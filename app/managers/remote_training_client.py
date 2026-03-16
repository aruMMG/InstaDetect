import os
import posixpath
import shlex
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import paramiko
from PyQt5.QtCore import QThread, pyqtSignal

from utils.dataset_builder import prepare_yolo_dataset


class TrainingCancelled(Exception):
    pass


@dataclass
class TrainingConfig:
    host: str
    user: str
    remote_project_dir: str   # scripts dir on remote
    remote_dataset_root: str
    remote_runs_root: str     # runs dir on remote
    session_path: str
    classes_csv: str
    model_name: str
    local_model_dest: str
    python_exec: str = "python3"
    use_mock: bool = True
    ssh_key: str = ""


class RemoteTrainingWorker(QThread):
    status_changed = pyqtSignal(str)
    log_line = pyqtSignal(str)
    latest_message = pyqtSignal(str)
    completed = pyqtSignal(bool, str)

    def __init__(self, config: TrainingConfig, parent=None):
        super().__init__(parent)
        self.config = config
        self._stopped = False
        self._ssh_client: Optional[paramiko.SSHClient] = None
        self._sftp_client: Optional[paramiko.SFTPClient] = None
        self._active_channel = None

    def stop(self):
        self._stopped = True
        if self._active_channel is not None:
            self._active_channel.close()
        if self._sftp_client is not None:
            self._sftp_client.close()
            self._sftp_client = None
        if self._ssh_client is not None:
            self._ssh_client.close()
            self._ssh_client = None

    def run(self):
        try:
            if self.config.use_mock:
                self._run_mock_pipeline()
            else:
                self._run_real_pipeline()
        except TrainingCancelled:
            self.latest_message.emit("Cancelled")
            self.log_line.emit("Remote training cancelled")
            self.completed.emit(False, "Cancelled")
        except Exception as exc:
            self.status_changed.emit("failed")
            self.latest_message.emit(str(exc))
            self.log_line.emit(f"Remote training failed: {exc}")
            self.completed.emit(False, str(exc))
        finally:
            self.stop()

    def _run_mock_pipeline(self):
        phases = [
            ("preparing dataset", "Preparing dataset from recent capture session", 0.5),
            ("uploading dataset", "Uploading prepared dataset to remote machine", 0.5),
            ("training started", "Preparing remote run", 1.0),
            ("training running", f"Few-shot training for model {self.config.model_name}", 3.0),
            ("validation running", "Running validation metrics", 2.0),
            ("export running", "Exporting optimized ONNX model", 2.0),
            ("pulling model", "Pulling model back to local machine", 1.5),
            ("completed", "Remote pipeline completed successfully", 0.2),
        ]
        for state, message, seconds in phases:
            if self._stopped:
                raise TrainingCancelled()
            self.status_changed.emit(state)
            self.latest_message.emit(message)
            self.log_line.emit(message)
            t_end = time.time() + seconds
            while time.time() < t_end:
                if self._stopped:
                    raise TrainingCancelled()
                time.sleep(0.2)
        self.completed.emit(True, "Mock training pipeline completed")

    def _run_real_pipeline(self):
        classes = [name.strip() for name in self.config.classes_csv.split(",") if name.strip()]
        if not self.config.session_path.strip():
            raise ValueError("Session path is required for remote training")
        if not classes:
            raise ValueError("Classes CSV must define at least one class")

        self.status_changed.emit("preparing dataset")
        self.latest_message.emit("Preparing local YOLO dataset from session")
        self.log_line.emit(f"Preparing dataset from session: {self.config.session_path}")
        prepared = prepare_yolo_dataset(self.config.session_path, classes)
        self.log_line.emit(f"Prepared dataset: {prepared.output_dir}")
        self.log_line.emit(f"dataset.yaml: {prepared.dataset_yaml}")
        self.log_line.emit(f"Split counts train={prepared.train_count} val={prepared.val_count}")

        ssh_client = self._connect_ssh()
        sftp_client = ssh_client.open_sftp()
        self._sftp_client = sftp_client

        remote_dataset_root = self._expand_remote_path(sftp_client, self.config.remote_dataset_root)
        remote_runs_root = self._expand_remote_path(sftp_client, self.config.remote_runs_root)
        remote_project_dir = self._expand_remote_path(sftp_client, self.config.remote_project_dir)
        dataset_name = f"{Path(prepared.session_dir).name}_{int(time.time())}"
        remote_dataset_dir = posixpath.join(remote_dataset_root, dataset_name)
        remote_dataset_yaml = posixpath.join(remote_dataset_dir, "dataset.yaml")

        self.status_changed.emit("uploading dataset")
        self.latest_message.emit("Uploading prepared dataset to remote machine")
        self.log_line.emit(f"Uploading dataset to {remote_dataset_dir}")
        self._sftp_mkdir_p(sftp_client, remote_dataset_dir)
        self._upload_dir(sftp_client, str(prepared.output_dir), remote_dataset_dir)

        remote_dir = shlex.quote(remote_project_dir)
        dataset_yaml = shlex.quote(remote_dataset_yaml)
        session_path = shlex.quote(self.config.session_path)
        classes_csv = shlex.quote(self.config.classes_csv)
        model_name = shlex.quote(self.config.model_name)
        py = shlex.quote(self.config.python_exec)

        phases: List[Dict[str, str]] = [
            {
                "status": "training started",
                "message": "Remote training command starting",
                "command": (
                    f"cd {remote_dir} && {py} fine_tune.py --session {session_path} "
                    f"--classes {classes_csv} --dataset-yaml {dataset_yaml} --model {model_name}"
                ),
            },
            {
                "status": "validation running",
                "message": "Validation started",
                "command": f"cd {remote_dir} && {py} validate.py --model {model_name} --dataset-yaml {dataset_yaml}",
            },
            {
                "status": "export running",
                "message": "Export started",
                "command": f"cd {remote_dir} && {py} export_model.py --model {model_name}",
            },
        ]

        for idx, phase in enumerate(phases):
            if self._stopped:
                raise TrainingCancelled()

            state = phase["status"]
            self.status_changed.emit(state)
            self.latest_message.emit(phase["message"])
            self.log_line.emit(f"[{state}] {phase['command']}")
            self._run_ssh_command(ssh_client, phase["command"], training_phase=(idx == 0))

        if self._stopped:
            raise TrainingCancelled()

        self.status_changed.emit("pulling model")
        self.latest_message.emit("Pulling exported ONNX back to local")
        self.log_line.emit("Pulling exported ONNX back to local")
        self._pull_exported_model(sftp_client, remote_runs_root)

        self.status_changed.emit("completed")
        self.latest_message.emit("Training, export, and model pull completed")
        self.log_line.emit("Remote pipeline completed successfully")
        self.completed.emit(True, "Remote pipeline completed")

    def _connect_ssh(self) -> paramiko.SSHClient:
        self.log_line.emit(f"Connecting to {self.config.user}@{self.config.host} with Paramiko")
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        connect_kwargs = {
            "hostname": self.config.host,
            "username": self.config.user,
            "timeout": 15,
            "allow_agent": True,
            "look_for_keys": not bool(self.config.ssh_key.strip()),
        }
        if self.config.ssh_key.strip():
            connect_kwargs["key_filename"] = os.path.expanduser(self.config.ssh_key)
        client.connect(**connect_kwargs)
        self._ssh_client = client
        return client

    def _run_ssh_command(self, ssh_client: paramiko.SSHClient, remote_command: str, training_phase: bool = False):
        _, stdout, _ = ssh_client.exec_command(remote_command, get_pty=True)
        channel = stdout.channel
        self._active_channel = channel

        if training_phase:
            self.status_changed.emit("training running")

        buffer = ""
        while True:
            if self._stopped:
                channel.close()
                raise TrainingCancelled()

            while channel.recv_ready():
                chunk = channel.recv(4096).decode("utf-8", errors="replace")
                buffer += chunk
                buffer = self._emit_complete_lines(buffer)

            if channel.exit_status_ready():
                while channel.recv_ready():
                    chunk = channel.recv(4096).decode("utf-8", errors="replace")
                    buffer += chunk
                    buffer = self._emit_complete_lines(buffer)
                if buffer.strip():
                    self.log_line.emit(buffer.rstrip())
                    self.latest_message.emit(buffer.rstrip())
                break

            time.sleep(0.1)

        self._active_channel = None
        ret = channel.recv_exit_status()
        if ret != 0:
            raise RuntimeError(f"SSH command failed with exit code {ret}: {remote_command}")

    def _pull_exported_model(self, sftp_client: paramiko.SFTPClient, remote_runs_root: str):
        remote_onnx = posixpath.join(
            remote_runs_root,
            self.config.model_name,
            "weights",
            "best.onnx",
        )
        local_dest = os.path.expanduser(self.config.local_model_dest)
        os.makedirs(os.path.dirname(local_dest), exist_ok=True)
        self.log_line.emit(f"Downloading {remote_onnx} -> {local_dest}")
        sftp_client.get(remote_onnx, local_dest)

    def _emit_complete_lines(self, buffer: str) -> str:
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            line = line.rstrip()
            if line:
                self.log_line.emit(line)
                self.latest_message.emit(line)
        return buffer

    @staticmethod
    def _expand_remote_path(sftp_client: paramiko.SFTPClient, path: str) -> str:
        if path == "~":
            return sftp_client.normalize(".")
        if path.startswith("~/"):
            return posixpath.join(sftp_client.normalize("."), path[2:])
        return path

    def _upload_dir(self, sftp_client: paramiko.SFTPClient, local_dir: str, remote_dir: str):
        for root, dirs, files in os.walk(local_dir):
            rel_root = os.path.relpath(root, local_dir)
            remote_root = remote_dir if rel_root == "." else posixpath.join(remote_dir, rel_root.replace(os.sep, "/"))
            self._sftp_mkdir_p(sftp_client, remote_root)
            for directory in dirs:
                self._sftp_mkdir_p(sftp_client, posixpath.join(remote_root, directory))
            for filename in files:
                if self._stopped:
                    raise TrainingCancelled()
                local_path = os.path.join(root, filename)
                remote_path = posixpath.join(remote_root, filename)
                self.log_line.emit(f"Uploading {local_path} -> {remote_path}")
                sftp_client.put(local_path, remote_path)

    @staticmethod
    def _sftp_mkdir_p(sftp_client: paramiko.SFTPClient, remote_dir: str):
        parts = []
        head = remote_dir
        while head not in ("", "/"):
            parts.append(head)
            head = posixpath.dirname(head)
        for directory in reversed(parts):
            try:
                sftp_client.stat(directory)
            except FileNotFoundError:
                sftp_client.mkdir(directory)


class RemoteTrainingClient:
    def __init__(self):
        self.worker = None

    def start(self, config: TrainingConfig) -> RemoteTrainingWorker:
        if self.worker and self.worker.isRunning():
            raise RuntimeError("Training is already running")
        self.worker = RemoteTrainingWorker(config)
        self.worker.start()
        return self.worker

    def stop(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
