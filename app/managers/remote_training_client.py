import os
import shlex
import subprocess
import time
from dataclasses import dataclass
from typing import Dict, List

from PyQt5.QtCore import QThread, pyqtSignal


@dataclass
class TrainingConfig:
    host: str
    user: str
    remote_project_dir: str
    session_path: str
    classes_csv: str
    model_name: str
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

    def stop(self):
        self._stopped = True

    def run(self):
        try:
            if self.config.use_mock:
                self._run_mock_pipeline()
            else:
                self._run_real_pipeline()
        except Exception as exc:
            self.status_changed.emit("failed")
            self.latest_message.emit(str(exc))
            self.log_line.emit(f"Remote training failed: {exc}")
            self.completed.emit(False, str(exc))

    def _run_mock_pipeline(self):
        phases = [
            ("training started", "Preparing remote run", 1.0),
            ("training running", f"Few-shot training for model {self.config.model_name}", 3.0),
            ("validation running", "Running validation metrics", 2.0),
            ("export running", "Exporting optimized edge model", 2.0),
            ("deployment running", "Deploying model back to Raspberry Pi", 2.0),
            ("completed", "Remote pipeline completed successfully", 0.2),
        ]
        for state, message, seconds in phases:
            if self._stopped:
                self.completed.emit(False, "Cancelled")
                return
            self.status_changed.emit(state)
            self.latest_message.emit(message)
            self.log_line.emit(message)
            t_end = time.time() + seconds
            while time.time() < t_end:
                if self._stopped:
                    self.completed.emit(False, "Cancelled")
                    return
                time.sleep(0.2)
        self.completed.emit(True, "Mock training pipeline completed")

    def _run_real_pipeline(self):
        remote_dir = shlex.quote(self.config.remote_project_dir)
        session_path = shlex.quote(self.config.session_path)
        classes_csv = shlex.quote(self.config.classes_csv)
        model_name = shlex.quote(self.config.model_name)
        py = shlex.quote(self.config.python_exec)

        phases: List[Dict[str, str]] = [
            {
                "status": "training started",
                "message": "Remote training command starting",
                "command": f"cd {remote_dir} && {py} fine_tune.py --session {session_path} --classes {classes_csv} --model {model_name}",
            },
            {
                "status": "validation running",
                "message": "Validation started",
                "command": f"cd {remote_dir} && {py} validate.py --model {model_name}",
            },
            {
                "status": "export running",
                "message": "Export started",
                "command": f"cd {remote_dir} && {py} export_model.py --model {model_name}",
            },
            {
                "status": "deployment running",
                "message": "Deployment started",
                "command": f"cd {remote_dir} && {py} deploy_back.py --model {model_name}",
            },
        ]

        for idx, phase in enumerate(phases):
            if self._stopped:
                self.completed.emit(False, "Cancelled")
                return
            state = phase["status"]
            self.status_changed.emit(state)
            self.latest_message.emit(phase["message"])
            self.log_line.emit(f"[{state}] {phase['command']}")
            self._run_ssh_command(phase["command"], training_phase=(idx == 0))

        self.status_changed.emit("completed")
        self.latest_message.emit("Training, validation, export, and deployment finished")
        self.log_line.emit("Remote pipeline completed successfully")
        self.completed.emit(True, "Remote pipeline completed")

    def _run_ssh_command(self, remote_command: str, training_phase: bool = False):
        ssh_parts = ["ssh"]
        if self.config.ssh_key:
            ssh_parts += ["-i", os.path.expanduser(self.config.ssh_key)]
        ssh_parts.append(f"{self.config.user}@{self.config.host}")
        ssh_parts.append(remote_command)

        process = subprocess.Popen(
            ssh_parts,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        if training_phase:
            self.status_changed.emit("training running")

        assert process.stdout is not None
        for line in process.stdout:
            line = line.rstrip()
            if line:
                self.log_line.emit(line)
                self.latest_message.emit(line)
            if self._stopped:
                process.terminate()
                self.completed.emit(False, "Cancelled")
                return

        ret = process.wait()
        if ret != 0:
            raise RuntimeError(f"SSH command failed with exit code {ret}: {remote_command}")


class RemoteTrainingClient:
    def __init__(self):
        self.worker: RemoteTrainingWorker | None = None

    def start(self, config: TrainingConfig) -> RemoteTrainingWorker:
        if self.worker and self.worker.isRunning():
            raise RuntimeError("Training is already running")
        self.worker = RemoteTrainingWorker(config)
        self.worker.start()
        return self.worker

    def stop(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
