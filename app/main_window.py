import os
import time
import json
from pathlib import Path
from typing import Optional

import cv2
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QTabWidget,
    QVBoxLayout,
    QWidget,
    QSizePolicy,
)

from app.interfaces.predictor import UltralyticsOnnxPredictor
from app.managers.camera_manager import CameraManager
from app.managers.capture_manager import CaptureManager
from app.managers.inference_manager import InferenceManager
from app.managers.log_manager import LogManager
from app.managers.model_manager import ModelManager
from app.managers.remote_training_client import RemoteTrainingClient, TrainingConfig
from app.widgets.status_chip import StatusChip
from app.widgets.data_capture_widget import DataCaptureWidget

def frame_to_pixmap(frame, width: int, height: int) -> QPixmap:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    image = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
    return QPixmap.fromImage(image).scaled(width, height, Qt.KeepAspectRatio, Qt.SmoothTransformation)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Edge AI Industrial Defect Detection Demo")
        self.resize(1420, 860)
        self.setMinimumSize(1420, 860)
        
        self.device_name = os.uname().nodename if hasattr(os, "uname") else "Raspberry Pi"
        self.current_frame = None
        self.last_inference_frame = None

        self._last_capture_preview_ts = 0.0
        self._last_inference_preview_ts = 0.0

        project_root = Path(__file__).resolve().parent.parent
        base_data_dir = os.path.expanduser("~/edge_ai_demo_data")
        self.ui_config_path = os.path.join(base_data_dir, "config", "ui_config.json")
        os.makedirs(os.path.dirname(self.ui_config_path), exist_ok=True)
        
        os.makedirs(base_data_dir, exist_ok=True)
        default_model_dir = str(project_root / "weights")
        os.makedirs(default_model_dir, exist_ok=True)
        self.default_model_path = os.path.join(default_model_dir, "current_model.onnx")

        self.log_manager = LogManager()
        self.predictor = UltralyticsOnnxPredictor(conf_threshold=0.25, imgsz=640, device="cpu")
        self.camera_manager = CameraManager(camera_index=0)
        self.inference_manager = InferenceManager(self.predictor)
        
        self.capture_manager = CaptureManager(base_dir=os.path.join(base_data_dir, "captures"))
        self.data_capture_widget = DataCaptureWidget(
            capture_manager=self.capture_manager,
            class_names=["scratch", "dent", "crack", "missing_part"],
            log_callback=self.log_manager.log,
        )
        self.remote_training_client = RemoteTrainingClient()
        self.model_manager = ModelManager(self.predictor, self.default_model_path, auto_reload=False)


        self._build_ui()
        self._wire_signals()
        self._apply_theme()
        self._load_ui_config()

        self.capture_manager.start_new_session("demo_session")
        self.model_manager.reload_model()
        self.camera_manager.start_camera()
        self.log_manager.log("Application initialized")

    def closeEvent(self, event):
        self.inference_manager.stop()
        self.remote_training_client.stop()
        self.camera_manager.stop_camera()
        super().closeEvent(event)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        header = QFrame()
        header_layout = QHBoxLayout(header)
        title = QLabel("Edge AI Industrial Defect Detection Demo")
        title.setStyleSheet("font-size: 22px; font-weight: 700;")
        self.header_device_label = QLabel(f"Device: {self.device_name}")
        self.header_device_label.setStyleSheet("font-size: 14px; color: #455A64;")
        header_layout.addWidget(title)
        header_layout.addStretch(1)
        header_layout.addWidget(self.header_device_label)
        root.addWidget(header)

        self.tabs = QTabWidget()
        root.addWidget(self.tabs)

        self.tab_inference = self._build_inference_tab()
        self.tab_capture = self.data_capture_widget
        # self.tab_capture = self._build_capture_tab()
        self.tab_training = self._build_training_tab()
        self.tab_logs = self._build_logs_tab()

        self.tabs.addTab(self.tab_inference, "Live Inference")
        self.tabs.addTab(self.tab_capture, "Data Capture")
        self.tabs.addTab(self.tab_training, "Training / Deployment")
        self.tabs.addTab(self.tab_logs, "Logs")

    def _build_inference_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        top = QHBoxLayout()
        self.inference_status_chip = StatusChip("stopped")
        self.model_name_value = QLabel("-")
        self.model_time_value = QLabel("-")
        self.fps_value = QLabel("0.0")
        self.latency_value = QLabel("0.0 ms")
        self.det_count_value = QLabel("0")
        self.inference_toggle_btn = QPushButton("Start Inference")
        self.inference_toggle_btn.setMinimumWidth(180)

        for name, widget in [
            ("Inference", self.inference_status_chip),
            ("Model", self.model_name_value),
            ("Loaded", self.model_time_value),
            ("FPS", self.fps_value),
            ("Latency", self.latency_value),
            ("Detections", self.det_count_value),
        ]:
            block = QVBoxLayout()
            label = QLabel(name)
            label.setStyleSheet("color: #546E7A; font-size: 12px;")
            block.addWidget(label)
            block.addWidget(widget)
            top.addLayout(block)

        top.addStretch(1)
        top.addWidget(self.inference_toggle_btn)
        layout.addLayout(top)

        self.inference_preview = QLabel("Waiting for camera...")
        self.inference_preview.setAlignment(Qt.AlignCenter)
        self.inference_preview.setMinimumSize(640, 480)
        self.inference_preview.setStyleSheet("background: #111; color: #EEE; border-radius: 8px;")
        self.inference_preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.inference_preview.setScaledContents(False)
        layout.addWidget(self.inference_preview)
        return tab

    def _build_capture_tab(self) -> QWidget:
        tab = QWidget()
        layout = QGridLayout(tab)

        self.capture_preview = QLabel("Waiting for camera...")
        self.capture_preview.setAlignment(Qt.AlignCenter)
        self.capture_preview.setMinimumSize(900, 620)
        self.capture_preview.setStyleSheet("background: #111; color: #EEE; border-radius: 8px;")
        layout.addWidget(self.capture_preview, 0, 0, 2, 1)

        side = QVBoxLayout()
        session_group = QGroupBox("Session Control")
        session_form = QFormLayout(session_group)
        self.capture_dir_edit = QLineEdit(self.capture_manager.base_dir)
        self.session_name_edit = QLineEdit("demo_session")
        self.current_session_value = QLabel("-")
        self.image_count_value = QLabel("0")
        self.new_session_btn = QPushButton("Start New Session")
        self.browse_capture_dir_btn = QPushButton("Browse...")
        dir_row = QHBoxLayout()
        dir_row.addWidget(self.capture_dir_edit)
        dir_row.addWidget(self.browse_capture_dir_btn)
        session_form.addRow("Base folder", dir_row)
        session_form.addRow("Session name", self.session_name_edit)
        session_form.addRow("Current session", self.current_session_value)
        session_form.addRow("Images in session", self.image_count_value)
        session_form.addRow(self.new_session_btn)
        side.addWidget(session_group)

        capture_group = QGroupBox("Capture")
        capture_form = QFormLayout(capture_group)
        self.class_label_combo = QComboBox()
        self.class_label_combo.setEditable(True)
        self.class_label_combo.addItems(["scratch", "dent", "crack", "missing_part"])
        self.capture_btn = QPushButton("Capture Image")
        self.capture_btn.setMinimumHeight(44)
        capture_form.addRow("Class label", self.class_label_combo)
        capture_form.addRow(self.capture_btn)
        side.addWidget(capture_group)
        side.addStretch(1)

        layout.addLayout(side, 0, 1)
        return tab

    def _build_training_tab(self) -> QWidget:
        tab = QWidget()
        layout = QHBoxLayout(tab)

        config_box = QGroupBox("Remote Training Configuration")
        config_form = QFormLayout(config_box)
        self.host_edit = QLineEdit("192.168.1.100")
        self.user_edit = QLineEdit("pi")
        self.remote_dir_edit = QLineEdit("~/edge_training_project")
        self.remote_dataset_root_edit = QLineEdit("~/edge_training_project/datasets")
        self.training_session_path_edit = QLineEdit(self.capture_manager.current_session_dir or "")
        self.training_classes_edit = QLineEdit(",".join(self.data_capture_widget.class_names))
        self.training_model_name_edit = QLineEdit("defect_demo_v1")
        self.training_python_edit = QLineEdit("python3")
        self.ssh_key_edit = QLineEdit("~/.ssh/id_rsa")
        self.mock_checkbox = QCheckBox("Use mock remote pipeline")
        self.mock_checkbox.setChecked(True)
        self.start_training_btn = QPushButton("Trigger Remote Training")
        self.stop_training_btn = QPushButton("Stop Training")
        self.stop_training_btn.setEnabled(False)

        self.remote_runs_root_edit = QLineEdit("~/side_work/edge_remote_project/runs")
        self.save_training_cfg_btn = QPushButton("Save Remote Config")

        config_form.addRow("SSH host", self.host_edit)
        config_form.addRow("SSH user", self.user_edit)
        config_form.addRow("Remote project dir", self.remote_dir_edit)
        config_form.addRow("Remote dataset root", self.remote_dataset_root_edit)
        config_form.addRow("Session path", self.training_session_path_edit)
        config_form.addRow("Classes CSV", self.training_classes_edit)
        config_form.addRow("Model name", self.training_model_name_edit)
        config_form.addRow("Python exec", self.training_python_edit)
        config_form.addRow("SSH key", self.ssh_key_edit)
        config_form.addRow(self.mock_checkbox)

        config_form.addRow("Remote runs root", self.remote_runs_root_edit)
        config_form.addRow(self.save_training_cfg_btn)

        button_row = QHBoxLayout()
        button_row.addWidget(self.start_training_btn)
        button_row.addWidget(self.stop_training_btn)
        config_form.addRow(button_row)

        status_box = QGroupBox("Deployment / Model Reload")
        status_form = QFormLayout(status_box)
        self.training_status_chip = StatusChip("idle")
        self.training_latest_msg = QLabel("No training started")
        self.training_latest_msg.setWordWrap(True)
        self.training_model_path_edit = QLineEdit(self.default_model_path)
        self.current_loaded_model_value = QLabel("-")
        self.last_deploy_time_value = QLabel("-")
        self.reload_model_btn = QPushButton("Reload Model")
        self.auto_reload_checkbox = QCheckBox("Auto reload on model file change")
        self.auto_reload_checkbox.setChecked(False)
        self.polling_checkbox = QCheckBox("Enable model polling")
        self.polling_checkbox.setChecked(True)

        status_form.addRow("Pipeline status", self.training_status_chip)
        status_form.addRow("Latest message", self.training_latest_msg)
        status_form.addRow("Model file", self.training_model_path_edit)
        status_form.addRow("Loaded model", self.current_loaded_model_value)
        status_form.addRow("Last deployment", self.last_deploy_time_value)
        status_form.addRow(self.auto_reload_checkbox)
        status_form.addRow(self.polling_checkbox)
        status_form.addRow(self.reload_model_btn)

        layout.addWidget(config_box, 2)
        layout.addWidget(status_box, 2)
        return tab

    def _build_logs_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.logs_text = QPlainTextEdit()
        self.logs_text.setReadOnly(True)
        clear_btn = QPushButton("Clear Logs")
        clear_btn.clicked.connect(self.logs_text.clear)
        layout.addWidget(self.logs_text)
        layout.addWidget(clear_btn, alignment=Qt.AlignRight)
        return tab

    def _wire_signals(self):
        self.log_manager.log_added.connect(self.logs_text.appendPlainText)

        self.camera_manager.frame_ready.connect(self._on_camera_frame)
        self.camera_manager.error.connect(self._on_error)
        self.camera_manager.camera_state_changed.connect(self._on_camera_state_changed)

        self.inference_toggle_btn.clicked.connect(self._toggle_inference)
        self.inference_manager.inference_result.connect(self._on_inference_result)
        self.inference_manager.error.connect(self._on_error)
        self.inference_manager.inference_state_changed.connect(self._on_inference_state_changed)

        # self.browse_capture_dir_btn.clicked.connect(self._choose_capture_dir)
        # self.new_session_btn.clicked.connect(self._start_new_capture_session)
        # self.capture_btn.clicked.connect(self._capture_current_frame)
        # self.capture_manager.session_changed.connect(self._on_session_changed)
        # self.capture_manager.count_changed.connect(lambda count: self.image_count_value.setText(str(count)))
        # self.capture_manager.image_captured.connect(self._on_image_captured)
        # self.capture_manager.error.connect(self._on_error)
        self.camera_manager.frame_ready.connect(self.data_capture_widget.update_live_frame)
        self.capture_manager.session_changed.connect(self._on_session_changed)
        
        self.start_training_btn.clicked.connect(self._start_remote_training)
        self.stop_training_btn.clicked.connect(self._stop_remote_training)
        self.reload_model_btn.clicked.connect(self._reload_model)
        self.save_training_cfg_btn.clicked.connect(self._save_ui_config)
        self.auto_reload_checkbox.stateChanged.connect(self._on_auto_reload_changed)
        self.polling_checkbox.stateChanged.connect(self._on_polling_changed)

        self.model_manager.model_loaded.connect(self._on_model_loaded)
        self.model_manager.model_changed_on_disk.connect(self._on_model_changed_on_disk)
        self.model_manager.error.connect(self._on_error)



    def _apply_theme(self):
        self.setStyleSheet(
            """
            QWidget {
                font-size: 13px;
            }
            QGroupBox {
                font-weight: 600;
                border: 1px solid #CFD8DC;
                border-radius: 8px;
                margin-top: 8px;
                padding-top: 12px;
            }
            QGroupBox::title {
                left: 10px;
                padding: 0 4px 0 4px;
            }
            QPushButton {
                background: #1565C0;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 10px 14px;
                font-weight: 600;
            }
            QPushButton:disabled {
                background: #90A4AE;
            }
            QLineEdit, QComboBox, QPlainTextEdit {
                border: 1px solid #B0BEC5;
                border-radius: 6px;
                padding: 6px;
                background: white;
            }
            QTabWidget::pane {
                border: 1px solid #CFD8DC;
                border-radius: 8px;
                background: #FAFAFA;
            }
            QTabBar::tab {
                padding: 10px 16px;
            }
            """
        )

    def _on_camera_frame(self, frame):
        self.current_frame = frame
        self.inference_manager.set_latest_frame(frame)

        now = time.perf_counter()

        if not self.inference_manager._running and now - self._last_inference_preview_ts >= 0.08:
            self.inference_preview.setPixmap(
                frame_to_pixmap(frame, self.inference_preview.width(), self.inference_preview.height())
            )
            self._last_inference_preview_ts = now

    def _on_inference_result(self, overlay_frame, detections, latency_ms, fps):
        self.last_inference_frame = overlay_frame.copy()
        self.inference_preview.setPixmap(
            frame_to_pixmap(overlay_frame, self.inference_preview.width(), self.inference_preview.height())
        )
        self.det_count_value.setText(str(len(detections)))
        self.fps_value.setText(f"{fps:.1f}")
        self.latency_value.setText(f"{latency_ms:.1f} ms")

    def _toggle_inference(self):
        if self.inference_manager._running:
            self.inference_manager.stop()
            self.log_manager.log("Inference stopped")
        else:
            self.inference_manager.start()
            self.log_manager.log("Inference started")

    def _on_inference_state_changed(self, running: bool):
        self.inference_status_chip.set_status("running" if running else "stopped")
        self.inference_toggle_btn.setText("Stop Inference" if running else "Start Inference")

    def _choose_capture_dir(self):
        directory = QFileDialog.getExistingDirectory(self, "Select capture base directory", self.capture_manager.base_dir)
        if directory:
            self.capture_manager.base_dir = directory
            self.capture_dir_edit.setText(directory)
            self.log_manager.log(f"Capture base directory set to {directory}")

    def _start_new_capture_session(self):
        self.capture_manager.base_dir = os.path.expanduser(self.capture_dir_edit.text().strip())
        os.makedirs(self.capture_manager.base_dir, exist_ok=True)
        session_name = self.session_name_edit.text().strip() or "capture_session"
        self.capture_manager.start_new_session(session_name)
        self.training_session_path_edit.setText(self.capture_manager.current_session_dir or "")
        self.log_manager.log(f"New capture session started: {self.capture_manager.current_session_dir}")

    def _on_session_changed(self, session_dir: str):
        self.training_session_path_edit.setText(session_dir)

    def _capture_current_frame(self):
        if self.current_frame is None:
            self._on_error("No camera frame available")
            return
        label = self.class_label_combo.currentText().strip() or "unlabeled"
        self.capture_manager.capture_image(self.current_frame, label)

    def _on_image_captured(self, path: str):
        self.log_manager.log(f"Image captured: {path}")

    def _start_remote_training(self):
        try:
            self.model_manager.set_model_path(self.training_model_path_edit.text().strip())
            config = TrainingConfig(
                host=self.host_edit.text().strip(),
                user=self.user_edit.text().strip(),
                remote_project_dir=self.remote_dir_edit.text().strip(),
                remote_dataset_root=self.remote_dataset_root_edit.text().strip(),
                remote_runs_root=self.remote_runs_root_edit.text().strip(),
                session_path=self.training_session_path_edit.text().strip(),
                classes_csv=self.training_classes_edit.text().strip(),
                model_name=self.training_model_name_edit.text().strip(),
                local_model_dest=self.training_model_path_edit.text().strip(),
                python_exec=self.training_python_edit.text().strip() or "python3",
                use_mock=self.mock_checkbox.isChecked(),
                ssh_key=self.ssh_key_edit.text().strip(),
            )
            worker = self.remote_training_client.start(config)
            worker.status_changed.connect(self._on_training_status_changed)
            worker.latest_message.connect(self.training_latest_msg.setText)
            worker.latest_message.connect(lambda msg: self.log_manager.log(f"Training status: {msg}"))
            worker.log_line.connect(self.log_manager.log)
            worker.completed.connect(self._on_training_completed)
            self.start_training_btn.setEnabled(False)
            self.stop_training_btn.setEnabled(True)
            self._save_ui_config()
            self.log_manager.log("Remote training triggered")
        except Exception as exc:
            self._on_error(str(exc))

    def _stop_remote_training(self):
        self.remote_training_client.stop()
        self.log_manager.log("Stop signal sent to remote training worker")

    def _on_training_status_changed(self, status: str):
        self.training_status_chip.set_status(status)

    def _on_training_completed(self, success: bool, message: str):
        self.start_training_btn.setEnabled(True)
        self.stop_training_btn.setEnabled(False)
        if success:
            self.training_status_chip.set_status("completed")
            self.training_latest_msg.setText("Model pulled back successfully. Reloading local model.")
            self.log_manager.log("Training/export/pull-back complete")
            self._reload_model()
        else:
            self.training_status_chip.set_status("failed")
            self.log_manager.log(f"Training failed: {message}")

    def _reload_model(self):
        path = self.training_model_path_edit.text().strip()
        self.model_manager.set_model_path(path)
        self.model_manager.reload_model()

    def _on_auto_reload_changed(self):
        self.model_manager.auto_reload = self.auto_reload_checkbox.isChecked()
        self.log_manager.log(f"Auto reload set to {self.model_manager.auto_reload}")

    def _on_polling_changed(self):
        if self.polling_checkbox.isChecked():
            self.model_manager.timer.start()
            self.log_manager.log("Model polling enabled")
        else:
            self.model_manager.timer.stop()
            self.log_manager.log("Model polling disabled")

    def _on_model_loaded(self, model_name: str, timestamp: str):
        self.model_name_value.setText(model_name)
        self.model_time_value.setText(timestamp)
        self.current_loaded_model_value.setText(model_name)
        self.last_deploy_time_value.setText(timestamp)
        self.log_manager.log(f"Model loaded: {model_name}")

    def _on_model_changed_on_disk(self, path: str):
        self.log_manager.log(f"Model file changed on disk: {path}")
        if not self.auto_reload_checkbox.isChecked():
            self.training_latest_msg.setText("Updated model detected on disk. Click Reload Model or enable auto reload.")

    def _on_camera_state_changed(self, running: bool):
        if running:
            self.log_manager.log("Camera started")
        else:
            self.log_manager.log("Camera stopped")

    def _on_error(self, message: str):
        self.log_manager.log(f"ERROR: {message}")
        QMessageBox.warning(self, "Edge AI Demo", message)

    def _save_ui_config(self):
        data = {
            "host": self.host_edit.text().strip(),
            "user": self.user_edit.text().strip(),
            "remote_project_dir": self.remote_dir_edit.text().strip(),
            "remote_dataset_root": self.remote_dataset_root_edit.text().strip(),
            "remote_runs_root": self.remote_runs_root_edit.text().strip(),
            "python_exec": self.training_python_edit.text().strip(),
            "ssh_key": self.ssh_key_edit.text().strip(),
            "model_file": self.training_model_path_edit.text().strip(),
            "mock_mode": self.mock_checkbox.isChecked(),
            "auto_reload": self.auto_reload_checkbox.isChecked(),
            "polling": self.polling_checkbox.isChecked(),
        }
        with open(self.ui_config_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        self.log_manager.log(f"Saved UI config: {self.ui_config_path}")


    def _load_ui_config(self):
        if not os.path.exists(self.ui_config_path):
            return
        try:
            with open(self.ui_config_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.host_edit.setText(data.get("host", self.host_edit.text()))
            self.user_edit.setText(data.get("user", self.user_edit.text()))
            self.remote_dir_edit.setText(data.get("remote_project_dir", self.remote_dir_edit.text()))
            self.remote_dataset_root_edit.setText(data.get("remote_dataset_root", self.remote_dataset_root_edit.text()))
            self.remote_runs_root_edit.setText(data.get("remote_runs_root", self.remote_runs_root_edit.text()))
            self.training_python_edit.setText(data.get("python_exec", self.training_python_edit.text()))
            self.ssh_key_edit.setText(data.get("ssh_key", self.ssh_key_edit.text()))
            self.training_model_path_edit.setText(data.get("model_file", self.training_model_path_edit.text()))
            self.mock_checkbox.setChecked(data.get("mock_mode", self.mock_checkbox.isChecked()))
            self.auto_reload_checkbox.setChecked(data.get("auto_reload", self.auto_reload_checkbox.isChecked()))
            self.polling_checkbox.setChecked(data.get("polling", self.polling_checkbox.isChecked()))

            self.log_manager.log(f"Loaded UI config: {self.ui_config_path}")
        except Exception as exc:
            self.log_manager.log(f"Failed to load UI config: {exc}")
