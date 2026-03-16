from __future__ import annotations

import os
import time
from typing import List, Optional

import cv2
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from app.managers.capture_manager import CaptureManager
from app.widgets.annotation_canvas import AnnotationCanvas


def frame_to_pixmap(frame, width: int, height: int) -> QPixmap:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    image = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
    return QPixmap.fromImage(image.copy()).scaled(width, height, Qt.KeepAspectRatio, Qt.SmoothTransformation)


class DataCaptureWidget(QWidget):
    def __init__(self, capture_manager: CaptureManager, class_names: List[str], log_callback=None, parent=None):
        super().__init__(parent)
        self.capture_manager = capture_manager
        self.class_names = class_names or ["defect"]
        self.log_callback = log_callback
        self.current_frame: Optional[np.ndarray] = None
        self.pending_image_path: Optional[str] = None
        self.pending_label_path: Optional[str] = None
        self._last_preview_ts = 0.0
        self._build_ui()
        self._wire_signals()

    def _build_ui(self):
        layout = QGridLayout(self)

        self.live_preview = QLabel("Waiting for camera...")
        self.live_preview.setAlignment(Qt.AlignCenter)
        self.live_preview.setMinimumSize(560, 360)
        self.live_preview.setStyleSheet("background:#111; color:#EEE; border-radius:8px;")

        self.annotation_canvas = AnnotationCanvas(self.class_names)
        self.annotation_canvas.setStyleSheet("background:#111; border-radius:8px;")

        layout.addWidget(self.live_preview, 0, 0)
        layout.addWidget(self.annotation_canvas, 0, 1)

        side = QVBoxLayout()

        session_box = QGroupBox("Session")
        session_form = QFormLayout(session_box)
        self.base_dir_edit = QLineEdit(self.capture_manager.base_dir)
        self.session_name_edit = QLineEdit("defect_session")
        self.current_session_label = QLabel("-")
        self.capture_count_label = QLabel("0")
        self.new_session_btn = QPushButton("Start New Session")
        session_form.addRow("Base dir", self.base_dir_edit)
        session_form.addRow("Session name", self.session_name_edit)
        session_form.addRow("Current session", self.current_session_label)
        session_form.addRow("Saved images", self.capture_count_label)
        session_form.addRow(self.new_session_btn)

        capture_box = QGroupBox("Capture")
        capture_form = QFormLayout(capture_box)
        self.capture_btn = QPushButton("Capture Frame")
        self.current_image_label = QLabel("-")
        capture_form.addRow("Pending image", self.current_image_label)
        capture_form.addRow(self.capture_btn)

        annotation_box = QGroupBox("Annotation")
        annotation_form = QFormLayout(annotation_box)
        self.class_combo = QComboBox()
        self.class_combo.addItems(self.class_names)
        self.prev_class_btn = QPushButton("Prev Class [")
        self.next_class_btn = QPushButton("Next Class ]")
        class_row = QHBoxLayout()
        class_row.addWidget(self.prev_class_btn)
        class_row.addWidget(self.next_class_btn)
        self.box_count_label = QLabel("0")
        self.dirty_label = QLabel("No")
        self.save_btn = QPushButton("Save (S)")
        self.undo_btn = QPushButton("Undo (U)")
        self.clear_btn = QPushButton("Clear (C)")
        self.cancel_btn = QPushButton("Cancel")
        action_row1 = QHBoxLayout()
        action_row1.addWidget(self.save_btn)
        action_row1.addWidget(self.undo_btn)
        action_row2 = QHBoxLayout()
        action_row2.addWidget(self.clear_btn)
        action_row2.addWidget(self.cancel_btn)
        annotation_form.addRow("Active class", self.class_combo)
        annotation_form.addRow(class_row)
        annotation_form.addRow("Boxes", self.box_count_label)
        annotation_form.addRow("Unsaved", self.dirty_label)
        annotation_form.addRow(action_row1)
        annotation_form.addRow(action_row2)

        help_box = QGroupBox("Controls")
        help_layout = QVBoxLayout(help_box)
        self.help_text = QTextEdit()
        self.help_text.setReadOnly(True)
        self.help_text.setMaximumHeight(190)
        self.help_text.setPlainText(
            "Mouse: left drag to draw box\n"
            "S: save\n"
            "U: undo last\n"
            "C: clear all\n"
            "[: previous class\n"
            "]: next class\n"
            "0-9: select class\n"
            "Workflow: Capture Frame -> draw boxes -> Save"
        )
        help_layout.addWidget(self.help_text)

        side.addWidget(session_box)
        side.addWidget(capture_box)
        side.addWidget(annotation_box)
        side.addWidget(help_box)
        side.addStretch(1)

        layout.addLayout(side, 1, 0, 1, 2)

    def _wire_signals(self):
        self.new_session_btn.clicked.connect(self._start_new_session)
        self.capture_btn.clicked.connect(self.capture_current_frame)
        self.prev_class_btn.clicked.connect(self.annotation_canvas.prev_class)
        self.next_class_btn.clicked.connect(self.annotation_canvas.next_class)
        self.class_combo.currentIndexChanged.connect(self.annotation_canvas.set_active_class)
        self.save_btn.clicked.connect(self.annotation_canvas.save_annotations)
        self.undo_btn.clicked.connect(self.annotation_canvas.undo_last)
        self.clear_btn.clicked.connect(self.annotation_canvas.clear_all)
        self.cancel_btn.clicked.connect(self._cancel_annotation)

        self.annotation_canvas.box_count_changed.connect(lambda n: self.box_count_label.setText(str(n)))
        self.annotation_canvas.dirty_changed.connect(lambda d: self.dirty_label.setText("Yes" if d else "No"))
        self.annotation_canvas.active_class_changed.connect(self._on_active_class_changed)
        self.annotation_canvas.save_completed.connect(self._on_save_completed)
        self.annotation_canvas.status_message.connect(self._log)

        self.capture_manager.session_changed.connect(self.current_session_label.setText)
        self.capture_manager.count_changed.connect(lambda n: self.capture_count_label.setText(str(n)))
        self.capture_manager.status_message.connect(self._log)

    def update_live_frame(self, frame: np.ndarray):
        self.current_frame = frame.copy()
        now = time.perf_counter()
        if now - self._last_preview_ts < 0.08:
            return
        self._last_preview_ts = now
        self.live_preview.setPixmap(frame_to_pixmap(frame, self.live_preview.width(), self.live_preview.height()))

    def capture_current_frame(self):
        if self.current_frame is None:
            self._log("No camera frame available")
            return
        self.capture_manager.base_dir = os.path.expanduser(self.base_dir_edit.text().strip())
        os.makedirs(self.capture_manager.base_dir, exist_ok=True)
        if not self.capture_manager.current_session_dir:
            self._start_new_session()
        self.pending_image_path, self.pending_label_path = self.capture_manager.create_capture_paths()
        self.current_image_label.setText(os.path.basename(self.pending_image_path))
        self.annotation_canvas.load_capture(self.current_frame, self.pending_image_path, self.pending_label_path)
        self._log(f"Captured frame for annotation: {self.pending_image_path}")

    def _start_new_session(self):
        self.capture_manager.base_dir = os.path.expanduser(self.base_dir_edit.text().strip())
        os.makedirs(self.capture_manager.base_dir, exist_ok=True)
        self.capture_manager.start_new_session(self.session_name_edit.text().strip() or "defect_session")

    def _cancel_annotation(self):
        self.pending_image_path = None
        self.pending_label_path = None
        self.current_image_label.setText("-")
        self.annotation_canvas.image_bgr = None
        self.annotation_canvas.boxes = []
        self.annotation_canvas.update()
        self.box_count_label.setText("0")
        self.dirty_label.setText("No")
        self._log("Annotation cancelled")

    def _on_active_class_changed(self, class_id: int, _name: str):
        if self.class_combo.currentIndex() != class_id:
            self.class_combo.blockSignals(True)
            self.class_combo.setCurrentIndex(class_id)
            self.class_combo.blockSignals(False)

    def _on_save_completed(self, image_path: str, label_path: str):
        self.capture_manager.mark_saved()
        self.current_image_label.setText(os.path.basename(image_path))
        self._log(f"Saved capture: {image_path}")
        self._log(f"Saved labels: {label_path}")

    def _log(self, message: str):
        if self.log_callback:
            self.log_callback(message)
