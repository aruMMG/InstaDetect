import os
from datetime import datetime
from typing import Optional

import cv2
import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal


class CaptureManager(QObject):
    session_changed = pyqtSignal(str)
    count_changed = pyqtSignal(int)
    image_captured = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, base_dir: str, parent=None):
        super().__init__(parent)
        self.base_dir = os.path.expanduser(base_dir)
        self.current_session_dir: Optional[str] = None
        self.image_count = 0
        os.makedirs(self.base_dir, exist_ok=True)

    def start_new_session(self, session_name: Optional[str] = None):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = session_name.strip().replace(" ", "_") if session_name else "capture_session"
        self.current_session_dir = os.path.join(self.base_dir, f"{safe_name}_{stamp}")
        os.makedirs(self.current_session_dir, exist_ok=True)
        self.image_count = 0
        self.session_changed.emit(self.current_session_dir)
        self.count_changed.emit(self.image_count)

    def capture_image(self, frame: np.ndarray, label: str):
        if frame is None:
            self.error.emit("No frame available to capture")
            return
        if not self.current_session_dir:
            self.start_new_session()

        clean_label = (label or "unlabeled").strip().replace(" ", "_")
        label_dir = os.path.join(self.current_session_dir, clean_label)
        os.makedirs(label_dir, exist_ok=True)

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        file_path = os.path.join(label_dir, f"{clean_label}_{stamp}.jpg")
        ok = cv2.imwrite(file_path, frame)
        if not ok:
            self.error.emit(f"Failed to save image: {file_path}")
            return

        self.image_count += 1
        self.count_changed.emit(self.image_count)
        self.image_captured.emit(file_path)
