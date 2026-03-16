import os
from datetime import datetime
from typing import Optional, Tuple

from PyQt5.QtCore import QObject, pyqtSignal


class CaptureManager(QObject):
    session_changed = pyqtSignal(str)
    count_changed = pyqtSignal(int)
    status_message = pyqtSignal(str)

    def __init__(self, base_dir: str, parent=None):
        super().__init__(parent)
        self.base_dir = os.path.expanduser(base_dir)
        self.current_session_dir: Optional[str] = None
        self.images_dir: Optional[str] = None
        self.labels_dir: Optional[str] = None
        self.image_count = 0
        os.makedirs(self.base_dir, exist_ok=True)

    def start_new_session(self, session_name: Optional[str] = None):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = session_name.strip().replace(" ", "_") if session_name else "capture_session"
        self.current_session_dir = os.path.join(self.base_dir, f"{safe_name}_{stamp}")
        self.images_dir = os.path.join(self.current_session_dir, "images")
        self.labels_dir = os.path.join(self.current_session_dir, "labels")
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.labels_dir, exist_ok=True)
        self.image_count = 0
        self.session_changed.emit(self.current_session_dir)
        self.count_changed.emit(0)
        self.status_message.emit(f"New capture session: {self.current_session_dir}")

    def create_capture_paths(self, prefix: str = "capture") -> Tuple[str, str]:
        if not self.current_session_dir:
            self.start_new_session()
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        name = f"{prefix}_{stamp}"
        image_path = os.path.join(self.images_dir, f"{name}.jpg")
        label_path = os.path.join(self.labels_dir, f"{name}.txt")
        return image_path, label_path

    def mark_saved(self):
        self.image_count += 1
        self.count_changed.emit(self.image_count)
