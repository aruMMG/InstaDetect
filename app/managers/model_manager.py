import os
from datetime import datetime
from typing import Optional

from PyQt5.QtCore import QObject, QTimer, pyqtSignal


class ModelManager(QObject):
    model_loaded = pyqtSignal(str, str)
    model_changed_on_disk = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, predictor, model_path: str, poll_ms: int = 5000, auto_reload: bool = False, parent=None):
        super().__init__(parent)
        self.predictor = predictor
        self.model_path = os.path.expanduser(model_path)
        self.auto_reload = auto_reload
        self.last_loaded_ts: Optional[str] = None
        self.last_seen_mtime: Optional[float] = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.check_for_updates)
        self.timer.start(poll_ms)

    def reload_model(self):
        try:
            self.predictor.load_model(self.model_path)
            file_mtime = os.path.getmtime(self.model_path) if os.path.exists(self.model_path) else None
            if file_mtime is not None:
                self.last_seen_mtime = file_mtime
                ts = datetime.fromtimestamp(file_mtime).strftime("%Y-%m-%d %H:%M:%S")
            else:
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.last_loaded_ts = ts
            self.model_loaded.emit(os.path.basename(self.model_path), ts)
        except Exception as exc:
            self.error.emit(f"Model reload failed: {exc}")

    def check_for_updates(self):
        try:
            if not os.path.exists(self.model_path):
                return
            current_mtime = os.path.getmtime(self.model_path)
            if self.last_seen_mtime is None:
                self.last_seen_mtime = current_mtime
                return
            if current_mtime > self.last_seen_mtime:
                self.last_seen_mtime = current_mtime
                self.model_changed_on_disk.emit(self.model_path)
                if self.auto_reload:
                    self.reload_model()
        except Exception as exc:
            self.error.emit(f"Model polling error: {exc}")

    def set_model_path(self, model_path: str):
        self.model_path = os.path.expanduser(model_path)
