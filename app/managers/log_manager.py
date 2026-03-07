from datetime import datetime

from PyQt5.QtCore import QObject, pyqtSignal


class LogManager(QObject):
    log_added = pyqtSignal(str)

    def log(self, message: str):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {message}"
        self.log_added.emit(line)
