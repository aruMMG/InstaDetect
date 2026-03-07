import time
from typing import Optional

import cv2
from PyQt5.QtCore import QThread, pyqtSignal


class CameraManager(QThread):
    frame_ready = pyqtSignal(object)
    camera_state_changed = pyqtSignal(bool)
    error = pyqtSignal(str)

    def __init__(self, camera_index: int = 0, width: int = 1280, height: int = 720, target_fps: int = 15, parent=None):
        super().__init__(parent)
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.target_fps = max(1, target_fps)
        self._frame_interval = 1.0 / self.target_fps
        self._running = False
        self._capture: Optional[cv2.VideoCapture] = None

    def start_camera(self):
        if self.isRunning():
            return
        self._running = True
        self.start()

    def stop_camera(self):
        self._running = False
        self.wait(1500)

    def run(self):
        self._capture = cv2.VideoCapture(self.camera_index)
        if not self._capture.isOpened():
            self.error.emit(f"Could not open camera index {self.camera_index}")
            self.camera_state_changed.emit(False)
            return

        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.camera_state_changed.emit(True)

        last_emit = 0.0
        while self._running:
            ok, frame = self._capture.read()
            if not ok:
                self.error.emit("Camera frame read failed")
                time.sleep(0.05)
                continue

            now = time.perf_counter()
            if now - last_emit < self._frame_interval:
                time.sleep(0.001)
                continue

            last_emit = now
            self.frame_ready.emit(frame.copy())

        self._capture.release()
        self._capture = None
        self.camera_state_changed.emit(False)