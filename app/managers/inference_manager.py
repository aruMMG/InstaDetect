import time
from typing import Optional

import numpy as np
from PyQt5.QtCore import QObject, QRunnable, QThreadPool, QTimer, pyqtSignal

from app.interfaces.predictor import draw_detections


class InferenceSignals(QObject):
    result = pyqtSignal(object, list, float, float)
    error = pyqtSignal(str)


class InferenceTask(QRunnable):
    def __init__(self, predictor, frame: np.ndarray):
        super().__init__()
        self.predictor = predictor
        self.frame = frame.copy()
        self.signals = InferenceSignals()

    def run(self):
        try:
            t0 = time.perf_counter()
            detections = self.predictor.predict(self.frame)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            fps = 1000.0 / latency_ms if latency_ms > 0 else 0.0
            overlay = draw_detections(self.frame, detections)
            self.signals.result.emit(overlay, detections, latency_ms, fps)
        except Exception as exc:
            self.signals.error.emit(str(exc))


class InferenceManager(QObject):
    inference_result = pyqtSignal(object, list, float, float)
    inference_state_changed = pyqtSignal(bool)
    error = pyqtSignal(str)

    def __init__(self, predictor, interval_ms: int = 250, parent=None):
        super().__init__(parent)
        self.predictor = predictor
        self.interval_ms = interval_ms
        self._latest_frame: Optional[np.ndarray] = None
        self._running = False
        self._busy = False
        self.thread_pool = QThreadPool(self)
        self.thread_pool.setMaxThreadCount(1)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._submit_task)

    def set_latest_frame(self, frame: np.ndarray):
        self._latest_frame = frame

    def start(self):
        if self._running:
            return
        self._running = True
        self.timer.start(self.interval_ms)
        self.inference_state_changed.emit(True)

    def stop(self):
        self._running = False
        self.timer.stop()
        self.inference_state_changed.emit(False)

    def _submit_task(self):
        if not self._running or self._busy or self._latest_frame is None:
            return
        self._busy = True
        task = InferenceTask(self.predictor, self._latest_frame)
        task.signals.result.connect(self._on_result)
        task.signals.error.connect(self._on_error)
        self.thread_pool.start(task)

    def _on_result(self, overlay_frame, detections, latency_ms, fps):
        self._busy = False
        self.inference_result.emit(overlay_frame, detections, latency_ms, fps)

    def _on_error(self, message: str):
        self._busy = False
        self.error.emit(message)