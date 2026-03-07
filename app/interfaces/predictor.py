import os
import threading
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np
from ultralytics import YOLO


@dataclass
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    class_name: str


class PredictorInterface:
    def load_model(self, model_path: str):
        raise NotImplementedError

    def predict(self, frame: np.ndarray) -> List[Detection]:
        raise NotImplementedError


class UltralyticsOnnxPredictor(PredictorInterface):
    """
    Predictor wrapper for Ultralytics YOLO exported ONNX models.

    Expected usage:
      predictor = UltralyticsOnnxPredictor(conf_threshold=0.25, imgsz=640)
      predictor.load_model("/path/to/weights/best.onnx")
      detections = predictor.predict(frame)

    Notes:
    - model_path can be a file path or a directory.
    - if a directory is passed, the predictor tries current_model.onnx, best.onnx,
      then the newest .onnx file in that directory.
    """

    def __init__(self, conf_threshold: float = 0.25, imgsz: int = 640, device: str = "cpu"):
        self.conf_threshold = conf_threshold
        self.imgsz = imgsz
        self.device = device
        self.model: Optional[YOLO] = None
        self.model_path: str = ""
        self.class_names = {}
        self._lock = threading.Lock()

    def load_model(self, model_path: str):
        resolved = self._resolve_model_path(model_path)
        with self._lock:
            self.model = YOLO(resolved, task="detect")
            self.model_path = resolved
            names = getattr(self.model, "names", None)
            self.class_names = names if isinstance(names, dict) else {}

    def predict(self, frame: np.ndarray) -> List[Detection]:
        if frame is None:
            return []

        with self._lock:
            if self.model is None:
                raise RuntimeError("No model loaded. Load an ONNX model before starting inference.")

            results = self.model.predict(
                source=frame,
                verbose=False,
                conf=self.conf_threshold,
                imgsz=self.imgsz,
                device=self.device,
            )

        detections: List[Detection] = []
        if not results:
            return detections

        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or boxes.xyxy is None:
            return detections

        names = getattr(result, "names", None) or self.class_names or {}

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy() if boxes.conf is not None else np.zeros((len(xyxy),), dtype=float)
        classes = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else np.zeros((len(xyxy),), dtype=int)

        for idx, box in enumerate(xyxy):
            x1, y1, x2, y2 = [int(v) for v in box.tolist()]
            cls_id = int(classes[idx]) if idx < len(classes) else 0
            cls_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)
            conf = float(confs[idx]) if idx < len(confs) else 0.0
            detections.append(
                Detection(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    confidence=conf,
                    class_name=cls_name,
                )
            )
        return detections

    @staticmethod
    def _resolve_model_path(model_path: str) -> str:
        expanded = os.path.expanduser(model_path)
        if os.path.isfile(expanded):
            return expanded

        if os.path.isdir(expanded):
            preferred = [
                os.path.join(expanded, "current_model.onnx"),
                os.path.join(expanded, "best.onnx"),
            ]
            for candidate in preferred:
                if os.path.isfile(candidate):
                    return candidate

            onnx_files = [
                os.path.join(expanded, name)
                for name in os.listdir(expanded)
                if name.lower().endswith(".onnx")
            ]
            if onnx_files:
                onnx_files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
                return onnx_files[0]

            raise FileNotFoundError(f"No .onnx model found in directory: {expanded}")

        raise FileNotFoundError(f"Model path does not exist: {expanded}")


def draw_detections(frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
    image = frame.copy()
    for det in detections:
        cv2.rectangle(image, (det.x1, det.y1), (det.x2, det.y2), (0, 255, 0), 2)
        label = f"{det.class_name} {det.confidence:.2f}"
        text_w = max(150, min(260, 10 * len(label)))
        cv2.rectangle(image, (det.x1, max(0, det.y1 - 24)), (det.x1 + text_w, det.y1), (0, 255, 0), -1)
        cv2.putText(
            image,
            label,
            (det.x1 + 4, max(16, det.y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    return image
