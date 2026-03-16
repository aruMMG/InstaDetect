from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PyQt5.QtCore import QPoint, QRect, Qt, pyqtSignal
from PyQt5.QtGui import QImage, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import QWidget


@dataclass
class Box:
    class_id: int
    x1: int
    y1: int
    x2: int
    y2: int

    def clamped(self, img_w: int, img_h: int) -> Tuple[int, int, int, int]:
        x1 = max(0, min(self.x1, img_w - 1))
        y1 = max(0, min(self.y1, img_h - 1))
        x2 = max(0, min(self.x2, img_w - 1))
        y2 = max(0, min(self.y2, img_h - 1))
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        return x1, y1, x2, y2

    def normalized(self, img_w: int, img_h: int) -> Tuple[float, float, float, float]:
        x1, y1, x2, y2 = self.clamped(img_w, img_h)
        bw = max(0, x2 - x1)
        bh = max(0, y2 - y1)
        xc = x1 + bw / 2.0
        yc = y1 + bh / 2.0
        return xc / img_w, yc / img_h, bw / img_w, bh / img_h

    def is_valid(self, min_size: int = 3) -> bool:
        return abs(self.x2 - self.x1) >= min_size and abs(self.y2 - self.y1) >= min_size


class YoloIO:
    @staticmethod
    def save_labels(label_path: str, boxes: List[Box], img_w: int, img_h: int) -> None:
        os.makedirs(os.path.dirname(label_path), exist_ok=True)
        with open(label_path, "w", encoding="utf-8") as f:
            for box in boxes:
                if not box.is_valid():
                    continue
                xc, yc, bw, bh = box.normalized(img_w, img_h)
                f.write(f"{int(box.class_id)} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

    @staticmethod
    def load_labels(label_path: str, img_w: int, img_h: int) -> List[Box]:
        if not os.path.exists(label_path):
            return []
        boxes: List[Box] = []
        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls_id = int(float(parts[0]))
                xc, yc, bw, bh = map(float, parts[1:])
                x1 = int(round((xc - bw / 2.0) * img_w))
                y1 = int(round((yc - bh / 2.0) * img_h))
                x2 = int(round((xc + bw / 2.0) * img_w))
                y2 = int(round((yc + bh / 2.0) * img_h))
                boxes.append(Box(cls_id, x1, y1, x2, y2))
        return boxes


class Colors:
    PALETTE = [
        (0, 255, 0),
        (0, 255, 255),
        (255, 0, 0),
        (255, 255, 0),
        (255, 0, 255),
        (0, 128, 255),
        (128, 255, 0),
        (255, 128, 0),
        (128, 0, 255),
        (0, 200, 200),
    ]

    @staticmethod
    def get(class_id: int) -> Tuple[int, int, int]:
        return Colors.PALETTE[class_id % len(Colors.PALETTE)]


class AnnotationCanvas(QWidget):
    box_count_changed = pyqtSignal(int)
    dirty_changed = pyqtSignal(bool)
    save_completed = pyqtSignal(str, str)
    status_message = pyqtSignal(str)
    active_class_changed = pyqtSignal(int, str)

    def __init__(self, classes: List[str], parent=None):
        super().__init__(parent)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMinimumSize(760, 520)
        self.classes = classes or ["defect"]
        self.active_class = 0

        self.image_bgr: Optional[np.ndarray] = None
        self.image_path: Optional[str] = None
        self.label_path: Optional[str] = None
        self.boxes: List[Box] = []
        self.dirty = False

        self.drawing = False
        self.temp_start: Optional[Tuple[int, int]] = None
        self.temp_end: Optional[Tuple[int, int]] = None
        self._display_rect = QRect()

    def set_classes(self, classes: List[str]):
        self.classes = classes or ["defect"]
        self.active_class = min(self.active_class, len(self.classes) - 1)
        self.active_class_changed.emit(self.active_class, self.classes[self.active_class])
        self.update()

    def load_capture(self, image_bgr: np.ndarray, image_path: str, label_path: str):
        self.image_bgr = image_bgr.copy()
        self.image_path = image_path
        self.label_path = label_path
        self.boxes = YoloIO.load_labels(label_path, self.image_bgr.shape[1], self.image_bgr.shape[0])
        self.dirty = False
        self.drawing = False
        self.temp_start = None
        self.temp_end = None
        self.box_count_changed.emit(len(self.boxes))
        self.dirty_changed.emit(False)
        self.status_message.emit(f"Loaded capture for annotation: {os.path.basename(image_path)}")
        self.update()
        self.setFocus()

    def has_image(self) -> bool:
        return self.image_bgr is not None

    def set_active_class(self, class_id: int):
        if 0 <= class_id < len(self.classes):
            self.active_class = class_id
            self.active_class_changed.emit(self.active_class, self.classes[self.active_class])
            self.update()

    def next_class(self):
        self.set_active_class((self.active_class + 1) % len(self.classes))

    def prev_class(self):
        self.set_active_class((self.active_class - 1) % len(self.classes))

    def undo_last(self):
        if self.boxes:
            self.boxes.pop()
            self._set_dirty(True)
            self.box_count_changed.emit(len(self.boxes))
            self.status_message.emit("Removed last box")
            self.update()

    def clear_all(self):
        if self.boxes:
            self.boxes.clear()
            self._set_dirty(True)
            self.box_count_changed.emit(0)
            self.status_message.emit("Cleared all boxes")
            self.update()

    def save_annotations(self):
        if self.image_bgr is None or not self.image_path or not self.label_path:
            self.status_message.emit("No captured image to save")
            return
        os.makedirs(os.path.dirname(self.image_path), exist_ok=True)
        ok = cv2.imwrite(self.image_path, self.image_bgr)
        if not ok:
            self.status_message.emit(f"Failed to save image: {self.image_path}")
            return
        h, w = self.image_bgr.shape[:2]
        YoloIO.save_labels(self.label_path, self.boxes, w, h)
        self._set_dirty(False)
        self.save_completed.emit(self.image_path, self.label_path)
        self.status_message.emit(f"Saved image and labels: {os.path.basename(self.image_path)}")

    def paintEvent(self, event):  # noqa: N802
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.black)

        if self.image_bgr is None:
            painter.setPen(Qt.white)
            painter.drawText(self.rect(), Qt.AlignCenter, "Capture an image to start annotation")
            return

        pixmap = self._to_pixmap(self.image_bgr)
        target = self._fit_rect(pixmap.width(), pixmap.height())
        self._display_rect = target
        painter.drawPixmap(target, pixmap)

        for idx, box in enumerate(self.boxes):
            x1, y1, x2, y2 = box.clamped(self.image_bgr.shape[1], self.image_bgr.shape[0])
            rx1, ry1 = self._img_to_widget(x1, y1)
            rx2, ry2 = self._img_to_widget(x2, y2)
            bgr = Colors.get(box.class_id)
            color = self._bgr_to_qt(bgr)
            painter.setPen(QPen(color, 2))
            painter.drawRect(QRect(QPoint(rx1, ry1), QPoint(rx2, ry2)))
            label = f"{idx}: {self.classes[box.class_id] if box.class_id < len(self.classes) else box.class_id}"
            painter.fillRect(rx1, max(0, ry1 - 22), 160, 20, color)
            painter.setPen(Qt.black)
            painter.drawText(rx1 + 4, max(14, ry1 - 6), label)

        if self.drawing and self.temp_start and self.temp_end:
            rx1, ry1 = self._img_to_widget(*self.temp_start)
            rx2, ry2 = self._img_to_widget(*self.temp_end)
            color = self._bgr_to_qt(Colors.get(self.active_class))
            painter.setPen(QPen(color, 2, Qt.DashLine))
            painter.drawRect(QRect(QPoint(rx1, ry1), QPoint(rx2, ry2)))

    def mousePressEvent(self, event):  # noqa: N802
        if event.button() != Qt.LeftButton or self.image_bgr is None:
            return
        point = self._widget_to_img(event.pos().x(), event.pos().y())
        if point is None:
            return
        self.drawing = True
        self.temp_start = point
        self.temp_end = point
        self.update()

    def mouseMoveEvent(self, event):  # noqa: N802
        if not self.drawing or self.image_bgr is None:
            return
        point = self._widget_to_img(event.pos().x(), event.pos().y())
        if point is None:
            return
        self.temp_end = point
        self.update()

    def mouseReleaseEvent(self, event):  # noqa: N802
        if event.button() != Qt.LeftButton or not self.drawing or self.image_bgr is None:
            return
        point = self._widget_to_img(event.pos().x(), event.pos().y())
        self.drawing = False
        if point is not None:
            self.temp_end = point
        if self.temp_start and self.temp_end:
            box = Box(self.active_class, self.temp_start[0], self.temp_start[1], self.temp_end[0], self.temp_end[1])
            if box.is_valid():
                self.boxes.append(box)
                self._set_dirty(True)
                self.box_count_changed.emit(len(self.boxes))
                self.status_message.emit(f"Added box: {self.classes[self.active_class]}")
            else:
                self.status_message.emit("Ignored tiny box")
        self.temp_start = None
        self.temp_end = None
        self.update()

    def keyPressEvent(self, event):  # noqa: N802
        key = event.key()
        if key == Qt.Key_S:
            self.save_annotations()
        elif key == Qt.Key_U:
            self.undo_last()
        elif key == Qt.Key_C:
            self.clear_all()
        elif key == Qt.Key_BracketRight:
            self.next_class()
        elif key == Qt.Key_BracketLeft:
            self.prev_class()
        elif Qt.Key_0 <= key <= Qt.Key_9:
            self.set_active_class(key - Qt.Key_0)
        else:
            super().keyPressEvent(event)

    def _set_dirty(self, value: bool):
        self.dirty = value
        self.dirty_changed.emit(value)

    def _fit_rect(self, img_w: int, img_h: int) -> QRect:
        area_w = max(1, self.width() - 20)
        area_h = max(1, self.height() - 20)
        scale = min(area_w / img_w, area_h / img_h)
        draw_w = int(img_w * scale)
        draw_h = int(img_h * scale)
        x = (self.width() - draw_w) // 2
        y = (self.height() - draw_h) // 2
        return QRect(x, y, draw_w, draw_h)

    def _widget_to_img(self, x: int, y: int) -> Optional[Tuple[int, int]]:
        if self.image_bgr is None or not self._display_rect.contains(x, y):
            return None
        rel_x = (x - self._display_rect.x()) / max(1, self._display_rect.width())
        rel_y = (y - self._display_rect.y()) / max(1, self._display_rect.height())
        img_x = int(rel_x * self.image_bgr.shape[1])
        img_y = int(rel_y * self.image_bgr.shape[0])
        img_x = max(0, min(img_x, self.image_bgr.shape[1] - 1))
        img_y = max(0, min(img_y, self.image_bgr.shape[0] - 1))
        return img_x, img_y

    def _img_to_widget(self, x: int, y: int) -> Tuple[int, int]:
        if self.image_bgr is None:
            return x, y
        px = self._display_rect.x() + int(x / self.image_bgr.shape[1] * self._display_rect.width())
        py = self._display_rect.y() + int(y / self.image_bgr.shape[0] * self._display_rect.height())
        return px, py

    @staticmethod
    def _to_pixmap(frame_bgr: np.ndarray) -> QPixmap:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        image = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        return QPixmap.fromImage(image.copy())

    @staticmethod
    def _bgr_to_qt(color_bgr: Tuple[int, int, int]):
        from PyQt5.QtGui import QColor

        b, g, r = color_bgr
        return QColor(r, g, b)
