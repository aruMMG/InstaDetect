#!/usr/bin/env python3
"""
Industrial Defect Annotation Tool for YOLO
Single-file OpenCV annotation utility with two modes:
1) Camera capture + annotate
2) Local directory annotation

Features:
- Mouse-based box drawing with live preview
- Multiple boxes per image
- Multiple classes with keyboard switching
- Load/save YOLO labels
- Undo / clear / skip / save / next / previous
- Empty txt files supported for no-defect images

Author: OpenAI
"""

import argparse
import os
import sys
import cv2
import time
import glob
import traceback
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple


VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class Box:
    class_id: int
    x1: int
    y1: int
    x2: int
    y2: int

    def normalized(self, img_w: int, img_h: int) -> Tuple[float, float, float, float]:
        x1, y1, x2, y2 = self.clamped(img_w, img_h)
        bw = max(0, x2 - x1)
        bh = max(0, y2 - y1)
        xc = x1 + bw / 2.0
        yc = y1 + bh / 2.0
        return xc / img_w, yc / img_h, bw / img_w, bh / img_h

    def clamped(self, img_w: int, img_h: int) -> Tuple[int, int, int, int]:
        x1 = max(0, min(self.x1, img_w - 1))
        y1 = max(0, min(self.y1, img_h - 1))
        x2 = max(0, min(self.x2, img_w - 1))
        y2 = max(0, min(self.y2, img_h - 1))
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        return x1, y1, x2, y2

    def is_valid(self, min_size: int = 3) -> bool:
        return abs(self.x2 - self.x1) >= min_size and abs(self.y2 - self.y1) >= min_size


class YoloIO:
    @staticmethod
    def load_labels(label_path: str, img_w: int, img_h: int) -> List[Box]:
        boxes = []
        if not os.path.exists(label_path):
            return boxes

        try:
            with open(label_path, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
        except Exception as e:
            print(f"[WARN] Failed to read label file: {label_path} | {e}")
            return boxes

        for line_no, line in enumerate(lines, start=1):
            parts = line.split()
            if len(parts) != 5:
                print(f"[WARN] Invalid label line {line_no} in {label_path}: {line}")
                continue
            try:
                class_id = int(float(parts[0]))
                xc = float(parts[1])
                yc = float(parts[2])
                bw = float(parts[3])
                bh = float(parts[4])

                x1 = int(round((xc - bw / 2.0) * img_w))
                y1 = int(round((yc - bh / 2.0) * img_h))
                x2 = int(round((xc + bw / 2.0) * img_w))
                y2 = int(round((yc + bh / 2.0) * img_h))
                boxes.append(Box(class_id, x1, y1, x2, y2))
            except Exception as e:
                print(f"[WARN] Failed parsing line {line_no} in {label_path}: {e}")
        return boxes

    @staticmethod
    def save_labels(label_path: str, boxes: List[Box], img_w: int, img_h: int) -> None:
        os.makedirs(os.path.dirname(label_path), exist_ok=True)
        with open(label_path, "w", encoding="utf-8") as f:
            for box in boxes:
                if not box.is_valid():
                    continue
                class_id = max(0, int(box.class_id))
                xc, yc, bw, bh = box.normalized(img_w, img_h)
                f.write(f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")


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


class Annotator:
    def __init__(self, window_name: str, classes: List[str]):
        self.window_name = window_name
        self.classes = classes if classes else ["defect"]
        self.active_class = 0

        self.image = None
        self.image_path = None
        self.label_path = None

        self.boxes: List[Box] = []
        self.temp_start: Optional[Tuple[int, int]] = None
        self.temp_end: Optional[Tuple[int, int]] = None
        self.drawing = False
        self.dirty = False

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)

    def load(self, image, image_path: str, label_path: str) -> None:
        self.image = image.copy()
        self.image_path = image_path
        self.label_path = label_path
        self.boxes = YoloIO.load_labels(label_path, self.image.shape[1], self.image.shape[0])
        self.temp_start = None
        self.temp_end = None
        self.drawing = False
        self.dirty = False

    def set_boxes(self, boxes: List[Box]) -> None:
        self.boxes = list(boxes)
        self.dirty = True

    def save(self) -> bool:
        if self.image is None or self.label_path is None:
            return False
        try:
            h, w = self.image.shape[:2]
            os.makedirs(os.path.dirname(self.label_path), exist_ok=True)

            with open(self.label_path, "w", encoding="utf-8") as f:
                for box in self.boxes:
                    if not box.is_valid():
                        continue
                    class_id = int(box.class_id)
                    xc, yc, bw, bh = box.normalized(w, h)
                    f.write(f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

            print(f"[INFO] Saved label file: {self.label_path}")
            self.dirty = False
            return True
        except Exception as e:
            print(f"[ERROR] Failed to save labels: {self.label_path} | {e}")
            return False

    def undo_last(self) -> None:
        if self.boxes:
            removed = self.boxes.pop()
            self.dirty = True
            print(f"[INFO] Removed box class={removed.class_id}")

    def clear_all(self) -> None:
        if self.boxes:
            self.boxes.clear()
            self.dirty = True
            print("[INFO] Cleared all boxes")

    def next_class(self) -> None:
        self.active_class = (self.active_class + 1) % len(self.classes)

    def prev_class(self) -> None:
        self.active_class = (self.active_class - 1) % len(self.classes)

    def set_class_by_digit(self, digit: int) -> None:
        if 0 <= digit < len(self.classes):
            self.active_class = digit

    def render(self) -> None:
        if self.image is None:
            return

        img = self.image.copy()

        # Draw saved boxes on image
        for idx, box in enumerate(self.boxes):
            color = Colors.get(box.class_id)
            x1, y1, x2, y2 = box.clamped(img.shape[1], img.shape[0])
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            class_name = self.classes[box.class_id] if 0 <= box.class_id < len(self.classes) else str(box.class_id)
            label = f"{idx}: {class_name}"
            cv2.putText(
                img, label, (x1, max(20, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA
            )

        # Draw preview box
        if self.drawing and self.temp_start and self.temp_end:
            x1, y1 = self.temp_start
            x2, y2 = self.temp_end
            color = Colors.get(self.active_class)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
            cv2.putText(
                img, f"NEW: {self.classes[self.active_class]}",
                (x1, max(20, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA
            )

        # Build side info panel
        panel = self._build_side_panel(img.shape[0])

        # Concatenate image + panel
        canvas = cv2.hconcat([img, panel])
        cv2.imshow(self.window_name, canvas)
    def _build_side_panel(self, height: int):
        panel_width = 320
        panel = np.zeros((height, panel_width, 3), dtype=np.uint8)
        panel[:] = (25, 25, 25)

        filename = os.path.basename(self.image_path) if self.image_path else "N/A"
        active_name = self.classes[self.active_class]
        dirty_text = "YES" if self.dirty else "NO"

        lines = [
            "YOLO ANNOTATOR",
            "",
            f"File: {filename}",
            f"Active class: [{self.active_class}] {active_name}",
            f"Box count: {len(self.boxes)}",
            f"Unsaved: {dirty_text}",
            "",
            "Mouse:",
            "  Left drag -> draw box",
            "",
            "Keys:",
            "  s -> save",
            "  u -> undo last",
            "  c -> clear all",
            "  [ / ] -> prev/next class",
            "  0-9 -> select class",
            "  n -> next image",
            "  p -> previous image",
            "  k -> skip image",
            "  q / Esc -> quit",
            "",
            "Classes:",
        ]

        y = 30
        line_h = 24

        for i, line in enumerate(lines):
            color = (230, 230, 230)
            scale = 0.6
            thickness = 1

            if i == 0:
                color = (0, 255, 255)
                scale = 0.75
                thickness = 2

            cv2.putText(
                panel, line, (12, y),
                cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA
            )
            y += line_h

        # Show class list with active highlight
        for cid, cname in enumerate(self.classes):
            color = Colors.get(cid) if cid == self.active_class else (180, 180, 180)
            prefix = "->" if cid == self.active_class else "  "
            text = f"{prefix} [{cid}] {cname}"
            cv2.putText(
                panel, text, (12, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2 if cid == self.active_class else 1, cv2.LINE_AA
            )
            y += 24

        return panel
    def _draw_overlay(self, canvas) -> None:
        h, w = canvas.shape[:2]
        filename = os.path.basename(self.image_path) if self.image_path else "N/A"
        active_name = self.classes[self.active_class]
        box_count = len(self.boxes)
        dirty_text = " *UNSAVED*" if self.dirty else ""

        lines = [
            f"File: {filename}",
            f"Active Class [{self.active_class}]: {active_name}",
            f"Boxes: {box_count}{dirty_text}",
            "Keys: s=save, u=undo, c=clear, n=next, p=prev, k=skip, q=quit",
            "Class: [ / ] or number keys 0-9",
            "Mouse: left-drag to draw box",
        ]

        pad = 10
        line_h = 22
        overlay_h = pad * 2 + line_h * len(lines)
        cv2.rectangle(canvas, (0, 0), (min(w, 800), overlay_h), (20, 20, 20), -1)
        cv2.rectangle(canvas, (0, 0), (min(w, 800), overlay_h), (80, 80, 80), 1)

        y = pad + 16
        for i, line in enumerate(lines):
            color = (255, 255, 255)
            if i == 1:
                color = Colors.get(self.active_class)
            cv2.putText(canvas, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 1, cv2.LINE_AA)
            y += line_h

    def _mouse_callback(self, event, x, y, flags, param) -> None:
        if self.image is None:
            return

        x = max(0, min(x, self.image.shape[1] - 1))
        y = max(0, min(y, self.image.shape[0] - 1))

        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.temp_start = (x, y)
            self.temp_end = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.temp_end = (x, y)

        elif event == cv2.EVENT_LBUTTONUP and self.drawing:
            self.drawing = False
            self.temp_end = (x, y)
            if self.temp_start is not None and self.temp_end is not None:
                box = Box(self.active_class, self.temp_start[0], self.temp_start[1], self.temp_end[0], self.temp_end[1])
                if box.is_valid():
                    self.boxes.append(box)
                    self.dirty = True
                    print(f"[INFO] Added box class={self.active_class}: {box}")
                else:
                    print("[INFO] Ignored tiny box")
            self.temp_start = None
            self.temp_end = None


class DirectoryMode:
    def __init__(self, input_dir: str, labels_dir: str, classes: List[str],
                 img_exts: List[str], start_index: int = 0):
        self.input_dir = input_dir
        self.labels_dir = labels_dir
        self.classes = classes
        self.img_exts = img_exts
        self.index = start_index
        self.annotator = Annotator("YOLO Annotator - Directory Mode", classes)
        self.images = self._collect_images()

        if not self.images:
            raise RuntimeError(f"No images found in: {input_dir}")

        if self.index < 0 or self.index >= len(self.images):
            self.index = 0

    def _collect_images(self) -> List[str]:
        image_paths = []
        for ext in self.img_exts:
            pattern = os.path.join(self.input_dir, f"*{ext}")
            image_paths.extend(glob.glob(pattern))
        image_paths = sorted(set(image_paths))
        return image_paths

    def _label_path_for_image(self, image_path: str) -> str:
        base = os.path.splitext(os.path.basename(image_path))[0]
        return os.path.join(self.labels_dir, base + ".txt")

    def _load_current(self) -> bool:
        if not (0 <= self.index < len(self.images)):
            return False

        image_path = self.images[self.index]
        image = cv2.imread(image_path)
        if image is None:
            print(f"[WARN] Failed to load image: {image_path}")
            return False

        label_path = self._label_path_for_image(image_path)
        self.annotator.load(image, image_path, label_path)
        return True

    def run(self) -> None:
        print("[INFO] Starting directory annotation mode")
        print(f"[INFO] Found {len(self.images)} images")

        if not self._load_current():
            raise RuntimeError("Failed to load initial image.")

        while True:
            self.annotator.render()
            key = cv2.waitKey(20) & 0xFF

            if key == 255:
                continue

            action = self._handle_key(key)
            if action == "quit":
                break
            elif action == "next":
                if self.index < len(self.images) - 1:
                    self.index += 1
                    self._load_current()
                else:
                    print("[INFO] Reached last image")
            elif action == "prev":
                if self.index > 0:
                    self.index -= 1
                    self._load_current()
                else:
                    print("[INFO] Already at first image")

        cv2.destroyAllWindows()

    def _handle_key(self, key: int) -> Optional[str]:
        # save
        if key in (ord('s'), ord('S')):
            self.annotator.save()

        # undo
        elif key in (ord('u'), ord('U')):
            self.annotator.undo_last()

        # clear
        elif key in (ord('c'), ord('C')):
            self.annotator.clear_all()

        elif key in (ord('n'), ord('N'), 83):
            self.annotator.save()
            return "next"

        elif key in (ord('p'), ord('P'), 81):
            self.annotator.save()
            return "prev"

        elif key in (ord('k'), ord('K')):
            self.annotator.save()
            print("[INFO] Skipping image")
            return "next"

        # quit
        elif key in (ord('q'), ord('Q'), 27):  # esc
            if self.annotator.dirty:
                print("[WARN] Unsaved changes exist. Press q again to quit or save first.")
                self.annotator.dirty = False  # one-time safe quit prompt behavior
            else:
                return "quit"

        # class switching
        elif key == ord(']'):
            self.annotator.next_class()
        elif key == ord('['):
            self.annotator.prev_class()

        # number keys 0-9
        elif ord('0') <= key <= ord('9'):
            self.annotator.set_class_by_digit(key - ord('0'))

        return None


class CameraMode:
    def __init__(self, camera_id: int, output_dir: str, labels_dir: str, classes: List[str]):
        self.camera_id = camera_id
        self.output_dir = output_dir
        self.labels_dir = labels_dir
        self.classes = classes
        self.annotator = Annotator("YOLO Annotator - Camera Mode", classes)
        self.capture_count = 0

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.labels_dir, exist_ok=True)

    def _generate_capture_paths(self) -> Tuple[str, str]:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"capture_{timestamp}_{self.capture_count:04d}.jpg"
        image_path = os.path.join(self.output_dir, filename)
        label_path = os.path.join(self.labels_dir, os.path.splitext(filename)[0] + ".txt")
        self.capture_count += 1
        return image_path, label_path

    def run(self) -> None:
        cap = cv2.VideoCapture(self.camera_id)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera: {self.camera_id}")

        print("[INFO] Starting camera mode")
        print("[INFO] Press SPACE or ENTER to capture, q to quit live view")

        cv2.namedWindow("Camera Live Feed", cv2.WINDOW_NORMAL)

        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    print("[WARN] Failed to read frame from camera")
                    key = cv2.waitKey(30) & 0xFF
                    if key in (ord('q'), ord('Q'), 27):
                        break
                    continue

                display = frame.copy()
                info_lines = [
                    f"Camera ID: {self.camera_id}",
                    "SPACE/ENTER: capture frame",
                    "Q/ESC: quit",
                ]
                y = 25
                for line in info_lines:
                    cv2.putText(display, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (0, 255, 0), 2, cv2.LINE_AA)
                    y += 30

                cv2.imshow("Camera Live Feed", display)
                key = cv2.waitKey(20) & 0xFF

                if key in (ord('q'), ord('Q'), 27):
                    break
                elif key in (32, 13):  # space or enter
                    image_path, label_path = self._generate_capture_paths()
                    if not cv2.imwrite(image_path, frame):
                        print(f"[ERROR] Failed to save captured image: {image_path}")
                        continue

                    print(f"[INFO] Captured: {image_path}")
                    self.annotator.load(frame, image_path, label_path)
                    result = self._run_annotation_loop()

                    if result == "quit_all":
                        break

        finally:
            cap.release()
            cv2.destroyAllWindows()

    def _run_annotation_loop(self) -> Optional[str]:
        while True:
            self.annotator.render()
            key = cv2.waitKey(20) & 0xFF

            if key == 255:
                continue

            # save
            if key in (ord('s'), ord('S')):
                self.annotator.save()

            # undo
            elif key in (ord('u'), ord('U')):
                self.annotator.undo_last()

            # clear
            elif key in (ord('c'), ord('C')):
                self.annotator.clear_all()

            # skip / leave annotation without deleting image
            elif key in (ord('k'), ord('K'), ord('n'), ord('N')):
                self.annotator.save()
                print("[INFO] Saved and returning to live camera view")
                break

            elif key in (ord('q'), ord('Q'), 27):
                self.annotator.save()
                return "quit_all"

            # class switching
            elif key == ord(']'):
                self.annotator.next_class()
            elif key == ord('['):
                self.annotator.prev_class()

            # number keys 0-9
            elif ord('0') <= key <= ord('9'):
                self.annotator.set_class_by_digit(key - ord('0'))

        return None


def parse_classes(classes_str: str) -> List[str]:
    if not classes_str:
        return ["defect"]
    classes = [c.strip() for c in classes_str.split(",") if c.strip()]
    return classes if classes else ["defect"]


def parse_img_exts(exts_str: str) -> List[str]:
    if not exts_str:
        return [".jpg", ".jpeg", ".png"]
    exts = []
    for ext in exts_str.split(","):
        ext = ext.strip().lower()
        if not ext:
            continue
        if not ext.startswith("."):
            ext = "." + ext
        exts.append(ext)
    return exts if exts else [".jpg", ".jpeg", ".png"]


def ensure_dir(path: Optional[str], desc: str) -> None:
    if not path:
        raise ValueError(f"{desc} path is empty")
    os.makedirs(path, exist_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Industrial defect annotation tool for YOLO using OpenCV"
    )
    parser.add_argument("--mode", required=True, choices=["camera", "dir"],
                        help="Run mode: camera or dir")
    parser.add_argument("--camera_id", type=int, default=0,
                        help="Camera ID for camera mode")
    parser.add_argument("--input_dir", type=str, default="",
                        help="Input directory for directory mode")
    parser.add_argument("--output_dir", type=str, default="captures",
                        help="Output directory for captured images")
    parser.add_argument("--labels_dir", type=str, default="labels",
                        help="Directory for YOLO txt labels")
    parser.add_argument("--classes", type=str, default="defect",
                        help="Comma-separated class names, e.g. scratch,dent,crack")
    parser.add_argument("--img_ext", type=str, default=".jpg,.jpeg,.png,.bmp",
                        help="Comma-separated image extensions")
    parser.add_argument("--start_index", type=int, default=0,
                        help="Start image index in directory mode")
    return parser


def validate_args(args) -> None:
    if args.mode == "dir":
        if not args.input_dir:
            raise ValueError("--input_dir is required in dir mode")
        if not os.path.isdir(args.input_dir):
            raise FileNotFoundError(f"Input directory not found: {args.input_dir}")

    ensure_dir(args.labels_dir, "Labels directory")

    if args.mode == "camera":
        ensure_dir(args.output_dir, "Output directory")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        validate_args(args)

        classes = parse_classes(args.classes)
        img_exts = parse_img_exts(args.img_ext)

        print("[INFO] Classes:", classes)
        print("[INFO] Image extensions:", img_exts)

        if args.mode == "camera":
            app = CameraMode(
                camera_id=args.camera_id,
                output_dir=args.output_dir,
                labels_dir=args.labels_dir,
                classes=classes,
            )
            app.run()

        elif args.mode == "dir":
            app = DirectoryMode(
                input_dir=args.input_dir,
                labels_dir=args.labels_dir,
                classes=classes,
                img_exts=img_exts,
                start_index=args.start_index,
            )
            app.run()

        return 0

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
        return 130
    except Exception as e:
        print(f"[ERROR] {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())