#!/usr/bin/env python3
"""
Camera capture script with ID-based labeling.
Runs camera in background thread so capture is instant on Enter.

- Enter an ID to set it (no capture)
- Press Enter (empty) to capture with current ID
- Type 'r' to remove last captured image
- Type 'q' to quit

Filenames: ID00077_i0001.jpg

Usage:
    python3 capture.py --camera 0
"""

import cv2
import os
import glob
import subprocess
import argparse
import threading


SAVE_DIR = "captures"
os.makedirs(SAVE_DIR, exist_ok=True)


class CameraStream:
    """Continuously reads frames in a background thread."""
    def __init__(self, camera_id: int):
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera {camera_id}")
        self.frame = None
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                with self.lock:
                    self.frame = frame

    def get_frame(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.running = False
        self.thread.join(timeout=2)
        self.cap.release()


def parse_args():
    parser = argparse.ArgumentParser(description="Camera capture with ID labeling")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index (default: 0)")
    return parser.parse_args()


def get_next_index(save_dir: str, current_id: str) -> int:
    pattern = os.path.join(save_dir, f"{current_id}_i*.jpg")
    existing = glob.glob(pattern)
    if not existing:
        return 1
    indices = []
    for path in existing:
        base = os.path.splitext(os.path.basename(path))[0]
        try:
            indices.append(int(base.split("_i")[-1]))
        except ValueError:
            continue
    return max(indices) + 1 if indices else 1


def icat(image_path: str) -> None:
    try:
        subprocess.run(["kitty", "+kitten", "icat", "--align", "left", image_path])
    except FileNotFoundError:
        print("[WARN] kitty icat not available, skipping preview.")


def format_id(raw: str) -> str:
    raw = raw.strip()
    if raw.isdigit():
        return f"ID{int(raw):05d}"
    if raw.upper().startswith("ID"):
        num_part = raw[2:]
        if num_part.isdigit():
            return f"ID{int(num_part):05d}"
    return raw


def remove_last(last_saved: str) -> None:
    if last_saved is None:
        print("[WARN] No image to remove.")
        return
    if os.path.exists(last_saved):
        os.remove(last_saved)
        print(f"[INFO] Removed: {last_saved}")
    else:
        print(f"[WARN] File not found: {last_saved}")


def main():
    args = parse_args()

    print(f"[INFO] Starting camera {args.camera}...")
    stream = CameraStream(args.camera)

    # Wait until we get at least one frame
    import time
    for _ in range(20):
        if stream.get_frame() is not None:
            break
        time.sleep(0.1)
    else:
        print("[ERROR] Camera never produced a frame.")
        stream.stop()
        return

    print(f"[INFO] Camera {args.camera} ready.")
    print("[INFO] Enter an ID to set it, ENTER to capture, 'r' to remove last, 'q' to quit.\n")

    current_id = None
    capture_count = 0
    last_saved = None

    try:
        while True:
            if current_id:
                next_idx = get_next_index(SAVE_DIR, current_id)
                prompt = f"[{current_id}_i{next_idx:04d}] > "
            else:
                prompt = "[no ID set] > "

            try:
                user_input = input(prompt).strip()
            except (EOFError, KeyboardInterrupt):
                print("\n[INFO] Quitting.")
                break

            if user_input.lower() == 'q':
                print("[INFO] Quitting.")
                break

            elif user_input.lower() == 'r':
                remove_last(last_saved)
                last_saved = None

            elif user_input == "":
                if current_id is None:
                    print("[WARN] No ID set. Enter an ID first.")
                    continue

                frame = stream.get_frame()
                if frame is None:
                    print("[WARN] No frame available, try again.")
                    continue

                idx = get_next_index(SAVE_DIR, current_id)
                filename = f"{current_id}_i{idx:04d}.jpg"
                filepath = os.path.join(SAVE_DIR, filename)
                cv2.imwrite(filepath, frame)
                capture_count += 1
                last_saved = filepath
                print(f"[INFO] Saved: {filepath}")
                icat(filepath)

            else:
                new_id = format_id(user_input)
                current_id = new_id
                existing_count = len(glob.glob(os.path.join(SAVE_DIR, f"{current_id}_i*.jpg")))
                print(f"[INFO] ID set to '{current_id}' ({existing_count} existing capture(s))")

    finally:
        stream.stop()
        print(f"[INFO] Done. {capture_count} image(s) captured this session.")


if __name__ == "__main__":
    main()
