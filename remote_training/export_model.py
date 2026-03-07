import argparse
from pathlib import Path

from ultralytics import YOLO

from common import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    cfg = load_config()
    best_pt = Path(cfg["workspace_dir"]) / args.model / "weights" / "best.pt"
    if not best_pt.exists():
        raise FileNotFoundError(f"best.pt not found: {best_pt}")

    print(f"[export] model={best_pt}")
    model = YOLO(str(best_pt))
    exported_path = model.export(format="onnx")
    print(f"[export] onnx={exported_path}")


if __name__ == "__main__":
    main()
