import argparse
from pathlib import Path

from ultralytics import YOLO

from common import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset-yaml", default="")
    args = parser.parse_args()

    cfg = load_config()
    best_pt = Path(cfg["workspace_dir"]) / args.model / "weights" / "best.pt"
    dataset_yaml = args.dataset_yaml or cfg["dataset_yaml"]

    if not best_pt.exists():
        raise FileNotFoundError(f"best.pt not found: {best_pt}")

    print(f"[validate] model={best_pt}")
    print(f"[validate] dataset_yaml={dataset_yaml}")
    model = YOLO(str(best_pt))
    metrics = model.val(data=dataset_yaml)

    map50 = getattr(metrics.box, "map50", None)
    map5095 = getattr(metrics.box, "map", None)
    print(f"[validate] mAP50={map50}")
    print(f"[validate] mAP50-95={map5095}")


if __name__ == "__main__":
    main()
