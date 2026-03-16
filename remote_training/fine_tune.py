import argparse
from pathlib import Path

from ultralytics import YOLO

from common import ensure_dir, load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", default="", help="Pi session path, kept for logging/auditing")
    parser.add_argument("--classes", default="", help="Comma-separated classes, optional for your own pipeline")
    parser.add_argument("--dataset-yaml", default="", help="Optional dataset YAML path for this run")
    parser.add_argument("--model", required=True, help="Run name / model name")
    args = parser.parse_args()

    cfg = load_config()
    workspace_dir = ensure_dir(cfg["workspace_dir"])
    dataset_yaml = args.dataset_yaml or cfg["dataset_yaml"]
    base_model_pt = cfg["base_model_pt"]
    epochs = int(cfg.get("train_epochs", 20))
    imgsz = int(cfg.get("train_imgsz", 640))
    batch = int(cfg.get("train_batch", 8))

    print(f"[fine_tune] session={args.session}")
    print(f"[fine_tune] classes={args.classes}")
    print(f"[fine_tune] dataset_yaml={dataset_yaml}")
    print(f"[fine_tune] base_model={base_model_pt}")
    print(f"[fine_tune] workspace={workspace_dir}")
    print(f"[fine_tune] model_name={args.model}")

    model = YOLO(base_model_pt)
    model.train(
        data=dataset_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=workspace_dir,
        name=args.model,
        exist_ok=True,
    )

    best_pt = Path(workspace_dir) / args.model / "weights" / "best.pt"
    print(f"[fine_tune] best_pt={best_pt}")


if __name__ == "__main__":
    main()
