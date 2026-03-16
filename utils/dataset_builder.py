from __future__ import annotations

import argparse
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence


@dataclass(frozen=True)
class PreparedDataset:
    session_dir: Path
    output_dir: Path
    dataset_yaml: Path
    train_count: int
    val_count: int
    classes: List[str]


def prepare_yolo_dataset(
    session_dir: str,
    class_names: Sequence[str],
    output_dir: str | None = None,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> PreparedDataset:
    session_path = Path(session_dir).expanduser().resolve()
    if not session_path.exists():
        raise FileNotFoundError(f"Session directory does not exist: {session_path}")

    images_dir = session_path / "images"
    labels_dir = session_path / "labels"
    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise FileNotFoundError(
            f"Expected YOLO session layout with images/ and labels/ under {session_path}"
        )

    classes = _normalize_classes(class_names)
    image_files = sorted(path for path in images_dir.iterdir() if path.is_file())
    pairs = [(image_path, labels_dir / f"{image_path.stem}.txt") for image_path in image_files]
    pairs = [(image_path, label_path) for image_path, label_path in pairs if label_path.exists()]

    if len(pairs) < 2:
        raise ValueError("Need at least 2 labeled captures in the session to build train/val splits")

    _validate_label_classes(pairs, classes)

    dataset_dir = Path(output_dir).expanduser().resolve() if output_dir else session_path / "prepared_dataset"
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)

    for split in ("train", "val"):
        (dataset_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (dataset_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    shuffled = list(pairs)
    random.Random(seed).shuffle(shuffled)

    val_count = max(1, int(round(len(shuffled) * val_ratio)))
    val_count = min(val_count, len(shuffled) - 1)
    train_pairs = shuffled[:-val_count]
    val_pairs = shuffled[-val_count:]

    _copy_pairs(train_pairs, dataset_dir / "train")
    _copy_pairs(val_pairs, dataset_dir / "val")

    dataset_yaml = dataset_dir / "dataset.yaml"
    dataset_yaml.write_text(_render_dataset_yaml(classes), encoding="utf-8")

    return PreparedDataset(
        session_dir=session_path,
        output_dir=dataset_dir,
        dataset_yaml=dataset_yaml,
        train_count=len(train_pairs),
        val_count=len(val_pairs),
        classes=list(classes),
    )


def _normalize_classes(class_names: Sequence[str]) -> List[str]:
    classes = [name.strip() for name in class_names if name.strip()]
    if not classes:
        raise ValueError("At least one class name is required")
    return classes


def _validate_label_classes(pairs: Iterable[tuple[Path, Path]], classes: Sequence[str]) -> None:
    max_class_id = len(classes) - 1
    for _image_path, label_path in pairs:
        for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) != 5:
                raise ValueError(f"Invalid YOLO label row in {label_path}:{line_number}")
            class_id = int(float(parts[0]))
            if class_id < 0 or class_id > max_class_id:
                raise ValueError(
                    f"Class id {class_id} in {label_path}:{line_number} exceeds configured classes {list(classes)}"
                )


def _copy_pairs(pairs: Sequence[tuple[Path, Path]], split_dir: Path) -> None:
    images_out = split_dir / "images"
    labels_out = split_dir / "labels"
    for image_path, label_path in pairs:
        shutil.copy2(image_path, images_out / image_path.name)
        shutil.copy2(label_path, labels_out / label_path.name)


def _render_dataset_yaml(classes: Sequence[str]) -> str:
    names = ", ".join(f"'{name}'" for name in classes)
    return "\n".join(
        [
            "train: train/images",
            "val: val/images",
            f"nc: {len(classes)}",
            f"names: [{names}]",
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare a YOLO dataset from a captured session")
    parser.add_argument("--session", required=True, help="Path to a capture session containing images/ and labels/")
    parser.add_argument("--classes", required=True, help="Comma-separated class names in label index order")
    parser.add_argument("--output", default="", help="Optional output directory for the prepared dataset")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio")
    args = parser.parse_args()

    prepared = prepare_yolo_dataset(
        session_dir=args.session,
        class_names=args.classes.split(","),
        output_dir=args.output or None,
        val_ratio=args.val_ratio,
    )

    print(f"session={prepared.session_dir}")
    print(f"output_dir={prepared.output_dir}")
    print(f"dataset_yaml={prepared.dataset_yaml}")
    print(f"train_count={prepared.train_count}")
    print(f"val_count={prepared.val_count}")
    print(f"classes={prepared.classes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
