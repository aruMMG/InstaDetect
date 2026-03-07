import json
import os
from pathlib import Path


def load_config():
    cfg_path = Path(__file__).resolve().parent / "remote_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"Missing config file: {cfg_path}. Copy remote_config.json.example to remote_config.json and edit it."
        )
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path
