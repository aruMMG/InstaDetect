import argparse
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from common import load_config


def run(cmd):
    print("[deploy]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    cfg = load_config()
    weights_dir = Path(cfg["workspace_dir"]) / args.model / "weights"
    local_best = weights_dir / "best.pt"
    local_onnx = weights_dir / "best.onnx"

    if not local_onnx.exists() and local_best.exists():
        fallback = local_best.with_suffix(".onnx")
        if fallback.exists():
            local_onnx = fallback

    if not local_onnx.exists():
        onnx_candidates = sorted(weights_dir.glob("*.onnx"), key=lambda p: p.stat().st_mtime, reverse=True)
        if onnx_candidates:
            local_onnx = onnx_candidates[0]

    if not local_onnx.exists():
        raise FileNotFoundError(f"No exported ONNX file found under {weights_dir}")

    pi_host = cfg["pi_host"]
    pi_user = cfg["pi_user"]
    pi_ssh_key = os.path.expanduser(cfg["pi_ssh_key"])
    pi_model_path = cfg["pi_deploy_model_path"]
    pi_version_json_path = cfg["pi_version_json_path"]

    metadata = {
        "model_name": args.model,
        "source_onnx": str(local_onnx),
        "deployed_at_utc": datetime.now(timezone.utc).isoformat(),
        "format": "onnx"
    }

    local_meta = weights_dir / "deploy_metadata.json"
    with open(local_meta, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    remote_model_dir = str(Path(pi_model_path).parent)
    remote_meta_dir = str(Path(pi_version_json_path).parent)

    run(["ssh", "-i", pi_ssh_key, f"{pi_user}@{pi_host}", f"mkdir -p {remote_model_dir} {remote_meta_dir}"])
    run(["scp", "-i", pi_ssh_key, str(local_onnx), f"{pi_user}@{pi_host}:{pi_model_path}"])
    run(["scp", "-i", pi_ssh_key, str(local_meta), f"{pi_user}@{pi_host}:{pi_version_json_path}"])

    print(f"[deploy] deployed model to {pi_user}@{pi_host}:{pi_model_path}")


if __name__ == "__main__":
    main()
