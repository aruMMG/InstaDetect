# Remote training scripts

These scripts are meant to run on the stronger remote machine that the local UI connects to.

## What they do

- `fine_tune.py`: train from the base `.pt` model already present on the remote machine
- `validate.py`: validate `best.pt`
- `export_model.py`: export the trained checkpoint to ONNX
- `deploy_back.py`: optional helper that pushes the model to another device with `ssh`/`scp`

## Current data flow

The local UI now prepares a YOLO dataset from the selected capture session before remote training starts.

The uploaded dataset looks like this on the remote machine:

```text
<remote_dataset_root>/<dataset_name>/
├── dataset.yaml
├── train/
│   ├── images/
│   └── labels/
└── val/
    ├── images/
    └── labels/
```

`fine_tune.py` accepts either:

- `--dataset-yaml <path>` for a session-specific uploaded dataset
- or the fallback `dataset_yaml` value from `remote_config.json`

## Setup on the remote machine

```bash
sudo apt update
sudo apt install -y python3-venv
mkdir -p ~/edge_remote_project
cd ~/edge_remote_project
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r remote_training/requirements.txt
cp remote_training/remote_config.json.example remote_training/remote_config.json
```

Then edit `remote_training/remote_config.json` with the remote machine paths for:

- `base_model_pt`
- `workspace_dir`
- optional fallback `dataset_yaml`
- deployment destinations if you still use `deploy_back.py`

## Folder layout example

```text
~/edge_remote_project/
├── .venv/
├── base_models/
│   └── yolo11n.pt
├── datasets/
│   └── <uploaded_session_dataset>/
├── workspace/
├── remote_training/
│   ├── fine_tune.py
│   ├── validate.py
│   ├── export_model.py
│   ├── deploy_back.py
│   ├── requirements.txt
│   └── remote_config.json
```

## Manual test examples

```bash
source .venv/bin/activate
cd ~/edge_remote_project/remote_training
python fine_tune.py --model defect_demo_v1 --dataset-yaml ~/edge_remote_project/datasets/sample_run/dataset.yaml
python validate.py --model defect_demo_v1
python export_model.py --model defect_demo_v1
```
