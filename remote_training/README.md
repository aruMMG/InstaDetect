# Remote training scripts for Ubuntu laptop

These scripts are meant to live on the stronger remote machine that the Raspberry Pi UI calls over SSH.

## What they do
- `fine_tune.py`: trains a detector from your base `.pt` model
- `validate.py`: validates the trained `.pt` checkpoint
- `export_model.py`: exports `best.pt` to `best.onnx`
- `deploy_back.py`: copies the ONNX model back to the Raspberry Pi

## Important data assumption
The current UI capture screen stores images as:

```text
<session>/<class_name>/<image>.jpg
```

That is **not** YOLO detection dataset format. For detection fine-tuning, your real dataset should already exist in YOLO format, or you should add your own preprocessing step before `fine_tune.py`.

The scripts below therefore use `dataset_yaml` from `remote_config.json` as the source of truth for training.

## Setup on Ubuntu laptop

```bash
sudo apt update
sudo apt install -y python3-venv rsync openssh-client
mkdir -p ~/edge_remote_project
cd ~/edge_remote_project
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r remote_training/requirements.txt
cp remote_training/remote_config.json.example remote_training/remote_config.json
```

Then edit `remote_training/remote_config.json` for your laptop and Pi paths.

## Folder layout example

```text
~/edge_remote_project/
├── .venv/
├── base_models/
│   └── yolo11n.pt
├── data/
│   └── defect_dataset.yaml
├── workspace/
├── remote_training/
│   ├── fine_tune.py
│   ├── validate.py
│   ├── export_model.py
│   ├── deploy_back.py
│   ├── requirements.txt
│   └── remote_config.json
```

## Test each script manually

```bash
source .venv/bin/activate
cd ~/edge_remote_project/remote_training
python fine_tune.py --model defect_demo_v1
python validate.py --model defect_demo_v1
python export_model.py --model defect_demo_v1
python deploy_back.py --model defect_demo_v1
```
