# Edge AI Defect Detection Demo

PyQt5 desktop application for two related workflows:

- run local ONNX inference with a model already available on the edge device
- collect a few annotated samples, send them to a stronger remote machine, fine-tune a detector there, export ONNX, and pull the new model back for local inference

## Current end-to-end flow

1. Launch the local UI.
2. Capture and annotate 5 to 10 images in the `Data Capture` tab.
3. Click remote training from the `Training / Deployment` tab.
4. The app prepares a YOLO dataset from the active session, uploads it to the remote machine, starts fine-tuning from the base model already present there, exports ONNX, and downloads the ONNX back into this repository.
5. Start inference with the downloaded model.

The default local model path is now:

```text
<repo>/weights/current_model.onnx
```

In this repository that is:

[weights/current_model.onnx](/home/aru/side_work/edge_ai_demo_ultralytics_onnx/edge_ai_demo/weights/current_model.onnx)

## What the local app does

- starts a live camera preview
- runs local ONNX inference through Ultralytics
- captures frames and saves YOLO-format labels with a manual bounding-box annotation UI
- prepares a train/val dataset plus `dataset.yaml` from the current capture session
- connects to the remote machine over SSH/SFTP using Paramiko
- uploads the prepared dataset
- triggers remote training, validation, and ONNX export
- downloads the exported ONNX model and reloads it locally

Local runtime data is still stored under:

```text
~/edge_ai_demo_data/
├── captures/
└── config/ui_config.json
```

## Repository layout

```text
edge_ai_demo/
├── app/
├── config/
├── remote_training/
├── utils/
├── weights/
├── main.py
├── requirements.txt
└── README.md
```

## Local setup

```bash
sudo apt update
sudo apt install -y python3-pip python3-pyqt5 python3-opencv
cd edge_ai_demo
python3 -m pip install -r requirements.txt
python3 main.py
```

Top-level Python dependencies:

- `PyQt5`
- `opencv-python-headless`
- `numpy`
- `ultralytics`
- `onnxruntime`
- `paramiko`

## Dataset preparation

The capture session already uses YOLO detection labels:

```text
<session_dir>/
├── images/
└── labels/
```

When remote training starts, the app prepares:

```text
<session_dir>/prepared_dataset/
├── dataset.yaml
├── train/
│   ├── images/
│   └── labels/
└── val/
    ├── images/
    └── labels/
```

The generated `dataset.yaml` is intentionally minimal:

```yaml
train: train/images
val: val/images
nc: <class_count>
names: ['class_a', 'class_b']
```

You can also build this manually:

```bash
python3 utils/dataset_builder.py \
  --session ~/edge_ai_demo_data/captures/your_session \
  --classes scratch,dent,crack,missing_part
```

## Remote training workflow

The training tab now expects:

- remote host
- remote user
- remote project directory containing `fine_tune.py`, `validate.py`, and `export_model.py`
- remote dataset root where uploaded prepared datasets should land
- remote runs root where Ultralytics training outputs are written
- SSH key if needed

For a real run the app:

1. prepares the local dataset from the selected session
2. uploads it with Paramiko SFTP
3. runs `fine_tune.py --dataset-yaml <uploaded_yaml>`
4. runs `validate.py`
5. runs `export_model.py`
6. downloads `best.onnx` from the configured remote runs directory into the local model path

The remote base `.pt` model is still expected to already exist on the remote machine.

## Important implementation notes

- The active session path is now wired into the training tab automatically when you start a new capture session.
- The default class CSV in the training tab now matches the capture widget class list.
- The app no longer depends on local `ssh` or `scp` binaries for its main remote workflow.
- The helper script [remote_training/deploy_back.py](/home/aru/side_work/edge_ai_demo_ultralytics_onnx/edge_ai_demo/remote_training/deploy_back.py) still exists, but the UI does not need it for the main train-and-pull-back path.

## Useful files

- [main.py](/home/aru/side_work/edge_ai_demo_ultralytics_onnx/edge_ai_demo/main.py)
- [app/main_window.py](/home/aru/side_work/edge_ai_demo_ultralytics_onnx/edge_ai_demo/app/main_window.py)
- [app/managers/remote_training_client.py](/home/aru/side_work/edge_ai_demo_ultralytics_onnx/edge_ai_demo/app/managers/remote_training_client.py)
- [app/interfaces/predictor.py](/home/aru/side_work/edge_ai_demo_ultralytics_onnx/edge_ai_demo/app/interfaces/predictor.py)
- [app/widgets/data_capture_widget.py](/home/aru/side_work/edge_ai_demo_ultralytics_onnx/edge_ai_demo/app/widgets/data_capture_widget.py)
- [utils/dataset_builder.py](/home/aru/side_work/edge_ai_demo_ultralytics_onnx/edge_ai_demo/utils/dataset_builder.py)
- [remote_training/README.md](/home/aru/side_work/edge_ai_demo_ultralytics_onnx/edge_ai_demo/remote_training/README.md)
