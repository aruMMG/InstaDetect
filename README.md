# Edge AI Industrial Defect Detection Demo

PyQt5 desktop UI for a Raspberry Pi or Ubuntu laptop demo with:
- live camera preview
- local YOLO inference
- few-shot image capture
- remote training trigger over SSH
- model polling and reload
- timestamped logs

## Local app setup (Ubuntu laptop)

```bash
sudo apt update
sudo apt install -y python3-pip python3-pyqt5 python3-opencv
cd edge_ai_demo
python3 -m pip install -r requirements.txt
mkdir -p weights
# put your exported ONNX here, for example:
# weights/best.onnx
python3 main.py
```

## Model path for your test
In the UI, set **Training / Deployment -> Model file** to either:
- a specific file, for example `~/your_project/weights/best.onnx`
- or a directory, for example `~/your_project/weights`

If you pass a directory, the app tries:
1. `current_model.onnx`
2. `best.onnx`
3. newest `.onnx` file in that directory

## Notes
- The UI uses `UltralyticsOnnxPredictor` in `app/interfaces/predictor.py`
- The ONNX model is loaded with `YOLO(model_path, task="detect")`
- For CPU demo runs on Pi or laptop, ONNX is a practical choice
