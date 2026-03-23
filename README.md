# ScreenFilter (offline messenger-screen collector)

Collect screenshots that contain messenger apps (WhatsApp, Telegram, Teams, Slack, Discord) using a **local YOLOv8 (PyTorch)** model.

## What this project does
- **Train** a YOLOv8 detector on your labeled screenshots (offline).
- **Predict** on screenshots.
- **Collect/filter**: copy only screenshots that contain any messenger app window.

This project does **not** extract message text; it only detects messenger app presence (and optionally bounding boxes).

## Quickstart

### 1) Create environment + install deps

> **Important:** PyTorch and Ultralytics currently support up to Python 3.11.  
> Use Python **3.10 or 3.11** for this project (not 3.12/3.13).

If you use `pyenv` (recommended on macOS):

```bash
pyenv install 3.11.9        # or any 3.10/3.11 you prefer
pyenv local 3.11.9          # inside this ScreenFilter folder

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

### 2) Prepare dataset (YOLO detection format)

Put your images + labels under:
- `data/messengers/images/train/`
- `data/messengers/images/val/`
- `data/messengers/labels/train/`
- `data/messengers/labels/val/`

Label format (one text file per image, same basename):

```
<class_id> <x_center> <y_center> <width> <height>
```

All coordinates are **normalized** (0..1). Classes are defined in `configs/messengers.yaml`.

### 3) Train (offline)

If you already have a local YOLOv8 base weight, place it in `weights/` and point `--model` to it.

Example:

```bash
screenfilter train --data configs/messengers.yaml --model weights/yolov8n.pt --epochs 50 --imgsz 960
```

Outputs go to `runs/detect/`.

### Resume interrupted training

Ultralytics saves **`weights/last.pt`** while training (full checkpoint: weights, optimizer, epoch). To continue after a crash or stop:

1. Find your run folder (check the training log line `save_dir=...`, or look under `runs/detect/`).
2. Point `--model` at **`.../weights/last.pt`** (not `best.pt`).
3. Pass **`--resume`**:

```bash
screenfilter train \
  --resume \
  --data configs/messengers.yaml \
  --model runs/detect/runs/detect/train/weights/last.pt \
  --epochs 100 \
  --imgsz 960
```

Adjust `--model` to your actual `last.pt` path. With `--resume`, Ultralytics restores progress and keeps writing to the same run when possible.

### 4) Collect screenshots containing messenger apps

```bash
screenfilter collect \
  --model runs/detect/train/weights/best.pt \
  --source /path/to/screenshot_folder \
  --out collected \
  --conf 0.25
```

## Notes on offline usage
- This uses the local `ultralytics` package (PyTorch). There is **no online inference service**.
- Training/inference is offline once dependencies are installed and model weights are available locally.
- If you want to avoid any automatic weight downloads, always pass `--model` as a **local path** (e.g. `weights/yolov8n.pt`) and keep your environment without internet.

