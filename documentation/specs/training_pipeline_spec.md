# YOLO Training Pipeline Specification

## Overview

Config-driven pipeline for capturing game frames, annotating them in
Roboflow, training a YOLOv8 object detection model, and validating the
results.  Designed to be **game-agnostic**: each game gets its own
training config in `configs/training/<game>.yaml`.

## Architecture

```
capture_dataset.py ──> Roboflow (annotate) ──> train_model.py ──> validate_model.py
      │                      │                       │                     │
      ▼                      ▼                       ▼                     ▼
  output/dataset_*/    Upload via API          weights/<game>/        mAP thresholds
  frame_*.png          upload_to_roboflow.py   best.pt               per-class AP
  manifest.json
```

## Configuration

### Training Config (`configs/training/<game>.yaml`)

| Field           | Type       | Description                                      |
|-----------------|------------|--------------------------------------------------|
| `game`          | str        | Links to `configs/games/<game>.yaml`             |
| `classes`       | list[str]  | YOLO class names for this game                   |
| `base_model`    | str        | Ultralytics pretrained checkpoint (e.g. `yolov8n.pt`) |
| `dataset_path`  | str\|null  | Path to Roboflow-exported `data.yaml`; null until export |
| `epochs`        | int        | Training epochs                                  |
| `imgsz`         | int        | Input image size (pixels)                        |
| `batch`         | int        | Batch size                                       |
| `amp`           | bool       | Automatic mixed precision                        |
| `device`        | str        | `"auto"` (xpu > cuda > cpu), or explicit         |
| `output_dir`    | str        | Relative path for trained weights                |
| `min_map50`     | float      | Minimum mAP@0.5 for validation pass              |
| `min_map50_95`  | float      | Minimum mAP@0.5:0.95 for validation pass         |

### Environment Variables (`.env`)

| Variable             | Description                    |
|----------------------|--------------------------------|
| `ROBOFLOW_API_KEY`   | Roboflow API key               |
| `ROBOFLOW_WORKSPACE` | Roboflow workspace slug        |
| `ROBOFLOW_PROJECT`   | Roboflow project slug          |

All scripts use `python-dotenv` to load `.env` automatically.
The `.env` file is gitignored; `.env.example` provides a template.

## Pipeline Scripts

### 1. `scripts/capture_dataset.py`

Captures frames from a running game for YOLO training data.

- Launches the game via `BrowserInstance` (Selenium)
- Plays with a random bot (random mouse movements to move paddle)
- Captures frames via `WindowCapture` (PrintWindow/BitBlt)
- Saves PNGs and a JSON manifest with metadata

**Key arguments:**

| Arg               | Default      | Description                         |
|-------------------|--------------|-------------------------------------|
| `--config`        | `breakout-71`| Game config name                    |
| `--frames`        | 500          | Number of frames to capture         |
| `--interval`      | 0.2          | Seconds between captures            |
| `--action-interval`| 5           | Random action every N frames        |
| `--browser`       | auto-detect  | Browser to use                      |
| `--skip-setup`    | false        | Skip npm install                    |

**Output:** `output/dataset_<timestamp>/` containing:
- `frame_00000.png` ... `frame_NNNNN.png`
- `manifest.json` with capture metadata

### 2. `scripts/upload_to_roboflow.py`

Uploads captured frames to Roboflow for annotation.

- Reads frames from a dataset directory
- Uploads via Roboflow Python API
- Supports resume via `.upload_state.json` tracking file
- Configurable dataset split assignment (train/valid/test)

**Key arguments:**

| Arg             | Default        | Description                      |
|-----------------|----------------|----------------------------------|
| `dataset_dir`   | (required)     | Path to dataset directory        |
| `--api-key`     | from `.env`    | Roboflow API key                 |
| `--workspace`   | from `.env`    | Roboflow workspace slug          |
| `--project`     | `breakout71`   | Roboflow project slug            |
| `--batch`       | 50             | Progress log interval            |
| `--split`       | `train`        | Dataset split (train/valid/test) |

### 3. `scripts/train_model.py`

Trains a YOLOv8 model using a game-specific training config.

- Loads config from `configs/training/<game>.yaml`
- Resolves device: XPU > CUDA > CPU (or explicit override)
- Fine-tunes from a pretrained Ultralytics checkpoint
- Saves best weights to `<output_dir>/best.pt`

**Key arguments:**

| Arg              | Default        | Description                     |
|------------------|----------------|---------------------------------|
| `--config`       | `breakout-71`  | Training config name            |
| `--epochs`       | from config    | Override epoch count            |
| `--device`       | from config    | Override device                 |
| `--dataset-path` | from config    | Override dataset path           |
| `--batch`        | from config    | Override batch size             |

### 4. `scripts/validate_model.py`

Validates a trained model against quality thresholds.

- Loads weights and runs `model.val()` on the validation split
- Checks mAP@0.5 and mAP@0.5:0.95 against config thresholds
- Reports per-class AP@0.5
- Optionally saves annotated sample images for visual inspection

**Key arguments:**

| Arg              | Default        | Description                     |
|------------------|----------------|---------------------------------|
| `--config`       | `breakout-71`  | Training config name            |
| `--weights`      | auto           | Override weights path           |
| `--save-samples` | 0              | Number of annotated samples     |

## User Workflow

```
1. python scripts/capture_dataset.py --frames 500
   → output/dataset_<ts>/frame_*.png + manifest.json

2. python scripts/upload_to_roboflow.py output/dataset_<ts>
   → Frames uploaded to Roboflow project

3. Annotate images in Roboflow UI (5 classes: paddle, ball, brick, powerup, wall)

4. Export dataset from Roboflow in YOLOv8 format
   → Download and extract; update dataset_path in configs/training/<game>.yaml

5. python scripts/train_model.py --config breakout-71
   → weights/breakout71/best.pt

6. python scripts/validate_model.py --config breakout-71 --save-samples 10
   → mAP metrics + annotated sample images
```

## Adding a New Game

1. Create `configs/games/<game>.yaml` (game loader config)
2. Create `configs/training/<game>.yaml` (training config with game-specific classes)
3. Run the same pipeline scripts with `--config <game>`

## Breakout 71 Classes

| Index | Class    | In-Game Object          | Notes                          |
|-------|----------|-------------------------|--------------------------------|
| 0     | paddle   | Player paddle           | Single instance                |
| 1     | ball     | Ball                    | May be multiple (multiball perk) |
| 2     | brick    | Destructible bricks     | Variable count per level       |
| 3     | powerup  | Coins (spawned by bricks)| Physics-based, fly toward paddle |
| 4     | wall     | Boundary walls/ceiling  | Static                         |

## Dependencies

| Package        | Purpose                    | Install                    |
|----------------|----------------------------|----------------------------|
| `ultralytics`  | YOLOv8 training/inference  | Already in environment.yml |
| `roboflow`     | API upload to Roboflow     | Added to environment.yml   |
| `python-dotenv`| .env file loading          | Added to environment.yml   |
| `selenium`     | Browser automation         | Already in environment.yml |
| `torch`        | PyTorch (XPU/CUDA/CPU)     | Already in environment.yml |
