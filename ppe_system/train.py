"""
train.py
────────
Download a PPE dataset from Roboflow and train a YOLOv8 model.

Steps:
  1. Set your Roboflow API key (or set env var ROBOFLOW_API_KEY)
  2. Run:  python train.py
  3. Trained weights land in:  models/ppe_best.pt

Roboflow PPE datasets to try (free):
  • "ppe-detection-using-yolov8" by roboflow-universe-projects
  • "construction-site-safety"   by roboflow-universe-projects
  • Search https://universe.roboflow.com for "PPE"
"""

import os
import shutil
from pathlib import Path
from roboflow import Roboflow
from ultralytics import YOLO

# ── Configuration — edit these ────────────────────────────────────────────────
ROBOFLOW_API_KEY  = os.getenv("ROBOFLOW_API_KEY", "YOUR_API_KEY_HERE")
ROBOFLOW_WORKSPACE = "roboflow-universe-projects"
ROBOFLOW_PROJECT   = "construction-site-safety"
ROBOFLOW_VERSION   = 1          # dataset version number

YOLO_BASE_MODEL    = "yolov8n.pt"   # nano=fast, yolov8s/m/l for more accuracy
EPOCHS             = 50
IMAGE_SIZE         = 640
BATCH_SIZE         = 16
DEVICE             = "cpu"          # change to "0" if you have a CUDA GPU

OUTPUT_DIR         = Path(__file__).parent / "models"
# ─────────────────────────────────────────────────────────────────────────────


def download_dataset():
    if ROBOFLOW_API_KEY == "YOUR_API_KEY_HERE":
        raise ValueError(
            "Set your Roboflow API key in train.py or via the "
            "ROBOFLOW_API_KEY environment variable.\n"
            "Get a free key at https://roboflow.com"
        )

    print(f"Connecting to Roboflow workspace: {ROBOFLOW_WORKSPACE}")
    rf      = Roboflow(api_key=ROBOFLOW_API_KEY)
    project = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT)
    dataset = project.version(ROBOFLOW_VERSION).download("yolov8")
    print(f"Dataset downloaded to: {dataset.location}")
    return dataset.location


def train(data_yaml: str):
    print(f"\nStarting YOLOv8 training — base: {YOLO_BASE_MODEL}")
    print(f"Epochs: {EPOCHS}  |  Image size: {IMAGE_SIZE}  |  Batch: {BATCH_SIZE}  |  Device: {DEVICE}\n")

    model   = YOLO(YOLO_BASE_MODEL)
    results = model.train(
        data    = data_yaml,
        epochs  = EPOCHS,
        imgsz   = IMAGE_SIZE,
        batch   = BATCH_SIZE,
        device  = DEVICE,
        project = str(OUTPUT_DIR / "runs"),
        name    = "ppe_train",
        exist_ok= True,
        patience= 15,           # early stopping
        save    = True,
        plots   = True,
    )
    return results


def export_best_weights(run_dir: Path):
    best = run_dir / "weights" / "best.pt"
    if best.exists():
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        dest = OUTPUT_DIR / "ppe_best.pt"
        shutil.copy(best, dest)
        print(f"\nBest weights saved to: {dest}")
        return dest
    else:
        print("Warning: best.pt not found — check training output.")
        return None


def validate(weights_path: Path, data_yaml: str):
    print(f"\nValidating model: {weights_path}")
    model   = YOLO(str(weights_path))
    metrics = model.val(data=data_yaml, imgsz=IMAGE_SIZE, device=DEVICE)
    print(f"mAP50   : {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    return metrics


if __name__ == "__main__":
    # 1. Download dataset
    dataset_location = download_dataset()
    data_yaml        = str(Path(dataset_location) / "data.yaml")

    # 2. Train
    results  = train(data_yaml)
    run_dir  = Path(results.save_dir)

    # 3. Copy best weights
    best_pt  = export_best_weights(run_dir)

    # 4. Validate
    if best_pt:
        validate(best_pt, data_yaml)

    print("\nDone. Run the detector with:")
    print("  python ppe_detector.py --model models/ppe_best.pt")
