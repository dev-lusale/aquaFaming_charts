# Industrial PPE Detection System

Real-time AI safety system that detects **Helmets**, **Reflective Vests**, **Gloves**, and flags **Unsafe Entry** (person without required PPE).

## Stack
| Tool | Role |
|---|---|
| YOLOv8 (Ultralytics) | Object detection model |
| Roboflow | Dataset download & management |
| OpenCV | Camera capture & frame rendering |
| Flask | Web dashboard server |

---

## Project Structure
```
ppe_system/
├── models/               ← trained weights go here (ppe_best.pt)
├── templates/
│   └── index.html        ← Flask dashboard UI
├── ppe_detector.py       ← core inference engine (use standalone or import)
├── app.py                ← Flask dashboard (MJPEG stream + REST API)
├── train.py              ← Roboflow dataset download + YOLOv8 training
└── requirements.txt
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run with fallback model (no training needed)
The system works immediately using YOLOv8's built-in COCO model.
It will detect **persons** and flag them as unsafe until a custom model is trained.

```bash
# Standalone detector (webcam)
python ppe_detector.py

# Standalone detector (video file)
python ppe_detector.py --source path/to/video.mp4

# Flask dashboard
python app.py
# → open http://localhost:5000
```

### 3. Train a custom PPE model (recommended)

**Step 1** — Get a free Roboflow API key at https://roboflow.com

**Step 2** — Set your key:
```bash
# Windows
set ROBOFLOW_API_KEY=your_key_here

# Or edit train.py line: ROBOFLOW_API_KEY = "your_key_here"
```

**Step 3** — Train:
```bash
python train.py
```
This downloads the dataset, trains YOLOv8n for 50 epochs, and saves
`models/ppe_best.pt` automatically.

**Step 4** — Run with custom model:
```bash
python ppe_detector.py --model models/ppe_best.pt
python app.py   # dashboard auto-loads models/ppe_best.pt
```

---

## Dashboard Features
- Live MJPEG video stream
- Real-time stats: FPS, persons detected, safe/unsafe counts
- Session totals
- Unsafe entry alert log with timestamps
- Pause/resume detection toggle
- Red alert banner on unsafe entry

## REST API
| Endpoint | Method | Description |
|---|---|---|
| `/video_feed` | GET | MJPEG stream |
| `/api/stats` | GET | JSON stats snapshot |
| `/api/toggle` | POST | Toggle detection on/off |
| `/api/alerts/clear` | POST | Clear alert log |

---

## PPE Classes (custom model)
| ID | Class | Required |
|---|---|---|
| 0 | Helmet | ✅ Yes |
| 1 | Vest | ✅ Yes |
| 2 | Gloves | Optional |
| 3 | Person | — (triggers safety check) |

A person is marked **SAFE** when both Helmet and Vest are detected overlapping their bounding box.

---

## Recommended Roboflow Datasets
- `construction-site-safety` by roboflow-universe-projects
- `ppe-detection-using-yolov8` by roboflow-universe-projects
- Search https://universe.roboflow.com for "PPE helmet vest"

---

## GPU Training (faster)
Edit `train.py`:
```python
DEVICE = "0"        # first CUDA GPU
YOLO_BASE_MODEL = "yolov8s.pt"   # small model — better accuracy
```
