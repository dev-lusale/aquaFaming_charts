"""
ppe_detector.py
───────────────
Real-time PPE detection using a two-model pipeline:

  Model 1 — YOLOv8n COCO  : detects persons in the frame
  Model 2 — YOLOv8n Helmet: classifies each person region as
            "With Helmet" or "Without Helmet"

This approach works reliably on webcam footage because:
  • COCO person detection is robust at any distance / angle
  • The helmet model was trained on close-up head/torso images
    matching typical webcam conditions

Usage:
    python ppe_detector.py                     # webcam
    python ppe_detector.py --source video.mp4  # video file
"""

import argparse
import time
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# ── Paths ─────────────────────────────────────────────────────────────────────
_DIR             = Path(__file__).parent
HELMET_MODEL_PATH = _DIR / "models" / "helmet_best.pt"
PERSON_MODEL_PATH = "yolov8n.pt"          # auto-downloaded by ultralytics

# ── Thresholds ────────────────────────────────────────────────────────────────
PERSON_CONF   = 0.45    # confidence to accept a person detection
HELMET_CONF   = 0.40    # confidence to accept helmet/no-helmet classification
IOU_THRESH    = 0.45
FRAME_SIZE    = (1280, 720)

# ── Colours (BGR) ─────────────────────────────────────────────────────────────
COL_SAFE      = (0,   210,  80)
COL_UNSAFE    = (0,   50,  220)
COL_HELMET    = (0,   210,  80)
COL_NO_HELMET = (0,   50,  220)
COL_PERSON    = (200, 200, 200)

# ── Head-region crop helper ───────────────────────────────────────────────────
def head_crop(frame, x1, y1, x2, y2):
    """
    Return the upper-third of a person bounding box (head/torso region)
    with a small padding, clipped to frame bounds.
    """
    h_box = y2 - y1
    head_y2 = y1 + int(h_box * 0.45)          # top 45 % of person box
    pad = int((x2 - x1) * 0.08)
    hx1 = max(0, x1 - pad)
    hy1 = max(0, y1 - pad)
    hx2 = min(frame.shape[1], x2 + pad)
    hy2 = min(frame.shape[0], head_y2 + pad)
    return frame[hy1:hy2, hx1:hx2], (hx1, hy1, hx2, hy2)


# ── Drawing helpers ───────────────────────────────────────────────────────────
def draw_person_box(frame, box, status, detail, color):
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # status badge
    badge = f"  {status}  "
    (bw, bh), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    cv2.rectangle(frame, (x1, y1 - bh - 12), (x1 + bw + 4, y1), color, -1)
    cv2.putText(frame, badge, (x1 + 2, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    # detail below box
    cv2.putText(frame, detail, (x1 + 4, y2 + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, color, 1, cv2.LINE_AA)


def draw_hud(frame, summary):
    h, w = frame.shape[:2]

    # top bar
    cv2.rectangle(frame, (0, 0), (w, 78), (12, 12, 18), -1)
    cv2.putText(frame, "PPE DETECTION SYSTEM", (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (180, 140, 50), 2)
    cv2.putText(frame, "Helmet Detection Active", (12, 54),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 200, 100), 1)

    # right stats
    stats = [
        f"FPS     : {summary['fps']}",
        f"Persons : {summary['persons']}",
        f"Safe    : {summary['safe']}",
        f"Unsafe  : {summary['unsafe']}",
    ]
    for i, s in enumerate(stats):
        col = COL_SAFE   if "Safe"   in s and summary["safe"]   > 0 else \
              COL_UNSAFE if "Unsafe" in s and summary["unsafe"] > 0 else \
              (200, 200, 200)
        cv2.putText(frame, s, (w - 240, 26 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, col, 1)

    # unsafe alert banner
    if summary["unsafe"] > 0:
        cv2.rectangle(frame, (0, h - 48), (w, h), (0, 0, 160), -1)
        msg = f"  UNSAFE ENTRY — {summary['unsafe']} person(s) without helmet"
        cv2.putText(frame, msg, (10, h - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2)


# ── Main detector class ───────────────────────────────────────────────────────
class PPEDetector:
    def __init__(self, conf_person=PERSON_CONF, conf_helmet=HELMET_CONF):
        # Person detector (COCO)
        print("Loading person detector (YOLOv8n COCO)...")
        self.person_model = YOLO(PERSON_MODEL_PATH)

        # Helmet classifier
        if not HELMET_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Helmet model not found at {HELMET_MODEL_PATH}.\n"
                "Run: python -c \"from huggingface_hub import hf_hub_download; "
                "import shutil; shutil.copy(hf_hub_download("
                "'iam-tsr/yolov8n-helmet-detection','best.pt'), 'models/helmet_best.pt')\""
            )
        print(f"Loading helmet model: {HELMET_MODEL_PATH}")
        self.helmet_model = YOLO(str(HELMET_MODEL_PATH))
        print("Helmet classes:", self.helmet_model.names)

        self.conf_person = conf_person
        self.conf_helmet = conf_helmet

        # stats
        self.frame_count  = 0
        self.safe_count   = 0
        self.unsafe_count = 0
        self.fps          = 0.0
        self._t0          = time.time()

    # ── Per-frame inference ────────────────────────────────────────────────
    def predict(self, frame):
        h, w = frame.shape[:2]

        # ── Step 1: detect persons ─────────────────────────────────────────
        person_results = self.person_model.predict(
            source=frame,
            conf=self.conf_person,
            iou=IOU_THRESH,
            classes=[0],        # class 0 = person in COCO
            verbose=False,
        )[0]

        persons = []
        for box in person_results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            persons.append((x1, y1, x2, y2))

        n_safe = n_unsafe = 0

        # ── Step 2: classify helmet for each person ────────────────────────
        for (x1, y1, x2, y2) in persons:
            crop, crop_box = head_crop(frame, x1, y1, x2, y2)

            has_helmet = False
            helmet_conf = 0.0

            if crop.size > 0:
                h_results = self.helmet_model.predict(
                    source=crop,
                    conf=self.conf_helmet,
                    verbose=False,
                )[0]

                for hbox in h_results.boxes:
                    cls_name = self.helmet_model.names[int(hbox.cls[0])]
                    conf_val = float(hbox.conf[0])
                    if cls_name == "With Helmet" and conf_val > helmet_conf:
                        has_helmet  = True
                        helmet_conf = conf_val

            if has_helmet:
                n_safe += 1
                status = "SAFE"
                detail = f"Helmet {helmet_conf:.0%}"
                color  = COL_SAFE
            else:
                n_unsafe += 1
                status = "UNSAFE"
                detail = "No Helmet Detected"
                color  = COL_UNSAFE

            draw_person_box(frame, (x1, y1, x2, y2), status, detail, color)

        # ── FPS ────────────────────────────────────────────────────────────
        self.frame_count += 1
        elapsed = time.time() - self._t0
        if elapsed >= 1.0:
            self.fps         = round(self.frame_count / elapsed, 1)
            self.frame_count = 0
            self._t0         = time.time()

        self.safe_count   += n_safe
        self.unsafe_count += n_unsafe

        summary = {
            "fps":          self.fps,
            "persons":      len(persons),
            "safe":         n_safe,
            "unsafe":       n_unsafe,
            "using_custom": True,
        }
        return frame, summary

    # ── Live loop ──────────────────────────────────────────────────────────
    def run(self, source=0):
        cap = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
        if not cap.isOpened():
            print(f"Cannot open source: {source}")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_SIZE[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_SIZE[1])
        print("PPE Detector running  |  Press 'q' to quit")

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame, summary = self.predict(frame)
                draw_hud(frame, summary)
                cv2.imshow("PPE Detection System", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print(f"\nSession — Safe: {self.safe_count}  Unsafe: {self.unsafe_count}")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",       default="0")
    parser.add_argument("--conf-person",  type=float, default=PERSON_CONF)
    parser.add_argument("--conf-helmet",  type=float, default=HELMET_CONF)
    args = parser.parse_args()

    det = PPEDetector(conf_person=args.conf_person, conf_helmet=args.conf_helmet)
    det.run(source=args.source)
