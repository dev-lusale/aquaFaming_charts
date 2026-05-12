"""
app.py
──────
Flask dashboard for the PPE Detection System.

Features:
  • Live MJPEG video stream from webcam / video file
  • Real-time stats panel (safe / unsafe counts, FPS)
  • Unsafe-entry alert log
  • REST API  GET /api/stats  →  JSON snapshot
  • Toggle detection on/off via UI

Run:
    python app.py
    Open browser → http://localhost:5000
"""

import threading
import time
import cv2
from flask import Flask, Response, render_template, jsonify, request
from ppe_detector import PPEDetector, draw_hud

# ── App & detector setup ──────────────────────────────────────────────────────
app      = Flask(__name__)
detector = PPEDetector()

# Shared state (protected by a lock)
_lock        = threading.Lock()
_latest_jpeg = b""
_stats       = {
    "fps": 0, "persons": 0, "safe": 0, "unsafe": 0,
    "total_safe": 0, "total_unsafe": 0,
    "alerts": [],          # list of {"time": ..., "count": ...}
    "running": True,
}
_detection_on = True
SOURCE        = 0          # webcam index; change to a video path if needed


# ── Background capture thread ─────────────────────────────────────────────────
def capture_loop():
    global _latest_jpeg, _stats, _detection_on

    cap = cv2.VideoCapture(SOURCE)
    if not cap.isOpened():
        print(f"Cannot open source: {SOURCE}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.05)
            continue

        if _detection_on:
            frame, summary = detector.predict(frame)
            draw_hud(frame, summary)

            with _lock:
                _stats["fps"]     = summary["fps"]
                _stats["persons"] = summary["persons"]
                _stats["safe"]    = summary["safe"]
                _stats["unsafe"]  = summary["unsafe"]
                _stats["total_safe"]   += summary["safe"]
                _stats["total_unsafe"] += summary["unsafe"]

                if summary["unsafe"] > 0:
                    _stats["alerts"].append({
                        "time":  time.strftime("%H:%M:%S"),
                        "count": summary["unsafe"],
                    })
                    # keep last 50 alerts
                    _stats["alerts"] = _stats["alerts"][-50:]

        # encode to JPEG
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        with _lock:
            _latest_jpeg = buf.tobytes()


# Start background thread
_thread = threading.Thread(target=capture_loop, daemon=True)
_thread.start()


# ── MJPEG stream generator ────────────────────────────────────────────────────
def generate_frames():
    while True:
        with _lock:
            frame = _latest_jpeg
        if frame:
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        time.sleep(0.03)   # ~30 fps cap


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/api/stats")
def api_stats():
    with _lock:
        return jsonify(dict(_stats))


@app.route("/api/toggle", methods=["POST"])
def api_toggle():
    global _detection_on
    _detection_on = not _detection_on
    return jsonify({"detection": _detection_on})


@app.route("/api/alerts/clear", methods=["POST"])
def api_clear_alerts():
    with _lock:
        _stats["alerts"] = []
    return jsonify({"cleared": True})


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("PPE Dashboard → http://localhost:8080")
    app.run(host="0.0.0.0", port=8080, debug=False, threaded=True)
