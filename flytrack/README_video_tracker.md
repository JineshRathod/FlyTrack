# 🚁 FlyTrack: Edge-Ready Video Tracker & Telemetry HUD

> **Real-time flying object detection, tracking, and active carrier control system designed for edge deployment.**

## 🌟 Engineering Highlight: Edge AI & Active Carrier Control
At the core of this project is a custom-engineered, edge-optimized AI model (`fuselage_detector.tflite`). This system represents a complete, closed-loop engineering solution:
- 🧠 **Custom Edge AI**: A custom box regressor model trained specifically for high-speed flying object detection, fully quantized and ready for edge deployment.
- ⚡ **Real-Time Edge Inference**: Designed to run efficiently on CPU/Edge hardware (via `ai-edge-litert` / `tflite-runtime`) with zero dependency on heavy GPUs.
- 🎯 **Active Carrier Control (Visual Servo)**: The AI doesn't just observe—it drives. By calculating target velocity, delta-V, and spatial coordinates, it actively computes the visual servo commands to adjust the **pan/tilt angles** and **speed** of the edge carrier to maintain a stable lock on the flying subject.

---

## 🛠️ Quick Start

No telemetry, no ground truth, no camera pose needed. Files: `track_flying_object.py` (tracker + video loop) and `hud.py` (the HUD; keep it next to the script).
**Dependencies:** OpenCV, NumPy, SciPy, and TensorFlow *or* `ai-edge-litert` / `tflite-runtime`.

### Command Line Usage
```bash
# recorded video  ->  annotated video + CSV
python track_flying_object.py --video ../clip.mp4 --model ../fuselage_detector.tflite --hud --hfov-deg 60 --target-size-m 12 --out tracked.mp4 --csv track.csv

# live camera / RTSP stream with a preview window (q or Esc to stop)
python track_flying_object.py --video 0 --model ../fuselage_detector.tflite --hud --show
python track_flying_object.py --video rtsp://192.168.1.10/stream --hud --show --out recording.mp4

# you pick the vehicle instead of auto-locking (x,y,w,h in original pixels)
python track_flying_object.py --video ../clip.mp4 --model ../m.tflite --init-box 410,180,60,24 --hud

# model-free (small bright/dark blobs); IR white-hot = bright, visible sky = dark
python track_flying_object.py --video ../clip.mp4 --polarity bright --hud
```

### Python API
```python
from track_flying_object import track_video
res = track_video("../clip.mp4", model="../fuselage_detector.tflite", hud=True, hfov_deg=60, target_size_m=12, out="tracked.mp4", csv_path="track.csv")
print(res.stats)      # fps, ms/frame, dropped frames, latency
```

## ⚡ Real-Time Performance
* **Sources:** A file, a camera index (`0`), or a stream URL (`rtsp://`, `http://`, `udp://`). Camera/URL sources are treated as live automatically. `--realtime` makes a *file* behave like a live camera to test real-time behaviour.
* **Newest-frame Policy:** A background thread keeps only the latest frame. If the tracker is slower than the camera, frames are *dropped* instead of queuing, so latency stays bounded. `res.stats` reports dropped frames and capture-to-output latency. A stalled stream stops after 5s.
* **Highly Optimized:** Engineered for speed (e.g., 50.7 → ~12 ms/frame at 640×512, no model) using rectangular morphology kernels, cheaper statistics, half-resolution camera-motion estimate, and candidate pre-filtering. 

**Measured Performance (Sandbox CPU):**
| Input Resolution | Blob Only | Blob + TFLite (3.4 MB model) | + HUD Overlay |
|---|---|---|---|
| **640×480** | 8.7 ms | 12.3 ms | 17.3 / 22.9 ms |
| **1280×720** | 19.2 ms | 23.2 ms | 29.0 / 32.1 ms |
| **1920×1080** | 14.7 ms | 17.9 ms | 33.0 / 36.1 ms |

*Note: Emulated live run of an IR clip (30 fps, HUD on) yielded **0 frames dropped, latency p50/p95 27/47 ms**. On slower machines, dropping frames is the intended behavior to maintain real-time sync.*

## 🛩️ Telemetry HUD (`--hud`)
This is the same HUD as in the simulation (`hud.py`), drawn directly on your frames. Since a single camera cannot measure everything, **every number is either measured, estimated from the image under an explicit assumption, or assumed**.

| HUD Item | Source | Status |
|---|---|---|
| **Ground speed / ALT / climb** | Hover at 50 m (`--alt-m`) | **ASSUMED** |
| **Subject range** | `f · target_size / box_size` | **EST** |
| **Range rate** | Slope of ln(blob size) over ~1.2 s × range | **EST** |
| **Subject velocity** | Radial + tangential velocity | **EST** |
| **DELTA-V (drone − subj)** | Assumed hover means exactly −(subject velocity) | **EST** |
| **Bearing / depression / shot angle** | Pixel offset + assumed camera pose | Measured + **ASSUMED** |
| **PAN / TILT now** | Assumed camera pose | **ASSUMED** |
| **PAN / TILT corr, rate, OFFSET** | Visual servo's commands from measured image error | Measured / Commanded |

* **Layout:** Uses a `dock` layout automatically below 1280px wide (video in the middle, panels in margins).
* **Robustness:** Handles cut/re-acquisition and box-size jumps smoothly. Speeds above 150 m/s (`--max-speed-mps`) are ignored.

## 🧠 Model Architecture & Pipeline
Your model: 224×224 RGB in [0,1] → `[ymin,xmin,ymax,xmax]` (SSD-style outputs auto-detected; `--input-range pm1|255`). 
- **Execution:** Runs on a crop around the predicted position and verifies candidates by agreement. 
- **Tracking:** Keeps identity with camera-motion compensation, joint assignment, ambiguity guard, cannot-link memory, and strict re-acquisition. 
- **Telemetry Data (`track.csv`):** Includes range estimates, subject speed/heading, delta-V, and assumed bearing/depression angles. Tests can be run with `python -m pytest tests`.

## ⚠️ Known Limits
* **Hover Assumption:** If the edge carrier moves, DELTA-V, subject speed, and heading calculations will be skewed by that motion.
* **Range Calculations:** Requires the target's physical size and correct HFOV. A rotating subject (e.g., helicopter) changes apparent length, affecting range calculations. Range rate comes from size change only (noisy, biased low).
* **Box Regressor:** Trained without negatives, meaning rejection of false positives relies entirely on motion/size/appearance gating and agreement tests. For multiple identical vehicles, use `--init-box` to lock onto the correct one.
