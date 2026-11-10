# VisionAI Demo

Real-time **Face Mesh**, **Hand Tracking** (21 landmarks per hand, with named finger labels) and **Body Pose** (33 landmarks)
in the browser, powered by MediaPipe, OpenCV and Flask.

## Setup
Requires Python 3.9 - 3.12 and a webcam.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate     macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000 and press **Start Camera**.

## Controls
- Face Mesh / Hands + fingers / Body Pose: toggle each detector live
- Hand tracking labels the thumb, index, middle, ring, and pinky fingertips
- Mirror View: flip the image like a mirror
- Stats panel shows faces, hands, visible pose landmarks and FPS

## Tips
- Different camera? `CAMERA_INDEX=1 python app.py` (Windows: `set CAMERA_INDEX=1`)
- Slow machine? Turn off detectors you don't need.

## Structure
```
app.py            Flask server + MediaPipe pipeline (MJPEG stream, JSON API)
templates/        index.html
static/           style.css, script.js
models/           optional custom model files
```
Project demo video:https://drive.google.com/file/d/1Hit0gg1B0bypSgfA8CjHKp4yaWhJaFyY/view?usp=sharing
