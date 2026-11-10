"""
VisionAI Demo - real-time Face Mesh, Hand Tracking and Body Pose.
Flask backend + MediaPipe + OpenCV. Run:  python app.py
Then open http://127.0.0.1:5000
"""
import os
import threading
import time

import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, Response, jsonify, render_template, request

CAMERA_INDEX = int(os.environ.get("CAMERA_INDEX", 0))
FRAME_W, FRAME_H = 1280, 720

app = Flask(__name__)

mp_face = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

# BGR colours
GREEN = (90, 230, 90)
BLUE = (246, 130, 59)
PURPLE = (241, 102, 99)
WHITE = (255, 255, 255)

FACE_SPEC = mp_draw.DrawingSpec(color=(70, 220, 70), thickness=1, circle_radius=0)
HAND_LM = mp_draw.DrawingSpec(color=(250, 120, 60), thickness=-1, circle_radius=4)
HAND_CN = mp_draw.DrawingSpec(color=WHITE, thickness=2)
POSE_LM = mp_draw.DrawingSpec(color=(60, 200, 255), thickness=-1, circle_radius=5)
POSE_CN = mp_draw.DrawingSpec(color=(240, 200, 120), thickness=3)
FINGER_TIPS = ((4, "Thumb"), (8, "Index"), (12, "Middle"), (16, "Ring"), (20, "Pinky"))


def bbox(landmarks, w, h, pad=0.02):
    xs = [l.x for l in landmarks]
    ys = [l.y for l in landmarks]
    x1 = int(max(0, (min(xs) - pad) * w))
    y1 = int(max(0, (min(ys) - pad) * h))
    x2 = int(min(w - 1, (max(xs) + pad) * w))
    y2 = int(min(h - 1, (max(ys) + pad) * h))
    return x1, y1, x2, y2


def tag(frame, text, x, y, color):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x, y - th - 12), (x + tw + 14, y), color, -1)
    cv2.putText(frame, text, (x + 7, y - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2, cv2.LINE_AA)


def draw_finger_labels(frame, landmarks):
    h, w = frame.shape[:2]
    offsets = {4: -8, 8: -24, 12: -40, 16: -24, 20: -8}
    for index, name in FINGER_TIPS:
        point = landmarks[index]
        text = f"{name} finger"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        x = max(4, min(int(point.x * w) + 8, w - tw - 8))
        y = max(th + 6, min(int(point.y * h) + offsets[index], h - 4))
        cv2.rectangle(frame, (x - 3, y - th - 4), (x + tw + 3, y + 3), (35, 35, 35), -1)
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    WHITE, 1, cv2.LINE_AA)


def placeholder_jpeg(text="Camera is off - press Start Camera"):
    img = np.zeros((FRAME_H, FRAME_W, 3), np.uint8)
    img[:] = (30, 20, 14)
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
    cv2.putText(img, text, ((FRAME_W - tw) // 2, FRAME_H // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (190, 170, 150), 2, cv2.LINE_AA)
    return cv2.imencode(".jpg", img)[1].tobytes()


class VisionEngine:
    def __init__(self):
        self.lock = threading.Lock()
        self.running = False
        self.thread = None
        self.error = None
        self.settings = {"face": True, "hands": True, "pose": True, "mirror": True}
        self.stats = {"faces": 0, "hands": 0, "pose": 0, "fps": 0}
        self.idle_frame = placeholder_jpeg()
        self.frame = self.idle_frame

    # ---------- lifecycle ----------
    def start(self):
        if self.running:
            return True
        cap = cv2.VideoCapture(CAMERA_INDEX)
        if not cap.isOpened():
            self.error = f"Could not open camera index {CAMERA_INDEX}."
            return False
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
        self.error = None
        self.running = True
        self.thread = threading.Thread(target=self._loop, args=(cap,), daemon=True)
        self.thread.start()
        return True

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=3)
        self.stats = {"faces": 0, "hands": 0, "pose": 0, "fps": 0}
        self.frame = self.idle_frame

    # ---------- main loop ----------
    def _loop(self, cap):
        face = mp_face.FaceMesh(max_num_faces=2, min_detection_confidence=0.5,
                                min_tracking_confidence=0.5)
        hands = mp_hands.Hands(max_num_hands=2, model_complexity=0,
                               min_detection_confidence=0.5, min_tracking_confidence=0.5)
        pose = mp_pose.Pose(model_complexity=0, min_detection_confidence=0.5,
                            min_tracking_confidence=0.5)
        last, fps = time.time(), 0.0
        try:
            while self.running:
                ok, frame = cap.read()
                if not ok:
                    self.error = "Camera stopped returning frames."
                    break
                s = dict(self.settings)
                mirror = s["mirror"]
                if mirror:
                    frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                n_faces = n_hands = n_pose = 0

                if s["pose"]:
                    r = pose.process(rgb)
                    if r.pose_landmarks:
                        mp_draw.draw_landmarks(frame, r.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                               POSE_LM, POSE_CN)
                        n_pose = sum(1 for l in r.pose_landmarks.landmark if l.visibility > 0.5)
                        x1, y1, x2, y2 = bbox(r.pose_landmarks.landmark, w, h)
                        tag(frame, "Body Pose", max(x1, 10), min(h - 10, y2 - 20), PURPLE)

                if s["face"]:
                    r = face.process(rgb)
                    if r.multi_face_landmarks:
                        n_faces = len(r.multi_face_landmarks)
                        for fl in r.multi_face_landmarks:
                            mp_draw.draw_landmarks(
                                frame, fl, mp_face.FACEMESH_TESSELATION, None, FACE_SPEC)
                            x1, y1, x2, y2 = bbox(fl.landmark, w, h)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), GREEN, 2)
                            tag(frame, "Face", x1, y1, (60, 200, 60))

                if s["hands"]:
                    r = hands.process(rgb)
                    if r.multi_hand_landmarks:
                        n_hands = len(r.multi_hand_landmarks)
                        for hl, hd in zip(r.multi_hand_landmarks, r.multi_handedness):
                            label = hd.classification[0].label  # assumes mirrored (selfie) input
                            if not mirror:
                                label = "Left" if label == "Right" else "Right"
                            mp_draw.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS,
                                                   HAND_LM, HAND_CN)
                            draw_finger_labels(frame, hl.landmark)
                            x1, y1, x2, y2 = bbox(hl.landmark, w, h, 0.03)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), BLUE, 2)
                            tag(frame, f"{label} Hand", x1, y1, BLUE)

                now = time.time()
                fps = 0.9 * fps + 0.1 * (1.0 / max(now - last, 1e-6))
                last = now
                self.stats = {"faces": n_faces, "hands": n_hands, "pose": n_pose,
                              "fps": int(round(fps))}
                self.frame = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()
        finally:
            cap.release()
            face.close()
            hands.close()
            pose.close()
            self.running = False
            self.frame = self.idle_frame


engine = VisionEngine()


@app.route("/")
def index():
    return render_template("index.html")


def mjpeg():
    while True:
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + engine.frame + b"\r\n")
        time.sleep(0.03)


@app.route("/video_feed")
def video_feed():
    return Response(mjpeg(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.post("/api/start")
def api_start():
    if engine.start():
        return jsonify(ok=True)
    return jsonify(ok=False, error=engine.error), 500


@app.post("/api/stop")
def api_stop():
    engine.stop()
    return jsonify(ok=True)


@app.post("/api/settings")
def api_settings():
    data = request.get_json(silent=True) or {}
    for key in engine.settings:
        if key in data:
            engine.settings[key] = bool(data[key])
    return jsonify(engine.settings)


@app.get("/api/stats")
def api_stats():
    return jsonify(running=engine.running, error=engine.error, **engine.stats)


if __name__ == "__main__":
    print("VisionAI Demo running at http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, threaded=True, debug=False)
