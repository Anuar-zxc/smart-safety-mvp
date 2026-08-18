"""Detection helpers for Smart Safety MVP.

Keeps things intentionally simple:
- detect_people(): runs YOLO(-pose) on a frame, returns boxes (+ keypoints if available)
- SimpleTracker: nearest-centroid tracker across frames (no ByteTrack, no re-id)
- check_fall(): aspect-ratio drop + optional torso-angle heuristic, evaluated over a short history
- check_crowd(): simple threshold on person count
"""

import time
import math
from pathlib import Path
from collections import deque

import numpy as np
from ultralytics import YOLO

WEIGHTS_DIR = Path(__file__).parent / "assets"

# COCO pose keypoint indices used by Ultralytics pose models
KP_LEFT_SHOULDER = 5
KP_RIGHT_SHOULDER = 6
KP_LEFT_HIP = 11
KP_RIGHT_HIP = 12
KP_CONF_THRESHOLD = 0.5


def load_model():
    """Try pose model first (gives us the torso-angle signal), fall back to plain detection.

    Weights are cached under assets/ so re-runs never hit the network. If a weight
    file isn't cached yet, ultralytics will download it once into assets/ (needs
    internet for that one-time download only, not at demo runtime).
    """
    WEIGHTS_DIR.mkdir(exist_ok=True)
    candidates = ["yolo11n-pose.pt", "yolov8n-pose.pt", "yolo11n.pt", "yolov8n.pt"]
    last_err = None
    for name in candidates:
        try:
            model = YOLO(str(WEIGHTS_DIR / name))
            has_pose = "pose" in name
            return model, has_pose
        except Exception as e:  # noqa: BLE001
            last_err = e
            continue
    raise RuntimeError(f"Could not load any YOLO model: {last_err}")


def detect_people(frame, model, has_pose, conf=0.4):
    """Run inference on a single frame, return list of person detections.

    Each detection: {"box": (x1,y1,x2,y2), "conf": float, "keypoints": np.ndarray|None}
    keypoints, when present, is shape (17, 3) -> x, y, conf
    """
    results = model.predict(frame, conf=conf, classes=[0], verbose=False)
    detections = []
    if not results:
        return detections

    r = results[0]
    if r.boxes is None or len(r.boxes) == 0:
        return detections

    boxes_xyxy = r.boxes.xyxy.cpu().numpy()
    confs = r.boxes.conf.cpu().numpy()

    kpts_arr = None
    if has_pose and r.keypoints is not None and r.keypoints.data is not None:
        kpts_arr = r.keypoints.data.cpu().numpy()  # (N, 17, 3)

    for i in range(len(boxes_xyxy)):
        x1, y1, x2, y2 = boxes_xyxy[i]
        kp = kpts_arr[i] if kpts_arr is not None and i < len(kpts_arr) else None
        detections.append({
            "box": (float(x1), float(y1), float(x2), float(y2)),
            "conf": float(confs[i]),
            "keypoints": kp,
        })
    return detections


def _centroid(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


class SimpleTracker:
    """Nearest-centroid tracker. Good enough for a handful of people in frame."""

    def __init__(self, max_distance=300, max_missed=15, history_len=30):
        self.next_id = 1
        self.tracks = {}  # id -> {"centroid": (x,y), "missed": int, "history": deque}
        self.max_distance = max_distance
        self.max_missed = max_missed
        self.history_len = history_len

    def update(self, detections):
        """Assign a track id to each detection, return list of (track_id, detection)."""
        assigned = []
        used_track_ids = set()

        for det in detections:
            c = _centroid(det["box"])
            best_id, best_dist = None, self.max_distance
            for tid, t in self.tracks.items():
                if tid in used_track_ids:
                    continue
                d = math.hypot(c[0] - t["centroid"][0], c[1] - t["centroid"][1])
                if d < best_dist:
                    best_dist, best_id = d, tid

            if best_id is None:
                best_id = self.next_id
                self.next_id += 1
                self.tracks[best_id] = {"centroid": c, "missed": 0,
                                         "history": deque(maxlen=self.history_len),
                                         "seen_standing": False}
            else:
                self.tracks[best_id]["centroid"] = c
                self.tracks[best_id]["missed"] = 0

            used_track_ids.add(best_id)
            assigned.append((best_id, det))

        # age out tracks that weren't matched this frame
        for tid in list(self.tracks.keys()):
            if tid not in used_track_ids:
                self.tracks[tid]["missed"] += 1
                if self.tracks[tid]["missed"] > self.max_missed:
                    del self.tracks[tid]

        return assigned

    def record_metrics(self, track_id, aspect_ratio, torso_angle, standing_ar=1.5):
        if track_id not in self.tracks:
            return
        track = self.tracks[track_id]
        track["history"].append({
            "t": time.time(),
            "ar": aspect_ratio,
            "angle": torso_angle,
        })
        # remembered permanently for this track's lifetime -- a short history buffer would
        # otherwise "forget" the standing evidence once someone has been down for a while
        if aspect_ratio >= standing_ar:
            track["seen_standing"] = True

    def get_history(self, track_id):
        t = self.tracks.get(track_id)
        return t["history"] if t else deque()

    def get_seen_standing(self, track_id):
        t = self.tracks.get(track_id)
        return t["seen_standing"] if t else False


def compute_aspect_ratio(box):
    x1, y1, x2, y2 = box
    w = max(x2 - x1, 1e-6)
    h = max(y2 - y1, 1e-6)
    return h / w


def compute_torso_angle(keypoints):
    """Angle (degrees) of the shoulders->hips line from vertical. None if unreliable."""
    if keypoints is None:
        return None

    ls, rs = keypoints[KP_LEFT_SHOULDER], keypoints[KP_RIGHT_SHOULDER]
    lh, rh = keypoints[KP_LEFT_HIP], keypoints[KP_RIGHT_HIP]

    pts = [ls, rs, lh, rh]
    if any(p[2] < KP_CONF_THRESHOLD for p in pts):
        return None

    shoulder_mid = ((ls[0] + rs[0]) / 2.0, (ls[1] + rs[1]) / 2.0)
    hip_mid = ((lh[0] + rh[0]) / 2.0, (lh[1] + rh[1]) / 2.0)

    dx = hip_mid[0] - shoulder_mid[0]
    dy = hip_mid[1] - shoulder_mid[1]
    if dx == 0 and dy == 0:
        return None

    angle_from_vertical = math.degrees(math.atan2(abs(dx), abs(dy)))
    return angle_from_vertical


def check_fall(history, seen_standing, window_sec=0.5, fallen_ar=1.0, angle_threshold=60):
    """A track counts as fallen once it has been seen standing at some point in its
    lifetime (seen_standing, tracked persistently by SimpleTracker -- a short rolling
    history window would otherwise "forget" the standing evidence once someone has been
    down for a while) and its aspect ratio / torso angle has stayed in "fallen" territory
    for at least window_sec, so brief crouches/bends don't trigger it.
    """
    if not seen_standing or len(history) < 3:
        return False

    now = history[-1]["t"]
    sustained_window = [h for h in history if now - h["t"] <= window_sec]
    if not sustained_window:
        return False

    fallen_by_ar = all(h["ar"] < fallen_ar for h in sustained_window)
    fallen_by_angle = all(
        h["angle"] is not None and h["angle"] > angle_threshold for h in sustained_window
    )

    return fallen_by_ar or fallen_by_angle


def check_crowd(detections, threshold=5):
    return len(detections) >= threshold
