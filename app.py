import time
import uuid
from pathlib import Path
from collections import deque

import cv2
import numpy as np
import streamlit as st

from detection import (
    load_model,
    detect_people,
    SimpleTracker,
    compute_aspect_ratio,
    compute_torso_angle,
    check_fall,
    check_crowd,
)

VIDEO_PATH = str(Path(__file__).parent / "test_videos" / "demo_fall.mp4")
FRAME_SKIP = 2          # process every Nth frame (speed on weak hardware)
CROWD_THRESHOLD = 5
FALL_COOLDOWN_SEC = 8
CROWD_COOLDOWN_SEC = 8
DETECT_CONF = 0.4

st.set_page_config(page_title="Smart Safety — Ситуационный центр", layout="wide")

CUSTOM_CSS = """
<style>
.block-container {padding-top: 1.5rem; padding-bottom: 1rem;}
h1 {font-size: 1.6rem; margin-bottom: 0.2rem;}
.status-banner {
    padding: 14px 18px; border-radius: 10px; font-weight: 700; font-size: 1.1rem;
    text-align: center; color: white; margin-bottom: 14px; transition: background 0.3s ease;
}
.status-green {background: linear-gradient(90deg,#1f9d55,#27ae60);}
.status-yellow {background: linear-gradient(90deg,#d4a017,#f1c40f); color:#20201a;}
.status-red {background: linear-gradient(90deg,#c0392b,#e74c3c);}
.alert-card {
    border-radius: 10px; padding: 10px 12px; margin-bottom: 10px;
    border-left: 6px solid #888; background: #1e1e1e10;
    display: flex; gap: 10px; align-items: center;
}
.alert-card.high {border-left-color: #e74c3c;}
.alert-card.medium {border-left-color: #f1c40f;}
.alert-meta {font-size: 0.82rem; opacity: 0.75;}
.alert-type {font-weight: 700; font-size: 0.95rem;}
.badge-new {color:#e74c3c; font-weight:700; font-size:0.78rem;}
.badge-dispatched {color:#7f8c8d; font-weight:600; font-size:0.78rem;}
.map-wrap {position: relative; border-radius: 10px; overflow: hidden; border: 1px solid #4443;}
.map-dot {
    position: absolute; width: 18px; height: 18px; border-radius: 50%;
    top: 50%; left: 50%; transform: translate(-50%, -50%);
    box-shadow: 0 0 0 6px rgba(0,0,0,0.05);
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.title("🛡️ Smart Safety — Ситуационный центр")


def init_state():
    if "alerts" not in st.session_state:
        st.session_state.alerts = []
    if "last_fall_ts" not in st.session_state:
        st.session_state.last_fall_ts = {}
    if "last_crowd_ts" not in st.session_state:
        st.session_state.last_crowd_ts = 0.0
    if "model" not in st.session_state:
        with st.spinner("Загружаю YOLO модель (первый запуск может скачать веса)..."):
            model, has_pose = load_model()
        st.session_state.model = model
        st.session_state.has_pose = has_pose
    if "tracker" not in st.session_state:
        st.session_state.tracker = SimpleTracker()
    if "running" not in st.session_state:
        st.session_state.running = False


init_state()


def add_alert(alert_type, frame_bgr, severity):
    thumb = cv2.resize(frame_bgr, (160, 120))
    thumb_rgb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
    alert = {
        "id": str(uuid.uuid4())[:8],
        "type": alert_type,
        "timestamp": time.strftime("%H:%M:%S"),
        "severity": severity,
        "thumbnail": thumb_rgb,
        "status": "new",
    }
    st.session_state.alerts.insert(0, alert)
    st.session_state.alerts = st.session_state.alerts[:30]


def overall_status():
    active = [a for a in st.session_state.alerts if a["status"] == "new"]
    if any(a["type"] == "fall" for a in active):
        return "red", "🔴 ТРЕВОГА: обнаружено падение"
    if any(a["type"] == "crowd" for a in active):
        return "yellow", "🟡 ВНИМАНИЕ: скопление людей"
    return "green", "🟢 Штатная ситуация"


# ---------- layout ----------
left_col, right_col = st.columns([2, 1])

with left_col:
    controls = st.columns([1, 1, 2])
    start_btn = controls[0].button("▶️ Старт", width="stretch")
    stop_btn = controls[1].button("⏸ Стоп", width="stretch")
    video_slot = st.empty()

with right_col:
    status_slot = st.empty()
    map_slot = st.empty()
    st.subheader("Лента алертов")
    alerts_slot = st.empty()


def render_status():
    color, label = overall_status()
    status_slot.markdown(
        f'<div class="status-banner status-{color}">{label}</div>',
        unsafe_allow_html=True,
    )
    dot_color = {"green": "#27ae60", "yellow": "#f1c40f", "red": "#e74c3c"}[color]
    map_slot.markdown(
        f"""
        <div class="map-wrap" style="height:140px; background:
            radial-gradient(circle at 50% 50%, #2c3e50 0%, #1a1f27 100%);">
            <div class="map-dot" style="background:{dot_color};
                box-shadow:0 0 24px 6px {dot_color}88;"></div>
        </div>
        <div style="text-align:center; font-size:0.8rem; opacity:0.7; margin-top:4px;">
            Камера №1 — вход на территорию
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_alerts():
    if not st.session_state.alerts:
        alerts_slot.info("Алертов пока нет.")
        return

    with alerts_slot.container():
        for alert in st.session_state.alerts:
            c1, c2, c3 = st.columns([1, 2.2, 1])
            with c1:
                st.image(alert["thumbnail"], width="stretch")
            with c2:
                type_label = "🚨 Падение" if alert["type"] == "fall" else "👥 Скопление людей"
                st.markdown(
                    f'<div class="alert-type">{type_label}</div>'
                    f'<div class="alert-meta">Время: {alert["timestamp"]} '
                    f'· severity: {alert["severity"]}</div>',
                    unsafe_allow_html=True,
                )
                if alert["status"] == "new":
                    st.markdown('<span class="badge-new">● НОВЫЙ</span>', unsafe_allow_html=True)
                else:
                    st.markdown('<span class="badge-dispatched">✔ Наряд передан</span>',
                                unsafe_allow_html=True)
            with c3:
                if alert["status"] == "new":
                    if st.button("Отправить наряд", key=f"dispatch_{alert['id']}"):
                        alert["status"] = "dispatched"
                        st.rerun()
            st.divider()


render_status()
render_alerts()

if start_btn:
    st.session_state.running = True
if stop_btn:
    st.session_state.running = False


def process_video():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        st.error(f"Не удалось открыть видео: {VIDEO_PATH}. "
                  f"Положи файл в test_videos/demo_fall.mp4")
        st.session_state.running = False
        return

    frame_idx = 0
    model = st.session_state.model
    has_pose = st.session_state.has_pose
    tracker = st.session_state.tracker

    while st.session_state.running:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop the video
            continue

        frame_idx += 1
        if frame_idx % FRAME_SKIP != 0:
            continue

        detections = detect_people(frame, model, has_pose, conf=DETECT_CONF)
        assigned = tracker.update(detections)

        display = frame.copy()
        now = time.time()

        for track_id, det in assigned:
            x1, y1, x2, y2 = [int(v) for v in det["box"]]
            ar = compute_aspect_ratio(det["box"])
            angle = compute_torso_angle(det["keypoints"])
            tracker.record_metrics(track_id, ar, angle)

            history = tracker.get_history(track_id)
            is_fall = check_fall(history)

            box_color = (0, 0, 255) if is_fall else (0, 200, 0)
            cv2.rectangle(display, (x1, y1), (x2, y2), box_color, 2)
            label = f"ID{track_id}" + (" FALL" if is_fall else "")
            cv2.putText(display, label, (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

            if is_fall:
                last_ts = st.session_state.last_fall_ts.get(track_id, 0)
                if now - last_ts > FALL_COOLDOWN_SEC:
                    st.session_state.last_fall_ts[track_id] = now
                    add_alert("fall", frame, "high")
                    render_alerts()
                    render_status()

        is_crowd = check_crowd(detections, threshold=CROWD_THRESHOLD)
        if is_crowd:
            cv2.putText(display, f"CROWD: {len(detections)} people", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2)
            if now - st.session_state.last_crowd_ts > CROWD_COOLDOWN_SEC:
                st.session_state.last_crowd_ts = now
                add_alert("crowd", frame, "medium")
                render_alerts()
                render_status()

        status_text = "FALL DETECTED" if any(
            check_fall(tracker.get_history(tid)) for tid, _ in assigned
        ) else ("CROWD" if is_crowd else "OK")
        status_color = (0, 0, 255) if status_text == "FALL DETECTED" else (
            (0, 165, 255) if status_text == "CROWD" else (0, 200, 0))
        cv2.putText(display, f"Status: {status_text}", (20, display.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        display_rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        video_slot.image(display_rgb, width="stretch")

    cap.release()


if st.session_state.running:
    process_video()
else:
    video_slot.info("Нажми ▶️ Старт, чтобы запустить обработку видео.")
