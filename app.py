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
FRAME_SKIP = 1          # process every Nth frame (speed on weak hardware)
CROWD_THRESHOLD = 5
FALL_COOLDOWN_SEC = 8
CROWD_COOLDOWN_SEC = 8
DETECT_CONF = 0.4

st.set_page_config(page_title="Smart Safety — Ситуационный центр", layout="wide", page_icon="🛡️")

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@500;700;800&family=JetBrains+Mono:wght@500&display=swap');

:root {
    --bg-0: #0a0d14;
    --bg-1: #0f1420;
    --panel: #131926;
    --panel-2: #171f30;
    --border: #232c40;
    --text: #e8ecf5;
    --text-dim: #8792a8;
    --accent: #3b82f6;
    --green: #22c55e;
    --yellow: #eab308;
    --red: #ef4444;
}

html, body, [class*="stApp"] {
    background: radial-gradient(ellipse 120% 80% at 50% -10%, #14213a 0%, var(--bg-0) 55%) !important;
    color: var(--text);
    font-family: 'Manrope', -apple-system, sans-serif;
}
.block-container {padding-top: 2rem; padding-bottom: 2rem; max-width: 1400px;}

/* header */
.app-header {
    display: flex; align-items: center; gap: 14px; margin-bottom: 4px;
}
.app-header .icon-badge {
    width: 46px; height: 46px; border-radius: 12px;
    background: linear-gradient(135deg, var(--accent), #8b5cf6);
    display: flex; align-items: center; justify-content: center;
    font-size: 22px; box-shadow: 0 6px 20px rgba(59,130,246,0.35);
}
.app-header h1 {
    font-size: 1.7rem; font-weight: 800; margin: 0; letter-spacing: -0.01em;
}
.app-header .subtitle {
    color: var(--text-dim); font-size: 0.85rem; font-weight: 500; margin-top: 2px;
}
.section-divider { height: 1px; background: var(--border); margin: 22px 0 18px 0; border: none; }

/* buttons */
.stButton > button {
    background: var(--panel-2) !important; color: var(--text) !important;
    border: 1px solid var(--border) !important; border-radius: 10px !important;
    font-weight: 700 !important; padding: 0.55rem 1rem !important;
    transition: all 0.15s ease !important;
}
.stButton > button:hover {
    border-color: var(--accent) !important; background: #1b2436 !important;
    transform: translateY(-1px);
}
.stButton > button p { font-weight: 700 !important; }

/* status banner */
.status-banner {
    padding: 16px 20px; border-radius: 14px; font-weight: 800; font-size: 1.05rem;
    text-align: center; color: white; margin-bottom: 16px; letter-spacing: -0.01em;
    border: 1px solid rgba(255,255,255,0.08);
    display: flex; align-items: center; justify-content: center; gap: 10px;
}
.status-dot { width: 10px; height: 10px; border-radius: 50%; background: white; flex-shrink: 0; }
.status-green { background: linear-gradient(135deg, #15803d, #22c55e); box-shadow: 0 8px 24px rgba(34,197,94,0.25); }
.status-yellow { background: linear-gradient(135deg, #a16207, #eab308); color:#1a1400; box-shadow: 0 8px 24px rgba(234,179,8,0.25); }
.status-yellow .status-dot { background: #1a1400; }
.status-red { background: linear-gradient(135deg, #b91c1c, #ef4444); box-shadow: 0 8px 24px rgba(239,68,68,0.35); animation: pulse-red 1.6s ease-in-out infinite; }
@keyframes pulse-red {
    0%, 100% { box-shadow: 0 8px 24px rgba(239,68,68,0.35); }
    50% { box-shadow: 0 8px 34px rgba(239,68,68,0.65); }
}

/* panel card */
.panel-card {
    background: var(--panel); border: 1px solid var(--border); border-radius: 14px;
    padding: 14px 16px; margin-bottom: 16px;
}
.panel-label {
    font-size: 0.72rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase;
    color: var(--text-dim); margin-bottom: 10px;
}

/* map */
.map-wrap {
    position: relative; border-radius: 12px; overflow: hidden; height: 150px;
    background:
        linear-gradient(rgba(59,130,246,0.06) 1px, transparent 1px) 0 0/24px 24px,
        linear-gradient(90deg, rgba(59,130,246,0.06) 1px, transparent 1px) 0 0/24px 24px,
        radial-gradient(circle at 50% 50%, #17213a 0%, #0d1220 100%);
    border: 1px solid var(--border);
}
.map-dot {
    position: absolute; width: 16px; height: 16px; border-radius: 50%;
    top: 50%; left: 50%; transform: translate(-50%, -50%);
}
.map-dot::after {
    content: ''; position: absolute; inset: -14px; border-radius: 50%;
    border: 1px solid currentColor; opacity: 0.35;
}
.map-caption {
    text-align: center; font-size: 0.78rem; color: var(--text-dim); margin-top: 10px; font-weight: 600;
}

/* alerts feed */
.alerts-heading {
    font-size: 0.95rem; font-weight: 800; margin-bottom: 12px; display: flex; align-items: center; gap: 8px;
}
.alert-type { font-weight: 800; font-size: 0.9rem; }
.alert-meta { font-size: 0.76rem; color: var(--text-dim); font-family: 'JetBrains Mono', monospace; margin-top: 2px; }
.badge-new {
    color: var(--red); font-weight: 700; font-size: 0.72rem; letter-spacing: 0.04em;
    background: rgba(239,68,68,0.12); padding: 2px 8px; border-radius: 20px; display: inline-block; margin-top: 6px;
}
.badge-dispatched {
    color: #4ade80; font-weight: 700; font-size: 0.72rem; letter-spacing: 0.04em;
    background: rgba(74,222,128,0.12); padding: 2px 8px; border-radius: 20px; display: inline-block; margin-top: 6px;
}
.empty-feed {
    text-align: center; padding: 30px 10px; color: var(--text-dim); font-size: 0.85rem;
    border: 1px dashed var(--border); border-radius: 12px;
}

/* video frame */
div[data-testid="stImage"] img {
    border-radius: 14px; border: 1px solid var(--border);
}

hr { border-color: var(--border) !important; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.markdown(
    """
    <div class="app-header">
        <div class="icon-badge">🛡️</div>
        <div>
            <h1>Smart Safety — Ситуационный центр</h1>
            <div class="subtitle">Видеоаналитика в реальном времени · детекция падений и скоплений людей</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


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
left_col, right_col = st.columns([2, 1], gap="large")

with left_col:
    controls = st.columns([1, 1, 3])
    start_btn = controls[0].button("▶️ Старт", width="stretch")
    stop_btn = controls[1].button("⏸ Стоп", width="stretch")
    video_slot = st.empty()

with right_col:
    status_slot = st.empty()
    map_slot = st.empty()
    alerts_heading_slot = st.empty()
    alerts_slot = st.empty()
    alerts_heading_slot.markdown(
        '<div class="alerts-heading">📋 Лента алертов</div>', unsafe_allow_html=True
    )


def render_status():
    color, label = overall_status()
    status_slot.markdown(
        f'<div class="status-banner status-{color}"><span class="status-dot"></span>{label}</div>',
        unsafe_allow_html=True,
    )
    dot_color = {"green": "#22c55e", "yellow": "#eab308", "red": "#ef4444"}[color]
    map_slot.markdown(
        f"""
        <div class="panel-card">
            <div class="panel-label">Карта объекта</div>
            <div class="map-wrap">
                <div class="map-dot" style="background:{dot_color}; color:{dot_color};
                    box-shadow:0 0 20px 4px {dot_color}99;"></div>
            </div>
            <div class="map-caption">📍 Камера №1 — вход на территорию</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_alerts():
    if not st.session_state.alerts:
        alerts_slot.markdown(
            '<div class="empty-feed">Алертов пока нет —<br>система следит за периметром</div>',
            unsafe_allow_html=True,
        )
        return

    with alerts_slot.container():
        for i, alert in enumerate(st.session_state.alerts):
            accent = "var(--red)" if alert["severity"] == "high" else "var(--yellow)"
            c1, c2, c3 = st.columns([1, 2.2, 1], vertical_alignment="center")
            with c1:
                st.image(alert["thumbnail"], width="stretch")
            with c2:
                type_label = "🚨 Падение" if alert["type"] == "fall" else "👥 Скопление людей"
                badge = ('<span class="badge-new">● НОВЫЙ</span>' if alert["status"] == "new"
                         else '<span class="badge-dispatched">✔ НАРЯД ПЕРЕДАН</span>')
                st.markdown(
                    f'<div class="alert-type" style="color:{accent}">{type_label}</div>'
                    f'<div class="alert-meta">{alert["timestamp"]} · severity: {alert["severity"]}</div>'
                    f'{badge}',
                    unsafe_allow_html=True,
                )
            with c3:
                if alert["status"] == "new":
                    if st.button("Отправить наряд", key=f"dispatch_{alert['id']}", width="stretch"):
                        alert["status"] = "dispatched"
                        st.rerun()
            if i < len(st.session_state.alerts) - 1:
                st.markdown('<hr style="border-color: var(--border); margin: 10px 0;">',
                            unsafe_allow_html=True)


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
            is_fall = check_fall(history, tracker.get_seen_standing(track_id))

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
            check_fall(tracker.get_history(tid), tracker.get_seen_standing(tid))
            for tid, _ in assigned
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
    video_slot.markdown(
        """
        <div style="border:1px dashed var(--border); border-radius:14px; padding:80px 20px;
            text-align:center; color:var(--text-dim); background:var(--panel);">
            <div style="font-size:2rem; margin-bottom:10px;">▶️</div>
            Нажми «Старт», чтобы запустить обработку видео
        </div>
        """,
        unsafe_allow_html=True,
    )
