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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500&display=swap');

:root {
    --bg-0: #08080a;
    --panel: #121214;
    --panel-2: #17181b;
    --border: #232326;
    --text: #f2f2f4;
    --text-dim: #8b8b93;
    --accent: #4f6bff;
    --accent-dim: rgba(79,107,255,0.12);
    --green: #34d399;
    --yellow: #f2b93d;
    --red: #f0546a;
}

html, body, [class*="stApp"] {
    background: var(--bg-0) !important;
    color: var(--text);
    font-family: 'Inter', -apple-system, sans-serif;
}
.block-container {padding-top: 2.5rem; padding-bottom: 2rem; max-width: 1360px;}
p, span, div, label { font-family: 'Inter', -apple-system, sans-serif; }

/* header */
.eyebrow-badge {
    display: inline-flex; align-items: center; gap: 8px;
    border: 1px solid var(--border); background: var(--panel);
    border-radius: 100px; padding: 6px 14px; font-size: 0.78rem; font-weight: 600;
    color: var(--text-dim); margin-bottom: 22px;
}
.eyebrow-badge .dot { width: 6px; height: 6px; border-radius: 50%; background: var(--accent); }
.app-header h1 {
    font-size: 2.4rem; font-weight: 800; margin: 0 0 10px 0; letter-spacing: -0.03em; line-height: 1.1;
    color: var(--text);
}
.app-header h1 .accent { color: var(--accent); }
.app-header .subtitle {
    color: var(--text-dim); font-size: 1rem; font-weight: 400; margin-bottom: 28px; max-width: 640px;
}

/* buttons */
.stButton > button {
    background: var(--panel) !important; color: var(--text) !important;
    border: 1px solid var(--border) !important; border-radius: 10px !important;
    font-weight: 600 !important; padding: 0.55rem 1.1rem !important;
    transition: all 0.15s ease !important; font-size: 0.88rem !important;
}
.stButton > button:hover {
    border-color: var(--accent) !important; background: var(--accent-dim) !important;
}
.stButton > button p { font-weight: 600 !important; }
div[data-testid="column"]:first-of-type .stButton > button {
    background: var(--accent) !important; border-color: var(--accent) !important; color: white !important;
}
div[data-testid="column"]:first-of-type .stButton > button:hover { filter: brightness(1.1); }

/* status banner -- styled like the reference site's risk-summary cards */
.status-banner {
    padding: 16px 18px; border-radius: 14px; font-weight: 700; font-size: 0.98rem;
    margin-bottom: 16px; letter-spacing: -0.005em;
    border: 1px solid var(--border); background: var(--panel);
    display: flex; align-items: center; gap: 12px;
}
.status-dot { width: 9px; height: 9px; border-radius: 50%; flex-shrink: 0; }
.status-green { color: var(--green); }
.status-green .status-dot { background: var(--green); box-shadow: 0 0 12px 2px rgba(52,211,153,0.5); }
.status-yellow { color: var(--yellow); }
.status-yellow .status-dot { background: var(--yellow); box-shadow: 0 0 12px 2px rgba(242,185,61,0.5); }
.status-red { color: var(--red); border-color: rgba(240,84,106,0.4); animation: pulse-red 1.6s ease-in-out infinite; }
.status-red .status-dot { background: var(--red); box-shadow: 0 0 12px 2px rgba(240,84,106,0.6); }
@keyframes pulse-red {
    0%, 100% { background: var(--panel); }
    50% { background: rgba(240,84,106,0.08); }
}

/* panel card */
.panel-card {
    background: var(--panel); border: 1px solid var(--border); border-radius: 14px;
    padding: 16px 18px; margin-bottom: 16px;
}
.panel-label {
    font-size: 0.7rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase;
    color: var(--text-dim); margin-bottom: 12px;
}

/* map */
.map-wrap {
    position: relative; border-radius: 10px; overflow: hidden; height: 140px;
    background:
        linear-gradient(rgba(79,107,255,0.05) 1px, transparent 1px) 0 0/22px 22px,
        linear-gradient(90deg, rgba(79,107,255,0.05) 1px, transparent 1px) 0 0/22px 22px,
        #0d0d10;
    border: 1px solid var(--border);
}
.map-dot {
    position: absolute; width: 14px; height: 14px; border-radius: 50%;
    top: 50%; left: 50%; transform: translate(-50%, -50%);
}
.map-dot::after {
    content: ''; position: absolute; inset: -12px; border-radius: 50%;
    border: 1px solid currentColor; opacity: 0.3;
}
.map-caption {
    text-align: center; font-size: 0.78rem; color: var(--text-dim); margin-top: 12px; font-weight: 500;
}

/* alerts feed */
.alerts-heading {
    font-size: 0.98rem; font-weight: 700; margin-bottom: 14px; display: flex; align-items: center; gap: 8px;
    color: var(--text);
}
.alert-type { font-weight: 700; font-size: 0.88rem; color: var(--text); }
.alert-meta { font-size: 0.75rem; color: var(--text-dim); font-family: 'JetBrains Mono', monospace; margin-top: 3px; }
.badge-new {
    color: var(--red); font-weight: 700; font-size: 0.68rem; letter-spacing: 0.04em;
    background: rgba(240,84,106,0.12); padding: 2px 9px; border-radius: 20px; display: inline-block; margin-top: 7px;
}
.badge-dispatched {
    color: var(--green); font-weight: 700; font-size: 0.68rem; letter-spacing: 0.04em;
    background: rgba(52,211,153,0.12); padding: 2px 9px; border-radius: 20px; display: inline-block; margin-top: 7px;
}
.empty-feed {
    text-align: center; padding: 34px 10px; color: var(--text-dim); font-size: 0.85rem; line-height: 1.6;
    border: 1px dashed var(--border); border-radius: 12px; background: var(--panel);
}

/* video frame */
div[data-testid="stImage"] img {
    border-radius: 14px; border: 1px solid var(--border);
}

hr { border-color: var(--border) !important; margin: 12px 0 !important; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.markdown(
    """
    <div class="app-header">
        <div class="eyebrow-badge"><span class="dot"></span>Видеоаналитика для городской безопасности</div>
        <h1>Smart Safety — <span class="accent">ситуационный центр</span></h1>
        <div class="subtitle">Детекция падений и скоплений людей в видеопотоке в реальном времени —
            с автоматическим оповещением и передачей наряда.</div>
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
        return "red", "ТРЕВОГА: обнаружено падение"
    if any(a["type"] == "crowd" for a in active):
        return "yellow", "ВНИМАНИЕ: скопление людей"
    return "green", "Штатная ситуация"


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


_render_call_id = 0


def render_alerts():
    # process_video()'s loop can call this several times within a single script
    # execution (once per new alert); a stable widget key would collide on the
    # 2nd+ call since Streamlit checks key uniqueness per run, not per placeholder,
    # so every call gets its own suffix.
    global _render_call_id
    _render_call_id += 1
    call_id = _render_call_id

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
                    if st.button("Отправить наряд", key=f"dispatch_{alert['id']}_{call_id}_{i}",
                                 width="stretch"):
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
