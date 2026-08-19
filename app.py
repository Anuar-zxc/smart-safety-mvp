import time
import uuid
from pathlib import Path

import cv2
import streamlit as st

from detection import (
    load_model,
    detect_people,
    SimpleTracker,
    compute_aspect_ratio,
    compute_torso_angle,
    check_fall,
    check_crowd,
    check_fight,
)

VIDEO_DIR = Path(__file__).parent / "test_videos"
CAMERAS = [
    {"id": "cam1", "name": "Камера 1", "location": "Вход на территорию",
     "video": str(VIDEO_DIR / "cam1_fall.mp4"), "map_pos": (28, 62)},
    {"id": "cam2", "name": "Камера 2", "location": "Центральная площадь",
     "video": str(VIDEO_DIR / "cam2_crowd.mp4"), "map_pos": (68, 32)},
    {"id": "cam3", "name": "Камера 3", "location": "Спортивный двор",
     "video": str(VIDEO_DIR / "cam3_fight.mp4"), "map_pos": (50, 78)},
]
CAMERA_BY_ID = {c["id"]: c for c in CAMERAS}

FRAME_SKIP = 2  # bumped alongside PLAYBACK_SPEED so 5x doesn't peg CPU flat-out
PLAYBACK_SPEED = 5  # x realtime -- events happen sooner, demo doesn't sit idle
CROWD_THRESHOLD = 5
FALL_COOLDOWN_SEC = 8
CROWD_COOLDOWN_SEC = 8
FIGHT_COOLDOWN_SEC = 8
FIGHT_SUSTAIN_FRAMES = 4
DETECT_CONF = 0.4

EVENT_META = {
    "fall": {"label": "Падение", "emoji": "🚨", "severity": "high"},
    "fight": {"label": "Драка", "emoji": "🥊", "severity": "high"},
    "crowd": {"label": "Скопление людей", "emoji": "👥", "severity": "medium"},
}

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
    --accent: #d2232a;
    --accent-dim: rgba(210,35,42,0.14);
    --panel-3: #1a1a1d;
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

/* camera tabs */
.cam-tab-active > button {
    border-color: var(--accent) !important; background: var(--accent-dim) !important;
    box-shadow: inset 0 0 0 1px var(--accent);
}

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

/* KPI stat row */
.stat-row {
    display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 24px;
}
.stat-tile {
    background: var(--panel); border: 1px solid var(--border); border-radius: 14px;
    padding: 16px 18px; position: relative; overflow: hidden;
}
.stat-tile::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: var(--tile-accent, var(--accent));
}
.stat-value {
    font-size: 1.7rem; font-weight: 800; letter-spacing: -0.02em; line-height: 1.1;
    color: var(--text); font-family: 'JetBrains Mono', monospace;
}
.stat-label {
    font-size: 0.72rem; font-weight: 600; color: var(--text-dim); margin-top: 6px;
    text-transform: uppercase; letter-spacing: 0.06em;
}
.stat-tile.alert .stat-value { color: var(--red); }
.stat-tile.ok .stat-value { color: var(--green); }

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
    position: relative; border-radius: 10px; overflow: hidden; height: 160px;
    background:
        linear-gradient(rgba(79,107,255,0.05) 1px, transparent 1px) 0 0/22px 22px,
        linear-gradient(90deg, rgba(79,107,255,0.05) 1px, transparent 1px) 0 0/22px 22px,
        #0d0d10;
    border: 1px solid var(--border);
}
.map-dot {
    position: absolute; width: 13px; height: 13px; border-radius: 50%;
    transform: translate(-50%, -50%);
}
.map-dot::after {
    content: ''; position: absolute; inset: -11px; border-radius: 50%;
    border: 1px solid currentColor; opacity: 0.3;
}
.map-dot.active::before {
    content: ''; position: absolute; inset: -18px; border-radius: 50%;
    border: 1px dashed currentColor; opacity: 0.6;
}
.map-legend {
    display: flex; flex-wrap: wrap; gap: 6px 14px; margin-top: 12px; font-size: 0.72rem;
    color: var(--text-dim); font-weight: 500;
}
.map-legend span.sw { width: 8px; height: 8px; border-radius: 50%; display: inline-block; margin-right: 5px; }

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

/* top nav */
.topnav {
    display: flex; align-items: center; justify-content: space-between;
    padding: 14px 0 22px 0; margin-bottom: 8px; border-bottom: 1px solid var(--border);
}
.topnav-brand {
    display: flex; align-items: center; gap: 10px; font-weight: 800; font-size: 1.05rem;
    letter-spacing: -0.01em; color: var(--text);
}
.topnav-brand .badge-icon {
    width: 30px; height: 30px; border-radius: 8px; background: var(--accent);
    display: flex; align-items: center; justify-content: center; font-size: 15px;
}
.topnav-links { display: flex; gap: 30px; }
.topnav-links a {
    color: var(--text-dim); text-decoration: none; font-weight: 600; font-size: 0.86rem;
    transition: color 0.15s ease;
}
.topnav-links a:hover { color: var(--text); }

/* hero */
.hero { padding: 56px 0 40px 0; max-width: 760px; }
.hero h1 {
    font-size: 3.1rem; font-weight: 800; letter-spacing: -0.035em; line-height: 1.08;
    margin: 0 0 20px 0; color: var(--text);
}
.hero h1 .accent { color: var(--accent); }
.hero-sub {
    font-size: 1.08rem; color: var(--text-dim); line-height: 1.6; margin-bottom: 28px; max-width: 620px;
}
.hero-cta {
    display: inline-flex; align-items: center; gap: 8px;
    background: var(--accent); color: white !important; text-decoration: none;
    font-weight: 700; font-size: 0.92rem; padding: 13px 26px; border-radius: 100px;
    transition: filter 0.15s ease;
}
.hero-cta:hover { filter: brightness(1.12); }

/* about bullets */
.about-row {
    display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin: 8px 0 44px 0;
}
.about-card {
    background: var(--panel); border: 1px solid var(--border); border-radius: 16px; padding: 22px;
}
.about-card .num {
    font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; font-weight: 700;
    color: var(--accent); margin-bottom: 10px; letter-spacing: 0.05em;
}
.about-card h4 { font-size: 1rem; font-weight: 700; margin: 0 0 8px 0; color: var(--text); }
.about-card p { font-size: 0.85rem; color: var(--text-dim); line-height: 1.55; margin: 0; }

/* section heading (used before dashboard / tech / faq) */
.section-heading {
    display: flex; align-items: baseline; justify-content: space-between;
    margin: 56px 0 24px 0; padding-top: 32px; border-top: 1px solid var(--border);
}
.section-heading h2 {
    font-size: 1.9rem; font-weight: 800; letter-spacing: -0.02em; margin: 0; color: var(--text);
}
.section-heading .section-kicker {
    font-size: 0.72rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase;
    color: var(--accent); margin-bottom: 8px; display: block;
}
.section-sub { color: var(--text-dim); font-size: 0.95rem; margin-top: 10px; max-width: 620px; }

/* tech cards */
.tech-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-top: 8px; }
.tech-card {
    background: linear-gradient(160deg, var(--panel-3), var(--panel));
    border: 1px solid var(--border); border-radius: 18px; padding: 26px;
    transition: border-color 0.2s ease;
}
.tech-card:hover { border-color: var(--accent); }
.tech-card .tech-icon {
    width: 44px; height: 44px; border-radius: 12px; background: var(--accent-dim);
    display: flex; align-items: center; justify-content: center; font-size: 21px; margin-bottom: 16px;
}
.tech-card h3 { font-size: 1.08rem; font-weight: 700; margin: 0 0 10px 0; color: var(--text); }
.tech-card p { font-size: 0.87rem; color: var(--text-dim); line-height: 1.6; margin: 0 0 12px 0; }
.tech-card .tech-tag {
    display: inline-block; font-family: 'JetBrains Mono', monospace; font-size: 0.7rem;
    color: var(--accent); background: var(--accent-dim); padding: 3px 9px; border-radius: 6px;
}

/* FAQ */
div[data-testid="stExpander"] {
    background: var(--panel) !important; border: 1px solid var(--border) !important;
    border-radius: 12px !important; margin-bottom: 10px;
}
div[data-testid="stExpander"] summary {
    font-weight: 600 !important; font-size: 0.92rem !important; color: var(--text) !important;
}
div[data-testid="stExpander"] p { color: var(--text-dim) !important; font-size: 0.88rem !important; line-height: 1.6 !important; }

/* footer */
.site-footer {
    margin-top: 60px; padding: 28px 0 10px 0; border-top: 1px solid var(--border);
    display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 10px;
}
.site-footer .foot-brand { font-weight: 700; color: var(--text); font-size: 0.92rem; }
.site-footer .foot-meta {
    color: var(--text-dim); font-size: 0.78rem; font-family: 'JetBrains Mono', monospace;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.markdown(
    """
    <div class="topnav">
        <div class="topnav-brand"><span class="badge-icon">🛡️</span> SMART SAFETY</div>
        <div class="topnav-links">
            <a href="#dashboard-anchor">Демо</a>
            <a href="#tech-anchor">Технологии</a>
            <a href="#faq-anchor">FAQ</a>
        </div>
    </div>

    <div class="hero">
        <div class="eyebrow-badge"><span class="dot"></span>Видеоаналитика для городской безопасности</div>
        <h1>Мы видим то,<br><span class="accent">что действительно важно.</span></h1>
        <div class="hero-sub">Smart Safety — прототип ситуационного центра для городских камер видеонаблюдения.
            Система разбирает видеопоток покадрово с помощью YOLO11-pose, отслеживает людей и автоматически
            распознаёт падения, драки и опасные скопления людей — без участия оператора, с оповещением
            и передачей наряда в один клик.</div>
        <a class="hero-cta" href="#dashboard-anchor">Смотреть демо ↓</a>
    </div>

    <div class="about-row">
        <div class="about-card">
            <div class="num">01</div>
            <h4>Зачем</h4>
            <p>Операторы физически не могут одинаково внимательно следить за десятками экранов часами —
                усталость и рутина снижают реакцию на реальные ЧП.</p>
        </div>
        <div class="about-card">
            <div class="num">02</div>
            <h4>Как</h4>
            <p>YOLO11-pose детектирует людей и их позу на каждом кадре, эвристики отслеживают резкое изменение
                позы, скорость и перекрытие боксов между людьми.</p>
        </div>
        <div class="about-card">
            <div class="num">03</div>
            <h4>Что дальше</h4>
            <p>При срабатывании — тревога на карте, запись в ленту событий и кнопка «Отправить наряд»
                для мгновенной передачи инцидента дежурной службе.</p>
        </div>
    </div>

    <div id="dashboard-anchor"></div>
    <div class="section-heading">
        <div>
            <span class="section-kicker">Живой демо-стенд</span>
            <h2>Ситуационный центр</h2>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


def init_state():
    if "alerts" not in st.session_state:
        st.session_state.alerts = []
    if "active_cam" not in st.session_state:
        st.session_state.active_cam = CAMERAS[0]["id"]
    if "last_fall_ts" not in st.session_state:
        st.session_state.last_fall_ts = {c["id"]: {} for c in CAMERAS}
    if "last_crowd_ts" not in st.session_state:
        st.session_state.last_crowd_ts = {c["id"]: 0.0 for c in CAMERAS}
    if "last_fight_ts" not in st.session_state:
        st.session_state.last_fight_ts = {c["id"]: 0.0 for c in CAMERAS}
    if "fight_streak" not in st.session_state:
        st.session_state.fight_streak = {c["id"]: 0 for c in CAMERAS}
    if "fight_last_pair" not in st.session_state:
        st.session_state.fight_last_pair = {c["id"]: None for c in CAMERAS}
    if "model" not in st.session_state:
        with st.spinner("Загружаю YOLO модель (первый запуск может скачать веса)..."):
            model, has_pose = load_model()
        st.session_state.model = model
        st.session_state.has_pose = has_pose
    if "trackers" not in st.session_state:
        st.session_state.trackers = {c["id"]: SimpleTracker() for c in CAMERAS}
    if "running" not in st.session_state:
        st.session_state.running = False


init_state()


def add_alert(camera_id, alert_type, frame_bgr, severity):
    thumb = cv2.resize(frame_bgr, (160, 120))
    thumb_rgb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
    alert = {
        "id": str(uuid.uuid4())[:8],
        "camera_id": camera_id,
        "camera_name": CAMERA_BY_ID[camera_id]["name"],
        "type": alert_type,
        "timestamp": time.strftime("%H:%M:%S"),
        "severity": severity,
        "thumbnail": thumb_rgb,
        "status": "new",
    }
    st.session_state.alerts.insert(0, alert)
    st.session_state.alerts = st.session_state.alerts[:200]


def camera_status(camera_id):
    active = [a for a in st.session_state.alerts
              if a["status"] == "new" and a["camera_id"] == camera_id]
    if any(a["type"] in ("fall", "fight") for a in active):
        return "red"
    if any(a["type"] == "crowd" for a in active):
        return "yellow"
    return "green"


def overall_status():
    active = [a for a in st.session_state.alerts if a["status"] == "new"]
    urgent = [a for a in active if a["type"] in ("fall", "fight")]
    if urgent:
        a = urgent[0]
        return "red", f"ТРЕВОГА: {EVENT_META[a['type']]['label'].lower()} — {a['camera_name']}"
    crowd = [a for a in active if a["type"] == "crowd"]
    if crowd:
        a = crowd[0]
        return "yellow", f"ВНИМАНИЕ: скопление людей — {a['camera_name']}"
    return "green", "Штатная ситуация — все камеры в норме"


def render_stats():
    active_alerts = [a for a in st.session_state.alerts if a["status"] == "new"]
    total_alerts = len(st.session_state.alerts)
    dispatched = total_alerts - len(active_alerts)
    cams_alarmed = len({a["camera_id"] for a in active_alerts})

    tiles = [
        (str(len(CAMERAS)), "Камер онлайн", ""),
        (str(len(active_alerts)), "Активных тревог", "alert" if active_alerts else "ok"),
        (str(cams_alarmed), "Камер в тревоге", "alert" if cams_alarmed else "ok"),
        (str(total_alerts), "Событий за сессию", ""),
    ]
    tiles_html = "".join(
        f'<div class="stat-tile {cls}"><div class="stat-value">{val}</div>'
        f'<div class="stat-label">{label}</div></div>'
        for val, label, cls in tiles
    )
    stats_slot.markdown(f'<div class="stat-row">{tiles_html}</div>', unsafe_allow_html=True)


# ---------- layout ----------
stats_slot = st.empty()
render_stats()

left_col, right_col = st.columns([2, 1], gap="large")

with left_col:
    cam_cols = st.columns(len(CAMERAS))
    cam_click = None
    for col, cam in zip(cam_cols, CAMERAS):
        dot = {"green": "🟢", "yellow": "🟡", "red": "🔴"}[camera_status(cam["id"])]
        is_active = st.session_state.active_cam == cam["id"]
        prefix = "▶ " if is_active else ""
        with col:
            if col.button(f"{prefix}{dot} {cam['name']}", key=f"camtab_{cam['id']}", width="stretch"):
                cam_click = cam["id"]

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

if cam_click is not None:
    st.session_state.active_cam = cam_click


def render_status():
    color, label = overall_status()
    status_slot.markdown(
        f'<div class="status-banner status-{color}"><span class="status-dot"></span>{label}</div>',
        unsafe_allow_html=True,
    )
    dot_colors = {"green": "#22c55e", "yellow": "#eab308", "red": "#ef4444"}
    dots_html = ""
    legend_html = ""
    for cam in CAMERAS:
        st_color = camera_status(cam["id"])
        c = dot_colors[st_color]
        x, y = cam["map_pos"]
        active_cls = "active" if cam["id"] == st.session_state.active_cam else ""
        dots_html += (
            f'<div class="map-dot {active_cls}" style="left:{x}%; top:{y}%; '
            f'background:{c}; color:{c}; box-shadow:0 0 16px 3px {c}99;"></div>'
        )
        legend_html += (
            f'<span><span class="sw" style="background:{c};"></span>{cam["name"]}</span>'
        )
    map_slot.markdown(
        f"""
        <div class="panel-card">
            <div class="panel-label">Карта объекта</div>
            <div class="map-wrap">{dots_html}</div>
            <div class="map-legend">{legend_html}</div>
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

    visible_alerts = st.session_state.alerts[:20]
    with alerts_slot.container():
        for i, alert in enumerate(visible_alerts):
            meta = EVENT_META[alert["type"]]
            accent = "var(--red)" if alert["severity"] == "high" else "var(--yellow)"
            c1, c2, c3 = st.columns([1, 2.2, 1], vertical_alignment="center")
            with c1:
                st.image(alert["thumbnail"], width="stretch")
            with c2:
                type_label = f"{meta['emoji']} {meta['label']}"
                badge = ('<span class="badge-new">● НОВЫЙ</span>' if alert["status"] == "new"
                         else '<span class="badge-dispatched">✔ НАРЯД ПЕРЕДАН</span>')
                st.markdown(
                    f'<div class="alert-type" style="color:{accent}">{type_label}</div>'
                    f'<div class="alert-meta">{alert["camera_name"]} · {alert["timestamp"]} · '
                    f'severity: {alert["severity"]}</div>'
                    f'{badge}',
                    unsafe_allow_html=True,
                )
            with c3:
                if alert["status"] == "new":
                    if st.button("Отправить наряд", key=f"dispatch_{alert['id']}_{call_id}_{i}",
                                 width="stretch"):
                        alert["status"] = "dispatched"
                        st.rerun()
            if i < len(visible_alerts) - 1:
                st.markdown('<hr style="border-color: var(--border); margin: 10px 0;">',
                            unsafe_allow_html=True)
        if len(st.session_state.alerts) > len(visible_alerts):
            st.markdown(
                f'<div class="alert-meta" style="text-align:center; margin-top:8px;">'
                f'+ ещё {len(st.session_state.alerts) - len(visible_alerts)} в истории</div>',
                unsafe_allow_html=True,
            )


render_status()
render_alerts()

if start_btn:
    st.session_state.running = True
if stop_btn:
    st.session_state.running = False


def process_video():
    cam = CAMERA_BY_ID[st.session_state.active_cam]
    cam_id = cam["id"]

    cap = cv2.VideoCapture(cam["video"])
    if not cap.isOpened():
        st.error(f"Не удалось открыть видео: {cam['video']}")
        st.session_state.running = False
        return

    frame_idx = 0
    model = st.session_state.model
    has_pose = st.session_state.has_pose
    tracker = st.session_state.trackers[cam_id]

    # Pace the loop instead of spinning the CPU flat-out (an unthrottled loop is
    # what got the whole app CPU-throttled on Streamlit Cloud's free tier during
    # testing) -- but play back at PLAYBACK_SPEED times the source fps so footage
    # doesn't sit at real-world 1x speed waiting for an event to happen.
    source_fps = cap.get(cv2.CAP_PROP_FPS) or 15
    frame_interval = 1.0 / (source_fps * PLAYBACK_SPEED)
    last_frame_time = time.time()

    while st.session_state.running and st.session_state.active_cam == cam_id:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop the video
            continue

        frame_idx += 1
        if frame_idx % FRAME_SKIP != 0:
            continue

        elapsed = time.time() - last_frame_time
        if elapsed < frame_interval:
            time.sleep(frame_interval - elapsed)
        last_frame_time = time.time()

        detections = detect_people(frame, model, has_pose, conf=DETECT_CONF)
        assigned = tracker.update(detections)

        display = frame.copy()
        now = time.time()

        fall_active = False
        for track_id, det in assigned:
            x1, y1, x2, y2 = [int(v) for v in det["box"]]
            ar = compute_aspect_ratio(det["box"])
            angle = compute_torso_angle(det["keypoints"])
            tracker.record_metrics(track_id, ar, angle)

            history = tracker.get_history(track_id)
            is_fall = check_fall(history, tracker.get_seen_standing(track_id))
            fall_active = fall_active or is_fall

            box_color = (0, 0, 255) if is_fall else (0, 200, 0)
            cv2.rectangle(display, (x1, y1), (x2, y2), box_color, 2)
            label = f"ID{track_id}" + (" FALL" if is_fall else "")
            cv2.putText(display, label, (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

            if is_fall:
                last_ts = st.session_state.last_fall_ts[cam_id].get(track_id, 0)
                if now - last_ts > FALL_COOLDOWN_SEC:
                    st.session_state.last_fall_ts[cam_id][track_id] = now
                    add_alert(cam_id, "fall", frame, "high")
                    render_alerts()
                    render_status()
                    render_stats()

        is_crowd = check_crowd(detections, threshold=CROWD_THRESHOLD)
        if is_crowd:
            cv2.putText(display, f"CROWD: {len(detections)} people", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2)
            if now - st.session_state.last_crowd_ts[cam_id] > CROWD_COOLDOWN_SEC:
                st.session_state.last_crowd_ts[cam_id] = now
                add_alert(cam_id, "crowd", frame, "medium")
                render_alerts()
                render_status()
                render_stats()

        is_fight_frame, fight_pair = check_fight(assigned, tracker)
        if is_fight_frame:
            # Require the SAME pair of people to stay overlapping+fast across
            # consecutive frames. A real fight keeps the same two people; a dense
            # crowd's occlusion-confused tracker pairs shuffle randomly frame to
            # frame, so this alone kills the crowd false positives that raw
            # "is_fight_frame" streaks let through (tested empirically).
            pair_key = frozenset(fight_pair)
            if pair_key == st.session_state.fight_last_pair[cam_id]:
                st.session_state.fight_streak[cam_id] += 1
            else:
                st.session_state.fight_streak[cam_id] = 1
            st.session_state.fight_last_pair[cam_id] = pair_key
        else:
            st.session_state.fight_streak[cam_id] = 0
            st.session_state.fight_last_pair[cam_id] = None
        is_fight = st.session_state.fight_streak[cam_id] >= FIGHT_SUSTAIN_FRAMES
        if is_fight:
            cv2.putText(display, "FIGHT DETECTED", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            if now - st.session_state.last_fight_ts[cam_id] > FIGHT_COOLDOWN_SEC:
                st.session_state.last_fight_ts[cam_id] = now
                add_alert(cam_id, "fight", frame, "high")
                render_alerts()
                render_status()
                render_stats()

        status_text = ("FALL DETECTED" if fall_active else
                        "FIGHT DETECTED" if is_fight else
                        "CROWD" if is_crowd else "OK")
        status_color = ((0, 0, 255) if status_text in ("FALL DETECTED", "FIGHT DETECTED") else
                         (0, 165, 255) if status_text == "CROWD" else (0, 200, 0))
        cv2.putText(display, f"Status: {status_text}", (20, display.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        display_rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        video_slot.image(display_rgb, width="stretch",
                          caption=f"{cam['name']} — {cam['location']}")

    cap.release()


if st.session_state.running:
    process_video()
else:
    active_cam = CAMERA_BY_ID[st.session_state.active_cam]
    video_slot.markdown(
        f"""
        <div style="border:1px dashed var(--border); border-radius:14px; padding:80px 20px;
            text-align:center; color:var(--text-dim); background:var(--panel);">
            <div style="font-size:2rem; margin-bottom:10px;">▶️</div>
            Нажми «Старт», чтобы запустить обработку видео — {active_cam['name']}
            ({active_cam['location']})
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------- static sections below the live dashboard ----------
st.markdown(
    """
    <div id="tech-anchor"></div>
    <div class="section-heading">
        <div>
            <span class="section-kicker">Под капотом</span>
            <h2>Ключевые технологии</h2>
            <div class="section-sub">Три независимые эвристики поверх одной модели детекции людей —
                каждая настроена и проверена на реальных видео, а не на синтетике.</div>
        </div>
    </div>
    <div class="tech-grid">
        <div class="tech-card">
            <div class="tech-icon">🚨</div>
            <h3>Детекция падений</h3>
            <p>Отслеживаем соотношение сторон бокса человека и угол торса по ключевым точкам YOLO11-pose.
                Падение засчитывается, только если человек до этого стоял и его поза устойчиво (не менее
                0.5 сек) осталась «лежачей» — так наклоны и приседания не дают ложных тревог.</p>
            <span class="tech-tag">YOLO11-pose · aspect ratio · torso angle</span>
        </div>
        <div class="tech-card">
            <div class="tech-icon">🥊</div>
            <h3>Детекция драк</h3>
            <p>Ищем пары людей с перекрывающимися боксами, которые одновременно быстро двигаются в кадре.
                Тревога срабатывает, только если это одна и та же пара несколько кадров подряд — так плотная
                толпа с постоянно меняющимися соседями не путается с реальной дракой.</p>
            <span class="tech-tag">IoU overlap · frame-to-frame speed</span>
        </div>
        <div class="tech-card">
            <div class="tech-icon">👥</div>
            <h3>Скопление людей</h3>
            <p>Считаем количество людей в кадре на каждый обработанный кадр. При превышении порога
                (по умолчанию 5 человек) формируется алерт «жёлтого» уровня — с меньшим приоритетом,
                чем падение или драка, но всё ещё требующий внимания оператора.</p>
            <span class="tech-tag">Person count threshold</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div id="faq-anchor"></div>
    <div class="section-heading">
        <div>
            <span class="section-kicker">Вопросы и ответы</span>
            <h2>FAQ</h2>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

FAQ_ITEMS = [
    ("Это настоящее видео или сгенерированное?",
     "Все три ролика — реальные видеозаписи: падения взяты из открытого академического датасета "
     "UR Fall Detection (создан именно для тестирования алгоритмов детекции падений), сцены с дракой "
     "и толпой — из открытых стоковых видео. Никакого синтетического или AI-сгенерированного контента."),
    ("Можно ли подключить настоящую камеру вместо видеофайла?",
     "Да, архитектурно это замена одной строки — вместо cv2.VideoCapture(путь_к_файлу) подставляется "
     "cv2.VideoCapture(rtsp_поток_камеры) или индекс веб-камеры. Вся остальная логика детекции и алертов "
     "работает без изменений."),
    ("Насколько точны эвристики драки и падения?",
     "Это эвристики поверх детекции позы, а не отдельно обученные модели — они откалиброваны и проверены "
     "на конкретных тестовых видео, но не гарантируют точность production-уровня. Для реального внедрения "
     "потребуется дообучение на размеченных данных конкретных объектов и ракурсов камер."),
    ("Что происходит при нажатии «Отправить наряд»?",
     "В этом прототипе — это заглушка: статус алерта в интерфейсе меняется на «Наряд передан». "
     "Реальная интеграция с диспетчерской службой (API, SMS, пуш-уведомление) не входит в объём MVP, "
     "но именно в эту точку она бы подключалась."),
    ("Почему видео проигрывается в ускоренном темпе?",
     "Чтобы не ждать реального времени до следующего события на демонстрации — воспроизведение идёт "
     "в 1.6 раза быстрее реального, при этом обработка кадров всё равно ограничена по CPU, "
     "чтобы не создавать чрезмерную нагрузку на сервер."),
    ("Почему в системе всего 3 камеры?",
     "Это прототип для демонстрации подхода, не production-система. Добавить N-ю камеру — значит добавить "
     "ещё один элемент в список CAMERAS с своим видео и локацией; остальная логика (трекер, алерты, карта) "
     "уже рассчитана на произвольное число камер."),
]

for question, answer in FAQ_ITEMS:
    with st.expander(question):
        st.write(answer)

st.markdown(
    """
    <div class="site-footer">
        <div class="foot-brand">🛡️ Smart Safety</div>
        <div class="foot-meta">YOLO11-pose · OpenCV · Streamlit — прототип для хакатона по цифровизации города</div>
    </div>
    """,
    unsafe_allow_html=True,
)
