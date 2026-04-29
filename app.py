import random
import streamlit as st
import pandas as pd
import plotly.express as px

from backend import (
    run_pipeline,
    CITY_PRESETS,
    generate_llm_recommendation,
    fetch_tad_positions,
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Stargazing Assistant",
    page_icon="🌌",
    layout="wide",
)


# ============================================================
# CUSTOM CSS — Nebula Dusk Design System
# ============================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,300&family=DM+Serif+Display:ital@0;1&display=swap');
    @import url('https://fonts.googleapis.com/icon?family=Material+Icons');

    /* ── Design tokens ─────────────────────────────────────── */
    :root {
        --bg:           #090614;
        --sidebar:      #0c0818;
        --card:         #100d1e;
        --card-hover:   #16122a;
        --accent:       #c478d2;
        --accent-soft:  rgba(196,120,210,0.15);
        --border:       rgba(196,120,210,0.16);
        --slider-track: #1a1430;
        --text:         #e4dff0;
        --text-muted:   #8880a0;
        --text-dim:     #554f6a;
        --accent-glow:  rgba(196,120,210,0.06);
        --tx: background 0.15s linear, color 0.15s linear, border-color 0.15s linear;
    }

    /* ── Base ─────────────────────────────────────────────── */
    html, body { background: var(--bg) !important; }
    .stApp, [data-testid="stAppViewContainer"], .main {
        background-color: var(--bg) !important;
        font-family: 'DM Sans', sans-serif;
        color: var(--text);
    }
    p, li, span, label, td, th, button, input, select, textarea {
        font-family: 'DM Sans', sans-serif;
    }
    h1, h2, h3, h4, h5, h6 {
        font-family: 'DM Serif Display', serif;
        font-weight: 400;
        color: var(--text);
    }
    .stMarkdown p {
        font-size: 14px;
        line-height: 1.67;
        color: var(--text);
        font-family: 'DM Sans', sans-serif;
    }
    .stMarkdown li {
        font-size: 14px;
        line-height: 1.7;
        color: var(--text-muted);
        font-family: 'DM Sans', sans-serif;
    }
    hr { border-color: var(--border) !important; }

    /* ── Sidebar — glassmorphism ──────────────────────────── */
    [data-testid="stSidebar"] {
        background: rgba(12,8,24,0.88) !important;
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border-right: 1px solid var(--border) !important;
    }
    [data-testid="stSidebar"] * { font-family: 'DM Sans', sans-serif; }
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span { color: var(--text) !important; font-size: 13px; }

    /* ── Buttons ──────────────────────────────────────────── */
    .stButton > button {
        font-family: 'DM Sans', sans-serif !important;
        font-size: 13px !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
        transition: var(--tx) !important;
        padding: 8px 16px !important;
    }
    .stButton > button[kind="primary"] {
        background: var(--accent-soft) !important;
        border: 1px solid var(--accent) !important;
        color: var(--accent) !important;
    }
    .stButton > button[kind="primary"]:hover {
        background: rgba(196,120,210,0.26) !important;
        border-color: rgba(196,120,210,0.55) !important;
    }
    .stButton > button:not([kind="primary"]) {
        background: transparent !important;
        border: 1px solid var(--border) !important;
        color: var(--text-muted) !important;
    }
    .stButton > button:not([kind="primary"]):hover {
        background: var(--card-hover) !important;
        border-color: rgba(196,120,210,0.3) !important;
        color: var(--text) !important;
    }

    /* ── Sliders ──────────────────────────────────────────── */
    [data-testid="stSlider"] [role="slider"] {
        background: var(--accent) !important;
        border: none !important;
        box-shadow: 0 0 8px rgba(196,120,210,0.6) !important;
        border-radius: 50% !important;
    }
    [data-testid="stSlider"] label {
        color: var(--text-muted) !important;
        font-size: 12px !important;
        font-family: 'DM Sans', sans-serif !important;
        font-weight: 500 !important;
    }

    /* ── Selectbox ────────────────────────────────────────── */
    [data-testid="stSelectbox"] > div > div {
        background: var(--slider-track) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        color: var(--text) !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 13px !important;
    }

    /* ── Checkboxes & radios ──────────────────────────────── */
    [data-testid="stCheckbox"] label span,
    [data-testid="stRadio"] label span {
        color: var(--text) !important;
        font-size: 13px !important;
        font-family: 'DM Sans', sans-serif !important;
    }

    /* ── Streamlit alerts ─────────────────────────────────── */
    [data-testid="stAlert"] {
        background: var(--card) !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
        font-family: 'DM Sans', sans-serif !important;
    }
    [data-testid="stCaptionContainer"] p {
        color: var(--text-dim) !important;
        font-size: 12px !important;
        font-family: 'DM Sans', sans-serif !important;
    }

    /* ── Expanders ────────────────────────────────────────── */
    [data-testid="stExpander"] {
        background: var(--card) !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
    }
    [data-testid="stExpander"] summary {
        color: var(--text) !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 13px !important;
    }

    /* ── Dataframe ────────────────────────────────────────── */
    [data-testid="stDataFrame"] {
        border-radius: 10px !important;
        border: 1px solid var(--border) !important;
    }

    /* ── Streamlit metric ─────────────────────────────────── */
    div[data-testid="stMetric"] {
        background: var(--card);
        padding: 14px 16px;
        border-radius: 10px;
        border: 1px solid var(--border);
    }

    /* ── Sidebar collapse button ──────────────────────────── */
    [data-testid="collapsedControl"],
    button[data-testid="baseButton-header"] {
        font-family: 'Material Icons' !important;
        color: var(--text-dim) !important;
        background: transparent !important;
        border: none !important;
    }
    [data-testid="collapsedControl"]:hover,
    button[data-testid="baseButton-header"]:hover {
        color: var(--accent) !important;
    }

    /* ── Headings from st.subheader ───────────────────────── */
    [data-testid="stHeadingWithActionElements"] h2,
    [data-testid="stHeadingWithActionElements"] h3 {
        font-family: 'DM Serif Display', serif !important;
        font-weight: 400 !important;
        color: var(--text) !important;
    }

    /* ── Sidebar Navigation (styled radio) ────────────────── */
    /* hide the default widget label — we render our own */
    [data-testid="stSidebar"] [data-testid="stRadio"] > label { display: none !important; }

    [data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] {
        display: flex !important;
        flex-direction: column !important;
        gap: 2px !important;
    }

    /* each nav item */
    [data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] label {
        display: flex !important;
        align-items: center !important;
        gap: 10px !important;
        padding: 8px 12px !important;
        border-radius: 8px !important;
        cursor: pointer !important;
        transition: var(--tx) !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 13px !important;
        font-weight: 400 !important;
        color: var(--text-muted) !important;
        position: relative !important;
        width: 100% !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] label:hover {
        background: rgba(196,120,210,0.08) !important;
        color: var(--text) !important;
    }

    /* hide Streamlit's native radio circle */
    [data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] label input { display: none !important; }
    [data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] label > div:first-of-type { display: none !important; }

    /* left bullet dot */
    [data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] label::before {
        content: "●";
        font-size: 7px;
        color: var(--text-dim);
        flex-shrink: 0;
        line-height: 1;
        transition: var(--tx);
    }

    /* active item */
    [data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) {
        background: rgba(196,120,210,0.12) !important;
        color: var(--accent) !important;
        font-weight: 500 !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] label:has(input:checked)::before {
        color: var(--accent) !important;
    }
    /* right dot on active item */
    [data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] label:has(input:checked)::after {
        content: "●";
        font-size: 7px;
        color: var(--accent);
        flex-shrink: 0;
        line-height: 1;
        margin-left: auto;
    }

    /* ═══════════════════════════════════════════════════════ */
    /* CUSTOM COMPONENTS                                       */
    /* ═══════════════════════════════════════════════════════ */

    /* ── Hero Card ────────────────────────────────────────── */
    .hero-card {
        background: linear-gradient(135deg, var(--card) 0%, var(--accent-glow) 100%);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 28px 32px;
        margin-bottom: 20px;
        position: relative;
        overflow: hidden;
        transition: var(--tx);
    }
    /* nebula glow — top-right radial, ~180px */
    .hero-glow {
        position: absolute;
        top: -60px;
        right: -60px;
        width: 220px;
        height: 220px;
        background: radial-gradient(circle, rgba(196,120,210,0.08) 0%, transparent 68%);
        pointer-events: none;
    }
    .hero-flex {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        gap: 20px;
        position: relative;
    }
    .hero-left {
        display: flex;
        align-items: center;
        gap: 20px;
    }
    .hero-icon-wrap {
        width: 52px;
        height: 52px;
        background: var(--accent-soft);
        border: 1px solid var(--border);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.35rem;
        flex-shrink: 0;
    }
    .hero-title {
        margin: 0 0 5px 0;
        font-family: 'DM Serif Display', serif;
        font-size: 2rem;
        font-weight: 400;
        color: var(--text);
        line-height: 1.15;
    }

    .hero-desc {
        margin: 0;
        font-family: 'DM Sans', sans-serif;
        font-weight: 300;
        color: var(--text-muted);
        font-size: 13px;
        line-height: 1.67;
        max-width: 540px;
    }
    .score-badge {
        display: flex;
        align-items: center;
        gap: 7px;
        background: var(--accent-soft);
        border: 1px solid rgba(196,120,210,0.35);
        color: var(--accent);
        padding: 7px 16px;
        border-radius: 99px;
        font-family: 'DM Sans', sans-serif;
        font-size: 12px;
        font-weight: 600;
        white-space: nowrap;
        flex-shrink: 0;
        transition: var(--tx);
    }
    .score-dot {
        width: 7px;
        height: 7px;
        background: var(--accent);
        border-radius: 50%;
        box-shadow: 0 0 6px var(--accent);
        display: inline-block;
        flex-shrink: 0;
    }

    /* ── Info / Metric Cards (spec: r=10px, p=14 16) ─────── */
    .metric-card {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 14px 16px;
        transition: var(--tx);
        cursor: default;
    }
    .metric-card:hover {
        background: var(--card-hover);
        border-color: rgba(196,120,210,0.28);
    }
    .metric-card-title {
        font-family: 'DM Sans', sans-serif;
        font-size: 10px;
        color: var(--text-dim);
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 10px;
        display: block;
    }
    .metric-value-row {
        display: flex;
        align-items: baseline;
        gap: 3px;
    }
    .metric-large {
        font-family: 'DM Serif Display', serif;
        font-size: 2rem;
        font-weight: 400;
        color: var(--text);
        line-height: 1;
    }
    .metric-unit {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.9rem;
        color: var(--text-muted);
        font-weight: 300;
    }

    /* ── Section Cards ────────────────────────────────────── */
    .section-card {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 24px 28px;
        margin-bottom: 16px;
        transition: var(--tx);
    }
    .section-card:hover { background: var(--card-hover); }

    .section-label {
        font-family: 'DM Sans', sans-serif;
        font-size: 10px;
        color: var(--text-dim);
        font-weight: 600;
        letter-spacing: 0.13em;
        text-transform: uppercase;
        margin: 0 0 14px 0;
        display: block;
    }
    .section-heading {
        font-family: 'DM Serif Display', serif;
        font-size: 20px;
        font-weight: 400;
        color: var(--text);
        margin: 0 0 8px 0;
    }

    /* ── Window Cards ─────────────────────────────────────── */
    .window-card {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 20px;
        min-height: 200px;
        transition: var(--tx);
    }
    .window-card:hover {
        background: var(--card-hover);
        border-color: rgba(196,120,210,0.28);
    }
    .window-rank {
        font-family: 'DM Sans', sans-serif;
        color: var(--accent);
        font-weight: 600;
        font-size: 11px;
        letter-spacing: 0.04em;
        margin-bottom: 8px;
    }
    .window-time {
        font-family: 'DM Serif Display', serif;
        color: var(--text);
        font-size: 1.15rem;
        font-weight: 400;
        margin-bottom: 6px;
    }
    .window-score {
        font-family: 'DM Serif Display', serif;
        font-size: 1.9rem;
        font-weight: 400;
        color: var(--text);
        line-height: 1.1;
    }
    .muted {
        font-family: 'DM Sans', sans-serif;
        font-weight: 300;
        color: var(--text-muted);
        font-size: 13px;
        line-height: 1.6;
    }

    /* ── Badges ───────────────────────────────────────────── */
    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 99px;
        font-family: 'DM Sans', sans-serif;
        font-size: 11px;
        font-weight: 600;
        margin-right: 6px;
        margin-top: 6px;
        transition: var(--tx);
    }
    .badge-excellent { background: rgba(34,197,94,0.12);  color: #4ade80; border: 1px solid rgba(74,222,128,0.25); }
    .badge-good      { background: rgba(20,184,166,0.12);  color: #5eead4; border: 1px solid rgba(94,234,212,0.25); }
    .badge-marginal  { background: rgba(245,158,11,0.12);  color: #fbbf24; border: 1px solid rgba(251,191,36,0.25); }
    .badge-poor      { background: rgba(249,115,22,0.12);  color: #fb923c; border: 1px solid rgba(251,146,60,0.25); }
    .badge-nogo      { background: rgba(239,68,68,0.12);   color: #f87171; border: 1px solid rgba(248,113,113,0.25); }

    /* ── Source pills ─────────────────────────────────────── */
    .source-pill {
        background: var(--accent-soft);
        color: var(--accent);
        border: 1px solid var(--border);
        padding: 4px 10px;
        border-radius: 99px;
        font-family: 'DM Sans', sans-serif;
        font-size: 11px;
        font-weight: 600;
        margin-right: 6px;
        display: inline-block;
        transition: var(--tx);
    }
    .warning-pill {
        background: rgba(245,158,11,0.10);
        color: #fbbf24;
        border: 1px solid rgba(251,191,36,0.22);
        padding: 4px 10px;
        border-radius: 99px;
        font-family: 'DM Sans', sans-serif;
        font-size: 11px;
        font-weight: 600;
        margin-right: 6px;
        display: inline-block;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# STAR FIELD
# ============================================================

def _star_field(n: int = 160) -> str:
    rng = random.Random(42)
    palette = ["#f0eaff", "#ddd0ff", "#c4a8f0", "#ffffff"]
    circles = []
    for _ in range(n):
        x   = round(rng.uniform(0, 100), 1)
        y   = round(rng.uniform(0, 100), 1)
        r   = round(rng.uniform(0.4, 1.4), 1)
        op  = round(rng.uniform(0.10, 0.70), 2)
        op2 = round(min(op + rng.uniform(0.10, 0.35), 0.92), 2)
        dur = round(rng.uniform(1.5, 4.5), 1)
        beg = round(rng.uniform(0, dur), 1)
        c   = rng.choice(palette)
        circles.append(
            f'<circle cx="{x}%" cy="{y}%" r="{r}" fill="{c}" opacity="{op}">'
            f'<animate attributeName="opacity" values="{op};{op2};{op}" '
            f'dur="{dur}s" begin="{beg}s" repeatCount="indefinite"/>'
            f'</circle>'
        )
    return "\n".join(circles)


st.markdown(
    f"""
    <svg xmlns="http://www.w3.org/2000/svg"
         style="position:fixed;top:0;left:0;width:100vw;height:100vh;
                pointer-events:none;z-index:0;"
         preserveAspectRatio="xMidYMid slice">
    {_star_field()}
    </svg>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown(
    '<p style="font-family:\'DM Sans\',sans-serif;font-size:10px;color:#554f6a;'
    'font-weight:600;letter-spacing:0.13em;text-transform:uppercase;'
    'margin:0 0 6px 0;padding:0 4px;">Navigation</p>',
    unsafe_allow_html=True,
)

selected_page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Forecast Timeline",
        "Best Windows",
        "Sky Conditions",
        "Sky Path",
        "AI Insight",
        "Methodology",
        "Raw Data",
    ],
    label_visibility="collapsed",
)

st.sidebar.divider()
st.sidebar.markdown(
    '<p style="font-family:\'DM Sans\',sans-serif;font-size:10px;color:#554f6a;'
    'font-weight:600;letter-spacing:0.13em;text-transform:uppercase;margin-bottom:4px;">'
    'User Input</p>',
    unsafe_allow_html=True,
)

input_mode = st.sidebar.radio(
    "Location input mode",
    ["City preset", "Custom latitude / longitude"],
)

city_name = "New York City"
lat = None
lon = None
timezone = None

if input_mode == "City preset":
    city_list = list(CITY_PRESETS.keys())
    city_name = st.sidebar.selectbox(
        "Select a city",
        city_list,
        index=city_list.index("New York City") if "New York City" in city_list else 0,
    )
else:
    city_name = st.sidebar.text_input("Location name", "Custom Location")
    lat = st.sidebar.number_input("Latitude", value=40.7128, format="%.6f")
    lon = st.sidebar.number_input("Longitude", value=-74.0060, format="%.6f")
    timezone = st.sidebar.text_input("Timezone", "America/New_York")

days = st.sidebar.slider("Forecast range", min_value=1, max_value=4, value=4)

bortle_index = st.sidebar.slider(
    "City lights index",
    min_value=1,
    max_value=9,
    value=5,
    help="1 = very dark rural sky; 9 = heavily light-polluted urban sky.",
)

include_tad = st.sidebar.checkbox(
    "Use Timeanddate event data",
    value=True,
    help="Adds detailed moon/twilight event features. This can affect the score.",
)

include_positions = st.sidebar.checkbox(
    "Use Sun/Moon position data",
    value=False,
    help="Only used for Sky Path visualization. This does not affect the score.",
)

include_llm = st.sidebar.checkbox(
    "Generate AI recommendation",
    value=False,
    help="Only explains fixed model output. This does not affect the score.",
)

run_button = st.sidebar.button("▶ Run Pipeline", use_container_width=True, type="primary")

if st.sidebar.button("Clear cached data", use_container_width=True):
    st.cache_data.clear()
    st.sidebar.success("Cache cleared. Run the pipeline again.")


# ============================================================
# CACHE FUNCTIONS
# ============================================================

@st.cache_data(ttl=1800, show_spinner=False)
def cached_run_pipeline(city_name, lat, lon, timezone, days, bortle_index, include_tad):
    return run_pipeline(
        city_name=city_name,
        lat=lat,
        lon=lon,
        timezone=timezone,
        days=days,
        bortle_index=bortle_index,
        include_tad=include_tad,
        include_positions=False,
        include_llm=False,
    )


@st.cache_data(ttl=1800, show_spinner=False)
def cached_fetch_positions(lat, lon):
    return fetch_tad_positions(lat=lat, lon=lon)


@st.cache_data(ttl=1800, show_spinner=False)
def cached_generate_llm(context):
    return generate_llm_recommendation(context)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_numeric(value, default=None):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def badge_class(label):
    label = str(label).lower()
    if label == "excellent":
        return "badge-excellent"
    if label == "good":
        return "badge-good"
    if label == "marginal":
        return "badge-marginal"
    if label == "poor":
        return "badge-poor"
    return "badge-nogo"


def explain_row(row):
    reasons = []
    cloud        = safe_numeric(row.get("cloud_value"), 100)
    transparency = safe_numeric(row.get("transparency_value"), 5)
    seeing       = safe_numeric(row.get("seeing_value"), 5)
    moon         = safe_numeric(row.get("moon_illuminated_pct"), 50)
    is_dark      = bool(row.get("is_dark_enough", False))
    is_moon_up   = bool(row.get("is_moon_up", False))

    if cloud is not None:
        if cloud <= 30:   reasons.append("low cloud cover")
        elif cloud <= 60: reasons.append("moderate cloud cover")
        else:             reasons.append("high cloud cover")

    if transparency is not None:
        if transparency <= 2:   reasons.append("strong transparency")
        elif transparency <= 4: reasons.append("acceptable transparency")
        else:                   reasons.append("weak transparency")

    if seeing is not None:
        if seeing <= 2:   reasons.append("stable seeing")
        elif seeing <= 4: reasons.append("moderate seeing")
        else:             reasons.append("poor seeing")

    if moon is not None:
        if moon <= 30:   reasons.append("low moon illumination")
        elif moon <= 70: reasons.append("moderate moon illumination")
        else:            reasons.append("bright moon")

    reasons.append("dark enough sky" if is_dark else "not fully dark")
    reasons.append("moon may interfere" if is_moon_up else "moon is not a major issue")
    return ", ".join(reasons)


def display_empty_result_debug(score_df, top_windows):
    st.warning("No recommended stargazing windows were found for this location and setting.")
    st.markdown(
        """
        This usually means one of these happened:

        - The score column is missing or all NaN
        - Forecast conditions are too poor
        - The API returned data in a different shape than expected
        - The scoring logic is too strict
        """
    )
    st.write("score_df shape:", score_df.shape if score_df is not None else None)
    st.write("top_windows shape:", top_windows.shape if top_windows is not None else None)

    if score_df is not None and not score_df.empty:
        debug_cols = [
            "local_dt", "date", "stargazing_score", "recommendation",
            "cloud_value", "transparency_value", "seeing_value",
            "moon_illuminated_pct", "is_dark_enough", "is_moon_up",
            "visibility_penalty", "transparency_norm", "seeing_norm",
            "humidity_quality", "moon_brightness_penalty", "effective_darkness",
            "atmospheric_score",
        ]
        debug_cols = [c for c in debug_cols if c in score_df.columns]
        st.subheader("First 30 scored rows")
        st.dataframe(score_df[debug_cols].head(30), use_container_width=True)
        if "stargazing_score" in score_df.columns:
            st.subheader("Score summary")
            st.write(score_df["stargazing_score"].describe())

    st.stop()


def _themed_layout(fig, height: int = 480):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans, sans-serif", color="#8880a0", size=12),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(196,120,210,0.16)",
            font=dict(color="#8880a0", size=11),
        ),
        xaxis=dict(
            gridcolor="rgba(196,120,210,0.08)",
            linecolor="rgba(196,120,210,0.12)",
            tickfont=dict(color="#554f6a"),
        ),
        yaxis=dict(
            gridcolor="rgba(196,120,210,0.08)",
            linecolor="rgba(196,120,210,0.12)",
            tickfont=dict(color="#554f6a"),
        ),
    )
    return fig


def build_score_heatmap(score_df):
    heatmap_df = score_df.copy()
    heatmap_df["local_dt"] = pd.to_datetime(heatmap_df["local_dt"], errors="coerce")
    heatmap_df = heatmap_df.dropna(subset=["local_dt"])
    heatmap_df["date_str"] = heatmap_df["local_dt"].dt.strftime("%b %d")
    heatmap_df["hour"] = heatmap_df["local_dt"].dt.hour
    pivot = heatmap_df.pivot_table(
        index="date_str", columns="hour", values="stargazing_score", aggfunc="mean",
    )
    fig = px.imshow(
        pivot, text_auto=".0f", aspect="auto",
        title="Stargazing Score Heatmap by Date and Hour",
        labels=dict(x="Hour of Day", y="Date", color="Score"),
        zmin=0, zmax=100,
        color_continuous_scale=[[0, "#100d1e"], [0.5, "#6b2f7a"], [1, "#c478d2"]],
    )
    return _themed_layout(fig, 440)


def build_factor_chart(score_df):
    factor_cols = [
        "cloud_value", "transparency_value", "seeing_value",
        "moon_illuminated_pct", "effective_darkness", "atmospheric_score", "visibility_penalty",
    ]
    existing_cols = [c for c in factor_cols if c in score_df.columns]
    plot_df = score_df.copy()
    plot_df["local_dt"] = pd.to_datetime(plot_df["local_dt"], errors="coerce")
    plot_df = plot_df.dropna(subset=["local_dt"])
    long_df = plot_df.melt(
        id_vars=["local_dt"], value_vars=existing_cols, var_name="factor", value_name="value",
    )
    fig = px.line(
        long_df, x="local_dt", y="value", color="factor",
        title="Key Stargazing Factors Over Time",
    )
    return _themed_layout(fig, 520)


def build_recommendation_distribution(score_df):
    if "recommendation" not in score_df.columns:
        return None
    dist = score_df["recommendation"].value_counts().reset_index()
    dist.columns = ["recommendation", "count"]
    fig = px.bar(
        dist, x="recommendation", y="count", color="recommendation",
        title="Distribution of Forecasted Stargazing Quality",
        color_discrete_sequence=["#c478d2", "#9b5ab0", "#7a3d8c", "#5c2d6b", "#3e1e4a"],
    )
    return _themed_layout(fig, 420)


def render_source_badges(result):
    weather_source   = result.get("weather_source", "Unknown")
    astronomy_source = result.get("astronomy_source", "Unknown")
    timezone_value   = result.get("timezone", "Unknown")
    w_cls = "warning-pill" if "fallback" in weather_source.lower() else "source-pill"
    a_cls = "warning-pill" if "fallback" in astronomy_source.lower() else "source-pill"
    st.markdown(
        f'<span class="{w_cls}">Weather: {weather_source}</span>'
        f'<span class="{a_cls}">Astronomy: {astronomy_source}</span>'
        f'<span class="source-pill">Timezone: {timezone_value}</span>',
        unsafe_allow_html=True,
    )


def render_window_card(rank, row):
    score      = safe_numeric(row.get("stargazing_score"), 0)
    label      = row.get("recommendation", "N/A")
    time_label = row.get("time_label", "N/A")
    explanation = explain_row(row)
    st.markdown(
        f"""
        <div class="window-card">
            <div class="window-rank">#{rank} Recommended Window</div>
            <div class="window-time">{time_label}</div>
            <div class="window-score">{score:.1f}</div>
            <span class="badge {badge_class(label)}">{label}</span>
            <p class="muted" style="margin-top:12px;">{explanation}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def fmt_hour(h):
    if h == 0:  return "12am"
    if h < 12:  return f"{h}am"
    if h == 12: return "12pm"
    return f"{h - 12}pm"


# ============================================================
# INITIAL STATE
# ============================================================

if not run_button:
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-glow"></div>
            <div class="hero-flex">
                <div class="hero-left">
                    <div class="hero-icon-wrap">🚀</div>
                    <div>
                        <h1 class="hero-title">Stargazing Assistant</h1>
                        <p class="hero-desc">
                            A live decision-support tool for finding the best stargazing windows
                            using weather, astronomy, light pollution, scoring logic, and optional
                            AI explanation.
                        </p>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-card">
            <h3 class="section-heading">Ready to find your best stargazing window?</h3>
            <p class="muted">
                Choose a location from the sidebar, adjust the Bortle light pollution index,
                and run the live recommendation pipeline.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-card">
            <span class="section-label">What this app does</span>
            <ul style="color:#8880a0; margin:0; padding-left:18px; line-height:2.1;
                       font-family:'DM Sans',sans-serif; font-size:14px; font-weight:300;">
                <li>Fetches live weather and astronomy data</li>
                <li>Scores each forecast hour using cloud, transparency, seeing, darkness, and moon conditions</li>
                <li>Ranks the best observing windows</li>
                <li>Visualizes forecast trends, factor diagnostics, and optional Sun/Moon sky path</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.stop()


# ============================================================
# RUN PIPELINE
# ============================================================

with st.spinner("Fetching live weather and astronomy data..."):
    try:
        result = cached_run_pipeline(
            city_name=city_name,
            lat=lat,
            lon=lon,
            timezone=timezone,
            days=days,
            bortle_index=bortle_index,
            include_tad=include_tad,
        )
    except Exception as e:
        st.error("The live scoring pipeline failed.")
        st.exception(e)
        st.stop()


# ============================================================
# LOAD RESULTS
# ============================================================

score_df     = result.get("score_df", pd.DataFrame())
top_windows  = result.get("top_windows", pd.DataFrame())
daily_summary = result.get("daily_summary", pd.DataFrame())
master_df    = result.get("master_df", pd.DataFrame())
weather_df   = result.get("weather_df", pd.DataFrame())
ip_geo_df    = result.get("ip_geo_df", pd.DataFrame())
event_df     = result.get("event_df", pd.DataFrame())

position_df = pd.DataFrame()
llm_text    = None

if top_windows is None or top_windows.empty:
    display_empty_result_debug(score_df, top_windows)


# ============================================================
# OPTIONAL POSITION DATA
# ============================================================

if include_positions:
    with st.spinner("Fetching Sun/Moon position data for visualization only..."):
        try:
            position_df = cached_fetch_positions(lat=result["lat"], lon=result["lon"])
        except Exception as e:
            st.warning(f"Sun/Moon position visualization unavailable: {e}")
            position_df = pd.DataFrame()


# ============================================================
# OPTIONAL LLM
# ============================================================

if include_llm:
    with st.spinner("Generating AI explanation from fixed model results..."):
        try:
            llm_text = cached_generate_llm(result["llm_context"])
        except Exception as e:
            llm_text = f"AI recommendation failed: {e}"


# ============================================================
# HERO CARD (with score badge)
# ============================================================

best_row   = top_windows.iloc[0]
best_score = safe_numeric(best_row.get("stargazing_score"), 0)
best_time  = best_row.get("time_label", "N/A")
best_label = best_row.get("recommendation", "N/A")
score_10   = best_score / 10

st.markdown(
    f"""
    <div class="hero-card">
        <div class="hero-glow"></div>
        <div class="hero-flex">
            <div class="hero-left">
                <div class="hero-icon-wrap">🚀</div>
                <div>
                    <h1 class="hero-title">Stargazing Assistant</h1>
                    <p class="hero-desc">
                        A live decision-support tool for finding the best stargazing windows
                        using weather, astronomy, light pollution, scoring logic, and optional
                        AI explanation.
                    </p>
                </div>
            </div>
            <div class="score-badge">
                <span class="score-dot"></span>Score: {score_10:.1f}/10
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

render_source_badges(result)
st.markdown(
    '<p style="font-family:\'DM Sans\',sans-serif;font-size:13px;color:#c478d2;'
    'background:rgba(196,120,210,0.08);border:1px solid rgba(196,120,210,0.22);'
    'border-radius:8px;padding:8px 14px;margin:6px 0 0 0;display:inline-block;">'
    "&#9432;&nbsp; Score is generated only from the weather/astronomy scoring pipeline. "
    "Sun/Moon position data and AI recommendation do not change the score.</p>",
    unsafe_allow_html=True,
)
st.success(
    f"Live data loaded for {result.get('city_name', city_name)} "
    f"({result.get('lat', 0):.4f}, {result.get('lon', 0):.4f})"
)


# ============================================================
# SUMMARY METRICS
# ============================================================

seeing_val = safe_numeric(best_row.get("seeing_value"), 0)
moon_pct   = safe_numeric(best_row.get("moon_illuminated_pct"), 0)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        f"""
        <div class="metric-card">
            <span class="metric-card-title">Best Window</span>
            <div class="metric-large" style="font-size:1.45rem; line-height:1.2;">{best_time}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        f"""
        <div class="metric-card">
            <span class="metric-card-title">Seeing</span>
            <div class="metric-value-row">
                <span class="metric-large">{seeing_val:.1f}</span>
                <span class="metric-unit">&thinsp;/ 5</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        f"""
        <div class="metric-card">
            <span class="metric-card-title">Moon Phase</span>
            <div class="metric-value-row">
                <span class="metric-large">{moon_pct:.0f}</span>
                <span class="metric-unit">&thinsp;% lit</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col4:
    st.markdown(
        f"""
        <div class="metric-card">
            <span class="metric-card-title">Bortle</span>
            <div class="metric-value-row">
                <span class="metric-large">{bortle_index}</span>
                <span class="metric-unit">&thinsp;index</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()


# ============================================================
# PAGE ROUTER
# ============================================================

if selected_page == "Dashboard":

    st.markdown(
        """
        <div class="section-card">
            <h3 class="section-heading">Ready to find your best stargazing window?</h3>
            <p class="muted">
                Choose a location from the sidebar, adjust the Bortle light pollution index,
                and run the live recommendation pipeline.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Tonight's Forecast Preview
    chart_df = score_df.copy()
    chart_df["local_dt"] = pd.to_datetime(chart_df["local_dt"], errors="coerce")
    chart_df = chart_df.dropna(subset=["local_dt"])

    today    = pd.Timestamp.now().date()
    tomorrow = today + pd.Timedelta(days=1)
    tonight_mask = (
        ((chart_df["local_dt"].dt.date == today)    & (chart_df["local_dt"].dt.hour >= 18)) |
        ((chart_df["local_dt"].dt.date == tomorrow) & (chart_df["local_dt"].dt.hour <= 6))
    )
    tonight_df = chart_df[tonight_mask].copy()

    st.markdown(
        '<span class="section-label" style="margin-top:8px; display:block;">'
        "Tonight's Forecast Preview</span>",
        unsafe_allow_html=True,
    )

    if not tonight_df.empty:
        tonight_df["hour_label"] = tonight_df["local_dt"].dt.hour.map(fmt_hour)
        fig_preview = px.bar(
            tonight_df, x="hour_label", y="stargazing_score",
            color_discrete_sequence=["#c478d2"],
        )
        fig_preview.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="DM Sans, sans-serif", color="#8880a0", size=11),
            height=200,
            margin=dict(l=8, r=8, t=8, b=28),
            showlegend=False,
            xaxis=dict(showgrid=False, showline=False, title=None,
                       tickfont=dict(color="#554f6a", family="DM Sans, sans-serif")),
            yaxis=dict(showgrid=False, showline=False, visible=False, range=[0, 100]),
        )
        fig_preview.update_traces(
            marker_opacity=0.85,
            marker_line_width=0,
            marker_cornerradius=3,
        )
        st.plotly_chart(fig_preview, use_container_width=True)
    else:
        st.caption("No forecast data available for tonight's hours.")

    st.markdown(
        """
        <div class="section-card">
            <span class="section-label">What this app does</span>
            <ul style="color:#8880a0; margin:0; padding-left:18px; line-height:2.1;
                       font-family:'DM Sans',sans-serif; font-size:14px; font-weight:300;">
                <li>Fetches live weather and astronomy data</li>
                <li>Scores each forecast hour using cloud, transparency, seeing, darkness, and moon conditions</li>
                <li>Ranks the best observing windows</li>
                <li>Visualizes forecast trends, factor diagnostics, and optional Sun/Moon sky path</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Top 3 Recommended Windows")
    top3 = top_windows.head(3)
    cols = st.columns(3)
    for i, (_, row) in enumerate(top3.iterrows()):
        with cols[i]:
            render_window_card(i + 1, row)

    st.subheader("Score Trend")
    full_chart_df = score_df.copy()
    full_chart_df["local_dt"] = pd.to_datetime(full_chart_df["local_dt"], errors="coerce")
    fig_score = px.line(
        full_chart_df, x="local_dt", y="stargazing_score",
        color="recommendation", markers=True,
        title="Hourly Stargazing Score",
        hover_data=[
            c for c in [
                "cloud_value", "transparency_value", "seeing_value",
                "moon_illuminated_pct", "is_dark_enough", "is_moon_up",
                "effective_darkness", "atmospheric_score",
            ]
            if c in full_chart_df.columns
        ],
    )
    fig_score.update_layout(yaxis_range=[0, 100])
    st.plotly_chart(_themed_layout(fig_score, 480), use_container_width=True)

    dist_fig = build_recommendation_distribution(score_df)
    if dist_fig is not None:
        st.plotly_chart(dist_fig, use_container_width=True)


elif selected_page == "Forecast Timeline":
    st.subheader("Forecast Timeline")

    chart_df = score_df.copy()
    chart_df["local_dt"] = pd.to_datetime(chart_df["local_dt"], errors="coerce")

    fig_score = px.line(
        chart_df, x="local_dt", y="stargazing_score",
        color="recommendation", markers=True,
        title="Hourly Stargazing Score Forecast",
        hover_data=[
            c for c in [
                "cloud_value", "transparency_value", "seeing_value",
                "moon_illuminated_pct", "is_dark_enough", "is_moon_up",
                "visibility_penalty", "transparency_norm", "seeing_norm",
                "humidity_quality", "moon_brightness_penalty", "effective_darkness",
                "atmospheric_score",
            ]
            if c in chart_df.columns
        ],
    )
    fig_score.update_layout(yaxis_range=[0, 100])
    st.plotly_chart(_themed_layout(fig_score, 560), use_container_width=True)
    st.plotly_chart(build_score_heatmap(score_df), use_container_width=True)


elif selected_page == "Best Windows":
    st.subheader("Best Windows")

    top_cols = st.columns(3)
    for i, (_, row) in enumerate(top_windows.head(3).iterrows()):
        with top_cols[i]:
            render_window_card(i + 1, row)

    st.subheader("Top Recommended Observing Windows")
    fig_top = px.bar(
        top_windows.sort_values("stargazing_score"),
        x="stargazing_score", y="time_label", orientation="h",
        color="recommendation",
        hover_data=[
            c for c in [
                "cloud_value", "transparency_value", "seeing_value",
                "moon_illuminated_pct", "is_dark_enough", "is_moon_up",
                "visibility_penalty", "effective_darkness", "atmospheric_score",
            ]
            if c in top_windows.columns
        ],
        title="Top 10 Stargazing Windows",
    )
    fig_top.update_layout(xaxis_range=[0, 100])
    st.plotly_chart(_themed_layout(fig_top, 520), use_container_width=True)
    st.dataframe(top_windows, use_container_width=True)


elif selected_page == "Sky Conditions":
    st.subheader("Sky Conditions")
    st.plotly_chart(build_factor_chart(score_df), use_container_width=True)

    feature_options = [
        c for c in [
            "cloud_value", "transparency_value", "seeing_value",
            "moon_illuminated_pct", "visibility_penalty", "transparency_norm",
            "seeing_norm", "humidity_quality", "moon_brightness_penalty",
            "effective_darkness", "atmospheric_score", "stargazing_score",
        ]
        if c in score_df.columns
    ]

    if feature_options:
        selected_feature = st.selectbox("Inspect one feature", feature_options)
        plot_df = score_df.copy()
        plot_df["local_dt"] = pd.to_datetime(plot_df["local_dt"], errors="coerce")
        fig_feature = px.line(
            plot_df, x="local_dt", y=selected_feature,
            markers=True, title=f"{selected_feature} Over Time",
        )
        st.plotly_chart(_themed_layout(fig_feature, 480), use_container_width=True)

    st.subheader("Scoring Feature Table")
    diagnostic_cols = [
        "local_dt", "stargazing_score", "recommendation",
        "cloud_value", "transparency_value", "seeing_value",
        "visibility_penalty", "transparency_norm", "seeing_norm",
        "humidity_quality", "moon_brightness_penalty", "effective_darkness", "atmospheric_score",
    ]
    diagnostic_cols = [c for c in diagnostic_cols if c in score_df.columns]
    st.dataframe(score_df[diagnostic_cols].head(60), use_container_width=True)


elif selected_page == "Sky Path":
    st.subheader("Sky Path")
    st.caption(
        "This visualization uses Sun/Moon position data only. "
        "It does not affect the recommendation score."
    )

    if not include_positions:
        st.info("Turn on 'Use Sun/Moon position data' in the sidebar, then run again.")
    elif position_df is None or position_df.empty:
        st.warning("No Sun/Moon position data was returned.")
    else:
        required_cols = ["Azimuth (°)", "Altitude (°)", "Object"]
        if all(c in position_df.columns for c in required_cols):
            pos_plot_df = position_df.dropna(subset=["Azimuth (°)", "Altitude (°)"])
            if not pos_plot_df.empty:
                fig_pos = px.scatter_polar(
                    pos_plot_df, r="Altitude (°)", theta="Azimuth (°)", color="Object",
                    hover_data=[
                        c for c in ["Hour", "Illuminated (%)", "Moon Phase"]
                        if c in pos_plot_df.columns
                    ],
                    title="Sun and Moon Position in the Sky",
                )
                st.plotly_chart(_themed_layout(fig_pos, 650), use_container_width=True)
            else:
                st.warning("Position data exists, but altitude/azimuth values are missing.")
        else:
            st.warning(
                "Position data does not contain the required columns: "
                "Azimuth (°), Altitude (°), Object."
            )
        with st.expander("Position Data"):
            st.dataframe(position_df, use_container_width=True)


elif selected_page == "AI Insight":
    st.subheader("AI Insight")
    st.caption(
        "AI explanation is generated from fixed model outputs. It does not change the score."
    )

    if not include_llm:
        st.info("Turn on 'Generate AI recommendation' in the sidebar, then run again.")
    elif llm_text:
        st.markdown(
            f'<div class="section-card">{llm_text}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.warning("AI recommendation was enabled, but no text was returned.")


elif selected_page == "Methodology":
    st.subheader("Methodology")
    st.markdown(
        """
        <div class="section-card">
            <h3 class="section-heading">Data Sources</h3>
            <p style="color:#8880a0; font-family:'DM Sans',sans-serif; font-size:14px;">
                <b style="color:#e4dff0; font-weight:500;">Weather Forecast</b>
            </p>
            <ul style="color:#8880a0; font-family:'DM Sans',sans-serif; font-size:14px;
                       line-height:1.9; font-weight:300;">
                <li>Primary source: Astrospheric</li>
                <li>Fallback source: Open-Meteo</li>
            </ul>
            <p style="color:#8880a0; font-family:'DM Sans',sans-serif; font-size:14px;">
                <b style="color:#e4dff0; font-weight:500;">Astronomy Data</b>
            </p>
            <ul style="color:#8880a0; font-family:'DM Sans',sans-serif; font-size:14px;
                       line-height:1.9; font-weight:300;">
                <li>Primary source: IPGeolocation astronomy data</li>
                <li>Optional detailed source: Timeanddate event data</li>
                <li>Fallback: simplified astronomy assumptions</li>
            </ul>
        </div>

        <div class="section-card">
            <h3 class="section-heading">Scoring Logic</h3>
            <p style="color:#8880a0; font-family:'DM Sans',sans-serif; font-size:14px;
                      font-weight:300;">
                The stargazing score combines three major components:
            </p>
            <ol style="color:#8880a0; font-family:'DM Sans',sans-serif; font-size:14px;
                       line-height:1.9; font-weight:300;">
                <li><b style="color:#e4dff0; font-weight:500;">Visibility penalty</b> — penalizes high cloud cover and non-dark hours.</li>
                <li><b style="color:#e4dff0; font-weight:500;">Atmospheric quality</b> — combines transparency, seeing, and humidity/dew spread.</li>
                <li><b style="color:#e4dff0; font-weight:500;">Darkness quality</b> — accounts for Bortle light pollution, moon illumination, moon altitude, and whether the moon is up.</li>
            </ol>
        </div>

        <div class="section-card">
            <h3 class="section-heading">Important Design Rule</h3>
            <p style="color:#8880a0; font-family:'DM Sans',sans-serif; font-size:14px;
                      font-weight:300; line-height:1.7;">
                Sun/Moon position data is used only for visualization.
                AI recommendation is used only for natural-language explanation.
                Neither feature changes the stargazing score.
            </p>
        </div>

        <div class="section-card">
            <h3 class="section-heading">Fallback Logic</h3>
            <p style="color:#8880a0; font-family:'DM Sans',sans-serif; font-size:14px;
                      font-weight:300; line-height:1.7;">
                Astrospheric is preferred because it provides astronomy-specific sky quality data.
                If it fails or returns empty data, Open-Meteo is used as a global weather fallback.
                Open-Meteo does not provide real seeing/transparency metrics, so those values are
                approximated.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


elif selected_page == "Raw Data":
    st.subheader("Raw Data")
    st.caption("These tables are mainly for debugging and transparency.")

    with st.expander("Generated Master DataFrame"):
        st.dataframe(master_df, use_container_width=True)
    with st.expander("Scored DataFrame"):
        st.dataframe(score_df, use_container_width=True)
    with st.expander("Top Windows DataFrame"):
        st.dataframe(top_windows, use_container_width=True)
    with st.expander("Weather Data"):
        st.dataframe(weather_df, use_container_width=True)
    with st.expander("IPGeolocation / Astronomy Data"):
        st.dataframe(ip_geo_df, use_container_width=True)

    if event_df is not None and not event_df.empty:
        with st.expander("Timeanddate Event Data"):
            st.dataframe(event_df, use_container_width=True)

    if position_df is not None and not position_df.empty:
        with st.expander("Sun/Moon Position Data"):
            st.dataframe(position_df, use_container_width=True)
