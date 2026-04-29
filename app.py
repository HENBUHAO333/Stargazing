import random
import streamlit as st
import pandas as pd
import plotly.express as px

# ============================================================
# BACKEND IMPORTS
# ============================================================

try:
    from backend import (
        run_pipeline,
        CITY_PRESETS,
        generate_llm_recommendation,
        generate_rag_recommendation,
        fetch_tad_positions,
    )
    HAS_RAG_BACKEND = True

except ImportError:
    from backend import (
        run_pipeline,
        CITY_PRESETS,
        generate_llm_recommendation,
        fetch_tad_positions,
    )
    generate_rag_recommendation = None
    HAS_RAG_BACKEND = False


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Stargazing Assistant",
    page_icon="🌌",
    layout="wide",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Serif+Display&display=swap');

    :root {
        --bg: #090614;
        --sidebar: #0c0818;
        --card: #100d1e;
        --card-hover: #16122a;
        --accent: #c478d2;
        --accent-soft: rgba(196,120,210,0.15);
        --border: rgba(196,120,210,0.16);
        --text: #e4dff0;
        --text-muted: #8880a0;
        --text-dim: #554f6a;
    }

    html, body, .stApp, [data-testid="stAppViewContainer"], .main {
        background-color: var(--bg) !important;
        font-family: 'DM Sans', sans-serif;
        color: var(--text);
    }

    h1, h2, h3, h4 {
        font-family: 'DM Serif Display', serif !important;
        font-weight: 400 !important;
        color: var(--text) !important;
    }

    p, li, span, label, td, th, button, input, select, textarea {
        font-family: 'DM Sans', sans-serif !important;
    }

    [data-testid="stSidebar"] {
        background: rgba(12,8,24,0.92) !important;
        border-right: 1px solid var(--border) !important;
    }

    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span {
        color: var(--text) !important;
        font-size: 13px !important;
    }

    .hero-card {
        background: linear-gradient(135deg, var(--card) 0%, rgba(196,120,210,0.06) 100%);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 28px 32px;
        margin-bottom: 20px;
        position: relative;
        overflow: hidden;
    }

    .hero-flex {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        gap: 20px;
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
        font-family: 'DM Serif Display', serif !important;
        font-size: 2rem;
        font-weight: 400 !important;
        color: var(--text);
        line-height: 1.15;
    }

    .hero-desc {
        margin: 0;
        color: var(--text-muted);
        font-size: 13px;
        line-height: 1.67;
        max-width: 680px;
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
        font-size: 12px;
        font-weight: 600;
        white-space: nowrap;
    }

    .score-dot {
        width: 7px;
        height: 7px;
        background: var(--accent);
        border-radius: 50%;
        box-shadow: 0 0 6px var(--accent);
        display: inline-block;
    }

    .metric-card {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 14px 16px;
        min-height: 112px;
    }

    .metric-card-title {
        font-size: 10px;
        color: var(--text-dim);
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 10px;
        display: block;
    }

    .metric-large {
        font-family: 'DM Serif Display', serif;
        font-size: 2rem;
        font-weight: 400;
        color: var(--text);
        line-height: 1;
    }

    .metric-unit {
        font-size: 0.9rem;
        color: var(--text-muted);
        font-weight: 300;
    }

    .section-card {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 24px 28px;
        margin-bottom: 16px;
    }

    .section-heading {
        font-family: 'DM Serif Display', serif !important;
        font-size: 20px;
        font-weight: 400 !important;
        color: var(--text);
        margin: 0 0 8px 0;
    }

    .section-label {
        font-size: 10px;
        color: var(--text-dim);
        font-weight: 600;
        letter-spacing: 0.13em;
        text-transform: uppercase;
        margin: 0 0 14px 0;
        display: block;
    }

    .window-card {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 20px;
        min-height: 200px;
    }

    .window-rank {
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
        color: var(--text-muted);
        font-size: 13px;
        line-height: 1.6;
    }

    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 99px;
        font-size: 11px;
        font-weight: 600;
        margin-right: 6px;
        margin-top: 6px;
    }

    .badge-excellent { background: rgba(34,197,94,0.12); color: #4ade80; border: 1px solid rgba(74,222,128,0.25); }
    .badge-good { background: rgba(20,184,166,0.12); color: #5eead4; border: 1px solid rgba(94,234,212,0.25); }
    .badge-marginal { background: rgba(245,158,11,0.12); color: #fbbf24; border: 1px solid rgba(251,191,36,0.25); }
    .badge-poor { background: rgba(249,115,22,0.12); color: #fb923c; border: 1px solid rgba(251,146,60,0.25); }
    .badge-nogo { background: rgba(239,68,68,0.12); color: #f87171; border: 1px solid rgba(248,113,113,0.25); }

    .source-pill {
        background: var(--accent-soft);
        color: var(--accent);
        border: 1px solid var(--border);
        padding: 4px 10px;
        border-radius: 99px;
        font-size: 11px;
        font-weight: 600;
        margin-right: 6px;
        display: inline-block;
    }

    .warning-pill {
        background: rgba(245,158,11,0.10);
        color: #fbbf24;
        border: 1px solid rgba(251,191,36,0.22);
        padding: 4px 10px;
        border-radius: 99px;
        font-size: 11px;
        font-weight: 600;
        margin-right: 6px;
        display: inline-block;
    }

    .cluster-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 99px;
        background: rgba(99,102,241,0.12);
        color: #a5b4fc;
        border: 1px solid rgba(165,180,252,0.25);
        font-size: 11px;
        font-weight: 600;
        margin-top: 10px;
    }

    .cluster-desc {
        font-size: 12px;
        font-style: italic;
        color: #6b6585;
        margin-top: 6px;
        line-height: 1.55;
    }

    .rag-box {
        background: rgba(196,120,210,0.08);
        border: 1px solid rgba(196,120,210,0.22);
        border-radius: 12px;
        padding: 18px 22px;
        margin-top: 14px;
        color: var(--text);
        line-height: 1.7;
    }

    div[data-testid="stMetric"] {
        background: var(--card);
        padding: 14px 16px;
        border-radius: 10px;
        border: 1px solid var(--border);
    }

    [data-testid="stAlert"] {
        background: var(--card) !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# STAR FIELD
# ============================================================

def _star_field(n: int = 140) -> str:
    rng = random.Random(42)
    palette = ["#f0eaff", "#ddd0ff", "#c4a8f0", "#ffffff"]
    circles = []

    for _ in range(n):
        x = round(rng.uniform(0, 100), 1)
        y = round(rng.uniform(0, 100), 1)
        r = round(rng.uniform(0.4, 1.4), 1)
        op = round(rng.uniform(0.10, 0.70), 2)
        op2 = round(min(op + rng.uniform(0.10, 0.35), 0.92), 2)
        dur = round(rng.uniform(1.5, 4.5), 1)
        beg = round(rng.uniform(0, dur), 1)
        c = rng.choice(palette)

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
    """
    <p style="font-size:10px;color:#554f6a;font-weight:600;
    letter-spacing:0.13em;text-transform:uppercase;margin-bottom:6px;">
    Navigation</p>
    """,
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
    """
    <p style="font-size:10px;color:#554f6a;font-weight:600;
    letter-spacing:0.13em;text-transform:uppercase;margin-bottom:4px;">
    User Input</p>
    """,
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
    st.session_state.pop("pipeline_result", None)
    st.session_state.pop("pipeline_bortle", None)
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


@st.cache_data(ttl=1800, show_spinner=False)
def cached_generate_rag(context, user_question):
    if generate_rag_recommendation is None:
        return (
            "RAG backend function not found. Please add "
            "`generate_rag_recommendation()` to backend.py first."
        )

    return generate_rag_recommendation(context, user_question=user_question)


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

    cloud = safe_numeric(row.get("cloud_value"), 100)
    transparency = safe_numeric(row.get("transparency_value"), 5)
    seeing = safe_numeric(row.get("seeing_value"), 5)
    moon = safe_numeric(row.get("moon_illuminated_pct"), 50)

    is_dark = bool(row.get("is_dark_enough", False))
    is_moon_up = bool(row.get("is_moon_up", False))

    if cloud is not None:
        if cloud <= 30:
            reasons.append("low cloud cover")
        elif cloud <= 60:
            reasons.append("moderate cloud cover")
        else:
            reasons.append("high cloud cover")

    if transparency is not None:
        if transparency <= 2:
            reasons.append("strong transparency")
        elif transparency <= 4:
            reasons.append("acceptable transparency")
        else:
            reasons.append("weak transparency")

    if seeing is not None:
        if seeing <= 2:
            reasons.append("stable seeing")
        elif seeing <= 4:
            reasons.append("moderate seeing")
        else:
            reasons.append("poor seeing")

    if moon is not None:
        if moon <= 30:
            reasons.append("low moon illumination")
        elif moon <= 70:
            reasons.append("moderate moon illumination")
        else:
            reasons.append("bright moon")

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
            "local_dt",
            "date",
            "stargazing_score",
            "recommendation",
            "cloud_value",
            "transparency_value",
            "seeing_value",
            "moon_illuminated_pct",
            "is_dark_enough",
            "is_moon_up",
            "visibility_penalty",
            "transparency_norm",
            "seeing_norm",
            "humidity_quality",
            "moon_brightness_penalty",
            "effective_darkness",
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
        index="date_str",
        columns="hour",
        values="stargazing_score",
        aggfunc="mean",
    )

    fig = px.imshow(
        pivot,
        text_auto=".0f",
        aspect="auto",
        title="Stargazing Score Heatmap by Date and Hour",
        labels=dict(x="Hour of Day", y="Date", color="Score"),
        zmin=0,
        zmax=100,
        color_continuous_scale=[
            [0, "#100d1e"],
            [0.5, "#6b2f7a"],
            [1, "#c478d2"],
        ],
    )

    return _themed_layout(fig, 440)


def build_factor_chart(score_df):
    factor_cols = [
        "cloud_value",
        "transparency_value",
        "seeing_value",
        "moon_illuminated_pct",
        "effective_darkness",
        "atmospheric_score",
        "visibility_penalty",
    ]

    existing_cols = [c for c in factor_cols if c in score_df.columns]

    plot_df = score_df.copy()
    plot_df["local_dt"] = pd.to_datetime(plot_df["local_dt"], errors="coerce")
    plot_df = plot_df.dropna(subset=["local_dt"])

    long_df = plot_df.melt(
        id_vars=["local_dt"],
        value_vars=existing_cols,
        var_name="factor",
        value_name="value",
    )

    fig = px.line(
        long_df,
        x="local_dt",
        y="value",
        color="factor",
        title="Key Stargazing Factors Over Time",
    )

    return _themed_layout(fig, 520)


def build_recommendation_distribution(score_df):
    if "recommendation" not in score_df.columns:
        return None

    dist = score_df["recommendation"].value_counts().reset_index()
    dist.columns = ["recommendation", "count"]

    fig = px.bar(
        dist,
        x="recommendation",
        y="count",
        color="recommendation",
        title="Distribution of Forecasted Stargazing Quality",
        color_discrete_sequence=[
            "#c478d2",
            "#9b5ab0",
            "#7a3d8c",
            "#5c2d6b",
            "#3e1e4a",
        ],
    )

    return _themed_layout(fig, 420)


def render_source_badges(result):
    weather_source = result.get("weather_source", "Unknown")
    astronomy_source = result.get("astronomy_source", "Unknown")
    timezone_value = result.get("timezone", "Unknown")

    w_cls = "warning-pill" if "fallback" in weather_source.lower() else "source-pill"
    a_cls = "warning-pill" if "fallback" in astronomy_source.lower() else "source-pill"

    st.markdown(
        f'<span class="{w_cls}">Weather: {weather_source}</span>'
        f'<span class="{a_cls}">Astronomy: {astronomy_source}</span>'
        f'<span class="source-pill">Timezone: {timezone_value}</span>',
        unsafe_allow_html=True,
    )


def render_window_card(rank, row):
    score = safe_numeric(row.get("stargazing_score"), 0)
    label = row.get("recommendation", "N/A")
    time_label = row.get("time_label", "N/A")
    explanation = explain_row(row)

    cluster_label = row.get("cluster_label", "")
    cluster_desc = row.get("cluster_description", "")

    cluster_html = ""

    if cluster_label and cluster_label != "Unknown":
        cluster_html = (
            f'<div><span class="cluster-badge">{cluster_label}</span></div>'
            f'<p class="cluster-desc">{cluster_desc}</p>'
        )

    st.markdown(
        f"""
        <div class="window-card">
            <div class="window-rank">#{rank} Recommended Window</div>
            <div class="window-time">{time_label}</div>
            <div class="window-score">{score:.1f}</div>
            <span class="badge {badge_class(label)}">{label}</span>
            <p class="muted" style="margin-top:12px;">{explanation}</p>
            {cluster_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# RUN PIPELINE
# ============================================================

if run_button:
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

            st.session_state["pipeline_result"] = result
            st.session_state["pipeline_bortle"] = bortle_index

        except Exception as e:
            st.error("The live scoring pipeline failed.")
            st.exception(e)
            st.stop()


if "pipeline_result" not in st.session_state:
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-flex">
                <div class="hero-left">
                    <div class="hero-icon-wrap">🚀</div>
                    <div>
                        <h1 class="hero-title">Stargazing Assistant</h1>
                        <p class="hero-desc">
                            A live decision-support tool for finding the best stargazing windows
                            using weather, astronomy, light pollution, scoring logic, and optional
                            AI / RAG explanation.
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
                Choose a location from the sidebar, adjust the city lights index,
                and run the live recommendation pipeline.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.stop()


result = st.session_state["pipeline_result"]
bortle_index = st.session_state.get("pipeline_bortle", bortle_index)


# ============================================================
# LOAD RESULTS
# ============================================================

score_df = result.get("score_df", pd.DataFrame())
top_windows = result.get("top_windows", pd.DataFrame())
daily_summary = result.get("daily_summary", pd.DataFrame())
master_df = result.get("master_df", pd.DataFrame())
weather_df = result.get("weather_df", pd.DataFrame())
ip_geo_df = result.get("ip_geo_df", pd.DataFrame())
event_df = result.get("event_df", pd.DataFrame())

position_df = pd.DataFrame()

if top_windows is None or top_windows.empty:
    display_empty_result_debug(score_df, top_windows)


# ============================================================
# OPTIONAL POSITION DATA
# ============================================================

if include_positions:
    with st.spinner("Fetching Sun/Moon position data for visualization only..."):
        try:
            position_df = cached_fetch_positions(
                lat=result["lat"],
                lon=result["lon"],
            )

        except Exception as e:
            st.warning(f"Sun/Moon position visualization unavailable: {e}")
            position_df = pd.DataFrame()


# ============================================================
# HERO CARD
# ============================================================

best_row = top_windows.iloc[0]
best_score = safe_numeric(best_row.get("stargazing_score"), 0)
best_time = best_row.get("time_label", "N/A")
best_label = best_row.get("recommendation", "N/A")
score_10 = best_score / 10

st.markdown(
    f"""
    <div class="hero-card">
        <div class="hero-flex">
            <div class="hero-left">
                <div class="hero-icon-wrap">🚀</div>
                <div>
                    <h1 class="hero-title">Stargazing Assistant</h1>
                    <p class="hero-desc">
                        A live decision-support tool for finding the best stargazing windows
                        using weather, astronomy, light pollution, scoring logic, clustering,
                        and optional AI/RAG explanation.
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
    """
    <p style="font-size:13px;color:#c478d2;
    background:rgba(196,120,210,0.08);border:1px solid rgba(196,120,210,0.22);
    border-radius:8px;padding:8px 14px;margin:6px 0 0 0;display:inline-block;">
    &#9432;&nbsp; Score is generated only from the weather/astronomy scoring pipeline.
    Sun/Moon position data, clustering, and AI/RAG explanation do not change the score.</p>
    """,
    unsafe_allow_html=True,
)

st.success(
    f"Live data loaded for {result.get('city_name', city_name)} "
    f"({result.get('lat', 0):.4f}, {result.get('lon', 0):.4f})"
)


# ============================================================
# SUMMARY METRICS
# ============================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        f"""
        <div class="metric-card">
            <span class="metric-card-title">Best Score</span>
            <span class="metric-large">{best_score:.1f}</span>
            <span class="metric-unit">/100</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        f"""
        <div class="metric-card">
            <span class="metric-card-title">Best Time</span>
            <div class="metric-large" style="font-size:1.45rem;">{best_time}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        f"""
        <div class="metric-card">
            <span class="metric-card-title">Recommendation</span>
            <span class="badge {badge_class(best_label)}">{best_label}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col4:
    st.markdown(
        f"""
        <div class="metric-card">
            <span class="metric-card-title">Forecast Hours</span>
            <span class="metric-large">{len(score_df)}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()


# ============================================================
# PAGE ROUTER
# ============================================================

if selected_page == "Dashboard":
    st.subheader("Dashboard")

    st.markdown(
        f"""
        <div class="section-card">
            <h3 class="section-heading">Best Observing Window</h3>
            <p class="muted">
                The best current observing window is <b style="color:#e4dff0;">{best_time}</b>,
                with a score of <b style="color:#e4dff0;">{best_score:.1f}/100</b>.
            </p>
            <span class="badge {badge_class(best_label)}">{best_label}</span>
            <p class="muted" style="margin-top:12px;">{explain_row(best_row)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Top 3 Recommended Windows")

    cols = st.columns(3)

    for i, (_, row) in enumerate(top_windows.head(3).iterrows()):
        with cols[i]:
            render_window_card(i + 1, row)

    st.subheader("Score Trend Preview")

    chart_df = score_df.copy()
    chart_df["local_dt"] = pd.to_datetime(chart_df["local_dt"], errors="coerce")

    fig_score = px.line(
        chart_df,
        x="local_dt",
        y="stargazing_score",
        color="recommendation",
        markers=True,
        title="Hourly Stargazing Score",
        hover_data=[
            c for c in [
                "cloud_value",
                "transparency_value",
                "seeing_value",
                "moon_illuminated_pct",
                "is_dark_enough",
                "is_moon_up",
                "effective_darkness",
                "atmospheric_score",
            ]
            if c in chart_df.columns
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
        chart_df,
        x="local_dt",
        y="stargazing_score",
        color="recommendation",
        markers=True,
        title="Hourly Stargazing Score Forecast",
        hover_data=[
            c for c in [
                "cloud_value",
                "transparency_value",
                "seeing_value",
                "moon_illuminated_pct",
                "is_dark_enough",
                "is_moon_up",
                "visibility_penalty",
                "transparency_norm",
                "seeing_norm",
                "humidity_quality",
                "moon_brightness_penalty",
                "effective_darkness",
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

    cols = st.columns(3)

    for i, (_, row) in enumerate(top_windows.head(3).iterrows()):
        with cols[i]:
            render_window_card(i + 1, row)

    st.subheader("Top Recommended Observing Windows")

    fig_top = px.bar(
        top_windows.sort_values("stargazing_score"),
        x="stargazing_score",
        y="time_label",
        orientation="h",
        color="recommendation",
        hover_data=[
            c for c in [
                "cloud_value",
                "transparency_value",
                "seeing_value",
                "moon_illuminated_pct",
                "is_dark_enough",
                "is_moon_up",
                "visibility_penalty",
                "effective_darkness",
                "atmospheric_score",
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
            "cloud_value",
            "transparency_value",
            "seeing_value",
            "moon_illuminated_pct",
            "visibility_penalty",
            "transparency_norm",
            "seeing_norm",
            "humidity_quality",
            "moon_brightness_penalty",
            "effective_darkness",
            "atmospheric_score",
            "stargazing_score",
        ]
        if c in score_df.columns
    ]

    if feature_options:
        selected_feature = st.selectbox("Inspect one feature", feature_options)

        plot_df = score_df.copy()
        plot_df["local_dt"] = pd.to_datetime(plot_df["local_dt"], errors="coerce")

        fig_feature = px.line(
            plot_df,
            x="local_dt",
            y=selected_feature,
            markers=True,
            title=f"{selected_feature} Over Time",
        )

        st.plotly_chart(_themed_layout(fig_feature, 480), use_container_width=True)

    st.subheader("Scoring Feature Table")

    diagnostic_cols = [
        "local_dt",
        "stargazing_score",
        "recommendation",
        "cloud_value",
        "transparency_value",
        "seeing_value",
        "visibility_penalty",
        "transparency_norm",
        "seeing_norm",
        "humidity_quality",
        "moon_brightness_penalty",
        "effective_darkness",
        "atmospheric_score",
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
            pos_plot_df = position_df.dropna(
                subset=["Azimuth (°)", "Altitude (°)"]
            )

            if not pos_plot_df.empty:
                fig_pos = px.scatter_polar(
                    pos_plot_df,
                    r="Altitude (°)",
                    theta="Azimuth (°)",
                    color="Object",
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
        "AI explanation is generated from fixed model outputs. "
        "RAG uses a local stargazing knowledge base to improve explanation quality. "
        "It does not change the score."
    )

    st.markdown(
        """
        <div class="section-card">
            <h3 class="section-heading">Ask the Stargazing Assistant</h3>
            <p class="muted">
                Use this page to ask why a window is recommended, what objects to observe,
                or how moonlight, clouds, transparency, and city lights affect the result.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    user_question = st.text_area(
        "Ask a stargazing question",
        value="Based on the current forecast, what should I observe and why?",
        height=110,
    )

    use_rag = st.checkbox(
        "Use local stargazing knowledge base",
        value=True,
        help=(
            "Retrieves explanations about sky darkness, cloud cover, moonlight, "
            "seeing, transparency, and data limitations."
        ),
    )

    if not include_llm:
        st.info("Turn on 'Generate AI recommendation' in the sidebar, then run again.")

    else:
        col_a, col_b = st.columns([1, 1])

        with col_a:
            generate_button = st.button(
                "Generate AI Insight",
                use_container_width=True,
                type="primary",
            )

        with col_b:
            st.caption(
                "RAG is on" if use_rag else "Standard LLM explanation only"
            )

        if generate_button:
            with st.spinner("Generating grounded AI insight..."):
                if use_rag:
                    answer = cached_generate_rag(
                        result["llm_context"],
                        user_question,
                    )
                else:
                    answer = cached_generate_llm(
                        result["llm_context"],
                    )

            st.markdown('<div class="rag-box">', unsafe_allow_html=True)
            st.markdown(answer)
            st.markdown('</div>', unsafe_allow_html=True)
        if use_rag and not HAS_RAG_BACKEND:
            st.warning(
                "RAG UI is ready, but backend.py does not yet have "
                "`generate_rag_recommendation()`. Add the backend RAG function next."
            )


elif selected_page == "Methodology":
    st.subheader("Methodology")

    st.markdown(
        """
        <div class="section-card">
            <h3 class="section-heading">Data Sources</h3>
            <ul class="muted">
                <li>Primary weather source: Astrospheric</li>
                <li>Fallback weather source: Open-Meteo</li>
                <li>Primary astronomy source: IPGeolocation</li>
                <li>Optional detailed event source: Timeanddate</li>
            </ul>
        </div>

        <div class="section-card">
            <h3 class="section-heading">Scoring Logic</h3>
            <p class="muted">
                The score combines visibility constraints, atmospheric quality,
                and darkness quality. Cloud cover and daylight act as hard penalties,
                while transparency, seeing, humidity, moon illumination, moon altitude,
                and city lights influence the final score.
            </p>
        </div>

        <div class="section-card">
            <h3 class="section-heading">AI / RAG Layer</h3>
            <p class="muted">
                The AI Insight page does not recalculate the score.
                It only explains the fixed model output.
                When RAG is enabled, the model retrieves local stargazing knowledge
                about sky darkness, moonlight, cloud cover, transparency, seeing,
                and data limitations before generating the explanation.
            </p>
        </div>

        <div class="section-card">
            <h3 class="section-heading">Important Design Rule</h3>
            <p class="muted">
                Sun/Moon position data, clustering, and AI/RAG explanations are
                interpretation layers. They do not change the stargazing score.
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
