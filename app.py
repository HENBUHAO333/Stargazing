import streamlit as st
import pandas as pd
import plotly.express as px

try:
    from streamlit_option_menu import option_menu
except Exception:
    option_menu = None

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
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
    .main {
        background-color: #0b0f19;
    }

    .hero-card {
        background: linear-gradient(135deg, #111827 0%, #1f2937 100%);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 22px;
        padding: 28px;
        margin-bottom: 24px;
        box-shadow: 0 12px 32px rgba(0,0,0,0.28);
    }

    .metric-card {
        background: linear-gradient(135deg, #111827 0%, #182235 100%);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px;
        padding: 22px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.22);
        min-height: 130px;
    }

    .metric-title {
        font-size: 0.88rem;
        color: #9ca3af;
        margin-bottom: 8px;
        font-weight: 600;
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        color: #f9fafb;
        line-height: 1.15;
    }

    .metric-sub {
        font-size: 0.82rem;
        color: #9ca3af;
        margin-top: 8px;
    }

    .section-card {
        background-color: #111827;
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px;
        padding: 24px;
        margin-bottom: 22px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.18);
    }

    .window-card {
        background: linear-gradient(135deg, #101827 0%, #172033 100%);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px;
        padding: 20px;
        min-height: 210px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.2);
    }

    .window-rank {
        color: #60a5fa;
        font-weight: 800;
        font-size: 0.9rem;
        margin-bottom: 8px;
    }

    .window-time {
        color: #f9fafb;
        font-weight: 800;
        font-size: 1.2rem;
        margin-bottom: 8px;
    }

    .window-score {
        font-size: 2rem;
        font-weight: 900;
        color: #f9fafb;
    }

    .muted {
        color: #9ca3af;
        font-size: 0.9rem;
    }

    .badge {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 800;
        margin-right: 6px;
        margin-top: 6px;
    }

    .badge-excellent {
        background-color: rgba(34,197,94,0.18);
        color: #4ade80;
        border: 1px solid rgba(74,222,128,0.35);
    }

    .badge-good {
        background-color: rgba(20,184,166,0.18);
        color: #5eead4;
        border: 1px solid rgba(94,234,212,0.35);
    }

    .badge-marginal {
        background-color: rgba(245,158,11,0.18);
        color: #fbbf24;
        border: 1px solid rgba(251,191,36,0.35);
    }

    .badge-poor {
        background-color: rgba(249,115,22,0.18);
        color: #fb923c;
        border: 1px solid rgba(251,146,60,0.35);
    }

    .badge-nogo {
        background-color: rgba(239,68,68,0.18);
        color: #f87171;
        border: 1px solid rgba(248,113,113,0.35);
    }

    .source-pill {
        background-color: rgba(96,165,250,0.12);
        color: #93c5fd;
        border: 1px solid rgba(147,197,253,0.25);
        padding: 6px 10px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 700;
        margin-right: 6px;
        display: inline-block;
    }

    .warning-pill {
        background-color: rgba(245,158,11,0.12);
        color: #fbbf24;
        border: 1px solid rgba(251,191,36,0.25);
        padding: 6px 10px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 700;
        margin-right: 6px;
        display: inline-block;
    }

    div[data-testid="stMetric"] {
        background: #111827;
        padding: 16px;
        border-radius: 16px;
        border: 1px solid rgba(255,255,255,0.08);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# TITLE
# ============================================================

st.markdown(
    """
    <div class="hero-card">
        <h1 style="margin-bottom: 8px;">🌌 Stargazing Assistant</h1>
        <p class="muted" style="font-size: 1rem;">
            A live decision-support product for finding the best stargazing windows using weather,
            astronomy, light pollution, scoring logic, and optional AI explanation.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR INPUT
# ============================================================

st.sidebar.header("Controls")

if option_menu is not None:
    with st.sidebar:
        selected_page = option_menu(
            menu_title="Navigation",
            options=[
                "Dashboard",
                "Forecast Timeline",
                "Best Windows",
                "Sky Conditions",
                "Sky Path",
                "AI Insight",
                "Methodology",
                "Raw Data",
            ],
            icons=[
                "speedometer2",
                "graph-up",
                "clock-history",
                "cloud-moon",
                "moon-stars",
                "robot",
                "info-circle",
                "database",
            ],
            default_index=0,
        )
else:
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
    )

st.sidebar.divider()
st.sidebar.subheader("User Input")

input_mode = st.sidebar.radio(
    "Choose location input mode",
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

    lat = st.sidebar.number_input(
        "Latitude",
        value=40.7128,
        format="%.6f",
    )

    lon = st.sidebar.number_input(
        "Longitude",
        value=-74.0060,
        format="%.6f",
    )

    timezone = st.sidebar.text_input(
        "Timezone",
        "America/New_York",
    )

days = st.sidebar.slider(
    "Forecast range",
    min_value=1,
    max_value=4,
    value=4,
)

bortle_index = st.sidebar.slider(
    "Bortle light pollution index",
    min_value=1,
    max_value=9,
    value=5,
    help="1 = very dark rural sky; 9 = heavily light-polluted urban sky.",
)

include_tad = st.sidebar.checkbox(
    "Use Timeanddate event data",
    value=True,
    help=(
        "Adds detailed moon/twilight event features. This can affect the score."
    ),
)

include_positions = st.sidebar.checkbox(
    "Use Sun/Moon position data",
    value=False,
    help=(
        "Only used for Sky Path visualization. This does not affect the score."
    ),
)

include_llm = st.sidebar.checkbox(
    "Generate AI recommendation",
    value=False,
    help=(
        "Only explains fixed model output. This does not affect the score."
    ),
)

run_button = st.sidebar.button("Find Stargazing Windows", use_container_width=True)

if st.sidebar.button("Clear cached data", use_container_width=True):
    st.cache_data.clear()
    st.sidebar.success("Cache cleared. Run the pipeline again.")


# ============================================================
# CACHE FUNCTIONS
# ============================================================

@st.cache_data(ttl=1800, show_spinner=False)
def cached_run_pipeline(
    city_name,
    lat,
    lon,
    timezone,
    days,
    bortle_index,
    include_tad,
):
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

    if is_dark:
        reasons.append("dark enough sky")
    else:
        reasons.append("not fully dark")

    if is_moon_up:
        reasons.append("moon may interfere")
    else:
        reasons.append("moon is not a major issue")

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
    )

    fig.update_layout(height=440)
    return fig


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

    fig.update_layout(height=520)
    return fig


def build_recommendation_distribution(score_df):
    if "recommendation" not in score_df.columns:
        return None

    dist = (
        score_df["recommendation"]
        .value_counts()
        .reset_index()
    )

    dist.columns = ["recommendation", "count"]

    fig = px.bar(
        dist,
        x="recommendation",
        y="count",
        color="recommendation",
        title="Distribution of Forecasted Stargazing Quality",
    )

    fig.update_layout(height=420)
    return fig


def render_source_badges(result):
    weather_source = result.get("weather_source", "Unknown")
    astronomy_source = result.get("astronomy_source", "Unknown")
    timezone_value = result.get("timezone", "Unknown")

    weather_class = "warning-pill" if "fallback" in weather_source.lower() else "source-pill"
    astronomy_class = "warning-pill" if "fallback" in astronomy_source.lower() else "source-pill"

    st.markdown(
        f"""
        <span class="{weather_class}">Weather: {weather_source}</span>
        <span class="{astronomy_class}">Astronomy: {astronomy_source}</span>
        <span class="source-pill">Timezone: {timezone_value}</span>
        """,
        unsafe_allow_html=True,
    )


def render_window_card(rank, row):
    score = safe_numeric(row.get("stargazing_score"), 0)
    label = row.get("recommendation", "N/A")
    time_label = row.get("time_label", "N/A")
    explanation = explain_row(row)

    st.markdown(
        f"""
        <div class="window-card">
            <div class="window-rank">#{rank} Recommended Window</div>
            <div class="window-time">{time_label}</div>
            <div class="window-score">{score:.1f}</div>
            <span class="badge {badge_class(label)}">{label}</span>
            <p class="muted" style="margin-top: 12px;">{explanation}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# INITIAL STATE
# ============================================================

if not run_button:
    st.markdown(
        """
        <div class="section-card">
            <h3>🌠 Ready to find your best stargazing window?</h3>
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
            <h4>What this app does</h4>
            <ul>
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
# RUN FIXED SCORING PIPELINE
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

score_df = result.get("score_df", pd.DataFrame())
top_windows = result.get("top_windows", pd.DataFrame())
daily_summary = result.get("daily_summary", pd.DataFrame())
master_df = result.get("master_df", pd.DataFrame())
weather_df = result.get("weather_df", pd.DataFrame())
ip_geo_df = result.get("ip_geo_df", pd.DataFrame())
event_df = result.get("event_df", pd.DataFrame())

position_df = pd.DataFrame()
llm_text = None

st.success(
    f"Live data loaded for {result.get('city_name', city_name)} "
    f"({result.get('lat', 0):.4f}, {result.get('lon', 0):.4f})"
)

render_source_badges(result)

st.caption(
    "Score is generated only from the weather/astronomy scoring pipeline. "
    "Sun/Moon position data and AI recommendation do not change the score."
)

if top_windows is None or top_windows.empty:
    display_empty_result_debug(score_df, top_windows)


# ============================================================
# OPTIONAL POSITION DATA — VISUALIZATION ONLY
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
# OPTIONAL LLM — EXPLANATION ONLY
# ============================================================

if include_llm:
    with st.spinner("Generating AI explanation from fixed model results..."):
        try:
            llm_text = cached_generate_llm(result["llm_context"])

        except Exception as e:
            llm_text = f"AI recommendation failed: {e}"


# ============================================================
# SUMMARY METRICS
# ============================================================

best_row = top_windows.iloc[0]

best_score = safe_numeric(best_row.get("stargazing_score"), 0)
best_time = best_row.get("time_label", "N/A")
best_label = best_row.get("recommendation", "N/A")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Best Score</div>
            <div class="metric-value">{best_score:.1f}/100</div>
            <div class="metric-sub">Highest forecast score</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Best Time</div>
            <div class="metric-value" style="font-size:1.35rem;">{best_time}</div>
            <div class="metric-sub">Local time</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Recommendation</div>
            <span class="badge {badge_class(best_label)}">{best_label}</span>
            <div class="metric-sub">Model label</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col4:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Forecast Hours</div>
            <div class="metric-value">{len(score_df)}</div>
            <div class="metric-sub">Hourly forecast rows</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()


# ============================================================
# PAGE ROUTER
# ============================================================

if selected_page == "Dashboard":
    st.subheader("🌌 Dashboard")

    st.markdown(
        f"""
        <div class="section-card">
            <h3>Best Observing Window</h3>
            <p>
                The best current observing window is <b>{best_time}</b>,
                with a score of <b>{best_score:.1f}/100</b>.
            </p>
            <span class="badge {badge_class(best_label)}">{best_label}</span>
            <p class="muted" style="margin-top: 12px;">{explain_row(best_row)}</p>
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

    fig_score.update_layout(yaxis_range=[0, 100], height=480)
    st.plotly_chart(fig_score, use_container_width=True)

    dist_fig = build_recommendation_distribution(score_df)
    if dist_fig is not None:
        st.plotly_chart(dist_fig, use_container_width=True)


elif selected_page == "Forecast Timeline":
    st.subheader("📈 Forecast Timeline")

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

    fig_score.update_layout(yaxis_range=[0, 100], height=560)
    st.plotly_chart(fig_score, use_container_width=True)

    st.plotly_chart(build_score_heatmap(score_df), use_container_width=True)


elif selected_page == "Best Windows":
    st.subheader("🕒 Best Windows")

    top_cols = st.columns(3)

    for i, (_, row) in enumerate(top_windows.head(3).iterrows()):
        with top_cols[i]:
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

    fig_top.update_layout(height=520, xaxis_range=[0, 100])
    st.plotly_chart(fig_top, use_container_width=True)

    st.dataframe(top_windows, use_container_width=True)


elif selected_page == "Sky Conditions":
    st.subheader("☁️ Sky Conditions")

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

        fig_feature.update_layout(height=480)
        st.plotly_chart(fig_feature, use_container_width=True)

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
    st.subheader("🌙 Sky Path")

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

                fig_pos.update_layout(height=650)
                st.plotly_chart(fig_pos, use_container_width=True)

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
    st.subheader("🤖 AI Insight")

    st.caption(
        "AI explanation is generated from fixed model outputs. It does not change the score."
    )

    if not include_llm:
        st.info("Turn on 'Generate AI recommendation' in the sidebar, then run again.")

    elif llm_text:
        st.markdown(
            f"""
            <div class="section-card">
                {llm_text}
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:
        st.warning("AI recommendation was enabled, but no text was returned.")


elif selected_page == "Methodology":
    st.subheader("ℹ️ Methodology")

    st.markdown(
        """
        <div class="section-card">
            <h3>Data Sources</h3>
            <p><b>Weather Forecast</b></p>
            <ul>
                <li>Primary source: Astrospheric</li>
                <li>Fallback source: Open-Meteo</li>
            </ul>
            <p><b>Astronomy Data</b></p>
            <ul>
                <li>Primary source: IPGeolocation astronomy data</li>
                <li>Optional detailed source: Timeanddate event data</li>
                <li>Fallback: simplified astronomy assumptions</li>
            </ul>
        </div>

        <div class="section-card">
            <h3>Scoring Logic</h3>
            <p>The stargazing score combines three major components:</p>
            <ol>
                <li><b>Visibility penalty</b>: penalizes high cloud cover and non-dark hours.</li>
                <li><b>Atmospheric quality</b>: combines transparency, seeing, and humidity/dew spread.</li>
                <li><b>Darkness quality</b>: accounts for Bortle light pollution, moon illumination, moon altitude, and whether the moon is up.</li>
            </ol>
        </div>

        <div class="section-card">
            <h3>Important Design Rule</h3>
            <p>
                Sun/Moon position data is used only for visualization.
                AI recommendation is used only for natural-language explanation.
                Neither feature changes the stargazing score.
            </p>
        </div>

        <div class="section-card">
            <h3>Fallback Logic</h3>
            <p>
                Astrospheric is preferred because it provides astronomy-specific sky quality data.
                If it fails or returns empty data, Open-Meteo is used as a global weather fallback.
                However, Open-Meteo does not provide real seeing/transparency metrics, so those values are approximated.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


elif selected_page == "Raw Data":
    st.subheader("🗄️ Raw Data")

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
