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

st.title("🌌 Stargazing Assistant")
st.caption(
    "A live decision-support tool for finding the best stargazing windows "
    "using weather, astronomy, scoring logic, and optional AI explanation."
)


# ============================================================
# SIDEBAR USER INPUT
# ============================================================

st.sidebar.header("User Input")

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
    help=(
        "1 = very dark rural sky; 9 = heavily light-polluted urban sky. "
        "This affects the darkness component of the score."
    ),
)

include_tad = st.sidebar.checkbox(
    "Use Timeanddate event data",
    value=True,
    help=(
        "Uses detailed Sun/Moon event data such as moon illumination, "
        "moon meridian, civil twilight, and nautical twilight. "
        "This can affect the score because it improves astronomy features."
    ),
)

include_positions = st.sidebar.checkbox(
    "Use Sun/Moon position data",
    value=False,
    help=(
        "Only used for the Sky Path visualization. "
        "This does NOT affect the stargazing score."
    ),
)

include_llm = st.sidebar.checkbox(
    "Generate AI recommendation",
    value=False,
    help=(
        "Generates a natural-language explanation from the fixed model output. "
        "This does NOT affect the score."
    ),
)

run_button = st.sidebar.button("Find Stargazing Windows")


# ============================================================
# CACHED PIPELINE
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
    """
    Cached scoring pipeline.

    Important:
    - include_positions is forced False because position data is visualization only.
    - include_llm is forced False because LLM should never affect scoring.
    """
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
    """
    Cached Sun/Moon position fetch.

    This is separate from scoring and used only for visualization.
    """
    return fetch_tad_positions(lat=lat, lon=lon)


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


def explain_row(row):
    """
    Simple rule-based explanation for one recommended window.
    """
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
            reasons.append("strong atmospheric transparency")
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
    st.warning(
        "No recommended stargazing windows were found for this location and setting."
    )

    st.write("This usually means one of these happened:")
    st.write(
        "- The score column is missing or all NaN\n"
        "- The forecast conditions are too poor\n"
        "- The API returned data in a different shape than expected\n"
        "- The scoring logic is too strict"
    )

    st.write("Debug info:")
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


# ============================================================
# MAIN APP
# ============================================================

if not run_button:
    st.info("Use the sidebar to choose a location and run the live recommendation pipeline.")
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

st.caption(
    f"Weather source: {result.get('weather_source', 'Unknown')} | "
    f"Astronomy source: {result.get('astronomy_source', 'Unknown')} | "
    f"Timezone: {result.get('timezone', 'Unknown')}"
)

st.caption(
    "Score is generated only from the weather/astronomy scoring pipeline. "
    "Sun/Moon position data and AI recommendation do not change the score."
)


# ============================================================
# EMPTY RESULT PROTECTION
# ============================================================

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
            llm_text = generate_llm_recommendation(result["llm_context"])

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

col1.metric("Best Score", f"{best_score:.1f}/100")
col2.metric("Best Time", str(best_time))
col3.metric("Recommendation", str(best_label))
col4.metric("Forecast Hours", len(score_df))

st.divider()


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    [
        "Overview",
        "Top Windows",
        "Hourly Score",
        "Daily Summary",
        "Feature Diagnostics",
        "Sky Path",
        "Raw Data",
    ]
)


# ============================================================
# TAB 1 — OVERVIEW
# ============================================================

with tab1:
    st.subheader("Overall Recommendation")

    st.markdown(
        f"""
        **Best observing window:** {best_time}  
        **Score:** {best_score:.1f}/100  
        **Rating:** {best_label}  
        **Main explanation:** {explain_row(best_row)}
        """
    )

    if include_llm:
        st.subheader("AI Recommendation")

        if llm_text:
            st.write(llm_text)
        else:
            st.info("AI recommendation was enabled, but no text was returned.")

    st.subheader("Top 5 Windows Preview")

    preview = top_windows.head(5).copy()
    preview["Explanation"] = preview.apply(explain_row, axis=1)

    preview_cols = [
        "time_label",
        "stargazing_score",
        "recommendation",
        "cloud_value",
        "transparency_value",
        "seeing_value",
        "moon_illuminated_pct",
        "is_dark_enough",
        "is_moon_up",
        "Explanation",
    ]

    preview_cols = [c for c in preview_cols if c in preview.columns]

    st.dataframe(preview[preview_cols], use_container_width=True)


# ============================================================
# TAB 2 — TOP WINDOWS
# ============================================================

with tab2:
    st.subheader("Top Recommended Stargazing Windows")

    fig_top = px.bar(
        top_windows.sort_values("stargazing_score"),
        x="stargazing_score",
        y="time_label",
        orientation="h",
        color="recommendation",
        hover_data=[
            c
            for c in [
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

    fig_top.update_layout(
        xaxis_title="Stargazing Score",
        yaxis_title="Time Window",
        height=500,
    )

    st.plotly_chart(fig_top, use_container_width=True)

    st.dataframe(top_windows, use_container_width=True)


# ============================================================
# TAB 3 — HOURLY SCORE
# ============================================================

with tab3:
    st.subheader("Hourly Stargazing Score")

    if score_df.empty:
        st.warning("No scored forecast data available.")

    elif "stargazing_score" not in score_df.columns:
        st.warning("The score_df does not contain stargazing_score.")

    else:
        chart_df = score_df.copy()

        if "local_dt" in chart_df.columns:
            chart_df["local_dt"] = pd.to_datetime(chart_df["local_dt"], errors="coerce")
            x_col = "local_dt"
        else:
            chart_df["row_index"] = range(len(chart_df))
            x_col = "row_index"

        hover_cols = [
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

        hover_cols = [c for c in hover_cols if c in chart_df.columns]

        color_col = "recommendation" if "recommendation" in chart_df.columns else None

        fig_score = px.line(
            chart_df,
            x=x_col,
            y="stargazing_score",
            color=color_col,
            markers=True,
            hover_data=hover_cols,
            title="Hourly Stargazing Score Forecast",
        )

        fig_score.update_layout(
            xaxis_title="Local Time",
            yaxis_title="Score",
            yaxis_range=[0, 100],
            height=550,
        )

        st.plotly_chart(fig_score, use_container_width=True)


# ============================================================
# TAB 4 — DAILY SUMMARY
# ============================================================

with tab4:
    st.subheader("Daily Forecast Summary")

    if daily_summary.empty:
        st.warning("No daily summary available.")

    else:
        y_cols = [
            c for c in ["avg_score", "peak_score"]
            if c in daily_summary.columns
        ]

        if "date" in daily_summary.columns and y_cols:
            fig_daily = px.bar(
                daily_summary,
                x="date",
                y=y_cols,
                barmode="group",
                title="Average vs Peak Stargazing Score by Day",
            )

            fig_daily.update_layout(
                xaxis_title="Date",
                yaxis_title="Score",
                yaxis_range=[0, 100],
                height=500,
            )

            st.plotly_chart(fig_daily, use_container_width=True)

        st.dataframe(daily_summary, use_container_width=True)


# ============================================================
# TAB 5 — FEATURE DIAGNOSTICS
# ============================================================

with tab5:
    st.subheader("Feature Diagnostics")

    if score_df.empty:
        st.warning("No scored data available for diagnostics.")

    else:
        feature_cols = [
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

        existing_cols = [c for c in feature_cols if c in score_df.columns]

        if not existing_cols:
            st.warning("No diagnostic feature columns found.")

        else:
            default_index = (
                existing_cols.index("stargazing_score")
                if "stargazing_score" in existing_cols
                else 0
            )

            selected_feature = st.selectbox(
                "Choose feature to inspect",
                existing_cols,
                index=default_index,
            )

            plot_df = score_df.copy()

            if "local_dt" in plot_df.columns:
                plot_df["local_dt"] = pd.to_datetime(plot_df["local_dt"], errors="coerce")
                x_col = "local_dt"
            else:
                plot_df["row_index"] = range(len(plot_df))
                x_col = "row_index"

            fig_feature = px.line(
                plot_df,
                x=x_col,
                y=selected_feature,
                markers=True,
                title=f"{selected_feature} Over Time",
            )

            fig_feature.update_layout(
                xaxis_title="Local Time",
                yaxis_title=selected_feature,
                height=500,
            )

            st.plotly_chart(fig_feature, use_container_width=True)

            st.subheader("Feature Correlation")

            numeric_df = score_df[existing_cols].apply(
                pd.to_numeric,
                errors="coerce",
            )

            corr_cols = [
                c for c in numeric_df.columns
                if numeric_df[c].notna().sum() > 1
            ]

            if len(corr_cols) >= 2:
                corr = numeric_df[corr_cols].corr(numeric_only=True)

                fig_corr = px.imshow(
                    corr,
                    text_auto=True,
                    title="Correlation Between Scoring Features",
                    aspect="auto",
                )

                st.plotly_chart(fig_corr, use_container_width=True)

            else:
                st.info("Not enough numeric columns for correlation chart.")

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

        st.dataframe(
            score_df[diagnostic_cols].head(50),
            use_container_width=True,
        )


# ============================================================
# TAB 6 — SKY PATH
# ============================================================

with tab6:
    st.subheader("Sun / Moon Sky Path")

    st.caption(
        "This visualization uses Sun/Moon position data only. "
        "It does not affect the recommendation score."
    )

    if not include_positions:
        st.info("Turn on 'Use Sun/Moon position data' in the sidebar to view this chart.")

    elif position_df is None or position_df.empty:
        st.warning("No Sun/Moon position data was returned.")

    else:
        required_cols = ["Azimuth (°)", "Altitude (°)", "Object"]

        st.dataframe(position_df, use_container_width=True)

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
                        c
                        for c in ["Hour", "Illuminated (%)", "Moon Phase"]
                        if c in pos_plot_df.columns
                    ],
                    title="Sun and Moon Position in the Sky",
                )

                fig_pos.update_layout(height=650)

                st.plotly_chart(fig_pos, use_container_width=True)

            else:
                st.warning(
                    "Position data exists, but altitude/azimuth values are missing."
                )

        else:
            st.warning(
                "Position data does not contain the required columns: "
                "Azimuth (°), Altitude (°), Object."
            )


# ============================================================
# TAB 7 — RAW DATA
# ============================================================

with tab7:
    st.subheader("Generated Master DataFrame")
    st.dataframe(master_df, use_container_width=True)

    st.subheader("Scored DataFrame")
    st.dataframe(score_df, use_container_width=True)

    st.subheader("Top Windows DataFrame")
    st.dataframe(top_windows, use_container_width=True)

    st.subheader("Weather Data")
    st.dataframe(weather_df, use_container_width=True)

    st.subheader("IPGeolocation / Astronomy Data")
    st.dataframe(ip_geo_df, use_container_width=True)

    if event_df is not None and not event_df.empty:
        st.subheader("Timeanddate Event Data")
        st.dataframe(event_df, use_container_width=True)

    if position_df is not None and not position_df.empty:
        st.subheader("Sun / Moon Position Data")
        st.dataframe(position_df, use_container_width=True)
