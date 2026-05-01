import json
import math
import random
import os
from html import escape
import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
import plotly.express as px
from datetime import datetime

try:
    import pydeck as pdk
except Exception:
    pdk = None

# ============================================================
# BACKEND IMPORTS
# ============================================================

try:
    import backend as backend_module

    run_pipeline = backend_module.run_pipeline
    CITY_PRESETS = backend_module.CITY_PRESETS
    generate_llm_recommendation = backend_module.generate_llm_recommendation
    fetch_tad_positions = backend_module.fetch_tad_positions

    generate_rag_recommendation = getattr(backend_module, "generate_rag_recommendation", None)
    generate_forecast_ai_insight = getattr(backend_module, "generate_forecast_ai_insight", None)
    answer_semantic_knowledge_question = getattr(backend_module, "answer_semantic_knowledge_question", None)
    generate_travel_plan_for_current_forecast = getattr(
        backend_module,
        "generate_travel_plan_for_current_forecast",
        None,
    )
    TRAVEL_PLAN_SCORE_THRESHOLD = getattr(backend_module, "TRAVEL_PLAN_SCORE_THRESHOLD", 70.0)

    BACKEND_IMPORT_ERROR = None
    HAS_RAG_BACKEND = generate_rag_recommendation is not None

except ImportError as backend_import_error:
    raise RuntimeError(f"Could not import backend.py: {backend_import_error}") from backend_import_error


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Stargazing Assistant",
    page_icon="🌌",
    layout="wide",
)


BROWSER_GEOLOCATION = components.declare_component(
    "browser_geolocation",
    path=os.path.join(os.path.dirname(__file__), "components", "browser_geolocation"),
)


def browser_geolocation(key: str = "browser_geolocation"):
    return BROWSER_GEOLOCATION(key=key, default=None, height=72)


# ============================================================
# NEBULA DUSK — FULL THEME CSS
# ============================================================

st.markdown(
    """
    <style>
    @font-face {
        font-family: 'DM Serif Display';
        font-style: normal;
        font-weight: 400;
        font-display: swap;
        src: url('https://cdn.jsdelivr.net/npm/@fontsource/dm-serif-display@5/files/dm-serif-display-latin-400-normal.woff2') format('woff2');
    }
    @font-face {
        font-family: 'DM Serif Display';
        font-style: italic;
        font-weight: 400;
        font-display: swap;
        src: url('https://cdn.jsdelivr.net/npm/@fontsource/dm-serif-display@5/files/dm-serif-display-latin-400-italic.woff2') format('woff2');
    }
    @font-face {
        font-family: 'DM Sans';
        font-style: normal;
        font-weight: 400;
        font-display: swap;
        src: url('https://cdn.jsdelivr.net/npm/@fontsource/dm-sans@5/files/dm-sans-latin-400-normal.woff2') format('woff2');
    }
    @font-face {
        font-family: 'DM Sans';
        font-style: normal;
        font-weight: 500;
        font-display: swap;
        src: url('https://cdn.jsdelivr.net/npm/@fontsource/dm-sans@5/files/dm-sans-latin-500-normal.woff2') format('woff2');
    }
    @font-face {
        font-family: 'DM Sans';
        font-style: normal;
        font-weight: 600;
        font-display: swap;
        src: url('https://cdn.jsdelivr.net/npm/@fontsource/dm-sans@5/files/dm-sans-latin-600-normal.woff2') format('woff2');
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <style>

    /* ── DESIGN TOKENS ──────────────────────────────────────── */
    :root {
        --bg:             #090614;
        --sidebar:        #0c0818;
        --card:           #100d1e;
        --card-hover:     #16122a;
        --border:         rgba(196,120,210,0.16);
        --border-soft:    rgba(196,120,210,0.09);
        --accent:         #c478d2;
        --accent-soft:    rgba(196,120,210,0.15);
        --accent-glow:    rgba(196,120,210,0.06);
        --text:           #e4dff0;
        --text-muted:     #8880a0;
        --text-dim:       #554f6a;
        --slider-track:   #1a1430;
        --font-display:   'DM Serif Display', Georgia, serif;
        --font-body:      'DM Sans', system-ui, sans-serif;
    }

    /* ── BASE ───────────────────────────────────────────────── */
    html, body,
    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    .main {
        background-color: var(--bg) !important;
        font-family: 'DM Sans', system-ui, sans-serif !important;
        color: var(--text) !important;
    }

    /* Remove Streamlit default top padding */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 3rem !important;
    }

    /* ── TYPOGRAPHY ─────────────────────────────────────────── */
    h1, h2, h3, h4 {
        font-family: 'DM Serif Display', Georgia, serif !important;
        font-weight: 400 !important;
        color: var(--text) !important;
        letter-spacing: -0.01em;
    }

    p, li, label, td, th,
    button, input, select, textarea,
    [data-testid="stMarkdownContainer"] p {
        font-family: 'DM Sans', system-ui, sans-serif !important;
    }
    .material-icons,
    .material-symbols-rounded,
    .material-symbols-outlined,
    [class^="material-symbols-"],
    [class*=" material-symbols-"] {
        font-family: "Material Symbols Rounded", "Material Symbols Outlined", "Material Icons" !important;
        font-style: normal !important;
        font-weight: normal !important;
        letter-spacing: normal !important;
        text-transform: none !important;
        white-space: nowrap !important;
        direction: ltr !important;
    }
    /* Subheader override */
    [data-testid="stHeadingWithActionElements"] h2,
    [data-testid="stHeadingWithActionElements"] h3 {
        font-family: 'DM Serif Display', Georgia, serif !important;
        font-weight: 400 !important;
        color: var(--text) !important;
    }

    /* ── SCROLLBAR ──────────────────────────────────────────── */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: var(--bg); }
    ::-webkit-scrollbar-thumb { background: rgba(196,120,210,0.25); border-radius: 99px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(196,120,210,0.45); }

    /* ── SIDEBAR ────────────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background: rgba(12,8,24,0.96) !important;
        border-right: 1px solid var(--border) !important;
    }
    [data-testid="stSidebar"] > div:first-child {
        background: transparent !important;
    }
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p {
        color: var(--text) !important;
        font-family: 'DM Sans', system-ui, sans-serif !important;
        font-size: 13px !important;
    }
    [data-testid="stSidebar"] .stMarkdown p {
        color: var(--text-muted) !important;
        font-size: 12px !important;
    }
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div {
        color: var(--text) !important;
    }
    [data-testid="stSidebarCollapseButton"] span,
    [data-testid="collapsedControl"] span,
    button[aria-label*="sidebar" i] span {
        font-family: "Material Symbols Rounded", "Material Symbols Outlined", "Material Icons" !important;
        font-style: normal !important;
        font-weight: normal !important;
        font-size: 20px !important;
        letter-spacing: normal !important;
        text-transform: none !important;
    }

    /* ── DIVIDER ────────────────────────────────────────────── */
    hr, [data-testid="stDivider"] {
        border-color: var(--border-soft) !important;
    }

    /* ── RADIO BUTTONS ──────────────────────────────────────── */
    [data-testid="stRadio"] label {
        color: var(--text-muted) !important;
        font-size: 13px !important;
        font-family: 'DM Sans', system-ui, sans-serif !important;
        transition: color 0.15s;
    }
    [data-testid="stRadio"] label:hover {
        color: var(--text) !important;
    }
    /* Radio outer ring */
    [data-testid="stRadio"] [data-testid="stWidgetLabel"] + div [role="radiogroup"] > label > div:first-child {
        border-color: var(--text-dim) !important;
    }
    /* Selected radio */
    [data-testid="stRadio"] input[type="radio"]:checked + div {
        border-color: var(--accent) !important;
        background-color: var(--accent) !important;
    }
    /* Radio dot color via accent trick */
    [data-testid="stRadio"] input[type="radio"]:checked ~ span {
        color: var(--accent) !important;
    }

    /* ── CHECKBOXES ─────────────────────────────────────────── */
    [data-testid="stCheckbox"] label {
        color: var(--text-muted) !important;
        font-size: 13px !important;
        font-family: 'DM Sans', system-ui, sans-serif !important;
    }
    [data-testid="stCheckbox"] label:hover {
        color: var(--text) !important;
    }
    [data-testid="stCheckbox"] input[type="checkbox"] + div {
        border-color: var(--text-dim) !important;
        background-color: transparent !important;
        border-radius: 4px !important;
    }
    [data-testid="stCheckbox"] input[type="checkbox"]:checked + div {
        background-color: var(--accent-soft) !important;
        border-color: var(--accent) !important;
    }
    [data-testid="stCheckbox"] input[type="checkbox"]:checked + div svg {
        stroke: var(--accent) !important;
        fill: var(--accent) !important;
    }

    /* ── SLIDERS ────────────────────────────────────────────── */
    [data-testid="stSlider"] label {
        color: var(--text-muted) !important;
        font-size: 12px !important;
        font-family: 'DM Sans', system-ui, sans-serif !important;
    }
    /* Track */
    [data-testid="stSlider"] [data-baseweb="slider"] > div {
        background: var(--slider-track) !important;
    }
    /* Filled portion */
    [data-testid="stSlider"] [data-baseweb="slider"] [role="progressbar"] {
        background: var(--accent) !important;
    }
    /* Thumb */
    [data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
        background: var(--accent) !important;
        border-color: var(--bg) !important;
        box-shadow: 0 0 8px rgba(196,120,210,0.55) !important;
        width: 14px !important;
        height: 14px !important;
    }
    /* Slider value label */
    [data-testid="stSlider"] [data-testid="stTickBarMin"],
    [data-testid="stSlider"] [data-testid="stTickBarMax"] {
        color: var(--text-dim) !important;
        font-size: 10px !important;
    }

    /* ── SELECT BOX ─────────────────────────────────────────── */
    [data-testid="stSelectbox"] label {
        color: var(--text-muted) !important;
        font-size: 12px !important;
        font-family: 'DM Sans', system-ui, sans-serif !important;
    }
    [data-testid="stSelectbox"] [data-baseweb="select"] > div {
        background-color: var(--slider-track) !important;
        border-color: var(--border) !important;
        border-radius: 8px !important;
        color: var(--text) !important;
        font-family: 'DM Sans', system-ui, sans-serif !important;
        font-size: 13px !important;
    }
    [data-testid="stSelectbox"] [data-baseweb="select"] > div:hover {
        border-color: var(--accent) !important;
    }
    [data-testid="stSelectbox"] [data-baseweb="select"] svg {
        fill: var(--text-dim) !important;
    }
    /* Dropdown list */
    [data-baseweb="popover"] ul {
        background: var(--card) !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
        padding: 4px !important;
    }
    [data-baseweb="popover"] li {
        background: transparent !important;
        color: var(--text-muted) !important;
        font-family: 'DM Sans', system-ui, sans-serif !important;
        font-size: 13px !important;
        border-radius: 6px !important;
    }
    [data-baseweb="popover"] li:hover,
    [data-baseweb="popover"] li[aria-selected="true"] {
        background: var(--accent-soft) !important;
        color: var(--accent) !important;
    }

    /* ── TEXT INPUT / NUMBER INPUT ──────────────────────────── */
    [data-testid="stTextInput"] label,
    [data-testid="stNumberInput"] label,
    [data-testid="stTextArea"] label {
        color: var(--text-muted) !important;
        font-size: 12px !important;
        font-family: 'DM Sans', system-ui, sans-serif !important;
    }
    [data-testid="stTextInput"] input,
    [data-testid="stNumberInput"] input,
    [data-testid="stTextArea"] textarea {
        background: var(--slider-track) !important;
        border-color: var(--border) !important;
        border-radius: 8px !important;
        color: var(--text) !important;
        font-family: 'DM Sans', system-ui, sans-serif !important;
        font-size: 13px !important;
    }
    [data-testid="stTextInput"] input:focus,
    [data-testid="stNumberInput"] input:focus,
    [data-testid="stTextArea"] textarea:focus {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 2px rgba(196,120,210,0.18) !important;
    }

    /* ── BUTTONS ────────────────────────────────────────────── */
    /* Primary button (Run Pipeline) */
    [data-testid="stButton"] > button[kind="primary"],
    [data-testid="stButton"] > button[data-testid="baseButton-primary"] {
        background: var(--accent-soft) !important;
        border: 1px solid rgba(196,120,210,0.45) !important;
        color: var(--accent) !important;
        font-family: 'DM Sans', system-ui, sans-serif !important;
        font-size: 13px !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
        transition: background 0.15s, border-color 0.15s !important;
    }
    [data-testid="stButton"] > button[kind="primary"]:hover {
        background: rgba(196,120,210,0.28) !important;
        border-color: var(--accent) !important;
    }
    /* Secondary button (Clear cache) */
    [data-testid="stButton"] > button[kind="secondary"],
    [data-testid="stButton"] > button[data-testid="baseButton-secondary"] {
        background: transparent !important;
        border: 1px solid var(--border) !important;
        color: var(--text-muted) !important;
        font-family: 'DM Sans', system-ui, sans-serif !important;
        font-size: 12px !important;
        border-radius: 8px !important;
        transition: border-color 0.15s, color 0.15s !important;
    }
    [data-testid="stButton"] > button[kind="secondary"]:hover {
        border-color: var(--accent) !important;
        color: var(--text) !important;
    }

    /* ── ALERTS (success / info / warning / error) ──────────── */
    [data-testid="stAlert"] {
        border-radius: 10px !important;
        font-family: 'DM Sans', system-ui, sans-serif !important;
        font-size: 13px !important;
    }
    /* Success */
    [data-testid="stAlert"][data-baseweb="notification"][kind="positive"],
    div[data-testid="stSuccessMessage"] {
        background: rgba(74,222,128,0.08) !important;
        border: 1px solid rgba(74,222,128,0.22) !important;
        color: #4ade80 !important;
    }
    /* Info */
    div[data-testid="stInfoMessage"] {
        background: var(--accent-glow) !important;
        border: 1px solid var(--border) !important;
        color: var(--text-muted) !important;
    }
    /* Warning */
    div[data-testid="stWarningMessage"] {
        background: rgba(251,191,36,0.08) !important;
        border: 1px solid rgba(251,191,36,0.22) !important;
        color: #fbbf24 !important;
    }
    /* Error */
    div[data-testid="stErrorMessage"] {
        background: rgba(248,113,113,0.08) !important;
        border: 1px solid rgba(248,113,113,0.22) !important;
        color: #f87171 !important;
    }
    /* Alert icons */
    [data-testid="stAlert"] svg { opacity: 0.8; }

    /* ── SPINNER ────────────────────────────────────────────── */
    [data-testid="stSpinner"] > div {
        border-top-color: var(--accent) !important;
    }
    [data-testid="stStatusWidget"] {
        color: var(--text-muted) !important;
        font-family: 'DM Sans', system-ui, sans-serif !important;
        font-size: 13px !important;
    }

    /* ── DATAFRAME / TABLE ──────────────────────────────────── */
    [data-testid="stDataFrame"] iframe,
    [data-testid="stDataFrame"] {
        border-radius: 10px !important;
        border: 1px solid var(--border) !important;
        overflow: hidden !important;
    }
    .dvn-scroller { background: var(--card) !important; }
    .dvn-scroller th {
        background: var(--slider-track) !important;
        color: var(--text-dim) !important;
        font-family: 'DM Sans', system-ui, sans-serif !important;
        font-size: 11px !important;
        letter-spacing: 0.05em !important;
        text-transform: uppercase !important;
        border-bottom: 1px solid var(--border) !important;
    }
    .dvn-scroller td {
        background: var(--card) !important;
        color: var(--text-muted) !important;
        font-family: 'DM Sans', system-ui, sans-serif !important;
        font-size: 12px !important;
        border-bottom: 1px solid var(--border-soft) !important;
    }
    .dvn-scroller tr:hover td {
        background: var(--card-hover) !important;
    }

    /* ── EXPANDER ───────────────────────────────────────────── */
    [data-testid="stExpander"] {
        background: var(--card) !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
        overflow: hidden !important;
        margin-bottom: 8px !important;
    }
    [data-testid="stExpander"] summary {
        background: transparent !important;
        color: var(--text-muted) !important;
        font-family: 'DM Sans', system-ui, sans-serif !important;
        font-size: 13px !important;
        padding: 12px 16px !important;
    }
    [data-testid="stExpander"] summary:hover {
        color: var(--text) !important;
    }
    [data-testid="stExpander"] summary svg {
        fill: var(--text-dim) !important;
    }
    [data-testid="stExpander"] > div[data-testid="stExpanderDetails"] {
        background: transparent !important;
        border-top: 1px solid var(--border-soft) !important;
        padding: 12px 16px !important;
    }

    /* ── CAPTION ────────────────────────────────────────────── */
    [data-testid="stCaptionContainer"],
    .stCaption {
        color: var(--text-dim) !important;
        font-size: 11px !important;
        font-family: 'DM Sans', system-ui, sans-serif !important;
    }

    /* ── METRIC WIDGET ──────────────────────────────────────── */
    div[data-testid="stMetric"] {
        background: var(--card) !important;
        padding: 14px 16px !important;
        border-radius: 10px !important;
        border: 1px solid var(--border) !important;
    }
    div[data-testid="stMetric"] label {
        color: var(--text-dim) !important;
        font-size: 10px !important;
        letter-spacing: 0.1em !important;
        text-transform: uppercase !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: var(--text) !important;
        font-family: 'DM Serif Display', Georgia, serif !important;
        font-size: 2rem !important;
        font-weight: 400 !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
        color: var(--accent) !important;
    }

    /* ── TABS ───────────────────────────────────────────────── */
    [data-testid="stTabs"] [data-baseweb="tab-list"] {
        background: transparent !important;
        border-bottom: 1px solid var(--border) !important;
        gap: 4px !important;
    }
    [data-testid="stTabs"] [data-baseweb="tab"] {
        background: transparent !important;
        color: var(--text-muted) !important;
        font-family: 'DM Sans', system-ui, sans-serif !important;
        font-size: 13px !important;
        border-radius: 7px 7px 0 0 !important;
        border: none !important;
        padding: 8px 16px !important;
    }
    [data-testid="stTabs"] [data-baseweb="tab"]:hover {
        color: var(--text) !important;
        background: var(--accent-glow) !important;
    }
    [data-testid="stTabs"] [aria-selected="true"][data-baseweb="tab"] {
        color: var(--accent) !important;
        background: var(--accent-soft) !important;
        border-bottom: 2px solid var(--accent) !important;
    }
    [data-testid="stTabs"] [data-baseweb="tab-highlight"] {
        background: var(--accent) !important;
    }
    [data-testid="stTabs"] [data-baseweb="tab-panel"] {
        background: transparent !important;
        padding-top: 16px !important;
    }

    /* ── PLOTLY CHART CONTAINER ─────────────────────────────── */
    [data-testid="stPlotlyChart"] {
        border-radius: 12px !important;
        border: 1px solid var(--border) !important;
        overflow: hidden !important;
        background: var(--card) !important;
    }

    /* ── TOOLTIP ────────────────────────────────────────────── */
    [data-testid="stTooltipHoverTarget"] svg {
        fill: var(--text-dim) !important;
    }

    /* ── SIDEBAR NAV RADIO OVERRIDE ─────────────────────────── */
    /* Make selected nav item stand out */
    [data-testid="stSidebar"] [data-testid="stRadio"] > div > label[data-checked="true"] > div:first-child,
    [data-testid="stSidebar"] .stRadio > div > label:has(input:checked) > div:first-child {
        background: var(--accent) !important;
        border-color: var(--accent) !important;
    }
    [data-testid="stSidebar"] .stRadio > div > label:has(input:checked) > div:last-child p {
        color: var(--text) !important;
        font-weight: 600 !important;
    }

    /* ── CUSTOM COMPONENT CLASSES ───────────────────────────── */
    .hero-card {
        background: linear-gradient(135deg, var(--card) 0%, rgba(196,120,210,0.06) 100%);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 28px 32px;
        margin-bottom: 20px;
        position: relative;
        overflow: hidden;
    }
    .hero-card::before {
        content: '';
        position: absolute;
        top: -60px; right: -60px;
        width: 220px; height: 220px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(196,120,210,0.08) 0%, transparent 70%);
        pointer-events: none;
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
        position: relative;
    }
    .hero-icon-wrap {
        width: 52px; height: 52px;
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
        font-family: 'DM Serif Display', Georgia, serif !important;
        font-size: 2rem;
        font-weight: 400 !important;
        color: #f5f0ff;
        line-height: 1.15;
        letter-spacing: -0.01em;
        text-shadow: 0 0 14px rgba(196,120,210,0.22), 0 0 2px rgba(255,255,255,0.2);
    }
    .hero-desc {
        margin: 0;
        color: #b4aacb;
        font-size: 15px;
        line-height: 1.6;
        max-width: 680px;
        text-shadow: 0 0 10px rgba(0,0,0,0.25);
    }
    .hero-copy {
        position: relative;
        padding-right: 40px;
    }
    .hero-sparkles {
        position: absolute;
        top: -6px;
        right: -95px;
        left: auto;
        width: 220px;
        height: 145px;
        pointer-events: none;
    }
    .hero-const-label {
        position: absolute;
        color: rgba(255, 248, 220, 0.9);
        font-size: 10px;
        font-weight: 600;
        letter-spacing: 0.03em;
        text-shadow: 0 0 6px rgba(255, 244, 196, 0.35);
        white-space: nowrap;
    }
    .hero-const-label.ursa { left: 40px; top: 25px; }
    .hero-const-label.polaris { right: 0px; top: 4px; }
    .hero-const-label.plough { left: 58px; top: 126px; }
    .hero-const-line {
        position: absolute;
        height: 1.2px;
        background: linear-gradient(90deg, rgba(255,245,190,0.08), rgba(255,245,190,0.28), rgba(255,245,190,0.08));
        transform-origin: left center;
        filter: drop-shadow(0 0 3px rgba(255,244,180,0.22));
    }
    .hero-const-line.l1  { left: 33px;  top: 72px;  width: 22px; transform: rotate(10deg); }
    .hero-const-line.l2  { left: 55px;  top: 75px;  width: 24px; transform: rotate(23deg); }
    .hero-const-line.l3  { left: 77px;  top: 85px;  width: 26px; transform: rotate(31deg); }
    .hero-const-line.l4  { left: 99px;  top: 98px;  width: 17px; transform: rotate(130deg); }
    .hero-const-line.l5  { left: 88px;  top: 111px; width: 33px; transform: rotate(-7deg); }
    .hero-const-line.l6  { left: 121px; top: 107px; width: 19px; transform: rotate(-55deg); }
    .hero-const-line.l7  { left: 132px; top: 91px;  width: 79px; transform: rotate(-57deg); }
    .hero-const-line.l8  { left: 110px; top: 39px;  width: 13px; transform: rotate(-31deg); }
    .hero-const-line.l9  { left: 99px;  top: 46px;  width: 13px; transform: rotate(-31deg); }
    .hero-const-line.l10 { left: 99px;  top: 46px;  width: 34px; transform: rotate(-11deg); }
    .hero-const-line.l11 { left: 132px; top: 39px;  width: 22px; transform: rotate(-7deg); }
    .hero-const-line.l12 { left: 154px; top: 36px;  width: 12px; transform: rotate(-20deg); }
    .hero-const-line.l13 { left: 165px; top: 33px;  width: 13px; transform: rotate(-31deg); }
    .hero-sparkle {
        position: absolute;
        width: 10px;
        height: 10px;
        background: linear-gradient(180deg, #fffde6 0%, #f8e7a5 60%, #e7c86e 100%);
        clip-path: polygon(
            50% 0%, 58% 32%, 88% 12%, 68% 45%, 100% 50%,
            68% 55%, 88% 88%, 58% 68%, 50% 100%, 42% 68%,
            12% 88%, 32% 55%, 0% 50%, 32% 45%, 12% 12%, 42% 32%
        );
        box-shadow: 0 0 8px rgba(255,235,150,0.72);
        animation: starShimmer 2s ease-in-out infinite;
    }
    .hero-sparkle::after {
        content: "";
        position: absolute;
        left: 50%;
        top: 50%;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(255,255,240,0.22) 0%, rgba(255,255,240,0) 70%);
        transform: translate(-50%, -50%);
        pointer-events: none;
    }
    .hero-sparkle.s1  { left: 33px;  top: 72px;  animation-delay: 0.0s; }
    .hero-sparkle.s2  { left: 55px;  top: 75px;  animation-delay: 0.2s; }
    .hero-sparkle.s3  { left: 77px;  top: 85px;  animation-delay: 0.4s; }
    .hero-sparkle.s4  { left: 99px;  top: 98px;  animation-delay: 0.6s; }
    .hero-sparkle.s5  { left: 88px;  top: 111px; animation-delay: 0.8s; }
    .hero-sparkle.s6  { left: 121px; top: 107px; animation-delay: 1.0s; }
    .hero-sparkle.s7  { left: 132px; top: 91px;  animation-delay: 1.2s; }
    .hero-sparkle.s9  { left: 121px; top: 33px;  animation-delay: 0.15s; transform: scale(0.9); }
    .hero-sparkle.s10 { left: 110px; top: 39px;  animation-delay: 0.35s; transform: scale(0.9); }
    .hero-sparkle.s11 { left: 99px;  top: 46px;  animation-delay: 0.55s; transform: scale(0.9); }
    .hero-sparkle.s12 { left: 132px; top: 39px;  animation-delay: 0.75s; transform: scale(0.88); }
    .hero-sparkle.s13 { left: 154px; top: 36px;  animation-delay: 0.95s; transform: scale(0.88); }
    .hero-sparkle.s14 { left: 165px; top: 33px;  animation-delay: 1.15s; transform: scale(0.88); }
    .hero-sparkle.s8 {
        left: 176px;
        top: 26px;
        width: 12px;
        height: 12px;
        background: #fffde8;
        box-shadow: 0 0 16px rgba(255,246,190,1);
        animation-delay: 1.35s;
    }
    @keyframes starShimmer {
        0%, 100% { opacity: 0.45; transform: scale(0.9); }
        50% { opacity: 1; transform: scale(1.2); }
    }
    @media (max-width: 1280px) {
        .hero-sparkles { right: -170px; transform: scale(0.9); transform-origin: top right; }
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
        font-family: 'DM Sans', system-ui, sans-serif;
    }
    .score-dot {
        width: 7px; height: 7px;
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
        font-family: 'DM Sans', system-ui, sans-serif;
    }
    .metric-large {
        font-family: 'DM Serif Display', Georgia, serif;
        font-size: 2rem;
        font-weight: 400;
        color: var(--text);
        line-height: 1;
    }
    .metric-unit {
        font-size: 0.9rem;
        color: var(--text-muted);
        font-weight: 300;
        font-family: 'DM Sans', system-ui, sans-serif;
    }
    .section-card {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 24px 28px;
        margin-bottom: 16px;
    }
    .section-heading {
        font-family: 'DM Serif Display', Georgia, serif !important;
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
        font-family: 'DM Sans', system-ui, sans-serif;
    }
    .window-card {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 20px;
        min-height: 200px;
        transition: background 0.15s;
    }
    .window-card:hover { background: var(--card-hover); }
    .window-rank {
        color: var(--accent);
        font-weight: 600;
        font-size: 11px;
        letter-spacing: 0.04em;
        margin-bottom: 8px;
        font-family: 'DM Sans', system-ui, sans-serif;
    }
    .window-time {
        font-family: 'DM Serif Display', Georgia, serif;
        color: var(--text);
        font-size: 1.15rem;
        font-weight: 400;
        margin-bottom: 6px;
    }
    .window-score {
        font-family: 'DM Serif Display', Georgia, serif;
        font-size: 1.9rem;
        font-weight: 400;
        color: var(--text);
        line-height: 1.1;
    }
    .muted {
        color: var(--text-muted);
        font-size: 13px;
        line-height: 1.6;
        font-family: 'DM Sans', system-ui, sans-serif;
    }
    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 99px;
        font-size: 11px;
        font-weight: 600;
        margin-right: 6px;
        margin-top: 6px;
        font-family: 'DM Sans', system-ui, sans-serif;
    }
    .badge-excellent { background: rgba(34,197,94,0.12); color: #4ade80; border: 1px solid rgba(74,222,128,0.25); }
    .badge-good      { background: rgba(20,184,166,0.12); color: #5eead4; border: 1px solid rgba(94,234,212,0.25); }
    .badge-marginal  { background: rgba(245,158,11,0.12); color: #fbbf24; border: 1px solid rgba(251,191,36,0.25); }
    .badge-poor      { background: rgba(249,115,22,0.12); color: #fb923c; border: 1px solid rgba(251,146,60,0.25); }
    .badge-nogo      { background: rgba(239,68,68,0.12);  color: #f87171; border: 1px solid rgba(248,113,113,0.25); }

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
        font-family: 'DM Sans', system-ui, sans-serif;
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
        font-family: 'DM Sans', system-ui, sans-serif;
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
        font-family: 'DM Sans', system-ui, sans-serif;
    }
    .cluster-desc {
        font-size: 12px;
        font-style: italic;
        color: var(--text-dim);
        margin-top: 6px;
        line-height: 1.55;
        font-family: 'DM Sans', system-ui, sans-serif;
    }
    .rag-box {
        background: rgba(196,120,210,0.08);
        border: 1px solid rgba(196,120,210,0.22);
        border-radius: 12px;
        padding: 18px 22px;
        margin-top: 14px;
        color: var(--text);
        line-height: 1.7;
        font-family: 'DM Sans', system-ui, sans-serif;
        font-size: 13px;
    }
    .info-note {
        font-size: 13px;
        color: var(--accent);
        background: rgba(196,120,210,0.08);
        border: 1px solid rgba(196,120,210,0.22);
        border-radius: 8px;
        padding: 8px 14px;
        margin: 6px 0 0 0;
        display: inline-block;
        font-family: 'DM Sans', system-ui, sans-serif;
    }
    .inline-cell {
        background: rgba(10, 10, 24, 0.82);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 10px 12px;
        margin: 8px 0;
    }
    .status-text {
        color: #7ee0a3;
        font-size: 14px;
        font-weight: 500;
        margin: 0;
        font-family: 'DM Sans', system-ui, sans-serif;
    }
    .reactive-star-field {
        transition: opacity 0.6s ease;
        opacity: var(--star-opacity, 0.72);
    }
    .reactive-clouds-overlay {
        position: fixed;
        inset: 0;
        pointer-events: none;
        z-index: 1;
        opacity: var(--cloud-opacity, 0);
        background:
          radial-gradient(circle at 20% 20%, rgba(255,255,255,0.07), rgba(255,255,255,0) 45%),
          radial-gradient(circle at 70% 35%, rgba(255,255,255,0.06), rgba(255,255,255,0) 55%),
          radial-gradient(circle at 45% 80%, rgba(255,255,255,0.05), rgba(255,255,255,0) 50%);
        animation: cloudDrift 24s linear infinite;
    }
    @keyframes cloudDrift {
        0%   { transform: translateX(-2%); }
        50%  { transform: translateX(2%); }
        100% { transform: translateX(-2%); }
    }
    .chrono-node {
        position: relative;
        margin: 0 0 14px 22px;
        padding: 12px 14px;
        border: 1px solid rgba(196,120,210,0.2);
        border-radius: 10px;
        background: rgba(16,13,30,0.62);
    }
    .chrono-node::before {
        content: "";
        position: absolute;
        left: -21px;
        top: 16px;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background: rgba(196,120,210,0.8);
        box-shadow: 0 0 0 3px rgba(196,120,210,0.18);
    }
    .chrono-node-optimal {
        border-color: rgba(251,191,36,0.55);
        box-shadow: 0 0 0 1px rgba(251,191,36,0.25), 0 0 16px rgba(251,191,36,0.25);
    }
    .chrono-node-optimal::before {
        background: #fbbf24;
        box-shadow: 0 0 0 3px rgba(251,191,36,0.25), 0 0 14px rgba(251,191,36,0.6);
    }
    .telemetry-console {
        background: #070511;
        border: 1px solid rgba(120,119,198,0.35);
        border-radius: 10px;
        padding: 14px;
        white-space: pre-wrap;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
        font-size: 12px;
        color: #cfe5ff;
        line-height: 1.5;
    }
    .phenomena-card {
        background: rgba(14, 12, 28, 0.9);
        border: 1px solid rgba(196, 120, 210, 0.24);
        border-radius: 12px;
        padding: 12px 14px;
        margin: 10px 0;
        box-shadow: 0 0 0 1px rgba(196,120,210,0.08) inset;
    }
    .phenomena-title {
        margin: 0 0 5px 0;
        color: #efe6ff;
        font-size: 15px;
        font-weight: 700;
        font-family: 'DM Sans', system-ui, sans-serif;
    }
    .phenomena-meta {
        margin: 0;
        color: #b8afd0;
        font-size: 13px;
        line-height: 1.45;
    }
    .phenomena-why {
        margin: 7px 0 0 0;
        color: #d3ccf0;
        font-size: 13px;
        line-height: 1.5;
    }
    .phenomena-score-pill {
        display: inline-block;
        margin-left: 8px;
        padding: 2px 8px;
        border-radius: 999px;
        font-size: 11px;
        font-weight: 600;
        color: #2a0f3a;
        background: linear-gradient(90deg, #fbbf24 0%, #f6d98f 100%);
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
    <svg class="reactive-star-field" xmlns="http://www.w3.org/2000/svg"
         style="position:fixed;top:0;left:0;width:100vw;height:100vh;
                pointer-events:none;z-index:0;"
         preserveAspectRatio="xMidYMid slice">
    {_star_field()}
    </svg>
    <div class="reactive-clouds-overlay"></div>
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
        "All Sections",
        "Overview",
        "Best Windows",
        "Sky Conditions",
        "Score Validation",
        "Sky Path",
        "Celestial Phenomena",
        "AI Insight",
        "Glossary",
        "Methodology",
        "Telemetry",
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
    ["City preset", "Custom latitude / longitude", "Use my current location"],
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
elif input_mode == "Custom latitude / longitude":
    city_name = st.sidebar.text_input("Location name", "Custom Location")
    lat      = st.sidebar.number_input("Latitude",  value=40.7128, format="%.6f")
    lon      = st.sidebar.number_input("Longitude", value=-74.0060, format="%.6f")
    timezone = st.sidebar.text_input("Timezone", "America/New_York")
else:
    with st.sidebar:
        geo_result = browser_geolocation()

    if geo_result:
        if geo_result.get("ok"):
            st.session_state["browser_location"] = {
                "lat": float(geo_result["lat"]),
                "lon": float(geo_result["lon"]),
                "city": "Current Location",
                "tz": geo_result.get("timezone") or "UTC",
                "accuracy": geo_result.get("accuracy"),
            }
        else:
            error_code = geo_result.get("code")
            error_message = geo_result.get("error", "Unknown error")
            if error_code == 1:
                st.sidebar.warning(
                    "Location permission is blocked for this browser/site. "
                    "Allow location for localhost in the browser permission prompt or settings, "
                    "then press Detect my location again. You can also use Custom latitude / longitude."
                )
            else:
                st.sidebar.error(f"Location detection failed: {error_message}")

    browser_loc = st.session_state.get("browser_location")
    if browser_loc:
        lat       = browser_loc["lat"]
        lon       = browser_loc["lon"]
        city_name = browser_loc["city"]
        timezone  = browser_loc["tz"]
        accuracy = browser_loc.get("accuracy")
        accuracy_text = (
            f" · accuracy about {accuracy:.0f} m"
            if isinstance(accuracy, (int, float))
            else ""
        )
        st.sidebar.success(f"Detected: {city_name} ({lat:.4f}, {lon:.4f}){accuracy_text}")
    else:
        st.sidebar.info("Allow browser location access to use precise coordinates.")

days = st.sidebar.slider("Forecast range", min_value=1, max_value=4, value=4)

bortle_index = st.sidebar.slider(
    "City lights index",
    min_value=1, max_value=9, value=5,
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
    help="Used for Sky Path and Celestial Phenomena guidance only. This does not affect the score.",
)

include_llm = st.sidebar.checkbox(
    "Generate AI recommendation",
    value=False,
    help="Only explains fixed model output. This does not affect the score.",
)

run_button = st.sidebar.button("▶  Run Pipeline", width="stretch", type="primary")

if st.sidebar.button("Clear cached data", width="stretch"):
    st.cache_data.clear()
    st.session_state.pop("pipeline_result", None)
    st.session_state.pop("pipeline_bortle", None)
    st.session_state.pop("browser_location", None)
    st.session_state.pop("browser_geolocation", None)
    st.session_state.pop("travel_result", None)
    st.sidebar.success("Cache cleared. Run the pipeline again.")


# ============================================================
# CACHE FUNCTIONS
# ============================================================

@st.cache_data(ttl=1800, show_spinner=False)
def cached_run_pipeline(city_name, lat, lon, timezone, days, bortle_index, include_tad, include_positions):
    return run_pipeline(
        city_name=city_name,
        lat=lat, lon=lon,
        timezone=timezone,
        days=days,
        bortle_index=bortle_index,
        include_tad=include_tad,
        include_positions=include_positions,
        include_llm=False,
    )


@st.cache_data(ttl=1800, show_spinner=False)
def cached_fetch_positions(lat, lon, timezone, days):
    return fetch_tad_positions(lat=lat, lon=lon, timezone=timezone, days=days)


@st.cache_data(ttl=1800, show_spinner=False)
def cached_generate_llm(context):
    return generate_llm_recommendation(context)


@st.cache_data(ttl=1800, show_spinner=False)
def cached_generate_rag(context, user_question):
    """
    Backward-compatible cache wrapper.
    """
    if generate_rag_recommendation is None:
        return (
            "RAG backend function not found. Please add "
            "`generate_rag_recommendation()` to backend.py first."
        )
    return generate_rag_recommendation(context, user_question=user_question)


@st.cache_data(ttl=1800, show_spinner=False)
def cached_generate_forecast_insight(context):
    if generate_forecast_ai_insight is None:
        return (
            "Forecast AI Insight backend function not found. Please add "
            "`generate_forecast_ai_insight()` to backend.py first.\n\n"
            f"Backend import error: {BACKEND_IMPORT_ERROR or 'unknown'}"
        )
    return generate_forecast_ai_insight(context)


@st.cache_data(ttl=1800, show_spinner=False)
def cached_generate_travel_plan(
    context,
    bortle_index,
    days,
    radius_km,
    max_candidates,
    score_threshold,
    use_ai_text,
):
    if generate_travel_plan_for_current_forecast is None:
        return {
            "search": {
                "status": "unavailable",
                "reason": (
                    "Travel-planning backend function not found. Please add "
                    "`generate_travel_plan_for_current_forecast()` to backend.py first. "
                    f"Backend import error: {BACKEND_IMPORT_ERROR or 'unknown'}"
                ),
                "candidates": [],
                "best_candidate": None,
            },
            "travel_plan": None,
        }

    return generate_travel_plan_for_current_forecast(
        context=context,
        bortle_index=bortle_index,
        days=days,
        radius_km=radius_km,
        max_candidates=max_candidates,
        score_threshold=score_threshold,
        use_ai_text=use_ai_text,
    )


@st.cache_data(ttl=1800, show_spinner=False)
def cached_answer_semantic_question(user_question, context, use_forecast_context):
    if answer_semantic_knowledge_question is None:
        return (
            "Semantic Search backend function not found. Please add "
            "`answer_semantic_knowledge_question()` to backend.py first.\n\n"
            f"Backend import error: {BACKEND_IMPORT_ERROR or 'unknown'}"
        )
    return answer_semantic_knowledge_question(
        user_question=user_question,
        context=context,
        use_forecast_context=use_forecast_context,
    )


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
    if label == "excellent": return "badge-excellent"
    if label == "good":      return "badge-good"
    if label == "marginal":  return "badge-marginal"
    if label == "poor":      return "badge-poor"
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
        reasons.append("low cloud cover" if cloud <= 30 else ("moderate cloud cover" if cloud <= 60 else "high cloud cover"))
    if transparency is not None:
        reasons.append("strong transparency" if transparency <= 2 else ("acceptable transparency" if transparency <= 4 else "weak transparency"))
    if seeing is not None:
        reasons.append("stable seeing" if seeing <= 2 else ("moderate seeing" if seeing <= 4 else "poor seeing"))
    if moon is not None:
        reasons.append("low moon illumination" if moon <= 30 else ("moderate moon illumination" if moon <= 70 else "bright moon"))

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
        debug_cols = [c for c in [
            "local_dt","date","stargazing_score","recommendation","cloud_value",
            "transparency_value","seeing_value","moon_illuminated_pct","is_dark_enough",
            "is_moon_up","visibility_penalty","transparency_norm","seeing_norm",
            "humidity_quality","moon_brightness_penalty","effective_darkness","atmospheric_score",
        ] if c in score_df.columns]
        st.subheader("First 30 scored rows")
        st.dataframe(score_df[debug_cols].head(30), width="stretch")
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


HUMAN_LABELS = {
    "local_dt": "Local Time",
    "stargazing_score": "Stargazing Score",
    "recommendation": "Recommendation",
    "cloud_value": "Cloud Cover",
    "transparency_value": "Transparency",
    "seeing_value": "Seeing",
    "moon_illuminated_pct": "Moon Illumination (%)",
    "is_dark_enough": "Dark Enough",
    "is_moon_up": "Moon Above Horizon",
    "visibility_penalty": "Visibility Penalty",
    "cloud_transmission": "Cloud Transmission",
    "darkness_gate": "Darkness Gate",
    "transparency_norm": "Transparency Quality",
    "seeing_norm": "Seeing Quality",
    "humidity_quality": "Humidity Quality",
    "haze_quality": "Haze Quality",
    "moon_brightness_penalty": "Moon Brightness Penalty",
    "effective_darkness": "Effective Darkness",
    "atmospheric_score": "Atmospheric Score",
    "observability_score": "Observability Score",
    "view_quality_score": "View Quality Score",
    "legacy_stargazing_score": "Legacy Score",
    "aerosol_optical_depth": "Aerosol Optical Depth",
    "pm2_5": "PM2.5",
    "dust": "Dust",
    "meteoblue_cloud_value": "Meteoblue Cloud Cover",
    "meteoblue_low_clouds": "Meteoblue Low Clouds",
    "meteoblue_mid_clouds": "Meteoblue Mid Clouds",
    "meteoblue_high_clouds": "Meteoblue High Clouds",
    "meteoblue_visibility_m": "Meteoblue Visibility",
    "meteoblue_transparency_value": "Meteoblue Transparency Proxy",
    "meteoblue_seeing_proxy_value": "Meteoblue Seeing Proxy",
    "meteoblue_relativehumidity": "Meteoblue Relative Humidity",
    "meteoblue_fog_probability": "Meteoblue Fog Probability",
    "cloud_model_delta": "Cloud Model Delta",
    "transparency_model_delta": "Transparency Model Delta",
    "expected_score": "Formula Recomputed Score",
    "score_residual": "Formula Difference",
    "expected_recommendation": "Formula Recomputed Label",
    "score_band": "Score Band",
    "darkness_state": "Darkness State",
    "moon_state": "Moon State",
    "cloud_band": "Cloud Band",
    "bortle_index": "Bortle Index",
    "median_score": "Median Score",
    "mean_score": "Mean Score",
    "peak_score": "Peak Score",
    "excellent_good_share": "Excellent / Good Share",
    "credibility_score": "Credibility Score",
    "score_scope": "Score Scope",
    "time_label": "Time Window",
    "count": "Count",
    "factor": "Factor",
    "value": "Value",
    "Hour": "Hour of Day",
}


def human_label(name: str) -> str:
    return HUMAN_LABELS.get(name, name.replace("_", " ").title())


def labels_for(*names: str) -> dict:
    return {name: human_label(name) for name in names}


def classify_score_label(score: float) -> str:
    score = safe_numeric(score, 0.0) or 0.0
    if score >= 85:
        return "Excellent"
    if score >= 70:
        return "Good"
    if score >= 50:
        return "Marginal"
    if score >= 25:
        return "Poor"
    return "No-Go"


def truthy_label(value, true_label: str, false_label: str) -> str:
    if pd.isna(value):
        return false_label
    return true_label if bool(value) else false_label


def _azimuth_to_direction(azimuth: float) -> str:
    if azimuth is None or pd.isna(azimuth):
        return "Unknown"
    az = float(azimuth) % 360.0
    compass = [
        "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
        "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW",
    ]
    idx = int((az + 11.25) // 22.5) % len(compass)
    return compass[idx]


def _nearest_score_snapshot(score_lookup: pd.DataFrame, target_dt: pd.Timestamp) -> tuple:
    if score_lookup is None or score_lookup.empty or pd.isna(target_dt):
        return 50.0, False, "Unknown"
    score_times = pd.to_datetime(score_lookup["local_dt"], errors="coerce")
    if hasattr(score_times.dt, "tz") and score_times.dt.tz is not None:
        score_times = score_times.dt.tz_localize(None)

    target_ts = pd.to_datetime(target_dt, errors="coerce")
    if pd.isna(target_ts):
        return 50.0, False, "Unknown"
    if getattr(target_ts, "tzinfo", None) is not None:
        target_ts = target_ts.tz_localize(None)

    deltas = (score_times - target_ts).abs()
    idx = deltas.idxmin()
    row = score_lookup.loc[idx]
    return (
        safe_numeric(row.get("stargazing_score"), 50.0) or 50.0,
        bool(row.get("is_dark_enough", False)),
        str(row.get("recommendation", "Unknown")),
    )


def build_celestial_recommendations(
    score_df: pd.DataFrame,
    event_df: pd.DataFrame,
    position_df: pd.DataFrame,
    latitude: float,
    timezone_name: str,
    top_n: int = 5,
) -> pd.DataFrame:
    if score_df is None or score_df.empty:
        return pd.DataFrame()

    score_lookup = score_df.copy()
    score_lookup["local_dt"] = pd.to_datetime(score_lookup["local_dt"], errors="coerce")
    score_lookup = score_lookup.dropna(subset=["local_dt"]).reset_index(drop=True)
    if score_lookup.empty:
        return pd.DataFrame()

    candidates = []

    if position_df is not None and not position_df.empty and "UTC Time" in position_df.columns:
        pos = position_df.copy()
        pos["utc_dt"] = pd.to_datetime(pos["UTC Time"], utc=True, errors="coerce")
        pos = pos.dropna(subset=["utc_dt"])
        if timezone_name:
            pos["local_dt"] = pos["utc_dt"].dt.tz_convert(timezone_name).dt.tz_localize(None)
        else:
            pos["local_dt"] = pos["utc_dt"].dt.tz_localize(None)

        if "Object" in pos.columns:
            moon_rows = pos[pos["Object"].astype(str).str.lower() == "moon"].copy()
        else:
            moon_rows = pd.DataFrame()
        for _, row in moon_rows.iterrows():
            altitude = safe_numeric(row.get("Altitude (°)"), None)
            azimuth = safe_numeric(row.get("Azimuth (°)"), None)
            if altitude is None or azimuth is None or altitude < 8:
                continue

            score_value, dark_enough, score_label = _nearest_score_snapshot(
                score_lookup, row.get("local_dt")
            )
            illumination = safe_numeric(row.get("Illuminated (%)"), 45.0)
            shape = max(0.0, 1.0 - abs(altitude - 35.0) / 55.0)
            moonlight_factor = max(0.45, 1.0 - (illumination / 200.0))
            phenomenon_quality = min(1.0, shape * moonlight_factor + (0.1 if dark_enough else 0.0))

            src = str(row.get("Data Source", "Timeanddate astrodata"))
            source_weight = 0.82 if "estimated fallback" in src.lower() else 1.0
            combined = (0.7 * score_value + 30.0 * phenomenon_quality) * source_weight

            candidates.append(
                {
                    "phenomenon": "Moon observing window",
                    "local_time": pd.to_datetime(row.get("local_dt")),
                    "direction": f"{_azimuth_to_direction(azimuth)} ({azimuth:.0f} deg azimuth)",
                    "elevation_deg": round(float(altitude), 1),
                    "score_at_time": round(float(score_value), 1),
                    "combined_score": round(float(combined), 1),
                    "score_label": score_label,
                    "source": src,
                    "why": (
                        f"Moon at {altitude:.0f} deg altitude with {illumination:.0f}% illumination. "
                        f"Weighted with forecast score at this time."
                    ),
                }
            )

    if event_df is not None and not event_df.empty:
        evt = event_df.copy()
        if "local_dt_naive" in evt.columns:
            evt["event_local_dt"] = pd.to_datetime(evt["local_dt_naive"], errors="coerce")
        else:
            evt["event_local_dt"] = pd.to_datetime(evt.get("UTC Time"), utc=True, errors="coerce")
            if timezone_name:
                evt["event_local_dt"] = (
                    evt["event_local_dt"].dt.tz_convert(timezone_name).dt.tz_localize(None)
                )
            else:
                evt["event_local_dt"] = evt["event_local_dt"].dt.tz_localize(None)
        evt = evt.dropna(subset=["event_local_dt"])

        for _, row in evt.iterrows():
            obj = str(row.get("Object", "Unknown"))
            event_name = str(row.get("Event", "")).strip()
            event_key = f"{obj} {event_name}".lower()

            if "nautical" in event_key and "begin" in event_key:
                phenomenon = "Nautical twilight begins"
                quality = 0.72
            elif "nautical" in event_key and "end" in event_key:
                phenomenon = "Nautical twilight ends"
                quality = 0.78
            elif obj.lower() == "moon" and "meridian" in event_key:
                phenomenon = "Moon at local meridian"
                quality = 0.82
            elif obj.lower() == "moon" and "phase" in event_key:
                phenomenon = "Moon phase event"
                quality = 0.62
            else:
                continue

            score_value, dark_enough, score_label = _nearest_score_snapshot(
                score_lookup, row["event_local_dt"]
            )
            combined = 0.75 * score_value + 25.0 * (quality + (0.08 if dark_enough else 0.0))

            candidates.append(
                {
                    "phenomenon": phenomenon,
                    "local_time": pd.to_datetime(row["event_local_dt"]),
                    "direction": "See Sky Path for live azimuth",
                    "elevation_deg": safe_numeric(row.get("Altitude (°)"), None),
                    "score_at_time": round(float(score_value), 1),
                    "combined_score": round(float(combined), 1),
                    "score_label": score_label,
                    "source": "Timeanddate events",
                    "why": f"{obj} {event_name} aligned with forecast quality at this time.",
                }
            )

    if not candidates:
        return pd.DataFrame()

    out = pd.DataFrame(candidates)
    out = out.dropna(subset=["local_time"]).copy()
    out["time_label"] = out["local_time"].dt.strftime("%b %d, %I:%M %p")
    out = out.sort_values("combined_score", ascending=False)
    out = out.drop_duplicates(subset=["phenomenon"], keep="first")
    out = out.head(top_n).reset_index(drop=True)
    out["rank"] = out.index + 1
    return out[
        [
            "rank",
            "phenomenon",
            "time_label",
            "direction",
            "elevation_deg",
            "score_at_time",
            "combined_score",
            "score_label",
            "source",
            "why",
        ]
    ]


def render_celestial_recommendations(reco_df: pd.DataFrame):
    if reco_df is None or reco_df.empty:
        st.info("No celestial recommendations available yet. Run pipeline with Timeanddate data enabled.")
        return
    st.caption(
        "Ranking blends live astronomy events/positions with the deterministic stargazing score. "
        "Higher combined score means better overall recommendation quality."
    )
    for _, row in reco_df.iterrows():
        st.markdown(
            f"""
            <div class="phenomena-card">
                <p class="phenomena-title">
                    #{int(row['rank'])} {escape(str(row['phenomenon']))}
                    <span class="phenomena-score-pill">{float(row['combined_score']):.1f}/100</span>
                </p>
                <p class="phenomena-meta">
                    <b>When:</b> {escape(str(row['time_label']))} &nbsp;|&nbsp;
                    <b>Direction:</b> {escape(str(row['direction']))} &nbsp;|&nbsp;
                    <b>Score label:</b> {escape(str(row['score_label']))}
                </p>
                <p class="phenomena-why">{escape(str(row['why']))}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_celestial_card(rank: int, row: pd.Series):
    st.markdown(
        f"""
        <div class="window-card">
            <h4 style="margin:0 0 6px 0;color:#efe6ff;font-size:15px;">
                #{int(rank)} {escape(str(row.get("phenomenon", "Celestial event")))}
            </h4>
            <p class="muted" style="margin:0 0 4px 0;">
                {escape(str(row.get("time_label", "Unknown time")))}
            </p>
            <p class="muted" style="margin:0 0 8px 0;">
                {escape(str(row.get("direction", "Unknown direction")))}
            </p>
            <span class="badge badge-good">Combined: {float(safe_numeric(row.get("combined_score"), 0.0) or 0.0):.1f}/100</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


GLOSSARY_TERMS = [
    {"term": "Bortle Scale", "category": "Sky Darkness", "unit": "Index 1-9", "definition": "A scale describing night-sky brightness from dark rural skies (1) to bright city skies (9)."},
    {"term": "Seeing", "category": "Atmospheric Stability", "unit": "Relative quality", "definition": "How steady the atmosphere is. Better seeing produces sharper star and planetary detail."},
    {"term": "Transparency", "category": "Atmospheric Clarity", "unit": "Relative quality", "definition": "How clear the atmosphere is. Better transparency improves visibility of faint deep-sky objects."},
    {"term": "Cloud Cover", "category": "Weather", "unit": "%", "definition": "Fraction of sky covered by clouds. High cloud cover strongly reduces stargazing quality."},
    {"term": "Moon Illumination", "category": "Moonlight", "unit": "%", "definition": "The illuminated fraction of the Moon's disk. Higher illumination usually brightens the sky background."},
    {"term": "Effective Darkness", "category": "Composite Metric", "unit": "0-1", "definition": "A darkness metric combining light pollution and moonlight penalty effects."},
    {"term": "Atmospheric Score", "category": "Composite Metric", "unit": "0-1", "definition": "Weighted score derived from transparency, seeing, and humidity quality."},
    {"term": "Visibility Penalty", "category": "Scoring Constraint", "unit": "Multiplier", "definition": "A hard penalty applied when cloud cover is high or the sky is not dark enough."},
    {"term": "Zenith", "category": "Celestial Geometry", "unit": "Direction", "definition": "The point in the sky directly overhead at the observer's location."},
    {"term": "Azimuth", "category": "Coordinate System", "unit": "Degrees", "definition": "Horizontal direction around the horizon measured in degrees."},
    {"term": "Altitude", "category": "Coordinate System", "unit": "Degrees", "definition": "Angular height of an object above the horizon."},
    {"term": "Nautical Twilight", "category": "Twilight Phase", "unit": "Time Window", "definition": "Solar depression between 6 and 12 degrees below the horizon; brighter than full astronomical darkness."},
]


def infer_reactive_sky_state(score_df: pd.DataFrame, top_windows: pd.DataFrame) -> str:
    # Demo mode: keep a single, fixed background style.
    return "neutral"


def apply_reactive_sky_style(state: str):
    style_map = {
        "overcast": {
            "bg": "#0b0a16",
            "star_opacity": "0.18",
            "cloud_opacity": "0.85",
            "glare": "none",
        },
        "moonlit": {
            "bg": "#090614",
            "star_opacity": "0.38",
            "cloud_opacity": "0.2",
            "glare": "radial-gradient(circle at 55% 18%, rgba(255,255,255,0.22), rgba(255,255,255,0) 32%)",
        },
        "clear_dark": {
            "bg": "#05030a",
            "star_opacity": "0.95",
            "cloud_opacity": "0.0",
            "glare": "none",
        },
        "neutral": {
            "bg": "#090614",
            "star_opacity": "0.72",
            "cloud_opacity": "0.0",
            "glare": "none",
        },
    }
    cfg = style_map.get(state, style_map["neutral"])
    state_bg = cfg["bg"]
    st.markdown(
        f"""
        <style>
        :root {{
            --bg: {cfg["bg"]};
            --star-opacity: {cfg["star_opacity"]};
            --cloud-opacity: {cfg["cloud_opacity"]};
        }}
        .stApp,
        [data-testid="stAppViewContainer"],
        [data-testid="stMain"] {{
            background-color: {state_bg} !important;
            background-image: none !important;
        }}
        [data-testid="stAppViewContainer"]::before {{
            content: "";
            position: fixed;
            inset: 0;
            pointer-events: none;
            z-index: 0;
            background: rgba(6, 6, 16, 0.42);
        }}
        .hero-card,
        .section-card,
        .metric-card,
        .window-card {{
            background: rgba(10, 10, 24, 0.76) !important;
        }}
        .hero-title,
        .hero-desc,
        .muted,
        .metric-card-title,
        .metric-large,
        .metric-unit {{
            opacity: 1 !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_glossary_panel():
    glossary_df = pd.DataFrame(GLOSSARY_TERMS)
    query = st.text_input("Search glossary terms", placeholder="e.g., Bortle, seeing, transparency")
    if query:
        q = query.strip().lower()
        glossary_df = glossary_df[
            glossary_df.apply(
                lambda r: q in str(r["term"]).lower()
                or q in str(r["definition"]).lower()
                or q in str(r["category"]).lower(),
                axis=1,
            )
        ]

    with st.container(height=800):
        if glossary_df.empty:
            st.info("No glossary entries matched your search.")
            return
        for _, row in glossary_df.iterrows():
            with st.expander(f"🛰️ {row['term']}"):
                st.markdown(
                    f"**Category:** {row['category']}  \n"
                    f"**Unit / Type:** {row['unit']}  \n\n"
                    f"{row['definition']}"
                )


def render_telemetry_console(telemetry: dict):
    logs = telemetry.get("logs", []) if isinstance(telemetry, dict) else []
    pipeline = telemetry.get("pipeline", {}) if isinstance(telemetry, dict) else {}
    scoring_model = telemetry.get("scoring_model", {}) if isinstance(telemetry, dict) else {}
    previews = telemetry.get("api_preview", {}) if isinstance(telemetry, dict) else {}

    st.markdown("### Live Pipeline Logs")
    st.markdown(
        f'<div class="telemetry-console">{escape(chr(10).join(logs) if logs else "No telemetry logs available.")}</div>',
        unsafe_allow_html=True,
    )

    st.markdown("### Pipeline Summary")
    st.code(json.dumps(pipeline, indent=2, default=str), language="json")

    st.markdown("### Scoring Model Weights")
    st.code(json.dumps(scoring_model, indent=2, default=str), language="json")

    st.markdown("### API Payload Preview")
    st.code(json.dumps(previews, indent=2, default=str), language="json")


def build_factor_chart(score_df):
    factor_cols = [
        "cloud_value","transparency_value","seeing_value","moon_illuminated_pct",
        "effective_darkness","atmospheric_score","visibility_penalty",
        "cloud_transmission","haze_quality","observability_score","view_quality_score",
        "meteoblue_cloud_value","meteoblue_transparency_value","meteoblue_seeing_proxy_value",
    ]
    existing_cols = [c for c in factor_cols if c in score_df.columns]
    plot_df = score_df.copy()
    plot_df["local_dt"] = pd.to_datetime(plot_df["local_dt"], errors="coerce")
    plot_df = plot_df.dropna(subset=["local_dt"])
    long_df = plot_df.melt(id_vars=["local_dt"], value_vars=existing_cols, var_name="factor", value_name="value")
    long_df["factor"] = long_df["factor"].map(human_label)
    fig = px.line(
        long_df,
        x="local_dt",
        y="value",
        color="factor",
        title="Key Stargazing Factors Over Time",
        labels=labels_for("local_dt", "value", "factor"),
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
        color_discrete_sequence=["#c478d2","#9b5ab0","#7a3d8c","#5c2d6b","#3e1e4a"],
        labels=labels_for("recommendation", "count"),
    )
    return _themed_layout(fig, 420)


def build_score_validation_frame(score_df: pd.DataFrame) -> pd.DataFrame:
    if score_df is None or score_df.empty:
        return pd.DataFrame()

    required = [
        "stargazing_score",
        "visibility_penalty",
        "atmospheric_score",
        "effective_darkness",
    ]
    if any(c not in score_df.columns for c in required):
        return pd.DataFrame()

    validation_df = score_df.copy()
    for col in required + [
        "cloud_value",
        "transparency_value",
        "seeing_value",
        "dewPoint_value",
        "temperature_value",
        "moon_illuminated_pct",
        "moon_meridian_altitude",
        "moon_brightness_penalty",
        "transparency_norm",
        "seeing_norm",
        "humidity_quality",
        "haze_quality",
        "cloud_transmission",
        "darkness_gate",
        "observability_score",
        "view_quality_score",
        "legacy_stargazing_score",
        "aerosol_optical_depth",
        "pm2_5",
        "dust",
        "meteoblue_cloud_value",
        "meteoblue_low_clouds",
        "meteoblue_mid_clouds",
        "meteoblue_high_clouds",
        "meteoblue_visibility_m",
        "meteoblue_transparency_value",
        "meteoblue_seeing_proxy_value",
        "meteoblue_relativehumidity",
        "meteoblue_fog_probability",
        "cloud_model_delta",
        "transparency_model_delta",
    ]:
        if col in validation_df.columns:
            validation_df[col] = pd.to_numeric(validation_df[col], errors="coerce")

    if {"observability_score", "view_quality_score"}.issubset(validation_df.columns):
        validation_df["expected_score"] = (
            100
            * np.power(validation_df["observability_score"] / 100.0, 0.62)
            * np.power(validation_df["view_quality_score"] / 100.0, 0.38)
        ).clip(0, 100)
    else:
        validation_df["expected_score"] = (
            100
            * validation_df["visibility_penalty"]
            * (
                0.65 * validation_df["atmospheric_score"]
                + 0.35 * validation_df["effective_darkness"]
            )
        ).clip(0, 100)

    if {"atmospheric_score", "effective_darkness"}.issubset(validation_df.columns):
        validation_df["expected_view_quality_score"] = (
            100
            * (
                0.52 * validation_df["atmospheric_score"]
                + 0.48 * validation_df["effective_darkness"]
            )
        ).clip(0, 100)

    if {"darkness_gate", "cloud_transmission", "effective_darkness"}.issubset(validation_df.columns):
        validation_df["expected_observability_score"] = (
            100
            * validation_df["darkness_gate"]
            * validation_df["cloud_transmission"]
            * (0.70 + 0.30 * validation_df["effective_darkness"])
        ).clip(0, 100)

    validation_df["score_residual"] = (
        validation_df["stargazing_score"] - validation_df["expected_score"]
    ).abs()
    validation_df["expected_recommendation"] = validation_df["expected_score"].apply(
        classify_score_label
    )

    validation_df["score_band"] = pd.cut(
        validation_df["stargazing_score"],
        bins=[-0.01, 25, 50, 70, 85, 100],
        labels=["No-Go <25", "Poor 25-49", "Marginal 50-69", "Good 70-84", "Excellent 85+"],
        include_lowest=True,
    )
    validation_df["darkness_state"] = validation_df.get(
        "is_dark_enough",
        pd.Series(False, index=validation_df.index),
    ).map(lambda x: truthy_label(x, "Dark enough", "Daylight / Twilight"))
    validation_df["moon_state"] = validation_df.get(
        "is_moon_up",
        pd.Series(False, index=validation_df.index),
    ).map(lambda x: truthy_label(x, "Moon up", "Moon down"))

    if "cloud_value" in validation_df.columns:
        validation_df["cloud_band"] = pd.cut(
            validation_df["cloud_value"],
            bins=[-0.01, 20, 40, 60, 80, 100],
            labels=["0-20%", "21-40%", "41-60%", "61-80%", "81-100%"],
            include_lowest=True,
        )

    if "local_dt" in validation_df.columns:
        validation_df["local_dt"] = pd.to_datetime(validation_df["local_dt"], errors="coerce")
        validation_df["hour"] = validation_df["local_dt"].dt.hour

    return validation_df


def build_bortle_sensitivity_frame(validation_df: pd.DataFrame) -> pd.DataFrame:
    if validation_df is None or validation_df.empty:
        return pd.DataFrame()
    required = ["atmospheric_score", "moon_brightness_penalty"]
    if any(c not in validation_df.columns for c in required):
        return pd.DataFrame()

    rows = []
    source = validation_df.dropna(subset=required).copy()
    if source.empty:
        return pd.DataFrame()

    for bortle in range(1, 10):
        light_pollution_factor = bortle / 9.0
        base_darkness = 1.0 - (light_pollution_factor * 0.8)
        effective_darkness = (
            base_darkness
            - source["moon_brightness_penalty"] * (1 - light_pollution_factor)
        ).clip(0, 1)
        if {"darkness_gate", "cloud_transmission"}.issubset(source.columns):
            observability_score = (
                100
                * source["darkness_gate"]
                * source["cloud_transmission"]
                * (0.70 + 0.30 * effective_darkness)
            ).clip(0, 100)
            view_quality_score = (
                100
                * (0.52 * source["atmospheric_score"] + 0.48 * effective_darkness)
            ).clip(0, 100)
            simulated_score = (
                100
                * np.power(observability_score / 100.0, 0.62)
                * np.power(view_quality_score / 100.0, 0.38)
            ).clip(0, 100)
        else:
            simulated_score = (
                100
                * source["visibility_penalty"]
                * (0.65 * source["atmospheric_score"] + 0.35 * effective_darkness)
            ).clip(0, 100)
        rows.append(
            {
                "bortle_index": bortle,
                "mean_score": simulated_score.mean(),
                "median_score": simulated_score.median(),
                "peak_score": simulated_score.max(),
                "excellent_good_share": (simulated_score >= 70).mean(),
            }
        )

    return pd.DataFrame(rows)


def _quality_flag(label: str, status: str, detail: str) -> dict:
    return {"check": label, "status": status, "detail": detail}


def _status_badge_class(status: str) -> str:
    status = str(status).lower()
    if status == "good":
        return "badge-good"
    if status == "watch":
        return "badge-marginal"
    if status == "high risk":
        return "badge-poor"
    return "badge-nogo"


def render_audit_cards(audit_df: pd.DataFrame):
    if audit_df is None or audit_df.empty:
        st.info("No audit rows were generated for this run.")
        return

    for _, row in audit_df.iterrows():
        status = str(row.get("status", "Unknown"))
        st.markdown(
            f"""
            <div class="section-card" style="margin-bottom:10px;">
                <div class="inline-cell" style="justify-content:space-between;">
                    <h4 style="margin:0;color:#e4dff0;font-size:15px;">{escape(str(row.get("check", "Audit check")))}</h4>
                    <span class="badge {_status_badge_class(status)}">{escape(status)}</span>
                </div>
                <p class="muted" style="margin-top:8px;">{escape(str(row.get("detail", "")))}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_validation_source_cards():
    sources = [
        {
            "source": "Open-Meteo Air Quality",
            "use": "Aerosol optical depth, PM2.5, and dust validate haze/transparency.",
            "integration": "Implemented as no-key enrichment when weather is fetched.",
        },
        {
            "source": "Timeanddate Astro Position",
            "use": "Hourly Moon altitude, azimuth, and illumination validate moonlight penalty.",
            "integration": "Used for moonlight penalty when position data is enabled.",
        },
        {
            "source": "Meteoblue Astronomy Seeing",
            "use": "Independent seeing indices, cloud layers, and jet stream context.",
            "integration": "Best paid/API candidate to replace fallback seeing placeholders.",
        },
        {
            "source": "Skyfield or Astropy",
            "use": "Local Sun/Moon altitude and azimuth without a remote API.",
            "integration": "Good package fallback if Timeanddate position API is unavailable.",
        },
        {
            "source": "GLOBE at Night / SQM logs",
            "use": "Observed limiting magnitude and sky quality readings validate real outcomes.",
            "integration": "Best calibration target for learning weights and thresholds.",
        },
        {
            "source": "VIIRS / NASA Black Marble / light-pollution rasters",
            "use": "Location-derived night-light brightness validates manual Bortle input.",
            "integration": "Future Bortle/skyglow estimator.",
        },
    ]

    for item in sources:
        st.markdown(
            f"""
            <div class="section-card" style="margin-bottom:10px;">
                <h4 style="margin:0;color:#e4dff0;font-size:15px;">{escape(item["source"])}</h4>
                <p class="muted" style="margin:8px 0 4px 0;"><b>Use:</b> {escape(item["use"])}</p>
                <p class="muted" style="margin:0;"><b>Integration:</b> {escape(item["integration"])}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_external_verification_recipe():
    st.markdown("### External Verification Recipe")
    st.markdown(
        """
        1. Collect observations for the app's top 1-3 windows: user rating, visible limiting magnitude, SQM reading if available, and whether the Milky Way or common constellations were visible.
        2. Store the matching forecast row: score, observability, view quality, cloud cover, haze, Moon altitude, Moon illumination, and Bortle/sky brightness.
        3. Compare forecast score to observed outcome using correlation, calibration curves, and confusion matrices for labels such as No-Go/Poor/Marginal/Good.
        4. Fit weights on historical observations, then keep a blind validation set so the model is not graded on the same data used to tune it.
        5. Display confidence separately from score: high confidence requires real seeing/transparency, hourly Moon geometry, low missingness, and observed calibration coverage near that location.
        """
    )


def _score_health_status(value: float, warn_at: float, fail_at: float, higher_is_worse: bool = True) -> str:
    if higher_is_worse:
        if value >= fail_at:
            return "High Risk"
        if value >= warn_at:
            return "Watch"
        return "Good"
    if value <= fail_at:
        return "High Risk"
    if value <= warn_at:
        return "Watch"
    return "Good"


def build_score_credibility_audit(
    validation_df: pd.DataFrame,
    result: dict,
    residual_max: float,
    label_mismatches: int,
) -> tuple:
    if validation_df is None or validation_df.empty:
        return pd.DataFrame(), 0, []

    result = result or {}
    weather_source = str(result.get("weather_source", "Unknown"))
    astronomy_source = str(result.get("astronomy_source", "Unknown"))

    data_cols = [
        c for c in [
            "cloud_value",
            "transparency_value",
            "seeing_value",
            "dewPoint_value",
            "temperature_value",
            "moon_illuminated_pct",
            "moon_meridian_altitude",
        ] if c in validation_df.columns
    ]
    missing_share = (
        validation_df[data_cols].isna().mean().mean()
        if data_cols else 1.0
    )

    dark_share = (
        validation_df.get("is_dark_enough", pd.Series(False, index=validation_df.index))
        .fillna(False)
        .astype(bool)
        .mean()
    )
    poor_or_nogo_share = (
        validation_df["recommendation"].isin(["Poor", "No-Go"]).mean()
        if "recommendation" in validation_df.columns else 1.0
    )
    score_std = validation_df["stargazing_score"].std()
    score_range = (
        validation_df["stargazing_score"].max()
        - validation_df["stargazing_score"].min()
    )
    seeing_std = (
        validation_df["seeing_value"].std()
        if "seeing_value" in validation_df.columns else 0.0
    )
    transparency_std = (
        validation_df["transparency_value"].std()
        if "transparency_value" in validation_df.columns else 0.0
    )
    has_meteoblue = "Meteoblue" in weather_source or (
        "meteoblue_cloud_value" in validation_df.columns
        and validation_df["meteoblue_cloud_value"].notna().any()
    )
    cloud_delta_mean = (
        validation_df["cloud_model_delta"].mean()
        if "cloud_model_delta" in validation_df.columns else None
    )
    transparency_delta_mean = (
        validation_df["transparency_model_delta"].mean()
        if "transparency_model_delta" in validation_df.columns else None
    )

    # This is model confidence, not proof of real-world calibration.
    # Start below 100 until external observations are available.
    credibility = 82.0
    issues = []
    issues.append(
        "No observed SQM, limiting-magnitude, user-rating, or astrophotography outcome dataset is connected yet, so this is not externally calibrated."
    )

    if "Open-Meteo fallback" in weather_source:
        credibility -= 25
        issues.append(
            "Weather source is Open-Meteo fallback, so transparency is visibility-derived and seeing is a neutral placeholder."
        )
    elif weather_source == "Unknown":
        credibility -= 18
        issues.append("Weather source is unknown.")

    if has_meteoblue:
        credibility += 6
    else:
        credibility -= 10
        issues.append("Meteoblue realtime validation is not available for this run.")

    if missing_share > 0.10:
        credibility -= 20
        issues.append(f"{missing_share:.0%} of key scoring inputs are missing before fallback/coercion.")
    elif missing_share > 0.02:
        credibility -= 8
        issues.append(f"{missing_share:.0%} of key scoring inputs are missing before fallback/coercion.")

    if residual_max > 0.001 or label_mismatches:
        credibility -= 30
        issues.append("Stored score or label does not exactly match frontend formula recomputation.")

    if seeing_std < 0.01:
        credibility -= 8
        issues.append("Seeing has almost no variance, which means the model may not be using real seeing forecasts.")

    if transparency_std < 0.01:
        credibility -= 6
        issues.append("Transparency has almost no variance across the forecast window.")

    if dark_share < 0.25:
        credibility -= 8
        issues.append("Most rows are daylight/twilight, so whole-forecast distributions understate usable night quality.")

    if poor_or_nogo_share > 0.90:
        credibility -= 6
        issues.append("The output is highly compressed into Poor/No-Go, limiting ranking usefulness.")

    if score_range < 10:
        credibility -= 6
        issues.append("Scores have a narrow range, so small input errors can change rankings.")

    credibility = int(max(0, min(100, round(credibility))))

    rows = [
        _quality_flag(
            "Formula reproducibility",
            "Good" if residual_max <= 0.001 and label_mismatches == 0 else "High Risk",
            f"Max residual {residual_max:.6f}; label mismatches {int(label_mismatches)}.",
        ),
        _quality_flag(
            "External calibration",
            "Watch",
            "No observed SQM, limiting-magnitude, user-rating, or image-quality validation set is connected yet.",
        ),
        _quality_flag(
            "Weather input credibility",
            "Watch" if "Open-Meteo fallback" in weather_source else "Good",
            f"Weather source: {weather_source}.",
        ),
        _quality_flag(
            "Meteoblue realtime validation",
            "Good" if has_meteoblue else "Watch",
            (
                "Meteoblue cloud/visibility/stability fields are merged into this run. "
                f"Mean cloud model delta: {cloud_delta_mean:.1f}; mean transparency delta: {transparency_delta_mean:.2f}."
                if has_meteoblue and cloud_delta_mean is not None and transparency_delta_mean is not None
                else "No Meteoblue Validation fields were available in this run."
            ),
        ),
        _quality_flag(
            "Astronomy input credibility",
            "Good" if astronomy_source != "Unknown" else "Watch",
            f"Astronomy source: {astronomy_source}.",
        ),
        _quality_flag(
            "Missing input pressure",
            _score_health_status(missing_share, warn_at=0.02, fail_at=0.10),
            f"Key input missingness: {missing_share:.1%}.",
        ),
        _quality_flag(
            "Night-window coverage",
            _score_health_status(dark_share, warn_at=0.35, fail_at=0.20, higher_is_worse=False),
            f"Dark-enough rows: {dark_share:.1%}. Whole-day charts include many intentionally suppressed daylight rows.",
        ),
        _quality_flag(
            "Output spread",
            _score_health_status(score_range, warn_at=20, fail_at=10, higher_is_worse=False),
            f"Score range: {score_range:.1f}; standard deviation: {score_std:.1f}.",
        ),
        _quality_flag(
            "Seeing signal",
            "Watch" if seeing_std < 0.01 else "Good",
            f"Seeing standard deviation: {seeing_std:.3f}.",
        ),
    ]

    return pd.DataFrame(rows), credibility, issues


def build_night_scope_frame(validation_df: pd.DataFrame) -> pd.DataFrame:
    if validation_df is None or validation_df.empty:
        return pd.DataFrame()

    all_rows = validation_df.copy()
    all_rows["score_scope"] = "All forecast hours"
    if "is_dark_enough" not in validation_df.columns:
        return all_rows

    night_rows = validation_df[
        validation_df["is_dark_enough"].fillna(False).astype(bool)
    ].copy()
    night_rows["score_scope"] = "Dark-enough hours only"
    return pd.concat([all_rows, night_rows], ignore_index=True)


def render_score_improvement_guidance(validation_df: pd.DataFrame, credibility_issues: list):
    st.markdown("### Validity Assessment")
    st.markdown(
        """
        The score is internally valid as an arithmetic product of the current inputs.
        It is not yet externally validated against observed sky quality, human observing logs,
        SQM readings, limiting magnitude, or astrophotography outcomes.
        """
    )

    if credibility_issues:
        st.markdown("### Current Weak Points")
        for issue in credibility_issues:
            st.markdown(f"- {issue}")

    st.markdown("### Highest-Impact Improvements")
    st.markdown(
        """
        - Separate forecast-hour scoring from recommendation scoring: rank only dark-enough rows by default, and show daytime rows as suppressed context.
        - Continue replacing fallback seeing/transparency placeholders with astronomy-weather inputs where available, and display lower confidence when using Open-Meteo.
        - Prefer hourly Moon altitude from the position feed for moonlight penalty; fall back to daily meridian altitude only when hourly positions are unavailable.
        - Use smooth cloud transmission instead of hard cloud thresholds so 39% and 41% cloud cover behave similarly.
        - Calibrate weights and thresholds against observed outcomes such as SQM, naked-eye limiting magnitude, user ratings, or archived clear-sky observations.
        - Keep the score split into `observability` for whether the sky is usable and `view quality` for how good it is once usable.
        """
    )

    st.markdown("### External APIs / Packages To Validate Against")
    render_validation_source_cards()
    render_external_verification_recipe()

    if validation_df is not None and not validation_df.empty and "is_dark_enough" in validation_df.columns:
        night_df = validation_df[validation_df["is_dark_enough"].fillna(False).astype(bool)]
        if not night_df.empty:
            st.markdown("### Night-Only Baseline")
            cols = st.columns(4)
            cols[0].metric("Dark rows", f"{len(night_df):,}")
            cols[1].metric("Night median", f"{night_df['stargazing_score'].median():.1f}")
            cols[2].metric("Night peak", f"{night_df['stargazing_score'].max():.1f}")
            cols[3].metric("Night >= 50", f"{(night_df['stargazing_score'] >= 50).mean():.0%}")


def render_score_validation_panel(score_df: pd.DataFrame, bortle_index: int, result: dict = None):
    validation_df = build_score_validation_frame(score_df)
    if validation_df.empty:
        st.warning("Score validation needs scored rows with visibility, atmosphere, and darkness columns.")
        return

    residual_max = validation_df["score_residual"].max()
    residual_mean = validation_df["score_residual"].mean()
    label_mismatches = 0
    if "recommendation" in validation_df.columns:
        label_mismatches = (
            validation_df["recommendation"] != validation_df["expected_recommendation"]
        ).sum()

    audit_df, credibility_score, credibility_issues = build_score_credibility_audit(
        validation_df=validation_df,
        result=result or {},
        residual_max=residual_max,
        label_mismatches=label_mismatches,
    )

    st.markdown(
        """
        The backend score is deterministic:
        `100 * (observability_score / 100)^0.62 * (view_quality_score / 100)^0.38`.
        This panel recomputes that formula in the frontend, then separates formula consistency from
        real-world score credibility.
        """
    )

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Rows checked", f"{len(validation_df):,}")
    m2.metric("Max formula difference", f"{residual_max:.6f}")
    m3.metric("Mean formula difference", f"{residual_mean:.6f}")
    m4.metric("Label mismatches", f"{int(label_mismatches):,}")
    m5.metric("Credibility", f"{credibility_score}/100")

    if residual_max > 0.001 or label_mismatches:
        st.warning(
            "Frontend recomputation differs from the stored backend score or label. "
            "Inspect the residual table below before trusting downstream summaries."
        )
    else:
        st.success("Frontend recomputation matches the backend score and recommendation labels.")

    dist_tab, factor_tab, sensitivity_tab = st.tabs(
        [
            "Score Distributions",
            "Factor Distributions",
            "Bortle Sensitivity",
        ]
    )

    with dist_tab:
        scoped_df = build_night_scope_frame(validation_df)
        c1, c2 = st.columns(2)
        with c1:
            fig_hist = px.histogram(
                scoped_df,
                x="stargazing_score",
                color="score_scope",
                nbins=20,
                barmode="overlay",
                opacity=0.75,
                title="Score Distribution: All Hours vs Dark-Enough Hours",
                labels=labels_for("stargazing_score", "score_scope", "count"),
            )
            fig_hist.update_layout(xaxis_range=[0, 100], bargap=0.04)
            st.plotly_chart(_themed_layout(fig_hist, 430), width="stretch")
        with c2:
            rec_fig = build_recommendation_distribution(validation_df)
            if rec_fig is not None:
                st.plotly_chart(rec_fig, width="stretch")

        c3, c4 = st.columns(2)
        with c3:
            fig_dark = px.box(
                validation_df,
                x="darkness_state",
                y="stargazing_score",
                color="darkness_state",
                points="all",
                title="Score Spread by Darkness State",
                labels=labels_for("darkness_state", "stargazing_score"),
            )
            fig_dark.update_layout(showlegend=False, yaxis_range=[0, 100])
            st.plotly_chart(_themed_layout(fig_dark, 430), width="stretch")
        with c4:
            fig_moon = px.box(
                validation_df,
                x="moon_state",
                y="stargazing_score",
                color="moon_state",
                points="all",
                title="Score Spread by Moon State",
                labels=labels_for("moon_state", "stargazing_score"),
            )
            fig_moon.update_layout(showlegend=False, yaxis_range=[0, 100])
            st.plotly_chart(_themed_layout(fig_moon, 430), width="stretch")

        if "hour" in validation_df.columns:
            hourly = (
                validation_df.dropna(subset=["hour"])
                .groupby("hour", as_index=False)
                .agg(
                    mean_score=("stargazing_score", "mean"),
                    median_score=("stargazing_score", "median"),
                    peak_score=("stargazing_score", "max"),
                )
            )
            fig_hour = px.line(
                hourly,
                x="hour",
                y=["mean_score", "median_score", "peak_score"],
                markers=True,
                title="Hourly Score Distribution Summary",
                labels=labels_for("hour", "value", "factor"),
            )
            fig_hour.update_layout(xaxis=dict(dtick=1), yaxis_range=[0, 100])
            st.plotly_chart(_themed_layout(fig_hour, 450), width="stretch")

    with factor_tab:
        component_cols = [
            c for c in [
                "visibility_penalty",
                "cloud_transmission",
                "darkness_gate",
                "atmospheric_score",
                "effective_darkness",
                "observability_score",
                "view_quality_score",
                "transparency_norm",
                "seeing_norm",
                "humidity_quality",
                "haze_quality",
                "moon_brightness_penalty",
            ] if c in validation_df.columns
        ]
        component_long = validation_df.melt(
            value_vars=component_cols,
            var_name="factor",
            value_name="value",
        ).dropna()
        component_long["factor"] = component_long["factor"].map(human_label)
        fig_components = px.box(
            component_long,
            x="factor",
            y="value",
            color="factor",
            points="all",
            title="Normalized Component Distributions",
            labels=labels_for("factor", "value"),
        )
        fig_components.update_layout(showlegend=False)
        st.plotly_chart(_themed_layout(fig_components, 520), width="stretch")

        if "cloud_band" in validation_df.columns:
            cloud_summary = (
                validation_df.groupby("cloud_band", observed=True)
                .agg(
                    count=("stargazing_score", "size"),
                    mean_score=("stargazing_score", "mean"),
                    peak_score=("stargazing_score", "max"),
                )
                .reset_index()
            )
            fig_cloud = px.bar(
                cloud_summary,
                x="cloud_band",
                y="mean_score",
                color="count",
                hover_data=["count", "peak_score"],
                title="Mean Score by Cloud Cover Band",
                labels=labels_for("cloud_band", "mean_score", "count", "peak_score"),
            )
            fig_cloud.update_layout(yaxis_range=[0, 100])
            st.plotly_chart(_themed_layout(fig_cloud, 450), width="stretch")

    with sensitivity_tab:
        sensitivity_df = build_bortle_sensitivity_frame(validation_df)
        if sensitivity_df.empty:
            st.info("Bortle sensitivity needs moon brightness, visibility, and atmosphere columns.")
        else:
            fig_sensitivity = px.line(
                sensitivity_df,
                x="bortle_index",
                y=["mean_score", "median_score", "peak_score"],
                markers=True,
                title=f"Score Sensitivity to City Lights, Holding This Forecast Fixed (current Bortle {bortle_index})",
                labels=labels_for("bortle_index", "value", "factor"),
            )
            fig_sensitivity.update_layout(xaxis=dict(dtick=1), yaxis_range=[0, 100])
            st.plotly_chart(_themed_layout(fig_sensitivity, 470), width="stretch")

            share_fig = px.bar(
                sensitivity_df,
                x="bortle_index",
                y="excellent_good_share",
                title="Share of Forecast Hours Scoring Good or Excellent by Bortle Index",
                labels=labels_for("bortle_index", "excellent_good_share"),
            )
            share_fig.update_layout(
                xaxis=dict(dtick=1),
                yaxis=dict(tickformat=".0%", range=[0, 1]),
            )
            st.plotly_chart(_themed_layout(share_fig, 390), width="stretch")

            st.dataframe(
                sensitivity_df.style.format(
                    {
                        "mean_score": "{:.1f}",
                        "median_score": "{:.1f}",
                        "peak_score": "{:.1f}",
                        "excellent_good_share": "{:.0%}",
                    }
                ),
                width="stretch",
                hide_index=True,
            )


def render_source_badges(result):
    weather_source  = str(result.get("weather_source", "Unknown")).replace(
        "Meteoblue validation",
        "Meteoblue Validation",
    )
    astronomy_source = result.get("astronomy_source", "Unknown")
    timezone_value  = result.get("timezone", "Unknown")
    w_cls = "warning-pill" if "fallback" in weather_source.lower() else "source-pill"
    a_cls = "warning-pill" if "fallback" in astronomy_source.lower() else "source-pill"
    st.markdown(
        f'<div class="inline-cell">'
        f'<span class="{w_cls}">Weather: {weather_source}</span>'
        f'<span class="{a_cls}">Astronomy: {astronomy_source}</span>'
        f'<span class="source-pill">Timezone: {timezone_value}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_window_card(rank, row):
    score       = safe_numeric(row.get("stargazing_score"), 0)
    label       = row.get("recommendation", "N/A")
    time_label  = row.get("time_label", "N/A")
    explanation = explain_row(row)
    cluster_label = row.get("cluster_label", "")
    cluster_desc  = row.get("cluster_description", "")
    cluster_html  = ""
    if cluster_label and cluster_label != "Unknown":
        cluster_html = (
            f'<div><span class="cluster-badge">{cluster_label}</span></div>'
            f'<p class="cluster-desc">{cluster_desc}</p>'
        )
    st.markdown(
        f"""
        <div class="window-card">
            <div class="window-rank">#{rank} Recommended Window</div>
            <div class="window-time" style="font-family: 'DM Serif Display', Georgia, serif; font-weight: 400; font-size:1.15rem; color:#e4dff0; margin-bottom:6px;">{time_label}</div>
            <div class="window-score" style="font-family: 'DM Serif Display', Georgia, serif; font-weight: 400; font-size:1.9rem; color:#e4dff0; line-height:1.1;">{score:.1f}</div>
            <span class="badge {badge_class(label)}">{label}</span>
            <p class="muted" style="margin-top:12px;">{explanation}</p>
            {cluster_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _score_color(score: float):
    score = safe_numeric(score, 0) or 0
    if score >= 85:
        return [74, 222, 128, 190]
    if score >= 70:
        return [94, 234, 212, 185]
    if score >= 50:
        return [251, 191, 36, 180]
    if score >= 25:
        return [251, 146, 60, 175]
    return [248, 113, 113, 170]


def _circle_polygon(lat: float, lon: float, radius_km: float, steps: int = 96):
    coords = []
    earth_radius_km = 6371.0
    lat1 = math.radians(lat)
    lon1 = math.radians(lon)
    angular_distance = radius_km / earth_radius_km

    for i in range(steps):
        bearing = math.radians((360 / steps) * i)
        lat2 = math.asin(
            math.sin(lat1) * math.cos(angular_distance)
            + math.cos(lat1) * math.sin(angular_distance) * math.cos(bearing)
        )
        lon2 = lon1 + math.atan2(
            math.sin(bearing) * math.sin(angular_distance) * math.cos(lat1),
            math.cos(angular_distance) - math.sin(lat1) * math.sin(lat2),
        )
        coords.append([((math.degrees(lon2) + 540) % 360) - 180, math.degrees(lat2)])

    return coords


def render_travel_map(search_result: dict, current_lat: float, current_lon: float):
    if pdk is None:
        st.info("Map unavailable because pydeck is not installed in this environment.")
        return

    candidates = search_result.get("candidates", [])
    if not candidates:
        return

    candidate_rows = []
    for item in candidates:
        score = safe_numeric(item.get("best_score"), 0) or 0
        destination_name = item.get("destination_name") or "Coordinate candidate"
        candidate_rows.append(
            {
                "lat": item.get("lat"),
                "lon": item.get("lon"),
                "score": score,
                "time_label": item.get("time_label", "Unknown"),
                "recommendation": item.get("recommendation", "Unknown"),
                "distance_km": item.get("distance_km", 0),
                "destination_name": destination_name,
                "color": _score_color(score),
                "radius": 450 + max(score, 0) * 18,
            }
        )

    candidate_df = pd.DataFrame(candidate_rows).dropna(subset=["lat", "lon"])
    layers = []

    radius_km = safe_numeric(search_result.get("radius_km"), 0)
    if radius_km:
        layers.append(
            pdk.Layer(
                "PolygonLayer",
                data=[{"polygon": _circle_polygon(current_lat, current_lon, radius_km)}],
                get_polygon="polygon",
                get_fill_color=[196, 120, 210, 20],
                get_line_color=[196, 120, 210, 130],
                line_width_min_pixels=2,
                stroked=True,
                filled=True,
            )
        )

    layers.append(
        pdk.Layer(
            "ScatterplotLayer",
            data=pd.DataFrame([{
                "lat": current_lat,
                "lon": current_lon,
                "label": "Current location",
                "color": [228, 223, 240, 230],
                "radius": 900,
            }]),
            id="current-location",
            get_position="[lon, lat]",
            get_color="color",
            get_radius="radius",
            pickable=True,
        )
    )

    if not candidate_df.empty:
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=candidate_df,
                id="candidate-points",
                get_position="[lon, lat]",
                get_color="color",
                get_radius="radius",
                pickable=True,
                auto_highlight=True,
            )
        )

    destination = search_result.get("destination")
    if destination:
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=pd.DataFrame([{
                    "lat": destination.get("lat"),
                    "lon": destination.get("lon"),
                    "destination_name": destination.get("name", "Selected destination"),
                    "score": search_result.get("best_candidate", {}).get("best_score", 0),
                    "time_label": search_result.get("best_candidate", {}).get("time_label", "Unknown"),
                    "recommendation": search_result.get("best_candidate", {}).get("recommendation", "Unknown"),
                    "distance_km": search_result.get("best_candidate", {}).get("distance_km", 0),
                    "color": [255, 255, 255, 245],
                    "radius": 1200,
                }]).dropna(subset=["lat", "lon"]),
                id="selected-destination",
                get_position="[lon, lat]",
                get_color="color",
                get_radius="radius",
                pickable=True,
                auto_highlight=True,
            )
        )

    chart = pdk.Deck(
        map_style=None,
        initial_view_state=pdk.ViewState(
            latitude=current_lat,
            longitude=current_lon,
            zoom=8,
            pitch=0,
        ),
        layers=layers,
        tooltip={
            "html": (
                "<b>{destination_name}</b><br/>"
                "Score: {score}<br/>"
                "Window: {time_label}<br/>"
                "Label: {recommendation}<br/>"
                "Distance: {distance_km} km"
            ),
            "style": {"backgroundColor": "#100d1e", "color": "#e4dff0"},
        },
    )
    st.pydeck_chart(chart, width="stretch", height=460)


def render_score_explanation(row):
    score = safe_numeric(row.get("stargazing_score"), 0) or 0
    visibility = safe_numeric(row.get("visibility_penalty"), 0)
    atmospheric = safe_numeric(row.get("atmospheric_score"), 0)
    darkness = safe_numeric(row.get("effective_darkness"), 0)
    cloud = safe_numeric(row.get("cloud_value"), None)
    transparency = safe_numeric(row.get("transparency_value"), None)
    seeing = safe_numeric(row.get("seeing_value"), None)
    moon = safe_numeric(row.get("moon_illuminated_pct"), None)
    bortle = safe_numeric(row.get("light_pollution_factor"), None)
    bortle_text = f"{bortle * 9:.1f}" if bortle is not None else "N/A"

    rows = [
        ("Final score", f"{score:.1f}/100", "The overall score used to rank the window."),
        ("Visibility penalty", f"{visibility:.2f}" if visibility is not None else "N/A", "Clouds and daylight can sharply reduce the score."),
        ("Atmospheric quality", f"{atmospheric:.2f}" if atmospheric is not None else "N/A", "Transparency, seeing, and humidity shape clarity."),
        ("Effective darkness", f"{darkness:.2f}" if darkness is not None else "N/A", "Moonlight and Bortle/light pollution reduce contrast."),
        ("Cloud cover", f"{cloud:.0f}" if cloud is not None else "N/A", "Lower is better."),
        ("Transparency / Seeing", f"{transparency} / {seeing}", "Lower raw values are better in the forecast feed."),
        ("Moon illumination", f"{moon:.0f}%" if moon is not None else "N/A", "Brighter moonlight washes out faint targets."),
        ("Estimated Bortle", bortle_text, "User-selected city lights index converted into the score."),
    ]

    html_rows = "".join(
        f"""
        <div class="inline-cell">
            <div class="metric-card-title">{escape(label)}</div>
            <div style="color:#e4dff0;font-size:1.05rem;font-weight:600;">{escape(value)}</div>
            <p class="muted" style="margin:6px 0 0 0;">{escape(desc)}</p>
        </div>
        """
        for label, value, desc in rows
    )

    st.markdown(
        f"""
        <div class="section-card">
            <h3 class="section-heading" style="font-family: 'DM Serif Display', Georgia, serif; font-weight: 400; font-size:20px; color:#e4dff0; margin:0 0 8px 0;">Why This Score?</h3>
            <p class="muted">This panel explains the top window using existing model columns. It does not change the score.</p>
            <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px;margin-top:12px;">
                {html_rows}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# SKY CLOCK + NIGHT TIMELINE CARDS
# ============================================================

def _sky_clock_card_html(city_name: str = "New York City") -> str:
    """Animated polar sky-clock card with moon position + legend."""
    now    = datetime.now()
    hour_f = now.hour + now.minute / 60.0

    # Approximate moon trajectory (spring, ~NYC)
    moon_rise = 18.8
    moon_set  = 27.2   # 3:12 am next day
    progress  = max(0.0, min(1.0, (hour_f - moon_rise) / (moon_set - moon_rise)))
    moon_above = moon_rise <= hour_f <= moon_set

    moon_az  = 80 + progress * 200
    moon_alt = max(0.0, 52.0 * math.sin(progress * math.pi))
    moon_r_p = (1 - moon_alt / 90.0) * 88
    moon_ang = (moon_az - 90) * math.pi / 180
    moon_x   = 110 + moon_r_p * math.cos(moon_ang)
    moon_y   = 110 + moon_r_p * math.sin(moon_ang)

    moon_az_disp  = int(moon_az % 360)
    moon_alt_disp = int(moon_alt)
    ms_h = int(moon_set) % 24
    ms_m = int((moon_set % 1) * 60)
    moonset_str = f"{ms_h % 12 or 12}:{ms_m:02d} {'am' if ms_h < 12 else 'pm'}"

    rng = random.Random(7)
    star_els = []
    for _ in range(72):
        az_s = rng.uniform(0, 2 * math.pi)
        r_s  = rng.uniform(0, 1) ** 0.5 * 85
        sx   = 110 + r_s * math.cos(az_s)
        sy   = 110 + r_s * math.sin(az_s)
        sr   = round(rng.uniform(0.3, 1.4), 1)
        op   = round(rng.uniform(0.12, 0.68), 2)
        op2  = round(min(op + rng.uniform(0.15, 0.35), 0.92), 2)
        dur  = round(rng.uniform(1.5, 4.2), 1)
        beg  = round(rng.uniform(0, dur), 1)
        star_els.append(
            f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="{sr}" fill="#e4dff0" opacity="{op}">'
            f'<animate attributeName="opacity" values="{op};{op2};{op}" '
            f'dur="{dur}s" begin="{beg}s" repeatCount="indefinite"/>'
            f'</circle>'
        )

    trail_pts = []
    p0 = max(0.0, progress - 0.28)
    for i in range(13):
        p     = p0 + (progress - p0) * i / 12
        az_t  = 80 + p * 200
        alt_t = max(0.0, 52.0 * math.sin(p * math.pi))
        if alt_t <= 0:
            continue
        r_t  = (1 - alt_t / 90.0) * 88
        an_t = (az_t - 90) * math.pi / 180
        trail_pts.append(f"{110 + r_t * math.cos(an_t):.1f},{110 + r_t * math.sin(an_t):.1f}")
    trail_svg = (
        f'<polyline points="{" ".join(trail_pts)}" fill="none" stroke="#c478d2" '
        f'stroke-width="1" stroke-opacity="0.28" stroke-dasharray="3 4"/>'
        if len(trail_pts) >= 2 else ""
    )

    moon_svg = ""
    if moon_above:
        moon_svg = (
            f'<circle cx="{moon_x:.1f}" cy="{moon_y:.1f}" r="16" fill="#c478d2" fill-opacity="0.06"/>'
            f'<circle cx="{moon_x:.1f}" cy="{moon_y:.1f}" r="9"  fill="#c478d2" fill-opacity="0.14"/>'
            f'<circle cx="{moon_x:.1f}" cy="{moon_y:.1f}" r="5.5" fill="#c478d2" fill-opacity="0.82">'
            f'<animate attributeName="fill-opacity" values="0.82;0.50;0.82" dur="3.2s" repeatCount="indefinite"/>'
            f'</circle>'
            f'<circle cx="{moon_x - 1.5:.1f}" cy="{moon_y - 1.5:.1f}" r="1.6" fill="white" fill-opacity="0.42"/>'
        )

    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="220" height="220" viewBox="0 0 220 220">'
        '<defs>'
        '<radialGradient id="skyBgNd" cx="50%" cy="50%" r="50%">'
        '<stop offset="0%"   stop-color="#c478d2" stop-opacity="0.05"/>'
        '<stop offset="60%"  stop-color="#090614" stop-opacity="0.95"/>'
        '<stop offset="100%" stop-color="#090614" stop-opacity="1"/>'
        '</radialGradient>'
        '<clipPath id="skyClipNd"><circle cx="110" cy="110" r="100"/></clipPath>'
        '</defs>'
        '<circle cx="110" cy="110" r="108" fill="none" stroke="rgba(196,120,210,0.16)" stroke-width="1.5"/>'
        '<circle cx="110" cy="110" r="100" fill="url(#skyBgNd)"/>'
        '<circle cx="110" cy="110" r="33" fill="none" stroke="rgba(196,120,210,0.10)" stroke-width="0.7" stroke-dasharray="3 5"/>'
        '<circle cx="110" cy="110" r="66" fill="none" stroke="rgba(196,120,210,0.10)" stroke-width="0.7" stroke-dasharray="3 5"/>'
        '<circle cx="110" cy="110" r="99" fill="none" stroke="rgba(196,120,210,0.10)" stroke-width="0.7" stroke-dasharray="3 5"/>'
        '<line x1="110" y1="12" x2="110" y2="208" stroke="rgba(196,120,210,0.07)" stroke-width="0.5"/>'
        '<line x1="12" y1="110" x2="208" y2="110" stroke="rgba(196,120,210,0.07)" stroke-width="0.5"/>'
        f'<g clip-path="url(#skyClipNd)">{"".join(star_els)}</g>'
        f'{trail_svg}'
        '<text x="110" y="17"  text-anchor="middle" dominant-baseline="middle" font-size="9" font-weight="700" fill="#8880a0" font-family="DM Sans,sans-serif">N</text>'
        '<text x="203" y="110" text-anchor="middle" dominant-baseline="middle" font-size="9" font-weight="700" fill="#8880a0" font-family="DM Sans,sans-serif">E</text>'
        '<text x="110" y="203" text-anchor="middle" dominant-baseline="middle" font-size="9" font-weight="700" fill="#8880a0" font-family="DM Sans,sans-serif">S</text>'
        '<text x="17"  y="110" text-anchor="middle" dominant-baseline="middle" font-size="9" font-weight="700" fill="#8880a0" font-family="DM Sans,sans-serif">W</text>'
        '<circle cx="110" cy="110" r="2.5" fill="#554f6a"/>'
        f'{moon_svg}'
        '</svg>'
    )

    legend_items = [
        ("Moon azimuth",  f"{moon_az_disp}°",        False),
        ("Moon altitude", f"{moon_alt_disp}°",       False),
        None,
        ("Moon phase",    "12% Waxing Crescent",     True),
        ("Moonset",       moonset_str,                False),
        ("Bortle class",  "5 — Suburban",             False),
    ]
    rows_html = ""
    for item in legend_items:
        if item is None:
            rows_html += '<div style="height:1px;background:rgba(196,120,210,0.13);margin:6px 0;"></div>'
        else:
            lbl, val, accent_val = item
            val_color = "#c478d2" if accent_val else "#e4dff0"
            rows_html += (
                f'<div style="display:flex;justify-content:space-between;'
                f'align-items:baseline;margin-bottom:8px;">'
                f'<span style="font-size:12px;color:#8880a0;">{lbl}</span>'
                f'<span style="font-size:13px;color:{val_color};font-weight:600;">{val}</span>'
                f'</div>'
            )

    best_window_hour = 22.0
    adj = hour_f if hour_f <= best_window_hour else hour_f - 24
    mins_until = max(0, int((best_window_hour - adj) * 60))
    countdown_html = (
        f'<span style="font-size:12px;color:#c478d2;">'
        f'★ Best window opens in <strong>{mins_until} min</strong> — run pipeline for details'
        f'</span>'
        if mins_until > 0 else
        f'<span style="font-size:12px;color:#c478d2;">'
        f'★ Prime observing window active now — run pipeline for details'
        f'</span>'
    )

    return (
        '<div class="section-card" style="display:flex;gap:28px;align-items:center;">'
        f'<div style="flex-shrink:0;">{svg}</div>'
        '<div style="flex:1;">'
        '<span class="section-label">Sky Now · ' + city_name + '</span>'
        + rows_html +
        '<div style="margin-top:12px;padding:8px 12px;border-radius:8px;'
        'background:rgba(196,120,210,0.10);border:1px solid rgba(196,120,210,0.20);">'
        + countdown_html +
        '</div>'
        '</div>'
        '</div>'
    )


def _night_timeline_card_html() -> str:
    """Dusk-to-dawn phase bar with animated NOW cursor."""
    now    = datetime.now()
    hour_f = now.hour + now.minute / 60.0

    t_start = 19.7
    t_end   = 29.03
    adj_h   = hour_f if hour_f >= t_start else hour_f + 24
    now_pct = max(0.02, min(0.97, (adj_h - t_start) / (t_end - t_start)))

    phases = [
        ("☀",  "Sunset",      "7:42 pm", 0.00),
        (None, "Civil",       "8:09 pm", 0.12),
        (None, "Nautical",    "8:38 pm", 0.22),
        ("★",  "Astro Dark",  "9:11 pm", 0.33),
        ("◎",  "Best Window", "10–11 pm",0.50),
        (None, "Dawn",        "5:02 am", 1.00),
    ]

    ticks_html = "".join(
        f'<div style="position:absolute;top:0;left:{p[3]*100:.1f}%;'
        f'width:1px;height:100%;background:rgba(196,120,210,0.22);"></div>'
        for p in phases
    )

    labels_html = ""
    for i, p in enumerate(phases):
        icon, label, time_s, pct = p
        align     = "flex-end" if i == len(phases) - 1 else "center"
        transform = "translateX(0%)" if i == len(phases) - 1 else "translateX(-50%)"
        lbl_color = "#e4dff0" if icon else "#8880a0"
        lbl_weight= "600" if icon else "400"
        icon_html = (
            f'<span style="font-size:10px;color:#c478d2;margin-bottom:1px;">{icon}</span>'
            if icon else ""
        )
        labels_html += (
            f'<div style="position:absolute;left:{pct*100:.1f}%;transform:{transform};'
            f'display:flex;flex-direction:column;align-items:{align};gap:1px;">'
            f'{icon_html}'
            f'<span style="font-size:10px;color:{lbl_color};font-weight:{lbl_weight};'
            f'white-space:nowrap;">{label}</span>'
            f'<span style="font-size:9px;color:#554f6a;white-space:nowrap;">{time_s}</span>'
            f'</div>'
        )

    return (
        '<style>'
        '@keyframes nd-now-glow {'
        '0%,100%{box-shadow:0 0 0 2px rgba(196,120,210,0.0),0 0 10px #c478d2;}'
        '50%{box-shadow:0 0 0 4px rgba(196,120,210,0.22),0 0 20px #c478d2;}'
        '}'
        '</style>'
        '<div class="section-card">'
        '<span class="section-label">Tonight\'s Sky Timeline</span>'
        '<div style="position:relative;margin-bottom:34px;">'
        '<div style="height:10px;border-radius:99px;'
        'background:linear-gradient(to right,'
        'rgba(224,130,64,0.20) 0%,rgba(80,96,192,0.38) 12%,'
        'rgba(10,10,40,0.88) 30%,rgba(5,5,20,0.96) 55%,'
        'rgba(10,10,40,0.88) 75%,rgba(80,96,192,0.38) 88%,'
        'rgba(224,130,64,0.20) 100%);'
        'border:1px solid rgba(196,120,210,0.14);"></div>'
        '<div style="position:absolute;top:0;left:48%;width:14%;height:100%;'
        'border-radius:99px;background:rgba(196,120,210,0.20);'
        'border:1px solid rgba(196,120,210,0.38);"></div>'
        f'<div style="position:absolute;inset:0;">{ticks_html}</div>'
        f'<div style="position:absolute;top:-6px;left:{now_pct*100:.1f}%;'
        'transform:translateX(-50%);'
        'display:flex;flex-direction:column;align-items:center;gap:2px;">'
        '<div style="width:14px;height:14px;border-radius:50%;'
        'background:#c478d2;border:2px solid #090614;'
        'animation:nd-now-glow 2s ease-in-out infinite;"></div>'
        '<div style="width:1px;height:22px;'
        'background:linear-gradient(#c478d2,transparent);"></div>'
        '<span style="font-size:9px;color:#c478d2;font-weight:700;'
        'white-space:nowrap;margin-top:-2px;">NOW</span>'
        '</div>'
        '</div>'
        f'<div style="position:relative;height:48px;">{labels_html}</div>'
        '</div>'
    )


# ============================================================
# RUN PIPELINE
# ============================================================

if run_button:
    with st.spinner("Fetching live weather and astronomy data..."):
        try:
            result = cached_run_pipeline(
                city_name=city_name,
                lat=lat, lon=lon,
                timezone=timezone,
                days=days,
                bortle_index=bortle_index,
                include_tad=include_tad,
                include_positions=include_positions,
            )
            st.session_state["pipeline_result"] = result
            st.session_state["pipeline_bortle"] = bortle_index
        except Exception as e:
            st.error("The live scoring pipeline failed.")
            st.exception(e)
            st.stop()


# Apply neutral background before pipeline data exists.
if "pipeline_result" not in st.session_state:
    apply_reactive_sky_style("neutral")


if "pipeline_result" not in st.session_state:
    # ── Hero ─────────────────────────────────────────────────────
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-flex">
                <div class="hero-left">
                    <div class="hero-icon-wrap">🔭</div>
                    <div>
                        <h1 class="hero-title" style="font-family:'DM Serif Display',Georgia,serif;
                            font-weight:400;font-size:2rem;color:#e4dff0;line-height:1.15;margin:0 0 5px 0;">
                            Stargazing Assistant
                        </h1>
                        <p class="hero-desc">
                            Find tonight's best observing window using live weather, astronomy,
                            and light-pollution data — in one click.
                        </p>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── 3-step guide ─────────────────────────────────────────────
    st.markdown(
        """
        <style>
        .steps-row {
            display: flex;
            gap: 12px;
            margin-bottom: 20px;
        }
        .step-card {
            flex: 1;
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 18px 20px;
            display: flex;
            align-items: flex-start;
            gap: 14px;
        }
        .step-num {
            width: 28px; height: 28px;
            border-radius: 50%;
            background: var(--accent-soft);
            border: 1px solid rgba(196,120,210,0.35);
            color: var(--accent);
            font-size: 13px;
            font-weight: 700;
            display: flex; align-items: center; justify-content: center;
            flex-shrink: 0;
            font-family: 'DM Sans', sans-serif;
        }
        .step-body {}
        .step-title {
            font-size: 13px;
            font-weight: 600;
            color: var(--text);
            margin-bottom: 3px;
            font-family: 'DM Sans', sans-serif;
        }
        .step-desc {
            font-size: 12px;
            color: var(--text-muted);
            line-height: 1.55;
            font-family: 'DM Sans', sans-serif;
        }
        .cta-row {
            display: flex;
            align-items: center;
            gap: 14px;
            margin-bottom: 20px;
        }
        .cta-hint {
            font-size: 12px;
            color: var(--text-dim);
            font-family: 'DM Sans', sans-serif;
        }
        </style>
        <div class="steps-row">
            <div class="step-card">
                <div class="step-num">1</div>
                <div class="step-body">
                    <div class="step-title">Pick a location</div>
                    <div class="step-desc">Choose a city preset or enter custom coordinates in the sidebar.</div>
                </div>
            </div>
            <div class="step-card">
                <div class="step-num">2</div>
                <div class="step-body">
                    <div class="step-title">Configure options</div>
                    <div class="step-desc">Set forecast range, Bortle index, and toggle optional data sources.</div>
                </div>
            </div>
            <div class="step-card">
                <div class="step-num">3</div>
                <div class="step-body">
                    <div class="step-title">Run the pipeline</div>
                    <div class="step-desc">Hit the button below — results appear in seconds.</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── CTA button ───────────────────────────────────────────────
    st.markdown(
        """
        <style>
        /* Make the primary button extra prominent in empty state */
        div[data-testid="stButton"].nd-cta > button {
            background: var(--accent-soft) !important;
            border: 1px solid rgba(196,120,210,0.55) !important;
            color: var(--accent) !important;
            font-size: 15px !important;
            font-weight: 700 !important;
            padding: 14px 36px !important;
            border-radius: 10px !important;
            letter-spacing: 0.02em;
            box-shadow: 0 0 28px rgba(196,120,210,0.12);
            transition: background 0.15s, box-shadow 0.15s !important;
        }
        div[data-testid="stButton"].nd-cta > button:hover {
            background: rgba(196,120,210,0.28) !important;
            box-shadow: 0 0 40px rgba(196,120,210,0.22) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    col_btn, col_hint = st.columns([2, 5])
    with col_btn:
        page_run = st.button(
            "▶  Run Pipeline",
            type="primary",
            use_container_width=True,
            key="page_run_btn",
        )
    with col_hint:
        st.markdown(
            f'<p class="cta-hint" style="padding-top:10px;">'
            f'Currently set to <strong style="color:#e4dff0;">{city_name}</strong> · '
            f'Bortle {bortle_index} · {days}-day forecast'
            f'</p>',
            unsafe_allow_html=True,
        )

    if page_run:
        with st.spinner("Fetching live weather and astronomy data..."):
            try:
                result = cached_run_pipeline(
                    city_name=city_name,
                    lat=lat, lon=lon,
                    timezone=timezone,
                    days=days,
                    bortle_index=bortle_index,
                    include_tad=include_tad,
                    include_positions=include_positions,
                )
                st.session_state["pipeline_result"] = result
                st.session_state["pipeline_bortle"] = bortle_index
                st.rerun()
            except Exception as e:
                st.error("The pipeline failed.")
                st.exception(e)
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(_sky_clock_card_html(city_name), unsafe_allow_html=True)
    with col_b:
        st.markdown(_night_timeline_card_html(), unsafe_allow_html=True)
    st.stop()


result       = st.session_state["pipeline_result"]
bortle_index = st.session_state.get("pipeline_bortle", bortle_index)


# ============================================================
# LOAD RESULTS
# ============================================================

score_df      = result.get("score_df",      pd.DataFrame())
top_windows   = result.get("top_windows",   pd.DataFrame())
daily_summary = result.get("daily_summary", pd.DataFrame())
master_df     = result.get("master_df",     pd.DataFrame())
weather_df    = result.get("weather_df",    pd.DataFrame())
ip_geo_df     = result.get("ip_geo_df",     pd.DataFrame())
event_df      = result.get("event_df",      pd.DataFrame())
position_df   = result.get("position_df",   pd.DataFrame())
telemetry     = result.get("telemetry",     {})

if top_windows is None or top_windows.empty:
    display_empty_result_debug(score_df, top_windows)

sky_state = infer_reactive_sky_state(score_df, top_windows)
apply_reactive_sky_style(sky_state)

if include_positions and (position_df is None or position_df.empty):
    with st.spinner("Fetching Sun/Moon position data for visualization..."):
        try:
            position_df = cached_fetch_positions(
                lat=result["lat"],
                lon=result["lon"],
                timezone=result.get("timezone", timezone or "UTC"),
                days=days,
            )
        except Exception as e:
            st.warning(f"Sun/Moon position visualization unavailable: {e}")
            position_df = pd.DataFrame()

celestial_df = build_celestial_recommendations(
    score_df=score_df,
    event_df=event_df,
    position_df=position_df,
    latitude=result.get("lat", 0.0),
    timezone_name=result.get("timezone", timezone or "UTC"),
    top_n=5,
)


# ============================================================
# HERO CARD
# ============================================================

best_row   = top_windows.iloc[0]
best_score = safe_numeric(best_row.get("stargazing_score"), 0)
best_time  = best_row.get("time_label", "N/A")
best_label = best_row.get("recommendation", "N/A")
score_10   = best_score / 10

st.markdown(
    f"""
    <div class="hero-card">
        <div class="hero-flex">
            <div class="hero-left">
                <div class="hero-icon-wrap">🌌</div>
                <div>
                    <h1 class="hero-title" style="font-family: 'DM Serif Display', Georgia, serif; font-weight: 400; font-size:2rem; color:#e4dff0; line-height:1.15; margin:0 0 5px 0;">Stargazing Assistant</h1>
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
    <p class="info-note">
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
            <span class="metric-large" style="font-family: 'DM Serif Display', Georgia, serif; font-weight: 400; font-size:2rem; color:#e4dff0; line-height:1;">{best_score:.1f}</span>
            <span class="metric-unit">/100</span>
        </div>
        """, unsafe_allow_html=True,
    )
with col2:
    st.markdown(
        f"""
        <div class="metric-card">
            <span class="metric-card-title">Best Time</span>
            <div class="metric-large" style="font-size:1.45rem;" style="font-family: 'DM Serif Display', Georgia, serif; font-weight: 400; font-size:1.45rem; color:#e4dff0; line-height:1;">{best_time}</div>
        </div>
        """, unsafe_allow_html=True,
    )
with col3:
    st.markdown(
        f"""
        <div class="metric-card">
            <span class="metric-card-title">Recommendation</span>
            <span class="badge {badge_class(best_label)}">{best_label}</span>
        </div>
        """, unsafe_allow_html=True,
    )
with col4:
    st.markdown(
        f"""
        <div class="metric-card">
            <span class="metric-card-title">Forecast Hours</span>
            <span class="metric-large" style="font-family: 'DM Serif Display', Georgia, serif; font-weight: 400; font-size:2rem; color:#e4dff0; line-height:1;">{len(score_df)}</span>
        </div>
        """, unsafe_allow_html=True,
    )

st.divider()


# ============================================================
# DASHBOARD SECTIONS
# ============================================================

def _section_open(name: str) -> bool:
    return selected_page == "All Sections" or selected_page == name


if selected_page != "All Sections":
    st.caption(f"Focused section: {selected_page}")

with st.expander("Overview", expanded=_section_open("Overview")):
    st.markdown(
        f"""
        <div class="section-card">
            <h3 class="section-heading" style="font-family: 'DM Serif Display', Georgia, serif; font-weight: 400; font-size:20px; color:#e4dff0; margin:0 0 8px 0;">Best Observing Window</h3>
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
    cols = st.columns(3)
    for i, (_, row) in enumerate(top_windows.head(3).iterrows()):
        with cols[i]:
            render_window_card(i + 1, row)
    if celestial_df is not None and not celestial_df.empty:
        st.markdown(
            """
            <div class="section-card">
                <h3 class="section-heading" style="font-family: 'DM Serif Display', Georgia, serif; font-weight: 400; font-size:20px; color:#e4dff0; margin:0 0 8px 0;">Top Celestial Phenomena</h3>
                <p class="muted">Best data-backed celestial opportunities for this forecast.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        ccols = st.columns(3)
        for i, (_, row) in enumerate(celestial_df.head(3).iterrows()):
            with ccols[i]:
                render_celestial_card(i + 1, row)

with st.expander("Best Windows", expanded=_section_open("Best Windows")):
    cols = st.columns(3)
    for i, (_, row) in enumerate(top_windows.head(3).iterrows()):
        with cols[i]:
            render_window_card(i + 1, row)
    render_score_explanation(best_row)
    fig_top = px.bar(
        top_windows.sort_values("stargazing_score"),
        x="stargazing_score", y="time_label",
        orientation="h", color="recommendation",
        hover_data=[c for c in [
            "cloud_value","transparency_value","seeing_value","moon_illuminated_pct",
            "is_dark_enough","is_moon_up","visibility_penalty","cloud_transmission",
            "effective_darkness","atmospheric_score","haze_quality",
            "observability_score","view_quality_score",
        ] if c in top_windows.columns],
        title="Top 10 Stargazing Windows",
        labels=labels_for(
            "stargazing_score", "time_label", "recommendation", "cloud_value",
            "transparency_value", "seeing_value", "moon_illuminated_pct",
            "is_dark_enough", "is_moon_up", "visibility_penalty",
            "cloud_transmission", "effective_darkness", "atmospheric_score",
            "haze_quality", "observability_score", "view_quality_score",
        ),
    )
    fig_top.update_layout(xaxis_range=[0, 100])
    st.plotly_chart(_themed_layout(fig_top, 520), width="stretch")

with st.expander("Sky Conditions", expanded=_section_open("Sky Conditions")):
    st.plotly_chart(build_factor_chart(score_df), width="stretch")
    feature_options = [c for c in [
        "cloud_value","transparency_value","seeing_value","moon_illuminated_pct",
        "visibility_penalty","cloud_transmission","darkness_gate",
        "transparency_norm","seeing_norm","humidity_quality","haze_quality",
        "moon_brightness_penalty","effective_darkness","atmospheric_score",
        "observability_score","view_quality_score","legacy_stargazing_score","stargazing_score",
        "aerosol_optical_depth","pm2_5","dust",
        "meteoblue_cloud_value","meteoblue_low_clouds","meteoblue_mid_clouds",
        "meteoblue_high_clouds","meteoblue_transparency_value",
        "meteoblue_seeing_proxy_value","meteoblue_relativehumidity",
        "meteoblue_fog_probability","cloud_model_delta","transparency_model_delta",
    ] if c in score_df.columns]
    if feature_options:
        feature_label_to_code = {human_label(c): c for c in feature_options}
        selected_feature_label = st.selectbox(
            "Inspect one feature",
            list(feature_label_to_code.keys()),
            key="dashboard_feature_selector",
        )
        selected_feature = feature_label_to_code[selected_feature_label]
        plot_df = score_df.copy()
        plot_df["local_dt"] = pd.to_datetime(plot_df["local_dt"], errors="coerce")
        fig_feature = px.line(
            plot_df,
            x="local_dt",
            y=selected_feature,
            markers=True,
            title=f"{selected_feature_label} Over Time",
            labels=labels_for("local_dt", selected_feature),
        )
        st.plotly_chart(_themed_layout(fig_feature, 480), width="stretch")

with st.expander("Score Validation", expanded=_section_open("Score Validation")):
    render_score_validation_panel(score_df, bortle_index, result)

with st.expander("Sky Path", expanded=_section_open("Sky Path")):
    st.caption("This visualization uses Sun/Moon position data only. It does not affect the recommendation score.")
    if not include_positions:
        st.info("Turn on 'Use Sun/Moon position data' in the sidebar, then run again.")
    elif position_df is None or position_df.empty:
        st.warning(
            "No Sun/Moon position data was returned for this location/time window. "
            "This can happen when the upstream astronomy provider has sparse coverage."
        )
    else:
        required_cols = ["Azimuth (°)", "Altitude (°)", "Object"]
        if "Data Source" in position_df.columns and (position_df["Data Source"] == "Estimated fallback").any():
            st.info(
                "Timeanddate position feed is unavailable for this location right now. "
                "Showing estimated Sun/Moon path for continuity."
            )
        if all(c in position_df.columns for c in required_cols):
            pos_plot_df = position_df.dropna(subset=["Azimuth (°)", "Altitude (°)"])
            if not pos_plot_df.empty:
                fig_pos = px.scatter_polar(
                    pos_plot_df,
                    r="Altitude (°)",
                    theta="Azimuth (°)",
                    color="Object",
                    hover_data=[c for c in ["Hour", "Illuminated (%)", "Moon Phase"] if c in pos_plot_df.columns],
                    title="Sun and Moon Position in the Sky",
                    labels=labels_for("Object", "Hour"),
                )
                st.plotly_chart(_themed_layout(fig_pos, 650), width="stretch")
            else:
                st.warning("Position data exists, but altitude/azimuth values are missing.")
        else:
            st.warning("Position data does not contain required columns: Azimuth (°), Altitude (°), Object.")

with st.expander("Celestial Phenomena", expanded=_section_open("Celestial Phenomena")):
    render_celestial_recommendations(celestial_df)

with st.expander("AI Insight", expanded=_section_open("AI Insight")):
    st.caption(
        "AI/RAG explains the fixed model output and searches the stargazing knowledge base. "
        "It does not change the score."
    )
    insight_tab, travel_tab, search_tab = st.tabs(
        ["Forecast AI Insight", "Travel Plan", "Semantic Knowledge Search"]
    )
    with insight_tab:
        if not include_llm:
            st.info("Turn on 'Generate AI recommendation' in the sidebar, then run again.")
        else:
            if st.button("Generate Forecast Insight", width="stretch", type="primary", key="dashboard_generate_forecast_insight"):
                with st.spinner("Generating forecast-based AI insight..."):
                    answer = cached_generate_forecast_insight(result["llm_context"])
                st.markdown('<div class="rag-box">', unsafe_allow_html=True)
                st.markdown(answer)
                st.markdown('</div>', unsafe_allow_html=True)
    with travel_tab:
        current_best_score = safe_numeric(best_score, 0)
        st.markdown(
            f"""
            <div class="section-card">
                <h3 class="section-heading" style="font-family: 'DM Serif Display', Georgia, serif; font-weight: 400; font-size:20px; color:#e4dff0; margin:0 0 8px 0;">Nearby Stargazing Trip</h3>
                <p class="muted">
                    Travel planning activates only when the current best score is at least
                    <b style="color:#e4dff0;">{TRAVEL_PLAN_SCORE_THRESHOLD:.0f}/100</b>.
                    Current best score:
                    <b style="color:#e4dff0;">{current_best_score:.1f}/100</b>.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if current_best_score < TRAVEL_PLAN_SCORE_THRESHOLD:
            st.info(
                "The current score is not high enough for an AI travel plan yet. "
                "Run this again when the forecast is Good or Excellent."
            )
        else:
            radius_km = st.slider(
                "Nearby search radius (km)",
                min_value=25,
                max_value=150,
                value=75,
                step=25,
                key="travel_plan_radius_km",
            )
            max_candidates = st.slider(
                "Candidate locations to score",
                min_value=4,
                max_value=16,
                value=8,
                step=4,
                key="travel_plan_candidate_count",
            )

            if st.button("Find Best Nearby Stargazing Trip", width="stretch", type="primary", key="dashboard_generate_travel_plan"):
                with st.status("Searching nearby forecast points...", expanded=True) as status:
                    st.write(f"Scoring up to {int(max_candidates)} locations inside {int(radius_km)} km.")
                    if include_llm:
                        st.write("AI text generation is enabled; the final plan may use OpenAI.")
                    else:
                        st.write("AI text generation is off; a deterministic travel plan will be shown.")
                    travel_result = cached_generate_travel_plan(
                        result["llm_context"],
                        bortle_index,
                        days,
                        float(radius_km),
                        int(max_candidates),
                        float(TRAVEL_PLAN_SCORE_THRESHOLD),
                        bool(include_llm),
                    )
                    st.session_state["travel_result"] = travel_result
                    status.update(label="Nearby travel search complete.", state="complete")

            travel_result = st.session_state.get("travel_result")
            if travel_result:
                search = travel_result.get("search", {})
                if search.get("status") != "eligible":
                    st.info(search.get("reason", "Travel plan was not generated."))
                else:
                    best_candidate = search.get("best_candidate", {})
                    destination = search.get("destination") or {}
                    destination_name = destination.get("name") or "coordinate-based candidate"
                    st.success(
                        "Best nearby trip target: "
                        f"{destination_name} · "
                        f"{best_candidate.get('best_score', 0):.1f}/100"
                    )

                    candidate_df = pd.DataFrame(search.get("candidates", []))
                    if not candidate_df.empty:
                        display_cols = [c for c in [
                            "best_score",
                            "recommendation",
                            "time_label",
                            "distance_km",
                            "destination_name",
                            "destination_distance_km",
                            "bearing_deg",
                            "estimated_bortle_index",
                            "lat",
                            "lon",
                            "weather_source",
                        ] if c in candidate_df.columns]
                        st.dataframe(
                            candidate_df[display_cols].head(10),
                            width="stretch",
                            hide_index=True,
                        )

                    render_travel_map(
                        search,
                        current_lat=float(result.get("lat", 0)),
                        current_lon=float(result.get("lon", 0)),
                    )

                    if search.get("candidate_errors"):
                        st.caption(
                            f"{len(search.get('candidate_errors', []))} nearby candidate(s) failed "
                            "and were skipped; partial results are shown."
                        )

                    if travel_result.get("travel_plan"):
                        st.markdown('<div class="rag-box">', unsafe_allow_html=True)
                        st.markdown(travel_result["travel_plan"])
                        st.markdown('</div>', unsafe_allow_html=True)
    with search_tab:
        user_question = st.text_area(
            "Ask the knowledge base",
            value="How do city lights and moon illumination affect stargazing?",
            height=110,
            key="dashboard_knowledge_question",
        )
        use_forecast_context = st.checkbox(
            "Use current forecast context",
            value=True,
            key="dashboard_use_forecast_context",
        )
        if not include_llm:
            st.info("Turn on 'Generate AI recommendation' in the sidebar, then run again.")
        else:
            if st.button("Search Knowledge Base", width="stretch", key="dashboard_search_knowledge"):
                with st.spinner("Searching vector knowledge base..."):
                    answer = cached_answer_semantic_question(
                        user_question,
                        result["llm_context"],
                        use_forecast_context,
                    )
                st.markdown('<div class="rag-box">', unsafe_allow_html=True)
                st.markdown(answer)
                st.markdown('</div>', unsafe_allow_html=True)

with st.expander("Glossary", expanded=_section_open("Glossary")):
    render_glossary_panel()

if selected_page == "Methodology":
    with st.expander("Methodology", expanded=True):
        st.markdown(
            """
            <div class="section-card">
                <h3 class="section-heading" style="font-family: 'DM Serif Display', Georgia, serif; font-weight: 400; font-size:20px; color:#e4dff0; margin:0 0 8px 0;">Data Sources</h3>
                <ul class="muted">
                    <li>Primary astronomy-weather source: Astrospheric.</li>
                    <li>Meteoblue Validation: realtime cloud, visibility, humidity, wind, and stability context. If Astrospheric is unavailable, Meteoblue can become the primary weather feed.</li>
                    <li>Open-Meteo Air Quality: aerosol optical depth, PM2.5, and dust for haze validation.</li>
                    <li>Open-Meteo weather fallback: cloud cover, visibility, temperature, dew point, and wind when higher-quality feeds are unavailable.</li>
                    <li>Primary astronomy source: IPGeolocation for twilight, Moon phase, moonrise, and moonset.</li>
                    <li>Optional Timeanddate data: detailed Sun/Moon events and hourly Moon position. When enabled, hourly Moon altitude/illumination are used for moonlight scoring.</li>
                </ul>
            </div>
            <div class="section-card">
                <h3 class="section-heading" style="font-family: 'DM Serif Display', Georgia, serif; font-weight: 400; font-size:20px; color:#e4dff0; margin:0 0 8px 0;">Scoring Logic</h3>
                <p class="muted">
                    The final score is a 0-100 deterministic synthesis of two sub-scores:
                    <b>observability_score</b>, which estimates whether the sky is practically usable,
                    and <b>view_quality_score</b>, which estimates how good the view should be once
                    the sky is usable.
                </p>
                <p class="muted">
                    <b>Observability</b> starts with astronomical darkness and cloud cover.
                    Daylight/twilight rows are strongly suppressed by a darkness gate. Cloud cover
                    is converted through a smooth transmission curve instead of hard 40/60/80%
                    cutoffs, so adjacent cloud percentages behave consistently. Observability also
                    includes effective darkness, so dark-sky locations and low Moon interference
                    improve the usable-window score.
                </p>
                <p class="muted">
                    <b>View quality</b> blends atmospheric and contrast conditions. Atmospheric score
                    is 40% transparency, 30% seeing or seeing proxy, 15% humidity/dew spread, and
                    15% haze from aerosol/PM data when available. Effective darkness is reduced by
                    Bortle light pollution and by Moon brightness when the Moon is above the horizon.
                    If hourly Moon position data is enabled, the Moon penalty uses hourly Moon altitude;
                    otherwise it falls back to Moon meridian altitude.
                </p>
                <p class="muted">
                    <b>Final formula:</b> stargazing_score = 100 ×
                    (observability_score / 100)<sup>0.62</sup> ×
                    (view_quality_score / 100)<sup>0.38</sup>.
                    This geometric blend keeps bad observability or poor view quality from being hidden
                    by the other component, while preserving ranking spread among viable nighttime hours.
                </p>
                <p class="muted">
                    Recommendation labels are assigned after scoring:
                    Excellent ≥ 85, Good ≥ 70, Marginal ≥ 50, Poor ≥ 25, and No-Go &lt; 25.
                </p>
            </div>
            <div class="section-card">
                <h3 class="section-heading" style="font-family: 'DM Serif Display', Georgia, serif; font-weight: 400; font-size:20px; color:#e4dff0; margin:0 0 8px 0;">AI / RAG Layer</h3>
                <p class="muted">
                    The AI Insight page does not recalculate the score.
                    It only explains the fixed model output.
                    When RAG is enabled, the model retrieves local stargazing knowledge
                    about sky darkness, moonlight, cloud cover, transparency, seeing,
                    and data limitations before generating the explanation.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

if selected_page == "Telemetry":
    with st.expander("Telemetry", expanded=True):
        render_telemetry_console(telemetry)
