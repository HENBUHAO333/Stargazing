import json
import math
import random
from html import escape
import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime

# ============================================================
# BACKEND IMPORTS
# ============================================================

try:
    from backend import (
        run_pipeline,
        CITY_PRESETS,
        generate_llm_recommendation,
        generate_rag_recommendation,
        generate_forecast_ai_insight,
        answer_semantic_knowledge_question,
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
    generate_forecast_ai_insight = None
    answer_semantic_knowledge_question = None
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
    lat      = st.sidebar.number_input("Latitude",  value=40.7128, format="%.6f")
    lon      = st.sidebar.number_input("Longitude", value=-74.0060, format="%.6f")
    timezone = st.sidebar.text_input("Timezone", "America/New_York")

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
    st.sidebar.success("Cache cleared. Run the pipeline again.")


# ============================================================
# CACHE FUNCTIONS
# ============================================================

@st.cache_data(ttl=1800, show_spinner=False)
def cached_run_pipeline(city_name, lat, lon, timezone, days, bortle_index, include_tad):
    return run_pipeline(
        city_name=city_name,
        lat=lat, lon=lon,
        timezone=timezone,
        days=days,
        bortle_index=bortle_index,
        include_tad=include_tad,
        include_positions=False,
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
            "`generate_forecast_ai_insight()` to backend.py first."
        )
    return generate_forecast_ai_insight(context)


@st.cache_data(ttl=1800, show_spinner=False)
def cached_answer_semantic_question(user_question, context, use_forecast_context):
    if answer_semantic_knowledge_question is None:
        return (
            "Semantic Search backend function not found. Please add "
            "`answer_semantic_knowledge_question()` to backend.py first."
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
    "transparency_norm": "Transparency Quality",
    "seeing_norm": "Seeing Quality",
    "humidity_quality": "Humidity Quality",
    "moon_brightness_penalty": "Moon Brightness Penalty",
    "effective_darkness": "Effective Darkness",
    "atmospheric_score": "Atmospheric Score",
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


def render_source_badges(result):
    weather_source  = result.get("weather_source", "Unknown")
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
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-flex" style="justify-content:center;">
                <div class="hero-left" style="flex-direction:column;align-items:center;">
                    <div class="hero-icon-wrap">🌌</div>
                    <div class="hero-copy" style="text-align:center;">
                        <div class="hero-sparkles">
                            <span class="hero-const-line l1"></span>
                            <span class="hero-const-line l2"></span>
                            <span class="hero-const-line l3"></span>
                            <span class="hero-const-line l4"></span>
                            <span class="hero-const-line l5"></span>
                            <span class="hero-const-line l6"></span>
                            <span class="hero-const-line l7"></span>
                            <span class="hero-const-line l8"></span>
                            <span class="hero-const-line l9"></span>
                            <span class="hero-const-line l10"></span>
                            <span class="hero-const-line l11"></span>
                            <span class="hero-const-line l12"></span>
                            <span class="hero-const-line l13"></span>
                            <span class="hero-sparkle s1"></span>
                            <span class="hero-sparkle s2"></span>
                            <span class="hero-sparkle s3"></span>
                            <span class="hero-sparkle s4"></span>
                            <span class="hero-sparkle s5"></span>
                            <span class="hero-sparkle s6"></span>
                            <span class="hero-sparkle s7"></span>
                            <span class="hero-sparkle s8"></span>
                            <span class="hero-sparkle s9"></span>
                            <span class="hero-sparkle s10"></span>
                            <span class="hero-sparkle s11"></span>
                            <span class="hero-sparkle s12"></span>
                            <span class="hero-sparkle s13"></span>
                            <span class="hero-sparkle s14"></span>
                            <span class="hero-const-label ursa">Ursa Minor</span>
                            <span class="hero-const-label polaris">Polaris</span>
                            <span class="hero-const-label plough">The Plough</span>
                        </div>
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
            <h3 class="section-heading" style="font-family: 'DM Serif Display', Georgia, serif; font-weight: 400; font-size:20px; color:#e4dff0; margin:0 0 8px 0;">Ready to find your best stargazing window?</h3>
            <p class="muted">
                Choose a location from the sidebar, adjust the city lights index,
                and run the live recommendation pipeline.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
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
position_df   = pd.DataFrame()
telemetry     = result.get("telemetry",     {})

if top_windows is None or top_windows.empty:
    display_empty_result_debug(score_df, top_windows)

sky_state = infer_reactive_sky_state(score_df, top_windows)
apply_reactive_sky_style(sky_state)

if include_positions:
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
                <div class="hero-copy">
                    <div class="hero-sparkles">
                        <span class="hero-const-line l1"></span>
                        <span class="hero-const-line l2"></span>
                        <span class="hero-const-line l3"></span>
                        <span class="hero-const-line l4"></span>
                        <span class="hero-const-line l5"></span>
                        <span class="hero-const-line l6"></span>
                        <span class="hero-const-line l7"></span>
                        <span class="hero-const-line l8"></span>
                        <span class="hero-const-line l9"></span>
                        <span class="hero-const-line l10"></span>
                        <span class="hero-const-line l11"></span>
                        <span class="hero-const-line l12"></span>
                        <span class="hero-const-line l13"></span>
                        <span class="hero-sparkle s1"></span>
                        <span class="hero-sparkle s2"></span>
                        <span class="hero-sparkle s3"></span>
                        <span class="hero-sparkle s4"></span>
                        <span class="hero-sparkle s5"></span>
                        <span class="hero-sparkle s6"></span>
                        <span class="hero-sparkle s7"></span>
                        <span class="hero-sparkle s8"></span>
                        <span class="hero-sparkle s9"></span>
                        <span class="hero-sparkle s10"></span>
                        <span class="hero-sparkle s11"></span>
                        <span class="hero-sparkle s12"></span>
                        <span class="hero-sparkle s13"></span>
                        <span class="hero-sparkle s14"></span>
                        <span class="hero-const-label ursa">Ursa Minor</span>
                        <span class="hero-const-label polaris">Polaris</span>
                        <span class="hero-const-label plough">The Plough</span>
                    </div>
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
st.caption(f"Reactive sky mode: {sky_state.replace('_', ' ').title()}")

st.markdown(
    """
    <div class="inline-cell">
        <p class="info-note">
        &#9432;&nbsp; Score is generated only from the weather/astronomy scoring pipeline.
        Sun/Moon position data, clustering, and AI/RAG explanation do not change the score.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="inline-cell">
        <p class="status-text">
            Live data loaded for {result.get('city_name', city_name)}
            ({result.get('lat', 0):.4f}, {result.get('lon', 0):.4f})
        </p>
    </div>
    """,
    unsafe_allow_html=True,
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
    fig_top = px.bar(
        top_windows.sort_values("stargazing_score"),
        x="stargazing_score", y="time_label",
        orientation="h", color="recommendation",
        hover_data=[c for c in [
            "cloud_value","transparency_value","seeing_value","moon_illuminated_pct",
            "is_dark_enough","is_moon_up","visibility_penalty","effective_darkness","atmospheric_score",
        ] if c in top_windows.columns],
        title="Top 10 Stargazing Windows",
        labels=labels_for(
            "stargazing_score", "time_label", "recommendation", "cloud_value",
            "transparency_value", "seeing_value", "moon_illuminated_pct",
            "is_dark_enough", "is_moon_up", "visibility_penalty",
            "effective_darkness", "atmospheric_score",
        ),
    )
    fig_top.update_layout(xaxis_range=[0, 100])
    st.plotly_chart(_themed_layout(fig_top, 520), width="stretch")

with st.expander("Sky Conditions", expanded=_section_open("Sky Conditions")):
    st.plotly_chart(build_factor_chart(score_df), width="stretch")
    feature_options = [c for c in [
        "cloud_value","transparency_value","seeing_value","moon_illuminated_pct",
        "visibility_penalty","transparency_norm","seeing_norm","humidity_quality",
        "moon_brightness_penalty","effective_darkness","atmospheric_score","stargazing_score",
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
    insight_tab, search_tab = st.tabs(["Forecast AI Insight", "Semantic Knowledge Search"])
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
                    <li>Primary weather source: Astrospheric</li>
                    <li>Fallback weather source: Open-Meteo</li>
                    <li>Primary astronomy source: IPGeolocation</li>
                    <li>Optional detailed event source: Timeanddate</li>
                </ul>
            </div>
            <div class="section-card">
                <h3 class="section-heading" style="font-family: 'DM Serif Display', Georgia, serif; font-weight: 400; font-size:20px; color:#e4dff0; margin:0 0 8px 0;">Scoring Logic</h3>
                <p class="muted">
                    The score combines visibility constraints, atmospheric quality,
                    and darkness quality. Cloud cover and daylight act as hard penalties,
                    while transparency, seeing, humidity, moon illumination, moon altitude,
                    and city lights influence the final score.
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
