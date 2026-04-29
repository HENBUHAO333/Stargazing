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
# NEBULA DUSK — FULL THEME CSS
# ============================================================


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

    p, li, span, label, td, th,
    button, input, select, textarea,
    [data-testid="stMarkdownContainer"] p {
        font-family: 'DM Sans', system-ui, sans-serif !important;
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
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div {
        color: var(--text) !important;
        font-family: 'DM Sans', system-ui, sans-serif !important;
        font-size: 13px !important;
    }
    [data-testid="stSidebar"] .stMarkdown p {
        color: var(--text-muted) !important;
        font-size: 12px !important;
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
    """
    <img src="x" style="display:none" onerror="
        (function() {
            if (window.__nd_font_applied) return;
            window.__nd_font_applied = true;
            var S = 'DM Serif Display, Georgia, serif';
            function fix() {
                var sel = '.hero-title,.section-heading,.window-time,.window-score,.metric-large';
                document.querySelectorAll(sel).forEach(function(el) {
                    el.style.setProperty('font-family', S, 'important');
                    el.style.setProperty('font-weight', '400', 'important');
                });
                document.querySelectorAll(
                    '[data-testid=stHeadingWithActionElements] h1,' +
                    '[data-testid=stHeadingWithActionElements] h2,' +
                    '[data-testid=stHeadingWithActionElements] h3'
                ).forEach(function(el) {
                    el.style.setProperty('font-family', S, 'important');
                    el.style.setProperty('font-weight', '400', 'important');
                });
            }
            fix();
            setInterval(fix, 300);
            new MutationObserver(fix).observe(document.documentElement, {childList:true, subtree:true});
        })();
    ">
    """,
    unsafe_allow_html=True,
)

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
    help="Only used for Sky Path visualization. This does not affect the score.",
)

include_llm = st.sidebar.checkbox(
    "Generate AI recommendation",
    value=False,
    help="Only explains fixed model output. This does not affect the score.",
)

run_button = st.sidebar.button("▶  Run Pipeline", use_container_width=True, type="primary")

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
        lat=lat, lon=lon,
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
        index="date_str", columns="hour",
        values="stargazing_score", aggfunc="mean",
    )
    fig = px.imshow(
        pivot, text_auto=".0f", aspect="auto",
        title="Stargazing Score Heatmap by Date and Hour",
        labels=dict(x="Hour of Day", y="Date", color="Score"),
        zmin=0, zmax=100,
        color_continuous_scale=[[0,"#100d1e"],[0.5,"#6b2f7a"],[1,"#c478d2"]],
    )
    return _themed_layout(fig, 440)


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
    fig = px.line(long_df, x="local_dt", y="value", color="factor", title="Key Stargazing Factors Over Time")
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
    )
    return _themed_layout(fig, 420)


def render_source_badges(result):
    weather_source  = result.get("weather_source", "Unknown")
    astronomy_source = result.get("astronomy_source", "Unknown")
    timezone_value  = result.get("timezone", "Unknown")
    w_cls = "warning-pill" if "fallback" in weather_source.lower() else "source-pill"
    a_cls = "warning-pill" if "fallback" in astronomy_source.lower() else "source-pill"
    st.markdown(
        f'<span class="{w_cls}">Weather: {weather_source}</span>'
        f'<span class="{a_cls}">Astronomy: {astronomy_source}</span>'
        f'<span class="source-pill">Timezone: {timezone_value}</span>',
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


if "pipeline_result" not in st.session_state:
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-flex">
                <div class="hero-left">
                    <div class="hero-icon-wrap">🚀</div>
                    <div>
                        <h1 class="hero-title" style="font-family: 'DM Serif Display', Georgia, serif; font-weight: 400; font-size:2rem; color:#e4dff0; line-height:1.15; margin:0 0 5px 0;">Stargazing Assistant</h1>
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
                <div class="hero-icon-wrap">🚀</div>
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
# PAGE ROUTER
# ============================================================

if selected_page == "Dashboard":
    st.subheader("Dashboard")
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
        """, unsafe_allow_html=True,
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
        chart_df, x="local_dt", y="stargazing_score",
        color="recommendation", markers=True,
        title="Hourly Stargazing Score",
        hover_data=[c for c in [
            "cloud_value","transparency_value","seeing_value","moon_illuminated_pct",
            "is_dark_enough","is_moon_up","effective_darkness","atmospheric_score",
        ] if c in chart_df.columns],
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
        hover_data=[c for c in [
            "cloud_value","transparency_value","seeing_value","moon_illuminated_pct",
            "is_dark_enough","is_moon_up","visibility_penalty","transparency_norm",
            "seeing_norm","humidity_quality","moon_brightness_penalty",
            "effective_darkness","atmospheric_score",
        ] if c in chart_df.columns],
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
        x="stargazing_score", y="time_label",
        orientation="h", color="recommendation",
        hover_data=[c for c in [
            "cloud_value","transparency_value","seeing_value","moon_illuminated_pct",
            "is_dark_enough","is_moon_up","visibility_penalty","effective_darkness","atmospheric_score",
        ] if c in top_windows.columns],
        title="Top 10 Stargazing Windows",
    )
    fig_top.update_layout(xaxis_range=[0, 100])
    st.plotly_chart(_themed_layout(fig_top, 520), use_container_width=True)
    st.dataframe(top_windows, use_container_width=True)


elif selected_page == "Sky Conditions":
    st.subheader("Sky Conditions")
    st.plotly_chart(build_factor_chart(score_df), use_container_width=True)
    feature_options = [c for c in [
        "cloud_value","transparency_value","seeing_value","moon_illuminated_pct",
        "visibility_penalty","transparency_norm","seeing_norm","humidity_quality",
        "moon_brightness_penalty","effective_darkness","atmospheric_score","stargazing_score",
    ] if c in score_df.columns]
    if feature_options:
        selected_feature = st.selectbox("Inspect one feature", feature_options)
        plot_df = score_df.copy()
        plot_df["local_dt"] = pd.to_datetime(plot_df["local_dt"], errors="coerce")
        fig_feature = px.line(plot_df, x="local_dt", y=selected_feature, markers=True, title=f"{selected_feature} Over Time")
        st.plotly_chart(_themed_layout(fig_feature, 480), use_container_width=True)

    st.subheader("Scoring Feature Table")
    diagnostic_cols = [c for c in [
        "local_dt","stargazing_score","recommendation","cloud_value","transparency_value",
        "seeing_value","visibility_penalty","transparency_norm","seeing_norm","humidity_quality",
        "moon_brightness_penalty","effective_darkness","atmospheric_score",
    ] if c in score_df.columns]
    st.dataframe(score_df[diagnostic_cols].head(60), use_container_width=True)


elif selected_page == "Sky Path":
    st.subheader("Sky Path")
    st.caption("This visualization uses Sun/Moon position data only. It does not affect the recommendation score.")
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
                    hover_data=[c for c in ["Hour","Illuminated (%)","Moon Phase"] if c in pos_plot_df.columns],
                    title="Sun and Moon Position in the Sky",
                )
                st.plotly_chart(_themed_layout(fig_pos, 650), use_container_width=True)
            else:
                st.warning("Position data exists, but altitude/azimuth values are missing.")
        else:
            st.warning("Position data does not contain the required columns: Azimuth (°), Altitude (°), Object.")
        with st.expander("Position Data"):
            st.dataframe(position_df, use_container_width=True)


elif selected_page == "AI Insight":
    st.subheader("AI Insight")
    st.caption("AI explanation is generated from fixed model outputs. RAG uses a local stargazing knowledge base to improve explanation quality. It does not change the score.")
    st.markdown(
        """
        <div class="section-card">
            <h3 class="section-heading" style="font-family: 'DM Serif Display', Georgia, serif; font-weight: 400; font-size:20px; color:#e4dff0; margin:0 0 8px 0;">Ask the Stargazing Assistant</h3>
            <p class="muted">
                Use this page to ask why a window is recommended, what objects to observe,
                or how moonlight, clouds, transparency, and city lights affect the result.
            </p>
        </div>
        """, unsafe_allow_html=True,
    )
    user_question = st.text_area(
        "Ask a stargazing question",
        value="Based on the current forecast, what should I observe and why?",
        height=110,
    )
    use_rag = st.checkbox(
        "Use local stargazing knowledge base",
        value=True,
        help="Retrieves explanations about sky darkness, cloud cover, moonlight, seeing, transparency, and data limitations.",
    )
    if not include_llm:
        st.info("Turn on 'Generate AI recommendation' in the sidebar, then run again.")
    else:
        col_a, col_b = st.columns([1, 1])
        with col_a:
            generate_button = st.button("Generate AI Insight", use_container_width=True, type="primary")
        with col_b:
            st.caption("RAG is on" if use_rag else "Standard LLM explanation only")
        if generate_button:
            with st.spinner("Generating grounded AI insight..."):
                answer = cached_generate_rag(result["llm_context"], user_question) if use_rag else cached_generate_llm(result["llm_context"])
            st.markdown('<div class="rag-box">', unsafe_allow_html=True)
            st.markdown(answer)
            st.markdown('</div>', unsafe_allow_html=True)
        if use_rag and not HAS_RAG_BACKEND:
            st.warning("RAG UI is ready, but backend.py does not yet have `generate_rag_recommendation()`. Add the backend RAG function next.")


elif selected_page == "Methodology":
    st.subheader("Methodology")
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
        <div class="section-card">
            <h3 class="section-heading" style="font-family: 'DM Serif Display', Georgia, serif; font-weight: 400; font-size:20px; color:#e4dff0; margin:0 0 8px 0;">Important Design Rule</h3>
            <p class="muted">
                Sun/Moon position data, clustering, and AI/RAG explanations are
                interpretation layers. They do not change the stargazing score.
            </p>
        </div>
        """, unsafe_allow_html=True,
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
