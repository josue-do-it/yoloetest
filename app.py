"""
YOLOE Studio  —  Block 1: Prompt → Mask → Crop
=================================================
Architecture
  styles.py  ←  this file owns ALL CSS (one place to edit)
  theme      ←  light / dark via CSS custom properties + .dark class
  Logic      ←  unchanged from original (utilities, runners, parsers)
  UI         ←  professional, no emojis, clean typography
"""

import sys, os, io, json, tempfile, importlib, traceback
from pathlib import Path

import streamlit as st

# ---------------------------------------------------------------------------
# OpenCV guard
# ---------------------------------------------------------------------------
try:
    import cv2
except ImportError as _cv_err:
    st.error(
        f"OpenCV import failed: {_cv_err}\n\n"
        "Ensure packages.txt lists libgl1 and libglib2.0-0t64, "
        "and requirements.txt uses opencv-python-headless."
    )
    st.stop()

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# App directory
# ---------------------------------------------------------------------------
try:
    APP_DIR = Path(__file__).resolve().parent
except NameError:
    APP_DIR = Path(os.getcwd())

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

# ---------------------------------------------------------------------------
# Page config  (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="YOLOE Studio",
    page_icon=None,  # set to a .png path if you have a custom favicon
    layout="wide",
    initial_sidebar_state="expanded",
)

# ===========================================================================
#  DESIGN SYSTEM
#  All visual decisions live here.  To retheme: edit tokens only.
# ===========================================================================
STYLES = """
<style>
/* ─── GOOGLE FONTS ─────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* ─── DESIGN TOKENS  (light mode defaults) ─────────────────────────────── */
:root {
  /* Background layers */
  --c-bg:          #f4f6fb;
  --c-surface:     #ffffff;
  --c-surface-alt: #eef1f8;
  --c-overlay:     #ffffff;

  /* Borders */
  --c-border:      #dce1f0;
  --c-border-mid:  #c5cde4;

  /* Text */
  --c-text-primary:   #0d1224;
  --c-text-secondary: #4b5678;
  --c-text-muted:     #8c96b5;

  /* Brand — indigo */
  --c-brand:       #3d5cf5;
  --c-brand-dim:   #eaedfd;
  --c-brand-border:#b5bffb;

  /* Accent palette */
  --c-violet:       #6d28d9;
  --c-violet-dim:   #f0ebfe;
  --c-violet-border:#c4b5fd;

  --c-teal:         #0d9488;
  --c-teal-dim:     #f0fdfa;
  --c-teal-border:  #99f6e4;

  --c-emerald:      #059669;
  --c-emerald-dim:  #ecfdf5;
  --c-emerald-border:#6ee7b7;

  --c-amber:        #b45309;
  --c-amber-dim:    #fffbeb;
  --c-amber-border: #fcd34d;

  --c-rose:         #be123c;
  --c-rose-dim:     #fff1f2;
  --c-rose-border:  #fda4af;

  /* Shadows */
  --shadow-sm: 0 1px 2px rgba(13,18,36,.06), 0 1px 4px rgba(13,18,36,.04);
  --shadow-md: 0 4px 12px rgba(13,18,36,.08), 0 2px 6px rgba(13,18,36,.05);

  /* Geometry */
  --radius-sm: 5px;
  --radius-md: 8px;
  --radius-lg: 12px;

  /* Typography */
  --font-sans: 'Inter', -apple-system, system-ui, sans-serif;
  --font-mono: 'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace;
}

/* ─── DARK MODE TOKENS ─────────────────────────────────────────────────── */
.dark-theme {
  --c-bg:          #080c18;
  --c-surface:     #0d1225;
  --c-surface-alt: #131828;
  --c-overlay:     #0d1225;

  --c-border:      #1a2040;
  --c-border-mid:  #222b4a;

  --c-text-primary:   #dde3f5;
  --c-text-secondary: #7a86aa;
  --c-text-muted:     #3e4a6a;

  --c-brand:       #5e78f8;
  --c-brand-dim:   #0d1438;
  --c-brand-border:#1e2e60;

  --c-violet:       #a78bfa;
  --c-violet-dim:   #140e36;
  --c-violet-border:#2d1e62;

  --c-teal:         #2dd4bf;
  --c-teal-dim:     #041e1a;
  --c-teal-border:  #0c3530;

  --c-emerald:      #34d399;
  --c-emerald-dim:  #03240f;
  --c-emerald-border:#064a28;

  --c-amber:        #fbbf24;
  --c-amber-dim:    #1c1100;
  --c-amber-border: #3a2800;

  --c-rose:         #fb7185;
  --c-rose-dim:     #200610;
  --c-rose-border:  #4a0c20;

  --shadow-sm: 0 1px 3px rgba(0,0,0,.5), 0 1px 6px rgba(0,0,0,.4);
  --shadow-md: 0 4px 16px rgba(0,0,0,.5), 0 2px 8px rgba(0,0,0,.4);
}

/* ─── RESET & BASE ─────────────────────────────────────────────────────── */
*, *::before, *::after {
  box-sizing: border-box;
  font-family: var(--font-sans);
}

.stApp {
  background-color: var(--c-bg) !important;
  color: var(--c-text-primary) !important;
}

/* ─── SIDEBAR ───────────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
  background-color: var(--c-surface) !important;
  border-right: 1px solid var(--c-border) !important;
}
section[data-testid="stSidebar"] > div {
  padding-top: 1rem;
}

/* ─── WORDMARK ──────────────────────────────────────────────────────────── */
.wordmark {
  display: flex;
  align-items: center;
  gap: 10px;
  padding-bottom: 20px;
  border-bottom: 1px solid var(--c-border);
  margin-bottom: 18px;
}
.wordmark-logotype {
  width: 32px;
  height: 32px;
  border-radius: var(--radius-md);
  background: linear-gradient(135deg, var(--c-brand) 0%, var(--c-violet) 100%);
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
  box-shadow: 0 2px 8px rgba(61,92,245,.3);
}
.wordmark-logotype svg {
  width: 16px;
  height: 16px;
  fill: #fff;
}
.wordmark-text {
  line-height: 1;
}
.wordmark-name {
  font-size: .9rem;
  font-weight: 700;
  color: var(--c-text-primary);
  letter-spacing: -.02em;
}
.wordmark-sub {
  font-family: var(--font-mono);
  font-size: .55rem;
  color: var(--c-text-muted);
  letter-spacing: .12em;
  text-transform: uppercase;
  margin-top: 3px;
}

/* ─── SIDEBAR SECTION HEADING ───────────────────────────────────────────── */
.sidebar-section {
  font-family: var(--font-mono);
  font-size: .58rem;
  font-weight: 600;
  letter-spacing: .14em;
  text-transform: uppercase;
  color: var(--c-text-muted);
  padding-bottom: 5px;
  border-bottom: 1px solid var(--c-border);
  margin: 14px 0 10px;
  display: block;
}

/* ─── STATUS INDICATOR ──────────────────────────────────────────────────── */
.status-row {
  display: flex;
  flex-direction: column;
  gap: 5px;
  margin-top: 4px;
}
.status-item {
  display: inline-flex;
  align-items: center;
  gap: 7px;
  font-family: var(--font-mono);
  font-size: .66rem;
  font-weight: 500;
  padding: 4px 10px;
  border-radius: 20px;
}
.status-ok {
  background: var(--c-emerald-dim);
  color: var(--c-emerald);
  border: 1px solid var(--c-emerald-border);
}
.status-warn {
  background: var(--c-amber-dim);
  color: var(--c-amber);
  border: 1px solid var(--c-amber-border);
}
.status-error {
  background: var(--c-rose-dim);
  color: var(--c-rose);
  border: 1px solid var(--c-rose-border);
}
.status-dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: currentColor;
  flex-shrink: 0;
}

/* ─── PIPELINE STRIP ────────────────────────────────────────────────────── */
.pipeline {
  display: flex;
  align-items: center;
  gap: 6px;
  flex-wrap: wrap;
  background: var(--c-surface);
  border: 1px solid var(--c-border);
  border-radius: var(--radius-lg);
  padding: 11px 16px;
  margin-bottom: 24px;
  box-shadow: var(--shadow-sm);
}
.pipeline-step {
  font-family: var(--font-mono);
  font-size: .58rem;
  font-weight: 600;
  letter-spacing: .07em;
  text-transform: uppercase;
  padding: 4px 12px;
  border-radius: 20px;
  white-space: nowrap;
}
.pipeline-step--active {
  background: var(--c-brand-dim);
  color: var(--c-brand);
  border: 1px solid var(--c-brand-border);
}
.pipeline-step--pending {
  background: var(--c-surface-alt);
  color: var(--c-text-muted);
  border: 1px solid var(--c-border);
}
.pipeline-arrow {
  font-size: .75rem;
  color: var(--c-border-mid);
  user-select: none;
}

/* ─── PAGE HEADER ───────────────────────────────────────────────────────── */
.page-header {
  margin-bottom: 22px;
}
.page-header-eyebrow {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-family: var(--font-mono);
  font-size: .62rem;
  font-weight: 600;
  letter-spacing: .08em;
  text-transform: uppercase;
  padding: 3px 11px;
  border-radius: 20px;
  margin-bottom: 10px;
}
.eyebrow-text   { background: var(--c-brand-dim);   color: var(--c-brand);   border: 1px solid var(--c-brand-border); }
.eyebrow-visual { background: var(--c-violet-dim);  color: var(--c-violet);  border: 1px solid var(--c-violet-border); }
.eyebrow-free   { background: var(--c-teal-dim);    color: var(--c-teal);    border: 1px solid var(--c-teal-border); }
.page-header-title {
  font-size: 1.55rem;
  font-weight: 700;
  color: var(--c-text-primary);
  letter-spacing: -.03em;
  line-height: 1.2;
  margin: 0 0 4px;
}
.page-header-desc {
  font-size: .8rem;
  color: var(--c-text-secondary);
  line-height: 1.5;
  margin: 0;
}

/* ─── FIELD LABEL ───────────────────────────────────────────────────────── */
.field-label {
  font-family: var(--font-mono);
  font-size: .58rem;
  font-weight: 600;
  letter-spacing: .14em;
  text-transform: uppercase;
  color: var(--c-text-muted);
  padding-bottom: 6px;
  margin-bottom: 8px;
  border-bottom: 1px solid var(--c-border);
  display: block;
}

/* ─── METRIC CARDS ──────────────────────────────────────────────────────── */
.metrics-row {
  display: flex;
  gap: 10px;
  margin: 16px 0 6px;
}
.metric-card {
  flex: 1;
  background: var(--c-surface);
  border: 1px solid var(--c-border);
  border-radius: var(--radius-md);
  padding: 14px 16px;
  box-shadow: var(--shadow-sm);
}
.metric-card__value {
  font-family: var(--font-mono);
  font-size: 1.5rem;
  font-weight: 700;
  color: var(--c-brand);
  line-height: 1.1;
  letter-spacing: -.02em;
}
.metric-card__label {
  font-size: .6rem;
  font-weight: 600;
  color: var(--c-text-muted);
  text-transform: uppercase;
  letter-spacing: .1em;
  margin-top: 4px;
}

/* ─── RESULT PANELS ─────────────────────────────────────────────────────── */
.panel-header {
  font-family: var(--font-mono);
  font-size: .6rem;
  font-weight: 600;
  color: var(--c-text-muted);
  text-transform: uppercase;
  letter-spacing: .1em;
  padding: 9px 12px;
  background: var(--c-surface-alt);
  border-bottom: 1px solid var(--c-border);
  border-radius: var(--radius-md) var(--radius-md) 0 0;
}

/* ─── CROP META ─────────────────────────────────────────────────────────── */
.crop-meta {
  background: var(--c-emerald-dim);
  border: 1px solid var(--c-emerald-border);
  border-radius: var(--radius-sm);
  padding: 10px 14px;
  margin-top: 10px;
  font-family: var(--font-mono);
  font-size: .68rem;
  color: var(--c-emerald);
  line-height: 2;
}
.crop-meta b {
  font-weight: 600;
}

/* ─── DETECTION LIST ────────────────────────────────────────────────────── */
.detection-item {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 10px 14px;
  margin-bottom: 6px;
  background: var(--c-surface);
  border: 1px solid var(--c-border);
  border-radius: var(--radius-md);
  box-shadow: var(--shadow-sm);
  transition: border-color .15s ease, box-shadow .15s ease;
}
.detection-item:hover {
  border-color: var(--c-brand-border);
  box-shadow: var(--shadow-md);
}
.detection-item__index {
  font-family: var(--font-mono);
  font-size: .6rem;
  font-weight: 700;
  padding: 2px 8px;
  border-radius: var(--radius-sm);
  background: var(--c-brand-dim);
  color: var(--c-brand);
  border: 1px solid var(--c-brand-border);
  min-width: 32px;
  text-align: center;
  flex-shrink: 0;
}
.detection-item__label {
  flex: 1;
  font-size: .82rem;
  font-weight: 500;
  color: var(--c-text-primary);
}
.detection-item__conf {
  font-family: var(--font-mono);
  font-size: .72rem;
  color: var(--c-brand);
  min-width: 55px;
  text-align: right;
  flex-shrink: 0;
}
.detection-item__px {
  font-size: .68rem;
  color: var(--c-text-muted);
  min-width: 72px;
  text-align: right;
  flex-shrink: 0;
}

/* ─── EXPORT SECTION ────────────────────────────────────────────────────── */
.export-header {
  font-family: var(--font-mono);
  font-size: .58rem;
  font-weight: 600;
  letter-spacing: .14em;
  text-transform: uppercase;
  color: var(--c-text-muted);
  margin-bottom: 8px;
  display: block;
}

/* ─── STREAMLIT WIDGET OVERRIDES ────────────────────────────────────────── */
/* File uploader */
div[data-testid="stFileUploader"] {
  background: var(--c-surface-alt) !important;
  border: 1.5px dashed var(--c-border-mid) !important;
  border-radius: var(--radius-lg) !important;
  transition: border-color .2s !important;
}
div[data-testid="stFileUploader"]:hover {
  border-color: var(--c-brand) !important;
}

/* Textarea */
.stTextArea textarea {
  background: var(--c-surface) !important;
  border: 1px solid var(--c-border-mid) !important;
  color: var(--c-text-primary) !important;
  font-family: var(--font-mono) !important;
  font-size: .81rem !important;
  border-radius: var(--radius-md) !important;
  line-height: 1.6 !important;
}
.stTextArea textarea:focus {
  border-color: var(--c-brand) !important;
  box-shadow: 0 0 0 3px rgba(61,92,245,.1) !important;
}

/* Primary button */
.stButton > button {
  background: var(--c-brand) !important;
  color: #ffffff !important;
  border: none !important;
  border-radius: var(--radius-md) !important;
  font-family: var(--font-sans) !important;
  font-size: .83rem !important;
  font-weight: 600 !important;
  letter-spacing: .01em !important;
  padding: .52rem 1.3rem !important;
  box-shadow: 0 2px 8px rgba(61,92,245,.28) !important;
  transition: opacity .15s ease, box-shadow .15s ease !important;
}
.stButton > button:hover {
  opacity: .88 !important;
  box-shadow: 0 4px 14px rgba(61,92,245,.38) !important;
}

/* Download button */
.stDownloadButton > button {
  background: var(--c-emerald-dim) !important;
  color: var(--c-emerald) !important;
  border: 1px solid var(--c-emerald-border) !important;
  border-radius: var(--radius-md) !important;
  font-family: var(--font-sans) !important;
  font-size: .78rem !important;
  font-weight: 600 !important;
}
.stDownloadButton > button:hover {
  opacity: .85 !important;
}

/* Labels */
label,
div[data-testid="stSidebar"] label {
  color: var(--c-text-secondary) !important;
  font-size: .79rem !important;
}

/* Dividers */
hr {
  border-color: var(--c-border) !important;
  margin: 14px 0 !important;
}

/* Sidebar widget labels */
.stRadio > div > label,
.stSelectbox label,
.stSlider label,
.stCheckbox label,
.stToggle label {
  color: var(--c-text-secondary) !important;
  font-size: .79rem !important;
}

/* Alerts */
.stAlert {
  border-radius: var(--radius-md) !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
  gap: 2px;
  border-bottom: 1px solid var(--c-border) !important;
}
.stTabs [data-baseweb="tab"] {
  font-family: var(--font-mono) !important;
  font-size: .72rem !important;
  font-weight: 600 !important;
  letter-spacing: .03em !important;
  border-radius: var(--radius-sm) var(--radius-sm) 0 0 !important;
}
</style>
"""

st.markdown(STYLES, unsafe_allow_html=True)


# ===========================================================================
#  THEME ENGINE
#  Injects CSS overrides for dark mode via a CSS class cascade.
#  Token values are defined inside .dark-theme to override :root defaults.
# ===========================================================================

def _theme_css_dark() -> str:
    """Returns the CSS that switches the app to dark mode."""
    return """<style>
.stApp, .stApp * {
  --c-bg:          #080c18;
  --c-surface:     #0d1225;
  --c-surface-alt: #131828;
  --c-overlay:     #0d1225;
  --c-border:      #1a2040;
  --c-border-mid:  #222b4a;
  --c-text-primary:   #dde3f5;
  --c-text-secondary: #7a86aa;
  --c-text-muted:     #3e4a6a;
  --c-brand:       #5e78f8;
  --c-brand-dim:   #0d1438;
  --c-brand-border:#1e2e60;
  --c-violet:       #a78bfa;
  --c-violet-dim:   #140e36;
  --c-violet-border:#2d1e62;
  --c-teal:         #2dd4bf;
  --c-teal-dim:     #041e1a;
  --c-teal-border:  #0c3530;
  --c-emerald:      #34d399;
  --c-emerald-dim:  #03240f;
  --c-emerald-border:#064a28;
  --c-amber:        #fbbf24;
  --c-amber-dim:    #1c1100;
  --c-amber-border: #3a2800;
  --c-rose:         #fb7185;
  --c-rose-dim:     #200610;
  --c-rose-border:  #4a0c20;
  --shadow-sm: 0 1px 3px rgba(0,0,0,.5), 0 1px 6px rgba(0,0,0,.4);
  --shadow-md: 0 4px 16px rgba(0,0,0,.5), 0 2px 8px rgba(0,0,0,.4);
}
.stApp                { background-color: #080c18 !important; color: #dde3f5 !important; }
section[data-testid="stSidebar"] { background-color: #0d1225 !important; border-right-color: #1a2040 !important; }
div[data-testid="stFileUploader"] { background: #131828 !important; border-color: #222b4a !important; }
.stTextArea textarea  { background: #0d1225 !important; border-color: #222b4a !important; color: #dde3f5 !important; }
label, div[data-testid="stSidebar"] label { color: #7a86aa !important; }
hr                    { border-color: #1a2040 !important; }
.stRadio > div > label, .stSelectbox label,
.stSlider label, .stCheckbox label, .stToggle label { color: #7a86aa !important; }
</style>"""


def _theme_css_light() -> str:
    """Returns CSS that activates light mode — covers every Streamlit widget."""
    return """<style>
/* ── Token reset ── */
.stApp, .stApp * {
  --c-bg:          #f4f6fb;
  --c-surface:     #ffffff;
  --c-surface-alt: #eef1f8;
  --c-border:      #dce1f0;
  --c-border-mid:  #c5cde4;
  --c-text-primary:   #0d1224;
  --c-text-secondary: #4b5678;
  --c-text-muted:     #8c96b5;
  --c-brand:       #3d5cf5;
  --c-brand-dim:   #eaedfd;
  --c-brand-border:#b5bffb;
  --c-violet:       #6d28d9;
  --c-violet-dim:   #f0ebfe;
  --c-violet-border:#c4b5fd;
  --c-teal:         #0d9488;
  --c-teal-dim:     #f0fdfa;
  --c-teal-border:  #99f6e4;
  --c-emerald:      #059669;
  --c-emerald-dim:  #ecfdf5;
  --c-emerald-border:#6ee7b7;
  --c-amber:        #b45309;
  --c-amber-dim:    #fffbeb;
  --c-amber-border: #fcd34d;
  --c-rose:         #be123c;
  --c-rose-dim:     #fff1f2;
  --c-rose-border:  #fda4af;
  --shadow-sm: 0 1px 2px rgba(13,18,36,.06), 0 1px 4px rgba(13,18,36,.04);
  --shadow-md: 0 4px 12px rgba(13,18,36,.08), 0 2px 6px rgba(13,18,36,.05);
}

/* ── App shell ── */
.stApp { background-color: #f4f6fb !important; color: #0d1224 !important; }
.main .block-container { color: #0d1224 !important; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
  background-color: #ffffff !important;
  border-right-color: #dce1f0 !important;
}

/* ── All text / labels ── */
label,
div[data-testid="stSidebar"] label,
.stMarkdown p, .stMarkdown span,
p, span, div { color: #0d1224; }

/* Widget labels (sidebar + main) */
.stRadio      > label,
.stSelectbox  > label,
.stSlider     > label,
.stCheckbox   > label,
.stToggle     > label,
.stTextArea   > label,
.stTextInput  > label,
.stNumberInput > label,
.stRadio     [data-testid="stWidgetLabel"],
.stSelectbox [data-testid="stWidgetLabel"],
.stSlider    [data-testid="stWidgetLabel"],
.stCheckbox  [data-testid="stWidgetLabel"],
.stToggle    [data-testid="stWidgetLabel"] {
  color: #4b5678 !important;
}

/* ── Radio buttons ── */
.stRadio > div > label,
.stRadio [data-testid="stMarkdownContainer"] p {
  color: #0d1224 !important;
}
.stRadio [data-baseweb="radio"] span { background-color: #ffffff !important; }

/* ── Selectbox / dropdown ── */
.stSelectbox [data-baseweb="select"] > div,
.stSelectbox [data-baseweb="select"] {
  background-color: #ffffff !important;
  border-color: #c5cde4 !important;
  color: #0d1224 !important;
}
.stSelectbox [data-baseweb="select"] span,
.stSelectbox [data-baseweb="select"] div { color: #0d1224 !important; }
/* Dropdown list items */
li[role="option"], li[role="option"] * { color: #0d1224 !important; background: #ffffff !important; }
li[role="option"]:hover { background: #eef1f8 !important; }

/* ── Slider ── */
.stSlider [data-baseweb="slider"] { background-color: transparent !important; }
.stSlider [data-testid="stTickBarMin"],
.stSlider [data-testid="stTickBarMax"],
.stSlider [data-baseweb="slider"] div { color: #0d1224 !important; }
/* Slider value label */
.stSlider [data-testid="stThumbValue"],
.stSlider p { color: #0d1224 !important; }

/* ── Checkbox ── */
.stCheckbox [data-testid="stMarkdownContainer"] p { color: #0d1224 !important; }
.stCheckbox [data-baseweb="checkbox"] span { border-color: #c5cde4 !important; }

/* ── Toggle ── */
.stToggle   [data-testid="stMarkdownContainer"] p { color: #0d1224 !important; }

/* ── Inputs ── */
.stTextArea  textarea  { background: #ffffff !important; border-color: #c5cde4 !important; color: #0d1224 !important; }
.stTextInput input     { background: #ffffff !important; border-color: #c5cde4 !important; color: #0d1224 !important; }
.stNumberInput input   { background: #ffffff !important; border-color: #c5cde4 !important; color: #0d1224 !important; }

/* ── File uploader ── */
div[data-testid="stFileUploader"] { background: #eef1f8 !important; border-color: #c5cde4 !important; }
div[data-testid="stFileUploader"] span,
div[data-testid="stFileUploader"] p  { color: #4b5678 !important; }

/* ── Caption / small text ── */
.stCaption, .stCaption p, small { color: #8c96b5 !important; }

/* ── Dividers ── */
hr { border-color: #dce1f0 !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab"]         { color: #4b5678 !important; background: #eef1f8 !important; }
.stTabs [aria-selected="true"]       { color: #3d5cf5 !important; background: #ffffff !important; }
.stTabs [data-baseweb="tab-border"]  { background: #dce1f0 !important; }

/* ── Info / warning / error boxes ── */
div[data-testid="stAlert"]           { background: #f4f6fb !important; color: #0d1224 !important; }

/* ── Spinner text ── */
.stSpinner > div > div { border-top-color: #3d5cf5 !important; }
</style>"""


if "dark_mode" not in st.session_state:
    st.session_state["dark_mode"] = True     # default: dark


def apply_theme():
    """Inject theme CSS based on current session state."""
    if st.session_state.get("dark_mode", True):
        st.markdown(_theme_css_dark(), unsafe_allow_html=True)
    else:
        st.markdown(_theme_css_light(), unsafe_allow_html=True)


apply_theme()


# ===========================================================================
#  HTML HELPER COMPONENTS
#  Small, self-contained functions that return HTML strings.
#  Keep presentation logic separate from data logic.
# ===========================================================================

def html_wordmark() -> str:
    # SVG: simple scissors icon (no external dependency)
    icon_svg = (
        '<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">'
        '<circle cx="6" cy="6" r="3"/><circle cx="6" cy="18" r="3"/>'
        '<line x1="20" y1="4" x2="8.12" y2="15.88" stroke="#fff" stroke-width="2"/>'
        '<line x1="14.47" y1="14.48" x2="20" y2="20" stroke="#fff" stroke-width="2"/>'
        '<line x1="8.12" y1="8.12" x2="12" y2="12" stroke="#fff" stroke-width="2"/>'
        '</svg>'
    )
    return f"""
<div class="wordmark">
  <div class="wordmark-logotype">{icon_svg}</div>
  <div class="wordmark-text">
    <div class="wordmark-name">YOLOE Studio</div>
    <div class="wordmark-sub">Block 1 &middot; Mask &amp; Crop</div>
  </div>
</div>"""


def html_sidebar_section(label: str) -> str:
    return f'<span class="sidebar-section">{label}</span>'


def html_status_item(label: str, state: str) -> str:
    """state: 'ok' | 'warn' | 'error'"""
    return (
        f'<div class="status-item status-{state}">'
        f'<span class="status-dot"></span>{label}'
        f'</div>'
    )


def html_pipeline() -> str:
    steps_active  = ["Image", "Prompt", "YOLOE Mask", "Crop"]
    steps_pending = ["Any6D Pose"]
    parts = []
    for step in steps_active:
        parts.append(f'<div class="pipeline-step pipeline-step--active">{step}</div>')
        parts.append('<div class="pipeline-arrow">&rarr;</div>')
    for step in steps_pending:
        parts.append(f'<div class="pipeline-step pipeline-step--pending">{step}</div>')
    return f'<div class="pipeline">{"".join(parts)}</div>'


def html_page_header(title: str, description: str, mode_key: str) -> str:
    eyebrow_classes = {
        "text":   ("Text Prompt",   "eyebrow-text"),
        "visual": ("Visual Prompt", "eyebrow-visual"),
        "free":   ("Free Prompt",   "eyebrow-free"),
    }
    label, css = eyebrow_classes.get(mode_key, ("", "eyebrow-text"))
    return f"""
<div class="page-header">
  <span class="page-header-eyebrow {css}">{label}</span>
  <h1 class="page-header-title">{title}</h1>
  <p class="page-header-desc">{description}</p>
</div>"""


def html_field_label(label: str) -> str:
    return f'<span class="field-label">{label}</span>'


def html_metrics(conf: float, mask_px: int, coverage: float, bbox_size: str) -> str:
    cards = [
        (f"{conf:.3f}", "Confidence"),
        (f"{mask_px:,}", "Mask pixels"),
        (f"{coverage:.1f}%", "Coverage"),
        (bbox_size, "Bbox size"),
    ]
    items = "".join(
        f'<div class="metric-card">'
        f'<div class="metric-card__value">{v}</div>'
        f'<div class="metric-card__label">{l}</div>'
        f'</div>'
        for v, l in cards
    )
    return f'<div class="metrics-row">{items}</div>'


def html_panel_header(label: str) -> str:
    return f'<div class="panel-header">{label}</div>'


def html_crop_meta(origin: tuple, size: tuple, padding: int) -> str:
    return (
        f'<div class="crop-meta">'
        f'<b>Origin</b>&ensp;({origin[0]}, {origin[1]})<br>'
        f'<b>Size</b>&emsp;&ensp;{size[0]} &times; {size[1]} px<br>'
        f'<b>Padding</b>&ensp;{padding} px'
        f'</div>'
    )


def html_detection_item(index: int, label: str, conf: float, mask_px: int) -> str:
    return (
        f'<div class="detection-item">'
        f'<span class="detection-item__index">#{index:02d}</span>'
        f'<span class="detection-item__label">{label}</span>'
        f'<span class="detection-item__conf">{conf:.3f}</span>'
        f'<span class="detection-item__px">{mask_px:,}&thinsp;px</span>'
        f'</div>'
    )


def html_export_header() -> str:
    return '<span class="export-header">Export for Any6D</span>'


# ===========================================================================
#  PURE LOGIC — unchanged from original
# ===========================================================================

def save_upload(up) -> str:
    suffix = Path(up.name).suffix or ".jpg"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    up.seek(0)
    tmp.write(up.read())
    tmp.flush()
    tmp.close()
    return tmp.name


def safe_unlink(*paths):
    for p in paths:
        try:
            if p and os.path.exists(p):
                os.unlink(p)
        except Exception:
            pass


def pil_to_bytes(img, fmt="PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def read_rgb(path: str) -> np.ndarray:
    bgr = cv2.imread(path)
    if bgr is None:
        raise ValueError(f"Cannot read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def ensure_bool_mask(mask: np.ndarray, H: int, W: int) -> np.ndarray:
    if mask.shape != (H, W):
        mask = cv2.resize(mask.astype(np.float32), (W, H))
    return mask.astype(bool)


def largest_component(mask: np.ndarray) -> np.ndarray:
    u = mask.astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(u, connectivity=8)
    if n <= 1:
        return mask
    return labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))


def refine_mask(mask, dilate=0, erode=0):
    m = mask.astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    if erode  > 0: m = cv2.erode (m, k, iterations=erode)
    if dilate > 0: m = cv2.dilate(m, k, iterations=dilate)
    return m.astype(bool)


def tight_bbox(mask):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return None
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    return int(x1), int(y1), int(x2), int(y2)


def crop_object(img_rgb, mask, padding=8):
    bb = tight_bbox(mask)
    if bb is None:
        return None, None, None
    H, W = img_rgb.shape[:2]
    x1, y1, x2, y2 = bb
    x1p = max(0, x1 - padding);  y1p = max(0, y1 - padding)
    x2p = min(W, x2 + padding);  y2p = min(H, y2 + padding)
    rr = img_rgb[y1p:y2p, x1p:x2p]
    rm = mask   [y1p:y2p, x1p:x2p]
    if rr.size == 0:
        return None, None, None
    alpha     = (rm * 255).astype(np.uint8)
    rgba      = np.dstack([rr.copy(), alpha])
    crop_rgba = Image.fromarray(rgba, "RGBA")
    rh, rw    = rr.shape[:2]
    side      = max(rh, rw)
    sq        = np.zeros((side, side, 4), dtype=np.uint8)
    oy = (side - rh) // 2
    ox = (side - rw) // 2
    sq[oy:oy+rh, ox:ox+rw] = rgba
    return crop_rgba, Image.fromarray(sq, "RGBA"), (x1p, y1p, x2p, y2p)


def overlay_mask(img, mask, color=(60, 180, 255), alpha=0.45):
    if mask is None or not mask.any():
        return img.copy()
    out = img.copy().astype(np.float32)
    out[mask] = out[mask] * (1 - alpha) + np.array(color, np.float32) * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def draw_bbox_img(img, bbox, color=(60, 180, 255), t=2):
    out = img.copy()
    x1, y1, x2, y2 = [int(v) for v in bbox]
    cv2.rectangle(out, (x1, y1), (x2, y2), color, t)
    return out


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------

def _load_yoloe(name: str):
    import torch
    from ultralytics import YOLOE
    orig = sys.argv[:]
    sys.argv = ["yoloe_app"]
    try:
        m = YOLOE(name, verbose=False)
    finally:
        sys.argv = orig
    m.to("cuda" if torch.cuda.is_available() else "cpu")
    return m


@st.cache_resource(show_spinner="Loading segmentation model…")
def load_model(name):
    return _load_yoloe(name)


@st.cache_resource(show_spinner="Loading free-prompt model…")
def load_model_pf(name):
    return _load_yoloe(name)


# ---------------------------------------------------------------------------
# Helpers importer
# ---------------------------------------------------------------------------

def _import_helper(fn_name: str):
    try:
        if "helpers" in sys.modules:
            importlib.reload(sys.modules["helpers"])
        import helpers
        fn = getattr(helpers, fn_name, None)
        if fn is None:
            st.warning(f"helpers.py: '{fn_name}' not defined — using fallback")
        return fn
    except ModuleNotFoundError:
        return None
    except Exception as exc:
        st.warning(f"helpers import error: {exc}")
        traceback.print_exc(file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Result parsers
# ---------------------------------------------------------------------------

def parse_best(results, H, W):
    r = results[0] if isinstance(results, list) else results
    if r.boxes is None or len(r.boxes) == 0:
        return None, np.zeros((H, W), dtype=bool), 0.0, results
    best = int(r.boxes.conf.argmax().item())
    bbox = r.boxes.xyxy[best].cpu().numpy().astype(int)
    conf = float(r.boxes.conf[best].item())
    mask = np.zeros((H, W), dtype=bool)
    if r.masks is not None:
        raw  = r.masks.data[best].cpu().numpy().astype(np.float32)
        mask = ensure_bool_mask(raw, H, W)
    else:
        x1, y1, x2, y2 = bbox
        mask[max(0,y1):min(H,y2), max(0,x1):min(W,x2)] = True
    return bbox, mask, conf, results


def parse_all(results, H, W):
    r    = results[0] if isinstance(results, list) else results
    dets = []
    if r.boxes is None:
        return dets
    for i in range(len(r.boxes)):
        bbox = r.boxes.xyxy[i].cpu().numpy().astype(int)
        conf = float(r.boxes.conf[i].item())
        cls  = int(r.boxes.cls[i].item())
        lbl  = r.names.get(cls, str(cls)) if hasattr(r, "names") and r.names else str(cls)
        mask = np.zeros((H, W), dtype=bool)
        if r.masks is not None:
            raw  = r.masks.data[i].cpu().numpy().astype(np.float32)
            mask = ensure_bool_mask(raw, H, W)
        else:
            x1, y1, x2, y2 = bbox
            mask[max(0,y1):min(H,y2), max(0,x1):min(W,x2)] = True
        dets.append({"bbox": bbox, "mask": mask, "conf": conf, "label": lbl})
    return dets


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------

def run_text(model, scene_path, scene_rgb, prompts, conf):
    H, W = scene_rgb.shape[:2]
    fn   = _import_helper("yoloe_text_prompt")
    if fn:
        try:
            bbox, mask, c, res = fn(model, scene_path=scene_path, text_prompts=prompts, conf=conf)
            if mask is not None:
                mask = ensure_bool_mask(np.array(mask, np.float32), H, W)
            return bbox, mask, c, res
        except Exception as exc:
            st.warning(f"helpers.yoloe_text_prompt failed ({exc}) — using fallback")
            traceback.print_exc(file=sys.stderr)
    model.set_classes(prompts)
    res = model.predict(source=scene_path, conf=conf, verbose=False)
    return parse_best(res, H, W)


def run_visual(model, scene_path, scene_rgb, anchor_path, anchor_bbox, conf):
    from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor
    H, W = scene_rgb.shape[:2]
    fn   = _import_helper("yoloe_visual_prompt")
    if fn:
        try:
            bbox, mask, c, res = fn(model, scene_path=scene_path,
                                    anchor_path=anchor_path, anchor_bbox=anchor_bbox, conf=conf)
            if mask is not None:
                mask = ensure_bool_mask(np.array(mask, np.float32), H, W)
            return bbox, mask, c, res
        except Exception as exc:
            st.warning(f"helpers.yoloe_visual_prompt failed ({exc}) — using fallback")
            traceback.print_exc(file=sys.stderr)
    bbox_arr = np.array(anchor_bbox, np.float32)
    if bbox_arr.ndim == 1:
        bbox_arr = bbox_arr[np.newaxis, :]
    res = model.predict(
        source=scene_path,
        visual_prompts={"bboxes": bbox_arr.tolist(), "cls": list(range(len(bbox_arr)))},
        refer_image=anchor_path,
        predictor=YOLOEVPSegPredictor,
        conf=conf,
        verbose=False,
    )
    return parse_best(res, H, W)


def run_free(model_pf, scene_path, scene_rgb, conf):
    H, W = scene_rgb.shape[:2]
    fn   = _import_helper("yoloe_free_prompt")
    if fn:
        try:
            bbox, mask, c, res = fn(model_pf, scene_path=scene_path, conf=conf)
            if mask is not None:
                mask = ensure_bool_mask(np.array(mask, np.float32), H, W)
            return bbox, mask, c, res
        except Exception as exc:
            st.warning(f"helpers.yoloe_free_prompt failed ({exc}) — using fallback")
            traceback.print_exc(file=sys.stderr)
    res = model_pf.predict(source=scene_path, conf=conf, verbose=False)
    return parse_best(res, H, W)


# ===========================================================================
#  RESULT RENDERER
# ===========================================================================

def show_results(
    scene_rgb, mask, bbox, conf_val,
    crop_padding, use_largest, erode_iters, dilate_iters,
    mode_label="", key_suffix=""
):
    H, W = scene_rgb.shape[:2]

    if mask is None or not np.asarray(mask).any():
        st.warning("No object detected — try lowering the confidence threshold or adjusting the prompt.")
        return

    # Mask post-processing
    mask = ensure_bool_mask(np.asarray(mask, np.float32), H, W)
    if use_largest:
        mask = largest_component(mask)
    if erode_iters > 0 or dilate_iters > 0:
        mask = refine_mask(mask, dilate=dilate_iters, erode=erode_iters)
    if not mask.any():
        st.warning("Mask is empty after refinement — reduce erosion or increase dilation.")
        return

    # Build visualisations
    overlay   = overlay_mask(scene_rgb, mask)
    if bbox is not None:
        overlay = draw_bbox_img(overlay, bbox)
    mask_vis  = np.stack([(mask * 255).astype(np.uint8)] * 3, axis=-1)
    crop_rgba, crop_sq, crop_bbox = crop_object(scene_rgb, mask, crop_padding)

    # Crop preview on neutral background
    dark = st.session_state.get("dark_mode", True)
    bg_color = (13, 18, 37) if dark else (244, 246, 251)
    if crop_sq:
        bg = Image.new("RGB", crop_sq.size, bg_color)
        bg.paste(crop_sq, mask=crop_sq.split()[3])
        crop_prev = np.array(bg)
    else:
        crop_prev = None

    # Metrics
    mask_px  = int(mask.sum())
    mask_pct = mask_px / (H * W) * 100
    bbox_str = f"{int(bbox[2]-bbox[0])} × {int(bbox[3]-bbox[1])}" if bbox is not None else "—"
    st.markdown(html_metrics(conf_val, mask_px, mask_pct, bbox_str), unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # Three result panels
    col_overlay, col_mask, col_crop = st.columns(3)

    with col_overlay:
        st.markdown(html_panel_header("Mask overlay"), unsafe_allow_html=True)
        st.image(overlay, use_container_width=True)

    with col_mask:
        st.markdown(html_panel_header("Binary mask"), unsafe_allow_html=True)
        st.image(mask_vis, use_container_width=True)

    with col_crop:
        st.markdown(html_panel_header("Cropped object — Any6D input"), unsafe_allow_html=True)
        if crop_prev is not None:
            st.image(crop_prev, use_container_width=True)
            if crop_bbox:
                cx1, cy1, cx2, cy2 = crop_bbox
                st.markdown(
                    html_crop_meta(
                        origin=(cx1, cy1),
                        size=(cx2 - cx1, cy2 - cy1),
                        padding=crop_padding,
                    ),
                    unsafe_allow_html=True,
                )
        else:
            st.warning("Crop failed — mask may be empty or at the image boundary.")

    # Downloads
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(html_export_header(), unsafe_allow_html=True)
    mask_pil = Image.fromarray((mask * 255).astype(np.uint8), "L")
    meta = {
        "mode":           mode_label,
        "conf":           round(float(conf_val), 4),
        "bbox_xyxy":      [int(v) for v in bbox] if bbox is not None else None,
        "crop_bbox_xyxy": list(crop_bbox) if crop_bbox else None,
        "mask_pixels":    mask_px,
        "coverage_pct":   round(mask_pct, 3),
        "image_hw":       [H, W],
        "crop_padding":   crop_padding,
    }
    dl1, dl2, dl3, dl4 = st.columns(4)
    dl1.download_button("mask.png",      pil_to_bytes(mask_pil),                            "mask.png",      "image/png",         key=f"dlm_{key_suffix}")
    dl2.download_button("crop_rgba.png", pil_to_bytes(crop_rgba) if crop_rgba else b"",      "crop_rgba.png", "image/png",         disabled=not crop_rgba, key=f"dlr_{key_suffix}")
    dl3.download_button("crop_sq.png",   pil_to_bytes(crop_sq)   if crop_sq   else b"",      "crop_sq.png",   "image/png",         disabled=not crop_sq,   key=f"dls_{key_suffix}")
    dl4.download_button("meta.json",     json.dumps(meta, indent=2).encode(),                "meta.json",     "application/json",  key=f"dlj_{key_suffix}")


# ===========================================================================
#  SIDEBAR
# ===========================================================================

with st.sidebar:
    st.markdown(html_wordmark(), unsafe_allow_html=True)

    # Theme toggle
    st.toggle("Dark mode", value=st.session_state["dark_mode"], key="dark_mode")
    apply_theme()

    # Prompt mode
    st.markdown(html_sidebar_section("Prompt mode"), unsafe_allow_html=True)
    mode = st.radio(
        label="mode",
        options=[" Text Prompt", " Visual Prompt", " Free-Prompt"],
        label_visibility="collapsed",
    )

    st.markdown("<hr>", unsafe_allow_html=True)

    # Models
    st.markdown(html_sidebar_section("Models"), unsafe_allow_html=True)
    model_choice = st.selectbox(
        "Segmentation model",
        ["yoloe-26l-seg.pt", "yoloe-11l-seg.pt", "yoloe-11m-seg.pt", "yoloe-11s-seg.pt"],
    )
    pf_choice = st.selectbox(
        "Free-prompt model",
        ["yoloe-26l-seg-pf.pt", "yoloe-11l-seg-pf.pt", "yoloe-11s-seg-pf.pt"],
    )

    st.markdown("<hr>", unsafe_allow_html=True)

    # Detection settings
    st.markdown(html_sidebar_section("Detection"), unsafe_allow_html=True)
    conf_thresh  = st.slider("Confidence threshold", 0.05, 0.95, 0.05, 0.05)

    st.markdown(html_sidebar_section("Mask refinement"), unsafe_allow_html=True)
    use_largest  = st.checkbox("Keep largest component only", value=True)
    _c1, _c2     = st.columns(2)
    erode_iters  = _c1.slider("Erode",  0, 5, 0)
    dilate_iters = _c2.slider("Dilate", 0, 5, 0)

    st.markdown(html_sidebar_section("Crop"), unsafe_allow_html=True)
    crop_padding = st.slider("Padding (px)", 0, 80, 12)

    st.markdown("<hr>", unsafe_allow_html=True)

    # Environment status
    st.markdown(html_sidebar_section("Environment"), unsafe_allow_html=True)
    _helpers_ok = (APP_DIR / "helpers.py").exists()
    _status_items = [html_status_item(
        "helpers.py found" if _helpers_ok else "helpers.py missing",
        "ok" if _helpers_ok else "error",
    )]
    try:
        import clip
        _status_items.append(html_status_item("CLIP ready", "ok"))
    except ImportError:
        _status_items.append(html_status_item("CLIP not installed", "error"))
    try:
        import mobileclip
        _status_items.append(html_status_item("MobileCLIP ready", "ok"))
    except ImportError:
        _status_items.append(html_status_item("MobileCLIP not installed", "warn"))

    st.markdown(
        f'<div class="status-row">{"".join(_status_items)}</div>',
        unsafe_allow_html=True,
    )
    st.caption(f"{APP_DIR.name}/")


# ===========================================================================
#  MAIN AREA
# ===========================================================================

# Pipeline strip
st.markdown(html_pipeline(), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# MODE 1 — TEXT PROMPT
# ---------------------------------------------------------------------------

if " Text Prompt" in mode:
    st.markdown(
        html_page_header(
            title="Text Prompt Detection",
            description=(
                "Enter class names, one per line. "
                "Select which tags to detect — every selected class is detected "
                "and cropped individually."
            ),
            mode_key="text",
        ),
        unsafe_allow_html=True,
    )

    col_img, col_cfg = st.columns([1.5, 1])

    with col_img:
        st.markdown(html_field_label("Scene image"), unsafe_allow_html=True)
        scene_up = st.file_uploader(
            "scene",
            type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed",
            key="t_scene",
        )
        if scene_up:
            scene_up.seek(0)
            st.image(scene_up, use_container_width=True)

    with col_cfg:
        st.markdown(html_field_label("Object class names — one per line"), unsafe_allow_html=True)
        raw_text = st.text_area(
            label="classes",
            height=150,
            label_visibility="collapsed",
            key="t_raw",
            placeholder="tennis ball\nyellow ball\nball\nmustard bottle",
        )

        lines        = [l.strip() for l in raw_text.strip().splitlines() if l.strip()]
        unique_lines = list(dict.fromkeys(lines))
        selected     = []

        if unique_lines:
            st.markdown(
                html_field_label("Select classes to detect"),
                unsafe_allow_html=True,
            )
            tag_cols = st.columns(min(len(unique_lines), 3))
            for i, tag in enumerate(unique_lines):
                safe_key = "tag_" + "".join(c if c.isalnum() else "_" for c in tag)[:40]
                with tag_cols[i % len(tag_cols)]:
                    if st.checkbox(tag, value=True, key=safe_key):
                        selected.append(tag)

            if selected:
                st.caption(f"{len(selected)} class(es) selected: {', '.join(selected)}")
            else:
                st.caption("Select at least one class.")
        else:
            selected = []

        st.markdown("<br>", unsafe_allow_html=True)
        run_t = st.button("Run detection", use_container_width=True, key="run_t")

    if run_t:
        if not scene_up:
            st.error("Please upload a scene image."); st.stop()
        if not selected:
            st.error("Please select at least one class."); st.stop()

        scene_path = None
        try:
            scene_path = save_upload(scene_up)
            scene_rgb  = read_rgb(scene_path)
            model      = load_model(model_choice)
            with st.spinner(f"Detecting {len(selected)} class(es)…"):
                # Prime the model with all classes
                run_text(model, scene_path, scene_rgb, selected, conf_thresh)
        except Exception as exc:
            st.error(f"Detection failed: {exc}")
            traceback.print_exc(file=sys.stderr)
            st.stop()

        st.markdown("<hr>", unsafe_allow_html=True)

        # Run per-class detection and crop — scene_path stays alive until all done
        try:
            if len(selected) == 1:
                _cls = selected[0]
                try:
                    _bbox, _mask, _conf, _ = run_text(model, scene_path, scene_rgb, [_cls], conf_thresh)
                except Exception:
                    _bbox, _mask, _conf = None, None, 0.0
                show_results(
                    scene_rgb, _mask, _bbox, _conf,
                    crop_padding, use_largest, erode_iters, dilate_iters,
                    mode_label=f"text:{_cls}", key_suffix="t0",
                )
            else:
                tabs = st.tabs(selected)
                for ti, _cls in enumerate(selected):
                    with tabs[ti]:
                        try:
                            _bbox, _mask, _conf, _ = run_text(model, scene_path, scene_rgb, [_cls], conf_thresh)
                        except Exception:
                            _bbox, _mask, _conf = None, None, 0.0
                        if _mask is None or not np.asarray(_mask).any():
                            st.warning(f"No detection for '{_cls}'. Try a lower confidence threshold.")
                        else:
                            show_results(
                                scene_rgb, _mask, _bbox, _conf,
                                crop_padding, use_largest, erode_iters, dilate_iters,
                                mode_label=f"text:{_cls}", key_suffix=f"t{ti}",
                            )
        finally:
            safe_unlink(scene_path)  # always runs, even if show_results raises


# ---------------------------------------------------------------------------
# MODE 2 — VISUAL PROMPT
# ---------------------------------------------------------------------------

elif " Visual Prompt" in mode:
    st.markdown(
        html_page_header(
            title="Visual Prompt Detection",
            description=(
                "Upload a reference image and define the object's bounding box "
                "using the sliders. The model finds the same object in the scene."
            ),
            mode_key="visual",
        ),
        unsafe_allow_html=True,
    )

    col_anc, col_scn = st.columns(2)
    anchor_up = scene_up = None

    with col_anc:
        st.markdown(html_field_label("Reference image"), unsafe_allow_html=True)
        anchor_up = st.file_uploader(
            "anchor",
            type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed",
            key="v_anchor",
        )
        drawn_bbox = None

        if anchor_up:
            anchor_up.seek(0)
            anchor_pil      = Image.open(anchor_up).convert("RGB")
            orig_W, orig_H  = anchor_pil.size

            # Init bbox in session state
            # Always clamp stored values to current image dimensions.
            # This prevents a Streamlit crash when the user uploads a second
            # anchor image that is smaller than the previous one (slider
            # value > max raises ValueError).
            defaults = {"bbox_x1": 0, "bbox_y1": 0, "bbox_x2": orig_W, "bbox_y2": orig_H}
            for k, default in defaults.items():
                if k not in st.session_state:
                    st.session_state[k] = default
            # Clamp to current image bounds
            st.session_state["bbox_x1"] = max(0,      min(st.session_state["bbox_x1"], orig_W - 1))
            st.session_state["bbox_y1"] = max(0,      min(st.session_state["bbox_y1"], orig_H - 1))
            st.session_state["bbox_x2"] = max(1,      min(st.session_state["bbox_x2"], orig_W))
            st.session_state["bbox_y2"] = max(1,      min(st.session_state["bbox_y2"], orig_H))

            st.markdown(html_field_label("Bounding box"), unsafe_allow_html=True)
            row1 = st.columns(2)
            row2 = st.columns(2)
            _bx1 = row1[0].slider("x1 — left",   0, orig_W - 1, st.session_state["bbox_x1"], key="vx1")
            _bx2 = row1[1].slider("x2 — right",  1, orig_W,     st.session_state["bbox_x2"], key="vx2")
            _by1 = row2[0].slider("y1 — top",    0, orig_H - 1, st.session_state["bbox_y1"], key="vy1")
            _by2 = row2[1].slider("y2 — bottom", 1, orig_H,     st.session_state["bbox_y2"], key="vy2")

            # Ensure valid box
            _bx1, _bx2 = min(_bx1, _bx2 - 1), max(_bx1 + 1, _bx2)
            _by1, _by2 = min(_by1, _by2 - 1), max(_by1 + 1, _by2)
            st.session_state.update({
                "bbox_x1": _bx1, "bbox_x2": _bx2,
                "bbox_y1": _by1, "bbox_y2": _by2,
            })

            # Live preview with bbox drawn
            preview = np.array(anchor_pil.copy())
            ov      = preview.copy()
            cv2.rectangle(ov, (_bx1, _by1), (_bx2, _by2), (91, 156, 246), -1)
            preview = cv2.addWeighted(ov, 0.18, preview, 0.82, 0)
            cv2.rectangle(preview, (_bx1, _by1), (_bx2, _by2), (91, 156, 246), 2)
            for cx, cy in [(_bx1, _by1), (_bx2, _by1), (_bx1, _by2), (_bx2, _by2)]:
                cv2.circle(preview, (cx, cy), 5, (91, 156, 246), -1)
            cv2.putText(
                preview, f"{_bx2 - _bx1} x {_by2 - _by1} px",
                (_bx1 + 4, max(_by1 - 6, 14)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (91, 156, 246), 2,
            )
            st.image(
                preview,
                use_container_width=True,
                caption=f"Box: [{_bx1}, {_by1}, {_bx2}, {_by2}]",
            )
            drawn_bbox = [_bx1, _by1, _bx2, _by2]

    with col_scn:
        st.markdown(html_field_label("Scene image"), unsafe_allow_html=True)
        scene_up = st.file_uploader(
            "scene",
            type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed",
            key="v_scene",
        )
        if scene_up:
            scene_up.seek(0)
            st.image(scene_up, use_container_width=True)
        if drawn_bbox:
            st.info(f"Reference box: [{drawn_bbox[0]}, {drawn_bbox[1]}, {drawn_bbox[2]}, {drawn_bbox[3]}]")
        else:
            st.info("Upload a reference image and set the bounding box with the sliders.")

    run_v = st.button("Run detection", use_container_width=True, key="run_v")

    if run_v:
        if not anchor_up:   st.error("Please upload a reference image."); st.stop()
        if not scene_up:    st.error("Please upload a scene image.");     st.stop()
        if not drawn_bbox:  st.error("Please set the bounding box.");     st.stop()

        # Validate bbox
        _db = [
            min(drawn_bbox[0], drawn_bbox[2]), min(drawn_bbox[1], drawn_bbox[3]),
            max(drawn_bbox[0], drawn_bbox[2]), max(drawn_bbox[1], drawn_bbox[3]),
        ]
        if _db[2] - _db[0] < 4 or _db[3] - _db[1] < 4:
            st.error(f"Bounding box too small: {_db}. Please draw a larger box."); st.stop()
        drawn_bbox = _db

        anchor_path = scene_path = None
        try:
            anchor_up.seek(0); scene_up.seek(0)
            anchor_path = save_upload(anchor_up)
            scene_path  = save_upload(scene_up)
            scene_rgb   = read_rgb(scene_path)
            anchor_bbox = np.array([drawn_bbox], dtype=np.float32)
            model       = load_model(model_choice)
            with st.spinner("Detecting via visual prompt…"):
                bbox, mask, conf_val, _ = run_visual(
                    model, scene_path, scene_rgb, anchor_path, anchor_bbox, conf_thresh
                )
        except Exception as exc:
            st.error(f"Detection failed: {exc}")
            traceback.print_exc(file=sys.stderr)
            st.stop()
        finally:
            safe_unlink(anchor_path, scene_path)

        st.markdown("<hr>", unsafe_allow_html=True)
        show_results(
            scene_rgb, mask, bbox, conf_val,
            crop_padding, use_largest, erode_iters, dilate_iters,
            mode_label="visual", key_suffix="v",
        )


# ---------------------------------------------------------------------------
# MODE 3 — FREE PROMPT
# ---------------------------------------------------------------------------

else:
    st.markdown(
        html_page_header(
            title="Free Prompt Detection",
            description=(
                "No class names or reference image required. "
                "The model detects all objects automatically. "
                "Select any detected object to crop and export for Any6D."
            ),
            mode_key="free",
        ),
        unsafe_allow_html=True,
    )

    col_img, col_info = st.columns([1.5, 1])
    with col_img:
        st.markdown(html_field_label("Scene image"), unsafe_allow_html=True)
        scene_up = st.file_uploader(
            "scene",
            type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed",
            key="f_scene",
        )
        if scene_up:
            scene_up.seek(0)
            st.image(scene_up, use_container_width=True)

    with col_info:
        st.info(
            "The prompt-free model detects every visible object "
            "without any prior knowledge. Results are ranked by confidence. "
            "Click **Crop** on any row to generate the Any6D input."
        )
        run_f = st.button("Run detection", use_container_width=True, key="run_f")

    # Cache detections — avoid re-running model on Crop button click
    if run_f:
        if not scene_up:
            st.error("Please upload a scene image."); st.stop()

        scene_path = None
        try:
            scene_up.seek(0)
            scene_path = save_upload(scene_up)
            scene_rgb  = read_rgb(scene_path)
            H, W       = scene_rgb.shape[:2]
            m_pf       = load_model_pf(pf_choice)
            with st.spinner("Detecting all objects…"):
                _, _, _, raw_results = run_free(m_pf, scene_path, scene_rgb, conf_thresh)
                all_dets = parse_all(raw_results, H, W)
            all_dets.sort(key=lambda d: d["conf"], reverse=True)
            st.session_state["free_dets"]      = all_dets
            st.session_state["free_scene_rgb"] = scene_rgb
        except Exception as exc:
            st.error(f"Detection failed: {exc}")
            traceback.print_exc(file=sys.stderr)
            st.stop()
        finally:
            safe_unlink(scene_path)

    all_dets       = st.session_state.get("free_dets",      [])
    scene_rgb_free = st.session_state.get("free_scene_rgb", None)

    if not all_dets or scene_rgb_free is None:
        if run_f:
            st.warning("No objects detected. Try lowering the confidence threshold.")
        st.stop()

    # Overview with all detections drawn
    PALETTE = [
        (91,  156, 246), (163, 122, 245), (61,  214, 160),
        (246, 180, 91),  (246, 91,  91),  (91,  246, 220),
    ]
    overview = scene_rgb_free.copy()
    for i, d in enumerate(all_dets):
        col_rgb = PALETTE[i % len(PALETTE)]
        overview = overlay_mask(overview, d["mask"], color=col_rgb, alpha=0.28)
        x1, y1, x2, y2 = [int(v) for v in d["bbox"]]
        cv2.rectangle(overview, (x1, y1), (x2, y2), col_rgb, 2)
        cv2.putText(
            overview,
            f"[{i:02d}] {d['label']}  {d['conf']:.2f}",
            (x1, max(y1 - 6, 14)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, col_rgb, 2,
        )

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(html_field_label("All detections"), unsafe_allow_html=True)
    st.image(overview, use_container_width=True)

    # Summary metrics
    _n   = len(all_dets)
    _cls = len(set(d["label"] for d in all_dets))
    _bc  = all_dets[0]["conf"]
    st.markdown(
        f'<div class="metrics-row">'
        f'<div class="metric-card"><div class="metric-card__value">{_n}</div><div class="metric-card__label">Objects</div></div>'
        f'<div class="metric-card"><div class="metric-card__value">{_cls}</div><div class="metric-card__label">Classes</div></div>'
        f'<div class="metric-card"><div class="metric-card__value">{_bc:.3f}</div><div class="metric-card__label">Best conf</div></div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Object list
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(html_field_label("Select an object to crop"), unsafe_allow_html=True)

    for i, d in enumerate(all_dets):
        col_det, col_btn = st.columns([5, 1])
        mask_px = int(d["mask"].sum())
        col_det.markdown(
            html_detection_item(i, d["label"], d["conf"], mask_px),
            unsafe_allow_html=True,
        )
        if col_btn.button("Crop", key=f"crop_{i}"):
            st.markdown(f"**Object {i:02d} — {d['label']}**")
            show_results(
                scene_rgb_free, d["mask"], d["bbox"], d["conf"],
                crop_padding, use_largest, erode_iters, dilate_iters,
                mode_label=f"free/{d['label']}", key_suffix=f"f{i}",
            )
            st.markdown("<hr>", unsafe_allow_html=True)
