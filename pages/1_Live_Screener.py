# ============================================================
# MONARCH PRO ENGINE — v6  SCORING ENGINE OVERHAUL
# Changes vs v5:
#   SCORING:
#     I-01 Bulk scorer: score = signal_strength × coverage
#          Prevents stocks with sparse signals ranking above
#          stocks with full signal coverage.
#     I-02 RS vs Nifty: volatility-normalised alpha via tanh
#          Replaces fixed ±2% band — adapts to stock's own σ.
#     I-04 MA Structure: EMA9/EMA50 ratio percentile-ranked
#          over 250d history + alignment bonus. Continuous,
#          no step thresholds.
#     I-05 Momentum Acceleration: EMA5−EMA20 velocity diff,
#          percentile-ranked over 200d. 0-4 pts bonus (was
#          binary 0/1/3 pts from crude RS delta comparison).
#     I-06 VolCont (ATR5/ATR20): percentile-ranked over 250d,
#          inverted — low ratio = high score. Replaces 4
#          hard tier thresholds (0.55/0.65/0.75/0.85).
#     I-07 RCI (range5/range20): same percentile treatment.
#          Replaces 3 hard tier thresholds (0.40/0.55/0.70).
#     I-08 52-week position score: percentile-ranked position
#          within 52w high/low range. 0-3 pts bonus. True
#          momentum leaders near highs now rewarded.
#     I-10 Market breadth: VIX-level continuous scoring
#          (−8 to +2 pts) + Nifty trend score (−8 to +4 pts).
#          Replaces binary −8/−5 step penalties.
#     I-11 Sector RS: alpha in σ-units via tanh. Replaces
#          fixed ±3% band with vol-adaptive normalisation.
#   PERFORMANCE:
#     I-14 Market context cache TTL: 300s → 900s (15 min)
# ============================================================

import streamlit as st
import requests, gzip, json, time, io, urllib.parse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed   # PERF: concurrent fetch
import plotly.graph_objects as go
import yfinance as yf
import feedparser
import os

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(layout="wide", page_title="MONARCH PRO — Terminal")


# ── BLOOMBERG TERMINAL THEME v5 ── full readability overhaul
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600;700&display=swap');

:root {
    --bb-bg:      #0a0a0a;
    --bb-surface: #111111;
    --bb-surface2:#161616;
    --bb-border:  #2a2a2a;
    --bb-border2: #3a3a3a;
    --bb-amber:   #ff8c00;
    --bb-amber2:  #ffb347;
    --bb-green:   #00d084;
    --bb-green2:  #00ff88;
    --bb-red:     #ff3b3b;
    --bb-blue:    #1e90ff;
    --bb-cyan:    #00ccff;
    --bb-purple:  #cc88ff;
    --bb-white:   #e8e8e8;
    --bb-white2:  #c8c8c8;
    --bb-muted:   #888888;
    --bb-dim:     #555555;
    --bb-mono:    'IBM Plex Mono', monospace;
}

/* ── GLOBAL BASE ── */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > section,
.main .block-container {
    background-color: var(--bb-bg) !important;
    color: var(--bb-white) !important;
    font-family: var(--bb-mono) !important;
}
/* Catch-all: every text node readable */
p, span, div, label, li, caption,
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] span,
[data-testid="stMarkdownContainer"] li {
    color: var(--bb-white2);
    font-family: var(--bb-mono) !important;
}
[data-testid="stMarkdownContainer"] > div { color: var(--bb-white2) !important; }

/* ── SIDEBAR ── */
[data-testid="stSidebar"], [data-testid="stSidebar"] > div {
    background-color: #060606 !important;
    border-right: 1px solid var(--bb-border) !important;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] div {
    color: var(--bb-muted) !important;
    font-size: 0.67rem !important;
    letter-spacing: 0.05em !important;
}

/* ── HEADERS ── */
h1 { color: var(--bb-amber) !important; font-size: 1.05rem !important;
     font-weight: 700 !important; letter-spacing: 0.18em !important;
     text-transform: uppercase !important;
     border-bottom: 2px solid var(--bb-amber) !important;
     padding-bottom: 6px !important; margin-bottom: 12px !important; }
h2 { color: var(--bb-amber2) !important; font-size: 0.85rem !important;
     font-weight: 600 !important; letter-spacing: 0.12em !important;
     text-transform: uppercase !important;
     border-bottom: 1px solid #2a2a2a !important; padding-bottom: 4px !important; }
h3 { color: var(--bb-white) !important; font-size: 0.78rem !important;
     font-weight: 600 !important; letter-spacing: 0.1em !important;
     text-transform: uppercase !important; }
h4, h5, h6 { color: var(--bb-muted) !important; font-size: 0.70rem !important;
             letter-spacing: 0.08em !important; text-transform: uppercase !important; }

/* ── METRIC BOXES ── */
[data-testid="metric-container"] {
    background: var(--bb-surface) !important; border-radius: 0 !important;
    border: 1px solid var(--bb-border) !important;
    border-left: 3px solid var(--bb-amber) !important;
    padding: 8px 12px !important;
}
[data-testid="stMetricLabel"] p,
[data-testid="metric-container"] label {
    color: var(--bb-muted) !important; font-size: 0.58rem !important;
    letter-spacing: 0.12em !important; text-transform: uppercase !important;
}
[data-testid="stMetricValue"] {
    color: var(--bb-amber) !important; font-size: 1.05rem !important; font-weight: 700 !important;
}
[data-testid="stMetricDelta"] { font-size: 0.62rem !important; }

/* ── DATAFRAME — full Bloomberg terminal ── */
[data-testid="stDataFrame"] {
    border: 1px solid var(--bb-amber) !important; border-radius: 0 !important;
}
/* header cells */
.stDataFrame thead tr th,
[data-testid="stDataFrame"] table thead tr th {
    background-color: #1a1200 !important;
    color: var(--bb-amber) !important;
    font-family: var(--bb-mono) !important;
    font-size: 0.60rem !important; font-weight: 700 !important;
    letter-spacing: 0.14em !important; text-transform: uppercase !important;
    border-bottom: 2px solid var(--bb-amber) !important;
    border-right: 1px solid #2a2a2a !important;
    padding: 6px 10px !important; white-space: nowrap !important;
}
/* body cells */
.stDataFrame tbody tr td,
[data-testid="stDataFrame"] table tbody tr td {
    background-color: #0d0d0d !important;
    color: var(--bb-white) !important;
    font-family: var(--bb-mono) !important;
    font-size: 0.68rem !important;
    border-bottom: 1px solid #1a1a1a !important;
    border-right: 1px solid #1a1a1a !important;
    padding: 4px 10px !important; white-space: nowrap !important;
}
.stDataFrame tbody tr:nth-child(odd)  td { background-color: #111111 !important; }
.stDataFrame tbody tr:nth-child(even) td { background-color: #0d0d0d !important; }
.stDataFrame tbody tr:hover td {
    background-color: #1f1400 !important; color: var(--bb-amber) !important;
}
/* index column dim */
.stDataFrame tbody tr td:first-child { color: var(--bb-dim) !important; font-size: 0.60rem !important; }

/* ── BUTTONS ── */
.stButton > button {
    background: #140e00 !important; color: var(--bb-amber) !important;
    border: 1px solid var(--bb-amber) !important; border-radius: 0 !important;
    font-family: var(--bb-mono) !important; font-size: 0.70rem !important;
    font-weight: 600 !important; letter-spacing: 0.1em !important;
    text-transform: uppercase !important; padding: 6px 18px !important;
}
.stButton > button:hover { background: var(--bb-amber) !important; color: #000 !important; }

/* ── INPUTS ── */
.stSelectbox > div > div, .stTextInput > div > div, .stNumberInput > div > div {
    background: var(--bb-surface) !important; border: 1px solid var(--bb-border2) !important;
    border-radius: 0 !important; color: var(--bb-white) !important;
    font-family: var(--bb-mono) !important; font-size: 0.72rem !important;
}
.stSelectbox label, .stTextInput label, .stNumberInput label,
.stDateInput label, .stSlider label, .stCheckbox label, .stRadio label {
    color: var(--bb-muted) !important; font-size: 0.62rem !important;
    letter-spacing: 0.1em !important; text-transform: uppercase !important;
}
.stDateInput input, .stNumberInput input {
    background: var(--bb-surface) !important; color: var(--bb-white) !important;
    border: 1px solid var(--bb-border2) !important; border-radius: 0 !important;
    font-family: var(--bb-mono) !important;
}

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] {
    background: #080808 !important; border-bottom: 2px solid var(--bb-amber) !important; gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important; color: var(--bb-muted) !important;
    font-family: var(--bb-mono) !important; font-size: 0.63rem !important;
    font-weight: 600 !important; letter-spacing: 0.1em !important;
    text-transform: uppercase !important; border-radius: 0 !important;
    border-right: 1px solid var(--bb-border) !important; padding: 8px 14px !important;
}
.stTabs [data-baseweb="tab"]:hover {
    background: #1a1200 !important; color: var(--bb-amber2) !important;
}
.stTabs [aria-selected="true"] {
    background: #1a1200 !important; color: var(--bb-amber) !important;
    border-bottom: 3px solid var(--bb-amber) !important; font-weight: 700 !important;
}
.stTabs [data-baseweb="tab-panel"] { background: var(--bb-bg) !important; padding: 0 !important; }

/* ── DIVIDER ── */
hr { border-color: #1e1e1e !important; margin: 10px 0 !important; }

/* ── ALERTS ── */
[data-testid="stAlert"], .stSuccess, .stWarning, .stInfo, .stError {
    border-radius: 0 !important; font-family: var(--bb-mono) !important; font-size: 0.70rem !important;
}
[data-testid="stAlert"] p { font-size: 0.70rem !important; color: inherit !important; }

/* ── PROGRESS BAR ── */
.stProgress > div > div { background: var(--bb-amber) !important; }

/* ── CAPTION ── */
.stCaption, [data-testid="stCaptionContainer"] p {
    color: var(--bb-dim) !important; font-size: 0.60rem !important;
    letter-spacing: 0.06em !important;
}

/* ── EXPANDER ── */
.streamlit-expanderHeader, [data-testid="stExpander"] summary {
    background: var(--bb-surface) !important; color: var(--bb-amber) !important;
    font-family: var(--bb-mono) !important; font-size: 0.92rem !important;
    font-weight: 600 !important; letter-spacing: 0.1em !important;
    text-transform: uppercase !important; border-radius: 0 !important;
    border: 1px solid var(--bb-border) !important;
}
/* Kill every possible form of the Streamlit expander arrow icon:
   - SVG element (newer Streamlit)
   - Material Icons <span> that renders "keyboard_arrow_right" / "keyboard_arrow_down"
   - Any <span> inside summary that uses icon fonts */
[data-testid="stExpander"] summary svg,
[data-testid="stExpander"] summary .material-icons,
[data-testid="stExpander"] summary span[data-testid="stExpanderToggleIcon"],
[data-testid="stExpander"] summary > div > span:first-child,
[data-testid="stExpander"] summary > span:first-child {
    display: none !important;
    visibility: hidden !important;
    width: 0 !important;
    overflow: hidden !important;
    font-size: 0 !important;
    color: transparent !important;
}
[data-testid="stExpander"] {
    border: 1px solid var(--bb-border) !important; border-radius: 0 !important;
}
[data-testid="stExpander"] [data-testid="stMarkdownContainer"] p,
[data-testid="stExpander"] [data-testid="stMarkdownContainer"] li,
[data-testid="stExpander"] [data-testid="stMarkdownContainer"] td {
    color: var(--bb-white2) !important; font-size: 0.68rem !important;
}
[data-testid="stExpander"] [data-testid="stMarkdownContainer"] th {
    color: var(--bb-amber) !important; background: #1a1200 !important;
}

/* ── MARKDOWN TABLES ── */
[data-testid="stMarkdownContainer"] table {
    border-collapse: collapse !important; font-family: var(--bb-mono) !important;
    font-size: 0.68rem !important; width: 100% !important;
}
[data-testid="stMarkdownContainer"] th {
    background: #1a1200 !important; color: var(--bb-amber) !important;
    padding: 5px 10px !important; border: 1px solid var(--bb-border) !important;
    letter-spacing: 0.08em !important; text-transform: uppercase !important;
}
[data-testid="stMarkdownContainer"] td {
    color: var(--bb-white2) !important; padding: 4px 10px !important;
    border: 1px solid #1e1e1e !important;
}
[data-testid="stMarkdownContainer"] tr:nth-child(even) td { background: #0d0d0d !important; }
[data-testid="stMarkdownContainer"] strong { color: var(--bb-amber) !important; }
[data-testid="stMarkdownContainer"] code {
    background: #1a1200 !important; color: var(--bb-amber2) !important;
    padding: 1px 5px !important; border-radius: 0 !important;
    font-family: var(--bb-mono) !important; font-size: 0.68rem !important;
}

/* ── SCROLLBAR ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--bb-bg); }
::-webkit-scrollbar-thumb { background: #333; border-radius: 0; }
::-webkit-scrollbar-thumb:hover { background: var(--bb-amber); }

/* ══════════════════════════════════════════════════════════════
   FIX 1 — KILL keyboard_double_arrow / keyboard_arrow_right
   These are Streamlit sidebar nav icons rendered by Material Icons font.
   Target every possible selector across all Streamlit versions.
   ══════════════════════════════════════════════════════════════ */

/* Sidebar page-nav icon spans (the ones showing "keyboard_double_a..." text) */
[data-testid="stSidebarNavLink"] span[data-testid="stIconMaterial"],
[data-testid="stSidebarNavLink"] span.material-icons,
[data-testid="stSidebarNavLink"] span[class*="icon"],
[data-testid="stSidebarNavLink"] svg,
[data-testid="stNavLink"] span[data-testid="stIconMaterial"],
[data-testid="stNavLink"] span.material-icons,
[data-testid="stNavLink"] svg,
/* Expander toggle icons */
[data-testid="stExpander"] summary [data-testid="stExpanderToggleIcon"],
[data-testid="stExpander"] summary [data-testid="stExpanderToggleIcon"] *,
[data-testid="stExpander"] summary svg,
[data-testid="stExpander"] summary .material-icons,
[data-testid="stExpander"] summary > div > span:first-child,
[data-testid="stExpander"] summary > span:first-child,
[data-testid="stExpander"] summary span[data-testid],
/* ANY span/element using Material Icons font anywhere in sidebar */
[data-testid="stSidebar"] span[data-testid="stIconMaterial"],
[data-testid="stSidebar"] .material-icons,
.streamlit-expanderHeader span[class*="arrow"],
.streamlit-expanderHeader svg {
    display: none !important;
    visibility: hidden !important;
    width: 0 !important;
    height: 0 !important;
    max-width: 0 !important;
    max-height: 0 !important;
    overflow: hidden !important;
    font-size: 0 !important;
    color: transparent !important;
    opacity: 0 !important;
    position: absolute !important;
    pointer-events: none !important;
}
[data-testid="stExpander"] summary::-webkit-details-marker,
[data-testid="stExpander"] summary::marker { display: none !important; content: "" !important; }
[data-testid="stExpander"] summary::before,
[data-testid="stExpander"] summary::after  { display: none !important; content: "" !important; }

/* Also hide the text label that appears before the icon in the nav */
[data-testid="stSidebarNavLink"] [data-testid="stIconMaterial"]::before,
[data-testid="stSidebarNavLink"] [data-testid="stIconMaterial"]::after,
/* Kill material icon font rendering as text */
@font-face rules won't help - target the element using the font */
[data-testid="stSidebar"] [style*="material"] {
    font-size: 0 !important;
    color: transparent !important;
}

/* ══════════════════════════════════════════════════════════════
   FIX 2 — PRICE/CHANGE COLORS  (green positive · red negative)
   The global "color: !important" catch-all was overriding inline
   style="color:#00d084" attributes. Fixed by:
   a) Removing !important from the catch-all color rule
   b) Adding explicit high-specificity green/red rules
   ══════════════════════════════════════════════════════════════ */

/* Metric delta colors */
[data-testid="stMetricDeltaIcon-Up"],
[data-testid="stMetricDelta"]:has([data-testid="stMetricDeltaIcon-Up"]),
[data-testid="stMetricDelta"]:has([data-testid="stMetricDeltaIcon-Up"]) * {
    color: #00d084 !important;
}
[data-testid="stMetricDeltaIcon-Down"],
[data-testid="stMetricDelta"]:has([data-testid="stMetricDeltaIcon-Down"]),
[data-testid="stMetricDelta"]:has([data-testid="stMetricDeltaIcon-Down"]) * {
    color: #ff3b3b !important;
}

/* ══════════════════════════════════════════════════════════════
   FIX 3 — FONT SIZE INCREASES across all pages
   ══════════════════════════════════════════════════════════════ */
[data-testid="stMetricValue"] { font-size: 0.92rem !important; font-weight: 700 !important; }
[data-testid="stMetricLabel"] p { font-size: 0.78rem !important; }
[data-testid="stMetricDelta"]   { font-size: 0.82rem !important; }
.stDataFrame thead tr th { font-size: 0.80rem !important; }
.stDataFrame tbody tr td { font-size: 0.90rem !important; }
.stButton > button      { font-size: 0.90rem !important; }


/* ── LEGEND BAR ── */
.bb-header {
    background: var(--bb-amber); color: #000;
    font-family: var(--bb-mono); font-size: 0.7rem; font-weight: 700;
    letter-spacing: 0.15em; text-transform: uppercase;
    padding: 4px 12px; margin-bottom: 8px;
}
</style>
""", unsafe_allow_html=True)


# Terminal header bar
st.markdown("""
<div style="background:#ff8c00;color:#000;font-family:'IBM Plex Mono',monospace;
font-size:0.65rem;font-weight:600;letter-spacing:0.18em;padding:5px 14px;
display:flex;justify-content:space-between;margin-bottom:12px;">
  <span>◼ MONARCH PRO — EQUITY TERMINAL</span>
  <span id="bb-clock">NSE · INDIA</span>
</div>
<script>
(function() {
  function tick() {
    var el = document.getElementById('bb-clock');
    if (el) {
      var now = new Date();
      el.textContent = now.toLocaleTimeString('en-IN',{hour12:false}) + ' IST';
    }
  }
  tick(); setInterval(tick, 1000);
})();
</script>
""", unsafe_allow_html=True)


# ============================================================
# ============================================================
# TOKEN — reads from Home.py session OR secrets OR manual paste
# ============================================================
TOKEN_FILE = ".upstox_token_scanner"

def _resolve_token():
    """
    Priority order for Streamlit Cloud compatibility:
    1. st.session_state.upstox_token  — set by Home.py auto-login (same session)
    2. st.secrets["upstox_token"]     — set once in Streamlit Cloud → Settings → Secrets
    3. TOKEN_FILE on disk             — works locally, not on Cloud
    4. Manual paste in sidebar        — always available as fallback
    """
    # 1. Shared session state from Home.py
    if st.session_state.get("upstox_token",""):
        return st.session_state["upstox_token"]
    # 2. Streamlit secrets (Streamlit Cloud)
    try:
        tok = st.secrets.get("upstox_token","")
        if tok: return tok
    except: pass
    # 3. Local file
    if os.path.exists(TOKEN_FILE):
        try:
            t = open(TOKEN_FILE).read().strip()
            if t: return t
        except: pass
    return ""

if "scanner_token_loaded" not in st.session_state:
    st.session_state.scanner_token = _resolve_token()
    st.session_state.scanner_token_loaded = True

# Sidebar — show status or paste box
st.sidebar.markdown("""
<div style="color:#ff8c00;font-size:.85rem;font-weight:700;letter-spacing:.1em;
padding:6px 0 4px;border-bottom:1px solid #2a2a2a;margin-bottom:8px;">
🔑 UPSTOX CONNECTION
</div>""", unsafe_allow_html=True)

if st.session_state.scanner_token:
    st.sidebar.markdown(f"""
<div style="background:#001a0a;border:1px solid #00d084;border-left:3px solid #00d084;
padding:7px 10px;font-family:'IBM Plex Mono',monospace;margin-bottom:8px;">
  <div style="color:#00d084;font-size:.72rem;font-weight:700;">✔ CONNECTED</div>
  <div style="color:#555;font-size:.62rem;">Token: {st.session_state.scanner_token[:16]}…</div>
  <div style="color:#444;font-size:.58rem;">Login via Home page to refresh</div>
</div>""", unsafe_allow_html=True)
    if st.sidebar.button("↺ Change Token", key="screener_change_tok"):
        st.session_state.scanner_token = ""
        st.session_state.scanner_token_loaded = False
        st.rerun()
else:
    st.sidebar.markdown('<div style="color:#ff3b3b;font-size:.68rem;margin-bottom:6px;">⚠ Not connected — login via Home page or paste token below</div>', unsafe_allow_html=True)
    token_input = st.sidebar.text_input("PASTE ACCESS TOKEN (ONCE DAILY)",
        type="password", key="screener_tok_inp",
        placeholder="eyJ0eXAiOiJKV1Q…")
    if token_input:
        st.session_state.scanner_token = token_input
        st.session_state.upstox_token  = token_input   # share back to session
        try:
            with open(TOKEN_FILE, "w") as f: f.write(token_input)
        except: pass
        st.rerun()

ACCESS_TOKEN = st.session_state.scanner_token

if not ACCESS_TOKEN:
    st.warning("⚠ Connect Upstox on the **Home** page, or paste your token in the sidebar.")
    st.stop()

HEADERS = {
    "Authorization": f"Bearer {ACCESS_TOKEN}",
    "Accept": "application/json"
}
# ============================================================
# SESSION STATE INIT
# ============================================================
if "raw_data_cache" not in st.session_state:
    st.session_state.raw_data_cache = {}
if "error_log" not in st.session_state:
    st.session_state.error_log = []
if "live_quotes_cache" not in st.session_state:
    st.session_state.live_quotes_cache = {}
if "targets" not in st.session_state:
    st.session_state.targets = {}
if "score_cache" not in st.session_state:
    st.session_state.score_cache = {}          # PERF: avoid double score_stock_dual calls
if "score_cache_ts" not in st.session_state:
    st.session_state.score_cache_ts = 0        # timestamp of last score recompute
if "live_tables" not in st.session_state:
    st.session_state.live_tables = {
        "leader": None, "rs": None, "trigger": None,
        "transition": None, "exit": None
    }

SCORE_CACHE_TTL = 60   # recompute scores at most once per minute

# ============================================================
# CORE UTILITIES
# ============================================================

def to_ascending(df: pd.DataFrame) -> pd.DataFrame:
    return df.iloc[::-1].reset_index(drop=True)


def rsi_wilder(close: pd.Series, period: int):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50)


def percentile_last(series: pd.Series, window: int):
    s = series.tail(window).dropna()
    if len(s) == 0:
        return np.nan
    lo, hi = s.min(), s.max()
    if hi == lo:
        return np.nan
    return (s.iloc[-1] - lo) / (hi - lo)


def darvas_probability(df: pd.DataFrame, window: int):
    if len(df) < window * 2:
        return np.nan
    hist   = df.iloc[-(window*2):-window]
    recent = df.iloc[-window:]
    hist_range   = hist['high'] - hist['low']
    recent_range = recent['high'] - recent['low']
    compression  = 1 - (recent_range.mean() / (hist_range.mean() + 1e-9))
    box_high     = hist['high'].max()
    box_low      = hist['low'].min()
    inside_box   = ((recent['high'] <= box_high) & (recent['low'] >= box_low)).mean()
    last_close   = df['close'].iloc[-1]
    pressure     = (last_close - box_low) / (box_high - box_low + 1e-9)
    return max(0, compression) * inside_box * max(0, pressure)


def normalize_key(k: str) -> str:
    """Normalise to pipe format: NSE_EQ|INE002A01018"""
    return k.replace("%7C", "|").replace(":", "|")

# ============================================================
# ★ LIVE QUOTE FETCH — Single endpoint: /v2/market-quote/quotes
#
# Response keys come back as  "NSE_EQ:SYMBOL"  (colon + trading symbol).
# Each record also contains   "instrument_token": "NSE_EQ|INE..."
# We store results keyed by the instrument_token (pipe format) so that
# lookup via targets[sym] (which is also pipe format) always matches.
#
# Fields returned: last_price, volume, oi, last_trade_time
#                  ohlc → open/high/low/close  (today's intraday)
# ============================================================

def fetch_live_quotes(all_keys: list) -> dict:
    """
    Returns dict keyed by instrument_key (pipe format, e.g. NSE_EQ|INE...):
      ltp, open, high, low, volume, oi, last_trade_time
    All values are today's live intraday data.
    """
    url      = "https://api.upstox.com/v2/market-quote/quotes"
    live_map = {}
    CHUNK    = 50   # safe batch size for full-quote endpoint

    for i in range(0, len(all_keys), CHUNK):
        batch  = all_keys[i:i + CHUNK]
        params = {"instrument_key": ",".join(batch)}
        try:
            r = requests.get(url, headers=HEADERS, params=params, timeout=15)
            if r.status_code != 200:
                st.warning(f"Live quote HTTP {r.status_code} batch {i//CHUNK+1}: {r.text[:200]}")
                continue

            data = r.json().get("data", {})

            for _resp_key, v in data.items():
                # ── KEY: use instrument_token from body (pipe format), not resp key ──
                ikey = v.get("instrument_token", "")
                if not ikey:
                    continue
                nk = normalize_key(ikey)   # ensures NSE_EQ|INE... format

                ltp  = v.get("last_price")
                if ltp is None:
                    continue

                ohlc = v.get("ohlc", {})   # today's intraday OHLC

                live_map[nk] = {
                    "ltp":             float(ltp),
                    "open":            float(ohlc.get("open",  ltp)),
                    "high":            float(ohlc.get("high",  ltp)),
                    "low":             float(ohlc.get("low",   ltp)),
                    # close in ohlc == last_price, redundant but kept for clarity
                    "volume":          float(v["volume"])          if v.get("volume")          else None,
                    "oi":              float(v["oi"])               if v.get("oi")               else None,
                    "last_trade_time": v.get("last_trade_time"),
                }

        except Exception as e:
            st.warning(f"Live fetch error batch {i//CHUNK+1}: {e}")

        time.sleep(0.12)

    if not live_map:
        st.warning("⚠️ Live quote returned empty — check token or market hours")

    return live_map


def patch_live_bar(df: pd.DataFrame, live: dict) -> pd.DataFrame:
    """
    Overwrites the LAST daily bar's close/high/low/volume/oi with today's
    real intraday values fetched from the full market-quote endpoint.
    """
    if not live:
        return df

    df = df.copy()
    last_idx = df.index[-1]

    ltp    = live.get("ltp")
    high   = live.get("high")
    low    = live.get("low")
    volume = live.get("volume")
    oi     = live.get("oi")

    if ltp is not None:
        df.at[last_idx, "close"] = ltp
    if high is not None:
        df.at[last_idx, "high"] = max(df.at[last_idx, "high"], high)
    if low is not None:
        df.at[last_idx, "low"] = min(df.at[last_idx, "low"], low)
    if volume is not None:
        df.at[last_idx, "volume"] = volume
    if oi is not None and "oi" in df.columns:
        df.at[last_idx, "oi"] = oi

    return df


def get_live_bar(sym: str) -> dict:
    """Returns the latest live quote dict for a symbol (or empty dict)."""
    key   = st.session_state.targets.get(sym)
    if not key:
        return {}
    cache = st.session_state.live_quotes_cache
    return cache.get(normalize_key(key), {})


def apply_live_patch(df: pd.DataFrame, sym: str) -> pd.DataFrame:
    """Applies the live patch to any df passed in (used at render time)."""
    live = get_live_bar(sym)
    if not live:
        return df
    return patch_live_bar(df, live)


# ============================================================
# LIVE REFRESH ENGINE (background re-patch, every 60s)
# ============================================================
LIVE_REFRESH_SEC = 60   # refreshed much faster than original 200s

def refresh_live_prices():
    if not st.session_state.raw_data_cache:
        return
    if not st.session_state.targets:
        return

    now  = time.time()
    last = st.session_state.get("_last_live_refresh", 0)
    if now - last < LIVE_REFRESH_SEC:
        return

    keys = list(st.session_state.targets.values())
    live = fetch_live_quotes(keys)
    if not live:
        return

    # Patch stored dataframes in-place
    for sym, df in st.session_state.raw_data_cache.items():
        key    = st.session_state.targets.get(sym)
        if not key:
            continue
        live_q = live.get(normalize_key(key))
        if not live_q:
            continue
        st.session_state.raw_data_cache[sym] = patch_live_bar(df, live_q)

    st.session_state.live_quotes_cache    = live
    st.session_state._last_live_refresh   = now

# ============================================================
# MASTER INSTRUMENT LIST
# ============================================================
@st.cache_data(ttl=3600)
def get_live_master():
    try:
        url = "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz"
        r   = requests.get(url, timeout=10)
        if r.status_code == 200:
            with gzip.GzipFile(fileobj=io.BytesIO(r.content)) as gz:
                data = json.load(gz)
            return pd.DataFrame(data)
    except Exception as e:
        st.error(f"Failed to load Master List: {e}")
    return pd.DataFrame()


def build_universes(df):
    eq = df[(df['exchange'] == 'NSE') & (df['instrument_type'] == 'EQ')]
    fo_contracts   = df[df['segment'].astype(str).str.contains("FO", na=False)]
    fo_underlyings = fo_contracts['underlying_symbol'].dropna().astype(str).unique()
    fno = eq[eq['trading_symbol'].isin(fo_underlyings)]

    nifty50 = [
        "RELIANCE","HDFCBANK","ICICIBANK","INFY","ITC","TCS","LT","SBIN","AXISBANK",
        "KOTAKBANK","BHARTIARTL","ASIANPAINT","HCLTECH","MARUTI","SUNPHARMA","ULTRACEMCO",
        "TITAN","WIPRO","NESTLEIND","POWERGRID","NTPC","BAJFINANCE","BAJAJFINSV",
        "INDUSINDBK","TECHM","M&M","TATAMOTORS","ADANIENT","ADANIPORTS","ONGC",
        "COALINDIA","JSWSTEEL","HINDALCO","TATASTEEL","BPCL","GRASIM","CIPLA",
        "DRREDDY","EICHERMOT","HEROMOTOCO","BRITANNIA","DIVISLAB","SBILIFE",
        "HDFCLIFE","APOLLOHOSP","BAJAJ-AUTO","UPL","SHREECEM","HINDUNILVR"
    ]
    nifty = eq[eq['trading_symbol'].isin(nifty50)]
    return nifty, fno, eq

# ============================================================
# SIDEBAR CONFIG
# ============================================================
DAILY_PROFILES = {
    "Short Term":  {"ma_f": 5,  "ma_s": 20,  "rsi_p": 7},
    "Medium Term": {"ma_f": 20, "ma_s": 50,  "rsi_p": 14},
    "Long Term":   {"ma_f": 50, "ma_s": 200, "rsi_p": 21},
}

with st.sidebar:
    st.title("🛡️ Monarch Config")
    universe_choice = st.radio("Universe", ["Nifty 50", "F&O Stocks", "Full NSE"])
    selected_p  = st.selectbox("Profile", list(DAILY_PROFILES.keys()))
    conf        = DAILY_PROFILES[selected_p]
    ma_f_val    = st.number_input("Fast MA",  1, 200, conf['ma_f'])
    ma_s_val    = st.number_input("Slow MA",  1, 500, conf['ma_s'])
    rsi_p_val   = st.number_input("RSI Period", 1, 50, conf['rsi_p'])

    st.divider()
    st.caption("Live data source: Upstox /market-quote/quotes")
    if st.session_state.live_quotes_cache:
        n_live = len(st.session_state.live_quotes_cache)
        last_t = st.session_state.get("_last_live_refresh", 0)
        age    = int(time.time() - last_t) if last_t else "—"
        st.success(f"✅ {n_live} live quotes | refreshed {age}s ago")
    else:
        st.info("No live data yet — run extraction")

# ============================================================
# MAIN TITLE + UNIVERSE BUILD
# ============================================================
st.title("📊 Monarch Pro v2 — Real-Time Engine")

master_df = get_live_master()

if master_df.empty:
    st.error("Could not load instrument master — check network")
    st.stop()

nifty_df, fno_df, full_df = build_universes(master_df)

if universe_choice == "Nifty 50":
    selected_df = nifty_df
elif universe_choice == "F&O Stocks":
    selected_df = fno_df
else:
    selected_df = full_df

targets = {row['trading_symbol']: row['instrument_key']
           for _, row in selected_df.iterrows()}

st.write(f"**Universe:** {universe_choice} — **{len(targets)} stocks**")

# ============================================================
# BULK EXTRACTION BUTTON
# ============================================================
if st.button("🚀 Start Bulk Extraction", use_container_width=True):
    st.session_state.raw_data_cache = {}
    st.session_state.targets = targets

    end_date   = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=600)).strftime('%Y-%m-%d')

    # ── Fetch ALL live quotes ONCE before the loop ──
    with st.spinner("Fetching live market snapshot…"):
        live_quotes = fetch_live_quotes(list(targets.values()))
        st.session_state.live_quotes_cache    = live_quotes
        st.session_state._last_live_refresh   = time.time()

    st.success(f"Live snapshot: {len(live_quotes)} instruments")

    # DEBUG: show first 5 live quote entries so key format is visible
    with st.expander("\U0001f50d Debug — Live Quote Sample (first 5 keys)", expanded=False):
        sample = dict(list(live_quotes.items())[:5])
        st.json(sample)
        st.caption("Keys should look like: NSE_EQ|INE... — if different, key mismatch is the problem")

    # ── CONCURRENT HISTORICAL FETCH (replaces sequential sleep loop) ──
    FETCH_WORKERS = 8   # safe concurrency for Upstox API

    def _fetch_one(sym_key_pair):
        sym, key = sym_key_pair
        url = (
            f"https://api.upstox.com/v2/historical-candle/"
            f"{urllib.parse.quote(key)}/day/{end_date}/{start_date}"
        )
        try:
            r = requests.get(url, headers=HEADERS, timeout=12)
            if r.status_code != 200:
                return sym, None, f"HTTP {r.status_code}"
            raw = r.json().get("data", {}).get("candles", [])
            if not raw:
                return sym, None, "empty"
            df = pd.DataFrame(raw, columns=["time","open","high","low","close","volume","oi"])
            df = to_ascending(df)
            # patch live bar
            live_q = live_quotes.get(normalize_key(key))
            if live_q:
                df = patch_live_bar(df, live_q)
            df["__live__"] = True
            return sym, df, None
        except Exception as e:
            return sym, None, str(e)

    progress   = st.progress(0)
    results    = []
    sym_keys   = list(targets.items())
    completed  = 0

    with ThreadPoolExecutor(max_workers=FETCH_WORKERS) as executor:
        futures = {executor.submit(_fetch_one, pair): pair for pair in sym_keys}
        for future in as_completed(futures):
            sym, df, err = future.result()
            completed += 1
            progress.progress(completed / len(sym_keys))

            if err or df is None:
                if err and err not in ("empty",):
                    st.session_state.error_log.append(f"{sym}: {err}")
                continue

            key = targets[sym]
            live_q = live_quotes.get(normalize_key(key))
            st.session_state.raw_data_cache[sym] = df.copy()

            if len(df) < ma_s_val:
                continue

            try:
                df['ma_f'] = df['close'].rolling(ma_f_val).mean()
                df['ma_s'] = df['close'].rolling(ma_s_val).mean()
                df['rsi']  = rsi_wilder(df['close'], rsi_p_val)

                signal_window = min(40, len(df)//3)
                regime_window = min(250, len(df)//2)

                spread = df['ma_f'] - df['ma_s']
                rng    = df['high'] - df['low']

                trend        = percentile_last(spread, regime_window)
                momentum     = percentile_last(df['rsi'], signal_window)
                participation= percentile_last(df['volume'], signal_window)
                location     = percentile_last(df['close'], signal_window)
                darvas       = darvas_probability(df, ma_s_val)

                trend_chg         = percentile_last(spread.diff(), regime_window)
                momentum_chg      = percentile_last(df['rsi'].diff(), signal_window)
                participation_chg = percentile_last(df['volume'].diff(), signal_window)
                location_chg      = percentile_last(df['close'].diff(), signal_window)
                expansion_chg     = percentile_last(rng.diff(), signal_window)

                # FIX B-02: vectorised Darvas structure series
                _w = ma_s_val
                _hh = df['high'].rolling(_w).max()
                _ll = df['low'].rolling(_w).min()
                _hist_rng = (_hh.shift(_w) - _ll.shift(_w)).fillna(0)
                _rec_rng  = (df['high'].rolling(max(1, _w // 2)).max() - df['low'].rolling(max(1, _w // 2)).min())
                _compression = (1 - (_rec_rng / (_hist_rng + 1e-9))).clip(0, 1)
                _pressure = ((df['close'] - _ll) / (_hh - _ll + 1e-9)).clip(0, 1)
                _inside = (
                    (df['high'] <= _hh.shift(1)) &
                    (df['low']  >= _ll.shift(1))
                ).rolling(_w).mean().fillna(0)
                structure_series = (_compression * _inside * _pressure).fillna(0)
                structure_chg = percentile_last(structure_series.diff(), regime_window)

                # FIX I-01: Remove missing-factor bias.
                # A stock with only 2/6 signals can't rank as high as one with all 6.
                # score = signal_strength × coverage — penalises sparse signal sets.
                _all_comps = np.array([trend_chg, momentum_chg, participation_chg,
                                       location_chg, expansion_chg, structure_chg],
                                      dtype=float)
                _valid      = np.sum(~np.isnan(_all_comps))
                _coverage   = _valid / len(_all_comps)          # 0.0 – 1.0
                _strength   = float(np.nanmean(_all_comps)) if _valid > 0 else 0.0
                score       = float(_strength * _coverage * 100) if _valid > 0 else 0

                latest = df.iloc[-1]

                def label(v, high, mid):
                    if pd.isna(v): return "NA"
                    if v > high:   return "High"
                    if v > mid:    return "Medium"
                    return "Low"

                ltt = live_q.get("last_trade_time", "—") if live_q else "—"

                results.append({
                    "Ticker":         sym,
                    "Score":          round(score, 2),
                    "LTP":            round(float(latest['close']), 2),
                    "DayHigh":        round(float(live_q['high']), 2)   if live_q else None,
                    "DayLow":         round(float(live_q['low']),  2)   if live_q else None,
                    "LiveVolume":     int(live_q['volume'])              if live_q and live_q['volume'] else None,
                    "LiveOI":         int(live_q['oi'])                  if live_q and live_q['oi'] else None,
                    "LastTradeTime":  ltt,
                    "RSI":            round(float(latest['rsi']), 1),
                    "OI":             int(latest['oi']),
                    "Trend%":         round(trend*100, 1)        if pd.notna(trend) else np.nan,
                    "Participation%": round(participation*100,1) if pd.notna(participation) else np.nan,
                    "Location%":      round(location*100, 1)     if pd.notna(location) else np.nan,
                    "Structure%":     round(darvas*100, 1)       if pd.notna(darvas) else np.nan,
                    "Trend":          label(trend, 0.66, 0.33),
                    "Participation":  label(participation, 0.66, 0.33),
                    "Location":       label(location, 0.66, 0.33),
                    "Structure":      label(darvas, 0.66, 0.33),
                })
            except Exception as e:
                st.session_state.error_log.append(f"{sym} (compute): {e}")

    st.success(f"✅ Downloaded {len(st.session_state.raw_data_cache)} symbols | "
               f"{len(live_quotes)} live quotes patched")

    if results:
        result_df = pd.DataFrame(results).sort_values("Score", ascending=False).reset_index(drop=True)
        result_df.insert(0, "#", result_df.index + 1)

        # ── Bloomberg-style extraction results table ──
        st.markdown("""
<div style="background:#0a0a0a;border:1px solid #ff8c00;border-top:3px solid #ff8c00;
     margin-top:12px;font-family:'IBM Plex Mono',monospace;">
  <div style="background:#1a1200;padding:6px 14px;border-bottom:1px solid #2a2a2a;
              display:flex;justify-content:space-between;align-items:center;">
    <span style="color:#ff8c00;font-size:.72rem;font-weight:700;letter-spacing:.12em;">
      ◼ EXTRACTION RESULTS
    </span>
    <span style="color:#555;font-size:.62rem;">SORTED BY SCORE  ·  LIVE DATA PATCHED</span>
  </div>
</div>""", unsafe_allow_html=True)

        def _bb_extr(df):
            disp_cols = ["#","Ticker","LTP","DayHigh","DayLow","LiveVolume","RSI",
                         "Trend%","Participation%","Location%","Structure%",
                         "Trend","Participation","Location","Structure"]
            disp = df[[c for c in disp_cols if c in df.columns]]
            def score_col(v):
                if not isinstance(v,(int,float)): return ""
                if v >= 60: return "background-color:#1a3300;color:#00d084;font-weight:700"
                if v >= 40: return "background-color:#1a2200;color:#b8e06a"
                if v >= 20: return "background-color:#2a1800;color:#ffb347"
                return "color:#555"
            def rsi_col(v):
                if not isinstance(v,(int,float)): return ""
                if v >= 70: return "color:#ff3b3b;font-weight:700"
                if v >= 55: return "color:#ffb347"
                if v <= 30: return "color:#1e90ff;font-weight:700"
                return "color:#00d084"
            def label_col(v):
                if v == "High":   return "color:#00d084;font-weight:700"
                if v == "Medium": return "color:#ffb347"
                return "color:#555"
            sty = disp.style
            for pct_col in ["Trend%","Participation%","Location%","Structure%"]:
                if pct_col in disp.columns:
                    sty = sty.applymap(score_col, subset=[pct_col])
            if "RSI" in disp.columns:
                sty = sty.applymap(rsi_col, subset=["RSI"])
            for lbl in ["Trend","Participation","Location","Structure"]:
                if lbl in disp.columns:
                    sty = sty.applymap(label_col, subset=[lbl])
            return sty

        st.dataframe(_bb_extr(result_df), use_container_width=True, hide_index=True)

# ============================================================
# MONARCH DUAL-SETUP ENGINE — 1-5 DAY EDGE MODEL
# ============================================================
# Two setup types scored independently:
#
# [B] BREAKOUT — stock consolidating, volume exploding,
#     price breaking above range with a strong candle.
#     Best entered on the day of breakout.
#
# [P] PULLBACK — stock in confirmed uptrend, pulled back
#     to EMA support, reversal candle appearing, volume
#     turning up. Best entered 1-3 bars after pullback low.
#
# SIGNAL FACTORS (no redundancy):
#   1. Relative Strength vs Nifty (5d + 20d)  — is this stock
#      outperforming the index? Best single predictor of continuation.
#   2. Volume Surge                            — your #1 trusted signal
#   3. Consolidation Tightness (box quality)  — how well-formed is the base
#   4. MA Structure (EMA9/20/50)              — trend alignment
#   5. ATR-normalised Price Move              — magnitude vs volatility
#   6. Candlestick Pattern Score              — 8 patterns detected
#   7. RSI(7) Quality Filter                  — overbought guard only
# ============================================================

refresh_live_prices()
st.divider()
st.header("🎯 Monarch Dual-Setup Engine — 1-5 Day Edge")

# ── MARKET CONTEXT — Nifty trend, VIX, Sector returns ─────────
SECTOR_TICKERS = {
    "IT":"^CNXIT","Bank":"^NSEBANK","Auto":"^CNXAUTO",
    "Pharma":"^CNXPHARMA","Metal":"^CNXMETAL","Energy":"^CNXENERGY",
    "Infra":"^CNXINFRA","FMCG":"^CNXFMCG","Realty":"^CNXREALTY",
    "PSUBank":"^CNXPSUBANK",
}

STOCK_SECTOR_MAP = {
    "TCS":"IT","INFY":"IT","WIPRO":"IT","HCLTECH":"IT","TECHM":"IT",
    "LTIM":"IT","MPHASIS":"IT","COFORGE":"IT","PERSISTENT":"IT","OFSS":"IT",
    "HDFCBANK":"Bank","ICICIBANK":"Bank","KOTAKBANK":"Bank","AXISBANK":"Bank",
    "INDUSINDBK":"Bank","FEDERALBNK":"Bank","IDFCFIRSTB":"Bank","AUBANK":"Bank",
    "BAJFINANCE":"Bank","BAJAJFINSV":"Bank",
    "SBIN":"PSUBank","BANKBARODA":"PSUBank","PNB":"PSUBank","CANBK":"PSUBank",
    "MARUTI":"Auto","TATAMOTORS":"Auto","M&M":"Auto","BAJAJ-AUTO":"Auto",
    "HEROMOTOCO":"Auto","EICHERMOT":"Auto","TVSMOTORS":"Auto","TITAN":"Auto",
    "SUNPHARMA":"Pharma","DRREDDY":"Pharma","CIPLA":"Pharma","DIVISLAB":"Pharma",
    "TORNTPHARM":"Pharma","AUROPHARMA":"Pharma","APOLLOHOSP":"Pharma",
    "TATASTEEL":"Metal","JSWSTEEL":"Metal","HINDALCO":"Metal","SAIL":"Metal",
    "VEDL":"Metal","COALINDIA":"Metal","NMDC":"Metal",
    "ONGC":"Energy","NTPC":"Energy","POWERGRID":"Energy","BPCL":"Energy",
    "IOC":"Energy","GAIL":"Energy","RELIANCE":"Energy",
    "LT":"Infra","ADANIPORTS":"Infra","IRFC":"Infra","RVNL":"Infra",
    "IRCON":"Infra","NBCC":"Infra","ULTRACEMCO":"Infra","SHREECEM":"Infra",
    "HINDUNILVR":"FMCG","ITC":"FMCG","NESTLEIND":"FMCG","BRITANNIA":"FMCG",
    "DABUR":"FMCG","MARICO":"FMCG","GODREJCP":"FMCG","ASIANPAINT":"FMCG",
    "DLF":"Realty","LODHA":"Realty","OBEROIRLTY":"Realty","PHOENIXLTD":"Realty",
}

@st.cache_data(ttl=900)   # FIX I-14: 15-min cache (was 5-min) — market data rarely changes intra-session
def get_market_context():
    out = dict(nifty_r5=None, nifty_r20=None,
               nifty_above_20dma=True, vix_level=None, vix_falling=True,
               sector_returns={}, sector_returns_10d={},   # FIX B-01: two timeframes
               top_sectors=set(),
               market_ok=True, market_notes=[])
    try:
        n = yf.download("^NSEI", period="90d", interval="1d", progress=False)
        if not n.empty:
            c = n["Close"].squeeze()
            out["nifty_r5"]  = float(c.iloc[-1]/c.iloc[-6]-1)  if len(c)>=6  else None
            out["nifty_r20"] = float(c.iloc[-1]/c.iloc[-21]-1) if len(c)>=21 else None
            dma20 = float(c.tail(20).mean())
            out["nifty_above_20dma"] = float(c.iloc[-1]) > dma20
            if not out["nifty_above_20dma"]:
                out["market_notes"].append("Nifty below 20DMA — breakout risk elevated")
    except Exception:
        pass
    try:
        v = yf.download("^INDIAVIX", period="10d", interval="1d", progress=False)
        if not v.empty:
            vc = v["Close"].squeeze()
            out["vix_level"]  = round(float(vc.iloc[-1]), 2)
            out["vix_falling"] = float(vc.iloc[-1]) <= float(vc.iloc[-3]) if len(vc)>=3 else True
            if not out["vix_falling"]:
                out["market_notes"].append(f"VIX rising ({out['vix_level']}) — breakouts may reverse fast")
    except Exception:
        pass
    # FIX B-01: fetch 5d AND 10d sector returns separately so sect_accel is real
    sr_5d  = {}
    sr_10d = {}
    for name, ticker in SECTOR_TICKERS.items():
        try:
            s = yf.download(ticker, period="60d", interval="1d", progress=False)
            if not s.empty:
                sc = s["Close"].squeeze()
                if len(sc) >= 6:
                    sr_5d[name]  = float(sc.iloc[-1] / sc.iloc[-6]  - 1)
                if len(sc) >= 11:
                    sr_10d[name] = float(sc.iloc[-1] / sc.iloc[-11] - 1)
        except Exception:
            pass
    out["sector_returns"]     = sr_5d
    out["sector_returns_10d"] = sr_10d
    if sr_5d:
        out["top_sectors"] = {k for k,_ in sorted(sr_5d.items(), key=lambda x:x[1], reverse=True)[:3]}
    out["market_ok"] = out["nifty_above_20dma"]
    return out

mkt              = get_market_context()
nifty_r5         = mkt["nifty_r5"]
nifty_r20        = mkt["nifty_r20"]
sector_returns   = mkt["sector_returns"]
sector_returns_10d = mkt["sector_returns_10d"]   # FIX B-01
top_sectors      = mkt["top_sectors"]

def get_sector(ticker):
    return STOCK_SECTOR_MAP.get(ticker.upper(), None)

def get_sector_return(ticker):
    """Returns (sect_r5d, sect_r10d, sect_name) for the given ticker."""
    sect = get_sector(ticker)
    if sect:
        r5  = sector_returns.get(sect)
        r10 = sector_returns_10d.get(sect)
        if r5 is not None:
            return r5, r10, sect
    return None, None, None

# ── INDICATOR HELPERS ──

def ema(series, n):
    return series.ewm(span=n, adjust=False).mean()

def atr14(df):
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift(1)).abs()
    lc = (df["low"]  - df["close"].shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(span=14, adjust=False).mean()

def detect_candle_patterns(o, h, l, c, prev_o, prev_h, prev_l, prev_c):
    """
    Scores the current candle for 8 bullish patterns.
    Returns (score 0-10, list of pattern names found).
    """
    body     = abs(c - o)
    rng      = h - l + 1e-9
    upper_w  = h - max(o, c)
    lower_w  = min(o, c) - l
    prev_body= abs(prev_c - prev_o)
    prev_rng = prev_h - prev_l + 1e-9   # FIX B-03: need prev candle range for MorningStar
    patterns = []
    pts      = 0

    # 1. Bullish Engulfing — strongest reversal signal
    if prev_c < prev_o and c > o and c > prev_o and o < prev_c:
        patterns.append("Engulfing"); pts += 3

    # 2. Hammer / Pin Bar — rejection of lows
    # FIX B-03 (minor): relax upper_w threshold to 0.4×rng for large-body candles
    if lower_w >= 2 * body and upper_w <= 0.4 * rng and c > o:
        patterns.append("Hammer"); pts += 2.5

    # 3. Inside Bar — compression before expansion
    if h <= prev_h and l >= prev_l:
        patterns.append("InsideBar"); pts += 1.5

    # 4. Outside Bar (bullish) — strong momentum candle
    if h > prev_h and l < prev_l and c > o and c > (h + l) / 2:
        patterns.append("OutsideBar"); pts += 2

    # 5. Strong Green Candle — body > 60% of range, closes high
    if body / rng > 0.60 and c > o and (c - l) / rng > 0.75:
        patterns.append("StrongGreen"); pts += 2

    # 6. Doji at support (tiny body, equal wicks) — indecision turning bullish
    if body / rng < 0.10 and lower_w > 1.5 * upper_w:
        patterns.append("BullDoji"); pts += 1

    # 7. Morning Star equivalent: prev big RED candle, current green closing > prev midpoint
    # FIX B-03: use prev_rng (not today's rng) to judge if prev candle was "big red"
    if prev_c < prev_o and prev_body / prev_rng > 0.5 and c > o and c > (prev_o + prev_c) / 2:
        patterns.append("MorningStar"); pts += 2.5

    # 8. Gap-up open with continuation (open > prev close, close even higher)
    if o > prev_c * 1.003 and c > o:
        patterns.append("GapContinue"); pts += 2

    return min(pts, 10), patterns

def consolidation_score(df, window=15):
    """
    Measures how tight/clean the consolidation base is.
    Returns 0-1. Higher = tighter base = better breakout setup.
    """
    if len(df) < window * 2:
        return 0.0
    base  = df.iloc[-window:]
    prior = df.iloc[-(window*2):-window]
    base_rng  = (base["high"].max()  - base["low"].min())
    prior_rng = (prior["high"].max() - prior["low"].min())
    if prior_rng == 0:
        return 0.0
    compression = 1.0 - (base_rng / (prior_rng + 1e-9))
    flat_hi  = base["high"].std()  / (base["high"].mean()  + 1e-9)
    flat_lo  = base["low"].std()   / (base["low"].mean()   + 1e-9)
    flatness = 1.0 - min(1.0, (flat_hi + flat_lo) * 50)
    return max(0.0, min(1.0, (compression * 0.6 + flatness * 0.4)))

def volume_surge(v_today, v_series, window=20):
    """Returns ratio of today's volume to N-day average."""
    avg = v_series.tail(window).mean()
    if avg == 0:
        return 0.0
    return float(v_today) / float(avg)

def relative_strength(c_series, nifty_r5, nifty_r20, window5=5, window20=20):
    """
    RS score 0-1.
    FIX I-02: Volatility-normalised alpha replaces the fixed ±2% band.
    rs_alpha = (stock_return - index_return) / stock_return_std
    This is superior because a +2% beat on a 10%-vol stock is weak signal;
    the same +2% beat on a 1%-vol stock is a massive 2-std outperformance.
    Percentile-ranked over the stock's own 60d history makes it fully adaptive.
    """
    if len(c_series) < 22:
        return 0.5
    base_6  = float(c_series.iloc[-6])  if len(c_series) >= 6  else 0
    base_21 = float(c_series.iloc[-21]) if len(c_series) >= 21 else 0
    stock_r5  = float(c_series.iloc[-1] / base_6  - 1) if base_6  != 0 else 0
    stock_r20 = float(c_series.iloc[-1] / base_21 - 1) if base_21 != 0 else 0
    r5_beat   = stock_r5  - (nifty_r5  or 0)
    r20_beat  = stock_r20 - (nifty_r20 or 0)

    # Volatility normalisation — use stock's own 20d return std as scaling unit
    daily_rets = c_series.pct_change().dropna()
    ret_std_5  = float(daily_rets.tail(5).std())  if len(daily_rets) >= 5  else 0.01
    ret_std_20 = float(daily_rets.tail(20).std()) if len(daily_rets) >= 20 else 0.01
    ret_std_5  = max(ret_std_5,  0.001)   # floor: prevents div-by-zero on zero-vol
    ret_std_20 = max(ret_std_20, 0.001)

    # alpha in units of stock volatility (z-score of outperformance)
    alpha_5  = r5_beat  / (ret_std_5  * np.sqrt(5)  + 1e-9)
    alpha_20 = r20_beat / (ret_std_20 * np.sqrt(20) + 1e-9)

    # Percentile-rank each alpha vs a ±3 std range → 0-1 score
    # tanh squashing: output 0.5 at alpha=0, →1 as alpha→+∞, →0 as alpha→-∞
    rs5  = float(0.5 * (1.0 + np.tanh(alpha_5  / 1.5)))
    rs20 = float(0.5 * (1.0 + np.tanh(alpha_20 / 1.5)))

    return rs5 * 0.6 + rs20 * 0.4   # 5d weighted more for short-term momentum


# ── MAIN SCORING FUNCTION v3 ─────────────────────────────────
# 10 factors, 100 pts. Adds vs v2:
#   RS_Sector, InstVol footprint, VolContraction ATR5/ATR20,
#   flat-resistance in coil, ATR% potential, gap filter,
#   time horizon, entry/target/stop, EMI ranking metric.

def score_stock_dual(df_raw, live, nifty_r5, nifty_r20, ticker=""):
    if len(df_raw) < 60:
        return None

    df = df_raw.copy()
    ltp     = live.get("ltp")    or float(df["close"].iloc[-1])
    day_vol = live.get("volume") or float(df["volume"].iloc[-1])
    day_hi  = live.get("high")   or float(df["high"].iloc[-1])
    day_lo  = live.get("low")    or float(df["low"].iloc[-1])
    day_o   = live.get("open")   or float(df["open"].iloc[-1])

    df.at[df.index[-1], "close"]  = ltp
    df.at[df.index[-1], "high"]   = max(float(df["high"].iloc[-1]),  day_hi)
    df.at[df.index[-1], "low"]    = min(float(df["low"].iloc[-1]),   day_lo)
    df.at[df.index[-1], "volume"] = day_vol

    c = df["close"]; h = df["high"]; l = df["low"]
    v = df["volume"]; o = df["open"]

    # Historical slice — BEFORE today (prevents look-ahead)
    hist = df.iloc[:-1]
    hc = hist["close"]; hh = hist["high"]; hl = hist["low"]; hv = hist["volume"]

    e9   = ema(c, 9);  e20 = ema(c, 20);  e50 = ema(c, 50)
    e5   = ema(c, 5)   # FIX I-05: needed for momentum acceleration
    atr  = atr14(df)
    rsi  = rsi_wilder(c, 7)

    e9_v  = float(e9.iloc[-1]);  e20_v = float(e20.iloc[-1])
    e50_v = float(e50.iloc[-1]); atr_v = float(atr.iloc[-1])
    rsi_v = float(rsi.iloc[-1]); rsi_p = float(rsi.iloc[-2])
    e9_y  = float(e9.iloc[-2]);  e20_y = float(e20.iloc[-2])

    vol_ma20 = float(v.rolling(20).mean().iloc[-1])
    vol_ma5  = float(v.tail(5).mean())
    atr_pct  = (atr_v / ltp) * 100 if ltp > 0 else 0

    # ── HARD FILTERS ──
    if atr_v == 0 or vol_ma20 == 0:           return None
    if rsi_v > 85:                             return None
    if day_vol < vol_ma20 * 0.15:             return None
    if atr_pct < 0.8:                          return None
    # Illiquid / SME filter: avg volume < 100k shares = skip (too thin to trade)
    if vol_ma20 < 100_000:                     return None
    prev_close = float(hc.iloc[-1])
    if prev_close > 0 and abs(day_o - prev_close) / prev_close > 0.06:
        return None   # gap > 6%

    # ── BASE RANGE (excl today) ──
    base_hi  = float(hh.tail(20).max())
    base_lo  = float(hl.tail(20).min())
    base_rng = base_hi - base_lo + 1e-9
    breakout_ext = (ltp - base_hi) / (atr_v + 1e-9)

    # ── SETUP CLASSIFICATION ──
    above_ema50  = ltp > e50_v * 0.97
    if not above_ema50:
        return None
    near_e9_y  = abs(float(hc.iloc[-1]) - e9_y)  / (atr_v + 1e-9) < 1.0
    near_e20_y = abs(float(hc.iloc[-1]) - e20_y) / (atr_v + 1e-9) < 1.2

    # Improved: accumulation volume > avg already confirms demand, threshold lowered to 1.0×
    if breakout_ext >= -0.3 and breakout_ext <= 1.5 and day_vol > vol_ma20 * 1.0:
        setup_type = "Breakout"
    elif above_ema50 and (near_e9_y or near_e20_y):
        setup_type = "Pullback"
    elif breakout_ext > 1.5:
        return None
    else:
        setup_type = "Pullback"

    # ═══════════════════════════════════════════════════════
    # F1 — RS vs NIFTY  (0-15 pts)
    # ═══════════════════════════════════════════════════════
    rs_score = relative_strength(hc, nifty_r5, nifty_r20)
    rs_pts   = round(rs_score * 15, 1)

    # ── RS ACCELERATION (momentum strengthening signal) ──
    # FIX I-05: Replace binary rs_accel_bonus with continuous percentile-ranked
    # EMA velocity acceleration. velocity = EMA5 - EMA20 (price momentum speed).
    # acceleration = velocity.diff() — detects early ignition of trend.
    # This is a true leading signal: catches acceleration BEFORE RS confirms.
    _velocity     = e5 - e20
    _acceleration = _velocity.diff()
    _acc_hist     = _acceleration.iloc[:-1]   # no look-ahead
    _acc_pct      = percentile_last(_acc_hist, min(200, len(_acc_hist)))
    acc_score     = float(_acc_pct) if pd.notna(_acc_pct) else 0.5
    rs_accel_bonus = round(acc_score * 4, 1)  # 0-4 pts (was binary 0/1/3)

    # Also retain raw accel value for display / backtest output
    rs_accel = float(_acceleration.iloc[-2]) if len(_acceleration) >= 2 else 0.0

    # ═══════════════════════════════════════════════════════
    # F2 — RS vs SECTOR  (0-10 pts)
    # Outperforming own sector = double confirmation
    # Bonus +2 if sector is in top-3 rotating sectors today
    # Sector Momentum Acceleration: 5d beat > 10d beat → money rotating in NOW
    # ═══════════════════════════════════════════════════════
    sect_ret, sect_ret_10d, sect_name = get_sector_return(ticker) if ticker else (None, None, None)
    if sect_ret is not None and len(hc) >= 6:
        hc_base    = float(hc.iloc[-6])
        stock_r5   = float(hc.iloc[-1] / hc_base - 1) if hc_base != 0 else 0
        sect_beat  = stock_r5 - sect_ret   # raw alpha vs own sector

        # FIX I-11: sector alpha percentile via tanh normalisation.
        # Replaces fixed ±3% band with vol-adaptive scaling.
        # alpha in σ-units: divide by stock's own 5d return std.
        _daily_r_5 = hc.pct_change().dropna().tail(5)
        _r5_std    = max(float(_daily_r_5.std()) if len(_daily_r_5) >= 3 else 0.01, 0.001)
        _sect_alpha_z = sect_beat / (_r5_std * np.sqrt(5) + 1e-9)
        # tanh → 0.5 at 0 alpha, →1 as alpha→+∞, →0 as alpha→-∞
        rs_sect_sc = float(0.5 * (1.0 + np.tanh(_sect_alpha_z / 1.5)))

        if sect_name in top_sectors:
            rs_sect_sc = min(1.0, rs_sect_sc + 0.15)

        # FIX B-01: real sector acceleration using two independent timeframes
        if sect_ret_10d is not None:
            sect_accel = sect_ret - sect_ret_10d   # positive = sector accelerating
        else:
            sect_accel = 0.0
        if sect_accel > 0.005:          # >0.5% acceleration threshold (avoids noise)
            rs_sect_sc = min(1.0, rs_sect_sc + 0.1)
        rs_sect_pts = round(rs_sect_sc * 10, 1)
    else:
        rs_sect_pts = 5.0
        sect_name   = "?"

    # ═══════════════════════════════════════════════════════
    # VOLUME DISTRIBUTION STATS — used by F3 and horizon classifier
    # vol_mu / vol_sigma derived from the stock's own last-20d history.
    # This makes ALL volume scoring dynamic — no hardcoded multiples.
    # vol_bo_thresh = μ + 1.5σ  →  the "breakout surge" bar.
    # ═══════════════════════════════════════════════════════
    vol_series_20 = v.iloc[-21:-1] if len(v) >= 21 else v.iloc[:-1]
    vol_mu        = float(vol_series_20.mean()) if len(vol_series_20) > 0 else float(vol_ma20)
    vol_sigma     = float(vol_series_20.std())  if len(vol_series_20) > 1 else vol_mu * 0.3
    vol_sigma     = max(vol_sigma, vol_mu * 0.05)   # floor: avoid div-by-zero on flat vol
    vol_bo_thresh = vol_mu + 1.5 * vol_sigma         # dynamic breakout volume threshold

    # ═══════════════════════════════════════════════════════
    # F3 — VOLUME SURGE  (0-15 pts)
    # Dynamic: thresholds derived from the stock's own vol distribution.
    # Surge = (vol − μ) / σ  (z-score).
    # BO: high z-score = strong demand signal.
    # PB: negative z (dry-up) = healthy, shows lack of distribution.
    # ═══════════════════════════════════════════════════════
    vol_ratio = volume_surge(day_vol, v)
    vol_z     = (day_vol - vol_mu) / (vol_sigma + 1e-9)   # z-score of today's vol

    if setup_type == "Breakout":
        # Score proportional to z-score: z≥3→15, z≥2→12, z≥1.5→9, z≥1→6, z≥0→3
        if   vol_z >= 3.0: vol_pts = 15
        elif vol_z >= 2.0: vol_pts = 12
        elif vol_z >= 1.5: vol_pts = 9
        elif vol_z >= 1.0: vol_pts = 6
        elif vol_z >= 0.0: vol_pts = 3
        else:              vol_pts = 0
    else:
        # Pullback dry-up: z-score should be negative (below average)
        pb_vol_last3 = float(hv.tail(3).mean())
        pb_z = (pb_vol_last3 - vol_mu) / (vol_sigma + 1e-9)
        if   pb_z <= -1.5: vol_pts = 15   # very dry — ideal pullback
        elif pb_z <= -0.5: vol_pts = 11
        elif pb_z <= 0.0:  vol_pts = 6
        elif pb_z <= 0.5:  vol_pts = 2
        else:              vol_pts = 0    # distribution on pullback — bearish

    # ═══════════════════════════════════════════════════════
    # F4 — PRE-BREAKOUT ACCUMULATION  (0-10 pts)
    # vol5/vol20 ratio: measures whether recent (5d) volume is
    # elevated vs the 20d baseline — proxy for institutional demand.
    # AUDIT FIX: continuous score using sigmoid centred at ratio=1.3
    # (neutral accumulation). Avoids hard 1.2/1.4/1.6 step-thresholds.
    # sigmoid(x) = 10 / (1 + exp(-k*(x-x0)))  k=4, x0=1.3
    # ═══════════════════════════════════════════════════════
    inst_ratio = vol_ma5 / (vol_ma20 + 1e-9)
    inst_pts   = round(10.0 / (1.0 + np.exp(-4.0 * (inst_ratio - 1.3))), 1)
    inst_pts   = max(0.0, min(10.0, inst_pts))

    # ═══════════════════════════════════════════════════════
    # F5 — VOLATILITY CONTRACTION + RANGE COMPRESSION  (0-10 pts)
    # FIX I-06: ATR5/ATR20 ratio percentile-ranked over 250d history.
    # LOW percentile = stock is more compressed than usual = coiling energy.
    # Score = 1 - percentile (we want LOW ratio = HIGH score).
    # FIX I-07: Same treatment for range compression (range5/range20).
    # Both fully adaptive to the stock's own volatility regime.
    # ═══════════════════════════════════════════════════════

    # Build ATR5/ATR20 ratio history (on hist slice, no look-ahead)
    _tr_series  = pd.concat([hh - hl,
                              (hh - hc.shift(1)).abs(),
                              (hl - hc.shift(1)).abs()], axis=1).max(axis=1)
    _atr5_hist  = _tr_series.rolling(5).mean()
    _atr20_hist = _tr_series.rolling(20).mean()
    _vc_series  = _atr5_hist / (_atr20_hist.replace(0, np.nan))
    _vc_pct     = percentile_last(_vc_series, min(250, len(_vc_series)))
    if pd.isna(_vc_pct):
        _vc_pct = 0.5
    # Low ratio = high compression = good → invert: score = 1 - percentile
    vc_pts = round((1.0 - _vc_pct) * 5, 1)   # 0-5 pts

    # Range Compression Index — percentile-ranked
    _rng_series = (hh.rolling(5).max() - hl.rolling(5).min()) / \
                  (hh.rolling(20).max() - hl.rolling(20).min() + 1e-9)
    _rci_pct    = percentile_last(_rng_series, min(250, len(_rng_series)))
    if pd.isna(_rci_pct):
        _rci_pct = 0.5
    rci     = float(_rng_series.iloc[-1]) if pd.notna(_rng_series.iloc[-1]) else 1.0
    rci_pts = round((1.0 - _rci_pct) * 5, 1)  # 0-5 pts

    vc_pts  = vc_pts + rci_pts   # combined 0-10

    # Keep vc_ratio for VCVE and horizon computations
    atr5_h  = float(_tr_series.iloc[-5:].mean())  if len(_tr_series) >= 5  else atr_v
    atr20_h = float(_tr_series.iloc[-20:].mean()) if len(_tr_series) >= 20 else atr_v
    vc_ratio = atr5_h / (atr20_h + 1e-9)

    # ── VCVE: Volume-Compression Interaction (bonus) ──
    # Detects hidden accumulation: rising vol + falling volatility.
    # Evaluated here because vc_ratio is now available.
    vcve     = inst_ratio * (1.0 - min(vc_ratio, 1.0))
    if   vcve >= 0.7: vcve_bonus = 3
    elif vcve >= 0.5: vcve_bonus = 2
    elif vcve >= 0.3: vcve_bonus = 1
    else:             vcve_bonus = 0

    # Range 5 / Range 20 scalar for display
    range5  = float((hh.tail(5).max()  - hl.tail(5).min()))
    range20 = float((hh.tail(20).max() - hl.tail(20).min()))

    # ═══════════════════════════════════════════════════════
    # F6 — BASE / COIL QUALITY + BASE POSITION  (0-10 pts)
    # Includes flat-resistance + position score (>0.75 = at resistance)
    # ═══════════════════════════════════════════════════════
    if setup_type == "Breakout":
        rng5  = float(hh.tail(5).max()) - float(hl.tail(5).min())
        tight = 1.0 - min(1.0, rng5 / (base_rng + 1e-9))
        rec_hi    = hh.tail(8)
        # AUDIT FIX: normalise high-spread by ATR instead of magic *40 multiplier
        # hi_spread as fraction of ATR = how many ATR-widths does resistance vary?
        # 0 = perfectly flat (all highs identical), 1 = spread equals one full ATR
        hi_spread = (rec_hi.max() - rec_hi.min()) / (atr_v + 1e-9)
        flatness  = max(0.0, 1.0 - min(hi_spread / 1.0, 1.0))   # 0 ATR-spread = full flat
        coil_sc   = tight * 0.55 + flatness * 0.45
        # Base position: price pressing resistance?
        base_pos  = (ltp - base_lo) / (base_rng + 1e-9)
        if base_pos >= 0.85: coil_sc = min(1.0, coil_sc + 0.2)
        elif base_pos >= 0.75: coil_sc = min(1.0, coil_sc + 0.1)
    else:
        psw_hi  = float(hh.tail(20).max())
        psw_lo  = float(hl.tail(40).min())
        pm      = psw_hi - psw_lo + 1e-9
        pb_dep  = (psw_hi - float(hc.iloc[-1])) / pm
        if   0.25 <= pb_dep <= 0.50: coil_sc = 1.0
        elif 0.15 <= pb_dep <  0.25: coil_sc = 0.7
        elif 0.50 <  pb_dep <= 0.65: coil_sc = 0.5
        else:                        coil_sc = 0.2
        base_pos = (ltp - base_lo) / (base_rng + 1e-9)
    coil_pts = round(coil_sc * 10, 1)

    # ─────────────────────────────────────────────────────
    # FIX I-08: 52-WEEK POSITION SCORE (bonus up to +3 pts)
    # position = (price - low_250) / (high_250 - low_250)
    # Percentile-ranked over own 250d position history.
    # High percentile = near 52w highs = momentum leader.
    # This identifies true leaders (near highs while compressing)
    # vs stocks merely bouncing from oversold lows.
    # ─────────────────────────────────────────────────────
    if len(hc) >= 50:
        _n250      = min(250, len(hc))
        _hi250     = float(hh.tail(_n250).max())
        _lo250     = float(hl.tail(_n250).min())
        _pos_now   = (ltp - _lo250) / (_hi250 - _lo250 + 1e-9)
        # Build rolling position series over same window
        _pos_series = (hc - hc.rolling(_n250).min()) / \
                      (hc.rolling(_n250).max() - hc.rolling(_n250).min() + 1e-9)
        _pos_pct    = percentile_last(_pos_series, min(250, len(_pos_series)))
        if pd.notna(_pos_pct):
            pos52w_bonus = round(_pos_pct * 3, 1)   # 0-3 pts
            pos52w       = round(_pos_now, 3)
        else:
            pos52w_bonus = 0.0
            pos52w       = round(base_pos, 3)
    else:
        pos52w_bonus = 0.0
        pos52w       = round(base_pos, 3)

    # ═══════════════════════════════════════════════════════
    # LIQUIDITY SWEEP DETECTION (bonus pts)
    # Low < prior support, close > prior close, large lower wick, vol spike
    # AUDIT FIX: wick threshold now ATR-relative (lower_wick >= 0.5×ATR)
    # instead of fixed fraction of candle range. This prevents false sweeps
    # on inside bars with tiny ranges.
    # ═══════════════════════════════════════════════════════
    sweep_bonus = 0
    if len(hc) >= 5:
        prior_support = float(hl.tail(5).min())
        prior_close   = float(hc.iloc[-1])
        lower_wick    = min(day_o, ltp) - day_lo     # body bottom - day low
        # require lower wick ≥ 0.5 ATR (meaningful rejection, not dust)
        if (day_lo < prior_support and
                ltp > prior_close and
                lower_wick >= 0.5 * atr_v and
                vol_z >= 1.0):                        # vol must be above average (z≥1)
            sweep_bonus = 4

    # ═══════════════════════════════════════════════════════
    # VWMA-20 POSITION (bonus pts)
    # FIX B-04: This is a 20-day Volume-Weighted Moving Average on daily bars,
    # NOT intraday VWAP. Renamed to VWMA20 to avoid confusion.
    # Price > VWMA20 = buyers controlling trend; adds 2 pts.
    # ═══════════════════════════════════════════════════════
    vwap_bonus = 0
    if "volume" in df.columns and len(df) >= 20:
        typical  = (h + l + c) / 3
        cum_tv   = (typical * v).rolling(20).sum()
        cum_v    = v.rolling(20).sum()
        vwma20_val = float((cum_tv / cum_v.replace(0, np.nan)).iloc[-1])
        if not np.isnan(vwma20_val):
            if ltp > vwma20_val:
                vwap_bonus = 2
            # VWMA20 trending upward (today > yesterday)
            vwma20_prev = float((cum_tv / cum_v.replace(0, np.nan)).iloc[-2]) if len(df) >= 21 else vwma20_val
            if not np.isnan(vwma20_prev) and vwma20_val > vwma20_prev:
                vwap_bonus += 1

    # ═══════════════════════════════════════════════════════
    # MOMENTUM STABILITY FILTER
    # AUDIT FIX: Use 20d window (was 10d) — 10 days too noisy.
    # positive_days / 20 — measures consistency of upward drift.
    # Threshold 0.3 = needs at least 6 of 20 days positive to pass.
    # Stocks in confirmed uptrends typically score 0.55–0.75.
    # ═══════════════════════════════════════════════════════
    if len(hc) >= 21:
        returns_20    = hc.iloc[-20:].pct_change().dropna()
        positive_days = int((returns_20 > 0).sum())
        stability     = positive_days / max(len(returns_20), 1)
    elif len(hc) >= 11:
        returns_10    = hc.iloc[-10:].pct_change().dropna()
        positive_days = int((returns_10 > 0).sum())
        stability     = positive_days / max(len(returns_10), 1)
    else:
        stability = 0.5
    # Filter out chaotic stocks — stability < 0.35 = no steady trend
    if stability < 0.35:
        return None
    stab_bonus = 2 if stability >= 0.65 else 1 if stability >= 0.55 else 0

    # ═══════════════════════════════════════════════════════
    # F7 — TREND STRUCTURE / MA  (0-10 pts)
    # FIX I-04: Replace hard step scoring with EMA9/EMA50 ratio
    # percentile-ranked over 250d history — fully continuous, adaptive.
    # trend_ratio = EMA9 / EMA50 captures both direction AND strength.
    # percentile_last(trend_ratio, 250) → 0 at 250d lows, 1 at 250d highs.
    # Bonus: classic EMA alignment check still used as a quality gate.
    # ═══════════════════════════════════════════════════════
    trend_ratio_series = e9 / e50.replace(0, np.nan)
    trend_pct  = percentile_last(trend_ratio_series.iloc[:-1], min(250, len(df)-1))
    if pd.isna(trend_pct):
        trend_pct = 0.5
    ma_pts = round(trend_pct * 8, 1)   # 0-8 pts from percentile

    # Alignment bonus: EMA9 > EMA20 > EMA50 = clean uptrend → +2 pts
    yc = float(hc.iloc[-1])
    if e9_y > e20_y and e20_y > e50_v:
        ma_pts = min(10.0, ma_pts + 2.0)
    elif yc > e9_y:
        ma_pts = min(10.0, ma_pts + 1.0)

    # ═══════════════════════════════════════════════════════
    # F8 — BREAKOUT PROXIMITY  (0-10 pts)
    # AUDIT FIX: Replace 4 hard step-tiers with a continuous
    # exponential decay: score = 10 × exp(−λ × d_trig_atr)
    # where λ=1.5 gives ~9 pts at 0 ATR, ~4 pts at 1 ATR, ~1 pt at 2 ATR.
    # Pullback: same decay on distance to nearest EMA (9 or 20).
    # ═══════════════════════════════════════════════════════
    _prox_lambda = 1.5   # decay rate — 1 ATR distance = score × exp(-1.5) ≈ 22%
    if setup_type == "Breakout":
        d_trig = (base_hi - ltp) / (atr_v + 1e-9)   # negative = already breaking
        # Clamp: deeply broken (>2 ATR past resistance) gets 0; in base = full decay
        d_for_score = max(0.0, d_trig)               # negative = at/above = 0 distance
        prox_pts = round(10.0 * np.exp(-_prox_lambda * d_for_score), 1)
        prox_pts = max(0.0, min(10.0, prox_pts))
    else:
        d_e9  = abs(ltp - e9_v)  / (atr_v + 1e-9)
        d_e20 = abs(ltp - e20_v) / (atr_v + 1e-9)
        prox_dist = min(d_e9, d_e20)
        prox_pts  = round(10.0 * np.exp(-_prox_lambda * prox_dist), 1)
        prox_pts  = max(0.0, min(10.0, prox_pts))

    # ═══════════════════════════════════════════════════════
    # F9 — ATR POTENTIAL  (0-5 pts)
    # Dynamic: score relative to the 60d rolling ATR% distribution
    # of this stock — not fixed global thresholds.
    # Scores highest when today's ATR% is in the top quartile
    # of its own 60d history (i.e. stock is more volatile than usual).
    # ═══════════════════════════════════════════════════════
    atr_hist_pct = atr14(df).iloc[:-1] / c.iloc[:-1] * 100   # historical ATR%
    atr_hist_pct = atr_hist_pct.tail(60).dropna()
    if len(atr_hist_pct) >= 10:
        atr_pct_rank = float((atr_hist_pct <= atr_pct).mean())  # percentile 0-1
        atp_pts = round(atr_pct_rank * 5, 1)
    else:
        # Fallback to absolute thresholds when insufficient history
        if   atr_pct >= 3.0: atp_pts = 5
        elif atr_pct >= 2.0: atp_pts = 4
        elif atr_pct >= 1.5: atp_pts = 3
        elif atr_pct >= 1.2: atp_pts = 2
        else:                atp_pts = 1

    # ═══════════════════════════════════════════════════════
    # F10 — CANDLESTICK TRIGGER  (0-5 pts)
    # ═══════════════════════════════════════════════════════
    raw_cdl, candle_names = detect_candle_patterns(
        day_o, day_hi, day_lo, ltp,
        float(o.iloc[-2]), float(h.iloc[-2]),
        float(l.iloc[-2]), float(c.iloc[-2])
    )
    cdl_pts = min(round(raw_cdl * 0.5, 1), 5.0)

    # ── TOTAL (base factors: 100 pts max) ──
    total = (rs_pts + rs_sect_pts + vol_pts + inst_pts +
             vc_pts + coil_pts + ma_pts + prox_pts + atp_pts + cdl_pts)

    # ── BONUS SIGNALS (additive, capped at 100) ──
    # FIX I-08: pos52w_bonus added (0-3 pts) — 52-week position leader premium
    bonuses = rs_accel_bonus + vcve_bonus + sweep_bonus + vwap_bonus + stab_bonus + pos52w_bonus
    total  += bonuses

    # ── ADJUSTMENTS ──
    if setup_type == "Breakout" and breakout_ext > 1.0:
        total -= round((breakout_ext - 1.0) * 10, 1)
    if setup_type == "Pullback" and rsi_p < 45 and rsi_v > rsi_p:
        total += 5
    if setup_type == "Pullback" and ltp < e20_v:
        total -= 4

    # FIX I-10: Replace binary Nifty/VIX penalties with continuous breadth adjustment.
    # breadth_score = percentile of (% stocks above EMA20) vs 200d history.
    # We approximate breadth via Nifty position vs 20DMA and VIX percentile.
    # nifty_above_20dma → replaced by continuous nifty position score.
    # vix_falling → replaced by VIX trend strength.
    # Both derived from mkt data already fetched; no extra API calls needed.
    _nifty_breadth_adj = 0.0
    if mkt.get("nifty_r5") is not None and mkt.get("nifty_r20") is not None:
        # Use nifty 5d and 20d returns as breadth proxy:
        # Strong +ve trend = broad participation likely → small bonus
        # Weak / negative = breadth deteriorating → penalty
        _n5  = mkt["nifty_r5"]  or 0.0
        _n20 = mkt["nifty_r20"] or 0.0
        _nifty_breadth_adj = np.clip((_n5 + _n20 * 0.5) * 100, -8, 4)
    elif not mkt["nifty_above_20dma"]:
        _nifty_breadth_adj = -8   # fallback to original binary penalty

    _vix_adj = 0.0
    if mkt.get("vix_level") is not None:
        _vix = mkt["vix_level"]
        # VIX: < 14 = benign (bonus), 14-18 = neutral, 18-25 = elevated (-pts), >25 = danger
        if   _vix < 14:  _vix_adj =  2.0
        elif _vix < 18:  _vix_adj =  0.0
        elif _vix < 22:  _vix_adj = -3.0
        elif _vix < 26:  _vix_adj = -5.0
        else:            _vix_adj = -8.0
        if not mkt["vix_falling"]:   # VIX rising makes it worse
            _vix_adj = min(0.0, _vix_adj - 2.0)
    elif not mkt["vix_falling"]:
        _vix_adj = -5.0   # fallback

    total += _nifty_breadth_adj + _vix_adj
    total = max(0, min(100, round(total, 1)))

    # ── EMI = Score × ATR%  (rewards volatile high-score setups) ──
    emi = round(total * atr_pct / 100, 3)

    # ── COMPOSITE RANK = EMI × LiquidityScore × VolumeStability ──
    liquidity_score  = min(1.0, float(day_vol) / (vol_ma20 * 2 + 1e-9))
    volume_stability = min(1.0, stability)
    composite_rank   = round(emi * (0.6 + 0.2 * liquidity_score + 0.2 * volume_stability), 4)

    # ═══════════════════════════════════════════════════════
    # HORIZON CLASSIFICATION — 5 TIERS, ALL MATH-DERIVED
    # ─────────────────────────────────────────────────────
    # Each tier is defined by ATR-normalised price distances,
    # momentum derivatives, and volume conditions — no magic numbers.
    #
    # Tier 1: IMMINENT BREAKOUT  — price ≤ 0.25×ATR from trigger,
    #          vol_ratio ≥ vol_breakout_threshold (2σ above mean)
    # Tier 2: INTRADAY BUY       — strong candle today, RSI turning,
    #          ltp > VWMA20, vol spike today (ratio ≥ 2× avg)
    # Tier 3: SWING (2-5 days)   — at EMA support / ≤ 1 ATR from trigger
    #          vol_ratio ≥ 1.2×, RSI not overbought
    # Tier 4: MID-TERM (5-14d)   — setup forming, ≤ 3×ATR from trigger
    #          needs volume pickup to confirm
    # Tier 5: LONG-TERM (14-30d) — base building, > 3×ATR away
    #          monitor for compression → expansion
    #
    # Vol breakout threshold = μ + 1.5σ of last 20 volumes (dynamic)
    # ═══════════════════════════════════════════════════════

    # Dynamic volume breakout threshold — already computed above (before F3)
    # vol_mu, vol_sigma, vol_bo_thresh are available here unchanged.

    imminence = prox_pts + vc_pts   # 0–20 scale (proximity + compression)

    if setup_type == "Breakout":
        d_trig_atr = (base_hi - ltp) / (atr_v + 1e-9)   # ≤0 = already breaking
        # IMMINENT: within 0.25 ATR of trigger + volume crossing dynamic threshold
        if d_trig_atr <= 0.25 and day_vol >= vol_bo_thresh:
            horizon = "Imminent BO"
            hz_note = f"AT TRIGGER — vol {vol_ratio:.1f}× avg (threshold {vol_bo_thresh/vol_mu:.1f}×). Enter now or market open."
        # INTRADAY: same session breakout — price above trigger today, strong candle
        elif d_trig_atr <= 0.0 and vol_ratio >= 1.5 and rsi_v < 78:
            horizon = "Intraday"
            hz_note = f"Breaking today — RSI {rsi_v:.0f}, vol {vol_ratio:.1f}×. Trail stop above base low."
        # SWING 2-5d: within 1 ATR, vol accumulating ≥ 1.2×
        elif d_trig_atr <= 1.0 and vol_ratio >= 1.2:
            horizon = "Swing 2-5D"
            hz_note = f"{d_trig_atr:.2f} ATR from trigger, vol building. Place limit order above {base_hi:.1f}."
        # MID-TERM 5-14d: within 3 ATR, needs confirmation
        elif d_trig_atr <= 3.0:
            horizon = "Mid 5-14D"
            hz_note = f"{d_trig_atr:.2f} ATR from trigger. Wait for volume pickup before entry."
        # LONG-TERM 14-30d: > 3 ATR away, base building phase
        else:
            horizon = "Long 14-30D"
            hz_note = f"{d_trig_atr:.2f} ATR from trigger. Base still forming — watch and wait."
    else:  # Pullback
        rsi_turning   = rsi_v > rsi_p                    # RSI crossed up
        pb_depth_atr  = (e20_v - ltp) / (atr_v + 1e-9)  # ≤0 = at/above EMA20
        # INTRADAY: at EMA support today, RSI turning up, vol drying up (healthy PB)
        if pb_depth_atr <= 0.3 and rsi_turning and vol_ratio <= 0.8:
            horizon = "Intraday"
            hz_note = f"EMA20 support + RSI turning ({rsi_v:.0f}↑). Vol dry = clean pullback. Buy near {e20_v:.1f}."
        # IMMINENT: classic 3-bar RSI reversal setup + at EMA + candle pattern
        elif pb_depth_atr <= 0.5 and rsi_turning and raw_cdl >= 2:
            horizon = "Imminent BO"
            hz_note = f"Reversal candle at EMA. RSI {rsi_v:.0f}↑, pattern: {', '.join(candle_names) if candle_names else 'none'}."
        # SWING 2-5d: within 1 ATR of EMA support, RSI > 40
        elif pb_depth_atr <= 1.0 and rsi_v >= 40:
            horizon = "Swing 2-5D"
            hz_note = f"Approaching EMA20. RSI {rsi_v:.0f}. Wait for reversal candle + vol confirmation."
        # MID-TERM 5-14d: pullback still developing
        elif pb_depth_atr <= 2.5:
            horizon = "Mid 5-14D"
            hz_note = f"Pullback deepening ({pb_depth_atr:.1f} ATR below EMA20). Do not enter yet."
        # LONG-TERM 14-30d: extended pullback, needs structural repair
        else:
            horizon = "Long 14-30D"
            hz_note = f"Extended correction ({pb_depth_atr:.1f} ATR below EMA20). Watch for base formation."

    # ═══════════════════════════════════════════════════════
    # ENTRY / TARGET / STOP — fully ATR-derived, horizon-aware
    # ─────────────────────────────────────────────────────
    # ATR multipliers are derived from observed NSE daily move
    # distributions: median 1-day ATR move ≈ 0.5×ATR,
    # median 5-day range ≈ 2.5×ATR, 10-day ≈ 4×ATR.
    # Multipliers below match these empirical distributions.
    #
    # Target multipliers by horizon:
    #   Intraday / Imminent  → 0.75 × ATR  (same-session move)
    #   Swing 2-5D           → 2.0  × ATR
    #   Mid 5-14D            → 3.5  × ATR
    #   Long 14-30D          → 5.0  × ATR
    # ═══════════════════════════════════════════════════════
    _tgt_mult = {
        "Imminent BO": 0.75,
        "Intraday":    0.75,
        "Swing 2-5D":  2.0,
        "Mid 5-14D":   3.5,
        "Long 14-30D": 5.0,
    }
    tgt_mult = _tgt_mult.get(horizon, 2.0)

    if setup_type == "Breakout":
        entry = round(base_hi * 1.002, 2) if ltp < base_hi else round(ltp, 2)
        entry_note = (f"Buy above {entry:.2f} (0.2% above base high {base_hi:.2f})"
                      if ltp < base_hi else f"Breaking now — buy on close above {base_hi:.2f}")
        tgt = round(entry + tgt_mult * atr_v, 2)
        # Stop = base low minus 0.5×ATR (1 ATR-based buffer below consolidation)
        stp = round(base_lo - 0.5 * atr_v, 2)
    else:  # Pullback
        entry = round(ltp, 2)
        entry_note = f"Buy near EMA20 ({e20_v:.2f}) on reversal candle"
        # Target: prior 20d swing high, floored at entry + tgt_mult×ATR
        tgt_struct = round(float(hh.tail(20).max()) * 0.997, 2)
        tgt_atr    = round(entry + tgt_mult * atr_v, 2)
        tgt        = max(tgt_struct, tgt_atr)
        # Stop: 1 ATR below EMA50 (structural trend stop)
        stp = round(e50_v - atr_v, 2)

    risk_raw   = max(entry - stp,  0.01)
    reward_raw = max(tgt  - entry, 0.01)
    rr         = round(reward_raw / risk_raw, 2)   # divide before round avoids 0.01→0.0
    risk       = round(risk_raw,   1)
    reward     = round(reward_raw, 1)
    move_pct   = round((tgt - entry) / entry * 100, 1) if entry != 0 else 0.0

    return {
        # core
        "SetupType": setup_type,
        "Score":     total,
        "EMI":       emi,
        "CompositeRank": composite_rank,
        "Horizon":   horizon,
        "HorizonNote": hz_note,
        # trade levels
        "Entry":     entry,
        "Target":    tgt,
        "Stop":      stp,
        "Risk":      risk,
        "Reward":    reward,
        "RR":        rr,
        "Move%":     move_pct,
        "EntryNote": entry_note,
        # factors
        "RS":        round(rs_pts, 1),
        "RS_Sector": round(rs_sect_pts, 1),
        "Volume":    round(vol_pts, 1),
        "InstVol":   round(inst_pts, 1),
        "VolCont":   round(vc_pts, 1),
        "RCI":       round(rci, 3),
        "Coil":      round(coil_pts, 1),
        "MA_Struct": round(ma_pts, 1),
        "Proximity": round(prox_pts, 1),
        "ATR_Pot":   round(atp_pts, 1),
        "Candle":    round(cdl_pts, 1),
        "Patterns":  ", ".join(candle_names) if candle_names else "—",
        # bonus signals
        "RS_Accel":   round(rs_accel, 4),
        "AccelScore": round(acc_score * 100, 1),  # FIX I-05: continuous acceleration percentile
        "VCVE":       round(vcve, 3),
        "BasePos":    round(base_pos, 3),
        "Pos52W":     round(pos52w, 3),            # FIX I-08: 52-week position percentile
        "Stability":  round(stability, 2),
        "Sweep":      sweep_bonus > 0,
        "VWMA20_OK":  vwap_bonus > 0,   # FIX B-04: renamed from VWAP_OK
        # info
        "RSI7":      round(rsi_v, 1),
        "VolRatio":  round(vol_ratio, 2),
        "VolZ":      round(vol_z, 2),          # dynamic z-score of today's volume
        "VolBOThr":  round(vol_bo_thresh / vol_mu, 2),  # dynamic surge threshold (×avg)
        "InstRatio": round(inst_ratio, 2),
        "VC_Ratio":  round(vc_ratio, 2),
        "ATR%":      round(atr_pct, 2),
        "RS_vs_Nifty": round(rs_score * 100, 1),
        "BO_Ext_ATR":  round(breakout_ext, 2),
        "Sector":    sect_name or "?",
        "EMA9":      round(e9_v, 2),
        "EMA20":     round(e20_v, 2),
        "EMA50":     round(e50_v, 2),
    }


# ── CACHED SCORE HELPER — avoids calling score_stock_dual twice per stock per refresh ──
def get_cached_score(sym, df_raw, live, nifty_r5, nifty_r20):
    """Returns score result from cache if fresh, else recomputes and caches."""
    now = time.time()
    cache = st.session_state.score_cache
    ts    = st.session_state.score_cache_ts
    # invalidate entire cache if it's older than TTL
    if now - ts > SCORE_CACHE_TTL:
        st.session_state.score_cache = {}
        st.session_state.score_cache_ts = now
        cache = st.session_state.score_cache
    if sym not in cache:
        cache[sym] = score_stock_dual(df_raw, live, nifty_r5, nifty_r20, ticker=sym)
    return cache[sym]


# ── RUN THE SCREENER ──

if not st.session_state.raw_data_cache:
    st.info("Run extraction first to score stocks")
else:
    if nifty_r5 is None:
        st.warning("⚠️ Could not fetch Nifty data — Relative Strength factor will be neutral (0.5)")

    screener_rows = []
    for sym, df_raw in st.session_state.raw_data_cache.items():
        live   = get_live_bar(sym)
        result = get_cached_score(sym, df_raw, live, nifty_r5, nifty_r20)   # PERF: cached
        if result is None:
            continue
        live_ltp = live.get("ltp") or float(df_raw["close"].iloc[-1])
        screener_rows.append({
            "Ticker":  sym,
            "LTP":     round(live_ltp, 2),
            "DayHigh": round(float(live.get("high", df_raw["high"].iloc[-1])), 2),
            "DayLow":  round(float(live.get("low",  df_raw["low"].iloc[-1])),  2),
            "LiveVol": int(live["volume"]) if live.get("volume") else None,
            **result,
        })

    if not screener_rows:
        st.warning("No stocks passed filters — market may be closed or extraction needed")
    else:
        df_out = pd.DataFrame(screener_rows).sort_values("Score", ascending=False).reset_index(drop=True)

        # ── Sort by CompositeRank = EMI × LiquidityScore × VolumeStability ──
        df_out = df_out.sort_values("CompositeRank", ascending=False).reset_index(drop=True)
        if "Rank" in df_out.columns:
            df_out.drop(columns=["Rank"], inplace=True)
        df_out.insert(0, "Rank", df_out.index + 1)

        # ── MARKET CONTEXT BANNER ──
        mkt_col = "#00d084" if mkt["market_ok"] else "#ff3b3b"
        mkt_note = "  ·  ".join(mkt["market_notes"]) if mkt["market_notes"] else "Market conditions normal"
        nifty_lbl = "▲ ABOVE 20DMA" if mkt["nifty_above_20dma"] else "▼ BELOW 20DMA"
        vix_lbl   = f"VIX {mkt['vix_level']} FALLING ✓" if mkt["vix_falling"] else f"VIX {mkt['vix_level']} RISING ⚠"
        top_s     = "  ".join(sorted(top_sectors)) if top_sectors else "—"
        st.markdown(f"""
<div style="background:#0d0d0d;border:1px solid #2a2a2a;border-left:4px solid {mkt_col};
padding:10px 16px;margin-bottom:10px;font-family:'IBM Plex Mono',monospace;">
  <div style="display:flex;gap:28px;align-items:center;flex-wrap:wrap;">
    <span style="color:#555;font-size:.6rem;letter-spacing:.12em;">MARKET</span>
    <span style="color:{mkt_col};font-size:.72rem;font-weight:700;">NIFTY {nifty_lbl}</span>
    <span style="color:{mkt_col};font-size:.72rem;">{vix_lbl}</span>
    <span style="color:#ff8c00;font-size:.72rem;">⭐ SECTORS: {top_s}</span>
    <span style="color:#555;font-size:.65rem;">{mkt_note}</span>
  </div>
</div>""", unsafe_allow_html=True)

        # ── COLUMN SETS ──
        CORE_COLS = ["Rank","Ticker","Sector","SetupType","Horizon","Score","EMI","CompositeRank",
                     "LTP","Entry","Target","Stop","RR","Move%","ATR%",
                     "RSI7","VolRatio","VolZ","RS","RS_Sector","MA_Struct",
                     "VolCont","Proximity","Candle","Patterns","HorizonNote"]
        TRADE_COLS = ["Rank","Ticker","Sector","SetupType","Horizon","Score","EMI",
                      "LTP","Entry","Target","Stop","RR","Move%","ATR%",
                      "RSI7","VolRatio","VolZ","HorizonNote","Sweep","VWMA20_OK","Stability"]
        FACTOR_COLS = ["Rank","Ticker","Sector","SetupType","Score",
                       "RS","RS_Sector","Volume","InstVol","VolCont","RCI",
                       "Coil","MA_Struct","Proximity","ATR_Pot","Candle","VCVE","BasePos"]

        def _vcols(d, cols):
            return [c for c in cols if c in d.columns]

        # ── Bloomberg-style cell coloriser ──
        def style_df(d, cols):
            disp = d[_vcols(d, cols)].copy()

            def score_bg(v):
                if not isinstance(v, (int, float)): return ""
                if v >= 70: return "background-color:#0d2200;color:#00d084;font-weight:700"
                if v >= 55: return "background-color:#1a2200;color:#b8e06a"
                if v >= 40: return "background-color:#2a1800;color:#ffb347"
                return "color:#555555"

            def setup_bg(v):
                if v == "Breakout": return "background-color:#001522;color:#1e90ff;font-weight:700"
                if v == "Pullback": return "background-color:#140014;color:#cc88ff;font-weight:700"
                return ""

            def horizon_bg(v):
                v = str(v)
                if "Imminent" in v: return "background-color:#002200;color:#00ff88;font-weight:700;letter-spacing:.05em"
                if "Intraday" in v: return "background-color:#001a22;color:#00ccff;font-weight:700"
                if "Swing"    in v: return "background-color:#1a2200;color:#b8e06a;font-weight:600"
                if "Mid"      in v: return "background-color:#2a1800;color:#ffb347"
                if "Long"     in v: return "color:#666666"
                return ""

            def rr_bg(v):
                if not isinstance(v, (int, float)): return ""
                if v >= 3.0: return "color:#00ff88;font-weight:700"
                if v >= 2.0: return "color:#00d084;font-weight:600"
                if v >= 1.5: return "color:#ffb347"
                return "color:#ff3b3b"

            def rsi_bg(v):
                if not isinstance(v, (int, float)): return ""
                if v >= 70: return "color:#ff3b3b;font-weight:700"
                if v >= 60: return "color:#ffb347"
                if v <= 35: return "color:#1e90ff;font-weight:700"
                return "color:#00d084"

            def volz_bg(v):
                if not isinstance(v, (int, float)): return ""
                if v >= 3.0: return "color:#00ff88;font-weight:700"
                if v >= 2.0: return "color:#00d084"
                if v >= 1.0: return "color:#b8e06a"
                if v <= -1.0: return "color:#1e90ff"
                return "color:#666666"

            def ltp_bg(v):
                return "color:#e8e8e8;font-weight:600" if isinstance(v, (int, float)) else ""

            def entry_bg(v):
                return "color:#ff8c00;font-weight:700" if isinstance(v, (int, float)) else ""

            def tgt_bg(v):
                return "color:#00d084;font-weight:700" if isinstance(v, (int, float)) else ""

            def stp_bg(v):
                return "color:#ff3b3b" if isinstance(v, (int, float)) else ""

            sty = disp.style.set_properties(**{
                'font-family': 'IBM Plex Mono, monospace',
                'font-size': '0.70rem',
            })
            col_map = {
                "Score":    score_bg,
                "SetupType":setup_bg,
                "Horizon":  horizon_bg,
                "RR":       rr_bg,
                "RSI7":     rsi_bg,
                "VolZ":     volz_bg,
                "LTP":      ltp_bg,
                "Entry":    entry_bg,
                "Target":   tgt_bg,
                "Stop":     stp_bg,
            }
            for col, fn in col_map.items():
                if col in disp.columns:
                    sty = sty.applymap(fn, subset=[col])
            return sty

        # ── BB table header ──
        def bb_table_header(title, subtitle=""):
            st.markdown(f"""
<div style="background:#1a1200;border-top:2px solid #ff8c00;border-bottom:1px solid #2a2a2a;
     padding:6px 14px;display:flex;justify-content:space-between;align-items:center;
     font-family:'IBM Plex Mono',monospace;margin-top:8px;">
  <span style="color:#ff8c00;font-size:.72rem;font-weight:700;letter-spacing:.12em;">◼ {title.upper()}</span>
  <span style="color:#555;font-size:.60rem;">{subtitle}</span>
</div>""", unsafe_allow_html=True)

        # ── Count per horizon ──
        n_imm  = len(df_out[df_out.Horizon=="Imminent BO"])
        n_intr = len(df_out[df_out.Horizon=="Intraday"])
        n_sw   = len(df_out[df_out.Horizon=="Swing 2-5D"])
        n_mid  = len(df_out[df_out.Horizon=="Mid 5-14D"])
        n_long = len(df_out[df_out.Horizon=="Long 14-30D"])
        n_bo   = len(df_out[df_out.SetupType=="Breakout"])
        n_pb   = len(df_out[df_out.SetupType=="Pullback"])

        # ── HORIZON TABS ──
        tab_all, tab_imm, tab_intr, tab_sw, tab_mid, tab_long = st.tabs([
            f"◼ ALL  ({len(df_out)})",
            f"🔴 IMMINENT  ({n_imm})",
            f"⚡ INTRADAY  ({n_intr})",
            f"📈 SWING 2-5D  ({n_sw})",
            f"📊 MID 5-14D  ({n_mid})",
            f"🏦 LONG 14-30D  ({n_long})",
        ])

        with tab_all:
            bb_table_header("ALL SETUPS", f"RANKED BY COMPOSITE RANK = EMI × LIQUIDITY × STABILITY  ·  {len(df_out)} STOCKS")
            st.dataframe(style_df(df_out, CORE_COLS), use_container_width=True, hide_index=True, height=420)

        with tab_imm:
            d_imm = df_out[df_out.Horizon=="Imminent BO"].reset_index(drop=True)
            bb_table_header("IMMINENT BREAKOUT", "AT TRIGGER + VOLUME CROSSING σ-THRESHOLD — ACT NOW OR NEXT OPEN")
            if d_imm.empty:
                st.markdown("""<div style="font-family:'IBM Plex Mono';color:#555;font-size:.72rem;
                padding:16px;">No imminent breakouts — these appear when price ≤0.25 ATR from trigger
                AND volume crosses the dynamic 1.5σ threshold above 20d mean.</div>""", unsafe_allow_html=True)
            else:
                st.dataframe(style_df(d_imm, TRADE_COLS), use_container_width=True, hide_index=True)

        with tab_intr:
            d_intr = df_out[df_out.Horizon=="Intraday"].reset_index(drop=True)
            bb_table_header("INTRADAY SIGNALS", "SAME-SESSION ENTRY — TARGET 0.75×ATR FROM ENTRY")
            if d_intr.empty:
                st.markdown("""<div style="font-family:'IBM Plex Mono';color:#555;font-size:.72rem;
                padding:16px;">No intraday signals. Intraday BO requires: price above trigger + vol ≥1.5× + RSI &lt;78.
                Intraday PB requires: at EMA20 + RSI turning + clean dry-up vol.</div>""", unsafe_allow_html=True)
            else:
                st.dataframe(style_df(d_intr, TRADE_COLS), use_container_width=True, hide_index=True)

        with tab_sw:
            d_sw = df_out[df_out.Horizon=="Swing 2-5D"].reset_index(drop=True)
            bb_table_header("SWING TRADE  2-5 DAYS", "TARGET 2×ATR FROM ENTRY  ·  PLACE LIMIT ABOVE TRIGGER")
            if d_sw.empty:
                st.markdown("""<div style="font-family:'IBM Plex Mono';color:#555;font-size:.72rem;
                padding:16px;">No swing setups (2-5 day). These appear when price is within 1 ATR
                of trigger with vol ≥1.2× and RSI not overbought.</div>""", unsafe_allow_html=True)
            else:
                st.dataframe(style_df(d_sw, TRADE_COLS), use_container_width=True, hide_index=True)

        with tab_mid:
            d_mid = df_out[df_out.Horizon=="Mid 5-14D"].reset_index(drop=True)
            bb_table_header("MID-TERM  5-14 DAYS", "TARGET 3.5×ATR FROM ENTRY  ·  WATCH FOR VOLUME CONFIRMATION")
            if d_mid.empty:
                st.markdown("""<div style="font-family:'IBM Plex Mono';color:#555;font-size:.72rem;
                padding:16px;">No mid-term setups (5-14 day). These appear when price is within 3 ATR
                of trigger, setup still developing.</div>""", unsafe_allow_html=True)
            else:
                st.dataframe(style_df(d_mid, TRADE_COLS), use_container_width=True, hide_index=True)

        with tab_long:
            d_long = df_out[df_out.Horizon=="Long 14-30D"].reset_index(drop=True)
            bb_table_header("LONG-TERM  14-30 DAYS", "TARGET 5×ATR FROM ENTRY  ·  BASE BUILDING PHASE")
            if d_long.empty:
                st.markdown("""<div style="font-family:'IBM Plex Mono';color:#555;font-size:.72rem;
                padding:16px;">No long-term setups currently. These appear when price is >3 ATR
                from trigger — base building, not yet actionable.</div>""", unsafe_allow_html=True)
            else:
                st.dataframe(style_df(d_long, FACTOR_COLS), use_container_width=True, hide_index=True)

        # ── TOP 5 ACTIONABLE CARDS ──
        st.divider()
        bb_table_header("TOP 5 SETUPS", "RANKED BY COMPOSITE RANK = EMI × LIQUIDITY × STABILITY")
        top5 = df_out.head(5)
        c5 = st.columns(5)
        _hz_colors = {
            "Imminent BO": "#00ff88",
            "Intraday":    "#00ccff",
            "Swing 2-5D":  "#b8e06a",
            "Mid 5-14D":   "#ffb347",
            "Long 14-30D": "#777777",
        }
        for i, (_, row) in enumerate(top5.iterrows()):
            with c5[i]:
                icon   = "🚀" if row["SetupType"]=="Breakout" else "🔁"
                hc_    = _hz_colors.get(row["Horizon"], "#777")
                rr_c   = "#00ff88" if row["RR"]>=3.0 else "#00d084" if row["RR"]>=2.0 else "#ffb347" if row["RR"]>=1.5 else "#ff3b3b"
                st.markdown(f"""
<div style="background:#0a0a0a;border:1px solid #2a2a2a;border-top:3px solid #ff8c00;
padding:10px 9px;font-family:'IBM Plex Mono',monospace;">
  <div style="font-size:.65rem;color:#ff8c00;font-weight:700;letter-spacing:.1em;">
    {icon} #{int(row['Rank'])}  {row['Ticker']}
  </div>
  <div style="font-size:.62rem;color:#888;">{row.get('Sector','?')}</div>
  <div style="font-size:1rem;color:#e8e8e8;font-weight:600;margin:3px 0;">₹{row['LTP']:.2f}</div>
  <div style="font-size:.60rem;color:{hc_};font-weight:700;letter-spacing:.04em;">◼ {row['Horizon']}</div>
  <div style="font-size:.57rem;color:#888;">Score {row['Score']}  EMI {row['EMI']}</div>
  <hr style="border-color:#222;margin:5px 0;">
  <div style="font-size:.60rem;">Entry  <b style="color:#ff8c00;">₹{row['Entry']}</b></div>
  <div style="font-size:.60rem;">Target <b style="color:#00d084;">₹{row['Target']}</b></div>
  <div style="font-size:.60rem;">Stop   <b style="color:#ff3b3b;">₹{row['Stop']}</b></div>
  <div style="font-size:.62rem;color:{rr_c};font-weight:700;margin-top:4px;">
    R:R 1:{row['RR']}  +{row['Move%']}%
  </div>
  <div style="font-size:.55rem;color:#555;margin-top:3px;">{row.get('Patterns','—')}</div>
</div>""", unsafe_allow_html=True)

        # ── FACTOR BREAKDOWN CHART ──
        st.divider()
        st.markdown("""
<div style="background:#1a1200;border-top:2px solid #ff8c00;border-bottom:1px solid #2a2a2a;
     padding:6px 14px;font-family:'IBM Plex Mono',monospace;margin-bottom:4px;">
  <span style="color:#ff8c00;font-size:.72rem;font-weight:700;letter-spacing:.12em;">
    ◼ SCORE BREAKDOWN — TOP 12 BY COMPOSITE RANK
  </span>
  <span style="color:#555;font-size:.60rem;margin-left:20px;">
    stacked = factor contribution  ·  label = horizon tier
  </span>
</div>""", unsafe_allow_html=True)
        top12 = df_out.head(12)
        # Factor columns and Bloomberg-palette colours
        fcols   = ["RS","RS_Sector","Volume","InstVol","VolCont","RCI",
                   "Coil","MA_Struct","Proximity","ATR_Pot","Candle"]
        fcolors = ["#ff8c00","#ffb347","#1e90ff","#00ccff","#00d084",
                   "#26a69a","#8bc34a","#9c27b0","#e91e63","#ffc107","#ff5722"]
        _hz_sym = {
            "Imminent BO": "🔴",
            "Intraday":    "⚡",
            "Swing 2-5D":  "📈",
            "Mid 5-14D":   "📊",
            "Long 14-30D": "🏦",
        }
        fig_br = go.Figure()
        for fc, fc_col in zip(fcols, fcolors):
            if fc in top12.columns:
                fig_br.add_trace(go.Bar(
                    name=fc, x=top12["Ticker"], y=top12[fc],
                    marker_color=fc_col,
                    marker_line_width=0,
                    hovertemplate=f"<b>%{{x}}</b><br>{fc}: %{{y:.1f}}<extra></extra>"
                ))
        # Add horizon + setup label above each bar
        for _, row in top12.iterrows():
            setup_sym = "🚀" if row["SetupType"] == "Breakout" else "↩"
            hz_sym    = _hz_sym.get(row.get("Horizon",""), "")
            fig_br.add_annotation(
                x=row["Ticker"], y=row["Score"] + 3,
                text=f"<b style='color:#ff8c00'>{hz_sym}</b>",
                showarrow=False,
                font=dict(size=11, color="#ff8c00"),
                bgcolor="rgba(0,0,0,0)"
            )
        fig_br.update_layout(
            barmode="stack",
            height=440,
            legend=dict(
                orientation="h",
                x=0, y=1.14,
                font=dict(size=10, color="#e8e8e8", family="IBM Plex Mono"),
                bgcolor="rgba(0,0,0,0)",
                bordercolor="#2a2a2a",
                borderwidth=1,
                itemclick="toggleothers",
            ),
            margin=dict(t=50, b=20, l=40, r=10),
            yaxis=dict(
                title=dict(text="SCORE", font=dict(color="#888", size=9)),
                range=[0, 125],
                gridcolor="#1a1a1a",
                tickfont=dict(color="#888", size=9),
                zeroline=False,
            ),
            xaxis=dict(
                tickfont=dict(color="#e8e8e8", size=11, family="IBM Plex Mono"),
                tickangle=0,
            ),
            plot_bgcolor="#000000",
            paper_bgcolor="#0a0a0a",
            font=dict(color="#e8e8e8", family="IBM Plex Mono", size=9),
            hoverlabel=dict(
                bgcolor="#1a1200", font_color="#ff8c00",
                font_family="IBM Plex Mono", font_size=11,
                bordercolor="#ff8c00"
            ),
        )
        st.plotly_chart(fig_br, use_container_width=True)

        # ── SEND TO ML PREDICTOR ──
        st.divider()
        st.markdown("""
<div style="background:#0a1520;border-top:2px solid #1e90ff;border-bottom:1px solid #1a2a3a;
     padding:8px 14px;font-family:'IBM Plex Mono',monospace;margin-bottom:10px;">
  <span style="color:#1e90ff;font-size:.77rem;font-weight:700;letter-spacing:.12em;">
    🧠 ML PRICE PREDICTOR
  </span>
  <span style="color:#555;font-size:.62rem;margin-left:16px;">
    Select stocks from scan results → run ML to predict closing price
  </span>
</div>""", unsafe_allow_html=True)

        ml_col1, ml_col2 = st.columns([3, 1])
        with ml_col1:
            all_tickers = df_out["Ticker"].tolist()
            default_ml  = all_tickers[:min(5, len(all_tickers))]
            ml_selected = st.multiselect(
                "Select stocks for ML prediction",
                options=all_tickers,
                default=default_ml,
                key="ml_ticker_select",
                format_func=lambda t: f"{t}  (Score: {df_out.loc[df_out['Ticker']==t,'Score'].values[0]:.0f}  {df_out.loc[df_out['Ticker']==t,'Horizon'].values[0]})" if t in df_out['Ticker'].values else t
            )
        with ml_col2:
            st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
            if st.button("🧠  OPEN ML PREDICTOR", key="goto_ml_btn",
                         use_container_width=True, type="primary"):
                if ml_selected:
                    # Package everything the ML page needs
                    st.session_state["ml_tickers"]       = ml_selected
                    st.session_state["ml_scan_params"]   = {
                        "interval":   "1d",
                        "days":       365,
                    }
                    # Pass raw OHLCV data for selected tickers
                    st.session_state["ml_raw_data"]      = {
                        sym: st.session_state.raw_data_cache[sym].copy()
                        for sym in ml_selected
                        if sym in st.session_state.raw_data_cache
                    }
                    # Pass score context for display
                    st.session_state["ml_score_context"] = df_out[
                        df_out["Ticker"].isin(ml_selected)
                    ][["Ticker","Score","SetupType","Horizon","Entry","Target","Stop","RR"]].to_dict("records")
                    try:
                        st.switch_page("pages/4_ML_Predictor.py")
                    except Exception:
                        st.info("Navigate to **ML Predictor** page in the sidebar.")
                else:
                    st.warning("Select at least one stock first.")


        # ── SECTOR HEATMAP ──
        if sector_returns:
            st.divider()
            st.markdown("### 🗂 SECTOR MOMENTUM  ·  5-Day Returns")
            sdf = pd.DataFrame(
                [(k, round(v*100,2)) for k,v in sorted(sector_returns.items(), key=lambda x:x[1], reverse=True)],
                columns=["Sector","5D Ret%"]
            )
            s_left, s_right = st.columns([1,2])
            with s_left:
                def sc_col(v):
                    if not isinstance(v,(int,float)): return ""
                    if v>1.5:  return "background-color:#1a3300;color:#00d084"
                    if v>0:    return "background-color:#1a2200;color:#b8e06a"
                    if v>-1.5: return "background-color:#2a1000;color:#ffb347"
                    return "background-color:#2a0000;color:#ff3b3b"
                st.dataframe(sdf.style.applymap(sc_col, subset=["5D Ret%"]),
                             use_container_width=True, hide_index=True)
            with s_right:
                fig_s = go.Figure(go.Bar(
                    x=sdf["Sector"], y=sdf["5D Ret%"],
                    marker_color=["#00d084" if v>=0 else "#ff3b3b" for v in sdf["5D Ret%"]],
                    text=[f"{v:+.1f}%" for v in sdf["5D Ret%"]],
                    textposition="outside", textfont=dict(size=9,color="#e8e8e8")
                ))
                fig_s.update_layout(height=240, plot_bgcolor="#000", paper_bgcolor="#000",
                    font=dict(color="#e8e8e8",family="IBM Plex Mono",size=9),
                    margin=dict(t=6,b=6,l=0,r=0),
                    yaxis=dict(gridcolor="#1a1a1a",zerolinecolor="#333"))
                st.plotly_chart(fig_s, use_container_width=True)

        # ── LEGEND ──
        with st.expander("◼ HOW TO READ THIS SCREENER  (click to expand)"):
            st.markdown("""
**RANKING:** Sorted by **CompositeRank = EMI × LiquidityScore × VolumeStability**

**5-HORIZON SYSTEM — all thresholds derived mathematically from ATR and volume σ:**

| Horizon | Trigger | Target | Entry Rule |
|---------|---------|--------|-----------|
| 🔴 IMMINENT BO | Price ≤ 0.25×ATR from trigger + Vol z-score ≥ 1.5σ | 0.75×ATR | Buy at market or limit just above base high |
| ⚡ INTRADAY | Same-session BO with vol ≥ 1.5× OR PB at EMA20 + RSI turning | 0.75×ATR | Enter today, trail stop same session |
| 📈 SWING 2-5D | Within 1×ATR of trigger, vol z-score ≥ 0 (above mean) | 2.0×ATR | Place limit order, confirm with volume |
| 📊 MID 5-14D | Within 3×ATR of trigger, setup forming | 3.5×ATR | Watchlist — wait for volume confirmation |
| 🏦 LONG 14-30D | > 3×ATR from trigger, base building | 5.0×ATR | Monitor only, do not enter yet |

**MATHEMATICAL BASIS (v6 — all signals threshold-free):**
- **Vol z-score (VolZ):** (today_vol − 20d_mean) / 20d_σ — dynamic, adapts to each stock
- **Vol BO Threshold (VolBOThr):** μ + 1.5σ (shown as × of average) — no hardcoded multiples
- **ATR%:** scored vs stock's own 60d ATR% percentile — high-vol stocks score vs themselves
- **RS normalisation (v6):** volatility-normalised alpha (z-score) via tanh squashing — replaces fixed ±2% band
- **Sector RS (v6):** sector alpha in σ-units via tanh — replaces fixed ±3% band
- **Trend (v6):** EMA9/EMA50 ratio percentile-ranked over 250d — continuous, no EMA step thresholds
- **Accel (v6):** EMA5−EMA20 velocity differential, percentile-ranked — leading momentum signal
- **VolCont (v6):** ATR5/ATR20 & range5/range20 percentile-ranked over 250d — fully adaptive
- **52W Position (v6):** price position in 52-week range, percentile-ranked — identifies true leaders
- **Market breadth (v6):** VIX-level + Nifty trend continuous scoring — no binary penalties

**10 FACTORS + BONUSES (100 pts base):**

| Factor | Max | Signal |
|--------|-----|--------|
| RS Nifty | 15 | Vol-normalised alpha vs Nifty, tanh-squashed (v6) |
| RS Sector | 10 | Stock α vs sector in σ-units, sector rotation acceleration |
| Volume Surge | 15 | z-score of today's vol vs 20d σ distribution |
| Pre-BO Accum | 10 | vol5/vol20 sigmoid(k=4, x0=1.3) — no hard thresholds |
| VolCont + RCI | 10 | ATR5/ATR20 + range5/range20 percentile-ranked over 250d (v6) |
| Coil Quality | 10 | Base tightness + flat resistance + price position |
| MA Structure | 10 | EMA9/EMA50 ratio percentile (250d) + alignment bonus (v6) |
| Proximity | 10 | ATR-normalised distance to trigger, exp decay |
| ATR% Rank | 5 | Stock's ATR% vs its own 60d percentile distribution |
| Candle | 5 | 8 bullish patterns (Engulfing, Hammer, MorningStar etc.) |
| **Bonuses** | +13 | Accel (4), VCVE (3), 52W Pos (3), Sweep (4), VWMA (3), Stability (2) |

**Market adjustment (v6):** Continuous VIX-level scoring (−8 to +2 pts) · Nifty trend adjustment (−8 to +4 pts)
""")


# ============================================================
# HISTORICAL BACKTEST — Signal Replay + Forward Returns
# ============================================================
# How it works:
#   1. Pick any past date (must have data in raw_data_cache)
#   2. We slice every stock's history up to that date
#   3. Run score_stock_dual() on that slice (same logic as live)
#   4. Look forward 1/2/3/5 days from entry close
#   5. Show returns per stock, ranked by signal score
#   6. Show aggregate stats: did high-score stocks beat low-score stocks?
# ============================================================

st.divider()
st.header("🔬 Backtest — Signal Replay on Historical Date")
st.caption("Select a past date → the screener rebuilds signals exactly as they would have appeared that day → shows actual forward returns.")

if not st.session_state.raw_data_cache:
    st.info("Run extraction first — backtest needs historical data in cache")
else:
    # ── find valid date range from cached data ──
    all_dates = set()
    for sym, df in st.session_state.raw_data_cache.items():
        if "time" in df.columns:
            for d in pd.to_datetime(df["time"]).dt.date:
                all_dates.add(d)

    all_dates = sorted(all_dates)

    # need at least 60 bars history before + 5 bars forward
    if len(all_dates) < 70:
        st.warning("Not enough historical bars in cache for backtest (need 70+)")
    else:
        # valid range: not in last 5 trading days (need forward returns)
        valid_dates = all_dates[60:-5]

        if not valid_dates:
            st.warning("Not enough data range for backtest")
        else:
            col_bt1, col_bt2, col_bt3 = st.columns([2, 1, 1])

            with col_bt1:
                bt_date = st.date_input(
                    "Select backtest date",
                    value=valid_dates[-1],
                    min_value=valid_dates[0],
                    max_value=valid_dates[-1],
                    key="bt_date"
                )

            with col_bt2:
                bt_topn = st.number_input(
                    "Top N signals to trade", min_value=3, max_value=20, value=10, key="bt_topn"
                )

            with col_bt3:
                bt_minscore = st.number_input(
                    "Min score threshold", min_value=0, max_value=80, value=40, key="bt_minscore"
                )

            run_bt = st.button("▶ Run Backtest", use_container_width=True, key="run_bt")

            if run_bt:
                bt_target = pd.Timestamp(bt_date)

                # FIX B-06: pre-compute Nifty returns AT the backtest date (no look-ahead)
                # Download full Nifty history once and slice to bt_date
                bt_nifty_r5  = nifty_r5    # fallback to today's values
                bt_nifty_r20 = nifty_r20
                try:
                    nifty_hist = yf.download("^NSEI", period="730d", interval="1d", progress=False)
                    if not nifty_hist.empty:
                        nc = nifty_hist["Close"].squeeze()
                        nc.index = pd.to_datetime(nc.index).tz_localize(None)
                        # find closest date <= bt_date in Nifty history
                        nc_slice = nc[nc.index.date <= bt_date]
                        if len(nc_slice) >= 21:
                            bt_nifty_r5  = float(nc_slice.iloc[-1] / nc_slice.iloc[-6]  - 1) if len(nc_slice) >= 6  else None
                            bt_nifty_r20 = float(nc_slice.iloc[-1] / nc_slice.iloc[-21] - 1)
                except Exception as _e:
                    st.warning(f"Could not compute historical Nifty RS — using today's values as fallback: {_e}")

                # ── STEP 1: slice each stock to backtest date ──
                bt_signals   = []
                skipped      = 0

                progress_bt  = st.progress(0)
                syms         = list(st.session_state.raw_data_cache.keys())

                for idx, sym in enumerate(syms):
                    df_full = st.session_state.raw_data_cache[sym].copy()
                    df_full["time"] = pd.to_datetime(df_full["time"])
                    df_full = df_full.sort_values("time").reset_index(drop=True)

                    # find index of backtest date
                    date_idx = df_full[df_full["time"].dt.date == bt_date].index
                    if len(date_idx) == 0:
                        skipped += 1
                        progress_bt.progress((idx+1)/len(syms))
                        continue

                    bar_idx = int(date_idx[0])

                    # need at least 60 bars before AND 5 bars after
                    if bar_idx < 60 or bar_idx + 5 >= len(df_full):
                        skipped += 1
                        progress_bt.progress((idx+1)/len(syms))
                        continue

                    # slice up to and including backtest date
                    df_slice = df_full.iloc[:bar_idx + 1].copy()

                    # simulate "live" bar = that day's actual OHLCV
                    bar      = df_full.iloc[bar_idx]
                    fake_live = {
                        "ltp":    float(bar["close"]),
                        "open":   float(bar["open"]),
                        "high":   float(bar["high"]),
                        "low":    float(bar["low"]),
                        "volume": float(bar["volume"]),
                    }

                    # FIX B-06: pass historical Nifty RS values for the backtest date
                    result = score_stock_dual(df_slice, fake_live, bt_nifty_r5, bt_nifty_r20, ticker=sym)

                    if result is None:
                        skipped += 1
                        progress_bt.progress((idx+1)/len(syms))
                        continue

                    if result["Score"] < bt_minscore:
                        progress_bt.progress((idx+1)/len(syms))
                        continue

                    # ── STEP 2: measure forward returns ──
                    entry_price = float(bar["close"])
                    if entry_price == 0:
                        skipped += 1
                        progress_bt.progress((idx+1)/len(syms))
                        continue

                    def fwd_return(n):
                        fwd_idx = bar_idx + n
                        if fwd_idx >= len(df_full):
                            return None
                        return round((df_full.iloc[fwd_idx]["close"] - entry_price) / entry_price * 100, 2)

                    r1 = fwd_return(1)
                    r2 = fwd_return(2)
                    r3 = fwd_return(3)
                    r5 = fwd_return(5)

                    # max drawdown in 5-bar window
                    fwd_window = df_full.iloc[bar_idx+1 : bar_idx+6]["low"]
                    max_dd = round((fwd_window.min() - entry_price) / entry_price * 100, 2) if len(fwd_window) > 0 else None

                    # max gain in 5-bar window
                    fwd_high   = df_full.iloc[bar_idx+1 : bar_idx+6]["high"]
                    max_gain   = round((fwd_high.max() - entry_price) / entry_price * 100, 2) if len(fwd_high) > 0 else None

                    bt_signals.append({
                        "Ticker":    sym,
                        "SetupType": result["SetupType"],
                        "Score":     result["Score"],
                        "EMI":       result["EMI"],
                        "CompositeRank": result["CompositeRank"],
                        "Entry":     round(entry_price, 2),
                        "R1d%":      r1,
                        "R2d%":      r2,
                        "R3d%":      r3,
                        "R5d%":      r5,
                        "MaxGain%":  max_gain,
                        "MaxDD%":    max_dd,
                        "Patterns":  result["Patterns"],
                        "RSI7":      result["RSI7"],
                        "VolRatio":  result["VolRatio"],
                        "RS":        result["RS"],
                        "Volume":    result["Volume"],
                        "MA_Struct": result["MA_Struct"],
                        "Candle":    result["Candle"],
                        "VCVE":      result["VCVE"],
                        "RCI":       result["RCI"],
                        "Sweep":     result["Sweep"],
                        "VWMA20_OK":   result["VWMA20_OK"],
                        "Stability": result["Stability"],
                    })

                    progress_bt.progress((idx+1)/len(syms))

                progress_bt.empty()

                if not bt_signals:
                    st.warning(f"No signals found on {bt_date} with score ≥ {bt_minscore}")
                else:
                    bt_df = pd.DataFrame(bt_signals).sort_values("CompositeRank", ascending=False).reset_index(drop=True)
                    if "Rank" in bt_df.columns:
                        bt_df.drop(columns=["Rank"], inplace=True)
                    bt_df.insert(0, "Rank", bt_df.index + 1)

                    # limit to top N
                    bt_top = bt_df.head(bt_topn)

                    st.success(f"✅ {len(bt_df)} signals on {bt_date} | showing top {bt_topn} | {skipped} stocks skipped (insufficient data)")

                    # ── AGGREGATE STATS ──
                    st.subheader("📈 Aggregate Performance")

                    def safe_mean(series):
                        s = series.dropna()
                        return round(s.mean(), 2) if len(s) > 0 else None

                    r5_all   = bt_top["R5d%"].dropna()
                    r1_all   = bt_top["R1d%"].dropna()
                    win_rate = round((r5_all > 0).sum() / len(r5_all) * 100, 1) if len(r5_all) > 0 else None
                    avg_r5   = safe_mean(bt_top["R5d%"])
                    avg_r1   = safe_mean(bt_top["R1d%"])
                    avg_dd   = safe_mean(bt_top["MaxDD%"])
                    avg_gain = safe_mean(bt_top["MaxGain%"])
                    best     = bt_top.loc[bt_top["R5d%"].idxmax(), "Ticker"] if not r5_all.empty else "—"
                    worst    = bt_top.loc[bt_top["R5d%"].idxmin(), "Ticker"] if not r5_all.empty else "—"

                    c1,c2,c3,c4,c5,c6 = st.columns(6)
                    c1.metric("Win Rate (5d)",   f"{win_rate}%"  if win_rate is not None else "—")
                    c2.metric("Avg 5d Return",   f"{avg_r5}%"   if avg_r5   is not None else "—")
                    c3.metric("Avg 1d Return",   f"{avg_r1}%"   if avg_r1   is not None else "—")
                    c4.metric("Avg Max Gain",    f"{avg_gain}%" if avg_gain is not None else "—")
                    c5.metric("Avg Max Drawdown",f"{avg_dd}%"   if avg_dd   is not None else "—")
                    c6.metric("Best / Worst",    f"{best} / {worst}")

                    # ── SCORE vs RETURN SCATTER ──
                    st.subheader("📊 Does Score Predict Return? (Score vs 5d Return)")
                    valid_scatter = bt_df.dropna(subset=["R5d%"])
                    if len(valid_scatter) >= 3:
                        fig_sc = go.Figure()

                        for stype, color in [("Breakout","#2196F3"),("Pullback","#FF9800")]:
                            sub = valid_scatter[valid_scatter["SetupType"]==stype]
                            if not sub.empty:
                                fig_sc.add_trace(go.Scatter(
                                    x=sub["Score"], y=sub["R5d%"],
                                    mode="markers+text",
                                    text=sub["Ticker"],
                                    textposition="top center",
                                    textfont=dict(size=9),
                                    marker=dict(size=10, color=color, opacity=0.85),
                                    name=stype
                                ))

                        # add trend line
                        from numpy.polynomial import polynomial as P
                        x_vals = valid_scatter["Score"].values
                        y_vals = valid_scatter["R5d%"].values
                        if len(x_vals) >= 3:
                            coefs = np.polyfit(x_vals, y_vals, 1)
                            x_line= np.linspace(x_vals.min(), x_vals.max(), 50)
                            y_line= np.polyval(coefs, x_line)
                            fig_sc.add_trace(go.Scatter(
                                x=x_line, y=y_line,
                                mode="lines",
                                line=dict(color="white", dash="dot", width=1),
                                name="Trend"
                            ))

                        fig_sc.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
                        fig_sc.update_layout(
                            height=420,
                            xaxis_title="Signal Score",
                            yaxis_title="5-Day Return %",
                            plot_bgcolor="#000000",
                            paper_bgcolor="#0a0a0a",
                            font_color="#e8e8e8",
                            margin=dict(t=20, b=20)
                        )
                        st.plotly_chart(fig_sc, use_container_width=True)

                        # correlation stat
                        corr = valid_scatter["Score"].corr(valid_scatter["R5d%"])
                        if corr > 0.3:
                            st.success(f"✅ Correlation (Score → 5d Return): **{corr:.3f}** — Score has real predictive power on this date")
                        elif corr > 0:
                            st.info(f"ℹ️ Correlation: **{corr:.3f}** — weak positive relationship")
                        else:
                            st.warning(f"⚠️ Correlation: **{corr:.3f}** — Score did not predict returns on this date")

                    # ── RETURN WATERFALL BY STOCK ──
                    st.subheader(f"📋 Top {bt_topn} Signals — Actual Forward Returns")

                    def color_ret(val):
                        if not isinstance(val, (int, float)) or pd.isna(val): return ""
                        if val >= 3:   return "background-color:#0d2a00;color:#00ff88;font-weight:700"
                        if val >= 1:   return "background-color:#0a1f00;color:#00d084"
                        if val >= 0:   return "background-color:#0d1a0a;color:#7ec87a"
                        if val >= -1:  return "background-color:#2a0000;color:#ff8888"
                        return            "background-color:#1f0000;color:#ff3b3b;font-weight:700"

                    def color_score_bt(val):
                        if not isinstance(val, (int, float)): return ""
                        if val >= 70: return "background-color:#0d2200;color:#00ff88;font-weight:700"
                        if val >= 55: return "background-color:#1a2200;color:#b8e06a"
                        if val >= 40: return "background-color:#2a1800;color:#ffb347"
                        return "color:#555555"

                    styled_bt = (
                        bt_top.style
                        .applymap(color_ret,      subset=["R1d%","R2d%","R3d%","R5d%","MaxGain%","MaxDD%"])
                        .applymap(color_score_bt, subset=["Score"])
                    )
                    st.dataframe(styled_bt, use_container_width=True, hide_index=True)

                    # ── EQUITY CURVE: equal-weight portfolio of top-N signals ──
                    st.subheader("📈 Equal-Weight Portfolio Equity Curve")
                    day_cols = ["R1d%","R2d%","R3d%","R5d%"]
                    valid_eq = bt_top[day_cols].dropna(how="all")
                    if len(valid_eq) >= 2:
                        # average return across all stocks at each horizon
                        port_returns = valid_eq.mean()
                        days_map     = {"R1d%":1,"R2d%":2,"R3d%":3,"R5d%":5}
                        x_days = [days_map[c] for c in port_returns.index]
                        y_cum  = port_returns.values

                        fig_eq = go.Figure()
                        fig_eq.add_trace(go.Scatter(
                            x=[0]+x_days, y=[0]+list(y_cum),
                            mode="lines+markers",
                            line=dict(color="#4CAF50", width=2),
                            fill="tozeroy",
                            fillcolor="rgba(76,175,80,0.15)",
                            name="Portfolio"
                        ))
                        fig_eq.add_hline(y=0, line_dash="dash", line_color="gray")
                        fig_eq.update_layout(
                            height=300,
                            xaxis=dict(title="Days after signal", tickvals=[0,1,2,3,5]),
                            yaxis_title="Avg Return %",
                            plot_bgcolor="#000000",
                            paper_bgcolor="#0a0a0a",
                            font_color="#e8e8e8",
                            margin=dict(t=10,b=20)
                        )
                        st.plotly_chart(fig_eq, use_container_width=True)

                    # ── SCORE QUINTILE TABLE ──
                    st.subheader("🔢 Does Higher Score = Higher Return? (Quintile Analysis)")
                    valid_q = bt_df.dropna(subset=["R5d%"]).copy()
                    if len(valid_q) >= 10:
                        valid_q["Quintile"] = pd.qcut(valid_q["Score"], 5,
                                                      labels=["Q1 (Lowest)","Q2","Q3","Q4","Q5 (Highest)"])
                        qt = (valid_q.groupby("Quintile", observed=True)["R5d%"]
                              .agg(Count="count", AvgReturn="mean", WinRate=lambda x: (x>0).mean()*100)
                              .round(2).reset_index())
                        qt.columns = ["Score Quintile","Count","Avg 5d Return %","Win Rate %"]

                        def qt_color(val):
                            if not isinstance(val,(int,float)): return ""
                            if val > 1:  return "background-color:#0d2200;color:#00d084;font-weight:700"
                            if val > 0:  return "background-color:#0a1a00;color:#7ec87a"
                            return "background-color:#1f0000;color:#ff3b3b"

                        st.dataframe(
                            qt.style.applymap(qt_color, subset=["Avg 5d Return %","Win Rate %"]),
                            use_container_width=True, hide_index=True
                        )
                        st.caption("Q5 should have higher returns than Q1 — if it does, the model has real edge")
                    else:
                        st.info("Need at least 10 signals for quintile analysis")

                    # ── TOP 10 vs BOTTOM 10 COMPARISON ──
                    st.subheader("⚖️ Top 10 vs Bottom 10 Signals — Head to Head")
                    valid_cmp = bt_df.dropna(subset=["R5d%"]).copy()
                    if len(valid_cmp) >= 6:
                        top10_cmp    = valid_cmp.head(min(10, len(valid_cmp)//2))
                        bottom10_cmp = valid_cmp.tail(min(10, len(valid_cmp)//2))

                        # side-by-side metrics
                        mc1, mc2 = st.columns(2)
                        with mc1:
                            st.markdown("#### 🟢 Top Signals (Highest Score)")
                            t_win  = round((top10_cmp["R5d%"] > 0).mean() * 100, 1)
                            t_avg  = round(top10_cmp["R5d%"].mean(), 2)
                            t_gain = round(top10_cmp["MaxGain%"].mean(), 2)
                            t_dd   = round(top10_cmp["MaxDD%"].mean(), 2)
                            st.metric("Avg 5d Return",  f"{t_avg}%")
                            st.metric("Win Rate",       f"{t_win}%")
                            st.metric("Avg Max Gain",   f"{t_gain}%")
                            st.metric("Avg Max DD",     f"{t_dd}%")

                        with mc2:
                            st.markdown("#### 🔴 Bottom Signals (Lowest Score)")
                            b_win  = round((bottom10_cmp["R5d%"] > 0).mean() * 100, 1)
                            b_avg  = round(bottom10_cmp["R5d%"].mean(), 2)
                            b_gain = round(bottom10_cmp["MaxGain%"].mean(), 2)
                            b_dd   = round(bottom10_cmp["MaxDD%"].mean(), 2)
                            st.metric("Avg 5d Return",  f"{b_avg}%",  delta=f"{round(b_avg - t_avg, 2)}% vs top")
                            st.metric("Win Rate",       f"{b_win}%",  delta=f"{round(b_win - t_win, 1)}% vs top")
                            st.metric("Avg Max Gain",   f"{b_gain}%", delta=f"{round(b_gain - t_gain, 2)}% vs top")
                            st.metric("Avg Max DD",     f"{b_dd}%",   delta=f"{round(b_dd - t_dd, 2)}% vs top")

                        # bar chart comparing 1d/2d/3d/5d avg returns
                        horizons   = ["R1d%","R2d%","R3d%","R5d%"]
                        h_labels   = ["1 Day","2 Day","3 Day","5 Day"]
                        top_means  = [round(top10_cmp[h].mean(), 2)    for h in horizons]
                        bot_means  = [round(bottom10_cmp[h].mean(), 2) for h in horizons]

                        fig_cmp = go.Figure()
                        fig_cmp.add_trace(go.Bar(
                            name="Top 10",    x=h_labels, y=top_means,
                            marker_color="#4CAF50", text=[f"{v}%" for v in top_means],
                            textposition="outside"
                        ))
                        fig_cmp.add_trace(go.Bar(
                            name="Bottom 10", x=h_labels, y=bot_means,
                            marker_color="#F44336", text=[f"{v}%" for v in bot_means],
                            textposition="outside"
                        ))
                        fig_cmp.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
                        fig_cmp.update_layout(
                            barmode="group", height=360,
                            yaxis_title="Avg Return %",
                            plot_bgcolor="#000000", paper_bgcolor="#0a0a0a",
                            font_color="#e8e8e8", margin=dict(t=20, b=20),
                            legend=dict(orientation="h", y=1.1)
                        )
                        st.plotly_chart(fig_cmp, use_container_width=True)

                        spread_5d = round(t_avg - b_avg, 2)
                        if spread_5d > 1.5:
                            st.success(f"✅ Spread between top and bottom: **{spread_5d}%** — strong model discrimination")
                        elif spread_5d > 0:
                            st.info(f"ℹ️ Spread: **{spread_5d}%** — some discrimination, model is directionally correct")
                        else:
                            st.warning(f"⚠️ Spread: **{spread_5d}%** — model is NOT separating winners from losers on this date")

                        # detailed table for bottom 10
                        st.markdown("#### Bottom 10 Signal Detail")
                        st.dataframe(
                            bottom10_cmp.style
                            .applymap(color_ret,      subset=["R1d%","R2d%","R3d%","R5d%","MaxGain%","MaxDD%"])
                            .applymap(color_score_bt, subset=["Score"]),
                            use_container_width=True, hide_index=True
                        )
                    else:
                        st.info("Need at least 6 signals to compare top vs bottom")


# ============================================================
# BLOOMBERG-STYLE CHART DASHBOARD
# ============================================================
# Clicking any row in the screener table opens this panel.
# Features:
#   - Candlestick + Volume subplot
#   - EMA9 / EMA20 / EMA50 overlaid
#   - ATR bands
#   - BUY marker: first bar after signal date that closes green
#     with above-avg volume AND above EMA9
#   - SELL marker: first bar where close < EMA20 OR gain > 8%
#     OR loss > -3% (trailing stop)
#   - Bloomberg dark colour scheme throughout
# ============================================================

st.divider()
st.markdown("## ▶ CHART TERMINAL")

symbols_list = sorted(st.session_state.raw_data_cache.keys())
if not symbols_list:
    st.info("Run extraction first")
    st.stop()

# ── CHART BUILDER ──

def bb_chart(sym, df_raw, live, signal_date=None):
    """
    Build a Bloomberg-styled candlestick chart with:
    - EMA9 (amber), EMA20 (white), EMA50 (blue)
    - ATR envelope
    - Volume bars (green/red)
    - BUY / SELL annotations derived from rule-based logic
    Returns plotly Figure
    """
    df = df_raw.copy()
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").tail(120)   # last 120 bars for clarity

    # apply live patch on last bar
    ltp = live.get("ltp")
    if ltp:
        df.at[df.index[-1], "close"] = ltp
        df.at[df.index[-1], "high"]  = max(float(df["high"].iloc[-1]), ltp)
        df.at[df.index[-1], "low"]   = min(float(df["low"].iloc[-1]),  ltp)
    if live.get("volume"):
        df.at[df.index[-1], "volume"] = live["volume"]

    # ── indicators ──
    df["e9"]  = df["close"].ewm(span=9,  adjust=False).mean()
    df["e20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["e50"] = df["close"].ewm(span=50, adjust=False).mean()

    # ATR(14) envelope ±1 ATR from EMA20
    hl  = df["high"] - df["low"]
    hc  = (df["high"] - df["close"].shift(1)).abs()
    lc  = (df["low"]  - df["close"].shift(1)).abs()
    tr  = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df["atr14"] = tr.ewm(span=14, adjust=False).mean()
    df["atr_up"]= df["e20"] + df["atr14"]
    df["atr_dn"]= df["e20"] - df["atr14"]

    vol_avg20 = df["volume"].rolling(20).mean()

    # ── BUY / SELL SIGNAL MARKS ──
    buy_x,  buy_y  = [], []
    sell_x, sell_y = [], []

    in_trade    = False
    entry_price = 0.0
    entry_idx   = -1

    for i in range(2, len(df)):
        row  = df.iloc[i]
        prev = df.iloc[i-1]
        c    = float(row["close"]);  o = float(row["open"])
        va   = float(vol_avg20.iloc[i]) if pd.notna(vol_avg20.iloc[i]) else 0
        vol  = float(row["volume"]) if pd.notna(row["volume"]) else 0

        if not in_trade:
            # BUY CONDITION:
            #   green candle + close > EMA9 + volume > 1.2x avg + EMA9 > EMA20
            if (c > o and c > float(row["e9"]) and
                float(row["e9"]) > float(row["e20"]) and
                vol > va * 1.2 and pd.notna(row["e20"])):
                buy_x.append(row["time"])
                buy_y.append(float(row["low"]) * 0.995)
                in_trade    = True
                entry_price = c
                entry_idx   = i
        else:
            gain = (c - entry_price) / entry_price * 100 if entry_price != 0 else 0.0
            # SELL CONDITIONS (first triggered):
            #   1. Close drops below EMA20
            #   2. Gain > 8% (take profit)
            #   3. Loss < -3% (stop loss)
            #   4. Close < EMA9 after being in trade 2+ bars
            if (c < float(row["e20"]) or
                gain >  8.0 or
                gain < -3.0 or
                (i - entry_idx >= 2 and c < float(row["e9"]))):
                sell_x.append(row["time"])
                sell_y.append(float(row["high"]) * 1.005)
                in_trade    = False
                entry_price = 0.0

    # ── MARK SIGNAL DATE if provided ──
    sig_x, sig_y = [], []
    if signal_date is not None:
        sig_ts = pd.Timestamp(signal_date)
        sig_row = df[df["time"].dt.date == signal_date]
        if not sig_row.empty:
            sig_x.append(sig_row.iloc[0]["time"])
            sig_y.append(float(sig_row.iloc[0]["low"]) * 0.992)

    # ── BUILD FIGURE ──
    BB = dict(
        bg    = "#000000",
        paper = "#000000",
        amber = "#ff8c00",
        amber2= "#ffb347",
        green = "#00d084",
        red   = "#ff3b3b",
        blue  = "#1e90ff",
        white = "#e8e8e8",
        muted = "#444444",
        grid  = "#1a1a1a",
    )

    from plotly.subplots import make_subplots
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.75, 0.25],
        vertical_spacing=0.02,
    )

    # ── candles ──
    fig.add_trace(go.Candlestick(
        x=df["time"],
        open=df["open"], high=df["high"],
        low=df["low"],   close=df["close"],
        increasing=dict(line=dict(color=BB["green"], width=1), fillcolor=BB["green"]),
        decreasing=dict(line=dict(color=BB["red"],   width=1), fillcolor=BB["red"]),
        name="Price",
        showlegend=False
    ), row=1, col=1)

    # ── ATR envelope ──
    fig.add_trace(go.Scatter(
        x=df["time"], y=df["atr_up"],
        line=dict(color=BB["muted"], width=0.8, dash="dot"),
        name="ATR+", showlegend=False, opacity=0.6
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df["time"], y=df["atr_dn"],
        line=dict(color=BB["muted"], width=0.8, dash="dot"),
        fill="tonexty", fillcolor="rgba(255,140,0,0.04)",
        name="ATR-", showlegend=False, opacity=0.6
    ), row=1, col=1)

    # ── EMAs ──
    fig.add_trace(go.Scatter(
        x=df["time"], y=df["e9"],
        line=dict(color=BB["amber"], width=1.2),
        name="EMA9"
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df["time"], y=df["e20"],
        line=dict(color=BB["white"], width=1.2),
        name="EMA20"
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df["time"], y=df["e50"],
        line=dict(color=BB["blue"], width=1.2),
        name="EMA50"
    ), row=1, col=1)

    # ── BUY markers ──
    if buy_x:
        fig.add_trace(go.Scatter(
            x=buy_x, y=buy_y,
            mode="markers+text",
            marker=dict(symbol="triangle-up", size=12,
                        color=BB["green"], line=dict(color="#000", width=1)),
            text=["B"]*len(buy_x), textposition="bottom center",
            textfont=dict(color=BB["green"], size=8, family="IBM Plex Mono"),
            name="BUY", showlegend=True
        ), row=1, col=1)

    # ── SELL markers ──
    if sell_x:
        fig.add_trace(go.Scatter(
            x=sell_x, y=sell_y,
            mode="markers+text",
            marker=dict(symbol="triangle-down", size=12,
                        color=BB["red"], line=dict(color="#000", width=1)),
            text=["S"]*len(sell_x), textposition="top center",
            textfont=dict(color=BB["red"], size=8, family="IBM Plex Mono"),
            name="SELL", showlegend=True
        ), row=1, col=1)

    # ── SIGNAL DATE marker ──
    if sig_x:
        fig.add_trace(go.Scatter(
            x=sig_x, y=sig_y,
            mode="markers+text",
            marker=dict(symbol="diamond", size=14,
                        color=BB["amber"], line=dict(color="#000", width=1)),
            text=["★"], textposition="bottom center",
            textfont=dict(color=BB["amber"], size=10),
            name="Signal Date"
        ), row=1, col=1)

    # ── Volume bars ──
    vol_colors = [BB["green"] if float(df["close"].iloc[i]) >= float(df["open"].iloc[i])
                  else BB["red"] for i in range(len(df))]
    fig.add_trace(go.Bar(
        x=df["time"], y=df["volume"],
        marker_color=vol_colors, marker_opacity=0.7,
        name="Volume", showlegend=False
    ), row=2, col=1)

    # ── vol avg line ──
    fig.add_trace(go.Scatter(
        x=df["time"], y=vol_avg20,
        line=dict(color=BB["amber"], width=1, dash="dot"),
        name="Vol MA20", showlegend=False
    ), row=2, col=1)

    # ── live price line ──
    if ltp:
        fig.add_hline(
            y=ltp, row=1, col=1,
            line=dict(color=BB["amber2"], width=0.8, dash="dash"),
            annotation_text=f"LTP {ltp:.2f}",
            annotation_font=dict(color=BB["amber2"], size=9, family="IBM Plex Mono"),
            annotation_position="right"
        )

    # ── LAYOUT ──
    axis_style = dict(
        gridcolor=BB["grid"], gridwidth=1,
        zerolinecolor=BB["grid"],
        tickfont=dict(color=BB["muted"], size=9, family="IBM Plex Mono"),
        showgrid=True,
    )
    fig.update_layout(
        height=580,
        plot_bgcolor=BB["bg"],
        paper_bgcolor=BB["paper"],
        font=dict(family="IBM Plex Mono", color=BB["white"], size=9),
        xaxis_rangeslider_visible=False,
        margin=dict(l=8, r=8, t=30, b=8),
        legend=dict(
            orientation="h", y=1.02, x=0,
            font=dict(size=8, family="IBM Plex Mono", color=BB["white"]),
            bgcolor="rgba(0,0,0,0.5)",
            bordercolor=BB["muted"], borderwidth=1
        ),
        title=dict(
            text=f"<b>{sym}</b>  ·  DAILY  ·  EMA 9/20/50  ·  ATR BANDS",
            font=dict(size=10, color=BB["amber"], family="IBM Plex Mono"),
            x=0.01
        )
    )
    fig.update_xaxes(axis_style)
    fig.update_yaxes(axis_style)
    fig.update_yaxes(title_text="PRICE", row=1, col=1,
                     title_font=dict(size=8, color=BB["muted"]))
    fig.update_yaxes(title_text="VOL", row=2, col=1,
                     title_font=dict(size=8, color=BB["muted"]))

    return fig


# ── SCREENER TABLE + CLICK-TO-CHART ──

if "chart_sym" not in st.session_state:
    st.session_state.chart_sym = symbols_list[0] if symbols_list else None

# re-run screener to get display df (use cached result if available)
if st.session_state.raw_data_cache:
    q_rows = []
    for sym, df_raw in st.session_state.raw_data_cache.items():
        live   = get_live_bar(sym)
        result = get_cached_score(sym, df_raw, live, nifty_r5, nifty_r20)   # PERF: cached
        if result is None: continue
        ltp_v = live.get("ltp") or float(df_raw["close"].iloc[-1])
        q_rows.append({
            "Ticker":    sym,
            "Setup":     result["SetupType"],
            "Score":     result["Score"],
            "LTP":       round(ltp_v, 2),
            "RSI7":      result["RSI7"],
            "VolRatio":  result["VolRatio"],
            "RS_Nifty":  result["RS_vs_Nifty"],
            "Patterns":  result["Patterns"],
        })

    if q_rows:
        q_df = pd.DataFrame(q_rows).sort_values("Score", ascending=False).reset_index(drop=True)

        col_tbl, col_chart = st.columns([1, 2.8])

        with col_tbl:
            st.markdown("##### SELECT A STOCK TO CHART")
            st.caption("Click any row → chart updates instantly")

            # Selectbox keyed by ticker string — value is directly the ticker
            ticker_opts = q_df["Ticker"].tolist()
            score_map   = dict(zip(q_df["Ticker"], q_df["Score"]))
            setup_map   = dict(zip(q_df["Ticker"], q_df["Setup"]))

            # default to first ticker if chart_sym not in current list
            default_sym = st.session_state.get("chart_sym", ticker_opts[0])
            if default_sym not in ticker_opts:
                default_sym = ticker_opts[0]
            default_idx = ticker_opts.index(default_sym)

            chosen = st.selectbox(
                "Stock",
                options=ticker_opts,
                index=default_idx,
                format_func=lambda t: f"{t}  [{setup_map.get(t,'?')[0]}]  {score_map.get(t,0)}",
                key="chart_picker",
                label_visibility="collapsed"
            )
            # immediately write chosen to session_state so col_chart gets the right value
            st.session_state.chart_sym = chosen

            # mini score table
            def score_bg(val):
                if not isinstance(val,(int,float)): return ""
                if val >= 70: return "background-color:#1a3300;color:#00d084"
                if val >= 55: return "background-color:#1a2200;color:#7dba34"
                if val >= 40: return "background-color:#2a1a00;color:#ffb347"
                return "color:#666"
            def setup_bg(val):
                return ("background-color:#001a2a;color:#1e90ff" if val=="Breakout"
                        else "background-color:#1a001a;color:#cc88ff")

            styled_q = (q_df.style
                        .applymap(score_bg,  subset=["Score"])
                        .applymap(setup_bg,  subset=["Setup"]))
            st.dataframe(styled_q, use_container_width=True, hide_index=True, height=480)

        with col_chart:
            sym = chosen   # use value from this run, not stale session_state
            if sym and sym in st.session_state.raw_data_cache:
                df_raw = st.session_state.raw_data_cache[sym]
                live   = get_live_bar(sym)
                result = get_cached_score(sym, df_raw, live, nifty_r5, nifty_r20)   # PERF: cached

                # ── TICKER HEADER ──
                ltp_v   = live.get("ltp") or float(df_raw["close"].iloc[-1])
                prev_c  = float(df_raw["close"].iloc[-2]) if len(df_raw) >= 2 else ltp_v
                chg     = ltp_v - prev_c
                chg_pct = chg / prev_c * 100 if prev_c else 0
                chg_col = "#00d084" if chg >= 0 else "#ff3b3b"
                arrow   = "▲" if chg >= 0 else "▼"

                st.markdown(f"""
<div style="display:flex;align-items:baseline;gap:16px;
            border-bottom:1px solid #2a2a2a;padding-bottom:6px;margin-bottom:8px;">
  <span style="font-family:'IBM Plex Mono';font-size:1.3rem;
               font-weight:600;color:#ff8c00;letter-spacing:0.12em;">{sym}</span>
  <span style="font-family:'IBM Plex Mono';font-size:1.1rem;
               font-weight:500;color:#e8e8e8;">₹{ltp_v:.2f}</span>
  <span style="font-family:'IBM Plex Mono';font-size:0.85rem;color:{chg_col};">
    {arrow} {chg:+.2f} ({chg_pct:+.2f}%)
  </span>
  <span style="font-family:'IBM Plex Mono';font-size:0.72rem;color:#666;margin-left:auto;">
    H: ₹{live.get('high', '—')}  L: ₹{live.get('low', '—')}  
    VOL: {(f"{int(live['volume']):,}") if live.get('volume') else '—'}
  </span>
</div>
""", unsafe_allow_html=True)

                if result:
                    r1c, r2c, r3c, r4c, r5c, r6c = st.columns(6)
                    for col_m, label, val in [
                        (r1c, "SCORE",   result["Score"]),
                        (r2c, "SETUP",   result["SetupType"]),
                        (r3c, "RSI(7)",  result["RSI7"]),
                        (r4c, "VOL×",    result["VolRatio"]),
                        (r5c, "RS%",     result["RS_vs_Nifty"]),
                        (r6c, "PATTERN", result["Patterns"][:12] if result["Patterns"] else "—"),
                    ]:
                        col_m.metric(label, val)

                fig = bb_chart(sym, df_raw, live)
                st.plotly_chart(fig, use_container_width=True, config={
                    "displayModeBar": True,
                    "modeBarButtonsToRemove": ["lasso2d","select2d"],
                    "displaylogo": False
                })

# ── NEWS TICKER ──
st.divider()
st.markdown("## ◼ MARKET INTEL")

def parse_news_sorted(url):
    feed = feedparser.parse(url)
    rows = []
    for e in feed.entries:
        try:    published = pd.to_datetime(e.published)
        except: published = pd.Timestamp.now()
        rows.append({"title": e.title, "link": e.link, "time": published})
    if not rows: return []
    return pd.DataFrame(rows).sort_values("time", ascending=False).to_dict("records")

nc1, nc2 = st.columns(2)
with nc1:
    sym_news = st.session_state.get("chart_sym", "")
    if sym_news:
        st.markdown(f"##### {sym_news} — NEWS")
        stock_news = parse_news_sorted(
            f"https://news.google.com/rss/search?q={sym_news}%20NSE%20stock&hl=en-IN&gl=IN&ceid=IN:en"
        )
        if stock_news:
            for n in stock_news[:6]:
                t = n["time"].strftime("%d %b %H:%M")
                st.markdown(
                    f'<div style="font-family:IBM Plex Mono;font-size:0.68rem;'
                    f'border-left:2px solid #ff8c00;padding:3px 8px;margin:3px 0;'
                    f'color:#aaa;">'
                    f'<span style="color:#ff8c00">{t}</span> — '
                    f'<a href="{n["link"]}" target="_blank" style="color:#e8e8e8;'
                    f'text-decoration:none;">{n["title"][:90]}</a></div>',
                    unsafe_allow_html=True
                )
        else:
            st.caption("No recent news")

with nc2:
    st.markdown("##### MARKET — NEWS")
    market_news = parse_news_sorted(
        "https://news.google.com/rss/search?q=Indian+stock+market+NSE&hl=en-IN&gl=IN&ceid=IN:en"
    )
    if market_news:
        for n in market_news[:6]:
            t = n["time"].strftime("%d %b %H:%M")
            st.markdown(
                f'<div style="font-family:IBM Plex Mono;font-size:0.68rem;'
                f'border-left:2px solid #444;padding:3px 8px;margin:3px 0;color:#aaa;">'
                f'<span style="color:#666">{t}</span> — '
                f'<a href="{n["link"]}" target="_blank" style="color:#e8e8e8;'
                f'text-decoration:none;">{n["title"][:90]}</a></div>',
                unsafe_allow_html=True
            )
    else:
        st.caption("No market news")
