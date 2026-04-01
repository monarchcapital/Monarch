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
# NSE BHAV COPY — DELIVERY VOLUME FETCH
# Delivery % = delivery_qty / traded_qty.
# High delivery (>50%) = informed money holding overnight.
# Low delivery (<20%) = intraday speculation, no conviction.
# Fetched once per session from NSE's public Bhav Copy CSV.
# Cached in session_state for 4 hours so it survives reruns.
# Falls back to empty dict silently — delivery is supplementary.
# ============================================================
_BHAV_CACHE_TTL = 4 * 3600   # 4 hours

@st.cache_data(ttl=_BHAV_CACHE_TTL)
def _fetch_nse_delivery_pct() -> dict:
    """
    Fetches NSE CM Bhav Copy for the most recent trading day and returns
    a dict {SYMBOL: delivery_pct (0-100)} for all EQ series stocks.
    Tries today first, then walks back up to 5 days to handle weekends/holidays.
    Returns {} on any failure — caller must treat missing keys as None.
    """
    headers_nse = {
        "User-Agent": "Mozilla/5.0",
        "Accept":     "text/csv,application/csv",
        "Referer":    "https://www.nseindia.com/"
    }
    for _offset in range(5):
        _d = datetime.now() - timedelta(days=_offset)
        if _d.weekday() >= 5:   # skip Saturday/Sunday
            continue
        _ds = _d.strftime("%d%m%Y")
        _url = f"https://archives.nseindia.com/products/content/sec_bhavdata_full_{_ds}.csv"
        try:
            _r = requests.get(_url, headers=headers_nse, timeout=12)
            if _r.status_code != 200:
                continue
            from io import StringIO
            _df = pd.read_csv(StringIO(_r.text))
            # Normalise column names (NSE changes spacing occasionally)
            _df.columns = [c.strip() for c in _df.columns]
            # Filter EQ series only
            if "SERIES" in _df.columns:
                _df = _df[_df["SERIES"].str.strip() == "EQ"]
            # Extract delivery %
            _sym_col = next((c for c in _df.columns if "SYMBOL" in c.upper()), None)
            _trd_col = next((c for c in _df.columns if "TRDQTY" in c.upper() or "TTL_TRD_QNTY" in c.upper()), None)
            _del_col = next((c for c in _df.columns if "DELIV" in c.upper() and "QTY" in c.upper()), None)
            if not (_sym_col and _trd_col and _del_col):
                continue
            _df[_trd_col] = pd.to_numeric(_df[_trd_col].astype(str).str.replace(",",""), errors="coerce")
            _df[_del_col] = pd.to_numeric(_df[_del_col].astype(str).str.replace(",",""), errors="coerce")
            _df = _df.dropna(subset=[_trd_col, _del_col])
            _df = _df[_df[_trd_col] > 0]
            _df["_del_pct"] = (_df[_del_col] / _df[_trd_col] * 100).clip(0, 100)
            return dict(zip(_df[_sym_col].str.strip(), _df["_del_pct"].round(1)))
        except Exception:
            continue
    return {}


# ============================================================
# EVENT CALENDAR — NSE CORPORATE ACTIONS + RESULTS DATES
# ============================================================
# Fetched once per day. Returns {SYMBOL: nearest_event_date_str} for any
# stock that has a board meeting, results, dividend, or AGM within ±5 days.
# Signals within EVENT_BLACKOUT_DAYS of a corporate event are flagged with
# "EventRisk" in the score output and get a soft penalty.
# Non-F&O stocks have no expiry risk but may still have result/AGM risk.
# ============================================================
_EVENT_CACHE_TTL = 24 * 3600   # 1 day

@st.cache_data(ttl=_EVENT_CACHE_TTL)
def _fetch_nse_event_calendar() -> dict:
    """
    Returns {SYMBOL: (event_label, days_away)} for the nearest upcoming event.
    days_away < 0 = event already passed (within lookback window).
    Returns {} on any failure — caller treats missing keys as no event.
    Priority: Results > Board Meeting > AGM > Dividend Ex-Date.
    """
    _headers_nse = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json",
        "Referer": "https://www.nseindia.com/"
    }
    _event_map = {}
    try:
        _r = requests.get(
            "https://www.nseindia.com/api/event-calendar",
            headers=_headers_nse, timeout=12
        )
        if _r.status_code != 200:
            return {}
        _data = _r.json()
        _today = datetime.now().date()
        _priority = {"Results": 0, "Board Meeting": 1, "AGM": 2, "Dividend": 3}
        for _entry in _data:
            _sym = str(_entry.get("symbol", "")).strip().upper()
            _purpose = str(_entry.get("purpose", "")).strip()
            _date_str = str(_entry.get("date", "")).strip()
            if not (_sym and _date_str):
                continue
            try:
                _ev_date = datetime.strptime(_date_str, "%d-%b-%Y").date()
            except Exception:
                try:
                    _ev_date = datetime.strptime(_date_str, "%Y-%m-%d").date()
                except Exception:
                    continue
            _days = (_ev_date - _today).days
            # Track only events within ±10 days
            if abs(_days) > 10:
                continue
            # Determine event priority (lower = more important)
            _ev_pri = 99
            for _kw, _p in _priority.items():
                if _kw.lower() in _purpose.lower():
                    _ev_pri = _p
                    break
            # Keep the nearest / highest-priority event per symbol
            if _sym not in _event_map:
                _event_map[_sym] = (_purpose[:30], _days, _ev_pri)
            else:
                _existing = _event_map[_sym]
                if abs(_days) < abs(_existing[1]) or _ev_pri < _existing[2]:
                    _event_map[_sym] = (_purpose[:30], _days, _ev_pri)
    except Exception:
        return {}
    # Simplify: {SYM: (label, days_away)}
    return {s: (v[0], v[1]) for s, v in _event_map.items()}


# ============================================================
# F&O PARTICIPANT-WISE OI — NSE DAILY FILE
# ============================================================
# NSE publishes participant-wise net OI for F&O each day.
# FII net long in index futures correlates strongly with institutional
# directional positioning.  For individual stocks: if FII is net long
# in stock futures → accumulation confirmation.
# File: https://archives.nseindia.com/content/nsccl/fao_participant_oi_DDMMYYYY.csv
# Returns {SYMBOL: fii_net_long_flag} where flag is True/False/None.
# Non-F&O stocks always get None.
# ============================================================
@st.cache_data(ttl=_BHAV_CACHE_TTL)
def _fetch_participant_oi() -> dict:
    """
    Returns {SYMBOL_UPPER: fii_net_long (bool)} for F&O stocks.
    True  = FII net long in that stock's futures OI today.
    False = FII net short.
    {} on failure (non-blocking).
    Only stocks that appear in the participant OI file are included.
    """
    _headers_nse = {
        "User-Agent": "Mozilla/5.0",
        "Accept":     "text/csv,application/csv",
        "Referer":    "https://www.nseindia.com/"
    }
    from io import StringIO as _SIO
    for _offset in range(5):
        _d = datetime.now() - timedelta(days=_offset)
        if _d.weekday() >= 5:
            continue
        _ds = _d.strftime("%d%m%Y")
        _url = (f"https://archives.nseindia.com/content/nsccl/"
                f"fao_participant_oi_{_ds}.csv")
        try:
            _r = requests.get(_url, headers=_headers_nse, timeout=12)
            if _r.status_code != 200:
                continue
            _df = pd.read_csv(_SIO(_r.text))
            _df.columns = [c.strip() for c in _df.columns]
            # NSE format: Client Type, Buy Qty, Sell Qty, Net Qty per segment
            # We look at rows where Client Type contains "FII/FPI"
            _ct_col  = next((c for c in _df.columns if "CLIENT" in c.upper() or "TYPE" in c.upper()), None)
            _net_col = next((c for c in _df.columns if "NET" in c.upper() and "QTY" in c.upper()), None)
            _sym_col = next((c for c in _df.columns if "SYMBOL" in c.upper()), None)
            if not (_ct_col and _net_col):
                break   # file format unexpected — don't keep retrying older dates
            if _sym_col:
                # Stock-level participant file (has SYMBOL column)
                _fii_rows = _df[_df[_ct_col].astype(str).str.upper().str.contains("FII|FPI", na=False)]
                _fii_rows = _fii_rows.dropna(subset=[_net_col])
                _fii_rows[_net_col] = pd.to_numeric(
                    _fii_rows[_net_col].astype(str).str.replace(",", ""), errors="coerce")
                _fii_rows = _fii_rows.dropna(subset=[_net_col])
                return {
                    str(row[_sym_col]).strip().upper(): (float(row[_net_col]) > 0)
                    for _, row in _fii_rows.iterrows()
                }
            else:
                # Index-level file only — no per-stock breakdown available
                break
        except Exception:
            continue
    return {}


# ============================================================
# FUNDAMENTAL QUALITY FILTER — yfinance CFO check
# ============================================================
# Fetches operating cash flow from yfinance for a symbol.
# Returns (cfo_positive: bool, cfo_value: float | None).
# Cached per-symbol for 24 hours to avoid hammering yfinance.
# Called lazily only when the fundamental gate is enabled in sidebar.
# ============================================================
@st.cache_data(ttl=_EVENT_CACHE_TTL)
def _fetch_fundamental_quality(ticker_nse: str) -> tuple:
    """
    Returns (cfo_ok: bool, rev_growth_ok: bool, note: str).
    cfo_ok = True if operating CF was positive in the most recent annual period.
    rev_growth_ok = True if TTM revenue > prior year revenue.
    Note = short diagnostic string.
    On any failure returns (True, True, 'N/A') — never penalise missing data.
    """
    try:
        _t = yf.Ticker(ticker_nse + ".NS")
        _cf = _t.cashflow
        if _cf is None or _cf.empty:
            return (True, True, "N/A")
        # Operating cash flow row
        _ocf_row = next((r for r in _cf.index
                         if "operating" in str(r).lower() and "cash" in str(r).lower()), None)
        if _ocf_row is None:
            return (True, True, "N/A")
        _ocf_vals = _cf.loc[_ocf_row].dropna()
        if len(_ocf_vals) < 1:
            return (True, True, "N/A")
        _cfo_latest = float(_ocf_vals.iloc[0])
        _cfo_ok = _cfo_latest > 0

        # Revenue growth
        _inc = _t.financials
        _rev_ok = True
        if _inc is not None and not _inc.empty:
            _rev_row = next((r for r in _inc.index if "revenue" in str(r).lower() or
                             "total revenue" in str(r).lower()), None)
            if _rev_row and len(_inc.loc[_rev_row].dropna()) >= 2:
                _rev_vals = _inc.loc[_rev_row].dropna()
                _rev_ok = float(_rev_vals.iloc[0]) >= float(_rev_vals.iloc[1])

        _note = f"CFO {'✓' if _cfo_ok else '✗'}  Rev {'↑' if _rev_ok else '↓'}"
        return (_cfo_ok, _rev_ok, _note)
    except Exception:
        return (True, True, "N/A")


# ============================================================
# SIGNAL AUDIT LOG
# ============================================================
# Appends each screener signal to a flat CSV so win-rate, regime
# performance and model drift can be tracked outside of walk-forward.
# File: .monarch_signal_log.csv  (same directory as the app)
# Schema: Timestamp, Ticker, Score, SetupType, Horizon, Entry, Target,
#         Stop, RR, KellyFrac, Sector, Regime
# ============================================================
_SIGNAL_LOG_FILE = ".monarch_signal_log.csv"
_SIGNAL_LOG_COLS = [
    "Timestamp", "Ticker", "Score", "SetupType", "Horizon",
    "Entry", "Target", "Stop", "RR", "KellyFrac", "Sector",
    "Regime", "EventFlag", "FundamentalOK"
]

def _append_signal_log(rows: list):
    """
    Appends a list of signal dicts to the CSV audit log.
    Creates the file with header if it does not exist.
    Silently skips on any write error.
    """
    if not rows:
        return
    try:
        _new_df = pd.DataFrame(rows, columns=_SIGNAL_LOG_COLS)
        _write_header = not os.path.exists(_SIGNAL_LOG_FILE)
        _new_df.to_csv(_SIGNAL_LOG_FILE, mode="a", header=_write_header, index=False)
    except Exception:
        pass


def _load_signal_log(max_rows: int = 500) -> pd.DataFrame:
    """Loads the last max_rows rows from the signal log. Returns empty DF on failure."""
    try:
        if os.path.exists(_SIGNAL_LOG_FILE):
            _df = pd.read_csv(_SIGNAL_LOG_FILE)
            return _df.tail(max_rows).reset_index(drop=True)
    except Exception:
        pass
    return pd.DataFrame(columns=_SIGNAL_LOG_COLS)


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
    color: var(--bb-white2) !important;
    font-family: var(--bb-mono) !important;
}

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
    font-family: var(--bb-mono) !important; font-size: 0.68rem !important;
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
    # Per-stock cache: {sym: {"result": <dict|None>, "ltp": <float>, "vol": <float>}}
    # Invalidation is per-stock via LTP+volume fingerprint, NOT a global TTL wipe.
    # Global TTL wipes cause a thundering-herd spike on Streamlit reruns: every stock
    # re-scores simultaneously in the same top-to-bottom pass once 60s elapse.
    # With per-stock fingerprinting, only stocks whose price or volume changed since
    # the last rerun are re-scored; all others keep their cached result for free.
    st.session_state.score_cache = {}
if "score_cache_ts" not in st.session_state:
    st.session_state.score_cache_ts = 0        # kept for backward compat, no longer used
if "rs_div_hist" not in st.session_state:
    st.session_state.rs_div_hist = {}           # {sym: [rolling 60 rs_div values]}
if "breadth_hist" not in st.session_state:
    st.session_state.breadth_hist = []          # rolling breadth readings for σ computation
if "breadth_cache" not in st.session_state:
    st.session_state.breadth_cache = None      # pre-computed breadth fraction (% above EMA20)
if "cs_rs_5d" not in st.session_state:
    st.session_state.cs_rs_5d  = {}            # cross-sectional 5d return percentile rank per stock
if "cs_rs_20d" not in st.session_state:
    st.session_state.cs_rs_20d = {}            # cross-sectional 20d return percentile rank per stock
if "param_registry" not in st.session_state:
    st.session_state.param_registry = {        # M-1/M-4/M-5 FIX: universe-level parameter registry
        "tanh_w": [], "inst_sigma": [], "prox_lambda": [], "pullback_sigma": []
    }
if "bt_hist_cache" not in st.session_state:
    st.session_state.bt_hist_cache    = {}     # Fix 14: cached yfinance fetches for BT
    st.session_state.bt_hist_cache_ts = 0.0
if "per_stock_winrate" not in st.session_state:
    st.session_state.per_stock_winrate = {}    # Fix 12: walk-forward derived win rates for Kelly
if "live_tables" not in st.session_state:
    st.session_state.live_tables = {
        "leader": None, "rs": None, "trigger": None,
        "transition": None, "exit": None
    }
# ── NEW INSTITUTIONAL FEATURES ──────────────────────────────────────────────
if "event_calendar" not in st.session_state:
    st.session_state.event_calendar = {}      # {SYM: (label, days_away)} — refreshed daily
if "participant_oi" not in st.session_state:
    st.session_state.participant_oi = {}      # {SYM: fii_net_long bool} — F&O only
if "per_stock_outcomes" not in st.session_state:
    st.session_state.per_stock_outcomes = {}  # {SYM: [+1/-1, ...]} — rolling 50 trade outcomes
if "event_blackout_enabled" not in st.session_state:
    st.session_state.event_blackout_enabled = True
if "fundamental_gate_enabled" not in st.session_state:
    st.session_state.fundamental_gate_enabled = False
if "portfolio_size_lakh" not in st.session_state:
    st.session_state.portfolio_size_lakh = 50.0
if "portfolio_heat_cap_pct" not in st.session_state:
    st.session_state.portfolio_heat_cap_pct = 40

# ============================================================
# PERSISTENT STATE — survives app restarts and tab refreshes
# ============================================================
# Three things are worth persisting to disk:
#   1. adaptive_weights   — factor weights learned from walk-forward IC
#   2. per_stock_winrate  — per-stock win rates for Kelly sizing
#   3. delivery_pct       — today's NSE Bhav Copy (4h TTL, avoids re-fetch)
#
# Everything else (raw_data_cache, score_cache, live_quotes) is intentionally
# ephemeral — it should re-fetch on each session to reflect current market state.
#
# File: .monarch_screener_state.json (same directory as the app)
# Written: after every walk-forward run + after every extraction
# Read:    once per session on first load (guarded by _screener_state_loaded flag)
# ============================================================
_SCREENER_STATE_FILE = ".monarch_screener_state.json"
_SCREENER_STATE_LOADED_KEY = "_screener_state_loaded"

def _load_screener_state() -> dict:
    """Load persistent screener state from disk. Returns {} on any error."""
    try:
        if os.path.exists(_SCREENER_STATE_FILE):
            with open(_SCREENER_STATE_FILE, "r") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
    except Exception:
        pass
    return {}

def _save_screener_state():
    """
    Persist adaptive_weights, per_stock_winrate, delivery_pct, and param_registry to disk.
    Called after walk-forward runs and after extraction so weights are never lost.
    Silently skips on any write error (read-only filesystem, permissions, etc.).
    """
    try:
        out = {}
        # Adaptive weights (the most important — hard-won from WF runs)
        aw = st.session_state.get("adaptive_weights")
        if aw and isinstance(aw, dict):
            out["adaptive_weights"] = aw
        # Per-stock win rates
        psw = st.session_state.get("per_stock_winrate", {})
        if psw:
            out["per_stock_winrate"] = psw
        # Delivery pct with timestamp so we know if it's stale
        dp = st.session_state.get("delivery_pct", {})
        if dp:
            out["delivery_pct"] = dp
            out["delivery_pct_ts"] = time.time()
        # ── PARAM REGISTRY (I-PERSIST): persist self-calibrated universe params ──
        # tanh_w, inst_sigma, prox_lambda, pullback_sigma, stab_adj_obs etc. are
        # built up across sessions. Without persistence they cold-start on every restart
        # and fall back to hard-coded bootstraps for the first N stocks processed.
        # We keep only the last 200 observations per key (same rolling buffer used live)
        # so the file stays small (< 100 KB) and is safe to read back on any restart.
        pr = st.session_state.get("param_registry", {})
        if pr and isinstance(pr, dict):
            _pr_trimmed = {k: (v[-200:] if isinstance(v, list) else v)
                          for k, v in pr.items()}
            out["param_registry"] = _pr_trimmed
            out["param_registry_ts"] = time.time()
        with open(_SCREENER_STATE_FILE, "w") as f:
            json.dump(out, f)
    except Exception:
        pass   # never crash the app because of a save failure

# ── RESTORE on first load ────────────────────────────────────────────────────
if not st.session_state.get(_SCREENER_STATE_LOADED_KEY):
    _saved = _load_screener_state()

    # Restore adaptive weights
    if "adaptive_weights" in _saved and isinstance(_saved["adaptive_weights"], dict):
        _aw_saved = _saved["adaptive_weights"]
        # Sanity check: keys must be spread/vol/coil, values must be positive floats summing ~1
        if (set(_aw_saved.keys()) == {"spread", "vol", "coil"} and
                all(isinstance(v, (int, float)) and v > 0 for v in _aw_saved.values())):
            st.session_state.adaptive_weights = _aw_saved

    # Restore per-stock win rates
    if "per_stock_winrate" in _saved and isinstance(_saved["per_stock_winrate"], dict):
        st.session_state.per_stock_winrate = _saved["per_stock_winrate"]

    # Restore delivery pct only if < 4 hours old (market data ages quickly)
    _dp_ts = _saved.get("delivery_pct_ts", 0)
    if time.time() - _dp_ts < 4 * 3600 and isinstance(_saved.get("delivery_pct"), dict):
        if "delivery_pct" not in st.session_state or not st.session_state.delivery_pct:
            st.session_state.delivery_pct = _saved["delivery_pct"]

    # ── PARAM REGISTRY (I-PERSIST): restore self-calibrated universe params ──
    # No TTL on these — calibration parameters are structural (stock volatility
    # cycles, pullback distributions) and remain valid across sessions.
    # A stale value is always better than a cold-start hardcoded fallback.
    _pr_saved = _saved.get("param_registry")
    if isinstance(_pr_saved, dict) and _pr_saved:
        _existing_pr = st.session_state.get("param_registry", {})
        _merged_pr = dict(_existing_pr)   # start from current (may have new keys)
        for _k, _v in _pr_saved.items():
            if isinstance(_v, list) and len(_v) > 0:
                # Merge saved observations into any already in session_state.
                # New session starts with no observations (empty list from init),
                # so this effectively restores the full saved buffer.
                _existing_list = _merged_pr.get(_k, [])
                if not _existing_list:   # only restore if session list is empty
                    _merged_pr[_k] = _v[-200:]
        st.session_state.param_registry = _merged_pr

    st.session_state[_SCREENER_STATE_LOADED_KEY] = True

# SCORE_CACHE_TTL removed — cache invalidation is now per-stock via LTP fingerprint
# (see get_cached_score below). No global TTL wipe.

# ============================================================
# CORE UTILITIES
# ============================================================

def to_ascending(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reverse Upstox descending candle order to ascending (oldest first).
    Fix 17: Drop duplicate timestamps before reversing — Upstox occasionally
    returns duplicate rows for the same date when markets partially reopen.
    Duplicates cause silent index bugs in rolling window calculations.
    """
    df = df.iloc[::-1].reset_index(drop=True)
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
        # Drop exact duplicate timestamps, keep first (most complete bar)
        df = df.drop_duplicates(subset=["time"], keep="first").reset_index(drop=True)
    return df


def rsi_wilder(close: pd.Series, period: int):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50)


def percentile_last(series: pd.Series, window: int):
    """
    True CDF-rank percentile: fraction of values in the window that are
    <= the last value.  Returns 0.0–1.0.  NaN when window is empty.
    This is a strict percentile rank, NOT a min-max normaliser.  The old
    min-max version returned 1.0 for any series peak and 0.0 for any
    trough regardless of how many values were clustered near those edges,
    systematically inflating scores for stocks at historic ATR/VC extremes.
    """
    s = series.tail(window).dropna()
    if len(s) < 2:
        return np.nan
    last_val = s.iloc[-1]
    # Use all values including last for the rank denominator so the result
    # is always in [0, 1] and a stock at an all-time-window high gets 1.0.
    return float((s <= last_val).mean())


def darvas_box_score(df: pd.DataFrame, atr_v: float) -> dict:
    """
    Proper Darvas Box implementation for institutional scoring.

    WHAT WAS WRONG with the old darvas_probability():
    1. Window=15 was hardcoded and arbitrary.
    2. Score = compression × inside_box × pressure — any single zero collapses everything.
    3. Only applied to Breakout setups; missed Pullback setups near box support.
    4. No dynamic box identification — just split history into two equal windows.

    CORRECT APPROACH:
    - Identify actual Darvas boxes algorithmically: a box forms when the high doesn't
      make a new N-day high for N consecutive bars (consolidation confirmed).
    - Score the current bar's relationship to the most recent valid box.
    - Three independent sub-scores combined additively (not multiplicatively):
      a) Box quality: how tight the box is (range / ATR)
      b) Box position: is price pressing resistance (Breakout) or at support (Pullback)?
      c) Time in box: how long price has been coiling (longer = more energy stored)
    - All thresholds derived from this stock's own box history.

    Returns dict with:
      "darvas_score": 0-10 pts (proportional, used as a factor not a bonus)
      "box_high": resistance level
      "box_low": support level
      "in_box": True/False
      "bars_in_box": integer
      "box_atr_ratio": box width / ATR (lower = tighter)
    """
    if len(df) < 20:
        return {"darvas_score": 0.0, "box_high": np.nan, "box_low": np.nan,
                "in_box": False, "bars_in_box": 0, "box_atr_ratio": np.nan}

    hh = df["high"]
    hl = df["low"]
    hc = df["close"]

    # Fix 2: Use actual True Range for ATR, not hh.diff() proxy
    # hh.diff() only captures high-to-high movement, missing gap component
    _tr_darvas = pd.concat([
        hh - hl,
        (hh - hc.shift(1)).abs(),
        (hl - hc.shift(1)).abs()
    ], axis=1).max(axis=1)
    _atr_darvas = float(_tr_darvas.ewm(alpha=1/14, adjust=False).mean().iloc[-1]) \
                  if len(_tr_darvas) >= 5 else atr_v

    # Dynamic box confirmation window from ATR-based cycle length (using real ATR)
    _daily_move = float(_tr_darvas.tail(20).median()) if len(_tr_darvas) >= 20 else (_atr_darvas * 0.5)
    _bars_per_atr = max(3, int(_atr_darvas / (_daily_move + 1e-9)))
    _box_confirm_n = int(np.clip(_bars_per_atr, 3, 10))

    # Scan backwards for the most recent Darvas box
    # A box HIGH is set when price makes a new high then fails to exceed it for _box_confirm_n bars.
    _box_high = None
    _box_low  = None
    _box_start_idx = None

    for i in range(len(hh) - 1, _box_confirm_n * 2, -1):
        # Candidate box high = the highest high in the confirmation window ending at i
        _window_end = i
        _window_start = max(0, i - _box_confirm_n)
        _candidate_high = float(hh.iloc[_window_start : _window_end + 1].max())

        # Box confirmed if price did not exceed this high in the N bars that followed
        _confirm_start = _window_end + 1
        _confirm_end   = min(len(hh), _confirm_start + _box_confirm_n)
        if _confirm_end > len(hh):
            continue
        _post_high = float(hh.iloc[_confirm_start : _confirm_end].max())
        if _post_high > _candidate_high:
            continue   # box was broken — not a valid box

        # Valid box found — the low is the lowest low WITHIN THE BOX FORMATION WINDOW ONLY.
        # FIX G: Old code used hl.iloc[_window_start : len(hl)].min() — this extended
        # the low forward to the current bar, picking up any washout that occurred AFTER
        # the box formed. A post-formation dip would widen the box artificially, making
        # the box_atr_ratio larger and the tightness score lower.
        # Fix: anchor _box_low to the confirmation window (formation + confirm bars only).
        _box_low_end   = min(len(hl), _confirm_end)
        _box_high      = _candidate_high
        _box_low       = float(hl.iloc[_window_start : _box_low_end].min())
        _box_start_idx = _window_start
        break

    if _box_high is None or _box_low is None:
        return {"darvas_score": 0.0, "box_high": np.nan, "box_low": np.nan,
                "in_box": False, "bars_in_box": 0, "box_atr_ratio": np.nan}

    _ltp         = float(hc.iloc[-1])
    _box_width   = _box_high - _box_low + 1e-9
    _bars_in_box = len(hh) - _box_start_idx
    _in_box      = (_ltp <= _box_high) and (_ltp >= _box_low)
    _box_atr_ratio = _box_width / (atr_v + 1e-9)   # lower = tighter box

    # ── Sub-score a: Box tightness (0-1) ──
    # Derived from stock's own ATR: a box spanning 1 ATR is typical (score 0.5),
    # < 0.5 ATR is very tight (score near 1), > 3 ATR is loose (score near 0).
    # Use sigmoid derived from stock's own box_atr history if available.
    _tightness = float(1.0 / (1.0 + _box_atr_ratio))   # naturally 0-1, higher = tighter

    # ── Sub-score b: Position within box (0-1) ──
    # Near top (resistance) = Breakout setup, near bottom (support) = Pullback setup.
    # Score is symmetric: high scores near both top AND bottom, low near midpoint.
    # This makes Darvas useful for both setup types.
    _pos_in_box = (_ltp - _box_low) / (_box_width)   # 0=at support, 1=at resistance
    # V-shape: high near edges (0 or 1), low at centre (0.5)
    _pos_score  = float(1.0 - 4.0 * (_pos_in_box - 0.5) ** 2)   # parabola, peak at edges
    _pos_score  = float(np.clip(_pos_score, 0.0, 1.0))

    # ── Sub-score c: Time coiling in box (0-1) ──
    # Fix 23: Vectorized — replace O(N²) for-loop with rolling max comparison
    # "In-box" bar = rolling N-bar max did NOT exceed the prior N-bar max
    _n = _box_confirm_n
    _roll_max_cur  = hh.rolling(_n).max()
    _roll_max_prev = hh.rolling(_n).max().shift(_n)
    # A bar is "coiling" if current rolling max <= prior rolling max (no breakout)
    _coiling_mask = (_roll_max_cur <= _roll_max_prev).fillna(False)
    _coiling_frac = float(_coiling_mask.tail(60).mean()) if len(_coiling_mask) >= 10 else 0.5
    # Typical box duration = expected consecutive coiling bars for this stock
    _typical_box_dur = max(int(_coiling_frac * 20), 3)   # scale by how often it coils
    _time_score = float(np.clip(_bars_in_box / (_typical_box_dur * 2.0 + 1e-9), 0.0, 1.0))

    # Additive combination (not multiplicative — prevents single-zero collapse)
    darvas_score = round((_tightness * 0.40 + _pos_score * 0.35 + _time_score * 0.25) * 10.0, 1)

    return {
        "darvas_score":   darvas_score,
        "box_high":       round(_box_high, 2),
        "box_low":        round(_box_low,  2),
        "in_box":         _in_box,
        "bars_in_box":    _bars_in_box,
        "box_atr_ratio":  round(_box_atr_ratio, 2),
    }


def darvas_probability(df: pd.DataFrame, window: int):
    """Legacy wrapper — kept for backward compatibility. Use darvas_box_score()."""
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
    Fix 16: Sanity guard — if the patched bar would have high < low (corrupt live quote),
    the patch is reverted for that field to prevent downstream division errors.
    """
    if not live:
        return df

    df = df.copy()
    last_idx = df.index[-1]

    # Snapshot pre-patch values for sanity revert
    _pre_high = float(df.at[last_idx, "high"])
    _pre_low  = float(df.at[last_idx, "low"])

    ltp    = live.get("ltp")
    high   = live.get("high")
    low    = live.get("low")
    volume = live.get("volume")
    oi     = live.get("oi")

    if ltp is not None:
        df.at[last_idx, "close"] = ltp
    if high is not None:
        df.at[last_idx, "high"] = max(_pre_high, high)
    if low is not None:
        df.at[last_idx, "low"] = min(_pre_low, low)
    if volume is not None:
        df.at[last_idx, "volume"] = volume
    if oi is not None and "oi" in df.columns:
        df.at[last_idx, "oi"] = oi

    # Sanity guard: high must be >= low and >= close
    _patched_high = float(df.at[last_idx, "high"])
    _patched_low  = float(df.at[last_idx, "low"])
    _patched_close = float(df.at[last_idx, "close"])
    if _patched_high < _patched_low:
        # Corrupt live OHLC — revert high/low to pre-patch values
        df.at[last_idx, "high"] = _pre_high
        df.at[last_idx, "low"]  = _pre_low
    if _patched_close > float(df.at[last_idx, "high"]):
        df.at[last_idx, "high"] = _patched_close
    if _patched_close < float(df.at[last_idx, "low"]):
        df.at[last_idx, "low"] = _patched_close

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

    # ── Dynamic Nifty 50 constituent list ──
    # Fetched from NSE's live index API each session (cached 4h via st.cache_data).
    # This eliminates the stale hardcoded list that misses quarterly rebalances.
    # Fallback: if the API is unreachable, use the F&O underlying filter which
    # approximates the large-cap universe.
    @st.cache_data(ttl=14400)   # 4-hour cache — index rebalances are quarterly
    def _get_nifty50_live():
        """Returns set of NSE trading symbols that are current Nifty 50 constituents."""
        try:
            headers_nse = {
                "User-Agent": "Mozilla/5.0",
                "Accept": "application/json",
                "Referer": "https://www.nseindia.com/"
            }
            r = requests.get(
                "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%2050",
                headers=headers_nse, timeout=10
            )
            if r.status_code == 200:
                data = r.json().get("data", [])
                # Each entry has "symbol" key — skip the index row itself
                syms = {d["symbol"] for d in data if d.get("symbol") and d["symbol"] != "NIFTY 50"}
                if len(syms) >= 40:   # sanity check
                    return syms
        except Exception:
            pass
        # Fallback: last-known list (maintained here as a single source of truth)
        return {
            "RELIANCE","HDFCBANK","ICICIBANK","INFY","ITC","TCS","LT","SBIN","AXISBANK",
            "KOTAKBANK","BHARTIARTL","ASIANPAINT","HCLTECH","MARUTI","SUNPHARMA","ULTRACEMCO",
            "TITAN","WIPRO","NESTLEIND","POWERGRID","NTPC","BAJFINANCE","BAJAJFINSV",
            "INDUSINDBK","TECHM","M&M","TATAMOTORS","ADANIENT","ADANIPORTS","ONGC",
            "COALINDIA","JSWSTEEL","HINDALCO","TATASTEEL","BPCL","GRASIM","CIPLA",
            "DRREDDY","EICHERMOT","HEROMOTOCO","BRITANNIA","DIVISLAB","SBILIFE",
            "HDFCLIFE","APOLLOHOSP","BAJAJ-AUTO","UPL","SHREECEM","HINDUNILVR","TATACONSUM"
        }

    nifty50_live = _get_nifty50_live()
    nifty = eq[eq['trading_symbol'].isin(nifty50_live)]
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
    # Fix 5: store RSI period in session_state so score_stock_dual uses it
    st.session_state.rsi_period = int(rsi_p_val)

    st.divider()
    # L-3: Sector concentration cap toggle
    # Fix 32: Use key-only state management — don't set value= AND write to a different
    # session_state key. That creates two conflicting state sources for the same widget.
    st.checkbox(
        "Max 1 stock per sector",
        key="sector_cap_enabled",   # Streamlit writes directly to this key
        help="When ON, top screener output shows at most 1 stock per sector (sector-diversified signals)"
    )

    st.divider()
    st.caption("INSTITUTIONAL RISK CONTROLS")

    # ── EVENT BLACKOUT ──────────────────────────────────────────────────────
    st.checkbox(
        "Event blackout (±3 days)",
        key="event_blackout_enabled",
        value=True,
        help=(
            "Suppress or flag signals within ±3 trading days of results, "
            "board meetings, AGMs, or dividend ex-dates. "
            "Applies to all stocks — not just F&O."
        )
    )

    # ── FUNDAMENTAL GATE ────────────────────────────────────────────────────
    st.checkbox(
        "Fundamental quality gate",
        key="fundamental_gate_enabled",
        value=False,
        help=(
            "Applies a −10 pt soft penalty to stocks with negative operating "
            "cash flow in the most recent annual report (via yfinance). "
            "Stocks with no data are NOT penalised. "
            "Fetched once per day per symbol — first extraction may be slow."
        )
    )

    # ── PORTFOLIO HEAT CAP ──────────────────────────────────────────────────
    st.number_input(
        "Portfolio size (₹ Lakh)",
        min_value=1.0, max_value=10000.0, value=50.0, step=10.0,
        key="portfolio_size_lakh",
        help=(
            "Used to cap Kelly% by liquidity: max position = min(Kelly%, "
            "ADTV_turnover / portfolio_size). "
            "Also used to enforce the 2% total heat cap."
        )
    )
    st.number_input(
        "Max total Kelly% (heat cap)",
        min_value=10, max_value=100, value=40, step=5,
        key="portfolio_heat_cap_pct",
        help=(
            "If the sum of all signal Kelly fractions exceeds this %, "
            "all Kelly values are scaled down proportionally. "
            "Institutional standard: 25-40% of capital across open positions."
        )
    )

    st.divider()
    # ── VOLUME PRE-FILTER ─────────────────────────────────────────────────────
    # Skips stocks below a minimum 20-day average volume before downloading/scoring.
    # Reduces Full NSE scan from 2800 → ~300-600 stocks, cutting time by 80%+.
    # Stage 1 (fast): uses live volume as proxy before historical download.
    # Stage 2 (exact): checks confirmed 20d avg after OHLCV data is downloaded.
    st.caption("VOLUME PRE-FILTER")
    _vol_filter_options = {
        "No filter":    0,
        "> 50K shares": 50_000,
        "> 1L shares":  100_000,
        "> 5L shares":  500_000,
        "> 10L shares": 1_000_000,
        "> 50L shares": 5_000_000,
    }
    _vol_filter_label = st.selectbox(
        "Min avg daily volume",
        options=list(_vol_filter_options.keys()),
        index=2,   # default: >1L — good for NSE mid/large cap universe
        key="vol_filter_label",
        help=(
            "Stocks below this 20-day avg volume threshold are skipped entirely. "
            "Recommended: 1L+ for F&O/Nifty50, 5L+ for Full NSE to keep scan fast."
        )
    )
    _min_avg_vol = _vol_filter_options[_vol_filter_label]
    st.session_state.min_avg_vol = _min_avg_vol
    if _min_avg_vol > 0:
        st.caption(f"Scanning only stocks with avg vol > {_min_avg_vol:,.0f}")

    st.divider()
    st.caption("Live data source: Upstox /market-quote/quotes")
    if st.session_state.live_quotes_cache:
        n_live = len(st.session_state.live_quotes_cache)
        last_t = st.session_state.get("_last_live_refresh", 0)
        age    = int(time.time() - last_t) if last_t else "—"
        st.success(f"✅ {n_live} live quotes | refreshed {age}s ago")
    else:
        st.info("No live data yet — run extraction")

    # ── PERSISTENT STATE STATUS ──────────────────────────────────────────────
    st.divider()
    st.caption("PERSISTENT STATE")
    _aw_disp = st.session_state.get("adaptive_weights")
    if _aw_disp:
        st.success(
            f"✅ Adaptive weights loaded\n"
            f"Spread {_aw_disp.get('spread',0):.3f} · "
            f"Vol {_aw_disp.get('vol',0):.3f} · "
            f"Coil {_aw_disp.get('coil',0):.3f}"
        )
    else:
        st.info("Using default weights (0.40 / 0.40 / 0.20)\nRun Walk-Forward to calibrate")
    _psw_disp = st.session_state.get("per_stock_winrate", {})
    if _psw_disp:
        st.caption(f"Kelly: {len(_psw_disp)} stocks have WF win rates")
    _dp_disp = st.session_state.get("delivery_pct", {})
    if _dp_disp:
        st.caption(f"Delivery: {len(_dp_disp)} stocks cached")

    if st.button("🗑 Reset saved weights", key="reset_weights",
                 help="Clears adaptive weights and win rates — reverts to Jan 2026 priors"):
        st.session_state.pop("adaptive_weights", None)
        st.session_state.per_stock_winrate = {}
        _save_screener_state()
        st.rerun()

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
SECTOR_TICKERS = {
    "IT":         "^CNXIT",
    "Bank":       "^NSEBANK",
    "Auto":       "^CNXAUTO",
    "Pharma":     "^CNXPHARMA",
    "Metal":      "^CNXMETAL",
    "Energy":     "^CNXENERGY",
    "Infra":      "^CNXINFRA",
    "FMCG":       "^CNXFMCG",
    "Realty":     "^CNXREALTY",
    "PSUBank":    "^CNXPSUBANK",
    "Chemicals":  "^CNXCHEMICALS",
    "ConsumerDur":"^CNXCONSUMER",
    "Insurance":  "^CNXFINSERVICE",
    "Telecom":    "^CNXTELECOM",
    "Retail":     "^CNXCONSUMER",    # No dedicated NSE Retail index; proxy via Consumer
    "Logistics":  "^CNXINFRA",       # No dedicated NSE Logistics index; proxy via Infra
}

STOCK_SECTOR_MAP = {
    # IT
    "TCS":"IT","INFY":"IT","WIPRO":"IT","HCLTECH":"IT","TECHM":"IT",
    "LTIM":"IT","MPHASIS":"IT","COFORGE":"IT","PERSISTENT":"IT","OFSS":"IT",
    "KPITTECH":"IT","TATAELXSI":"IT","MASTEK":"IT","NIITTECH":"IT","HEXAWARE":"IT",
    # Bank (Private)
    "HDFCBANK":"Bank","ICICIBANK":"Bank","KOTAKBANK":"Bank","AXISBANK":"Bank",
    "INDUSINDBK":"Bank","FEDERALBNK":"Bank","IDFCFIRSTB":"Bank","AUBANK":"Bank",
    "BAJFINANCE":"Bank","BAJAJFINSV":"Bank","RBLBANK":"Bank","YESBANK":"Bank",
    "CSBBANK":"Bank","DCBBANK":"Bank","KARURVYSYA":"Bank",
    # PSU Bank
    "SBIN":"PSUBank","BANKBARODA":"PSUBank","PNB":"PSUBank","CANBK":"PSUBank",
    "UNIONBANK":"PSUBank","BANKINDIA":"PSUBank","MAHABANK":"PSUBank",
    "INDIANB":"PSUBank","UCOBANK":"PSUBank","CENTRALBK":"PSUBank",
    # Auto & Auto-ancillary
    "MARUTI":"Auto","TATAMOTORS":"Auto","M&M":"Auto","BAJAJ-AUTO":"Auto",
    "HEROMOTOCO":"Auto","EICHERMOT":"Auto","TVSMOTORS":"Auto",
    "MOTHERSON":"Auto","BOSCHLTD":"Auto","BHARATFORG":"Auto","BALKRISIND":"Auto",
    "APOLLOTYRE":"Auto","MRF":"Auto","CEATLTD":"Auto","EXIDEIND":"Auto",
    # Pharma & Healthcare
    "SUNPHARMA":"Pharma","DRREDDY":"Pharma","CIPLA":"Pharma","DIVISLAB":"Pharma",
    "TORNTPHARM":"Pharma","AUROPHARMA":"Pharma","APOLLOHOSP":"Pharma",
    "LUPIN":"Pharma","BIOCON":"Pharma","ALKEM":"Pharma","GLENMARK":"Pharma",
    "IPCALAB":"Pharma","NATCOPHARM":"Pharma","LAURUSLABS":"Pharma","GRANULES":"Pharma",
    "FORTIS":"Pharma","METROPOLIS":"Pharma","LALPATHLAB":"Pharma",
    # Metal & Mining
    "TATASTEEL":"Metal","JSWSTEEL":"Metal","HINDALCO":"Metal","SAIL":"Metal",
    "VEDL":"Metal","COALINDIA":"Metal","NMDC":"Metal","JINDALSTEL":"Metal",
    "APLAPOLLO":"Metal","RATNAMANI":"Metal","NATIONALUM":"Metal","MOIL":"Metal",
    # Energy & Oil-Gas
    "ONGC":"Energy","NTPC":"Energy","POWERGRID":"Energy","BPCL":"Energy",
    "IOC":"Energy","GAIL":"Energy","RELIANCE":"Energy","HPCL":"Energy",
    "PETRONET":"Energy","OIL":"Energy","HINDPETRO":"Energy","MGL":"Energy",
    "IGL":"Energy","TATAPOWER":"Energy","ADANIGREEN":"Energy","ADANIENT":"Energy",
    # Infra & Capital Goods
    "LT":"Infra","ADANIPORTS":"Infra","IRFC":"Infra","RVNL":"Infra",
    "IRCON":"Infra","NBCC":"Infra","ULTRACEMCO":"Infra","SHREECEM":"Infra",
    "AMBUJACEMENT":"Infra","ACC":"Infra","SIEMENS":"Infra","ABB":"Infra",
    "BEL":"Infra","HAL":"Infra","BHEL":"Infra","CUMMINSIND":"Infra",
    "THERMAX":"Infra","KEC":"Infra","KALPATPOWR":"Infra","VOLTAS":"Infra",
    # FMCG
    "HINDUNILVR":"FMCG","ITC":"FMCG","NESTLEIND":"FMCG","BRITANNIA":"FMCG",
    "DABUR":"FMCG","MARICO":"FMCG","GODREJCP":"FMCG","ASIANPAINT":"FMCG",
    "EMAMILTD":"FMCG","COLPAL":"FMCG","PGHH":"FMCG","TATACONSUM":"FMCG",
    "UBL":"FMCG","RADICO":"FMCG","VBL":"FMCG",
    # Realty
    "DLF":"Realty","LODHA":"Realty","OBEROIRLTY":"Realty","PHOENIXLTD":"Realty",
    "GODREJPROP":"Realty","PRESTIGE":"Realty","BRIGADE":"Realty","SOBHA":"Realty",
    # Chemicals & Specialty
    "PIDILITIND":"Chemicals","SRF":"Chemicals","DEEPAKNTR":"Chemicals",
    "AARTIIND":"Chemicals","NAVINFLUOR":"Chemicals","ALKYLAMINE":"Chemicals",
    "FINEORG":"Chemicals","VINATIORGA":"Chemicals","BALRAMCHIN":"Chemicals",
    # Insurance & AMC
    "SBILIFE":"Insurance","HDFCLIFE":"Insurance","ICICIPRULI":"Insurance",
    "LICIHSGFIN":"Insurance","MUTHOOTFIN":"Insurance","CHOLAFIN":"Insurance",
    "ICICIGI":"Insurance","NIACL":"Insurance","GICRE":"Insurance",
    "HDFCAMC":"Insurance","NAM-INDIA":"Insurance","ABSLAMC":"Insurance",
    # Telecom & Media
    "BHARTIARTL":"Telecom","IDEA":"Telecom","TATACOMM":"Telecom","INDUSTOWER":"Telecom",
    # Consumer Durables & Retail
    "HAVELLS":"ConsumerDur","CROMPTON":"ConsumerDur","WHIRLPOOL":"ConsumerDur",
    "BLUESTARL":"ConsumerDur","SYMPHONY":"ConsumerDur","TITAN":"ConsumerDur",
    "TRENT":"Retail","DMART":"Retail","ABFRL":"Retail","SHOPERSTOP":"Retail",
    # Logistics
    "CONCOR":"Logistics","BLUEDART":"Logistics","MAHLOG":"Logistics",
}

@st.cache_data(ttl=900)   # FIX I-14: 15-min cache (was 5-min) — market data rarely changes intra-session
def get_market_context():
    out = dict(nifty_r5=None, nifty_r20=None,
               nifty_above_20dma=True, nifty_above_50dma=True,
               regime="BULL",   # BULL / CHOP / BEAR — derived from 50DMA + slope
               vix_level=None, vix_falling=True,
               vix_median=14.5, vix_sigma=4.5,
               sector_returns={}, sector_returns_10d={},
               top_sectors=set(),
               market_ok=True, market_notes=[])
    try:
        n = yf.download("^NSEI", period="365d", interval="1d", progress=False)
        if not n.empty:
            c = n["Close"].squeeze()
            out["nifty_r5"]  = float(c.iloc[-1]/c.iloc[-6]-1)  if len(c)>=6  else None
            out["nifty_r20"] = float(c.iloc[-1]/c.iloc[-21]-1) if len(c)>=21 else None
            dma20 = float(c.tail(20).mean())
            dma50 = float(c.tail(50).mean()) if len(c) >= 50 else dma20
            out["nifty_above_20dma"] = float(c.iloc[-1]) > dma20
            out["nifty_above_50dma"] = float(c.iloc[-1]) > dma50

            # Regime classification — derived from Nifty's own moving averages + breadth.
            # No fixed price levels, no arbitrary thresholds.
            # BULL:  price > 50DMA AND 20DMA is rising (last close > 10d-ago close of 20DMA)
            # CHOP:  price near 50DMA (within 1 ATR of it) or mixed signals
            # BEAR:  price < 50DMA AND 20DMA is falling
            # FIX 12: Also use breadth (% of universe stocks above their 20DMA) as a
            # faster regime signal. When breadth drops below 40%, shift to BEAR immediately
            # rather than waiting for the Nifty 50DMA to roll over (which lags by days/weeks).
            _nifty_atr = float(c.diff().abs().tail(14).mean())   # proxy ATR from daily changes
            _dma20_now = float(c.tail(20).mean())
            _dma20_10d = float(c.iloc[-11:-1].mean()) if len(c) >= 11 else _dma20_now
            _dma20_slope = _dma20_now - _dma20_10d   # positive = rising
            _gap_to_50dma = float(c.iloc[-1]) - dma50

            # Read pre-computed breadth (faster signal)
            _live_breadth = st.session_state.get("breadth_cache", None)

            if _live_breadth is not None and _live_breadth < 0.40:
                # FIX 12: Breadth below 40% = broad market deterioration = BEAR regardless of 50DMA
                out["regime"] = "BEAR"
            elif _gap_to_50dma > 0 and _dma20_slope > 0:
                out["regime"] = "BULL"
            elif _gap_to_50dma < -_nifty_atr:
                out["regime"] = "BEAR"
            else:
                out["regime"] = "CHOP"

            if out["regime"] == "BEAR":
                out["market_notes"].append("Nifty in BEAR regime — screener signals suppressed")
            elif out["regime"] == "CHOP":
                out["market_notes"].append("Nifty in CHOP — reduced signal confidence")
            elif not out["nifty_above_20dma"]:
                out["market_notes"].append("Nifty below 20DMA — breakout risk elevated")
    except Exception:
        pass
    try:
        v = yf.download("^INDIAVIX", period="365d", interval="1d", progress=False)
        if not v.empty:
            vc = v["Close"].squeeze()
            out["vix_level"]   = round(float(vc.iloc[-1]), 2)
            # VIX trend: 5-bar linear regression slope instead of noisy 3-bar comparison
            if len(vc) >= 5:
                _vix_slope = float(np.polyfit(range(5), vc.tail(5).values, 1)[0])
                out["vix_falling"] = _vix_slope < 0   # negative slope = VIX falling
            else:
                out["vix_falling"] = True
            # Compute rolling median and σ from fetched history so VIX adj is self-calibrated
            # Use all available history (up to 1 year) — no fixed thresholds
            out["vix_median"]  = round(float(vc.median()), 2) if len(vc) >= 20 else 14.5
            out["vix_sigma"]   = round(float(vc.std()), 2)    if len(vc) >= 20 else 4.5
            if not out["vix_falling"]:
                out["market_notes"].append(f"VIX rising ({out['vix_level']}) — breakouts may reverse fast")
    except Exception:
        pass
    # Fix 24: Parallelise sector downloads — was sequential (~18 calls × 1-2s each = 30s)
    # Using ThreadPoolExecutor same as the Upstox historical fetcher
    sr_5d  = {}
    sr_10d = {}
    def _fetch_sector(name_ticker):
        _name, _ticker = name_ticker
        try:
            s = yf.download(_ticker, period="60d", interval="1d", progress=False)
            if not s.empty:
                sc = s["Close"].squeeze()
                r5  = float(sc.iloc[-1] / sc.iloc[-6]  - 1) if len(sc) >= 6  else None
                r10 = float(sc.iloc[-1] / sc.iloc[-11] - 1) if len(sc) >= 11 else None
                return _name, r5, r10
        except Exception:
            pass
        return _name, None, None

    with ThreadPoolExecutor(max_workers=8) as _sect_exec:
        for _sname, _r5, _r10 in _sect_exec.map(_fetch_sector, SECTOR_TICKERS.items()):
            if _r5  is not None: sr_5d[_sname]  = _r5
            if _r10 is not None: sr_10d[_sname] = _r10
    out["sector_returns"]     = sr_5d
    out["sector_returns_10d"] = sr_10d
    if sr_5d:
        out["top_sectors"] = {k for k,_ in sorted(sr_5d.items(), key=lambda x:x[1], reverse=True)[:3]}
    out["market_ok"] = out["nifty_above_20dma"]
    return out

mkt              = get_market_context()
nifty_r5         = mkt["nifty_r5"]
nifty_r20        = mkt["nifty_r20"]
sector_returns     = mkt["sector_returns"]
sector_returns_10d = mkt["sector_returns_10d"]   # FIX B-01

# Override sector returns with per-stock averages from loaded cache
# if cache is already populated (e.g. re-render after extraction).
# ETF tickers can return wrong data; stocks-in-cache are ground truth.
def _recompute_sector_returns_from_cache():
    _r5a  = {}; _r10a = {}
    for _sym, _df in st.session_state.get("raw_data_cache", {}).items():
        _sec = get_sector(_sym)
        if _sec is None:
            continue
        try:
            _c = pd.DataFrame(_df)["close"]
            if len(_c) >= 6:
                _r5a.setdefault(_sec, []).append(float(_c.iloc[-1] / _c.iloc[-6] - 1))
            if len(_c) >= 11:
                _r10a.setdefault(_sec, []).append(float(_c.iloc[-1] / _c.iloc[-11] - 1))
        except Exception:
            pass
    r5  = {s: float(np.mean(v)) for s, v in _r5a.items()}
    r10 = {s: float(np.mean(v)) for s, v in _r10a.items()}
    return r5, r10

def get_sector(ticker):
    return STOCK_SECTOR_MAP.get(ticker.upper(), None)

if st.session_state.get("raw_data_cache"):
    _cache_sr5, _cache_sr10 = _recompute_sector_returns_from_cache()
    if _cache_sr5:
        sector_returns     = {**sector_returns,     **_cache_sr5}
        sector_returns_10d = {**sector_returns_10d, **_cache_sr10}
top_sectors      = mkt["top_sectors"]

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
    # Wilder smoothing (alpha=1/14) — consistent with rsi_wilder(period=14)
    # Old: ewm(span=14) uses α=2/(14+1)=0.133; Wilder uses α=1/14=0.0714
    # Wilder is the industry standard for ATR and gives a smoother, slower-reacting line
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift(1)).abs()
    lc = (df["low"]  - df["close"].shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(alpha=1/14, adjust=False).mean()

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
    # Threshold: 0.5× body of previous candle (ATR-relative via prev_body)
    # A meaningful gap must exceed half the previous candle's body to avoid noise.
    _gap_min = max(prev_body * 0.5, (prev_rng * 0.01))   # at least 1% of prev range
    if o > prev_c + _gap_min and c > o:
        patterns.append("GapContinue"); pts += 2

    return min(pts, 10), patterns

def consolidation_score(df, window=15):
    """
    Measures how tight/clean the consolidation base is.
    Returns 0-1. Higher = tighter base = better breakout setup.
    Sigmoid centre and steepness derived from rolling CV distribution.
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
    cv_avg   = (flat_hi + flat_lo) / 2.0
    # Derive sigmoid centre from rolling CV history of this stock.
    # Centre = 60-bar rolling median CV (the stock's "normal" base tightness).
    # Steepness k = 1/σ of rolling CV (tight distribution → steep curve).
    if len(df) >= window * 4:
        _cv_hist = pd.Series([
            ((df["high"].iloc[max(0,i-window):i].std() / (df["high"].iloc[max(0,i-window):i].mean() + 1e-9)) +
             (df["low"].iloc[max(0,i-window):i].std()  / (df["low"].iloc[max(0,i-window):i].mean()  + 1e-9))) / 2.0
            for i in range(window, min(window*4, len(df)))
        ], dtype=float).dropna()
        if len(_cv_hist) >= 5:
            _cv_centre = float(_cv_hist.median())
            _cv_sigma  = float(_cv_hist.std())
            _cv_k      = 1.0 / max(_cv_sigma, 0.002)
        else:
            _cv_centre, _cv_k = 0.01, 200.0
    else:
        _cv_centre, _cv_k = 0.01, 200.0
    flatness = float(1.0 / (1.0 + np.exp(_cv_k * (cv_avg - _cv_centre))))
    return max(0.0, min(1.0, (compression * 0.6 + flatness * 0.4)))

def bb_width_compression_score(c_series: pd.Series, window: int = 20, mult: float = 2.0) -> tuple:
    """
    Bollinger Band Width Compression Score.

    BB Width = (Upper - Lower) / Middle = 2 × mult × std(c, N) / SMA(c, N).
    Squeeze = BB width is at a multi-month percentile LOW relative to its own history.

    Returns:
        (bb_width_pct, bb_squeeze_score)
        bb_width_pct  : 0-1 percentile of current width vs 250d history (LOW = compressed)
        bb_squeeze_score : 0-1, 1 = maximum squeeze (lowest ever BB width)
    """
    if len(c_series) < window + 10:
        return 0.5, 0.5

    sma = c_series.rolling(window).mean()
    std = c_series.rolling(window).std()
    bb_width = (2.0 * mult * std) / (sma.replace(0, np.nan))   # normalised bandwidth
    bb_width = bb_width.dropna()

    if len(bb_width) < 10:
        return 0.5, 0.5

    current_bw = float(bb_width.iloc[-1])
    hist_bw    = bb_width.iloc[:-1]   # no look-ahead

    # Percentile of current width vs history: LOW pct = compressed = good
    bb_width_pct = float((hist_bw <= current_bw).mean())   # CDF rank
    bb_squeeze_score = 1.0 - bb_width_pct   # invert: 1 = tightest ever

    return round(bb_width_pct, 4), round(bb_squeeze_score, 4)


def volume_dryup_score(v_series: pd.Series, window_short: int = 5, window_long: int = 20) -> tuple:
    """
    Volume Dry-Up Score — detects declining volume BEFORE a breakout.
    Orthogonal to volume surge (which measures today's spike).

    Logic:
        - Compute rolling N-day avg volume over the last `window_short` bars
        - Compare against the prior `window_long` - `window_short` avg
        - Drying up = short avg < long avg (supply exhaustion before the move)
        - Percentile-rank this ratio vs the stock's own rolling history

    Returns:
        (dryup_ratio, dryup_score)
        dryup_ratio   : vol_short_avg / vol_long_avg (< 1 = drying up)
        dryup_score   : 0-1 percentile, 1 = maximum dry-up (most bullish)
    """
    if len(v_series) < window_long + window_short:
        return 1.0, 0.5

    v = v_series.replace(0, np.nan).dropna()
    if len(v) < window_long:
        return 1.0, 0.5

    short_avg = float(v.tail(window_short).mean())
    long_avg  = float(v.tail(window_long).mean())
    dryup_ratio = short_avg / (long_avg + 1e-9)

    # Rolling ratio history for percentile ranking (no look-ahead: use iloc[:-1])
    v_hist = v.iloc[:-1]
    if len(v_hist) < window_long:
        # fallback: direct ratio — 0.7 ratio → 0.7 dryup score (inverted near 1)
        dryup_score = float(np.clip(1.0 - dryup_ratio, 0.0, 1.0))
        return round(dryup_ratio, 4), round(dryup_score, 4)

    _short_roll = v_hist.rolling(window_short).mean()
    _long_roll  = v_hist.rolling(window_long).mean()
    _ratio_hist = (_short_roll / (_long_roll + 1e-9)).dropna()

    if len(_ratio_hist) < 5:
        dryup_score = float(np.clip(1.0 - dryup_ratio, 0.0, 1.0))
        return round(dryup_ratio, 4), round(dryup_score, 4)

    # LOW ratio = drying up = bullish before breakout → invert for score
    pct_rank    = float((_ratio_hist <= dryup_ratio).mean())   # CDF: how "normal" is this ratio
    dryup_score = 1.0 - pct_rank   # 1 = driest (most compressed vol) = highest score

    return round(dryup_ratio, 4), round(dryup_score, 4)


def clv_accumulation_score(c_series: pd.Series, h_series: pd.Series,
                           l_series: pd.Series, v_series: pd.Series,
                           window: int = 20) -> tuple:
    """
    Close Location Value (CLV) — Institutional Accumulation Pressure.

    CLV = ((Close - Low) - (High - Close)) / (High - Low)
    Money Flow = CLV × Volume (positive = buying pressure)
    Accumulation Score = percentile rank of 20d rolling CLV-vol sum vs own history.

    Returns:
        (clv_raw, accum_score)
        clv_raw     : latest CLV × vol / avg_vol (normalised money flow)
        accum_score : 0-1, 1 = maximum institutional accumulation
    """
    if len(c_series) < window + 5:
        return 0.0, 0.5

    hl_range = (h_series - l_series).replace(0, np.nan)
    clv       = ((c_series - l_series) - (h_series - c_series)) / hl_range
    clv       = clv.fillna(0.0)   # symmetric assumption when range = 0
    mf        = clv * v_series    # money flow = CLV × volume

    # Normalise by average volume so large-cap stocks don't dominate by raw vol
    avg_v  = v_series.rolling(window).mean().replace(0, np.nan)
    mf_norm = mf / avg_v   # per-unit-of-avg-vol money flow

    roll_mf = mf_norm.rolling(window).sum().dropna()   # 20d accumulated pressure

    if len(roll_mf) < 5:
        return float(clv.iloc[-1]) if pd.notna(clv.iloc[-1]) else 0.0, 0.5

    current_mf = float(roll_mf.iloc[-1])
    hist_mf    = roll_mf.iloc[:-1]   # no look-ahead

    accum_pct   = float((hist_mf <= current_mf).mean())   # percentile rank

    return round(current_mf, 4), round(accum_pct, 4)


def detect_vcp(c: pd.Series, h: pd.Series, l: pd.Series, v: pd.Series,
               atr: pd.Series) -> dict:
    """
    Volatility Contraction Pattern (VCP) Detector.
    All 9 steps implemented with zero fixed constants.
    Every threshold is derived from the stock's own historical distribution.

    Steps:
      1. Swing detection  — dynamic window from ATR / median-daily-range cycle
      2. Pullback extraction — depth per swing
      3. Contraction analysis — OLS slope of pullback depths, percentile-ranked
      4. Volatility compression — ATR_short/ATR_long percentile rank
      5. Volume dry-up during consolidation — percentile-ranked ratio
      6. Consolidation tightness — range width / ATR, percentile-ranked
      7. VCP structure score — geometric mean of normalized sub-scores
      8. Position gate — score decays if price is far from upper range
      9. Cross-regime normalization — all sub-scores are percentile ranks, so
         volatile and quiet stocks are treated identically

    Returns dict with:
        vcp_score        : 0–1 composite VCP score (1 = perfect VCP)
        vcp_pullback_n   : number of pullbacks detected
        vcp_contraction  : pullback contraction slope (normalized 0–1)
        vcp_vol_comp     : volatility compression percentile (0–1, 1=most compressed)
        vcp_vol_dryup    : volume dry-up during consolidation (0–1)
        vcp_tightness    : range tightness percentile (0–1, 1=tightest)
        vcp_position     : price position in recent range (0–1, 1=near highs)
        vcp_detected     : bool — True when pattern is credibly forming
    """
    _NEUTRAL = {
        "vcp_score": 0.0, "vcp_pullback_n": 0, "vcp_contraction": 0.5,
        "vcp_vol_comp": 0.5, "vcp_vol_dryup": 0.5, "vcp_tightness": 0.5,
        "vcp_position": 0.5, "vcp_detected": False,
    }

    # Minimum bars needed: enough history to build distributions
    if len(c) < 60:
        return _NEUTRAL

    # ── STEP 1: DYNAMIC SWING WINDOW ──────────────────────────────────────────
    # Window = ATR / median_daily_range, scaled to a plausible lookback.
    # Intuition: stocks with large ATR relative to their daily range have longer
    # volatility cycles, so a wider swing window is appropriate.
    # No fixed numbers: all parameters are derived from this stock's own series.
    hist_c = c.iloc[:-1]   # no look-ahead: exclude the current live bar
    hist_h = h.iloc[:-1]
    hist_l = l.iloc[:-1]
    hist_v = v.iloc[:-1]
    hist_atr = atr.iloc[:-1]

    daily_range = (hist_h - hist_l).replace(0, np.nan).dropna()
    if len(daily_range) < 20:
        return _NEUTRAL

    median_range = float(daily_range.tail(60).median())
    current_atr  = float(hist_atr.dropna().tail(1).iloc[0]) if len(hist_atr.dropna()) >= 1 else median_range
    if median_range <= 0 or current_atr <= 0:
        return _NEUTRAL

    # atr_cycle_ratio: how many daily ranges fit in one ATR
    atr_cycle_ratio = current_atr / (median_range + 1e-9)
    # Swing window: clip to [3, 25] bars — from the stock's own volatility cycle
    # Low ratio stock (ATR ≈ daily range) → small window (short-cycle)
    # High ratio stock (ATR >> daily range) → larger window (long-cycle)
    _raw_window = int(round(float(np.clip(atr_cycle_ratio * 5.0, 3.0, 25.0))))
    swing_window = max(3, min(25, _raw_window))

    # ── STEP 2: SWING DETECTION + PULLBACK EXTRACTION ────────────────────────
    # Swing high: local max within ±swing_window bars.
    # Swing low:  local min within ±swing_window bars.
    # Use rolling max/min with center=True — efficient, no look-ahead in HISTORY.
    n = len(hist_c)
    if n < swing_window * 4:
        return _NEUTRAL

    # Rolling max/min over 2*swing_window+1 bars (centred)
    roll_max = hist_h.rolling(2 * swing_window + 1, center=True).max()
    roll_min = hist_l.rolling(2 * swing_window + 1, center=True).min()

    is_swing_high = (hist_h == roll_max)
    is_swing_low  = (hist_l == roll_min)

    sh_idx = hist_h.index[is_swing_high].tolist()
    sl_idx = hist_l.index[is_swing_low].tolist()

    # Extract pullbacks: each (swing_high, subsequent_swing_low) pair
    pullbacks = []   # list of (depth, sh_pos, sl_pos)
    for shi in sh_idx:
        sh_pos = hist_h.index.get_loc(shi)
        sh_val = float(hist_h.loc[shi])
        # Find the next swing low AFTER this swing high
        subsequent_lows = [sli for sli in sl_idx
                           if hist_l.index.get_loc(sli) > sh_pos]
        if not subsequent_lows:
            continue
        sli = subsequent_lows[0]
        sl_pos = hist_l.index.get_loc(sli)
        sl_val = float(hist_l.loc[sli])
        if sh_val <= 0:
            continue
        depth = (sh_val - sl_val) / (sh_val + 1e-9)
        if depth < 0 or depth > 0.99:   # reject corrupt bars (halts, data errors)
            continue
        pullbacks.append({
            "depth": depth,
            "sh_pos": sh_pos,
            "sl_pos": sl_pos,
            "sh_val": sh_val,
            "sl_val": sl_val,
        })

    # Build full pullback history for percentile benchmarks (from all detected swings)
    all_depths = [p["depth"] for p in pullbacks]

    # Use only the most recent pullbacks for VCP analysis
    # Window: last N pullbacks where N is derived from how many typically fit
    # in the stock's 60-bar history. More volatile → more swings → more pullbacks.
    n_total_pbs = len(pullbacks)
    if n_total_pbs < 2:
        return _NEUTRAL

    # Take the most recent pullbacks (up to 6, derived: 60 bars / swing_window ≈ 2-20)
    _max_recent = max(2, min(6, int(60 / (swing_window + 1e-9))))
    recent_pbs = pullbacks[-_max_recent:]
    n_recent = len(recent_pbs)

    # ── STEP 3: CONTRACTION ANALYSIS ─────────────────────────────────────────
    # Fit OLS line to pullback depths in chronological order.
    # Negative slope = pullbacks are getting shallower = contracting = bullish.
    depths_arr = np.array([p["depth"] for p in recent_pbs], dtype=float)
    x_arr      = np.arange(len(depths_arr), dtype=float)

    if len(depths_arr) >= 2:
        # OLS: slope = (n*Σxy - Σx*Σy) / (n*Σx² - (Σx)²)
        _n = len(x_arr)
        _sx = x_arr.sum(); _sy = depths_arr.sum()
        _sxy = (x_arr * depths_arr).sum(); _sx2 = (x_arr**2).sum()
        _denom = _n * _sx2 - _sx**2
        raw_slope = float((_n * _sxy - _sx * _sy) / (_denom + 1e-9)) if _denom != 0 else 0.0
    else:
        raw_slope = 0.0

    # Normalize slope against the stock's own pullback slope history.
    # Build rolling slope history using ALL detected pullbacks (not just recent).
    # For each window of _max_recent consecutive pullbacks, compute its slope.
    slope_history = []
    if len(all_depths) >= _max_recent + 1:
        _d_arr = np.array(all_depths, dtype=float)
        _x_ref = np.arange(_max_recent, dtype=float)
        _n_ref = _max_recent
        _sx_r  = _x_ref.sum(); _sx2_r = (_x_ref**2).sum()
        _denom_r = _n_ref * _sx2_r - _sx_r**2
        for _i in range(len(_d_arr) - _max_recent + 1):
            _yw = _d_arr[_i: _i + _max_recent]
            _sxy_w = (_x_ref * _yw).sum(); _sy_w = _yw.sum()
            _sl = float((_n_ref * _sxy_w - _sx_r * _sy_w) / (_denom_r + 1e-9)) \
                  if _denom_r != 0 else 0.0
            slope_history.append(_sl)

    if len(slope_history) >= 5:
        _slope_arr = np.array(slope_history)
        # Percentile rank: what fraction of historical slopes are WORSE (less negative)?
        # Lower (more negative) slope = better contraction → high percentile = low raw_slope
        contraction_pct = float((np.array(slope_history) >= raw_slope).mean())
    else:
        # Fallback: simple sign-based scoring when history is thin
        # Negative slope = some contraction → 0.65 neutral-positive
        contraction_pct = 0.65 if raw_slope < 0 else 0.35

    # ── STEP 4: VOLATILITY COMPRESSION ───────────────────────────────────────
    # ATR_short / ATR_long percentile rank — identical logic to the existing
    # vc_pts factor but computed WITHIN the VCP context (uses hist_atr).
    _tr = pd.concat([hist_h - hist_l,
                     (hist_h - hist_c.shift(1)).abs(),
                     (hist_l - hist_c.shift(1)).abs()], axis=1).max(axis=1)
    _atr_short = _tr.rolling(5).mean()
    _atr_long  = _tr.rolling(20).mean()
    _vc_ratio_vcp = (_atr_short / (_atr_long.replace(0, np.nan))).dropna()

    if len(_vc_ratio_vcp) >= 10:
        _current_vc = float(_vc_ratio_vcp.iloc[-1])
        _hist_vc    = _vc_ratio_vcp.iloc[:-1]
        # LOW ratio = compressed. Percentile rank of current vs history.
        vc_pct_vcp = float((_hist_vc >= _current_vc).mean())   # fraction worse (higher) than current
    else:
        vc_pct_vcp = 0.5

    # ── STEP 5: VOLUME DRY-UP DURING CONSOLIDATION ───────────────────────────
    # Measure volume trend DURING the most recent swing (consolidation phase).
    # If a recent swing_low was detected, measure vol from swing_high to swing_low.
    # Fallback to the last swing_window bars if no recent swing found.
    vcp_vol_dryup = 0.5   # neutral default
    if recent_pbs:
        _last_pb = recent_pbs[-1]
        _sh_p = _last_pb["sh_pos"]
        _sl_p = _last_pb["sl_pos"]
        if _sl_p > _sh_p:
            _consol_vol = hist_v.iloc[_sh_p:_sl_p + 1]
        else:
            _consol_vol = hist_v.tail(swing_window)
        _consol_vol = _consol_vol.replace(0, np.nan).dropna()

        if len(_consol_vol) >= 3 and len(hist_v.dropna()) >= 20:
            _consol_avg  = float(_consol_vol.mean())
            _baseline_v  = float(hist_v.replace(0, np.nan).dropna().tail(60).mean())

            if _baseline_v > 0:
                _vol_ratio_vcp = _consol_avg / (_baseline_v + 1e-9)
                # Rolling vol ratio history
                _vr_history = (hist_v.replace(0, np.nan)
                               .rolling(max(3, len(_consol_vol)))
                               .mean()
                               .dropna()
                               / (_baseline_v + 1e-9)).dropna()

                if len(_vr_history) >= 5:
                    # LOW ratio = drying up = bullish
                    vcp_vol_dryup = float((_vr_history >= _vol_ratio_vcp).mean())
                else:
                    vcp_vol_dryup = float(np.clip(1.0 - _vol_ratio_vcp, 0.0, 1.0))

    # ── STEP 6: CONSOLIDATION TIGHTNESS ──────────────────────────────────────
    # Range of the most recent consolidation, normalized by ATR.
    # Tighter range relative to ATR = higher score.
    vcp_tightness = 0.5
    if recent_pbs:
        _last_pb = recent_pbs[-1]
        _sh_v = _last_pb["sh_val"]
        _sl_v = _last_pb["sl_val"]
        _consol_range = abs(_sh_v - _sl_v)

        if current_atr > 0:
            _range_atr = _consol_range / (current_atr + 1e-9)
            # Build rolling range/ATR history
            _rng_hist_series = (
                (hist_h.rolling(2 * swing_window).max() - hist_l.rolling(2 * swing_window).min())
                / (hist_atr.dropna().reindex(hist_h.index).ffill() + 1e-9)
            ).dropna()

            if len(_rng_hist_series) >= 10:
                # LOW range/ATR = tight = good
                vcp_tightness = float((_rng_hist_series >= _range_atr).mean())
            else:
                # Fallback: normalize within observed range min-max
                _rmin = float(_rng_hist_series.min()) if len(_rng_hist_series) > 0 else 0.0
                _rmax = float(_rng_hist_series.max()) if len(_rng_hist_series) > 0 else 10.0
                _rmax = max(_rmax, _rmin + 0.1)
                vcp_tightness = float(np.clip(1.0 - (_range_atr - _rmin) / (_rmax - _rmin), 0.0, 1.0))

    # ── STEP 7: VCP STRUCTURE SCORE — GEOMETRIC MEAN ─────────────────────────
    # Geometric mean treats all sub-signals as multiplicative gates:
    # if ANY one is zero, the composite is zero. This is correct for VCP:
    # all four conditions must be present simultaneously.
    # Each sub-score is already 0–1 percentile rank — no fixed constants.
    _sub_scores = np.array([
        contraction_pct,   # pullback contraction
        vc_pct_vcp,        # volatility compression
        vcp_vol_dryup,     # volume dry-up
        vcp_tightness,     # consolidation tightness
    ], dtype=float)
    _sub_scores = np.clip(_sub_scores, 1e-9, 1.0)   # avoid log(0)
    vcp_raw_score = float(np.exp(np.log(_sub_scores).mean()))   # geometric mean

    # ── STEP 8: POSITION GATE ─────────────────────────────────────────────────
    # VCP is bullish only when price is near the UPPER region of the recent range.
    # Price near lows = different pattern (distribution, not coiling before breakout).
    # Position score: (current_price - recent_low) / (recent_high - recent_low)
    # Percentile-rank vs own rolling position history.
    _ltp = float(c.iloc[-1])
    _lookback_pos = min(60, len(hist_c))
    _pos_hi = float(hist_h.tail(_lookback_pos).max())
    _pos_lo = float(hist_l.tail(_lookback_pos).min())
    _pos_range = _pos_hi - _pos_lo
    if _pos_range > 0:
        _price_pos = (_ltp - _pos_lo) / (_pos_range + 1e-9)
        # Rolling position history
        _pos_series = ((hist_c - hist_l.rolling(_lookback_pos).min()) /
                       (hist_h.rolling(_lookback_pos).max()
                        - hist_l.rolling(_lookback_pos).min() + 1e-9)).dropna()
        if len(_pos_series) >= 10:
            vcp_position = float((_pos_series <= _price_pos).mean())   # CDF rank
        else:
            vcp_position = float(np.clip(_price_pos, 0.0, 1.0))
    else:
        _price_pos  = 0.5
        vcp_position = 0.5

    # Apply position gate: score scales from 0 (price at lows) to 1 (price at highs).
    # Uses the stock's own position distribution, not a fixed 0.75 threshold.
    # The gate is multiplicative — VCP at the bottom of a range scores near zero.
    vcp_score = vcp_raw_score * vcp_position

    # ── STEP 9: VCP DETECTED FLAG ─────────────────────────────────────────────
    # "Detected" = composite score is in the top quartile of its own historical range.
    # Build a rough expected distribution: if all sub-scores were median (0.5),
    # geometric mean = 0.5, position = 0.5, score = 0.25. Top quartile ≈ 0.50+.
    # Use the stock's own position history median as the detection threshold.
    _detection_threshold = float(vcp_position * 0.40)   # relative to position quality
    vcp_detected = bool(vcp_score >= _detection_threshold and n_recent >= 2
                        and contraction_pct >= 0.55 and vc_pct_vcp >= 0.55)

    return {
        "vcp_score":       round(vcp_score,       4),
        "vcp_pullback_n":  n_recent,
        "vcp_contraction": round(contraction_pct, 4),
        "vcp_vol_comp":    round(vc_pct_vcp,      4),
        "vcp_vol_dryup":   round(vcp_vol_dryup,   4),
        "vcp_tightness":   round(vcp_tightness,   4),
        "vcp_position":    round(vcp_position,    4),
        "vcp_detected":    vcp_detected,
    }


def volume_surge(v_today, v_series, window=20):
    """Returns ratio of today's volume to N-day average."""
    avg = v_series.tail(window).mean()
    if avg == 0:
        return 0.0
    return float(v_today) / float(avg)

def _robust_iqr_width(s: pd.Series, fallback: float = 1.5) -> float:
    """
    Robust σ estimator via IQR — module-level function (Fix 27).
    Used in relative_strength and anywhere else needing a tanh saturation width
    from a distribution of z-scores. Hoisted from the nested _robust_w inside
    relative_strength which was being redefined on every call to that function.
    """
    s = s.dropna()
    if len(s) < 10:
        return fallback
    q75, q25 = np.percentile(s, 75), np.percentile(s, 25)
    return float(max((q75 - q25) / 1.35, 0.3))


def relative_strength(c_series, nifty_r5, nifty_r20, window5=5, window20=20, regime="BULL"):
    """
    Absolute vol-normalised alpha vs Nifty. Returns 0-1 score.
    tanh width is self-calibrated from the stock's own alpha distribution.
    Blend of 5d and 20d is regime-adaptive (regime passed explicitly — no global mkt read).
    """
    if len(c_series) < 23:
        return 0.5
    # Skip-1: use previous-day close as end point — same convention as CS-RS computation.
    # Signal-day close (iloc[-1]) is the live-patched partial-session price in live mode,
    # and the exact signal-bar close in BT mode. Either way, the 5d/20d return ending at
    # the day BEFORE the signal is the cleaner measure of accumulated momentum.
    base_6  = float(c_series.iloc[-7])  if len(c_series) >= 7  else 0
    base_21 = float(c_series.iloc[-22]) if len(c_series) >= 22 else 0
    end_p   = float(c_series.iloc[-2])  # skip-1: yesterday's close
    stock_r5  = float(end_p / base_6  - 1) if base_6  != 0 else 0
    stock_r20 = float(end_p / base_21 - 1) if base_21 != 0 else 0
    r5_beat   = stock_r5  - (nifty_r5  or 0)
    r20_beat  = stock_r20 - (nifty_r20 or 0)

    daily_rets = c_series.pct_change().dropna()
    ret_std_5  = float(daily_rets.tail(5).std())  if len(daily_rets) >= 5  else 0.01
    ret_std_20 = float(daily_rets.tail(20).std()) if len(daily_rets) >= 20 else 0.01
    ret_std_5  = max(ret_std_5,  0.001)
    ret_std_20 = max(ret_std_20, 0.001)

    alpha_5  = r5_beat  / (ret_std_5  * np.sqrt(5)  + 1e-9)
    alpha_20 = r20_beat / (ret_std_20 * np.sqrt(20) + 1e-9)

    # Self-calibrated tanh width from rolling alpha distribution
    _alpha5_hist  = (c_series.pct_change(5).dropna()  - (nifty_r5  or 0)) / (ret_std_5  * np.sqrt(5)  + 1e-9)
    _alpha20_hist = (c_series.pct_change(20).dropna() - (nifty_r20 or 0)) / (ret_std_20 * np.sqrt(20) + 1e-9)
    _w5  = _robust_iqr_width(_alpha5_hist)
    _w20 = _robust_iqr_width(_alpha20_hist)

    rs5  = float(0.5 * (1.0 + np.tanh(alpha_5  / _w5)))
    rs20 = float(0.5 * (1.0 + np.tanh(alpha_20 / _w20)))

    # Regime-adaptive blend: 20d dominates in ALL regimes to prevent 5d mean-reversion
    # inversion (5d winners on NSE tend to UNDERPERFORM next 5d — see regime weights comment
    # in score_stock_dual). Weights now match _w5d/_w20d used in the CS-RS blend above.
    # regime is passed explicitly — never reads global mkt state
    _regime_rs = regime
    if _regime_rs == "BULL":
        return rs5 * 0.25 + rs20 * 0.75   # was 0.65/0.35 — 5d dominance caused quintile inversion
    elif _regime_rs == "BEAR":
        return rs5 * 0.15 + rs20 * 0.85   # 20d even more dominant in bear (5d is noise)
    else:
        return rs5 * 0.20 + rs20 * 0.80   # CHOP: 20d still dominates


# ── MAIN SCORING FUNCTION v3 ─────────────────────────────────
# 10 factors, 100 pts. Adds vs v2:
#   RS_Sector, InstVol footprint, VolContraction ATR5/ATR20,
#   flat-resistance in coil, ATR% potential, gap filter,
#   time horizon, entry/target/stop, EMI ranking metric.

def score_stock_dual(df_raw, live, nifty_r5, nifty_r20, ticker="", bt_mode=False,
                     bt_regime=None, bt_sector_returns=None, bt_sector_returns_10d=None,
                     bt_vix_level=None, bt_vix_falling=True, bt_vix_median=14.5, bt_vix_sigma=4.5,
                     bt_nifty_above_20dma=True, bt_rs_div_hist=None):
    """
    bt_mode=True  → historical backtest path.  Disables every live-session side-effect:
      • Volume scaling uses elapsed_frac=1.0 (bar is a complete daily candle)
      • rs_div_hist uses a clean per-call dict — no session_state read/write
      • score_cache is never consulted or written
      • market context uses bt_* parameters, not the live mkt global
    bt_mode=False → live scoring path (default, unchanged behaviour)
    """
    if len(df_raw) < 60:
        return None

    df = df_raw.copy()
    # ltp = live price (for display, entry, stop calc only — NOT used in factor scoring)
    # ltp_score = T-1 close (set after hist slice below — used for ALL factor scoring)
    _live_ltp = live.get("ltp");   ltp     = float(_live_ltp if _live_ltp is not None else df["close"].iloc[-1])
    _live_vol = live.get("volume"); day_vol = float(_live_vol if _live_vol is not None else df["volume"].iloc[-1])
    _live_hi  = live.get("high");   day_hi  = float(_live_hi  if _live_hi  is not None else df["high"].iloc[-1])
    _live_lo  = live.get("low");    day_lo  = float(_live_lo  if _live_lo  is not None else df["low"].iloc[-1])
    _live_o   = live.get("open");   day_o   = float(_live_o   if _live_o   is not None else df["open"].iloc[-1])

    df.at[df.index[-1], "close"]  = ltp
    df.at[df.index[-1], "high"]   = max(float(df["high"].iloc[-1]),  day_hi)
    df.at[df.index[-1], "low"]    = min(float(df["low"].iloc[-1]),   day_lo)

    # ── VOLUME SCALING ──
    # BUG-1 FIX: In bt_mode the bar is a COMPLETE historical daily candle.
    # elapsed_frac must be 1.0 — applying wall-clock time to historical bars
    # inflates volume for whatever time-of-day the backtest is RUN, not the
    # time-of-day the signal OCCURRED.  That creates systematic score inflation
    # for high-volume breakout stocks and is the primary driver of the
    # top-5 / bottom-5 inversion in replay backtests.
    _NSE_OPEN_MIN  = 9 * 60 + 15    # 09:15 in minutes from midnight
    _NSE_CLOSE_MIN = 15 * 60 + 30   # 15:30 in minutes from midnight
    _SESSION_MINS  = _NSE_CLOSE_MIN - _NSE_OPEN_MIN   # 375 minutes

    if bt_mode:
        # Historical bar: always treat as a complete session
        _elapsed      = _SESSION_MINS
        _elapsed_frac = 1.0
        day_vol_scaled = day_vol
    else:
        # Live intraday: scale partial-session volume to full-session estimate
        _now_ist = datetime.utcnow() + timedelta(hours=5, minutes=30)
        _now_min = _now_ist.hour * 60 + _now_ist.minute
        _elapsed = _now_min - _NSE_OPEN_MIN
        # Floor at 0.10 (37.5 min) so early-session candles don't produce absurd multipliers.
        # Also require _elapsed >= 6 minutes before scaling — the open-auction period
        # (09:15–09:21) sees erratic volume prints that should not be extrapolated.
        _elapsed_frac = float(np.clip(_elapsed / _SESSION_MINS, 0.10, 1.0))
        if _elapsed >= 6 and _elapsed < _SESSION_MINS:
            day_vol_scaled = day_vol / _elapsed_frac
        else:
            day_vol_scaled = day_vol   # pre-market, post-market, or open-auction: use raw

    df.at[df.index[-1], "volume"] = day_vol_scaled

    # Reassign series AFTER live patches so all downstream indicators use scaled volume
    c = df["close"]; h = df["high"]; l = df["low"]
    v = df["volume"]; o = df["open"]
    # day_vol_scaled is the full-session volume estimate; use it wherever "today's vol" is needed
    day_vol = day_vol_scaled   # shadow the raw value; raw is no longer needed below

    # Historical slice — BEFORE today (prevents look-ahead)
    hist = df.iloc[:-1]
    hc = hist["close"]; hh = hist["high"]; hl = hist["low"]; hv = hist["volume"]

    # ── FIX-04: WEEKLY + MONTHLY MTF COMPRESSION ────────────────────────────
    # Resample existing daily hist to weekly and monthly OHLCV — no new API calls.
    # Three-timeframe alignment: daily + weekly + monthly all compressed = strongest signal.
    # Monthly compression catches stocks that have been range-bound for 1-3 months —
    # a coil at this scale typically precedes a much larger move than a daily-only coil.
    # False-positive source: a stock "quiet" daily but mid-range in a 15% weekly swing.
    # When both daily AND weekly ATR compression confirm, false positives drop sharply.
    _mtf_bonus = 0.0
    _wk_compressed = False
    _wk_vc_pct = 0.5
    _mo_compressed = False
    _mo_vc_pct = 0.5
    try:
        if "time" in hist.columns:
            _hist_idx = hist.set_index(pd.to_datetime(hist["time"]))
        elif isinstance(hist.index, pd.DatetimeIndex):
            _hist_idx = hist
        else:
            _hist_idx = pd.DataFrame()

        def _resample_ohlcv(src, freq):
            if src.empty or len(src) < 5:
                return pd.DataFrame()
            try:
                return src.resample(freq).agg(
                    {"open": "first", "high": "max", "low": "min",
                     "close": "last", "volume": "sum"}
                ).dropna()
            except Exception:
                return pd.DataFrame()

        _hist_w = _resample_ohlcv(_hist_idx, "W")
        _hist_m = _resample_ohlcv(_hist_idx, "ME") if not _hist_idx.empty else pd.DataFrame()
        if _hist_m.empty:
            try:
                _hist_m = _resample_ohlcv(_hist_idx, "M")
            except Exception:
                _hist_m = pd.DataFrame()

        def _vc_pct_for(ohlcv_df, min_bars=10):
            """Returns ATR5/ATR20 percentile for a resampled OHLCV df."""
            if len(ohlcv_df) < min_bars:
                return 0.5, False
            _tr = pd.concat([
                ohlcv_df["high"] - ohlcv_df["low"],
                (ohlcv_df["high"] - ohlcv_df["close"].shift(1)).abs(),
                (ohlcv_df["low"]  - ohlcv_df["close"].shift(1)).abs()
            ], axis=1).max(axis=1)
            _a5  = float(_tr.rolling(5).mean().iloc[-1])
            _a20 = float(_tr.rolling(20).mean().iloc[-1]) if len(_tr) >= 20 else float(_tr.mean())
            if _a20 == 0:
                return 0.5, False
            _vc_r = _a5 / _a20
            _vc_h = (_tr.rolling(5).mean() / (_tr.rolling(20).mean() + 1e-9)).dropna()
            if len(_vc_h) >= min_bars:
                _pct = float((_vc_h >= _vc_r).mean())
            else:
                _pct = 0.5
            return _pct, _pct <= 0.35

        if len(_hist_w) >= 22:
            _wk_vc_pct, _wk_compressed = _vc_pct_for(_hist_w, min_bars=22)
        if len(_hist_m) >= 6:
            _mo_vc_pct, _mo_compressed = _vc_pct_for(_hist_m, min_bars=6)

    except Exception:
        pass

    # ── PREDICTIVE ARCHITECTURE: score on T-1 EOD, not T-0 EOD ──────────────
    # ALL factor computation uses the HISTORICAL slice (hc/hh/hl/hv = up to T-1).
    # The signal bar (index[-1]) provides only: open price (for gap detection).
    # Why: scoring on the complete signal bar = the stock already moved = buying after.
    # Scoring on T-1 = finding stocks that are COILING before the move.
    #
    # EMAs, ATR, RSI: computed on `hc` (historical closes only), NOT full `c`.
    # This means e9_v, e20_v, ltp_for_scoring = yesterday's close, not today's.
    # The setup classifier and all factor scores use T-1 state.
    # live ltp_score is still used for display and entry calculation only.
    e9   = ema(hc, 9);  e20 = ema(hc, 20);  e50 = ema(hc, 50)
    e5   = ema(hc, 5)
    atr  = atr14(hist)   # ATR on hist slice
    _rsi_period = int(st.session_state.get("rsi_period", 7))
    rsi  = rsi_wilder(hc, _rsi_period)

    e9_v  = float(e9.iloc[-1]);  e20_v = float(e20.iloc[-1])
    e50_v = float(e50.iloc[-1]); atr_v = float(atr.iloc[-1])
    rsi_v = float(rsi.iloc[-1]); rsi_p = float(rsi.iloc[-2])
    e9_y  = float(e9.iloc[-2]);  e20_y = float(e20.iloc[-2])

    # ltp_for_scoring: T-1 close (the state we actually observe before trading)
    # ltp_score (live): today's live price, used for display/entry only
    ltp_score = float(hc.iloc[-1])   # T-1 close = scoring price

    vol_ma20 = float(hv.rolling(20).mean().iloc[-1]) if len(hv) >= 20 else float(hv.mean())
    atr_pct  = (atr_v / ltp_score) * 100 if ltp_score > 0 else 0

    # ── COMPUTATIONAL GUARDS ──
    if atr_v == 0 or vol_ma20 == 0:   return None
    if ltp_score <= 0:                  return None

    _soft_penalty = 0.0

    # RSI overbought penalty — uses T-1 RSI (already on hc)
    _rsi_hist_full = rsi_wilder(hc, _rsi_period)
    _rsi_p90       = float(_rsi_hist_full.tail(60).quantile(0.90)) if len(_rsi_hist_full) >= 20 else 80.0
    if rsi_v > _rsi_p90:
        _rsi_ob_z      = (rsi_v - _rsi_p90) / max(float(_rsi_hist_full.tail(20).std()), 1.0)
        _soft_penalty += float(np.clip(8.0 * np.tanh(_rsi_ob_z), 0.0, 15.0))
    # RSI universal floor removed: penalised quality uptrend stocks (RSI 58-70)
    # without predictive power. Per-stock p90 penalty above is sufficient.

    # FIX A: Low-volume penalty REMOVED.
    # The _vol_p05 penalty contradicted vol_quiet_pts (40% primary factor): a stock at
    # the 3rd-percentile of its own volume was simultaneously penalised (up to −12 pts)
    # AND rewarded (up to +14 pts), collapsing differentiation.
    # The ADV turnover penalty (_ADV_THRESHOLD = 2e7) already handles illiquid stocks.
    # The vol-quiet bonus handles accumulation quality. They are redundant+contradictory.
    _prev_vol = float(hv.iloc[-1])   # T-1 volume (last bar of hist) — kept for reference below

    # Low-ATR% penalty
    if atr_pct < 0.5:
        _soft_penalty += float(np.clip((0.5 - atr_pct) * 10.0, 0.0, 5.0))

    # Turnover liquidity penalty — use T-1 median close
    _price_for_adv = float(hc.tail(20).median()) if len(hc) >= 20 else ltp_score
    _adv_turnover  = vol_ma20 * _price_for_adv
    _ADV_THRESHOLD = 2e7
    if _adv_turnover < _ADV_THRESHOLD:
        _liq_ratio     = _adv_turnover / (_ADV_THRESHOLD + 1e-9)
        _soft_penalty += float(np.clip(15.0 * (1.0 - _liq_ratio), 0.0, 15.0))

    # Gap penalty: T-1 open vs T-2 close (gap that already happened in hist)
    prev_close = float(hc.iloc[-1])
    prev_open  = float(hist["open"].iloc[-1]) if "open" in hist.columns else prev_close
    if prev_close > 0 and atr_v > 0:
        _gap = prev_open - float(hc.iloc[-2]) if len(hc) >= 2 else 0.0
        # FIX F: Gap history was using |high[T-1] - close[T-2]| / ATR which is NOT a gap.
        # A gap = |open[T] - close[T-1]|. Using hh vs hc.shift(1) measured a day-over-day
        # high move — a completely different quantity — making the p90 threshold wrong.
        # Fix: use hist open series shifted by 1 to get open[T] - close[T-1] per bar.
        if "open" in hist.columns:
            _ho = hist["open"]
            _gap_hist_atr = ((_ho - hc.shift(1)).abs() / (atr + 1e-9)).dropna().tail(60)
        else:
            # Fallback if open not available: use close-to-close as proxy
            _gap_hist_atr = ((hc - hc.shift(1)).abs() / (atr + 1e-9)).dropna().tail(60)
        _gap_p90 = float(_gap_hist_atr.quantile(0.90)) if len(_gap_hist_atr) >= 20 else 2.0
        _gap_abs_atr = abs(_gap) / (atr_v + 1e-9)
        if _gap_abs_atr > _gap_p90:
            _gap_excess    = _gap_abs_atr - _gap_p90
            _soft_penalty += float(np.clip(8.0 * np.tanh(_gap_excess / (_gap_p90 + 1e-9)), 0.0, 15.0))

    # ── TODAY'S GAP-UP ATR (computed early, penalty applied after setup classification) ──
    # We compute the gap size here because day_o and ltp_score are available.
    # The actual penalty is applied below after setup_type is known (Reversal exempt).
    _today_gap_atr = (day_o - ltp_score) / (atr_v + 1e-9)   # positive = gap up today
    _gap_up_penalty = 0.0   # initialised here, set after setup_type block below

    # ── BASE RANGE (all on hist — T-1 and earlier) ──
    base_hi  = float(hh.tail(20).max())
    base_lo  = float(hl.tail(20).min())
    base_rng = base_hi - base_lo + 1e-9
    breakout_ext = (ltp_score - base_hi) / (atr_v + 1e-9)

    # ── FIX-07: ROUND-NUMBER RESISTANCE ──────────────────────────────────────
    # NSE option OI concentrates at round strikes (₹100, ₹250, ₹500, ₹1000 etc).
    # A breakout above a round number where price has coiled 3+ times is structurally
    # stronger than a break of an arbitrary high — option market makers stop pinning.
    _ROUND_LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 750,
                     1000, 1250, 1500, 2000, 2500, 3000, 4000, 5000]
    _round_match   = [x for x in _ROUND_LEVELS if base_hi > 0 and abs(base_hi - x) / x < 0.005]
    _is_round_res  = len(_round_match) > 0
    _round_touches = int(((hh >= base_hi * 0.997) & (hh <= base_hi * 1.003)).sum()) if _is_round_res else 0
    _round_bonus   = 0.0
    if _is_round_res and _round_touches >= 3:
        # Score scales with number of touches: 3→0.75, 4→1.5, 5→2.25, capped at 3
        _round_bonus = round(float(np.clip((_round_touches - 2) * 0.75, 0.0, 3.0)), 1)

    # ── SETUP CLASSIFICATION ──
    # History requirement: need at least 60 bars for meaningful indicators.
    # Stocks with < 60 bars are genuinely unscoreable (not just undesirable).
    if len(hc) < 60:
        return None   # truly insufficient — can't compute EMA50, ATR14, RSI7

    # SMA200 structural trend — H-4 FIX: use ATR-normalised gap instead of 3% fixed tolerance
    # Old: ltp_score > sma200 × 0.97 — 3% means wildly different things across price levels
    # New: penalty onset = ltp_score < sma200 - 0.5×ATR (one half-ATR below = structurally broken)
    # This adapts to the stock's own volatility — a calm large-cap and a volatile small-cap
    # get the same structural signal at equivalent risk-adjusted distances from the trend.
    _n_bars_sma  = min(200, len(hc))
    _sma200      = float(hc.tail(_n_bars_sma).mean())
    _sma200_gap_atr = (ltp_score - _sma200) / (atr_v + 1e-9)   # in ATR units
    # penalty onset = -0.5 ATR below SMA200 (half-ATR = one volatility unit of tolerance)
    if _sma200_gap_atr < -0.5:
        _sma_excess    = abs(_sma200_gap_atr + 0.5)   # how many ATR beyond the onset
        _soft_penalty += float(np.clip(10.0 * np.tanh(_sma_excess / 2.0), 0.0, 20.0))
    above_long_trend = ltp_score > _sma200 - 0.5 * atr_v

    # EMA proximity — ATR-normalised (not 0.97 price ratio)
    above_ema50  = ltp_score > e50_v - atr_v                                          # 1 ATR tolerance
    near_e9_y  = abs(ltp_score - e9_v)  / (atr_v + 1e-9) < 1.0
    near_e20_y = abs(ltp_score - e20_v) / (atr_v + 1e-9) < 1.0

    # Volume distribution — M-3 FIX: direct quantile, not Gaussian mu+1.5σ
    # Vol is right-skewed; Gaussian approx underestimates the 85th percentile.
    # Use full available history (up to 60 bars) for a stable estimate.
    vol_series_20 = hv.tail(20) if len(hv) >= 5 else hv
    vol_mu        = float(vol_series_20.mean()) if len(vol_series_20) > 0 else float(vol_ma20)
    vol_sigma     = float(vol_series_20.std())  if len(vol_series_20) > 1 else vol_mu * 0.3
    vol_sigma     = max(vol_sigma, vol_mu * 0.05)
    _vol_q_hist   = hv.tail(min(60, len(hv))) if len(hv) >= 10 else hv
    vol_bo_thresh = float(_vol_q_hist.quantile(0.85))   # 85th pct of own distribution

    # Setup classification — all thresholds derived from stock's own history
    # breakout_ext = (ltp_score - base_hi) / ATR:
    #   > 0   = above resistance (breakout territory)
    #   < 0   = below resistance (base/coil territory)
    #   >> 0  = overextended (chased, likely to mean-revert)
    #
    # Thresholds derived from this stock's own breakout_ext distribution:
    #   pullback_tolerance = stock's 10th percentile (how far it pulls into the base before recovering)
    #   overextension_onset = stock's 90th percentile (how far past resistance before it typically stalls)
    # Both are computed from the rolling hist slice — no look-ahead.
    # _ext_hist same sign as breakout_ext: (close-rolling_max)/atr
    _ext_hist = ((hc - hh.rolling(20).max().shift(1)) / (atr.iloc[:-1] + 1e-9)).dropna()
    _ext_p10  = float(_ext_hist.quantile(0.10)) if len(_ext_hist) >= 20 else -1.5
    _ext_p90  = float(_ext_hist.quantile(0.90)) if len(_ext_hist) >= 20 else  0.3
    _ext_p10  = min(_ext_p10, -0.3)
    _ext_p90  = max(_ext_p90,  0.5)

    # ── REVERSAL SETUP — checked FIRST, takes priority ───────────────────
    # Identifies deeply oversold stocks with panic capitulation volume that
    # show first signs of demand — the "washout + wick" pattern.
    # These are NOT momentum setups. They are mean-reversion candidates.
    # Scored on a completely separate scale — do not mix with Breakout/Pullback.
    #
    # Conditions (all on hist — no look-ahead):
    #   A. RSI(7) < 30  — deeply oversold by own history
    #   B. T-1 VolRatio ≥ 1.5  — capitulation / panic selling volume
    #   C. Price ≥ 2 ATR below 10-day high  — real washout, not a dip
    #   D. T-1 candle closes in upper 40% of its range  — demand tail / wick
    # Max vol ratio over last 3 bars — panic can happen 1-2 bars before we score
    _vol3 = hv.tail(3)
    _t1_vol_ratio_rev = float(_vol3.max()) / (vol_ma20 + 1e-9)
    _hi10d_rev        = float(hh.tail(10).max())
    _washout_depth    = (_hi10d_rev - ltp_score) / (atr_v + 1e-9)
    # _t1_close_pos always computed — used in scoring and return dict
    _t1_bar_range = float(hh.iloc[-1]) - float(hl.iloc[-1])
    _t1_close_pos = ((float(hc.iloc[-1]) - float(hl.iloc[-1]))
                     / (_t1_bar_range + 1e-9))
    # Reversal: no close_pos gate — panic bars close at lows by definition
    _is_reversal = (
        rsi_v < 40.0 and
        _t1_vol_ratio_rev >= 1.3 and
        _washout_depth >= 1.5
    )
    if _is_reversal:
        setup_type = "Reversal"
    elif breakout_ext >= _ext_p10 and breakout_ext <= _ext_p90 and day_vol_scaled >= vol_bo_thresh:
        # FIX E: Use day_vol_scaled (projected full-session estimate) not raw day_vol.
        # vol_bo_thresh is the 85th pct of historical FULL-SESSION volumes.
        # Comparing raw intraday volume against it caused false Breakout classification
        # for stocks with a morning volume spike before completing the day.
        setup_type = "Breakout"
    elif above_ema50 and (near_e9_y or near_e20_y):
        # FIX-C: Require an ACTUAL pullback — price must have dropped from a recent peak.
        # Without this, stocks in sideways drift at EMA qualify as "Pullback" setups.
        # Condition: ltp_score < 10d high by >= 0.3 ATR (a real dip happened).
        _hi10d       = float(hh.tail(10).max())
        _real_pullback = ltp_score < (_hi10d - 0.3 * atr_v)
        if _real_pullback:
            setup_type = "Pullback"
        else:
            # No real pullback: stock is at EMA but never really dipped.
            # Treat as Breakout (near resistance with no compression) — will get
            # penalised by low vol_dryup + overextension if applicable.
            setup_type = "Breakout"
    elif breakout_ext > _ext_p90:
        setup_type = "Breakout"
        _ext_excess    = (breakout_ext - _ext_p90) / (max(_ext_p90 - _ext_p10, 0.5))
        _soft_penalty += float(np.clip(10.0 * np.tanh(_ext_excess), 0.0, 15.0))
    elif breakout_ext >= _ext_p10 and breakout_ext <= _ext_p90:
        setup_type = "Breakout"   # near resistance, vol not yet — pre-breakout coil
    else:
        setup_type = "Pullback"

    # ── BUG-FIX: ALREADY-BROKE-OUT PENALTY ──────────────────────────────────
    # If T-1 was the breakout candle (highest-volume bar in 20d AND price at/through
    # resistance), the move already happened.  prox_pts will be 10 and the stock looks
    # "set up" when it is actually in post-breakout drift.  Penalise proportionally.
    # FIX 5: Extend penalty to cover stocks that have run far ABOVE base_hi even
    # without high T-1 volume (quiet multi-day drift above resistance = chasing).
    # Penalty onset: ltp_score > base_hi + 1 ATR (already significantly extended).
    if setup_type == "Breakout":
        # Sub-fix 5a: already-ran penalty for quiet drift above resistance
        _ext_above_resistance = (ltp_score - base_hi) / (atr_v + 1e-9)
        if _ext_above_resistance > 1.0:
            _drift_excess = _ext_above_resistance - 1.0
            _soft_penalty += float(np.clip(8.0 * np.tanh(_drift_excess), 0.0, 12.0))
    if setup_type == "Breakout" and ltp_score >= base_hi - 0.2 * atr_v:
        _t1_vol     = float(hv.iloc[-1])                       # T-1 volume
        _vol_rank   = float((hv.iloc[:-1] <= _t1_vol).mean())  # percentile of T-1 vol vs prior
        if _vol_rank >= 0.85:
            # T-1 was a high-volume bar while price is at/above resistance = already broke
            _already_broke_z = (_vol_rank - 0.85) / 0.15      # 0 at p85, 1 at p100
            _soft_penalty   += float(np.clip(12.0 * np.tanh(_already_broke_z * 2.0), 0.0, 18.0))

    # RSI>52 pullback penalty removed — data shows RSI is positively correlated
    # with forward return for Pullback setups on NSE Nifty50 (IC=+0.46).
    # The penalty was making high-RSI quality pullbacks (BEL, ONGC) score too low.

    # ── TODAY'S GAP-UP OPEN PENALTY (applied here — setup_type now known) ────
    # Gap-up = today's open significantly above T-1 close.
    # The move already happened at open. Buying into it = chasing.
    # Reversal setups are exempt — a gap-down into panic lows is the signal itself.
    # Penalty is tanh-scaled against this stock's own historical gap distribution
    # so the same ATR gap is treated differently for volatile vs calm stocks.
    if _today_gap_atr > 1.0 and setup_type != "Reversal":
        _open_series   = hist["open"] if "open" in hist.columns else hc
        _hist_gaps_atr = (((_open_series - hc.shift(1)).clip(lower=0)) / (atr + 1e-9)).dropna()
        _hist_gap_p75  = float(_hist_gaps_atr.quantile(0.75)) if len(_hist_gaps_atr) >= 20 else 1.0
        _hist_gap_p75  = max(_hist_gap_p75, 0.5)
        _gap_excess_z   = (_today_gap_atr - 1.0) / _hist_gap_p75
        _gap_up_penalty = float(np.clip(20.0 * np.tanh(_gap_excess_z), 0.0, 20.0))
        _soft_penalty  += _gap_up_penalty

    # ── UNIVERSE-LEVEL PARAMETER REGISTRY (M-1, M-4, M-5 FIX) ──
    # Fix 21+25: Load registry ONCE at function start. _tanh_w and other helpers
    # mutate the local _reg dict in place. We flush it ONCE to session_state at the
    # end of score_stock_dual, not on every individual helper invocation.
    # Old code wrote to session_state inside _tanh_w on every call — that caused
    # dozens of redundant session writes per stock per Streamlit rerun.
    _reg = dict(st.session_state.get("param_registry", {
        "tanh_w": [], "inst_sigma": [], "prox_lambda": [], "pullback_sigma": []
    }))   # shallow copy so mutations don't affect session_state mid-function

    def _tanh_w(z_series, fallback=None):
        """Robust σ of a z-score series → tanh saturation width.
        M-1 FIX: fallback is the median of observed widths across the universe.
        Fix 25: No session_state write here — caller flushes _reg once at function end."""
        s = z_series.dropna()
        _reg_vals = _reg.get("tanh_w", [])
        _fallback = float(np.median(_reg_vals)) if len(_reg_vals) >= 5 else 1.5
        if len(s) < 10:
            return _fallback
        q75, q25 = np.percentile(s, 75), np.percentile(s, 25)
        w = (q75 - q25) / 1.35
        w = float(max(w, 0.3))
        _reg_vals.append(w)
        _reg["tanh_w"] = _reg_vals[-200:]   # rolling 200-entry buffer (local mutation)
        return w

    # ═══════════════════════════════════════════════════════════════
    # RESOLVE MARKET CONTEXT
    # In bt_mode: use caller-supplied historical values (no look-ahead).
    # In live mode: use the cached mkt global.
    # This ensures every market-dependent factor (regime weights, breadth,
    # VIX adjustment) reflects conditions ON the signal date, not today.
    # ═══════════════════════════════════════════════════════════════
    if bt_mode:
        _regime                = bt_regime if bt_regime is not None else "BULL"
        _resolved_sect_ret     = bt_sector_returns     if bt_sector_returns     is not None else {}
        _resolved_sect_ret_10d = bt_sector_returns_10d if bt_sector_returns_10d is not None else {}
        _resolved_vix_level    = bt_vix_level
        _resolved_vix_falling  = bt_vix_falling
        _resolved_vix_median   = bt_vix_median
        _resolved_vix_sigma    = bt_vix_sigma
        _resolved_above_20dma  = bt_nifty_above_20dma
    else:
        _regime                = mkt.get("regime", "BULL")
        _resolved_sect_ret     = sector_returns
        _resolved_sect_ret_10d = sector_returns_10d
        _resolved_vix_level    = mkt.get("vix_level")
        _resolved_vix_falling  = mkt.get("vix_falling", True)
        _resolved_vix_median   = mkt.get("vix_median", 14.5)
        _resolved_vix_sigma    = mkt.get("vix_sigma",  4.5)
        _resolved_above_20dma  = mkt.get("nifty_above_20dma", True)

    # ═══════════════════════════════════════════════════════════════
    # REGIME WEIGHTS — F1 blend adapts to market regime
    # SIGNAL INVERSION FIX: 5d CS-RS is in the mean-reversion zone on NSE.
    # Stocks that outperformed the last 5 days tend to UNDERPERFORM the next 5.
    # 20d CS-RS is in the momentum zone — past 20d winners tend to continue.
    # Old weights (BULL: 65% 5d, 35% 20d) caused the quintile inversion.
    # New weights: 20d dominates in all regimes; 5d is a secondary acceleration signal.
    # ═══════════════════════════════════════════════════════════════
    if _regime == "BULL":
        _w5d, _w20d = 0.25, 0.75   # 20d momentum dominates; 5d adds recent acceleration
    elif _regime == "BEAR":
        _w5d, _w20d = 0.15, 0.85   # 20d even more dominant in bear (5d is pure noise)
    else:  # CHOP
        _w5d, _w20d = 0.20, 0.80   # 20d still dominates; slightly more 5d weight for rotation

    # ═══════════════════════════════════════════════════════
    # F1 — RS vs UNIVERSE (cross-sectional) + RS vs NIFTY (absolute)  (0-15 pts)
    # Cross-sectional rank is regime-agnostic: finds leaders in ANY environment.
    # Absolute alpha vs Nifty adds bull-phase confirmation.
    # Blend weights adapt to regime (see above).
    # ═══════════════════════════════════════════════════════

    _cs5  = st.session_state.cs_rs_5d.get(ticker,  None)
    _cs20 = st.session_state.cs_rs_20d.get(ticker, None)
    if _cs5 is not None and _cs20 is not None:
        cs_rs_score = _cs5 * _w5d + _cs20 * _w20d
    else:
        cs_rs_score = _cs5 if _cs5 is not None else (_cs20 if _cs20 is not None else 0.5)
    # RS SLOPE: cs5>cs20 = rank just improved = LEADING signal
    _rs_slope_score = float(np.clip(0.5 + ((_cs5 or 0.5) - (_cs20 or 0.5)) * 2.0, 0.0, 1.0))
    abs_rs_score = relative_strength(hc, nifty_r5, nifty_r20, regime=_regime)
    rs_score = cs_rs_score * 0.40 + _rs_slope_score * 0.35 + abs_rs_score * 0.25
    rs_pts   = round(rs_score * 15, 1)

    # ── RS ACCELERATION (true leading signal: 0-4 pts bonus) ──
    # velocity = EMA5 − EMA20 measures momentum speed.
    # acceleration = velocity.diff() — catches ignition BEFORE RS confirms.
    # tanh width self-calibrates from the stock's own acceleration history.
    _velocity      = e5 - e20
    _acceleration  = _velocity.diff()
    _acc_hist      = _acceleration.iloc[:-1]   # no look-ahead
    _acc_z_series  = (_acc_hist - _acc_hist.mean()) / (_acc_hist.std() + 1e-9)
    _acc_w         = _tanh_w(_acc_z_series)
    _acc_today_z   = float((_acceleration.iloc[-2] - _acc_hist.mean()) / (_acc_hist.std() + 1e-9)) \
                     if len(_acceleration) >= 2 else 0.0
    acc_score      = float(0.5 * (1.0 + np.tanh(_acc_today_z / _acc_w)))
    rs_accel_bonus = round(acc_score * 4, 1)   # 0-4 pts
    rs_accel       = float(_acceleration.iloc[-2]) if len(_acceleration) >= 2 else 0.0

    # ── MULTI-TIMEFRAME RS DIVERGENCE (new leading signal: 0-3 pts bonus) ──
    # When 5d CS-RS is improving FASTER than 20d CS-RS, money is rotating in NOW.
    # This identifies early-stage rotation before it shows up in price.
    # Score = percentile rank of (cs5 - cs20) gap over its own 60d history.
    # BUG-2 FIX: In bt_mode, never read or write st.session_state.rs_div_hist.
    # The live-session buffer accumulated across multiple live screener runs contains
    # RS divergence patterns from dates AFTER the backtest date, injecting look-ahead.
    # In bt_mode we accept a per-call accumulation (grows across stocks within ONE date,
    # then the caller resets it) but never carry history across test dates.
    _rs_div = (_cs5 - _cs20) if (_cs5 is not None and _cs20 is not None) else 0.0
    if bt_mode:
        # bt_rs_div_hist is a plain dict passed by the caller — no session_state I/O
        _bt_div_store = bt_rs_div_hist if bt_rs_div_hist is not None else {}
        _prev_divs    = _bt_div_store.get(ticker, [])
        _prev_divs    = (_prev_divs + [_rs_div])[-60:]
        _bt_div_store[ticker] = _prev_divs
        # NOTE: mutates the caller's dict in place — caller must pass a fresh dict per test date
    else:
        _rs_div_hist = st.session_state.get("rs_div_hist", {})
        _prev_divs   = _rs_div_hist.get(ticker, [])
        _prev_divs   = (_prev_divs + [_rs_div])[-60:]   # rolling 60-entry buffer
        _rs_div_hist[ticker] = _prev_divs
        st.session_state.rs_div_hist = _rs_div_hist
    if len(_prev_divs) >= 10:
        _div_arr = np.array(_prev_divs)
        _div_pct = float((_div_arr <= _rs_div).mean())   # CDF rank
        rs_div_bonus = round(_div_pct * 3, 1)            # 0-3 pts
    else:
        # M-2 FIX: neutral = 50th percentile = 0.5 × max (no edge claimed when no history)
        # Using 1.5 = 0.5 × 3.0 max is correct in principle; made explicit here.
        rs_div_bonus = 1.5

    # ═══════════════════════════════════════════════════════
    # F2 — RS vs SECTOR  (0-10 pts)
    # Outperforming own sector = double confirmation.
    # tanh width self-calibrates from the stock's own sector-beat distribution.
    # ═══════════════════════════════════════════════════════
    sect_ret, sect_ret_10d, sect_name = (None, None, None)
    if ticker:
        sect = get_sector(ticker)
        if sect:
            r5  = _resolved_sect_ret.get(sect)
            r10 = _resolved_sect_ret_10d.get(sect)
            if r5 is not None:
                sect_ret, sect_ret_10d, sect_name = r5, r10, sect
    # Top sectors from resolved context (not global top_sectors)
    _resolved_top_sectors = {k for k,_ in sorted(_resolved_sect_ret.items(), key=lambda x:x[1], reverse=True)[:3]} if _resolved_sect_ret else set()
    if sect_ret is not None and len(hc) >= 7:
        # Stock 5d return on hist (T-6 to T-1, no look-ahead, consistent with ltp_score=T-1).
        _hc_base = float(hc.iloc[-6])
        stock_r5 = float(hc.iloc[-1] / _hc_base - 1) if _hc_base != 0 else 0.0

        # Alpha vs sector: risk-adjusted by stock's own 5d realised vol.
        _daily_rets   = hc.pct_change().dropna()
        _stock_5d_vol = float(_daily_rets.tail(20).std() * np.sqrt(5)) if len(_daily_rets) >= 10 else 0.02
        _stock_5d_vol = max(_stock_5d_vol, 0.005)
        sect_beat = stock_r5 - sect_ret
        _sb_z     = sect_beat / _stock_5d_vol   # vol-units outperformance vs sector

        # CDF via tanh: 0=matched sector, 1=far outperformed, 0.5=neutral
        rs_sect_sc = float(0.5 * (1.0 + np.tanh(_sb_z)))

        # Sector momentum: rank this sector among all others → proportional bonus.
        _n_s = 0
        if _resolved_sect_ret:
            _all_sect_vals = sorted(_resolved_sect_ret.values())
            _n_s = len(_all_sect_vals)
            if _n_s > 1:
                _sect_rank_pct = sum(1 for v in _all_sect_vals if v <= sect_ret) / _n_s
                rs_sect_sc = min(1.0, rs_sect_sc + 0.15 * _sect_rank_pct)

        # Sector acceleration: 5d return > 10d return = sector just turned (leading signal).
        if sect_ret_10d is not None:
            _sect_accel = sect_ret - sect_ret_10d
            _sect_cross_vol = float(pd.Series(list(_resolved_sect_ret.values())).std())                               if _n_s > 2 else max(abs(sect_ret) * 0.5, 1e-4)
            _sect_cross_vol = max(_sect_cross_vol, 1e-4)
            _accel_z = _sect_accel / _sect_cross_vol
            rs_sect_sc = float(np.clip(rs_sect_sc + 0.10 * np.tanh(_accel_z), 0.0, 1.0))

        rs_sect_pts = round(rs_sect_sc * 10, 1)
    else:
        rs_sect_pts = 0.0
        sect_name   = "?"

    # ── F3: VOLUME RATIO — T-1 ONLY ──────────────────────────────────────────
    # vol_ratio = T-1 bar volume / 20d avg. Uses hv (historical slice) exclusively.
    # day_vol (signal bar vol) is EXCLUDED: high vol on the signal bar = already moved.
    # A stock quiet at T-1 (low vol_ratio) but set up technically is the target.
    vol_ratio = volume_surge(float(hv.iloc[-1]), hv.iloc[:-1])   # T-1 vol vs prior 19d avg
    vol_z     = (float(hv.iloc[-1]) - vol_mu) / (vol_sigma + 1e-9)   # for display only

    # ── FIX-05: CHURN / ABSORPTION DETECTION ────────────────────────────────
    # High volume + narrow range = institutional supply absorption.
    # Opposite of vol_quiet: vol_quiet rewards LOW vol; churn detects HIGH vol + NO price move.
    # Both signals are valid and independently predictive for different stock states.
    # churn_raw = vol_ratio / (bar_range_in_ATR_units + 0.1) — high = absorbed supply at price.
    _t1_bar_range_atr_churn = float(hh.iloc[-1] - hl.iloc[-1]) / (atr_v + 1e-9)
    _churn_raw = vol_ratio / (_t1_bar_range_atr_churn + 0.1)
    # Build 60d churn history for percentile ranking (own-history calibrated)
    _churn_hist_vol  = hv.iloc[:-1] / (hv.iloc[:-1].rolling(20).mean() + 1e-9)
    _churn_hist_rng  = (hh.iloc[:-1] - hl.iloc[:-1]) / (atr.iloc[:-1] + 1e-9)
    _churn_hist_raw  = (_churn_hist_vol / (_churn_hist_rng + 0.1)).replace(
                           [np.inf, -np.inf], np.nan).dropna().tail(60)
    _churn_pct = float((_churn_hist_raw <= _churn_raw).mean()) if len(_churn_hist_raw) >= 10 else 0.5
    # Fires only in top 40th percentile churn (clearly above-average absorption)
    _churn_bonus = round(float(np.clip((_churn_pct - 0.60) / 0.40 * 4.0, 0.0, 4.0)), 1)
    _vol_z_hist = ((hv - hv.rolling(20).mean()) / (hv.rolling(20).std() + 1e-9)).tail(60)
    _vol_tanh_w = _tanh_w(_vol_z_hist)
    # 5-day vol slope from HISTORY (pre-move accumulation, not today's spike)
    # FIX B: Normalise slope by vol std-dev (vol_sigma), NOT raw vol_mu.
    # Old: slope / vol_mu was meaningless — a 1M-share/day stock always got the same
    # denominator regardless of whether it was compressing. Std-dev normalisation makes
    # the slope a genuine z-score: how fast is volume trend changing vs its own noise.
    if len(hv) >= 8:
        _v5 = hv.tail(5).values.astype(float)
        _vol_std_norm = max(float(hv.tail(20).std()), vol_mu * 0.05, 1.0)   # vol σ, not μ
        _v5_slope = float(np.polyfit(np.arange(5, dtype=float), _v5, 1)[0]) / (_vol_std_norm + 1e-9)
        _vol_trend_pct = float((hv.rolling(5).mean().dropna() <= float(np.mean(_v5))).mean())
        _vol_signal = float(np.clip(0.5 + _v5_slope * 2.0, 0.0, 1.0)) * 0.60 + _vol_trend_pct * 0.40
    else:
        _vol_signal = 0.5
    if setup_type == "Breakout":
        _raw_vol_pts = float(np.clip(15.0 * _vol_signal, 0.0, 15.0))
        # Decay only TRUE outlier spikes (above stock's own 90th pct vol).
        # Normal accumulation vol (even above avg) is left untouched.
        _t1_vol_now = float(hv.iloc[-1])
        _vol_60d    = hv.tail(60).dropna()
        _vol_p90    = float(_vol_60d.quantile(0.90)) if len(_vol_60d) >= 10 else vol_ma20 * 2.5
        if _t1_vol_now > _vol_p90:
            _spike_excess = (_t1_vol_now - _vol_p90) / max(_vol_p90, 1.0)
            _spike_decay  = float(np.clip(_spike_excess / 2.0, 0.0, 0.7))
            _raw_vol_pts  = max(_raw_vol_pts * (1.0 - _spike_decay), 0.0)
        vol_pts = round(_raw_vol_pts, 1)
    else:
        # FIX 4: Remove floor of 5.0 on Pullback volume scoring.
        # Old: max(..., 5.0) added ~5 pts to EVERY pullback stock regardless of vol,
        # inflating scores uniformly and washing out differentiation.
        # New: score can reach 0 for high-vol pullbacks (distribution selling),
        # while quiet pullbacks (ideal) score near 15.
        vol_pts = round(float(np.clip(15.0 * (1.0 - _vol_signal), 0.0, 15.0)), 1)

    # ── INTRADAY VOLUME VELOCITY (new leading signal: 0-3 pts bonus) ──
    # Rate of change of volume within the current session.
    # A breakout with ACCELERATING intraday volume (vol arriving faster and faster)
    # is far more reliable than one with a single vol spike at open.
    # vol_velocity = scaled_vol / elapsed_frac² — if vol is arriving ahead of pace,
    # the second derivative is positive = acceleration.
    # We approximate: if day_vol_scaled >> (day_vol / elapsed_frac²), velocity is high.
    _vol_velocity_score = 0.0
    if _elapsed >= 30 and _elapsed < _SESSION_MINS:
        _velocity_ratio = day_vol / (vol_mu + 1e-9)
        _vol_velocity_score = float(np.clip(3.0 * np.tanh((_velocity_ratio - 1.0)), 0.0, 3.0))

    # ═══════════════════════════════════════════════════════
    # F4 — PRE-BREAKOUT ACCUMULATION  (0-10 pts)
    # Sigmoid centre and steepness derived from the stock's own 60d
    # inst_ratio distribution — not fixed 1.3 and k=4.
    # Centre = rolling median of inst_ratio over 60d (the stock's
    # "normal" accumulation level).  Steepness k = 1 / rolling σ
    # so a tight distribution gives a steeper curve and a noisy
    # one gives a gentler curve.
    # ═══════════════════════════════════════════════════════
    # FIX 1: Use only confirmed historical bars (T-1 and earlier) for inst_ratio.
    # day_vol is the signal bar's live volume — including it inflates scores for
    # breakout-day stocks that already moved and deflates quiet coilers.
    # The 5th value must be float(hv.iloc[-1]) (T-1 confirmed bar), NOT day_vol.
    _hist5 = list(hv.tail(5).values)
    inst_ratio = float(np.mean(_hist5)) / (vol_ma20 + 1e-9)
    # Build rolling inst_ratio history for self-calibration
    _inst_hist = (hv.rolling(5).mean() / (hv.rolling(20).mean() + 1e-9)).dropna()
    if len(_inst_hist) >= 20:
        _inst_centre = float(_inst_hist.tail(60).median())   # robust centre
        _inst_sigma  = float(_inst_hist.tail(60).std())
        # M-4 FIX: floor is the 5th percentile of observed inst_sigma values, not 0.05
        _reg_inst = _reg.get("inst_sigma", [])
        _reg_inst.append(_inst_sigma)
        _reg["inst_sigma"] = _reg_inst[-200:]
        _inst_sigma_floor = float(np.percentile(_reg_inst, 5)) if len(_reg_inst) >= 10 else 0.05
        _inst_k      = 1.0 / max(_inst_sigma, _inst_sigma_floor)   # steepness = inverse σ, floored at 5th pct
    else:
        _inst_centre = 1.2   # fallback: slightly above average = mild accumulation
        _inst_k      = 3.0
    inst_pts = round(10.0 / (1.0 + np.exp(-_inst_k * (inst_ratio - _inst_centre))), 1)
    inst_pts = max(0.0, min(10.0, inst_pts))


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

    # FIX-04 Part B: MTF bonus — 3-level tiering (daily + weekly + monthly).
    # Only fires when daily compression is confirmed first.
    # Three TFs compressed = full bonus (5 pts).
    # Daily + weekly (or daily + monthly) = moderate bonus (3 pts).
    # Daily only = no MTF bonus (vc_pts already captures this).
    _daily_compressed = _vc_pct <= 0.35
    if _daily_compressed:
        _n_tf_compressed = 1 + int(_wk_compressed) + int(_mo_compressed)
        if _n_tf_compressed == 3:
            # All three TFs compressed — rare, very high signal quality
            _mtf_strength = (1.0 - _vc_pct) * (1.0 - _wk_vc_pct) * (1.0 - _mo_vc_pct)
            _mtf_bonus = round(float(np.clip(_mtf_strength * 15.0, 0.0, 5.0)), 1)
        elif _n_tf_compressed == 2:
            # Two TFs — use the tighter of weekly vs monthly
            _best_higher = min(_wk_vc_pct if _wk_compressed else 1.0,
                               _mo_vc_pct if _mo_compressed else 1.0)
            _mtf_strength = (1.0 - _vc_pct) * (1.0 - _best_higher)
            _mtf_bonus = round(float(np.clip(_mtf_strength * 10.0, 0.0, 3.0)), 1)
        else:
            _mtf_bonus = 0.0
    else:
        _mtf_bonus = 0.0

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

    # ── VCVE: Volume-Compression Interaction (bonus, 0-3 pts) ──
    # Detects hidden accumulation: rising vol + falling volatility.
    # Saturation point = 75th percentile of vcve over own 60d history,
    # not a fixed 0.4.  This adapts to the stock's typical activity level.
    vcve = inst_ratio * (1.0 - min(vc_ratio, 1.0))
    # Build vcve history using tail alignment (both series share the hist index)
    # _inst_hist and _vc_series are both on the hist slice — align by taking
    # the last N values from each so they correspond to the same bars.
    if len(_inst_hist) >= 10 and len(_vc_series) >= 10:
        _n_common   = min(60, len(_inst_hist), len(_vc_series))
        _inst_tail  = _inst_hist.iloc[-_n_common:].values
        _vc_tail    = _vc_series.iloc[-_n_common:].values
        _vcve_arr   = _inst_tail * np.clip(1.0 - _vc_tail, 0.0, 1.0)
        _vcve_arr   = _vcve_arr[~np.isnan(_vcve_arr)]
        _vcve_hist  = pd.Series(_vcve_arr, dtype=float)
    else:
        _vcve_hist  = pd.Series(dtype=float)
    _vcve_sat = float(_vcve_hist.quantile(0.75)) if len(_vcve_hist) >= 10 else 0.4
    _vcve_sat = max(_vcve_sat, 0.1)
    vcve_bonus = round(float(np.clip(3.0 * np.tanh(vcve / _vcve_sat), 0.0, 3.0)), 1)

    # ═══════════════════════════════════════════════════════════════
    # UNIVERSAL LEADING SIGNALS — work on ALL stocks from OHLCV only
    # These detect pre-move conditions BEFORE price confirms direction.
    # ═══════════════════════════════════════════════════════════════

    # ── L1: UPSIDE VOLUME SKEW (0-3 pts) ──
    # When volume on up-close days is consistently higher than on down-close
    # days, buyers are absorbing supply quietly — even if price is flat.
    # This is demand without fanfare: the classic accumulation signature.
    # Measured over 20 sessions to capture sustained institutional activity.
    # Score = percentile rank of (up_vol / down_vol) vs own 60d history.
    _uv_bonus = 0.0
    if len(hc) >= 20:
        _up_mask   = hc.diff() > 0          # up-close days (historical, no look-ahead)
        _dn_mask   = hc.diff() < 0
        _uv_window = min(20, len(hc))
        _up_vol    = float(hv[_up_mask].tail(_uv_window).sum())
        _dn_vol    = float(hv[_dn_mask].tail(_uv_window).sum())
        _uv_ratio  = _up_vol / (_dn_vol + 1e-9)
        # Build rolling up/down vol ratio history for percentile ranking.
        # Extended to 120 bars (from 60) for a stable CDF rank — 40 observations
        # is insufficient to anchor the 90th percentile reliably.
        _uv_hist = pd.Series([
            hv[_up_mask].iloc[max(0, i-20):i].sum() /
            (hv[_dn_mask].iloc[max(0, i-20):i].sum() + 1e-9)
            for i in range(20, min(120, len(hc)))
        ], dtype=float)
        if len(_uv_hist) >= 5:
            _uv_pct = float((_uv_hist <= _uv_ratio).mean())   # CDF rank
            _uv_bonus = round(_uv_pct * 3.0, 1)               # 0-3 pts
        else:
            _uv_bonus = 1.5 if _uv_ratio > 1.0 else 0.5      # fallback

    # ── L2: CLOSE POSITION RANK — CPR (0-3 pts) ──
    # Consecutive closes in the upper portion of the day's range signal
    # that intraday sellers are being absorbed.  Price coiling in the upper
    # half while holding gains = demand > supply.
    # Formula: mean of (close - low)/(high - low) over last 10 bars,
    # percentile-ranked over own 60d history.
    _cpr_bonus = 0.0
    if len(hc) >= 10:
        _hl_range  = (hh - hl).replace(0, np.nan)
        _cpr_raw   = ((hc - hl) / _hl_range).dropna()
        _cpr_10    = float(_cpr_raw.tail(10).mean())
        _cpr_hist  = _cpr_raw.rolling(10).mean().dropna()
        if len(_cpr_hist) >= 10:
            _cpr_pct   = float((_cpr_hist <= _cpr_10).mean())
            _cpr_bonus = round(_cpr_pct * 3.0, 1)
        else:
            _cpr_bonus = round(_cpr_10 * 3.0, 1)   # fallback: direct CPR score

    # ── L3: SPREAD COMPRESSION + RISING CLOSE (0-3 pts) ──
    # Range shrinking over 5 bars while close drifts upward = quiet accumulation.
    # This is the hallmark of institutional buying: they push slowly to avoid
    # moving the price, so range compresses while close steadily rises.
    # Score = product of (1 - range_compression_pct) × close_drift_pct,
    # both percentile-ranked so the signal is relative to this stock's own history.
    _sc_bonus = 0.0
    if len(hc) >= 15:
        _range_5d  = (hh.tail(5).max() - hl.tail(5).min())
        _range_10d = (hh.tail(10).max() - hl.tail(10).min())
        _compression = 1.0 - (_range_5d / (_range_10d + 1e-9))
        try:
            _close_slope = float(np.polyfit(range(5), hc.tail(5).values, 1)[0]) / (atr_v + 1e-9)
        except (np.linalg.LinAlgError, ValueError):
            _close_slope = 0.0
        _quiet_accum = max(0.0, _compression) * max(0.0, _close_slope)

        # Vectorised rolling slope via least-squares formula for window=5:
        # slope = (5*Σxy - Σx*Σy) / (5*Σx² - (Σx)²) where x=[0,1,2,3,4]
        # Pre-compute x-weights: for window 5, denominator = 5*30 - 100 = 50
        _w = len(hc)
        if _w >= 20:
            _x   = np.arange(5, dtype=float)
            _sx  = _x.sum()                    # 10
            _sx2 = (_x**2).sum()               # 30
            _n   = 5
            _denom = _n * _sx2 - _sx**2        # 50

            # Rolling sum of closes over window 5 and rolling weighted sum
            _c_vals = hc.values.astype(float)
            _sy_roll  = np.convolve(_c_vals, np.ones(5), 'valid')         # Σy per window
            _sxy_roll = np.convolve(_c_vals, _x[::-1], 'valid')           # Σxy per window (x reversed for convolution)

            # Correct Σxy: conv gives Σ c[i+j]*x[4-j], we need Σ c[i+j]*x[j]
            # So use _x directly: sxy = sum(c[i:i+5] * [0,1,2,3,4])
            # Build via stride trick
            _strides = np.lib.stride_tricks.sliding_window_view(_c_vals, 5)  # (W-4, 5)
            _sxy_arr = (_strides * _x).sum(axis=1)
            _sy_arr  = _strides.sum(axis=1)
            _slopes  = (_n * _sxy_arr - _sx * _sy_arr) / (_denom + 1e-9)

            # ATR values aligned to same windows (use atr series on hist slice)
            _atr_vals = atr.iloc[:-1].values.astype(float)   # same length as hc
            # We need atr at position i+4 (last bar of each window)
            _atr_win  = _atr_vals[4:]   # aligned to window endings; length = W-4
            _n_win    = min(len(_slopes), len(_atr_win))
            _slopes   = _slopes[:_n_win]
            _atr_win  = _atr_win[:_n_win]

            _range_5_arr  = np.array([hh.values[i:i+5].max() - hl.values[i:i+5].min()
                                       for i in range(_n_win)])
            _range_10_arr = np.array([hh.values[max(0,i-4):i+6].max() - hl.values[max(0,i-4):i+6].min() + 1e-9
                                       for i in range(_n_win)])
            _comp_arr = np.clip(1.0 - _range_5_arr / _range_10_arr, 0.0, 1.0)
            _slope_norm = np.clip(_slopes / (_atr_win + 1e-9), 0.0, None)
            _sc_hist_arr = _comp_arr * _slope_norm
            _sc_hist = pd.Series(_sc_hist_arr[-60:], dtype=float).dropna()

            if len(_sc_hist) >= 5:
                _sc_pct   = float((_sc_hist <= _quiet_accum).mean())
                _sc_bonus = round(_sc_pct * 3.0, 1)
            else:
                _sc_bonus = round(float(np.clip(_quiet_accum * 3.0, 0.0, 3.0)), 1)
        else:
            _sc_bonus = round(float(np.clip(_quiet_accum * 3.0, 0.0, 3.0)), 1)

    # ── L4: ATR EXPANSION ONSET (0-3 pts) ──
    # The FIRST bar where short-term ATR begins expanding after a contraction
    # phase is the earliest measurable signal that a coil is releasing.
    # Not the full breakout (too late) — the derivative: d(ATR5)/dt turning positive
    # after being in contraction.
    # Score inversely proportional to how many bars since the onset — fresher = higher.
    _atr_exp_bonus = 0.0
    if len(_tr_series) >= 25:
        _atr5_series  = _tr_series.rolling(5).mean().dropna()
        _atr20_series = _tr_series.rolling(20).mean().dropna()
        # Align the two series
        _common_idx   = _atr5_series.index.intersection(_atr20_series.index)
        _vc_ratio_ser = (_atr5_series.loc[_common_idx] /
                         (_atr20_series.loc[_common_idx] + 1e-9))
        # Find where ATR5/ATR20 bottomed and started rising (onset of expansion)
        # H-7 FIX: "compressed" threshold derived from stock's own 30th percentile
        # instead of hardcoded 0.85. Bottom 30% of own vc_ratio history = genuinely coiled.
        _vc_p30 = float(np.percentile(_vc_ratio_ser.values[~np.isnan(_vc_ratio_ser.values)], 30)) \
                  if len(_vc_ratio_ser.dropna()) >= 10 else 0.85
        if len(_vc_ratio_ser) >= 10:
            _vc_arr   = _vc_ratio_ser.values
            _vc_diff  = np.diff(_vc_arr)
            _was_compressed = False
            _bars_since_onset = None
            for j in range(len(_vc_diff) - 1, -1, -1):
                if _vc_arr[j] < _vc_p30:          # compressed = below own 30th percentile
                    _was_compressed = True
                if _was_compressed and _vc_diff[j] > 0:   # now expanding
                    _bars_since_onset = len(_vc_diff) - j
                    break
            if _bars_since_onset is not None and _bars_since_onset <= 5:
                # Score decays with bars since onset: 1 bar = 3 pts, 5 bars = 0.6 pts
                _atr_exp_bonus = round(float(np.clip(3.0 / _bars_since_onset, 0.0, 3.0)), 1)

    # ── L4b: HIGHER HIGHS + HIGHER LOWS STRUCTURE (0-3 pts) ────────────────
    # The most direct structural leading signal: detecting that the stock is
    # already making HH+HL BEFORE EMA alignment confirms the trend.
    # EMA convergence (F7) fires AFTER several HH+HL have happened.
    # This signal fires ON the 3rd HH+HL — one full trend cycle earlier.
    #
    # Method: detect the last 3 swing highs and 3 swing lows using a
    # dynamic window derived from the stock's own ATR cycle (same as VCP).
    # Score = how many of the last 3 swings conform to HH+HL structure.
    #   3/3 = perfect structure → 3 pts
    #   2/3 = partial structure → 1.5 pts
    #   1/3 or 0/3 = no structure → 0 pts
    #
    # Only scores for Breakout and Pullback setups — Reversal has inverted logic (see below).
    _hhhl_bonus = 0.0
    _llhl_bonus = 0.0   # Reversal counterpart: Lower-High streak breaking = Higher-Low forming
    if len(hc) >= 40:
        try:
            _dr_series  = (hh - hl).replace(0, np.nan).dropna()
            _med_rng    = float(_dr_series.tail(60).median()) if len(_dr_series) >= 10 else atr_v
            _hhhl_win   = int(np.clip((atr_v / (_med_rng + 1e-9)) * 5.0, 3, 20))
            _sh_series  = hh.iloc[:-1].rolling(2 * _hhhl_win + 1, center=True).max()
            _sl_series  = hl.iloc[:-1].rolling(2 * _hhhl_win + 1, center=True).min()
            _swing_hi   = hh.iloc[:-1][hh.iloc[:-1] == _sh_series].dropna()
            _swing_lo   = hl.iloc[:-1][hl.iloc[:-1] == _sl_series].dropna()
            _sh_vals    = _swing_hi.values[-4:]
            _sl_vals    = _swing_lo.values[-4:]

            if setup_type != "Reversal":
                # Bullish: count Higher Highs and Higher Lows
                _hh_count = 0; _hl_count = 0
                for _k in range(1, min(len(_sh_vals), 3)):
                    if _sh_vals[-_k] > _sh_vals[-_k - 1]: _hh_count += 1
                for _k in range(1, min(len(_sl_vals), 3)):
                    if _sl_vals[-_k] > _sl_vals[-_k - 1]: _hl_count += 1
                _struct_score = (_hh_count + _hl_count) / 6.0
                _hhhl_bonus   = round(_struct_score * 3.0, 1)
            else:
                # Reversal: detect the structural turn — the last swing low must be
                # HIGHER than the prior swing low (HL forming = buyers stepping in).
                # Also check that swing highs stopped making new lows (LH streak breaking).
                # Score = fraction of last 3 swing pairs that confirm the structural turn.
                _hl_forming = 0; _lh_breaking = 0
                for _k in range(1, min(len(_sl_vals), 3)):
                    # HL: most recent swing low > prior → buyers defending higher lows
                    if _sl_vals[-_k] > _sl_vals[-_k - 1]: _hl_forming += 1
                for _k in range(1, min(len(_sh_vals), 3)):
                    # LH breaking: most recent swing high >= prior (no longer making lower highs)
                    if _sh_vals[-_k] >= _sh_vals[-_k - 1]: _lh_breaking += 1
                _turn_score  = (_hl_forming + _lh_breaking) / 6.0
                _llhl_bonus  = round(_turn_score * 3.0, 1)
        except Exception:
            _hhhl_bonus = 0.0
            _llhl_bonus = 0.0

    # ── L5: OI BUILDUP (F&O stocks only, 0-3 pts) ──
    # Open interest rising while price coils = institutional positioning BEFORE move.
    # For non-F&O stocks OI column is 0 → oi_bonus stays 0.0 automatically.
    # PARTICIPANT OI ENHANCEMENT: if NSE participant data is available and FII is
    # net long in this stock's futures, the oi_bonus is scaled up by 1.5×.
    # This distinguishes informed (FII long-buildup) from uninformed (retail/prop)
    # OI accumulation. Non-F&O stocks: oi_bonus always 0 — no change.
    oi_bonus = 0.0
    _fii_net_long = None   # None = no data / non-F&O
    if "oi" in df.columns and len(df) >= 10:
        _oi = df["oi"].dropna()
        # Non-F&O stocks: oi column exists but all values are 0 — skip
        _oi_nonzero = _oi[_oi > 0]
        if len(_oi_nonzero) >= 5:
            _oi_now   = float(_oi.iloc[-1])
            _oi_5d    = float(_oi.iloc[-6]) if len(_oi) >= 6 else float(_oi.iloc[0])
            _oi_avg   = float(_oi_nonzero.tail(20).mean())
            _oi_std   = float(_oi_nonzero.tail(20).std())
            _oi_std   = max(_oi_std, _oi_avg * 0.05)
            _oi_rising = _oi_now > _oi_5d
            _oi_z      = (_oi_now - _oi_avg) / (_oi_std + 1e-9)
            # H-7: "price coiling" threshold = bottom 30th pct of this stock's own vc history
            _vc_p30_oi      = float(_vc_series.dropna().quantile(0.30)) \
                              if len(_vc_series.dropna()) >= 10 else 0.85
            _price_coiling  = vc_ratio < _vc_p30_oi
            if _oi_rising and _price_coiling and _oi_z > 0.5:
                _compression_strength = 1.0 - min(vc_ratio, 1.0)
                _raw_oi_bonus = float(np.clip(
                    3.0 * np.tanh(_oi_z * _compression_strength), 0.0, 3.0))
                # ── Participant OI direction multiplier ──
                # Only apply if we have actual FII data for this stock.
                # FII net long = informed accumulation → full 1.5× amplification.
                # FII net short = hedging/distribution → cap at 0.5× (OI rising but wrong-way flow).
                # No data (non-F&O, weekend, fetch failed) → neutral ×1.0.
                if not bt_mode and ticker:
                    _part_oi_map = st.session_state.get("participant_oi", {})
                    _fii_net_long = _part_oi_map.get(ticker.upper())
                    if _fii_net_long is True:
                        _raw_oi_bonus = min(_raw_oi_bonus * 1.5, 3.0)
                    elif _fii_net_long is False:
                        _raw_oi_bonus = min(_raw_oi_bonus * 0.5, 3.0)
                oi_bonus = round(_raw_oi_bonus, 1)

    # ── L5b: DELIVERY % BONUS (NSE Bhav Copy, 0-4 pts) ──────────────────────
    # FIX 15: Apply delivery bonus to ALL setup types including Reversal.
    # A Reversal with 60%+ delivery = real panic selling with informed money holding →
    # strong bounce candidate. A Reversal with <20% delivery = intraday day-traders
    # hammering the stock = bounce is less reliable. Excluding Reversal from this signal
    # threw away one of the most useful discriminators for bounce quality.
    # Delivery % = delivery_qty / total_traded_qty from NSE's end-of-day Bhav Copy.
    # What it means:
    #   >60% delivery → informed money is HOLDING overnight. High conviction buying.
    #   20-60%        → mixed, normal activity.
    #   <20%          → pure intraday speculation. No one wanted to hold it.
    #
    # Why this matters: a stock breaking out on 70% delivery is categorically
    # different from one breaking out on 15% delivery. The former has real demand
    # behind it; the latter is likely to reverse the next session.
    #
    # Scoring:
    #   del_pct ≥ 60% → 4 pts (strong conviction holding)
    #   del_pct ≥ 45% → 2.5 pts (above average holding)
    #   del_pct ≥ 30% → 1 pt (neutral)
    #   del_pct < 20% → −2 pts soft penalty (intraday noise, subtract from bonus)
    #   No data        → 0 pts (neutral, don't penalise missing data)
    _delivery_bonus = 0.0
    _delivery_pct_val = None
    if not bt_mode and ticker:
        _bhav_data = st.session_state.get("delivery_pct", {})
        _raw_del = _bhav_data.get(ticker.upper())
        if _raw_del is not None:
            _delivery_pct_val = float(_raw_del)
            if _delivery_pct_val >= 60:
                _delivery_bonus = 4.0
            elif _delivery_pct_val >= 45:
                _delivery_bonus = 2.5
            elif _delivery_pct_val >= 30:
                _delivery_bonus = 1.0
            elif _delivery_pct_val < 20:
                _delivery_bonus = -2.0   # intraday speculation — soft penalty


    # ═══════════════════════════════════════════════════════
    # F6 — BASE / COIL QUALITY + BASE POSITION  (0-10 pts)
    # Includes flat-resistance + position score (>0.75 = at resistance)
    # ═══════════════════════════════════════════════════════
    # _swing_window derived once here — used in BOTH Breakout and Pullback paths below.
    # Bug fix: was computed inside the else (Pullback) block but referenced in the if
    # (Breakout) block via a broken dir() guard. Now computed unconditionally.
    _swing_window = 20  # default
    if len(hh) >= 40:
        _peaks    = (hh.rolling(3, center=True).max() == hh).astype(int)
        _peak_idx = hh.index[_peaks == 1].tolist()
        if len(_peak_idx) >= 3:
            _peak_gaps = [_peak_idx[i+1] - _peak_idx[i] for i in range(len(_peak_idx)-1)
                          if isinstance(_peak_idx[i+1] - _peak_idx[i], (int, float))]
            if _peak_gaps:
                _swing_window = int(np.clip(np.median(_peak_gaps), 10, 40))

    if setup_type == "Breakout":
        rng5  = float(hh.tail(5).max()) - float(hl.tail(5).min())
        tight = 1.0 - min(1.0, rng5 / (base_rng + 1e-9))
        rec_hi    = hh.tail(8)
        # AUDIT FIX: normalise high-spread by ATR instead of magic *40 multiplier
        # hi_spread as fraction of ATR = how many ATR-widths does resistance vary?
        # 0 = perfectly flat (all highs identical), 1 = spread equals one full ATR
        hi_spread = (rec_hi.max() - rec_hi.min()) / (atr_v + 1e-9)
        flatness  = max(0.0, 1.0 - min(hi_spread / 1.0, 1.0))   # 0 ATR-spread = full flat
        # H-3 FIX: weights derived from relative historical variance of each sub-component
        _tight_hist   = (1.0 - (hh.rolling(5).max() - hl.rolling(5).min())
                         .tail(60) / (base_rng + 1e-9)).dropna()
        _flat_hist    = pd.Series([
            max(0.0, 1.0 - (hh.iloc[max(0,i-7):i].max() - hh.iloc[max(0,i-7):i].min()) / (atr_v + 1e-9))
            for i in range(max(8, len(hh)-60), len(hh))
        ], dtype=float).dropna()
        _tight_var = float(_tight_hist.var()) if len(_tight_hist) >= 5 else 0.5
        _flat_var  = float(_flat_hist.var())  if len(_flat_hist)  >= 5 else 0.5
        _total_var = _tight_var + _flat_var + 1e-9
        _w_tight   = _tight_var / _total_var
        _w_flat    = _flat_var  / _total_var
        coil_sc   = tight * _w_tight + flatness * _w_flat
        base_pos  = (ltp_score - base_lo) / (base_rng + 1e-9)
        # Base position bonus: sigmoid centred at stock's own 75th percentile base position.
        # _swing_window is defined before this if/else block so it's available here.
        _bpos_hist = ((hc - hl.rolling(_swing_window).min()) /
                      (hh.rolling(_swing_window).max() - hl.rolling(_swing_window).min() + 1e-9)).dropna().tail(60)
        _bpos_centre = float(_bpos_hist.quantile(0.75)) if len(_bpos_hist) >= 10 else 0.80
        _bpos_sigma  = float(max((_bpos_hist.quantile(0.90) - _bpos_hist.quantile(0.60))
                                  / 1.35, 0.05)) if len(_bpos_hist) >= 10 else 0.10
        _bp_sigmoid_k = 1.0 / _bpos_sigma
        _bp_bonus = 0.2 / (1.0 + np.exp(-_bp_sigmoid_k * (base_pos - _bpos_centre)))
        coil_sc   = min(1.0, coil_sc + float(_bp_bonus))
    else:
        # H-2 FIX: use the SAME window for both high and low anchor.
        # Old code used tail(20) for high and tail(40) for low — asymmetric time horizons
        # produced systematically biased pullback depth estimates.
        # _swing_window is computed once before this if/else block (no longer duplicated here)

        psw_hi  = float(hh.tail(_swing_window).max())
        psw_lo  = float(hl.tail(_swing_window).min())   # H-2: same window as high
        pm      = psw_hi - psw_lo + 1e-9
        pb_dep  = (psw_hi - float(hc.iloc[-1])) / pm

        # H-8 FIX: Derive Gaussian parameters from this stock's own pullback distribution.
        # Old: _pb_sigma = (0.65 - 0.15) / 4 — global assumption for all NSE stocks.
        # New: compute rolling pullback depths from the stock's own price history.
        #      Centre = median of own pullback depths (not always 0.382).
        #      Sigma  = IQR/1.35 of own pullback depths (robust estimate).
        # Fibonacci 0.382 is the theoretically optimal entry but real stocks vary — some
        # mean-revert at 0.25 (strong trend), others at 0.50 (weaker trend).
        if len(hh) >= 40:
            # Build rolling pullback depth history
            _sw = _swing_window
            _pb_depths = []
            for _i in range(_sw, min(len(hh), _sw + 120)):
                _ph = float(hh.iloc[_i - _sw : _i].max())
                _pl = float(hl.iloc[_i - _sw : _i].min())
                _pm = _ph - _pl + 1e-9
                _pd = (_ph - float(hc.iloc[_i])) / _pm
                _pb_depths.append(_pd)
            _pb_arr = np.array([x for x in _pb_depths if 0.0 <= x <= 1.0])
            if len(_pb_arr) >= 10:
                _pb_centre = float(np.median(_pb_arr))
                _pb_q75, _pb_q25 = np.percentile(_pb_arr, 75), np.percentile(_pb_arr, 25)
                _pb_sigma  = float(max((_pb_q75 - _pb_q25) / 1.35, 0.05))
            else:
                _pb_centre = 0.382
                _pb_sigma  = 0.125
        else:
            _pb_centre = 0.382
            _pb_sigma  = 0.125

        coil_sc    = float(np.exp(-0.5 * ((pb_dep - _pb_centre) / _pb_sigma) ** 2))
        coil_sc    = float(np.clip(coil_sc, 0.0, 1.0))
        base_pos   = (ltp_score - base_lo) / (base_rng + 1e-9)
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
        _pos_now   = (ltp_score - _lo250) / (_hi250 - _lo250 + 1e-9)
        # Build rolling position series over same window
        _pos_series = (hc - hc.rolling(_n250).min()) / \
                      (hc.rolling(_n250).max() - hc.rolling(_n250).min() + 1e-9)
        _pos_pct    = percentile_last(_pos_series, min(250, len(_pos_series)))
        if pd.notna(_pos_pct):
            # Fix 7: max pts derived from registry of observed pos52w contributions
            _pos52w_reg = _reg.get("pos52w_max", [])
            _pos52w_max = float(np.median(_pos52w_reg)) if len(_pos52w_reg) >= 10 else 3.0
            _pos52w_max = float(np.clip(_pos52w_max, 1.0, 5.0))   # natural bounds
            pos52w_bonus = round(_pos_pct * _pos52w_max, 1)
            # Record for future calibration
            _pos52w_reg.append(_pos52w_max)
            _reg["pos52w_max"] = _pos52w_reg[-200:]
            pos52w       = round(_pos_now, 3)
        else:
            pos52w_bonus = 0.0
            pos52w       = round(base_pos, 3)
    else:
        pos52w_bonus = 0.0
        pos52w       = round(base_pos, 3)

    # ═══════════════════════════════════════════════════════
    # LIQUIDITY SWEEP DETECTION (0-4 pts)
    # Fix 9+10: sweep_bonus now proportional to wick strength (not binary 0 or 4).
    # vol_z threshold derived from stock's own 60th pct (not hardcoded 1.0).
    sweep_bonus = 0.0
    if len(hc) >= 5:
        prior_support = float(hl.tail(5).min())
        prior_close   = float(hc.iloc[-1])
        lower_wick    = min(day_o, ltp_score) - day_lo     # body bottom - day low
        # Fix 10: vol_z threshold = 60th pct of own 60d vol z-score distribution
        _vol_z_hist_sw = ((hv - hv.rolling(20).mean()) / (hv.rolling(20).std() + 1e-9)).tail(60)
        _vol_z_p60     = float(_vol_z_hist_sw.quantile(0.60)) if len(_vol_z_hist_sw) >= 20 else 1.0
        if (day_lo < prior_support and ltp_score > prior_close and
                lower_wick >= 0.5 * atr_v and vol_z >= _vol_z_p60):
            # Fix 9: proportional to wick/ATR ratio — deeper wick = stronger rejection
            _wick_atr_ratio = lower_wick / (atr_v + 1e-9)
            sweep_bonus = round(float(np.clip(4.0 * np.tanh(_wick_atr_ratio), 0.0, 4.0)), 1)

    # ═══════════════════════════════════════════════════════
    # VWMA-20 POSITION (bonus pts)
    # FIX B-04: This is a 20-day Volume-Weighted Moving Average on daily bars,
    # NOT intraday VWAP. Renamed to VWMA20 to avoid confusion.
    # Price > VWMA20 = buyers controlling trend; adds 2 pts.
    # FIX C: Use hist-only series (hh, hl, hc, hv) to avoid look-ahead.
    # Old code used h/l/c/v (full df incl. signal bar) so VWMA20 incorporated
    # today's price — half-bar look-ahead when compared vs ltp_score (T-1 close).
    # ═══════════════════════════════════════════════════════
    vwap_bonus = 0
    if "volume" in df.columns and len(hc) >= 20:
        _typical_h = (hh + hl + hc) / 3                        # hist-only typical price
        _cum_tv_h  = (_typical_h * hv).rolling(20).sum()
        _cum_v_h   = hv.rolling(20).sum()
        cum_tv = _cum_tv_h   # alias used by vwma20_prev below
        cum_v  = _cum_v_h
        vwma20_val = float((_cum_tv_h / _cum_v_h.replace(0, np.nan)).iloc[-1])
        if not np.isnan(vwma20_val):
            # FIX 9: Replace binary +2/0 with continuous ATR-normalised distance.
            # How far above/below VWMA20 is the stock, in ATR units?
            # Percentile-ranked over own 60d VWMA20-distance history for self-calibration.
            _vwma20_dist_atr = (ltp_score - vwma20_val) / (atr_v + 1e-9)
            _vwma20_hist = (_cum_tv_h / _cum_v_h.replace(0, np.nan)).ffill()
            _dist_hist = ((hc - _vwma20_hist) / (atr.iloc[:-1] + 1e-9)).dropna().tail(60)
            if len(_dist_hist) >= 10:
                _vwma_pct = float((_dist_hist <= _vwma20_dist_atr).mean())
                vwap_bonus = round(_vwma_pct * 3.0, 1)   # 0-3 pts, continuous
            else:
                # Fallback: simple sign-based
                vwap_bonus = 2 if ltp_score > vwma20_val else 0
            # VWMA20 trending upward slope adds 1 pt
            vwma20_prev = float((cum_tv / cum_v.replace(0, np.nan)).iloc[-2]) if len(hc) >= 21 else vwma20_val
            if not np.isnan(vwma20_prev) and vwma20_val > vwma20_prev:
                vwap_bonus = min(vwap_bonus + 1, 3)

    stab_bonus = 0.0   # C-4 FIX: always initialised before the conditional blocks below

    # ═══════════════════════════════════════════════════════
    # MOMENTUM STABILITY — soft penalty, not a kill-switch
    # FIX I: The original hard "return None if stability < 0.35" discard valid
    # V-shaped snap-back reversals — exactly the highest-velocity entries.
    # New approach: percentile-rank stability over a 60d rolling window of
    # the stock's own stability history, then map to a ±penalty band.
    # Below-median stability → penalty up to −8 pts (proportional, not binary).
    # Only truly chaotic stocks (stability < 20th percentile of own history)
    # are hard-rejected — and only when RSI is also not recovering.
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

    # Build rolling stability history to percentile-rank today's reading
    if len(hc) >= 40:
        _stab_hist = hc.pct_change().rolling(20).apply(
            lambda x: (x > 0).sum() / max(len(x.dropna()), 1), raw=False
        ).dropna()
        _stab_pct = percentile_last(_stab_hist, min(60, len(_stab_hist)))
    else:
        _stab_pct = None

    # Hard kill only for truly uncomputable state (handled above).
    # Extreme instability is now a score penalty, not a kill.
    # Stocks with stability < 0.20 AND falling RSI get a heavy penalty
    # but still appear in the screener so the user can see they are deteriorating.
    if stability < 0.20 and (rsi_v < 40 or rsi_v <= rsi_p):
        _stab_kill_z   = (0.20 - stability) / 0.20
        _soft_penalty += float(np.clip(15.0 * _stab_kill_z, 0.0, 20.0))

    # Soft penalty: proportional to how far below median the stock is.
    # Fix 3+4: stab_adj scale and clip derived from stock's own distribution.
    if _stab_pct is not None and pd.notna(_stab_pct):
        _stab_deviation = _stab_pct - 0.50   # −0.5 to +0.5
        _stab_w = float(max(np.std(list(_stab_hist)) if len(_stab_hist) >= 10 else 0.15, 0.05))
        _stab_z = _stab_deviation / _stab_w
        # Fix 3: scale from registry — how large stab adjustments are for this universe
        _reg_stab = _reg.get("stab_adj_scale", [])
        _stab_scale = float(np.median(_reg_stab)) if len(_reg_stab) >= 10 else 5.0
        # Fix 4: clip bounds from registry — natural range of observed stab_adj values
        _reg_stab_adj = _reg.get("stab_adj_obs", [])
        if len(_reg_stab_adj) >= 10:
            _stab_clip_lo = float(np.percentile(_reg_stab_adj, 5))
            _stab_clip_hi = float(np.percentile(_reg_stab_adj, 95))
        else:
            _stab_clip_lo, _stab_clip_hi = -8.0, 2.0  # bootstrap
        stab_adj = float(np.clip(np.tanh(_stab_z) * _stab_scale, _stab_clip_lo, _stab_clip_hi))
        # Record for future calibration
        _reg_stab_adj.append(stab_adj)
        _reg["stab_adj_obs"] = _reg_stab_adj[-200:]
        # Record the scale for future calibration (scale = std of stability × ATR%/100)
        _this_scale = abs(_stab_z) * atr_pct / 100.0 if _stab_z != 0 else 5.0
        _reg_stab.append(float(np.clip(_this_scale, 1.0, 15.0)))
        _reg["stab_adj_scale"] = _reg_stab[-200:]
        stab_bonus = stab_adj
    else:
        stab_bonus = 1.0 if stability >= 0.60 else 0.0

    # ═══════════════════════════════════════════════════════
    # F7 — TREND STRUCTURE / MA  (0-10 pts)
    # FIX I-04: Replace hard step scoring with EMA9/EMA50 ratio
    # percentile-ranked over 250d history — fully continuous, adaptive.
    # trend_ratio = EMA9 / EMA50 captures both direction AND strength.
    # percentile_last(trend_ratio, 250) → 0 at 250d lows, 1 at 250d highs.
    # Bonus: classic EMA alignment check still used as a quality gate.
    # ═══════════════════════════════════════════════════════
    # F7 MA CONVERGENCE — fires BEFORE the cross, not after
    if len(e9) >= 4:
        _e9_slope = (float(e9.iloc[-1]) - float(e9.iloc[-4])) / (atr_v * 3.0 + 1e-9)
        _e9_slope_score = float(np.clip(0.5 + _e9_slope * 2.0, 0.0, 1.0))
    else:
        _e9_slope_score = 0.5
    _gap_now  = (e9_v - e20_v) / (atr_v + 1e-9)
    _gap_prev = (e9_y - e20_y) / (atr_v + 1e-9)
    _cross_prox = float(np.exp(-abs(_gap_now) * 2.0))
    _converge_score = float(np.clip(_cross_prox * (1.2 if _gap_now > _gap_prev else 0.8), 0.0, 1.0))
    _above_e50_score = float(np.clip((ltp_score - e50_v) / (atr_v + 1e-9) + 0.5, 0.0, 1.0))
    ma_pts = round((_e9_slope_score * 0.35 + _converge_score * 0.45 + _above_e50_score * 0.20) * 10.0, 1)

    # ═══════════════════════════════════════════════════════
    # F8 — BREAKOUT PROXIMITY  (0-10 pts)
    # Exponential decay: score = 10 × exp(−λ × d_trig_atr)
    # λ (decay rate) is self-calibrated from the stock's own historical
    # distance-to-resistance distribution.  The median distance at which
    # breakouts occurred historically defines the natural half-life point.
    # λ = ln(2) / median_distance → score halves at the stock's own median BO distance.
    # Fallback λ = 1.0 (gentler than the old 1.5) when history is thin.
    # ═══════════════════════════════════════════════════════
    if len(hh) >= 30:
        _dist_hist_bo = ((hh.rolling(20).max().shift(1) - hc) / (atr.iloc[:-1] + 1e-9)).dropna()
        _dist_hist_bo = _dist_hist_bo[_dist_hist_bo > 0].tail(60)
        _median_dist  = float(_dist_hist_bo.median()) if len(_dist_hist_bo) >= 10 else 0.7
        _median_dist  = max(_median_dist, 0.2)   # floor: prevent infinite λ
        _prox_lambda  = float(np.log(2) / _median_dist)
        # M-5 FIX: clip bounds derived from distribution of observed lambda values
        _reg_lam = _reg.get("prox_lambda", [])
        _reg_lam.append(_prox_lambda)
        _reg["prox_lambda"] = _reg_lam[-200:]
        if len(_reg_lam) >= 10:
            _lam_lo = float(np.percentile(_reg_lam, 5))
            _lam_hi = float(np.percentile(_reg_lam, 95))
        else:
            _lam_lo, _lam_hi = 0.5, 3.0   # bootstrap fallback only until 10 observations
        _prox_lambda  = float(np.clip(_prox_lambda, _lam_lo, _lam_hi))
    else:
        _prox_lambda = 1.0   # fallback

    if setup_type == "Breakout":
        d_trig = (base_hi - ltp_score) / (atr_v + 1e-9)
        # IDEAL_D = the stock's own historical "last bar before breakout" distance.
        # Method: find every bar where close was below resistance but next bar
        # crossed above → that bar's distance = optimal pre-breakout entry distance.
        # Median of these = this stock's natural approach distance before its moves.
        # No arbitrary floor — entirely derived from OHLCV history.
        _resist_series = hh.rolling(20).max().shift(1)   # 20d resistance at each bar
        _below_mask = hc < _resist_series                # bars below resistance
        _above_next = hc.shift(-1) > _resist_series      # next bar crossed above
        # Use .values arrays to avoid pandas index alignment errors
        _n_bo = min(len(hc), len(_resist_series.dropna()), len(atr) - 1)
        _hc_bo   = hc.values[-_n_bo:].astype(float)
        _res_bo  = _resist_series.values[-_n_bo:].astype(float)
        _atr_bo  = atr.values[-(_n_bo + 1):-1].astype(float)
        _valid   = ~np.isnan(_res_bo)
        _below_v = _valid & (_hc_bo < _res_bo)
        _above_n = np.zeros(len(_hc_bo), dtype=bool)
        _above_n[:-1] = _valid[:-1] & (_hc_bo[1:] > _res_bo[:-1])
        _bo_entry_v = _below_v & _above_n
        _dist_bo = np.where(_valid & (_atr_bo > 0), (_res_bo - _hc_bo) / _atr_bo, np.nan)
        if _bo_entry_v.sum() >= 3:
            _d = _dist_bo[_bo_entry_v]
            _d = _d[(_d > 0) & ~np.isnan(_d)]
            _IDEAL_D = float(np.median(_d)) if len(_d) >= 2 else float(1.0 / _prox_lambda)
        else:
            _d = _dist_bo[(_dist_bo > 0) & ~np.isnan(_dist_bo)]
            _IDEAL_D = float(np.median(_d)) if len(_d) >= 5 else float(1.0 / _prox_lambda)
        _d_adj = abs(d_trig - _IDEAL_D)
        if d_trig < 0:   # already above resistance: symmetric extra penalty
            _d_adj += abs(d_trig)
        prox_pts = round(10.0 * np.exp(-_prox_lambda * _d_adj), 1)
        prox_pts = max(0.0, min(10.0, prox_pts))
    else:
        # Pullback IDEAL_D: distance above EMA20 at bars just before EMA9 crossed EMA20.
        # Uses numpy arrays to avoid pandas index-alignment errors.
        _de9  = ltp_score - e9_v
        _de20 = ltp_score - e20_v
        _closest_ema_dist = _de9 if abs(_de9) <= abs(_de20) else _de20

        # Align all series to common length via .values (no index issues)
        _n_common = min(len(hc), len(e9), len(e20), len(atr) - 1)
        _hc_arr   = hc.values[-_n_common:].astype(float)
        _e9_arr   = e9.values[-_n_common:].astype(float)
        _e20_arr  = e20.values[-_n_common:].astype(float)
        _atr_arr  = atr.values[-(  _n_common + 1):-1].astype(float)  # atr has one extra

        # EMA9 cross above EMA20: e9[i] > e20[i] and e9[i-1] <= e20[i-1]
        _cross_mask = np.zeros(len(_e9_arr), dtype=bool)
        if len(_e9_arr) >= 2:
            _cross_mask[1:] = (_e9_arr[1:] > _e20_arr[1:]) & (_e9_arr[:-1] <= _e20_arr[:-1])

        # Bar BEFORE each cross = pre-bounce entry point
        _pre_cross_mask = np.zeros(len(_e9_arr), dtype=bool)
        if _cross_mask.sum() >= 3:
            cross_idxs = np.where(_cross_mask)[0]
            pre_idxs   = cross_idxs[cross_idxs > 0] - 1
            _pre_cross_mask[pre_idxs] = True

        _dist_arr = (_hc_arr - _e20_arr) / (_atr_arr + 1e-9)

        if _pre_cross_mask.sum() >= 2:
            _pb_entry_dists = _dist_arr[_pre_cross_mask]
            _pb_entry_dists = _pb_entry_dists[_pb_entry_dists > 0]
            _IDEAL_D_PB = float(np.median(_pb_entry_dists)) if len(_pb_entry_dists) >= 2 else float(1.0 / _prox_lambda)
        else:
            _pos_dists = _dist_arr[_dist_arr > 0]
            _IDEAL_D_PB = float(np.median(_pos_dists)) if len(_pos_dists) >= 5 else float(1.0 / _prox_lambda)

        _d_from_ideal = (_closest_ema_dist / (atr_v + 1e-9)) - _IDEAL_D_PB
        prox_dist = abs(_d_from_ideal) if _d_from_ideal >= 0 else abs(_d_from_ideal) * 1.5
        prox_pts  = round(10.0 * np.exp(-_prox_lambda * prox_dist), 1)
        prox_pts  = max(0.0, min(10.0, prox_pts))

    # ═══════════════════════════════════════════════════════
    # F9 — ATR POTENTIAL  (0-5 pts)
    # Dynamic: score relative to the 60d rolling ATR% distribution
    # of this stock — not fixed global thresholds.
    # Scores highest when today's ATR% is in the top quartile
    # of its own 60d history (i.e. stock is more volatile than usual).
    # ═══════════════════════════════════════════════════════
    # F9 ATR COMPRESSION — low ATR = coiled = pre-move
    # FIX: atr14(df) used the FULL df including the signal bar (look-ahead).
    # atr14(hist) uses only the T-1 and earlier slice — consistent with all other factors.
    # Divide by hist closes (hc) so the ATR% history is look-ahead free end-to-end.
    atr_hist_pct = atr14(hist) / (hc.replace(0, np.nan)) * 100
    atr_hist_pct = atr_hist_pct.tail(60).dropna()
    if len(atr_hist_pct) >= 10:
        atr_pct_rank = float((atr_hist_pct <= atr_pct).mean())
        atp_pts = round((1.0 - atr_pct_rank) * 5, 1)
    else:
        _atr_lo = float(atr_hist_pct.min()) if len(atr_hist_pct) > 0 else 0.0
        _atr_hi = float(atr_hist_pct.max()) if len(atr_hist_pct) > 0 else 5.0
        _atr_hi = max(_atr_hi, _atr_lo + 0.1)
        atp_pts = round(float(np.clip((1.0-(atr_pct-_atr_lo)/(_atr_hi-_atr_lo))*5, 0.0, 5.0)), 1)

    # ═══════════════════════════════════════════════════════
    # F10 — CANDLESTICK TRIGGER  (0-5 pts)
    # ═══════════════════════════════════════════════════════
    raw_cdl, candle_names = detect_candle_patterns(
        day_o, day_hi, day_lo, ltp_score,
        float(o.iloc[-2]), float(h.iloc[-2]),
        float(l.iloc[-2]), float(c.iloc[-2])
    )
    # FIX 10: Context-aware candle weighting.
    # An Engulfing after 5 compressing bars is far more reliable than one after a trending run.
    # Multiplier: if the prior 5 bars were compressing (vc_ratio < own 40th pct), apply 1.3× bonus.
    # If prior 5 bars were trending up (close rising each bar), apply 0.7× penalty.
    _vc_p40_cdl = float(_vc_series.dropna().quantile(0.40)) if len(_vc_series.dropna()) >= 10 else 0.9
    _prior_5_closes = hc.tail(5).values
    _prior_trending = all(_prior_5_closes[i] < _prior_5_closes[i+1] for i in range(len(_prior_5_closes)-1))
    _prior_compressing = vc_ratio < _vc_p40_cdl
    if _prior_compressing and not _prior_trending:
        raw_cdl = min(raw_cdl * 1.3, 10)  # compressing base = more reliable pattern
    elif _prior_trending and not _prior_compressing:
        raw_cdl = raw_cdl * 0.7           # trending = mean-reversion risk, discount pattern
    cdl_pts = min(round(raw_cdl * 0.5, 1), 5.0)

    # ── DARVAS BOX FACTOR (0-10 pts) — FULL FACTOR, BOTH SETUPS ──
    # Old: 0-2 pt bonus, Breakout only, multiplicative formula (sparse signal)
    # New: proper 0-10 pt factor via darvas_box_score(), both setups.
    #   Breakout: high score when price is near box top (pressing resistance)
    #   Pullback: high score when price is near box bottom (testing support)
    _darvas_result = darvas_box_score(hist, atr_v)
    _d_box_hi = _darvas_result.get("box_high", np.nan)
    _d_box_lo = _darvas_result.get("box_low",  np.nan)
    if not (np.isnan(_d_box_hi) or np.isnan(_d_box_lo)):
        _d_width = _d_box_hi - _d_box_lo + 1e-9
        _d_pos   = (ltp_score - _d_box_lo) / _d_width
        _d_pos_score = float(np.clip(_d_pos, 0.0, 1.0)) if setup_type == "Breakout" \
                       else float(np.clip(1.0 - _d_pos, 0.0, 1.0))
        _d_atr_ratio = float(_darvas_result.get("box_atr_ratio", 1.0) or 1.0)
        if _d_atr_ratio != _d_atr_ratio: _d_atr_ratio = 1.0
        _d_tightness = 1.0 / (1.0 + _d_atr_ratio)
        _d_time = float(np.clip(_darvas_result.get("bars_in_box", 0) / 20.0, 0.0, 1.0))
        darvas_pts = round((_d_tightness*0.40 + _d_pos_score*0.35 + _d_time*0.25)*10.0, 1)
    else:
        darvas_pts = _darvas_result["darvas_score"]
    darvas_bonus = 0.0

    # ═══════════════════════════════════════════════════════════════
    # NEW PREDICTIVE FACTORS — read universe cross-sectional ranks
    # computed BEFORE the scoring loop (stored in session_state).
    # These are universe percentiles (0=bottom, 1=top) so they are
    # already cross-sectionally ranked — no further normalization needed.
    # In bt_mode these may be absent; fall back to 0.5 (neutral).
    # ═══════════════════════════════════════════════════════════════

    # F_BB — Bollinger Band Width Compression (universe percentile, 0-8 pts)
    # Low BB width = price coiling = pre-expansion signal.
    # Universe rank ensures only the most compressed stocks score highest.
    _bb_cs_pct = st.session_state.get("cs_bb_squeeze", {}).get(ticker, None)
    if _bb_cs_pct is None:
        # Fallback: compute self-calibrated score from own history only
        _, _bb_self = bb_width_compression_score(hc)
        _bb_cs_pct = _bb_self
    bb_pts = round(float(_bb_cs_pct) * 8.0, 1)   # 0-8 pts

    # F_VDU — Volume Dry-Up (universe percentile, 0-8 pts)
    # Declining 5d avg vol BEFORE a breakout = supply exhaustion.
    # Orthogonal to today's vol surge (which belongs in vol_pts).
    # Only meaningful for Breakout setup — pullbacks may show dry vol too,
    # but the interpretation is different (risk of further decline).
    _vdu_cs_pct = st.session_state.get("cs_vol_dryup", {}).get(ticker, None)
    if _vdu_cs_pct is None:
        _, _vdu_self = volume_dryup_score(hv)
        _vdu_cs_pct = _vdu_self
    if setup_type == "Breakout":
        vol_dryup_pts = round(float(_vdu_cs_pct) * 8.0, 1)   # 0-8 pts
    else:
        # For Pullback: moderate dry-up is ok; extreme dry-up may signal illiquidity.
        # Use a tent-shaped score centred at the 60th percentile (some dry-up good, excessive bad).
        _vdu_tent = float(np.clip(1.0 - abs(_vdu_cs_pct - 0.60) / 0.40, 0.0, 1.0))
        vol_dryup_pts = round(_vdu_tent * 5.0, 1)   # 0-5 pts for Pullback

    # F_CLV — CLV Institutional Accumulation (DIAGNOSTIC ONLY — not in primary score)
    # IC=-0.260 on NSE (Jan 2026 backtest): anti-predictive. Excluded from total.
    # Shown as "CLV (diag)" in screener table — use it to manually verify if
    # a stock with strong BBSqueeze + VolDryUp also has money flow supporting it.
    # High CLV alongside low VCP/BB = accumulation without compression = less reliable.
    # Low CLV alongside high BB squeeze = compression without buyers = wait for confirmation.
    _clv_cs_pct = st.session_state.get("cs_clv_accum", {}).get(ticker, None)
    if _clv_cs_pct is None:
        _, _clv_self = clv_accumulation_score(hc, hh, hl, hv)
        _clv_cs_pct = _clv_self
    clv_pts = round(float(_clv_cs_pct) * 8.0, 1)   # stored for display — weight=0 in score

    # F_VCP — Volatility Contraction Pattern (universe percentile — DIAGNOSTIC ONLY)
    # IC=-0.186 on NSE (Jan 2026 backtest): anti-predictive in small universe.
    # VCP score is computed and stored for the chart/table but NOT added to total.
    # It is shown as "VCP (diag)" in the screener to guide manual review:
    # a high VCP stock is forming the PATTERN — it still needs vol confirmation before entry.
    # Weight = 0 in score formula. Use it visually alongside BBSqueeze + VolDryUp.
    _vcp_cs_pct = st.session_state.get("cs_vcp", {}).get(ticker, None)
    if _vcp_cs_pct is None:
        _vcp_result_inline = detect_vcp(c, h, l, v, atr)
        _vcp_cs_pct = _vcp_result_inline["vcp_score"]
        _vcp_detail = _vcp_result_inline
    else:
        _vcp_detail = detect_vcp(c, h, l, v, atr)
    vcp_pts = round(float(_vcp_cs_pct) * 10.0, 1)   # stored for display — weight=0 in score

    # ── PREDICTIVE vs CONFIRMATORY SIGNAL SEPARATION ──────────────────────────
    # The quintile inversion diagnosis: high-scoring stocks had strong trailing RS,
    # strong MA trend, and strong CLV — all CONFIRMATORY (measuring what already happened).
    # Low-scoring stocks had compression, dry vol, VCP forming — PREDICTIVE (what's about to happen).
    # The old weight structure gave 40%+ to RS + MA (confirmatory), driving the inversion.
    #
    # Fix: separate signals into two tiers:
    #
    # TIER 1 — PREDICTIVE (what precedes a move): VCP, BB squeeze, vol dry-up, CLV, ATR compression
    #   These measure conditions BEFORE price expansion. High weight = 65% of total.
    #
    # TIER 2 — CONFIRMATORY (what confirms a setup exists): RS level, MA trend, sector
    #   These provide directional context but are mean-reverting at 5d horizon on NSE.
    #   Low weight = 35% of total. They act as quality gates, not primary drivers.
    #
    # RS ACCELERATION (change in RS, not level) is predictive — kept at moderate weight.
    # RS LEVEL (trailing 20d outperformance) is confirmatory — weight cut by 60%.
    #
    # Mean-reversion penalty: stocks with very high RS level (top quintile) get an
    # explicit penalty because they are the most mean-reverting over 1-5d on NSE.
    # This is derived from the stock's own RS distribution, not a fixed threshold.

    # ── RS MEAN-REVERSION PENALTY (new) ──────────────────────────────────────
    # If cs_rs_score (the trailing RS rank) is in the top quintile of the universe,
    # this stock is the most likely to mean-revert. Penalize proportionally.
    # Penalty = 0 at cs_rs_score = 0.80, grows to max at cs_rs_score = 1.0.
    # Formula: tanh((cs_rs_score - 0.80) / 0.10) × penalty_max
    # penalty_max derived from the distribution of cs_rs_score × forward_return
    # correlation: empirically ~8 pts on NSE at 5d horizon.
    # ── FACTOR ASSEMBLY — DATA-DRIVEN WEIGHTS (v7) ────────────────────────────
    # Weights derived from measured IC on actual backtest data (Jan 27 2026, Nifty50):
    #   -VolRatio:   IC=+0.723  ← PRIMARY. Quiet stocks outperform.
    #   SpreadComp:  IC=+0.648  ← PRIMARY. Rising close + narrow spread.
    #   F_Prox:      IC=+0.377  ← Proximity to trigger.
    #   VCP:         IC=-0.186  ← Anti-predictive in small universe. Weight cut.
    #   RS (level):  IC=-0.076  ← Anti-predictive. Excluded.
    #   CLV:         IC=-0.260  ← Anti-predictive. Excluded.

    # Vol-quiet score: use T-1 confirmed volume (hv.iloc[-1] is the last FULL day in hist).
    # IMPORTANT: hv here is the historical slice (hist["volume"]), so iloc[-1] is T-1,
    # not the live partial bar.  The percentile is ranked against the 20 full-day bars
    # preceding T-1 (shift(1) on rolling mean) to keep the reference window look-ahead free.
    _vr_confirmed  = hv                                                   # historical slice only
    _vr_rolling20  = _vr_confirmed.rolling(20).mean().shift(1)           # 20d avg ending day BEFORE each bar
    _vr_ratio_hist = (_vr_confirmed / (_vr_rolling20 + 1e-9)).dropna()  # ratio for every confirmed bar
    _vr_now        = float(_vr_ratio_hist.iloc[-1]) if len(_vr_ratio_hist) >= 1 else 1.0  # T-1 ratio
    _vr_hist_prior = _vr_ratio_hist.iloc[:-1]                            # history before T-1 (no self-reference)
    _quiet_pct = float((_vr_hist_prior >= _vr_now).mean()) if len(_vr_hist_prior) >= 10 \
                 else float(np.clip(1.0 - _vr_now, 0.0, 1.0))
    vol_quiet_pts = round(_quiet_pct * 14.0, 1)   # 0-14 pts

    # SpreadComp promoted to primary (was bonus max 3pts → now primary 0-11pts)
    # FIX 6: Cross-sectionally rank SpreadComp within the universe before mapping to pts.
    # Old: _sc_bonus was computed per-stock against own history only — a stock at its
    # own 80th percentile might only be at the 40th universe-percentile.
    # New: read universe percentile from session_state.cs_spread_comp if available,
    # else fall back to own-history score (backward-compatible).
    _sc_cs_pct = st.session_state.get("cs_spread_comp", {}).get(ticker, None) if not bt_mode else None
    if _sc_cs_pct is not None:
        spread_pts = round(float(_sc_cs_pct) * 11.0, 1)
    else:
        spread_pts = round(float(np.clip(_sc_bonus / 3.0, 0.0, 1.0)) * 11.0, 1)

    # ── REVERSAL SCORING — completely separate from Breakout/Pullback ──────
    # Factors with positive IC on mean-reversion bounce days:
    #   1. Oversold depth  — how far RSI below 30 (deeper = stronger snap)
    #   2. Vol spike       — panic capitulation magnitude (higher = cleaner washout)
    #   3. Washout depth   — price distance below 10d high in ATR units
    #   4. Candle tail     — close position in bar range (wick rejection strength)
    #   5. Proximity to key support (SMA200, EMA50) — structural bounce zone
    #   6. Sector context  — oversold sectors bounce together
    #
    # Penalties (things that kill bounces):
    #   - Fundamental breakdown signal: price below SMA200 by > 5 ATR (structural bear)
    #   - Very low liquidity (panic in illiquid stock = no buyers for the bounce)
    #   - Continued distribution: closing in LOWER half on high vol = distribution not washout
    # liquidity_score needed by both Reversal composite_rank and the main composite_rank below
    _liq_logadv    = float(np.log(_adv_turnover + 1.0))
    _LIQ_CENTRE    = float(np.log(5e7))
    _LIQ_SCALE     = 1.0
    liquidity_score = float(1.0 / (1.0 + np.exp(-_LIQ_SCALE * (_liq_logadv - _LIQ_CENTRE))))

    if setup_type == "Reversal":
        # Reversal scoring derived from measured IC on actual backtest data.
        # What separates good Reversal stocks from bad ones:
        #   POWERGRID(+11.3%): F_Coil=7.7, F_Prox=6.7, RSI=35
        #   APOLLOHOSP(+4.1%): SpreadComp=2.2, RSI=22
        #   vs TATACONSUM(-3%): F_Coil=4.4, F_Prox=2.8, low SpreadComp
        #
        # Vol spike is NOT the primary predictor — many recoveries are quiet.
        # The primary signals: how oversold + base forming at lows + near support.

        # ── Factor 1: RSI oversold depth (0-40 pts) ──
        # How far below own 60d p90 RSI is this stock?
        # Uses stock's OWN history — a stock with typical RSI of 60 at 35 is
        # more oversold than a stock with typical RSI of 40 at 35.
        _rsi_p90_rev = float(_rsi_hist_full.tail(60).quantile(0.90)) if len(_rsi_hist_full) >= 20 else 70.0
        _rsi_p10_rev = float(_rsi_hist_full.tail(60).quantile(0.10)) if len(_rsi_hist_full) >= 20 else 25.0
        _rsi_range   = max(_rsi_p90_rev - _rsi_p10_rev, 10.0)
        _rsi_oversold_depth = float(np.clip((_rsi_p90_rev - rsi_v) / _rsi_range, 0.0, 1.0))
        _rev_rsi_pts = round(40.0 * _rsi_oversold_depth, 1)

        # ── Factor 2: Coil quality at the lows (0-30 pts) ──
        # A base forming at the bottom = buyers absorbing sellers = spring loading.
        # Condition: coil_pts measures compression, but a Reversal coil must also show
        # closes forming in the UPPER HALF of each bar's range — buyers stepping in.
        # A tight range with closes at the lows = continued distribution, not accumulation.
        # Compute fraction of last 5 bars closing in upper half of their range.
        _rev_close_quality = 0.5   # neutral default
        if len(hc) >= 5 and len(hh) >= 5 and len(hl) >= 5:
            _rev_rng5   = (hh.iloc[-5:] - hl.iloc[-5:]).replace(0, np.nan)
            _rev_cpr5   = ((hc.iloc[-5:] - hl.iloc[-5:]) / _rev_rng5).dropna()
            _rev_close_quality = float((_rev_cpr5 > 0.50).sum()) / max(len(_rev_cpr5), 1)
        # Score: coil quality × close quality
        # A tight coil closing at lows scores 0; tight coil closing at highs scores full.
        _rev_coil_pts = round(30.0 * float(np.clip(coil_pts / 10.0, 0.0, 1.0))
                              * float(np.clip(_rev_close_quality * 2.0, 0.0, 1.0)), 1)

        # ── Factor 3: Proximity to structural support (0-20 pts) ──
        # For a Reversal, "support" means EMA50 or SMA200 — structural bounce zones.
        # prox_pts was computed using base_hi (20d high resistance) as the target —
        # the right metric for Breakout but WRONG for Reversal: a deep washout sitting
        # 3 ATR below resistance (low prox_pts) is EXACTLY the best Reversal candidate.
        # Fix: derive support proximity from distance to EMA50 and SMA200, ATR-normalised.
        # Score peaks when price is within 1 ATR of either support level.
        _dist_ema50_atr  = abs(ltp_score - e50_v) / (atr_v + 1e-9)
        _dist_sma200_atr = abs(ltp_score - _sma200) / (atr_v + 1e-9)
        # Derive natural approach distance from own history (same as breakout IDEAL_D logic)
        # "At support" = within the stock's own typical pre-bounce distance
        if len(hl) >= 30:
            _supp_dist_hist = ((hc - e50.iloc[:-1]).abs() / (atr.iloc[:-1] + 1e-9)).dropna().tail(60)
            _supp_ideal = float(_supp_dist_hist.quantile(0.20)) if len(_supp_dist_hist) >= 10 else 1.0
            _supp_ideal = max(_supp_ideal, 0.2)
        else:
            _supp_ideal = 1.0
        _supp_decay_lambda = float(np.log(2.0) / max(_supp_ideal, 0.2))
        _supp_prox_ema50  = float(np.exp(-_supp_decay_lambda * _dist_ema50_atr))
        _supp_prox_sma200 = float(np.exp(-_supp_decay_lambda * _dist_sma200_atr))
        _rev_support_raw  = max(_supp_prox_ema50, _supp_prox_sma200)   # nearest support wins
        _rev_prox_pts     = round(20.0 * float(np.clip(_rev_support_raw, 0.0, 1.0)), 1)
        _rev_support_pts  = _rev_prox_pts   # alias for return dict

        # ── Factor 4: Range compression at lows (0-10 pts) ──
        # SpreadComp = range narrowing + close rising = coiling energy at the bottom.
        _rev_spread_pts = round(10.0 * float(np.clip(_sc_bonus / 3.0, 0.0, 1.0)), 1)

        # Keep vol spike as a minor diagnostic — stored for return dict but low weight
        _rev_vol_pct = float((hv.iloc[:-1] <= float(hv.iloc[-1])).mean())
        _rev_vol_pts = round(5.0 * _rev_vol_pct, 1)   # 0-5 pts (minor)

        # Keep washout and tail for return dict (diagnostics)
        _rev_wash_score = float(np.clip((_washout_depth - 1.5) / 4.0, 0.0, 1.0))
        _rev_wash_pts   = round(10.0 * _rev_wash_score, 1)
        _rev_tail_pts   = round(5.0 * float(np.clip((_t1_close_pos - 0.30) / 0.70, 0.0, 1.0)), 1)

        # ── Penalty: structural downtrend (price well below SMA200) ──
        _rev_penalty = 0.0
        if _sma200_gap_atr < -5.0:
            _rev_penalty += float(np.clip((_sma200_gap_atr + 5.0) * -3.0, 0.0, 15.0))
        if liquidity_score < 0.2:   # very illiquid — derived from own ADV sigmoid
            _rev_penalty += 10.0 * (0.2 - liquidity_score) / 0.2

        # ── Total ──
        # _llhl_bonus captures the structural turn signal (HL forming) — Reversal-specific.
        _rev_raw = _rev_rsi_pts + _rev_coil_pts + _rev_prox_pts + _rev_spread_pts + _llhl_bonus
        total    = round(max(0.0, min(100.0, _rev_raw - _rev_penalty)), 1)

        emi        = round(total * atr_pct / 100, 3)
        # FIX I: composite_rank for Reversal is computed BELOW after _nifty_breadth_adj
        # and _vix_adj are applied to total (line ~3842).
        # Old code set composite_rank HERE using pre-adjustment total — so Reversal setups
        # were immune to market context scoring. A textbook panic bottom in a BEAR market
        # (the highest-probability reversal) received no breadth discount.
        # composite_rank is now set after the breadth/VIX block for ALL setup types.

    # ── UNIFIED SCORE ASSEMBLY — identical formula for ALL setup types ──────
    # Derived from grid-search on actual Jan 2026 Nifty50 backtest data.
    # Best 5d return spread (+2.4%): SpreadComp(40%) + vol_direction(40%) + Coil(20%)
    #
    # SpreadComp: range compressing + rising close = spring loading
    # vol_direction: setup-aware
    #   Reversal → vol_surge (high panic vol = capitulation = bounce fuel)
    #   Breakout/Pullback → vol_quiet (low vol = accumulation before move)
    # Coil: base quality at approach level — same signal for all setups
    #
    # Using a single formula means Reversal scores compete directly with
    # Breakout/Pullback scores on the same 0-100 scale.
    _t1_vol_ratio_pb = float(hv.iloc[-1]) / (vol_ma20 + 1e-9)
    vol_surge_pts    = round(float(np.clip((_t1_vol_ratio_pb / 3.0) * 14.0, 0.0, 14.0)), 1)
    _stab_pts_pb = _cpr_raw_pb = _stab_pts_assembly = _cpr_pts_assembly = 0.0
    _W_BB = _W_PROX = _W_VC = _W_VCP = _W_VOL_DRYUP = _W_STAB = _W_CPR = 0.0

    if setup_type == "Reversal":
        _primary_vol_pts = vol_surge_pts   # panic vol = bounce fuel
    else:
        _primary_vol_pts = vol_quiet_pts   # quiet vol = accumulation
    _primary_vol_max = 14.0

    # ── FACTOR WEIGHTS — ADAPTIVE (updated by walk-forward IC feedback) ──────
    # Weights start as Jan 2026 Nifty50 backtest priors.
    # After each walk-forward run they are blended toward the measured IC spread.
    # In bt_mode: always use the fixed priors — no adaptive weights in backtest
    # (would create look-ahead bias from future walk-forward data).
    if bt_mode:
        _W_SPREAD    = 0.40
        _W_VOL_QUIET = 0.40
        _W_COIL      = 0.20
    else:
        _aw = st.session_state.get("adaptive_weights",
                                   {"spread": 0.40, "vol": 0.40, "coil": 0.20})
        _W_SPREAD    = float(_aw.get("spread", 0.40))
        _W_VOL_QUIET = float(_aw.get("vol",    0.40))
        _W_COIL      = float(_aw.get("coil",   0.20))
        # Safety: re-normalise in case session_state was written with rounding error
        _w_sum = _W_SPREAD + _W_VOL_QUIET + _W_COIL
        if abs(_w_sum - 1.0) > 0.01:
            _W_SPREAD /= _w_sum; _W_VOL_QUIET /= _w_sum; _W_COIL /= _w_sum

    _MAX_SPREAD = 11.0; _MAX_VOL_QUIET = 14.0; _MAX_COIL = 10.0
    _MAX_BB = _MAX_PROX = _MAX_VC = _MAX_VCP = _MAX_VDRYUP = _MAX_STAB = _MAX_CPR = 10.0

    # FIX 2: Full 10-factor score assembly. Previously _weighted_raw only contained
    # SpreadComp + VolQuiet + Coil — every other factor (RS, Sector, InstVol, VC,
    # MA, Proximity, ATR, Candle) was computed but unused in the total.
    # Now all factors contribute proportionally via their adaptive weights.
    # The three primary factors retain their dominant role (default 40/40/20 split
    # of the 0-35 primary pool), while secondary factors fill the remaining 0-65.
    _weighted_raw = (
        _W_SPREAD    * spread_pts       +
        _W_VOL_QUIET * _primary_vol_pts +
        _W_COIL      * coil_pts         +
        # Secondary factors (fixed fractional weight within 65% pool)
        0.06 * rs_pts        +   # F1: RS vs Nifty (was anti-predictive at 15pt weight; reduced)
        0.05 * rs_sect_pts   +   # F2: Sector RS
        0.04 * inst_pts      +   # F4: Pre-BO accumulation
        0.05 * vc_pts        +   # F5: Volatility contraction
        0.05 * ma_pts        +   # F7: MA structure
        0.05 * prox_pts      +   # F8: Breakout proximity
        0.02 * atp_pts       +   # F9: ATR potential
        0.02 * cdl_pts           # F10: Candle pattern
    )
    _weighted_max = (
        _W_SPREAD    * _MAX_SPREAD      +
        _W_VOL_QUIET * _primary_vol_max +
        _W_COIL      * _MAX_COIL        +
        0.06 * 15.0  +   # rs_pts max
        0.05 * 10.0  +   # rs_sect_pts max
        0.04 * 10.0  +   # inst_pts max
        0.05 * 10.0  +   # vc_pts max
        0.05 * 10.0  +   # ma_pts max
        0.05 * 10.0  +   # prox_pts max
        0.02 * 5.0   +   # atp_pts max
        0.02 * 5.0       # cdl_pts max
    )
    _scale_to_100 = 100.0 / max(_weighted_max, 1e-9)
    total_base    = round(_weighted_raw * _scale_to_100, 1)
    total = round(total_base, 1)

    # Preserve raw values for display/debug
    rs_pts_raw,  vol_pts_raw,  rs_sect_pts_raw = rs_pts,  vol_pts,  rs_sect_pts
    inst_pts_raw, vc_pts_raw,  coil_pts_raw    = inst_pts, vc_pts,  coil_pts
    ma_pts_raw,  prox_pts_raw                  = ma_pts,  prox_pts

    # ── BONUS SIGNALS (additive, absolutely capped) ──
    # Each signal documented with its maximum:
    #   rs_accel_bonus   ≤ 4   (EMA momentum acceleration — leading)
    #   rs_div_bonus     ≤ 3   (5d/20d RS divergence — early rotation — leading)
    #   _uv_bonus        ≤ 3   (upside vol skew — quiet accumulation — universal)
    #   _cpr_bonus       ≤ 3   (close position rank — demand absorbing supply — universal)
    #   _sc_bonus        ≤ 3   (spread compression + rising close — universal)
    #   _atr_exp_bonus   ≤ 3   (ATR expansion onset — coil releasing — universal)
    #   oi_bonus         ≤ 3   (OI buildup — F&O stocks only)
    #   _vol_velocity    ≤ 3   (intraday vol velocity — universal)
    #   vcve_bonus       ≤ 3   (vol × compression interaction)
    #   sweep_bonus      ≤ 4   (liquidity sweep reversal)
    #   vwap_bonus       ≤ 3   (price above VWMA20)
    #   stab_bonus       ≤ 2   (trend stability, can be negative)
    #   pos52w_bonus     ≤ 3   (near 52w high = momentum leader)
    #   darvas_bonus     ≤ 2   (Darvas box quality — Breakout only)
    # Raw max ≈ 45 pts.
    #
    # SIGNAL STABILITY: bonuses are multiplied by a persistence factor.
    # If a signal fires on only 1 day out of the last 3 (isolated spike),
    # it gets 50% weight. Signals persistent over 2-3 days get full weight.
    # This prevents single-bar anomalies from boosting weak stocks.
    # Compression signals (BB, VC, range) must be sustained to count.
    # ── BONUS SIGNALS — COMPRESSION-ONLY (additive, capped) ──
    # RS momentum bonuses (rs_accel_bonus, rs_div_bonus) are REMOVED.
    # Reason: they are the single largest source of the quintile inversion.
    # rs_accel and rs_div both measure trailing momentum which mean-reverts
    # at 5d horizon on NSE, especially in Bear/Chop regimes.
    # The compression signals below have no such mean-reversion property.
    #
    # pos52w_bonus is also removed — near 52w highs = already moved = mean reverts.
    # stab_bonus kept: trend stability is structural, not momentum.
    # ── PERSISTENCE FACTORS — signal-specific, 5-bar window, majority rule ──────
    # Old: one global factor from 3-bar VC-only window → too short (NSE has 1-2 day
    # interruptions from results/expiry/macro), and VC compression has nothing to do
    # with whether upside-volume-skew or CPR have been sustained.
    # New: each signal class has its own 5-bar persistence check.
    #   "at least 3 of 5 bars compressed/confirmed" → full weight
    #   "2 of 5" → 0.75 weight
    #   "1 of 5" → 0.50 weight (isolated spike, still possible but discounted)
    # VC-based signals (streak, ib, mtf, cs) share vc_persist.
    # Volume-based signals (uv, vcve, churn) share vol_persist.
    # Direction signals (hhhl, cpr, sc) share dir_persist.
    _vc_clean     = _vc_series.dropna()
    _vc_p40_5b    = float(_vc_clean.quantile(0.40)) if len(_vc_clean) >= 10 else 0.9
    _vc_last5     = _vc_clean.iloc[-6:-1] if len(_vc_clean) >= 6 else _vc_clean
    _vc_comp5     = int((_vc_last5 < _vc_p40_5b).sum())
    _vc_persist   = float(np.clip(0.50 + (_vc_comp5 / max(len(_vc_last5), 1)) * 0.50, 0.50, 1.0))

    # Volume persistence: fraction of last 5 bars where up-vol > down-vol
    if len(hc) >= 6 and len(hv) >= 6:
        _up_days5   = int((hc.diff().iloc[-6:-1] > 0).sum())
        _vol_persist = float(np.clip(0.50 + (_up_days5 / 5.0) * 0.50, 0.50, 1.0))
    else:
        _vol_persist = 0.75

    # Direction persistence: fraction of last 5 bars closing in upper half of range
    if len(hc) >= 6 and len(hh) >= 6 and len(hl) >= 6:
        _hl_rng5    = (hh.iloc[-6:-1] - hl.iloc[-6:-1]).replace(0, np.nan)
        _cpr5       = ((hc.iloc[-6:-1] - hl.iloc[-6:-1]) / _hl_rng5).dropna()
        _dir_days5  = int((_cpr5 > 0.50).sum())
        _dir_persist = float(np.clip(0.50 + (_dir_days5 / max(len(_cpr5), 1)) * 0.50, 0.50, 1.0))
    else:
        _dir_persist = 0.75

    # Backward-compat alias — used in the bonus assembly below where a single
    # _persist_factor was applied to all signals together.  We now apply
    # signal-specific factors inline; _persist_factor is kept as the VC one
    # for the legacy path that multiplies the whole _bonus_raw block.
    _persist_factor = _vc_persist

    # ── COMPRESSION STREAK SIGNAL (0-4 pts bonus) ────────────────────────────
    # Problem: _persist_factor only looks back 3 bars — too short for real coils.
    # A genuine base/coil needs 5-10 consecutive days of narrowing range.
    # This signal counts how many consecutive bars have had a daily range
    # (high-low) below the stock's 20-bar average range — the "narrow bar" count.
    #
    # Streak of 1-2 bars: noise, no bonus.
    # Streak of 3-4 bars: starting to coil, small bonus.
    # Streak of 5-7 bars: genuine coil forming, meaningful bonus.
    # Streak of 8+ bars:  textbook base, maximum bonus.
    #
    # Why use avg range not ATR: ATR lags and smooths; raw range captures
    # today's actual compression vs recent trading activity directly.
    _compression_streak = 0
    _streak_bonus = 0.0
    if len(hc) >= 25:
        _daily_range   = (hh - hl).iloc[:-1]   # historical ranges, no look-ahead
        _avg_range_20  = float(_daily_range.tail(20).mean())
        if _avg_range_20 > 0:
            # FIX 8: Replace hardcoded 85% threshold with stock's own 30th percentile range.
            # Volatile stocks (ATR% > 3%) naturally have wider daily ranges — their "narrow"
            # day is at a different absolute level than a calm stock's narrow day.
            # Using the stock's own 30th percentile range as the "narrow" threshold is
            # self-calibrating: any bar below its own historical 30th percentile is genuinely narrow.
            _range_arr_full = _daily_range.values
            _range_p30 = float(np.percentile(_range_arr_full[_range_arr_full > 0], 30)) \
                         if (_range_arr_full > 0).sum() >= 10 else _avg_range_20 * 0.85
            # Walk backwards from T-1 counting consecutive narrow bars
            _range_arr = _daily_range.values[::-1]   # newest first
            for _rng in _range_arr:
                if float(_rng) < _range_p30:   # FIX 8: below own 30th percentile = "narrow"
                    _compression_streak += 1
                else:
                    break
            # Score: logarithmic so 5 bars is not 5x better than 1 bar
            if _compression_streak >= 3:
                _streak_raw = float(np.log2(_compression_streak - 1))   # 3→1, 5→2, 9→3
                _streak_bonus = round(float(np.clip(_streak_raw * 1.5, 0.0, 4.0)), 1)

    # FIX-06: TRUE INSIDE-BAR STREAK — containment structure, not just narrow range ──
    # A true inside bar: high[i] <= high[i-1] AND low[i] >= low[i-1].
    # Three consecutive inside bars = buyers AND sellers both refuse to push price —
    # a genuine standoff. Higher signal purity than the range-percentile streak above.
    _inside_bar_streak = 0
    _ib_bonus = 0.0
    if len(hh) >= 5:
        for _ib_i in range(len(hh) - 2, max(len(hh) - 12, 0), -1):
            if (float(hh.iloc[_ib_i]) <= float(hh.iloc[_ib_i - 1]) and
                    float(hl.iloc[_ib_i]) >= float(hl.iloc[_ib_i - 1])):
                _inside_bar_streak += 1
            else:
                break
        if _inside_bar_streak >= 2:
            _ib_bonus = round(float(np.clip((_inside_bar_streak - 1) * 1.5, 0.0, 4.0)), 1)

    if setup_type != "Reversal":
        # FIX H: consolidation_score() was defined but never called (dead code).
        # It measures how tight/clean the consolidation base is (0-1 ratio).
        # Now wired in: score on hist slice to avoid look-ahead, capped at 3 pts bonus.
        # Contributes to base-quality differentiation alongside _streak_bonus and coil_pts.
        _cs_raw   = consolidation_score(hist, window=15)   # 0.0-1.0
        _cs_bonus = round(float(np.clip(_cs_raw * 3.0, 0.0, 3.0)), 1)   # 0-3 pts

        _bonus_raw = (
            (_uv_bonus  + vcve_bonus + _churn_bonus) * _vol_persist +   # volume-class signals
            (_cpr_bonus + _sc_bonus  + _hhhl_bonus)  * _dir_persist +   # direction-class signals
            (_streak_bonus + _ib_bonus + _cs_bonus + _mtf_bonus + _atr_exp_bonus) * _vc_persist +  # compression-class
            oi_bonus + stab_bonus + sweep_bonus + _round_bonus           # independent signals, no persist haircut
            # FIX 18: _delivery_bonus excluded from _persist_factor multiplication.
            # Delivery % has nothing to do with price compression continuity.
            # It is added directly to _bonus_raw AFTER the persist_factor is applied.
        )
        _bonus_raw += _delivery_bonus   # FIX 18: delivery added post-persist_factor
        # FIX 3: Tiered bonus cap instead of hard 8-pt ceiling.
        # Old: all 11 signals capped to 8 pts total → a stock with 6 strong signals
        # scored identically to one with 2 strong signals.
        # New: cap scales with the number of signals firing.
        _n_firing = sum(1 for _v in [
            _uv_bonus, _cpr_bonus, _sc_bonus, _atr_exp_bonus,
            oi_bonus, vcve_bonus, sweep_bonus, stab_bonus,
            _streak_bonus, _hhhl_bonus, _delivery_bonus, _cs_bonus, _mtf_bonus, _churn_bonus, _ib_bonus, _round_bonus
        ] if _v > 0.5)
        _BONUS_CAP_ABS = float(np.clip(8.0 + max(0, _n_firing - 3) * 1.0, 8.0, 16.0))
        # FIX 13: Make bonus cap regime-adaptive.
        # In BEAR regime the primary score is already penalised by up to 8 pts from the
        # breadth/VIX adjustment. Applying the same hard 8-pt bonus cap on top doubly
        # compresses exceptional BEAR-regime leaders. In BEAR, widen the cap so strong
        # signals can still differentiate genuine leaders from mediocre setups.
        if _regime == "BEAR":
            _BONUS_CAP_ABS = min(_BONUS_CAP_ABS + 4.0, 20.0)  # wider in bear = more differentiation
        elif _regime == "CHOP":
            _BONUS_CAP_ABS = min(_BONUS_CAP_ABS + 2.0, 18.0)
        bonuses = round((_bonus_raw * (_BONUS_CAP_ABS / _bonus_raw)
                         if _bonus_raw > _BONUS_CAP_ABS else _bonus_raw), 1)
        total += bonuses
    else:
        bonuses = 0.0
        # FIX 15: Reversal setup also incorporates delivery bonus.
        # High delivery on reversal = real panic selling = informed money left → strong bounce.
        # Low delivery on reversal = intraday speculation → bounce less reliable.
        # Add directly to total (not through bonus pool which uses _persist_factor).
        total = round(max(0.0, min(100.0, total + _delivery_bonus)), 1)

    # ── APPLY ACCUMULATED SOFT PENALTIES ──
    # Reversal stocks bypass momentum penalties — the characteristics that signal
    # a reversal (high vol, below SMA200, gap-down) are exactly what the momentum
    # penalties target. Applying them would zero out legitimate reversal scores.
    if setup_type != "Reversal":
        total = max(0.0, total - _soft_penalty)

    # ── ADJUSTMENTS ──
    # Overextension handled by prox_pts IDEAL_D decay and soft_penalty (setup classification).

    # FIX I-10: Replace binary Nifty/VIX penalties with continuous breadth adjustment.
    # breadth_score = percentile of (% stocks above EMA20) vs 200d history.
    # We approximate breadth via Nifty position vs 20DMA and VIX percentile.
    # nifty_above_20dma → replaced by continuous nifty position score.
    # vix_falling → replaced by VIX trend strength.
    # Both derived from mkt data already fetched; no extra API calls needed.
    # FIX E: Replace Nifty 5d/20d returns as "breadth proxy" with a true breadth measure.
    # Nifty return ≠ market breadth — a narrow index rally hides broad deterioration.
    # True breadth = % of stocks in the loaded universe that are ABOVE their own 20-day EMA.
    # We compute this from raw_data_cache which is already in memory (no extra API calls).
    # If cache is empty (screener not yet run), fall back to Nifty-based approximation.
    # FIX (Perf): Breadth is pre-computed ONCE before the scoring loop and stored in
    # st.session_state.breadth_cache. Reading it here is O(1) instead of O(N×M).
    # If the cache is missing (first call), fall back to Nifty-based approximation.
    #
    # FIX (Q4 inversion): Breadth penalty must be modulated by the stock's own
    # cross-sectional RS. A stock in the 90th percentile of the universe while
    # breadth is only 30% is an exceptional outlier — penalising it flattens the
    # very signal we want. The penalty is scaled by (1 - cs_rs_score): stocks that
    # are BEATING the weak breadth get little or no penalty; stocks that are
    # IN LINE with the weak breadth get the full penalty.
    _nifty_breadth_adj = 0.0
    _breadth_cached = st.session_state.get("breadth_cache", None)
    if _breadth_cached is not None:
        _breadth = _breadth_cached
        if bt_mode:
            # BT FIX: breadth_hist in session_state is from LIVE runs, not historical.
            # Using live μ/σ to normalise a historical breadth reading produces wrong
            # z-scores — e.g. a historically normal 50% breadth looks like an outlier
            # when compared against live-session μ=0.65 after a strong rally.
            # In bt_mode use neutral priors: μ=0.50 (historical long-run mean for NSE),
            # σ=0.12 (typical cross-date dispersion). These are stable population params.
            _breadth_mu  = 0.50
            _breadth_sig = 0.12
        else:
            _breadth_hist = st.session_state.get("breadth_hist", [])
            # FIX 20: Use exponentially decay-weighted μ/σ if available (computed in pre-scoring block).
            # Falls back to simple mean/std if EWM stats not yet computed.
            _ewm_stats = st.session_state.get("breadth_hist_ewm")
            if _ewm_stats and len(_breadth_hist) >= 10:
                _breadth_mu  = float(_ewm_stats.get("mean", 0.50))
                _breadth_sig = float(_ewm_stats.get("std",  0.12))
            else:
                _breadth_mu   = float(np.mean(_breadth_hist)) if len(_breadth_hist) >= 5  else 0.50
                _breadth_sig  = float(np.std(_breadth_hist))  if len(_breadth_hist) >= 10 else 0.12
        _breadth_sig  = max(_breadth_sig, 0.03)
        _breadth_z    = (_breadth - _breadth_mu) / _breadth_sig
        _raw_breadth_adj = float(np.clip(6.0 * np.tanh(_breadth_z), -8.0, 4.0))
        if _raw_breadth_adj < 0:
            _penalty_scale = max(0.0, 1.0 - cs_rs_score)
            _nifty_breadth_adj = _raw_breadth_adj * _penalty_scale
        else:
            _nifty_breadth_adj = _raw_breadth_adj
    elif nifty_r5 is not None and nifty_r20 is not None:
        _n5  = nifty_r5  or 0.0
        _n20 = nifty_r20 or 0.0
        _nifty_breadth_adj = float(np.clip((_n5 + _n20 * 0.5) * 100, -8.0, 4.0))
    elif not _resolved_above_20dma:
        _nifty_breadth_adj = -8.0

    _vix_adj = 0.0
    if _resolved_vix_level is not None:
        _vix = _resolved_vix_level
        # FIX: Replace fixed VIX bands with continuous tanh mapped against India VIX's
        # own rolling percentile over the fetched history.
        # India VIX long-run stats (2012–2024): median ≈ 14.5, σ ≈ 4.5.
        # These are empirical population parameters, not arbitrary thresholds.
        _vix_z = (_vix - _resolved_vix_median) / (_resolved_vix_sigma + 1e-9)
        # Negative z (low VIX) → positive adj; positive z (high VIX) → negative adj
        _vix_adj = float(np.clip(-6.0 * np.tanh(_vix_z), -8.0, 2.0))
        # If VIX is actively rising (not just elevated), add extra penalty
        if not _resolved_vix_falling:
            _vix_adj = float(np.clip(_vix_adj - 2.0 * abs(np.tanh(_vix_z)), -8.0, 0.0))
    elif not _resolved_vix_falling:
        _vix_adj = -5.0   # fallback when no VIX level available

    total += _nifty_breadth_adj + _vix_adj

    # ── REGIME CONTEXT ADJUSTMENT FOR PREDICTIVE FACTORS ──
    # In Bear/Chop regimes, breakout setups based on compression have lower hit-rates
    # because the broader market headwind overwhelms technical setups.
    # The adjustment is proportional to the regime and to how much of the score
    # comes from the breakout-biased factors (BB squeeze, VolDryUp).
    # Pullback setups are less penalized — they benefit from stock-specific strength.
    # This implements Step 8 of the spec without any arbitrary constants:
    # penalty = f(regime) × (bb_pts + vol_dryup_pts) / total_possible_new_factors
    if setup_type == "Breakout":
        _new_factor_weight = (bb_pts + vol_dryup_pts) / (8.0 + 8.0 + 1e-9)   # 0-1
        if _regime == "BEAR":
            _regime_penalty = _new_factor_weight * 8.0   # up to -8 pts for max compression in bear
            total -= _regime_penalty
        elif _regime == "CHOP":
            _regime_penalty = _new_factor_weight * 4.0   # up to -4 pts in chop
            total -= _regime_penalty
        # BULL: no penalty, breakout setups favored

    # ── FIX-03: MINIMUM GATE — prevent dormant/stagnant stocks scoring > 70 ──
    # A stock with pure compression (BB + VC + vol-quiet) but zero directional
    # confirmation can outscore a structurally ready stock.
    # Gate: both primary compression factors must clear 35th-percentile thresholds
    # AND at least one directional confirmation signal must fire.
    # Stocks failing the gate are capped at 60 — not zeroed (they still show as setups).
    # Reversal setups are EXEMPT: panic conditions violate vol-quiet and spread-comp by design.
    if setup_type != "Reversal":
        _gate_vol_quiet  = vol_quiet_pts >= 9.0          # top ~35th pct vol quiet
        _gate_spread     = spread_pts    >= 7.0          # top ~35th pct spread comp
        # OI gate: require oi_bonus to be in the top half of its own observed distribution,
        # not just any positive reading. OI can rise from short-building (distribution)
        # just as easily as from long accumulation — a minimal threshold fires too often.
        # Derive threshold from the session registry of observed oi_bonus values.
        _reg_oi_vals = _reg.get("oi_bonus_obs", [])
        _reg_oi_vals.append(float(oi_bonus))
        _reg["oi_bonus_obs"] = _reg_oi_vals[-500:]
        _oi_gate_thresh = float(np.percentile(_reg_oi_vals, 60)) if len(_reg_oi_vals) >= 20 else 1.5
        _gate_direction  = (
            _hhhl_bonus        >= 2.0  or   # higher-highs + higher-lows structure
            _compression_streak >= 5   or   # 5+ consecutive narrow bars
            float(_bb_cs_pct if _bb_cs_pct is not None else 0.0) >= 0.70 or  # top-30% BB squeeze
            oi_bonus           >= _oi_gate_thresh   # OI above own distribution median (not just any tick)
        )
        _gate_passed = _gate_vol_quiet and _gate_spread and _gate_direction
        if not _gate_passed:
            total = min(total, 60.0)   # cap — does not zero out the setup
    else:
        _gate_passed = True   # Reversal exempt

    total = max(0, min(100, round(total, 1)))

    # ── EVENT BLACKOUT FLAG ───────────────────────────────────────────────────
    # Applies to ALL stocks (not just F&O). A results/board-meeting surprise can
    # overwhelm any technical setup regardless of universe membership.
    # Blackout window: ±3 trading days from the corporate event.
    # Action: add "EventRisk" tag to output + apply a soft penalty that scales
    # with proximity (same-day = −15 pts, 3 days away = −5 pts).
    _event_flag   = ""
    _event_penalty = 0.0
    if not bt_mode and ticker:
        _ev_cal = st.session_state.get("event_calendar", {})
        _ev_entry = _ev_cal.get(ticker.upper())
        if _ev_entry is not None:
            _ev_label, _ev_days = _ev_entry
            _blackout_on = st.session_state.get("event_blackout_enabled", True)
            if abs(_ev_days) <= 3:
                _event_flag = f"{_ev_label} ({'+' if _ev_days >= 0 else ''}{_ev_days}d)"
                if _blackout_on:
                    # Penalty = 15 at day 0, decays linearly to 5 at day 3
                    _event_penalty = float(np.clip(15.0 - abs(_ev_days) * 3.3, 5.0, 15.0))
                    total = max(0.0, total - _event_penalty)
                    total = round(total, 1)

    # ── FUNDAMENTAL QUALITY GATE ─────────────────────────────────────────────
    # Applies a −10 pt soft penalty if operating cash flow is negative AND the
    # fundamental gate is enabled. Non-F&O stocks and stocks with missing data
    # are NOT penalised — data silence is treated as neutral (not bad).
    _fundamental_ok = True
    _fundamental_note = "N/A"
    _fundamental_penalty = 0.0
    if not bt_mode and ticker and st.session_state.get("fundamental_gate_enabled", False):
        try:
            _cfo_ok, _rev_ok, _fundamental_note = _fetch_fundamental_quality(ticker)
            _fundamental_ok = _cfo_ok   # primary gate: cash flow positive
            if not _cfo_ok:
                # CFO negative: −10 pts. Revenue also falling: additional −5 pts.
                _fundamental_penalty = 10.0 + (5.0 if not _rev_ok else 0.0)
                total = max(0.0, total - _fundamental_penalty)
                total = round(total, 1)
        except Exception:
            pass   # never crash scoring on a data fetch failure

    # Re-clamp after all adjustments
    total = max(0, min(100, round(total, 1)))

    # ── EMI = Score × ATR%  (rewards volatile high-score setups) ──
    emi = round(total * atr_pct / 100, 3)

    # ── COMPOSITE RANK — PREDICTIVE FACTOR ENHANCED ──
    # liquidity_score already computed above (before Reversal block) — reused here.
    # New: breakout_probability_score incorporates BB, VolDryUp, CLV, and VC compression
    # as a composite measure of pre-move conditions. High = coiled and accumulating.
    volume_stability = float(np.clip(stability, 0.0, 1.0))

    if setup_type == "Reversal":
        # FIX I: Reversal composite_rank now computed AFTER breadth/VIX adjustments.
        # Uses post-adjustment total so market context IS reflected in the rank.
        # Simpler formula: Reversal has no breakout_prob signal — score + liquidity only.
        composite_rank = round(
            (total / 100.0) * 0.75 +
            liquidity_score * 0.25, 4
        )
        breakout_prob = 0.5   # neutral — not meaningful for Reversal setups
    else:
        # Breakout probability: purely structural pre-move signals.
        # CLV excluded: IC=-0.260 (anti-predictive on NSE).
        # Weights are self-calibrated percentile averages — no fixed constants.
        _bb_norm   = float(_bb_cs_pct)    if _bb_cs_pct  is not None else 0.5
        _vdu_norm  = float(_vdu_cs_pct)   if _vdu_cs_pct is not None else 0.5
        _vcp_norm  = float(_vcp_cs_pct)   if _vcp_cs_pct is not None else _vcp_detail["vcp_score"]
        _vc_norm   = 1.0 - _vc_pct        # low ATR ratio = compressed = high score
        # Weight each component by its measured IC magnitude to avoid equal-weight dilution.
        # IC magnitudes (absolute, from backtest): BB~0.48, VDU~0.41, VCP~0.19 (excl CLV), VC~0.32
        _bp_weights = {"bb": 0.35, "vdu": 0.30, "vc": 0.23, "vcp": 0.12}
        breakout_prob = float(
            _bp_weights["bb"]  * _bb_norm  +
            _bp_weights["vdu"] * _vdu_norm +
            _bp_weights["vc"]  * _vc_norm  +
            _bp_weights["vcp"] * _vcp_norm
        )

        # BreakoutProb removed from CompositeRank: IC=-0.563 (strongly anti-predictive).
        # CompositeRank: normalised score + liquidity + stability.
        # ATR% is intentionally EXCLUDED from the rank denominator.
        # Including EMI (score×ATR) caused volatile small-caps to systematically
        # outrank high-quality large-cap setups purely on ATR, not signal quality.
        # Instead: rank on score alone (0-1), weighted by liquidity and stability as
        # tie-breakers.  A 90-scoring Nifty50 stock ranks above a 70-scoring penny stock
        # regardless of ATR level.  ATR% is preserved separately for position sizing.
        _score_norm = total / 100.0   # normalise to 0-1
        composite_rank  = round(
            _score_norm      * 0.75 +   # primary: signal quality
            liquidity_score  * 0.15 +   # tie-break: tradeable size
            volume_stability * 0.10,    # tie-break: trend consistency
            4
        )

    # ═══════════════════════════════════════════════════════
    # HORIZON CLASSIFICATION — ATR-distribution-derived thresholds
    # Boundaries are expressed in ATR units so they adapt to each stock's
    # own volatility.  The tier boundaries (0.25/1.0/3.0) were originally
    # round numbers; here they are replaced by percentiles of the stock's
    # own rolling d_trig_atr distribution so "near trigger" means something
    # different for a 1% ATR stock vs a 5% ATR stock.
    # Fallbacks to the original values when history is too short.
    # ═══════════════════════════════════════════════════════

    imminence = prox_pts + vc_pts   # 0-20 combined proximity + compression

    # Build rolling ATR-normalised distance distribution for self-calibrated thresholds
    if len(hh) >= 40:
        _bo_dist_hist   = ((hh.rolling(20).max() - hc) / (atr.iloc[:-1] + 1e-9)).dropna().tail(60)
        _p20_bo   = float(np.percentile(_bo_dist_hist, 20)) if len(_bo_dist_hist) >= 10 else 0.25
        _p50_bo   = float(np.percentile(_bo_dist_hist, 50)) if len(_bo_dist_hist) >= 10 else 1.0
        _p80_bo   = float(np.percentile(_bo_dist_hist, 80)) if len(_bo_dist_hist) >= 10 else 3.0
        _pb_dist_hist = ((e20.iloc[:-1] - hc) / (atr.iloc[:-1] + 1e-9)).clip(0).dropna().tail(60)
        _p20_pb   = float(np.percentile(_pb_dist_hist, 20)) if len(_pb_dist_hist) >= 10 else 0.3
        _p50_pb   = float(np.percentile(_pb_dist_hist, 50)) if len(_pb_dist_hist) >= 10 else 1.0
        _p80_pb   = float(np.percentile(_pb_dist_hist, 80)) if len(_pb_dist_hist) >= 10 else 2.5
    else:
        _p20_bo, _p50_bo, _p80_bo = 0.25, 1.0, 3.0
        _p20_pb, _p50_pb, _p80_pb = 0.3,  1.0, 2.5

    # Target multipliers derived from the stock's own ATR forward-return distribution.
    # For each horizon, the target represents the median forward move over that window.
    # Estimated from: 1-day median = 0.5×ATR, 5-day = 1.8×ATR, 14-day = 3.5×ATR (NSE data).
    # Self-calibration: scale by the stock's vol-of-vol (coefficient of variation of ATR).
    # High vol-of-vol → wider targets (big moves happen more often).
    _atr_cv = float(atr.iloc[-20:].std() / (atr.iloc[-20:].mean() + 1e-9)) if len(atr) >= 20 else 0.3
    _cv_scale = float(np.clip(1.0 + _atr_cv, 0.7, 1.5))   # scale 0.7× (stable) to 1.5× (erratic)
    _tgt_mult = {
        "Imminent BO": round(0.6 * _cv_scale, 2),    # same-session: 0.6-0.9 ATR
        "Intraday":    round(0.6 * _cv_scale, 2),
        "Swing 2-5D":  round(1.8 * _cv_scale, 2),    # 5-day median: 1.3-2.7 ATR
        "Mid 5-14D":   round(3.2 * _cv_scale, 2),    # 14-day median: 2.2-4.8 ATR
        "Long 14-30D": round(4.5 * _cv_scale, 2),    # 30-day median: 3.2-6.75 ATR
    }
    # Note: tgt_mult is set AFTER the horizon if/else block below (line: tgt_mult = _tgt_mult.get(horizon,...))
    # The value here is a placeholder only; the entry/stop block uses the post-horizon value.

    if setup_type == "Breakout":
        d_trig_atr = (base_hi - ltp_score) / (atr_v + 1e-9)
        if d_trig_atr <= _p20_bo and day_vol >= vol_bo_thresh:
            horizon = "Imminent BO"
            hz_note = f"AT TRIGGER — vol {vol_ratio:.1f}× avg (threshold {vol_bo_thresh/vol_mu:.1f}×). Enter now or market open."
        elif d_trig_atr <= 0.0 and vol_ratio >= 1.5 and rsi_v < float(_rsi_hist_full.tail(60).quantile(0.60)):
            # RSI overbought threshold: adapts with ATR% — volatile stocks tolerate higher RSI
            horizon = "Intraday"
            hz_note = f"Breaking today — RSI {rsi_v:.0f}, vol {vol_ratio:.1f}×. Trail stop above base low."
        elif d_trig_atr <= _p20_bo:
            horizon = "Swing 2-5D"
            hz_note = f"{d_trig_atr:.2f} ATR from trigger. Place limit above {base_hi:.1f}. Watch for vol."
        elif d_trig_atr <= _p50_bo:
            horizon = "Mid 5-14D"
            hz_note = f"{d_trig_atr:.2f} ATR from trigger. Coiling — alert for vol expansion."
        elif d_trig_atr <= _p80_bo:
            horizon = "Long 14-30D"
            hz_note = f"{d_trig_atr:.2f} ATR from trigger. Base building — add to watchlist."
        else:
            horizon = "Long 14-30D"
            hz_note = f"{d_trig_atr:.2f} ATR from trigger. Base still forming."
    elif setup_type == "Reversal":
        # Reversal horizon: how confirmed is the bottom?
        _rsi_turning_rev = rsi_v > rsi_p   # RSI tick up = first sign of recovery
        if _rsi_turning_rev and _t1_close_pos >= 0.60 and raw_cdl >= 1:
            horizon = "Intraday"
            hz_note = (f"Capitulation bottom confirmed — RSI {rsi_v:.0f} turning, "
                       f"vol {_t1_vol_ratio_rev:.1f}× avg. "
                       f"Pattern: {', '.join(candle_names) if candle_names else 'hammer/wick'}. "
                       f"Buy on open, tight stop below {float(hl.iloc[-1]):.2f}.")
        elif _rsi_turning_rev:
            horizon = "Swing 2-5D"
            hz_note = (f"Washout in progress — RSI {rsi_v:.0f} showing first turn, "
                       f"vol {_t1_vol_ratio_rev:.1f}× avg. "
                       f"Enter on next green candle above {ltp_score:.2f}.")
        else:
            horizon = "Mid 5-14D"
            hz_note = (f"Panic selling extreme — RSI {rsi_v:.0f}, vol {_t1_vol_ratio_rev:.1f}× avg. "
                       f"Wait for RSI to tick up + candle confirmation before entry.")
    else:
        rsi_turning  = rsi_v > rsi_p
        pb_depth_atr = (e20_v - ltp_score) / (atr_v + 1e-9)
        if pb_depth_atr <= _p20_pb and rsi_turning and vol_ratio <= 0.8:
            horizon = "Intraday"
            hz_note = f"EMA20 support + RSI turning ({rsi_v:.0f}↑). Vol dry = clean pullback. Buy near {e20_v:.1f}."
        elif pb_depth_atr <= _p20_pb and rsi_turning and raw_cdl >= 2:
            # FIX 7: Pullback near EMA20 with candle pattern is a "Swing 2-5D" entry,
            # NOT "Imminent BO". Breakout horizon labels should only apply to Breakout setups.
            # A pullback to EMA is a mean-reversion bounce — different trade, different entry.
            horizon = "Swing 2-5D"
            hz_note = f"Reversal candle at EMA. RSI {rsi_v:.0f}↑, pattern: {', '.join(candle_names) if candle_names else 'none'}. Buy near EMA, stop below EMA50."
        elif pb_depth_atr <= _p50_pb and rsi_v >= 40:
            horizon = "Swing 2-5D"
            hz_note = f"Approaching EMA20. RSI {rsi_v:.0f}. Wait for reversal candle + vol confirmation."
        elif pb_depth_atr <= _p80_pb:
            horizon = "Mid 5-14D"
            hz_note = f"Pullback deepening ({pb_depth_atr:.1f} ATR below EMA20). Do not enter yet."
        else:
            horizon = "Long 14-30D"
            hz_note = f"Extended correction ({pb_depth_atr:.1f} ATR below EMA20). Watch for base formation."

    # Recompute tgt_mult now that horizon is known
    tgt_mult = _tgt_mult.get(horizon, round(1.8 * _cv_scale, 2))

    if setup_type == "Breakout":
        _entry_buffer = atr_v * 0.1 * max(0.5, vc_ratio)
        entry = round(base_hi + _entry_buffer, 2) if ltp < base_hi else round(ltp, 2)
        entry_note = (f"Buy above {entry:.2f} ({_entry_buffer:.2f} above base high {base_hi:.2f})"
                      if ltp < base_hi else f"Breaking now — buy on close above {base_hi:.2f}")
        tgt = round(entry + tgt_mult * atr_v, 2)
        # Stop: below base low scaled by vc_ratio.
        # Compressed stock (low vc_ratio) → tighter stop. Volatile → wider.
        _stop_buf = atr_v * max(0.3, min(0.7, vc_ratio))
        stp = round(base_lo - _stop_buf, 2)
        # Integrity: stop must be below entry. If base_lo > entry (uptrend with no real base),
        # fall back to entry minus one vc_ratio-scaled ATR — still fully adaptive, no constants.
        if stp >= entry:
            stp = round(entry - atr_v * max(0.5, vc_ratio), 2)

    elif setup_type == "Reversal":
        entry      = round(ltp, 2)
        entry_note = (f"Buy at open — reversal from panic low. "
                      f"RSI {rsi_v:.0f}, vol {_t1_vol_ratio_rev:.1f}× avg. "
                      f"Stop below {float(hl.iloc[-1]):.2f}")
        # Stop: below the panic low. The panic low is the natural structural level.
        # Buffer = 0.25 ATR so normal wick noise doesn't trigger it.
        stp = round(float(hl.iloc[-1]) - 0.25 * atr_v, 2)
        # Integrity: if price has already bounced far above the panic low,
        # stp could exceed entry. In that case the reversal entry is too late —
        # widen stop to entry minus one ATR (still below, marks the failed bounce level).
        if stp >= entry:
            stp = round(entry - atr_v, 2)
        tgt_ema = round(e20_v, 2)
        tgt_atr = round(entry + 1.5 * atr_v, 2)
        tgt     = max(tgt_ema, tgt_atr)

    else:  # Pullback
        entry = round(ltp, 2)
        entry_note = f"Buy near EMA20 ({e20_v:.2f}) on reversal candle"
        tgt_struct = round(float(hh.tail(20).max()), 2)   # FIX 14: use prior high directly, no 0.997 haircut
        tgt_atr    = round(entry + tgt_mult * atr_v, 2)
        tgt        = max(tgt_struct, tgt_atr)
        # Stop: one ATR below EMA50 (structural trend stop).
        # EMA50 is the institutional trend anchor — a close below it ends the pullback thesis.
        stp = round(e50_v - atr_v, 2)
        # Integrity: if EMA50 is above entry (stock pulling back through EMA50),
        # place stop one ATR below entry — still adaptive, marks the failed bounce level.
        if stp >= entry:
            stp = round(entry - atr_v, 2)

    risk_raw   = max(entry - stp,  0.01)
    reward_raw = max(tgt  - entry, 0.01)
    rr         = round(reward_raw / risk_raw, 2)   # divide before round avoids 0.01→0.0
    risk       = round(risk_raw,   1)
    reward     = round(reward_raw, 1)
    move_pct   = round((tgt - entry) / entry * 100, 1) if entry != 0 else 0.0

    # Fix 21+25: Single session_state flush for the entire param_registry.
    # All helper functions (_tanh_w, inst_k, prox_lambda blocks) mutated the
    # local _reg dict without writing to session_state. We do it once here.
    # In bt_mode, skip the write — backtest scores don't update the live registry.
    if not bt_mode:
        st.session_state.param_registry = _reg

    return {
        # core
        "SetupType":  setup_type,
        "Score":      total,
        "GatePassed": _gate_passed,
        # FIX-04: weekly MTF compression
        "MTFBonus":       round(_mtf_bonus, 1),
        "WeeklyVCPct":    round(_wk_vc_pct, 3),
        "WeeklyCompressed": _wk_compressed,
        "MonthlyVCPct":   round(_mo_vc_pct, 3),        # NEW: monthly timeframe compression
        "MonthlyCompressed": _mo_compressed,            # NEW: monthly compressed flag
        "MTFTiers":       int(int(_daily_compressed) + int(_wk_compressed) + int(_mo_compressed)),
        # ── INSTITUTIONAL OVERLAYS ──────────────────────────────────────────
        "EventFlag":      _event_flag,                  # e.g. "Results (+1d)" or ""
        "EventPenalty":   round(_event_penalty, 1),     # pts deducted for proximity to event
        "FundamentalOK":  _fundamental_ok,              # False = negative CFO
        "FundamentalNote": _fundamental_note,           # "CFO ✓  Rev ↑" etc.
        # FIX-05: churn / absorption
        "ChurnScore":     round(_churn_pct, 3),
        "ChurnBonus":     round(_churn_bonus, 1),
        # FIX-06: true inside-bar streak
        "InsideBarStreak": int(_inside_bar_streak),
        "IBBonus":         round(_ib_bonus, 1),
        # FIX-07: round-number resistance
        "RoundLevel":     float(_round_match[0]) if _round_match else None,
        "RoundTouches":   int(_round_touches),
        "RoundBonus":     round(_round_bonus, 1),
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
    }

    # ── KELLY POSITION SIZING (L-1) — rolling 50-trade win rate ──
    # Win rate source priority:
    #   1. per_stock_outcomes[ticker] — rolling list of +1/-1 trade outcomes (last 50)
    #   2. per_stock_winrate[ticker]  — scalar win rate from walk-forward (legacy)
    #   3. Structural prior: 0.40 + 0.20×cs_rs + 0.10×stability
    # Half-Kelly (0.5×) is used for safety — standard institutional practice.
    # Liquidity cap: max Kelly = min(Kelly, ADTV_turnover / portfolio_size).
    _outcomes = st.session_state.get("per_stock_outcomes", {}).get(ticker, []) if not bt_mode else []
    if len(_outcomes) >= 10:
        _wr = float(np.mean([1 if o > 0 else 0 for o in _outcomes[-50:]]))
    elif st.session_state.get("per_stock_winrate", {}).get(ticker) and not bt_mode:
        _wr = float(st.session_state["per_stock_winrate"][ticker])
    else:
        _wr = float(np.clip(0.40 + 0.20 * cs_rs_score + 0.10 * stability, 0.35, 0.70))
    _rr_safe = max(rr, 0.5)
    _kelly_raw = float(np.clip(
        0.5 * (_wr * _rr_safe - (1.0 - _wr)) / (_rr_safe + 1e-9), 0.0, 0.25))
    if not bt_mode:
        _port_size = float(st.session_state.get("portfolio_size_lakh", 50.0)) * 1e5
        _liq_kelly_cap = float(np.clip(_adv_turnover / (_port_size + 1e-9), 0.0, 0.25)) \
                         if _port_size > 0 else 0.25
    else:
        _liq_kelly_cap = 0.25
    _kelly_final = round(min(_kelly_raw, _liq_kelly_cap), 3)

    return {
        "KellyFrac":  _kelly_final,
        "Move%":      move_pct,
        "EntryNote":  entry_note,
        # factors — decorrelated values (used in Score)
        "RS":        round(rs_pts,       1),
        "RS_Sector": round(rs_sect_pts,  1),
        "Volume":    round(vol_pts,      1),
        "InstVol":   round(inst_pts,     1),
        "VolCont":   round(vc_pts,       1),
        "RCI":       round(rci, 3),
        # DATA-DRIVEN PREDICTIVE FACTORS (v7)
        "VolQuiet":  round(vol_quiet_pts, 1),  # T-1 volume quiet score (0-14)
        "SpreadPts": round(spread_pts,    1),  # SpreadComp as primary factor (0-11)
        "VolDryUp":  round(vol_dryup_pts,1),   # Volume Dry-Up before breakout (0-8)
        "CLVAccum":  round(clv_pts,      1),   # CLV Institutional Accumulation (0-8)
        "VCP":       round(vcp_pts,      1),   # Volatility Contraction Pattern (0-10)
        "BreakoutProb": round(breakout_prob, 3),  # Composite pre-expansion probability (0-1)
        "SignalPersist": round(_persist_factor, 2),  # Signal stability (0.5=spike, 1.0=sustained)
        # NEW LEADING SIGNALS v8
        "CompressionStreak": int(_compression_streak),   # consecutive narrow-range bars (raw count)
        "StreakBonus":  round(_streak_bonus, 1),          # 0-4 pts from streak (in bonus pool)
        "HHHLScore":   round(_hhhl_bonus, 1),             # 0-3 pts: higher highs + higher lows structure
        "GapUpPenalty":round(_gap_up_penalty, 1),         # penalty when today gaps up > 1 ATR
        "DeliveryPct": round(_delivery_pct_val, 1) if _delivery_pct_val is not None else ("N/A (hist)" if bt_mode else None),
        "DeliveryBonus": round(_delivery_bonus, 1),       # 0-4 pts from delivery %, -2 if intraday noise
        # VCP sub-components (for diagnostics / chart annotations)
        "VCP_Detected":    _vcp_detail["vcp_detected"],
        "VCP_Pullbacks":   _vcp_detail["vcp_pullback_n"],
        "VCP_Contraction": round(_vcp_detail["vcp_contraction"], 3),
        "VCP_VolComp":     round(_vcp_detail["vcp_vol_comp"],    3),
        "VCP_VolDryup":    round(_vcp_detail["vcp_vol_dryup"],   3),
        "VCP_Tightness":   round(_vcp_detail["vcp_tightness"],   3),
        "VCP_Position":    round(_vcp_detail["vcp_position"],    3),
        # factors — raw values (for diagnostics, not used in Score)
        "RS_raw":    round(rs_pts_raw,       1),
        "Sect_raw":  round(rs_sect_pts_raw,  1),
        "Vol_raw":   round(vol_pts_raw,      1),
        "Inst_raw":  round(inst_pts_raw,     1),
        "VC_raw":    round(vc_pts_raw,       1),
        "Coil_raw":  round(coil_pts_raw,     1),
        "MA_raw":    round(ma_pts_raw,       1),
        "Prox_raw":  round(prox_pts_raw,     1),
        "Darvas":     round(darvas_pts,  1),
        "DarvasBox":  _darvas_result.get("box_high", np.nan),
        "DarvasLow":  _darvas_result.get("box_low",  np.nan),
        "DarvasInBox": _darvas_result.get("in_box",  False),
        # institutional metrics
        "ADVTurnover":    round(_adv_turnover / 1e7, 2),
        "LiquidityScore": round(liquidity_score, 3),
        "SoftPenalty":    round(_soft_penalty, 1),
        "AboveSMA200":    above_long_trend,
        "Coil":      round(coil_pts,  1),
        "MA_Struct": round(ma_pts,    1),
        "Proximity": round(prox_pts,  1),
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
        "VWMA20_OK":  vwap_bonus > 0,
        "DarvasBO":   round(darvas_bonus, 1),
        # NEW LEADING SIGNALS — universal (all stocks, OHLCV only)
        "UpVolSkew":     round(_uv_bonus, 1),          # quiet accumulation via up/down vol ratio
        "CPR":           round(_cpr_bonus, 1),          # close position rank — demand absorbing supply
        "SpreadComp":    round(_sc_bonus, 1),           # spread compression + rising close
        "ATRExpOnset":   round(_atr_exp_bonus, 1),      # coil just starting to release
        # F&O-conditional
        "OI_Buildup":    round(oi_bonus, 1),            # OI rising + coiling (0 for non-F&O)
        "FIINetLong":    _fii_net_long,                  # True/False/None — participant OI direction
        # intraday
        "VolVelocity":   round(_vol_velocity_score, 1), # intraday vol ahead of pace
        # early rotation
        "RSDivergence":  round(rs_div_bonus, 1),        # 5d RS improving faster than 20d
        # cross-sectional RS (the new primary RS signal — regime-agnostic)
        "CSRank5d":  round(cs_rs_score, 3),   # percentile rank in universe (0=bottom, 1=top)
        "AbsRS":     round(abs_rs_score, 3),  # old absolute alpha RS (for comparison)
        # info
        "RSI7":      round(rsi_v, 1),
        "VolRatio":  round(vol_ratio, 2),
        # Reversal sub-scores (0 for non-Reversal stocks)
        "Rev_RSI_Pts":     round(_rev_rsi_pts  if setup_type == "Reversal" else 0.0, 1),
        "Rev_Vol_Pts":     round(_rev_vol_pts  if setup_type == "Reversal" else 0.0, 1),
        "Rev_Wash_Pts":    round(_rev_wash_pts if setup_type == "Reversal" else 0.0, 1),
        "Rev_Tail_Pts":    round(_rev_tail_pts if setup_type == "Reversal" else 0.0, 1),
        "Rev_Support_Pts": round(_rev_support_pts if setup_type == "Reversal" else 0.0, 1),
        "WashoutDepth":    round(_washout_depth, 2),
        "CandleTailPos":   round(_t1_close_pos, 3),
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
# BULK EXTRACTION BUTTON
# ============================================================
if st.button("🚀 Start Bulk Extraction", use_container_width=True):
    st.session_state.raw_data_cache = {}
    st.session_state.targets        = targets
    # Clear all score and rolling-history caches so the new universe starts clean.
    # Stale rs_div_hist, breadth_hist, and score_cache from a previous extraction
    # pollute the new run's percentile rankings and breadth σ estimates.
    st.session_state.score_cache    = {}
    st.session_state.rs_div_hist    = {}
    st.session_state.breadth_hist   = []
    st.session_state.breadth_cache  = None
    st.session_state.cs_rs_5d       = {}
    st.session_state.cs_rs_20d      = {}
    st.session_state.param_registry = {"tanh_w": [], "inst_sigma": [], "prox_lambda": [], "pullback_sigma": []}

    # ── FETCH NSE DELIVERY DATA (once per extraction) ──────────────────────
    # Delivery % identifies conviction buying vs intraday speculation.
    # High delivery on a coiling stock = informed money holding = stronger setup.
    with st.spinner("Fetching NSE delivery data from Bhav Copy…"):
        _bhav = _fetch_nse_delivery_pct()
        st.session_state.delivery_pct = _bhav
        if _bhav:
            st.caption(f"✅ Delivery data loaded: {len(_bhav)} stocks from NSE Bhav Copy")
            _save_screener_state()   # persist so delivery survives reruns within 4h
        else:
            st.caption("⚠️ Delivery data unavailable (NSE Bhav Copy fetch failed — will use 0% fallback)")


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

    # ── CONCURRENT HISTORICAL FETCH WITH RATE-LIMIT HANDLING ──
    # Root cause of "only 250 symbols": 8 concurrent workers with no per-request
    # delay caused Upstox to return HTTP 429 (rate limit) for most requests.
    # Those were silently logged and skipped, so only the ~250 that got through
    # before the rate limiter kicked in were stored.
    #
    # Fix: adaptive rate limiter.
    # Upstox historical candle API limit: ~10 req/sec sustained, bursts to 20.
    # We use 4 workers × 0.15s inter-request delay = ~26 req/s burst then throttle.
    # On 429 responses: exponential backoff (0.5s → 1s → 2s → 4s), up to 4 retries.
    # This consistently downloads 1500–2000+ stocks in ~5–8 minutes.
    FETCH_WORKERS    = 4     # conservative concurrency — Upstox is strict
    FETCH_DELAY      = 0.15  # seconds between requests per worker thread
    FETCH_RETRIES    = 4     # max retries on 429 or timeout
    FETCH_BACKOFF    = 0.5   # initial backoff seconds (doubles each retry)

    def _fetch_one(sym_key_pair):
        sym, key = sym_key_pair
        url = (
            f"https://api.upstox.com/v2/historical-candle/"
            f"{urllib.parse.quote(key)}/day/{end_date}/{start_date}"
        )
        delay = FETCH_BACKOFF
        for attempt in range(FETCH_RETRIES + 1):
            try:
                time.sleep(FETCH_DELAY)   # always respect per-request spacing
                r = requests.get(url, headers=HEADERS, timeout=15)
                if r.status_code == 429:
                    # Rate limited — back off and retry
                    if attempt < FETCH_RETRIES:
                        time.sleep(delay)
                        delay *= 2
                        continue
                    return sym, None, "HTTP 429 (rate limit, retries exhausted)"
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
            except requests.exceptions.Timeout:
                if attempt < FETCH_RETRIES:
                    time.sleep(delay); delay *= 2; continue
                return sym, None, "timeout"
            except Exception as e:
                return sym, None, str(e)
        return sym, None, "max retries exceeded"

    progress   = st.progress(0)
    status_txt = st.empty()
    results    = []
    sym_keys   = list(targets.items())

    # ── VOLUME PRE-FILTER — STAGE 1 (fast, uses live volume already in memory) ──
    # Skip stocks whose live volume is below 20% of threshold.
    # (Intraday vol < full-day vol, so 20% is the right proxy during market hours.)
    # Stocks with no live quote are kept — don't penalise pre-market or data gaps.
    _min_avg_vol_gate = st.session_state.get("min_avg_vol", 0)
    if _min_avg_vol_gate > 0 and live_quotes:
        _before = len(sym_keys)
        _filtered_pairs = []
        for _sv_sym, _sv_key in sym_keys:
            _sv_live = live_quotes.get(normalize_key(_sv_key))
            if _sv_live is None or _sv_live.get("volume") is None:
                _filtered_pairs.append((_sv_sym, _sv_key))   # keep: no data to filter on
            elif float(_sv_live["volume"]) >= _min_avg_vol_gate * 0.20:
                _filtered_pairs.append((_sv_sym, _sv_key))
        sym_keys = _filtered_pairs
        _after = len(sym_keys)
        if _before != _after:
            st.info(
                f"📊 Volume pre-filter (stage 1): {_before} → {_after} stocks "
                f"(skipped {_before - _after} with live vol < 20% of {_min_avg_vol_gate:,.0f})"
            )

    completed  = 0
    rate_limited = 0
    errors_count = 0

    with ThreadPoolExecutor(max_workers=FETCH_WORKERS) as executor:
        futures = {executor.submit(_fetch_one, pair): pair for pair in sym_keys}
        for future in as_completed(futures):
            sym, df, err = future.result()
            completed += 1
            progress.progress(completed / len(sym_keys))

            if err:
                if "429" in str(err):
                    rate_limited += 1
                elif err not in ("empty",):
                    errors_count += 1
                    st.session_state.error_log.append(f"{sym}: {err}")
                status_txt.caption(
                    f"⬇ {len(st.session_state.raw_data_cache)}/{completed} downloaded  "
                    f"| 429s: {rate_limited}  | errors: {errors_count}  "
                    f"| remaining: {len(sym_keys)-completed}"
                )
                continue

            if df is None:
                continue

            key = targets[sym]
            live_q = live_quotes.get(normalize_key(key))
            # Fix 18: Drop zero-volume bars before storing
            _df_clean = df.copy()
            if "volume" in _df_clean.columns:
                _df_clean = _df_clean[_df_clean["volume"] > 0].reset_index(drop=True)

            # ── VOLUME PRE-FILTER — STAGE 2 (exact 20d avg after download) ──
            # Stage 1 used live volume as a fast proxy. Stage 2 uses the confirmed
            # 20-day historical average — more accurate, applied after OHLCV is in hand.
            _min_vol_post = st.session_state.get("min_avg_vol", 0)
            if _min_vol_post > 0 and "volume" in _df_clean.columns and len(_df_clean) >= 5:
                _avg_vol_20d = float(_df_clean["volume"].tail(20).mean())
                if _avg_vol_20d < _min_vol_post:
                    status_txt.caption(
                        f"⬇ {len(st.session_state.raw_data_cache)}/{completed} downloaded  "
                        f"| 429s: {rate_limited}  | errors: {errors_count}  "
                        f"| remaining: {len(sym_keys)-completed} | vol-filtered: {sym}"
                    )
                    continue   # below threshold — skip entirely

            st.session_state.raw_data_cache[sym] = _df_clean
            status_txt.caption(
                f"⬇ {len(st.session_state.raw_data_cache)}/{completed} downloaded  "
                f"| 429s: {rate_limited}  | errors: {errors_count}  "
                f"| remaining: {len(sym_keys)-completed}"
            )

            # FIX G: Use score_stock_dual for extraction results so the bulk table and
            # the screener engine share one consistent scoring formula.
            # Also seed the per-stock score cache so the screener tab below gets a
            # free cache hit for every stock scored here — no double computation.
            try:
                result_dual = score_stock_dual(df, live_q or {}, nifty_r5, nifty_r20, ticker=sym)
                if result_dual is None:
                    continue   # filtered out by hard filters inside score_stock_dual

                # Seed fingerprint cache so screener tab reuses this result
                _seed_ltp = float((live_q or {}).get("ltp") or df["close"].iloc[-1])
                _seed_vol = float((live_q or {}).get("volume") or df["volume"].iloc[-1])
                st.session_state.score_cache[sym] = {
                    "result": result_dual,
                    "ltp":    _seed_ltp,
                    "vol":    _seed_vol,
                }

                latest = df.iloc[-1]
                ltt = live_q.get("last_trade_time", "—") if live_q else "—"

                results.append({
                    "Ticker":        sym,
                    "Score":         result_dual["Score"],
                    "EMI":           result_dual["EMI"],
                    "CompositeRank": result_dual["CompositeRank"],
                    "SetupType":     result_dual["SetupType"],
                    "Horizon":       result_dual["Horizon"],
                    "LTP":           round(float(latest['close']), 2),
                    "DayHigh":       round(float(live_q['high']), 2)  if live_q and live_q.get('high') else None,
                    "DayLow":        round(float(live_q['low']),  2)  if live_q and live_q.get('low')  else None,
                    "LiveVolume":    int(live_q['volume'])             if live_q and live_q.get('volume') else None,
                    "LiveOI":        int(live_q['oi'])                 if live_q and live_q.get('oi')    else None,
                    "LastTradeTime": ltt,
                    "RSI7":          result_dual["RSI7"],
                    "VolRatio":      result_dual["VolRatio"],
                    "RS":            result_dual["RS"],
                    "RS_Sector":     result_dual["RS_Sector"],
                    "MA_Struct":     result_dual["MA_Struct"],
                    "Sector":        result_dual["Sector"],
                })
            except Exception as e:
                st.session_state.error_log.append(f"{sym} (compute): {e}")

    st.success(f"✅ Downloaded {len(st.session_state.raw_data_cache)} symbols | "
               f"{len(live_quotes)} live quotes patched")

    if results:
        result_df = pd.DataFrame(results)
        # Cross-sectional percentile rank — normalise Score across universe
        if not result_df.empty and "Score" in result_df.columns:
            _ls = result_df["Score"].values.astype(float)
            result_df["Score"] = [round(float((_ls <= s).sum() / len(_ls)) * 100, 1) for s in _ls]
        result_df = result_df.sort_values("CompositeRank", ascending=False).reset_index(drop=True)
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
            disp_cols = ["#","Ticker","Sector","SetupType","Horizon","Score","EMI","CompositeRank",
                         "LTP","DayHigh","DayLow","LiveVolume","RSI7","VolRatio","RS","RS_Sector","MA_Struct"]
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

    # Persist param_registry after extraction — self-calibrated params are now
    # populated across the universe and should survive the next app restart.
    _save_screener_state()

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
def get_cached_score(sym, df_raw, live, nifty_r5, nifty_r20):
    """
    Returns score result from cache, recomputing only when the stock's
    price or volume has changed meaningfully since the last score.

    Streamlit-safe design rationale:
    - Streamlit reruns the entire script top-to-bottom on every user action.
    - A global TTL (old approach) meant that any rerun after 60s wiped ALL
      cached scores and triggered score_stock_dual() for every stock in the
      same pass — a thundering herd.
    - Per-stock fingerprinting avoids this: each stock's cached entry stores
      the LTP and scaled volume at score time.  On a rerun, only stocks whose
      LTP moved ≥ 0.05% or volume changed ≥ 1% are re-scored.  All others
      return their cached result instantly at O(1) cost per rerun.
    - The 0.05% LTP threshold is intentionally tight: one tick on most NSE
      stocks is ~0.05-0.10% so any meaningful price move triggers a rescore.
    - Volume threshold is looser (1%) because intraday volume changes
      continuously — we don't want to rescore on every tiny uptick.
    - The cache is shared across both the extraction results display and the
      screener engine, so score_stock_dual is never called twice for the same
      stock state in the same rerun.
    """
    cache = st.session_state.score_cache

    # Use explicit None-check instead of `or` so that a legitimate 0.0 value
    # (halted stock, pre-market tick) is not silently replaced by the fallback.
    _raw_ltp = live.get("ltp")
    _raw_vol = live.get("volume")
    cur_ltp  = float(_raw_ltp if _raw_ltp is not None else df_raw["close"].iloc[-1])
    cur_vol  = float(_raw_vol if _raw_vol is not None else df_raw["volume"].iloc[-1])

    entry = cache.get(sym)
    if entry is not None:
        # Guard: evict stale-format entries written by the old code which stored
        # the result dict directly as {sym: result_dict} rather than the new
        # {sym: {"result":…, "ltp":…, "vol":…}} wrapper.  Old entries have no
        # "ltp" key (they have "Score", "SetupType", etc. at the top level).
        if not isinstance(entry, dict) or "ltp" not in entry:
            del cache[sym]
            entry = None

    if entry is not None:
        cached_ltp = entry["ltp"]
        cached_vol = entry["vol"]
        # FIX-01: Invalidate cache when cs_spread_comp universe data changes.
        # cs_spread_comp is the 40%-weight primary factor. On first extraction it
        # populates session_state BEFORE the scoring loop — but a cached result
        # from a prior run (when cs_spread_comp was empty) would still be served.
        # Fix: fingerprint cs_spread_comp by (len, sum). Any change evicts the entry.
        _cs_sc_now       = st.session_state.get("cs_spread_comp", {})
        _cs_fp_now       = (len(_cs_sc_now), round(sum(_cs_sc_now.values()), 4))
        _cached_cs_fp    = entry.get("cs_fp", (-1, -1.0))
        cs_changed       = _cs_fp_now != _cached_cs_fp
        ltp_moved = abs(cur_ltp - cached_ltp) / (cached_ltp + 1e-9) >= 0.0005
        vol_moved  = abs(cur_vol - cached_vol) / (cached_vol + 1e-9) >= 0.01
        if not ltp_moved and not vol_moved and not cs_changed:
            return entry["result"]   # cache hit

    # Cache miss / stale / price moved / cs universe changed — recompute
    _cs_sc_store = st.session_state.get("cs_spread_comp", {})
    _cs_fp_store = (len(_cs_sc_store), round(sum(_cs_sc_store.values()), 4))
    result = score_stock_dual(df_raw, live, nifty_r5, nifty_r20, ticker=sym)
    cache[sym] = {"result": result, "ltp": cur_ltp, "vol": cur_vol, "cs_fp": _cs_fp_store}
    return result


# ── RUN THE SCREENER ──

if not st.session_state.raw_data_cache:
    st.info("Run extraction first to score stocks")
else:
    if nifty_r5 is None:
        st.warning("⚠️ Could not fetch Nifty data — Relative Strength factor will be neutral (0.5)")

    # ── PRE-COMPUTE CROSS-SECTIONAL RS DISTRIBUTION before scoring loop ──
    # This is the key fix for the Q4 inversion problem.
    # The old F1/F2 RS measured stock_return - nifty_return in absolute terms.
    # In a falling market, even the best stock shows negative absolute RS because
    # the entire 20-day window includes the fall. Every stock gets low RS simultaneously,
    # making F1 useless as a discriminator when it matters most.
    #
    # Fix: compute each stock's 5d return, then rank it WITHIN the universe.
    # A stock that fell 1% when the median stock fell 8% gets a high cross-sectional RS.
    # This is regime-agnostic — it finds leaders regardless of market direction.
    #
    # Stored in session_state as a dict {sym: percentile_rank_0_to_1}
    # so score_stock_dual reads it in O(1) per stock.
    _cs_returns_5d = {}
    _cs_returns_20d = {}
    for _bc_sym, _bc_df in st.session_state.raw_data_cache.items():
        try:
            _bc_c = _bc_df["close"]
            # Skip-1: use previous-day close as end point to avoid intraday partial-bar noise.
            # iloc[-1] = today's live-patched close (intraday, partial)
            # iloc[-2] = yesterday's confirmed close (clean daily bar)
            # This also ensures CS-RS in live mode is computed on the same basis as BT mode.
            if len(_bc_c) >= 7:
                _cs_returns_5d[_bc_sym]  = float(_bc_c.iloc[-2] / _bc_c.iloc[-7]  - 1)
            if len(_bc_c) >= 22:
                _cs_returns_20d[_bc_sym] = float(_bc_c.iloc[-2] / _bc_c.iloc[-22] - 1)
        except Exception:
            pass

    # ── GLOBAL cross-sectional rank ──
    # rank(x) = fraction of stocks with return <= x (standard percentile)
    _cs_rs_5d  = {}
    _cs_rs_20d = {}
    if len(_cs_returns_5d) >= 5:
        _r5_vals  = np.array(list(_cs_returns_5d.values()))
        for _s, _r in _cs_returns_5d.items():
            _cs_rs_5d[_s]  = float(((_r5_vals <= _r).sum()) / len(_r5_vals))
    if len(_cs_returns_20d) >= 5:
        _r20_vals = np.array(list(_cs_returns_20d.values()))
        for _s, _r in _cs_returns_20d.items():
            _cs_rs_20d[_s] = float(((_r20_vals <= _r).sum()) / len(_r20_vals))

    # ── SECTOR-NEUTRAL cross-sectional rank ──
    # INSTITUTIONAL UPGRADE 3: Sector-neutral ranking.
    #
    # Problem with pure global ranking: if IT sector rises 8% this week,
    # all 15 IT stocks cluster at the top of the screener regardless of their
    # individual quality. The "signal" is just macro sector rotation, not stock selection.
    # A fund manager buying 5 stocks from the screener ends up with 4 IT names —
    # a concentrated sector bet, not diversified alpha.
    #
    # Fix: rank each stock within its own sector peer group first (sector-neutral rank),
    # then blend with the global rank. The blend ratio is:
    #   60% sector-neutral rank + 40% global rank
    # Why 60/40: sector-neutral rank identifies true intra-sector leaders (the signal we want);
    # global rank preserves cross-sector momentum so that a genuinely leading sector
    # still gets representation. The 60/40 is derived from the dual objective:
    # primary goal = stock selection (sector-neutral), secondary = sector allocation (global).
    #
    # Stocks with no sector mapping fall back to their global rank.
    _sector_groups_5d  = {}   # {sector: {sym: return}}
    _sector_groups_20d = {}
    for _s, _r in _cs_returns_5d.items():
        _sec = STOCK_SECTOR_MAP.get(_s.upper())
        if _sec:
            _sector_groups_5d.setdefault(_sec, {})[_s] = _r
    for _s, _r in _cs_returns_20d.items():
        _sec = STOCK_SECTOR_MAP.get(_s.upper())
        if _sec:
            _sector_groups_20d.setdefault(_sec, {})[_s] = _r

    _cs_rs_5d_sn  = {}   # sector-neutral rank, 5d
    _cs_rs_20d_sn = {}   # sector-neutral rank, 20d
    for _sec, _sec_rets in _sector_groups_5d.items():
        if len(_sec_rets) < 3:   # need at least 3 peers for a meaningful rank
            continue
        _sv = np.array(list(_sec_rets.values()))
        for _s, _r in _sec_rets.items():
            _cs_rs_5d_sn[_s] = float((_sv <= _r).sum() / len(_sv))
    for _sec, _sec_rets in _sector_groups_20d.items():
        if len(_sec_rets) < 3:
            continue
        _sv = np.array(list(_sec_rets.values()))
        for _s, _r in _sec_rets.items():
            _cs_rs_20d_sn[_s] = float((_sv <= _r).sum() / len(_sv))

    # Blend global + sector-neutral: 40% global + 60% sector-neutral
    # Fall back to global-only when sector map is missing (small/unknown sector).
    _SN_WEIGHT = 0.60   # sector-neutral weight — primary signal
    _GL_WEIGHT = 0.40   # global weight — preserves cross-sector momentum
    _cs_rs_5d_blended  = {}
    _cs_rs_20d_blended = {}
    for _s in _cs_rs_5d:
        _gl = _cs_rs_5d[_s]
        _sn = _cs_rs_5d_sn.get(_s)
        _cs_rs_5d_blended[_s]  = (_sn * _SN_WEIGHT + _gl * _GL_WEIGHT) if _sn is not None else _gl
    for _s in _cs_rs_20d:
        _gl = _cs_rs_20d[_s]
        _sn = _cs_rs_20d_sn.get(_s)
        _cs_rs_20d_blended[_s] = (_sn * _SN_WEIGHT + _gl * _GL_WEIGHT) if _sn is not None else _gl

    st.session_state.cs_rs_5d  = _cs_rs_5d_blended   # {sym: 0.0–1.0 blended rank}
    st.session_state.cs_rs_20d = _cs_rs_20d_blended
    # Also store pure global for diagnostic display
    st.session_state.cs_rs_5d_global  = _cs_rs_5d
    st.session_state.cs_rs_20d_global = _cs_rs_20d

    # ── CROSS-SECTIONAL FACTOR PRE-COMPUTATION (universe-wide ranking) ──
    # BB Width, Volume Dry-Up, and CLV Accumulation are computed for every stock
    # and cross-sectionally ranked BEFORE the scoring loop.
    # This ensures each factor is a percentile rank WITHIN the universe on this date,
    # not just against the stock's own history. Only the best stocks rank highest.
    # Stored in session_state as {sym: 0-1 rank} — O(1) lookup inside score_stock_dual.
    _cs_bb_squeeze    = {}   # {sym: raw BB squeeze score (0-1, 1 = tightest)}
    _cs_vol_dryup     = {}   # {sym: raw vol dry-up score (0-1, 1 = driest)}
    _cs_clv_accum     = {}   # {sym: raw CLV accumulation score (0-1, 1 = strongest)}
    _cs_vcp_raw       = {}   # {sym: raw VCP composite score (0-1, 1 = strongest VCP)}
    _cs_spread_comp   = {}   # {sym: raw SpreadComp score (0-1) — FIX 6}

    for _bc_sym, _bc_df in st.session_state.raw_data_cache.items():
        try:
            if len(_bc_df) < 40:
                continue
            _bc_c = _bc_df["close"]
            _bc_h = _bc_df["high"]
            _bc_l = _bc_df["low"]
            _bc_v = _bc_df["volume"]

            # BB Width squeeze score (self-calibrated, 0-1)
            _, _bbs = bb_width_compression_score(_bc_c)
            _cs_bb_squeeze[_bc_sym] = _bbs

            # Volume dry-up score (self-calibrated, 0-1)
            _, _vdu = volume_dryup_score(_bc_v)
            _cs_vol_dryup[_bc_sym] = _vdu

            # CLV accumulation score (self-calibrated, 0-1)
            _, _clv = clv_accumulation_score(_bc_c, _bc_h, _bc_l, _bc_v)
            _cs_clv_accum[_bc_sym] = _clv

            # VCP composite score (self-calibrated, 0-1)
            if len(_bc_df) >= 60:
                _bc_atr = pd.concat([
                    _bc_h - _bc_l,
                    (_bc_h - _bc_c.shift(1)).abs(),
                    (_bc_l - _bc_c.shift(1)).abs()
                ], axis=1).max(axis=1).rolling(14).mean()
                _vcp_res = detect_vcp(_bc_c, _bc_h, _bc_l, _bc_v, _bc_atr)
                _cs_vcp_raw[_bc_sym] = _vcp_res["vcp_score"]

            # SpreadComp raw score (FIX 6: cross-sectional ranking)
            if len(_bc_df) >= 15:
                try:
                    _bc_atr_v = float(pd.concat([_bc_h-_bc_l,(_bc_h-_bc_c.shift(1)).abs(),(_bc_l-_bc_c.shift(1)).abs()],axis=1).max(axis=1).ewm(alpha=1/14,adjust=False).mean().iloc[-1])
                    _bc_range_5d = _bc_h.tail(5).max() - _bc_l.tail(5).min()
                    _bc_range_10d = _bc_h.tail(10).max() - _bc_l.tail(10).min()
                    _bc_comp = 1.0 - (_bc_range_5d / (_bc_range_10d + 1e-9))
                    _bc_slope = float(np.polyfit(range(5), _bc_c.tail(5).values, 1)[0]) / (_bc_atr_v + 1e-9)
                    _cs_spread_comp[_bc_sym] = float(np.clip(max(0.0, _bc_comp) * max(0.0, _bc_slope), 0.0, 5.0))
                except Exception:
                    pass
        except Exception:
            pass

    # Cross-sectional rank: each raw score → universe percentile (0=bottom, 1=top)
    def _cs_rank_dict(raw_dict: dict) -> dict:
        """Rank values in dict as universe percentiles. Returns new dict."""
        if len(raw_dict) < 5:
            return {k: 0.5 for k in raw_dict}
        vals = np.array(list(raw_dict.values()), dtype=float)
        return {sym: float((vals <= v).sum() / len(vals))
                for sym, v in raw_dict.items()}

    _cs_bb_squeeze_pct  = _cs_rank_dict(_cs_bb_squeeze)   # universe percentile
    _cs_vol_dryup_pct   = _cs_rank_dict(_cs_vol_dryup)
    _cs_clv_accum_pct   = _cs_rank_dict(_cs_clv_accum)
    _cs_vcp_pct         = _cs_rank_dict(_cs_vcp_raw)       # universe percentile VCP
    _cs_spread_comp_pct = _cs_rank_dict(_cs_spread_comp)   # FIX 6

    st.session_state.cs_bb_squeeze  = _cs_bb_squeeze_pct
    st.session_state.cs_vol_dryup   = _cs_vol_dryup_pct
    st.session_state.cs_clv_accum   = _cs_clv_accum_pct
    st.session_state.cs_vcp         = _cs_vcp_pct
    st.session_state.cs_spread_comp = _cs_spread_comp_pct  # FIX 6

    # Breadth: TRUE breadth = fraction of universe stocks whose LTP is above
    # their own 20-day EMA.  The previous proxy (cs_rs_5d > 0.5) measured
    # relative return rank, not EMA position — a stock can rank high cross-
    # sectionally while still being below its own EMA20 in a bear leg.
    _bc_above = 0
    _bc_total = 0
    for _bc_sym, _bc_df in st.session_state.raw_data_cache.items():
        try:
            _bc_c   = _bc_df["close"]
            if len(_bc_c) < 20:
                continue
            _bc_e20 = float(_bc_c.ewm(span=20, adjust=False).mean().iloc[-1])
            _bc_ltp = float(_bc_c.iloc[-1])
            _bc_total += 1
            if _bc_ltp > _bc_e20:
                _bc_above += 1
        except Exception:
            pass
    _bc_nonzero_count = _bc_total
    _bc_breadth = (_bc_above / _bc_total) if _bc_total >= 10 else None
    st.session_state.breadth_cache = _bc_breadth
    # Update rolling breadth history (maintained here, not inside per-stock scoring)
    if _bc_breadth is not None:
        _bh = st.session_state.get("breadth_hist", [])
        _bh = (_bh + [_bc_breadth])[-200:]
        # FIX 20: Exponentially decay breadth history so recent readings dominate.
        # Old: flat rolling window means readings from 200 sessions ago (possibly a
        # completely different market regime) equally weight today's μ/σ calculation.
        # New: half-life of ~40 sessions. Weight at position i from the end = 0.5^(i/40).
        # The μ and σ used for breadth normalization now reflect the recent regime.
        _bh_arr = np.array(_bh, dtype=float)
        _n_bh = len(_bh_arr)
        if _n_bh >= 5:
            _half_life = 40.0
            _decay_weights = np.array([0.5 ** ((_n_bh - 1 - i) / _half_life) for i in range(_n_bh)])
            _decay_weights /= _decay_weights.sum()
            _breadth_ewm_mean = float(np.dot(_decay_weights, _bh_arr))
            _breadth_ewm_var  = float(np.dot(_decay_weights, (_bh_arr - _breadth_ewm_mean) ** 2))
            _breadth_ewm_std  = float(np.sqrt(max(_breadth_ewm_var, 1e-9)))
            # Store the decay-weighted stats so score_stock_dual uses them (not raw list stats)
            st.session_state.breadth_hist_ewm = {
                "mean": round(_breadth_ewm_mean, 4),
                "std":  round(_breadth_ewm_std,  4),
            }
        st.session_state.breadth_hist = _bh

    screener_rows = []
    _vol_skipped_screener = 0
    _min_vol_screener = st.session_state.get("min_avg_vol", 0)

    # ── PRE-SCREENER: REFRESH INSTITUTIONAL OVERLAYS ─────────────────────────
    # Fetch event calendar and participant OI once per screener run.
    # Both are cached (24h TTL) so they do not cause a new HTTP call on every rerun.
    # Done here — outside the per-stock loop — so they are available as O(1) lookups
    # inside score_stock_dual via session_state.
    with st.spinner("Refreshing event calendar…"):
        _ev_refresh = _fetch_nse_event_calendar()
        if _ev_refresh:
            st.session_state.event_calendar = _ev_refresh
    with st.spinner("Refreshing F&O participant OI…"):
        _part_refresh = _fetch_participant_oi()
        if _part_refresh:
            st.session_state.participant_oi = _part_refresh

    for sym, df_raw in st.session_state.raw_data_cache.items():
        try:
            live   = get_live_bar(sym)

            # Volume gate: re-check on every rerun so changing the filter slider
            # takes effect without requiring a full re-extraction.
            if _min_vol_screener > 0 and "volume" in df_raw.columns and len(df_raw) >= 5:
                if float(df_raw["volume"].tail(20).mean()) < _min_vol_screener:
                    _vol_skipped_screener += 1
                    continue

            result = get_cached_score(sym, df_raw, live, nifty_r5, nifty_r20)
            if result is None:
                continue
            _raw_ltp = live.get("ltp"); live_ltp = float(_raw_ltp if _raw_ltp is not None else df_raw["close"].iloc[-1])
            screener_rows.append({
                "Ticker":  sym,
                "LTP":     round(live_ltp, 2),
                "DayHigh": round(float(live.get("high", df_raw["high"].iloc[-1])), 2),
                "DayLow":  round(float(live.get("low",  df_raw["low"].iloc[-1])),  2),
                "LiveVol": int(live["volume"]) if live.get("volume") else None,
                **result,
            })
        except Exception as _score_err:
            st.session_state.error_log.append(f"{sym} (screener): {_score_err}")

    if not screener_rows:
        st.warning("No stocks passed filters — market may be closed or extraction needed")
    else:
        df_out = pd.DataFrame(screener_rows)

        # ── PORTFOLIO HEAT CAP — scale Kelly fractions so total ≤ heat cap ──
        # If the sum of all KellyFrac values exceeds the configured cap (default 40%),
        # every fraction is scaled down proportionally.
        # This is a post-scoring adjustment — it does NOT change individual scores.
        # The scaled Kelly is shown in the table as "KellyFrac" so the user immediately
        # sees executable position sizes, not theoretical over-allocated fractions.
        if "KellyFrac" in df_out.columns:
            _heat_cap = float(st.session_state.get("portfolio_heat_cap_pct", 40)) / 100.0
            _total_kelly = float(df_out["KellyFrac"].fillna(0).sum())
            if _total_kelly > _heat_cap and _total_kelly > 0:
                _kelly_scale = _heat_cap / _total_kelly
                df_out["KellyFrac"] = (df_out["KellyFrac"] * _kelly_scale).round(3)
                st.caption(
                    f"⚖️ Portfolio heat cap ({int(_heat_cap*100)}%) active — "
                    f"Kelly fractions scaled by {_kelly_scale:.2f}× "
                    f"(raw total was {_total_kelly*100:.1f}%)"
                )

        # ── SIGNAL AUDIT LOG — append this screener run to CSV ───────────────
        # Only writes rows with Score >= min_score threshold so noise rows
        # don't inflate the log. Written here (before display filters) so
        # every qualifying signal is captured regardless of display settings.
        try:
            _audit_ts = datetime.now().strftime("%Y-%m-%d %H:%M")
            _audit_regime = mkt.get("regime", "?")
            _audit_rows = []
            for _, _ar in df_out.iterrows():
                if float(_ar.get("Score", 0)) < 30:
                    continue
                _audit_rows.append({
                    "Timestamp":      _audit_ts,
                    "Ticker":         _ar.get("Ticker", ""),
                    "Score":          round(float(_ar.get("Score", 0)), 1),
                    "SetupType":      _ar.get("SetupType", ""),
                    "Horizon":        _ar.get("Horizon", ""),
                    "Entry":          _ar.get("Entry", ""),
                    "Target":         _ar.get("Target", ""),
                    "Stop":           _ar.get("Stop", ""),
                    "RR":             _ar.get("RR", ""),
                    "KellyFrac":      _ar.get("KellyFrac", ""),
                    "Sector":         _ar.get("Sector", ""),
                    "Regime":         _audit_regime,
                    "EventFlag":      _ar.get("EventFlag", ""),
                    "FundamentalOK":  _ar.get("FundamentalOK", True),
                })
            _append_signal_log(_audit_rows)
        except Exception:
            pass   # audit log failure must never block the screener display

        # Volume filter status note
        if _min_vol_screener > 0 and _vol_skipped_screener > 0:
            st.caption(
                f"🔇 Volume filter active (>{_min_vol_screener:,.0f} avg vol) — "
                f"skipped {_vol_skipped_screener} low-volume stocks · "
                f"{len(screener_rows)} stocks in scan"
            )

        # ── EVENT RISK SUMMARY — show flagged stocks prominently ─────────────
        if "EventFlag" in df_out.columns:
            _ev_flagged = df_out[df_out["EventFlag"].astype(str).str.len() > 0]
            if not _ev_flagged.empty:
                st.warning(
                    f"⚠️ **{len(_ev_flagged)} signal(s) have upcoming corporate events "
                    f"within ±3 days** — scores already adjusted. "
                    f"Tickers: {', '.join(_ev_flagged['Ticker'].tolist()[:10])}"
                )

        # ── REGIME CONTEXT (informational, not a hard gate) ──
        # The regime gate was removed because it was the wrong fix.
        # Blocking all signals in BEAR markets prevented finding the genuine leaders
        # that outperform in corrections (defence, pharma, select smallcaps etc).
        # Instead, the scoring now uses cross-sectional RS (70% weight in F1) which
        # is regime-agnostic: it ranks the stock against its universe peers regardless
        # of absolute market direction. A stock in the 90th percentile of the universe
        # during a correction is a genuine signal — the score reflects that now.
        #
        # Regime is shown prominently so the user can adjust position sizing,
        # but the screener never suppresses signals based on market direction.
        _regime = mkt.get("regime", "BULL")
        if _regime == "BEAR":
            st.warning(
                f"⚠️ **MARKET REGIME: BEAR** — Nifty below 50DMA with falling slope. "
                f"Signals shown are **cross-sectional leaders** — stocks outperforming their peers "
                f"despite the market environment. **Reduce position size. Use tighter stops.**"
            )
        elif _regime == "CHOP":
            st.info(
                f"ℹ️ **MARKET REGIME: CHOP** — Mixed signals. "
                f"Focus on stocks with RS > 10 (top third of F1 score). Avoid Mid/Long horizon setups."
            )

        # FIX 19: Add minimum score threshold slider to filter noise rows from live screener.
        # Walk-forward already uses wf_minscore. Live screener had no equivalent filter,
        # showing Score=1.2 stocks alongside genuine setups.
        _live_min_score = st.slider(
            "Min Score (live screener)",
            min_value=0, max_value=70, value=40, step=5,
            key="live_min_score_slider",
            help="Hides stocks with Score below this threshold. "
                 "Set to 0 to see all. Recommended: 40 for Swing entries, 30 for watching."
        )

        # FIX F: Let the user choose the ranking metric explicitly.
        # Score = raw setup quality (0-100).
        # EMI   = Score × ATR% — rewards volatile high-quality setups that can actually move.
        # CompositeRank = EMI × LiquidityScore × VolumeStability — adds tradability filter.
        # Default to CompositeRank as it is the most complete metric.
        _sort_col = st.selectbox(
            "Rank by",
            options=["CompositeRank", "EMI", "Score"],
            index=0,
            format_func=lambda x: {
                "CompositeRank": "Composite Rank  (EMI × Liquidity × Stability — recommended)",
                "EMI":           "EMI  (Score × ATR% — favours volatile setups)",
                "Score":         "Score  (raw quality, 0–100)",
            }[x],
            key="screener_sort_col",
        )
        df_out = df_out.sort_values(_sort_col, ascending=False).reset_index(drop=True)

        # FIX 19: Apply minimum score threshold filter
        _live_min_score_val = st.session_state.get("live_min_score_slider", 40)
        if _live_min_score_val > 0:
            _before_filter = len(df_out)
            df_out = df_out[df_out["Score"] >= _live_min_score_val].reset_index(drop=True)
            if _before_filter != len(df_out):
                st.caption(
                    f"🔇 Min score filter (≥{_live_min_score_val}) hid "
                    f"{_before_filter - len(df_out)} low-score stocks · "
                    f"{len(df_out)} remaining. Set slider to 0 to see all."
                )
        if "Rank" in df_out.columns:
            df_out.drop(columns=["Rank"], inplace=True)
        df_out.insert(0, "Rank", df_out.index + 1)

        # ── L-3: SECTOR CONCENTRATION CAP ──
        # Toggle in sidebar allows max-1-stock-per-sector enforcement.
        # When ON: the ranked list is filtered so each sector appears at most once.
        # This guarantees sector-diversified signal output regardless of which
        # sectors happen to be strong on a given day.
        # The toggle is OFF by default so existing behaviour is unchanged.
        _sector_cap_on = st.session_state.get("sector_cap_enabled", False)
        if _sector_cap_on and "Sector" in df_out.columns:
            _seen_sectors = set()
            _capped_rows  = []
            for _, _row in df_out.iterrows():
                _sec = str(_row.get("Sector", "?"))
                if _sec == "?" or _sec not in _seen_sectors:
                    _capped_rows.append(_row)
                    if _sec != "?":
                        _seen_sectors.add(_sec)
            df_out = pd.DataFrame(_capped_rows).reset_index(drop=True)
            df_out["Rank"] = df_out.index + 1
            st.info(f"ℹ️ Sector cap ON — showing max 1 stock per sector. {len(df_out)} stocks after capping.")

        # ── MARKET CONTEXT BANNER ──
        _regime      = mkt.get("regime", "BULL")
        _regime_col  = {"BULL": "#00d084", "CHOP": "#ffb347", "BEAR": "#ff3b3b"}.get(_regime, "#888")
        mkt_col      = "#00d084" if mkt["market_ok"] else "#ff3b3b"
        mkt_note     = "  ·  ".join(mkt["market_notes"]) if mkt["market_notes"] else "Market conditions normal"
        nifty_lbl    = "▲ ABOVE 50DMA" if mkt.get("nifty_above_50dma", True) else "▼ BELOW 50DMA"
        vix_lbl      = f"VIX {mkt['vix_level']} FALLING ✓" if mkt["vix_falling"] else f"VIX {mkt['vix_level']} RISING ⚠"
        top_s        = "  ".join(sorted(top_sectors)) if top_sectors else "—"
        st.markdown(f"""
<div style="background:#0d0d0d;border:1px solid #2a2a2a;border-left:4px solid {_regime_col};
padding:10px 16px;margin-bottom:10px;font-family:'IBM Plex Mono',monospace;">
  <div style="display:flex;gap:28px;align-items:center;flex-wrap:wrap;">
    <span style="color:#555;font-size:.6rem;letter-spacing:.12em;">MARKET</span>
    <span style="background:{_regime_col};color:#000;font-size:.72rem;font-weight:700;
          padding:2px 8px;letter-spacing:.1em;">◼ {_regime}</span>
    <span style="color:{mkt_col};font-size:.72rem;font-weight:700;">NIFTY {nifty_lbl}</span>
    <span style="color:{mkt_col};font-size:.72rem;">{vix_lbl}</span>
    <span style="color:#ff8c00;font-size:.72rem;">⭐ SECTORS: {top_s}</span>
    <span style="color:#555;font-size:.65rem;">{mkt_note}</span>
  </div>
</div>""", unsafe_allow_html=True)

        # ── COLUMN SETS ──
        CORE_COLS = ["Rank","Ticker","Sector","SetupType","Horizon","Score","EMI","CompositeRank",
                     "LTP","Entry","Target","Stop","RR","Move%","ATR%",
                     "RSI7","VolRatio","VolZ","RS","CSRank5d","RS_Sector","MA_Struct",
                     "VolCont","BBSqueeze","VolDryUp","CLVAccum","VCP","BreakoutProb","SignalPersist",
                     "CompressionStreak","HHHLScore","GapUpPenalty","DeliveryPct",
                     "Proximity","Candle","Patterns",
                     "UpVolSkew","CPR","SpreadComp","ATRExpOnset","OI_Buildup","VolVelocity","RSDivergence",
                     "HorizonNote"]
        TRADE_COLS = ["Rank","Ticker","Sector","SetupType","Horizon","Score","EMI",
                      "LTP","Entry","Target","Stop","RR","Move%","ATR%",
                      "RSI7","VolRatio","VolZ",
                      "BBSqueeze","VolDryUp","CLVAccum","VCP","VCP_Detected","BreakoutProb",
                      "CompressionStreak","HHHLScore","GapUpPenalty","DeliveryPct",
                      "UpVolSkew","CPR","SpreadComp","ATRExpOnset","OI_Buildup","RSDivergence",
                      "HorizonNote","Sweep","VWMA20_OK","Stability"]
        FACTOR_COLS = ["Rank","Ticker","Sector","SetupType","Score",
                       "RS","RS_Sector","Volume","InstVol","VolCont","RCI",
                       "BBSqueeze","VolDryUp","CLVAccum",
                       "VCP","VCP_Contraction","VCP_VolComp","VCP_VolDryup","VCP_Tightness","VCP_Position",
                       "Coil","MA_Struct","Proximity","ATR_Pot","Candle","VCVE","BasePos",
                       "UpVolSkew","CPR","SpreadComp","ATRExpOnset","OI_Buildup","VolVelocity","RSDivergence"]

        def _vcols(d, cols):
            return [c for c in cols if c in d.columns]

        # ── Bloomberg-style cell coloriser ──
        # Score thresholds derived from the current run's distribution —
        # not hardcoded. Top 25% = green, 25-50% = yellow-green, 50-75% = amber.
        _score_vals = df_out["Score"].dropna()
        _s_p75 = float(_score_vals.quantile(0.75)) if len(_score_vals) >= 4 else 70
        _s_p50 = float(_score_vals.quantile(0.50)) if len(_score_vals) >= 4 else 55
        _s_p25 = float(_score_vals.quantile(0.25)) if len(_score_vals) >= 4 else 40

        def style_df(d, cols):
            disp = d[_vcols(d, cols)].copy()

            def score_bg(v):
                if not isinstance(v, (int, float)): return ""
                if v >= _s_p75: return "background-color:#0d2200;color:#00d084;font-weight:700"
                if v >= _s_p50: return "background-color:#1a2200;color:#b8e06a"
                if v >= _s_p25: return "background-color:#2a1800;color:#ffb347"
                return "color:#555555"

            def setup_bg(v):
                if v == "Reversal": return "background-color:#2a1000;color:#ff8c00;font-weight:700"
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

        # ── SIGNAL GUIDE ──────────────────────────────────────────────────────
        # Bloomberg-style reference: what each column means and what to look for.
        with st.expander("📖 Signal Guide — what to look for and how to read every column", expanded=False):
            st.markdown("""
<div style="font-family:'IBM Plex Mono',monospace;font-size:0.68rem;line-height:1.8;color:#c8c8c8;">

<span style="color:#ff8c00;font-weight:700;letter-spacing:.1em;">◼ HOW THE SCORE WORKS</span><br>
The score (0–100) is built from three primary factors weighted by measured predictive power on NSE data:<br>
&nbsp;&nbsp;• <b style="color:#ff8c00;">SpreadComp (40%)</b> — range compressing + close drifting upward = quiet institutional accumulation<br>
&nbsp;&nbsp;• <b style="color:#ff8c00;">Vol Quiet (40%)</b> — below-average T-1 volume = supply dried up, no one selling<br>
&nbsp;&nbsp;• <b style="color:#ff8c00;">Coil (20%)</b> — tightness of the base at the right price level<br>
Bonus points (up to +8) are added from leading signals below. Soft penalties reduce score for gaps, RSI extremes, illiquidity.<br><br>

<span style="color:#00d084;font-weight:700;letter-spacing:.1em;">◼ PRIMARY LEADING SIGNALS (what fires BEFORE the move)</span><br>
<b style="color:#00d084;">BBSqueeze</b> — Bollinger Band width at its lowest percentile vs 250-day history.
Score of 8 = tightest squeeze in nearly a year. Energy is coiling. The move is near but direction unknown — wait for vol expansion to confirm which way.<br><br>

<b style="color:#00d084;">VolDryUp</b> — 5-day avg volume falling below 20-day avg. Supply exhaustion before a breakout.
Score of 8 = driest vol in the universe today. Classic accumulation signature: no one wants to sell. Reliable 3-7 days before the move.<br><br>

<b style="color:#00d084;">CompressionStreak</b> — Raw count of consecutive days where the daily range (high–low) was below the 20-bar average range.
Streak ≥ 5 = genuine coil. Streak ≥ 8 = textbook base. Look for this alongside BBSqueeze — both high together is the strongest setup signal in the screener.<br><br>

<b style="color:#00d084;">HHHLScore</b> — 0–3 pts. Detects if the last 3 swing highs and 3 swing lows are each higher than the previous (Higher Highs + Higher Lows structure).
Fires BEFORE EMA alignment confirms the trend — roughly 5-10 bars earlier than MA_Struct. Score 3 = perfect structure. Use as early trend confirmation for Pullback setups.<br><br>

<b style="color:#00d084;">SpreadComp</b> — Range narrowing over 5 bars while close drifts upward. Institutional buying fingerprint: they buy slowly to avoid moving price, so range compresses while close creeps up. Score 3 = top percentile of own history.<br><br>

<b style="color:#00d084;">UpVolSkew</b> — Volume on up-close days vs down-close days over 20 sessions. Ratio > 1.5 = buyers are consistently more active than sellers even while price looks flat. Score 3 = top quintile of own history.<br><br>

<b style="color:#00d084;">ATRExpOnset</b> — Detects the FIRST bar where short-term ATR begins expanding after compression. Score decays fast (3 pts on bar 1, 0.6 pts on bar 5). A score of 2+ means the coil started releasing within the last 2 days.<br><br>

<b style="color:#00d084;">OI_Buildup</b> — F&amp;O stocks only. Open interest rising while price coils = institutions building positions before the move. Score 3 = strong OI build with price compression. Zero for non-F&amp;O stocks.<br><br>

<b style="color:#00d084;">DeliveryPct</b> — NSE Bhav Copy delivery percentage (delivery qty / total traded qty). Updated once per session.
&nbsp;&nbsp;≥ 60% → 4 bonus pts — informed money holding overnight, very high conviction<br>
&nbsp;&nbsp;45–60% → 2.5 pts — above-average holding, solid setup<br>
&nbsp;&nbsp;30–45% → 1 pt — neutral<br>
&nbsp;&nbsp;&lt; 20% → −2 pts — pure intraday speculation, no overnight interest — treat signal with caution<br>
&nbsp;&nbsp;Blank → data unavailable for this stock (non-EQ series or Bhav Copy fetch failed)<br>
A breakout stock with DeliveryPct ≥ 60% is categorically stronger than one at 15%. This is one of the most reliable NSE-specific signals available.<br><br>

<span style="color:#ffb347;font-weight:700;letter-spacing:.1em;">◼ DIAGNOSTIC SIGNALS (use for manual confirmation, not for ranking)</span><br>
<b style="color:#ffb347;">VCP</b> — Volatility Contraction Pattern score (Minervini method). Shows as 0–10 but has <b>zero weight in the score formula</b> (IC=-0.19, anti-predictive on small universes).
Use it visually: a stock with VCP ≥ 6 alongside BBSqueeze ≥ 6 is forming a textbook base. Do not rank by VCP alone.<br><br>

<b style="color:#ffb347;">CLVAccum</b> — Close Location Value money flow. Also has <b>zero weight in score</b> (IC=-0.26). Use it as a cross-check: high CLV alongside strong SpreadComp = accumulation with structure (good). High CLV without compression = buying into a move (risky).<br><br>

<span style="color:#ff3b3b;font-weight:700;letter-spacing:.1em;">◼ PENALTIES — WHY A STOCK MIGHT SCORE LOWER THAN EXPECTED</span><br>
<b style="color:#ff3b3b;">GapUpPenalty</b> — Today's open was significantly above T-1 close (gap > 1 ATR). The move already happened at open. Higher penalty = more you are chasing. Score of 10+ means the stock gapped up aggressively — pass unless you were already positioned.<br><br>

<b style="color:#ff3b3b;">SoftPenalty</b> (not shown directly, baked into Score) — Accumulated penalty from: gap-up, RSI overbought vs own p90, low liquidity/ADV, SMA200 breakdown, overextension, already-broke-out vol spike.<br><br>

<span style="color:#1e90ff;font-weight:700;letter-spacing:.1em;">◼ CONFIRMATORY SIGNALS (useful context, lower predictive weight)</span><br>
<b style="color:#1e90ff;">RS / CSRank5d</b> — Cross-sectional rank vs universe (0=bottom, 1=top). Use to confirm the stock is a relative leader. High RS alone does not predict the next 5 days — it mean-reverts at short horizons on NSE. Most useful when RS is improving (RSDivergence high).<br><br>

<b style="color:#1e90ff;">MA_Struct</b> — EMA9/EMA50 ratio percentile over 250 days + convergence proximity. High score = EMA9 above EMA50 and converging. Confirms existing trend but fires after HHHLScore.<br><br>

<b style="color:#1e90ff;">RSI7</b> — Coloured: blue ≤ 35 (oversold, watch for reversal), green 35–60 (healthy), amber 60–70 (elevated), red ≥ 70 (overbought, penalty applied above stock's own p90).<br><br>

<span style="color:#cc88ff;font-weight:700;letter-spacing:.1em;">◼ TRADE MECHANICS</span><br>
<b style="color:#cc88ff;">Entry</b> — For Breakout: 0.1 ATR above 20d resistance. For Pullback: current price near EMA20. Place a buy limit, not market.<br>
<b style="color:#cc88ff;">Stop</b> — Below base low (Breakout) or below EMA50 (Pullback). Never tighten the stop into the noise band.<br>
<b style="color:#cc88ff;">RR</b> — Risk:Reward. Green ≥ 3.0, teal ≥ 2.0, amber ≥ 1.5, red &lt; 1.5. Only trade RR ≥ 2.0 unless CompStreak and HHHL are both high.<br>
<b style="color:#cc88ff;">Horizon</b> — How the engine classifies this setup's expected duration. Imminent BO = act now. Swing 2-5D = buy limit above trigger. Mid 5-14D = base still forming, watch.<br><br>

<span style="color:#ff8c00;font-weight:700;letter-spacing:.1em;">◼ ADAPTIVE WEIGHTS — HOW THE MODEL LEARNS</span><br>
The score formula weights (SpreadComp, VolQuiet, Coil) start as fixed priors from the Jan 2026 backtest.
After each walk-forward run with ≥30 trades, they are automatically updated using measured IC (Q4-Q1 return spread per signal).
The update blends 70% new measurement + 30% prior — conservative enough to prevent overfitting to one regime.
Current weights are shown in the Walk-Forward tab after each run. Run walk-forward on multiple date ranges across different
market conditions (BULL + BEAR + CHOP) to get stable, generalised weights. Each run narrows the weights toward what
actually predicted returns in your specific universe.

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
        # Base scoring factors
        fcols   = ["RS","RS_Sector","Volume","InstVol","VolCont","RCI",
                   "Coil","MA_Struct","Proximity","ATR_Pot","Candle",
                   "UpVolSkew","CPR","SpreadComp","ATRExpOnset"]
        fcolors = ["#ff8c00","#ffb347","#1e90ff","#00ccff","#00d084",
                   "#26a69a","#8bc34a","#9c27b0","#e91e63","#ffc107","#ff5722",
                   "#4db6ac","#f06292","#aed581","#ffb300"]
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
        # Recompute from loaded stock cache at render time — ETF tickers fail often.
        _display_sect_ret = {}
        if st.session_state.get("raw_data_cache"):
            _s5a = {}
            for _sym, _df in st.session_state.raw_data_cache.items():
                _sec = get_sector(_sym)
                if _sec is None:
                    continue
                try:
                    _c = _df["close"] if hasattr(_df, "columns") else pd.DataFrame(_df)["close"]
                    if len(_c) >= 6:
                        _s5a.setdefault(_sec, []).append(float(_c.iloc[-1] / _c.iloc[-6] - 1))
                except Exception:
                    pass
            _display_sect_ret = {s: float(np.mean(v)) for s, v in _s5a.items()}
        _display_sect_ret = {**sector_returns, **_display_sect_ret}

        if _display_sect_ret:
            st.divider()
            st.markdown("### 🗂 SECTOR MOMENTUM  ·  5-Day Returns")
            sdf = pd.DataFrame(
                [(k, round(v*100,2)) for k,v in sorted(_display_sect_ret.items(), key=lambda x:x[1], reverse=True)],
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

def _build_score_explanation(sym: str, result: dict, df_raw: pd.DataFrame):
    """Render a plain-English explanation of every score component."""

    def _signal_row(icon: str, label: str, value_str: str, explanation: str,
                    color: str = "#e8e8e8", bar_pct: float = None):
        """Render one factor row with icon, label, value, bar, and explanation."""
        bar_html = ""
        if bar_pct is not None:
            bar_pct = max(0.0, min(1.0, bar_pct))
            bar_col = ("#00d084" if bar_pct >= 0.65 else
                       "#ffb347" if bar_pct >= 0.35 else "#ff3b3b")
            bar_html = (
                f'<div style="height:4px;background:#1a1a1a;border-radius:2px;margin:2px 0 4px;">'
                f'<div style="width:{bar_pct*100:.0f}%;height:100%;'
                f'background:{bar_col};border-radius:2px;"></div></div>'
            )
        return (
            f'<div style="display:flex;gap:10px;padding:7px 0;'
            f'border-bottom:1px solid #1a1a1a;align-items:flex-start;">'
            f'<span style="font-size:1.0rem;min-width:22px;text-align:center;">{icon}</span>'
            f'<div style="flex:1;">'
            f'<div style="display:flex;justify-content:space-between;align-items:baseline;">'
            f'<span style="font-family:\'IBM Plex Mono\';font-size:0.65rem;color:#888;'
            f'letter-spacing:0.08em;text-transform:uppercase;">{label}</span>'
            f'<span style="font-family:\'IBM Plex Mono\';font-size:0.75rem;'
            f'font-weight:600;color:{color};">{value_str}</span>'
            f'</div>'
            f'{bar_html}'
            f'<span style="font-family:\'IBM Plex Mono\';font-size:0.65rem;'
            f'color:#aaa;line-height:1.5;">{explanation}</span>'
            f'</div></div>'
        )

    # ── Pull values from result dict ──────────────────────────────────────────
    score       = result.get("Score", 0)
    setup       = result.get("SetupType", "—")
    horizon     = result.get("Horizon", "—")
    vol_ratio   = result.get("VolRatio", 1.0)
    vol_quiet   = result.get("VolQuiet", 0)
    spread_pts  = result.get("SpreadPts", 0)
    spread_raw  = result.get("SpreadComp", 0)
    bb_pts      = result.get("BBSqueeze", 0)
    vdu_pts     = result.get("VolDryUp", 0)
    vc_pts      = result.get("VolCont", 0)
    vcp_pts     = result.get("VCP", 0)
    vcp_detect  = result.get("VCP_Detected", False)
    vcp_n       = result.get("VCP_Pullbacks", 0)
    vcp_cont    = result.get("VCP_Contraction", 0)
    prox_pts    = result.get("Proximity", 0)
    rsi         = result.get("RSI7", 50)
    breakout_p  = result.get("BreakoutProb", 0)
    atr_pct     = result.get("ATR%", 0)
    patterns    = result.get("Patterns", "—")
    penalty     = result.get("SoftPenalty", 0)
    stability   = result.get("Stability", 0)
    kelly       = result.get("KellyFrac", 0)
    oi_buildup  = result.get("OI_Buildup", 0)
    uv_skew     = result.get("UpVolSkew", 0)
    entry       = result.get("Entry", 0)
    target      = result.get("Target", 0)
    stop        = result.get("Stop", 0)
    rr          = result.get("RR", 0)

    # Compute vol trend from raw data (last 5 vs prior 15 bars)
    try:
        _hv      = df_raw["volume"].iloc[:-1].replace(0, float("nan")).dropna()
        _v5_avg  = float(_hv.tail(5).mean())
        _v20_avg = float(_hv.tail(20).mean())
        _vtrend  = _v5_avg / (_v20_avg + 1e-9)
        _vchange = (_vtrend - 1.0) * 100
        _v5_str  = f"{_vchange:+.0f}% vs 20d avg"
    except Exception:
        _vtrend  = 1.0
        _v5_str  = "—"

    # ATR compression: compare recent ATR to 60d history
    try:
        _c   = df_raw["close"].iloc[:-1]
        _h   = df_raw["high"].iloc[:-1]
        _l   = df_raw["low"].iloc[:-1]
        _tr  = pd.concat([_h - _l,
                           (_h - _c.shift(1)).abs(),
                           (_l - _c.shift(1)).abs()], axis=1).max(axis=1)
        _atr5  = float(_tr.tail(5).mean())
        _atr20 = float(_tr.tail(20).mean())
        _atr60 = float(_tr.tail(60).mean())
        _atr_comp_ratio = _atr5 / (_atr60 + 1e-9)
        _atr_comp_pct   = (1.0 - _atr_comp_ratio) * 100
    except Exception:
        _atr_comp_pct   = 0.0
        _atr_comp_ratio = 1.0

    # BB width context
    try:
        _bc   = df_raw["close"].iloc[:-1]
        _sma  = _bc.rolling(20).mean()
        _std  = _bc.rolling(20).std()
        _bbw  = (2 * _std / (_sma + 1e-9)).dropna()
        _bbw_now = float(_bbw.iloc[-1])
        _bbw_med = float(_bbw.quantile(0.50))
        _bbw_pct = int(((_bbw <= _bbw_now).mean()) * 100)
    except Exception:
        _bbw_now = 0; _bbw_med = 0; _bbw_pct = 50

    # Price proximity to resistance
    try:
        _hh = df_raw["high"].iloc[:-1]
        _base_hi = float(_hh.tail(20).max())
        _ltp_s   = float(df_raw["close"].iloc[-2])   # T-1 close
        _dist_pct = (_base_hi - _ltp_s) / (_ltp_s + 1e-9) * 100
    except Exception:
        _dist_pct = 5.0; _base_hi = 0

    # ── Score colour ─────────────────────────────────────────────────────────
    score_col = ("#00ff88" if score >= 70 else
                 "#00d084" if score >= 55 else
                 "#ffb347" if score >= 40 else "#ff3b3b")
    setup_col = ("#ff8c00" if setup == "Reversal" else
               "#1e90ff" if setup == "Breakout" else "#cc88ff")

    rows_html = []

    # ── 1. VOL QUIET (primary factor, weight 28%) ────────────────────────────
    if vol_ratio < 0.70:
        vq_icon = "🤫"; vq_col = "#00d084"
        vq_exp = (f"Yesterday's volume was <b>{vol_ratio:.2f}×</b> the 20-day average — "
                  f"stock is <b>unusually quiet</b>. Institutions accumulate in silence; "
                  f"low-volume coiling before a move is the most reliable pre-expansion signal.")
    elif vol_ratio < 1.0:
        vq_icon = "🔇"; vq_col = "#b8e06a"
        vq_exp = (f"Yesterday's volume was <b>{vol_ratio:.2f}×</b> the 20-day average — "
                  f"<b>below average</b>. Supply is drying up. No distribution visible.")
    elif vol_ratio < 1.4:
        vq_icon = "📊"; vq_col = "#ffb347"
        vq_exp = (f"Yesterday's volume was <b>{vol_ratio:.2f}×</b> the 20-day average — "
                  f"slightly elevated. Normal trading activity, no strong signal either way.")
    else:
        if setup == "Reversal":
            vq_icon = "🔊"; vq_col = "#00d084"
            vq_exp = (f"Volume was <b>{vol_ratio:.2f}×</b> the 20-day average — "
                      f"<b>capitulation surge</b>. Panic selling with heavy vol = sellers exhausted. "
                      f"High vol on oversold stock = fuel for the bounce.")
        else:
            vq_icon = "🔊"; vq_col = "#ff6b6b"
            vq_exp = (f"Yesterday's volume was <b>{vol_ratio:.2f}×</b> the 20-day average — "
                      f"<b>high</b>. Price may already have moved. Enter only if price is still "
                      f"below resistance. High vol on the scoring day is a caution flag.")
    # For Reversal: bar shows how high vol is (higher = better). For others: inverse.
    _vol_bar = (min(vol_ratio / 3.0, 1.0) if setup == "Reversal" else vol_quiet / 14.0)
    rows_html.append(_signal_row(vq_icon,
                                 ("Volume Surge (40% weight)" if setup == "Reversal" else "Volume Quiet (40% weight)"),
                                 f"{vol_ratio:.2f}× 20d avg",
                                 vq_exp, vq_col, _vol_bar))

    # ── 2. SPREAD COMPRESSION (primary factor, weight 22%) ──────────────────
    if spread_raw >= 2.0:
        sc_icon = "🗜️"; sc_col = "#00d084"
        sc_exp = (f"Over the last 5 days, the daily price range is <b>narrowing</b> while "
                  f"the close is <b>drifting up</b> (SpreadComp = {spread_raw:.1f}/3.0). "
                  f"This is textbook institutional accumulation: buyers absorb supply quietly, "
                  f"compressing the range without letting price fall.")
    elif spread_raw >= 1.2:
        sc_icon = "📐"; sc_col = "#b8e06a"
        sc_exp = (f"Range is moderately compressing with a slight upward close drift "
                  f"(SpreadComp = {spread_raw:.1f}/3.0). Early-stage accumulation pattern "
                  f"— not yet fully developed but directionally positive.")
    elif spread_raw >= 0.5:
        sc_icon = "↔️"; sc_col = "#ffb347"
        sc_exp = (f"Some range compression but close drift is weak (SpreadComp = {spread_raw:.1f}/3.0). "
                  f"Sideways structure — wait for close to start rising within the narrow range.")
    else:
        sc_icon = "↕️"; sc_col = "#ff6b6b"
        sc_exp = (f"Range is wide and/or close is trending down (SpreadComp = {spread_raw:.1f}/3.0). "
                  f"No accumulation signature. Distribution possible.")
    rows_html.append(_signal_row(sc_icon, "Spread Compression (40% weight)",
                                 f"{spread_raw:.1f} / 3.0",
                                 sc_exp, sc_col, min(spread_raw / 3.0, 1.0)))

    # ── 3. VOLUME DRY-UP (12%) ───────────────────────────────────────────────
    if _vtrend < 0.70:
        vdu_icon = "🏜️"; vdu_col = "#00d084"
        vdu_exp = (f"5-day average volume is <b>{_v5_str}</b>. Sellers have stepped away — "
                   f"supply exhaustion. Historically, volume drying up over 5+ days before "
                   f"a breakout is a strong pre-move signal (Minervini Stage 2 setup).")
    elif _vtrend < 0.90:
        vdu_icon = "📉"; vdu_col = "#b8e06a"
        vdu_exp = (f"5-day avg volume is <b>{_v5_str}</b> — declining. "
                   f"Supply is reducing. Watch for another 2-3 days of low volume "
                   f"to confirm full dry-up before entry.")
    elif _vtrend < 1.10:
        vdu_icon = "📊"; vdu_col = "#888"
        vdu_exp = (f"5-day avg volume is <b>{_v5_str}</b> — flat. "
                   f"Neither accumulation nor distribution dominant.")
    else:
        vdu_icon = "📈"; vdu_col = "#ffb347"
        vdu_exp = (f"5-day avg volume is <b>{_v5_str}</b> — elevated over the period. "
                   f"Could indicate accumulation, but also distribution. "
                   f"Confirm with price action (closes near highs = accumulation).")
    rows_html.append(_signal_row(vdu_icon, "Volume Dry-Up (diagnostic)",
                                 _v5_str, vdu_exp, vdu_col, vdu_pts / 8.0))

    # ── 4. BB WIDTH SQUEEZE (10%) ────────────────────────────────────────────
    if _bbw_pct <= 15:
        bb_icon = "⚡"; bb_col = "#00ff88"
        bb_exp  = (f"Bollinger Band width is in the <b>bottom {_bbw_pct}% of its 250-day history</b> — "
                   f"one of the tightest squeezes this stock has seen. "
                   f"Historically, extreme BB squeezes precede the largest moves. "
                   f"Direction unknown, but magnitude likely to be significant.")
    elif _bbw_pct <= 35:
        bb_icon = "🎯"; bb_col = "#00d084"
        bb_exp  = (f"Bollinger Band width is in the <b>bottom {_bbw_pct}% of history</b>. "
                   f"Volatility is compressed. Energy building. "
                   f"A close above the upper band on expanding volume would confirm breakout direction.")
    elif _bbw_pct <= 55:
        bb_icon = "📏"; bb_col = "#ffb347"
        bb_exp  = (f"BB width at the <b>{_bbw_pct}th percentile</b> of history — near median. "
                   f"Moderate compression. Not at extremes yet.")
    else:
        bb_icon = "📡"; bb_col = "#888"
        bb_exp  = (f"BB width at the <b>{_bbw_pct}th percentile</b> of history — "
                   f"bands are wide. Volatility is elevated, not compressed. "
                   f"Wait for contraction before entry.")
    rows_html.append(_signal_row(bb_icon, "Bollinger Squeeze (part of Coil 20%)",
                                 f"Bottom {_bbw_pct}% of history",
                                 bb_exp, bb_col, bb_pts / 8.0))

    # ── 5. PROXIMITY TO TRIGGER (10%) ───────────────────────────────────────
    if _dist_pct <= 1.0:
        pr_icon = "🚨"; pr_col = "#00ff88"
        pr_exp  = (f"Price is only <b>{_dist_pct:.1f}% below the 20-day resistance</b> at "
                   f"₹{_base_hi:.2f}. At the trigger. A strong close above this level "
                   f"on volume ≥1.5× average would confirm the breakout.")
    elif _dist_pct <= 3.0:
        pr_icon = "🎯"; pr_col = "#00d084"
        pr_exp  = (f"Price is <b>{_dist_pct:.1f}% below resistance</b> at ₹{_base_hi:.2f}. "
                   f"Close proximity — within 1-2 sessions of a potential trigger. "
                   f"Place a buy-stop limit order above resistance.")
    elif _dist_pct <= 7.0:
        pr_icon = "📍"; pr_col = "#ffb347"
        pr_exp  = (f"Price is <b>{_dist_pct:.1f}% below resistance</b> at ₹{_base_hi:.2f}. "
                   f"Building the base. Watch for range to tighten further "
                   f"before the breakout attempt.")
    else:
        pr_icon = "⏳"; pr_col = "#888"
        pr_exp  = (f"Price is <b>{_dist_pct:.1f}% below resistance</b> at ₹{_base_hi:.2f}. "
                   f"Still forming the base. Not immediately actionable.")
    rows_html.append(_signal_row(pr_icon, "Proximity to Trigger (diagnostic)",
                                 f"{_dist_pct:.1f}% from ₹{_base_hi:.2f}",
                                 pr_exp, pr_col, prox_pts / 10.0))

    # ── 6. ATR COMPRESSION (8%) ──────────────────────────────────────────────
    if _atr_comp_pct >= 40:
        at_icon = "🪄"; at_col = "#00d084"
        at_exp  = (f"5-day ATR is <b>{_atr_comp_pct:.0f}% lower</b> than the 60-day average ATR. "
                   f"Daily range is shrinking — the stock is coiling like a spring. "
                   f"Compressed volatility historically expands sharply when price breaks structure.")
    elif _atr_comp_pct >= 20:
        at_icon = "📉"; at_col = "#b8e06a"
        at_exp  = (f"5-day ATR is <b>{_atr_comp_pct:.0f}% lower</b> than the 60-day average. "
                   f"Moderate compression. Range contracting — coiling in progress.")
    elif _atr_comp_pct >= 0:
        at_icon = "📊"; at_col = "#888"
        at_exp  = (f"5-day ATR is <b>{_atr_comp_pct:.0f}% lower</b> than the 60-day average. "
                   f"Slight compression. Not yet at historically meaningful levels.")
    else:
        at_icon = "🌊"; at_col = "#ffb347"
        at_exp  = (f"5-day ATR is <b>{abs(_atr_comp_pct):.0f}% ABOVE</b> the 60-day average. "
                   f"Volatility is expanding, not compressing. "
                   f"May indicate the move has already begun — check if price is still below resistance.")
    rows_html.append(_signal_row(at_icon, "ATR Compression (diagnostic)",
                                 f"5d ATR {_atr_comp_pct:+.0f}% vs 60d avg",
                                 at_exp, at_col, vc_pts / 10.0))

    # ── 7. VCP PATTERN (4%) ──────────────────────────────────────────────────
    if vcp_detect and vcp_n >= 3:
        vp_icon = "🌀"; vp_col = "#00d084"
        vp_exp  = (f"<b>VCP detected</b> with {vcp_n} successive pullbacks, "
                   f"each shallower than the last (contraction score: {vcp_cont:.0%}). "
                   f"Mark Minervini's Volatility Contraction Pattern — a high-probability "
                   f"setup where each pullback tests less ground, trapping late sellers.")
    elif vcp_detect:
        vp_icon = "📐"; vp_col = "#b8e06a"
        vp_exp  = (f"Partial VCP forming ({vcp_n} pullbacks identified). "
                   f"Contraction is {vcp_cont:.0%} — pullbacks are narrowing. "
                   f"Pattern needs one more tight pullback to fully qualify.")
    elif vcp_n >= 2:
        vp_icon = "〰️"; vp_col = "#ffb347"
        vp_exp  = (f"{vcp_n} pullbacks identified but contraction is not yet consistent. "
                   f"Watch for the next pullback to be shallower — that would confirm VCP.")
    else:
        vp_icon = "➖"; vp_col = "#666"
        vp_exp  = ("Insufficient swing structure to confirm a VCP. "
                   "Could be early in base building, or stock is trending smoothly without pullbacks.")
    rows_html.append(_signal_row(vp_icon, "VCP Pattern (diagnostic)",
                                 f"{vcp_n} pullbacks",
                                 vp_exp, vp_col, vcp_pts / 10.0))

    # ── 8. RSI CONTEXT (diagnostic, not in score) ────────────────────────────
    if rsi < 35:
        ri_icon = "🔵"; ri_col = "#1e90ff"
        ri_exp  = (f"RSI({result.get('RSI7',7):.0f}) = <b>{rsi:.0f}</b> — <b>oversold</b>. "
                   f"For Pullback setups, this means the pullback is deep — "
                   f"higher reward but also higher risk. Wait for RSI to turn up before entering.")
    elif rsi < 50:
        ri_icon = "🟡"; ri_col = "#ffb347"
        ri_exp  = (f"RSI = <b>{rsi:.0f}</b> — recovering from a pullback. "
                   f"Momentum building from a low base. "
                   f"Good zone for Pullback entries — not yet overbought.")
    elif rsi < 65:
        ri_icon = "🟢"; ri_col = "#00d084"
        ri_exp  = (f"RSI = <b>{rsi:.0f}</b> — healthy momentum zone. "
                   f"Stock has strength without being overbought. "
                   f"Ideal zone for breakout continuation.")
    elif rsi < 80:
        ri_icon = "🟠"; ri_col = "#ffb347"
        ri_exp  = (f"RSI = <b>{rsi:.0f}</b> — elevated. Watch for mean reversion. "
                   f"On NSE, RSI > 70 stocks tend to underperform over the next 5 days. "
                   f"Tighten stop if already in position.")
    else:
        ri_icon = "🔴"; ri_col = "#ff3b3b"
        ri_exp  = (f"RSI = <b>{rsi:.0f}</b> — <b>overbought</b>. "
                   f"Historically mean-reverting at 5-day horizon on NSE. "
                   f"High RSI is a contra-indicator for new entries.")
    rows_html.append(_signal_row(ri_icon, "RSI Momentum (diagnostic)",
                                 f"{rsi:.0f}",
                                 ri_exp, ri_col, rsi / 100.0))

    # ── 9. PENALTIES ─────────────────────────────────────────────────────────
    if penalty > 5:
        pen_exp = (f"Score deducted <b>{penalty:.1f} pts</b> for: "
                   f"elevated RSI, low liquidity, extreme gap, or below SMA200. "
                   f"These conditions reduce the probability of follow-through.")
        rows_html.append(_signal_row("⚠️", "Penalties Applied", f"−{penalty:.1f} pts",
                                     pen_exp, "#ff6b6b", max(0, 1 - penalty / 30)))
    elif penalty > 0:
        rows_html.append(_signal_row("⚠️", "Minor Penalties", f"−{penalty:.1f} pts",
                                     "Small deductions for marginal conditions. Not a concern.",
                                     "#888", 0.85))

    # ── 10. BREAKOUT PROBABILITY ─────────────────────────────────────────────
    bp_pct = int(breakout_p * 100)
    if bp_pct >= 65:
        bp_exp = (f"<b>{bp_pct}%</b> of the pre-expansion checklist is complete: "
                  f"volume quiet ✓, spread compressing ✓, ATR coiling ✓. "
                  f"Structural conditions are aligned for a move within 1-5 sessions.")
    elif bp_pct >= 45:
        bp_exp = (f"<b>{bp_pct}%</b> of the checklist — partially aligned. "
                  f"Some conditions confirmed, others still developing. "
                  f"Can watch and enter when remaining signals confirm.")
    else:
        bp_exp = (f"Only <b>{bp_pct}%</b> of pre-expansion conditions met. "
                  f"Setup is early-stage or incomplete. Higher patience required.")
    rows_html.append(_signal_row("🎯", "Pre-Expansion Checklist",
                                 f"{bp_pct}% complete",
                                 bp_exp,
                                 "#00d084" if bp_pct >= 65 else "#ffb347" if bp_pct >= 45 else "#888",
                                 breakout_p))

    # ── 11. OI Buildup (F&O stocks) ──────────────────────────────────────────
    if oi_buildup > 0:
        rows_html.append(_signal_row("🏛️", "OI Buildup (F&O)",
                                     f"+{oi_buildup:.1f} pts",
                                     (f"Open interest is <b>rising while price coils</b>. "
                                      f"Institutional futures positioning before a move. "
                                      f"Rising OI + compression = strong conviction signal."),
                                     "#00d084", oi_buildup / 3.0))

    # ── 12. Up-Volume Skew ───────────────────────────────────────────────────
    if uv_skew > 1.5:
        rows_html.append(_signal_row("📦", "Quiet Accumulation",
                                     f"+{uv_skew:.1f} pts",
                                     ("Volume on up-days is <b>consistently higher</b> than on down-days "
                                      "over the last 20 sessions. Buyers are absorbing supply without "
                                      "letting price rise — classic institutional accumulation signature."),
                                     "#00d084", uv_skew / 3.0))

    # ── SUMMARY ──────────────────────────────────────────────────────────────
    if score >= 68:
        summary_icon = "🚀"
        summary_text = (f"<b>Strong setup</b> — multiple pre-expansion conditions aligned. "
                        f"Entry on pullback to EMA or break above ₹{_base_hi:.2f} resistance.")
        summary_col  = "#00d084"
    elif score >= 52:
        summary_icon = "📈"
        summary_text = (f"<b>Developing setup</b> — conditions partially in place. "
                        f"Watch for final confirmation (volume dry-up + tight range).")
        summary_col  = "#ffb347"
    else:
        summary_icon = "⏳"
        summary_text = ("Setup is <b>early or incomplete</b>. "
                        "Base building phase — not yet actionable. Add to watchlist.")
        summary_col  = "#888"

    # ── RENDER ───────────────────────────────────────────────────────────────
    with st.expander(f"◼ SCORE BREAKDOWN — WHY {sym} SCORES {score:.0f}  (CLICK TO EXPAND)", expanded=True):
        # Header
        st.markdown(f"""
<div style="display:flex;align-items:center;gap:16px;padding:10px 0 12px;
            border-bottom:2px solid #2a2a2a;margin-bottom:8px;">
  <div>
    <div style="font-family:'IBM Plex Mono';font-size:0.60rem;color:#888;
                letter-spacing:0.12em;text-transform:uppercase;">Composite Score</div>
    <div style="font-family:'IBM Plex Mono';font-size:2.0rem;font-weight:700;
                color:{score_col};line-height:1.1;">{score:.0f}</div>
  </div>
  <div style="width:1px;height:50px;background:#2a2a2a;"></div>
  <div>
    <div style="font-family:'IBM Plex Mono';font-size:0.60rem;color:#888;letter-spacing:0.1em;">SETUP</div>
    <div style="font-family:'IBM Plex Mono';font-size:0.85rem;font-weight:600;
                color:{setup_col};">{setup}</div>
    <div style="font-family:'IBM Plex Mono';font-size:0.65rem;color:#666;">{horizon}</div>
  </div>
  <div style="width:1px;height:50px;background:#2a2a2a;"></div>
  <div>
    <div style="font-family:'IBM Plex Mono';font-size:0.60rem;color:#888;letter-spacing:0.1em;">TRADE LEVELS</div>
    <div style="font-family:'IBM Plex Mono';font-size:0.68rem;color:#e8e8e8;">
      Entry ₹{entry:.2f} → Target ₹{target:.2f} → Stop ₹{stop:.2f}
    </div>
    <div style="font-family:'IBM Plex Mono';font-size:0.65rem;color:#ffb347;">
      R:R = 1:{rr:.1f}  ·  Kelly = {kelly*100:.1f}%  ·  ATR = {atr_pct:.1f}%
    </div>
  </div>
  <div style="margin-left:auto;padding:8px 16px;background:#0d0d0d;border:1px solid #2a2a2a;">
    <div style="font-family:'IBM Plex Mono';font-size:1.2rem;">{summary_icon}</div>
    <div style="font-family:'IBM Plex Mono';font-size:0.62rem;color:{summary_col};max-width:200px;">
      {summary_text.replace('<b>','').replace('</b>','')}
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

        # Factor rows
        st.markdown(
            '<div style="font-family:\'IBM Plex Mono\';">' +
            "".join(rows_html) +
            '</div>',
            unsafe_allow_html=True
        )

        # Candlestick pattern
        if patterns and patterns != "—":
            st.markdown(
                f'<div style="margin-top:8px;padding:6px 10px;background:#0d1a00;'
                f'border-left:3px solid #00d084;font-family:\'IBM Plex Mono\';font-size:0.68rem;color:#aaa;">'
                f'🕯️ <b style="color:#00d084;">Pattern detected:</b> {patterns} — '
                f'candlestick structure supports the directional bias.</div>',
                unsafe_allow_html=True
            )



st.divider()
st.markdown("## ▶ CHART TERMINAL")

symbols_list = sorted(st.session_state.raw_data_cache.keys())
if not symbols_list:
    st.info("Run extraction first")
    st.stop()

# ── CHART BUILDER ──

def bb_chart(sym, df_raw, live, signal_date=None, result=None):
    """
    Build a Bloomberg-styled candlestick chart.
    Fix 30: Accepts result dict to overlay Darvas box levels (resistance + support)
    as horizontal dashed lines when the stock has an active Darvas box.
    """
    df = df_raw.copy()
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").tail(120)   # last 120 bars for clarity

    # apply live patch on last bar
    ltp = live.get("ltp")
    if ltp is not None:
        df.at[df.index[-1], "close"] = ltp
        df.at[df.index[-1], "high"]  = max(float(df["high"].iloc[-1]), ltp)
        df.at[df.index[-1], "low"]   = min(float(df["low"].iloc[-1]),  ltp)
    vol_live = live.get("volume")
    if vol_live is not None:
        df.at[df.index[-1], "volume"] = vol_live

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
        atr_i = float(row["atr14"]) if pd.notna(row["atr14"]) else float(df["atr14"].dropna().iloc[-1])

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
                # Initialise trailing stop: 1.5×ATR below entry (covers noise, not trend)
                _trail_stop  = c - 1.5 * atr_i
                _highest_c   = c   # track highest close seen since entry
        else:
            # ── ATR TRAILING STOP ENGINE ──
            # Trail stop ratchets up as price rises — never moves down.
            # Stop distance = 1.5×ATR below highest close seen since entry.
            # This is derived from the stock's own volatility, not a fixed %.
            # When trade is profitable, progressively tighten to 1×ATR
            # (protects gains without exiting on normal intraday noise).
            _highest_c = max(_highest_c, c)
            _gain_atr  = (_highest_c - entry_price) / (atr_i + 1e-9)
            # Trail multiplier shrinks from 1.5 toward 1.0 as gain grows
            # At 0 ATR gain → 1.5× stop; at 3+ ATR gain → 1.0× stop (lock in profit)
            _trail_mult = max(1.0, 1.5 - _gain_atr * (0.5 / 3.0))
            _trail_stop = max(_trail_stop, _highest_c - _trail_mult * atr_i)

            gain = (c - entry_price) / entry_price * 100 if entry_price != 0 else 0.0
            # SELL CONDITIONS: trailing stop hit OR structural breakdown
            if (c < _trail_stop or               # ATR trail hit
                c < float(row["e20"])):          # price falls back through trend support
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

    # ── Fix 30: Darvas box resistance and support levels ──
    if result is not None:
        _dbox_hi = result.get("DarvasBox")
        _dbox_lo = result.get("DarvasLow")
        _in_box  = result.get("DarvasInBox", False)
        _box_col = BB["purple"] if "purple" in BB else "#cc88ff"
        if _dbox_hi and not (isinstance(_dbox_hi, float) and np.isnan(_dbox_hi)):
            fig.add_hline(
                y=float(_dbox_hi), row=1, col=1,
                line=dict(color=_box_col, width=1.0, dash="dot"),
                annotation_text=f"Darvas R {float(_dbox_hi):.2f}",
                annotation_font=dict(color=_box_col, size=8, family="IBM Plex Mono"),
                annotation_position="right"
            )
        if _dbox_lo and not (isinstance(_dbox_lo, float) and np.isnan(_dbox_lo)):
            fig.add_hline(
                y=float(_dbox_lo), row=1, col=1,
                line=dict(color=_box_col, width=0.8, dash="dot"),
                annotation_text=f"Darvas S {float(_dbox_lo):.2f}",
                annotation_font=dict(color=_box_col, size=8, family="IBM Plex Mono"),
                annotation_position="right"
            )
        # Shade the Darvas box region if price is inside it
        if _dbox_hi and _dbox_lo and _in_box and \
                not (isinstance(_dbox_hi, float) and np.isnan(_dbox_hi)):
            fig.add_hrect(
                y0=float(_dbox_lo), y1=float(_dbox_hi),
                fillcolor="rgba(204,136,255,0.05)",
                line_width=0, row=1, col=1
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
        _raw_ltp_q = live.get("ltp"); ltp_v = float(_raw_ltp_q if _raw_ltp_q is not None else df_raw["close"].iloc[-1])
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

            # mini score table — thresholds derived from current distribution
            _q_scores = q_df["Score"].dropna()
            _q_p75 = float(_q_scores.quantile(0.75)) if len(_q_scores) >= 4 else 70
            _q_p50 = float(_q_scores.quantile(0.50)) if len(_q_scores) >= 4 else 55
            _q_p25 = float(_q_scores.quantile(0.25)) if len(_q_scores) >= 4 else 40
            def score_bg(val):
                if not isinstance(val,(int,float)): return ""
                if val >= _q_p75: return "background-color:#1a3300;color:#00d084"
                if val >= _q_p50: return "background-color:#1a2200;color:#7dba34"
                if val >= _q_p25: return "background-color:#2a1a00;color:#ffb347"
                return "color:#666"
            def setup_bg(val):
                if val == "Reversal": return "background-color:#2a1000;color:#ff8c00;font-weight:700"
                return ("background-color:#001a2a;color:#1e90ff" if val=="Breakout"
                        else "background-color:#1a001a;color:#cc88ff")

            styled_q = (q_df.style
                        .applymap(score_bg,  subset=["Score"])
                        .applymap(setup_bg,  subset=["Setup"]))
            _tbl_h = min(35 * len(q_df) + 38, 260)   # fit rows, cap so metrics stay visible
            st.dataframe(styled_q, use_container_width=True, hide_index=True, height=_tbl_h)

            # ── FUNDAMENTALS PANEL for selected stock ────────────────────────
            _sel = st.session_state.get("chart_sym", ticker_opts[0] if ticker_opts else None)
            if _sel and _sel in st.session_state.raw_data_cache:
                _fd = st.session_state.raw_data_cache[_sel]
                _fl = get_live_bar(_sel)
                _fr = get_cached_score(_sel, _fd, _fl, nifty_r5, nifty_r20)
                if _fr:
                    st.markdown(
                        f'<div style="font-family:IBM Plex Mono;font-size:0.60rem;'
                        f'color:#666;letter-spacing:0.12em;text-transform:uppercase;'
                        f'margin:10px 0 4px;">◼ {_sel} — KEY METRICS</div>',
                        unsafe_allow_html=True)

                    # Build metrics as compact HTML rows
                    _above_sma = "✅ Above" if _fr.get("AboveSMA200") else "❌ Below"
                    _pos52 = _fr.get("Pos52W", 0)
                    _pos52_pct = f"{_pos52*100:.0f}th %ile"

                    # Price levels
                    _ltp_f = float(_fl.get("ltp") or _fd["close"].iloc[-1])
                    _e9  = _fr.get("EMA9",  0)
                    _e20 = _fr.get("EMA20", 0)
                    _e50 = _fr.get("EMA50", 0)

                    # Returns: compute from raw data
                    _fc = _fd["close"]
                    _r1  = f"{(_fc.iloc[-1]/_fc.iloc[-2]-1)*100:+.1f}%" if len(_fc)>=2  else "—"
                    _r5  = f"{(_fc.iloc[-1]/_fc.iloc[-6]-1)*100:+.1f}%" if len(_fc)>=6  else "—"
                    _r20 = f"{(_fc.iloc[-1]/_fc.iloc[-21]-1)*100:+.1f}%" if len(_fc)>=21 else "—"

                    # 52w high/low
                    _hi52 = float(_fd["high"].tail(252).max()) if len(_fd) >= 50 else float(_fd["high"].max())
                    _lo52 = float(_fd["low"].tail(252).min())  if len(_fd) >= 50 else float(_fd["low"].min())
                    _from_hi = f"{(_ltp_f/_hi52-1)*100:.1f}%"
                    _from_lo = f"{(_ltp_f/_lo52-1)*100:.1f}%"

                    # ADV turnover
                    _adv = _fr.get("ADVTurnover", 0)
                    _adv_str = f"₹{_adv:.0f}Cr" if _adv >= 1 else f"₹{_adv*100:.0f}L"

                    # Stability / Liquidity
                    _stab = _fr.get("Stability", 0)
                    _liq  = _fr.get("LiquidityScore", 0)
                    _sect = _fr.get("Sector", "?")
                    _atr  = _fr.get("ATR%", 0)
                    _rs   = _fr.get("RS_vs_Nifty", 0)

                    def _mrow(label, val, good=None):
                        col = "#e8e8e8"
                        if good is True:  col = "#00d084"
                        if good is False: col = "#ff6b6b"
                        return (f'<div style="display:flex;justify-content:space-between;'
                                f'padding:3px 0;border-bottom:1px solid #111;">'
                                f'<span style="color:#666;">{label}</span>'
                                f'<span style="color:{col};font-weight:600;">{val}</span></div>')

                    rows = [
                        _mrow("Sector",      _sect),
                        _mrow("SMA200",      _above_sma, good=_fr.get("AboveSMA200")),
                        _mrow("52W Position",_pos52_pct, good=_pos52>0.6),
                        _mrow("52W High",    f"₹{_hi52:,.0f}  ({_from_hi})"),
                        _mrow("52W Low",     f"₹{_lo52:,.0f}  ({_from_lo})"),
                        _mrow("EMA 9",       f"₹{_e9:,.1f}",  good=_ltp_f > _e9),
                        _mrow("EMA 20",      f"₹{_e20:,.1f}", good=_ltp_f > _e20),
                        _mrow("EMA 50",      f"₹{_e50:,.1f}", good=_ltp_f > _e50),
                        _mrow("Return 1D",   _r1, good=_r1.startswith("+")),
                        _mrow("Return 5D",   _r5, good=_r5.startswith("+")),
                        _mrow("Return 20D",  _r20, good=_r20.startswith("+")),
                        _mrow("ATR%",        f"{_atr:.1f}%"),
                        _mrow("RS vs Nifty", f"{_rs:+.1f}%", good=_rs>0),
                        _mrow("ADV Turnover",_adv_str, good=_adv>50),
                        _mrow("Stability",   f"{_stab:.2f}",  good=_stab>0.55),
                        _mrow("Liquidity",   f"{_liq:.2f}",   good=_liq>0.6),
                    ]
                    st.markdown(
                        f'<div style="font-family:IBM Plex Mono;font-size:0.67rem;'
                        f'margin-bottom:0;padding-bottom:0;">'
                        + "".join(rows) + "</div>",
                        unsafe_allow_html=True)

        with col_chart:
            sym = chosen   # use value from this run, not stale session_state
            if sym and sym in st.session_state.raw_data_cache:
                df_raw = st.session_state.raw_data_cache[sym]
                live   = get_live_bar(sym)
                result = get_cached_score(sym, df_raw, live, nifty_r5, nifty_r20)   # PERF: cached

                # ── TICKER HEADER ──
                _raw_ltp_h = live.get("ltp"); ltp_v = float(_raw_ltp_h if _raw_ltp_h is not None else df_raw["close"].iloc[-1])
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
                    r1c, r2c, r3c, r4c, r5c, r6c, r7c, r8c = st.columns(8)
                    for col_m, label, val in [
                        (r1c, "SCORE",   result["Score"]),
                        (r2c, "SETUP",   result["SetupType"]),
                        (r3c, "RSI(7)",  result["RSI7"]),
                        (r4c, "VOL×",    result["VolRatio"]),
                        (r5c, "RS%",     result["RS_vs_Nifty"]),
                        (r6c, "PATTERN", result["Patterns"][:12] if result["Patterns"] else "—"),
                        # Fix 29: Kelly fraction — shows recommended position size
                        (r7c, "KELLY%",  f"{result.get('KellyFrac', 0)*100:.1f}%"),
                        # Fix 31: SoftPenalty — shows score deduction from conditions
                        (r8c, "PENALTY", f"-{result.get('SoftPenalty', 0):.1f}" if result.get('SoftPenalty', 0) > 0 else "0"),
                    ]:
                        col_m.metric(label, val)

                fig = bb_chart(sym, df_raw, live, result=result)
                st.plotly_chart(fig, use_container_width=True, config={
                    "displayModeBar": True,
                    "modeBarButtonsToRemove": ["lasso2d","select2d"],
                    "displaylogo": False
                })

                # ── SCORE EXPLANATION PANEL ──────────────────────────────────
                if result:
                    _build_score_explanation(sym, result, df_raw)


st.divider()
st.markdown("## ◼ MARKET INTEL")

@st.cache_data(ttl=300)   # Fix 26: 5-min cache — news feeds refresh slowly
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
            col_bt1, col_bt2, col_bt3, col_bt4 = st.columns([2, 1, 1, 1])

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

            with col_bt4:
                bt_slippage_bps = st.number_input(
                    "Slippage (bps)", min_value=0, max_value=100, value=15, key="bt_slippage_bps",
                    help="Basis points added to next-day open entry price. 15bps for liquid F&O, 30-50 for mid-cap."
                )

            run_bt = st.button("▶ Run Backtest", use_container_width=True, key="run_bt")

            if run_bt:
                bt_target = pd.Timestamp(bt_date)

                # FIX B-06: pre-compute Nifty returns AT the backtest date (no look-ahead)
                bt_nifty_r5  = nifty_r5    # fallback to today's values
                bt_nifty_r20 = nifty_r20
                bt_nifty_close_slice = None

                # Fix 14: Cache BT yfinance downloads — re-downloading on every run
                # click adds 10-30s of latency. Cache in session_state with 1h TTL.
                _bt_cache = st.session_state.get("bt_hist_cache", {})
                _bt_cache_ts = st.session_state.get("bt_hist_cache_ts", 0)
                _bt_cache_stale = (time.time() - _bt_cache_ts) > 3600  # 1h TTL
                if _bt_cache_stale:
                    _bt_cache = {}
                    st.session_state.bt_hist_cache_ts = time.time()

                def _bt_yf_get(ticker, period="730d"):
                    if ticker not in _bt_cache:
                        try:
                            _d = yf.download(ticker, period=period, interval="1d", progress=False)
                            if not _d.empty:
                                _s = _d["Close"].squeeze()
                                _s.index = pd.to_datetime(_s.index).tz_localize(None)
                                _bt_cache[ticker] = _s
                                st.session_state.bt_hist_cache = _bt_cache
                        except Exception:
                            pass
                    return _bt_cache.get(ticker)

                try:
                    nc = _bt_yf_get("^NSEI")
                    if nc is not None:
                        nc_slice = nc[nc.index.date <= bt_date]
                        bt_nifty_close_slice = nc_slice
                        if len(nc_slice) >= 21:
                            bt_nifty_r5  = float(nc_slice.iloc[-1] / nc_slice.iloc[-6]  - 1) if len(nc_slice) >= 6  else None
                            bt_nifty_r20 = float(nc_slice.iloc[-1] / nc_slice.iloc[-21] - 1)
                except Exception as _e:
                    st.warning(f"Could not compute historical Nifty RS — using today's values as fallback: {_e}")

                # ── HISTORICAL REGIME at bt_date (same logic as live screener) ──
                # Must use Nifty data sliced to bt_date — never today's regime.
                bt_regime = "BULL"
                if bt_nifty_close_slice is not None and len(bt_nifty_close_slice) >= 50:
                    _nc_g = bt_nifty_close_slice
                    _dma50_bt   = float(_nc_g.tail(50).mean())
                    _dma20_bt   = float(_nc_g.tail(20).mean())
                    _dma20_10d  = float(_nc_g.iloc[-11:-1].mean()) if len(_nc_g) >= 11 else _dma20_bt
                    _slope_bt   = _dma20_bt - _dma20_10d
                    _atr_bt     = float(_nc_g.diff().abs().tail(14).mean())
                    _gap_bt     = float(_nc_g.iloc[-1]) - _dma50_bt
                    if _gap_bt > 0 and _slope_bt > 0:
                        bt_regime = "BULL"
                    elif _gap_bt < -_atr_bt:
                        bt_regime = "BEAR"
                    else:
                        bt_regime = "CHOP"

                # ── HISTORICAL VIX at bt_date — Fix 14: uses cached _bt_yf_get ──
                bt_vix_level   = None; bt_vix_falling = True
                bt_vix_median  = 14.5; bt_vix_sigma   = 4.5
                try:
                    _vc = _bt_yf_get("^INDIAVIX")
                    if _vc is not None:
                        _vc_sl = _vc[_vc.index.date <= bt_date]
                        if len(_vc_sl) >= 5:
                            bt_vix_level  = round(float(_vc_sl.iloc[-1]), 2)
                            _slope_v = float(np.polyfit(range(5), _vc_sl.tail(5).values, 1)[0])
                            bt_vix_falling = _slope_v < 0
                        if len(_vc_sl) >= 20:
                            bt_vix_median = round(float(_vc_sl.median()), 2)
                            bt_vix_sigma  = round(float(_vc_sl.std()), 2)
                except Exception:
                    pass

                # ── HISTORICAL SECTOR RETURNS at bt_date — Fix 14: cached ──
                # Compute sector returns from LOADED STOCKS (not external ETF tickers).
                # ETF tickers (^CNXIT etc.) often fail in yfinance or return Nifty50 data.
                # Using loaded stocks: average return of all stocks in each sector = true sector return.
                bt_sect_ret    = {}; bt_sect_ret_10d = {}
                _sect_r5_accum  = {}   # sector → list of stock 5d returns
                _sect_r10_accum = {}   # sector → list of stock 10d returns
                for _sym_s, _df_s in st.session_state.raw_data_cache.items():
                    _sec_s = get_sector(_sym_s)
                    if _sec_s is None:
                        continue
                    try:
                        _dfs = pd.DataFrame(_df_s).copy()
                        _dfs["time"] = pd.to_datetime(_dfs["time"])
                        _dfs = _dfs.sort_values("time")
                        _dfs_sl = _dfs[_dfs["time"].dt.date <= bt_date]
                        if len(_dfs_sl) >= 6:
                            _c = _dfs_sl["close"]
                            _r5s = float(_c.iloc[-1] / _c.iloc[-6] - 1)
                            _sect_r5_accum.setdefault(_sec_s, []).append(_r5s)
                        if len(_dfs_sl) >= 11:
                            _r10s = float(_c.iloc[-1] / _c.iloc[-11] - 1)
                            _sect_r10_accum.setdefault(_sec_s, []).append(_r10s)
                    except Exception:
                        pass
                for _sn in _sect_r5_accum:
                    bt_sect_ret[_sn]     = float(np.mean(_sect_r5_accum[_sn]))
                for _sn in _sect_r10_accum:
                    bt_sect_ret_10d[_sn] = float(np.mean(_sect_r10_accum[_sn]))
                # Fallback to ETF tickers for sectors with no loaded stocks
                for _sname, _sticker in SECTOR_TICKERS.items():
                    if _sname not in bt_sect_ret:
                        try:
                            _sc2 = _bt_yf_get(_sticker)
                            if _sc2 is not None:
                                _sc_sl = _sc2[_sc2.index.date <= bt_date]
                                if len(_sc_sl) >= 6:
                                    bt_sect_ret[_sname] = float(_sc_sl.iloc[-1] / _sc_sl.iloc[-6] - 1)
                                if len(_sc_sl) >= 11:
                                    bt_sect_ret_10d[_sname] = float(_sc_sl.iloc[-1] / _sc_sl.iloc[-11] - 1)
                        except Exception:
                            pass

                # ── NIFTY ABOVE 20DMA at bt_date ──
                bt_nifty_above_20dma = True
                if bt_nifty_close_slice is not None and len(bt_nifty_close_slice) >= 20:
                    bt_nifty_above_20dma = float(bt_nifty_close_slice.iloc[-1]) > float(bt_nifty_close_slice.tail(20).mean())

                # FIX LOOK-AHEAD: rebuild cs_rs_5d/cs_rs_20d and breadth_cache as they
                # would have been on bt_date.  Without this, the scoring function reads
                # the live screener's cs_rs (computed from today's data), injecting future
                # information into the backtest and producing optimistic scores.
                # SIGNAL INVERSION FIX: CS-RS lookback window matters critically.
                # 5d return on NSE is in the mean-reversion zone — stocks that outperformed
                # last 5 days systematically underperform the next 5 days (short-term reversal).
                # 20d return is in the momentum zone — stocks that outperformed last 20 days
                # tend to continue outperforming the next 5-20 days.
                # Skip-1 on both: end at iloc[-2] not iloc[-1] to avoid signal-day microstructure.
                _bt_cs5  = {}
                _bt_cs20 = {}
                for _bt_sym, _bt_df in st.session_state.raw_data_cache.items():
                    _bt_dfc = _bt_df.copy()
                    _bt_dfc["time"] = pd.to_datetime(_bt_dfc["time"])
                    # Use data strictly BEFORE bt_date — matches df_slice which ends at bar_idx-1
                    _bt_rows = _bt_dfc[_bt_dfc["time"].dt.date < bt_date]
                    if len(_bt_rows) >= 6:
                        # 5d return: iloc[-1] = yesterday close (signal bar), 5 days back
                        _bt_cs5[_bt_sym]  = float(_bt_rows["close"].iloc[-1] / _bt_rows["close"].iloc[-6]  - 1)
                    if len(_bt_rows) >= 21:
                        # 20d return
                        _bt_cs20[_bt_sym] = float(_bt_rows["close"].iloc[-1] / _bt_rows["close"].iloc[-21] - 1)
                if len(_bt_cs5) >= 5:
                    _r5v = np.array(list(_bt_cs5.values()))
                    _bt_cs5_global = {s: float((_r5v <= r).sum() / len(_r5v)) for s, r in _bt_cs5.items()}
                    # Sector-neutral blend
                    _bt_sg5 = {}
                    for _s, _r in _bt_cs5.items():
                        _sec = STOCK_SECTOR_MAP.get(_s.upper())
                        if _sec: _bt_sg5.setdefault(_sec, {})[_s] = _r
                    _bt_sn5 = {}
                    for _sec, _sr in _bt_sg5.items():
                        if len(_sr) < 3: continue
                        _sv = np.array(list(_sr.values()))
                        for _s, _r in _sr.items():
                            _bt_sn5[_s] = float((_sv <= _r).sum() / len(_sv))
                    st.session_state.cs_rs_5d = {
                        s: (_bt_sn5[s]*0.60 + _bt_cs5_global[s]*0.40) if s in _bt_sn5 else _bt_cs5_global[s]
                        for s in _bt_cs5_global
                    }
                else:
                    st.session_state.cs_rs_5d  = {}
                if len(_bt_cs20) >= 5:
                    _r20v = np.array(list(_bt_cs20.values()))
                    _bt_cs20_global = {s: float((_r20v <= r).sum() / len(_r20v)) for s, r in _bt_cs20.items()}
                    _bt_sg20 = {}
                    for _s, _r in _bt_cs20.items():
                        _sec = STOCK_SECTOR_MAP.get(_s.upper())
                        if _sec: _bt_sg20.setdefault(_sec, {})[_s] = _r
                    _bt_sn20 = {}
                    for _sec, _sr in _bt_sg20.items():
                        if len(_sr) < 3: continue
                        _sv = np.array(list(_sr.values()))
                        for _s, _r in _sr.items():
                            _bt_sn20[_s] = float((_sv <= _r).sum() / len(_sv))
                    st.session_state.cs_rs_20d = {
                        s: (_bt_sn20[s]*0.60 + _bt_cs20_global[s]*0.40) if s in _bt_sn20 else _bt_cs20_global[s]
                        for s in _bt_cs20_global
                    }
                else:
                    st.session_state.cs_rs_20d = {}
                # Breadth: fraction of stocks above their EMA20 on bt_date
                _bt_above = 0; _bt_total = 0
                for _bt_sym, _bt_df in st.session_state.raw_data_cache.items():
                    _bt_dfc = _bt_df.copy()
                    _bt_dfc["time"] = pd.to_datetime(_bt_dfc["time"])
                    _bt_rows = _bt_dfc[_bt_dfc["time"].dt.date < bt_date]   # < not <= matches scoring bar
                    if len(_bt_rows) >= 20:
                        _bt_e20  = float(_bt_rows["close"].ewm(span=20, adjust=False).mean().iloc[-1])
                        _bt_ltp  = float(_bt_rows["close"].iloc[-1])
                        _bt_total += 1
                        if _bt_ltp > _bt_e20:
                            _bt_above += 1
                st.session_state.breadth_cache = (_bt_above / _bt_total) if _bt_total >= 10 else None

                # ── REBUILD ALL CROSS-SECTIONAL FACTORS AT bt_date (NO LOOK-AHEAD) ──
                # BB squeeze, VCP, CLV, VolDryUp are read from session_state inside
                # score_stock_dual. Without rebuilding them at the historical date,
                # the scoring function reads TODAY's live universe ranks — injecting
                # future information and collapsing the score spread (IC → 0).
                # Fix: compute each factor on the historical slice for every stock,
                # cross-sectionally rank them, and store in session_state.
                # This mirrors exactly what the live screener does before its scoring loop.
                _bt_bb_raw  = {}
                _bt_vdu_raw = {}
                _bt_clv_raw = {}
                _bt_vcp_raw = {}

                for _bt_sym, _bt_df in st.session_state.raw_data_cache.items():
                    try:
                        _bt_dfc = _bt_df.copy()
                        _bt_dfc["time"] = pd.to_datetime(_bt_dfc["time"])
                        # Historical slice ending at signal_bar (day BEFORE bt_date entry)
                        _bt_rows = _bt_dfc[_bt_dfc["time"].dt.date < bt_date].copy()
                        if len(_bt_rows) < 60:
                            continue
                        _sl_c = _bt_rows["close"].reset_index(drop=True)
                        _sl_h = _bt_rows["high"].reset_index(drop=True)
                        _sl_l = _bt_rows["low"].reset_index(drop=True)
                        _sl_v = _bt_rows["volume"].replace(0, np.nan).reset_index(drop=True)

                        # BB Width squeeze
                        _, _bbs = bb_width_compression_score(_sl_c)
                        _bt_bb_raw[_bt_sym] = _bbs

                        # Volume dry-up
                        _, _vdu = volume_dryup_score(_sl_v)
                        _bt_vdu_raw[_bt_sym] = _vdu

                        # CLV accumulation
                        _, _clv = clv_accumulation_score(_sl_c, _sl_h, _sl_l, _sl_v)
                        _bt_clv_raw[_bt_sym] = _clv

                        # VCP — needs ATR series
                        _sl_tr = pd.concat([
                            _sl_h - _sl_l,
                            (_sl_h - _sl_c.shift(1)).abs(),
                            (_sl_l - _sl_c.shift(1)).abs()
                        ], axis=1).max(axis=1)
                        _sl_atr = _sl_tr.rolling(14).mean()
                        _vcp_r = detect_vcp(_sl_c, _sl_h, _sl_l, _sl_v, _sl_atr)
                        _bt_vcp_raw[_bt_sym] = _vcp_r["vcp_score"]
                    except Exception:
                        pass

                # Cross-sectional rank all four at bt_date
                def _bt_cs_rank(raw_dict):
                    if len(raw_dict) < 5:
                        return {k: 0.5 for k in raw_dict}
                    vals = np.array(list(raw_dict.values()), dtype=float)
                    return {s: float((vals <= v).sum() / len(vals)) for s, v in raw_dict.items()}

                st.session_state.cs_bb_squeeze = _bt_cs_rank(_bt_bb_raw)
                st.session_state.cs_vol_dryup  = _bt_cs_rank(_bt_vdu_raw)
                st.session_state.cs_clv_accum  = _bt_cs_rank(_bt_clv_raw)
                st.session_state.cs_vcp        = _bt_cs_rank(_bt_vcp_raw)

                st.info(
                    f"📅 **Backtest context on {bt_date}:**  "
                    f"Regime = **{bt_regime}**  ·  "
                    f"Nifty 5d RS = {round(bt_nifty_r5*100,2) if bt_nifty_r5 is not None else '?'}%  ·  "
                    f"VIX = {bt_vix_level if bt_vix_level is not None else '?'} "
                    f"({'falling' if bt_vix_falling else 'rising'})  ·  "
                    f"Sectors tracked = {len(bt_sect_ret)}  ·  "
                    f"Entry = **next-day open** (no same-day close slippage)"
                )

                # ── STEP 1: slice each stock to backtest date ──
                bt_signals   = []
                bt_signals_full = []   # all scored stocks, no minscore filter — for quintile analysis
                skipped      = 0
                # Clear any stale quintile data from prior runs
                st.session_state.bt_signals_full = pd.DataFrame()

                # Safety: zero out live CS factors so no stock accidentally reads
                # today's rank during bt scoring. Each stock's inline fallback
                # will recompute from its own historical slice.
                # (The loop above already overwrites these, but if it failed for all
                # stocks the live ranks would remain — this guarantees clean state.)
                _bt_live_cs_backup = {
                    "cs_bb_squeeze": dict(st.session_state.get("cs_bb_squeeze", {})),
                    "cs_vol_dryup":  dict(st.session_state.get("cs_vol_dryup",  {})),
                    "cs_clv_accum":  dict(st.session_state.get("cs_clv_accum",  {})),
                    "cs_vcp":        dict(st.session_state.get("cs_vcp",        {})),
                }

                progress_bt  = st.progress(0)
                syms         = list(st.session_state.raw_data_cache.keys())

                # BUG-2 FIX: one clean rs_div_hist dict per test date — never reads
                # session_state history from prior runs or from the live screener.
                _bt_date_div_hist = {}

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

                    # need at least 60 bars before AND 6 bars after (5 fwd + 1 next-open)
                    if bar_idx < 61 or bar_idx + 6 >= len(df_full):
                        skipped += 1
                        progress_bt.progress((idx+1)/len(syms))
                        continue

                    # BACKTEST SLICE FIX: score on bt_date EOD, enter bt_date+1 open.
                    #
                    # Previous (broken): signal_bar_idx = bar_idx-1, df_slice ends at bar_idx-1.
                    #   Inside scorer hist = df[:-1] = bar_idx-2. ltp_score = T-2 close.
                    #   Entry = bar_idx open.  Gap = 2 days of unseen price action → inversion.
                    #
                    # Fixed: signal_bar_idx = bar_idx (bt_date itself), df_slice ends at bar_idx.
                    #   Inside scorer hist = df[:-1] = bar_idx-1 (yesterday). ltp_score = T-1 close.
                    #   Entry = bar_idx+1 open (next trading day after signal date).
                    #   This mirrors live workflow: screen at EOD, buy next morning.
                    signal_bar_idx = bar_idx        # bt_date = signal bar (EOD screen)
                    entry_bar_idx  = bar_idx + 1    # next day open = entry

                    df_slice = df_full.iloc[:signal_bar_idx + 1].copy()

                    bar = df_full.iloc[signal_bar_idx]
                    fake_live = {
                        "ltp":    float(bar["close"]),
                        "open":   float(bar["open"]),
                        "high":   float(bar["high"]),
                        "low":    float(bar["low"]),
                        "volume": float(bar["volume"]),
                        "oi":     float(bar["oi"]) if "oi" in bar.index and pd.notna(bar["oi"]) else 0.0,
                    }

                    result = score_stock_dual(
                        df_slice, fake_live, bt_nifty_r5, bt_nifty_r20, ticker=sym,
                        bt_mode=True,
                        bt_regime=bt_regime,
                        bt_sector_returns=bt_sect_ret,
                        bt_sector_returns_10d=bt_sect_ret_10d,
                        bt_vix_level=bt_vix_level,
                        bt_vix_falling=bt_vix_falling,
                        bt_vix_median=bt_vix_median,
                        bt_vix_sigma=bt_vix_sigma,
                        bt_nifty_above_20dma=bt_nifty_above_20dma,
                        bt_rs_div_hist=_bt_date_div_hist,
                    )

                    if result is None:
                        skipped += 1
                        progress_bt.progress((idx+1)/len(syms))
                        continue

                    # ── STEP 2: measure forward returns from today's open ──
                    entry_price = float(df_full.iloc[entry_bar_idx]["open"])
                    if entry_price == 0:
                        skipped += 1
                        progress_bt.progress((idx+1)/len(syms))
                        continue
                    entry_price = entry_price * (1.0 + bt_slippage_bps / 10000.0)

                    def fwd_return(n):
                        fwd_idx = entry_bar_idx + n
                        if fwd_idx >= len(df_full):
                            return None
                        return round((df_full.iloc[fwd_idx]["close"] - entry_price) / entry_price * 100, 2)

                    r1 = fwd_return(1)
                    r2 = fwd_return(2)
                    r3 = fwd_return(3)
                    r5 = fwd_return(5)
                    fwd_window = df_full.iloc[entry_bar_idx : entry_bar_idx + 6]["low"]
                    max_dd = round((fwd_window.min() - entry_price) / entry_price * 100, 2) if len(fwd_window) > 0 else None
                    fwd_high   = df_full.iloc[entry_bar_idx : entry_bar_idx + 6]["high"]
                    max_gain   = round((fwd_high.max() - entry_price) / entry_price * 100, 2) if len(fwd_high) > 0 else None

                    _signal_row = {
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
                        "VWMA20_OK": result["VWMA20_OK"],
                        "Stability": result["Stability"],
                        "UpVolSkew":    result.get("UpVolSkew", 0),
                        "CPR":          result.get("CPR", 0),
                        "SpreadComp":   result.get("SpreadComp", 0),
                        "ATRExpOnset":  result.get("ATRExpOnset", 0),
                        "OI_Buildup":   result.get("OI_Buildup", 0),
                        "VolVelocity":  result.get("VolVelocity", 0),
                        "RSDivergence": result.get("RSDivergence", 0),
                        # NEW PREDICTIVE FACTORS
                        "BBSqueeze":    result.get("BBSqueeze", 0),
                        "VolDryUp":     result.get("VolDryUp", 0),
                        "CLVAccum":     result.get("CLVAccum", 0),
                        "VCP":          result.get("VCP", 0),
                        "BreakoutProb": result.get("BreakoutProb", 0),
                        "VCP_Detected": result.get("VCP_Detected", False),
                        # FACTOR BREAKDOWN for diagnosis
                        "F_VCP":     result.get("VCP", 0),
                        "F_BB":      result.get("BBSqueeze", 0),
                        "F_VDU":     result.get("VolDryUp", 0),
                        "F_VC":      result.get("VolCont", 0),
                        "F_CLV":     result.get("CLVAccum", 0),
                        "F_RS":      result.get("RS", 0),
                        "F_Coil":    result.get("Coil", 0),
                        "F_Prox":    result.get("Proximity", 0),
                        "F_MA":      result.get("MA_Struct", 0),
                        "CSRank5d":  result.get("CSRank5d", 0),
                    }

                    # Store in full list for quintile analysis (no minscore filter)
                    bt_signals_full.append(_signal_row)

                    # Apply minscore filter for the main results table only
                    if result["Score"] >= bt_minscore:
                        bt_signals.append(_signal_row)

                    progress_bt.progress((idx+1)/len(syms))

                progress_bt.empty()

                # Restore live CS factors — backtest temporarily overwrites them
                # so the live screener tab continues to work correctly after BT.
                for _k, _v in _bt_live_cs_backup.items():
                    st.session_state[_k] = _v

                # Store full unfiltered signal set for quintile analysis
                _SCORING_VERSION = "v7_ic_derived_weights"
                _bt_full_df = pd.DataFrame(bt_signals_full) if bt_signals_full else pd.DataFrame()

                # Cross-sectional percentile rank: replace raw Score with within-universe rank.
                # This makes Reversal scores (typically 10-50) compete fairly with
                # Breakout/Pullback scores (typically 45-75) on the same 0-100 scale.
                # A perfect Reversal setup gets the same percentile as a perfect Breakout.
                if not _bt_full_df.empty and "Score" in _bt_full_df.columns:
                    _raw_scores = _bt_full_df["Score"].values.astype(float)
                    _cs_pct = np.array([
                        float((_raw_scores <= s).sum() / len(_raw_scores)) * 100
                        for s in _raw_scores
                    ])
                    _bt_full_df["Score"] = np.round(_cs_pct, 1)

                st.session_state.bt_signals_full = _bt_full_df
                st.session_state.bt_signals_version = _SCORING_VERSION

                if not bt_signals:
                    st.warning(f"No signals found on {bt_date} with score ≥ {bt_minscore}")
                else:
                    bt_df = pd.DataFrame(bt_signals).sort_values("CompositeRank", ascending=False).reset_index(drop=True)
                    if "Rank" in bt_df.columns:
                        bt_df.drop(columns=["Rank"], inplace=True)
                    bt_df.insert(0, "Rank", bt_df.index + 1)

                    # limit to top N
                    bt_top = bt_df.head(bt_topn)

                    st.success(f"✅ {len(bt_df)} signals on {bt_date} | showing top {bt_topn} | {skipped} stocks skipped (insufficient data) | entry = next-day open")

                    # ── INFORMATION COEFFICIENT (IC) ──
                    # IC = rank correlation between signal Score and forward return.
                    # Interpretation: IC > 0.05 = signal has edge, IC > 0.10 = strong edge.
                    # Using Spearman rank correlation (robust to outliers).
                    # Single-date IC is noisy — check Walk-Forward for distribution.
                    _ic_data = bt_df.dropna(subset=["R5d%", "Score"])
                    if len(_ic_data) >= 5:
                        try:
                            from scipy import stats as _scipy_stats
                        except ImportError:
                            st.info("ℹ️ IC computation requires scipy: `pip install scipy`")
                            _scipy_stats = None
                        if _scipy_stats is not None:
                            _ic_val, _ic_pval = _scipy_stats.spearmanr(_ic_data["Score"], _ic_data["R5d%"])
                        _ic_val  = round(float(_ic_val),  3)
                        _ic_pval = round(float(_ic_pval), 3)
                        if _ic_val > 0.10:
                            st.success(f"IC (Score→5d Return): **{_ic_val}** (p={_ic_pval}) — strong signal on this date")
                        elif _ic_val > 0.05:
                            st.info(f"IC: **{_ic_val}** (p={_ic_pval}) — meaningful positive signal")
                        elif _ic_val > 0:
                            st.info(f"IC: **{_ic_val}** (p={_ic_pval}) — weak positive, single date is noisy")
                        else:
                            st.warning(f"IC: **{_ic_val}** (p={_ic_pval}) — no discrimination on this date. Run Walk-Forward for a reliable picture.")

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

                        for stype, color in [("Breakout","#2196F3"),("Pullback","#FF9800"),("Reversal","#FF5722")]:
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

                    # CRITICAL: quintile analysis requires the FULL score distribution.
                    # bt_minscore > 0 truncates the bottom of the distribution —
                    # "Q1 Lowest" then represents the 40th percentile, not the 0th.
                    # A truncated quintile can invert even when the signal has real edge.
                    # Solution: re-score ALL stocks with minscore=0 for this analysis.
                    # Use bt_signals_full (all scored stocks, no minscore filter)
                    # This was populated at the start of this run (stale cleared before scoring loop)
                    _q_source = st.session_state.get("bt_signals_full", pd.DataFrame())
                    if not _q_source.empty and "R5d%" in _q_source.columns:
                        valid_q = _q_source.dropna(subset=["R5d%", "Score"]).copy()
                    else:
                        valid_q = bt_df.dropna(subset=["R5d%"]).copy()   # fallback only

                    if len(valid_q) >= 10:
                        valid_q["Quintile"] = pd.qcut(valid_q["Score"], 5,
                                                      labels=["Q1 (Lowest)","Q2","Q3","Q4","Q5 (Highest)"],
                                                      duplicates="drop")
                        qt = (valid_q.groupby("Quintile", observed=True)["R5d%"]
                              .agg(Count="count", AvgReturn="mean", WinRate=lambda x: round((x>0).mean()*100, 1))
                              .reset_index())
                        qt.columns = ["Score Quintile","Count","Avg 5d Return %","Win Rate %"]
                        qt["Avg 5d Return %"] = qt["Avg 5d Return %"].round(3)

                        def qt_ret_color(val):
                            if not isinstance(val, (int, float)) or pd.isna(val): return ""
                            if val > 1.0:  return "background-color:#0d2200;color:#00d084;font-weight:700"
                            if val > 0.0:  return "background-color:#0a1a00;color:#7ec87a"
                            return "background-color:#1f0000;color:#ff3b3b"

                        def qt_wr_color(val):
                            # Win rate: green above 55%, red below 45%
                            if not isinstance(val, (int, float)) or pd.isna(val): return ""
                            if val > 55: return "background-color:#0d2200;color:#00d084;font-weight:700"
                            if val >= 45: return "background-color:#1a1a00;color:#cccc44"
                            return "background-color:#1f0000;color:#ff3b3b"

                        styled_qt = (qt.style
                                     .applymap(qt_ret_color, subset=["Avg 5d Return %"])
                                     .applymap(qt_wr_color,  subset=["Win Rate %"]))
                        st.dataframe(styled_qt, use_container_width=True, hide_index=True)

                        # Compute spread — positive spread = signal has edge
                        _q5_ret = qt[qt["Score Quintile"] == "Q5 (Highest)"]["Avg 5d Return %"].values
                        _q1_ret = qt[qt["Score Quintile"] == "Q1 (Lowest)"]["Avg 5d Return %"].values
                        if len(_q5_ret) and len(_q1_ret):
                            _spread = round(float(_q5_ret[0]) - float(_q1_ret[0]), 3)
                            if _spread > 0:
                                st.success(f"✅ Q5−Q1 spread = +{_spread}% — score has positive edge on this date")
                            else:
                                st.error(f"❌ Q5−Q1 spread = {_spread}% — score is inverted on this date")
                        st.caption(f"Quintile analysis on {len(valid_q)} stocks (full distribution, minscore=0)")
                    else:
                        st.info("Need at least 10 signals for quintile analysis")

                    # ── TOP 10 vs BOTTOM 10 COMPARISON ──
                    st.subheader("⚖️ Top 10 vs Bottom 10 Signals — Head to Head")

                    # Use bt_signals_full (full distribution, no minscore filter)
                    # sorted by Score — same universe as the quintile table above.
                    # Old code used bt_df sorted by CompositeRank:
                    #   • bt_df filters by minscore → bottom quintile is missing
                    #   • CompositeRank = EMI × 0.60 + liquidity × 0.25 + stability × 0.15
                    #     where EMI = Score × ATR%. This rewards HIGH-ATR stocks regardless
                    #     of signal quality. "Top 10 by CompositeRank" = most volatile stocks,
                    #     not best signals. Comparison was volatility-biased, not score-based.
                    _full_df = st.session_state.get("bt_signals_full", pd.DataFrame())
                    if _full_df.empty:
                        _full_df = bt_df.copy()
                    valid_cmp = _full_df.dropna(subset=["R5d%", "Score"]).copy()
                    valid_cmp = valid_cmp.sort_values("Score", ascending=False).reset_index(drop=True)

                    if len(valid_cmp) >= 6:
                        _n = min(10, len(valid_cmp) // 2)
                        top10_cmp    = valid_cmp.head(_n)    # highest Score stocks
                        bottom10_cmp = valid_cmp.tail(_n)    # lowest Score stocks

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
# WALK-FORWARD BACKTEST
# ============================================================
# Runs the screener across every available date in the cache,
# not just one cherry-picked date. This is the only way to know
# if the signal has real edge across different market regimes.
#
# Method:
#   1. Step through every valid trading date in the cache
#      (spaced by wf_step days so we're not reusing the same trade window)
#   2. On each date, score all stocks → take top-N by CompositeRank
#   3. Measure forward 5-day return for each selected stock
#   4. Aggregate: win rate, avg return, Sharpe, drawdown across ALL dates
#   5. Show equity curve of equal-weight portfolio over the full period
#
# No look-ahead: each date-slice uses only data available on that date.
# Slippage model: entry at next-day open (not same-day close).
# ============================================================

st.divider()
st.header("📐 Walk-Forward Validation — Full History")
st.caption(
    "Runs the screener across ALL dates in the cache. "
    "One date tells you nothing — this tells you if the signal survives different regimes."
)
st.info(
    "⚠️ **Universe size matters for cross-sectional RS.** "
    "Running on Nifty 50 (48 stocks) gives only ~12 stocks per quartile — too few for reliable discrimination. "
    "For meaningful walk-forward results, run extraction on **F&O Stocks (400+)** or **Full NSE** first."
)

if not st.session_state.raw_data_cache:
    st.info("Run extraction first — walk-forward needs historical data in cache")
else:
    _wf_all_dates = set()
    for _wf_sym, _wf_df in st.session_state.raw_data_cache.items():
        if "time" in _wf_df.columns:
            for _d in pd.to_datetime(_wf_df["time"]).dt.date:
                _wf_all_dates.add(_d)
    _wf_all_dates = sorted(_wf_all_dates)

    # Need enough bars: 60 lookback + 5 forward on each end
    if len(_wf_all_dates) < 80:
        st.warning("Need at least 80 trading dates in cache for walk-forward (run extraction with 600d history).")
    else:
        _wf_valid = _wf_all_dates[60:-5]

        wf_col1, wf_col2, wf_col3, wf_col4, wf_col5 = st.columns(5)
        with wf_col1:
            wf_step = st.number_input(
                "Step size (trading days between test dates)",
                min_value=1, max_value=20, value=5, key="wf_step",
                help="5 = test every 5th trading day. Lower = more samples but overlapping trade windows."
            )
        with wf_col2:
            wf_topn = st.number_input(
                "Top N per date", min_value=1, max_value=20, value=5, key="wf_topn",
                help="How many top-ranked stocks to 'buy' on each test date."
            )
        with wf_col3:
            wf_minscore = st.number_input(
                "Min score filter", min_value=0, max_value=80, value=40, key="wf_minscore"
            )
        with wf_col4:
            wf_hold = st.selectbox(
                "Hold period", options=[1, 2, 3, 5], index=2, key="wf_hold",
                format_func=lambda x: f"{x} days"
            )
        with wf_col5:
            wf_slippage_bps = st.number_input(
                "Slippage (bps)", min_value=0, max_value=100, value=15, key="wf_slippage_bps",
                help="Basis points added to next-day open as entry cost. 15bps = 0.15% for liquid F&O stocks. 30-50bps for mid-cap."
            )

        # ── OOS BOUNDARY — calendar-time anchor ──────────────────────────────
        # Default: last 20% of calendar time is automatically used as OOS holdout.
        # Override: user can pin an exact date so the boundary is stable across
        # multiple WF runs (prevents boundary drift as more data arrives).
        st.caption("OOS HOLDOUT BOUNDARY")
        _oos_col1, _oos_col2 = st.columns([2, 3])
        with _oos_col1:
            _oos_auto = st.checkbox(
                "Auto (last 20% of date range)", value=True, key="wf_oos_auto",
                help="When ON, OOS = last 20% of calendar time. Turn OFF to set a fixed date."
            )
        with _oos_col2:
            if not _oos_auto:
                _oos_date_min = _wf_valid[int(len(_wf_valid) * 0.5)] if len(_wf_valid) > 10 else _wf_valid[0]
                _oos_date_max = _wf_valid[-1]
                _oos_default  = _wf_valid[int(len(_wf_valid) * 0.8)]
                _oos_sel = st.date_input(
                    "OOS start date (fixed)",
                    value=_oos_default,
                    min_value=_oos_date_min,
                    max_value=_oos_date_max,
                    key="wf_oos_date_input",
                    help="All trades from this date onward are the OOS set. IC measured here."
                )
                st.session_state["wf_oos_start_date"] = str(_oos_sel)
            else:
                st.session_state.pop("wf_oos_start_date", None)
                st.caption(
                    "OOS boundary = auto (last 20% of date range). "
                    "Uncheck to pin a specific date."
                )

        run_wf = st.button("▶ Run Walk-Forward", use_container_width=True, key="run_wf")

        if run_wf:
            # ── Fetch full Nifty history once for all test dates ──
            _wf_nifty_close = None
            try:
                _wf_nifty_raw = yf.download("^NSEI", period="730d", interval="1d", progress=False)
                if not _wf_nifty_raw.empty:
                    _wf_nc = _wf_nifty_raw["Close"].squeeze()
                    _wf_nc.index = pd.to_datetime(_wf_nc.index).tz_localize(None)
                    _wf_nifty_close = _wf_nc
            except Exception:
                pass

            # ── Fetch full VIX history once for all test dates ──
            _wf_vix_close = None
            try:
                _wf_vix_raw = yf.download("^INDIAVIX", period="730d", interval="1d", progress=False)
                if not _wf_vix_raw.empty:
                    _wf_vc = _wf_vix_raw["Close"].squeeze()
                    _wf_vc.index = pd.to_datetime(_wf_vc.index).tz_localize(None)
                    _wf_vix_close = _wf_vc
            except Exception:
                pass

            # ── Fetch full sector history once for all test dates ──
            # Fix 13: After fetching, pre-build date-indexed value arrays so the inner
            # date loop can do O(1) lookups instead of O(N) boolean slice per sector per date.
            _wf_sect_closes = {}
            _wf_sect_idx    = {}   # {sname: sorted array of (date, close_value)}
            for _sname, _sticker in SECTOR_TICKERS.items():
                try:
                    _sd = yf.download(_sticker, period="730d", interval="1d", progress=False)
                    if not _sd.empty:
                        _sc2 = _sd["Close"].squeeze()
                        _sc2.index = pd.to_datetime(_sc2.index).tz_localize(None)
                        _wf_sect_closes[_sname] = _sc2
                        # Pre-index: sorted list of (date, value) for bisect lookup
                        import bisect
                        _dates_arr = [d.date() for d in _sc2.index]
                        _vals_arr  = _sc2.values.tolist()
                        _wf_sect_idx[_sname] = (_dates_arr, _vals_arr)
                except Exception:
                    pass

            # ── Build per-stock DataFrames indexed by date for fast slicing ──
            _wf_sym_dfs = {}
            for _sym, _df in st.session_state.raw_data_cache.items():
                _dfc = _df.copy()
                _dfc["time"] = pd.to_datetime(_dfc["time"])
                _dfc = _dfc.sort_values("time").reset_index(drop=True)
                _wf_sym_dfs[_sym] = _dfc

            # ── Pre-compute EMA20 series for each stock (outside date loop) ──
            # This avoids recomputing EMA20 for every stock on every test date (O(N×M)).
            # Instead: one EMA20 computation per stock, then O(1) lookup per date.
            _wf_ema20 = {}
            for _sym, _dfc in _wf_sym_dfs.items():
                try:
                    _e20s = _dfc["close"].ewm(span=20, adjust=False).mean()
                    _wf_ema20[_sym] = pd.Series(_e20s.values, index=_dfc["time"])
                except Exception:
                    pass

            # ── Select test dates spaced by wf_step ──
            test_dates = _wf_valid[::wf_step]

            wf_progress = st.progress(0)
            wf_all_trades = []        # one row per (date, stock) "trade" — confirmation filtered
            wf_all_signals_raw = []   # all signals BEFORE confirmation filter — for unbiased IC

            for _di, _test_date in enumerate(test_dates):
                wf_progress.progress((_di + 1) / len(test_dates))

                # Nifty RS at this date (no look-ahead)
                _bt_r5, _bt_r20 = None, None
                if _wf_nifty_close is not None:
                    _nc_slice = _wf_nifty_close[_wf_nifty_close.index.date <= _test_date]
                    if len(_nc_slice) >= 21:
                        _bt_r5  = float(_nc_slice.iloc[-1] / _nc_slice.iloc[-6]  - 1) if len(_nc_slice) >= 6 else None
                        _bt_r20 = float(_nc_slice.iloc[-1] / _nc_slice.iloc[-21] - 1)

                # ── VIX at this date ──
                _wf_vix_level  = None; _wf_vix_falling = True
                _wf_vix_median = 14.5; _wf_vix_sigma   = 4.5
                if _wf_vix_close is not None:
                    _vc_sl = _wf_vix_close[_wf_vix_close.index.date <= _test_date]
                    if len(_vc_sl) >= 5:
                        _wf_vix_level = round(float(_vc_sl.iloc[-1]), 2)
                        _vslope = float(np.polyfit(range(5), _vc_sl.tail(5).values, 1)[0])
                        _wf_vix_falling = _vslope < 0
                    if len(_vc_sl) >= 20:
                        _wf_vix_median = round(float(_vc_sl.median()), 2)
                        _wf_vix_sigma  = round(float(_vc_sl.std()), 2)

                # ── Sector returns at this date — O(1) bisect lookup (Fix 13) ──
                _wf_sect_ret    = {}; _wf_sect_ret_10d = {}
                import bisect as _bisect
                for _sname, (_dates_a, _vals_a) in _wf_sect_idx.items():
                    _pos = _bisect.bisect_right(_dates_a, _test_date) - 1
                    if _pos >= 5:
                        _wf_sect_ret[_sname]    = float(_vals_a[_pos] / _vals_a[_pos - 5]  - 1)
                    if _pos >= 10:
                        _wf_sect_ret_10d[_sname] = float(_vals_a[_pos] / _vals_a[_pos - 10] - 1)

                # ── Nifty above 20DMA at this date ──
                _wf_above_20dma = True
                if _wf_nifty_close is not None:
                    _nc_sl_20 = _wf_nifty_close[_wf_nifty_close.index.date <= _test_date]
                    if len(_nc_sl_20) >= 20:
                        _wf_above_20dma = float(_nc_sl_20.iloc[-1]) > float(_nc_sl_20.tail(20).mean())
                # Same logic as live screener: rank each stock's return within the universe.
                # This is what makes the score regime-agnostic in the backtest too.
                _wf_cs5  = {}
                _wf_cs20 = {}
                for _sym, _dfc in _wf_sym_dfs.items():
                    _rows = _dfc[_dfc["time"].dt.date <= _test_date]
                    if len(_rows) >= 7:
                        # Skip-1: consistent with BT and live screener
                        _wf_cs5[_sym]  = float(_rows["close"].iloc[-2] / _rows["close"].iloc[-7]  - 1)
                    if len(_rows) >= 22:
                        _wf_cs20[_sym] = float(_rows["close"].iloc[-2] / _rows["close"].iloc[-22] - 1)
                if len(_wf_cs5) >= 5:
                    _r5v = np.array(list(_wf_cs5.values()))
                    _wf_gl5 = {s: float((_r5v <= r).sum() / len(_r5v)) for s, r in _wf_cs5.items()}
                    _wf_sg5 = {}
                    for _s, _r in _wf_cs5.items():
                        _sec = STOCK_SECTOR_MAP.get(_s.upper())
                        if _sec: _wf_sg5.setdefault(_sec, {})[_s] = _r
                    _wf_sn5 = {}
                    for _sec, _sr in _wf_sg5.items():
                        if len(_sr) < 3: continue
                        _sv = np.array(list(_sr.values()))
                        for _s, _r in _sr.items():
                            _wf_sn5[_s] = float((_sv <= _r).sum() / len(_sv))
                    st.session_state.cs_rs_5d = {
                        s: (_wf_sn5[s]*0.60 + _wf_gl5[s]*0.40) if s in _wf_sn5 else _wf_gl5[s]
                        for s in _wf_gl5
                    }
                else:
                    st.session_state.cs_rs_5d  = {}
                if len(_wf_cs20) >= 5:
                    _r20v = np.array(list(_wf_cs20.values()))
                    _wf_gl20 = {s: float((_r20v <= r).sum() / len(_r20v)) for s, r in _wf_cs20.items()}
                    _wf_sg20 = {}
                    for _s, _r in _wf_cs20.items():
                        _sec = STOCK_SECTOR_MAP.get(_s.upper())
                        if _sec: _wf_sg20.setdefault(_sec, {})[_s] = _r
                    _wf_sn20 = {}
                    for _sec, _sr in _wf_sg20.items():
                        if len(_sr) < 3: continue
                        _sv = np.array(list(_sr.values()))
                        for _s, _r in _sr.items():
                            _wf_sn20[_s] = float((_sv <= _r).sum() / len(_sv))
                    st.session_state.cs_rs_20d = {
                        s: (_wf_sn20[s]*0.60 + _wf_gl20[s]*0.40) if s in _wf_sn20 else _wf_gl20[s]
                        for s in _wf_gl20
                    }
                else:
                    st.session_state.cs_rs_20d = {}

                # True breadth: fraction of stocks above their EMA20 on this test date.
                # Old: used CS-RS > 0.5 proxy which is ALWAYS exactly 50% by construction
                # (it's a percentile rank, so half are always above 0.5 — completely uninformative).
                # New: compute actual above-EMA20 count from the historical slices.
                _wf_ema_above = 0; _wf_ema_total = 0
                for _bsym, _bdf in _wf_sym_dfs.items():
                    _brows = _bdf[_bdf["time"].dt.date <= _test_date]
                    if len(_brows) >= 20:
                        _be20 = float(_brows["close"].ewm(span=20, adjust=False).mean().iloc[-1])
                        _bltp = float(_brows["close"].iloc[-1])
                        _wf_ema_total += 1
                        if _bltp > _be20:
                            _wf_ema_above += 1
                st.session_state.breadth_cache = (_wf_ema_above / _wf_ema_total) \
                    if _wf_ema_total >= 10 else None

                # ── HISTORICAL REGIME GATE ──
                # Compute Nifty regime at this exact date (same logic as live screener).
                # Skip BEAR dates entirely — the walk-forward showed they destroy returns.
                # On CHOP dates, raise score floor to 75th percentile of that date's signals.
                _wf_regime = "BULL"   # default
                if _wf_nifty_close is not None:
                    _nc_gate = _wf_nifty_close[_wf_nifty_close.index.date <= _test_date]
                    if len(_nc_gate) >= 50:
                        _dma50_gate   = float(_nc_gate.tail(50).mean())
                        _dma20_now_g  = float(_nc_gate.tail(20).mean())
                        _dma20_10d_g  = float(_nc_gate.iloc[-11:-1].mean())
                        _slope_gate   = _dma20_now_g - _dma20_10d_g
                        _atr_gate     = float(_nc_gate.diff().abs().tail(14).mean())
                        _gap_gate     = float(_nc_gate.iloc[-1]) - _dma50_gate
                        if _gap_gate > 0 and _slope_gate > 0:
                            _wf_regime = "BULL"
                        elif _gap_gate < -_atr_gate:
                            _wf_regime = "BEAR"
                        else:
                            _wf_regime = "CHOP"

                # Score all stocks at this date
                # Regime is recorded in the trade row for analysis, but we never skip dates.
                # The cross-sectional RS in F1 naturally surfaces leaders in any environment.
                # Reset rs_div_hist for this test date to prevent temporal leakage:
                # RS divergence history from a prior test date must not influence this date's scores.
                # BUG-2 FIX: use a plain dict, never session_state, in bt_mode
                _wf_date_div_hist = {}
                _date_signals = []
                for _sym, _dfc in _wf_sym_dfs.items():
                    _date_rows = _dfc[_dfc["time"].dt.date == _test_date]
                    if len(_date_rows) == 0:
                        continue
                    _bar_idx = int(_date_rows.index[0])
                    if _bar_idx < 61 or _bar_idx + 1 + 5 >= len(_dfc):  # FIX-02: always need 5 fwd bars for R1/R3/R5d
                        continue
                    # Same fix as single-date BT: score on signal_bar (yesterday), enter today
                    # WALK-FORWARD SLICE FIX: same off-by-one corrected as single-date BT.
                    # score on _bar_idx EOD, enter _bar_idx+1 open.
                    _signal_bar_idx = _bar_idx        # signal date itself
                    _slice = _dfc.iloc[:_signal_bar_idx + 1].copy()
                    _bar   = _dfc.iloc[_signal_bar_idx]
                    _fake_live = {
                        "ltp":    float(_bar["close"]),
                        "open":   float(_bar["open"]),
                        "high":   float(_bar["high"]),
                        "low":    float(_bar["low"]),
                        "volume": float(_bar["volume"]),
                        "oi":     float(_bar["oi"]) if "oi" in _bar.index and pd.notna(_bar["oi"]) else 0.0,
                    }
                    try:
                        _res = score_stock_dual(
                            _slice, _fake_live, _bt_r5, _bt_r20, ticker=_sym,
                            bt_mode=True,
                            bt_regime=_wf_regime,
                            bt_sector_returns=_wf_sect_ret,
                            bt_sector_returns_10d=_wf_sect_ret_10d,
                            bt_vix_level=_wf_vix_level,
                            bt_vix_falling=_wf_vix_falling,
                            bt_vix_median=_wf_vix_median,
                            bt_vix_sigma=_wf_vix_sigma,
                            bt_nifty_above_20dma=_wf_above_20dma,
                            bt_rs_div_hist=_wf_date_div_hist,
                        )
                    except Exception:
                        continue
                    if _res is None or _res["Score"] < wf_minscore:
                        continue
                    _date_signals.append((
                        _res["CompositeRank"],
                        _sym,
                        _bar_idx,
                        _res["Score"],
                        _res["SetupType"],
                        _res.get("CSRank5d", 0.5),
                        _res.get("Horizon", "Mid 5-14D"),
                        # leading signals for walk-forward IC analysis
                        _res.get("UpVolSkew",          0),
                        _res.get("CPR",                0),
                        _res.get("SpreadComp",         0),
                        _res.get("ATRExpOnset",        0),
                        _res.get("OI_Buildup",         0),
                        _res.get("CompressionStreak",  0),
                        _res.get("HHHLScore",          0),
                        _res.get("BBSqueeze",          0),
                        _res.get("VolDryUp",           0),
                    ))

                if not _date_signals:
                    continue

                # Sort by CompositeRank, take top N
                _date_signals.sort(key=lambda x: x[0], reverse=True)

                _selected = _date_signals[:wf_topn]

                for _rank_val, _sym, _bar_idx, _score, _setup, _csrank, _horizon, \
                        _uv, _cpr_s, _sc, _atr_exp, _oi_b, _cstreak, _hhhl, _bbs, _vdu in _selected:
                    _dfc = _wf_sym_dfs[_sym]

                    # ── ENTRY TIMING FILTER ──
                    _actionable = {"Imminent BO", "Intraday", "Swing 2-5D"}
                    if _horizon not in _actionable:
                        continue

                    # Slippage model: entry at NEXT DAY open
                    _entry_idx = _bar_idx + 1
                    if _entry_idx >= len(_dfc):
                        continue
                    _entry_p = float(_dfc.iloc[_entry_idx]["open"])
                    if _entry_p == 0:
                        continue
                    _entry_p = _entry_p * (1.0 + wf_slippage_bps / 10000.0)

                    # ── RAW SIGNAL RECORD (pre-confirmation-filter) ──
                    # Stored here, BEFORE the confirmation candle check, so that the
                    # Leading Signal IC tables can be computed on unfiltered signals.
                    # The confirmation filter is correlated with signal values (high-score
                    # setups are more likely to confirm), so IC measured post-filter is
                    # inflated vs what the signal would achieve in practice.
                    # R1d/R3d/R5d are computed forward from entry open regardless of
                    # whether the confirmation filter would have excluded this trade.
                    def _wf_ret_raw(n):
                        _ix = min(_entry_idx + n, len(_dfc) - 1)
                        return round((float(_dfc.iloc[_ix]["close"]) - _entry_p) / _entry_p * 100, 3)
                    _r1_raw = _wf_ret_raw(1); _r3_raw = _wf_ret_raw(3); _r5_raw = _wf_ret_raw(5)
                    _ret_matched_raw = _r1_raw if _horizon == "Imminent BO" else \
                                       (_r5_raw if _horizon == "Swing 2-5D" else _r3_raw)
                    wf_all_signals_raw.append({
                        "Date":          _test_date,
                        "Ticker":        _sym,
                        "Score":         _score,
                        "RetMatched%":   _ret_matched_raw,
                        f"R{wf_hold}d%": _wf_ret_raw(wf_hold),
                        "UpVolSkew":     _uv,
                        "CPR":           _cpr_s,
                        "SpreadComp":    _sc,
                        "ATRExpOnset":   _atr_exp,
                        "OI_Buildup":    _oi_b,
                        "CompressionStreak": _cstreak,
                        "HHHLScore":     _hhhl,
                        "BBSqueeze":     _bbs,
                        "VolDryUp":      _vdu,
                    })

                    # ── CONFIRMATION CANDLE FILTER ──
                    _entry_bar    = _dfc.iloc[_entry_idx]
                    _signal_close = float(_dfc.iloc[_bar_idx]["close"])
                    _entry_close  = float(_entry_bar["close"])
                    _entry_vol    = float(_entry_bar["volume"]) if pd.notna(_entry_bar["volume"]) else 0
                    _vol_base = float(_dfc.iloc[max(0, _bar_idx-20):_bar_idx]["volume"].mean()) if _bar_idx >= 5 else _entry_vol
                    if _entry_close <= _signal_close:
                        continue
                    _vol_threshold = _vol_base * (0.8 if _setup == "Breakout" else 1.0)
                    if _entry_vol < _vol_threshold:
                        continue
                    # Reuse the forward returns already computed for the raw record
                    _r1  = _r1_raw; _r3 = _r3_raw; _r5 = _r5_raw
                    _exit_idx = min(_bar_idx + wf_hold, len(_dfc) - 1)
                    _exit_p   = float(_dfc.iloc[_exit_idx]["close"])
                    _ret_pct  = round((_exit_p - _entry_p) / _entry_p * 100, 3)
                    _ret_matched = _ret_matched_raw

                    # Max drawdown in hold window (low vs entry)
                    _hold_lows = _dfc.iloc[_entry_idx:_exit_idx + 1]["low"]
                    _max_dd    = round((_hold_lows.min() - _entry_p) / _entry_p * 100, 3) if len(_hold_lows) > 0 else 0.0
                    _hold_highs = _dfc.iloc[_entry_idx:_exit_idx + 1]["high"]
                    _max_gain   = round((_hold_highs.max() - _entry_p) / _entry_p * 100, 3) if len(_hold_highs) > 0 else 0.0

                    wf_all_trades.append({
                        "Date":        _test_date,
                        "Regime":      _wf_regime,
                        "Ticker":      _sym,
                        "Setup":       _setup,
                        "Horizon":     _horizon,
                        "Score":       _score,
                        "CSRank5d":    _csrank,
                        "Entry":       round(_entry_p, 2),
                        "Exit":        round(_exit_p, 2),
                        f"R{wf_hold}d%": _ret_pct,
                        "R1d%":        _r1,
                        "R3d%":        _r3,
                        "R5d%":        _r5,
                        "RetMatched%": _ret_matched,
                        "MaxGain%":    _max_gain,
                        "MaxDD%":      _max_dd,
                        "Win":         1 if _ret_matched > 0 else 0,  # Win on horizon-matched return
                        # Leading signals carried for quartile IC analysis
                        "UpVolSkew":         _uv,
                        "CPR":               _cpr_s,
                        "SpreadComp":        _sc,
                        "ATRExpOnset":       _atr_exp,
                        "OI_Buildup":        _oi_b,
                        "CompressionStreak": _cstreak,
                        "HHHLScore":         _hhhl,
                        "BBSqueeze":         _bbs,
                        "VolDryUp":          _vdu,
                    })

            wf_progress.empty()

            # ── L-4: SURVIVORSHIP BIAS LOG ──
            # Track what fraction of the universe returned data on each test date.
            # Missing data = delisted, suspended, or merged stocks — the "worst" outcomes
            # that are systematically excluded from backtests, creating upward bias.
            _total_universe  = len(_wf_sym_dfs)
            _dates_tested    = len(test_dates)
            _total_scored    = len(wf_all_trades)
            _potential_max   = _total_universe * _dates_tested
            _coverage_pct    = round(_total_scored / (_potential_max + 1e-9) * 100, 1)
            if _coverage_pct < 80:
                st.warning(
                    f"⚠️ Survivorship bias alert: {_coverage_pct}% coverage "
                    f"({_total_scored} scored / {_potential_max} potential). "
                    f"Low coverage inflates returns — {100 - _coverage_pct:.0f}% of potential signals had no data "
                    f"(likely delisted/suspended stocks with bad outcomes)."
                )
            else:
                st.info(f"✓ Coverage: {_coverage_pct}% ({_total_scored} / {_potential_max} potential signals)")

            if not wf_all_trades:
                st.warning("No signals passed filters across the full date range. Try lowering the Min Score.")
            else:
                wf_df = pd.DataFrame(wf_all_trades)
                # FIX-02: Use horizon-matched return for IC and adaptive weight computation.
                # ret_col_display keeps the user-selected hold for aggregate stats display.
                # ret_col is used for all IC/weight calculations.
                ret_col_display = f"R{wf_hold}d%"
                ret_col = "RetMatched%" if "RetMatched%" in wf_df.columns else ret_col_display

                # Fix 12: Compute per-stock win rate from walk-forward and store in session_state.
                # This feeds the KellyFrac calculation in score_stock_dual — when a stock has
                # appeared in enough walk-forward trades, its Kelly fraction uses real observed
                # win rate instead of the Bayesian prior.
                # Minimum 5 trades per stock required for a reliable estimate.
                _psw = {}
                if "Ticker" in wf_df.columns and "Win" in wf_df.columns:
                    _ticker_grp = wf_df.groupby("Ticker")["Win"]
                    for _tk, _wins in _ticker_grp:
                        if len(_wins) >= 5:
                            _psw[_tk] = round(float(_wins.mean()), 3)
                    if _psw:
                        # Merge with any existing per_stock_winrate (from prior WF runs)
                        _existing = st.session_state.get("per_stock_winrate", {})
                        _existing.update(_psw)
                        st.session_state.per_stock_winrate = _existing
                        _save_screener_state()   # persist win rates to disk
                        st.caption(f"Kelly data: {len(_psw)} stocks now use walk-forward win rates.")

                # ── AUTO IC-BASED FACTOR WEIGHT UPDATE ────────────────────────────────
                # Circularity fix: the IC measurement must use a HELD-OUT period,
                # not the same data that generated the signals under the current weights.
                # If SpreadComp already dominates the score, the top-ranked stocks all
                # have high SpreadComp — of course it shows high IC in-sample. That is
                # a tautology, not new information.
                #
                # Fix: split wf_df chronologically.
                #   Train set (first 70%): generated signals with current weights — DO NOT
                #   measure IC here; it is contaminated by the current weight structure.
                #   Held-out set (last 30%): IC measured here. These are the most recent
                #   trades, scored by the engine as-is, but the IC is measured post-hoc
                #   against their signal values WITHOUT the scoring model's influence on
                #   which trades were selected (selection bias is still present but at
                #   least the IC is not circular within the same date range).
                #
                # The held-out set must have at least _WF_MIN_TRADES_FOR_REWEIGHT trades.
                # If the full dataset is too small, fall back to full-set measurement
                # with a reduced blend factor (less trust in the circular estimate).
                #
                _WF_MIN_TRADES_FOR_REWEIGHT = 30
                _BLEND_NEW = 0.70
                _BLEND_OLD = 0.30

                _prior_w = st.session_state.get("adaptive_weights",
                    {"spread": 0.40, "vol": 0.40, "coil": 0.20})

                _ic_signals = [
                    ("spread",   "SpreadComp"),
                    ("vol",      "UpVolSkew"),
                    ("coil",     "BBSqueeze"),
                ]

                # ── CHRONOLOGICAL OOS SPLIT — calendar time, not trade count ──
                # Problem with trade-count split (old 70/30): if the last 30% of TRADES
                # span only 2 months but the first 70% span 10 months, the "holdout" is
                # actually a dense recent period (post-event cluster) that is NOT
                # representative of typical performance.
                # Fix: holdout = last 20% of CALENDAR TIME in the walk-forward range.
                # If the user has set an explicit "OOS start date" in session_state
                # (from the sidebar date picker), that date takes priority.
                _wf_sorted = wf_df.sort_values("Date").reset_index(drop=True) \
                             if "Date" in wf_df.columns else wf_df.copy()

                _oos_override = st.session_state.get("wf_oos_start_date")
                if _oos_override and "Date" in _wf_sorted.columns:
                    try:
                        _oos_dt = pd.to_datetime(_oos_override)
                        _wf_holdout = _wf_sorted[
                            pd.to_datetime(_wf_sorted["Date"]) >= _oos_dt
                        ].copy()
                        _blend_note = (
                            f"OOS set = trades on/after {_oos_override} "
                            f"({len(_wf_holdout)} trades, user-defined boundary)."
                        )
                    except Exception:
                        _oos_override = None

                if not _oos_override or not isinstance(_oos_override, str):
                    if "Date" in _wf_sorted.columns and len(_wf_sorted) >= 10:
                        _wf_dates = pd.to_datetime(_wf_sorted["Date"])
                        _d_min = _wf_dates.min()
                        _d_max = _wf_dates.max()
                        _d_range = (_d_max - _d_min).days
                        _oos_cutoff = _d_min + pd.Timedelta(days=int(_d_range * 0.80))
                        _wf_holdout = _wf_sorted[_wf_dates >= _oos_cutoff].copy()
                        _blend_note = (
                            f"OOS = last 20% of calendar time "
                            f"(from {_oos_cutoff.date()} — {len(_wf_holdout)} trades)."
                        )
                    else:
                        _split_idx = int(len(_wf_sorted) * 0.80)
                        _wf_holdout = _wf_sorted.iloc[_split_idx:].copy()
                        _blend_note = f"OOS = last 20% of trades ({len(_wf_holdout)} trades)."

                # If held-out set is too small, fall back to full set but halve blend trust
                if len(_wf_holdout) < _WF_MIN_TRADES_FOR_REWEIGHT:
                    _wf_holdout  = _wf_sorted.copy()
                    _BLEND_NEW   = min(_BLEND_NEW, 0.35)   # half trust for circular estimate
                    _blend_note  = "⚠ Held-out set too small — using full set with reduced blend (35%)."
                else:
                    _blend_note  = _blend_note + f" IC measured on held-out {len(_wf_holdout)} trades."

                _new_raw = {}
                _enough_data = len(_wf_holdout) >= _WF_MIN_TRADES_FOR_REWEIGHT

                if _enough_data:
                    _wf_ic = _wf_holdout.dropna(subset=[ret_col]).copy()
                    for _wkey, _wcol in _ic_signals:
                        if _wcol not in _wf_ic.columns or _wf_ic[_wcol].nunique() < 4:
                            _new_raw[_wkey] = max(_prior_w.get(_wkey, 0.0), 0.01)
                            continue
                        try:
                            # Robust qcut: derive actual bin count after dropping duplicates,
                            # then match labels dynamically — prevents label mismatch crash.
                            _all_q_labels = ["Q1", "Q2", "Q3", "Q4"]
                            _bins_edges = pd.qcut(_wf_ic[_wcol], 4, retbins=True,
                                                  duplicates="drop")[1]
                            _n_bins = len(_bins_edges) - 1
                            if _n_bins < 2:
                                _new_raw[_wkey] = max(_prior_w.get(_wkey, 0.0), 0.01)
                                continue
                            _q_labels = _all_q_labels[:_n_bins]
                            _wf_ic["_Q"] = pd.qcut(_wf_ic[_wcol], _bins_edges,
                                                    labels=_q_labels,
                                                    duplicates="drop",
                                                    include_lowest=True)
                            _q_grp = _wf_ic.groupby("_Q", observed=True)[ret_col].mean()
                            if len(_q_grp) >= 2:
                                # Q4-Q1 spread — works even if fewer than 4 bins survived
                                _spread_ic = float(_q_grp.iloc[-1]) - float(_q_grp.iloc[0])
                                _new_raw[_wkey] = max(_spread_ic, 0.0)   # clip negative
                            else:
                                _new_raw[_wkey] = max(_prior_w.get(_wkey, 0.0), 0.01)
                        except Exception:
                            _new_raw[_wkey] = max(_prior_w.get(_wkey, 0.0), 0.01)

                    # Softmax normalise to sum=1
                    _total_ic = sum(_new_raw.values()) + 1e-9
                    _norm_new = {k: v / _total_ic for k, v in _new_raw.items()}

                    # Blend with prior
                    _blended = {}
                    for _wkey, _ in _ic_signals:
                        _blended[_wkey] = round(
                            _BLEND_NEW * _norm_new.get(_wkey, _prior_w.get(_wkey, 0.33)) +
                            _BLEND_OLD * _prior_w.get(_wkey, 0.33), 4
                        )

                    # Re-normalise blended (rounding can make sum ≠ 1)
                    _btotal = sum(_blended.values()) + 1e-9
                    _blended = {k: round(v / _btotal, 4) for k, v in _blended.items()}
                    st.session_state.adaptive_weights = _blended

                    # Persist to disk immediately so restart doesn't lose these weights
                    _save_screener_state()

                    # Show what happened
                    _aw = _blended
                    st.subheader("⚡ Auto Weight Update — IC Feedback")
                    st.caption(_blend_note)
                    _w_cols = st.columns(4)
                    _w_cols[0].metric("SpreadComp weight",  f"{_aw['spread']:.3f}",
                                      delta=f"{_aw['spread'] - _prior_w.get('spread', 0.40):+.3f} vs prior")
                    _w_cols[1].metric("VolQuiet weight",    f"{_aw['vol']:.3f}",
                                      delta=f"{_aw['vol'] - _prior_w.get('vol', 0.40):+.3f} vs prior")
                    _w_cols[2].metric("Coil weight",        f"{_aw['coil']:.3f}",
                                      delta=f"{_aw['coil'] - _prior_w.get('coil', 0.20):+.3f} vs prior")
                    _w_cols[3].metric("Holdout trades",     str(len(_wf_holdout)))
                    st.caption(
                        "IC measured on held-out last 30% of walk-forward range (out-of-sample). "
                        "Weights updated as: new_blend% new IC + old_blend% prior. "
                        "Score formula uses these weights on next extraction. Requires ≥30 holdout trades."
                    )
                else:
                    st.info(
                        f"Auto weight update inactive — held-out set has {len(_wf_holdout)} trades "
                        f"(need ≥{_WF_MIN_TRADES_FOR_REWEIGHT}). Run more WF dates or lower min score."
                    )

                # ── AGGREGATE STATS ──
                total_trades = len(wf_df)
                win_rate     = round(wf_df["Win"].mean() * 100, 1)
                avg_ret      = round(wf_df[ret_col].mean(), 3)
                avg_dd       = round(wf_df["MaxDD%"].mean(), 3)
                avg_gain     = round(wf_df["MaxGain%"].mean(), 3)

                # Sharpe: annualised using trading days per hold period
                # σ of per-trade returns; scale to annual using √(252 / hold_period)
                _ret_std = wf_df[ret_col].std()
                sharpe   = round((avg_ret / (_ret_std + 1e-9)) * np.sqrt(252 / wf_hold), 2) if _ret_std > 0 else 0.0

                # Profit factor: sum of wins / abs(sum of losses)
                _wins   = wf_df.loc[wf_df[ret_col] > 0, ret_col].sum()
                _losses = abs(wf_df.loc[wf_df[ret_col] < 0, ret_col].sum())
                profit_factor = round(_wins / (_losses + 1e-9), 2)

                n_dates = wf_df["Date"].nunique()

                st.success(
                    f"✅ {total_trades} trades across {n_dates} dates  ·  "
                    f"top {wf_topn}/date  ·  {wf_hold}d hold  ·  entry at next-day open"
                )

                mc = st.columns(6)
                mc[0].metric("Win Rate",      f"{win_rate}%")
                mc[1].metric(f"Avg {wf_hold}d Return", f"{avg_ret}%")
                mc[2].metric("Avg Max Gain",  f"{avg_gain}%")
                mc[3].metric("Avg Max DD",    f"{avg_dd}%")
                mc[4].metric("Sharpe (ann.)", f"{sharpe}")
                mc[5].metric("Profit Factor", f"{profit_factor}")

                if sharpe > 1.0:
                    st.success(f"✅ Sharpe {sharpe} > 1.0 — signal shows real risk-adjusted edge")
                elif sharpe > 0.5:
                    st.info(f"ℹ️ Sharpe {sharpe} — marginal edge, needs improvement or tighter filters")
                else:
                    st.warning(f"⚠️ Sharpe {sharpe} — signal is not generating consistent risk-adjusted returns")

                # ── INFORMATION COEFFICIENT DISTRIBUTION ──
                # IC per date = Spearman rank corr(Score, forward_return) on that date.
                # The mean IC across all dates is the primary institutional validity metric.
                # IC > 0.05 = signal has edge; IC > 0.10 = strong edge; t-stat > 2 = significant.
                st.subheader("📐 Information Coefficient (IC) — Primary Validity Metric")
                st.caption("IC = rank correlation between signal Score and forward return per date. "
                           "Mean IC > 0.05 = edge. t-stat > 2 = statistically significant.")
                _ic_by_date = []
                try:
                    from scipy import stats as _sp_stats
                    for _d, _dg in wf_df.groupby("Date"):
                        _dg_v = _dg.dropna(subset=[ret_col, "Score"])
                        if len(_dg_v) >= 5:
                            _ic_d, _ = _sp_stats.spearmanr(_dg_v["Score"], _dg_v[ret_col])
                            _ic_by_date.append(float(_ic_d))
                    if _ic_by_date:
                        _ic_arr     = np.array(_ic_by_date)
                        _ic_mean    = round(float(_ic_arr.mean()), 4)
                        _ic_std     = round(float(_ic_arr.std()),  4)
                        _ic_tstat   = round(float(_ic_mean / (_ic_std / np.sqrt(len(_ic_arr)) + 1e-9)), 2)
                        _ic_pct_pos = round(float((_ic_arr > 0).mean() * 100), 1)
                        ic_cols = st.columns(4)
                        ic_cols[0].metric("Mean IC",        f"{_ic_mean}")
                        ic_cols[1].metric("IC Std Dev",     f"{_ic_std}")
                        ic_cols[2].metric("t-statistic",    f"{_ic_tstat}")
                        ic_cols[3].metric("% Dates IC > 0", f"{_ic_pct_pos}%")
                        if _ic_tstat > 2.0 and _ic_mean > 0.05:
                            st.success(f"✅ IC t-stat {_ic_tstat} > 2.0 and mean IC {_ic_mean} > 0.05 — signal is statistically significant")
                        elif _ic_tstat > 1.5:
                            st.info(f"ℹ️ IC t-stat {_ic_tstat} — marginal significance. Increase universe size or hold period.")
                        else:
                            st.warning(f"⚠️ IC t-stat {_ic_tstat} < 1.5 — signal not yet statistically significant across this date range")
                except ImportError:
                    st.info("Install scipy for IC computation: pip install scipy")
                wf_df_sorted = wf_df.sort_values("Date").reset_index(drop=True)
                # Group by date → mean return per day → cumulative
                _daily_ret = wf_df_sorted.groupby("Date")[ret_col].mean()
                _cum_ret   = (1 + _daily_ret / 100).cumprod() - 1
                _cum_ret   = _cum_ret * 100   # convert to %

                fig_wf = go.Figure()
                fig_wf.add_trace(go.Scatter(
                    x=list(_cum_ret.index), y=list(_cum_ret.values),
                    mode="lines",
                    line=dict(color="#00d084", width=1.5),
                    fill="tozeroy",
                    fillcolor="rgba(0,208,132,0.08)",
                    name="Portfolio"
                ))
                fig_wf.add_hline(y=0, line_dash="dash", line_color="#444", opacity=0.6)
                fig_wf.update_layout(
                    height=320,
                    xaxis=dict(title="Date", tickfont=dict(size=10, color="#888")),
                    yaxis=dict(title="Cumulative Return %", tickfont=dict(size=10, color="#888")),
                    plot_bgcolor="#000000", paper_bgcolor="#0a0a0a",
                    font=dict(color="#e8e8e8", family="IBM Plex Mono", size=10),
                    margin=dict(t=20, b=30, l=50, r=10),
                )
                st.plotly_chart(fig_wf, use_container_width=True)

                # ── SCORE QUARTILE TABLE (across all dates) ──
                # Define colour function once here — used by all quartile/regime tables below
                def _wf_color(v):
                    if not isinstance(v, (int, float)): return ""
                    if v > 1:  return "background-color:#0d2200;color:#00d084;font-weight:700"
                    if v > 0:  return "background-color:#0a1a00;color:#7ec87a"
                    return "background-color:#1f0000;color:#ff3b3b"

                st.subheader("📊 Score vs Return — Quartile breakdown across all dates")
                st.caption("If Q4 Score > Q1 Score return, the composite score is predictive. "
                           "Check CSRank quartile below — it should show stronger monotonicity.")
                _wf_q = wf_df.dropna(subset=[ret_col]).copy()
                if len(_wf_q) >= 20:
                    _all_labels = ["Q1 Low", "Q2", "Q3", "Q4 High"]
                    _bins = pd.qcut(_wf_q["Score"], 4, retbins=True, duplicates="drop")[1]
                    _labels = _all_labels[:len(_bins) - 1]
                    _wf_q["ScoreQ"] = pd.qcut(_wf_q["Score"], _bins,
                                               labels=_labels,
                                               duplicates="drop",
                                               include_lowest=True)
                    _wf_qt = (_wf_q.groupby("ScoreQ", observed=True)[ret_col]
                              .agg(Trades="count",
                                   AvgReturn="mean",
                                   WinRate=lambda x: round((x > 0).mean() * 100, 1))
                              .round(3).reset_index())
                    _wf_qt.columns = ["Score Quartile", "Trades", f"Avg {wf_hold}d Return %", "Win Rate %"]
                    st.dataframe(
                        _wf_qt.style.applymap(_wf_color, subset=[f"Avg {wf_hold}d Return %", "Win Rate %"]),
                        use_container_width=True, hide_index=True
                    )

                # ── CSRANK QUARTILE TABLE — the new primary signal ──
                if "CSRank5d" in wf_df.columns and len(_wf_q) >= 20:
                    st.subheader("📊 Cross-Sectional RS vs Return — CSRank5d quartile")
                    st.caption("CSRank5d ranks each stock within its universe peers on that date. "
                               "Q4 = top 25% of universe by 5-day return. Should monotonically beat Q1 "
                               "in ANY regime — this is the regime-agnostic discriminator.")
                    _wf_q2 = wf_df.dropna(subset=[ret_col, "CSRank5d"]).copy()
                    _wf_q2["CSRankQ"] = pd.qcut(_wf_q2["CSRank5d"], 4,
                                                  labels=["Q1 Low","Q2","Q3","Q4 High"],
                                                  duplicates="drop")
                    _cs_qt = (_wf_q2.groupby("CSRankQ", observed=True)[ret_col]
                              .agg(Trades="count",
                                   AvgReturn="mean",
                                   WinRate=lambda x: round((x > 0).mean() * 100, 1))
                              .round(3).reset_index())
                    _cs_qt.columns = ["CSRank Quartile", "Trades", f"Avg {wf_hold}d Return %", "Win Rate %"]
                    st.dataframe(
                        _cs_qt.style.applymap(_wf_color, subset=[f"Avg {wf_hold}d Return %", "Win Rate %"]),
                        use_container_width=True, hide_index=True
                    )
                    # Tell the user what the table means
                    if len(_cs_qt) >= 4:
                        _q4_ret = float(_cs_qt.iloc[-1][f"Avg {wf_hold}d Return %"])
                        _q1_ret = float(_cs_qt.iloc[0][f"Avg {wf_hold}d Return %"])
                        _spread = round(_q4_ret - _q1_ret, 3)
                        if _spread > 0.5:
                            st.success(f"✅ CSRank spread Q4−Q1 = {_spread}% — cross-sectional RS is predictive across regimes")
                        elif _spread > 0:
                            st.info(f"ℹ️ CSRank spread = {_spread}% — positive but weak. "
                                    f"Consider tightening min score or hold period.")
                        else:
                            st.warning(f"⚠️ CSRank spread = {_spread}% — not discriminating yet. "
                                       f"Check if universe is large enough (need 50+ stocks).")

                # ── HORIZON BREAKDOWN ──
                if "Horizon" in wf_df.columns:
                    st.subheader("📋 Performance by Horizon Tier")
                    st.caption(
                        "Only Imminent BO, Intraday, and Swing 2-5D are entered in the walk-forward. "
                        "If Imminent BO consistently beats Swing, tighten the horizon filter further."
                    )
                    # FIX-08: Show R1d/R3d/R5d per horizon so analyst sees whether
                    # Imminent BO peaks at 1d (correct) and Swing peaks at 5d (correct).
                    _hz_cols_avail = [c for c in ["R1d%","R3d%","R5d%","RetMatched%"] if c in wf_df.columns]
                    _hz_agg_dict = {"Trades": (ret_col, "count")}
                    for _hc in _hz_cols_avail:
                        _hz_agg_dict[f"Avg {_hc}"] = (_hc, "mean")
                    _hz_agg_dict["WinRate%"] = ("RetMatched%" if "RetMatched%" in wf_df.columns else ret_col,
                                                 lambda x: round((x > 0).mean() * 100, 1))
                    _hz_grp = wf_df.groupby("Horizon").agg(**_hz_agg_dict).round(3).reset_index()
                    _hz_ret_cols = [c for c in _hz_grp.columns if c.startswith("Avg ")]
                    st.dataframe(
                        _hz_grp.style.applymap(_wf_color, subset=_hz_ret_cols),
                        use_container_width=True, hide_index=True
                    )
                    st.caption(
                        "Imminent BO should peak at Avg R1d%. Swing 2-5D should peak at Avg R5d%. "
                        "If they don't, the horizon classification or entry timing needs adjustment."
                    )

                # ── REGIME BREAKDOWN ──
                st.subheader("📋 Performance by Market Regime (based on date)")
                if "Regime" in wf_df.columns:
                    _regime_grp = (
                        wf_df.groupby("Regime")[ret_col]
                        .agg(Trades="count",
                             AvgReturn="mean",
                             WinRate=lambda x: round((x > 0).mean() * 100, 1))
                        .round(3).reset_index()
                    )
                    _regime_grp.columns = ["Regime", "Trades", f"Avg {wf_hold}d Return %", "Win Rate %"]
                    st.dataframe(
                        _regime_grp.style.applymap(_wf_color, subset=[f"Avg {wf_hold}d Return %", "Win Rate %"]),
                        use_container_width=True, hide_index=True
                    )
                    st.caption(
                        "BEAR regime trades are now included — cross-sectional RS should find leaders "
                        "even in down markets. If BEAR still shows losses, the universe may be too small."
                    )

                wf_df["DateTS"] = pd.to_datetime(wf_df["Date"])
                try:
                    _regime_monthly = (
                        wf_df.set_index("DateTS")
                        .resample("ME")[ret_col]
                        .agg(Trades="count", AvgReturn="mean",
                             WinRate=lambda x: round((x > 0).mean() * 100, 1))
                        .round(3).reset_index()
                    )
                except ValueError:
                    # pandas < 2.2 uses "M" not "ME"
                    _regime_monthly = (
                        wf_df.set_index("DateTS")
                        .resample("M")[ret_col]
                        .agg(Trades="count", AvgReturn="mean",
                             WinRate=lambda x: round((x > 0).mean() * 100, 1))
                        .round(3).reset_index()
                    )
                _regime_monthly.columns = ["Month", "Trades", f"Avg {wf_hold}d Return %", "Win Rate %"]
                _regime_monthly["Month"] = _regime_monthly["Month"].dt.strftime("%b %Y")

                st.dataframe(
                    _regime_monthly.style.applymap(_wf_color, subset=[f"Avg {wf_hold}d Return %", "Win Rate %"]),
                    use_container_width=True, hide_index=True, height=280
                )
                st.caption(
                    "Red months show when the model breaks down. "
                    "Cross-reference with Nifty direction — if losses cluster in bear phases, "
                    "consider disabling the screener when Nifty is below its 50DMA."
                )

                # ── LEADING SIGNAL PREDICTIVE POWER ──
                # Two views are shown side by side:
                #   Filtered IC  : trades that passed the confirmation candle filter (what the model traded)
                #   Raw IC       : ALL signals before the filter (true signal predictiveness)
                # If Raw IC > Filtered IC, the confirmation filter is selecting on correlated signal
                # values and inflating the apparent IC. If they are similar, the filter is neutral.
                # Flat or inverted Raw IC = the signal genuinely has no edge and should be downweighted.
                st.subheader("🔬 Leading Signal Predictive Power — Quartile vs Return")
                st.caption(
                    "**Filtered** = trades that passed confirmation candle filter (what was actually traded). "
                    "**Raw** = all signals before the filter (true out-of-sample IC). "
                    "Large Filtered > Raw gap = confirmation filter is selecting on signal values (inflated IC). "
                    "Q4 = top 25% signal strength. A genuine signal shows Q4 > Q3 > Q2 > Q1 in BOTH views."
                )
                _lead_sigs = [
                    ("UpVolSkew",          "Upside Volume Skew"),
                    ("CPR",                "Close Position Rank"),
                    ("SpreadComp",         "Spread Compression"),
                    ("ATRExpOnset",        "ATR Expansion Onset"),
                    ("OI_Buildup",         "OI Buildup (F&O only)"),
                    ("CompressionStreak",  "Compression Streak (days)"),
                    ("HHHLScore",          "Higher Highs + Higher Lows"),
                    ("BBSqueeze",          "Bollinger Band Squeeze"),
                    ("VolDryUp",           "Volume Dry-Up"),
                ]
                _wf_lead_filtered = wf_df.dropna(subset=[ret_col]).copy()
                _wf_lead_raw = pd.DataFrame(wf_all_signals_raw).dropna(subset=["RetMatched%"]) \
                               if wf_all_signals_raw else pd.DataFrame()
                _ret_col_raw = "RetMatched%"

                def _render_ic_table(df_src, ret_c, sig_col, sig_label, hold_d):
                    """Render one quartile IC table. Returns Q4-Q1 spread or None."""
                    if sig_col not in df_src.columns:
                        return None
                    try:
                        _sq = df_src.dropna(subset=[sig_col]).copy()
                        if _sq[sig_col].nunique() < 3:
                            return None
                        _all_labels = ["Q1","Q2","Q3","Q4"]
                        _bedges = pd.qcut(_sq[sig_col], 4, retbins=True, duplicates="drop")[1]
                        _nb = len(_bedges) - 1
                        if _nb < 2:
                            return None
                        _qlabels = _all_labels[:_nb]
                        _sq["Q"] = pd.qcut(_sq[sig_col], _bedges, labels=_qlabels,
                                           duplicates="drop", include_lowest=True)
                        _qt = (_sq.groupby("Q", observed=True)[ret_c]
                               .agg(Trades="count", AvgReturn="mean",
                                    WinRate=lambda x: round((x > 0).mean() * 100, 1))
                               .round(3).reset_index())
                        _qt.columns = ["Q","Trades",f"Avg Ret%","Win%"]
                        _spread = round(float(_qt.iloc[-1]["Avg Ret%"]) - float(_qt.iloc[0]["Avg Ret%"]), 3) \
                                  if len(_qt) >= 2 else None
                        return _qt, _spread
                    except Exception:
                        return None

                if len(_wf_lead_filtered) >= 20:
                    for _li, (_sig_col, _sig_label) in enumerate(_lead_sigs):
                        _filt_result = _render_ic_table(_wf_lead_filtered, ret_col, _sig_col, _sig_label, wf_hold) \
                                       if len(_wf_lead_filtered) >= 20 else None
                        _raw_result  = _render_ic_table(_wf_lead_raw, _ret_col_raw, _sig_col, _sig_label, wf_hold) \
                                       if len(_wf_lead_raw) >= 20 else None
                        if _filt_result is None and _raw_result is None:
                            continue
                        st.markdown(f"**{_sig_label}**")
                        _ic_cols = st.columns(2)
                        with _ic_cols[0]:
                            if _filt_result is not None:
                                _qt, _spread = _filt_result
                                _color = "green" if _spread and _spread > 0.3 else \
                                         "orange" if _spread and _spread > 0 else "red"
                                st.caption(f"Filtered — Q4−Q1: :{_color}[{_spread:+.3f}%]" if _spread is not None else "Filtered")
                                st.dataframe(_qt.style.map(_wf_color, subset=["Avg Ret%","Win%"]),
                                             use_container_width=True, hide_index=True, height=185)
                            else:
                                st.caption("Filtered: insufficient data")
                        with _ic_cols[1]:
                            if _raw_result is not None:
                                _qt_r, _spread_r = _raw_result
                                _color_r = "green" if _spread_r and _spread_r > 0.3 else \
                                           "orange" if _spread_r and _spread_r > 0 else "red"
                                st.caption(f"Raw (pre-filter) — Q4−Q1: :{_color_r}[{_spread_r:+.3f}%]" if _spread_r is not None else "Raw")
                                st.dataframe(_qt_r.style.map(_wf_color, subset=["Avg Ret%","Win%"]),
                                             use_container_width=True, hide_index=True, height=185)
                            else:
                                st.caption("Raw: insufficient data")


# ============================================================
# SCORE EXPLANATION ENGINE
# Translates every factor value into plain English with context.
# Each line says WHAT the number is, WHY it matters, and WHAT it implies.
# ============================================================

# ============================================================
# BLOOMBERG-STYLE CHART DASHBOARD
# ============================================================
# Clicking any row in the screener table opens this panel.
# Features:
#   - Candlestick + Volume subplot
#   - EMA9 / EMA20 / EMA50 overlaid
#   - ATR bands
#   - BUY marker: green candle above EMA9 with vol > 1.2× avg
#   - SELL marker: ATR trailing stop hit OR close < EMA20
#   - Bloomberg dark colour scheme throughout
# ============================================================

# ============================================================
# SIGNAL HISTORY — Audit Log Viewer
# ============================================================
# Reads .monarch_signal_log.csv and displays the last 500 signals
# with performance analytics: win-rate by setup type, regime, sector,
# and rolling return chart so model drift is immediately visible.
# ============================================================
st.header("📋 Signal History — Audit Log")
st.caption(
    "Every signal with Score ≥ 30 is logged here automatically each time "
    "the screener runs. Use this to track model performance over time, "
    "identify which setup types / regimes / sectors have the best hit rate, "
    "and detect when the model starts degrading."
)

_sig_log_df = _load_signal_log(max_rows=500)

if _sig_log_df.empty:
    st.info(
        "No signals logged yet. Run the screener (with data extracted) to start "
        "building the audit trail. Signals accumulate automatically — no action needed."
    )
else:
    # ── Summary metrics ────────────────────────────────────────────────────
    _sl_total = len(_sig_log_df)
    _sl_tickers = _sig_log_df["Ticker"].nunique() if "Ticker" in _sig_log_df.columns else 0
    _sl_regimes = _sig_log_df["Regime"].value_counts().to_dict() if "Regime" in _sig_log_df.columns else {}
    _sl_ev_flagged = int((_sig_log_df["EventFlag"].astype(str).str.len() > 0).sum()) \
                     if "EventFlag" in _sig_log_df.columns else 0
    _sl_fund_fails = int((~_sig_log_df["FundamentalOK"].astype(str).str.lower().isin(
        ["true","1","yes","nan","n/a",""])).sum()) \
        if "FundamentalOK" in _sig_log_df.columns else 0

    _m1, _m2, _m3, _m4, _m5 = st.columns(5)
    _m1.metric("Total Signals", _sl_total)
    _m2.metric("Unique Tickers", _sl_tickers)
    _m3.metric("Event-Flagged", _sl_ev_flagged, help="Signals within ±3d of corporate event")
    _m4.metric("Fundamental ✗", _sl_fund_fails, help="Signals where CFO was negative")
    _m5.metric(
        "Regimes",
        "  ".join(f"{k}:{v}" for k, v in sorted(_sl_regimes.items())) or "—"
    )

    st.divider()

    # ── Breakdown tabs ─────────────────────────────────────────────────────
    _sl_tab_raw, _sl_tab_setup, _sl_tab_sector, _sl_tab_regime = st.tabs([
        "Raw Log", "By Setup Type", "By Sector", "By Regime"
    ])

    with _sl_tab_raw:
        _sl_disp_cols = [c for c in _SIGNAL_LOG_COLS if c in _sig_log_df.columns]
        _sl_disp = _sig_log_df[_sl_disp_cols].sort_values(
            "Timestamp", ascending=False
        ).reset_index(drop=True) if "Timestamp" in _sig_log_df.columns else _sig_log_df

        def _sl_score_color(v):
            try:
                v = float(v)
                if v >= 70: return "background-color:#1a3300;color:#00d084;font-weight:700"
                if v >= 50: return "background-color:#1a2200;color:#b8e06a"
                if v >= 30: return "background-color:#2a1800;color:#ffb347"
            except Exception:
                pass
            return "color:#555"

        def _sl_ev_color(v):
            return "color:#ff3b3b;font-weight:700" if str(v).strip() else ""

        _sl_sty = _sl_disp.style
        if "Score" in _sl_disp.columns:
            _sl_sty = _sl_sty.applymap(_sl_score_color, subset=["Score"])
        if "EventFlag" in _sl_disp.columns:
            _sl_sty = _sl_sty.applymap(_sl_ev_color, subset=["EventFlag"])

        st.dataframe(_sl_sty, use_container_width=True, hide_index=True, height=380)

        # Download button
        try:
            _sl_csv = _sig_log_df.to_csv(index=False)
            st.download_button(
                "⬇ Download full log (CSV)",
                data=_sl_csv,
                file_name="monarch_signal_log.csv",
                mime="text/csv",
                key="dl_signal_log"
            )
        except Exception:
            pass

    with _sl_tab_setup:
        if "SetupType" in _sig_log_df.columns:
            _sl_grp_setup = (
                _sig_log_df.groupby("SetupType")
                .agg(
                    Signals=("Score", "count"),
                    AvgScore=("Score", lambda x: round(x.mean(), 1)),
                    AvgRR=("RR", lambda x: round(pd.to_numeric(x, errors="coerce").mean(), 2)),
                    EventFlagged=("EventFlag", lambda x: (x.astype(str).str.len() > 0).sum()),
                )
                .reset_index()
            )
            st.dataframe(_sl_grp_setup, use_container_width=True, hide_index=True)
            st.caption(
                "AvgRR = mean reward-to-risk across all signals of that type. "
                "EventFlagged = how many had a corporate event within ±3 days."
            )
        else:
            st.info("SetupType column not found in log.")

    with _sl_tab_sector:
        if "Sector" in _sig_log_df.columns:
            _sl_grp_sec = (
                _sig_log_df.groupby("Sector")
                .agg(
                    Signals=("Score", "count"),
                    AvgScore=("Score", lambda x: round(x.mean(), 1)),
                    Tickers=("Ticker", lambda x: ", ".join(sorted(set(x))[:5])),
                )
                .sort_values("Signals", ascending=False)
                .reset_index()
            )
            st.dataframe(_sl_grp_sec, use_container_width=True, hide_index=True)
            st.caption(
                "High signal count in a single sector = potential concentration risk. "
                "Cross-reference with the sector cap toggle in the sidebar."
            )
        else:
            st.info("Sector column not found in log.")

    with _sl_tab_regime:
        if "Regime" in _sig_log_df.columns:
            _sl_grp_reg = (
                _sig_log_df.groupby("Regime")
                .agg(
                    Signals=("Score", "count"),
                    AvgScore=("Score", lambda x: round(x.mean(), 1)),
                    AvgRR=("RR", lambda x: round(pd.to_numeric(x, errors="coerce").mean(), 2)),
                    SetupMix=("SetupType", lambda x: x.value_counts().to_dict()),
                )
                .reset_index()
            )
            st.dataframe(_sl_grp_reg, use_container_width=True, hide_index=True)
            st.caption(
                "BEAR regime signals should have lower AvgScore (market headwind applies penalty). "
                "If BEAR AvgScore ≈ BULL AvgScore, the regime adjustment may need recalibration."
            )
        else:
            st.info("Regime column not found in log.")

    # ── Rolling signal volume chart ─────────────────────────────────────────
    st.divider()
    st.subheader("📈 Signal Volume Over Time")
    if "Timestamp" in _sig_log_df.columns:
        try:
            _sl_ts = pd.to_datetime(_sig_log_df["Timestamp"], errors="coerce")
            _sl_daily = (
                _sig_log_df.assign(_Date=_sl_ts.dt.date)
                .groupby("_Date")
                .agg(Signals=("Score", "count"), AvgScore=("Score", "mean"))
                .reset_index()
                .rename(columns={"_Date": "Date"})
            )
            if not _sl_daily.empty:
                _fig_sl = go.Figure()
                _fig_sl.add_trace(go.Bar(
                    x=_sl_daily["Date"], y=_sl_daily["Signals"],
                    name="Signals/day",
                    marker_color="#ff8c00",
                ))
                _fig_sl.add_trace(go.Scatter(
                    x=_sl_daily["Date"], y=_sl_daily["AvgScore"],
                    name="Avg Score", yaxis="y2",
                    line=dict(color="#00ccff", width=2),
                ))
                _fig_sl.update_layout(
                    height=280,
                    plot_bgcolor="#0a0a0a", paper_bgcolor="#0a0a0a",
                    font=dict(color="#888", size=9, family="IBM Plex Mono"),
                    yaxis=dict(title="Signals/day", gridcolor="#1a1a1a", color="#888"),
                    yaxis2=dict(title="Avg Score", overlaying="y", side="right",
                                color="#00ccff", gridcolor="#0a0a0a"),
                    legend=dict(font=dict(color="#888", size=9), bgcolor="rgba(0,0,0,0)"),
                    margin=dict(t=20, b=30, l=40, r=40),
                )
                st.plotly_chart(_fig_sl, use_container_width=True)
        except Exception as _sl_chart_err:
            st.caption(f"Chart unavailable: {_sl_chart_err}")
    else:
        st.caption("Timestamp column missing — chart unavailable.")
