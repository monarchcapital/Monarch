# ===========================================================
# MONARCH PRO — FUNDAMENTAL RESEARCH TERMINAL
# ============================================================
# Pure analyst-grade fundamental research page.
# Data source: yfinance (Yahoo Finance) — free, no auth needed.
#
# Sections:
#   1. SNAPSHOT       — current valuation scorecard
#   2. VALUATION      — P/E, P/B, P/S, EV/EBITDA vs sector peers
#   3. PROFITABILITY  — ROE, ROCE, ROA, margins over time
#   4. GROWTH         — Revenue, EBITDA, EPS CAGR (1y/3y/5y)
#   5. FINANCIAL HEALTH — D/E, Interest Coverage, Current Ratio, Altman Z
#   6. CASH FLOW      — OCF, FCF, FCF yield, Capex intensity
#   7. DIVIDENDS      — yield, payout ratio, DPS history
#   8. EARNINGS       — quarterly EPS trend, beat/miss, surprise %
#   9. PEER COMPARISON — rank vs sector on all key metrics
#  10. ANALYST RATINGS — consensus, price targets, upside/downside
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    layout="wide",
    page_title="MONARCH — Fundamental Research",
    page_icon="📊",
    initial_sidebar_state="expanded"
)

# ============================================================
# BLOOMBERG TERMINAL CSS
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600;700&display=swap');

:root {
    --bb-bg:      #0a0a0a;
    --bb-surface: #111111;
    --bb-border:  #2a2a2a;
    --bb-amber:   #ff8c00;
    --bb-amber2:  #ffb347;
    --bb-green:   #00d084;
    --bb-green2:  #00ff88;
    --bb-red:     #ff3b3b;
    --bb-blue:    #1e90ff;
    --bb-cyan:    #00ccff;
    --bb-white:   #e8e8e8;
    --bb-white2:  #c8c8c8;
    --bb-muted:   #888888;
    --bb-dim:     #555555;
    --bb-mono:    'IBM Plex Mono', monospace;
}

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > section,
.main .block-container {
    background-color: var(--bb-bg) !important;
    color: var(--bb-white) !important;
    font-family: var(--bb-mono) !important;
}
p, span, div, label, li, caption,
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] span,
[data-testid="stMarkdownContainer"] li {
    color: var(--bb-white2) !important;
    font-family: var(--bb-mono) !important;
}
[data-testid="stSidebar"], [data-testid="stSidebar"] > div {
    background-color: #060606 !important;
    border-right: 1px solid var(--bb-border) !important;
}
[data-testid="stSidebar"] label {
    color: #c8c8c8 !important; font-size: 0.65rem !important;
    letter-spacing: 0.08em !important; text-transform: uppercase !important;
}
[data-testid="stSidebar"] p, [data-testid="stSidebar"] span { color: #c8c8c8 !important; font-size: 0.67rem !important; }
[data-testid="stSidebar"] .stCaption p,
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p { color: #666 !important; font-size: 0.58rem !important; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    color: var(--bb-amber) !important; font-size: 0.72rem !important;
    border-bottom: 1px solid #2a2a2a !important; padding-bottom: 4px !important;
}
[data-testid="stSidebar"] [data-baseweb="tag"] { background: #1a1200 !important; border: 1px solid var(--bb-amber) !important; }
[data-testid="stSidebar"] [data-baseweb="tag"] span { color: var(--bb-amber) !important; }
h1 { color: var(--bb-amber) !important; font-size: 1.05rem !important;
     font-weight: 700 !important; letter-spacing: 0.18em !important;
     text-transform: uppercase !important;
     border-bottom: 2px solid var(--bb-amber) !important;
     padding-bottom: 6px !important; margin-bottom: 12px !important; }
h2 { color: var(--bb-amber2) !important; font-size: 0.85rem !important;
     font-weight: 600 !important; letter-spacing: 0.12em !important;
     text-transform: uppercase !important; }
h3 { color: var(--bb-white) !important; font-size: 0.75rem !important;
     font-weight: 600 !important; letter-spacing: 0.1em !important;
     text-transform: uppercase !important; }
[data-testid="metric-container"] {
    background: var(--bb-surface) !important;
    border: 1px solid var(--bb-border) !important;
    border-left: 3px solid var(--bb-amber) !important;
    padding: 8px 12px !important; border-radius: 0 !important;
}
[data-testid="stMetricLabel"] p {
    color: var(--bb-muted) !important; font-size: 0.58rem !important;
    letter-spacing: 0.12em !important; text-transform: uppercase !important;
}
[data-testid="stMetricValue"] {
    color: var(--bb-amber) !important; font-size: 1.0rem !important; font-weight: 700 !important;
}
[data-testid="stMetricDelta"] { font-size: 0.60rem !important; }
[data-testid="stDataFrame"] {
    border: 1px solid var(--bb-amber) !important; border-radius: 0 !important;
}
.stDataFrame thead tr th {
    background-color: #1a1200 !important; color: var(--bb-amber) !important;
    font-family: var(--bb-mono) !important; font-size: 0.60rem !important;
    font-weight: 700 !important; letter-spacing: 0.14em !important;
    text-transform: uppercase !important; border-bottom: 2px solid var(--bb-amber) !important;
    border-right: 1px solid #2a2a2a !important; padding: 6px 10px !important;
    white-space: nowrap !important;
}
.stDataFrame tbody tr td {
    background-color: #0d0d0d !important; color: var(--bb-white) !important;
    font-family: var(--bb-mono) !important; font-size: 0.68rem !important;
    border-bottom: 1px solid #1a1a1a !important; padding: 4px 10px !important;
    white-space: nowrap !important;
}
.stDataFrame tbody tr:nth-child(odd)  td { background-color: #111111 !important; }
.stDataFrame tbody tr:nth-child(even) td { background-color: #0d0d0d !important; }
.stDataFrame tbody tr:hover td { background-color: #1f1400 !important; color: var(--bb-amber) !important; }
.stButton > button {
    background: #140e00 !important; color: var(--bb-amber) !important;
    border: 1px solid var(--bb-amber) !important; border-radius: 0 !important;
    font-family: var(--bb-mono) !important; font-size: 0.70rem !important;
    font-weight: 600 !important; letter-spacing: 0.1em !important;
    text-transform: uppercase !important; padding: 6px 18px !important;
}
.stButton > button:hover { background: var(--bb-amber) !important; color: #000 !important; }
.stSelectbox > div > div, .stTextInput > div > div, .stMultiSelect > div > div {
    background: var(--bb-surface) !important; border: 1px solid #3a3a3a !important;
    border-radius: 0 !important; color: var(--bb-white) !important;
    font-family: var(--bb-mono) !important; font-size: 0.72rem !important;
}
.stSelectbox label, .stTextInput label, .stMultiSelect label {
    color: var(--bb-muted) !important; font-size: 0.62rem !important;
    letter-spacing: 0.1em !important; text-transform: uppercase !important;
}
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
.stTabs [data-baseweb="tab"]:hover { background: #1a1200 !important; color: var(--bb-amber2) !important; }
.stTabs [aria-selected="true"] {
    background: #1a1200 !important; color: var(--bb-amber) !important;
    border-bottom: 3px solid var(--bb-amber) !important; font-weight: 700 !important;
}
.stTabs [data-baseweb="tab-panel"] { background: var(--bb-bg) !important; padding: 0 !important; }
hr { border-color: #1e1e1e !important; margin: 10px 0 !important; }
[data-testid="stAlert"] { border-radius: 0 !important; font-family: var(--bb-mono) !important; font-size: 0.70rem !important; }
[data-testid="stAlert"] p { font-size: 0.70rem !important; color: inherit !important; }
.stCaption, [data-testid="stCaptionContainer"] p {
    color: var(--bb-dim) !important; font-size: 0.60rem !important; letter-spacing: 0.06em !important;
}
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
[data-testid="stExpander"] [data-testid="stMarkdownContainer"] p,
[data-testid="stExpander"] [data-testid="stMarkdownContainer"] li,
[data-testid="stExpander"] [data-testid="stMarkdownContainer"] td {
    color: var(--bb-white2) !important; font-size: 0.68rem !important;
}
[data-testid="stExpander"] [data-testid="stMarkdownContainer"] th {
    color: var(--bb-amber) !important; background: #1a1200 !important;
}
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
    color: var(--bb-white2) !important; padding: 4px 10px !important; border: 1px solid #1e1e1e !important;
}
[data-testid="stMarkdownContainer"] tr:nth-child(even) td { background: #0d0d0d !important; }
[data-testid="stMarkdownContainer"] strong { color: var(--bb-amber) !important; }
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--bb-bg); }
::-webkit-scrollbar-thumb { background: #333; border-radius: 0; }
::-webkit-scrollbar-thumb:hover { background: var(--bb-amber); }

/* ══ FORCE SIDEBAR ALWAYS VISIBLE ══ */
[data-testid="stSidebar"] {
    display: block !important;
    visibility: visible !important;
    opacity: 1 !important;
    width: 21rem !important;
    min-width: 200px !important;
    transform: none !important;
    position: relative !important;
}
[data-testid="stSidebar"] > div:first-child {
    display: block !important;
    visibility: visible !important;
    opacity: 1 !important;
}
/* Force the sidebar collapse/expand button visible */
[data-testid="stSidebarCollapseButton"] {
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
}
button[kind="header"] {
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
}

/* ── PRICE DELTA COLORS ── */
[data-testid="stMetricDeltaIcon-Up"]  { color: #00d084 !important; }
[data-testid="stMetricDeltaIcon-Down"] { color: #ff3b3b !important; }
[data-testid="stMetricDelta"]:has([data-testid="stMetricDeltaIcon-Up"]) span  { color: #00d084 !important; }
[data-testid="stMetricDelta"]:has([data-testid="stMetricDeltaIcon-Down"]) span { color: #ff3b3b !important; }
/* ── FONT SIZE INCREASES ── */
[data-testid="stMetricValue"]   { font-size: 1.35rem !important; font-weight: 700 !important; }
[data-testid="stMetricLabel"] p { font-size: 0.78rem !important; }
[data-testid="stMetricDelta"]   { font-size: 0.82rem !important; }
.stDataFrame thead tr th        { font-size: 0.80rem !important; }
.stDataFrame tbody tr td        { font-size: 0.90rem !important; }
.stButton > button              { font-size: 0.90rem !important; }
/* ── NAV ICON TEXT fix ── */
[data-testid="stSidebarNavLink"] [data-testid="stIconMaterial"] {
    font-size: 0 !important;
    line-height: 0 !important;
    color: transparent !important;
    overflow: hidden !important;
    max-width: 20px !important;
}

</style>""", unsafe_allow_html=True)

# ── Terminal Header ──
st.markdown("""
<div style="background:#ff8c00;color:#000;font-family:'IBM Plex Mono',monospace;
font-size:0.65rem;font-weight:700;letter-spacing:0.18em;padding:5px 14px;
display:flex;justify-content:space-between;margin-bottom:12px;">
  <span>◼ MONARCH PRO — FUNDAMENTAL RESEARCH TERMINAL</span>
  <span>NSE · INDIA · POWERED BY YFINANCE</span>
</div>""", unsafe_allow_html=True)


# ============================================================
# HELPERS — Bloomberg Plotly theme
# ============================================================
BB_LAYOUT = dict(
    plot_bgcolor="#000000",
    paper_bgcolor="#0a0a0a",
    font=dict(color="#e8e8e8", family="IBM Plex Mono", size=10),
    xaxis=dict(gridcolor="#1a1a1a", zeroline=False, tickfont=dict(color="#888", size=9)),
    yaxis=dict(gridcolor="#1a1a1a", zeroline=False, tickfont=dict(color="#888", size=9)),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#e8e8e8", size=10),
                bordercolor="#2a2a2a", borderwidth=1),
    hoverlabel=dict(bgcolor="#1a1200", font_color="#ff8c00",
                    font_family="IBM Plex Mono", font_size=11, bordercolor="#ff8c00"),
    margin=dict(t=40, b=30, l=50, r=20),
)

def bb_chart(fig, height=320):
    fig.update_layout(height=height, **BB_LAYOUT)
    return fig

def section_header(title, subtitle=""):
    st.markdown(f"""
<div style="background:#1a1200;border-top:2px solid #ff8c00;border-bottom:1px solid #2a2a2a;
     padding:7px 14px;display:flex;justify-content:space-between;align-items:center;
     font-family:'IBM Plex Mono',monospace;margin:14px 0 6px 0;">
  <span style="color:#ff8c00;font-size:.72rem;font-weight:700;letter-spacing:.14em;">
    ◼ {title.upper()}
  </span>
  <span style="color:#555;font-size:.60rem;">{subtitle}</span>
</div>""", unsafe_allow_html=True)

def score_badge(val, good_thresh, bad_thresh, fmt=".1f", invert=False):
    """Returns coloured HTML badge."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return '<span style="color:#555">N/A</span>'
    if invert:
        good = val <= good_thresh
        bad  = val >= bad_thresh
    else:
        good = val >= good_thresh
        bad  = val <= bad_thresh
    color = "#00ff88" if good else ("#ff3b3b" if bad else "#ffb347")
    return f'<span style="color:{color};font-weight:700;">{val:{fmt}}</span>'

def safe(v, default=None):
    """Safe float conversion."""
    try:
        f = float(v)
        return None if np.isnan(f) or np.isinf(f) else f
    except:
        return default

def pct(v):
    """Format as %."""
    return f"{v*100:.1f}%" if v is not None else "N/A"

def fmt_cr(v):
    """Format in Crores."""
    if v is None: return "N/A"
    if abs(v) >= 1e11: return f"₹{v/1e11:.1f}L Cr"
    if abs(v) >= 1e7:  return f"₹{v/1e7:.0f} Cr"
    return f"₹{v:.0f}"

def cagr(series, years):
    """CAGR over N years from a list/series of annual values (oldest→newest)."""
    s = [x for x in series if x and not np.isnan(x) and x > 0]
    if len(s) < 2: return None
    n = min(years, len(s)-1)
    if s[-(n+1)] <= 0: return None
    return (s[-1] / s[-(n+1)]) ** (1/n) - 1

def bar_chart(x, y, title, yaxis_title, color_positive=True, annotations=None):
    colors = ["#00d084" if v >= 0 else "#ff3b3b" for v in y] if color_positive else ["#ff8c00"] * len(y)
    fig = go.Figure(go.Bar(
        x=x, y=y, marker_color=colors, marker_line_width=0,
        text=[f"{v:.1f}" for v in y], textposition="outside",
        textfont=dict(size=9, color="#e8e8e8"),
    ))
    fig.update_layout(title=dict(text=title, font=dict(color="#ff8c00", size=11)), yaxis_title=yaxis_title)
    return bb_chart(fig)

def line_chart(x_list, y_list, names, title, yaxis_title, colors=None):
    _colors = colors or ["#ff8c00","#1e90ff","#00d084","#cc88ff","#ffb347"]
    fig = go.Figure()
    for i, (x, y, name) in enumerate(zip(x_list, y_list, names)):
        fig.add_trace(go.Scatter(
            x=x, y=y, name=name, mode="lines+markers",
            line=dict(color=_colors[i % len(_colors)], width=2),
            marker=dict(size=5),
        ))
    fig.update_layout(title=dict(text=title, font=dict(color="#ff8c00", size=11)), yaxis_title=yaxis_title)
    return bb_chart(fig)


# ============================================================
# DATA FETCHING — yfinance wrapper with cache
# ============================================================
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_fundamentals(ticker_ns: str):
    """
    Fetches all fundamental data from yfinance.
    ticker_ns: NSE symbol like 'RELIANCE' — auto-appended with .NS
    Returns dict with info, financials, balance sheet, cashflow, earnings.
    """
    yt = ticker_ns.upper()
    if not yt.endswith(".NS") and not yt.endswith(".BO"):
        yt = yt + ".NS"

    tk = yf.Ticker(yt)
    out = {
        "ticker": yt,
        "info": {},
        "income_annual": pd.DataFrame(),
        "income_quarterly": pd.DataFrame(),
        "balance_annual": pd.DataFrame(),
        "balance_quarterly": pd.DataFrame(),
        "cashflow_annual": pd.DataFrame(),
        "cashflow_quarterly": pd.DataFrame(),
        "earnings_quarterly": pd.DataFrame(),
        "dividends": pd.Series(dtype=float),
        "price_history": pd.DataFrame(),
        "analyst_price_targets": {},
        "recommendations": pd.DataFrame(),
        "error": None,
    }
    try:
        out["info"] = tk.info or {}
    except Exception as e:
        out["error"] = str(e)
        return out

    try: out["income_annual"]      = tk.financials
    except: pass
    try: out["income_quarterly"]   = tk.quarterly_financials
    except: pass
    try: out["balance_annual"]     = tk.balance_sheet
    except: pass
    try: out["balance_quarterly"]  = tk.quarterly_balance_sheet
    except: pass
    try: out["cashflow_annual"]    = tk.cashflow
    except: pass
    try: out["cashflow_quarterly"] = tk.quarterly_cashflow
    except: pass
    try: out["dividends"]          = tk.dividends
    except: pass
    try: out["price_history"]      = tk.history(period="5y", interval="1mo", auto_adjust=True)
    except: pass
    try: out["analyst_price_targets"] = tk.analyst_price_targets
    except: pass
    try: out["recommendations"]    = tk.recommendations
    except: pass

    return out


def get_row(df, *keys):
    """Get first matching row from a DataFrame by trying multiple key names."""
    if df is None or df.empty:
        return pd.Series(dtype=float)
    for k in keys:
        for idx in df.index:
            if k.lower() in str(idx).lower():
                return df.loc[idx]
    return pd.Series(dtype=float)


def annual_series(df, *keys):
    """Returns (dates, values) for an annual financial row."""
    row = get_row(df, *keys)
    if row.empty:
        return [], []
    dates = [str(d.year) if hasattr(d, 'year') else str(d) for d in row.index]
    vals  = [safe(v) for v in row.values]
    # sort ascending
    pairs = sorted(zip(dates, vals), key=lambda x: x[0])
    return [p[0] for p in pairs], [p[1] for p in pairs]


def quarterly_series(df, *keys):
    """Returns (quarters, values) for a quarterly financial row."""
    row = get_row(df, *keys)
    if row.empty:
        return [], []
    dates = [str(d)[:7] if hasattr(d, 'year') else str(d)[:7] for d in row.index]
    vals  = [safe(v) for v in row.values]
    pairs = sorted(zip(dates, vals), key=lambda x: x[0])
    return [p[0] for p in pairs], [p[1] for p in pairs]


# ============================================================
# PEER SECTOR MAPS — NSE symbols by sector
# ============================================================
SECTOR_PEERS = {
    "IT":       ["TCS","INFY","WIPRO","HCLTECH","TECHM","LTIM","MPHASIS","COFORGE","PERSISTENT"],
    "Bank":     ["HDFCBANK","ICICIBANK","KOTAKBANK","AXISBANK","INDUSINDBK","FEDERALBNK","IDFCFIRSTB"],
    "Pharma":   ["SUNPHARMA","DRREDDY","CIPLA","DIVISLAB","TORNTPHARM","AUROPHARMA","APOLLOHOSP"],
    "Auto":     ["MARUTI","TATAMOTORS","M&M","BAJAJ-AUTO","HEROMOTOCO","EICHERMOT","TVSMOTORS"],
    "FMCG":     ["HINDUNILVR","ITC","NESTLEIND","BRITANNIA","DABUR","MARICO","GODREJCP"],
    "Metal":    ["TATASTEEL","JSWSTEEL","HINDALCO","SAIL","VEDL","COALINDIA","NMDC"],
    "Energy":   ["RELIANCE","ONGC","NTPC","POWERGRID","BPCL","IOC","GAIL"],
    "Infra":    ["LT","ADANIPORTS","ULTRACEMCO","SHREECEM","IRFC","RVNL"],
    "Realty":   ["DLF","LODHA","OBEROIRLTY","PHOENIXLTD"],
    "PSUBank":  ["SBIN","BANKBARODA","PNB","CANBK"],
}

STOCK_SECTOR = {}
for sector, stocks in SECTOR_PEERS.items():
    for s in stocks:
        STOCK_SECTOR[s] = sector


# ============================================================
# SIDEBAR — STOCK SELECTOR
# ============================================================
st.sidebar.header("🔍 Stock Selector")

# Quick-entry text input + sector dropdown
all_stocks = sorted(set(s for peers in SECTOR_PEERS.values() for s in peers))
sector_sel = st.sidebar.selectbox("Sector", ["All"] + sorted(SECTOR_PEERS.keys()), key="fund_sector")
if sector_sel != "All":
    filtered_stocks = sorted(SECTOR_PEERS[sector_sel])
else:
    filtered_stocks = all_stocks

manual_input = st.sidebar.text_input("Or type NSE symbol", placeholder="e.g. RELIANCE", key="fund_manual").upper().strip()
if manual_input:
    ticker_sym = manual_input
else:
    ticker_sym = st.sidebar.selectbox("Select Stock", filtered_stocks, key="fund_stock")

# Peers — auto-derived from ticker sector, user can override
sector_name  = STOCK_SECTOR.get(ticker_sym, "")
peer_options = SECTOR_PEERS.get(sector_name, [])
default_peers = [p for p in peer_options if p != ticker_sym][:4]

if sector_name:
    st.sidebar.caption(f"◼ Sector detected: {sector_name}")

peers_sel = st.sidebar.multiselect(
    "Peers (auto by sector — override OK)",
    options=[p for p in peer_options if p != ticker_sym],
    default=default_peers,
    max_selections=5,
    key="fund_peers"
)

st.sidebar.markdown("---")
st.sidebar.caption(f"Data source: Yahoo Finance (yfinance)  ·  Prices in ₹  ·  Financials in ₹ Cr")
st.sidebar.caption("Refresh: data cached 1 hour")

load_btn = st.sidebar.button("◼ LOAD FUNDAMENTALS", use_container_width=True)


# ============================================================
# MAIN PANEL
# ============================================================

# ── Initialize all session state keys upfront ──
if "fund_data"       not in st.session_state: st.session_state.fund_data       = None
if "fund_ticker"     not in st.session_state: st.session_state.fund_ticker     = None
if "fund_sector_val" not in st.session_state: st.session_state.fund_sector_val = None
if "peer_data"       not in st.session_state: st.session_state.peer_data       = {}

if not load_btn and st.session_state.fund_data is None:
    st.markdown("""
<div style="background:#0d0d0d;border:1px solid #2a2a2a;border-left:4px solid #ff8c00;
padding:24px 28px;font-family:'IBM Plex Mono',monospace;margin-top:20px;">
  <div style="color:#ff8c00;font-size:.9rem;font-weight:700;letter-spacing:.12em;margin-bottom:10px;">
    ◼ MONARCH FUNDAMENTAL RESEARCH TERMINAL
  </div>
  <div style="color:#c8c8c8;font-size:.72rem;line-height:1.8;">
    Select a stock from the sidebar and click <b style="color:#ffb347">LOAD FUNDAMENTALS</b> to begin.<br><br>
    This terminal provides:<br>
    &nbsp; • Snapshot — quality score, DuPont, Piotroski F-Score, red flags<br>
    &nbsp; • Valuation — P/E, P/B, EV/EBITDA, Reverse DCF (implied growth)<br>
    &nbsp; • Profitability — ROE, ROCE, margins, DuPont decomposition<br>
    &nbsp; • Growth — Revenue, EPS, FCF CAGR · operating leverage<br>
    &nbsp; • Financial Health — D/E, Interest Cover, Cash Conversion Cycle, Altman Z<br>
    &nbsp; • Cash Flow — OCF, FCF yield, earnings quality ratio<br>
    &nbsp; • Dividends — yield, payout ratio, DPS history<br>
    &nbsp; • Earnings — quarterly EPS trend, beat/miss, surprise %<br>
    &nbsp; • Peer Ranking — percentile rank on every metric vs sector<br>
    &nbsp; • Analyst Ratings — consensus, price targets, timeframe to target
  </div>
</div>""", unsafe_allow_html=True)
    st.stop()

# ── Auto-derive peers + parallel fetch ──
_auto_peers     = [p for p in SECTOR_PEERS.get(STOCK_SECTOR.get(ticker_sym, ""), []) if p != ticker_sym][:4]
_peers_to_load  = peers_sel if peers_sel else _auto_peers
_ticker_changed = st.session_state.fund_ticker != ticker_sym
_peers_changed  = set(_peers_to_load) != set(st.session_state.peer_data.keys())

if load_btn or st.session_state.fund_data is None or _ticker_changed or _peers_changed:
    with st.spinner(f"Fetching {ticker_sym} + {len(_peers_to_load)} peers in parallel..."):
        all_to_fetch = [ticker_sym] + _peers_to_load
        results = {}
        with ThreadPoolExecutor(max_workers=min(6, len(all_to_fetch))) as ex:
            future_map = {ex.submit(fetch_fundamentals, t): t for t in all_to_fetch}
            for future in as_completed(future_map):
                t = future_map[future]
                try:    results[t] = future.result()
                except: results[t] = fetch_fundamentals(t)
        st.session_state.fund_data       = results[ticker_sym]
        st.session_state.fund_ticker     = ticker_sym
        st.session_state.fund_sector_val = sector_name
        st.session_state.peer_data       = {p: results[p] for p in _peers_to_load if p in results}

data   = st.session_state.fund_data
info   = data["info"]
peers  = st.session_state.peer_data
t_sym  = st.session_state.fund_ticker

if data.get("error"):
    st.error(f"Could not load data for {ticker_sym}: {data['error']}")
    st.stop()

# ── Stock identity bar ──
comp_name  = info.get("longName") or info.get("shortName") or t_sym
sector_str = info.get("sector","") or sector_name
industry   = info.get("industry","")
mktcap     = safe(info.get("marketCap"))
curr_price = safe(info.get("currentPrice") or info.get("regularMarketPrice"))
_52w_hi    = safe(info.get("fiftyTwoWeekHigh"))
_52w_lo    = safe(info.get("fiftyTwoWeekLow"))
pct_from_hi = ((curr_price - _52w_hi) / _52w_hi * 100) if curr_price and _52w_hi else None

st.markdown(f"""
<div style="background:#0d0d0d;border:1px solid #2a2a2a;border-left:4px solid #ff8c00;
     padding:10px 16px;font-family:'IBM Plex Mono',monospace;margin-bottom:12px;">
  <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px;">
    <div>
      <span style="color:#ff8c00;font-size:1.1rem;font-weight:700;letter-spacing:.1em;">{t_sym}</span>
      <span style="color:#888;font-size:.72rem;margin-left:12px;">{comp_name}</span>
    </div>
    <div style="display:flex;gap:24px;align-items:center;flex-wrap:wrap;">
      <span style="color:#e8e8e8;font-size:.8rem;font-weight:600;">
        ₹{curr_price:.2f}</span>
      <span style="color:#555;font-size:.65rem;">{sector_str}  ·  {industry}</span>
      <span style="color:#888;font-size:.65rem;">MCap: {fmt_cr(mktcap)}</span>
      <span style="color:{'#ff3b3b' if pct_from_hi and pct_from_hi < -20 else '#888'};font-size:.65rem;">
        {f'{pct_from_hi:.1f}% from 52w high' if pct_from_hi else ''}</span>
    </div>
  </div>
</div>""", unsafe_allow_html=True)


# ============================================================
# TAB STRUCTURE
# ============================================================
(tab_snap, tab_val, tab_prof, tab_growth,
 tab_health, tab_cf, tab_div, tab_earn,
 tab_peer, tab_analyst) = st.tabs([
    "⬡ SNAPSHOT",
    "📐 VALUATION",
    "📈 PROFITABILITY",
    "🚀 GROWTH",
    "🏦 HEALTH",
    "💵 CASH FLOW",
    "💰 DIVIDENDS",
    "📊 EARNINGS",
    "🏆 PEER RANK",
    "🎯 ANALYSTS",
])


# ══════════════════════════════════════════════════════════════
# TAB 1 — SNAPSHOT SCORECARD
# ══════════════════════════════════════════════════════════════
with tab_snap:
    section_header("ANALYST SCORECARD", "LIVE FUNDAMENTAL SNAPSHOT")

    # Pull key metrics from info dict
    pe      = safe(info.get("trailingPE"))
    fwd_pe  = safe(info.get("forwardPE"))
    pb      = safe(info.get("priceToBook"))
    ps      = safe(info.get("priceToSalesTrailing12Months"))
    ev_ebit = safe(info.get("enterpriseToEbitda"))
    roe     = safe(info.get("returnOnEquity"))
    roa     = safe(info.get("returnOnAssets"))
    gross_m = safe(info.get("grossMargins"))
    op_m    = safe(info.get("operatingMargins"))
    net_m   = safe(info.get("profitMargins"))
    rev_gr  = safe(info.get("revenueGrowth"))
    earn_gr = safe(info.get("earningsGrowth"))
    debt_eq = safe(info.get("debtToEquity"))
    cr      = safe(info.get("currentRatio"))
    div_yld = safe(info.get("dividendYield"))
    payout  = safe(info.get("payoutRatio"))
    beta    = safe(info.get("beta"))
    eps_ttm = safe(info.get("trailingEps"))
    eps_fwd = safe(info.get("forwardEps"))
    book_v  = safe(info.get("bookValue"))
    fcf     = safe(info.get("freeCashflow"))
    ocf     = safe(info.get("operatingCashflow"))
    revenue = safe(info.get("totalRevenue"))
    ev      = safe(info.get("enterpriseValue"))
    # Additional derived metrics
    total_debt  = safe(info.get("totalDebt"))
    total_cash  = safe(info.get("totalCash"))
    net_debt    = (total_debt - total_cash) if total_debt is not None and total_cash is not None else None
    ev_rev      = ev / revenue if ev and revenue and revenue != 0 else None
    net_debt_ev = net_debt / (ev_ebit * revenue * op_m) if (net_debt and ev_ebit and revenue and op_m and revenue * op_m != 0) else None
    # EBITDA proxy for Net Debt/EBITDA
    ebitda_proxy = revenue * op_m if revenue and op_m else None
    nd_ebitda    = net_debt / ebitda_proxy if net_debt is not None and ebitda_proxy and ebitda_proxy != 0 else None

    # ── METRIC ROWS ──
    st.markdown("#### VALUATION")
    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.metric("Trailing P/E",   f"{pe:.1f}x"     if pe    else "N/A")
    c2.metric("Forward P/E",    f"{fwd_pe:.1f}x"  if fwd_pe else "N/A",
              delta=f"{fwd_pe-pe:.1f}x vs TTM" if pe and fwd_pe else None)
    c3.metric("Price/Book",     f"{pb:.2f}x"     if pb    else "N/A")
    c4.metric("Price/Sales",    f"{ps:.2f}x"     if ps    else "N/A")
    c5.metric("EV/EBITDA",      f"{ev_ebit:.1f}x" if ev_ebit else "N/A")
    c6.metric("EV/Revenue",     f"{ev_rev:.2f}x"  if ev_rev  else "N/A")

    st.markdown("#### PROFITABILITY")
    p1,p2,p3,p4,p5,p6 = st.columns(6)
    p1.metric("ROE",             pct(roe),   delta="↑" if roe and roe > 0.15 else None)
    p2.metric("ROA",             pct(roa))
    p3.metric("Gross Margin",    pct(gross_m))
    p4.metric("Operating Margin",pct(op_m))
    p5.metric("Net Margin",      pct(net_m))
    p6.metric("EPS (TTM)",       f"₹{eps_ttm:.2f}" if eps_ttm else "N/A",
              delta=f"Fwd ₹{eps_fwd:.2f}" if eps_fwd else None)

    st.markdown("#### GROWTH · HEALTH · DEBT")
    g1,g2,g3,g4,g5,g6 = st.columns(6)
    g1.metric("Revenue Growth",  pct(rev_gr))
    g2.metric("Earnings Growth", pct(earn_gr))
    g3.metric("Debt/Equity",     f"{debt_eq/100:.2f}x" if debt_eq else "N/A")
    g4.metric("Net Debt",        fmt_cr(net_debt) if net_debt is not None else "Net Cash" if net_debt is not None else "N/A")
    g5.metric("Net Debt/EBITDA", f"{nd_ebitda:.1f}x" if nd_ebitda is not None else "N/A")
    g6.metric("FCF",             fmt_cr(fcf))

    # ── OVERALL SCORE CARD ──
    st.divider()
    section_header("FUNDAMENTAL QUALITY SCORE", "COMPOSITE RATING ACROSS 5 DIMENSIONS")

    def score_dim(label, value, good, bad, invert=False, weight=1.0):
        if value is None: return None, label, "N/A", "#555"
        if invert:
            norm = 1 - min(1, max(0, (value - good) / (bad - good + 1e-9)))
        else:
            norm = min(1, max(0, (value - bad) / (good - bad + 1e-9)))
        score = norm * 10 * weight
        color = "#00ff88" if norm >= 0.7 else "#ffb347" if norm >= 0.4 else "#ff3b3b"
        return score, label, f"{value*100:.1f}%" if abs(value) < 10 else f"{value:.1f}x", color

    dims = [
        score_dim("ROE",          roe,     0.20, 0.05),
        score_dim("Net Margin",   net_m,   0.15, 0.02),
        score_dim("Rev Growth",   rev_gr,  0.20, 0.0),
        score_dim("Debt/Equity",  (debt_eq/100) if debt_eq else None, 0.3, 2.0, invert=True),
        score_dim("Current Ratio",cr,      2.0,  1.0),
    ]

    valid_dims = [(s, l, v, c) for s, l, v, c in dims if s is not None]
    if valid_dims:
        total_score = sum(s for s, *_ in valid_dims) / len(valid_dims)
        overall_color = "#00ff88" if total_score >= 7 else "#ffb347" if total_score >= 5 else "#ff3b3b"

        sc_cols = st.columns(len(valid_dims) + 1)
        for i, (s, l, v, c) in enumerate(valid_dims):
            sc_cols[i].markdown(f"""
<div style="background:#0d0d0d;border:1px solid #2a2a2a;border-top:3px solid {c};
     padding:10px 8px;text-align:center;font-family:'IBM Plex Mono',monospace;">
  <div style="color:#555;font-size:.56rem;letter-spacing:.1em;text-transform:uppercase;">{l}</div>
  <div style="color:{c};font-size:1.1rem;font-weight:700;margin:4px 0;">{s:.1f}</div>
  <div style="color:#888;font-size:.60rem;">{v}</div>
</div>""", unsafe_allow_html=True)

        sc_cols[-1].markdown(f"""
<div style="background:#1a1200;border:1px solid #ff8c00;border-top:4px solid {overall_color};
     padding:10px 8px;text-align:center;font-family:'IBM Plex Mono',monospace;">
  <div style="color:#888;font-size:.56rem;letter-spacing:.1em;">OVERALL</div>
  <div style="color:{overall_color};font-size:1.4rem;font-weight:700;margin:4px 0;">{total_score:.1f}/10</div>
  <div style="color:#888;font-size:.60rem;">COMPOSITE</div>
</div>""", unsafe_allow_html=True)

    # ── 52-WEEK PRICE RANGE BAR ──
    st.divider()
    if curr_price and _52w_hi and _52w_lo:
        rng_pct = (curr_price - _52w_lo) / (_52w_hi - _52w_lo + 1e-9)
        bar_color = "#00ff88" if rng_pct > 0.7 else "#ffb347" if rng_pct > 0.4 else "#ff3b3b"
        st.markdown(f"""
<div style="font-family:'IBM Plex Mono',monospace;margin:10px 0;">
  <div style="display:flex;justify-content:space-between;color:#555;font-size:.60rem;margin-bottom:3px;">
    <span>52W LOW  ₹{_52w_lo:.2f}</span>
    <span style="color:{bar_color};">CMP ₹{curr_price:.2f}  ({rng_pct*100:.0f}th pct)</span>
    <span>52W HIGH  ₹{_52w_hi:.2f}</span>
  </div>
  <div style="background:#1a1a1a;height:6px;border-radius:0;position:relative;">
    <div style="background:{bar_color};width:{rng_pct*100:.1f}%;height:100%;"></div>
  </div>
</div>""", unsafe_allow_html=True)

    # ══════════════════════════════
    # PIOTROSKI F-SCORE
    # ══════════════════════════════
    st.divider()
    section_header("PIOTROSKI F-SCORE", "9-POINT ACCOUNTING QUALITY MODEL  ·  8–9 STRONG  ·  5–7 AVERAGE  ·  0–4 WEAK")
    try:
        _ia  = data["income_annual"]
        _ba  = data["balance_annual"]
        _cf  = data["cashflow_annual"]
        _ia2 = data.get("income_annual")

        _ni_d, _ni_v     = annual_series(_ia,  "Net Income","Net income")
        _ta_d, _ta_v     = annual_series(_ba,  "Total Assets")
        _ocf_d, _ocf_v   = annual_series(_cf,  "Operating Cash Flow","Cash Flow From Continuing Operating Activities")
        _ltd_d, _ltd_v   = annual_series(_ba,  "Long Term Debt","Total Debt","Long Term Debt And Capital Lease Obligation")
        _ca_d, _ca_v     = annual_series(_ba,  "Current Assets")
        _cl_d, _cl_v     = annual_series(_ba,  "Current Liabilities")
        _rev_d, _rev_v   = annual_series(_ia,  "Total Revenue","Revenue")
        _gp_d, _gp_v     = annual_series(_ia,  "Gross Profit")
        _sh_d, _sh_v     = annual_series(_ba,  "Common Stock","Share Issued","Ordinary Shares Number")
        _eq_d, _eq_v     = annual_series(_ba,  "Stockholders Equity","Total Equity","Common Stock Equity")
        _ebit_d, _ebit_v = annual_series(_ia,  "Operating Income","Ebit","EBIT")

        def _last(lst, n=0):
            return lst[-(1+n)] if lst and len(lst) > n else None

        ta0, ta1   = _last(_ta_v,0),  _last(_ta_v,1)
        ni0        = _last(_ni_v,0)
        ocf0       = _last(_ocf_v,0)
        ltd0, ltd1 = _last(_ltd_v,0), _last(_ltd_v,1)
        ca0, cl0   = _last(_ca_v,0),  _last(_cl_v,0)
        ca1, cl1   = _last(_ca_v,1),  _last(_cl_v,1)
        rev0, rev1 = _last(_rev_v,0), _last(_rev_v,1)
        gp0, gp1   = _last(_gp_v,0),  _last(_gp_v,1)
        sh0, sh1   = _last(_sh_v,0),  _last(_sh_v,1)
        eq0        = _last(_eq_v,0)
        ebit0      = _last(_ebit_v,0)

        roa0  = ni0/ta0   if ni0  and ta0  and ta0  != 0 else None
        roa1  = _last(_ni_v,1) / ta1 if _last(_ni_v,1) and ta1 and ta1 != 0 else None
        cfoa0 = ocf0/ta0  if ocf0 and ta0  and ta0  != 0 else None
        lev0  = ltd0/ta0  if ltd0 and ta0  and ta0  != 0 else None
        lev1  = ltd1/ta1  if ltd1 and ta1  and ta1  != 0 else None
        cr0   = ca0/cl0   if ca0  and cl0  and cl0  != 0 else None
        cr1   = ca1/cl1   if ca1  and cl1  and cl1  != 0 else None
        gm0   = gp0/rev0  if gp0  and rev0 and rev0 != 0 else None
        gm1   = gp1/rev1  if gp1  and rev1 and rev1 != 0 else None
        at0   = rev0/ta0  if rev0 and ta0  and ta0  != 0 else None
        at1   = rev1/ta1  if rev1 and ta1  and ta1  != 0 else None

        f_signals = [
            ("F1 ROA > 0",                roa0 is not None and roa0 > 0,             "Profitability"),
            ("F2 Operating CF > 0",       ocf0 is not None and ocf0 > 0,             "Profitability"),
            ("F3 ROA improving",          roa0 is not None and roa1 is not None and roa0 > roa1, "Profitability"),
            ("F4 Accruals (CF > ROA)",    cfoa0 is not None and roa0 is not None and cfoa0 > roa0, "Profitability"),
            ("F5 Leverage declining",     lev0 is not None and lev1 is not None and lev0 < lev1, "Leverage"),
            ("F6 Current ratio rising",   cr0  is not None and cr1  is not None and cr0  > cr1,  "Liquidity"),
            ("F7 No new shares issued",   sh0  is not None and sh1  is not None and sh0  <= sh1, "Dilution"),
            ("F8 Gross margin rising",    gm0  is not None and gm1  is not None and gm0  > gm1,  "Efficiency"),
            ("F9 Asset turnover rising",  at0  is not None and at1  is not None and at0  > at1,  "Efficiency"),
        ]
        f_score = sum(1 for _, v, _ in f_signals if v)
        f_color = "#00ff88" if f_score >= 8 else "#ffb347" if f_score >= 5 else "#ff3b3b"
        f_label = "STRONG" if f_score >= 8 else "AVERAGE" if f_score >= 5 else "WEAK"

        pf_cols = st.columns([1,3])
        with pf_cols[0]:
            st.markdown(f"""
<div style="background:#0d0d0d;border:2px solid {f_color};padding:16px 10px;text-align:center;font-family:'IBM Plex Mono';">
  <div style="color:#555;font-size:.56rem;letter-spacing:.1em;">PIOTROSKI</div>
  <div style="color:{f_color};font-size:2.2rem;font-weight:700;margin:6px 0;">{f_score}/9</div>
  <div style="color:{f_color};font-size:.60rem;font-weight:700;">{f_label}</div>
</div>""", unsafe_allow_html=True)
        with pf_cols[1]:
            f_rows = []
            for label, passed, category in f_signals:
                tick = "✓" if passed else "✗"
                color = "#00d084" if passed else "#ff3b3b"
                f_rows.append(f'<span style="color:{color};margin-right:16px;">{tick} {label}</span>')
            # 3 per row
            for i in range(0, len(f_rows), 3):
                st.markdown(" ".join(f_rows[i:i+3]), unsafe_allow_html=True)

    except Exception as e:
        st.info(f"Piotroski F-Score: insufficient data ({e})")

    # ══════════════════════════════
    # RED FLAG DETECTOR
    # ══════════════════════════════
    st.divider()
    section_header("RED FLAG DETECTOR", "AUTOMATED ACCOUNTING & FINANCIAL WARNING SIGNALS")
    try:
        flags = []
        # 1. FCF negative 2+ consecutive years
        _cf2   = data["cashflow_annual"]
        _ocf2_d, _ocf2_v = annual_series(_cf2, "Operating Cash Flow","Cash Flow From Continuing Operating Activities")
        _cap2_d, _cap2_v = annual_series(_cf2, "Capital Expenditure","Purchase Of Ppe","Purchase Of Property Plant And Equipment")
        if _ocf2_v and _cap2_v and len(_ocf2_v) == len(_cap2_v):
            _fcf2_v = [o+c if o is not None and c is not None else None for o,c in zip(_ocf2_v, _cap2_v)]
            neg_fcf = sum(1 for v in _fcf2_v[-3:] if v is not None and v < 0)
            if neg_fcf >= 2:
                flags.append(("⚠ FCF negative", f"Free cash flow negative in {neg_fcf} of last 3 years — earnings may not be cash-backed", "#ff3b3b"))

        # 2. Earnings quality (CFO / Net Income) < 0.8
        _ni2_d, _ni2_v = annual_series(data["income_annual"], "Net Income","Net income")
        if _ocf2_v and _ni2_v:
            _eq_ratio = _ocf2_v[-1] / _ni2_v[-1] if _ni2_v[-1] and _ni2_v[-1] != 0 and _ocf2_v[-1] else None
            if _eq_ratio is not None and _eq_ratio < 0.8:
                flags.append(("⚠ Low earnings quality", f"CFO/Net Income = {_eq_ratio:.2f}x  (< 0.8 indicates accrual-heavy earnings)", "#ff3b3b"))
            elif _eq_ratio is not None and _eq_ratio >= 1.0:
                flags.append(("✓ Strong earnings quality", f"CFO/Net Income = {_eq_ratio:.2f}x  (> 1.0 = fully cash-backed)", "#00d084"))

        # 3. Debt growing faster than revenue
        _td2_d, _td2_v = annual_series(data["balance_annual"], "Total Debt","Long Term Debt And Capital Lease Obligation")
        _rv2_d, _rv2_v = annual_series(data["income_annual"], "Total Revenue","Revenue")
        if _td2_v and len(_td2_v) >= 2 and _rv2_v and len(_rv2_v) >= 2:
            if _td2_v[-2] and _td2_v[-2] != 0 and _rv2_v[-2] and _rv2_v[-2] != 0:
                debt_gr = (_td2_v[-1] - _td2_v[-2]) / abs(_td2_v[-2])
                rev_gr2 = (_rv2_v[-1] - _rv2_v[-2]) / abs(_rv2_v[-2])
                if _td2_v[-1] is not None and debt_gr > rev_gr2 + 0.05:
                    flags.append(("⚠ Debt growing > Revenue", f"Debt grew {debt_gr*100:.1f}% vs Revenue {rev_gr2*100:.1f}% last year", "#ffb347"))

        # 4. Payout ratio > 100%
        _po = safe(info.get("payoutRatio"))
        if _po and _po > 1.0:
            flags.append(("⚠ Payout > 100%", f"Payout ratio = {_po*100:.0f}% — dividend being paid from borrowings or reserves", "#ff3b3b"))

        # 5. D/E > 2.0
        _de = safe(info.get("debtToEquity"))
        if _de and _de / 100 > 2.0:
            flags.append(("⚠ High leverage", f"D/E ratio = {_de/100:.2f}x — highly leveraged balance sheet", "#ffb347"))
        elif _de and _de / 100 < 0.3:
            flags.append(("✓ Low leverage", f"D/E ratio = {_de/100:.2f}x — clean balance sheet", "#00d084"))

        # 6. Current ratio < 1.0
        _cr3 = safe(info.get("currentRatio"))
        if _cr3 and _cr3 < 1.0:
            flags.append(("⚠ Current ratio < 1.0", f"Current ratio = {_cr3:.2f}x — short-term liquidity risk", "#ff3b3b"))

        # 7. Revenue growth decelerating (3Y vs 1Y)
        _rv_clean = [v for v in _rv2_v if v is not None and v > 0]
        if len(_rv_clean) >= 4:
            _cagr1 = (_rv_clean[-1] / _rv_clean[-2]) - 1 if _rv_clean[-2] else None
            _cagr3 = (_rv_clean[-1] / _rv_clean[-4]) ** (1/3) - 1 if len(_rv_clean) >= 4 and _rv_clean[-4] else None
            if _cagr1 is not None and _cagr3 is not None and _cagr1 < _cagr3 - 0.05:
                flags.append(("⚠ Growth decelerating", f"1Y revenue growth {_cagr1*100:.1f}% < 3Y CAGR {_cagr3*100:.1f}%", "#ffb347"))

        # 8. Promoter holding very low
        _promo = safe(info.get("heldPercentInsiders"))
        if _promo is not None and _promo < 0.25:
            flags.append(("⚠ Low promoter holding", f"Promoter/Insider holding = {_promo*100:.1f}% (< 25%)", "#ffb347"))
        elif _promo is not None and _promo >= 0.50:
            flags.append(("✓ High promoter conviction", f"Promoter/Insider holding = {_promo*100:.1f}%", "#00d084"))

        if flags:
            for flag_label, flag_desc, flag_col in flags:
                st.markdown(f"""
<div style="border-left:3px solid {flag_col};padding:6px 12px;margin:4px 0;background:#0d0d0d;
     font-family:'IBM Plex Mono';display:flex;gap:16px;align-items:center;">
  <span style="color:{flag_col};font-size:.68rem;font-weight:700;min-width:200px;">{flag_label}</span>
  <span style="color:#888;font-size:.63rem;">{flag_desc}</span>
</div>""", unsafe_allow_html=True)
        else:
            st.success("No major red flags detected based on available data.")

    except Exception as e:
        st.info(f"Red flag analysis: insufficient data ({e})")

    # ══════════════════════════════
    # DUPONT DECOMPOSITION (SUMMARY)
    # ══════════════════════════════
    st.divider()
    section_header("DUPONT DECOMPOSITION", "ROE = NET MARGIN × ASSET TURNOVER × EQUITY MULTIPLIER")
    try:
        _ni_dp   = safe(info.get("netIncomeToCommon")) or safe(info.get("profitMargins"))
        _rev_dp  = safe(info.get("totalRevenue"))
        _ta_dp   = None
        _eq_dp   = None
        _ba_dp   = data["balance_annual"]
        _ia_dp   = data["income_annual"]

        _ta_dd, _ta_dv = annual_series(_ba_dp, "Total Assets")
        _eq_dd, _eq_dv = annual_series(_ba_dp, "Stockholders Equity","Total Equity","Common Stock Equity")
        _ni_dd, _ni_dv = annual_series(_ia_dp, "Net Income","Net income")
        _rv_dd, _rv_dv = annual_series(_ia_dp, "Total Revenue","Revenue")

        if _ta_dv and _eq_dv and _ni_dv and _rv_dv:
            ta_dp = _ta_dv[-1]; eq_dp = _eq_dv[-1]; ni_dp = _ni_dv[-1]; rv_dp = _rv_dv[-1]
            if all(v and v != 0 for v in [ta_dp, eq_dp, ni_dp, rv_dp]):
                nm_dp  = ni_dp / rv_dp          # Net Margin
                at_dp  = rv_dp / ta_dp          # Asset Turnover
                em_dp  = ta_dp / eq_dp          # Equity Multiplier
                roe_dp = nm_dp * at_dp * em_dp  # = ROE

                dp1, dp2, dp3, dp4, dp5 = st.columns(5)
                dp1.metric("Net Margin",       f"{nm_dp*100:.1f}%",  help="Profit per ₹ of revenue")
                dp2.metric("×  Asset Turnover",f"{at_dp:.2f}x",      help="Revenue per ₹ of assets")
                dp3.metric("×  Equity Mult.",  f"{em_dp:.2f}x",      help="Assets per ₹ of equity (leverage)")
                dp4.metric("=  ROE (DuPont)",  f"{roe_dp*100:.1f}%", help="Derived ROE from decomposition")
                dp5.metric("Reported ROE",     f"{roe*100:.1f}%" if roe else "N/A", help="From yfinance info")

                # Driver diagnosis
                driver = "margin-driven" if nm_dp > 0.12 else "leverage-driven" if em_dp > 3 else "turnover-driven"
                driver_color = "#00d084" if driver == "margin-driven" else "#ffb347" if driver == "turnover-driven" else "#ff8c00"
                st.markdown(f"""
<div style="font-family:'IBM Plex Mono';font-size:.65rem;color:#888;margin-top:8px;">
  ROE is primarily <span style="color:{driver_color};font-weight:700;">{driver.upper()}</span>
  &nbsp;·&nbsp; Net Margin: {nm_dp*100:.1f}%
  &nbsp;·&nbsp; Asset Turnover: {at_dp:.2f}x
  &nbsp;·&nbsp; Leverage: {em_dp:.2f}x
</div>""", unsafe_allow_html=True)
    except Exception as e:
        st.info(f"DuPont decomposition: {e}")


# ══════════════════════════════════════════════════════════════
# TAB 2 — VALUATION
# ══════════════════════════════════════════════════════════════
with tab_val:
    section_header("VALUATION ANALYSIS", "P/E · P/B · P/S · EV/EBITDA — HISTORICAL + PEER COMPARISON")

    ia = data["income_annual"]
    ba = data["balance_annual"]
    ph = data["price_history"]

    # ── P/E History from price + EPS ──
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("#### TRAILING P/E HISTORY  ·  12-MONTH ROLLING")
        if not ph.empty:
            ph = ph.copy()
            ph.index = pd.to_datetime(ph.index).tz_localize(None)
            annual_eps_dates, annual_eps_vals = annual_series(ia, "Net Income", "Net income", "NetIncome")
            # Get shares to compute EPS from net income
            shares = safe(info.get("sharesOutstanding"))
            if annual_eps_vals and shares:
                # Compute trailing P/E from monthly price / (latest TTM EPS)
                eps_val = safe(info.get("trailingEps"))
                if eps_val:
                    ph["PE"] = ph["Close"] / eps_val
                    fig_pe = go.Figure()
                    fig_pe.add_trace(go.Scatter(
                        x=ph.index, y=ph["PE"], name="P/E", mode="lines",
                        line=dict(color="#ff8c00", width=2),
                        fill="tozeroy", fillcolor="rgba(255,140,0,0.08)"
                    ))
                    # Mean ± 1SD band
                    pe_mean = ph["PE"].mean()
                    pe_std  = ph["PE"].std()
                    fig_pe.add_hline(y=pe_mean, line_dash="dot", line_color="#888",
                                     annotation_text=f"5Y avg {pe_mean:.1f}x",
                                     annotation_font_color="#888")
                    fig_pe.add_hrect(y0=pe_mean-pe_std, y1=pe_mean+pe_std,
                                     fillcolor="rgba(255,140,0,0.05)", line_width=0)
                    fig_pe.update_layout(title="Trailing P/E (price ÷ TTM EPS)",
                                         yaxis_title="P/E Multiple", height=280)
                    fig_pe.update_layout(**BB_LAYOUT)
                    st.plotly_chart(fig_pe, use_container_width=True)
                else:
                    st.info("TTM EPS not available for P/E history")
            else:
                st.info("EPS or share count data not available")
        else:
            st.info("Price history not available")

    with col_right:
        st.markdown("#### PRICE/BOOK HISTORY")
        if not ph.empty and book_v:
            ph2 = ph.copy()
            ph2["PB"] = ph2["Close"] / book_v
            fig_pb = go.Figure(go.Scatter(
                x=ph2.index, y=ph2["PB"], mode="lines",
                line=dict(color="#1e90ff", width=2),
                fill="tozeroy", fillcolor="rgba(30,144,255,0.07)"
            ))
            pb_mean = ph2["PB"].mean()
            fig_pb.add_hline(y=pb_mean, line_dash="dot", line_color="#888",
                             annotation_text=f"5Y avg {pb_mean:.2f}x",
                             annotation_font_color="#888")
            fig_pb.update_layout(title="Price / Book Value",
                                  yaxis_title="P/B Multiple", height=280)
            fig_pb.update_layout(**BB_LAYOUT)
            st.plotly_chart(fig_pb, use_container_width=True)
        else:
            st.info("Book value or price history not available")

    # ── Peer Valuation Comparison ──
    section_header("PEER VALUATION TABLE", "KEY MULTIPLES VS SECTOR COMPARISON")
    val_metrics = ["trailingPE","forwardPE","priceToBook","priceToSalesTrailing12Months",
                   "enterpriseToEbitda","profitMargins","returnOnEquity"]
    val_labels  = ["Trail P/E","Fwd P/E","P/Book","P/Sales","EV/EBITDA","Net Margin%","ROE%"]

    peer_rows = []
    all_tickers = [t_sym] + list(peers.keys())
    all_data    = {t_sym: data}
    all_data.update(peers)

    for sym in all_tickers:
        d_info = all_data[sym]["info"]
        row = {"Ticker": sym}
        for col, lbl in zip(val_metrics, val_labels):
            v = safe(d_info.get(col))
            if v is not None and lbl.endswith("%"):
                row[lbl] = round(v * 100, 1)
            elif v is not None:
                row[lbl] = round(v, 2)
            else:
                row[lbl] = None
        peer_rows.append(row)

    peer_val_df = pd.DataFrame(peer_rows).set_index("Ticker")

    def style_val_table(df):
        def _pe(v):
            if not isinstance(v, (int, float)) or v != v: return "color:#555"
            if v < 15: return "color:#00ff88;font-weight:700"
            if v < 25: return "color:#b8e06a"
            if v < 40: return "color:#ffb347"
            return "color:#ff3b3b"
        def _pb(v):
            if not isinstance(v, (int, float)) or v != v: return "color:#555"
            if v < 2:  return "color:#00ff88;font-weight:700"
            if v < 4:  return "color:#b8e06a"
            if v < 8:  return "color:#ffb347"
            return "color:#ff3b3b"
        def _roe(v):
            if not isinstance(v, (int, float)) or v != v: return "color:#555"
            if v >= 20: return "color:#00ff88;font-weight:700"
            if v >= 12: return "color:#b8e06a"
            if v >= 5:  return "color:#ffb347"
            return "color:#ff3b3b"
        def _nm(v):
            if not isinstance(v, (int, float)) or v != v: return "color:#555"
            if v >= 15: return "color:#00ff88;font-weight:700"
            if v >= 8:  return "color:#b8e06a"
            if v >= 3:  return "color:#ffb347"
            return "color:#ff3b3b"
        sty = df.style
        for c in ["Trail P/E","Fwd P/E","EV/EBITDA"]:
            if c in df.columns: sty = sty.applymap(_pe, subset=[c])
        if "P/Book" in df.columns: sty = sty.applymap(_pb, subset=["P/Book"])
        if "ROE%"   in df.columns: sty = sty.applymap(_roe, subset=["ROE%"])
        if "Net Margin%" in df.columns: sty = sty.applymap(_nm, subset=["Net Margin%"])
        return sty

    st.dataframe(style_val_table(peer_val_df), use_container_width=True)

    # ── EV/EBITDA Peer Bar ──
    ev_ebitda_vals = [safe(all_data[s]["info"].get("enterpriseToEbitda")) for s in all_tickers]
    valid_ev = [(s, v) for s, v in zip(all_tickers, ev_ebitda_vals) if v is not None]
    if valid_ev:
        syms_, vals_ = zip(*valid_ev)
        colors_ = ["#ff8c00" if s == t_sym else "#2a2a2a" for s in syms_]
        fig_ev = go.Figure(go.Bar(x=list(syms_), y=list(vals_),
                                   marker_color=colors_, marker_line_width=0,
                                   text=[f"{v:.1f}x" for v in vals_],
                                   textposition="outside",
                                   textfont=dict(color="#e8e8e8", size=9)))
        fig_ev.update_layout(title="EV / EBITDA — Peer Comparison",
                              yaxis_title="EV/EBITDA Multiple", height=280)
        fig_ev.update_layout(**BB_LAYOUT)
        st.plotly_chart(fig_ev, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# TAB 3 — PROFITABILITY
# ══════════════════════════════════════════════════════════════
with tab_prof:
    section_header("PROFITABILITY TRENDS", "MARGINS · ROE · ROCE · ROA — ANNUAL TIME SERIES")

    ia = data["income_annual"]
    ba = data["balance_annual"]

    # ── Revenue & Net Profit trend ──
    rev_d, rev_v = annual_series(ia, "Total Revenue", "Revenue")
    ni_d,  ni_v  = annual_series(ia, "Net Income", "Net income")

    if rev_v and rev_d:
        col1, col2 = st.columns(2)
        with col1:
            rev_cr = [v/1e7 if v else None for v in rev_v]  # → Crores
            ni_cr  = [v/1e7 if v else None for v in ni_v]   if ni_v else []
            fig_rn = go.Figure()
            fig_rn.add_trace(go.Bar(x=rev_d, y=rev_cr, name="Revenue",
                                     marker_color="#1e90ff", marker_line_width=0))
            if ni_cr and len(ni_cr) == len(rev_d):
                fig_rn.add_trace(go.Bar(x=ni_d or rev_d, y=ni_cr, name="Net Profit",
                                         marker_color="#00d084", marker_line_width=0))
            fig_rn.update_layout(title="Revenue & Net Profit  (₹ Cr)",
                                   barmode="group", yaxis_title="₹ Crores", height=300)
            fig_rn.update_layout(**BB_LAYOUT)
            st.plotly_chart(fig_rn, use_container_width=True)

        with col2:
            # Margin trend
            gm_d, gm_v = annual_series(ia, "Gross Profit")
            op_d, op_v = annual_series(ia, "Operating Income", "Ebit", "EBIT")
            if rev_v and gm_v and len(gm_v) == len(rev_v):
                gm_pct = [g/r*100 if g and r else None for g, r in zip(gm_v, rev_v)]
                op_pct = [o/r*100 if o and r else None for o, r in zip(op_v, rev_v)] if op_v and len(op_v)==len(rev_v) else []
                ni_pct = [n/r*100 if n and r else None for n, r in zip(ni_v, rev_v)] if ni_v and len(ni_v)==len(rev_v) else []
                fig_mg = go.Figure()
                fig_mg.add_trace(go.Scatter(x=rev_d, y=gm_pct, name="Gross Margin",
                                             mode="lines+markers", line=dict(color="#00ccff", width=2)))
                if op_pct:
                    fig_mg.add_trace(go.Scatter(x=op_d or rev_d, y=op_pct, name="EBIT Margin",
                                                 mode="lines+markers", line=dict(color="#ff8c00", width=2)))
                if ni_pct:
                    fig_mg.add_trace(go.Scatter(x=ni_d or rev_d, y=ni_pct, name="Net Margin",
                                                 mode="lines+markers", line=dict(color="#00d084", width=2)))
                fig_mg.update_layout(title="Margin Trends  (%)", yaxis_title="Margin %", height=300)
                fig_mg.update_layout(**BB_LAYOUT)
                st.plotly_chart(fig_mg, use_container_width=True)
            else:
                st.info("Margin data incomplete for this period")

    # ── ROE / ROA trend ──
    section_header("RETURN RATIOS — ANNUAL", "ROE · ROA · ROCE")
    eq_d,  eq_v  = annual_series(ba, "Stockholders Equity", "Total Equity", "Common Stock Equity")
    ta_d,  ta_v  = annual_series(ba, "Total Assets")
    debt_d, debt_v = annual_series(ba, "Total Debt", "Long Term Debt")

    if ni_v and eq_v and len(ni_v) >= 2:
        roe_hist = [n/e*100 if n and e and e != 0 else None for n, e in zip(ni_v[-len(eq_v):], eq_v)]
        roa_hist = [n/a*100 if n and a and a != 0 else None for n, a in zip(ni_v[-len(ta_v):], ta_v)] if ta_v else []

        op_v2    = annual_series(ia, "Operating Income", "Ebit")[1]
        # ROCE = EBIT / (Total Assets - Current Liabilities)
        cl_d, cl_v = annual_series(ba, "Current Liabilities")
        roce_hist = []
        if op_v2 and ta_v and cl_v and len(op_v2)==len(ta_v)==len(cl_v):
            roce_hist = [o/(a-c)*100 if o and a and c and (a-c) != 0 else None
                         for o, a, c in zip(op_v2, ta_v, cl_v)]

        fig_ret = go.Figure()
        fig_ret.add_trace(go.Scatter(x=eq_d, y=roe_hist, name="ROE %",
                                      mode="lines+markers", line=dict(color="#ff8c00", width=2)))
        if roa_hist:
            fig_ret.add_trace(go.Scatter(x=ta_d, y=roa_hist, name="ROA %",
                                          mode="lines+markers", line=dict(color="#1e90ff", width=2)))
        if roce_hist:
            fig_ret.add_trace(go.Scatter(x=ta_d, y=roce_hist, name="ROCE %",
                                          mode="lines+markers", line=dict(color="#00d084", width=2)))
        fig_ret.add_hline(y=15, line_dash="dot", line_color="#555",
                           annotation_text="15% benchmark", annotation_font_color="#555")
        fig_ret.update_layout(title="ROE / ROA / ROCE  (%)", yaxis_title="%", height=320)
        fig_ret.update_layout(**BB_LAYOUT)
        st.plotly_chart(fig_ret, use_container_width=True)
    else:
        st.info("Insufficient data for return ratio history")

    # ── Quarterly margin trend ──
    section_header("QUARTERLY MARGINS", "LAST 8 QUARTERS")
    iq = data["income_quarterly"]
    if not iq.empty:
        qrev_d, qrev_v = quarterly_series(iq, "Total Revenue", "Revenue")
        qni_d,  qni_v  = quarterly_series(iq, "Net Income", "Net income")
        if qrev_v and qni_v and len(qrev_v) == len(qni_v):
            qni_pct = [n/r*100 if n and r else None for n, r in zip(qni_v, qrev_v)]
            # last 8 quarters
            n8 = 8
            fig_qm = go.Figure()
            fig_qm.add_trace(go.Bar(x=qrev_d[-n8:], y=[v/1e7 if v is not None else None for v in qrev_v[-n8:]],
                                     name="Revenue (Cr)", marker_color="#1e90ff",
                                     yaxis="y2", opacity=0.4))
            fig_qm.add_trace(go.Scatter(x=qni_d[-n8:], y=qni_pct[-n8:], name="Net Margin %",
                                         mode="lines+markers", line=dict(color="#ff8c00", width=2)))
            fig_qm.update_layout(
                title="Quarterly Net Margin % (line) + Revenue ₹Cr (bar)",
                yaxis=dict(title="Net Margin %", gridcolor="#1a1a1a", tickfont=dict(color="#888")),
                yaxis2=dict(title="₹ Cr", overlaying="y", side="right",
                             showgrid=False, tickfont=dict(color="#444")),
                height=300,
            )
            fig_qm.update_layout(**{k: v for k, v in BB_LAYOUT.items() if k not in ("yaxis",)})
            st.plotly_chart(fig_qm, use_container_width=True)
    else:
        st.info("Quarterly data not available")


# ══════════════════════════════════════════════════════════════
# TAB 4 — GROWTH
# ══════════════════════════════════════════════════════════════
with tab_growth:
    section_header("GROWTH ANALYSIS", "REVENUE · EPS · EBITDA — CAGR ACROSS HORIZONS")

    ia = data["income_annual"]

    rev_d, rev_v = annual_series(ia, "Total Revenue", "Revenue")
    ni_d,  ni_v  = annual_series(ia, "Net Income", "Net income")
    ebitda_d, ebitda_v = annual_series(ia, "EBITDA", "Ebitda")
    dep_d, dep_v = annual_series(ia, "Depreciation", "Depreciation And Amortization")

    # Compute EBITDA if not directly available
    if not ebitda_v and ni_v:
        op_d2, op_v2 = annual_series(ia, "Operating Income", "Ebit")
        if op_v2 and dep_v and len(op_v2) == len(dep_v):
            ebitda_v = [o + d if o and d else None for o, d in zip(op_v2, dep_v)]
            ebitda_d = op_d2

    # EPS from info (trailing) — use shares outstanding + net income for history
    shares = safe(info.get("sharesOutstanding"))
    if ni_v and shares:
        eps_hist = [n/shares if n else None for n in ni_v]
    else:
        eps_hist = []

    # ── CAGR Table ──
    cagr_rows = []
    for label, vals in [("Revenue", rev_v), ("Net Profit", ni_v),
                        ("EBITDA", ebitda_v), ("EPS", eps_hist)]:
        if not vals: continue
        clean = [v for v in vals if v is not None and v > 0]
        r = {"Metric": label}
        for yrs, lbl in [(1,"1Y CAGR"),(3,"3Y CAGR"),(5,"5Y CAGR")]:
            c = cagr(clean, yrs)
            r[lbl] = f"{c*100:.1f}%" if c is not None else "N/A"
        # Latest value
        r["Latest"] = fmt_cr(vals[-1] * (1 if label == "EPS" else 1)) if vals[-1] else "N/A"
        cagr_rows.append(r)

    if cagr_rows:
        cagr_df = pd.DataFrame(cagr_rows).set_index("Metric")
        def style_cagr(df):
            def c(v):
                if not isinstance(v, str) or v == "N/A": return "color:#555"
                try:
                    n = float(v.replace("%",""))
                    if n >= 15: return "color:#00ff88;font-weight:700"
                    if n >= 8:  return "color:#b8e06a"
                    if n >= 0:  return "color:#ffb347"
                    return "color:#ff3b3b"
                except: return "color:#555"
            sty = df.style
            for col in ["1Y CAGR","3Y CAGR","5Y CAGR"]:
                if col in df.columns: sty = sty.applymap(c, subset=[col])
            return sty
        st.dataframe(style_cagr(cagr_df), use_container_width=True)
        st.caption("CAGR computed from actual reported financials. Compares stock vs own history — not vs arbitrary benchmarks.")

    # ── Revenue & Net Profit Growth bars ──
    col_g1, col_g2 = st.columns(2)

    with col_g1:
        if rev_v and len(rev_v) >= 2:
            rev_yoy = [None] + [(rev_v[i] - rev_v[i-1]) / abs(rev_v[i-1]) * 100
                                if rev_v[i] and rev_v[i-1] and rev_v[i-1] != 0 else None
                                for i in range(1, len(rev_v))]
            colors = ["#00d084" if v and v >= 0 else "#ff3b3b" for v in rev_yoy[1:]]
            fig_rg = go.Figure(go.Bar(
                x=rev_d[1:], y=rev_yoy[1:], marker_color=colors, marker_line_width=0,
                text=[f"{v:.1f}%" if v else "" for v in rev_yoy[1:]],
                textposition="outside", textfont=dict(color="#e8e8e8", size=9)
            ))
            fig_rg.update_layout(title="Revenue YoY Growth %", yaxis_title="%", height=280)
            fig_rg.update_layout(**BB_LAYOUT)
            st.plotly_chart(fig_rg, use_container_width=True)

    with col_g2:
        if ni_v and len(ni_v) >= 2:
            ni_yoy = [None] + [(ni_v[i] - ni_v[i-1]) / abs(ni_v[i-1]) * 100
                               if ni_v[i] and ni_v[i-1] and ni_v[i-1] != 0 else None
                               for i in range(1, len(ni_v))]
            colors2 = ["#00d084" if v and v >= 0 else "#ff3b3b" for v in ni_yoy[1:]]
            fig_ng = go.Figure(go.Bar(
                x=ni_d[1:], y=ni_yoy[1:], marker_color=colors2, marker_line_width=0,
                text=[f"{v:.1f}%" if v else "" for v in ni_yoy[1:]],
                textposition="outside", textfont=dict(color="#e8e8e8", size=9)
            ))
            fig_ng.update_layout(title="Net Profit YoY Growth %", yaxis_title="%", height=280)
            fig_ng.update_layout(**BB_LAYOUT)
            st.plotly_chart(fig_ng, use_container_width=True)

    # ── EPS trend ──
    if eps_hist and rev_d:
        section_header("EPS TREND — ANNUAL", "NET INCOME ÷ SHARES OUTSTANDING")
        fig_eps = go.Figure()
        fig_eps.add_trace(go.Bar(x=rev_d, y=eps_hist, name="EPS ₹",
                                  marker_color=["#ff8c00" if e and e >= 0 else "#ff3b3b" for e in eps_hist],
                                  marker_line_width=0))
        fig_eps.update_layout(title="Annual EPS  (₹)", yaxis_title="₹ per share", height=260)
        fig_eps.update_layout(**BB_LAYOUT)
        st.plotly_chart(fig_eps, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# TAB 5 — FINANCIAL HEALTH
# ══════════════════════════════════════════════════════════════
with tab_health:
    section_header("FINANCIAL HEALTH", "DEBT · COVERAGE · LIQUIDITY · ALTMAN Z-SCORE")

    ba = data["balance_annual"]
    ia = data["income_annual"]

    eq_d, eq_v   = annual_series(ba, "Stockholders Equity","Total Equity","Common Stock Equity")
    ta_d, ta_v   = annual_series(ba, "Total Assets")
    td_d, td_v   = annual_series(ba, "Total Debt","Long Term Debt And Capital Lease Obligation")
    cl_d, cl_v   = annual_series(ba, "Current Liabilities")
    ca_d, ca_v   = annual_series(ba, "Current Assets")
    ni_d,  ni_v  = annual_series(ia,  "Net Income","Net income")
    int_d, int_v = annual_series(ia,  "Interest Expense")
    ebit_d,ebit_v= annual_series(ia,  "Operating Income","Ebit","EBIT")

    col_h1, col_h2 = st.columns(2)

    with col_h1:
        # D/E Ratio history
        if td_v and eq_v and len(td_v) == len(eq_v):
            de_hist = [d/e if d and e and e != 0 else None for d, e in zip(td_v, eq_v)]
            colors_de = ["#00ff88" if v and v < 0.5 else "#ffb347" if v and v < 1.5 else "#ff3b3b"
                         for v in de_hist]
            fig_de = go.Figure(go.Bar(x=eq_d, y=de_hist, marker_color=colors_de,
                                       marker_line_width=0,
                                       text=[f"{v:.2f}x" if v else "" for v in de_hist],
                                       textposition="outside",
                                       textfont=dict(color="#e8e8e8", size=9)))
            fig_de.add_hline(y=1.0, line_dash="dot", line_color="#ffb347",
                              annotation_text="D/E = 1.0 caution", annotation_font_color="#ffb347")
            fig_de.update_layout(title="Debt / Equity Ratio", yaxis_title="D/E", height=280)
            fig_de.update_layout(**BB_LAYOUT)
            st.plotly_chart(fig_de, use_container_width=True)

    with col_h2:
        # Interest Coverage = EBIT / Interest Expense
        if ebit_v and int_v and len(ebit_v) == len(int_v):
            ic_hist = [e / abs(i) if e and i and i != 0 else None for e, i in zip(ebit_v, int_v)]
            colors_ic = ["#00ff88" if v and v >= 5 else "#ffb347" if v and v >= 2 else "#ff3b3b"
                          for v in ic_hist]
            fig_ic = go.Figure(go.Bar(x=ebit_d, y=ic_hist, marker_color=colors_ic,
                                       marker_line_width=0,
                                       text=[f"{v:.1f}x" if v else "" for v in ic_hist],
                                       textposition="outside",
                                       textfont=dict(color="#e8e8e8", size=9)))
            fig_ic.add_hline(y=3.0, line_dash="dot", line_color="#ffb347",
                              annotation_text="3x minimum safe", annotation_font_color="#ffb347")
            fig_ic.update_layout(title="Interest Coverage Ratio  (EBIT ÷ Interest)",
                                   yaxis_title="Times", height=280)
            fig_ic.update_layout(**BB_LAYOUT)
            st.plotly_chart(fig_ic, use_container_width=True)

    # ── Current Ratio history ──
    if ca_v and cl_v and len(ca_v) == len(cl_v):
        cr_hist = [a/l if a and l and l != 0 else None for a, l in zip(ca_v, cl_v)]
        section_header("CURRENT RATIO TREND", "CURRENT ASSETS ÷ CURRENT LIABILITIES")
        colors_cr = ["#00ff88" if v and v >= 2.0 else "#ffb347" if v and v >= 1.0 else "#ff3b3b"
                      for v in cr_hist]
        fig_cr = go.Figure()
        fig_cr.add_trace(go.Scatter(x=ca_d, y=cr_hist, mode="lines+markers",
                                     line=dict(color="#00ccff", width=2),
                                     marker=dict(color=colors_cr, size=8)))
        fig_cr.add_hline(y=1.5, line_dash="dot", line_color="#555",
                          annotation_text="1.5 benchmark", annotation_font_color="#555")
        fig_cr.update_layout(title="Current Ratio", yaxis_title="Ratio", height=240)
        fig_cr.update_layout(**BB_LAYOUT)
        st.plotly_chart(fig_cr, use_container_width=True)

    # ── Cash Conversion Cycle ──
    section_header("CASH CONVERSION CYCLE", "DEBTOR DAYS + INVENTORY DAYS − CREDITOR DAYS  ·  LOWER = MORE EFFICIENT")
    try:
        _inv_d, _inv_v   = annual_series(ba, "Inventory","Inventories")
        _ar_d,  _ar_v    = annual_series(ba, "Accounts Receivable","Net Receivables","Receivables")
        _ap_d,  _ap_v    = annual_series(ba, "Accounts Payable","Payables And Accrued Expenses")
        _cogs_d, _cogs_v = annual_series(ia, "Cost Of Revenue","Cost of Goods Sold","Cost Of Goods And Services Sold")
        _rev_h_d, _rev_h_v = annual_series(ia, "Total Revenue","Revenue")

        ccc_years, dso_hist, dio_hist, dpo_hist, ccc_hist = [], [], [], [], []
        _n = min(len(v) for v in [_ar_v, _inv_v, _ap_v, _cogs_v, _rev_h_v] if v)
        if _n >= 2:
            for i in range(_n):
                try:
                    ar   = _ar_v[i];  inv  = _inv_v[i]
                    ap   = _ap_v[i];  cogs = _cogs_v[i]; rev = _rev_h_v[i]
                    if all(v and v > 0 for v in [ar, inv, ap, cogs, rev]):
                        dso = ar   / rev  * 365   # Days Sales Outstanding
                        dio = inv  / cogs * 365   # Days Inventory Outstanding
                        dpo = ap   / cogs * 365   # Days Payable Outstanding
                        ccc = dso + dio - dpo
                        dso_hist.append(dso); dio_hist.append(dio)
                        dpo_hist.append(dpo); ccc_hist.append(ccc)
                        ccc_years.append(_rev_h_d[i] if _rev_h_d else str(i))
                except: pass

        if ccc_hist:
            fig_ccc = go.Figure()
            fig_ccc.add_trace(go.Scatter(x=ccc_years, y=dso_hist, name="DSO (Debtor Days)",
                                          mode="lines+markers", line=dict(color="#1e90ff", width=2)))
            fig_ccc.add_trace(go.Scatter(x=ccc_years, y=dio_hist, name="DIO (Inventory Days)",
                                          mode="lines+markers", line=dict(color="#ffb347", width=2)))
            fig_ccc.add_trace(go.Scatter(x=ccc_years, y=dpo_hist, name="DPO (Creditor Days)",
                                          mode="lines+markers", line=dict(color="#00d084", width=2)))
            fig_ccc.add_trace(go.Scatter(x=ccc_years, y=ccc_hist, name="CCC (Net)",
                                          mode="lines+markers+text",
                                          text=[f"{v:.0f}d" for v in ccc_hist],
                                          textposition="top center",
                                          line=dict(color="#ff8c00", width=3)))
            fig_ccc.update_layout(title="Cash Conversion Cycle (Days)",
                                   yaxis_title="Days", height=300)
            fig_ccc.update_layout(**BB_LAYOUT)
            st.plotly_chart(fig_ccc, use_container_width=True)

            # CCC latest insight
            _latest_ccc = ccc_hist[-1]
            _ccc_color  = "#00ff88" if _latest_ccc < 30 else "#ffb347" if _latest_ccc < 90 else "#ff3b3b"
            st.markdown(f"""
<div style="font-family:'IBM Plex Mono';font-size:.65rem;color:#888;padding:6px 12px;
     border-left:3px solid {_ccc_color};background:#0d0d0d;">
  Latest CCC: <span style="color:{_ccc_color};font-weight:700;">{_latest_ccc:.0f} days</span>
  &nbsp;·&nbsp; DSO {dso_hist[-1]:.0f}d + DIO {dio_hist[-1]:.0f}d − DPO {dpo_hist[-1]:.0f}d
  &nbsp;·&nbsp; {"Asset-light / efficient" if _latest_ccc < 30 else "Moderate working capital need" if _latest_ccc < 90 else "High working capital intensity"}
</div>""", unsafe_allow_html=True)
        else:
            st.info("Cash Conversion Cycle: inventory, receivables, or COGS data not available for this company.")
    except Exception as e:
        st.info(f"Cash Conversion Cycle: {e}")

    # ── Altman Z-Score ──
    section_header("ALTMAN Z-SCORE", "BANKRUPTCY RISK MODEL — Z>2.99 SAFE · 1.81-2.99 GREY · <1.81 DISTRESS")
    with st.expander("◼ SHOW ALTMAN Z-SCORE (click to compute)", expanded=True):
        # Z = 1.2×X1 + 1.4×X2 + 3.3×X3 + 0.6×X4 + 1.0×X5
        # X1 = Working Capital / Total Assets
        # X2 = Retained Earnings / Total Assets
        # X3 = EBIT / Total Assets
        # X4 = Market Cap / Total Liabilities
        # X5 = Revenue / Total Assets
        try:
            ta_val  = ta_v[-1]  if ta_v  else None
            ca_val  = ca_v[-1]  if ca_v  else None
            cl_val  = cl_v[-1]  if cl_v  else None
            wc      = ca_val - cl_val if ca_val and cl_val else None
            re_val  = safe(data["balance_annual"].loc[
                next((i for i in data["balance_annual"].index if "retained" in str(i).lower()), ""), :
            ].iloc[-1]) if not ba.empty else None

            ebit_val = ebit_v[-1] if ebit_v else None
            td_val   = td_v[-1]   if td_v   else None
            rev_last = annual_series(ia, "Total Revenue","Revenue")[1]
            rev_val  = rev_last[-1] if rev_last else None
            mc       = safe(info.get("marketCap"))

            x1 = wc / ta_val     if wc      and ta_val and ta_val != 0 else None
            x2 = re_val / ta_val if re_val  and ta_val and ta_val != 0 else None
            x3 = ebit_val / ta_val if ebit_val and ta_val and ta_val != 0 else None
            x4 = mc / td_val     if mc      and td_val and td_val != 0 else None
            x5 = rev_val / ta_val if rev_val and ta_val and ta_val != 0 else None

            if all(v is not None for v in [x1, x3, x5]):
                z_score = (1.2 * (x1 or 0) + 1.4 * (x2 or 0) +
                           3.3 * (x3 or 0) + 0.6 * (x4 or 0) + 1.0 * (x5 or 0))
                z_color = "#00ff88" if z_score > 2.99 else "#ffb347" if z_score > 1.81 else "#ff3b3b"
                z_label = "SAFE ZONE" if z_score > 2.99 else "GREY ZONE" if z_score > 1.81 else "⚠ DISTRESS ZONE"
                zc1, zc2, zc3, zc4, zc5, zc6 = st.columns(6)
                zc1.metric("Z-Score",    f"{z_score:.2f}", delta=z_label)
                zc2.metric("X1 (WC/TA)", f"{x1:.3f}" if x1 else "N/A")
                zc3.metric("X2 (RE/TA)", f"{x2:.3f}" if x2 else "N/A")
                zc4.metric("X3 (EBIT/TA)",f"{x3:.3f}" if x3 else "N/A")
                zc5.metric("X4 (MC/Debt)",f"{x4:.3f}" if x4 else "N/A")
                zc6.metric("X5 (Rev/TA)", f"{x5:.3f}" if x5 else "N/A")
                st.progress(min(1.0, z_score / 4.0))
                st.markdown(f"""
<span style="color:{z_color};font-family:'IBM Plex Mono';font-size:.72rem;font-weight:700;">
  Z = {z_score:.2f}  →  {z_label}
</span>
<span style="color:#555;font-family:'IBM Plex Mono';font-size:.60rem;margin-left:16px;">
  Formula: 1.2×X1 + 1.4×X2 + 3.3×X3 + 0.6×X4 + 1.0×X5
</span>""", unsafe_allow_html=True)
            else:
                st.info("Insufficient balance sheet data to compute Altman Z-Score")
        except Exception as e:
            st.info(f"Could not compute Altman Z-Score: {e}")


# ══════════════════════════════════════════════════════════════
# TAB 6 — CASH FLOW
# ══════════════════════════════════════════════════════════════
with tab_cf:
    section_header("CASH FLOW ANALYSIS", "OCF · FCF · CAPEX · FCF YIELD — ANNUAL TREND")

    cf = data["cashflow_annual"]
    ia = data["income_annual"]

    ocf_d, ocf_v  = annual_series(cf, "Operating Cash Flow", "Cash Flow From Continuing Operating Activities")
    cap_d, cap_v  = annual_series(cf, "Capital Expenditure", "Purchase Of Ppe", "Purchase Of Property Plant And Equipment")
    fcf_d, fcf_v  = annual_series(cf, "Free Cash Flow")

    # Compute FCF if not available
    if not fcf_v and ocf_v and cap_v and len(ocf_v) == len(cap_v):
        fcf_v = [o + c if o and c else None for o, c in zip(ocf_v, cap_v)]  # capex is negative
        fcf_d = ocf_d

    col_cf1, col_cf2 = st.columns(2)

    with col_cf1:
        if ocf_v and cap_v:
            fig_cf = go.Figure()
            fig_cf.add_trace(go.Bar(x=ocf_d, y=[v/1e7 if v else None for v in ocf_v],
                                     name="Operating CF", marker_color="#00d084", marker_line_width=0))
            if cap_v:
                fig_cf.add_trace(go.Bar(x=cap_d, y=[v/1e7 if v else None for v in cap_v],
                                         name="CapEx", marker_color="#ff3b3b", marker_line_width=0))
            if fcf_v:
                fig_cf.add_trace(go.Scatter(x=fcf_d, y=[v/1e7 if v else None for v in fcf_v],
                                             name="FCF", mode="lines+markers",
                                             line=dict(color="#ff8c00", width=2, dash="dot")))
            fig_cf.update_layout(title="OCF / CapEx / FCF  (₹ Cr)",
                                   barmode="group", yaxis_title="₹ Crores", height=300)
            fig_cf.update_layout(**BB_LAYOUT)
            st.plotly_chart(fig_cf, use_container_width=True)

    with col_cf2:
        # FCF Yield = FCF / Market Cap
        mc = safe(info.get("marketCap"))
        if fcf_v and mc:
            fcf_yield = [v / mc * 100 if v else None for v in fcf_v]
            colors_fy = ["#00ff88" if v and v >= 5 else "#ffb347" if v and v >= 2 else "#ff3b3b"
                          for v in fcf_yield]
            fig_fy = go.Figure(go.Bar(x=fcf_d, y=fcf_yield, marker_color=colors_fy,
                                       marker_line_width=0,
                                       text=[f"{v:.1f}%" if v else "" for v in fcf_yield],
                                       textposition="outside",
                                       textfont=dict(color="#e8e8e8", size=9)))
            fig_fy.add_hline(y=5, line_dash="dot", line_color="#00d084",
                              annotation_text="5% = strong yield", annotation_font_color="#00d084")
            fig_fy.update_layout(title="FCF Yield %  (FCF ÷ Market Cap × 100)",
                                   yaxis_title="Yield %", height=300)
            fig_fy.update_layout(**BB_LAYOUT)
            st.plotly_chart(fig_fy, use_container_width=True)

    # ── Capex Intensity = CapEx / Revenue ──
    rev_d2, rev_v2 = annual_series(ia, "Total Revenue", "Revenue")
    if cap_v and rev_v2 and len(cap_v) == len(rev_v2):
        capex_int = [abs(c)/r*100 if c and r and r != 0 else None for c, r in zip(cap_v, rev_v2)]
        section_header("CAPEX INTENSITY", "CAPEX ÷ REVENUE — HIGH = CAPITAL HEAVY BUSINESS")
        fig_ci = go.Figure(go.Scatter(
            x=cap_d, y=capex_int, mode="lines+markers+text",
            text=[f"{v:.1f}%" if v else "" for v in capex_int],
            textposition="top center", textfont=dict(size=9, color="#888"),
            line=dict(color="#cc88ff", width=2),
            fill="tozeroy", fillcolor="rgba(204,136,255,0.07)"
        ))
        fig_ci.update_layout(title="CapEx Intensity %  (CapEx ÷ Revenue)",
                              yaxis_title="% of Revenue", height=240)
        fig_ci.update_layout(**BB_LAYOUT)
        st.plotly_chart(fig_ci, use_container_width=True)

    # ── Quarterly OCF ──
    section_header("QUARTERLY OPERATING CASH FLOW", "LAST 8 QUARTERS")
    cfq = data["cashflow_quarterly"]
    qocf_d, qocf_v = quarterly_series(cfq, "Operating Cash Flow",
                                       "Cash Flow From Continuing Operating Activities")
    if qocf_v:
        colors_qocf = ["#00d084" if v and v >= 0 else "#ff3b3b" for v in qocf_v[-8:]]
        fig_qocf = go.Figure(go.Bar(
            x=qocf_d[-8:], y=[v/1e7 if v else None for v in qocf_v[-8:]],
            marker_color=colors_qocf, marker_line_width=0,
        ))
        fig_qocf.update_layout(title="Quarterly Operating Cash Flow  (₹ Cr)",
                                 yaxis_title="₹ Crores", height=240)
        fig_qocf.update_layout(**BB_LAYOUT)
        st.plotly_chart(fig_qocf, use_container_width=True)

    # ── Earnings Quality (CFO / Net Income) ──
    section_header("EARNINGS QUALITY & OPERATING LEVERAGE", "CFO÷NET INCOME  ·  EBIT%÷REVENUE% — RISK AMPLIFIER")
    eq_col1, eq_col2 = st.columns(2)

    with eq_col1:
        # Earnings Quality = Operating CF / Net Income
        _eq_ni_d, _eq_ni_v = annual_series(ia, "Net Income","Net income")
        _eq_cf_d, _eq_cf_v = annual_series(cf, "Operating Cash Flow","Cash Flow From Continuing Operating Activities")
        if _eq_ni_v and _eq_cf_v:
            _n_eq = min(len(_eq_ni_v), len(_eq_cf_v))
            _eq_dates = _eq_ni_d[-_n_eq:]
            _eq_ratio = [c/n if c is not None and n and n != 0 else None
                         for c, n in zip(_eq_cf_v[-_n_eq:], _eq_ni_v[-_n_eq:])]
            _eq_colors = ["#00ff88" if v and v >= 1.0 else "#ffb347" if v and v >= 0.8 else "#ff3b3b"
                           for v in _eq_ratio]
            fig_eq = go.Figure(go.Bar(
                x=_eq_dates, y=_eq_ratio, marker_color=_eq_colors, marker_line_width=0,
                text=[f"{v:.2f}x" if v else "" for v in _eq_ratio],
                textposition="outside", textfont=dict(color="#e8e8e8", size=9)
            ))
            fig_eq.add_hline(y=1.0, line_dash="dot", line_color="#00d084",
                              annotation_text="> 1.0 = fully cash-backed", annotation_font_color="#00d084")
            fig_eq.add_hline(y=0.8, line_dash="dot", line_color="#ffb347",
                              annotation_text="0.8 = minimum acceptable", annotation_font_color="#ffb347")
            fig_eq.update_layout(title="Earnings Quality  (CFO ÷ Net Income)", yaxis_title="Ratio", height=280)
            fig_eq.update_layout(**BB_LAYOUT)
            st.plotly_chart(fig_eq, use_container_width=True)

    with eq_col2:
        # Operating Leverage = %ΔEBIT / %ΔRevenue
        _ol_ebit_d, _ol_ebit_v = annual_series(ia, "Operating Income","Ebit","EBIT")
        _ol_rev_d,  _ol_rev_v  = annual_series(ia, "Total Revenue","Revenue")
        if _ol_ebit_v and _ol_rev_v and len(_ol_ebit_v) >= 2 and len(_ol_rev_v) >= 2:
            _n_ol = min(len(_ol_ebit_v), len(_ol_rev_v))
            _ol_years = _ol_ebit_d[1:_n_ol]
            _ol_vals  = []
            for i in range(1, _n_ol):
                try:
                    ebit_chg = (_ol_ebit_v[i] - _ol_ebit_v[i-1]) / abs(_ol_ebit_v[i-1])
                    rev_chg  = (_ol_rev_v[i]  - _ol_rev_v[i-1])  / abs(_ol_rev_v[i-1])
                    ol = ebit_chg / rev_chg if rev_chg and rev_chg != 0 else None
                    _ol_vals.append(ol)
                except: _ol_vals.append(None)
            if _ol_vals:
                _ol_colors = ["#ff3b3b" if v and v > 4 else "#ffb347" if v and v > 2 else "#00d084"
                               for v in _ol_vals]
                fig_ol = go.Figure(go.Bar(
                    x=_ol_years, y=_ol_vals, marker_color=_ol_colors, marker_line_width=0,
                    text=[f"{v:.1f}x" if v else "" for v in _ol_vals],
                    textposition="outside", textfont=dict(color="#e8e8e8", size=9)
                ))
                fig_ol.add_hline(y=1.0, line_dash="dot", line_color="#555",
                                  annotation_text="1.0 = neutral leverage", annotation_font_color="#555")
                fig_ol.update_layout(title="Operating Leverage  (ΔEBIT% ÷ ΔRevenue%)",
                                      yaxis_title="Degree of Operating Leverage", height=280)
                fig_ol.update_layout(**BB_LAYOUT)
                st.plotly_chart(fig_ol, use_container_width=True)
                st.caption("High operating leverage (>3x) amplifies both upside and downside in a revenue slowdown.")


# ══════════════════════════════════════════════════════════════
# TAB 7 — DIVIDENDS
# ══════════════════════════════════════════════════════════════
with tab_div:
    section_header("DIVIDEND ANALYSIS", "YIELD · DPS · PAYOUT RATIO — HISTORY")

    divs  = data["dividends"]
    pe_v  = safe(info.get("trailingPE"))
    dps   = safe(info.get("lastDividendValue"))
    dyld  = safe(info.get("dividendYield"))
    pout  = safe(info.get("payoutRatio"))
    ex_dt = info.get("exDividendDate")
    d5y   = safe(info.get("fiveYearAvgDividendYield"))

    d1,d2,d3,d4,d5 = st.columns(5)
    d1.metric("Current Yield",   pct(dyld))
    d2.metric("DPS (Last)",      f"₹{dps:.2f}" if dps else "N/A")
    d3.metric("Payout Ratio",    pct(pout))
    d4.metric("5Y Avg Yield",    f"{d5y:.2f}%" if d5y else "N/A")
    try:
        ex_dt_str = datetime.fromtimestamp(int(ex_dt)).strftime("%d %b %Y") if ex_dt else "N/A"
    except Exception:
        ex_dt_str = str(ex_dt) if ex_dt else "N/A"
    d5.metric("Ex-Dividend Date", ex_dt_str)

    if not divs.empty:
        divs.index = pd.to_datetime(divs.index).tz_localize(None)

        # Annual DPS
        annual_dps = divs.resample("YE").sum()
        if not annual_dps.empty:
            fig_dps = go.Figure()
            fig_dps.add_trace(go.Bar(
                x=[str(d.year) for d in annual_dps.index],
                y=annual_dps.values,
                marker_color="#ff8c00", marker_line_width=0,
                text=[f"₹{v:.1f}" for v in annual_dps.values],
                textposition="outside", textfont=dict(color="#e8e8e8", size=9)
            ))
            fig_dps.update_layout(title="Annual Dividend Per Share  (₹)",
                                   yaxis_title="₹ per share", height=280)
            fig_dps.update_layout(**BB_LAYOUT)
            st.plotly_chart(fig_dps, use_container_width=True)

        # Dividend yield history (price-relative)
        ph = data["price_history"]
        if not ph.empty:
            ph.index = pd.to_datetime(ph.index).tz_localize(None)
            annual_dps2 = divs.resample("YE").sum().reindex(
                pd.date_range(ph.index[0], ph.index[-1], freq="YE"), fill_value=0
            )
            annual_price = ph["Close"].resample("YE").mean()
            if len(annual_dps2) > 0 and len(annual_price) > 0:
                common = annual_dps2.index.intersection(annual_price.index)
                if len(common) > 0:
                    hist_yield = (annual_dps2[common] / annual_price[common] * 100).dropna()
                    fig_dy = go.Figure(go.Scatter(
                        x=[str(d.year) for d in hist_yield.index],
                        y=hist_yield.values,
                        mode="lines+markers",
                        line=dict(color="#00d084", width=2),
                        fill="tozeroy", fillcolor="rgba(0,208,132,0.08)"
                    ))
                    mean_yield = hist_yield.mean()
                    fig_dy.add_hline(y=mean_yield, line_dash="dot", line_color="#888",
                                     annotation_text=f"Avg {mean_yield:.2f}%",
                                     annotation_font_color="#888")
                    fig_dy.update_layout(title="Historical Dividend Yield %",
                                          yaxis_title="Yield %", height=240)
                    fig_dy.update_layout(**BB_LAYOUT)
                    st.plotly_chart(fig_dy, use_container_width=True)

        # Payout Ratio trend
        ia = data["income_annual"]
        ni_d2, ni_v2 = annual_series(ia, "Net Income", "Net income")
        shares_ = safe(info.get("sharesOutstanding"))
        if ni_v2 and shares_ and not annual_dps.empty:
            annual_dps_al = annual_dps.values
            eps_al = [n/shares_ if n is not None else None for n in ni_v2[-len(annual_dps_al):]]
            dps_al = list(annual_dps_al[-len(eps_al):])
            if eps_al and dps_al:
                payout_hist = [d/e*100 if e and e != 0 else None for d, e in zip(dps_al, eps_al)]
                fig_po = go.Figure(go.Bar(
                    x=ni_d2[-len(payout_hist):], y=payout_hist,
                    marker_color=["#ffb347" if v and 30<=v<=60 else "#00d084" if v and v<30 else "#ff3b3b"
                                   for v in payout_hist],
                    marker_line_width=0,
                    text=[f"{v:.0f}%" if v else "" for v in payout_hist],
                    textposition="outside", textfont=dict(color="#e8e8e8", size=9)
                ))
                fig_po.add_hline(y=75, line_dash="dot", line_color="#ff3b3b",
                                  annotation_text="75% = high payout risk", annotation_font_color="#ff3b3b")
                fig_po.update_layout(title="Dividend Payout Ratio %  (DPS ÷ EPS × 100)",
                                      yaxis_title="Payout %", height=240)
                fig_po.update_layout(**BB_LAYOUT)
                st.plotly_chart(fig_po, use_container_width=True)
    else:
        st.info(f"{t_sym} has not paid dividends or dividend history is not available.")


# ══════════════════════════════════════════════════════════════
# TAB 8 — EARNINGS
# ══════════════════════════════════════════════════════════════
with tab_earn:
    section_header("EARNINGS ANALYSIS", "QUARTERLY EPS TREND · ESTIMATE BEAT/MISS · SURPRISE %")

    iq = data["income_quarterly"]

    # ── Quarterly EPS trend ──
    qni_d, qni_v  = quarterly_series(iq, "Net Income", "Net income")
    shares_ = safe(info.get("sharesOutstanding"))

    if qni_v and shares_:
        qeps = [n/shares_ if n else None for n in qni_v]
        n_show = min(12, len(qeps))

        colors_eps = ["#00d084" if v and v >= 0 else "#ff3b3b" for v in qeps[-n_show:]]
        fig_qeps = go.Figure(go.Bar(
            x=qni_d[-n_show:], y=qeps[-n_show:],
            marker_color=colors_eps, marker_line_width=0,
            text=[f"₹{v:.2f}" if v else "" for v in qeps[-n_show:]],
            textposition="outside", textfont=dict(color="#e8e8e8", size=9)
        ))
        fig_qeps.update_layout(title=f"Quarterly EPS  (last {n_show} quarters)",
                                yaxis_title="₹ per share", height=300)
        fig_qeps.update_layout(**BB_LAYOUT)
        st.plotly_chart(fig_qeps, use_container_width=True)

        # QoQ & YoY EPS growth
        if len(qeps) >= 5:
            qoq = [(qeps[i] - qeps[i-1])/abs(qeps[i-1])*100
                   if qeps[i] and qeps[i-1] and qeps[i-1] != 0 else None
                   for i in range(1, len(qeps))]
            yoy = [(qeps[i] - qeps[i-4])/abs(qeps[i-4])*100
                   if i >= 4 and qeps[i] and qeps[i-4] and qeps[i-4] != 0 else None
                   for i in range(len(qeps))]

            gc1, gc2 = st.columns(2)
            with gc1:
                colors_qoq = ["#00d084" if v and v>=0 else "#ff3b3b" for v in qoq[-8:]]
                fig_qoq = go.Figure(go.Bar(
                    x=qni_d[1:][-8:], y=qoq[-8:], marker_color=colors_qoq, marker_line_width=0,
                    text=[f"{v:.1f}%" if v else "" for v in qoq[-8:]],
                    textposition="outside", textfont=dict(color="#e8e8e8", size=9)
                ))
                fig_qoq.update_layout(title="EPS QoQ Growth %", yaxis_title="%", height=240)
                fig_qoq.update_layout(**BB_LAYOUT)
                st.plotly_chart(fig_qoq, use_container_width=True)
            with gc2:
                yoy_clean = [(d, v) for d, v in zip(qni_d[4:], yoy[4:]) if v is not None]
                if yoy_clean:
                    y_d, y_v = zip(*yoy_clean)
                    colors_yoy = ["#00d084" if v>=0 else "#ff3b3b" for v in y_v]
                    fig_yoy = go.Figure(go.Bar(
                        x=list(y_d)[-8:], y=list(y_v)[-8:], marker_color=colors_yoy[-8:],
                        marker_line_width=0,
                        text=[f"{v:.1f}%" for v in list(y_v)[-8:]],
                        textposition="outside", textfont=dict(color="#e8e8e8", size=9)
                    ))
                    fig_yoy.update_layout(title="EPS YoY Growth %", yaxis_title="%", height=240)
                    fig_yoy.update_layout(**BB_LAYOUT)
                    st.plotly_chart(fig_yoy, use_container_width=True)

    # ── Earnings surprise — analyst estimates vs actuals ──
    section_header("EARNINGS SURPRISE HISTORY", "ACTUAL EPS vs ESTIMATE — BEAT / MISS")
    try:
        hist_earn = yf.Ticker(t_sym + ".NS").earnings_history
        if hist_earn is not None and not hist_earn.empty:
            hist_earn = hist_earn.copy()
            hist_earn["Surprise%"] = ((hist_earn["epsActual"] - hist_earn["epsEstimate"])
                                       / hist_earn["epsEstimate"].abs() * 100)
            hist_earn = hist_earn.sort_index().tail(8)
            e1, e2 = st.columns([2,1])
            with e1:
                colors_surp = ["#00d084" if v >= 0 else "#ff3b3b" for v in hist_earn["Surprise%"]]
                fig_surp = go.Figure(go.Bar(
                    x=[str(d)[:10] for d in hist_earn.index],
                    y=hist_earn["Surprise%"],
                    marker_color=colors_surp, marker_line_width=0,
                    text=[f"{v:+.1f}%" for v in hist_earn["Surprise%"]],
                    textposition="outside", textfont=dict(color="#e8e8e8", size=9)
                ))
                fig_surp.add_hline(y=0, line_color="#555")
                fig_surp.update_layout(title="EPS Surprise %  (Actual vs Estimate)",
                                        yaxis_title="Surprise %", height=260)
                fig_surp.update_layout(**BB_LAYOUT)
                st.plotly_chart(fig_surp, use_container_width=True)
            with e2:
                disp_earn = hist_earn[["epsEstimate","epsActual","Surprise%"]].copy()
                disp_earn.columns = ["Est EPS","Act EPS","Surprise%"]
                disp_earn = disp_earn.round(2)
                disp_earn.index = [str(d)[:10] for d in disp_earn.index]
                def surp_color(v):
                    if not isinstance(v,(int,float)): return ""
                    return "color:#00ff88;font-weight:700" if v > 5 else \
                           "color:#b8e06a" if v > 0 else \
                           "color:#ff3b3b;font-weight:700" if v < -5 else "color:#ffb347"
                st.dataframe(disp_earn.style.applymap(surp_color, subset=["Surprise%"]),
                             use_container_width=True)
        else:
            st.info("Analyst earnings estimates not available — yfinance may not have this data for NSE stocks.")
    except Exception as e:
        st.info(f"Earnings surprise data not available: {e}")


# ══════════════════════════════════════════════════════════════
# TAB 9 — PEER RANKING
# ══════════════════════════════════════════════════════════════
with tab_peer:
    section_header("PEER RANKING", "PERCENTILE RANK VS SECTOR ON EVERY KEY METRIC")

    all_tickers2 = [t_sym] + list(peers.keys())
    all_data2    = {t_sym: data}
    all_data2.update(peers)

    # Build comparison matrix
    METRIC_DEFS = [
        ("Trail P/E",    "trailingPE",           False, True),   # (label, info_key, pct_fmt, lower_better)
        ("Fwd P/E",      "forwardPE",             False, True),
        ("P/Book",       "priceToBook",           False, True),
        ("EV/EBITDA",    "enterpriseToEbitda",    False, True),
        ("Net Margin",   "profitMargins",         True,  False),
        ("ROE",          "returnOnEquity",        True,  False),
        ("ROA",          "returnOnAssets",        True,  False),
        ("Rev Growth",   "revenueGrowth",         True,  False),
        ("EPS Growth",   "earningsGrowth",        True,  False),
        ("D/E Ratio",    "debtToEquity",          False, True),
        ("Current Ratio","currentRatio",          False, False),
        ("Div Yield",    "dividendYield",         True,  False),
        ("Beta",         "beta",                  False, True),
    ]

    rows = []
    for sym in all_tickers2:
        d_inf = all_data2[sym]["info"]
        row = {"Ticker": sym, "Sector": STOCK_SECTOR.get(sym, "?")}
        for lbl, key, is_pct, lower_better in METRIC_DEFS:
            v = safe(d_inf.get(key))
            if v is not None:
                row[lbl] = round(v*100, 2) if is_pct else round(v, 2)
            else:
                row[lbl] = None
        rows.append(row)

    comp_df = pd.DataFrame(rows).set_index("Ticker")

    # Compute percentile ranks
    rank_df = comp_df.copy()
    metric_cols = [m[0] for m in METRIC_DEFS]
    for lbl, _, _, lower_better in METRIC_DEFS:
        if lbl not in rank_df.columns: continue
        col = rank_df[lbl].dropna()
        if len(col) < 2: continue
        if lower_better:
            rank_df[lbl + " Rank"] = col.rank(ascending=True, pct=True) * 100
        else:
            rank_df[lbl + " Rank"] = col.rank(ascending=False, pct=True) * 100

    # ── Main comparison table ──
    st.dataframe(
        comp_df.style.apply(
            lambda col: [
                "background-color:#0d2200;color:#00ff88;font-weight:700"
                if isinstance(v, (int,float)) and not np.isnan(v) and v == col.dropna().max()
                else "background-color:#1f0000;color:#ff3b3b"
                if isinstance(v, (int,float)) and not np.isnan(v) and v == col.dropna().min()
                else ""
                for v in col
            ], axis=0
        ),
        use_container_width=True
    )
    st.caption(f"Green = best in group  ·  Red = worst in group  ·  {t_sym} highlighted in amber row")

    # ── Radar chart — focus stock vs sector average ──
    section_header("RADAR — FOCUS STOCK vs SECTOR AVERAGE")
    radar_metrics = ["Net Margin","ROE","ROA","Rev Growth","EPS Growth","Div Yield"]
    radar_available = [m for m in radar_metrics if m in comp_df.columns]

    if len(radar_available) >= 4:
        sector_avg = comp_df[radar_available].mean()
        focus_vals = comp_df.loc[t_sym, radar_available] if t_sym in comp_df.index else pd.Series()

        if not focus_vals.empty:
            # Normalise 0-1 for radar
            _max = comp_df[radar_available].max()
            _min = comp_df[radar_available].min()
            def norm_radar(s):
                return ((s - _min) / (_max - _min + 1e-9)).clip(0, 1)

            focus_norm = norm_radar(focus_vals)
            avg_norm   = norm_radar(sector_avg)

            fig_rad = go.Figure()
            fig_rad.add_trace(go.Scatterpolar(
                r=list(focus_norm) + [focus_norm.iloc[0]],
                theta=radar_available + [radar_available[0]],
                fill="toself", fillcolor="rgba(255,140,0,0.15)",
                line=dict(color="#ff8c00", width=2), name=t_sym
            ))
            fig_rad.add_trace(go.Scatterpolar(
                r=list(avg_norm) + [avg_norm.iloc[0]],
                theta=radar_available + [radar_available[0]],
                fill="toself", fillcolor="rgba(30,144,255,0.1)",
                line=dict(color="#1e90ff", width=2, dash="dot"), name="Sector Avg"
            ))
            fig_rad.update_layout(
                polar=dict(
                    bgcolor="#0a0a0a",
                    radialaxis=dict(visible=True, range=[0,1], tickfont=dict(color="#555", size=8),
                                    gridcolor="#1a1a1a"),
                    angularaxis=dict(tickfont=dict(color="#e8e8e8", size=10), gridcolor="#2a2a2a"),
                ),
                paper_bgcolor="#0a0a0a",
                font=dict(color="#e8e8e8", family="IBM Plex Mono", size=10),
                legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#e8e8e8")),
                height=400
            )
            st.plotly_chart(fig_rad, use_container_width=True)

    # ── Bar chart: ROE vs Peers ──
    roe_peer = [(s, safe(all_data2[s]["info"].get("returnOnEquity"))) for s in all_tickers2]
    roe_peer = [(s, v*100) for s, v in roe_peer if v is not None]
    if roe_peer:
        syms_r, vals_r = zip(*roe_peer)
        colors_r = ["#ff8c00" if s == t_sym else "#2a2a2a" for s in syms_r]
        fig_roe = go.Figure(go.Bar(
            x=list(syms_r), y=list(vals_r),
            marker_color=colors_r, marker_line_width=0,
            text=[f"{v:.1f}%" for v in vals_r], textposition="outside",
            textfont=dict(color="#e8e8e8", size=9)
        ))
        fig_roe.add_hline(y=sum(vals_r)/len(vals_r), line_dash="dot", line_color="#888",
                           annotation_text="Group avg", annotation_font_color="#888")
        fig_roe.update_layout(title="ROE % — Peer Comparison",
                               yaxis_title="ROE %", height=260)
        fig_roe.update_layout(**BB_LAYOUT)
        st.plotly_chart(fig_roe, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# TAB 10 — ANALYST RATINGS
# ══════════════════════════════════════════════════════════════
with tab_analyst:
    section_header("ANALYST CONSENSUS", "PRICE TARGETS · RATINGS · UPSIDE/DOWNSIDE · TIMEFRAME")

    apt = data.get("analyst_price_targets", {})
    recs = data.get("recommendations", pd.DataFrame())

    curr   = safe(info.get("currentPrice") or info.get("regularMarketPrice"))
    t_mean = safe(apt.get("mean"))             if isinstance(apt, dict) else None
    t_high = safe(apt.get("high"))             if isinstance(apt, dict) else None
    t_low  = safe(apt.get("low"))              if isinstance(apt, dict) else None
    n_ana  = safe(apt.get("numberOfAnalysts")) if isinstance(apt, dict) else None

    a1,a2,a3,a4,a5 = st.columns(5)
    a1.metric("Current Price",  f"₹{curr:.2f}"   if curr   else "N/A")
    a2.metric("Analyst Target", f"₹{t_mean:.2f}" if t_mean else "N/A",
              delta=f"{(t_mean/curr-1)*100:.1f}% upside" if t_mean and curr else None)
    a3.metric("Target High",    f"₹{t_high:.2f}" if t_high else "N/A")
    a4.metric("Target Low",     f"₹{t_low:.2f}"  if t_low  else "N/A")
    a5.metric("# Analysts",     f"{int(n_ana)}"  if n_ana  else "N/A")

    # ── Upside banner + TIMEFRAME TO TARGET ──
    if curr and t_mean:
        upside    = (t_mean - curr) / curr * 100
        up_color  = "#00ff88" if upside > 10 else "#ffb347" if upside > 0 else "#ff3b3b"

        # Timeframe: derived from analyst EPS growth + P/E re-rating
        # Logic: years = ln(target/cmp) / ln(1 + expected_annual_return)
        # Expected annual return uses forward EPS growth as proxy for earnings CAGR
        _fwd_pe_a  = safe(info.get("forwardPE"))
        _trail_pe_a= safe(info.get("trailingPE"))
        _earn_gr_a = safe(info.get("earningsGrowth")) or safe(info.get("revenueGrowth"))
        timeframe_str = "N/A"
        timeframe_note = ""
        if upside > 0 and _earn_gr_a and _earn_gr_a > 0.01:
            # Implied years at current earnings growth rate to justify target price
            # Target P/E is assumed to compress to midpoint of current and forward P/E
            _target_return = _earn_gr_a  # minimum return = earnings growth
            try:
                import math
                _years = math.log(t_mean / curr) / math.log(1 + _target_return)
                _years = round(_years, 1)
                if _years <= 0.5:   timeframe_str = "< 6 months"
                elif _years <= 1.0: timeframe_str = "6–12 months"
                elif _years <= 1.5: timeframe_str = "12–18 months"
                elif _years <= 2.0: timeframe_str = "12–24 months"
                elif _years <= 3.0: timeframe_str = "2–3 years"
                else:               timeframe_str = f"~{_years:.1f} years"
                timeframe_note = f"Based on {_earn_gr_a*100:.1f}% earnings growth as annual return driver"
            except: pass
        elif upside <= 0:
            timeframe_str = "Overvalued vs target"

        if t_high and t_low:
            st.markdown(f"""
<div style="font-family:'IBM Plex Mono';margin:12px 0;padding:14px 18px;
     background:#0d0d0d;border:1px solid #2a2a2a;border-left:4px solid {up_color};
     display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px;">
  <div>
    <div style="color:#888;font-size:.58rem;letter-spacing:.1em;">CONSENSUS UPSIDE TO MEAN TARGET</div>
    <div style="color:{up_color};font-size:1.5rem;font-weight:700;margin:4px 0;">{upside:+.1f}%</div>
    <div style="color:#555;font-size:.62rem;">CMP ₹{curr:.2f}  →  ₹{t_mean:.2f}  (Range ₹{t_low:.0f}–₹{t_high:.0f})</div>
  </div>
  <div style="text-align:right;">
    <div style="color:#888;font-size:.58rem;letter-spacing:.1em;">ESTIMATED TIMEFRAME TO TARGET</div>
    <div style="color:#ff8c00;font-size:1.1rem;font-weight:700;margin:4px 0;">⏱ {timeframe_str}</div>
    <div style="color:#555;font-size:.58rem;">{timeframe_note}</div>
  </div>
</div>""", unsafe_allow_html=True)

    # ── Price target gauge ──
    if curr and t_low and t_high:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=curr,
            delta={"reference": t_mean, "valueformat": ".1f", "prefix": "vs Target ₹"},
            number={"prefix": "₹", "font": {"color": "#ff8c00", "size": 28}},
            gauge={
                "axis": {"range": [t_low * 0.85, t_high * 1.1],
                         "tickcolor": "#555", "tickfont": {"color": "#888"}},
                "bar":  {"color": "#ff8c00", "thickness": 0.25},
                "steps": [
                    {"range": [t_low * 0.85, t_low],  "color": "#2a0000"},
                    {"range": [t_low, t_mean],          "color": "#1a1200"},
                    {"range": [t_mean, t_high],         "color": "#0d2200"},
                    {"range": [t_high, t_high * 1.1],  "color": "#0a3300"},
                ],
                "threshold": {"line": {"color": "#00ff88", "width": 2}, "value": t_mean},
                "bgcolor": "#0a0a0a", "bordercolor": "#2a2a2a",
            },
            title={"text": "CMP vs Analyst Target Range", "font": {"color": "#ff8c00", "size": 12}},
        ))
        fig_gauge.update_layout(height=280, paper_bgcolor="#0a0a0a",
                                 font=dict(color="#e8e8e8", family="IBM Plex Mono"))
        st.plotly_chart(fig_gauge, use_container_width=True)

    # ══════════════════════════════
    # REVERSE DCF
    # ══════════════════════════════
    section_header("REVERSE DCF — IMPLIED GROWTH RATE", "WHAT GROWTH IS THE CURRENT PRICE ALREADY PRICING IN?")
    try:
        _cmp    = curr or safe(info.get("currentPrice"))
        _eps_r  = safe(info.get("trailingEps")) or safe(info.get("forwardEps"))
        _shares_r = safe(info.get("sharesOutstanding"))
        _fcf_r  = safe(info.get("freeCashflow"))
        _mc_r   = safe(info.get("marketCap"))
        _wacc   = 0.12   # Conservative WACC for Indian market (12%)
        _tg     = 0.04   # Terminal growth (long-run nominal GDP)
        _yrs    = 10     # DCF horizon

        # Reverse DCF: solve for g such that DCF(FCF, g, wacc, tg, yrs) = market cap
        # NPV = sum(FCF*(1+g)^t / (1+wacc)^t, t=1..n) + TV / (1+wacc)^n = MC
        # Numerically solved via bisection
        if _fcf_r and _mc_r and _fcf_r > 0:
            def dcf_value(g):
                pv = sum(_fcf_r * (1+g)**t / (1+_wacc)**t for t in range(1, _yrs+1))
                tv = (_fcf_r * (1+g)**_yrs * (1+_tg)) / (_wacc - _tg) / (1+_wacc)**_yrs
                return pv + tv

            # Bisection: find g where dcf_value(g) = market_cap
            lo, hi = -0.30, 1.00
            implied_g = None
            for _ in range(60):
                mid = (lo + hi) / 2
                if dcf_value(mid) < _mc_r:
                    lo = mid
                else:
                    hi = mid
            implied_g = (lo + hi) / 2

            # Compare implied g to actual growth
            _actual_g  = safe(info.get("earningsGrowth")) or safe(info.get("revenueGrowth"))
            _fwd_eps_gr = (safe(info.get("forwardEps")) / safe(info.get("trailingEps")) - 1) if safe(info.get("trailingEps")) and safe(info.get("forwardEps")) and safe(info.get("trailingEps")) != 0 else None

            rd1, rd2, rd3, rd4 = st.columns(4)
            rd1.metric("Implied Growth Rate",   f"{implied_g*100:.1f}%",
                        help="The FCF growth rate the market is pricing in at current price")
            rd2.metric("Actual Earnings Growth", f"{_actual_g*100:.1f}%" if _actual_g else "N/A",
                        help="Trailing 12M earnings growth from yfinance")
            rd3.metric("Fwd EPS Growth (1Y)",   f"{_fwd_eps_gr*100:.1f}%" if _fwd_eps_gr else "N/A",
                        help="Forward EPS vs Trailing EPS — 1-year implied growth")
            rd4.metric("WACC Used",             f"{_wacc*100:.0f}%",
                        help="Assumed cost of capital. Conservative 12% for NSE stocks")

            # Verdict
            if _actual_g is not None:
                gap = _actual_g - implied_g
                if gap > 0.05:
                    verdict = f"UNDERVALUED — actual growth ({_actual_g*100:.1f}%) exceeds implied ({implied_g*100:.1f}%) by {gap*100:.1f}pp"
                    v_color = "#00ff88"
                elif gap < -0.05:
                    verdict = f"OVERVALUED — price implies {implied_g*100:.1f}% growth but actual is only {_actual_g*100:.1f}%"
                    v_color = "#ff3b3b"
                else:
                    verdict = f"FAIRLY VALUED — implied growth ({implied_g*100:.1f}%) ≈ actual ({_actual_g*100:.1f}%)"
                    v_color = "#ffb347"
                st.markdown(f"""
<div style="font-family:'IBM Plex Mono';font-size:.68rem;color:{v_color};font-weight:700;
     padding:8px 12px;background:#0d0d0d;border-left:3px solid {v_color};margin-top:8px;">
  ◼ REVERSE DCF VERDICT: {verdict}
</div>
<div style="font-family:'IBM Plex Mono';font-size:.58rem;color:#555;margin-top:4px;padding:0 4px;">
  Assumes WACC={_wacc*100:.0f}%, terminal growth={_tg*100:.0f}%, {_yrs}Y horizon. Based on FCF = {fmt_cr(_fcf_r)}.
  This is a model — real valuations depend on qualitative factors not captured here.
</div>""", unsafe_allow_html=True)

            # Sensitivity: show target price at different growth assumptions
            section_header("REVERSE DCF SENSITIVITY", "FAIR VALUE AT DIFFERENT GROWTH ASSUMPTIONS")
            growth_scenarios = [-0.05, 0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
            scen_rows = []
            for g_s in growth_scenarios:
                fv = dcf_value(g_s)
                updown = (fv / _mc_r - 1) * 100 if _mc_r else None
                per_share = fv / _shares_r if _shares_r else None
                scen_rows.append({
                    "FCF Growth Assumed": f"{g_s*100:.0f}%",
                    "DCF Fair Value (₹Cr)": fmt_cr(fv),
                    "Fair Value/Share": f"₹{per_share:.0f}" if per_share else "N/A",
                    "Upside/Downside": f"{updown:+.1f}%" if updown is not None else "N/A",
                    "vs Current": "▲ UPSIDE" if updown and updown > 0 else "▼ DOWNSIDE",
                })
            scen_df = pd.DataFrame(scen_rows).set_index("FCF Growth Assumed")

            def scen_color(v):
                if not isinstance(v, str): return ""
                if "UPSIDE" in v: return "color:#00ff88;font-weight:700"
                if "DOWNSIDE" in v: return "color:#ff3b3b;font-weight:700"
                return ""
            st.dataframe(scen_df.style.applymap(scen_color, subset=["vs Current"]),
                          use_container_width=True)
        else:
            st.info("Reverse DCF requires positive FCF. This company may be pre-profit or FCF data is unavailable.")
    except Exception as e:
        st.info(f"Reverse DCF: {e}")

    # ── Recommendations history ──
    if recs is not None and not recs.empty:
        section_header("RECOMMENDATION HISTORY", "ANALYST UPGRADES / DOWNGRADES")
        try:
            recs_disp = recs.copy()
            if "period" in recs_disp.columns:
                recs_disp = recs_disp.set_index("period")
            buy_cols  = [c for c in recs_disp.columns if any(x in c.lower() for x in ["buy","strong buy"])]
            hold_cols = [c for c in recs_disp.columns if "hold" in c.lower()]
            sell_cols = [c for c in recs_disp.columns if any(x in c.lower() for x in ["sell","underperform","underweight"])]
            if buy_cols or hold_cols or sell_cols:
                recs_disp["Buy"]  = recs_disp[buy_cols].sum(axis=1)  if buy_cols  else 0
                recs_disp["Hold"] = recs_disp[hold_cols].sum(axis=1) if hold_cols else 0
                recs_disp["Sell"] = recs_disp[sell_cols].sum(axis=1) if sell_cols else 0
                rec_plot = recs_disp[["Buy","Hold","Sell"]].tail(8)
                fig_rec = go.Figure()
                fig_rec.add_trace(go.Bar(x=rec_plot.index, y=rec_plot["Buy"],  name="Buy",  marker_color="#00d084"))
                fig_rec.add_trace(go.Bar(x=rec_plot.index, y=rec_plot["Hold"], name="Hold", marker_color="#ffb347"))
                fig_rec.add_trace(go.Bar(x=rec_plot.index, y=rec_plot["Sell"], name="Sell", marker_color="#ff3b3b"))
                fig_rec.update_layout(title="Analyst Recommendations — Buy / Hold / Sell",
                                       barmode="stack", yaxis_title="# Analysts", height=280)
                fig_rec.update_layout(**BB_LAYOUT)
                st.plotly_chart(fig_rec, use_container_width=True)
            else:
                st.dataframe(recs_disp.tail(8), use_container_width=True)
        except Exception as e:
            st.info(f"Could not render recommendation chart: {e}")
    else:
        st.info("Analyst recommendations not available via yfinance for this stock.")

    # ── Key info expander ──
    with st.expander("◼ FULL COMPANY PROFILE (click to expand)"):
        bio = info.get("longBusinessSummary","")
        if bio:
            st.markdown(f"""
<div style="font-family:'IBM Plex Mono';font-size:.68rem;color:#c8c8c8;line-height:1.7;
     background:#0d0d0d;padding:12px;border:1px solid #2a2a2a;">
{bio[:800]}{'...' if len(bio)>800 else ''}
</div>""", unsafe_allow_html=True)
        kp_rows = [
            ("Exchange",        info.get("exchange","")),
            ("Industry",        info.get("industry","")),
            ("Country",         info.get("country","")),
            ("Employees",       f"{info.get('fullTimeEmployees','N/A'):,}" if info.get("fullTimeEmployees") else "N/A"),
            ("Fiscal Year End", (lambda v: datetime.fromtimestamp(int(v)).strftime("%b %Y")
                                 if isinstance(v, (int, float)) else str(v)[:7]
                                 if v else "N/A")(info.get("mostRecentQuarter"))),
            ("Shares Out",      f"{safe(info.get('sharesOutstanding'))/1e7:.1f} Cr" if safe(info.get("sharesOutstanding")) else "N/A"),
            ("Float",           f"{safe(info.get('floatShares'))/1e7:.1f} Cr"       if safe(info.get("floatShares"))       else "N/A"),
            ("Promoter %",      f"{safe(info.get('heldPercentInsiders'))*100:.1f}%"  if safe(info.get("heldPercentInsiders"))      else "N/A"),
            ("Institution %",   f"{safe(info.get('heldPercentInstitutions'))*100:.1f}%" if safe(info.get("heldPercentInstitutions")) else "N/A"),
            ("Net Debt",        fmt_cr(net_debt) if net_debt is not None else "Net Cash"),
            ("Beta",            f"{beta:.2f}" if beta else "N/A"),
        ]
        kp_df = pd.DataFrame(kp_rows, columns=["Field","Value"]).set_index("Field")
        st.dataframe(kp_df, use_container_width=True)
