# ============================================================
# MONARCH OPTIONS INTELLIGENCE ENGINE
# pages/option.py  — standalone Streamlit page
#
# Fully self-contained. Reads Upstox token from shared file.
# No dependency on screener_pro.py runtime state.
#
# Inputs:
#   • Underlying symbol (index or stock)
#   • Expiry date (fetched live from Upstox)
#   • DTE override / risk-free rate
#
# Outputs:
#   • Directional Bias (7-factor technical model)
#   • Volatility Regime (IV rank vs HV)
#   • Strategy Recommendations (bias × vol × dte)
#   • Live Option Chain with signals
#   • Greeks Dashboard (BS engine, no external lib)
#   • OI Analysis (max pain, PCR, walls)
#   • Interactive Payoff Builder
# ============================================================

import streamlit as st
import requests, gzip, json, time, io, math, urllib.parse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import yfinance as yf
import os

st.set_page_config(layout="wide", page_title="MONARCH — Options Intel")

# ============================================================
# CENTRALIZED CONFIGURATION — change once, applies everywhere
# ============================================================
CFG = {
    "rfr_default":    6.5,    # Risk-free rate % (India repo rate)
    "hv_window":      20,     # Historical volatility look-back days
    "hv_window_fast": 10,     # Fast HV for comparison
    "iv_hist_max":    252,    # Max IV history (1 trading year)
    "chain_strikes":  8,      # Strikes shown each side in chain tab
    "oi_strikes":     10,     # Strikes each side in OI tab
    "pain_strikes":   15,     # Strikes each side for max pain calc
    "iv_overpriced":  1.20,   # IV ratio above HV → sell signal
    "iv_underpriced": 0.85,   # IV ratio below HV → buy signal
    "chain_cache_ttl": 30,    # Seconds to cache live option chain
    "expiry_cache_ttl": 300,  # Seconds to cache expiry list
    "master_cache_ttl": 3600, # Seconds to cache instrument master
}

# ── Bloomberg Terminal Theme ──────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&display=swap');
:root {
    --bb-bg: #0a0a0a; --bb-surface: #111111; --bb-border: #2a2a2a;
    --bb-amber: #ff8c00; --bb-amber2: #ffb347; --bb-green: #00d084;
    --bb-red: #ff3b3b; --bb-blue: #1e90ff; --bb-white: #e8e8e8; --bb-muted: #888888;
}

/* ── Base ── */
html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"],
.main, .block-container, [data-testid="stVerticalBlock"] {
    background-color: var(--bb-bg) !important;
    color: var(--bb-white) !important;
    font-family: 'IBM Plex Mono', monospace !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"], [data-testid="stSidebar"] > div {
    background-color: #060606 !important;
    border-right: 1px solid var(--bb-border) !important;
}
[data-testid="stSidebar"] * { color: var(--bb-white) !important; font-family: 'IBM Plex Mono', monospace !important; }
[data-testid="stSidebar"] label { color: var(--bb-muted) !important; font-size: .65rem !important; }
[data-testid="stSidebar"] .stDivider, [data-testid="stSidebar"] hr { border-color: var(--bb-border) !important; }

/* ── Typography ── */
h1 { font-family: 'IBM Plex Mono', monospace !important; color: var(--bb-amber) !important;
     font-size: 1.0rem !important; font-weight: 600 !important; letter-spacing: .15em !important;
     text-transform: uppercase !important; border-bottom: 1px solid var(--bb-amber) !important; padding-bottom: 4px !important; }
h2 { color: var(--bb-amber2) !important; font-size: .85rem !important; letter-spacing: .1em !important; text-transform: uppercase !important; }
h3 { color: var(--bb-white) !important; font-size: .78rem !important; letter-spacing: .08em !important; }
p, li, span, div { font-family: 'IBM Plex Mono', monospace !important; }

/* ── st.caption / small text ── */
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p,
small, .stCaption, .caption { color: var(--bb-muted) !important; font-size: .62rem !important; }

/* ── st.markdown prose text ── */
[data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li, [data-testid="stMarkdownContainer"] td,
[data-testid="stMarkdownContainer"] th {
    color: var(--bb-white) !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: .72rem !important;
}
[data-testid="stMarkdownContainer"] strong { color: var(--bb-amber2) !important; }
[data-testid="stMarkdownContainer"] code {
    background: #1a1400 !important; color: var(--bb-amber) !important;
    padding: 1px 4px !important; border-radius: 0 !important; font-size: .68rem !important;
}
[data-testid="stMarkdownContainer"] table { border-collapse: collapse !important; width: 100% !important; }
[data-testid="stMarkdownContainer"] th { background: #1a1400 !important; color: var(--bb-amber) !important; border: 1px solid var(--bb-border) !important; padding: 4px 8px !important; font-size: .62rem !important; }
[data-testid="stMarkdownContainer"] td { border: 1px solid var(--bb-border) !important; padding: 4px 8px !important; color: var(--bb-white) !important; font-size: .65rem !important; }

/* ── Metrics ── */
[data-testid="metric-container"] {
    background: var(--bb-surface) !important; border: 1px solid var(--bb-border) !important;
    border-left: 3px solid var(--bb-amber) !important; padding: 8px 12px !important; border-radius: 0 !important;
}
[data-testid="metric-container"] label, [data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] p { color: var(--bb-muted) !important; font-size: .58rem !important; letter-spacing: .1em !important; text-transform: uppercase !important; }
[data-testid="stMetricValue"], [data-testid="stMetricValue"] div { color: var(--bb-amber) !important; font-size: 1rem !important; font-weight: 600 !important; }
[data-testid="stMetricDelta"] { font-size: .62rem !important; }
[data-testid="stMetricDelta"] svg { display: none !important; }

/* ── DataFrames ── */
[data-testid="stDataFrame"] { border: 1px solid var(--bb-border) !important; }
[data-testid="stDataFrame"] *, .stDataFrame * { background-color: transparent !important; }
[data-testid="stDataFrame"] [data-testid="stDataFrameResizable"] { background: var(--bb-surface) !important; }
.stDataFrame thead tr th { background: #1a1400 !important; color: var(--bb-amber) !important; font-size: .62rem !important; letter-spacing: .1em !important; text-transform: uppercase !important; border-bottom: 1px solid var(--bb-amber) !important; }
.stDataFrame tbody tr td { font-size: .68rem !important; color: var(--bb-white) !important; border-bottom: 1px solid #1a1a1a !important; background: var(--bb-surface) !important; }
.stDataFrame tbody tr:hover td { background: #1a1400 !important; }
/* Streamlit 1.x iframe-based dataframe */
.stDataFrame iframe { background: var(--bb-surface) !important; }

/* ── Buttons ── */
.stButton > button { background: #1a1400 !important; color: var(--bb-amber) !important; border: 1px solid var(--bb-amber) !important;
    border-radius: 0 !important; font-size: .7rem !important; letter-spacing: .1em !important; text-transform: uppercase !important; padding: 6px 14px !important; }
.stButton > button:hover { background: var(--bb-amber) !important; color: #000 !important; }
.stButton > button:disabled { opacity: .4 !important; }

/* ── Inputs: text, number, selectbox ── */
.stSelectbox > div > div, .stTextInput > div > div, .stNumberInput > div > div {
    background: var(--bb-surface) !important; border: 1px solid var(--bb-border) !important;
    border-radius: 0 !important; color: var(--bb-white) !important; font-size: .72rem !important; }
.stSelectbox label, .stTextInput label, .stNumberInput label {
    color: var(--bb-muted) !important; font-size: .62rem !important; font-family: 'IBM Plex Mono', monospace !important; }
.stSelectbox div[data-baseweb="select"] > div { background: var(--bb-surface) !important; color: var(--bb-white) !important; }
/* Dropdown popup list */
ul[data-baseweb="menu"], [data-baseweb="popover"], [data-baseweb="popover"] li {
    background: #1a1a1a !important; color: var(--bb-white) !important; font-family: 'IBM Plex Mono', monospace !important; font-size: .7rem !important; border: 1px solid var(--bb-border) !important; }
[data-baseweb="option"]:hover { background: #1a1400 !important; }
/* Number input spinners */
.stNumberInput button { background: var(--bb-surface) !important; color: var(--bb-muted) !important; border: 1px solid var(--bb-border) !important; }
input[type="number"], input[type="text"], input[type="password"] {
    background: var(--bb-surface) !important; color: var(--bb-white) !important;
    border: 1px solid var(--bb-border) !important; font-family: 'IBM Plex Mono', monospace !important; font-size: .72rem !important; border-radius: 0 !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] { background: var(--bb-surface) !important; border-bottom: 1px solid var(--bb-border) !important; gap: 0 !important; }
.stTabs [data-baseweb="tab"] { background: transparent !important; color: var(--bb-muted) !important; font-size: .65rem !important; letter-spacing: .1em !important; text-transform: uppercase !important; border-radius: 0 !important; border-right: 1px solid var(--bb-border) !important; padding: 8px 14px !important; }
.stTabs [aria-selected="true"] { background: #1a1400 !important; color: var(--bb-amber) !important; border-bottom: 2px solid var(--bb-amber) !important; }
[data-testid="stTabContent"] { background: var(--bb-bg) !important; }

/* ── Expanders ── */
[data-testid="stExpander"] { background: var(--bb-surface) !important; border: 1px solid var(--bb-border) !important; border-radius: 0 !important; }
[data-testid="stExpander"] summary { color: var(--bb-amber2) !important; font-size: .66rem !important; letter-spacing: .08em !important; font-family: 'IBM Plex Mono', monospace !important; }
[data-testid="stExpander"] summary:hover { color: var(--bb-amber) !important; background: #1a1400 !important; }
[data-testid="stExpander"] svg { fill: var(--bb-amber2) !important; stroke: var(--bb-amber2) !important; }
[data-testid="stExpanderDetails"] { background: var(--bb-bg) !important; border-top: 1px solid var(--bb-border) !important; }

/* ── Divider ── */
hr, [data-testid="stDivider"] { border-color: var(--bb-border) !important; margin: 8px 0 !important; }
[data-testid="stDivider"] hr { border-top: 1px solid var(--bb-border) !important; }

/* ── Spinner / info / warning ── */
[data-testid="stAlert"] { background: var(--bb-surface) !important; border: 1px solid var(--bb-border) !important; color: var(--bb-white) !important; border-radius: 0 !important; font-size: .7rem !important; }
[data-testid="stAlert"] p { color: var(--bb-white) !important; }
.stInfo { border-left: 3px solid var(--bb-blue) !important; }
.stWarning { border-left: 3px solid var(--bb-amber) !important; }
.stError { border-left: 3px solid var(--bb-red) !important; }
.stSuccess { border-left: 3px solid var(--bb-green) !important; }

/* ── Spinner text ── */
[data-testid="stSpinner"] p, .stSpinner p { color: var(--bb-amber) !important; }

/* ── Scrollbars ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--bb-bg); }
::-webkit-scrollbar-thumb { background: var(--bb-border); }
::-webkit-scrollbar-thumb:hover { background: var(--bb-amber); }

/* ── Hide Streamlit branding ── */
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] { display: none !important; }

/* ── Expander arrow overlap fix (from previous session) ── */
[data-testid="stExpander"] summary svg,
[data-testid="stExpander"] summary span[data-testid="StyledFullScreenButton"],
[data-testid="stExpander"] summary > div > svg { display: none !important; visibility: hidden !important; width: 0 !important; font-size: 0 !important; }
</style>
""", unsafe_allow_html=True)

# Terminal header
st.markdown("""
<div style="background:#ff8c00;color:#000;font-family:'IBM Plex Mono',monospace;
font-size:.62rem;font-weight:600;letter-spacing:.18em;padding:5px 14px;
display:flex;justify-content:space-between;margin-bottom:12px;">
  <span>◼ MONARCH OPTIONS INTELLIGENCE — NSE F&O</span>
  <span>OPTIONS · DERIVATIVES · STRATEGY ENGINE</span>
</div>
""", unsafe_allow_html=True)

# ============================================================
# TOKEN — shared with screener_pro.py
# ============================================================
TOKEN_FILE = ".upstox_token_scanner"

if "opt_token_loaded" not in st.session_state:
    if os.path.exists(TOKEN_FILE):
        try:
            with open(TOKEN_FILE, "r") as f:
                st.session_state.opt_access_token = f.read().strip()
        except:
            st.session_state.opt_access_token = ""
    else:
        st.session_state.opt_access_token = ""
    st.session_state.opt_token_loaded = True

with st.sidebar:
    st.markdown("### 🔑 Upstox Token")
    tok_inp = st.text_input("Access Token", type="password",
                             value=st.session_state.opt_access_token, key="opt_tok_inp")
    if tok_inp and tok_inp != st.session_state.opt_access_token:
        st.session_state.opt_access_token = tok_inp
        try:
            with open(TOKEN_FILE, "w") as f: f.write(tok_inp)
            st.success("Token saved ✔")
        except: pass

ACCESS_TOKEN = st.session_state.opt_access_token
HEADERS = {"Authorization": f"Bearer {ACCESS_TOKEN}", "Accept": "application/json"}

if not ACCESS_TOKEN:
    st.warning("⚠️  Paste your Upstox access token in the sidebar to continue.")
    st.stop()

# ============================================================
# BLACK-SCHOLES ENGINE (pure Python, no scipy needed)
# ============================================================

def _ncdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def _npdf(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

def bs_price(S, K, T, r, sigma, opt="call"):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max(S - K, 0) if opt == "call" else max(K - S, 0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if opt == "call":
        return S * _ncdf(d1) - K * math.exp(-r * T) * _ncdf(d2)
    return K * math.exp(-r * T) * _ncdf(-d2) - S * _ncdf(-d1)

def bs_greeks(S, K, T, r, sigma, opt="call"):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return dict(delta=1.0 if opt=="call" else -1.0, gamma=0, theta=0, vega=0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    nd1 = _npdf(d1)
    delta = _ncdf(d1) if opt == "call" else _ncdf(d1) - 1
    gamma = nd1 / (S * sigma * math.sqrt(T))
    theta = (-(S * nd1 * sigma) / (2 * math.sqrt(T)) -
              r * K * math.exp(-r * T) * (_ncdf(d2) if opt=="call" else _ncdf(-d2))) / 365
    vega  = S * nd1 * math.sqrt(T) / 100
    return dict(delta=round(delta,4), gamma=round(gamma,6),
                theta=round(theta,4), vega=round(vega,4))

def implied_vol(mkt_px, S, K, T, r, opt="call"):
    if T <= 0 or mkt_px <= 0: return None
    sig = 0.30
    for _ in range(200):
        px   = bs_price(S, K, T, r, sig, opt)
        d1   = (math.log(max(S/K,1e-9)) + (r + 0.5*sig**2)*T) / (sig*math.sqrt(T))
        vega = S * _npdf(d1) * math.sqrt(T)
        if vega < 1e-10: break
        diff = mkt_px - px
        if abs(diff) < 1e-6: break
        sig += diff / vega
        sig  = max(0.001, min(sig, 10.0))
    return round(sig, 6) if 0.001 < sig < 9.9 else None

def bs_itm_prob(S, K, T, r, sigma, opt="call"):
    """Probability option finishes ITM = N(d2) for call, N(-d2) for put.
    This is the risk-neutral probability of exercise."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 1.0 if (opt == "call" and S > K) else (1.0 if (opt == "put" and S < K) else 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return round(_ncdf(d2) if opt == "call" else _ncdf(-d2), 4)

def atm_strike(spot, step):
    return round(round(spot / step) * step, 2)

def strikes_around(spot, step, n=6):
    atm = atm_strike(spot, step)
    return [round(atm + i * step, 2) for i in range(-n, n+1)]

# ============================================================
# UPSTOX API HELPERS
# ============================================================

@st.cache_data(ttl=3600)
def load_fno_master():
    try:
        url = "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz"
        r   = requests.get(url, timeout=12)
        if r.status_code == 200:
            with gzip.GzipFile(fileobj=io.BytesIO(r.content)) as gz:
                return pd.DataFrame(json.load(gz))
    except: pass
    return pd.DataFrame()

def find_instrument_key(master_df, symbol):
    """Find the NSE index or EQ instrument key for a symbol."""
    if master_df.empty: return None
    sym = symbol.upper().strip()
    for itype in ["INDEX", "EQ", "FUTIDX", "FUTSTK"]:
        mask = (
            master_df.get("trading_symbol", pd.Series(dtype=str))
                      .astype(str).str.upper() == sym
        ) & (
            master_df.get("instrument_type", pd.Series(dtype=str))
                      .astype(str).str.upper() == itype
        )
        rows = master_df[mask]
        if not rows.empty:
            return str(rows.iloc[0]["instrument_key"])
    # fallback: any match
    mask2 = master_df.get("trading_symbol", pd.Series(dtype=str)).astype(str).str.upper() == sym
    rows2 = master_df[mask2]
    return str(rows2.iloc[0]["instrument_key"]) if not rows2.empty else None

@st.cache_data(ttl=300, show_spinner=False)
def fetch_expiries(_token, instrument_key):
    """Fetch available option expiry dates for an instrument."""
    url = "https://api.upstox.com/v2/option/contract"
    hdrs = {"Authorization": f"Bearer {_token}", "Accept": "application/json"}
    try:
        r = requests.get(url, headers=hdrs, params={"instrument_key": instrument_key}, timeout=10)
        if r.status_code == 200:
            data = r.json().get("data", [])
            return sorted(set(d["expiry"] for d in data if d.get("expiry")))
    except: pass
    return []

@st.cache_data(ttl=CFG["chain_cache_ttl"], show_spinner=False)
def fetch_option_chain(_token, instrument_key, expiry):
    """Fetch live option chain from Upstox. Cached for 30s to avoid hammering API."""
    url = "https://api.upstox.com/v2/option/chain"
    hdrs = {"Authorization": f"Bearer {_token}", "Accept": "application/json"}
    try:
        r = requests.get(url, headers=hdrs,
                         params={"instrument_key": instrument_key, "expiry_date": expiry},
                         timeout=15)
        if r.status_code == 200:
            return r.json().get("data", [])
        else:
            st.warning(f"Chain API error {r.status_code}: {r.text[:200]}")
    except Exception as e:
        st.warning(f"Chain fetch failed: {e}")
    return []

def fetch_spot_quote(instrument_key):
    """Fetch live spot price from Upstox market-quote."""
    url    = "https://api.upstox.com/v2/market-quote/quotes"
    params = {"instrument_key": instrument_key}
    try:
        r = requests.get(url, headers=HEADERS, params=params, timeout=10)
        if r.status_code == 200:
            data = r.json().get("data", {})
            for v in data.values():
                lp = v.get("last_price")
                if lp: return float(lp)
    except: pass
    return None

def parse_chain(raw_data, spot, step=50):
    """Parse Upstox option chain into a clean DataFrame."""
    rows = []
    if not raw_data: return pd.DataFrame()
    items = raw_data if isinstance(raw_data, list) else raw_data.get("options", [])
    for item in items:
        strike = item.get("strike_price") or item.get("strike")
        if not strike: continue
        ce = item.get("call_options", {}) or {}
        pe = item.get("put_options",  {}) or {}
        def g(d, *keys):
            for k in keys:
                if k in d: return d[k]
            md = d.get("market_data", {}) or {}
            for k in keys:
                if k in md: return md[k]
            return 0
        rows.append({
            "Strike":  float(strike),
            "CE_LTP":  g(ce,"last_price","ltp")        or 0,
            "CE_OI":   g(ce,"open_interest","oi")      or 0,
            "CE_OIC":  g(ce,"oi_day_change","oichange") or 0,
            "CE_Vol":  g(ce,"volume","vol")             or 0,
            "CE_IV":   g(ce,"implied_volatility","iv")  or 0,
            "CE_Bid":  g(ce,"bid_price","bid")          or 0,
            "CE_Ask":  g(ce,"ask_price","ask")          or 0,
            "PE_LTP":  g(pe,"last_price","ltp")        or 0,
            "PE_OI":   g(pe,"open_interest","oi")      or 0,
            "PE_OIC":  g(pe,"oi_day_change","oichange") or 0,
            "PE_Vol":  g(pe,"volume","vol")             or 0,
            "PE_IV":   g(pe,"implied_volatility","iv")  or 0,
            "PE_Bid":  g(pe,"bid_price","bid")          or 0,
            "PE_Ask":  g(pe,"ask_price","ask")          or 0,
        })
    df = pd.DataFrame(rows).sort_values("Strike").reset_index(drop=True)
    if not df.empty:
        df["PCR"]       = df.apply(lambda r: round(r.PE_OI / (r.CE_OI + 1e-9), 3), axis=1)
        df["OI_Diff"]   = df["CE_OI"] - df["PE_OI"]
        df["Moneyness"] = df["Strike"].apply(
            lambda k: "ATM" if abs(k - spot) <= 0.5 * step
                      else ("ITM-C" if k < spot else "OTM-C"))
    return df

# ============================================================
# HISTORICAL DATA (yfinance fallback + Upstox)
# ============================================================

YF_TICKERS = {
    "NIFTY":      "^NSEI",
    "BANKNIFTY":  "^NSEBANK",
    "FINNIFTY":   "^CNXFIN",
    "MIDCPNIFTY": "^CNXMIDCAP",
    "SENSEX":     "^BSESN",
}

def get_ohlcv(symbol, token):
    """Get daily OHLCV. Tries Upstox historical first, then yfinance."""
    # yfinance
    yftick = YF_TICKERS.get(symbol.upper(), f"{symbol.upper()}.NS")
    try:
        d = yf.download(yftick, period="1y", interval="1d", progress=False, auto_adjust=True)
        if not d.empty:
            d = d.copy()
            d.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in d.columns]
            d = d.reset_index()
            d.columns = [c.lower() for c in d.columns]
            return d
    except: pass
    return pd.DataFrame()

def compute_hv(close_series, window=20):
    lr = np.log(close_series / close_series.shift(1)).dropna()
    if len(lr) < window: return None
    return float(lr.tail(window).std() * np.sqrt(252))

# ============================================================
# DIRECTIONAL ANALYSIS — 7-FACTOR MODEL
# ============================================================

def directional_bias(df, ltp):
    if df.empty or len(df) < 50:
        return {"bias":"NEUTRAL","score":0,"factors":{},"rsi":50,"macd_hist":0,
                "bb_pct":50,"vol_ratio":1,"atr_pct":1.5,"e9":ltp,"e20":ltp,"e50":ltp,"atr":ltp*0.015}
    c  = df["close"].astype(float)
    h  = df["high"].astype(float)
    l  = df["low"].astype(float)
    v  = df["volume"].astype(float)

    e9   = c.ewm(span=9,   adjust=False).mean()
    e20  = c.ewm(span=20,  adjust=False).mean()
    e50  = c.ewm(span=50,  adjust=False).mean()
    e200 = c.ewm(span=200, adjust=False).mean()

    tr   = pd.concat([h-l, (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
    atr  = tr.ewm(span=14, adjust=False).mean()
    atrv = float(atr.iloc[-1])

    delta = c.diff()
    gain  = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rsi   = float((100 - 100/(1 + gain/loss.replace(0, np.nan))).iloc[-1])

    macd_l  = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    macd_s  = macd_l.ewm(span=9, adjust=False).mean()
    macd_h  = float(macd_l.iloc[-1] - macd_s.iloc[-1])

    bm  = c.rolling(20).mean()
    bs  = c.rolling(20).std()
    bup = float((bm + 2*bs).iloc[-1])
    blo = float((bm - 2*bs).iloc[-1])
    bb_pct = (ltp - blo) / (bup - blo + 1e-9)

    vol_ma5  = float(v.tail(5).mean())
    vol_ma20 = float(v.tail(20).mean())
    vol_ratio = vol_ma5 / (vol_ma20 + 1e-9)

    e9v   = float(e9.iloc[-1]);  e20v  = float(e20.iloc[-1])
    e50v  = float(e50.iloc[-1]); e200v = float(e200.iloc[-1]) if len(df) >= 200 else ltp

    factors = {}; score = 0

    # Normaliser for MACD histogram: rolling std of histogram over last 60 bars
    # Avoids hard ±10 flip — score is proportional to histogram magnitude
    macd_hist_series = macd_l - macd_s
    macd_std = float(macd_hist_series.tail(60).std()) or 1e-9

    # F1: EMA Stack (±20) — 5 binary checks, ±4 each = ±20 total
    # Uniform weights: each check equally important
    es = 0
    for price, ref in [(ltp,e9v),(ltp,e20v),(ltp,e50v),(e9v,e20v),(e20v,e50v)]:
        es += 4 if price > ref else -4
    factors["EMA Stack"]  = es; score += es

    # F2: RSI — continuous tanh centred at 50, max ±15
    # tanh((rsi−50)/15): RSI 65→+0.83×15, RSI 35→−0.83×15
    rs = round(15 * math.tanh((rsi - 50.0) / 15.0), 1)
    factors["RSI(14)"]    = rs; score += rs

    # F3: MACD Histogram — continuous tanh, normalised by rolling std (max ±10)
    # tanh(hist / std): at 1 std = ±0.76×10, at 2 std = ±0.96×10
    # Avoids the binary ±10 flip that ignores magnitude
    ms = round(10 * math.tanh(macd_h / macd_std), 1)
    factors["MACD Hist"]  = ms; score += ms

    # F4: Bollinger %B — linear map [0,1] → [−10,+10]
    bs2 = round(max(-10.0, min(10.0, 10 * (2 * bb_pct - 1))), 1)
    factors["BB Position"]= bs2; score += bs2

    # F5: Volume — continuous tanh centred at 1.0×, max ±10
    vs = round(10 * math.tanh((vol_ratio - 1.0) / 0.5), 1)
    factors["Volume"]     = vs; score += vs

    # F6: 200 EMA — continuous tanh normalised by ATR (max ±15)
    # Distance from 200 EMA measured in ATR units: 0 ATR → 0, 1 ATR → tanh(1)×15 ≈ ±11.5
    # This avoids binary ±15 flip and makes the score proportional to distance/volatility
    atr_norm = atrv if atrv > 0 else (ltp * 0.01)
    e2 = round(15 * math.tanh((ltp - e200v) / atr_norm), 1)
    factors["200 EMA"]    = e2; score += e2

    # F7: 5D momentum — continuous tanh centred at 0%, max ±10
    # tanh(ret5 / 2.0): +3%→+0.905×10, +1%→+0.46×10
    if len(c) >= 6:
        base = float(c.iloc[-6])
        ret5 = (ltp/base - 1)*100 if base != 0 else 0
        m5 = round(10 * math.tanh(ret5 / 2.0), 1)
    else:
        m5 = 0
    factors["5D Return"]  = m5; score += m5

    bias = ("STRONGLY BULLISH" if score >= 30 else "BULLISH"   if score >= 12 else
            "NEUTRAL"          if score >  -12 else "BEARISH"  if score >= -30 else "STRONGLY BEARISH")

    return {
        "bias": bias, "score": int(round(score)), "rsi": round(rsi,1),
        "macd_hist": round(macd_h,3), "bb_pct": round(bb_pct*100,1),
        "vol_ratio": round(vol_ratio,2),
        "atr_pct":   round(atrv/ltp*100,2) if ltp>0 else 0,
        "e9": round(e9v,2), "e20": round(e20v,2), "e50": round(e50v,2),
        "atr": round(atrv,2), "factors": factors
    }

# ============================================================
# VOLATILITY REGIME
# ============================================================

def iv_rank(iv_series, current_iv):
    s = pd.Series(iv_series).dropna()
    if len(s) < 3: return 50.0
    lo, hi = s.min(), s.max()
    return round((current_iv - lo)/(hi - lo + 1e-9)*100, 1)

def vol_regime(ivr):
    # IV Rank quartile boundaries — principled percentile cuts:
    # < 25  = bottom quartile  → structurally cheap vol → BUY premium
    # 25–50 = second quartile  → below-median vol → slight buy lean
    # 50–75 = third quartile   → above-median vol → slight sell lean
    # 75–90 = elevated         → lean SELL premium
    # > 90  = top decile       → extreme premium → strong SELL signal
    if   ivr < 25: return "LOW VOL",      "BUY premium — debit spreads / long options / straddles", "#1e90ff"
    elif ivr < 50: return "NORMAL-LOW",   "Slight buy lean — calendars / ratio spreads",            "#7ec8e3"
    elif ivr < 75: return "NORMAL-HIGH",  "Slight sell lean — balanced spreads, light credits",     "#ffb347"
    elif ivr < 90: return "ELEVATED",     "Lean SELL — credit spreads / iron condor",               "#ff8c00"
    else:          return "HIGH VOL",     "SELL premium — iron condors / strangles / short straddle","#ff3b3b"

# ============================================================
# OI ANALYSIS
# ============================================================

def oi_analysis(chain_df, spot, step=50, T=0.02, r=0.065, atm_iv=0.20):
    """OI analysis with GEX, gamma flip, OI cluster walls, and skew metrics."""
    if chain_df.empty: return {}
    atm_approx = round(round(spot / step) * step, 2)
    df = chain_df[(chain_df.Strike >= atm_approx - CFG["pain_strikes"]*step) &
                  (chain_df.Strike <= atm_approx + CFG["pain_strikes"]*step)].copy()
    if df.empty: return {}

    def pain(ep, d):
        return (((ep - d.Strike).clip(lower=0) * d.CE_OI) +
                ((d.Strike - ep).clip(lower=0) * d.PE_OI)).sum()

    pain_map = {row.Strike: pain(row.Strike, df) for _, row in df.iterrows()}
    max_pain  = min(pain_map, key=pain_map.get) if pain_map else spot

    total_ce  = float(df.CE_OI.sum())
    total_pe  = float(df.PE_OI.sum())
    pcr_oi    = round(total_pe / (total_ce + 1e-9), 3)

    # ── OI cluster walls (3-strike sliding window — smoother than single-strike max) ──
    def oi_cluster_peak(oi_col, strikes_list):
        if len(strikes_list) < 3: return float(strikes_list[oi_col.argmax()])
        best_k, best_sum = strikes_list[0], 0
        for i in range(1, len(strikes_list) - 1):
            s = oi_col.iloc[i-1] + oi_col.iloc[i] + oi_col.iloc[i+1]
            if s > best_sum:
                best_sum = s
                best_k   = strikes_list[i]
        return float(best_k)

    call_wall = oi_cluster_peak(df.CE_OI.reset_index(drop=True), list(df.Strike))
    put_wall  = oi_cluster_peak(df.PE_OI.reset_index(drop=True), list(df.Strike))

    atm_r    = df.iloc[(df.Strike - spot).abs().argsort()[:1]]
    straddle = float(atm_r.CE_LTP.values[0] + atm_r.PE_LTP.values[0]) if not atm_r.empty else 0
    exp_move     = round(straddle / spot * 100, 2) if spot > 0 else 0
    exp_move_2sd = round(exp_move * 2.0, 2)

    # ── PCR signal ──
    pcr_values_in_chain = df["PCR"].replace([np.inf, -np.inf], np.nan).dropna()
    pcr_pct = float(
        (pcr_values_in_chain <= pcr_oi).mean() * 100
        if len(pcr_values_in_chain) > 0 else 50.0
    )
    if   pcr_pct >= 75: pcr_sig = "BULLISH — aggregate PCR in top quartile; heavy put writing = support"
    elif pcr_pct >= 55: pcr_sig = "SLIGHT BULLISH LEAN — PCR above median; put OI outweighs calls"
    elif pcr_pct >= 45: pcr_sig = "NEUTRAL — PCR near median; balanced OI both sides"
    elif pcr_pct >= 25: pcr_sig = "SLIGHT BEARISH LEAN — PCR below median; call OI building"
    else:               pcr_sig = "BEARISH — aggregate PCR in bottom quartile; heavy call writing = resistance"

    # ── Gamma Exposure (GEX) ──
    # GEX per strike = gamma × OI × spot (dollar-gamma, per-unit)
    # Dealer convention: dealers are net SHORT options, so their GEX = -(buyer's GEX)
    # Net GEX = calls positive, puts negative (from dealer perspective)
    t_safe = max(T, 1.0/365.0)
    gex_rows = []
    for _, row in df.iterrows():
        iv_c = float(row.CE_IV)
        iv_p = float(row.PE_IV)
        iv_c = max((iv_c/100 if iv_c > 2 else iv_c), 0.01) if iv_c else atm_iv
        iv_p = max((iv_p/100 if iv_p > 2 else iv_p), 0.01) if iv_p else atm_iv
        g_ce = bs_greeks(spot, float(row.Strike), t_safe, r, iv_c, "call")["gamma"]
        g_pe = bs_greeks(spot, float(row.Strike), t_safe, r, iv_p, "put")["gamma"]
        net  = (g_ce * float(row.CE_OI) - g_pe * float(row.PE_OI)) * spot
        gex_rows.append({"Strike": float(row.Strike), "NET_GEX": net})

    gex_df       = pd.DataFrame(gex_rows)
    net_gex_total = float(gex_df["NET_GEX"].sum())
    gex_regime   = ("POSITIVE GEX — range-bound / vol suppressed (dealers buy dips & sell rallies)"
                    if net_gex_total >= 0 else
                    "NEGATIVE GEX — trending / vol expansion likely (dealers chase price)")

    # ── Gamma Flip Level (strike where cumulative GEX crosses zero) ──
    gex_sorted = gex_df.copy()
    gex_sorted["dist"] = (gex_sorted.Strike - spot).abs()
    gex_sorted = gex_sorted.sort_values("dist").reset_index(drop=True)
    cum_gex    = gex_sorted["NET_GEX"].cumsum()
    gamma_flip = spot  # default
    for i in range(1, len(cum_gex)):
        if cum_gex.iloc[i-1] * cum_gex.iloc[i] <= 0:
            gamma_flip = float(gex_sorted.Strike.iloc[i])
            break

    # ── IV Skew (downside put IV vs upside call IV at ±1 strike) ──
    skew_val, skew_label = None, "—"
    try:
        dn1   = df.iloc[(df.Strike - (spot - step)).abs().argsort()[:1]]
        up1   = df.iloc[(df.Strike - (spot + step)).abs().argsort()[:1]]
        dn_iv = float(dn1.PE_IV.values[0])
        up_iv = float(up1.CE_IV.values[0])
        dn_iv = dn_iv/100 if dn_iv > 2 else dn_iv
        up_iv = up_iv/100 if up_iv > 2 else up_iv
        if dn_iv > 0 and up_iv > 0:
            skew_val = round((dn_iv - up_iv) * 100, 2)
            if   skew_val >  3: skew_label = f"BEARISH SKEW +{skew_val:.1f}pp — put protection demand elevated"
            elif skew_val > -1: skew_label = f"NEUTRAL SKEW {skew_val:+.1f}pp — balanced demand"
            else:               skew_label = f"CALL SKEW {skew_val:+.1f}pp — upside speculation elevated"
    except: pass

    return dict(
        max_pain=round(max_pain,2), pcr_oi=pcr_oi,
        call_wall=round(call_wall,2), put_wall=round(put_wall,2),
        total_ce_oi=int(total_ce), total_pe_oi=int(total_pe),
        atm_straddle=round(straddle,2), exp_move_pct=exp_move,
        exp_move_2sd_pct=exp_move_2sd, pcr_signal=pcr_sig,
        net_gex=round(net_gex_total, 2), gex_regime=gex_regime,
        gamma_flip=round(gamma_flip, 2), gex_df=gex_df,
        skew_pp=skew_val, skew_label=skew_label,
    )

# ============================================================
# STRATEGY RECOMMENDATION ENGINE
# ============================================================

def recommend_strategies(bias, vol_lbl, dte, spot, atm, step, ivr, bias_score=0):
    is_bull  = "BULL" in bias
    is_bear  = "BEAR" in bias
    hi_vol   = ivr >= 75   # top quartile = sell premium
    lo_vol   = ivr < 25    # bottom quartile = buy premium
    sv       = float(step)
    recs     = []

    # ── Computed Fit Score ──────────────────────────────────────
    # Score = bias_alignment × vol_alignment × dte_alignment, normalised to 0–100
    #
    # bias_alignment:  how well strategy direction matches the bias score
    #   |bias_score| / 80 × 100  for directional, (80 - |bias_score|) / 80 × 100 for neutral
    #
    # vol_alignment:  does the strategy want cheap (buy) or rich (sell) vol?
    #   debit  strategy in low IV  → ivr distance from 0   = (100 - ivr) / 100
    #   credit strategy in high IV → ivr distance from 100 = ivr / 100
    #   neutral strategies         → 1 - |ivr - 50| / 50
    #
    # dte_alignment:  dte within the ideal DTE range, linearly scored
    #   parsed from "lo–hi DTE" string; 1.0 if in range, decays linearly outside
    #
    # Final: round(bias_align × vol_align × dte_align × 100, 0)

    abs_score = abs(bias_score)  # used by fit_score via closure

    def _dte_align(ideal_dte_str, actual_dte):
        """Return 0–1 DTE alignment. 1.0 if within range, exponential decay outside.
        Decay constant κ = ln(2) / half_range → score halves every half_range days outside."""
        import re
        nums = re.findall(r'\d+', ideal_dte_str.split("DTE")[0])
        if len(nums) >= 2:
            lo, hi = int(nums[0]), int(nums[-1])
            if lo <= actual_dte <= hi: return 1.0
            dist = min(abs(actual_dte - lo), abs(actual_dte - hi))
            half_range = max((hi - lo) / 2.0, 1.0)
            return math.exp(-math.log(2) * dist / half_range)
        elif len(nums) == 1:
            ref = int(nums[0])
            dist = abs(actual_dte - ref)
            return math.exp(-math.log(2) * dist / max(ref / 2.0, 1.0))
        return 0.5

    def fit_score(strategy_type, ideal_dte_str, bias_pts):
        """
        strategy_type: 'debit_directional' | 'debit_neutral' | 'credit_directional' | 'credit_neutral'
        bias_pts: absolute value of the overall bias score (0–80)
        """
        # 1. Bias alignment
        if "directional" in strategy_type:
            b_align = bias_pts / 80.0
        else:  # neutral: better when bias is weak
            b_align = max(0.0, 1.0 - bias_pts / 80.0)

        # 2. Vol alignment
        if "debit" in strategy_type:
            v_align = (100.0 - ivr) / 100.0   # cheap options favour debit
        else:  # credit
            v_align = ivr / 100.0             # rich options favour credit

        # 3. DTE alignment
        d_align = _dte_align(ideal_dte_str, dte)

        # 4. Composite (geometric mean keeps all three honest)
        raw = (b_align * v_align * d_align) ** (1.0/3.0)
        return int(round(raw * 100))

    def add(name, type_, legs, rationale, risk, reward, ideal_dte, _strat_type, _bias_pts):
        sc = fit_score(_strat_type, ideal_dte, _bias_pts)
        recs.append({"Strategy":name,"Type":type_,"Legs":legs,
                     "Rationale":rationale,"Max Risk":risk,"Max Reward":reward,
                     "Ideal DTE":ideal_dte,"Score":sc})

    # bias_pts = absolute directional conviction, 0–80
    _bp_dir = min(80, abs(bias_score))   # for directional strategies
    _bp_neut_base = min(80, max(0, 80 - abs(bias_score)))  # inverse: neutral best when bias is weak

    # ── BULLISH ──────────────────────────────
    if is_bull:
        _bp = _bp_dir
        if lo_vol:
            add("Long ATM Call",
                "Debit · Directional",
                f"BUY {atm} CE",
                "Low IV = cheap premium. Pure directional. Max loss = premium paid. Best when you expect a swift move.",
                "Premium paid","Unlimited","15–45 DTE", "debit_directional", _bp)
            add("Bull Call Spread",
                "Debit · Defined Risk",
                f"BUY {atm} CE  +  SELL {atm+sv:.0f} CE",
                "Cuts premium cost vs naked call. Profits if stock closes above upper strike at expiry.",
                "Net debit",f"Spread width − debit","15–30 DTE", "debit_directional", _bp)
            add("Call Ratio Backspread",
                "Credit–even · Vol + Direction",
                f"SELL 1× {atm-sv:.0f} CE  +  BUY 2× {atm} CE",
                "Enter for credit or zero cost. Profits from big upside move OR vol expansion. Limited loss in the middle.",
                "Limited (near short strike)","Unlimited above upper BE","30–45 DTE", "debit_directional", _bp)
        elif hi_vol:
            add("Bull Put Spread",
                "Credit · Defined Risk",
                f"SELL {atm} PE  +  BUY {atm-sv:.0f} PE",
                "Sell expensive puts. Keep the credit as long as stock stays above short strike. High IV = fat credit.",
                "Spread width − credit","Net credit received","7–21 DTE", "credit_directional", _bp)
            add("Short Put (OTM)",
                "Credit · Income",
                f"SELL {atm-sv:.0f} PE",
                "Collect rich premium. Obligated to buy stock at strike if assigned — only use for stocks you want to own.",
                "Strike − premium","Premium received","7–21 DTE", "credit_directional", _bp)
            add("Jade Lizard",
                "Credit · Slight Bullish",
                f"SELL {atm} PE  +  SELL {atm+sv:.0f} CE  +  BUY {atm+2*sv:.0f} CE",
                "No upside risk if total credit > call spread width. Benefits from high IV in both directions.",
                "Put strike − total credit","Total credit","14–21 DTE", "credit_directional", _bp)
        else:
            add("Bull Call Spread",
                "Debit · Defined Risk",
                f"BUY {atm} CE  +  SELL {atm+sv:.0f} CE",
                "Clean risk/reward. Wins on moderate upside move. Lower breakeven than naked call.",
                "Net debit","Spread width − debit","15–30 DTE", "debit_directional", _bp)
            add("Long OTM Call (+1)",
                "Debit · High Leverage",
                f"BUY {atm+sv:.0f} CE",
                "Cheaper than ATM. Higher leverage, needs bigger move. Good for event-driven plays.",
                "Premium paid","Unlimited","10–25 DTE", "debit_directional", _bp)

    # ── BEARISH ──────────────────────────────
    if is_bear:
        _bp = _bp_dir
        if lo_vol:
            add("Long ATM Put",
                "Debit · Directional",
                f"BUY {atm} PE",
                "Low IV = cheap downside protection. Pure directional. Max loss = premium paid.",
                "Premium paid","Strike − premium","15–45 DTE", "debit_directional", _bp)
            add("Bear Put Spread",
                "Debit · Defined Risk",
                f"BUY {atm} PE  +  SELL {atm-sv:.0f} PE",
                "Reduces cost vs naked put. Wins if stock falls below lower strike.",
                "Net debit","Spread width − debit","15–30 DTE", "debit_directional", _bp)
        elif hi_vol:
            add("Bear Call Spread",
                "Credit · Defined Risk",
                f"SELL {atm} CE  +  BUY {atm+sv:.0f} CE",
                "Sell expensive calls above current price. Keep credit if stock stays below short strike.",
                "Spread width − credit","Net credit","7–21 DTE", "credit_directional", _bp)
            add("Short Call (OTM)",
                "Credit · Aggressive",
                f"SELL {atm+sv:.0f} CE",
                "Rich call premium to sell. High risk — use only with clear bearish conviction and stop loss.",
                "Theoretically unlimited","Premium received","7–14 DTE", "credit_directional", _bp)
        else:
            add("Bear Put Spread",
                "Debit · Defined Risk",
                f"BUY {atm} PE  +  SELL {atm-sv:.0f} PE",
                "Balanced risk/reward for moderate downside. Standard short-term bearish play.",
                "Net debit","Spread width − debit","15–30 DTE", "debit_directional", _bp)
            add("Bear Call Spread",
                "Credit · Defined Risk",
                f"SELL {atm} CE  +  BUY {atm+sv:.0f} CE",
                "Collect premium above current price. Wins if stock stays flat or falls.",
                "Spread width − credit","Net credit","7–21 DTE", "credit_directional", _bp)

    # ── NEUTRAL / RANGE ──────────────────────
    if "NEUTRAL" in bias or hi_vol:
        _bp_neut = _bp_neut_base
        if hi_vol:
            add("Iron Condor",
                "Credit · Non-Directional",
                f"SELL {atm-sv:.0f} PE + BUY {atm-2*sv:.0f} PE  |  SELL {atm+sv:.0f} CE + BUY {atm+2*sv:.0f} CE",
                "Maximum premium collection in high IV. Wins if stock stays between short strikes. "
                "Most popular professional strategy for range-bound markets.",
                "Spread width − total credit","Total credit received","14–30 DTE", "credit_neutral", _bp_neut)
            add("Short Strangle",
                "Credit · Uncapped Risk",
                f"SELL {atm-sv:.0f} PE  +  SELL {atm+sv:.0f} CE",
                "Higher credit than iron condor. No wing protection = unlimited risk both sides. "
                "Must manage aggressively at 50% profit or 2× loss.",
                "Theoretically unlimited","Total premium","7–21 DTE", "credit_neutral", _bp_neut)
            add("Short Straddle",
                "Credit · Max Theta",
                f"SELL {atm} CE  +  SELL {atm} PE",
                "Maximum theta at ATM. Needs stock to pin very close to ATM. Highest risk — "
                "delta-hedge or exit quickly if stock moves.",
                "Unlimited both sides","Total premium","7–14 DTE", "credit_neutral", _bp_neut)
        elif lo_vol:
            add("Long Straddle",
                "Debit · Vol Expansion",
                f"BUY {atm} CE  +  BUY {atm} PE",
                "Low IV = cheap double. Profits from ANY large move either direction, or from IV expansion. "
                "Needs move > combined premium to profit.",
                "Combined premium paid","Unlimited","30–60 DTE", "debit_neutral", _bp_neut)
            add("Long Strangle",
                "Debit · Cheaper Vol Play",
                f"BUY {atm+sv:.0f} CE  +  BUY {atm-sv:.0f} PE",
                "Cheaper than straddle, needs bigger move. Excellent if you expect a large event-driven move.",
                "Combined premium","Unlimited","30–60 DTE", "debit_neutral", _bp_neut)
            add("Calendar Spread",
                "Debit · Theta + Vol",
                f"SELL near {atm} CE  +  BUY far {atm} CE",
                "Sell near-term theta, buy longer-dated vega. Profits from flat market + "
                "any IV expansion. Best when front-month IV > back-month IV.",
                "Net debit","Limited (peaks at ATM on front-month expiry)","Near:7–14 / Far:30–45 DTE", "debit_neutral", _bp_neut)
        else:
            add("Iron Condor",
                "Credit · Non-Directional",
                f"SELL {atm-sv:.0f} PE + BUY {atm-2*sv:.0f} PE  |  SELL {atm+sv:.0f} CE + BUY {atm+2*sv:.0f} CE",
                "Collect premium from both sides with defined risk. Ideal for a sideways market expectation.",
                "Spread width − credit","Total credit","14–30 DTE", "credit_neutral", _bp_neut)
            add("Iron Butterfly",
                "Credit · Tighter Range",
                f"SELL {atm} CE + SELL {atm} PE  |  BUY {atm+sv:.0f} CE + BUY {atm-sv:.0f} PE",
                "Higher credit than condor. Needs stock to stay near ATM. "
                "Better reward, narrower profit zone.",
                "Spread width − credit","Net credit","14–21 DTE", "credit_neutral", _bp_neut)
            add("ATM Butterfly",
                "Debit · Precision Play",
                f"BUY {atm-sv:.0f} CE  +  SELL 2× {atm} CE  +  BUY {atm+sv:.0f} CE",
                "Low cost, defined risk, maximum profit if stock pins ATM at expiry. "
                "Use when expecting consolidation around ATM.",
                "Net debit","Spread − 2×debit","7–21 DTE", "debit_neutral", _bp_neut)

    # DTE-based hedges
    if dte <= 5:
        add("Same-Day / Weekly Straddle Sell",
                "Credit · Near Expiry",
                f"SELL {atm} CE  +  SELL {atm} PE (weekly/near expiry)",
                "Near expiry = explosive theta decay. ATM options lose most value in last 2–5 days. "
                "Must monitor constantly and exit at 50% profit. NEVER hold to expiry naked.",
                "Large if stock moves","Theta collected","1–5 DTE", "credit_neutral", min(80, ivr))

    recs.sort(key=lambda x: x["Score"], reverse=True)
    return recs

# ============================================================
# SESSION STATE
# ============================================================
if "opt_chain_data"  not in st.session_state: st.session_state.opt_chain_data  = pd.DataFrame()
if "opt_spot"        not in st.session_state: st.session_state.opt_spot        = 0.0
if "opt_atm_iv"      not in st.session_state: st.session_state.opt_atm_iv      = 0.20
if "opt_hv20"        not in st.session_state: st.session_state.opt_hv20        = 0.20
if "opt_bias"        not in st.session_state: st.session_state.opt_bias        = {}
if "opt_oi"          not in st.session_state: st.session_state.opt_oi          = {}
if "opt_symbol"      not in st.session_state: st.session_state.opt_symbol      = "NIFTY"
if "opt_expiry"      not in st.session_state: st.session_state.opt_expiry      = ""
if "opt_dte"         not in st.session_state: st.session_state.opt_dte         = 7
if "opt_step"        not in st.session_state: st.session_state.opt_step        = 50
if "payoff_legs"     not in st.session_state: st.session_state.payoff_legs     = []
if "opt_iv_history" not in st.session_state: st.session_state.opt_iv_history   = {}  # {symbol: [iv, ...]} rolling 252-day IV log
if "opt_loaded"      not in st.session_state: st.session_state.opt_loaded      = False

# ============================================================
# SIDEBAR — ALL INPUTS
# ============================================================

OPTION_SYMBOLS = [
    "NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY",
    "RELIANCE","HDFCBANK","ICICIBANK","INFY","TCS","LT",
    "SBIN","AXISBANK","KOTAKBANK","BHARTIARTL","ITC",
    "BAJFINANCE","WIPRO","HCLTECH","TATAMOTORS","MARUTI",
    "SUNPHARMA","TITAN","ADANIENT","ONGC","NTPC","JSWSTEEL",
    "TATASTEEL","HINDALCO","DRREDDY","CIPLA","DIVISLAB",
    "BAJAJ-AUTO","HEROMOTOCO","EICHERMOT","M&M","TECHM",
    "INDUSINDBK","POWERGRID","COALINDIA","VEDL","SAIL",
]

STRIKE_STEPS = {"NIFTY":50,"BANKNIFTY":100,"FINNIFTY":50,"MIDCPNIFTY":25}
DEFAULT_STEP  = 5   # for stocks

with st.sidebar:
    st.divider()
    st.markdown("### ⚡ Options Setup")

    sym_sel  = st.selectbox("Underlying", OPTION_SYMBOLS,
                             index=OPTION_SYMBOLS.index(st.session_state.opt_symbol)
                             if st.session_state.opt_symbol in OPTION_SYMBOLS else 0,
                             key="sym_sel_sidebar")

    step_val = STRIKE_STEPS.get(sym_sel.upper(), DEFAULT_STEP)

    # Load expiries for selected symbol
    master_df = load_fno_master()
    ikey      = find_instrument_key(master_df, sym_sel)

    if ikey:
        expiry_list = fetch_expiries(ACCESS_TOKEN, ikey)
    else:
        expiry_list = []
        st.warning(f"No instrument key found for {sym_sel}")

    if expiry_list:
        expiry_sel = st.selectbox("Expiry Date", expiry_list, key="expiry_sel_sidebar")
    else:
        expiry_sel = st.text_input("Expiry (YYYY-MM-DD)", value="", key="expiry_text_sidebar")
        if not expiry_sel:
            st.info("Enter expiry date manually if list is empty")

    # Advanced inputs tucked away — only show if user needs to override
    with st.expander("⚙ Advanced Overrides", expanded=False):
        spot_override = st.number_input("Spot Price Override (0 = live)", min_value=0.0,
                                         value=0.0, step=1.0, key="spot_ovr_sidebar")
        dte_sidebar   = st.number_input("DTE override (0 = auto from expiry)", min_value=0, max_value=90,
                                         value=0, key="dte_sidebar")
        rfr_sidebar   = st.number_input("Risk-Free Rate %", min_value=0.0, max_value=15.0,
                                         value=6.5, step=0.1, key="rfr_sidebar")
    # Use sensible defaults when not overridden
    if "dte_sidebar" not in st.session_state or st.session_state.dte_sidebar == 0:
        dte_sidebar = 0  # will be computed from expiry date below
    if "rfr_sidebar" not in st.session_state:
        rfr_sidebar = 6.5

    st.divider()
    load_btn = st.button("⚡ LOAD OPTIONS INTEL", use_container_width=True, key="load_opt_main")

    # Show a clean status card once loaded — no raw number clutter
    if st.session_state.opt_loaded:
        _s = st.session_state
        _bres = _s.opt_bias
        _bc2  = {"STRONGLY BULLISH":"#00d084","BULLISH":"#7dca84","NEUTRAL":"#ffb347",
                 "BEARISH":"#ff7777","STRONGLY BEARISH":"#ff3b3b"}.get(_bres.get("bias","NEUTRAL"),"#888")
        st.markdown(f"""
<div style="background:#0d0d0d;border:1px solid #2a2a2a;border-left:3px solid {_bc2};
padding:8px 10px;font-family:'IBM Plex Mono',monospace;font-size:.6rem;margin-top:4px;">
  <div style="color:#555;letter-spacing:.08em;margin-bottom:3px;">LOADED</div>
  <div style="color:#ff8c00;font-weight:700;">{_s.opt_symbol} · {_s.opt_expiry}</div>
  <div style="color:#e8e8e8;">₹{_s.opt_spot:,.1f} · DTE {_s.opt_dte}</div>
  <div style="color:{_bc2};">{_bres.get('bias','—')} ({int(round(_bres.get('score',0))):+d})</div>
  <div style="color:#555;">IV {_s.opt_atm_iv*100:.1f}% · HV {_s.opt_hv20*100:.1f}%</div>
</div>""", unsafe_allow_html=True)
    else:
        st.caption(f"Strike Step: {step_val}")
        if ikey: st.caption(f"Key: {ikey[:30]}…")

# ============================================================
# LOAD LOGIC
# ============================================================

if load_btn:
    with st.spinner(f"Loading {sym_sel} options intelligence…"):

        # Validate expiry format before any API calls
        if expiry_sel:
            try:
                datetime.strptime(expiry_sel, "%Y-%m-%d")
            except ValueError:
                st.error(f"Invalid expiry format '{expiry_sel}'. Expected YYYY-MM-DD.")
                st.stop()

        # 1. Spot
        spot = spot_override if spot_override > 0 else (fetch_spot_quote(ikey) if ikey else None)
        if not spot:
            # yfinance fallback
            yftick = YF_TICKERS.get(sym_sel.upper(), f"{sym_sel.upper()}.NS")
            try:
                d = yf.download(yftick, period="2d", interval="1d", progress=False, auto_adjust=True)
                if not d.empty:
                    spot = float(d["Close"].iloc[-1])
            except: pass
        if not spot or spot <= 0:
            st.error(f"Could not get spot price for {sym_sel}. Use 'Spot Price Override'.")
            st.stop()

        # 2. Historical data
        ohlcv_df = get_ohlcv(sym_sel, ACCESS_TOKEN)
        hv20     = compute_hv(ohlcv_df["close"].astype(float), 20) if not ohlcv_df.empty else None
        hv10     = compute_hv(ohlcv_df["close"].astype(float), 10) if not ohlcv_df.empty else None

        # 3. Direction
        bias_res = directional_bias(ohlcv_df, spot)

        # 4. Option chain
        chain_raw = fetch_option_chain(ACCESS_TOKEN, ikey, expiry_sel) if ikey and expiry_sel else []
        chain_df  = parse_chain(chain_raw, spot, step_val)

        # 5. ATM IV
        atm_iv = None
        atm_k  = atm_strike(spot, step_val)
        if not chain_df.empty:
            row = chain_df.iloc[(chain_df.Strike - spot).abs().argsort()[:1]]
            if not row.empty:
                ce_iv_r = float(row.CE_IV.values[0])
                pe_iv_r = float(row.PE_IV.values[0])
                ce_iv   = ce_iv_r/100 if ce_iv_r > 2 else ce_iv_r
                pe_iv   = pe_iv_r/100 if pe_iv_r > 2 else pe_iv_r
                if ce_iv + pe_iv > 0: atm_iv = (ce_iv + pe_iv) / 2
        if not atm_iv:
            # Compute T from expiry date first (dte_sidebar may be 0 = auto)
            _dte_tmp = 7  # safe fallback
            if expiry_sel:
                try:
                    _exp_d  = datetime.strptime(expiry_sel, "%Y-%m-%d").date()
                    _dte_tmp = max(((_exp_d - datetime.now().date()).days), 1)
                except: pass
            elif dte_sidebar and dte_sidebar > 0:
                _dte_tmp = dte_sidebar
            T_tmp = _dte_tmp / 365.0
            if not chain_df.empty:
                row = chain_df.iloc[(chain_df.Strike - spot).abs().argsort()[:1]]
                if not row.empty:
                    strd = float(row.CE_LTP.values[0]) + float(row.PE_LTP.values[0])
                    if strd > 0 and T_tmp > 0:
                        # Brenner-Subrahmanyam (1988): ATM straddle ≈ spot × IV × sqrt(2/π) × sqrt(T)
                        # Rearranged: IV = straddle / (spot × sqrt(T) × sqrt(2/π))
                        # sqrt(2/π) ≈ 0.79788 — replaces the arbitrary 0.8
                        _bs_const = math.sqrt(2.0 / math.pi)  # 0.79788...
                        atm_iv = strd / (_bs_const * spot * math.sqrt(T_tmp)) if spot > 0 else None
        if not atm_iv:
            atm_iv = hv20 or 0.20

        # 6. OI
        _T_for_oi  = _dte_tmp / 365.0
        _rfr_for_oi = st.session_state.get("rfr_sidebar", CFG["rfr_default"]) / 100.0
        oi_d = oi_analysis(chain_df, spot, step_val, T=_T_for_oi, r=_rfr_for_oi, atm_iv=atm_iv or 0.20)

        # 7. DTE from expiry date (auto when dte_sidebar == 0)
        actual_dte = dte_sidebar if dte_sidebar and dte_sidebar > 0 else 7
        if expiry_sel:
            try:
                exp_d      = datetime.strptime(expiry_sel, "%Y-%m-%d").date()
                actual_dte = max((exp_d - datetime.now().date()).days, 1)
            except: pass

        # Append current ATM IV to rolling history for this symbol (max 252 trading days = ~1 year)
        _iv_hist = st.session_state.opt_iv_history
        _sym_key = sym_sel.upper()
        if _sym_key not in _iv_hist:
            _iv_hist[_sym_key] = []
        _iv_hist[_sym_key].append(atm_iv)
        if len(_iv_hist[_sym_key]) > 252:
            _iv_hist[_sym_key] = _iv_hist[_sym_key][-252:]
        st.session_state.opt_iv_history = _iv_hist

        # Store in session
        st.session_state.opt_chain_data = chain_df
        st.session_state.opt_spot       = spot
        st.session_state.opt_atm_iv     = atm_iv
        st.session_state.opt_hv20       = hv20 or 0.20
        st.session_state.opt_bias       = bias_res
        st.session_state.opt_oi         = oi_d
        st.session_state.opt_symbol     = sym_sel
        st.session_state.opt_expiry     = expiry_sel
        st.session_state.opt_dte        = actual_dte
        st.session_state.opt_step       = step_val
        st.session_state.opt_atm_k      = atm_k
        st.session_state.opt_rfr        = st.session_state.get("rfr_sidebar", 6.5) / 100.0
        st.session_state.opt_hv10       = hv10 or 0.20
        st.session_state.opt_ohlcv      = ohlcv_df
        st.session_state.opt_loaded     = True
        st.session_state.payoff_legs    = []  # reset payoff on new load

# ============================================================
# MAIN RENDER (only if data is loaded)
# ============================================================

if not st.session_state.opt_loaded:
    st.markdown("""
<div style="background:#0d0d0d;border:1px solid #2a2a2a;padding:32px 24px;
font-family:'IBM Plex Mono',monospace;text-align:center;margin-top:20px;">
  <div style="color:#ff8c00;font-size:.9rem;font-weight:700;letter-spacing:.15em;">
    ⚡ MONARCH OPTIONS INTELLIGENCE
  </div>
  <div style="color:#444;font-size:.65rem;margin:12px 0 20px;">
    ──────────────────────────────────────────────────────────
  </div>
  <div style="color:#888;font-size:.7rem;line-height:2.2;">
    1. Select <span style="color:#ff8c00;">Underlying</span> from the sidebar<br/>
    2. Choose <span style="color:#ff8c00;">Expiry Date</span> (auto-loaded from Upstox)<br/>
    3. Click <span style="color:#ff8c00;">⚡ LOAD OPTIONS INTEL</span><br/>
  </div>
  <div style="color:#444;font-size:.62rem;margin-top:20px;line-height:1.8;">
    The engine will compute:<br/>
    <span style="color:#e8e8e8;">
    Directional Bias · IV Regime · Strategy Recommendations<br/>
    Live Option Chain with Signals · Greeks Dashboard · OI Analysis · Payoff Builder
    </span>
  </div>
</div>
""", unsafe_allow_html=True)
    st.stop()

# Pull from session state
spot     = st.session_state.opt_spot
chain_df = st.session_state.opt_chain_data
bias_res = st.session_state.opt_bias
oi_d     = st.session_state.opt_oi
atm_iv   = st.session_state.opt_atm_iv
hv20     = st.session_state.opt_hv20
hv10     = st.session_state.get("opt_hv10", hv20)
sym      = st.session_state.opt_symbol
expiry   = st.session_state.opt_expiry
dte      = st.session_state.opt_dte
step     = st.session_state.opt_step
atm_k    = st.session_state.get("opt_atm_k", atm_strike(spot, step))
r        = st.session_state.get("opt_rfr", 0.065)
T        = dte / 365.0
ohlcv_df = st.session_state.get("opt_ohlcv", pd.DataFrame())

bias       = bias_res.get("bias", "NEUTRAL")
bias_score = int(round(bias_res.get("score", 0)))  # always int — tanh scoring returns floats

# IV Rank from real rolling history (accumulated across page refreshes for this symbol)
# Falls back to HV-relative estimate on first load (before enough history is stored)
_iv_hist_sym = st.session_state.opt_iv_history.get(sym, [])
if len(_iv_hist_sym) >= 3:
    ivr = iv_rank(_iv_hist_sym, atm_iv)
else:
    # Bootstrap estimate: IV Rank ~50 when IV = HV (fair value), rises above 50 when IV > HV.
    # Formula: IVR = 50 + 50 × tanh((IV - HV) / (0.5 × HV))
    # At IV=HV: IVR=50. At IV=1.5×HV: IVR≈81. At IV=0.5×HV: IVR≈19.
    _hv_ref = hv20 if hv20 and hv20 > 0 else 0.15
    ivr = float(min(100.0, max(0.0, 50.0 + 50.0 * math.tanh((atm_iv - _hv_ref) / (0.5 * _hv_ref)))))
v_lbl, v_act, v_col = vol_regime(ivr)
strat_recs   = recommend_strategies(bias, v_lbl, dte, spot, atm_k, step, ivr, bias_score)

BIAS_COLORS = {
    "STRONGLY BULLISH":"#00d084","BULLISH":"#7dca84","NEUTRAL":"#ffb347",
    "BEARISH":"#ff7777","STRONGLY BEARISH":"#ff3b3b"
}
bc = BIAS_COLORS.get(bias, "#888")

# ── TOP HEADER BAR ──
iv_vs_hv = (atm_iv - hv20)*100
iv_sign  = "+" if iv_vs_hv >= 0 else ""

# IV momentum: change vs rolling mean of session IV history
_iv_hist_sym2 = st.session_state.opt_iv_history.get(sym, [])
if len(_iv_hist_sym2) >= 5:
    _iv_ma5 = float(np.mean(_iv_hist_sym2[-5:]))
    iv_momentum = (atm_iv - _iv_ma5) * 100
    iv_mom_sign = "+" if iv_momentum >= 0 else ""
    iv_mom_str  = f"IV Δ(5): {iv_mom_sign}{iv_momentum:.1f}%"
    iv_mom_c    = "#ff3b3b" if iv_momentum > 0.5 else ("#1e90ff" if iv_momentum < -0.5 else "#888")
else:
    iv_mom_str = "IV Δ: —"
    iv_mom_c   = "#555"

st.markdown(f"""
<div style="background:#0d0d0d;border:1px solid #2a2a2a;border-left:4px solid {bc};
padding:10px 16px;margin-bottom:6px;font-family:'IBM Plex Mono',monospace;">
  <div style="display:flex;gap:28px;align-items:center;flex-wrap:wrap;">
    <span style="color:#ff8c00;font-size:.65rem;font-weight:700;letter-spacing:.15em;">⚡ {sym}</span>
    <span style="font-size:1.15rem;color:#e8e8e8;font-weight:700;">₹{spot:,.2f}</span>
    <span style="color:#888;font-size:.65rem;">ATM {atm_k} · Step {step} · {expiry}</span>
    <span style="color:{bc};font-size:.82rem;font-weight:700;">{bias} ({bias_score:+d})</span>
    <span style="color:{v_col};font-size:.72rem;font-weight:600;">VOL: {v_lbl}</span>
    <span style="color:#888;font-size:.62rem;">ATM IV {atm_iv*100:.1f}% · HV20 {hv20*100:.1f}% · IV−HV {iv_sign}{iv_vs_hv:.1f}%</span>
    <span style="color:{iv_mom_c};font-size:.6rem;">{iv_mom_str}</span>
    <span style="color:#555;font-size:.6rem;">DTE: {dte}</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── TABS ──
t_ov, t_dir, t_strat, t_chain, t_greeks, t_oi, t_payoff = st.tabs([
    "📊 Overview",
    "🧭 Direction",
    "🎯 Strategies",
    "📋 Chain",
    "🔢 Greeks",
    "📌 OI Analysis",
    "💹 Payoff",
])

# ══════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════
with t_ov:
    st.markdown("### ◼ Options Intelligence Summary")

    k1,k2,k3,k4,k5,k6,k7 = st.columns(7)
    k1.metric("Spot",      f"₹{spot:,.1f}")
    k2.metric("ATM",       str(atm_k))
    k3.metric("DTE",       str(dte))
    k4.metric("ATM IV",    f"{atm_iv*100:.1f}%")
    k5.metric("HV(20)",    f"{hv20*100:.1f}%")
    k6.metric("IV−HV",     f"{iv_vs_hv:+.1f}%")
    k7.metric("Bias",      bias, delta=f"{bias_score:+d} pts")

    st.divider()

    ov1, ov2 = st.columns(2)
    with ov1:
        # Vol regime box
        st.markdown(f"""
<div style="background:#0d0d0d;border:1px solid {v_col};padding:12px 16px;
font-family:'IBM Plex Mono',monospace;height:100%;">
  <div style="color:{v_col};font-size:.7rem;font-weight:700;letter-spacing:.1em;margin-bottom:6px;">
    VOL REGIME: {v_lbl}
  </div>
  <div style="color:#e8e8e8;font-size:.68rem;line-height:1.7;">{v_act}</div>
  <div style="margin-top:8px;color:#555;font-size:.6rem;">
    ATM Straddle: ₹{oi_d.get('atm_straddle',0):.1f} &nbsp;·&nbsp;
    Exp Move ±1σ: ±{oi_d.get('exp_move_pct',0):.1f}% &nbsp;·&nbsp;
    ±2σ: ±{oi_d.get('exp_move_2sd_pct', oi_d.get('exp_move_pct',0)*2):.1f}% &nbsp;·&nbsp;
    IV Rank: {ivr:.0f}
  </div>
</div>""", unsafe_allow_html=True)

    with ov2:
        # OI key metrics
        if oi_d:
            # PCR colour: green if aggregate put OI > call OI (bullish support), else red
            pcr_c = "#00d084" if oi_d.get("total_pe_oi", 0) >= oi_d.get("total_ce_oi", 0) else "#ff3b3b"
            _gex_net = oi_d.get("net_gex", 0) or 0
            _gex_c   = "#00d084" if _gex_net >= 0 else "#ff3b3b"
            _gex_lbl = "POS GEX" if _gex_net >= 0 else "NEG GEX"
            _skew_pp = oi_d.get("skew_pp")
            _skew_str = f"{_skew_pp:+.1f}pp" if _skew_pp is not None else "—"
            _skew_c  = "#ff3b3b" if (_skew_pp or 0) > 2 else ("#1e90ff" if (_skew_pp or 0) < -1 else "#888")
            st.markdown(f"""
<div style="background:#0d0d0d;border:1px solid #2a2a2a;padding:12px 16px;
font-family:'IBM Plex Mono',monospace;">
  <div style="color:#ff8c00;font-size:.65rem;font-weight:700;letter-spacing:.1em;margin-bottom:8px;">OI SNAPSHOT</div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;">
    <div><div style="color:#555;font-size:.58rem;">MAX PAIN</div>
         <div style="color:#ff8c00;font-size:.9rem;font-weight:700;">₹{oi_d.get('max_pain',0):,.0f}</div></div>
    <div><div style="color:#555;font-size:.58rem;">PCR (OI)</div>
         <div style="color:{pcr_c};font-size:.9rem;font-weight:700;">{oi_d.get('pcr_oi',0):.3f}</div></div>
    <div><div style="color:#555;font-size:.58rem;">CALL WALL</div>
         <div style="color:#ff3b3b;font-size:.82rem;font-weight:600;">₹{oi_d.get('call_wall',0):,.0f}</div></div>
    <div><div style="color:#555;font-size:.58rem;">PUT WALL</div>
         <div style="color:#00d084;font-size:.82rem;font-weight:600;">₹{oi_d.get('put_wall',0):,.0f}</div></div>
    <div><div style="color:#555;font-size:.58rem;">GEX REGIME</div>
         <div style="color:{_gex_c};font-size:.78rem;font-weight:600;">{_gex_lbl} ({_gex_net:+,.0f})</div></div>
    <div><div style="color:#555;font-size:.58rem;">IV SKEW (±1 strike)</div>
         <div style="color:{_skew_c};font-size:.78rem;font-weight:600;">{_skew_str}</div></div>
  </div>
  <div style="margin-top:8px;color:{pcr_c};font-size:.62rem;">{oi_d.get('pcr_signal','—')}</div>
  <div style="margin-top:4px;color:#555;font-size:.58rem;">Gamma Flip: ₹{oi_d.get('gamma_flip',spot):,.0f}</div>
</div>""", unsafe_allow_html=True)

    # Best strategy card
    if strat_recs:
        best = strat_recs[0]
        st.divider()
        st.markdown(f"""
<div style="background:#0d1a00;border:1px solid #ff8c00;border-top:3px solid #ff8c00;
padding:14px 18px;font-family:'IBM Plex Mono',monospace;">
  <div style="color:#ff8c00;font-size:.6rem;letter-spacing:.12em;font-weight:700;">⭐ TOP RECOMMENDED STRATEGY</div>
  <div style="color:#e8e8e8;font-size:.95rem;font-weight:700;margin:8px 0 4px;">{best['Strategy']}</div>
  <div style="color:#7ec8e3;font-size:.72rem;margin-bottom:6px;">LEGS: {best['Legs']}</div>
  <div style="color:#aaa;font-size:.64rem;line-height:1.6;">{best['Rationale']}</div>
  <div style="display:flex;gap:24px;margin-top:10px;flex-wrap:wrap;">
    <span style="color:#ff3b3b;font-size:.62rem;">⬇ Risk: {best['Max Risk']}</span>
    <span style="color:#00d084;font-size:.62rem;">⬆ Reward: {best['Max Reward']}</span>
    <span style="color:#888;font-size:.62rem;">TYPE: {best['Type']}</span>
    <span style="color:#666;font-size:.62rem;">DTE: {best['Ideal DTE']}</span>
  </div>
</div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# TAB 2 — DIRECTIONAL ANALYSIS
# ══════════════════════════════════════════════════════════════
with t_dir:
    st.markdown("### 🧭 Directional Signal Stack")

    s_norm  = max(-80, min(80, bias_score))
    gauge_w = int((s_norm + 80) / 160 * 100)
    gc_     = "#00d084" if bias_score>0 else "#ff3b3b" if bias_score<0 else "#ffb347"

    st.markdown(f"""
<div style="background:#0d0d0d;border:1px solid #2a2a2a;padding:10px 16px;
font-family:'IBM Plex Mono',monospace;margin-bottom:10px;">
  <div style="color:#555;font-size:.58rem;letter-spacing:.1em;margin-bottom:4px;">
    BEARISH ◄──────────────────── 0 ────────────────────► BULLISH
  </div>
  <div style="background:#1a1a1a;height:14px;position:relative;overflow:hidden;">
    <div style="position:absolute;left:50%;top:0;bottom:0;width:2px;background:#333;"></div>
    <div style="width:{gauge_w}%;height:100%;background:{gc_};opacity:.85;"></div>
  </div>
  <div style="color:{gc_};font-size:1rem;font-weight:700;margin-top:6px;">
    {bias} &nbsp;·&nbsp; Score: {bias_score:+d}
  </div>
</div>""", unsafe_allow_html=True)

    # Factor table
    if bias_res.get("factors"):
        fdf = pd.DataFrame([
            {"Factor": k, "Points": v,
             "Signal": "🟢 BULL" if v>0 else "🔴 BEAR" if v<0 else "⚪ NEUT"}
            for k, v in bias_res["factors"].items()
        ])
        def pts_style(v):
            if isinstance(v,(int,float)):
                if v > 0: return "color:#00d084;font-weight:700"
                if v < 0: return "color:#ff3b3b;font-weight:700"
            return ""
        st.dataframe(fdf.style.applymap(pts_style, subset=["Points"]),
                     use_container_width=True, hide_index=True)

    st.divider()
    ic1,ic2,ic3,ic4,ic5 = st.columns(5)
    ic1.metric("RSI(14)",    f"{bias_res.get('rsi',50):.1f}")
    ic2.metric("MACD Hist",  f"{bias_res.get('macd_hist',0):+.3f}")
    ic3.metric("BB %",       f"{bias_res.get('bb_pct',50):.0f}%")
    ic4.metric("Vol Ratio",  f"{bias_res.get('vol_ratio',1):.2f}×")
    ic5.metric("ATR%",       f"{bias_res.get('atr_pct',0):.2f}%")

    em1,em2,em3 = st.columns(3)
    em1.metric("EMA 9",  f"₹{bias_res.get('e9',spot):.2f}",
               delta="Above" if spot>bias_res.get('e9',spot) else "Below")
    em2.metric("EMA 20", f"₹{bias_res.get('e20',spot):.2f}",
               delta="Above" if spot>bias_res.get('e20',spot) else "Below")
    em3.metric("EMA 50", f"₹{bias_res.get('e50',spot):.2f}",
               delta="Above" if spot>bias_res.get('e50',spot) else "Below")

    impl = ("CALL BUYER or PUT SELLER — use debit spreads in low IV, credit spreads in high IV" if "BULL" in bias else
            "PUT BUYER or CALL SELLER — use debit spreads in low IV, credit spreads in high IV" if "BEAR" in bias else
            "NON-DIRECTIONAL preferred — iron condor / straddle / strangle based on vol regime")
    impl_c = bc
    st.markdown(f"""
<div style="border-left:3px solid {impl_c};padding:8px 12px;
font-family:'IBM Plex Mono',monospace;font-size:.7rem;color:{impl_c};margin-top:8px;">
  OPTIONS IMPLICATION: {impl}
</div>""", unsafe_allow_html=True)

    # Chart
    if not ohlcv_df.empty and "close" in ohlcv_df.columns:
        st.divider()
        disp = ohlcv_df.tail(80).copy()
        if "time" not in disp.columns: disp = disp.reset_index()
        for sp_, col_ in [(9,"#ff8c00"),(20,"#1e90ff"),(50,"#9c27b0")]:
            disp[f"e{sp_}"] = disp["close"].astype(float).ewm(span=sp_,adjust=False).mean()
        x_col = "time" if "time" in disp.columns else (disp.index.name or "index")
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=disp[x_col] if x_col in disp.columns else disp.index,
            open=disp["open"],high=disp["high"],low=disp["low"],close=disp["close"],
            name="Price",increasing_line_color="#00d084",decreasing_line_color="#ff3b3b"
        ))
        for sp_, col_ in [(9,"#ff8c00"),(20,"#1e90ff"),(50,"#9c27b0")]:
            fig.add_trace(go.Scatter(
                x=disp[x_col] if x_col in disp.columns else disp.index,
                y=disp[f"e{sp_}"], name=f"EMA{sp_}", line=dict(color=col_,width=1.5)
            ))
        fig.add_hline(y=spot,  line=dict(color="#ffb347",dash="dot",width=1), annotation_text="Spot")
        fig.add_hline(y=atm_k, line=dict(color="#333",   dash="dash",width=1), annotation_text=f"ATM {atm_k}")
        if oi_d:
            fig.add_hline(y=oi_d.get("call_wall",0),line=dict(color="#ff3b3b",dash="dot",width=1),annotation_text="Call Wall")
            fig.add_hline(y=oi_d.get("put_wall",0), line=dict(color="#00d084",dash="dot",width=1),annotation_text="Put Wall")
        fig.update_layout(
            height=380,plot_bgcolor="#000",paper_bgcolor="#000",
            font=dict(color="#e8e8e8",family="IBM Plex Mono",size=9),
            xaxis_rangeslider_visible=False,
            legend=dict(orientation="h",y=1.1,font=dict(size=8)),
            margin=dict(t=30,b=10),yaxis=dict(gridcolor="#111")
        )
        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════
# TAB 3 — STRATEGIES
# ══════════════════════════════════════════════════════════════
with t_strat:
    st.markdown("### 🎯 Strategy Recommendations")
    st.markdown(f"""
<div style="background:#0a0a0a;border:1px solid #2a2a2a;padding:7px 14px;
font-family:'IBM Plex Mono',monospace;font-size:.65rem;color:#666;margin-bottom:10px;">
  Bias: <span style="color:{bc};">{bias}</span> &nbsp;·&nbsp;
  Vol: <span style="color:{v_col};">{v_lbl}</span> &nbsp;·&nbsp;
  IV Rank: {ivr:.0f} &nbsp;·&nbsp;
  DTE: {dte} &nbsp;·&nbsp;
  ATM: {atm_k} &nbsp;·&nbsp;
  Step: {step}
</div>""", unsafe_allow_html=True)

    for i, s in enumerate(strat_recs[:8]):
        rank_c  = "#ff8c00" if i==0 else "#444"
        score_c = "#00d084" if s["Score"]>=90 else "#ffb347" if s["Score"]>=75 else "#666"
        top_border = "border-top:3px solid #ff8c00;" if i==0 else ""
        st.markdown(f"""
<div style="background:#0d0d0d;border:1px solid #2a2a2a;{top_border}
padding:12px 16px;margin-bottom:7px;font-family:'IBM Plex Mono',monospace;">
  <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;">
    <div>
      <span style="color:{rank_c};font-size:.58rem;font-weight:700;">{'⭐ BEST FIT' if i==0 else f'#{i+1}'}</span>
      <span style="color:#e8e8e8;font-size:.9rem;font-weight:700;margin-left:8px;">{s['Strategy']}</span>
      <span style="background:#1a1a1a;color:#777;font-size:.56rem;padding:2px 7px;margin-left:8px;
                   display:inline-block;">{s['Type']}</span>
    </div>
    <span style="color:{score_c};font-size:.72rem;font-weight:700;">FIT SCORE {s['Score']}</span>
  </div>
  <div style="color:#7ec8e3;font-size:.7rem;margin:6px 0 4px;">
    LEGS: <b>{s['Legs']}</b>
  </div>
  <div style="color:#999;font-size:.64rem;line-height:1.55;">{s['Rationale']}</div>
  <div style="display:flex;gap:18px;margin-top:8px;flex-wrap:wrap;">
    <span style="color:#ff3b3b;font-size:.6rem;">⬇ Max Risk: {s['Max Risk']}</span>
    <span style="color:#00d084;font-size:.6rem;">⬆ Max Reward: {s['Max Reward']}</span>
    <span style="color:#ffb347;font-size:.6rem;">⏱ Ideal DTE: {s['Ideal DTE']}</span>
  </div>
</div>""", unsafe_allow_html=True)

    with st.expander("◼ STRATEGY SELECTION LOGIC"):
        st.markdown(f"""
**How the engine picks strategies:**

The engine cross-references **Directional Bias × Volatility Regime × DTE**:

| IV Rank | What to do | Best Structures |
|---------|-----------|-----------------|
| < 35 (Low IV) | **BUY vol** — options are cheap | Long calls/puts, straddles, strangles, calendars |
| 35–60 (Normal) | Use **spreads** both ways | Bull/bear spreads, condors, butterflies |
| > 60 (High IV) | **SELL vol** — premium is fat | Iron condors, short strangles, credit spreads |

**Current inputs:**
- Bias: **{bias}** (score {bias_score:+d})
- Vol Regime: **{v_lbl}** (IV Rank ~{ivr:.0f})
- DTE: **{dte}**
- Action: *{v_act}*

**Key rule:** *Never buy expensive options (high IV) and never sell cheap options (low IV).*
""")

# ══════════════════════════════════════════════════════════════
# TAB 4 — OPTION CHAIN
# ══════════════════════════════════════════════════════════════
with t_chain:
    st.markdown("### 📋 Option Chain with Signals")

    if chain_df.empty:
        st.warning("No live chain data from Upstox. Showing Black-Scholes synthetic chain.")
        syn_rows = []
        for k in strikes_around(spot, step, 6):
            ce_p = bs_price(spot, k, T, r, atm_iv, "call")
            pe_p = bs_price(spot, k, T, r, atm_iv, "put")
            cg   = bs_greeks(spot, k, T, r, atm_iv, "call")
            pg   = bs_greeks(spot, k, T, r, atm_iv, "put")
            mm   = "ATM" if abs(k-spot) <= 0.5*step else ("ITM-C" if k<spot else "OTM-C")
            syn_rows.append({
                "Strike":k,"Moneyness":mm,
                "CE Price":round(ce_p,2),"CE IV%":round(atm_iv*100,1),
                "CE Δ":cg["delta"],"CE θ":round(cg["theta"],3),"CE ν":round(cg["vega"],3),
                "PE Price":round(pe_p,2),"PE IV%":round(atm_iv*100,1),
                "PE Δ":pg["delta"],"PE θ":round(pg["theta"],3),"PE ν":round(pg["vega"],3),
            })
        st.dataframe(pd.DataFrame(syn_rows), use_container_width=True, hide_index=True)
    else:
        # Show ±8 strikes around ATM (not an arbitrary % band)
        _chain_lo = atm_k - 8 * step
        _chain_hi = atm_k + 8 * step
        disp_c = chain_df[(chain_df.Strike >= _chain_lo) & (chain_df.Strike <= _chain_hi)].copy()

        # Add directional + IV edge signal per row
        def row_signal(row):
            ce_iv_r = float(row.CE_IV); pe_iv_r = float(row.PE_IV)
            ce_iv   = ce_iv_r/100 if ce_iv_r>2 else (ce_iv_r or atm_iv)
            pe_iv   = pe_iv_r/100 if pe_iv_r>2 else (pe_iv_r or atm_iv)
            # IV edge: relative to HV — scale-invariant
            # >1.20× HV = 20% overpriced → sell signal
            # <0.85× HV = 15% underpriced → buy signal
            hv_ref  = hv20 if hv20 and hv20 > 0 else atm_iv
            ce_ratio = ce_iv / hv_ref if hv_ref > 0 else 1.0
            pe_ratio = pe_iv / hv_ref if hv_ref > 0 else 1.0
            ce_dir  = "BUY" if bias_score >= 12 else "SELL" if bias_score <= -12 else "—"
            pe_dir  = "BUY" if bias_score <= -12 else "SELL" if bias_score >= 12 else "—"
            ce_vol  = "SELL (rich)" if ce_ratio > 1.20 else "BUY (cheap)" if ce_ratio < 0.85 else "—"
            pe_vol  = "SELL (rich)" if pe_ratio > 1.20 else "BUY (cheap)" if pe_ratio < 0.85 else "—"
            return pd.Series({"CE_Dir":ce_dir,"CE_Vol_Sig":ce_vol,"PE_Dir":pe_dir,"PE_Vol_Sig":pe_vol})

        sigs = disp_c.apply(row_signal, axis=1)
        disp_c = pd.concat([disp_c, sigs], axis=1)

        chain_show = disp_c[[
            "Strike","Moneyness",
            "CE_LTP","CE_IV","CE_OI","CE_Vol","CE_OIC","CE_Dir","CE_Vol_Sig",
            "PCR",
            "PE_Dir","PE_Vol_Sig","PE_OIC","PE_Vol","PE_OI","PE_IV","PE_LTP"
        ]].copy()
        chain_show.columns = [
            "Strike","Money",
            "CE LTP","CE IV","CE OI","CE Vol","CE ΔOI","CE Signal","CE IV Sig",
            "PCR",
            "PE Signal","PE IV Sig","PE ΔOI","PE Vol","PE OI","PE IV","PE LTP"
        ]

        def sig_style(v):
            if v == "BUY":        return "background-color:#1a3300;color:#00d084;font-weight:700"
            if v == "SELL":       return "background-color:#2a0000;color:#ff3b3b;font-weight:700"
            if "rich" in str(v):  return "color:#ff8c00"
            if "cheap" in str(v): return "color:#1e90ff"
            return "color:#555"
        def mm_style(v):
            if v == "ATM":    return "background-color:#1a1400;color:#ff8c00;font-weight:700"
            if "ITM" in str(v): return "color:#7ec8e3"
            return "color:#555"

        styled = chain_show.style \
            .applymap(sig_style,  subset=["CE Signal","CE IV Sig","PE Signal","PE IV Sig"]) \
            .applymap(mm_style,   subset=["Money"])
        st.dataframe(styled, use_container_width=True, hide_index=True)
        st.caption("CE/PE Signal = directional signal from bias. IV Sig = IV vs HV edge (overpriced/underpriced).")

# ══════════════════════════════════════════════════════════════
# TAB 5 — GREEKS DASHBOARD
# ══════════════════════════════════════════════════════════════
with t_greeks:
    st.markdown("### 🔢 Greeks Dashboard")

    g_rows = []
    for k in strikes_around(spot, step, 5):
        # Use API IV if available
        ce_iv_use = pe_iv_use = atm_iv
        if not chain_df.empty:
            closest = chain_df.iloc[(chain_df.Strike - k).abs().argsort()[:1]]
            if not closest.empty:
                ce_r = float(closest.CE_IV.values[0])
                pe_r = float(closest.PE_IV.values[0])
                ce_iv_use = (ce_r/100 if ce_r>2 else ce_r) or atm_iv
                pe_iv_use = (pe_r/100 if pe_r>2 else pe_r) or atm_iv

        cg  = bs_greeks(spot, k, T, r, ce_iv_use, "call")
        pg  = bs_greeks(spot, k, T, r, pe_iv_use, "put")
        cp  = bs_price (spot, k, T, r, ce_iv_use, "call")
        pp  = bs_price (spot, k, T, r, pe_iv_use, "put")
        mm  = "ATM" if abs(k-spot) <= 0.5*step else ("ITM" if k<spot else "OTM")
        ce_itm = bs_itm_prob(spot, k, T, r, ce_iv_use, "call")
        pe_itm = bs_itm_prob(spot, k, T, r, pe_iv_use, "put")
        g_rows.append({
            "Strike":k, "Moneyness":mm,
            "CE Price":round(cp,2), "CE IV%":round(ce_iv_use*100,1),
            "CE Δ":cg["delta"], "CE Γ":cg["gamma"],
            "CE θ/d":round(cg["theta"],3), "CE ν/1%":round(cg["vega"],3),
            "CE P(ITM)":f"{ce_itm*100:.0f}%",
            "PE Price":round(pp,2), "PE IV%":round(pe_iv_use*100,1),
            "PE Δ":pg["delta"], "PE Γ":pg["gamma"],
            "PE θ/d":round(pg["theta"],3), "PE ν/1%":round(pg["vega"],3),
            "PE P(ITM)":f"{pe_itm*100:.0f}%",
        })

    g_df = pd.DataFrame(g_rows)
    st.dataframe(g_df, use_container_width=True, hide_index=True)

    # Gamma chart
    fig_g = go.Figure()
    fig_g.add_trace(go.Bar(x=g_df.Strike, y=g_df["CE Γ"], name="CE Gamma",
                           marker_color="#1e90ff", opacity=0.85))
    fig_g.add_trace(go.Bar(x=g_df.Strike, y=g_df["PE Γ"], name="PE Gamma",
                           marker_color="#ff8c00", opacity=0.85))
    fig_g.add_vline(x=spot, line=dict(color="#00d084",dash="dot",width=1.5),
                    annotation_text=f"Spot {spot:.0f}")
    fig_g.update_layout(
        title="Gamma by Strike — higher = delta changes faster (most sensitive near expiry at ATM)",
        height=260, barmode="group", plot_bgcolor="#000", paper_bgcolor="#000",
        font=dict(color="#e8e8e8",family="IBM Plex Mono",size=9),
        legend=dict(orientation="h",y=1.12), margin=dict(t=40,b=10)
    )
    st.plotly_chart(fig_g, use_container_width=True)

    st.divider()
    st.markdown("##### Greeks Quick Reference")
    gc1,gc2,gc3,gc4 = st.columns(4)
    with gc1:
        st.markdown("""**Δ Delta**
- Call: 0→1 | Put: −1→0
- ATM ≈ ±0.50
- ₹ move per ₹1 underlying move
- Deep ITM → acts like stock""")
    with gc2:
        st.markdown("""**Γ Gamma**
- Rate of delta change
- Peaks ATM near expiry
- Long options: +gamma
- Near-expiry spike = danger""")
    with gc3:
        st.markdown("""**θ Theta**
- Time decay per day (₹)
- ATM decays fastest
- Accelerates near expiry
- Sellers want high theta""")
    with gc4:
        st.markdown("""**ν Vega**
- P&L per 1% IV move
- Buy in low IV, sell in high IV
- Higher for longer DTE
- Key edge identification tool""")

# ══════════════════════════════════════════════════════════════
# TAB 6 — OI ANALYSIS
# ══════════════════════════════════════════════════════════════
with t_oi:
    st.markdown("### 📌 Open Interest Analysis")

    if not oi_d:
        st.info("No OI data available. Load chain first.")
    else:
        pcr_c = "#00d084" if oi_d.get("total_pe_oi", 0) >= oi_d.get("total_ce_oi", 0) else "#ff3b3b"
        pain_diff = spot - oi_d.get("max_pain", spot)

        o1,o2,o3,o4,o5,o6 = st.columns(6)
        o1.metric("Max Pain",       f"₹{oi_d['max_pain']:,.0f}",
                  delta=f"{pain_diff:+.0f} from spot")
        o2.metric("PCR (OI)",       f"{oi_d['pcr_oi']:.3f}")
        o3.metric("Call Wall",      f"₹{oi_d['call_wall']:,.0f}")
        o4.metric("Put Wall",       f"₹{oi_d['put_wall']:,.0f}")
        o5.metric("ATM Straddle",   f"₹{oi_d['atm_straddle']:.1f}")
        o6.metric("Exp Move ±1σ",   f"{oi_d['exp_move_pct']:.1f}%",
                  delta=f"±2σ = {oi_d.get('exp_move_2sd_pct', oi_d['exp_move_pct']*2):.1f}%")

        st.markdown(f"""
<div style="border-left:3px solid {pcr_c};padding:7px 12px;margin:8px 0;
font-family:'IBM Plex Mono',monospace;font-size:.7rem;color:{pcr_c};">
  PCR SIGNAL: {oi_d['pcr_signal']}
</div>""", unsafe_allow_html=True)

    if not chain_df.empty:
        oi_disp = chain_df[(chain_df.Strike >= atm_k - 10*step) & (chain_df.Strike <= atm_k + 10*step)].copy()

        # OI Bar Chart
        fig_oi = go.Figure()
        fig_oi.add_trace(go.Bar(x=oi_disp.Strike, y=oi_disp.CE_OI/1e5,
                                name="CE OI (lac)", marker_color="#ff3b3b", opacity=0.8))
        fig_oi.add_trace(go.Bar(x=oi_disp.Strike, y=oi_disp.PE_OI/1e5,
                                name="PE OI (lac)", marker_color="#00d084", opacity=0.8))
        fig_oi.add_vline(x=spot, line=dict(color="#ffb347",dash="dot",width=1.5),
                         annotation_text=f"Spot {spot:.0f}")
        if oi_d:
            fig_oi.add_vline(x=oi_d["max_pain"], line=dict(color="#ff8c00",dash="dash",width=1),
                             annotation_text=f"MaxPain {oi_d['max_pain']:.0f}")
        fig_oi.update_layout(
            title="OI Distribution — Call (red) vs Put (green)",
            height=340, barmode="group", plot_bgcolor="#000", paper_bgcolor="#000",
            font=dict(color="#e8e8e8",family="IBM Plex Mono",size=9),
            legend=dict(orientation="h",y=1.12), margin=dict(t=40,b=10),
            yaxis=dict(title="OI (lac)",gridcolor="#111")
        )
        st.plotly_chart(fig_oi, use_container_width=True)

        # PCR bar colours derived from the chain's own PCR distribution (25th/75th pct)
        _pcr_vals = oi_disp.PCR.replace([np.inf, -np.inf], np.nan).dropna()
        _pcr_hi   = float(_pcr_vals.quantile(0.75)) if len(_pcr_vals) >= 4 else 1.2
        _pcr_lo   = float(_pcr_vals.quantile(0.25)) if len(_pcr_vals) >= 4 else 0.8
        fig_pcr = go.Figure()
        fig_pcr.add_trace(go.Bar(
            x=oi_disp.Strike, y=oi_disp.PCR,
            marker_color=["#00d084" if v > _pcr_hi else "#ff3b3b" if v < _pcr_lo else "#ffb347"
                          for v in oi_disp.PCR],
            name="PCR"
        ))
        fig_pcr.add_hline(y=float(_pcr_vals.median()) if len(_pcr_vals) > 0 else 1.0,
                          line=dict(color="#555",dash="dot"),
                          annotation_text=f"Median {_pcr_vals.median():.2f}" if len(_pcr_vals) > 0 else "")
        fig_pcr.add_vline(x=spot, line=dict(color="#ffb347",dash="dot",width=1))
        fig_pcr.update_layout(
            title=f"PCR by Strike  (green > {_pcr_hi:.2f} = 75th pct / red < {_pcr_lo:.2f} = 25th pct)",
            height=220, plot_bgcolor="#000", paper_bgcolor="#000",
            font=dict(color="#e8e8e8",family="IBM Plex Mono",size=9),
            margin=dict(t=40,b=10)
        )
        st.plotly_chart(fig_pcr, use_container_width=True)

        # OI Change (who is building positions)
        if "CE_OIC" in oi_disp.columns and oi_disp.CE_OIC.abs().sum() > 0:
            fig_oic = go.Figure()
            fig_oic.add_trace(go.Bar(x=oi_disp.Strike, y=oi_disp.CE_OIC/1e5,
                                     name="CE OI Change", marker_color="#ff6b6b", opacity=0.8))
            fig_oic.add_trace(go.Bar(x=oi_disp.Strike, y=oi_disp.PE_OIC/1e5,
                                     name="PE OI Change", marker_color="#6bff9e", opacity=0.8))
            fig_oic.add_vline(x=spot, line=dict(color="#ffb347",dash="dot",width=1))
            fig_oic.update_layout(
                title="OI Change Today — New positions being built",
                height=220, barmode="group", plot_bgcolor="#000", paper_bgcolor="#000",
                font=dict(color="#e8e8e8",family="IBM Plex Mono",size=9),
                legend=dict(orientation="h",y=1.12), margin=dict(t=40,b=10)
            )
            st.plotly_chart(fig_oic, use_container_width=True)

            # ΔOI interpretation table
            with st.expander("◼ ΔOI POSITIONING GUIDE"):
                st.markdown("""
| Price | OI | Meaning | Implication |
|-------|-----|---------|-------------|
| ↑ | ↑ | Long build-up | Bullish — fresh longs entering |
| ↓ | ↑ | Short build-up | Bearish — fresh shorts entering |
| ↑ | ↓ | Short covering | Bullish — trapped shorts exiting |
| ↓ | ↓ | Long liquidation | Bearish — longs exiting |

**Read CE_ΔOI and PE_ΔOI independently at each strike.**
- Big CE_ΔOI build at a strike above spot = call resistance being written
- Big PE_ΔOI build at a strike below spot = put support being written
""")

        # ── Gamma Exposure (GEX) Chart ──
        if oi_d and "gex_df" in oi_d and not oi_d["gex_df"].empty:
            gex_plot = oi_d["gex_df"]
            gex_plot = gex_plot[(gex_plot.Strike >= atm_k - CFG["oi_strikes"]*step) &
                                (gex_plot.Strike <= atm_k + CFG["oi_strikes"]*step)]
            gex_colors = ["#00d084" if v >= 0 else "#ff3b3b" for v in gex_plot.NET_GEX]
            fig_gex = go.Figure()
            fig_gex.add_trace(go.Bar(
                x=gex_plot.Strike, y=gex_plot.NET_GEX,
                marker_color=gex_colors, name="Net GEX", opacity=0.85
            ))
            fig_gex.add_vline(x=spot, line=dict(color="#ffb347",dash="dot",width=1.5),
                              annotation_text=f"Spot {spot:.0f}")
            if oi_d.get("gamma_flip"):
                fig_gex.add_vline(x=oi_d["gamma_flip"],
                                  line=dict(color="#ff8c00",dash="dash",width=1.5),
                                  annotation_text=f"Gamma Flip {oi_d['gamma_flip']:.0f}",
                                  annotation=dict(font=dict(color="#ff8c00",size=8)))
            gex_net   = oi_d.get("net_gex", 0)
            gex_regime_short = "POSITIVE" if gex_net >= 0 else "NEGATIVE"
            gex_rc    = "#00d084" if gex_net >= 0 else "#ff3b3b"
            fig_gex.update_layout(
                title=f"Gamma Exposure (GEX) by Strike — Net: {gex_net:+.0f} [{gex_regime_short}]",
                height=280, plot_bgcolor="#000", paper_bgcolor="#000",
                font=dict(color="#e8e8e8",family="IBM Plex Mono",size=9),
                margin=dict(t=40,b=10), yaxis=dict(gridcolor="#111")
            )
            st.plotly_chart(fig_gex, use_container_width=True)

            # GEX regime card
            gex_flip_rel = "ABOVE spot" if oi_d["gamma_flip"] > spot else "BELOW spot"
            st.markdown(f"""
<div style="background:#0d0d0d;border:1px solid #2a2a2a;border-left:3px solid {gex_rc};
padding:9px 14px;font-family:'IBM Plex Mono',monospace;font-size:.65rem;">
  <span style="color:{gex_rc};font-weight:700;">GEX: {oi_d.get('gex_regime','—')}</span><br/>
  <span style="color:#888;">Gamma Flip: ₹{oi_d['gamma_flip']:,.0f} ({gex_flip_rel}) &nbsp;·&nbsp;
  Net GEX: {gex_net:+,.0f}</span>
</div>""", unsafe_allow_html=True)

        # ── Volatility Smile ──
        if not chain_df.empty:
            smile_df = chain_df[(chain_df.Strike >= atm_k - CFG["oi_strikes"]*step) &
                                (chain_df.Strike <= atm_k + CFG["oi_strikes"]*step)].copy()
            smile_df["CE_IV_pct"] = smile_df.CE_IV.apply(lambda v: (v/100 if v>2 else v)*100 if v else None)
            smile_df["PE_IV_pct"] = smile_df.PE_IV.apply(lambda v: (v/100 if v>2 else v)*100 if v else None)
            valid_smile = smile_df[(smile_df.CE_IV_pct > 0) | (smile_df.PE_IV_pct > 0)]
            if not valid_smile.empty:
                fig_smile = go.Figure()
                fig_smile.add_trace(go.Scatter(
                    x=valid_smile.Strike, y=valid_smile.CE_IV_pct,
                    mode="lines+markers", name="CE IV%",
                    line=dict(color="#ff3b3b",width=2), marker=dict(size=5)
                ))
                fig_smile.add_trace(go.Scatter(
                    x=valid_smile.Strike, y=valid_smile.PE_IV_pct,
                    mode="lines+markers", name="PE IV%",
                    line=dict(color="#00d084",width=2), marker=dict(size=5)
                ))
                fig_smile.add_hline(y=atm_iv*100, line=dict(color="#ffb347",dash="dot",width=1),
                                    annotation_text=f"ATM IV {atm_iv*100:.1f}%")
                if hv20:
                    fig_smile.add_hline(y=hv20*100, line=dict(color="#555",dash="dot",width=1),
                                        annotation_text=f"HV20 {hv20*100:.1f}%")
                fig_smile.add_vline(x=spot, line=dict(color="#ffb347",dash="dot",width=1),
                                    annotation_text=f"Spot {spot:.0f}")

                # Skew annotation
                skew_txt = ""
                if oi_d and oi_d.get("skew_pp") is not None:
                    skew_pp   = oi_d["skew_pp"]
                    skew_col  = "#ff3b3b" if skew_pp > 2 else ("#1e90ff" if skew_pp < -1 else "#888")
                    skew_txt  = f"Skew: {skew_pp:+.1f}pp"

                fig_smile.update_layout(
                    title=f"Volatility Smile — IV by Strike{'  ·  ' + skew_txt if skew_txt else ''}",
                    height=280, plot_bgcolor="#000", paper_bgcolor="#000",
                    font=dict(color="#e8e8e8",family="IBM Plex Mono",size=9),
                    legend=dict(orientation="h",y=1.12), margin=dict(t=40,b=10),
                    yaxis=dict(title="IV%",gridcolor="#111")
                )
                st.plotly_chart(fig_smile, use_container_width=True)

                # Skew card
                if oi_d and oi_d.get("skew_label") and oi_d["skew_label"] != "—":
                    skew_pp   = oi_d.get("skew_pp", 0) or 0
                    skew_c    = "#ff3b3b" if skew_pp > 2 else ("#1e90ff" if skew_pp < -1 else "#888")
                    st.markdown(f"""
<div style="border-left:3px solid {skew_c};padding:6px 12px;
font-family:'IBM Plex Mono',monospace;font-size:.66rem;color:{skew_c};margin:4px 0;">
  SKEW: {oi_d['skew_label']}
</div>""", unsafe_allow_html=True)

    with st.expander("◼ HOW TO READ OI DATA"):
        st.markdown("""
**Max Pain** — Strike where option writers lose the least. Price gravitates here near expiry.
- Spot far above Max Pain → downward pull likely
- Spot far below Max Pain → upward pull likely

**Put-Call Ratio (PCR OI)**
- > 1.3 → Heavy put writing → dealers net short puts → strong support → **contrarian BULLISH**
- 0.9–1.2 → Balanced
- < 0.7 → Heavy call buying → too much optimism → **contrarian BEARISH**

**Call Wall** — Highest CE OI cluster = hard resistance. Dealers short calls → sell futures above = supply.
**Put Wall** — Highest PE OI cluster = strong support. Dealers short puts → buy futures below = demand.

**Expected Move** = ATM straddle ÷ Spot. Market's priced ±1σ move to expiry.

**Gamma Exposure (GEX)**
- Positive GEX: Dealers are net short gamma → buy dips, sell rallies → **range-bound / vol suppression**
- Negative GEX: Dealers are net long gamma → chase price → **trending / vol expansion**
- Gamma Flip: Price level where dealer hedging direction reverses. Crossing it can trigger vol expansion.

**Volatility Skew**
- Downside put IV > upside call IV (normal for indices): protection demand elevated
- Skew narrowing: bearish hedging decreasing, market less worried about downside
- Upside call IV > downside put IV (rare): upside breakout speculation
""")

# ══════════════════════════════════════════════════════════════
# TAB 7 — PAYOFF BUILDER
# ══════════════════════════════════════════════════════════════
with t_payoff:
    st.markdown("### 💹 Strategy Payoff Builder")
    st.caption("Build multi-leg strategies. Quick-load buttons fill theoretical prices automatically.")

    # Correct NSE F&O lot sizes (as of 2024 SEBI revision)
    # Nifty: 75 (revised from 50 in Nov 2024), BankNifty: 15, FinNifty: 40, MidcapNifty: 75
    _LOT_SIZES = {
        "NIFTY": 75, "BANKNIFTY": 15, "FINNIFTY": 40, "MIDCPNIFTY": 75,
        "RELIANCE": 250, "HDFCBANK": 550, "ICICIBANK": 700, "INFY": 400,
        "TCS": 150, "LT": 150, "SBIN": 1500, "AXISBANK": 625,
        "KOTAKBANK": 400, "BHARTIARTL": 500, "ITC": 3200,
        "BAJFINANCE": 125, "WIPRO": 1500, "HCLTECH": 350,
        "TATAMOTORS": 1425, "MARUTI": 100,
    }
    LOT_SIZE = _LOT_SIZES.get(sym.upper(), 500)  # default 500 for unlisted stocks

    # ── Quick load buttons ──
    ql1,ql2,ql3,ql4,ql5,ql6 = st.columns(6)
    sv = float(step)

    def bs_c(k): return round(bs_price(spot, k, T, r, atm_iv, "call"), 2)
    def bs_p(k): return round(bs_price(spot, k, T, r, atm_iv, "put"),  2)

    if ql1.button("Bull Call Spread",  key="ql_bcs"):
        st.session_state.payoff_legs = [
            {"Opt":"CE","Strike":atm_k,     "Premium":bs_c(atm_k),     "Qty":1,"Action":"Buy"},
            {"Opt":"CE","Strike":atm_k+sv,  "Premium":bs_c(atm_k+sv),  "Qty":1,"Action":"Sell"},
        ]; st.rerun()
    if ql2.button("Bear Put Spread",   key="ql_bps"):
        st.session_state.payoff_legs = [
            {"Opt":"PE","Strike":atm_k,     "Premium":bs_p(atm_k),     "Qty":1,"Action":"Buy"},
            {"Opt":"PE","Strike":atm_k-sv,  "Premium":bs_p(atm_k-sv),  "Qty":1,"Action":"Sell"},
        ]; st.rerun()
    if ql3.button("Long Straddle",     key="ql_str"):
        st.session_state.payoff_legs = [
            {"Opt":"CE","Strike":atm_k,"Premium":bs_c(atm_k),"Qty":1,"Action":"Buy"},
            {"Opt":"PE","Strike":atm_k,"Premium":bs_p(atm_k),"Qty":1,"Action":"Buy"},
        ]; st.rerun()
    if ql4.button("Iron Condor",       key="ql_ic"):
        st.session_state.payoff_legs = [
            {"Opt":"PE","Strike":atm_k-sv,   "Premium":bs_p(atm_k-sv),   "Qty":1,"Action":"Sell"},
            {"Opt":"PE","Strike":atm_k-2*sv, "Premium":bs_p(atm_k-2*sv), "Qty":1,"Action":"Buy"},
            {"Opt":"CE","Strike":atm_k+sv,   "Premium":bs_c(atm_k+sv),   "Qty":1,"Action":"Sell"},
            {"Opt":"CE","Strike":atm_k+2*sv, "Premium":bs_c(atm_k+2*sv), "Qty":1,"Action":"Buy"},
        ]; st.rerun()
    if ql5.button("Butterfly",         key="ql_bf"):
        st.session_state.payoff_legs = [
            {"Opt":"CE","Strike":atm_k-sv,  "Premium":bs_c(atm_k-sv),  "Qty":1,"Action":"Buy"},
            {"Opt":"CE","Strike":atm_k,     "Premium":bs_c(atm_k),     "Qty":2,"Action":"Sell"},
            {"Opt":"CE","Strike":atm_k+sv,  "Premium":bs_c(atm_k+sv),  "Qty":1,"Action":"Buy"},
        ]; st.rerun()
    if ql6.button("Short Strangle",    key="ql_ss"):
        st.session_state.payoff_legs = [
            {"Opt":"PE","Strike":atm_k-sv,  "Premium":bs_p(atm_k-sv),  "Qty":1,"Action":"Sell"},
            {"Opt":"CE","Strike":atm_k+sv,  "Premium":bs_c(atm_k+sv),  "Qty":1,"Action":"Sell"},
        ]; st.rerun()

    st.divider()

    # ── Manual leg entry ──
    with st.expander("➕ Add Custom Leg", expanded=len(st.session_state.payoff_legs)==0):
        ac1,ac2,ac3,ac4,ac5 = st.columns([1,2,1.5,1,1.5])
        with ac1: l_opt    = st.selectbox("CE/PE", ["CE","PE"], key="nl_opt")
        with ac2: l_strike = st.number_input("Strike", value=float(atm_k), step=float(sv), key="nl_strike")
        with ac3: l_prem   = st.number_input("Premium ₹", value=0.0, min_value=0.0, step=0.5, key="nl_prem")
        with ac4: l_qty    = st.number_input("Lots", value=1, min_value=1, key="nl_qty")
        with ac5: l_act    = st.selectbox("Buy/Sell", ["Buy","Sell"], key="nl_act")
        if st.button("Add Leg ➕", key="add_leg_btn2"):
            prem = l_prem if l_prem > 0 else (bs_c(l_strike) if l_opt=="CE" else bs_p(l_strike))
            st.session_state.payoff_legs.append({
                "Opt":l_opt,"Strike":l_strike,"Premium":round(prem,2),"Qty":l_qty,"Action":l_act
            })
            st.rerun()

    # ── Display & edit legs ──
    if st.session_state.payoff_legs:
        legs_df = pd.DataFrame(st.session_state.payoff_legs)
        st.dataframe(legs_df, use_container_width=True, hide_index=True)

        pc_clear, pc_lot = st.columns([1,3])
        with pc_clear:
            if st.button("🗑 Clear All", key="clear_legs2"):
                st.session_state.payoff_legs = []; st.rerun()
        with pc_lot:
            lot_inp = st.number_input("Lot Size", min_value=1, value=LOT_SIZE, key="lot_size_inp")

        # Chart range = ±3σ based on expected move from straddle
        # exp_move_pct is ±1σ; ×3 gives the 99.7% probability range
        _exp_move_frac = oi_d.get("exp_move_pct", 0) / 100.0 if oi_d else 0
        if _exp_move_frac <= 0:
            # Fallback: derive from IV and T (no OI data)
            _exp_move_frac = atm_iv * math.sqrt(2.0 / math.pi) * math.sqrt(T) if T > 0 else 0.05
        _range_frac  = max(3 * _exp_move_frac, 0.05)  # at least ±5% for very short DTE
        px_range     = np.linspace(spot * (1 - _range_frac), spot * (1 + _range_frac), 400)
        payoff_total = np.zeros(len(px_range))
        total_cost   = 0.0

        for leg in st.session_state.payoff_legs:
            k_   = float(leg["Strike"])
            pr_  = float(leg["Premium"])
            qty_ = int(leg["Qty"]) * int(lot_inp)
            d_   = 1 if leg["Action"] == "Buy" else -1
            intr = np.maximum(px_range - k_, 0) if leg["Opt"]=="CE" else np.maximum(k_ - px_range, 0)
            payoff_total += d_ * (intr - pr_) * qty_
            total_cost   += d_ * pr_ * qty_

        # Breakevens
        bes = []
        for i in range(1, len(payoff_total)):
            if payoff_total[i-1] * payoff_total[i] < 0:
                frac = abs(payoff_total[i-1]) / (abs(payoff_total[i-1]) + abs(payoff_total[i]))
                bes.append(round(px_range[i-1] + frac*(px_range[i]-px_range[i-1]), 1))

        max_profit = payoff_total.max()
        max_loss   = payoff_total.min()

        # Payoff chart
        fig_pay = go.Figure()
        fig_pay.add_trace(go.Scatter(
            x=px_range, y=payoff_total,
            fill="tozeroy", fillcolor="rgba(0,208,132,0.06)",
            line=dict(color="#00d084",width=2.5), name="P&L (₹)"
        ))
        fig_pay.add_hline(y=0, line=dict(color="#333",dash="dot",width=1))
        fig_pay.add_vline(x=spot, line=dict(color="#ffb347",dash="dot",width=1.5),
                          annotation_text=f"Spot ₹{spot:.0f}")
        fig_pay.add_vline(x=atm_k, line=dict(color="#333",dash="dash",width=1),
                          annotation_text=f"ATM {atm_k}")
        for be in bes:
            fig_pay.add_vline(x=be, line=dict(color="#ff8c00",dash="dash",width=1),
                              annotation_text=f"BE ₹{be:.0f}",
                              annotation=dict(font=dict(color="#ff8c00",size=8)))
        if oi_d:
            fig_pay.add_vline(x=oi_d["max_pain"],
                              line=dict(color="#9c27b0",dash="dot",width=1),
                              annotation_text=f"MaxPain ₹{oi_d['max_pain']:.0f}")
            fig_pay.add_vrect(x0=oi_d["put_wall"], x1=oi_d["call_wall"],
                              fillcolor="rgba(0,208,132,0.04)", layer="below",
                              annotation_text="OI Range", annotation_position="top left",
                              annotation=dict(font=dict(size=8,color="#444")))
        fig_pay.update_layout(
            height=420, plot_bgcolor="#000", paper_bgcolor="#000",
            font=dict(color="#e8e8e8",family="IBM Plex Mono",size=9),
            xaxis=dict(title="Underlying Price at Expiry",gridcolor="#111"),
            yaxis=dict(title="P&L (₹)",gridcolor="#111",zeroline=True,zerolinecolor="#2a2a2a"),
            legend=dict(orientation="h",y=1.08),
            margin=dict(t=30,b=10)
        )
        st.plotly_chart(fig_pay, use_container_width=True)

        # Stats row
        s1,s2,s3,s4,s5,s6 = st.columns(6)
        s1.metric("Max Profit",  f"₹{max_profit:,.0f}")
        s2.metric("Max Loss",    f"₹{max_loss:,.0f}")
        cost_lbl = "Debit" if total_cost > 0 else "Credit"
        s3.metric("Net Cost",    f"₹{abs(total_cost):,.0f}", delta=cost_lbl)
        s4.metric("Breakevens",  ", ".join([f"₹{b:.0f}" for b in bes]) or "None")
        rr = abs(max_profit/max_loss) if max_loss != 0 else float("inf")
        s5.metric("Reward:Risk", f"{rr:.2f}×" if rr != float("inf") else "∞")
        s6.metric("Legs",        str(len(st.session_state.payoff_legs)))

    else:
        st.info("Click a Quick Load button above, or add legs manually to build a payoff diagram.")
        st.caption("Premiums auto-fill from Black-Scholes at current ATM IV if you don't enter them.")
