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

# ── Bloomberg Terminal Theme ──────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&display=swap');
:root {
    --bb-bg: #0a0a0a; --bb-surface: #111111; --bb-border: #2a2a2a;
    --bb-amber: #ff8c00; --bb-amber2: #ffb347; --bb-green: #00d084;
    --bb-red: #ff3b3b; --bb-blue: #1e90ff; --bb-white: #e8e8e8; --bb-muted: #666666;
}
html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bb-bg) !important;
    color: var(--bb-white) !important;
    font-family: 'IBM Plex Mono', monospace !important;
}
[data-testid="stSidebar"] { background-color: #060606 !important; border-right: 1px solid var(--bb-border) !important; }
h1 { font-family: 'IBM Plex Mono', monospace !important; color: var(--bb-amber) !important;
     font-size: 1.0rem !important; font-weight: 600 !important; letter-spacing: .15em !important;
     text-transform: uppercase !important; border-bottom: 1px solid var(--bb-amber) !important; padding-bottom: 4px !important; }
h2 { color: var(--bb-amber2) !important; font-size: .85rem !important; letter-spacing: .1em !important; text-transform: uppercase !important; }
h3 { color: var(--bb-white) !important; font-size: .78rem !important; letter-spacing: .08em !important; }
[data-testid="metric-container"] {
    background: var(--bb-surface) !important; border: 1px solid var(--bb-border) !important;
    border-left: 3px solid var(--bb-amber) !important; padding: 8px 12px !important; border-radius: 0 !important;
}
[data-testid="metric-container"] label { color: var(--bb-muted) !important; font-size: .58rem !important; letter-spacing: .1em !important; text-transform: uppercase !important; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color: var(--bb-amber) !important; font-size: 1rem !important; font-weight: 600 !important; }
[data-testid="stDataFrame"] { border: 1px solid var(--bb-border) !important; }
.stDataFrame thead tr th { background: #1a1400 !important; color: var(--bb-amber) !important; font-size: .62rem !important; letter-spacing: .1em !important; text-transform: uppercase !important; border-bottom: 1px solid var(--bb-amber) !important; }
.stDataFrame tbody tr td { font-size: .7rem !important; color: var(--bb-white) !important; border-bottom: 1px solid #1a1a1a !important; }
.stDataFrame tbody tr:hover td { background: #1a1400 !important; }
.stButton > button { background: #1a1400 !important; color: var(--bb-amber) !important; border: 1px solid var(--bb-amber) !important;
    border-radius: 0 !important; font-size: .7rem !important; letter-spacing: .1em !important; text-transform: uppercase !important; padding: 6px 14px !important; }
.stButton > button:hover { background: var(--bb-amber) !important; color: #000 !important; }
.stSelectbox > div > div, .stTextInput > div > div {
    background: var(--bb-surface) !important; border: 1px solid var(--bb-border) !important;
    border-radius: 0 !important; color: var(--bb-white) !important; font-size: .72rem !important; }
.stTabs [data-baseweb="tab-list"] { background: var(--bb-surface) !important; border-bottom: 1px solid var(--bb-border) !important; gap: 0 !important; }
.stTabs [data-baseweb="tab"] { background: transparent !important; color: var(--bb-muted) !important; font-size: .65rem !important; letter-spacing: .1em !important; text-transform: uppercase !important; border-radius: 0 !important; border-right: 1px solid var(--bb-border) !important; padding: 8px 14px !important; }
.stTabs [aria-selected="true"] { background: #1a1400 !important; color: var(--bb-amber) !important; border-bottom: 2px solid var(--bb-amber) !important; }
hr { border-color: var(--bb-border) !important; margin: 8px 0 !important; }
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--bb-bg); }
::-webkit-scrollbar-thumb { background: var(--bb-border); }
::-webkit-scrollbar-thumb:hover { background: var(--bb-amber); }
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

def fetch_option_chain(instrument_key, expiry):
    """Fetch live option chain from Upstox."""
    url = "https://api.upstox.com/v2/option/chain"
    try:
        r = requests.get(url, headers=HEADERS,
                         params={"instrument_key": instrument_key, "expiry_date": expiry},
                         timeout=15)
        if r.status_code == 200:
            return r.json().get("data", [])
    except: pass
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

def parse_chain(raw_data, spot):
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
            lambda k: "ATM" if abs(k - spot)/spot < 0.006
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

    # F1: EMA Stack (±20)
    es = 0
    for price, ref, pts in [(ltp,e9v,5),(ltp,e20v,5),(ltp,e50v,5),(e9v,e20v,3),(e20v,e50v,2)]:
        es += pts if price > ref else -pts
    factors["EMA Stack"]  = es; score += es

    # F2: RSI (±15)
    rs = 15 if rsi>60 else 8 if rsi>50 else 0 if rsi>45 else -8 if rsi>35 else -15
    factors["RSI(14)"]    = rs; score += rs

    # F3: MACD Histogram (±10)
    ms = 10 if macd_h > 0 else -10
    factors["MACD Hist"]  = ms; score += ms

    # F4: Bollinger Position (±10)
    bs2 = 10 if bb_pct>0.8 else 6 if bb_pct>0.6 else 0 if bb_pct>0.4 else -6 if bb_pct>0.2 else -10
    factors["BB Position"]= bs2; score += bs2

    # F5: Volume confirmation (±10)
    vs = 10 if vol_ratio>1.5 else 5 if vol_ratio>1.2 else 0 if vol_ratio>0.8 else -5
    factors["Volume"]     = vs; score += vs

    # F6: 200 EMA (±15)
    e2 = 15 if ltp > e200v else -15
    factors["200 EMA"]    = e2; score += e2

    # F7: 5D momentum (±10)
    if len(c) >= 6:
        base = float(c.iloc[-6])
        ret5 = (ltp/base - 1)*100 if base != 0 else 0
        m5 = 10 if ret5>3 else 5 if ret5>1 else 0 if ret5>-1 else -5 if ret5>-3 else -10
    else:
        m5 = 0
    factors["5D Return"]  = m5; score += m5

    bias = ("STRONGLY BULLISH" if score >= 30 else "BULLISH"   if score >= 12 else
            "NEUTRAL"          if score >  -12 else "BEARISH"  if score >= -30 else "STRONGLY BEARISH")

    return {
        "bias": bias, "score": score, "rsi": round(rsi,1),
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
    if   ivr < 20: return "LOW VOL",      "BUY premium — debit spreads / long options / straddles", "#1e90ff"
    elif ivr < 40: return "NORMAL-LOW",   "Slight buy lean — calendars / ratio spreads",            "#7ec8e3"
    elif ivr < 60: return "NORMAL",       "Neutral — use balanced spreads both sides",               "#ffb347"
    elif ivr < 75: return "ELEVATED",     "Lean SELL — credit spreads / iron condor",               "#ff8c00"
    else:          return "HIGH VOL",     "SELL premium — iron condors / strangles / short straddle","#ff3b3b"

# ============================================================
# OI ANALYSIS
# ============================================================

def oi_analysis(chain_df, spot):
    if chain_df.empty: return {}
    df = chain_df[(chain_df.Strike > spot*0.85) & (chain_df.Strike < spot*1.15)].copy()
    if df.empty: return {}

    def pain(ep, d):
        return (((ep - d.Strike).clip(lower=0) * d.CE_OI) +
                ((d.Strike - ep).clip(lower=0) * d.PE_OI)).sum()

    pain_map = {row.Strike: pain(row.Strike, df) for _, row in df.iterrows()}
    max_pain  = min(pain_map, key=pain_map.get) if pain_map else spot

    total_ce  = float(df.CE_OI.sum())
    total_pe  = float(df.PE_OI.sum())
    pcr_oi    = round(total_pe / (total_ce + 1e-9), 3)
    call_wall = float(df.loc[df.CE_OI.idxmax(), "Strike"])
    put_wall  = float(df.loc[df.PE_OI.idxmax(), "Strike"])

    atm_r = df.iloc[(df.Strike - spot).abs().argsort()[:1]]
    straddle = float(atm_r.CE_LTP.values[0] + atm_r.PE_LTP.values[0]) if not atm_r.empty else 0
    exp_move = round(straddle / spot * 100, 2) if spot > 0 else 0

    pcr_sig = ("BULLISH — heavy put writing = strong support below" if pcr_oi > 1.3 else
               "NEUTRAL — balanced OI"                             if pcr_oi > 0.9 else
               "SLIGHT BEARISH LEAN — more call OI"               if pcr_oi > 0.7 else
               "BEARISH — heavy call writing = resistance above")

    return dict(max_pain=round(max_pain,2), pcr_oi=pcr_oi,
                call_wall=round(call_wall,2), put_wall=round(put_wall,2),
                total_ce_oi=int(total_ce), total_pe_oi=int(total_pe),
                atm_straddle=round(straddle,2), exp_move_pct=exp_move, pcr_signal=pcr_sig)

# ============================================================
# STRATEGY RECOMMENDATION ENGINE
# ============================================================

def recommend_strategies(bias, vol_lbl, dte, spot, atm, step, ivr):
    is_bull  = "BULL" in bias
    is_bear  = "BEAR" in bias
    hi_vol   = ivr >= 60
    lo_vol   = ivr < 35
    sv       = float(step)
    recs     = []

    def add(name, type_, legs, rationale, risk, reward, ideal_dte, score):
        recs.append({"Strategy":name,"Type":type_,"Legs":legs,
                     "Rationale":rationale,"Max Risk":risk,"Max Reward":reward,
                     "Ideal DTE":ideal_dte,"Score":score})

    # ── BULLISH ──────────────────────────────
    if is_bull:
        if lo_vol:
            add("Long ATM Call",
                "Debit · Directional",
                f"BUY {atm} CE",
                "Low IV = cheap premium. Pure directional. Max loss = premium paid. Best when you expect a swift move.",
                "Premium paid","Unlimited","15–45 DTE", 96)
            add("Bull Call Spread",
                "Debit · Defined Risk",
                f"BUY {atm} CE  +  SELL {atm+sv:.0f} CE",
                "Cuts premium cost vs naked call. Profits if stock closes above upper strike at expiry.",
                "Net debit",f"Spread width − debit","15–30 DTE", 90)
            add("Call Ratio Backspread",
                "Credit–even · Vol + Direction",
                f"SELL 1× {atm-sv:.0f} CE  +  BUY 2× {atm} CE",
                "Enter for credit or zero cost. Profits from big upside move OR vol expansion. Limited loss in the middle.",
                "Limited (near short strike)","Unlimited above upper BE","30–45 DTE", 80)
        elif hi_vol:
            add("Bull Put Spread",
                "Credit · Defined Risk",
                f"SELL {atm} PE  +  BUY {atm-sv:.0f} PE",
                "Sell expensive puts. Keep the credit as long as stock stays above short strike. High IV = fat credit.",
                "Spread width − credit","Net credit received","7–21 DTE", 95)
            add("Short Put (OTM)",
                "Credit · Income",
                f"SELL {atm-sv:.0f} PE",
                "Collect rich premium. Obligated to buy stock at strike if assigned — only use for stocks you want to own.",
                "Strike − premium","Premium received","7–21 DTE", 82)
            add("Jade Lizard",
                "Credit · Slight Bullish",
                f"SELL {atm} PE  +  SELL {atm+sv:.0f} CE  +  BUY {atm+2*sv:.0f} CE",
                "No upside risk if total credit > call spread width. Benefits from high IV in both directions.",
                "Put strike − total credit","Total credit","14–21 DTE", 78)
        else:
            add("Bull Call Spread",
                "Debit · Defined Risk",
                f"BUY {atm} CE  +  SELL {atm+sv:.0f} CE",
                "Clean risk/reward. Wins on moderate upside move. Lower breakeven than naked call.",
                "Net debit","Spread width − debit","15–30 DTE", 90)
            add("Long OTM Call (+1)",
                "Debit · High Leverage",
                f"BUY {atm+sv:.0f} CE",
                "Cheaper than ATM. Higher leverage, needs bigger move. Good for event-driven plays.",
                "Premium paid","Unlimited","10–25 DTE", 72)

    # ── BEARISH ──────────────────────────────
    if is_bear:
        if lo_vol:
            add("Long ATM Put",
                "Debit · Directional",
                f"BUY {atm} PE",
                "Low IV = cheap downside protection. Pure directional. Max loss = premium paid.",
                "Premium paid","Strike − premium","15–45 DTE", 96)
            add("Bear Put Spread",
                "Debit · Defined Risk",
                f"BUY {atm} PE  +  SELL {atm-sv:.0f} PE",
                "Reduces cost vs naked put. Wins if stock falls below lower strike.",
                "Net debit","Spread width − debit","15–30 DTE", 88)
        elif hi_vol:
            add("Bear Call Spread",
                "Credit · Defined Risk",
                f"SELL {atm} CE  +  BUY {atm+sv:.0f} CE",
                "Sell expensive calls above current price. Keep credit if stock stays below short strike.",
                "Spread width − credit","Net credit","7–21 DTE", 94)
            add("Short Call (OTM)",
                "Credit · Aggressive",
                f"SELL {atm+sv:.0f} CE",
                "Rich call premium to sell. High risk — use only with clear bearish conviction and stop loss.",
                "Theoretically unlimited","Premium received","7–14 DTE", 68)
        else:
            add("Bear Put Spread",
                "Debit · Defined Risk",
                f"BUY {atm} PE  +  SELL {atm-sv:.0f} PE",
                "Balanced risk/reward for moderate downside. Standard short-term bearish play.",
                "Net debit","Spread width − debit","15–30 DTE", 88)
            add("Bear Call Spread",
                "Credit · Defined Risk",
                f"SELL {atm} CE  +  BUY {atm+sv:.0f} CE",
                "Collect premium above current price. Wins if stock stays flat or falls.",
                "Spread width − credit","Net credit","7–21 DTE", 82)

    # ── NEUTRAL / RANGE ──────────────────────
    if "NEUTRAL" in bias or hi_vol:
        if hi_vol:
            add("Iron Condor",
                "Credit · Non-Directional",
                f"SELL {atm-sv:.0f} PE + BUY {atm-2*sv:.0f} PE  |  SELL {atm+sv:.0f} CE + BUY {atm+2*sv:.0f} CE",
                "Maximum premium collection in high IV. Wins if stock stays between short strikes. "
                "Most popular professional strategy for range-bound markets.",
                "Spread width − total credit","Total credit received","14–30 DTE", 97)
            add("Short Strangle",
                "Credit · Uncapped Risk",
                f"SELL {atm-sv:.0f} PE  +  SELL {atm+sv:.0f} CE",
                "Higher credit than iron condor. No wing protection = unlimited risk both sides. "
                "Must manage aggressively at 50% profit or 2× loss.",
                "Theoretically unlimited","Total premium","7–21 DTE", 83)
            add("Short Straddle",
                "Credit · Max Theta",
                f"SELL {atm} CE  +  SELL {atm} PE",
                "Maximum theta at ATM. Needs stock to pin very close to ATM. Highest risk — "
                "delta-hedge or exit quickly if stock moves.",
                "Unlimited both sides","Total premium","7–14 DTE", 75)
        elif lo_vol:
            add("Long Straddle",
                "Debit · Vol Expansion",
                f"BUY {atm} CE  +  BUY {atm} PE",
                "Low IV = cheap double. Profits from ANY large move either direction, or from IV expansion. "
                "Needs move > combined premium to profit.",
                "Combined premium paid","Unlimited","30–60 DTE", 92)
            add("Long Strangle",
                "Debit · Cheaper Vol Play",
                f"BUY {atm+sv:.0f} CE  +  BUY {atm-sv:.0f} PE",
                "Cheaper than straddle, needs bigger move. Excellent if you expect a large event-driven move.",
                "Combined premium","Unlimited","30–60 DTE", 85)
            add("Calendar Spread",
                "Debit · Theta + Vol",
                f"SELL near {atm} CE  +  BUY far {atm} CE",
                "Sell near-term theta, buy longer-dated vega. Profits from flat market + "
                "any IV expansion. Best when front-month IV > back-month IV.",
                "Net debit","Limited (peaks at ATM on front-month expiry)","Near:7–14 / Far:30–45 DTE", 82)
        else:
            add("Iron Condor",
                "Credit · Non-Directional",
                f"SELL {atm-sv:.0f} PE + BUY {atm-2*sv:.0f} PE  |  SELL {atm+sv:.0f} CE + BUY {atm+2*sv:.0f} CE",
                "Collect premium from both sides with defined risk. Ideal for a sideways market expectation.",
                "Spread width − credit","Total credit","14–30 DTE", 88)
            add("Iron Butterfly",
                "Credit · Tighter Range",
                f"SELL {atm} CE + SELL {atm} PE  |  BUY {atm+sv:.0f} CE + BUY {atm-sv:.0f} PE",
                "Higher credit than condor. Needs stock to stay near ATM. "
                "Better reward, narrower profit zone.",
                "Spread width − credit","Net credit","14–21 DTE", 78)
            add("ATM Butterfly",
                "Debit · Precision Play",
                f"BUY {atm-sv:.0f} CE  +  SELL 2× {atm} CE  +  BUY {atm+sv:.0f} CE",
                "Low cost, defined risk, maximum profit if stock pins ATM at expiry. "
                "Use when expecting consolidation around ATM.",
                "Net debit","Spread − 2×debit","7–21 DTE", 72)

    # DTE-based hedges
    if dte <= 5:
        add("Same-Day / Weekly Straddle Sell",
                "Credit · Near Expiry",
                f"SELL {atm} CE  +  SELL {atm} PE (weekly/near expiry)",
                "Near expiry = explosive theta decay. ATM options lose most value in last 2–5 days. "
                "Must monitor constantly and exit at 50% profit. NEVER hold to expiry naked.",
                "Large if stock moves","Theta collected","1–5 DTE", 80)

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

    spot_override = st.number_input("Spot Price Override (0 = live)", min_value=0.0,
                                     value=0.0, step=1.0, key="spot_ovr_sidebar")
    dte_sidebar   = st.number_input("DTE (for Greeks/Strategy)", min_value=1, max_value=90,
                                     value=7, key="dte_sidebar")
    rfr_sidebar   = st.number_input("Risk-Free Rate %", min_value=0.0, max_value=15.0,
                                     value=6.5, step=0.1, key="rfr_sidebar")
    st.divider()
    load_btn = st.button("⚡ LOAD OPTIONS INTEL", use_container_width=True, key="load_opt_main")

    st.divider()
    st.caption(f"Strike Step: {step_val}")
    if ikey: st.caption(f"Key: {ikey[:30]}…")

# ============================================================
# LOAD LOGIC
# ============================================================

if load_btn:
    with st.spinner(f"Loading {sym_sel} options intelligence…"):

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
        chain_raw = fetch_option_chain(ikey, expiry_sel) if ikey and expiry_sel else []
        chain_df  = parse_chain(chain_raw, spot)

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
            T_tmp = dte_sidebar / 365.0
            if not chain_df.empty:
                row = chain_df.iloc[(chain_df.Strike - spot).abs().argsort()[:1]]
                if not row.empty:
                    strd = float(row.CE_LTP.values[0]) + float(row.PE_LTP.values[0])
                    if strd > 0 and T_tmp > 0:
                        atm_iv = strd / (0.8 * spot * math.sqrt(T_tmp)) if spot > 0 else None
        if not atm_iv:
            atm_iv = hv20 or 0.20

        # 6. OI
        oi_d = oi_analysis(chain_df, spot)

        # 7. DTE from expiry date
        actual_dte = dte_sidebar
        if expiry_sel:
            try:
                exp_d   = datetime.strptime(expiry_sel, "%Y-%m-%d").date()
                actual_dte = max((exp_d - datetime.now().date()).days, 1)
            except: pass

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
        st.session_state.opt_rfr        = rfr_sidebar / 100.0
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
bias_score = bias_res.get("score", 0)

ivr          = iv_rank([atm_iv]*30, atm_iv)   # simplified; real app stores IV history
v_lbl, v_act, v_col = vol_regime(ivr)
strat_recs   = recommend_strategies(bias, v_lbl, dte, spot, atm_k, step, ivr)

BIAS_COLORS = {
    "STRONGLY BULLISH":"#00d084","BULLISH":"#7dca84","NEUTRAL":"#ffb347",
    "BEARISH":"#ff7777","STRONGLY BEARISH":"#ff3b3b"
}
bc = BIAS_COLORS.get(bias, "#888")

# ── TOP HEADER BAR ──
iv_vs_hv = (atm_iv - hv20)*100
iv_sign  = "+" if iv_vs_hv >= 0 else ""
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
    Expected Move: ±{oi_d.get('exp_move_pct',0):.1f}% &nbsp;·&nbsp;
    IV Rank: {ivr:.0f}
  </div>
</div>""", unsafe_allow_html=True)

    with ov2:
        # OI key metrics
        if oi_d:
            pcr_c = "#00d084" if oi_d.get("pcr_oi",1) >= 1.0 else "#ff3b3b"
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
  </div>
  <div style="margin-top:8px;color:{pcr_c};font-size:.62rem;">{oi_d.get('pcr_signal','—')}</div>
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
            mm   = "ATM" if abs(k-spot)/spot<0.006 else ("ITM-C" if k<spot else "OTM-C")
            syn_rows.append({
                "Strike":k,"Moneyness":mm,
                "CE Price":round(ce_p,2),"CE IV%":round(atm_iv*100,1),
                "CE Δ":cg["delta"],"CE θ":round(cg["theta"],3),"CE ν":round(cg["vega"],3),
                "PE Price":round(pe_p,2),"PE IV%":round(atm_iv*100,1),
                "PE Δ":pg["delta"],"PE θ":round(pg["theta"],3),"PE ν":round(pg["vega"],3),
            })
        st.dataframe(pd.DataFrame(syn_rows), use_container_width=True, hide_index=True)
    else:
        # Filter ±8 strikes
        disp_c = chain_df[(chain_df.Strike >= spot*0.91) & (chain_df.Strike <= spot*1.09)].copy()

        # Add directional + IV edge signal per row
        def row_signal(row):
            ce_iv_r = float(row.CE_IV); pe_iv_r = float(row.PE_IV)
            ce_iv   = ce_iv_r/100 if ce_iv_r>2 else (ce_iv_r or atm_iv)
            pe_iv   = pe_iv_r/100 if pe_iv_r>2 else (pe_iv_r or atm_iv)
            ce_edge = (ce_iv - hv20)*100
            pe_edge = (pe_iv - hv20)*100
            ce_dir  = "BUY" if bias_score>=12 else "SELL" if bias_score<=-12 else "—"
            pe_dir  = "BUY" if bias_score<=-12 else "SELL" if bias_score>=12 else "—"
            ce_vol  = "SELL (rich)" if ce_edge>15 else "BUY (cheap)" if ce_edge<-15 else "—"
            pe_vol  = "SELL (rich)" if pe_edge>15 else "BUY (cheap)" if pe_edge<-15 else "—"
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
        mm  = "ATM" if abs(k-spot)/spot<0.006 else ("ITM" if k<spot else "OTM")
        g_rows.append({
            "Strike":k, "Moneyness":mm,
            "CE Price":round(cp,2), "CE IV%":round(ce_iv_use*100,1),
            "CE Δ":cg["delta"], "CE Γ":cg["gamma"],
            "CE θ/d":round(cg["theta"],3), "CE ν/1%":round(cg["vega"],3),
            "PE Price":round(pp,2), "PE IV%":round(pe_iv_use*100,1),
            "PE Δ":pg["delta"], "PE Γ":pg["gamma"],
            "PE θ/d":round(pg["theta"],3), "PE ν/1%":round(pg["vega"],3),
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
        pcr_c = "#00d084" if oi_d.get("pcr_oi",1) >= 1.0 else "#ff3b3b"
        pain_diff = spot - oi_d.get("max_pain", spot)

        o1,o2,o3,o4,o5,o6 = st.columns(6)
        o1.metric("Max Pain",       f"₹{oi_d['max_pain']:,.0f}",
                  delta=f"{pain_diff:+.0f} from spot")
        o2.metric("PCR (OI)",       f"{oi_d['pcr_oi']:.3f}")
        o3.metric("Call Wall",      f"₹{oi_d['call_wall']:,.0f}")
        o4.metric("Put Wall",       f"₹{oi_d['put_wall']:,.0f}")
        o5.metric("ATM Straddle",   f"₹{oi_d['atm_straddle']:.1f}")
        o6.metric("Exp Move ±",     f"{oi_d['exp_move_pct']:.1f}%")

        st.markdown(f"""
<div style="border-left:3px solid {pcr_c};padding:7px 12px;margin:8px 0;
font-family:'IBM Plex Mono',monospace;font-size:.7rem;color:{pcr_c};">
  PCR SIGNAL: {oi_d['pcr_signal']}
</div>""", unsafe_allow_html=True)

    if not chain_df.empty:
        oi_disp = chain_df[(chain_df.Strike>=spot*0.90)&(chain_df.Strike<=spot*1.10)].copy()

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

        # PCR by Strike
        fig_pcr = go.Figure()
        fig_pcr.add_trace(go.Bar(
            x=oi_disp.Strike, y=oi_disp.PCR,
            marker_color=["#00d084" if v>1.2 else "#ff3b3b" if v<0.8 else "#ffb347"
                          for v in oi_disp.PCR],
            name="PCR"
        ))
        fig_pcr.add_hline(y=1.0, line=dict(color="#555",dash="dot"))
        fig_pcr.add_vline(x=spot, line=dict(color="#ffb347",dash="dot",width=1))
        fig_pcr.update_layout(
            title="PCR by Strike (>1.2 bullish signal / <0.8 bearish signal)",
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

    with st.expander("◼ HOW TO READ OI DATA"):
        st.markdown("""
**Max Pain** — Strike where option writers lose the least. Price gravitates here near expiry.
- Spot far above Max Pain → downward pull likely
- Spot far below Max Pain → upward pull likely

**Put-Call Ratio (PCR OI)**
- > 1.3 → Heavy put writing → dealers net short puts → strong support → **contrarian BULLISH**
- 0.9–1.2 → Balanced
- < 0.7 → Heavy call buying → too much optimism → **contrarian BEARISH**

**Call Wall** — Highest CE OI strike = hard resistance. Dealers short calls → sell futures above = supply.
**Put Wall** — Highest PE OI strike = strong support. Dealers short puts → buy futures below = demand.

**Expected Move** = ATM straddle price ÷ Spot. This is the market's priced 1 standard deviation move by expiry.
""")

# ══════════════════════════════════════════════════════════════
# TAB 7 — PAYOFF BUILDER
# ══════════════════════════════════════════════════════════════
with t_payoff:
    st.markdown("### 💹 Strategy Payoff Builder")
    st.caption("Build multi-leg strategies. Quick-load buttons fill theoretical prices automatically.")

    LOT_SIZE = 50  # Nifty lot; adjust for stocks

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

        # ── Compute payoff ──
        px_range     = np.linspace(spot * 0.82, spot * 1.18, 400)
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
