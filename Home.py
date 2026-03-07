# ============================================================
# MONARCH — HOME  |  Global Risk Intelligence Dashboard
# pages/Home.py   (or Home.py at project root)
#
# Features:
#   • Upstox auto-login (phone → OTP → token, no manual copy-paste)
#   • Global Risk-On / Risk-Off composite score
#   • Live blinking ticker tape
#   • Global Macro panels: Equities · Bonds · FX · Commodities
#   • Fear & Greed proxy, VIX, yield spreads
#   • Polymarket-style probability bars (sourced from live data)
#   • Auto-refresh every 60 seconds
# ============================================================

import streamlit as st
import requests, json, time, math, os, re, urllib.parse, threading, webbrowser
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf

st.set_page_config(layout="wide", page_title="MONARCH — Global Risk Dashboard", page_icon="◼")

# ============================================================
# SHARED CSS — Bloomberg dark terminal (matches screener/options)
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600;700&display=swap');

:root {
    --bb-bg:      #0a0a0a;
    --bb-surface: #111111;
    --bb-surface2:#161616;
    --bb-border:  #2a2a2a;
    --bb-amber:   #ff8c00;
    --bb-amber2:  #ffb347;
    --bb-green:   #00d084;
    --bb-red:     #ff3b3b;
    --bb-blue:    #1e90ff;
    --bb-cyan:    #00ccff;
    --bb-white:   #e8e8e8;
    --bb-muted:   #888888;
    --bb-dim:     #444444;
    --bb-mono:    'IBM Plex Mono', monospace;
}

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > section,
.main, .main .block-container,
[data-testid="stMain"], [data-testid="stVerticalBlock"] {
    background-color: var(--bb-bg) !important;
    color: var(--bb-white) !important;
    font-family: var(--bb-mono) !important;
}

p, span, div, label, li, td, th, caption,
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] span,
[data-testid="stMarkdownContainer"] li {
    color: var(--bb-white) !important;
    font-family: var(--bb-mono) !important;
}

[data-testid="stSidebar"], [data-testid="stSidebar"] > div {
    background-color: #060606 !important;
    border-right: 1px solid var(--bb-border) !important;
}
[data-testid="stSidebar"] p, [data-testid="stSidebar"] span,
[data-testid="stSidebar"] label, [data-testid="stSidebar"] div {
    color: var(--bb-white) !important; font-size: 0.67rem !important;
}

h1 { color: var(--bb-amber) !important; font-size: 1rem !important;
     font-weight: 700 !important; letter-spacing: 0.18em !important;
     text-transform: uppercase !important;
     border-bottom: 2px solid var(--bb-amber) !important;
     padding-bottom: 4px !important; }
h2 { color: var(--bb-amber2) !important; font-size: 0.82rem !important;
     font-weight: 600 !important; letter-spacing: 0.12em !important;
     text-transform: uppercase !important; }
h3 { color: var(--bb-white) !important; font-size: 0.72rem !important;
     font-weight: 600 !important; letter-spacing: 0.1em !important;
     text-transform: uppercase !important; }

[data-testid="metric-container"] {
    background: var(--bb-surface) !important; border-radius: 0 !important;
    border: 1px solid var(--bb-border) !important;
    border-left: 3px solid var(--bb-amber) !important;
    padding: 8px 10px !important;
}
[data-testid="stMetricLabel"] p, [data-testid="metric-container"] label {
    color: var(--bb-muted) !important; font-size: 0.56rem !important;
    letter-spacing: 0.12em !important; text-transform: uppercase !important;
}
[data-testid="stMetricValue"] { color: var(--bb-amber) !important; font-size: 0.95rem !important; font-weight: 700 !important; }
[data-testid="stMetricDelta"] { font-size: 0.60rem !important; }

[data-testid="stDataFrame"] { border: 1px solid var(--bb-border) !important; border-radius: 0 !important; }
.stDataFrame thead tr th { background: #1a1200 !important; color: var(--bb-amber) !important;
    font-size: 0.58rem !important; font-weight: 700 !important; letter-spacing: 0.12em !important;
    text-transform: uppercase !important; border-bottom: 1px solid var(--bb-amber) !important;
    border-right: 1px solid #2a2a2a !important; padding: 5px 8px !important; white-space: nowrap !important; }
.stDataFrame tbody tr td { background: #0d0d0d !important; color: var(--bb-white) !important;
    font-size: 0.66rem !important; border-bottom: 1px solid #1a1a1a !important;
    border-right: 1px solid #1a1a1a !important; padding: 4px 8px !important; white-space: nowrap !important; }
.stDataFrame tbody tr:nth-child(odd) td { background: #111 !important; }
.stDataFrame tbody tr:hover td { background: #1a1400 !important; }

.stButton > button { background: #1a1400 !important; color: var(--bb-amber) !important;
    border: 1px solid var(--bb-amber) !important; border-radius: 0 !important;
    font-size: 0.68rem !important; letter-spacing: 0.1em !important;
    text-transform: uppercase !important; padding: 6px 14px !important; }
.stButton > button:hover { background: var(--bb-amber) !important; color: #000 !important; }

.stTextInput > div > div, .stNumberInput > div > div, .stSelectbox > div > div {
    background: var(--bb-surface) !important; border: 1px solid var(--bb-border) !important;
    border-radius: 0 !important; color: var(--bb-white) !important; font-size: 0.70rem !important; }
.stTextInput label, .stNumberInput label, .stSelectbox label {
    color: var(--bb-muted) !important; font-size: 0.60rem !important; font-family: var(--bb-mono) !important; }
input[type="text"], input[type="password"], input[type="number"] {
    background: var(--bb-surface) !important; color: var(--bb-white) !important;
    border: 1px solid var(--bb-border) !important; font-family: var(--bb-mono) !important;
    font-size: 0.70rem !important; border-radius: 0 !important; }

[data-testid="stExpander"] { background: var(--bb-surface) !important;
    border: 1px solid var(--bb-border) !important; border-radius: 0 !important; }

/* ── EXPANDER ARROW OVERLAP FIX ──
   Streamlit injects a <span> with Material Icons font rendering "keyboard_arrow_down"
   as a ligature. We must zero it out via font-size AND hide via multiple selectors.    */
[data-testid="stExpander"] summary,
[data-testid="stExpander"] details > summary {
    display: flex !important;
    align-items: center !important;
    padding: 8px 12px !important;
    list-style: none !important;
    cursor: pointer !important;
    overflow: hidden !important;
}
/* Kill the Material Icons ligature text node entirely */
[data-testid="stExpander"] summary span,
[data-testid="stExpander"] summary > div > span,
[data-testid="stExpander"] summary .material-icons,
[data-testid="stExpander"] summary [data-testid="StyledFullScreenButton"] {
    font-size: 0 !important;
    width: 0 !important;
    height: 0 !important;
    overflow: hidden !important;
    visibility: hidden !important;
    display: none !important;
}
/* But keep the actual label paragraph visible */
[data-testid="stExpander"] summary p,
[data-testid="stExpander"] summary > div > p {
    font-size: 0.64rem !important;
    color: var(--bb-amber2) !important;
    font-family: var(--bb-mono) !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    visibility: visible !important;
    display: block !important;
    width: auto !important;
    overflow: visible !important;
    margin: 0 !important;
}
/* Hide SVG chevrons too */
[data-testid="stExpander"] summary svg {
    display: none !important;
    visibility: hidden !important;
    width: 0 !important;
    height: 0 !important;
}
[data-testid="stExpander"] ::-webkit-details-marker { display: none !important; }
[data-testid="stExpanderDetails"] { background: var(--bb-bg) !important;
    border-top: 1px solid var(--bb-border) !important; }

[data-testid="stAlert"] { background: var(--bb-surface) !important;
    border: 1px solid var(--bb-border) !important; border-radius: 0 !important; }
[data-testid="stAlert"] p { color: var(--bb-white) !important; font-size: 0.68rem !important; }

[data-testid="stCaptionContainer"] p, small { color: var(--bb-muted) !important; font-size: 0.58rem !important; }

.stTabs [data-baseweb="tab-list"] { background: var(--bb-surface) !important;
    border-bottom: 1px solid var(--bb-border) !important; gap: 0 !important; }
.stTabs [data-baseweb="tab"] { background: transparent !important; color: var(--bb-muted) !important;
    font-size: 0.62rem !important; letter-spacing: 0.1em !important; text-transform: uppercase !important;
    border-radius: 0 !important; border-right: 1px solid var(--bb-border) !important; padding: 7px 12px !important; }
.stTabs [aria-selected="true"] { background: #1a1400 !important; color: var(--bb-amber) !important;
    border-bottom: 2px solid var(--bb-amber) !important; }

hr, [data-testid="stDivider"] hr { border-color: var(--bb-border) !important; }
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] { display: none !important; }
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-thumb { background: var(--bb-border); }
::-webkit-scrollbar-thumb:hover { background: var(--bb-amber); }

/* ── Blinking animation ── */
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.3} }
@keyframes pulse { 0%{box-shadow:0 0 0 0 rgba(255,140,0,.4)} 70%{box-shadow:0 0 0 6px rgba(255,140,0,0)} 100%{box-shadow:0 0 0 0 rgba(255,140,0,0)} }
@keyframes slideIn { from{opacity:0;transform:translateY(-4px)} to{opacity:1;transform:translateY(0)} }
@keyframes scanline { 0%{top:-10%} 100%{top:110%} }

.blink { animation: blink 1.4s ease-in-out infinite; }
.pulse-dot { display:inline-block; width:7px; height:7px; border-radius:50%;
    background:var(--bb-green); animation: pulse 2s infinite; margin-right:6px; }
.live-badge { display:inline-flex; align-items:center; gap:5px;
    background:#001a0a; border:1px solid var(--bb-green); color:var(--bb-green);
    font-size:0.55rem; letter-spacing:0.12em; padding:2px 8px; }

/* Ticker tape */
@keyframes ticker { from{transform:translateX(0)} to{transform:translateX(-50%)} }
.ticker-wrap { overflow:hidden; background:#0d0d0d; border-top:1px solid var(--bb-border);
    border-bottom:1px solid var(--bb-border); padding:5px 0; }
.ticker-inner { display:inline-flex; animation:ticker 60s linear infinite; white-space:nowrap; }
.ticker-inner:hover { animation-play-state:paused; }
.tick-item { padding:0 32px; font-size:0.62rem; font-family:var(--bb-mono); color:var(--bb-muted);
    border-right:1px solid #222; }
.tick-up { color:var(--bb-green) !important; }
.tick-dn { color:var(--bb-red) !important; }

/* Risk gauge */
.risk-bar-wrap { height:8px; background:#1a1a1a; border-radius:0; overflow:hidden; }
.risk-bar-fill { height:100%; transition:width .6s ease; }

/* Probability bar (Polymarket style) */
.prob-bar { height:6px; background:#1a1a1a; border-radius:0; overflow:hidden; margin:4px 0; }
.prob-fill-yes { height:100%; background:var(--bb-green); }
.prob-fill-no  { height:100%; background:var(--bb-red); }

/* Panel card */
.panel-card { background:var(--bb-surface); border:1px solid var(--bb-border);
    padding:12px 14px; font-family:var(--bb-mono); }
.panel-title { color:var(--bb-amber); font-size:0.58rem; font-weight:700;
    letter-spacing:0.14em; text-transform:uppercase; margin-bottom:8px;
    border-bottom:1px solid var(--bb-border); padding-bottom:5px; }

/* Scanline overlay (cosmetic) */
.scanline-wrap { position:relative; overflow:hidden; }
.scanline-wrap::after { content:''; position:absolute; left:0; right:0; height:2px;
    background:rgba(255,140,0,0.04); animation:scanline 4s linear infinite; pointer-events:none; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# UPSTOX CREDENTIALS — hardcoded from auth.json
# ============================================================
UPSTOX_CLIENT_ID     = "e720544b-52d6-4f92-941a-9f2fecb1ec72"
UPSTOX_CLIENT_SECRET = "eujrsvhzju"
UPSTOX_REDIRECT_URI  = "http://127.0.0.1"
UPSTOX_AUTH_URL      = "https://api.upstox.com/v2/login/authorization/dialog"
UPSTOX_TOKEN_URL     = "https://api.upstox.com/v2/login/authorization/token"
TOKEN_FILE           = ".upstox_token_scanner"
AUTH_CODE_FILE       = ".upstox_auth_code"   # temp file written by local redirect server

# ============================================================
# AUTO-LOGIN: local HTTP server to catch redirect code
# ============================================================
import socketserver, http.server, webbrowser

class _OAuthHandler(http.server.BaseHTTPRequestHandler):
    """Catches GET /?code=... from Upstox redirect and saves the code."""
    def do_GET(self):
        try:
            parsed = urllib.parse.urlparse(self.path)
            params = urllib.parse.parse_qs(parsed.query)
            code   = params.get("code", [""])[0]
            if code:
                with open(AUTH_CODE_FILE, "w") as f: f.write(code)
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(b"""<html><body style='background:#0a0a0a;color:#00d084;
font-family:monospace;text-align:center;padding-top:80px;font-size:1.2rem;'>
<b>&#10003; MONARCH PRO</b><br><span style='color:#888;font-size:.9rem;'>
Auth code received. Return to the dashboard.</span></body></html>""")
        except: pass
    def log_message(self, *args): pass

def _start_redirect_server():
    """
    Tries port 80 first (matches Upstox redirect URI http://127.0.0.1).
    On Windows, port 80 requires admin rights — if it fails the sidebar
    shows a fallback paste box so the user can copy the URL manually.
    """
    try:
        socketserver.TCPServer.allow_reuse_address = True
        with socketserver.TCPServer(("127.0.0.1", 80), _OAuthHandler) as srv:
            srv.handle_request()
    except PermissionError:
        # Port 80 blocked — write a sentinel so sidebar knows
        try:
            with open(AUTH_CODE_FILE, "w") as f: f.write("__PORT80_BLOCKED__")
        except: pass
    except Exception:
        pass

def upstox_get_access_token(client_id, client_secret, redirect_uri, code):
    try:
        r = requests.post(UPSTOX_TOKEN_URL,
            data={"code": code, "client_id": client_id,
                  "client_secret": client_secret,
                  "redirect_uri": redirect_uri,
                  "grant_type": "authorization_code"},
            headers={"accept": "application/json",
                     "Content-Type": "application/x-www-form-urlencoded"},
            timeout=15)
        if r.status_code == 200:
            return True, r.json().get("access_token",""), r.json()
        return False, "", r.text
    except Exception as e:
        return False, "", str(e)

def save_token(token):
    st.session_state.upstox_token = token
    try:
        with open(TOKEN_FILE, "w") as f: f.write(token)
    except: pass

# ── Load existing token on startup ──
if "home_token_loaded" not in st.session_state:
    if os.path.exists(TOKEN_FILE):
        try:
            with open(TOKEN_FILE) as f:
                st.session_state.upstox_token = f.read().strip()
        except:
            st.session_state.upstox_token = ""
    else:
        st.session_state.upstox_token = ""
    st.session_state.home_token_loaded = True

# ── Session state defaults ──
for k, v in [("ux_step","idle"), ("ux_server_started", False)]:
    if k not in st.session_state: st.session_state[k] = v

# ============================================================
# MARKET DATA — yfinance
# ============================================================
@st.cache_data(ttl=120, show_spinner=False)
def fetch_market_data():
    """Fetch global macro data from yfinance. Returns dict of DataFrames."""
    tickers = {
        # Equity indices
        "NIFTY":        "^NSEI",
        "BANKNIFTY":    "^NSEBANK",
        "SENSEX":       "^BSESN",
        "SPX":          "^GSPC",
        "NDX":          "^NDX",
        "DOW":          "^DJI",
        "DAX":          "^GDAXI",
        "FTSE":         "^FTSE",
        "NIKKEI":       "^N225",
        "SHANGHAI":     "000001.SS",
        "HSI":          "^HSI",
        # Bonds / Yields
        "US10Y":        "^TNX",
        "US2Y":         "^IRX",
        "IN10Y":        "IN10Y.NS",   # India 10Y (may not exist)
        # Commodities
        "GOLD":         "GC=F",
        "SILVER":       "SI=F",
        "CRUDEOIL":     "CL=F",
        "NATGAS":       "NG=F",
        "COPPER":       "HG=F",
        # FX
        "DXY":          "DX-Y.NYB",
        "EURUSD":       "EURUSD=X",
        "USDINR":       "INR=X",
        "USDJPY":       "JPY=X",
        "GBPUSD":       "GBPUSD=X",
        # Volatility
        "VIX":          "^VIX",
        "VIXINDIA":     "^INDIAVIX",
        # Crypto
        "BTC":          "BTC-USD",
        "ETH":          "ETH-USD",
    }
    results = {}
    symbols_list = list(tickers.values())
    try:
        raw = yf.download(symbols_list, period="5d", interval="1d",
                          progress=False, auto_adjust=True, group_by="ticker")
        for name, sym in tickers.items():
            try:
                if sym in raw.columns.get_level_values(0):
                    df = raw[sym].dropna()
                    if not df.empty:
                        last  = float(df["Close"].iloc[-1])
                        prev  = float(df["Close"].iloc[-2]) if len(df) > 1 else last
                        chg   = last - prev
                        pct   = (chg / prev * 100) if prev != 0 else 0
                        results[name] = {"last": last, "prev": prev, "chg": chg, "pct": pct,
                                         "high": float(df["High"].iloc[-1]),
                                         "low":  float(df["Low"].iloc[-1]),
                                         "series": list(df["Close"].tail(20))}
            except: pass
    except: pass
    return results

@st.cache_data(ttl=300, show_spinner=False)
def fetch_crypto_sentiment():
    """Fear & Greed Index from alternative.me"""
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=7", timeout=8)
        if r.status_code == 200:
            data = r.json().get("data", [])
            return [{"value": int(d["value"]), "label": d["value_classification"],
                     "date": d["timestamp"]} for d in data]
    except: pass
    return []

@st.cache_data(ttl=600, show_spinner=False)
def fetch_economic_calendar():
    """Lightweight economic event proxy from public RSS."""
    events = []
    try:
        import feedparser
        feed = feedparser.parse("https://rss.investing.com/rss/economic-calendar")
        for e in feed.entries[:8]:
            events.append({"title": e.get("title",""), "date": e.get("published","")})
    except: pass
    return events

# ============================================================
# RISK-ON / RISK-OFF COMPOSITE
# ============================================================
def compute_risk_score(md):
    """
    Composite Global Risk Score  -100 (extreme risk-off) → +100 (extreme risk-on)
    Components (total weight = 100):
      VIX               20pts  — low VIX = risk-on
      SPX trend         15pts  — positive = risk-on
      DXY               10pts  — falling $ = risk-on (EM bullish)
      GOLD              10pts  — rising gold = risk-off
      US10Y yield        8pts  — rising yield = risk-off (for bonds), risk-on for growth
      Copper            10pts  — rising copper = risk-on (growth proxy)
      BTC               10pts  — rising BTC = risk-on
      NIFTY trend        7pts  — Indian market sentiment
      CRUDEOIL          10pts  — moderate crude = risk-on
    """
    score = 0.0

    def s(name, lo_bad, lo_neut, hi_neut, hi_bad, direction=1):
        """Linear score: direction=+1 if higher=bullish, -1 if higher=bearish"""
        if name not in md: return 0
        pct = md[name]["pct"]
        if direction == 1:
            if pct >= hi_bad:   return  1.0
            if pct >= hi_neut:  return  pct / max(hi_bad, 0.01)
            if pct <= lo_bad:   return -1.0
            if pct <= lo_neut:  return  pct / max(abs(lo_bad), 0.01)
            return pct / 1.5
        else:
            return -s(name, lo_bad, lo_neut, hi_neut, hi_bad, direction=1)

    def vix_score():
        if "VIX" not in md: return 0
        vix = md["VIX"]["last"]
        if vix < 13:  return  1.0
        if vix < 18:  return  0.5
        if vix < 24:  return  0.0
        if vix < 32:  return -0.5
        return -1.0

    # VIX: 20pts
    score += vix_score() * 20

    # SPX: 15pts — rising = risk-on
    score += max(-1, min(1, (md.get("SPX",{}).get("pct",0) / 1.5))) * 15

    # DXY: 10pts — falling = risk-on (money flowing into risk assets)
    score += max(-1, min(1, (-md.get("DXY",{}).get("pct",0) / 0.8))) * 10

    # GOLD: 10pts — rising gold = risk-off
    score += max(-1, min(1, (-md.get("GOLD",{}).get("pct",0) / 1.0))) * 10

    # Copper: 10pts — rising = risk-on (growth proxy)
    score += max(-1, min(1, (md.get("COPPER",{}).get("pct",0) / 1.0))) * 10

    # BTC: 10pts
    score += max(-1, min(1, (md.get("BTC",{}).get("pct",0) / 3.0))) * 10

    # NIFTY: 7pts
    score += max(-1, min(1, (md.get("NIFTY",{}).get("pct",0) / 1.0))) * 7

    # US10Y: 8pts — rising yield moderate: good = growth; extreme = flight to safety
    tnx = md.get("US10Y",{}).get("pct",0)
    score += max(-1, min(1, tnx / 2.0)) * 8  # moderate rise = slight risk-on

    # Crude: 10pts — moderate rise = growth, extreme rise = inflationary risk
    cl = md.get("CRUDEOIL",{}).get("pct",0)
    if cl > 3:   score -= 5   # too hot = risk-off
    elif cl > 0: score += cl / 0.3 * 10 * 0.1  # moderate = risk-on
    elif cl < -3: score -= 5

    return round(max(-100, min(100, score)), 1)

def risk_label(score):
    if score >= 40:  return "RISK ON",  "#00d084"
    if score >= 15:  return "MILD RISK ON", "#7dca84"
    if score >= -15: return "NEUTRAL",   "#ffb347"
    if score >= -40: return "MILD RISK OFF", "#ff7777"
    return "RISK OFF", "#ff3b3b"

# ============================================================
# HEADER BAR
# ============================================================
now_str = datetime.now().strftime("%Y-%m-%d  %H:%M:%S  IST")
token_ok = bool(st.session_state.upstox_token)
tok_html = ('<span class="live-badge"><span class="pulse-dot"></span>UPSTOX CONNECTED</span>'
            if token_ok else
            '<span style="color:#ff3b3b;font-size:.58rem;border:1px solid #ff3b3b;'
            'padding:2px 8px;">⚠ UPSTOX NOT CONNECTED</span>')

st.markdown(f"""
<div style="background:#ff8c00;color:#000;font-family:'IBM Plex Mono',monospace;
font-size:.60rem;font-weight:700;letter-spacing:.18em;padding:5px 16px;
display:flex;justify-content:space-between;align-items:center;margin-bottom:0;">
  <span>◼ MONARCH GLOBAL RISK INTELLIGENCE</span>
  <span style="font-size:.55rem;font-weight:400;">{now_str}</span>
</div>
<div style="background:#0d0d0d;border-bottom:1px solid #2a2a2a;
padding:4px 16px;display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
  <span style="color:#888;font-size:.55rem;letter-spacing:.08em;">
    GLOBAL MACRO · RISK MONITOR · VOLATILITY · FX · BONDS · COMMODITIES · CRYPTO
  </span>
  <span>{tok_html}</span>
</div>
""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR — Upstox Auto-Login (one-click)
# ============================================================
with st.sidebar:
    st.markdown("""
<div style="color:#ff8c00;font-size:.72rem;font-weight:700;letter-spacing:.14em;
padding:10px 0 6px;border-bottom:1px solid #2a2a2a;margin-bottom:10px;">
◼ MONARCH PRO
</div>""", unsafe_allow_html=True)

    token_ok = bool(st.session_state.upstox_token)

    # ── Already connected ──
    if token_ok:
        st.markdown(f"""
<div style="background:#001a0a;border:1px solid #00d084;border-left:3px solid #00d084;
padding:10px 12px;font-family:'IBM Plex Mono',monospace;margin-bottom:10px;">
  <div style="color:#00d084;font-size:.68rem;font-weight:700;letter-spacing:.1em;margin-bottom:4px;">
    ✔ UPSTOX CONNECTED
  </div>
  <div style="color:#888;font-size:.56rem;">
    Token: <span style="color:#aaa;">{st.session_state.upstox_token[:20]}…</span>
  </div>
  <div style="color:#666;font-size:.52rem;margin-top:3px;">Auto-refreshes daily on re-login</div>
</div>""", unsafe_allow_html=True)
        if st.button("↺  Disconnect / Re-login", key="re_login_btn", use_container_width=True):
            st.session_state.ux_step = "idle"
            st.session_state.ux_server_started = False
            st.session_state.upstox_token = ""
            try:
                if os.path.exists(TOKEN_FILE): os.remove(TOKEN_FILE)
                if os.path.exists(AUTH_CODE_FILE): os.remove(AUTH_CODE_FILE)
            except: pass
            st.rerun()

    # ── Auto-login flow ──
    else:
        # Check if the redirect server already caught a code
        if os.path.exists(AUTH_CODE_FILE):
            try:
                with open(AUTH_CODE_FILE) as f:
                    caught_code = f.read().strip()
            except:
                caught_code = ""

            if caught_code == "__PORT80_BLOCKED__":
                try: os.remove(AUTH_CODE_FILE)
                except: pass
                st.session_state.ux_step = "port_blocked"
                st.session_state.ux_server_started = False
                st.rerun()
            elif caught_code:
                st.session_state.ux_step = "exchanging"
                with st.spinner("🔑 Exchanging code for token…"):
                    ok, tok, resp = upstox_get_access_token(
                        UPSTOX_CLIENT_ID, UPSTOX_CLIENT_SECRET,
                        UPSTOX_REDIRECT_URI, caught_code)
                try:
                    os.remove(AUTH_CODE_FILE)
                except: pass
                if ok and tok:
                    save_token(tok)
                    st.session_state.ux_step = "idle"
                    st.session_state.ux_server_started = False
                    st.success("✔ Connected to Upstox!")
                    time.sleep(0.8)
                    st.rerun()
                else:
                    st.session_state.ux_step = "idle"
                    st.session_state.ux_server_started = False
                    err = str(resp)[:300]
                    st.markdown(f"""
<div style="background:#1a0000;border:1px solid #ff3b3b;padding:8px 10px;
font-size:.58rem;color:#ff3b3b;margin-top:6px;">
  <b>Token exchange failed</b><br/>
  <span style="color:#aaa;font-size:.54rem;">{err}</span><br/><br/>
  <span style="color:#888;">Check Client Secret is correct, then try again.</span>
</div>""", unsafe_allow_html=True)

        # Show the one-click connect button
        if not token_ok:
            st.markdown("""
<div style="background:#0a0800;border:1px solid #ff8c00;padding:10px 12px;margin-bottom:10px;">
  <div style="color:#ffb347;font-size:.60rem;font-weight:700;letter-spacing:.1em;margin-bottom:6px;">
    ⚡ ONE-CLICK UPSTOX LOGIN
  </div>
  <div style="color:#888;font-size:.55rem;line-height:1.8;">
    1. Click <b style="color:#ff8c00;">CONNECT UPSTOX</b> below<br/>
    2. Browser opens Upstox login page<br/>
    3. Login with phone + OTP<br/>
    4. Come back here — token auto-fetched ✔
  </div>
</div>""", unsafe_allow_html=True)

            if st.button("⚡  CONNECT UPSTOX", key="auto_login_btn",
                         use_container_width=True, type="primary"):
                # Start local redirect catcher in background thread
                if not st.session_state.ux_server_started:
                    t = threading.Thread(target=_start_redirect_server, daemon=True)
                    t.start()
                    st.session_state.ux_server_started = True
                # Clean any stale code file
                try:
                    if os.path.exists(AUTH_CODE_FILE): os.remove(AUTH_CODE_FILE)
                except: pass
                # Build auth URL and open browser
                params   = {"response_type": "code",
                            "client_id":     UPSTOX_CLIENT_ID,
                            "redirect_uri":  UPSTOX_REDIRECT_URI}
                auth_url = UPSTOX_AUTH_URL + "?" + urllib.parse.urlencode(params)
                webbrowser.open(auth_url)
                st.session_state.ux_step = "waiting"
                st.rerun()

            if st.session_state.ux_step == "port_blocked":
                st.markdown("""
<div style="background:#1a0800;border:1px solid #ff8c00;padding:8px 10px;
font-size:.58rem;color:#ff8c00;margin-top:8px;">
  <b>⚠ Port 80 blocked</b> — Run as Administrator, or paste the redirect URL below after login.
</div>""", unsafe_allow_html=True)

            if st.session_state.ux_step == "waiting":
                st.markdown("""
<div style="background:#001520;border:1px solid #1e90ff;padding:8px 10px;
font-size:.58rem;color:#1e90ff;margin-top:8px;text-align:center;">
  <span class="blink">⏳ Waiting for Upstox login…</span><br/>
  <span style="color:#666;font-size:.52rem;">Complete login in the browser tab — token will auto-load</span>
</div>""", unsafe_allow_html=True)
                time.sleep(2)
                st.rerun()

            # Fallback: manual paste
            st.markdown('<div style="border-top:1px solid #1a1a1a;margin:12px 0 8px;"></div>',
                        unsafe_allow_html=True)
            st.markdown('<div style="color:#555;font-size:.54rem;margin-bottom:4px;">'
                        'If auto-capture fails, paste the redirect URL here:</div>',
                        unsafe_allow_html=True)
            manual_url = st.text_input("Redirect URL (fallback)", key="manual_url",
                                       placeholder="http://127.0.0.1/?code=AbCdEf…")
            if st.button("✔ Use This URL", key="manual_url_btn", use_container_width=True):
                if manual_url.strip():
                    try:
                        parsed = urllib.parse.urlparse(manual_url.strip())
                        params = urllib.parse.parse_qs(parsed.query)
                        code   = params.get("code", [""])[0]
                    except:
                        code = ""
                    if not code:
                        m = re.search(r'[?&]code=([^&\s]+)', manual_url)
                        code = m.group(1) if m else manual_url.strip()
                    if code:
                        with st.spinner("Exchanging code…"):
                            ok, tok, resp = upstox_get_access_token(
                                UPSTOX_CLIENT_ID, UPSTOX_CLIENT_SECRET,
                                UPSTOX_REDIRECT_URI, code)
                        if ok and tok:
                            save_token(tok)
                            st.session_state.ux_step = "idle"
                            st.rerun()
                        else:
                            st.error(f"Failed: {str(resp)[:200]}")
                    else:
                        st.error("Could not extract code from URL.")
                else:
                    st.warning("Paste the redirect URL first.")

            # Direct token paste
            st.markdown('<div style="color:#555;font-size:.54rem;margin-bottom:4px;margin-top:8px;">'
                        'Or paste access token directly:</div>', unsafe_allow_html=True)
            direct_tok = st.text_input("Access Token", key="direct_tok",
                                       type="password", placeholder="eyJ0eXAiOiJKV1Q…")
            if st.button("✔  Use Token", key="direct_tok_btn", use_container_width=True):
                if direct_tok.strip():
                    save_token(direct_tok.strip())
                    st.session_state.ux_step = "idle"
                    st.success("Token saved!")
                    st.rerun()
                else:
                    st.warning("Paste your token first.")

    st.markdown('<div style="border-top:1px solid #2a2a2a;margin:12px 0 8px;"></div>',
                unsafe_allow_html=True)
    st.markdown('<div style="color:#ff8c00;font-size:.60rem;font-weight:700;'
                'letter-spacing:.1em;margin-bottom:6px;">⚙ DASHBOARD</div>',
                unsafe_allow_html=True)
    auto_refresh = st.checkbox("Auto-refresh (60s)", value=False, key="auto_refresh")
    if st.button("↺  Refresh Data", key="refresh_btn", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.markdown('<div style="border-top:1px solid #2a2a2a;margin:12px 0 8px;"></div>',
                unsafe_allow_html=True)
    st.markdown("""
<div style="color:#888;font-size:.56rem;line-height:2;">
NAVIGATE<br/>
<span style="color:#e8e8e8;">→ Home</span><br/>
<span style="color:#666;">→ Screener Pro</span><br/>
<span style="color:#666;">→ Options Intel</span><br/>
<span style="color:#666;">→ Fundamental Research</span>
</div>""", unsafe_allow_html=True)


# ============================================================
# FETCH DATA
# ============================================================
with st.spinner("Fetching global market data…"):
    md = fetch_market_data()
    fng = fetch_crypto_sentiment()

if not md:
    st.error("Could not fetch market data. Check internet connection.")
    st.stop()

# Compute composite risk score
risk_score = compute_risk_score(md)
rl, rc = risk_label(risk_score)

# ============================================================
# AUTO-REFRESH SCRIPT
# ============================================================
if auto_refresh:
    st.markdown('<meta http-equiv="refresh" content="60">', unsafe_allow_html=True)

# ============================================================
# TICKER TAPE
# ============================================================
def fmt_tick(name, d):
    sign = "▲" if d["pct"] >= 0 else "▼"
    cls  = "tick-up" if d["pct"] >= 0 else "tick-dn"
    val  = f"{d['last']:,.2f}" if d['last'] < 1000 else f"{d['last']:,.0f}"
    return f'<span class="tick-item"><span class="{cls}">{name}: {val} {sign}{abs(d["pct"]):.2f}%</span></span>'

tick_order = ["NIFTY","BANKNIFTY","SENSEX","SPX","NDX","NIKKEI","DAX","FTSE","HSI",
              "VIX","VIXINDIA","GOLD","SILVER","CRUDEOIL","COPPER","DXY","EURUSD",
              "USDINR","USDJPY","BTC","ETH","US10Y","US2Y"]
tick_html = "".join(fmt_tick(n, md[n]) for n in tick_order if n in md)
# Duplicate for seamless loop
tick_html = tick_html * 2

st.markdown(f"""
<div class="ticker-wrap">
  <div class="ticker-inner">{tick_html}</div>
</div>
""", unsafe_allow_html=True)

st.markdown("<div style='margin-bottom:10px;'></div>", unsafe_allow_html=True)

# ============================================================
# ROW 1 — Risk Score + Key Metrics
# ============================================================
r1c1, r1c2 = st.columns([2, 5])

with r1c1:
    gauge_w = int((risk_score + 100) / 200 * 100)
    gauge_c = rc

    # Gradient risk bar
    vix_val  = md.get("VIX",{}).get("last", 20)
    vix_c    = "#00d084" if vix_val < 16 else "#ffb347" if vix_val < 26 else "#ff3b3b"
    fng_val  = fng[0]["value"] if fng else 50
    fng_lbl  = fng[0]["label"] if fng else "—"
    fng_c    = "#00d084" if fng_val > 60 else "#ff3b3b" if fng_val < 40 else "#ffb347"

    st.markdown(f"""
<div class="scanline-wrap" style="background:var(--bb-surface);border:1px solid var(--bb-border);
border-left:4px solid {gauge_c};padding:16px 18px;font-family:'IBM Plex Mono',monospace;height:100%;">
  <div style="color:#555;font-size:.54rem;letter-spacing:.14em;margin-bottom:4px;">
    GLOBAL RISK MONITOR
  </div>
  <div style="color:{gauge_c};font-size:1.6rem;font-weight:700;letter-spacing:.06em;margin-bottom:2px;">
    {rl}
  </div>
  <div style="color:#888;font-size:.62rem;margin-bottom:12px;">
    Composite Score: <span style="color:{gauge_c};font-weight:700;">{risk_score:+.1f}</span> / 100
  </div>

  <div style="color:#444;font-size:.52rem;letter-spacing:.1em;margin-bottom:3px;">
    RISK SPECTRUM ◄ RISK-OFF ─────────── RISK-ON ►
  </div>
  <div class="risk-bar-wrap" style="margin-bottom:12px;">
    <div class="risk-bar-fill" style="width:{gauge_w}%;
    background:linear-gradient(90deg,#ff3b3b,#ffb347,#00d084);"></div>
  </div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:4px;">
    <div style="background:#0d0d0d;border:1px solid #1a1a1a;padding:7px 10px;">
      <div style="color:#555;font-size:.52rem;letter-spacing:.1em;">VIX</div>
      <div style="color:{vix_c};font-size:1.1rem;font-weight:700;">{vix_val:.1f}</div>
      <div style="color:#444;font-size:.54rem;">{'FEAR' if vix_val>26 else 'NORMAL' if vix_val>16 else 'COMPLACENT'}</div>
    </div>
    <div style="background:#0d0d0d;border:1px solid #1a1a1a;padding:7px 10px;">
      <div style="color:#555;font-size:.52rem;letter-spacing:.1em;">CRYPTO F&G</div>
      <div style="color:{fng_c};font-size:1.1rem;font-weight:700;">{fng_val}</div>
      <div style="color:#444;font-size:.54rem;">{fng_lbl.upper()}</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

with r1c2:
    # Key global metrics blinking table
    metrics_config = [
        ("NIFTY 50",     "NIFTY"),
        ("BANK NIFTY",   "BANKNIFTY"),
        ("S&P 500",      "SPX"),
        ("NASDAQ 100",   "NDX"),
        ("NIKKEI 225",   "NIKKEI"),
        ("DAX",          "DAX"),
        ("GOLD",         "GOLD"),
        ("CRUDE OIL",    "CRUDEOIL"),
        ("DXY",          "DXY"),
        ("USD/INR",      "USDINR"),
        ("USD/JPY",      "USDJPY"),
        ("EUR/USD",      "EURUSD"),
        ("BTC",          "BTC"),
        ("US 10Y YIELD", "US10Y"),
        ("COPPER",       "COPPER"),
    ]
    rows = []
    for label, key in metrics_config:
        if key in md:
            d = md[key]
            arrow = "▲" if d["pct"] >= 0 else "▼"
            color = "#00d084" if d["pct"] >= 0 else "#ff3b3b"
            val   = f"{d['last']:>12,.2f}" if d['last'] < 1000 else f"{d['last']:>12,.1f}"
            rows.append({
                "Instrument": label,
                "Price":      val.strip(),
                "Chg":        f"{d['chg']:+.2f}",
                "Chg%":       f"{arrow} {abs(d['pct']):.2f}%",
                "_color":     color,
            })

    tbl_html = """
<table style="width:100%;border-collapse:collapse;font-family:'IBM Plex Mono',monospace;font-size:.62rem;">
<thead>
<tr style="background:#1a1200;border-bottom:1px solid #ff8c00;">
  <th style="color:#ff8c00;padding:5px 10px;text-align:left;letter-spacing:.1em;font-size:.56rem;">INSTRUMENT</th>
  <th style="color:#ff8c00;padding:5px 10px;text-align:right;letter-spacing:.1em;font-size:.56rem;">PRICE</th>
  <th style="color:#ff8c00;padding:5px 10px;text-align:right;letter-spacing:.1em;font-size:.56rem;">CHG</th>
  <th style="color:#ff8c00;padding:5px 10px;text-align:right;letter-spacing:.1em;font-size:.56rem;">CHG %</th>
</tr>
</thead>
<tbody>"""
    for i, row in enumerate(rows):
        bg = "#111" if i % 2 == 0 else "#0d0d0d"
        tbl_html += f"""
<tr style="background:{bg};border-bottom:1px solid #1a1a1a;">
  <td style="color:#e8e8e8;padding:5px 10px;">{row['Instrument']}</td>
  <td style="color:#e8e8e8;padding:5px 10px;text-align:right;">{row['Price']}</td>
  <td style="color:{row['_color']};padding:5px 10px;text-align:right;">{row['Chg']}</td>
  <td style="color:{row['_color']};padding:5px 10px;text-align:right;font-weight:700;">{row['Chg%']}</td>
</tr>"""
    tbl_html += "</tbody></table>"

    st.markdown(f"""
<div style="background:var(--bb-surface);border:1px solid var(--bb-border);
padding:10px;height:100%;">
  <div class="panel-title">GLOBAL MARKETS — LIVE SNAPSHOT</div>
  {tbl_html}
</div>""", unsafe_allow_html=True)

st.markdown("<div style='margin:10px 0;'></div>", unsafe_allow_html=True)

# ============================================================
# ROW 2 — Charts: Equities · Bonds · FX · Commodities
# ============================================================
st.markdown("""
<div style="color:#ff8c00;font-size:.6rem;font-weight:700;letter-spacing:.16em;
padding:5px 0 8px;border-bottom:1px solid #2a2a2a;margin-bottom:10px;">
  ◼ GLOBAL ASSET CLASS PERFORMANCE  (5-DAY TREND)
</div>""", unsafe_allow_html=True)

chart_groups = [
    ("EQUITY INDICES",  ["SPX","NDX","NIFTY","BANKNIFTY","NIKKEI","DAX"]),
    ("FX + DXY",        ["DXY","EURUSD","USDINR","USDJPY","GBPUSD"]),
    ("COMMODITIES",     ["GOLD","SILVER","CRUDEOIL","COPPER","NATGAS"]),
    ("CRYPTO + VIX",    ["BTC","ETH","VIX","VIXINDIA"]),
]

ch_cols = st.columns(4)
for ci, (title, keys) in enumerate(chart_groups):
    with ch_cols[ci]:
        fig = go.Figure()
        for key in keys:
            if key in md and md[key]["series"]:
                s = md[key]["series"]
                base = s[0] if s[0] != 0 else 1
                norm = [(v / base - 1) * 100 for v in s]
                last_pct = norm[-1]
                col = "#00d084" if last_pct >= 0 else "#ff3b3b"
                fig.add_trace(go.Scatter(
                    x=list(range(len(norm))), y=norm,
                    name=key, mode="lines",
                    line=dict(width=1.5, color=col),
                    hovertemplate=f"{key}: %{{y:.2f}}%<extra></extra>"
                ))
        fig.update_layout(
            title=dict(text=title, font=dict(color="#ff8c00", size=9,
                        family="IBM Plex Mono"), x=0),
            height=200, plot_bgcolor="#000", paper_bgcolor="#111",
            font=dict(color="#888", family="IBM Plex Mono", size=8),
            xaxis=dict(showgrid=False, showticklabels=False,
                       zeroline=False, color="#444"),
            yaxis=dict(gridcolor="#1a1a1a", tickformat=".1f",
                       ticksuffix="%", color="#888", zeroline=True,
                       zerolinecolor="#333"),
            legend=dict(orientation="h", y=-0.25, font=dict(size=7, color="#888"),
                        bgcolor="rgba(0,0,0,0)"),
            margin=dict(t=30, b=30, l=35, r=5),
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

st.markdown("<div style='margin:4px 0;'></div>", unsafe_allow_html=True)

# ============================================================
# ROW 3 — Polymarket-style Risk Signals + Yield Spread
# ============================================================
st.markdown("""
<div style="color:#ff8c00;font-size:.6rem;font-weight:700;letter-spacing:.16em;
padding:5px 0 8px;border-bottom:1px solid #2a2a2a;margin-bottom:10px;">
  ◼ RISK SIGNAL DASHBOARD  —  MARKET PROBABILITY INDICATORS
</div>""", unsafe_allow_html=True)

pc1, pc2, pc3 = st.columns([2, 2, 3])

with pc1:
    # Polymarket-style probability cards
    def prob_card(question, yes_pct, note="", yes_col="#00d084", no_col="#ff3b3b"):
        no_pct = 100 - yes_pct
        yes_c  = yes_col if yes_pct >= 50 else "#555"
        no_c   = no_col  if no_pct  >= 50 else "#555"
        return f"""
<div style="background:#0d0d0d;border:1px solid #1a1a1a;border-left:2px solid #ff8c00;
padding:8px 10px;margin-bottom:6px;font-family:'IBM Plex Mono',monospace;">
  <div style="color:#888;font-size:.56rem;letter-spacing:.06em;margin-bottom:5px;">{question}</div>
  <div style="display:flex;justify-content:space-between;margin-bottom:3px;">
    <span style="color:{yes_c};font-size:.68rem;font-weight:700;">YES {yes_pct:.0f}%</span>
    <span style="color:{no_c};font-size:.68rem;">NO {no_pct:.0f}%</span>
  </div>
  <div class="prob-bar"><div class="prob-fill-yes" style="width:{yes_pct}%;"></div></div>
  <div style="color:#444;font-size:.52rem;margin-top:3px;">{note}</div>
</div>"""

    # Compute live probability proxies from market data
    vix_prob_fear     = min(100, max(0, (md.get("VIX",{}).get("last",20) - 12) / 28 * 100))
    spx_bull_prob     = min(100, max(0, 50 + md.get("SPX",{}).get("pct",0) * 8))
    gold_hedge_prob   = min(100, max(0, 50 + md.get("GOLD",{}).get("pct",0) * 6))
    dxy_strength_prob = min(100, max(0, 50 + md.get("DXY",{}).get("pct",0) * 10))
    btc_bull_prob     = min(100, max(0, 50 + md.get("BTC",{}).get("pct",0) * 3))
    india_bull_prob   = min(100, max(0, 50 + md.get("NIFTY",{}).get("pct",0) * 8))

    cards = [
        ("VIX > 20 end of week?",       vix_prob_fear,   f"Current VIX: {md.get('VIX',{}).get('last',0):.1f}"),
        ("S&P 500 bullish today?",       spx_bull_prob,   f"SPX: {md.get('SPX',{}).get('pct',0):+.2f}%"),
        ("Gold as safe haven active?",   gold_hedge_prob, f"Gold: {md.get('GOLD',{}).get('pct',0):+.2f}%"),
        ("USD strengthening?",           dxy_strength_prob, f"DXY: {md.get('DXY',{}).get('pct',0):+.2f}%"),
        ("Bitcoin risk-on?",             btc_bull_prob,   f"BTC: {md.get('BTC',{}).get('pct',0):+.2f}%"),
        ("NIFTY bullish bias?",          india_bull_prob, f"Nifty: {md.get('NIFTY',{}).get('pct',0):+.2f}%"),
    ]

    st.markdown('<div class="panel-title" style="color:#ff8c00;font-size:.58rem;'
                'font-weight:700;letter-spacing:.12em;margin-bottom:8px;">'
                'POLYMARKET-STYLE PROBABILITY SIGNALS</div>',
                unsafe_allow_html=True)
    for q, pct, note in cards:
        st.markdown(prob_card(q, pct, note), unsafe_allow_html=True)

with pc2:
    # India VIX gauge + Fear & Greed history
    india_vix = md.get("VIXINDIA",{}).get("last", 0)
    ivix_c = "#00d084" if india_vix < 15 else "#ffb347" if india_vix < 22 else "#ff3b3b"
    ivix_lbl = "LOW FEAR" if india_vix < 15 else "MODERATE" if india_vix < 22 else "HIGH FEAR"

    st.markdown(f"""
<div class="panel-card" style="margin-bottom:8px;">
  <div class="panel-title">INDIA VIX + VOLATILITY</div>
  <div style="display:flex;align-items:center;gap:16px;margin-bottom:8px;">
    <div>
      <div style="color:#555;font-size:.52rem;letter-spacing:.1em;">INDIA VIX</div>
      <div style="color:{ivix_c};font-size:2.2rem;font-weight:700;line-height:1;">{india_vix:.2f}</div>
      <div style="color:{ivix_c};font-size:.58rem;">{ivix_lbl}</div>
    </div>
    <div style="flex:1;">
      <div style="color:#555;font-size:.52rem;margin-bottom:4px;">VIX LEVEL</div>
      <div class="risk-bar-wrap">
        <div class="risk-bar-fill" style="width:{min(100,india_vix/40*100):.0f}%;
        background:linear-gradient(90deg,#00d084,#ffb347,#ff3b3b);"></div>
      </div>
      <div style="display:flex;justify-content:space-between;font-size:.48rem;color:#444;margin-top:2px;">
        <span>0</span><span>LOW</span><span>HIGH</span><span>40</span>
      </div>
    </div>
  </div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;">
    <div style="background:#0d0d0d;border:1px solid #1a1a1a;padding:6px 8px;">
      <div style="color:#555;font-size:.50rem;">VIX (USA)</div>
      <div style="color:{vix_c};font-size:.85rem;font-weight:700;">{vix_val:.1f}</div>
    </div>
    <div style="background:#0d0d0d;border:1px solid #1a1a1a;padding:6px 8px;">
      <div style="color:#555;font-size:.50rem;">VIX Δ</div>
      <div style="color:{'#ff3b3b' if md.get('VIX',{}).get('pct',0)>0 else '#00d084'};
           font-size:.85rem;font-weight:700;">{md.get('VIX',{}).get('pct',0):+.2f}%</div>
    </div>
  </div>
</div>""", unsafe_allow_html=True)

    # Crypto Fear & Greed 7-day
    if fng:
        fng_vals = [d["value"] for d in fng[:7]][::-1]
        fng_labels = [d["label"] for d in fng[:7]][::-1]
        fng_colors = ["#00d084" if v > 60 else "#ff3b3b" if v < 40 else "#ffb347"
                      for v in fng_vals]
        fig_fng = go.Figure(go.Bar(
            x=list(range(len(fng_vals))), y=fng_vals,
            marker_color=fng_colors, text=[str(v) for v in fng_vals],
            textposition="outside", textfont=dict(size=8, color="#888")
        ))
        fig_fng.add_hline(y=50, line=dict(color="#333", dash="dot"))
        fig_fng.update_layout(
            title=dict(text="CRYPTO FEAR & GREED (7D)", font=dict(color="#ff8c00", size=9), x=0),
            height=160, plot_bgcolor="#000", paper_bgcolor="#111",
            font=dict(color="#888", family="IBM Plex Mono", size=8),
            xaxis=dict(showticklabels=False, showgrid=False),
            yaxis=dict(range=[0,105], showgrid=False, showticklabels=False),
            margin=dict(t=28, b=8, l=0, r=0), showlegend=False
        )
        st.plotly_chart(fig_fng, use_container_width=True, config={"displayModeBar": False})

with pc3:
    # Yield Spread + Equity/Bond ratio charts
    us10y = md.get("US10Y",{}).get("last", 4.2)
    us2y  = md.get("US2Y",{}).get("last", 4.5)
    spread = us10y - us2y
    spread_c = "#ff3b3b" if spread < 0 else "#00d084"
    spread_lbl = "INVERTED (recession signal)" if spread < 0 else "NORMAL" if spread < 0.5 else "STEEP"

    # SPX vs Gold ratio (risk appetite)
    spx_gold_ratio = None
    if "SPX" in md and "GOLD" in md and md["GOLD"]["last"] > 0:
        spx_gold_ratio = md["SPX"]["last"] / md["GOLD"]["last"]

    st.markdown(f"""
<div class="panel-card" style="margin-bottom:8px;">
  <div class="panel-title">YIELD CURVE + BOND MARKET</div>
  <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin-bottom:8px;">
    <div style="background:#0d0d0d;border:1px solid #1a1a1a;padding:8px;">
      <div style="color:#555;font-size:.50rem;letter-spacing:.1em;">US 10Y YIELD</div>
      <div style="color:#ffb347;font-size:1.1rem;font-weight:700;">{us10y:.2f}%</div>
    </div>
    <div style="background:#0d0d0d;border:1px solid #1a1a1a;padding:8px;">
      <div style="color:#555;font-size:.50rem;letter-spacing:.1em;">US 2Y YIELD</div>
      <div style="color:#ffb347;font-size:1.1rem;font-weight:700;">{us2y:.2f}%</div>
    </div>
    <div style="background:#0d0d0d;border:1px solid #1a1a1a;padding:8px;">
      <div style="color:#555;font-size:.50rem;letter-spacing:.1em;">10Y–2Y SPREAD</div>
      <div style="color:{spread_c};font-size:1.1rem;font-weight:700;">{spread:+.2f}%</div>
      <div style="color:{spread_c};font-size:.50rem;">{spread_lbl}</div>
    </div>
  </div>
</div>""", unsafe_allow_html=True)

    # Heatmap of 1-day returns
    hm_keys   = ["SPX","NDX","NIFTY","BANKNIFTY","DAX","NIKKEI","HSI","SENSEX",
                 "GOLD","SILVER","CRUDEOIL","COPPER","BTC","ETH","DXY","USDINR"]
    hm_labels = [k for k in hm_keys if k in md]
    hm_vals   = [md[k]["pct"] for k in hm_labels]

    if hm_vals:
        # 4-wide grid
        ncols = 4
        nrows = math.ceil(len(hm_labels) / ncols)
        z_matrix = []
        text_matrix = []
        for row in range(nrows):
            z_row, t_row = [], []
            for col in range(ncols):
                idx = row * ncols + col
                if idx < len(hm_labels):
                    z_row.append(hm_vals[idx])
                    t_row.append(f"{hm_labels[idx]}<br>{hm_vals[idx]:+.2f}%")
                else:
                    z_row.append(0)
                    t_row.append("")
            z_matrix.append(z_row)
            text_matrix.append(t_row)

        fig_hm = go.Figure(go.Heatmap(
            z=z_matrix, text=text_matrix, texttemplate="%{text}",
            textfont=dict(size=8, family="IBM Plex Mono", color="#fff"),
            colorscale=[[0,"#ff3b3b"],[0.5,"#222222"],[1,"#00d084"]],
            zmid=0, zmin=-3, zmax=3,
            showscale=False, xgap=2, ygap=2,
        ))
        fig_hm.update_layout(
            title=dict(text="DAILY RETURN HEATMAP", font=dict(color="#ff8c00",size=9), x=0),
            height=220, plot_bgcolor="#000", paper_bgcolor="#111",
            font=dict(color="#888", family="IBM Plex Mono", size=8),
            xaxis=dict(showticklabels=False, showgrid=False),
            yaxis=dict(showticklabels=False, showgrid=False),
            margin=dict(t=28,b=5,l=5,r=5)
        )
        st.plotly_chart(fig_hm, use_container_width=True, config={"displayModeBar": False})

st.markdown("<div style='margin:6px 0;'></div>", unsafe_allow_html=True)

# ============================================================
# ROW 4 — India Macro + Sector Performance
# ============================================================
st.markdown("""
<div style="color:#ff8c00;font-size:.6rem;font-weight:700;letter-spacing:.16em;
padding:5px 0 8px;border-bottom:1px solid #2a2a2a;margin-bottom:10px;">
  ◼ INDIA MACRO DASHBOARD
</div>""", unsafe_allow_html=True)

india_tickers_map = {
    "NIFTY IT":     "^CNXIT",
    "NIFTY BANK":   "^NSEBANK",
    "NIFTY PHARMA": "^CNXPHARMA",
    "NIFTY AUTO":   "^CNXAUTO",
    "NIFTY FMCG":   "^CNXFMCG",
    "NIFTY METAL":  "^CNXMETAL",
    "NIFTY ENERGY": "^CNXENERGY",
    "NIFTY REALTY": "^CNXREALTY",
}

@st.cache_data(ttl=300, show_spinner=False)
def fetch_india_sectors():
    try:
        tlist = list(india_tickers_map.values())
        raw = yf.download(tlist, period="5d", interval="1d",
                          progress=False, auto_adjust=True, group_by="ticker")
        res = {}
        for name, sym in india_tickers_map.items():
            try:
                if sym in raw.columns.get_level_values(0):
                    df = raw[sym].dropna()
                    if len(df) >= 2:
                        last = float(df["Close"].iloc[-1])
                        prev = float(df["Close"].iloc[-2])
                        pct  = (last / prev - 1) * 100
                        res[name] = {"last": last, "pct": pct,
                                     "series": list(df["Close"])}
            except: pass
        return res
    except: return {}

ind_sec = fetch_india_sectors()

i4c1, i4c2 = st.columns([3, 4])

with i4c1:
    # India key metrics
    i_metrics = [
        ("NIFTY 50",    "NIFTY"),
        ("BANK NIFTY",  "BANKNIFTY"),
        ("SENSEX",      "SENSEX"),
        ("INDIA VIX",   "VIXINDIA"),
        ("USD/INR",     "USDINR"),
    ]
    ind_rows_html = ""
    for i, (lbl, key) in enumerate(i_metrics):
        if key in md:
            d  = md[key]
            c  = "#00d084" if d["pct"] >= 0 else "#ff3b3b"
            arr = "▲" if d["pct"] >= 0 else "▼"
            val = f"{d['last']:,.2f}" if d['last'] < 1000 else f"{d['last']:,.0f}"
            bg = "#111" if i%2==0 else "#0d0d0d"
            ind_rows_html += f"""
<tr style="background:{bg};border-bottom:1px solid #1a1a1a;">
  <td style="color:#e8e8e8;padding:5px 10px;font-size:.62rem;">{lbl}</td>
  <td style="color:#e8e8e8;padding:5px 10px;text-align:right;font-size:.62rem;">{val}</td>
  <td style="color:{c};padding:5px 10px;text-align:right;font-size:.62rem;font-weight:700;">
    {arr} {abs(d['pct']):.2f}%</td>
</tr>"""

    st.markdown(f"""
<div class="panel-card">
  <div class="panel-title">INDIA INDICES</div>
  <table style="width:100%;border-collapse:collapse;">
  <thead><tr style="background:#1a1200;border-bottom:1px solid #ff8c00;">
    <th style="color:#ff8c00;padding:5px 10px;text-align:left;font-size:.54rem;letter-spacing:.1em;">INDEX</th>
    <th style="color:#ff8c00;padding:5px 10px;text-align:right;font-size:.54rem;letter-spacing:.1em;">PRICE</th>
    <th style="color:#ff8c00;padding:5px 10px;text-align:right;font-size:.54rem;letter-spacing:.1em;">CHG %</th>
  </tr></thead>
  <tbody>{ind_rows_html}</tbody>
  </table>
</div>""", unsafe_allow_html=True)

with i4c2:
    if ind_sec:
        sec_names = list(ind_sec.keys())
        sec_pcts  = [ind_sec[n]["pct"] for n in sec_names]
        sec_colors = ["#00d084" if p >= 0 else "#ff3b3b" for p in sec_pcts]

        fig_sec = go.Figure(go.Bar(
            x=sec_pcts, y=sec_names, orientation="h",
            marker_color=sec_colors, opacity=0.85,
            text=[f"{p:+.2f}%" for p in sec_pcts],
            textposition="outside",
            textfont=dict(size=8, color="#888")
        ))
        fig_sec.add_vline(x=0, line=dict(color="#333", width=1))
        fig_sec.update_layout(
            title=dict(text="INDIA SECTORAL PERFORMANCE (1D)", font=dict(color="#ff8c00",size=9), x=0),
            height=260, plot_bgcolor="#000", paper_bgcolor="#111",
            font=dict(color="#888", family="IBM Plex Mono", size=8),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, tickfont=dict(size=8, color="#888")),
            margin=dict(t=28, b=8, l=110, r=60)
        )
        st.plotly_chart(fig_sec, use_container_width=True, config={"displayModeBar": False})
    else:
        st.info("Sector data unavailable — market may be closed.")

# ============================================================
# ROW 5 — Commodity details + FX Matrix
# ============================================================
st.markdown("""
<div style="color:#ff8c00;font-size:.6rem;font-weight:700;letter-spacing:.16em;
padding:5px 0 8px;border-bottom:1px solid #2a2a2a;margin-bottom:10px;">
  ◼ COMMODITIES · FX · CRYPTO  —  DETAIL VIEW
</div>""", unsafe_allow_html=True)

r5c1, r5c2, r5c3 = st.columns(3)

def mini_sparkline(series, color="#ffb347"):
    if not series or len(series) < 2: return None
    base = series[0] if series[0] != 0 else 1
    norm = [(v/base-1)*100 for v in series]
    fig = go.Figure(go.Scatter(
        x=list(range(len(norm))), y=norm, mode="lines",
        line=dict(color=color, width=2), fill="tozeroy",
        fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.08)"
    ))
    fig.update_layout(
        height=70, margin=dict(t=2,b=2,l=0,r=0),
        plot_bgcolor="#000", paper_bgcolor="#000",
        xaxis=dict(showgrid=False,showticklabels=False,zeroline=False),
        yaxis=dict(showgrid=False,showticklabels=False,zeroline=True,
                   zerolinecolor="#222"),
        showlegend=False
    )
    return fig

with r5c1:
    st.markdown('<div class="panel-card"><div class="panel-title">COMMODITIES</div>', unsafe_allow_html=True)
    for key, lbl, unit in [("GOLD","Gold","$/oz"),("SILVER","Silver","$/oz"),
                            ("CRUDEOIL","Crude Oil","$/bbl"),("COPPER","Copper","$/lb"),
                            ("NATGAS","Nat Gas","$/MMBtu")]:
        if key in md:
            d  = md[key]
            c  = "#00d084" if d["pct"] >= 0 else "#ff3b3b"
            arr = "▲" if d["pct"] >= 0 else "▼"
            st.markdown(f"""
<div style="display:flex;justify-content:space-between;align-items:center;
padding:5px 0;border-bottom:1px solid #1a1a1a;">
  <span style="color:#888;font-size:.6rem;">{lbl} <span style="color:#444;font-size:.52rem;">{unit}</span></span>
  <span style="color:#e8e8e8;font-size:.68rem;font-weight:600;">{d['last']:,.2f}</span>
  <span style="color:{c};font-size:.62rem;font-weight:700;">{arr}{abs(d['pct']):.2f}%</span>
</div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with r5c2:
    st.markdown('<div class="panel-card"><div class="panel-title">CURRENCY PAIRS</div>', unsafe_allow_html=True)
    for key, lbl in [("USDINR","USD/INR"),("EURUSD","EUR/USD"),("USDJPY","USD/JPY"),
                     ("GBPUSD","GBP/USD"),("DXY","DXY Index")]:
        if key in md:
            d  = md[key]
            c  = "#00d084" if d["pct"] >= 0 else "#ff3b3b"
            arr = "▲" if d["pct"] >= 0 else "▼"
            val = f"{d['last']:.4f}" if d['last'] < 100 else f"{d['last']:.2f}"
            st.markdown(f"""
<div style="display:flex;justify-content:space-between;align-items:center;
padding:5px 0;border-bottom:1px solid #1a1a1a;">
  <span style="color:#888;font-size:.6rem;">{lbl}</span>
  <span style="color:#e8e8e8;font-size:.68rem;font-weight:600;">{val}</span>
  <span style="color:{c};font-size:.62rem;font-weight:700;">{arr}{abs(d['pct']):.2f}%</span>
</div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with r5c3:
    st.markdown('<div class="panel-card"><div class="panel-title">CRYPTO + DIGITAL ASSETS</div>', unsafe_allow_html=True)
    for key, lbl in [("BTC","Bitcoin"),("ETH","Ethereum")]:
        if key in md:
            d  = md[key]
            c  = "#00d084" if d["pct"] >= 0 else "#ff3b3b"
            arr = "▲" if d["pct"] >= 0 else "▼"
            st.markdown(f"""
<div style="margin-bottom:8px;">
  <div style="display:flex;justify-content:space-between;margin-bottom:3px;">
    <span style="color:#888;font-size:.6rem;">{lbl}</span>
    <span style="color:{c};font-size:.62rem;font-weight:700;">{arr}{abs(d['pct']):.2f}%</span>
  </div>
  <div style="color:#e8e8e8;font-size:1rem;font-weight:700;margin-bottom:3px;">${d['last']:,.0f}</div>
</div>""", unsafe_allow_html=True)
            sp = mini_sparkline(d.get("series",[]), c)
            if sp:
                st.plotly_chart(sp, use_container_width=True, config={"displayModeBar":False})
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# FOOTER
# ============================================================
st.markdown(f"""
<div style="background:#0d0d0d;border-top:1px solid #1a1a1a;margin-top:16px;
padding:8px 16px;display:flex;justify-content:space-between;align-items:center;
font-family:'IBM Plex Mono',monospace;font-size:.52rem;color:#555;">
  <span>◼ MONARCH PRO · GLOBAL RISK INTELLIGENCE</span>
  <span>Data: yfinance · alternative.me · Upstox  ·  Last updated: {datetime.now().strftime('%H:%M:%S')}</span>
  <span>Auto-refresh: {'ON (60s)' if auto_refresh else 'OFF'}</span>
</div>
""", unsafe_allow_html=True)
