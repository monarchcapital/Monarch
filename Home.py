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
    color: var(--bb-white) !important; font-size: 0.84rem !important;
}

h1 { color: var(--bb-amber) !important; font-size: 1.25rem !important;
     font-weight: 700 !important; letter-spacing: 0.18em !important;
     text-transform: uppercase !important;
     border-bottom: 2px solid var(--bb-amber) !important;
     padding-bottom: 4px !important; }
h2 { color: var(--bb-amber2) !important; font-size: 1.02rem !important;
     font-weight: 600 !important; letter-spacing: 0.12em !important;
     text-transform: uppercase !important; }
h3 { color: var(--bb-white) !important; font-size: 0.90rem !important;
     font-weight: 600 !important; letter-spacing: 0.1em !important;
     text-transform: uppercase !important; }

[data-testid="metric-container"] {
    background: var(--bb-surface) !important; border-radius: 0 !important;
    border: 1px solid var(--bb-border) !important;
    border-left: 3px solid var(--bb-amber) !important;
    padding: 8px 10px !important;
}
[data-testid="stMetricLabel"] p, [data-testid="metric-container"] label {
    color: var(--bb-muted) !important; font-size: 0.70rem !important;
    letter-spacing: 0.12em !important; text-transform: uppercase !important;
}
[data-testid="stMetricValue"] { color: var(--bb-amber) !important; font-size: 1.19rem !important; font-weight: 700 !important; }
[data-testid="stMetricDelta"] { font-size: 0.75rem !important; }

[data-testid="stDataFrame"] { border: 1px solid var(--bb-border) !important; border-radius: 0 !important; }
.stDataFrame thead tr th { background: #1a1200 !important; color: var(--bb-amber) !important;
    font-size: 0.72rem !important; font-weight: 700 !important; letter-spacing: 0.12em !important;
    text-transform: uppercase !important; border-bottom: 1px solid var(--bb-amber) !important;
    border-right: 1px solid #2a2a2a !important; padding: 5px 8px !important; white-space: nowrap !important; }
.stDataFrame tbody tr td { background: #0d0d0d !important; color: var(--bb-white) !important;
    font-size: 0.83rem !important; border-bottom: 1px solid #1a1a1a !important;
    border-right: 1px solid #1a1a1a !important; padding: 4px 8px !important; white-space: nowrap !important; }
.stDataFrame tbody tr:nth-child(odd) td { background: #111 !important; }
.stDataFrame tbody tr:hover td { background: #1a1400 !important; }

.stButton > button { background: #1a1400 !important; color: var(--bb-amber) !important;
    border: 1px solid var(--bb-amber) !important; border-radius: 0 !important;
    font-size: 0.85rem !important; letter-spacing: 0.1em !important;
    text-transform: uppercase !important; padding: 6px 14px !important; }
.stButton > button:hover { background: var(--bb-amber) !important; color: #000 !important; }

.stTextInput > div > div, .stNumberInput > div > div, .stSelectbox > div > div {
    background: var(--bb-surface) !important; border: 1px solid var(--bb-border) !important;
    border-radius: 0 !important; color: var(--bb-white) !important; font-size: 0.88rem !important; }
.stTextInput label, .stNumberInput label, .stSelectbox label {
    color: var(--bb-muted) !important; font-size: 0.75rem !important; font-family: var(--bb-mono) !important; }
input[type="text"], input[type="password"], input[type="number"] {
    background: var(--bb-surface) !important; color: var(--bb-white) !important;
    border: 1px solid var(--bb-border) !important; font-family: var(--bb-mono) !important;
    font-size: 0.88rem !important; border-radius: 0 !important; }

/* ── EXPANDER — nuclear arrow kill + label preserve ── */
.streamlit-expanderHeader,
[data-testid="stExpander"] summary {
    background: var(--bb-surface) !important;
    color: var(--bb-amber) !important;
    font-family: var(--bb-mono) !important;
    font-size: 0.80rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    border-radius: 0 !important;
    border: 1px solid var(--bb-border) !important;
    padding: 8px 12px !important;
    list-style: none !important;
    cursor: pointer !important;
}
/* Kill every possible form of the icon — SVG, Material Icons span,
   toggle icon testid, first-child spans used by all Streamlit versions */
[data-testid="stExpander"] summary svg,
[data-testid="stExpander"] summary .material-icons,
[data-testid="stExpander"] summary span[data-testid="stExpanderToggleIcon"],
[data-testid="stExpander"] summary > div > span:first-child,
[data-testid="stExpander"] summary > span:first-child {
    display: none !important;
    visibility: hidden !important;
    width: 0 !important;
    height: 0 !important;
    overflow: hidden !important;
    font-size: 0 !important;
    color: transparent !important;
}
[data-testid="stExpander"] summary::-webkit-details-marker,
[data-testid="stExpander"] summary::marker { display: none !important; }
[data-testid="stExpander"] {
    border: 1px solid var(--bb-border) !important;
    border-radius: 0 !important;
    background: var(--bb-surface) !important;
}
[data-testid="stExpanderDetails"] {
    background: var(--bb-bg) !important;
    border-top: 1px solid var(--bb-border) !important;
}
[data-testid="stExpander"] [data-testid="stMarkdownContainer"] p,
[data-testid="stExpander"] [data-testid="stMarkdownContainer"] li,
[data-testid="stExpander"] [data-testid="stMarkdownContainer"] td {
    color: var(--bb-white) !important; font-size: 0.83rem !important;
}
[data-testid="stExpander"] [data-testid="stMarkdownContainer"] th {
    color: var(--bb-amber) !important; background: #1a1200 !important;
}

[data-testid="stAlert"] { background: var(--bb-surface) !important;
    border: 1px solid var(--bb-border) !important; border-radius: 0 !important; }
[data-testid="stAlert"] p { color: var(--bb-white) !important; font-size: 0.85rem !important; }

[data-testid="stCaptionContainer"] p, small { color: var(--bb-muted) !important; font-size: 0.72rem !important; }

.stTabs [data-baseweb="tab-list"] { background: var(--bb-surface) !important;
    border-bottom: 1px solid var(--bb-border) !important; gap: 0 !important; }
.stTabs [data-baseweb="tab"] { background: transparent !important; color: var(--bb-muted) !important;
    font-size: 0.78rem !important; letter-spacing: 0.1em !important; text-transform: uppercase !important;
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
    font-size:0.69rem; letter-spacing:0.12em; padding:2px 8px; }

/* Ticker tape */
@keyframes ticker { from{transform:translateX(0)} to{transform:translateX(-50%)} }
.ticker-wrap { overflow:hidden; background:#0d0d0d; border-top:1px solid var(--bb-border);
    border-bottom:1px solid var(--bb-border); padding:5px 0; }
.ticker-inner { display:inline-flex; animation:ticker 60s linear infinite; white-space:nowrap; }
.ticker-inner:hover { animation-play-state:paused; }
.tick-item { padding:0 32px; font-size:0.78rem; font-family:var(--bb-mono); color:var(--bb-muted);
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
.panel-title { color:var(--bb-amber); font-size:0.72rem; font-weight:700;
    letter-spacing:0.14em; text-transform:uppercase; margin-bottom:8px;
    border-bottom:1px solid var(--bb-border); padding-bottom:5px; }

/* Scanline overlay (cosmetic) */
.scanline-wrap { position:relative; overflow:hidden; }
.scanline-wrap::after { content:''; position:absolute; left:0; right:0; height:2px;
    background:rgba(255,140,0,0.04); animation:scanline 4s linear infinite; pointer-events:none; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# ============================================================
# UPSTOX CREDENTIALS
# ============================================================
UPSTOX_CLIENT_ID     = "e720544b-52d6-4f92-941a-9f2fecb1ec72"
UPSTOX_CLIENT_SECRET = "eujrsvhzju"
UPSTOX_REDIRECT_URI  = "http://127.0.0.1"
UPSTOX_AUTH_URL      = "https://api.upstox.com/v2/login/authorization/dialog"
UPSTOX_TOKEN_URL     = "https://api.upstox.com/v2/login/authorization/token"
UPSTOX_TOKEN_REQ_URL = "https://api.upstox.com/v3/login/auth/token/request/{client_id}"
TOKEN_FILE           = ".upstox_token_scanner"
WEBHOOK_TOKEN_FILE   = ".upstox_webhook_token"  # written by webhook receiver
AUTH_CODE_FILE       = ".upstox_auth_code"

import socketserver, http.server, webbrowser

# ============================================================
# WEBHOOK RECEIVER — catches token pushed by Upstox after phone approval
# Upstox POSTs {"access_token": "...", "message_type": "access_token"} to this URL
# We run a tiny HTTP server on port 8765 to receive it
# ============================================================
WEBHOOK_PORT = 8765

class _WebhookHandler(http.server.BaseHTTPRequestHandler):
    """Receives POST from Upstox with access_token after user taps Approve."""
    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            body   = self.rfile.read(length)
            data   = json.loads(body)
            token  = data.get("access_token", "")
            if token:
                with open(WEBHOOK_TOKEN_FILE, "w") as f: f.write(token)
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')
        except:
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')
    def do_GET(self):
        # Also handle redirect-based flow as fallback
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
font-family:monospace;text-align:center;padding-top:80px;font-size:1.5rem;'>
<b>&#10003; MONARCH PRO</b><br><span style='color:#888;font-size:1.1rem;'>
Auth received. Return to the dashboard.</span></body></html>""")
        except: pass
    def log_message(self, *args): pass

def _start_webhook_server():
    """Start webhook receiver on port 8765 in background thread."""
    try:
        socketserver.TCPServer.allow_reuse_address = True
        with socketserver.TCPServer(("0.0.0.0", WEBHOOK_PORT), _WebhookHandler) as srv:
            srv.handle_request()
    except Exception:
        pass

# ── Legacy redirect server for local fallback (port 80) ──
class _OAuthHandler(http.server.BaseHTTPRequestHandler):
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
font-family:monospace;text-align:center;padding-top:80px;font-size:1.5rem;'>
<b>&#10003; MONARCH PRO</b><br><span style='color:#888;font-size:1.1rem;'>
Auth code received. Return to the dashboard.</span></body></html>""")
        except: pass
    def log_message(self, *args): pass

def _start_redirect_server():
    try:
        socketserver.TCPServer.allow_reuse_address = True
        with socketserver.TCPServer(("127.0.0.1", 80), _OAuthHandler) as srv:
            srv.handle_request()
    except PermissionError:
        try:
            with open(AUTH_CODE_FILE, "w") as f: f.write("__PORT80_BLOCKED__")
        except: pass
    except Exception:
        pass

def upstox_request_token_via_phone(client_id, notifier_url):
    """
    Calls Upstox Access Token Request API.
    Upstox sends push notification + WhatsApp to the account holder.
    On approval, token is POSTed to notifier_url.
    Returns (success, message)
    """
    try:
        url = UPSTOX_TOKEN_REQ_URL.format(client_id=client_id)
        r = requests.post(url,
            json={"notifier_url": notifier_url},
            headers={"Content-Type": "application/json", "accept": "application/json"},
            timeout=15)
        if r.status_code == 200:
            return True, r.json()
        return False, r.text
    except Exception as e:
        return False, str(e)

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
    # Share to all other pages via session state
    st.session_state.scanner_token      = token
    st.session_state.opt_access_token   = token
    try:
        with open(TOKEN_FILE, "w") as f: f.write(token)
    except: pass

# ── Load existing token on startup ──
if "home_token_loaded" not in st.session_state:
    tok = ""
    # 1. secrets
    try: tok = st.secrets.get("upstox_token","")
    except: pass
    # 2. token file
    if not tok and os.path.exists(TOKEN_FILE):
        try: tok = open(TOKEN_FILE).read().strip()
        except: pass
    st.session_state.upstox_token = tok
    st.session_state.home_token_loaded = True

# ── Session state defaults ──
for k, v in [("ux_step","idle"), ("ux_server_started",False), ("ux_webhook_started",False)]:
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
def compute_risk_score(md, fng_val=50):
    """
    Composite Global Risk Score  -100 (extreme risk-off) → +100 (extreme risk-on)
    Uses ALL available page data — 14 signals, total weight = 100 pts.

    VOLATILITY (25pts)
      VIX level           15pts  — low VIX = risk-on
      India VIX level      5pts  — India-specific fear gauge
      Fear & Greed index   5pts  — crypto sentiment proxy

    EQUITIES (25pts)
      SPX 1d %            10pts  — US market direction
      Global breadth       8pts  — DAX + NIKKEI + HSI average direction
      NIFTY 1d %           7pts  — Emerging market proxy

    SAFE HAVENS (20pts)
      GOLD 1d %           10pts  — rising = risk-off
      DXY 1d %             5pts  — rising $ = risk-off (flight to safety)
      US10Y–US2Y spread    5pts  — steep curve = growth; inverted = recession fear

    GROWTH PROXIES (20pts)
      Copper 1d %          8pts  — Dr. Copper = global growth barometer
      Crude Oil 1d %       7pts  — moderate rise = demand growth; spike = inflation risk
      BTC 1d %             5pts  — high-beta risk-on asset

    FX / GLOBAL STRESS (10pts)
      USDINR 1d %          5pts  — rising INR = EM risk-on
      EURUSD 1d %          5pts  — rising EUR = USD weakness = risk-on
    """
    def clamp(v, lo=-1.0, hi=1.0):
        return max(lo, min(hi, v))

    def pct(name):
        return md.get(name, {}).get("pct", 0)

    def last(name):
        return md.get(name, {}).get("last", 0)

    signals = {}   # name → (contribution, weight, raw_value, direction_label)

    # ── VOLATILITY ────────────────────────────────────────────
    vix = last("VIX")
    if   vix < 13: vix_s =  1.0
    elif vix < 18: vix_s =  0.6
    elif vix < 24: vix_s =  0.0
    elif vix < 32: vix_s = -0.6
    else:          vix_s = -1.0
    signals["VIX"] = (vix_s * 15, 15, f"{vix:.1f}", "risk-on" if vix_s > 0 else "risk-off")

    ivix = last("VIXINDIA")
    if   ivix < 13: ivix_s =  1.0
    elif ivix < 18: ivix_s =  0.5
    elif ivix < 25: ivix_s =  0.0
    elif ivix < 35: ivix_s = -0.5
    else:           ivix_s = -1.0
    signals["INDIA VIX"] = (ivix_s * 5, 5, f"{ivix:.1f}", "risk-on" if ivix_s > 0 else "risk-off")

    fng_s = clamp((fng_val - 50) / 50)
    signals["FEAR & GREED"] = (fng_s * 5, 5, str(fng_val), "greed" if fng_s > 0 else "fear")

    # ── EQUITIES ──────────────────────────────────────────────
    spx_s = clamp(pct("SPX") / 1.5)
    signals["S&P 500"] = (spx_s * 10, 10, f"{pct('SPX'):+.2f}%", "bullish" if spx_s > 0 else "bearish")

    global_eq = np.mean([pct("DAX"), pct("NIKKEI"), pct("HSI")])
    geq_s = clamp(global_eq / 1.2)
    signals["GLOBAL EQ"] = (geq_s * 8, 8, f"{global_eq:+.2f}%", "bullish" if geq_s > 0 else "bearish")

    nifty_s = clamp(pct("NIFTY") / 1.0)
    signals["NIFTY"] = (nifty_s * 7, 7, f"{pct('NIFTY'):+.2f}%", "bullish" if nifty_s > 0 else "bearish")

    # ── SAFE HAVENS ───────────────────────────────────────────
    gold_s = clamp(-pct("GOLD") / 1.2)
    signals["GOLD"] = (gold_s * 10, 10, f"{pct('GOLD'):+.2f}%", "risk-on" if gold_s > 0 else "risk-off")

    dxy_s = clamp(-pct("DXY") / 0.8)
    signals["DXY"] = (dxy_s * 5, 5, f"{pct('DXY'):+.2f}%", "risk-on" if dxy_s > 0 else "risk-off")

    us10 = last("US10Y"); us2 = last("US2Y")
    spread = us10 - us2
    if   spread >  1.0: yc_s =  1.0   # steep = growth
    elif spread >  0.0: yc_s =  0.3
    elif spread > -0.5: yc_s = -0.5   # slightly inverted
    else:               yc_s = -1.0   # deeply inverted = recession fear
    signals["YIELD CURVE"] = (yc_s * 5, 5, f"{spread:+.2f}%", "normal" if yc_s > 0 else "inverted")

    # ── GROWTH PROXIES ────────────────────────────────────────
    copper_s = clamp(pct("COPPER") / 1.2)
    signals["COPPER"] = (copper_s * 8, 8, f"{pct('COPPER'):+.2f}%", "risk-on" if copper_s > 0 else "risk-off")

    cl = pct("CRUDEOIL")
    if   cl >  4.0: oil_s = -1.0   # spike = inflation shock
    elif cl >  0.0: oil_s = clamp(cl / 2.0)
    elif cl > -3.0: oil_s = clamp(cl / 1.5)
    else:           oil_s = -0.8   # collapse = demand fear
    signals["CRUDE OIL"] = (oil_s * 7, 7, f"{cl:+.2f}%", "risk-on" if oil_s > 0 else "risk-off")

    btc_s = clamp(pct("BTC") / 4.0)
    signals["BITCOIN"] = (btc_s * 5, 5, f"{pct('BTC'):+.2f}%", "risk-on" if btc_s > 0 else "risk-off")

    # ── FX / GLOBAL STRESS ────────────────────────────────────
    # USDINR: falling USD/INR means INR strengthening = EM risk-on
    inr_s = clamp(-pct("USDINR") / 0.5)
    signals["USD/INR"] = (inr_s * 5, 5, f"{pct('USDINR'):+.2f}%", "risk-on" if inr_s > 0 else "risk-off")

    eur_s = clamp(pct("EURUSD") / 0.6)
    signals["EUR/USD"] = (eur_s * 5, 5, f"{pct('EURUSD'):+.2f}%", "risk-on" if eur_s > 0 else "risk-off")

    total = sum(v[0] for v in signals.values())
    return round(max(-100, min(100, total)), 1), signals

def risk_label(score):
    if score >= 50:  return "STRONG RISK ON",  "#00d084"
    if score >= 20:  return "RISK ON",          "#44cc88"
    if score >= 5:   return "MILD RISK ON",     "#88cc66"
    if score >= -5:  return "NEUTRAL",          "#ffb347"
    if score >= -20: return "MILD RISK OFF",    "#ff9944"
    if score >= -50: return "RISK OFF",         "#ff6644"
    return "STRONG RISK OFF", "#ff3b3b"

# ============================================================
# HEADER BAR
# ============================================================
now_str = datetime.now().strftime("%Y-%m-%d  %H:%M:%S  IST")
token_ok = bool(st.session_state.upstox_token)
tok_html = ('<span class="live-badge"><span class="pulse-dot"></span>UPSTOX CONNECTED</span>'
            if token_ok else
            '<span style="color:#ff3b3b;font-size:0.72rem;border:1px solid #ff3b3b;'
            'padding:2px 8px;">⚠ UPSTOX NOT CONNECTED</span>')

st.markdown(f"""
<div style="background:#ff8c00;color:#000;font-family:'IBM Plex Mono',monospace;
font-size:0.75rem;font-weight:700;letter-spacing:.18em;padding:5px 16px;
display:flex;justify-content:space-between;align-items:center;margin-bottom:0;">
  <span>◼ MONARCH GLOBAL RISK INTELLIGENCE</span>
  <span style="font-size:0.69rem;font-weight:400;">{now_str}</span>
</div>
<div style="background:#0d0d0d;border-bottom:1px solid #2a2a2a;
padding:4px 16px;display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
  <span style="color:#888;font-size:0.69rem;letter-spacing:.08em;">
    GLOBAL MACRO · RISK MONITOR · VOLATILITY · FX · BONDS · COMMODITIES · CRYPTO
  </span>
  <span>{tok_html}</span>
</div>
""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR — Upstox Login (Phone Approval + OTP fallback)
# ============================================================
with st.sidebar:
    st.markdown("""
<div style="color:#ff8c00;font-size:0.90rem;font-weight:700;letter-spacing:.14em;
padding:10px 0 6px;border-bottom:1px solid #2a2a2a;margin-bottom:10px;">
◼ MONARCH PRO
</div>""", unsafe_allow_html=True)

    token_ok = bool(st.session_state.upstox_token)

    # ── Already connected ──
    if token_ok:
        st.markdown(f"""
<div style="background:#001a0a;border:1px solid #00d084;border-left:3px solid #00d084;
padding:10px 12px;font-family:'IBM Plex Mono',monospace;margin-bottom:10px;">
  <div style="color:#00d084;font-size:0.85rem;font-weight:700;letter-spacing:.1em;margin-bottom:4px;">
    ✔ UPSTOX CONNECTED
  </div>
  <div style="color:#888;font-size:0.72rem;">Token: {st.session_state.upstox_token[:20]}…</div>
  <div style="color:#555;font-size:0.65rem;margin-top:3px;">Valid until 3:30 AM · shared across all pages</div>
</div>""", unsafe_allow_html=True)
        if st.button("↺  Disconnect / Re-login", key="re_login_btn", use_container_width=True):
            st.session_state.ux_step = "idle"
            st.session_state.ux_server_started = False
            st.session_state.ux_webhook_started = False
            st.session_state.upstox_token = ""
            st.session_state.scanner_token = ""
            st.session_state.opt_access_token = ""
            for f in [TOKEN_FILE, AUTH_CODE_FILE, WEBHOOK_TOKEN_FILE]:
                try:
                    if os.path.exists(f): os.remove(f)
                except: pass
            st.rerun()

    else:
        # ── Check if webhook already delivered a token ──
        if os.path.exists(WEBHOOK_TOKEN_FILE):
            try:
                tok = open(WEBHOOK_TOKEN_FILE).read().strip()
            except: tok = ""
            if tok:
                save_token(tok)
                try: os.remove(WEBHOOK_TOKEN_FILE)
                except: pass
                st.session_state.ux_step = "idle"
                st.session_state.ux_webhook_started = False
                st.rerun()

        # ── Check if redirect-based flow caught a code ──
        if os.path.exists(AUTH_CODE_FILE):
            try:
                caught = open(AUTH_CODE_FILE).read().strip()
            except: caught = ""
            if caught == "__PORT80_BLOCKED__":
                try: os.remove(AUTH_CODE_FILE)
                except: pass
                st.session_state.ux_step = "port_blocked"
                st.session_state.ux_server_started = False
                st.rerun()
            elif caught:
                with st.spinner("🔑 Exchanging code for token…"):
                    ok, tok, resp = upstox_get_access_token(
                        UPSTOX_CLIENT_ID, UPSTOX_CLIENT_SECRET,
                        UPSTOX_REDIRECT_URI, caught)
                try: os.remove(AUTH_CODE_FILE)
                except: pass
                if ok and tok:
                    save_token(tok)
                    st.session_state.ux_step = "idle"
                    st.session_state.ux_server_started = False
                    st.rerun()
                else:
                    st.session_state.ux_step = "idle"
                    st.error(f"Token exchange failed: {str(resp)[:200]}")

        # ── Login mode selector ──
        st.markdown("""
<div style="background:#0a0800;border:1px solid #ff8c00;padding:10px 12px;margin-bottom:10px;">
  <div style="color:#ffb347;font-size:0.72rem;font-weight:700;letter-spacing:.1em;margin-bottom:6px;">
    ⚡ UPSTOX LOGIN
  </div>
  <div style="color:#888;font-size:0.65rem;line-height:1.8;">
    Choose how to connect each morning:
  </div>
</div>""", unsafe_allow_html=True)

        login_mode = st.radio("Login method", 
            ["📱 Phone Approval (1 tap)", "🌐 OTP in Browser"],
            key="login_mode_radio",
            label_visibility="collapsed")

        st.markdown('<div style="border-top:1px solid #1a1a1a;margin:8px 0;"></div>', unsafe_allow_html=True)

        # ══════════════════════════════════════════════════════
        # MODE 1: PHONE APPROVAL (semi-automated webhook flow)
        # ══════════════════════════════════════════════════════
        if login_mode == "📱 Phone Approval (1 tap)":
            st.markdown("""
<div style="background:#001520;border:1px solid #1e90ff;padding:9px 11px;margin-bottom:10px;
font-family:'IBM Plex Mono',monospace;">
  <div style="color:#1e90ff;font-size:0.72rem;font-weight:700;margin-bottom:5px;">HOW IT WORKS</div>
  <div style="color:#888;font-size:0.62rem;line-height:1.9;">
    1. Click <b style="color:#ff8c00;">REQUEST TOKEN</b><br/>
    2. Check Upstox app / WhatsApp<br/>
    3. Tap <b style="color:#00d084;">Approve</b> on the notification<br/>
    4. Dashboard auto-connects ✔
  </div>
  <div style="color:#555;font-size:0.58rem;margin-top:5px;border-top:1px solid #1a1a1a;padding-top:4px;">
    ⚠ Requires <b style="color:#ffb347;">Notifier Webhook URL</b> set in your<br/>
    Upstox API app settings (see below)
  </div>
</div>""", unsafe_allow_html=True)

            # The webhook URL — user needs to set their Streamlit app URL here
            default_webhook = st.secrets.get("webhook_url", "") if True else ""
            try: default_webhook = st.secrets.get("webhook_url","")
            except: default_webhook = ""
            
            webhook_url = st.text_input("Your Webhook URL", key="webhook_url_inp",
                value=default_webhook,
                placeholder="https://your-app.streamlit.app/webhook")
            st.markdown('<div style="color:#555;font-size:0.60rem;margin:-4px 0 8px;line-height:1.6;">' +
                'Set this same URL as <b style="color:#ff8c00;">Notifier Webhook Endpoint</b> ' +
                'in your Upstox Developer App settings.' +
                '</div>', unsafe_allow_html=True)

            if st.button("📱  REQUEST TOKEN — NOTIFY MY PHONE", key="phone_req_btn",
                         use_container_width=True, type="primary"):
                if not webhook_url.strip():
                    st.error("Enter your webhook URL first.")
                else:
                    # Start local webhook listener
                    if not st.session_state.ux_webhook_started:
                        t = threading.Thread(target=_start_webhook_server, daemon=True)
                        t.start()
                        st.session_state.ux_webhook_started = True
                    ok, resp = upstox_request_token_via_phone(UPSTOX_CLIENT_ID, webhook_url.strip())
                    if ok:
                        st.session_state.ux_step = "waiting_phone"
                        st.rerun()
                    else:
                        st.error(f"Request failed: {str(resp)[:200]}")

            if st.session_state.ux_step == "waiting_phone":
                st.markdown("""
<div style="background:#001520;border:1px solid #1e90ff;padding:10px;
font-size:0.68rem;color:#1e90ff;margin-top:8px;text-align:center;">
  <span class="blink">📱 Waiting for your approval…</span><br/>
  <span style="color:#555;font-size:0.62rem;">Check Upstox app or WhatsApp notification</span>
</div>""", unsafe_allow_html=True)
                time.sleep(3)
                st.rerun()

        # ══════════════════════════════════════════════════════
        # MODE 2: OTP IN BROWSER — clickable link (works on Cloud)
        # ══════════════════════════════════════════════════════
        else:
            # Build auth URL always so the link is ready
            _params   = {"response_type": "code",
                         "client_id":     UPSTOX_CLIENT_ID,
                         "redirect_uri":  UPSTOX_REDIRECT_URI}
            _auth_url = UPSTOX_AUTH_URL + "?" + urllib.parse.urlencode(_params)

            st.markdown(f"""
<div style="background:#0a0800;border:1px solid #ff8c00;padding:10px 12px;margin-bottom:10px;">
  <div style="color:#ffb347;font-size:0.70rem;font-weight:700;letter-spacing:.08em;margin-bottom:6px;">
    HOW IT WORKS
  </div>
  <div style="color:#888;font-size:0.65rem;line-height:1.9;">
    1. Click the link below to open Upstox login<br/>
    2. Login with your phone + OTP<br/>
    3. After login, browser redirects to a URL like:<br/>
    <span style="color:#ff8c00;">http://127.0.0.1/?code=XXXXXX</span><br/>
    4. Copy that full URL and paste below
  </div>
  <a href="{_auth_url}" target="_blank"
     style="display:block;margin-top:10px;background:#ff8c00;color:#000;
     text-align:center;padding:9px;font-family:'IBM Plex Mono',monospace;
     font-size:0.75rem;font-weight:700;letter-spacing:.1em;text-decoration:none;">
    ↗ OPEN UPSTOX LOGIN PAGE
  </a>
</div>""", unsafe_allow_html=True)

            # Paste redirect URL
            st.markdown('<div style="color:#888;font-size:0.65rem;margin-bottom:4px;">After login, paste the redirect URL here:</div>', unsafe_allow_html=True)
            manual_url = st.text_input("Redirect URL", key="manual_url", label_visibility="collapsed",
                                       placeholder="http://127.0.0.1/?code=…")
            if st.button("✔ Use This URL", key="manual_url_btn", use_container_width=True):
                if manual_url.strip():
                    try:
                        p = urllib.parse.parse_qs(urllib.parse.urlparse(manual_url.strip()).query)
                        code = p.get("code",[""])[0]
                    except: code = ""
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

        # Direct token paste (always visible)
        st.markdown('<div style="border-top:1px solid #1a1a1a;margin:10px 0 6px;"></div>', unsafe_allow_html=True)
        st.markdown('<div style="color:#444;font-size:0.62rem;margin-bottom:4px;">Or paste token directly:</div>', unsafe_allow_html=True)
        direct_tok = st.text_input("Access Token", key="direct_tok",
                                   type="password", label_visibility="collapsed",
                                   placeholder="eyJ0eXAiOiJKV1Q…")
        if st.button("✔  Use Token", key="direct_tok_btn", use_container_width=True):
            if direct_tok.strip():
                save_token(direct_tok.strip())
                st.session_state.ux_step = "idle"
                st.rerun()
            else:
                st.warning("Paste your token first.")

    # ── Dashboard controls ──
    st.markdown('<div style="border-top:1px solid #2a2a2a;margin:12px 0 8px;"></div>', unsafe_allow_html=True)
    st.markdown('<div style="color:#ff8c00;font-size:0.75rem;font-weight:700;letter-spacing:.1em;margin-bottom:6px;">⚙ DASHBOARD</div>', unsafe_allow_html=True)
    auto_refresh = st.checkbox("Auto-refresh (60s)", value=False, key="auto_refresh")
    if st.button("↺  Refresh Data", key="refresh_btn", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.markdown('<div style="border-top:1px solid #2a2a2a;margin:12px 0 8px;"></div>', unsafe_allow_html=True)
    st.markdown("""
<div style="color:#888;font-size:0.70rem;line-height:2;">
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
fng_val_now = fng[0]["value"] if fng else 50
risk_score, risk_signals = compute_risk_score(md, fng_val=fng_val_now)
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
    gauge_w  = int((risk_score + 100) / 200 * 100)
    gauge_c  = rc
    vix_val  = md.get("VIX",{}).get("last", 20)
    vix_c    = "#00d084" if vix_val < 16 else "#ffb347" if vix_val < 26 else "#ff3b3b"
    fng_val  = fng[0]["value"] if fng else 50
    fng_lbl  = fng[0]["label"] if fng else "—"
    fng_c    = "#00d084" if fng_val > 60 else "#ff3b3b" if fng_val < 40 else "#ffb347"

    # Build signal breakdown rows HTML
    SIGNAL_GROUPS = [
        ("VOLATILITY",     ["VIX","INDIA VIX","FEAR & GREED"]),
        ("EQUITIES",       ["S&P 500","GLOBAL EQ","NIFTY"]),
        ("SAFE HAVENS",    ["GOLD","DXY","YIELD CURVE"]),
        ("GROWTH PROXY",   ["COPPER","CRUDE OIL","BITCOIN"]),
        ("FX / STRESS",    ["USD/INR","EUR/USD"]),
    ]
    GROUP_COLORS = {
        "VOLATILITY":   "#ff8c00",
        "EQUITIES":     "#00d084",
        "SAFE HAVENS":  "#1e90ff",
        "GROWTH PROXY": "#ffb347",
        "FX / STRESS":  "#cc88ff",
    }

    breakdown_html = ""
    for grp_name, sig_keys in SIGNAL_GROUPS:
        gc = GROUP_COLORS[grp_name]
        breakdown_html += f'''<div style="color:{gc};font-size:0.58rem;font-weight:700;
letter-spacing:.12em;margin:6px 0 3px;padding-bottom:2px;border-bottom:1px solid #1a1a1a;">
{grp_name}</div>'''
        for k in sig_keys:
            if k not in risk_signals: continue
            contrib, weight, raw_val, direction = risk_signals[k]
            bar_pct  = min(100, max(0, (contrib / weight + 1) / 2 * 100)) if weight else 50
            sig_c    = "#00d084" if contrib > 0.5 else "#ff3b3b" if contrib < -0.5 else "#888"
            arrow    = "▲" if contrib > 0.5 else "▼" if contrib < -0.5 else "▬"
            contrib_str = f"{contrib:+.1f}"
            breakdown_html += f"""
<div style="display:grid;grid-template-columns:70px 1fr 32px 30px;
gap:4px;align-items:center;margin-bottom:3px;">
  <span style="color:#888;font-size:0.60rem;white-space:nowrap;overflow:hidden;">{k}</span>
  <div style="height:3px;background:#1a1a1a;border-radius:0;">
    <div style="height:100%;width:{bar_pct:.0f}%;background:{sig_c};opacity:.8;"></div>
  </div>
  <span style="color:#555;font-size:0.58rem;text-align:right;">{raw_val}</span>
  <span style="color:{sig_c};font-size:0.62rem;font-weight:700;text-align:right;">{contrib_str}</span>
</div>"""

    st.markdown(f"""
<div class="scanline-wrap" style="background:var(--bb-surface);border:1px solid var(--bb-border);
border-left:4px solid {gauge_c};padding:14px 16px;font-family:'IBM Plex Mono',monospace;">
  <div style="color:#555;font-size:0.62rem;letter-spacing:.14em;margin-bottom:3px;">
    ◼ GLOBAL RISK MONITOR  —  14 SIGNALS
  </div>
  <div style="color:{gauge_c};font-size:1.88rem;font-weight:700;letter-spacing:.06em;margin-bottom:1px;">
    {rl}
  </div>
  <div style="color:#888;font-size:0.75rem;margin-bottom:10px;">
    Score: <span style="color:{gauge_c};font-weight:700;">{risk_score:+.1f}</span>
    <span style="color:#444;"> / ±100</span>
  </div>
  <div style="color:#444;font-size:0.60rem;letter-spacing:.08em;margin-bottom:2px;">
    ◄ RISK-OFF ───────────────── RISK-ON ►
  </div>
  <div style="height:7px;background:#1a1a1a;margin-bottom:10px;position:relative;">
    <div style="position:absolute;left:50%;top:0;width:1px;height:100%;background:#333;"></div>
    <div style="height:100%;width:{gauge_w}%;
    background:linear-gradient(90deg,#ff3b3b 0%,#ffb347 50%,#00d084 100%);"></div>
  </div>
  <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:5px;margin-bottom:10px;">
    <div style="background:#0d0d0d;border:1px solid #1a1a1a;padding:5px 7px;">
      <div style="color:#444;font-size:0.58rem;">VIX</div>
      <div style="color:{vix_c};font-size:1.12rem;font-weight:700;">{vix_val:.1f}</div>
      <div style="color:#444;font-size:0.55rem;">{"FEAR" if vix_val>26 else "NORMAL" if vix_val>16 else "LOW"}</div>
    </div>
    <div style="background:#0d0d0d;border:1px solid #1a1a1a;padding:5px 7px;">
      <div style="color:#444;font-size:0.58rem;">F&G INDEX</div>
      <div style="color:{fng_c};font-size:1.12rem;font-weight:700;">{fng_val}</div>
      <div style="color:#444;font-size:0.55rem;">{fng_lbl.upper()[:8]}</div>
    </div>
    <div style="background:#0d0d0d;border:1px solid #1a1a1a;padding:5px 7px;">
      <div style="color:#444;font-size:0.58rem;">INDIA VIX</div>
      <div style="color:{"#00d084" if md.get("VIXINDIA",{}).get("last",20)<18 else "#ff3b3b"};
      font-size:1.12rem;font-weight:700;">{md.get("VIXINDIA",{}).get("last",0):.1f}</div>
      <div style="color:#444;font-size:0.55rem;">{"LOW" if md.get("VIXINDIA",{}).get("last",20)<18 else "ELEVATED"}</div>
    </div>
  </div>
  <div style="border-top:1px solid #1a1a1a;padding-top:8px;">
    <div style="color:#555;font-size:0.58rem;letter-spacing:.12em;margin-bottom:5px;">
      SIGNAL BREAKDOWN  <span style="color:#2a2a2a;">(contribution pts)</span>
    </div>
    {breakdown_html}
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
<table style="width:100%;border-collapse:collapse;font-family:'IBM Plex Mono',monospace;font-size:0.78rem;">
<thead>
<tr style="background:#1a1200;border-bottom:1px solid #ff8c00;">
  <th style="color:#ff8c00;padding:5px 10px;text-align:left;letter-spacing:.1em;font-size:0.70rem;">INSTRUMENT</th>
  <th style="color:#ff8c00;padding:5px 10px;text-align:right;letter-spacing:.1em;font-size:0.70rem;">PRICE</th>
  <th style="color:#ff8c00;padding:5px 10px;text-align:right;letter-spacing:.1em;font-size:0.70rem;">CHG</th>
  <th style="color:#ff8c00;padding:5px 10px;text-align:right;letter-spacing:.1em;font-size:0.70rem;">CHG %</th>
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
<div style="color:#ff8c00;font-size:0.75rem;font-weight:700;letter-spacing:.16em;
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
# ROW 3 — Polymarket Live Bets (9 tabs × 8 markets) + Yield Spread + VIX
# ============================================================
st.markdown("""
<div style="color:#ff8c00;font-size:0.75rem;font-weight:700;letter-spacing:.16em;
padding:5px 0 8px;border-bottom:1px solid #2a2a2a;margin-bottom:10px;">
  ◼ RISK SIGNAL DASHBOARD  —  POLYMARKET LIVE BETS  ·  REAL MONEY PREDICTION MARKETS
</div>""", unsafe_allow_html=True)

poly_left, poly_right = st.columns([3, 2])

# ── Shared helpers (defined once, used inside tabs) ──────────────────────
BLACKLIST_PM = [
    "nba","nfl","mlb","nhl","premier league","champions league","la liga","serie a",
    "bundesliga","ligue 1","super bowl","world cup","fifa","ncaa","mvp","playoff",
    "championship","transfer","roster","draft pick","coach fired","manager fired",
    "game 1","game 2","game 3","game 4","game 5","game 6","game 7",
    "win the series","win the finals","win the title","season wins","win the cup",
    "oscar","grammy","emmy","golden globe","academy award","album","song","tour",
    "box office","movie","film","tv show","netflix","celebrity","kardashian",
    "taylor swift","beyonce","kanye","podcast","youtube","twitch","streamer",
    "will it rain","weather","temperature","fight","ufc","boxing","bout","round",
    "award show","who will host","nba draft","mlb draft","number 1 pick",
    "tennis","golf","formula 1","f1 race","grand prix","cricket","ipl","rugby",
    "survivor","bachelor","reality","tiktok ban","twitter","elon musk tweet",
]

# ── Each list uses EXACT PHRASES only — no loose single words that cause false matches ──
TAB_CONFIGS = {
    "⚔ War & Geo": [
        "ceasefire","peace deal","peace talks","invasion","ground offensive","military operation",
        "airstrike","missile strike","nuclear strike","nuclear weapon","nuclear war","wmd",
        "troops withdraw","troops deploy","frontline","war crimes","armed conflict",
        "coup attempt","regime change","insurgency","civil war","proxy war",
    ],
    "🛢 Oil & Energy": [
        "crude oil","oil price","oil above","oil below","brent crude","wti crude",
        "opec cut","opec+","oil production","oil output","oil embargo","oil sanction",
        "natural gas price","lng price","energy crisis","fuel price","gasoline price",
        "oil minister","oil supply","oil demand","barrel of oil","shale oil",
    ],
    "📊 Macro & Econ": [
        "federal reserve","rate cut","rate hike","interest rate","fed meeting","fomc",
        "powell","ecb rate","bank of england rate","rbi rate","central bank",
        "quantitative easing","quantitative tightening","monetary policy",
        "debt ceiling","sovereign default","trade deficit","budget deficit",
        "fiscal policy","stimulus package","government shutdown","credit rating",
    ],
    "📉 Inflation/GDP": [
        "inflation rate","cpi report","pce inflation","core inflation","disinflation",
        "gdp growth","gdp contraction","gdp forecast","recession","us recession",
        "global recession","stagflation","soft landing","hard landing",
        "unemployment rate","jobs report","nonfarm payroll","wage growth",
        "consumer price","producer price","cost of living","purchasing power",
    ],
    "₿ Crypto": [
        "bitcoin price","bitcoin etf","bitcoin above","bitcoin below","btc above","btc below",
        "ethereum price","eth above","eth below","crypto market","crypto crash","crypto ban",
        "sec crypto","crypto etf","bitcoin halving","bitcoin reserve",
        "solana price","xrp price","crypto regulation","stablecoin","defi hack",
    ],
    "📈 Markets": [
        "s&p 500","sp500","nasdaq","dow jones","stock market crash","market correction",
        "bear market","bull market","market rally","stock market","ipo",
        "10-year yield","yield curve","fed pivot","short squeeze","margin call",
        "volatility index","vix spike","hedge fund","market cap","earnings season",
    ],
    "🇺🇸 US Politics": [
        "trump","tariff","trade war","trade deal","executive order","impeachment",
        "border wall","immigration policy","us election","us congress","us senate",
        "us-china","us-russia","us-iran","us-israel","us sanctions","nato summit",
        "g7 summit","g20 summit","pentagon","us budget","us debt limit",
        "doge department","elon musk","deportation","us policy",
    ],
    "🕌 Middle East": [
        "israel","gaza","hamas","hezbollah","iran nuclear","iran deal","iran sanctions",
        "lebanon war","west bank","idf","palestine state","netanyahu","tehran",
        "red sea attack","houthi","yemen war","saudi arabia","uae","qatar","mbs",
        "persian gulf","strait of hormuz","middle east war","baghdad","two-state",
    ],
    "🇪🇺 Europe/NATO": [
        "ukraine war","ukraine ceasefire","ukraine russia","zelensky","putin",
        "nato expansion","nato article 5","eu sanctions","sanctions on russia",
        "russian oil","nord stream","europe recession","crimea","donbas",
        "kharkiv","kherson","zaporizhzhia","moldova","belarus","poland","finland",
        "germany recession","france election","uk election","macron","scholz",
    ],
    "🐉 China & Taiwan": [
        "taiwan invasion","taiwan strait","taiwan independence","taiwan war",
        "china invade","chinese military","pla","xi jinping","south china sea",
        "us-china","china tariff","chip war","semiconductor ban","decoupling",
        "china gdp","china recession","china property","evergrande","yuan devaluation",
        "hong kong","huawei ban","belt and road","one china","reunification",
    ],
}

TAB_COLORS = {
    "⚔ War & Geo":      "#ff4444",
    "🛢 Oil & Energy":  "#ffb347",
    "📊 Macro & Econ":  "#1e90ff",
    "📉 Inflation/GDP": "#ff6688",
    "₿ Crypto":         "#f7931a",
    "📈 Markets":        "#00d084",
    "🇺🇸 US Politics":  "#4488ff",
    "🕌 Middle East":    "#cc6644",
    "🇪🇺 Europe/NATO":  "#88aaff",
    "🐉 China & Taiwan": "#dd4444",
}

@st.cache_data(ttl=300, show_spinner=False)
def fetch_polymarket_tab(tab_name: str, keywords: list, limit: int = 8):
    """
    Fetch up to `limit` Polymarket markets matching `keywords` for a specific tab.
    Pulls 300 markets sorted by volume, filters by whitelist keywords,
    excludes blacklisted topics.
    """
    def is_match(q: str) -> bool:
        ql = q.lower()
        if any(bl in ql for bl in BLACKLIST_PM):
            return False
        return any(kw in ql for kw in keywords)

    def parse_price(m):
        prices = m.get("outcomePrices", "[]")
        if isinstance(prices, str):
            try: prices = json.loads(prices)
            except: return None
        if isinstance(prices, list) and len(prices) >= 1:
            try: return round(float(prices[0]) * 100, 1)
            except: return None
        return None

    results = []
    # Primary: Gamma API
    try:
        r = requests.get(
            "https://gamma-api.polymarket.com/markets",
            params={"limit": 300, "active": "true", "closed": "false",
                    "order": "volume", "ascending": "false"},
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
            timeout=12
        )
        if r.status_code == 200:
            raw  = r.json()
            data = raw if isinstance(raw, list) else raw.get("markets", [])
            for m in data:
                q = m.get("question", "")
                if not is_match(q): continue
                yes_pct = parse_price(m)
                if yes_pct is None: continue
                vol  = 0
                try: vol = float(m.get("volume") or 0)
                except: pass
                slug = m.get("slug", "")
                results.append({
                    "question": q,
                    "yes_pct":  yes_pct,
                    "volume":   vol,
                    "url": f"https://polymarket.com/event/{slug}" if slug else "https://polymarket.com",
                })
                if len(results) >= limit: break
    except Exception:
        pass

    # Fallback: CLOB API
    if not results:
        try:
            r2 = requests.get(
                "https://clob.polymarket.com/markets",
                params={"limit": 300},
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=12
            )
            if r2.status_code == 200:
                for m in r2.json().get("data", []):
                    q = m.get("question","")
                    if not is_match(q): continue
                    tokens  = m.get("tokens", [])
                    yes_tok = next((t for t in tokens if t.get("outcome","").upper()=="YES"), None)
                    if not yes_tok: continue
                    try: yes_pct = round(float(yes_tok.get("price",0))*100, 1)
                    except: continue
                    cid = m.get("condition_id","")
                    results.append({
                        "question": q,
                        "yes_pct":  yes_pct,
                        "volume":   0,
                        "url": f"https://polymarket.com/event/{cid}" if cid else "https://polymarket.com",
                    })
                    if len(results) >= limit: break
        except Exception:
            pass

    return results[:limit]

def poly_card(question, yes_pct, volume=0, url="#", color="#ff8c00"):
    no_pct   = round(100 - yes_pct, 1)
    yes_c    = "#00d084" if yes_pct >= 50 else "#666"
    no_c     = "#ff3b3b" if no_pct  >= 50 else "#666"
    vol_str  = f"${volume/1e6:.1f}M" if volume >= 1e6 else f"${volume/1e3:.0f}K" if volume >= 1000 else "—"
    q_display = question if len(question) <= 68 else question[:65] + "…"
    return f"""
<a href="{url}" target="_blank" style="text-decoration:none;">
<div style="background:#0d0d0d;border:1px solid #1e1e1e;border-left:3px solid {color};
padding:8px 10px;margin-bottom:5px;font-family:'IBM Plex Mono',monospace;">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">
    <span style="color:#ccc;font-size:0.68rem;line-height:1.4;flex:1;padding-right:8px;">{q_display}</span>
    <span style="color:#444;font-size:0.55rem;white-space:nowrap;">{vol_str}</span>
  </div>
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:3px;">
    <span style="color:{yes_c};font-size:0.85rem;font-weight:700;">YES {yes_pct:.0f}%</span>
    <span style="color:{no_c};font-size:0.80rem;font-weight:600;">NO {no_pct:.0f}%</span>
  </div>
  <div style="height:4px;background:#1a1a1a;overflow:hidden;">
    <div style="height:100%;width:{yes_pct}%;background:{color};opacity:0.8;"></div>
  </div>
  <div style="color:#2a2a2a;font-size:0.53rem;margin-top:2px;text-align:right;">polymarket.com ↗</div>
</div></a>"""

def poly_empty(color):
    return f"""
<div style="background:#0d0d0d;border:1px solid #1a1a1a;padding:20px;
text-align:center;font-family:'IBM Plex Mono',monospace;color:#444;font-size:0.70rem;">
  <div style="color:{color};margin-bottom:4px;">⚠ No markets found</div>
  <div style="font-size:0.62rem;margin-bottom:8px;">API unavailable or no active bets</div>
  <a href="https://polymarket.com" target="_blank"
     style="color:{color};font-size:0.65rem;text-decoration:none;">
    Browse polymarket.com ↗
  </a>
</div>"""

with poly_left:
    st.markdown('''<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
  <span style="color:#ff8c00;font-size:0.75rem;font-weight:700;letter-spacing:.12em;">◼ POLYMARKET LIVE BETS</span>
  <span style="background:#1a0a00;border:1px solid #ff8c00;color:#ff8c00;font-size:0.55rem;padding:1px 6px;letter-spacing:.08em;">REAL MONEY · 10 CATEGORIES · 8 MARKETS EACH</span>
</div>''', unsafe_allow_html=True)

    tab_names = list(TAB_CONFIGS.keys())
    tabs = st.tabs(tab_names)

    for tab_obj, tab_name in zip(tabs, tab_names):
        with tab_obj:
            color    = TAB_COLORS[tab_name]
            keywords = TAB_CONFIGS[tab_name]
            markets  = fetch_polymarket_tab(tab_name, keywords, limit=8)
            if markets:
                for m in markets:
                    st.markdown(poly_card(m["question"], m["yes_pct"],
                                          m["volume"], m["url"], color),
                                unsafe_allow_html=True)
            else:
                st.markdown(poly_empty(color), unsafe_allow_html=True)

with poly_right:
    # ── VIX + India VIX ──
    india_vix = md.get("VIXINDIA",{}).get("last", 0)
    ivix_c   = "#00d084" if india_vix < 15 else "#ffb347" if india_vix < 22 else "#ff3b3b"
    ivix_lbl = "LOW FEAR" if india_vix < 15 else "MODERATE" if india_vix < 22 else "HIGH FEAR"
    vix_val  = md.get("VIX",{}).get("last", 0)
    vix_c    = "#00d084" if vix_val < 18 else "#ffb347" if vix_val < 26 else "#ff3b3b"

    st.markdown(f"""
<div class="panel-card" style="margin-bottom:8px;">
  <div class="panel-title">INDIA VIX + VOLATILITY</div>
  <div style="display:flex;align-items:center;gap:16px;margin-bottom:8px;">
    <div>
      <div style="color:#555;font-size:0.65rem;letter-spacing:.1em;">INDIA VIX</div>
      <div style="color:{ivix_c};font-size:2.75rem;font-weight:700;line-height:1;">{india_vix:.2f}</div>
      <div style="color:{ivix_c};font-size:0.72rem;">{ivix_lbl}</div>
    </div>
    <div style="flex:1;">
      <div style="color:#555;font-size:0.65rem;margin-bottom:4px;">VIX LEVEL</div>
      <div class="risk-bar-wrap">
        <div class="risk-bar-fill" style="width:{min(100,india_vix/40*100):.0f}%;
        background:linear-gradient(90deg,#00d084,#ffb347,#ff3b3b);"></div>
      </div>
      <div style="display:flex;justify-content:space-between;font-size:0.60rem;color:#444;margin-top:2px;">
        <span>0</span><span>LOW</span><span>HIGH</span><span>40</span>
      </div>
    </div>
  </div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;">
    <div style="background:#0d0d0d;border:1px solid #1a1a1a;padding:6px 8px;">
      <div style="color:#555;font-size:0.62rem;">VIX (USA)</div>
      <div style="color:{vix_c};font-size:1.06rem;font-weight:700;">{vix_val:.1f}</div>
    </div>
    <div style="background:#0d0d0d;border:1px solid #1a1a1a;padding:6px 8px;">
      <div style="color:#555;font-size:0.62rem;">VIX Δ</div>
      <div style="color:{'#ff3b3b' if md.get('VIX',{}).get('pct',0)>0 else '#00d084'};
           font-size:1.06rem;font-weight:700;">{md.get('VIX',{}).get('pct',0):+.2f}%</div>
    </div>
  </div>
</div>""", unsafe_allow_html=True)

    # ── Crypto Fear & Greed ──
    if fng:
        fng_vals   = [d["value"] for d in fng[:7]][::-1]
        fng_colors = ["#00d084" if v > 60 else "#ff3b3b" if v < 40 else "#ffb347" for v in fng_vals]
        fig_fng = go.Figure(go.Bar(
            x=list(range(len(fng_vals))), y=fng_vals,
            marker_color=fng_colors, text=[str(v) for v in fng_vals],
            textposition="outside", textfont=dict(size=8, color="#888")
        ))
        fig_fng.add_hline(y=50, line=dict(color="#333", dash="dot"))
        fig_fng.update_layout(
            title=dict(text="CRYPTO FEAR & GREED (7D)", font=dict(color="#ff8c00",size=9), x=0),
            height=160, plot_bgcolor="#000", paper_bgcolor="#111",
            font=dict(color="#888", family="IBM Plex Mono", size=8),
            xaxis=dict(showticklabels=False, showgrid=False),
            yaxis=dict(range=[0,105], showgrid=False, showticklabels=False),
            margin=dict(t=28, b=8, l=0, r=0), showlegend=False
        )
        st.plotly_chart(fig_fng, use_container_width=True, config={"displayModeBar": False})

    # ── Yield Curve ──
    us10y  = md.get("US10Y",{}).get("last", 4.2)
    us2y   = md.get("US2Y",{}).get("last", 4.5)
    spread = us10y - us2y
    spread_c   = "#ff3b3b" if spread < 0 else "#00d084"
    spread_lbl = "INVERTED (recession signal)" if spread < 0 else "NORMAL" if spread < 0.5 else "STEEP"
    spx_gold_ratio = None
    if "SPX" in md and "GOLD" in md and md["GOLD"]["last"] > 0:
        spx_gold_ratio = md["SPX"]["last"] / md["GOLD"]["last"]

    st.markdown(f"""
<div class="panel-card" style="margin-bottom:8px;">
  <div class="panel-title">YIELD CURVE + BOND MARKET</div>
  <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin-bottom:8px;">
    <div style="background:#0d0d0d;border:1px solid #1a1a1a;padding:8px;">
      <div style="color:#555;font-size:0.62rem;letter-spacing:.1em;">US 10Y YIELD</div>
      <div style="color:#ffb347;font-size:1.38rem;font-weight:700;">{us10y:.2f}%</div>
    </div>
    <div style="background:#0d0d0d;border:1px solid #1a1a1a;padding:8px;">
      <div style="color:#555;font-size:0.62rem;letter-spacing:.1em;">US 2Y YIELD</div>
      <div style="color:#ffb347;font-size:1.38rem;font-weight:700;">{us2y:.2f}%</div>
    </div>
    <div style="background:#0d0d0d;border:1px solid #1a1a1a;padding:8px;">
      <div style="color:#555;font-size:0.62rem;letter-spacing:.1em;">10Y–2Y SPREAD</div>
      <div style="color:{spread_c};font-size:1.38rem;font-weight:700;">{spread:+.2f}%</div>
    </div>
  </div>
  <div style="color:{spread_c};font-size:0.68rem;font-weight:700;margin-bottom:6px;">
    {spread_lbl}
  </div>
  <div style="height:6px;background:#1a1a1a;overflow:hidden;margin-bottom:8px;">
    <div style="height:100%;width:{min(100,max(0,(spread+1)/2*100)):.0f}%;
    background:linear-gradient(90deg,#ff3b3b,#ffb347,#00d084);"></div>
  </div>
  {"" if not spx_gold_ratio else f'<div style="display:flex;justify-content:space-between;padding-top:6px;border-top:1px solid #1a1a1a;"><span style="color:#555;font-size:0.62rem;">SPX/GOLD RATIO</span><span style="color:#ffb347;font-size:0.88rem;font-weight:700;">{spx_gold_ratio:.2f}</span></div>'}
</div>""", unsafe_allow_html=True)
# ============================================================
# ROW 4 — India Macro + Sector Performance
# ============================================================
st.markdown("""
<div style="color:#ff8c00;font-size:0.75rem;font-weight:700;letter-spacing:.16em;
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
  <td style="color:#e8e8e8;padding:5px 10px;font-size:0.78rem;">{lbl}</td>
  <td style="color:#e8e8e8;padding:5px 10px;text-align:right;font-size:0.78rem;">{val}</td>
  <td style="color:{c};padding:5px 10px;text-align:right;font-size:0.78rem;font-weight:700;">
    {arr} {abs(d['pct']):.2f}%</td>
</tr>"""

    st.markdown(f"""
<div class="panel-card">
  <div class="panel-title">INDIA INDICES</div>
  <table style="width:100%;border-collapse:collapse;">
  <thead><tr style="background:#1a1200;border-bottom:1px solid #ff8c00;">
    <th style="color:#ff8c00;padding:5px 10px;text-align:left;font-size:0.68rem;letter-spacing:.1em;">INDEX</th>
    <th style="color:#ff8c00;padding:5px 10px;text-align:right;font-size:0.68rem;letter-spacing:.1em;">PRICE</th>
    <th style="color:#ff8c00;padding:5px 10px;text-align:right;font-size:0.68rem;letter-spacing:.1em;">CHG %</th>
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
<div style="color:#ff8c00;font-size:0.75rem;font-weight:700;letter-spacing:.16em;
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
  <span style="color:#888;font-size:0.75rem;">{lbl} <span style="color:#444;font-size:0.65rem;">{unit}</span></span>
  <span style="color:#e8e8e8;font-size:0.85rem;font-weight:600;">{d['last']:,.2f}</span>
  <span style="color:{c};font-size:0.78rem;font-weight:700;">{arr}{abs(d['pct']):.2f}%</span>
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
  <span style="color:#888;font-size:0.75rem;">{lbl}</span>
  <span style="color:#e8e8e8;font-size:0.85rem;font-weight:600;">{val}</span>
  <span style="color:{c};font-size:0.78rem;font-weight:700;">{arr}{abs(d['pct']):.2f}%</span>
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
    <span style="color:#888;font-size:0.75rem;">{lbl}</span>
    <span style="color:{c};font-size:0.78rem;font-weight:700;">{arr}{abs(d['pct']):.2f}%</span>
  </div>
  <div style="color:#e8e8e8;font-size:1.25rem;font-weight:700;margin-bottom:3px;">${d['last']:,.0f}</div>
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
font-family:'IBM Plex Mono',monospace;font-size:0.65rem;color:#555;">
  <span>◼ MONARCH PRO · GLOBAL RISK INTELLIGENCE</span>
  <span>Data: yfinance · alternative.me · Upstox  ·  Last updated: {datetime.now().strftime('%H:%M:%S')}</span>
  <span>Auto-refresh: {'ON (60s)' if auto_refresh else 'OFF'}</span>
</div>
""", unsafe_allow_html=True)
