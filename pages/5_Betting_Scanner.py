# pages/5_Arb_Scanner.py
# ─────────────────────────────────────────────────────────────────────────────
# MONARCH  ·  POLYMARKET STRATEGY SCANNER
#
# Polymarket-only deep market analysis. Uses every available Polymarket API
# endpoint to surface trading edges:
#
#   STRATEGY 1 · OVERROUND ARB
#     ask_YES + ask_NO < 1.00 → buy both sides for guaranteed profit
#
#   STRATEGY 2 · MOMENTUM DIVERGENCE
#     Price velocity + volume acceleration signals
#     Fast-moving markets where order flow creates predictable short-term drift
#
#   STRATEGY 3 · LIQUIDITY MISPRICING
#     Markets where bid-ask spread is anomalously wide relative to volume
#     Mid-price may be stale — VWAP fill reveals real cost vs mid
#
#   STRATEGY 4 · VOLUME ANOMALY
#     Unusual volume spikes relative to 24h average
#     Informed trading signals — price may not yet reflect new information
#
#   STRATEGY 5 · END-DATE DECAY
#     Near-expiry markets where time-value is underpriced
#     High annualised return even on small absolute edge
#
#   STRATEGY 6 · MARKET INEFFICIENCY SCORE
#     Composite score across all signals — ranks best overall opportunities
#
# ─────────────────────────────────────────────────────────────────────────────

import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timezone
import time
import re

st.set_page_config(page_title="Polymarket Strategy Scanner · MONARCH", layout="wide")

# ─────────────────────────────────────────────────────────────────────────────
# THEME
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600;700&display=swap');
:root {
    --bg:#0a0a0a;--surface:#111111;--border:#2a2a2a;
    --amber:#ff8c00;--amber2:#ffb347;
    --green:#00d084;--red:#ff3b3b;--blue:#1e90ff;--cyan:#00ccff;--purple:#cc88ff;
    --white:#e8e8e8;--white2:#c8c8c8;--muted:#888888;
    --mono:'IBM Plex Mono',monospace;
}
html,body,[data-testid="stAppViewContainer"],[data-testid="stAppViewContainer"]>section,.main .block-container{
    background-color:var(--bg)!important;color:var(--white)!important;font-family:var(--mono)!important;}
p,span,div,label,li,caption,[data-testid="stMarkdownContainer"] p,[data-testid="stMarkdownContainer"] span{
    color:var(--white2)!important;font-family:var(--mono)!important;}
[data-testid="stSidebar"],[data-testid="stSidebar"]>div{background-color:#060606!important;border-right:1px solid var(--border)!important;}
[data-testid="stSidebar"] p,[data-testid="stSidebar"] span,[data-testid="stSidebar"] label,[data-testid="stSidebar"] div{
    color:var(--muted)!important;font-size:0.67rem!important;letter-spacing:0.05em!important;}
h1{color:var(--amber)!important;font-size:1.05rem!important;font-weight:700!important;letter-spacing:0.18em!important;
   text-transform:uppercase!important;border-bottom:2px solid var(--amber)!important;padding-bottom:6px!important;margin-bottom:12px!important;}
h2{color:var(--amber2)!important;font-size:0.85rem!important;font-weight:600!important;letter-spacing:0.12em!important;
   text-transform:uppercase!important;border-bottom:1px solid #2a2a2a!important;padding-bottom:4px!important;}
[data-testid="metric-container"]{background:var(--surface)!important;border-radius:0!important;
    border:1px solid var(--border)!important;border-left:3px solid var(--amber)!important;padding:8px 12px!important;}
[data-testid="stMetricLabel"] p{color:var(--muted)!important;font-size:0.58rem!important;letter-spacing:0.12em!important;text-transform:uppercase!important;}
[data-testid="stMetricValue"]{color:var(--amber)!important;font-size:1.05rem!important;font-weight:700!important;}
[data-testid="stDataFrame"]{border:1px solid var(--amber)!important;border-radius:0!important;}
.stDataFrame thead tr th{background-color:#1a1200!important;color:var(--amber)!important;font-family:var(--mono)!important;
    font-size:0.60rem!important;font-weight:700!important;letter-spacing:0.14em!important;text-transform:uppercase!important;
    border-bottom:2px solid var(--amber)!important;padding:6px 10px!important;}
.stDataFrame tbody tr td{background-color:#0d0d0d!important;color:var(--white)!important;font-family:var(--mono)!important;
    font-size:0.68rem!important;border-bottom:1px solid #1a1a1a!important;padding:4px 10px!important;}
.stDataFrame tbody tr:nth-child(odd) td{background-color:#111111!important;}
.stDataFrame tbody tr:hover td{background-color:#1f1400!important;color:var(--amber)!important;}
.stButton>button{background:#140e00!important;color:var(--amber)!important;border:1px solid var(--amber)!important;
    border-radius:0!important;font-family:var(--mono)!important;font-size:0.70rem!important;
    font-weight:600!important;letter-spacing:0.1em!important;text-transform:uppercase!important;padding:6px 18px!important;}
.stButton>button:hover{background:var(--amber)!important;color:#000!important;}
.stTabs [data-baseweb="tab-list"]{background:#080808!important;border-bottom:2px solid var(--amber)!important;gap:0!important;}
.stTabs [data-baseweb="tab"]{background:transparent!important;color:var(--muted)!important;
    font-family:var(--mono)!important;font-size:0.63rem!important;font-weight:600!important;
    letter-spacing:0.1em!important;text-transform:uppercase!important;border-radius:0!important;
    border-right:1px solid var(--border)!important;padding:8px 14px!important;}
.stTabs [aria-selected="true"]{background:#1a1200!important;color:var(--amber)!important;
    border-bottom:3px solid var(--amber)!important;font-weight:700!important;}
.stProgress>div>div{background:var(--amber)!important;}
hr{border-color:#1e1e1e!important;margin:10px 0!important;}
.streamlit-expanderHeader,[data-testid="stExpander"] summary{
    background:var(--surface)!important;color:var(--amber)!important;
    font-family:var(--mono)!important;font-size:0.68rem!important;font-weight:600!important;
    letter-spacing:0.1em!important;text-transform:uppercase!important;border-radius:0!important;
    border:1px solid var(--border)!important;}
::-webkit-scrollbar{width:4px;height:4px;}
::-webkit-scrollbar-track{background:var(--bg);}
::-webkit-scrollbar-thumb{background:#333;}
::-webkit-scrollbar-thumb:hover{background:var(--amber);}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="background:#ff8c00;color:#000;font-family:'IBM Plex Mono',monospace;
font-size:0.65rem;font-weight:700;letter-spacing:0.18em;padding:5px 14px;
display:flex;justify-content:space-between;margin-bottom:12px;">
  <span>◼ MONARCH PRO — POLYMARKET STRATEGY SCANNER</span>
  <span>POLYMARKET · LIVE DATA · 6 STRATEGIES</span>
</div>
""", unsafe_allow_html=True)

st.title("⚡ POLYMARKET STRATEGY SCANNER")
st.caption("6 signal types · CLOB order-book VWAP · Gamma API enrichment · Momentum · Liquidity · Volume · Decay · Composite scoring")

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
POLY_BASE  = "https://clob.polymarket.com"
GAMMA_BASE = "https://gamma-api.polymarket.com"
POLY_FEE   = 0.02   # 2% of winnings (protocol-defined taker fee)

# Required headers — without these Polymarket returns 403/empty
HEADERS = {
    "Accept":     "application/json",
    "User-Agent": "Mozilla/5.0 (compatible; MonarchScanner/1.0)",
    "Origin":     "https://polymarket.com",
    "Referer":    "https://polymarket.com/",
}

DARK = dict(
    plot_bgcolor="#000", paper_bgcolor="#0a0a0a",
    font=dict(color="#e8e8e8", family="IBM Plex Mono", size=10),
    hoverlabel=dict(bgcolor="#1a1200", font_color="#ff8c00",
                    font_family="IBM Plex Mono", font_size=11),
)
_M = dict(t=44, b=28, l=60, r=24)

STRAT_COLORS = {
    "OVERROUND ARB":  "#00d084",
    "MOMENTUM":       "#1e90ff",
    "LIQUIDITY EDGE": "#cc88ff",
    "VOLUME ANOMALY": "#ff8c00",
    "DECAY TRADE":    "#00ccff",
    "INEFFICIENCY":   "#ffb347",
}


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _days_to_resolution(end_str: str) -> float:
    if not end_str:
        return float("inf")
    try:
        s = re.sub(r"\.\d+", "", str(end_str).replace("Z", "+00:00"))
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        delta = dt - datetime.now(timezone.utc)
        return max(0.0, delta.total_seconds() / 86400)
    except Exception:
        return float("inf")


def _annualised(net: float, days: float):
    if days <= 0 or days == float("inf") or days > 3650 or net <= 0:
        return None
    return ((1 + net) ** (365 / days) - 1) * 100


def _vwap_fill(asks: list, target_dollars: float):
    if not asks:
        return None
    spent = contracts = 0.0
    for price, n in asks:
        if price <= 0:
            continue
        avail = price * n
        take  = min(avail, target_dollars - spent)
        spent += take
        contracts += take / price
        if spent >= target_dollars - 1e-9:
            return spent / contracts if contracts else None
    return None


# ─────────────────────────────────────────────────────────────────────────────
# API  —  Polymarket CLOB + Gamma
# ─────────────────────────────────────────────────────────────────────────────

def _get(url: str, params: dict = None, timeout: int = 15):
    """
    Centralised GET with correct headers and detailed error capture.
    Returns (data_or_none, error_string_or_none).
    """
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=timeout)
        if r.status_code == 403:
            return None, f"403 Forbidden — {url}"
        if r.status_code == 404:
            return None, f"404 Not Found — {url}"
        r.raise_for_status()
        return r.json(), None
    except requests.exceptions.ConnectionError as e:
        return None, f"Connection error (no network?): {e}"
    except requests.exceptions.Timeout:
        return None, f"Timeout after {timeout}s — {url}"
    except requests.exceptions.HTTPError as e:
        return None, f"HTTP {e.response.status_code}: {url}"
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


# Track API health for the diagnostic panel
if "api_errors" not in st.session_state:
    st.session_state.api_errors = {}
if "api_ok"     not in st.session_state:
    st.session_state.api_ok = {}


@st.cache_data(ttl=30, show_spinner=False)
def fetch_poly_markets(limit: int = 300) -> list:
    """
    Primary: Gamma API /markets  (most reliable, richest data).
    Fallback: CLOB /markets.
    Gamma returns active binary markets with all pricing fields.
    """
    # ── Primary: Gamma API ───────────────────────────────────────────────────
    data, err = _get(f"{GAMMA_BASE}/markets",
                     params={"limit": limit, "active": "true", "closed": "false",
                             "order": "volume24hr", "ascending": "false"})
    if data is not None:
        markets = data if isinstance(data, list) else data.get("markets", data.get("data", []))
        if markets:
            st.session_state.api_ok["Gamma /markets"] = f"{len(markets)} markets"
            # Convert Gamma format → CLOB-compatible format expected by normalise_markets
            converted = []
            for m in markets:
                # Gamma markets have outcomes as separate token objects or flat yes/no fields
                yes_price = float(m.get("outcomePrices", [0.5])[0]  if m.get("outcomePrices") else
                                  m.get("bestBid", 0.5) or 0.5)
                no_price  = float(m.get("outcomePrices", [0.5, 0.5])[1] if m.get("outcomePrices") else
                                  1 - yes_price)
                # Clamp
                yes_price = max(0.01, min(0.99, yes_price))
                no_price  = max(0.01, min(0.99, no_price))

                tokens = m.get("tokens") or [
                    {"outcome": "YES", "token_id": m.get("clobTokenIds", ["",""])[0] if m.get("clobTokenIds") else "",
                     "price": yes_price},
                    {"outcome": "NO",  "token_id": m.get("clobTokenIds", ["",""])[1] if m.get("clobTokenIds") and len(m.get("clobTokenIds",[])) > 1 else "",
                     "price": no_price},
                ]
                converted.append({
                    "condition_id": m.get("conditionId") or m.get("condition_id", ""),
                    "question":     m.get("question", ""),
                    "active":       True,
                    "closed":       False,
                    "tokens":       tokens,
                    "volume_num":   float(m.get("volume24hr", 0) or 0),
                    "liquidity":    float(m.get("liquidity", 0) or 0),
                    "end_date_iso": m.get("endDate", ""),
                    "market_slug":  m.get("slug", ""),
                    "_gamma":       m,   # keep raw for enrichment
                })
            return converted

    st.session_state.api_errors["Gamma /markets"] = err or "empty response"

    # ── Fallback: CLOB API ────────────────────────────────────────────────────
    data, err2 = _get(f"{POLY_BASE}/markets",
                      params={"limit": limit, "active": "true", "closed": "false",
                              "order": "volume_num", "ascending": "false"})
    if data is not None:
        markets = data.get("data", [])
        if markets:
            st.session_state.api_ok["CLOB /markets"] = f"{len(markets)} markets"
            return markets

    st.session_state.api_errors["CLOB /markets"] = err2 or "empty response"
    return []


@st.cache_data(ttl=30, show_spinner=False)
def fetch_gamma_enrichment(condition_id: str) -> dict:
    """Fetch single market detail from Gamma for enrichment."""
    data, _ = _get(f"{GAMMA_BASE}/markets", params={"conditionId": condition_id})
    if data:
        markets = data if isinstance(data, list) else data.get("markets", [])
        return markets[0] if markets else {}
    return {}


@st.cache_data(ttl=15, show_spinner=False)
def fetch_orderbook(token_id: str) -> dict:
    if not token_id:
        return {"bids": [], "asks": []}
    data, err = _get(f"{POLY_BASE}/book", params={"token_id": token_id}, timeout=8)
    if data:
        bids = sorted([(float(b["price"]), float(b["size"])) for b in data.get("bids", [])], reverse=True)
        asks = sorted([(float(a["price"]), float(a["size"])) for a in data.get("asks", [])])
        return {"bids": bids, "asks": asks}
    if err:
        st.session_state.api_errors[f"OB:{token_id[:8]}"] = err
    return {"bids": [], "asks": []}


@st.cache_data(ttl=30, show_spinner=False)
def fetch_price_history(token_id: str, fidelity: int = 60) -> list:
    if not token_id:
        return []
    end_ts   = int(time.time())
    start_ts = end_ts - 86400 * 7
    data, _  = _get(f"{POLY_BASE}/prices-history",
                    params={"market": token_id, "startTs": start_ts,
                            "endTs": end_ts, "fidelity": fidelity},
                    timeout=10)
    return data.get("history", []) if data else []


@st.cache_data(ttl=60, show_spinner=False)
def fetch_trades(token_id: str, limit: int = 50) -> list:
    if not token_id:
        return []
    data, _ = _get(f"{POLY_BASE}/trades",
                   params={"market": token_id, "limit": limit}, timeout=8)
    return data.get("data", []) if data else []


# ─────────────────────────────────────────────────────────────────────────────
# MARKET NORMALISATION
# ─────────────────────────────────────────────────────────────────────────────

def normalise_markets(clob_raw: list, _unused=None) -> list:
    """
    Accepts markets in either:
      - CLOB native format (tokens array with YES/NO outcomes)
      - Gamma-converted format (from fetch_poly_markets, includes _gamma key)
    Produces a unified dict for all strategy engines.
    """
    markets = []
    for m in clob_raw:
        if m.get("closed") or not m.get("active", True):
            continue

        tokens = m.get("tokens", [])
        yes_tok = next((t for t in tokens if str(t.get("outcome", "")).upper() == "YES"), None)
        no_tok  = next((t for t in tokens if str(t.get("outcome", "")).upper() == "NO"),  None)

        # If tokens missing/malformed, try to get prices from _gamma raw
        g = m.get("_gamma", {})
        if yes_tok is None:
            op = g.get("outcomePrices", [])
            yp = float(op[0]) if op else float(g.get("bestBid", 0.5) or 0.5)
            yes_tok = {"outcome": "YES", "token_id": "", "price": yp}
        if no_tok is None:
            op = g.get("outcomePrices", [])
            np_ = float(op[1]) if len(op) > 1 else (1 - float(yes_tok.get("price", 0.5)))
            no_tok = {"outcome": "NO", "token_id": "", "price": np_}

        yes_price = max(0.01, min(0.99, float(yes_tok.get("price", 0.5) or 0.5)))
        no_price  = max(0.01, min(0.99, float(no_tok.get("price",  0.5) or 0.5)))

        # Token IDs — needed for order book + history fetches
        yes_tid = yes_tok.get("token_id", "")
        no_tid  = no_tok.get("token_id", "")
        # Try clobTokenIds from Gamma if CLOB token_id is empty
        ctids = g.get("clobTokenIds", [])
        if not yes_tid and len(ctids) > 0: yes_tid = ctids[0]
        if not no_tid  and len(ctids) > 1: no_tid  = ctids[1]

        # Pricing enrichment from Gamma raw data
        best_bid   = float(g.get("bestBid",        yes_price - 0.01) or yes_price - 0.01)
        best_ask   = float(g.get("bestAsk",         yes_price + 0.01) or yes_price + 0.01)
        vol_24h    = float(g.get("volume24hr",      m.get("volume_num", 0) or 0) or 0)
        vol_total  = float(g.get("volumeClob",      m.get("volume_num", 0) or 0) or vol_24h * 30)
        liquidity  = float(g.get("liquidity",       m.get("liquidity", 0)  or 0) or 0)
        last_price = float(g.get("lastTradePrice",  yes_price) or yes_price)
        spread_val = max(0.0, best_ask - best_bid)
        category   = g.get("groupItemTitle") or g.get("category") or ""

        end_str     = (m.get("end_date_iso") or m.get("end_date") or
                       g.get("endDate") or "")
        days_to_res = _days_to_resolution(end_str)

        markets.append({
            "cid":          m.get("condition_id", g.get("conditionId", "")),
            "question":     m.get("question", g.get("question", "Unnamed market")),
            "yes_price":    yes_price,
            "no_price":     no_price,
            "yes_token_id": yes_tid,
            "no_token_id":  no_tid,
            "best_bid":     best_bid,
            "best_ask":     best_ask,
            "spread":       spread_val,
            "vol_24h":      vol_24h,
            "vol_total":    vol_total,
            "liquidity":    liquidity,
            "last_price":   last_price,
            "overround":    yes_price + no_price,
            "days_to_res":  days_to_res,
            "end_date":     end_str,
            "category":     category,
            "url":          f"https://polymarket.com/event/{m.get('market_slug', g.get('slug',''))}",
        })

    return markets


# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY ENGINES
# ─────────────────────────────────────────────────────────────────────────────

def strategy_overround_arb(m, ob_yes, ob_no, lot):
    yes_vwap = _vwap_fill(ob_yes.get("asks", []), lot) or m.get("best_ask")
    no_vwap  = _vwap_fill(ob_no.get("asks",  []), lot) or (1.0 - m.get("best_bid", 0))
    if yes_vwap is None or no_vwap is None:
        return None
    cost  = yes_vwap + no_vwap
    if cost >= 1.0:
        return None
    gross    = 1.0 - cost
    fee_cost = POLY_FEE * (1.0 - min(yes_vwap, no_vwap))
    net      = gross - fee_cost
    if net <= 0:
        return None
    days = m["days_to_res"]
    return {
        "strategy": "OVERROUND ARB", "risk": "RISK-FREE",
        "question": m["question"],   "category": m["category"],
        "action_1": f"BUY YES @ {yes_vwap:.4f}", "action_2": f"BUY NO  @ {no_vwap:.4f}",
        "yes_vwap": yes_vwap, "no_vwap": no_vwap,
        "cost_per_unit": cost, "gross_profit": gross, "fee_cost": fee_cost,
        "net_profit": net, "net_pct": net * 100, "annualised_pct": _annualised(net, days),
        "days_to_res": days, "lot": lot, "total_cost": cost * lot, "total_profit": net * lot,
        "score": net * 100 * (1 + 1 / max(days, 1)),
        "url": m["url"], "cid": m["cid"], "yes_token_id": m["yes_token_id"],
    }


def strategy_momentum(m, history, lot):
    if len(history) < 6:
        return None
    prices = [float(h.get("p", 0)) for h in history if h.get("p")]
    if len(prices) < 6:
        return None
    recent   = prices[-1]
    baseline = prices[max(0, len(prices) - 24)]
    velocity = recent - baseline
    h6  = prices[-1] - prices[max(0, len(prices) - 6)]
    h12 = prices[max(0, len(prices) - 6)] - prices[max(0, len(prices) - 12)]
    accel = h6 - h12
    if abs(velocity) < 0.03:
        return None
    vol_score = min(m["vol_24h"] / max(m["vol_total"] / 30 + 1, 1), 3.0)
    signal    = abs(velocity) * (1 + max(accel, 0)) * vol_score
    direction = "YES" if velocity > 0 else "NO"
    entry     = recent if direction == "YES" else (1 - recent)
    expected  = min(entry + abs(velocity) * 0.5, 0.99)
    net       = (expected - entry) * (1 - POLY_FEE)
    if net <= 0.005:
        return None
    days = m["days_to_res"]
    return {
        "strategy": "MOMENTUM", "risk": "DIRECTIONAL",
        "question": m["question"], "category": m["category"],
        "action_1": f"BUY {direction} @ {entry:.4f}",
        "action_2": f"TARGET {expected:.4f}  (+{abs(velocity)*50:.1f}¢ continuation)",
        "entry_price": entry, "target_price": expected,
        "velocity_24h": velocity, "acceleration": accel, "vol_ratio": vol_score,
        "signal_strength": signal, "net_profit": net, "net_pct": net * 100,
        "annualised_pct": _annualised(net, days), "days_to_res": days, "lot": lot,
        "total_cost": entry * lot, "total_profit": net * lot, "score": signal * 10,
        "url": m["url"], "cid": m["cid"], "yes_token_id": m["yes_token_id"],
        "history": history,
    }


def strategy_liquidity_edge(m, ob_yes, ob_no, lot):
    spread = m["spread"]
    if spread < 0.05:
        return None
    yes_mid  = m["yes_price"]
    yes_vwap = _vwap_fill(ob_yes.get("asks", []), lot) or m.get("best_ask")
    no_vwap  = _vwap_fill(ob_no.get("asks",  []), lot) or (1 - m.get("best_bid", 0))
    if yes_vwap is None or no_vwap is None:
        return None
    yes_edge = yes_mid - yes_vwap
    no_edge  = (1 - yes_mid) - no_vwap
    best_edge   = max(yes_edge, no_edge)
    best_side   = "YES" if yes_edge >= no_edge else "NO"
    entry_vwap  = yes_vwap if best_side == "YES" else no_vwap
    mid_for_side = yes_mid if best_side == "YES" else (1 - yes_mid)
    if best_edge < 0.02:
        return None
    net  = best_edge * (1 - POLY_FEE)
    days = m["days_to_res"]
    return {
        "strategy": "LIQUIDITY EDGE", "risk": "MEAN-REVERT",
        "question": m["question"], "category": m["category"],
        "action_1": f"BUY {best_side} @ VWAP {entry_vwap:.4f}",
        "action_2": f"MID = {mid_for_side:.4f}  SPREAD = {spread:.4f}",
        "entry_price": entry_vwap, "mid_price": mid_for_side, "spread": spread,
        "edge_yes": yes_edge, "edge_no": no_edge, "best_edge": best_edge,
        "net_profit": net, "net_pct": net * 100, "annualised_pct": _annualised(net, days),
        "days_to_res": days, "lot": lot, "total_cost": entry_vwap * lot, "total_profit": net * lot,
        "score": best_edge * spread * 1000,
        "url": m["url"], "cid": m["cid"], "yes_token_id": m["yes_token_id"],
    }


def strategy_volume_anomaly(m, trades, lot):
    if len(trades) < 5:
        return None
    amounts = [float(t.get("size", 0)) for t in trades if t.get("size")]
    if not amounts:
        return None
    recent_10 = np.mean(amounts[:10]) if len(amounts) >= 10 else np.mean(amounts)
    baseline  = np.mean(amounts[-20:]) if len(amounts) >= 20 else recent_10
    vol_ratio = recent_10 / (baseline + 1e-9)
    if vol_ratio < 2.0:
        return None
    recent_prices = [float(t.get("price", 0)) for t in trades[:10] if t.get("price")]
    if len(recent_prices) < 3:
        return None
    price_drift = recent_prices[0] - recent_prices[-1]
    direction   = "YES" if price_drift > 0 else "NO"
    entry       = m["yes_price"] if direction == "YES" else m["no_price"]
    target      = min(entry + abs(price_drift) * 0.5, 0.98)
    net         = (target - entry) * (1 - POLY_FEE)
    if net < 0.01:
        return None
    days = m["days_to_res"]
    return {
        "strategy": "VOLUME ANOMALY", "risk": "DIRECTIONAL",
        "question": m["question"], "category": m["category"],
        "action_1": f"BUY {direction} @ {entry:.4f}  (follow informed flow)",
        "action_2": f"TARGET {target:.4f}  (volume ratio: {vol_ratio:.1f}×)",
        "direction": direction, "entry_price": entry, "target_price": target,
        "vol_ratio": vol_ratio, "price_drift": price_drift,
        "recent_avg_size": recent_10, "baseline_size": baseline,
        "net_profit": net, "net_pct": net * 100, "annualised_pct": _annualised(net, days),
        "days_to_res": days, "lot": lot, "total_cost": entry * lot, "total_profit": net * lot,
        "score": vol_ratio * abs(price_drift) * 100,
        "url": m["url"], "cid": m["cid"], "yes_token_id": m["yes_token_id"],
    }


def strategy_decay_trade(m, lot):
    days = m["days_to_res"]
    if days > 14 or days <= 0:
        return None
    yes_p = m["yes_price"]
    if yes_p >= 0.88:
        entry = m.get("best_ask") or yes_p
        net   = (1.0 - entry) * (1 - POLY_FEE)
        side  = "YES"
    elif yes_p <= 0.12:
        entry = 1.0 - (m.get("best_bid") or yes_p)
        net   = (1.0 - entry) * (1 - POLY_FEE)
        side  = "NO"
    else:
        return None
    if net <= 0:
        return None
    ann = _annualised(net, days)
    if ann is None or ann < 50:
        return None
    return {
        "strategy": "DECAY TRADE", "risk": "HIGH-PROB",
        "question": m["question"], "category": m["category"],
        "action_1": f"BUY {side} @ {entry:.4f}  ({days:.1f}d remaining)",
        "action_2": "HOLD TO RESOLUTION  (expected: $1.00 payout)",
        "entry_price": entry, "side": side, "yes_price": yes_p,
        "net_profit": net, "net_pct": net * 100, "annualised_pct": ann,
        "days_to_res": days, "lot": lot, "total_cost": entry * lot, "total_profit": net * lot,
        "score": (ann or 0) / 10,
        "url": m["url"], "cid": m["cid"], "yes_token_id": m["yes_token_id"],
    }


def strategy_inefficiency_score(m, ob_yes, ob_no, lot):
    signals = {}
    over_dev = abs(m["overround"] - 1.0)
    signals["overround_dev"] = min(over_dev * 10, 1.0)
    if m["liquidity"] > 1000:
        signals["spread_anomaly"] = min(m["spread"] / (m["liquidity"] / 10000 + 0.01), 1.0)
    else:
        signals["spread_anomaly"] = 0.0
    if m["liquidity"] > 0:
        signals["vol_liq"] = min(m["vol_24h"] / m["liquidity"] / 2, 1.0)
    else:
        signals["vol_liq"] = 0.0
    yes_depth = sum(s for _, s in ob_yes.get("asks", [])[:5])
    no_depth  = sum(s for _, s in ob_no.get("asks",  [])[:5])
    total_dep = yes_depth + no_depth
    signals["ob_imbalance"] = abs(yes_depth - no_depth) / total_dep if total_dep > 0 else 0.0
    signals["last_trade_dev"] = min(abs(m["yes_price"] - m["last_price"]) * 5, 1.0)
    composite = float(np.mean(list(signals.values())))
    if composite < 0.25:
        return None
    entry_yes  = m.get("best_ask") or m["yes_price"]
    entry_no   = 1.0 - (m.get("best_bid") or m["yes_price"])
    best_entry = min(entry_yes, entry_no)
    best_side  = "YES" if entry_yes <= entry_no else "NO"
    net        = (0.5 - best_entry) * (1 - POLY_FEE)
    days       = m["days_to_res"]
    return {
        "strategy": "INEFFICIENCY", "risk": "SPECULATIVE",
        "question": m["question"], "category": m["category"],
        "action_1": f"INVESTIGATE {best_side}  (score: {composite:.2f})",
        "action_2": "5 signals flagged — verify manually before trading",
        "composite": composite, "signals": signals,
        "net_profit": max(net, 0.001), "net_pct": composite * 20,
        "annualised_pct": _annualised(max(net, 0.001), days),
        "days_to_res": days, "lot": lot,
        "total_cost": best_entry * lot, "total_profit": max(net, 0.001) * lot,
        "score": composite * 100,
        "url": m["url"], "cid": m["cid"], "yes_token_id": m["yes_token_id"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────────────────────────────────────────

def price_history_chart(history, signal=None):
    if not history:
        return go.Figure()
    ts = [datetime.fromtimestamp(h.get("t", 0), tz=timezone.utc) for h in history]
    ps = [float(h.get("p", 0)) for h in history]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ts, y=ps, mode="lines",
        line=dict(color="#ff8c00", width=1.8),
        fill="tozeroy", fillcolor="rgba(255,140,0,0.06)",
        name="YES Price",
        hovertemplate="%{x|%b %d %H:%M}<br>%{y:.4f}<extra></extra>",
    ))
    if signal and signal.get("entry_price"):
        fig.add_hline(y=signal["entry_price"], line_dash="dot", line_color="#1e90ff",
                      annotation_text="Entry", annotation_font=dict(color="#1e90ff", size=8))
    if signal and signal.get("target_price"):
        fig.add_hline(y=signal["target_price"], line_dash="dot", line_color="#00d084",
                      annotation_text="Target", annotation_font=dict(color="#00d084", size=8))
    fig.update_layout(
        title=dict(text="PRICE HISTORY (7D)", font=dict(color="#ff8c00", size=11)),
        xaxis=dict(gridcolor="#1a1a1a", tickfont=dict(color="#888")),
        yaxis=dict(gridcolor="#1a1a1a", tickfont=dict(color="#888"),
                   tickformat=".3f", range=[0, 1]),
        margin=_M, **DARK,
    )
    return fig


def orderbook_chart(ob_yes, ob_no):
    fig = go.Figure()
    for ob, side, col in [(ob_yes, "YES", "#00d084"), (ob_no, "NO", "#ff3b3b")]:
        asks = ob.get("asks", [])[:20]
        if not asks:
            continue
        prices = [p for p, _ in asks]
        cum    = np.cumsum([s for _, s in asks])
        fig.add_trace(go.Scatter(
            x=prices, y=cum, mode="lines+markers",
            name=f"{side} Ask", line=dict(color=col, width=1.5),
            marker=dict(size=4, color=col),
            fill="tozeroy", fillcolor=col + "18",
            hovertemplate=f"{side}: %{{x:.4f}}<br>Cum: %{{y:.0f}}<extra></extra>",
        ))
    fig.update_layout(
        title=dict(text="ORDER BOOK DEPTH", font=dict(color="#ff8c00", size=11)),
        xaxis=dict(title="Price", gridcolor="#1a1a1a", tickfont=dict(color="#888")),
        yaxis=dict(title="Cumulative Contracts", gridcolor="#1a1a1a", tickfont=dict(color="#888")),
        margin=_M, **DARK,
    )
    return fig


def payoff_chart(result, lot):
    strat = result["strategy"]
    if strat == "OVERROUND ARB":
        y_v   = result.get("yes_vwap", 0.5)
        n_v   = result.get("no_vwap",  0.5)
        pl_y  = ((1 - y_v) * (1 - POLY_FEE) - n_v) * lot
        pl_n  = ((1 - n_v) * (1 - POLY_FEE) - y_v) * lot
        outcomes, pls = ["YES RESOLVES", "NO RESOLVES"], [pl_y, pl_n]
    else:
        entry  = result.get("entry_price", 0.5)
        net    = result.get("net_profit", 0)
        target = result.get("target_price", min(entry + net, 0.99))
        pl_hit  = (target - entry) * (1 - POLY_FEE) * lot
        pl_miss = -(entry * 0.5) * lot
        outcomes, pls = ["TARGET HIT", "TARGET MISSED"], [pl_hit, pl_miss]
    colors = ["#00d084" if p >= 0 else "#ff3b3b" for p in pls]
    fig = go.Figure(go.Bar(
        x=outcomes, y=pls, marker_color=colors, marker_line_width=1,
        text=[f"${p:+.2f}" for p in pls], textposition="outside",
        textfont=dict(color="#e8e8e8", size=11, family="IBM Plex Mono"),
    ))
    fig.add_hline(y=0, line_color="#555", line_width=1)
    fig.update_layout(
        title=dict(text="PAYOFF DIAGRAM", font=dict(color="#ff8c00", size=11)),
        xaxis=dict(gridcolor="#1a1a1a", tickfont=dict(color="#888")),
        yaxis=dict(gridcolor="#1a1a1a", tickprefix="$", tickfont=dict(color="#888")),
        margin=_M, showlegend=False, **DARK,
    )
    return fig


def annualised_chart(result):
    net = result.get("net_profit", 0)
    if net <= 0:
        return go.Figure()
    days_r = np.linspace(1, max(365, result.get("days_to_res", 30) * 2), 300)
    ann_r  = ((1 + net) ** (365 / days_r) - 1) * 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=days_r, y=ann_r, mode="lines",
                             line=dict(color="#ff8c00", width=2), name="Ann. Return"))
    d = result.get("days_to_res")
    if d and d < float("inf") and d > 0:
        a = ((1 + net) ** (365 / d) - 1) * 100
        fig.add_trace(go.Scatter(
            x=[d], y=[a], mode="markers+text",
            marker=dict(color="#00d084", size=10, symbol="diamond"),
            text=[f"  {a:.0f}%"], textfont=dict(color="#00d084", size=9),
            textposition="middle right", name="This Trade",
        ))
    fig.add_hline(y=5, line_dash="dot", line_color="#333",
                  annotation_text="5% risk-free", annotation_font=dict(color="#555", size=8))
    fig.update_layout(
        title=dict(text="ANNUALISED RETURN vs DAYS", font=dict(color="#ff8c00", size=11)),
        xaxis=dict(title="Days to Resolution", gridcolor="#1a1a1a", tickfont=dict(color="#888")),
        yaxis=dict(title="Annualised %", gridcolor="#1a1a1a", tickfont=dict(color="#888")),
        margin=_M, **DARK,
    )
    return fig


def signals_radar_chart(signals):
    cats = [k.replace("_", " ").upper() for k in signals]
    vals = list(signals.values())
    cats += [cats[0]]; vals += [vals[0]]
    fig = go.Figure(go.Scatterpolar(
        r=vals, theta=cats, fill="toself",
        line=dict(color="#ff8c00", width=2),
        fillcolor="rgba(255,140,0,0.12)",
    ))
    fig.update_layout(
        polar=dict(bgcolor="#0a0a0a",
                   radialaxis=dict(visible=True, range=[0, 1], gridcolor="#2a2a2a",
                                   tickfont=dict(color="#555", size=7)),
                   angularaxis=dict(gridcolor="#2a2a2a", tickfont=dict(color="#888", size=8))),
        title=dict(text="INEFFICIENCY SIGNALS", font=dict(color="#ff8c00", size=11)),
        margin=dict(t=50, b=30, l=50, r=50), **DARK,
    )
    return fig


def size_sensitivity_chart(result, lot):
    net  = result.get("net_profit", 0)
    lots = np.linspace(10, lot * 3, 200)
    fig  = go.Figure(go.Scatter(
        x=lots, y=net * lots, mode="lines",
        line=dict(color="#1e90ff", width=2),
        hovertemplate="Size: $%{x:,.0f}<br>Profit: $%{y:.2f}<extra></extra>",
    ))
    fig.add_vline(x=lot, line_color="#ff8c00", line_dash="dot",
                  annotation_text="Current", annotation_font=dict(color="#ff8c00", size=8))
    fig.update_layout(
        title=dict(text="PROFIT vs POSITION SIZE", font=dict(color="#ff8c00", size=11)),
        xaxis=dict(title="Position ($)", gridcolor="#1a1a1a", tickprefix="$", tickfont=dict(color="#888")),
        yaxis=dict(title="Net Profit ($)", gridcolor="#1a1a1a", tickprefix="$", tickfont=dict(color="#888")),
        margin=_M, **DARK,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙ SCANNER CONFIG")
    st.divider()
    lot_size   = st.number_input("Position size ($)", min_value=10, max_value=100_000, value=1_000, step=100)
    max_days   = st.number_input("Max days to resolution", min_value=1, max_value=365, value=90)
    min_volume = st.number_input("Min 24h volume ($)", min_value=0, max_value=1_000_000, value=500, step=500)
    st.divider()
    st.markdown("**STRATEGIES**")
    run_s1 = st.checkbox("S1 · Overround Arb",    value=True)
    run_s2 = st.checkbox("S2 · Momentum",          value=True)
    run_s3 = st.checkbox("S3 · Liquidity Edge",    value=True)
    run_s4 = st.checkbox("S4 · Volume Anomaly",    value=True)
    run_s5 = st.checkbox("S5 · Decay Trade",        value=True)
    run_s6 = st.checkbox("S6 · Inefficiency Score", value=True)
    st.divider()
    auto_refresh = st.checkbox("Auto-refresh every 60s", value=False)
    fetch_ob     = st.checkbox("Fetch order books (slower)", value=True)
    fetch_hist   = st.checkbox("Fetch price history (slower)", value=True)
    st.divider()
    st.markdown("""
<div style="font-size:0.58rem;color:#555;line-height:1.6;">
<b>DATA SOURCES</b><br>
Polymarket CLOB API<br>
Polymarket Gamma API<br>
CLOB /book  (order depth)<br>
CLOB /prices-history  (7d)<br>
CLOB /trades  (recent 50)<br><br>
<b>FEE MODEL</b><br>
Polymarket: 2% of winnings<br><br>
<b>RISK LABELS</b><br>
RISK-FREE · guaranteed at resolution<br>
DIRECTIONAL · depends on price move<br>
MEAN-REVERT · spread must close<br>
HIGH-PROB · near-certain, not locked<br>
SPECULATIVE · research signal only<br>
</div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SCAN
# ─────────────────────────────────────────────────────────────────────────────

col_btn, col_info = st.columns([1, 3])
with col_btn:
    run_scan = st.button("🔍  SCAN POLYMARKET", use_container_width=True, type="primary")
with col_info:
    active_strats = [n for on, n in [(run_s1,"S1"),(run_s2,"S2"),(run_s3,"S3"),
                                      (run_s4,"S4"),(run_s5,"S5"),(run_s6,"S6")] if on]
    st.markdown(f"""
<div style="padding:8px 12px;background:#111;border-left:3px solid #ff8c00;font-size:0.65rem;color:#888;">
Size: <span style="color:#ff8c00">${lot_size:,}</span> ·
Max days: <span style="color:#ff8c00">{max_days}d</span> ·
Min vol: <span style="color:#ff8c00">${min_volume:,}</span> ·
Strategies: <span style="color:#ff8c00">{' · '.join(active_strats)}</span> ·
Fee: <span style="color:#ff8c00">2% of profit</span>
</div>""", unsafe_allow_html=True)

for key in ("results", "scan_ts", "scan_mkts"):
    if key not in st.session_state:
        st.session_state[key] = [] if key == "results" else (None if key == "scan_ts" else 0)

if auto_refresh and st.session_state.scan_ts:
    if (time.time() - st.session_state.scan_ts) > 60:
        run_scan = True

if run_scan:
    # Clear previous error/ok state
    st.session_state.api_errors = {}
    st.session_state.api_ok     = {}

    with st.spinner("Connecting to Polymarket…"):
        prog = st.progress(0.0, text="Fetching markets from Gamma API…")
        clob_raw = fetch_poly_markets(limit=300)
        prog.progress(0.30, text="Normalising markets…")

        # ── API Diagnostics ─────────────────────────────────────────────────
        ok_items  = st.session_state.api_ok
        err_items = st.session_state.api_errors

        if err_items and not ok_items:
            # All APIs failed — show prominent error
            err_html = "".join(
                f'<div style="color:#ff3b3b;font-size:.65rem;margin:2px 0;">'
                f'✗ <b>{k}</b>: {v}</div>'
                for k, v in err_items.items()
            )
            st.error(f"""
**API Connection Failed — 0 markets loaded.**

All Polymarket endpoints returned errors. This usually means:
- Your network blocks requests to polymarket.com / gamma-api.polymarket.com
- You need a VPN or proxy
- Polymarket is temporarily down

**Errors:**
""")
            st.markdown(f'<div style="background:#1a0000;border:1px solid #ff3b3b;padding:10px;border-radius:0;">'
                        f'{err_html}</div>', unsafe_allow_html=True)
            st.stop()
        elif err_items:
            # Some failed — show as warning
            for k, v in err_items.items():
                st.warning(f"⚠ {k}: {v}")

        if ok_items:
            ok_html = " · ".join(f'<span style="color:#00d084">✓ {k}: {v}</span>'
                                  for k, v in ok_items.items())
            st.markdown(f'<div style="font-size:.62rem;padding:4px 0;">{ok_html}</div>',
                        unsafe_allow_html=True)

        markets = normalise_markets(clob_raw)
        prog.progress(0.33, text=f"Loaded {len(markets)} raw markets. Applying filters…")

        markets_before = len(markets)
        markets = [m for m in markets
                   if m["days_to_res"] <= max_days and m["vol_24h"] >= min_volume]

        if len(markets) == 0 and markets_before > 0:
            st.warning(
                f"✓ Fetched {markets_before} markets but **all were filtered out**. "
                f"Try: lower 'Min 24h volume' (currently ${min_volume:,}) or raise 'Max days' (currently {max_days}d)."
            )

        all_results = []
        n = len(markets)
        prog.progress(0.35, text=f"Running {len([x for x in [run_s1,run_s2,run_s3,run_s4,run_s5,run_s6] if x])} strategies on {n} markets…")

        for i, m in enumerate(markets):
            prog.progress(0.35 + 0.60 * (i / max(n, 1)),
                          text=f"[{i+1}/{n}] {m['question'][:48]}…")

            ob_yes = ob_no = {"bids": [], "asks": []}
            if fetch_ob and (run_s1 or run_s3 or run_s6):
                ob_yes = fetch_orderbook(m["yes_token_id"])
                ob_no  = fetch_orderbook(m["no_token_id"])

            if run_s1:
                r = strategy_overround_arb(m, ob_yes, ob_no, lot_size)
                if r: all_results.append(r)

            if run_s2 and fetch_hist:
                hist = fetch_price_history(m["yes_token_id"], fidelity=60)
                r = strategy_momentum(m, hist, lot_size)
                if r: all_results.append(r)

            if run_s3 and fetch_ob:
                r = strategy_liquidity_edge(m, ob_yes, ob_no, lot_size)
                if r: all_results.append(r)

            if run_s4:
                trades = fetch_trades(m["yes_token_id"], limit=50)
                r = strategy_volume_anomaly(m, trades, lot_size)
                if r: all_results.append(r)

            if run_s5:
                r = strategy_decay_trade(m, lot_size)
                if r: all_results.append(r)

            if run_s6 and fetch_ob:
                r = strategy_inefficiency_score(m, ob_yes, ob_no, lot_size)
                if r: all_results.append(r)

        all_results.sort(key=lambda x: x.get("score", 0), reverse=True)
        st.session_state.results   = all_results
        st.session_state.scan_ts   = time.time()
        st.session_state.scan_mkts = len(markets)

        if all_results:
            prog.progress(1.0, text=f"✓ Done — {len(all_results)} signals across {len(markets)} markets")
        else:
            prog.progress(1.0, text=f"Scan complete — no signals triggered. {len(markets)} markets scanned.")
            if markets:
                st.info(
                    f"Scanned {len(markets)} markets but no strategies triggered. "
                    f"Markets may be efficiently priced. Try: enable more strategies, "
                    f"lower min volume, or increase max days to resolution."
                )


# ─────────────────────────────────────────────────────────────────────────────
# RESULTS DISPLAY
# ─────────────────────────────────────────────────────────────────────────────

results = st.session_state.results

if st.session_state.scan_ts:
    age = int(time.time() - st.session_state.scan_ts)
    by_strat = {}
    for r in results:
        by_strat[r["strategy"]] = by_strat.get(r["strategy"], 0) + 1
    counts_str = " · ".join(f"{k}: {v}" for k, v in sorted(by_strat.items()))
    st.caption(f"Last scan: {age}s ago · {len(results)} signals · "
               f"{st.session_state.scan_mkts} markets · {counts_str}")

if not results:
    if st.session_state.scan_ts:
        st.info("No signals found. Try lowering filters (min volume, max days) or enabling more strategies.")
    else:
        st.info("Click **SCAN POLYMARKET** to search for live opportunities across all 6 strategies.")
    st.stop()

# Metrics row
risk_free  = [r for r in results if r.get("risk") == "RISK-FREE"]
directional = [r for r in results if r.get("risk") == "DIRECTIONAL"]
best = results[0]

m1, m2, m3, m4, m5, m6 = st.columns(6)
with m1: st.metric("RISK-FREE ARBS",  len(risk_free))
with m2: st.metric("DIRECTIONAL",      len(directional))
with m3: st.metric("TOTAL SIGNALS",    len(results))
with m4: st.metric("TOP P&L",          f"${best.get('total_profit',0):.2f}")
with m5:
    ann = best.get("annualised_pct")
    st.metric("TOP ANNUALISED", f"{ann:.0f}%" if ann else "N/A")
with m6: st.metric("TOP STRATEGY",     best["strategy"])

st.divider()

# Filter bar
cf1, cf2, cf3 = st.columns(3)
with cf1:
    strat_filter = st.multiselect("Filter strategy", options=list(STRAT_COLORS.keys()),
                                   default=list(STRAT_COLORS.keys()))
with cf2:
    risk_filter = st.multiselect("Filter risk",
                                  options=["RISK-FREE","DIRECTIONAL","MEAN-REVERT","HIGH-PROB","SPECULATIVE"],
                                  default=["RISK-FREE","DIRECTIONAL","MEAN-REVERT","HIGH-PROB","SPECULATIVE"])
with cf3:
    sort_by = st.selectbox("Sort by", ["Score","Net Profit $","Annualised %","Days to Resolution"])

filtered = [r for r in results
            if r.get("strategy","") in strat_filter and r.get("risk","") in risk_filter]

if   sort_by == "Net Profit $":       filtered.sort(key=lambda x: x.get("total_profit", 0), reverse=True)
elif sort_by == "Annualised %":        filtered.sort(key=lambda x: x.get("annualised_pct") or 0, reverse=True)
elif sort_by == "Days to Resolution":  filtered.sort(key=lambda x: x.get("days_to_res", 999))
else:                                  filtered.sort(key=lambda x: x.get("score", 0), reverse=True)

# Summary table
st.markdown("""<div style="color:#ff8c00;font-size:.75rem;font-weight:700;letter-spacing:.1em;margin-bottom:8px;">
◼ ALL SIGNALS — RANKED</div>""", unsafe_allow_html=True)

rows = []
for r in filtered:
    rows.append({
        "Strategy":  r["strategy"],
        "Risk":      r["risk"],
        "Market":    r["question"][:55] + ("…" if len(r["question"]) > 55 else ""),
        "Category":  r.get("category","")[:18],
        "Entry":     r["action_1"][:42],
        "Capital $": f"${r.get('total_cost', 0):.2f}",
        "Net P&L $": f"${r.get('total_profit', 0):.2f}",
        "Net %":     f"{r.get('net_pct', 0):.3f}%",
        "Ann. %":    f"{r['annualised_pct']:.0f}%" if r.get("annualised_pct") else "—",
        "Days":      f"{r.get('days_to_res', 0):.1f}d",
        "Score":     f"{r.get('score', 0):.1f}",
    })

if rows:
    def _color_row(row):
        styles = [""] * len(row)
        idx   = list(row.index)
        risk  = str(row.get("Risk",""))
        strat = str(row.get("Strategy",""))
        col   = STRAT_COLORS.get(strat, "#ff8c00")
        if "RISK-FREE"  in risk: styles[idx.index("Risk")] = "background-color:#001a0a;color:#00d084;font-weight:700"
        elif "HIGH-PROB" in risk: styles[idx.index("Risk")] = "color:#00ccff;font-weight:700"
        else:                     styles[idx.index("Risk")] = f"color:{col}"
        styles[idx.index("Strategy")] = f"color:{col};font-weight:700"
        if "RISK-FREE" in risk:
            styles[idx.index("Net P&L $")] = "color:#00d084;font-weight:700"
        return styles

    df = pd.DataFrame(rows)
    st.dataframe(df.style.apply(_color_row, axis=1), use_container_width=True, hide_index=True)

st.divider()

# Detailed tabs
st.markdown("""<div style="color:#ff8c00;font-size:.75rem;font-weight:700;letter-spacing:.1em;margin-bottom:8px;">
◼ DETAILED ANALYSIS — TOP 8 SIGNALS</div>""", unsafe_allow_html=True)

display = filtered[:8]
if not display:
    st.info("No signals match current filters.")
    st.stop()

tabs = st.tabs([f"{r['strategy'].split()[0]} #{i+1}" for i, r in enumerate(display)])

for tab, result in zip(tabs, display):
    with tab:
        strat      = result["strategy"]
        col_accent = STRAT_COLORS.get(strat, "#ff8c00")
        is_rf      = result["risk"] == "RISK-FREE"
        net_d      = result.get("total_profit", 0)
        cost_d     = result.get("total_cost",   0)
        ann_str    = f"{result['annualised_pct']:.1f}%" if result.get("annualised_pct") else "N/A"
        days_str   = f"{result['days_to_res']:.1f}" if result.get("days_to_res") != float("inf") else "?"

        # Hero card
        st.markdown(f"""
<div style="background:#0d0d0d;border:1px solid {col_accent};border-left:5px solid {col_accent};
padding:16px 20px;font-family:'IBM Plex Mono',monospace;margin-bottom:16px;">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:14px;">
    <div style="flex:3;min-width:220px;">
      <div style="color:#555;font-size:.50rem;letter-spacing:.14em;">
        {strat} · {result['risk']} · {result.get('category','POLYMARKET').upper()}</div>
      <div style="color:#e8e8e8;font-size:.88rem;font-weight:700;margin-top:5px;line-height:1.35;">
        {result['question'][:95]}{'…' if len(result['question'])>95 else ''}</div>
    </div>
    <div style="text-align:center;min-width:85px;">
      <div style="color:#555;font-size:.50rem;letter-spacing:.1em;">NET P&L</div>
      <div style="color:{col_accent};font-size:1.4rem;font-weight:700;">${net_d:.2f}</div>
      <div style="color:{col_accent};font-size:.60rem;">{result.get('net_pct',0):.3f}%/unit</div>
    </div>
    <div style="text-align:center;min-width:85px;">
      <div style="color:#555;font-size:.50rem;letter-spacing:.1em;">ANNUALISED</div>
      <div style="color:#ff8c00;font-size:1.1rem;font-weight:700;">{ann_str}</div>
    </div>
    <div style="text-align:center;min-width:85px;">
      <div style="color:#555;font-size:.50rem;letter-spacing:.1em;">EXPIRES IN</div>
      <div style="color:#e8e8e8;font-size:1.1rem;font-weight:700;">{days_str}d</div>
    </div>
    <div style="text-align:center;min-width:85px;">
      <div style="color:#555;font-size:.50rem;letter-spacing:.1em;">SCORE</div>
      <div style="color:#cc88ff;font-size:1.1rem;font-weight:700;">{result.get('score',0):.1f}</div>
    </div>
  </div>
</div>""", unsafe_allow_html=True)

        # Trade instructions
        border2 = "#00d084" if is_rf else col_accent
        bg2     = "#001a0a" if is_rf else "#0d0d0d"
        st.markdown(f"""
<div style="background:#111;border:1px solid #2a2a2a;padding:12px 16px;
font-family:'IBM Plex Mono',monospace;margin-bottom:14px;">
  <div style="color:#ff8c00;font-size:.60rem;font-weight:700;letter-spacing:.1em;margin-bottom:10px;">◼ TRADE INSTRUCTIONS</div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;">
    <div style="background:{bg2};border:1px solid {border2};padding:10px;">
      <div style="color:#555;font-size:.50rem;">LEG 1 / ENTRY</div>
      <div style="color:{border2};font-size:.80rem;font-weight:700;margin-top:4px;">{result['action_1']}</div>
    </div>
    <div style="background:#111;border:1px solid #2a2a2a;padding:10px;">
      <div style="color:#555;font-size:.50rem;">LEG 2 / TARGET</div>
      <div style="color:#c8c8c8;font-size:.80rem;font-weight:700;margin-top:4px;">{result['action_2']}</div>
    </div>
  </div>
  <div style="margin-top:10px;padding:8px;background:#0d0d0d;border:1px solid #1a1a1a;">
    <div style="display:flex;gap:20px;flex-wrap:wrap;font-size:.65rem;">
      <span style="color:#888;">Capital: <b style="color:#e8e8e8">${cost_d:.2f}</b></span>
      <span style="color:#ff3b3b;">Fee: ~${result.get('net_profit',0)*POLY_FEE/(1-POLY_FEE+1e-9)*lot_size:.4f}</span>
      <span style="color:{col_accent};font-weight:700;">Net: ${net_d:.4f}</span>
      <span style="color:#888;">ROI: {result.get('net_pct',0):.3f}%</span>
      {'<span style="color:#ff8c00;">Ann: ' + ann_str + '</span>' if result.get('annualised_pct') else ''}
    </div>
  </div>
</div>""", unsafe_allow_html=True)

        # Risk label
        if is_rf:
            st.success(f"✅ PURE ARB — Guaranteed ${net_d:.2f} regardless of outcome. "
                       f"Residual risks: platform solvency, fill slippage, resolution dispute.")
        elif result["risk"] == "HIGH-PROB":
            st.info(f"📌 HIGH-PROBABILITY — {days_str}d to resolution. Near-certain but NOT locked-in until resolved.")
        elif result["risk"] == "MEAN-REVERT":
            st.warning(f"⚡ LIQUIDITY EDGE — Edge: {result.get('best_edge',0)*100:.2f}¢. "
                       f"Spread of {result.get('spread',0):.4f} must close to mid for full P&L.")
        elif result["risk"] == "DIRECTIONAL":
            st.warning(f"📈 DIRECTIONAL — Profit depends on continued price movement. "
                       f"Signal strength: {result.get('signal_strength', result.get('vol_ratio',0)):.2f}. Size appropriately.")
        else:
            st.warning(f"🔬 SPECULATIVE — Composite inefficiency: {result.get('composite',0):.2f}. "
                       f"Research signal — verify manually.")

        if result.get("url"):
            st.markdown(f"[→ Open on Polymarket ↗]({result['url']})")

        # 3-column charts
        ch1, ch2, ch3 = st.columns(3)

        with ch1:
            st.plotly_chart(payoff_chart(result, lot_size),
                            use_container_width=True, config={"displayModeBar": False})
        with ch2:
            if result.get("history"):
                st.plotly_chart(price_history_chart(result["history"], result),
                                use_container_width=True, config={"displayModeBar": False})
            elif result.get("signals"):
                st.plotly_chart(signals_radar_chart(result["signals"]),
                                use_container_width=True, config={"displayModeBar": False})
            else:
                st.plotly_chart(annualised_chart(result),
                                use_container_width=True, config={"displayModeBar": False})
        with ch3:
            st.plotly_chart(size_sensitivity_chart(result, lot_size),
                            use_container_width=True, config={"displayModeBar": False})

        # Full stats expander
        with st.expander("▸ FULL STATISTICS"):
            sc1, sc2 = st.columns(2)
            with sc1:
                core = [("Strategy", strat), ("Risk", result["risk"]),
                        ("Score", f"{result.get('score',0):.2f}"),
                        ("Net/Unit", f"{result.get('net_profit',0):.6f}"),
                        ("Net %", f"{result.get('net_pct',0):.4f}%"),
                        ("Annualised", ann_str), ("Days to Res.", days_str),
                        ("Capital", f"${cost_d:.2f}"), ("Total P&L", f"${net_d:.4f}")]
                st.markdown("| Field | Value |\n|---|---|\n" +
                            "\n".join(f"| {k} | {v} |" for k, v in core))
            with sc2:
                extra_keys = {"yes_vwap":"YES VWAP","no_vwap":"NO VWAP",
                              "entry_price":"Entry","target_price":"Target",
                              "velocity_24h":"Velocity 24h","vol_ratio":"Vol Ratio",
                              "spread":"Spread","best_edge":"Best Edge",
                              "composite":"Composite Score","signal_strength":"Signal Strength"}
                extra_rows = [(lbl, f"{result[k]:.4f}" if isinstance(result.get(k), float) else str(result.get(k,"")))
                              for k, lbl in extra_keys.items() if result.get(k) is not None]
                if extra_rows:
                    st.markdown("| Field | Value |\n|---|---|\n" +
                                "\n".join(f"| {k} | {v} |" for k, v in extra_rows))
                if result.get("signals"):
                    st.markdown("\n**SIGNAL BREAKDOWN**")
                    for sig, val in result["signals"].items():
                        bar = "█" * int(val * 10)
                        st.markdown(f"`{sig.upper():<22}` {bar:<10} `{val:.2f}`")

# Auto-refresh
if auto_refresh:
    time.sleep(60)
    st.rerun()