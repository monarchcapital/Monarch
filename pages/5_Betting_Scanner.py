"""
MONARCH · POLYMARKET STRATEGY SCANNER
Strategies: Overround Arb · Momentum · Liquidity Edge · Volume Anomaly · Decay Trade · Inefficiency Score
APIs used:  Gamma /markets · CLOB /book · CLOB /prices-history · CLOB /trades
"""

import json
import re
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

st.set_page_config(page_title="Polymarket Scanner · MONARCH", layout="wide")

# ── THEME ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600;700&display=swap');
:root{--bg:#0a0a0a;--s:#111;--b:#2a2a2a;--a:#ff8c00;--a2:#ffb347;
     --g:#00d084;--r:#ff3b3b;--bl:#1e90ff;--cy:#00ccff;--pu:#cc88ff;
     --w:#e8e8e8;--w2:#c8c8c8;--mu:#888;--mo:'IBM Plex Mono',monospace;}
html,body,[data-testid="stAppViewContainer"],[data-testid="stAppViewContainer"]>section,.main .block-container{
  background:var(--bg)!important;color:var(--w)!important;font-family:var(--mo)!important;}
p,span,div,label,li,caption,[data-testid="stMarkdownContainer"] p,[data-testid="stMarkdownContainer"] span{
  color:var(--w2)!important;font-family:var(--mo)!important;}
[data-testid="stSidebar"],[data-testid="stSidebar"]>div{background:#060606!important;border-right:1px solid var(--b)!important;}
[data-testid="stSidebar"] p,[data-testid="stSidebar"] span,[data-testid="stSidebar"] label,[data-testid="stSidebar"] div{
  color:var(--mu)!important;font-size:.67rem!important;letter-spacing:.05em!important;}
h1{color:var(--a)!important;font-size:1.05rem!important;font-weight:700!important;letter-spacing:.18em!important;
   text-transform:uppercase!important;border-bottom:2px solid var(--a)!important;padding-bottom:6px!important;}
[data-testid="metric-container"]{background:var(--s)!important;border-radius:0!important;
  border:1px solid var(--b)!important;border-left:3px solid var(--a)!important;padding:8px 12px!important;}
[data-testid="stMetricLabel"] p{color:var(--mu)!important;font-size:.58rem!important;letter-spacing:.12em!important;text-transform:uppercase!important;}
[data-testid="stMetricValue"]{color:var(--a)!important;font-size:1.05rem!important;font-weight:700!important;}
[data-testid="stDataFrame"]{border:1px solid var(--a)!important;border-radius:0!important;}
.stDataFrame thead tr th{background:#1a1200!important;color:var(--a)!important;font-family:var(--mo)!important;
  font-size:.60rem!important;font-weight:700!important;letter-spacing:.14em!important;text-transform:uppercase!important;
  border-bottom:2px solid var(--a)!important;padding:6px 10px!important;}
.stDataFrame tbody tr td{background:#0d0d0d!important;color:var(--w)!important;font-family:var(--mo)!important;
  font-size:.68rem!important;border-bottom:1px solid #1a1a1a!important;padding:4px 10px!important;}
.stDataFrame tbody tr:nth-child(odd) td{background:#111!important;}
.stDataFrame tbody tr:hover td{background:#1f1400!important;color:var(--a)!important;}
.stButton>button{background:#140e00!important;color:var(--a)!important;border:1px solid var(--a)!important;
  border-radius:0!important;font-family:var(--mo)!important;font-size:.70rem!important;
  font-weight:600!important;letter-spacing:.1em!important;text-transform:uppercase!important;padding:6px 18px!important;}
.stButton>button:hover{background:var(--a)!important;color:#000!important;}
.stTabs [data-baseweb="tab-list"]{background:#080808!important;border-bottom:2px solid var(--a)!important;gap:0!important;}
.stTabs [data-baseweb="tab"]{background:transparent!important;color:var(--mu)!important;font-family:var(--mo)!important;
  font-size:.63rem!important;font-weight:600!important;letter-spacing:.1em!important;text-transform:uppercase!important;
  border-radius:0!important;border-right:1px solid var(--b)!important;padding:8px 14px!important;}
.stTabs [aria-selected="true"]{background:#1a1200!important;color:var(--a)!important;
  border-bottom:3px solid var(--a)!important;font-weight:700!important;}
.stProgress>div>div{background:var(--a)!important;}
hr{border-color:#1e1e1e!important;margin:10px 0!important;}
.streamlit-expanderHeader,[data-testid="stExpander"] summary{background:var(--s)!important;color:var(--a)!important;
  font-family:var(--mo)!important;font-size:.68rem!important;font-weight:600!important;
  letter-spacing:.1em!important;text-transform:uppercase!important;border-radius:0!important;
  border:1px solid var(--b)!important;list-style:none!important;}
[data-testid="stExpander"] summary{list-style:none!important;-webkit-appearance:none!important;}
[data-testid="stExpander"] summary::-webkit-details-marker{display:none!important;}
[data-testid="stExpander"] summary::marker{display:none!important;content:""!important;}
[data-testid="stExpander"] summary svg{display:none!important;width:0!important;height:0!important;}
[data-testid="stExpander"] summary p{display:inline!important;}
[data-testid="stExpander"] summary::before{content:"▸  ";color:var(--a)!important;font-size:.68rem;}
::-webkit-scrollbar{width:4px;height:4px;}
::-webkit-scrollbar-track{background:var(--bg);}
::-webkit-scrollbar-thumb{background:#333;}
::-webkit-scrollbar-thumb:hover{background:var(--a);}
</style>""", unsafe_allow_html=True)

st.markdown("""
<div style="background:#ff8c00;color:#000;font-family:'IBM Plex Mono',monospace;
font-size:.65rem;font-weight:700;letter-spacing:.18em;padding:5px 14px;
display:flex;justify-content:space-between;margin-bottom:12px;">
  <span>◼ MONARCH PRO — POLYMARKET STRATEGY SCANNER</span>
  <span>POLYMARKET · LIVE DATA · 6 STRATEGIES</span>
</div>""", unsafe_allow_html=True)

st.title("⚡ POLYMARKET STRATEGY SCANNER")
st.caption("6 strategies · CLOB VWAP · Gamma enrichment · Momentum · Liquidity · Volume · Decay · Inefficiency")

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
POLY_BASE  = "https://clob.polymarket.com"
GAMMA_BASE = "https://gamma-api.polymarket.com"
POLY_FEE   = 0.02

HEADERS = {
    "Accept":     "application/json",
    "User-Agent": "Mozilla/5.0 (compatible; MonarchScanner/1.0)",
    "Origin":     "https://polymarket.com",
    "Referer":    "https://polymarket.com/",
}

DARK = dict(
    plot_bgcolor="#000", paper_bgcolor="#0a0a0a",
    font=dict(color="#e8e8e8", family="IBM Plex Mono", size=10),
    hoverlabel=dict(bgcolor="#1a1200", font_color="#ff8c00", font_family="IBM Plex Mono", font_size=11),
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

# ── SAFE PARSERS ──────────────────────────────────────────────────────────────
# Gamma API returns many numeric fields as JSON strings e.g. "0.73" or '["0.73","0.27"]'

def _sf(val, default: float = 0.0) -> float:
    """Safe float: handles None, '', '0.73', '["0.73","0.27"]', 0.73"""
    if val is None:
        return default
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        v = val.strip()
        if not v or v in ("null", "None", "N/A", "—"):
            return default
        if v.startswith("["):
            try:
                lst = json.loads(v)
                return float(lst[0]) if lst else default
            except Exception:
                pass
        try:
            return float(v)
        except Exception:
            return default
    return default


def _sl(val) -> list:
    """Safe list: handles None, actual list, or JSON-string '["a","b"]'"""
    if val is None:
        return []
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        v = val.strip()
        if not v or v in ("null", "None"):
            return []
        try:
            parsed = json.loads(v)
            return parsed if isinstance(parsed, list) else []
        except Exception:
            return []
    return []


# ── MATH HELPERS ──────────────────────────────────────────────────────────────

def _days(end_str: str) -> float:
    if not end_str:
        return float("inf")
    try:
        s = re.sub(r"\.\d+", "", str(end_str).replace("Z", "+00:00"))
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        raw = (dt - datetime.now(timezone.utc)).total_seconds() / 86400
        # Floor at 1.0 — sub-day precision causes annualised overflow and is not useful
        return max(1.0, raw) if raw > 0 else float("inf")
    except Exception:
        return float("inf")


def _ann(net: float, days: float):
    if not net or not days:
        return None
    if days < 1 or days == float("inf") or days > 3650 or net <= 0:
        return None
    try:
        return min(((1 + net) ** (365.0 / days) - 1) * 100, 100_000.0)
    except (OverflowError, ZeroDivisionError):
        return None


def _vwap(asks: list, target_dollars: float):
    """VWAP fill cost from order book asks [(price, size), ...]"""
    if not asks or target_dollars <= 0:
        return None
    spent = contracts = 0.0
    for price, size in asks:
        price = _sf(price)
        size  = _sf(size)
        if price <= 0 or size <= 0:
            continue
        avail = price * size
        take  = min(avail, target_dollars - spent)
        spent     += take
        contracts += take / price
        if spent >= target_dollars - 1e-9:
            return spent / contracts if contracts > 0 else None
    return None  # insufficient depth


# ── SESSION STATE INIT ────────────────────────────────────────────────────────
for _k, _v in [("results", []), ("scan_ts", None), ("scan_mkts", 0),
               ("api_errors", {}), ("api_ok", {})]:
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ── API LAYER ─────────────────────────────────────────────────────────────────

def _get(url: str, params: dict = None, timeout: int = 15):
    """GET with headers; returns (json_data | None, error_str | None)"""
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=timeout)
        if r.status_code in (403, 404):
            return None, f"HTTP {r.status_code}: {url}"
        r.raise_for_status()
        return r.json(), None
    except requests.exceptions.ConnectionError as e:
        return None, f"Connection error — check network/VPN: {e}"
    except requests.exceptions.Timeout:
        return None, f"Timeout ({timeout}s): {url}"
    except requests.exceptions.HTTPError as e:
        return None, f"HTTP {e.response.status_code}: {url}"
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


@st.cache_data(ttl=30, show_spinner=False)
def fetch_markets(limit: int = 300) -> list:
    """
    Fetches from Gamma API (primary) then CLOB (fallback).
    Returns list of normalised market dicts ready for strategy engines.
    """
    # ── Gamma API (primary) ──────────────────────────────────────────────────
    data, err = _get(f"{GAMMA_BASE}/markets",
                     params={"limit": limit, "active": "true", "closed": "false",
                             "order": "volume24hr", "ascending": "false"})

    raw_markets = None
    if data is not None:
        raw_markets = data if isinstance(data, list) else data.get("markets", data.get("data"))

    if raw_markets:
        st.session_state.api_ok["Gamma"] = f"✓ {len(raw_markets)} markets"
        return _normalise(raw_markets, source="gamma")

    st.session_state.api_errors["Gamma"] = err or "empty response"

    # ── CLOB API (fallback) ──────────────────────────────────────────────────
    data2, err2 = _get(f"{POLY_BASE}/markets",
                       params={"limit": limit, "active": "true", "closed": "false",
                               "order": "volume_num", "ascending": "false"})
    if data2 is not None:
        raw2 = data2.get("data", [])
        if raw2:
            st.session_state.api_ok["CLOB"] = f"✓ {len(raw2)} markets"
            return _normalise(raw2, source="clob")

    st.session_state.api_errors["CLOB"] = err2 or "empty response"
    return []


def _normalise(raw: list, source: str) -> list:
    """Convert raw Gamma or CLOB market list into unified dicts."""
    out = []
    for m in raw:
        try:
            # Skip closed/inactive
            if m.get("closed") or not m.get("active", True):
                continue

            # ── Prices ───────────────────────────────────────────────────────
            if source == "gamma":
                op        = _sl(m.get("outcomePrices"))
                yes_price = _sf(op[0], 0.5) if len(op) > 0 else _sf(m.get("bestBid"), 0.5)
                no_price  = _sf(op[1], 0.5) if len(op) > 1 else (1.0 - yes_price)
                ctids     = _sl(m.get("clobTokenIds"))
                yes_tid   = str(ctids[0]) if len(ctids) > 0 else ""
                no_tid    = str(ctids[1]) if len(ctids) > 1 else ""
                vol_24h   = _sf(m.get("volume24hr"))
                vol_total = _sf(m.get("volumeClob"), vol_24h * 30)
                liquidity = _sf(m.get("liquidity"))
                best_bid  = _sf(m.get("bestBid"),  yes_price - 0.01)
                best_ask  = _sf(m.get("bestAsk"),  yes_price + 0.01)
                last_p    = _sf(m.get("lastTradePrice"), yes_price)
                end_str   = str(m.get("endDate") or "")
                question  = str(m.get("question") or "")
                cid       = str(m.get("conditionId") or m.get("condition_id") or "")
                slug      = str(m.get("slug") or "")
                category  = str(m.get("groupItemTitle") or m.get("category") or "")
            else:  # clob
                tokens    = m.get("tokens", [])
                yes_tok   = next((t for t in tokens if str(t.get("outcome","")).upper() == "YES"), {})
                no_tok    = next((t for t in tokens if str(t.get("outcome","")).upper() == "NO"),  {})
                yes_price = _sf(yes_tok.get("price"), 0.5)
                no_price  = _sf(no_tok.get("price"),  0.5)
                yes_tid   = str(yes_tok.get("token_id") or "")
                no_tid    = str(no_tok.get("token_id")  or "")
                vol_24h   = _sf(m.get("volume_num"))
                vol_total = _sf(m.get("volume_num"), vol_24h * 30)
                liquidity = _sf(m.get("liquidity"))
                best_bid  = yes_price - 0.01
                best_ask  = yes_price + 0.01
                last_p    = yes_price
                end_str   = str(m.get("end_date_iso") or m.get("end_date") or "")
                question  = str(m.get("question") or "")
                cid       = str(m.get("condition_id") or "")
                slug      = str(m.get("market_slug") or "")
                category  = ""

            yes_price = max(0.01, min(0.99, yes_price))
            no_price  = max(0.01, min(0.99, no_price))
            best_bid  = max(0.0, min(0.99, best_bid))
            best_ask  = max(0.01, min(1.0,  best_ask))

            out.append({
                "cid":       cid,
                "question":  question or "Unnamed market",
                "yes_price": yes_price,
                "no_price":  no_price,
                "yes_tid":   yes_tid,
                "no_tid":    no_tid,
                "best_bid":  best_bid,
                "best_ask":  best_ask,
                "spread":    max(0.0, best_ask - best_bid),
                "vol_24h":   max(0.0, vol_24h),
                "vol_total": max(0.0, vol_total),
                "liquidity": max(0.0, liquidity),
                "last_p":    max(0.01, min(0.99, last_p)),
                "overround": yes_price + no_price,
                "days":      _days(end_str),
                "category":  category,
                "url":       f"https://polymarket.com/event/{slug}" if slug else "",
            })
        except Exception:
            continue  # skip malformed market silently
    return out


@st.cache_data(ttl=15, show_spinner=False)
def fetch_ob(tid: str) -> dict:
    """Order book for a token id. Returns {bids: [(p,s)], asks: [(p,s)]}"""
    if not tid:
        return {"bids": [], "asks": []}
    data, _ = _get(f"{POLY_BASE}/book", params={"token_id": tid}, timeout=8)
    if not data:
        return {"bids": [], "asks": []}
    try:
        bids = sorted([(_sf(b["price"]), _sf(b["size"])) for b in data.get("bids", [])], reverse=True)
        asks = sorted([(_sf(a["price"]), _sf(a["size"])) for a in data.get("asks", [])])
        return {"bids": bids, "asks": asks}
    except Exception:
        return {"bids": [], "asks": []}


@st.cache_data(ttl=30, show_spinner=False)
def fetch_history(tid: str) -> list:
    """7-day hourly price history for a token id."""
    if not tid:
        return []
    now = int(time.time())
    data, _ = _get(f"{POLY_BASE}/prices-history",
                   params={"market": tid, "startTs": now - 86400*7, "endTs": now, "fidelity": 60},
                   timeout=10)
    return data.get("history", []) if data else []


@st.cache_data(ttl=60, show_spinner=False)
def fetch_trades(tid: str) -> list:
    """Recent 50 trades for a token id."""
    if not tid:
        return []
    data, _ = _get(f"{POLY_BASE}/trades", params={"market": tid, "limit": 50}, timeout=8)
    return data.get("data", []) if data else []


# ── STRATEGY ENGINES ──────────────────────────────────────────────────────────

def s1_overround(m: dict, ob_yes: dict, ob_no: dict, lot: float):
    """S1: Buy YES+NO when combined ask cost < $1 (guaranteed profit)."""
    y_vwap = _vwap(ob_yes["asks"], lot) or m["best_ask"]
    n_vwap = _vwap(ob_no["asks"],  lot) or (1.0 - m["best_bid"])
    cost   = y_vwap + n_vwap
    if cost >= 1.0:
        return None
    gross    = 1.0 - cost
    fee_cost = POLY_FEE * (1.0 - min(y_vwap, n_vwap))
    net      = gross - fee_cost
    if net <= 0:
        return None
    d = m["days"]
    return dict(
        strategy="OVERROUND ARB", risk="RISK-FREE",
        question=m["question"], category=m["category"],
        action_1=f"BUY YES @ {y_vwap:.4f}", action_2=f"BUY NO  @ {n_vwap:.4f}",
        yes_vwap=y_vwap, no_vwap=n_vwap,
        entry_price=y_vwap, target_price=1.0,
        cost_per_unit=cost, gross_profit=gross, fee_cost=fee_cost,
        net_profit=net, net_pct=net*100, annualised_pct=_ann(net, d),
        days_to_res=d, lot=lot,
        total_cost=cost*lot, total_profit=net*lot,
        score=net*100*(1 + 1/max(d, 1)),
        url=m["url"], cid=m["cid"], yes_tid=m["yes_tid"],
    )


def s2_momentum(m: dict, history: list, lot: float):
    """S2: 24h price velocity × volume acceleration momentum signal."""
    prices = [_sf(h.get("p"), -1) for h in history if _sf(h.get("p"), -1) > 0]
    if len(prices) < 6:
        return None
    velocity = prices[-1] - prices[max(0, len(prices) - 24)]
    if abs(velocity) < 0.03:
        return None
    h6    = prices[-1] - prices[max(0, len(prices) - 6)]
    h12   = prices[max(0, len(prices)-6)] - prices[max(0, len(prices)-12)]
    accel = h6 - h12
    vol_ratio = min(m["vol_24h"] / max(m["vol_total"] / 30.0 + 1, 1), 3.0)
    signal    = abs(velocity) * (1 + max(accel, 0)) * vol_ratio
    direction = "YES" if velocity > 0 else "NO"
    entry     = prices[-1] if direction == "YES" else (1 - prices[-1])
    target    = min(entry + abs(velocity) * 0.5, 0.99)
    net       = (target - entry) * (1 - POLY_FEE)
    if net <= 0.005:
        return None
    d = m["days"]
    return dict(
        strategy="MOMENTUM", risk="DIRECTIONAL",
        question=m["question"], category=m["category"],
        action_1=f"BUY {direction} @ {entry:.4f}",
        action_2=f"TARGET {target:.4f}  (+{abs(velocity)*50:.1f}¢ continuation)",
        entry_price=entry, target_price=target,
        velocity_24h=velocity, acceleration=accel, vol_ratio=vol_ratio,
        signal_strength=signal,
        net_profit=net, net_pct=net*100, annualised_pct=_ann(net, d),
        days_to_res=d, lot=lot,
        total_cost=entry*lot, total_profit=net*lot,
        score=signal*10,
        url=m["url"], cid=m["cid"], yes_tid=m["yes_tid"],
        history=history,
    )


def s3_liquidity(m: dict, ob_yes: dict, ob_no: dict, lot: float):
    """S3: VWAP fill cheaper than mid-price on wide-spread markets."""
    if m["spread"] < 0.05:
        return None
    y_vwap = _vwap(ob_yes["asks"], lot) or m["best_ask"]
    n_vwap = _vwap(ob_no["asks"],  lot) or (1.0 - m["best_bid"])
    y_edge = m["yes_price"] - y_vwap
    n_edge = (1.0 - m["yes_price"]) - n_vwap
    best_edge = max(y_edge, n_edge)
    if best_edge < 0.02:
        return None
    side       = "YES" if y_edge >= n_edge else "NO"
    entry_vwap = y_vwap if side == "YES" else n_vwap
    mid        = m["yes_price"] if side == "YES" else (1.0 - m["yes_price"])
    net        = best_edge * (1 - POLY_FEE)
    d          = m["days"]
    return dict(
        strategy="LIQUIDITY EDGE", risk="MEAN-REVERT",
        question=m["question"], category=m["category"],
        action_1=f"BUY {side} @ VWAP {entry_vwap:.4f}",
        action_2=f"MID = {mid:.4f}  SPREAD = {m['spread']:.4f}",
        entry_price=entry_vwap, target_price=mid,
        spread=m["spread"], edge_yes=y_edge, edge_no=n_edge, best_edge=best_edge,
        net_profit=net, net_pct=net*100, annualised_pct=_ann(net, d),
        days_to_res=d, lot=lot,
        total_cost=entry_vwap*lot, total_profit=net*lot,
        score=best_edge*m["spread"]*1000,
        url=m["url"], cid=m["cid"], yes_tid=m["yes_tid"],
    )


def s4_volume(m: dict, trades: list, lot: float):
    """S4: Unusual volume burst — follow informed order flow."""
    amounts = [_sf(t.get("size")) for t in trades if _sf(t.get("size")) > 0]
    if len(amounts) < 5:
        return None
    recent10  = float(np.mean(amounts[:10]) if len(amounts) >= 10 else np.mean(amounts))
    baseline  = float(np.mean(amounts[-20:]) if len(amounts) >= 20 else recent10)
    vol_ratio = recent10 / (baseline + 1e-9)
    if vol_ratio < 2.0:
        return None
    t_prices  = [_sf(t.get("price")) for t in trades[:10] if _sf(t.get("price")) > 0]
    if len(t_prices) < 3:
        return None
    price_drift = t_prices[0] - t_prices[-1]
    direction   = "YES" if price_drift > 0 else "NO"
    entry       = m["yes_price"] if direction == "YES" else m["no_price"]
    target      = min(entry + abs(price_drift) * 0.5, 0.98)
    net         = (target - entry) * (1 - POLY_FEE)
    if net < 0.01:
        return None
    d = m["days"]
    return dict(
        strategy="VOLUME ANOMALY", risk="DIRECTIONAL",
        question=m["question"], category=m["category"],
        action_1=f"BUY {direction} @ {entry:.4f}  (follow informed flow)",
        action_2=f"TARGET {target:.4f}  (volume ratio: {vol_ratio:.1f}×)",
        entry_price=entry, target_price=target,
        direction=direction, vol_ratio=vol_ratio, price_drift=price_drift,
        net_profit=net, net_pct=net*100, annualised_pct=_ann(net, d),
        days_to_res=d, lot=lot,
        total_cost=entry*lot, total_profit=net*lot,
        score=vol_ratio*abs(price_drift)*100,
        url=m["url"], cid=m["cid"], yes_tid=m["yes_tid"],
    )


def s5_decay(m: dict, lot: float):
    """S5: Near-expiry high-probability outcome — explosive annualised return."""
    d = m["days"]
    if d > 14 or d <= 0:
        return None
    yes_p = m["yes_price"]
    if yes_p >= 0.88:
        entry = m["best_ask"]
        side  = "YES"
    elif yes_p <= 0.12:
        entry = 1.0 - m["best_bid"]
        side  = "NO"
    else:
        return None
    net = (1.0 - entry) * (1 - POLY_FEE)
    if net <= 0:
        return None
    ann = _ann(net, d)
    if ann is None or ann < 50:
        return None
    return dict(
        strategy="DECAY TRADE", risk="HIGH-PROB",
        question=m["question"], category=m["category"],
        action_1=f"BUY {side} @ {entry:.4f}  ({d:.1f}d remaining)",
        action_2="HOLD TO RESOLUTION  (expected $1.00 payout)",
        entry_price=entry, target_price=1.0,
        side=side, yes_price=yes_p,
        net_profit=net, net_pct=net*100, annualised_pct=ann,
        days_to_res=d, lot=lot,
        total_cost=entry*lot, total_profit=net*lot,
        score=(ann or 0) / 10,
        url=m["url"], cid=m["cid"], yes_tid=m["yes_tid"],
    )


def s6_inefficiency(m: dict, ob_yes: dict, ob_no: dict, lot: float):
    """S6: Composite inefficiency — 5 independent signals ranked."""
    sigs = {}
    sigs["overround_dev"]  = min(abs(m["overround"] - 1.0) * 10, 1.0)
    if m["liquidity"] > 1000:
        sigs["spread_anomaly"] = min(m["spread"] / (m["liquidity"] / 10000 + 0.01), 1.0)
    else:
        sigs["spread_anomaly"] = 0.0
    sigs["vol_liq"]        = min(m["vol_24h"] / (m["liquidity"] + 1) / 2, 1.0)
    y_dep = sum(s for _, s in ob_yes["asks"][:5])
    n_dep = sum(s for _, s in ob_no["asks"][:5])
    total = y_dep + n_dep
    sigs["ob_imbalance"]   = abs(y_dep - n_dep) / total if total > 0 else 0.0
    sigs["last_trade_dev"] = min(abs(m["yes_price"] - m["last_p"]) * 5, 1.0)
    composite = float(np.mean(list(sigs.values())))
    if composite < 0.25:
        return None
    entry_y = m["best_ask"]
    entry_n = 1.0 - m["best_bid"]
    side    = "YES" if entry_y <= entry_n else "NO"
    entry   = entry_y if side == "YES" else entry_n
    net     = max((0.5 - entry) * (1 - POLY_FEE), 0.001)
    d       = m["days"]
    return dict(
        strategy="INEFFICIENCY", risk="SPECULATIVE",
        question=m["question"], category=m["category"],
        action_1=f"INVESTIGATE {side}  (composite score: {composite:.2f})",
        action_2="5 signals flagged — verify manually before trading",
        entry_price=entry, target_price=0.5,
        composite=composite, signals=sigs,
        net_profit=net, net_pct=composite*20, annualised_pct=_ann(net, d),
        days_to_res=d, lot=lot,
        total_cost=entry*lot, total_profit=net*lot,
        score=composite*100,
        url=m["url"], cid=m["cid"], yes_tid=m["yes_tid"],
    )


# ── CHARTS ────────────────────────────────────────────────────────────────────

def chart_payoff(r: dict, lot: float) -> go.Figure:
    if r["strategy"] == "OVERROUND ARB":
        yv = r.get("yes_vwap", 0.5)
        nv = r.get("no_vwap",  0.5)
        pl_y = ((1 - yv) * (1 - POLY_FEE) - nv) * lot
        pl_n = ((1 - nv) * (1 - POLY_FEE) - yv) * lot
        outcomes, pls = ["YES RESOLVES", "NO RESOLVES"], [pl_y, pl_n]
    else:
        entry  = r.get("entry_price", 0.5)
        target = r.get("target_price", min(entry + r.get("net_profit", 0), 0.99))
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
    fig.update_layout(title=dict(text="PAYOFF DIAGRAM", font=dict(color="#ff8c00", size=11)),
                      xaxis=dict(gridcolor="#1a1a1a", tickfont=dict(color="#888")),
                      yaxis=dict(gridcolor="#1a1a1a", tickprefix="$", tickfont=dict(color="#888")),
                      margin=_M, showlegend=False, **DARK)
    return fig


def chart_history(history: list, r: dict) -> go.Figure:
    if not history:
        return go.Figure()
    ts = []
    ps = []
    for h in history:
        t = h.get("t")
        p = _sf(h.get("p"), -1)
        if t and p > 0:
            ts.append(datetime.fromtimestamp(int(t), tz=timezone.utc))
            ps.append(p)
    if not ts:
        return go.Figure()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts, y=ps, mode="lines",
                             line=dict(color="#ff8c00", width=1.8),
                             fill="tozeroy", fillcolor="rgba(255,140,0,0.06)",
                             hovertemplate="%{x|%b %d %H:%M}<br>%{y:.4f}<extra></extra>"))
    ep = r.get("entry_price")
    tp = r.get("target_price")
    if ep:
        fig.add_hline(y=ep, line_dash="dot", line_color="#1e90ff",
                      annotation_text="Entry", annotation_font=dict(color="#1e90ff", size=8))
    if tp and tp != 1.0:
        fig.add_hline(y=tp, line_dash="dot", line_color="#00d084",
                      annotation_text="Target", annotation_font=dict(color="#00d084", size=8))
    fig.update_layout(title=dict(text="PRICE HISTORY (7D)", font=dict(color="#ff8c00", size=11)),
                      xaxis=dict(gridcolor="#1a1a1a", tickfont=dict(color="#888")),
                      yaxis=dict(gridcolor="#1a1a1a", tickfont=dict(color="#888"),
                                 tickformat=".3f", range=[0, 1]),
                      margin=_M, **DARK)
    return fig


def chart_annualised(r: dict) -> go.Figure:
    net = r.get("net_profit", 0)
    if net <= 0:
        return go.Figure()
    dr  = np.linspace(1, max(365, r.get("days_to_res", 30) * 2), 300)
    ann = ((1 + net) ** (365 / dr) - 1) * 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dr, y=ann, mode="lines",
                             line=dict(color="#ff8c00", width=2), name="Ann. Return"))
    d = r.get("days_to_res")
    if d and 0 < d < float("inf"):
        a = ((1 + net) ** (365 / d) - 1) * 100
        fig.add_trace(go.Scatter(x=[d], y=[a], mode="markers+text",
                                 marker=dict(color="#00d084", size=10, symbol="diamond"),
                                 text=[f"  {a:.0f}%"], textfont=dict(color="#00d084", size=9),
                                 textposition="middle right"))
    fig.add_hline(y=5, line_dash="dot", line_color="#333",
                  annotation_text="5% risk-free", annotation_font=dict(color="#555", size=8))
    fig.update_layout(title=dict(text="ANNUALISED vs DAYS", font=dict(color="#ff8c00", size=11)),
                      xaxis=dict(title="Days", gridcolor="#1a1a1a", tickfont=dict(color="#888")),
                      yaxis=dict(title="Ann. %", gridcolor="#1a1a1a", tickfont=dict(color="#888")),
                      margin=_M, **DARK)
    return fig


def chart_radar(signals: dict) -> go.Figure:
    cats = [k.replace("_", " ").upper() for k in signals]
    vals = list(signals.values())
    cats += [cats[0]]; vals += [vals[0]]
    fig = go.Figure(go.Scatterpolar(r=vals, theta=cats, fill="toself",
                                   line=dict(color="#ff8c00", width=2),
                                   fillcolor="rgba(255,140,0,0.12)"))
    fig.update_layout(
        polar=dict(bgcolor="#0a0a0a",
                   radialaxis=dict(visible=True, range=[0,1], gridcolor="#2a2a2a",
                                   tickfont=dict(color="#555", size=7)),
                   angularaxis=dict(gridcolor="#2a2a2a", tickfont=dict(color="#888", size=8))),
        title=dict(text="INEFFICIENCY SIGNALS", font=dict(color="#ff8c00", size=11)),
        margin=dict(t=50, b=30, l=50, r=50), **DARK)
    return fig


def chart_size(r: dict, lot: float) -> go.Figure:
    net  = r.get("net_profit", 0)
    lots = np.linspace(10, lot * 3, 200)
    fig  = go.Figure(go.Scatter(x=lots, y=net * lots, mode="lines",
                                line=dict(color="#1e90ff", width=2),
                                hovertemplate="Size: $%{x:,.0f}<br>Profit: $%{y:.2f}<extra></extra>"))
    fig.add_vline(x=lot, line_color="#ff8c00", line_dash="dot",
                  annotation_text="Current", annotation_font=dict(color="#ff8c00", size=8))
    fig.update_layout(title=dict(text="PROFIT vs SIZE", font=dict(color="#ff8c00", size=11)),
                      xaxis=dict(title="Position ($)", gridcolor="#1a1a1a",
                                 tickprefix="$", tickfont=dict(color="#888")),
                      yaxis=dict(title="Net Profit ($)", gridcolor="#1a1a1a",
                                 tickprefix="$", tickfont=dict(color="#888")),
                      margin=_M, **DARK)
    return fig


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙ SCANNER CONFIG")
    st.divider()
    lot_size   = st.number_input("Position size ($)", min_value=10, max_value=100_000, value=1_000, step=100)
    max_days   = st.number_input("Max days to resolution", min_value=1, max_value=365, value=90)
    min_volume = st.number_input("Min 24h volume ($)", min_value=0, max_value=1_000_000, value=500, step=500)
    st.divider()
    st.markdown("**STRATEGIES**")
    run_s1 = st.checkbox("S1 · Overround Arb",     value=True)
    run_s2 = st.checkbox("S2 · Momentum",           value=True)
    run_s3 = st.checkbox("S3 · Liquidity Edge",     value=True)
    run_s4 = st.checkbox("S4 · Volume Anomaly",     value=True)
    run_s5 = st.checkbox("S5 · Decay Trade",         value=True)
    run_s6 = st.checkbox("S6 · Inefficiency Score",  value=True)
    st.divider()
    auto_refresh = st.checkbox("Auto-refresh every 60s", value=False)
    use_ob       = st.checkbox("Fetch order books (slower)", value=True)
    use_hist     = st.checkbox("Fetch price history (slower)", value=True)
    st.divider()
    st.markdown("""<div style="font-size:.57rem;color:#555;line-height:1.6;">
<b>DATA SOURCES</b><br>
Gamma API /markets<br>CLOB /book · /prices-history · /trades<br><br>
<b>FEE MODEL</b><br>Polymarket: 2% of winnings<br><br>
<b>RISK LABELS</b><br>
RISK-FREE · guaranteed at resolution<br>
DIRECTIONAL · depends on price move<br>
MEAN-REVERT · spread must close<br>
HIGH-PROB · near-certain, not locked<br>
SPECULATIVE · research signal only</div>""", unsafe_allow_html=True)

# ── SCAN BUTTON ───────────────────────────────────────────────────────────────
cb, ci = st.columns([1, 3])
with cb:
    run_scan = st.button("🔍  SCAN POLYMARKET", use_container_width=True, type="primary")
with ci:
    on = [n for on, n in [(run_s1,"S1"),(run_s2,"S2"),(run_s3,"S3"),
                           (run_s4,"S4"),(run_s5,"S5"),(run_s6,"S6")] if on]
    st.markdown(f"""<div style="padding:8px 12px;background:#111;border-left:3px solid #ff8c00;font-size:.65rem;color:#888;">
Size: <span style="color:#ff8c00">${lot_size:,}</span> ·
Days: <span style="color:#ff8c00">{max_days}d</span> ·
Vol: <span style="color:#ff8c00">${min_volume:,}</span> ·
Active: <span style="color:#ff8c00">{' · '.join(on)}</span> ·
Fee: <span style="color:#ff8c00">2%</span></div>""", unsafe_allow_html=True)

if auto_refresh and st.session_state.scan_ts:
    if (time.time() - st.session_state.scan_ts) > 60:
        run_scan = True

# ── SCAN EXECUTION ────────────────────────────────────────────────────────────
if run_scan:
    st.session_state.api_errors = {}
    st.session_state.api_ok     = {}

    prog = st.progress(0.0, text="Fetching markets…")
    markets_raw = fetch_markets(limit=300)
    prog.progress(0.25, text="Markets loaded. Applying filters…")

    # Show API status
    ok_str  = "  ".join(f'<span style="color:#00d084">{v}</span>' for v in st.session_state.api_ok.values())
    err_str = "  ".join(f'<span style="color:#ff3b3b">✗ {k}: {v}</span>' for k, v in st.session_state.api_errors.items())
    if ok_str or err_str:
        st.markdown(f'<div style="font-size:.62rem;padding:4px 0;">{ok_str} {err_str}</div>',
                    unsafe_allow_html=True)

    if not markets_raw:
        st.error("**No markets returned.** Check your internet connection — Polymarket requires access to gamma-api.polymarket.com and clob.polymarket.com")
        st.stop()

    # Filter
    markets = [m for m in markets_raw if m["days"] <= max_days and m["vol_24h"] >= min_volume]

    if not markets:
        st.warning(f"Fetched {len(markets_raw)} markets but all were filtered out. "
                   f"Try lowering Min 24h volume (${min_volume:,}) or raising Max days ({max_days}d).")
        st.stop()

    all_results = []
    n = len(markets)

    for i, m in enumerate(markets):
        prog.progress(0.28 + 0.68 * (i / max(n, 1)),
                      text=f"[{i+1}/{n}] {m['question'][:52]}…")

        # Fetch order books once (shared by S1, S3, S6)
        ob_y = ob_n = {"bids": [], "asks": []}
        if use_ob and (run_s1 or run_s3 or run_s6):
            ob_y = fetch_ob(m["yes_tid"])
            ob_n = fetch_ob(m["no_tid"])

        if run_s1:
            r = s1_overround(m, ob_y, ob_n, lot_size)
            if r: all_results.append(r)

        if run_s2 and use_hist:
            hist = fetch_history(m["yes_tid"])
            r = s2_momentum(m, hist, lot_size)
            if r:
                all_results.append(r)

        if run_s3 and use_ob:
            r = s3_liquidity(m, ob_y, ob_n, lot_size)
            if r: all_results.append(r)

        if run_s4:
            trades = fetch_trades(m["yes_tid"])
            r = s4_volume(m, trades, lot_size)
            if r: all_results.append(r)

        if run_s5:
            r = s5_decay(m, lot_size)
            if r: all_results.append(r)

        if run_s6 and use_ob:
            r = s6_inefficiency(m, ob_y, ob_n, lot_size)
            if r: all_results.append(r)

    all_results.sort(key=lambda x: x.get("score", 0), reverse=True)
    st.session_state.results   = all_results
    st.session_state.scan_ts   = time.time()
    st.session_state.scan_mkts = len(markets)
    prog.progress(1.0, text=f"✓ {len(all_results)} signals found across {len(markets)} markets")

# ── RESULTS ───────────────────────────────────────────────────────────────────
results = st.session_state.results

if st.session_state.scan_ts:
    age = int(time.time() - st.session_state.scan_ts)
    by_s = {}
    for r in results:
        by_s[r["strategy"]] = by_s.get(r["strategy"], 0) + 1
    st.caption(f"Last scan: {age}s ago · {len(results)} signals · "
               f"{st.session_state.scan_mkts} markets · "
               + " · ".join(f"{k}: {v}" for k, v in sorted(by_s.items())))

if not results:
    if st.session_state.scan_ts:
        st.info("No signals found. Try lowering filters or enabling more strategies.")
    else:
        st.info("Click **SCAN POLYMARKET** to find live trading opportunities.")
    st.stop()

# Metrics
rf  = [r for r in results if r["risk"] == "RISK-FREE"]
dir_ = [r for r in results if r["risk"] == "DIRECTIONAL"]
best = results[0]
c1,c2,c3,c4,c5,c6 = st.columns(6)
c1.metric("RISK-FREE ARBS", len(rf))
c2.metric("DIRECTIONAL",    len(dir_))
c3.metric("TOTAL SIGNALS",  len(results))
c4.metric("TOP P&L",        f"${best.get('total_profit',0):.2f}")
ann_best = best.get("annualised_pct")
c5.metric("TOP ANNUALISED", f"{ann_best:.0f}%" if ann_best else "N/A")
c6.metric("TOP STRATEGY",   best["strategy"])

st.divider()

# Filters
f1, f2, f3 = st.columns(3)
with f1:
    sf = st.multiselect("Strategy", list(STRAT_COLORS.keys()), default=list(STRAT_COLORS.keys()))
with f2:
    rf2 = st.multiselect("Risk", ["RISK-FREE","DIRECTIONAL","MEAN-REVERT","HIGH-PROB","SPECULATIVE"],
                          default=["RISK-FREE","DIRECTIONAL","MEAN-REVERT","HIGH-PROB","SPECULATIVE"])
with f3:
    sb = st.selectbox("Sort by", ["Score","Net Profit $","Annualised %","Days to Resolution"])

filtered = [r for r in results if r.get("strategy","") in sf and r.get("risk","") in rf2]
if   sb == "Net Profit $":       filtered.sort(key=lambda x: x.get("total_profit", 0), reverse=True)
elif sb == "Annualised %":        filtered.sort(key=lambda x: x.get("annualised_pct") or 0, reverse=True)
elif sb == "Days to Resolution":  filtered.sort(key=lambda x: x.get("days_to_res", 999))
else:                             filtered.sort(key=lambda x: x.get("score", 0), reverse=True)

# Summary table
st.markdown('<div style="color:#ff8c00;font-size:.75rem;font-weight:700;letter-spacing:.1em;margin-bottom:8px;">◼ ALL SIGNALS — RANKED</div>',
            unsafe_allow_html=True)

rows = []
for r in filtered:
    rows.append({
        "Strategy":  r["strategy"],
        "Risk":      r["risk"],
        "Market":    r["question"][:55] + ("…" if len(r["question"]) > 55 else ""),
        "Category":  r.get("category","")[:18],
        "Entry":     r["action_1"][:42],
        "Capital $": f"${r.get('total_cost',0):.2f}",
        "Net P&L $": f"${r.get('total_profit',0):.2f}",
        "Net %":     f"{r.get('net_pct',0):.3f}%",
        "Ann. %":    f"{r['annualised_pct']:.0f}%" if r.get("annualised_pct") else "—",
        "Days":      f"{r.get('days_to_res',0):.1f}d",
        "Score":     f"{r.get('score',0):.1f}",
    })

if rows:
    def _color(row):
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

    st.dataframe(pd.DataFrame(rows).style.apply(_color, axis=1),
                 use_container_width=True, hide_index=True)

st.divider()

# Detail tabs
st.markdown('<div style="color:#ff8c00;font-size:.75rem;font-weight:700;letter-spacing:.1em;margin-bottom:8px;">◼ DETAILED ANALYSIS — TOP 8</div>',
            unsafe_allow_html=True)

display = filtered[:8]
if not display:
    st.info("No signals match current filters.")
    st.stop()

tabs = st.tabs([f"{r['strategy'].split()[0]} #{i+1}" for i, r in enumerate(display)])

for _ti, (tab, r) in enumerate(zip(tabs, display)):
    with tab:
        strat   = r["strategy"]
        col_ac  = STRAT_COLORS.get(strat, "#ff8c00")
        is_rf   = r["risk"] == "RISK-FREE"
        net_d   = r.get("total_profit", 0)
        cost_d  = r.get("total_cost",   0)
        ann_s   = f"{r['annualised_pct']:.1f}%" if r.get("annualised_pct") else "N/A"
        days_s  = f"{r['days_to_res']:.1f}" if r.get("days_to_res") not in (None, float("inf")) else "?"

        # Hero card
        st.markdown(f"""
<div style="background:#0d0d0d;border:1px solid {col_ac};border-left:5px solid {col_ac};
padding:16px 20px;font-family:'IBM Plex Mono',monospace;margin-bottom:16px;">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:14px;">
    <div style="flex:3;min-width:200px;">
      <div style="color:#555;font-size:.50rem;letter-spacing:.14em;">{strat} · {r['risk']} · {r.get('category','').upper() or 'POLYMARKET'}</div>
      <div style="color:#e8e8e8;font-size:.88rem;font-weight:700;margin-top:5px;line-height:1.35;">{r['question'][:95]}{'…' if len(r['question'])>95 else ''}</div>
    </div>
    <div style="text-align:center;min-width:80px;">
      <div style="color:#555;font-size:.50rem;">NET P&L</div>
      <div style="color:{col_ac};font-size:1.4rem;font-weight:700;">${net_d:.2f}</div>
      <div style="color:{col_ac};font-size:.60rem;">{r.get('net_pct',0):.3f}%/unit</div>
    </div>
    <div style="text-align:center;min-width:80px;">
      <div style="color:#555;font-size:.50rem;">ANNUALISED</div>
      <div style="color:#ff8c00;font-size:1.1rem;font-weight:700;">{ann_s}</div>
    </div>
    <div style="text-align:center;min-width:80px;">
      <div style="color:#555;font-size:.50rem;">EXPIRES IN</div>
      <div style="color:#e8e8e8;font-size:1.1rem;font-weight:700;">{days_s}d</div>
    </div>
    <div style="text-align:center;min-width:80px;">
      <div style="color:#555;font-size:.50rem;">SCORE</div>
      <div style="color:#cc88ff;font-size:1.1rem;font-weight:700;">{r.get('score',0):.1f}</div>
    </div>
  </div>
</div>""", unsafe_allow_html=True)

        # Trade instructions
        b2 = "#001a0a" if is_rf else "#0d0d0d"
        st.markdown(f"""
<div style="background:#111;border:1px solid #2a2a2a;padding:12px 16px;font-family:'IBM Plex Mono',monospace;margin-bottom:14px;">
  <div style="color:#ff8c00;font-size:.60rem;font-weight:700;letter-spacing:.1em;margin-bottom:10px;">◼ TRADE INSTRUCTIONS</div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;">
    <div style="background:{b2};border:1px solid {col_ac};padding:10px;">
      <div style="color:#555;font-size:.50rem;">LEG 1 / ENTRY</div>
      <div style="color:{col_ac};font-size:.80rem;font-weight:700;margin-top:4px;">{r['action_1']}</div>
    </div>
    <div style="background:#111;border:1px solid #2a2a2a;padding:10px;">
      <div style="color:#555;font-size:.50rem;">LEG 2 / TARGET</div>
      <div style="color:#c8c8c8;font-size:.80rem;font-weight:700;margin-top:4px;">{r['action_2']}</div>
    </div>
  </div>
  <div style="margin-top:10px;padding:8px;background:#0d0d0d;border:1px solid #1a1a1a;">
    <div style="display:flex;gap:20px;flex-wrap:wrap;font-size:.65rem;">
      <span style="color:#888;">Capital: <b style="color:#e8e8e8">${cost_d:.2f}</b></span>
      <span style="color:#ff3b3b;">Fee: ~${r.get('net_profit',0)*POLY_FEE/(1-POLY_FEE+1e-9)*lot_size:.4f}</span>
      <span style="color:{col_ac};font-weight:700;">Net: ${net_d:.4f}</span>
      <span style="color:#888;">ROI: {r.get('net_pct',0):.3f}%  Ann: {ann_s}</span>
    </div>
  </div>
</div>""", unsafe_allow_html=True)

        # Risk badge
        if is_rf:
            st.success(f"✅ PURE ARB — Guaranteed ${net_d:.2f} regardless of outcome.")
        elif r["risk"] == "HIGH-PROB":
            st.info(f"📌 HIGH-PROBABILITY — {days_s}d to resolution. Near-certain but NOT locked-in.")
        elif r["risk"] == "MEAN-REVERT":
            st.warning(f"⚡ LIQUIDITY EDGE — {r.get('best_edge',0)*100:.2f}¢ edge vs mid. Spread must close.")
        elif r["risk"] == "DIRECTIONAL":
            st.warning(f"📈 DIRECTIONAL — Signal: {r.get('signal_strength', r.get('vol_ratio',0)):.2f}. Not guaranteed.")
        else:
            st.warning(f"🔬 SPECULATIVE — Composite score {r.get('composite',0):.2f}. Research signal only.")

        if r.get("url"):
            st.markdown(f"[→ Open on Polymarket ↗]({r['url']})")

        # 3 charts
        c1, c2, c3 = st.columns(3)
        with c1:
            st.plotly_chart(chart_payoff(r, lot_size), use_container_width=True,
                            config={"displayModeBar": False}, key=f"payoff_{_ti}")
        with c2:
            if r.get("history"):
                st.plotly_chart(chart_history(r["history"], r), use_container_width=True,
                                config={"displayModeBar": False}, key=f"hist_{_ti}")
            elif r.get("signals"):
                st.plotly_chart(chart_radar(r["signals"]), use_container_width=True,
                                config={"displayModeBar": False}, key=f"radar_{_ti}")
            else:
                st.plotly_chart(chart_annualised(r), use_container_width=True,
                                config={"displayModeBar": False}, key=f"ann_{_ti}")
        with c3:
            st.plotly_chart(chart_size(r, lot_size), use_container_width=True,
                            config={"displayModeBar": False}, key=f"size_{_ti}")

        # Full stats
        with st.expander("FULL STATISTICS"):
            ec1, ec2 = st.columns(2)
            with ec1:
                rows_c = [("Strategy", strat), ("Risk", r["risk"]),
                          ("Score", f"{r.get('score',0):.2f}"),
                          ("Net/Unit", f"{r.get('net_profit',0):.6f}"),
                          ("Net %", f"{r.get('net_pct',0):.4f}%"),
                          ("Annualised", ann_s), ("Days to Res.", days_s),
                          ("Capital", f"${cost_d:.2f}"), ("Total P&L", f"${net_d:.4f}")]
                st.markdown("| Field | Value |\n|---|---|\n" +
                            "\n".join(f"| {k} | {v} |" for k, v in rows_c))
            with ec2:
                extra = {"yes_vwap":"YES VWAP","no_vwap":"NO VWAP","entry_price":"Entry",
                         "target_price":"Target","velocity_24h":"Velocity 24h",
                         "vol_ratio":"Vol Ratio","spread":"Spread","best_edge":"Best Edge",
                         "composite":"Composite","signal_strength":"Signal Strength"}
                xrows = [(lbl, f"{r[k]:.4f}" if isinstance(r.get(k), float) else str(r.get(k,"")))
                         for k, lbl in extra.items() if r.get(k) is not None]
                if xrows:
                    st.markdown("| Field | Value |\n|---|---|\n" +
                                "\n".join(f"| {k} | {v} |" for k, v in xrows))
                if r.get("signals"):
                    st.markdown("**SIGNAL BREAKDOWN**")
                    for sig, val in r["signals"].items():
                        bar = "█" * int(val * 10)
                        st.markdown(f"`{sig.upper():<22}` {bar:<10} `{val:.2f}`")

# ── AUTO REFRESH ──────────────────────────────────────────────────────────────
if auto_refresh:
    time.sleep(60)
    st.rerun()
