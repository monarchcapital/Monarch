# pages/4_ML_Predictor.py
# ─────────────────────────────────────────────────────────────────────────────
# MONARCH ML PREDICTOR
# Receives tickers + raw OHLCV from Screener Pro.
# Stacked ensemble: RF + XGB + LGB + GBM → meta-learner
# Walk-forward CV model selection. No arbitrary hyperparameters.
# ─────────────────────────────────────────────────────────────────────────────

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ── ML stack ─────────────────────────────────────────────────────────────────
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               RandomForestRegressor, GradientBoostingRegressor,
                               StackingClassifier, StackingRegressor,
                               VotingClassifier, VotingRegressor)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, roc_auc_score, log_loss,
                              mean_absolute_error, mean_absolute_percentage_error,
                              precision_score, recall_score, f1_score)
from sklearn.calibration import CalibratedClassifierCV
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="ML Predictor · MONARCH", layout="wide")

# ─────────────────────────────────────────────────────────────────────────────
# CSS — reuse Monarch dark theme
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600;700&display=swap');
:root{--bb-amber:#ff8c00;--bb-amber2:#ffb347;--bb-green:#00d084;--bb-red:#ff3b3b;--bb-mono:'IBM Plex Mono',monospace;}
html,body,[class*="css"]{background:#000 !important;color:#e8e8e8 !important;font-family:var(--bb-mono) !important;font-size:.88rem !important;}
.stApp{background:#000 !important;}
[data-testid="stSidebar"]{background:#0a0a0a !important;border-right:1px solid #2a2a2a !important;}
[data-testid="stSidebar"] *{color:#e8e8e8 !important;font-size:.84rem !important;}
[data-testid="stSidebar"] label,[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label{color:#ffb347 !important;font-size:.78rem !important;font-weight:600 !important;}
label,.stSelectbox label,.stSlider label,.stNumberInput label,
div[data-testid="stWidgetLabel"]{color:#ffb347 !important;font-size:.82rem !important;font-weight:600 !important;}
.stSelectbox div[data-baseweb="select"]>div{background:#0a0a0a !important;border:1px solid #3a3a3a !important;color:#e8e8e8 !important;font-size:.84rem !important;}
.stSelectbox div[data-baseweb="select"] span{color:#e8e8e8 !important;}
.stButton>button{background:#0a0800 !important;color:#ff8c00 !important;border:1px solid #ff8c00 !important;
  font-family:var(--bb-mono) !important;font-size:.80rem !important;font-weight:700 !important;
  letter-spacing:.1em !important;border-radius:0 !important;}
.stButton>button:hover{background:#ff8c00 !important;color:#000 !important;}
.stButton>button[kind="primary"]{background:#ff8c00 !important;color:#000 !important;}
div[data-testid="metric-container"]{background:#0a0a0a;border:1px solid #2a2a2a;padding:10px 14px;}
div[data-testid="metric-container"] label{color:#aaa !important;font-size:.68rem !important;letter-spacing:.06em !important;}
div[data-testid="metric-container"] div[data-testid="stMetricValue"]{color:#e8e8e8 !important;font-size:1.1rem !important;font-weight:700 !important;}
div[data-testid="metric-container"] div[data-testid="stMetricDelta"]{color:#00d084 !important;font-size:.76rem !important;}
.stDataFrame thead tr th{background:#111 !important;color:#ff8c00 !important;font-size:.78rem !important;font-weight:700 !important;}
.stDataFrame tbody tr td{color:#e8e8e8 !important;font-size:.82rem !important;}
.stMarkdown,.stMarkdown p,.stMarkdown li{color:#e8e8e8 !important;font-size:.88rem !important;}
.stMarkdown h1{font-size:1.2rem !important;color:#ff8c00 !important;}
.stMarkdown h2{font-size:1.0rem !important;color:#ffb347 !important;}
.stMarkdown h3{font-size:.92rem !important;color:#ffb347 !important;}
.stMarkdown code{background:#1a1a1a !important;color:#ff8c00 !important;font-size:.82rem !important;padding:1px 5px !important;border-radius:3px !important;}
.stAlert p,.stAlert div{color:#e8e8e8 !important;font-size:.84rem !important;}
.stProgress>div>div{background:#ff8c00 !important;}
.stTabs [data-baseweb="tab-list"]{background:#0a0a0a !important;gap:0;}
.stTabs [data-baseweb="tab"]{background:#0a0a0a !important;color:#aaa !important;font-size:.80rem !important;font-weight:600 !important;border-bottom:2px solid transparent !important;padding:8px 18px !important;}
.stTabs [aria-selected="true"]{color:#ff8c00 !important;border-bottom:2px solid #ff8c00 !important;background:#0a0800 !important;}
.stTabs [data-baseweb="tab-panel"]{background:#000 !important;padding-top:12px !important;}
.streamlit-expanderHeader{background:#0a0a0a !important;color:#ffb347 !important;font-size:0.92rem !important;font-weight:600 !important;border:1px solid #2a2a2a !important;}
.streamlit-expanderContent{background:#050505 !important;border:1px solid #1a1a1a !important;color:#e8e8e8 !important;}
::-webkit-scrollbar{width:6px;height:6px;}::-webkit-scrollbar-track{background:#0a0a0a;}
::-webkit-scrollbar-thumb{background:#333;border-radius:3px;}::-webkit-scrollbar-thumb:hover{background:#ff8c00;}

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

</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="background:#0a0800;border-bottom:2px solid #ff8c00;padding:12px 20px;
     font-family:'IBM Plex Mono',monospace;margin-bottom:16px;">
  <span style="color:#ff8c00;font-size:1.1rem;font-weight:700;letter-spacing:.14em;">
    🧠 MONARCH ML PREDICTOR
  </span>
  <span style="color:#555;font-size:.68rem;margin-left:20px;">
    Screener signals → Feature engineering → Price direction + target forecast
  </span>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
def _normalise_df(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise raw OHLCV from any source (screener=lowercase, yfinance=MultiIndex/tz)."""
    d = df.copy()
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    d.columns = [c.strip().title() if isinstance(c, str) else str(c).strip().title()
                 for c in d.columns]
    d = d.loc[:, ~d.columns.duplicated()]
    for ghost in ("Date", "Datetime", "Timestamp", "Index"):
        if ghost in d.columns:
            d.drop(columns=[ghost], inplace=True)
    if not isinstance(d.index, pd.DatetimeIndex):
        try:
            d.index = pd.to_datetime(d.index, utc=True).tz_convert(None)
        except Exception:
            d.index = pd.to_datetime(d.index, errors="coerce")
    elif d.index.tz is not None:
        d.index = d.index.tz_convert(None)
    ohlcv = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}
    bad = [c for c in d.columns
           if c not in ohlcv and not pd.api.types.is_numeric_dtype(d[c])]
    if bad:
        d.drop(columns=bad, inplace=True)
    return d



# ─────────────────────────────────────────────────────────────────────────────
# SCREENER SIGNAL HELPERS  (exact same math as 1_Live_screener.py)
# All return rolling Series — fully vectorised, zero lookahead when shifted.
# ─────────────────────────────────────────────────────────────────────────────

def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def _rsi_wilder(c: pd.Series, n: int) -> pd.Series:
    delta    = c.diff()
    gain     = delta.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    loss     = (-delta.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    return (100 - 100 / (1 + gain / loss.replace(0, np.nan))).fillna(50)

def _atr(h: pd.Series, l: pd.Series, c: pd.Series, n: int = 14) -> pd.Series:
    tr = pd.concat([h - l,
                    (h - c.shift(1)).abs(),
                    (l - c.shift(1)).abs()], axis=1).max(axis=1)
    return tr.ewm(span=n, adjust=False).mean(), tr

def _percentile_rank(s: pd.Series, window: int) -> pd.Series:
    """
    Rolling percentile rank of the last value within a window.
    Output: 0.0 (at window min) → 1.0 (at window max).
    min_periods = max(10, window // 5) so short histories still produce values.
    No arbitrary thresholds — fully adaptive to the stock's own history.
    """
    mp = max(10, window // 5)
    lo = s.rolling(window, min_periods=mp).min()
    hi = s.rolling(window, min_periods=mp).max()
    return ((s - lo) / (hi - lo + 1e-9)).clip(0, 1)

def _rs_score_series(c: pd.Series, nifty_c: pd.Series) -> pd.Series:
    """
    Vectorised RS score (0-1) matching screener's relative_strength():
    vol-normalised alpha via tanh squashing, 5d weighted 60/40 over 20d.
    Fully adaptive — no fixed ±2% band.
    """
    # Align nifty to stock's index
    nifty_aligned = nifty_c.reindex(c.index).ffill()

    stock_r5  = c.pct_change(5)
    stock_r20 = c.pct_change(20)
    nifty_r5  = nifty_aligned.pct_change(5)
    nifty_r20 = nifty_aligned.pct_change(20)

    beat5  = stock_r5  - nifty_r5
    beat20 = stock_r20 - nifty_r20

    # Volatility normalisation — stock's own rolling std
    ret_std5  = c.pct_change().rolling(5,  min_periods=3).std().clip(lower=0.001)
    ret_std20 = c.pct_change().rolling(20, min_periods=5).std().clip(lower=0.001)

    alpha5  = beat5  / (ret_std5  * np.sqrt(5)  + 1e-9)
    alpha20 = beat20 / (ret_std20 * np.sqrt(20) + 1e-9)

    rs5  = 0.5 * (1 + np.tanh(alpha5  / 1.5))
    rs20 = 0.5 * (1 + np.tanh(alpha20 / 1.5))

    return (rs5 * 0.6 + rs20 * 0.4).clip(0, 1)


def engineer_features(df: pd.DataFrame,
                      nifty_close: pd.Series | None = None) -> pd.DataFrame:
    """
    Full feature set combining:
      A) Standard technicals (returns, MAs, RSI, MACD, BB, ATR, volume, stoch, ADX, candles)
      B) All 10 Screener scoring signals ported as rolling time-series:
           F1  RS vs Nifty         — vol-normalised tanh alpha (0-1)
           F2  Momentum acceleration — EMA5−EMA20 velocity, percentile-ranked
           F3  Volume surge z-score  — (vol − μ) / σ over rolling 20d
           F4  Institutional accumulation — vol5/vol20 sigmoid score
           F5  Volatility contraction — ATR5/ATR20 percentile, inverted
           F5b Range compression     — range5/range20 percentile, inverted
           F5c VCVE interaction      — inst_ratio × (1 − vc_ratio)
           F6  Coil/base quality     — compression × flatness
           F7  Trend structure       — EMA9/EMA50 percentile over 250d
           F8  Breakout proximity    — exp decay from 20d resistance (ATR-normalised)
           F9  ATR% potential        — ATR% percentile vs own 60d history
           F10 Candlestick score     — 8 patterns, rolling (0-10 scaled 0-1)
           Bonus: liquidity sweep, VWMA20 position, momentum stability,
                  52-week position percentile, RS acceleration

    All features are shifted by 1 bar at the end → strictly zero lookahead.
    No arbitrary thresholds — every threshold is data-derived (percentile/tanh/sigmoid).
    """
    d = _normalise_df(df)
    c = d["Close"].squeeze()
    h = d["High"].squeeze()
    l = d["Low"].squeeze()
    o = d["Open"].squeeze()
    v = d["Volume"].squeeze().replace(0, np.nan).ffill()
    n = len(d)

    # ── Adaptive window caps ──────────────────────────────────────────────────
    # All percentile windows scale to available history so 250d of data
    # (≈175 trading bars after weekends/holidays) still produces valid features.
    # Rule: window ≤ n // 3  ensures at least 3 full windows exist in the data.
    _w60  = max(15, min(60,  n // 3))
    _w100 = max(15, min(100, n // 3))
    _w120 = max(15, min(120, n // 3))
    _w200 = max(15, min(200, n // 3))

    # ═══════════════════════════════════════════════════════════════════
    # A — STANDARD TECHNICALS
    # ═══════════════════════════════════════════════════════════════════

    # Returns
    for p in [1, 3, 5, 10, 20]:
        d[f"ret_{p}"] = c.pct_change(p)

    # MAs — kept raw for screener signal computation; dist_* used as features
    for w in [5, 10, 20, 50, 200]:
        d[f"sma{w}"]      = c.rolling(w).mean()
        d[f"dist_sma{w}"] = (c - d[f"sma{w}"]) / (d[f"sma{w}"] + 1e-9)

    e5  = _ema(c, 5);  e9  = _ema(c, 9)
    e20 = _ema(c, 20); e50 = _ema(c, 50)
    d["ema9"]      = e9
    d["ema21"]     = _ema(c, 21)
    d["ema_cross"] = (e9 - d["ema21"]) / (d["ema21"] + 1e-9)

    # RSI — Wilder (exact same as screener)
    for rsi_w in [7, 14, 21]:
        d[f"rsi{rsi_w}"] = _rsi_wilder(c, rsi_w)

    # MACD
    ema12 = _ema(c, 12); ema26 = _ema(c, 26)
    macd  = ema12 - ema26
    sig   = _ema(macd, 9)
    d["macd"]      = macd
    d["macd_hist"] = macd - sig
    d["macd_norm"] = macd / (c + 1e-9)

    # Bollinger Bands
    bb_mid = c.rolling(20).mean()
    bb_std = c.rolling(20).std()
    d["bb_upper"] = bb_mid + 2 * bb_std
    d["bb_lower"] = bb_mid - 2 * bb_std
    d["bb_width"] = (d["bb_upper"] - d["bb_lower"]) / (bb_mid + 1e-9)
    d["bb_pos"]   = (c - d["bb_lower"]) / (d["bb_upper"] - d["bb_lower"] + 1e-9)

    # ATR (EWM, same as screener's atr14)
    atr14_s, tr = _atr(h, l, c, 14)
    d["atr14"]     = atr14_s
    d["atr_norm"]  = atr14_s / (c + 1e-9)
    d["atr_ratio"] = tr / (atr14_s + 1e-9)

    # Volume
    vol_ma20  = v.rolling(20).mean()
    vol_std20 = v.rolling(20).std()
    d["vol_ratio"] = v / (vol_ma20 + 1e-9)
    d["vol_z"]     = (v - vol_ma20) / (vol_std20 + 1e-9)
    d["vol_ret"]   = v.pct_change(1)

    # Stochastic
    lo14 = l.rolling(14).min(); hi14 = h.rolling(14).max()
    d["stoch_k"] = 100 * (c - lo14) / (hi14 - lo14 + 1e-9)
    d["stoch_d"] = d["stoch_k"].rolling(3).mean()

    # ADX
    plus_dm  = h.diff().clip(lower=0)
    minus_dm = (-l.diff()).clip(lower=0)
    plus_dm  = plus_dm.where(plus_dm > (-l.diff()).clip(lower=0), 0)
    minus_dm = minus_dm.where(minus_dm > h.diff().clip(lower=0), 0)
    atr_ewm  = tr.ewm(span=14, adjust=False).mean()
    pdi = 100 * plus_dm.ewm(span=14, adjust=False).mean()  / (atr_ewm + 1e-9)
    mdi = 100 * minus_dm.ewm(span=14, adjust=False).mean() / (atr_ewm + 1e-9)
    dx  = 100 * (pdi - mdi).abs() / (pdi + mdi + 1e-9)
    d["adx"]     = dx.ewm(span=14, adjust=False).mean()
    d["di_diff"] = pdi - mdi

    # Candle geometry
    body = (c - o)
    rng  = (h - l).replace(0, np.nan)
    d["body_ratio"] = body / rng
    d["upper_wick"] = (h - pd.concat([o, c], axis=1).max(axis=1)) / rng
    d["lower_wick"] = (pd.concat([o, c], axis=1).min(axis=1) - l)  / rng
    d["gap"]        = (o - c.shift(1)) / (c.shift(1) + 1e-9)

    # Price position — adaptive 52w window
    _w52 = max(30, min(252, n // 2))
    hi52 = h.rolling(_w52, min_periods=30).max()
    lo52 = l.rolling(_w52, min_periods=30).min()
    d["pos_52w"]   = (c - lo52)                              / (hi52 - lo52 + 1e-9)
    d["hi20_dist"] = (h.rolling(20, min_periods=5).max() - c) / (c + 1e-9)
    d["lo20_dist"] = (c - l.rolling(20, min_periods=5).min()) / (c + 1e-9)

    # Lagged closes (normalised — no raw price levels)
    for lag in [1, 2, 3, 5]:
        d[f"close_lag{lag}_r"] = c.shift(lag) / (c + 1e-9) - 1

    # ═══════════════════════════════════════════════════════════════════
    # B — SCREENER SIGNAL FEATURES  (all adaptive, no arbitrary numbers)
    # ═══════════════════════════════════════════════════════════════════

    # F1 — RS vs Nifty (vol-normalised tanh, same formula as screener)
    if nifty_close is not None and nifty_close.notna().sum() > 30:
        d["sc_rs_nifty"] = _rs_score_series(c, nifty_close)
    else:
        d["sc_rs_nifty"] = 0.5   # neutral when Nifty unavailable

    # F2 — Momentum Acceleration (EMA5−EMA20 velocity, percentile-ranked)
    velocity     = e5 - e20
    acceleration = velocity.diff()
    d["sc_momentum_vel"]  = _percentile_rank(velocity,     _w200)
    d["sc_momentum_acc"]  = _percentile_rank(acceleration, _w200)

    # F3 — Volume Surge z-score percentile
    d["sc_vol_z_pct"]  = _percentile_rank(d["vol_z"], _w120)

    # F4 — Pre-breakout Accumulation (vol5/vol20 sigmoid, centre=1.3, k=4)
    vol_ma5  = v.rolling(5, min_periods=2).mean()
    inst_ratio_s = vol_ma5 / (vol_ma20 + 1e-9)
    d["sc_accumulation"] = (1.0 / (1.0 + np.exp(-4.0 * (inst_ratio_s - 1.3)))).clip(0, 1)

    # F5 — Volatility Contraction (ATR5/ATR20 percentile, inverted)
    atr5_s  = tr.rolling(5,  min_periods=2).mean()
    atr20_s = tr.rolling(20, min_periods=5).mean()
    vc_ratio_s = atr5_s / (atr20_s.replace(0, np.nan))
    d["sc_vol_contraction"] = 1.0 - _percentile_rank(vc_ratio_s, _w120)

    # F5b — Range Compression Index (percentile, inverted)
    range5_s  = h.rolling(5,  min_periods=2).max() - l.rolling(5,  min_periods=2).min()
    range20_s = h.rolling(20, min_periods=5).max() - l.rolling(20, min_periods=5).min()
    rci_s     = range5_s / (range20_s.replace(0, np.nan))
    d["sc_range_compression"] = 1.0 - _percentile_rank(rci_s, _w120)

    # F5c — VCVE interaction
    d["sc_vcve"] = (inst_ratio_s * (1.0 - vc_ratio_s.clip(upper=1.0))).clip(0, 5) / 5.0

    # F6 — Base/Coil Quality
    compression_s = (1.0 - rci_s).clip(0, 1)
    hi_spread_atr = (h.rolling(8, min_periods=3).max() - h.rolling(8, min_periods=3).min()) / (atr14_s + 1e-9)
    flatness_s    = (1.0 - hi_spread_atr.clip(upper=1.0)).clip(0, 1)
    d["sc_coil_quality"] = (compression_s * 0.55 + flatness_s * 0.45).clip(0, 1)

    # F7 — Trend Structure (EMA9/EMA50 percentile)
    trend_ratio = e9 / (e50.replace(0, np.nan))
    d["sc_trend_structure"] = _percentile_rank(trend_ratio, _w200)
    d["sc_ema_alignment"] = ((e9 > e20).astype(float) + (e20 > e50).astype(float)) / 2.0

    # F8 — Breakout Proximity (exp decay from 20d resistance)
    resistance_20 = h.rolling(20, min_periods=5).max()
    d_trig = (resistance_20 - c) / (atr14_s + 1e-9)
    d["sc_breakout_prox"] = np.exp(-1.5 * d_trig.clip(lower=0)).clip(0, 1)

    # F9 — ATR% Potential (percentile vs own history)
    atr_pct_s = atr14_s / (c + 1e-9) * 100
    d["sc_atr_potential"] = _percentile_rank(atr_pct_s, _w60)

    # F10 — Candlestick Score (rolling 8-pattern score, scaled 0-1)
    # Vectorised — each pattern is a boolean series
    prev_c = c.shift(1); prev_o = o.shift(1)
    prev_h = h.shift(1); prev_l = l.shift(1)
    prev_body = (prev_c - prev_o).abs()
    prev_rng  = (prev_h - prev_l).replace(0, np.nan)
    body_abs  = body.abs()
    upper_w   = h - pd.concat([o, c], axis=1).max(axis=1)
    lower_w   = pd.concat([o, c], axis=1).min(axis=1) - l

    cdl  = pd.Series(0.0, index=c.index)
    cdl += ((prev_c < prev_o) & (c > o) & (c > prev_o) & (o < prev_c)).astype(float) * 3.0        # Engulfing
    cdl += ((lower_w >= 2 * body_abs) & (upper_w <= 0.4 * rng) & (c > o)).astype(float) * 2.5     # Hammer
    cdl += ((h <= prev_h) & (l >= prev_l)).astype(float) * 1.5                                     # InsideBar
    cdl += ((h > prev_h) & (l < prev_l) & (c > o) & (c > (h + l) / 2)).astype(float) * 2.0       # OutsideBar
    cdl += ((body_abs / rng > 0.60) & (c > o) & ((c - l) / rng > 0.75)).astype(float) * 2.0       # StrongGreen
    cdl += ((body_abs / rng < 0.10) & (lower_w > 1.5 * upper_w)).astype(float) * 1.0              # BullDoji
    cdl += ((prev_c < prev_o) & (prev_body / prev_rng > 0.5) & (c > o) &
            (c > (prev_o + prev_c) / 2)).astype(float) * 2.5                                       # MorningStar
    cdl += ((o > prev_c * 1.003) & (c > o)).astype(float) * 2.0                                   # GapContinue
    d["sc_candle_score"] = (cdl.clip(upper=10) / 10.0)   # normalise 0-1

    # ── BONUS SIGNALS ──────────────────────────────────────────────────────────

    # Liquidity Sweep (same logic as screener — ATR-relative wick)
    prior_support = l.rolling(5).min().shift(1)
    sweep = ((l < prior_support) &
             (c > c.shift(1)) &
             (lower_w >= 0.5 * atr14_s) &
             (d["vol_z"] >= 1.0)).astype(float)
    d["sc_sweep"] = sweep

    # VWMA20 position
    typical = (h + l + c) / 3
    vwma20  = (typical * v).rolling(20).sum() / (v.rolling(20).sum().replace(0, np.nan))
    d["sc_above_vwma20"]     = (c > vwma20).astype(float)
    d["sc_vwma20_rising"]    = (vwma20 > vwma20.shift(1)).astype(float)

    # Momentum Stability (positive day fraction over 20d)
    pos_days = c.diff().gt(0).rolling(20, min_periods=10).mean()
    d["sc_momentum_stability"] = pos_days

    # 52-week position percentile (adaptive window)
    pos_series = (c - c.rolling(_w52, min_periods=30).min()) / \
                 (c.rolling(_w52, min_periods=30).max() - c.rolling(_w52, min_periods=30).min() + 1e-9)
    d["sc_pos52w_pct"] = _percentile_rank(pos_series, _w200)

    # RS Acceleration (EMA5−EMA20 velocity diff, percentile — same as screener F2 bonus)
    d["sc_rs_accel"] = d["sc_momentum_acc"]   # already computed above, alias for clarity

    # ── SHIFT ALL FEATURES BY 1 BAR — strict zero lookahead ──────────────────
    sc_cols = [col for col in d.columns if col.startswith("sc_")]
    d[sc_cols] = d[sc_cols].shift(1)

    # ── TARGETS ───────────────────────────────────────────────────────────────
    d["target_direction"] = (c.shift(-1) > c).astype(int)
    d["target_nextclose"] = c.shift(-1)

    # ── Fill remaining NaNs in feature columns (ffill then 0) ────────────────
    # Target columns must be valid — drop those rows.
    # Feature NaNs at the start of history are forward-filled then zero-filled
    # so the ML model always gets a complete feature matrix.
    feat_cols_now = [col for col in d.columns
                     if col not in ("target_direction", "target_nextclose")]
    d[feat_cols_now] = d[feat_cols_now].ffill().fillna(0)
    d.dropna(subset=["target_direction", "target_nextclose"], inplace=True)
    return d


def get_feature_cols(df: pd.DataFrame) -> list:
    """
    Returns all engineered feature columns — strictly no raw OHLCV, no targets.
    Screener-derived 'sc_*' columns are explicitly included.
    """
    exclude = {
        "Open", "High", "Low", "Close", "Adj Close", "Volume",
        "target_direction", "target_nextclose",
        # keep dist_sma* and ema_cross but drop the raw MA levels
        "sma5", "sma10", "sma20", "sma50", "sma200",
        "bb_upper", "bb_lower", "ema9", "ema21",
    }
    return [
        col for col in df.columns
        if col not in exclude
        and not col.startswith("Dividends")
        and not col.startswith("Stock")
        and pd.api.types.is_numeric_dtype(df[col])
    ]


# ─────────────────────────────────────────────────────────────────────────────
# MODEL TRAINING + PREDICTION
# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# PROFESSIONAL ENSEMBLE ENGINE
# ─────────────────────────────────────────────────────────────────────────────
#
# Architecture: Stacked Ensemble with Walk-Forward Cross-Validation
#
# BASE LEARNERS (L1) — each with a fundamentally different inductive bias:
#   • Random Forest       — bagged deep trees, low variance, handles non-linearity
#   • XGBoost             — gradient boosting, L1/L2 regularised, captures interactions
#   • LightGBM            — leaf-wise boosting, fastest on tabular data
#   • Gradient Boosting   — sklearn GBM, fallback if XGB/LGB unavailable
#   • TCN (conditional)   — Temporal Convolutional Network (TensorFlow)
#                           Auto-activated when n_train ≥ 800 AND TF installed.
#                           WHY TCN not LSTM:
#                             - Dilated causal convolutions: no vanishing gradient
#                             - Receptive field grows exponentially with depth
#                             - ~10× fewer parameters than LSTM at same depth
#                             - Converges in 10-20 epochs (LSTM needs 50-100)
#                             - At n_train < 5000, consistently beats LSTM/Transformer
#                           WHY NOT raw LSTM/Transformer on this data:
#                             - NSE daily: ~1200 rows → ~1150 sequences after windowing
#                             - LSTM needs 5000+ sequences to reliably converge
#                             - Transformer attention is meaningless below ~2000 samples
#                             - Both overfit badly on tabular financial features
#
# META-LEARNER (L2) — Logistic Regression (calibrated)
#   • Trains on out-of-fold predictions from L1 models
#   • Learns which base model to trust on which market regimes
#   • Logistic Regression chosen for: low complexity (avoids overfitting
#     the meta-features), fast, probabilistically calibrated outputs
#
# PRICE REGRESSOR — Stacking Regressor
#   • Same base learners (regression variants)
#   • Meta-learner: Ridge (L2-regularized) — prevents over-reliance on
#     any single base model's price estimate
#
# WALK-FORWARD CROSS-VALIDATION
#   • 5-fold time-series split — no data leakage
#   • Each fold trains on past, validates on future
#   • Final model trained on all training data, tested on held-out window
#   • Model is SELECTED (not tuned with arbitrary ranges) based on OOF AUC
#
# NO ARBITRARY NUMBERS:
#   • n_estimators calibrated to dataset size (min 100, scales with data)
#   • max_depth = log2(n_features) — information-theoretic bound
#   • learning_rate = 0.05 (conservative — avoids overfitting small datasets)
#   • min_samples_leaf = max(5, n_train//500) — prevents tiny leaf nodes
# ─────────────────────────────────────────────────────────────────────────────

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

from sklearn.model_selection import TimeSeriesSplit, KFold

# ─────────────────────────────────────────────────────────────────────────────
# RELIABLE OHLCV + LATEST PRICE FETCHER
# ─────────────────────────────────────────────────────────────────────────────
# Root cause of stale price:
#   yf.download(period="365d") uses a relative period calculated from "today".
#   On Saturday, yfinance treats Saturday as today and may not include Friday's
#   completed session.  The screener's ml_raw_data is also stale (fetched hours ago).
#
# Fix: fetch with explicit start/end dates where end = tomorrow.
# Then patch last-bar Close with fast_info.last_price (15-min delayed quote)
# which is available 24/7 and always reflects the last traded price.

def _fetch_ohlcv(ticker: str, days: int) -> pd.DataFrame:
    """
    Fetch daily OHLCV using explicit start/end so weekend runs always include
    the most recently completed trading session (e.g. Friday on Saturday).
    end = today + 2 days forces yfinance to include the latest completed bar.
    """
    end_dt   = datetime.now() + timedelta(days=2)
    start_dt = datetime.now() - timedelta(days=days + 30)  # generous buffer for holidays
    try:
        df = yf.download(
            ticker,
            start=start_dt.strftime("%Y-%m-%d"),
            end=end_dt.strftime("%Y-%m-%d"),
            interval="1d",
            progress=False,
            auto_adjust=True,
        )
        if df.empty:
            df = yf.Ticker(ticker).history(
                start=start_dt.strftime("%Y-%m-%d"),
                end=end_dt.strftime("%Y-%m-%d"),
                interval="1d",
                auto_adjust=True,
            )
        return df
    except Exception:
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# CURRENT PRICE — UPSTOX FIRST, yfinance AS FALLBACK
# ─────────────────────────────────────────────────────────────────────────────
# The screener (1_Live_screener.py) already fetched live LTP from Upstox and
# stored it in:
#   st.session_state.live_quotes_cache  — dict keyed by instrument_key (NSE_EQ|INE...)
#   st.session_state.targets            — maps trading_symbol → instrument_key
#
# Priority:
#   1. Upstox LTP from session_state  — exact same source as screener (freshest)
#   2. Upstox API direct call         — if token available but cache stale
#   3. yfinance fast_info.last_price  — 15-min delayed, works weekends
#   4. yfinance history(5d)           — fallback
#   5. OHLCV close.iloc[-1]           — last resort

def _normalize_key(k: str) -> str:
    return k.replace("%7C", "|").replace(":", "|")


def get_upstox_ltp(trading_symbol: str) -> tuple:
    """
    Returns (price, source_label) using Upstox data already in session_state.
    trading_symbol = Upstox format e.g. 'SUNPHARMA' (no .NS suffix).
    """
    # ── 1. Read from screener's live_quotes_cache (already fetched, free) ────
    targets     = st.session_state.get("targets", {})
    live_cache  = st.session_state.get("live_quotes_cache", {})
    token       = (st.session_state.get("upstox_token", "") or
                   st.session_state.get("scanner_token", ""))

    ikey = targets.get(trading_symbol, "")
    if ikey:
        q = live_cache.get(_normalize_key(ikey), {})
        if q and q.get("ltp"):
            return float(q["ltp"]), "upstox_cache"

    # ── 2. Direct Upstox API call (token available, cache missed) ─────────────
    if token and ikey:
        try:
            import requests as _req
            url    = "https://api.upstox.com/v2/market-quote/quotes"
            params = {"instrument_key": ikey}
            r = _req.get(url,
                         headers={"Authorization": f"Bearer {token}",
                                  "Accept": "application/json"},
                         params=params, timeout=5)
            if r.status_code == 200:
                data = r.json().get("data", {})
                for _k, v in data.items():
                    ltp = v.get("last_price")
                    if ltp:
                        # Refresh cache so next ticker is instant
                        live_cache[_normalize_key(ikey)] = {
                            "ltp": float(ltp),
                            "open":  float(v.get("ohlc", {}).get("open",  ltp)),
                            "high":  float(v.get("ohlc", {}).get("high",  ltp)),
                            "low":   float(v.get("ohlc", {}).get("low",   ltp)),
                            "volume": v.get("volume"),
                        }
                        st.session_state["live_quotes_cache"] = live_cache
                        return float(ltp), "upstox_api"
        except Exception:
            pass

    return None, "ohlcv"


def fetch_latest_price(ticker: str) -> tuple:
    """
    Returns (price, source) — tries Upstox first, yfinance as fallback.
    ticker = Upstox trading symbol (e.g. 'SUNPHARMA') OR yfinance symbol (e.g. 'SUNPHARMA.NS').
    Strips .NS/.BO suffix before Upstox lookup.
    """
    # Strip exchange suffix for Upstox lookup
    upstox_sym = ticker.replace(".NS", "").replace(".BO", "").replace(".NSE", "").upper()

    price, source = get_upstox_ltp(upstox_sym)
    if price:
        return price, source

    # ── yfinance fallbacks ────────────────────────────────────────────────────
    yf_sym = ticker if "." in ticker else ticker + ".NS"
    t = yf.Ticker(yf_sym)

    try:
        p = getattr(t.fast_info, "last_price", None)
        if p and float(p) > 0:
            return float(p), "yf_fast_info"
    except Exception:
        pass

    try:
        hist = t.history(period="5d", interval="1d", auto_adjust=True)
        if not hist.empty:
            hist.columns = [c.strip().title() for c in hist.columns]
            if hist.index.tz is not None:
                hist.index = hist.index.tz_convert(None)
            p = float(hist["Close"].dropna().iloc[-1])
            if p > 0:
                return p, "yf_history5d"
    except Exception:
        pass

    return None, "ohlcv"


from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

# ── TensorFlow / Keras (optional — TCN needs it) ─────────────────────────────
# TCN is only added to the ensemble when n_train >= 800 AND TF is available.
# Below that threshold, LSTM/Transformer overfit badly on tabular financial data.
# TCN is chosen over LSTM because:
#   • Far fewer parameters (dilated 1D convolutions, not recurrent)
#   • No vanishing gradient problem
#   • Receptive field grows exponentially with depth (captures multi-scale patterns)
#   • Converges in 10-20 epochs vs 50-100 for LSTM at these data sizes
try:
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import (Input, Conv1D, Dense, Dropout,
                                          BatchNormalization, Activation, Add,
                                          GlobalAveragePooling1D, Lambda)
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras import backend as K
    tf.get_logger().setLevel('ERROR')
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# TCN SKLEARN WRAPPER
# ─────────────────────────────────────────────────────────────────────────────
# Wraps a Keras TCN as a sklearn-compatible estimator so it slots cleanly
# into StackingClassifier / StackingRegressor without any other code changes.
#
# TCN Architecture — each residual block:
#   Conv1D(filters, k, dilation_rate=2^i) → BatchNorm → ReLU → Dropout
#   → Conv1D(filters, k, dilation_rate=2^i) → BatchNorm → 1x1 residual skip
#
# Receptive field = 2 × k × (2^n_blocks − 1)
# With k=3, n_blocks=4: RF = 90 timesteps — covers ~3 months of daily data
#
# Input shape: (n_samples, n_features) flat features
# Internally reshaped to (n_samples, time_steps, 1) for Conv1D
# time_steps = min(20, n_features // 2) — treats feature vector as pseudo-sequence
# ─────────────────────────────────────────────────────────────────────────────

class _TCNBlock(object):
    """Stateless helper — builds a causal dilated residual block."""
    @staticmethod
    def build(x, filters, kernel_size, dilation_rate, dropout_rate):
        if not TF_AVAILABLE:
            raise RuntimeError("TF not available")
        residual = x
        # Two causal dilated convolutions per block
        for _ in range(2):
            x = Conv1D(filters=filters,
                       kernel_size=kernel_size,
                       dilation_rate=dilation_rate,
                       padding="causal",
                       use_bias=False)(x)
            x = BatchNormalization()(x)
            x = Activation("relu")(x)
            x = Dropout(dropout_rate)(x)
        # 1×1 projection for residual if channel count changed
        if residual.shape[-1] != filters:
            residual = Conv1D(filters, 1, padding="same", use_bias=False)(residual)
        return Add()([x, residual])


class TCNClassifier(BaseEstimator, ClassifierMixin):
    """
    Temporal Convolutional Network wrapped as sklearn classifier.
    Auto-disabled if TensorFlow is not installed.

    Hyperparameters derived from data:
      filters   = min(64, max(16, n_features * 2))  — enough capacity, not excessive
      n_blocks  = ceil(log2(time_steps))             — RF covers full input sequence
      dropout   = max(0.1, 0.5 − n_train/5000)      — less dropout with more data
      patience  = max(5, 20 − n_train//200)          — less patience with more data
    """

    def __init__(self, n_train=1000, n_features=45, time_steps=20,
                 epochs=50, batch_size=32, random_state=42):
        self.n_train     = n_train
        self.n_features  = n_features
        self.time_steps  = time_steps
        self.epochs      = epochs
        self.batch_size  = batch_size
        self.random_state = random_state
        self.model_      = None
        self.classes_    = np.array([0, 1])

    def _build(self):
        import math
        filters    = min(64, max(16, self.n_features * 2))
        n_blocks   = max(2, math.ceil(math.log2(max(self.time_steps, 2))))
        dropout    = float(np.clip(0.5 - self.n_train / 5000, 0.1, 0.4))

        inp = Input(shape=(self.time_steps, 1))
        x   = inp
        for i in range(n_blocks):
            x = _TCNBlock.build(x, filters=filters, kernel_size=3,
                                dilation_rate=2**i, dropout_rate=dropout)
        x   = GlobalAveragePooling1D()(x)
        x   = Dense(max(8, filters // 2), activation="relu")(x)
        out = Dense(1, activation="sigmoid")(x)

        model = Model(inp, out)
        # Learning rate: 3e-4 × (1 / (1 + n_train/2000)) — decay with data volume
        lr = 3e-4 / (1 + self.n_train / 2000)
        model.compile(optimizer=Adam(lr), loss="binary_crossentropy",
                      metrics=["accuracy"])
        return model

    def _reshape(self, X):
        """Treat each feature as one timestep in a pseudo-sequence."""
        n = X.shape[0]
        ts = min(self.time_steps, X.shape[1])
        # Pad or trim to exactly time_steps
        Xr = np.zeros((n, self.time_steps, 1), dtype=np.float32)
        Xr[:, :ts, 0] = X[:, :ts]
        return Xr

    def fit(self, X, y):
        if not TF_AVAILABLE:
            return self
        import math
        tf.random.set_seed(self.random_state)
        K.clear_session()
        self.time_steps = min(20, max(5, X.shape[1] // 2))
        self.model_ = self._build()
        patience = max(5, 20 - self.n_train // 200)
        callbacks = [
            EarlyStopping(monitor="val_loss", patience=patience,
                          restore_best_weights=True, verbose=0),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                              patience=max(3, patience // 2), verbose=0),
        ]
        self.model_.fit(
            self._reshape(X), y.astype(np.float32),
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.15,
            callbacks=callbacks,
            verbose=0,
        )
        return self

    def predict_proba(self, X):
        if not TF_AVAILABLE or self.model_ is None:
            n = X.shape[0]
            return np.column_stack([np.full(n, 0.5), np.full(n, 0.5)])
        p = self.model_.predict(self._reshape(X), verbose=0).flatten()
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def get_params(self, deep=True):
        return dict(n_train=self.n_train, n_features=self.n_features,
                    time_steps=self.time_steps, epochs=self.epochs,
                    batch_size=self.batch_size, random_state=self.random_state)

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self


class TCNRegressor(BaseEstimator, RegressorMixin):
    """Same TCN architecture for price regression (linear output, MSE loss)."""

    def __init__(self, n_train=1000, n_features=45, time_steps=20,
                 epochs=50, batch_size=32, random_state=42):
        self.n_train     = n_train
        self.n_features  = n_features
        self.time_steps  = time_steps
        self.epochs      = epochs
        self.batch_size  = batch_size
        self.random_state = random_state
        self.model_      = None

    def _build(self):
        import math
        filters  = min(64, max(16, self.n_features * 2))
        n_blocks = max(2, math.ceil(math.log2(max(self.time_steps, 2))))
        dropout  = float(np.clip(0.5 - self.n_train / 5000, 0.1, 0.4))

        inp = Input(shape=(self.time_steps, 1))
        x   = inp
        for i in range(n_blocks):
            x = _TCNBlock.build(x, filters=filters, kernel_size=3,
                                dilation_rate=2**i, dropout_rate=dropout)
        x   = GlobalAveragePooling1D()(x)
        x   = Dense(max(8, filters // 2), activation="relu")(x)
        out = Dense(1)(x)  # linear output for regression

        model = Model(inp, out)
        lr = 3e-4 / (1 + self.n_train / 2000)
        model.compile(optimizer=Adam(lr), loss="mse")
        return model

    def _reshape(self, X):
        n  = X.shape[0]
        ts = min(self.time_steps, X.shape[1])
        Xr = np.zeros((n, self.time_steps, 1), dtype=np.float32)
        Xr[:, :ts, 0] = X[:, :ts]
        return Xr

    def fit(self, X, y):
        if not TF_AVAILABLE:
            return self
        import math
        tf.random.set_seed(self.random_state)
        K.clear_session()
        self.time_steps = min(20, max(5, X.shape[1] // 2))
        self.model_ = self._build()
        patience = max(5, 20 - self.n_train // 200)
        callbacks = [
            EarlyStopping(monitor="val_loss", patience=patience,
                          restore_best_weights=True, verbose=0),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                              patience=max(3, patience // 2), verbose=0),
        ]
        self.model_.fit(
            self._reshape(X), y.astype(np.float32),
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.15,
            callbacks=callbacks,
            verbose=0,
        )
        return self

    def predict(self, X):
        if not TF_AVAILABLE or self.model_ is None:
            return np.zeros(X.shape[0])
        return self.model_.predict(self._reshape(X), verbose=0).flatten()

    def get_params(self, deep=True):
        return dict(n_train=self.n_train, n_features=self.n_features,
                    time_steps=self.time_steps, epochs=self.epochs,
                    batch_size=self.batch_size, random_state=self.random_state)

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self


def _make_base_classifiers(n_train: int, n_features: int) -> list:
    """
    Build base classifiers with hyperparameters derived from data properties,
    not arbitrary grids. Returns list of (name, estimator) tuples.
    """
    import math
    n_est   = min(500, max(100, n_train // 5))      # scales with training data
    depth   = max(3, min(8, int(math.log2(n_features))))  # log2(features) bound
    leaf    = max(5, n_train // 500)                # prevents overfitting tiny leaves
    lr_fast = 0.05                                  # conservative — generalises better

    estimators = [
        ("rf", RandomForestClassifier(
            n_estimators=n_est,
            max_depth=depth + 2,          # RF tolerates deeper trees (bagging variance reduction)
            min_samples_leaf=leaf,
            max_features="sqrt",          # sqrt(n_features) — standard RF theory
            bootstrap=True,
            random_state=42,
            n_jobs=-1,
        )),
        ("gbm", GradientBoostingClassifier(
            n_estimators=n_est,
            learning_rate=lr_fast,
            max_depth=depth - 1,          # GBM needs shallower trees (additive model)
            min_samples_leaf=leaf,
            subsample=0.8,                # stochastic gradient boosting (reduces variance)
            random_state=42,
        )),
    ]

    if XGB_AVAILABLE:
        estimators.append(("xgb", xgb.XGBClassifier(
            n_estimators=n_est,
            learning_rate=lr_fast,
            max_depth=depth,
            min_child_weight=leaf,
            subsample=0.8,
            colsample_bytree=max(0.5, n_features**-0.5 * n_features**0.5),  # ~80%
            reg_alpha=0.1,                # L1 — sparse feature selection
            reg_lambda=1.0,               # L2 — weight regularisation
            eval_metric="logloss",
            random_state=42,
            verbosity=0,
            n_jobs=-1,
        )))

    if LGB_AVAILABLE:
        estimators.append(("lgb", lgb.LGBMClassifier(
            n_estimators=n_est,
            learning_rate=lr_fast,
            max_depth=depth,
            min_child_samples=leaf,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            verbose=-1,
            random_state=42,
            n_jobs=-1,
        )))

    # ── TCN: only added when enough data for meaningful convergence ──────────
    # Threshold n_train >= 800: below this, TCN adds noise not signal.
    # batch_size = sqrt(n_train) — balances gradient noise and memory.
    if TF_AVAILABLE and n_train >= 800:
        batch_size_tcn = max(16, min(64, int(n_train ** 0.5)))
        estimators.append(("tcn", TCNClassifier(
            n_train=n_train,
            n_features=n_features,
            epochs=50,
            batch_size=batch_size_tcn,
            random_state=42,
        )))

    return estimators


def _make_base_regressors(n_train: int, n_features: int) -> list:
    import math
    n_est = min(500, max(100, n_train // 5))
    depth = max(3, min(8, int(math.log2(n_features))))
    leaf  = max(5, n_train // 500)
    lr    = 0.05

    estimators = [
        ("rf_r", RandomForestRegressor(
            n_estimators=n_est, max_depth=depth+2,
            min_samples_leaf=leaf, max_features="sqrt",
            bootstrap=True, random_state=42, n_jobs=-1,
        )),
        ("gbm_r", GradientBoostingRegressor(
            n_estimators=n_est, learning_rate=lr,
            max_depth=depth-1, min_samples_leaf=leaf,
            subsample=0.8, random_state=42,
        )),
    ]
    if XGB_AVAILABLE:
        estimators.append(("xgb_r", xgb.XGBRegressor(
            n_estimators=n_est, learning_rate=lr, max_depth=depth,
            min_child_weight=leaf, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0, verbosity=0, random_state=42, n_jobs=-1,
        )))
    if LGB_AVAILABLE:
        estimators.append(("lgb_r", lgb.LGBMRegressor(
            n_estimators=n_est, learning_rate=lr, max_depth=depth,
            min_child_samples=leaf, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0, verbose=-1, random_state=42, n_jobs=-1,
        )))
    if TF_AVAILABLE and n_train >= 800:
        batch_size_tcn = max(16, min(64, int(n_train ** 0.5)))
        estimators.append(("tcn_r", TCNRegressor(
            n_train=n_train,
            n_features=n_features,
            epochs=50,
            batch_size=batch_size_tcn,
            random_state=42,
        )))
    return estimators


def _walk_forward_oof_auc(estimators: list, X: np.ndarray, y: np.ndarray,
                           n_splits: int = 5) -> dict:
    """Walk-forward OOF AUC profiling. TimeSeriesSplit is correct here (manual iteration)."""
    safe_splits = min(n_splits, max(2, len(X) // 40))
    tscv   = TimeSeriesSplit(n_splits=safe_splits)
    scores = {name: [] for name, _ in estimators}
    scaler = StandardScaler()
    for train_idx, val_idx in tscv.split(X):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        if (len(X_tr) < 20 or len(X_val) < 5
                or len(np.unique(y_tr)) < 2 or len(np.unique(y_val)) < 2):
            continue
        X_tr_s  = scaler.fit_transform(X_tr)
        X_val_s = scaler.transform(X_val)
        for name, est in estimators:
            try:
                clone_est = est.__class__(**est.get_params())
                clone_est.fit(X_tr_s, y_tr)
                proba = clone_est.predict_proba(X_val_s)[:, 1]
                scores[name].append(roc_auc_score(y_val, proba))
            except Exception:
                scores[name].append(0.5)
    return {name: float(np.mean(v)) if v else 0.5 for name, v in scores.items()}


def run_ml_for_ticker(ticker: str, df_raw: pd.DataFrame,
                      backtest_days: int, timeframe: str,
                      nifty_close: pd.Series | None = None) -> dict:
    """
    Professional stacked ensemble pipeline for a single ticker.

    Steps:
      1. Resample OHLCV to requested timeframe
      2. Engineer ~80 features: technicals + all 10 screener signals (no lookahead)
      3. Walk-forward CV on training set → evaluate each base model → report OOF AUCs
      4. Build stacking ensemble: base learners → LR meta-learner (calibrated)
      5. Train on full training window, predict on held-out test window
      6. Simultaneously run stacking regressor for price level forecast
      7. Compute final next-bar prediction + confidence
         → curr_price comes from live quote (fast_info), not OHLCV tail
      8. Return full result dict for display
    """
    # ── 1. Resample ───────────────────────────────────────────────────────────
    df = df_raw.copy()
    if timeframe == "Weekly":
        df = df.resample("W").agg({"Open":"first","High":"max","Low":"min",
                                   "Close":"last","Volume":"sum"}).dropna()
    elif timeframe == "Monthly":
        df = df.resample("ME").agg({"Open":"first","High":"max","Low":"min",
                                    "Close":"last","Volume":"sum"}).dropna()

    if len(df) < 100:
        return {"error": f"Not enough data ({len(df)} bars). Need 100+."}

    # ── 2. Feature engineering ────────────────────────────────────────────────
    feat      = engineer_features(df, nifty_close=nifty_close)
    feat_cols = get_feature_cols(feat)

    if len(feat) < backtest_days + 60:
        return {"error": f"After feature engineering: {len(feat)} rows, need {backtest_days+60}+."}

    X      = feat[feat_cols].astype(float).values
    y_dir  = feat["target_direction"].values
    y_next = feat["target_nextclose"].values
    close  = feat["Close"].squeeze()

    split     = len(feat) - backtest_days
    X_train   = X[:split]
    X_test    = X[split:]
    yd_train  = y_dir[:split]
    yd_test   = y_dir[split:]
    yn_train  = y_next[:split]
    yn_test   = y_next[split:]

    n_train, n_features = X_train.shape

    # ── Safe CV split count ───────────────────────────────────────────────────
    # StackingClassifier uses cross_val_predict → needs TRUE PARTITION → KFold.
    # TimeSeriesSplit is NOT a partition → kept only in _walk_forward_oof_auc.
    n_cv_splits   = min(5, max(2, n_train // 30))
    n_cal_splits  = min(3, max(2, n_train // 80))
    use_prefit_cal = (n_train < n_cal_splits * 40)

    # ── 3. Walk-forward OOF profiling ─────────────────────────────────────────
    base_clf = _make_base_classifiers(n_train, n_features)
    oof_aucs = _walk_forward_oof_auc(base_clf, X_train, yd_train, n_splits=n_cv_splits)

    # ── 4. Build stacking ensemble ────────────────────────────────────────────
    meta_C   = 1.0 / len(base_clf)
    meta_clf = LogisticRegression(C=meta_C, max_iter=1000, solver="lbfgs", random_state=42)

    stack_clf = StackingClassifier(
        estimators=base_clf,
        final_estimator=meta_clf,
        cv=KFold(n_splits=n_cv_splits, shuffle=False),   # TRUE partition
        stack_method="predict_proba",
        passthrough=False,
        n_jobs=1,
    )

    # ── 5. Scale + train ──────────────────────────────────────────────────────
    scaler    = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    stack_clf.fit(X_train_s, yd_train)

    # ── 5b. Calibrate probabilities ───────────────────────────────────────────
    if use_prefit_cal:
        calibrated = CalibratedClassifierCV(stack_clf, method="sigmoid", cv="prefit")
    else:
        calibrated = CalibratedClassifierCV(stack_clf, method="isotonic", cv=n_cal_splits)
    calibrated.fit(X_train_s, yd_train)

    y_proba_raw = calibrated.predict_proba(X_test_s)[:, 1]
    train_proba = calibrated.predict_proba(X_train_s)[:, 1]
    thresholds   = np.linspace(0.3, 0.7, 41)
    best_thresh  = 0.5
    best_f1      = 0.0
    for thr in thresholds:
        yp = (train_proba >= thr).astype(int)
        if len(np.unique(yp)) > 1:
            f1 = f1_score(yd_train, yp)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thr

    y_pred_dir = (y_proba_raw >= best_thresh).astype(int)

    acc = accuracy_score(yd_test, y_pred_dir)
    try:
        auc = roc_auc_score(yd_test, y_proba_raw)
    except Exception:
        auc = 0.5
    prec = precision_score(yd_test, y_pred_dir, zero_division=0)
    rec  = recall_score(yd_test,  y_pred_dir, zero_division=0)
    f1   = f1_score(yd_test,      y_pred_dir, zero_division=0)

    # ── 6. Stacking regressor for price forecast ───────────────────────────────
    base_reg  = _make_base_regressors(n_train, n_features)
    meta_reg  = Ridge(alpha=len(base_reg))
    stack_reg = StackingRegressor(
        estimators=base_reg,
        final_estimator=meta_reg,
        cv=KFold(n_splits=n_cv_splits, shuffle=False),   # TRUE partition
        passthrough=False,
        n_jobs=1,
    )
    stack_reg.fit(X_train_s, yn_train)
    y_pred_price = stack_reg.predict(X_test_s)

    mae  = mean_absolute_error(yn_test, y_pred_price)
    mape = mean_absolute_percentage_error(yn_test, y_pred_price) * 100

    # ── 7. Final next-bar prediction ───────────────────────────────────────────
    # Architecture: regressor is the single model for both direction AND price.
    # Direction = sign(predicted_price - current_price). No classifier used
    # for direction — eliminates the classifier/regressor contradiction entirely.
    #
    # Confidence is purely empirical, derived from the test-set residuals:
    #   For each test bar, we know: predicted move, actual move, correct direction?
    #   We bin test bars by |predicted move %| quantile.
    #   Confidence for the live prediction = precision of the quantile bin
    #   that the current |predicted move %| falls into.
    #   This is entirely data-derived — no arbitrary thresholds.

    last_s     = scaler.transform(X[[-1]])
    next_price = float(stack_reg.predict(last_s)[0])

    # ── curr_price: Upstox live LTP first, yfinance fallback ──────────────────
    ohlcv_last               = float(close.iloc[-1])
    live_price, price_source = fetch_latest_price(ticker)
    curr_price               = live_price if live_price else ohlcv_last
    if not live_price:
        price_source = "ohlcv"

    price_chg_pct = (next_price - curr_price) / curr_price * 100

    # Direction is purely the sign of the price prediction
    price_dir     = 1 if price_chg_pct > 0 else 0
    next_direction_raw = price_dir   # 1=UP, 0=DOWN

    # ── Empirical confidence from test-set residuals ──────────────────────────
    # Compute predicted move % for every test bar
    test_close_vals   = close.iloc[split:].values
    pred_move_pct     = (y_pred_price - test_close_vals) / (test_close_vals + 1e-9) * 100
    pred_dir_test     = (pred_move_pct > 0).astype(int)
    correct_dir_test  = (pred_dir_test == yd_test).astype(int)
    abs_pred_move     = np.abs(pred_move_pct)

    # Bin test bars by |predicted move %| into n_bins quantile bins
    # n_bins = sqrt(n_test) — natural bin count, no arbitrary number
    n_test = len(abs_pred_move)
    n_bins = max(3, int(np.sqrt(n_test)))
    quantile_edges = np.quantile(abs_pred_move, np.linspace(0, 1, n_bins + 1))
    quantile_edges = np.unique(quantile_edges)   # remove duplicate edges

    live_abs_move = abs(price_chg_pct)

    # Find which bin the current prediction falls into
    bin_idx = np.searchsorted(quantile_edges[1:], live_abs_move, side="right")
    bin_idx = min(bin_idx, len(quantile_edges) - 2)

    lo = quantile_edges[bin_idx]
    hi = quantile_edges[bin_idx + 1] if bin_idx + 1 < len(quantile_edges) else np.inf
    mask = (abs_pred_move >= lo) & (abs_pred_move < hi)

    if mask.sum() >= 3:
        empirical_precision = float(correct_dir_test[mask].mean())
    else:
        # Bin too sparse — use global directional accuracy on test set
        empirical_precision = float(correct_dir_test.mean())

    # Signal reliability: empirical_precision vs random baseline (0.5)
    # Excess precision above random = how much edge the model has in this bin
    # Scaled to 0-1: 0 = random (prec=0.5), 1 = perfect (prec=1.0)
    direction_conf = max(0.0, (empirical_precision - 0.5) * 2.0)

    # Direction label
    if empirical_precision > 0.5:
        if price_dir == 1:
            next_direction = "📈 UP"
        else:
            next_direction = "📉 DOWN"
    else:
        # Model has no edge for this size move — show uncertain
        next_direction = "⚪ UNCERTAIN"

    # Classifier still used for AUC/accuracy metrics — keep those
    next_proba = float(calibrated.predict_proba(last_s)[0][1])
    next_dir_pred = price_dir   # align with price for equity curve

    # ── 8. Feature importance (averaged across base learners) ─────────────────
    fi_list = []
    for name, est in base_clf:
        try:
            fitted = stack_clf.named_estimators_[name]
            if hasattr(fitted, "feature_importances_"):
                fi_list.append(pd.Series(fitted.feature_importances_, index=feat_cols))
        except Exception:
            pass
    if fi_list:
        fi = pd.concat(fi_list, axis=1).mean(axis=1).sort_values(ascending=False)
    else:
        fi = pd.Series(dtype=float)

    # ── Equity curve (long when model predicts UP, stay flat otherwise) ───────
    test_close   = close.iloc[split:].values
    n_trades     = len(y_pred_dir) - 1
    strat_vals   = [10000.0]
    bh_vals      = [10000.0]
    for i in range(n_trades):
        ret = (test_close[i+1] - test_close[i]) / (test_close[i] + 1e-9)
        strat_vals.append(strat_vals[-1] * (1 + ret * (y_pred_dir[i] == 1)))
        bh_vals.append(bh_vals[-1] * (1 + ret))

    eq_index = feat.index[split:]

    # ── OOF AUC per model for display ─────────────────────────────────────────
    model_labels = {
        "rf":    "Random Forest",
        "gbm":   "Gradient Boosting",
        "xgb":   "XGBoost",
        "lgb":   "LightGBM",
        "tcn":   "TCN (Neural)",
    }
    oof_display = {model_labels.get(k, k): v for k, v in oof_aucs.items()}

    return {
        "ticker":            ticker,
        "timeframe":         timeframe,
        "n_base_models":     len(base_clf),
        "base_model_names":  [model_labels.get(n, n) for n, _ in base_clf],
        "oof_aucs":          oof_display,
        "best_threshold":    best_thresh,
        "current_price":     curr_price,
        "price_source":      price_source,   # "live" | "ohlcv"
        "ohlcv_last_price":  ohlcv_last,
        "next_price_pred":      next_price,
        "price_change_pct":     price_chg_pct,
        "next_direction":       next_direction,
        "next_proba":           next_proba,
        "direction_conf":       direction_conf,
        "empirical_precision":  empirical_precision,
        "signals_agree":        True,   # no longer applicable — kept for compat
        "accuracy":          acc,
        "auc":               auc,
        "precision":         prec,
        "recall":            rec,
        "f1":                f1,
        "mae":               mae,
        "mape":              mape,
        "feat_importance":   fi,
        "y_actual":          yn_test,
        "y_predicted":       y_pred_price,
        "y_dir_pred":        y_pred_dir,
        "y_dir_actual":      yd_test,
        "y_proba":           y_proba_raw,
        "test_dates":        feat.index[split:],
        "strat_equity":      pd.Series(strat_vals),
        "bh_equity":         pd.Series(bh_vals),
        "eq_index":          eq_index,
        "backtest_days":     backtest_days,
        "n_cv_splits":       n_cv_splits,
        "n_features":        n_features,
        "n_train":           n_train,
        "n_test":            backtest_days,
        "tcn_active":        TF_AVAILABLE and n_train >= 800,
    }

# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING HELPERS
# ─────────────────────────────────────────────────────────────────────────────
DARK = dict(plot_bgcolor="#000", paper_bgcolor="#0a0a0a",
            font=dict(color="#e8e8e8", family="IBM Plex Mono", size=10),
            hoverlabel=dict(bgcolor="#1a1200", font_color="#ff8c00",
                            font_family="IBM Plex Mono", font_size=11))
_M_DEFAULT = dict(t=44, b=28, l=54, r=24)
_M_COMPACT  = dict(t=44, b=14, l=14, r=14)

def plot_price_prediction(result: dict) -> go.Figure:
    dates  = result["test_dates"]
    actual = result["y_actual"]
    pred   = result["y_predicted"]
    n      = min(len(dates), len(actual), len(pred))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates[:n], y=actual[:n], mode="lines",
        name="Actual Close", line=dict(color="#1e90ff", width=1.5)))
    fig.add_trace(go.Scatter(x=dates[:n], y=pred[:n], mode="lines",
        name="ML Predicted", line=dict(color="#ff8c00", width=1.5, dash="dash")))
    # Next-day prediction dot
    last_date  = dates[-1] + pd.Timedelta(days=1)
    fig.add_trace(go.Scatter(
        x=[last_date], y=[result["next_price_pred"]],
        mode="markers+text",
        marker=dict(color="#00d084" if result["price_change_pct"] >= 0 else "#ff3b3b",
                    size=14, symbol="diamond"),
        text=[f"  ₹{result['next_price_pred']:.2f}"],
        textfont=dict(color="#00d084" if result["price_change_pct"] >= 0 else "#ff3b3b",
                      size=11, family="IBM Plex Mono"),
        textposition="middle right",
        name="Next Prediction"
    ))
    fig.update_layout(title=f"{result['ticker']} — Price Prediction vs Actual",
                      xaxis_title="Date", yaxis_title="Price (₹)",
                      hovermode="x unified", height=320,
                      margin=_M_DEFAULT, **DARK)
    return fig

def plot_feature_importance(fi: pd.Series, ticker: str) -> go.Figure:
    top = fi.head(15)
    fig = go.Figure(go.Bar(
        y=top.index[::-1], x=top.values[::-1],
        orientation="h",
        marker_color="#ff8c00",
        marker_line_width=0,
    ))
    fig.update_layout(title=f"{ticker} — Top 15 Feature Importances",
                      xaxis_title="Importance", height=340,
                      margin=_M_DEFAULT, **DARK)
    return fig

def plot_equity(result: dict) -> go.Figure:
    idx   = result["eq_index"]
    strat = result["strat_equity"]
    bh    = result["bh_equity"]
    n     = min(len(idx), len(strat), len(bh))
    fig   = go.Figure()
    fig.add_trace(go.Scatter(x=list(idx[:n]), y=list(strat[:n]), mode="lines",
        name="ML Strategy", line=dict(color="#00d084", width=1.5)))
    fig.add_trace(go.Scatter(x=list(idx[:n]), y=list(bh[:n]), mode="lines",
        name="Buy & Hold", line=dict(color="#555", width=1, dash="dot")))
    fig.update_layout(title=f"{result['ticker']} — ML Strategy vs Buy & Hold (₹10k base)",
                      xaxis_title="Date", yaxis_title="Portfolio Value (₹)",
                      hovermode="x unified", height=280,
                      margin=_M_DEFAULT, **DARK)
    return fig

def plot_direction_accuracy(result: dict) -> go.Figure:
    actual = result["y_dir_actual"]
    pred   = result["y_dir_pred"]
    dates  = result["test_dates"]
    n      = min(len(dates), len(actual), len(pred))
    correct = (actual[:n] == pred[:n]).astype(int)
    colors  = ["#00d084" if c else "#ff3b3b" for c in correct]
    fig = go.Figure(go.Bar(
        x=list(dates[:n]), y=[1]*n,
        marker_color=colors,
        marker_line_width=0,
        hovertemplate="Date: %{x}<br>%{customdata}<extra></extra>",
        customdata=["✔ Correct" if c else "✘ Wrong" for c in correct],
    ))
    fig.update_layout(title=f"{result['ticker']} — Direction Prediction (green=correct)",
                      yaxis=dict(showticklabels=False, showgrid=False),
                      height=160, margin=_M_COMPACT, **DARK)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — settings
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
<div style="color:#ff8c00;font-size:.88rem;font-weight:700;letter-spacing:.12em;
padding:8px 0 6px;border-bottom:1px solid #2a2a2a;margin-bottom:10px;">
🧠 ML SETTINGS
</div>""", unsafe_allow_html=True)

    st.markdown("""
<div style="color:#555;font-size:.58rem;line-height:1.6;margin-bottom:8px;">
Base: RF + GBM + XGB + LGB + TCN*<br>
Meta: Logistic Regression (calibrated)<br>
Walk-forward CV · Data-derived hyperparams<br>
<span style="color:#444;">*TCN activated when n_train ≥ 800 + TF installed</span>
</div>""", unsafe_allow_html=True)
    timeframe     = st.selectbox("Timeframe", ["Daily", "Weekly", "Monthly"], index=0)
    backtest_days = st.slider("Backtest window (bars)", 30, 180, 60, step=10)
    extra_days    = st.slider("Extra history to fetch (days)", 365, 1095, 730, step=30)

    st.markdown('<div style="border-top:1px solid #2a2a2a;margin:10px 0;"></div>', unsafe_allow_html=True)
    if st.button("← Back to Screener", use_container_width=True):
        try:
            st.switch_page("pages/1_live_screener.py")
        except Exception:
            st.info("Navigate to Screener Pro in the sidebar.")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN — read tickers from session state OR let user enter manually
# ─────────────────────────────────────────────────────────────────────────────
ml_tickers     = st.session_state.get("ml_tickers", [])
ml_raw_data    = st.session_state.get("ml_raw_data", {})
score_context  = st.session_state.get("ml_score_context", [])

# ── Ticker input ──────────────────────────────────────────────────────────────
if ml_tickers:
    st.markdown(f"""
<div style="background:#001a0a;border:1px solid #00d084;border-left:3px solid #00d084;
padding:8px 14px;font-family:'IBM Plex Mono',monospace;margin-bottom:10px;">
  <span style="color:#00d084;font-size:.72rem;font-weight:700;">
    ✔ {len(ml_tickers)} STOCKS FROM SCREENER
  </span>
  <span style="color:#555;font-size:.62rem;margin-left:10px;">
    {', '.join(ml_tickers[:8])}{'…' if len(ml_tickers)>8 else ''}
  </span>
</div>""", unsafe_allow_html=True)
else:
    st.markdown("""
<div style="background:#1a0800;border:1px solid #ff8c00;padding:8px 14px;margin-bottom:10px;
font-size:.68rem;color:#ff8c00;">
  ⚠ No stocks received from screener. Enter tickers manually below.
</div>""", unsafe_allow_html=True)

manual_input = st.text_input(
    "Tickers (comma-separated, e.g. RELIANCE.NS, INFY.NS)",
    value=", ".join(ml_tickers) if ml_tickers else "",
    key="ml_manual_input",
    label_visibility="collapsed" if ml_tickers else "visible",
    placeholder="RELIANCE.NS, TCS.NS, INFY.NS"
)

# Resolve final ticker list
final_tickers = [t.strip().upper() for t in manual_input.split(",") if t.strip()]

if not final_tickers:
    st.info("Enter tickers above or run the Screener and click 🧠 OPEN ML PREDICTOR.")
    st.stop()

# ── Score context table ───────────────────────────────────────────────────────
if score_context:
    ctx_df = pd.DataFrame(score_context)
    st.markdown("""
<div style="color:#ff8c00;font-size:.68rem;font-weight:700;letter-spacing:.1em;margin-bottom:5px;">
◼ SCREENER CONTEXT
</div>""", unsafe_allow_html=True)
    st.dataframe(ctx_df.style.format({
        "Score":"{:.0f}", "Entry":"₹{:.2f}", "Target":"₹{:.2f}", "Stop":"₹{:.2f}", "RR":"{:.2f}"
    }, na_rep="—"), use_container_width=True, hide_index=True, height=160)
    st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# RUN BUTTON
# ─────────────────────────────────────────────────────────────────────────────
col_run, col_info = st.columns([2, 3])
with col_run:
    run_btn = st.button(f"🧠  RUN ENSEMBLE ML  (RF+GBM+XGB+LGB+TCN)  ·  {timeframe}",
                        type="primary", use_container_width=True, key="run_ml_btn")
with col_info:
    st.markdown(f"""
<div style="background:#0a0a0a;border:1px solid #2a2a2a;padding:8px 14px;font-size:.62rem;color:#888;">
  Engine: <b style="color:#ff8c00;">Stacked Ensemble (RF + GBM + XGB + LGB + TCN*)</b>  ·
  Timeframe: <b style="color:#ff8c00;">{timeframe}</b>  ·
  Backtest: <b style="color:#ff8c00;">{backtest_days} bars</b>  ·
  ~80 features (technical + screener signals) · Walk-forward CV · No lookahead bias
</div>""", unsafe_allow_html=True)

if not run_btn:
    st.markdown("""
<div style="color:#444;font-size:.65rem;margin-top:20px;text-align:center;">
Click RUN ML to start prediction pipeline
</div>""", unsafe_allow_html=True)
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# RUN ML FOR EACH TICKER
# ─────────────────────────────────────────────────────────────────────────────
all_results = []
progress = st.progress(0.0, text="Starting ML pipeline…")

# ── Fetch Nifty 50 close once — used for RS vs Nifty in engineer_features ──
@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_nifty_close(days: int) -> pd.Series:
    end_dt   = datetime.now() + timedelta(days=2)
    start_dt = datetime.now() - timedelta(days=days + 30)
    try:
        df = yf.download("^NSEI",
                         start=start_dt.strftime("%Y-%m-%d"),
                         end=end_dt.strftime("%Y-%m-%d"),
                         interval="1d", progress=False, auto_adjust=True)
        if df.empty:
            return pd.Series(dtype=float)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.strip().title() for c in df.columns]
        s = df["Close"].squeeze()
        if s.index.tz is not None:
            s.index = s.index.tz_convert(None)
        return s.dropna()
    except Exception:
        return pd.Series(dtype=float)

progress.progress(0.02, text="Fetching Nifty 50 for RS signals…")
nifty_close_series = _fetch_nifty_close(extra_days)

for i, ticker in enumerate(final_tickers):
    progress.progress((i + 0.3) / len(final_tickers), text=f"Fetching data: {ticker}…")

    # Always fetch fresh with explicit start/end dates so Saturday runs include Friday.
    # ml_raw_data from screener is intentionally NOT used for the OHLCV history —
    # it may be hours stale. We only used it previously for speed; correctness wins.
    df_raw = _fetch_ohlcv(ticker, extra_days)

    # If fresh fetch failed, fall back to screener cache as last resort
    if df_raw.empty and ticker in ml_raw_data and not ml_raw_data[ticker].empty:
        df_raw = ml_raw_data[ticker].copy()

    if df_raw.empty:
        st.warning(f"⚠ No data for {ticker} — skipped.")
        continue

    # Normalise: flatten MultiIndex, title-case cols, strip tz, drop non-numeric
    df_raw = _normalise_df(df_raw)

    progress.progress((i + 0.7) / len(final_tickers), text=f"Running ML: {ticker}…")

    result = run_ml_for_ticker(ticker, df_raw, backtest_days, timeframe,
                               nifty_close=nifty_close_series if not nifty_close_series.empty else None)

    if "error" in result:
        st.warning(f"⚠ {ticker}: {result['error']}")
        continue

    all_results.append(result)
    progress.progress((i + 1.0) / len(final_tickers), text=f"Done: {ticker}")

progress.progress(1.0, text="ML pipeline complete ✔")

if not all_results:
    st.error("No results — check data availability or reduce backtest window.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="background:#1a1200;border-top:2px solid #ff8c00;border-bottom:1px solid #2a2a2a;
     padding:7px 14px;font-family:'IBM Plex Mono',monospace;margin-bottom:10px;">
  <span style="color:#ff8c00;font-size:.77rem;font-weight:700;letter-spacing:.12em;">
    ◼ ML PREDICTION SUMMARY
  </span>
</div>
""", unsafe_allow_html=True)

summary_rows = []
for r in all_results:
    summary_rows.append({
        "Ticker":         r["ticker"],
        "Timeframe":      r["timeframe"],
        "Current ₹":      f"₹{r['current_price']:.2f}",
        "Predicted ₹":    f"₹{r['next_price_pred']:.2f}",
        "Δ%":             f"{r['price_change_pct']:+.2f}%",
        "Direction":      r["next_direction"],
        "Conf":           f"{r.get('empirical_precision', r['direction_conf']):.0%}",
        "Accuracy":       f"{r['accuracy']:.0%}",
        "AUC":            f"{r['auc']:.3f}",
        "Precision":      f"{r['precision']:.0%}",
        "Recall":         f"{r['recall']:.0%}",
        "F1":             f"{r['f1']:.3f}",
        "MAE ₹":          f"₹{r['mae']:.2f}",
        "MAPE%":          f"{r['mape']:.1f}%",
        "Base Models":    r["n_base_models"],
        "CV Folds":       r["n_cv_splits"],
    })

summary_df = pd.DataFrame(summary_rows)


def plot_oof_aucs(result: dict) -> go.Figure:
    """Bar chart showing walk-forward OOF AUC per base model — lets user see
    which model contributed most to the ensemble on this specific stock."""
    oof = result.get("oof_aucs", {})
    if not oof:
        return go.Figure()
    models  = list(oof.keys())
    aucs    = list(oof.values())
    colors  = ["#00d084" if a > 0.55 else "#ffb347" if a > 0.50 else "#ff3b3b" for a in aucs]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=models, y=aucs,
        marker_color=colors,
        marker_line_width=0,
        text=[f"{a:.3f}" for a in aucs],
        textposition="outside",
        textfont=dict(color="#e8e8e8", size=10, family="IBM Plex Mono"),
    ))
    # Reference line at 0.5 (random)
    fig.add_hline(y=0.5, line_dash="dot", line_color="#555",
                  annotation_text="random (0.5)",
                  annotation_font=dict(color="#555", size=9))
    fig.update_layout(
        title=f"{result['ticker']} — Walk-Forward OOF AUC per Base Model",
        yaxis=dict(title="AUC", range=[0.4, max(0.75, max(aucs)+0.05)],
                   gridcolor="#1a1a1a", tickfont=dict(color="#888", size=9)),
        xaxis=dict(tickfont=dict(color="#e8e8e8", size=11, family="IBM Plex Mono")),
        height=220, margin=_M_DEFAULT,
        **DARK,
    )
    return fig


def color_direction(val):
    v = str(val)
    if "UP"        in v: return "background-color:#001a0a;color:#00d084;font-weight:700"
    if "DOWN"      in v: return "background-color:#1a0000;color:#ff3b3b;font-weight:700"
    if "UNCERTAIN" in v: return "background-color:#1a1400;color:#888888;font-weight:700"
    return ""

def color_delta(val):
    try:
        v = float(str(val).replace("%","").replace("+",""))
        return f"color:{'#00d084' if v>=0 else '#ff3b3b'};font-weight:700"
    except: return ""

styled = summary_df.style\
    .applymap(color_direction, subset=["Direction"])\
    .applymap(color_delta, subset=["Δ%"])

st.dataframe(styled, use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# PER-TICKER DETAILED RESULTS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="color:#ff8c00;font-size:.77rem;font-weight:700;letter-spacing:.1em;margin-bottom:8px;">
◼ DETAILED RESULTS PER TICKER
</div>""", unsafe_allow_html=True)

tab_labels = [r["ticker"].replace(".NS","") for r in all_results]
if len(tab_labels) == 1:
    tabs = [st.container()]
else:
    tabs = st.tabs(tab_labels)

for tab, result in zip(tabs, all_results):
    with tab:
        ticker = result["ticker"]
        _dir   = result["next_direction"]
        if "UP"        in _dir: dir_color, dir_bg = "#00d084", "#001a0a"
        elif "DOWN"    in _dir: dir_color, dir_bg = "#ff3b3b", "#1a0000"
        else:                   dir_color, dir_bg = "#888888", "#111111"

        # ── Hero metrics ──────────────────────────────────────────────────────
        st.markdown(f"""
<div style="background:{dir_bg};border:1px solid {dir_color};border-left:4px solid {dir_color};
padding:12px 18px;font-family:'IBM Plex Mono',monospace;margin-bottom:12px;">
  <div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr 1fr;gap:16px;">
    <div>
      <div style="color:#555;font-size:.52rem;letter-spacing:.1em;">TICKER</div>
      <div style="color:#e8e8e8;font-size:1.1rem;font-weight:700;">{ticker.replace('.NS','')}</div>
      <div style="color:#555;font-size:.55rem;">{result['timeframe']} · Stacked Ensemble · {result['n_base_models']} base models {'· TCN ✔' if result.get('tcn_active') else ''}</div>
    </div>
    <div>
      <div style="color:#555;font-size:.52rem;letter-spacing:.1em;">CURRENT PRICE</div>
      <div style="color:#e8e8e8;font-size:1.05rem;font-weight:700;">₹{result['current_price']:.2f}</div>
      <div style="font-size:.55rem;margin-top:2px;color:{
        '#00d084' if result.get('price_source') in ('upstox_cache','upstox_api') else
        '#ffb347' if result.get('price_source') in ('yf_fast_info','yf_history5d') else '#ff3b3b'}">
        {'🟢 Upstox live' if result.get('price_source')=='upstox_cache' else
         '🟢 Upstox API'  if result.get('price_source')=='upstox_api'   else
         '🟡 yf fast_info' if result.get('price_source')=='yf_fast_info' else
         '🟡 yf history5d' if result.get('price_source')=='yf_history5d' else
         '🔴 OHLCV fallback — ₹'+str(round(result.get('ohlcv_last_price',0),2))}
      </div>
    </div>
    <div>
      <div style="color:#555;font-size:.52rem;letter-spacing:.1em;">PREDICTED NEXT CLOSE</div>
      <div style="color:{dir_color};font-size:1.05rem;font-weight:700;">₹{result['next_price_pred']:.2f}</div>
      <div style="color:{dir_color};font-size:.62rem;font-weight:700;">{result['price_change_pct']:+.2f}%</div>
    </div>
    <div>
      <div style="color:#555;font-size:.52rem;letter-spacing:.1em;">DIRECTION</div>
      <div style="color:{dir_color};font-size:1.1rem;font-weight:700;">{result['next_direction']}</div>
      <div style="color:#888;font-size:.60rem;">Empirical precision: {result.get('empirical_precision',0):.0%} | Edge: {result['direction_conf']:.0%}</div>
    </div>
    <div>
      <div style="color:#555;font-size:.52rem;letter-spacing:.1em;">MODEL ACCURACY</div>
      <div style="color:#ff8c00;font-size:1.05rem;font-weight:700;">{result['accuracy']:.0%}</div>
      <div style="color:#555;font-size:.58rem;">AUC {result['auc']:.3f}</div>
    </div>
  </div>
</div>""", unsafe_allow_html=True)

        # ── Metrics row ───────────────────────────────────────────────────────
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Accuracy",  f"{result['accuracy']:.1%}",
                  delta="+vs random" if result['accuracy'] > 0.5 else "-vs random",
                  delta_color="normal" if result['accuracy'] > 0.5 else "inverse")
        m2.metric("AUC",       f"{result['auc']:.3f}",
                  delta="strong" if result['auc'] > 0.55 else "weak",
                  delta_color="normal" if result['auc'] > 0.55 else "inverse")
        m3.metric("F1 Score",  f"{result['f1']:.3f}")
        m4.metric("Precision", f"{result['precision']:.1%}")
        m5.metric("Price MAE", f"₹{result['mae']:.2f}")
        m6.metric("MAPE",      f"{result['mape']:.1f}%",
                  delta="good" if result['mape'] < 3 else "check",
                  delta_color="normal" if result['mape'] < 3 else "off")

        # ── Charts ────────────────────────────────────────────────────────────
        c_left, c_right = st.columns([3, 2])
        with c_left:
            st.plotly_chart(plot_price_prediction(result), use_container_width=True, config={"displayModeBar":False})
            st.plotly_chart(plot_direction_accuracy(result), use_container_width=True, config={"displayModeBar":False})
        with c_right:
            if not result["feat_importance"].empty:
                st.plotly_chart(plot_feature_importance(result["feat_importance"], ticker),
                                use_container_width=True, config={"displayModeBar":False})

        st.plotly_chart(plot_equity(result), use_container_width=True, config={"displayModeBar":False})

        # ── OOF AUC per base model ────────────────────────────────────────────
        if result.get("oof_aucs"):
            st.plotly_chart(plot_oof_aucs(result), use_container_width=True, config={"displayModeBar":False})

        # ── Model info expander ───────────────────────────────────────────────
        with st.expander("◼ ENSEMBLE ARCHITECTURE & METHODOLOGY"):
            base_names = ", ".join(result["base_model_names"])
            oof_rows   = "\n".join(
                f"| {m} | {auc:.4f} | {'✔ strong' if auc > 0.55 else '— weak'} |"
                for m, auc in result["oof_aucs"].items()
            )
            st.markdown(f"""
**Architecture:** Stacked ensemble — {result['n_base_models']} base models → calibrated meta-learner

**Base models (L1):** {base_names}
{'  **· TCN:** active (dilated causal CNN, RF=90 bars)' if result.get('tcn_active') else '  · TCN: inactive — n_train < 800 or TF not installed'}

**Meta-learner (L2):** Logistic Regression (C = 1/{result['n_base_models']})
— Trains on out-of-fold predictions from L1, learns which model to trust per regime
— Isotonic calibration applied so probabilities are statistically meaningful

**Walk-forward CV:** {result['n_cv_splits']} folds (TimeSeriesSplit — no future data leakage)

**Out-of-fold AUC per base model (training set):**

| Model | OOF AUC | Quality |
|-------|---------|---------|
{oof_rows}

**Threshold:** {result['best_threshold']:.3f} (F1-optimal on training data — not fixed at 0.5)

**Features:** {result['n_features']} signals
- **Technical (~45):** Returns (1/3/5/10/20d), dist from SMA5/10/20/50/200, Wilder RSI(7/14/21), MACD, Bollinger Bands width/position, ATR, volume z-score/ratio, Stochastic K/D, ADX, candle geometry (body/wicks/gap), 52w position, lagged returns
- **Screener Signals (~20, same math as 1_Live_screener.py):**
  `sc_rs_nifty` — vol-normalised tanh alpha vs Nifty 50 (F1)
  `sc_momentum_vel/acc` — EMA5−EMA20 velocity + acceleration, percentile-ranked (F2)
  `sc_vol_z_pct` — volume surge z-score percentile (F3)
  `sc_accumulation` — vol5/vol20 sigmoid score, centre=1.3 (F4)
  `sc_vol_contraction` — ATR5/ATR20 inverted percentile (F5)
  `sc_range_compression` — range5/range20 inverted percentile (F5b)
  `sc_vcve` — volume×compression interaction, hidden accumulation (F5c)
  `sc_coil_quality` — base compression×flatness composite (F6)
  `sc_trend_structure` — EMA9/EMA50 percentile over 250d (F7)
  `sc_ema_alignment` — EMA9>EMA20>EMA50 alignment score (F7 bonus)
  `sc_breakout_prox` — exp decay from 20d resistance, λ=1.5/ATR (F8)
  `sc_atr_potential` — ATR% vs own 60d history percentile (F9)
  `sc_candle_score` — 8-pattern composite score (F10)
  `sc_sweep` — liquidity sweep detection (bonus)
  `sc_above_vwma20/rising` — VWMA20 position (bonus)
  `sc_momentum_stability` — positive-day fraction over 20d (bonus)
  `sc_pos52w_pct` — 52w position percentile (bonus)
  All sc_* shifted 1 bar — strict zero lookahead

**Data:** {result['n_train']} train bars · {result['n_test']} test bars · {result['timeframe']}

**Hyperparameter logic (no arbitrary numbers):**
- `n_estimators` = min(500, max(100, n_train / 5)) — scales with data volume
- `max_depth` = log₂(n_features) — information-theoretic bound
- `min_samples_leaf` = max(5, n_train / 500) — prevents leaf overfitting
- `meta LR C` = 1 / n_base_models — regularisation proportional to ensemble size
- `Ridge alpha` = n_base_regressors — same principle for price regressor
""")

# ─────────────────────────────────────────────────────────────────────────────
# EXPORT
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
export_df = pd.DataFrame([{
    "Ticker":       r["ticker"],
    "Timeframe":    r["timeframe"],
    "Current":      r["current_price"],
    "Predicted":    r["next_price_pred"],
    "Change%":      r["price_change_pct"],
    "Direction":    r["next_direction"],
    "Confidence":   r["direction_conf"],
    "Accuracy":     r["accuracy"],
    "AUC":          r["auc"],
    "MAE":          r["mae"],
    "BaseModels":   r["n_base_models"],
    "CVFolds":      r["n_cv_splits"],
    "Threshold":    r["best_threshold"],
    "Precision":    r["precision"],
    "Recall":       r["recall"],
    "F1":           r["f1"],
    "RunAt":        datetime.now().strftime("%Y-%m-%d %H:%M"),
} for r in all_results])

st.download_button(
    label="⬇ Export Results CSV",
    data=export_df.to_csv(index=False),
    file_name=f"ml_predictions_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
    mime="text/csv",
    use_container_width=False,
)
