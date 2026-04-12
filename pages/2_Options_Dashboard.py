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
from datetime import datetime, timedelta, date
import plotly.graph_objects as go
import yfinance as yf
import os
try:
    import pytz
except ImportError:
    pytz = None  # graceful fallback — market-hours detection will be skipped

st.set_page_config(layout="wide", page_title="MONARCH — Options Intel")

# ============================================================
# CENTRALIZED CONFIGURATION — change once, applies everywhere
# ============================================================
CFG = {
    "rfr_default":    6.5,    # Risk-free rate % (India repo rate)
    "hv_window":      20,     # Historical volatility look-back (trading days)
    "hv_window_fast": 10,     # Fast HV for comparison
    "iv_hist_max":    252,    # Max IV history (1 trading year)
    "chain_strikes":  8,      # Strikes shown each side in chain tab
    "oi_strikes":     10,     # Strikes each side in OI tab
    "pain_strikes":   15,     # Strikes each side for max pain calc
    # ── IV/HV classification: percentile-based, NOT fixed ratios ──────────────────
    # These are no longer used for signal generation (replaced by iv_hv_pct_sell/buy).
    # Retained only for legacy display labels in the chain tab edge classifier.
    "iv_rich_ratio":   1.20,  # kept for chain-tab IV edge label compatibility
    "iv_cheap_ratio":  0.85,  # kept for chain-tab IV edge label compatibility
    # ── Adaptive thresholds (percentile-based, replacing magic ratios) ──────────
    # Sell premium when IV/HV ratio is in the top N-th percentile of history
    "iv_hv_pct_sell":  75,    # IV/HV percentile above which vol is "rich" → sell
    "iv_hv_pct_buy":   30,    # IV/HV percentile below which vol is "cheap" → buy
    # ADX trend threshold (percentile of rolling ADX history, not fixed 25)
    "adx_trend_pct":   60,    # ADX above this percentile = confirmed trend
    # PCR percentile thresholds (adaptive to each chain's own distribution)
    "pcr_bull_pct":    65,    # PCR above this percentile = bullish (more puts written)
    "pcr_bear_pct":    35,    # PCR below this percentile = bearish (more calls written)
    # Liquidity: all thresholds now also computed as percentiles within chain
    "liq_spread_pct_sell": 70, # spread above 70th pct of chain = illiquid
    "liq_oi_pct_min":      30, # OI below 30th pct of chain = thin
    "liq_vol_pct_min":     30, # volume below 30th pct of chain = thin
    # Safety ratio for short strikes: distance / expected_move
    "safety_ratio_safe":    1.5,  # distance > 1.5x expected move = safe
    "safety_ratio_moderate": 1.0, # distance 1.0–1.5x = moderate
    # ── Signal model weights — leading indicators first (optimised for 1–5 day prediction) ──────
    # Markets move due to POSITIONING CHANGES, not lagging price indicators.
    # Flow and positioning are leading; trend/momentum are confirming (lagging).
    # factor_weights: COLD-START PRIOR ONLY.
    # These are immediately superseded by _rank_based_factor_weights() once
    # 20+ observations accumulate. Uniform priors avoid baking in any directional
    # bias before the model has seen real performance data.
    "factor_weights": {
        "flow":         0.20,
        "positioning":  0.20,
        "vol_regime":   0.20,
        "rel_strength": 0.20,
        "trend":        0.20,
    },
    "hv_fallback":     0.15,  # HV fallback when no historical data (15% annualised — NSE index baseline)
    "chain_cache_ttl": 30,    # Seconds to cache live option chain
    "expiry_cache_ttl": 300,  # Seconds to cache expiry list
    "master_cache_ttl": 3600, # Seconds to cache instrument master
    "iv_hist_file":   ".monarch_iv_history.json",
    "signal_log_file":".monarch_signal_log.json",  # forward signal log (replaces backtest)
    "signal_log_max": 200,   # keep last 200 signal entries (~2-3 months of daily use)
    # ── Flow conviction threshold (derived from distribution, not arbitrary) ──────────
    # flow_magnitude is in [0,1]. A "high conviction" flow event is one where the
    # composite magnitude exceeds the 70th percentile of recent magnitudes.
    # We use 0.35 as the session cold-start seed — it is replaced by the rolling
    # percentile once 5+ flow observations accumulate in session state.
    "flow_conviction_seed": 0.35,
    # ── Kelly cap — EV/MaxRisk Kelly capped at a fraction of account ─────────────────
    # 0.25 = 25% max of capital in any single position (standard half-Kelly cap)
    "kelly_cap_pct":  0.25,
    # ── Fractional Kelly multiplier ──────────────────────────────────────────────────
    # Apply fractional Kelly to reduce variance. 0.5 = half-Kelly (industry standard).
    "kelly_fraction": 0.50,
    "theta_days":     252,    # Theta convention: 252 trading days
    "ann_days":       252,    # Annualisation base
    # ── Liquidity filter — legacy absolute floors (fallback when chain sample < 5) ─
    "liq_min_oi":        1000,   # absolute floor contracts (only when no percentile data)
    "liq_min_vol":        100,   # absolute floor contracts (only when no percentile data)
    "liq_max_spread_pct":   5.0, # absolute cap % (only when no percentile data)
    # ── Centralised lot sizes (SEBI-mandated NSE F&O lot sizes, updated Jan 2025) ──
    "lot_sizes": {
        "NIFTY": 75, "BANKNIFTY": 15, "FINNIFTY": 40, "MIDCPNIFTY": 75,
        "SENSEX": 10,
        "RELIANCE": 250, "HDFCBANK": 550, "ICICIBANK": 700, "INFY": 400,
        "TCS": 150, "LT": 150, "SBIN": 1500, "AXISBANK": 625,
        "KOTAKBANK": 400, "BHARTIARTL": 500, "ITC": 3200,
        "BAJFINANCE": 125, "WIPRO": 1500, "HCLTECH": 350,
        "TATAMOTORS": 1425, "MARUTI": 100,
        "SUNPHARMA": 350, "TITAN": 175, "ADANIENT": 400, "ONGC": 1925,
        "NTPC": 2250, "JSWSTEEL": 600, "TATASTEEL": 5500, "HINDALCO": 1075,
        "DRREDDY": 125, "CIPLA": 650, "DIVISLAB": 200,
        "BAJAJ-AUTO": 75, "HEROMOTOCO": 150, "EICHERMOT": 175,
        "M&M": 350, "TECHM": 600, "INDUSINDBK": 500,
        "POWERGRID": 2400, "COALINDIA": 1400, "VEDL": 1550, "SAIL": 6750,
    },
    "lot_size_fallback": 500,
    "pop_simulations": 10000,
    # ── NSE/SEBI Transaction Cost Schedule (all rates as decimals of premium) ──
    # Source: NSE circular + SEBI schedule effective Jan 2025.
    # These are FIXED fee schedules — no data needed, just constants.
    # Applied per leg at entry AND exit (round-trip = 2×).
    # EV is already a round-trip calculation (entry cost embedded in premium_eff,
    # exit assumed at intrinsic), so we apply costs once at entry only.
    "tx_cost": {
        # STT: 0.0625% of premium on SELL legs only (option seller pays STT on premium)
        # Buyer pays STT at exercise only (usually negligible for OTM options)
        "stt_sell_pct":      0.000625,   # 0.0625% of premium, sell leg only
        # NSE Exchange Transaction Charges: 0.053% of premium turnover (both sides)
        "nse_charge_pct":    0.00053,    # 0.053% of premium
        # SEBI Turnover Fee: ₹10 per crore of turnover = 0.0001% of premium
        "sebi_fee_pct":      0.000001,   # 0.0001% of premium
        # Stamp duty: 0.003% of premium on BUY legs only (buyer pays)
        "stamp_duty_pct":    0.00003,    # 0.003% of premium, buy leg only
        # GST: 18% on (brokerage + exchange charges + SEBI fee)
        # Brokerage assumed flat ₹20/order — we model it as 0 here and let
        # the user's actual broker rate appear in their P&L, not the model EV.
        # We include GST only on the exchange charges and SEBI fee (the model-known costs).
        "gst_rate":          0.18,       # 18% GST on exchange + SEBI charges
    },
    # ── Intraday candle windows ───────────────────────────────────────────────
    # All window sizes as trading parameters, not magic numbers.
    # 5-minute candles: 6 candles = 30 min, 3 candles = 15 min, 18 candles = 90 min
    "intra_interval":         "5minute",   # Upstox candle interval
    "intra_opening_candles":  6,           # 30-min opening window (6 × 5-min candles)
    "intra_recent_candles":   6,           # 30-min recent window for vol acceleration
    "intra_structure_candles":3,           # candles for price structure early/late comparison
    "intra_lunch_candle_start": 14,        # candle index ~12:30 IST (14 × 5-min from open)
    "intra_lunch_candle_morn":  5,         # candle index ~end of first hour (~09:55)
    "intra_min_candles_vol":  6,           # min candles needed for vol acceleration signal
    "intra_min_candles_struct":6,          # min candles needed for price structure signal
    "intra_min_candles_lunch": 18,         # min candles needed for lunch reversal (~90 min)
    "intra_min_candles":       2,          # absolute minimum to compute any signal
}

_REGIME_TRENDING_UP   = "TRENDING_UP"
_REGIME_TRENDING_DOWN = "TRENDING_DOWN"
_REGIME_VOL_EXPANSION = "VOL_EXPANSION"
_REGIME_VOL_COMPRESS  = "VOL_COMPRESSION"
_REGIME_RANGE_BOUND   = "RANGE_BOUND"
_REGIME_TRANSITION    = "TRANSITION"

_REGIME_SIGNAL_WEIGHTS = {
    _REGIME_TRENDING_UP:   {"flow": 0.30, "positioning": 0.15, "vol_regime": 0.10,
                            "rel_strength": 0.30, "trend": 0.15},
    _REGIME_TRENDING_DOWN: {"flow": 0.30, "positioning": 0.15, "vol_regime": 0.10,
                            "rel_strength": 0.30, "trend": 0.15},
    _REGIME_VOL_EXPANSION: {"flow": 0.45, "positioning": 0.20, "vol_regime": 0.30,
                            "rel_strength": 0.05, "trend": 0.00},
    _REGIME_VOL_COMPRESS:  {"flow": 0.25, "positioning": 0.45, "vol_regime": 0.20,
                            "rel_strength": 0.05, "trend": 0.05},
    _REGIME_RANGE_BOUND:   {"flow": 0.15, "positioning": 0.55, "vol_regime": 0.15,
                            "rel_strength": 0.10, "trend": 0.05},
    _REGIME_TRANSITION:    {"flow": 0.50, "positioning": 0.20, "vol_regime": 0.15,
                            "rel_strength": 0.10, "trend": 0.05},
}

# ============================================================
# ADAPTIVE PARAMETER ENGINE
# All mixing weights, scale factors, and thresholds are learned
# from historical signal→return correlations stored in session state.
# Hard-coded constants are replaced by data-driven calibration with
# principled priors used only as cold-start seeds.
# ============================================================

# ── Cold-start priors (used ONLY before enough data accumulates) ──────────────
# Every prior is a single interpretable number that gets overwritten by data.
_PRIOR = {
    # Sub-model internal mixing weights (sum to 1.0 within each group)
    "trend_ema_vs_adx":       0.60,   # weight of EMA structure vs ADX within trend factor
    "momentum_rsi_vs_ret5":   0.50,   # RSI vs 5-day return within momentum
    "vol_bb_vs_atr":          0.60,   # BB regime vs ATR regime within vol factor
    "positioning_pcr_vs_oi_vs_mp": [0.40, 0.35, 0.25],  # PCR : OI-skew : max-pain
    "rs_level_vs_slope":      0.70,   # RS level vs RS slope within rel-strength
    "trend_combined_trend_vs_momentum": 0.60,   # combined trend: trend_score vs momentum
    # Regime pillars
    "regime_adx_vs_gex":      0.60,   # ADX vs GEX within trend_axis
    "regime_iv_vs_hv_accel":  0.55,   # IV pct vs HV accel within vol_axis
    "regime_conf_iv":         0.30,   # confidence: weight of IV pillar
    "regime_conf_adx":        0.30,   # confidence: weight of ADX pillar
    "regime_conf_hv":         0.25,   # confidence: weight of HV accel pillar
    "regime_conf_gex":        0.15,   # confidence: weight of GEX pillar
    # Composite EV score weights
    "ev_score_vs_dir_align":  0.60,   # ev_score vs directional alignment in composite
    # Within-factor RS decomposition
    "rs_z_vs_slope":          0.70,
    # Logistic sharpness for prob_up conversion
    # GAP-5 FIX: raised from 4.0 → 7.0.
    # At sharpness=4, a typical cold-start raw_score of ±0.15 maps to prob_up≈54%
    # — firmly inside the 45-55% neutral band.  At 7.0 the same signal maps to ≈60%,
    # which is outside the band and allows bull/bear strategies to be distinguished
    # immediately.  Calibration will lower sharpness automatically if signals prove
    # weak; we want to start HOT and let data cool us, not start cold and stay cold.
    # FIX 5: PRIOR_VERSION guards against stale persisted calibration (e.g. an older
    # logistic_sharpness=4.x on disk) silently overriding this hot-start value.
    # Bump _PRIOR_VERSION whenever a prior changes enough to warrant a reset.
    "logistic_sharpness":     7.0,    # raw_score × sharpness → logistic input
    # Sigmoid sharpness for safety factor
    "safety_sigmoid_sharpness": 2.0,
    # Tanh scale for term structure bonus
    "ts_tanh_scale":          20.0,   # ts_slope × scale
    # Tanh scale for EV normalisation: ev / (max_risk × ev_tanh_scale)
    "ev_tanh_scale":          0.50,
    # Max-pain gravity dampening (mp_dist_em × gravity → clamp)
    "mp_gravity":             0.40,
    # Vol-regime dampening: expensive vol → bearish lean, coefficient
    "vol_regime_damp":        0.50,
    # Term-structure z-score scale (pp of slope → z-score)
    "ts_slope_scale":         20.0,
    # RS slope scale
    "rs_slope_scale":         50.0,
    # HV accel tanh stretch
    "hv_accel_stretch":       3.0,
    # ADX weak-trend fraction of trend level
    "adx_weak_frac":          0.75,
    "adx_vs_rsi_within_trend": 0.75,   # fraction of non-EMA portion going to ADX vs RSI
    # MC blend weight (MC direction vs factor direction)
    "mc_blend":               0.50,
    # Liquidity composite weights
    "liq_spread_w":           0.50,
    "liq_oi_w":               0.30,
    "liq_vol_w":              0.20,
    # std floor as fraction of mean (for dOI/dPCR z-score denominators)
    "std_floor_frac":         0.05,
    # ── Intraday signal weights ───────────────────────────────────────────────
    # Opening 30-min momentum is the strongest NSE intraday signal.
    # VWAP position is second. Volume acceleration and OI build are supporting.
    "intra_w_opening_momentum":    0.30,   # first 30-min directional thrust
    "intra_w_vwap_position":       0.25,   # spot vs VWAP
    "intra_w_volume_acceleration": 0.20,   # recent 30-min vol vs session average
    "intra_w_oi_build":            0.15,   # net CE vs PE OI change direction
    "intra_w_price_structure":     0.07,   # higher highs / lower lows
    "intra_w_lunch_reversal":      0.03,   # post-lunch reversal signal
    # Weight of intraday score in final directional blend
    "intra_blend_weight":          0.20,   # 20% intraday, 80% factor+MC by default
    # ── Vector priors — derived from their scalar equivalents ──────────────────
    # These are the 2-element weight vectors that _calib_vec() looks up.
    # Cold-start: uniform or matching the scalar priors above.
    # They get overwritten by _update_vec_calib() once 15+ observations accumulate.
    "ev_score_vs_dir_align_vec":  [0.60, 0.40],  # ev_score : dir_align (matches ev_score_vs_dir_align)
    "mc_blend_vec":               [0.50, 0.50],  # MC direction : factor direction (matches mc_blend)
    "rs_level_vs_slope_vec":      [0.70, 0.30],  # RS level : RS slope (matches rs_level_vs_slope)
    "trend_ema_vs_adx_vec":       [0.60, 0.40],  # EMA : ADX (matches trend_ema_vs_adx)
    "adx_vs_rsi_vec":             [0.75, 0.25],  # ADX : RSI (matches adx_vs_rsi_within_trend)
    "vol_regime_z_vs_ts":         [0.55, 0.45],  # IV/HV z : term-slope z (matches regime_iv_vs_hv_accel)
    "regime_conf_pillars":        [0.30, 0.30, 0.25, 0.15],  # IV : ADX : HV : GEX (matches conf priors)
    "intra_weights_vec":          [0.30, 0.25, 0.20, 0.15, 0.07, 0.03],  # 6 intraday signals
    "intra_blend_vec":            [0.80, 0.20],  # factor+MC : intraday (matches 1-intra_blend_weight)
}

_CALIB_STORE_KEY = "monarch_calib"   # session_state key for persisted calibration
_CALIB_FILE      = ".monarch_calib.json"   # disk path — survives restarts
_CALIB_MIN_OBS_SEED = 20             # floor: never calibrate on fewer than 20 obs
_CALIB_MIN_OBS_CAP  = 60             # ceiling: require no more than 60 obs

def _dynamic_min_obs(symbol: str = "") -> int:
    """Dynamic minimum observation count based on recent outcome stability.
    Stable (low-variance) outcomes → fewer obs needed.
    Noisy (high-variance) outcomes → more obs required.
    Uses exponential decay to weight recent volatility more heavily.
    """
    sym  = symbol or st.session_state.get("opt_symbol", "").upper()
    rkey = f"{sym}:_calib_realised_ret_hist" if sym else "_calib_realised_ret_hist"
    hist = st.session_state.get(rkey, st.session_state.get("_calib_realised_ret_hist", []))
    if len(hist) < 5:
        return _CALIB_MIN_OBS_CAP   # cold start: be conservative
    arr  = np.array(hist[-60:], dtype=float)
    # Exponentially-weighted variance of returns
    decay = np.array([0.94 ** (len(arr) - 1 - i) for i in range(len(arr))])
    decay /= decay.sum()
    ew_var = float(np.dot(decay, (arr - np.dot(decay, arr)) ** 2))
    # High variance → need more obs; low variance → fewer
    # Baseline annualised vol ~ 20% → daily var ~ (0.20/sqrt(252))^2 ≈ 1.6e-4
    baseline_var = (0.20 / math.sqrt(252)) ** 2
    ratio = ew_var / max(baseline_var, 1e-8)
    n_dyn = int(_CALIB_MIN_OBS_SEED * max(0.5, min(3.0, ratio)))
    return max(_CALIB_MIN_OBS_SEED, min(_CALIB_MIN_OBS_CAP, n_dyn))

_CALIB_MIN_OBS = _CALIB_MIN_OBS_SEED  # legacy alias; code uses _dynamic_min_obs() at runtime
_CALIB_WINDOW    = 252               # rolling window for correlation computation

# FIX 5: Prior version sentinel.  Bump this integer whenever a _PRIOR value changes
# enough that stale disk calibration should be discarded for that key.
# _get_calib() checks the stored version and evicts keys that predate the current version.
_PRIOR_VERSION = 2   # v2: logistic_sharpness raised 4.0 → 7.0
# Keys that must be re-seeded from _PRIOR when an older version is detected on disk:
_VERSION_RESET_KEYS = {"logistic_sharpness"}

# ── Signal history persistence ─────────────────────────────────────────────────
# All calibration signal histories, pending outcomes, and flow histories are
# stored in session_state BUT must also be persisted to disk so calibration
# survives app restarts. Without this the system always trains from scratch.
_HIST_FILE       = ".monarch_hist.json"    # disk path for all signal histories
_HIST_LOADED_KEY = "_monarch_hist_loaded"  # session flag — load only once per session

# Keys to persist: all _record() targets, pending outcomes, flow histories
_HIST_PERSIST_PREFIXES = (
    "_calib_",          # calibration signal histories (e.g. _calib_raw_score_hist)
    "_outcome_pending_",# pending signal snapshots awaiting resolution
    "_flow_",           # flow signal histories (pcr_hist, oi_hist, skew_hist, gex_hist)
    # FIX: all symbol-namespaced histories — use ":_calib_" suffix match instead of
    # listing each symbol explicitly. The _save_hist() function already handles the
    # ":_calib_" in k check. This tuple is used by _restore_hist for the load filter.
    "NIFTY:_calib_", "BANKNIFTY:_calib_", "FINNIFTY:_calib_", "MIDCPNIFTY:_calib_",
    # Stock symbols
    "RELIANCE:_calib_", "HDFCBANK:_calib_", "ICICIBANK:_calib_", "INFY:_calib_",
    "TCS:_calib_", "LT:_calib_", "SBIN:_calib_", "AXISBANK:_calib_",
    "KOTAKBANK:_calib_", "BHARTIARTL:_calib_", "BAJFINANCE:_calib_",
    "WIPRO:_calib_", "HCLTECH:_calib_", "TATAMOTORS:_calib_", "MARUTI:_calib_",
    "SUNPHARMA:_calib_", "TITAN:_calib_", "ADANIENT:_calib_",
)


def _load_hist() -> dict:
    """Load all signal histories from disk. Returns {} on any error."""
    try:
        if os.path.exists(_HIST_FILE):
            with open(_HIST_FILE, "r") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
    except Exception:
        pass
    return {}


def _save_hist():
    """Persist all signal histories from session_state to disk.
    Called at the end of every LOAD cycle so data survives restarts.
    Saves keys matching calibration/flow/outcome patterns to avoid bloat.
    """
    try:
        out = {}
        for k, v in st.session_state.items():
            if not isinstance(k, str):
                continue
            # Match any calibration history, flow history, pending outcome,
            # OI snapshot, or intraday calibration key
            if (k.startswith("_calib_") or
                k.startswith("_flow_") or
                k.startswith("_outcome_pending_") or
                k.startswith("_oi_snap_") or
                k.startswith("_calib_intra_") or
                (":_calib_" in k) or
                k == "opt_factor_hist" or         # FIX: factor weight correlation histories
                k == "_flow_skew_oi_hist" or      # FIX: OI skew history for z-score baseline
                k == "_fhist_last_load_id" or     # FIX: factor hist load guard
                k == "_last_recorded_load_id" or  # FIX: prob_up record guard
                k == "_last_outcome_load_id" or   # FIX: outcome record guard
                k == "_flow_fii_hist"):            # FII/DII signal history
                if isinstance(v, (list, dict)):
                    out[k] = v
        with open(_HIST_FILE, "w") as f:
            json.dump(out, f)
    except Exception:
        pass


def _restore_hist():
    """Load persisted signal histories into session_state on first session call.
    Merges disk data into session_state without overwriting keys already present
    (in case the session was partially populated before this call).
    FIX: Uses suffix match ":_calib_" to cover ALL stock symbols, not just the 4 indices.
    Called once at startup via _HIST_LOADED_KEY flag.
    """
    if st.session_state.get(_HIST_LOADED_KEY):
        return
    data = _load_hist()
    for k, v in data.items():
        if k not in st.session_state:
            # Accept any key matching the persist prefixes OR any symbol-namespaced calib key
            _accept = (any(k.startswith(pfx) for pfx in _HIST_PERSIST_PREFIXES)
                       or ":_calib_" in k
                       or k in ("opt_factor_hist", "_flow_skew_oi_hist"))
            if _accept:
                st.session_state[k] = v
    st.session_state[_HIST_LOADED_KEY] = True


def _load_calib() -> dict:
    """Load persisted calibration from disk. Returns {} on any error."""
    try:
        if os.path.exists(_CALIB_FILE):
            with open(_CALIB_FILE, "r") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
    except Exception:
        pass
    return {}


def _save_calib(d: dict):
    """Persist calibration dict to disk. Silent on failure."""
    try:
        with open(_CALIB_FILE, "w") as f:
            json.dump(d, f, indent=2)
    except Exception:
        pass


def _get_calib(symbol: str = "") -> dict:
    """Return the calibration dict for the given symbol.
    Falls back to a cross-symbol 'global' store for any missing key.
    On first call per session, loads from disk.
    FIX 5: evicts keys listed in _VERSION_RESET_KEYS when the persisted
    calibration pre-dates _PRIOR_VERSION, so stale values (e.g. an old
    logistic_sharpness=4.x) never silently override updated priors.
    """
    if _CALIB_STORE_KEY not in st.session_state:
        raw = _load_calib()
        # Version check: if stored version < current, evict reset keys for all symbols
        stored_ver = raw.get("_version", 0)
        if stored_ver < _PRIOR_VERSION:
            for sym_dict in raw.values():
                if isinstance(sym_dict, dict):
                    for k in _VERSION_RESET_KEYS:
                        sym_dict.pop(k, None)
            raw["_version"] = _PRIOR_VERSION
        st.session_state[_CALIB_STORE_KEY] = raw
    store = st.session_state.get(_CALIB_STORE_KEY, {})
    sym_key = symbol.upper() if symbol else "_global"
    # Symbol-specific dict, falling back to global, falling back to empty
    sym_calib    = store.get(sym_key, {})
    global_calib = store.get("_global", {})
    # Merge: symbol-specific overrides global overrides prior
    merged = {**global_calib, **sym_calib}
    return merged


def _set_calib(key: str, value, symbol: str = ""):
    """Write a calibrated value for a symbol (or globally if symbol='')."""
    if _CALIB_STORE_KEY not in st.session_state:
        st.session_state[_CALIB_STORE_KEY] = _load_calib()
    store  = st.session_state[_CALIB_STORE_KEY]
    sym_key = symbol.upper() if symbol else "_global"
    if sym_key not in store:
        store[sym_key] = {}
    store[sym_key][key] = value
    store["_version"] = _PRIOR_VERSION   # FIX 5: keep version stamp current
    st.session_state[_CALIB_STORE_KEY] = store
    _save_calib(store)


def _calib(key: str, symbol: str = None) -> float:
    """Look up a calibrated scalar parameter for a symbol.
    If symbol is None, auto-reads opt_symbol from session state.
    Returns the data-driven value if available, otherwise the prior.
    """
    sym = symbol if symbol is not None else st.session_state.get("opt_symbol", "")
    return float(_get_calib(sym).get(key, _PRIOR.get(key, 0.5)))


def _calib_vec(key: str, symbol: str = None) -> list:
    """Look up a calibrated vector parameter for a symbol.
    If symbol is None, auto-reads opt_symbol from session state.
    Returns the data-driven value if available, otherwise the prior list.
    Returned list is always L1-normalised.
    """
    sym = symbol if symbol is not None else st.session_state.get("opt_symbol", "")
    v   = _get_calib(sym).get(key, _PRIOR.get(key, [0.5]))
    arr = [float(x) for x in v]
    s   = sum(arr)
    return [x / s for x in arr] if s > 1e-9 else [1.0 / len(arr)] * len(arr)


def _update_scalar_calib(key: str, signal_hist: list, outcome_hist: list,
                          transform=None, symbol: str = ""):
    """Replace OLS with rolling hit-rate + information ratio.

    weight = exp-decay-weighted hit_rate * |information_ratio|
    Normalized against prior via data_trust ramp.
    Never uses linear regression — avoids overfit on small samples.
    """
    n_min = _dynamic_min_obs(symbol)
    if len(signal_hist) < n_min or len(outcome_hist) < n_min:
        return
    n = min(len(signal_hist), len(outcome_hist), _CALIB_WINDOW)
    x = np.array(signal_hist[-n:], dtype=float)
    y = np.array(outcome_hist[-n:], dtype=float)
    if transform is not None:
        x = np.array([transform(v) for v in x])
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < n_min:
        return

    # Exponential decay weights (recent obs matter more)
    decay = np.array([0.97 ** (len(x) - 1 - i) for i in range(len(x))])
    decay /= decay.sum()

    # Hit rate: sign(signal) == sign(outcome)
    correct = (np.sign(x) == np.sign(y)).astype(float)
    hit_rate = float(np.dot(decay, correct))   # in [0, 1]

    # Information ratio: mean(signal * outcome) / std(signal * outcome)
    prod = x * y
    prod_std = float(np.std(prod))
    if prod_std < 1e-9:
        ir = 0.0
    else:
        ir = float(np.dot(decay, prod)) / prod_std
    ir = max(-3.0, min(3.0, ir))              # clamp

    # Scale parameter: hit_rate maps [0.5,1.0] → [0,1]; scaled by |ir|
    # This measures "how well does this signal calibrate this parameter?"
    perf_score = max(0.0, (hit_rate - 0.5) * 2.0) * max(0.0, abs(ir) / 2.0)

    prior_val  = _PRIOR.get(key, 0.5)
    data_trust = min(1.0, (len(x) - n_min) / max(1, _CALIB_WINDOW - n_min))

    # Map perf_score to same scale as prior (prior is used as anchor)
    if isinstance(prior_val, (int, float)):
        # Scale factor: perf_score in [0,1] → new value near prior
        # High perf → stay near prior * (1 + ir_direction)
        ir_sign  = 1.0 if ir >= 0 else -1.0
        w_new    = prior_val * (1.0 + 0.5 * perf_score * ir_sign)
        w_new    = max(prior_val * 0.1, min(prior_val * 3.0, w_new))
        w_blended = (1.0 - data_trust) * prior_val + data_trust * w_new
        if 0.0 <= prior_val <= 1.0:
            w_blended = max(0.05, min(0.95, w_blended))
    else:
        w_blended = prior_val   # non-scalar prior: skip

    _set_calib(key, round(w_blended, 6), symbol)


def _update_vec_calib(key: str, signal_matrix: np.ndarray, outcome_hist: list,
                       symbol: str = ""):
    """Replace ridge OLS with rank-based performance weighting.

    For each signal column:
      perf = exp-decay hit_rate * |corr_with_outcomes|
    Weights = softmax(ranks(perf))  →  prevents single-factor domination.
    Blended with uniform prior via data_trust ramp.
    """
    n_min = _dynamic_min_obs(symbol)
    if len(outcome_hist) < n_min:
        return
    n = min(signal_matrix.shape[0], len(outcome_hist), _CALIB_WINDOW)
    X = signal_matrix[-n:].astype(float)
    y = np.array(outcome_hist[-n:], dtype=float)
    mask = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
    X, y = X[mask], y[mask]
    if len(X) < n_min:
        return

    k = X.shape[1]
    decay = np.array([0.97 ** (len(X) - 1 - i) for i in range(len(X))])
    decay /= decay.sum()

    perf = np.zeros(k)
    for j in range(k):
        col = X[:, j]
        if np.std(col) < 1e-9:
            perf[j] = 0.0
            continue
        correct = (np.sign(col) == np.sign(y)).astype(float)
        hit_w   = float(np.dot(decay, correct))
        corr    = float(np.corrcoef(col, y)[0, 1]) if len(col) > 2 else 0.0
        if math.isnan(corr):
            corr = 0.0
        perf[j] = max(0.0, (hit_w - 0.5) * 2.0) * abs(corr)

    # Rank-based normalisation: rank position → weight (avoids raw perf domination)
    order  = np.argsort(perf)
    ranks  = np.empty(k); ranks[order] = np.arange(1, k + 1, dtype=float)
    w_rank = ranks / ranks.sum()

    prior_arr  = np.array(_PRIOR.get(key, [1.0 / k] * k), dtype=float)
    if len(prior_arr) != k:
        prior_arr = np.full(k, 1.0 / k)
    prior_arr  = prior_arr / prior_arr.sum()

    data_trust = min(1.0, (len(X) - n_min) / max(1, _CALIB_WINDOW - n_min))
    w_final    = (1.0 - data_trust) * prior_arr + data_trust * w_rank
    w_final    = np.clip(w_final, 0.01, None)
    w_final   /= w_final.sum()

    _set_calib(key, [round(float(v), 6) for v in w_final], symbol)

def _run_calibration_cycle(ohlcv_df, symbol: str = "", horizon: int = 4):
    """Top-level calibration dispatcher.
    Called once per LOAD after outcomes are resolved.
    Uses real forward returns from resolved signal snapshots as the primary
    training signal. Falls back to OHLCV-derived forward returns when
    insufficient resolved outcomes exist.
    Each parameter is calibrated independently using its own signal/outcome pair.
    """
    sym = symbol.upper() if symbol else ""

    # ── Primary outcome series: real resolved returns ─────────────────────────
    # These are log(spot_t+horizon / spot_t) computed from actual market prices
    # against signals recorded at time t. This is the correct closed-loop feedback.
    real_ret_hist = _get_hist("_calib_realised_ret_hist", sym)

    # ── Fallback: OHLCV-derived forward returns (no look-ahead gap enforced) ──
    # Used only when real_ret_hist has fewer than _dynamic_min_obs(sym) entries.
    ohlcv_fwd_ret = []
    if len(real_ret_hist) < _dynamic_min_obs(sym):
        if ohlcv_df is not None and not ohlcv_df.empty and len(ohlcv_df) >= horizon + _dynamic_min_obs(sym) + 5:
            c = ohlcv_df["close"].astype(float).reset_index(drop=True)
            ohlcv_fwd_ret = list(np.log(c.shift(-horizon) / c).dropna().values)

    # Choose which outcome series to use
    fwd_ret = real_ret_hist if len(real_ret_hist) >= _dynamic_min_obs(sym) else ohlcv_fwd_ret
    if len(fwd_ret) < _dynamic_min_obs(sym):
        return   # not enough data to calibrate anything

    fwd_ret = list(fwd_ret)

    # Helper: retrieve per-symbol history, fall back to global
    def H(key): return _get_hist(key, sym)

    # ── 1. Logistic sharpness ──────────────────────────────────────────────────
    _update_scalar_calib("logistic_sharpness", H("_calib_raw_score_hist"), fwd_ret,
                          transform=lambda x: 1 / (1 + math.exp(-max(-20, min(20, x)))),
                          symbol=sym)

    # ── 2. EV vs direction blend ───────────────────────────────────────────────
    ev_h   = H("_calib_ev_score_hist")
    dir_h  = H("_calib_dir_align_hist")
    n_ed   = min(len(ev_h), len(dir_h), len(fwd_ret))
    if n_ed >= _dynamic_min_obs(sym):
        X_ed = np.column_stack([ev_h[-n_ed:], dir_h[-n_ed:]])
        _update_vec_calib("ev_score_vs_dir_align_vec", X_ed, fwd_ret[-n_ed:], symbol=sym)
        vec = _calib_vec("ev_score_vs_dir_align_vec", sym)
        if len(vec) == 2:
            _set_calib("ev_score_vs_dir_align", round(vec[0], 6), sym)

    # ── 3. Positioning sub-weights ─────────────────────────────────────────────
    pcr_h = H("_calib_pcr_level_hist")
    oi_h  = H("_calib_oi_skew_hist")
    mp_h  = H("_calib_mp_z_hist")
    n_pos = min(len(pcr_h), len(oi_h), len(mp_h), len(fwd_ret))
    if n_pos >= _dynamic_min_obs(sym):
        X_pos = np.column_stack([pcr_h[-n_pos:], oi_h[-n_pos:], mp_h[-n_pos:]])
        _update_vec_calib("positioning_pcr_vs_oi_vs_mp", X_pos, fwd_ret[-n_pos:], symbol=sym)

    # ── 4. Vol-regime sub-weights ──────────────────────────────────────────────
    vrz_h = H("_calib_vol_regime_z_hist")
    tsz_h = H("_calib_term_slope_z_hist")
    n_vr  = min(len(vrz_h), len(tsz_h), len(fwd_ret))
    if n_vr >= _dynamic_min_obs(sym):
        X_vr = np.column_stack([vrz_h[-n_vr:], tsz_h[-n_vr:]])
        _update_vec_calib("vol_regime_z_vs_ts", X_vr, fwd_ret[-n_vr:], symbol=sym)
        vec2 = _calib_vec("vol_regime_z_vs_ts", sym)
        if len(vec2) == 2:
            _set_calib("vol_bb_vs_atr", round(vec2[0], 6), sym)

    # ── 5. RS level vs slope ───────────────────────────────────────────────────
    rs_lev_h = H("_calib_rs_z_hist")
    rs_slp_h = H("_calib_rs_slope_hist")
    n_rs     = min(len(rs_lev_h), len(rs_slp_h), len(fwd_ret))
    if n_rs >= _dynamic_min_obs(sym):
        X_rs = np.column_stack([rs_lev_h[-n_rs:], rs_slp_h[-n_rs:]])
        _update_vec_calib("rs_level_vs_slope_vec", X_rs, fwd_ret[-n_rs:], symbol=sym)
        vec_rs = _calib_vec("rs_level_vs_slope_vec", sym)
        if len(vec_rs) == 2:
            _set_calib("rs_level_vs_slope", round(vec_rs[0], 6), sym)

    # ── 6. MC blend weight ─────────────────────────────────────────────────────
    mc_h  = H("_calib_mc_dir_hist")
    fac_h = H("_calib_fac_dir_hist")
    n_mc  = min(len(mc_h), len(fac_h), len(fwd_ret))
    if n_mc >= _dynamic_min_obs(sym):
        X_mc = np.column_stack([mc_h[-n_mc:], fac_h[-n_mc:]])
        _update_vec_calib("mc_blend_vec", X_mc, fwd_ret[-n_mc:], symbol=sym)
        vec_mc = _calib_vec("mc_blend_vec", sym)
        if len(vec_mc) == 2:
            _set_calib("mc_blend", round(vec_mc[0], 6), sym)

    # ── 7. Safety sigmoid sharpness ────────────────────────────────────────────
    safe_h = H("_calib_safety_ratio_hist")
    # Use real return as proxy for safety quality (safe positions → better returns)
    n_sf   = min(len(safe_h), len(fwd_ret))
    if n_sf >= _dynamic_min_obs(sym):
        _update_scalar_calib("safety_sigmoid_sharpness", safe_h[-n_sf:], fwd_ret[-n_sf:],
                              transform=lambda x: max(0.0, x - 1.0), symbol=sym)

    # ── 8. Term-structure tanh scale ───────────────────────────────────────────
    ts_slp_h = H("_calib_ts_slope_raw_hist")
    n_ts     = min(len(ts_slp_h), len(fwd_ret))
    if n_ts >= _dynamic_min_obs(sym):
        _update_scalar_calib("ts_tanh_scale", ts_slp_h[-n_ts:], fwd_ret[-n_ts:], symbol=sym)

    # ── 9. HV accel stretch ────────────────────────────────────────────────────
    hva_h = H("_calib_hv_accel_raw_hist")
    n_hv  = min(len(hva_h), len(fwd_ret))
    if n_hv >= _dynamic_min_obs(sym):
        _update_scalar_calib("hv_accel_stretch", hva_h[-n_hv:], fwd_ret[-n_hv:], symbol=sym)

    # ── 10. Max-pain gravity ────────────────────────────────────────────────────
    mp_raw_h = H("_calib_mp_dist_raw_hist")
    n_mp     = min(len(mp_raw_h), len(fwd_ret))
    if n_mp >= _dynamic_min_obs(sym):
        _update_scalar_calib("mp_gravity", [-v for v in mp_raw_h[-n_mp:]], fwd_ret[-n_mp:],
                              symbol=sym)

    # ── 11. Regime pillar confidence weights ────────────────────────────────────
    iv_ph  = H("_calib_iv_pillar_hist")
    adx_ph = H("_calib_adx_pillar_hist")
    hva_ph = H("_calib_hva_pillar_hist")
    gex_ph = H("_calib_gex_pillar_hist")
    n_rp   = min(len(iv_ph), len(adx_ph), len(hva_ph), len(gex_ph), len(fwd_ret))
    if n_rp >= _dynamic_min_obs(sym):
        X_rp = np.column_stack([
            np.abs(iv_ph[-n_rp:]), np.abs(adx_ph[-n_rp:]),
            np.abs(hva_ph[-n_rp:]),np.abs(gex_ph[-n_rp:]),
        ])
        _update_vec_calib("regime_conf_pillars", X_rp,
                          list(np.abs(fwd_ret[-n_rp:])), symbol=sym)
        vrc = _calib_vec("regime_conf_pillars", sym)
        if len(vrc) == 4:
            for k, v in zip(["regime_conf_iv","regime_conf_adx",
                              "regime_conf_hv","regime_conf_gex"], vrc):
                _set_calib(k, round(v, 6), sym)

    # ── 12. Trend EMA vs ADX ────────────────────────────────────────────────────
    ema_h = H("_calib_ema_score_hist")
    adx_h = H("_calib_adx_score_hist")
    n_tr  = min(len(ema_h), len(adx_h), len(fwd_ret))
    if n_tr >= _dynamic_min_obs(sym):
        X_tr = np.column_stack([ema_h[-n_tr:], adx_h[-n_tr:]])
        _update_vec_calib("trend_ema_vs_adx_vec", X_tr, fwd_ret[-n_tr:], symbol=sym)
        vtr = _calib_vec("trend_ema_vs_adx_vec", sym)
        if len(vtr) == 2:
            _set_calib("trend_ema_vs_adx", round(vtr[0], 6), sym)

    # ── 13. ADX vs RSI within trend ────────────────────────────────────────────
    rsi_h2 = H("_calib_rsi_trend_hist")
    n_ar   = min(len(adx_h), len(rsi_h2), len(fwd_ret))
    if n_ar >= _dynamic_min_obs(sym):
        X_ar = np.column_stack([adx_h[-n_ar:], rsi_h2[-n_ar:]])
        _update_vec_calib("adx_vs_rsi_vec", X_ar, fwd_ret[-n_ar:], symbol=sym)
        var = _calib_vec("adx_vs_rsi_vec", sym)
        if len(var) == 2:
            _set_calib("adx_vs_rsi_within_trend", round(var[0], 6), sym)

    # ── 14. Intraday signal weights ─────────────────────────────────────────────
    _intra_keys = [
        "_calib_intra_opening_momentum_hist",
        "_calib_intra_vwap_position_hist",
        "_calib_intra_volume_acceleration_hist",
        "_calib_intra_oi_build_hist",
        "_calib_intra_price_structure_hist",
        "_calib_intra_lunch_reversal_hist",
    ]
    _intra_hists = [H(k) for k in _intra_keys]
    _n_intra = min(min(len(h) for h in _intra_hists), len(fwd_ret))
    if _n_intra >= _dynamic_min_obs(sym):
        X_intra = np.column_stack([h[-_n_intra:] for h in _intra_hists])
        _update_vec_calib("intra_weights_vec", X_intra, fwd_ret[-_n_intra:], symbol=sym)
        v_intra = _calib_vec("intra_weights_vec", sym)
        _intra_param_keys = [
            "intra_w_opening_momentum", "intra_w_vwap_position",
            "intra_w_volume_acceleration", "intra_w_oi_build",
            "intra_w_price_structure", "intra_w_lunch_reversal",
        ]
        if len(v_intra) == len(_intra_param_keys):
            for pk, wv in zip(_intra_param_keys, v_intra):
                _set_calib(pk, round(wv, 6), sym)

    # ── 15. Intraday blend weight ─────────────────────────────────────────────
    intra_score_h = H("_calib_intraday_score_hist")
    n_ib = min(len(intra_score_h), len(fwd_ret))
    if n_ib >= _dynamic_min_obs(sym):
        # Calibrate the blend weight: how much does intraday improve over factor-only?
        factor_h = H("_calib_raw_score_hist")
        n_blend  = min(len(intra_score_h), len(factor_h), len(fwd_ret))
        if n_blend >= _dynamic_min_obs(sym):
            X_blend = np.column_stack([factor_h[-n_blend:], intra_score_h[-n_blend:]])
            _update_vec_calib("intra_blend_vec", X_blend, fwd_ret[-n_blend:], symbol=sym)
            v_blend = _calib_vec("intra_blend_vec", sym)
            if len(v_blend) == 2:
                # intra_blend_weight = fraction of total weight going to intraday
                _set_calib("intra_blend_weight", round(v_blend[1], 6), sym)


def _bootstrap_signal_history(ohlcv_df, symbol: str = "", horizon: int = 4,
                               min_bars: int = 60, max_bars: int = 252):
    """Bootstrap historical calibration from past OHLCV data.

    Replays a simplified factor model over up to `max_bars` of daily price/volume
    history to produce paired (signal_score, forward_return, actual_up) observations.
    These are written directly into the calibration session-state histories so that
    _run_calibration_cycle immediately has real data to work with — no weeks of
    live Load clicks required.

    Signals reconstructed from OHLCV alone (no chain data needed):
      • EMA structure  (20d vs 50d vs 200d cross)  — trend factor proxy
      • RSI 14         (momentum / overbought-oversold)
      • 5-day return   (short-term momentum)
      • 20-day HV      (vol regime proxy — high HV → bearish lean)
      • ATR percentile (trend strength proxy for ADX)

    This is a deliberate approximation — it captures the directional factors
    but omits flow and positioning (which require chain data). The result is
    a calibrated logistic_sharpness and Brier baseline from day one rather
    than after weeks of accumulation.

    The function is idempotent: it checks whether bootstrap has already been
    run for this symbol + bar-count combination and skips if so.
    """
    sym = symbol.upper() if symbol else ""

    if ohlcv_df is None or ohlcv_df.empty:
        return

    try:
        c_all = ohlcv_df["close"].astype(float).dropna().reset_index(drop=True)
        n_all = len(c_all)
        if n_all < min_bars + horizon:
            return   # not enough history

        # ── Idempotency guard ──────────────────────────────────────────────────
        # Key encodes symbol + number of bars so re-loading with more data re-runs.
        _guard_key = f"_bootstrap_done:{sym}:{n_all}"
        if st.session_state.get(_guard_key, False):
            return
        st.session_state[_guard_key] = True

        # ── Use at most max_bars of history ────────────────────────────────────
        c = c_all.tail(max_bars + horizon).reset_index(drop=True)
        n = len(c)

        # Pre-compute indicators across the full window
        # EMA series
        e20  = c.ewm(span=20,  adjust=False).mean()
        e50  = c.ewm(span=50,  adjust=False).mean()
        e200 = c.ewm(span=200, adjust=False).mean()

        # RSI 14
        delta = c.diff()
        gain  = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
        loss  = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
        rsi_s = (100 - 100 / (1 + gain / loss.replace(0, float("nan")))).fillna(50.0)

        # 20-day HV (annualised)
        lr_s  = np.log(c / c.shift(1))
        hv20  = lr_s.rolling(20).std() * math.sqrt(252)

        # ATR 14-day proxy (using only close — no high/low in simple OHLCV schema)
        atr_s = lr_s.abs().rolling(14).mean() * math.sqrt(252)
        atr_pct_s = atr_s.rank(pct=True).fillna(0.5)   # percentile rank

        # 5-day log return
        ret5  = np.log(c / c.shift(5))

        # Minimum warm-up: need 200 bars of EMA + 20 HV + 14 ATR
        warmup = min(200, n // 2)

        raw_scores  = []
        prob_ups    = []
        actual_ups  = []
        fwd_returns = []

        _sharpness_prior = _PRIOR["logistic_sharpness"]  # use prior for bootstrap

        for i in range(warmup, n - horizon):
            try:
                spot_i  = float(c.iloc[i])
                e20_i   = float(e20.iloc[i])
                e50_i   = float(e50.iloc[i])
                e200_i  = float(e200.iloc[i])
                rsi_i   = float(rsi_s.iloc[i])
                hv_i    = float(hv20.iloc[i]) if not math.isnan(hv20.iloc[i]) else 0.15
                atr_i   = float(atr_pct_s.iloc[i])
                ret5_i  = float(ret5.iloc[i]) if not math.isnan(ret5.iloc[i]) else 0.0

                # ── Mini factor model (price/vol signals only) ─────────────────
                # EMA structure: fraction of checks bullish, mapped to [-1,+1]
                ema_checks = [spot_i > e20_i, spot_i > e50_i, e20_i > e50_i,
                              spot_i > e200_i, e50_i > e200_i]
                ema_z = (sum(ema_checks) / len(ema_checks)) * 2 - 1   # [-1,+1]

                # RSI z-score (50 = neutral)
                rsi_z = max(-1.0, min(1.0, (rsi_i - 50.0) / 25.0))

                # 5-day momentum z-score (annualise then clamp)
                mom_z = max(-1.0, min(1.0, ret5_i * math.sqrt(252 / 5) / 0.30))

                # HV regime: high HV → vol expensive → slight bearish lean
                hv_z  = max(-1.0, min(1.0, -(hv_i - 0.15) / 0.10))   # z around 15% baseline

                # ATR percentile as trend strength (high ATR → trend is in motion)
                atr_z = (atr_i - 0.5) * 2.0   # [-1,+1]

                # Weighted composite (no flow/positioning — unavailable from OHLCV)
                # Weights sum to 1.0; use the CFG factor weights as guide but
                # redistribute flow+positioning weight proportionally to what's available.
                # FIX13/14: RSI weight reduced to minimal (confirming only)
                # Momentum (5d return) also reduced — bootstrap is vol/trend only
                raw = (0.50 * ema_z      # EMA structure: primary trend signal
                     + 0.15 * mom_z      # 5d momentum: short confirming
                     + 0.05 * rsi_z      # RSI: minimal — easily gamed by mean-reversion
                     + 0.20 * hv_z       # vol regime: important for bootstrap
                     + 0.10 * atr_z)     # ATR pct: trend strength proxy
                raw = max(-1.0, min(1.0, raw))

                prob_up_i = 1.0 / (1.0 + math.exp(-raw * _sharpness_prior))

                # Forward outcome: was next-horizon-day close > today?
                fwd_c   = float(c.iloc[i + horizon])
                fwd_ret = math.log(fwd_c / spot_i) if spot_i > 0 else 0.0
                act_up  = 1.0 if fwd_c > spot_i else 0.0

                raw_scores.append(raw)
                prob_ups.append(prob_up_i)
                actual_ups.append(act_up)
                fwd_returns.append(fwd_ret)

            except Exception:
                continue

        if len(raw_scores) < _dynamic_min_obs():
            return

        # ── Write into calibration histories ──────────────────────────────────
        # Use _set_calib-style direct session-state writes (bypass _record to
        # avoid load-id guard — we want to write unconditionally here).
        _ns = lambda key: f"{sym}:{key}" if sym else key

        def _write_hist(key, values):
            existing = list(st.session_state.get(_ns(key), []))
            # Prepend bootstrap observations (older), keep live ones at end
            combined = (values + existing)[-_CALIB_WINDOW:]
            st.session_state[_ns(key)] = combined
            # Also write to global (no-prefix) key as fallback
            if sym:
                g_existing = list(st.session_state.get(key, []))
                st.session_state[key] = (values + g_existing)[-_CALIB_WINDOW:]

        # FIX8 (MONARCH v2 Tier1): Bootstrap ONLY writes vol/return histories.
        # Directional signals (raw_score, prob_up, actual_up) are intentionally
        # excluded — bootstrapped scores use price-only proxies that would corrupt
        # flow/positioning calibration. Only neutral volatility-based histories
        # and normalised forward returns are safe to bootstrap.
        _write_hist("_calib_realised_ret_hist",   fwd_returns)   # safe: raw log returns
        # Write HV proxy as vol_regime_z baseline (volatility signal only)
        hv_z_scores = [max(-1.0, min(1.0, -(float(hv20.iloc[i]) - 0.15) / 0.10))
                        for i in range(warmup, n - horizon)
                        if not math.isnan(float(hv20.iloc[i]))]
        _write_hist("_calib_vol_regime_z_hist",   hv_z_scores[:len(fwd_returns)])

        # EMA and RSI histories for sub-factor calibration
        ema_scores = []
        rsi_scores = []
        for i in range(warmup, n - horizon):
            try:
                spot_i = float(c.iloc[i])
                ema_checks = [spot_i > float(e20.iloc[i]),  spot_i > float(e50.iloc[i]),
                              float(e20.iloc[i]) > float(e50.iloc[i]),
                              spot_i > float(e200.iloc[i]), float(e50.iloc[i]) > float(e200.iloc[i])]
                ema_scores.append((sum(ema_checks) / len(ema_checks)) * 2 - 1)
                rsi_v = float(rsi_s.iloc[i])
                rsi_scores.append(max(-1.0, min(1.0, (rsi_v - 50.0) / 25.0)))
            except Exception:
                ema_scores.append(0.0); rsi_scores.append(0.0)

        # FIX8: EMA/RSI/ADX directional signal histories removed from bootstrap.
        # These would bias calibration toward price-only trend signals.
        # They will be populated from real resolved outcomes instead.

        # Now run the full calibration cycle with the bootstrapped data
        _run_calibration_cycle(ohlcv_df, symbol=sym, horizon=horizon)

    except Exception:
        pass   # bootstrap is best-effort; never crash the Load flow


def _record(key: str, val: float, symbol: str = None):
    """Append a scalar observation to a calibration history list.
    If symbol is None, auto-reads opt_symbol from session state.
    Keeps the last _CALIB_WINDOW entries.
    """
    sym    = symbol if symbol is not None else st.session_state.get("opt_symbol", "")
    ns_key = f"{sym.upper()}:{key}" if sym else key
    hist   = st.session_state.get(ns_key, [])
    hist.append(float(val))
    if len(hist) > _CALIB_WINDOW:
        hist = hist[-_CALIB_WINDOW:]
    st.session_state[ns_key] = hist

def _record_if_load(key: str, val: float, symbol: str = None):
    """Record only on genuine Load clicks (not every Streamlit re-render).
    Checks opt_load_id against _last_recorded_load_id before calling _record.
    Use this for all _record() calls inside functions that run every render
    (compute_probabilistic_score, ev_rank_strategies, etc.).
    The final prob_up/raw_score records in PART 8 are exempt — they have
    their own guard that also updates _last_recorded_load_id.
    """
    _cur  = st.session_state.get("opt_load_id", 0)
    _last = st.session_state.get("_last_recorded_load_id", -1)
    if _cur != _last:
        _record(key, val, symbol)


def _get_hist(key: str, symbol: str = "") -> list:
    """Retrieve a signal history list, trying symbol-specific first then global."""
    if symbol:
        ns = st.session_state.get(f"{symbol.upper()}:{key}", [])
        if ns:
            return ns
    return st.session_state.get(key, [])


def _record_outcome(symbol: str, signal_snapshot: dict, horizon_days: int = 4):
    """Record a signal snapshot with its entry price so forward returns can be
    computed when the next load happens.  Stored in a per-symbol pending queue.
    Called at every LOAD; resolved at the NEXT load of the same symbol.

    signal_snapshot must contain: raw_score, flow_score, mc_direction,
    factor_direction, ev_score, dir_align, safety_ratio, ts_slope,
    pcr_level_z, oi_skew_z, mp_z, vol_regime_z, term_slope_z,
    rs_z, rs_slope_z, ema_score, adx_score, rsi_z,
    iv_pillar, adx_pillar, hv_accel_pillar, gex_pillar.
    """
    key = f"_outcome_pending_{symbol.upper()}"
    pending = st.session_state.get(key, [])
    pending.append({
        "ts":            datetime.now().isoformat(timespec="seconds"),
        "spot":          signal_snapshot.get("spot", 0.0),
        "horizon":       horizon_days,
        **{k: float(v) for k, v in signal_snapshot.items()
           if k != "spot" and isinstance(v, (int, float))}
    })
    # Keep last 100 pending entries
    st.session_state[key] = pending[-100:]


def _adaptive_threshold(key: str, fallback: float, symbol: str = "",
                         percentile: float = 70.0) -> float:
    """Return a performance-adaptive threshold for a signal gate.

    Computes the `percentile`-th percentile of the signal's SUCCESS distribution
    (i.e. observations where the signal fired AND the subsequent return agreed),
    NOT the raw signal value distribution.

    When fewer than 20 success observations exist, returns `fallback` from CFG.
    This replaces all hard-coded CFG threshold usage in signal gating.
    """
    sym    = symbol or st.session_state.get("opt_symbol", "").upper()
    s_key  = f"{sym}:_thresh_success_{key}" if sym else f"_thresh_success_{key}"
    success_hist = st.session_state.get(s_key, [])
    if len(success_hist) < 20:
        return fallback
    return float(np.percentile(success_hist, percentile))


def _record_threshold_outcome(key: str, signal_value: float, outcome: float,
                               symbol: str = ""):
    """Record a signal value into the success history IFF it was followed by a
    correct directional outcome. Called from _ingest_resolved_outcomes.
    """
    if not (math.isfinite(signal_value) and math.isfinite(outcome)):
        return
    if float(signal_value) > 0 and float(outcome) > 0:
        _record(f"_thresh_success_{key}", signal_value, symbol)
    elif float(signal_value) < 0 and float(outcome) < 0:
        _record(f"_thresh_success_{key}", abs(signal_value), symbol)


def _resolve_outcomes(symbol: str, current_spot: float, ohlcv_df) -> list:
    """Resolve pending signal snapshots by computing realised forward returns.
    Called at LOAD time. For each pending entry whose horizon has elapsed,
    compute log(current_spot / entry_spot) and pair it with all stored signals.
    Returns list of resolved (signal_dict, realised_return) pairs.
    Removes resolved entries from the pending queue.
    """
    key     = f"_outcome_pending_{symbol.upper()}"
    pending = st.session_state.get(key, [])
    if not pending:
        return []

    # Use OHLCV close prices for accurate forward returns when available
    if ohlcv_df is not None and not ohlcv_df.empty:
        closes = ohlcv_df["close"].astype(float).values
    else:
        closes = None

    resolved  = []
    remaining = []
    today     = datetime.now().date()

    for entry in pending:
        try:
            entry_date = datetime.fromisoformat(entry["ts"]).date()
            elapsed    = int(np.busday_count(entry_date.isoformat(), today.isoformat()))
            horizon    = int(entry.get("horizon", 4))

            if elapsed >= horizon:
                entry_spot = float(entry.get("spot", 0))
                if entry_spot <= 0:
                    continue  # can't compute return without entry spot

                # FIX: Use the close at entry_date + horizon_days, not today's close.
                # closes[-1] was always today regardless of elapsed days — look-ahead contamination.
                # Now we find the actual index: closes[-elapsed] approximates the close
                # that was elapsed trading sessions ago (most recent close at resolution time).
                # If elapsed > len(closes), fall back to current spot (live price).
                if closes is not None and elapsed <= len(closes):
                    # closes is a daily array ending today: closes[-1]=today, closes[-2]=yesterday.
                    # We want the close AT entry_date + horizon_days, not today's close.
                    # When elapsed == horizon (just resolved): _idx=1 → closes[-1] (today). ✓
                    # When elapsed > horizon (resolved late):  _idx>1 → closes[-_idx].
                    # FIX 2: clamp _idx to len(closes) to prevent IndexError when elapsed is
                    # large relative to the OHLCV history window available.
                    _idx = max(1, elapsed - int(entry.get("horizon", 4)) + 1)
                    _idx = min(_idx, len(closes))   # bounds guard
                    _target_close = float(closes[-_idx])
                    realised_ret = float(np.log(_target_close / entry_spot))
                else:
                    realised_ret = float(np.log(current_spot / entry_spot))

                # Normalize by realized vol (return / vol → Sharpe-like unit)
                _rv_key = f"{symbol.upper()}:_calib_realised_ret_hist"
                _rv_hist = st.session_state.get(_rv_key, [])
                if len(_rv_hist) >= 5:
                    _rv_std = float(np.std(_rv_hist[-20:])) or 0.01
                else:
                    _rv_std = 0.01   # fallback: ~1% daily vol
                norm_ret = max(-3.0, min(3.0, realised_ret / (_rv_std + 1e-9)))
                resolved.append((entry, norm_ret))   # use vol-normalised return
            else:
                remaining.append(entry)
        except Exception:
            remaining.append(entry)

    st.session_state[key] = remaining
    return resolved


def _ingest_resolved_outcomes(symbol: str, resolved_pairs: list):
    """Feed resolved (signal, return) pairs into signal history buffers
    so the calibration cycle can train on real outcomes.
    This is the core feedback loop: market reality → calibration.
    """
    for entry, ret in resolved_pairs:
        sym = symbol.upper()
        _record("_calib_raw_score_hist",   entry.get("raw_score",   0.0), sym)
        _record("_calib_ev_score_hist",    entry.get("ev_score",    0.0), sym)
        _record("_calib_dir_align_hist",   entry.get("dir_align",   0.5), sym)
        _record("_calib_pcr_level_hist",   entry.get("pcr_level_z", 0.0), sym)
        _record("_calib_oi_skew_hist",     entry.get("oi_skew_z",   0.0), sym)
        _record("_calib_mp_z_hist",        entry.get("mp_z",        0.0), sym)
        _record("_calib_vol_regime_z_hist",entry.get("vol_regime_z",0.0), sym)
        _record("_calib_term_slope_z_hist",entry.get("term_slope_z",0.0), sym)
        _record("_calib_rs_z_hist",        entry.get("rs_z",        0.0), sym)
        _record("_calib_rs_slope_hist",    entry.get("rs_slope_z",  0.0), sym)
        _record("_calib_ema_score_hist",   entry.get("ema_score",   0.0), sym)
        _record("_calib_adx_score_hist",   entry.get("adx_score",   0.0), sym)
        _record("_calib_rsi_trend_hist",   entry.get("rsi_z",       0.0), sym)
        _record("_calib_mc_dir_hist",      entry.get("mc_direction",0.0), sym)
        _record("_calib_fac_dir_hist",     entry.get("factor_direction",0.0), sym)
        _record("_calib_iv_pillar_hist",   entry.get("iv_pillar",   0.0), sym)
        _record("_calib_adx_pillar_hist",  entry.get("adx_pillar",  0.0), sym)
        _record("_calib_hva_pillar_hist",  entry.get("hv_accel_pillar", 0.0), sym)
        _record("_calib_gex_pillar_hist",  entry.get("gex_pillar",  0.0), sym)
        # Record the realised return itself as a forward return observation
        _record("_calib_realised_ret_hist", ret, sym)
        # ── PART 8: Dedicated probability calibration histories ───────────────
        # These power the reliability diagram and Brier score in Edge Audit
        prob_up_stored = entry.get("raw_score", 0.0)  # fallback if prob_up not in entry
        # Use prob_up if stored in entry (newer snapshots), else derive from raw_score
        if "prob_up" in entry:
            prob_up_stored = float(entry["prob_up"])
        _record("_calib_prob_up_hist",    prob_up_stored, sym)
        _record("_calib_actual_up_hist",  1.0 if ret > 0 else 0.0, sym)
        # move_vs_iv: actual_move / expected_move — filled when we have spot data
        _em_stored = float(entry.get("expected_move", 0.0))
        _spot_stored = float(entry.get("spot", 0.0))
        if _spot_stored > 0 and _em_stored > 0:
            _actual_move = abs(math.exp(ret) - 1.0) * _spot_stored
            _record("_calib_move_vs_iv_hist", round(_actual_move / _em_stored, 4), sym)
        # FIX11: Feed adaptive threshold success histories (Fix 3)
        _record_threshold_outcome("iv_hv",  entry.get("vol_regime_z", 0.0), ret, sym)
        _record_threshold_outcome("pcr",    entry.get("pcr_level_z",  0.0), ret, sym)
        _record_threshold_outcome("adx",    entry.get("adx_score",    0.0), ret, sym)


# ── Statistical helpers ────────────────────────────────────────────────────────

def _zscore_clamp(series, current_val, clamp=3.0):
    """Return z-score of current_val in series, clamped to [-clamp, +clamp].
    Returns 0.0 when series is too short (< 5) or std ≈ 0.
    The clamp is adaptive: uses the 99th percentile of abs(z) in history
    rather than the fixed value of 3.0, once enough data accumulates.
    """
    s = pd.Series(series).dropna()
    if len(s) < 5:
        return 0.0
    mu, sd = float(s.mean()), float(s.std())
    if sd < 1e-9:
        return 0.0
    z = (current_val - mu) / sd
    # Adaptive clamp: 99th percentile of |z| in history (replaces fixed 3.0)
    if len(s) >= 30:
        adaptive_clamp = float(np.percentile(np.abs((s - mu) / sd), 99))
        clamp = max(2.0, min(5.0, adaptive_clamp))
    return float(max(-clamp, min(clamp, z)))


def _percentile_score(series, current_val):
    """Return what fraction of `series` is <= current_val (0.0–1.0).
    Returns 0.5 when series is too short (< 3).
    """
    s = pd.Series(series).dropna()
    if len(s) < 3:
        return 0.5
    return float((s <= current_val).mean())


def _rolling_zscore(series_full, window=252):
    """Given a pandas Series, compute rolling z-score using a trailing window."""
    s = pd.Series(series_full).dropna()
    if len(s) < 5:
        return 0.0
    tail = s.tail(window)
    mu, sd = float(tail.mean()), float(tail.std())
    if sd < 1e-9:
        return 0.0
    return float((float(s.iloc[-1]) - mu) / sd)


def _normalise_to_signal(raw: float, hist_key: str, record: bool = True) -> float:
    """Convert any raw value to a [-1, +1] signal using its own rolling distribution.
    Uses rank-based normalisation (percentile → [-1,+1]) so the output is
    always well-scaled regardless of the raw value's unit or magnitude.
    Automatically records the value into `hist_key` for future calibration.
    """
    hist = st.session_state.get(hist_key, [])
    if record:
        hist.append(float(raw))
        if len(hist) > _CALIB_WINDOW:
            hist = hist[-_CALIB_WINDOW:]
        st.session_state[hist_key] = hist
    if len(hist) < 3:
        return 0.0
    pct = _percentile_score(hist, float(raw))
    return float(2.0 * pct - 1.0)   # maps [0,1] percentile → [-1,+1]

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
[data-testid="stSidebar"] label { color: var(--bb-muted) !important; font-size: .82rem !important; }
[data-testid="stSidebar"] .stDivider, [data-testid="stSidebar"] hr { border-color: var(--bb-border) !important; }

/* ── Typography ── */
h1 { font-family: 'IBM Plex Mono', monospace !important; color: var(--bb-amber) !important;
     font-size: 1.25rem !important; font-weight: 600 !important; letter-spacing: .15em !important;
     text-transform: uppercase !important; border-bottom: 1px solid var(--bb-amber) !important; padding-bottom: 4px !important; }
h2 { color: var(--bb-amber2) !important; font-size: 1.05rem !important; letter-spacing: .1em !important; text-transform: uppercase !important; }
h3 { color: var(--bb-white) !important; font-size: .96rem !important; letter-spacing: .08em !important; }
p, li, span, div { font-family: 'IBM Plex Mono', monospace !important; }

/* ── st.caption / small text ── */
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p,
small, .stCaption, .caption { color: var(--bb-muted) !important; font-size: .80rem !important; }

/* ── st.markdown prose text ── */
[data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li, [data-testid="stMarkdownContainer"] td,
[data-testid="stMarkdownContainer"] th {
    color: var(--bb-white) !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: .90rem !important;
}
[data-testid="stMarkdownContainer"] strong { color: var(--bb-amber2) !important; }
[data-testid="stMarkdownContainer"] code {
    background: #1a1400 !important; color: var(--bb-amber) !important;
    padding: 1px 4px !important; border-radius: 0 !important; font-size: .86rem !important;
}
[data-testid="stMarkdownContainer"] table { border-collapse: collapse !important; width: 100% !important; }
[data-testid="stMarkdownContainer"] th { background: #1a1400 !important; color: var(--bb-amber) !important; border: 1px solid var(--bb-border) !important; padding: 5px 10px !important; font-size: .80rem !important; }
[data-testid="stMarkdownContainer"] td { border: 1px solid var(--bb-border) !important; padding: 5px 10px !important; color: var(--bb-white) !important; font-size: .84rem !important; }

/* ── Metrics ── */
[data-testid="metric-container"] {
    background: var(--bb-surface) !important; border: 1px solid var(--bb-border) !important;
    border-left: 3px solid var(--bb-amber) !important; padding: 10px 14px !important; border-radius: 0 !important;
}
[data-testid="metric-container"] label, [data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] p { color: var(--bb-muted) !important; font-size: .76rem !important; letter-spacing: .1em !important; text-transform: uppercase !important; }
[data-testid="stMetricValue"], [data-testid="stMetricValue"] div { color: var(--bb-amber) !important; font-size: 1.25rem !important; font-weight: 600 !important; }
[data-testid="stMetricDelta"] { font-size: .80rem !important; }
[data-testid="stMetricDelta"] svg { display: none !important; }

/* ── DataFrames ── */
[data-testid="stDataFrame"] { border: 1px solid var(--bb-border) !important; }
[data-testid="stDataFrame"] *, .stDataFrame * { background-color: transparent !important; }
[data-testid="stDataFrame"] [data-testid="stDataFrameResizable"] { background: var(--bb-surface) !important; }
.stDataFrame thead tr th { background: #1a1400 !important; color: var(--bb-amber) !important; font-size: .80rem !important; letter-spacing: .1em !important; text-transform: uppercase !important; border-bottom: 1px solid var(--bb-amber) !important; }
.stDataFrame tbody tr td { font-size: .86rem !important; color: var(--bb-white) !important; border-bottom: 1px solid #1a1a1a !important; background: var(--bb-surface) !important; }
.stDataFrame tbody tr:hover td { background: #1a1400 !important; }
/* Streamlit 1.x iframe-based dataframe */
.stDataFrame iframe { background: var(--bb-surface) !important; }

/* ── Buttons ── */
.stButton > button { background: #1a1400 !important; color: var(--bb-amber) !important; border: 1px solid var(--bb-amber) !important;
    border-radius: 0 !important; font-size: .88rem !important; letter-spacing: .1em !important; text-transform: uppercase !important; padding: 7px 16px !important; }
.stButton > button:hover { background: var(--bb-amber) !important; color: #000 !important; }
.stButton > button:disabled { opacity: .4 !important; }

/* ── Inputs: text, number, selectbox ── */
.stSelectbox > div > div, .stTextInput > div > div, .stNumberInput > div > div {
    background: var(--bb-surface) !important; border: 1px solid var(--bb-border) !important;
    border-radius: 0 !important; color: var(--bb-white) !important; font-size: .90rem !important; }
.stSelectbox label, .stTextInput label, .stNumberInput label {
    color: var(--bb-muted) !important; font-size: .80rem !important; font-family: 'IBM Plex Mono', monospace !important; }
.stSelectbox div[data-baseweb="select"] > div { background: var(--bb-surface) !important; color: var(--bb-white) !important; }
/* Dropdown popup list */
ul[data-baseweb="menu"], [data-baseweb="popover"], [data-baseweb="popover"] li {
    background: #1a1a1a !important; color: var(--bb-white) !important; font-family: 'IBM Plex Mono', monospace !important; font-size: .88rem !important; border: 1px solid var(--bb-border) !important; }
[data-baseweb="option"]:hover { background: #1a1400 !important; }
/* Number input spinners */
.stNumberInput button { background: var(--bb-surface) !important; color: var(--bb-muted) !important; border: 1px solid var(--bb-border) !important; }
input[type="number"], input[type="text"], input[type="password"] {
    background: var(--bb-surface) !important; color: var(--bb-white) !important;
    border: 1px solid var(--bb-border) !important; font-family: 'IBM Plex Mono', monospace !important; font-size: .90rem !important; border-radius: 0 !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] { background: var(--bb-surface) !important; border-bottom: 1px solid var(--bb-border) !important; gap: 0 !important; }
.stTabs [data-baseweb="tab"] { background: transparent !important; color: var(--bb-muted) !important; font-size: .82rem !important; letter-spacing: .1em !important; text-transform: uppercase !important; border-radius: 0 !important; border-right: 1px solid var(--bb-border) !important; padding: 9px 16px !important; }
.stTabs [aria-selected="true"] { background: #1a1400 !important; color: var(--bb-amber) !important; border-bottom: 2px solid var(--bb-amber) !important; }
[data-testid="stTabContent"] { background: var(--bb-bg) !important; }

/* ── Expanders ── */
[data-testid="stExpander"] { background: var(--bb-surface) !important; border: 1px solid var(--bb-border) !important; border-radius: 0 !important; }
[data-testid="stExpanderDetails"] { background: var(--bb-bg) !important; border-top: 1px solid var(--bb-border) !important; }

/* ── EXPANDER ARROW OVERLAP FIX ── */
[data-testid="stExpander"] summary,
[data-testid="stExpander"] details > summary {
    display: flex !important; align-items: center !important;
    padding: 9px 14px !important; list-style: none !important;
    cursor: pointer !important; overflow: hidden !important;
}
[data-testid="stExpander"] summary::-webkit-details-marker,
[data-testid="stExpander"] summary::marker { display: none !important; }
[data-testid="stExpander"] summary span,
[data-testid="stExpander"] summary > div > span,
[data-testid="stExpander"] summary .material-icons,
[data-testid="stExpander"] summary [class*="material"],
[data-testid="stExpander"] summary [data-testid="StyledFullScreenButton"] {
    font-size: 0 !important; width: 0 !important; height: 0 !important;
    overflow: hidden !important; visibility: hidden !important; display: none !important;
}
/* Keep the actual label paragraph readable */
[data-testid="stExpander"] summary p,
[data-testid="stExpander"] summary > div > p {
    font-size: 0.82rem !important; color: var(--bb-amber2) !important;
    font-family: 'IBM Plex Mono', monospace !important; letter-spacing: 0.08em !important;
    text-transform: uppercase !important; visibility: visible !important;
    display: block !important; margin: 0 !important; overflow: visible !important;
}
[data-testid="stExpander"] summary:hover p { color: var(--bb-amber) !important; }
/* Hide SVG chevrons */
[data-testid="stExpander"] summary svg {
    display: none !important; visibility: hidden !important; width: 0 !important; height: 0 !important;
}

/* ── Divider ── */
hr, [data-testid="stDivider"] { border-color: var(--bb-border) !important; margin: 8px 0 !important; }
[data-testid="stDivider"] hr { border-top: 1px solid var(--bb-border) !important; }
/* ── Alerts ── */
[data-testid="stAlert"] { background: var(--bb-surface) !important; border: 1px solid var(--bb-border) !important; color: var(--bb-white) !important; border-radius: 0 !important; font-size: .88rem !important; }
[data-testid="stAlert"] p { color: var(--bb-white) !important; }
.stInfo { border-left: 3px solid var(--bb-blue) !important; }
.stWarning { border-left: 3px solid var(--bb-amber) !important; }
.stError { border-left: 3px solid var(--bb-red) !important; }
.stSuccess { border-left: 3px solid var(--bb-green) !important; }
[data-testid="stSpinner"] p, .stSpinner p { color: var(--bb-amber) !important; }
/* ── Scrollbars ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--bb-bg); }
::-webkit-scrollbar-thumb { background: var(--bb-border); }
::-webkit-scrollbar-thumb:hover { background: var(--bb-amber); }
/* ── Hide Streamlit branding ── */
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] { display: none !important; }
</style>
""", unsafe_allow_html=True)

# Terminal header
st.markdown("""
<div style="background:#ff8c00;color:#000;font-family:'IBM Plex Mono',monospace;
font-size:0.79rem;font-weight:600;letter-spacing:.18em;padding:5px 14px;
display:flex;justify-content:space-between;margin-bottom:12px;">
  <span>◼ MONARCH OPTIONS INTELLIGENCE — NSE F&O</span>
  <span>OPTIONS · DERIVATIVES · STRATEGY ENGINE</span>
</div>
""", unsafe_allow_html=True)

# ============================================================
# TOKEN — shared with Home.py and screener_pro.py
# ============================================================
TOKEN_FILE = ".upstox_token_scanner"

# TOKEN HANDSHAKE FIX (was: always overwrite opt_access_token from file on first visit)
#
# The original block ran whenever "opt_token_loaded" was absent, which is true on
# every fresh page navigation — including after the user just logged in on Home.py.
# It unconditionally set opt_access_token from the file, wiping whatever Home.py's
# save_token() had already written into session_state.
#
# Correct priority order:
#   1. opt_access_token already in session_state and non-empty  → use it (Home.py set it)
#   2. upstox_token in session_state and non-empty              → use it (Home.py's key)
#   3. scanner_token in session_state and non-empty             → use it (screener key)
#   4. TOKEN_FILE on disk and non-empty                         → use it (previous session)
#   5. Empty string                                             → show token input
#
# The "opt_token_loaded" guard is kept so this runs only once per session, but it
# no longer overwrites a token that is already live in session_state.

if "opt_token_loaded" not in st.session_state:
    _existing = (
        st.session_state.get("opt_access_token", "")   # already set by Home.py or a previous run
        or st.session_state.get("upstox_token", "")    # Home.py primary key
        or st.session_state.get("scanner_token", "")   # screener_pro key
    )
    if not _existing and os.path.exists(TOKEN_FILE):
        try:
            with open(TOKEN_FILE, "r") as f:
                _existing = f.read().strip()
        except OSError:
            _existing = ""
    st.session_state.opt_access_token = _existing
    st.session_state.opt_token_loaded = True
else:
    # On subsequent re-renders: if Home.py updated upstox_token after option.py
    # already loaded, sync it forward so get_token() stays fresh.
    _home_tok = (st.session_state.get("upstox_token", "")
                 or st.session_state.get("scanner_token", ""))
    if _home_tok and not st.session_state.get("opt_access_token", ""):
        st.session_state.opt_access_token = _home_tok

with st.sidebar:
    st.markdown("### 🔑 Upstox Token")
    # Always show current live token (may have been set by Home.py after page load)
    _cur_tok = st.session_state.get("opt_access_token", "")
    tok_inp = st.text_input("Access Token", type="password",
                             value=_cur_tok, key="opt_tok_inp")
    if tok_inp and tok_inp != _cur_tok:
        st.session_state.opt_access_token = tok_inp
        # Sync to Home.py and screener keys so all pages stay in lockstep
        st.session_state.upstox_token  = tok_inp
        st.session_state.scanner_token = tok_inp
        # FIX: defer cache clearing until after all cached functions are defined.
        # Calling .clear() here (before their definitions at line ~1606+) raises
        # NameError on every token save. Use a session flag instead; the actual
        # clear() calls run in _clear_api_caches() which is called after all
        # @st.cache_data functions are defined (see call site below).
        st.session_state["_pending_cache_clear"] = True
        try:
            with open(TOKEN_FILE, "w") as f: f.write(tok_inp)
            st.success("Token saved ✔")
        except: pass

# ── Dynamic token helpers ─────────────────────────────────────────────────────
# CRITICAL FIX: Never freeze ACCESS_TOKEN or HEADERS at module load time.
# Streamlit re-executes from top on every interaction; the token may have been
# pasted AFTER the module first ran, so a module-level constant would be stale.
# Always read from session_state at call time.
def get_token() -> str:
    return st.session_state.get("opt_access_token", "")

def get_headers() -> dict:
    return {"Authorization": f"Bearer {get_token()}", "Accept": "application/json"}

# Legacy aliases so older code that references ACCESS_TOKEN still works.
# These re-evaluate on every use because they call the function above.
class _DynToken:
    """Proxy that evaluates to the current token string on every str() call."""
    def __str__(self):  return get_token()
    def __bool__(self): return bool(get_token())
    def __eq__(self, o): return get_token() == o
    def __ne__(self, o): return get_token() != o

ACCESS_TOKEN = _DynToken()   # acts like a string; re-reads session_state each time
HEADERS      = get_headers() # used only in places that call get_headers() explicitly

if not get_token():
    st.warning("⚠️  Paste your Upstox access token in the sidebar to continue.")
    st.stop()

# ============================================================
# IV HISTORY PERSISTENCE — survives page restarts
# ============================================================

def _load_iv_history() -> dict:
    """Load IV history dict {symbol: [iv_float, ...]} from disk.
    Returns empty dict on any error."""
    fp = CFG["iv_hist_file"]
    try:
        if os.path.exists(fp):
            with open(fp, "r") as f:
                data = json.load(f)
            # Validate structure
            if isinstance(data, dict):
                return {k: [float(x) for x in v if isinstance(x, (int, float))]
                        for k, v in data.items() if isinstance(v, list)}
    except Exception:
        pass
    return {}

def _save_iv_history(hist: dict):
    """Persist IV history dict to disk. Silent on failure."""
    try:
        with open(CFG["iv_hist_file"], "w") as f:
            json.dump(hist, f)
    except Exception:
        pass

def _append_iv(symbol: str, iv: float):
    """Append current ATM IV to the persistent history for a symbol.
    Trims to CFG['iv_hist_max'] entries."""
    hist = st.session_state.opt_iv_history
    sym  = symbol.upper()
    if sym not in hist:
        hist[sym] = []
    hist[sym].append(round(float(iv), 6))
    if len(hist[sym]) > CFG["iv_hist_max"]:
        hist[sym] = hist[sym][-CFG["iv_hist_max"]:]
    st.session_state.opt_iv_history = hist
    _save_iv_history(hist)

# ============================================================
# BLACK-SCHOLES ENGINE (pure Python, no scipy)
# — Merton continuous-dividend form: q = annualised dividend yield
# — Theta uses CFG["theta_days"] (252 trading days, not 365 calendar)
# — T should always be trading-day fraction: use trading_t() below
# ============================================================

def _ncdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def _npdf(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

def trading_t(expiry_date_str: str) -> float:
    """Return time-to-expiry as trading-day fraction of a year.
    Counts Mon–Fri weekdays between today and expiry, then subtracts NSE
    trading holidays that fall within that window.
    Uses CFG['ann_days'] as the annualisation base (252).
    Returns at least 1/252 so T is never zero on expiry day itself.

    NSE holiday list: updated for the current and next calendar year.
    Source: NSE India circular (https://www.nseindia.com/regulations/holiday-master).
    Add new years as needed — the list is filtered to the relevant window at runtime.
    """
    # NSE trading holidays (market closed, not weekends).
    # Format: "YYYY-MM-DD".  Keep this list current; stale entries do no harm
    # (they only matter if they fall between today and expiry).
    _NSE_HOLIDAYS = {
        # 2024
        "2024-01-22", "2024-03-25", "2024-03-29", "2024-04-11", "2024-04-14",
        "2024-04-17", "2024-04-21", "2024-05-23", "2024-06-17", "2024-07-17",
        "2024-08-15", "2024-10-02", "2024-11-01", "2024-11-15", "2024-11-20",
        "2024-12-25",
        # 2025
        "2025-02-26", "2025-03-14", "2025-03-31", "2025-04-10", "2025-04-14",
        "2025-04-18", "2025-05-01", "2025-08-15", "2025-08-27", "2025-10-02",
        "2025-10-20", "2025-10-21", "2025-11-05", "2025-12-25",
        # 2026
        "2026-01-26", "2026-03-19", "2026-04-02", "2026-04-03", "2026-04-06",
        "2026-04-14", "2026-04-30", "2026-05-01", "2026-08-15", "2026-09-16",
        "2026-10-02", "2026-11-09", "2026-12-25",
    }
    try:
        exp   = datetime.strptime(expiry_date_str, "%Y-%m-%d").date()
        today = datetime.now().date()
        if exp <= today:
            return 1.0 / CFG["ann_days"]
        # Count Mon–Fri weekdays (numpy busday_count)
        td = int(np.busday_count(today.isoformat(), exp.isoformat()))
        # Subtract NSE holidays that are weekdays within [today, exp)
        for h in _NSE_HOLIDAYS:
            hd = date.fromisoformat(h)
            if today <= hd < exp and hd.weekday() < 5:   # Mon=0 … Fri=4
                td -= 1
        td = max(td, 1)
        return td / CFG["ann_days"]
    except Exception:
        return 7.0 / CFG["ann_days"]   # safe fallback: 7 trading days

def _sanitise_iv(iv_raw: float, fallback: float) -> float:
    """Normalise IV from API (may be percent-form or decimal form),
    then clamp to [0.01, 5.0] (1% – 500% annualised).
    Returns fallback if result is still invalid.

    Threshold rationale: Upstox returns IV in percent form (e.g. 43.5 → 43.5%).
    We divide by 100 when iv_raw >= 3.0 to convert to decimal.
    Threshold is 3.0 (not 2.0) because:
      • No legitimate traded option has IV between 2% and 3% in decimal (0.02–0.03);
        if the API returned 2.3 it almost certainly means 2.3% percent-form.
      • iv_raw exactly 2.0 with threshold >2.0 would be treated as decimal (200% IV)
        when it could be 2% percent-form — a 100× error.
      • >= 3.0 safely captures all percent-form values while leaving genuine
        decimal IVs (e.g. 0.35 = 35%) untouched.
    """
    if not iv_raw or iv_raw <= 0:
        return fallback
    iv = iv_raw / 100.0 if iv_raw >= 3.0 else iv_raw
    if iv < 0.01 or iv > 5.0:
        return fallback
    return iv

def bs_price(S, K, T, r, sigma, opt="call", q=0.0):
    """European BSM with continuous dividend yield q (Merton 1973).
    q=0 is identical to the classic formula."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max(S - K, 0) if opt == "call" else max(K - S, 0)
    F  = S * math.exp((r - q) * T)          # cost-of-carry forward price
    d1 = (math.log(F / K) + 0.5 * sigma**2 * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    df = math.exp(-r * T)                    # discount factor
    if opt == "call":
        return math.exp(-q * T) * S * _ncdf(d1) - K * df * _ncdf(d2)
    return K * df * _ncdf(-d2) - math.exp(-q * T) * S * _ncdf(-d1)

def bs_greeks(S, K, T, r, sigma, opt="call", q=0.0):
    """Greeks with dividend yield q and 252-trading-day theta."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return dict(delta=1.0 if opt=="call" else -1.0, gamma=0, theta=0, vega=0)
    sqrtT = math.sqrt(T)
    F  = S * math.exp((r - q) * T)
    d1 = (math.log(F / K) + 0.5 * sigma**2 * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    nd1 = _npdf(d1)
    df  = math.exp(-r * T)
    dq  = math.exp(-q * T)

    delta = dq * (_ncdf(d1) if opt == "call" else _ncdf(d1) - 1)
    gamma = dq * nd1 / (S * sigma * sqrtT)
    # Theta: divide by CFG["theta_days"] (252) — per-trading-day decay
    if opt == "call":
        theta = (-(S * dq * nd1 * sigma) / (2 * sqrtT)
                 + q * S * dq * _ncdf(d1)
                 - r * K * df * _ncdf(d2)) / CFG["theta_days"]
    else:
        # FIX 4: Correct put theta formula (Hull 10th ed §19).
        # q·S term: put loses carry income → + q·S·e^(-qT)·N(-d1)  (positive)
        # r·K term: put benefits from interest on strike → - r·K·e^(-rT)·N(-d2)  (negative)
        # Previous code had BOTH signs wrong (−q and +r), making put theta too positive.
        theta = (-(S * dq * nd1 * sigma) / (2 * sqrtT)
                 + q * S * dq * _ncdf(-d1)
                 - r * K * df * _ncdf(-d2)) / CFG["theta_days"]
    vega  = S * dq * nd1 * sqrtT / 100   # per 1% IV change
    return dict(delta=round(delta,4), gamma=round(gamma,6),
                theta=round(theta,4), vega=round(vega,4))

def implied_vol(mkt_px, S, K, T, r, opt="call", q=0.0):
    """Robust IV solver: Newton-Raphson with bisection fallback.
    Initial guess: ATM formula near-the-money, intrinsic/time approx for deep OTM/ITM.
    Bisection guarantees convergence in [0.01, 3.0] if Newton fails (vega collapse, overshoot).
    """
    if T <= 0 or mkt_px <= 0 or S <= 0 or K <= 0:
        return None
    # FIX 3: Use true intrinsic value max(S-K, 0) / max(K-S, 0) as the lower bound.
    # The previous formula used present-value parity (S·e^(-qT) - K·e^(-rT)), which
    # is the theoretical lower bound for European options but not the true intrinsic.
    # For deep ITM options it can exceed the actual market price and cause spurious
    # None returns.  True intrinsic is always ≤ any real market price for a live option.
    intrinsic = max(S - K, 0.0) if opt == "call" else max(K - S, 0.0)
    if mkt_px <= intrinsic * 0.999:
        return None
    sqrtT = math.sqrt(T)
    _bs_const = math.sqrt(2.0 / math.pi)
    moneyness = abs(S - K) / S
    if moneyness < 0.05:
        # Near-the-money: Brenner-Subrahmanyam ATM approximation
        sig = max(0.05, min(mkt_px / (S * sqrtT * _bs_const * 0.4), 2.0))
    else:
        # Away from ATM: time-value approximation
        time_val = max(mkt_px - intrinsic, 1e-6)
        sig = max(0.05, min(math.sqrt(2.0 * math.pi / T) * time_val / S, 2.0))

    # Phase 1: Newton-Raphson (up to 100 iters)
    converged = False
    for _ in range(100):
        px  = bs_price(S, K, T, r, sig, opt, q)
        F   = S * math.exp((r - q) * T)
        d1  = (math.log(max(F / K, 1e-9)) + 0.5 * sig**2 * T) / (sig * sqrtT)
        vega = S * math.exp(-q * T) * _npdf(d1) * sqrtT
        if vega < 1e-10:
            break
        diff = mkt_px - px
        if abs(diff) < 1e-7:
            converged = True
            break
        sig += diff / vega
        sig  = max(0.001, min(sig, 10.0))

    # Phase 2: bisection fallback in [0.01, 3.0] if Newton did not converge
    if not converged:
        lo, hi = 0.01, 3.0
        f_lo = bs_price(S, K, T, r, lo, opt, q) - mkt_px
        f_hi = bs_price(S, K, T, r, hi, opt, q) - mkt_px
        if f_lo * f_hi > 0:
            return None   # price outside achievable range even at 300% vol
        for _ in range(60):
            mid = (lo + hi) / 2.0
            f_mid = bs_price(S, K, T, r, mid, opt, q) - mkt_px
            if abs(f_mid) < 1e-6 or (hi - lo) < 1e-6:
                sig = mid
                break
            if f_lo * f_mid <= 0:
                hi = mid; f_hi = f_mid
            else:
                lo = mid; f_lo = f_mid
        else:
            sig = (lo + hi) / 2.0

    return round(sig, 6) if 0.01 < sig < 4.99 else None

def bs_itm_prob(S, K, T, r, sigma, opt="call", q=0.0):
    """Risk-neutral probability of exercise = N(d2) for call, N(-d2) for put."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 1.0 if (opt == "call" and S > K) else (1.0 if (opt == "put" and S < K) else 0.0)
    F  = S * math.exp((r - q) * T)
    d1 = (math.log(F / K) + 0.5 * sigma**2 * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return round(_ncdf(d2) if opt == "call" else _ncdf(-d2), 4)

def bs_prob_touch(S, K, T, r, sigma, opt="call", q=0.0, real_drift=None):
    """Probability of price TOUCHING the strike K before expiry T.
    Uses the Reiner-Rubinstein (1991) first-passage barrier formula:
    For UP barrier H > S (call):   P = N(d+) + exp(2*mu*ln(H/S)/σ²) * N(d-)
    For DOWN barrier H < S (put):  P = N(-d+) + exp(-2*mu*ln(S/H)/σ²) * N(-d-)
    where mu = drift - σ²/2,  d± = (±ln(S/H) ± mu*T) / (σ√T)
    real_drift: if provided, use real-world mu instead of risk-neutral (r-q).
    Returns value in [0,1]. Already-ITM barriers return 1.0.
    """
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 1.0
    _base_drift = real_drift if real_drift is not None else (r - q)
    mu    = _base_drift - 0.5 * sigma**2
    sqrtT = math.sqrt(T)
    if opt == "call":
        if K <= S:
            return 1.0   # barrier already reached
        lnHoverS = math.log(K / S)   # positive
        d_plus   = (-lnHoverS + mu * T) / (sigma * sqrtT)   # negative
        d_minus  = (-lnHoverS - mu * T) / (sigma * sqrtT)   # more negative
        refl_exp = 2.0 * mu * lnHoverS / (sigma**2)
        refl     = math.exp(max(-700.0, min(700.0, refl_exp)))
        p = _ncdf(d_plus) + refl * _ncdf(d_minus)
    else:
        if K >= S:
            return 1.0   # barrier already reached
        lnSoverH = math.log(S / K)   # positive
        d_plus   = (lnSoverH + mu * T) / (sigma * sqrtT)
        d_minus  = (lnSoverH - mu * T) / (sigma * sqrtT)
        refl_exp = -2.0 * mu * lnSoverH / (sigma**2)
        refl     = math.exp(max(-700.0, min(700.0, refl_exp)))
        p = _ncdf(-d_plus) + refl * _ncdf(-d_minus)
    return round(max(0.0, min(1.0, p)), 4)


def build_iv_surface(chain_df, spot, atm_iv):
    """Build implied volatility surface from option chain data.
    Uses CE IV for strikes >= ATM, PE IV for strikes <= ATM.
    Returns a callable sigma(K) via linear interpolation over valid strikes.
    Falls back to atm_iv for any strike outside the observed range.
    Uses scipy if available; falls back to numpy np.interp (always available).
    """
    if chain_df is None or chain_df.empty:
        return lambda K: atm_iv

    strikes, ivs = [], []
    for _, row in chain_df.iterrows():
        k = float(row["Strike"])
        iv_raw = float(row.get("CE_IV", 0) or 0) if k >= spot else float(row.get("PE_IV", 0) or 0)
        iv = _sanitise_iv(iv_raw, None)
        if iv is not None and iv > 0:
            strikes.append(k)
            ivs.append(iv)

    if len(strikes) < 2:
        return lambda K: atm_iv

    # Sort by strike, build numpy arrays
    pairs   = sorted(zip(strikes, ivs), key=lambda x: x[0])
    ks_arr  = np.array([p[0] for p in pairs], dtype=float)
    vs_arr  = np.array([p[1] for p in pairs], dtype=float)

    # scipy gives cubic/linear interp; np.interp is always available as fallback
    try:
        from scipy.interpolate import interp1d as _interp1d
        f = _interp1d(ks_arr, vs_arr, kind="linear", bounds_error=False,
                      fill_value=(vs_arr[0], vs_arr[-1]))
        return lambda K: float(f(float(K)))
    except ImportError:
        # numpy linear interp — clamps to endpoints for extrapolation (correct behaviour)
        return lambda K: float(np.interp(float(K), ks_arr, vs_arr))
    except Exception:
        return lambda K: atm_iv


def _estimate_real_world_drift(ohlcv_df, r, q):
    """Estimate annualised real-world drift mu from last 60 trading days of log returns.
    Falls back to risk-neutral drift (r - q) if insufficient data.
    """
    if ohlcv_df is None or ohlcv_df.empty:
        return r - q
    try:
        c = ohlcv_df["close"].astype(float)
        if len(c) < 10:
            return r - q
        lr = np.log(c / c.shift(1)).dropna()
        tail = lr.tail(60)
        if len(tail) < 10:
            return r - q
        mu = float(tail.mean() * CFG["ann_days"])
        # Clamp to reasonable range: –100% to +200% annualised drift
        return max(-1.0, min(2.0, mu))
    except Exception:
        return r - q


def _tx_cost_per_leg(premium: float, action: str, qty: int = 1) -> float:
    """Return total transaction cost in ₹ for ONE leg (one unit, not lot-adjusted).

    Cost components applied at ENTRY (single side — exit is at intrinsic in MC,
    which has no premium-based charges):
      • STT:        0.0625% of premium on SELL legs (seller pays on entry)
      • Stamp duty: 0.003%  of premium on BUY  legs (buyer pays on entry)
      • NSE charge: 0.053%  of premium (both sides)
      • SEBI fee:   0.0001% of premium (both sides)
      • GST 18%:    on NSE charge + SEBI fee (not on STT or stamp duty)

    qty: number of units (not lots — caller multiplies by lot_size if needed).
    Returns ₹ cost per unit to subtract from the leg's effective premium.
    """
    tc  = CFG["tx_cost"]
    pr  = max(float(premium), 0.0)
    act = str(action).lower()

    # Side-specific charges
    stt        = pr * tc["stt_sell_pct"]   if act == "sell" else 0.0
    stamp      = pr * tc["stamp_duty_pct"] if act == "buy"  else 0.0

    # Both-side charges
    nse_charge = pr * tc["nse_charge_pct"]
    sebi_fee   = pr * tc["sebi_fee_pct"]
    gst        = (nse_charge + sebi_fee) * tc["gst_rate"]

    total_per_unit = (stt + stamp + nse_charge + sebi_fee + gst) * qty
    return round(total_per_unit, 4)


def _fit_jump_params(ohlcv_df, ann_days: int = 252):
    """Fit Merton jump-diffusion parameters from historical OHLCV log-returns.

    Method-of-moments on the empirical log-return distribution:
      • Identify jump days as |daily_log_return| > 2σ  (extreme moves)
      • λ  = jump_count / trading_years  (annualised jump frequency)
      • μ_j = mean(log_returns on jump days)
      • σ_j = std(log_returns on jump days)

    Falls back to Nifty-calibrated defaults if fewer than 60 daily bars.
    Returns (lambda, mu_j, sigma_j) as floats.
    """
    # Nifty-index defaults (calibrated to historical NSE data)
    _default = (3.0, -0.02, 0.04)

    if ohlcv_df is None or ohlcv_df.empty:
        return _default
    try:
        c  = ohlcv_df["close"].astype(float).dropna()
        if len(c) < 60:
            return _default

        lr = np.log(c / c.shift(1)).dropna().values
        if len(lr) < 30:
            return _default

        # Diffusion vol from core returns (trim top/bottom 5% to exclude jumps)
        p5, p95 = np.percentile(lr, 5), np.percentile(lr, 95)
        core    = lr[(lr >= p5) & (lr <= p95)]
        sigma_d = float(np.std(core)) if len(core) > 5 else float(np.std(lr))
        if sigma_d < 1e-6:
            return _default

        # Jump days: |return| > 2σ of the core distribution
        threshold  = 2.0 * sigma_d
        jump_mask  = np.abs(lr) > threshold
        jump_rets  = lr[jump_mask]
        n_jumps    = int(jump_mask.sum())
        n_years    = len(lr) / ann_days

        if n_jumps < 3 or n_years < 0.1:
            return _default

        lam   = float(n_jumps / n_years)
        mu_j  = float(np.mean(jump_rets))
        sig_j = float(np.std(jump_rets)) if n_jumps > 1 else abs(mu_j) * 0.5

        # Sanity clamps: λ ∈ [0.5, 20], μ_j ∈ [-0.15, 0.05], σ_j ∈ [0.01, 0.15]
        lam   = max(0.5, min(20.0, lam))
        mu_j  = max(-0.15, min(0.05, mu_j))
        sig_j = max(0.01, min(0.15, sig_j))

        return (lam, mu_j, sig_j)
    except Exception:
        return _default


def _mc_jump_prices(spot, T, mu, atm_sigma_sim, jump_lam, jump_mu, jump_sig, n_sims, rng):
    """Shared helper: generate Merton jump-diffusion terminal prices.
    Used by both strategy_prob_profit definitions (FIX 6).
    GBM drift is adjusted so that total E[log S_T] equals the real-world mu drift.
    """
    Z = rng.standard_normal(n_sims)
    jump_mean_ret = math.exp(jump_mu + 0.5 * jump_sig**2) - 1.0
    drift_adj = mu - jump_lam * jump_mean_ret
    drift = (drift_adj - 0.5 * atm_sigma_sim**2) * T
    diff  = atm_sigma_sim * math.sqrt(T) * Z
    # Jump component: compound Poisson
    n_jumps_per_path = rng.poisson(jump_lam * T, n_sims)
    max_j = int(n_jumps_per_path.max()) if n_jumps_per_path.max() > 0 else 0
    if max_j > 0:
        J_draws = rng.normal(jump_mu, jump_sig, (n_sims, max_j))
        mask = np.arange(max_j)[None, :] < n_jumps_per_path[:, None]
        jump_log_ret = (J_draws * mask).sum(axis=1)
    else:
        jump_log_ret = np.zeros(n_sims)
    return spot * np.exp(drift + diff + jump_log_ret)


def strategy_prob_profit(legs, spot, T, r, atm_iv, q=0.0, simulations=None,
                         chain_df=None, ohlcv_df=None):
    """Monte Carlo estimate of Probability of Profit for a multi-leg strategy.
    Improvements:
      1. VOL SURFACE: each leg repriced at its own strike IV (skew-aware), not ATM IV.
      2. REAL-WORLD DRIFT: uses historical mu from last 60 days instead of risk-neutral.
      3. JUMP DIFFUSION (FIX 6): Merton jump component fitted from OHLCV history.
    legs: list of dicts with keys: opt (CE/PE), strike, premium, action (Buy/Sell), qty
    Returns: prob_profit (float 0–1), expected_value (₹ per unit)
    """
    n_sims = simulations or CFG["pop_simulations"]
    if T <= 0 or atm_iv <= 0 or not legs:
        return 0.5, 0.0

    iv_surf = build_iv_surface(chain_df, spot, atm_iv)
    mu = _estimate_real_world_drift(ohlcv_df, r, q)
    jump_lam, jump_mu, jump_sig = _fit_jump_params(ohlcv_df, CFG["ann_days"])  # FIX 6

    _seed = int(abs(spot * 1000 + T * 1e6 + sum(l.get("strike", 0) for l in legs))) % (2**31)
    rng   = np.random.default_rng(_seed)
    atm_sigma_sim = iv_surf(spot)

    # FIX 6: Merton jump-diffusion prices via shared helper
    prices = _mc_jump_prices(spot, T, mu, atm_sigma_sim, jump_lam, jump_mu, jump_sig, n_sims, rng)

    total_pnl = np.zeros(n_sims)
    for leg in legs:
        k    = float(leg["strike"])
        pr   = float(leg["premium"])
        qty  = int(leg.get("qty", 1))
        d    = 1 if leg["action"].lower() == "buy" else -1
        if leg["opt"].upper() in ("CE", "CALL"):
            intr = np.maximum(prices - k, 0)
        else:
            intr = np.maximum(k - prices, 0)
        total_pnl += d * (intr - pr) * qty

    prob_profit = float((total_pnl > 0).mean())
    ev          = float(total_pnl.mean())
    return round(prob_profit, 4), round(ev, 4)


def iv_percentile(iv_series):
    """IV Percentile: fraction of past observations that current IV exceeds.
    Unlike IV Rank (range-normalized), IV Percentile is robust to outliers.
    Returns 0–100."""
    s = pd.Series(iv_series).dropna()
    if len(s) < 3:
        return 50.0
    cur = float(s.iloc[-1])
    pct = float((s < cur).mean() * 100)
    return round(pct, 1)


def _load_signal_log() -> list:
    """Load forward signal log from disk. Returns list of dicts."""
    fp = CFG["signal_log_file"]
    try:
        if os.path.exists(fp):
            with open(fp, "r") as f:
                data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception:
        pass
    return []

def _save_signal_log(log: list):
    """Persist signal log to disk. Keeps last CFG['signal_log_max'] entries."""
    try:
        log = log[-CFG["signal_log_max"]:]
        with open(CFG["signal_log_file"], "w") as f:
            json.dump(log, f)
    except Exception:
        pass

def _append_signal(symbol: str, prob_up: float, prob_down: float, flow_score: float,
                   flow_magnitude: float, raw_score: float, top_strategy: str,
                   strategy_ev: float, strategy_pop: float, strategy_kelly: float,
                   expected_move: float, atm_iv: float, ivr: float, bias: str,
                   # Edge Audit fields — full snapshot per spec
                   spot: float = 0.0, hv20: float = 0.0,
                   pcr: float = 1.0, oi_total: float = 0.0, oi_skew: float = 0.0,
                   max_pain: float = 0.0, max_pain_dist_pct: float = 0.0,
                   skew_pp: float = 0.0, term_slope: float = 0.0,
                   positioning_score: float = 0.0, vol_regime_score: float = 0.0,
                   trend_score: float = 0.0, intraday_score: float = 0.0,
                   dte: int = 1):
    """Append a forward signal snapshot to the persistent log.
    Now captures full Edge Audit fields so the Edge Diagnostic tab can measure
    what actually happens after each signal across 1/2/3/5-day horizons.
    """
    if "opt_signal_log" not in st.session_state:
        st.session_state.opt_signal_log = _load_signal_log()
    # Expected move = ATM_IV * sqrt(DTE / ann_days) using 252 trading-day base.
    # Using calendar days (365) inflates EM by sqrt(365/252) ≈ 1.20× — wrong.
    _em_calc = atm_iv * math.sqrt(max(dte, 1) / CFG["ann_days"]) * spot if spot > 0 else expected_move
    entry = {
        "ts":             datetime.now().isoformat(timespec="minutes"),
        "symbol":         symbol.upper(),
        # Direction signals
        "prob_up":        round(prob_up, 4),
        "prob_down":      round(prob_down, 4),
        "raw_score":      round(raw_score, 4),
        "flow_score":     round(flow_score, 4),
        "flow_magnitude": round(flow_magnitude, 4),
        "bias":           bias,
        # Strategy
        "top_strategy":   top_strategy,
        "strategy_ev":    round(strategy_ev, 2),
        "strategy_pop":   round(strategy_pop, 4),
        "strategy_kelly": round(strategy_kelly, 4),
        # Price & Vol at signal time
        "spot":           round(spot, 2),
        "atm_iv_pct":     round(atm_iv * 100, 2),
        "hv20_pct":       round(hv20 * 100, 2),
        "iv_hv_ratio":    round(atm_iv / (hv20 + 1e-9), 3) if hv20 > 0.01 else 0.0,
        "ivr":            round(ivr, 1),
        "dte":            dte,
        # Expected move
        "expected_move":  round(_em_calc, 2),
        # Positioning at signal time
        "pcr":            round(pcr, 3),
        "oi_total":       round(oi_total, 0),
        "oi_skew":        round(oi_skew, 0),
        "max_pain":       round(max_pain, 2),
        "max_pain_dist_pct": round(max_pain_dist_pct, 3),
        # Skew & term structure
        "skew_pp":        round(skew_pp, 3),
        "term_slope":     round(term_slope, 4),
        # Sub-scores for grouping
        "positioning_score": round(positioning_score, 4),
        "vol_regime_score":  round(vol_regime_score, 4),
        "trend_score":       round(trend_score, 4),
        "intraday_score":    round(intraday_score, 4),
        # Forward outcomes — filled in later by _resolve_edge_outcomes()
        "fwd_ret_1d": None, "fwd_ret_2d": None, "fwd_ret_3d": None, "fwd_ret_5d": None,
        "fwd_spot_1d": None, "fwd_spot_2d": None, "fwd_spot_3d": None, "fwd_spot_5d": None,
        "fwd_iv_1d": None, "fwd_iv_2d": None,
        "fwd_skew_1d": None, "fwd_skew_2d": None,
        "fwd_oi_1d": None, "fwd_oi_2d": None,
        "resolved": False,
    }
    log = st.session_state.opt_signal_log
    # Deduplicate: don't append if last entry is same symbol within same minute
    if log and log[-1]["ts"] == entry["ts"] and log[-1]["symbol"] == entry["symbol"]:
        return
    log.append(entry)
    st.session_state.opt_signal_log = log
    _save_signal_log(log)
def bs_charm(S, K, T, r, sigma, opt="call", q=0.0):
    """Charm = dDelta/dt per trading day.
    Derived analytically from the Merton BSM delta:
      Call delta  = exp(-qT) * N(d1)
      Put  delta  = exp(-qT) * (N(d1) - 1)
    Differentiating wrt T (holding S, K, sigma fixed):
      charm = exp(-qT) * npdf(d1) * [d1*(r-q)/(sigma*sqrt(T)) - (r-q) - sigma/(2*sqrt(T))*d2/d1_adj]
    Standard closed form (Hull, 10th ed §19):
      charm_call = -exp(-qT) * [npdf(d1)*(2*(r-q)*T - d2*sigma*sqrt(T)) / (2*T*sigma*sqrt(T))
                                - q*N(d1)]
      charm_put  = charm_call + q*exp(-qT)   [because delta_put = delta_call - exp(-qT)]
    Both divided by ann_days for per-trading-day value.
    """
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    sqrtT = math.sqrt(T)
    F     = S * math.exp((r - q) * T)
    d1    = (math.log(F / K) + 0.5 * sigma**2 * T) / (sigma * sqrtT)
    d2    = d1 - sigma * sqrtT
    nd1   = _npdf(d1)
    dq    = math.exp(-q * T)
    # Core term shared by call and put
    core  = -dq * nd1 * (2 * (r - q) * T - d2 * sigma * sqrtT) / (2 * T * sigma * sqrtT)
    if opt == "call":
        charm_raw = core - q * dq * _ncdf(d1)
    else:
        # charm_put = charm_call + q*exp(-qT)  (from delta_put = delta_call - exp(-qT))
        charm_call = core - q * dq * _ncdf(d1)
        charm_raw  = charm_call + q * dq
    return round(charm_raw / CFG["ann_days"], 6)


def bs_vanna(S, K, T, r, sigma, opt="call", q=0.0):
    """Vanna = dDelta/dIV = d²V/dSdσ.
    Vanna = -exp(-q*T) * npdf(d1) * d2 / sigma
    Same for calls and puts.
    Returns vanna (delta change per 1 unit IV change, not per 1%).
    """
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    sqrtT = math.sqrt(T)
    F  = S * math.exp((r - q) * T)
    d1 = (math.log(F / K) + 0.5 * sigma**2 * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    nd1 = _npdf(d1)
    dq  = math.exp(-q * T)
    vanna = -dq * nd1 * d2 / sigma
    return round(vanna, 6)


def atm_strike(spot, step):
    # Use round-half-up (standard trading convention) not Python's banker's rounding.
    # Python's round(0.5)=0 (rounds to even), but traders expect 24525→24550 not 24500.
    # math.floor(x + 0.5) gives true round-half-up for positive numbers.
    return float(math.floor(spot / step + 0.5) * step)

def strikes_around(spot, step, n=6):
    atm = atm_strike(spot, step)
    return [round(atm + i * step, 2) for i in range(-n, n+1)]

# NOTE: _clear_api_caches() is defined below, after all @st.cache_data functions
# are defined (fetch_option_chain, fetch_expiries, fetch_upstox_candles,
# fetch_upstox_intraday_candles). The pending-clear execution is also deferred
# to that point so it never references functions that do not yet exist.

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_fii_dii_data() -> dict:
    """Derive an institutional flow proxy from publicly available market data.

    Uses three yfinance tickers that are proven proxies for FII activity:
      USDINR=X  — USD/INR spot. Rupee strengthening = FII buying India equities.
                  (Correlation with FII net ~0.65-0.75 historically)
      ^INDIAVIX — India VIX. Rising VIX = fear/hedging = FII selling or exiting.
      ^NSEI     — Nifty 50. Directional confirmation.

    Why this works better than NSE API on cloud:
      • No cookies, no bot-detection, works on Streamlit Cloud
      • Real-time (15-min delay) vs T+1 for NSE's published FII data
      • USDINR is the most reliable real-time FII proxy available

    Encoding (all signals → bullish = positive):
      rupee_signal:  rupee strengthening (USDINR falling) → FII buying → positive
      vix_signal:    VIX falling → risk-on, institutions adding → positive
      nifty_signal:  Nifty up → confirming institutional buying → positive

    Weights: USDINR 50% (leading), VIX 30% (confirming), Nifty RS 20% (confirming)
    Returns combined signal in [-1, +1].
    """
    result = {
        "fii_net_crore": 0.0, "dii_net_crore": 0.0, "combined_net": 0.0,
        "fii_hist": [], "dii_hist": [],
        "fii_3d_avg": 0.0,
        "fii_signal": 0.0, "dii_signal": 0.0, "combined_signal": 0.0,
        "source_date": "", "data_available": False,
        # Proxy-specific fields
        "usdinr":        0.0,
        "usdinr_signal": 0.0,
        "indiavix":      0.0,
        "vix_signal":    0.0,
        "proxy_mode":    True,   # flag so UI can label correctly
    }
    try:
        # ── Fetch 20 days of USDINR, India VIX, Nifty ──────────────────────────
        _raw = yf.download(
            ["USDINR=X", "^INDIAVIX", "^NSEI"],
            period="30d", interval="1d",
            progress=False, auto_adjust=True, group_by="ticker"
        )
        if _raw.empty:
            return result

        def _close(ticker):
            try:
                if isinstance(_raw.columns, pd.MultiIndex):
                    return _raw[ticker]["Close"].dropna()
                return _raw["Close"].dropna()
            except Exception:
                return pd.Series(dtype=float)

        fx   = _close("USDINR=X")   # USD per INR — lower = stronger rupee
        vix  = _close("^INDIAVIX")
        nsei = _close("^NSEI")

        if len(fx) < 3 or len(vix) < 3 or len(nsei) < 3:
            return result

        # ── 1-day and 3-day changes ──────────────────────────────────────────────
        def _pct_chg(series, n=1):
            """Percentage change over n periods, most recent."""
            s = series.dropna()
            if len(s) <= n:
                return 0.0
            return float((s.iloc[-1] - s.iloc[-1-n]) / (s.iloc[-1-n] + 1e-9))

        fx_1d   = _pct_chg(fx,   1)   # positive = rupee weakened = FII selling
        vix_1d  = _pct_chg(vix,  1)   # positive = VIX rose = risk-off
        nsei_1d = _pct_chg(nsei, 1)   # positive = Nifty up

        # ── Normalise each to z-score within 20D rolling distribution ────────────
        def _z(series, current_pct_chg, window=20):
            s = series.dropna()
            if len(s) < 5:
                return float(math.tanh(current_pct_chg * 20))
            pct_changes = s.pct_change().dropna().tail(window).values
            if len(pct_changes) < 3:
                return float(math.tanh(current_pct_chg * 20))
            mu = float(pct_changes.mean())
            sd = float(pct_changes.std())
            sd = max(sd, abs(mu) * 0.1 + 1e-6)
            return float(max(-1.0, min(1.0, (current_pct_chg - mu) / sd / 2.0)))

        # Encode direction: rupee up (fx down) = bullish → negate fx signal
        usdinr_sig = -_z(fx,   fx_1d)    # rupee strengthening = positive
        vix_sig    = -_z(vix,  vix_1d)   # VIX falling = positive
        nsei_sig   =  _z(nsei, nsei_1d)  # Nifty up = positive

        # Weighted composite: USDINR leads, VIX confirms, Nifty confirms
        combined = round(0.50 * usdinr_sig + 0.30 * vix_sig + 0.20 * nsei_sig, 4)

        # BUG FIX (FII Proxy Fabricated Crore Numbers):
        # The old code stored combined*3000 in fii_net_crore — entirely synthetic.
        # A user seeing "FII: Rs2,400 Cr" had no way to know this was 0.80x3000.
        # Fix 1: remove all crore fields; expose only the dimensionless signal score.
        # Fix 2: dii_signal was wrongly assigned vix_sig (semantically unrelated).
        #         DII flow is not derivable from this proxy; set to 0.0.
        # Fix 3: combined_signal remains the sole model input — correct as-is.
        source_date = str(fx.index[-1].date()) if hasattr(fx.index[-1], 'date') else str(fx.index[-1])[:10]

        result.update({
            # Proxy signal (dimensionless) -- used by the signal engine
            "fii_proxy_signal":   round(usdinr_sig, 4),
            "dii_proxy_signal":   0.0,
            "combined_signal":    combined,
            # Legacy field aliases -- values are now None; UI must NOT display as Rs crore
            "fii_net_crore":      None,   # REMOVED: was fabricated (combined*3000)
            "dii_net_crore":      None,   # REMOVED: was always 0.0
            "combined_net":       None,   # REMOVED: was fabricated
            "fii_3d_avg":         None,   # REMOVED: was fabricated
            # Backward-compat signal fields (engine reads these)
            "fii_signal":         round(usdinr_sig, 4),
            "dii_signal":         0.0,    # BUG FIX: was vix_sig (wrong semantics)
            "fii_hist":           [],
            "dii_hist":           [],
            # Display fields
            "source_date":        source_date,
            "data_available":     True,
            "dii_data_available": False,
            "usdinr":             round(float(fx.iloc[-1]), 4),
            "usdinr_signal":      round(usdinr_sig, 4),
            "indiavix":           round(float(vix.iloc[-1]), 2),
            "vix_signal":         round(vix_sig, 4),
            "proxy_mode":         True,
            "proxy_warning":      (
                "PROXY MODE: FII/DII flow estimated from USD/INR + India VIX + Nifty. "
                "No real SEBI/NSE institutional flow data. Signal is directional only."
            ),
        })
    except Exception:
        pass
    return result



# NOTE: get_ohlcv, get_intraday_ohlcv, and compute_intraday_signals were previously
# defined here (duplicate first copies). BUG FIX (Duplicate Function Definitions):
# Python silently uses the LAST definition, making the first copies dead code.
# Removed first copies; canonical definitions remain below (after _clear_api_caches).

def compute_hv(close_series, window=20):
    """Annualised historical vol using CFG['ann_days'] (252 trading days)."""
    lr = np.log(close_series / close_series.shift(1)).dropna()
    if len(lr) < window: return None
    return float(lr.tail(window).std() * np.sqrt(CFG["ann_days"]))
# ============================================================
# DIRECTIONAL ANALYSIS — 7-FACTOR MODEL
# ============================================================

def compute_flow_scores(chain_df, ohlcv_df):
    """Compute LEADING flow signals from delta-changes in positioning metrics.

    For 1–5 day prediction, CHANGES matter more than levels.
    A PCR moving from 1.2→1.5 is more predictive than PCR=1.5 in isolation.

    Signals and their directional encoding:
      dIV    (signed): rising IV → institutions buying options → bearish for underlying longs
                        falling IV → dealers covering shorts → bullish
      dOI    (signed): rising OI + price rising = bullish (fresh longs)
                        rising OI + price falling = bearish (fresh shorts)
                        Note: without price context, use magnitude only (neutral sign)
      dPCR   (signed): RISING put/call ratio = MORE puts being written = BULLISH (support)
                        FALLING put/call ratio = puts being closed or calls written = BEARISH
      dSkew  (signed): steepening skew (put IV > call IV widening) = downside fear = BEARISH
                        flattening/reversing skew = complacency = BULLISH
      dGEX   (signed): GEX rising = dealers buying more = range-bound / vol-suppressed
                        GEX falling = dealers selling = directional move coming

    Returns dict with each delta z-score in [-1, +1] and a composite flow_score.
    """
    flow = {
        "dIV": 0.0, "dOI": 0.0, "dPCR": 0.0,
        "dSkew": 0.0, "dGEX": 0.0,
        "dFII": 0.0,             # FII/DII institutional net flow signal
        "flow_score": 0.0,
        "flow_magnitude": 0.0,   # abs(flow_score) — conviction level
    }
    if chain_df is None or chain_df.empty:
        return flow

    try:
        # ── dIV: Change in ATM IV vs 5-session average ──────────────────────────
        # Rising IV = institutions buying options = protective hedging = bearish for spot
        # Falling IV = fear unwind = bullish for spot
        iv_hist  = st.session_state.get("opt_iv_history", {})
        sym_hist = iv_hist.get(st.session_state.get("opt_symbol", ""), [])
        if len(sym_hist) >= 5:
            cur_iv   = float(sym_hist[-1])
            avg_5    = float(np.mean(sym_hist[-5:]))
            std_5    = float(np.std(sym_hist[-5:])) if len(sym_hist) >= 5 else (avg_5 * _calib("std_floor_frac"))
            std_5    = max(std_5, avg_5 * (_calib("std_floor_frac") * 0.2))  # floor at fraction of avg
            dIV_raw  = (cur_iv - avg_5) / (std_5 + 1e-9)    # z-score of IV change
            # Encode: rising IV = bearish for longs → negative score
            flow["dIV"] = round(max(-1.0, min(1.0, -dIV_raw / 2.0)), 3)

        # ── dPCR: Change in put/call ratio ───────────────────────────────────────
        # COMPUTED FIRST so dOI can use its sign to determine direction.
        # RISING PCR (more puts being added) = put writers providing support = BULLISH
        # FALLING PCR (puts being closed or calls added) = hedgers exiting = BEARISH
        ce_oi    = float(chain_df["CE_OI"].sum())
        pe_oi    = float(chain_df["PE_OI"].sum())
        pcr_now  = pe_oi / (ce_oi + 1e-9)
        _cur_load_id = st.session_state.get("opt_load_id", 0)
        _pcr_hist  = st.session_state.get("_flow_pcr_hist", [])
        _pcr_lids = st.session_state.get("_flow_pcr_load_ids", [])
        if _pcr_lids and _pcr_lids[-1] == _cur_load_id:
            _pcr_hist[-1] = pcr_now
        else:
            _pcr_hist.append(pcr_now)
            _pcr_lids.append(_cur_load_id)
        if len(_pcr_hist) > 30: _pcr_hist = _pcr_hist[-30:]; _pcr_lids = _pcr_lids[-30:]
        st.session_state["_flow_pcr_hist"]     = _pcr_hist
        st.session_state["_flow_pcr_load_ids"] = _pcr_lids
        if len(_pcr_hist) >= 5:
            pcr_arr  = np.array(_pcr_hist)
            pcr_mu   = float(np.mean(pcr_arr[:-1]))
            pcr_std  = float(np.std(pcr_arr[:-1])) or (pcr_mu * _calib("std_floor_frac"))
            dPCR_z   = (pcr_now - pcr_mu) / (pcr_std + 1e-9)
            # Rising PCR = more put writing = bullish support → POSITIVE directional score
            flow["dPCR"] = round(max(-1.0, min(1.0, dPCR_z / 2.0)), 3)

        # ── dOI: Change in total chain open interest ─────────────────────────────
        # OI build-up signals fresh positioning — a large OI surge means new bets are placed.
        # Net direction unknown from OI alone → use dPCR sign (now computed above) to orient.
        total_oi = float(chain_df["CE_OI"].sum() + chain_df["PE_OI"].sum())
        _oi_hist = st.session_state.get("_flow_oi_hist", [])
        _oi_lids = st.session_state.get("_flow_oi_load_ids", [])
        if _oi_lids and _oi_lids[-1] == _cur_load_id:
            _oi_hist[-1] = total_oi   # overwrite same Load entry
        else:
            _oi_hist.append(total_oi)
            _oi_lids.append(_cur_load_id)
        if len(_oi_hist) > 30: _oi_hist = _oi_hist[-30:]; _oi_lids = _oi_lids[-30:]
        st.session_state["_flow_oi_hist"]     = _oi_hist
        st.session_state["_flow_oi_load_ids"] = _oi_lids
        if len(_oi_hist) >= 5:
            oi_arr   = np.array(_oi_hist)
            oi_mu    = float(np.mean(oi_arr[:-1]))
            oi_std   = float(np.std(oi_arr[:-1])) or (oi_mu * _calib("std_floor_frac"))
            dOI_z    = (total_oi - oi_mu) / (oi_std + 1e-9)
            # dPCR is now available (computed above). Use its sign to orient the OI surge:
            # positive dPCR = more put writing = bullish side growing → positive dOI.
            _dPCR_sign = math.copysign(1.0, flow.get("dPCR", 0.0)) if flow.get("dPCR", 0.0) != 0 else 0.0
            dOI_directed = dOI_z * _dPCR_sign if _dPCR_sign != 0 else abs(dOI_z)
            flow["dOI"] = round(max(-1.0, min(1.0, dOI_directed / 2.0)), 3)

        # ── dSkew: Change in IV skew (OTM put IV minus OTM call IV) ─────────────
        # Steepening put skew = downside hedging demand = BEARISH
        # Flattening / reversal = complacency or call-buying = BULLISH
        _spot = st.session_state.get("opt_spot", 0)
        _step = st.session_state.get("opt_step", 50)
        if _spot > 0 and _step > 0:
            atm_k      = atm_strike(_spot, _step)
            otm_ce_row = chain_df[(chain_df.Strike - (atm_k + _step)).abs() < 0.5]
            otm_pe_row = chain_df[(chain_df.Strike - (atm_k - _step)).abs() < 0.5]
            if not otm_ce_row.empty and not otm_pe_row.empty:
                ce_iv_now  = _sanitise_iv(float(otm_ce_row.CE_IV.values[0]), 0) or 0.0
                pe_iv_now  = _sanitise_iv(float(otm_pe_row.PE_IV.values[0]), 0) or 0.0
                skew_now   = (pe_iv_now - ce_iv_now) if (ce_iv_now > 0 and pe_iv_now > 0) else 0.0
                _skew_hist  = st.session_state.get("_flow_skew_hist", [])
                _skew_lids = st.session_state.get("_flow_skew_load_ids", [])
                if _skew_lids and _skew_lids[-1] == _cur_load_id:
                    _skew_hist[-1] = skew_now
                else:
                    _skew_hist.append(skew_now)
                    _skew_lids.append(_cur_load_id)
                if len(_skew_hist) > 30: _skew_hist = _skew_hist[-30:]; _skew_lids = _skew_lids[-30:]
                st.session_state["_flow_skew_hist"]     = _skew_hist
                st.session_state["_flow_skew_load_ids"] = _skew_lids
                if len(_skew_hist) >= 5:
                    sk_arr   = np.array(_skew_hist)
                    sk_mu    = float(np.mean(sk_arr[:-1]))
                    sk_std   = float(np.std(sk_arr[:-1])) or (_calib("std_floor_frac") * 0.1)
                    dSkew_z  = (skew_now - sk_mu) / (sk_std + 1e-9)
                    # Steepening skew = more put demand = bearish → NEGATIVE score
                    flow["dSkew"] = round(max(-1.0, min(1.0, -dSkew_z / 2.0)), 3)

        # ── dGEX: Change in net gamma exposure ───────────────────────────────────
        # Rising GEX = dealers long gamma = price pinning = range-bound
        # Falling GEX = dealers short gamma = price trending = vol expansion
        # For directional: falling GEX + price trend = amplified move
        _gex_hist  = st.session_state.get("_flow_gex_hist", [])
        _gex_lids  = st.session_state.get("_flow_gex_load_ids", [])
        oi_d_cur  = st.session_state.get("opt_oi", {})
        gex_now   = float(oi_d_cur.get("net_gex", 0) or 0)
        if gex_now != 0:   # only track when we have real GEX data
            if _gex_lids and _gex_lids[-1] == _cur_load_id:
                _gex_hist[-1] = gex_now
            else:
                _gex_hist.append(gex_now)
                _gex_lids.append(_cur_load_id)
            if len(_gex_hist) > 30: _gex_hist = _gex_hist[-30:]; _gex_lids = _gex_lids[-30:]
            st.session_state["_flow_gex_hist"]     = _gex_hist
            st.session_state["_flow_gex_load_ids"] = _gex_lids
            if len(_gex_hist) >= 5:
                gex_arr  = np.array(_gex_hist)
                gex_mu   = float(np.mean(gex_arr[:-1]))
                gex_std  = float(np.std(gex_arr[:-1])) or (abs(gex_mu) * 0.1 + 1e-9)
                dGEX_z   = (gex_now - gex_mu) / (gex_std + 1e-9)
                flow["dGEX"] = round(max(-1.0, min(1.0, -dGEX_z / 2.0)), 3)

        # ── dFII: FII/DII institutional cash market net flow ──────────────────────
        # FII net buying in cash market = bullish signal for Nifty F&O.
        # FII net selling = bearish. DII is counter-cyclical (weaker signal).
        # This is the most reliable leading indicator for 1-5 day Nifty direction.
        # Data is T+0 (same day after ~18:00 IST) or T-1 if fetched before close.
        _fii_data = st.session_state.get("opt_fii_dii", {})
        if _fii_data.get("data_available"):
            flow["dFII"] = round(float(_fii_data.get("combined_signal", 0.0)), 3)
        else:
            flow["dFII"] = 0.0

        # ── Track FII signal history for weight-learning ──────────────────────
        # IMPORTANT: append AFTER flow["dFII"] is set above — otherwise history
        # records 0.0 every session and the adaptive weight-learner trains on
        # all-zeros, driving the FII weight to near-zero incorrectly.
        _fii_sig_now = flow["dFII"]
        _fii_hist_store = st.session_state.get("_flow_fii_hist", [])
        _fii_hist_store.append(_fii_sig_now)
        if len(_fii_hist_store) > 30: _fii_hist_store = _fii_hist_store[-30:]
        st.session_state["_flow_fii_hist"] = _fii_hist_store

        # ── Composite FLOW SCORE (data-driven weights) ────────────────────────
        # fixed fallback weights (FII added at 15%, others scaled down proportionally):
        _fw_fallback = {
            "dPCR": 0.30, "dSkew": 0.25, "dIV": 0.17, "dOI": 0.08,
            "dGEX": 0.05, "dFII": 0.15,
        }
        signal_keys = ["dPCR", "dSkew", "dIV", "dOI", "dGEX", "dFII"]

        def _compute_flow_weights(ohlcv_df_local, horizon=4):
            """Return dict of normalised abs-correlation weights for each flow signal.
            horizon: forecast horizon in sessions (default 4 ≈ 1 trading week).
            Requires ≥10 aligned observations; otherwise returns fallback weights.
            """
            try:
                if ohlcv_df_local is None or ohlcv_df_local.empty or len(ohlcv_df_local) < horizon + 10:
                    return _fw_fallback.copy()

                c_local = ohlcv_df_local["close"].astype(float).reset_index(drop=True)
                # Forward log return: return realised horizon sessions later
                fwd_ret = np.log(c_local.shift(-horizon) / c_local).dropna().values

                # Retrieve signal histories aligned to the same sessions
                hist_map = {
                    "dIV":   (st.session_state.get("opt_iv_history") or {}).get(
                                 st.session_state.get("opt_symbol", ""), []),
                    "dPCR":  st.session_state.get("_flow_pcr_hist",  []),
                    "dSkew": st.session_state.get("_flow_skew_hist", []),
                    "dOI":   st.session_state.get("_flow_oi_hist",   []),
                    "dGEX":  st.session_state.get("_flow_gex_hist",  []),
                    "dFII":  st.session_state.get("_flow_fii_hist",  []),
                }

                raw_weights = {}
                for sig_k in signal_keys:
                    hist = np.array(hist_map[sig_k], dtype=float)
                    n_common = min(len(hist), len(fwd_ret))
                    if n_common < 10:
                        raw_weights[sig_k] = _fw_fallback[sig_k]  # not enough data
                        continue
                    # Align: use the most-recent n_common observations
                    sig_aligned = hist[-n_common:]
                    ret_aligned = fwd_ret[-n_common:]
                    # Compute Pearson correlation; use abs value for weight
                    if np.std(sig_aligned) < 1e-9 or np.std(ret_aligned) < 1e-9:
                        raw_weights[sig_k] = _fw_fallback[sig_k]
                        continue
                    corr = float(np.corrcoef(sig_aligned, ret_aligned)[0, 1])
                    if np.isnan(corr):
                        corr = 0.0
                    raw_weights[sig_k] = abs(corr)

                total_w = sum(raw_weights.values())
                if total_w < 1e-9:
                    return _fw_fallback.copy()
                return {k: v / total_w for k, v in raw_weights.items()}

            except Exception:
                return _fw_fallback.copy()

        _flow_weights = _compute_flow_weights(
            st.session_state.get("opt_ohlcv_df", None)
        )

        fs_composite = sum(_flow_weights.get(k, _fw_fallback[k]) * flow.get(k, 0.0)
                           for k in signal_keys)
        flow["flow_score"]      = round(max(-1.0, min(1.0, fs_composite)), 3)
        flow["flow_magnitude"]  = round(abs(fs_composite), 3)
        flow["flow_weights"]    = {k: round(v, 4) for k, v in _flow_weights.items()}

    except Exception:
        pass

    # Flow acceleration (MONARCH v2 Block E)
    try:
        _regime_now = st.session_state.get("opt_regime_label", _REGIME_TRANSITION)

        def _accel(hist_list, min_len=5):
            if len(hist_list) < min_len:
                return 0.0
            arr = np.array(hist_list[-min_len:], dtype=float)
            d2  = np.diff(np.diff(arr))
            if len(d2) == 0:
                return 0.0
            raw = float(d2[-1])
            _sd = float(np.std(d2)) if len(d2) >= 2 else 1.0
            return float(max(-1.0, min(1.0, raw / (2.0 * (_sd or 1.0)))))

        _sym_fa = st.session_state.get("opt_symbol", "")
        oi_acc  = _accel(st.session_state.get("_flow_oi_hist", []))
        pcr_acc = _accel(st.session_state.get("_flow_pcr_hist", []))
        iv_acc  = _accel((st.session_state.get("opt_iv_history") or {}).get(_sym_fa, []))

        _aw = 0.40 if _regime_now == _REGIME_TRANSITION else 0.20
        acc_comp = max(-1.0, min(1.0, 0.40 * oi_acc + 0.35 * pcr_acc + 0.25 * iv_acc))

        flow["flow_score"]        = round(
            max(-1.0, min(1.0, (1.0 - _aw) * flow.get("flow_score", 0.0) + _aw * acc_comp)), 3)
        flow["flow_acceleration"] = round(acc_comp, 3)
        flow["oi_accel"]          = round(oi_acc, 3)
        flow["pcr_accel"]         = round(pcr_acc, 3)
        flow["iv_accel"]          = round(iv_acc, 3)
    except Exception:
        flow.setdefault("flow_acceleration", 0.0)
        flow.setdefault("oi_accel", 0.0)
        flow.setdefault("pcr_accel", 0.0)
        flow.setdefault("iv_accel", 0.0)

    return flow


def detect_shock_event(atm_iv, chain_df, ohlcv_df) -> dict:
    """
    Detect abnormal events (IV spike, OI jump, cross-asset shock) via z-scores.
    When detected, returns override_weights that increase FLOW+VOL, reduce TREND.
    All thresholds are z-score based -- no fixed numbers.
    """
    result = {
        "shock_detected": False, "iv_shock": False, "oi_shock": False, "ca_shock": False,
        "shock_intensity": 0.0, "iv_zscore": 0.0, "oi_zscore": 0.0, "override_weights": None,
    }
    try:
        # IV z-score
        _sym   = st.session_state.get("opt_symbol", "")
        _iv_h  = st.session_state.get("opt_iv_history", {}).get(_sym, [])
        iv_z   = 0.0; iv_shock = False
        if len(_iv_h) >= 10:
            _a = np.array(_iv_h, dtype=float)
            _mu = float(_a[:-1].mean()); _sd = float(_a[:-1].std()) or (_mu * 0.1)
            iv_z = (atm_iv - _mu) / (_sd + 1e-9)
            iv_shock = abs(iv_z) > 2.5

        # OI z-score
        _oi_h   = st.session_state.get("_flow_oi_hist", [])
        oi_z    = 0.0; oi_shock = False
        if len(_oi_h) >= 5 and chain_df is not None and not chain_df.empty:
            _tot = float(chain_df["CE_OI"].sum() + chain_df["PE_OI"].sum())
            _a   = np.array(_oi_h[:-1], dtype=float)
            if len(_a) >= 3:
                _mu = float(_a.mean()); _sd = float(_a.std()) or (_mu * 0.05)
                oi_z = (_tot - _mu) / (_sd + 1e-9)
                oi_shock = abs(oi_z) > 2.5

        # Cross-asset shock — zscores are normalised to [-1,+1] by _norm()
        # so the threshold must be in that range (>2.0 would never fire)
        _ca   = st.session_state.get("opt_cross_asset", {})
        ca_shock = (_ca.get("data_available") and
                    any(abs(v) > 0.75 for v in _ca.get("zscores", {}).values()))

        _n  = sum([iv_shock, oi_shock, ca_shock])
        det = _n >= 1
        intensity = min(1.0, (abs(iv_z) + abs(oi_z)) / 6.0 + (0.3 if ca_shock else 0.0))

        ow = None
        if det:
            _raw = {
                "flow":        min(0.55, 0.30 + 0.15 * intensity),
                "positioning": 0.20,
                "vol_regime":  min(0.35, 0.20 + 0.10 * intensity),
                "rel_strength":0.05,
                "trend":       max(0.00, 0.10 - 0.08 * intensity),
            }
            _t = sum(_raw.values()) or 1.0
            ow = {k: v / _t for k, v in _raw.items()}

        result.update({
            "shock_detected": det, "iv_shock": iv_shock, "oi_shock": oi_shock,
            "ca_shock": ca_shock, "shock_intensity": round(intensity, 3),
            "iv_zscore": round(iv_z, 2), "oi_zscore": round(oi_z, 2),
            "override_weights": ow,
        })
    except Exception:
        pass
    return result



def directional_bias(df, ltp, chain_df=None):
    """Directional bias model — fully adaptive, no magic scaling constants.

    Every sub-signal is converted to a z-score (from its own rolling history)
    or a percentile rank within the current data series, then clamped to [-1, +1].

    Five independent factor groups (zero multicollinearity by design):
      TREND       (weight 0.30): EMA structure + ADX percentile
      MOMENTUM    (weight 0.25): RSI z-score + 5-day return z-score
      VOLATILITY  (weight 0.15): ATR percentile + BB width percentile (regime signal)
      POSITIONING (weight 0.20): PCR percentile + OI skew + max pain distance / EM
      REL STRENGTH(weight 0.10): stock vs Nifty z-score (from session state)

    Each factor score ∈ [-1, +1]; final score = weighted sum ∈ [-1, +1].
    Displayed as –100 to +100 for UI compatibility.
    """
    if df.empty or len(df) < 50:
        return {"bias": "NEUTRAL", "score": 0, "factors": {}, "rsi": 50, "macd_hist": 0,
                "bb_pct": 50, "vol_ratio": 1, "atr_pct": 1.5,
                "e9": ltp, "e20": ltp, "e50": ltp, "atr": _atr_seed(ltp),
                "flow": {}, "adx": 0.0}

    c = df["close"].astype(float)
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    v = df["volume"].astype(float)

    # ── Common series ────────────────────────────────────────────────────────
    e9   = c.ewm(span=9,   adjust=False).mean()
    e20  = c.ewm(span=20,  adjust=False).mean()
    e50  = c.ewm(span=50,  adjust=False).mean()
    e200 = c.ewm(span=200, adjust=False).mean()

    tr    = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    atr14 = tr.ewm(span=14, adjust=False).mean()
    atrv  = float(atr14.iloc[-1])
    if ltp > 0: _record("_calib_atr_pct_hist", atrv / ltp)

    e9v   = float(e9.iloc[-1])
    e20v  = float(e20.iloc[-1])
    e50v  = float(e50.iloc[-1])
    e200v = float(e200.iloc[-1]) if len(c) >= 200 else ltp

    # ── ADX (14-period Wilder) ───────────────────────────────────────────────
    up_move   = h - h.shift(1)
    down_move = l.shift(1) - l
    dm_p = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=c.index)
    dm_m = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=c.index)
    di_p = 100 * dm_p.ewm(alpha=1/14, adjust=False).mean() / (atr14 + 1e-9)
    di_m = 100 * dm_m.ewm(alpha=1/14, adjust=False).mean() / (atr14 + 1e-9)
    dx   = 100 * (di_p - di_m).abs() / (di_p + di_m + 1e-9)
    adx_series = dx.ewm(alpha=1/14, adjust=False).mean().dropna()
    adx_val    = float(adx_series.iloc[-1])
    adx_dir    = 1.0 if float(di_p.iloc[-1]) > float(di_m.iloc[-1]) else -1.0
    # ADX percentile within its own history (no fixed 25 threshold)
    adx_pct    = _percentile_score(adx_series.values, adx_val)  # 0–1

    # ── RSI ─────────────────────────────────────────────────────────────────
    delta = c.diff()
    gain  = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rs_raw = gain / loss.replace(0, float('nan'))
    rsi_series = (100 - 100 / (1 + rs_raw)).fillna(
        gain.apply(lambda g: 100.0 if g > 0 else 0.0))
    rsi = float(rsi_series.iloc[-1])
    # RSI z-score over 1-year history (no fixed 50 centre, no fixed 15 scale)
    rsi_z = _zscore_clamp(rsi_series.tail(252).values, rsi, clamp=3.0) / 3.0  # → [-1,+1]

    # ── MACD histogram ──────────────────────────────────────────────────────
    macd_l  = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    macd_s  = macd_l.ewm(span=9, adjust=False).mean()
    macd_hist_series = macd_l - macd_s
    macd_h  = float(macd_hist_series.iloc[-1])
    # MACD histogram z-score (no fixed std normaliser — derived from rolling 1Y)
    macd_z  = _zscore_clamp(macd_hist_series.tail(252).values, macd_h, clamp=3.0) / 3.0

    # ── Bollinger Bands ──────────────────────────────────────────────────────
    bm     = c.rolling(20).mean()
    bb_std = c.rolling(20).std()
    bup    = (bm + 2 * bb_std)
    blo    = (bm - 2 * bb_std)
    bb_pct = float((ltp - float(blo.iloc[-1])) / (float(bup.iloc[-1]) - float(blo.iloc[-1]) + 1e-9))
    # Bollinger %B percentile: what fraction of historical %B values was <= today's
    bb_pct_pctile   = _percentile_score(
        ((c.tail(252) - blo.tail(252)) / (bup.tail(252) - blo.tail(252) + 1e-9)).dropna().values,
        bb_pct)  # 0–1; high = price near upper band

    # ── Volume ──────────────────────────────────────────────────────────────
    vol_ma5  = float(v.tail(5).mean())
    vol_ma20 = float(v.tail(20).mean())
    _vol_data_valid = vol_ma20 >= 100
    vol_ratio = (vol_ma5 / vol_ma20) if _vol_data_valid else 1.0
    # Volume percentile within 1-year rolling volume (no fixed 1.0× or 0.5 scale)
    if _vol_data_valid and len(v) >= 20:
        # Compute rolling 5-day avg volumes over history
        vol_roll5 = v.rolling(5).mean().dropna()
        vol_pct   = _percentile_score(vol_roll5.tail(252).values, vol_ma5)  # 0–1
    else:
        vol_pct = 0.5   # no data → neutral

    # ── 5-day return z-score ────────────────────────────────────────────────
    if len(c) >= 6:
        ret5_series = (c / c.shift(5) - 1).dropna() * 100
        ret5_now    = float((ltp / float(c.iloc[-6]) - 1) * 100) if float(c.iloc[-6]) != 0 else 0
        ret5_z      = _zscore_clamp(ret5_series.tail(252).values, ret5_now, clamp=3.0) / 3.0
    else:
        ret5_z = 0.0; ret5_now = 0.0

    # ── Distance from 200 EMA → z-score (no fixed ATR scale) ────────────────
    dist200_series = ((c - e200) / (atr14 + 1e-9)).dropna()
    dist200_now    = (ltp - e200v) / (atrv + 1e-9)
    dist200_z      = _zscore_clamp(dist200_series.tail(252).values, dist200_now, clamp=3.0) / 3.0

    # ── ATR percentile (vol regime) ─────────────────────────────────────────
    atr_pct_pctile = _percentile_score(atr14.tail(252).dropna().values, atrv)  # 0–1

    factors = {}

    # ════════════════════════════════════════════════════════════════════════
    # FACTOR GROUP 1 — TREND  (weight 0.30)
    # Inputs: EMA structure (sign-based) + ADX percentile × direction
    # ════════════════════════════════════════════════════════════════════════
    # EMA structure score: count how many of 3 checks pass, normalise to [-1,+1]
    # Uses sign only (no arbitrary weights 2,4,6)
    ema_checks = [ltp > e20v, ltp > e50v, e20v > e50v, ltp > e200v, e50v > e200v]
    ema_pass   = sum(ema_checks)
    ema_score  = (ema_pass / len(ema_checks)) * 2 - 1  # → [-1, +1]

    # ADX contribution: strength × direction (percentile replaces fixed 25)
    adx_score  = adx_pct * adx_dir  # high-percentile ADX + direction → strong trend signal

    trend_score = _calib("trend_ema_vs_adx") * ema_score + (1.0 - _calib("trend_ema_vs_adx")) * adx_score
    trend_score = max(-1.0, min(1.0, trend_score))
    # Record for calibration
    _record("_calib_ema_score_hist", ema_score)
    _record("_calib_adx_score_hist", adx_score)
    factors["TREND (EMA+ADX)"] = round(trend_score * 30, 1)  # display as ±30

    # ════════════════════════════════════════════════════════════════════════
    # FACTOR GROUP 2 — MOMENTUM  (weight 0.25)
    # Inputs: RSI z-score, 5-day return z-score
    # Both already in [-1,+1] via z/3 normalisation
    # ════════════════════════════════════════════════════════════════════════
    momentum_score = _calib("momentum_rsi_vs_ret5") * rsi_z + (1.0 - _calib("momentum_rsi_vs_ret5")) * ret5_z
    momentum_score = max(-1.0, min(1.0, momentum_score))
    factors["MOMENTUM (RSI+5D)"] = round(momentum_score * 25, 1)  # display as ±25

    # ════════════════════════════════════════════════════════════════════════
    # FACTOR GROUP 3 — VOLATILITY REGIME  (weight 0.15)
    # Low ATR percentile = mean-revert environment → slight bullish
    # High BB %B percentile = near upper band → slight bullish (momentum)
    # Combine as regime signal — high ATR = uncertainty = slight bearish weight
    # ════════════════════════════════════════════════════════════════════════
    # bb_pct_pctile high → price near upper band; contextually bullish in trending
    # Paired with ema_score to determine direction
    bb_regime  = (bb_pct_pctile * 2 - 1) * trend_score  # aligned with trend direction
    atr_regime = 1 - 2 * atr_pct_pctile  # high ATR pctile → -1 (noise/caution signal)
    vol_regime_score = _calib("vol_bb_vs_atr") * bb_regime + (1.0 - _calib("vol_bb_vs_atr")) * atr_regime
    vol_regime_score = max(-1.0, min(1.0, vol_regime_score))
    factors["VOL REGIME (ATR+BB)"] = round(vol_regime_score * 15, 1)  # display as ±15

    # ════════════════════════════════════════════════════════════════════════
    # FACTOR GROUP 4 — POSITIONING  (weight 0.20)
    # Inputs: PCR percentile, OI skew, max pain distance / expected_move
    # ════════════════════════════════════════════════════════════════════════
    positioning_score = 0.0
    if chain_df is not None and not chain_df.empty:
        total_ce = float(chain_df["CE_OI"].sum())
        total_pe = float(chain_df["PE_OI"].sum())
        pcr_val  = total_pe / (total_ce + 1e-9)

        # PCR percentile within full chain (not fixed 1.0 centre)
        all_pcr = chain_df["PCR"].replace([np.inf, -np.inf], np.nan).dropna().values
        pcr_pctile = _percentile_score(all_pcr, pcr_val)  # 0–1; high = bullish
        pcr_s      = 2 * pcr_pctile - 1  # → [-1, +1]

        # OI skew percentile: (put OI below spot vs call OI above spot)
        above_oi = chain_df[chain_df.Strike > ltp]["CE_OI"].sum()
        below_oi = chain_df[chain_df.Strike < ltp]["PE_OI"].sum()
        oi_skew_val = (below_oi - above_oi) / (total_ce + total_pe + 1e-9)
        # Convert raw skew to z-score via session history
        # BUG FIX (OI Skew Double-Append): appending here AND in compute_probabilistic_score
        # doubled every observation (2× history inflation). Fix: load-id dedup guard — same
        # pattern as _flow_oi_load_ids. One entry per Load, whichever function runs first.
        _skew_lids = st.session_state.get("_flow_skew_oi_load_ids", [])
        _skew_lid  = st.session_state.get("opt_load_id", 0)
        skew_hist  = st.session_state.get("_flow_skew_oi_hist", [])
        if _skew_lids and _skew_lids[-1] == _skew_lid:
            skew_hist[-1] = float(oi_skew_val)   # overwrite same Load entry
        else:
            skew_hist.append(float(oi_skew_val))
            _skew_lids.append(_skew_lid)
        if len(skew_hist) > 30: skew_hist = skew_hist[-30:]; _skew_lids = _skew_lids[-30:]
        st.session_state["_flow_skew_oi_hist"]     = skew_hist
        st.session_state["_flow_skew_oi_load_ids"] = _skew_lids
        # FIX: use tanh of raw value when history < 3 (z-score would be 0 with 1 sample)
        if len(skew_hist) >= 3:
            oi_skew_z = _zscore_clamp(skew_hist, float(oi_skew_val), clamp=2.0) / 2.0
        else:
            # Raw skew ∈ [-1,+1] already (it's a ratio). Amplify via tanh for signal.
            oi_skew_z = math.tanh(oi_skew_val * 3.0)

        # Max pain proximity: distance normalised by expected move (not % of spot)
        oi_d = st.session_state.get("opt_oi", {})
        mp   = oi_d.get("max_pain", ltp)
        # Expected move fallback: use ATR from OHLCV if available, else 1% of price
        _ohlcv_em = st.session_state.get("opt_ohlcv_df", None)
        if _ohlcv_em is not None and not _ohlcv_em.empty and len(_ohlcv_em) >= 5:
            _c_em = _ohlcv_em["close"].astype(float)
            _h_em = _ohlcv_em["high"].astype(float)
            _l_em = _ohlcv_em["low"].astype(float)
            _tr_em = pd.concat([_h_em - _l_em,
                                 (_h_em - _c_em.shift(1)).abs(),
                                 (_l_em - _c_em.shift(1)).abs()], axis=1).max(axis=1)
            _atr_em = float(_tr_em.tail(14).mean())
        else:
            _atr_em = _atr_seed(ltp)
        em_price = oi_d.get("atm_straddle", _atr_em)
        if em_price > 0:
            mp_dist_em = (ltp - mp) / (em_price + 1e-9)  # in units of expected move
            mp_s = max(-1.0, min(1.0, -mp_dist_em * _calib("mp_gravity")))
        else:
            mp_s = 0.0

        _pw = _calib_vec("positioning_pcr_vs_oi_vs_mp")
        positioning_score = _pw[0] * pcr_s + _pw[1] * oi_skew_z + _pw[2] * mp_s
        positioning_score = max(-1.0, min(1.0, positioning_score))

    factors["POSITIONING (OI+PCR)"] = round(positioning_score * 20, 1)  # display as ±20

    # ════════════════════════════════════════════════════════════════════════
    # FACTOR GROUP 5 — RELATIVE STRENGTH vs NIFTY  (weight 0.10)
    # RS ratio z-score from session history (no fixed 1.02 threshold)
    # ════════════════════════════════════════════════════════════════════════
    rs_factor = 0.0
    rs_data   = st.session_state.get("opt_rs_nifty", None)
    if rs_data and isinstance(rs_data, dict):
        rs_series_hist = rs_data.get("rs_series", [])
        rs_ratio       = rs_data.get("rs_ratio", 1.0)
        rs_z = _zscore_clamp(rs_series_hist, rs_ratio, clamp=2.0) / 2.0  # → [-1,+1]
        rs_factor = rs_z
    factors["REL STRENGTH (vs Nifty)"] = round(rs_factor * 10, 1)  # display as ±10

    # ════════════════════════════════════════════════════════════════════════
    # WEIGHTED FINAL SCORE — aligned with CFG["factor_weights"] key names
    # New key structure: flow, positioning, vol_regime, rel_strength, trend
    #
    # directional_bias computes trend + momentum + vol_regime separately.
    # Map them onto new keys:
    #   trend    → blend of trend_score (60%) + momentum_score (40%)
    #              Momentum is now a minor confirming input inside trend, not a
    #              separate primary factor (matches 1-5 day signal model design).
    #   vol_regime → vol_regime_score (ATR/BB regime signal)
    #   positioning → positioning_score (PCR + OI skew + max pain)
    #   rel_strength → rs_factor
    #   flow → from compute_flow_scores; computed after this block and stored in
    #           bias_res["flow"]. directional_bias does not weight flow here —
    #           flow is applied in compute_probabilistic_score where it has 0.30 weight.
    #           Here we use 0.0 for flow since it hasn't been computed yet.
    # ════════════════════════════════════════════════════════════════════════
    # Use regime-adaptive weights, but normalise over only the 4 factors
    # computed here (flow is handled separately in compute_probabilistic_score).
    _raw_fw = st.session_state.get("opt_regime_weights", CFG["factor_weights"])
    _db_keys = ["trend", "vol_regime", "positioning", "rel_strength"]
    _db_total = sum(_raw_fw.get(k, 0.0) for k in _db_keys) or 1.0
    fw = {k: _raw_fw.get(k, 0.0) / _db_total for k in _db_keys}

    # Blend trend + momentum → single "trend" contribution
    # Momentum (RSI z-score, 5D return) shrinks to minor role inside trend
    _ct_w = _calib("trend_combined_trend_vs_momentum")
    combined_trend = _ct_w * trend_score + (1.0 - _ct_w) * momentum_score
    combined_trend = max(-1.0, min(1.0, combined_trend))

    raw_score = (fw["trend"]        * combined_trend
               + fw["vol_regime"]   * vol_regime_score
               + fw["positioning"]  * positioning_score
               + fw["rel_strength"] * rs_factor
               + 0.0)                # flow: computed separately in compute_probabilistic_score
    raw_score = max(-1.0, min(1.0, raw_score))
    # Scale to ±100 for display and threshold compatibility
    score_100 = int(round(raw_score * 100))

    # Bias thresholds: 22/9 on ±100 scale (equivalent to old 30/12 on old ±90 range)
    bias = ("STRONGLY BULLISH" if score_100 >= 30 else "BULLISH"   if score_100 >= 12 else
            "NEUTRAL"          if score_100 >  -12 else "BEARISH"  if score_100 >= -30 else "STRONGLY BEARISH")

    # Flow scores (delta-based, also adaptive)
    flow = compute_flow_scores(chain_df, df)

    return {
        "bias": bias, "score": score_100,
        "rsi": round(rsi, 1), "macd_hist": round(macd_h, 3),
        "bb_pct": round(bb_pct * 100, 1),
        "vol_ratio": round(vol_ratio, 2),
        "atr_pct":   round(atrv / ltp * 100, 2) if ltp > 0 else 0,
        "e9": round(e9v, 2), "e20": round(e20v, 2), "e50": round(e50v, 2),
        "atr": round(atrv, 2), "factors": factors, "flow": flow,
        "adx": round(adx_val, 1),
        # Z-scores exposed for downstream use
        "rsi_z": round(rsi_z, 3), "macd_z": round(macd_z, 3),
        "ret5_z": round(ret5_z, 3), "dist200_z": round(dist200_z, 3),
        "adx_pct": round(adx_pct, 3), "vol_pct": round(vol_pct, 3),
    }

# ============================================================
# VOLATILITY REGIME
# ============================================================

def iv_rank(iv_series, current_iv):
    s = pd.Series(iv_series).dropna()
    if len(s) < 3: return 50.0
    lo, hi = s.min(), s.max()
    # When range is negligible (all IVs identical), return 50 — we have no useful information
    if (hi - lo) < 1e-6:
        return 50.0
    return round((current_iv - lo) / (hi - lo) * 100, 1)

def vol_regime(ivr):
    """Classify vol regime from IV Rank (0–100 percentile).
    Thresholds from CFG: iv_hv_pct_sell (sell premium) and iv_hv_pct_buy (buy premium).
    Midpoints between those are normal-high / normal-low.
    """
    _sell = _adaptive_threshold("iv_hv", CFG["iv_hv_pct_sell"], percentile=70.0)
    _buy  = _adaptive_threshold("iv_hv", CFG["iv_hv_pct_buy"],  percentile=30.0)
    _mid  = (_sell + _buy) / 2     # e.g. 52.5
    _very_hi = _sell + (100 - _sell) * 0.6  # 60% of way to 100 → extreme

    if   ivr >= _very_hi: return "HIGH VOL",     "SELL premium — iron condors / strangles / short straddle","#ff3b3b"
    elif ivr >= _sell:    return "ELEVATED",      "Lean SELL — credit spreads / iron condor",               "#ff8c00"
    elif ivr >= _mid:     return "NORMAL-HIGH",   "Slight sell lean — balanced spreads, light credits",     "#ffb347"
    elif ivr >= _buy:     return "NORMAL-LOW",    "Slight buy lean — calendars / ratio spreads",            "#7ec8e3"
    else:                 return "LOW VOL",       "BUY premium — debit spreads / long options / straddles", "#1e90ff"


# ============================================================
# CROSS-ASSET SIGNALS
# ============================================================

@st.cache_data(ttl=900, show_spinner=False)
def fetch_cross_asset_signals() -> dict:
    """
    Fetch US 10Y yield, DXY, Crude, VIX, India VIX and normalise to [-1,+1].
    Composite weighted by correlation with forward returns (learned dynamically).
    Returns dict with individual signals + composite + availability flag.
    """
    _result = {
        "rate_signal": 0.0, "dollar_signal": 0.0, "oil_signal": 0.0,
        "vix_signal": 0.0, "indiavix_signal": 0.0,
        "composite": 0.0, "data_available": False, "zscores": {},
    }
    try:
        _raw = yf.download(
            ["^TNX", "DX-Y.NYB", "CL=F", "^VIX", "^INDIAVIX"],
            period="60d", interval="1d",
            progress=False, auto_adjust=True, group_by="ticker"
        )
        if _raw.empty:
            return _result

        def _get_ca_close(ticker):
            try:
                return (_raw[ticker]["Close"].dropna()
                        if isinstance(_raw.columns, pd.MultiIndex)
                        else _raw["Close"].dropna())
            except Exception:
                return pd.Series(dtype=float)

        def _norm(series, window=20):
            s = series.dropna()
            if len(s) < 5:
                return 0.0
            pct_now = float((s.iloc[-1] - s.iloc[-2]) / (abs(s.iloc[-2]) + 1e-9))
            hist    = s.pct_change().dropna().tail(window).values
            if len(hist) < 3:
                return float(math.tanh(pct_now * 20))
            mu = float(hist.mean()); sd = max(float(hist.std()), abs(mu) * 0.1 + 1e-6)
            return float(max(-1.0, min(1.0, (pct_now - mu) / (2.0 * sd))))

        # Invert: rising rates/dollar/VIX = bearish for India equity
        sigs = {
            "rate":     -_norm(_get_ca_close("^TNX")),
            "dollar":   -_norm(_get_ca_close("DX-Y.NYB")),
            "oil":      -_norm(_get_ca_close("CL=F")) * 0.5,
            "vix":      -_norm(_get_ca_close("^VIX")),
            "indiavix": -_norm(_get_ca_close("^INDIAVIX")),
        }

        _ca_w = st.session_state.get("_cross_asset_weights",
                    {k: 0.20 for k in sigs})
        total_w = sum(_ca_w.values()) or 1.0
        composite = max(-1.0, min(1.0,
            sum(_ca_w.get(k, 0.20) / total_w * v for k, v in sigs.items())))

        _result.update({
            "rate_signal":     round(sigs["rate"],     4),
            "dollar_signal":   round(sigs["dollar"],   4),
            "oil_signal":      round(sigs["oil"],      4),
            "vix_signal":      round(sigs["vix"],      4),
            "indiavix_signal": round(sigs["indiavix"], 4),
            "composite":       round(composite, 4),
            "data_available":  True,
            "zscores":         {k: round(v, 4) for k, v in sigs.items()},
        })
    except Exception:
        pass
    return _result


def _update_cross_asset_weights():
    """Update cross-asset signal weights from correlation with forward returns."""
    try:
        sym    = st.session_state.get("opt_symbol", "").upper()
        ret_k  = f"{sym}:_calib_realised_ret_hist" if sym else "_calib_realised_ret_hist"
        r_hist = np.array(st.session_state.get(ret_k,
                     st.session_state.get("_calib_realised_ret_hist", [])), dtype=float)
        if len(r_hist) < 10:
            return
        _ca_hist = st.session_state.get("_cross_asset_signal_hist", {})
        new_w = {}
        for k in ["rate", "dollar", "oil", "vix", "indiavix"]:
            hist = np.array(_ca_hist.get(k, []), dtype=float)
            n    = min(len(hist), len(r_hist))
            if n < 10:
                new_w[k] = 0.20; continue
            corr = float(np.corrcoef(hist[-n:], r_hist[-n:])[0, 1])
            new_w[k] = max(0.01, abs(corr) if not math.isnan(corr) else 0.10)
        total = sum(new_w.values()) or 1.0
        st.session_state["_cross_asset_weights"] = {k: v / total for k, v in new_w.items()}
    except Exception:
        pass

# ============================================================
# MARKET REGIME DETECTION
# ============================================================

# NOTE (FIX 6): The canonical strategy_prob_profit (with Merton jump-diffusion)
# is defined above (after _fit_jump_params). This duplicate definition has been
# removed to prevent the earlier correct version from being shadowed.




@st.cache_data(ttl=3600)
def load_fno_master():
    try:
        url = "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz"
        r   = requests.get(url, timeout=12)
        if r.status_code == 200:
            with gzip.GzipFile(fileobj=io.BytesIO(r.content)) as gz:
                return pd.DataFrame(json.load(gz))
    except Exception as _e:
        st.warning(f"Failed to load F&O master from Upstox CDN: {_e}")
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
        elif r.status_code == 401:
            st.error("🔑 Upstox token expired. Please refresh your access token in the sidebar.")
    except Exception as _e:
        st.warning(f"fetch_expiries failed: {_e}")
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
            data = r.json().get("data", [])
            if not data:
                st.warning("⚠️ Upstox returned an empty option chain. Market may be closed or instrument key is wrong.")
            return data
        elif r.status_code == 401:
            st.error("🔑 Upstox token is expired or invalid. Please paste a fresh access token in the sidebar.")
        elif r.status_code == 429:
            st.warning("⚠️ Upstox API rate limit hit. Wait a few seconds and try again.")
        else:
            st.warning(f"Chain API error {r.status_code}: {r.text[:200]}")
    except requests.exceptions.Timeout:
        st.warning("⚠️ Upstox API timed out. Check your internet connection and retry.")
    except Exception as e:
        st.warning(f"Chain fetch failed: {e}")
    return []

def fetch_spot_quote(instrument_key):
    """Fetch live spot price from Upstox market-quote."""
    url    = "https://api.upstox.com/v2/market-quote/quotes"
    params = {"instrument_key": instrument_key}
    try:
        # FIX: use get_headers() so token is always current, not frozen at load time
        r = requests.get(url, headers=get_headers(), params=params, timeout=10)
        if r.status_code == 200:
            data = r.json().get("data", {})
            for v in data.values():
                lp = v.get("last_price")
                if lp: return float(lp)
        elif r.status_code == 401:
            st.warning("⚠️ Upstox token expired or invalid. Please paste a fresh token in the sidebar.")
    except Exception as e:
        st.warning(f"Spot quote fetch failed: {e}")
    return None

def parse_chain(raw_data, spot, step=50):
    """Parse Upstox option chain into a clean DataFrame.

    Upstox v2 /option/chain response structure (each item):
      {
        "strike_price": 22600,
        "call_options": {
          "instrument_key": "...",
          "market_data": {
            "ltp": 232.45, "volume": 9981560,
            "oi": 1864005, "bid_price": 232.4, "ask_price": 232.5,
            "prev_oi": ..., "bid_qty": ..., "ask_qty": ...
          },
          "option_greeks": {
            "vega": ..., "theta": ..., "gamma": ..., "delta": ...,
            "iv": 40.45          ← IV lives HERE, not in market_data
          }
        },
        "put_options": { ... same structure ... }
      }

    IV is in option_greeks.iv (percent form, e.g. 40.45 = 40.45% annualised).
    Fallback: compute IV from LTP via Brenner-Subrahmanyam when option_greeks missing.
    """
    rows = []
    if not raw_data: return pd.DataFrame()
    items = raw_data if isinstance(raw_data, list) else raw_data.get("options", [])

    # ── Helpers defined ONCE outside the loop ────────────────────────────────
    def _md(d, *keys):
        """Get a value from top-level dict OR its nested market_data sub-dict.
        Returns the first key found (including 0). Returns 0 only if no key exists.
        """
        for k in keys:
            if k in d:
                return d[k]
        md = d.get("market_data", {}) or {}
        for k in keys:
            if k in md:
                return md[k]
        return 0

    def _iv_from(d):
        """Extract IV % from option_greeks.iv (canonical Upstox v2 location).
        Falls back to market_data and top-level. Returns raw percent (e.g. 43.4).
        """
        og = d.get("option_greeks", {}) or {}
        for k in ("iv", "implied_volatility"):
            v = og.get(k)
            if v is not None and float(v) > 0:
                return float(v)
        md = d.get("market_data", {}) or {}
        for src in (d, md):
            for k in ("implied_volatility", "iv"):
                v = src.get(k)
                if v is not None and float(v) > 0:
                    return float(v)
        return 0.0

    for item in items:
        strike = float(item.get("strike_price") or item.get("strike") or 0)
        if not strike: continue

        ce = item.get("call_options", {}) or {}
        pe = item.get("put_options",  {}) or {}

        ce_ltp = float(_md(ce, "ltp", "last_price") or 0)
        pe_ltp = float(_md(pe, "ltp", "last_price") or 0)
        ce_iv  = _iv_from(ce)
        pe_iv  = _iv_from(pe)

        ce_oi      = float(_md(ce, "oi", "open_interest") or 0)
        pe_oi      = float(_md(pe, "oi", "open_interest") or 0)

        # ΔOI from API: use prev_oi if available in market_data, else 0.
        # Session-state tracking between loads will fill this in on subsequent clicks.
        _ce_prev   = _md(ce, "prev_oi")
        _pe_prev   = _md(pe, "prev_oi")
        ce_oic     = (ce_oi - float(_ce_prev)) if _ce_prev else 0.0
        pe_oic     = (pe_oi - float(_pe_prev)) if _pe_prev else 0.0

        rows.append({
            "Strike":  strike,
            "CE_LTP":  ce_ltp,
            "CE_OI":   ce_oi,
            "CE_OIC":  ce_oic,
            "CE_Vol":  float(_md(ce, "volume", "vol") or 0),
            "CE_IV":   ce_iv,
            "CE_Bid":  float(_md(ce, "bid_price", "bid") or 0),
            "CE_Ask":  float(_md(ce, "ask_price", "ask") or 0),
            "PE_LTP":  pe_ltp,
            "PE_OI":   pe_oi,
            "PE_OIC":  pe_oic,
            "PE_Vol":  float(_md(pe, "volume", "vol") or 0),
            "PE_IV":   pe_iv,
            "PE_Bid":  float(_md(pe, "bid_price", "bid") or 0),
            "PE_Ask":  float(_md(pe, "ask_price", "ask") or 0),
        })
    df = pd.DataFrame(rows).sort_values("Strike").reset_index(drop=True)
    if not df.empty:
        df["PCR"]       = df.apply(lambda r: round(r.PE_OI / (r.CE_OI + 1e-9), 3), axis=1)
        df["OI_Diff"]   = df["CE_OI"] - df["PE_OI"]
        # Moneyness relative to CALLS (standard chain convention):
        # strike < spot → ITM for CE, OTM for PE
        # strike > spot → OTM for CE, ITM for PE
        # Label uses call perspective: "ITM" means ITM-for-calls (strike below spot)
        df["Moneyness"] = df["Strike"].apply(
            lambda k: "ATM"    if abs(k - spot) <= 0.5 * step
                      else ("ITM" if k < spot else "OTM"))
    return df

# ============================================================
# HISTORICAL & INTRADAY DATA — Upstox primary, yfinance fallback
# ============================================================

YF_TICKERS = {
    "NIFTY":      "^NSEI",
    "BANKNIFTY":  "^NSEBANK",
    "FINNIFTY":   "^CNXFIN",
    "MIDCPNIFTY": "^CNXMIDCAP",
    "SENSEX":     "^BSESN",
}

# Upstox instrument key → exchange segment mapping for historical API
_UPSTOX_SEGMENT = {
    "INDEX": "NSE_INDEX",
    "EQ":    "NSE_EQ",
}

def _upstox_instrument_key_for_ohlcv(symbol: str, master_df) -> str | None:
    """Return the instrument_key suitable for the Upstox historical candles API."""
    if master_df is None or master_df.empty:
        return None
    sym = symbol.upper().strip()
    for itype in ["INDEX", "EQ"]:
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
    return None


@st.cache_data(ttl=300, show_spinner=False)
def fetch_upstox_candles(_token: str, instrument_key: str,
                          interval: str = "day", days: int = 365) -> pd.DataFrame:
    """Fetch OHLCV candles from Upstox historical data API.

    Endpoint: GET /v2/historical-candle/{instrument_key}/{interval}/{to_date}/{from_date}
    interval: 'day' | '30minute' | '15minute' | '5minute' | '1minute'
    Returns DataFrame with columns: date, open, high, low, close, volume.
    Empty DataFrame on any error.
    """
    if not _token or not instrument_key:
        return pd.DataFrame()
    try:
        to_dt   = datetime.now().date()
        fr_dt   = to_dt - timedelta(days=days)
        to_str  = to_dt.isoformat()
        fr_str  = fr_dt.isoformat()
        url     = (f"https://api.upstox.com/v2/historical-candle"
                   f"/{urllib.parse.quote(instrument_key, safe='')}"
                   f"/{interval}/{to_str}/{fr_str}")
        hdrs    = {"Authorization": f"Bearer {_token}", "Accept": "application/json"}
        r       = requests.get(url, headers=hdrs, timeout=15)
        if r.status_code != 200:
            st.warning(f"⚠️ OHLCV fetch failed (status {r.status_code}) for {instrument_key[:40]}. "
                       f"Falling back to yfinance. EMA/HV may lag by 1 day.")
            return pd.DataFrame()
        candles = r.json().get("data", {}).get("candles", [])
        if not candles:
            st.warning(f"⚠️ Upstox returned empty OHLCV candles for {instrument_key[:40]}. "
                       f"Falling back to yfinance.")
            return pd.DataFrame()
        # Each candle: [timestamp, open, high, low, close, volume, oi]
        rows = []
        for c in candles:
            try:
                rows.append({
                    "date":   pd.to_datetime(c[0]).date(),
                    "open":   float(c[1]),
                    "high":   float(c[2]),
                    "low":    float(c[3]),
                    "close":  float(c[4]),
                    "volume": float(c[5]),
                })
            except Exception:
                continue
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
        # Validate: last candle should be recent (within 5 trading days)
        last_date = df["date"].iloc[-1]
        days_ago  = (datetime.now().date() - last_date).days
        if days_ago > 5:
            st.warning(f"⚠️ Upstox OHLCV last candle is {days_ago} days old ({last_date}). "
                       f"Data may be stale — EMAs/HV will lag current price.")
        return df
    except Exception as e:
        st.warning(f"⚠️ fetch_upstox_candles failed: {e}. Falling back to yfinance.")
        return pd.DataFrame()


@st.cache_data(ttl=60, show_spinner=False)
def fetch_upstox_intraday_candles(_token: str, instrument_key: str,
                                   interval: str = "5minute") -> pd.DataFrame:
    """Fetch today's intraday OHLCV candles from Upstox intraday API.

    Endpoint: GET /v2/historical-candle/intraday/{instrument_key}/{interval}
    Returns DataFrame: datetime, open, high, low, close, volume.
    Cached 60 seconds — refreshes every minute during market hours.
    """
    if not _token or not instrument_key:
        return pd.DataFrame()
    try:
        url  = (f"https://api.upstox.com/v2/historical-candle/intraday"
                f"/{urllib.parse.quote(instrument_key, safe='')}/{interval}")
        hdrs = {"Authorization": f"Bearer {_token}", "Accept": "application/json"}
        r    = requests.get(url, headers=hdrs, timeout=10)
        if r.status_code != 200:
            return pd.DataFrame()
        candles = r.json().get("data", {}).get("candles", [])
        if not candles:
            return pd.DataFrame()
        rows = []
        for c in candles:
            try:
                rows.append({
                    "datetime": pd.to_datetime(c[0]),
                    "open":   float(c[1]),
                    "high":   float(c[2]),
                    "low":    float(c[3]),
                    "close":  float(c[4]),
                    "volume": float(c[5]),
                })
            except Exception:
                continue
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows).sort_values("datetime").reset_index(drop=True)
        return df
    except Exception:
        return pd.DataFrame()


def _clear_api_caches():
    """Clear all Upstox @st.cache_data caches.
    Must be called AFTER all cached functions are defined (not at module top-level
    where the functions don't exist yet).  Called once per session whenever the
    access token changes (flag set by the sidebar token widget).
    """
    fetch_option_chain.clear()
    fetch_expiries.clear()
    fetch_upstox_candles.clear()
    fetch_upstox_intraday_candles.clear()

# Execute any pending cache clear that was requested by the sidebar token widget
# (which runs before the cached functions were defined).
if st.session_state.pop("_pending_cache_clear", False):
    _clear_api_caches()


def get_ohlcv(symbol: str, token: str, master_df=None) -> pd.DataFrame:
    """Get daily OHLCV. Tries Upstox historical API first, then yfinance.
    Returns DataFrame with columns: date, open, high, low, close, volume.
    1 year of daily data for volatility, drift, and calibration computation.
    """
    # ── Primary: Upstox historical candles ──────────────────────────────────
    if token:
        _mdf = master_df if master_df is not None else load_fno_master()
        ikey = _upstox_instrument_key_for_ohlcv(symbol, _mdf)
        if ikey:
            df_up = fetch_upstox_candles(token, ikey, interval="day", days=365)
            if not df_up.empty and len(df_up) >= 20:
                return df_up

    # ── Fallback: yfinance ───────────────────────────────────────────────────
    yftick = YF_TICKERS.get(symbol.upper(), f"{symbol.upper()}.NS")
    try:
        d = yf.download(yftick, period="1y", interval="1d", progress=False, auto_adjust=True)
        if not d.empty:
            d = d.copy()
            # Handle MultiIndex columns from yfinance 0.2+ (e.g. ('Close','NSEI'))
            if isinstance(d.columns, pd.MultiIndex):
                # Drop ticker level — keep only the price-type level
                d.columns = [c[0].lower() if isinstance(c, tuple) else str(c).lower()
                             for c in d.columns]
            else:
                d.columns = [str(c).lower() for c in d.columns]
            d = d.reset_index()
            # After reset_index, date is in 'date' or 'datetime' or 'index' depending on version
            d.columns = [str(c).lower() for c in d.columns]
            # Normalise date column name to 'date'
            for _dc in ("datetime", "index", "date"):
                if _dc in d.columns and _dc != "date":
                    d = d.rename(columns={_dc: "date"})
                    break
            # Ensure required columns exist
            if "close" not in d.columns:
                return pd.DataFrame()
            # FIX: yfinance lags by 1 day during market hours — no today's candle yet.
            # Append a synthetic today row using live spot so EMAs/HV are current.
            _live_spot = st.session_state.get("opt_spot", 0.0)
            _today     = datetime.now().date()
            if "date" in d.columns:
                _last_d = pd.to_datetime(d["date"].iloc[-1]).date()
            else:
                _last_d = _today
            if _live_spot > 0 and _last_d < _today:
                _today_row = pd.DataFrame([{
                    "date": _today, "open": _live_spot, "high": _live_spot,
                    "low": _live_spot, "close": _live_spot, "volume": 0.0,
                }])
                d = pd.concat([d, _today_row], ignore_index=True)
            return d
    except Exception as e:
        st.warning(f"⚠️ yfinance OHLCV failed for {symbol} ({yftick}): {e}")
    return pd.DataFrame()


def get_intraday_ohlcv(symbol: str, token: str,
                        interval: str = "5minute",
                        master_df=None) -> pd.DataFrame:
    """Get today's intraday OHLCV candles from Upstox.
    Returns DataFrame with columns: datetime, open, high, low, close, volume.
    Returns empty DataFrame outside market hours or on API error.
    """
    if not token:
        return pd.DataFrame()
    _mdf = master_df if master_df is not None else load_fno_master()
    ikey = _upstox_instrument_key_for_ohlcv(symbol, _mdf)
    if not ikey:
        return pd.DataFrame()
    return fetch_upstox_intraday_candles(token, ikey, interval=interval)


# ============================================================
# INTRADAY SIGNAL COMPUTATION
# ============================================================

def compute_intraday_signals(intraday_df: pd.DataFrame,
                              chain_df: pd.DataFrame,
                              spot: float) -> dict:
    """Derive intraday behavioural signals from live 5-minute candles + chain.

    Signals computed:
      opening_momentum  — first 30-min directional move (opening auction bias)
      vwap_position     — spot vs VWAP (above = bullish intraday flow)
      intraday_range_pct— today's range as % of spot (realised volatility so far)
      volume_acceleration — current 30-min volume vs session average (flow surge detector)
      oi_build_direction  — OI change sign: CE OI rising faster than PE = bearish flow
      price_structure     — higher highs / lower lows intraday (momentum confirmation)
      lunch_reversal      — post-12:30 reversal vs opening direction

    Returns dict with each signal ∈ [-1, +1] and composite intraday_score.
    """
    result = {
        "opening_momentum":    0.0,
        "vwap_position":       0.0,
        "intraday_range_pct":  0.0,
        "volume_acceleration": 0.0,
        "oi_build_direction":  0.0,
        "price_structure":     0.0,
        "lunch_reversal":      0.0,
        "intraday_score":      0.0,
        "intraday_available":  False,
        "vwap":                spot,
        "session_high":        spot,
        "session_low":         spot,
        "candles_so_far":      0,
    }

    if intraday_df is None or intraday_df.empty or len(intraday_df) < CFG["intra_min_candles"]:
        return result

    df = intraday_df.copy()
    result["intraday_available"] = True
    result["candles_so_far"] = len(df)

    # ── VWAP — volume-weighted average price ─────────────────────────────────
    df["typical"] = (df["high"] + df["low"] + df["close"]) / 3.0
    total_vol = float(df["volume"].sum())
    if total_vol > 0:
        vwap = float((df["typical"] * df["volume"]).sum() / total_vol)
    else:
        vwap = float(df["close"].mean())
    result["vwap"] = round(vwap, 2)

    session_high = float(df["high"].max())
    session_low  = float(df["low"].min())
    result["session_high"] = round(session_high, 2)
    result["session_low"]  = round(session_low, 2)
    intra_range  = session_high - session_low
    if intra_range > 0:
        vwap_pos = (spot - vwap) / (intra_range / 2.0 + 1e-9)
        result["vwap_position"] = round(max(-1.0, min(1.0, vwap_pos)), 4)

    # ── Opening momentum — configurable window from CFG ───────────────────────
    _oc = CFG["intra_opening_candles"]
    opening_candles = df.head(_oc)
    if len(opening_candles) >= 2:
        open_price  = float(opening_candles.iloc[0]["open"])
        close_open  = float(opening_candles.iloc[-1]["close"])
        if open_price > 0 and intra_range > 0:
            open_move     = (close_open - open_price) / open_price
            open_momentum = open_move / (intra_range / spot + 1e-9)
            result["opening_momentum"] = round(max(-1.0, min(1.0, open_momentum)), 4)

    # ── Intraday range % ─────────────────────────────────────────────────────
    result["intraday_range_pct"] = round(intra_range / spot * 100, 3) if spot > 0 else 0.0

    # ── Volume acceleration — configurable recent window ─────────────────────
    _rc = CFG["intra_recent_candles"]
    _mv = CFG["intra_min_candles_vol"]
    session_avg_vol = float(df["volume"].mean())
    if session_avg_vol > 0 and len(df) >= _mv:
        recent_vol  = float(df.tail(_rc)["volume"].mean())
        vol_accel   = (recent_vol - session_avg_vol) / (session_avg_vol + 1e-9)
        result["volume_acceleration"] = round(max(-1.0, min(1.0, vol_accel)), 4)

    # ── Price structure — configurable early/late window ─────────────────────
    _sc = CFG["intra_structure_candles"]
    _ms = CFG["intra_min_candles_struct"]
    if len(df) >= _ms:
        early_high = float(df.head(_sc)["high"].mean())
        late_high  = float(df.tail(_sc)["high"].mean())
        early_low  = float(df.head(_sc)["low"].mean())
        late_low   = float(df.tail(_sc)["low"].mean())
        if intra_range > 0:
            hh_score = (late_high - early_high) / (intra_range + 1e-9)
            ll_score = (early_low  - late_low)  / (intra_range + 1e-9)
            price_struct = (hh_score + ll_score) / 2.0
            result["price_structure"] = round(max(-1.0, min(1.0, price_struct)), 4)

    # ── Lunch-hour reversal — configurable candle indices from CFG ───────────
    _lm  = CFG["intra_min_candles_lunch"]
    _lcs = CFG["intra_lunch_candle_start"]   # pre-lunch candle index
    if len(df) >= _lm:
        prelunch_close  = float(df.iloc[_lcs]["close"])
        postlunch_close = float(df.tail(1)["close"].values[0])
        open_p          = float(df.iloc[0]["open"])
        if open_p > 0 and intra_range > 0:
            # BUG FIX (Lunch Reversal): signal = where price is going AFTER lunch.
            # postlunch_dir already encodes that direction. The old negation logic
            # flipped a bullish afternoon reversal into a bearish signal. Fix: use
            # postlunch_dir directly — no sign flip needed regardless of morning dir.
            postlunch_dir = (postlunch_close - prelunch_close) / (intra_range + 1e-9)
            result["lunch_reversal"] = round(max(-1.0, min(1.0, postlunch_dir)), 4)

    # ── OI build direction from chain ─────────────────────────────────────────
    # CE OI change vs PE OI change: net call build = bearish, net put build = bullish
    if chain_df is not None and not chain_df.empty:
        try:
            ce_oic = float(chain_df["CE_OIC"].sum())
            pe_oic = float(chain_df["PE_OIC"].sum())
            total_oic = abs(ce_oic) + abs(pe_oic)
            if total_oic > 0:
                # More put OI building = bullish (put sellers adding support)
                # More call OI building = bearish (call sellers capping upside)
                oi_dir = (pe_oic - ce_oic) / (total_oic + 1e-9)
                result["oi_build_direction"] = round(max(-1.0, min(1.0, oi_dir)), 4)
        except Exception:
            pass

    # ── Composite intraday score ──────────────────────────────────────────────
    # Weights derived from empirical importance for NSE intraday:
    # Opening momentum and VWAP position are strongest intraday signals.
    # OI build direction and volume acceleration are secondary.
    # Price structure and lunch reversal are tertiary.
    # All weights are calibrated in _run_calibration_cycle if history is available.
    _iw = {
        "opening_momentum":    _calib("intra_w_opening_momentum"),
        "vwap_position":       _calib("intra_w_vwap_position"),
        "volume_acceleration": _calib("intra_w_volume_acceleration"),
        "oi_build_direction":  _calib("intra_w_oi_build"),
        "price_structure":     _calib("intra_w_price_structure"),
        "lunch_reversal":      _calib("intra_w_lunch_reversal"),
    }
    total_w = sum(_iw.values())
    if total_w < 1e-9:
        total_w = 1.0

    intra_composite = sum(
        (_iw[k] / total_w) * result.get(k, 0.0)
        for k in _iw
    )
    result["intraday_score"] = round(max(-1.0, min(1.0, intra_composite)), 4)

    # Record signals for calibration
    for k in _iw:
        _record_if_load(f"_calib_intra_{k}_hist", result.get(k, 0.0))

    return result


# ============================================================
# MARKET REGIME DETECTION
# ============================================================

def detect_market_regime(ohlcv_df, atm_iv, hv20, cross_asset_signal=0.0):
    """
    Regime detection — runs FIRST in the signal pipeline.

    6 observable inputs, all distribution-derived:
      IV percentile, IV/HV percentile, HV acceleration z-score,
      ADX percentile x direction, term structure slope z-score,
      cross-asset composite signal.

    Confidence = 1 - normalised stddev of pillar vector
    (measures pillar agreement, not a fixed number).

    Outputs regime label + confidence + per-regime signal weights.
    """
    result = {
        "trend": "UNKNOWN", "range": "UNKNOWN", "vol": "UNKNOWN",
        "regime": _REGIME_TRANSITION, "adx": 0.0, "bb_width_pct": 0.0,
        "atr_pct": 0.0, "color": "#888",
        "iv_pct": 50.0, "adx_pct": 50.0, "hv_accel": 0.0,
        "gex_sign": 0, "regime_confidence": 0.0,
        "signal_weights": _REGIME_SIGNAL_WEIGHTS[_REGIME_TRANSITION],
        "pillars": {},
    }
    if ohlcv_df is None or ohlcv_df.empty or len(ohlcv_df) < 30:
        return result

    c = ohlcv_df["close"].astype(float)
    h = ohlcv_df["high"].astype(float)
    l = ohlcv_df["low"].astype(float)

    # P1: IV percentile
    _sym = st.session_state.get("opt_symbol", "")
    _iv_hist = st.session_state.get("opt_iv_history", {}).get(_sym, [])
    iv_pct   = _percentile_score(_iv_hist, atm_iv) * 100 if len(_iv_hist) >= 5 else 50.0
    iv_pillar = (iv_pct / 100.0) * 2 - 1

    # P2: IV/HV ratio percentile
    _hv_s    = hv20 if (hv20 and hv20 > 0.01) else CFG["hv_fallback"]
    _iv_hv_h = _get_hist("_calib_iv_hv_ratio_hist")
    iv_hv_r  = atm_iv / (_hv_s + 1e-9)
    iv_hv_p  = _percentile_score(_iv_hv_h, iv_hv_r) if len(_iv_hv_h) >= 5 else 0.5
    iv_hv_pillar = iv_hv_p * 2 - 1

    # P3: HV acceleration z-score
    hv5 = (float(np.log(c / c.shift(1)).dropna().tail(5).std() * np.sqrt(252))
           if len(c) >= 6 else _hv_s)
    hv_accel = (hv5 - _hv_s) / (_hv_s + 1e-9)
    hv_accel_pillar = float(np.tanh(hv_accel * _calib("hv_accel_stretch")))
    _record("_calib_hv_accel_raw_hist", hv_accel)
    _record("_calib_hva_pillar_hist",   hv_accel_pillar)

    # P4: ADX percentile x direction
    tr     = pd.concat([h-l, (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
    atr14  = tr.ewm(alpha=1/14, adjust=False).mean()
    up_m   = h - h.shift(1); dn_m = l.shift(1) - l
    dm_p   = pd.Series(np.where((up_m > dn_m) & (up_m > 0), up_m, 0.0), index=c.index)
    dm_m   = pd.Series(np.where((dn_m > up_m) & (dn_m > 0), dn_m, 0.0), index=c.index)
    di_p   = 100 * dm_p.ewm(alpha=1/14, adjust=False).mean() / (atr14 + 1e-9)
    di_m_s = 100 * dm_m.ewm(alpha=1/14, adjust=False).mean() / (atr14 + 1e-9)
    dx     = 100 * (di_p - di_m_s).abs() / (di_p + di_m_s + 1e-9)
    adx_s  = dx.ewm(alpha=1/14, adjust=False).mean().dropna()
    adx    = float(adx_s.iloc[-1])
    adx_dir = 1.0 if float(di_p.iloc[-1]) > float(di_m_s.iloc[-1]) else -1.0
    adx_pct_v = _percentile_score(adx_s.values, adx) * 100 if len(adx_s) >= 10 else 50.0
    adx_pillar = (adx_pct_v / 100.0) * adx_dir
    _record("_calib_adx_pillar_hist", adx_pillar)

    # P5: Term structure slope
    # Derive term slope from opt_multi_expiry (same source as compute_probabilistic_score)
    # opt_term_slope was never set, so we compute it directly here.
    _ts_data = st.session_state.get("opt_multi_expiry", [])
    if len(_ts_data) >= 2:
        _ts = float(_ts_data[1].get("atm_iv", atm_iv)) - float(_ts_data[0].get("atm_iv", atm_iv))
    else:
        _ts = 0.0
    _tsh = st.session_state.get("_calib_ts_slope_hist", [])
    ts_p = (-(_zscore_clamp(_tsh, _ts, clamp=2.0) / 2.0) if len(_tsh) >= 5
            else -math.tanh(_ts * _calib("ts_tanh_scale")))

    # P6: Cross-asset
    ca_p = max(-1.0, min(1.0, float(cross_asset_signal)))
    _record("_calib_cross_asset_hist", ca_p)

    # P7: GEX sign
    _oi_d   = st.session_state.get("opt_oi", {})
    net_gex = float(_oi_d.get("net_gex", 0) or 0)
    gex_s   = 1 if net_gex > 0 else (-1 if net_gex < 0 else 0)
    _gexh   = st.session_state.get("_flow_gex_hist", [net_gex] if net_gex != 0 else [0])
    if len(_gexh) >= 3 and net_gex != 0:
        gex_p = -gex_s * _percentile_score([abs(g) for g in _gexh], abs(net_gex))
    elif net_gex != 0:
        gex_p = -gex_s * 0.5
    else:
        gex_p = 0.0
    _record("_calib_iv_pillar_hist",  iv_pillar)
    _record("_calib_gex_pillar_hist", gex_p)

    # Composite axes
    _ra_w    = _calib("regime_adx_vs_gex")
    trend_ax = max(-1.0, min(1.0,
        _ra_w * adx_pillar + (1.0 - _ra_w) * (-gex_p) + 0.15 * ca_p))
    _rv_w    = _calib("regime_iv_vs_hv_accel")
    vol_ax   = max(-1.0, min(1.0,
        _rv_w * iv_hv_pillar + (1.0 - _rv_w) * hv_accel_pillar + 0.10 * abs(iv_pillar)))

    # Regime confidence: 1 - normalised stddev
    _pvec = np.array([iv_pillar, iv_hv_pillar, hv_accel_pillar,
                      adx_pillar, ts_p, ca_p, gex_p])
    regime_conf = round(max(0.0, 1.0 - float(np.std(_pvec)) / 0.6), 3)

    # Regime label from (trend_ax, vol_ax)
    _any_shock = any(abs(x) > 0.85 for x in [iv_pillar, hv_accel_pillar, ca_p])
    _vol_hi    = vol_ax  >  0.20
    _vol_lo    = vol_ax  < -0.20
    _trending  = abs(trend_ax) > 0.25
    _ranging   = abs(trend_ax) < 0.15 and not _vol_hi

    if _any_shock or (_vol_hi and not _trending):
        regime, col = _REGIME_VOL_EXPANSION, "#ff3b3b"
    elif _trending and trend_ax > 0.25:
        regime, col = _REGIME_TRENDING_UP, "#00d084"
    elif _trending and trend_ax < -0.25:
        regime, col = _REGIME_TRENDING_DOWN, "#ff7777"
    elif _vol_lo and _ranging:
        regime, col = _REGIME_VOL_COMPRESS, "#1e90ff"
    elif _ranging:
        regime, col = _REGIME_RANGE_BOUND, "#7ec8e3"
    else:
        regime, col = _REGIME_TRANSITION, "#888"

    # Legacy labels
    trend  = ("UPTREND" if trend_ax > 0.3 else "DOWNTREND" if trend_ax < -0.3
              else "RANGE" if abs(trend_ax) < 0.15 else "CHOPPY")
    volst  = ("HIGH VOL" if vol_ax > 0.4 else "VOL EXPANDING" if vol_ax > 0.2
              else "VOL COMPRESSING" if vol_ax < -0.2 else "LOW VOL" if vol_ax < -0.4
              else "NORMAL VOL")
    atr_pct = float(atr14.iloc[-1] / c.iloc[-1] * 100) if float(c.iloc[-1]) > 0 else 0.0

    # Persist for downstream
    st.session_state["opt_regime_label"]      = regime
    st.session_state["opt_regime_confidence"] = regime_conf
    st.session_state["opt_regime_weights"]    = _REGIME_SIGNAL_WEIGHTS[regime]

    _strat_map = {
        "TRENDING HIGH VOL": ["Long ATM Call","Long ATM Put","Long Straddle"],
        "TRENDING LOW VOL":  ["Bull Call Spread","Bear Put Spread","Covered Call"],
        "RANGE LOW VOL":     ["Short Strangle","Iron Condor","Short Straddle"],
        "RANGE HIGH VOL":    ["Iron Condor","Iron Butterfly","Jade Lizard"],
        "TRANSITIONAL":      ["Iron Condor","ATM Straddle (small)","Wait"],
    }
    _compat = {
        _REGIME_TRENDING_UP: "TRENDING LOW VOL",   _REGIME_TRENDING_DOWN: "TRENDING LOW VOL",
        _REGIME_VOL_EXPANSION: "TRENDING HIGH VOL", _REGIME_VOL_COMPRESS: "RANGE LOW VOL",
        _REGIME_RANGE_BOUND: "RANGE LOW VOL",       _REGIME_TRANSITION: "TRANSITIONAL",
    }

    return {
        "trend": trend, "range": volst, "vol": volst, "regime": regime, "color": col,
        "adx": round(adx, 1), "adx_pct": round(adx_pct_v, 1), "iv_pct": round(iv_pct, 1),
        "hv_accel": round(hv_accel, 4), "atr_pct": round(atr_pct, 2), "bb_width_pct": 0.0,
        "gex_sign": gex_s, "regime_confidence": regime_conf,
        "signal_weights": _REGIME_SIGNAL_WEIGHTS[regime],
        "pillars": {
            "iv": round(iv_pillar, 3), "iv_hv": round(iv_hv_pillar, 3),
            "hv_accel": round(hv_accel_pillar, 3), "adx": round(adx_pillar, 3),
            "ts_slope": round(ts_p, 3), "cross_asset": round(ca_p, 3), "gex": round(gex_p, 3),
        },
        "strategy_ev_by_regime": _strat_map.get(_compat.get(regime, "TRANSITIONAL"), []),
    }

# ============================================================
# EVENT DETECTION
# ============================================================

# Known Indian market event calendar patterns
_EVENT_KEYWORDS = {
    # RBI MPC meetings — typically 6 per year (Feb, Apr, Jun, Aug, Oct, Dec)
    "RBI MPC":         {"months": [2, 4, 6, 8, 10, 12], "day_range": (4, 10), "color": "#9c27b0"},
    # Union Budget — typically 1st Feb
    "Union Budget":    {"months": [2], "day_range": (1, 2), "color": "#ff3b3b"},
    # Quarterly results — Q1 Jul-Aug, Q2 Oct-Nov, Q3 Jan-Feb, Q4 Apr-May
    "Quarterly Results Q1": {"months": [7, 8],  "day_range": (1, 31), "color": "#ff8c00"},
    "Quarterly Results Q2": {"months": [10, 11],"day_range": (1, 31), "color": "#ff8c00"},
    "Quarterly Results Q3": {"months": [1, 2],  "day_range": (1, 28), "color": "#ff8c00"},
    "Quarterly Results Q4": {"months": [4, 5],  "day_range": (1, 31), "color": "#ff8c00"},
    # Nifty expiry — last Thursday of every month
    "Monthly Expiry":  {"type": "last_thursday", "color": "#ffb347"},
}

def detect_events(expiry_date_str):
    """Detect upcoming market events between today and the expiry date.
    Returns list of dicts: [{event, date_str, days_away, color, impact}]
    """
    events = []
    try:
        today  = datetime.now().date()
        exp    = datetime.strptime(expiry_date_str, "%Y-%m-%d").date()

        # Pre-compute last-Thursday of each month in the scan range (cache per month)
        _last_thu_cache = {}
        def last_thursday_of_month(yr, mo):
            key = (yr, mo)
            if key not in _last_thu_cache:
                # Go to first day of next month, subtract days until Thursday
                if mo == 12:
                    first_next = date(yr + 1, 1, 1)
                else:
                    first_next = date(yr, mo + 1, 1)
                last_d = first_next - timedelta(days=1)
                # weekday(): Mon=0 … Thu=3 … Sun=6
                days_back = (last_d.weekday() - 3) % 7
                _last_thu_cache[key] = last_d - timedelta(days=days_back)
            return _last_thu_cache[key]

        # Pre-compute first-Wed of each RBI month in the scan range
        _rbi_wed_cache = {}
        def rbi_dates_of_month(yr, mo):
            key = (yr, mo)
            if key not in _rbi_wed_cache:
                first_d = date(yr, mo, 1)
                offset  = (2 - first_d.weekday()) % 7   # 2 = Wednesday
                wed = first_d + timedelta(days=offset)
                _rbi_wed_cache[key] = (wed, wed + timedelta(days=1))  # (Wed, Thu)
            return _rbi_wed_cache[key]

        cursor = today
        while cursor <= exp:
            m  = cursor.month
            yr = cursor.year
            d  = cursor.day

            # RBI MPC — first Wed+Thu of Feb, Apr, Jun, Aug, Oct, Dec
            if m in [2, 4, 6, 8, 10, 12]:
                rbi_wed, rbi_thu = rbi_dates_of_month(yr, m)
                if cursor in (rbi_wed, rbi_thu):
                    events.append({
                        "event":     "RBI MPC Meeting",
                        "date_str":  cursor.isoformat(),
                        "days_away": (cursor - today).days,
                        "color":     "#9c27b0",
                        "impact":    "HIGH — repo rate / policy stance. IV typically spikes 2–5pp before.",
                    })

            # Union Budget — 1 Feb
            if m == 2 and d == 1:
                events.append({
                    "event":     "Union Budget",
                    "date_str":  cursor.isoformat(),
                    "days_away": (cursor - today).days,
                    "color":     "#ff3b3b",
                    "impact":    "EXTREME — single largest annual vol event for NSE. Avoid naked shorts.",
                })

            # Last Thursday of every month = Nifty Monthly Expiry
            if cursor == last_thursday_of_month(yr, m):
                events.append({
                    "event":     "NIFTY Monthly Expiry",
                    "date_str":  cursor.isoformat(),
                    "days_away": (cursor - today).days,
                    "color":     "#ffb347",
                    "impact":    "MEDIUM — gamma squeeze risk, max pain pull, wide bid-ask near close.",
                })

            cursor += timedelta(days=1)

        # Quarterly results window — flag only when the expiry window genuinely overlaps
        # Results seasons: Jan-Feb (Q3), Apr-May (Q4), Jul-Aug (Q1), Oct-Nov (Q2)
        # An event is only meaningful if expiry falls within or just after the season.
        qr_seasons = [
            ((1,  1), (2, 28), "Q3 Results Season (Jan–Feb)"),
            ((4,  1), (5, 31), "Q4 Results Season (Apr–May)"),
            ((7,  1), (8, 31), "Q1 Results Season (Jul–Aug)"),
            ((10, 1), (11,30), "Q2 Results Season (Oct–Nov)"),
        ]
        for (sm, sd), (em, ed), label in qr_seasons:
            season_start = date(today.year, sm, sd)
            # Use next year if season already passed this year
            if season_start < today - timedelta(days=30):
                try:
                    season_start = date(today.year + 1, sm, sd)
                except ValueError:
                    continue
            season_end_day = 28 if em == 2 else ed
            try:
                season_end = date(season_start.year, em, season_end_day)
            except ValueError:
                continue
            # Only add if expiry falls within or shortly after the season
            if today <= season_end and exp >= season_start:
                days_to_season = max(0, (season_start - today).days)
                events.append({
                    "event":     label,
                    "date_str":  f"{season_start.isoformat()} → {season_end.isoformat()}",
                    "days_away": days_to_season,
                    "color":     "#ff8c00",
                    "impact":    "MEDIUM — stock-specific earnings risk. Check company-level result dates.",
                })

    except Exception:
        pass

    # Deduplicate by event name, sort by days_away
    seen = set()
    deduped = []
    for ev in events:
        k = ev["event"]
        if k not in seen:
            seen.add(k)
            deduped.append(ev)
    return sorted(deduped, key=lambda x: x["days_away"])

# ============================================================
# LIQUIDITY FILTER
# ============================================================

def liquidity_analysis(chain_df, spot, step, atm_iv, T):
    """Percentile-based liquidity scoring — adaptive to each instrument.
    All three components (spread, OI, volume) are ranked within the chain's own
    distribution so the score is meaningful for Nifty (huge OI) and mid-cap stocks alike.
    Falls back to absolute CFG thresholds only when chain sample < 5 strikes.
    liquid_score: 0–100 composite (higher = better execution quality).
    """
    if chain_df is None or chain_df.empty:
        return {"liquid_score": 0, "atm_spread_pct": 999, "atm_oi_ok": False,
                "atm_vol_ok": False, "verdict": "NO DATA", "color": "#555", "rows": [],
                "spread_pct_rank": 0, "oi_pct_rank": 0, "vol_pct_rank": 0}

    atm_k = atm_strike(spot, step)
    lo    = atm_k - 3 * step
    hi    = atm_k + 3 * step
    df    = chain_df[(chain_df.Strike >= lo) & (chain_df.Strike <= hi)].copy()
    if df.empty:
        df = chain_df.copy()  # fallback: use full chain if no strikes in ±3 range

    # ── Collect raw liquidity metrics for all strikes in window ──────────────
    rows = []
    all_spreads_ce, all_spreads_pe = [], []
    all_oi_ce, all_oi_pe = [], []
    all_vol_ce, all_vol_pe = [], []

    for _, row in df.iterrows():
        k = float(row.Strike)
        ce_bid, ce_ask = float(row.CE_Bid), float(row.CE_Ask)
        pe_bid, pe_ask = float(row.PE_Bid), float(row.PE_Ask)
        ce_mid  = (ce_bid + ce_ask) / 2 if ce_ask > 0 else float(row.CE_LTP)
        pe_mid  = (pe_bid + pe_ask) / 2 if pe_ask > 0 else float(row.PE_LTP)
        ce_spread_pct = (ce_ask - ce_bid) / ce_mid * 100 if ce_mid > 0.5 else 999
        pe_spread_pct = (pe_ask - pe_bid) / pe_mid * 100 if pe_mid > 0.5 else 999

        ce_oi  = int(row.CE_OI);  pe_oi  = int(row.PE_OI)
        ce_vol = int(row.CE_Vol); pe_vol = int(row.PE_Vol)

        if ce_spread_pct < 999: all_spreads_ce.append(ce_spread_pct)
        if pe_spread_pct < 999: all_spreads_pe.append(pe_spread_pct)
        if ce_oi  > 0: all_oi_ce.append(ce_oi)
        if pe_oi  > 0: all_oi_pe.append(pe_oi)
        if ce_vol > 0: all_vol_ce.append(ce_vol)
        if pe_vol > 0: all_vol_pe.append(pe_vol)

        rows.append({
            "Strike":      k,
            "CE Spread%":  round(ce_spread_pct, 1),
            "PE Spread%":  round(pe_spread_pct, 1),
            "CE OI":       ce_oi,
            "PE OI":       pe_oi,
            "CE Vol":      ce_vol,
            "PE Vol":      pe_vol,
        })

    # ── ATM-specific raw metrics ──────────────────────────────────────────────
    atm_rows = df.iloc[(df.Strike - spot).abs().argsort()[:1]]
    if not atm_rows.empty:
        ar = atm_rows.iloc[0]
        ce_bid_a, ce_ask_a = float(ar.CE_Bid), float(ar.CE_Ask)
        ce_mid_a  = (ce_bid_a + ce_ask_a) / 2 if ce_ask_a > 0 else float(ar.CE_LTP)
        pe_bid_a, pe_ask_a = float(ar.PE_Bid), float(ar.PE_Ask)
        pe_mid_a  = (pe_bid_a + pe_ask_a) / 2 if pe_ask_a > 0 else float(ar.PE_LTP)
        atm_ce_spread = (ce_ask_a - ce_bid_a) / ce_mid_a * 100 if ce_mid_a > 0.5 else 999
        atm_pe_spread = (pe_ask_a - pe_bid_a) / pe_mid_a * 100 if pe_mid_a > 0.5 else 999
        atm_spread_pct = (atm_ce_spread + atm_pe_spread) / 2
        atm_ce_oi  = int(ar.CE_OI);   atm_pe_oi  = int(ar.PE_OI)
        atm_ce_vol = int(ar.CE_Vol);  atm_pe_vol = int(ar.PE_Vol)
        atm_oi     = min(atm_ce_oi, atm_pe_oi)
        atm_vol    = min(atm_ce_vol, atm_pe_vol)
    else:
        atm_spread_pct = 999; atm_oi = 0; atm_vol = 0
        atm_ce_oi = atm_pe_oi = atm_ce_vol = atm_pe_vol = 0

    # ── Percentile-based scoring ──────────────────────────────────────────────
    use_percentile = len(all_spreads_ce) >= 5 and len(all_oi_ce) >= 5

    if use_percentile:
        all_spreads = all_spreads_ce + all_spreads_pe
        all_oi      = all_oi_ce + all_oi_pe
        all_vol     = all_vol_ce + all_vol_pe

        # Spread: lower spread = better. Score = 1 - spread_percentile.
        # (ATM spread at 10th pct of chain → excellent; at 90th pct → illiquid)
        spread_pct_rank = _percentile_score(all_spreads, atm_spread_pct)
        spread_score    = max(0.0, min(100.0, (1 - spread_pct_rank) * 100))

        # OI: higher OI = better. Score = oi_percentile.
        oi_pct_rank  = _percentile_score(all_oi, atm_oi)
        oi_score     = max(0.0, min(100.0, oi_pct_rank * 100))

        # Volume: higher volume = better. Score = vol_percentile.
        vol_pct_rank = _percentile_score(all_vol, atm_vol) if atm_vol > 0 else 0.0
        vol_score    = max(0.0, min(100.0, vol_pct_rank * 100))

        # Boolean helpers for display (based on percentile, not fixed threshold)
        atm_oi_ok  = oi_pct_rank  >= CFG["liq_oi_pct_min"]  / 100.0
        atm_vol_ok = vol_pct_rank >= CFG["liq_vol_pct_min"]  / 100.0

        # Annotate rows with OK status
        for r in rows:
            r_spr = (r["CE Spread%"] + r["PE Spread%"]) / 2
            r_oi  = min(r["CE OI"], r["PE OI"])
            r_vol = min(r["CE Vol"], r["PE Vol"])
            r_spr_pct  = _percentile_score(all_spreads, r_spr)
            r_oi_pct   = _percentile_score(all_oi,      r_oi)
            r_vol_pct  = _percentile_score(all_vol,     r_vol)
            ok = (r_spr_pct < CFG["liq_spread_pct_sell"] / 100.0
                  and r_oi_pct  >= CFG["liq_oi_pct_min"]  / 100.0
                  and r_vol_pct >= CFG["liq_vol_pct_min"]  / 100.0)
            r["CE OK"] = "✓" if ok else "⚠"
            r["PE OK"] = "✓" if ok else "⚠"

    else:
        # Absolute fallback when chain is tiny (< 5 sampled strikes)
        _min_oi  = CFG["liq_min_oi"]
        _min_vol = CFG["liq_min_vol"]
        _max_spr = CFG["liq_max_spread_pct"]
        spread_score = max(0.0, min(100.0, 100.0 * math.exp(-atm_spread_pct / max(_max_spr, 1e-9))))
        oi_score     = 100.0 if atm_oi  >= _min_oi  else 25.0
        vol_score    = 100.0 if atm_vol >= _min_vol else 25.0
        atm_oi_ok    = atm_oi  >= _min_oi
        atm_vol_ok   = atm_vol >= _min_vol
        spread_pct_rank = 1 - spread_score / 100.0
        oi_pct_rank     = oi_score  / 100.0
        vol_pct_rank    = vol_score / 100.0
        for r in rows:
            r["CE OK"] = "✓" if (r["CE OI"] >= _min_oi and r["CE Vol"] >= _min_vol
                                  and r["CE Spread%"] <= _max_spr) else "⚠"
            r["PE OK"] = "✓" if (r["PE OI"] >= _min_oi and r["PE Vol"] >= _min_vol
                                  and r["PE Spread%"] <= _max_spr) else "⚠"

    # ── Composite liquidity score: calibrated weights ────────────────────────
    liquid_score = int(
        _calib("liq_spread_w") * spread_score
      + _calib("liq_oi_w")     * oi_score
      + _calib("liq_vol_w")    * vol_score
    )

    if liquid_score >= 75:
        verdict = "LIQUID — safe to trade multi-leg"
        color   = "#00d084"
    elif liquid_score >= 50:
        verdict = "MODERATE — use limit orders, expect some slippage"
        color   = "#ffb347"
    else:
        verdict = (f"ILLIQUID — spread {atm_spread_pct:.1f}% / "
                   f"OI {'OK' if atm_oi_ok else 'THIN'} / "
                   f"Vol {'OK' if atm_vol_ok else 'THIN'}")
        color   = "#ff3b3b"

    return {
        "liquid_score":    liquid_score,
        "atm_spread_pct":  round(atm_spread_pct, 2),
        "atm_oi_ok":       atm_oi_ok,
        "atm_vol_ok":      atm_vol_ok,
        "verdict":         verdict,
        "color":           color,
        "rows":            rows,
        "spread_pct_rank": round(spread_pct_rank * 100, 1),
        "oi_pct_rank":     round(oi_pct_rank  * 100, 1),
        "vol_pct_rank":    round(vol_pct_rank  * 100, 1),
        "percentile_mode": use_percentile,
    }


# ============================================================
# RELATIVE STRENGTH vs NIFTY
# ============================================================

def relative_strength_vs_nifty(symbol, ohlcv_df, window=20):
    """Compute RS ratio of symbol vs Nifty over `window` trading days.
    RS = cumulative return of symbol / cumulative return of Nifty.
    RS > 1 = outperforming, RS < 1 = underperforming.
    Returns dict with rs_ratio, trend, color, rs_series (rolling, last 60 pts).
    """
    if symbol.upper() in ("NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "SENSEX"):
        return None  # meaningless to compare index with itself
    if ohlcv_df is None or ohlcv_df.empty or len(ohlcv_df) < window + 2:
        return None
    try:
        nifty_raw = yf.download("^NSEI", period="6mo", interval="1d",
                                progress=False, auto_adjust=True)
        if nifty_raw.empty:
            return None

        def _extract_close(df_raw):
            """Robustly extract a date-indexed Close series from any yfinance output format."""
            df = df_raw.copy()
            # Flatten MultiIndex columns if present (yfinance ≥0.2 may have ticker as level)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0].lower() for c in df.columns]
            else:
                df.columns = [str(c).lower() for c in df.columns]
            # Ensure date index is plain date (not Datetime)
            if not isinstance(df.index, pd.DatetimeIndex):
                df = df.reset_index()
                # find the date column
                date_col = next((c for c in df.columns if c in ("date", "datetime", "index")), None)
                if date_col:
                    df = df.set_index(date_col)
            df.index = pd.to_datetime(df.index).normalize()   # strip time component
            df.index = df.index.date                           # plain date objects
            return df["close"].astype(float).dropna()

        sym_close   = _extract_close(ohlcv_df) if "close" in ohlcv_df.columns or (
            isinstance(ohlcv_df.columns, pd.MultiIndex) and any("close" in str(c).lower() for c in ohlcv_df.columns)
        ) else None

        if sym_close is None:
            return None

        nifty_close = _extract_close(nifty_raw)

        # Align on common dates
        common = sorted(set(sym_close.index) & set(nifty_close.index))
        if len(common) < window + 5:
            return None

        s = sym_close.loc[common]
        n = nifty_close.loc[common]

        # Point-in-time RS ratio (last `window` days)
        s_ret   = float(s.iloc[-1] / s.iloc[-1 - window] - 1) * 100
        n_ret   = float(n.iloc[-1] / n.iloc[-1 - window] - 1) * 100
        rs      = round((1 + s_ret / 100) / (1 + n_ret / 100 + 1e-9), 4)

        # Rolling RS series — use full history not just tail(window+1)
        rs_series = []
        s_arr = s.values.astype(float)
        n_arr = n.values.astype(float)
        for i in range(window, len(s_arr)):
            sr = s_arr[i] / s_arr[i - window] - 1
            nr = n_arr[i] / n_arr[i - window] - 1
            rs_series.append(round((1 + sr) / (1 + nr + 1e-9), 4))
        rs_series = rs_series[-60:]   # keep last 60 rolling points for chart

        # RS slope: last 5 rolling points
        rs_slope = (rs_series[-1] - rs_series[-5]) if len(rs_series) >= 5 else 0

        if rs > 1.02 and rs_slope > 0:
            rs_trend, rs_color = "STRONGLY OUTPERFORMING", "#00d084"
        elif rs > 1.0:
            rs_trend, rs_color = "OUTPERFORMING",           "#7dca84"
        elif rs < 0.98 and rs_slope < 0:
            rs_trend, rs_color = "STRONGLY UNDERPERFORMING","#ff3b3b"
        elif rs < 1.0:
            rs_trend, rs_color = "UNDERPERFORMING",          "#ff7777"
        else:
            rs_trend, rs_color = "IN LINE WITH NIFTY",       "#888"

        return {
            "rs_ratio":      rs,
            "sym_ret_pct":   round(s_ret, 2),
            "nifty_ret_pct": round(n_ret, 2),
            "trend":         rs_trend,
            "color":         rs_color,
            "rs_series":     rs_series,
            "rs_slope":      round(rs_slope, 4),
        }
    except Exception:
        return None

# ============================================================
# STRATEGY BACKTEST BY REGIME
# ============================================================

def backtest_strategies_by_regime(ohlcv_df, iv_history, hv20):
    """Simple walk-forward backtest: for each 30-day window in history,
    classify the regime and simulate the P&L of 4 canonical strategies.
    Returns: dict of {strategy_name: {wins, losses, avg_pnl_pct, best_regime}}
    This is a simplified regime-P&L attribution (not a full tick-level backtest).
    It uses: close-to-close returns within the 30-day window, and the approximate
    P&L of each strategy given the actual move vs the implied straddle.
    """
    results = {}
    if ohlcv_df is None or ohlcv_df.empty or len(ohlcv_df) < 60:
        return results
    c   = ohlcv_df["close"].astype(float).reset_index(drop=True)
    h   = ohlcv_df["high"].astype(float).reset_index(drop=True)
    l   = ohlcv_df["low"].astype(float).reset_index(drop=True)

    strategies = {
        "Short Straddle":  {"type": "credit_neutral",  "best_regime": []},
        "Long Straddle":   {"type": "debit_neutral",   "best_regime": []},
        "Bull Call Spread":{"type": "debit_bull",      "best_regime": []},
        "Bear Put Spread": {"type": "debit_bear",      "best_regime": []},
        "Iron Condor":     {"type": "credit_neutral",  "best_regime": []},
    }

    window = 21  # trading days per expiry cycle
    pnl_by_regime = {s: {} for s in strategies}

    for i in range(window, len(c) - window, window):
        spot_entry = float(c.iloc[i])
        spot_exit  = float(c.iloc[min(i + window, len(c)-1)])
        # HV of entry window (to estimate IV proxy)
        lr_w = np.log(c.iloc[i-window:i] / c.iloc[i-window:i].shift(1)).dropna()
        iv_proxy = float(lr_w.std() * np.sqrt(252)) if len(lr_w) >= 5 else hv20
        # Regime classification: pass the FULL ohlcv_df up to point i so that
        # detect_market_regime can compute a meaningful 200-EMA trend filter.
        # Using only the 21-row sub-window caused the 200-EMA to fall back to the
        # 50-EMA every iteration, systematically misclassifying trend regimes.
        full_sub_df = pd.DataFrame({
            "close": c.iloc[:i].values,
            "high":  h.iloc[:i].values,
            "low":   l.iloc[:i].values,
        })
        regime_d = detect_market_regime(full_sub_df, iv_proxy, hv20, st.session_state.get("opt_cross_asset", {}).get("composite", 0.0))
        reg_label = regime_d.get("regime", "UNKNOWN")

        # Actual move
        actual_ret  = (spot_exit - spot_entry) / spot_entry

        # ── Strategy P&L using Black-Scholes pricing at delta-based strikes ──
        # No heuristic fractions. All legs priced at iv_proxy, T = window/252.
        # Short strikes chosen at delta 0.30 (≈1σ OTM) for all credit strategies
        # and at delta 0.20 (≈1.5σ OTM) for wings — same logic as live universe.
        T_window = window / 252.0
        r_window = CFG["rfr_default"] / 100.0

        def _c_bt(K): return bs_price(spot_entry, K, T_window, r_window, iv_proxy, "call")
        def _p_bt(K): return bs_price(spot_entry, K, T_window, r_window, iv_proxy, "put")
        def _d1_bt(K):
            if T_window <= 0 or iv_proxy <= 0: return 0.5
            F = spot_entry * math.exp(r_window * T_window)
            return (math.log(max(F/K,1e-9)) + 0.5*iv_proxy**2*T_window) / (iv_proxy*math.sqrt(T_window))
        def _delta_c_bt(K): return float(_ncdf(_d1_bt(K)))

        # Find nearest strikes at target deltas
        _sigma_move = iv_proxy * spot_entry * math.sqrt(T_window)
        # Grid: ±0.5σ to ±2.5σ around spot
        _grid = [spot_entry + i * _sigma_move * 0.5 for i in range(-5, 6)]
        _grid = [max(g, spot_entry * 0.5) for g in _grid]  # floor at 50% of spot

        def _find_k_above(target_delta):
            cands = [k for k in _grid if k > spot_entry]
            if not cands: return spot_entry + _sigma_move
            return min(cands, key=lambda k: abs(_delta_c_bt(k) - target_delta))

        def _find_k_below(target_abs_delta):
            cands = [k for k in _grid if k < spot_entry]
            if not cands: return spot_entry - _sigma_move
            return min(cands, key=lambda k: abs(abs(_delta_c_bt(k) - 1) - target_abs_delta))

        # Δ0.30 OTM call = short call for credit strategies
        _k_sc30 = _find_k_above(0.30)
        _k_sp30 = _find_k_below(0.30)
        _k_sc20 = _find_k_above(0.20)
        _k_sp20 = _find_k_below(0.20)

        # ── Short Straddle ──
        _ss_credit = _c_bt(spot_entry) + _p_bt(spot_entry)
        _ss_intr   = abs(spot_exit - spot_entry)
        ss_raw     = (_ss_credit - _ss_intr) / spot_entry * 100
        ss_pnl     = max(ss_raw, -2.0 * _ss_credit / spot_entry * 100)

        # ── Long Straddle ── mirror
        ls_pnl = -ss_raw

        # ── Bull Call Spread: BUY ATM, SELL 30Δ OTM call ──
        _debit_bcs = max(_c_bt(spot_entry) - _c_bt(_k_sc30), 0.01)
        _width_bcs = _k_sc30 - spot_entry
        if actual_ret > 0:
            _gain   = min(spot_exit - spot_entry, _width_bcs)
            bcs_pnl = (_gain - _debit_bcs) / spot_entry * 100
        else:
            bcs_pnl = -_debit_bcs / spot_entry * 100

        # ── Bear Put Spread: BUY ATM put, SELL 30Δ OTM put ──
        _debit_bps = max(_p_bt(spot_entry) - _p_bt(_k_sp30), 0.01)
        _width_bps = spot_entry - _k_sp30
        if actual_ret < 0:
            _gain_p = min(spot_entry - spot_exit, _width_bps)
            bps_pnl = (_gain_p - _debit_bps) / spot_entry * 100
        else:
            bps_pnl = -_debit_bps / spot_entry * 100

        # ── Iron Condor: SELL 30Δ strangle, BUY 20Δ wings ──
        _ic_credit = ((_p_bt(_k_sp30) - _p_bt(_k_sp20)) +
                      (_c_bt(_k_sc30) - _c_bt(_k_sc20)))
        _ic_width  = min(_k_sc30 - _k_sc20, _k_sp20 - _k_sp30)
        _ic_max_loss = max(_ic_width - _ic_credit, 0.01)
        if _k_sp30 <= spot_exit <= _k_sc30:
            ic_pnl = _ic_credit / spot_entry * 100   # full credit kept
        elif spot_exit > _k_sc20 or spot_exit < _k_sp20:
            ic_pnl = -_ic_max_loss / spot_entry * 100  # max loss
        else:
            if spot_exit > _k_sc30:
                frac = (spot_exit - _k_sc30) / max(_k_sc20 - _k_sc30, 1e-6)
            else:
                frac = (_k_sp30 - spot_exit) / max(_k_sp30 - _k_sp20, 1e-6)
            ic_pnl = (_ic_credit - frac * _ic_width) / spot_entry * 100

        for strat, pnl in [("Short Straddle", ss_pnl), ("Long Straddle", ls_pnl),
                            ("Bull Call Spread", bcs_pnl), ("Bear Put Spread", bps_pnl),
                            ("Iron Condor", ic_pnl)]:
            if reg_label not in pnl_by_regime[strat]:
                pnl_by_regime[strat][reg_label] = []
            pnl_by_regime[strat][reg_label].append(pnl)

    # Summarise
    for strat in strategies:
        all_pnls = []
        reg_summary = {}
        for reg, pnls in pnl_by_regime[strat].items():
            avg = round(float(np.mean(pnls)), 2)
            wins = sum(1 for p in pnls if p > 0)
            reg_summary[reg] = {"avg_pnl": avg, "win_rate": round(wins/len(pnls)*100, 0), "n": len(pnls)}
            all_pnls.extend(pnls)
        best_reg = max(reg_summary, key=lambda r: reg_summary[r]["avg_pnl"]) if reg_summary else "—"
        results[strat] = {
            "regime_pnl":  reg_summary,
            "best_regime": best_reg,
            "overall_avg": round(float(np.mean(all_pnls)), 2) if all_pnls else 0,
            "overall_wr":  round(sum(1 for p in all_pnls if p > 0) / len(all_pnls) * 100, 0) if all_pnls else 0,
        }
    return results

# ============================================================
# OI ANALYSIS
# ============================================================

def oi_analysis(chain_df, spot, step=50, T=None, r=None, atm_iv=None, lot_size=None):
    """OI analysis with GEX, gamma flip, OI cluster walls, and skew metrics.
    All numeric parameters should be passed explicitly by the caller.
    Defaults are used only as a last-resort fallback (should never be hit in normal operation).
    lot_size: used to convert GEX to rupee-notional units.
    """
    # Fallbacks — should always be overridden by caller; warn if hitting these
    if T        is None: T        = 7.0 / CFG["ann_days"]   # 7-day fallback
    if r        is None: r        = CFG["rfr_default"] / 100.0
    if atm_iv   is None: atm_iv   = CFG["hv_fallback"]       # use NSE baseline HV
    if lot_size is None: lot_size = CFG["lot_size_fallback"]
    if chain_df is None or chain_df.empty: return {}
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
    # PCR: always use the epsilon denominator form — never return 0 when CE_OI=0.
    # If total_ce=0 but total_pe>0, PCR should be very high (all puts = maximum bullish support).
    # The old `if total_ce > 0 else 0` returned PCR=0 (bearish signal) — exactly backwards.
    pcr_oi    = round(total_pe / (total_ce + 1e-9), 3)

    # ── OI cluster walls (3-strike sliding window) ──
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
    if not atm_r.empty:
        # Use bid/ask mid for straddle pricing — more accurate than LTP which can be stale
        # If bid/ask not available, fall back to LTP
        ce_mid = (float(atm_r.CE_Bid.values[0]) + float(atm_r.CE_Ask.values[0])) / 2 if (
            float(atm_r.CE_Bid.values[0]) > 0 and float(atm_r.CE_Ask.values[0]) > 0
        ) else float(atm_r.CE_LTP.values[0])
        pe_mid = (float(atm_r.PE_Bid.values[0]) + float(atm_r.PE_Ask.values[0])) / 2 if (
            float(atm_r.PE_Bid.values[0]) > 0 and float(atm_r.PE_Ask.values[0]) > 0
        ) else float(atm_r.PE_LTP.values[0])
        straddle = ce_mid + pe_mid
    else:
        straddle = 0.0
    exp_move     = round(straddle / spot * 100, 2) if spot > 0 else 0
    # 2-sigma scaling: straddle ≈ sigma*S*sqrt(T)*sqrt(2/pi), so 2-sigma move = 2*sigma*S*sqrt(T)
    # Therefore: 2-sigma / straddle = 2 / sqrt(2/pi) = 2*sqrt(pi/2) = sqrt(2*pi) ≈ 2.507
    exp_move_2sd = round(exp_move * math.sqrt(2 * math.pi), 2)

    # ── PCR signal — fully percentile-based, adaptive to THIS chain's distribution ──
    full_pcr = chain_df["PCR"].replace([np.inf, -np.inf], np.nan).dropna()
    pcr_pct  = float((full_pcr <= pcr_oi).mean() * 100) if len(full_pcr) > 0 else 50.0
    _pb  = _adaptive_threshold("pcr", CFG["pcr_bull_pct"],  percentile=65.0)
    _pbe = _adaptive_threshold("pcr", CFG["pcr_bear_pct"],  percentile=35.0)
    if   pcr_pct >= _pb:           pcr_sig = f"BULLISH — PCR at {pcr_pct:.0f}th pct; heavy put writing = support"
    elif pcr_pct >= (_pb + _pbe)/2:pcr_sig = f"SLIGHT BULLISH — PCR {pcr_pct:.0f}th pct; put OI outweighs calls"
    elif pcr_pct >= (_pbe + _pb)/2:pcr_sig = f"NEUTRAL — PCR {pcr_pct:.0f}th pct; balanced OI both sides"
    elif pcr_pct >= _pbe:           pcr_sig = f"SLIGHT BEARISH — PCR {pcr_pct:.0f}th pct; call OI building"
    else:                           pcr_sig = f"BEARISH — PCR at {pcr_pct:.0f}th pct; heavy call writing = resistance"

    # ── Gamma Exposure (GEX) — scaled to rupee-notional by lot_size ──
    # GEX (₹) = γ × OI × lot_size × spot²
    # This makes GEX comparable across instruments (not raw contract count).
    # Dealer convention: long call OI = dealer short → positive GEX
    #                    long put  OI = dealer short → negative GEX
    t_safe = max(T, 1.0/CFG["ann_days"])
    gex_rows = []
    for _, row in df.iterrows():
        iv_c = _sanitise_iv(float(row.CE_IV), atm_iv)
        iv_p = _sanitise_iv(float(row.PE_IV), atm_iv)
        g_ce = bs_greeks(spot, float(row.Strike), t_safe, r, iv_c, "call")["gamma"]
        g_pe = bs_greeks(spot, float(row.Strike), t_safe, r, iv_p, "put")["gamma"]
        # Multiply by lot_size × spot² to get standard dollar-gamma units (rupee-GEX)
        # Dollar-gamma convention: GEX = γ × OI × lot_size × S²
        # Rationale: gamma is in delta/₹, so γ × S² gives the full notional gamma sensitivity
        # This matches the SpotGamma / Tier1Alpha convention and makes GEX comparable across instruments
        net  = (g_ce * float(row.CE_OI) - g_pe * float(row.PE_OI)) * lot_size * spot * spot
        gex_rows.append({"Strike": float(row.Strike), "NET_GEX": net})

    gex_df       = pd.DataFrame(gex_rows)
    net_gex_total = float(gex_df["NET_GEX"].sum())
    gex_regime   = ("POSITIVE GEX — range-bound / vol suppressed (dealers buy dips & sell rallies)"
                    if net_gex_total >= 0 else
                    "NEGATIVE GEX — trending / vol expansion likely (dealers chase price)")

    # ── Gamma Flip Level ──
    gex_sorted = gex_df.copy()
    gex_sorted["dist"] = (gex_sorted.Strike - spot).abs()
    gex_sorted = gex_sorted.sort_values("dist").reset_index(drop=True)
    cum_gex    = gex_sorted["NET_GEX"].cumsum()
    gamma_flip = spot
    for i in range(1, len(cum_gex)):
        if cum_gex.iloc[i-1] * cum_gex.iloc[i] <= 0:
            gamma_flip = float(gex_sorted.Strike.iloc[i])
            break

    # ── IV Skew (downside put IV vs upside call IV at ±1 strike) ──
    skew_val, skew_label = None, "—"
    skew_curve = []  # full skew curve: list of {strike, moneyness_pct, put_iv, call_iv, skew_pp}

    # BUG FIX (Skew Percentile Against Empty List): skew_curve was populated AFTER
    # the percentile computation below, so _skew_vals was always [] and _skew_pct
    # was always 0.5 (a hardcoded false signal). Fix: build the full skew curve FIRST,
    # then compute the ATM skew value and its percentile against the real distribution.
    try:
        for _, row in df.iterrows():
            k     = float(row.Strike)
            c_iv  = _sanitise_iv(float(row.CE_IV), 0)
            p_iv  = _sanitise_iv(float(row.PE_IV), 0)
            m_pct = round((k - spot) / spot * 100, 2)   # +ve = OTM call, -ve = OTM put
            if c_iv > 0.01 or p_iv > 0.01:
                skew_curve.append({
                    "strike":       k,
                    "moneyness":    m_pct,
                    "call_iv":      round(c_iv * 100, 2),
                    "put_iv":       round(p_iv * 100, 2),
                    "skew_pp":      round((p_iv - c_iv) * 100, 2),
                })
    except Exception:
        pass

    try:
        dn1   = df.iloc[(df.Strike - (spot - step)).abs().argsort()[:1]]
        up1   = df.iloc[(df.Strike - (spot + step)).abs().argsort()[:1]]
        dn_iv = _sanitise_iv(float(dn1.PE_IV.values[0]), 0)
        up_iv = _sanitise_iv(float(up1.CE_IV.values[0]), 0)
        if dn_iv > 0.01 and up_iv > 0.01:
            skew_val = round((dn_iv - up_iv) * 100, 2)
            # Adaptive: compare skew_val to full skew curve distribution (now populated above)
            _skew_vals = [abs(row.get("skew_pp", 0)) for row in skew_curve if row.get("skew_pp") is not None]
            _skew_pct  = _percentile_score(_skew_vals, abs(skew_val)) if len(_skew_vals) >= 3 else 0.5
            # skew_pct > 0.7 = elevated skew (above 70th pct of current chain)
            if   skew_val > 0 and _skew_pct >= 0.70:
                skew_label = f"BEARISH SKEW +{skew_val:.1f}pp ({_skew_pct*100:.0f}th pct) — elevated put protection demand"
            elif skew_val > 0:
                skew_label = f"MILD BEARISH SKEW +{skew_val:.1f}pp ({_skew_pct*100:.0f}th pct) — moderate put demand"
            elif skew_val < 0 and _skew_pct >= 0.70:
                skew_label = f"CALL SKEW {skew_val:+.1f}pp ({_skew_pct*100:.0f}th pct) — elevated upside speculation"
            else:
                skew_label = f"NEUTRAL SKEW {skew_val:+.1f}pp ({_skew_pct*100:.0f}th pct) — balanced demand"
    except Exception:
        pass

    return dict(
        max_pain=round(max_pain,2), pcr_oi=pcr_oi,
        call_wall=round(call_wall,2), put_wall=round(put_wall,2),
        total_ce_oi=int(total_ce), total_pe_oi=int(total_pe),
        atm_straddle=round(straddle,2), exp_move_pct=exp_move,
        exp_move_2sd_pct=exp_move_2sd, pcr_signal=pcr_sig,
        net_gex=round(net_gex_total, 2), gex_regime=gex_regime,
        gamma_flip=round(gamma_flip, 2), gex_df=gex_df,
        skew_pp=skew_val, skew_label=skew_label,
        skew_curve=skew_curve,
    )

# ============================================================
# PROBABILISTIC SCORING ENGINE
# ============================================================

def _atr_seed(price: float) -> float:
    """Return an adaptive ATR estimate for `price` when no OHLCV data is available.
    Uses the median of all previously recorded ATR% values from this symbol's history.
    Falls back to the median NSE index ATR% (~1.2%) when no history exists.
    The median of observed ATR% values is a much better prior than a fixed 1.5%.
    """
    atr_pct_hist = st.session_state.get("_calib_atr_pct_hist", [])
    if len(atr_pct_hist) >= 3:
        median_atr_pct = float(np.median(atr_pct_hist))
    else:
        # NSE Nifty 20-year median daily ATR ≈ 1.1-1.3%; use 1.2% as neutral seed
        median_atr_pct = 0.012
    return price * median_atr_pct


def _logistic(x):
    """Sigmoid function: maps any real x → (0, 1)."""
    return 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, x))))



def _gate_signals_by_regime(factor_scores: dict, regime: str,
                             confidence: float) -> dict:
    """Regime-first signal gating (MONARCH v2 Tier 2 Fix 5).

    Suppresses factors that are structurally uninformative in the current regime
    by setting them to zero before the weighted composite is computed.
    Confidence scales the suppression: low confidence → partial suppression only.

    Gating rules (empirically motivated):
      TRENDING_UP / TRENDING_DOWN:
        - trend + rel_strength fully active
        - flow active (confirms momentum)
        - vol_regime reduced (vol not predictive in clean trends)
        - positioning suppressed (positioning reverts, trend runs)
      VOL_EXPANSION:
        - flow + vol_regime fully active (vol events are flow-driven)
        - trend SUPPRESSED (mean-reversion dominates after vol spikes)
        - rel_strength reduced
      VOL_COMPRESSION:
        - positioning fully active (range-bound → mean-revert to strikes)
        - flow reduced (flow meaningless in coiled market)
        - trend suppressed
      RANGE_BOUND:
        - positioning dominant
        - trend + rel_strength suppressed
      TRANSITION:
        - all signals active but flow up-weighted (spec: 0.50)
        - acceleration already handled in flow engine
    """
    # Gate multipliers per regime (1.0 = fully on, 0.0 = fully suppressed)
    # Uses module-level _REGIME_* constants to avoid duplication
    _gates = {
        _REGIME_TRENDING_UP:   {"flow": 1.0, "positioning": 0.4, "vol_regime": 0.5,
                                "rel_strength": 1.0, "trend": 1.0},
        _REGIME_TRENDING_DOWN: {"flow": 1.0, "positioning": 0.4, "vol_regime": 0.5,
                                "rel_strength": 1.0, "trend": 1.0},
        _REGIME_VOL_EXPANSION: {"flow": 1.0, "positioning": 0.6, "vol_regime": 1.0,
                                "rel_strength": 0.5, "trend": 0.0},
        _REGIME_VOL_COMPRESS:  {"flow": 0.5, "positioning": 1.0, "vol_regime": 0.8,
                                "rel_strength": 0.4, "trend": 0.2},
        _REGIME_RANGE_BOUND:   {"flow": 0.5, "positioning": 1.0, "vol_regime": 0.6,
                                "rel_strength": 0.3, "trend": 0.1},
    }
    # TRANSITION regime: all signals active, flow up-weighted
    _gates[_REGIME_TRANSITION] = {"flow": 1.0, "positioning": 0.9, "vol_regime": 0.9,
                                   "rel_strength": 0.7, "trend": 0.6}
    default_gates = {"flow": 1.0, "positioning": 1.0, "vol_regime": 1.0,
                     "rel_strength": 0.8, "trend": 0.6}
    gates = _gates.get(regime, default_gates)

    # Scale suppression by confidence: low conf → partial gates only
    # conf=1.0 → full gating; conf=0.0 → no gating (all signals pass)
    conf = max(0.0, min(1.0, float(confidence)))
    gated = {}
    for k, v in factor_scores.items():
        g = gates.get(k, 1.0)
        # Effective gate: interpolate between 1.0 (no gate) and g (full gate)
        eff_gate = 1.0 - conf * (1.0 - g)
        gated[k] = float(v) * eff_gate
    return gated


def _rank_based_factor_weights(factor_scores_now: dict, fw_cfg: dict) -> dict:
    """
    Exponential-weighted rank-based adaptive weights (MONARCH v2 Block F).

    For each factor:
      hit_rate_w   = exponentially-weighted directional accuracy
      corr         = Pearson correlation with forward return (signed)
      perf_score   = hit_rate_w x |corr|

    Weights = softmax of ranks (not raw perf_scores -> prevents domination).
    Blended with CFG prior via data_trust = min(1, n_obs/100).
    """
    try:
        sym    = st.session_state.get("opt_symbol", "").upper()
        ret_k  = f"{sym}:_calib_realised_ret_hist" if sym else "_calib_realised_ret_hist"
        r_hist = st.session_state.get(ret_k,
                 st.session_state.get("_calib_realised_ret_hist", []))
        if len(r_hist) < 10:
            return fw_cfg.copy()

        r_arr_f = np.array(r_hist, dtype=float)
        fh      = st.session_state.get("opt_factor_hist", {})
        perf    = {}; signs = {}; n_max = 0

        for fn in factor_scores_now:
            hist = np.array(fh.get(fn, []), dtype=float)
            n    = min(len(hist), len(r_arr_f))
            n_max = max(n_max, n)
            if n < 10:
                perf[fn] = fw_cfg.get(fn, 0.2); signs[fn] = 1; continue
            s_a  = hist[-n:]; r_a = r_arr_f[-n:]
            if np.std(s_a) < 1e-9:
                perf[fn] = fw_cfg.get(fn, 0.2); signs[fn] = 1; continue
            corr = float(np.corrcoef(s_a, r_a)[0, 1])
            if math.isnan(corr):
                corr = 0.0
            # Exponentially weighted hit rate
            decay = np.array([0.95 ** (n - 1 - i) for i in range(n)])
            decay /= decay.sum()
            match  = (np.sign(s_a) == np.sign(r_a)).astype(float)
            hit_w  = float(np.dot(decay, match))
            perf[fn]  = max(0.01, hit_w * max(0.0, abs(corr)))
            signs[fn] = 1 if corr >= 0 else -1

        # Rank-based normalisation (average-rank for ties, prevents first-index bias)
        _perf_keys = list(perf.keys())
        _perf_vals = np.array([perf[k] for k in _perf_keys], dtype=float)
        _order     = np.argsort(_perf_vals)              # ascending
        _ranks     = np.empty(len(_perf_vals))
        _ranks[_order] = np.arange(1, len(_perf_vals)+1, dtype=float)
        # Average-rank for ties
        for _v in np.unique(_perf_vals):
            _mask = _perf_vals == _v
            _ranks[_mask] = _ranks[_mask].mean()
        ranks  = {k: float(r) for k, r in zip(_perf_keys, _ranks)}
        total  = sum(ranks.values()) or 1.0
        norm_w = {k: v / total for k, v in ranks.items()}

        # Blend with prior
        trust  = min(1.0, n_max / 100.0)
        final  = {k: (1.0 - trust) * fw_cfg.get(k, 0.2) + trust * norm_w.get(k, 0.2)
                  for k in factor_scores_now}
        tot_f  = sum(final.values()) or 1.0
        final  = {k: v / tot_f for k, v in final.items()}

        st.session_state["_factor_corr_signs"] = signs
        return final
    except Exception:
        return fw_cfg.copy()

def compute_probabilistic_score(
        bias_res, chain_df, ohlcv_df, spot, atm_iv, hv20,
        ivr, oi_d, r, q, T, step):
    """
    Short-term directional signal model (1–5 day horizon).

    Factor hierarchy — leading indicators have highest weight:
      FLOW        (0.30) — ΔIV, ΔPCR, ΔSkew, ΔOI, ΔGEX  [most predictive]
      POSITIONING (0.25) — PCR level, OI walls, max pain distance
      VOL REGIME  (0.20) — IV/HV percentile, term structure slope
      REL STRENGTH(0.15) — stock vs Nifty 20-day
      TREND       (0.10) — EMA structure + ADX  [confirming, lagging]

    RSI and MACD contribute only weakly inside the trend factor.
    Signal → logistic → prob_up / prob_down.
    """
    fs = {}   # feature scores, each ∈ [-1, +1]

    # Guard: ohlcv_df may be None or empty
    _ohlcv   = ohlcv_df if (ohlcv_df is not None and not ohlcv_df.empty) else pd.DataFrame()
    c_series = _ohlcv["close"].astype(float) if not _ohlcv.empty else pd.Series(dtype=float)
    h_series = _ohlcv["high"].astype(float)  if not _ohlcv.empty else pd.Series(dtype=float)
    l_series = _ohlcv["low"].astype(float)   if not _ohlcv.empty else pd.Series(dtype=float)

    # ════════════════════════════════════════════════════════════════════════
    # FACTOR 1 — FLOW  (weight 0.30) — LEADING, highest predictive value
    # Already computed in directional_bias → bias_res["flow"]
    # ════════════════════════════════════════════════════════════════════════
    flow_d      = bias_res.get("flow", {})
    flow_score  = float(flow_d.get("flow_score", 0.0))   # composite from compute_flow_scores
    flow_mag    = float(flow_d.get("flow_magnitude", 0.0))

    # Individual flow components (used for display breakdown)
    dPCR  = float(flow_d.get("dPCR",  0.0))   # positive = rising PCR = bullish
    dSkew = float(flow_d.get("dSkew", 0.0))   # positive = flattening skew = bullish
    dIV   = float(flow_d.get("dIV",   0.0))   # positive = falling IV = bullish
    dOI   = float(flow_d.get("dOI",   0.0))   # magnitude signal
    dGEX  = float(flow_d.get("dGEX",  0.0))   # falling GEX = move incoming

    fs["flow_score"] = round(max(-1.0, min(1.0, flow_score)), 4)
    fs["dPCR"]       = round(dPCR, 4)
    fs["dSkew"]      = round(dSkew, 4)
    fs["dIV"]        = round(dIV, 4)
    fs["dOI"]        = round(dOI, 4)
    fs["dGEX"]       = round(dGEX, 4)

    # ════════════════════════════════════════════════════════════════════════
    # FACTOR 2 — POSITIONING  (weight 0.25) — LEADING, structural price magnets
    # PCR level percentile (where put/call OI sits in history)
    # OI skew (more puts below vs calls above spot)
    # Distance from max pain (gravitational pull)
    # ════════════════════════════════════════════════════════════════════════
    _chain_safe = chain_df if (chain_df is not None and not chain_df.empty) else None

    if _chain_safe is not None:
        total_ce = float(_chain_safe["CE_OI"].sum())
        total_pe = float(_chain_safe["PE_OI"].sum())
        pcr_v    = total_pe / (total_ce + 1e-9)
        all_pcr  = _chain_safe["PCR"].replace([np.inf, -np.inf], np.nan).dropna().values
        pcr_pct  = _percentile_score(all_pcr, pcr_v)  # 0=bearish, 1=bullish
        # PCR level: high PCR = heavy put writing = support below = bullish
        pcr_level_z = 2 * pcr_pct - 1   # [-1, +1]

        # OI skew: more put OI below spot than call OI above = asymmetric put support = bullish
        above_oi = float(_chain_safe[_chain_safe.Strike > spot]["CE_OI"].sum())
        below_oi = float(_chain_safe[_chain_safe.Strike < spot]["PE_OI"].sum())
        oi_skew_val = (below_oi - above_oi) / (total_ce + total_pe + 1e-9)
        # BUG FIX (OI Skew Double-Append): dedup guard — directional_bias() already wrote
        # this Load's value above. If this Load id was already appended, overwrite in place.
        _ps_skew_lids = st.session_state.get("_flow_skew_oi_load_ids", [])
        _ps_skew_lid  = st.session_state.get("opt_load_id", 0)
        oi_skew_hist  = st.session_state.get("_flow_skew_oi_hist", [])
        if _ps_skew_lids and _ps_skew_lids[-1] == _ps_skew_lid:
            oi_skew_hist[-1] = float(oi_skew_val)   # overwrite — no new append
        else:
            oi_skew_hist.append(float(oi_skew_val))
            _ps_skew_lids.append(_ps_skew_lid)
        if len(oi_skew_hist) > 30: oi_skew_hist = oi_skew_hist[-30:]; _ps_skew_lids = _ps_skew_lids[-30:]
        st.session_state["_flow_skew_oi_hist"]     = oi_skew_hist
        st.session_state["_flow_skew_oi_load_ids"] = _ps_skew_lids
        # FIX: tanh of raw value when history < 3 (z-score is 0 with single sample)
        if len(oi_skew_hist) >= 3:
            oi_skew_z = _zscore_clamp(oi_skew_hist, float(oi_skew_val), clamp=2.0) / 2.0
        else:
            oi_skew_z = math.tanh(oi_skew_val * 3.0)
    else:
        pcr_level_z = 0.0; oi_skew_z = 0.0; pcr_pct = 0.5; pcr_v = 1.0

    # Max pain proximity: normalised by expected move
    _T_safe  = max(T, 1.0 / CFG["ann_days"])   # guard: T is always ≥ 1/252, but be explicit
    em_price = float(oi_d.get("atm_straddle", atm_iv * spot * math.sqrt(_T_safe * 2 / math.pi)) or 1)
    em_pct   = float(oi_d.get("exp_move_pct", round(em_price / spot * 100, 2)) if oi_d else 0)
    mp       = float(oi_d.get("max_pain", spot) or spot)
    mp_dist_em = (spot - mp) / (em_price + 1e-9)   # in units of expected move
    # Max pain gravity: if spot >> max pain, expect mean reversion down (bearish)
    mp_z = max(-1.0, min(1.0, -mp_dist_em * _calib("mp_gravity")))

    _pw2 = _calib_vec("positioning_pcr_vs_oi_vs_mp")
    positioning_score = _pw2[0] * pcr_level_z + _pw2[1] * oi_skew_z + _pw2[2] * mp_z
    positioning_score = max(-1.0, min(1.0, positioning_score))
    # Record sub-scores for calibration
    _record_if_load("_calib_pcr_level_hist", pcr_level_z)
    _record_if_load("_calib_oi_skew_hist",   oi_skew_z)
    _record_if_load("_calib_mp_z_hist",      mp_z)

    fs["positioning_score"] = round(positioning_score, 4)
    fs["pcr_level_z"]       = round(pcr_level_z, 4)
    fs["oi_skew_z"]         = round(oi_skew_z, 4)
    fs["mp_z"]              = round(mp_z, 4)

    # ════════════════════════════════════════════════════════════════════════
    # FACTOR 3 — VOLATILITY REGIME  (weight 0.20) — LEADING/CONCURRENT
    # IV/HV ratio percentile: where current vol sits in history
    # Term structure slope: contango vs backwardation
    # Volatility direction: is IV expanding or compressing?
    # ════════════════════════════════════════════════════════════════════════
    hv_ref     = hv20 if hv20 and hv20 > 0.01 else CFG["hv_fallback"]
    iv_hv_r    = atm_iv / (hv_ref + 1e-9)
    sym_key    = st.session_state.get("opt_symbol", "")
    iv_hist    = st.session_state.get("opt_iv_history", {}).get(sym_key, [])
    # BUG FIX 3: _append_iv adds today's IV to iv_hist BEFORE compute_probabilistic_score
    # runs, so iv_hist[-1] == atm_iv.  Ranking today's value against a distribution
    # that already includes today produces a mild self-fulfilling look-ahead bias in the
    # vol-regime signal.  Fix: use iv_hist[:-1] (all prior sessions) as the reference
    # distribution, then rank today's atm_iv against it.  Falls back to full history
    # when fewer than 2 entries exist (cold-start safe).
    _iv_hist_ref = iv_hist[:-1] if len(iv_hist) >= 2 else iv_hist
    iv_hv_hist   = [iv / hv_ref for iv in _iv_hist_ref if iv > 0] if _iv_hist_ref else []

    if len(iv_hv_hist) >= 5:
        iv_hv_pct = _percentile_score(iv_hv_hist, iv_hv_r)
    else:
        # BUG FIX (IV/HV Key Mismatch): unified key — see Fix B2 comment above.
        iv_hv_pct = (_normalise_to_signal(iv_hv_r, "_calib_iv_hv_ratio_hist") + 1.0) / 2.0

    # BUG FIX (Vol Regime Dampening Direction):
    # Old code: vol_regime_z = -(2*iv_hv_pct - 1) * damp
    # → expensive IV (high iv_hv_pct) produced a BEARISH directional lean.
    # This directly contradicts the strategy engine: when IV is expensive the model
    # recommends "SELL vol" (short strangle / iron condor), strategies that benefit
    # from the same high-IV environment. Applying a bearish directional dampening
    # simultaneously is internally contradictory.
    #
    # Root cause: IV LEVEL is the wrong input for a directional signal.
    # High IV does NOT predict direction — it predicts vol contraction, not price move.
    #
    # Fix: replace IV level with IV MOMENTUM (direction of recent IV change).
    # Rising IV = institutions hedging / buying protection = bearish for spot (correct).
    # Falling IV = fear unwind = bullish for spot (correct).
    # This is directionally coherent AND consistent with the short-vol strategy when IV
    # is high but FALLING (ideal short-vol entry: rich premium starting to compress).
    #
    # Implementation: use the 3-session rate of change of IV/HV ratio, z-scored against
    # its own history, as the directional component. The vol_edge (buy/sell vol) remains
    # driven by IV/HV LEVEL (iv_hv_pct) separately — the two signals are now decoupled.
    _iv_momentum_z = 0.0
    if len(iv_hv_hist) >= 4:
        # 3-period rate of change of IV/HV ratio, normalised
        _iv_roc = (iv_hv_r - iv_hv_hist[-4]) / (abs(iv_hv_hist[-4]) + 1e-9)
        _iv_momentum_z = _normalise_to_signal(_iv_roc, "_calib_iv_momentum_hist")
        # Rising IV/HV → bearish lean on spot direction (negative sign)
        _iv_momentum_z = max(-1.0, min(1.0, -_iv_momentum_z))
    elif len(iv_hv_hist) >= 2:
        _iv_roc = (iv_hv_r - iv_hv_hist[-2]) / (abs(iv_hv_hist[-2]) + 1e-9)
        _iv_momentum_z = max(-1.0, min(1.0, -math.tanh(_iv_roc * 3.0)))

    _record_if_load("_calib_iv_momentum_hist", -_iv_momentum_z)   # store unsigned roc
    vol_regime_z = _iv_momentum_z * _calib("vol_regime_damp")
    _record_if_load("_calib_vol_regime_z_hist", vol_regime_z)

    # Term structure z-score: slope × calibrated scale factor
    # Backwardation (near IV > far IV, ts_slope < 0) = stress = bearish for spot.
    # Contango (far IV > near IV, ts_slope > 0) = calm = bullish for spot.
    ts_data      = st.session_state.get("opt_multi_expiry", [])
    term_slope_z = 0.0
    if len(ts_data) >= 2:
        iv_near  = float(ts_data[0].get("atm_iv", atm_iv))
        iv_far   = float(ts_data[1].get("atm_iv", atm_iv))
        ts_slope = iv_far - iv_near
        _record_if_load("_calib_ts_slope_raw_hist", ts_slope)
        st.session_state["opt_term_slope"] = ts_slope   # persist for detect_market_regime
        term_slope_z = max(-1.0, min(1.0, ts_slope * _calib("ts_slope_scale")))

    _record_if_load("_calib_term_slope_z_hist", term_slope_z)
    _vr_w = _calib("vol_bb_vs_atr")   # blend: IV momentum vs term slope
    vol_regime_score = _vr_w * vol_regime_z + (1.0 - _vr_w) * term_slope_z
    vol_regime_score = max(-1.0, min(1.0, vol_regime_score))

    fs["vol_regime_score"]  = round(vol_regime_score, 4)
    fs["iv_hv_pct"]         = round(iv_hv_pct, 4)
    fs["vol_regime_z"]      = round(vol_regime_z, 4)
    fs["iv_momentum_z"]     = round(_iv_momentum_z, 4)
    fs["term_slope_z"]      = round(term_slope_z, 4)

    # ════════════════════════════════════════════════════════════════════════
    # FACTOR 4 — RELATIVE STRENGTH  (weight 0.15) — CONFIRMING (medium lag)
    # Stock return vs Nifty over 20 days
    # z-score of RS ratio in its own history
    # ════════════════════════════════════════════════════════════════════════
    rs_data = st.session_state.get("opt_rs_nifty", None)
    if rs_data and isinstance(rs_data, dict):
        rs_ratio    = rs_data.get("rs_ratio", 1.0)
        rs_slope    = rs_data.get("rs_slope", 0.0)
        rs_z        = _normalise_to_signal(rs_ratio, "_calib_rs_z_raw_hist")
        rs_slope_z  = _normalise_to_signal(rs_slope, "_calib_rs_slope_raw_hist")
        _rs_lv_w    = _calib("rs_level_vs_slope")
        rs_score    = _rs_lv_w * rs_z + (1.0 - _rs_lv_w) * rs_slope_z
        _record_if_load("_calib_rs_z_hist",     rs_z)
        _record_if_load("_calib_rs_slope_hist", rs_slope_z)
    else:
        rs_score = 0.0; rs_z = 0.0; rs_slope_z = 0.0

    fs["rs_score"]   = round(max(-1.0, min(1.0, rs_score)), 4)
    fs["rs_z"]       = round(rs_z, 4)

    # ════════════════════════════════════════════════════════════════════════
    # FACTOR 5 — TREND  (weight 0.10) — CONFIRMING (most lagging)
    # EMA structure: where price sits relative to moving averages
    # ADX: trend strength percentile
    # RSI and MACD contribute a small fraction WITHIN this factor only
    # ════════════════════════════════════════════════════════════════════════
    trend_z = 0.0; atrv = _atr_seed(spot); rsi_v = 50.0

    if len(c_series) >= 50:
        # EMA structure
        e20  = c_series.ewm(span=20, adjust=False).mean()
        e50  = c_series.ewm(span=50, adjust=False).mean()
        e200 = c_series.ewm(span=200, adjust=False).mean()
        tr   = pd.concat([h_series - l_series,
                          (h_series - c_series.shift(1)).abs(),
                          (l_series - c_series.shift(1)).abs()], axis=1).max(axis=1)
        atr14 = tr.ewm(alpha=1/14, adjust=False).mean()
        atrv  = float(atr14.iloc[-1])
        if spot > 0: _record_if_load("_calib_atr_pct_hist", atrv / spot)
        e20v  = float(e20.iloc[-1])
        e50v  = float(e50.iloc[-1])
        e200v = float(e200.iloc[-1]) if len(c_series) >= 200 else spot
        ema_checks = [spot > e20v, spot > e50v, e20v > e50v, spot > e200v, e50v > e200v]
        ema_score  = (sum(ema_checks) / len(ema_checks)) * 2 - 1   # [-1, +1]

        # ADX — trend strength percentile × direction
        up_m  = h_series - h_series.shift(1)
        dn_m  = l_series.shift(1) - l_series
        dmp   = pd.Series(np.where((up_m > dn_m) & (up_m > 0), up_m, 0.0), index=c_series.index)
        dmm   = pd.Series(np.where((dn_m > up_m) & (dn_m > 0), dn_m, 0.0), index=c_series.index)
        di_p  = 100 * dmp.ewm(alpha=1/14, adjust=False).mean() / (atr14 + 1e-9)
        di_m  = 100 * dmm.ewm(alpha=1/14, adjust=False).mean() / (atr14 + 1e-9)
        dx    = 100 * (di_p - di_m).abs() / (di_p + di_m + 1e-9)
        adx_s = dx.ewm(alpha=1/14, adjust=False).mean().dropna()
        adx_v = float(adx_s.iloc[-1])
        adx_d = 1.0 if float(di_p.iloc[-1]) > float(di_m.iloc[-1]) else -1.0
        adx_pct = _percentile_score(adx_s.values, adx_v)

        # RSI — small contribution within trend factor (not a primary driver)
        if len(c_series) >= 20:
            delta  = c_series.diff()
            gain   = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
            loss   = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
            rs_raw = gain / loss.replace(0, float("nan"))
            rsi_s  = (100 - 100 / (1 + rs_raw)).fillna(
                gain.apply(lambda g: 100.0 if g > 0 else 0.0))
            rsi_v  = float(rsi_s.iloc[-1])
            rsi_z  = _zscore_clamp(rsi_s.tail(252).values, rsi_v, clamp=3.0) / 3.0
        else:
            rsi_z = 0.0

        # Trend score: EMA vs (ADX + RSI) with calibrated blend
        # Within non-EMA portion: ADX gets 3× the weight of RSI (structural vs momentum)
        _ema_w2       = _calib("trend_ema_vs_adx")
        _non_ema      = 1.0 - _ema_w2
        _adx_frac     = _calib("adx_vs_rsi_within_trend")
        trend_z = (_ema_w2 * ema_score
                   + _non_ema * _adx_frac       * (adx_pct * adx_d)
                   + _non_ema * (1.0 - _adx_frac) * rsi_z)
        _record_if_load("_calib_ema_score_hist", ema_score)
        _record_if_load("_calib_adx_score_hist", adx_pct * adx_d)
        _record_if_load("_calib_rsi_trend_hist", rsi_z)
    else:
        e20v = e50v = e200v = spot

    trend_z = max(-1.0, min(1.0, trend_z))
    fs["trend_z"]  = round(trend_z, 4)
    fs["rsi_v"]    = round(rsi_v, 1)

    # ════════════════════════════════════════════════════════════════════════
    # FINAL SIGNAL SCORE — data-driven factor weights (Improvement #4)
    # Instead of fixed weights from CFG["factor_weights"], compute each
    # factor weight as abs(correlation(factor_score, future_return_N_days))
    # normalised to sum=1.  Falls back to CFG weights when < 10 observations.
    # ════════════════════════════════════════════════════════════════════════

    # Raw (ungated) factor scores — stored in history BEFORE regime gating.
    # History must reflect true signal strength, not gated-zero values.
    # Gating is applied only to the final composite, not to the learning signal.
    _factor_scores_raw = {
        "flow":         fs["flow_score"],
        "positioning":  fs["positioning_score"],
        "vol_regime":   fs["vol_regime_score"],
        "rel_strength": fs["rs_score"],
        "trend":        fs["trend_z"],
    }
    _fw_cfg = CFG["factor_weights"]   # uniform prior fallback (superseded by rank-based weights)

    # Update rolling factor score history from UNGATED scores (only on genuine Load)
    _fhist = st.session_state.get("opt_factor_hist", {})
    _fhist_load_id = st.session_state.get("opt_load_id", 0)
    _fhist_last_id = st.session_state.get("_fhist_last_load_id", -1)
    if _fhist_load_id != _fhist_last_id:
        st.session_state["_fhist_last_load_id"] = _fhist_load_id
        for _fname, _fval in _factor_scores_raw.items():
            _fhist.setdefault(_fname, []).append(float(_fval))
            _fhist[_fname] = _fhist[_fname][-252:]
        st.session_state["opt_factor_hist"] = _fhist

    # Regime-first gating: suppress signals not informative in this regime.
    # Applied AFTER history recording so gated zeros never corrupt the learning signal.
    _regime_now   = st.session_state.get("opt_regime_label", _REGIME_TRANSITION)
    _regime_conf  = st.session_state.get("opt_regime_confidence", 0.5)
    _factor_scores_now = _gate_signals_by_regime(_factor_scores_raw, _regime_now, _regime_conf)

    # Weights computed from ungated scores (true signal performance)
    fw = _rank_based_factor_weights(_factor_scores_raw, _fw_cfg)


    # BUG FIX (abs(corr) sign): retrieve per-factor correlation signs persisted above.
    # A factor with negative historical correlation is contrarian — its score must be
    # inverted before blending so it contributes in the correct predictive direction.
    # Default sign = +1 (confirming) when calibration hasn't run yet (< 10 obs).
    _fc_signs = st.session_state.get("_factor_corr_signs", {})
    _fs = lambda fname: _fc_signs.get(fname, 1)   # +1 confirming, -1 contrarian

    # Rank-aggregation composite (Fix 9: replaces linear weighted sum).
    # Each factor score is first mapped to a percentile rank within its own
    # rolling history (→ comparable [-1,+1] rank signal), then weighted.
    # This neutralises scale differences and prevents any single fat signal
    # from dominating the composite.
    # _fhist_ps: use the same factor history (already updated with ungated scores above)
    _fhist_ps = st.session_state.get("opt_factor_hist", {})
    def _rank_score(fname, raw_val):
        h = np.array(_fhist_ps.get(fname, []), dtype=float)
        if len(h) < 10:
            return float(raw_val)  # cold start: pass through
        pct = float((h < raw_val).mean())  # 0..1
        return max(-1.0, min(1.0, (pct - 0.5) * 2.0))  # → [-1,+1]

    # Rank from ungated scores (true signal percentile), then apply gate multiplier
    _fscores_ranked = {}
    for _fname_r in _factor_scores_raw:
        _raw_r  = _factor_scores_raw[_fname_r]
        _gate_r = (_factor_scores_now[_fname_r] / (_raw_r + 1e-9)
                   if abs(_raw_r) > 1e-9 else 0.0)
        _gate_r = max(0.0, min(1.0, abs(_gate_r)))  # gate ratio 0..1
        _ranked_r = _rank_score(_fname_r, _raw_r)   # rank based on ungated history
        _fscores_ranked[_fname_r] = _ranked_r * _gate_r  # apply gate to ranked score

    raw_score = sum(
        fw.get(k, 0.0) * _fs(k) * _fscores_ranked.get(k, 0.0)
        for k in _factor_scores_raw
    )
    raw_score = max(-1.0, min(1.0, raw_score))

    # Map to probability via logistic with calibrated sharpness
    _sharpness = _calib("logistic_sharpness")
    prob_up   = _logistic(raw_score * _sharpness)
    prob_down = 1.0 - prob_up
    # raw_score recorded in Load-guarded PART 8 block only (FIX A).

    # ── Intraday blend — integrate live 5-min signals when market is open ─────
    # The intraday score is computed from Upstox live candles and blended into
    # the final direction. Blend weight is calibrated (default 20%).
    # When intraday data is unavailable (pre-market, after-hours, weekend),
    # the blend weight is set to 0 and the factor model drives entirely.
    _intra_data  = st.session_state.get("opt_intraday_signals", {})
    _intra_score = float(_intra_data.get("intraday_score", 0.0))
    _intra_avail = bool(_intra_data.get("intraday_available", False))
    _intra_w     = _calib("intra_blend_weight") if _intra_avail else 0.0

    # Only blend intraday when the signal is meaningfully above its own noise floor.
    # "Meaningful" = abs(score) exceeds the 30th percentile of its own history.
    # Cold-start (< 3 observations): always blend if intraday is available.
    _intra_hist_so_far = st.session_state.get("_calib_intraday_score_hist", [])
    if len(_intra_hist_so_far) >= 3:
        _intra_noise_floor = float(np.percentile(np.abs(_intra_hist_so_far), 30))
    else:
        _intra_noise_floor = 0.0   # no history yet → always blend when available
    if _intra_avail and abs(_intra_score) > _intra_noise_floor:
        # Blend: (1 − intra_w) × factor_raw + intra_w × intra_score
        blended_raw = (1.0 - _intra_w) * raw_score + _intra_w * _intra_score
        blended_raw = max(-1.0, min(1.0, blended_raw))
        prob_up   = _logistic(blended_raw * _sharpness)
        prob_down = 1.0 - prob_up
        _record_if_load("_calib_intraday_score_hist", _intra_score)

    # ── Monte Carlo direction signal (Improvement #1) ─────────────────────────
    # Run a lightweight MC to get a distribution-based directional score.
    # Uses ATM IV from IV surface and real-world drift; reuses existing helpers.
    # This signal is later blended (50/50) with the factor-model prob in ev_rank_strategies.
    _mc_direction = 0.0   # default neutral
    _mc_expected_move = em_price   # default to straddle-based EM
    try:
        _ohlcv_ps = st.session_state.get("opt_ohlcv_df", None)
        _r_ps = r
        _q_ps = q
        _T_ps = T
        if _ohlcv_ps is not None and not _ohlcv_ps.empty and _T_ps > 0 and atm_iv > 0:
            _iv_surf_ps = build_iv_surface(
                chain_df if (chain_df is not None and not chain_df.empty) else None,
                spot, atm_iv)
            _mu_ps    = _estimate_real_world_drift(_ohlcv_ps, _r_ps, _q_ps)
            _seed_ps  = int(abs(spot * 100 + _T_ps * 1e5)) % (2**31)
            _rng_ps   = np.random.default_rng(_seed_ps)
            _n_ps     = 5000   # lighter simulation for speed
            _Z_ps     = _rng_ps.standard_normal(_n_ps)
            _sigma_ps = _iv_surf_ps(spot)
            _ST_ps    = spot * np.exp((_mu_ps - 0.5 * _sigma_ps**2) * _T_ps
                                      + _sigma_ps * math.sqrt(_T_ps) * _Z_ps)
            _mc_prob_up_ps   = float((_ST_ps > spot).mean())
            _mc_prob_down_ps = float((_ST_ps < spot).mean())
            _mc_direction    = _mc_prob_up_ps - _mc_prob_down_ps   # ∈ [-1, +1]
            _mc_expected_move = float(np.std(_ST_ps))              # std of terminal prices
    except Exception:
        pass

    # FIX10: MC now conditions core probability (not just EV)
    if abs(_mc_direction) > 0.05:
        _mc_prob = max(0.10, min(0.90, 0.50 + _mc_direction * 0.40))
        _mc_w    = max(0.15, min(0.45, _calib("mc_blend")))
        prob_up  = max(0.05, min(0.95, (1.0 - _mc_w) * prob_up + _mc_w * _mc_prob))
        prob_down = 1.0 - prob_up

    # ── STEP 4: Implied probability from IV ──────────────────────────────────
    # expected_move_iv = spot * atm_iv * sqrt(DTE/252)  (spec formula)
    _dte_days = max(T * CFG["ann_days"], 1.0)
    em_iv_spec = spot * atm_iv * math.sqrt(_dte_days / 252.0)   # spec: sqrt(DTE/252)
    em_iv_pct  = round(em_iv_spec / (spot + 1e-9) * 100, 2)

    # Implied prob of up move from options pricing
    # implied_prob_up ≈ 0.50 + (expected_move / spot) / 2
    implied_prob_up = min(0.90, max(0.10, 0.50 + (em_iv_spec / (spot + 1e-9)) / 2.0))

    # ── PART 4: Model expected move ───────────────────────────────────────────
    # FIX: Replace heuristic (0.5 + |score|) with empirically grounded estimate.
    # Use the ATR-based 1-day move scaled to DTE, then bias by directional conviction.
    # When OHLCV is available: use realised volatility (HV20) scaled to DTE.
    # This is the same logic used for the straddle pricing but uses realised vol, not IV.
    # model_move = spot * HV20 * sqrt(DTE/252) = HV-implied move for the holding period
    # Then scale by (1 + 0.5 * |score|) — a moderate conviction multiplier (max 1.5x at |score|=1).
    # This is far better than implied*1.5 because: (a) it uses realised vol not IV,
    # (b) the multiplier is bounded and principled, (c) |score| captures model conviction.
    _hv_for_move = hv20 if (hv20 and hv20 > 0.01) else CFG["hv_fallback"]
    _hv_move_base  = spot * _hv_for_move * math.sqrt(max(T * CFG["ann_days"], 1) / 252.0)
    _final_score_for_move = raw_score   # best available final score
    # Conviction multiplier: 1.0 (neutral) → 1.5 (max conviction). Bounded, not explosive.
    _conviction_mult = 1.0 + 0.5 * abs(_final_score_for_move)
    model_move     = _hv_move_base * _conviction_mult
    model_move_pct = round(model_move / (spot + 1e-9) * 100, 2)

    # ── PART 5: Edge detection ────────────────────────────────────────────────
    # Direction edge: prob_up - 0.50 (simpler, absolute measure of directional conviction)
    direction_edge  = round(prob_up - 0.50, 4)
    # Move edge: model_move - expected_move (positive = expect bigger move than IV implies)
    move_edge       = round(model_move - em_iv_spec, 2)
    move_edge_pct   = round(move_edge / (spot + 1e-9) * 100, 2)
    # Vol edge: use IV/HV percentile rank from live history, NOT model_move heuristic.
    # iv_hv_pct is in [0,1]: 0=cheapest IV ever, 1=richest IV ever.
    # SELL vol when IV is rich (top quartile of history); BUY when cheap (bottom quartile).
    # Fall back to move_edge signal only when IV history is too thin to rank (< 5 obs).
    _iv_hv_hist = _get_hist("_calib_iv_hv_ratio_hist") or []
    if len(_iv_hv_hist) >= 5:
        _iv_hv_now    = atm_iv / (hv20 + 1e-9) if hv20 and hv20 > 0.01 else 1.0
        _iv_hv_pct_rv = _percentile_score(_iv_hv_hist, _iv_hv_now)   # 0–1
        if _iv_hv_pct_rv >= _adaptive_threshold("iv_hv", CFG["iv_hv_pct_sell"], percentile=70.0) / 100.0:
            vol_edge = "SELL"   # IV expensive vs history → sell premium
        elif _iv_hv_pct_rv <= CFG["iv_hv_pct_buy"] / 100.0:
            vol_edge = "BUY"    # IV cheap vs history → buy vol
        else:
            vol_edge = "NEUTRAL"
    else:
        # Cold-start: use move_edge but label NEUTRAL unless clearly one-sided
        vol_edge = ("BUY" if move_edge_pct > 0.5
                    else "SELL" if move_edge_pct < -0.5
                    else "NEUTRAL")
    # Record IV/HV ratio for future ranking
    _record_if_load("_calib_iv_hv_ratio_hist", atm_iv / (hv20 + 1e-9) if hv20 and hv20 > 0.01 else 1.0)

    edge_label = ("Bullish Edge" if direction_edge > 0.05
                  else "Bearish Edge" if direction_edge < -0.05
                  else "No Edge")

    # ── PART 3: Signal strength label ────────────────────────────────────────
    _pu_for_label = prob_up
    if _pu_for_label > 0.75:   signal_strength = "Very Strong Bullish"
    elif _pu_for_label > 0.65: signal_strength = "Strong Bullish"
    elif _pu_for_label > 0.60: signal_strength = "Moderate Bullish"
    elif _pu_for_label > 0.55: signal_strength = "Weak Bullish"
    elif _pu_for_label < 0.25: signal_strength = "Very Strong Bearish"
    elif _pu_for_label < 0.35: signal_strength = "Strong Bearish"
    elif _pu_for_label < 0.40: signal_strength = "Moderate Bearish"
    elif _pu_for_label < 0.45: signal_strength = "Weak Bearish"
    else:                      signal_strength = "No Edge"

    # ── PART 8: Store dedicated calibration histories ─────────────────────────
    # FIX A: Only record on genuine Load click, not every Streamlit re-render.
    _cur_load_id = st.session_state.get("opt_load_id", 0)
    _last_rec_id = st.session_state.get("_last_recorded_load_id", -1)
    if _cur_load_id != _last_rec_id:
        st.session_state["_last_recorded_load_id"] = _cur_load_id
        _record("_calib_prob_up_hist", prob_up)
        _record("_calib_raw_score_hist", raw_score)
    # _calib_actual_up_hist and _calib_move_vs_iv_hist filled in _ingest_resolved_outcomes

    return {
        "raw_score":         round(raw_score, 4),
        "prob_up":           round(prob_up, 4),
        "prob_down":         round(prob_down, 4),
        "feature_scores":    fs,
        "expected_move":     round(_mc_expected_move, 2),
        "expected_move_pct": round(_mc_expected_move / spot * 100, 2) if spot > 0 else round(em_pct, 2),
        "iv_hv_pct":         round(iv_hv_pct, 4),
        "pcr_pct":           round(pcr_pct, 4),
        "rsi":               round(rsi_v, 1),
        "atr":               round(atrv, 2),
        "flow_magnitude":    round(flow_mag, 3),
        "factor_weights":    {k: round(v, 4) for k, v in fw.items()},
        "mc_direction":      round(_mc_direction, 4),
        "mc_expected_move":  round(_mc_expected_move, 2),
        # Parts 3-5
        "implied_prob_up":   round(implied_prob_up, 4),
        "implied_move_pct":  em_iv_pct,
        "model_move_pct":    model_move_pct,
        "direction_edge":    direction_edge,
        "move_edge_pct":     move_edge_pct,
        "vol_edge":          vol_edge,
        "edge_label":        edge_label,
        "signal_strength":   signal_strength,
        "final_score":       round(raw_score, 4),   # alias for clarity in Decision Panel
    }


# ============================================================
# STRATEGY UNIVERSE — canonical definitions
# ============================================================

def _build_strategy_universe(spot, atm, step, T, r, q, atm_iv,
                              bs_call_fn, bs_put_fn, chain_df,
                              front_iv=None, back_iv=None):
    """
    Build a COMPLETE strategy universe by scanning ALL real liquid strikes from
    chain_df — not just ATM±1 step. Every directional spread, condor, strangle,
    and butterfly is generated across multiple real strike offsets so the EV
    ranker can pick the genuinely best configuration, not a hardcoded one.

    Strike selection pipeline:
      1. Pull all strikes from chain_df with non-zero OI and LTP on both sides.
      2. Compute per-strike delta using BS to identify 0.50/0.30/0.20/0.15 delta
         strikes for calls and puts.
      3. Generate strategies at each meaningful offset combination.
      4. Deduplicate by (strategy_template, short_strike_call, short_strike_put).
    Falls back to ATM±step if chain_df is empty or has no live data.
    """
    _c  = bs_call_fn
    _p  = bs_put_fn
    sv  = float(step)   # kept as fallback step only

    def _leg(opt, strike, action, qty=1):
        px = (_c(float(strike)) if opt == "CE" else _p(float(strike)))
        # GAP-2: carry bid/ask from chain so ev_rank_strategies can deduct slippage.
        # Fall back to mid±0 if chain data not available for this strike.
        _k = float(strike)
        if opt == "CE":
            _bid = float(strike_bid_ce.get(_k, px))
            _ask = float(strike_ask_ce.get(_k, px))
        else:
            _bid = float(strike_bid_pe.get(_k, px))
            _ask = float(strike_ask_pe.get(_k, px))
        # If bid/ask not populated yet (fallback), use a 0.5% spread heuristic
        if _bid <= 0 or _ask <= 0 or _ask < _bid:
            _half = max(px * 0.005, 0.05)
            _bid, _ask = max(px - _half, 0.05), px + _half
        return {"opt": opt, "strike": _k,
                "premium": float(max(px, 0.05)),
                "bid": round(_bid, 2), "ask": round(_ask, 2),
                "action": action, "qty": qty}

    universe = []
    seen_keys = set()   # dedup key: (template_name, short_c_strike, short_p_strike)

    def add(name, type_, legs, display_legs, max_risk, max_reward,
            ideal_lo, ideal_hi, short_strike=None, term_slope=None, dedup_key=None):
        dk = dedup_key or (name, short_strike)
        if dk in seen_keys:
            return
        seen_keys.add(dk)
        universe.append({
            "name":          name,
            "type":          type_,
            "legs":          legs,
            "display_legs":  display_legs,
            "max_risk":      float(max_risk),
            "max_reward":    float(max_reward),
            "ideal_dte_lo":  ideal_lo,
            "ideal_dte_hi":  ideal_hi,
            "short_strike":  short_strike,
            "term_slope":    term_slope,
        })

    # ── STEP 1: Extract real liquid strikes from chain_df ────────────────────
    # A strike is "liquid" if it has non-zero OI on both CE and PE sides.
    # We use the actual chain strikes (not synthetic grid) for all multi-leg strats.
    chain_strikes_all = []   # list of float strikes, sorted ascending
    strike_ltp_ce = {}       # strike → CE_LTP (for premium lookup)
    strike_ltp_pe = {}       # strike → PE_LTP
    strike_oi_ce  = {}
    strike_oi_pe  = {}
    # GAP-2: bid/ask dicts for slippage computation in ev_rank_strategies
    strike_bid_ce = {}
    strike_ask_ce = {}
    strike_bid_pe = {}
    strike_ask_pe = {}

    if chain_df is not None and not chain_df.empty and "CE_LTP" in chain_df.columns:
        _cdf = chain_df.copy()
        # Use strikes where both sides have positive LTP (live data indicator)
        _liq_mask = (_cdf["CE_LTP"] > 0) & (_cdf["PE_LTP"] > 0)
        _liq_df   = _cdf[_liq_mask].sort_values("Strike")
        for _, row in _liq_df.iterrows():
            k = float(row["Strike"])
            chain_strikes_all.append(k)
            strike_ltp_ce[k] = float(row.get("CE_LTP", 0) or 0)
            strike_ltp_pe[k] = float(row.get("PE_LTP", 0) or 0)
            strike_oi_ce[k]  = float(row.get("CE_OI", 0) or 0)
            strike_oi_pe[k]  = float(row.get("PE_OI", 0) or 0)
            # GAP-2: capture bid/ask from chain
            strike_bid_ce[k] = float(row.get("CE_Bid", 0) or 0)
            strike_ask_ce[k] = float(row.get("CE_Ask", 0) or 0)
            strike_bid_pe[k] = float(row.get("PE_Bid", 0) or 0)
            strike_ask_pe[k] = float(row.get("PE_Ask", 0) or 0)

    # Fallback: use synthetic grid around ATM when chain_df is not live
    if len(chain_strikes_all) < 4:
        chain_strikes_all = [atm + i * sv for i in range(-8, 9)]
        for k in chain_strikes_all:
            strike_ltp_ce[k] = _c(k)
            strike_ltp_pe[k] = _p(k)
            strike_oi_ce[k]  = 1
            strike_oi_pe[k]  = 1
            # GAP-2: synthetic 0.5% half-spread for fallback grid
            _ce_mid = _c(k); _pe_mid = _p(k)
            _hsp_ce = max(_ce_mid * 0.005, 0.05); _hsp_pe = max(_pe_mid * 0.005, 0.05)
            strike_bid_ce[k] = max(_ce_mid - _hsp_ce, 0.05); strike_ask_ce[k] = _ce_mid + _hsp_ce
            strike_bid_pe[k] = max(_pe_mid - _hsp_pe, 0.05); strike_ask_pe[k] = _pe_mid + _hsp_pe

    all_ks = sorted(chain_strikes_all)

    # ── STEP 2: Compute delta for each strike using BS ───────────────────────
    # delta_ce[k] ≈ N(d1) for call, delta_pe[k] ≈ N(d1)-1 for put
    def _delta_ce(k):
        if T <= 0 or atm_iv <= 0:
            return 0.5 if k <= spot else 0.1
        try:
            F  = spot * math.exp((r - q) * T)
            d1 = (math.log(max(F / k, 1e-9)) + 0.5 * atm_iv**2 * T) / (atm_iv * math.sqrt(T))
            return float(_ncdf(d1))
        except Exception:
            return 0.5

    def _delta_pe(k):
        return _delta_ce(k) - 1.0   # put delta = call delta - 1

    # ── STEP 3: Find nearest strikes at target deltas ────────────────────────
    # Target call deltas for OTM calls (positive): 0.40, 0.30, 0.20, 0.15
    # Target put deltas for OTM puts (negative abs): 0.40, 0.30, 0.20, 0.15
    # ATM = nearest strike to spot from chain

    def _nearest_k_above(target_delta_ce, min_k=None):
        """Find call strike above ATM with CE delta nearest to target_delta_ce."""
        candidates = [k for k in all_ks if k > (min_k or spot)]
        if not candidates:
            return atm + sv
        return min(candidates, key=lambda k: abs(_delta_ce(k) - target_delta_ce))

    def _nearest_k_below(target_abs_delta_pe, max_k=None):
        """Find put strike below ATM with |PE delta| nearest to target."""
        candidates = [k for k in all_ks if k < (max_k or spot)]
        if not candidates:
            return atm - sv
        return min(candidates, key=lambda k: abs(abs(_delta_pe(k)) - target_abs_delta_pe))

    # ATM from chain
    _atm_chain = min(all_ks, key=lambda k: abs(k - spot)) if all_ks else atm

    # Key call strikes (OTM side)
    ks_ce_40 = _nearest_k_above(0.40)   # near ATM call (~0.40Δ)
    ks_ce_30 = _nearest_k_above(0.30)   # 1σ OTM call
    ks_ce_20 = _nearest_k_above(0.20)   # 1.5σ OTM call
    ks_ce_15 = _nearest_k_above(0.15)   # 2σ OTM call (wing)

    # Key put strikes (OTM side)
    ks_pe_40 = _nearest_k_below(0.40)   # near ATM put (~0.40Δ)
    ks_pe_30 = _nearest_k_below(0.30)
    ks_pe_20 = _nearest_k_below(0.20)
    ks_pe_15 = _nearest_k_below(0.15)

    # For wing protection: one step further out than 20Δ
    ks_ce_wing = _nearest_k_above(0.10, min_k=ks_ce_20)
    ks_pe_wing = _nearest_k_below(0.10, max_k=ks_pe_20)

    # ATM premiums
    _atm_c = _c(_atm_chain); _atm_p = _p(_atm_chain)

    # ── STEP 4: DIRECTIONAL STRATEGIES at multiple offsets ───────────────────

    # Long ATM Call / Put (single leg)
    add("Long ATM Call", "debit_bull",
        [_leg("CE", _atm_chain, "buy")],
        f"BUY {_atm_chain:.0f} CE",
        _atm_c, 999, 15, 45)

    add("Long ATM Put", "debit_bear",
        [_leg("PE", _atm_chain, "buy")],
        f"BUY {_atm_chain:.0f} PE",
        _atm_p, 999, 15, 45)

    # Bull Call Spreads: buy ATM/near-ATM, sell at 30Δ and 20Δ offsets
    for _sell_k in sorted(set([ks_ce_40, ks_ce_30, ks_ce_20])):
        if _sell_k <= _atm_chain:
            continue
        _d = max(_atm_c - _c(_sell_k), 0.01)
        _w = _sell_k - _atm_chain
        add(f"Bull Call Spread {_sell_k:.0f}", "debit_bull",
            [_leg("CE", _atm_chain, "buy"), _leg("CE", _sell_k, "sell")],
            f"BUY {_atm_chain:.0f} CE + SELL {_sell_k:.0f} CE",
            _d, _w - _d, 15, 30, short_strike=_sell_k,
            dedup_key=("BCS", _atm_chain, _sell_k))

    # Bear Put Spreads: buy ATM/near-ATM, sell at 30Δ and 20Δ offsets
    for _sell_k in sorted(set([ks_pe_40, ks_pe_30, ks_pe_20]), reverse=True):
        if _sell_k >= _atm_chain:
            continue
        _d = max(_atm_p - _p(_sell_k), 0.01)
        _w = _atm_chain - _sell_k
        add(f"Bear Put Spread {_sell_k:.0f}", "debit_bear",
            [_leg("PE", _atm_chain, "buy"), _leg("PE", _sell_k, "sell")],
            f"BUY {_atm_chain:.0f} PE + SELL {_sell_k:.0f} PE",
            _d, _w - _d, 15, 30, short_strike=_sell_k,
            dedup_key=("BPS", _atm_chain, _sell_k))

    # Bear Call Spreads (credit): sell OTM call, buy further OTM call
    for _short_c, _long_c in [
        (ks_ce_30, ks_ce_20), (ks_ce_40, ks_ce_30),
        (ks_ce_20, ks_ce_15), (ks_ce_30, ks_ce_wing),
    ]:
        if _short_c >= _long_c or _short_c <= spot:
            continue
        _cr = max(_c(_short_c) - _c(_long_c), 0.01)
        _w  = _long_c - _short_c
        add(f"Bear Call Spread {_short_c:.0f}/{_long_c:.0f}", "credit_bear",
            [_leg("CE", _short_c, "sell"), _leg("CE", _long_c, "buy")],
            f"SELL {_short_c:.0f} CE + BUY {_long_c:.0f} CE",
            _w - _cr, _cr, 7, 21, short_strike=_short_c,
            dedup_key=("BEAR_CS", _short_c, _long_c))

    # Bull Put Spreads (credit): sell OTM put, buy further OTM put
    for _short_p, _long_p in [
        (ks_pe_30, ks_pe_20), (ks_pe_40, ks_pe_30),
        (ks_pe_20, ks_pe_15), (ks_pe_30, ks_pe_wing),
    ]:
        if _short_p <= _long_p or _short_p >= spot:
            continue
        _cr = max(_p(_short_p) - _p(_long_p), 0.01)
        _w  = _short_p - _long_p
        add(f"Bull Put Spread {_short_p:.0f}/{_long_p:.0f}", "credit_bull",
            [_leg("PE", _short_p, "sell"), _leg("PE", _long_p, "buy")],
            f"SELL {_short_p:.0f} PE + BUY {_long_p:.0f} PE",
            _w - _cr, _cr, 7, 21, short_strike=_short_p,
            dedup_key=("BULL_PS", _short_p, _long_p))

    # ── STEP 5: NEUTRAL / VOLATILITY STRATEGIES at multiple offsets ──────────

    # Short Straddle (always ATM)
    _strd = _atm_c + _atm_p
    add("Short Straddle", "credit_neutral",
        [_leg("CE", _atm_chain, "sell"), _leg("PE", _atm_chain, "sell")],
        f"SELL {_atm_chain:.0f} CE + SELL {_atm_chain:.0f} PE",
        999, _strd, 7, 14, short_strike=_atm_chain)

    add("Long Straddle", "debit_neutral",
        [_leg("CE", _atm_chain, "buy"), _leg("PE", _atm_chain, "buy")],
        f"BUY {_atm_chain:.0f} CE + BUY {_atm_chain:.0f} PE",
        _strd, 999, 30, 60)

    # Short Strangles: multiple width combinations using real delta strikes
    for _ce_k, _pe_k in [
        (ks_ce_30, ks_pe_30),   # 30Δ strangle (most common institutional)
        (ks_ce_20, ks_pe_20),   # 20Δ strangle (wider, safer)
        (ks_ce_40, ks_pe_40),   # tighter strangle
        (ks_ce_15, ks_pe_15),   # very wide strangle
    ]:
        if _ce_k <= spot or _pe_k >= spot or _ce_k == _pe_k:
            continue
        _ss_cr = _c(_ce_k) + _p(_pe_k)
        add(f"Short Strangle {_pe_k:.0f}/{_ce_k:.0f}", "credit_neutral",
            [_leg("PE", _pe_k, "sell"), _leg("CE", _ce_k, "sell")],
            f"SELL {_pe_k:.0f} PE + SELL {_ce_k:.0f} CE",
            999, _ss_cr, 7, 21, short_strike=_ce_k,
            dedup_key=("SHORT_STRNG", _pe_k, _ce_k))

    # Long Strangles
    for _ce_k, _pe_k in [
        (ks_ce_30, ks_pe_30),
        (ks_ce_20, ks_pe_20),
    ]:
        if _ce_k <= spot or _pe_k >= spot:
            continue
        _ls_db = _c(_ce_k) + _p(_pe_k)
        add(f"Long Strangle {_pe_k:.0f}/{_ce_k:.0f}", "debit_neutral",
            [_leg("CE", _ce_k, "buy"), _leg("PE", _pe_k, "buy")],
            f"BUY {_ce_k:.0f} CE + BUY {_pe_k:.0f} PE",
            _ls_db, 999, 30, 60,
            dedup_key=("LONG_STRNG", _pe_k, _ce_k))

    # Iron Condors: ALL combinations of (short_ce, short_pe) with wing protection
    # Short side at 30Δ, 20Δ; wings at 15Δ, 10Δ — producing multiple condors
    _ic_short_pairs = [
        (ks_ce_30, ks_pe_30, ks_ce_20, ks_pe_20),   # 30Δ short, 20Δ long wings
        (ks_ce_20, ks_pe_20, ks_ce_15, ks_pe_15),   # 20Δ short, 15Δ long wings
        (ks_ce_30, ks_pe_30, ks_ce_15, ks_pe_15),   # 30Δ short, wide wings
        (ks_ce_40, ks_pe_40, ks_ce_30, ks_pe_30),   # near-ATM short, 30Δ wings
        (ks_ce_20, ks_pe_20, ks_ce_wing, ks_pe_wing), # 20Δ short, 10Δ wings
    ]
    for (_sc, _sp, _lc, _lp) in _ic_short_pairs:
        if _sc <= spot or _sp >= spot or _lc <= _sc or _lp >= _sp:
            continue
        _ic_cr = ((_p(_sp) - _p(_lp)) + (_c(_sc) - _c(_lc)))
        _ic_w  = min(_sc - _lc, _sp - _lp)   # narrower wing is binding
        if _ic_cr <= 0 or _ic_w <= 0:
            continue
        add(f"Iron Condor {_sp:.0f}/{_sc:.0f}", "credit_neutral",
            [_leg("PE", _sp, "sell"), _leg("PE", _lp, "buy"),
             _leg("CE", _sc, "sell"), _leg("CE", _lc, "buy")],
            f"SELL {_sp:.0f}P/{_sc:.0f}C + BUY {_lp:.0f}P/{_lc:.0f}C",
            _ic_w - _ic_cr, _ic_cr, 14, 30, short_strike=_sc,
            dedup_key=("IC", _sp, _sc, _lp, _lc))

    # Iron Butterfly (always ATM short body)
    _ibf_cr = _atm_c + _atm_p - _c(ks_ce_30) - _p(ks_pe_30)
    if _ibf_cr > 0 and ks_ce_30 > _atm_chain and ks_pe_30 < _atm_chain:
        _ibf_w = min(ks_ce_30 - _atm_chain, _atm_chain - ks_pe_30)
        add("Iron Butterfly", "credit_neutral",
            [_leg("CE", _atm_chain, "sell"), _leg("PE", _atm_chain, "sell"),
             _leg("CE", ks_ce_30, "buy"), _leg("PE", ks_pe_30, "buy")],
            f"SELL {_atm_chain:.0f}C/{_atm_chain:.0f}P + BUY {ks_ce_30:.0f}C/{ks_pe_30:.0f}P",
            _ibf_w - _ibf_cr, _ibf_cr, 14, 21, short_strike=_atm_chain)

    # ATM Butterfly (call fly at multiple widths)
    for _fly_k in sorted(set([ks_ce_30, ks_ce_20])):
        if _fly_k <= _atm_chain:
            continue
        _fly_low  = _atm_chain - (_fly_k - _atm_chain)  # symmetric lower wing
        if _fly_low not in [round(k, 2) for k in all_ks]:
            # find nearest real strike below ATM at same distance
            _dist = _fly_k - _atm_chain
            _cands = [k for k in all_ks if k < _atm_chain]
            _fly_low = min(_cands, key=lambda k: abs((_atm_chain - k) - _dist)) if _cands else _atm_chain - _dist
        _bf_db = max(_c(_fly_low) - 2 * _atm_c + _c(_fly_k), 0.01)
        _bf_w  = _fly_k - _atm_chain
        add(f"ATM Butterfly {_fly_k:.0f}", "debit_neutral",
            [_leg("CE", _fly_low, "buy"), _leg("CE", _atm_chain, "sell", 2), _leg("CE", _fly_k, "buy")],
            f"BUY {_fly_low:.0f}C − 2×SELL {_atm_chain:.0f}C + BUY {_fly_k:.0f}C",
            _bf_db, _bf_w - _bf_db, 7, 21,
            dedup_key=("BFly", _atm_chain, _fly_k))

    # Calendar spread — only if term structure data available
    if front_iv is not None and back_iv is not None:
        _term_slope = back_iv - front_iv
        add("Calendar Spread", "debit_neutral",
            [_leg("CE", _atm_chain, "sell"), _leg("CE", _atm_chain, "buy")],
            f"SELL near {_atm_chain:.0f}CE + BUY far {_atm_chain:.0f}CE",
            _atm_c, 999, 7, 45, term_slope=_term_slope)

    return universe


def ev_rank_strategies(universe, spot, T, r, atm_iv, q,
                        prob_score, chain_df, ohlcv_df,
                        actual_dte, lot_size=1, simulations=None):
    """
    For every strategy in the universe:
      1. Run Monte Carlo to get POP and EV
      2. Compute Kelly fraction = EV / MaxLoss
      3. Compute DTE alignment (continuous, no threshold)
      4. Compute safety ratio (continuous, no threshold)
      5. Compute composite EV-adjusted score

    Selection criterion: highest EV adjusted for risk (not rules).
    Returns list of strategy dicts sorted by ev_score descending.
    """
    n_sims = simulations or CFG["pop_simulations"]
    # Defensive guards — never evaluate DataFrame truthiness with 'or'/'if df'
    _chain_safe = chain_df  if (chain_df  is not None and isinstance(chain_df, pd.DataFrame)
                                and not chain_df.empty) else None
    _ohlcv_safe = ohlcv_df  if (ohlcv_df  is not None and isinstance(ohlcv_df, pd.DataFrame)
                                 and not ohlcv_df.empty) else pd.DataFrame()
    iv_surf = build_iv_surface(_chain_safe, spot, atm_iv)
    mu      = _estimate_real_world_drift(_ohlcv_safe, r, q)
    _T_safe = max(T, 1.0 / CFG["ann_days"])   # guard against T=0 (expiry-day edge case)
    em      = prob_score.get("expected_move", atm_iv * spot * math.sqrt(_T_safe * 2 / math.pi))

    # Simulate terminal price distribution ONCE, reuse for all strategies
    # Seed mixes spot, T, and nanosecond timestamp so each Load press produces
    # genuinely independent Z paths — no more identical EV estimates every session.
    _seed   = int(abs(spot * 1000 + T * 1e6 + time.time_ns() % 1_000_000_000)) % (2**31)
    rng     = np.random.default_rng(_seed)
    Z       = rng.standard_normal(n_sims)
    sigma_sim = iv_surf(spot)

    # ── GAP-1 FIX: Merton Jump-Diffusion (replacing plain GBM) ──────────────
    # Plain GBM underestimates the probability of 3-4σ moves by 30-50% because
    # NSE options are priced with a volatility smile that reflects fat tails and
    # gap risk.  We add a Poisson-distributed jump component (Merton 1976):
    #   ln(S_T/S_0) = (μ - ½σ² - λ*κ)*T + σ√T*Z + Σ_{i=1}^{N(T)} Y_i
    # where N(T) ~ Poisson(λ*T) is the jump count and Y_i ~ N(m_j, v_j²) is
    # the log-jump size.
    # RECALIBRATION FIX: parameters now fitted per-symbol from OHLCV history
    # using _fit_jump_params() instead of hardcoded Nifty-only defaults.
    _jmp_lambda, _jmp_mu_j, _jmp_sig_j = _fit_jump_params(_ohlcv_safe, CFG["ann_days"])
    _compensator = _jmp_lambda * (math.exp(_jmp_mu_j + 0.5 * _jmp_sig_j**2) - 1.0)

    # Draw jump counts for each path from Poisson(λ*T)
    _N_jumps = rng.poisson(_jmp_lambda * T, size=n_sims)
    # Aggregate log-jumps: Σ Y_i ~ N(N*m_j, N*v_j²) — sum of N normals
    _log_jump_total = np.where(
        _N_jumps > 0,
        rng.normal(_N_jumps * _jmp_mu_j, np.sqrt(_N_jumps) * _jmp_sig_j),
        0.0
    )
    prices = spot * np.exp(
        (mu - 0.5 * sigma_sim**2 - _compensator) * T
        + sigma_sim * math.sqrt(T) * Z
        + _log_jump_total
    )

    # ── MC-derived direction signals (computed once over shared price paths) ──
    # Improvement #1: directional signal from MC terminal price distribution
    mc_prob_up   = float((prices > spot).mean())
    mc_prob_down = float((prices < spot).mean())
    mc_direction = mc_prob_up - mc_prob_down   # ∈ [-1, +1]

    # Improvement #2: expected move from MC std (replaces straddle-based EM)
    mc_expected_move = float(np.std(prices))   # in ₹, same units as spot

    # Final directional probability blends factor model (prob_score) with MC distribution
    # If prob_score already has a mc_direction (from compute_probabilistic_score), use it
    # directly; otherwise fall back to the prices computed from the full simulation above.
    _ps_mc_dir = prob_score.get("mc_direction", None)
    if _ps_mc_dir is not None:
        mc_direction = float(_ps_mc_dir)   # already computed, reuse

    factor_prob_up   = float(prob_score.get("prob_up",   0.5))
    factor_prob_down = float(prob_score.get("prob_down", 0.5))
    factor_direction = factor_prob_up - factor_prob_down   # ∈ [-1, +1]

    # Combined direction: equal weight between factor model and MC distribution
    combined_direction = 0.5 * factor_direction + 0.5 * mc_direction
    # Convert back to probability form for downstream dir_align computations
    final_prob_up   = max(0.01, min(0.99, (combined_direction + 1.0) / 2.0))
    final_prob_down = 1.0 - final_prob_up

    # Use MC expected move for safety ratio and strike selection (overrides straddle EM)
    em = mc_expected_move if mc_expected_move > 0 else em

    results = []
    for s in universe:
        # ── Strategy-specific IV for MC simulation ───────────────────────────
        # Use the average IV across the strategy's strikes (from the surface),
        # not just ATM IV. This gives proper vol for OTM strangles etc.
        _leg_strikes = [float(leg["strike"]) for leg in s["legs"]]
        _leg_ivs     = [iv_surf(k) for k in _leg_strikes]
        sigma_strat  = float(np.mean(_leg_ivs)) if _leg_ivs else sigma_sim
        # GAP-1 FIX: reuse the jump component (same Z, same _log_jump_total paths)
        # so all strategies are evaluated on the same fat-tail-aware paths.
        prices_strat = spot * np.exp(
            (mu - 0.5 * sigma_strat**2 - _compensator) * T
            + sigma_strat * math.sqrt(T) * Z
            + _log_jump_total
        )

        # ── MC P&L ──────────────────────────────────────────────
        # GAP-2 FIX: Deduct bid-ask half-spread from each leg's effective premium.
        # A sell leg receives mid - ½spread; a buy leg pays mid + ½spread.
        # On a 4-leg Iron Condor this costs ~₹300-700/lot before entry.
        pnl = np.zeros(n_sims)
        _total_tx_cost = 0.0   # ₹ per unit, accumulated across all legs
        for leg in s["legs"]:
            k   = float(leg["strike"])
            pr  = float(leg["premium"])
            qty = int(leg.get("qty", 1))
            d   = 1 if leg["action"] == "buy" else -1
            # Extract bid/ask if stored in leg (populated from chain row); fall back to mid.
            _bid = float(leg.get("bid", pr))
            _ask = float(leg.get("ask", pr))
            _half_spread = max(0.0, (_ask - _bid) / 2.0)
            # Buy legs pay ask (mid + ½spread); sell legs receive bid (mid - ½spread).
            pr_eff = pr + (_half_spread if leg["action"] == "buy" else -_half_spread)
            intr = np.maximum(prices_strat - k, 0) if leg["opt"] == "CE" else np.maximum(k - prices_strat, 0)
            pnl += d * (intr - pr_eff) * qty
            # RECALIBRATION FIX: accumulate SEBI/NSE transaction costs per leg
            _total_tx_cost += _tx_cost_per_leg(pr, leg["action"], qty)

        # Deduct total transaction cost from every path (it's a fixed ₹ drag, path-independent)
        pnl -= _total_tx_cost

        pop = float((pnl > 0).mean())
        ev  = float(pnl.mean())
        ev_per_lot = ev * lot_size

        max_risk   = float(s["max_risk"])
        max_reward = float(s["max_reward"])

        # FIX Bug 6: For unlimited-risk/reward strategies (max_risk=999 or max_reward=999),
        # use MC-derived actual risk/reward instead of the placeholder sentinel value.
        _mc_pnl_5pct  = float(np.percentile(pnl, 5))   # worst 5% outcome
        _mc_pnl_95pct = float(np.percentile(pnl, 95))  # best 5% outcome
        _mc_risk_eff   = max(abs(_mc_pnl_5pct), sum(abs(leg["premium"]) for leg in s["legs"]), 1.0)
        _mc_reward_eff = max(abs(_mc_pnl_95pct), 1.0)
        if max_risk   >= 999: max_risk   = _mc_risk_eff
        if max_reward >= 999: max_reward = _mc_reward_eff

        # ── PART 6: EV blend — model probability + MC distribution ───────────
        # Blend MC-derived EV with a model-probability-based EV estimate.
        # This ensures model conviction (prob_up/prob_down) feeds into EV,
        # not just the risk-neutral MC distribution.
        # model_ev = prob_up * avg_win_mc - prob_down * avg_loss_mc
        wins_arr   = pnl[pnl > 0]
        losses_arr = pnl[pnl < 0]
        _avg_win_mc  = float(wins_arr.mean())  if len(wins_arr)  > 0 else max_reward
        _avg_loss_mc = float(abs(losses_arr.mean())) if len(losses_arr) > 0 else max_risk
        model_ev = (final_prob_up * _avg_win_mc) - (final_prob_down * _avg_loss_mc)
        # Blend: 60% MC paths (unbiased), 40% model probability (directional conviction)
        ev = round(0.60 * ev + 0.40 * model_ev, 4)
        ev_per_lot = ev * lot_size

        # ── PART 7: Kelly using explicit model probability formula ────────────
        # kelly = (prob_up * win - prob_down * loss) / loss  (spec formula)
        # Use model prob_up/prob_down (blended with MC) for conviction-based sizing
        if _avg_win_mc > 0 and _avg_loss_mc > 0:
            _kelly_model = ((final_prob_up * _avg_win_mc
                             - final_prob_down * _avg_loss_mc)
                            / (_avg_loss_mc + 1e-9))
            # Also compute pure MC Kelly for comparison
            p_win    = len(wins_arr) / len(pnl) if len(wins_arr) > 0 else 0.0
            _kelly_mc = ((p_win * _avg_win_mc - (1.0 - p_win) * _avg_loss_mc)
                         / (_avg_win_mc + 1e-9))
            # Blend: model Kelly carries directional conviction, MC Kelly is unbiased
            kelly_raw = 0.60 * _kelly_mc + 0.40 * _kelly_model
        elif max_risk > 0 and max_risk < 1e6:
            kelly_raw = ev / (max_risk + 1e-9)
        elif max_risk >= 1e6:
            _proxy_risk = sum(abs(leg["premium"]) for leg in s["legs"]) * 2
            kelly_raw   = ev / (_proxy_risk + 1e-9)
        else:
            kelly_raw = 0.0
        kelly_capped = max(0.0, min(CFG.get("kelly_cap", 0.25), kelly_raw))
        kelly_fractional = kelly_capped * CFG.get("kelly_fraction", 0.5)

        # ── DTE alignment — continuous exponential decay, no binary in/out ──
        dte_lo, dte_hi = s["ideal_dte_lo"], s["ideal_dte_hi"]
        if dte_lo <= actual_dte <= dte_hi:
            dte_align = 1.0
        else:
            dist      = min(abs(actual_dte - dte_lo), abs(actual_dte - dte_hi))
            half_rng  = max((dte_hi - dte_lo) / 2.0, 1.0)
            # Steeper decay: halves every half_range DTE outside ideal window.
            # Floor lowered to 0.01 so severely mismatched strategies (e.g. Long Strangle
            # at DTE=1 with ideal=30-60) score near-zero rather than 0.26.
            dte_align = max(0.01, math.exp(-math.log(2) * dist / half_rng))

        # ── Safety factor — calibrated sigmoid sharpness ─────────────────────
        sk = s.get("short_strike")
        if sk is not None and em > 0:
            safety_ratio = abs(float(sk) - spot) / (em + 1e-9)
            safety_factor = _logistic(_calib("safety_sigmoid_sharpness") * (safety_ratio - 1.0))
            _record_if_load("_calib_safety_ratio_hist", safety_ratio)
        else:
            safety_ratio  = 2.0
            safety_factor = 1.0

        # ── Calendar term structure bonus — calibrated tanh scale ─────────────
        ts_factor = 1.0
        ts_slope  = s.get("term_slope")
        if ts_slope is not None:
            _record_if_load("_calib_ts_slope_raw_hist", ts_slope)
            ts_factor = 1.0 + 0.5 * math.tanh(ts_slope * _calib("ts_tanh_scale"))

        # ── EV-adjusted score — calibrated ev_tanh_scale ─────────────────────
        # FIX: max_risk can be near-zero on expiry day (DTE=1) when premiums collapse.
        # Floor at 1% of spot (≈ ₹226 for Nifty at 22600) to prevent tanh overflow → score=1 for all.
        _min_risk = max(spot * 0.005, sum(abs(leg["premium"]) for leg in s["legs"]) * 0.5, 1.0)
        _eff_risk = max(float(max_risk), _min_risk)
        ev_sign  = 1.0 if ev >= 0 else -1.0
        ev_norm  = math.tanh(abs(ev) / (_eff_risk * _calib("ev_tanh_scale")))
        ev_score = ev_norm * pop * dte_align * safety_factor * ts_factor * ev_sign
        ev_score = max(-1.0, min(1.0, ev_score))

        # ── Directional alignment — MC+factor blended probabilities ──────────
        stype     = s["type"]
        if "bull" in stype:
            dir_align = final_prob_up
        elif "bear" in stype:
            dir_align = final_prob_down
        else:
            # Neutral strategies: alignment depends on whether they need vol or calm
            # centrality = how close prob_up is to 0.5 (0=strong directional, 1=pure neutral)
            centrality = 1.0 - 2.0 * abs(final_prob_up - 0.5)   # 0→1
            if "debit" in stype:
                # Debit neutral (Long Straddle/Strangle, Butterfly buy): profit from BIG moves.
                # PENALISE when market is range-bound (centrality high = bad for long vol)
                # REWARD when strong directional signal exists (big move expected either way)
                dir_align = 1.0 - centrality   # strong direction → high alignment for debit neutral
            else:
                # Credit neutral (Short Straddle/Strangle, Iron Condor/Butterfly): profit from CALM.
                # REWARD when market is range-bound, PENALISE when strong directional signal
                dir_align = centrality

        # ── Composite — calibrated blend of EV score vs directional alignment ──
        # GAP-3 FIX: When EV is negative, the dir_align term is zeroed out.
        # This prevents a strategy with EV < 0 from outranking a positive-EV
        # strategy purely because directional alignment happens to be high.
        # The (dir_align - 0.5) * 2 term ranges up to +1.0, which at weight 0.40
        # can add 0.40 to composite — easily overriding a small negative ev_score.
        # Gating dir_align on sign(ev) closes this loophole.
        _ev_w    = _calib("ev_score_vs_dir_align")
        _dir_contribution = (dir_align - 0.5) * 2 if ev >= 0 else 0.0
        composite = _ev_w * ev_score + (1.0 - _ev_w) * _dir_contribution
        composite = max(-1.0, min(1.0, composite))

        # Hard penalty for negative-EV strategies: cap composite so they never
        # outrank a strategy with positive EV regardless of dir_align score.
        # Use MC-corrected max_risk (already replaced 999 sentinel above) so
        # unlimited-risk strategies (Short Straddle) are correctly penalised.
        _orig_risk = max(float(max_risk), 1.0)   # max_risk already corrected above
        if ev < 0:
            # Severity = |EV| / actual_risk, capped at 1
            _severity = min(1.0, abs(ev) / _orig_risk)
            # Cap composite: scales from 0.30 (barely negative) down to 0.0 (catastrophic)
            _composite_cap = 0.30 * (1.0 - _severity)
            composite = min(composite, _composite_cap)

        composite = max(-1.0, min(1.0, composite))
        # Record for calibration
        _record_if_load("_calib_ev_score_hist",   ev_score)
        _record_if_load("_calib_dir_align_hist",  dir_align)
        _record_if_load("_calib_realised_pnl_hist", ev)
        # Convert to 0-100 display score
        display_score = int(round((composite + 1.0) / 2.0 * 100))

        results.append({
            "Strategy":      s["name"],
            "Type":          s["type"],
            "Legs":          s["display_legs"],
            "pop":           round(pop, 4),
            "ev":            round(ev, 2),
            "ev_per_lot":    round(ev_per_lot, 2),
            "tx_cost_unit":  round(_total_tx_cost, 2),        # ₹ tx cost per unit
            "tx_cost_lot":   round(_total_tx_cost * lot_size, 2),  # ₹ tx cost per lot
            "kelly":         round(kelly_fractional, 4),
            "kelly_raw":     round(kelly_raw, 4),
            "kelly_capped":  round(kelly_capped, 4),
            "max_risk":      round(max_risk, 2),
            "max_reward":    round(max_reward, 2),
            "dte_align":     round(dte_align, 3),
            "safety_ratio":  round(safety_ratio, 3),
            "safety_factor": round(safety_factor, 3),
            "dir_align":     round(dir_align, 4),
            "mc_prob_up":    round(mc_prob_up, 4),
            "mc_prob_down":  round(mc_prob_down, 4),
            "mc_direction":  round(mc_direction, 4),
            "mc_expected_move": round(mc_expected_move, 2),
            "ev_score":      round(ev_score, 4),
            "composite":     round(composite, 4),
            "Score":         display_score,
            # Jump params used for this strategy's simulation (fitted per symbol)
            "jump_lambda":   round(_jmp_lambda, 2),
            "jump_mu_j":     round(_jmp_mu_j, 4),
            "jump_sig_j":    round(_jmp_sig_j, 4),
            # Legacy UI fields
            "Max Risk":      f"₹{max_risk:,.0f}" if max_risk < 1e5 else "Unlimited",
            "Max Reward":    f"₹{max_reward:,.0f}" if max_reward < 1e5 else "Unlimited",
            "Ideal DTE":     f"{s['ideal_dte_lo']}–{s['ideal_dte_hi']} DTE",
            "ideal_dte_lo":  s["ideal_dte_lo"],
            "ideal_dte_hi":  s["ideal_dte_hi"],
            "Rationale":     (
                f"POP {pop*100:.1f}% · EV ₹{ev:+.0f} · Safety {safety_ratio:.2f}× EM · "
                f"DTE-align {dte_align:.2f} · Dir-align {dir_align:.2f} · "
                f"TxCost ₹{_total_tx_cost*lot_size:.0f}/lot · "
                f"Jump λ={_jmp_lambda:.1f} μ={_jmp_mu_j*100:.1f}% σ={_jmp_sig_j*100:.1f}%"
            ),
        })

    results.sort(key=lambda x: x["composite"], reverse=True)
    return results


def recommend_strategies(bias, vol_lbl, dte, spot, atm, step, ivr, bias_score=0,
                          bs_call_fn=None, bs_put_fn=None,
                          front_iv=None, back_iv=None,
                          expected_move=None,
                          prob_score=None, chain_df=None, ohlcv_df=None,
                          T=None, r=None, q=None, atm_iv=None, lot_size=1):
    """
    Unified entry point — returns EV-ranked strategy list.
    Falls back to heuristic-free display when MC inputs unavailable.
    """
    # Build universe
    if bs_call_fn is None:
        return []

    universe = _build_strategy_universe(
        spot, atm, step, T or 0.05, r or 0.065, q or 0.0, atm_iv or 0.20,
        bs_call_fn, bs_put_fn,
        chain_df  if (chain_df  is not None and not chain_df.empty)  else None,
        front_iv, back_iv)

    if prob_score is None:
        prob_score = {"prob_up": 0.5, "prob_down": 0.5, "expected_move": expected_move or 0,
                      "raw_score": 0.0}

    # Expected move fallback: ATR-based if OHLCV available, else adaptive seed
    _ohlcv_em2 = st.session_state.get("opt_ohlcv_df", None)
    if _ohlcv_em2 is not None and not _ohlcv_em2.empty and len(_ohlcv_em2) >= 5:
        _c_em2  = _ohlcv_em2["close"].astype(float)
        _h_em2  = _ohlcv_em2["high"].astype(float)
        _l_em2  = _ohlcv_em2["low"].astype(float)
        _tr_em2 = pd.concat([_h_em2 - _l_em2,
                              (_h_em2 - _c_em2.shift(1)).abs(),
                              (_l_em2 - _c_em2.shift(1)).abs()], axis=1).max(axis=1)
        _atr_fallback = float(_tr_em2.tail(14).mean())
    else:
        _atr_fallback = _atr_seed(atm)

    # Inject fallback EM into prob_score if it has no usable expected_move
    if not prob_score.get("expected_move"):
        prob_score = dict(prob_score)   # shallow copy — don't mutate caller's dict
        prob_score["expected_move"] = expected_move or _atr_fallback

    # Safe DataFrame fallbacks — never use 'or' with DataFrames (raises ValueError)
    _chain_df_safe = chain_df if (chain_df is not None and not chain_df.empty) else None
    _ohlcv_df_safe = ohlcv_df if (ohlcv_df is not None and not ohlcv_df.empty) else pd.DataFrame()

    return ev_rank_strategies(
        universe, spot, T or 0.05, r or 0.065, atm_iv or 0.20, q or 0.0,
        prob_score, _chain_df_safe, _ohlcv_df_safe,
        actual_dte=dte, lot_size=lot_size)


# ============================================================
# PORTFOLIO GREEKS MODULE
# ============================================================

def compute_portfolio_greeks(legs, spot, T, r, atm_iv, q=0.0, chain_df=None, lot_size=1):
    """Aggregate net Greeks for a multi-leg portfolio.
    Uses per-strike IV from the vol surface where available (skew-aware).
    Returns: net Delta, Gamma, Vega, Theta (₹/day), Charm, Vanna.
    """
    iv_surf = build_iv_surface(chain_df, spot, atm_iv) if chain_df is not None else (lambda K: atm_iv)
    net = {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0,
           "charm": 0.0, "vanna": 0.0}
    for leg in legs:
        k     = float(leg.get("strike", spot))
        qty   = int(leg.get("qty", 1))
        d     = 1 if str(leg.get("action", "buy")).lower() == "buy" else -1
        opt   = "call" if str(leg.get("opt", "CE")).upper() in ("CE", "CALL") else "put"
        sigma = iv_surf(k)
        g = bs_greeks(spot, k, T, r, sigma, opt, q)
        scale = d * qty * lot_size
        net["delta"] += g["delta"] * scale
        net["gamma"] += g["gamma"] * scale
        net["vega"]  += g["vega"]  * scale
        net["theta"] += g["theta"] * scale
        net["charm"] += bs_charm(spot, k, T, r, sigma, opt, q) * scale
        net["vanna"] += bs_vanna(spot, k, T, r, sigma, opt, q) * scale
    return {k: round(v, 4) for k, v in net.items()}


# ============================================================
# KELLY POSITION SIZING ENGINE  (Improvement #9)
# ============================================================

def kelly_position_size(win_prob, avg_win_pct, avg_loss_pct, capital,
                        max_kelly=0.25, fractional=0.5):
    """Compute Kelly fraction and position size.
    Kelly = (WinProb × AvgWin - (1-WinProb) × AvgLoss) / AvgWin
    Capped at max_kelly (default 0.25) for safety.
    fractional Kelly (default 0.5) further reduces variance.
    Returns: kelly_raw, kelly_capped, kelly_fractional, position_size (₹).
    """
    if avg_win_pct <= 0 or avg_loss_pct <= 0:
        return {"kelly_raw": 0.0, "kelly_capped": 0.0, "kelly_f": 0.0, "position_size": 0.0,
                "remark": "Invalid win/loss inputs"}
    p   = max(0.0, min(1.0, win_prob))
    q_k = 1.0 - p
    kelly_raw  = (p * avg_win_pct - q_k * avg_loss_pct) / avg_win_pct
    kelly_cap  = max(0.0, min(max_kelly, kelly_raw))
    kelly_frac = kelly_cap * fractional
    pos_size   = kelly_frac * capital

    if   kelly_raw <= 0:   remark = "Negative Kelly — edge is unfavourable; do not trade"
    elif kelly_raw > 0.5:  remark = f"Very high Kelly ({kelly_raw:.1%}) — capped at {kelly_cap:.1%}"
    elif kelly_raw > 0.25: remark = f"High Kelly ({kelly_raw:.1%}) — capped at {kelly_cap:.1%}"
    else:                   remark = f"Kelly {kelly_raw:.1%} — fractional ({fractional:.0%}) = {kelly_frac:.1%}"
    return {
        "kelly_raw":     round(kelly_raw,  4),
        "kelly_capped":  round(kelly_cap,  4),
        "kelly_f":       round(kelly_frac, 4),
        "position_size": round(pos_size,   2),
        "remark":        remark,
    }


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
if "opt_fii_dii"     not in st.session_state: st.session_state.opt_fii_dii     = {}
# ── Load-ID guard counters — must be initialised BEFORE the load_btn block ──────
# opt_load_id is incremented on every Load click and used by _record_if_load()
# to prevent duplicate calibration records on every Streamlit re-render.
if "opt_load_id"              not in st.session_state: st.session_state["opt_load_id"]              = 0
if "_last_recorded_load_id"   not in st.session_state: st.session_state["_last_recorded_load_id"]   = -1
if "_last_outcome_load_id"    not in st.session_state: st.session_state["_last_outcome_load_id"]    = -1
if "_fhist_last_load_id"      not in st.session_state: st.session_state["_fhist_last_load_id"]      = -1
if "opt_div_yield"   not in st.session_state: st.session_state.opt_div_yield   = {}
# Chain live-data flags — set on each load, read by render section
if "_chain_has_live" not in st.session_state: st.session_state["_chain_has_live"] = False
if "_market_open"    not in st.session_state: st.session_state["_market_open"]    = True
# Multi-expiry: stores a list of dicts, one per loaded expiry
# Each dict: {expiry, dte, T, chain_df, atm_iv, straddle, exp_move_pct, oi_d}
if "opt_multi_expiry" not in st.session_state: st.session_state.opt_multi_expiry = []
if "opt_multi_loaded" not in st.session_state: st.session_state.opt_multi_loaded = False
# Load IV history from disk on first session init (persists across restarts)
if "opt_iv_history"  not in st.session_state:
    st.session_state.opt_iv_history = _load_iv_history()
# Forward signal log — loaded from disk once per session
if "opt_signal_log"  not in st.session_state:
    st.session_state.opt_signal_log = _load_signal_log()
# User capital — persists in session, used for Kelly position sizing everywhere
if "opt_capital"     not in st.session_state: st.session_state.opt_capital     = 500_000
# Flow conviction threshold — updated dynamically from rolling magnitude history
if "opt_flow_conv_threshold" not in st.session_state:
    st.session_state.opt_flow_conv_threshold = CFG["flow_conviction_seed"]
# New institutional features
if "opt_iv_percentile" not in st.session_state: st.session_state.opt_iv_percentile = 50.0
if "opt_regime"        not in st.session_state: st.session_state.opt_regime        = {}
if "opt_events"        not in st.session_state: st.session_state.opt_events        = []
if "opt_rs_nifty"      not in st.session_state: st.session_state.opt_rs_nifty      = None
if "opt_liquidity"     not in st.session_state: st.session_state.opt_liquidity     = {}
# Adaptive parameter engine — calibration store and signal histories
# Load from disk on first session so weights survive restarts
if _CALIB_STORE_KEY not in st.session_state:
    st.session_state[_CALIB_STORE_KEY] = _load_calib()
if "opt_prob_score"     not in st.session_state: st.session_state.opt_prob_score     = {}
if "opt_ohlcv_df"       not in st.session_state: st.session_state.opt_ohlcv_df       = None
if "opt_factor_hist"    not in st.session_state: st.session_state.opt_factor_hist    = {}
if "opt_intraday_df"    not in st.session_state: st.session_state.opt_intraday_df    = pd.DataFrame()
if "opt_intraday_signals" not in st.session_state: st.session_state.opt_intraday_signals = {}

# ── CRITICAL: Restore all signal histories from disk ──────────────────────────
# Without this, every app restart wipes calibration signal histories, pending
# outcomes, and flow histories → the adaptive engine always trains from scratch
# and the weights never improve beyond the cold-start priors.
_restore_hist()

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
        expiry_list = fetch_expiries(get_token(), ikey)
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

    # ── Capital input — persists across loads, used for Kelly sizing everywhere ──
    st.divider()
    st.markdown("### 💼 Capital & Sizing")
    _cap_input = st.number_input(
        "Trading Capital (₹)",
        min_value=10_000, max_value=100_000_000,
        value=int(st.session_state.opt_capital),
        step=10_000, key="capital_sidebar",
        help="Used for Kelly position sizing on every strategy card. Set once, applies everywhere.")
    if _cap_input != st.session_state.opt_capital:
        st.session_state.opt_capital = int(_cap_input)
    _kelly_frac_display = CFG["kelly_fraction"]
    _kelly_cap_display  = CFG["kelly_cap_pct"]
    st.caption(f"Kelly: {_kelly_frac_display:.0%} fractional · capped at {_kelly_cap_display:.0%}")

    st.divider()
    load_btn = st.button("⚡ LOAD OPTIONS INTEL", use_container_width=True, key="load_opt_main")

    # ── Multi-expiry loader ──────────────────────────────────
    st.divider()
    st.markdown("### 📅 Term Structure")
    st.caption("Load multiple expiries to analyse IV term structure, roll costs, and calendar opportunities.")

    if expiry_list and len(expiry_list) >= 2:
        _max_ts = min(len(expiry_list), 5)
        _ts_default = min(3, _max_ts)
        ts_n = st.slider("Expiries to load", min_value=2, max_value=_max_ts,
                         value=_ts_default, key="ts_n_slider")
        ts_expiries = expiry_list[:ts_n]
        st.caption("Will load: " + "  ·  ".join(ts_expiries))
        multi_load_btn = st.button("📅 LOAD TERM STRUCTURE", use_container_width=True, key="multi_load_btn")
    else:
        multi_load_btn = False
        st.caption("Need ≥2 expiries from API to build term structure.")

    # Show a clean status card once loaded — no raw number clutter
    if st.session_state.opt_loaded:
        _s = st.session_state
        _bres = _s.opt_bias
        _bc2  = {"STRONGLY BULLISH":"#00d084","BULLISH":"#7dca84","NEUTRAL":"#ffb347",
                 "BEARISH":"#ff7777","STRONGLY BEARISH":"#ff3b3b"}.get(_bres.get("bias","NEUTRAL"),"#888")
        st.markdown(f"""
<div style="background:#0d0d0d;border:1px solid #2a2a2a;border-left:3px solid {_bc2};
padding:8px 10px;font-family:'IBM Plex Mono',monospace;font-size:0.77rem;margin-top:4px;">
  <div style="color:#555;letter-spacing:.08em;margin-bottom:3px;">LOADED</div>
  <div style="color:#ff8c00;font-weight:700;">{_s.opt_symbol} · {_s.opt_expiry}</div>
  <div style="color:#e8e8e8;">₹{_s.opt_spot:,.1f} · DTE {_s.opt_dte}</div>
  <div style="color:{_bc2};">{_bres.get('bias','—')} ({int(round(_bres.get('score',0))):+d})</div>
  <div style="color:#555;">IV {_s.opt_atm_iv*100:.1f}% · HV {(_s.opt_hv20 or CFG['hv_fallback'])*100:.1f}%</div>
</div>""", unsafe_allow_html=True)
    else:
        st.caption(f"Strike Step: {step_val}")
        if ikey: st.caption(f"Key: {ikey[:30]}…")

# ============================================================
# LOAD LOGIC
# ============================================================

if load_btn:
    st.session_state["opt_load_id"] = st.session_state.get("opt_load_id", 0) + 1   # FIX A
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
                    # Handle MultiIndex columns (yfinance 0.2+)
                    if isinstance(d.columns, pd.MultiIndex):
                        d.columns = [c[0].lower() if isinstance(c, tuple) else str(c).lower()
                                     for c in d.columns]
                    else:
                        d.columns = [str(c).lower() for c in d.columns]
                    _close_col = "close" if "close" in d.columns else (
                                 "Close" if "Close" in d.columns else None)
                    if _close_col:
                        spot = float(d[_close_col].iloc[-1])
            except Exception as _yf_err:
                st.warning(f"yfinance spot fallback failed for {sym_sel}: {_yf_err}")
        if not spot or spot <= 0:
            st.error(f"Could not get spot price for {sym_sel}. Use 'Spot Price Override'.")
            st.stop()

        # 2. Historical data
        ohlcv_df = get_ohlcv(sym_sel, get_token(), master_df)
        hv20     = compute_hv(ohlcv_df["close"].astype(float), CFG["hv_window"])      if not ohlcv_df.empty else None
        hv10     = compute_hv(ohlcv_df["close"].astype(float), CFG["hv_window_fast"]) if not ohlcv_df.empty else None

        # 2b. Intraday OHLCV — 5-min candles from Upstox (live, cached 60s)
        intraday_df = get_intraday_ohlcv(sym_sel, get_token(), interval="5minute", master_df=master_df)
        st.session_state["opt_intraday_df"] = intraday_df
        # Persist ohlcv_df so flow/factor weight functions can access it without a parameter chain
        st.session_state["opt_ohlcv_df"] = ohlcv_df if not ohlcv_df.empty else None

        # Cross-asset signals (MONARCH v2 Block G — Step 1)
        _ca_data = fetch_cross_asset_signals()
        st.session_state["opt_cross_asset"] = _ca_data
        if _ca_data.get("data_available"):
            _ca_hist = st.session_state.get("_cross_asset_signal_hist", {})
            for _k, _v in _ca_data.get("zscores", {}).items():
                _ca_hist.setdefault(_k, []).append(float(_v))
                _ca_hist[_k] = _ca_hist[_k][-252:]
            st.session_state["_cross_asset_signal_hist"] = _ca_hist
            _update_cross_asset_weights()
        _ca_composite = _ca_data.get("composite", 0.0)

        # Preliminary IV (before chain fetch) for early regime detection
        _hv_prelim = hv20 if hv20 and hv20 > 0.01 else CFG["hv_fallback"]
        atm_iv_prelim = _hv_prelim * 1.15

        # REGIME FIRST - Step 1 (MONARCH v2 Block G)
        regime_d = detect_market_regime(
            ohlcv_df if not ohlcv_df.empty else None,
            atm_iv_prelim, _hv_prelim, _ca_composite)
        st.session_state["opt_regime_d"]           = regime_d
        st.session_state["opt_regime_label"]       = regime_d.get("regime", _REGIME_TRANSITION)
        st.session_state["opt_regime_confidence"]  = regime_d.get("regime_confidence", 0.5)
        st.session_state["opt_regime_weights"]     = regime_d.get("signal_weights", CFG["factor_weights"])

        # ── Outcome resolution: compute real forward returns for past signals ──
        # This is the core feedback loop. Pending snapshots whose horizon has
        # elapsed are resolved against the current spot price, producing real
        # (signal, return) pairs that calibrate the model on actual market outcomes.
        _resolved = _resolve_outcomes(sym_sel, spot, ohlcv_df if not ohlcv_df.empty else None)
        if _resolved:
            _ingest_resolved_outcomes(sym_sel, _resolved)

        # ── RECALIBRATION FIX: bootstrap historical calibration from OHLCV ────
        # Replays the factor model over up to 252 days of past price data to
        # produce (signal, forward_return, actual_up) pairs immediately — no
        # weeks of live Load clicks needed. Idempotent: skips if already done
        # for this symbol+bar-count combination this session.
        _bootstrap_signal_history(
            ohlcv_df if not ohlcv_df.empty else None,
            symbol=sym_sel,
            horizon=4
        )

        # Run calibration cycle — uses real outcomes first, OHLCV fallback
        _run_calibration_cycle(
            ohlcv_df if not ohlcv_df.empty else None,
            symbol=sym_sel,
            horizon=4
        )

        # 3a. FII/DII institutional flow data (leading signal for Nifty)
        # Fetch once per Load and store in session_state for use in compute_flow_scores.
        # Only relevant for index options (Nifty/BankNifty etc.) — skip for stocks.
        _sym_upper = sym_sel.upper()
        _is_index  = _sym_upper in ("NIFTY","BANKNIFTY","FINNIFTY","MIDCPNIFTY","SENSEX")
        if _is_index:
            _fii_raw = fetch_fii_dii_data()
            st.session_state["opt_fii_dii"] = _fii_raw
            if _fii_raw.get("data_available"):
                _fii_sig = _fii_raw.get("fii_signal", 0.0)
                _fi_usd  = float(_fii_raw.get("usdinr", 0))
                _fi_vix2 = float(_fii_raw.get("indiavix", 0))
                _fi_mode = " [PROXY: USD/INR+VIX — no real SEBI flow data]" if _fii_raw.get("proxy_mode") else ""
                # BUG FIX (FII display): crore fields removed (were fabricated).
                # Display signal score only, with explicit proxy label.
                st.caption(
                    f"📊 Inst Flow ({_fii_raw.get('source_date','')}): "
                    f"USD/INR {_fi_usd:.2f} | "
                    f"India VIX {_fi_vix2:.1f} | "
                    f"Signal: {_fii_sig:+.3f}{_fi_mode}"
                )
        else:
            # For stocks: clear FII/DII signal (not relevant at stock level)
            st.session_state["opt_fii_dii"] = {"data_available": False,
                                                 "combined_signal": 0.0}

        # 4. Option chain
        # FIX: Clear chain cache on every explicit Load button press so LTP/OI always refresh.
        # The 30s cache is useful for tab-switching re-renders, not for explicit user reloads.
        fetch_option_chain.clear()
        chain_raw = fetch_option_chain(get_token(), ikey, expiry_sel) if ikey and expiry_sel else []
        chain_df  = parse_chain(chain_raw, spot, step_val)

        # ── Debug expander: show raw API response to diagnose field mapping issues ──
        with st.expander("🔬 Raw API Debug (first item)", expanded=False):
            if chain_raw and len(chain_raw) > 0:
                # Find ATM strike item for display
                _dbg_item = min(chain_raw, key=lambda x: abs(float(x.get("strike_price", 0)) - spot))
                st.json(_dbg_item)
                _dbg_ce = _dbg_item.get("call_options", {})
                _dbg_md = _dbg_ce.get("market_data", {})
                _dbg_og = _dbg_ce.get("option_greeks", {})
                st.caption(f"market_data keys: {list(_dbg_md.keys())}")
                st.caption(f"option_greeks keys: {list(_dbg_og.keys())}")
                st.caption(f"Parsed → CE_LTP: {chain_df[chain_df.Strike == float(_dbg_item.get('strike_price',0))]['CE_LTP'].values if not chain_df.empty else 'N/A'}")
            else:
                st.warning("chain_raw is empty — API returned no data")

        # ── FIX: Detect whether chain has real live data ───────────────────────
        # During market hours Upstox returns non-zero LTPs. Outside hours the
        # chain structure arrives but all LTP/OI/IV fields are 0. We detect this
        # and warn the user so misleading signals are clearly flagged.
        _chain_has_live = (not chain_df.empty and chain_df["CE_LTP"].sum() > 0
                           and chain_df["CE_OI"].sum() > 0)
        _market_open = True  # default: assume open if pytz unavailable
        if pytz:
            _ist       = pytz.timezone("Asia/Kolkata")
            _now_ist   = datetime.now(_ist)
            _is_weekday = _now_ist.weekday() < 5
            _mkt_open   = _now_ist.replace(hour=9,  minute=15, second=0, microsecond=0)
            _mkt_close  = _now_ist.replace(hour=15, minute=30, second=0, microsecond=0)
            _market_open = _is_weekday and _mkt_open <= _now_ist <= _mkt_close

            if not _market_open:
                st.info(f"ℹ️ Market is currently **closed** (IST {_now_ist.strftime('%H:%M')}). "
                        "Option chain LTP/OI/IV will be zero — signals based on OI/flow/IV are "
                        "unreliable until market opens at 09:15 IST. Use data for planning only.")
            elif not _chain_has_live:
                st.warning("⚠️ Market is open but chain data shows all zeros. "
                           "Possible causes: (1) token expired — paste a fresh token, "
                           "(2) instrument key mismatch — try reloading. "
                           "IV, OI and flow-based signals will be unreliable until real data loads.")
        elif not _chain_has_live:
            st.warning("⚠️ Option chain returned all-zero LTP/OI values. "
                       "Check your token validity and retry.")

        st.session_state["_chain_has_live"] = _chain_has_live
        st.session_state["_market_open"]    = _market_open

        # ── ΔOI tracking between loads ─────────────────────────────────────────
        # Upstox /option/chain may or may not return prev_oi.
        # We always track OI in session state between loads for reliable ΔOI.
        # Key uses int(strike) to avoid float precision mismatches (22600.0 vs 22600.000000004).
        _oi_snap_key = f"_oi_snap_{sym_sel.upper()}_{expiry_sel}"
        _prev_snap   = st.session_state.get(_oi_snap_key, {})
        if _chain_has_live and not chain_df.empty and _prev_snap:
            chain_df = chain_df.copy()
            for idx, row in chain_df.iterrows():
                k = int(round(row["Strike"]))
                prev_ce = _prev_snap.get(f"ce_{k}", None)
                prev_pe = _prev_snap.get(f"pe_{k}", None)
                # Only override if prev_oi from API is zero/missing
                if prev_ce is not None and float(row.get("CE_OIC", 0)) == 0:
                    chain_df.at[idx, "CE_OIC"] = row["CE_OI"] - prev_ce
                if prev_pe is not None and float(row.get("PE_OIC", 0)) == 0:
                    chain_df.at[idx, "PE_OIC"] = row["PE_OI"] - prev_pe
        # Save current OI snapshot for next load (Python 3.8-compatible merge)
        if _chain_has_live and not chain_df.empty:
            _new_snap = {}
            for _, _r in chain_df.iterrows():
                _k = int(round(_r["Strike"]))
                _new_snap[f"ce_{_k}"] = float(_r["CE_OI"])
                _new_snap[f"pe_{_k}"] = float(_r["PE_OI"])
            st.session_state[_oi_snap_key] = _new_snap

        # 4b. Compute intraday signals from live 5-min candles + OI change data
        _intra_sigs = compute_intraday_signals(intraday_df, chain_df, spot)
        st.session_state["opt_intraday_signals"] = _intra_sigs

        # Shock detection (MONARCH v2 Block G — Step 2)
        _shock = detect_shock_event(atm_iv_prelim, chain_df if not chain_df.empty else None, ohlcv_df if not ohlcv_df.empty else None)
        st.session_state["opt_shock"] = _shock
        if _shock["shock_detected"] and _shock["override_weights"]:
            st.session_state["opt_regime_weights"] = _shock["override_weights"]

        # 3b. Direction — MOVED here (was before chain fetch — BUG FIX 2).
        # directional_bias calls compute_flow_scores which reads dPCR, dSkew, dIV, dOI
        # from the LIVE chain. Running it before fetch_option_chain meant flow signals
        # were always computed from the PREVIOUS load's chain — one full cycle stale.
        # Now it receives the freshly fetched, OI-snapshot-enriched chain_df.
        bias_res = directional_bias(ohlcv_df, spot,
                                    chain_df=chain_df if not chain_df.empty else None)

        # 5. T — trading-day fraction using np.busday_count (weekdays only, NOT calendar/365)
        hv_ref = hv20 if hv20 and hv20 > 0.01 else CFG["hv_fallback"]
        if expiry_sel:
            actual_dte_td = max(int(np.busday_count(
                datetime.now().date().isoformat(), expiry_sel)), 1)
            T_val = actual_dte_td / CFG["ann_days"]
        elif dte_sidebar and dte_sidebar > 0:
            actual_dte_td = dte_sidebar
            T_val = dte_sidebar / CFG["ann_days"]
        else:
            actual_dte_td = 7
            T_val = 7.0 / CFG["ann_days"]
            st.warning("⚠️  No expiry date provided; DTE defaulted to 7 trading days. "
                       "Select an expiry or use the DTE override for accurate results.")

        # 6. ATM IV — PRIMARY: Brenner-Subrahmanyam from ATM straddle midpoint LTP
        atm_k  = atm_strike(spot, step_val)
        atm_iv = None

        if not chain_df.empty:
            row = chain_df.iloc[(chain_df.Strike - spot).abs().argsort()[:1]]
            if not row.empty:
                strd = float(row.CE_LTP.values[0]) + float(row.PE_LTP.values[0])
                if strd > 0 and T_val > 0 and spot > 0:
                    _bs_const = math.sqrt(2.0 / math.pi)
                    _iv_bs    = strd / (_bs_const * spot * math.sqrt(T_val))
                    atm_iv = _sanitise_iv(_iv_bs, None)

        if atm_iv is None and not chain_df.empty:
            # Secondary: average of sanitised per-leg IVs
            row = chain_df.iloc[(chain_df.Strike - spot).abs().argsort()[:1]]
            if not row.empty:
                valid = [v for v in [
                    _sanitise_iv(float(row.CE_IV.values[0]), None),
                    _sanitise_iv(float(row.PE_IV.values[0]), None)
                ] if v is not None]
                if valid:
                    atm_iv = sum(valid) / len(valid)

        if atm_iv is None:
            atm_iv = hv_ref  # final fallback: use HV (better than arbitrary constant)

        # 7. Dividend yield from yfinance info
        q_yield = 0.0
        try:
            _yft  = YF_TICKERS.get(sym_sel.upper(), f"{sym_sel.upper()}.NS")
            _info = yf.Ticker(_yft).info
            _dy   = _info.get("dividendYield") or 0.0
            q_yield = float(_dy) if 0 <= _dy < 0.20 else 0.0
        except Exception:
            q_yield = 0.0
        st.session_state.opt_div_yield[sym_sel.upper()] = q_yield

        # 8. Persist IV to disk-backed rolling history
        _append_iv(sym_sel, atm_iv)

        # 9. OI analysis — lot size from centralized CFG
        _lot = CFG["lot_sizes"].get(sym_sel.upper(), CFG["lot_size_fallback"])
        _rfr_for_oi = float(st.session_state.get("rfr_sidebar", CFG["rfr_default"])) / 100.0
        oi_d = oi_analysis(chain_df, spot, step_val, T=T_val, r=_rfr_for_oi,
                           atm_iv=atm_iv, lot_size=_lot)

        # Store in session
        st.session_state.opt_chain_data = chain_df
        st.session_state.opt_spot       = spot
        st.session_state.opt_atm_iv     = atm_iv
        st.session_state.opt_hv20       = hv20 or hv_ref
        st.session_state.opt_bias       = bias_res
        st.session_state.opt_oi         = oi_d
        st.session_state.opt_symbol     = sym_sel
        st.session_state.opt_expiry     = expiry_sel
        st.session_state.opt_dte        = actual_dte_td
        st.session_state.opt_step       = step_val
        st.session_state.opt_atm_k      = atm_k
        st.session_state.opt_rfr        = _rfr_for_oi
        st.session_state.opt_T          = T_val
        st.session_state.opt_hv10       = hv10 or hv_ref
        st.session_state.opt_ohlcv      = ohlcv_df
        # opt_ohlcv_df already set earlier (line ~4666) before calibration cycle
        st.session_state.opt_loaded     = True
        st.session_state.payoff_legs    = []

        # ── NEW: Institutional metrics ──────────────────────────
        # IV Percentile
        _iv_hist_load = st.session_state.opt_iv_history.get(sym_sel, [])
        if len(_iv_hist_load) >= 3:
            st.session_state.opt_iv_percentile = iv_percentile(_iv_hist_load)
        else:
            st.session_state.opt_iv_percentile = 50.0

        # Market Regime (MONARCH v2 Block G — Step 3: final call with real atm_iv)
        _hv_for_regime = hv20 or hv_ref
        _ca_composite_final = st.session_state.get("opt_cross_asset", {}).get("composite", 0.0)
        regime_d_final = detect_market_regime(
            ohlcv_df if not ohlcv_df.empty else None,
            atm_iv or atm_iv_prelim, _hv_for_regime, _ca_composite_final)
        st.session_state.opt_regime   = regime_d_final
        st.session_state["opt_regime_d"] = regime_d_final
        st.session_state["opt_regime_label"]      = regime_d_final.get("regime")
        st.session_state["opt_regime_confidence"] = regime_d_final.get("regime_confidence")
        # Keep shock-overridden weights if shock still active
        _shock_now = st.session_state.get("opt_shock", {})
        if not (_shock_now.get("shock_detected") and _shock_now.get("override_weights")):
            st.session_state["opt_regime_weights"] = regime_d_final.get("signal_weights", CFG["factor_weights"])

        # Event Detection
        if expiry_sel:
            st.session_state.opt_events = detect_events(expiry_sel)
        else:
            st.session_state.opt_events = []

        # Liquidity Analysis
        st.session_state.opt_liquidity = liquidity_analysis(chain_df, spot, step_val, atm_iv, T_val)

        # RS vs Nifty (skip for indices, they're the benchmark)
        if sym_sel.upper() not in ("NIFTY","BANKNIFTY","FINNIFTY","MIDCPNIFTY","SENSEX"):
            with st.spinner("Computing RS vs Nifty…"):
                st.session_state.opt_rs_nifty = relative_strength_vs_nifty(sym_sel, ohlcv_df)
        else:
            st.session_state.opt_rs_nifty = None

        # ── Flow conviction threshold: update from rolling magnitude history ──────
        # Track flow_magnitude across loads; threshold = 70th percentile of history
        # This makes "high conviction" adaptive — on a quiet day it's lower, on volatile
        # days it auto-raises so only truly exceptional flow fires the alert
        _flow_mag_now = bias_res.get("flow", {}).get("flow_magnitude", 0.0)
        _fmag_hist = st.session_state.get("_flow_mag_hist", [])
        _fmag_hist.append(float(_flow_mag_now))
        if len(_fmag_hist) > 50: _fmag_hist = _fmag_hist[-50:]
        st.session_state["_flow_mag_hist"] = _fmag_hist
        if len(_fmag_hist) >= 5:
            # 70th percentile of observed magnitudes = high conviction threshold
            st.session_state.opt_flow_conv_threshold = float(np.percentile(_fmag_hist, 70))
        # else: keep seed value from session init

        # ── Forward signal log — append this load's signal ───────────────────────
        # Computed after prob_score is available; store for forward performance tracking
        # prob_score is computed in the render section, but we need it at load time.
        # Solution: store the raw inputs; prob_score gets computed fresh on each render.
        # We append a minimal snapshot here — full prob_score appended in render section.
        st.session_state["_pending_signal_log"] = {
            "symbol":    sym_sel,
            "atm_iv":    atm_iv,
            "ivr":       st.session_state.opt_iv_percentile,
            "bias":      bias_res.get("bias", "NEUTRAL"),
            "flow_score": _flow_mag_now,
        }

        # ── CRITICAL: Persist all signal histories to disk ────────────────────────
        # This is the fix for the calibration system never accumulating data across
        # restarts. _record() writes to session_state only; _save_hist() syncs to disk.
        # Weights in _CALIB_FILE are also saved here via _set_calib → _save_calib.
        _save_hist()

# ============================================================
# MULTI-EXPIRY LOAD LOGIC
# ============================================================

if multi_load_btn:
    _spot_me = st.session_state.opt_spot if st.session_state.opt_loaded else None
    if not _spot_me:
        # try to get spot quickly
        _ikey_me = find_instrument_key(load_fno_master(), sym_sel)
        _spot_me = fetch_spot_quote(_ikey_me) if _ikey_me else None
    if not _spot_me or _spot_me <= 0:
        st.error("Load single expiry first (⚡ LOAD OPTIONS INTEL) to establish spot price, then load Term Structure.")
    else:
        _ikey_me   = find_instrument_key(load_fno_master(), sym_sel)
        _rfr_me    = float(st.session_state.get("rfr_sidebar", CFG["rfr_default"])) / 100.0
        _step_me   = STRIKE_STEPS.get(sym_sel.upper(), DEFAULT_STEP)
        _lot_me    = CFG["lot_sizes"].get(sym_sel.upper(), CFG["lot_size_fallback"])
        _atm_me    = atm_strike(_spot_me, _step_me)
        _bs_const  = math.sqrt(2.0 / math.pi)
        _q_me      = st.session_state.opt_div_yield.get(sym_sel.upper(), 0.0)
        _hv_me     = st.session_state.opt_hv20 if st.session_state.opt_loaded else CFG["hv_fallback"]

        multi_data = []
        _prog = st.progress(0, text="Loading term structure…")
        for _idx, _exp in enumerate(ts_expiries):
            _prog.progress(int((_idx+1)/len(ts_expiries)*100),
                           text=f"Fetching {_exp}…")
            try:
                _dte_td = max(int(np.busday_count(
                    datetime.now().date().isoformat(), _exp)), 1)
                _T_me   = _dte_td / CFG["ann_days"]

                _raw    = fetch_option_chain(get_token(), _ikey_me, _exp) if _ikey_me else []
                _cdf    = parse_chain(_raw, _spot_me, _step_me) if _raw else pd.DataFrame()

                # ATM IV via straddle midpoint
                _atm_iv_me = None
                if not _cdf.empty:
                    _row = _cdf.iloc[(_cdf.Strike - _spot_me).abs().argsort()[:1]]
                    if not _row.empty:
                        _strd = float(_row.CE_LTP.values[0]) + float(_row.PE_LTP.values[0])
                        if _strd > 0 and _T_me > 0:
                            _iv_bs = _strd / (_bs_const * _spot_me * math.sqrt(_T_me))
                            _atm_iv_me = _sanitise_iv(_iv_bs, None)
                if _atm_iv_me is None:
                    _atm_iv_me = _hv_me

                # per-strike IVs for smile
                _smile = []
                if not _cdf.empty:
                    for _, _sr in _cdf.iterrows():
                        _civ = _sanitise_iv(float(_sr.CE_IV), _atm_iv_me)
                        _piv = _sanitise_iv(float(_sr.PE_IV), _atm_iv_me)
                        _smile.append({"Strike": float(_sr.Strike),
                                       "CE_IV": round(_civ*100, 2),
                                       "PE_IV": round(_piv*100, 2)})

                # straddle & expected move
                _strd_val = 0.0
                _exp_move = 0.0
                if not _cdf.empty:
                    _ar = _cdf.iloc[(_cdf.Strike - _spot_me).abs().argsort()[:1]]
                    if not _ar.empty:
                        _strd_val = float(_ar.CE_LTP.values[0]) + float(_ar.PE_LTP.values[0])
                        _exp_move = round(_strd_val / _spot_me * 100, 2)

                # OI totals — always use epsilon denominator, never return 0 when CE_OI=0
                _total_ce = float(_cdf.CE_OI.sum()) if not _cdf.empty else 0
                _total_pe = float(_cdf.PE_OI.sum()) if not _cdf.empty else 0
                _pcr      = round(_total_pe / (_total_ce + 1e-9), 3)

                multi_data.append({
                    "expiry":      _exp,
                    "dte":         _dte_td,
                    "T":           round(_T_me, 5),
                    "atm_iv":      round(_atm_iv_me, 5),
                    "atm_iv_pct":  round(_atm_iv_me * 100, 2),
                    "straddle":    round(_strd_val, 2),
                    "exp_move_pct":_exp_move,
                    "chain_df":    _cdf,
                    "smile":       _smile,
                    "total_ce_oi": int(_total_ce),
                    "total_pe_oi": int(_total_pe),
                    "pcr":         _pcr,
                })
            except Exception as _e:
                st.warning(f"Could not load {_exp}: {_e}")

        _prog.empty()

        # Compute forward volatility between consecutive tenors
        # Forward vol: σ_f(T1,T2) = sqrt( (σ₂²·T2 − σ₁²·T1) / (T2 − T1) )
        for _i in range(1, len(multi_data)):
            _prev = multi_data[_i-1]
            _curr = multi_data[_i]
            _t1, _t2 = _prev["T"], _curr["T"]
            _v1, _v2 = _prev["atm_iv"], _curr["atm_iv"]
            _dt = _t2 - _t1
            if _dt > 0 and _v2**2 * _t2 >= _v1**2 * _t1:
                _fv = math.sqrt((_v2**2 * _t2 - _v1**2 * _t1) / _dt)
            else:
                _fv = _v2  # fallback if variance curve inverted
            multi_data[_i]["fwd_vol"] = round(_fv, 5)
            multi_data[_i]["fwd_vol_pct"] = round(_fv * 100, 2)
        if multi_data:
            multi_data[0]["fwd_vol"] = multi_data[0]["atm_iv"]
            multi_data[0]["fwd_vol_pct"] = multi_data[0]["atm_iv_pct"]

        st.session_state.opt_multi_expiry = multi_data
        st.session_state.opt_multi_loaded = True
        st.rerun()

# ── Tradebook: persistence helpers ──────────────────────────────────────────
_TRADEBOOK_FILE = ".monarch_tradebook.json"
_HISTPNL_FILE   = ".monarch_hist_pnl.json"

def _load_tradebook() -> list:
    try:
        if os.path.exists(_TRADEBOOK_FILE):
            with open(_TRADEBOOK_FILE, "r") as f:
                d = json.load(f)
            return d if isinstance(d, list) else []
    except Exception:
        pass
    return []

def _save_tradebook(data: list):
    try:
        with open(_TRADEBOOK_FILE, "w") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass

def _load_hist_pnl() -> list:
    try:
        if os.path.exists(_HISTPNL_FILE):
            with open(_HISTPNL_FILE, "r") as f:
                d = json.load(f)
            return d if isinstance(d, list) else []
    except Exception:
        pass
    return []

def _save_hist_pnl(data: list):
    try:
        with open(_HISTPNL_FILE, "w") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass

if "opt_tradebook" not in st.session_state:
    st.session_state.opt_tradebook = _load_tradebook()
if "opt_hist_pnl"  not in st.session_state:
    st.session_state.opt_hist_pnl  = _load_hist_pnl()

# ── Create ALL tabs here — before st.stop() so Tradebook always renders ──────
# Other tabs only show content when opt_loaded=True (guarded inside each tab).
# Tradebook reads exclusively from session_state so it works at all times.
(t_signal, t_trade, t_chain, t_structure,
 t_reference, t_edge, t_prob, t_tradebook) = st.tabs([
    "⚡ Signal",
    "🎯 Trade",
    "📋 Chain & Data",
    "📅 Structure",
    "📐 Reference",
    "🔬 Edge Audit",
    "🎲 Probability Engine",
    "📒 Tradebook",
])

# ── Tradebook tab — fully self-contained, reads only session_state ────────────
def _estimate_trade_pnl(te: dict, live_spot: float = None) -> float:
    """Estimate directional PnL for a logged trade.
    Uses live_spot if provided (after load), else falls back to spot_current stored in entry.
    For neutral strategies (condor/straddle): negative when move exceeds expected move.
    """
    spot_e = float(te.get("spot_entry", 0) or 0)
    spot_c = float(live_spot if live_spot else te.get("spot_current", spot_e) or spot_e)
    pos    = float(te.get("pos_size_rs", 0) or 0)
    if spot_e <= 0 or pos <= 0:
        return 0.0
    pct_move = (spot_c - spot_e) / spot_e
    strat = (te.get("strategy") or "").lower()
    if any(x in strat for x in ["buy call", "call buy", "bull call", "bull spread",
                                  "sell put", "put sell", "put spread"]):
        mult = 1.0
    elif any(x in strat for x in ["buy put", "put buy", "bear put", "bear spread",
                                    "sell call", "call sell", "bear call"]):
        mult = -1.0
    elif any(x in strat for x in ["condor", "strangle", "straddle", "calendar", "butterfly"]):
        em_pct = float(te.get("exp_move_pct", 2.0) or 2.0) / 100.0
        excess = abs(pct_move) - em_pct
        mult   = -1.0 if excess > 0 else 1.0
        pct_move = abs(pct_move)
    else:
        bias_entry = te.get("bias", "NEUTRAL")
        mult = 1.0 if "BULL" in str(bias_entry) else (-1.0 if "BEAR" in str(bias_entry) else 1.0)
    return round(pct_move * mult * pos, 2)

with t_tradebook:
    st.markdown("### 📒 Tradebook — Active Trades & Historical PnL")

    # Refresh spot_current for active trades from live session state (post-load)
    _live_spot_tb = st.session_state.get("opt_spot", None)
    _live_iv_tb   = st.session_state.get("opt_atm_iv", None)
    _tb_list      = st.session_state.opt_tradebook
    _changed_tb   = False
    for _te in _tb_list:
        if _te.get("status") == "ACTIVE" and _live_spot_tb:
            _te["spot_current"]   = float(_live_spot_tb)
            _te["atm_iv_current"] = float(_live_iv_tb) if _live_iv_tb else _te.get("atm_iv_current", 0)
            _changed_tb = True
    if _changed_tb:
        st.session_state.opt_tradebook = _tb_list
        _save_tradebook(_tb_list)

    _active_trades = [t for t in st.session_state.opt_tradebook if t.get("status") == "ACTIVE"]
    _hist_trades   = st.session_state.opt_hist_pnl

    # ── Summary header metrics ────────────────────────────────────────────────
    _hm1, _hm2, _hm3, _hm4 = st.columns(4)
    _hm1.metric("Active Trades",    str(len(_active_trades)))
    _hm2.metric("Historical (Closed)", str(len(_hist_trades)))
    _total_unreal = sum(_estimate_trade_pnl(t, _live_spot_tb) for t in _active_trades)
    _hist_realised = sum(float(t.get("realised_pnl", 0) or 0) for t in _hist_trades)
    _hm3.metric("Est. Unrealised PnL",
                f"{'+'if _total_unreal>=0 else ''}₹{_total_unreal:,.0f}")
    _hm4.metric("Realised PnL (Closed)",
                f"{'+'if _hist_realised>=0 else ''}₹{_hist_realised:,.0f}")

    if not st.session_state.get("opt_loaded"):
        st.info("💡 Load options intel (⚡ LOAD OPTIONS INTEL) to enable live PnL updates. "
                "Logged trades and historical PnL are always visible here.")

    st.caption(
        "⚠ PnL estimates are directional approximations: (spot_now − spot_entry) / spot_entry × position size, "
        "adjusted for strategy direction. Neutral strategies go negative when move exceeds expected move. "
        "Verify actual P&L in your broker terminal."
    )
    st.divider()

    # ── ACTIVE TRADES ─────────────────────────────────────────────────────────
    st.markdown("#### 🟢 Active Trades")

    if not _active_trades:
        st.info(
            "No active trades yet. Go to the **🎯 Trade** tab → load options intel → "
            "find a strategy → press **📋 LOG Trade** to snapshot it here."
        )
    else:
        for _te in list(st.session_state.opt_tradebook):
            if _te.get("status") != "ACTIVE":
                continue
            _pnl_est  = _estimate_trade_pnl(_te, _live_spot_tb)
            _pnl_pct  = (_pnl_est / max(float(_te.get("pos_size_rs", 1) or 1), 1)) * 100
            _pnl_col  = "#00d084" if _pnl_est >= 0 else "#ff3b3b"
            _pnl_sign = "+" if _pnl_est >= 0 else ""
            _spot_e   = float(_te.get("spot_entry", 0) or 0)
            _spot_c   = float(_te.get("spot_current", _spot_e) or _spot_e)
            _sdelta   = _spot_c - _spot_e
            _sdel_col = "#00d084" if _sdelta >= 0 else "#ff3b3b"
            _sdel_sgn = "+" if _sdelta >= 0 else ""
            _bc_te    = ("#00d084" if "BULL" in str(_te.get("bias",""))
                         else "#ff3b3b" if "BEAR" in str(_te.get("bias",""))
                         else "#ffb347")
            _iv_e     = float(_te.get("atm_iv_entry", 0) or 0)
            _iv_c     = float(_te.get("atm_iv_current", _iv_e) or _iv_e)
            _iv_chg   = _iv_c - _iv_e
            _iv_col   = "#ff3b3b" if _iv_chg > 0.005 else "#1e90ff" if _iv_chg < -0.005 else "#888"

            st.markdown(f"""
<div style="background:#0d0d0d;border:1px solid #2a2a2a;border-left:4px solid {_pnl_col};
padding:14px 18px;margin-bottom:4px;font-family:'IBM Plex Mono',monospace;">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:6px;margin-bottom:10px;">
    <div>
      <span style="color:#ff8c00;font-size:0.72rem;font-weight:700;letter-spacing:.1em;">ACTIVE</span>
      &nbsp;·&nbsp;<span style="color:#555;font-size:0.72rem;">{_te.get('logged_at','—')}</span>
      &nbsp;·&nbsp;<span style="color:#888;font-size:0.72rem;">{_te.get('symbol','—')} · {_te.get('expiry','—')} · DTE@entry {_te.get('dte_at_entry','—')}</span>
    </div>
    <div style="text-align:right;">
      <span style="color:{_pnl_col};font-size:1.1rem;font-weight:700;">{_pnl_sign}₹{_pnl_est:,.0f}</span>
      <span style="color:{_pnl_col};font-size:0.78rem;margin-left:6px;">({_pnl_sign}{_pnl_pct:.1f}%)</span>
      <div style="color:#555;font-size:0.68rem;">Est. Unrealised PnL</div>
    </div>
  </div>
  <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:10px;">
    <div>
      <div style="color:#555;font-size:0.70rem;">STRATEGY</div>
      <div style="color:#e8e8e8;font-size:0.88rem;font-weight:700;">{_te.get('strategy','—')}</div>
      <div style="color:#777;font-size:0.70rem;">{_te.get('strategy_type','—')}</div>
    </div>
    <div>
      <div style="color:#555;font-size:0.70rem;">SIGNAL CLASS</div>
      <div style="color:#7ec8e3;font-size:0.84rem;font-weight:600;">{_te.get('strategy_class','—')}</div>
      <div style="color:#777;font-size:0.70rem;">EV Score {_te.get('ev_score',0)}</div>
    </div>
    <div>
      <div style="color:#555;font-size:0.70rem;">DIRECTION @ ENTRY</div>
      <div style="color:{_bc_te};font-size:0.88rem;font-weight:700;">{_te.get('bias','—')}</div>
      <div style="color:#777;font-size:0.70rem;">P(↑) {_te.get('prob_up',0.5)*100:.1f}% · Score {_te.get('raw_score',0):+.3f}</div>
    </div>
    <div>
      <div style="color:#555;font-size:0.70rem;">POSITION SIZE</div>
      <div style="color:#ff8c00;font-size:0.88rem;font-weight:700;">₹{float(_te.get('pos_size_rs',0) or 0):,.0f}</div>
      <div style="color:#777;font-size:0.70rem;">{float(_te.get('kelly_pct',0) or 0):.1f}% Kelly</div>
    </div>
  </div>
  <div style="display:grid;grid-template-columns:repeat(5,1fr);gap:8px;margin-bottom:10px;
  background:#080808;border:1px solid #1a1a1a;padding:8px 10px;">
    <div><div style="color:#555;font-size:0.68rem;">SPOT ENTRY</div>
         <div style="color:#e8e8e8;font-size:0.84rem;font-weight:700;">₹{_spot_e:,.1f}</div></div>
    <div><div style="color:#555;font-size:0.68rem;">SPOT NOW</div>
         <div style="color:{_sdel_col};font-size:0.84rem;font-weight:700;">₹{_spot_c:,.1f}</div></div>
    <div><div style="color:#555;font-size:0.68rem;">SPOT Δ</div>
         <div style="color:{_sdel_col};font-size:0.84rem;font-weight:700;">{_sdel_sgn}₹{_sdelta:,.1f}</div></div>
    <div><div style="color:#555;font-size:0.68rem;">ATM IV @ ENTRY</div>
         <div style="color:#888;font-size:0.84rem;">{_iv_e*100:.1f}%</div></div>
    <div><div style="color:#555;font-size:0.68rem;">IV Δ</div>
         <div style="color:{_iv_col};font-size:0.84rem;">{'+'if _iv_chg>=0 else ''}{_iv_chg*100:.2f}pp</div></div>
  </div>
  <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-bottom:8px;">
    <div><div style="color:#555;font-size:0.68rem;">MAX RISK</div>
         <div style="color:#ffb347;font-size:0.80rem;">{_te.get('max_risk','—')}</div></div>
    <div><div style="color:#555;font-size:0.68rem;">MAX REWARD</div>
         <div style="color:#00d084;font-size:0.80rem;">{_te.get('max_reward','—')}</div></div>
    <div><div style="color:#555;font-size:0.68rem;">POP @ ENTRY</div>
         <div style="color:#7ec8e3;font-size:0.80rem;">{float(_te.get('pop',0.5) or 0.5)*100:.1f}%</div></div>
    <div><div style="color:#555;font-size:0.68rem;">SAFETY RATIO</div>
         <div style="color:#888;font-size:0.80rem;">{float(_te.get('safety_ratio',0) or 0):.2f}× EM</div></div>
  </div>
  <div style="color:#444;font-size:0.72rem;line-height:1.5;border-top:1px solid #1a1a1a;padding-top:7px;">
    LEGS: <span style="color:#7ec8e3;">{_te.get('legs','—')}</span>
    &nbsp;·&nbsp; Vol Edge: <span style="color:#e8e8e8;">{_te.get('vol_edge','—')}</span>
    &nbsp;·&nbsp; Regime: <span style="color:#e8e8e8;">{_te.get('regime','—')}</span>
    &nbsp;·&nbsp; Flow: {float(_te.get('flow_score',0) or 0):+.3f}
    &nbsp;·&nbsp; Max Pain: ₹{float(_te.get('max_pain',0) or 0):,.0f}
    &nbsp;·&nbsp; Call Wall: ₹{float(_te.get('call_wall',0) or 0):,.0f}
    &nbsp;·&nbsp; Put Wall: ₹{float(_te.get('put_wall',0) or 0):,.0f}
  </div>
  <div style="color:#333;font-size:0.70rem;margin-top:4px;line-height:1.4;">{_te.get('dynamic_rationale','—')}</div>
</div>""", unsafe_allow_html=True)

            _btn1, _btn2, _btn3 = st.columns([1, 1.6, 4])
            with _btn1:
                if st.button("🗑 Delete", key=f"del_{_te['id']}",
                             help="Remove from Tradebook without recording PnL."):
                    st.session_state.opt_tradebook = [
                        t for t in st.session_state.opt_tradebook if t["id"] != _te["id"]
                    ]
                    _save_tradebook(st.session_state.opt_tradebook)
                    st.rerun()
            with _btn2:
                if st.button("📁 → Historical PnL", key=f"close_{_te['id']}",
                             help="Lock in current PnL and move to Historical."):
                    _closed = dict(_te)
                    _closed["status"]           = "HISTORICAL"
                    _closed["closed_at"]        = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    _closed["spot_close"]       = float(_live_spot_tb or _spot_c)
                    _closed["realised_pnl"]     = float(_pnl_est)
                    _closed["realised_pnl_pct"] = float(_pnl_pct)
                    _hp = st.session_state.opt_hist_pnl
                    _hp.append(_closed)
                    st.session_state.opt_hist_pnl = _hp
                    _save_hist_pnl(_hp)
                    st.session_state.opt_tradebook = [
                        t for t in st.session_state.opt_tradebook if t["id"] != _te["id"]
                    ]
                    _save_tradebook(st.session_state.opt_tradebook)
                    st.success(f"Trade closed. Realised PnL locked: {'+'if _pnl_est>=0 else ''}₹{_pnl_est:,.0f}")
                    st.rerun()

    st.divider()

    # ── HISTORICAL PnL ────────────────────────────────────────────────────────
    st.markdown("#### 📁 Historical PnL — Closed Trades")
    if not _hist_trades:
        st.info("No closed trades yet. Press **📁 → Historical PnL** on an active trade to record results.")
    else:
        _pnl_arr  = [float(t.get("realised_pnl", 0) or 0) for t in _hist_trades]
        _wins     = sum(1 for p in _pnl_arr if p > 0)
        _losses   = len(_pnl_arr) - _wins
        _wr       = _wins / len(_pnl_arr) * 100
        _avg_w    = float(np.mean([p for p in _pnl_arr if p > 0])) if _wins else 0
        _avg_l    = float(np.mean([p for p in _pnl_arr if p <= 0])) if _losses else 0
        _total_r  = float(np.sum(_pnl_arr))
        _rr       = abs(_avg_w / _avg_l) if _avg_l != 0 else float("inf")

        _sc1, _sc2, _sc3, _sc4, _sc5 = st.columns(5)
        _sc1.metric("Closed Trades", str(len(_pnl_arr)))
        _sc2.metric("Win Rate", f"{_wr:.1f}%", delta=f"{_wins}W / {_losses}L")
        _sc3.metric("Avg Win",  f"₹{_avg_w:,.0f}")
        _sc4.metric("Avg Loss", f"₹{_avg_l:,.0f}")
        _sc5.metric("Total Realised", f"{'+'if _total_r>=0 else ''}₹{_total_r:,.0f}",
                    delta=f"R:R {_rr:.2f}" if _avg_l != 0 else "R:R ∞")

        # Equity curve
        if len(_pnl_arr) >= 2:
            _cumul  = list(np.cumsum(_pnl_arr))
            _eq_c   = ["#00d084" if p >= 0 else "#ff3b3b" for p in _pnl_arr]
            _fig_eq = go.Figure()
            _fig_eq.add_trace(go.Scatter(
                x=list(range(1, len(_cumul)+1)), y=_cumul,
                mode="lines+markers", name="Cumulative PnL",
                line=dict(color="#ff8c00", width=2),
                marker=dict(color=_eq_c, size=8),
                text=[f"Trade {i+1}: {'+'if p>=0 else ''}₹{p:,.0f}<br>{_hist_trades[i].get('closed_at','—')}"
                      for i, p in enumerate(_pnl_arr)],
                hovertemplate="%{text}<extra></extra>",
            ))
            _fig_eq.add_hline(y=0, line=dict(color="#333", dash="dot", width=1))
            _fig_eq.update_layout(
                height=220, plot_bgcolor="#000", paper_bgcolor="#000",
                font=dict(color="#e8e8e8", family="IBM Plex Mono", size=9),
                margin=dict(t=20, b=20, l=40, r=10),
                xaxis=dict(gridcolor="#111", title="Trade #"),
                yaxis=dict(gridcolor="#111", title="Cumulative ₹ PnL"),
                showlegend=False,
            )
            st.plotly_chart(_fig_eq, use_container_width=True)

        st.divider()
        for _hidx, _ht in enumerate(reversed(_hist_trades)):
            _rp     = float(_ht.get("realised_pnl", 0) or 0)
            _rp_pct = float(_ht.get("realised_pnl_pct", 0) or 0)
            _rp_col = "#00d084" if _rp >= 0 else "#ff3b3b"
            _rp_sgn = "+" if _rp >= 0 else ""
            with st.expander(
                f"{'✅' if _rp>=0 else '❌'}  "
                f"{_ht.get('symbol','—')} · {_ht.get('strategy','—')} · "
                f"{_rp_sgn}₹{_rp:,.0f} ({_rp_sgn}{_rp_pct:.1f}%) · "
                f"Closed {_ht.get('closed_at','—')}",
                expanded=False
            ):
                _hc1,_hc2,_hc3,_hc4 = st.columns(4)
                _hc1.metric("Realised PnL",   f"{_rp_sgn}₹{_rp:,.0f}")
                _hc2.metric("Return %",        f"{_rp_sgn}{_rp_pct:.1f}%")
                _hc3.metric("Strategy",        _ht.get("strategy","—"))
                _hc4.metric("Bias @ Entry",    _ht.get("bias","—"))
                _hd1,_hd2,_hd3,_hd4 = st.columns(4)
                _hd1.metric("Spot Entry",  f"₹{float(_ht.get('spot_entry',0) or 0):,.1f}")
                _hd2.metric("Spot Close",  f"₹{float(_ht.get('spot_close',0) or 0):,.1f}")
                _hd3.metric("IV @ Entry",  f"{float(_ht.get('atm_iv_entry',0) or 0)*100:.1f}%")
                _hd4.metric("IVR @ Entry", f"{float(_ht.get('ivr_entry',0) or 0):.0f}")
                st.markdown(f"""
<div style="font-family:'IBM Plex Mono',monospace;font-size:0.76rem;color:#555;padding:8px 0;line-height:1.7;">
  <span style="color:#888;">LEGS:</span> <span style="color:#7ec8e3;">{_ht.get('legs','—')}</span>
  &nbsp;·&nbsp; P(↑): {float(_ht.get('prob_up',0.5) or 0.5)*100:.1f}%
  &nbsp;·&nbsp; POP: {float(_ht.get('pop',0.5) or 0.5)*100:.1f}%
  &nbsp;·&nbsp; EV Score: {_ht.get('ev_score',0)}
  &nbsp;·&nbsp; Size: ₹{float(_ht.get('pos_size_rs',0) or 0):,.0f}
  &nbsp;·&nbsp; Expiry: {_ht.get('expiry','—')} (DTE {_ht.get('dte_at_entry','—')} @ entry)
  &nbsp;·&nbsp; Logged: {_ht.get('logged_at','—')}
</div>""", unsafe_allow_html=True)
                if st.button("🗑 Delete from Historical", key=f"dh_{_ht['id']}_{_hidx}"):
                    st.session_state.opt_hist_pnl = [
                        t for t in st.session_state.opt_hist_pnl if t["id"] != _ht["id"]
                    ]
                    _save_hist_pnl(st.session_state.opt_hist_pnl)
                    st.rerun()

    st.markdown("""
<div style="font-family:'IBM Plex Mono',monospace;font-size:0.74rem;color:#444;
border-top:1px solid #1a1a1a;padding-top:10px;margin-top:8px;line-height:1.7;">
  <span style="color:#ff8c00;font-weight:700;">HOW LIVE PnL WORKS</span><br/>
  PnL auto-updates each time you reload options intel (⚡ LOAD) — spot price refreshes all active trades.<br/>
  Directional trades: (spot_now − spot_entry) / spot_entry × position size.<br/>
  Neutral strategies (condor/straddle): negative PnL when spot moves beyond expected move.<br/>
  <span style="color:#ff3b3b;">Approximation only — check your broker for actual leg-level P&L.</span><br/><br/>
  <span style="color:#ff8c00;font-weight:700;">WORKFLOW</span><br/>
  🎯 Trade tab → press <b>📋 LOG Trade</b> → appears as Active Trade here →
  reload to update PnL → <b>📁 → Historical PnL</b> to close.
</div>""", unsafe_allow_html=True)

# ── Gate all remaining content on opt_loaded ──────────────────────────────────
if not st.session_state.opt_loaded:
    with t_signal:
        st.markdown("""
<div style="background:#0d0d0d;border:1px solid #2a2a2a;padding:32px 24px;
font-family:'IBM Plex Mono',monospace;text-align:center;margin-top:20px;">
  <div style="color:#ff8c00;font-size:1.0rem;font-weight:700;letter-spacing:.15em;">
    ⚡ MONARCH OPTIONS INTELLIGENCE
  </div>
  <div style="color:#444;font-size:0.83rem;margin:12px 0 20px;">
    ──────────────────────────────────────────────────────────
  </div>
  <div style="color:#888;font-size:0.9rem;line-height:2.2;">
    1. Select <span style="color:#ff8c00;">Underlying</span> from the sidebar<br/>
    2. Choose <span style="color:#ff8c00;">Expiry Date</span> (auto-loaded from Upstox)<br/>
    3. Click <span style="color:#ff8c00;">⚡ LOAD OPTIONS INTEL</span><br/>
  </div>
  <div style="color:#444;font-size:0.79rem;margin-top:20px;line-height:1.8;">
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
# FIX: hv20/hv10 can be None when no OHLCV data is available — guard with fallback
hv20     = st.session_state.opt_hv20 or CFG["hv_fallback"]
hv10     = st.session_state.get("opt_hv10") or hv20
sym      = st.session_state.opt_symbol
expiry   = st.session_state.opt_expiry
dte      = st.session_state.opt_dte
step     = st.session_state.opt_step
atm_k    = st.session_state.get("opt_atm_k", atm_strike(spot, step))
r        = st.session_state.get("opt_rfr", CFG["rfr_default"] / 100.0)
# T: use the trading-day fraction stored at load time (NOT recomputed from calendar DTE)
T        = st.session_state.get("opt_T", dte / CFG["ann_days"])
q        = st.session_state.opt_div_yield.get(sym.upper(), 0.0)
ohlcv_df = st.session_state.get("opt_ohlcv", pd.DataFrame())

# New institutional features from session state
iv_pct       = st.session_state.get("opt_iv_percentile", 50.0)
regime_d     = st.session_state.get("opt_regime", {})
events_list  = st.session_state.get("opt_events", [])
liquidity_d  = st.session_state.get("opt_liquidity", {})
rs_nifty     = st.session_state.get("opt_rs_nifty", None)

# IV history for current symbol — used in Flow tab and Backtest tab
_iv_hist_sym_flow = st.session_state.opt_iv_history.get(sym, [])

bias       = bias_res.get("bias", "NEUTRAL")
bias_score = int(round(bias_res.get("score", 0)))

# IV Rank — from persistent disk-backed history
# Bootstrap when fewer than 3 observations: IV/HV ratio mapped via tanh.
# Denominator σ = 0.30 × HV is empirically calibrated:
#   NSE indices show IV/HV std of ~25-35%; 0.30 × HV gives a 1σ stretch of ≈30%.
#   At IV = 1.3×HV (one std rich) → IVR ≈ 75 (elevated sell zone). Principled.
_iv_hist_sym = st.session_state.opt_iv_history.get(sym, [])
if len(_iv_hist_sym) >= 3:
    ivr = iv_rank(_iv_hist_sym, atm_iv)
else:
    _hv_ref = hv20 if hv20 and hv20 > 0.01 else CFG["hv_fallback"]
    # Bootstrap: use rolling std of IV history if available, else adaptive estimate
    _iv_std = float(np.std(_iv_hist_sym)) if len(_iv_hist_sym) >= 2 else (_hv_ref * 0.30)
    _iv_std = max(_iv_std, _hv_ref * 0.05)   # floor: 5% of HV to prevent div/zero
    ivr = float(min(100.0, max(0.0,
        50.0 + 50.0 * math.tanh((atm_iv - _hv_ref) / _iv_std))))

v_lbl, v_act, v_col = vol_regime(ivr)

# BS helpers that close over live T, r, q, atm_iv — used by strategy engine and payoff builder
def _bs_c(k): return round(bs_price(spot, k, T, r, atm_iv, "call", q), 2)
def _bs_p(k): return round(bs_price(spot, k, T, r, atm_iv, "put",  q), 2)

# ── PROBABILISTIC DIRECTIONAL SCORE ──────────────────────────────────────
_lot = CFG["lot_sizes"].get(sym.upper(), CFG["lot_size_fallback"])
prob_score = compute_probabilistic_score(
    bias_res=bias_res, chain_df=chain_df, ohlcv_df=ohlcv_df,
    spot=spot, atm_iv=atm_iv, hv20=hv20,
    ivr=ivr, oi_d=oi_d, r=r, q=q, T=T, step=step)

# Persist prob_score for downstream access (e.g. IV smile chart range)
st.session_state["opt_prob_score"] = prob_score

# Record MC and factor direction signals for MC-blend calibration
_mc_dir_now  = float(prob_score.get("mc_direction", 0.0))
_fac_dir_now = float(prob_score.get("prob_up", 0.5) - prob_score.get("prob_down", 0.5))
_record_if_load("_calib_mc_dir_hist",  _mc_dir_now)
_record_if_load("_calib_fac_dir_hist", _fac_dir_now)

strat_recs = recommend_strategies(
    bias, v_lbl, dte, spot, atm_k, step, ivr, bias_score,
    bs_call_fn=_bs_c, bs_put_fn=_bs_p,
    front_iv=st.session_state.opt_multi_expiry[0]["atm_iv"] if len(st.session_state.opt_multi_expiry) >= 1 else None,
    back_iv =st.session_state.opt_multi_expiry[1]["atm_iv"] if len(st.session_state.opt_multi_expiry) >= 2 else None,
    expected_move=oi_d.get("atm_straddle", None),
    prob_score=prob_score,
    chain_df=chain_df, ohlcv_df=ohlcv_df,
    T=T, r=r, q=q, atm_iv=atm_iv, lot_size=_lot,
)

# ── Record signal snapshot for future outcome resolution ──────────────────────
# FIX B: Only record on genuine Load click (opt_load_id changed), not every rerender.
# Without this guard, every tab click appends a duplicate pending outcome.
_fs         = prob_score.get("feature_scores", {})
_top_s_snap = strat_recs[0] if strat_recs else {}
if st.session_state.get("opt_load_id",0) != st.session_state.get("_last_outcome_load_id",-1):
    st.session_state["_last_outcome_load_id"] = st.session_state.get("opt_load_id",0)
    _record_outcome(sym, {
        "spot":               spot,
        "raw_score":          prob_score.get("raw_score",    0.0),
        "prob_up":            float(prob_score.get("prob_up", 0.5)),  # FIX E: store directly
        "expected_move":      float(prob_score.get("expected_move", 0.0)),  # FIX E: for move_vs_iv
        "flow_score":         _fs.get("flow_score",          0.0),
        "mc_direction":       _mc_dir_now,
        "factor_direction":   _fac_dir_now,
        "ev_score":           float(_top_s_snap.get("ev_score",  0.0)),
        "dir_align":          float(_top_s_snap.get("dir_align", 0.5)),
        "safety_ratio":       float(_top_s_snap.get("safety_ratio", 2.0)),
        "ts_slope":           float(_top_s_snap.get("term_slope",   0.0) or 0.0),
        "pcr_level_z":        _fs.get("pcr_level_z",         0.0),
        "oi_skew_z":          _fs.get("oi_skew_z",            0.0),
        "mp_z":               _fs.get("mp_z",                 0.0),
        "vol_regime_z":       _fs.get("vol_regime_z",         0.0),
        "term_slope_z":       _fs.get("term_slope_z",         0.0),
        "rs_z":               _fs.get("rs_z",                 0.0),
        "rs_slope_z":         float((st.session_state.get("opt_rs_nifty") or {}).get(
                                   "rs_slope_z", 0.0) or 0.0),
        "ema_score":          _fs.get("trend_z",              0.0),
        "adx_score":          float((st.session_state.get("opt_regime_d") or {}).get("adx", 0.0)),
        "rsi_z":              max(-1.0, min(1.0, (_fs.get("rsi_v", 50.0) - 50.0) / 25.0)),
        "iv_pillar":          (st.session_state.get("opt_regime_d") or {}).get("pillars", {}).get("iv", 0.0),
        "adx_pillar":         (st.session_state.get("opt_regime_d") or {}).get("pillars", {}).get("adx", 0.0),
        "hv_accel_pillar":    (st.session_state.get("opt_regime_d") or {}).get("pillars", {}).get("hv_accel", 0.0),
        "gex_pillar":         (st.session_state.get("opt_regime_d") or {}).get("pillars", {}).get("gex", 0.0),
    }, horizon_days=max(4, dte // 2))

BIAS_COLORS = {
    "STRONGLY BULLISH":"#00d084","BULLISH":"#7dca84","NEUTRAL":"#ffb347",
    "BEARISH":"#ff7777","STRONGLY BEARISH":"#ff3b3b"
}
bc = BIAS_COLORS.get(bias, "#888")

# ── Forward signal log: append now that prob_score and strat_recs are available ──
_pending = st.session_state.get("_pending_signal_log", {})
if _pending and _pending.get("symbol") == sym:
    _top_s = strat_recs[0] if strat_recs else {}
    _fs_now2 = prob_score.get("feature_scores", {})
    _append_signal(
        symbol        = sym,
        prob_up       = prob_score.get("prob_up", 0.5),
        prob_down     = prob_score.get("prob_down", 0.5),
        flow_score    = prob_score.get("feature_scores", {}).get("flow_score", 0),
        flow_magnitude= prob_score.get("flow_magnitude", 0),
        raw_score     = prob_score.get("raw_score", 0.0),
        top_strategy  = _top_s.get("Strategy", "—"),
        strategy_ev   = _top_s.get("ev", 0),
        strategy_pop  = _top_s.get("pop", 0.5),
        strategy_kelly= _top_s.get("kelly", 0),
        expected_move = prob_score.get("expected_move", 0.0),
        atm_iv        = atm_iv,
        ivr           = ivr,
        bias          = bias,
        # Edge Audit full snapshot
        spot              = spot,
        hv20              = hv20,
        pcr               = float(oi_d.get("pcr_oi", 1.0) if oi_d else 1.0),
        oi_total          = float((chain_df["CE_OI"].sum() + chain_df["PE_OI"].sum()) if not chain_df.empty else 0),
        oi_skew           = float((chain_df["CE_OI"].sum() - chain_df["PE_OI"].sum()) if not chain_df.empty else 0),
        max_pain          = float(oi_d.get("max_pain", spot) if oi_d else spot),
        max_pain_dist_pct = float((spot - oi_d.get("max_pain", spot)) / (spot + 1e-9) * 100 if oi_d else 0),
        skew_pp           = float(oi_d.get("skew_pp", 0.0) or 0.0),
        term_slope        = float(st.session_state.opt_multi_expiry[1]["atm_iv"] - st.session_state.opt_multi_expiry[0]["atm_iv"]
                                  if len(st.session_state.opt_multi_expiry) >= 2 else 0.0),
        positioning_score = float(_fs_now2.get("pcr_level_z", 0.0)),
        vol_regime_score  = float(_fs_now2.get("vol_regime_z", 0.0)),
        trend_score       = float(_fs_now2.get("trend_z", 0.0)),
        intraday_score    = float((st.session_state.get("opt_intraday_signals") or {}).get("intraday_score", 0.0)),
        dte               = dte,
    )
    st.session_state["_pending_signal_log"] = {}  # clear pending flag

# ── TOP HEADER BAR ──
iv_vs_hv = (atm_iv - hv20)*100
iv_sign  = "+" if iv_vs_hv >= 0 else ""

# IV momentum: change vs rolling mean of session IV history
_iv_hist_sym2 = st.session_state.opt_iv_history.get(sym, [])
if len(_iv_hist_sym2) >= 5:
    _iv_ma5 = float(np.mean(_iv_hist_sym2[-5:]))
    iv_momentum = (atm_iv - _iv_ma5) * 100
    iv_mom_sign = "+" if iv_momentum >= 0 else ""
    iv_mom_str  = f"IV Δ(5): {iv_mom_sign}{iv_momentum:.1f}pp"   # pp = percentage points, not %
    iv_mom_c    = "#ff3b3b" if iv_momentum > 0.5 else ("#1e90ff" if iv_momentum < -0.5 else "#888")
else:
    iv_mom_str = "IV Δ: —"
    iv_mom_c   = "#555"

st.markdown(f"""
<div style="background:#0d0d0d;border:1px solid #2a2a2a;border-left:4px solid {bc};
padding:10px 16px;margin-bottom:6px;font-family:'IBM Plex Mono',monospace;">
  <div style="display:flex;gap:28px;align-items:center;flex-wrap:wrap;">
    <span style="color:#ff8c00;font-size:0.83rem;font-weight:700;letter-spacing:.15em;">⚡ {sym}</span>
    <span style="font-size:1.15rem;color:#e8e8e8;font-weight:700;">₹{spot:,.2f}</span>
    <span style="color:#888;font-size:0.83rem;">ATM {atm_k} · Step {step} · {expiry}</span>
    <span style="color:{bc};font-size:1.0rem;font-weight:700;">{bias} ({bias_score:+d})</span>
    <span style="color:#00d084;font-size:0.92rem;font-weight:700;">P(↑) {prob_score['prob_up']*100:.1f}%</span>
    <span style="color:#ff3b3b;font-size:0.92rem;font-weight:700;">P(↓) {prob_score['prob_down']*100:.1f}%</span>
    <span style="color:{v_col};font-size:0.88rem;font-weight:600;">EM ₹{prob_score['expected_move']:.0f}</span>
    <span style="color:#888;font-size:0.79rem;">ATM IV {atm_iv*100:.1f}% · HV20 {hv20*100:.1f}%</span>
    <span style="color:{iv_mom_c};font-size:0.77rem;">{iv_mom_str}</span>
    <span style="color:#555;font-size:0.77rem;">DTE: {dte}</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── TABS — 5 focused tabs matching the trader's decision workflow ──
# ⚡ Signal    — What is the market doing? (prob_up, flow, positioning, vol regime)
# 🎯 Trade    — What should I do? (EV strategies + position size + trade plan)
# 📋 Chain    — Raw data (option chain + OI + Greeks + payoff)
# 📅 Structure — Context (term structure + regime + forward signal log)
# 📐 Reference — Reference (maths + documentation)
def _resolve_edge_outcomes(ohlcv_df: pd.DataFrame, signal_log: list,
                           current_spot: float, current_iv: float,
                           current_skew: float, current_oi_total: float) -> list:
    """Fill in forward outcomes for unresolved signal log entries.
    Uses OHLCV close prices for accurate N-day forward returns.
    Mutates entries in-place and returns the updated log.
    """
    if not signal_log or ohlcv_df is None or ohlcv_df.empty:
        return signal_log

    # Build a date→close lookup from OHLCV
    try:
        _c = ohlcv_df.copy()
        if "date" in _c.columns:
            _c["_date"] = pd.to_datetime(_c["date"]).dt.date
        elif "datetime" in _c.columns:
            _c["_date"] = pd.to_datetime(_c["datetime"]).dt.date
        else:
            return signal_log
        _c = _c.sort_values("_date").reset_index(drop=True)
        _close_map = dict(zip(_c["_date"], _c["close"].astype(float)))
        _dates_sorted = sorted(_close_map.keys())
    except Exception:
        return signal_log

    today = datetime.now().date()
    modified = False

    for entry in signal_log:
        if entry.get("resolved"):
            continue
        try:
            sig_date = datetime.fromisoformat(entry["ts"]).date()
            sig_spot = float(entry.get("spot", 0.0))
            if sig_spot <= 0:
                continue

            # Find position of signal date in sorted dates
            future_dates = [d for d in _dates_sorted if d > sig_date]

            def _fwd_spot(n_days):
                if len(future_dates) >= n_days:
                    return _close_map.get(future_dates[n_days - 1])
                return None

            for n, key in [(1, "1d"), (2, "2d"), (3, "3d"), (5, "5d")]:
                fs = _fwd_spot(n)
                if fs and fs > 0:
                    entry[f"fwd_spot_{key}"] = round(fs, 2)
                    entry[f"fwd_ret_{key}"]  = round(math.log(fs / sig_spot), 5)

            # Mark resolved when horizon 1d is available and 5+ trading days have passed
            elapsed = int(np.busday_count(sig_date.isoformat(), today.isoformat()))
            if elapsed >= 5 and entry.get("fwd_ret_1d") is not None:
                # Use current session values for IV/skew/OI (approximate — best we have)
                entry["fwd_iv_1d"]   = round(current_iv * 100, 2)
                entry["fwd_skew_1d"] = round(current_skew, 3)
                entry["fwd_oi_1d"]   = round(current_oi_total, 0)
                entry["resolved"]    = True
                modified = True
        except Exception:
            continue

    if modified:
        _save_signal_log(signal_log)

    return signal_log


def _compute_edge_metrics(log: list, sym: str) -> dict:
    """Compute edge diagnostic metrics from resolved signal log entries.
    Returns a dict of group → metric rows for display in Edge Audit tab.
    """
    # Filter to resolved entries for this symbol
    resolved = [e for e in log
                if e.get("resolved") and e.get("symbol", "").upper() == sym.upper()
                and e.get("fwd_ret_1d") is not None]

    if len(resolved) < 3:
        return {}

    def _metrics(subset, horizon="1d"):
        if not subset:
            return None
        ret_key  = f"fwd_ret_{horizon}"
        spot_key = f"fwd_spot_{horizon}"
        rets     = [e[ret_key] for e in subset if e.get(ret_key) is not None]
        if not rets:
            return None

        n = len(rets)
        avg_ret  = float(np.mean(rets))

        # Direction accuracy: +score=bullish signal, check if return is positive
        dir_correct = []
        for e in subset:
            r = e.get(ret_key)
            if r is None: continue
            sig = e.get("bias", "NEUTRAL")
            if sig in ("BULLISH", "STRONGLY BULLISH"):
                dir_correct.append(1 if r > 0 else 0)
            elif sig in ("BEARISH", "STRONGLY BEARISH"):
                dir_correct.append(1 if r < 0 else 0)
            # NEUTRAL: check |r| vs expected move
        dir_acc = float(np.mean(dir_correct)) if dir_correct else 0.5

        # Move vs implied
        move_ratios = []
        for e in subset:
            sp0 = e.get("spot", 0.0)
            spN = e.get(spot_key)
            em  = e.get("expected_move", 0.0)
            if sp0 > 0 and spN and em > 0:
                abs_move = abs(spN - sp0)
                move_ratios.append(abs_move / em)
        avg_move_iv = float(np.mean(move_ratios)) if move_ratios else 0.0
        pct_gt_iv   = float(np.mean([1 if m > 1 else 0 for m in move_ratios])) if move_ratios else 0.5

        # IV change (1d only)
        iv_changes = []
        for e in subset:
            iv0 = e.get("atm_iv_pct")
            iv1 = e.get("fwd_iv_1d")
            if iv0 and iv1:
                iv_changes.append(iv1 - iv0)
        avg_iv_chg = float(np.mean(iv_changes)) if iv_changes else 0.0

        # Skew change
        skew_changes = []
        for e in subset:
            s0 = e.get("skew_pp")
            s1 = e.get("fwd_skew_1d")
            if s0 is not None and s1 is not None:
                skew_changes.append(s1 - s0)
        avg_skew_chg = float(np.mean(skew_changes)) if skew_changes else 0.0

        # OI change
        oi_changes = []
        for e in subset:
            o0 = e.get("oi_total")
            o1 = e.get("fwd_oi_1d")
            if o0 and o1:
                oi_changes.append(o1 - o0)
        avg_oi_chg = float(np.mean(oi_changes)) if oi_changes else 0.0

        # Max pain pinning: did price move toward max pain?
        toward_mp = []
        for e in subset:
            sp0 = e.get("spot", 0.0)
            spN = e.get(spot_key)
            mp  = e.get("max_pain", sp0)
            if sp0 > 0 and spN and mp > 0:
                dist_before = abs(sp0 - mp)
                dist_after  = abs(spN - mp)
                toward_mp.append(1 if dist_after < dist_before else 0)
        pct_toward_mp = float(np.mean(toward_mp)) if toward_mp else 0.5

        # Edge classification
        edges = []
        if dir_acc > 0.55 and len(dir_correct) >= 5:
            edges.append("Directional")
        if avg_move_iv > 1.05 and len(move_ratios) >= 5:
            edges.append("Long Vol")
        elif avg_move_iv < 0.90 and len(move_ratios) >= 5:
            edges.append("Short Vol")
        if avg_iv_chg < -0.5 and len(iv_changes) >= 5:
            edges.append("Sell IV")
        elif avg_iv_chg > 0.5 and len(iv_changes) >= 5:
            edges.append("Buy IV")
        if pct_toward_mp > 0.60 and len(toward_mp) >= 5:
            edges.append("Pinning")
        if abs(avg_skew_chg) > 0.3 and len(skew_changes) >= 5:
            edges.append("Skew")
        if not edges:
            edges.append("No Edge")

        return {
            "n":            n,
            "avg_ret":      round(avg_ret * 100, 3),   # in %
            "dir_acc":      round(dir_acc * 100, 1),
            "move_iv":      round(avg_move_iv, 3),
            "pct_gt_iv":    round(pct_gt_iv * 100, 1),
            "iv_chg":       round(avg_iv_chg, 2),
            "skew_chg":     round(avg_skew_chg, 3),
            "oi_chg":       round(avg_oi_chg / 1e6, 2),  # in millions
            "pct_toward_mp":round(pct_toward_mp * 100, 1),
            "edge":         " + ".join(edges),
        }

    # Build signal groups
    groups = {}

    # Direction groups
    bull = [e for e in resolved if e.get("bias") in ("BULLISH", "STRONGLY BULLISH")]
    bear = [e for e in resolved if e.get("bias") in ("BEARISH", "STRONGLY BEARISH")]
    neut = [e for e in resolved if e.get("bias") == "NEUTRAL"]
    if bull: groups["🟢 Bullish Signals"] = _metrics(bull)
    if bear: groups["🔴 Bearish Signals"] = _metrics(bear)
    if neut: groups["⚪ Neutral Signals"]  = _metrics(neut)

    # IV regime groups (ivr threshold = 60)
    hi_iv = [e for e in resolved if e.get("ivr", 50) >= 60]
    lo_iv = [e for e in resolved if e.get("ivr", 50) < 40]
    if hi_iv: groups["📈 High IV Regime (IVR≥60)"]  = _metrics(hi_iv)
    if lo_iv: groups["📉 Low IV Regime (IVR<40)"]    = _metrics(lo_iv)

    # PCR groups
    hi_pcr = [e for e in resolved if e.get("pcr", 1.0) >= 1.3]
    lo_pcr = [e for e in resolved if e.get("pcr", 1.0) < 0.8]
    if hi_pcr: groups["🐂 High PCR (≥1.3 bullish)"] = _metrics(hi_pcr)
    if lo_pcr: groups["🐻 Low PCR (<0.8 bearish)"]  = _metrics(lo_pcr)

    # Max pain proximity
    near_mp = [e for e in resolved if abs(e.get("max_pain_dist_pct", 5)) < 1.0]
    far_mp  = [e for e in resolved if abs(e.get("max_pain_dist_pct", 0)) >= 2.0]
    if near_mp: groups["📍 Near Max Pain (<1%)"]  = _metrics(near_mp)
    if far_mp:  groups["↔️ Far from Max Pain (≥2%)"] = _metrics(far_mp)

    # Flow score groups
    hi_flow = [e for e in resolved if abs(e.get("flow_score", 0)) >= 0.3]
    lo_flow = [e for e in resolved if abs(e.get("flow_score", 0)) < 0.1]
    if hi_flow: groups["⚡ High Flow (|score|≥0.3)"]  = _metrics(hi_flow)
    if lo_flow: groups["💤 Low Flow (|score|<0.1)"]    = _metrics(lo_flow)

    # Signal alignment groups
    flow_pos_aligned = [e for e in resolved
                        if abs(e.get("flow_score", 0)) >= 0.2
                        and abs(e.get("positioning_score", 0)) >= 0.2
                        and (e.get("flow_score", 0) * e.get("positioning_score", 0)) > 0]
    all_aligned = [e for e in flow_pos_aligned
                   if abs(e.get("vol_regime_score", 0)) >= 0.1]
    if flow_pos_aligned: groups["🔗 Flow + Positioning Aligned"] = _metrics(flow_pos_aligned)
    if all_aligned:      groups["🔗🔗 Flow + Pos + Vol Aligned"]  = _metrics(all_aligned)

    # All signals (baseline)
    groups["📊 All Signals"] = _metrics(resolved)

    # Remove None entries
    return {k: v for k, v in groups.items() if v is not None}


# ── TABS — aliases (tabs were already created before the load gate above) ──────
# ⚡ Signal    — What is the market doing?
# 🎯 Trade    — What should I do?
# 📋 Chain    — Raw data
# 📅 Structure — Context
# 📐 Reference — Reference
# 🔬 Edge Audit — Does the signal actually have edge?
# 📒 Tradebook  — Logged trades with live PnL

# Alias old tab names to new tabs so all existing rendering code works unchanged
t_ov       = t_signal    # Overview → Signal
t_dir      = t_signal    # Direction → Signal
t_strat    = t_trade     # Strategies → Trade
t_trade_plan = t_trade   # Trade Plan → Trade  (internal alias, used below)
t_chain_tab= t_chain     # Chain → Chain
t_greeks   = t_chain     # Greeks → Chain
t_oi       = t_chain     # OI Analysis → Chain
t_payoff   = t_chain     # Payoff → Chain
t_ts       = t_structure # Term Structure → Structure
t_flow     = t_structure # Flow & Skew → Structure
t_regime   = t_structure # Regime → Structure
t_backtest = t_structure # Signal Log (was Backtest) → Structure
t_math     = t_reference # Maths → Reference

# ══════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════
with t_ov:
    st.markdown("### ◼ Options Intelligence Summary")

    # ── Probabilistic core outputs ─────────────────────────────────────────
    _pu  = prob_score.get("prob_up", 0.5)
    _pd  = prob_score.get("prob_down", 0.5)
    _em  = prob_score.get("expected_move", 0.0)
    _emp = prob_score["expected_move_pct"]
    _rs  = prob_score.get("raw_score", 0.0)
    _pu_col  = "#00d084" if _pu > 0.55 else ("#ff3b3b" if _pu < 0.45 else "#ffb347")
    _pd_col  = "#ff3b3b" if _pd > 0.55 else ("#00d084" if _pd < 0.45 else "#ffb347")

    p1, p2, p3, p4, p5, p6, p7 = st.columns(7)
    p1.metric("Spot",        f"₹{spot:,.1f}")
    p2.metric("Prob Up ↑",   f"{_pu*100:.1f}%",  delta=f"score {_rs:+.3f}", delta_color="normal")
    p3.metric("Prob Down ↓", f"{_pd*100:.1f}%")
    p4.metric("Exp Move",    f"₹{_em:.0f}",       delta=f"±{_emp:.1f}%")
    p5.metric("ATM IV",      f"{atm_iv*100:.1f}%", delta=f"HV {hv20*100:.1f}%")
    p6.metric("IV Rank",     f"{ivr:.0f}",
              delta=f"Pctile {iv_pct:.0f}",
              help=f"IV Rank = (current−min)/(max−min) × 100 over session history. "
                   f"IV Pctile = % of sessions where IV was lower ({iv_pct:.0f}%). "
                   f"Gap between them is normal when historical IV had outlier spikes.")
    p7.metric("DTE",         str(dte))

    _fii_d = st.session_state.get("opt_fii_dii", {})
    if _fii_d.get("data_available"):
        _fi_sig    = float(_fii_d.get("combined_signal", 0.0))
        _fi_col    = "#00d084" if _fi_sig > 0.1 else "#ff3b3b" if _fi_sig < -0.1 else "#888"
        _fi_arrow  = "↑" if _fi_sig > 0.1 else "↓" if _fi_sig < -0.1 else "→"
        _fi_usdinr = float(_fii_d.get("usdinr",        0.0))
        _fi_vix    = float(_fii_d.get("indiavix",       0.0))
        _fi_usdsig = float(_fii_d.get("usdinr_signal", 0.0))
        _fi_vixsig = float(_fii_d.get("vix_signal",    0.0))
        _fi_date   = _fii_d.get("source_date", "")
        st.markdown(
            f"<div style='font-family:IBM Plex Mono,monospace;font-size:.79rem;"
            f"background:#111;border:1px solid #222;border-left:3px solid {_fi_col};"
            f"padding:5px 12px;margin-bottom:6px;'>"
            f"<span style='color:{_fi_col};font-weight:700;'>INST FLOW {_fi_arrow}</span>"
            f"&nbsp;&nbsp;USD/INR <b style='color:{_fi_col};'>{_fi_usdinr:.2f}</b>"
            f" (sig {_fi_usdsig:+.3f})&nbsp;·&nbsp;"
            f"India VIX <b>{_fi_vix:.1f}</b> (sig {_fi_vixsig:+.3f})&nbsp;·&nbsp;"
            f"<b style='color:{_fi_col};'>Combined: {_fi_sig:+.3f}</b>"
            f"&nbsp;·&nbsp;<span style='color:#555;font-size:.70rem;'>"
            f"{_fi_date} · USD/INR+VIX proxy</span></div>",
            unsafe_allow_html=True
        )

    # ── PART 12: Decision Panel ───────────────────────────────────────────────
    # Full output: Signals → Score → Probability → Edge → EV → Kelly → Strategy
    _imp_pu      = prob_score.get("implied_prob_up",  0.5)
    _imp_mv      = prob_score.get("implied_move_pct", _emp)
    _model_mv    = prob_score.get("model_move_pct",   _emp)
    _dir_edge    = prob_score.get("direction_edge",   0.0)
    _move_edge   = prob_score.get("move_edge_pct",    0.0)
    _vol_edge    = prob_score.get("vol_edge",         "—")
    _sig_str     = prob_score.get("signal_strength",  "No Edge")
    _edg_lbl     = prob_score.get("edge_label",       "No Edge")

    # Top strategy EV + Kelly for Decision Panel
    _dp_strat = strat_recs[0] if strat_recs else {}
    _dp_ev    = _dp_strat.get("ev", 0.0)
    _dp_kelly = _dp_strat.get("kelly", 0.0)
    _dp_kelly_pct = round(_dp_kelly * 100, 1)

    # Colours
    _ss_col   = ("#00d084" if "Bullish" in _sig_str
                 else "#ff3b3b" if "Bearish" in _sig_str else "#888")
    _de_col   = ("#00d084" if _dir_edge > 0.05
                 else "#ff3b3b" if _dir_edge < -0.05 else "#888")
    _me_col   = ("#00d084" if _move_edge > 0 else "#ff3b3b")
    _ve_col   = ("#1e90ff" if _vol_edge == "BUY" else "#ff8c00")
    _ev_col   = ("#00d084" if _dp_ev > 0 else "#ff3b3b")
    _pu_col2  = ("#00d084" if _pu > 0.55 else "#ff3b3b" if _pu < 0.45 else "#ffb347")

    def _dp_cell(label, value, color, sublabel=""):
        sub = f'<div style="color:#444;font-size:0.68rem;margin-top:1px;">{sublabel}</div>' if sublabel else ""
        return (f'<div style="min-width:90px;">'
                f'<div style="color:#555;font-size:0.68rem;letter-spacing:.07em;'
                f'text-transform:uppercase;">{label}</div>'
                f'<div style="color:{color};font-size:0.90rem;font-weight:700;'
                f'line-height:1.2;">{value}</div>{sub}</div>')

    _sign_de  = "+" if _dir_edge >= 0 else ""
    _sign_me  = "+" if _move_edge >= 0 else ""

    st.markdown(f"""
<div style="background:#080808;border:1px solid #2a2a2a;border-left:4px solid {_ss_col};
padding:12px 18px;margin:6px 0 6px;font-family:'IBM Plex Mono',monospace;">
  <div style="color:#555;font-size:0.70rem;letter-spacing:.10em;margin-bottom:8px;">
    ◼ DECISION PANEL &nbsp;·&nbsp; Signals → Score → Probability → Edge → EV → Kelly
  </div>
  <div style="display:flex;flex-wrap:wrap;gap:18px;align-items:flex-start;">
    {_dp_cell("Signal Strength", _sig_str,  _ss_col)}
    {_dp_cell("Prob Up ↑",  f"{_pu*100:.1f}%",   _pu_col2,  f"score {_rs:+.3f}")}
    {_dp_cell("Prob Down ↓",f"{(1-_pu)*100:.1f}%", "#ff3b3b")}
    {_dp_cell("Implied Move", f"±{_imp_mv:.1f}%", "#888",  f"IV×√(DTE/252)")}
    {_dp_cell("Model Move",   f"±{_model_mv:.1f}%", _me_col, f"EM×(0.5+|score|)")}
    {_dp_cell("Dir Edge",  f"{_sign_de}{_dir_edge*100:.1f}pp", _de_col, _edg_lbl)}
    {_dp_cell("Move Edge",    f"{_sign_me}{_move_edge:.1f}%",  _me_col, "model−implied")}
    {_dp_cell("Vol Edge",  _vol_edge,  _ve_col, "buy/sell options")}
    {_dp_cell("Top EV",   f"₹{_dp_ev:+.0f}", _ev_col, _dp_strat.get("Strategy","—")[:16])}
    {_dp_cell("Kelly Size", f"{_dp_kelly_pct:.1f}%", "#ff8c00", "of capital")}
  </div>
</div>""", unsafe_allow_html=True)

    # ── Probability bar ───────────────────────────────────────────────────
    _bar_up   = int(_pu * 100)
    _bar_down = 100 - _bar_up
    st.markdown(f"""
<div style="font-family:'IBM Plex Mono',monospace;margin:8px 0 4px;">
  <div style="display:flex;height:22px;border-radius:0;overflow:hidden;border:1px solid #2a2a2a;">
    <div style="width:{_bar_up}%;background:{_pu_col};display:flex;align-items:center;
                justify-content:center;font-size:0.74rem;font-weight:700;color:#000;
                min-width:30px;">↑{_bar_up}%</div>
    <div style="width:{_bar_down}%;background:{_pd_col};display:flex;align-items:center;
                justify-content:center;font-size:0.74rem;font-weight:700;color:#000;
                min-width:30px;">↓{_bar_down}%</div>
  </div>
  <div style="display:flex;justify-content:space-between;color:#555;font-size:0.72rem;margin-top:2px;">
    <span>Probability of upside move by expiry</span>
    <span>logistic(raw_score={_rs:+.3f})</span>
  </div>
</div>""", unsafe_allow_html=True)

    # Events + Liquidity + Regime banner
    if events_list or liquidity_d or regime_d:
        ev_cols = st.columns(3)
        with ev_cols[0]:
            _reg_color = regime_d.get("color", "#888")
            _reg_lbl   = regime_d.get("regime", "—")
            _adx       = regime_d.get("adx", 0)
            st.markdown(f"""<div style="background:#0d0d0d;border:1px solid #2a2a2a;
border-left:4px solid {_reg_color};padding:10px 14px;font-family:'IBM Plex Mono',monospace;">
  <div style="color:#555;font-size:0.76rem;letter-spacing:.08em;">MARKET REGIME</div>
  <div style="color:{_reg_color};font-size:0.96rem;font-weight:700;">{_reg_lbl}</div>
  <div style="color:#888;font-size:0.78rem;">ADX {_adx:.1f} · {regime_d.get('trend','—')} · {regime_d.get('vol','—')}</div>
</div>""", unsafe_allow_html=True)
        with ev_cols[1]:
            _liq_col   = liquidity_d.get("color", "#888")
            _liq_score = liquidity_d.get("liquid_score", 0)
            _liq_spr   = liquidity_d.get("atm_spread_pct", 0)
            st.markdown(f"""<div style="background:#0d0d0d;border:1px solid #2a2a2a;
border-left:4px solid {_liq_col};padding:10px 14px;font-family:'IBM Plex Mono',monospace;">
  <div style="color:#555;font-size:0.76rem;letter-spacing:.08em;">LIQUIDITY</div>
  <div style="color:{_liq_col};font-size:0.96rem;font-weight:700;">Score {_liq_score}/100</div>
  <div style="color:#888;font-size:0.78rem;">ATM spread {_liq_spr:.1f}% · {liquidity_d.get('verdict','—')[:30]}</div>
</div>""", unsafe_allow_html=True)
        with ev_cols[2]:
            _ev_count  = len([e for e in events_list if e.get("days_away", 99) <= 14])
            _ev_col    = "#ff3b3b" if _ev_count > 0 else "#555"
            _ev_txt    = events_list[0]["event"] if events_list else "None detected"
            st.markdown(f"""<div style="background:#0d0d0d;border:1px solid #2a2a2a;
border-left:4px solid {_ev_col};padding:10px 14px;font-family:'IBM Plex Mono',monospace;">
  <div style="color:#555;font-size:0.76rem;letter-spacing:.08em;">EVENTS IN WINDOW</div>
  <div style="color:{_ev_col};font-size:0.96rem;font-weight:700;">{_ev_count} within 14 days</div>
  <div style="color:#888;font-size:0.78rem;">{_ev_txt[:40]}</div>
</div>""", unsafe_allow_html=True)

    st.divider()

    # ── Adaptive Engine Calibration Status ───────────────────────────────────
    _calib_now   = _get_calib(sym)
    _n_params    = len(_calib_now)
    _sym_store   = st.session_state.get(_CALIB_STORE_KEY, {})
    _n_symbols   = len([k for k in _sym_store if not k.startswith("_")])
    _total_obs   = sum(len(v) for k, v in st.session_state.items()
                       if k.startswith(f"{sym.upper()}:_calib_") and isinstance(v, list))
    _real_ret_h  = _get_hist("_calib_realised_ret_hist", sym)
    _n_real      = len(_real_ret_h)
    _pending_n   = len(st.session_state.get(f"_outcome_pending_{sym.upper()}", []))
    _fw_live     = prob_score.get("factor_weights", CFG["factor_weights"])
    _mc_blend_live = round(_calib("mc_blend") * 100)
    _sharp_live    = round(_calib("logistic_sharpness"), 2)
    _calib_color   = "#00d084" if _n_real >= _dynamic_min_obs() else ("#ffb347" if _n_params >= 1 else "#555")
    _calib_status  = (f"LIVE — {_n_real} real outcomes · {_pending_n} pending"
                      if _n_real >= _dynamic_min_obs()
                      else f"WARMING UP — {_n_real}/{_dynamic_min_obs()} real outcomes needed")

    # ── Always-visible compact calibration progress bar ───────────────────────
    _pct_cal_bar  = min(100, int(_n_real / max(_dynamic_min_obs(), 1) * 100))
    _flow_pcr_n   = len(st.session_state.get("_flow_pcr_hist", []))
    _flow_pct_bar = min(100, int(_flow_pcr_n / 10 * 100))   # 10 loads = full flow history
    _bar_bg       = "#00d084" if _pct_cal_bar >= 100 else ("#ffb347" if _pct_cal_bar >= 50 else "#ff3b3b")
    _flow_bg      = "#00d084" if _flow_pct_bar >= 100 else ("#ffb347" if _flow_pct_bar >= 30 else "#ff3b3b")
    _hist_saved   = "✔ saved" if os.path.exists(_HIST_FILE) else "not saved yet"
    st.markdown(f"""
<div style="background:#0d0d0d;border:1px solid #2a2a2a;padding:8px 14px;
font-family:'IBM Plex Mono',monospace;font-size:0.76rem;margin-bottom:6px;">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:5px;">
    <span style="color:#555;letter-spacing:.06em;">⚙ ADAPTIVE ENGINE</span>
    <span style="color:{_calib_color};font-weight:700;">{_calib_status}</span>
    <span style="color:#444;font-size:0.70rem;">hist {_hist_saved}</span>
  </div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;">
    <div>
      <div style="color:#555;font-size:0.70rem;margin-bottom:2px;">
        CALIBRATION  {_n_real}/{_dynamic_min_obs()} real outcomes
      </div>
      <div style="background:#1a1a1a;height:6px;border-radius:2px;overflow:hidden;">
        <div style="width:{_pct_cal_bar}%;height:100%;background:{_bar_bg};border-radius:2px;"></div>
      </div>
    </div>
    <div>
      <div style="color:#555;font-size:0.70rem;margin-bottom:2px;">
        FLOW HISTORY  {_flow_pcr_n}/10 loads · {_pending_n} pending outcomes
      </div>
      <div style="background:#1a1a1a;height:6px;border-radius:2px;overflow:hidden;">
        <div style="width:{_flow_pct_bar}%;height:100%;background:{_flow_bg};border-radius:2px;"></div>
      </div>
    </div>
  </div>
</div>""", unsafe_allow_html=True)
    _hist_exists   = os.path.exists(_HIST_FILE)
    _hist_kb       = round(os.path.getsize(_HIST_FILE) / 1024, 1) if _hist_exists else 0

    with st.expander("⚙ Adaptive Engine — Live Calibration State", expanded=False):
        cc1, cc2, cc3, cc4 = st.columns(4)
        cc1.metric("Real Outcomes",     str(_n_real),        help="Resolved signal→return pairs from actual market moves")
        cc2.metric("Pending Snapshots", str(_pending_n),     help="Signals recorded, awaiting horizon to elapse")
        cc3.metric("MC Blend Weight",   f"{_mc_blend_live}%",help="MC direction weight in final prob blend (learned)")
        cc4.metric("Logistic Sharpness",str(_sharp_live),    help="Calibrated raw_score scaling for prob_up (learned)")

        # Persistence status
        _persist_col = "#00d084" if _hist_exists else "#ff3b3b"
        _persist_lbl = f"SAVED ({_hist_kb} KB)" if _hist_exists else "NOT SAVED YET — click Load first"
        st.markdown(
            f"""<div style="font-family:'IBM Plex Mono',monospace;font-size:0.78rem;
color:#555;padding:6px 0 2px;">
<span style="color:#888;">SYMBOL:</span> <span style="color:#ff8c00;font-weight:700;">{sym}</span>
&nbsp;·&nbsp;
<span style="color:#888;">STATUS:</span> <span style="color:{_calib_color};font-weight:700;">{_calib_status}</span>
&nbsp;·&nbsp;
<span style="color:#888;">HIST DISK:</span> <span style="color:{_persist_col};font-weight:700;">{_persist_lbl}</span>
&nbsp;·&nbsp;
<span style="color:#555;">{_n_symbols} symbol(s) · weights: {_CALIB_FILE} · histories: {_HIST_FILE}</span>
</div>""", unsafe_allow_html=True)

        # Calibration quality bar — fraction of parameters with real data vs priors
        _pct_calibrated = min(100, int(_n_real / max(_dynamic_min_obs(), 1) * 100))
        _bar_col = "#00d084" if _pct_calibrated >= 100 else ("#ffb347" if _pct_calibrated >= 50 else "#ff3b3b")
        st.markdown(f"""
<div style="font-family:'IBM Plex Mono',monospace;margin:6px 0 4px;">
  <div style="display:flex;height:16px;border:1px solid #2a2a2a;overflow:hidden;">
    <div style="width:{_pct_calibrated}%;background:{_bar_col};"></div>
    <div style="width:{100-_pct_calibrated}%;background:#1a1a1a;"></div>
  </div>
  <div style="color:#555;font-size:0.72rem;margin-top:2px;">
    Learning progress: {_pct_calibrated}% · {_n_real} real outcomes / {_dynamic_min_obs()} needed per parameter
  </div>
</div>""", unsafe_allow_html=True)

        # Live factor weights table
        _fw_rows = "".join(
            f"<tr><td style='color:#ff8c00;padding:3px 10px;'>{k.upper()}</td>"
            f"<td style='color:#e8e8e8;padding:3px 10px;text-align:right;'>"
            f"{v*100:.1f}%</td>"
            f"<td style='color:#555;padding:3px 10px;font-size:0.74rem;'>"
            f"prior={CFG['factor_weights'].get(k,0)*100:.0f}%</td></tr>"
            for k, v in sorted(_fw_live.items(), key=lambda x: -x[1])
        )
        st.markdown(f"""
<table style="font-family:'IBM Plex Mono',monospace;font-size:0.82rem;border-collapse:collapse;
width:100%;background:#0d0d0d;border:1px solid #2a2a2a;margin-top:6px;">
  <thead><tr>
    <th style="color:#ff8c00;padding:4px 10px;text-align:left;border-bottom:1px solid #2a2a2a;">FACTOR</th>
    <th style="color:#ff8c00;padding:4px 10px;text-align:right;border-bottom:1px solid #2a2a2a;">LIVE WEIGHT</th>
    <th style="color:#555;padding:4px 10px;text-align:left;border-bottom:1px solid #2a2a2a;">COLD-START PRIOR</th>
  </tr></thead>
  <tbody>{_fw_rows}</tbody>
</table>""", unsafe_allow_html=True)

        # Sub-model parameters
        _sub_w = {
            "EMA vs ADX (trend)":         f"{_calib('trend_ema_vs_adx')*100:.1f}% EMA",
            "ADX vs RSI (trend residual)": f"{_calib('adx_vs_rsi_within_trend')*100:.1f}% ADX",
            "Positioning: PCR:OI:MP":      " : ".join(f"{w*100:.0f}%" for w in _calib_vec("positioning_pcr_vs_oi_vs_mp")),
            "RS: level vs slope":          f"{_calib('rs_level_vs_slope')*100:.1f}% level",
            "Vol-regime damp":             f"{_calib('vol_regime_damp'):.3f}",
            "Safety sigmoid sharpness":    f"{_calib('safety_sigmoid_sharpness'):.3f}",
            "EV vs dir_align":             f"{_calib('ev_score_vs_dir_align')*100:.1f}% EV",
            "ADX weak-trend fraction":     f"{_calib('adx_weak_frac'):.3f}",
            "Term-structure tanh scale":   f"{_calib('ts_tanh_scale'):.1f}",
            "EV tanh scale":               f"{_calib('ev_tanh_scale'):.3f}",
            "HV accel stretch":            f"{_calib('hv_accel_stretch'):.3f}",
            "Max-pain gravity":            f"{_calib('mp_gravity'):.3f}",
            "Liq: spread:OI:vol":          " : ".join([
                f"{_calib('liq_spread_w')*100:.0f}%",
                f"{_calib('liq_oi_w')*100:.0f}%",
                f"{_calib('liq_vol_w')*100:.0f}%",
            ]),
        }
        _sw_rows = "".join(
            f"<tr><td style='color:#888;padding:2px 10px;'>{k}</td>"
            f"<td style='color:#e8e8e8;padding:2px 10px;text-align:right;'>{v}</td></tr>"
            for k, v in _sub_w.items()
        )
        st.markdown(f"""
<table style="font-family:'IBM Plex Mono',monospace;font-size:0.79rem;border-collapse:collapse;
width:100%;background:#0d0d0d;border:1px solid #2a2a2a;margin-top:8px;">
  <thead><tr>
    <th style="color:#ff8c00;padding:4px 10px;text-align:left;border-bottom:1px solid #2a2a2a;">SUB-MODEL PARAMETER</th>
    <th style="color:#ff8c00;padding:4px 10px;text-align:right;border-bottom:1px solid #2a2a2a;">CALIBRATED VALUE</th>
  </tr></thead>
  <tbody>{_sw_rows}</tbody>
</table>""", unsafe_allow_html=True)

        # Regime pillar confidence
        # Pillar breakdown
        _pillars_d = regime_d.get("pillars", {})
        st.caption(
            f"Pillars: iv={_pillars_d.get('iv',0):+.2f} "
            f"iv_hv={_pillars_d.get('iv_hv',0):+.2f} "
            f"hv_acc={_pillars_d.get('hv_accel',0):+.2f} "
            f"adx={_pillars_d.get('adx',0):+.2f} "
            f"ts={_pillars_d.get('ts_slope',0):+.2f} "
            f"xasset={_pillars_d.get('cross_asset',0):+.2f} "
            f"gex={_pillars_d.get('gex',0):+.2f} · "
            f"Conf: {regime_d.get('regime_confidence', 0):.2f}"
        )
        # Shock + acceleration status
        _shock_d  = st.session_state.get("opt_shock", {})
        _flow_d   = (st.session_state.get("opt_bias") or {}).get("flow", {})
        _fa       = float(_flow_d.get("flow_acceleration", 0.0))
        _oia      = float(_flow_d.get("oi_accel", 0.0))
        _pcra     = float(_flow_d.get("pcr_accel", 0.0))
        if _shock_d.get("shock_detected"):
            st.warning(
                f"⚡ SHOCK DETECTED — intensity {_shock_d.get('shock_intensity',0):.2f} · "
                f"IV z={_shock_d.get('iv_zscore',0):+.2f} "
                f"OI z={_shock_d.get('oi_zscore',0):+.2f} "
                + ("· CA shock" if _shock_d.get("ca_shock") else ""),
                icon="⚠️"
            )
        st.caption(
            f"Flow accel: composite={_fa:+.3f} "
            f"OI={_oia:+.3f} PCR={_pcra:+.3f}"
        )

        # Real outcome performance summary
        if _n_real >= 5:
            _ret_arr = np.array(_real_ret_h, dtype=float)
            _win_r   = float((_ret_arr > 0).mean() * 100)
            _avg_ret = float(_ret_arr.mean() * 100)
            _sharpe  = float(_ret_arr.mean() / (_ret_arr.std() + 1e-9) * np.sqrt(252 / 4))
            st.markdown(f"""
<div style="background:#0d0d0d;border:1px solid #2a2a2a;padding:10px 14px;
font-family:'IBM Plex Mono',monospace;font-size:0.80rem;margin-top:8px;">
  <div style="color:#ff8c00;font-weight:700;margin-bottom:6px;">REAL OUTCOME SUMMARY — {sym}</div>
  <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;">
    <div><div style="color:#555;font-size:0.72rem;">WIN RATE</div>
         <div style="color:{'#00d084' if _win_r>50 else '#ff3b3b'};font-weight:700;">{_win_r:.1f}%</div></div>
    <div><div style="color:#555;font-size:0.72rem;">AVG 4-DAY RETURN</div>
         <div style="color:{'#00d084' if _avg_ret>0 else '#ff3b3b'};font-weight:700;">{_avg_ret:+.2f}%</div></div>
    <div><div style="color:#555;font-size:0.72rem;">ANNUALISED SHARPE</div>
         <div style="color:{'#00d084' if _sharpe>0.5 else '#ffb347'};font-weight:700;">{_sharpe:.2f}</div></div>
  </div>
</div>""", unsafe_allow_html=True)

    # ── Intraday Signal Panel ─────────────────────────────────────────────────
    _intra = st.session_state.get("opt_intraday_signals", {})
    if _intra.get("intraday_available"):
        _i_score  = float(_intra.get("intraday_score", 0.0))
        _i_vwap   = float(_intra.get("vwap", spot))
        _i_shi    = float(_intra.get("session_high", spot))
        _i_slo    = float(_intra.get("session_low", spot))
        _i_candles= int(_intra.get("candles_so_far", 0))
        _i_range  = float(_intra.get("intraday_range_pct", 0.0))
        _i_vol_acc= float(_intra.get("volume_acceleration", 0.0))
        _i_oi_dir = float(_intra.get("oi_build_direction", 0.0))
        _i_open_m = float(_intra.get("opening_momentum", 0.0))
        _i_vwap_p = float(_intra.get("vwap_position", 0.0))
        _i_lunch  = float(_intra.get("lunch_reversal", 0.0))
        _i_blend  = round(_calib("intra_blend_weight") * 100)

        # ── Adaptive display thresholds — derived from rolling signal histories ──
        # "Meaningful" signal = above the 70th percentile of its own history.
        # This makes the colour coding adaptive to each instrument's intraday vol.
        def _adaptive_col(val: float, hist_key: str,
                           bull_col="#00d084", bear_col="#ff3b3b", neutral_col="#888") -> str:
            hist = _get_hist(hist_key, sym)
            if len(hist) >= 5:
                p70 = float(np.percentile(np.abs(hist), 70))
            else:
                # Cold-start: treat any non-trivial signal (>10% of range) as meaningful
                p70 = 0.10
            return bull_col if val > p70 else (bear_col if val < -p70 else neutral_col)

        _score_col = _adaptive_col(_i_score,   "_calib_intraday_score_hist")
        _vwap_col  = "#00d084" if spot > _i_vwap else "#ff3b3b"
        _vol_col   = _adaptive_col(_i_vol_acc, "_calib_intra_volume_acceleration_hist")
        _oi_col    = _adaptive_col(_i_oi_dir,  "_calib_intra_oi_build_hist")

        # Minimum signal magnitude to show as "active" (50th pctile of abs signal history)
        def _min_active(hist_key: str) -> float:
            hist = _get_hist(hist_key, sym)
            return float(np.percentile(np.abs(hist), 50)) if len(hist) >= 5 else 0.01

        def _bar(val: float, hist_key: str, width: int = 80) -> str:
            """Render a mini horizontal bar. Colour is adaptive to signal history."""
            pct = int((val + 1) / 2 * 100)
            col = _adaptive_col(val, hist_key)
            return (f'<div style="width:{width}px;height:8px;background:#1a1a1a;border-radius:2px;display:inline-block;vertical-align:middle;">'
                    f'<div style="width:{pct}%;height:100%;background:{col};border-radius:2px;"></div></div>')

        with st.expander(f"📊 Intraday Live Signals — {sym}  [{_i_candles} × 5-min candles]  "
                         f"Score: {_i_score:+.2f}", expanded=True):
            ic1, ic2, ic3, ic4 = st.columns(4)
            ic1.metric("Intraday Score",   f"{_i_score:+.3f}",   help=f"Blended into final prob at {_i_blend}% weight")
            ic2.metric("VWAP",             f"₹{_i_vwap:,.1f}",   delta=f"{'above' if spot > _i_vwap else 'below'} ₹{abs(spot - _i_vwap):.1f}", delta_color="normal")
            ic3.metric("Session Range",    f"{_i_range:.2f}%",   help=f"High ₹{_i_shi:,.1f} · Low ₹{_i_slo:,.1f}")
            ic4.metric("Vol Acceleration", f"{_i_vol_acc:+.2f}", help=f"Last {CFG['intra_recent_candles']} candles vs session avg. +ve = surge")

            _ps_val = float(_intra.get("price_structure", 0.0))
            st.markdown(f"""
<div style="background:#0d0d0d;border:1px solid #2a2a2a;padding:12px 16px;
font-family:'IBM Plex Mono',monospace;font-size:0.80rem;margin-top:8px;">
  <div style="color:#ff8c00;font-weight:700;margin-bottom:10px;font-size:0.84rem;">
    INTRADAY SIGNAL BREAKDOWN  ·  blend weight {_i_blend}%  ·  {_i_candles} candles
  </div>
  <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;">
    <div>
      <div style="color:#555;font-size:0.72rem;margin-bottom:3px;">OPENING MOMENTUM</div>
      <div style="color:{_adaptive_col(_i_open_m, "_calib_intra_opening_momentum_hist")};font-weight:700;">{_i_open_m:+.3f}</div>
      {_bar(_i_open_m, "_calib_intra_opening_momentum_hist")}
      <div style="color:#444;font-size:0.68rem;margin-top:2px;">First {CFG["intra_opening_candles"]} candles thrust</div>
    </div>
    <div>
      <div style="color:#555;font-size:0.72rem;margin-bottom:3px;">VWAP POSITION</div>
      <div style="color:{_vwap_col};font-weight:700;">{_i_vwap_p:+.3f}</div>
      {_bar(_i_vwap_p, "_calib_intra_vwap_position_hist")}
      <div style="color:#444;font-size:0.68rem;margin-top:2px;">Spot vs VWAP · {'ABOVE' if _i_vwap_p>0 else 'BELOW'}</div>
    </div>
    <div>
      <div style="color:#555;font-size:0.72rem;margin-bottom:3px;">VOLUME ACCELERATION</div>
      <div style="color:{_vol_col};font-weight:700;">{_i_vol_acc:+.3f}</div>
      {_bar(_i_vol_acc, "_calib_intra_volume_acceleration_hist")}
      <div style="color:#444;font-size:0.68rem;margin-top:2px;">Last {CFG["intra_recent_candles"]} candles vs avg</div>
    </div>
    <div>
      <div style="color:#555;font-size:0.72rem;margin-bottom:3px;">OI BUILD DIRECTION</div>
      <div style="color:{_oi_col};font-weight:700;">{_i_oi_dir:+.3f}</div>
      {_bar(_i_oi_dir, "_calib_intra_oi_build_hist")}
      <div style="color:#444;font-size:0.68rem;margin-top:2px;">PE OI change − CE OI change</div>
    </div>
    <div>
      <div style="color:#555;font-size:0.72rem;margin-bottom:3px;">PRICE STRUCTURE</div>
      <div style="color:{_adaptive_col(_ps_val, "_calib_intra_price_structure_hist")};font-weight:700;">{_ps_val:+.3f}</div>
      {_bar(_ps_val, "_calib_intra_price_structure_hist")}
      <div style="color:#444;font-size:0.68rem;margin-top:2px;">HH/LL ({CFG["intra_structure_candles"]} candles)</div>
    </div>
    <div>
      <div style="color:#555;font-size:0.72rem;margin-bottom:3px;">LUNCH REVERSAL</div>
      <div style="color:{_adaptive_col(_i_lunch, "_calib_intra_lunch_reversal_hist", bull_col="#ff8c00", bear_col="#ff8c00")};font-weight:700;">{_i_lunch:+.3f}</div>
      {_bar(_i_lunch, "_calib_intra_lunch_reversal_hist")}
      <div style="color:#444;font-size:0.68rem;margin-top:2px;">Post-candle-{CFG["intra_lunch_candle_start"]} shift</div>
    </div>
  </div>
</div>""", unsafe_allow_html=True)
    else:
        st.caption("📊 Intraday signals: not available (pre-market, after-hours, or Upstox API unavailable)")

    ov1, ov2 = st.columns(2)
    with ov1:
        # Vol regime box
        st.markdown(f"""
<div style="background:#0d0d0d;border:1px solid {v_col};padding:12px 16px;
font-family:'IBM Plex Mono',monospace;height:100%;">
  <div style="color:{v_col};font-size:0.9rem;font-weight:700;letter-spacing:.1em;margin-bottom:6px;">
    VOL REGIME: {v_lbl}
  </div>
  <div style="color:#e8e8e8;font-size:0.87rem;line-height:1.7;">{v_act}</div>
  <div style="margin-top:8px;color:#555;font-size:0.77rem;">
    ATM Straddle: ₹{oi_d.get('atm_straddle',0):.1f} &nbsp;·&nbsp;
    Exp Move ±1σ: ±{oi_d.get('exp_move_pct',0):.1f}% &nbsp;·&nbsp;
    ±2σ: ±{oi_d.get('exp_move_2sd_pct', oi_d.get('exp_move_pct',0)*2):.1f}% &nbsp;·&nbsp;
    IV Rank: {ivr:.0f} &nbsp;·&nbsp; IV Pctile: {iv_pct:.0f}
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
  <div style="color:#ff8c00;font-size:0.83rem;font-weight:700;letter-spacing:.1em;margin-bottom:8px;">OI SNAPSHOT</div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;">
    <div><div style="color:#555;font-size:0.74rem;">MAX PAIN</div>
         <div style="color:#ff8c00;font-size:1.0rem;font-weight:700;">₹{oi_d.get('max_pain',0):,.0f}</div></div>
    <div><div style="color:#555;font-size:0.74rem;">PCR (OI)</div>
         <div style="color:{pcr_c};font-size:1.0rem;font-weight:700;">{oi_d.get('pcr_oi',0):.3f}</div></div>
    <div><div style="color:#555;font-size:0.74rem;">CALL WALL</div>
         <div style="color:#ff3b3b;font-size:1.0rem;font-weight:600;">₹{oi_d.get('call_wall',0):,.0f}</div></div>
    <div><div style="color:#555;font-size:0.74rem;">PUT WALL</div>
         <div style="color:#00d084;font-size:1.0rem;font-weight:600;">₹{oi_d.get('put_wall',0):,.0f}</div></div>
    <div><div style="color:#555;font-size:0.74rem;">GEX REGIME</div>
         <div style="color:{_gex_c};font-size:1.0rem;font-weight:600;">{_gex_lbl} ({_gex_net:+,.0f})</div></div>
    <div><div style="color:#555;font-size:0.74rem;">IV SKEW (±1 strike)</div>
         <div style="color:{_skew_c};font-size:1.0rem;font-weight:600;">{_skew_str}</div></div>
  </div>
  <div style="margin-top:8px;color:{pcr_c};font-size:0.79rem;">{oi_d.get('pcr_signal','—')}</div>
  <div style="margin-top:4px;color:#555;font-size:0.74rem;">Gamma Flip: ₹{oi_d.get('gamma_flip',spot):,.0f}</div>
</div>""", unsafe_allow_html=True)

    # Best strategy card
    if strat_recs:
        best = strat_recs[0]
        st.divider()
        st.markdown(f"""
<div style="background:#0d1a00;border:1px solid #ff8c00;border-top:3px solid #ff8c00;
padding:14px 18px;font-family:'IBM Plex Mono',monospace;">
  <div style="color:#ff8c00;font-size:0.77rem;letter-spacing:.12em;font-weight:700;">⭐ TOP RECOMMENDED STRATEGY</div>
  <div style="color:#e8e8e8;font-size:1.0rem;font-weight:700;margin:8px 0 4px;">{best['Strategy']}</div>
  <div style="color:#7ec8e3;font-size:0.92rem;margin-bottom:6px;">LEGS: {best['Legs']}</div>
  <div style="color:#aaa;font-size:0.82rem;line-height:1.6;">{best['Rationale']}</div>
  <div style="display:flex;gap:24px;margin-top:10px;flex-wrap:wrap;">
    <span style="color:#ff3b3b;font-size:0.79rem;">⬇ Risk: {best['Max Risk']}</span>
    <span style="color:#00d084;font-size:0.79rem;">⬆ Reward: {best['Max Reward']}</span>
    <span style="color:#888;font-size:0.79rem;">TYPE: {best['Type']}</span>
    <span style="color:#666;font-size:0.79rem;">DTE: {best['Ideal DTE']}</span>
  </div>
</div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# TAB 2 — DIRECTIONAL ANALYSIS
# ══════════════════════════════════════════════════════════════
with t_dir:
    st.markdown("### 🧭 Directional Signal Stack")

    # ── Probabilistic summary ─────────────────────────────────────────────
    _pu_d  = prob_score.get("prob_up", 0.5)
    _pd_d  = prob_score.get("prob_down", 0.5)
    _rs_d  = prob_score.get("raw_score", 0.0)
    _pu_dc = "#00d084" if _pu_d > 0.55 else ("#ff3b3b" if _pu_d < 0.45 else "#ffb347")
    _pd_dc = "#ff3b3b" if _pd_d > 0.55 else ("#00d084" if _pd_d < 0.45 else "#ffb347")

    pc1, pc2, pc3, pc4 = st.columns(4)
    pc1.metric("P(↑) Prob Up",   f"{_pu_d*100:.2f}%", help="logistic(raw_score × 3)")
    pc2.metric("P(↓) Prob Down", f"{_pd_d*100:.2f}%")
    pc3.metric("Raw Score",      f"{_rs_d:+.4f}", help="Weighted z-score sum ∈ [-1,+1]")
    pc4.metric("Expected Move",  f"₹{prob_score['expected_move']:.0f}  ±{prob_score['expected_move_pct']:.1f}%")

    # Probability bar
    _bar_u = int(_pu_d * 100)
    st.markdown(f"""
<div style="font-family:'IBM Plex Mono',monospace;margin:6px 0 10px;">
  <div style="display:flex;height:20px;border:1px solid #2a2a2a;overflow:hidden;">
    <div style="width:{_bar_u}%;background:{_pu_dc};display:flex;align-items:center;
                justify-content:center;font-size:0.72rem;font-weight:700;color:#000;min-width:28px;">
                ↑{_bar_u}%</div>
    <div style="width:{100-_bar_u}%;background:{_pd_dc};display:flex;align-items:center;
                justify-content:center;font-size:0.72rem;font-weight:700;color:#000;min-width:28px;">
                ↓{100-_bar_u}%</div>
  </div>
</div>""", unsafe_allow_html=True)

    # Feature scores table — leading indicators first
    _fs = prob_score.get("feature_scores", {})

    # Flow history status — always visible, not just during warmup
    _flow_pcr_len = len(st.session_state.get("_flow_pcr_hist", []))
    _flow_oi_len  = len(st.session_state.get("_flow_oi_hist",  []))
    _flow_loads   = min(_flow_pcr_len, _flow_oi_len)
    if _flow_loads < 3:
        st.info(f"ℹ️ **Flow signals warming up** ({_flow_loads}/3 loads). "
                "ΔPCR, ΔIV, ΔOI, ΔSkew, ΔGEX are *change* signals — they need at least 3 "
                "consecutive loads to build a history baseline. "
                "Click **⚡ LOAD OPTIONS INTEL** 2-3 more times to activate them.")

    if _fs:
        _fs_rows = [
            # ── LEADING ──────────────────────────────────────────────────
            {"Factor": "① FLOW — ΔPCR",        "Score": _fs.get("dPCR", 0),           "Weight": "30%↑", "Type": "LEADING"},
            {"Factor": "① FLOW — ΔSkew",        "Score": _fs.get("dSkew", 0),          "Weight": "30%↑", "Type": "LEADING"},
            {"Factor": "① FLOW — ΔIV",          "Score": _fs.get("dIV", 0),            "Weight": "30%↑", "Type": "LEADING"},
            {"Factor": "① FLOW — ΔOI",          "Score": _fs.get("dOI", 0),            "Weight": "30%↑", "Type": "LEADING"},
            {"Factor": "① FLOW — ΔGEX",         "Score": _fs.get("dGEX", 0),           "Weight": "30%↑", "Type": "LEADING"},
            {"Factor": "② POSITIONING — PCR Lvl","Score": _fs.get("pcr_level_z", 0),   "Weight": "25%↑", "Type": "LEADING"},
            {"Factor": "② POSITIONING — OI Skew","Score": _fs.get("oi_skew_z", 0),    "Weight": "25%↑", "Type": "LEADING"},
            {"Factor": "② POSITIONING — Max Pain","Score": _fs.get("mp_z", 0),         "Weight": "25%↑", "Type": "LEADING"},
            {"Factor": "③ VOL REGIME — IV/HV",  "Score": _fs.get("vol_regime_z", 0),  "Weight": "20%↑", "Type": "CONCURRENT"},
            {"Factor": "③ VOL REGIME — TS Slope","Score": _fs.get("term_slope_z", 0), "Weight": "20%↑", "Type": "CONCURRENT"},
            # ── CONFIRMING ───────────────────────────────────────────────
            {"Factor": "④ REL STRENGTH",        "Score": _fs.get("rs_score", 0),       "Weight": "15%↓", "Type": "CONFIRMING"},
            {"Factor": "⑤ TREND (EMA+ADX+RSI)", "Score": _fs.get("trend_z", 0),       "Weight": "10%↓", "Type": "CONFIRMING"},
        ]
        _fdf = pd.DataFrame(_fs_rows)
        _fdf["Signal"] = _fdf["Score"].apply(
            lambda v: "🟢 BULL" if v > 0.05 else ("🔴 BEAR" if v < -0.05 else "⚪ NEUT"))
        def _zscore_style(v):
            if isinstance(v, float):
                if v > 0.05: return "color:#00d084;font-weight:700"
                if v < -0.05: return "color:#ff3b3b;font-weight:700"
            return "color:#888"
        st.dataframe(_fdf.style.map(_zscore_style, subset=["Score"]),
                     use_container_width=True, hide_index=True)

        # Flow conviction bar
        _fm = prob_score.get("flow_magnitude", 0)
        _fc_col = "#00d084" if _fs.get("flow_score", 0) > 0 else "#ff3b3b"
        st.markdown(
            f"<div style='font-family:IBM Plex Mono,monospace;font-size:0.78rem;color:#555;"
            f"padding:4px 0;'>FLOW CONVICTION: "
            f"<span style='color:{_fc_col};font-weight:700;'>{_fm*100:.0f}%</span>"
            f" &nbsp;·&nbsp; Composite flow score: "
            f"<span style='color:{_fc_col};'>{_fs.get('flow_score',0):+.3f}</span></div>",
            unsafe_allow_html=True)

    st.divider()

    s_norm  = max(-100, min(100, bias_score))
    gauge_w = int((s_norm + 100) / 200 * 100)
    gc_     = "#00d084" if bias_score > 0 else "#ff3b3b" if bias_score < 0 else "#ffb347"

    st.markdown(f"""
<div style="background:#0d0d0d;border:1px solid #2a2a2a;padding:10px 16px;
font-family:'IBM Plex Mono',monospace;margin-bottom:10px;">
  <div style="color:#555;font-size:0.74rem;letter-spacing:.1em;margin-bottom:4px;">
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
        st.dataframe(fdf.style.map(pts_style, subset=["Points"]),
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
font-family:'IBM Plex Mono',monospace;font-size:0.9rem;color:{impl_c};margin-top:8px;">
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
    st.markdown("### 🎯 Strategy Recommendations — EV-Ranked")

    # ── Capital shortcut — editable inline if not set in sidebar ─────────────
    _capital = st.session_state.opt_capital
    _kelly_frac = CFG["kelly_fraction"]
    _kelly_cap  = CFG["kelly_cap_pct"]

    # ── Probabilistic summary bar ─────────────────────────────────────────
    _pu_s  = prob_score.get("prob_up", 0.5)
    _pd_s  = prob_score.get("prob_down", 0.5)
    _em_s  = prob_score.get("expected_move", 0.0)
    _rs_s  = prob_score.get("raw_score", 0.0)
    _pu_c  = "#00d084" if _pu_s > 0.55 else ("#ff3b3b" if _pu_s < 0.45 else "#ffb347")

    # IVR bootstrap label — tells user if IVR is real or estimated
    _ivr_sessions = len(st.session_state.opt_iv_history.get(sym, []))
    _ivr_label = f"IVR {ivr:.0f}" if _ivr_sessions >= 3 else f"IVR ~{ivr:.0f} (est. {_ivr_sessions} sess)"
    _ivr_warn  = " ⚠" if _ivr_sessions < 3 else ""

    st.markdown(f"""
<div style="background:#0a0a0a;border:1px solid #2a2a2a;padding:7px 14px;
font-family:'IBM Plex Mono',monospace;font-size:0.83rem;margin-bottom:4px;">
  <span style="color:{_pu_c};font-weight:700;">P(↑) {_pu_s*100:.1f}%</span>
  &nbsp;·&nbsp;
  <span style="color:#ff3b3b;font-weight:700;">P(↓) {_pd_s*100:.1f}%</span>
  &nbsp;·&nbsp; EM ₹{_em_s:.0f} &nbsp;·&nbsp; Score {_rs_s:+.3f}
  &nbsp;·&nbsp; {_ivr_label}{_ivr_warn}
  &nbsp;·&nbsp; DTE {dte} &nbsp;·&nbsp; Capital ₹{_capital:,.0f}
</div>""", unsafe_allow_html=True)

    # ── Step 7: Probability × IV Regime Strategy Mapping ─────────────────────
    # Show the recommended strategy class based on prob_up and IV regime
    _s7_pu    = _pu_s
    _s7_ivr   = ivr
    _s7_str   = prob_score.get("signal_strength", "No Edge")
    _s7_edge  = prob_score.get("edge_label", "No Edge")
    _s7_de    = prob_score.get("direction_edge", 0.0)
    _s7_imp   = prob_score.get("implied_move_pct", 0.0)
    _s7_model = prob_score.get("model_move_pct", 0.0)
    _s7_me    = prob_score.get("move_edge_pct", 0.0)
    _s7_ve    = prob_score.get("vol_edge", "—")

    # Determine recommended strategy class
    _high_prob = _s7_pu > 0.60
    _low_prob  = _s7_pu < 0.40
    _high_iv   = _s7_ivr >= 60
    _low_iv    = _s7_ivr < 40

    if _high_prob and _low_iv:
        _s7_rec = "Buy Calls / Call Spread"
        _s7_rec_col = "#00d084"
        _s7_rec_why = "Model bullish + IV cheap → buy direction"
    elif _high_prob and _high_iv:
        _s7_rec = "Sell Puts / Put Spread"
        _s7_rec_col = "#00d084"
        _s7_rec_why = "Model bullish + IV rich → sell downside premium"
    elif _low_prob and _high_iv:
        _s7_rec = "Sell Calls / Call Spread"
        _s7_rec_col = "#ff3b3b"
        _s7_rec_why = "Model bearish + IV rich → sell upside premium"
    elif _low_prob and _low_iv:
        _s7_rec = "Buy Puts / Put Spread"
        _s7_rec_col = "#ff3b3b"
        _s7_rec_why = "Model bearish + IV cheap → buy direction"
    elif _high_iv:
        _s7_rec = "Iron Condor / Short Strangle"
        _s7_rec_col = "#ff8c00"
        _s7_rec_why = "Neutral signal + IV rich → sell premium both sides"
    else:
        _s7_rec = "Calendar Spread"
        _s7_rec_col = "#1e90ff"
        _s7_rec_why = "Neutral signal + IV cheap → buy vol via calendar"

    st.markdown(f"""
<div style="background:#0d0d0d;border:1px solid #2a2a2a;border-left:3px solid {_s7_rec_col};
padding:8px 16px;margin-bottom:6px;font-family:'IBM Plex Mono',monospace;
display:flex;gap:24px;align-items:center;flex-wrap:wrap;font-size:0.80rem;">
  <div>
    <span style="color:#555;font-size:0.70rem;letter-spacing:.06em;">STRATEGY CLASS</span><br/>
    <span style="color:{_s7_rec_col};font-weight:700;font-size:0.92rem;">{_s7_rec}</span>
  </div>
  <div>
    <span style="color:#555;font-size:0.70rem;">Signal: </span>
    <span style="color:#e8e8e8;">{_s7_str}</span>
  </div>
  <div>
    <span style="color:#555;font-size:0.70rem;">Dir Edge: </span>
    <span style="color:{'#00d084' if _s7_de>0 else '#ff3b3b' if _s7_de<0 else '#888'};">
      {'+' if _s7_de>=0 else ''}{_s7_de*100:.1f}pp ({_s7_edge})</span>
  </div>
  <div>
    <span style="color:#555;font-size:0.70rem;">Implied Move: </span>
    <span style="color:#888;">±{_s7_imp:.1f}%</span>
  </div>
  <div>
    <span style="color:#555;font-size:0.70rem;">Model Move: </span>
    <span style="color:{'#00d084' if _s7_model > _s7_imp else '#ff8c00'};">±{_s7_model:.1f}%</span>
  </div>
  <div>
    <span style="color:#555;font-size:0.70rem;">Vol Edge: </span>
    <span style="color:{'#1e90ff' if _s7_ve=='BUY' else '#ff8c00'};font-weight:700;">{_s7_ve} options</span>
    <span style="color:#444;font-size:0.68rem;"> (move edge {'+' if _s7_me>=0 else ''}{_s7_me:.1f}%)</span>
  </div>
  <div style="margin-left:auto;color:#555;font-size:0.72rem;">{_s7_rec_why}</div>
</div>""", unsafe_allow_html=True)

    # ── Step 7: Full matrix table (expandable) ────────────────────────────────
    with st.expander("📋 Full Probability × IV Regime Matrix", expanded=False):
        _matrix_rows = [
            {"P(↑)": "> 60%",  "IV Regime": "Low (IVR < 40)",  "Strategy": "Buy Calls / Call Spread",
             "Logic": "Strong bullish signal, options cheap — buy direction"},
            {"P(↑)": "> 60%",  "IV Regime": "High (IVR ≥ 60)", "Strategy": "Sell Puts / Put Spread",
             "Logic": "Strong bullish signal, options expensive — sell downside vol"},
            {"P(↑)": "< 40%",  "IV Regime": "High (IVR ≥ 60)", "Strategy": "Sell Calls / Bear Call Spread",
             "Logic": "Strong bearish signal, options expensive — sell upside vol"},
            {"P(↑)": "< 40%",  "IV Regime": "Low (IVR < 40)",  "Strategy": "Buy Puts / Put Spread",
             "Logic": "Strong bearish signal, options cheap — buy direction"},
            {"P(↑)": "45–55%", "IV Regime": "High (IVR ≥ 60)", "Strategy": "Iron Condor",
             "Logic": "Neutral, options rich — sell vol both sides"},
            {"P(↑)": "45–55%", "IV Regime": "Low (IVR < 40)",  "Strategy": "Calendar Spread",
             "Logic": "Neutral, vol cheap — buy term structure"},
        ]
        _mat_df = pd.DataFrame(_matrix_rows)

        def _mat_style(v):
            if "Buy" in str(v) and "Sell" not in str(v): return "color:#00d084"
            if "Sell" in str(v) and "Buy" not in str(v): return "color:#ff8c00"
            if "Condor" in str(v) or "Calendar" in str(v): return "color:#1e90ff"
            return "color:#e8e8e8"

        # Highlight current row
        def _highlight_current(row):
            if row["Strategy"] == _s7_rec:
                return ["background-color:#1a1400;font-weight:700"] * len(row)
            return [""] * len(row)

        st.dataframe(
            _mat_df.style
                .map(_mat_style, subset=["Strategy"])
                .apply(_highlight_current, axis=1),
            use_container_width=True, hide_index=True
        )

    # ── Flow Alert Banner — fires only when flow magnitude exceeds adaptive threshold ──
    _flow_mag_now   = prob_score.get("flow_magnitude", 0.0)
    _flow_threshold = st.session_state.get("opt_flow_conv_threshold", CFG["flow_conviction_seed"])
    _fs_now         = prob_score.get("feature_scores", {})
    _flow_dir       = prob_score.get("feature_scores", {}).get("flow_score", 0)

    # ── Expiry day warning — all new strategies are poor on DTE=1 ────────────
    if dte <= 1:
        st.warning("⚠️ **DTE = 1 (Expiry Day):** All option strategies have poor EV on expiry day "
                   "because time-value has collapsed. MC simulations show negative EV for most structures. "
                   "**Recommended use:** Close existing positions, not open new ones. "
                   "If you must trade, small directional plays (Long Call/Put) have limited loss = premium only.")

    if _flow_mag_now >= _flow_threshold and _flow_threshold > 0:
        _fb_col  = "#00d084" if _flow_dir > 0 else "#ff3b3b"
        _fb_dir  = "BULLISH" if _flow_dir > 0 else "BEARISH"
        _dpcr_v  = _fs_now.get("dPCR",  0); _dpcr_s = f"{'+' if _dpcr_v>=0 else ''}{_dpcr_v:.3f}"
        _dsk_v   = _fs_now.get("dSkew", 0); _dsk_s  = f"{'+' if _dsk_v>=0 else ''}{_dsk_v:.3f}"
        _div_v   = _fs_now.get("dIV",   0); _div_s  = f"{'+' if _div_v>=0 else ''}{_div_v:.3f}"
        _conv_pct = round(_flow_mag_now / (_flow_threshold + 1e-9) * 100 - 100, 0)
        st.markdown(f"""
<div style="background:#0a0e0a;border:1px solid {_fb_col};border-left:5px solid {_fb_col};
padding:8px 14px;margin-bottom:8px;font-family:'IBM Plex Mono',monospace;">
  <span style="color:{_fb_col};font-weight:700;font-size:0.86rem;">
    ⚡ HIGH-CONVICTION FLOW SIGNAL — {_fb_dir}
  </span>
  &nbsp;&nbsp;
  <span style="color:#888;font-size:0.80rem;">
    ΔPCR {_dpcr_s} &nbsp;·&nbsp; ΔSkew {_dsk_s} &nbsp;·&nbsp; ΔIV {_div_s}
    &nbsp;·&nbsp; Conviction {_flow_mag_now:.3f} ({_conv_pct:+.0f}% above threshold)
  </span>
</div>""", unsafe_allow_html=True)

    if strat_recs:
        # ── Feature score breakdown ──────────────────────────────────────
        with st.expander("◼ FEATURE SCORES (inputs to P(↑)/P(↓))"):
            fs = prob_score.get("feature_scores", {})
            _fc = st.columns(4)
            def _fscore_bar(label, val, col_idx, weight_label=""):
                color = "#00d084" if val > 0.05 else ("#ff3b3b" if val < -0.05 else "#888")
                dir_  = "▶" if val >= 0 else "◀"
                wt_c  = "#ff8c00" if "LEADING" in weight_label else "#555"
                _fc[col_idx % 4].markdown(
                    f"<div style='font-family:IBM Plex Mono,monospace;font-size:0.76rem;"
                    f"padding:3px 0;border-bottom:1px solid #1a1a1a;'>"
                    f"<span style='color:{wt_c};font-size:0.68rem;'>{weight_label}</span><br/>"
                    f"<span style='color:#555;'>{label}</span><br/>"
                    f"<span style='color:{color};font-weight:700;'>{dir_} {val:+.3f}</span></div>",
                    unsafe_allow_html=True)
            _scores = [
                ("ΔPCR flow",      fs.get("dPCR", 0),          "LEADING①"),
                ("ΔSkew flow",     fs.get("dSkew", 0),         "LEADING①"),
                ("ΔIV flow",       fs.get("dIV", 0),           "LEADING①"),
                ("ΔOI flow",       fs.get("dOI", 0),           "LEADING①"),
                ("PCR level",      fs.get("pcr_level_z", 0),   "LEADING②"),
                ("OI skew",        fs.get("oi_skew_z", 0),     "LEADING②"),
                ("Max pain",       fs.get("mp_z", 0),          "LEADING②"),
                ("Vol regime",     fs.get("vol_regime_z", 0),  "CONCURRENT③"),
                ("Term structure", fs.get("term_slope_z", 0),  "CONCURRENT③"),
                ("Rel Strength",   fs.get("rs_score", 0),      "confirming④"),
                ("Trend (EMA+ADX)",fs.get("trend_z", 0),       "confirming⑤"),
            ]
            for _ci, (_lbl, _val, _wt) in enumerate(_scores):
                _fscore_bar(_lbl, _val, _ci, _wt)

        # ── Strategy cards with Kelly in ₹ and dynamic rationale ────────
        for i, s in enumerate(strat_recs[:8]):
            rank_c     = "#ff8c00" if i == 0 else "#444"
            top_border = "border-top:3px solid #ff8c00;" if i == 0 else ""
            ev_raw     = s.get("ev", 0)
            ev_col     = "#00d084" if ev_raw >= 0 else "#ff3b3b"
            pop_val    = s.get("pop", 0.5)
            pop_col    = "#00d084" if pop_val >= 0.6 else ("#ffb347" if pop_val >= 0.45 else "#ff3b3b")
            sc         = s.get("Score", 50)
            sc_col     = "#00d084" if sc >= 70 else ("#ffb347" if sc >= 50 else "#666")
            sr         = s.get("safety_ratio", 2.0)
            sr_col     = "#00d084" if sr >= CFG["safety_ratio_safe"] else (
                         "#ffb347" if sr >= CFG["safety_ratio_moderate"] else "#ff3b3b")

            # Kelly position size in ₹ from live capital
            _kelly_raw_s = s.get("kelly_raw", 0)
            _kelly_cap_s = max(0.0, min(_kelly_cap, _kelly_raw_s))
            _kelly_f_s   = _kelly_cap_s * _kelly_frac
            _pos_size_rs = round(_kelly_f_s * _capital)
            _kelly_pct_s = _kelly_f_s * 100
            _k_col = "#00d084" if _kelly_f_s > 0 else "#555"

            # Theta daily decay (for credit strategies, compute days to 50% profit)
            _max_reward_raw = s.get("max_reward", 0)
            _theta_day_est  = 0.0
            _days_to_target = None
            if _max_reward_raw > 0 and _max_reward_raw < 1e5:
                # Approximate theta using BS at current params for ATM
                _theta_day_est = abs(bs_greeks(spot, atm_k, T, r, atm_iv, "call", q).get("theta", 0)) * 2
                if _theta_day_est > 0 and "credit" in s.get("type", "").lower():
                    _days_to_target = round(_max_reward_raw * 0.5 / (_theta_day_est + 1e-9))

            # Dynamic rationale — built from actual feature scores, not generic text
            _dom_factors = []
            if abs(_flow_dir) > 0.15:
                _dom_factors.append(f"flow {'bullish' if _flow_dir>0 else 'bearish'} ({_flow_dir:+.2f})")
            _pcr_lz = fs.get("pcr_level_z", 0)
            if abs(_pcr_lz) > 0.15:
                _dom_factors.append(f"PCR {'support' if _pcr_lz>0 else 'resistance'} ({_pcr_lz:+.2f})")
            _vr = fs.get("vol_regime_z", 0)
            if abs(_vr) > 0.10:
                _dom_factors.append(f"vol {'cheap' if _vr<0 else 'rich'} ({ivr:.0f} IVR)")
            _dte_a = s.get("dte_align", 1.0)
            _dir_a = s.get("dir_align", 0.5)
            if not _dom_factors:
                _dom_factors.append("balanced signals")
            _dynamic_rationale = (
                f"Ranked #{i+1} by EV · Drivers: {', '.join(_dom_factors)} · "
                f"Dir-fit {_dir_a*100:.0f}% · DTE-align {_dte_a:.2f}"
                + (f" · ~{_days_to_target}d to 50% profit" if _days_to_target else "")
            )

            # ── Conflict / warning badges ────────────────────────────────────
            _warnings = []
            _stype = s.get("type", "")
            # DTE mismatch warning
            if _dte_a < 0.15:
                _dte_lo_s = s.get("Ideal DTE", "?")
                _warnings.append(f"⚠️ DTE mismatch — ideal {s.get('ideal_dte_lo',0)}-{s.get('ideal_dte_hi',0)}d, current {dte}d")
            # Vol regime conflict: debit neutral in high-IV environment (better to sell)
            if "debit" in _stype and "neutral" in _stype and ivr > 70:
                _warnings.append("⚠️ IV Rich (IVR>70) — debit neutral strategies overpay for vol; consider credit spreads")
            # Credit neutral when strong directional signal
            if "credit" in _stype and "neutral" in _stype and abs(prob_score.get("raw_score", 0)) > 0.15:
                _warnings.append("⚠️ Strong directional signal — credit neutral strategies risk large directional loss")
            # Directional strategy against flow signal
            if "bull" in _stype and _flow_dir < -0.3:
                _warnings.append("⚠️ Bearish flow signal conflicts with bullish strategy")
            if "bear" in _stype and _flow_dir > 0.3:
                _warnings.append("⚠️ Bullish flow signal conflicts with bearish strategy")

            st.markdown(f"""
<div style="background:#0d0d0d;border:1px solid #2a2a2a;{top_border}
padding:12px 16px;margin-bottom:7px;font-family:'IBM Plex Mono',monospace;">
  <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;">
    <div>
      <span style="color:{rank_c};font-size:0.74rem;font-weight:700;">{'⭐ BEST EV' if i==0 else f'#{i+1}'}</span>
      <span style="color:#e8e8e8;font-size:1.0rem;font-weight:700;margin-left:8px;">{s['Strategy']}</span>
      <span style="background:#1a1a1a;color:#777;font-size:0.72rem;padding:2px 7px;margin-left:8px;
                   display:inline-block;">{s['Type']}</span>
    </div>
    <span style="color:{sc_col};font-size:0.92rem;font-weight:700;">EV SCORE {sc}</span>
  </div>
  <div style="color:#7ec8e3;font-size:0.9rem;margin:6px 0 4px;">LEGS: <b>{s['Legs']}</b></div>
  <div style="display:flex;gap:12px;margin-top:8px;flex-wrap:wrap;align-items:center;">
    <span style="color:{ev_col};font-size:0.82rem;font-weight:700;">EV {'+' if ev_raw>=0 else ''}₹{ev_raw:,.0f}</span>
    <span style="color:{pop_col};font-size:0.82rem;">POP {pop_val*100:.1f}%</span>
    <span style="color:#ffb347;font-size:0.77rem;">⬇ Risk: {s['Max Risk']}</span>
    <span style="color:#00d084;font-size:0.77rem;">⬆ Reward: {s['Max Reward']}</span>
    <span style="color:{sr_col};font-size:0.77rem;">🛡 {sr:.2f}× EM</span>
    <span style="color:{_k_col};font-size:0.82rem;font-weight:700;">
      {'₹' + f'{_pos_size_rs:,.0f}' if _pos_size_rs > 0 else '—'} ({_kelly_pct_s:.1f}% Kelly)
    </span>
    <span style="color:#555;font-size:0.77rem;">DTE {s['Ideal DTE']}</span>
  </div>
  <div style="color:#555;font-size:0.77rem;margin-top:5px;line-height:1.5;">{_dynamic_rationale}</div>
  {chr(10).join(f'<div style="color:#ffb347;font-size:0.74rem;margin-top:3px;">{w}</div>' for w in _warnings) if _warnings else ''}
</div>""", unsafe_allow_html=True)

            # ── LOG button: snapshot this strategy into the Tradebook ────────
            _log_col, _log_spacer = st.columns([1, 5])
            with _log_col:
                if st.button("📋 LOG Trade", key=f"log_strat_{i}_{s.get('Strategy','x')}",
                             help="Snapshot this signal + strategy into the Tradebook tab. Live PnL will update on each reload."):
                    _trade_entry = {
                        "id":               f"{int(time.time()*1000)}_{i}",
                        "logged_at":        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "symbol":           sym,
                        "expiry":           expiry,
                        "dte_at_entry":     dte,
                        "strategy":         s.get("Strategy", "—"),
                        "strategy_type":    s.get("Type", "—"),
                        "legs":             s.get("Legs", "—"),
                        "rank":             i + 1,
                        "ev_score":         int(sc),
                        "ev_rs":            float(ev_raw),
                        "pop":              float(pop_val),
                        "max_risk":         s.get("Max Risk", "—"),
                        "max_reward":       s.get("Max Reward", "—"),
                        "ideal_dte":        s.get("Ideal DTE", "—"),
                        "safety_ratio":     float(sr),
                        "kelly_pct":        float(_kelly_pct_s),
                        "pos_size_rs":      float(_pos_size_rs),
                        # Market snapshot at log time
                        "spot_entry":       float(spot),
                        "spot_current":     float(spot),
                        "atm_k":            float(atm_k),
                        "atm_iv_entry":     float(atm_iv),
                        "atm_iv_current":   float(atm_iv),
                        "hv20_entry":       float(hv20),
                        "ivr_entry":        float(ivr),
                        "bias":             bias,
                        "bias_score":       int(bias_score),
                        "prob_up":          float(prob_score.get("prob_up", 0.5)),
                        "prob_down":        float(prob_score.get("prob_down", 0.5)),
                        "raw_score":        float(prob_score.get("raw_score", 0.0)),
                        "vol_edge":         prob_score.get("vol_edge", "—"),
                        "strategy_class":   _s7_rec,
                        "expected_move":    float(prob_score.get("expected_move", 0.0)),
                        "exp_move_pct":     float(prob_score.get("expected_move_pct", 0.0)),
                        "max_pain":         float(oi_d.get("max_pain", spot) if oi_d else spot),
                        "call_wall":        float(oi_d.get("call_wall", 0) if oi_d else 0),
                        "put_wall":         float(oi_d.get("put_wall", 0) if oi_d else 0),
                        "pcr_oi":           float(oi_d.get("pcr_oi", 1.0) if oi_d else 1.0),
                        "flow_score":       float(prob_score.get("feature_scores", {}).get("flow_score", 0.0)),
                        "flow_magnitude":   float(prob_score.get("flow_magnitude", 0.0)),
                        "regime":           regime_d.get("label", regime_d.get("trend", "—")),
                        "dynamic_rationale":_dynamic_rationale,
                        "status":           "ACTIVE",
                    }
                    _tb = st.session_state.opt_tradebook
                    _tb.append(_trade_entry)
                    st.session_state.opt_tradebook = _tb
                    _save_tradebook(_tb)
                    st.success(f"✅ Logged! '{s.get('Strategy')}' → Tradebook ({len(_tb)} active). Switch to 📒 Tradebook tab.")
                    st.rerun()

    else:
        st.info("Load options intel to generate EV-ranked strategy recommendations.")


    with st.expander("◼ HOW EV RANKING WORKS"):
        st.markdown(f"""
**Selection criterion: highest EV adjusted for risk — no if/then rules.**

Every strategy in the universe (14 structures) is evaluated simultaneously:

| Component | Method |
|-----------|---------|
| POP & EV | Monte Carlo {CFG['pop_simulations']:,} paths · vol surface skew-aware · real-world drift |
| Directional alignment | `prob_up` / `prob_down` from logistic(raw_score) |
| DTE alignment | Continuous exponential decay (no binary in/out window) |
| Safety ratio | `distance_to_short / expected_move` via smooth sigmoid |
| EV score | `tanh(EV/MaxRisk) × POP × dte_align × safety × ts_factor` |
| Composite | 60% EV score + 40% directional alignment |

**Current P(↑) = {_pu_s*100:.1f}%** → raw_score = {_rs_s:+.4f} → logistic({_rs_s*3:+.3f})

Strategy selection requires **no thresholds**, no `if bullish → bull spread`.
The Iron Condor can rank #1 even when P(↑)=60% if its EV after skew-aware MC
is higher than a directional trade. The model picks the highest-EV outcome.
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
            ce_p = bs_price(spot, k, T, r, atm_iv, "call", q)
            pe_p = bs_price(spot, k, T, r, atm_iv, "put",  q)
            cg   = bs_greeks(spot, k, T, r, atm_iv, "call", q)
            pg   = bs_greeks(spot, k, T, r, atm_iv, "put",  q)
            mm   = "ATM" if abs(k-spot) <= 0.5*step else ("ITM" if k<spot else "OTM")
            syn_rows.append({
                "Strike":k,"Moneyness":mm,
                "CE Price":round(ce_p,2),"CE IV%":round(atm_iv*100,1),
                "CE Δ":cg["delta"],"CE θ/d":round(cg["theta"],3),"CE ν/1%":round(cg["vega"],3),
                "PE Price":round(pe_p,2),"PE IV%":round(atm_iv*100,1),
                "PE Δ":pg["delta"],"PE θ/d":round(pg["theta"],3),"PE ν/1%":round(pg["vega"],3),
            })
        st.dataframe(pd.DataFrame(syn_rows), use_container_width=True, hide_index=True)
    else:
        _chain_lo = atm_k - CFG["chain_strikes"] * step
        _chain_hi = atm_k + CFG["chain_strikes"] * step
        disp_c = chain_df[(chain_df.Strike >= _chain_lo) & (chain_df.Strike <= _chain_hi)].copy()

        # Warn when chain data is present but LTP/OI/IV are all zero
        _live = st.session_state.get("_chain_has_live", True)
        if not _live:
            st.warning("⚠️ Chain data is all zeros (market closed or token expired). "
                       "IV signals below are computed from HV only and are NOT reliable.")

        # Add directional + IV edge signal per row
        # IV edge: percentile-based. Collect all chain IV/HV ratios, classify by percentile.
        hv_ref = hv20 if hv20 and hv20 > 0.01 else atm_iv
        _all_iv_ratios = []
        for _, _cr in disp_c.iterrows():
            _ce_iv = _sanitise_iv(float(_cr.CE_IV), None)
            _pe_iv = _sanitise_iv(float(_cr.PE_IV), None)
            if _ce_iv: _all_iv_ratios.append(_ce_iv / hv_ref)
            if _pe_iv: _all_iv_ratios.append(_pe_iv / hv_ref)
        # Sell threshold = CFG iv_hv_pct_sell-th percentile of this chain's IV/HV ratios
        # FIX: also guard against all-zero IV case (chain has data but IVs are 0)
        _has_real_iv = len(_all_iv_ratios) >= 4 and max(_all_iv_ratios) > 0.01
        if _has_real_iv:
            _rich_thresh  = float(np.percentile(_all_iv_ratios, CFG["iv_hv_pct_sell"]))
            _cheap_thresh = float(np.percentile(_all_iv_ratios, CFG["iv_hv_pct_buy"]))
        else:
            # Absolute fallback when chain is tiny or IVs are all zero
            _rich_thresh  = CFG["iv_rich_ratio"]
            _cheap_thresh = CFG["iv_cheap_ratio"]

        def row_signal(row):
            ce_iv = _sanitise_iv(float(row.CE_IV), None)
            pe_iv = _sanitise_iv(float(row.PE_IV), None)
            ce_dir = "BUY" if bias_score >= 12 else "SELL" if bias_score <= -12 else "—"
            pe_dir = "BUY" if bias_score <= -12 else "SELL" if bias_score >= 12 else "—"
            # Suppress IV vol signal when: IV missing/zero, no real IV, or DTE=1 (structurally distorted)
            if ce_iv is None or not _has_real_iv or dte <= 1:
                ce_vol = "—"
            else:
                ce_ratio = ce_iv / (hv_ref + 1e-9)
                ce_vol = (f"SELL (rich ×{ce_ratio:.2f})"  if ce_ratio >= _rich_thresh else
                          f"BUY (cheap ×{ce_ratio:.2f})" if ce_ratio <= _cheap_thresh else "—")
            if pe_iv is None or not _has_real_iv or dte <= 1:
                pe_vol = "—"
            else:
                pe_ratio = pe_iv / (hv_ref + 1e-9)
                pe_vol = (f"SELL (rich ×{pe_ratio:.2f})"  if pe_ratio >= _rich_thresh else
                          f"BUY (cheap ×{pe_ratio:.2f})" if pe_ratio <= _cheap_thresh else "—")
            return pd.Series({"CE_Dir": ce_dir, "CE_Vol_Sig": ce_vol,
                               "PE_Dir": pe_dir, "PE_Vol_Sig": pe_vol})

        sigs = disp_c.apply(row_signal, axis=1)
        disp_c = pd.concat([disp_c, sigs], axis=1)

        # DTE context note — on expiry day (DTE=1) all IVs are elevated vs HV which
        # is a 20-day measure; "SELL rich" across the board is expected and not a signal.
        if dte <= 1:
            st.info("ℹ️ **DTE = 1 (expiry day):** IV/HV ratios are elevated across all strikes "
                    "because annualised IV reflects same-day optionality, not 20-day realised vol. "
                    "IV Sig 'SELL rich' on expiry day is structural, not a tradeable edge.")

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

        # Format numeric columns cleanly — no trailing .000000
        # Use pandas nullable integer type to force integer display in Streamlit dataframe
        for _icol in ["Strike", "CE OI", "PE OI", "CE Vol", "PE Vol", "CE ΔOI", "PE ΔOI"]:
            chain_show[_icol] = pd.to_numeric(chain_show[_icol], errors="coerce").fillna(0).astype("Int64")
        # Keep floats as floats (rounded) — Streamlit renders them without trailing zeros
        for _fcol, _dp in [("CE LTP",2),("PE LTP",2),("CE IV",2),("PE IV",2),("PCR",3)]:
            if _fcol in chain_show.columns:
                chain_show[_fcol] = pd.to_numeric(chain_show[_fcol], errors="coerce").round(_dp)

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
            .map(sig_style,  subset=["CE Signal","CE IV Sig","PE Signal","PE IV Sig"]) \
            .map(mm_style,   subset=["Money"])
        # Use column_config for number formatting on the plain dataframe level
        # (column_config works alongside Styler in Streamlit ≥1.22)
        st.dataframe(styled, use_container_width=True, hide_index=True,
                     column_config={
                         "CE LTP": st.column_config.NumberColumn(format="%.2f"),
                         "PE LTP": st.column_config.NumberColumn(format="%.2f"),
                         "CE IV":  st.column_config.NumberColumn(format="%.2f"),
                         "PE IV":  st.column_config.NumberColumn(format="%.2f"),
                         "PCR":    st.column_config.NumberColumn(format="%.3f"),
                     })
        st.caption("CE/PE Signal = directional signal from bias. IV Sig = IV vs HV edge (overpriced/underpriced).")

# ══════════════════════════════════════════════════════════════
# TAB 5 — GREEKS DASHBOARD
# ══════════════════════════════════════════════════════════════
with t_greeks:
    st.markdown("### 🔢 Greeks Dashboard")
    if q > 0:
        st.caption(f"Dividend yield q = {q*100:.2f}% applied (Merton form). Theta uses {CFG['theta_days']} trading-day convention.")
    else:
        st.caption(f"Theta uses {CFG['theta_days']} trading-day convention (not calendar days).")

    g_rows = []
    for k in strikes_around(spot, step, 5):
        ce_iv_use = pe_iv_use = atm_iv
        if not chain_df.empty:
            closest = chain_df.iloc[(chain_df.Strike - k).abs().argsort()[:1]]
            if not closest.empty:
                ce_iv_use = _sanitise_iv(float(closest.CE_IV.values[0]), atm_iv)
                pe_iv_use = _sanitise_iv(float(closest.PE_IV.values[0]), atm_iv)

        cg    = bs_greeks(spot, k, T, r, ce_iv_use, "call", q)
        pg    = bs_greeks(spot, k, T, r, pe_iv_use, "put",  q)
        cp    = bs_price (spot, k, T, r, ce_iv_use, "call", q)
        pp    = bs_price (spot, k, T, r, pe_iv_use, "put",  q)
        mm    = "ATM" if abs(k-spot) <= 0.5*step else ("ITM" if k<spot else "OTM")
        ce_itm   = bs_itm_prob(spot, k, T, r, ce_iv_use, "call", q)
        pe_itm   = bs_itm_prob(spot, k, T, r, pe_iv_use, "put",  q)
        # New: Prob of Touch (barrier probability)
        ce_touch = bs_prob_touch(spot, k, T, r, ce_iv_use, "call", q)
        pe_touch = bs_prob_touch(spot, k, T, r, pe_iv_use, "put",  q)
        # New: Charm (delta decay per day)
        ce_charm = bs_charm(spot, k, T, r, ce_iv_use, "call", q)
        pe_charm = bs_charm(spot, k, T, r, pe_iv_use, "put",  q)
        # New: Vanna (dDelta/dIV)
        ce_vanna = bs_vanna(spot, k, T, r, ce_iv_use, "call", q)
        pe_vanna = bs_vanna(spot, k, T, r, pe_iv_use, "put",  q)
        g_rows.append({
            "Strike":k, "Moneyness":mm,
            "CE Price":round(cp,2), "CE IV%":round(ce_iv_use*100,1),
            "CE Δ":cg["delta"], "CE Γ":cg["gamma"],
            "CE θ/d":round(cg["theta"],3), "CE ν/1%":round(cg["vega"],3),
            "CE Charm":ce_charm, "CE Vanna":ce_vanna,
            "CE P(ITM)":f"{ce_itm*100:.0f}%",
            "CE P(Touch)":f"{ce_touch*100:.0f}%",
            "PE Price":round(pp,2), "PE IV%":round(pe_iv_use*100,1),
            "PE Δ":pg["delta"], "PE Γ":pg["gamma"],
            "PE θ/d":round(pg["theta"],3), "PE ν/1%":round(pg["vega"],3),
            "PE Charm":pe_charm, "PE Vanna":pe_vanna,
            "PE P(ITM)":f"{pe_itm*100:.0f}%",
            "PE P(Touch)":f"{pe_touch*100:.0f}%",
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

    # ── Dealer Gamma by Strike (GEX chart with proper dealer sign convention) ──
    if oi_d and not oi_d.get("gex_df", pd.DataFrame()).empty:
        st.divider()
        st.markdown("##### Dealer Gamma Exposure (GEX) by Strike")
        st.caption("GEX > 0 (green) = dealers long gamma → they buy dips & sell rallies (stabilising). "
                   "GEX < 0 (red) = dealers short gamma → they chase price (destabilising).")
        _gex_df_chart = oi_d["gex_df"].copy()
        _gex_df_chart = _gex_df_chart[
            (_gex_df_chart.Strike >= atm_k - 8 * step) &
            (_gex_df_chart.Strike <= atm_k + 8 * step)
        ]
        fig_gex = go.Figure()
        _bar_cols = ["#00d084" if v >= 0 else "#ff3b3b" for v in _gex_df_chart.NET_GEX]
        # Auto-scale: choose unit that keeps values readable (not 0.00)
        _gex_max_abs = max(abs(_gex_df_chart.NET_GEX.max()), abs(_gex_df_chart.NET_GEX.min()), 1)
        if _gex_max_abs >= 1e9:
            _gex_scale, _gex_unit = 1e7, "₹Cr"
        elif _gex_max_abs >= 1e6:
            _gex_scale, _gex_unit = 1e5, "₹Lakh"
        else:
            _gex_scale, _gex_unit = 1e3, "₹K"
        fig_gex.add_trace(go.Bar(
            x=_gex_df_chart.Strike,
            y=_gex_df_chart.NET_GEX / _gex_scale,
            marker_color=_bar_cols,
            name="Net GEX",
        ))
        fig_gex.add_vline(x=spot, line=dict(color="#ffb347", dash="dot", width=1.5),
                          annotation_text=f"Spot {spot:.0f}")
        _gflip = oi_d.get("gamma_flip", spot)
        fig_gex.add_vline(x=_gflip, line=dict(color="#9c27b0", dash="dash", width=1.5),
                          annotation_text=f"Γ Flip {_gflip:.0f}")
        fig_gex.add_hline(y=0, line=dict(color="#444", width=1))
        fig_gex.update_layout(
            title=f"Dealer Gamma Exposure (GEX) by Strike — {_gex_unit} Notional",
            height=280, plot_bgcolor="#000", paper_bgcolor="#000",
            font=dict(color="#e8e8e8", family="IBM Plex Mono", size=9),
            xaxis=dict(title="Strike", gridcolor="#111"),
            yaxis=dict(title=f"GEX ({_gex_unit})", gridcolor="#111", zeroline=True, zerolinecolor="#2a2a2a"),
            margin=dict(t=40, b=10),
        )
        st.plotly_chart(fig_gex, use_container_width=True)

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
font-family:'IBM Plex Mono',monospace;font-size:0.9rem;color:{pcr_c};">
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
padding:9px 14px;font-family:'IBM Plex Mono',monospace;font-size:0.83rem;">
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
font-family:'IBM Plex Mono',monospace;font-size:0.84rem;color:{skew_c};margin:4px 0;">
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

    LOT_SIZE = CFG["lot_sizes"].get(sym.upper(), CFG["lot_size_fallback"])

    # ── EV context line ──────────────────────────────────────────────────
    if strat_recs:
        _top = strat_recs[0]
        _top_ev  = _top.get("ev", 0)
        _top_pop = _top.get("pop", 0.5)
        _top_evc = "#00d084" if _top_ev >= 0 else "#ff3b3b"
        st.markdown(
            f"<span style='font-family:IBM Plex Mono,monospace;font-size:0.80rem;color:#555;'>"
            f"Top EV strategy: <span style='color:#ff8c00;font-weight:700;'>{_top['Strategy']}</span>"
            f" · EV <span style='color:{_top_evc};'>{'+' if _top_ev>=0 else ''}₹{_top_ev:,.0f}</span>"
            f" · POP {_top_pop*100:.1f}%"
            f" · Score {_top['Score']}/100</span>",
            unsafe_allow_html=True)

    # ── Quick load buttons — use _bs_c/_bs_p which close over T, r, q, atm_iv ──
    ql0,ql1,ql2,ql3,ql4,ql5,ql6 = st.columns(7)
    sv = float(step)

    # Top EV pick from the probabilistic engine
    if ql0.button("⭐ Top EV Pick", key="ql_top_ev"):
        if strat_recs:
            _top_s = strat_recs[0]
            _top_legs = _top_s.get("legs", [])  # MC legs from universe
            if _top_legs:
                _new_legs = []
                for _lg in _top_legs:
                    _new_legs.append({
                        "Opt":     _lg["opt"],
                        "Strike":  float(_lg["strike"]),
                        "Premium": float(_lg["premium"]),
                        "Qty":     int(_lg.get("qty", 1)),
                        "Action":  _lg["action"].title(),
                    })
                st.session_state.payoff_legs = _new_legs
                st.rerun()

    if ql1.button("Bull Call Spread",  key="ql_bcs"):
        st.session_state.payoff_legs = [
            {"Opt":"CE","Strike":atm_k,     "Premium":_bs_c(atm_k),     "Qty":1,"Action":"Buy"},
            {"Opt":"CE","Strike":atm_k+sv,  "Premium":_bs_c(atm_k+sv),  "Qty":1,"Action":"Sell"},
        ]; st.rerun()
    if ql2.button("Bear Put Spread",   key="ql_bps"):
        st.session_state.payoff_legs = [
            {"Opt":"PE","Strike":atm_k,     "Premium":_bs_p(atm_k),     "Qty":1,"Action":"Buy"},
            {"Opt":"PE","Strike":atm_k-sv,  "Premium":_bs_p(atm_k-sv),  "Qty":1,"Action":"Sell"},
        ]; st.rerun()
    if ql3.button("Long Straddle",     key="ql_str"):
        st.session_state.payoff_legs = [
            {"Opt":"CE","Strike":atm_k,"Premium":_bs_c(atm_k),"Qty":1,"Action":"Buy"},
            {"Opt":"PE","Strike":atm_k,"Premium":_bs_p(atm_k),"Qty":1,"Action":"Buy"},
        ]; st.rerun()
    if ql4.button("Iron Condor",       key="ql_ic"):
        st.session_state.payoff_legs = [
            {"Opt":"PE","Strike":atm_k-sv,   "Premium":_bs_p(atm_k-sv),   "Qty":1,"Action":"Sell"},
            {"Opt":"PE","Strike":atm_k-2*sv, "Premium":_bs_p(atm_k-2*sv), "Qty":1,"Action":"Buy"},
            {"Opt":"CE","Strike":atm_k+sv,   "Premium":_bs_c(atm_k+sv),   "Qty":1,"Action":"Sell"},
            {"Opt":"CE","Strike":atm_k+2*sv, "Premium":_bs_c(atm_k+2*sv), "Qty":1,"Action":"Buy"},
        ]; st.rerun()
    if ql5.button("Butterfly",         key="ql_bf"):
        st.session_state.payoff_legs = [
            {"Opt":"CE","Strike":atm_k-sv,  "Premium":_bs_c(atm_k-sv),  "Qty":1,"Action":"Buy"},
            {"Opt":"CE","Strike":atm_k,     "Premium":_bs_c(atm_k),     "Qty":2,"Action":"Sell"},
            {"Opt":"CE","Strike":atm_k+sv,  "Premium":_bs_c(atm_k+sv),  "Qty":1,"Action":"Buy"},
        ]; st.rerun()
    if ql6.button("Short Strangle",    key="ql_ss"):
        st.session_state.payoff_legs = [
            {"Opt":"PE","Strike":atm_k-sv,  "Premium":_bs_p(atm_k-sv),  "Qty":1,"Action":"Sell"},
            {"Opt":"CE","Strike":atm_k+sv,  "Premium":_bs_c(atm_k+sv),  "Qty":1,"Action":"Sell"},
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
            prem = l_prem if l_prem > 0 else (_bs_c(l_strike) if l_opt=="CE" else _bs_p(l_strike))
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

        # Chart range = ±3σ using trading-day T (consistent with all other BS calcs)
        _exp_move_frac = oi_d.get("exp_move_pct", 0) / 100.0 if oi_d else 0
        if _exp_move_frac <= 0:
            # Fallback from BS: E[|move|] = σ × sqrt(T) × sqrt(2/π)
            _bs_const = math.sqrt(2.0 / math.pi)
            _exp_move_frac = atm_iv * _bs_const * math.sqrt(T) if T > 0 else atm_iv * 0.05
        # ±3σ covers 99.7% of probability mass; floor at ±3% for very short DTE
        _range_frac  = max(3 * _exp_move_frac, 0.03)
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

        # Check if any leg is uncapped short (naked short CE or PE)
        _has_naked_short = any(
            l["Action"] == "Sell" and not any(
                l2["Action"] == "Buy" and l2["Opt"] == l["Opt"] and
                (float(l2["Strike"]) > float(l["Strike"]) if l["Opt"]=="CE"
                 else float(l2["Strike"]) < float(l["Strike"]))
                for l2 in st.session_state.payoff_legs
            )
            for l in st.session_state.payoff_legs
        )
        _loss_label = f"₹{max_loss:,.0f} (within ±3σ)" if _has_naked_short else f"₹{max_loss:,.0f}"

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
        s2.metric("Max Loss",    _loss_label)
        cost_lbl = "Debit" if total_cost > 0 else "Credit"
        s3.metric("Net Cost",    f"₹{abs(total_cost):,.0f}", delta=cost_lbl)
        s4.metric("Breakevens",  ", ".join([f"₹{b:.0f}" for b in bes]) or "None")
        rr = abs(max_profit/max_loss) if max_loss != 0 else float("inf")
        s5.metric("Reward:Risk", f"{rr:.2f}×" if rr != float("inf") else "∞")
        s6.metric("Legs",        str(len(st.session_state.payoff_legs)))

        # ── Probability of Profit + Expected Value (Monte Carlo) ──
        _mc_legs = []
        for _lmc in st.session_state.payoff_legs:
            _mc_legs.append({
                "opt":     _lmc["Opt"],
                "strike":  float(_lmc["Strike"]),
                "premium": float(_lmc["Premium"]),
                "action":  _lmc["Action"],
                "qty":     int(_lmc["Qty"]) * int(lot_inp),
            })
        _pop, _ev = strategy_prob_profit(_mc_legs, spot, T, r, atm_iv, q,
                                         chain_df=chain_df,
                                         ohlcv_df=st.session_state.get("opt_ohlcv", pd.DataFrame()))
        _pop_col  = "#00d084" if _pop >= 0.6 else ("#ffb347" if _pop >= 0.45 else "#ff3b3b")
        _ev_col   = "#00d084" if _ev >= 0 else "#ff3b3b"
        pp1, pp2, pp3 = st.columns(3)
        pp1.markdown(f"""<div style="background:#0d0d0d;border:1px solid #2a2a2a;
border-left:4px solid {_pop_col};padding:10px 14px;font-family:'IBM Plex Mono',monospace;">
  <div style="color:#555;font-size:0.76rem;letter-spacing:.08em;">PROB OF PROFIT (MC 5K sims)</div>
  <div style="color:{_pop_col};font-size:1.3rem;font-weight:700;">{_pop*100:.1f}%</div>
  <div style="color:#888;font-size:0.78rem;">Real-world drift · Vol surface skew-aware</div>
</div>""", unsafe_allow_html=True)
        pp2.markdown(f"""<div style="background:#0d0d0d;border:1px solid #2a2a2a;
border-left:4px solid {_ev_col};padding:10px 14px;font-family:'IBM Plex Mono',monospace;">
  <div style="color:#555;font-size:0.76rem;letter-spacing:.08em;">EXPECTED VALUE (₹/unit)</div>
  <div style="color:{_ev_col};font-size:1.3rem;font-weight:700;">{'+' if _ev>=0 else ''}₹{_ev:,.0f}</div>
  <div style="color:#888;font-size:0.78rem;">Avg P&L across simulated paths</div>
</div>""", unsafe_allow_html=True)
        # Prob of Touch for each leg
        _touch_info = []
        for _lmc in st.session_state.payoff_legs:
            _k_t  = float(_lmc["Strike"])
            _ot   = "call" if _lmc["Opt"] == "CE" else "put"
            _iv_t = atm_iv
            if not chain_df.empty:
                _closest_t = chain_df.iloc[(chain_df.Strike - _k_t).abs().argsort()[:1]]
                if not _closest_t.empty:
                    _col_t = "CE_IV" if _ot == "call" else "PE_IV"
                    _iv_t  = _sanitise_iv(float(_closest_t[_col_t].values[0]), atm_iv)
            _pt   = bs_prob_touch(spot, _k_t, T, r, _iv_t, _ot, q)
            _touch_info.append(f"{_lmc['Opt']} {_k_t:.0f}: {_pt*100:.0f}%")
        pp3.markdown(f"""<div style="background:#0d0d0d;border:1px solid #2a2a2a;
border-left:4px solid #9c27b0;padding:10px 14px;font-family:'IBM Plex Mono',monospace;">
  <div style="color:#555;font-size:0.76rem;letter-spacing:.08em;">PROB OF TOUCH (per leg)</div>
  <div style="color:#9c27b0;font-size:0.86rem;font-weight:700;">{"  ·  ".join(_touch_info[:4])}</div>
  <div style="color:#888;font-size:0.78rem;">Barrier prob · real-world drift adjusted</div>
</div>""", unsafe_allow_html=True)

        st.divider()

        # ── Portfolio Greeks ────────────────────────────────────────────────
        st.markdown("#### ◼ Portfolio Greeks (Net Exposure)")
        _pg_legs = []
        for _lmc in st.session_state.payoff_legs:
            _pg_legs.append({
                "opt":     _lmc["Opt"],
                "strike":  float(_lmc["Strike"]),
                "premium": float(_lmc["Premium"]),
                "action":  _lmc["Action"],
                "qty":     int(_lmc["Qty"]) * int(lot_inp),
            })
        _pg = compute_portfolio_greeks(
            _pg_legs, spot, T, r, atm_iv, q,
            chain_df=chain_df, lot_size=1)
        pg1, pg2, pg3, pg4, pg5, pg6 = st.columns(6)
        def _greek_color(v, is_gamma=False):
            if is_gamma: return "#00d084" if v > 0 else "#ff3b3b"
            return "#00d084" if v > 0 else ("#ff3b3b" if v < 0 else "#888")
        pg1.metric("Net Δ Delta",  f"{_pg['delta']:+.4f}",  help="P&L per ₹1 spot move")
        pg2.metric("Net Γ Gamma",  f"{_pg['gamma']:+.6f}",  help="Delta change per ₹1 move")
        pg3.metric("Net V Vega",   f"₹{_pg['vega']:+.2f}",  help="P&L per 1% IV change")
        pg4.metric("Net Θ Theta",  f"₹{_pg['theta']:+.2f}", help="Daily time decay ₹/day")

        # Theta/Premium ratio — most actionable metric for credit strategies
        # How many days to collect 50% of max credit (exit target)
        _total_credit = sum(
            (-1 if str(l.get("action","buy")).lower()=="sell" else 1)
            * float(l.get("premium", 0)) * int(l.get("qty",1))
            for l in _pg_legs
        )
        _theta_day = abs(_pg["theta"]) if _pg["theta"] != 0 else 1e-9
        if _total_credit < 0 and _theta_day > 0:
            # Credit strategy: theta is working for you
            _days_50pct = abs(_total_credit) * 0.5 / _theta_day
            pg5.metric("Days → 50% profit",
                       f"~{_days_50pct:.0f}d",
                       help="Estimated days for theta to collect 50% of credit (exit target)")
        else:
            _theta_pct = abs(_pg["theta"]) / max(abs(_total_credit), 1) * 100
            pg5.metric("Θ/Premium %",
                       f"{_theta_pct:.1f}%/day",
                       help="Daily theta as % of premium paid (debit) or credit received")

        # Delta notional as a cleaner risk measure than vanna
        _delta_notional = _pg["delta"] * spot
        pg6.metric("Delta ₹ Notional",
                   f"₹{_delta_notional:,.0f}",
                   help="Effective ₹ exposure per ₹1 move in spot")

        # Delta notional
        _theta_annual   = _pg["theta"] * CFG["ann_days"]
        st.caption(f"Delta notional: ₹{_delta_notional:,.0f} · "
                   f"Annualised theta: ₹{_theta_annual:,.0f} · "
                   f"Vega/Theta: {abs(_pg['vega'] / (_pg['theta'] + 1e-9)):.1f}× "
                   f"({'vol-long' if _pg['vega'] > 0 else 'vol-short'})")

        st.divider()

        # ── Kelly Position Sizing ───────────────────────────────────────────
        st.markdown("#### ◼ Kelly Position Sizing")
        st.caption(f"Fractional Kelly ({CFG['kelly_fraction']:.0%}) applied. Capped at {CFG['kelly_cap_pct']:.0%}. Capital defaults to sidebar.")
        _ks_cols = st.columns([1, 1, 1, 2])
        with _ks_cols[0]:
            _ks_capital = st.number_input("Capital (₹)", min_value=10000, max_value=100_000_000,
                                          value=st.session_state.opt_capital,
                                          step=10_000, key="kelly_capital_inp")
        with _ks_cols[1]:
            _ks_win_pct = st.number_input("Avg Win (%)", min_value=1.0, max_value=500.0,
                                           value=float(max(1.0, round(abs(_pop * 100), 1))),
                                           step=0.5, key="kelly_win_inp",
                                           help="Expected avg winning trade as % of capital at risk")
        with _ks_cols[2]:
            _ks_loss_pct = st.number_input("Avg Loss (%)", min_value=1.0, max_value=500.0,
                                            value=float(max(1.0, round((1 - _pop) * 100, 1))),
                                            step=0.5, key="kelly_loss_inp",
                                            help="Expected avg losing trade as % of capital at risk")
        _kelly_res = kelly_position_size(
            _pop, _ks_win_pct / 100.0, _ks_loss_pct / 100.0, _ks_capital)
        with _ks_cols[3]:
            _kc1, _kc2, _kc3 = st.columns(3)
            _kc1.metric("Kelly Raw",     f"{_kelly_res['kelly_raw']*100:.1f}%")
            _kc2.metric("Kelly (capped)",f"{_kelly_res['kelly_capped']*100:.1f}%")
            _kc3.metric("Position Size", f"₹{_kelly_res['position_size']:,.0f}",
                        delta=f"{_kelly_res['kelly_f']*100:.1f}% of capital")
        _kremark_col = "#00d084" if _kelly_res["kelly_raw"] > 0 else "#ff3b3b"
        st.markdown(f"<span style='color:{_kremark_col};font-size:0.83rem;font-family:IBM Plex Mono,monospace;'>"
                    f"◆ {_kelly_res['remark']}</span>", unsafe_allow_html=True)

    else:
        st.info("Click a Quick Load button above, or add legs manually to build a payoff diagram.")
        st.caption("Premiums auto-fill from Black-Scholes at current ATM IV if you don't enter them.")

# ══════════════════════════════════════════════════════════════
# TAB 8 — TRADE PLAN
# ══════════════════════════════════════════════════════════════
with t_trade:
    st.markdown("### 🚦 Trade Plan — Step-by-Step Actionable Brief")
    st.caption("Everything synthesised into one page. Read top to bottom. Each section answers one question.")

    # ── Helper styles ────────────────────────────────────────
    def _card(title, value, sub, border_col, bg="#0d0d0d"):
        return f"""<div style="background:{bg};border:1px solid #2a2a2a;border-left:4px solid {border_col};
padding:12px 16px;font-family:'IBM Plex Mono',monospace;height:100%;">
  <div style="color:#555;font-size:0.78rem;letter-spacing:.08em;text-transform:uppercase;">{title}</div>
  <div style="color:{border_col};font-size:1.05rem;font-weight:700;margin:3px 0;">{value}</div>
  <div style="color:#888;font-size:0.80rem;line-height:1.6;">{sub}</div>
</div>"""

    def _section(num, title, col="#ff8c00"):
        st.markdown(f"""<div style="border-left:4px solid {col};padding:6px 14px;margin:18px 0 8px;
font-family:'IBM Plex Mono',monospace;">
  <span style="color:#555;font-size:0.78rem;">STEP {num}</span>
  <span style="color:{col};font-size:0.96rem;font-weight:700;margin-left:10px;">{title}</span>
</div>""", unsafe_allow_html=True)

    def _rule(label, status, detail, ok_col="#00d084", fail_col="#ff3b3b", warn_col="#ffb347"):
        col = ok_col if status == "GO" else (fail_col if status == "NO" else warn_col)
        icon = "▶" if status == "GO" else ("✖" if status == "NO" else "◆")
        st.markdown(f"""<div style="display:flex;align-items:flex-start;gap:12px;padding:7px 0;
border-bottom:1px solid #1a1a1a;font-family:'IBM Plex Mono',monospace;">
  <span style="color:{col};font-size:1.0rem;min-width:18px;">{icon}</span>
  <div style="flex:1;">
    <span style="color:#e8e8e8;font-size:0.86rem;font-weight:600;">{label}</span>
    <span style="background:{col}22;color:{col};font-size:0.76rem;padding:1px 7px;
    margin-left:8px;border:1px solid {col}44;">{status}</span>
    <div style="color:#888;font-size:0.82rem;margin-top:2px;line-height:1.55;">{detail}</div>
  </div>
</div>""", unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════
    # STEP 1 — MARKET CONDITION SNAPSHOT
    # ═══════════════════════════════════════════════════════
    _section(1, "WHAT IS THE MARKET DOING RIGHT NOW?")
    _c1, _c2, _c3, _c4 = st.columns(4)
    _pu_tp  = prob_score.get("prob_up", 0.5)
    _pd_tp  = prob_score.get("prob_down", 0.5)
    _pu_tpc = "#00d084" if _pu_tp > 0.55 else ("#ff3b3b" if _pu_tp < 0.45 else "#ffb347")
    _c1.markdown(_card("Probability",
                        f"P(↑) {_pu_tp*100:.1f}%  P(↓) {_pd_tp*100:.1f}%",
                        f"logistic score {prob_score['raw_score']:+.3f} · {bias} ({bias_score:+d})",
                        _pu_tpc), unsafe_allow_html=True)
    _c2.markdown(_card("Expected Move", f"₹{prob_score['expected_move']:.0f}  ±{prob_score['expected_move_pct']:.1f}%",
                        f"ATM straddle · IV {atm_iv*100:.1f}% · DTE {dte}", v_col), unsafe_allow_html=True)

    _iv_hv_ratio = atm_iv / hv20 if hv20 and hv20 > 0.01 else 1.0
    _iv_hv_pct_now = prob_score.get("iv_hv_pct", 0.5)
    _vol_edge_lbl = (f"RICH — sell premium (IVR {ivr:.0f})" if ivr >= CFG["iv_hv_pct_sell"]
                     else f"CHEAP — buy premium (IVR {ivr:.0f})" if ivr < CFG["iv_hv_pct_buy"]
                     else f"FAIR — spread strategies (IVR {ivr:.0f})")
    _vol_edge_col = ("#ff3b3b" if ivr >= CFG["iv_hv_pct_sell"]
                     else "#1e90ff" if ivr < CFG["iv_hv_pct_buy"] else "#ffb347")
    _c3.markdown(_card("IV Regime", _vol_edge_lbl,
                        f"IV/HV={_iv_hv_ratio:.2f}× · {_iv_hv_pct_now*100:.0f}th pct of history",
                        _vol_edge_col), unsafe_allow_html=True)

    _mp = oi_d.get("max_pain", spot) if oi_d else spot
    _mp_dist    = spot - _mp
    _em_price   = prob_score.get("expected_move", step) or step
    _mp_dist_em = abs(_mp_dist) / (_em_price + 1e-9)
    _mp_dir     = f"{abs(_mp_dist):.0f}pts {'above' if _mp_dist>0 else 'below'} pain · {_mp_dist_em:.2f}× EM"
    _mp_pull_thr = CFG["safety_ratio_moderate"]   # distance > 1× EM = meaningful pull
    _mp_pull    = ("Downward pull to max pain" if _mp_dist > _em_price * _mp_pull_thr
                   else "Upward pull to max pain" if _mp_dist < -_em_price * _mp_pull_thr
                   else "Near max pain — pinning likely")
    _c4.markdown(_card("Max Pain", f"₹{_mp:,.0f}", _mp_dir, "#9c27b0"), unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════
    # STEP 2 — GO / NO-GO CHECKLIST
    # All rules driven by computed signal scores — no raw threshold re-derivation
    # ═══════════════════════════════════════════════════════
    _section(2, "IS IT SAFE TO TRADE TODAY?", "#1e90ff")
    st.caption("Rules derived from computed signal scores. All checks GO = proceed. CAUTION = reduce size. NO = wait.")

    _fs_tp       = prob_score.get("feature_scores", {})
    _flow_s      = _fs_tp.get("flow_score", 0.0)
    _flow_m      = prob_score.get("flow_magnitude", 0.0)
    _pos_s       = _fs_tp.get("positioning_score", 0.0)
    _vol_rs      = _fs_tp.get("vol_regime_score", 0.0)
    _iv_hv_pct_g = prob_score.get("iv_hv_pct", 0.5)
    _pu_conv     = prob_score.get("prob_up", 0.5)
    _dir_edge    = abs(_pu_conv - 0.5)
    _flow_thr    = st.session_state.get("opt_flow_conv_threshold", CFG["flow_conviction_seed"])

    # ── CHECK 1: FLOW SIGNAL ─────────────────────────────────────────────
    # Leading signal — if flow strongly contradicts direction, wait
    _flow_dir_ok = (_flow_s >= 0 and _pu_conv > 0.5) or (_flow_s <= 0 and _pu_conv < 0.5) or abs(_flow_s) < 0.1
    _dpcr_g  = _fs_tp.get("dPCR", 0); _dsk_g = _fs_tp.get("dSkew", 0); _div_g = _fs_tp.get("dIV", 0)
    if _flow_m >= _flow_thr and not _flow_dir_ok:
        _rule("Flow signal direction", "CAUTION",
              f"High-conviction flow ({_flow_m:.3f}) but AGAINST directional lean. "
              f"ΔPCR {_dpcr_g:+.3f} · ΔSkew {_dsk_g:+.3f} · ΔIV {_div_g:+.3f}. "
              "Flow leads price — consider non-directional or wait for alignment.")
    elif _flow_m >= _flow_thr:
        _flow_dir_lbl = "bullish" if _flow_s > 0 else "bearish"
        _rule("Flow signal", "GO",
              f"High-conviction {_flow_dir_lbl} flow ({_flow_m:.3f} ≥ threshold {_flow_thr:.3f}). "
              f"ΔPCR {_dpcr_g:+.3f} · ΔSkew {_dsk_g:+.3f} · ΔIV {_div_g:+.3f}.")
    else:
        _rule("Flow signal", "GO",
              f"Normal flow activity ({_flow_m:.3f} < threshold {_flow_thr:.3f}). "
              f"Flow score {_flow_s:+.3f} — no extreme positioning change detected.")

    # ── CHECK 2: POSITIONING ─────────────────────────────────────────────
    _pcr_lz_g = _fs_tp.get("pcr_level_z", 0)
    _oi_sk_g  = _fs_tp.get("oi_skew_z", 0)
    _pos_conflict = (abs(_pos_s) > 0.3 and
                     ((_pos_s > 0 and _pu_conv < 0.45) or (_pos_s < 0 and _pu_conv > 0.55)))
    if _pos_conflict:
        _rule("Positioning vs direction", "CAUTION",
              f"OI positioning ({_pos_s:+.3f}) conflicts with P(↑)={_pu_conv*100:.1f}%. "
              f"PCR level {_pcr_lz_g:+.3f} · OI skew {_oi_sk_g:+.3f}. "
              "Structural OI argues against the directional lean — use defined-risk only.")
    else:
        _rule("Positioning aligned", "GO",
              f"Positioning score {_pos_s:+.3f} consistent with P(↑)={_pu_conv*100:.1f}%. "
              f"PCR level {_pcr_lz_g:+.3f} · OI skew {_oi_sk_g:+.3f}.")

    # ── CHECK 3: VOLATILITY REGIME ───────────────────────────────────────
    # Extreme IV percentiles = dangerous for both buyers and sellers
    if _iv_hv_pct_g > 0.92:
        _rule("Vol regime", "CAUTION",
              f"IV/HV at {_iv_hv_pct_g*100:.0f}th pct — extreme vol. Vol regime score {_vol_rs:+.3f}. "
              "Mean-reversion of IV likely. Size down credit strategies; avoid naked long vega.")
    elif _iv_hv_pct_g < 0.08:
        _rule("Vol regime", "CAUTION",
              f"IV/HV at {_iv_hv_pct_g*100:.0f}th pct — vol compressed. Vol regime score {_vol_rs:+.3f}. "
              "Options very cheap but breakout risk elevated. Prefer debit strategies.")
    else:
        _rule("Vol regime", "GO",
              f"IV/HV at {_iv_hv_pct_g*100:.0f}th pct · regime score {_vol_rs:+.3f} — tradeable range.")

    # ── CHECK 4: STRUCTURAL RISK (DTE + Gamma Flip + Events) ────────────
    _struct_issues = []
    _gflip_d_g = float('inf')   # safe default: infinitely far from gamma flip
    _liq_score = liquidity_d.get("liquid_score", 75)
    if dte < 2:
        _struct_issues.append(f"DTE={dte} (expiry day — avoid new positions)")
    elif dte < 5:
        _struct_issues.append(f"DTE={dte} (near expiry — gamma risk elevated)")
    if oi_d:
        _gflip_g   = oi_d.get("gamma_flip", spot)
        _gflip_d_g = abs(spot - _gflip_g) / (prob_score.get("expected_move", step) + 1e-9)
        if _gflip_d_g < 0.25:
            _struct_issues.append(f"Gamma flip at {_gflip_d_g:.2f}× EM (very close — vol expansion risk)")
    _near_events = [e for e in events_list if e.get("days_away", 99) <= dte]
    if _near_events:
        _struct_issues.append(f"{len(_near_events)} event(s) within expiry window")
    _liq_score = liquidity_d.get("liquid_score", 75)
    if _liq_score < 50:
        _struct_issues.append(f"Liquidity score {_liq_score}/100 — wide spreads")

    if any("expiry day" in s for s in _struct_issues):
        _rule("Structural risk", "NO",
              " | ".join(_struct_issues))
    elif _struct_issues:
        _rule("Structural risk", "CAUTION",
              " | ".join(_struct_issues))
    else:
        _rule("Structural risk", "GO",
              f"DTE {dte} · Gamma flip {_gflip_d_g:.2f}× EM away · Liquidity {_liq_score}/100 · No events in window."
              if oi_d else f"DTE {dte} · No structural issues detected.")

    # ═══════════════════════════════════════════════════════
    # STEP 3 — WHICH STRATEGY TO USE
    # ═══════════════════════════════════════════════════════
    _section(3, "WHICH STRATEGY TO USE?", "#00d084")

    if strat_recs:
        _best = strat_recs[0]
        _ev_b   = _best.get("ev", 0)
        _pop_b  = _best.get("pop", 0.5)
        _kelly_b = _best.get("kelly", 0) * 100
        _score_col = "#00d084" if _best["Score"] >= 70 else "#ffb347" if _best["Score"] >= 50 else "#ff3b3b"
        _ev_col    = "#00d084" if _ev_b >= 0 else "#ff3b3b"
        st.markdown(f"""
<div style="background:#0d1a00;border:1px solid #ff8c00;border-top:4px solid #ff8c00;
padding:16px 20px;font-family:'IBM Plex Mono',monospace;margin-bottom:12px;">
  <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;">
    <div>
      <div style="color:#555;font-size:0.78rem;letter-spacing:.1em;">TOP STRATEGY BY EV  ·  {_best['Ideal DTE']}</div>
      <div style="color:#e8e8e8;font-size:1.1rem;font-weight:700;">{_best['Strategy']}</div>
      <div style="color:#7ec8e3;font-size:0.84rem;margin-top:4px;">{_best['Type']}</div>
      <div style="color:#888;font-size:0.82rem;margin-top:6px;line-height:1.6;">{_best['Rationale']}</div>
    </div>
    <div style="text-align:right;min-width:100px;">
      <div style="color:#555;font-size:0.78rem;">EV SCORE</div>
      <div style="color:{_score_col};font-size:1.8rem;font-weight:700;">{min(100,_best['Score'])}</div>
      <div style="color:{_ev_col};font-size:0.82rem;font-weight:600;">EV {'+' if _ev_b>=0 else ''}₹{_ev_b:,.0f}</div>
      <div style="color:#888;font-size:0.78rem;">POP {_pop_b*100:.1f}% · Kelly {_kelly_b:.1f}%</div>
    </div>
  </div>
</div>""", unsafe_allow_html=True)

        if len(strat_recs) > 1:
            _alt_cols = st.columns(min(3, len(strat_recs)-1))
            for _ai, _alt in enumerate(strat_recs[1:4]):
                _ev_a = _alt.get("ev", 0)
                _ac   = "#ffb347" if _alt["Score"] >= 60 else "#555"
                _ev_ac = "#00d084" if _ev_a >= 0 else "#ff3b3b"
                _alt_cols[_ai].markdown(f"""
<div style="background:#0d0d0d;border:1px solid #2a2a2a;padding:10px 12px;
font-family:'IBM Plex Mono',monospace;height:100%;">
  <div style="color:#555;font-size:0.76rem;">ALT #{_ai+2}</div>
  <div style="color:#e8e8e8;font-size:0.88rem;font-weight:600;">{_alt['Strategy']}</div>
  <div style="color:{_ac};font-size:0.80rem;margin-top:3px;">Score {min(100,_alt['Score'])}/100</div>
  <div style="color:{_ev_ac};font-size:0.78rem;">EV {'+' if _ev_a>=0 else ''}₹{_ev_a:,.0f} · POP {_alt.get('pop',0.5)*100:.0f}%</div>
  <div style="color:#555;font-size:0.78rem;margin-top:4px;">{_alt['Type']}</div>
</div>""", unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════
    # STEP 4 — TRADE TICKET: EXACT ORDERS TO PLACE
    # ═══════════════════════════════════════════════════════
    _section(4, "TRADE TICKET — EXACT ORDERS TO PLACE", "#ff8c00")

    # Initialise _leg_def here so Step 7's stop-loss logic always has it in scope,
    # even if the strategy name doesn't match any template below.
    _leg_def = []

    if strat_recs:
        _best  = strat_recs[0]
        _sv    = float(step)
        _strat = _best["Strategy"]
        _lot_d = CFG["lot_sizes"].get(sym.upper(), CFG["lot_size_fallback"])

        # ── Central leg definition lookup ──────────────────────
        # Each entry: (action, option_type, strike, qty_multiplier, role_description)
        # qty_multiplier = 1 normally, 2 for the body of a butterfly
        _leg_def = []

        if "Near-Expiry Straddle Sell" in _strat or ("Short Straddle" in _strat and "Near" in _strat):
            _leg_def = [
                ("SELL", "CE", atm_k,        _bs_c(atm_k),        1, "Sell ATM call — collect full premium"),
                ("SELL", "PE", atm_k,        _bs_p(atm_k),        1, "Sell ATM put  — collect full premium"),
            ]
        elif "Short Straddle" in _strat:
            _leg_def = [
                ("SELL", "CE", atm_k,        _bs_c(atm_k),        1, "Sell ATM call — profit if spot stays near ATM"),
                ("SELL", "PE", atm_k,        _bs_p(atm_k),        1, "Sell ATM put  — profit if spot stays near ATM"),
            ]
        elif "Short Strangle" in _strat:
            _leg_def = [
                ("SELL", "PE", atm_k - _sv,  _bs_p(atm_k-_sv),   1, f"Sell {atm_k-_sv:.0f} PE — OTM put, collect credit below spot"),
                ("SELL", "CE", atm_k + _sv,  _bs_c(atm_k+_sv),   1, f"Sell {atm_k+_sv:.0f} CE — OTM call, collect credit above spot"),
            ]
        elif "Bull Call Spread" in _strat:
            _leg_def = [
                ("BUY",  "CE", atm_k,        _bs_c(atm_k),        1, f"Buy  {atm_k:.0f} CE — long call, profits above breakeven"),
                ("SELL", "CE", atm_k + _sv,  _bs_c(atm_k+_sv),   1, f"Sell {atm_k+_sv:.0f} CE — caps upside, reduces net cost"),
            ]
        elif "Bear Put Spread" in _strat:
            _leg_def = [
                ("BUY",  "PE", atm_k,        _bs_p(atm_k),        1, f"Buy  {atm_k:.0f} PE — long put, profits below breakeven"),
                ("SELL", "PE", atm_k - _sv,  _bs_p(atm_k-_sv),   1, f"Sell {atm_k-_sv:.0f} PE — caps downside gain, reduces cost"),
            ]
        elif "Bull Put Spread" in _strat:
            _leg_def = [
                ("SELL", "PE", atm_k,        _bs_p(atm_k),        1, f"Sell {atm_k:.0f} PE — short put, collect premium, stay above this"),
                ("BUY",  "PE", atm_k - _sv,  _bs_p(atm_k-_sv),   1, f"Buy  {atm_k-_sv:.0f} PE — wing protection, max loss capped here"),
            ]
        elif "Bear Call Spread" in _strat:
            _leg_def = [
                ("SELL", "CE", atm_k,        _bs_c(atm_k),        1, f"Sell {atm_k:.0f} CE — short call, collect premium, stay below this"),
                ("BUY",  "CE", atm_k + _sv,  _bs_c(atm_k+_sv),   1, f"Buy  {atm_k+_sv:.0f} CE — wing protection, max loss capped here"),
            ]
        elif "Iron Condor" in _strat:
            _leg_def = [
                ("SELL", "PE", atm_k - _sv,   _bs_p(atm_k-_sv),   1, f"Sell {atm_k-_sv:.0f} PE — short put, profit zone starts here"),
                ("BUY",  "PE", atm_k - 2*_sv, _bs_p(atm_k-2*_sv), 1, f"Buy  {atm_k-2*_sv:.0f} PE — defines max loss on downside"),
                ("SELL", "CE", atm_k + _sv,   _bs_c(atm_k+_sv),   1, f"Sell {atm_k+_sv:.0f} CE — short call, profit zone ends here"),
                ("BUY",  "CE", atm_k + 2*_sv, _bs_c(atm_k+2*_sv), 1, f"Buy  {atm_k+2*_sv:.0f} CE — defines max loss on upside"),
            ]
        elif "Iron Butterfly" in _strat:
            _leg_def = [
                ("SELL", "CE", atm_k,        _bs_c(atm_k),        1, f"Sell {atm_k:.0f} CE — short ATM call, max profit if spot pins here"),
                ("SELL", "PE", atm_k,        _bs_p(atm_k),        1, f"Sell {atm_k:.0f} PE — short ATM put, max profit if spot pins here"),
                ("BUY",  "CE", atm_k + _sv,  _bs_c(atm_k+_sv),   1, f"Buy  {atm_k+_sv:.0f} CE — upper wing, caps upside loss"),
                ("BUY",  "PE", atm_k - _sv,  _bs_p(atm_k-_sv),   1, f"Buy  {atm_k-_sv:.0f} PE — lower wing, caps downside loss"),
            ]
        elif "Long Straddle" in _strat:
            _leg_def = [
                ("BUY", "CE", atm_k, _bs_c(atm_k), 1, f"Buy  {atm_k:.0f} CE — profits if spot rallies beyond upper breakeven"),
                ("BUY", "PE", atm_k, _bs_p(atm_k), 1, f"Buy  {atm_k:.0f} PE — profits if spot falls below lower breakeven"),
            ]
        elif "Long Strangle" in _strat:
            _leg_def = [
                ("BUY", "CE", atm_k + _sv, _bs_c(atm_k+_sv), 1, f"Buy  {atm_k+_sv:.0f} CE — OTM call, needs strong rally to profit"),
                ("BUY", "PE", atm_k - _sv, _bs_p(atm_k-_sv), 1, f"Buy  {atm_k-_sv:.0f} PE — OTM put, needs strong sell-off to profit"),
            ]
        elif "Long ATM Call" in _strat or ("Long OTM Call" in _strat):
            _k_c = atm_k + _sv if "OTM" in _strat else atm_k
            _leg_def = [("BUY", "CE", _k_c, _bs_c(_k_c), 1, f"Buy  {_k_c:.0f} CE — directional long, profits above breakeven")]
        elif "Long ATM Put" in _strat:
            _leg_def = [("BUY", "PE", atm_k, _bs_p(atm_k), 1, f"Buy  {atm_k:.0f} PE — directional short, profits below breakeven")]
        elif "Short Put" in _strat:
            _leg_def = [("SELL", "PE", atm_k - _sv, _bs_p(atm_k-_sv), 1, f"Sell {atm_k-_sv:.0f} PE — naked short put, keep premium if spot stays above")]
        elif "Short Call" in _strat:
            _leg_def = [("SELL", "CE", atm_k + _sv, _bs_c(atm_k+_sv), 1, f"Sell {atm_k+_sv:.0f} CE — naked short call, keep premium if spot stays below")]
        elif "Jade Lizard" in _strat:
            _leg_def = [
                ("SELL", "PE", atm_k,        _bs_p(atm_k),        1, f"Sell {atm_k:.0f} PE — short put, bullish anchor"),
                ("SELL", "CE", atm_k + _sv,  _bs_c(atm_k+_sv),   1, f"Sell {atm_k+_sv:.0f} CE — short call, upside premium"),
                ("BUY",  "CE", atm_k + 2*_sv,_bs_c(atm_k+2*_sv), 1, f"Buy  {atm_k+2*_sv:.0f} CE — call wing, caps upside loss"),
            ]
        elif "Call Ratio Backspread" in _strat:
            _leg_def = [
                ("SELL", "CE", atm_k - _sv,  _bs_c(atm_k-_sv),   1, f"Sell 1× {atm_k-_sv:.0f} CE — short lower call (×1 lot), enter for credit"),
                ("BUY",  "CE", atm_k,        _bs_c(atm_k),        2, f"Buy  2× {atm_k:.0f} CE — long ATM calls (×2 lots), profits from big move up"),
            ]
        elif "Calendar" in _strat:
            # Near month premium is known; back month is unknown without loading that expiry.
            # Estimate: back month ≈ near month × sqrt(T_far/T_near) from B-S scaling.
            # For a typical near:far = 10d:35d ratio, scale ≈ sqrt(35/10) ≈ 1.87.
            # We use sqrt(T_far_est/T_near) with T_far_est = T*3.5 as a rough estimate.
            _near_prem = _bs_c(atm_k)
            _far_est   = round(_near_prem * math.sqrt(3.5), 2)  # sqrt(3.5) ≈ far/near T ratio
            _leg_def = [
                ("SELL", "CE", atm_k, _near_prem, 1, f"Sell {atm_k:.0f} CE near expiry — sell theta-rich front month"),
                ("BUY",  "CE", atm_k, _far_est,   1, f"Buy  {atm_k:.0f} CE far expiry  — est. ₹{_far_est:.2f} (√(T_far/T_near) scaling)"),
            ]
        elif "Butterfly" in _strat:
            _leg_def = [
                ("BUY",  "CE", atm_k - _sv,  _bs_c(atm_k-_sv),  1, f"Buy  {atm_k-_sv:.0f} CE — left wing (lower strike)"),
                ("SELL", "CE", atm_k,         _bs_c(atm_k),       2, f"Sell 2× {atm_k:.0f} CE — body (×2 lots), short the middle"),
                ("BUY",  "CE", atm_k + _sv,  _bs_c(atm_k+_sv),  1, f"Buy  {atm_k+_sv:.0f} CE — right wing (upper strike)"),
            ]

        if _leg_def:
            # ── Trade Ticket Card ───────────────────────────────
            _net_cost  = 0.0
            _ticket_rows = []
            for _act, _opt, _k, _pr, _qty_m, _desc in _leg_def:
                _sign        = 1 if _act == "BUY" else -1
                _net_cost   += _sign * _pr * _qty_m
                _per_lot     = _pr * _lot_d * _qty_m
                _ticket_rows.append({
                    "ORDER":      _act,
                    "QTY":        f"{_qty_m} lot{'s' if _qty_m>1 else ''} = {_qty_m*_lot_d:,} shares",
                    "TYPE":       _opt,
                    "STRIKE":     f"₹{_k:,.0f}",
                    "EXPIRY":     expiry,
                    "PREM/SHARE": f"₹{_pr:.2f}",
                    "TOTAL/LOT":  f"₹{_per_lot:,.0f}",
                    "ROLE":       _desc,
                })

            _tdf = pd.DataFrame(_ticket_rows)
            def _order_style(v):
                if v == "BUY":  return "background:#0a2200;color:#00d084;font-weight:700;font-size:0.88rem"
                if v == "SELL": return "background:#220000;color:#ff3b3b;font-weight:700;font-size:0.88rem"
                return ""
            def _type_style(v):
                if v == "CE": return "color:#1e90ff;font-weight:600"
                if v == "PE": return "color:#ff8c00;font-weight:600"
                return ""
            st.dataframe(
                _tdf.style
                    .map(_order_style, subset=["ORDER"])
                    .map(_type_style,  subset=["TYPE"]),
                use_container_width=True, hide_index=True
            )

            # ── Net cost + breakevens ───────────────────────────
            _cost_lbl = "NET DEBIT — you PAY this upfront" if _net_cost > 0 else "NET CREDIT — you RECEIVE this upfront"
            _cost_col = "#ff3b3b" if _net_cost > 0 else "#00d084"
            _net_abs  = abs(_net_cost)
            _net_lot  = _net_abs * _lot_d

            # ── Breakeven computation — verified against P&L=0 at expiry ──
            # Rule: BE is the price where total payoff (intrinsic - premium paid + premium received) = 0
            # For DEBIT strategies:  BE = long_strike ± net_debit
            # For CREDIT strategies: BE = short_strike ± net_credit  (NOT long_strike)
            # Short strangle: BEs = short_put_strike - credit  AND  short_call_strike + credit
            _bes = []
            if "Long ATM Call" in _strat or "Long OTM Call" in _strat:
                # Single long call: BE = strike + premium
                _k0 = _leg_def[0][2]; _pr0 = _leg_def[0][3]
                _bes = [f"₹{_k0 + _pr0:,.0f}  (strike {_k0:.0f} + premium {_pr0:.2f})"]

            elif "Long ATM Put" in _strat:
                _k0 = _leg_def[0][2]; _pr0 = _leg_def[0][3]
                _bes = [f"₹{_k0 - _pr0:,.0f}  (strike {_k0:.0f} − premium {_pr0:.2f})"]

            elif "Bull Call Spread" in _strat or "Bear Put Spread" in _strat:
                # Debit spreads: BE = long_strike ± net_debit
                _long_l  = [l for l in _leg_def if l[0]=="BUY"][0]
                if _long_l[1] == "CE":
                    _bes = [f"₹{_long_l[2] + _net_abs:,.0f}  (long call {_long_l[2]:.0f} + debit {_net_abs:.2f})"]
                else:
                    _bes = [f"₹{_long_l[2] - _net_abs:,.0f}  (long put {_long_l[2]:.0f} − debit {_net_abs:.2f})"]

            elif "Bull Put Spread" in _strat:
                # Credit spread: BE = short_put_strike - net_credit
                _short_l = [l for l in _leg_def if l[0]=="SELL" and l[1]=="PE"][0]
                _bes = [f"₹{_short_l[2] - _net_abs:,.0f}  (short put {_short_l[2]:.0f} − credit {_net_abs:.2f})"]

            elif "Bear Call Spread" in _strat:
                # Credit spread: BE = short_call_strike + net_credit
                _short_l = [l for l in _leg_def if l[0]=="SELL" and l[1]=="CE"][0]
                _bes = [f"₹{_short_l[2] + _net_abs:,.0f}  (short call {_short_l[2]:.0f} + credit {_net_abs:.2f})"]

            elif "Short Straddle" in _strat or "Near-Expiry Straddle" in _strat:
                # Both shorts at ATM: BEs = ATM ± total_credit
                _tot_cr = sum(l[3] for l in _leg_def)
                _bes = [f"₹{atm_k - _tot_cr:,.0f}  (ATM {atm_k:.0f} − credit {_tot_cr:.2f})",
                        f"₹{atm_k + _tot_cr:,.0f}  (ATM {atm_k:.0f} + credit {_tot_cr:.2f})"]

            elif "Short Strangle" in _strat:
                # Short put at atm-sv, short call at atm+sv: DIFFERENT strikes
                _short_pe = [l for l in _leg_def if l[0]=="SELL" and l[1]=="PE"][0]
                _short_ce = [l for l in _leg_def if l[0]=="SELL" and l[1]=="CE"][0]
                _tot_cr = _short_pe[3] + _short_ce[3]
                _bes = [f"₹{_short_pe[2] - _tot_cr:,.0f}  (put strike {_short_pe[2]:.0f} − credit {_tot_cr:.2f})",
                        f"₹{_short_ce[2] + _tot_cr:,.0f}  (call strike {_short_ce[2]:.0f} + credit {_tot_cr:.2f})"]

            elif "Long Straddle" in _strat:
                _bes = [f"₹{atm_k - _net_abs:,.0f}  (ATM {atm_k:.0f} − debit {_net_abs:.2f})",
                        f"₹{atm_k + _net_abs:,.0f}  (ATM {atm_k:.0f} + debit {_net_abs:.2f})"]

            elif "Long Strangle" in _strat:
                _long_pe = [l for l in _leg_def if l[1]=="PE"][0]
                _long_ce = [l for l in _leg_def if l[1]=="CE"][0]
                _bes = [f"₹{_long_pe[2] - _net_abs:,.0f}  (put strike {_long_pe[2]:.0f} − total debit {_net_abs:.2f})",
                        f"₹{_long_ce[2] + _net_abs:,.0f}  (call strike {_long_ce[2]:.0f} + total debit {_net_abs:.2f})"]

            elif "Iron Condor" in _strat or "Iron Butterfly" in _strat:
                _sells_ce = [l for l in _leg_def if l[0]=="SELL" and l[1]=="CE"]
                _sells_pe = [l for l in _leg_def if l[0]=="SELL" and l[1]=="PE"]
                if _sells_ce and _sells_pe:
                    _net_cr = sum(l[3] for l in _leg_def if l[0]=="SELL") - sum(l[3] for l in _leg_def if l[0]=="BUY")
                    _bes = [f"₹{_sells_pe[0][2] - _net_cr:,.0f}  (short put {_sells_pe[0][2]:.0f} − credit {_net_cr:.2f})",
                            f"₹{_sells_ce[0][2] + _net_cr:,.0f}  (short call {_sells_ce[0][2]:.0f} + credit {_net_cr:.2f})"]

            elif "Calendar" in _strat:
                # Cannot compute BE without knowing back-month price at near expiry
                _bes = ["Cannot compute at entry — depends on back-month IV at near-expiry date"]

            st.markdown(f"""
<div style="background:#0d0d0d;border:1px solid #ff8c00;padding:14px 20px;margin-top:6px;
font-family:'IBM Plex Mono',monospace;">
  <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-bottom:12px;">
    <div>
      <div style="color:#555;font-size:0.78rem;letter-spacing:.08em;">NET COST / CREDIT</div>
      <div style="color:{_cost_col};font-size:1.0rem;font-weight:700;">
        {'−' if _net_cost > 0 else '+'}₹{_net_abs:.2f} per share</div>
      <div style="color:{_cost_col};font-size:0.88rem;">
        {'−' if _net_cost > 0 else '+'}₹{_net_lot:,.0f} per lot</div>
      <div style="color:#888;font-size:0.80rem;margin-top:3px;">{_cost_lbl}</div>
    </div>
    <div>
      <div style="color:#555;font-size:0.78rem;letter-spacing:.08em;">LOT SIZE</div>
      <div style="color:#e8e8e8;font-size:1.0rem;font-weight:700;">{_lot_d} shares</div>
      <div style="color:#888;font-size:0.80rem;">{sym} F&O lot size</div>
    </div>
    <div>
      <div style="color:#555;font-size:0.78rem;letter-spacing:.08em;">EXPIRY</div>
      <div style="color:#ff8c00;font-size:1.0rem;font-weight:700;">{expiry}</div>
      <div style="color:#888;font-size:0.80rem;">DTE: {dte} trading days</div>
    </div>
  </div>
  {'<div style="border-top:1px solid #2a2a2a;padding-top:10px;"><div style="color:#555;font-size:0.78rem;letter-spacing:.08em;margin-bottom:6px;">BREAKEVEN PRICES AT EXPIRY</div>' + "".join([f'<div style="color:#ffb347;font-size:0.88rem;margin:2px 0;">📍 {be}</div>' for be in _bes]) + '</div>' if _bes else ''}
</div>""", unsafe_allow_html=True)

            # ── P&L at key levels table ─────────────────────────
            st.caption("P&L at expiry across key price levels (1 lot, theoretical):")
            _exp_move_pct = oi_d.get("exp_move_pct", atm_iv*100*math.sqrt(T)*math.sqrt(2/math.pi)) if oi_d else 5.0
            _key_prices = sorted(set([
                round(spot * (1 - _exp_move_pct/100), 0),
                round(spot * (1 - _exp_move_pct/200), 0),
                oi_d.get("put_wall", spot - 2*_sv) if oi_d else spot - 2*_sv,
                atm_k - _sv,
                atm_k,
                atm_k + _sv,
                oi_d.get("call_wall", spot + 2*_sv) if oi_d else spot + 2*_sv,
                round(spot * (1 + _exp_move_pct/200), 0),
                round(spot * (1 + _exp_move_pct/100), 0),
            ]))
            _pnl_rows = []
            for _px in _key_prices:
                _total_pnl = 0.0
                for _act, _opt, _k, _pr, _qty_m, _ in _leg_def:
                    _intr = max(_px - _k, 0) if _opt == "CE" else max(_k - _px, 0)
                    _d    = 1 if _act == "BUY" else -1
                    _total_pnl += _d * (_intr - _pr) * _qty_m * _lot_d
                _pnl_rows.append({
                    "Price at Expiry": f"₹{_px:,.0f}",
                    # Format: +₹NNN for profit, -₹NNN for loss (minus before ₹ for CSS detection)
                    "P&L (₹)": f"+₹{_total_pnl:,.0f}" if _total_pnl >= 0 else f"-₹{abs(_total_pnl):,.0f}",
                    "vs Spot":  f"{'↑' if _px > spot else '↓' if _px < spot else '—'} {abs(_px-spot):.0f}pts",
                })
            _pnl_df = pd.DataFrame(_pnl_rows)
            def _pnl_style(v):
                if isinstance(v, str) and v.startswith("+₹"): return "color:#00d084;font-weight:600"
                if isinstance(v, str) and v.startswith("-₹") or (isinstance(v, str) and "−" in v): return "color:#ff3b3b;font-weight:600"
                return ""
            st.dataframe(_pnl_df.style.map(_pnl_style, subset=["P&L (₹)"]),
                         use_container_width=True, hide_index=True)

        else:
            st.info(f"Exact leg breakdown not yet templated for **{_strat}**. "
                    "Use the Payoff Builder tab to construct it manually — "
                    f"legs are: {_best['Legs']}")


    # ═══════════════════════════════════════════════════════
    # STEP 5 — KEY LEVELS: WHERE TO ENTER, STOP, TARGET
    # ═══════════════════════════════════════════════════════
    _section(5, "KEY PRICE LEVELS", "#ffb347")

    _call_wall = oi_d.get("call_wall", spot+step*3) if oi_d else spot+step*3
    _put_wall  = oi_d.get("put_wall",  spot-step*3) if oi_d else spot-step*3
    _exp_move  = oi_d.get("exp_move_pct", atm_iv*100*math.sqrt(T)*math.sqrt(2/math.pi)) if oi_d else 0
    _exp_r_abs = spot * _exp_move / 100
    _atr       = bias_res.get("atr", _atr_seed(spot))

    _l1, _l2, _l3 = st.columns(3)
    with _l1:
        st.markdown(f"""
<div style="background:#0d0d0d;border:1px solid #2a2a2a;padding:12px 16px;
font-family:'IBM Plex Mono',monospace;border-top:3px solid #1e90ff;">
  <div style="color:#1e90ff;font-size:0.80rem;font-weight:600;letter-spacing:.08em;
  margin-bottom:8px;">SUPPORT LEVELS</div>
  <div style="color:#e8e8e8;font-size:0.86rem;line-height:2.0;">
    Put Wall:  <b style="color:#00d084;">₹{_put_wall:,.0f}</b><br>
    Max Pain:  <b style="color:#9c27b0;">₹{_mp:,.0f}</b><br>
    EMA 20:    <b style="color:#1e90ff;">₹{bias_res.get('e20', spot):,.2f}</b><br>
    −1σ Move:  <b style="color:#888;">₹{spot - _exp_r_abs:,.0f}</b><br>
    −1 ATR:    <b style="color:#555;">₹{spot - _atr:,.0f}</b>
  </div>
</div>""", unsafe_allow_html=True)
    with _l2:
        st.markdown(f"""
<div style="background:#0d0d0d;border:1px solid #2a2a2a;padding:12px 16px;
font-family:'IBM Plex Mono',monospace;border-top:3px solid #ff8c00;">
  <div style="color:#ff8c00;font-size:0.80rem;font-weight:600;letter-spacing:.08em;
  margin-bottom:8px;">CURRENT SPOT</div>
  <div style="color:#ff8c00;font-size:1.3rem;font-weight:700;margin-bottom:6px;">
  ₹{spot:,.2f}</div>
  <div style="color:#888;font-size:0.82rem;line-height:1.9;">
    ATM Strike: ₹{atm_k:,.0f}<br>
    Bias: {bias} ({bias_score:+d})<br>
    ATM IV: {atm_iv*100:.1f}%<br>
    DTE: {dte} trading days
  </div>
</div>""", unsafe_allow_html=True)
    with _l3:
        st.markdown(f"""
<div style="background:#0d0d0d;border:1px solid #2a2a2a;padding:12px 16px;
font-family:'IBM Plex Mono',monospace;border-top:3px solid #ff3b3b;">
  <div style="color:#ff3b3b;font-size:0.80rem;font-weight:600;letter-spacing:.08em;
  margin-bottom:8px;">RESISTANCE LEVELS</div>
  <div style="color:#e8e8e8;font-size:0.86rem;line-height:2.0;">
    Call Wall:  <b style="color:#ff3b3b;">₹{_call_wall:,.0f}</b><br>
    Max Pain:   <b style="color:#9c27b0;">₹{_mp:,.0f}</b><br>
    EMA 20:     <b style="color:#1e90ff;">₹{bias_res.get('e20', spot):,.2f}</b><br>
    +1σ Move:   <b style="color:#888;">₹{spot + _exp_r_abs:,.0f}</b><br>
    +1 ATR:     <b style="color:#555;">₹{spot + _atr:,.0f}</b>
  </div>
</div>""", unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════
    # STEP 6 — POSITION SIZING
    # ═══════════════════════════════════════════════════════
    _section(6, "POSITION SIZING — HOW MANY LOTS?", "#9c27b0")
    st.caption("Capital from sidebar. Kelly fraction computed by EV engine from POP and max risk.")

    _account_size   = st.session_state.opt_capital
    _risk_pct_cfg   = CFG["kelly_cap_pct"] * CFG["kelly_fraction"] * 100

    _ps1, _ps2 = st.columns(2)
    with _ps1:
        _account_override = st.number_input(
            "Capital (₹)", min_value=10_000, max_value=100_000_000,
            value=_account_size, step=10_000, key="ps_account",
            help="Defaults to sidebar capital.")
        _risk_pct = st.slider(
            "Max risk per trade (%)", min_value=0.5,
            max_value=float(CFG["kelly_cap_pct"] * 100),
            value=round(_risk_pct_cfg, 1), step=0.5, key="ps_risk_pct")
    with _ps2:
        _max_risk_rs   = _account_override * _risk_pct / 100
        _lot_sz_ps     = CFG["lot_sizes"].get(sym.upper(), CFG["lot_size_fallback"])
        _is_unlimited_risk = False
        if strat_recs:
            _mr     = strat_recs[0]["Max Risk"]
            _mr_str = str(_mr).strip()
            _is_unlimited_risk = any(w in _mr_str for w in ["Unlimited","unlimited","Large","large"])
            try:
                _risk_per_share = float(_mr_str.replace("₹","").replace(",","").strip())
                _risk_per_lot   = _risk_per_share * _lot_sz_ps
            except Exception:
                _risk_per_lot = 0
            _safe_lots = max(1, int(_max_risk_rs / _risk_per_lot)) if _risk_per_lot > 0 else 1
            # EV-based Kelly size
            _ev_kelly_raw = strat_recs[0].get("kelly_raw", 0)
            _ev_kelly_cap = max(0.0, min(CFG["kelly_cap_pct"], _ev_kelly_raw))
            _ev_kelly_f   = _ev_kelly_cap * CFG["kelly_fraction"]
            _ev_pos_size  = round(_ev_kelly_f * _account_override)
        else:
            _safe_lots = 1; _risk_per_lot = 0
            _ev_kelly_raw = 0; _ev_kelly_f = 0; _ev_pos_size = 0

        if _is_unlimited_risk:
            st.warning("⚠️ Strategy has **unlimited risk**. Use 1 lot max with a hard stop-loss.")

        st.markdown(f"""
<div style="background:#0d0d0d;border:1px solid {'#ff3b3b' if _is_unlimited_risk else '#2a2a2a'};
padding:14px 18px;font-family:'IBM Plex Mono',monospace;">
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;">
    <div>
      <div style="color:#555;font-size:0.78rem;">RISK BUDGET</div>
      <div style="color:#ff8c00;font-size:1.0rem;font-weight:700;">₹{_max_risk_rs:,.0f}</div>
      <div style="color:#888;font-size:0.80rem;">{_risk_pct:.1f}% of ₹{_account_override:,.0f}</div>
    </div>
    <div>
      <div style="color:#555;font-size:0.78rem;">SAFE LOTS (risk-based)</div>
      <div style="color:{'#ff3b3b' if _is_unlimited_risk else '#00d084'};font-size:1.0rem;font-weight:700;">
        {'⚠ 1 lot MAX' if _is_unlimited_risk else f"{_safe_lots} lot{'s' if _safe_lots!=1 else ''}"}</div>
      <div style="color:#888;font-size:0.80rem;">= {_safe_lots * _lot_sz_ps:,} shares</div>
    </div>
    <div>
      <div style="color:#555;font-size:0.78rem;">KELLY POSITION SIZE</div>
      <div style="color:#9c27b0;font-size:1.0rem;font-weight:700;">
        {'₹' + f'{_ev_pos_size:,.0f}' if _ev_pos_size > 0 else '—'}</div>
      <div style="color:#888;font-size:0.80rem;">{_ev_kelly_f*100:.1f}% Kelly (raw {_ev_kelly_raw*100:.1f}% → cap → ×{CFG['kelly_fraction']:.0%})</div>
    </div>
    <div>
      <div style="color:#555;font-size:0.78rem;">RISK / REWARD PER LOT</div>
      <div style="color:#e8e8e8;font-size:0.96rem;">
        {f'₹{_risk_per_lot:,.0f}' if _risk_per_lot > 0 and not _is_unlimited_risk else (strat_recs[0]['Max Risk'] if strat_recs else '—')}
        &nbsp;/&nbsp; {strat_recs[0]['Max Reward'] if strat_recs else '—'}</div>
    </div>
  </div>
</div>""", unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════
    # STEP 7 — ENTRY / EXIT / STOP RULES
    # ═══════════════════════════════════════════════════════
    _section(7, "ENTRY, EXIT AND STOP-LOSS RULES", "#ff8c00")

    # Dynamic stop levels based on ATR, gamma flip, and strategy type
    _is_credit   = strat_recs and "Credit" in strat_recs[0]["Type"] if strat_recs else False
    _is_neutral  = strat_recs and "Non-Directional" in strat_recs[0]["Type"] if strat_recs else True

    # Entry timing: bias + vol confirmation
    _entry_conds = []
    if bias_score >= 12:
        _entry_conds.append(f"• Bullish bias confirmed — enter on any dip toward EMA20 (₹{bias_res.get('e20',spot):,.0f}) or retrace to ATM")
    elif bias_score <= -12:
        _entry_conds.append(f"• Bearish bias confirmed — enter on any bounce toward EMA20 (₹{bias_res.get('e20',spot):,.0f}) or retrace to ATM")
    else:
        _entry_conds.append("• Neutral bias — enter when spot is within ±0.5 strike of ATM for optimal non-directional entry")

    _entry_conds.append(f"• Confirm entry: ATM IV should be {'stable or declining (good for sellers)' if _is_credit else 'stable or rising (good for buyers)'}")
    _entry_conds.append("• Best time: first 30–60 min or last 60 min of session (liquidity is highest, spreads are tighter)")
    _entry_conds.append("• Use LIMIT orders at mid of bid-ask. Never market-order multi-leg strategies.")

    # Stop logic — derive short strikes directly from leg_def if available,
    # otherwise fall back to OI walls (never brittle string-parse the leg description)
    if _is_credit:
        # Find the short strikes directly from leg_def built in Step 4
        _short_ce_strikes = [l[2] for l in _leg_def if l[0]=="SELL" and l[1]=="CE"] if _leg_def else []
        _short_pe_strikes = [l[2] for l in _leg_def if l[0]=="SELL" and l[1]=="PE"] if _leg_def else []
        if _short_ce_strikes and _short_pe_strikes:
            _stop_spot_desc = f"if spot breaches ₹{min(_short_pe_strikes):,.0f} (short put) or ₹{max(_short_ce_strikes):,.0f} (short call), begin closing immediately"
        elif _short_ce_strikes:
            _stop_spot_desc = f"if spot rallies above ₹{max(_short_ce_strikes):,.0f} (short call strike), begin closing"
        elif _short_pe_strikes:
            _stop_spot_desc = f"if spot falls below ₹{min(_short_pe_strikes):,.0f} (short put strike), begin closing"
        else:
            _stop_spot_desc = f"if spot breaks above call wall ₹{_call_wall:,.0f} or below put wall ₹{_put_wall:,.0f}, begin closing"
        _stop_rule   = "• Stop: close if unrealised loss reaches 1.5× to 2× the credit received per lot"
        _target_rule = "• Target: exit at 50% of maximum profit (50% of credit). Do not hold to expiry — gamma risk increases sharply in final 5 DTE"
        _stop_spot   = f"• Spot stop: {_stop_spot_desc}"
    else:
        _stop_rule   = "• Stop: if the position loses 40–50% of premium paid, exit. Do not average down."
        _target_rule = "• Target: exit when the position gains 80–100% of premium paid, or when your directional target is reached"
        _stop_spot   = f"• Spot stop: if spot breaks below EMA50 (₹{bias_res.get('e50',spot):,.0f}) for longs, or above EMA50 for shorts — exit"

    _adjust_rule  = f"• Adjustment trigger: if spot moves more than ±1 ATR (₹{_atr:.0f}) against you, re-evaluate the position"

    for _ec in _entry_conds:
        st.markdown(f"<div style='color:#e8e8e8;font-size:0.86rem;font-family:IBM Plex Mono,monospace;"
                    f"padding:3px 0;border-bottom:1px solid #111;'>{_ec}</div>", unsafe_allow_html=True)
    st.markdown(f"""
<div style="margin-top:10px;padding:10px 14px;border:1px solid #2a2a2a;background:#0d0d0d;
font-family:'IBM Plex Mono',monospace;font-size:0.84rem;line-height:2.0;color:#e8e8e8;">
  <span style="color:#ff3b3b;font-weight:600;">STOP LOSS:</span> {_stop_rule}<br>
  <span style="color:#00d084;font-weight:600;">TARGET:</span>    {_target_rule}<br>
  <span style="color:#ffb347;font-weight:600;">SPOT STOP:</span> {_stop_spot}<br>
  <span style="color:#9c27b0;font-weight:600;">ADJUST:</span>    {_adjust_rule}
</div>""", unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════
    # STEP 8 — WHAT TO WATCH AFTER ENTRY
    # ═══════════════════════════════════════════════════════
    _section(8, "WHAT TO MONITOR AFTER ENTRY", "#1e90ff")

    _gex_regime_short = "POSITIVE (range-bound)" if (oi_d and oi_d.get("net_gex",0) >= 0) else "NEGATIVE (trending)"

    st.markdown(f"""
<div style="background:#0d0d0d;border:1px solid #2a2a2a;padding:14px 18px;
font-family:'IBM Plex Mono',monospace;font-size:0.84rem;line-height:2.2;color:#e8e8e8;">

  <span style="color:#ffb347;font-weight:600;">IV CHANGE:</span>
  If IV spikes sharply after entry, {'your sold premium becomes more expensive to buy back — consider closing early if IV rises >20% from entry' if _is_credit else 'your bought options gain value — consider booking partial profits on the IV spike'}.<br>

  <span style="color:#ffb347;font-weight:600;">SPOT vs KEY LEVELS:</span>
  Watch ₹{_put_wall:,.0f} (Put Wall / support) and ₹{_call_wall:,.0f} (Call Wall / resistance).
  A break of either wall with volume = trend continuation. A rejection = reversal.<br>

  <span style="color:#ffb347;font-weight:600;">GAMMA FLIP:</span>
  Current GEX regime is {_gex_regime_short}.
  If spot crosses Gamma Flip ₹{oi_d.get('gamma_flip', spot):,.0f}, dealer hedging flips direction — vol can expand rapidly.
  {'Reduce short-gamma exposure before this level.' if _is_credit else 'This level can accelerate your directional move.'}<br>

  <span style="color:#ffb347;font-weight:600;">DTE COUNTDOWN:</span>
  At DTE ≤ 5, theta decay accelerates dramatically.
  {'For sellers: theta is working strongly for you — but gamma risk spikes. Consider taking profit early.' if _is_credit else 'For buyers: your options are losing value fastest now. If the move has not happened, exit.'}.<br>

  <span style="color:#ffb347;font-weight:600;">PCR SHIFT:</span>
  If PCR drops sharply (from >1.2 to <0.8), institutions are unwinding put protection — bullish signal.
  If PCR spikes sharply, fear is rising — bearish signal. Reload the engine to check.<br>

  <span style="color:#ffb347;font-weight:600;">MAX PAIN DRIFT:</span>
  Max pain currently ₹{_mp:,.0f}. As expiry approaches, spot tends to drift toward this level.
  {'If spot is far above max pain and you are short, risk increases.' if spot > _mp + step else
   'If spot is far below max pain and you are long, upward pull is likely.' if spot < _mp - step else
   'Spot is near max pain — pinning risk is elevated.'}

</div>""", unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════
    # STEP 9 — TRADE SUMMARY CARD (printable)
    # ═══════════════════════════════════════════════════════
    _section(9, "TRADE SUMMARY CARD", "#ff8c00")
    st.caption("Snapshot of the full plan. Reload after market moves to refresh all levels.")

    _best_name   = strat_recs[0]["Strategy"]  if strat_recs else "—"
    _best_legs   = strat_recs[0]["Legs"]      if strat_recs else "—"
    _best_risk   = strat_recs[0]["Max Risk"]  if strat_recs else "—"
    _best_reward = strat_recs[0]["Max Reward"] if strat_recs else "—"
    _best_dte    = strat_recs[0]["Ideal DTE"] if strat_recs else "—"
    _best_score  = strat_recs[0]["Score"]     if strat_recs else 0
    _best_ev     = strat_recs[0].get("ev", 0)     if strat_recs else 0
    _best_pop    = strat_recs[0].get("pop", 0.5)  if strat_recs else 0.5
    _best_kelly  = strat_recs[0].get("kelly", 0)  if strat_recs else 0

    st.markdown(f"""
<div style="background:#0d0d0d;border:2px solid #ff8c00;padding:20px 24px;
font-family:'IBM Plex Mono',monospace;margin-top:8px;">
  <div style="color:#ff8c00;font-size:0.78rem;letter-spacing:.15em;font-weight:700;
  border-bottom:1px solid #ff8c00;padding-bottom:8px;margin-bottom:14px;">
    ⚡ MONARCH TRADE BRIEF — {sym} — {expiry}
  </div>
  <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;margin-bottom:14px;">
    <div>
      <div style="color:#555;font-size:0.76rem;">STRATEGY</div>
      <div style="color:#e8e8e8;font-size:0.92rem;font-weight:700;">{_best_name}</div>
    </div>
    <div>
      <div style="color:#555;font-size:0.76rem;">DIRECTION</div>
      <div style="color:{bc};font-size:0.92rem;font-weight:700;">{bias} ({bias_score:+d})</div>
    </div>
    <div>
      <div style="color:#555;font-size:0.76rem;">VOL REGIME</div>
      <div style="color:{v_col};font-size:0.92rem;font-weight:700;">{v_lbl}  IVR {ivr:.0f}</div>
    </div>
    <div>
      <div style="color:#555;font-size:0.76rem;">SPOT / ATM</div>
      <div style="color:#e8e8e8;font-size:0.92rem;">₹{spot:,.0f}  /  ₹{atm_k:,.0f}</div>
    </div>
    <div>
      <div style="color:#555;font-size:0.76rem;">SUPPORT</div>
      <div style="color:#00d084;font-size:0.92rem;">₹{_put_wall:,.0f}  (put wall)</div>
    </div>
    <div>
      <div style="color:#555;font-size:0.76rem;">RESISTANCE</div>
      <div style="color:#ff3b3b;font-size:0.92rem;">₹{_call_wall:,.0f}  (call wall)</div>
    </div>
  </div>
  <div style="color:#7ec8e3;font-size:0.84rem;margin-bottom:10px;">
    LEGS: {_best_legs}
  </div>
  <div style="display:flex;gap:24px;flex-wrap:wrap;font-size:0.82rem;">
    <span style="color:#ff3b3b;">MAX RISK: {_best_risk}</span>
    <span style="color:#00d084;">MAX REWARD: {_best_reward}</span>
    <span style="color:#ffb347;">IDEAL DTE: {_best_dte}</span>
    <span style="color:#00d084;">EV: {'+' if _best_ev>=0 else ''}₹{_best_ev:,.0f}</span>
    <span style="color:#7ec8e3;">POP: {_best_pop*100:.1f}%</span>
    <span style="color:#9c27b0;">KELLY: {_best_kelly*100:.1f}%</span>
    <span style="color:#888;">P(↑): {prob_score['prob_up']*100:.1f}%  P(↓): {prob_score['prob_down']*100:.1f}%</span>
    <span style="color:#555;">Γ FLIP: ₹{oi_d.get('gamma_flip', spot):,.0f}</span>
    <span style="color:#555;">MAX PAIN: ₹{_mp:,.0f}</span>
  </div>
  <div style="margin-top:12px;color:#ff3b3b;font-size:0.82rem;line-height:1.7;">
    ⚠ This is a decision-support engine. Always verify premiums with live bid/ask before placing.
    Past signals do not guarantee future results. Use defined-risk structures until consistently profitable.
  </div>
</div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# TAB 9 — TERM STRUCTURE
# ══════════════════════════════════════════════════════════════
with t_ts:
    st.markdown("### 📅 Multi-Expiry Term Structure")

    _me = st.session_state.opt_multi_expiry

    if not st.session_state.opt_multi_loaded or not _me:
        st.info("Use **📅 LOAD TERM STRUCTURE** in the sidebar to fetch multiple expiries.")
        st.caption("First load single-expiry intel (⚡ LOAD OPTIONS INTEL), then use the Term Structure "
                   "loader to fetch 2–5 expiries simultaneously for cross-tenor analysis.")
    else:
        _spot_ts  = st.session_state.opt_spot
        _hv_ts    = st.session_state.opt_hv20
        _sym_ts   = st.session_state.opt_symbol
        _step_ts  = st.session_state.opt_step
        _r_ts     = st.session_state.get("opt_rfr", 0.065)
        _q_ts     = st.session_state.opt_div_yield.get(_sym_ts.upper(), 0.0)

        # ── Metrics Row ─────────────────────────────────────────
        _mcols = st.columns(len(_me))
        _colors_ts = ["#ff8c00","#1e90ff","#00d084","#9c27b0","#ff3b3b"]
        for _ci, _ed in enumerate(_me):
            _cc = _colors_ts[_ci % len(_colors_ts)]
            _mcols[_ci].markdown(f"""
<div style="background:#0d0d0d;border:1px solid #2a2a2a;border-top:3px solid {_cc};
padding:10px 12px;font-family:'IBM Plex Mono',monospace;">
  <div style="color:#555;font-size:0.78rem;letter-spacing:.08em;">{_ed['expiry']}</div>
  <div style="color:{_cc};font-size:1.1rem;font-weight:700;">{_ed['atm_iv_pct']:.1f}%</div>
  <div style="color:#888;font-size:0.80rem;">DTE {_ed['dte']}  ·  PCR {_ed['pcr']:.2f}</div>
  <div style="color:#e8e8e8;font-size:0.80rem;">Straddle ₹{_ed['straddle']:.0f}  ±{_ed['exp_move_pct']:.1f}%</div>
  <div style="color:#555;font-size:0.78rem;">FwdVol {_ed.get('fwd_vol_pct',_ed['atm_iv_pct']):.1f}%</div>
</div>""", unsafe_allow_html=True)

        st.divider()

        # ── Chart 1: IV Term Structure Curve ────────────────────
        _ts_dtes  = [d["dte"]         for d in _me]
        _ts_ivs   = [d["atm_iv_pct"]  for d in _me]
        _ts_fvs   = [d.get("fwd_vol_pct", d["atm_iv_pct"]) for d in _me]
        _ts_exps  = [d["expiry"]       for d in _me]

        fig_ts = go.Figure()
        fig_ts.add_trace(go.Scatter(
            x=_ts_dtes, y=_ts_ivs,
            mode="lines+markers+text",
            name="ATM IV",
            line=dict(color="#ff8c00", width=2.5),
            marker=dict(size=9, color="#ff8c00"),
            text=[f"{v:.1f}%" for v in _ts_ivs],
            textposition="top center",
            textfont=dict(size=9, color="#ff8c00"),
        ))
        fig_ts.add_trace(go.Scatter(
            x=_ts_dtes, y=_ts_fvs,
            mode="lines+markers",
            name="Forward Vol (between tenors)",
            line=dict(color="#1e90ff", width=1.5, dash="dash"),
            marker=dict(size=7, color="#1e90ff", symbol="diamond"),
        ))
        if _hv_ts:
            fig_ts.add_hline(
                y=_hv_ts * 100,
                line=dict(color="#00d084", dash="dot", width=1.5),
                annotation_text=f"HV20 {_hv_ts*100:.1f}%",
                annotation_font=dict(color="#00d084", size=9)
            )
        fig_ts.update_layout(
            title="IV Term Structure — ATM Implied Vol by Expiry",
            height=320, plot_bgcolor="#000", paper_bgcolor="#000",
            font=dict(color="#e8e8e8", family="IBM Plex Mono", size=9),
            xaxis=dict(title="DTE (trading days)", gridcolor="#111"),
            yaxis=dict(title="IV %", gridcolor="#111"),
            legend=dict(orientation="h", y=1.12),
            margin=dict(t=50, b=10),
        )
        st.plotly_chart(fig_ts, use_container_width=True)

        # ── Chart 2: IV Term Structure Shape Annotation ─────────
        _shape_analysis = ""
        if len(_me) >= 2:
            _slope = _me[-1]["atm_iv_pct"] - _me[0]["atm_iv_pct"]
            if _slope > 1.5:
                _shape_c = "#00d084"
                _shape_analysis = (f"CONTANGO (+{_slope:.1f}pp) — Far-month IV > near-month IV. "
                                   "Normal structure. Selling near-term and buying back-month is adverse carry. "
                                   "Calendar spread less attractive.")
            elif _slope < -1.5:
                _shape_c = "#ff3b3b"
                _shape_analysis = (f"BACKWARDATION ({_slope:.1f}pp) — Near-month IV > far-month IV. "
                                   "Inverted structure — often around events (earnings, budget, RBI). "
                                   "Calendar spread ATTRACTIVE: sell expensive near-term, buy cheap back-month.")
            else:
                _shape_c = "#ffb347"
                _shape_analysis = (f"FLAT ({_slope:+.1f}pp) — Term structure near flat. "
                                   "No strong carry edge in either direction. Use straddles or condors.")
            st.markdown(f"""
<div style="border-left:3px solid {_shape_c};padding:8px 14px;
font-family:'IBM Plex Mono',monospace;font-size:0.86rem;color:{_shape_c};margin:4px 0 12px;">
  TERM STRUCTURE: {_shape_analysis}
</div>""", unsafe_allow_html=True)

        # ── Chart 3: Forward Volatility Table ───────────────────
        st.divider()
        st.markdown("#### Forward Volatility Between Tenors")
        st.caption("σ_fwd(T₁→T₂) = √[(σ₂²·T₂ − σ₁²·T₁) / (T₂ − T₁)]  —  the vol the market implies for the period BETWEEN two expiries.")

        _fwd_rows = []
        for _i in range(1, len(_me)):
            _prev, _curr = _me[_i-1], _me[_i]
            _fv = _curr.get("fwd_vol_pct", _curr["atm_iv_pct"])
            # Carry = forward_vol - far_spot_vol
            # carry > 0: forward vol > far spot vol → market prices the FORWARD period richer
            #            than the far tenor's overall vol → far option is CHEAP → BUY far (calendar)
            # carry < 0: forward vol < far spot vol → far option is EXPENSIVE → avoid/sell far
            # The original code had this BACKWARDS ("FWD RICH" mapped to "sell far").
            _carry = _fv - _curr["atm_iv_pct"]
            _fwd_rows.append({
                "Period":      f"{_prev['expiry']}  →  {_curr['expiry']}",
                "From DTE":    _prev["dte"],
                "To DTE":      _curr["dte"],
                "Near IV%":    f"{_prev['atm_iv_pct']:.2f}%",
                "Far IV%":     f"{_curr['atm_iv_pct']:.2f}%",
                "Forward Vol%":f"{_fv:.2f}%",
                "Carry (pp)":  f"{_carry:+.2f}",
                # carry>0 → fwd>far → far is cheap → BUY far = calendar favoured
                # carry<0 → fwd<far → far is expensive → avoid calendar / sell far
                "Signal":      ("BUY FAR — calendar favoured (fwd > far spot)" if _carry > 1.0
                                else "SELL FAR — reverse calendar (fwd < far spot)" if _carry < -1.0
                                else "NEUTRAL — no clear edge"),
            })
        if _fwd_rows:
            st.dataframe(pd.DataFrame(_fwd_rows), use_container_width=True, hide_index=True)

        # ── Chart 4: Calendar Spread Opportunities ──────────────
        st.divider()
        st.markdown("#### Calendar Spread Opportunities")
        st.caption("For each adjacent pair of expiries — theoretical debit, theta advantage, and entry condition.")

        _cal_rows = []
        for _i in range(1, len(_me)):
            _near  = _me[_i-1]
            _far   = _me[_i]
            _T_n   = _near["T"]
            _T_f   = _far["T"]
            _iv_n  = _near["atm_iv"]
            _iv_f  = _far["atm_iv"]
            _atm_k_ts = atm_strike(_spot_ts, _step_ts)

            # BS price of near and far ATM call
            _p_near = bs_price(_spot_ts, _atm_k_ts, _T_n, _r_ts, _iv_n, "call", _q_ts)
            _p_far  = bs_price(_spot_ts, _atm_k_ts, _T_f, _r_ts, _iv_f, "call", _q_ts)
            # Net cash flow = far - near: positive = debit (normal), negative = credit (backwardation)
            _net_flow = _p_far - _p_near
            _debit_str = (f"₹{_net_flow:.2f} debit" if _net_flow >= 0
                          else f"₹{abs(_net_flow):.2f} CREDIT (backwardation)")

            # Greeks of each leg
            _g_near = bs_greeks(_spot_ts, _atm_k_ts, _T_n, _r_ts, _iv_n, "call", _q_ts)
            _g_far  = bs_greeks(_spot_ts, _atm_k_ts, _T_f, _r_ts, _iv_f, "call", _q_ts)
            _net_theta = _g_far["theta"] - _g_near["theta"]  # net daily decay (want near to decay faster)
            _net_vega  = _g_far["vega"]  - _g_near["vega"]   # net vega (want to be long back-month vega)

            # Entry condition
            _entry_ok   = _iv_n > _iv_f   # backwardation = favourable
            _entry_lbl  = "✓ FAVOURABLE" if _entry_ok else "✗ ADVERSE carry"
            _entry_c    = "#00d084" if _entry_ok else "#ff3b3b"

            # Breakeven range at near-expiry: spot must stay within ±straddle_near
            _be_width = _near["straddle"]

            _cal_rows.append({
                "Calendar":       f"SELL {_near['expiry']}  /  BUY {_far['expiry']}",
                "Net Cash Flow":  _debit_str,
                "Near IV%":       f"{_iv_n*100:.2f}%",
                "Far IV%":        f"{_iv_f*100:.2f}%",
                "Net θ/day":      f"{_net_theta:.3f}",
                "Net Vega/1%":    f"{_net_vega:.3f}",
                "Profit Zone ±":  f"₹{_be_width:.0f}",
                "Entry":          _entry_lbl,
            })

        if _cal_rows:
            _cdf_display = pd.DataFrame(_cal_rows)

            def _cal_style(v):
                if "FAVOURABLE" in str(v): return "color:#00d084;font-weight:700"
                if "ADVERSE"    in str(v): return "color:#ff3b3b;font-weight:700"
                return ""

            st.dataframe(
                _cdf_display.style.map(_cal_style, subset=["Entry"]),
                use_container_width=True, hide_index=True
            )

        # ── Chart 5: Volatility Smile Overlay ───────────────────
        st.divider()
        st.markdown("#### Volatility Smile Across Expiries")
        st.caption("IV smile by strike for each loaded expiry — shows skew evolution as tenor increases.")

        fig_smile_ts = go.Figure()
        _smile_colors = ["#ff8c00","#1e90ff","#00d084","#9c27b0","#ff3b3b"]
        for _ci, _ed in enumerate(_me):
            if _ed["smile"]:
                _sdf = pd.DataFrame(_ed["smile"])
                # Dynamic chart range: ±3 MC expected moves, floored at ±2× ATR
                _em_ts    = (st.session_state.get("opt_prob_score") or {}).get("mc_expected_move", _atr_seed(_spot_ts) * 5)
                _floor_ts = max(_atr_seed(_spot_ts) * 2, _spot_ts * 0.02)  # 2× ATR or 2% floor
                _range_ts = max(3.0 * _em_ts, _floor_ts)
                _sdf = _sdf[(_sdf.Strike >= _spot_ts - _range_ts) & (_sdf.Strike <= _spot_ts + _range_ts)]
                if not _sdf.empty:
                    _cc = _smile_colors[_ci % len(_smile_colors)]
                    fig_smile_ts.add_trace(go.Scatter(
                        x=_sdf.Strike, y=_sdf.CE_IV,
                        mode="lines", name=f"CE {_ed['expiry']} (DTE {_ed['dte']})",
                        line=dict(color=_cc, width=1.8),
                    ))
                    fig_smile_ts.add_trace(go.Scatter(
                        x=_sdf.Strike, y=_sdf.PE_IV,
                        mode="lines", name=f"PE {_ed['expiry']}",
                        line=dict(color=_cc, width=1.2, dash="dot"),
                    ))

        fig_smile_ts.add_vline(x=_spot_ts,
                               line=dict(color="#ffb347", dash="dot", width=1.5),
                               annotation_text=f"Spot ₹{_spot_ts:.0f}")
        if _hv_ts:
            fig_smile_ts.add_hline(y=_hv_ts*100,
                                   line=dict(color="#444", dash="dot", width=1),
                                   annotation_text=f"HV20 {_hv_ts*100:.1f}%",
                                   annotation_font=dict(size=8))
        fig_smile_ts.update_layout(
            title="Volatility Smile by Strike — Multi-Expiry Overlay",
            height=340, plot_bgcolor="#000", paper_bgcolor="#000",
            font=dict(color="#e8e8e8", family="IBM Plex Mono", size=9),
            xaxis=dict(title="Strike", gridcolor="#111"),
            yaxis=dict(title="IV %", gridcolor="#111"),
            legend=dict(orientation="h", y=1.12, font=dict(size=8)),
            margin=dict(t=50, b=10),
        )
        st.plotly_chart(fig_smile_ts, use_container_width=True)

        # ── Chart 6: OI Distribution Across Expiries ────────────
        st.divider()
        st.markdown("#### OI Distribution Across Expiries")
        st.caption("Total CE and PE OI per expiry — shows where market interest is concentrated.")

        fig_oi_ts = go.Figure()
        fig_oi_ts.add_trace(go.Bar(
            x=_ts_exps,
            y=[d["total_ce_oi"] / 1e5 for d in _me],
            name="CE OI (lac)", marker_color="#ff3b3b", opacity=0.8
        ))
        fig_oi_ts.add_trace(go.Bar(
            x=_ts_exps,
            y=[d["total_pe_oi"] / 1e5 for d in _me],
            name="PE OI (lac)", marker_color="#00d084", opacity=0.8
        ))
        fig_oi_ts.update_layout(
            title="Total OI per Expiry — Call vs Put",
            height=260, barmode="group", plot_bgcolor="#000", paper_bgcolor="#000",
            font=dict(color="#e8e8e8", family="IBM Plex Mono", size=9),
            yaxis=dict(title="OI (lac)", gridcolor="#111"),
            legend=dict(orientation="h", y=1.12),
            margin=dict(t=40, b=10),
        )
        st.plotly_chart(fig_oi_ts, use_container_width=True)

        # PCR across expiries
        fig_pcr_ts = go.Figure()
        fig_pcr_ts.add_trace(go.Bar(
            x=_ts_exps,
            y=[d["pcr"] for d in _me],
            marker_color=["#00d084" if d["pcr"] > 1.1 else "#ff3b3b" if d["pcr"] < 0.9 else "#ffb347"
                          for d in _me],
            name="PCR"
        ))
        fig_pcr_ts.add_hline(y=1.0, line=dict(color="#444", dash="dot"),
                             annotation_text="PCR = 1.0 (neutral)")
        fig_pcr_ts.update_layout(
            title="Put-Call Ratio by Expiry",
            height=200, plot_bgcolor="#000", paper_bgcolor="#000",
            font=dict(color="#e8e8e8", family="IBM Plex Mono", size=9),
            yaxis=dict(title="PCR", gridcolor="#111"),
            margin=dict(t=40, b=10),
        )
        st.plotly_chart(fig_pcr_ts, use_container_width=True)

        # ── Summary Table ────────────────────────────────────────
        st.divider()
        st.markdown("#### Full Term Structure Summary")
        _sum_rows = []
        for _ed in _me:
            _iv_vs_hv = _ed["atm_iv_pct"] - (_hv_ts or CFG["hv_fallback"]) * 100
            _regime   = ("LOW VOL"    if _ed["atm_iv_pct"] < 15 else
                         "NORMAL"     if _ed["atm_iv_pct"] < 25 else
                         "ELEVATED"   if _ed["atm_iv_pct"] < 35 else "HIGH VOL")
            _sum_rows.append({
                "Expiry":         _ed["expiry"],
                "DTE":            _ed["dte"],
                "ATM IV%":        f"{_ed['atm_iv_pct']:.2f}%",
                "Fwd Vol%":       f"{_ed.get('fwd_vol_pct', _ed['atm_iv_pct']):.2f}%",
                "IV vs HV (pp)":  f"{_iv_vs_hv:+.2f}",
                "Straddle ₹":     f"₹{_ed['straddle']:.0f}",
                "Exp Move ±%":    f"±{_ed['exp_move_pct']:.2f}%",
                "PCR":            f"{_ed['pcr']:.3f}",
                "CE OI (lac)":    f"{_ed['total_ce_oi']/1e5:.1f}",
                "PE OI (lac)":    f"{_ed['total_pe_oi']/1e5:.1f}",
                "Regime":         _regime,
            })
        st.dataframe(pd.DataFrame(_sum_rows), use_container_width=True, hide_index=True)

        with st.expander("◼ HOW TO READ TERM STRUCTURE"):
            st.markdown("""
**IV Term Structure** shows how implied volatility changes with expiry date.

| Shape | Meaning | Strategy Edge |
|-------|---------|---------------|
| **Contango** (near IV < far IV) | Normal — uncertainty grows over time | No calendar edge. Use single-expiry condors/spreads |
| **Backwardation** (near IV > far IV) | Inverted — near-term event risk elevated | Calendar spread ATTRACTIVE — sell expensive near, buy cheap far |
| **Flat** | Balanced | No carry advantage |

**Forward Volatility** is the vol the market implies for the period *between* two expiries:
σ_fwd(T₁→T₂) = √[(σ₂²·T₂ − σ₁²·T₁) / (T₂ − T₁)]

If forward vol > far-month spot vol → the market expects vol to *increase* after the near expiry (event risk).
If forward vol < far-month spot vol → the market expects vol to *decrease* → possible mean-reversion.

**Calendar Spread entry rule:** Only enter when near-month IV > far-month IV (backwardation).
You want to sell expensive near-term theta and buy cheap back-month vega.

**OI concentration across expiries** tells you where institutional positioning is heaviest.
The expiry with the largest OI is the "pinning" expiry — max pain gravity is strongest there.
""")

# ══════════════════════════════════════════════════════════════
# TAB 10 — FLOW & SKEW
# ══════════════════════════════════════════════════════════════
with t_flow:
    st.markdown("### 🌊 Options Flow, Skew & Liquidity")

    # ── IV Percentile vs IV Rank comparison ─────────────────
    st.markdown("#### IV Rank vs IV Percentile")
    _ivr_col  = "#ff3b3b" if ivr >= 75 else ("#ffb347" if ivr >= 50 else "#1e90ff")
    _ivp_col  = "#ff3b3b" if iv_pct >= 75 else ("#ffb347" if iv_pct >= 50 else "#1e90ff")
    fc1, fc2, fc3 = st.columns(3)
    fc1.markdown(f"""<div style="background:#0d0d0d;border:1px solid #2a2a2a;
border-left:4px solid {_ivr_col};padding:12px 16px;font-family:'IBM Plex Mono',monospace;">
  <div style="color:#555;font-size:0.76rem;letter-spacing:.08em;">IV RANK (range-normalised)</div>
  <div style="color:{_ivr_col};font-size:1.5rem;font-weight:700;">{ivr:.1f}</div>
  <div style="color:#888;font-size:0.78rem;">
    (IV − min) / (max − min) × 100<br>Sensitive to outlier spikes in history
  </div>
</div>""", unsafe_allow_html=True)
    fc2.markdown(f"""<div style="background:#0d0d0d;border:1px solid #2a2a2a;
border-left:4px solid {_ivp_col};padding:12px 16px;font-family:'IBM Plex Mono',monospace;">
  <div style="color:#555;font-size:0.76rem;letter-spacing:.08em;">IV PERCENTILE (robust)</div>
  <div style="color:{_ivp_col};font-size:1.5rem;font-weight:700;">{iv_pct:.1f}</div>
  <div style="color:#888;font-size:0.78rem;">
    % of past IVs below current IV<br>Outlier-robust — preferred for signal quality
  </div>
</div>""", unsafe_allow_html=True)
    fc3.markdown(f"""<div style="background:#0d0d0d;border:1px solid #2a2a2a;
border-left:4px solid #888;padding:12px 16px;font-family:'IBM Plex Mono',monospace;">
  <div style="color:#555;font-size:0.76rem;letter-spacing:.08em;">IV HISTORY DEPTH</div>
  <div style="color:#e8e8e8;font-size:1.5rem;font-weight:700;">{len(_iv_hist_sym_flow)}</div>
  <div style="color:#888;font-size:0.78rem;">
    observations stored<br>Min 3 needed for rank/percentile
  </div>
</div>""", unsafe_allow_html=True)

    # ── IV History chart ──────────────────────────────────────
    if len(_iv_hist_sym_flow) >= 5:
        st.divider()
        fig_iv_hist = go.Figure()
        _iv_x = list(range(len(_iv_hist_sym_flow)))
        fig_iv_hist.add_trace(go.Scatter(
            x=_iv_x, y=[v * 100 for v in _iv_hist_sym_flow],
            mode="lines", name="ATM IV History",
            line=dict(color="#ff8c00", width=1.8),
            fill="tozeroy", fillcolor="rgba(255,140,0,0.05)",
        ))
        fig_iv_hist.add_hline(y=atm_iv * 100,
                               line=dict(color="#ffb347", dash="dot", width=1.5),
                               annotation_text=f"Current {atm_iv*100:.1f}%")
        if hv20:
            fig_iv_hist.add_hline(y=hv20 * 100,
                                   line=dict(color="#00d084", dash="dot", width=1),
                                   annotation_text=f"HV20 {hv20*100:.1f}%")
        fig_iv_hist.update_layout(
            title=f"{sym} — ATM IV History (session observations)",
            height=230, plot_bgcolor="#000", paper_bgcolor="#000",
            font=dict(color="#e8e8e8", family="IBM Plex Mono", size=9),
            xaxis=dict(title="Observation #", gridcolor="#111"),
            yaxis=dict(title="IV %", gridcolor="#111"),
            margin=dict(t=40, b=10),
        )
        st.plotly_chart(fig_iv_hist, use_container_width=True)

    # ── Vol Term Structure Slope ──────────────────────────────
    st.divider()
    st.markdown("#### Volatility Term Structure Slope")
    _me_flow = st.session_state.opt_multi_expiry
    if _me_flow and len(_me_flow) >= 2:
        _ts_slope_pp = _me_flow[-1]["atm_iv_pct"] - _me_flow[0]["atm_iv_pct"]
        _ts_slope_per_dte = round(_ts_slope_pp / max(_me_flow[-1]["dte"] - _me_flow[0]["dte"], 1), 3)
        _slope_col  = "#ff3b3b" if _ts_slope_pp < -1.5 else ("#00d084" if _ts_slope_pp > 1.5 else "#ffb347")
        _slope_lbl  = ("BACKWARDATION — near > far IV (inverted)" if _ts_slope_pp < -1.5
                       else "CONTANGO — far > near IV (normal)" if _ts_slope_pp > 1.5
                       else "FLAT — near ≈ far IV")
        st.markdown(f"""<div style="background:#0d0d0d;border:1px solid #2a2a2a;
border-left:4px solid {_slope_col};padding:12px 16px;font-family:'IBM Plex Mono',monospace;margin-bottom:10px;">
  <div style="color:#555;font-size:0.76rem;letter-spacing:.08em;">TERM STRUCTURE SLOPE</div>
  <div style="color:{_slope_col};font-size:1.2rem;font-weight:700;">{_ts_slope_pp:+.2f}pp  ·  {_slope_lbl}</div>
  <div style="color:#888;font-size:0.80rem;">
    Near: {_me_flow[0]['atm_iv_pct']:.1f}% (DTE {_me_flow[0]['dte']})  →  
    Far: {_me_flow[-1]['atm_iv_pct']:.1f}% (DTE {_me_flow[-1]['dte']})  ·  
    Slope: {_ts_slope_per_dte:+.3f}pp/DTE
  </div>
</div>""", unsafe_allow_html=True)

        # Slope interpretation
        _slope_interp = {
            "BACKWARDATION": "Near-term fear is elevated. Calendar spread is attractive: "
                             "sell expensive near-term, buy cheap far-term. "
                             "Watch for event risk in the near expiry.",
            "CONTANGO":      "Normal structure. Far-month uncertainty priced higher. "
                             "Calendar spread has adverse carry — avoid unless there is a specific near-term catalyst.",
            "FLAT":          "No clear carry edge in either calendar direction. "
                             "Use outright straddles or iron condors rather than time-spread plays.",
        }
        _slope_key  = "BACKWARDATION" if _ts_slope_pp < -1.5 else ("CONTANGO" if _ts_slope_pp > 1.5 else "FLAT")
        st.caption(_slope_interp[_slope_key])
    else:
        st.info("Load Term Structure (📅 LOAD TERM STRUCTURE in sidebar) to see slope analysis.")

    # ── Full Put/Call IV Skew Curve ───────────────────────────
    st.divider()
    st.markdown("#### Put/Call IV Skew Curve — Full Strike Range")
    st.caption("Shows IV vs moneyness for all strikes. Steep left wing = heavy put buying (crash fear). "
               "Flat or inverted = complacent or call-skewed market.")

    _skew_curve = oi_d.get("skew_curve", []) if oi_d else []
    if _skew_curve:
        _sdf = pd.DataFrame(_skew_curve)
        _sdf = _sdf.sort_values("moneyness")

        fig_skew = go.Figure()
        fig_skew.add_trace(go.Scatter(
            x=_sdf["moneyness"], y=_sdf["put_iv"],
            mode="lines+markers", name="Put IV",
            line=dict(color="#ff3b3b", width=2),
            marker=dict(size=5),
        ))
        fig_skew.add_trace(go.Scatter(
            x=_sdf["moneyness"], y=_sdf["call_iv"],
            mode="lines+markers", name="Call IV",
            line=dict(color="#1e90ff", width=2),
            marker=dict(size=5),
        ))
        fig_skew.add_trace(go.Scatter(
            x=_sdf["moneyness"], y=_sdf["skew_pp"],
            mode="lines", name="Skew (Put−Call pp)",
            line=dict(color="#ff8c00", width=1.5, dash="dot"),
            yaxis="y2",
        ))
        if hv20:
            fig_skew.add_hline(y=hv20 * 100,
                                line=dict(color="#00d084", dash="dot", width=1),
                                annotation_text=f"HV20 {hv20*100:.1f}%")
        fig_skew.add_vline(x=0, line=dict(color="#ffb347", dash="dot", width=1.5),
                           annotation_text="ATM")
        fig_skew.update_layout(
            title="IV Skew Curve — Put & Call IV by Moneyness",
            height=360, plot_bgcolor="#000", paper_bgcolor="#000",
            font=dict(color="#e8e8e8", family="IBM Plex Mono", size=9),
            xaxis=dict(title="Moneyness % (−ve = OTM put, +ve = OTM call)", gridcolor="#111"),
            yaxis=dict(title="IV %", gridcolor="#111"),
            yaxis2=dict(title="Skew (pp)", overlaying="y", side="right",
                        gridcolor="#1a1a1a", zeroline=True, zerolinecolor="#2a2a2a"),
            legend=dict(orientation="h", y=1.12),
            margin=dict(t=50, b=10),
        )
        st.plotly_chart(fig_skew, use_container_width=True)

        # Skew table
        st.dataframe(_sdf[["strike","moneyness","call_iv","put_iv","skew_pp"]].rename(columns={
            "strike": "Strike", "moneyness": "Moneyness%",
            "call_iv": "Call IV%", "put_iv": "Put IV%", "skew_pp": "Skew(pp)"
        }), use_container_width=True, hide_index=True)
    else:
        st.info("Live chain data needed for skew curve. Load option chain (⚡ LOAD OPTIONS INTEL).")

    # ── Liquidity Filter ─────────────────────────────────────
    st.divider()
    st.markdown("#### Liquidity Filter — Bid-Ask Spread, OI & Volume")
    _liq_verdict = liquidity_d.get("verdict", "—")
    _liq_col     = liquidity_d.get("color", "#888")
    _liq_sc      = liquidity_d.get("liquid_score", 0)
    _liq_spr     = liquidity_d.get("atm_spread_pct", 0)

    lq1, lq2, lq3, lq4 = st.columns(4)
    lq1.metric("Liquidity Score",  f"{_liq_sc}/100")
    lq2.metric("ATM Spread %",     f"{_liq_spr:.1f}%")
    lq3.metric("ATM OI Adequate",  "✓" if liquidity_d.get("atm_oi_ok") else "⚠")
    lq4.metric("ATM Vol Adequate", "✓" if liquidity_d.get("atm_vol_ok") else "⚠")

    st.markdown(f"""<div style="border-left:4px solid {_liq_col};padding:8px 14px;
font-family:'IBM Plex Mono',monospace;font-size:0.86rem;color:{_liq_col};margin:8px 0;">
  {_liq_verdict}
</div>""", unsafe_allow_html=True)

    _liq_rows = liquidity_d.get("rows", [])
    if _liq_rows:
        _ldf = pd.DataFrame(_liq_rows)
        def _liq_ok_style(v):
            if v == "✓": return "color:#00d084;font-weight:700"
            if v == "⚠": return "color:#ffb347;font-weight:700"
            return ""
        def _spread_style(v):
            try:
                fv = float(v)
                if fv < 3:   return "color:#00d084"
                if fv < 8:   return "color:#ffb347"
                return "color:#ff3b3b"
            except: return ""
        st.dataframe(
            _ldf.style
                .map(_liq_ok_style, subset=["CE OK","PE OK"])
                .map(_spread_style, subset=["CE Spread%","PE Spread%"]),
            use_container_width=True, hide_index=True
        )
        st.caption("✓ = spread <5% AND OI ≥1000 AND volume ≥100. ⚠ = fails one or more criteria. "
                   "Use LIMIT orders at bid-ask mid. Never market-order multi-leg strategies.")

# ══════════════════════════════════════════════════════════════
# TAB 11 — REGIME & EVENTS
# ══════════════════════════════════════════════════════════════
with t_regime:
    st.markdown("### 🔭 Market Regime Detection & Event Calendar")

    # ── Regime Panel ─────────────────────────────────────────
    st.markdown("#### Market Regime")
    _rc = regime_d.get("color", "#888")
    _rl = regime_d.get("regime", "—")

    rc1, rc2, rc3, rc4, rc5 = st.columns(5)
    rc1.metric("Regime",       _rl)
    rc2.metric("Trend",        regime_d.get("trend", "—"))
    rc3.metric("Vol State",    regime_d.get("vol", "—"))
    rc4.metric("ADX(14)",      f"{regime_d.get('adx', 0):.1f}")
    rc5.metric("BB Width%",    f"{regime_d.get('bb_width_pct', 0):.2f}%")

    # Regime strategy guide
    _REGIME_GUIDE = {
        "TRENDING HIGH VOL": {
            "best":   ["Bear/Bull Spread", "Long Call/Put", "Debit Spread"],
            "avoid":  ["Iron Condor", "Short Straddle", "Short Strangle"],
            "note":   "Trend + high IV = directional debit spreads (defined risk). "
                      "Avoid credit strategies — dealers are short gamma, amplifying moves. "
                      "Use smaller size and wider stops.",
        },
        "TRENDING LOW VOL": {
            "best":   ["Long ATM Call/Put", "Bull/Bear Call/Put Spread", "Calendar"],
            "avoid":  ["Short Straddle", "Iron Condor"],
            "note":   "Low IV = buy premium cheaply and ride the trend. "
                      "Debit spreads maximise directional capture per rupee of premium. "
                      "Calendar spreads work well — back-month IV cheap relative to realised.",
        },
        "RANGE LOW VOL": {
            "best":   ["Short Straddle (near expiry)", "Iron Condor", "Short Strangle"],
            "avoid":  ["Long Call/Put", "Long Straddle"],
            "note":   "Range + low IV = ideal credit environment. "
                      "Spot pinned, theta working for sellers, IV not rich enough to justify debit. "
                      "Use iron condors with strikes at OI walls.",
        },
        "RANGE HIGH VOL": {
            "best":   ["Iron Condor", "Short Straddle", "Bull Put / Bear Call Spread"],
            "avoid":  ["Long Straddle", "Long Strangle (without hedge)"],
            "note":   "IV is elevated but range-bound structure means big moves unlikely. "
                      "Sell premium with defined risk (condors > naked straddle). "
                      "Take profit early — IV mean-reversion likely.",
        },
        "TRANSITIONAL": {
            "best":   ["Iron Condor (conservative)", "Calendar"],
            "avoid":  ["Naked shorts", "large directional debit"],
            "note":   "Market in transition — no clear structural edge. Reduce size, "
                      "widen strikes, and wait for regime clarity.",
        },
    }
    _guide = _REGIME_GUIDE.get(_rl, {"best": ["—"], "avoid": ["—"], "note": "Regime not classified."})

    rg1, rg2 = st.columns(2)
    with rg1:
        st.markdown(f"""<div style="background:#0d1a00;border:1px solid #00d084;
border-left:4px solid #00d084;padding:12px 16px;font-family:'IBM Plex Mono',monospace;">
  <div style="color:#00d084;font-size:0.80rem;font-weight:600;letter-spacing:.08em;margin-bottom:8px;">
    ✓ STRATEGIES SUITED TO THIS REGIME
  </div>
  {"".join(f'<div style="color:#e8e8e8;font-size:0.86rem;padding:2px 0;">• {s}</div>' for s in _guide["best"])}
</div>""", unsafe_allow_html=True)
    with rg2:
        st.markdown(f"""<div style="background:#1a0000;border:1px solid #ff3b3b;
border-left:4px solid #ff3b3b;padding:12px 16px;font-family:'IBM Plex Mono',monospace;">
  <div style="color:#ff3b3b;font-size:0.80rem;font-weight:600;letter-spacing:.08em;margin-bottom:8px;">
    ✗ AVOID IN THIS REGIME
  </div>
  {"".join(f'<div style="color:#e8e8e8;font-size:0.86rem;padding:2px 0;">• {s}</div>' for s in _guide["avoid"])}
</div>""", unsafe_allow_html=True)

    st.markdown(f"""<div style="border-left:3px solid {_rc};padding:8px 14px;margin-top:8px;
font-family:'IBM Plex Mono',monospace;font-size:0.86rem;color:#e8e8e8;">
  <span style="color:{_rc};font-weight:600;">{_rl}</span> — {_guide["note"]}
</div>""", unsafe_allow_html=True)

    # ── RS vs Nifty ─────────────────────────────────────────
    if rs_nifty:
        st.divider()
        st.markdown("#### Relative Strength vs Nifty")
        _rs_ratio = rs_nifty.get("rs_ratio", 1.0)
        _rs_trend = rs_nifty.get("trend", "—")
        _rs_col   = rs_nifty.get("color", "#888")
        _rs_sym   = rs_nifty.get("sym_ret_pct", 0)
        _rs_nif   = rs_nifty.get("nifty_ret_pct", 0)

        rs1, rs2, rs3, rs4 = st.columns(4)
        rs1.metric("RS Ratio (20d)",   f"{_rs_ratio:.4f}",
                   delta="OUTPERFORMING" if _rs_ratio > 1 else "UNDERPERFORMING")
        rs2.metric(f"{sym} Return (20d)", f"{_rs_sym:+.2f}%")
        rs3.metric("Nifty Return (20d)", f"{_rs_nif:+.2f}%")
        rs4.metric("RS Trend",           _rs_trend)

        # RS chart
        _rs_ser = rs_nifty.get("rs_series", [])
        if len(_rs_ser) >= 5:
            fig_rs = go.Figure()
            fig_rs.add_trace(go.Scatter(
                x=list(range(len(_rs_ser))), y=_rs_ser,
                mode="lines", name=f"{sym} RS vs Nifty",
                line=dict(color=_rs_col, width=2),
                fill="tozeroy",
                fillcolor=f"rgba({int(_rs_col[1:3],16)},{int(_rs_col[3:5],16)},{int(_rs_col[5:7],16)},0.07)",
            ))
            fig_rs.add_hline(y=1.0, line=dict(color="#555", dash="dot", width=1.5),
                              annotation_text="Nifty parity (RS = 1.0)")
            fig_rs.update_layout(
                title=f"{sym} Relative Strength vs Nifty (rolling 20-day return ratio)",
                height=240, plot_bgcolor="#000", paper_bgcolor="#000",
                font=dict(color="#e8e8e8", family="IBM Plex Mono", size=9),
                xaxis=dict(title="Session #", gridcolor="#111"),
                yaxis=dict(title="RS Ratio", gridcolor="#111"),
                margin=dict(t=40, b=10),
            )
            st.plotly_chart(fig_rs, use_container_width=True)

        # RS strategy implication
        _rs_impl = {
            "STRONGLY OUTPERFORMING": "Stock is leading the market — bullish bias confirmed by market structure. "
                                       "Bull call spreads and long calls supported by both technicals and RS.",
            "OUTPERFORMING":           "Stock beating Nifty — mild bullish confirmation. "
                                       "Directional long trades have RS tailwind.",
            "IN LINE WITH NIFTY":      "No relative edge vs market. "
                                       "Strategy choice should rely purely on IV and technical bias.",
            "UNDERPERFORMING":         "Stock lagging Nifty — weak money flow. "
                                       "Long strategies carry RS headwind; consider bear spreads or neutral.",
            "STRONGLY UNDERPERFORMING":"Clear RS breakdown — money is flowing away from this stock. "
                                        "Put spreads and bearish credit spreads are regime-appropriate.",
        }
        st.caption(_rs_impl.get(_rs_trend, ""))
    elif sym.upper() not in ("NIFTY","BANKNIFTY","FINNIFTY","MIDCPNIFTY"):
        st.info("RS vs Nifty not available — yfinance data may be unavailable. Reload to retry.")

    # ── Event Calendar ───────────────────────────────────────
    st.divider()
    st.markdown("#### Event Calendar — Within Expiry Window")
    st.caption(f"Events detected between today and {expiry}.")

    if events_list:
        for _ev in events_list:
            _ev_c   = _ev.get("color", "#888")
            _ev_d   = _ev.get("days_away", 0)
            _ev_lbl = "TODAY" if _ev_d == 0 else (f"in {_ev_d}d" if _ev_d > 0 else "past")
            st.markdown(f"""<div style="background:#0d0d0d;border:1px solid #2a2a2a;
border-left:4px solid {_ev_c};padding:10px 14px;margin-bottom:6px;font-family:'IBM Plex Mono',monospace;">
  <div style="display:flex;justify-content:space-between;align-items:center;">
    <div>
      <span style="color:{_ev_c};font-size:0.92rem;font-weight:700;">{_ev['event']}</span>
      <span style="color:#888;font-size:0.80rem;margin-left:12px;">{_ev.get('date_str','')}</span>
    </div>
    <span style="color:{_ev_c};font-size:0.80rem;font-weight:600;">{_ev_lbl}</span>
  </div>
  <div style="color:#888;font-size:0.80rem;margin-top:4px;">{_ev.get('impact','')}</div>
</div>""", unsafe_allow_html=True)

        # Event-based IV advice
        high_impact = [e for e in events_list if e.get("days_away", 99) <= 5]
        if high_impact:
            st.markdown("""<div style="background:#1a0000;border:1px solid #ff3b3b;
border-left:4px solid #ff3b3b;padding:12px 16px;font-family:'IBM Plex Mono',monospace;margin-top:8px;">
  <div style="color:#ff3b3b;font-size:0.86rem;font-weight:600;">⚠ HIGH-IMPACT EVENT WITHIN 5 DAYS</div>
  <div style="color:#e8e8e8;font-size:0.82rem;margin-top:4px;line-height:1.7;">
    IV typically expands 20–40% in the 2 days before an RBI/Budget event and collapses 30–50% on the day.<br>
    • If event is within DTE: avoid naked short premium — IV spike risk is significant.<br>
    • Buy straddle before event + sell immediately after for vol-crush capture (advanced).<br>
    • Defined-risk spreads are safest: max loss is capped regardless of IV move.
  </div>
</div>""", unsafe_allow_html=True)
    else:
        st.success("✓ No high-impact events detected within the expiry window.")


# ══════════════════════════════════════════════════════════════
# TAB 12 — BACKTEST BY REGIME
# ══════════════════════════════════════════════════════════════
with t_backtest:
    st.markdown("### 📋 Forward Signal Log")
    st.caption(
        "Every load appends a signal snapshot to the log. This is real forward performance tracking — "
        "not synthetic backtest approximations. Use this to evaluate whether the engine's signals "
        "are leading actual price moves over time.")

    _sig_log = st.session_state.get("opt_signal_log", [])
    _sym_log = [e for e in _sig_log if e.get("symbol") == sym.upper()]

    if not _sym_log:
        st.info(f"No signal history yet for {sym}. Each time you click ⚡ LOAD, a snapshot is recorded.")
        st.caption("After 5+ loads you'll see signal trends. After 20+ you can evaluate forward accuracy.")
    else:
        # Summary stats from log
        _n_log   = len(_sym_log)
        _bull    = sum(1 for e in _sym_log if e.get("prob_up", 0.5) > 0.55)
        _bear    = sum(1 for e in _sym_log if e.get("prob_up", 0.5) < 0.45)
        _neut    = _n_log - _bull - _bear
        _avg_pu  = float(np.mean([e.get("prob_up", 0.5) for e in _sym_log]))
        _avg_mag = float(np.mean([e.get("flow_magnitude", 0) for e in _sym_log]))
        _avg_ev  = float(np.mean([e.get("strategy_ev", 0) for e in _sym_log]))

        # Adaptive conviction threshold from log history
        _mag_hist_log = [e.get("flow_magnitude", 0) for e in _sym_log]
        _conv_thr_log = float(np.percentile(_mag_hist_log, 70)) if len(_mag_hist_log) >= 5 else CFG["flow_conviction_seed"]

        sl1, sl2, sl3, sl4, sl5 = st.columns(5)
        sl1.metric("Log entries",     str(_n_log))
        sl2.metric("Avg P(↑)",        f"{_avg_pu*100:.1f}%")
        sl3.metric("Bull/Neut/Bear",  f"{_bull}/{_neut}/{_bear}")
        sl4.metric("Avg flow mag",    f"{_avg_mag:.3f}")
        sl5.metric("Avg top EV",      f"₹{_avg_ev:,.0f}")

        st.caption(f"Flow conviction threshold (70th pct of log): {_conv_thr_log:.3f}")

        # Signal log table — most recent first
        _log_rows = []
        for e in reversed(_sym_log[-50:]):
            _pu_l  = e.get("prob_up", 0.5)
            _fm_l  = e.get("flow_magnitude", 0)
            _ev_l  = e.get("strategy_ev", 0)
            _log_rows.append({
                "Time":       e.get("ts", "—")[-8:],    # HH:MM only
                "Date":       e.get("ts", "—")[:10],
                "P(↑)%":      f"{_pu_l*100:.1f}",
                "Flow Mag":   f"{_fm_l:.3f}",
                "Flow Score": f"{e.get('flow_score', 0):+.3f}",
                "Bias":       e.get("bias", "—")[:9],
                "Top Strategy":e.get("top_strategy", "—")[:18],
                "EV":         f"₹{_ev_l:,.0f}",
                "POP%":       f"{e.get('strategy_pop', 0.5)*100:.1f}",
                "IVR":        f"{e.get('ivr', 0):.0f}",
                "IV%":        f"{e.get('atm_iv_pct', 0):.1f}",
            })

        _log_df = pd.DataFrame(_log_rows)

        def _log_pu_style(v):
            try:
                vf = float(v)
                if vf > 55: return "color:#00d084;font-weight:700"
                if vf < 45: return "color:#ff3b3b;font-weight:700"
            except: pass
            return "color:#888"

        def _log_ev_style(v):
            if isinstance(v, str) and v.startswith("₹") and "-" not in v: return "color:#00d084"
            if isinstance(v, str) and "-" in v: return "color:#ff3b3b"
            return ""

        st.dataframe(
            _log_df.style
                .map(_log_pu_style, subset=["P(↑)%"])
                .map(_log_ev_style, subset=["EV"]),
            use_container_width=True, hide_index=True)

        # P(↑) trend chart
        if len(_sym_log) >= 3:
            _pu_series = [e.get("prob_up", 0.5) * 100 for e in _sym_log]
            _fm_series = [e.get("flow_magnitude", 0) for e in _sym_log]
            _ts_labels = [e.get("ts", "")[-5:] for e in _sym_log]

            _log_fig = go.Figure()
            _log_fig.add_trace(go.Scatter(
                x=list(range(len(_pu_series))), y=_pu_series,
                mode="lines+markers", name="P(↑)%",
                line=dict(color="#00d084", width=2),
                marker=dict(size=5)))
            _log_fig.add_trace(go.Scatter(
                x=list(range(len(_fm_series))), y=[v * 100 for v in _fm_series],
                mode="lines", name="Flow Magnitude ×100",
                line=dict(color="#ff8c00", width=1.5, dash="dot"),
                yaxis="y2"))
            # Neutral zone band — 45% to 55% = no edge
            _log_fig.add_hrect(y0=45, y1=55, fillcolor="#333", opacity=0.15,
                                line_width=0, annotation_text="no-edge zone",
                                annotation_font_color="#555", annotation_font_size=9)
            _log_fig.update_layout(
                title=f"{sym} — Signal History ({_n_log} loads)",
                height=240, plot_bgcolor="#000", paper_bgcolor="#000",
                font=dict(color="#e8e8e8", family="IBM Plex Mono", size=9),
                xaxis=dict(gridcolor="#111", tickvals=list(range(len(_ts_labels))),
                           ticktext=_ts_labels, tickangle=-45),
                yaxis=dict(title="P(↑)%", gridcolor="#111", range=[30, 70]),
                yaxis2=dict(title="Flow Mag ×100", overlaying="y", side="right",
                            gridcolor="#111", range=[0, 60]),
                legend=dict(orientation="h", yanchor="bottom", y=1.0),
                margin=dict(t=40, b=40),
            )
            st.plotly_chart(_log_fig, use_container_width=True)

        st.divider()
        if st.button("🗑 Clear signal log for this symbol", key="clear_sig_log"):
            _all_log = st.session_state.opt_signal_log
            st.session_state.opt_signal_log = [e for e in _all_log if e.get("symbol") != sym.upper()]
            _save_signal_log(st.session_state.opt_signal_log)
            st.success(f"Cleared {sym} entries from signal log.")
            st.rerun()

        st.caption(
            f"Log persists to {CFG['signal_log_file']} · max {CFG['signal_log_max']} entries · "
            "Use this log to evaluate whether high-conviction flow signals (flow_magnitude > threshold) "
            "precede actual price moves in the direction predicted.")


# ══════════════════════════════════════════════════════════════
# TAB 9 — MATHS & LOGIC REFERENCE
# ══════════════════════════════════════════════════════════════
with t_math:
    st.markdown("### 📐 Engine Maths & Logic Reference")
    st.caption("Complete derivations, formulas, and decision logic used by every module in this engine.")

    SEC = {
        "bb": "border-bottom:1px solid #ff8c00;padding-bottom:4px;margin-bottom:14px;",
        "hd": "color:#ff8c00;font-size:1.0rem;font-weight:700;letter-spacing:.12em;text-transform:uppercase;font-family:'IBM Plex Mono',monospace;",
        "sh": "color:#ffb347;font-size:0.88rem;font-weight:600;font-family:'IBM Plex Mono',monospace;margin-top:16px;margin-bottom:6px;",
        "body": "color:#e8e8e8;font-size:0.86rem;line-height:1.85;font-family:'IBM Plex Mono',monospace;",
        "formula": "background:#0d1400;border-left:3px solid #ff8c00;padding:10px 16px;margin:8px 0 14px;font-size:0.88rem;color:#ffb347;font-family:'IBM Plex Mono',monospace;line-height:2.0;",
        "note": "background:#111;border:1px solid #2a2a2a;border-left:3px solid #1e90ff;padding:8px 14px;font-size:0.82rem;color:#7ec8e3;font-family:'IBM Plex Mono',monospace;margin:8px 0 12px;line-height:1.7;",
        "warn": "background:#1a0d00;border:1px solid #2a2a2a;border-left:3px solid #ff8c00;padding:8px 14px;font-size:0.82rem;color:#ffb347;font-family:'IBM Plex Mono',monospace;margin:8px 0 12px;line-height:1.7;",
        "tag": "background:#1a1400;color:#ff8c00;font-size:0.78rem;padding:2px 8px;border:1px solid #ff8c00;font-family:'IBM Plex Mono',monospace;margin-right:6px;",
    }

    def sec(title):
        st.markdown(f'<div style="{SEC["bb"]}"><span style="{SEC["hd"]}">{title}</span></div>', unsafe_allow_html=True)
    def sh(title):
        st.markdown(f'<div style="{SEC["sh"]}">{title}</div>', unsafe_allow_html=True)
    def body(text):
        st.markdown(f'<div style="{SEC["body"]}">{text}</div>', unsafe_allow_html=True)
    def formula(text):
        st.markdown(f'<div style="{SEC["formula"]}">{text}</div>', unsafe_allow_html=True)
    def note(text):
        st.markdown(f'<div style="{SEC["note"]}">ℹ {text}</div>', unsafe_allow_html=True)
    def warn(text):
        st.markdown(f'<div style="{SEC["warn"]}">⚠ {text}</div>', unsafe_allow_html=True)

    # ── 1. TIME TO EXPIRY ─────────────────────────────────────
    with st.expander("1 · Time to Expiry — Trading-Day Fraction T", expanded=True):
        sec("Time to Expiry T")
        body("""All Black-Scholes calculations use <b>T</b> expressed as a fraction of a trading year,
        not calendar days. This is because option theta accrues only on days the market is open — weekends
        and holidays do not eat into time value.""")
        formula("""T = busday_count(today, expiry) / 252<br><br>
where busday_count counts weekdays (Mon–Fri) between today and the expiry date.<br>
252 = conventional number of NSE trading days per year (CFG["ann_days"]).<br><br>
Example: 7-DTE expiry with 2 weekend days → busday_count = 5 → T = 5/252 = 0.01984<br>
Old (wrong) method: T = 7/365 = 0.01918 — understates time by ~3%""")
        note("T is computed once at load time via numpy.busday_count and stored in session_state['opt_T']. "
             "No downstream calculation ever recomputes it from calendar days.")

    # ── 2. BLACK-SCHOLES ENGINE ───────────────────────────────
    with st.expander("2 · Black-Scholes Merton Pricing Model"):
        sec("Black-Scholes Merton (BSM) with Dividend Yield")
        body("""The engine uses the <b>Merton (1973) continuous-dividend</b> extension of Black-Scholes.
        For pure index options (NIFTY, BANKNIFTY) q ≈ 0 and the formula reduces to classic BSM.
        For dividend-paying stocks, q is fetched from yfinance and corrects for the
        ex-dividend premium reduction that standard BSM ignores.""")

        sh("Forward Price")
        formula("F = S · e^(r − q)·T")
        body("S = spot price, r = risk-free rate (RFR), q = continuous dividend yield, T = trading-day fraction.")

        sh("d₁ and d₂")
        formula("""d₁ = [ ln(F/K) + 0.5·σ²·T ] / (σ·√T)<br>
d₂ = d₁ − σ·√T<br><br>
K = strike price, σ = implied or historical volatility (annualised decimal)""")

        sh("Option Prices")
        formula("""Call = e^(−q·T) · S · N(d₁) − K · e^(−r·T) · N(d₂)<br>
Put  = K · e^(−r·T) · N(−d₂) − e^(−q·T) · S · N(−d₁)<br><br>
N(x) = standard normal CDF = 0.5 · [1 + erf(x/√2)]  — implemented with math.erf, no scipy needed""")

        sh("Put-Call Parity (sanity check)")
        formula("Call − Put = e^(−q·T) · S − K · e^(−r·T) = PV(Forward) − PV(Strike)")
        note("If live CE_LTP and PE_LTP violate put-call parity by more than the bid-ask spread, "
             "an arbitrage exists. The engine does not exploit this but it is visible in the chain tab.")

        sh("Edge cases")
        body("""• T ≤ 0: returns intrinsic value max(S−K, 0) or max(K−S, 0).<br>
• σ ≤ 0 or S ≤ 0 or K ≤ 0: same intrinsic fallback.<br>
• IV bounds: _sanitise_iv() clamps to [0.01, 5.0] (1%–500% annualised). Values outside are replaced
  with the HV fallback to prevent API garbage from poisoning Greeks.""")

    # ── 3. IMPLIED VOLATILITY ─────────────────────────────────
    with st.expander("3 · Implied Volatility — Newton-Raphson Solver"):
        sec("Implied Volatility (IV)")
        body("""IV is the σ that makes the BSM price equal the observed market price.
        There is no closed-form inverse, so we solve numerically via Newton-Raphson iteration.""")

        sh("Algorithm")
        formula("""Initial guess (Brenner-Subrahmanyam):
  σ₀ = max(0.05, min(C / (S · √T · √(2/π) · 0.4), 2.0))<br><br>
Newton step (up to 200 iterations):
  σₙ₊₁ = σₙ + (market_price − BSM(σₙ)) / vega(σₙ)<br><br>
Stop when |market_price − BSM(σ)| < 1e-6 or vega < 1e-10<br>
Clamp: σ ∈ [0.001, 10.0] after each step""")

        sh("ATM IV — Primary Method (Brenner-Subrahmanyam 1988)")
        formula("""Straddle ≈ S · σ · √(2T/π)  →  σ_ATM = Straddle / (S · √T · √(2/π))<br><br>
√(2/π) = 0.79788...<br><br>
This uses the straddle midpoint LTP (CE_LTP + PE_LTP), not individual IV fields.
Avoids call/put IV averaging bias and is robust to API format inconsistencies.""")

        note("The per-leg IV from the Upstox API may be in percent format (e.g. 18.5) or decimal (0.185). "
             "_sanitise_iv() normalises: if iv_raw >= 3.0 → divide by 100 (percent→decimal). "
             "Threshold is 3.0 not 2.0 to avoid misclassifying IVs in the 2–3% range. "
             "Then clamps to [0.01, 5.0].")

    # ── 4. GREEKS ─────────────────────────────────────────────
    with st.expander("4 · The Greeks — Delta, Gamma, Theta, Vega"):
        sec("Option Greeks (First-Order Sensitivities)")

        sh("Delta  Δ")
        formula("""Call Δ = e^(−q·T) · N(d₁)       range: [0, 1]<br>
Put  Δ = e^(−q·T) · (N(d₁) − 1)  range: [−1, 0]<br><br>
Interpretation: ₹ change in option value per ₹1 change in spot.<br>
ATM option: |Δ| ≈ 0.50.  Deep ITM: |Δ| → 1.  Deep OTM: |Δ| → 0.""")

        sh("Gamma  Γ")
        formula("""Γ = e^(−q·T) · φ(d₁) / (S · σ · √T)       same for calls and puts<br><br>
φ(x) = standard normal PDF = e^(−x²/2) / √(2π)<br><br>
Interpretation: rate of delta change per ₹1 move in spot.<br>
Peaks at ATM and explodes near expiry — greatest gamma risk on expiry day.""")

        sh("Theta  θ  (per trading day)")
        formula("""Call θ = [ −S · e^(−q·T) · φ(d₁) · σ / (2·√T)  +  q·S·e^(−q·T)·N(d₁)  −  r·K·e^(−r·T)·N(d₂)  ] / 252<br>
Put  θ = [ −S · e^(−q·T) · φ(d₁) · σ / (2·√T)  −  q·S·e^(−q·T)·N(−d₁)  +  r·K·e^(−r·T)·N(−d₂) ] / 252<br><br>
Divided by 252 (trading days) NOT 365 (calendar days).
A position only decays on market-open days; using 365 understates daily theta by ~13%.""")
        warn("Old engines dividing theta by 365 understate daily decay by ~(252/365 − 1) ≈ 31%. "
             "This engine uses CFG['theta_days'] = 252 throughout.")

        sh("Vega  ν")
        formula("""ν = S · e^(−q·T) · φ(d₁) · √T / 100<br><br>
Divided by 100 → ν represents ₹ P&L per 1 percentage point (1%) change in IV.<br>
Example: ν = 45 means the option gains ₹45 if IV rises 1%.""")

        sh("ITM Probability  P(ITM)")
        formula("""Call: P(ITM) = N(d₂)    (risk-neutral probability of S_T > K)<br>
Put:  P(ITM) = N(−d₂)<br><br>
This is NOT the same as delta. Delta = e^(−q·T)·N(d₁).
N(d₂) is the true risk-neutral exercise probability; N(d₁) is delta-hedge ratio.""")

    # ── 5. HISTORICAL VOLATILITY ──────────────────────────────
    with st.expander("5 · Historical Volatility — HV Calculation"):
        sec("Historical Volatility (HV)")
        body("Computed from daily log returns of the close price series over a rolling window.")
        formula("""Log return: rᵢ = ln(Cᵢ / Cᵢ₋₁)<br><br>
HV(window) = std(r₁, r₂, ..., r_window) × √252<br><br>
window = 20 trading days for HV20 (CFG["hv_window"])<br>
window = 10 trading days for HV10 (CFG["hv_window_fast"])<br>
Annualisation base = 252 (CFG["ann_days"]) — consistent with T computation""")
        note("HV is a backward-looking realised volatility. IV is forward-looking implied volatility. "
             "The IV/HV ratio is the primary edge signal: IV > HV → premium is rich → sell. "
             "IV < HV → premium is cheap → buy.")

    # ── 6. IV RANK ────────────────────────────────────────────
    with st.expander("6 · IV Rank — Volatility Regime Scoring"):
        sec("IV Rank (IVR)")
        body("""IV Rank measures where the current ATM IV sits within its own history.
        It is more informative than raw IV because the same 20% IV means different things
        for NIFTY (always ~14-22%) vs a biotech stock (can range 15-80%).""")

        sh("Formula")
        formula("""IVR = (IV_current − IV_min) / (IV_max − IV_min) × 100<br><br>
IV_min, IV_max drawn from up to 252 stored ATM IV observations (1 trading year).<br>
History is persisted to .monarch_iv_history.json and survives app restarts.""")

        sh("Bootstrap (fewer than 3 observations)")
        formula("""IVR_bootstrap = 50 + 50 × tanh( (IV − HV) / (0.30 × HV) )<br><br>
At IV = HV:         IVR ≈ 50  (fair value — no premium edge)<br>
At IV = 1.30 × HV:  IVR ≈ 75  (elevated — sell zone)<br>
At IV = 0.70 × HV:  IVR ≈ 25  (cheap — buy zone)<br><br>
Denominator 0.30 × HV calibrated to NSE index IV/HV std (~25-35%).
At 1 std above HV, IVR maps to ~75 (top quartile = sell signal).""")

        sh("Regime Zones")
        formula("""IVR < 25:   LOW VOL       → Buy premium (debit spreads, long options, straddles)<br>
25 ≤ IVR < {CFG['iv_hv_pct_buy']+15}:  NORMAL-LOW    → Slight buy lean (calendars, ratio spreads)<br>
{CFG['iv_hv_pct_buy']+15} ≤ IVR < {CFG['iv_hv_pct_sell']}:  NORMAL-HIGH   → Slight sell lean (balanced spreads)<br>
{CFG['iv_hv_pct_sell']} ≤ IVR < {int(CFG['iv_hv_pct_sell']+(100-CFG['iv_hv_pct_sell'])*0.6)}: ELEVATED → Lean sell (credit spreads, iron condors)<br>
IVR ≥ {int(CFG['iv_hv_pct_sell']+(100-CFG['iv_hv_pct_sell'])*0.6)}: HIGH VOL → Sell premium (strangles, short straddles)<br><br>
<i>Thresholds are CFG-driven (iv_hv_pct_sell={CFG['iv_hv_pct_sell']}, iv_hv_pct_buy={CFG['iv_hv_pct_buy']}) — adaptive, not fixed quartile cuts.</i>""")

    # ── 7. DIRECTIONAL BIAS ───────────────────────────────────
    with st.expander("7 · Directional Bias — Adaptive Z-Score Factor Model"):
        sec("Directional Bias Score  (−100 to +100)")
        body("""The bias score aggregates 5 independent, non-collinear factor groups into a single score.
        Every sub-signal is converted to a <b>z-score</b> (using its own 1-year rolling history)
        or a <b>percentile rank</b> (within current data distribution) — no fixed tanh centres or 
        magic scaling constants. Each factor is normalised to [−1, +1].
        Final score = CFG-weighted sum, scaled to ±100 for display.""")

        sh("Factor Groups — Leading Indicators First")
        formula(f"""Signal model optimised for 1–5 day prediction horizon.<br>
Markets move due to POSITIONING CHANGES, not price indicators.<br><br>
GROUP 1 — FLOW  (weight {CFG['factor_weights']['flow']:.0%}) — LEADING: ΔIV, ΔPCR, ΔSkew, ΔOI, ΔGEX<br>
  All deltas z-scored against rolling history → each ∈ [−1, +1]<br>
  dPCR  rising = more put writing = support = BULLISH (+)<br>
  dSkew steepening = downside fear = BEARISH (−)<br>
  dIV   rising = hedging demand = BEARISH (−)<br>
  Flow = 0.35×dPCR + 0.30×dSkew + 0.20×dIV + 0.10×dOI + 0.05×dGEX<br><br>
GROUP 2 — POSITIONING  (weight {CFG['factor_weights']['positioning']:.0%}) — LEADING: PCR level, OI walls, max pain<br>
  PCR level  = 2 × PCR_percentile_in_chain − 1  (high PCR = put support = bullish)<br>
  OI skew    = zscore((put_OI_below − call_OI_above) / total_OI)<br>
  Max pain   = −(spot − max_pain) / expected_move  (gravitational pull)<br>
  Positioning = 0.45×PCR_level + 0.35×OI_skew + 0.20×max_pain<br><br>
GROUP 3 — VOL REGIME  (weight {CFG['factor_weights']['vol_regime']:.0%}) — CONCURRENT: IV/HV pct, term structure<br>
  IV/HV pct  = percentile of current IV/HV ratio in 1-year history<br>
  Term slope = IV_far − IV_near  (backwardation = stress = bearish)<br>
  Vol regime = 0.60 × (−IV_pct_z) + 0.40 × term_slope_z<br><br>
GROUP 4 — RELATIVE STRENGTH  (weight {CFG['factor_weights']['rel_strength']:.0%}) — CONFIRMING: RS ratio vs Nifty<br>
  RS z-score = zscore(RS_ratio_vs_Nifty) — confirming, not driving<br><br>
GROUP 5 — TREND  (weight {CFG['factor_weights']['trend']:.0%}) — CONFIRMING (most lagging): EMA + ADX + RSI<br>
  EMA score  = (passes / 5 checks) × 2 − 1<br>
  ADX pct    = percentile × direction (no fixed ADX > 25 threshold)<br>
  RSI z-score contributes only 10% within this factor<br>
  Trend = 0.60×EMA + 0.30×ADX + 0.10×RSI_z<br><br>
Note: RSI and MACD are absorbed into Trend at 10% of 10% = ~1% total signal influence.""")

        sh("Final Score & Bias Thresholds")
        formula(f"""raw_score = {CFG['factor_weights']['flow']:.2f}×flow + {CFG['factor_weights']['positioning']:.2f}×positioning + {CFG['factor_weights']['vol_regime']:.2f}×vol_regime + {CFG['factor_weights']['rel_strength']:.2f}×rel_strength + {CFG['factor_weights']['trend']:.2f}×trend  ∈ [−1, +1]<br>
score_100 = round(raw_score × 100)   displayed as ±100<br><br>
score ≥  30:  STRONGLY BULLISH<br>
score ≥  12:  BULLISH<br>
score >  −12: NEUTRAL<br>
score ≥  −30: BEARISH<br>
score <  −30: STRONGLY BEARISH<br><br>
Factor weights: {dict(CFG['factor_weights'])}""")

        note("RSI and MACD are intentionally demoted. A stock can sit at RSI 75 for 20 days. "
             "But a 3-sigma ΔPCR event plus skew steepening resolves in 1–5 days — exactly the "
             "prediction horizon this engine targets. Flow and positioning are leading; trend confirms.")

    # ── 8. OI ANALYSIS ───────────────────────────────────────
    with st.expander("8 · Open Interest Analysis — Max Pain, GEX, PCR, Skew"):
        sec("Open Interest Analysis")

        sh("Max Pain")
        formula("""Max Pain = argmin_K  Σ_i [ max(K − K_i, 0) × CE_OI_i  +  max(K_i − K, 0) × PE_OI_i ]<br><br>
For each candidate expiry price K, compute total dollar pain to option writers.<br>
The strike that minimises aggregate writer loss is Max Pain.<br>
Price gravitates toward max pain as expiry approaches due to dealer delta-hedging.""")

        sh("Call Wall & Put Wall (OI Cluster Peak)")
        formula("""For each side, find strike i that maximises the 3-strike sliding sum:<br>
  wall = argmax_i ( OI[i-1] + OI[i] + OI[i+1] )<br><br>
3-strike window smooths single-bar OI spikes and finds the true cluster centre.
Call Wall = resistance (dealers short calls → sell futures above it).
Put Wall  = support    (dealers short puts  → buy futures below it).""")

        sh("Put-Call Ratio (PCR)")
        formula("""PCR_OI = Total_PE_OI / Total_CE_OI  (across all strikes in chain)<br><br>
PCR percentile = fraction of per-strike PCR values ≤ aggregate PCR (full chain, not just pain window)<br><br>
Percentile ≥ 75%: BULLISH — heavy put writing = support below<br>
Percentile 45–55%: NEUTRAL — balanced OI<br>
Percentile ≤ 25%: BEARISH — heavy call writing = resistance above""")

        sh("Expected Move")
        formula("""ATM Straddle = CE_LTP_ATM + PE_LTP_ATM<br><br>
Expected Move ±1σ = Straddle / Spot × 100  (in %)<br>
Expected Move ±2σ = 2 × EM_1σ<br><br>
Derived from Brenner-Subrahmanyam: Straddle ≈ S · σ · √(2T/π)
→ σ = Straddle / (S · √T · √(2/π))""")

        sh("Gamma Exposure (GEX)")
        formula("""GEX per strike (₹) = [ γ_call × CE_OI − γ_put × PE_OI ] × lot_size × spot<br><br>
Dealer convention: dealers are net SHORT options → their GEX = −(buyer's GEX)<br>
  CE_OI × γ_call → positive GEX (dealers short calls → long delta → buy dips)<br>
  PE_OI × γ_put  → negative GEX (dealers short puts  → short delta → sell rallies)<br><br>
Scaled by lot_size × spot → rupee-notional units (comparable across instruments)<br><br>
Net GEX > 0: POSITIVE regime → dealers amplify mean-reversion → range-bound, vol suppressed<br>
Net GEX < 0: NEGATIVE regime → dealers amplify trending → breakout risk, vol expansion""")

        sh("Gamma Flip Level")
        formula("""Strikes sorted by distance from spot (nearest first).<br>
Cumulative GEX accumulated outward until sign changes.<br>
Gamma Flip = strike where cumulative GEX crosses zero.<br><br>
Significance: above flip → positive GEX regime (pinning). Below → negative (trending).
Crossing the flip can trigger rapid vol expansion and gap moves.""")

        sh("IV Skew")
        formula("""Skew (pp) = IV_put(ATM − 1 step) − IV_call(ATM + 1 step)  × 100<br><br>
Positive skew (put IV > call IV): downside protection demand elevated — normal for indices<br>
Negative skew (call IV > put IV): upside speculation — rare, signals breakout positioning<br>
Near-zero skew: balanced demand — market not hedging directionally""")

    # ── 9. STRATEGY FIT SCORE ─────────────────────────────────
    with st.expander("9 · Strategy Scoring — Probabilistic EV Engine"):
        sec("Strategy Selection: Expected Value Ranking (no if/then rules)")
        body("""The engine no longer uses directional rules like 'if bullish → bull spread'.
        All 14 canonical strategies are evaluated simultaneously by Expected Value
        computed from Monte Carlo simulation. The highest composite EV score wins.""")

        sh("Step 1 — Directional Signal (Leading Indicators First)")
        formula(f"""Signal model optimised for 1–5 day prediction horizon.
Markets move due to POSITIONING CHANGES, not price indicators.<br><br>
<b>FACTOR 1 — FLOW (weight {CFG['factor_weights']['flow']:.0%}) — LEADING</b><br>
  dPCR  = zscore(PCR_today − PCR_5day_avg)   → rising PCR = put writing = BULLISH (+)<br>
  dSkew = zscore(Skew_today − Skew_5day_avg)  → steepening skew = fear = BEARISH (−)<br>
  dIV   = zscore(IV_today − IV_5day_avg)      → rising IV = hedging = BEARISH (−)<br>
  dOI   = zscore(OI_today − OI_5day_avg)      → magnitude of new positioning<br>
  dGEX  = zscore(GEX_today − GEX_5day_avg)   → falling GEX = trending move coming<br>
  Flow composite = 0.35×dPCR + 0.30×dSkew + 0.20×dIV + 0.10×dOI + 0.05×dGEX<br><br>
<b>FACTOR 2 — POSITIONING (weight {CFG['factor_weights']['positioning']:.0%}) — LEADING</b><br>
  PCR level  = 2 × PCR_percentile_in_chain − 1   (high PCR = put support = bullish)<br>
  OI skew    = zscore((put_OI_below − call_OI_above) / total_OI)<br>
  Max pain   = −(spot − max_pain) / expected_move  (gravitational pull)<br>
  Positioning = 0.45×PCR_level + 0.35×OI_skew + 0.20×max_pain_pull<br><br>
<b>FACTOR 3 — VOL REGIME (weight {CFG['factor_weights']['vol_regime']:.0%}) — CONCURRENT</b><br>
  IV/HV percentile = percentile of current IV/HV in 1-year history<br>
  Term slope = IV_far − IV_near  (backwardation = stress = bearish)<br>
  Vol regime = 0.60 × (−IV_percentile_z) + 0.40 × term_slope_z<br><br>
<b>FACTOR 4 — RELATIVE STRENGTH (weight {CFG['factor_weights']['rel_strength']:.0%}) — CONFIRMING</b><br>
  RS z-score = zscore(RS_ratio vs Nifty 20D) + 0.3 × RS_slope<br><br>
<b>FACTOR 5 — TREND (weight {CFG['factor_weights']['trend']:.0%}) — CONFIRMING (most lagging)</b><br>
  EMA score = (passes/5 checks) × 2 − 1<br>
  ADX pct   = percentile of ADX in rolling history × direction<br>
  RSI z-score contributes only 10% within this factor<br>
  Trend = 0.60×EMA + 0.30×ADX + 0.10×RSI<br><br>
raw_score = {CFG['factor_weights']['flow']:.2f}×flow + {CFG['factor_weights']['positioning']:.2f}×positioning + {CFG['factor_weights']['vol_regime']:.2f}×vol_regime + {CFG['factor_weights']['rel_strength']:.2f}×rel_strength + {CFG['factor_weights']['trend']:.2f}×trend<br>
prob_up   = logistic(raw_score × 4)  =  1 / (1 + e^{{−raw_score × 4}})<br>
prob_down = 1 − prob_up<br><br>
Scale: raw=0.50 → prob_up≈98% · raw=0.25 → prob_up≈73% · raw=0.10 → prob_up≈60%""")

        sh("Step 2 — Monte Carlo EV per Strategy")
        formula(f"""For each of 14 strategies:<br><br>
  Simulate {CFG['pop_simulations']:,} terminal prices using:<br>
    ST = S × exp((μ − 0.5σ²)T + σ√T × Z)   (real-world drift μ from 60-day history)<br>
    σ = iv_surface(strike)   (per-strike IV from cubic spline interpolation)<br><br>
  POP = P(total_PnL > 0)   across all paths<br>
  EV  = E[total_PnL]       = mean PnL across all paths""")

        sh("Step 3 — EV-Adjusted Composite Score")
        formula("""ev_norm      = tanh(EV / MaxRisk × 0.5)              ∈ [0, 1]<br>
dte_align    = exp(−ln(2) × dist / half_range)      ∈ [0.05, 1]   (no binary in/out)<br>
safety_factor= logistic(2 × (safety_ratio − 1.0))   ∈ [0, 1]   (smooth sigmoid)<br>
ts_factor    = 1 + 0.5 × tanh(IV_slope × 20)        ∈ [0.5, 1.5]  (calendar bonus)<br>
dir_align    = prob_up   for bull strategies<br>
              = prob_down for bear strategies<br>
              = 1 − 2|prob_up − 0.5|  for neutral<br><br>
ev_score     = ev_norm × POP × dte_align × safety_factor × ts_factor<br>
composite    = 0.60 × ev_score + 0.40 × (dir_align − 0.5) × 2  ∈ [−1, +1]<br>
display_score= (composite + 1) / 2 × 100   (0–100)<br><br>
Selection criterion: sort strategies by composite descending. No if/then rules.""")

        sh("Step 4 — Kelly Fraction")
        formula("""Kelly = EV / MaxRisk   (EV-based Kelly, not win-rate formula)<br>
Capped at 25% of capital. Fractional (50%) applied for variance reduction.<br>
Position size = Kelly_capped × 0.5 × capital""")

        note("The Iron Condor can rank #1 even when P(↑)=60% if its EV is higher than "
             "a directional call after accounting for skew-aware Monte Carlo paths. "
             "There are no strategy exclusions — every structure competes on EV.")

    # ── 10. PAYOFF BUILDER ────────────────────────────────────
    with st.expander("10 · Payoff Builder — P&L Computation"):
        sec("Multi-Leg Payoff at Expiry")

        sh("Per-Leg Payoff")
        formula("""For each leg (option type, strike K, premium P, quantity Q, direction d = ±1):<br><br>
  Intrinsic(spot) = max(spot − K, 0)  for CE<br>
                  = max(K − spot, 0)  for PE<br><br>
  Leg P&L = d × (Intrinsic − P) × Q × lot_size<br><br>
  d = +1 for Buy, −1 for Sell<br>
  P = entry premium (BS-theoretical or manually entered)<br>
  Total P&L = Σ all legs""")

        sh("Breakeven Detection")
        formula("""Scan payoff array for sign changes between adjacent price points i-1 and i:<br><br>
  if payoff[i-1] × payoff[i] < 0:<br>
      frac = |payoff[i-1]| / (|payoff[i-1]| + |payoff[i]|)<br>
      BE = price[i-1] + frac × (price[i] − price[i-1])<br><br>
Linear interpolation between the two surrounding points — accurate to ±0.1 strike units.""")

        sh("Chart Range")
        formula("""Base range = ±3σ using trading-day T:<br><br>
  exp_move_frac = IV × √(2T/π)        (expected |move| = E[|S_T − S|] / S)<br>
  range_frac    = max(3 × exp_move_frac,  0.03)<br><br>
  price_range = linspace(spot × (1 − range_frac), spot × (1 + range_frac), 400)<br><br>
3σ captures 99.7% of the lognormal distribution.
Floor of ±3% prevents degenerate zero-width charts on same-day expiry.""")

        sh("Reward-to-Risk Ratio")
        formula("""R:R = |max_profit / max_loss|<br><br>
Reports ∞ when max_loss = 0 (e.g. long straddle — no loss scenario on expiry payoff).<br>
Note: payoff is at-expiry intrinsic only — does not include time value of intermediate exit.""")

    # ── 11. IV EDGE SIGNALS ───────────────────────────────────
    with st.expander("11 · IV Edge Signals — Adaptive Rich vs Cheap Per Strike"):
        sec("Per-Strike IV Edge Classification (Adaptive Percentile)")
        body("""Every strike in the chain tab is labelled SELL (rich) or BUY (cheap)
        based on its IV/HV ratio ranked within the CURRENT CHAIN's own distribution.
        Thresholds are computed as percentiles of all observed IV/HV ratios across strikes —
        fully adaptive to each instrument and regime. No fixed 1.20×/0.85× hard-coded constants.""")

        formula(f"""For each strike: ratio = IV_strike / HV20<br><br>
Collect all valid CE and PE IV/HV ratios across the chain window.<br>
SELL threshold = {CFG['iv_hv_pct_sell']}th percentile of chain ratios<br>
BUY  threshold = {CFG['iv_hv_pct_buy']}th percentile of chain ratios<br><br>
ratio ≥ SELL threshold → SELL (rich) — this strike's IV is elevated vs the chain<br>
ratio ≤ BUY  threshold → BUY  (cheap) — this strike's IV is depressed vs the chain<br>
Otherwise: neutral (no edge)<br><br>
Fallback: uses CFG["iv_rich_ratio"] / CFG["iv_cheap_ratio"] only when chain has < 4 valid strikes.<br>
Thresholds are configurable via CFG["iv_hv_pct_sell"] and CFG["iv_hv_pct_buy"].""")

        note("The directional signal (BUY / SELL on the CE/PE columns) is separate from the IV edge signal. "
             "CE_Dir = BUY if bias_score ≥ 12 (bullish). PE_Dir = BUY if bias_score ≤ −12 (bearish). "
             "Strongest edge = directional signal AND vol signal aligned on the same side.")

    # ── 12. CALENDAR SPREAD TERM STRUCTURE ───────────────────
    with st.expander("12 · Calendar Spread — Term Structure Logic"):
        sec("Calendar Spread Entry Condition")
        body("""A calendar spread (sell near-expiry CE, buy far-expiry CE at same strike) is only
        attractive when the term structure is inverted: front-month IV > back-month IV.
        This ensures you are selling expensive near-term vol and buying cheap far-term vol.""")

        formula("""Entry condition:  front_IV > back_IV<br><br>
Profit mechanism:<br>
  1. Near-month option decays faster (higher theta)<br>
  2. If IV term structure normalises, back-month gains more than near-month loses<br>
  3. P&L peaks when spot pins ATM on near-month expiry<br><br>
When front_IV ≤ back_IV (normal upward-sloping term structure):<br>
  You are BUYING the expensive vol and SELLING the cheap vol — adverse carry.<br>
  The engine flags this in the strategy rationale text.""")

        warn("The calendar signal requires fetching the ATM IV for the NEXT expiry. "
             "This is only available if you load two expiries separately and compare them. "
             "The engine displays a term structure warning when front/back IVs are not available.")

    # ── 13. DIVIDEND YIELD ADJUSTMENT ────────────────────────
    with st.expander("13 · Dividend Yield — Merton Model Adjustment"):
        sec("Continuous Dividend Yield q (Merton 1973)")
        body("""Standard BSM assumes the underlying pays no dividends. For dividend-paying
        stocks, the call is cheaper (dividend reduces forward price) and the put is richer.
        The Merton model adjusts by replacing S with S·e^(−q·T) in the forward price.""")

        formula("""Forward price with dividends:  F = S · e^(r − q) · T<br><br>
Effect on call:  lower forward → call cheaper by ~q × S × T × Δ<br>
Effect on put:   lower forward → put richer by the same amount<br><br>
q is fetched from yfinance Ticker.info["dividendYield"] (annual %).<br>
Sanity clamp: q must be in [0, 0.20] — rejects erroneous values > 20%.<br>
For indices and non-dividend stocks: q = 0 → identical to classic BSM.""")

        note("For NSE indices (NIFTY, BANKNIFTY etc.), q = 0 because index futures already "
             "embed dividends in the cost-of-carry. For stocks like ITC (yield ~3.5%) or "
             "COALINDIA (~7%), the Merton adjustment materially changes ATM prices.")

# ══════════════════════════════════════════════════════════════
# TAB — EDGE AUDIT
# Determines WHERE the signal engine actually has edge:
# Direction, Vol, Skew, Pinning, or None.
# ══════════════════════════════════════════════════════════════
with t_edge:
    st.markdown("### 🔬 Edge Audit — Signal Diagnostic")
    st.caption("Measures what actually happens AFTER each signal across 1d/2d/3d/5d horizons. "
               "This is not a backtest — it is a live edge diagnostic computed from real forward prices.")

    # ── Resolve forward outcomes for unresolved signals ───────────────────────
    _edge_log = st.session_state.get("opt_signal_log", [])
    _edge_oi_total = float(chain_df["CE_OI"].sum() + chain_df["PE_OI"].sum()) if not chain_df.empty else 0.0
    _edge_skew     = float(oi_d.get("skew_pp", 0.0) or 0.0) if oi_d else 0.0
    _edge_log = _resolve_edge_outcomes(
        ohlcv_df, _edge_log,
        current_spot    = spot,
        current_iv      = atm_iv,
        current_skew    = _edge_skew,
        current_oi_total= _edge_oi_total,
    )
    st.session_state["opt_signal_log"] = _edge_log

    # ── Summary counts ────────────────────────────────────────────────────────
    _all_sym    = [e for e in _edge_log if e.get("symbol","").upper() == sym.upper()]
    _resolved_n = len([e for e in _all_sym if e.get("resolved")])
    _pending_n  = len([e for e in _all_sym if not e.get("resolved")])
    _total_n    = len(_all_sym)

    ea1, ea2, ea3, ea4 = st.columns(4)
    ea1.metric("Total Signals", str(_total_n),    help="All signals recorded for this symbol")
    ea2.metric("Resolved",      str(_resolved_n), help="Signals with full 5-day forward outcome data")
    ea3.metric("Pending",       str(_pending_n),  help="Signals awaiting 5 trading days to elapse")
    ea4.metric("Min for Edge",  "5",              help="Minimum resolved signals needed per group")

    if _resolved_n < 5:
        st.info(f"ℹ️ **{_resolved_n} resolved signals** for {sym}. "
                "Edge metrics require at least 5 resolved signals per group (5 trading days per signal). "
                f"Currently {_pending_n} pending — check back after {max(0, 5 - _resolved_n)} more trading days.")

        # Show the pending pipeline so user can see data is being collected
        if _all_sym:
            st.markdown("#### 📥 Signal Pipeline (pending resolution)")
            _pipe_rows = []
            for e in sorted(_all_sym, key=lambda x: x.get("ts",""), reverse=True)[:20]:
                sig_date = e.get("ts","")[:10]
                elapsed  = 0
                try:
                    elapsed = int(np.busday_count(sig_date, datetime.now().date().isoformat()))
                except Exception:
                    pass
                _pipe_rows.append({
                    "Date":     sig_date,
                    "Bias":     e.get("bias", "—"),
                    "Score":    f"{e.get('raw_score', 0):+.3f}",
                    "IV%":      f"{e.get('atm_iv_pct', 0):.1f}",
                    "IVR":      f"{e.get('ivr', 0):.0f}",
                    "PCR":      f"{e.get('pcr', 0):.2f}",
                    "Flow":     f"{e.get('flow_score', 0):+.3f}",
                    "EM ₹":     f"{e.get('expected_move', 0):.0f}",
                    "Days Ago": str(elapsed),
                    "Status":   "✅ Resolved" if e.get("resolved") else f"⏳ {elapsed}/5d",
                })
            if _pipe_rows:
                _pipe_df = pd.DataFrame(_pipe_rows)
                st.dataframe(_pipe_df, use_container_width=True, hide_index=True)
        st.stop()

    # ── Compute edge metrics ──────────────────────────────────────────────────
    _edge_groups = _compute_edge_metrics(_edge_log, sym)

    if not _edge_groups:
        st.warning("Not enough resolved signals to compute edge metrics. Check back after more loads.")
    else:
        # ── HORIZON SELECTOR ────────────────────────────────────────────────
        _hz = st.radio("Analysis horizon", ["1d", "2d", "3d", "5d"],
                       index=0, horizontal=True,
                       help="Forward horizon used for return and accuracy metrics")

        st.divider()

        # ── EDGE SUMMARY TABLE ───────────────────────────────────────────────
        st.markdown("#### 📊 Edge Metrics by Signal Group")
        st.caption("Dir Acc = % signals where direction was correct · Move/IV = |actual move| / expected move · "
                   "IV Δ = avg change in ATM IV next day · Skew Δ = avg skew change · "
                   "→MP = % price moved toward max pain")

        _tbl_rows = []
        for grp, m in _edge_groups.items():
            if m is None:
                continue
            # Colour-code edge type
            edge_str = m["edge"]
            _tbl_rows.append({
                "Signal Group":  grp,
                "N":             m["n"],
                f"Avg Ret ({_hz})": f"{m['avg_ret']:+.2f}%",
                "Dir Acc":       f"{m['dir_acc']:.0f}%",
                "Move/IV":       f"{m['move_iv']:.2f}×",
                ">IV Move":      f"{m['pct_gt_iv']:.0f}%",
                "IV Δ":          f"{m['iv_chg']:+.2f}pp",
                "Skew Δ":        f"{m['skew_chg']:+.3f}pp",
                "OI Δ (M)":      f"{m['oi_chg']:+.1f}",
                "→MaxPain":      f"{m['pct_toward_mp']:.0f}%",
                "Edge Type":     edge_str,
            })

        if _tbl_rows:
            _tbl_df = pd.DataFrame(_tbl_rows)

            def _edge_style(v):
                if "Directional" in str(v): return "color:#00d084;font-weight:700"
                if "Long Vol"    in str(v): return "color:#1e90ff;font-weight:700"
                if "Short Vol"   in str(v): return "color:#ff8c00;font-weight:700"
                if "Pinning"     in str(v): return "color:#9c27b0;font-weight:700"
                if "Skew"        in str(v): return "color:#7ec8e3;font-weight:700"
                if "Sell IV"     in str(v): return "color:#ff8c00"
                if "Buy IV"      in str(v): return "color:#1e90ff"
                if "No Edge"     in str(v): return "color:#555"
                return "color:#888"

            def _acc_style(v):
                try:
                    pct = float(str(v).replace("%",""))
                    if pct > 58: return "color:#00d084;font-weight:700"
                    if pct > 52: return "color:#ffb347"
                    if pct < 45: return "color:#ff3b3b"
                except Exception:
                    pass
                return "color:#888"

            def _ret_style(v):
                try:
                    val = float(str(v).replace("%","").replace("+",""))
                    if val > 0.1:  return "color:#00d084;font-weight:700"
                    if val < -0.1: return "color:#ff3b3b;font-weight:700"
                except Exception:
                    pass
                return "color:#888"

            styled_tbl = _tbl_df.style \
                .map(_edge_style, subset=["Edge Type"]) \
                .map(_acc_style,  subset=["Dir Acc"]) \
                .map(_ret_style,  subset=[f"Avg Ret ({_hz})"])
            st.dataframe(styled_tbl, use_container_width=True, hide_index=True)

        st.divider()

        # ── INTERPRETATION PANEL ──────────────────────────────────────────────
        st.markdown("#### 🔍 Edge Interpretation")

        # Get all-signals metrics as baseline
        _baseline = _edge_groups.get("📊 All Signals")
        if _baseline:
            _interp_cols = st.columns(2)

            with _interp_cols[0]:
                st.markdown("**What edge does the system have?**")
                _findings = []

                if _baseline["dir_acc"] > 55:
                    _findings.append(("✅ Directional Edge",
                                      f"Direction accuracy {_baseline['dir_acc']:.0f}% > 55% threshold. "
                                      "The signal correctly predicts direction more often than chance."))
                elif _baseline["dir_acc"] < 45:
                    _findings.append(("❌ Reverse Directional Signal",
                                      f"Direction accuracy only {_baseline['dir_acc']:.0f}%. "
                                      "Consider reversing the signal interpretation."))
                else:
                    _findings.append(("⚪ No Directional Edge",
                                      f"Direction accuracy {_baseline['dir_acc']:.0f}% — not significantly above 50%."))

                if _baseline["move_iv"] > 1.05:
                    _findings.append(("✅ Long Vol Edge",
                                      f"Actual moves average {_baseline['move_iv']:.2f}× the implied move. "
                                      "Options are systematically underpriced — buy vol strategies preferred."))
                elif _baseline["move_iv"] < 0.90:
                    _findings.append(("✅ Short Vol Edge",
                                      f"Actual moves average {_baseline['move_iv']:.2f}× the implied move. "
                                      "Options are systematically overpriced — sell vol strategies preferred."))

                if _baseline["pct_toward_mp"] > 60:
                    _findings.append(("✅ Max Pain Pinning",
                                      f"Price moves toward max pain {_baseline['pct_toward_mp']:.0f}% of the time. "
                                      "Strong gravitational pull — use max pain for target levels."))

                if abs(_baseline["iv_chg"]) > 0.5:
                    direction = "rises" if _baseline["iv_chg"] > 0 else "falls"
                    _findings.append(("✅ IV Edge",
                                      f"IV {direction} avg {abs(_baseline['iv_chg']):.1f}pp after signal. "
                                      f"{'Buy IV' if _baseline['iv_chg'] > 0 else 'Sell IV'} strategies have positive expected IV change."))

                if abs(_baseline["skew_chg"]) > 0.3:
                    _findings.append(("✅ Skew Edge",
                                      f"Skew changes avg {_baseline['skew_chg']:+.2f}pp after signal. "
                                      "Skew-based strategies (risk reversals, skew trades) may have edge."))

                if not _findings:
                    _findings.append(("⚪ No Edge Detected",
                                      "No statistically meaningful edge found in any dimension. "
                                      "Collect more signals (need 20+ per group for confidence)."))

                for title, desc in _findings:
                    st.markdown(f"**{title}**")
                    st.caption(desc)
                    st.markdown("")

            with _interp_cols[1]:
                st.markdown("**Best-performing signal conditions:**")

                # Find group with highest dir accuracy
                _best_dir = max(
                    [(g, m["dir_acc"]) for g, m in _edge_groups.items() if m and m["n"] >= 3],
                    key=lambda x: x[1], default=(None, 0))
                # Find group with best return
                _best_ret = max(
                    [(g, m["avg_ret"]) for g, m in _edge_groups.items() if m and m["n"] >= 3],
                    key=lambda x: abs(x[1]), default=(None, 0))
                # Find highest pinning
                _best_pin = max(
                    [(g, m["pct_toward_mp"]) for g, m in _edge_groups.items() if m and m["n"] >= 3],
                    key=lambda x: x[1], default=(None, 0))

                if _best_dir[0]:
                    st.markdown(f"🎯 **Best direction:** {_best_dir[0]} — {_best_dir[1]:.0f}% accuracy")
                if _best_ret[0]:
                    st.markdown(f"📈 **Best return:** {_best_ret[0]} — avg {_best_ret[1]:+.2f}%/{_hz}")
                if _best_pin[0]:
                    st.markdown(f"📍 **Best pinning:** {_best_pin[0]} — {_best_pin[1]:.0f}% toward max pain")

                st.divider()
                st.markdown("**Confidence guide:**")
                st.caption("< 5 signals: insufficient · 5–20: indicative · 20–50: moderate · 50+: high confidence")

                # Show confidence bar
                _conf_pct = min(100, int(_resolved_n / 50 * 100))
                _conf_col = "#00d084" if _conf_pct >= 60 else ("#ffb347" if _conf_pct >= 20 else "#ff3b3b")
                st.markdown(f"""
<div style="font-family:'IBM Plex Mono',monospace;margin:6px 0;">
  <div style="color:#555;font-size:0.74rem;margin-bottom:3px;">
    Confidence: {_resolved_n} resolved / 50 recommended
  </div>
  <div style="background:#1a1a1a;height:10px;border-radius:3px;overflow:hidden;">
    <div style="width:{_conf_pct}%;height:100%;background:{_conf_col};border-radius:3px;"></div>
  </div>
</div>""", unsafe_allow_html=True)

        st.divider()

        # ── RAW RESOLVED SIGNAL DETAIL TABLE ────────────────────────────────
        with st.expander("📋 Raw Resolved Signal Log", expanded=False):
            _res_rows = []
            for e in sorted([x for x in _all_sym if x.get("resolved")],
                            key=lambda x: x.get("ts",""), reverse=True)[:100]:
                r1 = e.get("fwd_ret_1d")
                r5 = e.get("fwd_ret_5d")
                sp0 = e.get("spot", 0.0)
                em  = e.get("expected_move", 0.0)
                sp1 = e.get("fwd_spot_1d")
                move_iv = abs(sp1 - sp0) / em if sp1 and sp0 > 0 and em > 0 else None

                _bias = e.get("bias", "—")
                _dir_ok = None
                if r1 is not None:
                    if "BULLISH" in _bias:   _dir_ok = "✅" if r1 > 0 else "❌"
                    elif "BEARISH" in _bias: _dir_ok = "✅" if r1 < 0 else "❌"
                    else:                    _dir_ok = "—"

                _res_rows.append({
                    "Date":      e.get("ts","")[:10],
                    "Bias":      _bias,
                    "Score":     f"{e.get('raw_score',0):+.3f}",
                    "IV%":       f"{e.get('atm_iv_pct',0):.1f}",
                    "PCR":       f"{e.get('pcr',0):.2f}",
                    "Flow":      f"{e.get('flow_score',0):+.3f}",
                    "EM ₹":      f"{em:.0f}",
                    "Ret 1d":    f"{r1*100:+.2f}%" if r1 else "—",
                    "Ret 5d":    f"{r5*100:+.2f}%" if r5 else "—",
                    "Move/IV":   f"{move_iv:.2f}×" if move_iv else "—",
                    "Dir ✓":     _dir_ok or "—",
                })

            if _res_rows:
                _res_df = pd.DataFrame(_res_rows)

                def _dir_ok_style(v):
                    if v == "✅": return "color:#00d084;font-weight:700"
                    if v == "❌": return "color:#ff3b3b;font-weight:700"
                    return "color:#555"

                st.dataframe(_res_df.style.map(_dir_ok_style, subset=["Dir ✓"]),
                             use_container_width=True, hide_index=True)

        # ── VISUAL: Direction accuracy over time ──────────────────────────
        with st.expander("📈 Direction Accuracy — Rolling 20-Signal Window", expanded=False):
            _all_bias_signals = [e for e in _all_sym
                                 if e.get("resolved") and e.get("fwd_ret_1d") is not None
                                 and e.get("bias","NEUTRAL") != "NEUTRAL"]
            if len(_all_bias_signals) >= 5:
                _roll_acc = []
                for i in range(len(_all_bias_signals)):
                    window = _all_bias_signals[max(0, i-19):i+1]
                    correct = []
                    for e in window:
                        r = e.get("fwd_ret_1d", 0)
                        b = e.get("bias","")
                        if "BULLISH" in b:   correct.append(1 if r > 0 else 0)
                        elif "BEARISH" in b: correct.append(1 if r < 0 else 0)
                    _roll_acc.append({
                        "Signal #": i + 1,
                        "Date":     e.get("ts","")[:10],
                        "Rolling Accuracy": round(float(np.mean(correct)) * 100, 1) if correct else 50.0
                    })
                _acc_df = pd.DataFrame(_roll_acc)
                fig_acc = go.Figure()
                fig_acc.add_trace(go.Scatter(
                    x=_acc_df["Signal #"], y=_acc_df["Rolling Accuracy"],
                    mode="lines+markers", name="Rolling Accuracy",
                    line=dict(color="#ff8c00", width=2),
                    marker=dict(size=5)
                ))
                fig_acc.add_hline(y=55, line=dict(color="#00d084", dash="dash", width=1),
                                  annotation_text="55% edge threshold")
                fig_acc.add_hline(y=50, line=dict(color="#555", dash="dot", width=1),
                                  annotation_text="random (50%)")
                fig_acc.update_layout(
                    title="Rolling 20-Signal Direction Accuracy",
                    height=280, plot_bgcolor="#000", paper_bgcolor="#000",
                    font=dict(color="#e8e8e8", family="IBM Plex Mono", size=9),
                    xaxis=dict(title="Signal #", gridcolor="#111"),
                    yaxis=dict(title="Accuracy %", gridcolor="#111", range=[30, 80]),
                    margin=dict(t=40, b=10)
                )
                st.plotly_chart(fig_acc, use_container_width=True)
            else:
                st.caption("Need at least 5 resolved directional signals to plot rolling accuracy.")

        # ── VISUAL: Move vs Implied distribution ─────────────────────────
        with st.expander("📊 Actual Move vs Implied Move Distribution", expanded=False):
            _move_data = []
            for e in [x for x in _all_sym if x.get("resolved")]:
                sp0 = e.get("spot", 0.0)
                sp1 = e.get("fwd_spot_1d")
                em  = e.get("expected_move", 0.0)
                if sp0 > 0 and sp1 and em > 0:
                    _move_data.append(abs(sp1 - sp0) / em)
            if len(_move_data) >= 5:
                fig_mv = go.Figure()
                fig_mv.add_trace(go.Histogram(
                    x=_move_data, nbinsx=20,
                    marker_color="#1e90ff", opacity=0.8, name="Move/IV ratio"
                ))
                fig_mv.add_vline(x=1.0, line=dict(color="#ff8c00", dash="dash", width=2),
                                 annotation_text="IV = actual move")
                _mean_move = float(np.mean(_move_data))
                fig_mv.add_vline(x=_mean_move, line=dict(color="#00d084", dash="dot", width=1.5),
                                 annotation_text=f"avg {_mean_move:.2f}×")
                fig_mv.update_layout(
                    title=f"Distribution of |Actual 1d Move| / Expected Move  (n={len(_move_data)})",
                    height=260, plot_bgcolor="#000", paper_bgcolor="#000",
                    font=dict(color="#e8e8e8", family="IBM Plex Mono", size=9),
                    xaxis=dict(title="Actual / Implied", gridcolor="#111"),
                    yaxis=dict(title="Count", gridcolor="#111"),
                    margin=dict(t=40, b=10)
                )
                st.plotly_chart(fig_mv, use_container_width=True)
                _pct_over = float(np.mean([1 if m > 1 else 0 for m in _move_data])) * 100
                _vol_edge = "**Long Vol Edge** — options underpriced" if _mean_move > 1.05 else \
                            "**Short Vol Edge** — options overpriced" if _mean_move < 0.90 else \
                            "**No Vol Edge** — moves ≈ implied"
                st.markdown(f"Avg move = {_mean_move:.2f}× implied · {_pct_over:.0f}% of moves exceeded implied · {_vol_edge}")
            else:
                st.caption("Need at least 5 resolved signals to plot distribution.")

        # ── METHODOLOGY NOTE ─────────────────────────────────────────────
        with st.expander("ℹ️ Edge Audit Methodology", expanded=False):
            st.markdown("""
**What this module measures:**

Every time you click ⚡ LOAD OPTIONS INTEL, a full snapshot is recorded with:
spot, IV, IVR, IV/HV ratio, PCR, OI total, OI skew, max pain distance, skew, term structure slope,
flow score, positioning score, vol regime score, trend score, and expected move.

**Forward outcomes are resolved 5 trading days later** using OHLCV close prices:
- 1d/2d/3d/5d forward log return
- Absolute move vs expected move
- IV change, skew change, OI change
- Whether price moved toward max pain

**Signal groups** are formed by direction bias, IV regime, PCR level, proximity to max pain,
flow magnitude, and signal alignment (flow + positioning + vol all pointing same direction).

**Edge thresholds (interpretation rules):**

| Metric | Threshold | Edge |
|---|---|---|
| Direction accuracy | > 55% | Directional Edge |
| Avg move / implied | > 1.05× | Long Vol Edge |
| Avg move / implied | < 0.90× | Short Vol Edge |
| Avg IV change | > +0.5pp | Buy IV Edge |
| Avg IV change | < -0.5pp | Sell IV Edge |
| % toward max pain | > 60% | Pinning Edge |
| Avg skew change | > 0.3pp | Skew Edge |

**Confidence:** 5+ signals = indicative, 20+ = moderate, 50+ = high confidence.
Collect signals daily for 2–3 months for statistically meaningful results.
""")

        # ══════════════════════════════════════════════════════════════
        # STEP 8 — PROBABILITY CALIBRATION RESULTS
        # Predicted probability ≈ Actual probability
        # ══════════════════════════════════════════════════════════════
        st.divider()
        st.markdown("#### 🎯 Step 8 — Probability Calibration")
        st.caption("Does the model's predicted P(↑) match the actual frequency of up-moves? "
                   "A well-calibrated model has: when it says 70% bullish, the market goes up ~70% of the time.")

        # Pull raw/return history from session state (symbol-namespaced)
        _s8_raw   = _get_hist("_calib_raw_score_hist", sym)
        _s8_ret   = _get_hist("_calib_realised_ret_hist", sym)
        _s8_sharp = _calib("logistic_sharpness")
        n_s8      = min(len(_s8_raw), len(_s8_ret))

        c8a, c8b, c8c, c8d = st.columns(4)
        c8a.metric("Observations",    str(n_s8),
                   help="Paired (raw_score, realised_return) observations used for calibration")
        c8b.metric("Logistic k",       f"{_s8_sharp:.3f}",
                   help="Calibrated sharpness. Default=4.0. Higher=steeper prob curve. "
                        "Learned from correlation between raw_score and forward returns.")
        c8c.metric("Min needed",       str(_dynamic_min_obs()),
                   help=f"Minimum {_dynamic_min_obs()} observations before sharpness overrides cold-start prior")
        _s8_status = ("🟢 LIVE — using learned k" if n_s8 >= _dynamic_min_obs()
                      else f"🟡 WARMING UP — {n_s8}/{_dynamic_min_obs()}")
        c8d.metric("Status",  _s8_status)

        if n_s8 >= 5:
            # ── Reliability Diagram (Calibration Plot) ────────────────────
            with st.expander("📊 Reliability Diagram — Predicted vs Actual Probability", expanded=True):
                st.caption("Each point = one bucket of signals with similar predicted P(↑). "
                           "Perfect calibration = all points on the diagonal.")

                # Convert raw scores to predicted probs using current sharpness
                _raw_arr = np.array(_s8_raw[-min(n_s8, 200):])
                _ret_arr = np.array(_s8_ret[-min(n_s8, 200):])
                _pred_pu = 1.0 / (1.0 + np.exp(-_s8_sharp * _raw_arr))
                _actual_up = (_ret_arr > 0).astype(float)

                # Bin into spec-defined 0.05-wide buckets
                # 0.50–0.55, 0.55–0.60, 0.60–0.65, 0.65–0.70, 0.70–0.75, 0.75–0.80, 0.80–1.00
                # Plus bearish mirror: 0.20–0.25, 0.25–0.30, 0.30–0.35, 0.35–0.40, 0.40–0.45, 0.45–0.50
                _bins = [0.0, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
                         0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 1.0]
                _bin_labels = ["<20%","20-25%","25-30%","30-35%","35-40%","40-45%","45-50%",
                               "50-55%","55-60%","60-65%","65-70%","70-75%","75-80%",">80%"]
                _rel_rows = []
                for i in range(len(_bins)-1):
                    mask = (_pred_pu >= _bins[i]) & (_pred_pu < _bins[i+1])
                    n_bin = int(mask.sum())
                    if n_bin < 2:
                        continue
                    pred_mean   = float(_pred_pu[mask].mean())
                    actual_freq = float(_actual_up[mask].mean())
                    _move_vs_iv_h = _get_hist("_calib_move_vs_iv_hist", sym)
                    _mvi_subset   = float(np.mean(_move_vs_iv_h[-n_bin:])) if len(_move_vs_iv_h) >= n_bin else 0.0
                    _rel_rows.append({
                        "Predicted P(↑)": _bin_labels[i],
                        "Avg Predicted":  round(pred_mean * 100, 1),
                        "Actual Up %":    round(actual_freq * 100, 1),
                        "N":              n_bin,
                        "Avg Move/IV":    round(_mvi_subset, 2) if _mvi_subset else "—",
                        "Gap":            round((actual_freq - pred_mean) * 100, 1),
                    })

                if _rel_rows:
                    _rel_df = pd.DataFrame(_rel_rows)

                    # Plot reliability diagram
                    fig_rel = go.Figure()
                    # Perfect calibration line
                    fig_rel.add_trace(go.Scatter(
                        x=[0, 100], y=[0, 100],
                        mode="lines", name="Perfect calibration",
                        line=dict(color="#555", dash="dash", width=1)
                    ))
                    # Actual calibration points
                    _col_pts = ["#ff3b3b" if abs(r["Gap"]) > 10 else
                                "#ffb347" if abs(r["Gap"]) > 5 else "#00d084"
                                for r in _rel_rows]
                    fig_rel.add_trace(go.Scatter(
                        x=[r["Avg Predicted"] for r in _rel_rows],
                        y=[r["Actual Up %"] for r in _rel_rows],
                        mode="markers+lines",
                        name="Model calibration",
                        marker=dict(size=[max(8, r["N"]*2) for r in _rel_rows],
                                   color=_col_pts, line=dict(color="#fff", width=1)),
                        line=dict(color="#ff8c00", width=1.5),
                        text=[f"N={r['N']}, Gap={r['Gap']:+.1f}pp" for r in _rel_rows],
                        hovertemplate="%{text}<extra></extra>"
                    ))
                    fig_rel.update_layout(
                        height=300, plot_bgcolor="#000", paper_bgcolor="#000",
                        font=dict(color="#e8e8e8", family="IBM Plex Mono", size=9),
                        xaxis=dict(title="Predicted P(↑) %", range=[0,100], gridcolor="#111"),
                        yaxis=dict(title="Actual Up %",       range=[0,100], gridcolor="#111"),
                        margin=dict(t=20, b=10),
                        legend=dict(orientation="h", y=1.08)
                    )
                    st.plotly_chart(fig_rel, use_container_width=True)

                    # Table with gap highlighting
                    def _gap_style(v):
                        try:
                            g = float(str(v).replace("+",""))
                            if abs(g) > 10: return "color:#ff3b3b;font-weight:700"
                            if abs(g) > 5:  return "color:#ffb347"
                            return "color:#00d084"
                        except Exception:
                            return ""
                    def _mvi_style(v):
                        try:
                            m = float(str(v))
                            if m > 1.05: return "color:#1e90ff;font-weight:700"   # long vol edge
                            if m < 0.90: return "color:#ff8c00;font-weight:700"   # short vol edge
                            return "color:#888"
                        except Exception:
                            return ""
                    st.dataframe(
                        _rel_df.style
                            .map(_gap_style, subset=["Gap"])
                            .map(_mvi_style, subset=["Avg Move/IV"]),
                        use_container_width=True, hide_index=True
                    )
                    # Calibration verdict
                    _max_gap = max(abs(r["Gap"]) for r in _rel_rows)
                    if _max_gap <= 5:
                        st.success(f"✅ Well-calibrated — max gap {_max_gap:.1f}pp. "
                                   "Predicted probabilities closely match actual frequencies.")
                    elif _max_gap <= 10:
                        st.warning(f"⚠️ Moderate miscalibration — max gap {_max_gap:.1f}pp. "
                                   "Adjust logistic sharpness (k) or collect more data.")
                    else:
                        st.error(f"❌ Poor calibration — max gap {_max_gap:.1f}pp. "
                                 "The model over/underestimates probabilities significantly. "
                                 "More data needed for auto-calibration to correct this.")

            # ── Brier Score ───────────────────────────────────────────────
            with st.expander("📐 Brier Score — Probability Forecast Accuracy", expanded=False):
                st.caption("Brier Score = mean((predicted_prob − actual_outcome)²). "
                           "Lower is better. Random model = 0.25. Perfect = 0.00.")

                _brier = float(np.mean((_pred_pu - _actual_up) ** 2))
                _brier_skill = 1.0 - _brier / 0.25   # skill vs random baseline
                _brier_col = ("#00d084" if _brier < 0.20
                              else "#ffb347" if _brier < 0.23
                              else "#ff3b3b")

                bc1, bc2, bc3 = st.columns(3)
                bc1.metric("Brier Score",  f"{_brier:.4f}",
                           delta=f"vs random: {0.25:.4f}",
                           delta_color="inverse",
                           help="Mean squared error of probability forecasts. Lower = better. Random = 0.25")
                bc2.metric("Brier Skill",  f"{_brier_skill*100:.1f}%",
                           help="% improvement over a random (50/50) baseline. Positive = model adds value.")
                _log_score = float(-np.mean(
                    _actual_up * np.log(_pred_pu + 1e-9) +
                    (1 - _actual_up) * np.log(1 - _pred_pu + 1e-9)
                ))
                bc3.metric("Log Score",    f"{_log_score:.4f}",
                           help="Binary cross-entropy. Lower is better. Random = ln(2) ≈ 0.693")

                st.markdown(f"""
<div style="font-family:'IBM Plex Mono',monospace;font-size:0.78rem;color:#555;margin-top:6px;">
  <span style="color:{_brier_col};font-weight:700;">Brier Score {_brier:.4f}</span>
  &nbsp;·&nbsp; n={n_s8} observations &nbsp;·&nbsp; k={_s8_sharp:.3f}
  &nbsp;·&nbsp;
  {'✅ Better than random' if _brier_skill > 0 else '❌ Worse than random'}
</div>""", unsafe_allow_html=True)

            # ── Sharpness Calibration History ─────────────────────────────
            with st.expander("📈 Logistic Sharpness k — Learning Curve", expanded=False):
                st.caption("Shows how the calibrated sharpness k has evolved. "
                           "Starts at prior (4.0), updated as real outcomes accumulate.")

                # Reconstruct what k would have been at each step using OLS
                if n_s8 >= 10:
                    _k_history = []
                    for _i in range(10, n_s8 + 1):
                        _x = np.array(_s8_raw[:_i])
                        _y = np.array(_s8_ret[:_i])
                        _xx = float(np.dot(_x, _x))
                        if _xx > 1e-9:
                            _k_ols = float(np.dot(_x, _y)) / _xx
                            # Shrink toward prior
                            _shrink = max(0.0, 1.0 - (_i - _dynamic_min_obs()) / _CALIB_WINDOW)
                            _k_blended = (1.0 - _shrink) * _k_ols + _shrink * 4.0
                            _k_history.append(round(max(0.5, min(20.0, _k_blended)), 4))
                        else:
                            _k_history.append(4.0)

                    fig_k = go.Figure()
                    fig_k.add_hline(y=4.0, line=dict(color="#555", dash="dot", width=1),
                                    annotation_text="Prior k=4.0")
                    fig_k.add_trace(go.Scatter(
                        x=list(range(10, n_s8 + 1)), y=_k_history,
                        mode="lines", name="Calibrated k",
                        line=dict(color="#ff8c00", width=2)
                    ))
                    fig_k.add_hline(y=_s8_sharp,
                                    line=dict(color="#00d084", dash="dash", width=1.5),
                                    annotation_text=f"Current k={_s8_sharp:.3f}")
                    fig_k.update_layout(
                        title="Logistic Sharpness k Over Time",
                        height=240, plot_bgcolor="#000", paper_bgcolor="#000",
                        font=dict(color="#e8e8e8", family="IBM Plex Mono", size=9),
                        xaxis=dict(title="Observations", gridcolor="#111"),
                        yaxis=dict(title="k", gridcolor="#111"),
                        margin=dict(t=40, b=10)
                    )
                    st.plotly_chart(fig_k, use_container_width=True)

                    _k_interp = ("Sharpness above prior → model signals are more decisive than cold-start" if _s8_sharp > 4.5
                                 else "Sharpness below prior → signals are weaker than expected, wider prob spread" if _s8_sharp < 3.5
                                 else "Sharpness near prior → model behaving as expected")
                    st.caption(f"Current k={_s8_sharp:.3f} · {_k_interp}")
                else:
                    st.caption(f"Need 10+ observations to show learning curve. Currently {n_s8}.")

            # ── Predicted vs Actual Scatter ───────────────────────────────
            with st.expander("🔍 Predicted Probability vs Actual 1d Return", expanded=False):
                st.caption("Each point is one signal. X = model's predicted P(↑), Y = actual 1-day log return. "
                           "Good calibration → positive correlation, points trending upward left-to-right.")

                fig_scat = go.Figure()
                _colors_scat = ["#00d084" if r > 0 else "#ff3b3b" for r in _ret_arr]
                fig_scat.add_trace(go.Scatter(
                    x=(_pred_pu * 100).tolist(),
                    y=(_ret_arr * 100).tolist(),
                    mode="markers",
                    marker=dict(size=6, color=_colors_scat, opacity=0.7),
                    name="Signals",
                    hovertemplate="P(↑): %{x:.1f}%<br>Ret: %{y:.2f}%<extra></extra>"
                ))
                # Trend line
                if len(_pred_pu) >= 5:
                    _z = np.polyfit(_pred_pu, _ret_arr * 100, 1)
                    _px = np.linspace(_pred_pu.min(), _pred_pu.max(), 50)
                    _py = np.polyval(_z, _px)
                    fig_scat.add_trace(go.Scatter(
                        x=(_px * 100).tolist(), y=_py.tolist(),
                        mode="lines", name="Trend",
                        line=dict(color="#ff8c00", width=1.5, dash="dot")
                    ))
                    _corr_raw = np.corrcoef(_pred_pu, _ret_arr)[0, 1]
                    _corr = float(_corr_raw) if not (isinstance(_corr_raw, float) and math.isnan(_corr_raw)) and not np.isnan(_corr_raw) else 0.0
                    _corr_col = "#00d084" if _corr > 0.1 else "#ff3b3b" if _corr < -0.1 else "#888"

                fig_scat.add_hline(y=0, line=dict(color="#333", width=1))
                fig_scat.add_vline(x=50, line=dict(color="#333", width=1))
                fig_scat.update_layout(
                    height=280, plot_bgcolor="#000", paper_bgcolor="#000",
                    font=dict(color="#e8e8e8", family="IBM Plex Mono", size=9),
                    xaxis=dict(title="Predicted P(↑) %", gridcolor="#111", range=[20, 80]),
                    yaxis=dict(title="Actual 1d Return %", gridcolor="#111"),
                    margin=dict(t=20, b=10)
                )
                st.plotly_chart(fig_scat, use_container_width=True)

                if len(_pred_pu) >= 5:
                    st.markdown(
                        f"Correlation: <span style='color:{_corr_col};font-weight:700;font-family:monospace;'>"
                        f"{_corr:+.3f}</span> "
                        f"({'positive — model direction aligns with returns' if _corr > 0.05 else 'negative — model may need recalibration' if _corr < -0.05 else 'near zero — no directional correlation yet'})",
                        unsafe_allow_html=True
                    )

        else:
            _sym_bs = st.session_state.get("opt_symbol", "").upper()
            _guard_keys = [k for k in st.session_state if k.startswith(f"_bootstrap_done:{_sym_bs}:")]
            _bootstrapped = len(_guard_keys) > 0
            if _bootstrapped:
                st.info(
                    f"ℹ️ Bootstrap calibration ran from OHLCV history and seeded {n_s8} observations. "
                    f"Needs 5 to show calibration charts. "
                    f"Live Load clicks will add real forward-test pairs on top of the historical seed."
                )
            else:
                st.info(
                    f"ℹ️ Probability calibration requires at least 5 paired (score, return) observations. "
                    f"Currently {n_s8}. These accumulate automatically with daily use — "
                    "each Load records a snapshot and resolves it after 4 trading days. "
                    "Bootstrap from OHLCV history will run on next Load if OHLCV data is available."
                )

# ══════════════════════════════════════════════════════════════
# TAB — PROBABILITY ENGINE
# Full probability-based decision panel:
#   Signals → Score → Probability → Implied Move comparison
#   → Edge → EV → Kelly → Strategy recommendation
# All sections are self-contained and work with data already
# computed in the render section above (prob_score, strat_recs,
# chain_df, oi_d, atm_iv, hv20, dte, spot, etc.)
# ══════════════════════════════════════════════════════════════
with t_prob:
    # Read from session_state only - never from module-level vars
    _pe_loaded = st.session_state.get("opt_loaded", False)
    _pe_ps     = st.session_state.get("opt_prob_score", {})
    _pe_spot   = float(st.session_state.get("opt_spot", 0.0))
    _pe_iv     = float(st.session_state.get("opt_atm_iv", 0.20))
    _pe_hv     = float(st.session_state.get("opt_hv20") or 0.15)
    _pe_dte    = max(int(st.session_state.get("opt_dte", 7)), 1)
    _pe_sym    = str(st.session_state.get("opt_symbol", ""))

    st.markdown("### Probability Engine")

    if not _pe_loaded or _pe_spot <= 0:
        st.warning("Load a symbol first — press LOAD OPTIONS INTEL in the sidebar.")
    else:

        # Pull values from prob_score
        _pe_pu   = float(_pe_ps.get("prob_up",   0.50))
        _pe_pd   = float(_pe_ps.get("prob_down", 0.50))
        _pe_rs   = float(_pe_ps.get("raw_score", 0.0))
        _pe_edge = _pe_pu - 0.50
        _pe_str  = str(_pe_ps.get("signal_strength", "No Edge"))
        _pe_iv_pct = _pe_iv * 100.0
        _pe_hv_pct = _pe_hv * 100.0
        _pe_ivhv   = _pe_iv / (_pe_hv + 1e-9)

        # Implied move: spot * IV * sqrt(DTE/252)
        _pe_impl_pct = _pe_iv * math.sqrt(_pe_dte / 252.0) * 100.0
        _pe_impl_rs  = _pe_spot * _pe_impl_pct / 100.0

        # Colours
        _pe_pu_col  = "#00d084" if _pe_pu >= 0.60 else "#ff3b3b" if _pe_pu <= 0.40 else "#ffb347"
        _pe_edg_col = "#00d084" if _pe_edge > 0.05 else "#ff3b3b" if _pe_edge < -0.05 else "#888"
        _pe_edg_lbl = "Bullish Edge" if _pe_edge > 0.05 else "Bearish Edge" if _pe_edge < -0.05 else "Neutral"

        # ── SECTION 1: PROBABILITIES ──────────────────────────────────────────────
        st.markdown("---")
        st.markdown("#### Section 1 — Model Probabilities")

        _s1a, _s1b, _s1c, _s1d = st.columns(4)
        _s1a.metric("Prob Up",       f"{_pe_pu*100:.1f}%",
                    delta=f"score {_pe_rs:+.3f}", delta_color="normal")
        _s1b.metric("Prob Down",     f"{_pe_pd*100:.1f}%")
        _s1c.metric("Direction Edge",f"{_pe_edge*100:+.1f}pp")
        _s1d.metric("Signal",        _pe_str)

        # Simple progress bars using st.progress (always works)
        st.caption(f"Bull {_pe_pu*100:.0f}% vs Bear {_pe_pd*100:.0f}% | {_pe_edg_lbl}")
        st.progress(min(int(_pe_pu * 100), 100),
                    text=f"Prob Up: {_pe_pu*100:.1f}%  |  Edge: {_pe_edge*100:+.1f}pp")

        with st.expander("How to read probabilities", expanded=False):
            st.markdown("""
| Prob Up | Meaning |
|---------|---------|
| > 65% | Strong Bullish |
| 55-65% | Mild Bullish |
| 45-55% | Neutral — vol trade only |
| 35-45% | Mild Bearish |
| < 35% | Strong Bearish |
        """)

    # ── SECTION 2: MOVE COMPARISON ────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Section 2 — Move vs Implied Move")

    # Calibration histories
    _sym_pfx   = _pe_sym.upper()
    _prob_hist = (st.session_state.get(f"{_sym_pfx}:_calib_prob_up_hist",   []) or
                  st.session_state.get("_calib_prob_up_hist",   []))
    _act_hist  = (st.session_state.get(f"{_sym_pfx}:_calib_actual_up_hist", []) or
                  st.session_state.get("_calib_actual_up_hist", []))
    _sc_hist   = (st.session_state.get(f"{_sym_pfx}:_calib_raw_score_hist", []) or
                  st.session_state.get("_calib_raw_score_hist",  []))
    _ret_hist  = (st.session_state.get(f"{_sym_pfx}:_calib_realised_ret_hist", []) or
                  st.session_state.get("_calib_realised_ret_hist", []))
    _mvi_hist  = (st.session_state.get(f"{_sym_pfx}:_calib_move_vs_iv_hist", []) or
                  st.session_state.get("_calib_move_vs_iv_hist", []))
    _n_obs     = min(len(_prob_hist), len(_act_hist))

    # Model move from score buckets (data-driven) or fallback to implied
    _model_move_pct  = None
    _buckets = [(0.00,0.25,"0-0.25"),(0.25,0.50,"0.25-0.50"),
                (0.50,0.75,"0.50-0.75"),(0.75,1.00,"0.75-1.00"),
                (1.00,1.50,"1.00-1.50"),(1.50,99.0,"1.50+")]

    if len(_sc_hist) >= 5 and len(_ret_hist) >= 5:
        _sc_arr = np.array(_sc_hist[-min(len(_sc_hist),len(_ret_hist)):], dtype=float)
        _rt_arr = np.array(_ret_hist[-min(len(_sc_hist),len(_ret_hist)):], dtype=float)
        _mv_arr = np.abs(np.exp(_rt_arr) - 1.0) * 100.0
        _abs_rs = abs(_pe_rs)
        for _blo, _bhi, _ in _buckets:
            _msk = (np.abs(_sc_arr) >= _blo) & (np.abs(_sc_arr) < _bhi)
            if _msk.sum() > 0 and _blo <= _abs_rs < _bhi:
                _model_move_pct = float(np.mean(_mv_arr[_msk]))
                break

    _mdl_pct = _model_move_pct if _model_move_pct else _pe_impl_pct
    _mdl_rs  = _pe_spot * _mdl_pct / 100.0
    _me_pct  = _mdl_pct - _pe_impl_pct if _model_move_pct else None

    # Vol edge from avg move vs IV history
    _avg_mvi   = float(np.mean(_mvi_hist[-50:])) if len(_mvi_hist) >= 3 else None
    # ── Vol edge for panel display ──────────────────────────────────────────
    # PRIMARY: use IV/HV percentile history (same logic as compute_probabilistic_score).
    # This avoids the cold-start SELL bias from the old mvi_hist / model_move heuristic.
    _iv_hv_hist_disp = (_get_hist("_calib_iv_hv_ratio_hist") or
                        st.session_state.get("_calib_iv_hv_ratio_hist", []))
    if len(_iv_hv_hist_disp) >= 5:
        _iv_hv_now_disp = _pe_ivhv
        _iv_hv_pct_disp = _percentile_score(_iv_hv_hist_disp, _iv_hv_now_disp)
        if _iv_hv_pct_disp >= CFG["iv_hv_pct_sell"] / 100.0:
            _vol_edge = "SELL"
        elif _iv_hv_pct_disp <= CFG["iv_hv_pct_buy"] / 100.0:
            _vol_edge = "BUY"
        else:
            _vol_edge = "NEUTRAL"
    elif _avg_mvi is not None:
        _vol_edge = ("BUY"  if _avg_mvi > 1.05 else
                     "SELL" if _avg_mvi < 0.90 else
                     str(_pe_ps.get("vol_edge", "NEUTRAL")))
    else:
        _vol_edge = str(_pe_ps.get("vol_edge", "NEUTRAL"))
    # Override from move_edge only when strongly one-sided (>0.5pp) to avoid noise
    if _me_pct is not None:
        if _me_pct > 0.5:    _vol_edge = "BUY"
        elif _me_pct < -0.5: _vol_edge = "SELL"

    _m1, _m2, _m3, _m4 = st.columns(4)
    _m1.metric("Implied Move", f"Rs{_pe_impl_rs:,.0f}",
               delta=f"+-{_pe_impl_pct:.2f}%")
    _m2.metric("Model Move",   f"Rs{_mdl_rs:,.0f}",
               delta=f"+-{_mdl_pct:.2f}%", delta_color="off")
    _m3.metric("IV / HV",      f"{_pe_ivhv:.2f}x",
               delta=f"IV {_pe_iv_pct:.1f}% HV {_pe_hv_pct:.1f}%", delta_color="off")
    if _me_pct is not None:
        _me_lbl = "Underpriced" if _me_pct > 0.3 else "Overpriced" if _me_pct < -0.3 else "Fair"
        _m4.metric("Move Edge", f"{_me_pct:+.2f}pp", delta=_me_lbl, delta_color="off")
    elif _avg_mvi:
        _m4.metric("Avg Move/IV", f"{_avg_mvi:.2f}x",
                   delta="Long vol" if _avg_mvi > 1.05 else "Short vol" if _avg_mvi < 0.90 else "Fair",
                   delta_color="off")
    else:
        _m4.metric("Move Edge", "< 3 obs")

    st.caption(f"Observations for data-driven move: {len(_sc_hist)} score / {len(_ret_hist)} return / {len(_mvi_hist)} move-vs-IV")

    # ── SECTION 3: CALIBRATION TABLE ─────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Section 3 — Probability Calibration")
    st.caption(f"Observations: {_n_obs} (needs 5 to show table, 20+ for reliability)")

    if _n_obs >= 5:
        _ph = np.array(_prob_hist[-_n_obs:], dtype=float)
        _ah = np.array(_act_hist[-_n_obs:],  dtype=float)
        _cal_rows = []
        for _bl, _bh, _blbl in [(0.20,0.40,"0.20-0.40 Bearish"),(0.40,0.50,"0.40-0.50"),
                                  (0.50,0.60,"0.50-0.60"),(0.60,0.80,"0.60-0.80 Bullish"),
                                  (0.80,1.01,"0.80+ Strong Bull")]:
            _msk = (_ph >= _bl) & (_ph < _bh)
            _cnt = int(_msk.sum())
            if _cnt == 0: continue
            _ap = float(_ph[_msk].mean())
            _af = float(_ah[_msk].mean())
            _gap = _ap - _af
            _cal_rows.append({
                "Bucket": _blbl,
                "Model Prob": f"{_ap*100:.1f}%",
                "Actual Up":  f"{_af*100:.1f}%",
                "Count":      _cnt,
                "Gap":        f"{_gap*100:+.1f}pp"
            })
        if _cal_rows:
            st.dataframe(pd.DataFrame(_cal_rows), use_container_width=True, hide_index=True)
    else:
        st.info(f"Calibration table needs 5 resolved observations. Currently: {_n_obs}. "
                "Observations accumulate automatically — each Load + 4 trading days = 1 observation.")

    # ── SECTION 4: BRIER SCORE ────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Section 4 — Model Accuracy (Brier Score)")

    if _n_obs >= 5:
        _ph_b  = np.array(_prob_hist[-_n_obs:], dtype=float)
        _ah_b  = np.array(_act_hist[-_n_obs:],  dtype=float)
        _brier = float(np.mean((_ph_b - _ah_b) ** 2))
        _skill = 1.0 - _brier / 0.25
        _blbl  = ("Very Strong" if _brier < 0.12 else "Strong"    if _brier < 0.15 else
                  "Tradable"    if _brier < 0.18 else "Weak"       if _brier < 0.20 else
                  "Near Random" if _brier < 0.23 else "Below Random")

        _b1, _b2, _b3, _b4 = st.columns(4)
        _b1.metric("Brier Score",   f"{_brier:.4f}",
                   help="Lower = better. Random = 0.25, Perfect = 0.00")
        _b2.metric("Brier Skill",   f"{_skill*100:.1f}%",
                   help="% improvement over random baseline")
        _b3.metric("Assessment",    _blbl)
        _b4.metric("Observations",  str(_n_obs))

        _bar_pct = max(0, min(100, int((0.25 - _brier) / 0.25 * 100)))
        st.progress(_bar_pct,
                    text=f"Brier: {_brier:.4f} | Skill: {_skill*100:.0f}% | {_blbl}")
        st.caption("Reference: 0.25 = random  |  0.20 = weak  |  0.18 = tradable  |  0.15 = strong  |  0.12 = very strong")

        # FIX15: Calibration curve (prob bins vs outcome frequency)
        if _n_obs >= 20:
            with st.expander("📊 Calibration Curve (bins vs actual outcomes)", expanded=False):
                _pu_arr  = np.array(st.session_state.get("_calib_prob_up_hist", [])[-_n_obs:], dtype=float)
                _au_arr  = np.array(st.session_state.get("_calib_actual_up_hist", [])[-_n_obs:], dtype=float)
                _bins    = np.linspace(0.0, 1.0, 11)  # 10 equal-width bins
                _rows15  = []
                for _bi in range(len(_bins)-1):
                    _m = (_pu_arr >= _bins[_bi]) & (_pu_arr < _bins[_bi+1])
                    if _m.sum() < 2: continue
                    _pred_mid = float(_pu_arr[_m].mean())
                    _act_freq = float(_au_arr[_m].mean())
                    _gap15    = _act_freq - _pred_mid
                    _rows15.append({
                        "Prob Bin":        f"{int(_bins[_bi]*100)}-{int(_bins[_bi+1]*100)}%",
                        "Avg Predicted":   round(_pred_mid * 100, 1),
                        "Actual Up %":     round(_act_freq * 100, 1),
                        "Gap (pp)":        round(_gap15 * 100, 1),
                        "N":               int(_m.sum()),
                    })
                if _rows15:
                    st.dataframe(pd.DataFrame(_rows15), hide_index=True,
                                 use_container_width=True)
                    # Calibration ECE (Expected Calibration Error)
                    _total_binned = sum(r["N"] for r in _rows15)
                    _ece = sum(abs(r["Gap (pp)"]) * r["N"] for r in _rows15) / max(_total_binned, 1)
                    _ece_lbl = "🟢 Good" if _ece < 5 else "🟡 Fair" if _ece < 10 else "🔴 Poor"
                    st.metric("ECE (Expected Calibration Error)", f"{_ece:.1f}pp", help="Lower is better. <5pp = well-calibrated.")
                    st.caption(_ece_lbl + " | ECE measures average gap between predicted probabilities and actual frequencies")

    else:
        st.info(f"Brier score needs 5 observations (currently {_n_obs}). "
                "Each Load click followed by market close adds one observation after ~4 trading days.")

    # ── SECTION 5: EDGE + RECOMMENDATION ─────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Section 5 — Edge and Trade Recommendation")

    # ── FIX: vol_edge in this panel uses the SAME IV/HV percentile logic as
    # compute_probabilistic_score — NOT the model_move heuristic.
    # This is computed from the ranked history in session_state.
    _iv_hv_hist_pe = (_get_hist("_calib_iv_hv_ratio_hist") or
                      st.session_state.get("_calib_iv_hv_ratio_hist", []))
    if len(_iv_hv_hist_pe) >= 5:
        _iv_hv_now_pe = _pe_ivhv   # already computed as atm_iv / hv above
        _iv_hv_pct_pe = _percentile_score(_iv_hv_hist_pe, _iv_hv_now_pe)
        if _iv_hv_pct_pe >= _adaptive_threshold("iv_hv", CFG["iv_hv_pct_sell"], percentile=70.0) / 100.0:
            _vol_edge = "SELL"
        elif _iv_hv_pct_pe <= _adaptive_threshold("iv_hv", CFG["iv_hv_pct_buy"], percentile=30.0) / 100.0:
            _vol_edge = "BUY"
        else:
            _vol_edge = "NEUTRAL"
    # else: _vol_edge already computed above from mvi_hist / prob_score fallback

    # ── FIX: Narrow neutral zone to ±5pp (45-55%), not the original 40-60%.
    # A 20pp neutral band traps nearly every cold-start session → always Iron Condor.
    # 10pp band (45-55%) lets weak directional signals through properly.
    _is_bull    = _pe_pu > 0.55
    _is_bear    = _pe_pu < 0.45
    _is_neutral = not _is_bull and not _is_bear
    _long_vol   = _vol_edge == "BUY"
    _short_vol  = _vol_edge == "SELL"

    # ── FIX: PRIMARY recommendation from EV-ranked strategy list.
    # ev_rank_strategies() runs at LOAD time and returns strategies sorted by
    # composite EV score (MC POP × EV × DTE-align × Safety × Dir-align).
    # We use that result directly — no hardcoded rule tree.
    _strat_recs_panel = st.session_state.get("opt_strat_recs", [])
    if _strat_recs_panel:
        # Top strategy by composite EV score
        _top_rec   = _strat_recs_panel[0]
        _trade     = _top_rec.get("Strategy", "—")
        _trade_legs= _top_rec.get("Legs", "—")
        _trade_pop = float(_top_rec.get("pop", 0.5))
        _trade_ev  = float(_top_rec.get("ev_per_lot", 0))
        _trade_sc  = int(_top_rec.get("Score", 50))
        _trade_kelly = float(_top_rec.get("kelly", 0))
        _trade_rat = _top_rec.get("Rationale", "")
        # Also show top-3 for context
        _top3 = _strat_recs_panel[:3]
    else:
        # Fallback when LOAD has not been pressed yet
        if   _is_bull  and _long_vol:   _trade = "BUY CALLS"
        elif _is_bull  and _short_vol:  _trade = "BULL PUT SPREAD"
        elif _is_bull:                   _trade = "BUY CALLS (small)"
        elif _is_bear  and _long_vol:   _trade = "BUY PUTS"
        elif _is_bear  and _short_vol:  _trade = "BEAR CALL SPREAD"
        elif _is_bear:                   _trade = "BUY PUTS (small)"
        elif _is_neutral and _short_vol: _trade = "SHORT STRANGLE"
        elif _is_neutral and _long_vol:  _trade = "LONG STRADDLE"
        else:                            _trade = "WAIT — no clear edge"
        _trade_legs = "—"; _trade_pop = 0.5; _trade_ev = 0
        _trade_sc = 50; _trade_kelly = 0; _trade_rat = ""; _top3 = []

    _e1, _e2 = st.columns(2)
    with _e1:
        st.metric("Direction Edge", _pe_edg_lbl,
                  delta=f"Prob Up {_pe_pu*100:.1f}%  |  score {_pe_rs:+.3f}")
        st.metric("Volatility Edge", _vol_edge,
                  delta=f"IV/HV {_pe_ivhv:.2f}x" +
                        (f"  |  Avg Move/IV {_avg_mvi:.2f}x" if _avg_mvi else ""),
                  delta_color="off")
        st.metric("EV-Ranked Best", _trade, delta=f"Score {_trade_sc}/100  |  POP {_trade_pop*100:.1f}%")

    with _e2:
        st.markdown("**Decision Logic**")
        st.markdown(f"""
- Prob Up **{_pe_pu*100:.1f}%** → {_pe_edg_lbl}
- Vol Edge: **{_vol_edge}** (IV/HV {_pe_ivhv:.2f}x, percentile rank used)
- Score: **{_pe_rs:+.3f}** → {_pe_str}
- Brier: **{"%.4f" % _brier if _n_obs >= 5 else "< 5 obs"}**

---
**{_trade}**

*{_trade_legs}*
""")

    # ── Top-3 EV-Ranked Strategies table ────────────────────────────────────
    if _top3:
        st.markdown("**Top EV-Ranked Strategies (from live chain, all real strikes)**")
        _t3_rows = []
        # FIX 7: use enumerate instead of list.index() — index() finds the FIRST
        # occurrence by equality, giving rank #1 to all duplicate entries.
        for _rank, _r3 in enumerate(_top3):
            _t3_rows.append({
                "Rank":       f"#{_rank + 1}",
                "Strategy":   _r3.get("Strategy","—"),
                "Legs":       _r3.get("Legs","—"),
                "Score":      f"{_r3.get('Score',0)}/100",
                "POP":        f"{float(_r3.get('pop',0))*100:.1f}%",
                "EV/Lot":     f"₹{float(_r3.get('ev_per_lot',0)):+,.0f}",
                "TxCost/Lot": f"₹{float(_r3.get('tx_cost_lot',0)):,.0f}",
                "Kelly":      f"{float(_r3.get('kelly',0))*100:.1f}%",
                "Dir Align":  f"{float(_r3.get('dir_align',0.5))*100:.0f}%",
                "Safety":     f"{float(_r3.get('safety_ratio',1)):.2f}x",
                "DTE Fit":    f"{float(_r3.get('dte_align',1)):.2f}",
                "Jump λ":     f"{float(_r3.get('jump_lambda',3.0)):.1f}/yr",
            })
        st.dataframe(pd.DataFrame(_t3_rows), use_container_width=True, hide_index=True)
        st.caption(
            "Source: Monte Carlo EV ranking over ALL real chain strikes. "
            "EV/Lot is net of bid-ask spread + STT + NSE charges + stamp duty + GST. "
            "TxCost/Lot = total regulatory drag (not brokerage). "
            "Jump λ = fitted jump frequency from this symbol's own return history. "
            "Score = composite of EV × POP × directional alignment × DTE fit × safety."
        )

    # ── SECTION 6: COMPLETE DASHBOARD TABLE ──────────────────────────────────
    st.markdown("---")
    st.markdown("#### Section 6 — Full Decision Dashboard")

    _brier_disp = f"{_brier:.4f} ({_blbl})" if _n_obs >= 5 else "< 5 obs"
    _me_disp    = (f"{_me_pct:+.2f}pp ({'Buy options' if _me_pct>0.3 else 'Sell options' if _me_pct<-0.3 else 'Fair'})"
                   if _me_pct is not None else
                   (f"{_avg_mvi:.2f}x avg" if _avg_mvi else "< 3 obs"))
    _cal_disp   = "< 5 obs"
    # FIX 1: Only the calibration-accuracy computation requires _n_obs >= 5.
    # The dashboard table itself must always render so new users (< 5 obs) can
    # still see all signals, vol edge, and best-trade output immediately.
    if _n_obs >= 5:
        _rpa     = np.array(_prob_hist[-20:], dtype=float)
        _raa     = np.array(_act_hist[-20:],  dtype=float)
        _cacc    = float(np.mean((_rpa > 0.5) == (_raa > 0.5))) * 100
        _cal_disp = f"{_cacc:.0f}% directional accuracy (last 20 obs)"

    _dashboard = pd.DataFrame({
        "Metric": [
            "Prob Up", "Prob Down", "Direction Edge", "Signal Strength",
            "Implied Move", "Model Move", "Move Edge", "Vol Edge (IV/HV percentile)",
            "ATM IV", "IV / HV Ratio", "HV20",
            "Brier Score", "Calibration",
            "EV-Ranked Best Trade", "EV Trade Legs",
        ],
        "Value": [
            f"{_pe_pu*100:.1f}%",
            f"{_pe_pd*100:.1f}%",
            f"{_pe_edge*100:+.1f}pp — {_pe_edg_lbl}",
            _pe_str,
            f"+-{_pe_impl_pct:.2f}%  (Rs{_pe_impl_rs:,.0f})",
            f"+-{_mdl_pct:.2f}%  (Rs{_mdl_rs:,.0f})" + ("" if _model_move_pct else "  [implied fallback]"),
            _me_disp,
            f"{_vol_edge}  (IV/HV {_pe_ivhv:.2f}x)",
            f"{_pe_iv_pct:.1f}%",
            f"{_pe_ivhv:.2f}x",
            f"{_pe_hv_pct:.1f}%",
            _brier_disp,
            _cal_disp,
            f"{_trade}  [Score {_trade_sc}/100, POP {_trade_pop*100:.1f}%, EV ₹{_trade_ev:+,.0f}/lot]",
            _trade_legs,
        ]
    })
    st.dataframe(_dashboard, use_container_width=True, hide_index=True)

    st.caption(
        f"Workflow: Signals -> Score ({_pe_rs:+.3f}) -> "
        f"Prob Up {_pe_pu*100:.1f}% -> Implied +-{_pe_impl_pct:.1f}% -> "
        f"Vol Edge {_vol_edge} -> {_trade}"
    )
    st.caption(
        "NOTE: Prob Up is currently a logistic transform of lagging technical signals. "
        "It becomes a real calibrated probability after 50+ resolved observations "
        "accumulate in the Brier score section above."
    )
