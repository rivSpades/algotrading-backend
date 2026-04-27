"""
Hybrid VIX regime-switching hedge simulation (lookahead-free), per hedge_feature.md.
Default: daily OHLCV from DB with Yahoo backfill for SPY, VIXM, VIXY, ^VIX.
Optional ``yahoo_only=True``: skip DB reads and fetch each series from Yahoo Finance only (hedge lab preview).
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone as datetime_timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from django.utils import timezone

from market_data.models import Exchange, OHLCV, Symbol
from market_data.providers.yahoo_finance import YahooFinanceProvider
from market_data.services.ohlcv_service import OHLCVService

logger = logging.getLogger(__name__)

HEDGE_TICKERS = ("SPY", "VIXM", "VIXY", "^VIX")
# VIXM/VIXY are the tradeable vol legs used in the spread (normal) or VIXY-only (panic); they must
# not be run as primary gap-strategy symbols when hybrid hedge is on (backtest does not do that).
HEDGE_VOL_ETF_TICKERS = frozenset({"VIXM", "VIXY"})
# Minimum daily bars to trust DB-only; below this we try Yahoo in-memory (no DB write).
_MIN_DAILY_BARS = 20
BENCHMARK_EXCHANGE_CODE = "BENCHMARK"

DEFAULT_HEDGE_CONFIG: Dict[str, Any] = {
    "z_threshold": 1.5,
    "vix_floor": 25.0,
    "smooth_win": 3,
    "panic_spy_weight": 0.60,
    "panic_vixy_weight": 0.40,
    "normal_spy_weight": 0.80,
    "normal_spread_weight": 0.20,
    "rolling_vix_window": 20,
    "rolling_beta_window": 60,
    "min_warmup_days": 60,
}


def hedge_config_keys() -> frozenset:
    return frozenset(DEFAULT_HEDGE_CONFIG.keys())


def filter_hedge_config_user_keys(d: Optional[Dict]) -> Dict[str, Any]:
    """Keep only known hedge parameter keys (ignore unknown / null)."""
    if not d or not isinstance(d, dict):
        return {}
    allowed = hedge_config_keys()
    return {k: v for k, v in d.items() if k in allowed and v is not None}


def get_hedge_lab_saved_overrides() -> Dict[str, Any]:
    """JSON overrides persisted from the Hedge lab page (singleton row)."""
    try:
        from backtest_engine.models import HedgeLabSettings

        row = HedgeLabSettings.get_solo()
        return filter_hedge_config_user_keys(row.hedge_config)
    except Exception:
        return {}


def merge_lab_overrides_with_request(request_overrides: Optional[Dict]) -> Dict[str, Any]:
    """Lab defaults first, then per-request / per-backtest overrides (override wins)."""
    lab = get_hedge_lab_saved_overrides()
    req = filter_hedge_config_user_keys(request_overrides)
    return {**lab, **req}


def resolved_hedge_config_for_backtest(request_overrides: Optional[Dict]) -> Dict[str, Any]:
    """Full effective hedge config to store on Backtest when hedge_enabled (immutable snapshot)."""
    return _merge_hedge_config(merge_lab_overrides_with_request(request_overrides))


def merge_defaults_into_hedge_config(user: Optional[Dict]) -> Dict[str, Any]:
    """Public: fill missing keys from DEFAULT_HEDGE_CONFIG (for API responses)."""
    return _merge_hedge_config(user)


def _merge_hedge_config(user: Optional[Dict]) -> Dict[str, Any]:
    cfg = DEFAULT_HEDGE_CONFIG.copy()
    if user and isinstance(user, dict):
        for k, v in user.items():
            if k not in cfg or v is None:
                continue
            orig = cfg[k]
            try:
                if isinstance(orig, bool):
                    cfg[k] = bool(v)
                elif isinstance(orig, int):
                    cfg[k] = int(v)
                else:
                    cfg[k] = float(v)
            except (TypeError, ValueError):
                pass
    return cfg


def _ts_to_utc_midnight(dt) -> pd.Timestamp:
    """Pandas Timestamp at UTC midnight, comparable to hedge DataFrame index."""
    t = dt if isinstance(dt, pd.Timestamp) else pd.Timestamp(dt)
    if t.tzinfo is None:
        t = t.tz_localize(datetime_timezone.utc)
    else:
        t = t.tz_convert(datetime_timezone.utc)
    return t.normalize()


def _get_or_create_hedge_symbol(ticker: str) -> Symbol:
    exchange, _ = Exchange.objects.get_or_create(
        code=BENCHMARK_EXCHANGE_CODE,
        defaults={
            "name": "Benchmark indices",
            "country": "US",
            "timezone": "America/New_York",
        },
    )
    provider = OHLCVService.get_or_create_yahoo_provider()
    sym, _ = Symbol.objects.get_or_create(
        ticker=ticker,
        defaults={
            "exchange": exchange,
            "provider": provider,
            "type": "etf",
            "name": ticker,
            "status": "active",
        },
    )
    return sym


def _yahoo_may_persist(symbol: Symbol) -> bool:
    """OHLCVService rejects Yahoo saves when symbol is active with another provider (e.g. ALPACA)."""
    prov = getattr(symbol, "provider", None)
    code = (prov.code.upper() if prov and prov.code else "") or ""
    if code == "" or code == "YAHOO":
        return True
    if symbol.status == "disabled":
        return True
    return False


def _ensure_ohlcv(symbol: Symbol, start: datetime, end: datetime) -> None:
    if not _yahoo_may_persist(symbol):
        return
    covered, _ = OHLCVService.check_date_range_coverage(symbol, start, end, "daily")
    count = OHLCV.objects.filter(
        symbol=symbol, timeframe="daily", timestamp__gte=start, timestamp__lte=end
    ).count()
    if covered and count >= 20:
        return
    try:
        rows = YahooFinanceProvider.get_historical_data(
            symbol.ticker, start_date=start, end_date=end, interval="1d"
        )
        if rows:
            OHLCVService.save_ohlcv_data(
                symbol,
                rows,
                timeframe="daily",
                provider=OHLCVService.get_or_create_yahoo_provider(),
                replace_existing=False,
            )
    except Exception as ex:
        logger.warning("Yahoo OHLCV fetch failed for %s: %s", symbol.ticker, ex)


def _close_series_from_db(symbol: Symbol, start: datetime, end: datetime) -> pd.Series:
    rows = (
        OHLCV.objects.filter(
            symbol=symbol,
            timeframe="daily",
            timestamp__gte=start,
            timestamp__lte=end,
        )
        .order_by("timestamp")
        .values_list("timestamp", "close")
    )
    pts = []
    for ts, cl in rows:
        if ts is None or cl is None:
            continue
        t = ts
        if timezone.is_naive(t):
            t = timezone.make_aware(t)
        day = _ts_to_utc_midnight(t)
        pts.append((day, float(cl)))
    if not pts:
        return pd.Series(dtype=float)
    s = pd.Series({p[0]: p[1] for p in pts})
    s = s.sort_index()
    s = s[~s.index.duplicated(keep="last")]
    return s


def _close_series_from_yahoo_memory(ticker: str, start: datetime, end: datetime) -> pd.Series:
    """Yahoo prices for simulation only — does not write to DB (avoids ALPACA/Yahoo provider conflict)."""
    try:
        rows = YahooFinanceProvider.get_historical_data(
            ticker, start_date=start, end_date=end, interval="1d"
        )
    except Exception as ex:
        logger.warning("Yahoo in-memory fetch failed for %s: %s", ticker, ex)
        return pd.Series(dtype=float)
    pts = []
    for row in rows:
        ts = row.get("timestamp")
        cl = row.get("close")
        if ts is None or cl is None:
            continue
        if timezone.is_naive(ts):
            ts = timezone.make_aware(ts)
        day = _ts_to_utc_midnight(ts)
        pts.append((day, float(cl)))
    if not pts:
        return pd.Series(dtype=float)
    s = pd.Series({p[0]: p[1] for p in pts})
    s = s.sort_index()
    s = s[~s.index.duplicated(keep="last")]
    return s


def _close_series(
    ticker: str, start: datetime, end: datetime, *, yahoo_only: bool = False
) -> pd.Series:
    if yahoo_only:
        return _close_series_from_yahoo_memory(ticker, start, end)
    sym = _get_or_create_hedge_symbol(ticker)
    if _yahoo_may_persist(sym):
        _ensure_ohlcv(sym, start, end)
    s_db = _close_series_from_db(sym, start, end)
    if len(s_db) >= _MIN_DAILY_BARS:
        return s_db
    s_yh = _close_series_from_yahoo_memory(ticker, start, end)
    if len(s_yh) > len(s_db):
        return s_yh
    return s_db


def _build_aligned_frame(
    start: datetime, end: datetime, *, yahoo_only: bool = False
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    meta: Dict[str, Any] = {"errors": [], "data_source": "yahoo_finance" if yahoo_only else "database"}
    series_map = {}
    for t in HEDGE_TICKERS:
        s = _close_series(t, start, end, yahoo_only=yahoo_only)
        if s.empty:
            meta["errors"].append(f"No OHLCV for {t}")
        series_map[t] = s
    if any(series_map[t].empty for t in HEDGE_TICKERS):
        return pd.DataFrame(), meta

    common_idx = None
    for t in HEDGE_TICKERS:
        idx = series_map[t].index
        common_idx = idx if common_idx is None else common_idx.intersection(idx)
    if common_idx is None or len(common_idx) < 5:
        meta["errors"].append("Insufficient overlapping dates")
        return pd.DataFrame(), meta

    df = pd.DataFrame(index=sorted(common_idx))
    df["SPY_Ret"] = series_map["SPY"].reindex(df.index).pct_change()
    df["VIXM_Ret"] = series_map["VIXM"].reindex(df.index).pct_change()
    df["VIXY_Ret"] = series_map["VIXY"].reindex(df.index).pct_change()
    df["VIX_Spot"] = series_map["^VIX"].reindex(df.index)
    df = df.dropna()
    return df, meta


def _overlay_row_day_ns(idx_val) -> int:
    t = pd.Timestamp(idx_val)
    if t.tzinfo is None:
        t = t.tz_localize(datetime_timezone.utc)
    else:
        t = t.tz_convert(datetime_timezone.utc)
    t = t.normalize()
    return int(t.value)


def compute_trade_hedge_overlay(
    window_start,
    window_end,
    hedge_config: Optional[Dict],
    *,
    yahoo_only: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Precompute per trading day: strategy vs hedge capital weights (same as hybrid SPY/VIX model)
    and the hedge sleeve's daily return (VIXM/VIXY spread or VIXY in panic).

    Used by BacktestExecutor to split each entry's bet between the strategy leg and the hedge leg.
    """
    cfg = _merge_hedge_config(hedge_config)
    Z_TH = float(cfg["z_threshold"])
    VIX_FLOOR = float(cfg["vix_floor"])
    SMOOTH_WIN = int(cfg["smooth_win"])
    PANIC_SPY = float(cfg["panic_spy_weight"])
    PANIC_VIXY = float(cfg["panic_vixy_weight"])
    STATIC_SPY = float(cfg["normal_spy_weight"])
    STATIC_SPREAD = float(cfg["normal_spread_weight"])
    RW = int(cfg["rolling_vix_window"])
    BETA_W = int(cfg["rolling_beta_window"])
    MIN_W = int(cfg["min_warmup_days"])

    start_dt = _normalize_dt(window_start)
    end_dt = _normalize_dt(window_end)
    if end_dt < start_dt:
        start_dt, end_dt = end_dt, start_dt

    load_start = start_dt - timedelta(days=200)
    df, meta = _build_aligned_frame(load_start, end_dt, yahoo_only=yahoo_only)
    if df.empty:
        return {
            "error": "; ".join(meta.get("errors", [])) or "No hedge overlay data",
            "index_ns": [],
            "w_strategy": [],
            "w_hedge": [],
            "r_hedge": [],
            "min_warmup": MIN_W,
            "data_source": meta.get("data_source", "database"),
        }

    n = len(df)
    index_ns: List[int] = []
    w_strategy: List[float] = []
    w_hedge: List[float] = []
    r_hedge: List[float] = []

    waiting_reset = False
    for i in range(n):
        index_ns.append(_overlay_row_day_ns(df.index[i]))
        if i < MIN_W:
            w_strategy.append(STATIC_SPY)
            w_hedge.append(STATIC_SPREAD)
            r_hedge.append(0.0)
            continue

        vix_hist = df["VIX_Spot"].iloc[i - (RW + 3) : i]
        rolling_m = vix_hist.rolling(RW).mean()
        rolling_s = vix_hist.rolling(RW).std().replace(0, np.nan)
        z_raw = (vix_hist - rolling_m) / rolling_s
        z_last = z_raw.iloc[-SMOOTH_WIN:].dropna()
        smoothed_z = float(z_last.mean()) if len(z_last) else 0.0

        vix_yesterday = float(df["VIX_Spot"].iloc[i - 1])

        if waiting_reset and smoothed_z < 0:
            waiting_reset = False

        is_panic = (smoothed_z > Z_TH) and (vix_yesterday > VIX_FLOOR) and (not waiting_reset)
        row = df.iloc[i]

        if is_panic:
            w_strategy.append(PANIC_SPY)
            w_hedge.append(PANIC_VIXY)
            r_hedge.append(float(row["VIXY_Ret"]))
        else:
            if smoothed_z > Z_TH:
                waiting_reset = True
            v_m_win = df["VIXM_Ret"].iloc[i - BETA_W : i].values
            v_y_win = df["VIXY_Ret"].iloc[i - BETA_W : i].values
            var_y = np.var(v_y_win)
            if var_y < 1e-16:
                dyn_ratio = 1.0
            else:
                dyn_ratio = float(np.cov(v_m_win, v_y_win)[0, 1] / var_y)
                if not np.isfinite(dyn_ratio):
                    dyn_ratio = 1.0
            den = 1.0 + dyn_ratio
            spread_ret = (
                float(row["VIXM_Ret"]) * dyn_ratio - float(row["VIXY_Ret"])
            ) / den
            w_strategy.append(STATIC_SPY)
            w_hedge.append(STATIC_SPREAD)
            r_hedge.append(float(spread_ret))

    return {
        "index_ns": index_ns,
        "w_strategy": w_strategy,
        "w_hedge": w_hedge,
        "r_hedge": r_hedge,
        "min_warmup": MIN_W,
        "data_source": meta.get("data_source", "database"),
    }


def _hedge_panic_daily_rows(df: pd.DataFrame, cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """One row per session after warmup — same gating as :func:`compute_trade_hedge_overlay`."""
    Z_TH = float(cfg["z_threshold"])
    VIX_FLOOR = float(cfg["vix_floor"])
    SMOOTH_WIN = int(cfg["smooth_win"])
    PANIC_SPY = float(cfg["panic_spy_weight"])
    PANIC_VIXY = float(cfg["panic_vixy_weight"])
    STATIC_SPY = float(cfg["normal_spy_weight"])
    STATIC_SPREAD = float(cfg["normal_spread_weight"])
    RW = int(cfg["rolling_vix_window"])
    BETA_W = int(cfg["rolling_beta_window"])
    MIN_W = int(cfg["min_warmup_days"])

    n = len(df)
    waiting_reset = False
    out: List[Dict[str, Any]] = []
    for i in range(n):
        if i < MIN_W:
            continue

        vix_hist = df["VIX_Spot"].iloc[i - (RW + 3) : i]
        rolling_m = vix_hist.rolling(RW).mean()
        rolling_s = vix_hist.rolling(RW).std().replace(0, np.nan)
        z_raw = (vix_hist - rolling_m) / rolling_s
        z_last = z_raw.iloc[-SMOOTH_WIN:].dropna()
        smoothed_z = float(z_last.mean()) if len(z_last) else 0.0

        vix_yesterday = float(df["VIX_Spot"].iloc[i - 1])
        row = df.iloc[i]
        vix_today = float(row["VIX_Spot"])

        if waiting_reset and smoothed_z < 0:
            waiting_reset = False

        is_panic = (smoothed_z > Z_TH) and (vix_yesterday > VIX_FLOOR) and (not waiting_reset)

        if is_panic:
            w_s, w_h = PANIC_SPY, PANIC_VIXY
        else:
            if smoothed_z > Z_TH:
                waiting_reset = True
            v_m_win = df["VIXM_Ret"].iloc[i - BETA_W : i].values
            v_y_win = df["VIXY_Ret"].iloc[i - BETA_W : i].values
            var_y = np.var(v_y_win)
            if var_y < 1e-16:
                dyn_ratio = 1.0
            else:
                dyn_ratio = float(np.cov(v_m_win, v_y_win)[0, 1] / var_y)
                if not np.isfinite(dyn_ratio):
                    dyn_ratio = 1.0
            w_s, w_h = STATIC_SPY, STATIC_SPREAD

        z_arm_met = smoothed_z > Z_TH
        vix_arm_met = vix_yesterday > VIX_FLOOR
        z_points_to_stress = max(0.0, Z_TH - smoothed_z)
        vix_points_above_floor = max(0.0, VIX_FLOOR - vix_yesterday)

        idx_ts = df.index[i]
        as_of = pd.Timestamp(idx_ts).strftime("%Y-%m-%d")
        as_of_ns = _overlay_row_day_ns(idx_ts)

        out.append(
            {
                "as_of": as_of,
                "as_of_index_ns": as_of_ns,
                "z_threshold": Z_TH,
                "vix_floor": VIX_FLOOR,
                "smoothed_vix_z": round(smoothed_z, 6),
                "vix_for_rule_prior_day": round(vix_yesterday, 4),
                "vix_spot_on_as_of": round(vix_today, 4),
                "z_stress_satisfied": z_arm_met,
                "vix_level_satisfied": vix_arm_met,
                "z_points_still_below_threshold": round(z_points_to_stress, 6)
                if not z_arm_met
                else 0.0,
                "vix_points_still_needed_above_floor": round(vix_points_above_floor, 4)
                if not vix_arm_met
                else 0.0,
                "z_headroom": round(Z_TH - smoothed_z, 6),
                "vix_excess_over_floor": round(vix_yesterday - VIX_FLOOR, 4),
                "both_arms_satisfied": bool(z_arm_met and vix_arm_met),
                "panic_blocked_by_hysteresis": bool(
                    (not is_panic) and z_arm_met and vix_arm_met
                ),
                "is_panic": bool(is_panic),
                "waiting_reset": bool(waiting_reset),
                "w_strategy": float(w_s),
                "w_hedge": float(w_h),
            }
        )
    return out


def compute_hedge_panic_snapshot(
    hedge_config: Optional[Dict] = None,
    *,
    yahoo_only: bool = True,
    end_at: Optional[datetime] = None,
    include_chart: bool = True,
    chart_tail_days: int = 90,
) -> Dict[str, Any]:
    """
    Live / dashboard: latest hybrid-VIX panic gating and distance to the two arms
    (Z-stress and prior-day VIX vs floor), using the same loop as
    :func:`compute_trade_hedge_overlay` and the same windowing as
    :func:`live_hedge_weights_at` (end_at aligned to last common session in frame).

    When ``include_chart`` is True, appends a ``chart`` object with the last
    ``chart_tail_days`` sessions (for dashboard time-series / threshold lines).

    The capital split :math:`w_{strategy} / w_{hedge}` does not differ between long
    and short deployments; only the main symbol's direction changes in live trading.
    """
    cfg = _merge_hedge_config(hedge_config)
    MIN_W = int(cfg["min_warmup_days"])
    end_dt = _normalize_dt(end_at) if end_at is not None else timezone.now()
    start_dt = end_dt - timedelta(days=400)
    load_start = start_dt - timedelta(days=200)
    df, meta = _build_aligned_frame(load_start, end_dt, yahoo_only=yahoo_only)
    if df.empty:
        return {
            "error": "; ".join(meta.get("errors", [])) or "No hedge overlay data",
            "data_source": meta.get("data_source", "database"),
            "config": cfg,
        }

    n = len(df)
    if n < MIN_W + 1:
        return {
            "regime": "warmup",
            "min_warmup_days": MIN_W,
            "bars_loaded": n,
            "data_source": meta.get("data_source", "database"),
            "config": cfg,
        }

    daily = _hedge_panic_daily_rows(df, cfg)
    if not daily:
        return {
            "regime": "warmup",
            "min_warmup_days": MIN_W,
            "bars_loaded": n,
            "data_source": meta.get("data_source", "database"),
            "config": cfg,
        }

    snap: Dict[str, Any] = dict(daily[-1])
    for k in ("z_threshold", "vix_floor"):
        del snap[k]
    snap["data_source"] = meta.get("data_source", "database")
    snap["config"] = cfg
    snap["z_threshold"] = float(cfg["z_threshold"])
    snap["vix_floor"] = float(cfg["vix_floor"])

    if snap.get("panic_blocked_by_hysteresis"):
        snap["hysteresis_note"] = (
            "VIX and Z criteria are in stress territory, but the overlay still uses the "
            "normal spread path until mean Z rolls below 0 (hysteresis latch)."
        )
    else:
        snap["hysteresis_note"] = None
    snap["position_modes"] = {
        "long": {
            "w_strategy": snap["w_strategy"],
            "w_hedge": snap["w_hedge"],
            "interpretation": "Strategy leg: long the traded symbol. Hedge: vol sleeve (VIXY proxy).",
        },
        "short": {
            "w_strategy": snap["w_strategy"],
            "w_hedge": snap["w_hedge"],
            "interpretation": "Strategy leg: short the traded symbol. Hedge: same vol-sleeve split as long.",
        },
    }
    snap["note"] = (
        "Regime and w_strategy / w_hedge are the same for long and short; only the main "
        "position direction changes in live."
    )
    snap["regime"] = (
        "panic"
        if snap["is_panic"]
        else (
            "hysteresis"
            if snap.get("panic_blocked_by_hysteresis")
            else "normal"
        )
    )

    if include_chart and chart_tail_days > 0:
        zt = float(cfg["z_threshold"])
        vf = float(cfg["vix_floor"])
        tail = daily[-chart_tail_days:]
        snap["chart"] = {
            "z_threshold": zt,
            "vix_floor": vf,
            "y_label_z": "VIX z-score (smoothed)",
            "y_label_vix": "VIX, prior day (rule input)",
            "caption": (
                "Z and VIX vs thresholds, then a three-way sleeve bar: normal (grey), "
                "hysteresis (amber) = both rules true but model holds spread until mean z < 0, "
                "panic (red) = heavy VIXY sleeve. "
            ),
            "points": [
                {
                    "d": r["as_of"],
                    "z": r["smoothed_vix_z"],
                    "vixP": r["vix_for_rule_prior_day"],
                    "panic": bool(r["is_panic"]),
                    "hysteresis_block": bool(r["panic_blocked_by_hysteresis"]),
                    "waiting_reset": bool(r["waiting_reset"]),
                }
                for r in tail
            ],
        }
    return snap


def _metrics_from_returns(equity: np.ndarray, daily_ret: np.ndarray) -> Dict[str, float]:
    if len(equity) < 2 or len(daily_ret) < 2:
        return {"total_return_pct": 0.0, "sharpe_ratio": 0.0, "max_drawdown_pct": 0.0}
    total = (equity[-1] / equity[0] - 1.0) * 100.0
    r = daily_ret[np.isfinite(daily_ret)]
    if len(r) < 2:
        return {"total_return_pct": round(total, 4), "sharpe_ratio": 0.0, "max_drawdown_pct": 0.0}
    ann_vol = float(r.std() * np.sqrt(252))
    sharpe = float((r.mean() * 252) / ann_vol) if ann_vol > 1e-12 else 0.0
    cum = equity / np.maximum.accumulate(equity)
    mdd = float((cum.min() - 1.0) * 100.0)
    return {
        "total_return_pct": round(total, 4),
        "sharpe_ratio": round(sharpe, 4),
        "max_drawdown_pct": round(mdd, 4),
    }


def simulate_hybrid_vix_hedge(
    start_dt: datetime,
    end_dt: datetime,
    initial_capital: float,
    hedge_config: Optional[Dict] = None,
    *,
    yahoo_only: bool = False,
) -> Dict[str, Any]:
    """
    Run hybrid VIX regime simulation between start_dt and end_dt (inclusive intent).
    Loads extra history before start_dt for signal warmup (180 calendar days).

    If ``yahoo_only`` is True, OHLCV is read only from Yahoo Finance (no DB), fresh each call.
    """
    cfg = _merge_hedge_config(hedge_config)
    Z_TH = float(cfg["z_threshold"])
    VIX_FLOOR = float(cfg["vix_floor"])
    SMOOTH_WIN = int(cfg["smooth_win"])
    PANIC_SPY = float(cfg["panic_spy_weight"])
    PANIC_VIXY = float(cfg["panic_vixy_weight"])
    STATIC_SPY = float(cfg["normal_spy_weight"])
    STATIC_SPREAD = float(cfg["normal_spread_weight"])
    RW = int(cfg["rolling_vix_window"])
    BETA_W = int(cfg["rolling_beta_window"])
    MIN_W = int(cfg["min_warmup_days"])

    start_dt = _normalize_dt(start_dt)
    end_dt = _normalize_dt(end_dt)
    if end_dt < start_dt:
        start_dt, end_dt = end_dt, start_dt

    load_start = start_dt - timedelta(days=200)
    df, meta = _build_aligned_frame(load_start, end_dt, yahoo_only=yahoo_only)
    if df.empty:
        return {
            "equity_curve": [],
            "spy_equity_curve": [],
            "metrics": {},
            "spy_metrics": {},
            "panic_timestamps": [],
            "error": "; ".join(meta.get("errors", [])) or "No data",
            "config": cfg,
            "data_source": meta.get("data_source", "database"),
        }

    if len(df) < MIN_W + 5:
        return {
            "equity_curve": [],
            "spy_equity_curve": [],
            "metrics": {},
            "spy_metrics": {},
            "panic_timestamps": [],
            "error": "Not enough overlapping bars (need warmup history before window)",
            "config": cfg,
            "data_source": meta.get("data_source", "database"),
        }

    n = len(df)
    backtest_results: List[float] = []
    panic_dates: List[pd.Timestamp] = []
    waiting_reset = False

    for i in range(n):
        if i < MIN_W:
            backtest_results.append(float(df["SPY_Ret"].iloc[i]))
            continue

        vix_hist = df["VIX_Spot"].iloc[i - (RW + 3) : i]
        rolling_m = vix_hist.rolling(RW).mean()
        rolling_s = vix_hist.rolling(RW).std().replace(0, np.nan)
        z_raw = (vix_hist - rolling_m) / rolling_s
        z_last = z_raw.iloc[-SMOOTH_WIN:].dropna()
        smoothed_z = float(z_last.mean()) if len(z_last) else 0.0

        vix_yesterday = float(df["VIX_Spot"].iloc[i - 1])

        if waiting_reset and smoothed_z < 0:
            waiting_reset = False

        is_panic = (smoothed_z > Z_TH) and (vix_yesterday > VIX_FLOOR) and (not waiting_reset)
        row = df.iloc[i]

        if is_panic:
            panic_dates.append(df.index[i])
            daily_ret = float(
                row["SPY_Ret"] * PANIC_SPY + row["VIXY_Ret"] * PANIC_VIXY
            )
        else:
            if smoothed_z > Z_TH:
                waiting_reset = True
            v_m_win = df["VIXM_Ret"].iloc[i - BETA_W : i].values
            v_y_win = df["VIXY_Ret"].iloc[i - BETA_W : i].values
            var_y = np.var(v_y_win)
            if var_y < 1e-16:
                dyn_ratio = 1.0
            else:
                dyn_ratio = float(np.cov(v_m_win, v_y_win)[0, 1] / var_y)
                if not np.isfinite(dyn_ratio):
                    dyn_ratio = 1.0
            den = 1.0 + dyn_ratio
            spread_ret = (
                float(row["VIXM_Ret"]) * dyn_ratio - float(row["VIXY_Ret"])
            ) / den
            daily_ret = float(row["SPY_Ret"] * STATIC_SPY + spread_ret * STATIC_SPREAD)

        backtest_results.append(daily_ret)

    strat_rets_all = np.array(backtest_results, dtype=float)
    spy_rets_all = df["SPY_Ret"].iloc[: len(strat_rets_all)].values.astype(float)

    start_ts = _ts_to_utc_midnight(start_dt)
    mask_arr = np.array(df.index >= start_ts)
    if mask_arr.sum() < 2:
        return {
            "equity_curve": [],
            "spy_equity_curve": [],
            "metrics": {},
            "spy_metrics": {},
            "panic_timestamps": [],
            "error": "Selected window too short after warmup",
            "config": cfg,
            "data_source": meta.get("data_source", "database"),
        }

    strat_rets = strat_rets_all[mask_arr]
    spy_rets = spy_rets_all[mask_arr]
    idx = df.index[mask_arr]

    eq = np.empty(len(strat_rets))
    eq[0] = float(initial_capital) * (1.0 + strat_rets[0])
    for j in range(1, len(strat_rets)):
        eq[j] = eq[j - 1] * (1.0 + strat_rets[j])

    spy_eq = np.empty(len(spy_rets))
    spy_eq[0] = float(initial_capital) * (1.0 + spy_rets[0])
    for j in range(1, len(spy_rets)):
        spy_eq[j] = spy_eq[j - 1] * (1.0 + spy_rets[j])

    def curve(points):
        out = []
        for k, ts in enumerate(idx):
            t = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts
            if timezone.is_naive(t):
                t = timezone.make_aware(t)
            out.append(
                {"timestamp": t.isoformat(), "equity": round(float(points[k]), 2)}
            )
        return out

    equity_curve = curve(eq)
    spy_equity_curve = curve(spy_eq)

    panic_iso = []
    for p in panic_dates:
        p_utc = _ts_to_utc_midnight(p)
        if p_utc < start_ts:
            continue
        t = p.to_pydatetime() if hasattr(p, "to_pydatetime") else p
        if timezone.is_naive(t):
            t = timezone.make_aware(t)
        panic_iso.append(t.isoformat())

    return {
        "equity_curve": equity_curve,
        "spy_equity_curve": spy_equity_curve,
        "metrics": _metrics_from_returns(eq, strat_rets),
        "spy_metrics": _metrics_from_returns(spy_eq, spy_rets),
        "panic_timestamps": panic_iso,
        "panic_days_count": len(panic_iso),
        "config": cfg,
        "error": None,
        "data_source": meta.get("data_source", "database"),
    }


def _coerce_aware_datetime(value) -> datetime:
    """
    Equity curves and APIs often pass ISO strings; Django timezone helpers need datetime.
    Accepts datetime, ISO 8601 str, or pandas Timestamp.
    """
    if value is None:
        raise ValueError("timestamp is None")
    if isinstance(value, pd.Timestamp):
        dt = value.to_pydatetime()
    elif isinstance(value, datetime):
        dt = value
    elif isinstance(value, str):
        s = value.strip().replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(s)
        except ValueError:
            dt = pd.Timestamp(s, utc=True).to_pydatetime()
    else:
        raise TypeError(f"Unsupported datetime type: {type(value)}")
    if timezone.is_naive(dt):
        dt = timezone.make_aware(dt)
    return dt


def _normalize_dt(dt) -> datetime:
    return _coerce_aware_datetime(dt)


def run_hybrid_vix_hedge_for_backtest(
    backtest,
    equity_window_start,
    equity_window_end,
    initial_capital: float,
) -> Dict[str, Any]:
    """Use strategy equity window + hedge_config from Backtest model."""
    cfg = getattr(backtest, "hedge_config", None) or {}
    return simulate_hybrid_vix_hedge(
        equity_window_start,
        equity_window_end,
        float(initial_capital),
        cfg,
    )


def live_hedge_weights_at(
    fire_at,
    hedge_config: Optional[Dict],
    *,
    yahoo_only: bool = True,
) -> Tuple[float, float, Dict[str, Any]]:
    """Strategy vs hedge capital weights for *fire_at*'s trading day (live entry).

    Uses the same overlay as backtests. On failure, returns ``(1.0, 0.0, meta)``
    so the entry falls back to a single strategy leg.
    """
    meta: Dict[str, Any] = {}
    end_dt = _normalize_dt(fire_at)
    start_dt = end_dt - timedelta(days=400)
    overlay = compute_trade_hedge_overlay(
        start_dt, end_dt, hedge_config, yahoo_only=yahoo_only,
    )
    if not overlay or overlay.get('error'):
        meta['error'] = (overlay or {}).get('error', 'no_overlay')
        return 1.0, 0.0, meta

    index_ns = overlay.get('index_ns') or []
    ws = overlay.get('w_strategy') or []
    wh = overlay.get('w_hedge') or []
    if not index_ns or not ws or not wh:
        meta['error'] = 'empty_overlay'
        return 1.0, 0.0, meta

    target = _overlay_row_day_ns(fire_at)
    try:
        idx = list(index_ns).index(target)
    except ValueError:
        # Use last day at or before fire_at (rare edge if holiday mismatch)
        best_i = -1
        for i, ns in enumerate(index_ns):
            if ns <= target:
                best_i = i
        idx = best_i if best_i >= 0 else len(ws) - 1

    w_s = float(ws[idx])
    w_h = float(wh[idx])
    meta['data_source'] = overlay.get('data_source', '')
    meta['w_strategy'] = w_s
    meta['w_hedge'] = w_h
    return w_s, w_h, meta
