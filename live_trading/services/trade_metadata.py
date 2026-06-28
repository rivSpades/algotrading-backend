"""Shared LiveTrade `metadata` keys aligned with strategy backtests / frontend.

UI (`BacktestHedgeTradeTableCols`) reads optional `bet_strategy`, `bet_hedge`,
`strategy_pnl`, `hedge_pnl`, `hedge_pnl_pre_portion` (full hedge-leg roll-up sum),
and `hedge_pnl_scale_version` alongside legacy `bet_amount`.

Hybrid **main** rows may also persist `hedge_leg_live_trade_id`, `hedge_leg_ticker`,
and `hedge_leg_quantity` (vol-ETF sleeve size at open).

Hybrid main exits scale aggregate hedge-leg PnL by the fraction of the original
strategy sleeve still represented by ``closed_main_quantity`` (opening snapshot /
broker reconcile fallback).
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Mapping, MutableMapping, Optional

from django.db.models import Q, Sum

META_BET_AMOUNT = 'bet_amount'
META_BET_STRATEGY = 'bet_strategy'
META_BET_HEDGE = 'bet_hedge'
META_STRATEGY_PNL = 'strategy_pnl'
META_HEDGE_PNL = 'hedge_pnl'
META_HEDGE_PNL_PRE_PORTION = 'hedge_pnl_pre_portion'
META_HEDGE_PNL_SCALE_VERSION = 'hedge_pnl_scale_version'
META_HEDGE_LEG_LIVE_TRADE_ID = 'hedge_leg_live_trade_id'
META_HEDGE_LEG_TICKER = 'hedge_leg_ticker'
META_HEDGE_LEG_QUANTITY = 'hedge_leg_quantity'
META_HEDGE_LEG_ENTRY_QUANTITY = 'hedge_leg_entry_quantity'


def _qmoney(d: Decimal) -> str:
    return str(d.quantize(Decimal('0.0001')))


def _qshares(d: Decimal) -> str:
    return str(d.quantize(Decimal('0.00000001')))


def patch_main_trade_hedge_leg_link(
    meta: MutableMapping[str, Any],
    *,
    hedge_live_trade_id: int,
    hedge_ticker: str,
    hedge_quantity: Decimal,
) -> None:
    """After creating a hedge `LiveTrade`, record its link on the hybrid main row."""
    meta[META_HEDGE_LEG_LIVE_TRADE_ID] = int(hedge_live_trade_id)
    meta[META_HEDGE_LEG_TICKER] = str(hedge_ticker).strip()
    try:
        hq = hedge_quantity or Decimal('0')
        if hq > 0:
            meta[META_HEDGE_LEG_QUANTITY] = _qshares(hq)
    except Exception:
        pass


def enrich_flat_entry_metadata(
    meta: MutableMapping[str, Any],
    *,
    entry_price: Decimal,
    quantity: Decimal,
) -> None:
    """Add `bet_amount` for non-hybrid entries (strategy-only notional)."""
    try:
        notional = (entry_price or Decimal('0')) * (quantity or Decimal('0'))
    except Exception:
        return
    if notional > 0:
        meta[META_BET_AMOUNT] = _qmoney(notional)


def enrich_hybrid_entry_main_metadata(
    meta: MutableMapping[str, Any],
    *,
    strat_entry_price: Decimal,
    strat_quantity: Decimal,
) -> None:
    """Attach strategy-leg dollar size on main row when hedge is enabled."""
    try:
        notional = (strat_entry_price or Decimal('0')) * (strat_quantity or Decimal('0'))
    except Exception:
        return
    if notional > 0:
        meta[META_BET_STRATEGY] = _qmoney(notional)
        meta[META_BET_AMOUNT] = _qmoney(notional)


def enrich_hybrid_entry_hedge_metadata(
    meta: MutableMapping[str, Any],
    *,
    hedge_entry_price: Decimal,
    hedge_quantity: Decimal,
) -> None:
    """Attach hedge-leg dollar notion on hedge `LiveTrade` rows."""
    try:
        notional = (hedge_entry_price or Decimal('0')) * (hedge_quantity or Decimal('0'))
    except Exception:
        return
    if notional > 0:
        meta[META_BET_HEDGE] = _qmoney(notional)
    try:
        hq = hedge_quantity or Decimal('0')
        if hq > 0:
            meta[META_HEDGE_LEG_ENTRY_QUANTITY] = _qshares(hq)
    except Exception:
        pass


def closed_hedge_pnl_aggregate(parent_live_trade_id: int):
    """Sum realized PnL of closed hedge legs for a main trade id."""

    from ..models import LiveTrade

    pid = parent_live_trade_id
    row = (
        LiveTrade.objects.filter(
            status='closed',
            metadata__is_hedge_leg=True,
        )
        .filter(
            Q(metadata__hedge_parent_live_trade_id=pid)
            | Q(metadata__hedge_parent_live_trade_id=str(pid))
        )
        .aggregate(total=Sum('pnl'))
    )
    raw = row.get('total')
    if raw is None:
        return None
    return Decimal(str(raw))


_ORIG_STRATEGYLEG_META_KEYS = (
    'main_opening_strategyleg_qty',
    'opening_strategyleg_quantity',
    'requested_quantity',
    'strategyleg_quantity',
)


def _decimal_positive(raw: Any) -> Optional[Decimal]:
    if raw is None or raw == '':
        return None
    try:
        d = Decimal(str(raw))
    except Exception:
        return None
    return d if d > 0 else None


def snapshot_orig_strategyleg_quantity(meta: Mapping[str, Any]) -> Optional[Decimal]:
    """Baseline strategy-sleeve share count before partial broker drift (for hedge PnL portioning).

    Prefer explicit opening snapshot keys. If legacy rows omit them, approximate from
    broker quantity reconciles (`from_db` is the overstated ledger qty before shrinking).
    Do **not** use ``full_bet_shares`` — that is total book size before applying ``w_strategy``.
    """
    for key in _ORIG_STRATEGYLEG_META_KEYS:
        cand = _decimal_positive(meta.get(key))
        if cand is not None:
            return cand
    recon = meta.get('broker_quantity_reconcile')
    if isinstance(recon, list):
        best: Optional[Decimal] = None
        for entry in recon:
            if not isinstance(entry, dict):
                continue
            d = _decimal_positive(entry.get('from_db'))
            if d is None:
                continue
            best = d if best is None else max(best, d)
        if best is not None:
            return best
    return None


def main_qty_portion_factor(meta: Mapping[str, Any], closed_main_quantity: Decimal) -> Decimal:
    """How much of the original strategy sleeve is represented by ``closed_main_quantity``."""
    try:
        q_cur = Decimal(str(closed_main_quantity or '0'))
    except Exception:
        return Decimal('1')
    if q_cur <= 0:
        return Decimal('1')
    orig = snapshot_orig_strategyleg_quantity(meta)
    if orig is None or orig <= 0:
        return Decimal('1')
    frac = q_cur / orig
    if frac > Decimal('1'):
        frac = Decimal('1')
    if frac <= 0:
        return Decimal('1')
    return frac


def merge_exit_rollups_for_main_trade(
    meta: Mapping[str, Any],
    *,
    main_pnl: Optional[Decimal],
    parent_trade_id: int,
    closed_main_quantity: Optional[Decimal] = None,
) -> dict[str, Any]:
    """Attach `strategy_pnl` / `hedge_pnl` for hybrid UI when closing the main row."""
    out = dict(meta)
    if main_pnl is not None:
        try:
            out[META_STRATEGY_PNL] = _qmoney(Decimal(str(main_pnl)))
        except Exception:
            out[META_STRATEGY_PNL] = str(main_pnl)
    portion = Decimal('1')
    if closed_main_quantity is not None:
        try:
            portion = main_qty_portion_factor(meta, closed_main_quantity)
        except Exception:
            portion = Decimal('1')
    try:
        hsum = closed_hedge_pnl_aggregate(parent_trade_id)
    except Exception:
        hsum = None
    if hsum is not None:
        try:
            hdec = Decimal(str(hsum))
            scaled = hdec * portion
            out[META_HEDGE_PNL_PRE_PORTION] = _qmoney(hdec)
            out[META_HEDGE_PNL] = _qmoney(scaled)
            out[META_HEDGE_PNL_SCALE_VERSION] = 2
        except Exception:
            out[META_HEDGE_PNL] = str(hsum)
    return out
