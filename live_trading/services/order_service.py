"""Live order placement + LiveTrade lifecycle service.

Glue between an actionable `LiveSignal` (produced by the live engine) and the
underlying `BrokerAdapter` / `LiveTrade` model. The orchestration tasks call
`place_signal_order(...)` once a signal is fired; this service is the only
place the rest of the codebase reaches into to actually submit an order.

Responsibilities:

- Translate a `LiveSignal` into broker call parameters (side / quantity).
- Compute the trade quantity from the deployment's bet sizing rules and the
  broker's **account equity** (cash + positions value) so a fixed % of
  portfolio is consistent as positions are opened; fall back to the
  deployment's `initial_capital` when the broker call fails.
- Persist a `LiveTrade` row in the right lifecycle state (`open` for entries,
  `closed` for exits) and link it to the originating `DeploymentSymbol`.
- Emit `order_placed`, `order_filled`, `order_failed`, `trade_opened`,
  `trade_closed` audit events with rich context so the dashboard can render
  the full chain (signal -> order -> trade).
- Idempotency: refuse to open a duplicate position for the same
  (`deployment_symbol`, `position_mode`, calendar date) so a re-fire cannot
  double up. Exits short-circuit when no matching open trade exists.

The caller chain is::

    live_engine -> evaluate_deployment_symbol() -> place_signal_order(...)
                                              -> log_event()
                                              -> LiveTrade.create(...)
                                              -> broker.place_order(...)

`update_open_trades(...)` is a sibling helper used by `update_positions_task`
to reconcile open trades against broker state (close LiveTrade rows when the
broker reports the position is gone, mark the realized PnL when known).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from datetime import datetime
from decimal import Decimal, InvalidOperation, ROUND_DOWN
from enum import Enum
from typing import Any, Optional, Union

from django.db import transaction
from django.utils import timezone

from market_data.models import Symbol

from ..adapters.base import BaseBrokerAdapter, OrderResult, PositionInfo
from ..adapters.factory import get_broker_adapter
from ..engines.base import (
    SIGNAL_EXIT_LONG,
    SIGNAL_EXIT_SHORT,
    SIGNAL_LONG,
    SIGNAL_SHORT,
    LiveSignal,
)
from ..models import DeploymentSymbol, LiveTrade, StrategyDeployment
from strategies.signals import action_to_side
from .audit import log_event
from .trade_metadata import (
    enrich_flat_entry_metadata,
    enrich_hybrid_entry_hedge_metadata,
    patch_main_trade_hedge_leg_link,
    enrich_hybrid_entry_main_metadata,
    merge_exit_rollups_for_main_trade,
)

logger = logging.getLogger(__name__)

# Hedge sleeve tickers (normal/hysteresis uses VIXM; panic uses VIXY).
HEDGE_TICKER_NORMAL = 'VIXM'
HEDGE_TICKER_PANIC = 'VIXY'


def _positive_decimal(val: Any) -> Optional[Decimal]:
    """Return ``Decimal`` value only when strictly positive ('' and 0 are unknown/absent)."""
    if val is None or val == '':
        return None
    try:
        d = Decimal(str(val))
        return d if d > 0 else None
    except (InvalidOperation, ValueError, TypeError):
        return None


def _resolve_exit_execution_price(
    *,
    order_price: Any,
    entry_price_fallback: Decimal,
    metadata_exit_subset: Optional[dict] = None,
    broker_adapter: Optional[BaseBrokerAdapter] = None,
    quote_symbol: Optional[str] = None,
) -> Decimal:
    """Resolve realised exit avg for PnL and ``LiveTrade.exit_price``.

    ``if order_result.price`` is unsafe: ``Decimal('0')`` is falsy yet is a common
    placeholder before ``filled_avg_price`` is populated. Prefer positive prices
    only, then persisted exit-metadata from the synchronous response, then a live
    quote, then sleeve entry as last resort.
    """
    c = _positive_decimal(order_price)
    if c is not None:
        return c
    ex = metadata_exit_subset if isinstance(metadata_exit_subset, dict) else {}
    for key in (
        'synced_avg_price',
        'filled_avg_price',
        'executed_avg_price',
        'filled_price',
    ):
        c = _positive_decimal(ex.get(key))
        if c is not None:
            return c
    if broker_adapter and quote_symbol:
        try:
            c = _positive_decimal(broker_adapter.get_current_price(quote_symbol))
            if c is not None:
                return c
        except Exception:
            pass
    try:
        return Decimal(str(entry_price_fallback))
    except Exception:
        return Decimal('0')


class OrderPlacementError(Exception):
    """Raised when a broker order cannot be placed (validation level)."""


def _intended_closing_side(trade: LiveTrade) -> str:
    if trade.trade_type == 'buy':
        return 'sell'
    if trade.trade_type == 'sell':
        return 'buy'
    return 'sell' if trade.position_mode == SIGNAL_LONG else 'buy'


def _manual_close_error_user_message(reason: str, close_ctx: dict) -> str:
    if close_ctx.get('hint'):
        return str(close_ctx['hint'])
    defaults = {
        'no_broker_position': 'Broker reports no open position for this symbol. Use force: true to reset the app ledger only.',
        'broker_position_unavailable': 'Could not read broker position. Retry, or use force: true to reset the app ledger only.',
        'broker_position_mismatch_for_long_close': 'Broker is not long this name; a sell will not cover this app trade as written.',
        'broker_position_mismatch_for_short_cover': 'Broker is not short this name; a buy will not cover this app trade as written.',
        'zero_close_qty': 'Close quantity resolved to zero.',
    }
    return defaults.get(reason, reason)


def _manual_close_side_and_qty(
    trade: LiveTrade,
    broker_adapter: BaseBrokerAdapter,
) -> tuple[Optional[str], Optional[Decimal], Optional[str], dict]:
    """Resolve close side from entry `trade_type` and cap quantity to the broker position.

    Alpaca rejects sells that would *open* a fractional short (422: fractional orders cannot
    be sold short) when the account has no matching long. We avoid that by only submitting
    what the broker actually holds, and by failing clearly on DB/broker mismatch.
    """
    ctx: dict = {
        'ticker': (trade.symbol.ticker or '').strip(),
        'live_trade_id': trade.id,
    }
    closing_side = _intended_closing_side(trade)
    ctx['closing_side'] = closing_side

    if (trade.trade_type == 'buy' and trade.position_mode != SIGNAL_LONG) or (
        trade.trade_type == 'sell' and trade.position_mode != SIGNAL_SHORT
    ):
        ctx['db_inconsistent'] = {
            'trade_type': trade.trade_type,
            'position_mode': trade.position_mode,
        }
        logger.warning(
            'LiveTrade %s: trade_type %s vs position_mode %s — using trade_type for close side',
            trade.id,
            trade.trade_type,
            trade.position_mode,
        )

    try:
        if hasattr(broker_adapter, 'get_position_resolved'):
            pos: Optional[PositionInfo] = broker_adapter.get_position_resolved(
                (trade.symbol.ticker or '').strip(),
            )
        else:  # pragma: no cover
            pos = broker_adapter.get_position((trade.symbol.ticker or '').strip())
    except Exception as exc:  # pragma: no cover - network
        logger.exception('get_position_resolved failed for manual close %s', trade.symbol.ticker)
        return None, None, 'broker_position_unavailable', {**ctx, 'error': str(exc)}

    if pos is None:
        ctx['hint'] = (
            'No open position for this symbol at the broker. POST {"force": true} to close '
            'and reset the row in the app only (no Alpaca order).'
        )
        return None, None, 'no_broker_position', ctx

    tq = trade.quantity
    if closing_side == 'sell':
        if pos.position_type != 'long' or pos.quantity <= 0:
            return None, None, 'broker_position_mismatch_for_long_close', {
                **ctx,
                'broker': {'side': pos.position_type, 'qty': str(pos.quantity)},
            }
        close_qty = min(tq, pos.quantity)
    else:
        if pos.position_type != 'short' or pos.quantity >= 0:
            return None, None, 'broker_position_mismatch_for_short_cover', {
                **ctx,
                'broker': {'side': pos.position_type, 'qty': str(pos.quantity)},
            }
        cover = abs(pos.quantity)
        close_qty = min(tq, cover)

    if close_qty <= 0:
        return None, None, 'zero_close_qty', {**ctx, 'close_qty': str(close_qty)}

    if close_qty != tq:
        ctx['quantity_capped'] = {'from_db': str(tq), 'at_broker': str(close_qty)}

    return closing_side, close_qty, None, ctx


class OrderOutcomeStatus(str, Enum):
    """Internal outcome of an order attempt (distinct from broker OrderStatus)."""
    PLACED = 'placed'
    FILLED = 'filled'
    FAILED = 'failed'
    SKIPPED = 'skipped'


@dataclass
class OrderOutcome:
    """Result of an order attempt, returned to the caller for the audit feed."""

    status: Union[OrderOutcomeStatus, str]  # OrderOutcomeStatus (or its string value)
    reason: str = ''
    order_result: Optional[OrderResult] = None
    live_trade_id: Optional[int] = None
    duplicate_trade_id: Optional[int] = None
    quantity: Optional[Decimal] = None
    message: str = ''
    side: str = ''  # intended or executed closing side when relevant

    def to_dict(self) -> dict:
        order = self.order_result
        d = {
            'status': self.status,
            'reason': self.reason,
            'live_trade_id': self.live_trade_id,
            'duplicate_trade_id': self.duplicate_trade_id,
            'quantity': str(self.quantity) if self.quantity is not None else None,
            'order': (
                {
                    'broker_order_id': order.broker_order_id,
                    'status': order.status,
                    'filled_quantity': str(order.filled_quantity),
                    'filled_price': str(order.price),
                    'error': order.error_message,
                }
                if order
                else None
            ),
        }
        if self.message:
            d['message'] = self.message
        if self.side:
            d['side'] = self.side
        return d


def _hedge_parent_live_trade_ids_match(meta_parent: Any, main_id: Any) -> bool:
    """Match JSON ``hedge_parent_live_trade_id`` (int/str) to ``LiveTrade.id``."""
    if meta_parent is None or main_id is None:
        return False
    try:
        return int(meta_parent) == int(main_id)
    except (TypeError, ValueError):
        return str(meta_parent).strip() == str(main_id).strip()


def _hedge_leg_child_trades(trade: LiveTrade) -> list:
    """Open hedge legs whose metadata points at this main trade as parent."""
    out: list = []
    for ht in (
        trade.deployment.live_trades.filter(status='open')
        .select_related('symbol')
    ):
        m = ht.metadata or {}
        if m.get('is_hedge_leg') and _hedge_parent_live_trade_ids_match(
            m.get('hedge_parent_live_trade_id'),
            trade.id,
        ):
            out.append(ht)
    return out


def _finalize_hedge_leg_on_exit_fill(
    ht: LiveTrade,
    order_result,
    fire_at: datetime,
    *,
    broker_adapter: Optional[BaseBrokerAdapter] = None,
) -> None:
    """Persist closed state for a hedge LiveTrade when the broker reports a fill."""
    filled = order_result.filled_quantity or Decimal('0')
    if filled <= 0:
        return
    meta_u = dict(ht.metadata or {})
    exit_block_raw = meta_u.get('exit') if isinstance(meta_u.get('exit'), dict) else {}
    exit_block = dict(exit_block_raw)
    ticker = getattr(ht.symbol, 'ticker', None) if ht.symbol_id else None
    exit_price = _resolve_exit_execution_price(
        order_price=order_result.price,
        entry_price_fallback=ht.entry_price or Decimal('0'),
        metadata_exit_subset=exit_block,
        broker_adapter=broker_adapter,
        quote_symbol=ticker,
    )
    exit_block['executed_avg_price'] = str(exit_price)
    meta_u['exit'] = exit_block
    quantity_closed = min(filled, ht.quantity) if ht.quantity else filled
    pnl = _compute_pnl(
        ht.entry_price, exit_price, quantity_closed, ht.position_mode,
    )
    pnl_pct = _safe_pct(ht.entry_price, exit_price, ht.position_mode)
    with transaction.atomic():
        ht.refresh_from_db()
        fresh_meta = dict(ht.metadata or {})
        db_exit = dict(
            fresh_meta.get('exit') if isinstance(fresh_meta.get('exit'), dict) else {}
        )
        merged_exit = {**db_exit, **exit_block}
        merged_exit['executed_avg_price'] = str(exit_price)
        fresh_meta['exit'] = merged_exit
        ht.metadata = fresh_meta
        ht.exit_price = exit_price
        ht.exit_timestamp = fire_at
        ht.pnl = pnl
        ht.pnl_percentage = pnl_pct
        ht.is_winner = pnl is not None and pnl > 0
        ht.status = 'closed'
        if quantity_closed != ht.quantity:
            ht.quantity = quantity_closed
        ht.save(
            update_fields=[
                'exit_price', 'exit_timestamp', 'pnl', 'pnl_percentage',
                'is_winner', 'status', 'quantity', 'metadata', 'updated_at',
            ],
        )


def _submit_hedge_leg_exits_for_main_trade(
    main_trade: LiveTrade,
    *,
    broker_adapter: BaseBrokerAdapter,
    fire_at: datetime,
    source: str,
    actor_type: str,
    actor_id: str,
) -> None:
    """Market-close VIXM/VIXY hedge rows linked to this main leg (long hedges → sell)."""
    deployment = main_trade.deployment
    deployment_symbol = main_trade.deployment_symbol
    for ht in _hedge_leg_child_trades(main_trade):
        try:
            hx = broker_adapter.place_order(
                symbol=ht.symbol.ticker,
                side='sell',
                quantity=ht.quantity,
                order_type='market',
            )
            meta = dict(ht.metadata or {})
            meta['exit'] = {
                'broker_order_id': hx.broker_order_id,
                'order_id': hx.order_id,
                'status': hx.status,
                'filled_quantity': str(hx.filled_quantity),
                'filled_price': str(hx.price),
                'requested_at': fire_at.isoformat(),
                'source': source,
            }
            ht.metadata = meta
            if hx.status != 'rejected' and (hx.filled_quantity or Decimal('0')) > 0:
                _finalize_hedge_leg_on_exit_fill(
                    ht, hx, fire_at, broker_adapter=broker_adapter,
                )
                log_event(
                    deployment,
                    deployment_symbol=deployment_symbol,
                    event_type='trade_closed',
                    actor_type=actor_type,
                    actor_id=actor_id,
                    message=(
                        f"{main_trade.symbol.ticker} exit: hedge sleeve "
                        f"{ht.symbol.ticker} leg #{ht.id} filled (parent main #{main_trade.id})"
                    ),
                    context={
                        'main_ticker': main_trade.symbol.ticker,
                        'hedge_ticker': ht.symbol.ticker,
                        'ticker': ht.symbol.ticker,
                        'live_trade_id': ht.id,
                        'hedge_parent_live_trade_id': main_trade.id,
                        'source': source,
                    },
                )
            else:
                ht.save(update_fields=['metadata', 'updated_at'])
            if hx.status == 'rejected':
                log_event(
                    deployment,
                    deployment_symbol=deployment_symbol,
                    event_type='order_failed',
                    actor_type=actor_type,
                    actor_id=actor_id,
                    level='warning',
                    message=(
                        f"{ht.symbol.ticker}: broker rejected hedge exit "
                        f"({hx.error_message or 'no reason'})"
                    ),
                    context={
                        'ticker': ht.symbol.ticker,
                        'live_trade_id': ht.id,
                        'broker_order_id': hx.broker_order_id,
                        'error': hx.error_message,
                    },
                )
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception('Hedge exit failed for trade %s', ht.id)
            meta = dict(ht.metadata or {})
            meta['exit_error'] = str(exc)
            ht.metadata = meta
            ht.save(update_fields=['metadata', 'updated_at'])


def _apply_force_db_close_row(
    trade: LiveTrade,
    *,
    fire_at: datetime,
    parent_id: Optional[int] = None,
) -> None:
    """Set exit = entry, zero PnL, and force flag (caller saves inside transaction)."""
    meta = dict(trade.metadata or {})
    note = (
        'No broker order; position not at broker or account mismatch. App ledger only.'
    )
    if parent_id is not None:
        note = f'Closed with main trade #{parent_id} (app ledger only).'
    meta['force_db_close'] = True
    exit_block = {
        'source': 'manual_close_force',
        'requested_at': fire_at.isoformat(),
        'note': note,
    }
    if parent_id is not None:
        exit_block['parent_live_trade_id'] = parent_id
    meta['exit'] = exit_block
    trade.exit_price = trade.entry_price
    trade.exit_timestamp = fire_at
    trade.pnl = Decimal('0')
    trade.pnl_percentage = Decimal('0')
    trade.is_winner = False
    trade.status = 'closed'
    trade.metadata = meta


def _manual_close_force_db_reset(
    trade: LiveTrade,
    *,
    fire_at: datetime,
    actor_type: str,
    actor_id: str,
) -> OrderOutcome:
    """Mark trade closed in DB only. No broker call. Closes linked hedge leg rows if any."""
    deployment = trade.deployment
    if trade.status != 'open':
        return OrderOutcome(status='skipped', reason='trade_not_open', live_trade_id=trade.id)

    closing_side = _intended_closing_side(trade)
    children = _hedge_leg_child_trades(trade)
    with transaction.atomic():
        _apply_force_db_close_row(trade, fire_at=fire_at, parent_id=None)
        trade.save(
            update_fields=[
                'exit_price', 'exit_timestamp', 'pnl', 'pnl_percentage',
                'is_winner', 'status', 'metadata', 'updated_at',
            ],
        )
        for ht in children:
            _apply_force_db_close_row(
                ht, fire_at=fire_at, parent_id=trade.id,
            )
            ht.save(
                update_fields=[
                    'exit_price', 'exit_timestamp', 'pnl', 'pnl_percentage',
                    'is_winner', 'status', 'metadata', 'updated_at',
                ],
            )

    log_event(
        deployment,
        deployment_symbol=trade.deployment_symbol,
        event_type='trade_closed',
        actor_type=actor_type,
        actor_id=actor_id,
        message=(
            f"{(trade.symbol.ticker or '').strip()}: force DB close; trade #{trade.id} "
            f"(+ {len(children)} hedge leg(s)) — no broker order"
        ),
        context={
            'ticker': (trade.symbol.ticker or '').strip(),
            'side': closing_side,
            'live_trade_id': trade.id,
            'hedge_leg_ids_closed': [h.id for h in children],
            'source': 'manual_close_force',
        },
    )

    return OrderOutcome(
        status='filled',
        reason='manual_exit_db_reset',
        live_trade_id=trade.id,
        message=(
            'Closed in the app with no broker order (exit=entry, PnL 0). '
            'Use this when the position no longer exists at the broker.'
        ),
        side=closing_side,
        quantity=trade.quantity,
    )


def manual_close_live_trade(
    trade: LiveTrade,
    *,
    actor_type: str,
    actor_id: str,
    fire_at: Optional[datetime] = None,
    broker_adapter: Optional[BaseBrokerAdapter] = None,
    trust_db: bool = False,
) -> OrderOutcome:
    """Manually submit a market close for a single LiveTrade and persist state.

    If the broker accepts but doesn't fill immediately, the trade remains `open`
    and `update_open_trades` will reconcile later.

    With ``trust_db=True`` (API: ``{"force": true}``), only update the app: mark the
    trade (and any linked hedge leg rows) closed with no Alpaca order. Use when
    the position is already flat at the broker.

    When submitting a real close, linked hedge (VIXM/VIXY) legs are market-sold
    right after the main order is accepted, same as the automated exit path.
    """
    fire_at = fire_at or timezone.now()
    deployment = trade.deployment
    if trade.status != 'open':
        return OrderOutcome(status='skipped', reason='trade_not_open', live_trade_id=trade.id)

    if trust_db:
        return _manual_close_force_db_reset(
            trade, fire_at=fire_at, actor_type=actor_type, actor_id=actor_id,
        )

    broker_adapter = broker_adapter or get_adapter_for_deployment(deployment)
    if broker_adapter is None:
        log_event(
            deployment,
            deployment_symbol=trade.deployment_symbol,
            event_type='order_failed',
            actor_type=actor_type,
            actor_id=actor_id,
            level='warning',
            message=f"{trade.symbol.ticker}: manual close skipped — no broker adapter.",
            context={'ticker': trade.symbol.ticker, 'live_trade_id': trade.id},
        )
        return OrderOutcome(status='failed', reason='no_broker_adapter', live_trade_id=trade.id)

    closing_side, close_qty, close_err, close_ctx = _manual_close_side_and_qty(
        trade, broker_adapter
    )
    if close_err:
        intended_side = close_ctx.get('closing_side', '')
        log_event(
            deployment,
            deployment_symbol=trade.deployment_symbol,
            event_type='order_failed',
            actor_type=actor_type,
            actor_id=actor_id,
            level='warning',
            message=f"{trade.symbol.ticker}: manual close blocked — {close_err}.",
            context={
                'ticker': trade.symbol.ticker,
                'side': intended_side,
                'live_trade_id': trade.id,
                'source': 'manual_close',
                'error': close_err,
                **{k: v for k, v in close_ctx.items() if v is not None},
            },
        )
        return OrderOutcome(
            status='failed',
            reason=close_err,
            live_trade_id=trade.id,
            message=_manual_close_error_user_message(close_err, close_ctx),
            side=str(intended_side) if intended_side else '',
        )

    ticker_s = (trade.symbol.ticker or '').strip()
    msg = (
        f"{ticker_s}: manual close submitting {closing_side.upper()} {close_qty} "
        f"for trade #{trade.id}"
    )
    log_event(
        deployment,
        deployment_symbol=trade.deployment_symbol,
        event_type='order_placed',
        actor_type=actor_type,
        actor_id=actor_id,
        message=msg,
        context={
            'ticker': ticker_s,
            'side': closing_side,
            'quantity': str(close_qty),
            'live_trade_id': trade.id,
            'source': 'manual_close',
            'resolved': close_ctx,
        },
    )

    order_result = broker_adapter.place_order(
        symbol=ticker_s,
        side=closing_side,
        quantity=close_qty,
        order_type='market',
    )
    if order_result.status == 'rejected':
        meta = dict(trade.metadata or {})
        if order_result.error_message and '40310100' in order_result.error_message:
            meta['blocked_by_pdt'] = True
        trade.metadata = meta
        trade.save(update_fields=['metadata', 'updated_at'])
        log_event(
            deployment,
            deployment_symbol=trade.deployment_symbol,
            event_type='order_failed',
            actor_type=actor_type,
            actor_id=actor_id,
            level='warning',
            message=f"{trade.symbol.ticker}: manual close rejected ({order_result.error_message or 'no reason'})",
            context={
                'ticker': trade.symbol.ticker,
                'live_trade_id': trade.id,
                'broker_order_id': order_result.broker_order_id,
                'error': order_result.error_message,
                'source': 'manual_close',
            },
        )
        return OrderOutcome(
            status='failed',
            reason='broker_rejected',
            order_result=order_result,
            live_trade_id=trade.id,
        )

    _submit_hedge_leg_exits_for_main_trade(
        trade,
        broker_adapter=broker_adapter,
        fire_at=fire_at,
        source='manual_close',
        actor_type=actor_type,
        actor_id=actor_id,
    )

    filled = order_result.filled_quantity or Decimal('0')
    meta = dict(trade.metadata or {})
    meta['exit'] = {
        'broker_order_id': order_result.broker_order_id,
        'order_id': order_result.order_id,
        'status': order_result.status,
        'filled_quantity': str(order_result.filled_quantity),
        'filled_price': str(order_result.price),
        'requested_at': fire_at.isoformat(),
        'source': 'manual_close',
    }
    if filled <= 0:
        trade.metadata = meta
        trade.save(update_fields=['metadata', 'updated_at'])
        # Alpaca often reports 0 fill on the synchronous response while the market
        # order fills moments later — align DB with broker positions (main + hedges).
        try:
            update_open_trades(
                deployment,
                broker_adapter=broker_adapter,
                actor_type=actor_type,
                actor_id=actor_id,
            )
        except Exception:
            logger.exception(
                'update_open_trades after manual_close (pending fill) failed '
                'deployment=%s trade=%s',
                deployment.id,
                trade.id,
            )
        return OrderOutcome(
            status='placed',
            reason='manual_exit_submitted',
            order_result=order_result,
            live_trade_id=trade.id,
            quantity=close_qty,
        )

    quantity_closed = filled
    if quantity_closed != trade.quantity:
        meta['quantity_reconciled_to_fill'] = {
            'db_quantity': str(trade.quantity),
            'closed_quantity': str(quantity_closed),
        }
        trade.quantity = quantity_closed
    trade.metadata = meta

    exit_exit_sub = meta.get('exit') if isinstance(meta.get('exit'), dict) else {}
    exit_price = _resolve_exit_execution_price(
        order_price=order_result.price,
        entry_price_fallback=trade.entry_price,
        metadata_exit_subset=exit_exit_sub,
        broker_adapter=broker_adapter,
        quote_symbol=getattr(trade.symbol, 'ticker', None) if trade.symbol_id else None,
    )
    exit_exit_written = dict(
        meta.get('exit') if isinstance(meta.get('exit'), dict) else {}
    )
    exit_exit_written['executed_avg_price'] = str(exit_price)
    meta['exit'] = exit_exit_written
    pnl = _compute_pnl(
        trade.entry_price, exit_price, quantity_closed, trade.position_mode,
    )
    pnl_pct = _safe_pct(trade.entry_price, exit_price, trade.position_mode)
    if not meta.get('is_hedge_leg') and (
        meta.get('hedge_enabled') or getattr(trade.deployment, 'hedge_enabled', False)
    ):
        meta = merge_exit_rollups_for_main_trade(
            meta,
            main_pnl=pnl,
            parent_trade_id=trade.id,
            closed_main_quantity=quantity_closed,
        )
    trade.metadata = meta
    with transaction.atomic():
        trade.exit_price = exit_price
        trade.exit_timestamp = fire_at
        trade.pnl = pnl
        trade.pnl_percentage = pnl_pct
        trade.is_winner = pnl is not None and pnl > 0
        trade.status = 'closed'
        trade.save(
            update_fields=[
                'exit_price', 'exit_timestamp', 'pnl', 'pnl_percentage',
                'is_winner', 'status', 'quantity', 'metadata', 'updated_at',
            ],
        )
    log_event(
        deployment,
        deployment_symbol=trade.deployment_symbol,
        event_type='trade_closed',
        actor_type=actor_type,
        actor_id=actor_id,
        message=f"{trade.symbol.ticker}: manual close filled; trade #{trade.id} closed",
        context={
            'ticker': trade.symbol.ticker,
            'live_trade_id': trade.id,
            'pnl': str(pnl) if pnl is not None else None,
            'pnl_percentage': str(pnl_pct) if pnl_pct is not None else None,
            'source': 'manual_close',
        },
    )

    # Close any legs the sync path picks up (e.g. hedge filled async, or extra safety).
    try:
        update_open_trades(
            deployment,
            broker_adapter=broker_adapter,
            actor_type=actor_type,
            actor_id=actor_id,
        )
    except Exception:
        logger.exception(
            'update_open_trades after manual_close (filled) failed '
            'deployment=%s trade=%s',
            deployment.id,
            trade.id,
        )

    return OrderOutcome(
        status='filled',
        reason='manual_exit_filled',
        order_result=order_result,
        live_trade_id=trade.id,
        quantity=trade.quantity,
    )


def get_adapter_for_deployment(
    deployment: StrategyDeployment,
) -> Optional[BaseBrokerAdapter]:
    """Return the broker adapter for a deployment (paper vs real money)."""
    paper = deployment.deployment_type != 'real_money'
    return get_broker_adapter(deployment.broker, paper_trading=paper)


def place_signal_order(
    deployment_symbol: DeploymentSymbol,
    signal: LiveSignal,
    *,
    broker_adapter: Optional[BaseBrokerAdapter] = None,
    actor_type: str = 'task',
    actor_id: str = '',
    fire_at: Optional[datetime] = None,
) -> OrderOutcome:
    """Convert an actionable `LiveSignal` into a broker order and a `LiveTrade`.

    Returns an `OrderOutcome` describing the result; the function never raises
    on broker errors (those are captured as `status='failed'` so the calling
    task can keep iterating across other symbols).
    """

    fire_at = fire_at or timezone.now()
    deployment = deployment_symbol.deployment

    if not signal.is_actionable:
        return OrderOutcome(status='skipped', reason='non_actionable_signal')

    broker_adapter = broker_adapter or get_adapter_for_deployment(deployment)
    if broker_adapter is None:
        message = (
            f"No broker adapter available for {deployment.broker.code} "
            f"({'paper' if deployment.deployment_type != 'real_money' else 'real'})"
        )
        log_event(
            deployment,
            deployment_symbol=deployment_symbol,
            event_type='order_failed',
            actor_type=actor_type,
            actor_id=actor_id,
            level='error',
            message=message,
            context={
                'ticker': deployment_symbol.symbol.ticker,
                'signal': signal.to_dict(),
                'fire_at': fire_at.isoformat(),
            },
        )
        return OrderOutcome(status='failed', reason='no_broker_adapter')

    if signal.is_exit:
        return _place_exit_order(
            deployment_symbol=deployment_symbol,
            signal=signal,
            broker_adapter=broker_adapter,
            actor_type=actor_type,
            actor_id=actor_id,
            fire_at=fire_at,
        )
    return _place_entry_order(
        deployment_symbol=deployment_symbol,
        signal=signal,
        broker_adapter=broker_adapter,
        actor_type=actor_type,
        actor_id=actor_id,
        fire_at=fire_at,
    )


def exit_open_trades_for_deployment(
    deployment: StrategyDeployment,
    *,
    broker_adapter: Optional[BaseBrokerAdapter] = None,
    actor_type: str = 'task',
    actor_id: str = '',
    fire_at: Optional[datetime] = None,
) -> dict:
    """Submit exit orders for every open trade on a deployment (deployment symbols only).

    This is used by the Stop action: it attempts to close positions belonging
    to this deployment without requiring a strategy signal.

    Returns a JSON-serialisable summary. Failures on one symbol do not stop the
    rest.
    """

    fire_at = fire_at or timezone.now()
    broker_adapter = broker_adapter or get_adapter_for_deployment(deployment)

    open_trades = list(
        deployment.live_trades.select_related('deployment_symbol', 'symbol')
        .filter(status='open')
        .order_by('symbol__ticker', 'entry_timestamp')
    )

    # De-duplicate by (deployment_symbol_id, position_mode) so we don't try to
    # close the same logical position twice in one stop request.
    unique: dict[tuple[int, str], LiveTrade] = {}
    missing_deployment_symbol = 0
    for trade in open_trades:
        if trade.deployment_symbol_id is None:
            missing_deployment_symbol += 1
            continue
        key = (trade.deployment_symbol_id, trade.position_mode)
        if key not in unique:
            unique[key] = trade

    attempted = 0
    placed = 0
    filled = 0
    failed = 0
    skipped = 0
    outcomes: list[dict] = []

    for trade in unique.values():
        ds = trade.deployment_symbol
        if ds is None:
            continue

        signal_action = SIGNAL_EXIT_LONG if trade.position_mode == SIGNAL_LONG else SIGNAL_EXIT_SHORT
        signal = LiveSignal(action=signal_action, confidence=1.0, price=None, context={'source': 'stop_exit_all'})

        attempted += 1
        try:
            outcome = place_signal_order(
                ds,
                signal,
                broker_adapter=broker_adapter,
                actor_type=actor_type,
                actor_id=actor_id,
                fire_at=fire_at,
            )
        except Exception as exc:  # pragma: no cover - defensive
            failed += 1
            log_event(
                deployment,
                deployment_symbol=ds,
                event_type='order_failed',
                actor_type=actor_type,
                actor_id=actor_id,
                level='error',
                message=f"{ds.symbol.ticker}: stop exit failed: {exc}",
                error=exc,
                context={'ticker': ds.symbol.ticker, 'reason': 'stop_exit_exception'},
            )
            continue

        outcomes.append({'ticker': ds.symbol.ticker, **outcome.to_dict()})
        if outcome.status == 'filled':
            filled += 1
        elif outcome.status == 'placed':
            placed += 1
        elif outcome.status == 'skipped':
            skipped += 1
        else:
            failed += 1

    summary = {
        'status': 'success',
        'deployment_id': deployment.id,
        'open_trade_count': len(open_trades),
        'unique_positions': len(unique),
        'missing_deployment_symbol': missing_deployment_symbol,
        'attempted': attempted,
        'placed': placed,
        'filled': filled,
        'skipped': skipped,
        'failed': failed,
        'fire_at': fire_at.isoformat(),
        'outcomes': outcomes,
    }

    log_event(
        deployment,
        event_type='deploy_stop_exit',
        actor_type=actor_type,
        actor_id=actor_id,
        message=(
            f"Stop: exit-all attempted={attempted} filled={filled} placed={placed} "
            f"skipped={skipped} failed={failed}"
        ),
        context=summary,
    )

    return summary


# ---------------------------------------------------------------------------
# Internals — entries
# ---------------------------------------------------------------------------


def _place_entry_order(
    *,
    deployment_symbol: DeploymentSymbol,
    signal: LiveSignal,
    broker_adapter: BaseBrokerAdapter,
    actor_type: str,
    actor_id: str,
    fire_at: datetime,
) -> OrderOutcome:
    deployment = deployment_symbol.deployment
    ticker = deployment_symbol.symbol.ticker
    position_mode = SIGNAL_LONG if signal.action == SIGNAL_LONG else SIGNAL_SHORT
    side = action_to_side(signal.action)

    # Global idempotency: if the symbol already has an open *main* trade anywhere,
    # don't enter again (hedge legs are excluded and may accumulate).
    duplicate = (
        LiveTrade.objects.filter(
            symbol=deployment_symbol.symbol,
            status='open',
        )
        # Use `contains` so rows with no key aren't excluded.
        .exclude(metadata__contains={'is_hedge_leg': True})
        .order_by('-entry_timestamp')
        .first()
    )
    if duplicate is not None:
        log_event(
            deployment,
            deployment_symbol=deployment_symbol,
            event_type='order_failed',
            actor_type=actor_type,
            actor_id=actor_id,
            level='warning',
            message=(
                f"{ticker}: skipping duplicate {position_mode} entry — "
                f"open trade #{duplicate.id} already exists."
            ),
            context={
                'ticker': ticker,
                'duplicate_trade_id': duplicate.id,
                'duplicate_deployment_id': duplicate.deployment_id,
                'signal': signal.to_dict(),
                'fire_at': fire_at.isoformat(),
            },
        )
        return OrderOutcome(
            status='skipped',
            reason='duplicate_open_trade_symbol_global',
            duplicate_trade_id=duplicate.id,
        )

    reference_price = _resolve_reference_price(signal, broker_adapter, ticker)
    if reference_price is None or reference_price <= 0:
        log_event(
            deployment,
            deployment_symbol=deployment_symbol,
            event_type='order_failed',
            actor_type=actor_type,
            actor_id=actor_id,
            level='error',
            message=f"{ticker}: cannot size order — no reference price available.",
            context={
                'ticker': ticker,
                'signal': signal.to_dict(),
                'fire_at': fire_at.isoformat(),
            },
        )
        return OrderOutcome(status='failed', reason='no_reference_price')

    quantity = _compute_quantity(deployment, broker_adapter, reference_price, ticker=ticker)
    if quantity is None or quantity <= 0:
        # Recompute sizing context for audit visibility.
        bet_pct = float(deployment.bet_size_percentage or 0)
        capital = _sizing_base_capital(deployment, broker_adapter)
        bet_amount = None
        try:
            bet_amount = (capital * (Decimal(str(bet_pct)) / Decimal('100'))) if capital else None
        except Exception:
            bet_amount = None

        log_event(
            deployment,
            deployment_symbol=deployment_symbol,
            event_type='order_failed',
            actor_type=actor_type,
            actor_id=actor_id,
            level='warning',
            message=(
                f"{ticker}: computed entry quantity is zero/none "
                f"(price={reference_price})."
            ),
            context={
                'ticker': ticker,
                'reference_price': str(reference_price),
                'bet_size_percentage': bet_pct,
                'initial_capital': str(deployment.initial_capital or 0),
                'sizing_base_capital': str(capital) if capital is not None else None,
                'sizing_base_note': 'account equity (cash + positions) or initial_capital fallback',
                'bet_amount': str(bet_amount) if bet_amount is not None else None,
            },
        )
        return OrderOutcome(status='failed', reason='zero_quantity')

    if getattr(deployment, 'hedge_enabled', False):
        return _place_hedged_entry_order(
            deployment_symbol=deployment_symbol,
            signal=signal,
            broker_adapter=broker_adapter,
            actor_type=actor_type,
            actor_id=actor_id,
            fire_at=fire_at,
            reference_price=reference_price,
            quantity=quantity,
            position_mode=position_mode,
            side=side,
            ticker=ticker,
        )

    log_event(
        deployment,
        deployment_symbol=deployment_symbol,
        event_type='order_placed',
        actor_type=actor_type,
        actor_id=actor_id,
        message=(
            f"{ticker}: submitting {side.upper()} {quantity} @ market"
            f" ({position_mode} entry)"
        ),
        context={
            'ticker': ticker,
            'side': side,
            'quantity': str(quantity),
            'reference_price': str(reference_price),
            'position_mode': position_mode,
            'signal': signal.to_dict(),
        },
    )

    try:
        order_result = broker_adapter.place_order(
            symbol=ticker,
            side=side,
            quantity=quantity,
            order_type='market',
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception(
            'Broker.place_order raised for deployment=%s ticker=%s',
            deployment.id, ticker,
        )
        log_event(
            deployment,
            deployment_symbol=deployment_symbol,
            event_type='order_failed',
            actor_type=actor_type,
            actor_id=actor_id,
            level='error',
            message=f"{ticker}: broker raised: {exc}",
            error=exc,
            context={'ticker': ticker, 'signal': signal.to_dict()},
        )
        return OrderOutcome(status='failed', reason='broker_exception')

    if order_result.status == 'rejected':
        log_event(
            deployment,
            deployment_symbol=deployment_symbol,
            event_type='order_failed',
            actor_type=actor_type,
            actor_id=actor_id,
            level='error',
            message=(
                f"{ticker}: broker rejected order "
                f"({order_result.error_message or 'no reason'})"
            ),
            context={
                'ticker': ticker,
                'order_status': order_result.status,
                'broker_order_id': order_result.broker_order_id,
                'error': order_result.error_message,
            },
        )
        return OrderOutcome(
            status='failed',
            reason='broker_rejected',
            order_result=order_result,
            quantity=quantity,
        )

    filled = order_result.filled_quantity or Decimal('0')
    entry_price = order_result.price if filled > 0 else reference_price

    entry_meta = {
        'signal': signal.to_dict(),
        'reference_price': str(reference_price),
        'requested_quantity': str(quantity),
        'broker': {
            'order_id': order_result.order_id,
            'broker_order_id': order_result.broker_order_id,
            'status': order_result.status,
            'filled_quantity': str(order_result.filled_quantity),
            'filled_price': str(order_result.price),
        },
    }
    enrich_flat_entry_metadata(entry_meta, entry_price=entry_price, quantity=quantity)

    with transaction.atomic():
        live_trade = LiveTrade.objects.create(
            deployment=deployment,
            deployment_symbol=deployment_symbol,
            symbol=deployment_symbol.symbol,
            position_mode=position_mode,
            trade_type=side,
            entry_price=entry_price,
            quantity=quantity,
            entry_timestamp=fire_at,
            status='open',
            broker_order_id=order_result.broker_order_id or order_result.order_id or '',
            metadata=entry_meta,
        )

    if filled > 0 and order_result.status in ('filled', 'partially_filled', 'partial'):
        log_event(
            deployment,
            deployment_symbol=deployment_symbol,
            event_type='order_filled',
            actor_type=actor_type,
            actor_id=actor_id,
            message=(
                f"{ticker}: filled {filled} @ {order_result.price}"
            ),
            context={
                'ticker': ticker,
                'live_trade_id': live_trade.id,
                'broker_order_id': order_result.broker_order_id,
                'filled_quantity': str(filled),
                'filled_price': str(order_result.price),
            },
        )

    log_event(
        deployment,
        deployment_symbol=deployment_symbol,
        event_type='trade_opened',
        actor_type=actor_type,
        actor_id=actor_id,
        message=(
            f"{ticker}: opened {position_mode} trade #{live_trade.id} "
            f"({quantity} @ {entry_price})"
        ),
        context={
            'ticker': ticker,
            'live_trade_id': live_trade.id,
            'position_mode': position_mode,
            'quantity': str(quantity),
            'entry_price': str(entry_price),
            'broker_order_id': live_trade.broker_order_id,
            'order_status': order_result.status,
        },
    )

    return OrderOutcome(
        status='placed' if filled <= 0 else 'filled',
        reason='entry_submitted',
        order_result=order_result,
        live_trade_id=live_trade.id,
        quantity=quantity,
    )


def _place_hedged_entry_order(
    *,
    deployment_symbol: DeploymentSymbol,
    signal: LiveSignal,
    broker_adapter: BaseBrokerAdapter,
    actor_type: str,
    actor_id: str,
    fire_at: datetime,
    reference_price: Decimal,
    quantity: Decimal,
    position_mode: str,
    side: str,
    ticker: str,
) -> OrderOutcome:
    """Split notional: strategy leg + hedge sleeve (VIXM or VIXY by regime; matches w_s / w_h)."""
    from backtest_engine.services.hybrid_vix_hedge import (
        live_hedge_weights_at,
        resolved_hedge_config_for_backtest,
    )

    # Backtest sleeve: in normal, daily hedge return = β·VIXM_Ret − VIXY_Ret (scaled); in panic,
    # long VIXY return. Live approximates with a *single* BUY in VIXM (normal) or VIXY (panic) for
    # the hedge slice — not a full two-leg spread. Strategy symbols must not include VIXM/VIXY
    # when hedge is on (see GapUpGapDownLiveEngine).

    deployment = deployment_symbol.deployment
    cfg = resolved_hedge_config_for_backtest(deployment.hedge_config or {})
    w_s, w_h, hw_meta = live_hedge_weights_at(fire_at, cfg, yahoo_only=True)

    # Choose which hedge instrument to trade based on the same regime logic used in the dashboard/backtest.
    hedge_ticker = HEDGE_TICKER_PANIC
    hedge_regime = None
    try:
        from backtest_engine.services.hybrid_vix_hedge import compute_hedge_panic_snapshot

        snap = compute_hedge_panic_snapshot(
            cfg,
            yahoo_only=True,
            end_at=fire_at,
            include_chart=False,
        )
        hedge_regime = snap.get('regime')
        hedge_ticker = HEDGE_TICKER_PANIC if hedge_regime == 'panic' else HEDGE_TICKER_NORMAL
    except Exception:  # pragma: no cover - defensive
        hedge_regime = None
        hedge_ticker = HEDGE_TICKER_PANIC
    if w_h < 1e-9 or w_s >= 1.0 - 1e-9:
        # Overlay unavailable for this day — full strategy size (no VIXY leg)
        w_s, w_h = 1.0, 0.0

    main_caps = {}
    try:
        main_caps = broker_adapter.get_symbol_capabilities(ticker) or {}
    except Exception:  # pragma: no cover
        main_caps = {}
    main_fractionable = bool(main_caps.get('fractionable'))

    strat_qty_d = (quantity * Decimal(str(w_s))) if w_h > 1e-9 else quantity
    if main_fractionable:
        strat_qty_d = strat_qty_d.quantize(Decimal('0.000001'), rounding=ROUND_DOWN)
    else:
        strat_qty_d = Decimal(int(strat_qty_d))

    if strat_qty_d is None or strat_qty_d <= 0 or ((not main_fractionable) and strat_qty_d < 1):
        log_event(
            deployment,
            deployment_symbol=deployment_symbol,
            event_type='order_failed',
            actor_type=actor_type,
            actor_id=actor_id,
            level='warning',
            message=f"{ticker}: hedged entry skipped — insufficient strategy quantity after split.",
            context={
                'ticker': ticker,
                'w_strategy': w_s,
                'w_hedge': w_h,
                'hedge_meta': hw_meta,
                'fractionable': main_fractionable,
                'computed_strategy_qty': str(strat_qty_d),
                'total_qty': str(quantity),
            },
        )
        return OrderOutcome(status='failed', reason='zero_strategy_shares_hedge')

    hedge_leg: dict = {
        'mode': 'vixy_proxy',
        'w_strategy': w_s,
        'w_hedge': w_h,
        'overlay': hw_meta,
        'regime': hedge_regime,
        'hedge_ticker': hedge_ticker,
    }

    if w_h > 1e-9:
        total_notional = reference_price * quantity
        hedge_dollars = float(total_notional) * w_h
        vixy_px: Optional[Decimal] = None
        try:
            vixy_px = broker_adapter.get_current_price(hedge_ticker)
        except Exception:  # pragma: no cover
            vixy_px = None
        if vixy_px and vixy_px > 0:
            hedge_qty = int(Decimal(str(hedge_dollars)) / vixy_px)
        else:
            hedge_qty = 0
        if hedge_qty < 1:
            hedge_leg['skipped'] = 'no_hedge_price_or_tiny'
        else:
            hedge_leg['ticker'] = hedge_ticker
            hedge_leg['target_quantity'] = str(hedge_qty)
        if hedge_qty >= 1:
            hedge_leg['ticker'] = hedge_ticker
            hedge_leg['target_quantity'] = str(hedge_qty)

    log_event(
        deployment,
        deployment_symbol=deployment_symbol,
        event_type='order_placed',
        actor_type=actor_type,
        actor_id=actor_id,
        message=(
            f"{ticker}: hedged entry {side.upper()} {strat_qty_d} @ market + "
            f"hedge w_s={w_s:.3f} w_h={w_h:.3f}"
        ),
        context={
            'ticker': ticker,
            'side': side,
            'strat_qty': str(strat_qty_d),
            'hedge_leg': hedge_leg,
            'position_mode': position_mode,
            'signal': signal.to_dict(),
        },
    )

    try:
        order_result = broker_adapter.place_order(
            symbol=ticker,
            side=side,
            quantity=strat_qty_d,
            order_type='market',
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception(
            'Broker.place_order (hedged entry, main) failed deployment=%s ticker=%s',
            deployment.id, ticker,
        )
        log_event(
            deployment,
            deployment_symbol=deployment_symbol,
            event_type='order_failed',
            actor_type=actor_type,
            actor_id=actor_id,
            level='error',
            message=f"{ticker}: broker raised: {exc}",
            error=exc,
            context={'ticker': ticker, 'signal': signal.to_dict()},
        )
        return OrderOutcome(status='failed', reason='broker_exception')

    if order_result.status == 'rejected':
        log_event(
            deployment,
            deployment_symbol=deployment_symbol,
            event_type='order_failed',
            actor_type=actor_type,
            actor_id=actor_id,
            level='error',
            message=f"{ticker}: broker rejected main leg (hedged entry)",
            context={
                'ticker': ticker,
                'side': side,
                'requested_quantity': str(strat_qty_d),
                'position_mode': position_mode,
                'w_strategy': w_s,
                'w_hedge': w_h,
                'hedge_leg': hedge_leg,
                'broker_rejection': {
                    'status': order_result.status,
                    'error_message': order_result.error_message,
                    'order_id': order_result.order_id,
                    'broker_order_id': order_result.broker_order_id,
                },
            },
        )
        return OrderOutcome(
            status='failed',
            reason='broker_rejected',
            order_result=order_result,
            quantity=strat_qty_d,
        )

    hedge_result = None
    h_qty = 0
    if (
        w_h > 1e-9
        and hedge_leg.get('ticker') in (HEDGE_TICKER_NORMAL, HEDGE_TICKER_PANIC)
        and (hedge_leg.get('target_quantity'))
    ):
        h_qty = int(hedge_leg['target_quantity'])
        try:
            hedge_result = broker_adapter.place_order(
                symbol=hedge_leg['ticker'],
                side='buy',
                quantity=Decimal(str(h_qty)),
                order_type='market',
            )
            hedge_leg['broker'] = {
                'order_id': hedge_result.order_id,
                'broker_order_id': hedge_result.broker_order_id,
                'status': hedge_result.status,
            }
            if hedge_result.status == 'rejected':
                log_event(
                    deployment,
                    deployment_symbol=deployment_symbol,
                    event_type='order_failed',
                    actor_type=actor_type,
                    actor_id=actor_id,
                    level='warning',
                    message=f"{ticker}: broker rejected hedge leg ({hedge_leg.get('ticker')})",
                    context={
                        'ticker': ticker,
                        'hedge_ticker': hedge_leg.get('ticker'),
                        'hedge_quantity': str(h_qty),
                        'broker_rejection': {
                            'status': hedge_result.status,
                            'error_message': hedge_result.error_message,
                            'order_id': hedge_result.order_id,
                            'broker_order_id': hedge_result.broker_order_id,
                        },
                    },
                )
        except Exception as exc:  # pragma: no cover
            logger.exception('Hedge VIXY order failed for %s', ticker)
            hedge_leg['hedge_error'] = str(exc)
            log_event(
                deployment,
                deployment_symbol=deployment_symbol,
                event_type='order_failed',
                actor_type=actor_type,
                actor_id=actor_id,
                level='warning',
                message=f"{ticker}: hedge leg raised ({hedge_leg.get('ticker')}): {exc}",
                error=exc,
                context={
                    'ticker': ticker,
                    'hedge_ticker': hedge_leg.get('ticker'),
                    'hedge_quantity': str(h_qty),
                },
            )

    filled = order_result.filled_quantity or Decimal('0')
    entry_price = order_result.price if filled > 0 else reference_price

    meta: dict = {
        'hedge_enabled': True,
        'w_strategy': w_s,
        'w_hedge': w_h,
        'hedge_leg': hedge_leg,
        'signal': signal.to_dict(),
        'reference_price': str(reference_price),
        'full_bet_shares': str(quantity),
        'main_opening_strategyleg_qty': str(strat_qty_d),
        'strategyleg_quantity': str(strat_qty_d),
        'requested_quantity': str(strat_qty_d),
        'broker': {
            'order_id': order_result.order_id,
            'broker_order_id': order_result.broker_order_id,
            'status': order_result.status,
            'filled_quantity': str(order_result.filled_quantity),
            'filled_price': str(order_result.price),
        },
    }
    enrich_hybrid_entry_main_metadata(
        meta, strat_entry_price=entry_price, strat_quantity=strat_qty_d,
    )

    with transaction.atomic():
        live_trade = LiveTrade.objects.create(
            deployment=deployment,
            deployment_symbol=deployment_symbol,
            symbol=deployment_symbol.symbol,
            position_mode=position_mode,
            trade_type=side,
            entry_price=entry_price,
            quantity=strat_qty_d,
            entry_timestamp=fire_at,
            status='open',
            broker_order_id=order_result.broker_order_id or order_result.order_id or '',
            metadata=meta,
        )

        # Persist the hedge leg as its own LiveTrade row so it shows up in the UI/trade ledger.
        if hedge_result is not None and hedge_result.status != 'rejected' and h_qty > 0:
            hedge_sym_ticker = hedge_leg.get('ticker') or HEDGE_TICKER_PANIC
            hedge_symbol = Symbol.objects.filter(ticker=hedge_sym_ticker).first()
            if hedge_symbol is None:
                log_event(
                    deployment,
                    deployment_symbol=deployment_symbol,
                    event_type='order_failed',
                    actor_type=actor_type,
                    actor_id=actor_id,
                    level='warning',
                    message=f"{ticker}: hedge leg filled/placed but Symbol({hedge_sym_ticker}) not found in DB",
                    context={'ticker': ticker, 'hedge_ticker': hedge_sym_ticker},
                )
            else:
                hedge_filled = hedge_result.filled_quantity or Decimal('0')
                hedge_entry_px = (
                    hedge_result.price
                    if hedge_filled > 0
                    else broker_adapter.get_current_price(hedge_sym_ticker) or Decimal('0')
                )
                target_decimal = Decimal(str(h_qty))
                hedge_qty_effective = target_decimal
                if hedge_filled > Decimal('0'):
                    hedge_qty_effective = min(hedge_filled, target_decimal)
                hedge_md = {
                    'is_hedge_leg': True,
                    'hedge_parent_ticker': ticker,
                    'hedge_parent_live_trade_id': live_trade.id,
                    'hedge_target_quantity': str(h_qty),
                    'overlay': hw_meta,
                    'w_strategy': w_s,
                    'w_hedge': w_h,
                    'hedge_ticker': hedge_sym_ticker,
                    'regime': hedge_regime,
                }
                enrich_hybrid_entry_hedge_metadata(
                    hedge_md,
                    hedge_entry_price=hedge_entry_px,
                    hedge_quantity=hedge_qty_effective,
                )
                hedge_row = LiveTrade.objects.create(
                    deployment=deployment,
                    deployment_symbol=deployment_symbol,
                    symbol=hedge_symbol,
                    position_mode='long',
                    trade_type='buy',
                    entry_price=hedge_entry_px,
                    quantity=hedge_qty_effective,
                    entry_timestamp=fire_at,
                    status='open',
                    broker_order_id=hedge_result.broker_order_id or hedge_result.order_id or '',
                    metadata=hedge_md,
                )
                main_refresh = dict(live_trade.metadata or {})
                patch_main_trade_hedge_leg_link(
                    main_refresh,
                    hedge_live_trade_id=hedge_row.id,
                    hedge_ticker=hedge_sym_ticker,
                    hedge_quantity=hedge_qty_effective,
                )
                live_trade.metadata = main_refresh
                live_trade.save(update_fields=['metadata', 'updated_at'])

    if filled > 0 and order_result.status in ('filled', 'partially_filled', 'partial'):
        log_event(
            deployment,
            deployment_symbol=deployment_symbol,
            event_type='order_filled',
            actor_type=actor_type,
            actor_id=actor_id,
            message=f"{ticker}: main leg filled {filled} @ {order_result.price}",
            context={'ticker': ticker, 'live_trade_id': live_trade.id},
        )

    log_event(
        deployment,
        deployment_symbol=deployment_symbol,
        event_type='trade_opened',
        actor_type=actor_type,
        actor_id=actor_id,
        message=(
            f"{ticker}: opened {position_mode} trade #{live_trade.id} (hedged) "
            f"({strat_qty_d} @ {entry_price})"
        ),
        context={
            'ticker': ticker,
            'live_trade_id': live_trade.id,
            'position_mode': position_mode,
            'quantity': str(strat_qty_d),
        },
    )

    return OrderOutcome(
        status='placed' if filled <= 0 else 'filled',
        reason='hedged_entry_submitted',
        order_result=order_result,
        live_trade_id=live_trade.id,
        quantity=strat_qty_d,
    )


def _place_exit_order(
    *,
    deployment_symbol: DeploymentSymbol,
    signal: LiveSignal,
    broker_adapter: BaseBrokerAdapter,
    actor_type: str,
    actor_id: str,
    fire_at: datetime,
) -> OrderOutcome:
    deployment = deployment_symbol.deployment
    ticker = deployment_symbol.symbol.ticker
    position_mode = SIGNAL_LONG if signal.action == SIGNAL_EXIT_LONG else SIGNAL_SHORT
    closing_side = action_to_side(signal.action)

    open_trade = (
        deployment_symbol.live_trades.filter(
            status='open', position_mode=position_mode,
        )
        .order_by('-entry_timestamp')
        .first()
    )
    if open_trade is None:
        log_event(
            deployment,
            deployment_symbol=deployment_symbol,
            event_type='order_failed',
            actor_type=actor_type,
            actor_id=actor_id,
            level='warning',
            message=(
                f"{ticker}: exit signal received but no open {position_mode} trade."
            ),
            context={
                'ticker': ticker,
                'signal': signal.to_dict(),
                'fire_at': fire_at.isoformat(),
            },
        )
        return OrderOutcome(status='skipped', reason='no_open_trade_for_exit')

    hedge_children_before = list(_hedge_leg_child_trades(open_trade))
    hedge_inventory = [
        {'live_trade_id': h.id, 'hedge_ticker': h.symbol.ticker}
        for h in hedge_children_before
    ]

    log_event(
        deployment,
        deployment_symbol=deployment_symbol,
        event_type='order_placed',
        actor_type=actor_type,
        actor_id=actor_id,
        message=(
            f"{ticker}: submitting {closing_side.upper()} {open_trade.quantity} "
            f"to close {position_mode} trade #{open_trade.id}; "
            f"hedge legs to unwind: {[h['hedge_ticker'] for h in hedge_inventory]}"
        ),
        context={
            'main_ticker': ticker,
            'ticker': ticker,
            'side': closing_side,
            'quantity': str(open_trade.quantity),
            'live_trade_id': open_trade.id,
            'hedge_symbols': [h['hedge_ticker'] for h in hedge_inventory],
            'hedge_leg_live_trade_ids': [h['live_trade_id'] for h in hedge_inventory],
            'signal': signal.to_dict(),
            'source': 'exit_signal',
        },
    )

    try:
        order_result = broker_adapter.place_order(
            symbol=ticker,
            side=closing_side,
            quantity=open_trade.quantity,
            order_type='market',
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception(
            'Broker.place_order (exit) raised for deployment=%s ticker=%s',
            deployment.id, ticker,
        )
        log_event(
            deployment,
            deployment_symbol=deployment_symbol,
            event_type='order_failed',
            actor_type=actor_type,
            actor_id=actor_id,
            level='error',
            message=f"{ticker}: broker raised on exit: {exc}",
            error=exc,
            context={'ticker': ticker, 'live_trade_id': open_trade.id},
        )
        return OrderOutcome(
            status='failed',
            reason='broker_exception',
            live_trade_id=open_trade.id,
        )

    if order_result.status == 'rejected':
        log_event(
            deployment,
            deployment_symbol=deployment_symbol,
            event_type='order_failed',
            actor_type=actor_type,
            actor_id=actor_id,
            level='error',
            message=(
                f"{ticker}: broker rejected exit order "
                f"({order_result.error_message or 'no reason'})"
            ),
            context={
                'ticker': ticker,
                'live_trade_id': open_trade.id,
                'broker_order_id': order_result.broker_order_id,
                'error': order_result.error_message,
            },
        )
        return OrderOutcome(
            status='failed',
            reason='broker_rejected',
            order_result=order_result,
            live_trade_id=open_trade.id,
        )

    _submit_hedge_leg_exits_for_main_trade(
        open_trade,
        broker_adapter=broker_adapter,
        fire_at=fire_at,
        source='exit_signal',
        actor_type=actor_type,
        actor_id=actor_id,
    )

    filled = order_result.filled_quantity or Decimal('0')
    meta = dict(open_trade.metadata or {})
    meta['exit'] = {
        'broker_order_id': order_result.broker_order_id,
        'order_id': order_result.order_id,
        'status': order_result.status,
        'filled_quantity': str(order_result.filled_quantity),
        'filled_price': str(order_result.price),
        'signal': signal.to_dict(),
        'requested_at': fire_at.isoformat(),
    }
    open_trade.metadata = meta
    open_trade.save(update_fields=['metadata', 'updated_at'])

    # Only mark the trade closed once the broker reports a fill. When the exit
    # order is accepted but not filled, we leave the trade `open` and rely on
    # position-sync (`update_open_trades`) to close it once exposure is gone.
    if filled <= 0:
        return OrderOutcome(
            status='placed',
            reason='exit_submitted',
            order_result=order_result,
            live_trade_id=open_trade.id,
            quantity=open_trade.quantity,
        )

    oq = open_trade.quantity or Decimal('0')
    qty_closed = oq
    if filled > 0:
        try:
            fc = Decimal(str(filled))
            if fc > 0:
                qty_closed = min(fc, oq)
        except Exception:
            pass
    exit_price = _resolve_exit_execution_price(
        order_price=order_result.price,
        entry_price_fallback=open_trade.entry_price,
        metadata_exit_subset=meta.get('exit') if isinstance(meta.get('exit'), dict) else {},
        broker_adapter=broker_adapter,
        quote_symbol=ticker,
    )

    pnl = _compute_pnl(open_trade.entry_price, exit_price, qty_closed, position_mode)
    pnl_pct = _safe_pct(open_trade.entry_price, exit_price, position_mode)

    with transaction.atomic():
        open_trade.refresh_from_db()
        merge_meta = dict(open_trade.metadata or {})
        ex_inner = merge_meta.get('exit') if isinstance(merge_meta.get('exit'), dict) else {}
        ex_inner_merge = dict(ex_inner)
        ex_inner_merge['executed_avg_price'] = str(exit_price)
        merge_meta['exit'] = ex_inner_merge
        if not merge_meta.get('is_hedge_leg') and (
            merge_meta.get('hedge_enabled') or getattr(deployment, 'hedge_enabled', False)
        ):
            merge_meta = merge_exit_rollups_for_main_trade(
                merge_meta,
                main_pnl=pnl,
                parent_trade_id=open_trade.id,
                closed_main_quantity=qty_closed,
            )
        open_trade.metadata = merge_meta
        open_trade.exit_price = exit_price
        open_trade.exit_timestamp = fire_at
        open_trade.pnl = pnl
        open_trade.pnl_percentage = pnl_pct
        open_trade.is_winner = pnl is not None and pnl > 0
        open_trade.status = 'closed'
        open_trade.save(
            update_fields=[
                'metadata',
                'exit_price', 'exit_timestamp', 'pnl', 'pnl_percentage',
                'is_winner', 'status', 'updated_at',
            ],
        )

    if filled > 0:
        log_event(
            deployment,
            deployment_symbol=deployment_symbol,
            event_type='order_filled',
            actor_type=actor_type,
            actor_id=actor_id,
            message=(
                f"{ticker}: exit filled {filled} @ {exit_price}"
            ),
            context={
                'ticker': ticker,
                'live_trade_id': open_trade.id,
                'broker_order_id': order_result.broker_order_id,
                'filled_quantity': str(filled),
                'filled_price': str(exit_price),
            },
        )

    log_event(
        deployment,
        deployment_symbol=deployment_symbol,
        event_type='trade_closed',
        actor_type=actor_type,
        actor_id=actor_id,
        message=(
            f"{ticker} exit: main leg #{open_trade.id} closed (PnL={pnl}); "
            f"VIX hedge rows submitted for unwind: {[h['hedge_ticker'] for h in hedge_inventory]}"
        ),
        context={
            'main_ticker': ticker,
            'ticker': ticker,
            'live_trade_id': open_trade.id,
            'main_live_trade_id': open_trade.id,
            'position_mode': position_mode,
            'entry_price': str(open_trade.entry_price),
            'exit_price': str(exit_price),
            'pnl': str(pnl) if pnl is not None else None,
            'pnl_percentage': str(pnl_pct) if pnl_pct is not None else None,
            'is_winner': open_trade.is_winner,
            'broker_order_id': order_result.broker_order_id,
            'hedge_symbols': [h['hedge_ticker'] for h in hedge_inventory],
            'hedge_leg_live_trade_ids': [h['live_trade_id'] for h in hedge_inventory],
            'source': 'exit_signal',
        },
    )

    return OrderOutcome(
        status='filled',
        reason='exit_submitted',
        order_result=order_result,
        live_trade_id=open_trade.id,
        quantity=open_trade.quantity,
    )


# ---------------------------------------------------------------------------
# Position polling reconciliation
# ---------------------------------------------------------------------------


def _try_close_main_trade_via_exit_order_metadata(
    trade: LiveTrade,
    broker_adapter: BaseBrokerAdapter,
    deployment: StrategyDeployment,
    *,
    actor_type: str,
    actor_id: str,
) -> bool:
    """When a market close was submitted but returned 0 fill, finalize once the exit order fills.

    Manual/signal exits persist ``metadata['exit']['broker_order_id']``. Unlike hedge legs, the
    main row's model ``broker_order_id`` often still points at the *entry* order, so polling
    that id never observes the close — this path mirrors
    ``_try_close_hedge_leg_via_exit_order_metadata`` for strategy sleeve rows.
    """
    if trade.status != 'open':
        return False
    m = trade.metadata or {}
    if m.get('is_hedge_leg'):
        return False
    exit_block_raw = m.get('exit') or {}
    if not isinstance(exit_block_raw, dict):
        return False
    exit_oid = exit_block_raw.get('broker_order_id') or exit_block_raw.get('order_id')
    if not exit_oid:
        return False
    if exit_block_raw.get('source') == 'manual_close_force':
        return False

    try:
        ord_res = broker_adapter.get_order_status(str(exit_oid).strip())
    except Exception:  # pragma: no cover - network
        logger.exception('Exit order poll failed trade=%s order=%s', trade.id, exit_oid)
        return False

    st = (ord_res.status or '').lower()
    if st in ('rejected', 'canceled', 'cancelled', 'expired'):
        return False
    if st in (
        'new', 'pending_new', 'accepted', 'pending_replace',
        'pending_cancel', 'accepted_for_bidding',
    ):
        return False

    filled = ord_res.filled_quantity or Decimal('0')
    if filled <= 0 and st in ('filled', 'done', 'complete'):
        filled = trade.quantity or ord_res.quantity or Decimal('0')
    if filled <= 0 and st in ('partially_filled', 'partial'):
        return False
    if filled <= 0:
        return False

    ord_for_finalize = replace(ord_res, filled_quantity=filled)

    fire_at = timezone.now()
    ts = exit_block_raw.get('requested_at')
    if ts:
        try:
            from django.utils.dateparse import parse_datetime

            p = parse_datetime(str(ts).replace('Z', '+00:00'))
            if p is not None:
                fire_at = p
        except Exception:
            pass

    quantity_closed = filled
    meta = dict(trade.metadata or {})
    exit_block = dict(exit_block_raw)
    exit_block['synced_status'] = ord_res.status
    exit_block['synced_filled_quantity'] = str(filled)
    exit_block['synced_avg_price'] = str(ord_for_finalize.price or '')
    meta['exit'] = exit_block

    if quantity_closed != trade.quantity:
        meta['quantity_reconciled_to_fill'] = {
            'db_quantity': str(trade.quantity),
            'closed_quantity': str(quantity_closed),
        }

    exit_price = _resolve_exit_execution_price(
        order_price=ord_for_finalize.price,
        entry_price_fallback=trade.entry_price,
        metadata_exit_subset=exit_block,
        broker_adapter=broker_adapter,
        quote_symbol=getattr(trade.symbol, 'ticker', None) if trade.symbol_id else None,
    )
    exit_block['executed_avg_price'] = str(exit_price)
    meta['exit'] = exit_block
    pnl = _compute_pnl(
        trade.entry_price, exit_price, quantity_closed, trade.position_mode,
    )
    pnl_pct = _safe_pct(trade.entry_price, exit_price, trade.position_mode)
    if not meta.get('is_hedge_leg') and (
        meta.get('hedge_enabled') or getattr(trade.deployment, 'hedge_enabled', False)
    ):
        meta = merge_exit_rollups_for_main_trade(
            meta,
            main_pnl=pnl,
            parent_trade_id=trade.id,
            closed_main_quantity=quantity_closed,
        )

    with transaction.atomic():
        trade.refresh_from_db()
        if trade.status != 'open':
            return False
        trade.metadata = meta
        if quantity_closed != trade.quantity:
            trade.quantity = quantity_closed
        trade.exit_price = exit_price
        trade.exit_timestamp = fire_at
        trade.pnl = pnl
        trade.pnl_percentage = pnl_pct
        trade.is_winner = pnl is not None and pnl > 0
        trade.status = 'closed'
        trade.save(
            update_fields=[
                'exit_price', 'exit_timestamp', 'pnl', 'pnl_percentage',
                'is_winner', 'status', 'quantity', 'metadata', 'updated_at',
            ],
        )

    log_event(
        deployment,
        deployment_symbol=trade.deployment_symbol,
        event_type='trade_closed',
        actor_type=actor_type,
        actor_id=actor_id,
        message=(
            f"{trade.symbol.ticker}: main leg #{trade.id} closed via exit order sync "
            f"(order {exit_oid})"
        ),
        context={
            'ticker': trade.symbol.ticker,
            'live_trade_id': trade.id,
            'source': 'exit_order_sync_main',
            'exit_order_id': str(exit_oid),
            'pnl': str(pnl) if pnl is not None else None,
        },
    )
    return True


def _try_close_hedge_leg_via_exit_order_metadata(
    trade: LiveTrade,
    broker_adapter: BaseBrokerAdapter,
    deployment: StrategyDeployment,
    *,
    actor_type: str,
    actor_id: str,
) -> bool:
    """If this is an open hedge with a recorded exit order, close the row when that order fills.

    We cannot rely on ``get_position(VIXM) == 0`` alone: the account may still
    hold VIXM/VIXY for other strategies, so net quantity never hits zero even
    after *our* hedge exit fills.
    """
    m = trade.metadata or {}
    if not m.get('is_hedge_leg') or trade.status != 'open':
        return False
    exit_block = m.get('exit') or {}
    exit_oid = exit_block.get('broker_order_id') or exit_block.get('order_id')
    if not exit_oid:
        return False
    try:
        ord_res = broker_adapter.get_order_status(str(exit_oid).strip())
    except Exception:  # pragma: no cover - network
        return False
    st = (ord_res.status or '').lower()
    if st in ('rejected', 'canceled', 'cancelled', 'expired'):
        return False
    if st in (
        'new', 'pending_new', 'accepted', 'pending_replace',
        'pending_cancel', 'accepted_for_bidding',
    ):
        return False
    filled = ord_res.filled_quantity or Decimal('0')
    if filled <= 0 and st in ('filled', 'done', 'complete'):
        # Some responses omit filled_qty even when the order is done.
        filled = trade.quantity or ord_res.quantity or Decimal('0')
    if filled <= 0 and st in ('partially_filled', 'partial'):
        return False
    if filled <= 0:
        return False

    # _finalize_hedge_leg_on_exit_fill bails on filled_quantity<=0; broker GET may
    # still return 0 for filled_qty while status is already filled.
    ord_for_finalize = replace(ord_res, filled_quantity=filled)

    fire_at = timezone.now()
    ts = exit_block.get('requested_at')
    if ts:
        try:
            from django.utils.dateparse import parse_datetime

            p = parse_datetime(str(ts).replace('Z', '+00:00'))
            if p is not None:
                fire_at = p
        except Exception:
            pass

    ex_up = dict(exit_block) if isinstance(exit_block, dict) else {}
    ex_up['synced_status'] = ord_res.status
    ex_up['synced_filled_quantity'] = str(filled)
    if ord_for_finalize.price is not None:
        ex_up['synced_avg_price'] = str(ord_for_finalize.price)
    tm = dict(trade.metadata or {})
    tm['exit'] = ex_up
    trade.metadata = tm

    _finalize_hedge_leg_on_exit_fill(
        trade, ord_for_finalize, fire_at, broker_adapter=broker_adapter,
    )
    log_event(
        deployment,
        deployment_symbol=trade.deployment_symbol,
        event_type='trade_closed',
        actor_type=actor_type,
        actor_id=actor_id,
        message=(
            f"{trade.symbol.ticker}: hedge leg #{trade.id} closed via exit order sync"
        ),
        context={
            'ticker': trade.symbol.ticker,
            'live_trade_id': trade.id,
            'source': 'exit_order_sync',
            'exit_order_id': str(exit_oid),
        },
    )
    return True


def _reconcile_open_main_trade_quantity_to_broker(
    trade: LiveTrade,
    position: PositionInfo,
    deployment: StrategyDeployment,
    *,
    actor_type: str,
    actor_id: str,
) -> bool:
    """If the broker holds fewer shares than this open main row claims, shrink ``quantity``.

    Alpaca (and manual partial exits outside the app) can leave the ledger claiming e.g. 13
    shares while the account only has ~1.1 — manual close then sells the broker cap, but the
    UI and statistics stay wrong until we align the row to the broker's net position.

    Hedge legs are skipped: VIXM/VIXY quantity is often a slice of a shared account position.
    """
    if trade.status != 'open':
        return False
    meta0 = trade.metadata or {}
    if meta0.get('is_hedge_leg'):
        return False

    pq = position.quantity or Decimal('0')
    if pq == 0:
        return False
    tq = trade.quantity or Decimal('0')
    if tq <= 0:
        return False

    new_q: Optional[Decimal] = None
    if trade.position_mode == 'long':
        if position.position_type != 'long' or pq <= 0:
            return False
        if pq >= tq:
            return False
        new_q = pq
    elif trade.position_mode == 'short':
        if position.position_type != 'short' or pq >= 0:
            return False
        cover = abs(pq)
        if cover >= tq:
            return False
        new_q = cover
    else:
        return False

    if new_q is None or new_q <= 0:
        return False

    with transaction.atomic():
        trade.refresh_from_db()
        if trade.status != 'open':
            return False
        meta = dict(trade.metadata or {})
        log_entry = {
            'from_db': str(trade.quantity),
            'to_broker_qty': str(new_q),
            'synced_at': timezone.now().isoformat(),
        }
        prev = meta.get('broker_quantity_reconcile')
        if isinstance(prev, list):
            prev = prev + [log_entry]
        elif prev:
            prev = [prev, log_entry]
        else:
            prev = [log_entry]
        meta['broker_quantity_reconcile'] = prev[-10:]
        trade.quantity = new_q
        trade.metadata = meta
        trade.save(update_fields=['quantity', 'metadata', 'updated_at'])

    log_event(
        deployment,
        deployment_symbol=trade.deployment_symbol,
        event_type='info',
        actor_type=actor_type,
        actor_id=actor_id,
        message=(
            f"{trade.symbol.ticker}: reconciled open trade #{trade.id} quantity "
            f"{tq} -> {new_q} (broker-held shares lower than DB row)"
        ),
        context={
            'ticker': trade.symbol.ticker,
            'live_trade_id': trade.id,
            'source': 'broker_quantity_reconcile',
            'from_quantity': str(tq),
            'to_quantity': str(new_q),
        },
    )
    return True


def update_open_trades(
    deployment: StrategyDeployment,
    *,
    broker_adapter: Optional[BaseBrokerAdapter] = None,
    actor_type: str = 'task',
    actor_id: str = '',
) -> dict:
    """Reconcile open `LiveTrade` rows against the broker's reported positions.

    For every open trade we fetch the current broker position; if the broker
    has zero exposure the trade is closed (using the broker's last reported
    `current_price` if available, falling back to the entry price). Trades
    that the broker still tracks have their `metadata['last_position_sync']`
    refreshed but remain `open`.

    When the broker reports a smaller position than ``trade.quantity`` (main sleeves only),
    we shrink ``quantity`` to match so closes and KPIs reflect reality.
    """

    broker_adapter = broker_adapter or get_adapter_for_deployment(deployment)
    if broker_adapter is None:
        log_event(
            deployment,
            event_type='positions_synced',
            actor_type=actor_type,
            actor_id=actor_id,
            level='warning',
            message='No broker adapter available; cannot reconcile positions.',
        )
        return {'status': 'noop', 'reason': 'no_adapter'}

    open_trades = list(
        deployment.live_trades.select_related('symbol', 'deployment_symbol')
        .filter(status='open')
    )
    closed = 0
    refreshed = 0
    order_refreshed = 0
    order_terminal = 0
    quantity_reconciled = 0
    errors = 0
    for trade in open_trades:
        # Hedge sleeve: close from the *sell* order id in metadata, not net position in
        # the symbol (other deployments may still hold VIXM/VIXY).
        if _try_close_hedge_leg_via_exit_order_metadata(
            trade,
            broker_adapter,
            deployment,
            actor_type=actor_type,
            actor_id=actor_id,
        ):
            closed += 1
            continue

        if _try_close_main_trade_via_exit_order_metadata(
            trade,
            broker_adapter,
            deployment,
            actor_type=actor_type,
            actor_id=actor_id,
        ):
            closed += 1
            continue

        # 1) Refresh broker order status first.
        # This prevents us from incorrectly "closing" a trade just because the position
        # hasn't appeared yet while the order is still pending.
        pending_order = False
        if trade.broker_order_id:
            try:
                ord_status = broker_adapter.get_order_status(trade.broker_order_id)
                meta = dict(trade.metadata or {})
                meta['last_order_sync'] = {
                    'status': ord_status.status,
                    'filled_quantity': str(ord_status.filled_quantity),
                    'filled_price': str(ord_status.price),
                    'synced_at': timezone.now().isoformat(),
                    'error_message': ord_status.error_message,
                }
                trade.metadata = meta
                trade.save(update_fields=['metadata', 'updated_at'])
                order_refreshed += 1

                st = (ord_status.status or '').lower()
                if st in ('new', 'accepted', 'accepted_for_bidding', 'pending_new', 'pending_cancel', 'pending_replace'):
                    pending_order = True
                if st in ('canceled', 'expired', 'rejected'):
                    # If the broker says the order is terminal and we also have no position,
                    # mark the trade as cancelled so it doesn't block new entries forever.
                    try:
                        position = broker_adapter.get_position(trade.symbol.ticker)
                    except Exception:  # pragma: no cover
                        position = None
                    if position is None or (position.quantity or Decimal('0')) == 0:
                        with transaction.atomic():
                            trade.status = 'cancelled'
                            meta = dict(trade.metadata or {})
                            meta['closed_via_order_sync'] = True
                            trade.metadata = meta
                            trade.save(update_fields=['status', 'metadata', 'updated_at'])
                        log_event(
                            deployment,
                            deployment_symbol=trade.deployment_symbol,
                            event_type='order_failed',
                            actor_type=actor_type,
                            actor_id=actor_id,
                            level='warning',
                            message=(
                                f"{trade.symbol.ticker}: order terminal ({ord_status.status}); "
                                f"trade #{trade.id} marked cancelled."
                            ),
                            context={
                                'ticker': trade.symbol.ticker,
                                'live_trade_id': trade.id,
                                'broker_order_id': trade.broker_order_id,
                                'order_status': ord_status.status,
                                'error_message': ord_status.error_message,
                                'source': 'order_sync',
                            },
                        )
                        order_terminal += 1
                        continue
            except Exception as exc:  # pragma: no cover - defensive
                errors += 1
                log_event(
                    deployment,
                    deployment_symbol=trade.deployment_symbol,
                    event_type='error',
                    actor_type=actor_type,
                    actor_id=actor_id,
                    level='warning',
                    message=(
                        f"Order status lookup raised for {trade.symbol.ticker}: {exc}"
                    ),
                    error=exc,
                    context={
                        'ticker': trade.symbol.ticker,
                        'live_trade_id': trade.id,
                        'broker_order_id': trade.broker_order_id,
                    },
                )

        try:
            position = broker_adapter.get_position(trade.symbol.ticker)
        except Exception as exc:  # pragma: no cover - defensive
            errors += 1
            log_event(
                deployment,
                deployment_symbol=trade.deployment_symbol,
                event_type='error',
                actor_type=actor_type,
                actor_id=actor_id,
                level='warning',
                message=(
                    f"Position lookup raised for {trade.symbol.ticker}: {exc}"
                ),
                error=exc,
                context={'ticker': trade.symbol.ticker, 'live_trade_id': trade.id},
            )
            continue

        # If the order is pending and the broker has not opened a position yet,
        # keep the trade open (do not close it as "missing position").
        if pending_order and (position is None or (position.quantity or Decimal('0')) == 0):
            refreshed += 1
            continue

        if position is not None and (position.quantity or Decimal('0')) != 0:
            if _reconcile_open_main_trade_quantity_to_broker(
                trade,
                position,
                deployment,
                actor_type=actor_type,
                actor_id=actor_id,
            ):
                quantity_reconciled += 1
                try:
                    position = broker_adapter.get_position(trade.symbol.ticker)
                except Exception:  # pragma: no cover
                    position = None

        if position is None or (position.quantity or Decimal('0')) == 0:
            # Broker has no position -> close the trade.
            exit_price = (
                position.current_price if position is not None and position.current_price
                else trade.entry_price
            )
            pnl = _compute_pnl(
                trade.entry_price, exit_price, trade.quantity, trade.position_mode,
            )
            pnl_pct = _safe_pct(trade.entry_price, exit_price, trade.position_mode)
            with transaction.atomic():
                trade.exit_price = exit_price
                trade.exit_timestamp = timezone.now()
                trade.pnl = pnl
                trade.pnl_percentage = pnl_pct
                trade.is_winner = pnl is not None and pnl > 0
                trade.status = 'closed'
                meta = dict(trade.metadata or {})
                meta['closed_via_position_sync'] = True
                if not meta.get('is_hedge_leg') and (
                    meta.get('hedge_enabled')
                    or getattr(trade.deployment, 'hedge_enabled', False)
                ):
                    meta = merge_exit_rollups_for_main_trade(
                        meta,
                        main_pnl=pnl,
                        parent_trade_id=trade.id,
                        closed_main_quantity=trade.quantity,
                    )
                trade.metadata = meta
                trade.save(
                    update_fields=[
                        'exit_price', 'exit_timestamp', 'pnl', 'pnl_percentage',
                        'is_winner', 'status', 'metadata', 'updated_at',
                    ],
                )
            log_event(
                deployment,
                deployment_symbol=trade.deployment_symbol,
                event_type='trade_closed',
                actor_type=actor_type,
                actor_id=actor_id,
                message=(
                    f"{trade.symbol.ticker}: position-sync closed trade "
                    f"#{trade.id} PnL={pnl}"
                ),
                context={
                    'ticker': trade.symbol.ticker,
                    'live_trade_id': trade.id,
                    'pnl': str(pnl) if pnl is not None else None,
                    'pnl_percentage': str(pnl_pct) if pnl_pct is not None else None,
                    'source': 'position_sync',
                },
            )
            closed += 1
            continue

        meta = dict(trade.metadata or {})
        meta['last_position_sync'] = {
            'quantity': str(position.quantity),
            'current_price': str(position.current_price),
            'unrealized_pnl': str(position.unrealized_pnl),
            'synced_at': timezone.now().isoformat(),
        }
        trade.metadata = meta
        trade.save(update_fields=['metadata', 'updated_at'])
        refreshed += 1

    summary = {
        'status': 'success',
        'open_count': len(open_trades),
        'closed': closed,
        'refreshed': refreshed,
        'orders_refreshed': order_refreshed,
        'orders_terminal': order_terminal,
        'quantity_reconciled': quantity_reconciled,
        'errors': errors,
    }
    log_event(
        deployment,
        event_type='positions_synced',
        actor_type=actor_type,
        actor_id=actor_id,
        message=(
            f"Position sync: {len(open_trades)} open, "
            f"{closed} closed, {refreshed} refreshed."
        ),
        context=summary,
    )
    return summary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_reference_price(
    signal: LiveSignal,
    broker_adapter: BaseBrokerAdapter,
    ticker: str,
) -> Optional[Decimal]:
    if signal.price is not None and signal.price > 0:
        return signal.price
    try:
        return broker_adapter.get_current_price(ticker)
    except Exception:  # pragma: no cover - defensive
        logger.exception('Broker.get_current_price raised for %s', ticker)
        return None


# Minimum bet notional (equity * bet_pct) below which we stop scanning new entries.
LIVE_ENTRY_MIN_NOTIONAL_USD = Decimal('2')


def entry_phase_bankroll_exhausted(
    deployment: StrategyDeployment,
    broker_adapter: BaseBrokerAdapter,
) -> bool:
    """True when deployable sizing base cannot fund a minimal new entry."""
    cap = _sizing_base_capital(deployment, broker_adapter)
    if cap is None or cap <= 0:
        return True
    bet_pct = Decimal(str(deployment.bet_size_percentage or 0)) / Decimal('100')
    if bet_pct <= 0:
        return True
    return (cap * bet_pct) < LIVE_ENTRY_MIN_NOTIONAL_USD


def _sizing_base_capital(
    deployment: StrategyDeployment,
    broker_adapter: BaseBrokerAdapter,
) -> Optional[Decimal]:
    """Dollar base for `bet_size_percentage`: total equity, not cash alone.

    Alpaca (and other adapters) map this to **account equity** = cash + open
    positions value, so e.g. 5% of portfolio stays stable as cash is deployed
    into stocks. If equity cannot be read, fall back to ``initial_capital``.
    """
    capital: Optional[Decimal] = None
    try:
        capital = broker_adapter.get_account_equity()
    except Exception:  # pragma: no cover - defensive
        capital = None
    if capital is None or capital <= 0:
        capital = Decimal(str(deployment.initial_capital or 0))
    if capital <= 0:
        return None
    return capital


def _compute_quantity(
    deployment: StrategyDeployment,
    broker_adapter: BaseBrokerAdapter,
    reference_price: Decimal,
    *,
    ticker: str,
) -> Optional[Decimal]:
    bet_pct = Decimal(str(deployment.bet_size_percentage or 0)) / Decimal('100')
    if bet_pct <= 0:
        return None

    capital = _sizing_base_capital(deployment, broker_adapter)
    if capital is None or capital <= 0:
        return None

    bet_amount = capital * bet_pct
    if bet_amount <= 0:
        return None

    raw_qty = bet_amount / reference_price
    caps = {}
    try:
        caps = broker_adapter.get_symbol_capabilities(ticker) or {}
    except Exception:  # pragma: no cover - defensive
        caps = {}
    fractionable = bool(caps.get('fractionable'))
    try:
        if fractionable:
            qty = raw_qty.quantize(Decimal('0.000001'), rounding=ROUND_DOWN)
            return qty if qty > 0 else None
        qty_i = int(raw_qty)
        return Decimal(qty_i) if qty_i >= 1 else None
    except (InvalidOperation, ValueError):
        return None


def _compute_pnl(
    entry_price: Decimal,
    exit_price: Decimal,
    quantity: Decimal,
    position_mode: str,
) -> Optional[Decimal]:
    if entry_price is None or exit_price is None or quantity is None:
        return None
    direction = Decimal('1') if position_mode == SIGNAL_LONG else Decimal('-1')
    return (Decimal(exit_price) - Decimal(entry_price)) * Decimal(quantity) * direction


def _safe_pct(
    entry_price: Decimal,
    exit_price: Decimal,
    position_mode: str,
) -> Optional[Decimal]:
    if not entry_price:
        return None
    direction = Decimal('1') if position_mode == SIGNAL_LONG else Decimal('-1')
    raw = ((Decimal(exit_price) - Decimal(entry_price)) / Decimal(entry_price)) * Decimal('100') * direction
    return raw.quantize(Decimal('0.0001'))
