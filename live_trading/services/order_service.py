"""Live order placement + LiveTrade lifecycle service.

Glue between an actionable `LiveSignal` (produced by the live engine) and the
underlying `BrokerAdapter` / `LiveTrade` model. The orchestration tasks call
`place_signal_order(...)` once a signal is fired; this service is the only
place the rest of the codebase reaches into to actually submit an order.

Responsibilities:

- Translate a `LiveSignal` into broker call parameters (side / quantity).
- Compute the trade quantity from the deployment's bet sizing rules and the
  broker's available cash (falling back to the deployment's
  `initial_capital` when the broker call fails — useful for paper smoke-tests
  and offline development).
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
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Optional

from django.db import transaction
from django.utils import timezone

from ..adapters.base import BaseBrokerAdapter, OrderResult
from ..adapters.factory import get_broker_adapter
from ..engines.base import (
    SIGNAL_EXIT_LONG,
    SIGNAL_EXIT_SHORT,
    SIGNAL_LONG,
    SIGNAL_SHORT,
    LiveSignal,
)
from ..models import DeploymentSymbol, LiveTrade, StrategyDeployment
from .audit import log_event

logger = logging.getLogger(__name__)

# Live hedge sleeve uses a single liquid vol proxy; overlay math matches backtest sleeve notional.
HEDGE_PROXY_SYMBOL = 'VIXY'


class OrderPlacementError(Exception):
    """Raised when a broker order cannot be placed (validation level)."""


@dataclass
class OrderOutcome:
    """Result of an order attempt, returned to the caller for the audit feed."""

    status: str  # 'placed' / 'filled' / 'failed' / 'skipped'
    reason: str = ''
    order_result: Optional[OrderResult] = None
    live_trade_id: Optional[int] = None
    duplicate_trade_id: Optional[int] = None
    quantity: Optional[Decimal] = None

    def to_dict(self) -> dict:
        order = self.order_result
        return {
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
    side = 'buy' if position_mode == SIGNAL_LONG else 'sell'

    duplicate = _find_open_trade_for_today(
        deployment_symbol, position_mode, fire_at,
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
                f"open trade #{duplicate.id} already exists for today."
            ),
            context={
                'ticker': ticker,
                'duplicate_trade_id': duplicate.id,
                'signal': signal.to_dict(),
                'fire_at': fire_at.isoformat(),
            },
        )
        return OrderOutcome(
            status='skipped',
            reason='duplicate_open_trade_today',
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

    quantity = _compute_quantity(deployment, broker_adapter, reference_price)
    if quantity is None or quantity <= 0:
        log_event(
            deployment,
            deployment_symbol=deployment_symbol,
            event_type='order_failed',
            actor_type=actor_type,
            actor_id=actor_id,
            level='warning',
            message=(
                f"{ticker}: computed entry quantity is zero "
                f"(price={reference_price})."
            ),
            context={
                'ticker': ticker,
                'reference_price': str(reference_price),
                'bet_size_percentage': float(deployment.bet_size_percentage or 0),
                'initial_capital': str(deployment.initial_capital or 0),
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
            metadata={
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
            },
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
    """Split notional: strategy leg + VIXY proxy (matches backtest w_strategy / w_hedge)."""
    from backtest_engine.services.hybrid_vix_hedge import (
        live_hedge_weights_at,
        resolved_hedge_config_for_backtest,
    )

    deployment = deployment_symbol.deployment
    cfg = resolved_hedge_config_for_backtest(deployment.hedge_config or {})
    w_s, w_h, hw_meta = live_hedge_weights_at(fire_at, cfg, yahoo_only=True)
    if w_h < 1e-9 or w_s >= 1.0 - 1e-9:
        # Overlay unavailable for this day — full strategy size (no VIXY leg)
        w_s, w_h = 1.0, 0.0

    qty_f = float(quantity)
    strat_qty = int(qty_f * w_s) if w_h > 1e-9 else int(qty_f)
    if strat_qty < 1 and qty_f >= 1:
        strat_qty = 1
    if strat_qty < 1:
        log_event(
            deployment,
            deployment_symbol=deployment_symbol,
            event_type='order_failed',
            actor_type=actor_type,
            actor_id=actor_id,
            level='warning',
            message=f"{ticker}: hedged entry — zero strategy shares after split.",
            context={'ticker': ticker, 'w_strategy': w_s, 'w_hedge': w_h, 'hedge_meta': hw_meta},
        )
        return OrderOutcome(status='failed', reason='zero_strategy_shares_hedge')

    strat_qty_d = Decimal(strat_qty)
    hedge_leg: dict = {
        'mode': 'vixy_proxy',
        'w_strategy': w_s,
        'w_hedge': w_h,
        'overlay': hw_meta,
    }

    if w_h > 1e-9:
        total_notional = reference_price * quantity
        hedge_dollars = float(total_notional) * w_h
        vixy_px: Optional[Decimal] = None
        try:
            vixy_px = broker_adapter.get_current_price(HEDGE_PROXY_SYMBOL)
        except Exception:  # pragma: no cover
            vixy_px = None
        if vixy_px and vixy_px > 0:
            hedge_qty = int(Decimal(str(hedge_dollars)) / vixy_px)
        else:
            hedge_qty = 0
        if hedge_qty < 1:
            hedge_leg['skipped'] = 'no_hedge_price_or_tiny'
        else:
            hedge_leg['ticker'] = HEDGE_PROXY_SYMBOL
            hedge_leg['target_quantity'] = str(hedge_qty)

    log_event(
        deployment,
        deployment_symbol=deployment_symbol,
        event_type='order_placed',
        actor_type=actor_type,
        actor_id=actor_id,
        message=(
            f"{ticker}: hedged entry {side.upper()} {strat_qty} @ market + "
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
            context={'ticker': ticker},
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
        and hedge_leg.get('ticker') == HEDGE_PROXY_SYMBOL
        and (hedge_leg.get('target_quantity'))
    ):
        h_qty = int(hedge_leg['target_quantity'])
        try:
            hedge_result = broker_adapter.place_order(
                symbol=HEDGE_PROXY_SYMBOL,
                side='buy',
                quantity=Decimal(str(h_qty)),
                order_type='market',
            )
            hedge_leg['broker'] = {
                'order_id': hedge_result.order_id,
                'broker_order_id': hedge_result.broker_order_id,
                'status': hedge_result.status,
            }
        except Exception as exc:  # pragma: no cover
            logger.exception('Hedge VIXY order failed for %s', ticker)
            hedge_leg['hedge_error'] = str(exc)

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
    closing_side = 'sell' if position_mode == SIGNAL_LONG else 'buy'

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

    log_event(
        deployment,
        deployment_symbol=deployment_symbol,
        event_type='order_placed',
        actor_type=actor_type,
        actor_id=actor_id,
        message=(
            f"{ticker}: submitting {closing_side.upper()} {open_trade.quantity} "
            f"to close {position_mode} trade #{open_trade.id}"
        ),
        context={
            'ticker': ticker,
            'side': closing_side,
            'quantity': str(open_trade.quantity),
            'live_trade_id': open_trade.id,
            'signal': signal.to_dict(),
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

    tm = open_trade.metadata or {}
    hedge_block = tm.get('hedge_leg') or {}
    if (
        hedge_block.get('ticker') == HEDGE_PROXY_SYMBOL
        and hedge_block.get('target_quantity')
        and not hedge_block.get('skipped')
    ):
        try:
            hq = Decimal(str(hedge_block['target_quantity']))
            hx = broker_adapter.place_order(
                symbol=HEDGE_PROXY_SYMBOL,
                side='sell',
                quantity=hq,
                order_type='market',
            )
            hedge_block['exit_broker'] = {
                'order_id': hx.order_id,
                'broker_order_id': hx.broker_order_id,
                'status': hx.status,
            }
            tm['hedge_leg'] = hedge_block
            open_trade.metadata = tm
            open_trade.save(update_fields=['metadata', 'updated_at'])
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception('Hedge VIXY exit failed for trade %s', open_trade.id)
            hedge_block['exit_error'] = str(exc)
            tm['hedge_leg'] = hedge_block
            open_trade.metadata = tm
            open_trade.save(update_fields=['metadata', 'updated_at'])

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

    exit_price = order_result.price
    pnl = _compute_pnl(open_trade.entry_price, exit_price, open_trade.quantity, position_mode)
    pnl_pct = _safe_pct(open_trade.entry_price, exit_price, position_mode)

    with transaction.atomic():
        open_trade.exit_price = exit_price
        open_trade.exit_timestamp = fire_at
        open_trade.pnl = pnl
        open_trade.pnl_percentage = pnl_pct
        open_trade.is_winner = pnl is not None and pnl > 0
        open_trade.status = 'closed'
        open_trade.save(
            update_fields=[
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
            f"{ticker}: closed {position_mode} trade #{open_trade.id} "
            f"PnL={pnl}"
        ),
        context={
            'ticker': ticker,
            'live_trade_id': open_trade.id,
            'position_mode': position_mode,
            'entry_price': str(open_trade.entry_price),
            'exit_price': str(exit_price),
            'pnl': str(pnl) if pnl is not None else None,
            'pnl_percentage': str(pnl_pct) if pnl_pct is not None else None,
            'is_winner': open_trade.is_winner,
            'broker_order_id': order_result.broker_order_id,
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
    errors = 0
    for trade in open_trades:
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


def _find_open_trade_for_today(
    deployment_symbol: DeploymentSymbol,
    position_mode: str,
    fire_at: datetime,
) -> Optional[LiveTrade]:
    today = fire_at.date()
    return (
        deployment_symbol.live_trades.filter(
            status='open',
            position_mode=position_mode,
            entry_timestamp__date=today,
        )
        .order_by('-entry_timestamp')
        .first()
    )


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


def _compute_quantity(
    deployment: StrategyDeployment,
    broker_adapter: BaseBrokerAdapter,
    reference_price: Decimal,
) -> Optional[Decimal]:
    bet_pct = Decimal(str(deployment.bet_size_percentage or 0)) / Decimal('100')
    if bet_pct <= 0:
        return None

    capital: Optional[Decimal]
    try:
        capital = broker_adapter.get_account_balance()
    except Exception:  # pragma: no cover - defensive
        capital = None
    if capital is None or capital <= 0:
        capital = Decimal(str(deployment.initial_capital or 0))
    if capital <= 0:
        return None

    bet_amount = capital * bet_pct
    if bet_amount <= 0:
        return None

    raw_qty = bet_amount / reference_price
    try:
        return Decimal(int(raw_qty))
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
