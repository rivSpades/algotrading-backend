"""Live trading engine for the `Gap-Up and Gap-Down` strategy.

Reuses the same indicator pipeline as the backtest (`compute_strategy_indicators_for_ohlcv`)
and the pure rule helper in `_rules.gap`. The engine is intentionally
read-only with respect to broker state: it returns an `EngineEvaluation`
describing the signal; order placement happens in the orchestrating task
(Phase 5).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone as dt_tz
from decimal import Decimal
from typing import Optional

from django.utils import timezone

from backtest_engine.services.hybrid_vix_hedge import HEDGE_VOL_ETF_TICKERS

from market_data.models import OHLCV
from market_data.models import ExchangeSchedule

from ..models import LiveTrade
from .base import (
    SIGNAL_HOLD,
    BaseLiveTradingEngine,
    EngineEvaluation,
    LiveSignal,
)
from .registry import register_live_engine

logger = logging.getLogger(__name__)

STRATEGY_NAME = 'Gap-Up and Gap-Down'

#: Buffer added on top of `std_period` so the rolling indicators have settled.
DEFAULT_BAR_BUFFER = 5


@register_live_engine(STRATEGY_NAME)
class GapUpGapDownLiveEngine(BaseLiveTradingEngine):
    """Live engine for the gap-up / gap-down strategy.

    Each evaluation:
    1. Loads the latest `std_period + buffer` daily OHLCV rows for the symbol.
    2. Recomputes the strategy indicators via `compute_strategy_indicators_for_ohlcv`
       so the live engine and the backtest engine use the *exact same* values.
    3. Picks the latest indicator values and applies `gap_up_gap_down_decision`.
    4. Wraps the result in a `LiveSignal` honouring the deployment / symbol
       position mode and the broker's long/short capability flags.
    5. Returns an `EngineEvaluation` describing what happened.
    """

    name = STRATEGY_NAME
    expected_signal_window = 'open'
    min_bars_required = 5

    def evaluate(self, deployment_symbol, fire_at: datetime) -> EngineEvaluation:
        fire_at = fire_at or self.now()
        ticker = deployment_symbol.symbol.ticker

        # Backtest: hybrid sleeve uses VIXM/VIXY returns as one overlay (β-weighted VIXM−VIXY spread
        # in normal, or VIXY in panic) — it does *not* open separate gap strategy trades on these
        # tickers. Live must mirror that: with hedge on, do not treat them as regular symbols.
        if self.deployment.hedge_enabled and ticker in HEDGE_VOL_ETF_TICKERS:
            return EngineEvaluation(
                deployment_symbol_id=deployment_symbol.id,
                ticker=ticker,
                fire_at=fire_at,
                skipped_reason='hedge_vol_etf_excluded',
                ohlcv_count=0,
                signal=LiveSignal(
                    action=SIGNAL_HOLD,
                    context={
                        'reason': (
                            'VIXM/VIXY are hedge-sleeve inputs only when hybrid hedge is enabled. '
                            'Remove them from this deployment; hedge orders are placed automatically.'
                        ),
                    },
                ),
            )

        std_period = int(self.parameters.get('std_period', 90))
        threshold = float(self.parameters.get('threshold', 0.25))
        bars_to_load = max(std_period + DEFAULT_BAR_BUFFER, self.min_bars_required)

        # Live concept: compute today's gap using (today_open from broker data) vs (prev_close from DB).
        # We intentionally exclude "today" daily bars from the DB frame because those rows may be
        # missing or incomplete around the open.
        session_date = fire_at.date()
        hist_rows = list(
            OHLCV.objects.filter(
                symbol=deployment_symbol.symbol,
                timeframe='daily',
                timestamp__date__lt=session_date,
            ).order_by('-timestamp')[:bars_to_load]
        )
        hist_rows.reverse()  # oldest -> newest

        if len(hist_rows) < self.min_bars_required:
            return EngineEvaluation(
                deployment_symbol_id=deployment_symbol.id,
                ticker=ticker,
                fire_at=fire_at,
                ohlcv_count=len(hist_rows),
                skipped_reason='insufficient_history',
                signal=LiveSignal(
                    action=SIGNAL_HOLD,
                    bar_timestamp=hist_rows[-1].timestamp if hist_rows else None,
                    context={
                        'reason': 'insufficient_history',
                        'have': len(hist_rows),
                        'need': self.min_bars_required,
                    },
                    error='insufficient_history',
                ),
            )

        prev_bar = hist_rows[-1]
        prev_close = float(prev_bar.close)

        # Resolve the exchange session open time (UTC) for the symbol.
        open_utc = None
        try:
            schedules = list(
                ExchangeSchedule.objects.filter(
                    exchange=deployment_symbol.symbol.exchange,
                    active=True,
                )
            )
            # Choose a schedule that matches today's weekday when possible; otherwise fall back to earliest open.
            weekday = session_date.isoweekday()  # 1..7
            candidates = [s for s in schedules if weekday in (s.weekday_list() or [])]
            pick = (candidates or schedules)[0] if (candidates or schedules) else None
            if pick is not None:
                open_utc = pick.open_utc
        except Exception:  # pragma: no cover - defensive
            open_utc = None

        if open_utc is None:
            return EngineEvaluation(
                deployment_symbol_id=deployment_symbol.id,
                ticker=ticker,
                fire_at=fire_at,
                ohlcv_count=len(hist_rows),
                skipped_reason='missing_exchange_schedule',
                signal=LiveSignal(
                    action=SIGNAL_HOLD,
                    bar_timestamp=prev_bar.timestamp,
                    bar_date=session_date,
                    context={'reason': 'missing_exchange_schedule'},
                    error='missing_exchange_schedule',
                ),
            )

        session_open = timezone.make_aware(
            datetime.combine(session_date, open_utc),
            timezone=dt_tz.utc,
        )

        # Fetch today's session open from the broker adapter (minute bars).
        today_open = None
        if self.broker_adapter is not None and hasattr(self.broker_adapter, 'get_session_open_price'):
            today_open_dec = self.broker_adapter.get_session_open_price(
                ticker,
                session_open=session_open,
                window_minutes=6,
            )
            today_open = float(today_open_dec) if today_open_dec is not None else None

        if today_open is None or not (today_open > 0 and prev_close > 0):
            return EngineEvaluation(
                deployment_symbol_id=deployment_symbol.id,
                ticker=ticker,
                fire_at=fire_at,
                ohlcv_count=len(hist_rows),
                skipped_reason='missing_open_or_prev_close',
                signal=LiveSignal(
                    action=SIGNAL_HOLD,
                    bar_timestamp=session_open,
                    bar_date=session_date,
                    context={
                        'reason': 'missing_open_or_prev_close',
                        'today_open': today_open,
                        'prev_close': prev_close,
                        'prev_close_timestamp': prev_bar.timestamp.isoformat() if prev_bar.timestamp else None,
                        'session_open_utc': session_open.isoformat(),
                    },
                    error='missing_open_or_prev_close',
                ),
            )

        ohlcv_dicts = [
            {
                'timestamp': row.timestamp,
                'open': float(row.open),
                'high': float(row.high),
                'low': float(row.low),
                'close': float(row.close),
                'volume': int(row.volume or 0),
            }
            for row in hist_rows
        ]

        from market_data.services.indicator_service import compute_strategy_indicators_for_ohlcv

        try:
            indicators = compute_strategy_indicators_for_ohlcv(
                self.strategy,
                ohlcv_dicts,
                deployment_symbol.symbol,
                strategy_parameters=self.parameters,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception(
                "Gap-Up/Gap-Down engine: indicator computation failed for %s",
                ticker,
            )
            return EngineEvaluation(
                deployment_symbol_id=deployment_symbol.id,
                ticker=ticker,
                fire_at=fire_at,
                ohlcv_count=len(hist_rows),
                indicators_count=0,
                skipped_reason='indicator_failure',
                signal=LiveSignal(
                    action=SIGNAL_HOLD,
                    bar_timestamp=session_open,
                    bar_date=session_date,
                    context={'error_type': type(exc).__name__},
                    error=str(exc),
                ),
            )

        std_value = _last_value(
            indicators,
            [f'RollingSTD_{std_period}', 'RollingSTD', f'STD_{std_period}'],
        )
        if std_value is None or not (float(std_value) > 0):
            return EngineEvaluation(
                deployment_symbol_id=deployment_symbol.id,
                ticker=ticker,
                fire_at=fire_at,
                ohlcv_count=len(hist_rows),
                indicators_count=len(indicators),
                skipped_reason='invalid_std',
                signal=LiveSignal(
                    action=SIGNAL_HOLD,
                    bar_timestamp=session_open,
                    bar_date=session_date,
                    context={'reason': 'invalid_std', 'std': std_value},
                    error='invalid_std',
                ),
            )

        # Today's gap return uses broker session open vs DB previous close.
        returns_value = (today_open - prev_close) / prev_close

        from strategies.signals import (
            GAP_STRATEGY_NAME,
            StrategySignalContext,
            check_strategy_signal,
            position_state_from_live,
            resolve_broker_side_capabilities,
            to_live_action,
        )

        long_allowed, short_allowed = resolve_broker_side_capabilities(
            deployment_symbol.symbol,
            self.deployment.broker,
        )
        position = self._latest_open_position(deployment_symbol)
        pos_state = (
            position_state_from_live(position.position_mode)
            if position is not None
            else position_state_from_live(None)
        )

        signal_result = check_strategy_signal(
            GAP_STRATEGY_NAME,
            StrategySignalContext(
                returns=returns_value,
                std=std_value,
                threshold=threshold,
                position_mode=deployment_symbol.position_mode,
                position_state=pos_state,
                long_allowed=long_allowed,
                short_allowed=short_allowed,
                parameters=dict(self.parameters or {}),
            ),
        )
        action = to_live_action(signal_result)
        reason = signal_result.reason
        decision = signal_result.decision

        signal = LiveSignal(
            action=action,
            confidence=_decision_confidence(decision),
            price=Decimal(str(today_open)),
            bar_timestamp=session_open,
            bar_date=session_date,
            context={
                'returns': decision.returns,
                'std': decision.std,
                'threshold': decision.threshold,
                'long_threshold': decision.long_threshold,
                'short_threshold': decision.short_threshold,
                'long_signal': decision.long_signal,
                'short_signal': decision.short_signal,
                'raw_direction': decision.direction,
                'today_open': today_open,
                'prev_close': prev_close,
                'prev_close_timestamp': prev_bar.timestamp.isoformat() if prev_bar.timestamp else None,
                'session_open_utc': session_open.isoformat(),
                'long_allowed': long_allowed,
                'short_allowed': short_allowed,
                'position_mode': deployment_symbol.position_mode,
                'has_position': position is not None,
                'position_side': position.position_mode if position else None,
                'reason': reason,
                'rule_context': decision.context,
                **signal_result.context,
            },
        )

        return EngineEvaluation(
            deployment_symbol_id=deployment_symbol.id,
            ticker=ticker,
            fire_at=fire_at,
            ohlcv_count=len(hist_rows),
            indicators_count=len(indicators),
            signal=signal,
            skipped_reason=None if signal.is_actionable else reason,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _latest_open_position(self, deployment_symbol) -> Optional[LiveTrade]:
        return (
            LiveTrade.objects.filter(
                deployment=self.deployment,
                deployment_symbol=deployment_symbol,
                status='open',
            )
            .order_by('-entry_timestamp')
            .first()
        )

    def now(self) -> datetime:
        return timezone.now() if self._clock is None else self._clock()


def _last_value(indicators: dict, candidate_keys: list[str]) -> Optional[float]:
    """Return the last non-null value among the candidate indicator keys."""
    for key in candidate_keys:
        block = indicators.get(key)
        if not block:
            continue
        values = block.get('values') if isinstance(block, dict) else None
        if not values:
            continue
        for value in reversed(values):
            if value is None:
                continue
            try:
                f = float(value)
            except (TypeError, ValueError):
                continue
            if f != f:  # NaN guard
                continue
            return f
    return None


def _decision_confidence(decision) -> float:
    """Map the gap magnitude to a 0..1 confidence value for audit display."""
    if decision.direction is None or decision.std <= 0:
        return 0.0
    try:
        magnitude = abs(decision.returns / decision.std)
    except ZeroDivisionError:
        return 0.0
    if magnitude != magnitude:  # NaN guard
        return 0.0
    if magnitude >= 1.0:
        return 1.0
    return float(magnitude)
