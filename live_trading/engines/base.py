"""Base class and shared types for live trading engines.

The live trading engine is intentionally distinct from the backtest engine:
it is event-driven (one tick per scheduled fire) and operates on the *latest*
slice of OHLCV data plus broker state. Each strategy subclasses
`BaseLiveTradingEngine` and is registered against its strategy name via
`register_live_engine`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Optional


SIGNAL_LONG = 'long'
SIGNAL_SHORT = 'short'
SIGNAL_EXIT_LONG = 'exit_long'
SIGNAL_EXIT_SHORT = 'exit_short'
SIGNAL_HOLD = 'hold'

VALID_SIGNALS = {
    SIGNAL_LONG,
    SIGNAL_SHORT,
    SIGNAL_EXIT_LONG,
    SIGNAL_EXIT_SHORT,
    SIGNAL_HOLD,
}


@dataclass
class LiveSignal:
    """A single signal evaluation produced by a live engine.

    `action` is one of `long` / `short` / `exit_long` / `exit_short` / `hold`.
    `context` carries arbitrary per-engine debugging data (indicator values,
    thresholds, gap, etc.) so the audit log can render *why* a signal fired.
    """

    action: str
    confidence: float = 0.0
    price: Optional[Decimal] = None
    bar_timestamp: Optional[datetime] = None
    bar_date: Optional[date] = None
    context: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def __post_init__(self) -> None:
        if self.action not in VALID_SIGNALS:
            raise ValueError(
                f"Invalid signal action {self.action!r}; must be one of "
                f"{sorted(VALID_SIGNALS)}",
            )

    @property
    def is_entry(self) -> bool:
        return self.action in (SIGNAL_LONG, SIGNAL_SHORT)

    @property
    def is_exit(self) -> bool:
        return self.action in (SIGNAL_EXIT_LONG, SIGNAL_EXIT_SHORT)

    @property
    def is_actionable(self) -> bool:
        return self.is_entry or self.is_exit

    def to_dict(self) -> dict[str, Any]:
        return {
            'action': self.action,
            'confidence': self.confidence,
            'price': str(self.price) if self.price is not None else None,
            'bar_timestamp': (
                self.bar_timestamp.isoformat() if self.bar_timestamp else None
            ),
            'bar_date': self.bar_date.isoformat() if self.bar_date else None,
            'context': self.context,
            'error': self.error,
        }


@dataclass
class EngineEvaluation:
    """Outcome of a single engine fire.

    Distinct from `LiveSignal` so engines can return rich diagnostic info even
    when no actionable signal was emitted.
    """

    deployment_symbol_id: int
    ticker: str
    fire_at: datetime
    signal: Optional[LiveSignal] = None
    skipped_reason: Optional[str] = None
    ohlcv_count: int = 0
    indicators_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            'deployment_symbol_id': self.deployment_symbol_id,
            'ticker': self.ticker,
            'fire_at': self.fire_at.isoformat(),
            'signal': self.signal.to_dict() if self.signal else None,
            'skipped_reason': self.skipped_reason,
            'ohlcv_count': self.ohlcv_count,
            'indicators_count': self.indicators_count,
        }


class BaseLiveTradingEngine(ABC):
    """Abstract per-strategy live engine.

    Lifecycle:
    - The market-open Celery beat fires `evaluate(deployment_symbol, fire_at)`
      for each enrolled symbol; engines return an `EngineEvaluation`.
    - If `EngineEvaluation.signal.is_actionable`, the orchestrating task
      (Phase 5) calls the broker adapter to place the order.
    - The engine itself does not place orders; it only computes the signal.
    """

    #: Identifier returned by `engine_id`; defaults to the registered strategy
    #: name. Concrete subclasses can override if they expose multiple variants.
    name: str = ''

    #: Minimum number of historical OHLCV bars required to produce a signal.
    min_bars_required: int = 1

    #: When the engine expects to fire within the trading day. Used by the
    #: scheduler to slot the engine into the right beat (only `open` for now).
    expected_signal_window: str = 'open'

    def __init__(
        self,
        deployment,
        *,
        broker_adapter=None,
        clock=None,
    ) -> None:
        self.deployment = deployment
        self.strategy = deployment.strategy
        self.parameters = dict(deployment.strategy_parameters or {})
        self.broker_adapter = broker_adapter
        self._clock = clock

    # ------------------------------------------------------------------
    # Hooks subclasses MUST implement
    # ------------------------------------------------------------------

    @abstractmethod
    def evaluate(self, deployment_symbol, fire_at: datetime) -> EngineEvaluation:
        """Evaluate the engine for a single deployment symbol at `fire_at`."""

    # ------------------------------------------------------------------
    # Optional hooks
    # ------------------------------------------------------------------

    def on_market_open(self, deployment_symbol, fire_at: datetime) -> EngineEvaluation:
        """Default market-open hook delegates to `evaluate`."""
        return self.evaluate(deployment_symbol, fire_at)

    def on_market_close(self, deployment_symbol, fire_at: datetime) -> Optional[EngineEvaluation]:
        """Default close-hook is a no-op; override when the strategy needs it."""
        return None

    def on_weekend_recalc_done(self, deployment_symbol) -> None:
        """Hook fired after the weekend snapshot recalc settles."""
        return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def engine_id(self) -> str:
        return self.name or self.__class__.__name__

    def now(self) -> datetime:
        if self._clock is not None:
            return self._clock()
        from django.utils import timezone

        return timezone.now()
