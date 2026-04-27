"""Copy hybrid VIX hedge flags from symbol backtest runs into `StrategyDeployment`."""

from __future__ import annotations

from typing import Any

from backtest_engine.models import SymbolBacktestParameterSet, SymbolBacktestRun
from strategies.models import StrategyDefinition


def inherit_hedge_from_symbol_runs(
    strategy: StrategyDefinition,
    parameter_set: SymbolBacktestParameterSet,
) -> tuple[bool, dict[str, Any]]:
    """Return (hedge_enabled, hedge_config) from the latest matching `SymbolBacktestRun`.

    Prefers a run with `hedge_enabled=True` for the same strategy + parameter set;
    otherwise uses the most recently updated run for that pair (may be False).
    """

    base = SymbolBacktestRun.objects.filter(
        strategy=strategy,
        parameter_set=parameter_set,
    ).order_by('-updated_at')

    preferred = base.filter(hedge_enabled=True).first()
    run = preferred or base.first()
    if run is None:
        return False, {}
    return bool(run.hedge_enabled), dict(run.hedge_config or {})
