"""Default symbol selection helper for `StrategyDeployment`s.

Eligible symbols are restricted to the snapshot symbols of the deployment's
`SymbolBacktestParameterSet`. By default we keep only **green** symbols
(`bucket_color == 'green'`) for the deployed `position_mode` (or for either
side when `position_mode == 'all'`), and order them by trade-count tiers
(`gt50 > gt20 > gt10 > gt0`) then by Sharpe descending.

The helper returns a list of `dict`s describing each candidate so the API can
either preview or persist the selection. Persistence (creating
`DeploymentSymbol` rows) is handled by the viewset.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

from backtest_engine.models import (
    SymbolBacktestParameterSet,
    SymbolBacktestRun,
    SymbolBacktestStatistics,
)
from backtest_engine.position_modes import normalize_position_modes
from market_data.models import Symbol

from ..utils.colors import (
    GRAY,
    GREEN,
    TIER_RANK,
    bucket_color,
    trade_count_tier,
)


def _safe_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if result != result:
        return None
    return result


def _safe_int(value) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None


@dataclass
class SymbolCandidate:
    """Per-symbol snapshot stats with classification helpers."""

    symbol: Symbol
    run_id: int
    sharpe_long: Optional[float] = None
    sharpe_short: Optional[float] = None
    max_dd_long: Optional[float] = None
    max_dd_short: Optional[float] = None
    total_trades_long: Optional[int] = None
    total_trades_short: Optional[int] = None
    color_long: str = GRAY
    color_short: str = GRAY
    color_overall: str = GRAY
    tier: str = 'none'
    primary_sharpe: Optional[float] = None  # used for ordering tie-break
    primary_trades: Optional[int] = None

    def to_preview(self) -> dict:
        return {
            'symbol_id': self.symbol.pk,
            'ticker': self.symbol.ticker,
            'run_id': self.run_id,
            'sharpe_long': self.sharpe_long,
            'sharpe_short': self.sharpe_short,
            'max_dd_long': self.max_dd_long,
            'max_dd_short': self.max_dd_short,
            'total_trades_long': self.total_trades_long,
            'total_trades_short': self.total_trades_short,
            'color_long': self.color_long,
            'color_short': self.color_short,
            'color_overall': self.color_overall,
            'tier': self.tier,
        }


@dataclass
class SelectionResult:
    """Output of `build_symbol_candidates` / `select_default_symbols`."""

    parameter_set: SymbolBacktestParameterSet
    position_mode: str
    candidates: list[SymbolCandidate] = field(default_factory=list)

    @property
    def green_candidates(self) -> list[SymbolCandidate]:
        sides = _sides_for_mode(self.position_mode)
        return [c for c in self.candidates if _candidate_is_green(c, sides)]

    def to_preview_payload(self, default_only: bool = True) -> dict:
        chosen = self.green_candidates if default_only else self.candidates
        chosen = sorted(chosen, key=_sort_key_for_mode(self.position_mode))
        return {
            'parameter_set': self.parameter_set.signature,
            'position_mode': self.position_mode,
            'count': len(chosen),
            'symbols': [c.to_preview() for c in chosen],
        }


def _sides_for_mode(position_mode: str) -> tuple[str, ...]:
    if position_mode == 'long':
        return ('long',)
    if position_mode == 'short':
        return ('short',)
    return ('long', 'short')


def _candidate_is_green(candidate: SymbolCandidate, sides: Iterable[str]) -> bool:
    for side in sides:
        color = candidate.color_long if side == 'long' else candidate.color_short
        if color == GREEN:
            return True
    return False


def _primary_side_for_mode(position_mode: str) -> str:
    if position_mode == 'short':
        return 'short'
    return 'long'


def _sort_key_for_mode(position_mode: str):
    primary = _primary_side_for_mode(position_mode)

    def key(candidate: SymbolCandidate):
        tier_rank = TIER_RANK.get(candidate.tier, len(TIER_RANK))
        sharpe = (
            candidate.sharpe_long if primary == 'long' else candidate.sharpe_short
        )
        # Prefer presence of a sharpe value first, then highest sharpe.
        sharpe_sort = -sharpe if sharpe is not None else float('inf')
        ticker = candidate.symbol.ticker
        return (tier_rank, sharpe_sort, ticker)

    return key


def _classify_candidate(candidate: SymbolCandidate, position_mode: str) -> None:
    candidate.color_long = bucket_color(candidate.sharpe_long, candidate.max_dd_long)
    candidate.color_short = bucket_color(candidate.sharpe_short, candidate.max_dd_short)

    sides = _sides_for_mode(position_mode)
    overall_colors = {
        candidate.color_long if 'long' in sides else None,
        candidate.color_short if 'short' in sides else None,
    } - {None}
    if GREEN in overall_colors:
        candidate.color_overall = GREEN
    elif overall_colors:
        for color_pref in ('yellow', 'orange', 'red', 'black', 'gray'):
            if color_pref in overall_colors:
                candidate.color_overall = color_pref
                break
        else:
            candidate.color_overall = GRAY
    else:
        candidate.color_overall = GRAY

    primary = _primary_side_for_mode(position_mode)
    candidate.primary_sharpe = (
        candidate.sharpe_long if primary == 'long' else candidate.sharpe_short
    )
    candidate.primary_trades = (
        candidate.total_trades_long if primary == 'long' else candidate.total_trades_short
    )

    # Tier is computed from the deployed-mode trade count when available;
    # falls back to the other side / total to avoid losing eligible symbols
    # when a snapshot only stored stats for one direction.
    tier_basis = candidate.primary_trades
    if tier_basis is None:
        other = (
            candidate.total_trades_short if primary == 'long' else candidate.total_trades_long
        )
        tier_basis = other
    candidate.tier = trade_count_tier(tier_basis)


def build_symbol_candidates(
    parameter_set: SymbolBacktestParameterSet,
    position_mode: str = 'long',
) -> SelectionResult:
    """Return all candidate symbols for a parameter set with classification."""

    runs = (
        SymbolBacktestRun.objects.filter(parameter_set=parameter_set)
        .select_related('symbol')
        .order_by('symbol__ticker', '-created_at')
    )
    latest_by_ticker = {}
    for run in runs:
        ticker = run.symbol.ticker
        if ticker not in latest_by_ticker:
            latest_by_ticker[ticker] = run

    run_ids = [r.id for r in latest_by_ticker.values()]
    stats_rows = SymbolBacktestStatistics.objects.filter(
        run_id__in=run_ids,
        symbol__isnull=False,
    ).select_related('run', 'symbol')
    # Symbol's primary key is its ticker, so symbol_id is the ticker string.
    stats_by_run_symbol = {(s.run_id, s.symbol_id): s for s in stats_rows}

    candidates: list[SymbolCandidate] = []
    for ticker in sorted(latest_by_ticker.keys()):
        run = latest_by_ticker[ticker]
        stats = stats_by_run_symbol.get((run.id, run.symbol_id))
        if not stats:
            continue

        modes = normalize_position_modes(run.position_modes)
        primary = 'long' if 'long' in modes else 'short'
        secondary = 'short' if primary == 'long' else 'long'

        primary_sharpe = _safe_float(stats.sharpe_ratio)
        primary_dd = _safe_float(stats.max_drawdown)
        primary_trades = _safe_int(getattr(stats, 'total_trades', None))

        extra = stats.additional_stats if isinstance(stats.additional_stats, dict) else {}
        primary_block = extra.get(primary) if isinstance(extra.get(primary), dict) else {}
        if primary_trades is None and primary_block:
            primary_trades = _safe_int(primary_block.get('total_trades'))

        sec_block = extra.get(secondary) if isinstance(extra.get(secondary), dict) else {}
        secondary_sharpe = _safe_float(sec_block.get('sharpe_ratio'))
        secondary_dd = _safe_float(sec_block.get('max_drawdown'))
        secondary_trades = _safe_int(sec_block.get('total_trades'))

        if primary == 'long':
            candidate = SymbolCandidate(
                symbol=run.symbol,
                run_id=run.id,
                sharpe_long=primary_sharpe,
                sharpe_short=secondary_sharpe,
                max_dd_long=primary_dd,
                max_dd_short=secondary_dd,
                total_trades_long=primary_trades,
                total_trades_short=secondary_trades,
            )
        else:
            candidate = SymbolCandidate(
                symbol=run.symbol,
                run_id=run.id,
                sharpe_long=secondary_sharpe,
                sharpe_short=primary_sharpe,
                max_dd_long=secondary_dd,
                max_dd_short=primary_dd,
                total_trades_long=secondary_trades,
                total_trades_short=primary_trades,
            )
        _classify_candidate(candidate, position_mode)
        candidates.append(candidate)

    return SelectionResult(
        parameter_set=parameter_set,
        position_mode=position_mode,
        candidates=candidates,
    )


def select_default_symbols(
    parameter_set: SymbolBacktestParameterSet,
    position_mode: str = 'long',
    *,
    max_symbols: Optional[int] = None,
) -> list[SymbolCandidate]:
    """Return only the green-tier candidates ordered for deployment."""

    result = build_symbol_candidates(parameter_set, position_mode)
    chosen = sorted(result.green_candidates, key=_sort_key_for_mode(position_mode))
    if max_symbols is not None and max_symbols >= 0:
        chosen = chosen[:max_symbols]
    return chosen
