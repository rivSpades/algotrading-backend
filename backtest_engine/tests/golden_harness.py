"""
Shared helpers for Gap golden QA: synthetic OHLCV, executor runs, serialize/compare.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from django.utils import timezone

from backtest_engine.services.backtest_executor import BacktestExecutor
from market_data.models import Exchange, OHLCV, Symbol
from strategies.config import STRATEGY_DEFINITIONS
from strategies.models import StrategyDefinition

FIXTURES_DIR = Path(__file__).resolve().parent / 'fixtures'
SYNTHETIC_OHLCV_PATH = FIXTURES_DIR / 'gap_synthetic_ohlcv.json'
GOLDEN_LONG_PATH = FIXTURES_DIR / 'gap_golden_long.json'
GOLDEN_SHORT_PATH = FIXTURES_DIR / 'gap_golden_short.json'

GOLDEN_TICKER = 'GAPQA'
GOLDEN_PARAMS = {
    'threshold': 0.25,
    'std_period': 90,
}
GOLDEN_SPLIT_RATIO = 0.7
GOLDEN_INITIAL_CAPITAL = 10000.0
GOLDEN_BET_SIZE = 100.0

# Float tolerances for golden comparison
MONEY_TOL = 0.02
RATIO_TOL = 1e-4
STAT_TOL = 0.01


def build_synthetic_ohlcv_bars(num_bars: int = 150) -> list[dict]:
    """
    Deterministic daily bars: quiet overnight noise in the train window only,
    then flat opens (ret=0) in the test window except engineered gap events.

    Returns = (open - prev_close) / prev_close. Train-window ±0.1% noise keeps
    RollingSTD_90 > 0. Test-window gaps at 110/115/120/125 force known long/short
    round-trips (split_ratio=0.7 → test starts at index 105).
    """
    start = date(2020, 1, 2)
    bars: list[dict] = []
    prev_close = 100.0
    split_idx = int(num_bars * GOLDEN_SPLIT_RATIO)  # 105 for 150 bars
    gap_events = {
        110: 0.05,   # gap up → long entry
        115: -0.05,  # gap down → long exit
        120: -0.05,  # gap down → short entry
        125: 0.05,   # gap up → short exit
    }

    day = start
    for i in range(num_bars):
        while day.weekday() >= 5:  # skip weekends
            day += timedelta(days=1)
        if i in gap_events:
            ret = gap_events[i]
        elif i < split_idx:
            ret = 0.001 if (i % 2 == 0) else -0.001
        else:
            ret = 0.0
        open_px = round(prev_close * (1.0 + ret), 6)
        close_px = open_px  # flat session after open
        high_px = max(open_px, close_px) * 1.001
        low_px = min(open_px, close_px) * 0.999
        ts = timezone.make_aware(datetime(day.year, day.month, day.day))
        bars.append(
            {
                'timestamp': ts.isoformat(),
                'open': open_px,
                'high': round(high_px, 6),
                'low': round(low_px, 6),
                'close': close_px,
                'volume': 1_000_000,
            }
        )
        prev_close = close_px
        day += timedelta(days=1)
    return bars


def write_synthetic_ohlcv_fixture(path: Path | None = None) -> Path:
    path = path or SYNTHETIC_OHLCV_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'ticker': GOLDEN_TICKER,
        'timeframe': 'daily',
        'description': (
            'Synthetic Gap QA bars: ±0.1% noise in the train window (keep RollingSTD>0), '
            'flat opens in the test window except engineered gaps at indices 110/115/120/125 '
            '(std_period=90, split=0.7).'
        ),
        'bars': build_synthetic_ohlcv_bars(),
    }
    path.write_text(json.dumps(payload, indent=2) + '\n')
    return path


def load_synthetic_ohlcv_fixture(path: Path | None = None) -> dict:
    path = path or SYNTHETIC_OHLCV_PATH
    if not path.exists():
        write_synthetic_ohlcv_fixture(path)
    return json.loads(path.read_text())


def ensure_gap_strategy() -> StrategyDefinition:
    cfg = next(s for s in STRATEGY_DEFINITIONS if s['name'] == 'Gap-Up and Gap-Down')
    strategy, _ = StrategyDefinition.objects.update_or_create(
        name=cfg['name'],
        defaults={
            'description_short': cfg.get('description_short', ''),
            'description_long': cfg.get('description_long', ''),
            'default_parameters': cfg.get('default_parameters', {}),
            'analytic_tools_used': cfg.get('analytic_tools_used', []),
            'required_tool_configs': cfg.get('required_tool_configs', []),
            'globally_enabled': cfg.get('globally_enabled', False),
        },
    )
    return strategy


def load_symbol_and_ohlcv_from_fixture(fixture: dict | None = None) -> Symbol:
    fixture = fixture or load_synthetic_ohlcv_fixture()
    exchange, _ = Exchange.objects.get_or_create(
        code='GAPQA_EX',
        defaults={'name': 'Gap QA Exchange', 'country': 'US', 'timezone': 'UTC'},
    )
    symbol, _ = Symbol.objects.update_or_create(
        ticker=fixture['ticker'],
        defaults={
            'exchange': exchange,
            'type': 'stock',
            'status': 'active',
            'name': 'Gap QA Synthetic',
            'validation_status': 'valid',
        },
    )
    OHLCV.objects.filter(symbol=symbol, timeframe='daily').delete()
    rows = []
    for bar in fixture['bars']:
        ts = bar['timestamp']
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts.replace('Z', '+00:00'))
        if timezone.is_naive(ts):
            ts = timezone.make_aware(ts)
        rows.append(
            OHLCV(
                symbol=symbol,
                timeframe='daily',
                timestamp=ts,
                open=bar['open'],
                high=bar['high'],
                low=bar['low'],
                close=bar['close'],
                volume=bar['volume'],
            )
        )
    OHLCV.objects.bulk_create(rows)
    return symbol


def _make_run(strategy: StrategyDefinition, symbol: Symbol):
    """SymbolBacktestRun-shaped object for BacktestExecutor (persisted optional)."""
    from backtest_engine.models import SymbolBacktestRun

    bars = load_synthetic_ohlcv_fixture()['bars']
    start = datetime.fromisoformat(bars[0]['timestamp'].replace('Z', '+00:00'))
    end = datetime.fromisoformat(bars[-1]['timestamp'].replace('Z', '+00:00'))
    if timezone.is_naive(start):
        start = timezone.make_aware(start)
    if timezone.is_naive(end):
        end = timezone.make_aware(end)

    return SymbolBacktestRun(
        name='Gap golden QA',
        strategy=strategy,
        symbol=symbol,
        broker=None,
        start_date=start,
        end_date=end,
        split_ratio=GOLDEN_SPLIT_RATIO,
        initial_capital=GOLDEN_INITIAL_CAPITAL,
        bet_size_percentage=GOLDEN_BET_SIZE,
        strategy_parameters=dict(GOLDEN_PARAMS),
        hedge_enabled=False,
        run_strategy_only_baseline=True,
        hedge_config={},
        position_modes=['long', 'short'],
        status='pending',
    )


def _ts_iso(value) -> str | None:
    if value is None:
        return None
    if hasattr(value, 'isoformat'):
        ts = value
        if timezone.is_naive(ts):
            ts = timezone.make_aware(ts)
        return ts.astimezone(timezone.UTC).isoformat().replace('+00:00', 'Z')
    s = str(value)
    if s.endswith('+00:00'):
        return s[:-6] + 'Z'
    return s


def serialize_trades(trades: list[dict]) -> list[dict]:
    out = []
    for t in trades:
        sym = t.get('symbol')
        ticker = getattr(sym, 'ticker', None) or (sym if isinstance(sym, str) else None)
        out.append(
            {
                'ticker': ticker,
                'trade_type': t.get('trade_type'),
                'entry_timestamp': _ts_iso(t.get('entry_timestamp')),
                'exit_timestamp': _ts_iso(t.get('exit_timestamp')),
                'entry_price': round(float(t['entry_price']), 6) if t.get('entry_price') is not None else None,
                'exit_price': round(float(t['exit_price']), 6) if t.get('exit_price') is not None else None,
                'quantity': round(float(t['quantity']), 6) if t.get('quantity') is not None else None,
                'pnl': round(float(t['pnl']), 6) if t.get('pnl') is not None else None,
                'pnl_percentage': round(float(t['pnl_percentage']), 6)
                if t.get('pnl_percentage') is not None
                else None,
                'is_winner': bool(t.get('is_winner')),
            }
        )
    out.sort(key=lambda x: (x['entry_timestamp'] or '', x['trade_type'] or ''))
    return out


def serialize_equity_curve(curve: list) -> list[dict]:
    points = []
    for point in curve or []:
        if isinstance(point, dict):
            ts = point.get('timestamp')
            eq = point.get('equity')
        else:
            ts, eq = point
        points.append({'timestamp': _ts_iso(ts), 'equity': round(float(eq), 4)})
    return points


STAT_KEYS = (
    'total_trades',
    'winning_trades',
    'losing_trades',
    'win_rate',
    'total_pnl',
    'total_pnl_percentage',
    'average_pnl',
    'average_winner',
    'average_loser',
    'profit_factor',
    'max_drawdown',
    'max_drawdown_duration',
    'sharpe_ratio',
    'cagr',
    'total_return',
)


def serialize_stats(stats: dict, symbol: Symbol) -> dict:
    portfolio = stats.get(None) or {}
    symbol_stats = stats.get(symbol) or {}
    # Prefer symbol-level for single-symbol runs; fall back to portfolio key
    primary = symbol_stats if symbol_stats.get('total_trades') is not None else portfolio

    scalars = {}
    for key in STAT_KEYS:
        val = primary.get(key)
        if isinstance(val, float):
            scalars[key] = round(val, 6)
        else:
            scalars[key] = val

    equity = serialize_equity_curve(primary.get('equity_curve') or [])
    return {'scalars': scalars, 'equity_curve': equity}


def run_gap_executor(position_mode: str, persist_run: bool = False) -> dict[str, Any]:
    """Load fixture into DB, run BacktestExecutor in-process, return serializable result."""
    fixture = load_synthetic_ohlcv_fixture()
    strategy = ensure_gap_strategy()
    symbol = load_symbol_and_ohlcv_from_fixture(fixture)
    run = _make_run(strategy, symbol)
    if persist_run:
        run.save()

    executor = BacktestExecutor(run, position_mode=position_mode)
    executor.execute_strategy()
    stats = executor.calculate_statistics()

    return {
        'meta': {
            'ticker': symbol.ticker,
            'position_mode': position_mode,
            'strategy': strategy.name,
            'parameters': dict(GOLDEN_PARAMS),
            'split_ratio': GOLDEN_SPLIT_RATIO,
            'initial_capital': GOLDEN_INITIAL_CAPITAL,
            'bet_size_percentage': GOLDEN_BET_SIZE,
            'bar_count': len(fixture['bars']),
        },
        'trades': serialize_trades(executor.trades),
        'stats': serialize_stats(stats, symbol),
    }


def dump_golden_files() -> tuple[Path, Path, Path]:
    """Regenerate synthetic OHLCV + long/short golden JSON files."""
    ohlcv_path = write_synthetic_ohlcv_fixture()
    long_result = run_gap_executor('long')
    short_result = run_gap_executor('short')
    GOLDEN_LONG_PATH.write_text(json.dumps(long_result, indent=2) + '\n')
    GOLDEN_SHORT_PATH.write_text(json.dumps(short_result, indent=2) + '\n')
    return ohlcv_path, GOLDEN_LONG_PATH, GOLDEN_SHORT_PATH


def _almost_equal(a, b, tol: float) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return abs(float(a) - float(b)) <= tol


def assert_golden_match(actual: dict, expected: dict, *, money_tol=MONEY_TOL, ratio_tol=RATIO_TOL, stat_tol=STAT_TOL):
    """Raise AssertionError with detail if actual diverges from golden."""
    a_trades = actual['trades']
    e_trades = expected['trades']
    if len(a_trades) != len(e_trades):
        raise AssertionError(f'trade count {len(a_trades)} != golden {len(e_trades)}')

    for i, (a, e) in enumerate(zip(a_trades, e_trades)):
        for key in ('ticker', 'trade_type', 'is_winner'):
            if a.get(key) != e.get(key):
                raise AssertionError(f'trade[{i}].{key}: {a.get(key)!r} != {e.get(key)!r}')
        for key in ('entry_timestamp', 'exit_timestamp'):
            if (a.get(key) or '') != (e.get(key) or ''):
                raise AssertionError(f'trade[{i}].{key}: {a.get(key)!r} != {e.get(key)!r}')
        for key in ('entry_price', 'exit_price', 'quantity', 'pnl', 'pnl_percentage'):
            if not _almost_equal(a.get(key), e.get(key), money_tol):
                raise AssertionError(
                    f'trade[{i}].{key}: {a.get(key)!r} != {e.get(key)!r} (tol={money_tol})'
                )

    a_sc = actual['stats']['scalars']
    e_sc = expected['stats']['scalars']
    for key in STAT_KEYS:
        av, ev = a_sc.get(key), e_sc.get(key)
        if key in ('total_trades', 'winning_trades', 'losing_trades', 'max_drawdown_duration'):
            if av != ev:
                raise AssertionError(f'stats.{key}: {av!r} != {ev!r}')
        else:
            tol = money_tol if key in ('total_pnl', 'average_pnl', 'average_winner', 'average_loser') else (
                ratio_tol if key in ('win_rate', 'total_pnl_percentage', 'total_return', 'cagr', 'sharpe_ratio', 'profit_factor', 'max_drawdown') else stat_tol
            )
            if not _almost_equal(av, ev, tol):
                raise AssertionError(f'stats.{key}: {av!r} != {ev!r} (tol={tol})')

    a_eq = actual['stats']['equity_curve']
    e_eq = expected['stats']['equity_curve']
    if len(a_eq) != len(e_eq):
        raise AssertionError(f'equity_curve length {len(a_eq)} != golden {len(e_eq)}')
    for i, (a, e) in enumerate(zip(a_eq, e_eq)):
        if (a.get('timestamp') or '') != (e.get('timestamp') or ''):
            raise AssertionError(f'equity[{i}].timestamp: {a.get("timestamp")!r} != {e.get("timestamp")!r}')
        if not _almost_equal(a.get('equity'), e.get('equity'), money_tol):
            raise AssertionError(
                f'equity[{i}].equity: {a.get("equity")!r} != {e.get("equity")!r} (tol={money_tol})'
            )
