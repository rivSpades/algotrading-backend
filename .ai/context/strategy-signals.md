# Strategy signals (blackbox)

## Principle

Each strategy is a **blackbox**: classification and rule complexity live inside
`strategies/signals/`. The exterior API is the **same for backtest and live**.

Engines only prepare a `StrategySignalContext`, call `check_strategy_signal`, and
map the canonical result to runtime-specific order/action strings.

Reference implementation (full parity): **Gap-Up and Gap-Down**.

## Where logic lives

- **Public API:** `strategies.signals.check_strategy_signal(strategy_name, ctx)`
- **Pure rules:** `strategies/signals/rules/` (e.g. `gap.py` — inequality + classification)
- **Handlers:** `strategies/signals/handlers/` — compose rule + register via `@register_strategy_signal`
- **Output adapters (allowed exceptions):** `to_backtest_order()` → `buy`/`sell`/`None`;
  `to_live_action()` → live signal strings (`long` / `short` / `exit_*` / `hold`)

## Same contract, backtest and live

| Runtime | Prepare context | Classify | Map output |
|---------|-----------------|----------|------------|
| **Backtest** | indicators per bar | `check_strategy_signal` | `to_backtest_order` |
| **Live** | returns/std (broker session open, etc.) | `check_strategy_signal` | `to_live_action` |

Runtimes must **NOT** duplicate classification. Allowed differences: how inputs are
built (OHLCV indicators vs live broker open) and how the canonical `StrategySignalResult`
is mapped to buy/sell vs live action strings.

## Reference vs legacy

| Strategy | Blackbox? | Notes |
|----------|-----------|-------|
| Gap-Up and Gap-Down | Yes | Backtest + live both call `check_strategy_signal`; parity tests in `strategies/tests/` |
| SMA / MA crossover, RSI, Bollinger, MACD | No (legacy) | Still inline in `backtest_executor._generate_signal`; no live engine. When touched or given live trading, migrate to this blackbox |

## Adding a new strategy

1. Add pure rule + `classify_*_action` in `strategies/signals/rules/`
2. Add handler in `strategies/signals/handlers/` with `@register_strategy_signal('Strategy Name')`
3. Wire backtest `_generate_signal` and live engine as **thin wrappers** only (build ctx → check → map)
4. Add parity tests in `strategies/tests/`

Do **not** put new classification logic only inside `backtest_executor` or a live engine.

## Backtest result goldens (trades / stats / equity)

Signal parity tests do **not** lock fills or statistics. For Gap-Up and Gap-Down, numeric
regression lives in:

- `backtest_engine/tests/test_statistics_helpers.py` — equity/drawdown/PnL math
- `backtest_engine/tests/test_gap_golden_results.py` — end-to-end vs `fixtures/gap_golden_*.json`

**Who runs them:** the AGENT_TEAM **QA agent**, whenever QA scope includes the
backtest engine (see workspace `AGENT_TEAM.md`). Failures are defects, not optional.

**Who regenerates fixtures:** the **backend** agent only, via
`python manage.py dump_gap_golden`, after intentional engine changes — review the
trade/stats/equity JSON diff before commit. QA must not regenerate goldens just
to make tests pass.

## Broker capabilities

Use `resolve_broker_side_capabilities(symbol, broker)` — no association means both sides **disabled**.

## Related paths

- `strategies/signals/` — blackbox entry, registry, rules, handlers, adapters
- `backtest_engine/services/backtest_executor.py` — backtest signal wiring (Gap delegates; legacy still inline)
- `live_trading/engines/` — live engine implementations (Gap uses blackbox)
