# Strategy signals (blackbox)

## Where logic lives

- **Public API:** `strategies.signals.check_strategy_signal(strategy_name, ctx)`
- **Pure rules:** `strategies/signals/rules/` (e.g. `gap.py` — inequality + classification)
- **Handlers:** `strategies/signals/handlers/` — compose rule + register via `@register_strategy_signal`
- **Adapters:** `to_backtest_order()` → `buy`/`sell`/`None`; `to_live_action()` → live signal strings

## Runtimes must NOT duplicate classification

- **Backtest:** prepare indicators per bar → build `StrategySignalContext` → `check_strategy_signal` → `to_backtest_order`
- **Live:** prepare returns/std (broker open, etc.) → same context → `check_strategy_signal` → `to_live_action`

## Adding a new strategy

1. Add pure rule + `classify_*_action` in `strategies/signals/rules/`
2. Add handler in `strategies/signals/handlers/` with `@register_strategy_signal('Strategy Name')`
3. Wire backtest `_generate_signal` and live engine as thin wrappers
4. Add parity tests in `strategies/tests/`

## Broker capabilities

Use `resolve_broker_side_capabilities(symbol, broker)` — no association means both sides **disabled**.

## Related paths

- `backtest_engine/services/backtest_executor.py` — backtest signal wiring
- `live_trading/engines/` — live engine implementations
