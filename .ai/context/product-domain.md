# Product domain (PRD summary)

Condensed from the platform PRD. Full original: monorepo `Context.md.bak`.

## Modules

| Module | Description |
|--------|-------------|
| Market Data | OHLCV ingestion, symbols, exchanges, providers |
| Analytical Tools | Indicators, per-symbol assignments |
| Strategies | Definitions, parameters, per-symbol activation |
| Backtest Engine | Historical execution, trades, statistics, equity curves |
| Live Trading | Brokers, deployments, paper → real money workflow |

## Backtest

- Entry: global (`/strategy/<name>/backtest/`) or per-symbol
- Configurable: strategy, symbols, broker (optional), date range, `split_ratio`, parameters
- **Broker-aware:** position mode ALL/LONG/SHORT filters via `SymbolBrokerAssociation.long_active` / `short_active`
- Loop: add bar → indicators → strategy signal → trades → equity curve
- Runs in Celery; progress via websockets

## Live trading workflow

1. Select completed backtest + position mode (ALL/LONG/SHORT)
2. Select broker
3. Select symbols (filtered by broker association flags)
4. **Paper trading mandatory** — uses paper API key
5. Evaluation criteria: min trades, Sharpe > 1.0, PnL > 0
6. After min trades: auto-evaluate; real money blocked if any criterion fails
7. Real money uses real API key only when evaluation passes

## Strategy engines

Each strategy may have separate backtest and live engine implementations. Signal
classification must be shared via `strategies.signals` (see `strategy-signals.md`).

## Key models

- `StrategyDefinition`, `StrategyAssignment`
- `Backtest`, `Trade`, `BacktestStatistics`
- `Broker`, `SymbolBrokerAssociation`
- `LiveTradingDeployment`, `LiveTrade`

See `database-schema.md` for column-level detail.
