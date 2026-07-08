# Database schema

Implementation-accurate summary. For full column tables see monorepo `AI_PROJECT_HANDOFF.md` (if available).

## Django apps → tables

| App | Models |
|-----|--------|
| `market_data` | `Exchange`, `Provider`, `Symbol`, `OHLCV` |
| `analytical_tools` | `ToolDefinition`, `ToolAssignment`, `IndicatorValue` |
| `strategies` | `StrategyDefinition`, `StrategyAssignment` |
| `backtest_engine` | `Backtest`, `Trade`, `BacktestStatistics` |
| `live_trading` | `Broker`, `SymbolBrokerAssociation`, `LiveTradingDeployment`, `LiveTrade` |

## Key relationships

- `Symbol` → `Exchange`, `Provider`; M2M to brokers via `SymbolBrokerAssociation`
- `Backtest` → strategy, optional `broker`, symbols, produces `Trade` + `BacktestStatistics`
- `LiveTradingDeployment` → references `Backtest`, `Broker`, M2M `symbols`; tracks evaluation JSON
- `LiveTrade` → `LiveTradingDeployment`, `Symbol`, broker order metadata

## Broker association flags

`SymbolBrokerAssociation.long_active` / `short_active` drive:
- Backtest symbol filtering by position mode (ALL/LONG/SHORT)
- Live deployment symbol selection

## Migrations

- Always run `python manage.py makemigrations` after model changes
- Review generated SQL before applying in shared environments
