# Algo Trading Backend — Agent Context

> Single source of truth for AI tools when working in this repo (standalone).
> Detailed overflow docs are in `.ai/context/`.

## Overview

Django REST API for a modular algorithmic trading platform: market data ingestion,
analytical tools, strategy definitions, backtest engine, and live trading (brokers,
paper/real deployments). Celery runs long tasks; Django Channels streams progress via
websockets. All business logic and calculations live here — the React frontend only
displays API results.

## Stack

| Component | Technology |
|-----------|------------|
| Framework | Django 5.2+, Django REST Framework |
| Async tasks | Celery 5.5+, django-celery-beat |
| Realtime | Django Channels 4+, Daphne, Redis |
| Database | PostgreSQL 16 |
| Data | pandas, numpy, yfinance |
| Runtime | Python (conda env `trading`) |

## Commands

| Command | Purpose |
|---------|---------|
| `conda activate trading` | Activate Python environment |
| `python manage.py runserver` | Dev server (port 8000) |
| `python manage.py migrate` | Apply migrations |
| `python manage.py makemigrations` | Create migrations |
| `celery -A algo_trading_backend worker --loglevel=info` | Celery worker |
| `celery -A algo_trading_backend beat --loglevel=info` | Celery beat |
| `docker compose up` | Full stack (web, db, redis, celery) |

API base: `http://localhost:8000/api/`. Copy `.env` from `.env.example` before running.

## Architecture

- **Django apps:** `market_data`, `analytical_tools`, `strategies`, `backtest_engine`, `live_trading`
- **Provider blackboxes** in `market_data/providers/` — orchestration only in services/tasks/views
- **Strategy signals** shared between backtest and live via `strategies.signals.check_strategy_signal`
- **Separate engines:** backtest executor vs live trading engines per strategy
- **Broker-aware** backtests and deployments filter symbols by `SymbolBrokerAssociation` flags
- **Paper trading mandatory** before real money; evaluation criteria enforced in `live_trading`

## Directory Structure

```
algo_trading_backend/
├── algo_trading_backend/   # settings, urls, asgi, celery
├── market_data/            # symbols, OHLCV, providers, services
├── analytical_tools/       # indicators, assignments
├── strategies/             # definitions, signals blackbox
├── backtest_engine/        # executor, tasks, statistics
├── live_trading/           # brokers, deployments, adapters
├── docker-compose.yml
└── requirements.txt
```

## Context Documents

| File | Role |
|------|------|
| `.ai/context/product-domain.md` | Modules, backtest/live workflows, evaluation rules |
| `.ai/context/market-data.md` | Provider blackboxes, service layer, OHLCV UX |
| `.ai/context/strategy-signals.md` | Signal blackbox, backtest/live parity |
| `.ai/context/database-schema.md` | Models and relationships |
| `.ai/context/docker-ops.md` | Docker Compose, env vars, services |

## Conventions

- Tasks and views **orchestrate** — call `market_data/services/`; never duplicate provider API logic
- New OHLCV provider: blackbox class → `ProviderFactory` + `OHLCV_PROVIDERS` → use `get_daily_data`
- New strategy: pure rule in `strategies/signals/rules/` + handler with `@register_strategy_signal`
- Backtest and live engines must call the same signal API — add parity tests in `strategies/tests/`
- API keys in `.env` (`ALPACA_API_KEY`, etc.); never commit real credentials
- Migrations: always review before applying in production

## Don't

- Import Celery tasks or views inside provider blackboxes
- Duplicate signal classification in backtest vs live runtimes
- Call provider blackboxes directly from views/tasks — use service layer
- Bypass paper-trading evaluation before real-money deployment
- Put business calculations in the frontend
