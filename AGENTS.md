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
| `python manage.py runserver` | Dev server (port 8000 local / Docker host **8001**) |
| `python manage.py migrate` | Apply migrations |
| `python manage.py makemigrations` | Create migrations |
| `celery -A algo_trading_backend worker --loglevel=info` | Celery worker |
| `celery -A algo_trading_backend beat --loglevel=info` | Celery beat |
| `docker compose up` | Full stack (web, db, redis, celery) |
| `python manage.py test backtest_engine.tests.test_statistics_helpers backtest_engine.tests.test_gap_golden_results` | **QA agent:** Gap golden (trades, stats, equity) — required when QA covers backtest engine |
| `python manage.py dump_gap_golden` | **Backend agent only:** regenerate goldens after intentional engine change (review JSON diff) |

API base (Docker): `http://localhost:8001/api/`. Copy `.env` from `.env.example` before running.

## Architecture

- **Django apps:** `market_data`, `analytical_tools`, `strategies`, `backtest_engine`, `live_trading`
- **Provider blackboxes** in `market_data/providers/` — orchestration only in services/tasks/views
- **Strategy signals** shared between backtest and live via `strategies.signals.check_strategy_signal`
- **Broker adapters** via `get_broker_adapter` + `BaseBrokerAdapter` — HTTP/SDK only inside `live_trading/adapters/`
- **Separate engines:** backtest executor vs live trading engines per strategy
- **Broker-aware** backtests and deployments filter symbols by `SymbolBrokerAssociation` flags
- **Paper trading mandatory** before real money; evaluation criteria enforced in `live_trading`

## Blackbox exterior

Strategies, brokers, and OHLCV providers keep complexity **inside** a blackbox; the
**exterior API is the same** for every caller. Engines, services, tasks, and views
orchestrate — they do not reimplement provider/broker/strategy internals.

| Domain | Exterior (uniform) | Interior (blackbox) | Docs |
|--------|--------------------|---------------------|------|
| Strategies | `check_strategy_signal(name, ctx)` | `strategies/signals/rules/` + `handlers/` | `.ai/context/strategy-signals.md` |
| Brokers | `get_broker_adapter` → `place_order` / cancel / positions / balance | `live_trading/adapters/{broker}.py` | `.ai/context/broker-adapters.md` |
| OHLCV providers | `get_daily_data` (service layer) | `market_data/providers/` | `.ai/context/market-data.md` |

**Strategies:** backtest and live must call the same signal API. Allowed exceptions:
context preparation (indicators vs broker session open) and output mapping
(`to_backtest_order` / `to_live_action`). Never duplicate signal classification.
New strategies: pure rule + `@register_strategy_signal` + parity tests; engines are
thin wrappers only. Legacy inline signals in `backtest_executor` (SMA, MACD, RSI,
Bollinger, etc.): when touched or given a live engine, migrate to the blackbox —
Gap-Up and Gap-Down is the reference.

**Brokers:** obtain adapters only via `get_broker_adapter(broker, paper_trading)`;
call `BaseBrokerAdapter` methods. No broker HTTP/SDK outside `live_trading/adapters/`.
Backtest may simulate fills without a real broker (sensible exception).

**Providers:** unchanged — tasks/views call `market_data/services/`, never provider
classes directly.

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
| `.ai/context/broker-adapters.md` | Broker adapter blackbox, factory, order surface |
| `.ai/context/database-schema.md` | Models and relationships |
| `.ai/context/docker-ops.md` | Docker Compose, env vars, services |

## Conventions

- Respect **Blackbox exterior** above — same exterior API; complexity stays inside
- Tasks and views **orchestrate** — call `market_data/services/`; never duplicate provider API logic
- New OHLCV provider: blackbox class → `ProviderFactory` + `OHLCV_PROVIDERS` → use `get_daily_data`
- New strategy: pure rule in `strategies/signals/rules/` + handler with `@register_strategy_signal`
- Backtest and live engines must call the same signal API — add parity tests in `strategies/tests/`
- New broker: subclass `BaseBrokerAdapter` → `register_broker_adapter` → callers use factory only
- Order placement goes through `order_service` + `BaseBrokerAdapter.place_order` — not broker-specific APIs
- **Backtest result correctness (QA agent):** when AGENT_TEAM QA covers
  `backtest_engine` / Gap / stats / equity / trades, the QA agent **must** run
  `test_statistics_helpers` + `test_gap_golden_results` and treat failures as
  defects (log in workspace `AGENT_TEAM_REQUESTS.md`). Browser smoke does **not**
  replace this. Regenerating goldens (`dump_gap_golden`) is a **backend** task
  after intentional engine changes — not a way for QA to greenwash a break.
- API keys in `.env` (`ALPACA_API_KEY`, etc.); never commit real credentials
- Migrations: always review before applying in production

## Don't

- Import Celery tasks or views inside provider or broker blackboxes
- Duplicate signal classification in backtest vs live runtimes
- Call provider blackboxes directly from views/tasks — use service layer
- Call broker HTTP/SDK (or import `adapters.alpaca` etc.) from views/tasks/engines — use `get_broker_adapter`
- Bypass paper-trading evaluation before real-money deployment
- Put business calculations in the frontend
- Add a new strategy with classification logic only inside `backtest_executor` or a live engine
