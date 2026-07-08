# Market Data Architecture

## Provider blackboxes

- Files under `market_data/providers/` (except `registry.py`, `factory.py`) are **API-specific blackboxes**.
- Each OHLCV provider exposes `get_historical_data(...)` (and optionally bulk helpers) with provider-native logic inside.
- Blackboxes must **not** import Celery tasks, views, or orchestration code.
- EOD and Polygon are internal catalog/import providers — not listed in the OHLCV UI catalog.

## Reusable service layer

Implement provider-agnostic logic in `market_data/services/` first:

| Function | Module | Purpose |
|----------|--------|---------|
| `get_daily_data` | `market_data_service.py` | Unified OHLCV fetch entry |
| `get_daily_data_bulk` | `market_data_service.py` | Multi-ticker fetch |
| `normalize_date_range` / `parse_task_dates` | `market_data_service.py` | Date defaults (`end_date` empty → today) |
| `check_data_quality` | `data_validation.py` | OHLCV quality checks |
| `ensure_symbol` / `resolve_symbol` | `symbol_resolution.py` | EOD symbol create/search |
| `ensure_exchange_symbols` | `symbol_resolution.py` | Exchange import when empty |

**Tasks and views orchestrate only** — they call these services; they do not duplicate provider API logic.

## OHLCV-first UX

- User-facing flow is **Fetch OHLCV** only; symbol import runs automatically when a symbol or exchange is missing.
- Hardcoded OHLCV providers: Yahoo Finance, Alpaca, Alpha Vantage (`providers/registry.py`).

## Frontend contract

- No calculations in the frontend for market data quality or OHLCV metrics.
- Use `resolve-symbol` before single-symbol fetch when ambiguity is possible.
- Frontend imports market data operations from `data/symbols.js`, not raw HTTP.

## Credentials

- OHLCV provider API keys live in **`.env`** (`ALPACA_API_KEY`, `ALPACA_API_SECRET`, `ALPHA_VANTAGE_API_KEY`, etc.).
- `market_data/providers/credentials.py` reads env first; Provider DB rows are optional fallback.
- Copy from `.env.example`; never commit real keys.

## Adding a new provider

1. Add blackbox class in `market_data/providers/`.
2. Register in `ProviderFactory` and `OHLCV_PROVIDERS` catalog.
3. Wire credentials via `Provider` model or env vars.
4. Do **not** call the blackbox from views/tasks directly — use `get_daily_data`.
