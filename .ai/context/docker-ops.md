# Docker operations

## Services (`docker-compose.yml`)

| Service | Role | Port |
|---------|------|------|
| `db` | PostgreSQL 16 | internal |
| `redis` | Celery broker + Channels | internal |
| `web` | Daphne ASGI (Django) | 8000 |
| `celery_worker` | Background tasks | — |
| `celery_beat` | Scheduled tasks | — |

## Quick start

```bash
cp .env.example .env   # configure DB, API keys
docker compose up --build
```

`web` runs migrations when `RUN_MIGRATIONS=true`.

## Environment

- `DB_NAME`, `DB_USER`, `DB_PASSWORD` — Postgres (defaults in compose)
- Provider keys: `ALPACA_API_KEY`, `ALPACA_API_SECRET`, `ALPHA_VANTAGE_API_KEY`, etc.
- See `.env.example` for full list

## Local dev (without Docker)

```bash
conda activate trading
python manage.py runserver
celery -A algo_trading_backend worker --loglevel=info
```

Requires local PostgreSQL and Redis matching `settings.py` / `.env`.

## Troubleshooting

- `web` depends on healthy `db` and `redis` — check `docker compose logs db redis`
- Celery worker starts after `web` is up
- Volume `postgres_data` persists DB between restarts
