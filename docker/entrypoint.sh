#!/usr/bin/env bash
set -euo pipefail

DB_HOST="${DB_HOST:-db}"
DB_PORT="${DB_PORT:-5432}"
REDIS_HOST="${REDIS_HOST:-redis}"
REDIS_PORT="${REDIS_PORT:-6379}"

/app/docker/wait-for-it.sh "${DB_HOST}:${DB_PORT}" -t 60
/app/docker/wait-for-it.sh "${REDIS_HOST}:${REDIS_PORT}" -t 60

if [[ "${RUN_MIGRATIONS:-false}" == "true" ]]; then
    echo "Running migrations..."
    python manage.py migrate --noinput

    echo "Bootstrapping market schedules..."
    python manage.py bootstrap_market_schedules
fi

exec "$@"
