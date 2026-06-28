#!/usr/bin/env bash
# wait-for-it.sh — wait for a host:port to become available
# Usage: wait-for-it.sh host:port [-t timeout] [-- command args]

TIMEOUT=60
QUIET=0
HOST=""
PORT=""
CLI=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -t)
            TIMEOUT="$2"
            shift 2
            ;;
        -q)
            QUIET=1
            shift
            ;;
        --)
            shift
            CLI=("$@")
            break
            ;;
        *)
            if [[ "$1" == *:* ]]; then
                HOST="${1%%:*}"
                PORT="${1##*:}"
            fi
            shift
            ;;
    esac
done

if [[ -z "$HOST" || -z "$PORT" ]]; then
    echo "Error: host:port required" >&2
    exit 1
fi

log() {
    if [[ "$QUIET" -eq 0 ]]; then
        echo "$@"
    fi
}

log "Waiting for $HOST:$PORT (timeout ${TIMEOUT}s)..."

for ((i=0; i<TIMEOUT; i++)); do
    if (echo >"/dev/tcp/$HOST/$PORT") >/dev/null 2>&1; then
        log "$HOST:$PORT is available."
        if [[ ${#CLI[@]} -gt 0 ]]; then
            exec "${CLI[@]}"
        fi
        exit 0
    fi
    sleep 1
done

echo "Timeout waiting for $HOST:$PORT" >&2
exit 1
