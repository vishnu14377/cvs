#!/usr/bin/env bash
# Poll the API /health endpoint until it returns 200 or we time out.
# Usage: scripts/ci/wait_for_api.sh [base_url] [timeout_seconds]

set -euo pipefail

BASE_URL="${1:-http://localhost:8000}"
TIMEOUT="${2:-180}"
INTERVAL=3
ELAPSED=0

echo "Waiting for ${BASE_URL}/health (timeout=${TIMEOUT}s)..."
while [ "${ELAPSED}" -lt "${TIMEOUT}" ]; do
    if curl --silent --fail --max-time 5 "${BASE_URL}/health" > /dev/null 2>&1; then
        echo "API is healthy after ${ELAPSED}s."
        exit 0
    fi
    sleep "${INTERVAL}"
    ELAPSED=$((ELAPSED + INTERVAL))
    echo "  still waiting... ${ELAPSED}s elapsed"
done

echo "API did not become healthy within ${TIMEOUT}s."
echo "---- docker compose ps ----"
docker compose ps || true
echo "---- docker compose logs api (last 100 lines) ----"
docker compose logs --tail=100 api || true
exit 1
