#!/usr/bin/env bash
# Refresh the FTL fixture from a running local legacy service.
# Usage:
#   LEGACY_URL=http://localhost:8080 ./scripts/refresh-ftl-fixture.sh
set -euo pipefail
LEGACY_URL="${LEGACY_URL:-http://localhost:8080}"
OUT="src/api/static/fixtures/clinical-viewer-display.html"

payload='{
  "memberId": "TEST-MEMBER-001",
  "correlationId": "TEST-CORR-001",
  "dateOfBirth": "1950-01-01",
  "firstName": "TEST",
  "lastName": "PATIENT"
}'

resp=$(curl -sS -X POST "${LEGACY_URL}/memberADR/renderHtml" \
  -H "Content-Type: application/json" \
  -H "x-correlation-id: TEST-CORR-001" \
  -d "$payload")

# Legacy returns { "renderedHtml": "..." } — extract with python since jq may not be available.
echo "$resp" | python3 -c 'import json,sys; print(json.load(sys.stdin)["renderedHtml"])' > "$OUT"
echo "Wrote $OUT ($(wc -c < "$OUT") bytes)"
