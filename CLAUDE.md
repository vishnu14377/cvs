# CareConnect ADR AI Agent

## Testing

- Unit tests: `.venv/bin/python -m pytest tests/unit/api/ tests/unit/agents/ -v --tb=short`
- Validation tests: `.venv/bin/python -m pytest tests/unit/api/validation/ -v`
- Integration (needs Docker): `.venv/bin/python -m pytest tests/integration/ -v --timeout=180`
- Full suite has a pre-existing env var isolation bug: tests/unit/api/ tests fail with 401 when run alongside tests/unit/core/ (API_AUTH_TOKEN cleared by `patch.dict(os.environ, {}, clear=True)` in test_local_directory_handler.py). Run API tests in isolation or fix the conftest.

## Docker

- Stack: `docker-compose up -d --build`
- Rebuild single service: `docker-compose up -d --build api`
- Seed GCS emulator: `bash docker/seed-gcs.sh`
- Test harness UI: http://localhost:8000/dev/test
- Auth token (dev): `dev-token-12345`

## Deployment

- Branches: `non-prod` (deploys) and `release/2026.05.R1` (Docker publish) — keep in sync
- Docker publish requires `[publish]` in commit message OR push to `release/**` branch
- Deploy flow: commit code with `[publish]` → CD builds image → update `deploy-configs/careconnect-ai-dev1/values/v1.yaml` with new SHA → push triggers App Deployment
- Values file changes to `deploy-configs/` trigger `cd-dev.yaml` (App Deployment) only, NOT `cd.yaml` (Docker build)
- The "Fetch Setup Repo" step in App Deployment can take 20-40 min on slow runners; cancel+retry if stuck past 45 min

## Architecture

- FastAPI app factory: `src/api/app.py` → `create_app()`
- Agent graph: LangGraph StateGraph in `src/agents/graph.py` with generate → grounding_gate → tools_condition → tools loop
- Session registry: in-memory Dict[str, Tuple[SessionManager, float]] in `src/api/dependencies.py`
- Templates: `src/api/templates/` (Jinja2 for test harness and chat UI)
- Validation/guardrails: `src/api/validation/` (phi_redaction, input_safety, grounding_judge)
- Secrets bootstrap: `src/core/secrets.py` loads env vars from GCP Secret Manager at startup (before config reads them)
- Cloud SQL: `src/core/cloudsql_pg_client.py` — singleton engine, verifies pgvector extension (read-only, no CREATE needed)
- Vector store: `src/core/pgvector_store.py` + `src/adr_vector_database/` — LangChain PGVector in `carapp` schema

## Infrastructure

- GKE cluster: `careconnect-gke` in `us-east4`, project `hcb-dev-careconnect-etl`
- Cloud SQL: instance `cargpgsd1`, database `cargpgsd1_db`, schema `carapp`, user `cargpgsd1_nh_user`
- Vector tables (`langchain_pg_collection`, `langchain_pg_embedding`) live in `carapp` schema — NOT public
- GCS bucket: `care_connect_ai_initiatives` (ADR documents in `test_full_adrs/`)
- Endpoint: `internal-careconnect-ai-dev1.careconnect-gke.cvshealth.com`

## Environment

Secrets loaded from GCP Secret Manager (project `hcb-dev-careconnect-etl`):
- `CLOUDSQL_HOST`, `CLOUDSQL_PORT`, `CLOUDSQL_DATABASE`, `CLOUDSQL_DATABASE_SCHEMA`, `CLOUDSQL_USER`, `CLOUDSQL_PASSWORD`
- `MONGODB_URI`, `MONGODB_DATABASE`
- `API_AUTH_TOKEN`, `CARECONNECT_WIDGET_TOKEN`

For local dev, set these in `.env` (see `.env.example`).

## Code Patterns

- Python 3.10; use `from __future__ import annotations`
- Async routes with `pytest.mark.asyncio` and `AsyncClient` (httpx) in tests
- Tests use `from src.api.dependencies import get_session_registry` to inject mock sessions as `(mock_manager, time.time())` tuples
- Logger: `from src.core.logger import get_logger; logger = get_logger(__name__)`
- LangChain client: `from src.core.langchain_client import LangChainClient` → `.get_client()` returns bound LLM

## Conventions

- Route files in `src/api/routes/`, one per domain (query, sessions, feedback, etc.)
- Models in `src/api/models/`, matching the route name
- All external errors must be sanitized — never expose internal details to clients
- PHI redaction applied to all external surfaces (history, feedback, logs) but NOT to internal agent context
