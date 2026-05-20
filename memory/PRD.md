# Care Connect ADR AI Agent — Repository Reorganization

## Original Problem Statement
> Analyse the repo and organize files properly as per Confluence page and architecture and file names.

The "Confluence" reference was the architecture PDF in the repo
(`CC_ADR_AI_AGENT_ARCHITECTURE.pdf`) together with the handoff document
(`SEA-ADR AI Agent — Handoff Document-190526-174703.pdf`). User confirmed:
layered architecture, current `/app` workspace, preserve current functionality.

## What the Architecture Mandates
- Python FastAPI service (`src/api/app.py` → `create_app()`).
- Layered packages, one per bounded context (per architecture diagram):
  - `api` (REST controllers / routes / middleware / templates)
  - `agents` (LangGraph orchestration)
  - `tools` (ADR/Policy search/summary tools)
  - `core` (clients: GCS, CloudSQL, pgvector, embeddings, logger, secrets)
  - `ocr` (document OCR pipeline)
  - `adr_vector_database`, `policy_vector_database` (RAG stores)
  - `session_manager` (conversational state)
  - `feedback_manager` (MongoDB feedback store)
  - `eval`, `utils`
- `pyproject.toml` declares `where = ["src"]`, `testpaths = ["tests"]`,
  `pythonpath = ["src"]`, and the Dockerfile does `COPY src/ ./src/`.
- All source code uses `from src.<pkg>...` import style.

## What Was Found (Before)
Source packages and tests lived at the **repo root** — the layout
contradicted `pyproject.toml`, the `Dockerfile`, README, and the imports
already written into the code (`from src.api.dependencies`, `from src.core.logger`, …).

A few files still used bare imports (`from core.x`, `from ocr.x`, …)
which would break the moment the code is run with the declared
`src`-layout install.

## Changes Made (Jan 2026)
1. **Source moved under `src/`** via `git mv` (history preserved):
   `api/`, `core/`, `agents/`, `tools/`, `ocr/`, `session_manager/`,
   `feedback_manager/`, `utils/`, `adr_vector_database/`,
   `policy_vector_database/`, `eval/`, `adr_document_processor.py`.
2. **Tests moved under `tests/`** via `git mv`:
   `unit/`, `integration/`, `e2e/`, root `conftest.py`.
3. **Stale `/app/__init__.py`** (mislabeled "Tests for the ADR AI Agent
   system" at repo root) deleted — it would have shadowed the `src`
   package.
4. **Added** `src/__init__.py` and `tests/__init__.py` markers.
5. **Bare imports fixed**: every `from <pkg>.x` and `from <pkg> import` in
   the moved source/tests was rewritten to `from src.<pkg>.x`
   (24 files touched in `src/api`, `src/core`, `src/ocr`,
   `src/adr_vector_database`, and matching `tests/unit/` files).
6. **Pre-existing bug fixed**: `tests/unit/eval/test_scoring.py` used
   `from tests.eval.scoring` (a path that never existed). Corrected to
   `from src.eval.scoring` so the test actually runs against the real
   scoring module.

## Final Layout
```
/app
├── src/                       # source root (pyproject.toml where=["src"])
│   ├── __init__.py
│   ├── adr_document_processor.py
│   ├── adr_vector_database/
│   ├── agents/
│   ├── api/                   # routes, models, middleware, templates, static, validation, rendering, dev
│   ├── core/                  # config, logger, secrets, GCS, CloudSQL, pgvector, embedding, langchain, vertex, stubs
│   ├── eval/
│   ├── feedback_manager/
│   ├── ocr/                   # orchestrator (sync+async), mistral/llm clients, pdf_handler, sub_file_handler, prompts, data_models
│   ├── policy_vector_database/
│   ├── session_manager/       # core/, conversation_handler, deletion, initialization, warmup
│   ├── tools/                 # adr_search, adr_summary, policy_search, policy_summary, policy_list
│   └── utils/
├── tests/                     # testpaths=["tests"]
│   ├── __init__.py
│   ├── conftest.py
│   ├── unit/                  # one subpackage per src/* package
│   ├── integration/
│   └── e2e/
├── data/                      # golden/sample data (untouched)
├── deploy-configs/            # GKE/Helm values (untouched)
├── scripts/                   # CI helpers (untouched)
├── unqork/                    # Java/JS integration (untouched)
├── docker-compose*.yml, Dockerfile
├── pyproject.toml, requirements*.txt, requirements-dev.txt
├── README.md, CLAUDE.md, .commitlintrc.json
├── CC_ADR_AI_AGENT_ARCHITECTURE.pdf
├── SEA-ADR AI Agent — Handoff Document-190526-174703.pdf
└── Observability & Metrics for AI Agent.docx
```

## Verification
- `ruff check` on `src/` and `tests/`: ✅ all checks passed.
- Direct import check on 34 representative modules across every package: 34/34 OK.
- `pytest --collect-only -q tests/`: **1068 tests collected, 0 errors**.
- Targeted run (`tests/unit/eval/`, `tests/unit/agents/test_state.py`,
  `tests/unit/api/test_health.py`, `tests/unit/feedback_manager/`,
  `tests/unit/tools/test_adr_search.py`, `tests/unit/policy_vector_database/`):
  **69 passed, 0 failed**.
- `from src.api.app import create_app; create_app()` works — 28 routes registered.

## Backlog / Next Action Items
- P1 — pyproject.toml mentions `docs/` (Setup, Architecture, API docs) but
  no `docs/` directory exists. Consider creating it and moving the PDFs +
  docx into `docs/architecture/` for consistency.
- P1 — `Observability & Metrics for AI Agent.docx` is a binary doc at
  repo root; convert/move to `docs/observability.md` for diff-friendly
  versioning.
- P2 — README references `data/samples/` but actual path is
  `data/golden/`. Update the README "Layout" table.
- P2 — Pre-existing test isolation bug noted in `CLAUDE.md`:
  `tests/unit/core/test_local_directory_handler.py` clears
  `API_AUTH_TOKEN` for the whole environment, breaking `tests/unit/api/`
  if run together. Worth fixing the conftest.
- P2 — `tests/unit/eval/test_scoring.py` was importing from a path that
  never existed. Audit other tests for similarly stale imports.
- Future — Add a `Makefile` or `tox.ini` mirroring the CI matrix
  (`lint`, `type-check`, `test-unit`, `test-integration`, `build`).
