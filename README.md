# Care Connect ADR AI

An intelligent document Q&A system for processing and analyzing Additional Document Request (ADR) reports using Retrieval-Augmented Generation (RAG).

## Overview

Care Connect ADR AI is an AI-powered agent that processes ADR (Additional Document Request) PDF documents and enables natural language Q&A interactions. The system extracts text from documents using OCR, creates vector embeddings for semantic search, and uses RAG to provide accurate, context-aware answers. The system supports multi-RAG capabilities with tool calling, allowing you to query across multiple document types (e.g., ADR documents and Policy documents) simultaneously.

## Key Features

- **Document Processing**: Extract text from PDFs and images using Gemini Flash 2.5 or Mistral OCR
- **GCS Support**: Process files from local storage or Google Cloud Storage (gs:// URIs)
- **Robust Error Handling**: Automatic retries with exponential backoff for transient errors
- **Configurable**: Environment-based configuration for timeouts, retries, and model selection
- **Extensible**: Modular architecture for adding new document types and processing pipelines

## Quick Start

```bash
# Install dependencies
pip install -e .

# Authenticate with Google Cloud
gcloud auth application-default login

```

## Layout

| Path | Purpose |
|------|---------|
| `src/core/` | Configuration, logging, and AI clients (GenAI, Vertex AI) |
| `src/ocr/` | Document processing (LLM OCR, Mistral OCR) |
| `src/agents/` | Agent orchestration |
| `src/api/` | HTTP API and routes |
| `src/adr_vector_database/` | ADR document embeddings and retrieval |
| `src/policy_vector_database/` | Policy document embeddings and retrieval |
| `src/session_manager/` | Conversation and session state |
| `src/feedback_manager/` | User feedback storage and analysis |
| `tests/` | Unit and integration tests |
| `scripts/` | Environment setup and utilities |
| `docs/` | Architecture and setup documentation |
| `data/samples/` | Sample input files (not for production) |

## CI

Continuous integration runs on every PR and on merges to `main`. Workflow:
`.github/workflows/ci.yml`. Jobs:

| Job | What it runs | Blocking? |
|-----|--------------|-----------|
| `lint` | `ruff check` + `ruff format --check` | Soft (v1), blocking after baseline |
| `type-check` | `mypy src/` | Soft (v1), blocking after baseline |
| `test-unit` | `pytest -m unit` with coverage | Soft (v1), blocking after baseline |
| `test-integration` | `pytest -m integration` against live docker-compose stack | Soft (v1), blocking after baseline |
| `build` | `docker build` (no push) | Soft (v1), blocking after baseline |
| `security` | `pip-audit` + `trivy image` | Advisory (v1 and v2) |
| `semantic-release` | Conventional-commit → tag + CHANGELOG (main only) | Not required for PRs |

A separate `.github/workflows/commitlint.yml` enforces conventional commit
messages on PRs.

### Running checks locally

```bash
pip install -e ".[dev]"
ruff check src/ tests/
ruff format --check src/ tests/
mypy src/
pytest -m unit
docker compose up -d --build
./scripts/ci/wait_for_api.sh
pytest -m integration
docker compose down -v
```

> **Note:** CI layers `-f docker-compose.ci.yml` onto `docker compose up` to drop the
> host ADC bind mount (runners have no `gcloud auth application-default` credentials)
> and sets `VERTEX_AI_MODE=stub`. Locally, if you've run
> `gcloud auth application-default login`, the plain `docker compose up` above is
> correct — no override needed.

### Open CI questions (tracked in spec)

See `docs/superpowers/specs/2026-04-22-github-actions-ci-design.md` "Open
Questions for Engineering Lead" — runner label, semantic-release token,
Vertex AI mocking pattern, coverage tooling.

## Documentation

- [Quick Setup Guide](docs/guides/quick_setup.md) - End-to-end testing with `testing.py`
- [Setup Guide](docs/setup.md) - Installation and configuration
- [Architecture](docs/architecture.md) - System design and component details
- [API Documentation](docs/api_documentation.md) - API reference
