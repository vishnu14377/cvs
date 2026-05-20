# Care Connect ADR AI Agent — Project Analysis

> Synthesised from the architecture diagram (`CC_ADR_AI_AGENT_ARCHITECTURE.pdf`),
> the handoff document (`SEA-ADR AI Agent — Handoff Document-190526-174703.pdf`),
> the observability spec (`Observability & Metrics for AI Agent.docx`),
> `README.md`, `CLAUDE.md`, `pyproject.toml`, the Unqork BYO widget README,
> and the actual source code under `src/`.

---

## 1. Product Goal

An **AI-powered clinical assistant** embedded inside CVS Health's *CareConnect
Clinical Data Viewer (CDV)*. It lets clinical reviewers ask natural-language
questions about a patient's **Additional Document Request (ADR)** packet — the
chart, itemised bill, discharge summary, etc. — and cross-reference clinical
**policy bulletins**, all while keeping every answer grounded in the uploaded
documents.

| Concern             | Decision                                                                  |
|---------------------|---------------------------------------------------------------------------|
| Reviewer experience | Conversational, multi-turn chat inside an iframe in the existing CDV UI   |
| Grounding           | Strict — every claim must be verifiable in retrieved chunks; refuses otherwise |
| PHI safety          | Regex redaction on every external surface (logs, MongoDB, API responses)  |
| Continuity          | Sessions survive pod restarts (Cloud SQL checkpointer)                    |
| Cost / latency      | Gemini 2.5 Flash + text-embedding-004; HNSW pgvector; warmup on session creation |

Business owners (per the handoff document):

- **AI Agent (Python app):** Nathan's team — repo `careconnect-adr_ai_agent`
- **Java widget integration:** Nathan's team — repo `care-connect_member-adr-oci`
- **AI Engine (Gemini prompts, grounding):** Farhan's team
- **GKE / Cloud SQL / GCS / Secrets infra:** Si / Leif
- **Unqork CDV platform:** CareConnect platform team

---

## 2. High-Level Architecture (from the diagram)

```
                ┌─────────────────────────────────────────────────────┐
   User (CDV) ─►│  Unqork UI  ──── iframe + postMessage ────►  Widget │
                │  (Java OCI) ──── POST /api/v1/sessions ──►   API    │
                └─────────────────────────────────────────────────────┘
                                          │ bearer-token auth
                                          ▼
                              ┌──────────────────────┐
                              │  FastAPI REST API    │ ◄── /health, /widget/v1/*, /api/v1/*
                              │  (src/api/app.py)    │
                              └─────────┬────────────┘
                                        │
                                        ▼
                              ┌──────────────────────┐
                              │  LangGraph Agent     │ ◄── per-process singleton
                              │  generate → grounding│      (src/agents/graph.py)
                              │  → tools → loop      │
                              └─────────┬────────────┘
                                        │ tool calls
        ┌─────────────────────────┬─────┴─────┬──────────────────────────┐
        ▼                         ▼           ▼                          ▼
 ┌────────────┐          ┌────────────┐ ┌────────────┐         ┌──────────────────┐
 │ adr_search │          │ adr_summary│ │policy_*    │         │ document         │
 │ (BM25 +    │          │ (LLM call) │ │tools (list/│         │ processors       │
 │  semantic) │          │            │ │search/summ)│         │ (OCR → ingest)   │
 └─────┬──────┘          └─────┬──────┘ └─────┬──────┘         └─────────┬────────┘
       │ pgvector              │ Gemini 2.5   │ pgvector                 │
       ▼                       ▼ Flash        ▼                          ▼
 ┌─────────────────────────────────────────────────────┐         ┌──────────────┐
 │  Cloud SQL Postgres — schema "carapp"               │         │ Mistral OCR  │
 │  - langchain_pg_collection  (collection registry)   │         │ (25.05) or   │
 │  - langchain_pg_embedding   (vectors + metadata)    │         │ Gemini OCR   │
 │  - ai_sessions / checkpoints / checkpoint_blobs /   │         └──────┬───────┘
 │    checkpoint_writes / checkpoint_migrations        │                │
 │  Collections: "ADR_session_documents", "policy_documents"             │
 └─────────────────────────────────────────────────────┘                ▼
                                              ┌─────────────────────────────────┐
                                              │ Google Cloud Storage             │
                                              │ bucket: care_connect_ai_initiatives│
                                              │ prefix: adr_ai_agent/             │
                                              │   uploads/{id}/{filename}.pdf     │
                                              │   tmp/                            │
                                              │   extracted_text/{session_id}/    │
                                              └─────────────────────────────────┘

                              MongoDB ◄── Feedback Manager (thumbs up/down + comments)
```

### Layers (per `pyproject.toml` `where=["src"]`)

| Layer              | Package                              | Responsibility                                            |
|--------------------|--------------------------------------|-----------------------------------------------------------|
| Presentation       | `src/api/`                           | FastAPI app, routes, middleware, templates, validation    |
| Application/Orchestration | `src/agents/`, `src/session_manager/` | LangGraph workflow + per-session lifecycle           |
| Domain tools       | `src/tools/`                         | LangChain `BaseTool`s the agent picks from                |
| Domain pipelines   | `src/ocr/`, `src/adr_document_processor.py` | OCR + ingest orchestration                          |
| Data access        | `src/adr_vector_database/`, `src/policy_vector_database/`, `src/feedback_manager/` | RAG stores + feedback store     |
| Infrastructure     | `src/core/`                          | Clients for GCS / CloudSQL / pgvector / Vertex / LangChain; config; logger; secrets |
| Cross-cutting      | `src/eval/`, `src/utils/`            | Evaluation harness + shared helpers                       |

---

## 3. Data Flows

### 3.1 Document ingestion (one-shot at session creation)

```
Java OCI ──POST /api/v1/sessions/upload {file}──► sessions.py
                                                     │
                                                     ▼
                                       initialize_session()           src/session_manager/initialization.py
                                                     │
                                                     ▼
                                          SessionManager()            src/session_manager/core/session_manager.py
                                                     │  manager.initialize()
                                                     ▼
                                  AdrDocumentProcessor.process()      src/adr_document_processor.py
                                                     │
                          ┌──────────────────────────┴──────────────────────────┐
                          ▼                                                     ▼
                 (1) OcrOrchestrator                                  (2) ingest_session()
                 src/ocr/ocr_orchestrator.py                          src/adr_vector_database/ingestion_pipeline.py
                          │                                                     │
                          ▼                                                     ▼
                 - split PDF by size_limit_mb (default 5 MB) /         - read extracted JSONs from GCS
                   pages_per_chunk                                      - DocumentChunker (1000 chars / 200 overlap)
                 - per sub-file, call Mistral OCR                       - get_embedding_client → text-embedding-004
                   (model `mistral-ocr-2505`) or Gemini Vision          - VectorStoreManager.batch_insert (HNSW pgvector)
                 - persist raw extracted text JSON to GCS               - tag every chunk with metadata.session_id
                                                                        - returns BatchIngestionResult
                                                     │
                                                     ▼
                              session_id (generated), AdrProcessingResult
                                                     │
                                                     ▼
                                 in-memory registry[session_id] = (manager, time.time())
                                                     │
                                                     ▼
                                background_tasks.add_task(warmup_session)   src/session_manager/warmup.py
                                  → "Summarize the key clinical findings…" through a `{session_id}-warmup` thread
                                    so it pre-warms LLM + retriever WITHOUT polluting real chat history
```

OCR engine choices (per `_OCR_ENGINE_MAP` in `routes/sessions.py`):
`mistral`, `mistral-ocr` → `"mistral"` model; `gemini-vision`, `document-ai` → `"llm"` model (the `document-ai` mapping is a TODO).

### 3.2 Query / answer flow (every user message)

```
Widget JS ──POST /widget/v1/chat/query──► widget.py        (also: /api/v1/sessions/{id}/query)
                                              │
                                              ▼
                            (a) regex injection check     src/api/validation/input_safety.py
                                              │
                            (b) async LLM safety classifier (timeout 3s, fail-open SAFE)
                                              │
                                              ▼
                            invoke_graph(graph, message, session_id)    src/agents/graph.py
                                              │
                                              ▼
                            ┌──────────────  StateGraph(AgentState)  ──────────────┐
                            │                                                       │
                            │  generate ── grounding_gate ── tools_condition       │
                            │    │            (observer)        │                  │
                            │    │                              ├─► inject_session │
                            │    │                              │      │           │
                            │    │                              │      ▼           │
                            │    │                              │   ToolNode       │
                            │    │                              │      │           │
                            │    │                              ▼      ▼           │
                            │    └─◄────────────────────────  loop   END           │
                            └───────────────────────────────────────────────────────┘
                                              │
                                              ▼
                            judge_grounding(ai_content, tool_messages, session_id)
                                              │
                                              ├─► GROUNDED  → return as-is
                                              ├─► PARTIAL   → append medical disclaimer
                                              └─► UNGROUNDED→ replace with "I could not verify…"
                                              │
                                              ▼
                            render_to_safe_html + render_to_base64 (Markdown→HTML, bleach-sanitised)
                                              │
                                              ▼
                            QueryResponse { message_id, content, content_html, content_base64,
                                            sources[ ], metadata{ processing_time_ms, tokenUsage, grounding } }
```

Streaming variant (`/api/v1/sessions/{id}/query/stream`) emits SSE
`event: token | tool_call | tool_result | done | error`. The final
`done` payload contains the grounded content so the client overwrites
streamed tokens if the grounding judge rewrote them.

### 3.3 Session lifecycle

| Phase            | Mechanism                                                                                       |
|------------------|-------------------------------------------------------------------------------------------------|
| Create           | `POST /api/v1/sessions` (GCS URI) or `POST /api/v1/sessions/upload` (multipart PDF)             |
| Live state       | In-memory dict in `src/api/dependencies.py`: `session_id → (SessionManager, created_ts)`        |
| Conversation     | LangGraph checkpointer keyed on `thread_id = session_id` (MemorySaver in dev, AsyncPostgresSaver in prod) |
| Warmup           | `BackgroundTask` fires `{session_id}-warmup` thread to pre-warm Vertex + pgvector connection    |
| Expiry           | `cleanup_expired_sessions(ttl_hours=24)` hourly loop in `app.py` lifespan                       |
| Rehydration      | On miss in memory, the Java OCI service re-creates the session; checkpointer recovers history   |
| Explicit delete  | `DELETE /api/v1/sessions/{id}` → `delete_session()` clears vectors + checkpoint                 |
| Widget signal    | `careconnect:session-expired` postMessage → Unqork triggers `buttonClick(null,"ADRRefresh")`    |

---

## 4. The LangGraph Agent (`src/agents/`)

### 4.1 State

```python
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]  # accumulating reducer
    session_id: str                                            # scopes retrieval + checkpointing
```

### 4.2 Graph topology

```
START → generate → grounding_gate (observational) → tools_condition
                                                       │
                                                       ├──"tools"──→ inject_session_id → ToolNode ──┐
                                                       │                                            │
                                                       └──"__end__"──→ END                          │
                                                                                                    ▼
                                                                                                generate (loop)
```

* **`generate_node`** — uses `LangChainClient` (Gemini 2.5 Flash) with
  `bind_tools(tools)`. Trims history to last `max_turns=10` turns.
  System prompt (in `generate_node.py`) enforces 7 strict rules:
  always search before answering, never use training knowledge for
  clinical facts, always cite, refuse medical-advice requests, never
  speculate beyond document text.
* **`grounding_gate`** — pure observer; logs a `GROUNDING_ALERT` whenever
  the final AI message has no preceding `ToolMessage` (i.e. answered
  without searching).
* **`inject_session_id`** — patches every tool call whose name is in
  `{"adr_search","adr_summary","policy_search","policy_summary"}` to
  inject `session_id` from `AgentState` into the tool args. The arg is
  hidden from the LLM via `model_json_schema` override.
* **`ToolNode`** — LangGraph's prebuilt executor. Tools available:
  `adr_search`, `adr_summary`, `policy_search`, `policy_list`,
  `policy_summary` (wired in `agent_factory._build_agent`).

The agent itself is a **process-wide singleton** (`src/session_manager/core/agent_factory.py`).
Session isolation is achieved purely by `AgentState.session_id`
(injected into tool args) and the checkpointer's `thread_id`. No
per-session graphs are compiled — that's a deliberate cost-saving
choice.

### 4.3 Tools

| Tool             | Backed by                                                              | Notes                                                |
|------------------|------------------------------------------------------------------------|------------------------------------------------------|
| `adr_search`     | `src/adr_vector_database/retriever.py` — semantic or BM25+semantic ensemble | `session_id` filter on every pgvector query; can switch to hybrid (default weights 0.5/0.5) |
| `adr_summary`    | `src/tools/adr_summary.py` (LLM-driven roll-up over retrieved chunks)  | Returns ADR summary text                             |
| `policy_search`  | `src/policy_vector_database/` via `VectorStoreSingleton`, collection `policy_documents` | Persistent corpus across sessions             |
| `policy_list`    | Reads policy metadata (no embeddings needed)                           | Returns list of available policy bulletins            |
| `policy_summary` | LLM summary of one or more policy bulletins                            |                                                      |

### 4.4 Retrieval details

* **Embedding model**: `text-embedding-004` (768 dims) — *but* the
  architecture diagram says `text-embedding-005`; the code defaults to
  `text-embedding-004` (`vectorstore_config.EMBEDDING_MODEL_ID`) and is
  overridable via `VECTORSTORE_EMBEDDING_MODEL_ID`.
* **Chunk size / overlap**: `1000 / 200` characters (overridable via
  `INGESTION_CHUNK_SIZE` / `INGESTION_CHUNK_OVERLAP`).
* **Index**: pgvector HNSW (`M=16`, `efConstruction=64`, `efSearch=40`)
  in Cloud SQL schema `carapp`.
* **Search type**: `similarity` by default; `mmr`
  (`fetch_k=20`, `lambda_mult=0.5`) and `similarity_score_threshold`
  also supported.
* **Top-k**: `4` documents per call (`RETRIEVER_DEFAULT_K`).
* **Hybrid mode**: optional BM25 + semantic via
  `langchain_classic.retrievers.EnsembleRetriever`. BM25 index is built
  on demand from `get_session_documents(session_id)` and cached per
  session in `HybridRetrieverManager`.

---

## 5. Safety & Guardrails

Three concentric rings before content leaves the API:

1. **Input safety** (`src/api/validation/input_safety.py`)
   * Fast regex catches: `ignore previous instructions`, `[SYSTEM]`,
     `<<SYS>>`, `[INST]`, `you are now`, `pretend to be`, etc.
   * Slow LLM classifier (3 s timeout, fail-open `SAFE`).
   * Both `/widget/v1/chat/query` and `/api/v1/sessions/{id}/query`
     reject `UNSAFE` queries with HTTP 400 and a generic message.

2. **System prompt + grounding gate** (in-graph)
   * Strict 7-rule prompt forces tool use and citation.
   * Observational gate logs `GROUNDING_ALERT` when the agent answers
     without searching.

3. **Grounding judge** (`src/api/validation/grounding_judge.py`)
   * A second LLM call evaluates whether every factual claim is
     supported by the retrieved tool messages.
   * Verdicts:
     * `GROUNDED` → return content as-is
     * `PARTIAL` → append a *medical disclaimer*
     * `UNGROUNDED` → replace content with
       `"I could not verify this information from the uploaded documents…"`
   * 10 s timeout, fail-open `PARTIAL` + disclaimer.

4. **PHI redaction** (`src/api/validation/phi_redaction.py`)
   * Applied on **external surfaces only** (logs, MongoDB feedback,
     API responses), **not** to internal agent context.
   * Patterns: MRN, SSN (two forms), DOB, phone, email, Member/Insurance
     ID, "Patient/Name/Subscriber:" lines.

5. **HTML rendering** (`src/api/rendering/html_renderer.py`)
   * Markdown → HTML through `markdown` + `bleach` sanitisation; both
     UTF-8 and base64-encoded output for the Unqork iframe.

---

## 6. Persistence

| Store               | What                                                                    | Path / table                                     |
|---------------------|-------------------------------------------------------------------------|--------------------------------------------------|
| Cloud SQL `carapp`  | RAG embeddings + collection registry                                    | `langchain_pg_collection`, `langchain_pg_embedding` |
| Cloud SQL `carapp`  | LangGraph checkpoints (multi-turn conversation state)                   | `ai_sessions`, `checkpoints`, `checkpoint_blobs`, `checkpoint_writes`, `checkpoint_migrations` |
| GCS bucket          | Raw uploaded PDFs                                                       | `gs://care_connect_ai_initiatives/uploads/{id}/{filename}.pdf` |
| GCS bucket          | OCR intermediate files                                                  | `gs://…/tmp/` + `gs://…/extracted_text/{session_id}/` |
| MongoDB             | User feedback                                                           | `FeedbackRecord` — session_id, message_id, redacted user msg + AI response, rating, comment, document_names, tools_used, created_at |
| Process memory      | `session_id → (SessionManager, created_ts)`                            | `src/api/dependencies._session_registry`         |
| Process memory      | `VectorStoreSingleton._vector_stores[collection_name]`                  | Singleton pgvector clients                       |
| Process memory      | `HybridRetrieverManager._ensemble_retrievers[session_id]`               | Per-session BM25+semantic ensemble cache         |

Cloud SQL is reached via `psycopg[binary,pool]` through
`SQLAlchemy`. `get_cloudsql_client()` is a thread-safe singleton with
pool sizing from env (`POOL_SIZE=10`, `MAX_OVERFLOW=20`, recycle
1800 s). pgvector extension is **assumed to exist** — the client just
verifies it.

---

## 7. REST API surface

Auth: `Bearer ${API_AUTH_TOKEN}` or `Bearer ${CARECONNECT_WIDGET_TOKEN}`
on every route except `/health*`, `/chat/{session_id}` (HTML page),
`/widget/v1/chat/ui`, `/widget/v1/chat/fragment`.

| Group        | Method & path                                            | Purpose                                                   |
|--------------|----------------------------------------------------------|-----------------------------------------------------------|
| Health       | `GET /health`                                            | Liveness                                                  |
|              | `GET /health/ready`                                      | Probes Cloud SQL, GCS, MongoDB, Vertex AI                 |
| Sessions     | `GET    /api/v1/sessions`                                | List sessions                                             |
|              | `POST   /api/v1/sessions`                                | Create from GCS URIs                                      |
|              | `POST   /api/v1/sessions/upload`                         | Multipart PDF upload → uploads to GCS → processes        |
|              | `GET    /api/v1/sessions/{id}`                           | Status + metadata                                         |
|              | `DELETE /api/v1/sessions/{id}`                           | Delete vectors + checkpoint                               |
|              | `GET    /api/v1/sessions/initialize/stream`              | SSE stream of initialization progress                     |
| Query        | `POST   /api/v1/sessions/{id}/query`                     | One-shot agent answer                                     |
|              | `POST   /api/v1/sessions/{id}/query/stream`              | SSE stream of agent answer + tool events                  |
| History      | `GET    /api/v1/sessions/{id}/history`                   | Reconstructed conversation                                |
| Feedback     | `POST   /api/v1/sessions/{id}/feedback`                  | Thumbs/comment → Mongo (PHI-redacted)                     |
|              | `GET    /api/v1/sessions/{id}/feedback`                  | List feedback for session                                 |
| Policies     | `GET    /api/v1/policies`                                | List clinical policy bulletins                            |
|              | `POST   /api/v1/policies/batch`                          | Batch operations                                          |
|              | `GET    /api/v1/policies/{id}`                           | Single bulletin                                           |
| Widget       | `POST   /widget/v1/chat/query`                           | Unqork-formatted answer (base64 + sanitized HTML)         |
|              | `GET    /widget/v1/chat/ui`                              | Self-contained iframe chat UI                             |
|              | `GET    /widget/v1/chat/fragment`                        | HTML fragment (when Unqork can't iframe)                  |
| Chat        | `GET    /chat/{id}`                                       | Standalone full-page chat UI                              |
| Dev         | `/dev/test`                                              | Test harness (gated by `ENABLE_DEV_ROUTES=1`)             |

`ObservabilityMiddleware` stamps every response with `x-request-id` and
`x-response-time-ms`, and emits a structured log line with `request_id`,
`method`, `path`, `status`, `duration_ms`, `session_id` (parsed from
`/api/v1/sessions/{id}/…`), and `endpoint`.

---

## 8. Unqork / Java integration

The widget is registered as a **BYO** (Bring Your Own) component in
Unqork (`unqork/adr-chatbot.js` + `manifest.json`). Configuration
properties: `apiBaseUrl`, `authToken`, `sessionId`. The Java OCI
service (`care-connect_member-adr-oci`, not in this repo) is the
intermediary between Unqork and this API:

```
CDV (Unqork) ──FreeMarker template ClinicalViewerDisplay.ftl──► Java OCI
                                                                  │
                          ConcurrentHashMap session cache         │
                          (500 max, LRU)                          │
                                                                  ▼
                                              POST /api/v1/sessions/upload
                                                  │ Bearer ${API_AUTH_TOKEN}
                                                  ▼
                                              ADR AI Agent (this repo)
                                                  │ returns session_id
                                                  ▼
                                              Unqork iframes:
                                                  GET /widget/v1/chat/ui?sessionId=…
                                                       &layout=embedded&mode=iframe
                                                  Bearer ${CARECONNECT_WIDGET_TOKEN}
```

**iframe ↔ parent postMessage protocol** (all messages carry `source: "careconnect"`):

| Event                          | Direction         | Payload              | Purpose                                  |
|--------------------------------|-------------------|----------------------|------------------------------------------|
| `careconnect:ready`            | iframe → parent   | `{session_id, mode}` | Widget is ready                          |
| `careconnect:height`           | iframe → parent   | `{height}`           | Resize iframe to content                 |
| `careconnect:session-expired`  | iframe → parent   | `{session_id}`       | Tells Unqork to call `ADRRefresh`        |
| `careconnect:scroll-top`       | iframe → parent   | `{}`                 | Scroll parent to top                     |
| `careconnect:close`            | iframe → parent   | `{}`                 | User wants to close widget               |

---

## 9. Deployment

* **GKE cluster**: `careconnect-gke` in `us-east4`, project `hcb-dev-careconnect-etl`.
* **Namespace**: `careconnect-ai-dev1`.
* **Pod**: FastAPI on `uvicorn` port 8000, `2 ≤ replicas ≤ 3`,
  `targetCPU=80%`.
* **Istio sticky sessions**: `ConsistentHash` on header `X-Session-Id`
  → routes the same session_id to the same pod whenever possible.
* **Internal host**: `internal-careconnect-ai-dev1.careconnect-gke.cvshealth.com`.
* **Image**: built from `Dockerfile` (Python 3.10-slim, `COPY src/`,
  `pip install -e .`); CI flow `GHA → GAR → GKE`. Stargate CD flow is
  still pending end-to-end validation (PR #335 merged but untested).
* **Branches**: `non-prod` deploys to dev1; `release/2026.05.R1`
  triggers Docker publish (commit must include `[publish]` or be on a
  `release/**` branch).
* **Helm values**: `deploy-configs/careconnect-ai-dev1/values/v1.yaml`
  (currently embeds secret literals — flagged as tech debt; should
  switch to external-secrets / Vault).
* **Workload Identity**: KSA `careconnect-adr-ai-agent` →
  GSA `careconnect-ai-features@hcb-dev-careconnect-etl.iam.gserviceaccount.com`.
  Roles: Secret Manager Secret Accessor, Storage Object Viewer, Vertex AI User.

`src/core/secrets.py::load_secrets_from_gcp` runs at FastAPI import
time, hydrating `os.environ` from GCP Secret Manager **before**
`src/core/config.py` reads them.

---

## 10. Configuration (env vars)

| Variable                       | Default                  | Read by                                   |
|--------------------------------|--------------------------|-------------------------------------------|
| `APP_ENV`                      | —                        | (informational)                           |
| `LOG_LEVEL`                    | `INFO`                   | `src/core/logger.py`                      |
| `API_PORT`                     | `8000`                   | uvicorn                                   |
| `API_AUTH_TOKEN`               | (Secret Mgr)             | `src/api/dependencies.verify_token`       |
| `CARECONNECT_WIDGET_TOKEN`     | (Secret Mgr)             | Widget JS + `verify_token`                |
| `CARECONNECT_PARENT_ORIGIN`    | —                        | Widget UI postMessage origin check        |
| `CARECONNECT_API_BASE`         | —                        | Widget HTML template                      |
| `GCP_PROJECT_ID`               | —                        | All GCP clients                           |
| `GCP_REGION`                   | `us-central1`            | Vertex region                             |
| `GCS_BUCKET_NAME`              | —                        | `core.config.GlobalConfig`                |
| `AI_AGENT_PREFIX_GCS`          | —                        | Working prefix in GCS                     |
| `DEV_GCS_SEED_BUCKET`          | —                        | Dev-only fake-gcs bucket auto-create      |
| `OCR_GCS_TEMP_FOLDER`          | `tmp`                    | OCR scratch                               |
| `OCR_GCS_EXTRACTED_TEXT_FOLDER`| `extracted_text`         | Where extracted JSONs land                |
| `OCR_SMALL_SIZE_FILE_LIMIT_MB` | `4`                      | PDF size threshold                        |
| `OCR_MAX_PAGES_PER_DOCUMENT`   | `30`                     | Hard cap per doc                          |
| `OCR_PAGES_PER_CHUNK`          | `20`                     | Pages-per-sub-file (when not size-split)  |
| `MISTRAL_MODEL_ID`             | `mistral-ocr-2505`       | OCR client                                |
| `MISTRAL_PUBLISHER`            | `mistralai`              | Vertex publisher                          |
| `GEMINI_MODEL_ID`              | `gemini-2.5-flash`       | LangChainClient                           |
| `LLM_*`                        | (temperature 0.7, top_p 0.95, top_k 40, max_out 8192, retries 3) | Generation params |
| `VECTORSTORE_EMBEDDING_MODEL_ID`| `text-embedding-004`    | Embedding client                          |
| `VECTORSTORE_EMBEDDING_DIMENSION`| `768`                  | pgvector column dim                       |
| `VECTORSTORE_HNSW_M / EF_CONSTRUCTION / EF_SEARCH` | `16 / 64 / 40` | Index tuning                          |
| `RAG_COLLECTION_NAME`          | `ADR_session_documents`  | ADR pgvector collection                   |
| `RETRIEVER_DEFAULT_K / FETCH_K`| `4 / 20`                 | Retriever knobs                           |
| `RETRIEVER_LAMBDA_MULT`        | `0.5`                    | MMR diversity                             |
| `RETRIEVER_SEARCH_TYPE`        | `similarity`             | Default search type                       |
| `INGESTION_CHUNK_SIZE / OVERLAP / MAX_WORKERS` | `1000 / 200 / 4` | Chunker + ingest                    |
| `RAG_SESSION_TTL_HOURS`        | `24`                     | Session expiry                            |
| `CLOUDSQL_HOST / PORT / DATABASE / USER / PASSWORD / DATABASE_SCHEMA` | (Secret Mgr) | psycopg                              |
| `CARECONNECT_DEV_DATABASE`     | —                        | Cloud SQL Connector instance name (optional) |
| `CLOUDSQL_POOL_SIZE / MAX_OVERFLOW / POOL_TIMEOUT / POOL_RECYCLE` | `10 / 20 / 30 / 1800` | SQLAlchemy pool |
| `MONGODB_URI / MONGODB_DATABASE` | (Secret Mgr)           | feedback_manager                          |
| `VERTEX_AI_MODE`               | `real` (or `stub` for CI)| Toggles deterministic offline mode        |
| `ENABLE_DEV_ROUTES`            | unset                    | Enables `/dev/test` + CORS                |

---

## 11. Observability (current + target)

**Already wired** (`src/api/middleware/observability.py`):
* OpenTelemetry FastAPI instrumentation (auto traces).
* Per-request `x-request-id`, `x-response-time-ms` headers.
* Structured log line per request: `{request_id, method, path, status, duration_ms, session_id, endpoint}`.
* `token_usage` log event from `routes/query.py` with `input_tokens`,
  `output_tokens`, `session_id`, `endpoint`.
* `GROUNDING_ALERT` warn when the agent answers without searching.

**Target metrics** (per the observability doc — *aspirational*):

| Category    | Metrics                                                                                                  |
|-------------|----------------------------------------------------------------------------------------------------------|
| Retrieval   | Recall@k, citation/source-page accuracy                                                                  |
| Generation  | Groundedness/faithfulness, answer correctness, hallucination rate, refusal correctness, tool-use correctness |
| Privacy     | PHI leak rate (% of external responses where redaction missed something)                                 |
| Latency     | End-to-end p50/p95, per-stage latency (retrieval, LLM, grounding gate), ingest latency, tool-loop count  |
| Reliability | Error rate (5xx / timeout)                                                                              |
| Cost        | Tokens/query, $/query, $/session, $/1000 queries                                                         |

Dashboards, alert thresholds, SLO targets, sampling, and owners are
**not yet specified** in the observability doc — those are open
follow-ups.

---

## 12. Testing strategy (`tests/`)

| Layer       | Path                                      | Notes                                                                 |
|-------------|-------------------------------------------|-----------------------------------------------------------------------|
| Unit        | `tests/unit/*` (1 sub-pkg per `src/*` package) | Mocks `langchain_postgres` via `tests/unit/{core,session_manager}/conftest.py` to skip optional deps |
| Integration | `tests/integration/`                      | Requires `docker compose up` + `bash docker/seed-gcs.sh`              |
| E2E         | `tests/e2e/`                              | Playwright; opt-in via `pip install -e ".[e2e]"`                      |

`tests/conftest.py` auto-tags by directory (`-m unit`, `-m integration`, `-m e2e`).
Total collected suite: **1068 tests** (post-reorg).

Known limitation per `CLAUDE.md`: `tests/unit/api/` + `tests/unit/core/`
run together fail because `test_local_directory_handler.py` calls
`patch.dict(os.environ, {}, clear=True)` which nukes `API_AUTH_TOKEN`
for the whole process. Run them in isolation or fix the conftest.

---

## 13. CI / CD

`.github/workflows/ci.yml` runs on every PR + push to `main`:

| Job                | Tool                                                    |
|--------------------|---------------------------------------------------------|
| `lint`             | `ruff check src/ tests/` + `ruff format --check`        |
| `type-check`       | `mypy src/`                                             |
| `test-unit`        | `pytest -m unit` (with coverage)                        |
| `test-integration` | `pytest -m integration` against `docker compose` stack  |
| `build`            | `docker build` (no push)                                |
| `security`         | `pip-audit` + `trivy image` (advisory)                  |
| `semantic-release` | Conventional-commit driven tag + CHANGELOG (main only)  |

Plus `.github/workflows/commitlint.yml` for conventional-commit PRs.
The CI uses `docker-compose.ci.yml` layered on `docker-compose.yml` to
drop the host ADC bind mount and set `VERTEX_AI_MODE=stub`.

---

## 14. Known issues / tech debt (from the handoff doc)

| #  | Issue                                                                                     | Severity         |
|----|-------------------------------------------------------------------------------------------|------------------|
| 1  | Stargate CD bypass — current deploys are manual `docker build && push GAR`                | P1 near-term     |
| 2  | `/health/ready` doesn't verify DB; if DB is down the agent silently falls back to MemorySaver and loses sessions on pod restart | P1 |
| 3  | `v1.yaml` ships hardcoded secret values — migrate to external-secrets/Vault              | P1 (low risk now)|
| 4  | `test_dependencies.py` (5 tests) requires local Postgres; not blocking CI                 | P2               |
| 5  | Grounding judge over-refuses meta-questions (safety by design, but tuneable)              | P2               |
| 6  | In-memory vector store per session — embeddings lost on pod restart (docs re-processable) | P2 (future)      |
| 7  | Java `ConcurrentHashMap` has no cross-pod sync; Java restart → extra session creations     | P3 (handled)     |
| 8  | `langgraph-checkpoint-postgres.PostgresSaver.from_conn_string` is a `@contextmanager` footgun — already worked around with `AsyncPostgresSaver` | Doc-only |

---

## 15. Top-of-mind observations

1. **The architecture diagram refers to `text-embedding-005`, but the
   code default is `text-embedding-004` (768 dims).** Either the
   diagram is slightly out of date, or the code needs an env override
   in prod. Worth confirming with Farhan's team.

2. **The agent is a global singleton**, deliberately — the per-session
   work is in (a) the in-memory `_session_registry` (manager + retriever
   cache), and (b) the `thread_id=session_id` checkpointer key. Any
   refactor must keep that invariant or memory cost explodes.

3. **Sticky sessions via Istio `X-Session-Id` hash** are a soft
   optimisation — if a pod dies, the checkpointer rehydrates from
   Cloud SQL. But the in-memory `HybridRetrieverManager` + BM25 index
   will be lost and rebuilt on first query. Worth measuring first-query
   latency after a pod restart.

4. **PHI redaction is regex-based and applied only on external
   surfaces.** It will not catch every name or every member ID format.
   The "Patient/Name/Subscriber: …" pattern, in particular, only
   matches a very narrow shape. This is a known limitation; the
   grounding judge isn't a substitute, since it doesn't reason about PHI.

5. **The grounding judge is itself a Gemini call** — so a sustained
   incident in Vertex AI cascades to (a) the answer, (b) the grounding
   verdict, and (c) the safety classifier. All three fall open
   (`SAFE` / `PARTIAL` + disclaimer) on timeout, which is the right
   safety/availability trade-off but worth alerting on.

6. **Unqork is doing a lot of UX heavy-lifting outside this repo** —
   anything that touches `careconnect:*` postMessage events, the
   FreeMarker template, or `ADRRefresh` button wiring lives in the
   sibling `care-connect_member-adr-oci` Java repo and the Unqork
   module. The contract between the two is the **REST API + the
   `careconnect:*` postMessage protocol** — keep that contract stable.

7. **The repo has no `docs/` folder yet** even though the README links
   to `docs/setup.md`, `docs/architecture.md`, `docs/api_documentation.md`,
   `docs/guides/quick_setup.md`. That's a documentation gap — converting
   the architecture PDF + handoff PDF + observability docx into MD
   files under `docs/` would close it. *This file is a starting point.*
