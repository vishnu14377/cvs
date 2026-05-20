# Monitoring & Validation — Engineering Implementation Guide
## Care Connect ADR AI Agent

> **For**: Backend, SRE, AI Engineering, Platform/DevOps.
> **From**: AI/ML Engineering.
> **Pre-reads**: `docs/architecture/PROJECT_ANALYSIS.md`,
> `docs/observability/01_BUSINESS_BRIEF.md`.
> **Status**: Detailed implementation guide. Code-level prescriptive.

---

## 0. TL;DR (read this first)

We are adding monitoring + validation to a production agentic-RAG system
that handles medical claims. Approach:

1. **Use what's already in the stack.** OpenTelemetry, structured logs,
   `src/eval/`, the golden corpus in `data/golden/`. Don't add new infra.
2. **One new log event** is the source of ~10 of our metrics.
   We don't ship StatsD/Prometheus — we use **Cloud Logging log-based
   metrics** + **Cloud Monitoring**.
3. **Validation has two paths**: (a) per-request safety/grounding (already
   in code, we just instrument it); (b) nightly golden-set regression
   (extend `src/eval/run_eval.py`).
4. **PHI handling is non-negotiable**: only redacted external surfaces ever
   leave the pod; sampled audit goes through Cloud DLP; PagerDuty payloads
   carry IDs only.

Total scope: ~19 engineer-days. Six phases, each independently shippable.

---

## 1. Final architecture

```
┌─────────────────── GKE pod: careconnect-adr-ai-agent ───────────────────────┐
│                                                                             │
│ FastAPI (src/api/app.py)                                                    │
│   ├─ OpenTelemetry FastAPIInstrumentor ──► OTLP exporter ──► Cloud Trace   │
│   ├─ ObservabilityMiddleware ──► JSON logs (stdout)                        │
│   │                                                                         │
│   ├─ routes/{query, widget, query_stream}.py                                │
│   │     └─► emit log: "agent.query.completed"   ◄── §3.1 (CRITICAL)        │
│   │                                                                         │
│   ├─ api/validation/input_safety.py                                         │
│   │     └─► emit log: "safety.injection.regex" / ".llm"   ◄── §3.2         │
│   │                                                                         │
│   ├─ session_manager/initialization.py + adr_document_processor.py          │
│   │     └─► emit log: "session.create.completed" / "ocr.completed" /       │
│   │                   "ingest.completed"                ◄── §3.3            │
│   │                                                                         │
│   └─ src/api/app.py lifespan ──► poll _session_registry size every 5 min   │
│           emit log: "session.registry.size"                                  │
└─────────────────────────────────────────────────────────────────────────────┘
                              │ stdout
                              ▼
        ┌─────────────────────────────────────────────────────────────┐
        │   Cloud Logging  (project: hcb-dev-careconnect-etl)         │
        │   - bucket _Default  (90 d retention)                       │
        │   - log router sinks:                                       │
        │       1.  filter:logName="...stdout" AND jsonPayload.event  │
        │           ──► BQ table careconnect_logs.app_events          │
        │       2.  random_sampling=0.01 AND event="agent.query.…"    │
        │           ──► BQ table careconnect_logs.responses_sample    │
        │           (CMEK, IAM-restricted, 30 d TTL)                  │
        └────────────────┬───────────────────────────┬────────────────┘
                         │                           │
                         ▼ (real-time)               ▼ (batch, nightly)
        ┌───────────────────────────────┐    ┌────────────────────────┐
        │ Cloud Monitoring              │    │ Cloud Workflows        │
        │ - 20 log-based metrics        │    │ ──► Cloud DLP inspect  │
        │ - 4 dashboards (Looker)       │    │     responses_sample   │
        │ - 15 alert policies           │    │ ──► dlp_findings BQ    │
        │ - 3 uptime checks             │    │ ──► PG-6 alert if hit  │
        └────────────────┬──────────────┘    └────────────────────────┘
                         │
                         ▼
        ┌───────────────────────────────┐    ┌────────────────────────┐
        │ PagerDuty (page-grade)        │    │ ServiceNow (tickets)   │
        └───────────────────────────────┘    └────────────────────────┘

        ┌──────────────────────────────────────────────────────────────┐
        │ Cloud Scheduler ──► Cloud Run Job "golden-eval" (nightly 03:00 ET)│
        │   docker image = same prod image                              │
        │   command = python -m src.eval.run_eval --golden-set --output-bq │
        │   ──► BQ table careconnect_eval.golden_runs                  │
        │   ──► Looker tile "Recall@k 7-day trend"                      │
        └──────────────────────────────────────────────────────────────┘
```

---

## 2. Phase plan & ownership

| Phase | Days | Owner | Blocks |
|---|---|---|---|
| 0. Wire OpenTelemetry + JSON logger | 2 | Backend | nothing |
| 1. Emit 3 new structured log events | 2 | Backend + AI eng | nothing |
| 2. Log-based metrics + dashboards | 3 | SRE + Backend | Phase 1 |
| 3. Alerts + runbooks | 2 | SRE | Phase 2 |
| 4. Offline validation (golden set + CI gate) | 5 | AI eng + Clinical reviewer | nothing |
| 5. PHI audit (sampler + DLP + alert) | 5 | SecOps + Backend | Phase 1 |

Phase 0/1 can start in parallel with Phase 4 (golden authoring).

---

## 3. Code changes — exact event contracts

> All events use this base envelope (`extra` to `logger.info(...)`):
> ```python
> {
>   "event": "<event_name>",
>   "request_id": str,                  # from ObservabilityMiddleware
>   "session_id": str | None,           # already PII-safe (generated UUID)
>   "ts": "ISO-8601 UTC",               # auto-added by formatter
> }
> ```
> The JSON formatter (Phase 0) ensures `extra` keys become top-level JSON
> fields so Cloud Logging promotes them automatically. **No PHI may
> appear in `extra` — only IDs, enums, counts, durations.**

### 3.1 `agent.query.completed`  (THE primary metric source)

**Emit from**: `src/api/routes/query.py` (after `judge_grounding`) and
`src/api/routes/widget.py` (same point) and at the end of
`src/agents/graph.py::stream_graph` for SSE.

```python
# src/api/routes/query.py — after the existing return block is computed
logger.info(
    "agent.query.completed",
    extra={
        "event": "agent.query.completed",
        "request_id": getattr(request.state, "request_id", None),  # set by middleware
        "session_id": session_id,
        "endpoint": "/api/v1/sessions/{id}/query",                 # template, not concrete
        "duration_ms": elapsed_ms,
        "tool_calls": [m.name for m in tool_messages] or [],       # list[str]
        "tool_loop_count": _count_loops(messages),                 # int
        "retrieval_empty": _was_retrieval_empty(tool_messages),    # bool
        "grounding_verdict": verdict,                              # GROUNDED|PARTIAL|UNGROUNDED|TIMEOUT
        "refused": ai_content.startswith("I could not verify"),    # bool
        "input_tokens": token_usage.get("prompt", 0),
        "output_tokens": token_usage.get("completion", 0),
        "cost_usd_est": _estimate_cost(token_usage),
        "safety_classifier": "SAFE" if safety == "SAFE" else "UNSAFE",
    },
)
```

Helpers to add (small, pure, unit-testable) in `src/api/routes/_metrics.py`:

```python
def _count_loops(messages: list) -> int:
    """Number of ToolMessage->AIMessage transitions in the conversation."""
    return sum(
        1 for prev, curr in zip(messages, messages[1:])
        if getattr(prev, "type", "") == "tool" and getattr(curr, "type", "") == "ai"
    )

def _was_retrieval_empty(tool_messages: list) -> bool:
    """True if EVERY adr_search/policy_search returned 'No relevant documents'."""
    search_msgs = [m for m in tool_messages if getattr(m, "name", "") in ("adr_search", "policy_search")]
    if not search_msgs:
        return False
    return all("No relevant documents found" in (m.content or "") for m in search_msgs)

# Gemini 2.5 Flash pricing as of Jan 2026 — keep in env so it's tweakable
_PRICE_PER_1K_IN = float(os.environ.get("GEMINI_PRICE_PER_1K_INPUT_USD", "0.000075"))
_PRICE_PER_1K_OUT = float(os.environ.get("GEMINI_PRICE_PER_1K_OUTPUT_USD", "0.0003"))

def _estimate_cost(usage: dict) -> float:
    return round(
        usage.get("prompt", 0) / 1000 * _PRICE_PER_1K_IN
        + usage.get("completion", 0) / 1000 * _PRICE_PER_1K_OUT,
        6,
    )
```

**`grounding_judge.py` change**: return `"TIMEOUT"` (not `"PARTIAL"`) when
`asyncio.TimeoutError` fires, so we can distinguish timeouts from real
PARTIAL verdicts:

```python
# src/api/validation/grounding_judge.py
except (TimeoutError, asyncio.TimeoutError):
    logger.warning("Grounding judge timed out: session=%s", session_id)
    return ("TIMEOUT", ai_content + _MEDICAL_DISCLAIMER)
```

### 3.2 `safety.injection.*`

**Emit from**: `src/api/validation/input_safety.py`.

```python
def check_injection_regex(message: str, request_id: str | None = None,
                          session_id: str | None = None) -> bool:
    if not message:
        return False
    for pattern in _COMPILED_PATTERNS:
        if pattern.search(message):
            logger.warning(
                "safety.injection.regex",
                extra={
                    "event": "safety.injection.regex",
                    "request_id": request_id,
                    "session_id": session_id,
                    "pattern": pattern.pattern[:40],
                    "blocked": True,
                },
            )
            return True
    return False

# Inside _classify_with_llm: when result is UNSAFE
logger.warning(
    "safety.injection.llm",
    extra={"event": "safety.injection.llm", "request_id": request_id,
           "session_id": session_id, "blocked": True},
)
```

Callers in `routes/query.py` and `routes/widget.py` pass `request_id` /
`session_id`. The middleware already stores `request_id` on
`request.state` (extend `ObservabilityMiddleware` to do so — one line).

### 3.3 Pipeline events

| Event | File | Required fields |
|---|---|---|
| `session.create.completed` | `routes/sessions.py` (both `create_session` and `create_session_upload`) | `success: bool`, `duration_ms`, `pages_processed`, `model_type`, `session_id` |
| `ocr.completed` | `src/ocr/ocr_orchestrator.py::run()` | `success`, `duration_ms`, `total_pages`, `total_sub_files`, `successful_sub_files`, `failed_sub_files`, `session_id` |
| `ingest.completed` | `src/adr_vector_database/ingestion_pipeline.py::ingest_session()` | `success`, `duration_ms`, `total_documents`, `successful_documents`, `total_chunks_stored`, `session_id` |
| `session.warmup.completed` | already in `src/session_manager/warmup.py` — promote to structured event | `success`, `duration_ms`, `session_id` |
| `session.registry.size` | new periodic task in `src/api/app.py` lifespan, every 5 min | `active_sessions: int`, `memory_bytes_estimated` |

### 3.4 JSON log formatter (Phase 0)

```python
# src/core/logger.py  (add a JSON formatter)
import json, logging, os, sys

class JsonFormatter(logging.Formatter):
    _STD_KEYS = {"args","asctime","created","exc_info","exc_text","filename",
                 "funcName","levelname","levelno","lineno","message","module",
                 "msecs","msg","name","pathname","process","processName",
                 "relativeCreated","stack_info","thread","threadName"}

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S.%fZ"),
            "severity": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Promote everything from `extra=...` to top-level
        for k, v in record.__dict__.items():
            if k not in self._STD_KEYS and not k.startswith("_"):
                payload[k] = v
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)

def get_logger(name: str) -> logging.Logger:
    log = logging.getLogger(name)
    if not log.handlers:
        h = logging.StreamHandler(sys.stdout)
        if os.environ.get("LOG_FORMAT", "json").lower() == "json":
            h.setFormatter(JsonFormatter())
        else:
            h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        log.addHandler(h)
        log.setLevel(os.environ.get("LOG_LEVEL", "INFO"))
        log.propagate = False
    return log
```

Set `LOG_FORMAT=json` in `deploy-configs/careconnect-ai-dev1/values/v1.yaml`.

### 3.5 OpenTelemetry → Cloud Trace (Phase 0)

```python
# src/api/app.py  (top of create_app, after middleware mount)
def _wire_otel(app: FastAPI) -> None:
    if os.environ.get("OTEL_ENABLED", "1") != "1":
        return
    from opentelemetry import trace
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.sdk.resources import Resource, SERVICE_NAME
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

    provider = TracerProvider(resource=Resource.create({
        SERVICE_NAME: "careconnect-adr-ai-agent",
        "service.namespace": os.environ.get("APP_ENV", "dev"),
        "service.version": os.environ.get("APP_VERSION", "0.0.0"),
    }))
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(provider)
    FastAPIInstrumentor.instrument_app(app)
```

Add dep: `opentelemetry-exporter-otlp-proto-grpc>=1.25.0` (others already
present). Cloud Trace receives OTLP directly when running on GKE with the
Cloud Trace API enabled — no agent required.

**Add custom spans** at the hot points using `tracer.start_as_current_span`:

| Span name | Wrap | Attributes |
|---|---|---|
| `agent.invoke` | `invoke_graph` body | `session_id`, `tool_calls_count` |
| `agent.tool.adr_search` | `ADRSearchTool._arun` | `session_id`, `mode`, `k`, `result_count` |
| `agent.tool.policy_search` | `PolicySearchTool._run` | `k`, `result_count` |
| `agent.grounding_judge` | `judge_grounding` | `verdict`, `tool_messages_count` |
| `agent.safety_classifier` | `_classify_with_llm` | `mode=regex|llm` |
| `pipeline.ocr` | `OcrOrchestrator.run` | `model_type`, `pages` |
| `pipeline.ingest` | `ingest_session` | `chunks` |

---

## 4. GCP resources (Terraform — module skeleton)

> Place under `deploy-configs/observability/`.  Use the existing
> Stargate `terraform/` workflow for apply.

### 4.1 Log-based metrics

```hcl
# deploy-configs/observability/metrics.tf
locals {
  log_metric = "logging.googleapis.com/user"
}

# Counter — UNGROUNDED responses
resource "google_logging_metric" "ungrounded" {
  name        = "careconnect/agent/ungrounded_count"
  description = "Number of agent answers judged UNGROUNDED."
  filter      = <<EOT
    resource.type="k8s_container"
    resource.labels.namespace_name="careconnect-ai-dev1"
    jsonPayload.event="agent.query.completed"
    jsonPayload.grounding_verdict="UNGROUNDED"
  EOT
  metric_descriptor {
    metric_kind = "DELTA"
    value_type  = "INT64"
    unit        = "1"
    labels {
      key         = "endpoint"
      value_type  = "STRING"
    }
  }
  label_extractors = {
    "endpoint" = "EXTRACT(jsonPayload.endpoint)"
  }
}

# Distribution — query duration_ms
resource "google_logging_metric" "query_duration_ms" {
  name        = "careconnect/agent/query_duration_ms"
  description = "Per-request agent query duration."
  filter      = <<EOT
    jsonPayload.event="agent.query.completed"
  EOT
  metric_descriptor {
    metric_kind = "DELTA"
    value_type  = "DISTRIBUTION"
    unit        = "ms"
    labels { key = "endpoint" value_type = "STRING" }
    labels { key = "grounding_verdict" value_type = "STRING" }
  }
  value_extractor = "EXTRACT(jsonPayload.duration_ms)"
  label_extractors = {
    "endpoint"          = "EXTRACT(jsonPayload.endpoint)"
    "grounding_verdict" = "EXTRACT(jsonPayload.grounding_verdict)"
  }
  bucket_options {
    exponential_buckets {
      num_finite_buckets = 64
      growth_factor      = 1.4
      scale              = 10
    }
  }
}
```

Repeat the **counter** pattern for: `grounding_partial_count`,
`grounding_timeout_count`, `retrieval_empty_count`, `no_tool_answer_count`,
`refused_count`, `injection_regex_count`, `injection_llm_count`,
`ocr_failure_count`, `ingest_failure_count`, `session_create_failure_count`.

Repeat the **distribution** pattern for: `tool_loop_count`,
`input_tokens`, `output_tokens`, `cost_usd_est`, `ocr_duration_ms`,
`ingest_duration_ms`.

### 4.2 Alerts (sample — replicate per PG-* / TK-* in §6 of the long plan)

```hcl
# deploy-configs/observability/alerts.tf
resource "google_monitoring_alert_policy" "ungrounded_rate" {
  display_name = "PG-4 — UNGROUNDED rate > 5% (30 min)"
  combiner     = "OR"
  conditions {
    display_name = "ungrounded share > 5%"
    condition_threshold {
      filter = <<EOT
        metric.type="logging.googleapis.com/user/careconnect/agent/ungrounded_count"
        resource.type="k8s_container"
      EOT
      comparison      = "COMPARISON_GT"
      threshold_value = 0.05
      duration        = "1800s"
      aggregations {
        alignment_period     = "300s"
        per_series_aligner   = "ALIGN_RATE"
        cross_series_reducer = "REDUCE_SUM"
      }
      denominator_filter = <<EOT
        metric.type="logging.googleapis.com/user/careconnect/agent/query_total_count"
      EOT
      denominator_aggregations {
        alignment_period     = "300s"
        per_series_aligner   = "ALIGN_RATE"
        cross_series_reducer = "REDUCE_SUM"
      }
    }
  }
  notification_channels = [google_monitoring_notification_channel.pagerduty_ai.id]
  documentation {
    content   = file("${path.module}/runbooks/PG-4.md")
    mime_type = "text/markdown"
  }
}
```

You will need a sibling counter `query_total_count` (the denominator);
emit the same way, no `grounding_verdict` filter.

### 4.3 Uptime checks

```hcl
resource "google_monitoring_uptime_check_config" "health" {
  display_name = "ADR-AI /health"
  timeout      = "10s"
  period       = "60s"
  http_check {
    path           = "/health"
    port           = 443
    use_ssl        = true
    validate_ssl   = true
  }
  monitored_resource {
    type = "uptime_url"
    labels = {
      host       = "internal-careconnect-ai-dev1.careconnect-gke.cvshealth.com"
      project_id = var.project_id
    }
  }
}
```

Repeat for `/health/ready` (5 min). Synthetic-query uptime check is more
involved — see §7.

### 4.4 Log sinks → BigQuery

```hcl
resource "google_logging_project_sink" "app_events" {
  name        = "careconnect-app-events"
  destination = "bigquery.googleapis.com/projects/${var.project_id}/datasets/careconnect_logs"
  filter      = <<EOT
    resource.labels.namespace_name="careconnect-ai-dev1"
    jsonPayload.event=~"^(agent|safety|ocr|ingest|session)\\..+"
  EOT
  unique_writer_identity = true
  bigquery_options { use_partitioned_tables = true }
}

resource "google_logging_project_sink" "responses_sample" {
  name        = "careconnect-responses-sample"
  destination = "bigquery.googleapis.com/projects/${var.project_id}/datasets/careconnect_logs"
  filter      = <<EOT
    jsonPayload.event="agent.query.completed"
    sample(insertId, 0.01)
  EOT
  unique_writer_identity = true
  bigquery_options { use_partitioned_tables = true }
}
```

> **IAM**: grant the sink's `writer_identity` `roles/bigquery.dataEditor`
> on the dataset only. The `responses_sample` dataset must have CMEK and
> default 30-day expiration.

### 4.5 Dashboards (Looker Studio JSON, exported to repo)

Store JSON exports at:

```
deploy-configs/observability/dashboards/
  ├─ 01_sre_health.json
  ├─ 02_agent_quality.json
  ├─ 03_safety_compliance.json
  └─ 04_cost_capacity.json
```

Recommend one dashboard per stakeholder group so RBAC is simple. See §5
for tile contents.

---

## 5. Dashboard tile catalogue

### 5.1 SRE Health (audience: on-call)

| Tile | Source | Aggregation |
|---|---|---|
| Request rate per endpoint | Cloud Monitoring (Istio) | `rate(5m)` |
| 5xx error rate | `loadbalancing.googleapis.com/https/backend_request_count` | filter `response_code_class=500` / total |
| Latency p50 / p95 / p99 | `query_duration_ms` distribution | `percentile(0.5/0.95/0.99, 5m)` |
| Pod restarts | `kubernetes.io/container/restart_count` | sum 1h |
| Cloud SQL pool usage | `cloudsql.googleapis.com/database/postgresql/num_backends` | last value |
| Vertex AI error rate | `aiplatform.googleapis.com/prediction/online/error_count` | rate(5m) |
| `/health/ready` dep status (table) | log query against `health.ready.dep` event | last 5 min |

### 5.2 Agent Quality (audience: AI eng)

| Tile | Metric | Aggregation |
|---|---|---|
| UNGROUNDED rate | `ungrounded_count` / `query_total_count` | rate ratio (5 m) |
| Empty retrieval rate | `retrieval_empty_count` / `query_total_count` | rate ratio (5 m) |
| No-tool answer rate | `no_tool_answer_count` / `query_total_count` | rate ratio (5 m) |
| Refusal rate | `refused_count` / `query_total_count` | rate ratio (1 h) |
| Tool-loop p95 | `tool_loop_count` distribution | percentile(0.95, 30 m) |
| Tool-call distribution (stacked area) | `query_total_count` with `tool_calls` label | rate(5m) by tool |
| Top sessions by token spend (table) | `input_tokens`+`output_tokens` | sum(24h) by session_id |
| Golden recall@4 7-day trend | BQ `careconnect_eval.golden_runs` | avg by day |

### 5.3 Safety & Compliance (audience: SecOps)

| Tile | Source | Aggregation |
|---|---|---|
| Injection blocks per hour (regex vs LLM) | counters | sum(1h) by mode |
| Auth failures (401) per hour | Istio access log | filter `status=401` |
| DLP findings (must read 0) | BQ `careconnect_logs.dlp_findings` | count(24h) — **anything > 0 is red** |
| Sampled-audit row count (must be ~1 % of total queries) | BQ `responses_sample` | count(24h) |
| Top blocked injection patterns | log query | distinct `pattern` last 7 d |
| Grounding-judge self-failure rate | `grounding_timeout_count` / total | rate ratio (15 m) |

### 5.4 Cost & Capacity (audience: FinOps / Eng Mgr)

| Tile | Source | Aggregation |
|---|---|---|
| Daily Vertex spend (line) | Cloud Billing export → BQ | sum(day) |
| $/1000 queries (gauge + trend) | `cost_usd_est` distribution, `query_total_count` | sum/sum × 1000 |
| Active sessions (line) | `session.registry.size` gauge | last value (5 min) |
| Cloud SQL storage GB | `cloudsql.googleapis.com/database/disk/bytes_used` | last value |
| GCS bucket size GB | `storage.googleapis.com/storage/total_bytes` | last value |
| Ingest throughput (chunks/min) | `ingest.completed.total_chunks_stored` | rate(5m) |

---

## 6. Alert policies — full list

> Convention: PG-* go to PagerDuty `careconnect-ai-oncall` service; TK-*
> create a ServiceNow incident, no page.

### Pages (8)

| ID | Condition (Cloud Monitoring MQL or PromQL-style) | Owner |
|---|---|---|
| PG-1 | 5xx rate > 1 % for 5 min on any route | SRE |
| PG-2 | `/health/ready` returns `degraded` for 5 min (uptime check) | SRE |
| PG-3 | `query_duration_ms` p95 > 8000 for 10 min | SRE |
| PG-4 | `ungrounded_count` / `query_total_count` > 0.05 for 30 min | AI on-call |
| PG-5 | `grounding_timeout_count` / `query_total_count` > 0.03 for 15 min | AI on-call |
| PG-6 | `count(dlp_findings)` > 0 in the last hour | SecOps |
| PG-7 | Auth 401 rate > 20 / 5 min | SecOps |
| PG-8 | Cloud SQL `up` metric = 0 for 5 min | DBA + SRE |

### Tickets (7)

| ID | Condition | Owner |
|---|---|---|
| TK-1 | `retrieval_empty_count` / total > 0.05 over 24 h | AI eng |
| TK-2 | `no_tool_answer_count` / total > 0.02 over 24 h | AI eng |
| TK-3 | `refused_count` / total > 0.15 over 1 h *(over-refusal)* | AI eng |
| TK-4 | `refused_count` / total < 0.005 over 24 h *(under-refusal — sanity check)* | AI eng |
| TK-5 | Daily Vertex AI cost > $ budget (BQ-derived) | FinOps |
| TK-6 | Nightly golden recall@4 drops > 5 pp vs 7-day avg | AI eng |
| TK-7 | `ocr_failure_count` / total > 0.05 for 30 min | Data eng |

---

## 7. Synthetic monitoring (canary session)

Cloud Monitoring native synthetic checks can't drive a multi-step flow,
so we run our own:

```python
# scripts/synthetic_canary.py  (run by Cloud Scheduler every 15 min)
import os, sys, requests, time
BASE = os.environ["CARECONNECT_API_BASE"]
TOKEN = os.environ["API_AUTH_TOKEN"]
CANARY_SESSION = os.environ["CANARY_SESSION_ID"]
CANARY_QUERY = "What is the patient's primary diagnosis?"

def main():
    t0 = time.monotonic()
    r = requests.post(
        f"{BASE}/api/v1/sessions/{CANARY_SESSION}/query",
        headers={"Authorization": f"Bearer {TOKEN}"},
        json={"message": CANARY_QUERY},
        timeout=15,
    )
    elapsed_ms = int((time.monotonic() - t0) * 1000)
    if r.status_code != 200:
        sys.exit(f"FAIL status={r.status_code} body={r.text[:200]}")
    body = r.json()
    grounding = body.get("metadata", {}).get("grounding")
    if grounding == "UNGROUNDED":
        sys.exit(f"FAIL ungrounded answer in {elapsed_ms} ms")
    if elapsed_ms > 8000:
        sys.exit(f"FAIL slow {elapsed_ms} ms")
    print(f"OK {elapsed_ms} ms grounding={grounding}")

if __name__ == "__main__":
    main()
```

Deploy as a Cloud Run Job, schedule via Cloud Scheduler every 15 minutes,
job failure → PG-1 (via Cloud Monitoring alert on Cloud Run Job failure
count).

The canary session is created once at the start of each environment with
a known anonymised PDF in `data/golden/adr/sample_adr_1.pdf`.

---

## 8. Offline validation — golden set & CI gate (Phase 4)

### 8.1 Fixture format

```
data/golden/qa/
  ├─ adr_questions.jsonl        # 30 questions, one per line
  └─ policy_questions.jsonl     # 20 questions, one per line
```

```json
{
  "id": "adr1-q1",
  "kind": "adr",
  "seed_doc": "data/golden/adr/sample_adr_1.pdf",
  "session_seed_uri": "gs://careconnect-eval/golden/adr/sample_adr_1.pdf",
  "question": "What is the patient's primary diagnosis?",
  "expected_keywords": ["myocardial infarction", "STEMI"],
  "min_keywords_matched": 1,
  "expected_source": "sample_adr_1.pdf",
  "expected_page": 3,
  "must_not_refuse": true,
  "max_latency_ms": 5000
}
```

### 8.2 `src/eval/run_eval.py` upgrade

Extend the existing entry point to drive the **full agent**, not just
the retriever:

```python
# src/eval/run_eval.py
import asyncio, json, time, uuid, os
from pathlib import Path

from src.session_manager.initialization import initialize_session
from src.agents.graph import invoke_graph
from src.eval.scoring import (
    compute_keyword_recall,
    compute_source_accuracy,
    compute_latency_stats,
    format_markdown_summary,
)

async def evaluate_question(fx: dict) -> dict:
    sid, _result, mgr = initialize_session(gcs_uri=fx["session_seed_uri"])
    graph = mgr.agent
    t0 = time.monotonic()
    r = await invoke_graph(graph, fx["question"], sid)
    latency_ms = int((time.monotonic() - t0) * 1000)
    ai_msg = next((m for m in reversed(r["messages"]) if getattr(m,"type","")=="ai" and m.content), None)
    answer = ai_msg.content if ai_msg else ""
    tool_msgs = [m for m in r["messages"] if getattr(m,"type","")=="tool"]
    cited = _extract_cited_sources(tool_msgs)
    return {
        "id": fx["id"],
        "answer": answer,
        "cited_sources": cited,
        "latency_ms": latency_ms,
        "refused": answer.startswith("I could not verify"),
        "keyword_recall": compute_keyword_recall(answer, fx["expected_keywords"]),
        "source_correct": compute_source_accuracy(cited, fx["expected_source"]),
        "passed": _check_pass(fx, answer, cited, latency_ms),
    }

def _check_pass(fx, answer, cited, latency_ms) -> bool:
    if fx.get("must_not_refuse") and answer.startswith("I could not verify"):
        return False
    if compute_keyword_recall(answer, fx["expected_keywords"]) < fx.get("min_keywords_matched",1) / max(1, len(fx["expected_keywords"])):
        return False
    if fx["expected_source"] not in cited:
        return False
    if latency_ms > fx.get("max_latency_ms", 5000):
        return False
    return True

async def main(output: str = "json"):
    fixtures = []
    for path in Path("data/golden/qa").glob("*.jsonl"):
        with open(path) as f:
            fixtures.extend(json.loads(line) for line in f if line.strip())
    results = [await evaluate_question(fx) for fx in fixtures]
    summary = {
        "run_id": uuid.uuid4().hex,
        "total": len(results),
        "passed": sum(r["passed"] for r in results),
        "recall_at_k_mean": sum(r["keyword_recall"] for r in results) / len(results),
        "source_accuracy_mean": sum(r["source_correct"] for r in results) / len(results),
        "latency_p95_ms": compute_latency_stats([r["latency_ms"] for r in results])["p95"],
        "results": results,
    }
    if output == "bq":
        _emit_to_bigquery(summary)
    print(format_markdown_summary(summary))
    return 0 if summary["passed"] / summary["total"] >= float(os.environ.get("EVAL_PASS_THRESHOLD", "0.85")) else 1

if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main(output=sys.argv[1] if len(sys.argv) > 1 else "json")))
```

### 8.3 CI gate

```yaml
# .github/workflows/ci.yml — new job
  eval:
    name: golden-eval
    runs-on: self-hosted
    needs: [build]
    env:
      VERTEX_AI_MODE: "real"
      API_AUTH_TOKEN: ${{ secrets.EVAL_TOKEN }}
      EVAL_PASS_THRESHOLD: "0.85"
    steps:
      - uses: actions/checkout@v4
      - name: Auth GCP
        uses: google-github-actions/auth@v2
        with:
          workload_identity_provider: ${{ secrets.WIP }}
          service_account: careconnect-eval@hcb-dev-careconnect-etl.iam.gserviceaccount.com
      - name: Install
        run: pip install -e ".[dev]"
      - name: Run golden eval
        run: python -m src.eval.run_eval json
```

Fail the job (non-zero exit) if pass-rate < 85%. Add a manual override
label `bypass-eval` for emergencies.

### 8.4 Nightly Cloud Run Job

```hcl
resource "google_cloud_run_v2_job" "golden_eval" {
  name     = "careconnect-golden-eval"
  location = "us-east4"
  template {
    template {
      service_account = "careconnect-eval@hcb-dev-careconnect-etl.iam.gserviceaccount.com"
      containers {
        image = "gar-host/careconnect-adr-ai-agent:${var.image_tag}"
        command = ["python","-m","src.eval.run_eval","bq"]
        env { name = "VERTEX_AI_MODE" value = "real" }
        env { name = "EVAL_OUTPUT_DATASET" value = "careconnect_eval.golden_runs" }
      }
      timeout = "1800s"
    }
  }
}

resource "google_cloud_scheduler_job" "golden_eval" {
  name      = "careconnect-golden-eval-nightly"
  schedule  = "0 3 * * *"      # 03:00 ET
  time_zone = "America/New_York"
  http_target {
    uri         = "https://${var.region}-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/${var.project_id}/jobs/careconnect-golden-eval:run"
    http_method = "POST"
    oidc_token { service_account_email = "careconnect-scheduler@..." }
  }
}
```

`careconnect_eval.golden_runs` schema:

| Column | Type |
|---|---|
| `run_id` | STRING |
| `ts` | TIMESTAMP |
| `total` | INT |
| `passed` | INT |
| `recall_at_k_mean` | FLOAT64 |
| `source_accuracy_mean` | FLOAT64 |
| `latency_p95_ms` | INT |
| `results` | RECORD REPEATED (id, answer, latency_ms, refused, keyword_recall, source_correct, passed) |

---

## 9. PHI audit (Phase 5)

### 9.1 Sampled audit sink

Already specified in §4.4 (`responses_sample` sink). Sample rate is set
via the Cloud Logging `sample()` filter — change to `0.05` for staging,
`0.01` for prod.

### 9.2 Cloud DLP inspect template

```hcl
resource "google_data_loss_prevention_inspect_template" "phi" {
  parent       = "projects/${var.project_id}"
  display_name = "careconnect-phi"

  inspect_config {
    info_types { name = "US_SOCIAL_SECURITY_NUMBER" }
    info_types { name = "PHONE_NUMBER" }
    info_types { name = "EMAIL_ADDRESS" }
    info_types { name = "DATE_OF_BIRTH" }
    info_types { name = "US_HEALTHCARE_NPI" }
    info_types { name = "MEDICAL_RECORD_NUMBER" }
    info_types { name = "PERSON_NAME" }
    info_types { name = "US_DRIVERS_LICENSE_NUMBER" }
    min_likelihood = "LIKELY"
    include_quote  = false                      # do NOT store the matched text
    limits { max_findings_per_request = 100 }
  }
}
```

### 9.3 Nightly DLP workflow

```yaml
# deploy-configs/observability/workflows/dlp_scan.yaml
main:
  steps:
    - inspect:
        call: googleapis.dlp.v2.projects.dlpJobs.create
        args:
          parent: projects/${project_id}
          body:
            inspectJob:
              inspectTemplateName: projects/${project_id}/inspectTemplates/careconnect-phi
              storageConfig:
                bigQueryOptions:
                  tableReference:
                    projectId: ${project_id}
                    datasetId: careconnect_logs
                    tableId: responses_sample
                  rowsLimit: 10000
                  sampleMethod: RANDOM_START
                  identifyingFields:
                    - name: insertId
              actions:
                - saveFindings:
                    outputConfig:
                      table:
                        projectId: ${project_id}
                        datasetId: careconnect_logs
                        tableId: dlp_findings
                      outputSchema: BIG_QUERY_COLUMNS
```

Scheduler triggers at 03:30 ET nightly (after the eval run).

### 9.4 Alert on any finding

```hcl
resource "google_monitoring_alert_policy" "dlp_hit" {
  display_name = "PG-6 — PHI detected in sampled responses"
  conditions {
    display_name = "any DLP finding in last hour"
    condition_threshold {
      filter          = "resource.type=\"bigquery_dataset\" metric.type=\"logging.googleapis.com/user/careconnect/dlp/finding_count\""
      comparison      = "COMPARISON_GT"
      threshold_value = 0
      duration        = "0s"
    }
  }
  notification_channels = [google_monitoring_notification_channel.pagerduty_secops.id]
}
```

`dlp_finding_count` is a log-based counter against the `dlp_findings`
BQ table (Workflows logs a structured event `dlp.scan.completed` with
`findings_count: N`).

---

## 10. Runbook template

One file per page-grade alert under `docs/runbooks/`.

```markdown
# Runbook — PG-4: UNGROUNDED rate > 5%

**Severity**: P1 (page)
**Owner**: AI on-call

## What it means
The grounding judge has flagged more than 5% of agent answers as
UNGROUNDED in the last 30 minutes. The agent is producing answers
unsupported by the retrieved documents.

## First 3 things to check (≤ 5 minutes)
1. Cloud Trace: open the **Agent Quality** dashboard → click any UNGROUNDED
   sample. Look at the tool-call span. Did the agent call `adr_search`?
2. Run the canary: `gcloud run jobs execute careconnect-synthetic-canary --wait`.
   If the canary fails, this is system-wide; escalate to SRE.
3. Check the latest deploy: `git log -10 --oneline`. If a deploy happened
   in the last 2 hours, suspect prompt or system-message regression.

## Common causes & remediations
| Cause | Signal | Fix |
|---|---|---|
| Vertex AI latency spike → grounding judge timing out | TIMEOUT verdict count is up too | Verify §`grounding_timeout_count` alert; raise judge timeout temporarily via env. |
| New prompt regression | UNGROUNDED rate jumps right after a deploy | Roll back via `kubectl rollout undo`. |
| Document set degraded (corrupt OCR) | Empty-retrieval rate also up | Investigate OCR pipeline (PG / TK-7). |

## How to silence for a maintenance window
`gcloud alpha monitoring policies update ${POLICY_ID} --suppress-for=4h`

## Escalation
If unresolved after 30 min, escalate to AI Engineering Lead (Nathan)
and Engine team on-call (Farhan's team).
```

Replicate for PG-1 through PG-8.

---

## 11. Acceptance criteria (definition of done — v1)

- [ ] **Phase 0**: requests have `traceparent` headers; Cloud Trace shows
      spans for `agent.invoke` and `agent.grounding_judge`.
- [ ] **Phase 0**: a sample log line in Cloud Logging shows
      `severity`, `event`, `request_id`, `session_id` as top-level fields.
- [ ] **Phase 1**: every 200 response from `/query` and `/widget/v1/chat/query`
      is followed within 1 s by an `agent.query.completed` log entry.
- [ ] **Phase 2**: 20 log-based metrics visible in Metrics Explorer; 4
      dashboards saved and linkable.
- [ ] **Phase 2**: 3 uptime checks green for 24 h.
- [ ] **Phase 3**: 8 page policies + 7 ticket policies created. PG-1 chaos
      test (artificial 5xx via a feature flag) fires PagerDuty within 5
      min in the staging project.
- [ ] **Phase 4**: 50 fixtures merged; `eval` CI job green on `main`;
      nightly job has run successfully for 7 consecutive nights; BQ table
      `careconnect_eval.golden_runs` has rows.
- [ ] **Phase 5**: `responses_sample` table has rows for 14 consecutive
      days; DLP scan job has run 14 nights; **zero** findings; PG-6
      verified via a deliberate test PHI insertion in staging only.
- [ ] All 8 runbooks merged under `docs/runbooks/`.

---

## 12. Risks & mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Logging cost overrun (high-volume `agent.query.completed`) | Med | Cap retention at 30 d in `_Default` bucket; only `responses_sample` (1%) goes to BQ |
| Alert noise during first weeks | High | All new alerts deploy as **ticket-only** for first 2 weeks; promote to page after baseline |
| Golden-set bias (only 3 documents → 50 questions = repetition) | Med | Plan v2 corpus expansion at end of Phase 4 |
| DLP false positives on names that look like PHI but aren't | Med | Tune `min_likelihood = LIKELY` (not `POSSIBLE`); start in staging first |
| Cloud Trace sampling drops critical spans | Low | Set `OTEL_TRACES_SAMPLER=parentbased_traceidratio`, ratio `1.0` in prod (cost is acceptable at our volume) |
| Synthetic canary itself goes stale | Low | Refresh canary session weekly via Cloud Scheduler |

---

## 13. Out of scope for v1 (tracked for v2)

| Item | Why deferred | Trigger to reopen |
|---|---|---|
| Per-token TTFB / streaming metrics | E2E latency is the actionable signal | If users start reporting slow streaming feel |
| Embedding-drift detection | Need 3 months of baseline first | After Phase 4 has run for a quarter |
| Active learning from thumbs-down | Out of monitoring scope | Separate fine-tuning workstream |
| Multi-region eval | Single-region prod | When prod goes multi-region |
| Adversarial red-team suite | Needs Sec workstream | After Phase 5 stable |
| Replace regex PHI redactor with Cloud DLP de-id | Bigger workstream | If Phase 5 surfaces > 0 findings in 30 d |

---

## 14. Reading list for incoming engineers

1. `docs/architecture/PROJECT_ANALYSIS.md` (this repo)
2. `src/agents/graph.py` (esp. `_create_grounding_gate`, `stream_graph`)
3. `src/api/validation/{grounding_judge,input_safety,phi_redaction}.py`
4. `src/eval/{run_eval,scoring}.py`
5. [Cloud Logging log-based metrics docs](https://cloud.google.com/logging/docs/logs-based-metrics)
6. [Cloud DLP for HIPAA-covered workloads](https://cloud.google.com/dlp/docs/dlp-bigquery)
7. [OpenTelemetry Python on GKE → Cloud Trace](https://cloud.google.com/trace/docs/setup/python-ot)

---

## 15. Quick reference — file diff index

| File | Change | Phase |
|---|---|---|
| `src/core/logger.py` | Add `JsonFormatter`, `LOG_FORMAT` env switch | 0 |
| `src/api/app.py` | Add `_wire_otel()`; add `session.registry.size` periodic emit in lifespan | 0 |
| `src/api/middleware/observability.py` | Store `request_id` on `request.state` | 0 |
| `src/api/routes/_metrics.py` *(new)* | `_count_loops`, `_was_retrieval_empty`, `_estimate_cost` | 1 |
| `src/api/routes/query.py` | Emit `agent.query.completed` after grounding judge | 1 |
| `src/api/routes/widget.py` | Same as above | 1 |
| `src/agents/graph.py` | Emit `agent.query.completed` in `stream_graph` `done` event | 1 |
| `src/api/validation/input_safety.py` | Add `request_id`, `session_id` params + `safety.injection.*` events | 1 |
| `src/api/validation/grounding_judge.py` | Return `"TIMEOUT"` on timeout (distinct from `"PARTIAL"`) | 1 |
| `src/ocr/ocr_orchestrator.py` | Emit `ocr.completed` event | 1 |
| `src/adr_vector_database/ingestion_pipeline.py` | Emit `ingest.completed` event | 1 |
| `src/api/routes/sessions.py` | Emit `session.create.completed` event | 1 |
| `src/session_manager/warmup.py` | Promote existing log to structured `session.warmup.completed` event | 1 |
| `src/eval/run_eval.py` | Drive full agent; emit BQ rows; honour `EVAL_PASS_THRESHOLD` | 4 |
| `data/golden/qa/adr_questions.jsonl` *(new)* | 30 questions | 4 |
| `data/golden/qa/policy_questions.jsonl` *(new)* | 20 questions | 4 |
| `scripts/synthetic_canary.py` *(new)* | Canary check used by Cloud Run Job | 2 |
| `.github/workflows/ci.yml` | Add `eval` job | 4 |
| `deploy-configs/observability/*.tf` *(new)* | Terraform module: metrics, alerts, sinks, uptime, DLP, scheduler, dashboards | 2-5 |
| `deploy-configs/observability/runbooks/*.md` *(new)* | 8 runbooks | 3 |
| `deploy-configs/careconnect-ai-dev1/values/v1.yaml` | `LOG_FORMAT=json`, `OTEL_ENABLED=1`, OTLP endpoint, pricing env | 0 |
