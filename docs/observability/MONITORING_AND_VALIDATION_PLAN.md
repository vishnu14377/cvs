# Monitoring & Validation Plan — Care Connect ADR AI Agent

> **Audience**: Engineering Lead, AI Engineering, SRE, Compliance.
> **Scope**: Production deployment on GKE (`careconnect-ai-dev1`/QA2 today,
> prod after Stargate CD validation). Medical-claims domain → HIPAA-relevant.
> **Goal of v1**: a *basic-but-complete* observability + validation layer that
> is buildable in **3 weeks** and answers four questions on demand:
> 1. *Is the system up and fast?*
> 2. *Are answers grounded, safe, and PHI-clean?*
> 3. *Did anything regress today vs the golden set?*
> 4. *What did it cost?*

---

## 0. Non-goals (explicit, to keep v1 small)

- **No** custom ML drift detector. We rely on golden-set regression + sampled
  human review.
- **No** Prometheus/Grafana stack. We use **Cloud Monitoring + Cloud Logging
  + BigQuery + Looker Studio** — all GCP-managed, zero new infra.
- **No** automated PHI redaction *replacement*. We *validate* the existing
  regex redactor; replacing it is a separate workstream.
- **No** A/B testing harness. Single-variant prod for now.

---

## 1. Design principles

| Principle | Implication |
|---|---|
| **Don't ship new infra unless you have to** | OpenTelemetry, Cloud Logging, log-based metrics, Cloud Monitoring, BigQuery → all already in our org. Re-use. |
| **Every log line is a metric in waiting** | We emit one structured JSON log per request (already do) + one per agent invocation. Cloud Logging converts to metrics. |
| **PHI never leaves the cluster un-redacted** | Logs and BigQuery sinks consume **only** the already-redacted external surfaces. Internal agent context stays in-pod. |
| **Online cheap, offline thorough** | Per-request grounding judge is a single Gemini call (online). Heavy quality eval runs on a golden set, nightly + per-PR (offline). |
| **Fail open on observability** | A logging or metric error must never break a clinical query. All emitters are best-effort. |

---

## 2. Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                       GKE — careconnect-adr-ai-agent                     │
│                                                                          │
│  FastAPI ─► ObservabilityMiddleware ─► structured JSON logs (stdout)     │
│      │                                                                   │
│      ├─► OpenTelemetry SDK ─► OTLP exporter ─► Cloud Trace               │
│      │                                                                   │
│      └─► explicit metric events  (grounding, tools, tokens, refusals)    │
└──────────────────┬───────────────────────────────────────────────────────┘
                   │ stdout
                   ▼
        ┌────────────────────────┐
        │     Cloud Logging      │  ── log router ──► BigQuery (`careconnect_logs.*`)
        │  (HIPAA-compliant)     │  ── log-based metrics ──► Cloud Monitoring
        └──────────┬─────────────┘
                   │
                   ▼
        ┌────────────────────────┐
        │   Cloud Monitoring     │ ── Alerting Policy ──► PagerDuty / Slack / e-mail
        │  metrics + dashboards  │
        └──────────┬─────────────┘
                   │
                   ▼
        ┌────────────────────────┐         ┌──────────────────────────────┐
        │   Looker Studio        │         │ Cloud Scheduler ──► Cloud Run│
        │   exec / SRE dashboard │         │ "golden_eval" job  (nightly)│
        └────────────────────────┘         │   pytest src/eval ──► BQ    │
                                           └──────────────────────────────┘

         ┌────────────────────────────────────────────────────────┐
         │     Cloud Monitoring Uptime Checks                     │
         │     - GET /health                       (1 min)        │
         │     - GET /health/ready                 (5 min)        │
         │     - synthetic query against canary session (15 min)  │
         └────────────────────────────────────────────────────────┘
```

**Why these GCP services**:
- **Cloud Logging** — already receives stdout from GKE pods; has a **HIPAA BAA**.
- **Cloud Monitoring + log-based metrics** — zero-config conversion from
  structured fields (`grounding_verdict`, `tool_name`, `latency_ms`, …)
  to metrics. No StatsD/Prometheus to operate.
- **Cloud Trace** — `opentelemetry-instrumentation-fastapi` is already
  in `requirements.txt`; flip one exporter env var.
- **BigQuery** — sink target for log retention (90 days), the nightly
  golden eval rows, and sampled query audit (1 % sample).
- **Looker Studio** — free, native BigQuery connector; non-eng-friendly.
- **Cloud Scheduler + Cloud Run Job** — nightly eval runner.

---

## 3. The metric catalog (basic v1)

> **Convention**: every metric below maps either to a **GKE/Cloud SQL
> built-in metric** (free, no code change) or to a **log-based metric**
> derived from a single new structured log line. Nothing here requires
> a separate metrics pipeline.

### 3.1 RED metrics — service health  ⚙️ *infra-built-in*

| # | Metric | Source | Threshold (page) |
|---|---|---|---|
| H-1 | Request **rate** per endpoint | GKE / Istio | n/a (capacity planning) |
| H-2 | Request **error rate** (5xx) | GKE / Istio | > 1 % over 5 min |
| H-3 | Request **duration** p50/p95/p99 per endpoint | GKE / Istio | p95 `/query` > 8 s for 10 min |
| H-4 | Pod CPU / memory / restart count | GKE | > 3 restarts in 10 min |
| H-5 | HPA scaling events | GKE | informational |
| H-6 | Cloud SQL connection pool exhaustion | Cloud SQL Insights | > 80 % pool utilisation 5 min |
| H-7 | Cloud SQL query p95 | Cloud SQL Insights | > 500 ms 10 min |
| H-8 | GCS 4xx / 5xx | Cloud Monitoring | > 0.5 % 5 min |
| H-9 | Vertex AI call latency / error rate | Cloud Monitoring (Vertex) | error > 2 % 5 min |

> All nine are out-of-the-box on GCP. Zero code change. **Day-1 dashboard.**

### 3.2 Agent quality metrics  🧠 *one new log line per query*

Emitted once per `invoke_graph` call from `routes/query.py` and `routes/widget.py`:

```python
logger.info("agent.query.completed", extra={
    "session_id": session_id,                     # PII-safe identifier
    "request_id": request_id,
    "duration_ms": elapsed_ms,
    "tool_calls": ["adr_search"],                 # list[str]
    "tool_loop_count": 1,                         # int (how many ToolNode iters)
    "retrieval_empty": False,                     # bool — any tool returned 0 docs
    "grounding_verdict": "GROUNDED",              # GROUNDED|PARTIAL|UNGROUNDED|TIMEOUT
    "refused": False,                             # True if final content is the canned refusal
    "input_tokens": 412,
    "output_tokens": 187,
    "cost_usd_est": 0.00046,                      # derived
    "endpoint": "/widget/v1/chat/query",
})
```

| # | Metric (log-based) | Threshold (page) | Threshold (ticket) |
|---|---|---|---|
| Q-1 | **UNGROUNDED rate** = count(`grounding_verdict="UNGROUNDED"`) / count(*) | > 5 % over 30 min | > 2 % over 24 h |
| Q-2 | **Empty retrieval rate** = count(`retrieval_empty=True`) / count(*) | > 10 % over 30 min | > 5 % over 24 h |
| Q-3 | **No-tool answer rate** = count(`tool_calls=[]`) / count(*)  *(captures `GROUNDING_ALERT`)* | > 5 % over 30 min | > 2 % over 24 h |
| Q-4 | **Tool-loop p95** | > 4 over 30 min (indicates the LLM is thrashing) | — |
| Q-5 | **Refusal rate** = count(`refused=True`) / count(*) | > 15 % over 1 h (over-refusal) | < 0.5 % (under-refusal) over 24 h |
| Q-6 | **Grounding-judge timeout rate** = count(`grounding_verdict="TIMEOUT"`) / count(*) | > 3 % over 15 min | > 1 % over 24 h |
| Q-7 | **Per-query latency** p95 (end-to-end agent path) | > 8 s over 10 min | > 5 s over 24 h |
| Q-8 | **Input + output tokens** p95 | > 12 k over 24 h | — |
| Q-9 | **Cost / 1000 queries** (derived) | informational | dollar-budget alert |

### 3.3 Safety & compliance metrics  🔒 *critical for HIPAA*

Emitted from `validation/input_safety.py`, `validation/grounding_judge.py`,
and a new sampled audit hook.

| # | Metric | Source | Threshold |
|---|---|---|---|
| S-1 | **Prompt-injection blocks (regex)** | log event `safety.injection.regex` | > 10 / hour → SecOps ticket (recon attempt) |
| S-2 | **Prompt-injection blocks (LLM classifier)** | log event `safety.injection.llm` | > 5 / hour → SecOps ticket |
| S-3 | **Auth failures (401)** rate | Istio access logs | > 20 / 5 min → page (credential leak / brute force) |
| S-4 | **PHI-leak audit hits** | nightly Cloud DLP scan of `responses_sample` BQ table | **any** hit → page |
| S-5 | **Sampled query audit volume** | rows / day in `responses_sample` | < expected 1 % → ticket (sampler broken) |
| S-6 | **Sessions without `careconnect-widget-token`** | log filter | > 0 → SecOps ticket |
| S-7 | **Grounding-judge failure rate** (the judge itself errored) | log event | > 3 % over 30 min → page |

### 3.4 Pipeline metrics  📥 *ingest path*

| # | Metric | Source | Threshold |
|---|---|---|---|
| P-1 | Session-creation **success rate** | log event `session.create.completed` | < 98 % over 15 min → page |
| P-2 | **OCR latency** p95 per document | log event `ocr.completed` | > 60 s for 30 min → ticket |
| P-3 | **OCR failure rate** | log event | > 5 % over 30 min → page |
| P-4 | **Ingestion failure rate** (chunks written / chunks attempted) | log event | > 2 % over 1 h → page |
| P-5 | **Session warmup failure rate** (already non-fatal) | log event `session_manager/warmup` | > 10 % over 1 h → ticket |

### 3.5 Capacity / cost  💰

| # | Metric | Source | Threshold |
|---|---|---|---|
| C-1 | Active sessions (size of `_session_registry`) | log event `session.registry.size` (5-min poll) | > 80 % of memory budget → ticket |
| C-2 | Cloud SQL storage growth | Cloud SQL metric | budget-based |
| C-3 | GCS bucket size | GCS metric | budget-based |
| C-4 | Daily $ spent on Vertex AI | Cloud Billing export → BQ | > daily budget → ticket |

> **Total metric count: ~35.** All but ~10 are zero-code (GCP built-in).
> The remaining ~10 come from **3 new structured log events** plus a 1 %
> sampler.

---

## 4. Validation layer

### 4.1 Online (per-request) — already in code, just instrument

| Layer | Already does | What we add |
|---|---|---|
| Input safety regex | blocks `[SYSTEM]`, `ignore previous instructions`, … | log event `safety.injection.regex` |
| Input safety LLM classifier | 3 s LLM call, fail-open SAFE | log event `safety.injection.llm` |
| `grounding_gate` (observational) | `GROUNDING_ALERT` warning | already a log; promote to metric |
| `judge_grounding` | LLM verdict GROUNDED/PARTIAL/UNGROUNDED | promote verdict to a log-based metric |
| `redact_phi` | regex sub on external surfaces | nightly **DLP scan** of sampled responses to catch misses |

### 4.2 Offline — golden set regression  ✅ *non-negotiable for medical*

We **already have** the building blocks:
- `data/golden/adr/sample_adr_{1,2,3}.pdf` and `data/golden/policy/sample_policy_{1,2,3}.pdf`
- `src/eval/run_eval.py` + `src/eval/scoring.py` (recall, citation, latency, markdown summary)
- Unit tests for the scorers (already pass post-reorg)

What's missing — and what we build:

1. **Question / expected-answer JSONL** under `data/golden/qa/`:
   ```json
   {"id": "adr1-q1", "session_seed": "adr_sample_1", "question": "What is the patient's primary diagnosis?", "expected_keywords": ["myocardial infarction", "STEMI"], "expected_source": "sample_adr_1.pdf", "expected_page": 3}
   ```
   Initial set: **30 ADR questions + 20 policy questions = 50 fixtures.**
   This is the smallest set that gives statistically meaningful recall numbers.

2. **`src/eval/run_eval.py` upgrade** (already exists — extend it):
   - Spin up a session per seed document via `initialize_session`
   - For each question: invoke the agent, score against the fixture
   - Emit `eval.result` log line and a `BatchEvalResult` JSON

3. **CI gate** (`.github/workflows/ci.yml` job `eval`):
   ```
   pytest -m eval  → fails PR if:
     - recall@4 < 0.70
     - citation_accuracy < 0.80
     - groundedness_judge >= PARTIAL on >= 90% of questions
     - p95 latency > 5 s
   ```

4. **Nightly Cloud Run Job** writes the same eval rows to BigQuery
   table `careconnect_eval.golden_runs` → Looker dashboard. Detects
   regressions caused by **model drift** even when the code didn't
   change.

### 4.3 Offline — sampled production audit  🔍

- 1 % of **redacted** `agent.query.completed` rows are tee-d to
  `careconnect_logs.responses_sample` (Cloud Logging sink + filter).
- Nightly **Cloud DLP** job scans `content` and `content_html` columns
  with InfoTypes: `US_SOCIAL_SECURITY_NUMBER`, `PHONE_NUMBER`,
  `EMAIL_ADDRESS`, `DATE_OF_BIRTH`, `US_HEALTHCARE_NPI`,
  `MEDICAL_RECORD_NUMBER`, `PERSON_NAME`.
- **Any hit ⇒ pager + automatic ticket + quarantine row in a
  restricted BQ dataset.**
- Weekly: compliance reviewer reads 20 random sampled rows + all
  thumbs-down feedback rows.

### 4.4 Offline — feedback loop

- `feedback_manager` already stores thumbs-up/down + redacted
  user-message + redacted AI-response in MongoDB.
- BQ federated query (or daily batch export) into
  `careconnect_eval.feedback`.
- Looker tile: % thumbs-down per day, top tools-used on thumbs-down,
  worst-rated documents.

---

## 5. Dashboards (Looker Studio, 4 tabs)

| Tab | Audience | Key tiles |
|---|---|---|
| **1. SRE Health** | On-call | RED metrics, pod restarts, dependency health (`/health/ready` breakdown), uptime check status |
| **2. Agent Quality** | AI eng | UNGROUNDED %, empty-retrieval %, no-tool-answer %, refusal %, latency p50/p95, tool-call distribution |
| **3. Safety & Compliance** | SecOps / Compliance | Injection blocks per hour, auth failures, DLP hits (must be 0), sampled-audit volume, top blocked patterns |
| **4. Cost & Capacity** | Eng manager / FinOps | $/day, $/1000 queries, active sessions, ingest throughput, Cloud SQL storage growth |

Each tab has the standard **time-range selector**, **environment
filter** (dev1 / QA2 / prod), and a **release-marker** overlay
(from the GHA workflow that publishes deploy events).

---

## 6. Alerts (Cloud Monitoring policies)

> Rule of thumb: **paging alerts** wake humans up; **ticketing alerts**
> create a Jira/ServiceNow item. Don't mix.

### 6.1 Pages (PagerDuty, on-call rotation)

| ID | Condition | Owner |
|---|---|---|
| PG-1 | 5xx error rate > 1 % for 5 min | SRE |
| PG-2 | `/health/ready` returns `degraded` for 5 min | SRE |
| PG-3 | p95 `/query` latency > 8 s for 10 min | SRE |
| PG-4 | UNGROUNDED rate > 5 % for 30 min | AI on-call |
| PG-5 | Grounding-judge timeout > 3 % for 15 min | AI on-call |
| PG-6 | DLP hit on sampled responses (count > 0 in last hour) | SecOps |
| PG-7 | Auth failure rate > 20 / 5 min | SecOps |
| PG-8 | Cloud SQL down (engine `SELECT 1` fails on `/health/ready`) | DBA + SRE |

### 6.2 Tickets (ServiceNow/Jira)

| ID | Condition | Owner |
|---|---|---|
| TK-1 | Empty-retrieval > 5 % over 24 h | AI eng |
| TK-2 | No-tool answer > 2 % over 24 h | AI eng |
| TK-3 | Refusal rate > 15 % over 1 h *(over-refusal — tune grounding judge)* | AI eng |
| TK-4 | Refusal rate < 0.5 % over 24 h *(under-refusal — verify judge fired)* | AI eng |
| TK-5 | Daily Vertex AI spend > budget | FinOps |
| TK-6 | Golden-set nightly: recall@4 drop > 5 pp vs 7-day avg | AI eng |
| TK-7 | OCR failure rate > 5 % for 30 min | Data eng |

---

## 7. Implementation plan

> **Effort estimates assume 1 backend engineer + 0.5 SRE.**
> Each phase is independently shippable.

### Phase 0 — wire what's already there  *(2 days)*

| Step | File | Change |
|---|---|---|
| 0.1 | `src/api/app.py` | Add `FastAPIInstrumentor.instrument_app(app)` + `OTLPSpanExporter` pointing at Cloud Trace via `CLOUD_TRACE_*` env. |
| 0.2 | `src/core/logger.py` | Switch to JSON formatter (one line per event) so Cloud Logging auto-parses. |
| 0.3 | `deploy-configs/.../v1.yaml` | Set `OTEL_EXPORTER_OTLP_ENDPOINT`, `LOG_FORMAT=json`. |
| 0.4 | `src/api/middleware/observability.py` | Add `request_id` to every log line in scope (already partially done). |

**Deliverable**: every request shows up as a span in Cloud Trace and a
JSON log line in Cloud Logging.

### Phase 1 — emit the 3 new event logs  *(2 days)*

| Event | Where to emit | Fields |
|---|---|---|
| `agent.query.completed` | `routes/query.py`, `routes/widget.py`, end of handler | see §3.2 |
| `safety.injection.{regex,llm}` | `validation/input_safety.py` | `pattern`, `mode`, `session_id`, `request_id` |
| `ocr.completed` / `ingest.completed` / `session.create.completed` | the corresponding pipelines | `success`, `duration_ms`, `pages`, `chunks` |

**Deliverable**: every business metric in §3.2–§3.4 is now derivable
from logs.

### Phase 2 — log-based metrics + dashboards  *(3 days)*

1. Define 20 log-based **counter** + **distribution** metrics in
   Cloud Monitoring (Terraform module).
2. Build the 4 Looker Studio tabs.
3. Wire Cloud Monitoring **Uptime Checks** (`/health`,
   `/health/ready`, synthetic query against a canary session).

**Deliverable**: dashboards live.

### Phase 3 — alerts  *(2 days)*

1. Create 8 page policies + 7 ticket policies (§6) as Terraform.
2. Route page policies to PagerDuty service, tickets to ServiceNow
   webhook.
3. Add **runbooks** in `docs/runbooks/` for each PG-* alert — minimum
   "What does this mean / first 3 things to check / how to silence
   for a maintenance window".

**Deliverable**: someone pages **someone** when things go wrong.

### Phase 4 — offline validation  *(1 week)*

1. Author 50 golden Q/A JSONL fixtures (`data/golden/qa/*.jsonl`).
2. Extend `src/eval/run_eval.py` to drive the full agent (not just
   the retriever) and write a JSON report.
3. New GHA job `eval`: runs on every PR, blocks merge on threshold
   miss.
4. Cloud Run Job (containerised same image, command
   `python -m src.eval.run_eval --output bq`) scheduled nightly via
   Cloud Scheduler.
5. BigQuery table `careconnect_eval.golden_runs` + Looker tile.

**Deliverable**: regressions caught **before** they reach prod.

### Phase 5 — PHI audit  *(1 week)*

1. Log sink: 1 % sample of `agent.query.completed` with content fields
   → BQ `careconnect_logs.responses_sample` (retention 30 days,
   restricted IAM).
2. Cloud DLP inspection template (InfoTypes listed in §4.3).
3. Cloud Scheduler → Cloud Workflows: nightly DLP scan job; results
   to `careconnect_logs.dlp_findings`.
4. Alert policy PG-6 on any DLP finding.
5. Weekly human review — Looker board sorted by thumbs-down +
   sampled rows.

**Deliverable**: HIPAA-defensible audit trail.

---

## 8. What we explicitly defer (v2)

| Defer | Why | When to revisit |
|---|---|---|
| Per-token streaming metrics (TTFB, tokens/s) | Marginal value; expensive | When p95 latency stops being end-to-end-sufficient |
| Vector store drift detection (embedding distribution shift) | No clear actionable threshold yet | Once we have 3 months of golden-eval baseline |
| Active learning loop from thumbs-down → fine-tuning | Out of scope for monitoring | After Gemini fine-tuning approval |
| Multi-region eval | Single-region prod for now | When we go multi-region |
| Synthetic adversarial red-team suite | Need separate Sec workstream | After Phase 5 |

---

## 9. HIPAA-specific call-outs (read these before approval)

1. **No PHI in metrics.** Log-based metrics derive only from
   *non-PHI* fields (verdict, tokens, latency, tool_calls). Content
   strings never become metric labels.
2. **Cloud Logging is HIPAA-eligible** under the org-level BAA — but
   only `_Required` and `_Default` log buckets in the `hcb-dev-*`
   project. We will *not* sink logs to non-HIPAA destinations.
3. **BigQuery dataset `careconnect_logs.responses_sample`** is
   IAM-restricted (Compliance + SecOps groups only) and has CMEK
   enabled.
4. **Cloud DLP findings are themselves PHI** — they live in a
   separate restricted dataset with 30-day TTL, not in monitoring
   dashboards.
5. **Audit-log everything**: enable Cloud Audit Logs for the GKE
   namespace, Cloud SQL instance, GCS bucket, Secret Manager
   secrets — already on by default; verify with InfoSec.
6. **PagerDuty payload** must not include PHI. Alerts reference
   `request_id` / `session_id` only — engineer pivots to logs in
   Cloud Console.

---

## 10. Acceptance criteria for v1 (definition of done)

- [ ] All 35 metrics emit values in Cloud Monitoring.
- [ ] 4 dashboards live, viewable by Eng + SecOps + FinOps.
- [ ] 15 alert policies firing in test (chaos-injection or synthetic).
- [ ] 8 runbooks merged under `docs/runbooks/`.
- [ ] 50-question golden Q/A fixture merged.
- [ ] CI `eval` job green on `main`, blocks on regression.
- [ ] Nightly eval job + Looker tile live for 14 consecutive days.
- [ ] DLP scan job ran ≥ 14 nights with **zero** findings.
- [ ] On-call rotation defined; PG-1 to PG-8 have an owner.

---

## 11. Open decisions for the Eng Lead

1. **PagerDuty service** — reuse the existing CDV on-call or stand up a
   dedicated CareConnect-AI service?
2. **Eval budget** — nightly runs at ~50 questions × Gemini Flash ≈
   $0.30/night. OK to proceed unbudgeted?
3. **Thresholds in §3 are starting values** — we baseline them for
   2 weeks then tune. Acceptable?
4. **Golden Q/A authorship** — who writes the 50 fixtures? AI eng
   alone, or with a clinical reviewer? (Recommend: clinical reviewer
   approves expected answers, AI eng writes the JSONL.)
5. **CMEK on the sample BQ dataset** — confirm key ownership with InfoSec.
