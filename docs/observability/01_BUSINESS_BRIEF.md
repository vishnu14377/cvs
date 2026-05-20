# Monitoring & Validation for the ADR AI Agent
## A 3-Week Plan to Make Our Medical-Claims AI Safe, Reliable, and Audit-Ready

> **For**: Business stakeholders, Product, Compliance, Engineering Management.
> **From**: AI/ML Engineering.
> **Length**: 5-minute read.

---

## 1. The situation in plain English

We've shipped an AI assistant that helps our clinical reviewers read patient
records. Every day it reads hundreds of pages of medical documents, answers
questions about them, and shapes how our reviewers process insurance claims.

That power comes with three risks we have to actively manage:

| Risk | What it looks like in practice | Why it matters to the business |
|---|---|---|
| **The AI invents an answer** | It "hallucinates" a diagnosis that isn't in the chart | Reviewer makes the wrong claim decision → financial loss + regulatory exposure |
| **The AI leaks patient information** | A patient name, SSN, or MRN ends up in a log file or response we didn't intend | HIPAA violation → potential fines, brand damage, mandatory breach disclosure |
| **The AI gets slow or breaks** | Pages take 30 seconds to load, or the chatbox times out | Reviewers fall back to manual reading → throughput drops, SLAs miss |

Right now, **we have very little visibility into whether any of these are
happening in production**. We need that visibility before we scale usage.

This document explains, in business terms, **what we will measure, why each
measurement matters, what it costs to set up, and what we need from you.**

---

## 2. What we will measure

We've grouped 35 measurements into four buckets, ordered by business
importance. Each is one paragraph, no jargon.

### Bucket A — *"Is the AI giving correct, safe answers?"*  🧠

This is the most important bucket. These tell us if the AI is doing its job
well or quietly failing.

| Measurement | What it answers | Why it matters |
|---|---|---|
| **Hallucination rate** | "Out of every 100 answers, how many contained a claim the AI *made up* — i.e., not supported by the document?" | Direct measure of AI safety. Target: **under 2%**. |
| **Empty-search rate** | "How often does the AI search the document and find nothing?" | A high rate means either the document is bad or the AI is asking the wrong questions. |
| **Skipped-search rate** | "How often does the AI answer *without* checking the document first?" | The AI is supposed to always search before answering. This catches when it cuts corners. |
| **Refusal rate** | "How often does the AI refuse to answer?" | Too high → frustrating for reviewers. Too low → safety check has stopped working. |
| **Citation accuracy** | "When the AI cites 'page 7,' does page 7 actually contain that information?" | Reviewers rely on citations to verify. Wrong citations destroy trust. |

### Bucket B — *"Are we leaking patient data?"*  🔒

This is the **HIPAA bucket**. We treat every measurement here as a fire alarm.

| Measurement | What it answers | Why it matters |
|---|---|---|
| **PHI leak detector** | "Did any patient identifier (name, SSN, MRN, DOB) slip past our redactor into a log or response?" | This is our HIPAA defense. **Any positive result triggers an immediate page to SecOps.** |
| **Prompt-injection attempts blocked** | "Did anyone try to trick the AI into ignoring its safety rules?" | Early warning of malicious or curious users. |
| **Authentication failures** | "Did anyone try to use the system with an invalid credential?" | Detects credential leaks or scraping attempts. |

### Bucket C — *"Is the system fast and available?"*  ⚙️

The usual reliability checks. These are mostly free — Google Cloud
already collects them; we just need to put them on a dashboard.

| Measurement | What it answers |
|---|---|
| **Response time** | "How long, on average, does it take for the AI to answer a question?" Target: **under 5 seconds for 95% of questions.** |
| **Error rate** | "What % of requests fail with a server error?" Target: **under 1%.** |
| **Uptime** | "Is the service responding to health checks?" Target: **99.5%.** |
| **Database & Cloud Storage health** | "Are the storage services we depend on available?" |
| **Document processing success** | "When a reviewer uploads a chart, did we successfully read and index it?" |

### Bucket D — *"What is this costing us?"*  💰

| Measurement | What it answers |
|---|---|
| **Daily AI spend** | "How much are we paying Google for AI calls per day?" |
| **Cost per session** | "What does it cost us, on average, to process one reviewer's chat session?" |
| **Active sessions** | "How many simultaneous conversations are we handling?" |
| **Database storage growth** | "How fast is our document database growing?" |

---

## 3. How we'll validate quality (not just monitor it)

Monitoring tells us *what just happened*. Validation tells us *whether the
system is getting better or worse over time*. We need both.

### Live validation — happens automatically on every question

Our AI has built-in safety layers that run on every question:
- A check that blocks obvious manipulation attempts.
- A second AI that judges whether each answer is actually supported by the
  documents (we call this the **"grounding judge"**).
- A redactor that strips out patient identifiers before any answer leaves
  our system.

All three are already running. **What we're adding is the ability to count
how often each one fires and to investigate when something looks wrong.**

### Reference validation — runs nightly against known answers

This is the part that catches **silent quality drift** — the kind that
happens when Google updates the underlying AI model and nobody told us.

We will:
1. Pick **50 representative questions** with known-correct answers
   (30 chart questions + 20 policy questions). These will be co-authored
   with a clinical reviewer.
2. Replay all 50 questions through our AI every night.
3. Score the answers automatically against the known-correct answers.
4. Email a report. **If quality drops more than 5% from last week's
   baseline, we get a ticket the next morning.**

### Compliance validation — 1% sampled audit

Every day, we'll take 1% of all real AI responses (with patient details
already redacted) and run them through a Google service called **Cloud DLP**,
which is specifically designed to detect health information that slipped
past a redactor. If it finds *anything*, SecOps gets paged immediately.

---

## 4. Timeline and milestones

**Total elapsed: 3 weeks of engineering work** (1 backend engineer + 0.5
site-reliability engineer).

| Week | Milestone | Visible to business as… |
|---|---|---|
| **Week 1** | Foundation in place. Every AI interaction now produces a structured record we can search and count. | New dashboard showing real-time response times and error rate. |
| **Week 2** | All 35 measurements live. Alerts wired into PagerDuty (for urgent issues) and ServiceNow (for tickets). Runbooks written. | "Agent Quality" dashboard goes live. On-call rotation defined. We can answer the four questions on page 1. |
| **Week 3** | Reference validation running nightly. PHI audit running nightly. CI checks block bad code from being deployed. | Weekly compliance report. PR-blocking quality gate. |

**Hard milestones for stakeholders:**

- **End of Week 1**: Show live dashboard in standup.
- **End of Week 2**: Walk-through with Compliance & SecOps; sign-off
  required before HIPAA audit prep.
- **End of Week 3**: Two consecutive weeks of green nightly compliance
  reports → declare v1 complete.

---

## 5. Cost

| Category | Estimated monthly cost | Notes |
|---|---|---|
| Google Cloud monitoring + logging | ~$50 | We're already storing logs; this just structures them. |
| Cloud DLP scanning (PHI audit) | ~$30 | 1% sampled rate keeps cost minimal. |
| Nightly reference-validation runs | ~$10 | 50 questions × $0.0067 per AI call × 30 nights. |
| BigQuery storage (90-day retention) | ~$15 | Audit logs only. |
| **Total** | **~$105/month** | A rounding error vs. one prevented incident. |

**Engineering effort: one-time, 3 weeks.** Ongoing maintenance is
minimal — about half a day per month for threshold tuning.

---

## 6. The cost of *not* doing this

If we ship a medical-claims AI to production without this layer:

1. **A single undetected PHI leak** can trigger HIPAA penalties starting at
   $100 per record, mandatory breach disclosure, and remediation costs
   that dwarf the entire engineering investment of this plan.
2. **A quiet quality regression** — for example, a Gemini model update that
   makes the AI subtly worse — could go unnoticed for weeks, with every
   reviewer in that window relying on degraded answers.
3. **No audit trail** means that when Compliance asks "show me that the AI
   refused to answer a clinical question on January 14," we can't answer.
   That's a finding in any HIPAA audit.
4. **No early warning** for prompt-injection probes means the first time we
   learn about an attack is when it succeeds.

The plan above is what a senior architect would call **"the minimum
defensible posture"** for an AI system in a regulated domain.

---

## 7. What I need from you

| Decision | Owner | Needed by |
|---|---|---|
| Approve the 3-week plan and $105/month cloud spend | Engineering Manager + Director | Start of Week 1 |
| Nominate the **clinical reviewer** who will co-author the 50 reference questions | Clinical Operations Lead | Mid Week 1 |
| Confirm **on-call rotation** for the new pages (SRE + AI on-call + SecOps) | Engineering Manager + SRE Lead | End of Week 2 |
| Compliance sign-off after the Week-2 walk-through | Compliance / Privacy Officer | End of Week 2 |
| Acknowledge that thresholds in the plan are **starting values** that we'll tune for 2 weeks based on real data | Product + Engineering | Week 3 |

---

## 8. Frequently asked questions

> **"Why now? It's been working fine."**
> "Working fine" is currently an assumption, not a measured fact. We don't
> have data either way. The whole point of this plan is to replace
> assumption with evidence — *before* an incident forces us to.

> **"Can't we just rely on user feedback (thumbs up / down)?"**
> Feedback is one input — and it's already collected. But users see only
> the answer; they cannot tell if a name leaked into a log they never see,
> or if a citation was fabricated to a page they didn't open. We need
> system-level measurement, not just user-level.

> **"Is this going to slow the system down?"**
> No. The grounding judge and safety classifier already exist in code and
> already run on every request. We are *measuring* them, not adding them.
> Logging overhead is < 1 millisecond per request.

> **"What's the risk if a metric is wrong or noisy?"**
> Each alert routes to either a pager (urgent) or a ticket (next-day).
> First 2 weeks, **all** alerts are ticket-only while we baseline real
> behaviour. We move to pager-grade only after the noise floor is
> understood. This is standard SRE practice.

> **"Will this work in production with real PHI?"**
> Yes — every component is HIPAA-eligible under our existing Google Cloud
> Business Associate Agreement. The PHI audit specifically uses
> Google's HIPAA-covered DLP service, which is designed for exactly this
> use case.

> **"Can we do half of this now and the rest later?"**
> Bucket B (the PHI / compliance bucket) is non-negotiable for production.
> Buckets A and C can be staggered, but skipping Bucket A means we're
> flying blind on AI quality — which defeats the purpose. Bucket D (cost)
> can slip a week if needed.

---

## 9. Bottom line

For **3 weeks of engineering effort** and **~$105/month** in cloud spend,
we get:

- ✅ A live picture of how good, safe, fast, and expensive our AI is.
- ✅ A defensible audit trail for HIPAA compliance.
- ✅ Automatic regression detection when models or code change.
- ✅ Clear escalation paths when something breaks.
- ✅ The data we'll need anyway to optimise cost and quality next quarter.

The alternative is hoping nothing goes wrong. In a medical-claims domain,
hope is not a strategy.

**I recommend approval to start at the beginning of next sprint.**
