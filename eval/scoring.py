"""Scoring functions for ADR AI Agent evaluation."""

from __future__ import annotations

import statistics


def compute_keyword_recall(response_text: str, expected_keywords: list[str]) -> float:
    if not expected_keywords:
        return 1.0
    if not response_text:
        return 0.0
    response_lower = response_text.lower()
    found = sum(1 for kw in expected_keywords if kw.lower() in response_lower)
    return found / len(expected_keywords)


def compute_source_accuracy(actual_pages: list[int], expected_pages: list[int]) -> float:
    if not expected_pages:
        return 1.0
    if not actual_pages:
        return 0.0
    actual_set = set(actual_pages)
    found = sum(1 for p in expected_pages if p in actual_set)
    return found / len(expected_pages)


def compute_latency_stats(latencies_ms: list[float]) -> dict[str, float]:
    if not latencies_ms:
        return {"p50": 0.0, "p95": 0.0, "mean": 0.0}
    sorted_lat = sorted(latencies_ms)
    n = len(sorted_lat)
    p95_index = min(int(n * 0.95), n - 1)
    return {
        "p50": round(statistics.median(sorted_lat), 1),
        "p95": round(sorted_lat[p95_index], 1),
        "mean": round(statistics.mean(sorted_lat), 1),
    }


def format_markdown_summary(results: list[dict]) -> str:
    if not results:
        return "# Evaluation Results\n\nNo results — 0 questions evaluated.\n"

    keyword_recalls = [r["keyword_recall"] for r in results]
    source_accuracies = [r["source_accuracy"] for r in results]
    latencies = [r["latency_ms"] for r in results]
    total_prompt = sum(r["tokens"].get("prompt", 0) for r in results)
    total_completion = sum(r["tokens"].get("completion", 0) for r in results)

    avg_kr = statistics.mean(keyword_recalls)
    avg_sa = statistics.mean(source_accuracies)
    lat = compute_latency_stats(latencies)

    lines = [
        "# Evaluation Results",
        "",
        f"**Questions evaluated:** {len(results)}",
        "",
        "## Aggregate Metrics",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Avg Keyword Recall | {avg_kr:.2%} |",
        f"| Avg Source Accuracy | {avg_sa:.2%} |",
        f"| Latency P50 | {lat['p50']:.0f} ms |",
        f"| Latency P95 | {lat['p95']:.0f} ms |",
        f"| Latency Mean | {lat['mean']:.0f} ms |",
        f"| Total Prompt Tokens | {total_prompt} |",
        f"| Total Completion Tokens | {total_completion} |",
        "",
        "## Per-Question Results",
        "",
        "| Document | Question | Keyword Recall | Source Accuracy | Latency (ms) |",
        "|----------|----------|---------------|-----------------|---------------|",
    ]

    for r in results:
        q = r["question"][:50] + "..." if len(r["question"]) > 50 else r["question"]
        lines.append(
            f"| {r['document']} | {q} | {r['keyword_recall']:.2%} "
            f"| {r['source_accuracy']:.2%} | {r['latency_ms']:.0f} |"
        )
    lines.append("")
    return "\n".join(lines)
