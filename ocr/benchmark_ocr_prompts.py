"""
Benchmark script to test OCR prompts on multiple PDFs and profile latency.

Usage:
    python -m src.ocr.benchmark_ocr_prompts
"""

import time
import json
from typing import List, Dict, Any
from src.ocr.llm_ocr_client import get_llm_ocr_client
from src.core.logger import get_logger

logger = get_logger(__name__)

# ── Configure test PDFs (GCS URIs) ──────────────────────────────────────────
TEST_PDFS = [
    "gs://care_connect_ai_initiatives/test_full_adrs/anderson_adr_redacted.pdf",
    "gs://care_connect_ai_initiatives/test_full_adrs/BINGHAM,CALLIE_91202308017_FLC5_REDACTED.pdf"
]


def run_benchmark(gcs_uris: List[str]) -> List[Dict[str, Any]]:
    client = get_llm_ocr_client()
    results = []

    for uri in gcs_uris:
        logger.info("Benchmarking: %s", uri)
        start = time.perf_counter()

        result = client.process_document(uri)

        elapsed = time.perf_counter() - start
        page_count = len(result.get("pages", []))

        summary = {
            "uri": uri,
            "success": result["success"],
            "page_count": page_count,
            "latency_seconds": round(elapsed, 3),
            "latency_per_page": round(elapsed / page_count, 3) if page_count else None,
            "error": result.get("error"),
        }

        logger.info(
            "Result: success=%s | pages=%d | latency=%.3fs | per_page=%.3fs",
            summary["success"], page_count, elapsed,
            summary["latency_per_page"] or 0,
        )
        results.append(summary)

    return results


if __name__ == "__main__":
    results = run_benchmark(TEST_PDFS)
    print("\n=== BENCHMARK SUMMARY ===")
    print(json.dumps(results, indent=2))

    successes = sum(1 for r in results if r["success"])
    avg_latency = sum(r["latency_seconds"] for r in results) / len(results)
    print(f"\nSuccess rate: {successes}/{len(results)}")
    print(f"Avg latency: {avg_latency:.3f}s")