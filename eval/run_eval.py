"""Evaluation harness for ADR AI Agent golden datasets.

Drives the live API with golden Q&A pairs, scores keyword recall,
source accuracy, latency, and token usage, then outputs a markdown summary.

Usage:
    python tests/eval/run_eval.py --api-url http://localhost:8000

NOT wired into CI — requires live Vertex AI.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import httpx

from tests.eval.scoring import (
    compute_keyword_recall,
    compute_source_accuracy,
    format_markdown_summary,
)

GOLDEN_DIR = Path(__file__).parent.parent.parent / "data" / "golden"
DEFAULT_API_URL = "http://localhost:8000"
DEFAULT_TIMEOUT = 120.0


def load_qa_pairs(dataset: str) -> list[dict]:
    qa_path = GOLDEN_DIR / dataset / "qa_pairs.json"
    if not qa_path.exists():
        print(f"ERROR: Q&A file not found: {qa_path}", file=sys.stderr)
        sys.exit(1)
    with open(qa_path) as f:
        return json.load(f)


def create_session(client: httpx.Client, api_url: str, pdf_path: Path, token: str) -> str | None:
    url = f"{api_url}/api/v1/sessions/upload"
    headers = {"Authorization": f"Bearer {token}"}
    with open(pdf_path, "rb") as f:
        files = {"files": (pdf_path.name, f, "application/pdf")}
        response = client.post(url, files=files, headers=headers, timeout=DEFAULT_TIMEOUT)
    if response.status_code != 201:
        print(
            f"  ERROR creating session for {pdf_path.name}: {response.status_code}", file=sys.stderr
        )
        return None
    return response.json().get("session_id")


def query_session(
    client: httpx.Client, api_url: str, session_id: str, question: str, token: str
) -> dict:
    url = f"{api_url}/api/v1/sessions/{session_id}/query"
    headers = {"Authorization": f"Bearer {token}"}
    start = time.monotonic()
    response = client.post(
        url, json={"message": question}, headers=headers, timeout=DEFAULT_TIMEOUT
    )
    latency_ms = (time.monotonic() - start) * 1000

    if response.status_code != 200:
        return {
            "content": "",
            "sources": [],
            "latency_ms": latency_ms,
            "tokens": {"prompt": 0, "completion": 0},
            "error": f"{response.status_code}: {response.text[:200]}",
        }

    data = response.json()
    message = data.get("message", {})
    sources = data.get("sources", [])
    token_usage = data.get("metadata", {}).get("tokenUsage") or {}
    return {
        "content": message.get("content", ""),
        "sources": sources,
        "latency_ms": latency_ms,
        "tokens": {
            "prompt": token_usage.get("prompt", 0),
            "completion": token_usage.get("completion", 0),
        },
    }


def delete_session(client: httpx.Client, api_url: str, session_id: str, token: str) -> None:
    client.delete(
        f"{api_url}/api/v1/sessions/{session_id}",
        headers={"Authorization": f"Bearer {token}"},
        timeout=30.0,
    )


def run_evaluation(api_url: str, token: str, datasets: list[str]) -> list[dict]:
    results = []
    with httpx.Client() as client:
        for dataset in datasets:
            qa_pairs = load_qa_pairs(dataset)
            print(f"\n{'=' * 60}\nEvaluating: {dataset} ({len(qa_pairs)} questions)\n{'=' * 60}")

            doc_questions: dict[str, list[dict]] = {}
            for pair in qa_pairs:
                doc_questions.setdefault(pair["document"], []).append(pair)

            for doc_name, questions in doc_questions.items():
                pdf_path = GOLDEN_DIR / dataset / doc_name
                if not pdf_path.exists():
                    print(f"  SKIP: {pdf_path}", file=sys.stderr)
                    continue

                print(f"\n  Document: {doc_name}")
                session_id = create_session(client, api_url, pdf_path, token)
                if not session_id:
                    for q in questions:
                        results.append(
                            {
                                "document": doc_name,
                                "question": q["question"],
                                "keyword_recall": 0.0,
                                "source_accuracy": 0.0,
                                "latency_ms": 0.0,
                                "tokens": {"prompt": 0, "completion": 0},
                            }
                        )
                    continue

                for q in questions:
                    print(f"    Q: {q['question'][:60]}...")
                    resp = query_session(client, api_url, session_id, q["question"], token)
                    if "error" in resp:
                        print(f"    ERROR: {resp['error']}")
                        results.append(
                            {
                                "document": doc_name,
                                "question": q["question"],
                                "keyword_recall": 0.0,
                                "source_accuracy": 0.0,
                                "latency_ms": resp["latency_ms"],
                                "tokens": resp["tokens"],
                            }
                        )
                        continue

                    kr = compute_keyword_recall(resp["content"], q["expected_keywords"])
                    pages = [s.get("page") for s in resp["sources"] if s.get("page") is not None]
                    sa = compute_source_accuracy(pages, q.get("expected_page_refs", []))
                    results.append(
                        {
                            "document": doc_name,
                            "question": q["question"],
                            "keyword_recall": kr,
                            "source_accuracy": sa,
                            "latency_ms": resp["latency_ms"],
                            "tokens": resp["tokens"],
                        }
                    )
                    print(f"    Recall={kr:.0%} Source={sa:.0%} Latency={resp['latency_ms']:.0f}ms")

                delete_session(client, api_url, session_id, token)
    return results


def main():
    parser = argparse.ArgumentParser(description="ADR AI Agent evaluation")
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument("--token", default="test-token-secret")
    parser.add_argument("--datasets", nargs="+", default=["adr", "policy"])
    parser.add_argument("--output", "-o", help="Write report to file")
    args = parser.parse_args()

    print(f"API: {args.api_url}\nDatasets: {', '.join(args.datasets)}")
    results = run_evaluation(args.api_url, args.token, args.datasets)
    summary = format_markdown_summary(results)

    if args.output:
        Path(args.output).write_text(summary)
        print(f"\nReport: {args.output}")
    else:
        print(f"\n{'=' * 60}\n{summary}")


if __name__ == "__main__":
    main()
