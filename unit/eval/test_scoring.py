"""Unit tests for evaluation scoring functions."""

from tests.eval.scoring import (
    compute_keyword_recall,
    compute_latency_stats,
    compute_source_accuracy,
    format_markdown_summary,
)


class TestKeywordRecall:
    def test_all_keywords_present(self):
        assert (
            compute_keyword_recall(
                "Patient has acute asthma exacerbation.", ["asthma", "exacerbation"]
            )
            == 1.0
        )

    def test_no_keywords_present(self):
        assert compute_keyword_recall("Routine visit.", ["asthma", "exacerbation"]) == 0.0

    def test_partial_keywords_present(self):
        result = compute_keyword_recall("Patient has asthma.", ["asthma", "exacerbation", "acute"])
        assert abs(result - 1 / 3) < 0.01

    def test_case_insensitive(self):
        assert (
            compute_keyword_recall("ASTHMA EXACERBATION noted.", ["asthma", "exacerbation"]) == 1.0
        )

    def test_empty_keywords_returns_one(self):
        assert compute_keyword_recall("Some text.", []) == 1.0

    def test_empty_response_returns_zero(self):
        assert compute_keyword_recall("", ["asthma"]) == 0.0


class TestSourceAccuracy:
    def test_exact_match(self):
        assert compute_source_accuracy([1, 3], [1, 3]) == 1.0

    def test_no_overlap(self):
        assert compute_source_accuracy([2, 4], [1, 3]) == 0.0

    def test_partial_overlap(self):
        assert compute_source_accuracy([1, 2, 5], [1, 3]) == 0.5

    def test_empty_expected_returns_one(self):
        assert compute_source_accuracy([1, 2], []) == 1.0

    def test_empty_actual_returns_zero(self):
        assert compute_source_accuracy([], [1, 2]) == 0.0


class TestLatencyStats:
    def test_single_value(self):
        stats = compute_latency_stats([500.0])
        assert stats["p50"] == 500.0
        assert stats["mean"] == 500.0

    def test_multiple_values(self):
        stats = compute_latency_stats([100.0, 200.0, 300.0, 400.0, 500.0])
        assert stats["mean"] == 300.0
        assert stats["p50"] == 300.0
        assert stats["p95"] >= 400.0

    def test_empty_list_returns_zeros(self):
        stats = compute_latency_stats([])
        assert stats == {"p50": 0.0, "p95": 0.0, "mean": 0.0}

    def test_returns_expected_keys(self):
        stats = compute_latency_stats([100.0, 200.0])
        assert set(stats.keys()) == {"p50", "p95", "mean"}


class TestFormatMarkdownSummary:
    def test_produces_markdown_header(self):
        results = [
            {
                "document": "t.pdf",
                "question": "Q?",
                "keyword_recall": 0.8,
                "source_accuracy": 1.0,
                "latency_ms": 450.0,
                "tokens": {"prompt": 100, "completion": 50},
            }
        ]
        assert "# Evaluation Results" in format_markdown_summary(results)

    def test_includes_aggregate_stats(self):
        results = [
            {
                "document": "t.pdf",
                "question": "Q1",
                "keyword_recall": 0.5,
                "source_accuracy": 1.0,
                "latency_ms": 200.0,
                "tokens": {"prompt": 100, "completion": 50},
            },
            {
                "document": "t.pdf",
                "question": "Q2",
                "keyword_recall": 1.0,
                "source_accuracy": 0.5,
                "latency_ms": 400.0,
                "tokens": {"prompt": 120, "completion": 60},
            },
        ]
        output = format_markdown_summary(results)
        assert "Avg Keyword Recall" in output
        assert "Avg Source Accuracy" in output
        assert "P50" in output or "p50" in output.lower()

    def test_empty_results(self):
        output = format_markdown_summary([])
        assert "0 questions" in output
