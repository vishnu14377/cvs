"""Tests for stub_raw_predict_response (Mistral OCR stub payload)."""

from __future__ import annotations

from src.core.stubs.vertex_raw_predict_stub import stub_raw_predict_response


class TestStubRawPredictResponse:
    def test_returns_dict(self) -> None:
        r = stub_raw_predict_response({"document": {"document_url": "gs://b/f.pdf"}})
        assert isinstance(r, dict)

    def test_contains_pages_list(self) -> None:
        r = stub_raw_predict_response({"document": {"document_url": "gs://b/f.pdf"}})
        assert "pages" in r
        assert isinstance(r["pages"], list)
        assert len(r["pages"]) >= 1

    def test_page_shape_matches_mistral(self) -> None:
        r = stub_raw_predict_response({"document": {"document_url": "gs://b/f.pdf"}})
        page = r["pages"][0]
        assert "index" in page
        assert page["index"] == 0
        assert "markdown" in page
        assert isinstance(page["markdown"], str)

    def test_deterministic(self) -> None:
        r1 = stub_raw_predict_response({"document": {"document_url": "gs://b/f.pdf"}})
        r2 = stub_raw_predict_response({"document": {"document_url": "gs://b/f.pdf"}})
        assert r1 == r2

    def test_validates_against_pydantic_model(self) -> None:
        """Confirm stub response parses cleanly as the real response model."""
        from src.ocr.data_models.mistral_response import MistralOcrResponse

        r = stub_raw_predict_response({"document": {"document_url": "gs://b/f.pdf"}})
        parsed = MistralOcrResponse.model_validate(r)
        assert len(parsed.pages) >= 1
        assert parsed.pages[0].index == 0
