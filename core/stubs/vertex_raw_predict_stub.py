"""Deterministic Mistral-OCR raw_predict response stub."""

from __future__ import annotations

from typing import Any

from src.core.logger import get_logger

logger = get_logger(__name__)


def stub_raw_predict_response(_payload: dict[str, Any]) -> dict[str, Any]:
    """Return a deterministic Mistral-OCR response payload.

    Shape matches src/ocr/data_models/mistral_response.py:MistralOcrResponse:
      {"pages": [{"index": int, "markdown": str}, ...]}
    """
    logger.debug("stub_raw_predict_response called (VERTEX_AI_MODE=stub)")
    return {
        "pages": [
            {
                "index": 0,
                "markdown": "[stub] Mistral OCR page 0",
            }
        ],
        "model": "mistral-ocr-2505",
        "usage_info": {"pages_processed": 1, "doc_size_bytes": 0},
    }
