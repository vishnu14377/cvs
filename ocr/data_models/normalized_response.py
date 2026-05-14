"""
Normalized OCR response models used by OcrModelClient.

These models provide a unified format that abstracts the differences
between Mistral and LLM OCR backends. The OcrModelClient converts
raw responses from either backend into this normalized format.

The SubFileHandler consumes NormalizedOcrResponse for further processing.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field
from src.ocr.data_models.llm_response import DocumentExtraction, PageExtraction
from src.ocr.data_models.mistral_response import MistralOcrResponse, MistralPageResponse

# Type alias for model names
ModelName = Literal["mistral", "llm"]


class NormalizedPage(BaseModel):
    """
    Normalized page format used by OcrModelClient.

    This is the unified format that both Mistral and LLM responses
    are converted to. Uses 'index' and 'extracted_text' as standard keys.
    """

    index: int = Field(..., description="Page index (0-based)", ge=0)
    extracted_text: str = Field(default="", description="Extracted text content")

    @classmethod
    def from_mistral(cls, page: MistralPageResponse) -> NormalizedPage:
        """Create from Mistral page response (converts 'markdown' to 'extracted_text')."""
        return cls(index=page.index, extracted_text=page.markdown)

    @classmethod
    def from_llm(cls, page: PageExtraction) -> NormalizedPage:
        """Create from LLM page response."""
        return cls(index=page.index, extracted_text=page.extracted_text)

    @classmethod
    def from_dict(cls, data: dict, default_index: int = 0) -> NormalizedPage:
        """
        Create from a dictionary with fallback handling.

        Handles various key names from different OCR backends:
        - extracted_text, markdown, text, content -> extracted_text
        """
        text = (
            data.get("extracted_text")
            or data.get("markdown")
            or data.get("text")
            or data.get("content")
            or ""
        )
        return cls(
            index=data.get("index", default_index),
            extracted_text=text,
        )

    def to_dict(self) -> dict:
        """Convert to simple dictionary."""
        return {"index": self.index, "extracted_text": self.extracted_text}


class NormalizedOcrResponse(BaseModel):
    """
    Normalized OCR response from OcrModelClient.

    This is the standard response format that SubFileHandler consumes.
    It abstracts away the differences between Mistral and LLM backends.

    Includes tracking for which model was used and whether fallback occurred.
    """

    success: bool = Field(..., description="Whether OCR processing succeeded")
    pages: list[NormalizedPage] = Field(
        default_factory=list,
        description="List of normalized pages with index and extracted_text",
    )
    error: str | None = Field(
        default=None,
        description="Error message if processing failed",
    )
    model_used: ModelName | None = Field(
        default=None,
        description="Which model was used to produce the response (mistral or llm)",
    )
    fallback_used: bool = Field(
        default=False,
        description="Whether fallback from primary to secondary model was used",
    )
    primary_error: str | None = Field(
        default=None,
        description="Error message from primary model if fallback was triggered",
    )

    @classmethod
    def from_mistral_response(
        cls,
        response: MistralOcrResponse,
        fallback_used: bool = False,
        primary_error: str | None = None,
    ) -> NormalizedOcrResponse:
        """Create from Mistral OCR response."""
        pages = [NormalizedPage.from_mistral(p) for p in response.pages]
        return cls(
            success=True,
            pages=pages,
            model_used="mistral",
            fallback_used=fallback_used,
            primary_error=primary_error,
        )

    @classmethod
    def from_llm_response(
        cls,
        response: DocumentExtraction,
        fallback_used: bool = False,
        primary_error: str | None = None,
    ) -> NormalizedOcrResponse:
        """Create from LLM OCR response."""
        pages = [NormalizedPage.from_llm(p) for p in response.pages]
        return cls(
            success=True,
            pages=pages,
            model_used="llm",
            fallback_used=fallback_used,
            primary_error=primary_error,
        )

    @classmethod
    def from_error(
        cls,
        error: str,
        model_used: ModelName | None = None,
        fallback_used: bool = False,
        primary_error: str | None = None,
    ) -> NormalizedOcrResponse:
        """Create an error response."""
        return cls(
            success=False,
            pages=[],
            error=error,
            model_used=model_used,
            fallback_used=fallback_used,
            primary_error=primary_error,
        )

    def to_dict(self) -> dict:
        """Convert to simple dictionary for JSON serialization."""
        return {
            "success": self.success,
            "pages": [p.to_dict() for p in self.pages],
            "error": self.error,
            "model_used": self.model_used,
            "fallback_used": self.fallback_used,
            "primary_error": self.primary_error,
        }
