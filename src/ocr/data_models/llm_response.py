"""
Pydantic models for LLM OCR structured output.

These models are used with LangChain's `.with_structured_output()` method
to get typed responses directly from the LLM without manual parsing.

Usage:
    from src.ocr.data_models.llm_response import DocumentExtraction, PageExtraction

    # Use with LangChain's structured output
    structured_llm = llm.with_structured_output(DocumentExtraction)
    result: DocumentExtraction = structured_llm.invoke(messages)
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class PageExtraction(BaseModel):
    """Represents extracted data for a single page."""
    page_number: int = Field(..., description="1-based page index")
    page_text: Optional[str] = Field(None, description="Full extracted text from the page")
    page_summary: str = Field(..., description="Concise summary of the page")
    page_insight: str = Field(..., description="Key insights, keywords, and important flags")


class DocumentExtraction(BaseModel):
    """Represents the full document extraction result."""
    pages: List[PageExtraction] = Field(..., description="List of page extractions")


__all__ = [
    "PageExtraction",
    "DocumentExtraction",
]
