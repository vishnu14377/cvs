"""
Pydantic data models for Mistral OCR responses.

These models represent the raw JSON response structure from the Mistral OCR model.
Mistral returns pages with 'index' and 'markdown' keys.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class MistralPageResponse(BaseModel):
    """
    Raw page response from Mistral OCR.

    Mistral returns pages with 'index' and 'markdown' keys.
    """

    index: int = Field(..., description="Page index (0-based)")
    markdown: str = Field(default="", description="Extracted text in markdown format")


class MistralOcrResponse(BaseModel):
    """
    Raw response structure from Mistral OCR.

    This represents the actual JSON structure returned by the Mistral model.
    """

    pages: list[MistralPageResponse] = Field(
        default_factory=list,
        description="List of pages with markdown content",
    )
