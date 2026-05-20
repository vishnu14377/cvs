"""
Data models for SubFileHandler.

These models define the data structures used by the SubFileHandler
for processing PDF sub-files with page number mapping.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

# Type alias for model names
ModelName = Literal["mistral", "llm"]


@dataclass
class PageInfo:
    """
    Information about a single extracted page with original document mapping.

    This model extends the basic page extraction with mapping information
    that relates the sub-file page index to the original document page number.
    """

    # Page index within the sub-file (0-based, from OCR response)
    sub_file_index: int
    # Actual page number in the original document (1-based)
    original_page_number: int
    # Extracted text content for this page
    extracted_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "sub_file_index": self.sub_file_index,
            "original_page_number": self.original_page_number,
            "extracted_text": self.extracted_text,
        }


@dataclass
class SubFileMetadata:
    """
    Metadata for a processed sub-file.

    Contains information about the document name and page range
    extracted from the sub-file filename pattern.
    """

    # Extracted document name (without page range suffix)
    document_name: str
    # Base page number (start of range in original document, 1-based)
    base_page_number: int
    # End page number (end of range in original document, 1-based)
    end_page_number: int

    @property
    def expected_page_count(self) -> int:
        """Calculate the expected number of pages in this sub-file."""
        return self.end_page_number - self.base_page_number + 1

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "document_name": self.document_name,
            "base_page_number": self.base_page_number,
            "end_page_number": self.end_page_number,
        }


@dataclass
class SubFileResult:
    """
    Complete result from processing a sub-file.

    Contains the metadata, extracted pages with mapping, status information,
    and fallback tracking.
    """

    # Processing metadata
    metadata: SubFileMetadata
    # List of extracted pages with mapped page numbers
    pages: list[PageInfo] = field(default_factory=list)
    # Processing success status
    success: bool = False
    # Error message if processing failed
    error: str | None = None
    # Which model was used to produce the result ("mistral" or "llm")
    model_used: ModelName | None = None
    # Whether fallback from primary to secondary model was used
    fallback_used: bool = False
    # Error from primary model if fallback was triggered
    primary_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "document_name": self.metadata.document_name,
            "base_page_number": self.metadata.base_page_number,
            "end_page_number": self.metadata.end_page_number,
            "pages": [page.to_dict() for page in self.pages],
            "success": self.success,
            "error": self.error,
            "model_used": self.model_used,
            "fallback_used": self.fallback_used,
            "primary_error": self.primary_error,
        }

    @property
    def page_count(self) -> int:
        """Get the number of extracted pages."""
        return len(self.pages)

    def get_combined_text(self, separator: str = "\n\n") -> str:
        """
        Get combined text from all pages.

        Args:
            separator: String to use between pages

        Returns:
            Combined text from all pages
        """
        return separator.join(page.extracted_text for page in self.pages)


def map_page_to_original(sub_file_index: int, base_page_number: int) -> int:
    """
    Map a sub-file page index to the original document page number.

    Args:
        sub_file_index: 0-based index from OCR response
        base_page_number: Starting page number in original document (1-based)

    Returns:
        Actual page number in the original document (1-based)

    Example:
        If base_page_number=19 (for pages 19-36):
        - sub_file_index=0 -> original page 19
        - sub_file_index=5 -> original page 24
    """
    return base_page_number + sub_file_index
