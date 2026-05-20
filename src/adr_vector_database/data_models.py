"""
Data models for ADR Vector Database.

These models define the structure of extracted OCR documents that will be
chunked, embedded, and stored in the vector database for RAG retrieval.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field


class ExtractedPage(BaseModel):
    """
    A single page from the extracted OCR JSON.

    Maps directly to the page structure in the extracted JSON files.
    """

    sub_file_index: int = Field(..., description="0-based index within the sub-file")
    original_page_number: int = Field(..., description="1-based page number in original document")
    extracted_text: str = Field(default="", description="Extracted text content from OCR")


class ExtractedDocument(BaseModel):
    """
    Complete extracted document from OCR processing.

    This model represents the JSON structure stored in GCS after OCR processing.
    """

    document_name: str = Field(..., description="Name of the original document")
    base_page_number: int = Field(..., description="Starting page number (1-based)")
    end_page_number: int = Field(..., description="Ending page number (1-based)")
    pages: list[ExtractedPage] = Field(default_factory=list, description="List of extracted pages")

    # Optional metadata fields that may be present
    success: bool | None = Field(default=None, description="Whether OCR was successful")
    model_used: str | None = Field(default=None, description="OCR model used (mistral or llm)")
    fallback_used: bool | None = Field(default=None, description="Whether fallback model was used")
    error: str | None = Field(default=None, description="Error message if OCR failed")

    @property
    def page_count(self) -> int:
        """Get the number of pages in this document."""
        return len(self.pages)

    def get_combined_text(self, separator: str = "\n\n") -> str:
        """Combine all page texts into a single string."""
        return separator.join(page.extracted_text for page in self.pages if page.extracted_text)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExtractedDocument:
        """Create from a dictionary (parsed JSON)."""
        pages = [
            ExtractedPage(**page) if isinstance(page, dict) else page
            for page in data.get("pages", [])
        ]
        return cls(
            document_name=data.get("document_name", "unknown"),
            base_page_number=data.get("base_page_number", 1),
            end_page_number=data.get("end_page_number", 1),
            pages=pages,
            success=data.get("success"),
            model_used=data.get("model_used"),
            fallback_used=data.get("fallback_used"),
            error=data.get("error"),
        )


@dataclass
class DocumentChunk:
    """
    A chunk of text from an extracted document, ready for embedding.

    Contains the text content plus metadata needed for retrieval and
    tracing back to the source document.
    """

    # Core content
    text: str

    # Source tracking
    document_name: str
    page_numbers: list[int]  # List of page numbers this chunk spans
    chunk_index: int  # Index of this chunk within the document

    # Session/processing metadata
    session_id: str
    gcs_source_uri: str | None = None

    # Additional metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_metadata_dict(self) -> dict[str, Any]:
        """
        Convert chunk metadata to a dictionary for vector store storage.

        This metadata will be stored alongside the embedding in PGVector.
        """
        return {
            "document_name": self.document_name,
            "page_numbers": self.page_numbers,
            "chunk_index": self.chunk_index,
            "session_id": self.session_id,
            "gcs_source_uri": self.gcs_source_uri or "",
            "page_start": min(self.page_numbers) if self.page_numbers else 0,
            "page_end": max(self.page_numbers) if self.page_numbers else 0,
            **self.metadata,
        }


@dataclass
class IngestionResult:
    """
    Result of ingesting a document into the vector database.
    """

    document_name: str
    session_id: str
    success: bool
    chunks_created: int = 0
    chunks_stored: int = 0
    vector_ids: list[str] = field(default_factory=list)
    error: str | None = None
    gcs_source_uri: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "document_name": self.document_name,
            "session_id": self.session_id,
            "success": self.success,
            "chunks_created": self.chunks_created,
            "chunks_stored": self.chunks_stored,
            "vector_ids": self.vector_ids,
            "error": self.error,
            "gcs_source_uri": self.gcs_source_uri,
        }


@dataclass
class BatchIngestionResult:
    """
    Result of ingesting multiple documents into the vector database.
    """

    session_id: str
    total_documents: int
    successful_documents: int = 0
    failed_documents: int = 0
    total_chunks_stored: int = 0
    results: list[IngestionResult] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        """Check if all documents were successfully ingested."""
        return self.failed_documents == 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "session_id": self.session_id,
            "total_documents": self.total_documents,
            "successful_documents": self.successful_documents,
            "failed_documents": self.failed_documents,
            "total_chunks_stored": self.total_chunks_stored,
            "success": self.success,
            "results": [r.to_dict() for r in self.results],
            "errors": self.errors,
        }
