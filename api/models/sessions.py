"""Session request/response models."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class CreateSessionRequest(BaseModel):
    """Request body for POST /api/v1/sessions (JSON mode)."""

    gcs_uris: list[str] = Field(
        ..., min_length=1, description="GCS URIs of PDF documents to process"
    )
    ocr_engine: str = Field(
        default="mistral", description="OCR engine: mistral, gemini-vision, or document-ai"
    )
    metadata: dict[str, str] | None = Field(
        default=None, description="Optional metadata (encounterName, userId, etc.)"
    )


class SessionResponse(BaseModel):
    """Response for session operations."""

    session_id: str
    status: str
    documents_processed: int = 0
    processing_time_ms: int = 0
    created_at: datetime | None = None
    message_count: int = 0
    metadata: dict[str, str] | None = None


class SessionDeleteResponse(BaseModel):
    """Response for DELETE /api/v1/sessions/{sessionId}."""

    session_id: str
    status: str = "deleted"
    vectors_deleted: int = 0
    errors: list[str] = Field(default_factory=list)


class SessionListResponse(BaseModel):
    """Response for GET /api/v1/sessions."""

    sessions: list[SessionResponse] = Field(default_factory=list)
