"""Query request/response models."""

from __future__ import annotations

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request body for POST /api/v1/sessions/{sessionId}/query."""

    message: str = Field(..., min_length=1, max_length=5000, description="User question")
    include_source_references: bool = Field(
        default=True, description="Include source document references in response"
    )


class SourceReference(BaseModel):
    """A source document reference for a response."""

    document: str
    page: int | None = None
    chunk_text: str | None = None
    relevance_score: float | None = None


class MessageResponse(BaseModel):
    """The agent's response message."""

    role: str = "assistant"
    content: str
    content_html: str
    content_base64: str


class QueryResponse(BaseModel):
    """Response for POST /api/v1/sessions/{sessionId}/query."""

    session_id: str
    message_id: str
    message: MessageResponse
    sources: list[SourceReference] = Field(default_factory=list)
    metadata: dict | None = None


class HistoryMessage(BaseModel):
    """A single message in conversation history."""

    message_id: str
    role: str
    content: str
    content_html: str | None = None
    timestamp: str | None = None


class HistoryResponse(BaseModel):
    """Response for GET /api/v1/sessions/{sessionId}/history."""

    session_id: str
    messages: list[HistoryMessage] = Field(default_factory=list)
