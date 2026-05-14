"""Widget API request/response models for Unqork integration."""

from __future__ import annotations

from pydantic import BaseModel, Field


class WidgetQueryRequest(BaseModel):
    """Request body for POST /widget/v1/chat/query.

    Session ID is in the body (not URL path) because Unqork Plug-In
    components may have limitations on dynamic URL construction.
    """

    session_id: str = Field(..., description="Session ID")
    message: str = Field(..., min_length=1, max_length=5000, description="User question")


class WidgetQueryResponse(BaseModel):
    """Unqork-optimized response with base64-encoded HTML."""

    content_base64: str = Field(description="Base64-encoded safe HTML")
    content_html: str = Field(description="Safe HTML string")
    content_text: str = Field(description="Plain text version")
    message_id: str
    generated_at: str | None = Field(default=None, description="ISO 8601 timestamp")
    has_more: bool = False
    error: str | None = None
