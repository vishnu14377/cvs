"""Feedback request/response models."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class FeedbackRequest(BaseModel):
    """Request body for POST /api/v1/sessions/{sessionId}/feedback."""

    message_id: str = Field(..., description="ID of the agent message being rated")
    rating: Literal["positive", "negative"] = Field(..., description="User rating")
    comment: str | None = Field(default=None, max_length=2000, description="Optional user comment")


class FeedbackResponse(BaseModel):
    """Response for feedback submission."""

    feedback_id: str
    session_id: str
    message_id: str
    status: str = "stored"


class FeedbackEntry(BaseModel):
    """A single feedback record in a list response."""

    feedback_id: str
    message_id: str
    rating: str
    comment: str | None = None
    created_at: str | None = None


class FeedbackListResponse(BaseModel):
    """Response for GET /api/v1/sessions/{sessionId}/feedback."""

    session_id: str
    feedback: list[FeedbackEntry] = Field(default_factory=list)
