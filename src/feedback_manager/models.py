"""Feedback data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal


@dataclass
class FeedbackRecord:
    """A feedback record to store in MongoDB."""

    session_id: str
    message_id: str
    user_message: str
    ai_response: str
    rating: Literal["positive", "negative"]
    comment: str | None = None
    document_names: list[str] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
