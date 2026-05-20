"""Policy document data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class PolicyDocument:
    """Metadata for a policy document stored in the system."""

    policy_id: str
    policy_name: str
    gcs_uri: str
    page_count: int = 0
    ocr_engine: str = "mistral"
    category: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    status: str = "active"
