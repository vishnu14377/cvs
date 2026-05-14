"""
Session ID Generator.

Generates unique, human-readable session identifiers used to track
a single ADR document-processing run across all pipeline stages.

Format: ``adr-<YYYYMMDD>-<short-uuid>``
Example: ``adr-20260413-a1b2c3d4``
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone


def generate_session_id() -> str:
    """
    Generate a unique session ID.

    Returns:
        A string in the format ``adr-YYYYMMDD-<8-hex-chars>``.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    short_uuid = uuid.uuid4().hex[:8]
    return f"adr-{timestamp}-{short_uuid}"
