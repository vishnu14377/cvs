"""Stub session manager for the /dev/unqork-mock harness.

Creates an in-memory session with no documents, so the widget chat UI can be
exercised end-to-end without OCR + vector DB + GCS.
"""
from __future__ import annotations

from src.session_manager.core.agent_factory import get_agent


class StubSessionManager:
    """Minimal drop-in for SessionManager that bypasses OCR/ingestion.

    The agent still runs; its ADR search tool will return zero hits, and the
    grounding judge will reply with an "insufficient information" response.
    That's acceptable for demonstrating the widget UI in the harness.
    """

    def __init__(self, session_id: str):
        self._session_id = session_id

    @property
    def agent(self):
        return get_agent()
