"""CI/test stubs for Vertex-AI-backed clients."""

from __future__ import annotations

from src.core.stubs.chat_stub import StubChatModel, StubStructuredRunnable
from src.core.stubs.embedding_stub import StubEmbeddings
from src.core.stubs.vertex_raw_predict_stub import stub_raw_predict_response

__all__ = [
    "StubChatModel",
    "StubStructuredRunnable",
    "StubEmbeddings",
    "stub_raw_predict_response",
]
