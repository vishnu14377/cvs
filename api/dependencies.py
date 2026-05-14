"""Shared FastAPI dependencies: auth, session registry."""

from __future__ import annotations

import os
import time as _time

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from src.session_manager.core.session_manager import SessionManager

_bearer_scheme = HTTPBearer(auto_error=False)


def _get_valid_tokens() -> set[str]:
    """Get all valid auth tokens from environment."""
    tokens: set[str] = set()
    api_token = os.environ.get("API_AUTH_TOKEN", "")
    if not api_token:
        raise RuntimeError("API_AUTH_TOKEN environment variable must be set")
    tokens.add(api_token)
    widget_token = os.environ.get("CARECONNECT_WIDGET_TOKEN", "")
    if widget_token:
        tokens.add(widget_token)
    return tokens


async def verify_token(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),  # noqa: B008
) -> str:
    """Verify Bearer token. Raises 401 if invalid."""
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authorization header",
        )
    valid_tokens = _get_valid_tokens()
    if credentials.credentials not in valid_tokens:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
        )
    return credentials.credentials


# In-memory session registry (maps session_id -> (SessionManager, created_timestamp))
_session_registry: dict[str, tuple[SessionManager, float]] = {}


def get_session_registry() -> dict[str, tuple[SessionManager, float]]:
    """Return the session registry dict. Values are (manager, created_timestamp)."""
    return _session_registry


def get_session_manager(session_id: str) -> SessionManager:
    """Look up a SessionManager by session_id. Raises 404 if not found."""
    registry = get_session_registry()
    entry = registry.get(session_id)
    if entry is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found",
        )
    manager, _created = entry
    return manager


from src.core.logger import get_logger as _get_logger  # noqa: E402

_logger = _get_logger(__name__)


def cleanup_expired_sessions(ttl_hours: int = 24) -> int:
    """Remove sessions older than ttl_hours. Returns count removed."""
    now = _time.time()
    expired = [
        sid for sid, (_, created) in _session_registry.items() if now - created > ttl_hours * 3600
    ]
    for sid in expired:
        _session_registry.pop(sid, None)
        _logger.info("Expired session removed: %s", sid)
    return len(expired)
