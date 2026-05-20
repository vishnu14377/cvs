"""Shared fixtures for API tests."""

from __future__ import annotations

import os

import pytest
from httpx import ASGITransport, AsyncClient

# Set a test auth token before importing the app
os.environ.setdefault("API_AUTH_TOKEN", "test-token-secret")
# Enable dev routes for unit tests that exercise /dev/test. Production builds
# must leave ENABLE_DEV_ROUTES unset so the no-auth harness stays hidden.
os.environ.setdefault("ENABLE_DEV_ROUTES", "1")


@pytest.fixture(autouse=True)
def clear_session_registry():
    """Clear the in-memory session registry before each test to prevent state leakage."""
    from src.api.dependencies import get_session_registry

    registry = get_session_registry()
    registry.clear()
    yield
    registry.clear()


@pytest.fixture
def auth_headers() -> dict:
    """Authorization headers for authenticated requests."""
    return {"Authorization": "Bearer test-token-secret"}


@pytest.fixture
async def client():
    """Async test client for the FastAPI app."""
    from src.api.app import create_app

    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
