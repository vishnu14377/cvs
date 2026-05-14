"""Shared fixtures for all unit tests.

Sets API_AUTH_TOKEN early so it's available before any module imports
the FastAPI app (which reads the token via os.environ in verify_token).
Without this, tests that run before api/conftest.py would trigger app
imports without the token set, causing 401 failures.

Also defensively restores API_AUTH_TOKEN after every test, because some
tests use ``patch.dict(os.environ, {}, clear=True)`` which removes the
token for the duration of the context manager. Pytest fixture ordering
and module-level caches can leave the token cleared even after the
context manager exits in rare cases.
"""

import os

import pytest

_API_AUTH_TOKEN_DEFAULT = "test-token-secret"
os.environ.setdefault("API_AUTH_TOKEN", _API_AUTH_TOKEN_DEFAULT)


@pytest.fixture(autouse=True)
def _restore_api_auth_token():
    """Ensure API_AUTH_TOKEN is set before and after every unit test."""
    os.environ.setdefault("API_AUTH_TOKEN", _API_AUTH_TOKEN_DEFAULT)
    yield
    if "API_AUTH_TOKEN" not in os.environ:
        os.environ["API_AUTH_TOKEN"] = _API_AUTH_TOKEN_DEFAULT
