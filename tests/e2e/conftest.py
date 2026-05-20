"""Shared fixtures for Playwright E2E tests.

Requires Docker stack running:
    docker-compose up --build
    bash docker/seed-gcs.sh
"""

import os

import httpx
import pytest

BASE_URL = os.environ.get("TEST_BASE_URL", "http://localhost:8000")
AUTH_TOKEN = os.environ.get("TEST_AUTH_TOKEN", "dev-token-12345")
SAMPLE_ADR_URI = "gs://adr-ai-agent-dev/adr_ai_agent/BINGHAM_CALLIE_FLC5_REDACTED.pdf"


def is_api_reachable():
    try:
        r = httpx.get(f"{BASE_URL}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


skip_if_no_api = pytest.mark.skipif(
    not is_api_reachable(),
    reason="API not reachable — run docker-compose up first",
)


@pytest.fixture(scope="session")
def base_url_val():
    return BASE_URL


@pytest.fixture(scope="session")
def auth_token():
    return AUTH_TOKEN
