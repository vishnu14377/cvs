"""Playwright E2E tests for the ADR AI Agent chat page (/chat/{session_id}).

Prerequisites:
    docker-compose up --build
    bash docker/seed-gcs.sh

Run:
    .venv/bin/python -m pytest tests/e2e/test_chat_e2e.py -v
    .venv/bin/python -m pytest tests/e2e/test_chat_e2e.py -v --headed
"""

from __future__ import annotations

import pytest

# Playwright is an opt-in dev dep (see [project.optional-dependencies] e2e).
# Skip collection when it isn't installed so CI's pytest run isn't blocked.
pytest.importorskip("playwright")

import httpx
from playwright.sync_api import Page, expect

from tests.e2e.conftest import AUTH_TOKEN, BASE_URL, SAMPLE_ADR_URI, skip_if_no_api


@pytest.fixture(scope="module")
def chat_session_id():
    """Create a real session for chat page tests; delete it after the module finishes.

    Session creation can take 10–30 seconds (OCR + embedding).
    """
    headers = {"Authorization": f"Bearer {AUTH_TOKEN}"}
    session_id = None
    with httpx.Client(base_url=BASE_URL, timeout=120) as client:
        r = client.post(
            "/api/v1/sessions",
            json={
                "gcs_uris": [SAMPLE_ADR_URI],
                "ocr_engine": "mistral",
                "metadata": {"test": "e2e-chat"},
            },
            headers=headers,
        )
        if r.status_code == 201:
            session_id = r.json().get("session_id")

        yield session_id

        # Teardown: delete session if it was created
        if session_id:
            client.delete(f"/api/v1/sessions/{session_id}", headers=headers)


@skip_if_no_api
@pytest.mark.e2e
def test_chat_page_loads(page: Page, base_url_val: str, chat_session_id: str):
    """Chat page renders the ADR AI Assistant UI for a valid session."""
    if not chat_session_id:
        pytest.skip("Session creation failed — skipping chat page test")

    page.goto(base_url_val + f"/chat/{chat_session_id}")

    # Header
    chat_header = page.locator("#chat-header")
    expect(chat_header).to_be_visible()
    expect(chat_header).to_contain_text("ADR AI Assistant")

    # Message container
    expect(page.locator("#chat-messages")).to_be_visible()

    # Input field
    chat_input = page.locator("#chat-input")
    expect(chat_input).to_be_visible()

    # Send button
    send_btn = page.locator("#send-btn")
    expect(send_btn).to_be_visible()


@skip_if_no_api
@pytest.mark.e2e
def test_chat_page_404_invalid_session(page: Page, base_url_val: str):
    """Navigating to /chat/nonexistent returns a 404 response."""
    response = page.goto(base_url_val + "/chat/nonexistent-session-id-that-does-not-exist")
    assert response is not None, "Expected a response object from page.goto"
    assert response.status == 404, f"Expected HTTP 404 for invalid session, got {response.status}"
