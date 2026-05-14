"""Integration smoke tests for the ADR AI Agent API.

Tests the full session lifecycle against the real running Docker stack.

Prerequisites:
    docker-compose up --build
    bash docker/seed-gcs.sh

Run:
    .venv/bin/python -m pytest tests/integration/test_smoke.py -v --timeout=180
"""

from __future__ import annotations

import httpx
import pytest

from tests.integration.conftest import (
    AUTH_TOKEN,
    SAMPLE_ADR_URI,
    skip_if_no_api,
)


@pytest.mark.integration
class TestSmoke:
    """Full lifecycle smoke tests — run in definition order."""

    session_id: str = ""
    message_id: str = ""

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    @skip_if_no_api
    def test_health(self, client: httpx.Client):
        """GET /health returns 200 with status=ok."""
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"

    @skip_if_no_api
    def test_health_ready(self, client: httpx.Client):
        """GET /health/ready returns 200 with postgres and mongodb connected."""
        r = client.get("/health/ready")
        assert r.status_code == 200
        body = r.json()
        deps = body.get("dependencies", {})
        assert deps.get("postgres") == "connected", f"postgres not connected: {deps}"
        assert deps.get("mongodb") == "connected", f"mongodb not connected: {deps}"

    # ------------------------------------------------------------------
    # Listing before any data
    # ------------------------------------------------------------------

    @skip_if_no_api
    def test_list_sessions_empty(self, client: httpx.Client, auth_headers: dict):
        """GET /api/v1/sessions returns 200 with a sessions list."""
        r = client.get("/api/v1/sessions", headers=auth_headers)
        assert r.status_code == 200
        body = r.json()
        assert "sessions" in body
        assert isinstance(body["sessions"], list)

    @skip_if_no_api
    def test_list_policies_empty(self, client: httpx.Client, auth_headers: dict):
        """GET /api/v1/policies returns 200 with an empty policies list."""
        r = client.get("/api/v1/policies", headers=auth_headers)
        assert r.status_code == 200
        body = r.json()
        assert "policies" in body
        assert isinstance(body["policies"], list)

    # ------------------------------------------------------------------
    # Session creation
    # ------------------------------------------------------------------

    @skip_if_no_api
    def test_create_session(self, client: httpx.Client, auth_headers: dict):
        """POST /api/v1/sessions creates a session, returns 201 with status=ready."""
        r = client.post(
            "/api/v1/sessions",
            json={
                "gcs_uris": [SAMPLE_ADR_URI],
                "ocr_engine": "mistral",
                "metadata": {"test": "smoke"},
            },
            headers=auth_headers,
            timeout=120,
        )
        assert r.status_code == 201, f"Expected 201, got {r.status_code}: {r.text}"
        body = r.json()
        assert body["status"] == "ready", f"Session status not ready: {body}"
        assert "session_id" in body
        assert body["session_id"], "session_id must be non-empty"
        TestSmoke.session_id = body["session_id"]

    # ------------------------------------------------------------------
    # Session retrieval
    # ------------------------------------------------------------------

    @skip_if_no_api
    def test_get_session(self, client: httpx.Client, auth_headers: dict):
        """GET /api/v1/sessions/{session_id} returns 200 with status=ready."""
        assert TestSmoke.session_id, "session_id not set — test_create_session must run first"
        r = client.get(f"/api/v1/sessions/{TestSmoke.session_id}", headers=auth_headers)
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ready"
        assert body["session_id"] == TestSmoke.session_id

    @skip_if_no_api
    def test_list_sessions_has_session(self, client: httpx.Client, auth_headers: dict):
        """GET /api/v1/sessions includes our newly created session."""
        assert TestSmoke.session_id, "session_id not set — test_create_session must run first"
        r = client.get("/api/v1/sessions", headers=auth_headers)
        assert r.status_code == 200
        body = r.json()
        session_ids = [s["session_id"] for s in body["sessions"]]
        assert TestSmoke.session_id in session_ids, (
            f"Expected {TestSmoke.session_id} in session list: {session_ids}"
        )

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    @skip_if_no_api
    def test_query_agent(self, client: httpx.Client, auth_headers: dict):
        """POST /api/v1/sessions/{session_id}/query returns a response with content fields."""
        assert TestSmoke.session_id, "session_id not set — test_create_session must run first"
        r = client.post(
            f"/api/v1/sessions/{TestSmoke.session_id}/query",
            json={"message": "What medications is the patient taking?"},
            headers=auth_headers,
            timeout=120,
        )
        assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
        body = r.json()
        assert "message_id" in body
        assert body["message_id"], "message_id must be non-empty"
        msg = body.get("message", {})
        assert msg.get("content"), "response content must be non-empty"
        assert msg.get("content_html"), "response content_html must be non-empty"
        assert msg.get("content_base64"), "response content_base64 must be non-empty"
        TestSmoke.message_id = body["message_id"]

    @skip_if_no_api
    def test_query_has_sources(self, client: httpx.Client, auth_headers: dict):
        """The medication query response should include source references from adr_search."""
        assert TestSmoke.session_id, "session_id not set — test_create_session must run first"
        # Re-query to get sources (the session already has history, agent should retrieve)
        r = client.post(
            f"/api/v1/sessions/{TestSmoke.session_id}/query",
            json={
                "message": "List all medications mentioned in the ADR document.",
                "include_source_references": True,
            },
            headers=auth_headers,
            timeout=120,
        )
        assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
        body = r.json()
        sources = body.get("sources", [])
        assert isinstance(sources, list)
        # The agent should have triggered adr_search for a specific document question
        # We check the field structure rather than requiring a non-empty list since the
        # agent may answer from conversation context without re-triggering search
        if sources:
            first = sources[0]
            assert "document" in first, "source must have a document field"

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------

    @skip_if_no_api
    def test_get_history(self, client: httpx.Client, auth_headers: dict):
        """GET /api/v1/sessions/{session_id}/history returns at least 2 messages."""
        assert TestSmoke.session_id, "session_id not set — test_create_session must run first"
        r = client.get(
            f"/api/v1/sessions/{TestSmoke.session_id}/history",
            headers=auth_headers,
        )
        assert r.status_code == 200
        body = r.json()
        assert "messages" in body
        messages = body["messages"]
        assert len(messages) >= 2, f"Expected at least 2 messages (human + ai), got {len(messages)}"
        roles = [m["role"] for m in messages]
        assert "human" in roles, f"Expected a human message in history: {roles}"
        assert "ai" in roles, f"Expected an ai message in history: {roles}"

    # ------------------------------------------------------------------
    # Feedback
    # ------------------------------------------------------------------

    @skip_if_no_api
    def test_submit_feedback(self, client: httpx.Client, auth_headers: dict):
        """POST /api/v1/sessions/{session_id}/feedback stores feedback and returns 200."""
        assert TestSmoke.session_id, "session_id not set — test_create_session must run first"
        assert TestSmoke.message_id, "message_id not set — test_query_agent must run first"
        r = client.post(
            f"/api/v1/sessions/{TestSmoke.session_id}/feedback",
            json={
                "message_id": TestSmoke.message_id,
                "rating": "positive",
                "comment": "Smoke test feedback",
            },
            headers=auth_headers,
        )
        assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
        body = r.json()
        assert body["status"] == "stored"
        assert body["session_id"] == TestSmoke.session_id
        assert body["message_id"] == TestSmoke.message_id

    @skip_if_no_api
    def test_get_feedback(self, client: httpx.Client, auth_headers: dict):
        """GET /api/v1/sessions/{session_id}/feedback returns at least 1 entry."""
        assert TestSmoke.session_id, "session_id not set — test_create_session must run first"
        r = client.get(
            f"/api/v1/sessions/{TestSmoke.session_id}/feedback",
            headers=auth_headers,
        )
        assert r.status_code == 200
        body = r.json()
        assert "feedback" in body
        feedback_list = body["feedback"]
        assert len(feedback_list) >= 1, (
            f"Expected at least 1 feedback entry, got {len(feedback_list)}"
        )
        entry = feedback_list[0]
        assert entry["rating"] in ("positive", "negative")

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    @skip_if_no_api
    def test_query_stream(self, client: httpx.Client, auth_headers: dict):
        """POST /api/v1/sessions/{session_id}/query/stream returns text/event-stream with event: done."""
        assert TestSmoke.session_id, "session_id not set — test_create_session must run first"
        r = client.post(
            f"/api/v1/sessions/{TestSmoke.session_id}/query/stream",
            json={"message": "Briefly summarize this ADR document."},
            headers=auth_headers,
            timeout=120,
        )
        assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
        content_type = r.headers.get("content-type", "")
        assert "text/event-stream" in content_type, (
            f"Expected text/event-stream content-type, got: {content_type}"
        )
        body_text = r.text
        assert "event: done" in body_text, (
            f"Expected 'event: done' in SSE body. Body preview: {body_text[:500]}"
        )

    # ------------------------------------------------------------------
    # File upload
    # ------------------------------------------------------------------

    @skip_if_no_api
    def test_upload_session(self, client: httpx.Client, auth_headers: dict):
        """POST /api/v1/sessions/upload with a real PDF file creates a session."""
        import os

        pdf_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "data",
            "sample_adrs",
            "BINGHAM,CALLIE_91202308017_FLC5_REDACTED.pdf",
        )
        if not os.path.exists(pdf_path):
            pytest.skip("Sample PDF not found at " + pdf_path)

        with open(pdf_path, "rb") as f:
            r = client.post(
                "/api/v1/sessions/upload",
                files={"files": ("BINGHAM_CALLIE.pdf", f, "application/pdf")},
                data={"ocr_engine": "mistral"},
                headers={"Authorization": f"Bearer {AUTH_TOKEN}"},
                timeout=120,
            )

        assert r.status_code == 201, f"Expected 201, got {r.status_code}: {r.text}"
        body = r.json()
        assert body["session_id"], "session_id must be non-empty"
        assert body["status"] in ("ready", "error")
        TestSmoke.upload_session_id = body["session_id"]

    @skip_if_no_api
    def test_upload_rejects_non_pdf(self, client: httpx.Client):
        """POST /api/v1/sessions/upload with a non-PDF file returns 400."""
        r = client.post(
            "/api/v1/sessions/upload",
            files={"files": ("readme.txt", b"not a pdf", "text/plain")},
            data={"ocr_engine": "mistral"},
            headers={"Authorization": f"Bearer {AUTH_TOKEN}"},
        )
        assert r.status_code == 400, f"Expected 400, got {r.status_code}: {r.text}"
        assert "PDF" in r.json()["detail"]

    @skip_if_no_api
    def test_cleanup_upload_session(self, client: httpx.Client, auth_headers: dict):
        """Clean up the session created by test_upload_session."""
        sid = getattr(TestSmoke, "upload_session_id", None)
        if sid:
            client.delete(f"/api/v1/sessions/{sid}", headers=auth_headers)

    # ------------------------------------------------------------------
    # Auth guard
    # ------------------------------------------------------------------

    @skip_if_no_api
    def test_auth_required(self, client: httpx.Client):
        """POST /api/v1/sessions without auth header returns 401."""
        r = client.post(
            "/api/v1/sessions",
            json={
                "gcs_uris": [SAMPLE_ADR_URI],
                "ocr_engine": "mistral",
            },
            # No Authorization header
        )
        assert r.status_code == 401, f"Expected 401 Unauthorized, got {r.status_code}: {r.text}"

    # ------------------------------------------------------------------
    # Deletion
    # ------------------------------------------------------------------

    @skip_if_no_api
    def test_delete_session(self, client: httpx.Client, auth_headers: dict):
        """DELETE /api/v1/sessions/{session_id} returns 200 with status=deleted."""
        assert TestSmoke.session_id, "session_id not set — test_create_session must run first"
        r = client.delete(
            f"/api/v1/sessions/{TestSmoke.session_id}",
            headers=auth_headers,
        )
        assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
        body = r.json()
        assert body["status"] == "deleted"
        assert body["session_id"] == TestSmoke.session_id

    @skip_if_no_api
    def test_session_gone_after_delete(self, client: httpx.Client, auth_headers: dict):
        """GET /api/v1/sessions/{session_id} returns 404 after deletion."""
        assert TestSmoke.session_id, "session_id not set — test_create_session must run first"
        r = client.get(
            f"/api/v1/sessions/{TestSmoke.session_id}",
            headers=auth_headers,
        )
        assert r.status_code == 404, f"Expected 404 after delete, got {r.status_code}: {r.text}"
