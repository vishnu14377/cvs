"""Tests for health check endpoints."""

import pytest


class TestHealthEndpoints:
    """Tests for /health and /health/ready."""

    @pytest.mark.asyncio
    async def test_health_returns_ok(self, client):
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"

    @pytest.mark.asyncio
    async def test_health_ready_returns_status(self, client):
        response = await client.get("/health/ready")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ("ready", "degraded")
        assert "dependencies" in data

    @pytest.mark.asyncio
    async def test_authenticated_endpoint_rejects_no_token(self, client):
        response = await client.post("/api/v1/sessions", json={"gcs_uris": ["gs://bucket/doc.pdf"]})
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_authenticated_endpoint_rejects_bad_token(self, client):
        response = await client.post(
            "/api/v1/sessions",
            json={"gcs_uris": ["gs://bucket/doc.pdf"]},
            headers={"Authorization": "Bearer wrong-token"},
        )
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_health_does_not_require_auth(self, client):
        response = await client.get("/health")
        assert response.status_code == 200
