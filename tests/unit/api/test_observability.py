"""Tests for observability middleware."""

import pytest


class TestObservabilityMiddleware:
    """Tests for request logging and metrics."""

    @pytest.mark.asyncio
    async def test_health_endpoint_has_request_id_header(self, client):
        response = await client.get("/health")
        assert response.status_code == 200
        assert "x-request-id" in response.headers

    @pytest.mark.asyncio
    async def test_request_id_is_unique(self, client):
        r1 = await client.get("/health")
        r2 = await client.get("/health")
        assert r1.headers["x-request-id"] != r2.headers["x-request-id"]

    @pytest.mark.asyncio
    async def test_response_includes_timing_header(self, client):
        response = await client.get("/health")
        assert "x-response-time-ms" in response.headers
