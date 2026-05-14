"""Tests for dev test harness endpoint."""

import pytest


class TestDevHarness:
    """Tests for GET /dev/test."""

    @pytest.mark.asyncio
    async def test_harness_returns_html(self, client):
        response = await client.get("/dev/test")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "Test Harness" in response.text
        assert "Health Checks" in response.text

    @pytest.mark.asyncio
    async def test_root_redirects_to_harness(self, client):
        response = await client.get("/", follow_redirects=False)
        assert response.status_code == 302
        assert response.headers["location"] == "/dev/test"


class TestLegacyJavaProxy:
    """Tests for legacy-java hostname in proxy allowlist."""

    @pytest.mark.asyncio
    async def test_legacy_java_in_default_allowlist(self, client):
        """The legacy-java docker hostname should be allowed by default."""
        from src.api.routes.dev import _allowed_proxy_hosts
        hosts = _allowed_proxy_hosts()
        assert "legacy-java" in hosts
