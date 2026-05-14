"""Tests for session registry TTL cleanup in dependencies."""

import time
from unittest.mock import MagicMock


class TestCleanupExpiredSessions:
    """Tests for cleanup_expired_sessions()."""

    def setup_method(self):
        """Clear registry before each test."""
        from src.api.dependencies import get_session_registry

        registry = get_session_registry()
        registry.clear()

    def teardown_method(self):
        """Clear registry after each test."""
        from src.api.dependencies import get_session_registry

        registry = get_session_registry()
        registry.clear()

    def test_removes_sessions_older_than_ttl(self):
        from src.api.dependencies import cleanup_expired_sessions, get_session_registry

        registry = get_session_registry()
        old_ts = time.time() - 25 * 3600  # 25 hours ago — expired
        registry["old_session"] = (MagicMock(), old_ts)

        removed = cleanup_expired_sessions(ttl_hours=24)

        assert removed == 1
        assert "old_session" not in registry

    def test_keeps_sessions_within_ttl(self):
        from src.api.dependencies import cleanup_expired_sessions, get_session_registry

        registry = get_session_registry()
        recent_ts = time.time() - 1 * 3600  # 1 hour ago — not expired
        registry["fresh_session"] = (MagicMock(), recent_ts)

        removed = cleanup_expired_sessions(ttl_hours=24)

        assert removed == 0
        assert "fresh_session" in registry

    def test_returns_count_of_removed_sessions(self):
        from src.api.dependencies import cleanup_expired_sessions, get_session_registry

        registry = get_session_registry()
        old_ts = time.time() - 48 * 3600  # 48 hours ago
        registry["expired_1"] = (MagicMock(), old_ts)
        registry["expired_2"] = (MagicMock(), old_ts)
        registry["alive"] = (MagicMock(), time.time())

        removed = cleanup_expired_sessions(ttl_hours=24)

        assert removed == 2
        assert "expired_1" not in registry
        assert "expired_2" not in registry
        assert "alive" in registry

    def test_returns_zero_when_registry_empty(self):
        from src.api.dependencies import cleanup_expired_sessions

        removed = cleanup_expired_sessions(ttl_hours=24)

        assert removed == 0

    def test_custom_ttl_hours(self):
        from src.api.dependencies import cleanup_expired_sessions, get_session_registry

        registry = get_session_registry()
        two_hours_ago = time.time() - 2 * 3600
        registry["session_a"] = (MagicMock(), two_hours_ago)

        # 3-hour TTL: 2 hours ago is not expired
        removed = cleanup_expired_sessions(ttl_hours=3)
        assert removed == 0
        assert "session_a" in registry

        # 1-hour TTL: 2 hours ago is expired
        removed = cleanup_expired_sessions(ttl_hours=1)
        assert removed == 1
        assert "session_a" not in registry
