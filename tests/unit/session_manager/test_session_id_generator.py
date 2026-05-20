"""
Unit tests for session_id_generator.

Tests cover:
- Format of generated session IDs
- Uniqueness across calls
- Date component correctness

Run with: pytest tests/unit/session_manager/test_session_id_generator.py -v
"""

import re
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from src.session_manager.core.session_id_generator import generate_session_id

# =============================================================================
# Test: Format
# =============================================================================


class TestSessionIdFormat:
    """Tests for the format of generated session IDs."""

    def test_matches_expected_pattern(self):
        """Should match the format adr-YYYYMMDD-<8 hex chars>."""
        session_id = generate_session_id()
        assert re.fullmatch(r"adr-\d{8}-[0-9a-f]{8}", session_id)

    def test_starts_with_adr_prefix(self):
        """Should start with 'adr-'."""
        session_id = generate_session_id()
        assert session_id.startswith("adr-")

    def test_has_three_parts(self):
        """Should have exactly three dash-separated parts."""
        parts = generate_session_id().split("-")
        assert len(parts) == 3

    def test_date_component_is_today(self):
        """The date component should be today's UTC date."""
        session_id = generate_session_id()
        date_part = session_id.split("-")[1]
        expected_date = datetime.now(timezone.utc).strftime("%Y%m%d")
        assert date_part == expected_date

    def test_uuid_component_is_8_hex_chars(self):
        """The UUID suffix should be exactly 8 hex characters."""
        session_id = generate_session_id()
        uuid_part = session_id.split("-")[2]
        assert len(uuid_part) == 8
        assert all(c in "0123456789abcdef" for c in uuid_part)


# =============================================================================
# Test: Uniqueness
# =============================================================================


class TestSessionIdUniqueness:
    """Tests that generated IDs are unique."""

    def test_successive_calls_produce_different_ids(self):
        """Two consecutive calls should produce different IDs."""
        id1 = generate_session_id()
        id2 = generate_session_id()
        assert id1 != id2

    def test_many_ids_are_unique(self):
        """100 generated IDs should all be distinct."""
        ids = {generate_session_id() for _ in range(100)}
        assert len(ids) == 100


# =============================================================================
# Test: Deterministic date via mock
# =============================================================================


class TestSessionIdWithMockedDate:
    """Tests with a fixed UUID to verify deterministic output."""

    @patch("src.session_manager.core.session_id_generator.uuid")
    def test_uuid_component_is_deterministic(self, mock_uuid):
        """With a fixed UUID, the suffix should be predictable."""
        mock_uuid.uuid4.return_value = MagicMock(hex="aabbccdd11223344")

        session_id = generate_session_id()
        uuid_part = session_id.split("-")[2]
        assert uuid_part == "aabbccdd"
