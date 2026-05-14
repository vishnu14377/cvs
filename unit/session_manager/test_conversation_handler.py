"""
Unit tests for the conversation_handler module.

Tests cover:
- Input validation (empty / whitespace session IDs)
- Happy path — extracting human and AI messages
- Filtering out ToolMessage, SystemMessage, and other non-conversation types
- Content normalisation (plain string, list of blocks, dict blocks)
- Empty history / missing state
- Error handling when checkpointer read fails

All external dependencies (graph, checkpointer state) are mocked.

Run with: pytest tests/unit/session_manager/test_conversation_handler.py -v
"""

from unittest.mock import MagicMock, patch

import pytest
from src.session_manager.conversation_handler import (
    _extract_content,
    _format_messages,
    get_conversation_history,
)

# =============================================================================
# Patch targets
# =============================================================================

_CH = "src.session_manager.conversation_handler"


# =============================================================================
# Helpers — fake LangChain message objects
# =============================================================================


def _make_msg(cls_name: str, content):
    """Create a MagicMock that passes isinstance checks for the given class."""
    from langchain_core.messages import AIMessage, HumanMessage

    if cls_name == "human":
        msg = HumanMessage(content=content)
    elif cls_name == "ai":
        msg = AIMessage(content=content)
    else:
        # For ToolMessage, SystemMessage, etc. — just a mock with .content
        msg = MagicMock()
        msg.content = content
        msg.type = cls_name
    return msg


# =============================================================================
# Test: Input validation
# =============================================================================


class TestInputValidation:
    """Tests for session_id validation."""

    def test_empty_session_id_raises(self):
        with pytest.raises(ValueError, match="session_id must not be empty"):
            get_conversation_history(MagicMock(), "")

    def test_whitespace_only_session_id_raises(self):
        with pytest.raises(ValueError, match="session_id must not be empty"):
            get_conversation_history(MagicMock(), "   ")

    @patch(f"{_CH}._get_raw_messages", return_value=[])
    def test_strips_whitespace(self, mock_raw):
        result = get_conversation_history(MagicMock(), "  adr-123  ")
        assert result["session_id"] == "adr-123"


# =============================================================================
# Test: Happy path
# =============================================================================


class TestHappyPath:
    """Tests for normal conversation retrieval."""

    @patch(f"{_CH}._get_raw_messages")
    def test_returns_human_and_ai_messages(self, mock_raw):
        mock_raw.return_value = [
            _make_msg("human", "What is the diagnosis?"),
            _make_msg("ai", "The diagnosis is hypertension."),
            _make_msg("human", "What medications?"),
            _make_msg("ai", "Lisinopril 10mg daily."),
        ]

        result = get_conversation_history(MagicMock(), "adr-123")

        assert result["session_id"] == "adr-123"
        assert result["message_count"] == 4
        assert len(result["messages"]) == 4
        assert result["messages"][0] == {"role": "human", "content": "What is the diagnosis?"}
        assert result["messages"][1] == {"role": "ai", "content": "The diagnosis is hypertension."}
        assert result["messages"][2] == {"role": "human", "content": "What medications?"}
        assert result["messages"][3] == {"role": "ai", "content": "Lisinopril 10mg daily."}

    @patch(f"{_CH}._get_raw_messages", return_value=[])
    def test_empty_history(self, mock_raw):
        result = get_conversation_history(MagicMock(), "adr-123")

        assert result["session_id"] == "adr-123"
        assert result["message_count"] == 0
        assert result["messages"] == []

    @patch(f"{_CH}._get_raw_messages")
    def test_single_human_message(self, mock_raw):
        mock_raw.return_value = [_make_msg("human", "Hello")]

        result = get_conversation_history(MagicMock(), "adr-123")

        assert result["message_count"] == 1
        assert result["messages"][0] == {"role": "human", "content": "Hello"}


# =============================================================================
# Test: Filtering non-conversation messages
# =============================================================================


class TestFiltering:
    """Tests that ToolMessage, SystemMessage, etc. are excluded."""

    @patch(f"{_CH}._get_raw_messages")
    def test_filters_out_tool_messages(self, mock_raw):
        mock_raw.return_value = [
            _make_msg("human", "What is the diagnosis?"),
            _make_msg("ai", "Let me search for that."),
            _make_msg("tool", "search results here"),  # should be excluded
            _make_msg("ai", "The diagnosis is hypertension."),
        ]

        result = get_conversation_history(MagicMock(), "adr-123")

        assert result["message_count"] == 3
        roles = [m["role"] for m in result["messages"]]
        assert roles == ["human", "ai", "ai"]

    @patch(f"{_CH}._get_raw_messages")
    def test_filters_out_system_messages(self, mock_raw):
        mock_raw.return_value = [
            _make_msg("system", "You are a medical AI."),  # should be excluded
            _make_msg("human", "Hello"),
            _make_msg("ai", "Hi there!"),
        ]

        result = get_conversation_history(MagicMock(), "adr-123")

        assert result["message_count"] == 2
        assert result["messages"][0]["role"] == "human"
        assert result["messages"][1]["role"] == "ai"

    @patch(f"{_CH}._get_raw_messages")
    def test_only_tool_messages_returns_empty(self, mock_raw):
        mock_raw.return_value = [
            _make_msg("tool", "result 1"),
            _make_msg("tool", "result 2"),
        ]

        result = get_conversation_history(MagicMock(), "adr-123")

        assert result["message_count"] == 0
        assert result["messages"] == []


# =============================================================================
# Test: Content normalisation
# =============================================================================


class TestContentNormalisation:
    """Tests for _extract_content handling various content formats."""

    def test_plain_string(self):
        assert _extract_content("Hello world") == "Hello world"

    def test_empty_string(self):
        assert _extract_content("") == ""

    def test_list_of_strings(self):
        result = _extract_content(["Part 1", "Part 2"])
        assert result == "Part 1\nPart 2"

    def test_list_of_dicts_with_text_key(self):
        result = _extract_content([{"text": "Block 1"}, {"text": "Block 2"}])
        assert result == "Block 1\nBlock 2"

    def test_list_with_mixed_types(self):
        result = _extract_content(["plain", {"text": "dict block"}, 42])
        assert "plain" in result
        assert "dict block" in result
        assert "42" in result

    def test_non_string_non_list_fallback(self):
        result = _extract_content(12345)
        assert result == "12345"

    @patch(f"{_CH}._get_raw_messages")
    def test_list_content_in_ai_message(self, mock_raw):
        """AI message with list content should be normalised to a string."""
        mock_raw.return_value = [
            _make_msg("ai", [{"text": "The diagnosis"}, {"text": "is hypertension."}]),
        ]

        result = get_conversation_history(MagicMock(), "adr-123")

        assert result["message_count"] == 1
        assert result["messages"][0]["content"] == "The diagnosis\nis hypertension."


# =============================================================================
# Test: _get_raw_messages error handling
# =============================================================================


class TestGetRawMessages:
    """Tests for the internal _get_raw_messages helper."""

    def test_returns_messages_from_state(self):
        mock_graph = MagicMock()
        mock_state = MagicMock()
        mock_state.values = {"messages": ["msg1", "msg2"]}
        mock_graph.get_state.return_value = mock_state

        from src.session_manager.conversation_handler import _get_raw_messages

        result = _get_raw_messages(mock_graph, "adr-123")

        assert result == ["msg1", "msg2"]

    def test_returns_empty_list_when_no_state(self):
        mock_graph = MagicMock()
        mock_graph.get_state.return_value = None

        from src.session_manager.conversation_handler import _get_raw_messages

        result = _get_raw_messages(mock_graph, "adr-123")

        assert result == []

    def test_returns_empty_list_when_no_values(self):
        mock_graph = MagicMock()
        mock_state = MagicMock()
        mock_state.values = {}
        mock_graph.get_state.return_value = mock_state

        from src.session_manager.conversation_handler import _get_raw_messages

        result = _get_raw_messages(mock_graph, "adr-123")

        assert result == []

    def test_returns_empty_list_on_exception(self):
        mock_graph = MagicMock()
        mock_graph.get_state.side_effect = RuntimeError("checkpointer down")

        from src.session_manager.conversation_handler import _get_raw_messages

        result = _get_raw_messages(mock_graph, "adr-123")

        assert result == []


# =============================================================================
# Test: _format_messages
# =============================================================================


class TestFormatMessages:
    """Tests for the internal _format_messages helper."""

    def test_empty_list(self):
        assert _format_messages([]) == []

    def test_preserves_order(self):
        msgs = [
            _make_msg("human", "Q1"),
            _make_msg("ai", "A1"),
            _make_msg("human", "Q2"),
            _make_msg("ai", "A2"),
        ]
        result = _format_messages(msgs)
        assert [m["role"] for m in result] == ["human", "ai", "human", "ai"]
        assert [m["content"] for m in result] == ["Q1", "A1", "Q2", "A2"]
