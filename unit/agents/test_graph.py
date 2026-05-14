"""
Unit tests for the graph module.

Tests cover:
- create_checkpointer factory
- get_session_config helper
- invoke_graph and invoke_graph_sync helpers (with mocked graph)
- get_session_history retrieval
- Module constants

Note: Full graph creation tests require actual LLM integration and are
better suited for integration tests. These unit tests focus on the helper
functions and utilities that can be tested in isolation.

Run with: pytest tests/unit/agents/test_graph.py -v
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from src.agents.graph import (
    NODE_GENERATE,
    NODE_TOOLS,
    clear_session_history,
    create_checkpointer,
    get_session_config,
    get_session_history,
    invoke_graph,
    invoke_graph_sync,
)

# =============================================================================
# Test: Module Constants
# =============================================================================


class TestConstants:
    """Tests for module constants."""

    def test_node_generate_constant(self):
        """NODE_GENERATE should be the generate node name."""
        assert NODE_GENERATE == "generate"

    def test_node_tools_constant(self):
        """NODE_TOOLS should be the tools node name."""
        assert NODE_TOOLS == "tools"

    def test_constants_are_strings(self):
        """Constants should be strings."""
        assert isinstance(NODE_GENERATE, str)
        assert isinstance(NODE_TOOLS, str)


# =============================================================================
# Test: create_checkpointer
# =============================================================================


class TestCreateCheckpointer:
    """Tests for create_checkpointer factory."""

    def test_returns_memory_saver(self):
        """Should return a MemorySaver instance."""
        checkpointer = create_checkpointer()
        assert isinstance(checkpointer, MemorySaver)

    def test_returns_new_instance_each_call(self):
        """Should return a new instance each call."""
        cp1 = create_checkpointer()
        cp2 = create_checkpointer()
        assert cp1 is not cp2

    def test_memory_saver_is_functional(self):
        """MemorySaver should have expected interface."""
        checkpointer = create_checkpointer()
        # MemorySaver should have these methods
        assert hasattr(checkpointer, "put")
        assert hasattr(checkpointer, "get")


# =============================================================================
# Test: get_session_config
# =============================================================================


class TestGetSessionConfig:
    """Tests for get_session_config helper."""

    def test_returns_config_dict(self):
        """Should return a config dict with thread_id."""
        config = get_session_config("session-123")
        assert isinstance(config, dict)
        assert "configurable" in config
        assert "thread_id" in config["configurable"]

    def test_uses_session_id_as_thread_id(self):
        """Should use session_id as the thread_id."""
        config = get_session_config("my-session-abc")
        assert config["configurable"]["thread_id"] == "my-session-abc"

    def test_handles_uuid_format(self):
        """Should handle UUID-style session IDs."""
        uuid = "550e8400-e29b-41d4-a716-446655440000"
        config = get_session_config(uuid)
        assert config["configurable"]["thread_id"] == uuid

    def test_handles_simple_string(self):
        """Should handle simple string session IDs."""
        config = get_session_config("user-123")
        assert config["configurable"]["thread_id"] == "user-123"

    def test_handles_special_characters(self):
        """Should handle session IDs with special characters."""
        config = get_session_config("session_with_underscore-and-dash")
        assert config["configurable"]["thread_id"] == "session_with_underscore-and-dash"

    def test_handles_empty_string(self):
        """Should handle empty string (edge case)."""
        config = get_session_config("")
        assert config["configurable"]["thread_id"] == ""


# =============================================================================
# Test: invoke_graph (with mocked graph)
# =============================================================================


class TestInvokeGraph:
    """Tests for invoke_graph async helper."""

    @pytest.mark.asyncio
    async def test_converts_string_to_human_message(self):
        """Should convert string message to HumanMessage."""
        mock_graph = MagicMock()
        mock_graph.ainvoke = AsyncMock(
            return_value={"messages": [HumanMessage(content="Hello"), AIMessage(content="Hi")]}
        )

        with patch("builtins.print"):  # Suppress print
            await invoke_graph(mock_graph, "Hello", "session-123")

        call_args = mock_graph.ainvoke.call_args
        input_state = call_args[0][0]
        assert isinstance(input_state["messages"][0], HumanMessage)
        assert input_state["messages"][0].content == "Hello"

    @pytest.mark.asyncio
    async def test_accepts_human_message_directly(self):
        """Should accept HumanMessage directly."""
        mock_graph = MagicMock()
        mock_graph.ainvoke = AsyncMock(
            return_value={"messages": [HumanMessage(content="Hello"), AIMessage(content="Hi")]}
        )

        message = HumanMessage(content="Test message")

        with patch("builtins.print"):
            await invoke_graph(mock_graph, message, "session-123")

        call_args = mock_graph.ainvoke.call_args
        input_state = call_args[0][0]
        assert input_state["messages"][0] is message

    @pytest.mark.asyncio
    async def test_uses_session_id_as_thread_id(self):
        """Should use session_id as thread_id in config."""
        mock_graph = MagicMock()
        mock_graph.ainvoke = AsyncMock(return_value={"messages": [AIMessage(content="Hi")]})

        with patch("builtins.print"):
            await invoke_graph(mock_graph, "Hello", "my-session")

        call_args = mock_graph.ainvoke.call_args
        config = call_args[1].get("config") or call_args[0][1]
        assert config["configurable"]["thread_id"] == "my-session"

    @pytest.mark.asyncio
    async def test_returns_graph_result(self):
        """Should return the graph result."""
        expected_result = {
            "messages": [HumanMessage(content="Hello"), AIMessage(content="Hi there!")]
        }
        mock_graph = MagicMock()
        mock_graph.ainvoke = AsyncMock(return_value=expected_result)

        with patch("builtins.print"):
            result = await invoke_graph(mock_graph, "Hello", "session-123")

        assert result == expected_result

    @pytest.mark.asyncio
    async def test_sets_session_id_in_state(self):
        """Should set session_id in the input state."""
        mock_graph = MagicMock()
        mock_graph.ainvoke = AsyncMock(return_value={"messages": [AIMessage(content="Hi")]})

        with patch("builtins.print"):
            await invoke_graph(mock_graph, "Hello", "test-session-xyz")

        call_args = mock_graph.ainvoke.call_args
        input_state = call_args[0][0]
        assert input_state["session_id"] == "test-session-xyz"


# =============================================================================
# Test: invoke_graph_sync
# =============================================================================


class TestInvokeGraphSync:
    """Tests for invoke_graph_sync helper."""

    def test_converts_string_to_human_message(self):
        """Should convert string message to HumanMessage."""
        mock_graph = MagicMock()
        mock_graph.invoke = MagicMock(
            return_value={"messages": [HumanMessage(content="Hello"), AIMessage(content="Hi")]}
        )

        invoke_graph_sync(mock_graph, "Hello", "session-123")

        call_args = mock_graph.invoke.call_args
        input_state = call_args[0][0]
        assert isinstance(input_state["messages"][0], HumanMessage)

    def test_returns_graph_result(self):
        """Should return the graph result."""
        expected_result = {
            "messages": [HumanMessage(content="Hello"), AIMessage(content="Response")]
        }
        mock_graph = MagicMock()
        mock_graph.invoke = MagicMock(return_value=expected_result)

        result = invoke_graph_sync(mock_graph, "Hello", "session-123")

        assert result == expected_result

    def test_uses_session_id_as_thread_id(self):
        """Should use session_id as thread_id in config."""
        mock_graph = MagicMock()
        mock_graph.invoke = MagicMock(return_value={"messages": []})

        invoke_graph_sync(mock_graph, "Hello", "sync-session")

        call_args = mock_graph.invoke.call_args
        config = call_args[1].get("config") or call_args[0][1]
        assert config["configurable"]["thread_id"] == "sync-session"

    def test_sets_session_id_in_state(self):
        """Should set session_id in the input state."""
        mock_graph = MagicMock()
        mock_graph.invoke = MagicMock(return_value={"messages": []})

        invoke_graph_sync(mock_graph, "Hello", "sync-test-xyz")

        call_args = mock_graph.invoke.call_args
        input_state = call_args[0][0]
        assert input_state["session_id"] == "sync-test-xyz"


# =============================================================================
# Test: get_session_history
# =============================================================================


class TestGetSessionHistory:
    """Tests for get_session_history function."""

    def test_returns_messages_from_state(self):
        """Should return messages from checkpointed state."""
        messages = [
            HumanMessage(content="Hello"),
            AIMessage(content="Hi there!"),
        ]

        mock_state = MagicMock()
        mock_state.values = {"messages": messages}

        mock_graph = MagicMock()
        mock_graph.get_state = MagicMock(return_value=mock_state)

        result = get_session_history(mock_graph, "session-123")

        assert result == messages

    def test_returns_empty_list_when_no_state(self):
        """Should return empty list when no state found."""
        mock_graph = MagicMock()
        mock_graph.get_state = MagicMock(return_value=None)

        result = get_session_history(mock_graph, "nonexistent-session")

        assert result == []

    def test_returns_empty_list_when_no_messages(self):
        """Should return empty list when state has no messages."""
        mock_state = MagicMock()
        mock_state.values = {}

        mock_graph = MagicMock()
        mock_graph.get_state = MagicMock(return_value=mock_state)

        result = get_session_history(mock_graph, "session-123")

        assert result == []

    def test_handles_exception_gracefully(self):
        """Should handle exceptions and return empty list."""
        mock_graph = MagicMock()
        mock_graph.get_state = MagicMock(side_effect=Exception("Database error"))

        result = get_session_history(mock_graph, "session-123")

        assert result == []

    def test_uses_session_id_for_lookup(self):
        """Should use session_id for state lookup."""
        mock_state = MagicMock()
        mock_state.values = {"messages": []}

        mock_graph = MagicMock()
        mock_graph.get_state = MagicMock(return_value=mock_state)

        get_session_history(mock_graph, "my-session-id")

        call_args = mock_graph.get_state.call_args[0][0]
        assert call_args["configurable"]["thread_id"] == "my-session-id"

    def test_returns_empty_list_when_state_values_is_none(self):
        """Should handle state.values being None."""
        mock_state = MagicMock()
        mock_state.values = None

        mock_graph = MagicMock()
        mock_graph.get_state = MagicMock(return_value=mock_state)

        result = get_session_history(mock_graph, "session-123")

        assert result == []

    def test_preserves_message_order(self):
        """Should preserve the order of messages."""
        messages = [
            HumanMessage(content="First"),
            AIMessage(content="Second"),
            HumanMessage(content="Third"),
            AIMessage(content="Fourth"),
        ]

        mock_state = MagicMock()
        mock_state.values = {"messages": messages}

        mock_graph = MagicMock()
        mock_graph.get_state = MagicMock(return_value=mock_state)

        result = get_session_history(mock_graph, "session-123")

        assert len(result) == 4
        assert result[0].content == "First"
        assert result[3].content == "Fourth"


# =============================================================================
# Test: clear_session_history
# =============================================================================


class TestClearSessionHistory:
    """Tests for clear_session_history."""

    def test_returns_false_when_no_checkpointer(self):
        """Should return False when graph has no checkpointer attribute."""
        graph = MagicMock(spec=[])  # no attributes at all
        assert clear_session_history(graph, "sess-1") is False

    def test_returns_false_when_checkpointer_is_none(self):
        """Should return False when graph.checkpointer is None."""
        graph = MagicMock()
        graph.checkpointer = None
        assert clear_session_history(graph, "sess-1") is False

    def test_calls_delete_thread_and_returns_true(self):
        """Should call delete_thread with the session_id and return True."""
        checkpointer = MagicMock()
        graph = MagicMock()
        graph.checkpointer = checkpointer

        result = clear_session_history(graph, "sess-42")

        checkpointer.delete_thread.assert_called_once_with("sess-42")
        assert result is True

    def test_propagates_exception_from_delete_thread(self):
        """Exceptions from delete_thread should propagate to the caller."""
        checkpointer = MagicMock()
        checkpointer.delete_thread.side_effect = RuntimeError("DB error")
        graph = MagicMock()
        graph.checkpointer = checkpointer

        with pytest.raises(RuntimeError, match="DB error"):
            clear_session_history(graph, "sess-err")
