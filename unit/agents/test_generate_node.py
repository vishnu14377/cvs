"""
Unit tests for the generate_node module.

Tests cover:
- Message filtering by turns (_filter_messages_by_turns)
- Message preparation (_prepare_messages)
- Generate node factory (create_generate_node)
- Generate node execution (generate_node, generate_node_sync)
- Module constants

Run with: pytest tests/unit/agents/test_generate_node.py -v
"""

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from src.agents.generate_node import (
    DEFAULT_MAX_TURNS,
    DEFAULT_SYSTEM_PROMPT,
    _filter_messages_by_turns,
    _prepare_messages,
    create_generate_node,
    generate_node,
    generate_node_sync,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_llm():
    """Create a mock LLM client."""
    mock = MagicMock()
    mock.ainvoke = AsyncMock(return_value=AIMessage(content="Mock response"))
    mock.invoke = MagicMock(return_value=AIMessage(content="Mock response"))
    mock.bind_tools = MagicMock(return_value=mock)
    return mock


# =============================================================================
# Test: _filter_messages_by_turns
# =============================================================================


class TestFilterMessagesByTurns:
    """Tests for _filter_messages_by_turns function."""

    def test_empty_messages(self):
        """Should return empty list for empty input."""
        result = _filter_messages_by_turns([])
        assert result == []

    def test_single_human_message(self):
        """Should keep single message."""
        messages = [HumanMessage(content="Hello")]
        result = _filter_messages_by_turns(messages, max_turns=5)
        assert len(result) == 1
        assert result[0].content == "Hello"

    def test_preserves_system_messages(self):
        """Should always preserve system messages regardless of turn limit."""
        messages = [
            SystemMessage(content="System prompt"),
            HumanMessage(content="Turn 1"),
            AIMessage(content="Response 1"),
            HumanMessage(content="Turn 2"),
            AIMessage(content="Response 2"),
        ]
        result = _filter_messages_by_turns(messages, max_turns=1)

        # Should have system message + last turn (2 messages)
        system_msgs = [m for m in result if isinstance(m, SystemMessage)]
        assert len(system_msgs) == 1
        assert system_msgs[0].content == "System prompt"

    def test_limits_turns_correctly(self):
        """Should limit to max_turns most recent turns."""
        messages = [
            HumanMessage(content="Turn 1"),
            AIMessage(content="Response 1"),
            HumanMessage(content="Turn 2"),
            AIMessage(content="Response 2"),
            HumanMessage(content="Turn 3"),
            AIMessage(content="Response 3"),
        ]
        result = _filter_messages_by_turns(messages, max_turns=2)

        # Should keep only last 2 turns (4 messages)
        assert len(result) == 4
        assert result[0].content == "Turn 2"
        assert result[1].content == "Response 2"
        assert result[2].content == "Turn 3"
        assert result[3].content == "Response 3"

    def test_keeps_all_when_under_limit(self):
        """Should keep all messages when under the limit."""
        messages = [
            HumanMessage(content="Turn 1"),
            AIMessage(content="Response 1"),
        ]
        result = _filter_messages_by_turns(messages, max_turns=10)
        assert len(result) == 2

    def test_handles_tool_messages_in_turn(self):
        """Should include tool messages as part of a turn."""
        messages = [
            HumanMessage(content="Question"),
            AIMessage(content="", tool_calls=[{"name": "search", "args": {}, "id": "1"}]),
            ToolMessage(content="Tool result", tool_call_id="1"),
            AIMessage(content="Final answer"),
        ]
        result = _filter_messages_by_turns(messages, max_turns=1)

        # All messages belong to one turn
        assert len(result) == 4

    def test_only_system_messages(self):
        """Should return only system messages if no other messages."""
        messages = [SystemMessage(content="System")]
        result = _filter_messages_by_turns(messages, max_turns=5)
        assert len(result) == 1
        assert isinstance(result[0], SystemMessage)

    def test_default_max_turns(self):
        """Should use DEFAULT_MAX_TURNS when not specified."""
        messages = [HumanMessage(content="Hello")]
        result = _filter_messages_by_turns(messages)
        assert len(result) == 1

    def test_multiple_system_messages(self):
        """Should preserve all system messages."""
        messages = [
            SystemMessage(content="System 1"),
            SystemMessage(content="System 2"),
            HumanMessage(content="Hello"),
        ]
        result = _filter_messages_by_turns(messages, max_turns=1)
        system_msgs = [m for m in result if isinstance(m, SystemMessage)]
        assert len(system_msgs) == 2


# =============================================================================
# Test: _prepare_messages
# =============================================================================


class TestPrepareMessages:
    """Tests for _prepare_messages function."""

    def test_adds_system_prompt_when_missing(self):
        """Should add system prompt if not present."""
        messages = [HumanMessage(content="Hello")]
        result = _prepare_messages(messages, "System prompt", max_turns=10)

        assert isinstance(result[0], SystemMessage)
        assert result[0].content == "System prompt"
        assert len(result) == 2

    def test_preserves_existing_system_prompt(self):
        """Should not duplicate system prompt if already present."""
        messages = [
            SystemMessage(content="Existing system"),
            HumanMessage(content="Hello"),
        ]
        result = _prepare_messages(messages, "New system", max_turns=10)

        system_msgs = [m for m in result if isinstance(m, SystemMessage)]
        assert len(system_msgs) == 1
        assert system_msgs[0].content == "Existing system"

    def test_applies_turn_filtering(self):
        """Should apply turn filtering."""
        messages = [
            HumanMessage(content="Turn 1"),
            AIMessage(content="Response 1"),
            HumanMessage(content="Turn 2"),
            AIMessage(content="Response 2"),
        ]
        result = _prepare_messages(messages, "System", max_turns=1)

        # System + 1 turn (2 messages) = 3 total
        assert len(result) == 3
        human_msgs = [m for m in result if isinstance(m, HumanMessage)]
        assert len(human_msgs) == 1
        assert human_msgs[0].content == "Turn 2"

    def test_empty_messages_adds_system_prompt(self):
        """Should add system prompt to empty message list."""
        result = _prepare_messages([], "System prompt", max_turns=10)
        assert len(result) == 1
        assert isinstance(result[0], SystemMessage)
        assert result[0].content == "System prompt"


# =============================================================================
# Test: create_generate_node
# =============================================================================


class TestCreateGenerateNode:
    """Tests for create_generate_node factory function."""

    def test_returns_callable(self):
        """Should return a callable function."""
        node = create_generate_node()
        assert callable(node)

    def test_returns_async_function(self):
        """Should return an async function."""
        import asyncio

        node = create_generate_node()
        assert asyncio.iscoroutinefunction(node)

    def test_captures_tools(self):
        """Should capture tools in closure."""
        mock_tool = MagicMock()
        mock_tool.name = "test_tool"

        node = create_generate_node(tools=[mock_tool])
        assert callable(node)

    def test_captures_system_prompt(self):
        """Should capture custom system prompt."""
        node = create_generate_node(system_prompt="Custom prompt")
        assert callable(node)

    def test_captures_max_turns(self):
        """Should capture max_turns configuration."""
        node = create_generate_node(max_turns=5)
        assert callable(node)

    def test_captures_all_parameters(self):
        """Should capture all parameters in closure."""
        mock_tool = MagicMock()
        mock_tool.name = "tool"

        node = create_generate_node(
            tools=[mock_tool],
            system_prompt="Custom",
            max_turns=3,
        )
        assert callable(node)


# =============================================================================
# Test: generate_node (async) - mocking the LangChainClient
# =============================================================================


class TestGenerateNode:
    """Tests for generate_node async function."""

    @pytest.mark.asyncio
    async def test_returns_messages_dict(self, basic_state, mock_llm):
        """Should return dict with messages key."""
        # Patch at the location where LangChainClient is imported and used
        with patch.object(
            sys.modules["src.agents.generate_node"], "LangChainClient"
        ) as mock_client_class:
            mock_client = MagicMock()
            mock_client.client = mock_llm
            mock_client_class.return_value = mock_client

            result = await generate_node(basic_state)

            assert "messages" in result
            assert len(result["messages"]) == 1
            assert isinstance(result["messages"][0], AIMessage)

    @pytest.mark.asyncio
    async def test_invokes_llm(self, basic_state, mock_llm):
        """Should invoke the LLM with prepared messages."""
        with patch.object(
            sys.modules["src.agents.generate_node"], "LangChainClient"
        ) as mock_client_class:
            mock_client = MagicMock()
            mock_client.client = mock_llm
            mock_client_class.return_value = mock_client

            await generate_node(basic_state)
            mock_llm.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_binds_tools_when_provided(self, basic_state, mock_llm):
        """Should bind tools to LLM when tools are provided."""
        mock_tool = MagicMock()
        mock_tool.name = "test_tool"

        with patch.object(
            sys.modules["src.agents.generate_node"], "LangChainClient"
        ) as mock_client_class:
            mock_client = MagicMock()
            mock_client.client = mock_llm
            mock_client_class.return_value = mock_client

            await generate_node(basic_state, tools=[mock_tool])
            mock_llm.bind_tools.assert_called_once_with([mock_tool])

    @pytest.mark.asyncio
    async def test_no_tools_no_bind(self, basic_state, mock_llm):
        """Should not bind tools when none provided."""
        with patch.object(
            sys.modules["src.agents.generate_node"], "LangChainClient"
        ) as mock_client_class:
            mock_client = MagicMock()
            mock_client.client = mock_llm
            mock_client_class.return_value = mock_client

            await generate_node(basic_state, tools=None)
            mock_llm.bind_tools.assert_not_called()

    @pytest.mark.asyncio
    async def test_uses_custom_system_prompt(self, basic_state, mock_llm):
        """Should use custom system prompt when provided."""
        custom_prompt = "Custom prompt"

        with patch.object(
            sys.modules["src.agents.generate_node"], "LangChainClient"
        ) as mock_client_class:
            mock_client = MagicMock()
            mock_client.client = mock_llm
            mock_client_class.return_value = mock_client

            await generate_node(basic_state, system_prompt=custom_prompt)

            call_args = mock_llm.ainvoke.call_args
            messages = call_args[0][0]
            system_msgs = [m for m in messages if isinstance(m, SystemMessage)]
            assert len(system_msgs) == 1
            assert system_msgs[0].content == custom_prompt

    @pytest.mark.asyncio
    async def test_uses_default_system_prompt(self, basic_state, mock_llm):
        """Should use default system prompt when none provided."""
        with patch.object(
            sys.modules["src.agents.generate_node"], "LangChainClient"
        ) as mock_client_class:
            mock_client = MagicMock()
            mock_client.client = mock_llm
            mock_client_class.return_value = mock_client

            await generate_node(basic_state)

            call_args = mock_llm.ainvoke.call_args
            messages = call_args[0][0]
            system_msgs = [m for m in messages if isinstance(m, SystemMessage)]
            assert len(system_msgs) == 1
            assert system_msgs[0].content == DEFAULT_SYSTEM_PROMPT

    @pytest.mark.asyncio
    async def test_handles_empty_messages(self, mock_llm):
        """Should handle state with empty messages."""
        state = {"messages": [], "session_id": "test"}

        with patch.object(
            sys.modules["src.agents.generate_node"], "LangChainClient"
        ) as mock_client_class:
            mock_client = MagicMock()
            mock_client.client = mock_llm
            mock_client_class.return_value = mock_client

            result = await generate_node(state)
            assert "messages" in result

    @pytest.mark.asyncio
    async def test_propagates_llm_error(self, basic_state, mock_llm):
        """Should propagate LLM errors."""
        mock_llm.ainvoke = AsyncMock(side_effect=Exception("LLM Error"))

        with patch.object(
            sys.modules["src.agents.generate_node"], "LangChainClient"
        ) as mock_client_class:
            mock_client = MagicMock()
            mock_client.client = mock_llm
            mock_client_class.return_value = mock_client

            with pytest.raises(Exception, match="LLM Error"):
                await generate_node(basic_state)

    @pytest.mark.asyncio
    async def test_handles_tool_call_response(self, basic_state, mock_llm):
        """Should handle AIMessage with tool calls."""
        tool_call_response = AIMessage(
            content="", tool_calls=[{"name": "search", "args": {"query": "test"}, "id": "call_1"}]
        )
        mock_llm.ainvoke = AsyncMock(return_value=tool_call_response)

        with patch.object(
            sys.modules["src.agents.generate_node"], "LangChainClient"
        ) as mock_client_class:
            mock_client = MagicMock()
            mock_client.client = mock_llm
            mock_client_class.return_value = mock_client

            result = await generate_node(basic_state)

            assert len(result["messages"]) == 1
            assert len(result["messages"][0].tool_calls) == 1


# =============================================================================
# Test: generate_node_sync
# =============================================================================


class TestGenerateNodeSync:
    """Tests for generate_node_sync function."""

    def test_returns_messages_dict(self, basic_state, mock_llm):
        """Should return dict with messages key."""
        with patch.object(
            sys.modules["src.agents.generate_node"], "LangChainClient"
        ) as mock_client_class:
            mock_client = MagicMock()
            mock_client.client = mock_llm
            mock_client_class.return_value = mock_client

            result = generate_node_sync(basic_state)

            assert "messages" in result
            assert len(result["messages"]) == 1

    def test_invokes_llm_sync(self, basic_state, mock_llm):
        """Should invoke the LLM synchronously."""
        with patch.object(
            sys.modules["src.agents.generate_node"], "LangChainClient"
        ) as mock_client_class:
            mock_client = MagicMock()
            mock_client.client = mock_llm
            mock_client_class.return_value = mock_client

            generate_node_sync(basic_state)
            mock_llm.invoke.assert_called_once()

    def test_binds_tools_when_provided(self, basic_state, mock_llm):
        """Should bind tools to LLM."""
        mock_tool = MagicMock()
        mock_tool.name = "test_tool"

        with patch.object(
            sys.modules["src.agents.generate_node"], "LangChainClient"
        ) as mock_client_class:
            mock_client = MagicMock()
            mock_client.client = mock_llm
            mock_client_class.return_value = mock_client

            generate_node_sync(basic_state, tools=[mock_tool])
            mock_llm.bind_tools.assert_called_once()

    def test_propagates_llm_error(self, basic_state, mock_llm):
        """Should propagate LLM errors."""
        mock_llm.invoke = MagicMock(side_effect=Exception("LLM Error"))

        with patch.object(
            sys.modules["src.agents.generate_node"], "LangChainClient"
        ) as mock_client_class:
            mock_client = MagicMock()
            mock_client.client = mock_llm
            mock_client_class.return_value = mock_client

            with pytest.raises(Exception, match="LLM Error"):
                generate_node_sync(basic_state)


# =============================================================================
# Test: Module Constants
# =============================================================================


class TestConstants:
    """Tests for module constants."""

    def test_default_max_turns_is_reasonable(self):
        """DEFAULT_MAX_TURNS should be a reasonable value."""
        assert DEFAULT_MAX_TURNS > 0
        assert DEFAULT_MAX_TURNS <= 50

    def test_default_max_turns_is_integer(self):
        """DEFAULT_MAX_TURNS should be an integer."""
        assert isinstance(DEFAULT_MAX_TURNS, int)

    def test_default_system_prompt_exists(self):
        """DEFAULT_SYSTEM_PROMPT should be non-empty."""
        assert DEFAULT_SYSTEM_PROMPT
        assert len(DEFAULT_SYSTEM_PROMPT) > 10

    def test_default_system_prompt_mentions_context(self):
        """DEFAULT_SYSTEM_PROMPT should mention relevant context."""
        prompt_lower = DEFAULT_SYSTEM_PROMPT.lower()
        # Should mention ADR or assistant or helpful
        assert any(term in prompt_lower for term in ["adr", "assistant", "helpful"])
