"""
Unit tests for the tool_node module.

Tests cover:
- Re-exports from LangGraph (ToolNode, tools_condition)
- Helper functions (_get_last_ai_message, _create_no_tools_response)
- Session ID injection (_inject_session_id)
- Inject-session graph node (create_inject_session_node)
- Fallback tool node when no tools available
- create_tool_node factory function
- tools_condition routing

Run with: pytest tests/unit/agents/test_tool_node.py -v
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from src.agents.tool_node import (
    NO_TOOLS_MESSAGE,
    ToolNode,
    _create_no_tools_response,
    _get_last_ai_message,
    _inject_session_id,
    _inject_session_id_node,
    create_inject_session_node,
    create_tool_node,
    tools_condition,
)

# =============================================================================
# Real Tools for Testing
# =============================================================================


@tool
def search_tool(query: str) -> str:
    """Search for documents matching the query."""
    return f"Search results for: {query}"


@tool
def calculator_tool(expression: str) -> str:
    """Evaluate a mathematical expression."""
    return f"Result: {eval(expression)}"


# =============================================================================
# Test: Re-exports
# =============================================================================


class TestReExports:
    """Tests for module re-exports from LangGraph."""

    def test_tool_node_exported(self):
        """ToolNode should be exported from the module."""
        assert ToolNode is not None
        assert hasattr(ToolNode, "tools_by_name")

    def test_tools_condition_exported(self):
        """tools_condition should be exported and callable."""
        assert tools_condition is not None
        assert callable(tools_condition)

    def test_no_tools_message_constant(self):
        """NO_TOOLS_MESSAGE should be a non-empty string."""
        assert isinstance(NO_TOOLS_MESSAGE, str)
        assert len(NO_TOOLS_MESSAGE) > 10


# =============================================================================
# Test: _get_last_ai_message
# =============================================================================


class TestGetLastAIMessage:
    """Tests for _get_last_ai_message helper."""

    def test_returns_last_ai_message(self):
        messages = [
            HumanMessage(content="Hello"),
            AIMessage(content="First AI"),
            HumanMessage(content="Follow up"),
            AIMessage(content="Second AI"),
        ]
        assert _get_last_ai_message(messages).content == "Second AI"

    def test_returns_none_when_no_ai_message(self):
        assert _get_last_ai_message([HumanMessage(content="Hello")]) is None

    def test_returns_none_for_empty_list(self):
        assert _get_last_ai_message([]) is None

    def test_returns_only_ai_message(self):
        messages = [
            HumanMessage(content="Q1"),
            AIMessage(content="A1"),
            HumanMessage(content="Q2"),
        ]
        assert _get_last_ai_message(messages).content == "A1"

    def test_skips_tool_messages(self):
        messages = [
            HumanMessage(content="Q1"),
            AIMessage(content="I'll search", tool_calls=[{"name": "s", "args": {}, "id": "1"}]),
            ToolMessage(content="Results", tool_call_id="1"),
        ]
        assert _get_last_ai_message(messages).content == "I'll search"


# =============================================================================
# Test: _create_no_tools_response
# =============================================================================


class TestCreateNoToolsResponse:
    """Tests for _create_no_tools_response helper."""

    def test_creates_tool_messages(self):
        tool_calls = [
            {"name": "t1", "args": {}, "id": "call_1"},
            {"name": "t2", "args": {}, "id": "call_2"},
        ]
        result = _create_no_tools_response(tool_calls)
        assert len(result["messages"]) == 2
        assert all(isinstance(m, ToolMessage) for m in result["messages"])

    def test_uses_no_tools_message(self):
        result = _create_no_tools_response([{"name": "t", "args": {}, "id": "c1"}])
        assert result["messages"][0].content == NO_TOOLS_MESSAGE

    def test_uses_correct_tool_call_ids(self):
        tool_calls = [
            {"name": "t1", "args": {}, "id": "id_abc"},
            {"name": "t2", "args": {}, "id": "id_xyz"},
        ]
        result = _create_no_tools_response(tool_calls)
        assert result["messages"][0].tool_call_id == "id_abc"
        assert result["messages"][1].tool_call_id == "id_xyz"

    def test_handles_missing_id(self):
        result = _create_no_tools_response([{"name": "t", "args": {}}])
        assert result["messages"][0].tool_call_id == "no-id-0"

    def test_empty_tool_calls(self):
        assert _create_no_tools_response([])["messages"] == []


# =============================================================================
# Test: _inject_session_id
# =============================================================================


class TestInjectSessionId:
    """Tests for _inject_session_id helper function."""

    def test_injects_session_id_for_adr_search(self):
        state = {
            "messages": [
                HumanMessage(content="Search aspirin"),
                AIMessage(
                    content="",
                    tool_calls=[{"name": "adr_search", "args": {"query": "aspirin"}, "id": "c1"}],
                ),
            ],
            "session_id": "session-abc",
        }
        patched = _inject_session_id(state)
        last_ai = _get_last_ai_message(patched["messages"])
        assert last_ai.tool_calls[0]["args"]["session_id"] == "session-abc"
        assert last_ai.tool_calls[0]["args"]["query"] == "aspirin"

    def test_does_not_inject_for_non_adr_tools(self):
        state = {
            "messages": [
                HumanMessage(content="calc"),
                AIMessage(
                    content="",
                    tool_calls=[{"name": "calculator", "args": {"expression": "1+1"}, "id": "c1"}],
                ),
            ],
            "session_id": "session-abc",
        }
        patched = _inject_session_id(state)
        assert "session_id" not in _get_last_ai_message(patched["messages"]).tool_calls[0]["args"]

    def test_injects_only_for_adr_search_in_mixed_calls(self):
        state = {
            "messages": [
                HumanMessage(content="do stuff"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "adr_search", "args": {"query": "q1"}, "id": "c1"},
                        {"name": "calculator", "args": {"expression": "2+2"}, "id": "c2"},
                    ],
                ),
            ],
            "session_id": "session-xyz",
        }
        patched = _inject_session_id(state)
        last_ai = _get_last_ai_message(patched["messages"])
        assert last_ai.tool_calls[0]["args"]["session_id"] == "session-xyz"
        assert "session_id" not in last_ai.tool_calls[1]["args"]

    def test_does_not_mutate_original_state(self):
        ai_msg = AIMessage(
            content="",
            tool_calls=[{"name": "adr_search", "args": {"query": "aspirin"}, "id": "c1"}],
        )
        state = {
            "messages": [HumanMessage(content="Search"), ai_msg],
            "session_id": "session-abc",
        }
        patched = _inject_session_id(state)

        assert "session_id" not in ai_msg.tool_calls[0]["args"]
        assert (
            _get_last_ai_message(patched["messages"]).tool_calls[0]["args"]["session_id"]
            == "session-abc"
        )

    def test_noop_when_no_session_id(self):
        state = {
            "messages": [
                HumanMessage(content="Hi"),
                AIMessage(
                    content="",
                    tool_calls=[{"name": "adr_search", "args": {"query": "q"}, "id": "c1"}],
                ),
            ],
        }
        assert _inject_session_id(state) is state

    def test_noop_when_no_ai_message(self):
        state = {"messages": [HumanMessage(content="Hello")], "session_id": "s"}
        assert _inject_session_id(state) is state

    def test_noop_when_no_tool_calls(self):
        state = {
            "messages": [HumanMessage(content="Hi"), AIMessage(content="Hi!")],
            "session_id": "s",
        }
        assert _inject_session_id(state) is state

    def test_noop_when_empty_messages(self):
        state = {"messages": [], "session_id": "s"}
        assert _inject_session_id(state) is state


# =============================================================================
# Test: _inject_session_id_node and create_inject_session_node
# =============================================================================


class TestInjectSessionIdNode:
    """Tests for the inject_session_id graph node and its factory."""

    def test_factory_returns_the_node_function(self):
        assert create_inject_session_node() is _inject_session_id_node

    @pytest.mark.asyncio
    async def test_node_injects_session_id(self):
        state = {
            "messages": [
                HumanMessage(content="Search"),
                AIMessage(
                    content="",
                    tool_calls=[{"name": "adr_search", "args": {"query": "aspirin"}, "id": "c1"}],
                ),
            ],
            "session_id": "session-123",
        }
        patched = await _inject_session_id_node(state)
        assert (
            _get_last_ai_message(patched["messages"]).tool_calls[0]["args"]["session_id"]
            == "session-123"
        )

    @pytest.mark.asyncio
    async def test_node_noop_when_no_session_id(self):
        state = {
            "messages": [
                HumanMessage(content="Hi"),
                AIMessage(
                    content="",
                    tool_calls=[{"name": "adr_search", "args": {"query": "q"}, "id": "c1"}],
                ),
            ],
        }
        assert await _inject_session_id_node(state) is state


# =============================================================================
# Test: create_tool_node factory
# =============================================================================


class TestCreateToolNode:
    """Tests for create_tool_node factory function."""

    def test_returns_fallback_without_tools(self):
        node = create_tool_node(tools=None)
        assert callable(node)
        assert not isinstance(node, ToolNode)

    def test_returns_fallback_with_empty_tools(self):
        node = create_tool_node(tools=[])
        assert callable(node)
        assert not isinstance(node, ToolNode)

    def test_returns_tool_node_with_tools(self):
        node = create_tool_node(tools=[search_tool])
        assert isinstance(node, ToolNode)

    def test_exposes_tools_by_name(self):
        node = create_tool_node(tools=[search_tool])
        assert "search_tool" in node.tools_by_name

    def test_multiple_tools(self):
        node = create_tool_node(tools=[search_tool, calculator_tool])
        assert isinstance(node, ToolNode)
        assert "search_tool" in node.tools_by_name
        assert "calculator_tool" in node.tools_by_name


# =============================================================================
# Test: Fallback Node (no tools configured)
# =============================================================================


class TestFallbackToolNode:
    """Tests for the fallback tool node when no tools available."""

    @pytest.mark.asyncio
    async def test_no_ai_message_returns_empty(self, state_no_ai_message):
        node = create_tool_node(tools=None)
        assert await node(state_no_ai_message) == {"messages": []}

    @pytest.mark.asyncio
    async def test_no_tool_calls_returns_empty(self, state_ai_no_tool_calls):
        node = create_tool_node(tools=None)
        assert await node(state_ai_no_tool_calls) == {"messages": []}

    @pytest.mark.asyncio
    async def test_returns_no_tools_message(self, state_with_tool_calls):
        node = create_tool_node(tools=None)
        result = await node(state_with_tool_calls)

        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], ToolMessage)
        assert result["messages"][0].content == NO_TOOLS_MESSAGE
        assert result["messages"][0].tool_call_id == "call_123"

    @pytest.mark.asyncio
    async def test_handles_multiple_tool_calls(self, state_with_multiple_tool_calls):
        node = create_tool_node(tools=None)
        result = await node(state_with_multiple_tool_calls)

        assert len(result["messages"]) == 2
        assert all(m.content == NO_TOOLS_MESSAGE for m in result["messages"])
        assert result["messages"][0].tool_call_id == "call_1"
        assert result["messages"][1].tool_call_id == "call_2"


# =============================================================================
# Test: tools_condition
# =============================================================================


class TestToolsCondition:
    """Tests for tools_condition routing function."""

    def test_returns_tools_when_tool_calls_present(self, state_with_tool_calls):
        assert tools_condition(state_with_tool_calls) == "tools"

    def test_returns_end_when_no_tool_calls(self, state_ai_no_tool_calls):
        assert tools_condition(state_ai_no_tool_calls) == "__end__"

    def test_handles_empty_tool_calls_list(self):
        state = {
            "messages": [
                HumanMessage(content="Hello"),
                AIMessage(content="Response", tool_calls=[]),
            ],
        }
        assert tools_condition(state) == "__end__"

    def test_handles_multiple_tool_calls(self, state_with_multiple_tool_calls):
        assert tools_condition(state_with_multiple_tool_calls) == "tools"
