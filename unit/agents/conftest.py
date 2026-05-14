"""
Pytest configuration and fixtures for agent unit tests.

These tests require langchain_postgres. Skip this test package when the
dependency is not installed instead of globally injecting mock modules into
sys.modules during pytest collection.
"""

from unittest.mock import MagicMock

import pytest

pytest.importorskip("langchain_postgres")
from unittest.mock import AsyncMock

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import BaseTool


@pytest.fixture
def basic_state():
    """Create a basic agent state for testing."""
    return {
        "messages": [HumanMessage(content="Hello")],
        "session_id": "test-session-123",
    }


@pytest.fixture
def state_no_ai_message():
    """State with only human message."""
    return {
        "messages": [HumanMessage(content="Hello")],
        "session_id": "test-session",
    }


@pytest.fixture
def state_ai_no_tool_calls():
    """State with AI message but no tool calls."""
    return {
        "messages": [
            HumanMessage(content="Hello"),
            AIMessage(content="Hi there!"),
        ],
        "session_id": "test-session",
    }


@pytest.fixture
def state_with_tool_calls():
    """State with AI message containing tool calls."""
    return {
        "messages": [
            HumanMessage(content="Search for aspirin"),
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "search", "args": {"query": "aspirin"}, "id": "call_123"},
                ],
            ),
        ],
        "session_id": "test-session",
    }


@pytest.fixture
def state_with_multiple_tool_calls():
    """State with multiple tool calls."""
    return {
        "messages": [
            HumanMessage(content="Search"),
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "search", "args": {"query": "a"}, "id": "call_1"},
                    {"name": "search", "args": {"query": "b"}, "id": "call_2"},
                ],
            ),
        ],
        "session_id": "test-session",
    }


@pytest.fixture
def mock_tool():
    """Create a mock tool."""
    tool = MagicMock(spec=BaseTool)
    tool.name = "search"
    tool.description = "Search for documents"
    tool.invoke.return_value = "Search result"
    tool.ainvoke = AsyncMock(return_value="Search result")
    return tool


@pytest.fixture
def mock_llm():
    """Create a mock LLM client."""
    mock = MagicMock()
    mock.ainvoke = AsyncMock(return_value=AIMessage(content="Mock response"))
    mock.invoke = MagicMock(return_value=AIMessage(content="Mock response"))
    mock.bind_tools = MagicMock(return_value=mock)
    return mock
