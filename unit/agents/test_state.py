"""
Unit tests for the AgentState module.

Tests cover:
- AgentState TypedDict structure
- add_messages reducer function

Run with: pytest tests/unit/agents/test_state.py -v
"""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from src.agents.state import AgentState, add_messages


class TestAddMessages:
    """Tests for the add_messages reducer function."""

    def test_empty_lists(self):
        """Should return empty list when both inputs are empty."""
        result = add_messages([], [])
        assert result == []

    def test_left_empty(self):
        """Should return right messages when left is empty."""
        right = [HumanMessage(content="Hello")]
        result = add_messages([], right)
        assert len(result) == 1
        assert result[0].content == "Hello"

    def test_right_empty(self):
        """Should return left messages when right is empty."""
        left = [HumanMessage(content="Hello")]
        result = add_messages(left, [])
        assert len(result) == 1
        assert result[0].content == "Hello"

    def test_combines_lists(self):
        """Should combine left and right messages in order."""
        left = [HumanMessage(content="First")]
        right = [AIMessage(content="Second")]
        result = add_messages(left, right)

        assert len(result) == 2
        assert result[0].content == "First"
        assert result[1].content == "Second"

    def test_preserves_order(self):
        """Should preserve message order when combining."""
        left = [HumanMessage(content="1"), AIMessage(content="2")]
        right = [HumanMessage(content="3"), AIMessage(content="4")]
        result = add_messages(left, right)

        assert len(result) == 4
        assert [m.content for m in result] == ["1", "2", "3", "4"]

    def test_handles_all_message_types(self):
        """Should handle all LangChain message types."""
        left = [SystemMessage(content="System")]
        right = [
            HumanMessage(content="Human"),
            AIMessage(content="AI"),
            ToolMessage(content="Tool", tool_call_id="123"),
        ]
        result = add_messages(left, right)

        assert len(result) == 4
        assert isinstance(result[0], SystemMessage)
        assert isinstance(result[1], HumanMessage)
        assert isinstance(result[2], AIMessage)
        assert isinstance(result[3], ToolMessage)


class TestAgentState:
    """Tests for the AgentState TypedDict."""

    def test_creation_with_messages(self):
        """Should create a valid AgentState with messages."""
        state: AgentState = {
            "messages": [HumanMessage(content="Hello")],
            "session_id": "test-session-123",
        }

        assert len(state["messages"]) == 1
        assert state["session_id"] == "test-session-123"

    def test_empty_messages(self):
        """Should allow empty messages list."""
        state: AgentState = {
            "messages": [],
            "session_id": "test-session",
        }

        assert state["messages"] == []
        assert state["session_id"] == "test-session"

    def test_multiple_messages(self):
        """Should store multiple messages of different types."""
        state: AgentState = {
            "messages": [
                HumanMessage(content="Hello"),
                AIMessage(content="Hi there!"),
                HumanMessage(content="How are you?"),
                AIMessage(content="I'm fine!"),
            ],
            "session_id": "multi-msg-session",
        }

        assert len(state["messages"]) == 4
        assert isinstance(state["messages"][0], HumanMessage)
        assert isinstance(state["messages"][1], AIMessage)

    def test_message_content_access(self):
        """Should allow accessing message content."""
        state: AgentState = {
            "messages": [
                HumanMessage(content="Question"),
                AIMessage(content="Answer"),
            ],
            "session_id": "access-test",
        }

        assert state["messages"][0].content == "Question"
        assert state["messages"][1].content == "Answer"

    def test_session_id_formats(self):
        """Should accept various session_id formats."""
        # UUID-style
        state1: AgentState = {"messages": [], "session_id": "550e8400-e29b-41d4-a716-446655440000"}
        assert state1["session_id"] == "550e8400-e29b-41d4-a716-446655440000"

        # Simple string
        state2: AgentState = {"messages": [], "session_id": "user-123"}
        assert state2["session_id"] == "user-123"
