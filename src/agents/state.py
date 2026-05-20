"""
Agent state definitions for the CareConnect ADR AI Agent.

This module defines the TypedDict-based state models used in the LangGraph
agent workflow. The state is passed between nodes and maintains the
conversation history, tool results, and other relevant data.

Usage:
    from src.agents.state import AgentState

    state: AgentState = {
        "messages": [HumanMessage(content="Hello")],
        "session_id": "user-123",
    }
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict):
    """
    State dictionary for the ADR AI Agent workflow.

    This TypedDict defines the shape of the state that flows through
    the LangGraph agent. Each field represents a piece of data that
    nodes can read from or write to.

    Attributes:
        messages: Conversation history with reducer for appending.
            Uses the add_messages reducer to accumulate messages.
        session_id: Unique identifier for the user session.
            Used for retrieving session-specific documents from the vector store.
    """

    messages: Annotated[Sequence[BaseMessage], add_messages]
    session_id: str
