"""
Generate node for the ADR AI Agent using LangChain ReAct pattern.

This module implements the generate node that processes user queries using an LLM
with tool-calling capabilities via LangChain's ReAct pattern. The node uses the
LangChainClient to invoke the LLM with bound tools.

Features:
- Uses LangChain's bind_tools for tool-calling LLM
- Processes conversation history with configurable message filtering
- Handles system prompts and context management
- Returns AIMessage with tool_calls or content

Usage:
    from src.agents.generate_node import create_generate_node

    # Create generate node with tools
    generate = create_generate_node(tools=[retriever_tool])

    # Use in LangGraph
    workflow.add_node("generate", generate)
"""

from __future__ import annotations

from collections.abc import Callable, Coroutine, Sequence
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables.config import RunnableConfig
from langchain_core.tools import BaseTool
from src.agents.state import AgentState
from src.core.langchain_client import LangChainClient
from src.core.logger import get_logger

logger = get_logger(__name__)

# =============================================================================
# Configuration Constants
# =============================================================================

DEFAULT_MAX_TURNS = 10

DEFAULT_SYSTEM_PROMPT = """You are a document analysis assistant for the CareConnect ADR (Adverse Drug Reaction) system.
Your role is to help healthcare professionals understand adverse drug reaction reports by searching uploaded documents.

RULES — follow these strictly:
1. ALWAYS use the adr_search or policy_search tools before answering questions about documents, patients, medications, diagnoses, or clinical information.
2. ONLY answer based on information returned by the search tools. Do not use your training knowledge for medical or clinical facts.
3. If the search returns no relevant results, respond: "I could not find information about that in the uploaded documents."
4. ALWAYS cite the document source for every piece of information you provide.
5. If asked for medical advice, diagnosis, or treatment recommendations, respond: "I cannot provide medical advice. I can only help you find information in the uploaded documents. Please consult a healthcare professional."
6. For questions outside the scope of the uploaded documents, respond: "That information does not appear to be in the uploaded documents."
7. Never speculate, infer, or extrapolate beyond what the documents explicitly state."""


# =============================================================================
# Helper Functions
# =============================================================================


def _filter_messages_by_turns(
    messages: Sequence[BaseMessage],
    max_turns: int = DEFAULT_MAX_TURNS,
) -> list[BaseMessage]:
    """
    Filter conversation history to keep only the most recent turns.

    A "turn" is a human message followed by assistant/tool messages.

    Args:
        messages: Full conversation history.
        max_turns: Maximum number of turns to keep.

    Returns:
        Filtered list of messages with at most max_turns recent turns.
    """
    if not messages:
        return []

    system_messages = [m for m in messages if isinstance(m, SystemMessage)]
    non_system_messages = [m for m in messages if not isinstance(m, SystemMessage)]

    if not non_system_messages:
        return list(system_messages)

    # Group messages into turns (each turn starts with HumanMessage)
    turns: list[list[BaseMessage]] = []
    current_turn: list[BaseMessage] = []

    for msg in non_system_messages:
        if isinstance(msg, HumanMessage):
            if current_turn:
                turns.append(current_turn)
            current_turn = [msg]
        else:
            current_turn.append(msg)

    if current_turn:
        turns.append(current_turn)

    # Keep only the most recent turns
    recent_turns = turns[-max_turns:] if len(turns) > max_turns else turns

    # Flatten turns back to message list
    filtered_messages = []
    for turn in recent_turns:
        filtered_messages.extend(turn)

    return list(system_messages) + filtered_messages


def _prepare_messages(
    messages: Sequence[BaseMessage],
    system_prompt: str,
    max_turns: int,
) -> list[BaseMessage]:
    """
    Prepare messages for LLM invocation.

    Args:
        messages: Raw conversation messages.
        system_prompt: System prompt to prepend.
        max_turns: Maximum turns to include.

    Returns:
        Prepared message list with system prompt.
    """
    filtered = _filter_messages_by_turns(messages, max_turns)

    # Ensure system prompt is first
    if not any(isinstance(m, SystemMessage) for m in filtered):
        filtered.insert(0, SystemMessage(content=system_prompt))

    return filtered


# =============================================================================
# Generate Node
# =============================================================================


async def generate_node(
    state: AgentState,
    config: RunnableConfig | None = None,
    *,
    tools: Sequence[BaseTool] | None = None,
    system_prompt: str | None = None,
    max_turns: int = DEFAULT_MAX_TURNS,
) -> dict[str, list[AIMessage]]:
    """
    Generate node using LangChain ReAct pattern.

    Invokes the LLM with bound tools. The LLM will either:
    - Return an AIMessage with tool_calls (to be processed by ToolNode)
    - Return an AIMessage with content (final response)

    Args:
        state: Current agent state with messages.
        config: Optional LangGraph config.
        tools: Tools to bind to the LLM.
        system_prompt: Custom system prompt.
        max_turns: Max conversation turns in context.

    Returns:
        State update with new AIMessage.
    """
    messages = state.get("messages", [])
    session_id = state.get("session_id", "unknown")

    logger.debug(
        "Generate node: session_id=%s, messages=%d, tools=%d",
        session_id,
        len(messages),
        len(tools) if tools else 0,
    )

    # Prepare messages with system prompt and filtering
    prepared_messages = _prepare_messages(
        messages,
        system_prompt or DEFAULT_SYSTEM_PROMPT,
        max_turns,
    )

    # Get LLM from LangChainClient
    client = LangChainClient()
    llm: Any = client.client

    # Bind tools if provided (ReAct pattern)
    if tools:
        llm = llm.bind_tools(tools)

    # Invoke LLM
    try:
        response: AIMessage = await llm.ainvoke(prepared_messages)

        has_tool_calls = bool(getattr(response, "tool_calls", None))
        logger.debug(
            "LLM response: session_id=%s, has_tool_calls=%s",
            session_id,
            has_tool_calls,
        )

    except Exception as e:
        logger.error("LLM error: session_id=%s, error=%s", session_id, e)
        raise

    return {"messages": [response]}


def generate_node_sync(
    state: AgentState,
    config: RunnableConfig | None = None,
    *,
    tools: Sequence[BaseTool] | None = None,
    system_prompt: str | None = None,
    max_turns: int = DEFAULT_MAX_TURNS,
) -> dict[str, list[AIMessage]]:
    """
    Synchronous version of generate_node.

    Args:
        state: Current agent state with messages.
        config: Optional LangGraph config.
        tools: Tools to bind to the LLM.
        system_prompt: Custom system prompt.
        max_turns: Max conversation turns in context.

    Returns:
        State update with new AIMessage.
    """
    messages = state.get("messages", [])
    session_id = state.get("session_id", "unknown")

    prepared_messages = _prepare_messages(
        messages,
        system_prompt or DEFAULT_SYSTEM_PROMPT,
        max_turns,
    )

    client = LangChainClient()
    llm: Any = client.client

    if tools:
        llm = llm.bind_tools(tools)

    try:
        response: AIMessage = llm.invoke(prepared_messages)
    except Exception as e:
        logger.error("LLM error: session_id=%s, error=%s", session_id, e)
        raise

    return {"messages": [response]}


# =============================================================================
# Factory Function
# =============================================================================


def create_generate_node(
    tools: Sequence[BaseTool] | None = None,
    system_prompt: str | None = None,
    max_turns: int = DEFAULT_MAX_TURNS,
) -> Callable[[AgentState, RunnableConfig | None], Coroutine[Any, Any, dict[str, list[AIMessage]]]]:
    """
    Factory to create a generate node for LangGraph.

    Creates a closure that captures tools and configuration for use
    as a LangGraph node.

    Args:
        tools: Tools to bind to the LLM.
        system_prompt: Custom system prompt.
        max_turns: Max conversation turns in context.

    Returns:
        Async function for use as LangGraph node.

    Example:
        >>> generate = create_generate_node(tools=[retriever_tool])
        >>> workflow.add_node("generate", generate)
    """

    async def _node(
        state: AgentState,
        config: RunnableConfig | None = None,
    ) -> dict[str, list[AIMessage]]:
        return await generate_node(
            state,
            config,
            tools=tools,
            system_prompt=system_prompt,
            max_turns=max_turns,
        )

    return _node
