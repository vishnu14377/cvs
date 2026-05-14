"""
LangGraph definition for the ADR AI Agent.

This module defines the agent workflow graph using LangGraph. The graph
connects the generate node, tool node, and router to create a reactive
agent that can process queries and optionally use tools.

Graph Structure:
    START -> generate -> tools_condition -> tools -> generate (loop)
                                        |
                                        -> END (if no tool calls)

Features:
- Compiles a LangGraph StateGraph for the agent workflow
- Supports tool-calling loop with conditional routing via tools_condition
- Configurable with custom tools and system prompts
- Built-in checkpointing support using session_id as thread_id

Usage:
    from src.agents.graph import create_agent_graph, invoke_graph
    from langgraph.checkpoint.memory import MemorySaver
    from langchain_core.messages import HumanMessage

    # Create graph with checkpointing
    checkpointer = MemorySaver()
    graph = create_agent_graph(tools=[...], checkpointer=checkpointer)

    # Invoke with session_id (used as thread_id for state persistence)
    result = await invoke_graph(
        graph,
        message="What is ADR?",
        session_id="user-123"
    )
"""

from __future__ import annotations

import json as _json
import uuid as _uuid
from collections.abc import Sequence
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from src.agents.generate_node import create_generate_node
from src.agents.state import AgentState
from src.agents.tool_node import create_inject_session_node, create_tool_node, tools_condition
from src.core.logger import get_logger

logger = get_logger(__name__)

# =============================================================================
# Node Names
# =============================================================================

NODE_GENERATE = "generate"
NODE_INJECT_SESSION = "inject_session_id"
NODE_TOOLS = "tools"


# =============================================================================
# Grounding Gate
# =============================================================================


def _create_grounding_gate():
    """Observational node that logs when the agent responds without using search tools."""

    def grounding_gate(state: AgentState) -> AgentState:
        messages = state.get("messages", [])
        session_id = state.get("session_id", "unknown")

        if not messages:
            return state

        last_msg = messages[-1]

        if getattr(last_msg, "type", None) != "ai":
            return state
        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            return state

        has_tool_results = any(getattr(m, "type", None) == "tool" for m in messages)
        if not has_tool_results:
            logger.warning(
                "GROUNDING_ALERT: Agent responded without using search tools: session=%s",
                session_id,
            )

        return state

    return grounding_gate


# =============================================================================
# Graph Factory
# =============================================================================


def create_agent_graph(
    tools: Sequence[BaseTool] | None = None,
    system_prompt: str | None = None,
    max_turns: int = 10,
    checkpointer: BaseCheckpointSaver | None = None,
):
    """
    Create the LangGraph agent workflow graph.

    This function builds a StateGraph with the following structure:
    - generate: Processes user queries and generates responses
    - inject_session_id: Injects session_id into tool-call args
    - tools: Executes tool calls via LangGraph's ToolNode
    - Conditional routing based on whether tool calls are present

    Graph flow:
        START → generate → tools_condition → inject_session_id → tools → generate
                                           |
                                           → END (if no tool calls)

    Args:
        tools: Optional sequence of tools to make available to the agent.
        system_prompt: Optional custom system prompt for the agent.
        max_turns: Maximum conversation turns to include in context.
        checkpointer: Optional checkpointer for state persistence.
            If provided, enables conversation history persistence across invocations.
            Use session_id as thread_id in config to isolate conversations.

    Returns:
        Compiled LangGraph that can be invoked with agent state.

    Example:
        >>> from langgraph.checkpoint.memory import MemorySaver
        >>> graph = create_agent_graph(tools=[retriever_tool], checkpointer=MemorySaver())
        >>> result = await graph.ainvoke(
        ...     {"messages": [HumanMessage(content="What is ADR?")]},
        ...     config={"configurable": {"thread_id": "session-123"}}
        ... )
    """
    logger.info(
        "Creating agent graph: tools=%d, max_turns=%d, checkpointer=%s",
        len(tools) if tools else 0,
        max_turns,
        type(checkpointer).__name__ if checkpointer else "None",
    )

    # Create the state graph
    workflow = StateGraph(AgentState)

    # Create node functions with captured configuration
    generate_fn = create_generate_node(
        tools=tools,
        system_prompt=system_prompt,
        max_turns=max_turns,
    )
    inject_session_fn = create_inject_session_node()
    tool_fn = create_tool_node(tools=tools)
    grounding_gate_fn = _create_grounding_gate()

    # Add nodes to the graph
    workflow.add_node(NODE_GENERATE, generate_fn)  # type: ignore[call-overload]
    workflow.add_node("grounding_gate", grounding_gate_fn)
    workflow.add_node(NODE_INJECT_SESSION, inject_session_fn)
    workflow.add_node(NODE_TOOLS, tool_fn)

    # Set the entry point
    workflow.set_entry_point(NODE_GENERATE)

    # Generate → grounding gate (observational check)
    workflow.add_edge(NODE_GENERATE, "grounding_gate")

    # Grounding gate → tools_condition (route to tools or end)
    workflow.add_conditional_edges(
        "grounding_gate",
        tools_condition,
        {
            "tools": NODE_INJECT_SESSION,
            "__end__": END,
        },
    )

    # inject_session_id always flows into the actual ToolNode
    workflow.add_edge(NODE_INJECT_SESSION, NODE_TOOLS)

    # After tool execution, go back to generate for the final response
    workflow.add_edge(NODE_TOOLS, NODE_GENERATE)

    # Compile with optional checkpointer
    compiled_graph = workflow.compile(checkpointer=checkpointer)

    logger.info("Agent graph compiled successfully (checkpointing=%s)", checkpointer is not None)
    return compiled_graph


def create_simple_graph(checkpointer: BaseCheckpointSaver | None = None):
    """
    Create a simple agent graph with no tools.

    This is a convenience function for basic usage without tools.
    The agent will simply respond to queries using the LLM.

    Args:
        checkpointer: Optional checkpointer for state persistence.

    Returns:
        Compiled LangGraph for simple query-response interactions.
    """
    return create_agent_graph(tools=None, checkpointer=checkpointer)


def create_checkpointer() -> MemorySaver:
    """
    Create an in-memory checkpointer for the agent.

    This creates a MemorySaver instance that can be used to persist
    conversation state across invocations. For production use with
    multiple instances, consider using a persistent checkpointer
    like PostgresSaver or RedisSaver.

    Returns:
        A MemorySaver instance for checkpointing.

    Example:
        >>> checkpointer = create_checkpointer()
        >>> graph = create_agent_graph(checkpointer=checkpointer)
    """
    return MemorySaver()


# =============================================================================
# Graph Invocation Helpers
# =============================================================================


def get_session_config(session_id: str) -> dict[str, Any]:
    """
    Create a config dict with session_id as thread_id for checkpointing.

    Args:
        session_id: Unique session identifier to use as thread_id.

    Returns:
        Config dict for graph invocation with thread_id set.

    Example:
        >>> config = get_session_config("user-123")
        >>> result = await graph.ainvoke(state, config=config)
    """
    return {"configurable": {"thread_id": session_id}}


async def invoke_graph(
    graph,
    message: str | BaseMessage,
    session_id: str,
) -> dict[str, Any]:
    """
    Invoke the agent graph with a message and session_id.

    This is a convenience function that:
    - Creates the input state with the message
    - Sets up the config with session_id as thread_id
    - Invokes the graph asynchronously

    Args:
        graph: Compiled LangGraph to invoke.
        message: User message (string or BaseMessage).
        session_id: Unique session identifier (used as thread_id for checkpointing).

    Returns:
        The graph result containing updated messages.

    Example:
        >>> graph = create_agent_graph(checkpointer=MemorySaver())
        >>> result = await invoke_graph(graph, "What is ADR?", "user-123")
        >>> print(result["messages"][-1].content)
    """
    # Convert string to HumanMessage if needed
    if isinstance(message, str):
        message = HumanMessage(content=message)

    # Create input state
    input_state: AgentState = {
        "messages": [message],
        "session_id": session_id,
    }

    # Create config with session_id as thread_id
    config = get_session_config(session_id)

    logger.debug("Invoking graph: session_id=%s", session_id)

    # Invoke the graph
    result = await graph.ainvoke(input_state, config=config)

    return result


async def stream_graph(
    graph,
    message: str | BaseMessage,
    session_id: str,
):
    """
    Stream the agent graph execution as SSE events.

    Yields SSE-formatted strings for each event:
    - event: token  (partial AI content)
    - event: tool_call  (tool invocation started)
    - event: tool_result  (tool invocation completed)
    - event: done  (final response with full content)
    - event: error  (if something fails)
    """
    if isinstance(message, str):
        message = HumanMessage(content=message)

    input_state: AgentState = {
        "messages": [message],
        "session_id": session_id,
    }
    config = get_session_config(session_id)

    logger.debug("Streaming graph: session_id=%s", session_id)

    final_content = ""
    tool_messages = []
    try:
        async for chunk in graph.astream(input_state, config=config):
            for _node_name, node_output in chunk.items():
                messages = node_output.get("messages", [])
                for msg in messages:
                    if getattr(msg, "type", None) == "ai":
                        if hasattr(msg, "tool_calls") and msg.tool_calls:
                            for tc in msg.tool_calls:
                                tool_name = tc.get("name", "unknown")
                                yield f"event: tool_call\ndata: {_json.dumps({'tool': tool_name, 'status': 'started'})}\n\n"
                        elif msg.content:
                            raw = msg.content
                            if isinstance(raw, list):
                                text = " ".join(
                                    part.get("text", "") if isinstance(part, dict) else str(part)
                                    for part in raw
                                ).strip()
                            else:
                                text = str(raw)
                            final_content = text
                            yield f"event: token\ndata: {_json.dumps({'content': text})}\n\n"
                    elif getattr(msg, "type", None) == "tool":
                        tool_name = getattr(msg, "name", "unknown")
                        tool_messages.append(msg)
                        yield f"event: tool_result\ndata: {_json.dumps({'tool': tool_name, 'status': 'completed'})}\n\n"
    except Exception as e:
        logger.error("Stream error: %s", str(e), exc_info=True)
        yield f"event: error\ndata: {_json.dumps({'detail': 'Unable to process your question. Please try again.'})}\n\n"
        return

    # Run grounding judge before final render so ungrounded claims don't reach the client.
    # Tokens have already streamed, so the client must overwrite its display with the `done`
    # payload's content when a grounding verdict is attached.
    from src.api.rendering.html_renderer import render_to_base64, render_to_safe_html
    from src.api.validation.grounding_judge import judge_grounding

    verdict, final_content = await judge_grounding(final_content, tool_messages, session_id)

    message_id = f"msg_{_uuid.uuid4().hex[:12]}"
    done_payload = {
        "message_id": message_id,
        "content": final_content,
        "content_html": render_to_safe_html(final_content),
        "content_base64": render_to_base64(final_content),
        "grounding": verdict,
    }
    yield f"event: done\ndata: {_json.dumps(done_payload)}\n\n"


def invoke_graph_sync(
    graph,
    message: str | BaseMessage,
    session_id: str,
) -> dict[str, Any]:
    """
    Synchronous version of invoke_graph.

    Args:
        graph: Compiled LangGraph to invoke.
        message: User message (string or BaseMessage).
        session_id: Unique session identifier (used as thread_id for checkpointing).

    Returns:
        The graph result containing updated messages.
    """
    # Convert string to HumanMessage if needed
    if isinstance(message, str):
        message = HumanMessage(content=message)

    # Create input state
    input_state: AgentState = {
        "messages": [message],
        "session_id": session_id,
    }

    # Create config with session_id as thread_id
    config = get_session_config(session_id)

    logger.debug("Invoking graph (sync): session_id=%s", session_id)

    # Invoke the graph
    result = graph.invoke(input_state, config=config)

    return result


def get_session_history(graph, session_id: str) -> list:
    """
    Get the conversation history for a session.

    Retrieves messages from the checkpointer using session_id as thread_id.

    Args:
        graph: Compiled LangGraph with checkpointer.
        session_id: Unique session identifier.

    Returns:
        List of messages in the session, or empty list if not found.

    Example:
        >>> history = get_session_history(graph, "user-123")
        >>> for msg in history:
        ...     print(f"{type(msg).__name__}: {msg.content}")
    """
    try:
        config = get_session_config(session_id)
        state = graph.get_state(config)

        if state and state.values:
            return state.values.get("messages", [])
        return []
    except Exception as e:
        logger.warning("Error getting session history: %s", str(e))
        return []


def clear_session_history(graph, session_id: str) -> bool:
    """
    Delete all checkpointed state for a session.

    Uses the checkpointer's ``delete_thread`` method to remove every
    checkpoint and pending write associated with the given
    ``session_id`` (which is stored as the ``thread_id``).

    Args:
        graph: Compiled LangGraph whose checkpointer holds the state.
        session_id: The session / thread to purge.

    Returns:
        ``True`` if the history was successfully deleted (or there was
        nothing to delete).  ``False`` if the graph has no checkpointer.

    Raises:
        Exception: Propagates any error from the checkpointer's
            ``delete_thread`` call.

    Example:
        >>> cleared = clear_session_history(graph, "adr-20260414-abc")
        >>> print(cleared)  # True
    """
    checkpointer = getattr(graph, "checkpointer", None)
    if checkpointer is None:
        logger.debug("Graph has no checkpointer — nothing to clear for session '%s'", session_id)
        return False

    checkpointer.delete_thread(session_id)
    logger.info("Cleared checkpointer history for session '%s'", session_id)
    return True


if __name__ == "__main__":
    import asyncio

    # Example usage
    async def main():
        graph = create_simple_graph(checkpointer=create_checkpointer())
        result = await invoke_graph(graph, "What is ADR?", "test-session-123")
        print(result["messages"][-1].content)

    # Example usage for multi session with different messages and session IDs
    # NOTE: Messages for the same session must be sent SEQUENTIALLY (not in parallel)
    # to properly accumulate conversation history in the checkpointer
    async def multi_session_example():
        graph = create_simple_graph(checkpointer=create_checkpointer())

        # Define messages per session
        session_messages = {
            "user-123": [
                "What is ADR?",
                "What are the challenges of ADR?",
                "What are the best practices for ADR?",
                "What are the common use cases for ADR?",
            ],
            "user-456": [
                "Explain the benefits of ADR.",
                "What is the future of ADR?",
                "How does ADR compare to traditional architecture?",
            ],
            "user-789": [
                "How to implement ADR?",
                "How to evaluate ADR?",
                "What are the key components of ADR?",
                "How to get started with ADR?",
            ],
        }

        # Process each session's messages SEQUENTIALLY to preserve history
        for session_id, messages in session_messages.items():
            print(f"\n--- Processing session: {session_id} ---")
            for message in messages:
                result = await invoke_graph(graph, message, session_id)
                print(f"  Response: {result['messages'][-1].content[:80]}...")

        # Print session histories
        print("\n" + "=" * 60)
        print("SESSION HISTORIES")
        print("=" * 60)

        for session_id in ["user-123", "user-456", "user-789"]:
            history = get_session_history(graph, session_id)
            print(f"\n=== Session: {session_id} ({len(history)} messages) ===")
            if not history:
                print("  (No messages)")
            else:
                for i, msg in enumerate(history):
                    role = msg.type.upper()
                    content = msg.content if msg.content else "(no content)"
                    # Truncate long content for display
                    if len(content) > 100:
                        content = content[:100] + "..."
                    print(f"  {i + 1}. [{role}]: {content}")

    asyncio.run(multi_session_example())
