"""
Agent modules for the CareConnect ADR AI Agent.

This package contains the agent workflow components including:
- State definitions for the agent workflow
- Generate node for LLM-based response generation with tool calling
- Tool node for executing tool calls (using LangGraph's prebuilt ToolNode)
- Graph definition using LangGraph with checkpointing support

Usage:
    from src.agents import create_agent_graph, AgentState
    from langgraph.checkpoint.memory import MemorySaver
    from langchain_core.messages import HumanMessage

    # Create graph with checkpointing
    checkpointer = MemorySaver()
    graph = create_agent_graph(tools=[...], checkpointer=checkpointer)

    # Invoke the graph
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="Hello")], "session_id": "123"},
        config={"configurable": {"thread_id": "123"}}
    )
"""

from src.agents.generate_node import (
    DEFAULT_MAX_TURNS,
    DEFAULT_SYSTEM_PROMPT,
    create_generate_node,
    generate_node,
    generate_node_sync,
)
from src.agents.graph import (
    NODE_GENERATE,
    NODE_INJECT_SESSION,
    NODE_TOOLS,
    create_agent_graph,
    create_checkpointer,
    create_simple_graph,
    get_session_config,
    get_session_history,
    invoke_graph,
    invoke_graph_sync,
)
from src.agents.state import AgentState, add_messages
from src.agents.tool_node import (
    NO_TOOLS_MESSAGE,
    ToolNode,
    create_inject_session_node,
    create_tool_node,
    tools_condition,
)

__all__ = [
    # State
    "AgentState",
    "add_messages",
    # Generate node
    "generate_node",
    "generate_node_sync",
    "create_generate_node",
    "DEFAULT_MAX_TURNS",
    "DEFAULT_SYSTEM_PROMPT",
    # Tool node & session-ID injection
    "ToolNode",
    "tools_condition",
    "create_tool_node",
    "create_inject_session_node",
    "NO_TOOLS_MESSAGE",
    # Graph
    "create_agent_graph",
    "create_simple_graph",
    "create_checkpointer",
    "get_session_config",
    "invoke_graph",
    "invoke_graph_sync",
    "get_session_history",
    "NODE_GENERATE",
    "NODE_INJECT_SESSION",
    "NODE_TOOLS",
]
