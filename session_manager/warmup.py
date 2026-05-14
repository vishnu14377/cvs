"""Session warmup.

After session creation, fires a lightweight query through the agent graph
to pre-warm LLM connections and vector DB indexes. Runs as a FastAPI
BackgroundTask so it never blocks the session-creation response.

Uses a dedicated thread_id ({session_id}-warmup) so warmup messages
never appear in the real conversation history.
"""

from __future__ import annotations

import time

from src.agents.graph import invoke_graph
from src.core.logger import get_logger
from src.session_manager.core.agent_factory import get_agent

logger = get_logger(__name__)

WARMUP_QUERY = "Summarize the key clinical findings in this document."


async def warmup_session(session_id: str) -> None:
    start = time.monotonic()
    try:
        graph = get_agent()
        warmup_thread_id = f"{session_id}-warmup"
        await invoke_graph(graph, WARMUP_QUERY, warmup_thread_id)
        elapsed_ms = int((time.monotonic() - start) * 1000)
        logger.info(
            "Session warmup completed: session_id=%s, elapsed_ms=%d",
            session_id,
            elapsed_ms,
        )
    except Exception as exc:
        elapsed_ms = int((time.monotonic() - start) * 1000)
        logger.warning(
            "Session warmup failed (non-fatal): session_id=%s, elapsed_ms=%d, error=%s",
            session_id,
            elapsed_ms,
            exc,
        )
