"""LLM grounding judge — verifies AI responses are supported by retrieved documents.

Called after the agent produces a response. Makes a separate LLM call to evaluate
whether every factual claim is grounded in the tool-retrieved documents.
"""

from __future__ import annotations

import asyncio

from src.core.logger import get_logger

logger = get_logger(__name__)

_JUDGE_PROMPT = """You are a medical document grounding verifier. Determine whether an AI response is supported by the retrieved source documents.

RETRIEVED DOCUMENTS:
{tool_results}

AI RESPONSE:
{ai_response}

Evaluate whether every factual claim in the AI response is directly supported by the retrieved documents.

Respond with exactly one of:
- GROUNDED — every claim is supported by the documents
- PARTIAL — some claims are supported but others are not
- UNGROUNDED — the response contains claims not found in the documents"""

_MEDICAL_DISCLAIMER = (
    "\n\n---\n*This analysis is based on the uploaded documents and is not intended "
    "for medical decision-making. Please consult a healthcare professional for clinical decisions.*"
)

_UNGROUNDED_RESPONSE = (
    "I could not verify this information from the uploaded documents. "
    "Please rephrase your question, and I will search the documents again."
)


async def _call_judge(ai_content: str, tool_text: str) -> str:
    """Call the LLM judge. Returns raw verdict string."""
    from langchain_core.messages import HumanMessage
    from src.core.config import llm_config
    from src.core.langchain_client import LangChainClient

    if llm_config.VERTEX_AI_MODE == "stub":
        return "GROUNDED"

    prompt = _JUDGE_PROMPT.format(tool_results=tool_text[:3000], ai_response=ai_content[:2000])
    client = LangChainClient()
    llm = client.client
    result = await asyncio.to_thread(lambda: llm.invoke([HumanMessage(content=prompt)]))
    content = result.content
    return (str(content) if content else "").strip()


async def judge_grounding(
    ai_content: str,
    tool_messages: list,
    session_id: str,
    timeout: float = 10.0,
) -> tuple[str, str]:
    """Judge whether the AI response is grounded in retrieved documents.

    Args:
        ai_content: The AI's response text.
        tool_messages: List of tool message objects from the conversation.
        session_id: Session ID for logging.
        timeout: Max seconds to wait for judge LLM call.

    Returns:
        Tuple of (verdict, modified_content) where verdict is
        GROUNDED, PARTIAL, or UNGROUNDED, and modified_content
        may include a disclaimer or replacement text.
    """
    if not ai_content or not ai_content.strip():
        return ("GROUNDED", ai_content)

    # If no tool messages at all, we can't verify grounding
    tool_text = "\n".join(str(getattr(m, "content", "")) for m in tool_messages)
    if not tool_messages or not tool_text.strip():
        logger.warning("No tool messages for grounding check: session=%s", session_id)
        return ("PARTIAL", ai_content + _MEDICAL_DISCLAIMER)

    try:
        raw_verdict = await asyncio.wait_for(_call_judge(ai_content, tool_text), timeout=timeout)
    except (TimeoutError, asyncio.TimeoutError):
        logger.warning("Grounding judge timed out: session=%s", session_id)
        return ("PARTIAL", ai_content + _MEDICAL_DISCLAIMER)
    except Exception as e:
        logger.error("Grounding judge error: session=%s, error=%s", session_id, e)
        return ("PARTIAL", ai_content + _MEDICAL_DISCLAIMER)

    verdict = raw_verdict.upper()
    if "UNGROUNDED" in verdict:
        logger.warning("UNGROUNDED response detected: session=%s", session_id)
        return ("UNGROUNDED", _UNGROUNDED_RESPONSE)
    elif "PARTIAL" in verdict:
        logger.info("PARTIAL grounding: session=%s", session_id)
        return ("PARTIAL", ai_content + _MEDICAL_DISCLAIMER)
    else:
        logger.debug("GROUNDED response: session=%s", session_id)
        return ("GROUNDED", ai_content)
