"""Tests for LLM grounding judge."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from src.api.validation.grounding_judge import judge_grounding

_MEDICAL_DISCLAIMER = "This analysis is based on the uploaded documents and is not intended for medical decision-making."


class TestGroundingJudge:
    @pytest.mark.asyncio
    async def test_grounded_returns_content_unchanged(self):
        with patch(
            "src.api.validation.grounding_judge._call_judge", new_callable=AsyncMock
        ) as mock:
            mock.return_value = "GROUNDED"
            verdict, content = await judge_grounding(
                ai_content="The patient has hemangioma.",
                tool_messages=[MagicMock(content="Source: doc.pdf\nContent: hemangioma diagnosis")],
                session_id="sess_1",
            )
        assert verdict == "GROUNDED"
        assert content == "The patient has hemangioma."

    @pytest.mark.asyncio
    async def test_partial_appends_disclaimer(self):
        with patch(
            "src.api.validation.grounding_judge._call_judge", new_callable=AsyncMock
        ) as mock:
            mock.return_value = "PARTIAL — claim about diabetes not found in documents"
            verdict, content = await judge_grounding(
                ai_content="The patient has hemangioma and diabetes.",
                tool_messages=[MagicMock(content="Source: doc.pdf\nContent: hemangioma")],
                session_id="sess_1",
            )
        assert verdict == "PARTIAL"
        assert _MEDICAL_DISCLAIMER in content
        assert "hemangioma and diabetes" in content

    @pytest.mark.asyncio
    async def test_ungrounded_replaces_content(self):
        with patch(
            "src.api.validation.grounding_judge._call_judge", new_callable=AsyncMock
        ) as mock:
            mock.return_value = "UNGROUNDED"
            verdict, content = await judge_grounding(
                ai_content="The patient definitely has cancer.",
                tool_messages=[MagicMock(content="No relevant documents found")],
                session_id="sess_1",
            )
        assert verdict == "UNGROUNDED"
        assert "could not verify" in content.lower()
        assert "cancer" not in content

    @pytest.mark.asyncio
    async def test_timeout_defaults_to_partial(self):
        with patch(
            "src.api.validation.grounding_judge._call_judge", new_callable=AsyncMock
        ) as mock:
            mock.side_effect = TimeoutError()
            verdict, content = await judge_grounding(
                ai_content="Some response.",
                tool_messages=[],
                session_id="sess_1",
            )
        assert verdict == "PARTIAL"
        assert _MEDICAL_DISCLAIMER in content

    @pytest.mark.asyncio
    async def test_no_tool_messages_defaults_to_partial(self):
        with patch(
            "src.api.validation.grounding_judge._call_judge", new_callable=AsyncMock
        ) as mock:
            mock.return_value = "GROUNDED"
            verdict, content = await judge_grounding(
                ai_content="I could not find that information.",
                tool_messages=[],
                session_id="sess_1",
            )
        assert verdict == "PARTIAL"
        assert _MEDICAL_DISCLAIMER in content

    @pytest.mark.asyncio
    async def test_empty_content_skips_judge(self):
        verdict, content = await judge_grounding(
            ai_content="",
            tool_messages=[],
            session_id="sess_1",
        )
        assert verdict == "GROUNDED"
        assert content == ""
