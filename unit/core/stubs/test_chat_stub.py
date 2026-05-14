"""Tests for StubChatModel."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from src.core.stubs.chat_stub import StubChatModel, StubStructuredRunnable


class TestStubChatModelInvoke:
    def test_invoke_returns_aimessage(self) -> None:
        model = StubChatModel()
        result = model.invoke([HumanMessage(content="hello")])
        assert isinstance(result, AIMessage)
        assert "[stub]" in result.content
        assert "hello" in result.content

    def test_invoke_empty_messages_returns_placeholder(self) -> None:
        model = StubChatModel()
        result = model.invoke([])
        assert isinstance(result, AIMessage)
        assert "[stub]" in result.content

    def test_invoke_with_string_prompt(self) -> None:
        model = StubChatModel()
        result = model.invoke("what is the adr count?")
        assert isinstance(result, AIMessage)
        assert "what is the adr count?" in result.content

    def test_invoke_ignores_system_messages_for_echo(self) -> None:
        model = StubChatModel()
        result = model.invoke(
            [
                SystemMessage(content="be concise"),
                HumanMessage(content="user input"),
            ]
        )
        assert "user input" in result.content

    def test_no_tool_calls(self) -> None:
        """Stub must NOT emit tool calls — otherwise agent graph loops forever."""
        model = StubChatModel()
        result = model.invoke([HumanMessage(content="search adrs")])
        assert result.tool_calls == []


class TestStubChatModelAinvoke:
    @pytest.mark.asyncio
    async def test_ainvoke_returns_aimessage(self) -> None:
        model = StubChatModel()
        result = await model.ainvoke([HumanMessage(content="hello async")])
        assert isinstance(result, AIMessage)
        assert "hello async" in result.content


class TestStubChatModelBindTools:
    def test_bind_tools_returns_self(self) -> None:
        model = StubChatModel()
        bound = model.bind_tools([])
        assert bound is model

    def test_bind_tools_then_invoke_still_works(self) -> None:
        model = StubChatModel()
        bound = model.bind_tools([])
        result = bound.invoke([HumanMessage(content="q")])
        assert isinstance(result, AIMessage)

    def test_bind_returns_self(self) -> None:
        """`.bind(**kwargs)` mirrors LangChain's Runnable.bind — ignored in stub mode."""
        model = StubChatModel()
        bound = model.bind(temperature=0.7, max_output_tokens=1024)
        assert bound is model
        result = bound.invoke([HumanMessage(content="q")])
        assert isinstance(result, AIMessage)


class SamplePage(BaseModel):
    index: int = Field(default=0)
    extracted_text: str = Field(default="")


class SampleExtraction(BaseModel):
    pages: list[SamplePage] = Field(default_factory=list)


class TestStubChatModelStructuredOutput:
    def test_with_structured_output_returns_runnable(self) -> None:
        model = StubChatModel()
        runnable = model.with_structured_output(SampleExtraction)
        assert isinstance(runnable, StubStructuredRunnable)

    def test_structured_invoke_returns_schema_instance(self) -> None:
        model = StubChatModel()
        runnable = model.with_structured_output(SampleExtraction)
        result = runnable.invoke([HumanMessage(content="extract")])
        assert isinstance(result, SampleExtraction)

    @pytest.mark.asyncio
    async def test_structured_ainvoke_returns_schema_instance(self) -> None:
        model = StubChatModel()
        runnable = model.with_structured_output(SampleExtraction)
        result = await runnable.ainvoke([HumanMessage(content="extract")])
        assert isinstance(result, SampleExtraction)

    def test_structured_pages_default_populated(self) -> None:
        """The OCR pipeline expects at least one page. Stub yields one placeholder page."""
        model = StubChatModel()
        runnable = model.with_structured_output(SampleExtraction)
        result = runnable.invoke([HumanMessage(content="")])
        assert len(result.pages) >= 1
        assert result.pages[0].index == 0
        # SamplePage has default="" for extracted_text; required fields use "[stub]"
        # (see test_structured_output_with_required_fields for the production-schema check)
        assert isinstance(result.pages[0].extracted_text, str)

    def test_structured_output_with_required_list_of_scalars(self) -> None:
        """Required list[str] with no default must resolve to []."""

        class SchemaWithRequiredStringList(BaseModel):
            tags: list[str]

        model = StubChatModel()
        runnable = model.with_structured_output(SchemaWithRequiredStringList)
        result = runnable.invoke([HumanMessage(content="")])
        assert isinstance(result, SchemaWithRequiredStringList)
        assert result.tags == []

    def test_structured_output_with_required_fields(self) -> None:
        """Real production schema has required fields with no defaults — must still build."""
        from src.ocr.data_models.llm_response import DocumentExtraction

        model = StubChatModel()
        runnable = model.with_structured_output(DocumentExtraction)
        result = runnable.invoke([HumanMessage(content="")])
        assert isinstance(result, DocumentExtraction)
        assert len(result.pages) >= 1
        assert result.pages[0].page_number == 0
        assert isinstance(result.pages[0].page_summary, str)
        assert isinstance(result.pages[0].page_insight, str)
