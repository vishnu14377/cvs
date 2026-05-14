"""Deterministic chat-model stub for CI integration tests.

Implements just enough of the ChatGoogleGenerativeAI surface area to satisfy the
agent graph (generate_node, tool_node) and the OCR pipeline (llm_ocr_client).
Never emits tool calls — the graph terminates on the first generate pass.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel
from src.core.logger import get_logger

logger = get_logger(__name__)


def _last_human_content(messages: str | list[BaseMessage]) -> str:
    """Extract the last human-turn content for deterministic echoing."""
    if isinstance(messages, str):
        return messages
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            return str(m.content)
    return ""


def _build_schema_default(schema: type[BaseModel]) -> BaseModel:
    """Construct a schema instance using field defaults.

    For list[BaseModel] fields, produce a one-element list so pipelines that
    iterate over results (e.g. OCR page list) have something to work with.
    For required scalar fields with no default, supply a type-appropriate
    placeholder so Pydantic validation passes.
    """
    from pydantic_core import PydanticUndefined

    values: dict[str, Any] = {}
    for name, field in schema.model_fields.items():
        annotation = field.annotation
        origin = getattr(annotation, "__origin__", None)
        if origin is list:
            inner_args = getattr(annotation, "__args__", ())
            inner = inner_args[0] if inner_args else None
            if inner is not None and isinstance(inner, type) and issubclass(inner, BaseModel):
                values[name] = [_build_schema_default(inner)]
            elif field.default is PydanticUndefined and field.default_factory is None:
                # Required list of non-BaseModel items — empty list is valid.
                values[name] = []
        elif field.default is PydanticUndefined and field.default_factory is None:
            if annotation is int:
                values[name] = 0
            elif annotation is str:
                values[name] = "[stub]"
            elif annotation is float:
                values[name] = 0.0
            elif annotation is bool:
                values[name] = False
    return schema(**values)


class StubStructuredRunnable:
    """Runnable returned by StubChatModel.with_structured_output()."""

    def __init__(self, schema: type[BaseModel]) -> None:
        self._schema = schema

    def invoke(self, messages: str | list[BaseMessage], **_: Any) -> BaseModel:
        logger.debug("StubStructuredRunnable.invoke schema=%s", self._schema.__name__)
        return _build_schema_default(self._schema)

    async def ainvoke(self, messages: str | list[BaseMessage], **_: Any) -> BaseModel:
        logger.debug("StubStructuredRunnable.ainvoke schema=%s", self._schema.__name__)
        return _build_schema_default(self._schema)


class StubChatModel:
    """Deterministic offline stand-in for ChatGoogleGenerativeAI."""

    def __init__(self) -> None:
        logger.info("StubChatModel initialized (VERTEX_AI_MODE=stub)")

    def _echo(self, messages: str | list[BaseMessage]) -> AIMessage:
        last = _last_human_content(messages)
        return AIMessage(content=f"[stub] echo: {last}", tool_calls=[])

    def invoke(self, messages: str | list[BaseMessage], **_: Any) -> AIMessage:
        return self._echo(messages)

    async def ainvoke(self, messages: str | list[BaseMessage], **_: Any) -> AIMessage:
        return self._echo(messages)

    def bind_tools(
        self,
        _tools: Sequence[BaseTool] | None = None,
        **_: Any,
    ) -> StubChatModel:
        return self

    def bind(self, **_: Any) -> StubChatModel:
        """Mirror Runnable.bind — tool/generation overrides are ignored in stub mode."""
        return self

    def with_structured_output(
        self,
        schema: type[BaseModel],
        **_: Any,
    ) -> StubStructuredRunnable:
        return StubStructuredRunnable(schema)
