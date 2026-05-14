"""Tests for render_conversation_html — full-conversation Unqork rendering."""

import re

import pytest


class TestRenderConversationHtml:
    """Tests for render_conversation_html."""

    def test_single_turn_produces_user_and_assistant_cards(self):
        from src.api.rendering.html_renderer import render_conversation_html

        messages = [
            {"role": "user", "content": "What is the timeline?"},
            {"role": "assistant", "content": "**Timeline:**\n- 08:00 AM – Admission"},
        ]
        result = render_conversation_html(messages)

        assert "You" in result
        assert "What is the timeline?" in result
        assert "AI Assistant" in result
        assert "<strong>Timeline:</strong>" in result
        assert "08:00 AM" in result

    def test_multiple_turns_all_present_in_order(self):
        from src.api.rendering.html_renderer import render_conversation_html

        messages = [
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "First answer"},
            {"role": "user", "content": "Second question"},
            {"role": "assistant", "content": "Second answer"},
        ]
        result = render_conversation_html(messages)

        first_q = result.index("First question")
        first_a = result.index("First answer")
        second_q = result.index("Second question")
        second_a = result.index("Second answer")
        assert first_q < first_a < second_q < second_a

    def test_assistant_markdown_rendered_to_html(self):
        from src.api.rendering.html_renderer import render_conversation_html

        messages = [
            {"role": "user", "content": "Show me a list"},
            {"role": "assistant", "content": "- item 1\n- item 2"},
        ]
        result = render_conversation_html(messages)

        assert "<li>" in result
        assert "item 1" in result

    def test_source_references_in_footer(self):
        from src.api.rendering.html_renderer import render_conversation_html

        messages = [
            {"role": "user", "content": "Query"},
            {
                "role": "assistant",
                "content": "Response",
                "sources": [
                    {"document": "EMR Chart", "page": 1},
                    {"document": "Itemized Bill", "page": 3},
                ],
            },
        ]
        result = render_conversation_html(messages)

        assert "EMR Chart" in result
        assert "Itemized Bill" in result

    def test_timestamp_in_footer(self):
        from src.api.rendering.html_renderer import render_conversation_html

        messages = [
            {"role": "user", "content": "Query"},
            {
                "role": "assistant",
                "content": "Response",
                "generated_at": "2026-04-23T10:06:00Z",
            },
        ]
        result = render_conversation_html(messages)

        assert "23 Apr 2026" in result

    def test_empty_conversation_returns_wrapper(self):
        from src.api.rendering.html_renderer import render_conversation_html

        result = render_conversation_html([])

        assert "<div" in result
        assert "font-family" in result

    def test_user_message_html_escaped(self):
        from src.api.rendering.html_renderer import render_conversation_html

        messages = [
            {"role": "user", "content": "<script>alert('xss')</script>"},
        ]
        result = render_conversation_html(messages)

        assert "<script>" not in result
        assert "&lt;script&gt;" in result

    def test_no_script_tags_in_output(self):
        from src.api.rendering.html_renderer import render_conversation_html

        messages = [
            {"role": "user", "content": "test"},
            {"role": "assistant", "content": "<script>bad</script>Safe content"},
        ]
        result = render_conversation_html(messages)

        assert "<script>" not in result
        assert "Safe content" in result

    def test_no_event_handlers_in_output(self):
        from src.api.rendering.html_renderer import render_conversation_html

        messages = [
            {"role": "user", "content": "test"},
            {"role": "assistant", "content": '<div onclick="evil()">text</div>'},
        ]
        result = render_conversation_html(messages)

        assert "onclick" not in result

    def test_all_css_is_inline(self):
        from src.api.rendering.html_renderer import render_conversation_html

        messages = [
            {"role": "user", "content": "test"},
            {"role": "assistant", "content": "response"},
        ]
        result = render_conversation_html(messages)

        assert "<style>" not in result
        assert "style=" in result

    def test_wrapper_has_overflow_scroll(self):
        from src.api.rendering.html_renderer import render_conversation_html

        result = render_conversation_html([])

        assert "overflow-y:auto" in result

    def test_purple_theme_colors(self):
        from src.api.rendering.html_renderer import render_conversation_html

        messages = [
            {"role": "user", "content": "test"},
            {"role": "assistant", "content": "response"},
        ]
        result = render_conversation_html(messages)

        assert "#6f2c91" in result
        assert "#f3e8ff" in result

    def test_assistant_without_sources_no_sources_line(self):
        from src.api.rendering.html_renderer import render_conversation_html

        messages = [
            {"role": "user", "content": "test"},
            {"role": "assistant", "content": "response"},
        ]
        result = render_conversation_html(messages)

        assert "Sources:" not in result

    def test_tool_messages_skipped(self):
        from src.api.rendering.html_renderer import render_conversation_html

        messages = [
            {"role": "user", "content": "test"},
            {"role": "tool", "content": "internal tool output"},
            {"role": "assistant", "content": "response"},
        ]
        result = render_conversation_html(messages)

        assert "internal tool output" not in result
