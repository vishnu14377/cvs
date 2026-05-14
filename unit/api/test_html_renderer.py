"""Tests for the Unqork-safe HTML renderer."""

import base64
from unittest.mock import patch

from src.api.rendering.html_renderer import (
    render_conversation_html,
    render_to_base64,
    render_to_safe_html,
)


class TestRenderToSafeHtml:
    """Tests for render_to_safe_html."""

    def test_plain_text_wrapped_in_div(self):
        result = render_to_safe_html("Hello world")
        assert "<div" in result
        assert "Hello world" in result

    def test_markdown_bold_converted(self):
        result = render_to_safe_html("This is **bold** text")
        assert "<strong>bold</strong>" in result

    def test_markdown_list_converted(self):
        result = render_to_safe_html("- item 1\n- item 2")
        assert "<li>" in result
        assert "item 1" in result

    def test_script_tags_stripped(self):
        result = render_to_safe_html("<script>alert('xss')</script>Hello")
        assert "<script>" not in result
        assert "alert" not in result
        assert "Hello" in result

    def test_event_handlers_stripped(self):
        result = render_to_safe_html('<div onclick="alert()">Click</div>')
        assert "onclick" not in result
        assert "Click" in result

    def test_iframe_stripped(self):
        result = render_to_safe_html('<iframe src="http://evil.com"></iframe>Safe')
        assert "<iframe" not in result
        assert "Safe" in result

    def test_inline_style_preserved(self):
        result = render_to_safe_html('<p style="color:red">Red text</p>')
        assert 'style="color:red"' in result

    def test_allowed_tags_preserved(self):
        html = "<p>Text</p><ul><li>Item</li></ul><strong>Bold</strong>"
        result = render_to_safe_html(html)
        assert "<p>" in result
        assert "<ul>" in result
        assert "<li>" in result
        assert "<strong>" in result

    def test_img_src_preserved(self):
        result = render_to_safe_html('<img src="data:image/png;base64,abc" alt="chart">')
        assert "src=" in result
        assert "alt=" in result

    def test_strips_data_href_on_anchor(self):
        """data: URI on <a href> is an XSS vector — must be removed."""
        result = render_to_safe_html("[click](data:text/html,<script>alert(1)</script>)")
        assert 'href="data:' not in result
        assert "data:text/html" not in result

    def test_img_data_src_still_allowed(self):
        """data: on <img src> is still allowed (inline images are legitimate)."""
        result = render_to_safe_html('<img src="data:image/png;base64,iVBOR" alt="x">')
        assert 'src="data:image/png;base64,iVBOR"' in result


class TestRenderToBase64:
    """Tests for render_to_base64."""

    def test_returns_valid_base64(self):
        result = render_to_base64("Hello world")
        decoded = base64.b64decode(result).decode("utf-8")
        assert "Hello world" in decoded

    def test_base64_contains_safe_html(self):
        result = render_to_base64("**bold** text")
        decoded = base64.b64decode(result).decode("utf-8")
        assert "<strong>bold</strong>" in decoded
        assert "<script>" not in decoded


class TestRenderConversationHtmlSources:
    """Tests for source citation rendering in render_conversation_html."""

    def _make_messages(self, sources):
        return [
            {"role": "assistant", "content": "Answer text", "sources": sources},
        ]

    def test_sources_rendered_as_anchor_tags(self):
        sources = [
            {"document": "sample_adr_1_p1-2.json", "page": 1, "gcs_uri": "gs://bucket/base/session/extracted_text/sample_adr_1_p1-2.json"},
        ]
        with patch("src.api.rendering.html_renderer._resolve_source_url", return_value="https://storage.googleapis.com/signed"):
            result = render_conversation_html(self._make_messages(sources))
        assert "<a " in result
        assert 'href="https://storage.googleapis.com/signed#page=1"' in result
        assert 'target="_blank"' in result

    def test_sources_deduped_by_document_name(self):
        sources = [
            {"document": "sample_adr_1_p1-2.json", "page": 1, "gcs_uri": "gs://bucket/base/session/extracted_text/sample_adr_1_p1-2.json"},
            {"document": "sample_adr_1_p1-2.json", "page": 1, "gcs_uri": "gs://bucket/base/session/extracted_text/sample_adr_1_p1-2.json"},
            {"document": "sample_adr_1_p1-2.json", "page": 1, "gcs_uri": "gs://bucket/base/session/extracted_text/sample_adr_1_p1-2.json"},
        ]
        with patch("src.api.rendering.html_renderer._resolve_source_url", return_value="https://storage.googleapis.com/signed"):
            result = render_conversation_html(self._make_messages(sources))
        assert result.count("sample_adr_1_p1-2") == 1

    def test_source_label_shows_basename_not_full_path(self):
        sources = [
            {"document": "gs://bucket/base/session/extracted_text/sample_adr_1_p1-2.json", "page": 2, "gcs_uri": "gs://bucket/base/session/extracted_text/sample_adr_1_p1-2.json"},
        ]
        with patch("src.api.rendering.html_renderer._resolve_source_url", return_value="https://signed.url"):
            result = render_conversation_html(self._make_messages(sources))
        assert "gs://bucket" not in result
        assert "sample_adr_1_p1-2" in result

    def test_page_fragment_appended_to_href(self):
        sources = [
            {"document": "doc.json", "page": 5, "gcs_uri": "gs://bucket/extracted_text/doc.json"},
        ]
        with patch("src.api.rendering.html_renderer._resolve_source_url", return_value="https://signed.url"):
            result = render_conversation_html(self._make_messages(sources))
        assert "#page=5" in result

    def test_no_page_fragment_when_page_missing(self):
        sources = [
            {"document": "doc.json", "gcs_uri": "gs://bucket/extracted_text/doc.json"},
        ]
        with patch("src.api.rendering.html_renderer._resolve_source_url", return_value="https://signed.url"):
            result = render_conversation_html(self._make_messages(sources))
        assert "#page=" not in result
        assert 'href="https://signed.url"' in result

    def test_fallback_when_signing_unavailable(self):
        """When signed URL generation fails, sources still render as text."""
        sources = [
            {"document": "doc.json", "page": 1, "gcs_uri": "gs://bucket/extracted_text/doc.json"},
        ]
        with patch("src.api.rendering.html_renderer._resolve_source_url", return_value=None):
            result = render_conversation_html(self._make_messages(sources))
        assert "Sources:" in result
        assert "doc" in result
        assert "<a " not in result

    def test_pdf_extension_shown_in_label(self):
        """Display label should reference the PDF, not the JSON."""
        sources = [
            {"document": "sample_adr_1_p1-2.json", "page": 1, "gcs_uri": "gs://bucket/base/session/extracted_text/sample_adr_1_p1-2.json"},
        ]
        with patch("src.api.rendering.html_renderer._resolve_source_url", return_value="https://signed.url"):
            result = render_conversation_html(self._make_messages(sources))
        assert "sample_adr_1_p1-2.pdf" in result
        assert ".json" not in result
