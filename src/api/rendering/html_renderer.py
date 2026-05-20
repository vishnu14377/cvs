"""Render agent responses to Unqork-safe HTML.

Unqork HTML Element constraints:
- No <script> tags
- No event handler attributes (onclick, onload, etc.)
- No <iframe> elements
- CSS via inline style attributes only
- Allowed tags: div, p, span, h1-h6, ul, ol, li, table, tr, td, th,
  strong, em, a, img, br, hr
- <a> tags must use target="_blank"
- <img> tags may use src and alt
"""

from __future__ import annotations

import base64
import os
import re

import bleach
import markdown
from bleach.css_sanitizer import CSSSanitizer

_ALLOWED_TAGS = [
    "div",
    "p",
    "span",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "ul",
    "ol",
    "li",
    "table",
    "tr",
    "td",
    "th",
    "thead",
    "tbody",
    "strong",
    "em",
    "a",
    "img",
    "br",
    "hr",
    "pre",
    "code",
]

_ALLOWED_ATTRIBUTES = {
    "*": ["class", "style", "id", "title", "role", "width", "height"],
    "a": ["href", "target"],
    "img": ["src", "alt"],
    "td": ["colspan", "rowspan"],
    "th": ["colspan", "rowspan"],
}

# Broad set of safe CSS properties — no layout/positioning that could break Unqork UI.
_CSS_SANITIZER = CSSSanitizer(
    allowed_css_properties=[
        "color",
        "background-color",
        "font-size",
        "font-weight",
        "font-style",
        "font-family",
        "text-align",
        "text-decoration",
        "line-height",
        "margin",
        "margin-top",
        "margin-bottom",
        "margin-left",
        "margin-right",
        "padding",
        "padding-top",
        "padding-bottom",
        "padding-left",
        "padding-right",
        "border",
        "border-color",
        "border-width",
        "border-style",
        "border-radius",
        "width",
        "height",
        "max-width",
        "max-height",
        "min-width",
        "min-height",
        "display",
        "float",
        "clear",
        "overflow",
        "white-space",
        "word-break",
        "opacity",
        "visibility",
    ]
)

_WRAPPER_STYLE = (
    "font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;"
    " font-size: 14px; line-height: 1.5; color: #1a1a1a;"
)

# Matches trailing semicolons inside style attribute values, e.g. style="color:red;"
_STYLE_TRAILING_SEMI_RE = re.compile(r'style="([^"]*?);\s*"')

# Matches <script> elements including their content
_SCRIPT_RE = re.compile(r"<script[^>]*>.*?</script>", re.DOTALL | re.IGNORECASE)

# bleach's protocols= allowlist must include "data:" to keep inline
# <img src="data:image/...;base64,..."> working, but a data: URI on <a href>
# is an XSS vector (clicking navigates to data:text/html,... which executes
# script). Strip data: hrefs from anchors post-sanitization.
_A_HREF_DATA_RE = re.compile(
    r'(<a\b[^>]*?)\s+href="\s*data:[^"]*"',
    re.IGNORECASE,
)


def _strip_scripts(html: str) -> str:
    """Remove <script> elements and their content entirely."""
    return _SCRIPT_RE.sub("", html)


def _strip_anchor_data_hrefs(html: str) -> str:
    """Remove `data:` hrefs from <a> tags while leaving <img src="data:..."> alone."""
    return _A_HREF_DATA_RE.sub(r"\1", html)


def _normalize_styles(html: str) -> str:
    """Remove trailing semicolons from inline style attribute values.

    bleach's CSSSanitizer always appends a trailing semicolon; this normalizes
    the output so callers receive the value exactly as supplied.
    """
    return _STYLE_TRAILING_SEMI_RE.sub(
        lambda m: f'style="{m.group(1).rstrip(";").strip()}"',
        html,
    )


def render_to_safe_html(text: str) -> str:
    """Convert agent text to Unqork-safe HTML.

    Converts markdown to HTML, then sanitizes to remove unsafe elements.

    Args:
        text: Agent response text (may contain markdown).

    Returns:
        Sanitized HTML string safe for Unqork HTML Element rendering.
    """
    if not isinstance(text, str):
        text = str(text)

    html = markdown.markdown(
        text,
        extensions=["tables", "fenced_code"],
    )

    # Remove script elements (tag + content) before bleach processes the input.
    # bleach strips the tag but leaves text content, so we must pre-strip.
    html = _strip_scripts(html)

    clean = bleach.clean(
        html,
        tags=_ALLOWED_TAGS,
        attributes=_ALLOWED_ATTRIBUTES,
        protocols=["http", "https", "mailto", "data"],
        css_sanitizer=_CSS_SANITIZER,
        strip=True,
    )

    clean = _strip_anchor_data_hrefs(clean)

    # Normalize trailing semicolons added by CSSSanitizer
    clean = _normalize_styles(clean)

    # Force target="_blank" on all links
    clean = clean.replace("<a ", '<a target="_blank" ')

    return f'<div style="{_WRAPPER_STYLE}">{clean}</div>'


def render_to_base64(text: str) -> str:
    """Convert agent text to base64-encoded Unqork-safe HTML.

    Args:
        text: Agent response text (may contain markdown).

    Returns:
        Base64-encoded string of the sanitized HTML.
    """
    html = render_to_safe_html(text)
    return base64.b64encode(html.encode("utf-8")).decode("ascii")


def _format_timestamp(iso_str: str) -> str:
    """Format an ISO 8601 timestamp to '23 Apr 2026, 10:06 AM'."""
    from datetime import datetime, timezone

    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        return dt.strftime("%-d %b %Y, %-I:%M %p")
    except (ValueError, AttributeError):
        return iso_str


def _escape_html(text: str) -> str:
    """Escape HTML special characters in user input."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


_CONVERSATION_WRAPPER_STYLE = (
    "font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Arial,sans-serif;"
    "color:#2c3e50;overflow-y:auto;"
)

_USER_CARD_STYLE = (
    "margin:12px 16px;padding:10px 14px;background:#f3e8ff;"
    "border:1px solid #d8b4fe;border-radius:8px;"
)

_USER_LABEL_STYLE = "font-weight:600;color:#6f2c91;font-size:13px;margin-bottom:4px;"

_USER_CONTENT_STYLE = "font-size:14px;"

_ASSISTANT_CARD_STYLE = (
    "margin:12px 16px;padding:10px 14px;background:#fff;"
    "border:1px solid #e5e5e5;border-radius:8px;"
)

_ASSISTANT_LABEL_STYLE = "font-weight:600;color:#6f2c91;font-size:13px;margin-bottom:4px;"

_ASSISTANT_CONTENT_STYLE = "font-size:14px;line-height:1.6;"

_FOOTER_STYLE = (
    "display:flex;justify-content:space-between;align-items:center;"
    "margin-top:8px;padding-top:6px;border-top:1px solid #f0f0f0;"
)

_TIMESTAMP_STYLE = "font-size:11px;color:#999;"

_SOURCES_STYLE = "font-size:11px;color:#6f2c91;"

_SOURCE_LINK_STYLE = "color:#6f2c91;text-decoration:underline;"


def _resolve_source_url(gcs_uri: str) -> str | None:
    """Convert a GCS URI to a signed HTTPS URL for the source PDF.

    Returns None when signing is unavailable (no credentials, etc.).
    """
    from src.core.gcs_client import generate_signed_url, json_uri_to_pdf_uri

    pdf_uri = json_uri_to_pdf_uri(gcs_uri)
    return generate_signed_url(pdf_uri)


def _source_display_label(document: str) -> str:
    """Build a short display label from a document name or full GCS URI."""
    basename = os.path.basename(document)
    if basename.endswith(".json"):
        basename = basename.removesuffix(".json") + ".pdf"
    return basename


def _render_sources_html(sources: list[dict]) -> str:
    """Render deduped source citations as clickable links (or plain text as fallback)."""
    seen: set[str] = set()
    unique_sources: list[dict] = []
    for s in sources:
        key = (s.get("document", ""), s.get("page"))
        if key not in seen:
            seen.add(key)
            unique_sources.append(s)

    parts: list[str] = []
    for s in unique_sources:
        label = _source_display_label(s.get("document", "Unknown"))
        page = s.get("page")
        gcs_uri = s.get("gcs_uri")

        url = _resolve_source_url(gcs_uri) if gcs_uri else None

        if url:
            href = f"{url}#page={page}" if page else url
            display = f"{label} (Page {page})" if page else label
            parts.append(
                f'<a href="{href}" target="_blank" style="{_SOURCE_LINK_STYLE}">{display}</a>'
            )
        else:
            display = f"{label} (Page {page})" if page else label
            parts.append(display)

    return ", ".join(parts)


def render_conversation_html(
    messages: list[dict],
    theme: str = "purple",
) -> str:
    """Render a full conversation as Unqork-safe HTML.

    Each API response returns the entire conversation history as one HTML
    block. Unqork's HTML Element component replaces its content on every
    data change (no append), so we must return everything each time.

    Args:
        messages: List of message dicts with keys: role, content,
                  and optionally sources and generated_at.
        theme: Color theme (only "purple" supported currently).

    Returns:
        A single HTML string with all messages as styled cards.
        All CSS inline. No scripts. Safe for Unqork HTML Element.
    """
    cards: list[str] = []

    for msg in messages:
        role = msg.get("role", "")

        if role == "user":
            escaped = _escape_html(str(msg.get("content", "")))
            cards.append(
                f'<div style="{_USER_CARD_STYLE}">'
                f'<div style="{_USER_LABEL_STYLE}">You</div>'
                f'<div style="{_USER_CONTENT_STYLE}">{escaped}</div>'
                f"</div>"
            )

        elif role == "assistant":
            content = str(msg.get("content", ""))
            rendered = render_to_safe_html(content)
            # Strip the outer wrapper div that render_to_safe_html adds
            # so we can wrap it in our own card styling.
            inner = rendered
            if inner.startswith("<div"):
                close_bracket = inner.index(">")
                inner = inner[close_bracket + 1 :]
                if inner.endswith("</div>"):
                    inner = inner[: -len("</div>")]

            footer_parts: list[str] = []

            generated_at = msg.get("generated_at")
            if generated_at:
                ts = _format_timestamp(generated_at)
                footer_parts.append(
                    f'<span style="{_TIMESTAMP_STYLE}">Generated: {ts}</span>'
                )

            sources = msg.get("sources")
            if sources:
                citations = _render_sources_html(sources)
                footer_parts.append(
                    f'<span style="{_SOURCES_STYLE}">Sources: {citations}</span>'
                )

            footer_html = ""
            if footer_parts:
                footer_html = (
                    f'<div style="{_FOOTER_STYLE}">'
                    + "".join(footer_parts)
                    + "</div>"
                )

            cards.append(
                f'<div style="{_ASSISTANT_CARD_STYLE}">'
                f'<div style="{_ASSISTANT_LABEL_STYLE}">AI Assistant</div>'
                f'<div style="{_ASSISTANT_CONTENT_STYLE}">{inner}</div>'
                f"{footer_html}"
                f"</div>"
            )
        # Skip tool messages and other types

    return f'<div style="{_CONVERSATION_WRAPPER_STYLE}">{"".join(cards)}</div>'
