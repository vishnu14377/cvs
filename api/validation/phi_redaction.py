"""PHI (Protected Health Information) redaction utility.

Applies regex-based redaction to text before it reaches external surfaces
(logs, MongoDB, API responses). The agent's internal context is NOT redacted
— it needs full patient details to function.
"""

from __future__ import annotations

import re

_PHI_PATTERNS = [
    (r"\bMRN\s*[:=]?\s*[\d][\d\s-]{3,}", "[REDACTED_MRN]"),
    (r"\bSSN\s*[:=]?\s*\d{3}[-.\s]?\d{2}[-.\s]?\d{4}", "[REDACTED_SSN]"),
    (r"\b\d{3}[-.\s]\d{2}[-.\s]\d{4}\b", "[REDACTED_SSN]"),
    (r"(?:DOB|date of birth)\s*[:=]?\s*\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}", "[REDACTED_DOB]"),
    (r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b", "[REDACTED_PHONE]"),
    (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "[REDACTED_EMAIL]"),
    (r"(?:Member\s*ID|Insurance\s*ID)\s*[:=]?\s*[A-Z]?\d{5,}", "[REDACTED_MEMBER_ID]"),
    (r"(?:Patient|Name|Subscriber)\s*[:=]?\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+", "[REDACTED_NAME]"),
    (r"(?:Patient|Name|Subscriber)\s*[:=]?\s*[A-Z]{2,}(?:,\s*[A-Z]{2,})+", "[REDACTED_NAME]"),
]

_COMPILED_PATTERNS = [(re.compile(p, re.IGNORECASE), r) for p, r in _PHI_PATTERNS]


def redact_phi(text) -> str:
    """Redact PHI patterns from text.

    Args:
        text: Input text (str or None).

    Returns:
        Text with PHI patterns replaced by redaction tokens.
    """
    if not text:
        return ""
    if not isinstance(text, str):
        text = str(text)

    result = text
    for pattern, replacement in _COMPILED_PATTERNS:
        result = pattern.sub(replacement, result)
    return result
