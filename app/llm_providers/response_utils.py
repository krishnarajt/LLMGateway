"""Helpers for normalizing provider response text."""

from __future__ import annotations

import re
from typing import Any, Iterable

THINKING_FIELD_NAMES = (
    "reasoning_content",
    "reasoning",
    "thinking",
    "thought",
    "thoughts",
)

_TAG_BLOCK_RE = re.compile(
    r"(?is)<\s*(think|thinking|thought|reasoning)\b[^>]*>(.*?)<\s*/\s*\1\s*>"
)
_BRACKET_BLOCK_RE = re.compile(
    r"(?is)\[\s*(think|thinking|thought|reasoning)\s*\](.*?)\[\s*/\s*\1\s*\]"
)
_UNCLOSED_TAG_RE = re.compile(
    r"(?is)<\s*(think|thinking|thought|reasoning)\b[^>]*>.*$"
)
_UNCLOSED_BRACKET_RE = re.compile(
    r"(?is)\[\s*(think|thinking|thought|reasoning)\s*\].*$"
)
_LABELLED_PREFIX_RE = re.compile(
    r"(?is)^\s*(?:thoughts?|thinking|reasoning)(?:\s+summary)?\s*:\s*"
    r"(?P<thinking>.*?)"
    r"\n\s*(?:answer|final(?:\s+answer)?)\s*:\s*"
)


def text_from_content(value: Any) -> str:
    """Convert common provider content shapes into plain text."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(text_from_content(item) for item in value)
    if isinstance(value, dict):
        for key in ("text", "content", "output_text"):
            if key in value:
                return text_from_content(value[key])
        return ""
    return str(value)


def join_text_parts(values: Iterable[Any]) -> str:
    """Join non-empty text-ish values with readable spacing."""
    parts = [text_from_content(value).strip() for value in values]
    return "\n\n".join(part for part in parts if part)


def split_thinking_from_text(text: Any) -> tuple[str, str]:
    """Return final-answer text and extracted thinking/reasoning text."""
    content = text_from_content(text)
    thinking_parts: list[str] = []

    def collect(match: re.Match[str]) -> str:
        thinking = match.group(2).strip()
        if thinking:
            thinking_parts.append(thinking)
        return ""

    content = _TAG_BLOCK_RE.sub(collect, content)
    content = _BRACKET_BLOCK_RE.sub(collect, content)

    labelled = _LABELLED_PREFIX_RE.match(content)
    if labelled:
        thinking = labelled.group("thinking").strip()
        if thinking:
            thinking_parts.append(thinking)
        content = content[labelled.end() :]

    for pattern in (_UNCLOSED_TAG_RE, _UNCLOSED_BRACKET_RE):
        unclosed = pattern.search(content)
        if unclosed:
            thinking = unclosed.group(0).strip()
            if thinking:
                thinking_parts.append(thinking)
            content = content[: unclosed.start()]

    return content.strip(), join_text_parts(thinking_parts)


def normalized_response(
    content: Any,
    usage: dict | None = None,
    include_thinking: bool = False,
    thinking_parts: Iterable[Any] = (),
) -> dict:
    """Build a gateway response with hidden thinking by default."""
    final_content, inline_thinking = split_thinking_from_text(content)
    thinking = join_text_parts([*thinking_parts, inline_thinking])

    result = {"content": final_content, "usage": usage}
    if include_thinking and thinking:
        result["thinking"] = thinking
    return result
