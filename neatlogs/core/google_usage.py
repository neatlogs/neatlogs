"""Normalize Google Gemini usage metadata to the shared completion convention."""

from __future__ import annotations

from typing import Any

# Span attribute telling the backend that ``neatlogs.llm.token_count.completion``
# already includes thinking, so it must not add ``reasoning`` to it again.
# Spans from older SDKs lack it; the backend recognizes those separately.
COMPLETION_INCLUDES_REASONING = "neatlogs.llm.token_count.completion_includes_reasoning"


def _count(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def google_completion_tokens(candidates_tokens: Any, thoughts_tokens: Any) -> Any:
    """Return Gemini output tokens with thinking included.

    Gemini reports thinking (``thoughts_token_count``) outside the answer count
    (``candidates_token_count``); ``total_token_count`` is prompt + tool-use
    prompt + thoughts + candidates, and thinking is billed at the output rate.
    Every other provider NeatLogs maps reports reasoning inside completion, and
    ``neatlogs.llm.token_count.reasoning`` stays a breakdown of it, so the
    completion count here is candidates + thoughts.

    Values that are not plain non-negative ints are left as reported.
    """
    candidates = _count(candidates_tokens)
    thoughts = _count(thoughts_tokens)
    if candidates is None and thoughts is None:
        return candidates_tokens
    return (candidates or 0) + (thoughts or 0)


def mark_completion_includes_reasoning(span: Any) -> None:
    """Record on ``span`` that its completion count already includes thinking."""
    span.set_attribute(COMPLETION_INCLUDES_REASONING, True)
