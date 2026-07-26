"""Span reading for the cost engine.

Extracts token usage from raw or processed span JSONL records written
by ``NEATLOGS_LOG_SPANS=true`` / ``NEATLOGS_LOG_RAW_SPANS=true``.

The reader is intentionally permissive: missing fields default to 0,
malformed lines are skipped, and the same parser handles both the
processed ``span_data`` shape and the raw ``ReadableSpan.to_json()``
shape by reading whatever attributes the record declares.
"""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterator, List, Sequence, Union

PathLike = Union[str, os.PathLike]


@dataclasses.dataclass
class SpanUsage:
    """One LLM span's token usage, extracted from span attributes."""

    span_id: str
    trace_id: str
    model: str
    provider: str | None
    prompt_tokens: int
    completion_tokens: int
    cache_creation_tokens: int
    cache_read_tokens: int
    reasoning_tokens: int

    @property
    def input_total(self) -> int:
        return self.prompt_tokens + self.cache_creation_tokens + self.cache_read_tokens

    @property
    def output_total(self) -> int:
        return self.completion_tokens + self.reasoning_tokens

    @property
    def has_tokens(self) -> bool:
        return any(
            t > 0
            for t in (
                self.prompt_tokens,
                self.completion_tokens,
                self.cache_creation_tokens,
                self.cache_read_tokens,
                self.reasoning_tokens,
            )
        )

    @property
    def uses_prompt_cache(self) -> bool:
        return self.cache_creation_tokens > 0 or self.cache_read_tokens > 0

    @property
    def uses_reasoning(self) -> bool:
        return self.reasoning_tokens > 0


def _int_attr(attrs: Dict[str, Any], *keys: str) -> int:
    for k in keys:
        v = attrs.get(k)
        if isinstance(v, bool):
            continue
        if isinstance(v, int) and v >= 0:
            return v
        if isinstance(v, float) and v.is_integer() and v >= 0:
            return int(v)
    return 0


def _extract_usage(obj: Dict[str, Any]) -> SpanUsage:
    attrs = obj.get("attributes") or {}
    if not isinstance(attrs, dict):
        attrs = {}
    model = (
        attrs.get("neatlogs.llm.model_name")
        or attrs.get("gen_ai.response.model")
        or attrs.get("gen_ai.request.model")
        or ""
    )
    provider = (
        attrs.get("neatlogs.llm.provider")
        or attrs.get("neatlogs.llm.system")
        or attrs.get("gen_ai.system")
    )
    if isinstance(provider, str):
        provider = provider.lower() or None
    return SpanUsage(
        span_id=str(obj.get("span_id") or obj.get("context", {}).get("span_id") or ""),
        trace_id=str(obj.get("trace_id") or obj.get("context", {}).get("trace_id") or ""),
        model=str(model),
        provider=provider,
        prompt_tokens=_int_attr(
            attrs,
            "neatlogs.llm.token_count.prompt",
            "gen_ai.usage.input_tokens",
            "gen_ai.usage.prompt_tokens",
        ),
        completion_tokens=_int_attr(
            attrs,
            "neatlogs.llm.token_count.completion",
            "gen_ai.usage.output_tokens",
            "gen_ai.usage.completion_tokens",
        ),
        cache_creation_tokens=_int_attr(
            attrs,
            "neatlogs.llm.token_count.cache_creation",
            "gen_ai.usage.cache_creation_input_tokens",
        ),
        cache_read_tokens=_int_attr(
            attrs,
            "neatlogs.llm.token_count.cache_read",
            "gen_ai.usage.cache_read_input_tokens",
        ),
        reasoning_tokens=_int_attr(
            attrs,
            "neatlogs.llm.token_count.reasoning",
            "gen_ai.usage.reasoning_tokens",
        ),
    )


def _iter_json_objects(text: str) -> Iterator[Dict[str, Any]]:
    """Brace-balanced parser; tolerates newlines inside string values."""
    depth = 0
    start = -1
    in_string = False
    escape = False
    for i, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                try:
                    yield json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    pass
                start = -1
            elif depth < 0:
                depth = 0


def _read_usages(path: PathLike) -> List[SpanUsage]:
    p = Path(path)
    if not p.exists():
        return []
    text = p.read_text(encoding="utf-8", errors="replace")
    if not text.strip():
        return []
    return [_extract_usage(obj) for obj in _iter_json_objects(text) if isinstance(obj, dict)]


def _read_paths(paths: Sequence[PathLike]) -> List[SpanUsage]:
    out: List[SpanUsage] = []
    for raw in paths:
        out.extend(_read_usages(raw))
    return out
