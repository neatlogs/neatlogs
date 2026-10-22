"""Canonical span-kind resolution shared by normalization and filtering."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

SEMANTIC_SPAN_KINDS = frozenset(
    {
        "WORKFLOW",
        "AGENT",
        "CHAIN",
        "TOOL",
        "RETRIEVER",
        "EMBEDDING",
        "GUARDRAIL",
        "LLM",
        "RERANKER",
        "VECTOR_STORE",
        "TASK",
        "EVALUATOR",
        "LOG",
        "MEMORY",
        "MCP_TOOL",
    }
)


def resolve_explicit_span_kind(attributes: Mapping[str, Any]) -> str:
    """Resolve explicit kinds with the canonical Neatlogs attribute first."""

    for key in (
        "neatlogs.span.kind",
        "openinference.span.kind",
        "traceloop.span.kind",
    ):
        value = str(attributes.get(key) or "").strip()
        if value:
            return value
    return ""


def normalize_explicit_span_kind(attributes: Mapping[str, Any], values: Mapping[str, str]) -> str:
    value = resolve_explicit_span_kind(attributes)
    if not value:
        return ""
    return values.get(value, values.get(value.upper(), value))
