"""Shared pieces of the trace export pipeline.

init(), NeatlogsClient and the env bootstrap each build their own tracer
provider. The span limits live here so the three stay in step.
"""

from __future__ import annotations

import os

from opentelemetry.sdk.environment_variables import (
    OTEL_ATTRIBUTE_COUNT_LIMIT,
    OTEL_SPAN_ATTRIBUTE_COUNT_LIMIT,
)
from opentelemetry.sdk.trace import SpanLimits

DEFAULT_MAX_SPAN_ATTRIBUTES = 10_000


def default_span_limits() -> SpanLimits:
    """
    OpenTelemetry defaults to 128 span attributes, which can silently drop semantic
    attributes when instrumenting LLM apps (retrieval docs, tool IO, etc).

    If the user explicitly sets OTel limits via env vars, respect that. Otherwise
    default to a larger max-span-attributes value (matching OpenInference's approach).
    """
    span_limit = os.getenv(OTEL_SPAN_ATTRIBUTE_COUNT_LIMIT, "")
    general_limit = os.getenv(OTEL_ATTRIBUTE_COUNT_LIMIT, "")
    if span_limit.strip() or general_limit.strip():
        return SpanLimits()
    return SpanLimits(max_span_attributes=DEFAULT_MAX_SPAN_ATTRIBUTES)
