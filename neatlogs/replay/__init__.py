"""Counterfactual trace replay (MVP: LLM-span replay + comparison).

Safe by default: only LLM spans are re-issued, tool calls in replayed output
are recorded as data and never executed. See ``docs/counterfactual-replay.md``.
"""

from .compare import compare_spans, compare_traces
from .executor import replay_trace
from .extract import extract_llm_request, extract_messages, extract_tools
from .openai_caller import openai_caller
from .store import InMemoryTraceStore, TraceStore
from .types import (
    LLMRequest,
    LLMResponse,
    ProviderError,
    ReplayedSpan,
    ReplayError,
    ReplayOverrides,
    ReplayResult,
    SpanFieldDiff,
    ToolCallData,
    TraceComparison,
    TraceNotFoundError,
    UnsupportedSpanError,
)

__all__ = [
    "replay_trace",
    "compare_traces",
    "compare_spans",
    "extract_llm_request",
    "extract_messages",
    "extract_tools",
    "openai_caller",
    "InMemoryTraceStore",
    "TraceStore",
    "LLMRequest",
    "LLMResponse",
    "ToolCallData",
    "ReplayedSpan",
    "ReplayError",
    "TraceNotFoundError",
    "UnsupportedSpanError",
    "ProviderError",
    "ReplayOverrides",
    "ReplayResult",
    "SpanFieldDiff",
    "TraceComparison",
]
