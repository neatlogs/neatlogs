"""Shared types for counterfactual trace replay.

MVP scope is LLM-span replay: re-issue one captured LLM invocation with an
explicit override and record the result as a NEW trace linked to the original.
Tool calls in replayed output are recorded as data only and never executed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional


class ReplayError(Exception):
    """Base error for replay failures."""


class TraceNotFoundError(ReplayError):
    """Raised when a trace_id cannot be resolved via the TraceStore."""


class UnsupportedSpanError(ReplayError):
    """Raised when a span kind/provider cannot be replayed in this MVP."""

    def __init__(self, message: str, *, kind: str = "", provider: str = "") -> None:
        super().__init__(message)
        self.kind = kind
        self.provider = provider


class ProviderError(ReplayError):
    """Raised when the underlying LLM call fails during replay."""


def span_id(span: Any) -> str:
    """Hex span id for ReadableSpan, SDK Span, or ReplayedSpan snapshots."""
    if hasattr(span, "get_span_context"):
        return f"{span.get_span_context().span_id:016x}"
    context = getattr(span, "context", None)
    if context is not None:
        return f"{context.span_id:016x}"
    return str(getattr(span, "span_id", "unknown"))


def trace_id(span: Any) -> str:
    """Hex trace id for ReadableSpan, SDK Span, or ReplayedSpan snapshots."""
    if hasattr(span, "get_span_context"):
        return f"{span.get_span_context().trace_id:032x}"
    context = getattr(span, "context", None)
    if context is not None:
        return f"{context.trace_id:032x}"
    return str(getattr(span, "trace_id", "unknown"))


@dataclass(frozen=True)
class ReplayOverrides:
    """Explicit counterfactual deltas. Only set fields are applied."""

    model: Optional[str] = None
    messages: Optional[List[Dict[str, Any]]] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None

    _FIELDS = frozenset(
        {
            "model",
            "messages",
            "temperature",
            "top_p",
            "max_tokens",
            "frequency_penalty",
            "presence_penalty",
        }
    )

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "ReplayOverrides":
        """Build overrides from a plain dict, rejecting unknown keys."""
        unknown = set(values) - cls._FIELDS
        if unknown:
            raise ReplayError(f"Unknown override keys: {sorted(unknown)}")
        messages = values.get("messages")
        if messages is not None:
            if not isinstance(messages, list) or not all(
                isinstance(m, dict) and "role" in m and "content" in m for m in messages
            ):
                raise ReplayError("overrides.messages must be a list of {role, content} dicts")
            messages = [dict(m) for m in messages]
        return cls(
            model=values.get("model"),
            messages=messages,
            temperature=values.get("temperature"),
            top_p=values.get("top_p"),
            max_tokens=values.get("max_tokens"),
            frequency_penalty=values.get("frequency_penalty"),
            presence_penalty=values.get("presence_penalty"),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Only the explicitly set fields (deltas), never defaults."""
        return {k: v for k, v in asdict(self).items() if v is not None}

    def is_empty(self) -> bool:
        """True when no override is set (replay re-runs as a baseline probe)."""
        return not self.to_dict()


@dataclass(frozen=True)
class LLMRequest:
    """Reconstructed provider invocation from a captured LLM span."""

    provider: str
    model: str
    messages: List[Dict[str, Any]] = field(default_factory=list)
    tools: List[Dict[str, Any]] = field(default_factory=list)
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None


@dataclass(frozen=True)
class ToolCallData:
    """One function call requested by the model. Recorded, never executed."""

    name: str
    arguments: Any
    id: str = ""


@dataclass(frozen=True)
class LLMResponse:
    """Provider-agnostic completion used to stamp the replay span."""

    content: str = ""
    tool_calls: List[ToolCallData] = field(default_factory=list)
    model: str = ""
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    latency_ms: Optional[float] = None
    finish_reason: Optional[str] = None


@dataclass(frozen=True)
class ReplayedSpan:
    """Lightweight snapshot of one emitted replay span.

    Mirrors the OTel span's identity + attributes so comparison works without
    draining an exporter. ``parent_span_id`` is the replay-trace parent.
    """

    name: str
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    attributes: Dict[str, Any]


@dataclass
class ReplayResult:
    """Outcome of replaying one trace. Original spans are never mutated.

    ``spans`` holds one :class:`ReplayedSpan` snapshot per replayed LLM span
    (the grouping ``counterfactual-replay`` root exists in the backend trace
    but is excluded here so comparison focuses on replayed content).
    """

    original_trace_id: str
    replay_trace_id: str
    replay_id: str
    overrides: ReplayOverrides
    spans: List[ReplayedSpan] = field(default_factory=list)
    skipped: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe outcome summary (span snapshots live on `.spans`)."""
        return {
            "original_trace_id": self.original_trace_id,
            "replay_trace_id": self.replay_trace_id,
            "replay_id": self.replay_id,
            "overrides": self.overrides.to_dict(),
            "replayed_spans": len(self.spans),
            "skipped": self.skipped,
        }


@dataclass
class SpanFieldDiff:
    """Per-span field changes between one matched original/replay pair."""

    span_name: str
    span_kind: str
    original_span_id: str
    replay_span_id: str
    changed_fields: List[str] = field(default_factory=list)
    original: Dict[str, Any] = field(default_factory=dict)
    replay: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TraceComparison:
    """Structured original-vs-replay diff for human debugging and eval handoff."""

    original_trace_id: str
    replay_trace_id: str
    overrides: ReplayOverrides
    changed_spans: List[SpanFieldDiff] = field(default_factory=list)
    added_spans: List[str] = field(default_factory=list)
    removed_spans: List[str] = field(default_factory=list)
    output_changes: List[Dict[str, Any]] = field(default_factory=list)
    tool_call_changes: List[Dict[str, Any]] = field(default_factory=list)
    latency_delta_ms: Optional[float] = None
    prompt_tokens_delta: Optional[int] = None
    completion_tokens_delta: Optional[int] = None
    total_tokens_delta: Optional[int] = None
    cost_delta_usd: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe comparison for UI rendering or eval-system ingestion."""
        return {
            "original_trace_id": self.original_trace_id,
            "replay_trace_id": self.replay_trace_id,
            "overrides": self.overrides.to_dict(),
            "changed_spans": [asdict(d) for d in self.changed_spans],
            "added_spans": self.added_spans,
            "removed_spans": self.removed_spans,
            "output_changes": self.output_changes,
            "tool_call_changes": self.tool_call_changes,
            "latency_delta_ms": self.latency_delta_ms,
            "prompt_tokens_delta": self.prompt_tokens_delta,
            "completion_tokens_delta": self.completion_tokens_delta,
            "total_tokens_delta": self.total_tokens_delta,
            "cost_delta_usd": self.cost_delta_usd,
        }

    def summary(self) -> str:
        """One line per changed span: name, kind, and changed field names."""
        lines = [
            f"original={self.original_trace_id} replay={self.replay_trace_id} "
            f"changed={len(self.changed_spans)} "
            f"added={len(self.added_spans)} removed={len(self.removed_spans)}"
        ]
        for diff in self.changed_spans:
            lines.append(
                f"- {diff.span_name} [{diff.span_kind}]: " + ", ".join(diff.changed_fields)
            )
        return "\n".join(lines)

    def to_eval_case(self) -> Dict[str, Any]:
        """Build a regression/eval handoff payload from this comparison.

        This is intentionally a plain JSON-safe dict, not an eval runner:
        it captures the failing input, the rejected (original) output, and
        the preferred (replay) output with full provenance, ready to feed
        the existing Neatlogs experiments system. Raises :class:`ReplayError`
        when the comparison contains no output change to regress on.
        """
        if not self.output_changes:
            raise ReplayError("Cannot build an eval case: no output change in comparison")
        first = self.output_changes[0]
        return {
            "input_text": first.get("original_input"),
            "rejected_output": first["original_output"],
            "preferred_output": first["replay_output"],
            "overrides": self.overrides.to_dict(),
            "tool_call_changes": self.tool_call_changes,
            "provenance": {
                "original_trace_id": self.original_trace_id,
                "replay_trace_id": self.replay_trace_id,
                "original_span_id": first["original_span_id"],
                "replay_span_id": first["replay_span_id"],
                "span": first["span"],
            },
        }
