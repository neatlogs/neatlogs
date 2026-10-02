"""Re-execute captured LLM spans with explicit overrides.

MVP semantics (see ``docs/counterfactual-replay.md``):

REPLAYED (reused): original messages, tool schemas, model/params unless
  overridden, retrieval context stays as captured data (never re-fetched).
OVERRIDDEN: only fields set on :class:`ReplayOverrides`.
NON-DETERMINISTIC (new): output, tool calls, tokens, latency, timestamps.
NEVER executed: tools, external APIs, workflow functions. ``replay_trace``
  emits new spans only; the original spans are only read.
"""

from __future__ import annotations

import json
import uuid
from typing import Any, Callable, Dict, List, Mapping, Optional, Union

from .extract import extract_llm_request, span_kind
from .openai_caller import openai_caller
from .types import (
    LLMRequest,
    LLMResponse,
    ProviderError,
    ReplayError,
    ReplayOverrides,
    ReplayResult,
    UnsupportedSpanError,
    span_id,
    trace_id,
)

LlmCaller = Callable[[LLMRequest], LLMResponse]

REPLAY_ID_ATTR = "neatlogs.replay.id"
REPLAY_OF_TRACE_ATTR = "neatlogs.replay.of_trace_id"
REPLAY_OF_SPAN_ATTR = "neatlogs.replay.of_span_id"
REPLAY_OVERRIDES_ATTR = "neatlogs.replay.overrides"


def _dumps(value: Any) -> str:
    return json.dumps(value, default=str, ensure_ascii=False)


def _resolve_spans(
    trace_or_spans: Union[str, List[Any]], store: Any = None
) -> tuple[str, List[Any]]:
    if isinstance(trace_or_spans, str):
        if store is None:
            raise ReplayError("A TraceStore is required to replay by trace_id")
        spans = store.get_trace(trace_or_spans)
        return trace_or_spans, spans
    spans = list(trace_or_spans)
    if not spans:
        raise ReplayError("Cannot replay an empty span list")
    trace_ids = {trace_id(span) for span in spans}
    if len(trace_ids) > 1:
        raise ReplayError(
            "Refusing to replay spans from multiple traces in one call; "
            f"got {len(trace_ids)} trace ids. Group spans per trace first."
        )
    return trace_id(spans[0]), spans


def _resolve_overrides(
    overrides: Union[ReplayOverrides, Mapping[str, Any], None],
) -> ReplayOverrides:
    if overrides is None:
        return ReplayOverrides()
    if isinstance(overrides, ReplayOverrides):
        return overrides
    if isinstance(overrides, Mapping):
        return ReplayOverrides.from_dict(overrides)
    raise ReplayError("overrides must be ReplayOverrides, a dict, or None")


def _apply_overrides(request: LLMRequest, overrides: ReplayOverrides) -> LLMRequest:
    return LLMRequest(
        provider=request.provider,
        model=overrides.model or request.model,
        messages=list(overrides.messages) if overrides.messages is not None else request.messages,
        tools=request.tools,
        temperature=request.temperature if overrides.temperature is None else overrides.temperature,
        top_p=request.top_p if overrides.top_p is None else overrides.top_p,
        max_tokens=request.max_tokens if overrides.max_tokens is None else overrides.max_tokens,
        frequency_penalty=(
            request.frequency_penalty
            if overrides.frequency_penalty is None
            else overrides.frequency_penalty
        ),
        presence_penalty=(
            request.presence_penalty
            if overrides.presence_penalty is None
            else overrides.presence_penalty
        ),
    )


def _emit_llm_span(
    tracer: Any,
    name: str,
    request: LLMRequest,
    response: LLMResponse,
    *,
    original_span: Any,
    original_trace_id: str,
    replay_id: str,
    overrides_json: str,
) -> "ReplayedSpan":
    from .types import ReplayedSpan

    attributes: Dict[str, Any] = {
        "neatlogs.span.kind": "llm",
        "openinference.span.kind": "LLM",
        "neatlogs.llm.provider": "openai",
        "neatlogs.llm.system": "openai",
        "neatlogs.llm.model_name": request.model,
        REPLAY_OF_TRACE_ATTR: original_trace_id,
        REPLAY_OF_SPAN_ATTR: span_id(original_span),
        REPLAY_ID_ATTR: replay_id,
        REPLAY_OVERRIDES_ATTR: overrides_json,
        "neatlogs.replay.mode": "llm-span",
    }
    for i, message in enumerate(request.messages):
        attributes[f"neatlogs.llm.input_messages.{i}.role"] = str(message.get("role", ""))
        attributes[f"neatlogs.llm.input_messages.{i}.content"] = str(message.get("content", ""))
    for i, tool in enumerate(request.tools):
        attributes[f"neatlogs.llm.tools.{i}.type"] = str(tool.get("type") or "function")
        attributes[f"neatlogs.llm.tools.{i}.name"] = str(tool.get("name") or "")
        attributes[f"neatlogs.llm.tools.{i}.definition"] = _dumps(tool.get("definition"))
    if request.temperature is not None:
        attributes["neatlogs.llm.temperature"] = request.temperature
    if request.top_p is not None:
        attributes["neatlogs.llm.top_p"] = request.top_p
    if request.max_tokens is not None:
        attributes["neatlogs.llm.max_tokens"] = request.max_tokens
    if request.frequency_penalty is not None:
        attributes["neatlogs.llm.frequency_penalty"] = request.frequency_penalty
    if request.presence_penalty is not None:
        attributes["neatlogs.llm.presence_penalty"] = request.presence_penalty

    with tracer.start_as_current_span(name, attributes=attributes) as span:
        span.set_attribute("neatlogs.llm.output_messages.0.role", "assistant")
        if response.content:
            span.set_attribute("neatlogs.llm.output_messages.0.content", response.content)
        for i, call in enumerate(response.tool_calls):
            span.set_attribute(f"neatlogs.llm.tool_calls.{i}.id", call.id or f"replay_{i}")
            span.set_attribute(f"neatlogs.llm.tool_calls.{i}.name", call.name)
            span.set_attribute(
                f"neatlogs.llm.tool_calls.{i}.arguments",
                call.arguments if isinstance(call.arguments, str) else _dumps(call.arguments),
            )
            span.set_attribute(f"neatlogs.llm.tool_calls.{i}.choice_index", 0)
            span.set_attribute(f"neatlogs.llm.tool_calls.{i}.tool_call_index", i)
        if response.model:
            span.set_attribute("neatlogs.llm.model_name", response.model)
        if response.finish_reason:
            span.set_attribute("neatlogs.llm.finish_reason", response.finish_reason)
            span.set_attribute("neatlogs.llm.choices.0.finish_reason", response.finish_reason)
        if response.prompt_tokens is not None:
            span.set_attribute("neatlogs.llm.token_count.prompt", response.prompt_tokens)
        if response.completion_tokens is not None:
            span.set_attribute("neatlogs.llm.token_count.completion", response.completion_tokens)
        if response.total_tokens is not None:
            span.set_attribute("neatlogs.llm.token_count.total", response.total_tokens)
        if response.latency_ms is not None:
            span.set_attribute("neatlogs.llm.metrics.duration_ms", response.latency_ms)
        snapshot_attributes = dict(getattr(span, "attributes", {}) or {})
        context = span.get_span_context()
        parent = getattr(span, "parent", None)
        return ReplayedSpan(
            name=name,
            trace_id=f"{context.trace_id:032x}",
            span_id=f"{context.span_id:016x}",
            parent_span_id=f"{parent.span_id:016x}" if parent is not None else None,
            attributes=snapshot_attributes,
        )


def replay_trace(
    trace_or_spans: Union[str, List[Any]],
    overrides: Union[ReplayOverrides, Mapping[str, Any], None] = None,
    *,
    store: Any = None,
    llm_caller: Optional[LlmCaller] = None,
    tracer: Any = None,
    allow_tool_execution: bool = False,
) -> ReplayResult:
    """Replay supported LLM spans with explicit overrides.

    Each replayable LLM span is re-issued **independently from its own
    captured messages** (fan-out probes, not chained re-execution):
    downstream spans are never re-run with replayed outputs, so for
    multi-step ``LLM → tool → LLM`` traces every LLM span sees original
    context. The replay context for one span is exactly what that span
    captured (messages incl. prior tool results in message history, tool
    schemas, model/params unless overridden); sibling retriever/tool spans
    are not merged in.

    Args:
        trace_or_spans: a trace_id (resolved via ``store``) or a span list.
            A span list must belong to a single trace.
        overrides: :class:`ReplayOverrides` or equivalent dict. Only set
            fields are applied; everything else is reused from the original.
        store: required when replaying by trace_id.
        llm_caller: ``(LLMRequest) -> LLMResponse``. Defaults to a live
            OpenAI call (one provider call per replayed LLM span, billed to
            the caller's own credentials); tests inject a stub so no network
            is needed.
        tracer: OTel tracer for the new spans. Defaults to Neatlogs'
            provider tracer (requires ``neatlogs.init()``) so replay flows
            through the normal pipeline.
        allow_tool_execution: accepted for future API stability but MUST be
            False in this MVP — tools are recorded, never executed.

    Returns:
        ReplayResult with a fresh ``replay_trace_id``. Original spans are
        never mutated. ``skipped`` lists ``{span_id, kind, reason}`` dicts
        for spans that are not LLM-replayable. The replay root carries no
        session/end-user identity:
        replays stay out of production session analytics and link back via
        ``neatlogs.replay.of_trace_id``.

    Privacy: replay re-sends captured inputs to the provider through the
    same masking pipeline — redact production traces (``mask=`` / server-side
    PII) before replaying them.
    """
    if allow_tool_execution:
        raise ReplayError("Tool execution is not supported in this MVP replay (safe mode)")
    resolved = _resolve_overrides(overrides)
    original_trace_id, spans = _resolve_spans(trace_or_spans, store)
    caller = llm_caller or openai_caller
    if tracer is None:
        from .._wrap_utils import get_provider_tracer

        tracer = get_provider_tracer()

    replay_id = uuid.uuid4().hex
    overrides_json = _dumps(resolved.to_dict())
    skipped: List[Dict[str, Any]] = []

    # Phase 1: resolve every span before emitting anything, so a trace with
    # no replayable LLM spans raises without leaving a stray empty trace.
    planned: List[Any] = []
    for span in spans:
        kind = span_kind(span)
        if kind != "LLM":
            skipped.append(
                {
                    "span_id": span_id(span),
                    "kind": kind or "UNKNOWN",
                    "reason": "only LLM spans are replayable in this MVP",
                }
            )
            continue
        try:
            request = extract_llm_request(span)
        except UnsupportedSpanError as exc:
            skipped.append({"span_id": span_id(span), "kind": kind, "reason": str(exc)})
            continue
        assert request is not None
        planned.append((span, request))
    if not planned:
        raise UnsupportedSpanError(
            "No replayable LLM spans in trace " f"{original_trace_id} ({len(skipped)} skipped)",
            kind="",
            provider="",
        )

    replayed: List[Any] = []
    root_attributes = {
        "neatlogs.span.kind": "workflow",
        "openinference.span.kind": "WORKFLOW",
        REPLAY_OF_TRACE_ATTR: original_trace_id,
        REPLAY_ID_ATTR: replay_id,
        REPLAY_OVERRIDES_ATTR: overrides_json,
        "neatlogs.replay.mode": "llm-span",
    }
    with tracer.start_as_current_span("counterfactual-replay", attributes=root_attributes) as root:
        replay_trace_id = f"{root.get_span_context().trace_id:032x}"
        for span, request in planned:
            effective = _apply_overrides(request, resolved)
            name = getattr(span, "name", None) or "openai.chat.completions.create"
            try:
                response = caller(effective)
            except ProviderError:
                raise
            except ReplayError:
                raise
            except Exception as exc:
                raise ProviderError(f"LLM caller failed during replay: {exc}") from exc
            snapshot = _emit_llm_span(
                tracer,
                str(name),
                effective,
                response,
                original_span=span,
                original_trace_id=original_trace_id,
                replay_id=replay_id,
                overrides_json=overrides_json,
            )
            replayed.append(snapshot)

    return ReplayResult(
        original_trace_id=original_trace_id,
        replay_trace_id=replay_trace_id,
        replay_id=replay_id,
        overrides=resolved,
        spans=replayed,
        skipped=skipped,
    )
