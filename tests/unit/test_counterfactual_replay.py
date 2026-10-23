"""Counterfactual replay MVP: replay + comparison behavior."""

import json

import pytest
from opentelemetry import trace as otel_trace

import neatlogs
from neatlogs._wrap_utils import set_neatlogs_provider
from neatlogs.replay import (
    InMemoryTraceStore,
    LLMResponse,
    ProviderError,
    ReplayError,
    ReplayOverrides,
    ToolCallData,
    TraceNotFoundError,
    UnsupportedSpanError,
    compare_traces,
    replay_trace,
)


def _install(tracer_provider):
    otel_trace.set_tracer_provider(tracer_provider)
    set_neatlogs_provider(tracer_provider)


def _make_llm_span(
    tracer_provider,
    *,
    name="openai.chat.completions.create",
    model="gpt-4o-mini",
    user_content="Where is my order?",
    output="I'll issue a refund",
    tool_calls=None,
    provider="openai",
    prompt_tokens=20,
    completion_tokens=10,
    duration_ms=120.0,
    frequency_penalty=None,
    presence_penalty=None,
    parent_context=None,
):
    tracer = tracer_provider.get_tracer("test-original")
    kwargs = {"context": parent_context} if parent_context is not None else {}
    with tracer.start_as_current_span(name, **kwargs) as span:
        span.set_attribute("neatlogs.span.kind", "llm")
        span.set_attribute("openinference.span.kind", "LLM")
        span.set_attribute("neatlogs.llm.provider", provider)
        span.set_attribute("neatlogs.llm.system", provider)
        span.set_attribute("neatlogs.llm.model_name", model)
        span.set_attribute("neatlogs.llm.input_messages.0.role", "user")
        span.set_attribute("neatlogs.llm.input_messages.0.content", user_content)
        span.set_attribute(
            "neatlogs.llm.tools.0.definition",
            json.dumps({"type": "function", "function": {"name": "refund_order"}}),
        )
        span.set_attribute("neatlogs.llm.tools.0.name", "refund_order")
        span.set_attribute("neatlogs.llm.output_messages.0.role", "assistant")
        span.set_attribute("neatlogs.llm.output_messages.0.content", output)
        for i, call in enumerate(tool_calls or []):
            span.set_attribute(f"neatlogs.llm.tool_calls.{i}.name", call["name"])
            span.set_attribute(f"neatlogs.llm.tool_calls.{i}.arguments", call["arguments"])
            span.set_attribute(f"neatlogs.llm.tool_calls.{i}.id", call.get("id", f"call-{i}"))
        span.set_attribute("neatlogs.llm.token_count.prompt", prompt_tokens)
        span.set_attribute("neatlogs.llm.token_count.completion", completion_tokens)
        span.set_attribute("neatlogs.llm.token_count.total", prompt_tokens + completion_tokens)
        span.set_attribute("neatlogs.llm.metrics.duration_ms", duration_ms)
        if frequency_penalty is not None:
            span.set_attribute("neatlogs.llm.frequency_penalty", frequency_penalty)
        if presence_penalty is not None:
            span.set_attribute("neatlogs.llm.presence_penalty", presence_penalty)


def _make_tool_span(tracer_provider, name="refund_order", parent_context=None):
    tracer = tracer_provider.get_tracer("test-original")
    kwargs = {"context": parent_context} if parent_context is not None else {}
    with tracer.start_as_current_span(name, **kwargs) as span:
        span.set_attribute("neatlogs.span.kind", "tool")
        span.set_attribute("tool.name", name)
        span.set_attribute("input.value", '{"order_id": 123}')
        span.set_attribute("output.value", '{"refunded": true}')


def _trace_id(span):
    return f"{span.get_span_context().trace_id:032x}"


def _span_id(span):
    return f"{span.get_span_context().span_id:016x}"


def _stub_caller(response=None, seen=None):
    def _call(request):
        if seen is not None:
            seen.append(request)
        return response or LLMResponse(content="I'll cancel the order")

    return _call


def _llm_spans(exporter):
    return [
        s for s in exporter.get_finished_spans() if s.attributes.get("neatlogs.span.kind") == "llm"
    ]


def test_top_level_exports():
    assert callable(neatlogs.replay_trace)
    assert callable(neatlogs.compare_traces)


def test_successful_replay(tracer_provider, in_memory_span_exporter):
    _install(tracer_provider)
    _make_llm_span(tracer_provider)
    original = _llm_spans(in_memory_span_exporter)
    assert len(original) == 1

    store = InMemoryTraceStore()
    trace_id = store.put_spans(original)
    result = replay_trace(
        trace_id,
        ReplayOverrides(),
        store=store,
        llm_caller=_stub_caller(),
        tracer=tracer_provider.get_tracer("test-replay"),
    )
    assert result.original_trace_id == trace_id
    assert result.replay_trace_id != trace_id
    assert len(result.spans) == 1
    assert result.replay_id
    assert result.spans[0].attributes["neatlogs.replay.of_trace_id"] == trace_id


def test_replay_with_model_override(tracer_provider, in_memory_span_exporter):
    _install(tracer_provider)
    _make_llm_span(tracer_provider, model="gpt-4o")
    original = _llm_spans(in_memory_span_exporter)
    seen = []
    result = replay_trace(
        original,
        ReplayOverrides(model="gpt-4o-mini", temperature=0.0),
        llm_caller=_stub_caller(LLMResponse(content="ok", model="gpt-4o-mini"), seen),
        tracer=tracer_provider.get_tracer("test-replay"),
    )
    assert seen[0].model == "gpt-4o-mini"
    assert seen[0].temperature == 0.0
    # Non-overridden context is reused.
    assert seen[0].messages[0]["content"] == "Where is my order?"
    assert result.spans[0].attributes["neatlogs.llm.model_name"] == "gpt-4o-mini"


def test_replay_with_messages_override(tracer_provider, in_memory_span_exporter):
    _install(tracer_provider)
    _make_llm_span(tracer_provider)
    original = _llm_spans(in_memory_span_exporter)
    seen = []
    messages = [
        {"role": "system", "content": "Never issue refunds."},
        {"role": "user", "content": "Where is my order?"},
    ]
    replay_trace(
        original,
        {"messages": messages},
        llm_caller=_stub_caller(seen=seen),
        tracer=tracer_provider.get_tracer("test-replay"),
    )
    assert seen[0].messages == messages


def test_original_trace_unchanged(tracer_provider, in_memory_span_exporter):
    _install(tracer_provider)
    _make_llm_span(tracer_provider)
    original = _llm_spans(in_memory_span_exporter)
    before = dict(original[0].attributes)
    replay_trace(
        original,
        ReplayOverrides(model="other"),
        llm_caller=_stub_caller(),
        tracer=tracer_provider.get_tracer("test-replay"),
    )
    assert dict(original[0].attributes) == before


def test_replay_references_original(tracer_provider, in_memory_span_exporter):
    _install(tracer_provider)
    _make_llm_span(tracer_provider)
    original = _llm_spans(in_memory_span_exporter)
    trace_id = _trace_id(original[0])
    result = replay_trace(
        original,
        ReplayOverrides(model="gpt-4o-mini"),
        llm_caller=_stub_caller(),
        tracer=tracer_provider.get_tracer("test-replay"),
    )
    attrs = result.spans[0].attributes
    assert attrs["neatlogs.replay.of_trace_id"] == trace_id
    assert attrs["neatlogs.replay.of_span_id"] == _span_id(original[0])
    assert attrs["neatlogs.replay.id"] == result.replay_id
    assert json.loads(attrs["neatlogs.replay.overrides"]) == {"model": "gpt-4o-mini"}


def test_comparison_detects_changed_output(tracer_provider, in_memory_span_exporter):
    _install(tracer_provider)
    _make_llm_span(tracer_provider, output="I'll issue a refund")
    original = _llm_spans(in_memory_span_exporter)
    result = replay_trace(
        original,
        ReplayOverrides(),
        llm_caller=_stub_caller(LLMResponse(content="I'll cancel the order")),
        tracer=tracer_provider.get_tracer("test-replay"),
    )
    comparison = compare_traces(original, result.spans, result.overrides)
    assert comparison.original_trace_id == _trace_id(original[0])
    assert comparison.replay_trace_id == result.spans[0].trace_id
    assert len(comparison.changed_spans) == 1
    assert "output" in comparison.changed_spans[0].changed_fields
    assert len(comparison.output_changes) == 1
    assert comparison.output_changes[0]["original_output"] == "I'll issue a refund"
    assert comparison.output_changes[0]["replay_output"] == "I'll cancel the order"


def test_comparison_detects_changed_tool_call(tracer_provider, in_memory_span_exporter):
    _install(tracer_provider)
    _make_llm_span(
        tracer_provider,
        tool_calls=[{"name": "refund_order", "arguments": '{"order_id": 123}'}],
    )
    original = _llm_spans(in_memory_span_exporter)
    result = replay_trace(
        original,
        ReplayOverrides(),
        llm_caller=_stub_caller(
            LLMResponse(
                content="Cancelling",
                tool_calls=[
                    ToolCallData(name="cancel_order", arguments='{"order_id": 123}', id="call-9")
                ],
            )
        ),
        tracer=tracer_provider.get_tracer("test-replay"),
    )
    comparison = compare_traces(original, result.spans)
    assert len(comparison.tool_call_changes) == 1
    change = comparison.tool_call_changes[0]
    assert change["original_tool_calls"][0]["name"] == "refund_order"
    assert change["replay_tool_calls"][0]["name"] == "cancel_order"


def test_comparison_latency_and_token_deltas(tracer_provider, in_memory_span_exporter):
    _install(tracer_provider)
    _make_llm_span(tracer_provider, prompt_tokens=20, completion_tokens=10, duration_ms=100.0)
    original = _llm_spans(in_memory_span_exporter)
    result = replay_trace(
        original,
        ReplayOverrides(),
        llm_caller=_stub_caller(
            LLMResponse(
                content="new",
                prompt_tokens=30,
                completion_tokens=5,
                total_tokens=35,
                latency_ms=150.0,
            )
        ),
        tracer=tracer_provider.get_tracer("test-replay"),
    )
    comparison = compare_traces(original, result.spans)
    assert comparison.latency_delta_ms == pytest.approx(50.0)
    assert comparison.prompt_tokens_delta == 10
    assert comparison.completion_tokens_delta == -5
    assert comparison.total_tokens_delta == 5


def test_missing_trace_raises(tracer_provider):
    _install(tracer_provider)
    store = InMemoryTraceStore()
    with pytest.raises(TraceNotFoundError):
        replay_trace(
            "0" * 32, ReplayOverrides(), store=store, tracer=tracer_provider.get_tracer("t")
        )


def test_replay_by_trace_id_requires_store(tracer_provider):
    _install(tracer_provider)
    with pytest.raises(ReplayError):
        replay_trace("0" * 32, ReplayOverrides(), tracer=tracer_provider.get_tracer("t"))


def test_empty_span_list_raises(tracer_provider):
    _install(tracer_provider)
    with pytest.raises(ReplayError):
        replay_trace([], ReplayOverrides(), tracer=tracer_provider.get_tracer("t"))


def test_malformed_llm_span_raises(tracer_provider, in_memory_span_exporter):
    _install(tracer_provider)
    tracer = tracer_provider.get_tracer("test-original")
    with tracer.start_as_current_span("broken"):
        pass
    # A span with no kind/input at all is skipped as non-LLM; force the LLM path
    # with a kind but no model/messages.
    with tracer.start_as_current_span("broken-llm") as span:
        span.set_attribute("neatlogs.span.kind", "llm")
        span.set_attribute("neatlogs.llm.provider", "openai")
    spans = in_memory_span_exporter.get_finished_spans()
    llm_only = [s for s in spans if s.name == "broken-llm"]
    with pytest.raises(ReplayError):
        replay_trace(
            llm_only,
            ReplayOverrides(),
            llm_caller=_stub_caller(),
            tracer=tracer_provider.get_tracer("t"),
        )


def test_unsupported_span_type_skipped_or_raises(tracer_provider, in_memory_span_exporter):
    _install(tracer_provider)
    _make_tool_span(tracer_provider)
    tool_spans = [s for s in in_memory_span_exporter.get_finished_spans()]
    with pytest.raises(UnsupportedSpanError):
        replay_trace(
            tool_spans,
            ReplayOverrides(),
            llm_caller=_stub_caller(),
            tracer=tracer_provider.get_tracer("t"),
        )

    # A mixed single trace replays the LLM span and records the rest as skipped.
    tracer = tracer_provider.get_tracer("test-original")
    with tracer.start_as_current_span("workflow") as parent:
        parent.set_attribute("neatlogs.span.kind", "workflow")
        child_context = otel_trace.set_span_in_context(parent)
        _make_tool_span(tracer_provider, parent_context=child_context)
        _make_llm_span(tracer_provider, parent_context=child_context)
    mixed_trace_id = _trace_id(
        next(
            s
            for s in in_memory_span_exporter.get_finished_spans()
            if s.name == "openai.chat.completions.create"
        )
    )
    all_spans = [
        s for s in in_memory_span_exporter.get_finished_spans() if _trace_id(s) == mixed_trace_id
    ]
    assert len(all_spans) == 3
    result = replay_trace(
        all_spans,
        ReplayOverrides(),
        llm_caller=_stub_caller(),
        tracer=tracer_provider.get_tracer("t"),
    )
    assert len(result.spans) == 1
    assert {s["kind"] for s in result.skipped} == {"WORKFLOW", "TOOL"}


def test_spans_from_multiple_traces_rejected(tracer_provider, in_memory_span_exporter):
    _install(tracer_provider)
    _make_tool_span(tracer_provider)
    _make_llm_span(tracer_provider)
    all_spans = in_memory_span_exporter.get_finished_spans()
    assert len({_trace_id(s) for s in all_spans}) == 2
    with pytest.raises(ReplayError):
        replay_trace(
            all_spans,
            ReplayOverrides(),
            llm_caller=_stub_caller(),
            tracer=tracer_provider.get_tracer("t"),
        )
    store = InMemoryTraceStore()
    with pytest.raises(ValueError):
        store.put_spans(all_spans)


def test_unsupported_provider(tracer_provider, in_memory_span_exporter):
    _install(tracer_provider)
    _make_llm_span(tracer_provider, provider="some-future-provider")
    original = in_memory_span_exporter.get_finished_spans()
    with pytest.raises(UnsupportedSpanError):
        replay_trace(
            original,
            ReplayOverrides(),
            llm_caller=_stub_caller(),
            tracer=tracer_provider.get_tracer("t"),
        )


def test_provider_failure_wrapped(tracer_provider, in_memory_span_exporter):
    _install(tracer_provider)
    _make_llm_span(tracer_provider)
    original = _llm_spans(in_memory_span_exporter)

    def _boom(request):
        raise RuntimeError("downstream exploded")

    with pytest.raises(ProviderError):
        replay_trace(
            original, ReplayOverrides(), llm_caller=_boom, tracer=tracer_provider.get_tracer("t")
        )

    def _provider_boom(request):
        raise ProviderError("api 500")

    with pytest.raises(ProviderError):
        replay_trace(
            original,
            ReplayOverrides(),
            llm_caller=_provider_boom,
            tracer=tracer_provider.get_tracer("t"),
        )


def test_allow_tool_execution_refused(tracer_provider, in_memory_span_exporter):
    _install(tracer_provider)
    _make_llm_span(tracer_provider)
    original = _llm_spans(in_memory_span_exporter)
    called = []

    def _caller(request):
        called.append(request)
        return LLMResponse(content="x")

    with pytest.raises(ReplayError):
        replay_trace(
            original,
            ReplayOverrides(),
            llm_caller=_caller,
            tracer=tracer_provider.get_tracer("t"),
            allow_tool_execution=True,
        )
    assert called == []


def test_unknown_override_keys_rejected():
    with pytest.raises(ReplayError):
        ReplayOverrides.from_dict({"model": "x", "bogus": 1})
    with pytest.raises(ReplayError):
        ReplayOverrides.from_dict({"messages": [{"role": "user"}]})


def test_provider_preserved_in_request(tracer_provider, in_memory_span_exporter):
    _install(tracer_provider)
    _make_llm_span(tracer_provider, provider="azure_openai")
    original = _llm_spans(in_memory_span_exporter)
    seen = []
    result = replay_trace(
        original,
        ReplayOverrides(),
        llm_caller=_stub_caller(seen=seen),
        tracer=tracer_provider.get_tracer("t"),
    )
    assert seen[0].provider == "azure_openai"
    assert len(result.spans) == 1


def test_missing_provider_refused(tracer_provider, in_memory_span_exporter):
    _install(tracer_provider)
    tracer = tracer_provider.get_tracer("test-original")
    with tracer.start_as_current_span("providerless") as span:
        span.set_attribute("neatlogs.span.kind", "llm")
        span.set_attribute("neatlogs.llm.model_name", "gpt-4o-mini")
        span.set_attribute("neatlogs.llm.input_messages.0.role", "user")
        span.set_attribute("neatlogs.llm.input_messages.0.content", "hi")
    spans = [s for s in in_memory_span_exporter.get_finished_spans() if s.name == "providerless"]
    with pytest.raises(UnsupportedSpanError):
        replay_trace(
            spans,
            ReplayOverrides(),
            llm_caller=_stub_caller(),
            tracer=tracer_provider.get_tracer("t"),
        )


def test_frequency_presence_penalties_round_trip(tracer_provider, in_memory_span_exporter):
    _install(tracer_provider)
    _make_llm_span(tracer_provider, frequency_penalty=0.5, presence_penalty=0.2)
    original = _llm_spans(in_memory_span_exporter)
    seen = []
    result = replay_trace(
        original,
        ReplayOverrides(frequency_penalty=0.9),
        llm_caller=_stub_caller(seen=seen),
        tracer=tracer_provider.get_tracer("t"),
    )
    # Original values reused unless overridden.
    assert seen[0].frequency_penalty == 0.9
    assert seen[0].presence_penalty == 0.2
    attrs = result.spans[0].attributes
    assert attrs["neatlogs.llm.frequency_penalty"] == 0.9
    assert attrs["neatlogs.llm.presence_penalty"] == 0.2


def test_to_eval_case(tracer_provider, in_memory_span_exporter):
    _install(tracer_provider)
    _make_llm_span(tracer_provider, output="I'll issue a refund")
    original = _llm_spans(in_memory_span_exporter)
    result = replay_trace(
        original,
        ReplayOverrides(),
        llm_caller=_stub_caller(LLMResponse(content="I'll cancel the order")),
        tracer=tracer_provider.get_tracer("t"),
    )
    comparison = compare_traces(original, result.spans, result.overrides)
    case = comparison.to_eval_case()
    assert case["rejected_output"] == "I'll issue a refund"
    assert case["preferred_output"] == "I'll cancel the order"
    assert "Where is my order?" in case["input_text"]
    assert case["provenance"]["original_trace_id"] == comparison.original_trace_id
    assert case["provenance"]["replay_trace_id"] == comparison.replay_trace_id
    # JSON-safe handoff payload.
    json.dumps(case)


def test_to_eval_case_without_output_change_raises(tracer_provider, in_memory_span_exporter):
    _install(tracer_provider)
    _make_llm_span(tracer_provider, output="same answer")
    original = _llm_spans(in_memory_span_exporter)
    result = replay_trace(
        original,
        ReplayOverrides(),
        llm_caller=_stub_caller(LLMResponse(content="same answer")),
        tracer=tracer_provider.get_tracer("t"),
    )
    comparison = compare_traces(original, result.spans, result.overrides)
    assert comparison.output_changes == []
    with pytest.raises(ReplayError):
        comparison.to_eval_case()
