"""Canonical telemetry v2 keeps wrapper-emitted LLM invocation parameters.

Wrappers (bedrock, langchain, hermes, ...) record invocation parameters as
flat attributes such as ``neatlogs.llm.temperature``. Canonical normalization
only read the ``neatlogs.llm.invocation_parameters.*`` namespace, so those
values surfaced as null. Canonical keys stay authoritative; the flat wrapper
keys are fallbacks.
"""

from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.trace import SpanContext, SpanKind, TraceFlags, TraceState
from opentelemetry.trace.status import Status, StatusCode

from neatlogs.core.telemetry_v2 import normalize_span_v2


def _llm_span(attributes: dict) -> ReadableSpan:
    context = SpanContext(
        trace_id=0x1234567890ABCDEF1234567890ABCDEF,
        span_id=0x1234567890ABCDEF,
        is_remote=False,
        trace_flags=TraceFlags.SAMPLED,
        trace_state=TraceState(),
    )
    base = {"neatlogs.span.kind": "llm"}
    base.update(attributes)
    return ReadableSpan(
        name="llm",
        context=context,
        parent=None,
        resource=Resource.create({}),
        attributes=base,
        events=[],
        links=[],
        kind=SpanKind.INTERNAL,
        status=Status(StatusCode.OK),
        start_time=1,
        end_time=2,
        instrumentation_scope=None,
    )


def _parameters(span: ReadableSpan) -> dict:
    return normalize_span_v2(span).to_dict()["semantic"]["request"]["parameters"]


def test_flat_wrapper_invocation_parameters_are_preserved() -> None:
    span = _llm_span(
        {
            "neatlogs.llm.temperature": 0.2,
            "neatlogs.llm.top_p": 0.8,
            "neatlogs.llm.top_k": 40,
            "neatlogs.llm.max_tokens": 321,
            "neatlogs.llm.frequency_penalty": 0.1,
            "neatlogs.llm.presence_penalty": 0.3,
        }
    )

    parameters = _parameters(span)

    assert parameters["temperature"] == 0.2
    assert parameters["top_p"] == 0.8
    assert parameters["top_k"] == 40.0
    assert parameters["max_output_tokens"] == 321
    assert parameters["frequency_penalty"] == 0.1
    assert parameters["presence_penalty"] == 0.3


def test_canonical_invocation_parameters_take_precedence_over_flat_keys() -> None:
    span = _llm_span(
        {
            "neatlogs.llm.invocation_parameters.temperature": 0.9,
            "neatlogs.llm.temperature": 0.2,
            "neatlogs.llm.invocation_parameters.max_output_tokens": 128,
            "neatlogs.llm.max_tokens": 321,
        }
    )

    parameters = _parameters(span)

    assert parameters["temperature"] == 0.9
    assert parameters["max_output_tokens"] == 128


def test_missing_invocation_parameters_stay_null() -> None:
    parameters = _parameters(_llm_span({}))

    assert parameters["temperature"] is None
    assert parameters["top_p"] is None
    assert parameters["max_output_tokens"] is None
    assert parameters["frequency_penalty"] is None
    assert parameters["presence_penalty"] is None
