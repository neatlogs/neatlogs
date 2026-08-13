"""Tests for the trace-annotations API (neatlogs.annotate / neatlogs.add_event)."""

import json
import math
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from opentelemetry import trace as otel_trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


def _setup_tracer():
    """Force a fresh TracerProvider so neatlogs' cached tracer rebinds."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    otel_trace.set_tracer_provider(provider)
    import neatlogs._wrap_utils as _wu

    _wu._wrapper_tracer = None
    return exporter


def _recording_neatlogs_span():
    """Open a recording neatlogs scope span via the public API; return
    (span, exporter). The span is the current span and is recording."""
    exporter = _setup_tracer()
    tracer = otel_trace.get_tracer("neatlogs.test")
    span = tracer.start_span("test-span")
    # Make the span look like a neatlogs scope span by setting the
    # canonical attribute. _is_neatlogs_span() checks for this.
    span.set_attribute("neatlogs.span.kind", "test")
    return span, exporter


def test_annotate_sets_attribute_on_active_span():
    span, exporter = _recording_neatlogs_span()
    with otel_trace.use_span(span, end_on_exit=False):
        from neatlogs import annotate

        applied = annotate(needs_review=True, severity="high")
    assert applied is True
    span.end()
    finished = exporter.get_finished_spans()
    assert len(finished) == 1
    attrs = dict(finished[0].attributes or {})
    assert attrs.get("neatlogs.annotation.needs_review") is True
    assert attrs.get("neatlogs.annotation.severity") == "high"


def test_annotate_returns_false_when_no_active_span():
    _setup_tracer()
    # No active span — get_current_span() returns INVALID_SPAN
    from neatlogs import annotate

    applied = annotate(needs_review=True)
    assert applied is False


def test_annotate_returns_false_when_init_not_called(monkeypatch):
    # Force a "not initialized" state by clearing the cached tracer
    import neatlogs._wrap_utils as _wu

    _wu._wrapper_tracer = None
    # Don't init(); the resolver should return None.
    from neatlogs import annotate

    applied = annotate(needs_review=True)
    assert applied is False


def test_annotate_returns_false_for_foreign_span():
    """In default mode, if the current span is not a neatlogs scope span,
    annotate must refuse to set attributes (no co-tenant leak)."""
    exporter = _setup_tracer()
    tracer = otel_trace.get_tracer("not-neatlogs")
    foreign = tracer.start_span("foreign-span")
    with otel_trace.use_span(foreign, end_on_exit=False):
        from neatlogs import annotate

        applied = annotate(needs_review=True)
    assert applied is False
    foreign.end()
    # The foreign span must NOT carry the annotation.
    finished = exporter.get_finished_spans()
    assert len(finished) == 1
    attrs = dict(finished[0].attributes or {})
    assert "neatlogs.annotation.needs_review" not in attrs


def test_annotate_returns_false_for_ended_span():
    """If the active span has already ended, annotate must return False."""
    span, _exporter = _recording_neatlogs_span()
    with otel_trace.use_span(span, end_on_exit=False):
        from neatlogs import annotate

        span.end()
        applied = annotate(needs_review=True)
    assert applied is False


def test_annotate_rejects_dotted_keys():
    span, exporter = _recording_neatlogs_span()
    with otel_trace.use_span(span, end_on_exit=False):
        from neatlogs import annotate

        applied = annotate(**{"customer.address.city": "SF"})
    # The dotted key is rejected, so the call returns False.
    assert applied is False
    span.end()


def test_annotate_rejects_empty_string_keys():
    span, exporter = _recording_neatlogs_span()
    with otel_trace.use_span(span, end_on_exit=False):
        from neatlogs import annotate

        applied = annotate(**{"": "value"})
    assert applied is False
    span.end()


def test_annotate_rejects_keys_with_reserved_prefix():
    span, exporter = _recording_neatlogs_span()
    with otel_trace.use_span(span, end_on_exit=False):
        from neatlogs import annotate

        # Keys starting with "neatlogs." are rejected to prevent
        # double-namespacing.
        applied = annotate(**{"neatlogs.x": 1, "valid": 2})
    # Only the "valid" key was applied, so applied is True.
    assert applied is True
    span.end()
    finished = exporter.get_finished_spans()
    attrs = dict(finished[0].attributes or {})
    assert "neatlogs.annotation.valid" in attrs
    assert "neatlogs.annotation.neatlogs.x" not in attrs
    assert "neatlogs.annotation.neatlogs.annotation.x" not in attrs


def test_annotate_skips_none_values():
    span, exporter = _recording_neatlogs_span()
    with otel_trace.use_span(span, end_on_exit=False):
        from neatlogs import annotate

        applied = annotate(a=1, b=None, c="x")
    assert applied is True
    span.end()
    attrs = dict(exporter.get_finished_spans()[0].attributes or {})
    assert "neatlogs.annotation.a" in attrs
    assert "neatlogs.annotation.b" not in attrs
    assert "neatlogs.annotation.c" in attrs


def test_annotate_preserves_primitive_types():
    """Primitives (int/float/bool/str) pass through natively so backend
    numeric and boolean filters remain meaningful."""
    span, exporter = _recording_neatlogs_span()
    with otel_trace.use_span(span, end_on_exit=False):
        from neatlogs import annotate

        applied = annotate(int_val=5, float_val=1.5, bool_val=True, str_val="high")
    assert applied is True
    span.end()
    attrs = dict(exporter.get_finished_spans()[0].attributes or {})
    # int stays int, not "5"
    assert attrs["neatlogs.annotation.int_val"] == 5
    assert isinstance(attrs["neatlogs.annotation.int_val"], int)
    assert attrs["neatlogs.annotation.float_val"] == 1.5
    assert attrs["neatlogs.annotation.bool_val"] is True
    assert isinstance(attrs["neatlogs.annotation.bool_val"], bool)
    assert attrs["neatlogs.annotation.str_val"] == "high"


def test_annotate_coerces_complex_types_to_json():
    span, exporter = _recording_neatlogs_span()
    with otel_trace.use_span(span, end_on_exit=False):
        from neatlogs import annotate

        applied = annotate(
            payload={"key": "value", "n": 1},
            items=[1, 2, 3],
        )
    assert applied is True
    span.end()
    attrs = dict(exporter.get_finished_spans()[0].attributes or {})
    # Both dict and list were JSON-serialized.
    assert json.loads(attrs["neatlogs.annotation.payload"]) == {
        "key": "value",
        "n": 1,
    }
    assert json.loads(attrs["neatlogs.annotation.items"]) == [1, 2, 3]


def test_annotate_handles_nan_and_infinity():
    """NaN/Inf floats must not emit unquoted JSON tokens (ClickHouse rejects)."""
    span, exporter = _recording_neatlogs_span()
    with otel_trace.use_span(span, end_on_exit=False):
        from neatlogs import annotate

        applied = annotate(nan_v=math.nan, inf_v=math.inf, neg_inf_v=-math.inf)
    assert applied is True
    span.end()
    attrs = dict(exporter.get_finished_spans()[0].attributes or {})
    # NaN/Inf are coerced to their string form, not raw float.
    assert attrs["neatlogs.annotation.nan_v"] == "NaN"
    assert attrs["neatlogs.annotation.inf_v"] == "Infinity"
    assert attrs["neatlogs.annotation.neg_inf_v"] == "-Infinity"


def test_annotate_empty_kwargs_returns_false():
    span, _ = _recording_neatlogs_span()
    with otel_trace.use_span(span, end_on_exit=False):
        from neatlogs import annotate

        applied = annotate()
    assert applied is False


def test_annotate_partial_application_returns_true_if_any_valid():
    """Mixed valid/invalid kwargs: True if at least one was applied."""
    span, exporter = _recording_neatlogs_span()
    with otel_trace.use_span(span, end_on_exit=False):
        from neatlogs import annotate

        # valid=2 should land; the other two are rejected
        applied = annotate(valid=2, **{"with.dot": "x"}, **{"": "empty"})
    assert applied is True
    span.end()
    attrs = dict(exporter.get_finished_spans()[0].attributes or {})
    assert attrs["neatlogs.annotation.valid"] == 2


def test_add_event_creates_event_on_active_span():
    span, exporter = _recording_neatlogs_span()
    with otel_trace.use_span(span, end_on_exit=False):
        from neatlogs import add_event

        applied = add_event("user_feedback", rating=1, comment="wrong")
    assert applied is True
    span.end()
    finished = exporter.get_finished_spans()
    assert len(finished) == 1
    events = finished[0].events
    assert len(events) == 1
    assert events[0].name == "neatlogs.annotation.user_feedback"
    # Attributes are coerced through the same path as annotate.
    assert events[0].attributes.get("rating") == 1
    assert events[0].attributes.get("comment") == "wrong"


def test_add_event_returns_false_for_invalid_name():
    span, _ = _recording_neatlogs_span()
    with otel_trace.use_span(span, end_on_exit=False):
        from neatlogs import add_event

        assert add_event("") is False
        assert add_event("with.dot") is False
        assert add_event("neatlogs.x") is False
        assert add_event(123) is False  # non-string


def test_add_event_returns_false_for_no_active_span():
    _setup_tracer()
    from neatlogs import add_event

    assert add_event("x", rating=1) is False


def test_add_event_returns_false_for_foreign_span():
    exporter = _setup_tracer()
    tracer = otel_trace.get_tracer("not-neatlogs")
    foreign = tracer.start_span("foreign-span")
    with otel_trace.use_span(foreign, end_on_exit=False):
        from neatlogs import add_event

        applied = add_event("user_feedback", rating=1)
    assert applied is False
    foreign.end()
    finished = exporter.get_finished_spans()
    assert len(finished) == 1
    # OTel's ReadableSpan.events is a tuple, not a list.
    assert len(finished[0].events) == 0


def test_add_event_with_no_attrs_still_emits_event():
    """add_event('x') with no attributes still adds the event, returns True."""
    span, exporter = _recording_neatlogs_span()
    with otel_trace.use_span(span, end_on_exit=False):
        from neatlogs import add_event

        applied = add_event("checkpoint")
    assert applied is True
    span.end()
    events = exporter.get_finished_spans()[0].events
    assert len(events) == 1
    assert events[0].name == "neatlogs.annotation.checkpoint"


def test_annotate_thread_safety_via_otel_context():
    """OTel context propagates correctly across threads when a token is used.

    Note: OTel's contextvars-based propagation only works in the same thread
    or via explicit context.attach(). The annotation API relies on
    get_current_span() which reads the current OTel context. A thread spawned
    without context propagation will see no active span and the call will be
    a no-op (which is the documented behavior)."""
    span, exporter = _recording_neatlogs_span()
    token = otel_trace.context_api.attach(otel_trace.set_span_in_context(span))
    try:
        from neatlogs import annotate

        # Attached context: should land on the span.
        applied = annotate(needs_review=True)
        assert applied is True
    finally:
        otel_trace.context_api.detach(token)
    span.end()
    attrs = dict(exporter.get_finished_spans()[0].attributes or {})
    assert attrs.get("neatlogs.annotation.needs_review") is True


def test_annotate_pydantic_model_uses_model_dump():
    """Pydantic-style objects are dumped via model_dump() if available."""
    span, exporter = _recording_neatlogs_span()

    class _PydanticLike:
        def model_dump(self):
            return {"a": 1, "b": "two"}

    with otel_trace.use_span(span, end_on_exit=False):
        from neatlogs import annotate

        applied = annotate(blob=_PydanticLike())
    assert applied is True
    span.end()
    attrs = dict(exporter.get_finished_spans()[0].attributes or {})
    # Serialized as JSON of the model_dump output, not the repr.
    assert json.loads(attrs["neatlogs.annotation.blob"]) == {"a": 1, "b": "two"}


def test_annotate_repeated_calls_accumulate_attributes():
    """Multiple annotate() calls on the same span accumulate (no overwrite of
    earlier values for different keys; same key is last-write-wins)."""
    span, exporter = _recording_neatlogs_span()
    with otel_trace.use_span(span, end_on_exit=False):
        from neatlogs import annotate

        annotate(a=1)
        annotate(b=2)
        annotate(a=99)  # overwrites the earlier a
    span.end()
    attrs = dict(exporter.get_finished_spans()[0].attributes or {})
    assert attrs["neatlogs.annotation.a"] == 99
    assert attrs["neatlogs.annotation.b"] == 2
