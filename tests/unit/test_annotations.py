"""Tests for the trace-annotations API (neatlogs.annotate / neatlogs.add_event)."""

import json
import math

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


# ---------------------------------------------------------------------------
# Isolated mode (private / secondary provider)
# ---------------------------------------------------------------------------


def test_annotate_works_in_isolated_mode_via_secondary_client():
    """In isolated mode (secondary Client activated), annotate() must land on
    the private parent set via attach_as_current, and NOT on the OTel global
    current-span (which is the host's in this mode)."""
    global_exporter = _setup_tracer()
    import neatlogs
    from neatlogs._wrap_utils import attach_as_current
    from neatlogs._wrap_utils import detach as _nl_detach

    client = neatlogs.Client(
        api_key="iso-key",
        workflow_name="iso",
        disable_export=True,
    )
    private_exporter = InMemorySpanExporter()
    client.tracer_provider.add_span_processor(SimpleSpanProcessor(private_exporter))
    try:
        with client.activate():
            tracer = client.tracer_provider.get_tracer("neatlogs.iso")
            span = tracer.start_span("iso-span")
            token = attach_as_current(span)
            try:
                from neatlogs import annotate

                applied = annotate(needs_review=True, severity="high")
            finally:
                _nl_detach(token)
            span.end()
        assert applied is True
        # The private provider's exporter should have the annotated span.
        # (The NeatlogsSpanProcessor also emits a synthetic trace-complete span,
        # so the count is >=1, not ==1. We assert on the named span, not the
        # total count.)
        private_finished = private_exporter.get_finished_spans()
        iso_spans = [s for s in private_finished if s.name == "iso-span"]
        assert len(iso_spans) == 1
        attrs = dict(iso_spans[0].attributes or {})
        assert attrs.get("neatlogs.annotation.needs_review") is True
        assert attrs.get("neatlogs.annotation.severity") == "high"
        # The OTel global exporter must NOT have seen the annotation.
        for s in global_exporter.get_finished_spans():
            assert "neatlogs.annotation.needs_review" not in (s.attributes or {})
    finally:
        client.shutdown()


def test_annotate_returns_false_in_isolated_mode_with_no_active_parent():
    """A secondary Client is activate()'d but no span is attached on the
    private parent key: annotate() must no-op (the resolver looks up the
    private key, not the OTel global, in isolated mode)."""
    import neatlogs

    client = neatlogs.Client(
        api_key="iso-key",
        workflow_name="iso",
        disable_export=True,
    )
    try:
        with client.activate():
            from neatlogs import annotate

            # No span attached on the private key, so no parent resolves.
            applied = annotate(needs_review=True)
        assert applied is False
    finally:
        client.shutdown()


def test_annotate_in_isolated_mode_ignores_foreign_otel_span():
    """In isolated mode, a foreign span on the OTel global current-span must
    NOT be annotated — the resolver uses the private parent key only, so
    the global current-span is irrelevant."""
    import neatlogs

    _setup_tracer()
    client = neatlogs.Client(
        api_key="iso-key",
        workflow_name="iso",
        disable_export=True,
    )
    try:
        with client.activate():
            foreign_tracer = otel_trace.get_tracer("not-neatlogs")
            foreign = foreign_tracer.start_span("foreign-on-global")
            with otel_trace.use_span(foreign, end_on_exit=False):
                from neatlogs import annotate

                applied = annotate(needs_review=True)
            foreign.end()
        # No neatlogs parent on the private key, so the foreign span is
        # not annotated and the call returns False.
        assert applied is False
    finally:
        client.shutdown()


# ---------------------------------------------------------------------------
# Value-type coercion
# ---------------------------------------------------------------------------


def test_annotate_coerces_utf8_decodable_bytes_to_string():
    span, exporter = _recording_neatlogs_span()
    with otel_trace.use_span(span, end_on_exit=False):
        from neatlogs import annotate

        applied = annotate(raw=b"hello world")
    assert applied is True
    span.end()
    attrs = dict(exporter.get_finished_spans()[0].attributes or {})
    assert attrs["neatlogs.annotation.raw"] == "hello world"


def test_annotate_falls_back_to_repr_for_undecodable_bytes():
    span, exporter = _recording_neatlogs_span()
    bad = b"\xff\xfe\x00\x01"  # not valid utf-8
    with otel_trace.use_span(span, end_on_exit=False):
        from neatlogs import annotate

        applied = annotate(raw=bad)
    assert applied is True
    span.end()
    attrs = dict(exporter.get_finished_spans()[0].attributes or {})
    # repr() of bytes uses escape sequences
    assert attrs["neatlogs.annotation.raw"].startswith("b'")


def test_annotate_coerces_naive_datetime_via_isoformat():
    from datetime import datetime

    span, exporter = _recording_neatlogs_span()
    dt = datetime(2026, 8, 14, 1, 30, 0)
    with otel_trace.use_span(span, end_on_exit=False):
        from neatlogs import annotate

        applied = annotate(ts=dt)
    assert applied is True
    span.end()
    attrs = dict(exporter.get_finished_spans()[0].attributes or {})
    assert attrs["neatlogs.annotation.ts"] == "2026-08-14T01:30:00"


def test_annotate_coerces_aware_datetime_via_isoformat():
    from datetime import datetime, timezone

    span, exporter = _recording_neatlogs_span()
    dt = datetime(2026, 8, 14, 1, 30, 0, tzinfo=timezone.utc)
    with otel_trace.use_span(span, end_on_exit=False):
        from neatlogs import annotate

        applied = annotate(ts=dt)
    assert applied is True
    span.end()
    attrs = dict(exporter.get_finished_spans()[0].attributes or {})
    # tz-aware ISO contains the offset (UTC → +00:00).
    assert "2026-08-14T01:30:00" in attrs["neatlogs.annotation.ts"]
    assert "+00:00" in attrs["neatlogs.annotation.ts"]


def test_annotate_coerces_tuple_to_json_array():
    span, exporter = _recording_neatlogs_span()
    with otel_trace.use_span(span, end_on_exit=False):
        from neatlogs import annotate

        applied = annotate(items=(1, 2, 3))
    assert applied is True
    span.end()
    attrs = dict(exporter.get_finished_spans()[0].attributes or {})
    # json.dumps renders tuples as JSON arrays.
    assert json.loads(attrs["neatlogs.annotation.items"]) == [1, 2, 3]


def test_annotate_coerces_deeply_nested_structures_to_json():
    span, exporter = _recording_neatlogs_span()
    nested = {"a": [{"b": [{"c": 1, "d": [True, None, "x"]}]}]}
    with otel_trace.use_span(span, end_on_exit=False):
        from neatlogs import annotate

        applied = annotate(tree=nested)
    assert applied is True
    span.end()
    attrs = dict(exporter.get_finished_spans()[0].attributes or {})
    assert json.loads(attrs["neatlogs.annotation.tree"]) == nested


def test_annotate_falls_back_to_str_for_unknown_object():
    """A custom object with no model_dump / no isoformat falls back to str()."""

    class _Unknown:
        def __repr__(self):
            return "<Unknown 0xDEAD>"

    span, exporter = _recording_neatlogs_span()
    with otel_trace.use_span(span, end_on_exit=False):
        from neatlogs import annotate

        applied = annotate(obj=_Unknown())
    assert applied is True
    span.end()
    attrs = dict(exporter.get_finished_spans()[0].attributes or {})
    assert attrs["neatlogs.annotation.obj"] == "<Unknown 0xDEAD>"


def test_annotate_handles_failing_model_dump_by_falling_through():
    """If model_dump() raises, the coercion must fall through (not propagate)."""

    class _Broken:
        def model_dump(self):
            raise RuntimeError("nope")

    span, exporter = _recording_neatlogs_span()
    with otel_trace.use_span(span, end_on_exit=False):
        from neatlogs import annotate

        applied = annotate(blob=_Broken())
    assert applied is True
    span.end()
    attrs = dict(exporter.get_finished_spans()[0].attributes or {})
    # No model_dump, no isoformat → str() fallback. Just confirm the attr
    # landed as a string, regardless of exact value.
    assert isinstance(attrs["neatlogs.annotation.blob"], str)


# ---------------------------------------------------------------------------
# Key edge cases
# ---------------------------------------------------------------------------


def test_annotate_rejects_unicode_keys():
    span, _ = _recording_neatlogs_span()
    with otel_trace.use_span(span, end_on_exit=False):
        from neatlogs import annotate

        applied = annotate(**{"café": 1})
    # Non-ASCII chars not in _VALID_KEY_CHARS → rejected
    assert applied is False


def test_annotate_rejects_special_char_keys():
    span, _ = _recording_neatlogs_span()
    with otel_trace.use_span(span, end_on_exit=False):
        from neatlogs import annotate

        applied = annotate(**{"!@#$": 1, "a b": 2, "x,y": 3})
    # All rejected → no attr applied → False
    assert applied is False


def test_annotate_all_none_values_returns_false():
    """When every value is None, no attribute is set, so the call returns False."""
    span, _ = _recording_neatlogs_span()
    with otel_trace.use_span(span, end_on_exit=False):
        from neatlogs import annotate

        applied = annotate(a=None, b=None, c=None)
    assert applied is False


def test_annotate_accepts_alphanumeric_underscore_and_hyphen():
    """Sanity: the documented allowed character set actually works."""
    span, exporter = _recording_neatlogs_span()
    with otel_trace.use_span(span, end_on_exit=False):
        from neatlogs import annotate

        applied = annotate(plain="x", with_under="y", with_hyphen="z", mixed_123="w")
    assert applied is True
    span.end()
    attrs = dict(exporter.get_finished_spans()[0].attributes or {})
    assert attrs["neatlogs.annotation.plain"] == "x"
    assert attrs["neatlogs.annotation.with_under"] == "y"
    assert attrs["neatlogs.annotation.with_hyphen"] == "z"
    assert attrs["neatlogs.annotation.mixed_123"] == "w"


# ---------------------------------------------------------------------------
# add_event edge cases
# ---------------------------------------------------------------------------


def test_add_event_with_all_none_attrs_still_emits_event():
    """add_event('x') with all-None attrs still emits the event, returns True."""
    span, exporter = _recording_neatlogs_span()
    with otel_trace.use_span(span, end_on_exit=False):
        from neatlogs import add_event

        applied = add_event("checkpoint", a=None, b=None)
    assert applied is True
    span.end()
    events = exporter.get_finished_spans()[0].events
    assert len(events) == 1
    assert events[0].name == "neatlogs.annotation.checkpoint"


def test_add_event_rejects_reserved_prefix_attr_keys():
    """Event attrs with a 'neatlogs.' key are rejected (same as annotate), but
    the event itself is still emitted."""
    span, exporter = _recording_neatlogs_span()
    with otel_trace.use_span(span, end_on_exit=False):
        from neatlogs import add_event

        applied = add_event("user_feedback", **{"neatlogs.injected": "x"}, valid=1)
    assert applied is True
    span.end()
    events = exporter.get_finished_spans()[0].events
    assert len(events) == 1
    assert events[0].attributes.get("valid") == 1
    assert "neatlogs.injected" not in (events[0].attributes or {})


def test_add_event_with_same_name_emits_multiple_events():
    """Two add_event calls with the same name produce two distinct events."""
    span, exporter = _recording_neatlogs_span()
    with otel_trace.use_span(span, end_on_exit=False):
        from neatlogs import add_event

        add_event("retry", attempt=1)
        add_event("retry", attempt=2)
    span.end()
    events = exporter.get_finished_spans()[0].events
    assert len(events) == 2
    assert all(e.name == "neatlogs.annotation.retry" for e in events)
    assert events[0].attributes.get("attempt") == 1
    assert events[1].attributes.get("attempt") == 2


def test_add_event_with_pydantic_attr_uses_model_dump():
    """An attr that is a pydantic-like object is serialized via model_dump."""

    class _Nested:
        def model_dump(self):
            return {"k": "v"}

    span, exporter = _recording_neatlogs_span()
    with otel_trace.use_span(span, end_on_exit=False):
        from neatlogs import add_event

        applied = add_event("feedback", payload=_Nested())
    assert applied is True
    span.end()
    events = exporter.get_finished_spans()[0].events
    assert json.loads(events[0].attributes["payload"]) == {"k": "v"}


def test_add_event_with_object_no_model_dump_no_isoformat():
    """Plain object: coercion falls back to str()."""

    class _Plain:
        def __repr__(self):
            return "PLAIN_REPR"

    span, exporter = _recording_neatlogs_span()
    with otel_trace.use_span(span, end_on_exit=False):
        from neatlogs import add_event

        applied = add_event("feedback", payload=_Plain())
    assert applied is True
    span.end()
    events = exporter.get_finished_spans()[0].events
    assert events[0].attributes["payload"] == "PLAIN_REPR"


def test_add_event_with_datetime_attr_isoformat():
    from datetime import datetime, timezone

    span, exporter = _recording_neatlogs_span()
    with otel_trace.use_span(span, end_on_exit=False):
        from neatlogs import add_event

        applied = add_event("ping", at=datetime(2026, 8, 14, tzinfo=timezone.utc))
    assert applied is True
    span.end()
    events = exporter.get_finished_spans()[0].events
    assert "2026-08-14" in events[0].attributes["at"]


def test_add_event_with_bytes_attr_decoded():
    span, exporter = _recording_neatlogs_span()
    with otel_trace.use_span(span, end_on_exit=False):
        from neatlogs import add_event

        applied = add_event("ping", payload=b"raw-bytes")
    assert applied is True
    span.end()
    events = exporter.get_finished_spans()[0].events
    assert events[0].attributes["payload"] == "raw-bytes"


# ---------------------------------------------------------------------------
# Fail-open contract
# ---------------------------------------------------------------------------


def test_resolve_neatlogs_span_swallows_internal_exception(monkeypatch):
    """If _isolation_active() itself raises, the resolver must return None
    (the documented fail-open contract) — never propagate."""
    import neatlogs._annotations as _ann

    def boom():
        raise RuntimeError("simulated internal failure")

    monkeypatch.setattr(_ann, "_isolation_active", boom)
    from neatlogs import annotate

    # Must NOT raise; must return False.
    assert annotate(needs_review=True) is False


def test_resolve_neatlogs_span_swallows_get_current_exception(monkeypatch):
    """If otel_trace.get_current_span() raises (e.g. context corruption), the
    resolver must still return None — never propagate."""

    def boom():
        raise RuntimeError("otel state corrupted")

    monkeypatch.setattr(otel_trace, "get_current_span", boom)
    from neatlogs import annotate

    assert annotate(needs_review=True) is False


# ---------------------------------------------------------------------------
# End-to-end public API smoke test (real init / trace / annotate)
# ---------------------------------------------------------------------------


def test_end_to_end_init_trace_annotate_add_event():
    """Real-world path: neatlogs.init() + neatlogs.trace() + annotate() +
    add_event() all cooperate and the annotations land on the WORKFLOW span.
    """
    import neatlogs

    neatlogs.init(
        api_key="smoke-key",
        workflow_name="smoke",
        disable_export=True,
        instrumentations=[],
    )
    try:
        with neatlogs.trace("smoke-flow", kind="WORKFLOW") as span:
            from neatlogs import add_event, annotate

            applied_attr = annotate(review="ok", score=0.95)
            applied_evt = add_event("user_ping", latency_ms=42)
        assert applied_attr is True
        assert applied_evt is True
        attrs = dict(span.attributes or {})
        assert attrs.get("neatlogs.annotation.review") == "ok"
        assert attrs.get("neatlogs.annotation.score") == 0.95
        events = list(span.events or [])
        assert any(e.name == "neatlogs.annotation.user_ping" for e in events)
    finally:
        neatlogs.shutdown()


# ---------------------------------------------------------------------------
# Concurrency (asyncio + threads)
# ---------------------------------------------------------------------------


def test_annotate_works_inside_asyncio_task():
    """Annotation API works inside an asyncio task — OTel context propagates
    via contextvars across `await` points."""
    import asyncio

    span, exporter = _recording_neatlogs_span()

    async def annotate_inside():
        from neatlogs import add_event, annotate

        # The asyncio scheduler runs this on the same task, so the current
        # OTel context (set by use_span above) is preserved.
        applied_a = annotate(async_attr="x")
        await asyncio.sleep(0)
        applied_b = add_event("async_event", tick=1)
        return applied_a, applied_b

    with otel_trace.use_span(span, end_on_exit=False):
        a, b = asyncio.run(annotate_inside())
    span.end()
    assert a is True
    assert b is True
    finished = exporter.get_finished_spans()
    assert finished[0].attributes.get("neatlogs.annotation.async_attr") == "x"
    assert any(e.name == "neatlogs.annotation.async_event" for e in finished[0].events)


def test_annotate_concurrent_threads_with_attached_context():
    """Multi-thread: when the OTel context is attached in each worker, annotation
    from multiple threads lands on the same span. OTel spans are thread-safe for
    set_attribute. (ContextVar does not propagate to new threads automatically —
    each worker must attach the captured context.)"""
    import threading

    span, exporter = _recording_neatlogs_span()
    # Capture the context in the main thread, then re-attach in each worker.
    parent_ctx = otel_trace.set_span_in_context(span)

    results = []
    errors = []

    def worker(i: int) -> None:
        token = otel_trace.context_api.attach(parent_ctx)
        try:
            from neatlogs import annotate

            # Use unique keys per worker so all 8 writes are visible.
            results.append(annotate(**{f"worker_{i}": i, f"thread_{i}": f"t{i}"}))
        except Exception as e:
            errors.append(e)
        finally:
            otel_trace.context_api.detach(token)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    span.end()

    assert errors == []
    assert all(r is True for r in results)
    # All 8 workers' unique attrs must be on the span.
    finished = exporter.get_finished_spans()
    attrs = dict(finished[0].attributes or {})
    for i in range(8):
        assert attrs.get(f"neatlogs.annotation.worker_{i}") == i
        assert attrs.get(f"neatlogs.annotation.thread_{i}") == f"t{i}"


# ---------------------------------------------------------------------------
# Integration with other neatlogs public APIs
# ---------------------------------------------------------------------------


def test_annotate_works_inside_span_decorator():
    """Annotation from inside an @span-decorated function lands on that span."""
    import neatlogs

    neatlogs.init(
        api_key="dec-key",
        workflow_name="dec",
        disable_export=True,
        instrumentations=[],
    )
    try:

        @neatlogs.span(kind="AGENT")
        def my_agent(query: str) -> str:
            from neatlogs import annotate

            applied = annotate(agent_query=query, step="process")
            assert applied is True
            return "done"

        result = my_agent("hello")
        assert result == "done"
    finally:
        neatlogs.shutdown()


def test_annotate_with_identify_context_manager():
    """annotate() inside a neatlogs.identify() context works — the
    annotations land on the active span without disturbing identity attrs."""
    import neatlogs

    neatlogs.init(
        api_key="ident-key",
        workflow_name="ident",
        disable_export=True,
        instrumentations=[],
    )
    try:
        with neatlogs.identify(session_id="sess-1", end_user_id="user-1"):
            with neatlogs.trace("ident-flow", kind="WORKFLOW") as span:
                from neatlogs import annotate

                applied = annotate(review="ok")
        assert applied is True
        attrs = dict(span.attributes or {})
        assert attrs.get("neatlogs.annotation.review") == "ok"
        # Identity attrs from identify() are stamped on the root span.
        assert attrs.get("neatlogs.session.id") == "sess-1"
        assert attrs.get("neatlogs.end_user.id") == "user-1"
    finally:
        neatlogs.shutdown()


def test_annotate_then_log_then_annotate_chain():
    """annotate() before and after neatlogs.log() — both annotations land
    on the same span; log() does not disturb the active-span resolver."""
    import neatlogs

    neatlogs.init(
        api_key="chain-key",
        workflow_name="chain",
        disable_export=True,
        instrumentations=[],
    )
    try:
        with neatlogs.trace("chain-flow", kind="WORKFLOW") as span:
            from neatlogs import annotate, log

            annotate(before_log="x")
            log("in the middle")
            annotate(after_log="y")
        attrs = dict(span.attributes or {})
        assert attrs.get("neatlogs.annotation.before_log") == "x"
        assert attrs.get("neatlogs.annotation.after_log") == "y"
    finally:
        neatlogs.shutdown()


# ---------------------------------------------------------------------------
# Public API surface
# ---------------------------------------------------------------------------


def test_annotate_and_add_event_in_module_dir_and_all():
    """`neatlogs.annotate` and `neatlogs.add_event` are accessible both as
    attributes on the module and via `from neatlogs import ...`."""
    import neatlogs

    # Attribute access
    assert callable(neatlogs.annotate)
    assert callable(neatlogs.add_event)
    # Listed in __all__ so star-imports work
    assert "annotate" in neatlogs.__all__
    assert "add_event" in neatlogs.__all__
    # `dir(neatlogs)` includes them
    assert "annotate" in dir(neatlogs)
    assert "add_event" in dir(neatlogs)


# ---------------------------------------------------------------------------
# Stress / scale
# ---------------------------------------------------------------------------


def test_annotate_handles_thousand_attributes():
    """A single annotate() call with many attrs all land on the span. The
    OTel SDK's default span attribute limit is 128; we set a higher limit to
    match the production TracerProvider config (10_000)."""
    from opentelemetry.sdk.trace import SpanLimits, TracerProvider

    exporter = InMemorySpanExporter()
    provider = TracerProvider(span_limits=SpanLimits(max_span_attributes=2_000))
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    otel_trace.set_tracer_provider(provider)

    tracer = otel_trace.get_tracer("neatlogs.test")
    span = tracer.start_span("test-span")
    span.set_attribute("neatlogs.span.kind", "test")

    attrs = {f"k{i}": i for i in range(1000)}
    with otel_trace.use_span(span, end_on_exit=False):
        from neatlogs import annotate

        applied = annotate(**attrs)
    assert applied is True
    span.end()
    finished = exporter.get_finished_spans()
    span_attrs = dict(finished[0].attributes or {})
    annotated_count = sum(1 for k in span_attrs if k.startswith("neatlogs.annotation.k"))
    assert annotated_count == 1000


def test_add_event_handles_thousand_events():
    """1000 add_event() calls produce 1000 events on the span. The OTel SDK
    default event limit is 128; we set a higher limit to match production."""
    from opentelemetry.sdk.trace import SpanLimits, TracerProvider

    exporter = InMemorySpanExporter()
    provider = TracerProvider(span_limits=SpanLimits(max_events=2_000))
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    otel_trace.set_tracer_provider(provider)

    tracer = otel_trace.get_tracer("neatlogs.test")
    span = tracer.start_span("test-span")
    span.set_attribute("neatlogs.span.kind", "test")

    with otel_trace.use_span(span, end_on_exit=False):
        from neatlogs import add_event

        for i in range(1000):
            assert add_event("tick", n=i) is True
    span.end()
    events = exporter.get_finished_spans()[0].events
    assert len(events) == 1000
    assert all(e.name == "neatlogs.annotation.tick" for e in events)


# ---------------------------------------------------------------------------
# Value-type edge cases
# ---------------------------------------------------------------------------


def test_annotate_with_empty_bytes_value():
    """Empty bytes: utf-8 decode returns '', which is a valid string."""
    span, exporter = _recording_neatlogs_span()
    with otel_trace.use_span(span, end_on_exit=False):
        from neatlogs import annotate

        applied = annotate(empty=b"")
    assert applied is True
    span.end()
    assert exporter.get_finished_spans()[0].attributes["neatlogs.annotation.empty"] == ""


def test_annotate_accepts_digit_only_and_single_char_keys():
    """Digit-only keys and single-char keys are allowed (alphanumeric)."""
    span, exporter = _recording_neatlogs_span()
    with otel_trace.use_span(span, end_on_exit=False):
        from neatlogs import annotate

        applied = annotate(**{"0": "zero", "x": "x-val", "1": "one"})
    assert applied is True
    span.end()
    attrs = dict(exporter.get_finished_spans()[0].attributes or {})
    assert attrs["neatlogs.annotation.0"] == "zero"
    assert attrs["neatlogs.annotation.x"] == "x-val"
    assert attrs["neatlogs.annotation.1"] == "one"


def test_annotate_does_not_double_namespace_neatlogs_key():
    """A user key that happens to start with 'neatlogs' (no dot) is allowed
    and namespaced to 'neatlogs.annotation.<key>' — no double-namespacing."""
    span, exporter = _recording_neatlogs_span()
    with otel_trace.use_span(span, end_on_exit=False):
        from neatlogs import annotate

        # "neatlogsx" starts with "neatlogs" but not "neatlogs." — allowed.
        # The reserved-prefix check is on "neatlogs." (with the dot).
        applied = annotate(neatlogsx="ok", neatlogs_span="ok2")
    assert applied is True
    span.end()
    attrs = dict(exporter.get_finished_spans()[0].attributes or {})
    # Both keys land under neatlogs.annotation.<key> — no "neatlogs.annotation.neatlogs.annotation..."
    assert attrs["neatlogs.annotation.neatlogsx"] == "ok"
    assert attrs["neatlogs.annotation.neatlogs_span"] == "ok2"


# ---------------------------------------------------------------------------
# Idempotence / chain semantics
# ---------------------------------------------------------------------------


def test_annotate_inside_annotate_loop_does_not_stack():
    """annotate() calls don't interfere with each other; each call
    independently resolves the active span."""
    span, exporter = _recording_neatlogs_span()
    with otel_trace.use_span(span, end_on_exit=False):
        from neatlogs import add_event, annotate

        for i in range(5):
            annotate(loop_step=i)
            add_event("step", i=i)
    span.end()
    attrs = dict(exporter.get_finished_spans()[0].attributes or {})
    for i in range(5):
        assert attrs["neatlogs.annotation.loop_step"] == 4  # last wins
    events = exporter.get_finished_spans()[0].events
    assert len(events) == 5
    for i, e in enumerate(events):
        assert e.attributes.get("i") == i


# ---------------------------------------------------------------------------
# consult-kiro panel findings — gap coverage
# ---------------------------------------------------------------------------
# The 4-model panel (deepseek-v4-flash-free, hy3-free, mimo-v2.5-free, plus
# big-pickle which failed to produce output) reviewed the implementation and
# identified scenarios not covered by the 59-test baseline. F2 and F5 used
# to be xfail-marked known bugs; they are now fixed and the tests run as
# plain asserts.


def test_add_event_returns_false_on_ended_span():
    """Gap: equivalent of test_annotate_returns_false_for_ended_span for
    add_event. The implementation handles this via the resolver, but the
    path is untested."""
    span, _ = _recording_neatlogs_span()
    with otel_trace.use_span(span, end_on_exit=False):
        from neatlogs import add_event

        span.end()
        applied = add_event("after_end", x=1)
    assert applied is False


def test_resolve_neatlogs_span_swallows_parent_exception_in_isolated_mode(
    monkeypatch,
):
    """Gap: the existing fail-open test covers _isolation_active raising;
    this covers _current_neatlogs_parent raising in the isolated branch
    (the outer except at the resolver still catches it)."""
    import neatlogs._wrap_utils as _wu

    def boom_isolation():
        return True

    def boom_parent():
        raise RuntimeError("context corrupted")

    monkeypatch.setattr(_wu, "_isolation_active", boom_isolation)
    monkeypatch.setattr(_wu, "_current_neatlogs_parent", boom_parent)
    from neatlogs import annotate

    assert annotate(needs_review=True) is False


def test_annotate_returns_false_with_all_invalid_keys_on_active_span():
    """Gap: no test for the all-keys-invalid case. annotate() must return
    False when every supplied key is rejected."""
    span, _ = _recording_neatlogs_span()
    with otel_trace.use_span(span, end_on_exit=False):
        from neatlogs import annotate

        applied = annotate(**{"with.dot": 1, "": 2, "neatlogs.x": 3})
    assert applied is False


def test_isolated_mode_through_flag_not_client_activate(monkeypatch):
    """Gap: every isolated-mode test uses client.activate() which sets
    _active_client. The other path is _isolated = True (set via
    set_neatlogs_provider with a private provider). Verify that path
    resolves through the private parent key."""
    import neatlogs._wrap_utils as _wu
    from neatlogs._wrap_utils import attach_as_current
    from neatlogs._wrap_utils import detach as _nl_detach

    _setup_tracer()
    monkeypatch.setattr(_wu, "_isolated", True)
    # Force _active_client to None so the only isolation signal is the
    # _isolated flag. ContextVar.set returns a token; we reset in finally.
    client_token = _wu._active_client.set(None)
    try:
        tracer = otel_trace.get_tracer("neatlogs.iso")
        span = tracer.start_span("iso-span")
        token = attach_as_current(span)
        try:
            from neatlogs import annotate

            applied = annotate(flag_path=True)
        finally:
            _nl_detach(token)
            span.end()
    finally:
        _wu._active_client.reset(client_token)
        monkeypatch.setattr(_wu, "_isolated", False)
    assert applied is True


def test_add_event_name_with_special_chars_is_accepted():
    """Gap: add_event only checks for dots and the 'neatlogs.' prefix; the
    rest is accepted. This is an asymmetry vs _valid_key which restricts
    to [a-zA-Z0-9_-]. The test documents the current behavior so any
    future tightening shows up as a behavior change."""
    span, exporter = _recording_neatlogs_span()
    with otel_trace.use_span(span, end_on_exit=False):
        from neatlogs import add_event

        applied = add_event("user feedback", rating=1)
    assert applied is True
    span.end()
    events = exporter.get_finished_spans()[0].events
    assert events[0].name == "neatlogs.annotation.user feedback"


def test_bare_thread_without_attached_context_returns_false():
    """Gap: the multi-thread test attaches context inside each worker.
    This test documents the spec'd behavior: a thread spawned without
    context propagation sees no active span and the call is a no-op."""
    import threading

    _setup_tracer()
    from neatlogs import annotate

    results = {}

    def worker():
        # No context_api.attach here — this thread sees no current span.
        results["applied"] = annotate(needs_review=True)

    t = threading.Thread(target=worker)
    t.start()
    t.join()
    assert results["applied"] is False


def test_annotate_handles_set_value_via_str_fallback():
    """Gap: set/frozenset values fall through to str() producing
    non-JSON output like '{1, 2, 3}'. Document the current behavior:
    the call succeeds, the value lands as a Python repr string."""
    span, exporter = _recording_neatlogs_span()
    with otel_trace.use_span(span, end_on_exit=False):
        from neatlogs import annotate

        applied = annotate(tags={1, 2, 3})
    assert applied is True
    span.end()
    attrs = dict(exporter.get_finished_spans()[0].attributes or {})
    # set is serialized via str() — not JSON.
    assert attrs["neatlogs.annotation.tags"].startswith("{")


# Tests that document F1 — fail-open contract.
# Fix verified: annotate() now wraps _coerce and set_attribute in try/except
# so a single bad value doesn't crash the call. Other kwargs in the same
# call still apply (partial application).


def test_annotate_swallows_circular_reference_dict():
    """F1 fix: circular-reference dict must not crash annotate(). The
    failing value is silently skipped, and earlier/sibling attrs in the
    same call still apply."""
    span, exporter = _recording_neatlogs_span()
    payload = {}
    payload["self"] = payload
    with otel_trace.use_span(span, end_on_exit=False):
        from neatlogs import annotate

        # The circular ref's coerce fails; the good attr still lands.
        applied = annotate(payload=payload, good=1)
    span.end()
    # Returns True because at least one attr landed.
    assert applied is True
    finished = exporter.get_finished_spans()
    attrs = dict(finished[0].attributes or {})
    assert attrs.get("neatlogs.annotation.good") == 1
    # The bad value was silently skipped.
    assert "neatlogs.annotation.payload" not in attrs


def test_annotate_swallows_hostile_str():
    """F1 fix: object whose __str__ raises must not crash annotate()."""

    class _Hostile:
        def __str__(self):
            raise RuntimeError("boom")

        def __repr__(self):
            raise RuntimeError("boom")

    span, exporter = _recording_neatlogs_span()
    with otel_trace.use_span(span, end_on_exit=False):
        from neatlogs import annotate

        # Hostile coerce fails; sibling still lands.
        applied = annotate(blob=_Hostile(), ok=42)
    span.end()
    assert applied is True
    attrs = dict(exporter.get_finished_spans()[0].attributes or {})
    assert attrs.get("neatlogs.annotation.ok") == 42
    assert "neatlogs.annotation.blob" not in attrs


def test_annotate_swallows_set_attribute_exception(monkeypatch):
    """F1 fix: span.set_attribute raising must be caught; the fail-open
    contract is at the API boundary, not just the resolver."""
    from neatlogs import annotate

    span, exporter = _recording_neatlogs_span()

    def fake_set_attr(key, value):
        raise RuntimeError("OTel SDK rejected this attribute")

    monkeypatch.setattr(span, "set_attribute", fake_set_attr)
    with otel_trace.use_span(span, end_on_exit=False):
        # The bad attr is silently skipped.
        applied = annotate(any_key=1, other=2)
    span.end()
    # No attrs landed (set_attribute raises for every value).
    assert applied is False
    attrs = dict(exporter.get_finished_spans()[0].attributes or {})
    assert "neatlogs.annotation.any_key" not in attrs
    assert "neatlogs.annotation.other" not in attrs


def test_nested_nan_inside_dict_is_stringified():
    """NaN/Inf nested inside a dict must be stringified, not left as bare
    JSON tokens (which ClickHouse would reject)."""
    import math

    span, exporter = _recording_neatlogs_span()
    with otel_trace.use_span(span, end_on_exit=False):
        from neatlogs import annotate

        applied = annotate(payload={"score": math.nan, "limit": math.inf})
    assert applied is True
    span.end()
    raw = exporter.get_finished_spans()[0].attributes["neatlogs.annotation.payload"]
    # The serialized JSON must contain quoted 'NaN' / 'Infinity', not bare tokens.
    assert '"NaN"' in raw
    assert '"Infinity"' in raw


def test_annotate_truncates_oversized_values():
    """Large values are truncated to match the rest of the SDK's behavior
    (100KB ceiling via serialize())."""
    span, exporter = _recording_neatlogs_span()
    huge = "x" * 200_000
    with otel_trace.use_span(span, end_on_exit=False):
        from neatlogs import annotate

        applied = annotate(big=huge)
    assert applied is True
    span.end()
    attrs = dict(exporter.get_finished_spans()[0].attributes or {})
    raw = attrs["neatlogs.annotation.big"]
    # ~100KB + the "...[truncated]" marker.
    assert len(raw) < 110_000


def test_annotate_with_oversized_value_does_not_crash():
    """Sanity: a large value should at minimum not crash; whatever the
    truncation policy is, the call must return True and not raise."""
    span, exporter = _recording_neatlogs_span()
    huge = "x" * 200_000
    with otel_trace.use_span(span, end_on_exit=False):
        from neatlogs import annotate

        applied = annotate(big=huge)
    assert applied is True
    span.end()
    attrs = dict(exporter.get_finished_spans()[0].attributes or {})
    assert "neatlogs.annotation.big" in attrs
