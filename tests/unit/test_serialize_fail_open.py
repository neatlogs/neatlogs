"""Telemetry capture must never crash the traced application, even when a
captured value's __str__/__repr__ raises (fail-open serialization).

Regression test for the unguarded str() fallbacks in
neatlogs.decorators._base._serialize_obj / _safe_json_dumps.
"""

from opentelemetry import trace

from neatlogs._wrap_utils import set_neatlogs_provider
from neatlogs.decorators._base import _safe_json_dumps, _serialize_obj
from neatlogs.decorators.orchestration import span


class _BadRepr:
    """An object whose __repr__/__str__ raises — e.g. a half-initialized
    model or a lazy attribute that errors."""

    def __repr__(self):
        raise RuntimeError("repr raised during telemetry capture")


def _install(tracer_provider):
    trace.set_tracer_provider(tracer_provider)
    set_neatlogs_provider(tracer_provider)


def _finished(in_memory_span_exporter, name):
    return next(item for item in in_memory_span_exporter.get_finished_spans() if item.name == name)


def test_serialize_obj_does_not_raise_on_bad_repr():
    # Must degrade to a placeholder string, not raise.
    result = _serialize_obj(_BadRepr())
    assert isinstance(result, str)
    assert "unserializable" in result


def test_safe_json_dumps_does_not_raise_on_bad_repr():
    # Must always return a JSON string, never propagate an exception.
    result = _safe_json_dumps(_BadRepr())
    assert isinstance(result, str)
    assert "unserializable" in result


def test_span_output_capture_does_not_crash_on_bad_repr(tracer_provider, in_memory_span_exporter):
    _install(tracer_provider)

    @span(kind="TOOL", name="returns_bad_object")
    def returns_bad_object(x: int):
        return _BadRepr()

    # The decorated call must return normally — telemetry capture of the
    # return value must not crash the application.
    obj = returns_bad_object(42)
    assert isinstance(obj, _BadRepr)

    # The span is still recorded, with a placeholder output.value.
    captured = _finished(in_memory_span_exporter, "returns_bad_object")
    assert "output.value" in captured.attributes
    assert "unserializable" in captured.attributes["output.value"]


def test_span_input_capture_does_not_crash_on_bad_repr(tracer_provider, in_memory_span_exporter):
    _install(tracer_provider)

    @span(kind="TOOL", name="takes_bad_object")
    def takes_bad_object(obj):
        return "ok"

    result = takes_bad_object(_BadRepr())
    assert result == "ok"


def test_safe_json_dumps_non_finite_floats_are_valid_json():
    """NaN/Infinity/-Infinity are not valid JSON (RFC 8259) and are rejected
    by strict parsers like JavaScript's JSON.parse. _safe_json_dumps must not
    emit them as bare tokens."""
    import json as _json

    for value in (float("nan"), float("inf"), float("-inf")):
        out = _safe_json_dumps(value)
        # Must be parseable by a STRICT JSON parser (no NaN/Infinity extension).
        _json.loads(out, parse_constant=_reject_constant)


def test_safe_json_dumps_non_finite_nested_are_valid_json():
    import json as _json

    payload = {"score": float("nan"), "items": [1.0, float("inf"), 2.0]}
    out = _safe_json_dumps(payload)
    _json.loads(out, parse_constant=_reject_constant)


def _reject_constant(token):
    # json.loads calls parse_constant for NaN/Infinity/-Infinity tokens.
    raise AssertionError(f"non-standard JSON constant emitted: {token!r}")
