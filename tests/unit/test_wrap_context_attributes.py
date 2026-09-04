"""Tests for apply_wrap_context_attributes — the workflow-attribute path
used by neatlogs.wrap(client, **workflow_attributes).

Regression coverage for the str()-everything bug: previously every value
was coerced via str() so ints became "42", bools became "True", lists
became Python reprs — silently breaking backend filtering on the resulting
neatlogs.workflow.* attributes.
"""

import json
import math

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import neatlogs._wrap_utils as _wu
from neatlogs._wrap_utils import (
    _wrap_context,
    apply_wrap_context_attributes,
)


def _setup_provider():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider, exporter


def _record_attributes(values):
    """Stamp a set of workflow attrs on a fresh span via
    apply_wrap_context_attributes and return the resulting attribute dict.
    """
    provider, exporter = _setup_provider()
    tracer = provider.get_tracer("test")
    token = _wrap_context.set({"workflow": values})
    try:
        with tracer.start_as_current_span("root") as span:
            apply_wrap_context_attributes(span, is_root=True)
    finally:
        _wrap_context.reset(token)
    return dict(exporter.get_finished_spans()[0].attributes or {})


def test_wrap_context_preserves_string_value():
    attrs = _record_attributes({"project_id": "proj-123"})
    assert attrs["neatlogs.workflow.project_id"] == "proj-123"


def test_wrap_context_preserves_int_value_natively():
    """The bug: int 42 was stored as the string "42", breaking
    backend numeric filters on neatlogs.workflow.count."""
    attrs = _record_attributes({"count": 42})
    assert attrs["neatlogs.workflow.count"] == 42
    assert isinstance(attrs["neatlogs.workflow.count"], int)


def test_wrap_context_preserves_bool_value_natively():
    """The bug: bool True was stored as the string "True"."""
    attrs = _record_attributes({"enabled": True, "archived": False})
    assert attrs["neatlogs.workflow.enabled"] is True
    assert attrs["neatlogs.workflow.archived"] is False
    assert isinstance(attrs["neatlogs.workflow.enabled"], bool)
    assert isinstance(attrs["neatlogs.workflow.archived"], bool)


def test_wrap_context_preserves_float_value_natively():
    """The bug: float 0.95 was stored as the string "0.95"."""
    attrs = _record_attributes({"ratio": 0.95})
    assert attrs["neatlogs.workflow.ratio"] == 0.95
    assert isinstance(attrs["neatlogs.workflow.ratio"], float)


def test_wrap_context_serializes_list_as_json():
    """The bug: list ["a", "b"] was stored as Python repr
    "['a', 'b']", not valid JSON, breaking any list-shaped filter."""
    attrs = _record_attributes({"tags": ["a", "b", "c"]})
    raw = attrs["neatlogs.workflow.tags"]
    assert json.loads(raw) == ["a", "b", "c"]


def test_wrap_context_serializes_dict_as_json():
    """The bug: dict {"k": "v"} was stored as Python repr."""
    attrs = _record_attributes({"meta": {"k": "v", "n": 1}})
    raw = attrs["neatlogs.workflow.meta"]
    assert json.loads(raw) == {"k": "v", "n": 1}


def test_wrap_context_stringifies_nan_and_inf():
    """NaN/Inf floats must be stringified so strict JSON parsers
    (ClickHouse) don't reject the attribute. Mirrors the
    neatlogs._annotations._coerce contract."""
    attrs = _record_attributes({"nan_v": math.nan, "inf_v": math.inf, "neg_inf_v": -math.inf})
    assert attrs["neatlogs.workflow.nan_v"] == "NaN"
    assert attrs["neatlogs.workflow.inf_v"] == "Infinity"
    assert attrs["neatlogs.workflow.neg_inf_v"] == "-Infinity"


def test_wrap_context_does_not_set_attribute_for_non_root():
    """is_root=False must skip stamping — the auto-root is the only span
    that gets workflow attributes, not nested children."""
    provider, exporter = _setup_provider()
    tracer = provider.get_tracer("test")
    token = _wrap_context.set({"workflow": {"k": "v"}})
    try:
        with tracer.start_as_current_span("root") as root:
            apply_wrap_context_attributes(root, is_root=True)
            with tracer.start_as_current_span("child") as child:
                apply_wrap_context_attributes(child, is_root=False)
    finally:
        _wrap_context.reset(token)
    by_name = {s.name: dict(s.attributes or {}) for s in exporter.get_finished_spans()}
    # The root span has the attribute; the child does not.
    assert "neatlogs.workflow.k" in by_name["root"]
    assert by_name["root"]["neatlogs.workflow.k"] == "v"
    assert "neatlogs.workflow.k" not in by_name["child"]


def test_wrap_context_swallows_attribute_set_exception():
    """If span.set_attribute raises (e.g. an OTel attribute limit), the
    function must not propagate — the wrap call should never crash the
    user's app on a metadata-stamp failure."""
    provider, exporter = _setup_provider()
    tracer = provider.get_tracer("test")

    class _RaisingSpan:
        def set_attribute(self, key, value):
            raise RuntimeError("OTel rejected this attribute")

    token = _wrap_context.set({"workflow": {"k": "v"}})
    try:
        # Must not raise.
        apply_wrap_context_attributes(_RaisingSpan(), is_root=True)
    finally:
        _wrap_context.reset(token)
    # No spans were started on the tracer, so the finished list is empty.
    assert len(exporter.get_finished_spans()) == 0


def test_wrap_context_with_empty_context_is_noop():
    """No context set → no attributes stamped (no NPE)."""
    provider, exporter = _setup_provider()
    tracer = provider.get_tracer("test")
    span = tracer.start_span("x")
    apply_wrap_context_attributes(span, is_root=True)
    span.end()
    attrs = dict(exporter.get_finished_spans()[0].attributes or {})
    assert "neatlogs.workflow" not in {
        k.split(".")[1] for k in attrs if k.startswith("neatlogs.workflow.")
    }


# ---------------------------------------------------------------------------
# Additional edge cases surfaced by /code-review on PR #70
# ---------------------------------------------------------------------------


def test_wrap_context_decodes_utf8_bytes():
    """Bytes values are utf-8 decoded (per the docstring)."""
    attrs = _record_attributes({"raw": b"hello world"})
    assert attrs["neatlogs.workflow.raw"] == "hello world"


def test_wrap_context_handles_non_decodable_bytes():
    """Non-utf-8 bytes fall back to repr() so the attribute still lands."""
    attrs = _record_attributes({"raw": b"\xff\xfe\x00"})
    # repr() of bytes gives an escape-sequence string.
    assert attrs["neatlogs.workflow.raw"].startswith("b'")


def test_wrap_context_handles_empty_bytes():
    """Empty bytes: utf-8 decode returns ''. Stored as empty string."""
    attrs = _record_attributes({"raw": b""})
    assert attrs["neatlogs.workflow.raw"] == ""


def test_wrap_context_serializes_tuple_as_json():
    """Tuple goes through the dict/list/tuple branch → JSON array."""
    attrs = _record_attributes({"coords": (1, 2, 3)})
    assert json.loads(attrs["neatlogs.workflow.coords"]) == [1, 2, 3]


def test_wrap_context_stamps_multiple_values_in_one_call():
    """Many attrs in one call all land — verifies the loop, not just one."""
    attrs = _record_attributes(
        {
            "a": 1,
            "b": "two",
            "c": True,
            "d": 0.5,
            "e": ["x", "y"],
            "f": {"k": 1},
        }
    )
    assert attrs["neatlogs.workflow.a"] == 1
    assert attrs["neatlogs.workflow.b"] == "two"
    assert attrs["neatlogs.workflow.c"] is True
    assert attrs["neatlogs.workflow.d"] == 0.5
    assert json.loads(attrs["neatlogs.workflow.e"]) == ["x", "y"]
    assert json.loads(attrs["neatlogs.workflow.f"]) == {"k": 1}


def test_wrap_context_stringifies_nested_nan_in_dict():
    """Real bug (nested NaN/Inf must not emit bare JSON tokens — ClickHouse
    would reject). Fixed by _sanitize_nan_inf preprocessing."""
    import math

    attrs = _record_attributes({"scores": {"a": math.nan, "b": math.inf, "c": -math.inf}})
    raw = attrs["neatlogs.workflow.scores"]
    # The raw string must contain quoted "NaN" / "Infinity" tokens,
    # never the bare unquoted form json.dumps would emit by default.
    assert '"NaN"' in raw
    assert '"Infinity"' in raw
    assert '"-Infinity"' in raw
    # Belt-and-suspenders: a strict parser would round-trip cleanly.
    parsed = json.loads(raw)
    assert parsed == {"a": "NaN", "b": "Infinity", "c": "-Infinity"}


def test_wrap_context_stringifies_nested_nan_in_list():
    """Same fix applies inside list values, not just dicts."""
    import math

    attrs = _record_attributes({"xs": [math.nan, 1.0, math.inf]})
    raw = attrs["neatlogs.workflow.xs"]
    assert '"NaN"' in raw
    assert '"Infinity"' in raw
    assert json.loads(raw) == ["NaN", 1.0, "Infinity"]


def test_wrap_context_handles_datetime_via_isoformat():
    """datetime values are formatted via .isoformat() (parity with
    neatlogs._annotations._coerce)."""
    from datetime import datetime, timezone

    attrs = _record_attributes({"at": datetime(2026, 8, 15, 2, 11, tzinfo=timezone.utc)})
    assert attrs["neatlogs.workflow.at"] == "2026-08-15T02:11:00+00:00"


def test_wrap_context_handles_pydantic_like_via_model_dump():
    """Pydantic-v2-style objects (any .model_dump()) are JSON-serialized
    via the dump, not via repr. Parity with _annotations._coerce."""

    class _PydanticLike:
        def model_dump(self):
            return {"k": 1, "n": "two"}

    attrs = _record_attributes({"blob": _PydanticLike()})
    assert json.loads(attrs["neatlogs.workflow.blob"]) == {"k": 1, "n": "two"}


def test_wrap_context_handles_unknown_object_via_str_fallback():
    """A plain object with no isoformat / model_dump / JSON-serializable
    shape falls back to str(). Documented behavior."""

    class _Unknown:
        def __repr__(self):
            return "<Unknown 0xDEAD>"

    attrs = _record_attributes({"obj": _Unknown()})
    assert attrs["neatlogs.workflow.obj"] == "<Unknown 0xDEAD>"


def test_wrap_context_skips_none_values():
    """None is not crashed on: even though _filtered_mapping filters None
    upstream, a directly-set None via _wrap_context must not raise. The
    value lands as the str() fallback ("None"), which is acceptable —
    the upstream filter is the real defense against None in the public
    API."""
    attrs = _record_attributes({"a": 1, "b": None, "c": "x"})
    assert attrs.get("neatlogs.workflow.a") == 1
    # The string "None" is fine — see comment above.
    assert attrs.get("neatlogs.workflow.b") == "None"
    assert attrs.get("neatlogs.workflow.c") == "x"
