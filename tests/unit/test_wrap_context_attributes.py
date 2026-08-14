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
