import contextlib
import json

import pytest

import neatlogs
from neatlogs._wrap_utils import set_neatlogs_provider


@pytest.mark.parametrize("value", [False, 0])
@pytest.mark.parametrize("existing_workflow", [False, True])
def test_tool_capture_keeps_falsy_values_under_one_workflow(
    monkeypatch, tracer_provider, in_memory_span_exporter, value, existing_workflow
):
    monkeypatch.setenv("NEATLOGS_AUTO_ROOT", "true")
    set_neatlogs_provider(tracer_provider)

    @neatlogs.span(kind="TOOL", name="operation")
    def operation(value):
        return value

    context = (
        neatlogs.trace("parent", kind="WORKFLOW") if existing_workflow else contextlib.nullcontext()
    )
    with context:
        assert operation(value) is value

    spans = in_memory_span_exporter.get_finished_spans()
    roots = [span for span in spans if span.parent is None]
    assert len(roots) == 1
    root_kind = roots[0].attributes.get("neatlogs.span.kind") or roots[0].attributes.get(
        "openinference.span.kind"
    )
    assert root_kind.lower() == "workflow"
    operation_span = next(span for span in spans if span.name == "operation")
    assert operation_span.parent.span_id == roots[0].context.span_id
    assert operation_span.attributes["neatlogs.span.origin"] == "custom"
    assert operation_span.attributes["input.value"] == json.dumps({"value": value})
    assert operation_span.attributes["output.value"] == json.dumps(value)


@pytest.mark.parametrize("kind", ["CHAIN", "AGENT"])
@pytest.mark.parametrize("value", [True, False, 0, {"status": "ok"}])
def test_authored_roots_keep_kind_marker_and_io_with_child(
    tracer_provider, in_memory_span_exporter, kind, value
):
    set_neatlogs_provider(tracer_provider)

    @neatlogs.span(kind="TOOL", name="child")
    def child():
        return "done"

    @neatlogs.span(kind=kind, name="authored")
    def operation(value):
        child()
        return value

    assert operation(value) is value
    spans = {span.name: span for span in in_memory_span_exporter.get_finished_spans()}
    root = spans["authored"]
    assert root.parent is None
    assert root.attributes["neatlogs.span.kind"] == kind.lower()
    assert root.attributes["neatlogs.span.origin"] == "custom"
    assert root.attributes["input.value"] == json.dumps({"value": value})
    assert root.attributes["output.value"] == json.dumps(value)
    assert spans["child"].parent.span_id == root.context.span_id


@pytest.mark.parametrize("value", [True, False, 0, {"status": "ok"}])
def test_manual_trace_marks_authored_root_without_changing_kind(
    tracer_provider, in_memory_span_exporter, value
):
    set_neatlogs_provider(tracer_provider)
    with neatlogs.trace("decision", kind="CHAIN") as span:
        span.set_attribute("input.value", json.dumps({"value": value}))
        span.set_attribute("output.value", json.dumps(value))
    root = in_memory_span_exporter.get_finished_spans()[0]
    assert root.parent is None
    assert root.attributes["openinference.span.kind"] == "CHAIN"
    assert root.attributes["neatlogs.span.origin"] == "custom"
    assert root.attributes["input.value"] == json.dumps({"value": value})
    assert root.attributes["output.value"] == json.dumps(value)
