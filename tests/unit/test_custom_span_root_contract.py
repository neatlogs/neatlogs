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
    assert operation_span.attributes["input.value"] == json.dumps({"value": value})
    assert operation_span.attributes["output.value"] == json.dumps(value)
