import importlib
import json
import sys
import types
from datetime import datetime, timezone

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.trace import StatusCode

import neatlogs


def _install_test_tracer(in_memory_span_exporter) -> None:
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(in_memory_span_exporter))
    trace.set_tracer_provider(provider)
    neatlogs.init(api_key="test-key", disable_export=True, instrumentations=[])


def _crewai_module():
    return importlib.import_module("neatlogs.crewai")


def _finished_span(in_memory_span_exporter, name: str):
    return next(span for span in in_memory_span_exporter.get_finished_spans() if span.name == name)


def test_base_tool_preserves_long_structured_input_output_and_return_value(
    in_memory_span_exporter,
) -> None:
    _install_test_tracer(in_memory_span_exporter)
    crewai_instrumentation = _crewai_module()
    input_tail = "TOOL_INPUT_TAIL_AFTER_10000"
    output_tail = "TOOL_OUTPUT_TAIL_AFTER_10000"
    received = {}
    result = {"items": ["o" * 120_000, output_tail]}

    class FakeTool:
        name = "long_tool"
        description = "Returns a long structured result"

        def run(self, *args, **kwargs):
            received["args"] = args
            received["kwargs"] = kwargs
            return result

    crewai_instrumentation._patch_tool_run(FakeTool)
    argument = {"query": f"{'i' * 120_000}{input_tail}"}
    returned = FakeTool().run(payload=argument)

    assert returned is result
    assert received == {"args": (), "kwargs": {"payload": argument}}

    tool_span = _finished_span(in_memory_span_exporter, "crewai.tool.long_tool")
    captured_input = tool_span.attributes["input.value"]
    captured_output = tool_span.attributes["output.value"]
    assert input_tail in captured_input
    assert output_tail in captured_output
    assert len(captured_input) > 100_000
    assert len(captured_output) > 100_000
    assert json.loads(captured_input) == {"payload": argument}
    assert json.loads(captured_output) == result
    assert tool_span.attributes["neatlogs.framework"] == "crewai"
    assert tool_span.attributes["neatlogs.span.kind"] == "tool"


def test_base_tool_preserves_plain_string_output_without_json_quotes(
    in_memory_span_exporter,
) -> None:
    _install_test_tracer(in_memory_span_exporter)
    crewai_instrumentation = _crewai_module()

    class PlainStringTool:
        name = "plain_string_tool"

        def run(self):
            return "plain tool result"

    crewai_instrumentation._patch_tool_run(PlainStringTool)
    returned = PlainStringTool().run()

    assert returned == "plain tool result"
    tool_span = _finished_span(
        in_memory_span_exporter, "crewai.tool.plain_string_tool"
    )
    assert tool_span.attributes["output.value"] == "plain tool result"


def test_structured_tool_matches_base_tool_fidelity(in_memory_span_exporter, monkeypatch) -> None:
    _install_test_tracer(in_memory_span_exporter)
    crewai_instrumentation = _crewai_module()
    input_tail = "STRUCTURED_INPUT_TAIL_AFTER_10000"
    output_tail = "STRUCTURED_OUTPUT_TAIL_AFTER_10000"
    received = {}
    result = {"content": f"{'r' * 12_000}{output_tail}"}

    class FakeStructuredTool:
        name = "structured_long_tool"
        description = "Returns structured output"

        def invoke(self, *args, **kwargs):
            received["args"] = args
            received["kwargs"] = kwargs
            return result

    crewai_module = types.ModuleType("crewai")
    tools_module = types.ModuleType("crewai.tools")
    structured_module = types.ModuleType("crewai.tools.structured_tool")
    structured_module.CrewStructuredTool = FakeStructuredTool
    tools_module.structured_tool = structured_module
    crewai_module.tools = tools_module
    monkeypatch.setitem(sys.modules, "crewai", crewai_module)
    monkeypatch.setitem(sys.modules, "crewai.tools", tools_module)
    monkeypatch.setitem(sys.modules, "crewai.tools.structured_tool", structured_module)

    crewai_instrumentation._patch_structured_tool()
    payload = {"query": f"{'q' * 12_000}{input_tail}"}
    returned = FakeStructuredTool().invoke(payload)

    assert returned is result
    assert received == {"args": (payload,), "kwargs": {}}
    tool_span = _finished_span(in_memory_span_exporter, "crewai.tool.structured_long_tool")
    assert input_tail in tool_span.attributes["input.value"]
    assert output_tail in tool_span.attributes["output.value"]
    assert json.loads(tool_span.attributes["input.value"]) == payload
    assert json.loads(tool_span.attributes["output.value"]) == result
    assert tool_span.attributes["neatlogs.framework"] == "crewai"


@pytest.mark.parametrize("result", [[], {}, ""])
def test_base_tool_captures_empty_non_none_results(in_memory_span_exporter, result) -> None:
    _install_test_tracer(in_memory_span_exporter)
    crewai_instrumentation = _crewai_module()

    class EmptyResultTool:
        name = f"empty_{type(result).__name__}"

        def run(self):
            return result

    crewai_instrumentation._patch_tool_run(EmptyResultTool)
    returned = EmptyResultTool().run()

    assert returned is result
    tool_span = _finished_span(in_memory_span_exporter, f"crewai.tool.{EmptyResultTool.name}")
    assert "output.value" in tool_span.attributes
    expected = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
    assert tool_span.attributes["output.value"] == expected


def test_base_tool_serializes_non_json_native_values_without_changing_return(
    in_memory_span_exporter,
) -> None:
    _install_test_tracer(in_memory_span_exporter)
    crewai_instrumentation = _crewai_module()
    timestamp = datetime(2026, 8, 6, 12, 30, tzinfo=timezone.utc)
    result = {"created_at": timestamp}

    class DatetimeTool:
        name = "datetime_tool"

        def run(self):
            return result

    crewai_instrumentation._patch_tool_run(DatetimeTool)
    returned = DatetimeTool().run()

    assert returned is result
    tool_span = _finished_span(in_memory_span_exporter, "crewai.tool.datetime_tool")
    assert json.loads(tool_span.attributes["output.value"]) == {"created_at": str(timestamp)}


def test_base_tool_serialization_failure_does_not_replace_business_result(
    in_memory_span_exporter,
) -> None:
    _install_test_tracer(in_memory_span_exporter)
    crewai_instrumentation = _crewai_module()

    class BrokenString:
        def __str__(self):
            raise RuntimeError("telemetry serialization failed")

    result = BrokenString()

    class UnserializableTool:
        name = "unserializable_tool"

        def run(self):
            return result

    crewai_instrumentation._patch_tool_run(UnserializableTool)
    returned = UnserializableTool().run()

    assert returned is result
    tool_span = _finished_span(in_memory_span_exporter, "crewai.tool.unserializable_tool")
    assert tool_span.status.status_code == StatusCode.OK


def test_base_tool_preserves_exception_and_records_error_span(
    in_memory_span_exporter,
) -> None:
    _install_test_tracer(in_memory_span_exporter)
    crewai_instrumentation = _crewai_module()

    class FailingTool:
        name = "failing_tool"

        def run(self):
            raise RuntimeError("tool failed")

    crewai_instrumentation._patch_tool_run(FailingTool)

    with pytest.raises(RuntimeError, match="tool failed"):
        FailingTool().run()

    tool_span = _finished_span(in_memory_span_exporter, "crewai.tool.failing_tool")
    assert tool_span.status.status_code == StatusCode.ERROR
    assert tool_span.attributes["neatlogs.framework"] == "crewai"


def test_representative_crewai_span_kinds_include_framework_identity(
    in_memory_span_exporter,
) -> None:
    _install_test_tracer(in_memory_span_exporter)
    crewai_instrumentation = _crewai_module()

    class Result:
        raw = "ok"

    class FakeCrew:
        tasks = []
        agents = []

        def kickoff(self):
            return Result()

    class FakeFlow:
        state = {"ready": True}

        def kickoff(self):
            return {"done": True}

    class FakeTask:
        description = "task"

        def _execute_core(self):
            return Result()

    class FakeAgent:
        role = "researcher"
        tools = []

        def execute_task(self):
            return "agent result"

    class FakeTool:
        name = "identity_tool"

        def run(self):
            return "tool result"

    class FakeLlm:
        model = "test-model"

        def call(self, messages, *args, **kwargs):
            return "llm result"

    crewai_instrumentation._patch_crew_class(FakeCrew)
    crewai_instrumentation._patch_flow_class(FakeFlow)
    task = FakeTask()
    crewai_instrumentation._patch_task_execute(task)
    agent = FakeAgent()
    crewai_instrumentation._patch_agent_execute(agent)
    crewai_instrumentation._patch_tool_run(FakeTool)
    crewai_instrumentation._patch_llm_call(FakeLlm)

    FakeCrew().kickoff()
    FakeFlow().kickoff()
    task._execute_core()
    agent.execute_task()
    FakeTool().run()
    FakeLlm().call([{"role": "user", "content": "hello"}])

    expected_kinds = {
        "crewai.crew.kickoff": "workflow",
        "crewai.flow.kickoff": "workflow",
        "crewai.task": "task",
        "crewai.agent.researcher": "agent",
        "crewai.tool.identity_tool": "tool",
        "crewai.llm.call": "llm",
    }
    spans = {span.name: span for span in in_memory_span_exporter.get_finished_spans()}
    for span_name, kind in expected_kinds.items():
        span = spans[span_name]
        assert span.attributes["neatlogs.framework"] == "crewai"
        assert span.attributes["neatlogs.span.kind"] == kind
