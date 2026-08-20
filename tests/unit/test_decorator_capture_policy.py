"""Decorator capture defaults to full fidelity with explicit local controls."""

from opentelemetry import trace

from neatlogs._wrap_utils import set_neatlogs_provider
from neatlogs.decorators.orchestration import span


def _install(tracer_provider):
    trace.set_tracer_provider(tracer_provider)
    set_neatlogs_provider(tracer_provider)


def _finished(in_memory_span_exporter, name):
    return next(item for item in in_memory_span_exporter.get_finished_spans() if item.name == name)


def test_default_capture_records_input_and_output(tracer_provider, in_memory_span_exporter):
    _install(tracer_provider)

    @span(kind="WORKFLOW")
    def answer(question: str):
        return {"answer": question.upper()}

    answer("hello")

    captured = _finished(in_memory_span_exporter, "answer")
    assert captured.attributes["input.value"] == '{"question": "hello"}'
    assert captured.attributes["output.value"] == '{"answer": "HELLO"}'


def test_explicit_capture_controls_remain_local_and_deterministic(
    tracer_provider, in_memory_span_exporter
):
    _install(tracer_provider)

    @span(kind="WORKFLOW", capture_input=False, capture_output=False)
    def answer(question: str):
        return question.upper()

    answer("hello")

    captured = _finished(in_memory_span_exporter, "answer")
    assert "input.value" not in captured.attributes
    assert "output.value" not in captured.attributes
