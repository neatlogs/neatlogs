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


def test_method_capture_excludes_receiver_state(tracer_provider, in_memory_span_exporter):
    _install(tracer_provider)

    class Retriever:
        def __init__(self):
            self.api_key = "must-not-capture"
            self.client = self

        @span(kind="RETRIEVER", name="method-retrieve")
        def retrieve(self, query, namespace="faq"):
            return [query]

        @classmethod
        @span(kind="TOOL", name="class-tool")
        def tool(cls, query):
            return query

    retriever = Retriever()
    assert retriever.retrieve("delivery") == ["delivery"]
    assert retriever.tool("returns") == "returns"
    captured = _finished(in_memory_span_exporter, "method-retrieve")
    assert captured.attributes["input.value"] == '{"query": "delivery", "namespace": "faq"}'
    assert captured.attributes["neatlogs.retriever.query"] == "delivery"
    assert Retriever.retrieve(self=retriever, query="tracking") == ["tracking"]
    captured = [
        item
        for item in in_memory_span_exporter.get_finished_spans()
        if item.name == "method-retrieve"
    ][-1]
    assert captured.attributes["input.value"] == '{"query": "tracking", "namespace": "faq"}'
    assert (
        _finished(in_memory_span_exporter, "class-tool").attributes["input.value"]
        == '{"query": "returns"}'
    )


def test_capture_keeps_ordinary_parameters_named_self(tracer_provider, in_memory_span_exporter):
    _install(tracer_provider)

    @span(kind="TOOL", name="ordinary-self")
    def ordinary(self):
        return self

    class StaticTool:
        @staticmethod
        @span(kind="TOOL", name="static-self")
        def tool(self):
            return self

    assert ordinary("value") == "value"
    assert StaticTool.tool("value") == "value"
    for name in ("ordinary-self", "static-self"):
        assert (
            _finished(in_memory_span_exporter, name).attributes["input.value"]
            == '{"self": "value"}'
        )
