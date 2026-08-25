import neatlogs
from neatlogs._wrap_utils import set_neatlogs_provider


def test_legacy_trace_content_environment_variable_is_ignored(
    monkeypatch, tracer_provider, in_memory_span_exporter
):
    monkeypatch.setenv("NEATLOGS_TRACE_CONTENT", "false")
    set_neatlogs_provider(tracer_provider)

    @neatlogs.span(kind="TOOL")
    def echo(value):
        return value

    assert echo("visible") == "visible"
    span = in_memory_span_exporter.get_finished_spans()[0]
    assert "visible" in span.attributes["input.value"]
    assert span.attributes["output.value"] == '"visible"'


def test_explicit_per_span_capture_controls_remain_supported(
    tracer_provider, in_memory_span_exporter
):
    set_neatlogs_provider(tracer_provider)

    @neatlogs.span(kind="TOOL", capture_input=False, capture_output=False)
    def echo(value):
        return value

    assert echo("private") == "private"
    span = in_memory_span_exporter.get_finished_spans()[0]
    assert "input.value" not in span.attributes
    assert "output.value" not in span.attributes
