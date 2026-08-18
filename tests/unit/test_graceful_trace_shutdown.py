from opentelemetry import trace as otel_trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from neatlogs._wrap_utils import set_neatlogs_provider
from neatlogs.core.span_processor import NeatlogsSpanProcessor


def test_end_active_spans_closes_children_then_root_and_emits_completion_marker():
    provider = TracerProvider()
    lifecycle = NeatlogsSpanProcessor()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(lifecycle)
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    set_neatlogs_provider(provider)

    try:
        tracer = provider.get_tracer("neatlogs.test")
        root = tracer.start_span("workflow")
        root_context = otel_trace.set_span_in_context(root)
        child = tracer.start_span("agent", context=root_context)

        assert lifecycle.end_active_spans("SIGTERM") == 2

        spans = exporter.get_finished_spans()
        names = [span.name for span in spans]
        assert names.index("agent") < names.index("workflow")
        assert "neatlogs.trace.complete" in names

        finished_root = next(span for span in spans if span.name == "workflow")
        finished_child = next(span for span in spans if span.name == "agent")
        assert finished_root.parent is None
        assert finished_child.parent.span_id == finished_root.context.span_id
        assert finished_root.status.status_code is StatusCode.ERROR
        assert finished_root.attributes["neatlogs.trace.interrupted"] is True
        assert finished_root.attributes["neatlogs.trace.termination.reason"] == "SIGTERM"
    finally:
        lifecycle.end_active_spans("test-cleanup")
        provider.shutdown()
        set_neatlogs_provider(None)
