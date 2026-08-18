import importlib
import signal

import pytest
from opentelemetry import trace as otel_trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from neatlogs._wrap_utils import set_neatlogs_provider
from neatlogs.core.span_processor import NeatlogsSpanProcessor

init_module = importlib.import_module("neatlogs.init")


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
        root.set_status(StatusCode.OK)
        root_context = otel_trace.set_span_in_context(root)
        child = provider.get_tracer("openinference.test").start_span("agent", context=root_context)

        assert lifecycle.end_active_spans("SIGTERM") == 2
        assert lifecycle.end_active_spans("SIGTERM") == 0

        spans = exporter.get_finished_spans()
        names = [span.name for span in spans]
        assert names.index("agent") < names.index("workflow")
        assert "neatlogs.trace.complete" in names

        finished_root = next(span for span in spans if span.name == "workflow")
        finished_child = next(span for span in spans if span.name == "agent")
        assert finished_root.parent is None
        assert finished_child.parent.span_id == finished_root.context.span_id
        assert finished_root.status.status_code is StatusCode.OK
        assert finished_child.status.status_code is StatusCode.UNSET
        assert finished_root.attributes["neatlogs.trace.interrupted"] is True
        assert finished_root.attributes["neatlogs.trace.termination.reason"] == "SIGTERM"
        assert finished_child.attributes["neatlogs.trace.interrupted"] is True
        assert finished_child.attributes["neatlogs.trace.termination.reason"] == "SIGTERM"
        assert finished_root.events == ()
        assert finished_child.events == ()
    finally:
        lifecycle.end_active_spans("test-cleanup")
        provider.shutdown()
        set_neatlogs_provider(None)


@pytest.mark.parametrize(
    ("signum", "reason", "exception", "exit_code"),
    [
        (signal.SIGINT, "SIGINT", KeyboardInterrupt, None),
        (signal.SIGTERM, "SIGTERM", SystemExit, 128 + signal.SIGTERM),
    ],
)
def test_shutdown_signal_handlers_forward_reason_and_terminate(
    monkeypatch, signum, reason, exception, exit_code
):
    shutdown_reasons = []
    previous_calls = []

    def previous_handler(previous_signum, frame):
        previous_calls.extend([previous_signum, frame])

    monkeypatch.setattr(
        init_module,
        "shutdown",
        lambda **kwargs: shutdown_reasons.append(kwargs["termination_reason"]),
    )
    monkeypatch.setattr(init_module, "_signal_handlers", {signum: previous_handler})
    monkeypatch.setattr(init_module, "_signal_shutdown_in_progress", False)

    with pytest.raises(exception) as raised:
        init_module._shutdown_signal_handler(signum, None)

    assert shutdown_reasons == [reason]
    assert previous_calls == [signum, None]
    if exit_code is not None:
        assert raised.value.code == exit_code
