import importlib
import signal
import subprocess
import sys
import threading
import time

import pytest
from opentelemetry import trace as otel_trace
from opentelemetry.sdk.trace import SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from neatlogs._wrap_utils import set_neatlogs_provider
from neatlogs.client import Client
from neatlogs.core.deadline import DeadlineWorker, bounded_call
from neatlogs.core.masking_exporter import _MaskRunner
from neatlogs.core.span_processor import CompletionMarkerSpanProcessor, NeatlogsSpanProcessor

init_module = importlib.import_module("neatlogs.init")


@pytest.mark.parametrize("pipeline", ["default", "client"])
def test_python_312_atexit_shutdown_never_starts_a_thread(tmp_path, pipeline):
    marker = tmp_path / f"{pipeline}.shutdown"
    if pipeline == "default":
        setup = f"""
import importlib
import neatlogs
neatlogs.init(
    api_key="test-key",
    disable_export=True,
    instrumentations=[],
    register_shutdown_handlers=False,
)
module = importlib.import_module("neatlogs.init")
class Provider:
    def shutdown(self):
        open({str(marker)!r}, "w").write("closed")
module._tracer_provider = Provider()
module._owns_tracer_provider = True
module._log_provider = None
module._span_processor = None
module._completion_span_processor = None
module._instrumentation_manager = None
"""
    else:
        setup = f"""
from neatlogs.client import Client
client = Client(api_key="test-key", workflow_name="test", disable_export=True)
class Provider:
    def shutdown(self):
        open({str(marker)!r}, "w").write("closed")
client.tracer_provider = Provider()
"""

    script = setup + """
import threading
def forbidden(*args, **kwargs):
    raise RuntimeError("cannot create new thread at interpreter shutdown")
threading.Thread.start = forbidden
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert marker.read_text() == "closed"
    assert "cannot create new thread" not in completed.stderr


def test_prestarted_deadline_worker_returns_at_deadline_without_starting_a_thread(monkeypatch):
    worker = DeadlineWorker("neatlogs-test-shutdown")
    monkeypatch.setattr(
        threading.Thread,
        "start",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("late thread start")),
    )
    started = time.monotonic()

    completed, result = bounded_call(
        lambda: time.sleep(0.2),
        time.monotonic() + 0.02,
        synchronous=True,
        worker=worker,
    )

    elapsed = time.monotonic() - started
    assert completed is False
    assert isinstance(result, TimeoutError)
    assert elapsed < 0.1
    worker.close()


def test_mask_runner_uses_workers_started_before_atexit(monkeypatch):
    runner = _MaskRunner(timeout_seconds=0.1)
    monkeypatch.setattr(
        threading.Thread,
        "start",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("late thread start")),
    )

    assert runner.apply(lambda snapshot: snapshot, {"signal": "span"}) == {"signal": "span"}
    runner.shutdown()


def test_atexit_flush_with_mask_uses_only_prestarted_workers(tmp_path):
    marker = tmp_path / "masked.shutdown"
    script = f"""
import atexit
import threading
import time
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from neatlogs.core.deadline import DeadlineWorker, bounded_call
from neatlogs.core.masking_exporter import MaskingSpanExporter

inner = InMemorySpanExporter()
provider = TracerProvider()
provider.add_span_processor(BatchSpanProcessor(
    MaskingSpanExporter(inner, lambda snapshot: snapshot),
    schedule_delay_millis=60_000,
))
worker = DeadlineWorker("atexit-test")
provider.get_tracer("test").start_span("queued").end()
def cleanup():
    completed, _ = bounded_call(
        provider.shutdown,
        time.monotonic() + 1,
        synchronous=True,
        worker=worker,
    )
    open({str(marker)!r}, "w").write(f"{{completed}}:{{len(inner.get_finished_spans())}}")
atexit.register(cleanup)
def forbidden(*args, **kwargs):
    raise RuntimeError("cannot create new thread at interpreter shutdown")
threading.Thread.start = forbidden
"""

    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert marker.read_text() == "True:1"
    assert "cannot create new thread" not in completed.stderr


def test_concurrent_client_shutdown_wait_is_deadline_bounded():
    client = Client(api_key="test-key", workflow_name="bounded", disable_export=True)
    entered = threading.Event()
    release = threading.Event()

    class SlowProvider:
        def shutdown(self):
            entered.set()
            release.wait(1)

    client.tracer_provider = SlowProvider()
    owner = threading.Thread(target=lambda: client.shutdown(timeout_millis=500), daemon=True)
    owner.start()
    assert entered.wait(0.2)
    started = time.monotonic()

    assert client.shutdown(timeout_millis=20) is False
    assert time.monotonic() - started < 0.1

    release.set()
    owner.join(1)


def test_concurrent_default_shutdown_wait_is_deadline_bounded(monkeypatch):
    entered = threading.Event()
    release = threading.Event()

    class SlowProvider:
        def shutdown(self):
            entered.set()
            release.wait(1)

    monkeypatch.setattr(init_module, "_tracer_provider", SlowProvider())
    monkeypatch.setattr(init_module, "_owns_tracer_provider", True)
    monkeypatch.setattr(init_module, "_log_provider", None)
    monkeypatch.setattr(init_module, "_span_processor", None)
    monkeypatch.setattr(init_module, "_completion_span_processor", None)
    monkeypatch.setattr(init_module, "_instrumentation_manager", None)

    owner = threading.Thread(
        target=lambda: init_module.shutdown(timeout_millis=500),
        daemon=True,
    )
    owner.start()
    assert entered.wait(0.2)
    started = time.monotonic()

    assert init_module.shutdown(timeout_millis=20) is False
    assert time.monotonic() - started < 0.1

    release.set()
    owner.join(1)


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
        # OTel treats an explicit OK as terminal; preserve it, but convert the
        # usual open-span UNSET state to ERROR on interruption.
        assert finished_root.status.status_code is StatusCode.OK
        assert finished_child.status.status_code is StatusCode.ERROR
        assert finished_root.attributes["neatlogs.trace.interrupted"] is True
        assert finished_root.attributes["neatlogs.trace.termination.reason"] == "SIGTERM"
        assert finished_child.attributes["neatlogs.trace.interrupted"] is True
        assert finished_child.attributes["neatlogs.trace.termination.reason"] == "SIGTERM"
        assert finished_root.events[0].name == "neatlogs.trace.interrupted"
        assert finished_child.events[0].name == "neatlogs.trace.interrupted"
    finally:
        lifecycle.end_active_spans("test-cleanup")
        provider.shutdown()
        set_neatlogs_provider(None)


@pytest.mark.parametrize("signum,reason", [(signal.SIGINT, "SIGINT"), (signal.SIGTERM, "SIGTERM")])
def test_shutdown_signal_handler_returns_control_to_previous_callable(monkeypatch, signum, reason):
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

    init_module._shutdown_signal_handler(signum, None)

    assert shutdown_reasons == [reason]
    assert previous_calls == [signum, None]


@pytest.mark.parametrize(
    ("signum", "exception", "exit_code"),
    [
        (signal.SIGINT, KeyboardInterrupt, None),
        (signal.SIGTERM, SystemExit, 128 + signal.SIGTERM),
    ],
)
def test_shutdown_signal_handler_preserves_default_termination(
    monkeypatch, signum, exception, exit_code
):
    monkeypatch.setattr(init_module, "shutdown", lambda **_: True)
    monkeypatch.setattr(init_module, "_signal_handlers", {signum: signal.SIG_DFL})
    monkeypatch.setattr(init_module, "_signal_shutdown_in_progress", False)

    with pytest.raises(exception) as raised:
        init_module._shutdown_signal_handler(signum, None)

    if exit_code is not None:
        assert raised.value.code == exit_code


def test_shutdown_signal_handler_honors_ignored_signal(monkeypatch):
    calls = []
    monkeypatch.setattr(init_module, "shutdown", lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr(init_module, "_signal_handlers", {signal.SIGTERM: signal.SIG_IGN})
    monkeypatch.setattr(init_module, "_signal_shutdown_in_progress", False)

    init_module._shutdown_signal_handler(signal.SIGTERM, None)

    assert calls == []


def test_batch_processor_receives_root_before_completion_marker():
    provider = TracerProvider()
    lifecycle = NeatlogsSpanProcessor(emit_completion_markers=False)
    exporter = InMemorySpanExporter()
    provider.add_span_processor(lifecycle)
    provider.add_span_processor(
        BatchSpanProcessor(
            exporter,
            max_export_batch_size=1,
            schedule_delay_millis=60_000,
        )
    )
    provider.add_span_processor(
        CompletionMarkerSpanProcessor(
            lifecycle,
            provider.get_tracer("neatlogs.internal"),
        )
    )

    try:
        root = provider.get_tracer("neatlogs.test").start_span("workflow")
        root.end()
        provider.force_flush()
        names = [span.name for span in exporter.get_finished_spans()]
        assert names.index("workflow") < names.index("neatlogs.trace.complete")
    finally:
        provider.shutdown()


def test_cached_tracer_span_started_during_shutdown_is_closed_and_sanitized():
    provider = TracerProvider()
    lifecycle = NeatlogsSpanProcessor(emit_completion_markers=False)
    exporter = InMemorySpanExporter()
    provider.add_span_processor(lifecycle)
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("neatlogs.test")

    try:
        assert lifecycle.end_active_spans("SIGTERM\nforged=value") == 0
        late = tracer.start_span("late")
        assert not late.is_recording()
        finished = next(span for span in exporter.get_finished_spans() if span.name == "late")
        assert finished.status.status_code is StatusCode.ERROR
        assert finished.attributes["neatlogs.trace.termination.reason"] == "SIGTERM forged=value"
    finally:
        provider.shutdown()


def test_shutdown_is_same_thread_reentrant(monkeypatch):
    class ReentrantProcessor:
        def begin_shutdown(self, _reason):
            return None

        def end_active_spans(self, _reason):
            assert init_module.shutdown() is True
            return 0

        def _log_performance_stats(self):
            return None

        def wait_for_downstream(self, _timeout_millis):
            return True

    monkeypatch.setattr(init_module, "_span_processor", ReentrantProcessor())
    monkeypatch.setattr(init_module, "_tracer_provider", None)
    monkeypatch.setattr(init_module, "_log_provider", None)
    monkeypatch.setattr(init_module, "_instrumentation_manager", None)

    assert init_module.shutdown() is True


def test_shutdown_drains_logs_before_trace_completion(monkeypatch):
    order = []

    class Provider:
        def shutdown(self):
            order.append("traces")
            return True

    class Logs:
        def shutdown(self):
            order.append("logs")
            return True

    class Lifecycle:
        def begin_shutdown(self, _reason):
            order.append("fence")

        def end_active_spans(self, _reason):
            order.append("root-marker")
            return 1

        def _log_performance_stats(self):
            return None

        def wait_for_downstream(self, _timeout_millis):
            return True

    monkeypatch.setattr(init_module, "_tracer_provider", Provider())
    monkeypatch.setattr(init_module, "_owns_tracer_provider", True)
    monkeypatch.setattr(init_module, "_log_provider", Logs())
    monkeypatch.setattr(init_module, "_span_processor", Lifecycle())
    monkeypatch.setattr(init_module, "_completion_span_processor", None)
    monkeypatch.setattr(init_module, "_instrumentation_manager", None)

    assert init_module.shutdown() is True
    assert order == ["fence", "logs", "root-marker", "traces"]


def test_shutdown_deadline_detaches_a_stuck_exporter_generation(monkeypatch):
    release = threading.Event()

    class Provider:
        def shutdown(self):
            release.wait()

    monkeypatch.setattr(init_module, "_tracer_provider", Provider())
    monkeypatch.setattr(init_module, "_owns_tracer_provider", True)
    monkeypatch.setattr(init_module, "_log_provider", None)
    monkeypatch.setattr(init_module, "_span_processor", None)
    monkeypatch.setattr(init_module, "_completion_span_processor", None)
    monkeypatch.setattr(init_module, "_instrumentation_manager", None)

    assert init_module.shutdown(timeout_millis=20) is False
    assert init_module._tracer_provider is None
    release.set()


def test_completion_marker_is_deferred_until_requested():
    provider = TracerProvider()
    lifecycle = NeatlogsSpanProcessor(
        emit_completion_markers=False,
        own_all_spans=True,
    )
    exporter = InMemorySpanExporter()
    provider.add_span_processor(lifecycle)
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    completion = CompletionMarkerSpanProcessor(
        lifecycle,
        provider.get_tracer("neatlogs.internal"),
    )
    provider.add_span_processor(completion)

    try:
        completion.begin_shutdown()
        provider.get_tracer("custom.application").start_span("workflow").end()
        assert [span.name for span in exporter.get_finished_spans()] == ["workflow"]
        completion.emit_deferred()
        assert [span.name for span in exporter.get_finished_spans()] == [
            "workflow",
            "neatlogs.trace.complete",
        ]
    finally:
        provider.shutdown()


def test_disabled_export_does_not_accumulate_completion_eligibility():
    lifecycle = NeatlogsSpanProcessor(emit_completion_markers=False, own_all_spans=True)
    provider = TracerProvider()
    provider.add_span_processor(lifecycle)
    try:
        tracer = provider.get_tracer("neatlogs.test")
        for _ in range(100):
            tracer.start_span("root").end()
        assert lifecycle._completion_eligible_roots == set()
    finally:
        provider.shutdown()


def test_deferred_boundary_does_not_strand_late_roots():
    provider = TracerProvider()
    lifecycle = NeatlogsSpanProcessor(
        emit_completion_markers=False,
        own_all_spans=True,
    )
    exporter = InMemorySpanExporter()
    provider.add_span_processor(lifecycle)
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    completion = CompletionMarkerSpanProcessor(
        lifecycle,
        provider.get_tracer("neatlogs.internal"),
    )
    provider.add_span_processor(completion)

    try:
        completion.begin_shutdown()
        provider.get_tracer("custom").start_span("before-boundary").end()
        completion.emit_deferred()
        provider.get_tracer("custom").start_span("after-boundary").end()
        names = [span.name for span in exporter.get_finished_spans()]
        assert names.count("neatlogs.trace.complete") == 2
    finally:
        provider.shutdown()


def test_shutdown_waits_for_root_already_inside_downstream_processors():
    entered = threading.Event()
    release = threading.Event()

    class BlockingProcessor(SpanProcessor):
        def on_start(self, _span, parent_context=None):
            return None

        def on_end(self, span):
            if span.name == "workflow":
                entered.set()
                assert release.wait(2)

        def shutdown(self):
            return None

        def force_flush(self, timeout_millis=30000):
            return True

    provider = TracerProvider()
    lifecycle = NeatlogsSpanProcessor(
        emit_completion_markers=False,
        own_all_spans=True,
    )
    provider.add_span_processor(lifecycle)
    provider.add_span_processor(BlockingProcessor())
    completion = CompletionMarkerSpanProcessor(
        lifecycle,
        provider.get_tracer("neatlogs.internal"),
    )
    provider.add_span_processor(completion)
    root = provider.get_tracer("custom").start_span("workflow")
    ending = threading.Thread(target=root.end)
    ending.start()
    assert entered.wait(1)

    wait_result = []
    waiter = threading.Thread(
        target=lambda: wait_result.append(lifecycle.wait_for_downstream(2000))
    )
    waiter.start()
    waiter.join(0.05)
    assert waiter.is_alive()

    release.set()
    ending.join(1)
    waiter.join(1)
    assert wait_result == [True]
    provider.shutdown()
