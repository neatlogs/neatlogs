import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import requests
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanKind

import neatlogs
from neatlogs.core.filtering_exporter import HttpFilteringSpanExporter
from neatlogs.core.masking_exporter import MaskingSpanExporter
from neatlogs.core.span_processor import (
    CompletionMarkerSpanProcessor,
    NeatlogsSpanProcessor,
    is_http_span,
)
from neatlogs.instrumentation.manager import InstrumentationManager


class _SnapshotExporter(SpanExporter):
    def __init__(self):
        self.spans = []

    def export(self, spans):
        self.spans.extend(
            {"name": span.name, "attributes": dict(span.attributes or {})} for span in spans
        )
        return SpanExportResult.SUCCESS


class _FailureExporter(SpanExporter):
    def export(self, spans):
        return SpanExportResult.FAILURE


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        body = b"ok"
        self.send_response(200)
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture
def local_http_url():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/health"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_sdk_processor_does_not_backfill_from_nested_http_span():
    provider = TracerProvider()
    exporter = _SnapshotExporter()
    neatlogs.init(
        api_key="unused",
        disable_export=True,
        instrumentations=[],
        tracer_provider=provider,
        workflow_name="http-export-privacy",
    )
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    tracer = provider.get_tracer("app.http-export-privacy")
    with tracer.start_as_current_span("workflow"):
        with tracer.start_as_current_span(
            "GET",
            kind=SpanKind.CLIENT,
            attributes={
                "http.request.method": "GET",
                "url.full": "https://user:pass@example.com/health?token=secret#fragment",
                "url.query": "token=secret",
                "http.request.header.authorization": "Bearer secret",
                "http.request.body": "secret-body",
                "input.value": "secret-input",
                "output.value": "secret-output",
            },
        ):
            pass

    root = next(span for span in exporter.spans if span["name"] == "workflow")
    assert "input.value" not in root["attributes"]
    assert "output.value" not in root["attributes"]


def test_http_payloads_cannot_leak_through_workflow_root_backfill():
    provider = TracerProvider()
    exporter = _SnapshotExporter()
    processor = NeatlogsSpanProcessor(own_all_spans=True)
    provider.add_span_processor(processor)
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    tracer = provider.get_tracer("app.http-root-privacy")
    with tracer.start_as_current_span("workflow"):
        with tracer.start_as_current_span(
            "GET",
            kind=SpanKind.CLIENT,
            attributes={
                "http.request.method": "GET",
                "url.full": "https://user:pass@example.com/health?token=secret#fragment",
                "input.value": "secret-input",
                "output.value": "secret-output",
            },
        ):
            pass

    root = next(span for span in exporter.spans if span["name"] == "workflow")
    assert "input.value" not in root["attributes"]
    assert "output.value" not in root["attributes"]
    serialized = json.dumps(root, default=str)
    for secret in ("user:pass", "token=secret", "fragment", "secret-input", "secret-output"):
        assert secret not in serialized


def test_raw_http_debug_log_is_sanitized(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NEATLOGS_LOG_RAW_SPANS", "true")
    provider = TracerProvider()
    processor = NeatlogsSpanProcessor(own_all_spans=True)
    provider.add_span_processor(processor)

    tracer = provider.get_tracer("app.http-raw-log-privacy")
    with tracer.start_as_current_span(
        "GET",
        kind=SpanKind.CLIENT,
        attributes={
            "http.request.method": "GET",
            "url.full": "https://user:pass@example.com/health?token=secret#fragment",
            "http.request.header.authorization": "Bearer secret",
            "http.request.body": "secret-body",
            "input.value": "secret-input",
            "output.value": "secret-output",
        },
    ):
        pass

    raw_log = (tmp_path / "spans_raw_optimized.log").read_text(encoding="utf-8")
    assert raw_log == ""
    for secret in (
        "user:pass",
        "token=secret",
        "fragment",
        "Bearer secret",
        "secret-body",
        "secret-input",
        "secret-output",
    ):
        assert secret not in raw_log

    provider.shutdown()


def test_empty_instrumentation_list_emits_no_client_span(local_http_url):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    manager = InstrumentationManager(provider)

    try:
        manager.instrument(libraries=[])
        with provider.get_tracer("neatlogs.http-test").start_as_current_span("workflow"):
            assert requests.get(local_http_url, timeout=2).text == "ok"

        assert [span.name for span in exporter.get_finished_spans()] == ["workflow"]
    finally:
        manager.uninstrument_all()
        provider.shutdown()


def test_canonical_semantic_kind_wins_over_http_metadata():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(NeatlogsSpanProcessor(own_all_spans=True))
    provider.add_span_processor(
        SimpleSpanProcessor(
            MaskingSpanExporter(
                HttpFilteringSpanExporter(exporter),
                lambda snapshot: snapshot,
            )
        )
    )
    span = provider.get_tracer("opentelemetry.instrumentation.httpx").start_span(
        "rerank",
        kind=SpanKind.CLIENT,
        attributes={
            "neatlogs.span.kind": "RERANKER",
            "openinference.span.kind": "HTTP",
            "url.full": "https://provider.example/rerank",
        },
    )
    span.end()

    finished = exporter.get_finished_spans()
    assert len(finished) == 1
    assert not is_http_span(finished[0])
    assert finished[0].attributes["neatlogs.span.kind"] == "reranker"
    provider.shutdown()


def test_post_mask_http_root_does_not_emit_a_completion_marker():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()

    def reclassify_root(snapshot):
        if snapshot["name"] == "workflow":
            snapshot["attributes"]["neatlogs.span.kind"] = "HTTP"
        return snapshot

    lifecycle = NeatlogsSpanProcessor(mask=reclassify_root, own_all_spans=True)
    completion = CompletionMarkerSpanProcessor(lifecycle, provider.get_tracer("neatlogs.internal"))
    final_filter = HttpFilteringSpanExporter(
        exporter,
        on_accepted=completion.accept_exported_root,
        on_rejected=completion.reject_exported_root,
    )
    provider.add_span_processor(lifecycle)
    provider.add_span_processor(
        SimpleSpanProcessor(
            MaskingSpanExporter(
                final_filter,
                reclassify_root,
                on_dropped=completion.reject_exported_root,
            )
        )
    )
    provider.add_span_processor(completion)

    tracer = provider.get_tracer("neatlogs.test")
    with tracer.start_as_current_span("workflow", attributes={"neatlogs.span.kind": "WORKFLOW"}):
        tracer.start_span("tool", attributes={"neatlogs.span.kind": "TOOL"}).end()

    provider.force_flush()
    assert [span.name for span in exporter.get_finished_spans()] == ["tool"]
    provider.shutdown()


def test_failed_root_export_does_not_emit_a_completion_marker():
    marker_exporter = InMemorySpanExporter()
    provider = TracerProvider()
    lifecycle = NeatlogsSpanProcessor(own_all_spans=True)
    completion = CompletionMarkerSpanProcessor(lifecycle, provider.get_tracer("neatlogs.internal"))
    provider.add_span_processor(lifecycle)
    provider.add_span_processor(
        SimpleSpanProcessor(
            HttpFilteringSpanExporter(
                _FailureExporter(),
                on_accepted=completion.accept_exported_root,
                on_rejected=completion.reject_exported_root,
            )
        )
    )
    provider.add_span_processor(SimpleSpanProcessor(marker_exporter))
    provider.add_span_processor(completion)

    provider.get_tracer("neatlogs.test").start_span(
        "workflow", attributes={"neatlogs.span.kind": "WORKFLOW"}
    ).end()

    provider.force_flush()
    assert all(
        span.name != "neatlogs.trace.complete" for span in marker_exporter.get_finished_spans()
    )
    provider.shutdown()
