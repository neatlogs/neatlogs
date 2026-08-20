import threading
import gzip
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import requests
from opentelemetry.exporter.otlp.proto.http import Compression
from opentelemetry.exporter.otlp.proto.http import trace_exporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExportResult

from neatlogs.core.transport import build_otlp_session


def test_python_transport_retries_429_post_without_overlapping_otel_5xx_policy():
    requests_seen = 0

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            nonlocal requests_seen
            requests_seen += 1
            self.rfile.read(int(self.headers.get("Content-Length", "0")))
            self.send_response(429 if requests_seen < 3 else 200)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        response = build_otlp_session().post(
            f"http://127.0.0.1:{server.server_port}/v1/traces",
            data=b"protobuf",
            timeout=1,
        )
    finally:
        server.shutdown()
        server.server_close()

    assert response.status_code == 200
    assert requests_seen == 3


def test_upstream_otel_retries_503_and_gzip_is_receiver_compatible(monkeypatch):
    requests_seen = 0
    decoded = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            nonlocal requests_seen
            requests_seen += 1
            body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            assert self.headers.get("Content-Encoding") == "gzip"
            decoded.append(gzip.decompress(body))
            self.send_response(503 if requests_seen == 1 else 200)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *_args):
            pass

    monkeypatch.setattr(trace_exporter.random, "uniform", lambda *_args: 0)
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        exporter = OTLPSpanExporter(
            endpoint=f"http://127.0.0.1:{server.server_port}/v1/traces",
            compression=Compression.Gzip,
            session=build_otlp_session(),
        )
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        provider.get_tracer("retry-test").start_span("span").end()
        provider.shutdown()
    finally:
        server.shutdown()
        server.server_close()

    assert requests_seen == 2
    assert all(decoded)


def test_transport_bounds_read_timeout_retries():
    requests_seen = 0

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            nonlocal requests_seen
            requests_seen += 1
            self.rfile.read(int(self.headers.get("Content-Length", "0")))
            time.sleep(0.05)
            try:
                self.send_response(200)
                self.send_header("Content-Length", "0")
                self.end_headers()
            except BrokenPipeError:
                pass

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        with pytest.raises(requests.exceptions.RequestException):
            build_otlp_session().post(
                f"http://127.0.0.1:{server.server_port}/v1/traces",
                data=b"protobuf",
                timeout=0.01,
            )
    finally:
        server.shutdown()
        server.server_close()
    assert requests_seen == 3


def test_upstream_exporter_rejects_work_after_shutdown():
    exporter = OTLPSpanExporter(
        endpoint="http://127.0.0.1:1/v1/traces",
        session=build_otlp_session(),
    )
    exporter.shutdown()
    assert exporter.export(()) is SpanExportResult.FAILURE
