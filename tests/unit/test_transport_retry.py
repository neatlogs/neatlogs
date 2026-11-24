import gzip
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import requests
from opentelemetry.exporter.otlp.proto.http import Compression
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
            self.send_header("Retry-After", "0")
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


def test_upstream_otel_retries_503_and_gzip_is_receiver_compatible():
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
    # The response is ambiguous after the body has been accepted, so neither
    # requests nor urllib3 may replay this POST.
    assert requests_seen == 1


def test_rate_limit_retry_after_cannot_escape_export_deadline():
    requests_seen = 0

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            nonlocal requests_seen
            requests_seen += 1
            self.rfile.read(int(self.headers.get("Content-Length", "0")))
            self.send_response(429)
            self.send_header("Retry-After", "30")
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    started = time.monotonic()
    try:
        response = build_otlp_session().post(
            f"http://127.0.0.1:{server.server_port}/v1/traces",
            data=b"protobuf",
            timeout=0.05,
        )
        elapsed = time.monotonic() - started
    finally:
        server.shutdown()
        server.server_close()

    assert response.status_code == 429
    assert requests_seen == 1
    assert elapsed < 0.5


def test_transport_uses_no_version_sensitive_urllib3_retry_configuration():
    session = build_otlp_session()
    assert session.get_adapter("https://").max_retries.total == 0


def test_upstream_exporter_rejects_work_after_shutdown():
    exporter = OTLPSpanExporter(
        endpoint="http://127.0.0.1:1/v1/traces",
        session=build_otlp_session(),
    )
    exporter.shutdown()
    assert exporter.export(()) is SpanExportResult.FAILURE


@pytest.mark.parametrize("receipt", ["rejected", "warning", "malformed"])
def test_upstream_exporter_requires_complete_trace_receipt(monkeypatch, receipt):
    from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceResponse

    body = ExportTraceServiceResponse()
    body.partial_success.rejected_spans = 1 if receipt == "rejected" else 0
    body.partial_success.error_message = "private receiver details"
    calls = []

    def request(self, method, url, **kwargs):
        calls.append(url)
        response = requests.Response()
        response.status_code = 200
        response._content = (
            b"invalid-protobuf" if receipt == "malformed" else body.SerializeToString()
        )
        return response

    monkeypatch.setattr(requests.Session, "request", request)
    exporter = OTLPSpanExporter(
        endpoint="https://ingest.example.com/v1/traces", session=build_otlp_session()
    )
    try:
        expected = SpanExportResult.SUCCESS if receipt == "warning" else SpanExportResult.FAILURE
        assert exporter.export(()) is expected
        assert len(calls) == 1
    finally:
        exporter.shutdown()


def _fake_response(status: int, *, retry_after: str | None = None) -> requests.Response:
    resp = requests.Response()
    resp.status_code = status
    resp._content = b""
    resp._content_consumed = True
    if retry_after is not None:
        resp.headers["Retry-After"] = retry_after
    return resp


def test_retry_after_exceeding_max_cap_skips_retry_immediately_instead_of_capping_early(
    monkeypatch,
):
    """When server sends Retry-After > _MAX_RETRY_AFTER_SECONDS (even inside a larger caller
    timeout), transport must skip the internal retry immediately rather than capping to 5s
    and retrying 25s early."""
    from neatlogs.core import transport

    sleeps: list[float] = []
    calls: list[int] = []

    def fake_request(self, method, url, **kwargs):
        calls.append(1)
        return _fake_response(429, retry_after="30")

    monkeypatch.setattr(requests.Session, "request", fake_request)
    monkeypatch.setattr(transport.time, "sleep", sleeps.append)

    session = build_otlp_session(circuit_breaker_threshold=1)
    try:
        resp = session.post("https://ingest.example.com/v1/traces", data=b"", timeout=15.0)
        assert resp.status_code == 429
        assert len(calls) == 1
        assert sleeps == []
        assert session.circuit_breaker_state == "open"
    finally:
        session.close()


def test_retry_after_minimum_is_never_shortened_by_negative_jitter(monkeypatch):
    """Server-provided Retry-After within budget must never be shortened by negative jitter."""
    from neatlogs.core import transport

    sleeps: list[float] = []
    responses = [_fake_response(429, retry_after="2"), _fake_response(200)]

    monkeypatch.setattr(requests.Session, "request", lambda self, m, u, **kw: responses.pop(0))
    monkeypatch.setattr(transport.random, "uniform", lambda a, b: a)  # force minimum jitter (0.8)
    monkeypatch.setattr(transport.time, "sleep", sleeps.append)

    session = build_otlp_session()
    try:
        resp = session.post("https://ingest.example.com/v1/traces", data=b"", timeout=10.0)
        assert resp.status_code == 200
        assert len(sleeps) == 1
        assert sleeps[0] >= 2.0
    finally:
        session.close()


def test_retry_after_http_date_honors_minimum_and_skips_when_over_cap(monkeypatch):
    """HTTP-date Retry-After obeys the exact same no-shortening and over-cap skip rules."""
    from datetime import datetime, timedelta, timezone
    from email.utils import format_datetime

    from neatlogs.core import transport

    # 1. HTTP-date 30s in future exceeds 5s cap -> skip internal retry immediately
    future_far = format_datetime(datetime.now(timezone.utc) + timedelta(seconds=30), usegmt=True)
    sleeps_far: list[float] = []
    calls_far: list[int] = []

    def fake_far(self, method, url, **kwargs):
        calls_far.append(1)
        return _fake_response(429, retry_after=future_far)

    monkeypatch.setattr(requests.Session, "request", fake_far)
    monkeypatch.setattr(transport.time, "sleep", sleeps_far.append)

    session = build_otlp_session()
    try:
        resp = session.post("https://ingest.example.com/v1/traces", data=b"", timeout=15.0)
        assert resp.status_code == 429
        assert len(calls_far) == 1
        assert sleeps_far == []
    finally:
        session.close()

    # 2. HTTP-date ~3s in future -> not shortened by negative jitter
    future_near = format_datetime(datetime.now(timezone.utc) + timedelta(seconds=3), usegmt=True)
    sleeps_near: list[float] = []
    script = [_fake_response(429, retry_after=future_near), _fake_response(200)]

    monkeypatch.setattr(requests.Session, "request", lambda self, m, u, **kw: script.pop(0))
    monkeypatch.setattr(transport.random, "uniform", lambda a, b: 0.5)
    monkeypatch.setattr(transport.time, "sleep", sleeps_near.append)

    session2 = build_otlp_session()
    try:
        resp2 = session2.post("https://ingest.example.com/v1/traces", data=b"", timeout=10.0)
        assert resp2.status_code == 200
        assert len(sleeps_near) == 1
        # HTTP-date has 1-second resolution; must NOT be multiplied by 0.5 jitter (~1.0s)
        assert sleeps_near[0] >= 1.9
    finally:
        session2.close()


def test_fallback_backoff_without_retry_after_still_applies_client_jitter(monkeypatch):
    """When server omits Retry-After, client-owned exponential backoff still applies jitter."""
    from neatlogs.core import transport

    sleeps: list[float] = []
    script = [_fake_response(429), _fake_response(200)]

    monkeypatch.setattr(requests.Session, "request", lambda self, m, u, **kw: script.pop(0))
    monkeypatch.setattr(transport.random, "uniform", lambda a, b: a)  # 0.8
    monkeypatch.setattr(transport.time, "sleep", sleeps.append)

    session = build_otlp_session()
    try:
        resp = session.post("https://ingest.example.com/v1/traces", data=b"", timeout=5.0)
        assert resp.status_code == 200
        assert len(sleeps) == 1
        assert sleeps[0] == pytest.approx(0.08)
    finally:
        session.close()
