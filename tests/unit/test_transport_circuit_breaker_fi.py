"""Marked deterministic fault-injection tests (Pilot Acceptance Check #2).

Verifies fail-open circuit breaker on real loopback HTTP + OTLPSpanExporter,
fail-open checkpoint/diagnostic behavior under corrupted inputs, and init/Client parity.
"""

from __future__ import annotations

import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceResponse
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExportResult

import neatlogs
from neatlogs.core.backtracking import (
    CHECKPOINT_DELTA_TRUNCATION_MARKER,
    checkpoint,
    report_diagnostic,
)
from neatlogs.core.transport import CircuitOpenError, _OtlpSession
from neatlogs.errors import NeatlogsConfigurationError


class _FlakyOtlpServer:
    """Real loopback HTTP server injecting deterministic 503 faults then recovering to 200."""

    def __init__(self, fail_first: int) -> None:
        self.fail_first = fail_first
        self.requests_received = 0
        self._lock = threading.Lock()
        srv = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", "0"))
                self.rfile.read(length)
                with srv._lock:
                    srv.requests_received += 1
                    count = srv.requests_received
                if count <= srv.fail_first:
                    self.send_response(503)
                    self.end_headers()
                    return
                body = ExportTraceServiceResponse().SerializeToString()
                self.send_response(200)
                self.send_header("Content-Type", "application/x-protobuf")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format, *args):
                return

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}/v1/traces"
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=2)


@pytest.mark.fault_injection
def test_fi_circuit_breaker_fails_fast_without_otel_retry_storm_and_recovers():
    """Given a real loopback server returning 503 for 3 calls then 200,
    When the circuit breaker trips after 3 failures,
    Then subsequent OTLPSpanExporter.export() calls fail-fast with 0 server hits and no backoff sleep,
    and recover to CLOSED after cooldown."""
    server = _FlakyOtlpServer(fail_first=3)
    session = _OtlpSession(circuit_breaker_threshold=3, circuit_breaker_cooldown_s=0.15)
    try:
        # 3 consecutive 503 responses trip the breaker
        for _ in range(3):
            resp = session.post(server.url, data=b"", timeout=2.0)
            assert resp.status_code == 503
        assert server.requests_received == 3
        assert session.circuit_breaker_state == "open"

        # Direct call raises CircuitOpenError immediately without hitting server
        with pytest.raises(CircuitOpenError):
            session.post(server.url, data=b"", timeout=2.0)
        assert server.requests_received == 3

        # Real OTLPSpanExporter must abort immediately (no 6-retry OTel backoff sleep!)
        exporter = OTLPSpanExporter(endpoint=server.url, session=session, timeout=5.0)
        provider = TracerProvider()
        spans = []

        class _Collect(SimpleSpanProcessor):
            def on_end(self, span):
                spans.append(span)

        provider.add_span_processor(_Collect(exporter))
        with provider.get_tracer("fi-test").start_as_current_span("op"):
            pass

        started = time.perf_counter()
        result = exporter.export(spans)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        assert result == SpanExportResult.FAILURE
        assert server.requests_received == 3
        assert elapsed_ms < 100.0, f"Open circuit should fail-fast, took {elapsed_ms:.1f}ms"

        # After cooldown, half-open probe succeeds and closes circuit
        time.sleep(0.18)
        assert session.circuit_breaker_state == "half_open"
        recovered = exporter.export(spans)
        assert recovered == SpanExportResult.SUCCESS
        assert server.requests_received == 4
        assert session.circuit_breaker_state == "closed"
        provider.shutdown()
    finally:
        session.close()
        server.close()


@pytest.mark.fault_injection
def test_fi_checkpoint_and_diagnostic_fail_open_on_cyclic_and_oversized_inputs():
    """Given cyclic dicts, huge state_deltas, and unserializable objects,
    When checkpoint() and report_diagnostic() run inside an active span,
    Then no exception escapes and state_delta uses the safe non-replay-poisoning marker."""
    neatlogs.shutdown()
    neatlogs.init(
        api_key="fi-key",
        disable_export=True,
        instrumentations=[],
        register_shutdown_handlers=False,
    )
    try:
        cyclic: dict = {}
        cyclic["self"] = cyclic

        class _Exploding:
            def __repr__(self):
                raise RuntimeError("boom")

        with neatlogs.trace("fi-root", kind="WORKFLOW") as span:
            cid = checkpoint("cp-cyclic", state=cyclic, state_delta={"blob": "x" * 40_000})
            assert cid == "cp-cyclic"
            assert (
                report_diagnostic(
                    "broken_tool", "err api_key=sk-1234", details={"bad": _Exploding()}
                )
                is True
            )
            attrs = dict(span.attributes or {})
            assert attrs["neatlogs.checkpoint.id"] == "cp-cyclic"
            assert attrs["neatlogs.checkpoint.state_delta_truncated"] is True
            assert CHECKPOINT_DELTA_TRUNCATION_MARKER in attrs["neatlogs.checkpoint.state_delta"]
            assert "...[neatlogs-truncated" not in attrs["neatlogs.checkpoint.state_delta"]
            assert "sk-1234" not in attrs["neatlogs.diagnostic.message"]
    finally:
        neatlogs.shutdown()


@pytest.mark.fault_injection
def test_fi_init_signature_provenance_parity_and_no_ambient_git_sha(monkeypatch):
    """Given ambient GIT_SHA in environment, When init() runs without NEATLOGS_RELEASE,
    Then GIT_SHA is NOT leaked, and conflicting init(environment=...) raises NeatlogsConfigurationError.
    """
    monkeypatch.setenv("GIT_SHA", "should-not-leak-ambient-sha")
    monkeypatch.delenv("NEATLOGS_ENVIRONMENT", raising=False)
    monkeypatch.delenv("NEATLOGS_RELEASE", raising=False)

    neatlogs.shutdown()
    neatlogs.init(
        api_key="sig-key",
        disable_export=True,
        instrumentations=[],
        environment="staging",
        release="rel-1",
        register_shutdown_handlers=False,
    )
    try:
        # Same config is idempotent
        neatlogs.init(
            api_key="sig-key",
            disable_export=True,
            instrumentations=[],
            environment="staging",
            release="rel-1",
            register_shutdown_handlers=False,
        )
        # Different release must fail closed
        with pytest.raises(NeatlogsConfigurationError):
            neatlogs.init(
                api_key="sig-key",
                disable_export=True,
                instrumentations=[],
                environment="staging",
                release="rel-2",
                register_shutdown_handlers=False,
            )
    finally:
        neatlogs.shutdown()

    # Client parity without release does NOT read ambient GIT_SHA
    client = neatlogs.Client(
        api_key="client-key",
        workflow_name="client-wf",
        disable_export=True,
        environment="prod",
    )
    try:
        assert client._span_processor.environment == "prod"
        assert client._span_processor.release is None
    finally:
        client.shutdown()
