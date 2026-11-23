"""Regression unit tests for P1-2 (foreign-span isolation), P1-3 (retriever step replay),
and P1-4 (doctor --trace real API parsing, step_ordering_clean inversions, malformed spans)."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult

import neatlogs
from neatlogs.__main__ import main
from neatlogs.core.backtracking import checkpoint, report_diagnostic
from neatlogs.replay import ReplaySession, replay_recording_from_case
from neatlogs.replay.session import lookup_replay
from neatlogs.trace_audit import audit_trace, run_trace_audit


class _MemoryExporter(SpanExporter):
    def __init__(self) -> None:
        self.spans: list[Any] = []

    def export(self, spans):
        self.spans.extend(spans)
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        return None


# ---------------------------------------------------------------------------
# P1-2: Foreign-span isolation for checkpoint() and report_diagnostic()
# ---------------------------------------------------------------------------


def test_checkpoint_and_diagnostic_do_not_leak_onto_foreign_otel_span_when_uninitialized():
    """Plain OTel span outside Neatlogs must remain completely untouched."""
    neatlogs.shutdown()
    exporter = _MemoryExporter()
    foreign_provider = TracerProvider()
    foreign_provider.add_span_processor(SimpleSpanProcessor(exporter))
    foreign_tracer = foreign_provider.get_tracer("host.fastapi.app")

    try:
        with foreign_tracer.start_as_current_span("GET /orders"):
            assert checkpoint("cp-foreign", state={"secret": 1}, state_delta={"x": 2}) is None
            assert report_diagnostic("host_warn", "should not leak", details={"k": "v"}) is False
    finally:
        foreign_provider.shutdown()

    assert len(exporter.spans) == 1
    foreign_span = exporter.spans[0]
    attrs = dict(foreign_span.attributes or {})
    assert not any(k.startswith("neatlogs.") for k in attrs)
    assert list(foreign_span.events) == []


def test_checkpoint_and_diagnostic_do_not_leak_onto_foreign_span_in_isolated_mode():
    """Even after neatlogs.init() (isolated mode), a foreign OTel span with no active
    Neatlogs span must stay untouched; when nested inside neatlogs.trace(), attributes land
    on the Neatlogs span rather than the foreign span."""
    neatlogs.shutdown()
    neatlogs.init(
        api_key="isolation-key",
        disable_export=True,
        instrumentations=[],
        register_shutdown_handlers=False,
    )
    foreign_exporter = _MemoryExporter()
    foreign_provider = TracerProvider()
    foreign_provider.add_span_processor(SimpleSpanProcessor(foreign_exporter))
    foreign_tracer = foreign_provider.get_tracer("thirdparty.langfuse")

    try:
        # Outside any Neatlogs span -> no-op, foreign span untouched
        with foreign_tracer.start_as_current_span("foreign-root"):
            assert checkpoint("cp-outside", state={"a": 1}) is None
            assert report_diagnostic("cat", "msg") is False

        # Nested inside a Neatlogs span -> writes to Neatlogs span, NOT foreign span
        with neatlogs.trace("nl-root", kind="WORKFLOW") as nl_span:
            with foreign_tracer.start_as_current_span("foreign-child"):
                assert checkpoint("cp-inside", state={"a": 1}) == "cp-inside"
                assert report_diagnostic("cat", "inside-msg") is True
            nl_attrs = dict(nl_span.attributes or {})
            assert nl_attrs["neatlogs.checkpoint.id"] == "cp-inside"
            assert nl_attrs["neatlogs.diagnostic.category"] == "cat"
    finally:
        foreign_provider.shutdown()
        neatlogs.shutdown()

    assert len(foreign_exporter.spans) == 2
    for fspan in foreign_exporter.spans:
        fattrs = dict(fspan.attributes or {})
        assert not any(k.startswith("neatlogs.") for k in fattrs)
        assert list(fspan.events) == []


# ---------------------------------------------------------------------------
# P1-3: Retriever replay=True must NOT switch CLI replay to live mode
# ---------------------------------------------------------------------------


def test_retriever_replay_true_does_not_switch_downstream_tools_or_models_live():
    """In CLI replay (replay_recording_from_case without include_steps, or App-supplied
    recordings lacking 'step' entries), a RETRIEVER decorated with replay=True must run live
    for that call only without flipping switchedToLive=True for downstream tools/models."""
    live_calls = {"retriever": 0, "charge": 0, "model": 0}

    @neatlogs.span(kind="RETRIEVER", name="fetch_policy", replay=True)
    def fetch_policy(query: str) -> list[dict[str, Any]]:
        live_calls["retriever"] += 1
        return [{"id": "doc-1", "content": "30-day refunds"}]

    @neatlogs.span(kind="TOOL", tool_name="charge_card", replay=True)
    def charge_card(order_id: str, amount: int) -> dict[str, Any]:
        live_calls["charge"] += 1
        return {"charged": True, "order_id": order_id, "amount": amount}

    # Build a recording exactly like replay/cli.py and replay/all_command.py do
    # (replay_recording_from_case with default include_steps=False -> NO step entries!)
    frozen_case = {
        "source": {"kind": "trace", "traceId": "trace-cli-1"},
        "context": {
            "spans": [
                {
                    "span_id": "ret-1",
                    "span_type": "RETRIEVER",
                    "name": "fetch_policy",
                    "input_value": json.dumps({"query": "refund policy"}),
                    "output_value": json.dumps([{"id": "doc-1", "content": "30-day refunds"}]),
                    "attributes": {},
                },
                {
                    "span_id": "tool-1",
                    "span_type": "TOOL",
                    "tool_name": "charge_card",
                    "input_value": json.dumps({"order_id": "ORD-1", "amount": 50}),
                    "output_value": json.dumps({"charged": False, "replayed": True}),
                    "attributes": {},
                },
            ]
        },
    }
    converted = replay_recording_from_case(frozen_case)
    assert [e.kind for e in converted["recording"].entries] == ["tool"]

    # 1. tools_only mode: retriever runs live, downstream tool MUST still replay from cache
    with ReplaySession(converted["recording"], mode="tools_only") as session:
        docs = fetch_policy("refund policy")
        result = charge_card("ORD-1", 50)
        status = session.get_status()

    assert docs == [{"id": "doc-1", "content": "30-day refunds"}]
    assert result == {"charged": False, "replayed": True}
    assert live_calls == {"retriever": 1, "charge": 0, "model": 0}
    assert status["switchedToLive"] is False
    assert status["reason"] is None
    assert status["firstMiss"] is None
    assert status["hits"] == 1
    assert status["liveCalls"] == 1

    # 2. until_span mode: retriever miss must NOT switch downstream model/tool before checkpoint live
    recording_with_model = {
        "version": 1,
        "sourceTraceId": "trace-cli-2",
        "entries": [
            {
                "spanId": "m1",
                "kind": "model",
                "name": "gpt-4o",
                "input": {"prompt": "hi"},
                "output": {"text": "replayed-model"},
            },
            {
                "spanId": "t1",
                "kind": "tool",
                "name": "charge_card",
                "input": {"order_id": "ORD-2", "amount": 25},
                "output": {"charged": False, "replayed": True},
            },
            {
                "spanId": "cp1",
                "kind": "tool",
                "name": "send_email",
                "input": {"to": "a@b.com"},
                "output": {"sent": True},
            },
        ],
    }
    live_calls = {"retriever": 0, "charge": 0, "model": 0}
    with ReplaySession(recording_with_model, mode="until_span", until_span="cp1") as session:
        fetch_policy("refund policy")
        model_decision = lookup_replay("model", "gpt-4o", {"prompt": "hi"})
        tool_result = charge_card("ORD-2", 25)
        status = session.get_status()

    assert model_decision.outcome == "HIT"
    assert model_decision.entry.output == {"text": "replayed-model"}
    assert tool_result == {"charged": False, "replayed": True}
    assert live_calls["retriever"] == 1
    assert live_calls["charge"] == 0
    assert status["switchedToLive"] is False
    assert status["hits"] == 2


# ---------------------------------------------------------------------------
# P1-4: doctor --trace real backend APIs, step_ordering_clean, malformed spans
# ---------------------------------------------------------------------------


def test_step_ordering_clean_catches_timestamp_group_inversions_and_mixed_timestamps():
    """Sequence (t1:1, t1:3, t2:2) and (t1:3, t2:2) must FAIL step_ordering_clean."""
    base_trace = {"trace_id": "t-ord", "finalizationStatus": "finalized"}

    def _span(sid: str, parent: str | None, ts: Any, seq: int) -> dict[str, Any]:
        return {
            "span_id": sid,
            "parent_span_id": parent,
            "name": f"op-{sid}",
            "span_type": "WORKFLOW" if parent is None else "TOOL",
            "input": {"q": 1},
            "output": {"r": 2},
            "start_time_unix_nano": ts if isinstance(ts, int) else None,
            "startedAt": ts if isinstance(ts, str) else None,
            "attributes": {"neatlogs.step.seq": seq, "neatlogs.provenance.release": "v1"},
        }

    # Case 1: (t1:1, t1:3, t2:2) -> inversion across timestamp group must FAIL
    report = audit_trace(
        base_trace,
        [
            _span("s1", None, 1000, 1),
            _span("s2", "s1", 1000, 3),
            _span("s3", "s1", 2000, 2),
        ],
    )
    by_name = {c["name"]: c["status"] for c in report["checks"]}
    assert by_name["step_ordering_clean"] == "fail"

    # Case 2: (t1:3, t2:2) -> direct inversion must FAIL
    report2 = audit_trace(
        base_trace,
        [
            _span("s1", None, 1000, 3),
            _span("s2", "s1", 2000, 2),
        ],
    )
    by_name2 = {c["name"]: c["status"] for c in report2["checks"]}
    assert by_name2["step_ordering_clean"] == "fail"

    # Case 3: (t1:1, t1:2, t2:3) -> valid monotonic sequence passes (including mixed ISO/ns timestamps)
    report3 = audit_trace(
        base_trace,
        [
            _span("s1", None, "2026-11-20T00:00:01Z", 1),
            _span("s2", "s1", "2026-11-20T00:00:01Z", 2),
            _span("s3", "s1", "2026-11-20T00:00:02Z", 3),
        ],
    )
    by_name3 = {c["name"]: c["status"] for c in report3["checks"]}
    assert by_name3["step_ordering_clean"] == "pass"


def test_audit_trace_and_cli_handle_malformed_non_dict_spans_without_crashing(tmp_path: Path):
    """Non-dict entries in spans must not raise AttributeError; CLI --file must exit 5."""
    # Direct Python call: safe failure, no AttributeError
    report = audit_trace({"trace_id": "t-bad"}, ["not-a-dict", 123, None])  # type: ignore[list-item]
    by_name = {c["name"]: c["status"] for c in report["checks"]}
    assert by_name["has_spans"] == "fail"
    assert report["grade"] == "incomplete"

    # CLI --file with non-dict span -> exit 5 (EXIT_ERROR) with clean stderr message
    bad_file = tmp_path / "malformed.json"
    bad_file.write_text(
        json.dumps({"trace": {"trace_id": "t1"}, "spans": ["bad"]}), encoding="utf-8"
    )
    err_lines: list[str] = []
    code = run_trace_audit(None, file_path=str(bad_file), err=err_lines.append)
    assert code == 5
    assert any("JSON objects" in line for line in err_lines)


def test_doctor_trace_v3_large_payload_single_turn_session_and_endpoint_flag():
    """v3 route (> 1 MiB response, node_type='tool_call', sessionId==normalizedTraceId single-turn
    fallback, --endpoint override) must parse cleanly and grade 'ready'."""
    trace_id = "0123456789abcdef0123456789abcdef"
    huge_padding = "x" * (1_200_000)  # > 1 MiB!
    v3_payload = {
        "_id": trace_id,
        "status": "success",
        "finalizationStatus": "finalized",
        # Real v3 route falls back to normalizedTraceId when no sdkSessionId was set:
        "sessionId": trace_id,
        "spans": [
            {
                "span_id": "root-1",
                "node_name": "refund_workflow",
                "node_type": "workflow",
                "start_time": "2026-11-20T00:00:01Z",
                "data": {"input_value": '{"q":"help"}', "output_value": '{"a":"done"}'},
                "span_metadata": {
                    "neatlogs.step.seq": 1,
                    "neatlogs.provenance.release": "rel-1",
                    "padding": huge_padding,
                },
            },
            {
                "span_id": "tool-1",
                "parent_span_id": "root-1",
                "node_name": "issue_refund",
                "node_type": "tool_call",
                "start_time": "2026-11-20T00:00:02Z",
                "data": {"tool_input": '{"order":"1"}', "tool_output": '{"ok":true}'},
                "span_metadata": {"neatlogs.step.seq": 2},
            },
        ],
    }
    raw_bytes = json.dumps(v3_payload).encode("utf-8")
    assert len(raw_bytes) > (1 << 20)
    seen_paths: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            seen_paths.append(self.path)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw_bytes)))
            self.end_headers()
            self.wfile.write(raw_bytes)

        def log_message(self, format, *args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    endpoint = f"http://127.0.0.1:{server.server_port}"
    try:
        out: list[str] = []
        err: list[str] = []
        # Single-turn trace (expect_session=False): v3 sessionId==trace_id fallback must PASS
        code = run_trace_audit(
            trace_id,
            endpoint=endpoint,
            as_json=True,
            env={"NEATLOGS_API_KEY": "proj-key"},
            out=out.append,
            err=err.append,
        )
        assert code == 0, err
        report = json.loads(out[0])
        assert report["grade"] == "ready"
        assert report["score"] == 100
        assert seen_paths[-1] == f"/api/traces/v3/{trace_id}"

        # With expect_session=True, v3 fallback sessionId==trace_id must FAIL session_hygiene
        out2: list[str] = []
        code2 = run_trace_audit(
            trace_id,
            endpoint=endpoint,
            as_json=True,
            expect_session=True,
            env={"NEATLOGS_API_KEY": "proj-key"},
            out=out2.append,
            err=err.append,
        )
        assert code2 == 1
        by_name2 = {c["name"]: c["status"] for c in json.loads(out2[0])["checks"]}
        assert by_name2["session_hygiene"] == "fail"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_doctor_cli_forwards_endpoint_flag_and_handles_v3_202_and_409_dlq(monkeypatch, capsys):
    """`neatlogs doctor --trace <id> --endpoint <url>` must forward --endpoint to run_trace_audit;
    HTTP 202 returns exit 2 (not ready) and HTTP 409 DLQ returns exit 5 (terminal error)."""
    captured: list[dict[str, Any]] = []

    def fake_run_trace_audit(trace_id, **kwargs):
        captured.append({"trace_id": trace_id, **kwargs})
        return 0

    monkeypatch.setattr("neatlogs.trace_audit.run_trace_audit", fake_run_trace_audit)
    rc = main(
        ["doctor", "--trace", "abc", "--endpoint", "https://eu.ingest.neatlogs.com", "--json"]
    )
    assert rc == 0
    assert captured[0]["endpoint"] == "https://eu.ingest.neatlogs.com"
    assert captured[0]["trace_id"] == "abc"

    # Direct v3 status handling: 202 -> EXIT_NOT_READY (2), 409 dlq -> EXIT_ERROR (5)
    assert (
        run_trace_audit(
            "t-202",
            env={"NEATLOGS_API_KEY": "k"},
            fetch=lambda url, h, t: (202, b'{"status":"processing","spans":[]}'),
            err=lambda _: None,
        )
        == 2
    )
    assert (
        run_trace_audit(
            "t-dlq",
            env={"NEATLOGS_API_KEY": "k"},
            fetch=lambda url, h, t: (409, b'{"finalizationStatus":"dlq","error":"failed"}'),
            err=lambda _: None,
        )
        == 5
    )


def test_doctor_trace_public_api_metadata_only_warns_instead_of_false_payload_failures():
    """When only NEATLOGS_TOKEN + NEATLOGS_PROJECT_ID are set, public API spans are metadata-only
    (no input/output/attributes, timestamp is startedAt). Payload checks must WARN with guidance
    instead of falsely failing root_io_present / tool_linkage_valid / retriever_freshness."""
    trace_id = "pub-trace-1"
    public_trace = {
        "traceId": trace_id,
        "sessionId": "conv-1",
        "status": "success",
        "finalizationStatus": "finalized",
        "spansCount": 3,
    }
    public_spans = [
        {
            "spanId": "s1",
            "parentSpanId": None,
            "spanName": "refund_workflow",
            "spanType": "workflow",
            "startedAt": "2026-11-20T00:00:01Z",
            "status": "OK",
        },
        {
            "spanId": "s2",
            "parentSpanId": "s1",
            "spanName": "fetch_policy",
            "spanType": "retriever",
            "startedAt": "2026-11-20T00:00:02Z",
            "status": "OK",
        },
        {
            "spanId": "s3",
            "parentSpanId": "s1",
            "spanName": "issue_refund",
            "spanType": "tool",
            "startedAt": "2026-11-20T00:00:03Z",
            "status": "OK",
        },
    ]

    def public_fetch(url, headers, timeout):
        if "/spans" in url:
            body = json.dumps(
                {"success": True, "data": {"spans": public_spans, "page": {"nextCursor": None}}}
            ).encode()
        else:
            body = json.dumps({"success": True, "data": public_trace}).encode()
        return 200, body

    out: list[str] = []
    err: list[str] = []
    code = run_trace_audit(
        trace_id,
        as_json=True,
        expect_session=True,
        expect_retriever=True,
        env={"NEATLOGS_TOKEN": "tok-1", "NEATLOGS_PROJECT_ID": "proj-1"},
        fetch=public_fetch,
        out=out.append,
        err=err.append,
    )
    assert code == 1  # needs_attention (cannot confirm ready without payloads)
    report = json.loads(out[0])
    assert report["grade"] == "needs_attention"
    by_name = {c["name"]: c["status"] for c in report["checks"]}
    assert by_name["has_spans"] == "pass"
    assert by_name["single_workflow_root"] == "pass"
    assert by_name["trace_finalized"] == "pass"
    assert by_name["session_hygiene"] == "pass"
    assert by_name["root_io_present"] == "warn"
    assert by_name["tool_linkage_valid"] == "warn"
    assert by_name["retriever_freshness"] == "warn"
    assert by_name["step_ordering_clean"] == "warn"
    assert by_name["provenance_and_replayability"] == "warn"
    assert any("NEATLOGS_API_KEY" in rec for rec in report["recommendations"])

    # When BOTH NEATLOGS_API_KEY and NEATLOGS_TOKEN are set, prefer payload-capable /api/traces/v3
    called_urls: list[str] = []

    def dual_fetch(url, headers, timeout):
        called_urls.append(url)
        return (
            200,
            json.dumps(
                {
                    "_id": trace_id,
                    "status": "success",
                    "sessionId": "conv-1",
                    "spans": [
                        {
                            "span_id": "s1",
                            "node_name": "wf",
                            "node_type": "workflow",
                            "start_time": "2026-11-20T00:00:01Z",
                            "data": {"input_value": "hi", "output_value": "hello"},
                            "span_metadata": {
                                "neatlogs.step.seq": 1,
                                "neatlogs.provenance.release": "v1",
                            },
                        }
                    ],
                }
            ).encode(),
        )

    out2: list[str] = []
    code2 = run_trace_audit(
        trace_id,
        as_json=True,
        env={
            "NEATLOGS_API_KEY": "proj-key",
            "NEATLOGS_TOKEN": "tok-1",
            "NEATLOGS_PROJECT_ID": "proj-1",
        },
        fetch=dual_fetch,
        out=out2.append,
        err=err.append,
    )
    assert code2 == 0
    assert called_urls == [f"https://ingest.neatlogs.com/api/traces/v3/{trace_id}"]
    assert json.loads(out2[0])["grade"] == "ready"
