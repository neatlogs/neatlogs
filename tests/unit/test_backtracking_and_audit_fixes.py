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
    assert report["replay_ready"] is False
    assert report["jev_ready"] is False
    assert report["replay_readiness_verified"] is False
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


# ---------------------------------------------------------------------------
# PR #290 review Finding 1: Recorded step order and semantic kind identity
# ---------------------------------------------------------------------------


def test_frozen_case_sorts_overlapping_spans_by_step_seq_for_fifo_and_checkpoints():
    """Overlapping spans exported out of start order must be sorted deterministically by
    validated integer neatlogs.step.seq so FIFO matching and until_span cutoffs stay exact."""
    frozen_case = {
        "source": {"kind": "trace", "traceId": "trace-overlap-1"},
        "context": {
            "spans": [
                # Exported in end-time order (3, 2, 1), NOT start/seq order (1, 2, 3)!
                {
                    "span_id": "call-3",
                    "span_type": "TOOL",
                    "tool_name": "after",
                    "tool_input": {"q": "y"},
                    "tool_output": {"v": 3},
                    "start_time_unix_nano": 3000,
                    "attributes": {"neatlogs.step.seq": 3},
                },
                {
                    "span_id": "call-2",
                    "span_type": "TOOL",
                    "tool_name": "lookup",
                    "tool_input": {"q": "x"},
                    "tool_output": {"v": 2},
                    "start_time_unix_nano": 2000,
                    "attributes": {"neatlogs.step.seq": 2},
                },
                {
                    "span_id": "call-1",
                    "span_type": "TOOL",
                    "tool_name": "lookup",
                    "tool_input": {"q": "x"},
                    "tool_output": {"v": 1},
                    "start_time_unix_nano": 1000,
                    "attributes": {"neatlogs.step.seq": 1},
                },
            ]
        },
    }
    converted = replay_recording_from_case(frozen_case)
    recording = converted["recording"]
    assert recording.version == 1
    assert [e.span_id for e in recording.entries] == ["call-1", "call-2", "call-3"]

    # FIFO returns v=1 before v=2 despite call-2 appearing first in the raw snapshot
    with ReplaySession(recording, mode="tools_only"):
        first = lookup_replay("tool", "lookup", {"q": "x"})
        second = lookup_replay("tool", "lookup", {"q": "x"})
        assert first.entry.output == {"v": 1}
        assert second.entry.output == {"v": 2}

    # Checkpoint at call-2 must replay call-1 (before checkpoint), then go LIVE at call-2 & call-3
    with ReplaySession(recording, mode="until_span", until_span="call-2") as session:
        d1 = lookup_replay("tool", "lookup", {"q": "x"})
        d2 = lookup_replay("tool", "lookup", {"q": "x"})
        d3 = lookup_replay("tool", "after", {"q": "y"})
        assert d1.outcome == "HIT" and d1.entry.output == {"v": 1}
        assert d2.outcome == "LIVE"
        assert d3.outcome == "LIVE"
        assert session.get_status()["reason"] == "checkpoint"


def test_frozen_case_legacy_fallback_ordering_by_timestamp_and_snapshot_index():
    """When neatlogs.step.seq is absent or invalid (bool, 0, negative, non-int), fall back
    deterministically to start timestamp, and then original snapshot index."""
    # 1. Invalid/missing step.seq with ISO (sub-microsecond) & unix timestamps -> timestamp order
    ts_case = {
        "source": {"kind": "trace", "traceId": "trace-legacy-ts"},
        "context": {
            "spans": [
                {
                    "span_id": "late",
                    "span_type": "TOOL",
                    "tool_name": "op",
                    "tool_input": {"k": 1},
                    "tool_output": {"seq": "late"},
                    "start_time": "2026-11-20T00:00:01.000000900Z",
                    "attributes": {"neatlogs.step.seq": True},  # bool is invalid
                },
                {
                    "span_id": "early",
                    "span_type": "TOOL",
                    "tool_name": "op",
                    "tool_input": {"k": 1},
                    "tool_output": {"seq": "early"},
                    "start_time": "2026-11-20T00:00:01.000000100Z",
                    "attributes": {"neatlogs.step.seq": -1},  # negative is invalid
                },
                {
                    "span_id": "mid",
                    "span_type": "TOOL",
                    "tool_name": "op",
                    "tool_input": {"k": 1},
                    "tool_output": {"seq": "mid"},
                    "start_time": "2026-11-20T00:00:01.000000500Z",
                    "attributes": {"neatlogs.step.seq": "not-an-int"},
                },
            ]
        },
    }
    ts_rec = replay_recording_from_case(ts_case)["recording"]
    assert [e.span_id for e in ts_rec.entries] == ["early", "mid", "late"]

    # 2. No step.seq and no timestamps -> preserve original snapshot array order
    no_ts_case = {
        "source": {"kind": "trace", "traceId": "trace-legacy-idx"},
        "context": {
            "spans": [
                {
                    "span_id": "first-in-array",
                    "span_type": "TOOL",
                    "tool_name": "op",
                    "tool_input": {"k": 1},
                    "tool_output": {"order": 1},
                    "attributes": {},
                },
                {
                    "span_id": "second-in-array",
                    "span_type": "TOOL",
                    "tool_name": "op",
                    "tool_input": {"k": 1},
                    "tool_output": {"order": 2},
                    "attributes": {},
                },
            ]
        },
    }
    idx_rec = replay_recording_from_case(no_ts_case)["recording"]
    assert [e.span_id for e in idx_rec.entries] == ["first-in-array", "second-in-array"]


def test_semantic_kinds_preserve_identity_and_never_collide_on_same_name_and_input():
    """Distinct span kinds (RETRIEVER, MEMORY, GUARDRAIL, TOOL) sharing the exact same name
    and input must preserve their semantic kind in recording and lookup keys and never consume
    each other's recorded results."""
    from neatlogs.replay.contracts import replay_key

    live_calls = {"retriever": 0, "memory": 0, "guardrail": 0, "tool": 0}

    @neatlogs.span(kind="RETRIEVER", name="lookup", replay=True)
    def lookup_retriever(query: str) -> dict[str, str]:
        live_calls["retriever"] += 1
        return {"live": "retriever"}

    @neatlogs.span(kind="MEMORY", name="lookup", replay=True)
    def lookup_memory(query: str) -> dict[str, str]:
        live_calls["memory"] += 1
        return {"live": "memory"}

    @neatlogs.span(kind="GUARDRAIL", name="lookup", replay=True)
    def lookup_guardrail(query: str) -> dict[str, str]:
        live_calls["guardrail"] += 1
        return {"live": "guardrail"}

    @neatlogs.span(kind="TOOL", tool_name="lookup", replay=True)
    def lookup_tool(query: str) -> dict[str, str]:
        live_calls["tool"] += 1
        return {"live": "tool"}

    # Keys for distinct kinds with identical name and input must all differ
    keys = {
        replay_key(k, "lookup", {"query": "refund"})
        for k in ("retriever", "memory", "guardrail", "reranker", "vector_store", "tool", "model")
    }
    assert len(keys) == 7

    frozen_case = {
        "source": {"kind": "trace", "traceId": "trace-kinds-1"},
        "context": {
            "spans": [
                {
                    "span_id": "s-ret",
                    "span_type": "RETRIEVER",
                    "name": "lookup",
                    "input_value": json.dumps({"query": "refund"}),
                    "output_value": json.dumps({"saved": "retriever"}),
                    "attributes": {"neatlogs.step.seq": 1},
                },
                {
                    "span_id": "s-mem",
                    "span_type": "MEMORY",
                    "name": "lookup",
                    "input_value": json.dumps({"query": "refund"}),
                    "output_value": json.dumps({"saved": "memory"}),
                    "attributes": {"neatlogs.step.seq": 2},
                },
                {
                    "span_id": "s-grd",
                    "span_type": "GUARDRAIL",
                    "name": "lookup",
                    "input_value": json.dumps({"query": "refund"}),
                    "output_value": json.dumps({"saved": "guardrail"}),
                    "attributes": {"neatlogs.step.seq": 3},
                },
                {
                    "span_id": "s-tool",
                    "span_type": "TOOL",
                    "tool_name": "lookup",
                    "tool_input": {"query": "refund"},
                    "tool_output": {"saved": "tool"},
                    "attributes": {"neatlogs.step.seq": 4},
                },
            ]
        },
    }
    extracted = replay_recording_from_case(frozen_case, include_steps=True)
    recording = extracted["recording"]
    assert recording.version == 2
    assert [e.kind for e in recording.entries] == ["retriever", "memory", "guardrail", "tool"]

    # Call in reverse order with identical arguments — each kind hits ONLY its own entry
    with ReplaySession(recording, mode="tools_only") as session:
        assert lookup_tool("refund") == {"saved": "tool"}
        assert lookup_guardrail("refund") == {"saved": "guardrail"}
        assert lookup_memory("refund") == {"saved": "memory"}
        assert lookup_retriever("refund") == {"saved": "retriever"}
        assert session.get_status()["hits"] == 4
        assert session.get_status()["liveCalls"] == 0
    assert live_calls == {"retriever": 0, "memory": 0, "guardrail": 0, "tool": 0}


# ---------------------------------------------------------------------------
# PR #290 review Finding 2: Strict shared v1 contract vs opted-in v2 contract
# ---------------------------------------------------------------------------


def test_replay_contract_v1_strictly_excludes_step_and_error_entries():
    """Default replay_recording_from_case() must emit strict version=1 recordings matching
    the shared TS validator (tool/model only, no error field). Extended step/error entries
    are emitted ONLY under version=2 and rejected under version=1."""
    from neatlogs.replay import ReplayError, parse_replay_recording

    mixed_case = {
        "source": {"kind": "trace", "traceId": "trace-contract-1"},
        "context": {
            "spans": [
                {
                    "span_id": "r1",
                    "span_type": "RETRIEVER",
                    "name": "fetch",
                    "input_value": json.dumps({"q": "hi"}),
                    "output_value": json.dumps([{"id": "1"}]),
                    "attributes": {"neatlogs.step.seq": 1},
                },
                {
                    "span_id": "t1",
                    "span_type": "TOOL",
                    "tool_name": "ok_tool",
                    "tool_input": {"x": 1},
                    "tool_output": {"y": 2},
                    "attributes": {"neatlogs.step.seq": 2},
                },
                {
                    "span_id": "t2",
                    "span_type": "TOOL",
                    "tool_name": "bad_tool",
                    "tool_input": {"x": 2},
                    "status": "ERROR",
                    "error_message": "boom",
                    "attributes": {"neatlogs.step.seq": 3},
                },
            ]
        },
    }

    # Default (v1): only valid v1 tool/model entries, no error field, version == 1
    v1_out = replay_recording_from_case(mixed_case)
    v1_dict = v1_out["recording"].to_dict()
    assert v1_dict["version"] == 1
    assert v1_out["skippedTools"] == 1
    assert len(v1_dict["entries"]) == 1
    assert set(v1_dict["entries"][0]) == {"spanId", "kind", "name", "input", "output"}
    assert v1_dict["entries"][0]["kind"] == "tool"
    assert parse_replay_recording(v1_dict).version == 1

    # Opted-in include_steps / include_errors: emits version == 2 and round-trips cleanly
    v2_steps = replay_recording_from_case(mixed_case, include_steps=True)["recording"].to_dict()
    assert v2_steps["version"] == 2
    assert [e["kind"] for e in v2_steps["entries"]] == ["retriever", "tool"]
    assert parse_replay_recording(v2_steps).version == 2

    v2_errors = replay_recording_from_case(mixed_case, include_errors=True)["recording"].to_dict()
    assert v2_errors["version"] == 2
    assert v2_errors["entries"][1]["error"]["message"] == "boom"
    assert parse_replay_recording(v2_errors).version == 2

    # Strict v1 validation MUST reject step kinds, collapsed "step", and "error" fields
    for bad_v1_entry in (
        {"spanId": "s1", "kind": "step", "name": "f", "input": {}, "output": {}},
        {"spanId": "s1", "kind": "retriever", "name": "f", "input": {}, "output": {}},
        {
            "spanId": "s1",
            "kind": "tool",
            "name": "f",
            "input": {},
            "output": None,
            "error": {"type": "RuntimeError", "message": "boom"},
        },
    ):
        with pytest.raises(ReplayError):
            parse_replay_recording({"version": 1, "sourceTraceId": "t", "entries": [bad_v1_entry]})

    # Even in v2, collapsed "step" kind is rejected in favor of real semantic kinds
    with pytest.raises(ReplayError):
        parse_replay_recording(
            {
                "version": 2,
                "sourceTraceId": "t",
                "entries": [
                    {"spanId": "s1", "kind": "step", "name": "f", "input": {}, "output": {}}
                ],
            }
        )


# ---------------------------------------------------------------------------
# PR #290 review Finding 3: Align trace_audit replay readiness with frozen_case
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "defect",
    [
        "missing_output",
        "missing_input",
        "error_tool",
        "truncated_state",
        "redacted_tool",
        "credential_in_output",
        "legacy_mcp_string",
        "duplicate_span_id",
    ],
)
def test_trace_audit_never_marks_replay_ready_when_replay_builder_skips_tool(defect: str):
    """Any tool span that replay_recording_from_case() skips MUST fail tool_linkage_valid
    and set replay_ready=False in audit_trace()."""
    root = {
        "span_id": "root-1",
        "parent_span_id": None,
        "name": "wf",
        "span_type": "WORKFLOW",
        "input": {"q": "help"},
        "output": {"a": "done"},
        "start_time_unix_nano": 1000,
        "attributes": {"neatlogs.step.seq": 1, "neatlogs.provenance.release": "v1"},
    }
    tool: dict[str, Any] = {
        "span_id": "tool-1",
        "parent_span_id": "root-1",
        "name": "lookup",
        "span_type": "TOOL",
        "tool_name": "lookup",
        "tool_input": {"order_id": "1"},
        "tool_output": {"ok": True},
        "start_time_unix_nano": 2000,
        "attributes": {"neatlogs.step.seq": 2},
    }
    spans = [root, tool]
    if defect == "missing_output":
        del tool["tool_output"]
    elif defect == "missing_input":
        del tool["tool_input"]
    elif defect == "error_tool":
        tool["status"] = "ERROR"
    elif defect == "truncated_state":
        tool["output_content_state"] = "truncated"
    elif defect == "redacted_tool":
        tool["redacted"] = True
    elif defect == "credential_in_output":
        tool["tool_output"] = {"api_key": "sk-secret-123"}
    elif defect == "legacy_mcp_string":
        tool["span_type"] = "MCP_TOOL"
        tool["tool_output"] = {"result": "plain string"}
    elif defect == "duplicate_span_id":
        spans.append(dict(tool))

    converted = replay_recording_from_case(
        {"source": {"kind": "trace", "traceId": "t-align"}, "context": {"spans": spans}}
    )
    assert converted["skippedTools"] >= 1

    report = audit_trace({"trace_id": "t-align", "finalizationStatus": "finalized"}, spans)
    by_name = {c["name"]: c["status"] for c in report["checks"]}
    assert by_name["tool_linkage_valid"] == "fail"
    assert report["replay_ready"] is False
    assert report["replay_readiness_verified"] is True


def test_trace_audit_accepts_zero_arg_tool_and_marks_metadata_only_unverified():
    """A zero-arg tool with empty dict input and valid output passes both frozen_case and
    audit_trace; metadata-only reads MUST report replay_ready=False and unverified even with
    0 tool spans."""
    spans = [
        {
            "span_id": "root-1",
            "parent_span_id": None,
            "name": "wf",
            "span_type": "WORKFLOW",
            "input": {"q": "ping"},
            "output": {"a": "pong"},
            "start_time_unix_nano": 1000,
            "attributes": {"neatlogs.step.seq": 1, "neatlogs.provenance.release": "v1"},
        },
        {
            "span_id": "tool-1",
            "parent_span_id": "root-1",
            "name": "healthcheck",
            "span_type": "TOOL",
            "tool_name": "healthcheck",
            "tool_input": {},
            "tool_output": {"healthy": True},
            "start_time_unix_nano": 2000,
            "attributes": {"neatlogs.step.seq": 2},
        },
    ]
    converted = replay_recording_from_case(
        {"source": {"kind": "trace", "traceId": "t-zero"}, "context": {"spans": spans}}
    )
    assert converted["skippedTools"] == 0
    full_report = audit_trace({"trace_id": "t-zero", "finalizationStatus": "finalized"}, spans)
    assert full_report["replay_ready"] is True
    assert full_report["replay_readiness_verified"] is True

    # Metadata-only with zero tool spans must STILL never claim replay_ready=True
    meta_only_report = audit_trace(
        {"traceId": "t-meta", "finalizationStatus": "finalized"},
        [
            {
                "spanId": "s1",
                "parentSpanId": None,
                "spanName": "wf",
                "spanType": "workflow",
                "startedAt": "2026-11-20T00:00:01Z",
                "status": "OK",
            }
        ],
        metadata_only=True,
    )
    assert meta_only_report["replay_ready"] is False
    assert meta_only_report["jev_ready"] is False
    assert meta_only_report["replay_readiness_verified"] is False

    # Text output explicitly labels metadata-only replay readiness as unverified
    text_lines: list[str] = []

    def pub_fetch(url, headers, timeout):
        if "/spans" in url:
            body = json.dumps(
                {
                    "success": True,
                    "data": {
                        "spans": [
                            {
                                "spanId": "s1",
                                "parentSpanId": None,
                                "spanName": "wf",
                                "spanType": "workflow",
                                "startedAt": "2026-11-20T00:00:01Z",
                                "status": "OK",
                            }
                        ],
                        "page": {"nextCursor": None},
                    },
                }
            ).encode()
        else:
            body = json.dumps(
                {
                    "success": True,
                    "data": {"traceId": "t-meta", "finalizationStatus": "finalized"},
                }
            ).encode()
        return 200, body

    run_trace_audit(
        "t-meta",
        as_json=False,
        env={"NEATLOGS_TOKEN": "tok", "NEATLOGS_PROJECT_ID": "proj"},
        fetch=pub_fetch,
        out=text_lines.append,
        err=lambda _: None,
    )
    assert any("unverified: metadata-only" in line for line in text_lines)
