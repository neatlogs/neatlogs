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


# ---------------------------------------------------------------------------
# PR #290 Codex adversarial review P1/P2 regression tests
# ---------------------------------------------------------------------------


def test_mixed_step_seq_and_unsequenced_spans_preserve_chronological_order_and_until_span():
    """Finding 1 (P1): An earlier span (ts=100) without step.seq must sort BEFORE a later span
    (ts=200) with step.seq=2 so identical tool calls match in chronological FIFO order and
    until_span on the second call replays the first before switching live."""
    first_span = {
        "span_id": "call-1",
        "span_type": "TOOL",
        "tool_name": "lookup",
        "tool_input": {"q": "same"},
        "tool_output": {"res": "first"},
        "start_time_unix_nano": 100,
        "attributes": {},
    }
    second_span = {
        "span_id": "call-2",
        "span_type": "TOOL",
        "tool_name": "lookup",
        "tool_input": {"q": "same"},
        "tool_output": {"res": "second"},
        "start_time_unix_nano": 200,
        "attributes": {"neatlogs.step.seq": 2},
    }
    for raw_order in ([first_span, second_span], [second_span, first_span]):
        converted = replay_recording_from_case(
            {"source": {"kind": "trace", "traceId": "t-mixed"}, "context": {"spans": raw_order}}
        )
        rec = converted["recording"]
        assert [e.span_id for e in rec.entries] == ["call-1", "call-2"]

        with ReplaySession(rec, mode="tools_only"):
            assert lookup_replay("tool", "lookup", {"q": "same"}).entry.output == {"res": "first"}
            assert lookup_replay("tool", "lookup", {"q": "same"}).entry.output == {"res": "second"}

        with ReplaySession(rec, mode="until_span", until_span="call-2") as session:
            d1 = lookup_replay("tool", "lookup", {"q": "same"})
            d2 = lookup_replay("tool", "lookup", {"q": "same"})
            assert d1.outcome == "HIT" and d1.entry.output == {"res": "first"}
            assert d2.outcome == "LIVE"
            assert session.get_status()["reason"] == "checkpoint"

    # Equal timestamps with complete step.seq still disambiguate by step.seq
    equal_ts_case = {
        "source": {"kind": "trace", "traceId": "t-eq-ts"},
        "context": {
            "spans": [
                {
                    "span_id": "s2",
                    "span_type": "TOOL",
                    "tool_name": "lookup",
                    "tool_input": {"q": "same"},
                    "tool_output": {"res": 2},
                    "start_time_unix_nano": 100,
                    "attributes": {"neatlogs.step.seq": 2},
                },
                {
                    "span_id": "s1",
                    "span_type": "TOOL",
                    "tool_name": "lookup",
                    "tool_input": {"q": "same"},
                    "tool_output": {"res": 1},
                    "start_time_unix_nano": 100,
                    "attributes": {"neatlogs.step.seq": 1},
                },
            ]
        },
    }
    eq_rec = replay_recording_from_case(equal_ts_case)["recording"]
    assert [e.span_id for e in eq_rec.entries] == ["s1", "s2"]

    # Absent timestamps with complete step.seq still sort by step.seq even alongside a WORKFLOW span
    no_ts_case = {
        "source": {"kind": "trace", "traceId": "t-no-ts"},
        "context": {
            "spans": [
                {"span_id": "wf", "span_type": "WORKFLOW", "name": "wf", "attributes": {}},
                {
                    "span_id": "s2",
                    "span_type": "TOOL",
                    "tool_name": "lookup",
                    "tool_input": {"q": "same"},
                    "tool_output": {"res": 2},
                    "attributes": {"neatlogs.step.seq": 2},
                },
                {
                    "span_id": "s1",
                    "span_type": "TOOL",
                    "tool_name": "lookup",
                    "tool_input": {"q": "same"},
                    "tool_output": {"res": 1},
                    "attributes": {"neatlogs.step.seq": 1},
                },
            ]
        },
    }
    no_ts_rec = replay_recording_from_case(no_ts_case)["recording"]
    assert [e.span_id for e in no_ts_rec.entries] == ["s1", "s2"]


def test_64bit_otlp_nanosecond_timestamps_do_not_break_frozen_case_replay(tmp_path: Path):
    """Finding 2 (P1): Standard 64-bit integer OTLP timestamps (> 2^53) must be accepted and
    ordered with exact nanosecond precision while strict float53 checks remain enforced on I/O."""
    from neatlogs.replay import ReplayError, parse_replay_recording
    from neatlogs.replay.contracts import parse_json_bytes

    frozen_case = {
        "source": {"kind": "trace", "traceId": "t-ns64"},
        "context": {
            "spans": [
                {
                    "span_id": "second",
                    "span_type": "TOOL",
                    "tool_name": "lookup",
                    "tool_input": {"k": 1},
                    "tool_output": {"order": 2},
                    "start_time_unix_nano": 1700000000000000002,
                    "end_time_unix_nano": 1700000000000000999,
                    "attributes": {"neatlogs.step.seq": 2},
                },
                {
                    "span_id": "first",
                    "span_type": "TOOL",
                    "tool_name": "lookup",
                    "tool_input": {"k": 1},
                    "tool_output": {"order": 1},
                    "start_time_unix_nano": 1700000000000000001,
                    "end_time_unix_nano": 1700000000000000500,
                    "attributes": {"neatlogs.step.seq": 1},
                },
            ]
        },
    }
    converted = replay_recording_from_case(frozen_case)
    assert converted["skippedTools"] == 0
    assert [e.span_id for e in converted["recording"].entries] == ["first", "second"]

    # CLI file loading (allow_int64=True) parses 64-bit ns frozen case without ReplayError
    raw_bytes = json.dumps(frozen_case).encode("utf-8")
    loaded = parse_json_bytes(raw_bytes, allow_int64=True)
    assert loaded["context"]["spans"][1]["start_time_unix_nano"] == 1700000000000000001
    assert [e.span_id for e in replay_recording_from_case(loaded)["recording"].entries] == [
        "first",
        "second",
    ]

    # Strict float53 representability check MUST still reject 2^53+1 inside replay I/O payloads
    bad_io_case = {
        "source": {"kind": "trace", "traceId": "t-ns64-bad-io"},
        "context": {
            "spans": [
                {
                    "span_id": "bad",
                    "span_type": "TOOL",
                    "tool_name": "lookup",
                    "tool_input": {"unsafe_int": 1700000000000000001},
                    "tool_output": {"ok": True},
                    "start_time_unix_nano": 1700000000000000001,
                }
            ]
        },
    }
    bad_io_converted = replay_recording_from_case(bad_io_case)
    assert bad_io_converted["skippedTools"] == 1
    assert bad_io_converted["recording"].entries == ()

    with pytest.raises(ReplayError):
        parse_replay_recording(
            {
                "version": 1,
                "sourceTraceId": "t",
                "entries": [
                    {
                        "spanId": "s1",
                        "kind": "tool",
                        "name": "lookup",
                        "input": {"unsafe_int": 1700000000000000001},
                        "output": "ok",
                    }
                ],
            }
        )


def test_doctor_and_replay_builder_agree_on_input_output_aliases_and_truncation_flags():
    """Finding 3 (P2): Doctor and replay_recording_from_case must agree 100% on normalized
    TOOL spans (input={}, output='ok') AND on input_truncated / output_truncated flags."""
    root = {
        "span_id": "root-1",
        "parent_span_id": None,
        "name": "wf",
        "span_type": "WORKFLOW",
        "input": {"q": "hi"},
        "output": {"a": "bye"},
        "start_time_unix_nano": 1000,
        "attributes": {"neatlogs.step.seq": 1, "neatlogs.provenance.release": "v1"},
    }

    # Case A: TOOL with input={} and output="ok" -> BOTH accept
    tool_alias = {
        "span_id": "tool-1",
        "parent_span_id": "root-1",
        "name": "ping",
        "span_type": "TOOL",
        "input": {},
        "output": "ok",
        "start_time_unix_nano": 2000,
        "attributes": {"neatlogs.step.seq": 2},
    }
    spans_a = [root, tool_alias]
    conv_a = replay_recording_from_case(
        {"source": {"kind": "trace", "traceId": "t-a"}, "context": {"spans": spans_a}}
    )
    audit_a = audit_trace({"trace_id": "t-a", "finalizationStatus": "finalized"}, spans_a)
    assert conv_a["skippedTools"] == 0
    assert len(conv_a["recording"].entries) == 1
    assert conv_a["recording"].entries[0].input == {}
    assert conv_a["recording"].entries[0].output == "ok"
    assert audit_a["replay_ready"] is True

    # Case B: TOOL with canonical I/O + output_truncated=True (or input_truncated=True) -> BOTH reject
    for trunc_field in ("output_truncated", "input_truncated"):
        tool_trunc = {
            "span_id": "tool-1",
            "parent_span_id": "root-1",
            "name": "fetch",
            "span_type": "TOOL",
            "tool_name": "fetch",
            "tool_input": {"id": 1},
            "tool_output": {"data": "partial"},
            trunc_field: True,
            "start_time_unix_nano": 2000,
            "attributes": {"neatlogs.step.seq": 2},
        }
        spans_b = [root, tool_trunc]
        conv_b = replay_recording_from_case(
            {"source": {"kind": "trace", "traceId": "t-b"}, "context": {"spans": spans_b}}
        )
        audit_b = audit_trace({"trace_id": "t-b", "finalizationStatus": "finalized"}, spans_b)
        by_name_b = {c["name"]: c["status"] for c in audit_b["checks"]}
        assert conv_b["skippedTools"] == 1
        assert conv_b["recording"].entries == ()
        assert by_name_b["tool_linkage_valid"] == "fail"
        assert audit_b["replay_ready"] is False


def test_offline_trace_get_json_export_is_detected_as_metadata_only_and_not_replay_ready(
    tmp_path: Path,
):
    """Finding 4 (P2): Passing `neatlogs trace get --json` summary output to `doctor --file`
    must detect metadata-only spans (including `type` span kind) and NEVER report replay_ready."""
    from neatlogs.trace_cli import run_trace_get

    public_trace = {
        "traceId": "t-summary-1",
        "status": "success",
        "finalizationStatus": "finalized",
        "spansCount": 2,
        "totalTokens": 10,
    }
    public_spans = [
        {"spanId": "s1", "parentSpanId": None, "spanName": "wf", "spanType": "workflow"},
        {"spanId": "s2", "parentSpanId": "s1", "spanName": "lookup", "spanType": "tool"},
    ]

    def fake_fetch(url, headers, timeout):
        if "/spans" in url:
            body = json.dumps(
                {"success": True, "data": {"spans": public_spans, "page": {"nextCursor": None}}}
            ).encode()
        else:
            body = json.dumps({"success": True, "data": public_trace}).encode()
        return 200, body

    exported_lines: list[str] = []
    rc_get = run_trace_get(
        "t-summary-1",
        True,
        env={"NEATLOGS_TOKEN": "tok", "NEATLOGS_PROJECT_ID": "proj"},
        fetch=fake_fetch,
        out=exported_lines.append,
        err=lambda _: None,
    )
    assert rc_get == 0
    summary_file = tmp_path / "trace_get_export.json"
    summary_file.write_text(exported_lines[0], encoding="utf-8")

    audit_out: list[str] = []
    rc_doc = run_trace_audit(
        None,
        file_path=str(summary_file),
        as_json=True,
        out=audit_out.append,
        err=lambda _: None,
    )
    assert rc_doc == 1
    report = json.loads(audit_out[0])
    assert report["replay_ready"] is False
    assert report["jev_ready"] is False
    assert report["replay_readiness_verified"] is False
    by_name = {c["name"]: c["status"] for c in report["checks"]}
    assert by_name["tool_linkage_valid"] == "warn"
    assert by_name["root_io_present"] == "warn"


def test_replay_ready_rejects_inverted_step_seq_and_over_5000_span_traces():
    """Finding 5 (P2): Doctor replay_ready must be False when step_ordering_clean fails
    (inverted step.seq) OR when span count exceeds the replay builder's 5000-span cap."""
    from neatlogs.replay import ReplayError
    from neatlogs.replay.frozen_case import MAX_FROZEN_CASE_SPANS

    # 1. Inverted step.seq across two identical tool calls
    inverted_spans = [
        {
            "span_id": "root-1",
            "parent_span_id": None,
            "name": "wf",
            "span_type": "WORKFLOW",
            "input": {"q": 1},
            "output": {"a": 2},
            "start_time_unix_nano": 1000,
            "attributes": {"neatlogs.step.seq": 1, "neatlogs.provenance.release": "v1"},
        },
        {
            "span_id": "tool-1",
            "parent_span_id": "root-1",
            "name": "lookup",
            "span_type": "TOOL",
            "tool_name": "lookup",
            "tool_input": {"k": "same"},
            "tool_output": {"v": 1},
            "start_time_unix_nano": 2000,
            "attributes": {"neatlogs.step.seq": 3},
        },
        {
            "span_id": "tool-2",
            "parent_span_id": "root-1",
            "name": "lookup",
            "span_type": "TOOL",
            "tool_name": "lookup",
            "tool_input": {"k": "same"},
            "tool_output": {"v": 2},
            "start_time_unix_nano": 3000,
            "attributes": {"neatlogs.step.seq": 2},
        },
    ]
    inv_report = audit_trace(
        {"trace_id": "t-inv", "finalizationStatus": "finalized"}, inverted_spans
    )
    inv_checks = {c["name"]: c["status"] for c in inv_report["checks"]}
    assert inv_checks["step_ordering_clean"] == "fail"
    assert inv_report["replay_ready"] is False

    # 2. 5,001-span trace rejected by replay builder -> replay_ready MUST be False
    many_spans = [
        {
            "span_id": "s-0",
            "parent_span_id": None,
            "name": "wf",
            "span_type": "WORKFLOW",
            "input": {"q": 1},
            "output": {"a": 2},
            "start_time_unix_nano": 1000,
            "attributes": {"neatlogs.step.seq": 1, "neatlogs.provenance.release": "v1"},
        }
    ] + [
        {
            "span_id": f"s-{i}",
            "parent_span_id": "s-0",
            "name": "step",
            "span_type": "CHAIN",
            "start_time_unix_nano": 1000 + i,
            "attributes": {"neatlogs.step.seq": i + 1},
        }
        for i in range(1, MAX_FROZEN_CASE_SPANS + 1)
    ]
    assert len(many_spans) == MAX_FROZEN_CASE_SPANS + 1
    with pytest.raises(ReplayError):
        replay_recording_from_case(
            {"source": {"kind": "trace", "traceId": "t-5001"}, "context": {"spans": many_spans}}
        )
    many_report = audit_trace({"trace_id": "t-5001", "finalizationStatus": "finalized"}, many_spans)
    many_checks = {c["name"]: c["status"] for c in many_report["checks"]}
    assert many_checks["provenance_and_replayability"] == "fail"
    assert many_report["replay_ready"] is False


def test_semantic_tool_call_missing_arguments_rejected_by_both_builder_and_doctor():
    """Finding 6 (P2): Omitting call.arguments (or setting it to None) on a semantic tool span
    must fail both replay_recording_from_case and Doctor; explicit {} must pass both."""
    root = {
        "span_id": "root-1",
        "parent_span_id": None,
        "name": "wf",
        "span_type": "WORKFLOW",
        "input": {"q": "ping"},
        "output": {"a": "pong"},
        "start_time_unix_nano": 1000,
        "attributes": {"neatlogs.step.seq": 1, "neatlogs.provenance.release": "v1"},
    }

    for bad_call in (
        {"id": "c1", "name": "ping"},  # missing arguments
        {"id": "c1", "name": "ping", "arguments": None},  # null arguments
    ):
        sem_tool = {
            "span_id": "tool-1",
            "parent_span_id": "root-1",
            "span_type": "TOOL",
            "start_time_unix_nano": 2000,
            "attributes": {"neatlogs.step.seq": 2},
            "semantic": {
                "kind": "TOOL",
                "call": bad_call,
                "result": {"call_id": "c1", "is_error": False, "value": "pong"},
                "definition": {"name": "ping"},
            },
        }
        spans = [root, sem_tool]
        conv = replay_recording_from_case(
            {"source": {"kind": "trace", "traceId": "t-sem-bad"}, "context": {"spans": spans}}
        )
        audit = audit_trace({"trace_id": "t-sem-bad", "finalizationStatus": "finalized"}, spans)
        assert conv["skippedTools"] == 1
        assert conv["recording"].entries == ()
        assert audit["replay_ready"] is False

    # Valid zero-arg semantic tool with arguments={} passes both and matches decorated tool
    good_sem_tool = {
        "span_id": "tool-1",
        "parent_span_id": "root-1",
        "span_type": "TOOL",
        "start_time_unix_nano": 2000,
        "attributes": {"neatlogs.step.seq": 2},
        "semantic": {
            "kind": "TOOL",
            "call": {"id": "c1", "name": "ping", "arguments": {}},
            "result": {"call_id": "c1", "is_error": False, "value": "pong"},
            "definition": {"name": "ping"},
        },
    }
    good_spans = [root, good_sem_tool]
    conv_good = replay_recording_from_case(
        {"source": {"kind": "trace", "traceId": "t-sem-ok"}, "context": {"spans": good_spans}}
    )
    audit_good = audit_trace(
        {"trace_id": "t-sem-ok", "finalizationStatus": "finalized"}, good_spans
    )
    assert conv_good["skippedTools"] == 0
    assert conv_good["recording"].entries[0].input == {}
    assert audit_good["replay_ready"] is True

    live_runs = 0

    @neatlogs.span(kind="TOOL", tool_name="ping", replay=True)
    def ping() -> str:
        nonlocal live_runs
        live_runs += 1
        return "live-pong"

    with ReplaySession(conv_good["recording"], mode="tools_only"):
        assert ping() == "pong"
    assert live_runs == 0


def test_v2_model_error_entry_replays_recorded_error_without_running_live_callback():
    """Finding 7 (P2): A cached v2 model entry with error before until_span must raise
    ReplayRecordedError instead of falling through to live() while reporting hits=1."""
    from neatlogs.replay import ReplayRecordedError, replay_model
    from neatlogs.replay.provider import provider_replay

    recording_v2 = {
        "version": 2,
        "sourceTraceId": "t-model-err",
        "entries": [
            {
                "spanId": "m-err",
                "kind": "model",
                "name": "openai.chat.completions.create",
                "input": {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
                "output": None,
                "error": {"type": "RateLimitError", "message": "429 quota exceeded"},
            },
            {
                "spanId": "cp-tool",
                "kind": "tool",
                "name": "after",
                "input": {},
                "output": "ok",
            },
        ],
    }

    live_calls = 0

    def live():
        nonlocal live_calls
        live_calls += 1
        return {"text": "should-never-run"}

    with ReplaySession(recording_v2, mode="until_span", until_span="cp-tool") as session:
        with pytest.raises(ReplayRecordedError, match="RateLimitError: 429 quota exceeded"):
            replay_model(
                "openai.chat.completions.create",
                {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
                lambda v: v,
                live,
            )
        status = session.get_status()
    assert live_calls == 0
    assert status["hits"] == 1
    assert status["liveCalls"] == 0

    # provider_replay must also raise ReplayRecordedError instead of returning None (live fallback)
    with ReplaySession(recording_v2, mode="until_span", until_span="cp-tool") as session2:
        with pytest.raises(ReplayRecordedError, match="RateLimitError"):
            provider_replay(
                "openai.chat.completions.create",
                {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
                supported=True,
                span=None,
            )
        status2 = session2.get_status()
    assert status2["hits"] == 1
    assert status2["liveCalls"] == 0


def test_malformed_unicode_ordering_metadata_does_not_abort_recording_or_audit():
    """Finding 8 (P2): Unicode superscript '²' passes str.isdigit() in Python but raises
    ValueError in int(); malformed step.seq / timestamp strings must be safely ignored."""
    spans = [
        {
            "span_id": "root-1",
            "parent_span_id": None,
            "name": "wf",
            "span_type": "WORKFLOW",
            "input": {"q": 1},
            "output": {"a": 2},
            "start_time": "²",
            "attributes": {"neatlogs.step.seq": "²", "neatlogs.provenance.release": "v1"},
        },
        {
            "span_id": "tool-1",
            "parent_span_id": "root-1",
            "name": "ok_tool",
            "span_type": "TOOL",
            "tool_name": "ok_tool",
            "tool_input": {"x": 1},
            "tool_output": {"y": 2},
            "start_time": "9" * 5000,
            "attributes": {"neatlogs.step.seq": "²"},
        },
    ]
    converted = replay_recording_from_case(
        {"source": {"kind": "trace", "traceId": "t-unicode-meta"}, "context": {"spans": spans}}
    )
    assert converted["skippedTools"] == 0
    assert [e.span_id for e in converted["recording"].entries] == ["tool-1"]

    report = audit_trace({"trace_id": "t-unicode-meta", "finalizationStatus": "finalized"}, spans)
    assert report["span_count"] == 2
    assert report["replay_ready"] is True


def test_unhashable_entry_kind_raises_replay_error_and_is_caught_by_cli(tmp_path: Path, capsys):
    """Finding 9 (P2): Non-string / unhashable entry kinds (list, dict) must raise ReplayError
    rather than leaking TypeError out of parse_replay_recording and run_simulate_cli."""
    from neatlogs.replay import ReplayError, parse_replay_recording
    from neatlogs.replay.cli import run_simulate_cli

    for bad_kind in (["tool"], {"kind": "tool"}, 123):
        bad_recording = {
            "version": 1,
            "sourceTraceId": "t-bad-kind",
            "entries": [
                {"spanId": "s1", "kind": bad_kind, "name": "op", "input": {}, "output": {}}
            ],
        }
        with pytest.raises(ReplayError, match="Unsupported local replay entry kind"):
            parse_replay_recording(bad_recording)

    agent_cfg = tmp_path / "agent.yaml"
    agent_cfg.write_text("command: ['python', '-c', 'pass']\n", encoding="utf-8")
    rec_file = tmp_path / "bad_kind.json"
    rec_file.write_text(
        json.dumps(
            {
                "version": 1,
                "sourceTraceId": "t-bad-kind",
                "entries": [
                    {"spanId": "s1", "kind": ["tool"], "name": "op", "input": {}, "output": {}}
                ],
            }
        ),
        encoding="utf-8",
    )
    rc = run_simulate_cli(
        ["simulate", "local", "--recording", str(rec_file), "--agent", str(agent_cfg)]
    )
    captured = capsys.readouterr()
    assert rc == 1
    assert "Unsupported local replay entry kind" in captured.err


def test_doctor_eu_endpoint_resolves_eu_app_host_for_token_and_respects_explicit_host():
    """Finding 11 (P1): With NEATLOGS_TOKEN + NEATLOGS_PROJECT_ID and EU NEATLOGS_ENDPOINT
    (or --endpoint https://eu.ingest.neatlogs.com/...), Doctor must hit eu.app.neatlogs.com
    rather than US app.neatlogs.com, while respecting explicit NEATLOGS_HOST."""

    def _ok_public(url, headers, timeout):
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
                    "data": {"traceId": "t-eu", "finalizationStatus": "finalized"},
                }
            ).encode()
        return 200, body

    # 1. EU NEATLOGS_ENDPOINT without NEATLOGS_HOST -> derives https://eu.app.neatlogs.com
    urls_env: list[str] = []

    def fetch_env(url, headers, timeout):
        urls_env.append(url)
        return _ok_public(url, headers, timeout)

    run_trace_audit(
        "t-eu",
        as_json=True,
        env={
            "NEATLOGS_TOKEN": "tok-eu",
            "NEATLOGS_PROJECT_ID": "proj-eu",
            "NEATLOGS_ENDPOINT": "https://eu.ingest.neatlogs.com/api/v1/traces",
        },
        fetch=fetch_env,
        out=lambda _: None,
        err=lambda _: None,
    )
    assert urls_env[0] == "https://eu.app.neatlogs.com/api/v1/public/traces/t-eu"

    # 2. Explicit --endpoint https://eu.ingest.neatlogs.com/api/v1/traces -> maps to eu.app.neatlogs.com
    urls_flag: list[str] = []

    def fetch_flag(url, headers, timeout):
        urls_flag.append(url)
        return _ok_public(url, headers, timeout)

    run_trace_audit(
        "t-eu",
        endpoint="https://eu.ingest.neatlogs.com/api/v1/traces",
        as_json=True,
        env={"NEATLOGS_TOKEN": "tok-eu", "NEATLOGS_PROJECT_ID": "proj-eu"},
        fetch=fetch_flag,
        out=lambda _: None,
        err=lambda _: None,
    )
    assert urls_flag[0] == "https://eu.app.neatlogs.com/api/v1/public/traces/t-eu"

    # 3. Explicit NEATLOGS_HOST wins over NEATLOGS_ENDPOINT when --endpoint is not passed
    urls_host: list[str] = []

    def fetch_host(url, headers, timeout):
        urls_host.append(url)
        return _ok_public(url, headers, timeout)

    run_trace_audit(
        "t-eu",
        as_json=True,
        env={
            "NEATLOGS_TOKEN": "tok-eu",
            "NEATLOGS_PROJECT_ID": "proj-eu",
            "NEATLOGS_HOST": "https://custom.eu.internal",
            "NEATLOGS_ENDPOINT": "https://eu.ingest.neatlogs.com/api/v1/traces",
        },
        fetch=fetch_host,
        out=lambda _: None,
        err=lambda _: None,
    )
    assert urls_host[0] == "https://custom.eu.internal/api/v1/public/traces/t-eu"

    # 4. NEATLOGS_API_KEY + NEATLOGS_HOST=https://eu.app.neatlogs.com -> maps to eu.ingest.neatlogs.com
    urls_v3: list[str] = []

    def fetch_v3(url, headers, timeout):
        urls_v3.append(url)
        return (
            200,
            json.dumps(
                {
                    "_id": "t-eu",
                    "status": "success",
                    "spans": [
                        {
                            "span_id": "s1",
                            "node_name": "wf",
                            "node_type": "workflow",
                            "data": {"input_value": "hi", "output_value": "ok"},
                            "span_metadata": {"neatlogs.step.seq": 1},
                        }
                    ],
                }
            ).encode(),
        )

    run_trace_audit(
        "t-eu",
        as_json=True,
        env={"NEATLOGS_API_KEY": "key-eu", "NEATLOGS_HOST": "https://eu.app.neatlogs.com"},
        fetch=fetch_v3,
        out=lambda _: None,
        err=lambda _: None,
    )
    assert urls_v3 == ["https://eu.ingest.neatlogs.com/api/traces/v3/t-eu"]


# ---------------------------------------------------------------------------
# PR #290 Codex second-pass P1/P2 regression tests
# ---------------------------------------------------------------------------


def test_frozen_case_extract_before_sort_and_all_sequenced_step_seq_precedence():
    """Finding 1 (P1): Extract eligible replay entries BEFORE sorting so non-replayable
    spans (e.g. unsequenced LLM without capture) cannot break all-sequenced ordering; when
    every eligible entry has step.seq, step.seq takes precedence over skewed timestamps."""
    # (a) All-sequenced with oppositely ordered/skewed timestamps -> sorted 1, 2
    skewed_case = {
        "source": {"kind": "trace", "traceId": "t-skew"},
        "context": {
            "spans": [
                {
                    "span_id": "step-2",
                    "span_type": "TOOL",
                    "tool_name": "lookup",
                    "tool_input": {"q": "same"},
                    "tool_output": {"order": 2},
                    "start_time_unix_nano": 100,
                    "attributes": {"neatlogs.step.seq": 2},
                },
                {
                    "span_id": "step-1",
                    "span_type": "TOOL",
                    "tool_name": "lookup",
                    "tool_input": {"q": "same"},
                    "tool_output": {"order": 1},
                    "start_time_unix_nano": 200,
                    "attributes": {"neatlogs.step.seq": 1},
                },
            ]
        },
    }
    skewed_rec = replay_recording_from_case(skewed_case)["recording"]
    assert [e.span_id for e in skewed_rec.entries] == ["step-1", "step-2"]

    # (b) Timestamp-free [tool seq=2, unsequenced non-replayable LLM, tool seq=1] -> sorted 1, 2
    with_unsequenced_llm = {
        "source": {"kind": "trace", "traceId": "t-llm-drop"},
        "context": {
            "spans": [
                {
                    "span_id": "tool-2",
                    "span_type": "TOOL",
                    "tool_name": "lookup",
                    "tool_input": {"q": "same"},
                    "tool_output": {"order": 2},
                    "attributes": {"neatlogs.step.seq": 2},
                },
                {
                    "span_id": "llm-no-capture",
                    "span_type": "LLM",
                    "name": "gpt-4o",
                    "attributes": {},
                },
                {
                    "span_id": "tool-1",
                    "span_type": "TOOL",
                    "tool_name": "lookup",
                    "tool_input": {"q": "same"},
                    "tool_output": {"order": 1},
                    "attributes": {"neatlogs.step.seq": 1},
                },
            ]
        },
    }
    dropped_rec = replay_recording_from_case(with_unsequenced_llm)["recording"]
    assert [e.span_id for e in dropped_rec.entries] == ["tool-1", "tool-2"]

    # (c) Mixed eligible [tool t=100 no seq, tool t=200 seq=2] -> chronological t=100, t=200
    mixed_case = {
        "source": {"kind": "trace", "traceId": "t-mixed-2"},
        "context": {
            "spans": [
                {
                    "span_id": "later-seq",
                    "span_type": "TOOL",
                    "tool_name": "lookup",
                    "tool_input": {"q": "same"},
                    "tool_output": {"order": 2},
                    "start_time_unix_nano": 200,
                    "attributes": {"neatlogs.step.seq": 2},
                },
                {
                    "span_id": "earlier-unseq",
                    "span_type": "TOOL",
                    "tool_name": "lookup",
                    "tool_input": {"q": "same"},
                    "tool_output": {"order": 1},
                    "start_time_unix_nano": 100,
                    "attributes": {},
                },
            ]
        },
    }
    mixed_rec = replay_recording_from_case(mixed_case)["recording"]
    assert [e.span_id for e in mixed_rec.entries] == ["earlier-unseq", "later-seq"]


def test_top_level_input_output_strings_are_not_double_json_decoded():
    """Finding 2 (P1): Already-typed top-level span['output'] / span['input'] values such as
    'false' or 'null' must NOT be JSON-decoded a second time into False / None, whereas
    OTLP stringified attributes (output.value / input.value) still decode JSON."""
    frozen_case = {
        "source": {"kind": "trace", "traceId": "t-no-double-decode"},
        "context": {
            "spans": [
                {
                    "span_id": "t-false",
                    "span_type": "TOOL",
                    "name": "flag_tool",
                    "input": "true",
                    "output": "false",
                    "attributes": {"neatlogs.step.seq": 1},
                },
                {
                    "span_id": "t-null",
                    "span_type": "TOOL",
                    "name": "null_tool",
                    "input": {},
                    "output": "null",
                    "attributes": {"neatlogs.step.seq": 2},
                },
                {
                    "span_id": "r-step",
                    "span_type": "RETRIEVER",
                    "name": "step_tool",
                    "input": "null",
                    "output": "false",
                    "attributes": {"neatlogs.step.seq": 3},
                },
                {
                    "span_id": "t-otel-attr",
                    "span_type": "TOOL",
                    "name": "otel_tool",
                    "attributes": {
                        "neatlogs.step.seq": 4,
                        "input.value": '{"x": 1}',
                        "output.value": "false",
                    },
                },
            ]
        },
    }
    rec = replay_recording_from_case(frozen_case, include_steps=True)["recording"]
    by_id = {e.span_id: e for e in rec.entries}
    assert by_id["t-false"].input == "true" and type(by_id["t-false"].input) is str
    assert by_id["t-false"].output == "false" and type(by_id["t-false"].output) is str
    assert by_id["t-null"].output == "null" and type(by_id["t-null"].output) is str
    assert by_id["r-step"].input == "null" and type(by_id["r-step"].input) is str
    assert by_id["r-step"].output == "false" and type(by_id["r-step"].output) is str
    assert by_id["t-otel-attr"].input == {"x": 1}
    assert by_id["t-otel-attr"].output is False


def test_doctor_iso_timestamp_preserves_full_nanosecond_precision_and_timestamp_free_order():
    """Findings 4 & 5 (P2): Doctor preserves nanosecond precision on ISO timestamps (distinguishing
    .000000001Z vs .000000002Z) AND does not falsely fail timestamp-free child-first snapshots."""
    # Finding 4: .000000001Z (seq=2) vs .000000002Z (seq=1) -> inversion MUST FAIL
    inv_ns_spans = [
        {
            "span_id": "root-1",
            "parent_span_id": None,
            "name": "wf",
            "span_type": "WORKFLOW",
            "input": {"q": 1},
            "output": {"a": 2},
            "start_time": "2026-11-20T00:00:01.000000001Z",
            "attributes": {"neatlogs.step.seq": 2, "neatlogs.provenance.release": "v1"},
        },
        {
            "span_id": "tool-1",
            "parent_span_id": "root-1",
            "name": "op",
            "span_type": "TOOL",
            "tool_input": {"k": 1},
            "tool_output": {"v": 1},
            "start_time": "2026-11-20T00:00:01.000000002Z",
            "attributes": {"neatlogs.step.seq": 1},
        },
    ]
    inv_report = audit_trace(
        {"trace_id": "t-ns-inv", "finalizationStatus": "finalized"}, inv_ns_spans
    )
    inv_checks = {c["name"]: c["status"] for c in inv_report["checks"]}
    assert inv_checks["step_ordering_clean"] == "fail"
    assert inv_report["replay_ready"] is False

    # Same nanosecond timestamps in monotonic order (seq=1 at .000000001Z, seq=2 at .000000002Z) -> PASS
    ok_ns_spans = [
        {
            **inv_ns_spans[0],
            "attributes": {"neatlogs.step.seq": 1, "neatlogs.provenance.release": "v1"},
        },
        {**inv_ns_spans[1], "attributes": {"neatlogs.step.seq": 2}},
    ]
    ok_report = audit_trace({"trace_id": "t-ns-ok", "finalizationStatus": "finalized"}, ok_ns_spans)
    ok_checks = {c["name"]: c["status"] for c in ok_report["checks"]}
    assert ok_checks["step_ordering_clean"] == "pass"
    assert ok_report["replay_ready"] is True

    # Finding 5: Timestamp-free child-first snapshot (seq=2, 1) -> MUST PASS
    child_first_no_ts = [
        {
            "span_id": "tool-1",
            "parent_span_id": "root-1",
            "name": "op",
            "span_type": "TOOL",
            "tool_input": {"k": 1},
            "tool_output": {"v": 1},
            "attributes": {"neatlogs.step.seq": 2},
        },
        {
            "span_id": "root-1",
            "parent_span_id": None,
            "name": "wf",
            "span_type": "WORKFLOW",
            "input": {"q": 1},
            "output": {"a": 2},
            "attributes": {"neatlogs.step.seq": 1, "neatlogs.provenance.release": "v1"},
        },
    ]
    cf_report = audit_trace(
        {"trace_id": "t-cf-nots", "finalizationStatus": "finalized"}, child_first_no_ts
    )
    cf_checks = {c["name"]: c["status"] for c in cf_report["checks"]}
    assert cf_checks["step_ordering_clean"] == "pass"
    assert cf_report["replay_ready"] is True

    # Timestamp-free duplicate seq (1, 1) still FAILS
    dup_no_ts = [
        {**child_first_no_ts[0], "attributes": {"neatlogs.step.seq": 1}},
        child_first_no_ts[1],
    ]
    dup_report = audit_trace(
        {"trace_id": "t-dup-nots", "finalizationStatus": "finalized"}, dup_no_ts
    )
    dup_checks = {c["name"]: c["status"] for c in dup_report["checks"]}
    assert dup_checks["step_ordering_clean"] == "fail"
    assert dup_report["replay_ready"] is False


def test_doctor_json_encoded_semantic_tool_and_empty_attributes_metadata_only():
    """Findings 6 & 7 (P2): Doctor parses JSON-encoded `semantic` strings when resolving tool kind
    (including rejecting missing call.arguments) and treats empty `attributes: {}` with no
    payloads as metadata-only."""
    root = {
        "span_id": "root-1",
        "parent_span_id": None,
        "name": "wf",
        "span_type": "WORKFLOW",
        "input": {"q": "hi"},
        "output": {"a": "ok"},
        "attributes": {"neatlogs.step.seq": 1, "neatlogs.provenance.release": "v1"},
    }

    # Finding 6: JSON-encoded semantic string without span_type and missing call.arguments -> FAIL
    bad_sem_str_tool = {
        "span_id": "tool-1",
        "parent_span_id": "root-1",
        "attributes": {"neatlogs.step.seq": 2},
        "semantic": json.dumps(
            {
                "kind": "tool",
                "call": {"id": "c1", "name": "ping"},
                "result": {"call_id": "c1", "is_error": False, "value": "pong"},
                "definition": {"name": "ping"},
            }
        ),
    }
    bad_spans = [root, bad_sem_str_tool]
    conv_bad = replay_recording_from_case(
        {"source": {"kind": "trace", "traceId": "t-semstr-bad"}, "context": {"spans": bad_spans}}
    )
    audit_bad = audit_trace(
        {"trace_id": "t-semstr-bad", "finalizationStatus": "finalized"}, bad_spans
    )
    by_name_bad = {c["name"]: c["status"] for c in audit_bad["checks"]}
    assert conv_bad["skippedTools"] == 1
    assert by_name_bad["tool_linkage_valid"] == "fail"
    assert audit_bad["replay_ready"] is False

    # Same JSON-encoded semantic string WITH arguments={} -> PASS in both
    good_sem_str_tool = {
        "span_id": "tool-1",
        "parent_span_id": "root-1",
        "attributes": {"neatlogs.step.seq": 2},
        "semantic": json.dumps(
            {
                "kind": "tool",
                "call": {"id": "c1", "name": "ping", "arguments": {}},
                "result": {"call_id": "c1", "is_error": False, "value": "pong"},
                "definition": {"name": "ping"},
            }
        ),
    }
    good_spans = [root, good_sem_str_tool]
    conv_good = replay_recording_from_case(
        {"source": {"kind": "trace", "traceId": "t-semstr-ok"}, "context": {"spans": good_spans}}
    )
    audit_good = audit_trace(
        {"trace_id": "t-semstr-ok", "finalizationStatus": "finalized"}, good_spans
    )
    assert conv_good["skippedTools"] == 0
    assert audit_good["replay_ready"] is True

    # Finding 7: Empty `attributes: {}` with no payloads is NOT full capture evidence
    empty_attr_report = audit_trace(
        {"trace_id": "t-empty-attrs", "finalizationStatus": "finalized"},
        [
            {
                "span_id": "s1",
                "parent_span_id": None,
                "name": "wf",
                "type": "workflow",
                "attributes": {},
            },
            {
                "span_id": "s2",
                "parent_span_id": "s1",
                "name": "lookup",
                "type": "tool",
                "attributes": {},
            },
        ],
    )
    assert empty_attr_report["replay_ready"] is False
    assert empty_attr_report["jev_ready"] is False
    assert empty_attr_report["replay_readiness_verified"] is False


def test_eu_host_resolution_handles_trailing_fqdn_dot_and_eu_app_endpoint():
    """Finding 8 (P2): Trailing FQDN dot and eu.app.neatlogs.com in NEATLOGS_ENDPOINT must
    resolve cleanly to https://eu.app.neatlogs.com (and eu.ingest.neatlogs.com for v3)."""
    from neatlogs.trace_cli import resolve_public_api_host

    assert (
        resolve_public_api_host(
            {"NEATLOGS_ENDPOINT": "https://eu.ingest.neatlogs.com./api/v1/traces"}
        )
        == "https://eu.app.neatlogs.com"
    )
    assert (
        resolve_public_api_host({"NEATLOGS_ENDPOINT": "https://eu.app.neatlogs.com"})
        == "https://eu.app.neatlogs.com"
    )
    assert (
        resolve_public_api_host({"NEATLOGS_ENDPOINT": "https://eu.app.neatlogs.com."})
        == "https://eu.app.neatlogs.com"
    )
    assert (
        resolve_public_api_host({}, endpoint="https://eu.ingest.neatlogs.com./api/v1/traces")
        == "https://eu.app.neatlogs.com"
    )

    urls_v3: list[str] = []

    def fetch_v3(url, headers, timeout):
        urls_v3.append(url)
        return (
            200,
            json.dumps(
                {
                    "_id": "t-eu-dot",
                    "status": "success",
                    "spans": [
                        {
                            "span_id": "s1",
                            "node_name": "wf",
                            "node_type": "workflow",
                            "data": {"input_value": "hi", "output_value": "ok"},
                            "span_metadata": {"neatlogs.step.seq": 1},
                        }
                    ],
                }
            ).encode(),
        )

    run_trace_audit(
        "t-eu-dot",
        as_json=True,
        env={"NEATLOGS_API_KEY": "key-eu", "NEATLOGS_HOST": "https://eu.app.neatlogs.com."},
        fetch=fetch_v3,
        out=lambda _: None,
        err=lambda _: None,
    )
    assert urls_v3 == ["https://eu.ingest.neatlogs.com/api/traces/v3/t-eu-dot"]


def test_huge_json_integer_raises_replay_error_instead_of_overflow_error():
    """Finding 9 (P2): A 310-digit JSON integer must raise ReplayError (never uncaught
    OverflowError from math.isfinite/float coercion) across json_value, parse_json_bytes,
    and parse_replay_recording."""
    from neatlogs.replay import ReplayError, parse_replay_recording
    from neatlogs.replay.contracts import json_value, parse_json_bytes

    huge = int("9" * 310)
    with pytest.raises(ReplayError, match="exactly representable"):
        json_value({"n": huge})
    with pytest.raises(ReplayError, match="exactly representable"):
        json_value({"n": huge}, allow_int64=True)

    raw_bytes = (
        b'{"version":1,"sourceTraceId":"t","entries":[{"spanId":"s1","kind":"tool","name":"f","input":{"n":'
        + (b"9" * 310)
        + b'},"output":"ok"}]}'
    )
    with pytest.raises(ReplayError):
        parse_json_bytes(raw_bytes)
    with pytest.raises(ReplayError):
        parse_json_bytes(raw_bytes, allow_int64=True)
    with pytest.raises(ReplayError):
        parse_replay_recording(
            {
                "version": 1,
                "sourceTraceId": "t",
                "entries": [
                    {
                        "spanId": "s1",
                        "kind": "tool",
                        "name": "f",
                        "input": {"n": huge},
                        "output": "ok",
                    }
                ],
            }
        )


def test_polish_explicit_none_output_plain_text_mime_iso_overflow_and_whitespace_attrs():
    """Verify final edge-case polishes:
    1. Explicit top-level `"output": None` is preserved (not skipped as missing).
    2. `output.mime_type == "text/plain"` prevents JSON-decoding string `output.value`.
    3. Lowercase `t`/`z` ISO timestamps and UTC-conversion OverflowError (`0001-01-01T00:00:00+01:00`)
       are handled cleanly in both replay builder and Doctor.
    4. Spaced JSON-string `attributes: "{ }"` stays metadata-only, and whitespace `semantic.kind: " "`
       does not mask `span_type: "TOOL"`.
    """
    # 1 & 2: Explicit top-level output=None + text/plain mime_type on output.value
    root = {
        "span_id": "root-1",
        "parent_span_id": None,
        "name": "wf",
        "span_type": "WORKFLOW",
        "input": {"q": "go"},
        "output": {"a": "done"},
        "start_time": "2026-11-20t00:00:01.000000001z",
        "attributes": {"neatlogs.step.seq": 1, "neatlogs.provenance.release": "v1"},
    }
    none_tool = {
        "span_id": "t-none",
        "parent_span_id": "root-1",
        "name": "noop",
        "span_type": "TOOL",
        "input": {},
        "output": None,
        "start_time": "2026-11-20t00:00:01.000000002z",
        "attributes": {"neatlogs.step.seq": 2},
    }
    plain_tool = {
        "span_id": "t-plain",
        "parent_span_id": "root-1",
        "name": "plain_text_tool",
        "span_type": "TOOL",
        "start_time": "0001-01-01T00:00:00+01:00",  # UTC conversion OverflowError -> safely ignored
        "attributes": {
            "neatlogs.step.seq": 3,
            "input.value": '{"q": 1}',
            "output.value": "false",
            "output.mime_type": "text/plain",
        },
    }
    spans = [root, none_tool, plain_tool]
    conv = replay_recording_from_case(
        {"source": {"kind": "trace", "traceId": "t-polish"}, "context": {"spans": spans}}
    )
    assert conv["skippedTools"] == 0
    by_id = {e.span_id: e for e in conv["recording"].entries}
    assert by_id["t-none"].output is None
    assert by_id["t-plain"].output == "false" and type(by_id["t-plain"].output) is str

    audit = audit_trace({"trace_id": "t-polish", "finalizationStatus": "finalized"}, spans)
    assert audit["replay_ready"] is True

    # 3: Lowercase 't' + nanosecond inversion caught cleanly by Doctor
    inv_lower_t = [
        {
            **root,
            "start_time": "2026-11-20t00:00:01.000000001z",
            "attributes": {"neatlogs.step.seq": 2},
        },
        {
            **none_tool,
            "start_time": "2026-11-20t00:00:01.000000002z",
            "attributes": {"neatlogs.step.seq": 1},
        },
    ]
    inv_audit = audit_trace(
        {"trace_id": "t-lower-inv", "finalizationStatus": "finalized"}, inv_lower_t
    )
    assert {c["name"]: c["status"] for c in inv_audit["checks"]}["step_ordering_clean"] == "fail"
    assert inv_audit["replay_ready"] is False

    # 4a: JSON-string `attributes: "{ }"` with no payloads is still metadata-only
    spaced_empty_report = audit_trace(
        {"trace_id": "t-spaced-empty", "finalizationStatus": "finalized"},
        [
            {
                "span_id": "s1",
                "parent_span_id": None,
                "name": "wf",
                "type": "workflow",
                "attributes": "{ }",
            }
        ],
    )
    assert spaced_empty_report["replay_ready"] is False
    assert spaced_empty_report["replay_readiness_verified"] is False

    # 4b: Whitespace `semantic.kind: " "` does NOT mask `span_type: "TOOL"`
    ws_sem_tool = {
        "span_id": "t-ws",
        "parent_span_id": "root-1",
        "name": "broken_tool",
        "span_type": "TOOL",
        "semantic": {"kind": "   "},
        "attributes": {"neatlogs.step.seq": 2},
    }
    ws_audit = audit_trace(
        {"trace_id": "t-ws-sem", "finalizationStatus": "finalized"}, [root, ws_sem_tool]
    )
    assert {c["name"]: c["status"] for c in ws_audit["checks"]}["tool_linkage_valid"] == "fail"
    assert ws_audit["replay_ready"] is False
