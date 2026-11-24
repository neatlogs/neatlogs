"""Real-stack Given/When/Then BDD tests for backtracking, provenance, and trace readiness.

Zero fakes: exercises the real OpenTelemetry TracerProvider, NeatlogsSpanProcessor,
MaskingSpanExporter, OTLPSpanExporter, real gzip OTLP protobuf over loopback HTTP,
real ReplaySession, and real trace_audit CLI.
"""

from __future__ import annotations

import gzip
import json
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
    ExportTraceServiceRequest,
    ExportTraceServiceResponse,
)

import neatlogs
from neatlogs.core.backtracking import StepSequenceTracker
from neatlogs.replay import ReplayRecordedError, ReplaySession, replay_recording_from_case
from neatlogs.trace_audit import audit_trace


def _decode_any_value(val: Any) -> Any:
    kind = val.WhichOneof("value")
    if kind == "string_value":
        return val.string_value
    if kind == "bool_value":
        return val.bool_value
    if kind == "int_value":
        return val.int_value
    if kind == "double_value":
        return val.double_value
    if kind == "array_value":
        return [_decode_any_value(v) for v in val.array_value.values]
    if kind == "kvlist_value":
        return {kv.key: _decode_any_value(kv.value) for kv in val.kvlist_value.values}
    return None


class _RealOtlpCollector:
    """Real loopback HTTP OTLP protobuf receiver (zero fakes)."""

    def __init__(self) -> None:
        self.spans: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        collector = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length)
                if self.headers.get("Content-Encoding", "").lower() == "gzip":
                    body = gzip.decompress(body)
                req = ExportTraceServiceRequest()
                req.ParseFromString(body)
                decoded: list[dict[str, Any]] = []
                for rs in req.resource_spans:
                    for ss in rs.scope_spans:
                        for s in ss.spans:
                            attrs = {kv.key: _decode_any_value(kv.value) for kv in s.attributes}
                            events = [
                                {
                                    "name": ev.name,
                                    "attributes": {
                                        kv.key: _decode_any_value(kv.value) for kv in ev.attributes
                                    },
                                }
                                for ev in s.events
                            ]
                            decoded.append(
                                {
                                    "trace_id": s.trace_id.hex(),
                                    "span_id": s.span_id.hex(),
                                    "parent_span_id": (
                                        s.parent_span_id.hex() if s.parent_span_id else None
                                    ),
                                    "name": s.name,
                                    "span_type": attrs.get("neatlogs.span.kind")
                                    or attrs.get("openinference.span.kind")
                                    or "",
                                    "start_time": datetime.fromtimestamp(
                                        s.start_time_unix_nano / 1_000_000_000, tz=timezone.utc
                                    ).isoformat(),
                                    "end_time": datetime.fromtimestamp(
                                        s.end_time_unix_nano / 1_000_000_000, tz=timezone.utc
                                    ).isoformat(),
                                    "status": {
                                        "code": s.status.code,
                                        "description": s.status.message,
                                    },
                                    "attributes": attrs,
                                    "events": events,
                                }
                            )
                with collector._lock:
                    collector.spans.extend(decoded)
                resp_bytes = ExportTraceServiceResponse().SerializeToString()
                self.send_response(200)
                self.send_header("Content-Type", "application/x-protobuf")
                self.send_header("Content-Length", str(len(resp_bytes)))
                self.end_headers()
                self.wfile.write(resp_bytes)

            def log_message(self, format, *args):
                return

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self._server.server_address[1]
        self.endpoint = f"http://127.0.0.1:{self.port}"
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def spans_for_trace(self, trace_id: str) -> list[dict[str, Any]]:
        with self._lock:
            return [s for s in self.spans if s["trace_id"] == trace_id]

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=2)


@pytest.fixture
def real_otlp_pipeline(monkeypatch):
    """Boot a real loopback OTLP receiver and configure neatlogs against it."""
    monkeypatch.setenv("NEATLOGS_DISABLE_EXPORT", "false")
    collector = _RealOtlpCollector()
    neatlogs.shutdown()
    neatlogs.init(
        api_key="bdd-real-otlp-key",
        endpoint=collector.endpoint,
        workflow_name="bdd-refund-agent",
        instrumentations=[],
        environment="staging",
        release="sha-bdd-20261121",
        flush_interval=0.2,
        register_shutdown_handlers=False,
    )
    try:
        yield collector
    finally:
        neatlogs.flush()
        neatlogs.shutdown()
        collector.close()


def test_bdd_scenario_1_monotonic_step_seq_provenance_checkpoints_and_audit(
    real_otlp_pipeline, tmp_path: Path
):
    """Given a real OTLP pipeline, When a multi-step RAG agent runs with checkpoints and diagnostics,
    Then OTLP protobuf spans carry monotonic step_seq, provenance, sanitized diagnostics,
    retriever freshness metadata, and pass `neatlogs doctor --file`."""

    @neatlogs.span(kind="RETRIEVER", name="fetch_policy", replay=True)
    def fetch_policy(query: str) -> list[dict[str, Any]]:
        return [
            {
                "id": "pol-001",
                "content": "Refunds are allowed within 30 days.",
                "score": 0.94,
                "updated_at": "2026-11-20T00:00:00Z",
            }
        ]

    @neatlogs.span(kind="TOOL", tool_name="issue_refund", replay=True)
    def issue_refund(order_id: str, amount: int) -> dict[str, Any]:
        return {"order_id": order_id, "amount": amount, "status": "refunded"}

    @neatlogs.span(
        kind="WORKFLOW",
        name="refund_workflow",
        prompt_version="refund-v4",
        session_id="conv-session-42",
        end_user_id="user-99",
    )
    def run_agent(user_request: str) -> dict[str, Any]:
        trace_id = neatlogs.current_trace_id()
        docs = fetch_policy(user_request)
        cid = neatlogs.checkpoint(
            "after_retrieval",
            state={"order_id": "ORD-1", "docs": len(docs)},
            state_delta={"eligible": True},
        )
        assert cid == "after_retrieval"
        reported = neatlogs.report_diagnostic(
            "stale_data",
            "policy lookup api_key=sk-secret-12345 flagged age check",
            severity="info",
            details={"authorization": "Bearer super-secret-token", "doc_id": "pol-001"},
        )
        assert reported is True
        receipt = issue_refund("ORD-1", 50)
        return {
            "trace_id": trace_id,
            "reply": f"Refunded {receipt['amount']} for {receipt['order_id']}",
        }

    # When: execute the real agent workflow and flush over real HTTP protobuf
    result = run_agent("Please refund order ORD-1")
    neatlogs.flush()
    trace_id = result["trace_id"]
    exported = real_otlp_pipeline.spans_for_trace(trace_id)

    # Then: spans arrived over real OTLP with completion marker & monotonic step_seq
    markers = [s for s in exported if s["name"] == "neatlogs.trace.complete"]
    assert len(markers) == 1
    work_spans = [s for s in exported if s["name"] != "neatlogs.trace.complete"]
    work_spans.sort(key=lambda s: s["attributes"].get("neatlogs.step.seq", 0))
    seqs = [s["attributes"]["neatlogs.step.seq"] for s in work_spans]
    assert seqs == [1, 2, 3]

    root = next(s for s in work_spans if s["parent_span_id"] is None)
    assert root["attributes"]["deployment.environment.name"] == "staging"
    assert root["attributes"]["neatlogs.provenance.release"] == "sha-bdd-20261121"
    assert root["attributes"]["neatlogs.prompt.version"] == "refund-v4"
    assert root["attributes"]["neatlogs.checkpoint.id"] == "after_retrieval"
    assert root["attributes"]["neatlogs.checkpoint.state_digest"].startswith("sha256:")
    assert json.loads(root["attributes"]["neatlogs.checkpoint.state_delta"]) == {"eligible": True}

    # Credentials scrubbed from diagnostic message & details
    assert "sk-secret-12345" not in root["attributes"]["neatlogs.diagnostic.message"]
    assert "super-secret-token" not in root["attributes"]["neatlogs.diagnostic.details"]
    assert any(ev["name"] == "neatlogs.diagnostic" for ev in root["events"])

    # Retriever top-level updated_at folded into metadata
    ret_span = next(s for s in work_spans if s["name"] == "fetch_policy")
    ret_meta = json.loads(ret_span["attributes"]["neatlogs.retriever.documents.0.metadata"])
    assert ret_meta["updated_at"] == "2026-11-20T00:00:00Z"

    # Real-trace readiness audit passes via API and via real CLI subprocess
    audit = audit_trace(
        {"trace_id": trace_id, "session_id": "conv-session-42"},
        exported,
        expect_session=True,
        expect_retriever=True,
    )
    assert audit["grade"] == "ready"
    assert audit["jev_ready"] is True
    assert audit["replay_ready"] is True
    assert audit["score"] == 100

    trace_file = tmp_path / "healthy_trace.json"
    trace_file.write_text(
        json.dumps(
            {"trace": {"trace_id": trace_id, "session_id": "conv-session-42"}, "spans": exported}
        ),
        encoding="utf-8",
    )
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "neatlogs",
            "doctor",
            "--file",
            str(trace_file),
            "--expect-session",
            "--expect-retriever",
            "--json",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    cli_report = json.loads(proc.stdout)
    assert cli_report["grade"] == "ready"
    assert cli_report["score"] == 100


def test_bdd_scenario_2_and_3_backtrack_to_any_span_and_fork_with_override(real_otlp_pipeline):
    """Given a recorded 3-step RAG trace (RETRIEVER -> TOOL -> TOOL),
    When we backtrack via until_span AND fork with overrides={retriever_span_id: fresh_policy},
    Then prior steps replay without live calls, the override injects fresh data, downstream steps
    run live, and fork lineage attributes are exported over real OTLP."""

    live_counts = {"fetch_policy": 0, "check_order": 0, "decide_refund": 0}

    @neatlogs.span(kind="RETRIEVER", name="fetch_policy", replay=True)
    def fetch_policy(topic: str) -> list[dict[str, Any]]:
        live_counts["fetch_policy"] += 1
        return [{"id": "pol-old", "days_allowed": 7, "updated_at": "2024-01-01"}]

    @neatlogs.span(kind="TOOL", tool_name="check_order", replay=True)
    def check_order(order_id: str) -> dict[str, Any]:
        live_counts["check_order"] += 1
        return {"order_id": order_id, "age_days": 14}

    @neatlogs.span(kind="TOOL", tool_name="decide_refund", replay=True)
    def decide_refund(days_allowed: int, age_days: int) -> dict[str, Any]:
        live_counts["decide_refund"] += 1
        return {"approved": age_days <= days_allowed}

    @neatlogs.span(kind="WORKFLOW", name="refund_pipeline")
    def run_pipeline(order_id: str) -> dict[str, Any]:
        tid = neatlogs.current_trace_id()
        docs = fetch_policy("refunds")
        order = check_order(order_id)
        decision = decide_refund(docs[0]["days_allowed"], order["age_days"])
        return {"trace_id": tid, "approved": decision["approved"]}

    # Given: initial live run fails because policy says 7 days and order is 14 days old
    baseline = run_pipeline("ORD-14")
    neatlogs.flush()
    assert baseline["approved"] is False
    assert live_counts == {"fetch_policy": 1, "check_order": 1, "decide_refund": 1}

    recorded_spans = [
        s
        for s in real_otlp_pipeline.spans_for_trace(baseline["trace_id"])
        if s["name"] != "neatlogs.trace.complete"
    ]
    frozen_case = {
        "source": {"kind": "trace", "traceId": baseline["trace_id"]},
        "context": {"spans": recorded_spans},
    }
    extracted = replay_recording_from_case(frozen_case, include_steps=True)
    recording = extracted["recording"]
    assert extracted["skippedTools"] == 0
    assert recording.version == 2
    assert [e.kind for e in recording.entries] == ["retriever", "tool", "tool"]

    policy_span_id = recording.entries[0].span_id
    decide_span_id = recording.entries[2].span_id

    # Scenario 2: Backtrack to decide_refund — RETRIEVER and check_order replay with 0 live calls
    live_counts = {"fetch_policy": 0, "check_order": 0, "decide_refund": 0}
    with ReplaySession(recording, mode="until_span", until_span=decide_span_id) as session:
        replay_res = run_pipeline("ORD-14")
        status = session.get_status()
        lineage = session.get_fork_lineage()
    neatlogs.flush()

    assert replay_res["approved"] is False
    assert live_counts == {"fetch_policy": 0, "check_order": 0, "decide_refund": 1}
    assert status["hits"] == 2
    assert status["liveCalls"] == 1
    assert status["reason"] == "checkpoint"
    assert lineage["forkedFromTraceId"] == baseline["trace_id"]
    assert lineage["forkedFromSpanId"] == decide_span_id

    # Scenario 3: Fork at fetch_policy with override injecting fresh 30-day policy
    live_counts = {"fetch_policy": 0, "check_order": 0, "decide_refund": 0}
    fresh_policy = [{"id": "pol-new", "days_allowed": 30, "updated_at": "2026-11-20"}]
    with ReplaySession(
        recording,
        mode="until_span",
        until_span=policy_span_id,
        overrides={policy_span_id: fresh_policy},
    ) as fork_session:
        fork_res = run_pipeline("ORD-14")
        fork_status = fork_session.get_status()
        fork_lineage = fork_session.get_fork_lineage()
    neatlogs.flush()

    assert fork_res["approved"] is True
    assert live_counts == {"fetch_policy": 0, "check_order": 1, "decide_refund": 1}
    assert fork_status["reason"] == "override_fork"
    assert fork_status["overridesApplied"] == 1
    assert fork_lineage["overrideApplied"] is True

    fork_spans = [
        s
        for s in real_otlp_pipeline.spans_for_trace(fork_res["trace_id"])
        if s["name"] != "neatlogs.trace.complete"
    ]
    forked_policy = next(s for s in fork_spans if s["name"] == "fetch_policy")
    assert (
        forked_policy["attributes"]["neatlogs.replay.forked_from_trace_id"] == baseline["trace_id"]
    )
    assert forked_policy["attributes"]["neatlogs.replay.forked_from_span_id"] == policy_span_id
    assert forked_policy["attributes"]["neatlogs.replay.override_applied"] is True


def test_bdd_scenario_4_error_span_replay_and_recovery(real_otlp_pipeline):
    """Given a recorded trace where step 2 failed with an error,
    When extracted with include_errors=True,
    Then backtracking to step 2 recovers cleanly AND replaying past step 2 raises ReplayRecordedError.
    """

    should_fail = {"flag": True}
    calls = {"fetch": 0, "charge": 0}

    @neatlogs.span(kind="RETRIEVER", name="load_customer", replay=True)
    def load_customer(cid: str) -> dict[str, Any]:
        calls["fetch"] += 1
        return {"cid": cid, "tier": "pro"}

    @neatlogs.span(kind="TOOL", tool_name="charge_card", replay=True)
    def charge_card(cid: str) -> dict[str, Any]:
        calls["charge"] += 1
        if should_fail["flag"]:
            raise RuntimeError("GatewayTimeout api_key=sk-leak-999")
        return {"charged": True}

    @neatlogs.span(kind="TOOL", tool_name="send_receipt", replay=True)
    def send_receipt(cid: str) -> dict[str, Any]:
        return {"sent": True}

    @neatlogs.span(kind="WORKFLOW", name="checkout")
    def run_checkout(cid: str) -> dict[str, Any]:
        tid = neatlogs.current_trace_id()
        load_customer(cid)
        charge_card(cid)
        return {"trace_id": tid, "ok": True}

    with pytest.raises(RuntimeError, match="GatewayTimeout"):
        run_checkout("C-1")
    neatlogs.flush()

    all_spans = [s for s in real_otlp_pipeline.spans if s["name"] != "neatlogs.trace.complete"]
    failed_trace_id = all_spans[-1]["trace_id"]
    trace_spans = [s for s in all_spans if s["trace_id"] == failed_trace_id]

    extracted = replay_recording_from_case(
        {
            "source": {"kind": "trace", "traceId": failed_trace_id},
            "context": {"spans": trace_spans},
        },
        include_steps=True,
        include_errors=True,
    )
    recording = extracted["recording"]
    assert recording.version == 2
    assert len(recording.entries) == 2
    assert [e.kind for e in recording.entries] == ["retriever", "tool"]
    assert recording.entries[1].error is not None
    assert "sk-leak-999" not in recording.entries[1].error["message"]

    # Backtracking to failing step 2 skips live fetch and recovers live at charge_card
    should_fail["flag"] = False
    calls = {"fetch": 0, "charge": 0}
    with ReplaySession(recording, mode="until_span", until_span=recording.entries[1].span_id):
        recovered = run_checkout("C-1")
    assert recovered["ok"] is True
    assert calls == {"fetch": 0, "charge": 1}

    # Replaying in tools_only mode hits the recorded error deterministically without calling live charge
    calls = {"fetch": 0, "charge": 0}
    with ReplaySession(recording, mode="tools_only"):
        with pytest.raises(ReplayRecordedError, match="GatewayTimeout"):
            run_checkout("C-1")
    assert calls == {"fetch": 0, "charge": 0}


def test_bdd_scenario_5_broken_trace_audit_catches_all_defects(tmp_path: Path):
    """Given a broken trace missing root output, with session_id==trace_id, an orphan span,
    and missing retriever timestamps, When audited via CLI, Then exit code is 1 with exact check IDs.
    """
    broken = {
        "trace": {"trace_id": "abc123", "session_id": "abc123", "finalizationStatus": "pending"},
        "spans": [
            {
                "span_id": "s1",
                "parent_span_id": None,
                "name": "chat",
                "span_type": "WORKFLOW",
                "input": "hello",
                "output": "",
                "attributes": {},
            },
            {
                "span_id": "s2",
                "parent_span_id": "missing-parent",
                "name": "search_docs",
                "span_type": "RETRIEVER",
                "input": {"query": "refunds"},
                "output": [{"id": "d1", "content": "text without timestamp"}],
                "attributes": {},
            },
        ],
    }
    bad_file = tmp_path / "broken.json"
    bad_file.write_text(json.dumps(broken), encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "neatlogs",
            "doctor",
            "--file",
            str(bad_file),
            "--expect-session",
            "--expect-retriever",
            "--json",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 1
    report = json.loads(proc.stdout)
    assert report["grade"] == "incomplete"
    assert report["jev_ready"] is False
    by_name = {c["name"]: c["status"] for c in report["checks"]}
    assert by_name["single_workflow_root"] == "fail"
    assert by_name["root_io_present"] == "fail"
    assert by_name["trace_finalized"] == "fail"
    assert by_name["session_hygiene"] == "fail"
    assert by_name["retriever_freshness"] == "warn"
    assert len(report["recommendations"]) >= 4

    # Separate `trace audit` subcommand is intentionally removed in favor of `doctor --trace/--file`
    removed = subprocess.run(
        [sys.executable, "-m", "neatlogs", "trace", "audit", "--file", str(bad_file)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert removed.returncode == 4


def test_bdd_scenario_6_incremental_overhead_and_bounded_tracker():
    """Given a real NeatlogsSpanProcessor pipeline, When running 300 spans with backtracking hooks,
    Then incremental hot-path overhead stays bounded and StepSequenceTracker never exceeds its cap.
    """
    tracker = StepSequenceTracker(max_traces=50)
    for tid in range(200):
        assert tracker.next_seq(tid) == 1
        assert tracker.next_seq(tid) == 2
    assert len(tracker) == 50

    neatlogs.shutdown()
    neatlogs.init(
        api_key="perf-key",
        disable_export=True,
        instrumentations=[],
        environment="test",
        release="perf-sha",
        register_shutdown_handlers=False,
    )
    try:

        @neatlogs.span(kind="TOOL", tool_name="plain_op")
        def plain_op(x: int) -> int:
            return x + 1

        @neatlogs.span(kind="TOOL", tool_name="checkpointed_op", checkpoint_id="cp-1")
        def checkpointed_op(x: int) -> int:
            neatlogs.checkpoint("cp-inner", state={"x": x}, state_delta={"y": x + 1})
            return x + 1

        # Warm up
        for i in range(20):
            plain_op(i)
            checkpointed_op(i)

        base_times = []
        hook_times = []
        for i in range(250):
            t0 = time.perf_counter()
            plain_op(i)
            base_times.append((time.perf_counter() - t0) * 1000.0)

            t1 = time.perf_counter()
            checkpointed_op(i)
            hook_times.append((time.perf_counter() - t1) * 1000.0)

        base_p50 = sorted(base_times)[len(base_times) // 2]
        hook_p50 = sorted(hook_times)[len(hook_times) // 2]
        incremental_ms = hook_p50 - base_p50
        assert (
            incremental_ms < 1.0
        ), f"Incremental backtracking hook overhead too high: {incremental_ms:.3f}ms"

        # Huge 5 MB workflow state must stop streaming early at MAX_STATE_DIGEST_BYTES
        from neatlogs.core.backtracking import MAX_STATE_DIGEST_BYTES, _bounded_json_bytes

        huge_state = {"items": [{"payload": "x" * 10_000} for _ in range(500)]}
        t_huge = time.perf_counter()
        raw, truncated = _bounded_json_bytes(huge_state, max_bytes=MAX_STATE_DIGEST_BYTES)
        huge_ms = (time.perf_counter() - t_huge) * 1000.0
        assert truncated is True
        assert len(raw) == MAX_STATE_DIGEST_BYTES
        assert huge_ms < 20.0, f"Streaming bounded serializer took too long: {huge_ms:.2f}ms"
    finally:
        neatlogs.shutdown()
