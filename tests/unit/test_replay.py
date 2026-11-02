"""Network-free fixtures for native local replay and the shared recording protocol."""

import asyncio
import json
import os
from pathlib import Path

import pytest

import neatlogs
from neatlogs.replay import ReplayError, ReplaySession, get_replay_status, replay_model
from neatlogs.replay.contracts import parse_replay_recording, replay_key
from neatlogs.replay.session import ReplayHit, lookup_replay


def recording(entries=None):
    return {"version": 1, "sourceTraceId": "saved-trace", "entries": entries or []}


def entry(span="tool-1", *, kind="tool", name="lookup", input=None, output=None):
    return {
        "spanId": span,
        "kind": kind,
        "name": name,
        "input": input or {"value": 1},
        "output": output if output is not None else {"saved": True},
    }


@pytest.fixture(autouse=True)
def reset_replay_environment(monkeypatch):
    from neatlogs.replay import environment, session

    monkeypatch.setattr(environment, "_configured", None)
    monkeypatch.setattr(environment, "_report", None)
    monkeypatch.setattr(session, "_environment", None)
    for name in list(os.environ):
        if name.startswith("NEATLOGS_REPLAY_"):
            monkeypatch.delenv(name)


def test_ts_produced_canonical_keys_match_floats_unicode_and_type_identity():
    cases = json.loads((Path(__file__).parent / "fixtures/replay-ts-golden.json").read_text())
    for case in cases:
        assert replay_key("tool", "fixture", case["input"]) == case["key"], case["label"]
    assert replay_key("tool", "fixture", -0.0) == replay_key("tool", "fixture", 0)
    assert replay_key("tool", "fixture", 1.0) == replay_key("tool", "fixture", 1)
    assert replay_key("tool", "fixture", True) != replay_key("tool", "fixture", 1)


def test_fifo_exact_inputs_and_one_way_missing_transition():
    session = ReplaySession(recording([entry(), entry("tool-2", output={"saved": 2})]))
    with session:
        first = lookup_replay("tool", "lookup", {"value": 1.0})
        assert isinstance(first, ReplayHit) and first.entry.output == {"saved": True}
        first.entry.output["saved"] = "mutated"
        assert lookup_replay("tool", "lookup", {"value": 1}).entry.output == {"saved": 2}
        assert lookup_replay("tool", "lookup", {"value": 1}).outcome == "MISS"
        assert lookup_replay("tool", "unrelated", {}).outcome == "LIVE"
    assert session.get_status() == {
        "sourceTraceId": "saved-trace",
        "mode": "tools_only",
        "hits": 2,
        "liveCalls": 2,
        "switchedToLive": True,
        "reason": "missing_recording",
        "firstMiss": {"kind": "tool", "name": "lookup"},
        "traceIds": [],
    }
    assert get_replay_status() is None
    with pytest.raises(ReplayError, match="new ReplaySession"):
        with session:
            pass


def test_tools_only_models_stay_live_without_disabling_tool_cache():
    session = ReplaySession(recording([entry()]))
    with session:
        assert replay_model("model", {}, lambda saved: saved, lambda: "live") == "live"
        assert lookup_replay("tool", "lookup", {"value": 1}).outcome == "HIT"
    assert session.get_status()["switchedToLive"] is False


def test_exact_checkpoint_and_unseen_later_entry_cannot_replay_early():
    saved = recording(
        [entry(), entry("checkpoint", kind="model", name="model"), entry("later", name="later")]
    )
    with ReplaySession(saved, mode="until_span", until_span="checkpoint") as session:
        assert lookup_replay("tool", "lookup", {"value": 1}).outcome == "HIT"
        assert lookup_replay("model", "model", {"value": 1}).outcome == "LIVE"
        assert lookup_replay("tool", "later", {"value": 1}).outcome == "LIVE"
        assert session.get_status()["reason"] == "checkpoint"
    with ReplaySession(saved, mode="until_span", until_span="checkpoint") as session:
        assert lookup_replay("tool", "later", {"value": 1}).outcome == "MISS"
        assert session.get_status()["reason"] == "missing_recording"


@pytest.mark.parametrize(
    "defect", ["duplicate", "missing_checkpoint", "bad_mode", "tool_checkpoint", "run_trial"]
)
def test_invalid_recordings_and_options_are_rejected(defect):
    saved = recording([entry()])
    options = {}
    if defect == "duplicate":
        saved["entries"].append(entry())
    if defect == "missing_checkpoint":
        options = {"mode": "until_span", "until_span": "missing"}
    if defect == "bad_mode":
        options = {"mode": "other"}
    if defect == "tool_checkpoint":
        options = {"until_span": "tool-1"}
    if defect == "run_trial":
        options = {"run": {"runId": "not-a-uuid", "caseId": "not-a-uuid", "trial": True}}
    with pytest.raises(ReplayError):
        ReplaySession(saved, **options)


@pytest.mark.parametrize(
    "bad", [float("nan"), float("inf"), {1: "not-a-string"}, {"value": object()}, 2**53 + 1]
)
def test_invalid_json_is_not_coerced_or_logged(bad):
    with pytest.raises(ReplayError):
        parse_replay_recording(recording([entry(input=bad)]))


def test_sync_tools_stamp_one_trace_and_preserve_run_bindings(
    tracer_provider, in_memory_span_exporter
):
    neatlogs.init(
        tracer_provider=tracer_provider, disable_export=True, register_shutdown_handlers=False
    )
    calls = []

    @neatlogs.span(kind="TOOL", tool_name="lookup", replay=True)
    def tool(value):
        calls.append(value)
        return {"live": True}

    @neatlogs.span(kind="WORKFLOW")
    def workflow():
        assert tool(1) == {"saved": True}
        assert tool(2) == {"live": True}
        return tool(1)

    binding = {
        "runId": "00000000-0000-4000-8000-000000000001",
        "caseId": "00000000-0000-4000-8000-000000000002",
        "trial": 1,
    }
    session = ReplaySession(recording([entry()]), run=binding)
    assert session.run(workflow) == {"live": True}
    assert calls == [2, 1]
    spans = [
        span
        for span in in_memory_span_exporter.get_finished_spans()
        if span.name != "neatlogs.trace.complete"
    ]
    assert len(session.get_status()["traceIds"]) == 1
    assert all(span.attributes["neatlogs.origin"] == "simulation" for span in spans)
    assert all(span.attributes["neatlogs.replay.run_id"] == binding["runId"] for span in spans)
    tools = [span for span in spans if span.attributes.get("neatlogs.replay.outcome")]
    assert [span.attributes["neatlogs.replay.outcome"] for span in tools] == ["HIT", "MISS", "LIVE"]
    assert tools[0].attributes["neatlogs.replay.source_span_id"] == "tool-1"


@pytest.mark.asyncio
async def test_async_tools_factories_and_independent_sessions():
    calls = []

    @neatlogs.span(kind="TOOL", tool_name="lookup", replay=True)
    async def tool(value):
        calls.append(value)
        return {"live": True}

    @neatlogs.span(kind="TOOL", tool_name="factory", replay="awaitable")
    def factory(value):
        async def result():
            calls.append(value)
            return {"factory": True}

        return result()

    async def workflow(session, value):
        async with session:
            await asyncio.sleep(0)
            assert await tool(value) == {"saved": value}
            assert await factory(value) == {"factory": value}
            return get_replay_status()

    sessions = [
        ReplaySession(
            recording(
                [
                    entry(input={"value": value}, output={"saved": value}),
                    entry(
                        "factory", name="factory", input={"value": value}, output={"factory": value}
                    ),
                ]
            )
        )
        for value in (1, 2)
    ]
    reports = await asyncio.gather(
        *(workflow(session, value) for session, value in zip(sessions, (1, 2)))
    )
    assert calls == []
    assert [report["hits"] for report in reports] == [2, 2]
    assert get_replay_status() is None


@pytest.mark.asyncio
async def test_exceptions_cancellation_and_nested_sessions_restore_context():
    session = ReplaySession(recording())
    async with session:
        with pytest.raises(ReplayError, match="Nested"):
            with ReplaySession(recording()):
                pass
        assert lookup_replay("tool", "missing", {}).outcome == "MISS"
    assert get_replay_status() is None

    async def cancelled():
        async with ReplaySession(recording()):
            raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await cancelled()
    assert get_replay_status() is None


def test_environment_init_is_credential_free_and_report_is_private(monkeypatch, tmp_path):
    from neatlogs.replay.environment import save_environment_report

    source = tmp_path / "recording.json"
    report = tmp_path / "report.json"
    source.write_text(json.dumps(recording([entry()])))
    monkeypatch.setenv("NEATLOGS_REPLAY_FILE", str(source))
    monkeypatch.setenv("NEATLOGS_REPLAY_REPORT", str(report))
    monkeypatch.setenv("NEATLOGS_REPLAY_TRACE_ID", "saved-trace")
    monkeypatch.delenv("NEATLOGS_API_KEY", raising=False)
    neatlogs.init(disable_export=True, instrumentations=[], register_shutdown_handlers=False)
    assert lookup_replay("tool", "lookup", {"value": 1}).outcome == "HIT"
    save_environment_report()
    assert json.loads(report.read_text())["hits"] == 1
    assert report.stat().st_mode & 0o777 == 0o600
    assert neatlogs.shutdown()


def test_environment_rejects_partial_binding_before_initializing(monkeypatch, tmp_path):
    source = tmp_path / "recording.json"
    source.write_text(json.dumps(recording()))
    monkeypatch.setenv("NEATLOGS_REPLAY_FILE", str(source))
    monkeypatch.setenv("NEATLOGS_REPLAY_RUN_ID", "00000000-0000-4000-8000-000000000001")
    with pytest.raises(ReplayError):
        neatlogs.init(disable_export=True)


def test_manual_model_hit_cannot_export_new_billed_tokens():
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from neatlogs.core.span_processor import NeatlogsSpanProcessor

    provider = TracerProvider()
    sink = InMemorySpanExporter()
    provider.add_span_processor(
        NeatlogsSpanProcessor(emit_completion_markers=False, own_all_spans=True)
    )
    provider.add_span_processor(SimpleSpanProcessor(sink))
    saved = recording(
        [
            entry(kind="model", name="manual", input={}, output={"text": "fixture"}),
            entry("checkpoint"),
        ]
    )
    with ReplaySession(saved, mode="until_span", until_span="checkpoint"):
        span = provider.get_tracer("neatlogs.test").start_span(
            "manual", attributes={"neatlogs.span.kind": "llm"}
        )
        assert replay_model(
            "manual", {"value": 1}, lambda value: value, lambda: None, span=span
        ) == {"text": "fixture"}
        span.set_attribute("neatlogs.llm.token_count.prompt", 100)
        span.set_attribute("neatlogs.llm.token_count.completion", 50)
        span.set_attribute("neatlogs.llm.cost.total", 1.0)
        span.end()
    exported = sink.get_finished_spans()[0]
    assert exported.attributes["neatlogs.llm.token_count.prompt"] == 0
    assert exported.attributes["neatlogs.llm.token_count.completion"] == 0
    assert exported.attributes["neatlogs.llm.cost.total"] == 0
    provider.shutdown()
