import json
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

import neatlogs
from neatlogs.agent import AgentClient
from neatlogs.agent.protocol import attributes, parse_config, unit

VECTORS = json.loads((Path(__file__).parents[1] / "fixtures" / "agent-v2.json").read_text())


def iso(value):
    return datetime.fromtimestamp(value, timezone.utc).isoformat()


@pytest.fixture
def connected():
    now = time.time()
    state = {
        "config": json.loads(json.dumps(VECTORS["config"])),
        "assignment": json.loads(json.dumps(VECTORS["assignment"])),
        "config_status": 200,
        "assignment_status": 200,
        "exposure_status": 200,
        "changed": False,
        "requests": [],
        "now": now,
    }
    state["config"]["expiresAt"] = state["assignment"]["expiresAt"] = iso(now + 60)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_GET(self):
            state["requests"].append((self.path, None))
            self.send_response(state["config_status"])
            self.send_header("ETag", '"config"')
            self.send_header("X-AB-Config-Expires-At", state["config"]["expiresAt"])
            self.end_headers()
            if state["config_status"] != 304:
                self.wfile.write(json.dumps(state["config"]).encode())

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state["requests"].append((self.path, body))
            if self.path.endswith("/assignments"):
                if state["changed"]:
                    state["changed"] = False
                    state["config"]["revision"] += 1
                    state["assignment"]["revision"] += 1
                    status, response = 409, {"code": "CONFIG_CHANGED"}
                else:
                    status, response = state["assignment_status"], state["assignment"]
            else:
                status = state["exposure_status"]
                response = {"code": "AB_TRACE_NOT_READY"} if status == 409 else {"accepted": True}
            self.send_response(status)
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())

    http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=http.serve_forever, daemon=True)
    thread.start()
    client = AgentClient(
        base_url=f"http://127.0.0.1:{http.server_port}", api_key="test-key", keys=["support"]
    )
    yield client, state
    client.close()
    http.shutdown()
    http.server_close()
    thread.join(timeout=2)


async def choose(client):
    return await client.choose_agent(
        "support",
        unit="session:42",
        attributes={"environment": "production", "premium": True},
        fallback={"model": "deployed"},
    )


def test_shared_vectors():
    for vector in VECTORS["completionVectors"]:
        config = json.loads(json.dumps(VECTORS["config"]))
        config["live"].update({key: vector[key] for key in ("unit", "completionPolicy")})
        if vector["valid"]:
            assert parse_config(config) == config
        else:
            with pytest.raises(ValueError):
                parse_config(config)
    for vector in VECTORS["attributeVectors"]:
        if vector["valid"]:
            assert attributes(vector["attributes"]) == vector["attributes"]
        else:
            with pytest.raises(ValueError):
                attributes(vector["attributes"])
    for vector in VECTORS["unitVectors"]:
        if vector["valid"]:
            assert unit(vector["unit"]) == vector["unit"]
        else:
            with pytest.raises(ValueError):
                unit(vector["unit"])


@pytest.mark.asyncio
@pytest.mark.parametrize("lease", ["config", "assignment"])
async def test_earlier_lease_expiry_and_transient_refresh(connected, monkeypatch, lease):
    client, state = connected
    state[lease]["expiresAt"] = iso(state["now"] + 10)
    first = await choose(client)
    assert first.reason == "assigned" and first.arm == "b"
    assert first.expires_at == pytest.approx(state["now"] + 10)
    state["config_status"] = state["assignment_status"] = 503
    await client.refresh_async()
    assert (await choose(client)).reason == "assigned"
    monkeypatch.setattr(time, "time", lambda: state["now"] + 11)
    result = await choose(client)
    assert result.reason == "unavailable" and result.params == {"model": "deployed"}


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [401, 403, 404, 302])
async def test_revocation(connected, status):
    client, state = connected
    await choose(client)
    state["config_status"] = status
    await client.refresh_async()
    assert (await choose(client)).reason == "unavailable"


@pytest.mark.asyncio
async def test_pause_and_invalid_completion_policy(connected):
    client, state = connected
    await choose(client)
    state["config"]["revision"] += 1
    state["config"]["live"]["status"] = "paused"
    await client.refresh_async()
    result = await choose(client)
    assert result.reason == "paused" and result.params == {"model": "current"}
    state["config"]["revision"] += 1
    state["config"]["live"]["unit"] = "end_user"
    await client.refresh_async()
    assert (await choose(client)).reason == "unavailable"


@pytest.mark.asyncio
async def test_changed_publication_retries_once_with_typed_attributes(connected):
    client, state = connected
    state["changed"] = True
    assert (await choose(client)).revision == 2
    assignments = [body for path, body in state["requests"] if path.endswith("/assignments")]
    assert len(assignments) == 2
    assert assignments[0] == {
        "protocolVersion": 2,
        "revision": 1,
        "unitId": "session:42",
        "attributes": {"environment": "production", "premium": True},
    }
    assert all(path.startswith("/api/v2/experiment-agents/") for path, _ in state["requests"])


@pytest.mark.asyncio
async def test_mismatched_assignment_and_unknown_config_fail_closed(connected):
    client, state = connected
    state["assignment"]["fingerprint"] = "d" * 64
    assert (await choose(client)).reason == "unavailable"
    state["assignment"]["fingerprint"] = "b" * 64
    state["config"]["futureTargeting"] = True
    assert (await choose(client)).reason == "unavailable"


@pytest.mark.asyncio
async def test_root_only_attribution_ordered_retries_and_original_completion(
    connected, monkeypatch
):
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from neatlogs._wrap_utils import get_neatlogs_provider, set_neatlogs_provider

    client, state = connected
    selected = await choose(client)
    assert client.get_diagnostics()["pending"] == 0
    assert not client.record_agent_run(selected)
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    previous = get_neatlogs_provider()
    set_neatlogs_provider(provider)
    try:
        with neatlogs.trace("root") as root:
            trace_id = format(root.get_span_context().trace_id, "032x")
            with neatlogs.trace("nested", kind="TOOL"):
                assert client.record_agent_run(selected)
                assert client.record_agent_run(selected)
        spans = exporter.get_finished_spans()
        roots = [span for span in spans if span.parent is None]
        assert roots[0].attributes["neatlogs.agent.support.assignment_id"] == selected.assignment_id
        assert roots[0].attributes["neatlogs.agent.support.fingerprint"] == selected.fingerprint
        assert all(
            "neatlogs.agent.support.assignment_id" not in span.attributes
            for span in spans
            if span is not roots[0]
        )
        state["exposure_status"] = 409
        assert not await client.flush_async()
        state["exposure_status"] = 200
        assert await client.flush_async()
        monkeypatch.setattr(time, "time", lambda: state["now"] + 61)
        assert client.complete_agent_decision(selected, trace_id)
        assert await client.flush_async()
        sent = [body for path, body in state["requests"] if path.endswith("/exposures")]
        assert len(sent) == 3 and sent[0] == sent[1]
        assert sent[2]["assignmentId"] == selected.assignment_id and sent[2]["unitComplete"] is True
        assert sent[2]["conversationComplete"] is True
        assert "arm" not in sent[2]
    finally:
        set_neatlogs_provider(previous)
        provider.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [408, 429, 500])
async def test_transient_status_preserves_original_lease(connected, status):
    client, state = connected
    original = await choose(client)
    state["config_status"] = status
    await client.refresh_async()
    assert await choose(client) == original


@pytest.mark.asyncio
async def test_concurrent_admission_and_shortened_config_lease(connected):
    import asyncio

    client, state = connected
    first, second = await asyncio.gather(choose(client), choose(client))
    assert first == second
    assert len([path for path, _ in state["requests"] if path.endswith("/assignments")]) == 1
    state["config"]["expiresAt"] = iso(state["now"] + 3)
    state["config_status"] = 304
    await client.refresh_async()
    assert (await choose(client)).expires_at == pytest.approx(state["now"] + 3)


@pytest.mark.asyncio
async def test_replay_never_admits_or_records_live_receipts(connected, monkeypatch):
    import importlib

    module = importlib.import_module("neatlogs.agent.client")
    client, state = connected
    selected = await choose(client)
    monkeypatch.setattr(module, "current_replay", lambda: object())
    assert (await choose(client)).reason == "unavailable"
    assert not client.record_agent_run(selected)
    assert client.get_diagnostics()["pending"] == 0
    assert len([path for path, _ in state["requests"] if path.endswith("/assignments")]) == 1


@pytest.mark.asyncio
async def test_natural_test_expiry_without_revision_bump(connected):
    client, state = connected
    await choose(client)
    state["config"]["live"] = None
    await client.refresh_async()
    result = await choose(client)
    assert result.reason == "current" and result.params == {"model": "current"}


@pytest.mark.asyncio
async def test_customer_conversations_keep_assignment_and_complete_conversations_only(connected):
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from neatlogs._wrap_utils import get_neatlogs_provider, set_neatlogs_provider

    client, state = connected
    state["config"]["live"].update(unit="end_user", completionPolicy="fixed_horizon")
    state["assignment"].update(unit="end_user", completionPolicy="fixed_horizon")
    telemetry = TracerProvider()
    exporter = InMemorySpanExporter()
    telemetry.add_span_processor(SimpleSpanProcessor(exporter))
    previous = get_neatlogs_provider()
    set_neatlogs_provider(telemetry)
    trace_ids = []
    try:
        for index, step in enumerate(VECTORS["customerJourney"]["traces"]):
            selected = await client.choose_agent(
                "support",
                unit=VECTORS["customerJourney"]["unitId"],
                attributes={"environment": "production", "premium": True},
                fallback={"model": "deployed"},
            )
            assert selected.reason == "assigned"
            assert (selected.unit, selected.completion_policy) == ("end_user", "fixed_horizon")
            with neatlogs.trace(
                "customer-turn", session_id=step["sessionId"], end_user_id=selected.unit_id
            ) as span:
                trace_ids.append(format(span.get_span_context().trace_id, "032x"))
                assert client.record_agent_run(selected)
                assert client.record_agent_run(selected)
            telemetry.force_flush()
            if index == 2:
                state["config"]["revision"] += 1
                state["config"]["live"].update(unit="session", completionPolicy="explicit")
                await client.refresh_async()
            if step["conversationComplete"]:
                assert client.complete_agent_decision(selected, trace_ids[-1])
                assert client.complete_agent_decision(selected, trace_ids[-1])
            assert await client.flush_async()
        roots = [span for span in exporter.get_finished_spans() if span.parent is None]
        assert len(roots) == len(set(trace_ids)) == 3
        for root, step in zip(roots, VECTORS["customerJourney"]["traces"]):
            assert root.attributes["neatlogs.session.id"] == step["sessionId"]
            assert root.attributes["neatlogs.end_user.id"] == VECTORS["customerJourney"]["unitId"]
            assert (
                root.attributes["neatlogs.agent.support.assignment_id"]
                == VECTORS["assignment"]["assignmentId"]
            )
        admissions = [body for path, body in state["requests"] if path.endswith("/assignments")]
        assert admissions == [
            {
                "protocolVersion": 2,
                "revision": 1,
                "unitId": VECTORS["customerJourney"]["unitId"],
                "attributes": {"environment": "production", "premium": True},
            }
        ]
        sent = [body for path, body in state["requests"] if path.endswith("/exposures")]
        assert len(sent) == len({item["eventId"] for item in sent}) == 5
        assert [item["conversationComplete"] for item in sent] == [False, False, True, False, True]
        for item in sent:
            assert item == {
                "protocolVersion": 2,
                "eventId": item["eventId"],
                "assignmentId": VECTORS["assignment"]["assignmentId"],
                "unitId": VECTORS["customerJourney"]["unitId"],
                "traceId": item["traceId"],
                "conversationComplete": item["conversationComplete"],
                "unitComplete": False,
            }
    finally:
        set_neatlogs_provider(previous)
        telemetry.shutdown()


@pytest.mark.asyncio
async def test_assignment_policy_must_match_publication(connected):
    client, state = connected
    state["assignment"].update(unit="end_user", completionPolicy="fixed_horizon")
    assert (await choose(client)).reason == "unavailable"
    state["assignment"].update(unit="session", completionPolicy="unknown")
    assert (await choose(client)).reason == "unavailable"
    del state["assignment"]["completionPolicy"]
    assert (await choose(client)).reason == "unavailable"
