import importlib
import json
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from uuid import uuid4

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider

import neatlogs
from neatlogs.ab_test import AbTestClient
from neatlogs.ab_test.protocol import assignment_bucket
from neatlogs.agent import AgentClient
from neatlogs.agent.protocol import AgentDecision

client_module = importlib.import_module("neatlogs.ab_test.client")
shared = importlib.import_module("neatlogs.ab_test")


def iso(seconds=0):
    return datetime.fromtimestamp(time.time() + seconds, timezone.utc).isoformat()


def config():
    return {
        "protocolVersion": 1,
        "id": str(uuid4()),
        "key": "example",
        "revision": 1,
        "allocationVersion": 1,
        "salt": "0123456789abcdef" * 4,
        "unit": "session",
        "status": "running",
        "enabled": True,
        "defaultArm": "a",
        "expiresAt": iso(60),
        "endsAt": iso(3600),
        "refreshAfterSeconds": 60,
        "assignment": "sha256_utf8_salt_colon_unit_first_uint32_be_mod_10000",
        "arms": [
            {"key": "a", "label": "Control", "weight": 4105, "params": {"model": "small"}},
            {"key": "b", "label": "Treatment", "weight": 5895, "params": {"model": "candidate"}},
        ],
        "etag": '"revision-1"',
    }


@pytest.fixture
def server():
    state = {
        "config": config(),
        "status": 200,
        "requests": [],
        "exposure_status": 200,
        "exposure_body": {"accepted": True},
    }

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_GET(self):
            state["requests"].append(("GET", self.path, dict(self.headers), None))
            self.send_response(state["status"])
            self.send_header("X-AB-Config-Expires-At", iso(60))
            self.end_headers()
            if state["status"] != 304:
                self.wfile.write(json.dumps(state["config"]).encode())

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state["requests"].append(("POST", self.path, dict(self.headers), payload))
            status, body = (
                state["exposure_responder"](self.path, payload)
                if state.get("exposure_responder")
                else (state["exposure_status"], state["exposure_body"])
            )
            self.send_response(status)
            self.end_headers()
            self.wfile.write(json.dumps(body).encode())

    http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=http.serve_forever, daemon=True)
    thread.start()
    state["url"] = f"http://127.0.0.1:{http.server_port}"
    yield state
    http.shutdown()
    http.server_close()
    thread.join(timeout=2)


@contextmanager
def connected(server):
    client = AbTestClient(base_url=server["url"], api_key="project-test-key", keys=["example"])
    client.refresh()
    try:
        yield client
    finally:
        client.close()


@pytest.mark.parametrize("unit,bucket", [("user-1", 2509), ("用户-α", 2445), ("session:42", 4105)])
def test_shared_assignment_vectors(unit, bucket):
    assert assignment_bucket("0123456789abcdef" * 4, unit) == bucket


def test_assignment_is_cached_stable_and_does_no_network(server):
    with connected(server) as client:
        count = len(server["requests"])
        for _ in range(10):
            result = client.assign("example", unit="session:42")
            assert (result.arm, result.params, result.reason) == (
                "b",
                {"model": "candidate"},
                "assigned",
            )
        result.params["model"] = "changed"
        assert client.assign("example", unit="session:42").params["model"] == "candidate"
        assert len(server["requests"]) == count
        assert server["requests"][0][1] == "/api/v1/ab-tests/example/config"
        assert server["requests"][0][2]["Authorization"] == "Bearer project-test-key"


def test_refresh_304_preserves_valid_cache_on_transient_failure(server, monkeypatch):
    with connected(server) as client:
        server["status"] = 304
        client.refresh()
        assert server["requests"][-1][2]["If-None-Match"] == '"revision-1"'
        assert client.assign("example", unit="session:42").arm == "b"
        with monkeypatch.context() as patch:
            now = time.time()
            patch.setattr(client_module.time, "time", lambda: now + 61)
            assert client.assign(
                "example", unit="session:42", fallback={"model": "safe"}
            ).params == {"model": "safe"}
        server["status"] = 503
        client.refresh()
        assert client.assign("example", unit="session:42").reason == "assigned"
        server["status"] = 200
        server["config"]["revision"] = 0
        client.refresh()
        assert client.assign("example", unit="session:42").arm == "a"


@pytest.mark.parametrize("failure", [408, 429, 500, 503, "disconnect", "interrupted"])
def test_transient_refresh_never_extends_server_expiry(server, monkeypatch, failure):
    import requests

    server["config"]["expiresAt"] = iso(20)
    expires_at = client_module.timestamp(server["config"]["expiresAt"])
    with connected(server) as client:
        if failure in {"disconnect", "interrupted"}:

            def disconnected(*args, **kwargs):
                if failure == "interrupted":
                    raise requests.exceptions.ChunkedEncodingError("interrupted")
                raise requests.ConnectionError("disconnected")

            monkeypatch.setattr(client, "_request", disconnected)
        else:
            server["status"] = failure
        monkeypatch.setattr(client_module.time, "time", lambda: expires_at - 0.001)
        client.refresh()
        assert client.choose("example", unit="session:42").reason == "assigned"
        monkeypatch.setattr(client_module.time, "time", lambda: expires_at)
        client.refresh()
        assert client.choose("example", unit="session:42", fallback={"model": "safe"}).params == {
            "model": "safe"
        }


@pytest.mark.parametrize("failure", [401, 403, 404, 302, "invalid"])
def test_revoked_or_invalid_refresh_discards_unexpired_cache(server, failure):
    with connected(server) as client:
        if failure == "invalid":
            server["config"]["revision"] = 0
        else:
            server["status"] = failure
        client.refresh()
        assert client.choose("example", unit="session:42").reason == "unavailable"


@pytest.mark.parametrize("kind", ["WORKFLOW", "CHAIN"])
def test_nested_assignment_tags_neatlogs_root_without_tagging_host(server, kind):
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from neatlogs._wrap_utils import get_neatlogs_provider, set_neatlogs_provider

    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    previous = get_neatlogs_provider()
    host_provider = TracerProvider()
    set_neatlogs_provider(provider)
    try:
        with host_provider.get_tracer("host").start_as_current_span("host") as host:
            with neatlogs.trace("outer", kind=kind):
                with neatlogs.trace("nested", kind="TOOL"):
                    with connected(server) as client:
                        decision = client.assign("example", unit="session:42")
                        assert decision.reason == "assigned"
                assert "neatlogs.ab.example" not in host.attributes
        spans = exporter.get_finished_spans()
        roots = [span for span in spans if span.parent is None]
        assert len(roots) == 1
        assert roots[0].attributes["neatlogs.ab.example"] == "b"
        assert all(
            "neatlogs.ab.example" not in span.attributes for span in spans if span is not roots[0]
        )
    finally:
        set_neatlogs_provider(previous)
        provider.shutdown()
        host_provider.shutdown()


def test_disabled_lifecycle_serves_explicit_default_without_exposure(server):
    server["config"].update(status="shipped", enabled=False, defaultArm="b")
    server["config"]["arms"][0]["weight"] = 0
    server["config"]["arms"][1]["weight"] = 10000
    with connected(server) as client:
        assert client.assign("example", unit="user-1").reason == "default"
        assert client.assign("example", unit="user-1").arm == "b"
        assert client.get_diagnostics()["pending"] == 0
        client.close()
        assert client.assign("example", unit="user-1").reason == "unavailable"


def test_auto_stamp_dedup_and_explicit_closure_wait_for_trace_finalization(server, monkeypatch):
    provider = TracerProvider()
    span = provider.get_tracer("test").start_span("owned")
    monkeypatch.setattr(
        client_module, "active_neatlogs_context", lambda: trace.set_span_in_context(span)
    )
    trace_id = format(span.get_span_context().trace_id, "032x")
    with connected(server) as client:
        assigned = client.assign("example", unit="session:42")
        client.assign("example", unit="session:42")
        assert span.attributes["neatlogs.ab.example"] == assigned.arm
        assert client.complete_unit(
            "example", unit_id="session:42", arm=assigned.arm, trace_id=trace_id
        )
        assert client.get_diagnostics()["pending"] == 2
        server["exposure_status"] = 409
        server["exposure_body"] = {"code": "AB_TRACE_NOT_READY"}
        assert not client.flush()
        first_event = server["requests"][-1][3]["eventId"]
        assert client.get_diagnostics()["pending"] == 2
        server["exposure_status"] = 200
        server["exposure_body"] = {"accepted": True}
        assert client.flush()
        assert server["requests"][-2][3]["eventId"] == first_event
        assert server["requests"][-2][3]["unitComplete"] is False
        assert server["requests"][-1][3]["unitComplete"] is True
        assert all(row[3]["source"] == "sdk" for row in server["requests"] if row[0] == "POST")
    span.end()
    provider.shutdown()


def test_choose_does_not_count_abandoned_traffic_but_records_started_failure(server, monkeypatch):
    provider = TracerProvider()
    span = provider.get_tracer("test").start_span("failed-agent")
    monkeypatch.setattr(
        client_module, "active_neatlogs_context", lambda: trace.set_span_in_context(span)
    )
    trace_id = format(span.get_span_context().trace_id, "032x")
    with connected(server) as client:
        abandoned = client.choose("example", unit="abandoned")
        assert abandoned.reason == "assigned"
        assert client.get_diagnostics()["pending"] == 0
        decision = client.choose("example", unit="session:42")
        assert decision.arm == "b"
        assert client.record_run(decision)
        assert client.get_diagnostics()["pending"] == 1
        assert client.flush()
        assert client.complete_decision(decision, trace_id)
        assert client.flush()
        receipts = [row[3] for row in server["requests"] if row[0] == "POST"]
        assert len(receipts) == 2
        assert all(row["unitId"] == "session:42" for row in receipts)
        assert [row["unitComplete"] for row in receipts] == [False, True]
    span.end()
    provider.shutdown()


def test_external_exposure_conflict_is_observable_and_bounded(server):
    with connected(server) as client:
        event = str(uuid4())
        assert client.log_exposure(
            "example", unit_id="user", arm="a", trace_id="trace", event_id=event
        )
        assert not client.log_exposure(
            "example", unit_id="other", arm="b", trace_id="other", event_id=event
        )
        server["exposure_status"] = 409
        server["exposure_body"] = {"code": "AB_UNIT_CONFLICT"}
        assert not client.flush()
        assert server["requests"][-1][3]["source"] == "external"
        assert client.get_diagnostics() == {"pending": 0, "rejected": 2}


@pytest.mark.parametrize(
    "delayed", [(409, {"code": "AB_TRACE_NOT_READY"}), (200, {"accepted": False})]
)
def test_delayed_unit_preserves_closure_but_does_not_block_other_units(server, delayed):
    ready = False
    server["exposure_responder"] = lambda _path, body: (
        delayed if body["unitId"] == "blocked" and not ready else (200, {"accepted": True})
    )
    with connected(server) as client:
        for unit, complete in [
            ("blocked", False),
            ("blocked", True),
            ("independent", False),
            ("independent", True),
        ]:
            assert client.log_exposure(
                "example", unit_id=unit, arm="a", trace_id=f"{unit}-trace", unit_complete=complete
            )
        assert not client.flush()
        sent = [row[3] for row in server["requests"] if row[0] == "POST"]
        assert [(row["unitId"], row["unitComplete"]) for row in sent] == [
            ("blocked", False),
            ("independent", False),
            ("independent", True),
        ]
        assert client.get_diagnostics() == {"pending": 2, "rejected": 0}
        event_id = sent[0]["eventId"]
        ready = True
        assert client.flush()
        later = [row[3] for row in server["requests"] if row[0] == "POST"][3:]
        assert later[0]["eventId"] == event_id
        assert [row["unitComplete"] for row in later] == [False, True]


def test_agent_delayed_unit_blocks_renewed_lease_closure_but_not_other_keys(server):
    ready = False
    server["exposure_responder"] = lambda _path, body: (
        (409, {"code": "AB_TRACE_NOT_READY"})
        if body["traceId"] == "old-run" and not ready
        else (200, {"accepted": True})
    )
    client = AgentClient(base_url=server["url"], api_key="test-key", keys=["example", "other"])
    try:
        for key, unit, trace_id, complete in [
            ("example", "blocked", "old-run", False),
            ("example", "blocked", "renewed-closure", True),
            ("example", "independent", "other-unit", False),
            ("other", "blocked", "other-key", False),
        ]:
            decision = AgentDecision(
                key,
                unit,
                "a",
                {},
                "assigned",
                assignment_id=str(uuid4()),
                unit="session",
                completion_policy="explicit",
            )
            assert client._queue_agent(decision, trace_id, complete)
        assert not client.flush()
        sent = [(row[1], row[3]) for row in server["requests"] if row[0] == "POST"]
        assert [body["traceId"] for _path, body in sent] == ["old-run", "other-unit", "other-key"]
        assert all(path.startswith("/api/v2/experiment-agents/") for path, _body in sent)
        assert client.get_diagnostics() == {"pending": 2, "rejected": 0}
        first = sent[0][1]
        ready = True
        assert client.flush()
        later = [row[3] for row in server["requests"] if row[0] == "POST"][3:]
        assert later[0] == first
        assert later[1]["assignmentId"] != first["assignmentId"]
        assert later[1]["conversationComplete"] is True
    finally:
        client.close()


@pytest.mark.parametrize("status", [408, 429, 500, 503, 409])
def test_global_delivery_failures_still_stop_independent_units(server, status):
    server["exposure_status"] = status
    server["exposure_body"] = {}
    with connected(server) as client:
        for unit in ["first", "other"]:
            assert client.log_exposure("example", unit_id=unit, arm="a", trace_id=f"{unit}-trace")
        assert not client.flush()
        assert len([row for row in server["requests"] if row[0] == "POST"]) == 1
        assert client.get_diagnostics() == {"pending": 2, "rejected": 0}


def test_shared_init_and_flush_deliver_after_telemetry(server, monkeypatch):
    shared.close_ab_tests()
    monkeypatch.setattr(
        importlib.import_module("neatlogs.init"),
        "get_session_config",
        lambda: {"_api_key": "project-test-key", "_base_url": server["url"]},
    )
    client = neatlogs.configure_ab_tests(keys=["example"])
    try:
        assert neatlogs.ab_test("example", unit="user-1").arm == "a"
        assert neatlogs.log_exposure("example", unit_id="user", arm="b", trace_id="trace")
        assert neatlogs.flush()
        assert client.get_diagnostics()["pending"] == 0
    finally:
        shared.close_ab_tests()
    assert neatlogs.ab_test("example", unit="user-1").reason == "unavailable"


def test_init_owns_cache_lifecycle_and_disabled_export_never_fetches(server, monkeypatch):
    monkeypatch.setenv("NEATLOGS_DISABLE_EXPORT", "false")
    options = dict(
        api_key="project-test-key",
        endpoint=server["url"],
        workflow_name="ab-sdk",
        instrumentations=[],
        register_shutdown_handlers=False,
        ab_test_keys=["example"],
    )
    neatlogs.init(**options)
    try:
        assert neatlogs.ab_test("example", unit="session:42").arm == "b"
        count = len(server["requests"])
        neatlogs.init(**options)
        assert len(server["requests"]) == count
    finally:
        assert neatlogs.shutdown()
    assert neatlogs.ab_test("example", unit="session:42").reason == "unavailable"
    neatlogs.init(**options, disable_export=True)
    try:
        assert len(server["requests"]) == count
        assert neatlogs.ab_test("example", unit="session:42").reason == "unavailable"
    finally:
        assert neatlogs.shutdown()


def test_exposure_queue_is_bounded_and_large_settings_fail_closed(server):
    with connected(server) as client:
        for index in range(1000):
            assert client.log_exposure("example", unit_id=str(index), arm="a", trace_id=str(index))
        assert not client.log_exposure("example", unit_id="overflow", arm="a", trace_id="overflow")
        assert client.get_diagnostics() == {"pending": 1000, "rejected": 1}
        server["config"]["padding"] = "x" * 70000
        client.refresh()
        assert client.assign("example", unit="session:42").reason == "unavailable"


@pytest.mark.parametrize(
    "url",
    [
        "https://user:password@example.com",
        "https://example.com/path",
        "http://example.com",
        "https://example.com?secret=yes",
    ],
)
def test_rejects_unsafe_origins(url):
    with pytest.raises(ValueError):
        AbTestClient(base_url=url, api_key="key", keys=["example"])


def rollout(state, percentage=50, epoch=2):
    settings = state["config"]
    settings.update(
        phase="rollout",
        rolloutPercentage=percentage,
        allocationVersion=epoch,
        status="ramping" if percentage == 50 else "shipped",
        enabled=True,
        defaultArm="a",
        revision=epoch,
        expiresAt=iso(60),
        endsAt=iso(3600),
    )
    settings["arms"][0]["weight"] = 10000 - percentage * 100
    settings["arms"][1]["weight"] = percentage * 100


def test_rollout_monotonic_assignment_and_expired_disabled_control(server, monkeypatch):
    rollout(server)
    with connected(server) as client:
        original = {str(i): client.assign("example", unit=str(i)) for i in range(100)}
        assert all(value.allocation_version == 2 for value in original.values())
        rollout(server, 100, 3)
        client.refresh()
        for unit, prior in original.items():
            current = client.assign("example", unit=unit)
            assert current.arm == "b"
            assert current.allocation_version == 3
        now = time.time()
        with monkeypatch.context() as patch:
            patch.setattr(client_module.time, "time", lambda: now + 3601)
            assert client.assign("example", unit="1").arm == "a"
        server["config"].update(status="auto_paused", enabled=False)
        server["config"]["arms"][0]["weight"] = 10000
        server["config"]["arms"][1]["weight"] = 0
        client.refresh()
        assert client.assign("example", unit="1").arm == "a"


def test_rollout_deferred_closure_keeps_original_epoch_and_rejects_ambiguity(server):
    rollout(server)
    with connected(server) as client:
        assert client.log_exposure("example", unit_id="unit", arm="b", trace_id="trace")
        rollout(server, 100, 3)
        client.refresh()
        assert client.complete_unit(
            "example", unit_id="unit", arm="b", trace_id="trace", source="external"
        )
        assert client.flush()
        assert [row[3]["allocationVersion"] for row in server["requests"] if row[0] == "POST"] == [
            2,
            2,
        ]
        assert client.log_exposure(
            "example", unit_id="unit", arm="b", trace_id="trace", allocation_version=3
        )
        assert not client.complete_unit(
            "example", unit_id="unit", arm="b", trace_id="trace", source="external"
        )
        assert client.complete_unit(
            "example",
            unit_id="unit",
            arm="b",
            trace_id="trace",
            source="external",
            allocation_version=3,
        )
        assert not client.complete_unit("example", unit_id="unknown", arm="b", trace_id="unknown")


@pytest.mark.parametrize(
    "patch",
    [
        {"allocationVersion": 2147483648},
        {"allocationVersion": True},
        {"phase": "analysis"},
        {"rolloutPercentage": 75},
        {"rolloutPercentage": True},
        {"status": "running"},
        {"defaultArm": "b"},
        {"allocationVersion": 1},
    ],
)
def test_rollout_malformed_config_fails_control(server, patch):
    rollout(server)
    server["config"].update(patch)
    with connected(server) as client:
        assert client.assign("example", unit="unit").reason == "unavailable"
