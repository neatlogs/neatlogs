import json
from pathlib import Path
from types import SimpleNamespace

from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExportResult
from opentelemetry.trace import SpanContext, SpanKind, Status, StatusCode, TraceFlags, TraceState

from neatlogs.__main__ import main
from neatlogs.core.masking_exporter import MaskingSpanExporter
from neatlogs.doctor_v2 import (
    clear_doctor_capture,
    doctor_captured_local_v2,
    doctor_local_v2,
    doctor_probe_v2,
    doctor_semantic_digest,
    get_captured_envelope,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "doctor-v2"


def envelope():
    return json.loads((FIXTURES / "valid-envelope.json").read_text())


def test_canonical_digest_matches_cross_language_golden_value():
    assert doctor_semantic_digest(envelope()) == (
        "sha256:76d8726734664dacaa4e6da4ffc547cc" "5b7c8edde4721a485b5875378c233381"
    )


def test_valid_envelope_conforms_to_doctor_v2_shape():
    result = doctor_local_v2(envelope(), flush_duration_ms=12)
    assert result["format_version"] == "neatlogs.doctor/v2"
    assert result["mode"] == "local"
    assert result["status"] == "pass"
    assert result["first_failure"] is None
    assert result["runtime"]["language"] == "python"
    assert result["checks"][0]["reason_code"] == "LOCAL_ENVELOPE_VALID"


def test_validation_reports_first_contract_order_failure():
    invalid = envelope()
    invalid["trace_id"] = "bad"
    invalid["spans"][1]["parent_span_id"] = "missing"
    result = doctor_local_v2(invalid)
    assert result["status"] == "fail"
    assert result["first_failure"] == "TRACE_ID_INVALID"
    assert "capture" not in result


def _span(attributes, *, trace_id=1, span_id=2, parent_id=0):
    context = SpanContext(
        trace_id=trace_id,
        span_id=span_id,
        is_remote=False,
        trace_flags=TraceFlags.SAMPLED,
        trace_state=TraceState(),
    )
    parent = (
        None
        if not parent_id
        else SpanContext(
            trace_id=trace_id,
            span_id=parent_id,
            is_remote=False,
            trace_flags=TraceFlags.SAMPLED,
            trace_state=TraceState(),
        )
    )
    return ReadableSpan(
        name="root",
        context=context,
        parent=parent,
        resource=Resource.create({}),
        attributes=attributes,
        events=[],
        links=[],
        kind=SpanKind.INTERNAL,
        status=Status(StatusCode.OK),
        start_time=1,
        end_time=2,
        instrumentation_scope=None,
    )


class Exporter:
    def __init__(self):
        self.spans = []

    def export(self, spans):
        self.spans.extend(spans)
        return SpanExportResult.SUCCESS

    def force_flush(self, timeout_millis=30000):
        return True

    def shutdown(self):
        pass


def persisted_trace(exporter):
    spans = [span for span in exporter.spans if span.name != "neatlogs.trace.complete"]
    kinds = {
        "WORKFLOW": "workflow",
        "AGENT": "agent_action",
        "LLM": "llm",
        "TOOL": "tool_call",
    }
    return {
        "_id": f"{spans[0].get_span_context().trace_id:032x}",
        "workflowName": "neatlogs.doctor.v2",
        "spanCount": len(spans),
        "promptTokens": 11,
        "completionTokens": 7,
        "totalTokensUsed": 18,
        "spans": [
            {
                "span_id": f"{item.get_span_context().span_id:016x}",
                **(
                    {"parent_span_id": f"{item.parent.span_id:016x}"}
                    if item.parent and item.parent.is_valid
                    else {}
                ),
                "node_name": item.name,
                "node_type": kinds[
                    str(
                        item.attributes.get("openinference.span.kind")
                        or item.attributes.get("neatlogs.span.kind")
                    )
                    .removeprefix("Neatlogs.")
                    .upper()
                ],
                "data": {
                    "input_value": item.attributes.get("input.value", "{}"),
                    "output_value": item.attributes.get("output.value", "{}"),
                },
                "span_metadata": {
                    "neatlogs.doctor": item.attributes.get("neatlogs.doctor"),
                    "neatlogs.doctor.version": item.attributes.get("neatlogs.doctor.version"),
                    "telemetry.sdk.language": item.attributes.get("telemetry.sdk.language"),
                },
            }
            for item in spans
        ],
    }


def test_real_exporter_path_captures_only_post_mask_data():
    clear_doctor_capture()
    inner = Exporter()

    def mask(snapshot):
        snapshot["attributes"]["input.value"] = '{"email":"[masked]"}'
        return snapshot

    exporter = MaskingSpanExporter(inner, mask)
    assert (
        exporter.export(
            [
                _span(
                    {
                        "neatlogs.span.kind": "WORKFLOW",
                        "input.value": '{"email":"secret@example.com"}',
                    }
                )
            ]
        )
        is SpanExportResult.SUCCESS
    )
    captured = get_captured_envelope("00000000000000000000000000000001")
    assert captured["spans"][0]["input"] == {"email": "[masked]"}
    serialized = json.dumps(captured)
    assert "secret@example.com" not in serialized
    assert doctor_captured_local_v2("00000000000000000000000000000001")["status"] == "pass"


def test_capture_store_is_bounded_to_sixteen_traces():
    clear_doctor_capture()
    from neatlogs.doctor_v2 import capture_prepared_spans

    for trace_id in range(1, 18):
        capture_prepared_spans([_span({"neatlogs.span.kind": "WORKFLOW"}, trace_id=trace_id)])
    assert get_captured_envelope("00000000000000000000000000000001") is None
    assert get_captured_envelope("00000000000000000000000000000011") is not None


def test_probe_exports_and_reads_back_the_exact_trace(monkeypatch):
    exporter = Exporter()
    calls = []

    def get(url, **kwargs):
        calls.append((url, kwargs))
        return SimpleNamespace(
            ok=True,
            status_code=200,
            json=lambda: persisted_trace(exporter),
            raise_for_status=lambda: None,
        )

    monkeypatch.setattr("neatlogs.doctor_v2.requests.get", get)
    result = doctor_probe_v2(
        api_key="local-key",
        endpoint="http://localhost:4100",
        timeout_seconds=1,
        _exporter=exporter,
    )
    assert result["status"] == "pass"
    assert result["capture"]["span_count"] == 4
    assert result["probe"] == {
        "ingest_route": "/v1/traces",
        "marker_header": "x-neatlogs-doctor",
        "marker_version": "v1",
        "visible": True,
        "readback_span_count": 4,
        "hierarchy_valid": True,
        "attributes_valid": True,
        "input_output_valid": True,
        "metadata_valid": True,
        "typed_tokens_valid": True,
    }
    assert calls[0][0].startswith("http://localhost:4100/api/traces/v3/")
    assert calls[0][1]["headers"] == {"x-api-key": "local-key"}
    assert "local-key" not in json.dumps(result)


def test_probe_exporter_uses_normal_trace_route_and_versioned_header(monkeypatch):
    import importlib

    init_module = importlib.import_module("neatlogs.init")
    captured = {}

    class ConstructorExporter(Exporter):
        def __init__(self, **kwargs):
            super().__init__()
            captured.update(kwargs)

    monkeypatch.setattr(init_module, "OTLPSpanExporter", ConstructorExporter)
    init_module.init(
        api_key="project-key",
        endpoint="http://localhost:4100",
        workflow_name="neatlogs.doctor.v2",
        register_shutdown_handlers=False,
        _doctor_probe=True,
    )
    try:
        assert captured["endpoint"] == "http://localhost:4100/v1/traces"
        assert captured["headers"] == {
            "x-api-key": "project-key",
            "x-neatlogs-doctor": "v1",
        }
    finally:
        init_module.shutdown(termination_reason="doctor-test-complete")


def test_probe_missing_credential_never_echoes_environment(monkeypatch):
    monkeypatch.delenv("NEATLOGS_API_KEY", raising=False)
    result = doctor_probe_v2()
    assert result["first_failure"] == "CREDENTIAL_MISSING"
    assert result["capture"]["span_count"] == 4


def test_probe_invalid_endpoint_returns_stable_configuration_failure():
    result = doctor_probe_v2(api_key="test-key", endpoint="not-a-url")
    assert result["status"] == "fail"
    assert result["first_failure"] == "ENDPOINT_INVALID"
    assert result["checks"][-1]["remediation_code"] == "SET_ENDPOINT"
    assert "api_key" not in json.dumps(result).lower()


def test_standalone_local_runs_real_isolated_exporter_pipeline(monkeypatch, capsys):
    clear_doctor_capture()
    monkeypatch.delenv("NEATLOGS_API_KEY", raising=False)

    def forbidden(*args, **kwargs):
        raise AssertionError("local Doctor attempted backend access")

    monkeypatch.setattr("requests.sessions.Session.request", forbidden)
    assert main(["doctor", "--local", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "pass"
    assert result["first_failure"] is None
    assert result["capture"]["span_count"] == 4
    assert result["capture"]["semantic_digest"].startswith("sha256:")
    assert result["checks"] == [
        {
            "name": "local_envelope",
            "status": "pass",
            "reason_code": "LOCAL_ENVELOPE_VALID",
            "message": "The final normalized local envelope is valid",
            "remediation_code": "NONE",
        }
    ]
    assert result["runtime"]["language"] == "python"
    assert result["sampling"]["sampled"] is True
    assert result["ownership"]["provider"] == "private"
    assert result["queue"]["mode"] == "diagnostic_capture"
    assert result["flush"]["outcome"] == "success"


def test_probe_never_treats_processing_readback_as_success(monkeypatch):
    exporter = Exporter()
    response = SimpleNamespace(
        ok=True,
        status_code=202,
        json=lambda: {"status": "processing"},
        raise_for_status=lambda: None,
    )
    monkeypatch.setattr("neatlogs.doctor_v2.requests.get", lambda *a, **k: response)
    monkeypatch.setattr("neatlogs.doctor_v2.time.sleep", lambda *_: None)
    ticks = iter(float(value) for value in range(1_000))
    monkeypatch.setattr("neatlogs.doctor_v2.time.monotonic", lambda: next(ticks))
    result = doctor_probe_v2(
        api_key="local-key",
        endpoint="http://localhost:4100",
        timeout_seconds=1,
        _exporter=exporter,
    )
    assert result["status"] == "fail"
    assert result["first_failure"] == "BACKEND_PROBE_UNAVAILABLE"


def test_probe_rejects_redacted_token_counts(monkeypatch):
    exporter = Exporter()

    def get(*args, **kwargs):
        trace_data = persisted_trace(exporter)
        trace_data["promptTokens"] = "[REDACTED]"
        return SimpleNamespace(
            ok=True,
            status_code=200,
            json=lambda: trace_data,
            raise_for_status=lambda: None,
        )

    monkeypatch.setattr("neatlogs.doctor_v2.requests.get", get)
    result = doctor_probe_v2(
        api_key="local-key",
        endpoint="http://localhost:4100",
        timeout_seconds=1,
        _exporter=exporter,
    )
    assert result["status"] == "fail"
    assert result["first_failure"] == "TYPED_TOKENS_VALID_FAILED"
    assert result["probe"]["typed_tokens_valid"] is False


def test_standalone_local_reports_missing_configuration(monkeypatch, capsys):
    clear_doctor_capture()
    monkeypatch.delenv("NEATLOGS_API_KEY", raising=False)
    assert main(["doctor", "--local", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["first_failure"] is None
    assert all(check["reason_code"] != "CREDENTIAL_MISSING" for check in result["checks"])
