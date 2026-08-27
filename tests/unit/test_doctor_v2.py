import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExportResult
from opentelemetry.trace import SpanContext, SpanKind, Status, StatusCode, TraceFlags, TraceState

from neatlogs.core.masking_exporter import MaskingSpanExporter
from neatlogs.doctor_v2 import (
    clear_doctor_capture,
    doctor_captured_local_v2,
    doctor_local_v2,
    doctor_probe_v2,
    doctor_semantic_digest,
    get_captured_envelope,
)
from neatlogs.__main__ import main

FIXTURES = Path("/Users/shyam-neatlogs/neatlogs-chotu/contracts/doctor/v2/fixtures")


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


def test_probe_polling_never_passes_on_auth_receipt_only(monkeypatch):
    created = {
        "diagnostic_id": "diag_0123456789abcdef",
        "probe_token": "x" * 32,
        "expires_at": "2030-01-01T00:05:00Z",
    }
    receipt = {
        "status": "pending",
        "diagnostic_id": created["diagnostic_id"],
        "created_at": "2030-01-01T00:00:00Z",
        "expires_at": created["expires_at"],
        "stages": [
            {
                "stage": "auth",
                "status": "accepted",
                "reason_code": "AUTH_ACCEPTED",
                "at": "2030-01-01T00:00:01Z",
            }
        ],
    }
    response = lambda value, ok=True: SimpleNamespace(
        ok=ok, json=lambda: value, raise_for_status=lambda: None
    )
    monkeypatch.setattr("neatlogs.doctor_v2.requests.post", lambda *a, **k: response(created))
    monkeypatch.setattr("neatlogs.doctor_v2.requests.get", lambda *a, **k: response(receipt))
    monkeypatch.setattr("neatlogs.doctor_v2.requests.delete", lambda *a, **k: response({}))
    monkeypatch.setattr("neatlogs.doctor_v2.time.sleep", lambda *_: None)
    ticks = iter([0.0, 1.0, 2.0])
    monkeypatch.setattr("neatlogs.doctor_v2.time.monotonic", lambda: next(ticks, 2.0))
    result = doctor_probe_v2(
        api_key="local-key", endpoint="http://localhost:4100", timeout_seconds=1
    )
    assert result["status"] == "fail"
    assert result["first_failure"] == "STAGE_PENDING"
    assert result["probe"]["receipt_status"] == "pending"
    assert result["checks"][0]["remediation_code"] == "WAIT_FOR_RECEIPT"
    assert "probe_token" not in json.dumps(result)


def test_probe_missing_credential_never_echoes_environment(monkeypatch):
    monkeypatch.delenv("NEATLOGS_API_KEY", raising=False)
    result = doctor_probe_v2()
    assert result["first_failure"] == "CREDENTIAL_MISSING"
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
    assert result["capture"]["span_count"] == 2
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


def test_probe_digest_mismatch_prevents_success(monkeypatch):
    created = {
        "diagnostic_id": "diag_0123456789abcdef",
        "probe_token": "x" * 32,
        "expires_at": "2030-01-01T00:05:00Z",
    }
    stages = [
        {
            "stage": stage,
            "status": "accepted",
            "reason_code": code,
            "at": "2030-01-01T00:00:01Z",
        }
        for stage, code in (
            ("auth", "AUTH_ACCEPTED"),
            ("schema_decode", "SCHEMA_DECODED"),
            ("pii", "PII_PROCESSED"),
            ("kafka", "KAFKA_PUBLISHED"),
            ("raw_durable", "RAW_DURABLE"),
            ("root_resolution", "ROOT_RESOLVED"),
            ("simplified_durable", "SIMPLIFIED_DURABLE"),
            ("visibility", "DIAGNOSTIC_VISIBLE"),
        )
    ]
    receipt = {
        "status": "pass",
        "expires_at": created["expires_at"],
        "backend_semantic_digest": "sha256:" + "f" * 64,
        "stages": stages,
    }
    response = lambda value: SimpleNamespace(
        ok=True, json=lambda: value, raise_for_status=lambda: None
    )
    monkeypatch.setattr("neatlogs.doctor_v2.requests.post", lambda *a, **k: response(created))
    monkeypatch.setattr("neatlogs.doctor_v2.requests.get", lambda *a, **k: response(receipt))
    monkeypatch.setattr("neatlogs.doctor_v2.requests.delete", lambda *a, **k: response({}))
    result = doctor_probe_v2(
        api_key="local-key", endpoint="http://localhost:4100", timeout_seconds=1
    )
    assert result["status"] == "fail"
    assert result["first_failure"] == "DIGEST_MISMATCH"


@pytest.mark.parametrize(
    ("receipt", "expected"),
    [
        (
            {
                "status": "expired",
                "expires_at": "2030-01-01T00:05:00Z",
                "stages": [
                    {
                        "stage": "auth",
                        "status": "accepted",
                        "reason_code": "AUTH_ACCEPTED",
                        "at": "2030-01-01T00:00:01Z",
                    }
                ],
            },
            "DIAGNOSTIC_EXPIRED",
        ),
        (
            {
                "status": "fail",
                "first_failure": "RAW_DURABILITY_FAILED",
                "expires_at": "2030-01-01T00:05:00Z",
                "stages": [
                    {
                        "stage": "raw_durable",
                        "status": "failed",
                        "reason_code": "RAW_DURABILITY_FAILED",
                        "at": "2030-01-01T00:00:01Z",
                    }
                ],
            },
            "RAW_DURABILITY_FAILED",
        ),
    ],
)
def test_probe_preserves_expiry_and_first_backend_failure(monkeypatch, receipt, expected):
    created = {
        "diagnostic_id": "diag_0123456789abcdef",
        "probe_token": "x" * 32,
        "expires_at": "2030-01-01T00:05:00Z",
    }
    response = lambda value: SimpleNamespace(
        ok=True, json=lambda: value, raise_for_status=lambda: None
    )
    monkeypatch.setattr("neatlogs.doctor_v2.requests.post", lambda *a, **k: response(created))
    monkeypatch.setattr("neatlogs.doctor_v2.requests.get", lambda *a, **k: response(receipt))
    monkeypatch.setattr("neatlogs.doctor_v2.requests.delete", lambda *a, **k: response({}))
    result = doctor_probe_v2(
        api_key="local-key", endpoint="http://localhost:4100", timeout_seconds=1
    )
    assert result["status"] == "fail"
    assert result["first_failure"] == expected


def test_standalone_local_reports_missing_configuration(monkeypatch, capsys):
    clear_doctor_capture()
    monkeypatch.delenv("NEATLOGS_API_KEY", raising=False)
    assert main(["doctor", "--local", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["first_failure"] is None
    assert all(check["reason_code"] != "CREDENTIAL_MISSING" for check in result["checks"])
