import importlib
import json
import math
import subprocess
import sys

import neatlogs

init_module = importlib.import_module("neatlogs.init")


def check(result, name):
    return next(item for item in result.checks if item.name == name)


def test_doctor_is_stable_read_only_and_network_free(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("doctor attempted a network request")

    monkeypatch.setattr("requests.sessions.Session.request", forbidden)
    before = (
        init_module._initialized,
        init_module._tracer_provider,
        tuple(init_module._export_health),
        init_module.get_session_config(),
    )

    result = neatlogs.doctor(disable_export=True)

    after = (
        init_module._initialized,
        init_module._tracer_provider,
        tuple(init_module._export_health),
        init_module.get_session_config(),
    )
    assert after == before
    assert result.format_version == "neatlogs.doctor/v1"
    assert result.sdk_version == neatlogs.__version__
    assert result.ready
    assert [item.name for item in result.checks] == [
        "runtime",
        "package",
        "schema",
        "transport",
        "endpoint",
        "sampler",
        "ownership",
        "queue",
        "export_health",
        "root",
    ]
    assert check(result, "queue").reason_code == "EXPORT_QUEUE_DISABLED"
    assert check(result, "export_health").status == "unknown"
    assert check(result, "root").status == "unknown"


def test_doctor_rejects_endpoint_and_sampler_without_leaking_secret(monkeypatch):
    secret = "doctor-super-secret"
    monkeypatch.setenv(
        "NEATLOGS_ENDPOINT", f"https://user:{secret}@ingest.neatlogs.com/path?token={secret}"
    )

    result = neatlogs.doctor(sample_rate=math.nan)
    encoded = result.to_json()

    assert not result.ready
    assert check(result, "endpoint").reason_code == "ENDPOINT_INVALID"
    assert check(result, "sampler").reason_code == "SAMPLER_INVALID"
    assert secret not in encoded


def test_doctor_observes_private_runtime_health_and_active_root():
    neatlogs.init(
        workflow_name="doctor-test",
        disable_export=True,
        register_shutdown_handlers=False,
    )
    try:
        with neatlogs.trace(name="doctor.root"):
            result = neatlogs.doctor(disable_export=True)
            root = check(result, "root")
            assert root.status == "pass"
            assert root.reason_code == "ROOT_IDS_VALID"
            assert len(root.details["trace_id"]) == 32
            assert len(root.details["span_id"]) == 16
            assert check(result, "ownership").reason_code == "OTEL_PROVIDER_PRIVATE"
            assert check(result, "export_health").reason_code == "EXPORT_HEALTHY"
    finally:
        neatlogs.shutdown()


def test_doctor_reports_existing_export_health_failure():
    class Health:
        failures = 2
        drops = 1

    class Exporter:
        health = Health()

    class Runtime:
        tracer_provider = object()
        _span_processor = None
        _exporters = [Exporter()]
        _log_exporters = []

    result = neatlogs.doctor(client=Runtime())
    health = check(result, "export_health")
    assert not result.ready
    assert health.reason_code == "EXPORT_HEALTH_UNHEALTHY"
    assert health.details == {"dropped_spans": "1", "export_failures": "2"}


def test_doctor_cli_emits_common_json_contract():
    process = subprocess.run(
        [sys.executable, "-m", "neatlogs", "doctor", "--disable-export", "--json"],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(process.stdout)
    assert result["format_version"] == "neatlogs.doctor/v1"
    assert result["ready"] is True
    assert process.stderr == ""
