import hashlib
import json

import pytest
from jsonschema import ValidationError

import neatlogs

EXPECTED_PRECEDENCE = [
    "native-v2",
    "neatlogs-direct",
    "otel-genai",
    "openinference",
    "provider-specific",
    "external-legacy",
    "unknown-raw",
]


def test_canonical_telemetry_schema_is_packaged_verbatim():
    schema_bytes = neatlogs.telemetry_schema_bytes()

    assert hashlib.sha256(schema_bytes).hexdigest() == neatlogs.TELEMETRY_SCHEMA_SHA256
    assert json.loads(schema_bytes) == neatlogs.telemetry_schema()
    neatlogs.verify_telemetry_schema()
    manifest = neatlogs.telemetry_manifest()
    assert manifest["schema_sha256"] == neatlogs.TELEMETRY_SCHEMA_SHA256
    assert manifest["schema_id"] == json.loads(schema_bytes)["$id"]


def test_canonical_telemetry_policy_is_consumed_by_the_sdk():
    schema = neatlogs.telemetry_schema()
    policy = schema["x-neatlogs-policy"]

    assert neatlogs.TELEMETRY_SCHEMA_VERSION == 2
    assert policy["contract_version"] == neatlogs.TELEMETRY_CONTRACT_VERSION
    assert policy["conflict_precedence"] == EXPECTED_PRECEDENCE
    assert policy["tool_calls"]["execution_is_separate_tool_span"] is True
    assert policy["root_finalization"]["launch_sdk_auto_workflow_roots"] is True


def test_stream_contract_requires_an_explicit_terminal_completion_state():
    stream = neatlogs.telemetry_schema()["$defs"]["llmStream"]

    assert "completion_state" in stream["required"]
    assert stream["properties"]["completion_state"]["enum"] == [
        "not_streamed",
        "complete",
        "consumer_cancelled",
        "provider_error",
    ]


def _generated_workflow_fixture():
    """Smallest useful output shape shared fixture generators can emit."""

    return {
        "schema_version": 2,
        "trace_id": "1" * 32,
        "span_id": "2" * 16,
        "parent_span_id": None,
        "name": "generated-workflow",
        "kind": "WORKFLOW",
        "ownership": {
            "owner": "neatlogs-sdk",
            "provider_generation": 1,
            "project_key_id": None,
            "propagation": "local-private-context",
        },
        "wrapper": {
            "captured": False,
            "integration": None,
            "integration_version": None,
            "capture_fidelity": "native",
        },
        "input": {"type": "json", "value": {"question": "fixed"}, "media": []},
        "output": {"type": "text", "value": "FIXTURE_OK", "media": []},
        "status": {"code": "OK", "message": None, "source": "application"},
        "error": None,
        "code": None,
        "provenance": [],
        "conflicts": [],
        "semantic": {
            "kind": "WORKFLOW",
            "operation": "fixture-generation",
            "role": None,
            "metadata": {},
            "recovery": None,
        },
        "attributes": {},
    }


def _generated_streaming_fixture(completion_state):
    fixture = _generated_workflow_fixture()
    fixture["name"] = "generated-stream"
    fixture["kind"] = "LLM"
    fixture["semantic"] = {
        "kind": "LLM",
        "request": {
            "provider": "fixture",
            "model": "fixture-model",
            "operation": "generate",
            "messages": [],
            "tools": [],
            "parameters": {
                "temperature": None,
                "top_p": None,
                "top_k": None,
                "max_output_tokens": None,
                "stop": [],
                "seed": None,
                "frequency_penalty": None,
                "presence_penalty": None,
                "response_format": None,
                "reasoning": None,
                "service_tier": None,
                "provider_options": {},
            },
        },
        "response": {
            "id": None,
            "model": "fixture-model",
            "choices": [],
            "finish_reasons": [],
        },
        "usage": {
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
            "reasoning_tokens": None,
            "cache_read_tokens": None,
            "cache_write_tokens": None,
            "cost_usd": None,
        },
        "stream": {
            "completion_state": completion_state,
            "time_to_first_token_ms": None,
            "chunk_count": 1,
            "choice_count": 0,
            "events": [],
            "raw_chunks": None,
        },
    }
    return fixture


def test_generated_fixture_validation_hook_accepts_one_or_many_envelopes():
    fixture = _generated_workflow_fixture()

    assert neatlogs.validate_telemetry_fixture(fixture) is None
    assert neatlogs.validate_telemetry_fixture([fixture, fixture]) is None


@pytest.mark.parametrize(
    "completion_state",
    ["not_streamed", "complete", "consumer_cancelled", "provider_error"],
)
def test_generated_stream_fixture_requires_a_terminal_completion_state(completion_state):
    assert (
        neatlogs.validate_telemetry_fixture(_generated_streaming_fixture(completion_state)) is None
    )


def test_generated_fixture_validation_hook_rejects_contract_drift():
    fixture = _generated_workflow_fixture()
    fixture["trace_id"] = "not-a-trace-id"

    with pytest.raises(ValidationError):
        neatlogs.validate_telemetry_fixture(fixture)


@pytest.mark.parametrize("fixture", ["invalid", [None]])
def test_generated_fixture_validation_hook_rejects_non_mapping_items(fixture):
    with pytest.raises(TypeError):
        neatlogs.validate_telemetry_fixture(fixture)
