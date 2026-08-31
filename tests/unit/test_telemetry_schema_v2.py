import hashlib
import json

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


def test_canonical_telemetry_policy_is_consumed_by_the_sdk():
    schema = neatlogs.telemetry_schema()
    policy = schema["x-neatlogs-policy"]

    assert neatlogs.TELEMETRY_SCHEMA_VERSION == 2
    assert policy["contract_version"] == neatlogs.TELEMETRY_CONTRACT_VERSION
    assert policy["conflict_precedence"] == EXPECTED_PRECEDENCE
    assert policy["tool_calls"]["execution_is_separate_tool_span"] is True
    assert policy["root_finalization"]["launch_sdk_auto_workflow_roots"] is True
