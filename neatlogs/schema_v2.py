"""Access to the canonical, language-neutral NeatLogs telemetry contract v2."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from importlib import resources
from typing import Any

TELEMETRY_CONTRACT_VERSION = "2.0.0"
TELEMETRY_SCHEMA_VERSION = 2
TELEMETRY_SCHEMA_SHA256 = "1ce32734138c2ffc316c4299f5ae3eebec2f94381a538a383af49ba93eec9f9d"


def _contract_resource(name: str):
    return resources.files("neatlogs").joinpath(f"contracts/v2/{name}")


def telemetry_schema_bytes() -> bytes:
    """Return the exact public schema bytes shipped with this SDK."""

    return _contract_resource("neatlogs-telemetry.schema.json").read_bytes()


def telemetry_schema() -> dict[str, Any]:
    """Return the parsed canonical telemetry schema."""

    return json.loads(telemetry_schema_bytes())


def telemetry_manifest() -> dict[str, Any]:
    """Return the packaged contract manifest used by fixture generators."""

    return json.loads(_contract_resource("manifest.json").read_bytes())


def verify_telemetry_schema() -> None:
    """Fail when the packaged schema, digest, and manifest disagree."""

    actual = hashlib.sha256(telemetry_schema_bytes()).hexdigest()
    if actual != TELEMETRY_SCHEMA_SHA256:
        raise RuntimeError(
            "NeatLogs telemetry schema v2 digest mismatch: "
            f"expected {TELEMETRY_SCHEMA_SHA256}, received {actual}"
        )
    manifest = telemetry_manifest()
    schema = telemetry_schema()
    expected = {
        "contract_version": TELEMETRY_CONTRACT_VERSION,
        "schema_version": TELEMETRY_SCHEMA_VERSION,
        "schema_sha256": TELEMETRY_SCHEMA_SHA256,
        "schema_id": schema.get("$id"),
        "schema_file": "neatlogs-telemetry.schema.json",
    }
    mismatches = {
        key: (manifest.get(key), value)
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"NeatLogs telemetry manifest mismatch: {mismatches}")


def validate_telemetry_fixture(fixture: Mapping[str, Any] | Sequence[Mapping[str, Any]]) -> None:
    """Validate one generated span envelope, or a sequence of envelopes, against v2.

    Validation support is intentionally optional so importing the SDK does not add
    a JSON Schema dependency. Install ``neatlogs[schema-validation]`` in fixture
    generators and conformance jobs.
    """

    try:
        from jsonschema.validators import validator_for
    except ImportError as exc:  # pragma: no cover - exercised in a clean consumer test
        raise RuntimeError(
            "Telemetry fixture validation requires neatlogs[schema-validation]"
        ) from exc

    schema = telemetry_schema()
    validator_class = validator_for(schema)
    validator_class.check_schema(schema)
    validator = validator_class(schema)
    fixtures = [fixture] if isinstance(fixture, Mapping) else fixture
    if isinstance(fixtures, (str, bytes)) or not isinstance(fixtures, Sequence):
        raise TypeError("fixture must be a mapping or a sequence of mappings")
    for index, item in enumerate(fixtures):
        if not isinstance(item, Mapping):
            raise TypeError(f"fixture[{index}] must be a mapping")
        validator.validate(dict(item))
