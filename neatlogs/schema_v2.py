"""Access to the canonical, language-neutral NeatLogs telemetry contract v2."""

from __future__ import annotations

import hashlib
import json
from importlib import resources
from typing import Any

TELEMETRY_CONTRACT_VERSION = "2.0.0"
TELEMETRY_SCHEMA_VERSION = 2
TELEMETRY_SCHEMA_SHA256 = "b0242cfd18aac41da9619449376a80d100b83a9c3abe48b31068771468275ef5"


def telemetry_schema_bytes() -> bytes:
    """Return the exact public schema bytes shipped with this SDK."""

    return (
        resources.files("neatlogs")
        .joinpath("contracts/v2/neatlogs-telemetry.schema.json")
        .read_bytes()
    )


def telemetry_schema() -> dict[str, Any]:
    """Return the parsed canonical telemetry schema."""

    return json.loads(telemetry_schema_bytes())


def verify_telemetry_schema() -> None:
    """Fail if packaging or a local edit changed the frozen contract bytes."""

    actual = hashlib.sha256(telemetry_schema_bytes()).hexdigest()
    if actual != TELEMETRY_SCHEMA_SHA256:
        raise RuntimeError(
            "NeatLogs telemetry schema v2 digest mismatch: "
            f"expected {TELEMETRY_SCHEMA_SHA256}, received {actual}"
        )
