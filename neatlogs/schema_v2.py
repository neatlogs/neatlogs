"""Access to the canonical, language-neutral NeatLogs telemetry contract v2."""

from __future__ import annotations

import hashlib
import json
from importlib import resources
from typing import Any

TELEMETRY_CONTRACT_VERSION = "2.1.0"
TELEMETRY_SCHEMA_VERSION = 2
TELEMETRY_SCHEMA_SHA256 = "75bb25d2162be846927a2687787e01c5c09333ee0d75db1fc0e20056931c5fc5"


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
