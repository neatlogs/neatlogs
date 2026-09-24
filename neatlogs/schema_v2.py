"""Access to the canonical, language-neutral NeatLogs telemetry contract v2."""

from __future__ import annotations

import hashlib
import json
from importlib import resources
from typing import Any

TELEMETRY_CONTRACT_VERSION = "2.0.0"
TELEMETRY_SCHEMA_VERSION = 2
# Digest of the canonical schema bytes normalized to LF line endings. Git's
# `core.autocrlf` (Windows) checks out the JSON with CRLF, which changes the
# raw SHA-256 without changing any semantics — so all digest comparisons must
# go through the normalized form. This value equals the LF blob hash.
TELEMETRY_SCHEMA_SHA256 = "50bbd9f1e6eaa6c83f08dcb84da3a98867c962fc8c4e1edd629da561fe5fe5a8"


def _normalize_schema_bytes(data: bytes) -> bytes:
    """Normalize line endings so the digest is identical on every OS."""

    return data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def telemetry_schema_digest(data: bytes | None = None) -> str:
    """Return the canonical SHA-256 of the telemetry schema.

    Normalizes CRLF/CR to LF before hashing so Windows checkouts
    (`core.autocrlf=true`), LF checkouts, and sdists/wheels all agree.
    """

    raw = data if data is not None else telemetry_schema_bytes()
    return hashlib.sha256(_normalize_schema_bytes(raw)).hexdigest()


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
    """Fail if packaging or a local edit changed the frozen contract bytes.

    Compares the LF-normalized digest so the check is OS-independent.
    """

    actual = telemetry_schema_digest()
    if actual != TELEMETRY_SCHEMA_SHA256:
        raise RuntimeError(
            "NeatLogs telemetry schema v2 digest mismatch: "
            f"expected {TELEMETRY_SCHEMA_SHA256}, received {actual}"
        )
