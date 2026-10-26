"""Version-one cached A/B allocation protocol shared with the backend and other SDKs."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal
from urllib.parse import urlparse

Params = dict[str, str | float | int | bool | None]
ArmKey = Literal["a", "b"]
KEY = re.compile(r"^[a-z][a-z0-9_-]{0,79}$")
ASSIGNMENT = "sha256_utf8_salt_colon_unit_first_uint32_be_mod_10000"
STATES = {
    "draft",
    "running",
    "paused",
    "auto_paused",
    "concluded",
    "ramping",
    "shipped",
    "rolled_back",
}


def validate_origin(base_url: str) -> str:
    url = urlparse(base_url)
    if (
        url.username
        or url.password
        or url.query
        or url.fragment
        or url.path not in {"", "/"}
        or not url.hostname
        or (
            url.scheme != "https"
            and not (url.scheme == "http" and url.hostname in {"localhost", "127.0.0.1", "::1"})
        )
    ):
        raise ValueError("A/B base_url must be an HTTPS origin or local HTTP origin")
    return base_url.rstrip("/")


def valid_keys(keys: list[str]) -> list[str]:
    if (
        not isinstance(keys, list)
        or len(keys) > 20
        or any(not isinstance(key, str) or not KEY.fullmatch(key) for key in keys)
    ):
        raise ValueError("A/B keys must contain at most 20 valid experiment keys")
    return list(dict.fromkeys(keys))


def params(value: Any) -> Params:
    if not isinstance(value, dict) or len(value) > 20:
        raise ValueError("Invalid A/B parameters")
    for key, item in value.items():
        if not isinstance(key, str) or not 1 <= len(key) <= 100:
            raise ValueError("Invalid A/B parameter name")
        if item is None or isinstance(item, bool):
            continue
        if isinstance(item, str) and len(item) <= 2000:
            continue
        if isinstance(item, (int, float)) and math.isfinite(item):
            continue
        raise ValueError("Invalid A/B parameter value")
    return dict(value)


def timestamp(value: Any) -> float:
    if not isinstance(value, str):
        raise ValueError("Invalid A/B timestamp")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("A/B timestamps require a timezone")
    return parsed.timestamp()


def integer(value: Any, lower: int, upper: int) -> int:
    if type(value) is not int or not lower <= value <= upper:
        raise ValueError("Invalid A/B integer")
    return value


@dataclass(frozen=True)
class Arm:
    key: ArmKey
    params: Params
    weight: int


@dataclass(frozen=True)
class Config:
    id: str
    key: str
    revision: int
    salt: str
    unit: str
    enabled: bool
    default_arm: ArmKey
    expires_at: float
    ends_at: float
    arms: tuple[Arm, Arm]
    etag: str
    allocation_version: int
    phase: str


def parse_config(value: Any) -> Config:
    from uuid import UUID

    if not isinstance(value, dict):
        raise ValueError("Invalid A/B settings")
    integer(value.get("protocolVersion"), 1, 1)
    epoch = integer(value.get("allocationVersion"), 1, 2147483647)
    phase = value.get("phase", "analysis")
    percentage = value.get("rolloutPercentage")
    if phase == "analysis":
        if epoch != 1 or percentage is not None or value.get("status") == "ramping":
            raise ValueError("Invalid analysis epoch")
    elif phase == "rollout":
        if epoch < 2 or type(percentage) is not int or percentage not in {50, 100}:
            raise ValueError("Invalid rollout epoch")
    else:
        raise ValueError("Unsupported A/B phase")
    UUID(value["id"])
    valid_keys([value["key"]])
    if not isinstance(value.get("salt"), str) or not re.fullmatch(r"[a-f0-9]{64}", value["salt"]):
        raise ValueError("Invalid A/B salt")
    if value.get("assignment") != ASSIGNMENT or value.get("refreshAfterSeconds") != 60:
        raise ValueError("Unsupported A/B allocation")
    if value.get("status") not in STATES or value.get("unit") not in {"session", "end_user"}:
        raise ValueError("Invalid A/B lifecycle or unit")
    if type(value.get("enabled")) is not bool or value.get("defaultArm") not in {"a", "b"}:
        raise ValueError("Invalid A/B default")
    rows = value.get("arms")
    if not isinstance(rows, list) or len(rows) != 2:
        raise ValueError("Expected two A/B arms")
    arms = []
    for row, key in zip(rows, ("a", "b")):
        if not isinstance(row, dict) or row.get("key") != key:
            raise ValueError("Invalid A/B arm order")
        arms.append(Arm(key, params(row["params"]), integer(row.get("weight"), 0, 10000)))
    enabled, default = value["enabled"], value["defaultArm"]
    if arms[0].weight + arms[1].weight != 10000:
        raise ValueError("Invalid A/B allocation weights")
    expected_status = (
        ("ramping" if percentage == 50 else "shipped") if phase == "rollout" else "running"
    )
    if enabled and value["status"] != expected_status:
        raise ValueError("Invalid enabled A/B lifecycle")
    if phase == "rollout" and enabled and arms[1].weight != percentage * 100:
        raise ValueError("Invalid rollout percentage")
    if default != ("b" if phase == "analysis" and value["status"] == "shipped" else "a"):
        raise ValueError("Invalid A/B default version")
    if not enabled and arms[0 if default == "a" else 1].weight != 10000:
        raise ValueError("Invalid disabled A/B weights")
    if not isinstance(value.get("etag"), str) or not 1 <= len(value["etag"]) <= 200:
        raise ValueError("Invalid A/B revision tag")
    return Config(
        value["id"],
        value["key"],
        integer(value.get("revision"), 1, 2**53 - 1),
        value["salt"],
        value["unit"],
        enabled,
        default,
        timestamp(value["expiresAt"]),
        timestamp(value["endsAt"]),
        (arms[0], arms[1]),
        value["etag"],
        epoch,
        phase,
    )


def assignment_bucket(salt: str, unit_id: str) -> int:
    digest = hashlib.sha256(f"{salt}:{unit_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % 10000


@dataclass(frozen=True)
class AbAssignment:
    key: str
    arm: ArmKey
    params: Params
    reason: Literal["assigned", "default", "unavailable"]
    revision: int | None
    allocation_version: int | None = None


@dataclass(frozen=True)
class AbDecision:
    """Cached version choice before its agent run has been recorded."""

    key: str
    unit_id: str
    arm: ArmKey
    params: Params
    reason: Literal["assigned", "default", "unavailable"]
    revision: int | None
    allocation_version: int | None = None
