"""Customer and session agent protocol. Validation occurs before a decision enters the cache."""

import re
from dataclasses import dataclass
from uuid import UUID

from ..ab_test.protocol import Params, integer, params, timestamp, valid_keys

PROFILE = "serving-trace-v1"
CAPABILITY_HEADER = "X-Neatlogs-Agent-Capability"


def length(value):
    return len(value.encode("utf-16-le")) // 2


def attributes(value):
    if not isinstance(value, dict) or len(value) > 20:
        raise ValueError("Invalid agent attributes")
    for key, item in value.items():
        if not isinstance(key, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]{0,63}", key):
            raise ValueError("Invalid attribute key")
        if type(item) is not bool and not (isinstance(item, str) and length(item) <= 256):
            raise ValueError("Attributes accept strings and booleans only")
    return dict(value)


def unit(value):
    if not isinstance(value, str) or not 1 <= length(value) <= 512:
        raise ValueError("A customer or session ID is required")
    return value


def _fields(value, fields):
    if not isinstance(value, dict) or set(value) != set(fields.split()):
        raise ValueError("Invalid agent protocol fields")


def _fingerprint(value):
    if not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{64}", value):
        raise ValueError("Invalid version fingerprint")


def _params(value):
    if not isinstance(value, dict):
        raise ValueError("Invalid params")
    result = {}
    for key, item in value.items():
        result.update(params({key: item}))
    if any(
        length(key) > 100 or isinstance(item, str) and length(item) > 2000
        for key, item in result.items()
    ):
        raise ValueError("Invalid agent params")
    return result


def _base(value):
    integer(value["protocolVersion"], 2, 2)
    valid_keys([value["agentKey"]])
    integer(value["revision"], 1, 2**53 - 1)
    timestamp(value["expiresAt"])


def _completion(value):
    if (value.get("unit"), value.get("completionPolicy")) not in {
        ("session", "explicit"),
        ("end_user", "fixed_horizon"),
    }:
        raise ValueError("Unsupported agent completion policy")


def _version(value):
    _fields(value, "fingerprint params")
    _fingerprint(value["fingerprint"])
    _params(value["params"])


def serving_ref(value, publication=False):
    _fields(value, "revisionId revision digest" + (" publicationId" if publication else ""))
    UUID(value["revisionId"])
    integer(value["revision"], 1, 2**53 - 1)
    _fingerprint(value["digest"])
    if publication:
        UUID(value["publicationId"])


def _serving(value):
    _fields(value, "revisionId revision digest unit requiredAttributeKeys allowedIdentities")
    serving_ref({key: value[key] for key in ("revisionId", "revision", "digest")})
    if value["unit"] not in {"end_user", "session"}:
        raise ValueError("Invalid serving unit")
    keys = value["requiredAttributeKeys"]
    if type(keys) is not list or len(keys) > 20 or keys != sorted(set(keys)):
        raise ValueError("Invalid serving attribute keys")
    attributes({key: True for key in keys})
    identities = value["allowedIdentities"]
    if type(identities) is not list or len(identities) > 32:
        raise ValueError("Invalid serving roster")
    seen = set()
    for identity in identities:
        _fields(identity, "publicationId fingerprint")
        UUID(identity["publicationId"])
        _fingerprint(identity["fingerprint"])
        if identity["publicationId"] in seen:
            raise ValueError("Duplicate publication identity")
        seen.add(identity["publicationId"])


def parse_config(value):
    enhanced = isinstance(value, dict) and value.get("profile") == PROFILE
    _fields(
        value,
        "protocolVersion agentKey revision expiresAt current live"
        + (" profile serving" if enhanced else ""),
    )
    _base(value)
    _version(value["current"])
    serving = value.get("serving")
    if serving is not None:
        _serving(serving)
    live = value["live"]
    if live is not None:
        _fields(
            live,
            "testId experimentId candidateRevision allocationVersion unit completionPolicy candidateFingerprint policyDigest status effectiveTrafficBps targetTrafficBps split"
            + (" agentProfile current baselineServing" if enhanced else " currentFingerprint"),
        )
        for key in ("testId", "experimentId"):
            UUID(live[key])
        for key in ("candidateFingerprint", "policyDigest"):
            _fingerprint(live[key])
        if enhanced:
            if live["agentProfile"] not in {"legacy-v2", PROFILE}:
                raise ValueError("Invalid agent policy profile")
            _version(live["current"])
            if live["baselineServing"] is not None:
                serving_ref(live["baselineServing"])
            if serving is not None and serving["unit"] != live["unit"]:
                raise ValueError("Conflicting serving unit")
        else:
            _fingerprint(live["currentFingerprint"])
            if live["currentFingerprint"] != value["current"]["fingerprint"]:
                raise ValueError("Invalid legacy Current fingerprint")
        integer(live["candidateRevision"], 1, 2**53 - 1)
        integer(live["allocationVersion"], 1, 2147483647)
        integer(live["effectiveTrafficBps"], 0, 10000)
        integer(live["targetTrafficBps"], live["effectiveTrafficBps"], 10000)
        _fields(live["split"], "a b")
        if integer(live["split"]["a"], 0, 10000) + integer(live["split"]["b"], 0, 10000) != 10000:
            raise ValueError("Invalid split")
        _completion(live)
        if live["status"] not in {"running", "paused", "auto_paused"}:
            raise ValueError("Unsupported agent allocation")
    return value


def parse_assignment(value):
    enhanced = isinstance(value, dict) and value.get("profile") == PROFILE
    reason = value.get("reason") if isinstance(value, dict) else None
    assigned, published = reason == "assigned", reason == "published" and enhanced
    fields = "protocolVersion agentKey revision expiresAt params fingerprint reason" + (
        " profile" if enhanced else ""
    )
    if published:
        fields += " unit unitId attributesDigest serving"
    else:
        fields += " arm" + (
            " experimentId candidateRevision testId allocationVersion enrollmentId assignmentId unit completionPolicy"
            if assigned
            else ""
        )
    _fields(value, fields)
    _base(value)
    _params(value["params"])
    _fingerprint(value["fingerprint"])
    if assigned:
        _completion(value)
        for key in ("experimentId", "testId", "enrollmentId", "assignmentId"):
            UUID(value[key])
        integer(value["candidateRevision"], 1, 2**53 - 1)
        integer(value["allocationVersion"], 1, 2147483647)
        if value["arm"] not in {"a", "b"}:
            raise ValueError("Invalid arm")
    elif published:
        if value["unit"] not in {"session", "end_user"}:
            raise ValueError("Invalid published unit")
        unit(value["unitId"])
        _fingerprint(value["attributesDigest"])
        serving_ref(value["serving"], publication=True)
    elif (
        reason not in {"outside_audience", "outside_share", "paused", "current"}
        or value["arm"] != "a"
    ):
        raise ValueError("Invalid fallback")
    return value


@dataclass(frozen=True)
class AgentDecision:
    agent_key: str
    unit_id: str
    arm: str | None
    params: Params
    reason: str
    revision: int | None = None
    expires_at: float | None = None
    fingerprint: str | None = None
    assignment_id: str | None = None
    test_id: str | None = None
    allocation_version: int | None = None
    experiment_id: str | None = None
    candidate_revision: int | None = None
    enrollment_id: str | None = None
    unit: str | None = None
    completion_policy: str | None = None
    profile: str | None = None
    serving: dict | None = None
    attributes_digest: str | None = None
    known_revision: int | None = None
    known_unit: str | None = None
    cause: str | None = None


def decision(value, unit_id, expires):
    return AgentDecision(
        value["agentKey"],
        unit_id,
        value.get("arm"),
        dict(value["params"]),
        value["reason"],
        value["revision"],
        expires,
        value["fingerprint"],
        value.get("assignmentId"),
        value.get("testId"),
        value.get("allocationVersion"),
        value.get("experimentId"),
        value.get("candidateRevision"),
        value.get("enrollmentId"),
        value.get("unit"),
        value.get("completionPolicy"),
        value.get("profile"),
        value.get("serving"),
        value.get("attributesDigest"),
    )
