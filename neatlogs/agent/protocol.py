"""Session-only agent protocol. Validation occurs before a decision enters the cache."""
from dataclasses import dataclass
import re
from uuid import UUID

from ..ab_test.protocol import Params, integer, params, timestamp, valid_keys


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
        raise ValueError("A session ID is required")
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
    if any(length(key) > 100 or isinstance(item, str) and length(item) > 2000 for key, item in result.items()):
        raise ValueError("Invalid agent params")
    return result


def _base(value):
    integer(value["protocolVersion"], 2, 2)
    valid_keys([value["agentKey"]])
    integer(value["revision"], 1, 2**53 - 1)
    timestamp(value["expiresAt"])


def parse_config(value):
    _fields(value, "protocolVersion agentKey revision expiresAt current live")
    _base(value)
    _fields(value["current"], "fingerprint params")
    _fingerprint(value["current"]["fingerprint"])
    _params(value["current"]["params"])
    live = value["live"]
    if live is not None:
        _fields(live, "testId experimentId candidateRevision allocationVersion unit currentFingerprint candidateFingerprint policyDigest status effectiveTrafficBps targetTrafficBps split")
        for key in ("testId", "experimentId"):
            UUID(live[key])
        for key in ("currentFingerprint", "candidateFingerprint", "policyDigest"):
            _fingerprint(live[key])
        integer(live["candidateRevision"], 1, 2**53 - 1)
        integer(live["allocationVersion"], 1, 2147483647)
        integer(live["effectiveTrafficBps"], 0, 10000)
        integer(live["targetTrafficBps"], live["effectiveTrafficBps"], 10000)
        _fields(live["split"], "a b")
        if integer(live["split"]["a"], 0, 10000) + integer(live["split"]["b"], 0, 10000) != 10000:
            raise ValueError("Invalid split")
        if live["unit"] != "session" or live["status"] not in {"running", "paused", "auto_paused"} or live["currentFingerprint"] != value["current"]["fingerprint"]:
            raise ValueError("Unsupported agent allocation")
    return value


def parse_assignment(value):
    assigned = isinstance(value, dict) and value.get("reason") == "assigned"
    _fields(value, "protocolVersion agentKey revision expiresAt params fingerprint reason arm" + (" experimentId candidateRevision testId allocationVersion enrollmentId assignmentId" if assigned else ""))
    _base(value)
    _params(value["params"])
    _fingerprint(value["fingerprint"])
    if assigned:
        for key in ("experimentId", "testId", "enrollmentId", "assignmentId"):
            UUID(value[key])
        integer(value["candidateRevision"], 1, 2**53 - 1)
        integer(value["allocationVersion"], 1, 2147483647)
        if value["arm"] not in {"a", "b"}:
            raise ValueError("Invalid arm")
    elif value["reason"] not in {"outside_audience", "outside_share", "paused", "current"} or value["arm"] != "a":
        raise ValueError("Invalid fallback")
    return value


@dataclass(frozen=True)
class AgentDecision:
    agent_key: str
    unit_id: str
    arm: str
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


def decision(value, unit_id, expires):
    return AgentDecision(value["agentKey"], unit_id, value["arm"], dict(value["params"]), value["reason"], value["revision"], expires, value["fingerprint"], value.get("assignmentId"), value.get("testId"), value.get("allocationVersion"), value.get("experimentId"), value.get("candidateRevision"), value.get("enrollmentId"))
