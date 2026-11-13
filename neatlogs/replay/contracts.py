"""Strict, bounded local replay protocol shared with the TypeScript SDK."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Literal, Union
from uuid import UUID

import rfc8785

from ..errors import NeatlogsError

Json = Union[None, bool, int, float, str, list["Json"], dict[str, "Json"]]
Kind = Literal["tool", "model"]
MAX_RECORDING_BYTES = 16 * 1024 * 1024
MAX_INPUT_BYTES = 1024 * 1024


class ReplayError(NeatlogsError, ValueError):
    """Invalid local replay configuration; never includes recorded content or credentials."""


def json_value(value: object, *, limit: int = MAX_RECORDING_BYTES) -> Json:
    ancestors: set[int] = set()
    nodes = 0

    def visit(item, depth):
        nonlocal nodes
        nodes += 1
        if nodes > 100_000 or depth > 64:
            raise ReplayError("Replay JSON exceeds its traversal limit.")
        if item is None or type(item) in (str, bool):
            return item
        if type(item) in (int, float):
            if not math.isfinite(item):
                raise ReplayError("Replay JSON numbers must be finite.")
            if type(item) is int and abs(item) > 2**53 - 1:
                number = float(item)
                if int(number) != item:
                    raise ReplayError("Replay JSON integers must be exactly representable.")
                return number
            return item
        if type(item) not in (dict, list) or id(item) in ancestors:
            raise ReplayError("Replay values must contain finite, acyclic plain JSON.")
        ancestors.add(id(item))
        try:
            if type(item) is list:
                return [visit(child, depth + 1) for child in item]
            if any(type(key) is not str for key in item):
                raise ReplayError("Replay JSON object keys must be strings.")
            return {key: visit(child, depth + 1) for key, child in item.items()}
        finally:
            ancestors.remove(id(item))

    try:
        result = visit(value, 0)
        if len(rfc8785.dumps(result)) > limit:
            raise ReplayError("Replay JSON exceeds its size limit.")
        return result
    except (rfc8785.CanonicalizationError, UnicodeError, OverflowError):
        raise ReplayError("Replay value is not lossless JSON.") from None


def canonical_json(value: object, *, limit: int = MAX_RECORDING_BYTES) -> bytes:
    return rfc8785.dumps(json_value(value, limit=limit))


def replay_key(kind: Kind, name: str, value: object) -> str:
    return (
        f"{kind}:{name}:{hashlib.sha256(canonical_json(value, limit=MAX_INPUT_BYTES)).hexdigest()}"
    )


def object_fields(value: object, required: set[str], optional: set[str] = frozenset()) -> dict:
    if type(value) is not dict or set(value) - required - optional or required - set(value):
        raise ReplayError("Invalid local replay object fields.")
    return value


def identifier(value: object) -> str:
    if type(value) is not str or not 1 <= len(value) <= 200:
        raise ReplayError("Replay identifiers must contain 1 to 200 characters.")
    return value


def uuid(value: object) -> str:
    try:
        if type(value) is not str or len(value) != 36:
            raise ValueError()
        return str(UUID(value))
    except (ValueError, TypeError, AttributeError):
        raise ReplayError("Replay binding identifiers must be UUIDs.") from None


@dataclass(frozen=True)
class ReplayEntry:
    span_id: str
    kind: Kind
    name: str
    input: Json
    output: Json

    def to_dict(self):
        return {
            "spanId": self.span_id,
            "kind": self.kind,
            "name": self.name,
            "input": self.input,
            "output": self.output,
        }


@dataclass(frozen=True)
class ReplayRecording:
    source_trace_id: str
    entries: tuple[ReplayEntry, ...]

    def to_dict(self):
        return {
            "version": 1,
            "sourceTraceId": self.source_trace_id,
            "entries": [entry.to_dict() for entry in self.entries],
        }


@dataclass(frozen=True)
class ReplayRunBinding:
    run_id: str
    case_id: str
    trial: int

    @classmethod
    def parse(cls, value):
        fields = object_fields(value, {"runId", "caseId", "trial"})
        trial = fields["trial"]
        if type(trial) is not int or not 1 <= trial <= 10:
            raise ReplayError("Replay trial must be an integer from 1 to 10.")
        return cls(uuid(fields["runId"]), uuid(fields["caseId"]), trial)


def parse_replay_recording(value: object) -> ReplayRecording:
    if isinstance(value, ReplayRecording):
        value = value.to_dict()
    copied = json_value(value)
    fields = object_fields(copied, {"version", "sourceTraceId", "entries"})
    if type(fields["version"]) is not int or fields["version"] != 1:
        raise ReplayError("Unsupported local replay recording version.")
    entries = fields["entries"]
    if type(entries) is not list or len(entries) > 5000:
        raise ReplayError("Replay recordings allow at most 5000 entries.")
    result = []
    for raw in entries:
        item = object_fields(raw, {"spanId", "kind", "name", "input", "output"})
        if item["kind"] not in ("tool", "model"):
            raise ReplayError("Unsupported local replay entry kind.")
        result.append(
            ReplayEntry(
                identifier(item["spanId"]),
                item["kind"],
                identifier(item["name"]),
                item["input"],
                item["output"],
            )
        )
    if len({entry.span_id for entry in result}) != len(result):
        raise ReplayError("Replay span identities must be unique.")
    return ReplayRecording(identifier(fields["sourceTraceId"]), tuple(result))


def parse_json_bytes(content: bytes, *, limit=MAX_RECORDING_BYTES):
    if len(content) > limit:
        raise ReplayError("Replay file exceeds its size limit.")

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ReplayError("Replay JSON contains duplicate object keys.")
            result[key] = value
        return result

    try:
        value = json.loads(content, object_pairs_hook=pairs)
        return json_value(value, limit=limit)
    except (ValueError, TypeError, RecursionError, UnicodeError):
        raise ReplayError("Replay file must contain bounded, valid JSON.") from None
