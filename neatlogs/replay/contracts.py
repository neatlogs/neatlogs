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
V1_KINDS = frozenset({"tool", "model"})
STEP_KINDS = frozenset({"retriever", "reranker", "vector_store", "memory", "guardrail"})
V2_KINDS = V1_KINDS | STEP_KINDS
Kind = Literal[
    "tool",
    "model",
    "retriever",
    "reranker",
    "vector_store",
    "memory",
    "guardrail",
]
MAX_RECORDING_BYTES = 16 * 1024 * 1024
MAX_INPUT_BYTES = 1024 * 1024
MAX_REPLAY_ENTRIES = 5000


class ReplayError(NeatlogsError, ValueError):
    """Invalid local replay configuration; never includes recorded content or credentials."""


class ReplayRecordedError(NeatlogsError, RuntimeError):
    """Raised when replaying a historically failed span captured with include_errors=True."""

    def __init__(self, error_type: str, message: str = "") -> None:
        self.error_type = str(error_type or "RecordedError")
        self.error_message = str(message or "")
        detail = (
            f"{self.error_type}: {self.error_message}" if self.error_message else self.error_type
        )
        super().__init__(detail)


def json_value(
    value: object, *, limit: int = MAX_RECORDING_BYTES, allow_int64: bool = False
) -> Json:
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
                if allow_int64 and -(2**63) <= item <= 2**64 - 1:
                    return item
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

    result = visit(value, 0)
    try:
        if allow_int64:
            encoded = json.dumps(
                result, allow_nan=False, ensure_ascii=False, separators=(",", ":")
            ).encode("utf-8")
        else:
            encoded = rfc8785.dumps(result)
    except (rfc8785.CanonicalizationError, ValueError, TypeError, UnicodeError, OverflowError):
        raise ReplayError("Replay value is not lossless JSON.") from None
    if len(encoded) > limit:
        raise ReplayError("Replay JSON exceeds its size limit.")
    return result


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
    error: dict[str, Json] | None = None

    def to_dict(self):
        out = {
            "spanId": self.span_id,
            "kind": self.kind,
            "name": self.name,
            "input": self.input,
            "output": self.output,
        }
        if self.error is not None:
            out["error"] = self.error
        return out


@dataclass(frozen=True)
class ReplayRecording:
    source_trace_id: str
    entries: tuple[ReplayEntry, ...]
    version: int = 1

    def to_dict(self):
        return {
            "version": self.version,
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
    version = fields["version"]
    if type(version) is not int or version not in (1, 2):
        raise ReplayError("Unsupported local replay recording version.")
    entries = fields["entries"]
    if type(entries) is not list or len(entries) > MAX_REPLAY_ENTRIES:
        raise ReplayError(f"Replay recordings allow at most {MAX_REPLAY_ENTRIES} entries.")
    allowed_kinds = V1_KINDS if version == 1 else V2_KINDS
    optional_fields = frozenset() if version == 1 else frozenset({"error"})
    result = []
    for raw in entries:
        item = object_fields(
            raw, {"spanId", "kind", "name", "input", "output"}, optional=optional_fields
        )
        if type(item["kind"]) is not str or item["kind"] not in allowed_kinds:
            raise ReplayError("Unsupported local replay entry kind.")
        raw_error = item.get("error")
        parsed_error = None
        if raw_error is not None:
            err_obj = object_fields(raw_error, {"message"}, optional={"type"})
            if type(err_obj["message"]) is not str or (
                "type" in err_obj and type(err_obj["type"]) is not str
            ):
                raise ReplayError("Replay entry error fields must be strings.")
            parsed_error = dict(err_obj)
        result.append(
            ReplayEntry(
                identifier(item["spanId"]),
                item["kind"],
                identifier(item["name"]),
                item["input"],
                item["output"],
                parsed_error,
            )
        )
    if len({entry.span_id for entry in result}) != len(result):
        raise ReplayError("Replay span identities must be unique.")
    return ReplayRecording(identifier(fields["sourceTraceId"]), tuple(result), version=version)


def parse_json_bytes(content: bytes, *, limit=MAX_RECORDING_BYTES, allow_int64: bool = False):
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
        return json_value(value, limit=limit, allow_int64=allow_int64)
    except ReplayError:
        raise
    except (ValueError, TypeError, RecursionError, UnicodeError):
        raise ReplayError("Replay file must contain bounded, valid JSON.") from None
