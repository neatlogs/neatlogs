"""Convert complete frozen trace evidence into the shared replay recording."""

from __future__ import annotations

from collections import Counter

from .contracts import ReplayError, identifier, json_value, parse_json_bytes, parse_replay_recording
from .provider import MODEL_CAPTURE_ATTRIBUTE, read_provider_capture

_MISSING = object()
_UNAVAILABLE = {"unavailable", "retentionExpired", "truncated", "previewOnly", "redacted", "masked"}


def _object(value):
    if value is None:
        return {}
    if type(value) is str:
        value = parse_json_bytes(value.encode())
    if type(value) is not dict:
        raise ReplayError("Saved execution metadata must be an object.")
    return value


def _first(*values):
    return next(
        (value for value in values if value is not None and value is not _MISSING), _MISSING
    )


def _stored(value):
    if type(value) is str:
        try:
            return parse_json_bytes(value.encode())
        except ReplayError:
            return value
    return json_value(value)


def _complete(span, attributes):
    status = span.get("status")
    if type(status) is dict:
        status = status.get("code")
    states = [
        span.get(key) for key in ("payload_state", "input_content_state", "output_content_state")
    ]
    flags = [span.get("redacted"), span.get("masked")] + [
        attributes.get(key)
        for key in (
            "pii.server_redacted",
            "pii.server_redaction_failed",
            "neatlogs.payload.truncated",
            "neatlogs.capture.truncated",
            "neatlogs.stream.incomplete",
        )
    ]
    return (
        not any(state in _UNAVAILABLE for state in states if type(state) is str)
        and not any(flag is True or flag == "true" for flag in flags)
        and attributes.get("neatlogs.capture_fidelity") not in _UNAVAILABLE
        and attributes.get("pii.server_redaction_status") != "redacted"
        and status not in ("ERROR", 2, "2")
        and span.get("status_code") not in ("ERROR", 2, "2")
        and span.get("error") in (None, "")
        and span.get("error_message") in (None, "")
    )


def replay_recording_from_case(value):
    item = json_value(value)
    if type(item) is not dict or type(item.get("source")) is not dict:
        raise ReplayError("A frozen case needs its recorded source trace.")
    source = identifier(item["source"].get("traceId"))
    context = _object(item.get("context"))
    spans = context.get("spans")
    if (
        type(spans) is not list
        or len(spans) > 5000
        or any(type(span) is not dict for span in spans)
    ):
        raise ReplayError("A frozen case needs a bounded saved span list.")
    counts = Counter(span.get("span_id") for span in spans if type(span.get("span_id")) is str)
    entries, skipped = [], 0
    for span in spans:
        kind = span.get("span_type")
        try:
            semantic = _object(span.get("semantic"))
            kind = semantic.get("kind", kind)
            attributes = _object(span.get("attributes"))
            if kind not in ("LLM", "TOOL", "MCP_TOOL"):
                continue
            span_id = identifier(span.get("span_id"))
            if counts[span_id] != 1 or not _complete(span, attributes):
                raise ReplayError("Saved replay execution is incomplete or ambiguous.")
            if kind == "LLM":
                captured = read_provider_capture(attributes.get(MODEL_CAPTURE_ATTRIBUTE))
                entries.append({"spanId": span_id, "kind": "model", **captured})
                continue
            if semantic.get("kind"):
                call, result, definition = (
                    _object(semantic.get(key)) for key in ("call", "result", "definition")
                )
                if (
                    not call.get("id")
                    or result.get("call_id") != call["id"]
                    or definition.get("name") != call.get("name")
                    or result.get("is_error") is not False
                    or "value" not in result
                    or result.get("media")
                ):
                    raise ReplayError("Saved tool result has inconsistent linkage.")
                name, input, output = (
                    call.get("name"),
                    json_value(call.get("arguments")),
                    json_value(result["value"]),
                )
            else:
                name = _first(
                    span.get("tool_name"),
                    attributes.get("neatlogs.tool.name"),
                    attributes.get("neatlogs.raw.tool.name"),
                )
                input = _stored(
                    _first(
                        attributes.get("neatlogs.tool.input"),
                        attributes.get("neatlogs.raw.tool.input"),
                        span.get("tool_input"),
                        span.get("input_value", _MISSING),
                    )
                )
                output = _stored(
                    _first(
                        attributes.get("neatlogs.tool.output"),
                        attributes.get("neatlogs.raw.tool.output"),
                        span.get("tool_output"),
                        span.get("output_value", _MISSING),
                    )
                )
                if (
                    kind == "MCP_TOOL"
                    and type(output) is dict
                    and set(output) == {"result"}
                    and type(output["result"]) is str
                ):
                    raise ReplayError("Legacy MCP string output does not preserve its return type.")
            entries.append(
                {
                    "spanId": span_id,
                    "kind": "tool",
                    "name": identifier(name),
                    "input": input,
                    "output": output,
                }
            )
        except (ReplayError, ValueError, TypeError, KeyError):
            if kind in ("TOOL", "MCP_TOOL"):
                skipped += 1
    return {
        "recording": parse_replay_recording(
            {"version": 1, "sourceTraceId": source, "entries": entries}
        ),
        "skippedTools": skipped,
    }
