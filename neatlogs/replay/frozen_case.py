"""Convert complete frozen trace evidence into the shared replay recording."""

from __future__ import annotations

from collections import Counter

from ..core.error_sanitization import sanitize_error_text
from .contracts import ReplayError, identifier, json_value, parse_json_bytes, parse_replay_recording
from .provider import MODEL_CAPTURE_ATTRIBUTE, read_provider_capture, safe_body

_STEP_REPLAY_KINDS = frozenset({"RETRIEVER", "RERANKER", "VECTOR_STORE", "MEMORY", "GUARDRAIL"})
_MISSING = object()
_UNAVAILABLE = {
    "unavailable",
    "retentionExpired",
    "truncated",
    "previewOnly",
    "redacted",
    "masked",
    "lossy",
}


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


def _complete(span, attributes, *, allow_error: bool = False):
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
    base_ok = (
        not any(state in _UNAVAILABLE for state in states if type(state) is str)
        and not any(flag is True or flag == "true" for flag in flags)
        and attributes.get("neatlogs.capture_fidelity") not in _UNAVAILABLE
        and attributes.get("pii.server_redaction_status") != "redacted"
    )
    if not base_ok:
        return False
    if allow_error:
        return True
    return (
        status not in ("ERROR", 2, "2")
        and span.get("status_code") not in ("ERROR", 2, "2")
        and span.get("error") in (None, "")
        and span.get("error_message") in (None, "")
    )


def _span_error_record(span, attributes):
    status = span.get("status")
    status_desc = ""
    if type(status) is dict:
        status_desc = str(status.get("description") or "")
        status = status.get("code")
    is_error = (
        status in ("ERROR", 2, "2")
        or span.get("status_code") in ("ERROR", 2, "2")
        or span.get("error") not in (None, "")
        or span.get("error_message") not in (None, "")
    )
    if not is_error:
        return None
    exc_attrs = {}
    events = span.get("events")
    if isinstance(events, list):
        for ev in events:
            if isinstance(ev, dict) and ev.get("name") == "exception":
                ev_attrs = ev.get("attributes")
                if isinstance(ev_attrs, dict):
                    exc_attrs = ev_attrs
                    break
    err_type = str(
        attributes.get("exception.type")
        or attributes.get("error.type")
        or exc_attrs.get("exception.type")
        or (span.get("error").get("type") if isinstance(span.get("error"), dict) else "")
        or "RecordedError"
    )[:128]
    raw_msg = (
        span.get("error_message")
        or (
            span.get("error").get("message")
            if isinstance(span.get("error"), dict)
            else span.get("error")
        )
        or attributes.get("exception.message")
        or attributes.get("error.message")
        or exc_attrs.get("exception.message")
        or status_desc
        or "Recorded span error"
    )
    return {"type": err_type, "message": sanitize_error_text(str(raw_msg))[:1024]}


def replay_recording_from_case(value, *, include_steps: bool = False, include_errors: bool = False):
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
    if item["source"].get("kind") == "span":
        selected = identifier(item["source"].get("spanId"))
        if sum(span.get("span_id") == selected for span in spans) != 1:
            raise ReplayError("Selected span is missing or ambiguous.")
        children = {}
        for span in spans:
            if type(span.get("span_id")) is str:
                children.setdefault(span.get("parent_span_id"), []).append(span["span_id"])
        subtree, pending = set(), [selected]
        while pending:
            identity = pending.pop()
            if identity not in subtree:
                subtree.add(identity)
                pending.extend(children.get(identity, []))
        spans = [span for span in spans if span.get("span_id") in subtree]
    counts = Counter(span.get("span_id") for span in spans if type(span.get("span_id")) is str)
    entries, skipped = [], 0
    for span in spans:
        kind = span.get("span_type")
        try:
            semantic = _object(span.get("semantic"))
            kind = semantic.get("kind", kind)
            if isinstance(kind, str):
                kind = kind.upper()
                if kind == "TOOL_CALL":
                    kind = "TOOL"
            attributes = _object(span.get("attributes"))
            is_step = include_steps and kind in _STEP_REPLAY_KINDS
            if kind not in ("LLM", "TOOL", "MCP_TOOL") and not is_step:
                continue
            span_id = identifier(span.get("span_id"))
            allow_err = include_errors and kind != "LLM"
            if counts[span_id] != 1 or not _complete(span, attributes, allow_error=allow_err):
                raise ReplayError("Saved replay execution is incomplete or ambiguous.")
            error_record = _span_error_record(span, attributes) if allow_err else None
            if kind == "LLM":
                captured = read_provider_capture(attributes.get(MODEL_CAPTURE_ATTRIBUTE))
                entries.append({"spanId": span_id, "kind": "model", **captured})
                continue
            if is_step:
                kind_lower = kind.lower()
                name = _first(
                    span.get("name"),
                    span.get("span_name"),
                    span.get("spanName"),
                )
                raw_in = _first(
                    attributes.get(f"neatlogs.{kind_lower}.input"),
                    attributes.get("input.value"),
                    span.get("input_value", _MISSING),
                    span.get("input", _MISSING),
                )
                raw_out = _first(
                    attributes.get(f"neatlogs.{kind_lower}.output"),
                    attributes.get("output.value"),
                    span.get("output_value", _MISSING),
                    span.get("output", _MISSING),
                )
                if raw_in is _MISSING or (raw_out is _MISSING and error_record is None):
                    raise ReplayError("Saved step replay entry is missing I/O.")
                input = _stored(raw_in)
                output = (
                    None if (raw_out is _MISSING and error_record is not None) else _stored(raw_out)
                )
                safe_body(input, provider_transport=False)
                safe_body(output, provider_transport=False)
                entry_dict = {
                    "spanId": span_id,
                    "kind": "step",
                    "name": identifier(name),
                    "input": input,
                    "output": output,
                }
                if error_record is not None:
                    entry_dict["error"] = error_record
                entries.append(entry_dict)
                continue
            if semantic.get("kind") and error_record is None:
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
                    attributes.get("tool.name"),
                    attributes.get("neatlogs.tool.name"),
                    attributes.get("neatlogs.raw.tool.name"),
                    span.get("name"),
                    span.get("span_name"),
                )
                raw_in = _first(
                    attributes.get("neatlogs.tool.input"),
                    attributes.get("neatlogs.raw.tool.input"),
                    attributes.get("input.value"),
                    span.get("tool_input"),
                    span.get("input_value", _MISSING),
                )
                raw_out = _first(
                    attributes.get("neatlogs.tool.output"),
                    attributes.get("neatlogs.raw.tool.output"),
                    attributes.get("output.value"),
                    span.get("tool_output"),
                    span.get("output_value", _MISSING),
                )
                if raw_in is _MISSING or (raw_out is _MISSING and error_record is None):
                    raise ReplayError("Saved tool replay entry is missing I/O.")
                input = _stored(raw_in)
                output = (
                    None if (raw_out is _MISSING and error_record is not None) else _stored(raw_out)
                )
                if (
                    error_record is None
                    and kind == "MCP_TOOL"
                    and type(output) is dict
                    and set(output) == {"result"}
                    and type(output["result"]) is str
                ):
                    raise ReplayError("Legacy MCP string output does not preserve its return type.")
            safe_body(input, provider_transport=False)
            safe_body(output, provider_transport=False)
            entry_dict = {
                "spanId": span_id,
                "kind": "tool",
                "name": identifier(name),
                "input": input,
                "output": output,
            }
            if error_record is not None:
                entry_dict["error"] = error_record
            entries.append(entry_dict)
        except (ReplayError, ValueError, TypeError, KeyError):
            if kind in ("TOOL", "MCP_TOOL"):
                skipped += 1
    return {
        "recording": parse_replay_recording(
            {"version": 1, "sourceTraceId": source, "entries": entries}
        ),
        "skippedTools": skipped,
    }
