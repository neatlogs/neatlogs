"""Convert complete frozen trace evidence into the shared replay recording."""

from __future__ import annotations

import math
import re
from collections import Counter
from datetime import datetime, timedelta, timezone

from ..core.error_sanitization import sanitize_error_text
from .contracts import (
    MAX_REPLAY_ENTRIES,
    ReplayError,
    identifier,
    json_value,
    parse_json_bytes,
    parse_replay_recording,
)
from .provider import MODEL_CAPTURE_ATTRIBUTE, read_provider_capture, safe_body

MAX_FROZEN_CASE_SPANS = MAX_REPLAY_ENTRIES
_STEP_REPLAY_KINDS = frozenset({"RETRIEVER", "RERANKER", "VECTOR_STORE", "MEMORY", "GUARDRAIL"})
_KIND_ALIASES = {
    "TOOL_CALL": "TOOL",
    "MCP": "MCP_TOOL",
    "RE_RANKER": "RERANKER",
    "VECTOR": "VECTOR_STORE",
}
_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
_ISO_TS_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2})(?:\.(\d+))?(Z|[+-]\d{2}:?\d{2})?$",
    re.ASCII,
)
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
        value = parse_json_bytes(value.encode(), allow_int64=True)
    if type(value) is not dict:
        raise ReplayError("Saved execution metadata must be an object.")
    return value


def _merge_attributes(span):
    attrs = _object(span.get("attributes"))
    meta = span.get("span_metadata")
    if type(meta) is dict:
        return {**meta, **attrs}
    return attrs


def _span_id_value(span):
    sid = span.get("span_id")
    if type(sid) is str and sid:
        return sid
    alt = span.get("spanId")
    if type(alt) is str and alt:
        return alt
    return sid


def _parent_span_id_value(span):
    pid = span.get("parent_span_id")
    if type(pid) is str and pid:
        return pid
    alt = span.get("parentSpanId")
    if type(alt) is str and alt:
        return alt
    return pid


def _first(*values):
    return next(
        (value for value in values if value is not None and value is not _MISSING), _MISSING
    )


def _first_io(*candidates: tuple[object, bool]):
    for value, decode_json_string in candidates:
        if value is not None and value is not _MISSING:
            return value, decode_json_string
    return _MISSING, False


def _stored(value):
    if type(value) is str:
        try:
            return parse_json_bytes(value.encode())
        except ReplayError:
            return value
    return json_value(value)


def _decode_io(value, *, decode_json_string: bool):
    return _stored(value) if decode_json_string else json_value(value)


def _complete(span, attributes, *, allow_error: bool = False):
    status = span.get("status")
    if type(status) is dict:
        status = status.get("code")
    status = getattr(status, "name", status)
    status_code = getattr(span.get("status_code"), "name", span.get("status_code"))
    states = [
        span.get(key) for key in ("payload_state", "input_content_state", "output_content_state")
    ]
    flags = [
        span.get("redacted"),
        span.get("masked"),
        span.get("input_truncated"),
        span.get("output_truncated"),
    ] + [
        attributes.get(key)
        for key in (
            "pii.server_redacted",
            "pii.server_redaction_failed",
            "neatlogs.payload.truncated",
            "neatlogs.capture.truncated",
            "neatlogs.stream.incomplete",
        )
    ]
    capture_fidelity = attributes.get("neatlogs.capture_fidelity")
    base_ok = (
        not any(state in _UNAVAILABLE for state in states if type(state) is str)
        and not any(flag is True or flag == "true" for flag in flags)
        and not (type(capture_fidelity) is str and capture_fidelity in _UNAVAILABLE)
        and attributes.get("pii.server_redaction_status") != "redacted"
    )
    if not base_ok:
        return False
    if allow_error:
        return True
    return (
        status not in ("ERROR", 2, "2")
        and status_code not in ("ERROR", 2, "2")
        and span.get("error") in (None, "")
        and span.get("error_message") in (None, "")
    )


def _validated_step_seq(span) -> int | None:
    raw = None
    try:
        attrs = _merge_attributes(span)
    except ReplayError:
        attrs = span.get("span_metadata") if type(span.get("span_metadata")) is dict else {}
    if "neatlogs.step.seq" in attrs:
        raw = attrs["neatlogs.step.seq"]
    if type(raw) is int and raw >= 1:
        return raw
    if type(raw) is str:
        text = raw.strip()
        if text.isascii() and text.isdigit():
            try:
                parsed = int(text)
            except ValueError:
                return None
            if parsed >= 1:
                return parsed
    return None


def _parse_start_ts_ns(span) -> int | None:
    raw = None
    for key in (
        "start_time_unix_nano",
        "start_time",
        "startTime",
        "startedAt",
        "createdAt",
    ):
        candidate = span.get(key)
        if candidate is not None and candidate != "":
            raw = candidate
            break
    if raw is None or type(raw) is bool:
        return None
    if type(raw) is int:
        return raw
    if type(raw) is float:
        return int(raw) if math.isfinite(raw) else None
    if type(raw) is str:
        text = raw.strip()
        if not text:
            return None
        if text.isascii() and text.isdigit():
            try:
                return int(text)
            except ValueError:
                return None
        match = _ISO_TS_RE.match(text)
        if match:
            base, frac, tz = match.groups()
            frac = frac or ""
            micro = frac[:6].ljust(6, "0") if frac else ""
            tz_norm = "+00:00" if tz == "Z" else (tz or "+00:00")
            if len(tz_norm) == 5 and tz_norm[0] in "+-":
                tz_norm = f"{tz_norm[:3]}:{tz_norm[3:]}"
            iso = f"{base}{'.' + micro if micro else ''}{tz_norm}"
            try:
                extra_ns = int(frac[6:9].ljust(3, "0")) if len(frac) > 6 else 0
                dt = datetime.fromisoformat(iso)
                delta = dt.astimezone(timezone.utc) - _EPOCH
                return (delta // timedelta(microseconds=1)) * 1_000 + extra_ns
            except ValueError:
                return None
        try:
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            delta = dt.astimezone(timezone.utc) - _EPOCH
            return (delta // timedelta(microseconds=1)) * 1_000
        except ValueError:
            return None
    return None


def _sort_replay_candidates(
    candidates: list[tuple[int, int | None, int | None, dict]],
) -> list[dict]:
    if not candidates:
        return []

    all_have_seq = all(seq is not None for _, seq, _, _ in candidates)
    if all_have_seq:
        # When every eligible replay entry carries a valid step.seq, step.seq is the
        # primary deterministic order (taking precedence over skewed/clock-jitter timestamps).
        return [
            entry
            for _, _, _, entry in sorted(
                candidates,
                key=lambda item: (
                    item[1],
                    0 if item[2] is not None else 1,
                    item[2] if item[2] is not None else 0,
                    item[0],
                ),
            )
        ]

    any_have_ts = any(ts is not None for _, _, ts, _ in candidates)
    if not any_have_ts:
        return [entry for _, _, _, entry in candidates]

    # Mixed or unsequenced entries: sort chronologically by start timestamp so an earlier
    # unsequenced call is never moved behind a later sequenced call; break equal-timestamp
    # ties with step.seq only when every entry at that timestamp carries a valid step.seq.
    ts_group_all_seq: dict[int, bool] = {}
    for _, seq, ts, _ in candidates:
        if ts is not None:
            ts_group_all_seq[ts] = ts_group_all_seq.get(ts, True) and (seq is not None)

    def sort_key(
        item: tuple[int, int | None, int | None, dict],
    ) -> tuple[int, int, int, int]:
        orig_idx, seq, ts, _ = item
        if ts is not None:
            tie = seq if (ts_group_all_seq.get(ts) and seq is not None) else orig_idx
            return (0, ts, tie, orig_idx)
        return (1, 0, orig_idx, orig_idx)

    return [entry for _, _, _, entry in sorted(candidates, key=sort_key)]


def _normalize_kind(kind):
    if isinstance(kind, str):
        upper = kind.strip().upper()
        return _KIND_ALIASES.get(upper, upper)
    return ""


def _resolve_span_kind(span, attributes, semantic):
    for candidate in (
        semantic.get("kind"),
        span.get("spanType"),
        span.get("span_type"),
        span.get("node_type"),
        span.get("kind"),
        attributes.get("neatlogs.span.kind"),
        attributes.get("openinference.span.kind"),
        span.get("type"),
    ):
        if isinstance(candidate, str) and candidate.strip():
            return _normalize_kind(candidate)
    return ""


def _extract_tool_entry(span, attributes, semantic, kind, span_id, *, error_record=None):
    if semantic.get("kind") and error_record is None:
        call, result, definition = (
            _object(semantic.get(key)) for key in ("call", "result", "definition")
        )
        if (
            not call.get("id")
            or result.get("call_id") != call["id"]
            or definition.get("name") != call.get("name")
            or result.get("is_error") is not False
            or "arguments" not in call
            or call["arguments"] is None
            or "value" not in result
            or result.get("media")
        ):
            raise ReplayError("Saved tool result has inconsistent linkage.")
        name, input, output = (
            call.get("name"),
            json_value(call["arguments"]),
            json_value(result["value"]),
        )
    else:
        data = span.get("data") if type(span.get("data")) is dict else {}
        name = _first(
            span.get("tool_name"),
            attributes.get("tool.name"),
            attributes.get("neatlogs.tool.name"),
            attributes.get("neatlogs.raw.tool.name"),
            span.get("name"),
            span.get("span_name"),
            span.get("spanName"),
            span.get("node_name"),
        )
        raw_in, decode_in = _first_io(
            (attributes.get("neatlogs.tool.input"), True),
            (attributes.get("neatlogs.raw.tool.input"), True),
            (attributes.get("input.value"), True),
            (span.get("tool_input"), True),
            (span.get("input_value", _MISSING), True),
            (span.get("inputValue", _MISSING), True),
            (span.get("input", _MISSING), False),
            (data.get("tool_input", _MISSING), True),
            (data.get("tool_arguments", _MISSING), True),
            (data.get("input_value", _MISSING), True),
            (data.get("input", _MISSING), True),
        )
        raw_out, decode_out = _first_io(
            (attributes.get("neatlogs.tool.output"), True),
            (attributes.get("neatlogs.raw.tool.output"), True),
            (attributes.get("output.value"), True),
            (span.get("tool_output"), True),
            (span.get("output_value", _MISSING), True),
            (span.get("outputValue", _MISSING), True),
            (span.get("output", _MISSING), False),
            (data.get("tool_output", _MISSING), True),
            (data.get("tool_result", _MISSING), True),
            (data.get("output_value", _MISSING), True),
            (data.get("output", _MISSING), True),
            (data.get("result", _MISSING), True),
        )
        if raw_in is _MISSING or (raw_out is _MISSING and error_record is None):
            raise ReplayError("Saved tool replay entry is missing I/O.")
        input = _decode_io(raw_in, decode_json_string=decode_in)
        output = (
            None
            if (raw_out is _MISSING and error_record is not None)
            else _decode_io(raw_out, decode_json_string=decode_out)
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
    return entry_dict


def validate_tool_span_for_replay(span: dict, span_id_counts: Counter) -> bool:
    """Return True iff the default v1 replay builder would accept this tool span."""
    try:
        semantic = _object(span.get("semantic"))
        attributes = _merge_attributes(span)
        kind = _resolve_span_kind(span, attributes, semantic)
        if kind not in ("TOOL", "MCP_TOOL"):
            return False
        span_id = identifier(_span_id_value(span))
        if span_id_counts[span_id] != 1 or not _complete(span, attributes, allow_error=False):
            return False
        _extract_tool_entry(span, attributes, semantic, kind, span_id, error_record=None)
        return True
    except (ReplayError, ValueError, TypeError, KeyError):
        return False


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


def _span_kind_best_effort(span: dict) -> str:
    try:
        semantic = _object(span.get("semantic"))
    except ReplayError:
        semantic = {}
    try:
        attributes = _merge_attributes(span)
    except ReplayError:
        attributes = span.get("span_metadata") if type(span.get("span_metadata")) is dict else {}
    return _resolve_span_kind(span, attributes, semantic)


def replay_recording_from_case(value, *, include_steps: bool = False, include_errors: bool = False):
    item = json_value(value, allow_int64=True)
    if type(item) is not dict or type(item.get("source")) is not dict:
        raise ReplayError("A frozen case needs its recorded source trace.")
    source = identifier(item["source"].get("traceId"))
    context = _object(item.get("context"))
    spans = context.get("spans")
    if (
        type(spans) is not list
        or len(spans) > MAX_FROZEN_CASE_SPANS
        or any(type(span) is not dict for span in spans)
    ):
        raise ReplayError("A frozen case needs a bounded saved span list.")
    if item["source"].get("kind") == "span":
        selected = identifier(item["source"].get("spanId"))
        if sum(_span_id_value(span) == selected for span in spans) != 1:
            raise ReplayError("Selected span is missing or ambiguous.")
        children = {}
        for span in spans:
            sid = _span_id_value(span)
            if type(sid) is str:
                children.setdefault(_parent_span_id_value(span), []).append(sid)
        subtree, pending = set(), [selected]
        while pending:
            identity = pending.pop()
            if identity not in subtree:
                subtree.add(identity)
                pending.extend(children.get(identity, []))
        spans = [span for span in spans if _span_id_value(span) in subtree]
    counts = Counter(
        sid for sid in (_span_id_value(span) for span in spans) if type(sid) is str and sid
    )
    candidates: list[tuple[int, int | None, int | None, dict]] = []
    skipped = 0
    for orig_idx, span in enumerate(spans):
        kind = _span_kind_best_effort(span)
        try:
            semantic = _object(span.get("semantic"))
            attributes = _merge_attributes(span)
            kind = _resolve_span_kind(span, attributes, semantic)
            is_step = include_steps and kind in _STEP_REPLAY_KINDS
            if kind not in ("LLM", "TOOL", "MCP_TOOL") and not is_step:
                continue
            span_id = identifier(_span_id_value(span))
            allow_err = include_errors and kind != "LLM"
            if counts[span_id] != 1 or not _complete(span, attributes, allow_error=allow_err):
                raise ReplayError("Saved replay execution is incomplete or ambiguous.")
            error_record = _span_error_record(span, attributes) if allow_err else None
            if kind == "LLM":
                captured = read_provider_capture(attributes.get(MODEL_CAPTURE_ATTRIBUTE))
                entry_dict = {"spanId": span_id, "kind": "model", **captured}
            elif is_step:
                kind_lower = kind.lower()
                data = span.get("data") if type(span.get("data")) is dict else {}
                name = _first(
                    span.get("name"),
                    span.get("span_name"),
                    span.get("spanName"),
                    span.get("node_name"),
                )
                raw_in, decode_in = _first_io(
                    (attributes.get(f"neatlogs.{kind_lower}.input"), True),
                    (attributes.get("input.value"), True),
                    (span.get("input_value", _MISSING), True),
                    (span.get("inputValue", _MISSING), True),
                    (span.get("input", _MISSING), False),
                    (data.get("input_value", _MISSING), True),
                    (data.get("input", _MISSING), True),
                )
                raw_out, decode_out = _first_io(
                    (attributes.get(f"neatlogs.{kind_lower}.output"), True),
                    (attributes.get("output.value"), True),
                    (span.get("output_value", _MISSING), True),
                    (span.get("outputValue", _MISSING), True),
                    (span.get("output", _MISSING), False),
                    (data.get("output_value", _MISSING), True),
                    (data.get("output", _MISSING), True),
                )
                if raw_in is _MISSING or (raw_out is _MISSING and error_record is None):
                    raise ReplayError("Saved step replay entry is missing I/O.")
                input = _decode_io(raw_in, decode_json_string=decode_in)
                output = (
                    None
                    if (raw_out is _MISSING and error_record is not None)
                    else _decode_io(raw_out, decode_json_string=decode_out)
                )
                safe_body(input, provider_transport=False)
                safe_body(output, provider_transport=False)
                entry_dict = {
                    "spanId": span_id,
                    "kind": kind_lower,
                    "name": identifier(name),
                    "input": input,
                    "output": output,
                }
                if error_record is not None:
                    entry_dict["error"] = error_record
            else:
                entry_dict = _extract_tool_entry(
                    span, attributes, semantic, kind, span_id, error_record=error_record
                )
            candidates.append(
                (orig_idx, _validated_step_seq(span), _parse_start_ts_ns(span), entry_dict)
            )
        except (ReplayError, ValueError, TypeError, KeyError):
            if kind in ("TOOL", "MCP_TOOL"):
                skipped += 1
    entries = _sort_replay_candidates(candidates)
    version = 2 if (include_steps or include_errors) else 1
    return {
        "recording": parse_replay_recording(
            {"version": version, "sourceTraceId": source, "entries": entries}
        ),
        "skippedTools": skipped,
    }
