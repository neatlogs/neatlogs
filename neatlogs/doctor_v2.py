"""Doctor v2 envelope capture, validation, export, and exact trace read-back."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence
from urllib.parse import quote, urlparse

import requests
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.trace import StatusCode, TraceFlags, get_current_span

from .schema_v2 import TELEMETRY_SCHEMA_VERSION
from .version import __version__

DOCTOR_V2_FORMAT_VERSION = "neatlogs.doctor/v2"
_TRACE_ID = re.compile(r"^[0-9a-f]{32}$")
_SPAN_ID = re.compile(r"^[0-9a-f]{16}$")
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_MAX_CAPTURED_TRACES = 16
_MAX_CAPTURED_SPANS_PER_TRACE = 64
_MAX_CAPTURED_BYTES_PER_TRACE = 256 * 1024
_MAX_CAPTURED_BYTES_TOTAL = 1024 * 1024
_capture_lock = threading.RLock()
_captures: OrderedDict[str, dict[str, dict[str, Any]]] = OrderedDict()
_capture_sizes: dict[str, int] = {}
_capture_span_sizes: dict[str, dict[str, int]] = {}
_latest_trace_id: str | None = None


def _canonical_span_kind(value: Any) -> str:
    return str(value or "").removeprefix("Neatlogs.").upper()


def _mark_doctor_span(span_type: str) -> None:
    from ._wrap_utils import active_neatlogs_context

    context = active_neatlogs_context()
    span = get_current_span(context) if context is not None else get_current_span()
    span.set_attributes(
        {
            "neatlogs.doctor": True,
            "neatlogs.doctor.version": "v1",
            "service.name": "neatlogs.doctor.v2",
            "telemetry.sdk.language": "python",
            "telemetry.sdk.version": __version__,
            "neatlogs.span.kind": span_type.lower(),
        }
    )


# This maps every reason emitted by this SDK version. It is intentionally not an
# exhaustive registry for future/backend reason codes: unknown codes fail safely
# to CONTACT_SUPPORT so a newer server can never trigger an unsafe client action.
_REMEDIATION = {
    "CREDENTIAL_MISSING": "SET_CREDENTIAL",
    "AUTH_FAILED": "CHECK_INGEST_CREDENTIAL",
    "BACKEND_PROBE_UNAVAILABLE": "CHECK_TRACE_ENDPOINT",
    "ENDPOINT_INVALID": "SET_ENDPOINT",
    "PROVIDER_OWNERSHIP_AMBIGUOUS": "USE_PRIVATE_PROVIDER",
    "TRACE_ID_INVALID": "RECREATE_TRACE",
    "SPAN_ID_INVALID": "RECREATE_SPAN",
    "SPAN_ID_DUPLICATE": "RECREATE_SPAN",
    "PARENT_ID_INVALID": "FIX_PARENT_CONTEXT",
    "PARENT_MISSING": "FIX_PARENT_CONTEXT",
    "ROOT_MISSING": "CREATE_ROOT_SPAN",
    "ROOT_MULTIPLE": "USE_SINGLE_ROOT",
    "ROOT_NOT_ENDED": "END_ROOT_SPAN",
    "INPUT_JSON_INVALID": "SERIALIZE_INPUT_JSON",
    "OUTPUT_JSON_INVALID": "SERIALIZE_OUTPUT_JSON",
    "TOOL_CALL_MISSING": "CAPTURE_TOOL_REQUEST",
    "TOOL_EXECUTION_MISSING": "CAPTURE_TOOL_EXECUTION",
    "CHOICE_LOSS": "PRESERVE_ALL_CHOICES",
    "STREAM_FRAGMENT_MISSING": "PRESERVE_STREAM_FRAGMENTS",
    "PAYLOAD_ATTACHMENT_REQUIRED": "UPLOAD_PAYLOAD_ATTACHMENT",
    "SAMPLING_INCONSISTENT": "FIX_PARENT_BASED_SAMPLING",
    "QUEUE_SATURATED": "INCREASE_OR_DRAIN_QUEUE",
    "EXPORT_RETRY_EXHAUSTED": "CHECK_TRANSPORT",
    "FLUSH_TIMEOUT": "INCREASE_FLUSH_BUDGET",
    "MASKING_FAILED_CLOSED": "FIX_MASK_CALLBACK",
    "LOCAL_ENVELOPE_VALID": "NONE",
    "PROBE_FIXTURE_INVALID": "FIX_DOCTOR_INSTRUMENTATION",
}


def _json_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text or text[0] not in "[{":
        return value
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return value


def _first(attributes: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in attributes:
            return _json_value(attributes[key])
    return None


def _tool_calls(attributes: Mapping[str, Any]) -> list[dict[str, Any]] | None:
    """Read only canonical indexed keys from attribute-mapping.json."""
    exploded: dict[int, dict[str, Any]] = {}
    for key, item in attributes.items():
        match = re.fullmatch(
            r"neatlogs\.llm\.tool_calls\.(\d+)\."
            r"(id|name|arguments|choice_index|tool_call_index)$",
            key,
        )
        if match:
            exploded.setdefault(int(match.group(1)), {})[match.group(2)] = _json_value(item)
    calls = [exploded[index] for index in sorted(exploded) if "id" in exploded[index]]
    return calls or None


def _choices(attributes: Mapping[str, Any]) -> list[dict[str, Any]]:
    indexes: set[int] = set()
    messages: dict[int, dict[str, Any]] = {}
    finishes: dict[int, Any] = {}
    for key, value in attributes.items():
        match = re.fullmatch(r"neatlogs\.llm\.output_messages\.(\d+)\.(.+)", key)
        if match:
            index = int(match.group(1))
            indexes.add(index)
            messages.setdefault(index, {})[match.group(2)] = _json_value(value)
            continue
        match = re.fullmatch(r"neatlogs\.llm\.choices\.(\d+)\.finish_reason", key)
        if match:
            index = int(match.group(1))
            indexes.add(index)
            finishes[index] = _json_value(value)
    choices = []
    for index in sorted(indexes):
        message = dict(messages.get(index, {}))
        choice: dict[str, Any] = {"index": index, "message": message}
        if index in finishes:
            choice["finish_reason"] = finishes[index]
        choices.append(choice)
    return choices


def _stream_fragments(span: ReadableSpan) -> list[Any]:
    fragments = []
    for event in span.events or ():
        if event.name == "neatlogs.stream.chunk" and event.attributes:
            summary = event.attributes.get("neatlogs.stream.chunk.summary")
            if summary is not None:
                fragments.append(_json_value(summary))
    return fragments


def _payload_references(attributes: Mapping[str, Any]) -> list[dict[str, Any]]:
    records: dict[tuple[str, int], dict[str, Any]] = {}
    pattern = re.compile(
        r"^(neatlogs\..+\.media)\.(\d+)\." r"(id|reference|sha256|mime_type|byte_length|state)$"
    )
    for key, value in attributes.items():
        match = pattern.fullmatch(key)
        if match:
            records.setdefault((match.group(1), int(match.group(2))), {})[match.group(3)] = (
                _json_value(value)
            )
    references = []
    for identity in sorted(records):
        record = records[identity]
        digest = record.get("sha256")
        size = record.get("byte_length")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            continue
        references.append(
            {
                "digest": f"sha256:{digest}",
                "size": size if isinstance(size, int) and not isinstance(size, bool) else 0,
                "mime_type": str(record.get("mime_type") or "application/octet-stream"),
            }
        )
    return references


def _diagnostic_span(span: ReadableSpan) -> dict[str, Any]:
    context = span.get_span_context()
    attributes = dict(span.attributes or {})
    kind_value = attributes.get("neatlogs.span.kind", attributes.get("openinference.span.kind"))
    kind = _canonical_span_kind(kind_value or "INTERNAL")
    parent = span.parent
    output: dict[str, Any] = {
        "span_id": f"{context.span_id:016x}",
        "parent_span_id": f"{parent.span_id:016x}" if parent and parent.is_valid else None,
        "name": span.name,
        "kind": kind,
        "status": ("ERROR" if span.status.status_code is StatusCode.ERROR else "OK"),
        "sampled": bool(context.trace_flags & TraceFlags.SAMPLED),
        "ended": span.end_time is not None,
        "start_time_ns": span.start_time,
        "duration_ns": max(0, (span.end_time or span.start_time) - span.start_time),
        "attributes": attributes,
    }
    fields = {
        "input": _first(attributes, (f"neatlogs.{kind.lower()}.input",)),
        "output": _first(attributes, (f"neatlogs.{kind.lower()}.output",)),
    }
    for key, value in fields.items():
        if value is not None:
            output[key] = value
    calls = _tool_calls(attributes)
    if calls:
        output["tool_calls"] = calls
    choices = _choices(attributes)
    if choices:
        output["choices"] = choices
    expected_choices = attributes.get("neatlogs.llm.generation_choices")
    if isinstance(expected_choices, int) and not isinstance(expected_choices, bool):
        output["expected_choice_count"] = expected_choices
    elif isinstance(expected_choices, (list, tuple)):
        output["expected_choice_count"] = len(expected_choices)
    elif choices:
        output["expected_choice_count"] = len(choices)
    call_id = _first(attributes, ("neatlogs.tool_call.id",))
    if kind == "TOOL" and isinstance(call_id, str):
        output["tool_call"] = {"id": call_id}
        if isinstance(attributes.get("neatlogs.tool.name"), str):
            output["tool_call"]["name"] = attributes["neatlogs.tool.name"]
        if "neatlogs.tool.input" in attributes:
            output["tool_call"]["arguments"] = _json_value(attributes["neatlogs.tool.input"])
        if "neatlogs.tool.output" in attributes:
            output["tool_call"]["result"] = _json_value(attributes["neatlogs.tool.output"])
    fragments = _stream_fragments(span)
    if fragments:
        output["stream_fragments"] = fragments
    streaming = bool(
        attributes.get("neatlogs.llm.is_streaming") or attributes.get("neatlogs.tool.is_streaming")
    )
    if streaming:
        output["streaming"] = True
    references = _payload_references(attributes)
    if references:
        output["payload_references"] = references
    if attributes.get("neatlogs.capture.truncated") is True:
        output["oversized"] = True
    return output


def capture_prepared_spans(spans: Sequence[ReadableSpan]) -> None:
    """Capture the final masked spans immediately before the network exporter."""

    global _latest_trace_id
    prepared: list[tuple[str, dict[str, Any], int]] = []
    for span in spans:
        # Completion is a finalizer control record and is folded out of the
        # product trace. Compare Doctor read-back against semantic spans.
        if span.name == "neatlogs.trace.complete":
            continue
        trace_id = f"{span.get_span_context().trace_id:032x}"
        item = _diagnostic_span(span)
        item_size = len(
            json.dumps(_canonical(item), separators=(",", ":"), sort_keys=True).encode("utf-8")
        )
        if item_size <= _MAX_CAPTURED_BYTES_PER_TRACE:
            prepared.append((trace_id, item, item_size))

    with _capture_lock:
        for trace_id, item, item_size in prepared:
            current = _captures.pop(trace_id, {})
            span_sizes = _capture_span_sizes.setdefault(trace_id, {})
            if item["span_id"] not in current and len(current) >= _MAX_CAPTURED_SPANS_PER_TRACE:
                _captures[trace_id] = current
                continue
            previous_size = span_sizes.get(item["span_id"], 0)
            next_size = _capture_sizes.get(trace_id, 0) - previous_size + item_size
            if next_size > _MAX_CAPTURED_BYTES_PER_TRACE:
                _captures[trace_id] = current
                continue
            current[item["span_id"]] = item
            span_sizes[item["span_id"]] = item_size
            _captures[trace_id] = current
            _capture_sizes[trace_id] = next_size
            _latest_trace_id = trace_id
        while len(_captures) > _MAX_CAPTURED_TRACES:
            evicted, _ = _captures.popitem(last=False)
            _capture_sizes.pop(evicted, None)
            _capture_span_sizes.pop(evicted, None)
        while sum(_capture_sizes.values()) > _MAX_CAPTURED_BYTES_TOTAL and _captures:
            evicted, _ = _captures.popitem(last=False)
            _capture_sizes.pop(evicted, None)
            _capture_span_sizes.pop(evicted, None)


def clear_doctor_capture() -> None:
    global _latest_trace_id
    with _capture_lock:
        _captures.clear()
        _capture_sizes.clear()
        _capture_span_sizes.clear()
        _latest_trace_id = None


def get_captured_envelope(trace_id: str | None = None) -> dict[str, Any] | None:
    with _capture_lock:
        selected = trace_id or _latest_trace_id
        spans = list(_captures.get(selected or "", {}).values())
        if not selected or not spans:
            return None
        roots = [span for span in spans if span["parent_span_id"] is None]
        if not roots:
            return None
        return {"trace_id": selected, "root_span_id": roots[0]["span_id"], "spans": spans}


def _canonical(value: Any) -> Any:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("diagnostic envelope contains a non-finite number")
        return 0 if value == 0 else value
    if isinstance(value, Mapping):
        return {str(key): _canonical(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise TypeError("diagnostic envelope contains a non-JSON value")


def doctor_semantic_digest(envelope: Mapping[str, Any]) -> str:
    spans = [dict(item) for item in envelope["spans"]]
    names_by_id = {str(item.get("span_id")): str(item.get("name")) for item in spans}
    stable_fields = (
        "name",
        "kind",
        "status",
        "input",
        "output",
        "choices",
        "expected_choice_count",
        "tool_calls",
        "tool_call",
        "streaming",
        "oversized",
        "payload_references",
        "sampled",
        "ended",
    )
    projection = {
        "spans": sorted(
            [
                {
                    **{key: item[key] for key in stable_fields if key in item},
                    "parent": names_by_id.get(str(item.get("parent_span_id"))),
                }
                for item in spans
                if item.get("name") != "neatlogs.trace.complete"
            ],
            key=lambda item: (str(item.get("name")), str(item.get("kind"))),
        )
    }
    encoded = json.dumps(
        _canonical(projection),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _check(name: str, status: str, reason: str, message: str, details=None) -> dict[str, Any]:
    return {
        "name": name,
        "status": status,
        "reason_code": reason,
        "message": message,
        "remediation_code": _REMEDIATION.get(reason, "CONTACT_SUPPORT"),
        **({"details": details} if details else {}),
    }


def _probe_fixture_check(spans: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    expected = {
        "doctor.probe.root": ("WORKFLOW", None),
        "doctor.probe.agent": ("AGENT", "doctor.probe.root"),
        "doctor.probe.llm": ("LLM", "doctor.probe.agent"),
        "doctor.probe.tool": ("TOOL", "doctor.probe.root"),
    }
    by_name = {str(span.get("name")): span for span in spans}
    if len(spans) != 4 or set(by_name) != set(expected):
        return _check(
            "probe_fixture",
            "fail",
            "PROBE_FIXTURE_INVALID",
            "Doctor must capture exactly the four versioned semantic fixture spans",
        )
    names_by_id = {str(span.get("span_id")): name for name, span in by_name.items()}
    for name, (kind, parent_name) in expected.items():
        span = by_name[name]
        actual_parent = names_by_id.get(str(span.get("parent_span_id")))
        if span.get("kind") != kind or actual_parent != parent_name:
            return _check(
                "probe_fixture",
                "fail",
                "PROBE_FIXTURE_INVALID",
                "Doctor fixture span types or parent edges are incorrect",
                {"span": name},
            )
        attrs = span.get("attributes") if isinstance(span.get("attributes"), Mapping) else {}
        if (
            attrs.get("neatlogs.doctor") is not True
            or attrs.get("neatlogs.doctor.version") != "v1"
            or attrs.get("service.name") != "neatlogs.doctor.v2"
            or attrs.get("telemetry.sdk.language") != "python"
            or attrs.get("telemetry.sdk.version") != __version__
            or _canonical_span_kind(attrs.get("neatlogs.span.kind")) != kind
            or span.get("input") is None
            or span.get("output") is None
        ):
            return _check(
                "probe_fixture",
                "fail",
                "PROBE_FIXTURE_INVALID",
                "Doctor fixture metadata and deterministic input/output must be complete",
                {"span": name},
            )
    return None


def doctor_local_v2(
    envelope: Mapping[str, Any],
    *,
    flush_outcome: str = "success",
    flush_timeout_ms: int = 5000,
    flush_duration_ms: int | None = None,
    private_provider: bool = True,
    sample_rate: float = 1.0,
    queue_capacity: int = 2048,
    dropped_spans: int = 0,
    expected_probe_fixture: bool = False,
) -> dict[str, Any]:
    """Validate one final normalized and masked export envelope."""

    checks: list[dict[str, Any]] = []
    trace_id = envelope.get("trace_id")
    root_id = envelope.get("root_span_id")
    spans = list(envelope.get("spans") or [])
    if not isinstance(trace_id, str) or not _TRACE_ID.fullmatch(trace_id):
        checks.append(
            _check(
                "trace_id",
                "fail",
                "TRACE_ID_INVALID",
                "Trace ID must be 32 lowercase hexadecimal characters",
            )
        )
    ids: set[str] = set()
    for span in spans:
        span_id = span.get("span_id")
        if not isinstance(span_id, str) or not _SPAN_ID.fullmatch(span_id):
            checks.append(
                _check(
                    "span_id",
                    "fail",
                    "SPAN_ID_INVALID",
                    "Span ID must be 16 lowercase hexadecimal characters",
                )
            )
        elif span_id in ids:
            checks.append(
                _check(
                    "span_id",
                    "fail",
                    "SPAN_ID_DUPLICATE",
                    "Span IDs must be unique",
                    {"span_id": span_id},
                )
            )
        ids.add(span_id)
    roots = [span for span in spans if span.get("parent_span_id") is None]
    if not roots or root_id not in ids:
        checks.append(
            _check("root", "fail", "ROOT_MISSING", "Exactly one declared root span is required")
        )
    elif len(roots) != 1 or roots[0].get("span_id") != root_id:
        checks.append(
            _check(
                "root", "fail", "ROOT_MULTIPLE", "The envelope must contain one declared root span"
            )
        )
    elif roots[0].get("ended") is not True:
        checks.append(
            _check("root", "fail", "ROOT_NOT_ENDED", "The root span must be ended before capture")
        )
    for span in spans:
        span_id = span.get("span_id")
        parent = span.get("parent_span_id")
        if parent is not None and (not isinstance(parent, str) or not _SPAN_ID.fullmatch(parent)):
            checks.append(
                _check(
                    "parent",
                    "fail",
                    "PARENT_ID_INVALID",
                    "Parent ID must be a valid span ID",
                    {"span_id": span_id},
                )
            )
        elif parent is not None and parent not in ids:
            checks.append(
                _check(
                    "parent",
                    "fail",
                    "PARENT_MISSING",
                    "Parent span is absent from the envelope",
                    {"span_id": span_id},
                )
            )
        for field, reason in (("input", "INPUT_JSON_INVALID"), ("output", "OUTPUT_JSON_INVALID")):
            if field in span:
                try:
                    _canonical(span[field])
                except (TypeError, ValueError):
                    checks.append(
                        _check(
                            field,
                            "fail",
                            reason,
                            f"Span {field} is not canonical JSON",
                            {"span_id": span_id},
                        )
                    )
        if span.get("expected_choice_count", 0) > len(span.get("choices") or []):
            checks.append(
                _check(
                    "choices",
                    "fail",
                    "CHOICE_LOSS",
                    "The normalized response lost model choices",
                    {"span_id": span_id},
                )
            )
        if span.get("streaming") and not span.get("stream_fragments"):
            checks.append(
                _check(
                    "stream",
                    "fail",
                    "STREAM_FRAGMENT_MISSING",
                    "A streaming span has no captured canonical chunk events",
                    {"span_id": span_id},
                )
            )
        references = span.get("payload_references") or []
        if span.get("oversized") and not any(
            isinstance(ref, Mapping)
            and isinstance(ref.get("digest"), str)
            and _DIGEST.fullmatch(ref["digest"])
            and ref.get("size", 0) > 0
            and ref.get("mime_type")
            for ref in references
        ):
            checks.append(
                _check(
                    "payload",
                    "fail",
                    "PAYLOAD_ATTACHMENT_REQUIRED",
                    "Oversized content requires a valid payload reference",
                    {"span_id": span_id},
                )
            )
    requested = {
        call["id"]
        for span in spans
        for call in (span.get("tool_calls") or [])
        if isinstance(call, Mapping) and isinstance(call.get("id"), str)
    }
    # Schema-v2 choices preserve assistant-requested calls inside each choice.
    for span in spans:
        for choice in span.get("choices") or []:
            if not isinstance(choice, Mapping):
                continue
            message = choice.get("message")
            if not isinstance(message, Mapping):
                continue
            for call in message.get("tool_calls") or []:
                if isinstance(call, Mapping) and isinstance(call.get("id"), str):
                    requested.add(call["id"])
    executed = {
        span["tool_call"]["id"]
        for span in spans
        if isinstance(span.get("tool_call"), Mapping)
        and isinstance(span["tool_call"].get("id"), str)
    }
    for call_id in sorted(requested - executed):
        checks.append(
            _check(
                "tools",
                "fail",
                "TOOL_EXECUTION_MISSING",
                "Assistant-requested tool call has no execution span",
                {"call_id": call_id},
            )
        )
    for call_id in sorted(executed - requested):
        checks.append(
            _check(
                "tools",
                "fail",
                "TOOL_CALL_MISSING",
                "Tool execution has no preserved assistant request",
                {"call_id": call_id},
            )
        )
    decisions = {span.get("sampled") for span in spans if isinstance(span.get("sampled"), bool)}
    if len(decisions) > 1:
        checks.append(
            _check(
                "sampling",
                "fail",
                "SAMPLING_INCONSISTENT",
                "All spans must share the root sampling decision",
            )
        )
    if expected_probe_fixture:
        fixture_failure = _probe_fixture_check(spans)
        if fixture_failure:
            checks.append(fixture_failure)
    if not private_provider:
        checks.append(
            _check(
                "ownership",
                "fail",
                "PROVIDER_OWNERSHIP_AMBIGUOUS",
                "Doctor could not prove private provider ownership",
            )
        )
    if flush_outcome == "timeout":
        checks.append(
            _check(
                "flush",
                "fail",
                "FLUSH_TIMEOUT",
                "Diagnostic capture did not flush within the deadline",
            )
        )
    if not checks:
        checks.append(
            _check(
                "local_envelope",
                "pass",
                "LOCAL_ENVELOPE_VALID",
                "The final normalized local envelope is valid",
            )
        )
    first = next((item for item in checks if item["status"] == "fail"), None)
    digest = None
    try:
        digest = doctor_semantic_digest(envelope)
    except (TypeError, ValueError, KeyError):
        pass
    capture = None
    if (
        digest
        and isinstance(trace_id, str)
        and _TRACE_ID.fullmatch(trace_id)
        and isinstance(root_id, str)
        and _SPAN_ID.fullmatch(root_id)
    ):
        capture = {
            "trace_id": trace_id,
            "root_span_id": root_id,
            "span_count": len(spans),
            "semantic_digest": digest,
        }
    result = {
        "format_version": DOCTOR_V2_FORMAT_VERSION,
        "mode": "local",
        "status": (
            "fail"
            if first
            else "warn" if any(item["status"] == "warn" for item in checks) else "pass"
        ),
        "first_failure": first["reason_code"] if first else None,
        "runtime": {
            "language": "python",
            "sdk_version": __version__,
            "schema_version": str(TELEMETRY_SCHEMA_VERSION),
            "transport": "otlp_http_protobuf",
        },
        "sampling": {
            "effective_sampler": "parentbased_traceidratio",
            "root_sample_rate": sample_rate,
            "sampled": next(iter(decisions), True),
        },
        "ownership": {
            "provider": "private" if private_provider else "ambiguous",
            "instrumentor_count": 0,
        },
        "queue": {
            "mode": "diagnostic_capture",
            "pending_spans": 0,
            "dropped_spans": max(0, dropped_spans),
            "capacity": queue_capacity,
        },
        "retry": {"attempts": 0, "window_ms": 0, "exhausted": False},
        "flush": {
            "outcome": flush_outcome,
            "timeout_ms": flush_timeout_ms,
            "duration_ms": flush_duration_ms,
        },
        "checks": checks,
    }
    if capture:
        result["capture"] = capture
    return result


def doctor_captured_local_v2(trace_id: str | None = None, **options: Any) -> dict[str, Any] | None:
    envelope = get_captured_envelope(trace_id)
    return doctor_local_v2(envelope, **options) if envelope else None


def _safe_endpoint(value: str) -> str:
    parsed = urlparse(value)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.username
        or parsed.password
    ):
        raise ValueError("invalid endpoint")
    return f"{parsed.scheme}://{parsed.netloc}"


def _controlled_probe_capture(
    *, api_key: str, endpoint: str, exporter: Any = None
) -> dict[str, Any]:
    """Export one deterministic trace through the normal SDK pipeline."""

    from .core.context import trace
    from .core.logger import get_logger
    from .decorators import span
    from .init import flush, init, shutdown

    clear_doctor_capture()
    sdk_logger = get_logger()
    logging_was_disabled = sdk_logger.disabled
    sdk_logger.disabled = True
    try:
        init(
            api_key=api_key,
            endpoint=endpoint,
            workflow_name="neatlogs.doctor.v2",
            batch_size=32,
            flush_interval=60,
            register_shutdown_handlers=False,
            _doctor_probe=True,
            _doctor_probe_exporter=exporter,
        )
    except Exception:
        sdk_logger.disabled = logging_was_disabled
        raise

    @span(kind="AGENT", name="doctor.probe.agent", role="diagnostic-agent")
    def diagnostic_agent(prompt: str) -> dict[str, str]:
        _mark_doctor_span("AGENT")
        with trace(name="doctor.probe.llm", kind="LLM") as llm:
            # trace() marks manual helper spans internal by default. This is an
            # intentional user-facing Doctor semantic span, so keep it in the
            # normal materialized trace.
            llm.set_attribute("neatlogs.internal", False)
            _mark_doctor_span("LLM")
            llm.set_attribute(
                "input.value",
                json.dumps({"messages": [{"role": "user", "content": prompt}]}),
            )
            llm.set_attribute("neatlogs.llm.token_count.prompt", 11)
            llm.set_attribute("neatlogs.llm.token_count.completion", 7)
            llm.set_attribute("neatlogs.llm.token_count.total", 18)
            output = {"text": "generated diagnostic output"}
            llm.set_attribute("output.value", json.dumps(output))
            return output

    @span(kind="TOOL", name="doctor.probe.tool")
    def diagnostic_tool(value: int) -> dict[str, int]:
        _mark_doctor_span("TOOL")
        return {"value": value + 1}

    started = time.monotonic()
    trace_id = ""
    try:
        with trace(
            name="doctor.probe.root",
            kind="WORKFLOW",
            session_id="neatlogs-doctor-probe",
        ) as root:
            _mark_doctor_span("WORKFLOW")
            trace_id = f"{root.get_span_context().trace_id:032x}"
            root.set_attribute("input.value", json.dumps({"prompt": "generated diagnostic input"}))
            diagnostic_agent("generated diagnostic input")
            tool_output = diagnostic_tool(1)
            root.set_attribute("output.value", json.dumps({"result": tool_output}))
        flushed = flush(timeout_millis=5_000)
        local = doctor_captured_local_v2(
            trace_id,
            flush_outcome="success" if flushed else "timeout",
            flush_duration_ms=max(0, round((time.monotonic() - started) * 1_000)),
            expected_probe_fixture=True,
        )
        if local is None:
            raise RuntimeError("Doctor export capture was unavailable")
        return local
    finally:
        try:
            shutdown(timeout_millis=5_000, termination_reason="doctor-probe-complete")
        finally:
            clear_doctor_capture()
            sdk_logger.disabled = logging_was_disabled


def _record(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _matches_materialized_value(value: Any, candidates: Sequence[Any]) -> bool:
    """Match one of the exact, versioned v3 materialization representations."""
    actual = json.dumps(_canonical(_json_value(value)), separators=(",", ":"), sort_keys=True)
    return any(
        actual == json.dumps(_canonical(candidate), separators=(",", ":"), sort_keys=True)
        for candidate in candidates
    )


def _persisted_probe_result(local: dict[str, Any], trace_data: Mapping[str, Any]) -> dict[str, Any]:
    spans = [dict(item) for item in trace_data.get("spans", []) if isinstance(item, Mapping)]
    ids = [item.get("span_id") if isinstance(item.get("span_id"), str) else "" for item in spans]
    id_set = set(ids)
    duplicate_span_count = len(ids) - len(id_set)
    expected_types = {
        "doctor.probe.root": "workflow",
        "doctor.probe.agent": "agent_action",
        "doctor.probe.llm": "llm",
        "doctor.probe.tool": "tool_call",
    }
    normalized = [
        {
            "id": item.get("span_id"),
            "parent_id": item.get("parent_span_id"),
            "name": str(item.get("node_name") or item.get("span_name") or ""),
            "type": str(item.get("node_type") or item.get("span_type") or "").lower(),
            "data": _record(item.get("data")),
            "metadata": _record(item.get("span_metadata")),
        }
        for item in spans
    ]
    by_name = {item["name"]: item for item in normalized}
    meaningful_root_count = sum(
        item["parent_id"] in (None, "") and item["name"] != "neatlogs.trace.complete"
        for item in normalized
    )
    exact_set = len(normalized) == 4 and set(by_name) == set(expected_types)
    hierarchy_valid = bool(
        exact_set
        and len(id_set) == 4
        and duplicate_span_count == 0
        and meaningful_root_count == 1
        and all(_SPAN_ID.fullmatch(item) for item in ids)
        and by_name["doctor.probe.root"]["parent_id"] in (None, "")
        and by_name["doctor.probe.agent"]["parent_id"] == by_name["doctor.probe.root"]["id"]
        and by_name["doctor.probe.llm"]["parent_id"] == by_name["doctor.probe.agent"]["id"]
        and by_name["doctor.probe.tool"]["parent_id"] == by_name["doctor.probe.root"]["id"]
    )
    attributes_valid = bool(
        exact_set and all(by_name[name]["type"] == kind for name, kind in expected_types.items())
    )
    # The v3 read path intentionally returns the UI-facing simplified view.
    # It may preserve the normalized JSON value or render the same deterministic
    # semantic value for display. Keep this allowlist identical across SDKs.
    expected_io = {
        "doctor.probe.root": (
            ({"prompt": "generated diagnostic input"}, "generated diagnostic input"),
            ({"result": {"value": 2}}, "Value: 2"),
        ),
        "doctor.probe.agent": (
            ({"prompt": "generated diagnostic input"}, "Prompt: generated diagnostic input"),
            ({"text": "generated diagnostic output"}, "Text: generated diagnostic output"),
        ),
        "doctor.probe.llm": (
            (
                {"messages": [{"role": "user", "content": "generated diagnostic input"}]},
                {"prompt": "generated diagnostic input"},
            ),
            ({"text": "generated diagnostic output"}, "Text: generated diagnostic output"),
        ),
        "doctor.probe.tool": (
            ({"value": 1}, "Value: 1"),
            ({"value": 2}, "Value: 2"),
        ),
    }
    input_output_valid = bool(
        exact_set
        and all(
            _matches_materialized_value(by_name[name]["data"].get("input_value"), expected_inputs)
            and _matches_materialized_value(
                by_name[name]["data"].get("output_value"), expected_outputs
            )
            for name, (expected_inputs, expected_outputs) in expected_io.items()
        )
    )
    metadata_valid = bool(
        exact_set
        and all(
            by_name[name]["metadata"].get("neatlogs.doctor") is True
            and by_name[name]["metadata"].get("neatlogs.doctor.version") == "v1"
            and by_name[name]["metadata"].get("service.name") == "neatlogs.doctor.v2"
            and by_name[name]["metadata"].get("telemetry.sdk.language") == "python"
            and by_name[name]["metadata"].get("telemetry.sdk.version") == __version__
            and _canonical_span_kind(by_name[name]["metadata"].get("neatlogs.span.kind"))
            == expected_types[name]
            .replace("agent_action", "agent")
            .replace("tool_call", "tool")
            .upper()
            for name in expected_types
        )
    )
    token_values = [
        trace_data.get("promptTokens"),
        trace_data.get("completionTokens"),
        trace_data.get("totalTokensUsed"),
    ]
    typed_tokens_valid = all(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value == expected
        for value, expected in zip(token_values, (11, 7, 18), strict=True)
    )
    readback_span_count = (
        trace_data.get("spanCount") if isinstance(trace_data.get("spanCount"), int) else len(spans)
    )
    readback_trace_id = trace_data.get("_id") if isinstance(trace_data.get("_id"), str) else ""
    visible = readback_trace_id == local.get("capture", {}).get("trace_id")
    # A terminal failure is materialized, but it is not a successful Doctor
    # probe. Only the product API's documented successful terminal value passes.
    finalized = str(trace_data.get("status") or "").lower() == "success"
    validations = (
        (
            "probe_visibility",
            visible and readback_span_count == 4 and len(spans) == 4,
            "TRACE_VISIBLE",
            "WAIT_FOR_TRACE",
            "The exact Doctor trace is visible through the authenticated trace API",
        ),
        (
            "probe_finalization",
            finalized,
            "TRACE_FINALIZED",
            "WAIT_FOR_TRACE",
            "The exact Doctor trace reached a terminal materialized state",
        ),
        (
            "probe_hierarchy",
            hierarchy_valid,
            "HIERARCHY_VALID",
            "CHECK_TRACE_FINALIZER",
            "The persisted Doctor hierarchy has one root and valid parents",
        ),
        (
            "probe_attributes",
            attributes_valid,
            "ATTRIBUTES_VALID",
            "CHECK_ATTRIBUTE_MAPPING",
            "The persisted Doctor span names and types are complete",
        ),
        (
            "probe_input_output",
            input_output_valid,
            "INPUT_OUTPUT_VALID",
            "CHECK_PAYLOAD_MAPPING",
            "The persisted Doctor spans retain input and output",
        ),
        (
            "probe_metadata",
            metadata_valid,
            "METADATA_VALID",
            "CHECK_METADATA_FINALIZATION",
            "The versioned Doctor SDK metadata survived finalization",
        ),
        (
            "probe_typed_tokens",
            typed_tokens_valid,
            "TYPED_TOKENS_VALID",
            "CHECK_TOKEN_MAPPING",
            "Persisted token totals remain numeric",
        ),
    )
    probe_checks = [
        {
            "name": name,
            "status": "pass" if passed else "fail",
            "reason_code": pass_code if passed else f"{pass_code}_FAILED",
            "remediation_code": "NONE" if passed else remediation,
            "message": message,
        }
        for name, passed, pass_code, remediation, message in validations
    ]
    first = next((item for item in probe_checks if item["status"] == "fail"), None)
    return {
        **local,
        "mode": "probe",
        "status": "fail" if first or local["status"] != "pass" else "pass",
        "first_failure": first["reason_code"] if first else local.get("first_failure"),
        "probe": {
            "ingest_route": "/v1/traces",
            "marker_header": "x-neatlogs-doctor",
            "marker_version": "v1",
            "visible": visible,
            "readback_trace_id": readback_trace_id,
            "finalized": finalized,
            "meaningful_root_count": meaningful_root_count,
            "duplicate_span_count": duplicate_span_count,
            "readback_span_count": readback_span_count,
            "hierarchy_valid": hierarchy_valid,
            "attributes_valid": attributes_valid,
            "input_output_valid": input_output_valid,
            "metadata_valid": metadata_valid,
            "typed_tokens_valid": typed_tokens_valid,
        },
        "checks": [*local.get("checks", []), *probe_checks],
    }


def doctor_probe_v2(
    *,
    api_key: str | None = None,
    endpoint: str | None = None,
    timeout_seconds: float = 45.0,
    _exporter: Any = None,
) -> dict[str, Any]:
    """Export a controlled trace and read the exact persisted trace back."""
    key = (api_key if api_key is not None else os.getenv("NEATLOGS_API_KEY", "")).strip()
    if not key:
        return {
            "format_version": DOCTOR_V2_FORMAT_VERSION,
            "mode": "probe",
            "status": "fail",
            "first_failure": "CREDENTIAL_MISSING",
            "runtime": {
                "language": "python",
                "sdk_version": __version__,
                "schema_version": str(TELEMETRY_SCHEMA_VERSION),
                "transport": "otlp_http_protobuf",
            },
            "checks": [
                _check(
                    "credentials",
                    "fail",
                    "CREDENTIAL_MISSING",
                    "Configure an ingestion credential to run a backend probe",
                )
            ],
        }
    try:
        url = _safe_endpoint(
            (endpoint or os.getenv("NEATLOGS_ENDPOINT") or "https://ingest.neatlogs.com").strip()
        )
    except ValueError:
        return {
            "format_version": DOCTOR_V2_FORMAT_VERSION,
            "mode": "probe",
            "status": "fail",
            "first_failure": "ENDPOINT_INVALID",
            "runtime": {
                "language": "python",
                "sdk_version": __version__,
                "schema_version": str(TELEMETRY_SCHEMA_VERSION),
                "transport": "otlp_http_protobuf",
            },
            "checks": [
                _check(
                    "endpoint",
                    "fail",
                    "ENDPOINT_INVALID",
                    "Configure an absolute HTTP or HTTPS diagnostic endpoint",
                ),
            ],
        }
    capture: dict[str, Any] | None = None
    try:
        local = _controlled_probe_capture(
            api_key=key,
            endpoint=url,
            exporter=_exporter,
        )
        capture = local["capture"]
        readback_url = f"{url}/api/traces/v3/{quote(capture['trace_id'], safe='')}"
        deadline = time.monotonic() + timeout_seconds
        trace_data: dict[str, Any] | None = None
        while time.monotonic() < deadline:
            response = requests.get(
                readback_url,
                headers={"x-api-key": key},
                timeout=min(5.0, max(0.1, deadline - time.monotonic())),
            )
            if response.status_code in {401, 403}:
                response.raise_for_status()
            if response.ok and response.status_code != 202:
                value = response.json()
                if not isinstance(value, dict):
                    raise ValueError("invalid trace read-back")
                trace_data = value
                break
            if response.status_code not in {202, 404}:
                response.raise_for_status()
            time.sleep(min(1.0, max(0, deadline - time.monotonic())))
        if trace_data is None:
            raise requests.Timeout("timed out waiting for exact Doctor trace")
        return _persisted_probe_result(local, trace_data)
    except (
        requests.RequestException,
        RuntimeError,
        ValueError,
        TypeError,
        json.JSONDecodeError,
    ) as exc:
        response = getattr(exc, "response", None)
        response_status = getattr(response, "status_code", None)
        reason_code = (
            "AUTH_FAILED" if response_status in {401, 403} else "BACKEND_PROBE_UNAVAILABLE"
        )
        failure = {
            "format_version": DOCTOR_V2_FORMAT_VERSION,
            "mode": "probe",
            "status": "fail",
            "first_failure": reason_code,
            "runtime": {
                "language": "python",
                "sdk_version": __version__,
                "schema_version": str(TELEMETRY_SCHEMA_VERSION),
                "transport": "otlp_http_protobuf",
            },
            "checks": [
                _check(
                    "probe",
                    "fail",
                    reason_code,
                    (
                        "The project key was rejected by the existing trace API"
                        if reason_code == "AUTH_FAILED"
                        else "The existing trace ingestion or read path is unavailable"
                    ),
                )
            ],
        }
        if capture is not None:
            failure["capture"] = capture
        return failure
