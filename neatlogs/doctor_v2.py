"""Doctor v2 envelope capture, validation, and backend receipt probing."""

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
from urllib.parse import urljoin, urlparse

import requests
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.trace import StatusCode, TraceFlags

from .schema_v2 import TELEMETRY_SCHEMA_VERSION
from .version import __version__

DOCTOR_V2_FORMAT_VERSION = "neatlogs.doctor/v2"
_TRACE_ID = re.compile(r"^[0-9a-f]{32}$")
_SPAN_ID = re.compile(r"^[0-9a-f]{16}$")
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_MAX_CAPTURED_TRACES = 16
_capture_lock = threading.RLock()
_captures: OrderedDict[str, dict[str, dict[str, Any]]] = OrderedDict()
_latest_trace_id: str | None = None

_REMEDIATION = {
    "CREDENTIAL_MISSING": "SET_CREDENTIAL",
    "AUTH_FAILED": "CHECK_INGEST_CREDENTIAL",
    "BACKEND_PROBE_UNAVAILABLE": "CHECK_DIAGNOSTIC_ENDPOINT",
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
    "STAGE_PENDING": "WAIT_FOR_RECEIPT",
    "DIAGNOSTIC_EXPIRED": "CREATE_NEW_SESSION",
    "DIAGNOSTIC_NOT_VISIBLE": "CONTACT_SUPPORT",
    "DIGEST_MISMATCH": "CONTACT_SUPPORT",
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


def _tool_calls(attributes: Mapping[str, Any]) -> list[dict[str, str]] | None:
    value = _first(attributes, ("gen_ai.assistant.tool_calls", "neatlogs.llm.tool_calls"))
    if isinstance(value, list):
        calls = [
            {key: item[key] for key in ("id", "name") if isinstance(item.get(key), str)}
            for item in value
            if isinstance(item, dict) and isinstance(item.get("id"), str)
        ]
        return calls or None
    exploded: dict[int, dict[str, str]] = {}
    for key, item in attributes.items():
        match = re.match(r"^(?:neatlogs\.)?llm\.tool_calls\.(\d+)\.(id|name)$", key)
        if match and isinstance(item, str):
            exploded.setdefault(int(match.group(1)), {})[match.group(2)] = item
    calls = [exploded[index] for index in sorted(exploded) if "id" in exploded[index]]
    return calls or None


def _diagnostic_span(span: ReadableSpan) -> dict[str, Any]:
    context = span.get_span_context()
    attributes = dict(span.attributes or {})
    kind_value = attributes.get("neatlogs.span.kind", attributes.get("openinference.span.kind"))
    kind = str(kind_value or "INTERNAL").removeprefix("Neatlogs.").upper()
    parent = span.parent
    output: dict[str, Any] = {
        "span_id": f"{context.span_id:016x}",
        "parent_span_id": f"{parent.span_id:016x}" if parent and parent.is_valid else None,
        "name": span.name,
        "kind": kind,
        "status": (
            "ERROR"
            if span.status.status_code is StatusCode.ERROR
            else "OK" if span.status.status_code is StatusCode.OK else "UNSET"
        ),
        "sampled": bool(context.trace_flags & TraceFlags.SAMPLED),
        "ended": span.end_time is not None,
        "start_time_ns": span.start_time,
        "duration_ns": max(0, (span.end_time or span.start_time) - span.start_time),
        "attributes": attributes,
    }
    fields = {
        "input": _first(
            attributes,
            ("gen_ai.input.messages", "input.value", "neatlogs.input", "neatlogs.llm.input"),
        ),
        "output": _first(
            attributes,
            ("gen_ai.output.messages", "output.value", "neatlogs.output", "neatlogs.llm.output"),
        ),
        "choices": _first(attributes, ("gen_ai.response.choices", "neatlogs.llm.choices")),
        "stream_fragments": _first(
            attributes, ("gen_ai.stream.fragments", "neatlogs.stream.fragments")
        ),
    }
    for key, value in fields.items():
        if value is not None:
            output[key] = value
    calls = _tool_calls(attributes)
    if calls:
        output["tool_calls"] = calls
    call_id = _first(attributes, ("neatlogs.tool.call_id", "neatlogs.tool_call.id", "tool_call_id"))
    if kind == "TOOL" and isinstance(call_id, str):
        output["tool_call"] = {"id": call_id}
        if isinstance(attributes.get("neatlogs.tool.name"), str):
            output["tool_call"]["name"] = attributes["neatlogs.tool.name"]
    streaming = bool(
        attributes.get("neatlogs.llm.is_streaming")
        or attributes.get("neatlogs.tool.is_streaming")
        or "stream_fragments" in output
    )
    if streaming:
        output["streaming"] = True
    return output


def capture_prepared_spans(spans: Sequence[ReadableSpan]) -> None:
    """Capture the final masked spans immediately before the network exporter."""

    global _latest_trace_id
    with _capture_lock:
        for span in spans:
            trace_id = f"{span.get_span_context().trace_id:032x}"
            current = _captures.pop(trace_id, {})
            item = _diagnostic_span(span)
            current[item["span_id"]] = item
            _captures[trace_id] = current
            _latest_trace_id = trace_id
        while len(_captures) > _MAX_CAPTURED_TRACES:
            _captures.popitem(last=False)


def clear_doctor_capture() -> None:
    global _latest_trace_id
    with _capture_lock:
        _captures.clear()
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
    projection = {
        "trace_id": envelope["trace_id"],
        "root_span_id": envelope["root_span_id"],
        "spans": sorted(envelope["spans"], key=lambda span: span["span_id"].lower()),
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
                    "Streaming span has no captured fragments",
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
    return f"{parsed.scheme}://{parsed.netloc}/api/diagnostics/v2/sessions"


def doctor_probe_v2(
    *, api_key: str | None = None, endpoint: str | None = None, timeout_seconds: float = 10.0
) -> dict[str, Any]:
    """Create and poll a scoped backend diagnostic session without exposing its token."""

    import secrets

    synthetic = {
        "trace_id": secrets.token_hex(16),
        "root_span_id": secrets.token_hex(8),
        "spans": [],
    }
    synthetic["spans"].append(
        {
            "span_id": synthetic["root_span_id"],
            "parent_span_id": None,
            "name": "doctor.workflow",
            "kind": "WORKFLOW",
            "status": "OK",
            "input": {"prompt": "generated diagnostic input"},
            "output": {"result": "generated diagnostic output"},
            "sampled": True,
            "ended": True,
        }
    )
    local = doctor_local_v2(synthetic)
    capture = local["capture"]
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
            "capture": capture,
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
            **local,
            "mode": "probe",
            "status": "fail",
            "first_failure": "ENDPOINT_INVALID",
            "checks": [
                *local["checks"],
                _check(
                    "endpoint",
                    "fail",
                    "ENDPOINT_INVALID",
                    "Configure an absolute HTTP or HTTPS diagnostic endpoint",
                ),
            ],
        }
    headers = {"x-api-key": key, "content-type": "application/json"}
    session_id = None
    try:
        created_response = requests.post(
            url,
            headers=headers,
            json={
                "envelope_digest": capture["semantic_digest"],
                "fixture_version": "doctor-v2",
                "trace_id": capture["trace_id"],
            },
            timeout=min(5.0, timeout_seconds),
        )
        created_response.raise_for_status()
        created = created_response.json()
        session_id, token = created.get("diagnostic_id"), created.get("probe_token")
        if not isinstance(session_id, str) or not isinstance(token, str):
            raise ValueError("invalid diagnostic session")
        receipt_url = urljoin(url.rstrip("/") + "/", session_id)
        deadline = time.monotonic() + timeout_seconds
        receipt: dict[str, Any] = {
            "status": "pending",
            "stages": [],
            "expires_at": created.get("expires_at"),
        }
        while time.monotonic() < deadline:
            response = requests.get(
                receipt_url,
                headers={"x-api-key": key, "x-neatlogs-diagnostic-token": token},
                timeout=min(5.0, max(0.1, deadline - time.monotonic())),
            )
            if response.ok:
                receipt = response.json()
            if receipt.get("status") in {"pass", "fail", "expired"}:
                break
            time.sleep(min(0.25, max(0, deadline - time.monotonic())))
        stages = [item for item in receipt.get("stages", []) if isinstance(item, dict)]
        required = (
            "auth",
            "schema_decode",
            "pii",
            "kafka",
            "raw_durable",
            "root_resolution",
            "simplified_durable",
            "visibility",
        )
        complete = receipt.get("status") == "pass" and all(
            any(item.get("stage") == stage and item.get("status") == "accepted" for item in stages)
            for stage in required
        )
        failed = next((item for item in stages if item.get("status") == "failed"), None)
        local_digest = receipt.get("local_semantic_digest")
        backend_digest = receipt.get("backend_semantic_digest")
        digest_mismatch = any(
            isinstance(value, str) and value != capture["semantic_digest"]
            for value in (local_digest, backend_digest)
        ) or (
            isinstance(local_digest, str)
            and isinstance(backend_digest, str)
            and local_digest != backend_digest
        )
        reason = receipt.get("first_failure") or (failed or {}).get("reason_code")
        if reason is None and digest_mismatch:
            reason = "DIGEST_MISMATCH"
        if reason is None and not complete:
            reason = (
                "DIAGNOSTIC_EXPIRED"
                if receipt.get("status") == "expired"
                else (
                    "STAGE_PENDING"
                    if receipt.get("status") == "pending"
                    else "DIAGNOSTIC_NOT_VISIBLE"
                )
            )
        if digest_mismatch:
            complete = False
        check = _check(
            "probe_visibility",
            "pass" if complete else "fail",
            "DIAGNOSTIC_VISIBLE" if complete else reason,
            (
                "The diagnostic trace reached the authenticated read path"
                if complete
                else "The backend diagnostic did not reach every required stage"
            ),
        )
        return {
            "format_version": DOCTOR_V2_FORMAT_VERSION,
            "mode": "probe",
            "status": "pass" if complete else "fail",
            "first_failure": reason,
            "runtime": {
                "language": "python",
                "sdk_version": __version__,
                "schema_version": str(TELEMETRY_SCHEMA_VERSION),
                "transport": "otlp_http_protobuf",
            },
            "capture": capture,
            "probe": {
                "diagnostic_id": session_id,
                "receipt_status": receipt.get("status", "pending"),
                "expires_at": receipt.get("expires_at") or created.get("expires_at"),
                "stages": stages,
            },
            "checks": [check],
        }
    except (requests.RequestException, ValueError, TypeError, json.JSONDecodeError) as exc:
        response = getattr(exc, "response", None)
        response_status = getattr(response, "status_code", None)
        reason_code = (
            "AUTH_FAILED" if response_status in {401, 403} else "BACKEND_PROBE_UNAVAILABLE"
        )
        return {
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
            "capture": capture,
            "checks": [
                _check(
                    "probe",
                    "fail",
                    reason_code,
                    (
                        "The authenticated diagnostic session was rejected"
                        if reason_code == "AUTH_FAILED"
                        else "The backend diagnostic session is unavailable"
                    ),
                )
            ],
        }
    finally:
        if session_id:
            try:
                requests.delete(
                    urljoin(url.rstrip("/") + "/", session_id),
                    headers={"x-api-key": key},
                    timeout=2.0,
                )
            except requests.RequestException:
                pass
