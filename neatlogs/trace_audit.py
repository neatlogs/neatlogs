"""Real-trace Judge & Replay Readiness auditor. Standard library only."""

from __future__ import annotations

import json
import math
import os
import re
import sys
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Callable, Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlparse

from .trace_cli import (
    EXIT_AUTH,
    EXIT_CHECKS_FAILED,
    EXIT_ERROR,
    EXIT_NOT_READY,
    EXIT_OK,
    EXIT_USAGE,
    Fetcher,
    _NoRedirect,
    fetch_public_trace,
    resolve_public_api_host,
)

_FRESHNESS_KEYS = ("updated_at", "created_at", "timestamp", "last_modified", "date")
_TRUNCATION_KEYS = ("neatlogs.payload.truncated", "neatlogs.capture.truncated")
_UNAVAILABLE_STATES = frozenset(
    {"unavailable", "retentionExpired", "truncated", "previewOnly", "redacted", "masked", "lossy"}
)
_INCOMPLETE_ATTR_FLAGS = (
    "pii.server_redacted",
    "pii.server_redaction_failed",
    "neatlogs.payload.truncated",
    "neatlogs.capture.truncated",
    "neatlogs.stream.incomplete",
)
MAX_V3_TRACE_BYTES = 16 * 1024 * 1024
_KIND_ALIASES = {
    "TOOL_CALL": "TOOL",
    "MCP": "MCP_TOOL",
    "RE_RANKER": "RERANKER",
    "VECTOR": "VECTOR_STORE",
    "AGENT_ACTION": "AGENT",
}
_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
_ISO_TS_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2})(?:\.(\d+))?(Z|[+-]\d{2}:?\d{2})?$",
    re.ASCII,
)
_FULL_CONTAINER_EVIDENCE_KEYS = ("attributes", "span_metadata", "data", "semantic")
_FULL_VALUE_EVIDENCE_KEYS = (
    "input",
    "output",
    "input_value",
    "output_value",
    "inputValue",
    "outputValue",
    "tool_input",
    "tool_output",
    "payload_state",
    "input_content_state",
    "output_content_state",
    "redacted",
    "masked",
    "input_truncated",
    "output_truncated",
)


def _span_has_full_capture_evidence(span: dict[str, Any]) -> bool:
    for key in _FULL_CONTAINER_EVIDENCE_KEYS:
        val = span.get(key)
        if isinstance(val, dict) and bool(val):
            return True
        if isinstance(val, str):
            stripped = val.strip()
            if stripped and stripped != "{}":
                return True
    return any(span.get(key) is not None for key in _FULL_VALUE_EVIDENCE_KEYS)


def _is_metadata_only_spans(spans: Sequence[dict[str, Any]]) -> bool:
    """Return True when spans carry only summary/metadata fields and no full capture evidence."""
    if not spans:
        return False
    return not any(_span_has_full_capture_evidence(span) for span in spans)


class _ResponseTooLargeError(ValueError):
    """Raised when a trace read response exceeds the bounded size cap."""


def _v3_default_fetch(url: str, headers: dict, timeout: float) -> tuple[int, bytes]:
    request = urllib.request.Request(url, headers=headers, method="GET")
    opener = urllib.request.build_opener(_NoRedirect)
    try:
        with opener.open(request, timeout=timeout) as response:
            body = response.read(MAX_V3_TRACE_BYTES + 1)
            if len(body) > MAX_V3_TRACE_BYTES:
                raise _ResponseTooLargeError(
                    f"Trace response exceeded {MAX_V3_TRACE_BYTES // (1024 * 1024)} MiB limit"
                )
            return response.status, body
    except urllib.error.HTTPError as exc:
        try:
            body = exc.read(64 * 1024)
        except Exception:
            body = b""
        return exc.code, body


def _normalize_id(value: str | None) -> str:
    return (value or "").strip().replace("-", "").lower()


def _parse_ts_ns(span: dict[str, Any]) -> int | None:
    raw = (
        span.get("start_time_unix_nano")
        or span.get("start_time")
        or span.get("startTime")
        or span.get("startedAt")
        or span.get("createdAt")
    )
    if raw is None or isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        return raw
    if isinstance(raw, float):
        return int(raw) if math.isfinite(raw) else None
    if isinstance(raw, str):
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


def _step_seq(span: dict[str, Any]) -> int | None:
    raw = _attrs(span).get("neatlogs.step.seq")
    if isinstance(raw, int) and not isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        text = raw.strip()
        if text.isascii() and text.isdigit():
            try:
                return int(text)
            except ValueError:
                return None
    return None


def _text(value: Any) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _attrs(span: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    meta = span.get("span_metadata")
    if isinstance(meta, dict):
        out.update(meta)
    raw = span.get("attributes")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            raw = None
    if isinstance(raw, dict):
        out.update(raw)
    return out


def _span_id(span: dict[str, Any]) -> str | None:
    return _text(span.get("spanId")) or _text(span.get("span_id"))


def _parent_id(span: dict[str, Any]) -> str | None:
    return _text(span.get("parentSpanId")) or _text(span.get("parent_span_id"))


def _span_name(span: dict[str, Any]) -> str | None:
    return (
        _text(span.get("spanName"))
        or _text(span.get("span_name"))
        or _text(span.get("node_name"))
        or _text(span.get("name"))
    )


def _tool_name(span: dict[str, Any]) -> str | None:
    attrs = _attrs(span)
    return (
        _text(span.get("tool_name"))
        or _text(attrs.get("tool.name"))
        or _text(attrs.get("neatlogs.tool.name"))
        or _text(attrs.get("neatlogs.raw.tool.name"))
        or _span_name(span)
    )


def _is_lossy_or_incomplete(span: dict[str, Any]) -> bool:
    attrs = _attrs(span)
    states = (
        span.get("payload_state"),
        span.get("input_content_state"),
        span.get("output_content_state"),
        attrs.get("neatlogs.capture_fidelity"),
    )
    if any(state in _UNAVAILABLE_STATES for state in states if isinstance(state, str)):
        return True
    if attrs.get("pii.server_redaction_status") == "redacted":
        return True
    span_flags = (
        span.get("redacted"),
        span.get("masked"),
        span.get("input_truncated"),
        span.get("output_truncated"),
    )
    attr_flags = tuple(attrs.get(k) for k in _INCOMPLETE_ATTR_FLAGS)
    return any(flag is True or flag == "true" for flag in (*span_flags, *attr_flags))


def _tool_is_replay_acceptable(span: dict[str, Any], span_id_counts: Counter[str]) -> bool:
    from .replay.frozen_case import validate_tool_span_for_replay

    return validate_tool_span_for_replay(span, span_id_counts)


def _semantic(span: dict[str, Any]) -> dict[str, Any]:
    raw = span.get("semantic")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            return {}
    return raw if isinstance(raw, dict) else {}


def _span_kind(span: dict[str, Any]) -> str:
    attrs = _attrs(span)
    semantic = _semantic(span)
    raw = (
        semantic.get("kind")
        or span.get("spanType")
        or span.get("span_type")
        or span.get("node_type")
        or span.get("kind")
        or attrs.get("neatlogs.span.kind")
        or attrs.get("openinference.span.kind")
        or span.get("type")
        or ""
    )
    normalized = str(raw).strip().upper()
    return _KIND_ALIASES.get(normalized, normalized)


def _is_completion_marker(span: dict[str, Any]) -> bool:
    attrs = _attrs(span)
    return (
        _span_name(span) == "neatlogs.trace.complete"
        or attrs.get("neatlogs.trace.complete") is True
    )


def _non_empty(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        stripped = value.strip()
        return bool(stripped) and stripped not in {'""', "null", "None", "{}", "[]"}
    if isinstance(value, (dict, list, tuple)):
        return len(value) > 0
    return True


def _span_input(span: dict[str, Any]) -> Any:
    attrs = _attrs(span)
    data = span.get("data") if isinstance(span.get("data"), dict) else {}
    kind = _span_kind(span).lower()
    for candidate in (
        span.get("input"),
        span.get("inputValue"),
        span.get("input_value"),
        span.get("tool_input"),
        data.get("input_value"),
        data.get("input"),
        data.get("tool_input"),
        data.get("tool_arguments"),
        data.get("user_input"),
        attrs.get(f"neatlogs.{kind}.input") if kind else None,
        attrs.get("neatlogs.tool.input"),
        attrs.get("input.value"),
    ):
        if _non_empty(candidate):
            return candidate
    return None


def _span_output(span: dict[str, Any]) -> Any:
    attrs = _attrs(span)
    data = span.get("data") if isinstance(span.get("data"), dict) else {}
    kind = _span_kind(span).lower()
    for candidate in (
        span.get("output"),
        span.get("outputValue"),
        span.get("output_value"),
        span.get("tool_output"),
        data.get("output_value"),
        data.get("output"),
        data.get("tool_output"),
        data.get("tool_result"),
        data.get("result"),
        attrs.get(f"neatlogs.{kind}.output") if kind else None,
        attrs.get("neatlogs.tool.output"),
        attrs.get("output.value"),
    ):
        if _non_empty(candidate):
            return candidate
    return None


def _extract_docs_with_freshness(span: dict[str, Any]) -> tuple[int, int]:
    """Return (doc_count, docs_with_freshness) from retriever attributes or output JSON."""
    attrs = _attrs(span)
    doc_indices: set[str] = set()
    for key in attrs:
        if key.startswith("neatlogs.retriever.documents."):
            parts = key.split(".")
            if len(parts) >= 4:
                doc_indices.add(parts[3])
    if doc_indices:
        fresh = 0
        for idx in doc_indices:
            prefix = f"neatlogs.retriever.documents.{idx}."
            if any(attrs.get(prefix + k) for k in _FRESHNESS_KEYS):
                fresh += 1
                continue
            meta = attrs.get(prefix + "metadata")
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except ValueError:
                    meta = None
            if isinstance(meta, dict) and any(meta.get(k) for k in _FRESHNESS_KEYS):
                fresh += 1
        return len(doc_indices), fresh

    out = _span_output(span)
    if isinstance(out, str):
        try:
            out = json.loads(out)
        except ValueError:
            return 0, 0
    docs = None
    if isinstance(out, list):
        docs = out
    elif isinstance(out, dict):
        for key in ("documents", "docs", "results", "matches", "items"):
            if isinstance(out.get(key), list):
                docs = out[key]
                break
    if not docs:
        return 0, 0
    fresh = 0
    for doc in docs:
        if not isinstance(doc, dict):
            continue
        meta = doc.get("metadata") if isinstance(doc.get("metadata"), dict) else {}
        if any(doc.get(k) for k in _FRESHNESS_KEYS) or any(meta.get(k) for k in _FRESHNESS_KEYS):
            fresh += 1
    return len(docs), fresh


def readable_spans_to_audit_payload(
    spans: Sequence[Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Convert real OpenTelemetry ReadableSpans into (trace_dict, span_dicts) for auditing."""
    converted: list[dict[str, Any]] = []
    trace_id = ""
    has_marker = False
    session_id = None
    for span in spans:
        ctx = getattr(span, "context", None)
        if ctx is not None and not trace_id:
            trace_id = f"{ctx.trace_id:032x}"
        attrs = dict(getattr(span, "attributes", None) or {})
        parent = getattr(span, "parent", None)
        if getattr(span, "name", "") == "neatlogs.trace.complete" or attrs.get(
            "neatlogs.trace.complete"
        ):
            has_marker = True
        if session_id is None:
            session_id = _text(attrs.get("neatlogs.session.id")) or _text(attrs.get("session.id"))
        converted.append(
            {
                "span_id": f"{ctx.span_id:016x}" if ctx is not None else "",
                "parent_span_id": f"{parent.span_id:016x}" if parent is not None else None,
                "name": getattr(span, "name", ""),
                "span_type": attrs.get("neatlogs.span.kind")
                or attrs.get("openinference.span.kind")
                or "",
                "start_time_unix_nano": getattr(span, "start_time", None),
                "end_time_unix_nano": getattr(span, "end_time", None),
                "attributes": attrs,
                "status": getattr(getattr(span, "status", None), "status_code", None),
            }
        )
    trace: dict[str, Any] = {
        "trace_id": trace_id,
        "finalizationStatus": "finalized" if has_marker else "pending",
    }
    if session_id:
        trace["session_id"] = session_id
    return trace, converted


def audit_trace(
    trace: dict[str, Any],
    spans: list[dict[str, Any]],
    *,
    expect_session: bool = False,
    expect_retriever: bool = False,
    metadata_only: bool = False,
    v3_session_fallback: bool = False,
) -> dict[str, Any]:
    """Evaluate a real trace for Jev judge-readiness and ReplaySession backtrack-readiness."""
    safe_trace = trace if isinstance(trace, dict) else {}
    raw_spans = spans if isinstance(spans, Sequence) and not isinstance(spans, (str, bytes)) else []
    dict_spans = [s for s in raw_spans if isinstance(s, dict)]
    malformed_spans = len(raw_spans) - len(dict_spans)

    trace_id = (
        _text(safe_trace.get("traceId"))
        or _text(safe_trace.get("trace_id"))
        or _text(safe_trace.get("_id"))
        or ""
    )
    has_marker = any(_is_completion_marker(s) for s in dict_spans)
    work_spans = [s for s in dict_spans if not _is_completion_marker(s)]
    metadata_only = metadata_only or _is_metadata_only_spans(work_spans)

    checks: list[dict[str, Any]] = []
    recommendations: list[str] = []
    score = 0

    def record(name: str, status: str, weight: int, message: str, rec: str | None = None) -> None:
        nonlocal score
        checks.append({"name": name, "status": status, "weight": weight, "message": message})
        if status == "pass":
            score += weight
        elif status == "warn":
            score += weight // 2
        if status != "pass" and rec:
            recommendations.append(rec)

    # 1. has_spans (15)
    if malformed_spans > 0:
        record(
            "has_spans",
            "fail",
            15,
            f"{len(work_spans)} operational span(s) captured, but {malformed_spans} malformed non-object span(s) found",
            "Verify the trace payload contains only JSON span objects.",
        )
    elif work_spans:
        record("has_spans", "pass", 15, f"{len(work_spans)} operational span(s) captured")
    else:
        record(
            "has_spans",
            "fail",
            15,
            "0 operational spans captured",
            "Ensure neatlogs.init() runs before LLM imports and flush() before exit.",
        )

    # 2. single_workflow_root (15)
    ids = {_span_id(s) for s in work_spans} - {None}
    orphans = [s for s in work_spans if _parent_id(s) not in (None, *ids)]
    roots = [s for s in work_spans if _parent_id(s) is None]
    if len(roots) == 1 and not orphans:
        record("single_workflow_root", "pass", 15, "Single root span with resolved parent links")
    else:
        record(
            "single_workflow_root",
            "fail",
            15,
            f"Found {len(roots)} root(s) and {len(orphans)} orphan span(s)",
            "Wrap each user-facing entry point in one @neatlogs.span(kind='WORKFLOW') root.",
        )

    # 3. root_io_present (15)
    root = roots[0] if roots else None
    has_in = bool(root and _non_empty(_span_input(root)))
    has_out = bool(root and _non_empty(_span_output(root)))
    if metadata_only and root is not None and not (has_in and has_out):
        record(
            "root_io_present",
            "warn",
            15,
            "Public API spans are metadata-only (no input/output payloads returned)",
            "Set NEATLOGS_API_KEY (for /api/traces/v3) or audit a local --file export to verify root I/O.",
        )
    elif has_in and has_out:
        record("root_io_present", "pass", 15, "Root span has non-empty input and output")
    else:
        missing = [lbl for lbl, ok in (("input", has_in), ("output", has_out)) if not ok]
        record(
            "root_io_present",
            "fail",
            15,
            f"Root span missing: {', '.join(missing)}",
            "Capture clean user_request input and final_reply output on the WORKFLOW root.",
        )

    # 4. trace_finalized (10)
    fin_status = safe_trace.get("finalizationStatus") or safe_trace.get("finalization_status")
    v3_status = str(safe_trace.get("status") or "").lower()
    if has_marker or fin_status == "finalized" or v3_status in ("success", "error", "completed"):
        record("trace_finalized", "pass", 10, "Trace finalized with completion marker")
    else:
        record(
            "trace_finalized",
            "fail",
            10,
            f"Trace not finalized (finalizationStatus={fin_status}, marker={has_marker})",
            "End the root span cleanly and call neatlogs.flush() on script/worker exit.",
        )

    # 5. step_ordering_clean (10)
    seq_entries: list[tuple[int | None, int]] = []
    for s in work_spans:
        seq = _step_seq(s)
        if seq is not None:
            ts = _parse_ts_ns(s)
            seq_entries.append((ts, seq))
    seqs = [seq for _, seq in seq_entries]
    if seqs:
        # Real OTLP exports spans in child-first end order; require unique positive sequences
        # and, among spans with start timestamps, verify step.seq never inverts when start
        # time strictly advances. Without timestamps, unique positive step.seq defines order.
        unique_positive = min(seqs) >= 1 and len(set(seqs)) == len(seqs)
        strictly_inc = unique_positive
        ts_entries = [(ts, seq) for ts, seq in seq_entries if ts is not None]
        if strictly_inc and len(ts_entries) >= 2:
            ordered = sorted(ts_entries, key=lambda pair: (pair[0], pair[1]))
            prev_ts = ordered[0][0]
            prev_max_seq = 0
            cur_max_seq = ordered[0][1]
            for ts, seq in ordered[1:]:
                if ts != prev_ts:
                    prev_ts = ts
                    prev_max_seq = cur_max_seq
                if seq <= prev_max_seq:
                    strictly_inc = False
                    break
                if seq > cur_max_seq:
                    cur_max_seq = seq
        if strictly_inc:
            record(
                "step_ordering_clean", "pass", 10, f"Monotonic step sequence ({len(seqs)} steps)"
            )
        else:
            record(
                "step_ordering_clean",
                "fail",
                10,
                f"Non-monotonic neatlogs.step.seq values: {seqs[:10]}",
                "Check span creation order and upgrade to the latest neatlogs SDK.",
            )
    elif metadata_only:
        record(
            "step_ordering_clean",
            "warn",
            10,
            "Public API spans are metadata-only (no neatlogs.step.seq attributes returned)",
            "Set NEATLOGS_API_KEY (for /api/traces/v3) or audit a local --file export to verify step ordering.",
        )
    else:
        record(
            "step_ordering_clean",
            "warn",
            10,
            "No neatlogs.step.seq attributes found (older SDK or external OTLP source)",
            "Upgrade SDK to stamp monotonic neatlogs.step.seq for deterministic backtracking.",
        )

    # 6. tool_linkage_valid (10)
    tool_spans = [s for s in work_spans if _span_kind(s) in {"TOOL", "MCP_TOOL"}]
    span_id_counts: Counter[str] = Counter(
        sid for sid in (_span_id(s) for s in dict_spans) if sid is not None
    )
    if metadata_only:
        unnamed_tools = [s for s in tool_spans if not _tool_name(s)]
        if unnamed_tools:
            record(
                "tool_linkage_valid",
                "fail",
                10,
                f"{len(unnamed_tools)} tool span(s) missing name",
                "Decorate custom tools with @neatlogs.span(kind='TOOL', tool_name='...', replay=True).",
            )
        elif tool_spans:
            record(
                "tool_linkage_valid",
                "warn",
                10,
                f"{len(tool_spans)} tool span(s) named, but public API spans are metadata-only (I/O not returned)",
                "Set NEATLOGS_API_KEY (for /api/traces/v3) or audit a local --file export to verify tool I/O.",
            )
        else:
            record("tool_linkage_valid", "pass", 10, "0 tool span(s) have names and I/O")
    else:
        bad_tools = [s for s in tool_spans if not _tool_is_replay_acceptable(s, span_id_counts)]
        if not bad_tools:
            record(
                "tool_linkage_valid",
                "pass",
                10,
                f"{len(tool_spans)} tool span(s) satisfy replay builder acceptance rules",
            )
        else:
            record(
                "tool_linkage_valid",
                "fail",
                10,
                f"{len(bad_tools)} tool span(s) would be skipped by replay builder (missing name, both I/O sides, or lossless capture)",
                "Decorate custom tools with @neatlogs.span(kind='TOOL', tool_name='...', replay=True) and ensure complete lossless input and output.",
            )

    # 7. retriever_freshness (10)
    ret_spans = [s for s in work_spans if _span_kind(s) == "RETRIEVER"]
    if not ret_spans:
        if expect_retriever:
            record(
                "retriever_freshness",
                "fail",
                10,
                "Expected RETRIEVER span(s) but found none",
                "Wrap RAG/search lookups with @neatlogs.span(kind='RETRIEVER', replay=True).",
            )
        else:
            record("retriever_freshness", "pass", 10, "No RETRIEVER spans in trace (not expected)")
    elif metadata_only:
        record(
            "retriever_freshness",
            "warn",
            10,
            f"{len(ret_spans)} RETRIEVER span(s) present, but public API spans are metadata-only (documents not returned)",
            "Set NEATLOGS_API_KEY (for /api/traces/v3) or audit a local --file export to verify document freshness.",
        )
    else:
        total_docs = fresh_docs = 0
        for rs in ret_spans:
            d, f = _extract_docs_with_freshness(rs)
            total_docs += d
            fresh_docs += f
        if total_docs > 0 and fresh_docs == total_docs:
            record(
                "retriever_freshness",
                "pass",
                10,
                f"{fresh_docs}/{total_docs} retrieved document(s) carry freshness timestamps",
            )
        elif total_docs > 0:
            record(
                "retriever_freshness",
                "warn",
                10,
                f"Only {fresh_docs}/{total_docs} retrieved document(s) carry freshness timestamps",
                "Include updated_at or timestamp on retrieved documents so stale retrieval is detectable.",
            )
        else:
            record(
                "retriever_freshness",
                "fail",
                10,
                f"{len(ret_spans)} RETRIEVER span(s) recorded no parseable documents",
                "Return structured document dicts with id, content, score, and updated_at.",
            )

    # 8. session_hygiene (10)
    span_session_id = None
    if root is not None:
        r_attrs = _attrs(root)
        span_session_id = _text(r_attrs.get("neatlogs.session.id")) or _text(
            r_attrs.get("session.id")
        )
    trace_session_id = _text(safe_trace.get("sessionId")) or _text(safe_trace.get("session_id"))
    session_id = span_session_id or trace_session_id
    norm_trace = _normalize_id(trace_id)
    norm_session = _normalize_id(session_id)
    is_backend_default_session = (
        v3_session_fallback
        and span_session_id is None
        and bool(norm_trace)
        and norm_session == norm_trace
    )
    if session_id and not is_backend_default_session:
        if norm_trace and norm_session == norm_trace:
            record(
                "session_hygiene",
                "fail",
                10,
                "session_id equals trace_id (breaks multi-turn conversation grouping)",
                "Pass a stable conversation/thread ID to identify(session_id=...), never trace_id.",
            )
        else:
            record(
                "session_hygiene", "pass", 10, f"Distinct session_id present ({session_id[:24]})"
            )
    elif expect_session:
        record(
            "session_hygiene",
            "fail",
            10,
            "Expected session_id for multi-turn app, but none found",
            "Bind session_id via neatlogs.identify(session_id=...) or @span(kind='WORKFLOW', session_id=...).",
        )
    else:
        record("session_hygiene", "pass", 10, "Single-turn trace (session_id not required)")

    # 9. provenance_and_replayability (5)
    from .replay.frozen_case import MAX_FROZEN_CASE_SPANS

    exceeds_replay_span_limit = len(dict_spans) > MAX_FROZEN_CASE_SPANS
    truncated = any(_is_lossy_or_incomplete(s) for s in work_spans)
    has_provenance = any(
        _attrs(s).get("neatlogs.provenance.release")
        or _attrs(s).get("neatlogs.prompt.version")
        or _attrs(s).get("deployment.environment.name")
        or _attrs(s).get("trace_catalog_environment")
        for s in work_spans
    )
    if truncated:
        record(
            "provenance_and_replayability",
            "fail",
            5,
            "One or more spans are truncated, redacted, or lossy (blocks lossless ReplaySession)",
            "Avoid lossy return types on replayable tools/retrievers or enable presigned overflow.",
        )
    elif exceeds_replay_span_limit:
        record(
            "provenance_and_replayability",
            "fail",
            5,
            f"Trace has {len(dict_spans)} spans, exceeding the replay builder limit of {MAX_FROZEN_CASE_SPANS}",
            f"Split long runs or replay a narrower span subtree within {MAX_FROZEN_CASE_SPANS} spans.",
        )
    elif metadata_only and not has_provenance:
        record(
            "provenance_and_replayability",
            "warn",
            5,
            "Public API spans are metadata-only (no provenance/truncation attributes returned)",
            "Set NEATLOGS_API_KEY (for /api/traces/v3) or audit a local --file export to verify provenance.",
        )
    elif not has_provenance:
        record(
            "provenance_and_replayability",
            "warn",
            5,
            "No release/environment/prompt_version provenance attached",
            "Pass environment= and release= to neatlogs.init() and prompt_version= on WORKFLOW/AGENT spans.",
        )
    else:
        record(
            "provenance_and_replayability",
            "pass",
            5,
            "Lossless capture with deploy/prompt provenance present",
        )

    failed = {c["name"] for c in checks if c["status"] == "fail"}
    jev_ready = not metadata_only and not (
        failed
        & {
            "has_spans",
            "single_workflow_root",
            "root_io_present",
            "trace_finalized",
            "session_hygiene",
        }
    )
    replay_ready = (
        not metadata_only
        and not exceeds_replay_span_limit
        and not (
            failed
            & {
                "has_spans",
                "single_workflow_root",
                "step_ordering_clean",
                "tool_linkage_valid",
                "provenance_and_replayability",
            }
        )
    )
    grade = (
        "ready"
        if score >= 85 and not failed
        else ("needs_attention" if score >= 60 else "incomplete")
    )
    return {
        "trace_id": trace_id,
        "score": score,
        "grade": grade,
        "jev_ready": jev_ready,
        "replay_ready": replay_ready,
        "replay_readiness_verified": not metadata_only,
        "span_count": len(work_spans),
        "checks": checks,
        "recommendations": recommendations,
    }


def audit_readable_spans(
    spans: Sequence[Any],
    *,
    expect_session: bool = False,
    expect_retriever: bool = False,
) -> dict[str, Any]:
    """Audit real in-process OpenTelemetry ReadableSpans directly."""
    trace, span_dicts = readable_spans_to_audit_payload(spans)
    return audit_trace(
        trace,
        span_dicts,
        expect_session=expect_session,
        expect_retriever=expect_retriever,
    )


def _validate_span_dicts(spans: list[Any], *, source: str) -> list[dict[str, Any]]:
    if any(not isinstance(s, dict) for s in spans):
        raise ValueError(f"{source} spans must all be JSON objects")
    return spans


def _load_trace_file(path: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Trace file must contain a JSON object")
    if isinstance(raw.get("trace"), dict) and isinstance(raw.get("spans"), list):
        return raw["trace"], _validate_span_dicts(raw["spans"], source="Trace file")
    if isinstance(raw.get("context"), dict) and isinstance(raw["context"].get("spans"), list):
        source = raw.get("source") if isinstance(raw.get("source"), dict) else {}
        return {
            "trace_id": source.get("traceId", ""),
            "finalizationStatus": "finalized",
        }, _validate_span_dicts(raw["context"]["spans"], source="Trace file")
    if isinstance(raw.get("spans"), list):
        return raw, _validate_span_dicts(raw["spans"], source="Trace file")
    raise ValueError("Trace file must contain 'spans' or a frozen case 'context.spans'")


def _fetch_v3_trace_with_project_key(
    trace_id: str,
    *,
    endpoint: str | None = None,
    env: dict[str, str],
    fetch: Fetcher | None,
    timeout: float,
    err: Callable[[str], None],
) -> tuple[dict[str, Any], list[dict[str, Any]]] | int:
    api_key = (env.get("NEATLOGS_API_KEY") or "").strip()
    if not api_key:
        err(
            "Set NEATLOGS_API_KEY (recommended) or NEATLOGS_TOKEN + NEATLOGS_PROJECT_ID to fetch a trace"
        )
        return EXIT_AUTH
    resolved_endpoint = (
        endpoint
        or env.get("NEATLOGS_ENDPOINT")
        or env.get("NEATLOGS_HOST")
        or "https://ingest.neatlogs.com"
    ).strip()
    parsed = urlparse(resolved_endpoint)
    if (
        parsed.scheme not in ("http", "https")
        or not parsed.hostname
        or parsed.username
        or parsed.password
    ):
        err("NEATLOGS_ENDPOINT must be an absolute http(s) URL")
        return EXIT_USAGE
    app_to_ingest = {
        "app.neatlogs.com": "ingest.neatlogs.com",
        "eu.app.neatlogs.com": "eu.ingest.neatlogs.com",
    }
    mapped_host = app_to_ingest.get(parsed.hostname.rstrip(".").lower())
    if mapped_host is not None:
        port = f":{parsed.port}" if parsed.port is not None else ""
        netloc = f"{mapped_host}{port}"
    else:
        netloc = parsed.netloc
    url = f"{parsed.scheme}://{netloc}/api/traces/v3/{quote(trace_id, safe='')}"
    fetcher = fetch or _v3_default_fetch
    headers = {"Authorization": f"Bearer {api_key}", "x-api-key": api_key}
    try:
        status, body = fetcher(url, headers, timeout)
    except _ResponseTooLargeError as exc:
        err(str(exc))
        return EXIT_ERROR
    except (TimeoutError, urllib.error.URLError, OSError):
        err("Could not reach /api/traces/v3")
        return EXIT_ERROR
    if status in (401, 403):
        err(f"Trace read rejected the API key (HTTP {status})")
        return EXIT_AUTH
    if status == 409:
        try:
            err_payload = json.loads(body) if body else {}
        except (TypeError, ValueError):
            err_payload = {}
        if isinstance(err_payload, dict) and err_payload.get("finalizationStatus") == "dlq":
            err("Trace ingestion failed for good (finalizationStatus=dlq); retrying will not help")
            return EXIT_ERROR
        err("Trace not ready (HTTP 409)")
        return EXIT_NOT_READY
    if status in (202, 404):
        err(f"Trace not ready or not found (HTTP {status})")
        return EXIT_NOT_READY
    if not 200 <= status < 300:
        err(f"Trace read failed (HTTP {status})")
        return EXIT_ERROR
    try:
        payload = json.loads(body)
    except (TypeError, ValueError):
        err("Trace read returned invalid JSON")
        return EXIT_ERROR
    if not isinstance(payload, dict):
        err("Trace read returned an unexpected response")
        return EXIT_ERROR
    data = payload.get("data") if isinstance(payload.get("data"), dict) else payload
    if not isinstance(data, dict) or not isinstance(data.get("spans"), list):
        err("Trace read response did not include spans")
        return EXIT_ERROR
    if any(not isinstance(s, dict) for s in data["spans"]):
        err("Trace read response contained malformed non-object spans")
        return EXIT_ERROR
    return data, data["spans"]


def run_trace_audit(
    trace_id: str | None,
    *,
    file_path: str | None = None,
    endpoint: str | None = None,
    as_json: bool = False,
    expect_session: bool = False,
    expect_retriever: bool = False,
    env: dict | None = None,
    fetch: Fetcher | None = None,
    timeout: float = 5.0,
    out: Callable[[str], None] = print,
    err: Callable[[str], None] | None = None,
) -> int:
    err = err or (lambda line: print(line, file=sys.stderr))
    if bool(trace_id) == bool(file_path):
        err("Specify exactly one of <trace_id> or --file <path>")
        return EXIT_USAGE
    metadata_only = False
    v3_session_fallback = False
    if file_path is not None:
        try:
            trace, spans = _load_trace_file(file_path)
        except (OSError, ValueError) as exc:
            err(f"Could not read trace file: {exc}")
            return EXIT_ERROR
    else:
        effective_env = os.environ if env is None else env
        if (effective_env.get("NEATLOGS_API_KEY") or "").strip():
            fetched = _fetch_v3_trace_with_project_key(
                str(trace_id),
                endpoint=endpoint,
                env=effective_env,
                fetch=fetch,
                timeout=timeout,
                err=err,
            )
            v3_session_fallback = True
        elif (effective_env.get("NEATLOGS_TOKEN") or "").strip():
            public_env = dict(effective_env)
            public_env["NEATLOGS_HOST"] = resolve_public_api_host(effective_env, endpoint=endpoint)
            fetched = fetch_public_trace(
                str(trace_id), env=public_env, fetch=fetch, timeout=timeout, err=err
            )
            metadata_only = True
            v3_session_fallback = True
        else:
            err(
                "Set NEATLOGS_API_KEY (recommended for full payload audit) or "
                "NEATLOGS_TOKEN + NEATLOGS_PROJECT_ID to fetch a trace"
            )
            return EXIT_AUTH
        if isinstance(fetched, int):
            return fetched
        trace, spans = fetched

    report = audit_trace(
        trace,
        spans,
        expect_session=expect_session,
        expect_retriever=expect_retriever,
        metadata_only=metadata_only,
        v3_session_fallback=v3_session_fallback,
    )
    if as_json:
        out(json.dumps(report, indent=2, sort_keys=True))
    else:
        replay_label = (
            "False (unverified: metadata-only)"
            if not report["replay_readiness_verified"]
            else str(report["replay_ready"])
        )
        out(
            f"trace {report['trace_id'] or '(file)'}: {report['grade'].upper()} "
            f"(score={report['score']}/100, jev_ready={report['jev_ready']}, "
            f"replay_ready={replay_label})"
        )
        for check in report["checks"]:
            out(f"  {check['status'].upper():4s} {check['name']}: {check['message']}")
        for rec in report["recommendations"]:
            out(f"  tip: {rec}")
    return EXIT_OK if report["grade"] == "ready" else EXIT_CHECKS_FAILED
