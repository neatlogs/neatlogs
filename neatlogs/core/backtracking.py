"""Fail-open backtracking helpers: step sequencing, provenance, checkpoints, and diagnostics."""

from __future__ import annotations

import hashlib
import json
import os
from collections import OrderedDict
from typing import Any, Optional

from .error_sanitization import sanitize_error_text

MAX_TRACKED_TRACES = 10_000
MAX_STATE_DIGEST_BYTES = 256 * 1024
MAX_STATE_DELTA_BYTES = 32 * 1024
CHECKPOINT_DELTA_TRUNCATION_MARKER = "...[checkpoint-delta-truncated]"
MAX_DIAGNOSTIC_MESSAGE_CHARS = 1024
MAX_DIAGNOSTIC_DETAILS_BYTES = 8 * 1024


class StepSequenceTracker:
    """Bounded per-trace monotonic sequence generator (caller holds synchronization lock)."""

    def __init__(self, max_traces: int = MAX_TRACKED_TRACES) -> None:
        self._max_traces = max(1, int(max_traces))
        self._counts: OrderedDict[int, int] = OrderedDict()

    def next_seq(self, trace_id: int) -> int:
        seq = self._counts.get(trace_id, 0) + 1
        self._counts[trace_id] = seq
        self._counts.move_to_end(trace_id)
        while len(self._counts) > self._max_traces:
            self._counts.popitem(last=False)
        return seq

    def release(self, trace_id: int) -> None:
        self._counts.pop(trace_id, None)

    def __len__(self) -> int:
        return len(self._counts)


def resolve_provenance(
    environment: Optional[str] = None,
    release: Optional[str] = None,
) -> tuple[Optional[str], Optional[str]]:
    """Resolve explicit or NEATLOGS_-prefixed provenance values without reading ambient env."""
    env_val = environment if environment is not None else os.getenv("NEATLOGS_ENVIRONMENT")
    rel_val = release if release is not None else os.getenv("NEATLOGS_RELEASE")
    resolved_env = str(env_val).strip() if env_val is not None else ""
    resolved_rel = str(rel_val).strip() if rel_val is not None else ""
    return (resolved_env or None), (resolved_rel or None)


def apply_provenance_attributes(
    span: Any,
    *,
    environment: Optional[str],
    release: Optional[str],
    is_root: bool,
) -> None:
    """Stamp optional deployment/release provenance on owned root spans only."""
    if not is_root or span is None:
        return
    try:
        if environment:
            span.set_attribute("deployment.environment.name", environment)
        if release:
            span.set_attribute("neatlogs.provenance.release", release)
    except Exception:
        pass


def _active_recording_span() -> Any:
    """Resolve the active Neatlogs span across private/isolated and global providers."""
    try:
        from opentelemetry.trace import get_current_span

        from .._wrap_utils import active_neatlogs_context

        ctx = active_neatlogs_context()
        span = get_current_span(ctx) if ctx is not None else get_current_span()
        if span is not None and span.is_recording():
            return span
    except Exception:
        return None
    return None


def _bounded_json_bytes(value: Any, *, max_bytes: int) -> tuple[bytes, bool]:
    """Stream-serialize arbitrary Python objects to UTF-8 JSON, stopping early at max_bytes."""
    seen: set[int] = set()

    def _safe_default(obj: Any) -> Any:
        obj_id = id(obj)
        if obj_id in seen:
            return "<cycle>"
        seen.add(obj_id)
        try:
            for attr in ("model_dump", "to_dict", "dict"):
                fn = getattr(obj, attr, None)
                if callable(fn):
                    return fn()
            return repr(obj)
        except Exception:
            return f"<{type(obj).__name__}>"

    encoder = json.JSONEncoder(
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=_safe_default,
    )
    buf = bytearray()
    truncated = False
    try:
        for chunk in encoder.iterencode(value):
            raw = chunk.encode("utf-8")
            remaining = max_bytes - len(buf)
            if len(raw) > remaining:
                buf.extend(raw[:remaining])
                truncated = True
                break
            buf.extend(raw)
    except (TypeError, ValueError, RecursionError):
        raw = json.dumps(repr(value)[:max_bytes], ensure_ascii=False).encode("utf-8")
        if len(raw) > max_bytes:
            return raw[:max_bytes], True
        return raw, False
    return bytes(buf), truncated


def checkpoint(
    checkpoint_id: str,
    *,
    state: Any = None,
    state_delta: Any = None,
) -> Optional[str]:
    """Attach a backtrackable checkpoint anchor (and optional digest/delta) to the active span.

    Fail-open: never raises into caller code. Returns ``checkpoint_id`` when recorded,
    else ``None``.
    """
    try:
        cid = str(checkpoint_id or "").strip()
        if not cid:
            return None
        span = _active_recording_span()
        if span is None:
            return None
        span.set_attribute("neatlogs.checkpoint.id", cid[:200])
        if state is not None:
            state_bytes, state_clipped = _bounded_json_bytes(
                state, max_bytes=MAX_STATE_DIGEST_BYTES
            )
            digest = hashlib.sha256(state_bytes).hexdigest()
            span.set_attribute("neatlogs.checkpoint.state_digest", f"sha256:{digest}")
            if state_clipped:
                span.set_attribute("neatlogs.checkpoint.state_digest_truncated", True)
        if state_delta is not None:
            delta_bytes, delta_clipped = _bounded_json_bytes(
                state_delta, max_bytes=MAX_STATE_DELTA_BYTES
            )
            delta_text = delta_bytes.decode("utf-8", errors="ignore")
            if delta_clipped:
                delta_text = f"{delta_text}{CHECKPOINT_DELTA_TRUNCATION_MARKER}"
                span.set_attribute("neatlogs.checkpoint.state_delta_truncated", True)
            span.set_attribute("neatlogs.checkpoint.state_delta", delta_text)
        return cid[:200]
    except Exception:
        return None


def report_diagnostic(
    category: str,
    message: str,
    *,
    severity: str = "warning",
    details: Optional[dict[str, Any]] = None,
) -> bool:
    """Record a sanitized agent/tool self-diagnostic event and summary attrs on the active span.

    Fail-open: never raises into caller code. Returns ``True`` when recorded, else ``False``.
    """
    try:
        cat = str(category or "").strip()[:128]
        if not cat:
            return False
        sev = str(severity or "warning").strip().lower()
        if sev not in {"info", "warning", "error", "critical"}:
            sev = "warning"
        msg = sanitize_error_text(str(message or "").strip())[:MAX_DIAGNOSTIC_MESSAGE_CHARS]
        span = _active_recording_span()
        if span is None:
            return False
        event_attrs: dict[str, Any] = {
            "neatlogs.diagnostic.category": cat,
            "neatlogs.diagnostic.severity": sev,
            "neatlogs.diagnostic.message": msg,
        }
        if details is not None:
            details_bytes, details_clipped = _bounded_json_bytes(
                details, max_bytes=MAX_DIAGNOSTIC_DETAILS_BYTES
            )
            details_text = sanitize_error_text(details_bytes.decode("utf-8", errors="ignore"))
            if details_clipped:
                details_text = f"{details_text}{CHECKPOINT_DELTA_TRUNCATION_MARKER}"
            event_attrs["neatlogs.diagnostic.details"] = details_text
        span.add_event("neatlogs.diagnostic", attributes=event_attrs)
        for key, value in event_attrs.items():
            span.set_attribute(key, value)
        return True
    except Exception:
        return False
