"""Span buffer that collects spans grouped by trace and flushes on root completion."""

from __future__ import annotations

import atexit
import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("neatlogs.mcp.buffer")


class SpanRecord:
    """A single span in the buffer."""

    __slots__ = (
        "span_id",
        "parent_span_id",
        "name",
        "kind",
        "start_time",
        "end_time",
        "status_code",
        "status_message",
        "attributes",
        "events",
        "is_root",
    )

    def __init__(
        self,
        span_id: str,
        parent_span_id: Optional[str],
        name: str,
        kind: str,
        start_time: str,
        *,
        end_time: Optional[str] = None,
        status_code: str = "UNSET",
        status_message: str = "",
        attributes: Optional[Dict[str, Any]] = None,
        events: Optional[List[Dict[str, Any]]] = None,
        is_root: bool = False,
    ):
        self.span_id = span_id
        self.parent_span_id = parent_span_id
        self.name = name
        self.kind = kind
        self.start_time = start_time
        self.end_time = end_time
        self.status_code = status_code
        self.status_message = status_message
        self.attributes = attributes or {}
        self.events = events or []
        self.is_root = is_root

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "span_id": self.span_id,
            "name": self.name,
            "kind": self.kind,
            "start_time": self.start_time,
            "status_code": self.status_code,
            "attributes": self.attributes,
        }
        if self.parent_span_id:
            d["parent_span_id"] = self.parent_span_id
        if self.end_time:
            d["end_time"] = self.end_time
        if self.status_message:
            d["status_message"] = self.status_message
        if self.events:
            d["events"] = self.events
        return d


class TraceBuffer:
    """Holds all spans for a single trace."""

    def __init__(self, trace_id: str, workflow_name: str, framework: Optional[str] = None):
        self.trace_id = trace_id
        self.workflow_name = workflow_name
        self.framework = framework
        self.metadata: Dict[str, Any] = {}
        self.spans: Dict[str, SpanRecord] = {}
        self.root_span_id: Optional[str] = None
        self.created_at = time.monotonic()

    def add_span(self, span: SpanRecord) -> None:
        self.spans[span.span_id] = span
        if span.is_root:
            self.root_span_id = span.span_id

    def get_span(self, span_id: str) -> Optional[SpanRecord]:
        return self.spans.get(span_id)

    def is_complete(self) -> bool:
        if not self.root_span_id:
            return False
        root = self.spans.get(self.root_span_id)
        return root is not None and root.end_time is not None

    def to_log_trace_payload(self) -> Dict[str, Any]:
        root = self.spans.get(self.root_span_id, None) if self.root_span_id else None

        child_spans = []
        for span in self.spans.values():
            if span.span_id == self.root_span_id:
                continue
            child_spans.append(span.to_dict())

        payload: Dict[str, Any] = {
            "workflow_name": self.workflow_name,
            "spans": child_spans,
        }
        if self.framework:
            payload["framework"] = self.framework
        if self.metadata:
            payload["metadata"] = self.metadata
        if root:
            payload["timestamp"] = root.start_time
            if root.end_time:
                start = datetime.fromisoformat(root.start_time.replace("Z", "+00:00"))
                end = datetime.fromisoformat(root.end_time.replace("Z", "+00:00"))
                payload["latency_ms"] = (end - start).total_seconds() * 1000

        return payload


class SpanBuffer:
    """Thread-safe buffer managing multiple trace buffers with auto-flush on completion."""

    def __init__(
        self,
        on_flush: Callable[[Dict[str, Any]], None],
        stale_timeout_s: float = 300.0,
    ):
        self._traces: Dict[str, TraceBuffer] = {}
        self._lock = threading.Lock()
        self._on_flush = on_flush
        self._stale_timeout_s = stale_timeout_s
        self._span_log_handle = self._init_span_log()
        atexit.register(self.flush_all)

    def _init_span_log(self) -> Any:
        """Initialize span file logging if NEATLOGS_LOG_SPANS=true (same env var as SDK)."""
        if os.getenv("NEATLOGS_LOG_SPANS", "").lower() not in ("true", "1", "yes"):
            return None
        path = os.path.join(
            os.getcwd(),
            os.getenv("NEATLOGS_LOG_SPANS_FILE", "spans_mcp.log"),
        )
        try:
            handle = open(path, "a", encoding="utf-8")
            logger.info(f"MCP span logging enabled: {path}")
            return handle
        except Exception as e:
            logger.warning(f"Failed to open span log file {path}: {e}")
            return None

    def _log_span(self, trace_id: str, span: SpanRecord) -> None:
        if not self._span_log_handle or self._span_log_handle.closed:
            return
        try:
            entry = {
                "trace_id": trace_id,
                **span.to_dict(),
            }
            self._span_log_handle.write(json.dumps(entry, default=str) + "\n")
            self._span_log_handle.flush()
        except Exception:
            pass

    def get_or_create_trace(
        self,
        trace_id: str,
        workflow_name: str,
        framework: Optional[str] = None,
    ) -> TraceBuffer:
        with self._lock:
            if trace_id not in self._traces:
                self._traces[trace_id] = TraceBuffer(trace_id, workflow_name, framework)
            return self._traces[trace_id]

    def add_span(self, trace_id: str, span: SpanRecord) -> None:
        with self._lock:
            buf = self._traces.get(trace_id)
            if buf:
                buf.add_span(span)
                self._log_span(trace_id, span)

    def complete_span(self, trace_id: str, span_id: str, end_time: str, **updates: Any) -> None:
        flush_payload = None
        with self._lock:
            buf = self._traces.get(trace_id)
            if not buf:
                return
            span = buf.get_span(span_id)
            if not span:
                return
            span.end_time = end_time
            if "status_code" in updates:
                span.status_code = updates["status_code"]
            if "status_message" in updates:
                span.status_message = updates["status_message"]
            if "output_attrs" in updates:
                span.attributes.update(updates["output_attrs"])

            self._log_span(trace_id, span)

            if buf.is_complete():
                flush_payload = buf.to_log_trace_payload()
                del self._traces[trace_id]

        if flush_payload is not None:
            self._on_flush(flush_payload)

    def flush_all(self) -> None:
        with self._lock:
            pending = list(self._traces.values())
            self._traces.clear()

        for buf in pending:
            try:
                self._on_flush(buf.to_log_trace_payload())
            except Exception:
                pass

    def flush_stale(self) -> None:
        now = time.monotonic()
        stale_payloads = []
        with self._lock:
            stale_ids = [
                tid
                for tid, buf in self._traces.items()
                if now - buf.created_at > self._stale_timeout_s
            ]
            for tid in stale_ids:
                stale_payloads.append(self._traces.pop(tid).to_log_trace_payload())

        for payload in stale_payloads:
            try:
                self._on_flush(payload)
            except Exception:
                pass
