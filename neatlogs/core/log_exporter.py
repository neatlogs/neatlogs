"""
NeatlogsLogExporter — converts OTel LogRecords into span-format payloads
and routes them through the existing NeatlogsExporter HTTP batch pipeline.

Each log record is stored as a row in ClickHouse spans table with
span_type = 'LOG', appearing as a child of the active span in the timeline.
"""

from __future__ import annotations

import random
import sys
import time
from typing import TYPE_CHECKING, Sequence

from opentelemetry.sdk._logs.export import LogRecordExporter, LogRecordExportResult

if TYPE_CHECKING:
    from opentelemetry.sdk._logs import ReadableLogRecord

    from .exporter import NeatlogsExporter

# Python 3.10+ exposes stdlib module names directly; fall back to empty set on older versions.
_STDLIB_MODULE_NAMES: frozenset = getattr(sys, "stdlib_module_names", frozenset())


def _is_external_module(logger_name: str) -> bool:
    """Return True if logger_name belongs to stdlib or a site-packages library."""
    if not logger_name:
        return False
    parts = logger_name.split(".")
    top_pkg = parts[0]
    # stdlib check (Python 3.10+)
    if top_pkg in _STDLIB_MODULE_NAMES:
        return True
    # site-packages check: walk from most-specific to top-level module.
    # Namespace packages (e.g. "openinference") have __file__=None, so checking
    # only the top-level package misses libraries like openinference-instrumentation-*.
    # Scanning downward finds the first concrete submodule that has a real __file__.
    for i in range(len(parts), 0, -1):
        mod_name = ".".join(parts[:i])
        mod = sys.modules.get(mod_name)
        if mod is not None:
            mod_file = getattr(mod, "__file__", "") or ""
            if mod_file:  # concrete module (namespace packages have __file__=None)
                return "site-packages" in mod_file or "/dist-packages/" in mod_file
    return False


class NeatlogsLogExporter(LogRecordExporter):
    """
    Converts OTel LogRecords into the span payload format used by NeatlogsExporter,
    then forwards them through the existing HTTP batch pipeline.

    Filtering:
    - Records with no trace_id (logged outside a traced span) are dropped.
    - Records from neatlogs internals (logger name "neatlogs") are dropped.
    """

    def __init__(self, span_exporter: "NeatlogsExporter") -> None:
        self._span_exporter = span_exporter

    def export(self, batch: Sequence["ReadableLogRecord"]) -> LogRecordExportResult:
        for readable in batch:
            payload = self._to_span_payload(readable)
            if payload is not None:
                self._span_exporter.export(payload)
        return LogRecordExportResult.SUCCESS

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return True

    def _to_span_payload(self, readable: "ReadableLogRecord") -> dict | None:
        lr = readable.log_record

        # Drop records with no active trace (outside a span)
        if not lr.trace_id or lr.trace_id == 0:
            return None

        # Drop logs from stdlib or site-packages (httpcore, asyncio, openai SDK, etc.)
        # Check instrumentation_scope.name (= Python logger name set by LoggingInstrumentor)
        scope = getattr(readable, "instrumentation_scope", None)
        scope_name = getattr(scope, "name", "") or ""
        if _is_external_module(scope_name):
            return None

        # Fallback: check code.filepath if set by LoggingInstrumentor
        filepath = (lr.attributes or {}).get("code.filepath", "") or ""
        if "site-packages" in filepath or "/dist-packages/" in filepath:
            return None

        trace_id = f"{lr.trace_id:032x}"
        # lr.span_id is the ACTIVE span — becomes the parent of this log row
        parent_span_id = f"{lr.span_id:016x}" if lr.span_id else None

        # Generate a new unique span_id for this log row in ClickHouse
        new_span_id = f"{random.getrandbits(64):016x}"

        # Timing — use nanoseconds (same as spans)
        ts_ns = lr.timestamp or lr.observed_timestamp or time.time_ns()

        body = str(lr.body) if lr.body is not None else ""
        level = (lr.severity_text or "info").lower()

        # Build attributes
        attrs: dict = {
            "openinference.span.kind": "LOG",
            "neatlogs.span.kind": "log",
            "neatlogs.internal": True,
            "neatlogs.input.value": body,  # kafka consumer maps this → input_value column
            "input.value": body,  # OpenInference standard key (kept for compatibility)
            "input.mime_type": "text/plain",
            "log.level": level,
        }

        # Merge structured attributes from LogRecord (log.template, log.{key}, etc.)
        if lr.attributes:
            for k, v in lr.attributes.items():
                # Coerce to str for OTel attribute compatibility
                attrs[k] = v if isinstance(v, (str, int, float, bool)) else str(v)

        # Use log.template as span name (low-cardinality) if neatlogs.log() set it,
        # otherwise fall back to the rendered body
        span_name = attrs.get("log.template") or body

        # Include resource attributes (workflow_name, session_id, etc.)
        resource_attrs: dict = {}
        if readable.resource and readable.resource.attributes:
            for k, v in readable.resource.attributes.items():
                if isinstance(v, (str, int, float, bool)):
                    resource_attrs[k] = v
                elif isinstance(v, (list, tuple)):
                    resource_attrs[k] = list(v)
                else:
                    resource_attrs[k] = str(v)

        return {
            "trace_id": trace_id,
            "span_id": new_span_id,
            "parent_span_id": parent_span_id,
            "name": span_name,
            "kind": "log",
            "start_time": ts_ns,
            "end_time": ts_ns,
            "duration_ns": 0,
            "attributes": attrs,
            "resource": {"attributes": resource_attrs},
            "status": {"code": "OK", "description": ""},
            "events": [],
        }
