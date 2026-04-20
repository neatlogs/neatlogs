"""
NeatlogsLogFilter — a LogRecordProcessor that drops unwanted log records
(no active trace, stdlib/site-packages internals) before forwarding to
the downstream exporter (OTLPLogExporter → /v1/logs).

Replaces the old NeatlogsLogExporter which converted LogRecords into
span-shaped JSON payloads for /api/data/v4/batch. The conversion is now
done server-side by the /v1/logs OTLP receiver.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from opentelemetry.sdk._logs import LogRecordProcessor

if TYPE_CHECKING:
    from opentelemetry.sdk._logs import LogData, LoggerProvider

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


class NeatlogsLogFilter(LogRecordProcessor):
    """
    Filtering LogRecordProcessor that drops records before they reach OTLPLogExporter.

    Drops:
    - Records with no trace_id (logged outside a traced span)
    - Records from stdlib or site-packages (httpcore, asyncio, openai SDK internals, etc.)

    All other records are forwarded to the wrapped processor (typically a
    BatchLogRecordProcessor wrapping OTLPLogExporter → /v1/logs).
    """

    def __init__(self, downstream: LogRecordProcessor) -> None:
        self._downstream = downstream

    def on_emit(self, log_data: "LogData") -> None:
        lr = log_data.log_record

        # Drop records with no active trace (logged outside a span)
        if not lr.trace_id or lr.trace_id == 0:
            return

        # Drop logs from stdlib or site-packages (httpcore, asyncio, openai SDK, etc.)
        # instrumentation_scope.name is the Python logger name set by LoggingInstrumentor
        scope = getattr(log_data, "instrumentation_scope", None)
        scope_name = getattr(scope, "name", "") or ""
        if _is_external_module(scope_name):
            return

        # Fallback: check code.filepath attribute set by LoggingInstrumentor
        filepath = (lr.attributes or {}).get("code.filepath", "") or ""
        if "site-packages" in filepath or "/dist-packages/" in filepath:
            return

        self._downstream.on_emit(log_data)

    def shutdown(self) -> None:
        self._downstream.shutdown()

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return self._downstream.force_flush(timeout_millis)
