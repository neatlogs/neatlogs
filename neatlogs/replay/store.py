"""In-memory trace index for replay.

The SDK has no local trace database (spans export via OTLP to the Neatlogs
backend). This store is a minimal local index over already-captured spans so
``replay_trace(trace_id, ...)`` has a source of truth in tests and offline
use. A future backend fetch (``GET /api/traces/v3/:traceId``) can implement
the same protocol without changing callers.
"""

from __future__ import annotations

from threading import RLock
from typing import Any, Dict, List, Protocol

from .types import TraceNotFoundError, trace_id


class TraceStore(Protocol):
    """Minimal trace source for replay. A backend fetch can implement this."""

    def get_trace(self, trace_id: str) -> List[Any]:
        """Return the spans of one trace; raise TraceNotFoundError if unknown."""
        ...

    def put_trace(self, trace_id: str, spans: List[Any]) -> None:
        """Index spans under an explicit trace id."""
        ...


class InMemoryTraceStore:
    """Thread-safe dict-backed store. Values are stored by reference; callers
    must treat retrieved spans as read-only (replay never mutates them)."""

    def __init__(self) -> None:
        self._traces: Dict[str, List[Any]] = {}
        self._lock = RLock()

    def put_trace(self, trace_id: str, spans: List[Any]) -> None:
        """Index spans under an explicit trace id."""
        with self._lock:
            self._traces[trace_id] = list(spans)

    def put_spans(self, spans: List[Any]) -> str:
        """Index spans by their own trace id. Returns the trace id.

        All spans must belong to a single trace — exporter drains covering
        several traces must be grouped first (e.g. by trace id).
        """
        if not spans:
            raise ValueError("put_spans requires at least one span")
        trace_ids = {trace_id(span) for span in spans}
        if len(trace_ids) > 1:
            raise ValueError(
                "put_spans requires spans from a single trace; " f"got {len(trace_ids)} trace ids"
            )
        resolved = trace_id(spans[0])
        self.put_trace(resolved, spans)
        return resolved

    def get_trace(self, trace_id: str) -> List[Any]:
        """Return a copy of the stored spans; raise TraceNotFoundError if unknown."""
        with self._lock:
            spans = self._traces.get(trace_id)
        if spans is None:
            raise TraceNotFoundError(f"Unknown trace_id: {trace_id}")
        return list(spans)
