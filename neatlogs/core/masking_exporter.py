"""Fail-closed masking at Neatlogs' final telemetry export boundary."""

from __future__ import annotations

import asyncio
import copy
import inspect
import logging
import queue
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Callable

try:
    from opentelemetry.sdk._logs import ReadableLogRecord
    from opentelemetry.sdk._logs.export import LogRecordExporter, LogRecordExportResult
except ImportError:  # OpenTelemetry 1.35-1.38 compatibility
    from opentelemetry.sdk._logs import LogData as ReadableLogRecord
    from opentelemetry.sdk._logs.export import (
        LogExporter as LogRecordExporter,
        LogExportResult as LogRecordExportResult,
    )

from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import Event, ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.trace import Link, Status, StatusCode

from .delivery import DeliveryDiagnostics
from .mask import effective_mask

logger = logging.getLogger(__name__)
_MASK_POOL_SIZE = 4


@dataclass(frozen=True)
class MaskContext:
    """Deadline/cancellation information supplied to context-aware masks."""

    signal_type: str
    timeout_seconds: float
    deadline_monotonic: float
    cancelled: threading.Event


def _accepts_context(mask: Callable) -> bool:
    """Require an explicit keyword-only context opt-in.

    The original public callback contract accepted one positional snapshot.
    Binding a second positional argument breaks callbacks that happen to have
    an unrelated optional positional parameter.
    """
    try:
        parameter = inspect.signature(mask).parameters.get("context")
    except (TypeError, ValueError):
        return False
    return parameter is not None and parameter.kind is inspect.Parameter.KEYWORD_ONLY


def _mask_call(
    mask: Callable,
    snapshot: dict[str, Any],
    timeout_seconds: float,
    context: MaskContext,
):
    result = mask(snapshot, context=context) if _accepts_context(mask) else mask(snapshot)
    if inspect.isawaitable(result):
        return asyncio.run(asyncio.wait_for(result, timeout=timeout_seconds))
    return result


class _MaskWorkerPool:
    """Fixed workers created before atexit; submissions never create threads."""

    def __init__(self, max_workers: int = _MASK_POOL_SIZE) -> None:
        self._tasks: queue.Queue[Callable[[], None]] = queue.Queue()
        self._slots = threading.BoundedSemaphore(max_workers)
        self._workers = [
            threading.Thread(
                target=self._run,
                name=f"neatlogs-mask-{index}",
                daemon=True,
            )
            for index in range(max_workers)
        ]
        for worker in self._workers:
            worker.start()

    def _run(self) -> None:
        while True:
            operation = self._tasks.get()
            try:
                operation()
            finally:
                self._slots.release()

    def submit(self, operation: Callable[[], None]) -> bool:
        if not self._slots.acquire(blocking=False):
            return False
        self._tasks.put_nowait(operation)
        return True


_mask_pool: _MaskWorkerPool | None = None
_mask_pool_lock = threading.Lock()


def _get_mask_pool() -> _MaskWorkerPool:
    global _mask_pool
    with _mask_pool_lock:
        if _mask_pool is None:
            _mask_pool = _MaskWorkerPool()
        return _mask_pool


class _MaskRunner:
    def __init__(self, timeout_seconds: float = 5.0, max_workers: int = 4) -> None:
        self._timeout_seconds = timeout_seconds
        self._max_workers = max_workers
        self._pool = _get_mask_pool()
        self._closed = threading.Event()
        self._active_lock = threading.Lock()
        self._active_cancellations: set[threading.Event] = set()

    def apply(self, mask: Callable, snapshot: dict[str, Any]) -> dict[str, Any] | None:
        return self.apply_many(((mask, snapshot),))[0]

    def apply_many(
        self,
        items: Sequence[tuple[Callable, dict[str, Any]]],
    ) -> list[dict[str, Any] | None]:
        """Mask a batch concurrently under one bounded export-worker deadline."""
        if not items:
            return []
        results: list[dict[str, Any] | None] = [None] * len(items)
        deadline = time.monotonic() + self._timeout_seconds
        result_queue: queue.Queue[tuple[int, bool, Any, dict[str, Any]]] = queue.Queue()
        pending = list(range(len(items)))
        active: dict[int, threading.Event] = {}

        def launch(index: int) -> bool:
            if self._closed.is_set():
                return False
            mask, snapshot = items[index]
            candidate = copy.deepcopy(snapshot)
            cancelled = threading.Event()
            context = MaskContext(
                signal_type=str(snapshot.get("signal") or "span"),
                timeout_seconds=max(0.0, deadline - time.monotonic()),
                deadline_monotonic=deadline,
                cancelled=cancelled,
            )
            active[index] = cancelled
            with self._active_lock:
                self._active_cancellations.add(cancelled)

            def run() -> None:
                try:
                    remaining = max(0.001, deadline - time.monotonic())
                    result = _mask_call(mask, candidate, remaining, context)
                    result_queue.put((index, True, result, candidate))
                except BaseException as exc:
                    result_queue.put((index, False, exc, candidate))
                finally:
                    with self._active_lock:
                        self._active_cancellations.discard(cancelled)

            if not self._pool.submit(run):
                with self._active_lock:
                    self._active_cancellations.discard(cancelled)
                active.pop(index, None)
                return False
            return True

        while pending and len(active) < self._max_workers:
            if not launch(pending[0]):
                break
            pending.pop(0)

        while active and time.monotonic() < deadline:
            try:
                index, succeeded, result, candidate = result_queue.get(
                    timeout=max(0.001, deadline - time.monotonic())
                )
            except queue.Empty:
                break
            active.pop(index, None)
            if succeeded and result is None:
                results[index] = candidate
            elif succeeded and isinstance(result, Mapping):
                results[index] = dict(result)
            elif succeeded:
                logger.error("[neatlogs] mask returned a non-mapping; telemetry item dropped")
            else:
                logger.error(
                    "[neatlogs] mask failed; telemetry item dropped (%s)",
                    type(result).__name__,
                )
            while pending and len(active) < self._max_workers:
                if not launch(pending[0]):
                    break
                pending.pop(0)

        if active or pending:
            for cancelled in active.values():
                cancelled.set()
            logger.error(
                "[neatlogs] mask deadline/capacity exhausted; %d telemetry item(s) dropped",
                len(active) + len(pending),
            )
        return results

    def shutdown(self) -> None:
        self._closed.set()
        with self._active_lock:
            for cancelled in self._active_cancellations:
                cancelled.set()


def _attrs(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): item for key, item in value.items() if item is not None}


def _span_snapshot(span: ReadableSpan) -> dict[str, Any]:
    return {
        "signal": "span",
        "name": span.name,
        "attributes": dict(span.attributes or {}),
        "events": [
            {
                "name": event.name,
                "timestamp": event.timestamp,
                "attributes": dict(event.attributes or {}),
            }
            for event in span.events
        ],
        "links": [
            {
                "trace_id": f"{link.context.trace_id:032x}",
                "span_id": f"{link.context.span_id:016x}",
                "attributes": dict(link.attributes or {}),
            }
            for link in span.links
        ],
        "resource": {"attributes": dict(span.resource.attributes or {})},
        "status": {
            "code": span.status.status_code.name,
            "description": span.status.description,
        },
    }


def _masked_span(span: ReadableSpan, snapshot: Mapping[str, Any]) -> ReadableSpan:
    events = []
    for item in snapshot.get("events", ()):
        if not isinstance(item, Mapping):
            continue
        events.append(
            Event(
                str(item.get("name") or "event"),
                attributes=_attrs(item.get("attributes")),
                timestamp=item.get("timestamp"),
            )
        )

    links = []
    for index, item in enumerate(snapshot.get("links", ())):
        if index >= len(span.links) or not isinstance(item, Mapping):
            continue
        links.append(Link(span.links[index].context, attributes=_attrs(item.get("attributes"))))

    resource_data = snapshot.get("resource")
    resource_attrs = (
        _attrs(resource_data.get("attributes")) if isinstance(resource_data, Mapping) else {}
    )
    status_data = snapshot.get("status")
    status = span.status
    if isinstance(status_data, Mapping):
        code_name = str(status_data.get("code") or span.status.status_code.name)
        try:
            code = StatusCode[code_name]
        except KeyError:
            code = span.status.status_code
        status = Status(code, status_data.get("description"))

    attributes = _attrs(snapshot.get("attributes"))
    attributes.pop("neatlogs.mask_id", None)
    return ReadableSpan(
        name=str(snapshot.get("name") or span.name),
        context=span.context,
        parent=span.parent,
        resource=Resource(resource_attrs, schema_url=span.resource.schema_url),
        attributes=attributes,
        events=events,
        links=links,
        kind=span.kind,
        status=status,
        start_time=span.start_time,
        end_time=span.end_time,
        instrumentation_scope=span.instrumentation_scope,
    )


class MaskingSpanExporter(SpanExporter):
    """Clone, mask and export spans; callback failures drop only the affected span."""

    def __init__(
        self,
        inner: SpanExporter,
        mask: Callable | None,
        timeout_seconds: float = 5.0,
        diagnostics: DeliveryDiagnostics | None = None,
    ) -> None:
        self._inner = inner
        self._global_mask = mask
        self._runner = _MaskRunner(timeout_seconds)
        self._diagnostics = diagnostics

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        prepared: list[ReadableSpan | None] = [None] * len(spans)
        masked_inputs: list[tuple[Callable, dict[str, Any]]] = []
        masked_indexes: list[int] = []
        for index, span in enumerate(spans):
            snapshot = _span_snapshot(span)
            mask = effective_mask(snapshot, self._global_mask)
            if mask is None:
                prepared[index] = span
                continue
            masked_inputs.append((mask, snapshot))
            masked_indexes.append(index)
        for index, masked in zip(masked_indexes, self._runner.apply_many(masked_inputs)):
            if masked is not None:
                prepared[index] = _masked_span(spans[index], masked)
            elif self._diagnostics is not None:
                self._diagnostics.record_masked_drop("span")
        kept = [span for span in prepared if span is not None]
        if not kept:
            return SpanExportResult.SUCCESS
        return self._inner.export(kept)

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        result = self._inner.force_flush(timeout_millis)
        return True if result is None else bool(result)

    def shutdown(self) -> None:
        self._runner.shutdown()
        self._inner.shutdown()


def _log_snapshot(item: ReadableLogRecord) -> dict[str, Any]:
    record = item.log_record
    resource = getattr(item, "resource", None) or getattr(record, "resource", None)
    return {
        "signal": "log",
        "name": item.instrumentation_scope.name if item.instrumentation_scope else "log",
        "body": record.body,
        "attributes": dict(record.attributes or {}),
        "resource": {"attributes": dict(resource.attributes or {}) if resource else {}},
        "severity_text": record.severity_text,
        "event_name": record.event_name,
    }


class MaskingLogExporter(LogRecordExporter):
    """Apply the same global fail-closed mask contract to correlated logs."""

    def __init__(
        self,
        inner: LogRecordExporter,
        mask: Callable | None,
        timeout_seconds: float = 5.0,
        diagnostics: DeliveryDiagnostics | None = None,
    ) -> None:
        self._inner = inner
        self._mask = mask
        self._runner = _MaskRunner(timeout_seconds)
        self._diagnostics = diagnostics

    def export(self, batch: Sequence[ReadableLogRecord]) -> LogRecordExportResult:
        if self._mask is None:
            return self._inner.export(batch)
        kept = []
        snapshots = [(self._mask, _log_snapshot(item)) for item in batch]
        for item, masked in zip(batch, self._runner.apply_many(snapshots)):
            if masked is None:
                if self._diagnostics is not None:
                    self._diagnostics.record_masked_drop("log")
                continue
            clone = copy.copy(item)
            clone_record = copy.copy(item.log_record)
            clone_record.body = masked.get("body")
            clone_record.attributes = _attrs(masked.get("attributes"))
            clone_record.severity_text = masked.get("severity_text")
            clone_record.event_name = masked.get("event_name")
            object.__setattr__(clone, "log_record", clone_record)
            resource_data = masked.get("resource")
            resource_attrs = (
                _attrs(resource_data.get("attributes"))
                if isinstance(resource_data, Mapping)
                else {}
            )
            original_resource = getattr(item, "resource", None) or getattr(
                item.log_record, "resource", None
            )
            masked_resource = Resource(
                resource_attrs,
                schema_url=original_resource.schema_url if original_resource else None,
            )
            if hasattr(clone, "resource"):
                object.__setattr__(clone, "resource", masked_resource)
            elif hasattr(clone_record, "resource"):
                object.__setattr__(clone_record, "resource", masked_resource)
            kept.append(clone)
        if not kept:
            return LogRecordExportResult.SUCCESS
        return self._inner.export(kept)

    def force_flush(self, timeout_millis: int = 10000) -> bool:
        # LogRecordExporter.force_flush became abstract in OpenTelemetry 1.44.
        # NeatLogs still supports earlier SDK releases whose exporters do not
        # expose it, so absence means there is no exporter-level buffer to flush.
        force_flush = getattr(self._inner, "force_flush", None)
        if force_flush is None:
            return True
        result = force_flush(timeout_millis)
        return True if result is None else bool(result)

    def shutdown(self) -> None:
        self._runner.shutdown()
        self._inner.shutdown()
