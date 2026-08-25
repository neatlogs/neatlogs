"""Fail-closed masking at Neatlogs' final telemetry export boundary."""

from __future__ import annotations

import asyncio
import concurrent.futures
import copy
import inspect
import logging
import threading
from collections.abc import Mapping, Sequence
from typing import Any, Callable

from opentelemetry.sdk._logs import ReadableLogRecord
from opentelemetry.sdk._logs.export import LogRecordExporter, LogRecordExportResult
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import Event, ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.trace import Link, Status, StatusCode

from .mask import effective_mask

logger = logging.getLogger(__name__)


def _invoke(mask: Callable, snapshot: dict[str, Any], timeout: float):
    result = mask(snapshot)
    if inspect.isawaitable(result):
        return asyncio.run(asyncio.wait_for(result, timeout=timeout))
    return result


class _MaskRunner:
    def __init__(self, timeout_seconds: float) -> None:
        self.timeout_seconds = timeout_seconds
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="neatlogs-mask"
        )

    def apply(self, mask: Callable, snapshot: dict[str, Any]) -> dict[str, Any] | None:
        candidate = copy.deepcopy(snapshot)
        future = self._executor.submit(_invoke, mask, candidate, self.timeout_seconds)
        try:
            result = future.result(timeout=self.timeout_seconds)
        except Exception as exc:
            future.cancel()
            logger.error("[neatlogs] mask failed; telemetry item dropped (%s)", type(exc).__name__)
            return None
        # None is an explicit privacy-preserving drop. In-place callbacks must
        # return their candidate; silently exporting on None is not fail closed.
        if result is None:
            return None
        if not isinstance(result, Mapping):
            logger.error("[neatlogs] mask returned non-mapping; telemetry item dropped")
            return None
        return dict(result)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)


class _Health:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.failures = 0
        self.drops = 0

    def fail(self, *, dropped: bool = False) -> None:
        with self._lock:
            self.failures += 1
            if dropped:
                self.drops += 1

    @property
    def healthy(self) -> bool:
        with self._lock:
            return self.failures == 0


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
        "status": {"code": span.status.status_code.name, "description": span.status.description},
    }


def _masked_span(span: ReadableSpan, snapshot: Mapping[str, Any]) -> ReadableSpan:
    events = [
        Event(
            str(item.get("name") or "event"),
            attributes=_attrs(item.get("attributes")),
            timestamp=item.get("timestamp"),
        )
        for item in snapshot.get("events", ())
        if isinstance(item, Mapping)
    ]
    links = [
        Link(span.links[index].context, attributes=_attrs(item.get("attributes")))
        for index, item in enumerate(snapshot.get("links", ()))
        if index < len(span.links) and isinstance(item, Mapping)
    ]
    resource_data = snapshot.get("resource")
    resource_attrs = (
        _attrs(resource_data.get("attributes")) if isinstance(resource_data, Mapping) else {}
    )
    status_data = snapshot.get("status")
    status = span.status
    if isinstance(status_data, Mapping):
        try:
            code = StatusCode[str(status_data.get("code") or span.status.status_code.name)]
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
    """Clone and mask spans without exposing the shared ReadableSpan to callbacks."""

    def __init__(
        self, inner: SpanExporter, mask: Callable | None, timeout_seconds: float = 5.0
    ) -> None:
        self._inner = inner
        self._global_mask = mask
        self._runner = _MaskRunner(timeout_seconds)
        self.health = _Health()

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        kept = []
        for span in spans:
            snapshot = _span_snapshot(span)
            mask = effective_mask(snapshot, self._global_mask)
            if mask is None:
                kept.append(span)
                continue
            masked = self._runner.apply(mask, snapshot)
            if masked is None:
                self.health.fail(dropped=True)
            else:
                kept.append(_masked_span(span, masked))
        if not kept:
            return SpanExportResult.FAILURE if spans else SpanExportResult.SUCCESS
        try:
            result = self._inner.export(kept)
        except Exception:
            self.health.fail()
            raise
        if result is not SpanExportResult.SUCCESS:
            self.health.fail()
        return result

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        result = self._inner.force_flush(timeout_millis)
        return (result is None or bool(result)) and self.health.healthy

    def shutdown(self) -> None:
        self._runner.shutdown()
        self._inner.shutdown()


def _log_snapshot(item: ReadableLogRecord) -> dict[str, Any]:
    record = item.log_record
    return {
        "signal": "log",
        "name": item.instrumentation_scope.name if item.instrumentation_scope else "log",
        "body": record.body,
        "attributes": dict(record.attributes or {}),
        "resource": {"attributes": dict(item.resource.attributes or {})},
        "severity_text": record.severity_text,
        "event_name": record.event_name,
    }


class MaskingLogExporter(LogRecordExporter):
    def __init__(
        self, inner: LogRecordExporter, mask: Callable | None, timeout_seconds: float = 5.0
    ) -> None:
        self._inner = inner
        self._mask = mask
        self._runner = _MaskRunner(timeout_seconds)
        self.health = _Health()

    def export(self, batch: Sequence[ReadableLogRecord]) -> LogRecordExportResult:
        if self._mask is None:
            try:
                result = self._inner.export(batch)
            except Exception:
                self.health.fail()
                raise
            if result is not LogRecordExportResult.SUCCESS:
                self.health.fail()
            return result
        kept = []
        for item in batch:
            masked = self._runner.apply(self._mask, _log_snapshot(item))
            if masked is None:
                self.health.fail(dropped=True)
                continue
            clone = copy.copy(item)
            record = copy.copy(item.log_record)
            record.body = masked.get("body")
            record.attributes = _attrs(masked.get("attributes"))
            record.severity_text = masked.get("severity_text")
            record.event_name = masked.get("event_name")
            object.__setattr__(clone, "log_record", record)
            resource_data = masked.get("resource")
            resource_attrs = (
                _attrs(resource_data.get("attributes"))
                if isinstance(resource_data, Mapping)
                else {}
            )
            object.__setattr__(
                clone, "resource", Resource(resource_attrs, schema_url=item.resource.schema_url)
            )
            kept.append(clone)
        if not kept:
            return LogRecordExportResult.FAILURE if batch else LogRecordExportResult.SUCCESS
        try:
            result = self._inner.export(kept)
        except Exception:
            self.health.fail()
            raise
        if result is not LogRecordExportResult.SUCCESS:
            self.health.fail()
        return result

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        method = getattr(self._inner, "force_flush", None)
        result = method(timeout_millis) if method else True
        return (result is None or bool(result)) and self.health.healthy

    def shutdown(self) -> None:
        self._runner.shutdown()
        self._inner.shutdown()
