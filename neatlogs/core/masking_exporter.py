"""Fail-closed masking at Neatlogs' final telemetry export boundary."""

from __future__ import annotations

import asyncio
import concurrent.futures
import copy
import inspect
import logging
from collections.abc import Mapping, Sequence
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

from .mask import effective_mask

logger = logging.getLogger(__name__)


def _mask_call(mask: Callable, snapshot: dict[str, Any], timeout_seconds: float):
    result = mask(snapshot)
    if inspect.isawaitable(result):
        return asyncio.run(asyncio.wait_for(result, timeout=timeout_seconds))
    return result


class _MaskRunner:
    def __init__(self, timeout_seconds: float = 5.0) -> None:
        self._timeout_seconds = timeout_seconds
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="neatlogs-mask"
        )

    def apply(self, mask: Callable, snapshot: dict[str, Any]) -> dict[str, Any] | None:
        candidate = copy.deepcopy(snapshot)
        future = self._executor.submit(_mask_call, mask, candidate, self._timeout_seconds)
        try:
            result = future.result(timeout=self._timeout_seconds)
        except Exception as exc:
            future.cancel()
            logger.error(
                "[neatlogs] mask failed; telemetry item dropped (%s)",
                type(exc).__name__,
            )
            return None
        if result is None:
            return candidate
        if not isinstance(result, Mapping):
            logger.error("[neatlogs] mask returned a non-mapping; telemetry item dropped")
            return None
        return dict(result)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)


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

    def __init__(self, inner: SpanExporter, mask: Callable | None) -> None:
        self._inner = inner
        self._global_mask = mask
        self._runner = _MaskRunner()

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        kept = []
        for span in spans:
            snapshot = _span_snapshot(span)
            mask = effective_mask(snapshot, self._global_mask)
            if mask is None:
                kept.append(span)
                continue
            masked = self._runner.apply(mask, snapshot)
            if masked is not None:
                kept.append(_masked_span(span, masked))
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

    def __init__(self, inner: LogRecordExporter, mask: Callable | None) -> None:
        self._inner = inner
        self._mask = mask
        self._runner = _MaskRunner()

    def export(self, batch: Sequence[ReadableLogRecord]) -> LogRecordExportResult:
        if self._mask is None:
            return self._inner.export(batch)
        kept = []
        for item in batch:
            masked = self._runner.apply(self._mask, _log_snapshot(item))
            if masked is None:
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
