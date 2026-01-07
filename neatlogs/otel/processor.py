"""
Neatlogs Span Processor
=======================

Captures OpenTelemetry spans, serializes them safely,
derives a stable externalTraceId, and sends them to Neatlogs backend.
"""

import json
import logging
import traceback
from datetime import datetime
from typing import Optional

from opentelemetry.sdk.trace import SpanProcessor, ReadableSpan

from ..core import LLMTracker, LLMCallData

logger = logging.getLogger(__name__)


def _hex(n: Optional[int], width: int) -> Optional[str]:
    if n is None:
        return None
    return format(n, f"0{width}x")


def serialize_span(span: ReadableSpan) -> dict:
    """
    Convert ReadableSpan -> JSON-serializable dict
    (matches what server v3 expects)
    """
    ctx = span.context

    attributes = {}
    for k, v in (span.attributes or {}).items():
        if isinstance(v, (str, int, float, bool, list, dict)) or v is None:
            attributes[k] = v
        else:
            attributes[k] = str(v)

    resource_attrs = {}
    if span.resource:
        for k, v in (span.resource.attributes or {}).items():
            if isinstance(v, (str, int, float, bool, list, dict)) or v is None:
                resource_attrs[k] = v
            else:
                resource_attrs[k] = str(v)

    events = []
    for event in span.events:
        evt_attrs = {}
        for k, v in (event._attributes or {}).items():
            if isinstance(v, (str, int, float, bool, list, dict)) or v is None:
                evt_attrs[k] = v
            else:
                evt_attrs[k] = str(v)

        events.append({
            "name": event.name,
            "timestamp": event.timestamp,
            "attributes": evt_attrs
        })

    return {
        "name": span.name,
        "context": {
            "trace_id": _hex(ctx.trace_id, 32),
            "span_id": _hex(ctx.span_id, 16),
        },
        "parent_id": _hex(span.parent.span_id, 16) if span.parent else None,
        "start_time": span.start_time,
        "end_time": span.end_time,
        "attributes": attributes,
        "resource": {
            "attributes": resource_attrs
        },
        "events": events,
        "status": {
            "status_code": span.status.status_code.name
            if span.status else "UNSET"
        },
    }


def choose_external_trace_id(serialized_span: dict) -> str:
    """
    Canonical trace grouping rule:
    1. neatlogs.session_id
    2. neatlogs.thread_id
    3. OTEL trace_id (fallback)
    """
    res_attrs = serialized_span.get("resource", {}).get("attributes", {})

    session_id = res_attrs.get("neatlogs.session_id")
    if session_id:
        return str(session_id)

    thread_id = res_attrs.get("neatlogs.thread_id")
    if thread_id:
        return str(thread_id)

    trace_id = serialized_span.get("context", {}).get("trace_id")
    if trace_id:
        return trace_id

    # absolute fallback (should almost never happen)
    from uuid import uuid4
    return str(uuid4())


class NeatlogsSpanProcessor(SpanProcessor):
    def __init__(self, tracker: LLMTracker):
        self.tracker = tracker

    def on_start(self, span, parent_context=None):
        pass

    def on_end(self, span: ReadableSpan) -> None:
        if not span:
            return

        try:
            attributes = span.attributes or {}
            span_kind = attributes.get("openinference.span.kind")

            # Ignore infra spans
            if not span_kind:
                return

            serialized = serialize_span(span)

            # Add tags to the resource attributes of the span
            with self.tracker._lock:
                if self.tracker.tags:
                    if "attributes" not in serialized.get("resource", {}):
                        if "resource" not in serialized:
                            serialized["resource"] = {}
                        serialized["resource"]["attributes"] = {}
                    serialized["resource"]["attributes"]["neatlogs.tags"] = self.tracker.tags

            external_trace_id = choose_external_trace_id(serialized)

            call_data = LLMCallData(
                span=serialized,
                trace_id=external_trace_id,
                api_key=self.tracker.api_key,
            )

            if self.tracker.enable_server_sending and not self.tracker.dry_run:
                self.tracker._enqueue_span(call_data)
            elif self.tracker.dry_run:
                logger.info(
                    f"[Dry Run] Captured span for trace {external_trace_id}"
                )

        except Exception as e:
            logger.error(f"Neatlogs: Failed to process span: {e}")
            logger.debug(traceback.format_exc())

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True
