"""Context-local membership recording on the existing Neatlogs provider."""

from __future__ import annotations

import importlib
import json
import threading
import weakref
from collections.abc import Mapping, Sequence
from contextvars import ContextVar

from opentelemetry.sdk.trace import SpanProcessor
from opentelemetry.sdk.trace.sampling import ALWAYS_ON, ParentBased, TraceIdRatioBased

from .._wrap_utils import get_active_client, get_neatlogs_provider
from ..core.span_processor import is_http_span

current_recording: ContextVar[Recording | None] = ContextVar(
    "neatlogs.dataset_capture", default=None
)
_lock = threading.Lock()
_processors = weakref.WeakKeyDictionary()
_owners = weakref.WeakKeyDictionary()


def pipeline():
    client = get_active_client()
    provider = get_neatlogs_provider()
    if provider is None:
        raise ValueError("Initialize Neatlogs before dataset capture")
    limits = provider._span_limits
    if any(
        getattr(limits, field, None) is not None
        for field in ("max_attribute_length", "max_span_attribute_length")
    ):
        raise ValueError(
            "Dataset capture requires OpenTelemetry attribute-value truncation disabled"
        )
    sampler = provider.sampler
    if isinstance(sampler, ParentBased):
        if (
            sampler._local_parent_sampled != ALWAYS_ON
            or sampler._remote_parent_sampled != ALWAYS_ON
        ):
            raise ValueError("Dataset capture requires fully sampled child spans")
        sampler = sampler._root
    if sampler != ALWAYS_ON and not (
        isinstance(sampler, TraceIdRatioBased) and sampler.rate == 1.0
    ):
        raise ValueError("Dataset capture requires a fully sampled Neatlogs provider")
    if client is not None:
        if not client._is_running():
            raise ValueError("Dataset capture requires a running Client")
        transports = client._transport_processors
        owner = client._span_processor
        flush = client.flush
    else:
        default = importlib.import_module("neatlogs.init")
        transports = default._transport_span_processors
        owner = default._span_processor
        flush = default.flush
    if not transports:
        raise ValueError("Dataset capture requires normal Neatlogs export to be enabled")
    with _lock:
        processor = _processors.get(provider)
        if processor is None:
            processor = CaptureProcessor(owner)
            provider.add_span_processor(processor)
            _processors[provider] = processor
            _owners[owner] = processor
    return processor, flush


class Recording:
    def __init__(self, processor):
        self.processor = processor
        self.lock = threading.RLock()
        self.roots: dict[int, int] = {}
        self.spans: dict[int, set[int]] = {}
        self.active: set[tuple[int, int]] = set()
        self.accepted: dict[int, set[int]] = {}
        self.completed: set[int] = set()
        self.filtered: set[tuple[int, int]] = set()
        self.export_error = False
        self.closed = False
        self.error: str | None = None
        self.started_count = 0

    def start(self, span) -> bool:
        with self.lock:
            if self.closed or self.error or span.name == "neatlogs.trace.complete":
                return False
            trace_id, span_id = span.context.trace_id, span.context.span_id
            if span.parent is None or not span.parent.is_valid:
                if is_http_span(span):
                    return False
                if trace_id in self.roots:
                    self.error = "A captured trace has more than one root"
                    return False
                self.roots[trace_id] = span_id
                self.spans[trace_id] = set()
                self.accepted[trace_id] = set()
            if trace_id not in self.roots:
                return False
            # Bound memory even before the serialized manifest size check.
            if len(self.roots) > 1000 or self.started_count >= 3000:
                self.error = "Capture exceeds its manifest limit; use a smaller block"
                return False
            self.started_count += 1
            self.active.add((trace_id, span_id))
            return True

    def end(self, span):
        with self.lock:
            trace_id, span_id = span.context.trace_id, span.context.span_id
            self.active.discard((trace_id, span_id))
            if any(
                getattr(span, field, 0)
                for field in ("dropped_attributes", "dropped_events", "dropped_links")
            ) or any(
                getattr(item, "dropped_attributes", 0) for item in (*span.events, *span.links)
            ):
                self.error = "Captured telemetry exceeded OpenTelemetry span limits"
            if not span.context.trace_flags.sampled:
                self.error = "A captured span was not sampled"
            elif (
                not is_http_span(span)
                and span.name != "neatlogs.trace.complete"
                and (trace_id, span_id) not in self.filtered
            ):
                self.spans[trace_id].add(span_id)

    def freeze(self):
        with self.lock:
            self.closed = True
            if self.active:
                raise ValueError(
                    "Capture has unfinished spans; await all work before leaving the block"
                )
            if self.error:
                raise ValueError(self.error)
            traces = []
            for trace_id, root_id in self.roots.items():
                spans = self.spans[trace_id]
                if root_id not in spans:
                    raise ValueError("A captured root was suppressed or not exported")
                traces.append(
                    {"traceId": f"{trace_id:032x}", "spanIds": [f"{s:016x}" for s in sorted(spans)]}
                )
            return {"traces": traces}

    def exported(self, span, accepted):
        with self.lock:
            trace_id, span_id = span.context.trace_id, span.context.span_id
            if span.name == "neatlogs.trace.complete":
                if accepted:
                    self.completed.add(trace_id)
                else:
                    self.export_error = True
            elif is_http_span(span):
                self.filtered.add((trace_id, span_id))
                self.spans[trace_id].discard(span_id)
            elif accepted:
                if _incomplete_value(dict(span.attributes or {})) or any(
                    _incomplete_value(dict(event.attributes or {})) for event in span.events
                ):
                    self.error = "Captured telemetry was truncated or contains unavailable media"
                self.accepted[trace_id].add(span_id)
            else:
                self.export_error = True

    def exported_manifest(self):
        with self.lock:
            if (
                self.error
                or self.export_error
                or self.completed != self.roots.keys()
                or any(self.spans[t] != self.accepted[t] for t in self.roots)
            ):
                raise ValueError(
                    self.error or "Captured spans were not acknowledged by the exporter"
                )
            return self.freeze()


class CaptureProcessor(SpanProcessor):
    def __init__(self, owner):
        self.owner = weakref.proxy(owner)
        self.lock = threading.Lock()
        self.pending: dict[tuple[int, int], Recording] = {}
        self.traces: dict[int, Recording] = {}

    def on_start(self, span, parent_context=None):
        with self.lock:
            recording = self.traces.get(span.context.trace_id)
        if recording is None:
            recording = current_recording.get()
        if (
            recording is not None
            and recording.processor is self
            and self.owner.owns_span(span)
            and recording.start(span)
        ):
            with self.lock:
                self.pending[(span.context.trace_id, span.context.span_id)] = recording
                self.traces[span.context.trace_id] = recording

    def on_end(self, span):
        with self.lock:
            recording = self.pending.get((span.context.trace_id, span.context.span_id))
        if recording is not None:
            recording.end(span)

    def exported(self, span, accepted):
        with self.lock:
            recording = (
                self.traces.get(span.context.trace_id)
                if span.name == "neatlogs.trace.complete"
                else self.pending.get((span.context.trace_id, span.context.span_id))
            )
        if recording is not None:
            recording.exported(span, accepted)

    def release(self, recording):
        with self.lock:
            self.pending = {
                key: value for key, value in self.pending.items() if value is not recording
            }
            self.traces = {
                key: value for key, value in self.traces.items() if value is not recording
            }

    def shutdown(self):
        return None

    def force_flush(self, timeout_millis=30000):
        return True


def notify_export(owner, span, *, accepted):
    """Observe the existing final post-mask export receipt without changing transport."""
    with _lock:
        processor = _owners.get(owner)
    if processor is not None:
        processor.exported(span, accepted)


def _incomplete_value(value, depth=0):
    if depth > 64:
        return True
    if isinstance(value, Mapping):
        if value.get("state") in ("failed", "pending-upload", "unavailable") and (
            value.get("type") in ("image", "audio", "video", "document", "media")
            or str(value.get("id", "")).startswith("nl_media_")
        ):
            return True
        if any(
            str(key).endswith(".state")
            and item in ("failed", "pending-upload", "unavailable")
            and ".media." in str(key)
            for key, item in value.items()
        ):
            return True
        return any(_incomplete_value(item, depth + 1) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return any(_incomplete_value(item, depth + 1) for item in value)
    if isinstance(value, str):
        if "...[neatlogs-truncated" in value:
            return True
        if value.lstrip().startswith(("{", "[")):
            try:
                return _incomplete_value(json.loads(value), depth + 1)
            except (ValueError, RecursionError):
                pass
    return False
