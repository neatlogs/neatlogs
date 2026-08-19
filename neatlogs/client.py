"""Independent, context-scoped Neatlogs clients.

``neatlogs.init()`` remains the process-wide default. ``Client`` is additive:
inside ``client.activate()`` only, Neatlogs spans and structured logs use the
client's private credentials and exporters.
"""

from __future__ import annotations

import atexit
import contextlib
import threading
from collections.abc import Iterator
from typing import Any
from urllib.parse import urlparse

from opentelemetry import trace as otel_trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import SpanLimits, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from ._wrap_utils import (
    _ForeignParentGuardTracer,
    _normalize_traces_endpoint,
    activate_client,
    reset_active_client,
)
from .core.log_exporter import NeatlogsLogFilter
from .core.span_processor import CompletionMarkerSpanProcessor, NeatlogsSpanProcessor
from .version import __version__


class _ClientLifecycleTracer:
    """Prevent cached tracers from starting spans after Client shutdown starts."""

    def __init__(self, client: "Client", tracer: Any) -> None:
        self._client = client
        self._tracer = tracer
        self._noop = otel_trace.NoOpTracerProvider().get_tracer("neatlogs.client.closed")

    def _current(self):
        return self._tracer if self._client._is_running() else self._noop

    def start_span(self, *args: Any, **kwargs: Any):
        return self._current().start_span(*args, **kwargs)

    def start_as_current_span(self, *args: Any, **kwargs: Any):
        return self._current().start_as_current_span(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._current(), name)


class Client:
    """An independent Neatlogs export pipeline activated with a ContextVar."""

    def __init__(
        self,
        *,
        api_key: str,
        workflow_name: str,
        endpoint: str = "https://ingest.neatlogs.com",
        tags: list[str] | None = None,
        capture_logs: bool = False,
        batch_size: int = 100,
        flush_interval: float = 5.0,
        disable_export: bool = False,
        tracer_provider: TracerProvider | None = None,
    ) -> None:
        key = str(api_key or "").strip()
        name = str(workflow_name or "").strip()
        if not key and not disable_export:
            raise ValueError("api_key is required unless disable_export=True")
        if not name:
            raise ValueError("workflow_name is required")
        if tags is not None and not all(isinstance(tag, str) for tag in tags):
            raise ValueError("All tags must be strings")

        self.workflow_name = name
        self._state = "running"
        self._shutdown_result = True
        self._shutdown_owner: int | None = None
        self._state_changed = threading.Condition(threading.RLock())
        self._owns_provider = tracer_provider is None
        self._tracers: dict[str, Any] = {}

        resource_attrs: dict[str, Any] = {
            SERVICE_NAME: name,
            "service.version": __version__,
            "neatlogs.workflow_name": name,
        }
        if tags:
            resource_attrs["neatlogs.tags"] = ",".join(tags)
        resource = Resource.create(resource_attrs)

        self.tracer_provider = tracer_provider or TracerProvider(
            resource=resource,
            span_limits=SpanLimits(max_span_attributes=10_000),
        )
        if tracer_provider is not None:
            try:
                self.tracer_provider._resource = self.tracer_provider.resource.merge(resource)
            except Exception:
                pass

        self._span_processor = NeatlogsSpanProcessor(
            emit_completion_markers=False,
            # Client activation is an isolated project pipeline. Caller-owned
            # provider shutdown semantics do not change span ownership here.
            own_all_spans=True,
        )
        self.tracer_provider.add_span_processor(self._span_processor)
        self._transport_processors: list[Any] = []
        self._completion_processor: CompletionMarkerSpanProcessor | None = None

        traces_endpoint = _normalize_traces_endpoint(endpoint)
        parsed = urlparse(traces_endpoint)
        base_url = f"{parsed.scheme}://{parsed.netloc}"
        if not disable_export:
            exporter = OTLPSpanExporter(
                endpoint=traces_endpoint,
                headers={"x-api-key": key},
            )
            batch_processor = BatchSpanProcessor(
                exporter,
                max_export_batch_size=batch_size,
                schedule_delay_millis=int(flush_interval * 1000),
            )
            completion_processor = CompletionMarkerSpanProcessor(
                self._span_processor,
                self.tracer_provider.get_tracer("neatlogs.internal"),
            )
            self.tracer_provider.add_span_processor(batch_processor)
            self.tracer_provider.add_span_processor(completion_processor)
            self._transport_processors = [batch_processor, completion_processor]
            self._completion_processor = completion_processor

        self.log_provider: LoggerProvider | None = None
        if capture_logs:
            self.log_provider = LoggerProvider(resource=resource)
            if not disable_export:
                from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter

                log_exporter = OTLPLogExporter(
                    endpoint=f"{base_url}/v1/logs",
                    headers={"x-api-key": key},
                )
                self.log_provider.add_log_record_processor(
                    NeatlogsLogFilter(BatchLogRecordProcessor(log_exporter))
                )

        atexit.register(self.shutdown)

    def get_tracer(self, scope: str):
        tracer = self._tracers.get(scope)
        if tracer is None:
            guarded = _ForeignParentGuardTracer(self.tracer_provider.get_tracer(scope))
            tracer = _ClientLifecycleTracer(self, guarded)
            self._tracers[scope] = tracer
        return tracer

    def _is_running(self) -> bool:
        with self._state_changed:
            return self._state == "running"

    @contextlib.contextmanager
    def activate(self) -> Iterator[Client]:
        with self._state_changed:
            if self._state != "running":
                raise RuntimeError("Client is closing or closed")
        token = activate_client(self)
        try:
            yield self
        finally:
            reset_active_client(token)

    def wrap(self, target: Any, **workflow_attributes: Any) -> Any:
        """Patch/wrap a supported integration for context-time client routing."""
        import neatlogs

        with self.activate():
            return neatlogs.wrap(target, **workflow_attributes)

    def flush(self, timeout_millis: int = 30000) -> bool:
        success = True
        if self.log_provider is not None:
            try:
                self.log_provider.force_flush(timeout_millis=timeout_millis)
            except Exception:
                success = False
        try:
            result = self.tracer_provider.force_flush(timeout_millis=timeout_millis)
            success = bool(result) and success
        except Exception:
            success = False
        return success

    def shutdown(
        self,
        timeout_millis: int = 30000,
        termination_reason: str = "shutdown",
    ) -> bool:
        with self._state_changed:
            if self._state == "closed":
                return self._shutdown_result
            if self._state == "closing":
                if self._shutdown_owner == threading.get_ident():
                    return self._shutdown_result
                self._state_changed.wait_for(lambda: self._state == "closed")
                return self._shutdown_result
            self._state = "closing"
            self._shutdown_owner = threading.get_ident()

        if self._completion_processor is not None:
            self._completion_processor.begin_shutdown()
        self._span_processor.begin_shutdown(termination_reason)
        success = False
        try:
            success = self._perform_shutdown(timeout_millis, termination_reason)
            return success
        finally:
            with self._state_changed:
                self._shutdown_result = success
                self._state = "closed"
                self._shutdown_owner = None
                self._state_changed.notify_all()
            try:
                atexit.unregister(self.shutdown)
            except Exception:
                pass

    def _perform_shutdown(self, timeout_millis: int, termination_reason: str) -> bool:
        success = True
        # Drain logs before ending roots: root end creates the completion marker.
        if self.log_provider is not None:
            try:
                self.log_provider.shutdown()
            except Exception:
                success = False
        try:
            self._span_processor.end_active_spans(termination_reason)
        except Exception:
            success = False
        if not self._span_processor.wait_for_downstream(timeout_millis):
            success = False
        if self._completion_processor is not None:
            try:
                self._completion_processor.emit_deferred()
            except Exception:
                success = False
        if self._owns_provider:
            try:
                self.tracer_provider.shutdown()
            except Exception:
                success = False
        else:
            try:
                result = self.tracer_provider.force_flush(timeout_millis=timeout_millis)
                success = bool(result) and success
            except Exception:
                success = False
            for processor in reversed(self._transport_processors):
                try:
                    processor.shutdown()
                except Exception:
                    success = False
            try:
                self._span_processor.shutdown()
            except Exception:
                success = False

        return success

    close = shutdown
