"""Independent, context-scoped Neatlogs clients.

``neatlogs.init()`` remains the process-wide default. ``Client`` is additive:
inside ``client.activate()`` only, Neatlogs spans and structured logs use the
client's private credentials and exporters.
"""

from __future__ import annotations

import atexit
import contextlib
import math
import threading
import time
from collections.abc import Iterator
from typing import Any, Callable
from urllib.parse import urlparse

from opentelemetry import trace as otel_trace
from opentelemetry.exporter.otlp.proto.http import Compression
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import SpanLimits, TracerProvider
from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased

from ._wrap_utils import (
    _ForeignParentGuardTracer,
    _normalize_traces_endpoint,
    activate_client,
    reset_active_client,
)
from .constants import DEFAULT_INGEST_ENDPOINT, export_queue_capacity
from .core.byte_limited_exporter import ByteLimitedSpanExporter
from .core.byte_limited_log_exporter import ByteLimitedLogExporter
from .core.client_registry import register_client, unregister_client
from .core.deadline import bounded_call
from .core.delivery import (
    DeliveryDiagnostics,
    ObservableBatchLogRecordProcessor,
    ObservableBatchSpanProcessor,
)
from .core.log_exporter import NeatlogsLogFilter
from .core.masking_exporter import MaskingLogExporter, MaskingSpanExporter
from .core.span_processor import CompletionMarkerSpanProcessor, NeatlogsSpanProcessor
from .core.transport import build_otlp_session
from .errors import NeatlogsConfigurationError
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
        endpoint: str = DEFAULT_INGEST_ENDPOINT,
        tags: list[str] | None = None,
        capture_logs: bool = False,
        batch_size: int = 100,
        flush_interval: float = 5.0,
        sample_rate: float = 1.0,
        mask: Callable[[dict[str, Any]], Any] | None = None,
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
        if isinstance(sample_rate, bool) or not isinstance(sample_rate, (int, float)):
            raise NeatlogsConfigurationError("sample_rate must be a finite number from 0.0 to 1.0")
        rate = float(sample_rate)
        if not math.isfinite(rate) or rate < 0.0 or rate > 1.0:
            raise NeatlogsConfigurationError("sample_rate must be a finite number from 0.0 to 1.0")
        if tracer_provider is not None and rate != 1.0:
            raise NeatlogsConfigurationError(
                "sample_rate cannot configure a caller-owned tracer_provider; "
                "configure its sampler directly or omit tracer_provider"
            )
        if tracer_provider is not None and tracer_provider is otel_trace.get_tracer_provider():
            raise NeatlogsConfigurationError(
                "tracer_provider must be private and must not be the process-global provider"
            )

        self.workflow_name = name
        self._state = "running"
        self._shutdown_result = True
        self._shutdown_owner: int | None = None
        self._state_changed = threading.Condition(threading.RLock())
        self._owns_provider = tracer_provider is None
        self._tracers: dict[str, Any] = {}
        self._delivery_diagnostics = DeliveryDiagnostics()

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
            sampler=ParentBased(root=TraceIdRatioBased(rate)),
            span_limits=SpanLimits(max_span_attributes=10_000),
        )
        if tracer_provider is not None:
            try:
                self.tracer_provider._resource = self.tracer_provider.resource.merge(resource)
            except Exception:
                pass

        self._span_processor = NeatlogsSpanProcessor(
            mask=mask,
            emit_completion_markers=False,
            # A private provider is an isolated project pipeline. A caller-owned
            # provider may be shared with host telemetry, so ownership is marked
            # per Client tracer start instead of claiming every provider span.
            own_all_spans=self._owns_provider,
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
                compression=Compression.Gzip,
                session=build_otlp_session(),
            )
            batch_processor = ObservableBatchSpanProcessor(
                MaskingSpanExporter(
                    ByteLimitedSpanExporter(exporter, diagnostics=self._delivery_diagnostics),
                    mask,
                    diagnostics=self._delivery_diagnostics,
                ),
                max_export_batch_size=batch_size,
                max_queue_size=export_queue_capacity(batch_size),
                schedule_delay_millis=int(flush_interval * 1000),
                diagnostics=self._delivery_diagnostics,
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
                    compression=Compression.Gzip,
                    session=build_otlp_session(),
                )
                self.log_provider.add_log_record_processor(
                    NeatlogsLogFilter(
                        ObservableBatchLogRecordProcessor(
                            MaskingLogExporter(
                                ByteLimitedLogExporter(
                                    log_exporter,
                                    diagnostics=self._delivery_diagnostics,
                                ),
                                mask,
                                diagnostics=self._delivery_diagnostics,
                            ),
                            max_export_batch_size=batch_size,
                            max_queue_size=export_queue_capacity(batch_size),
                            schedule_delay_millis=int(flush_interval * 1000),
                            diagnostics=self._delivery_diagnostics,
                        )
                    )
                )

        atexit.register(self._atexit_shutdown)
        register_client(self)

    def get_tracer(self, scope: str):
        tracer = self._tracers.get(scope)
        if tracer is None:
            guarded = _ForeignParentGuardTracer(
                self.tracer_provider.get_tracer(scope),
                context_transform=self._span_processor.owned_span_context,
            )
            tracer = _ClientLifecycleTracer(self, guarded)
            self._tracers[scope] = tracer
        return tracer

    def _is_running(self) -> bool:
        with self._state_changed:
            return self._state == "running"

    def get_delivery_diagnostics(self) -> dict[str, Any]:
        return self._delivery_diagnostics.snapshot()

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
        for processor in self._transport_processors:
            try:
                result = processor.force_flush(timeout_millis=timeout_millis)
                success = (result is None or bool(result)) and success
            except Exception:
                success = False
        return success

    def shutdown(
        self,
        timeout_millis: int = 30000,
        termination_reason: str = "shutdown",
        *,
        _synchronous: bool = False,
    ) -> bool:
        with self._state_changed:
            if self._state == "closed":
                return self._shutdown_result
            if self._state == "closing":
                if self._shutdown_owner == threading.get_ident():
                    return self._shutdown_result
                completed = self._state_changed.wait_for(
                    lambda: self._state == "closed",
                    timeout=max(0, timeout_millis) / 1000,
                )
                return self._shutdown_result if completed else False
            self._state = "closing"
            self._shutdown_owner = threading.get_ident()

        if self._completion_processor is not None:
            self._completion_processor.begin_shutdown()
        self._span_processor.begin_shutdown(termination_reason)
        success = False
        try:
            success = self._perform_shutdown(
                timeout_millis,
                termination_reason,
                synchronous=_synchronous,
            )
            return success
        finally:
            with self._state_changed:
                self._shutdown_result = success
                self._state = "closed"
                self._shutdown_owner = None
                self._state_changed.notify_all()
            try:
                atexit.unregister(self._atexit_shutdown)
            except Exception:
                pass
            unregister_client(self)

    def _atexit_shutdown(self) -> None:
        """Synchronous interpreter-exit cleanup (Python 3.12 forbids new threads)."""

        self.shutdown(_synchronous=True)

    def _perform_shutdown(
        self,
        timeout_millis: int,
        termination_reason: str,
        *,
        synchronous: bool = False,
    ) -> bool:
        success = True
        deadline = time.monotonic() + max(0, timeout_millis) / 1000
        # Drain logs before ending roots: root end creates the completion marker.
        if self.log_provider is not None:
            completed, _ = bounded_call(
                self.log_provider.shutdown,
                deadline,
                synchronous=synchronous,
            )
            success = completed and success
        try:
            self._span_processor.end_active_spans(termination_reason)
        except Exception:
            success = False
        remaining_millis = max(0, int((deadline - time.monotonic()) * 1000))
        if not self._span_processor.wait_for_downstream(remaining_millis):
            success = False
        if self._completion_processor is not None:
            try:
                self._completion_processor.emit_deferred()
            except Exception:
                success = False
        if self._owns_provider:
            completed, _ = bounded_call(
                self.tracer_provider.shutdown,
                deadline,
                synchronous=synchronous,
            )
            success = completed and success
        else:
            completed, result = bounded_call(
                lambda: self.tracer_provider.force_flush(
                    timeout_millis=max(0, int((deadline - time.monotonic()) * 1000))
                ),
                deadline,
                synchronous=synchronous,
            )
            success = completed and (result is None or bool(result)) and success
            for processor in reversed(self._transport_processors):
                completed, _ = bounded_call(
                    processor.shutdown,
                    deadline,
                    synchronous=synchronous,
                )
                success = completed and success
            completed, _ = bounded_call(
                self._span_processor.shutdown,
                deadline,
                synchronous=synchronous,
            )
            success = completed and success

        return success

    close = shutdown
