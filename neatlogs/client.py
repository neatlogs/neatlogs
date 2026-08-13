"""Independent, context-scoped Neatlogs clients.

``neatlogs.init()`` remains the process-wide default. ``Client`` is additive:
inside ``client.activate()`` only, Neatlogs spans and structured logs use the
client's private credentials and exporters.
"""

from __future__ import annotations

import atexit
import contextlib
from collections.abc import Iterator
from typing import Any
from urllib.parse import urlparse

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
from .core.span_processor import NeatlogsSpanProcessor
from .version import __version__


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
        self._closed = False
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

        self._span_processor = NeatlogsSpanProcessor()
        self.tracer_provider.add_span_processor(self._span_processor)

        traces_endpoint = _normalize_traces_endpoint(endpoint)
        parsed = urlparse(traces_endpoint)
        base_url = f"{parsed.scheme}://{parsed.netloc}"
        if not disable_export:
            exporter = OTLPSpanExporter(
                endpoint=traces_endpoint,
                headers={"x-api-key": key},
            )
            self.tracer_provider.add_span_processor(
                BatchSpanProcessor(
                    exporter,
                    max_export_batch_size=batch_size,
                    schedule_delay_millis=int(flush_interval * 1000),
                )
            )

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
            tracer = _ForeignParentGuardTracer(self.tracer_provider.get_tracer(scope))
            self._tracers[scope] = tracer
        return tracer

    @contextlib.contextmanager
    def activate(self) -> Iterator[Client]:
        if self._closed:
            raise RuntimeError("Client is closed")
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

    def shutdown(self, timeout_millis: int = 30000) -> bool:
        if self._closed:
            return True
        success = self.flush(timeout_millis=timeout_millis)
        if self.log_provider is not None:
            try:
                self.log_provider.shutdown()
            except Exception:
                success = False
        if self._owns_provider:
            try:
                self.tracer_provider.shutdown()
            except Exception:
                success = False
        self._closed = True
        try:
            atexit.unregister(self.shutdown)
        except Exception:
            pass
        return success

    close = shutdown
