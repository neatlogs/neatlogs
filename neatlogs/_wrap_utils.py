"""
Shared infrastructure for Neatlogs provider wrappers.

Only contains truly shared concerns:
  - TracerProvider bootstrap (auto from env or reuse from init())
  - configure() for wrapper-only mode
  - Sync/async stream wrapper classes
  - Safe JSON serialization
"""

import json
import os
import time
from typing import Any, Callable, Dict, List, Optional

from opentelemetry import context as context_api
from opentelemetry import trace as otel_trace
from opentelemetry.sdk.trace import TracerProvider

from .core.logger import get_logger

logger = get_logger()

_wrapper_tracer: Optional[otel_trace.Tracer] = None
_wrapper_bootstrapped = False
_bootstrap_warned = False

_wrapper_config: Dict[str, Any] = {}


def configure(**kwargs: Any) -> None:
    """
    Optional configuration for wrapper-only mode (no neatlogs.init() needed).

    Args:
        workflow_name: Logical grouping for traces
        session_id: Session identifier
        endpoint: Backend URL (default: https://cloud.neatlogs.com)
        api_key: Project write key (or set NEATLOGS_API_KEY env var)
    """
    _wrapper_config.update(kwargs)
    global _wrapper_tracer
    _wrapper_tracer = None


def get_tracer() -> otel_trace.Tracer:
    """
    Return a Tracer from init()'s provider, or auto-bootstrap from env.
    """
    global _wrapper_tracer, _wrapper_bootstrapped, _bootstrap_warned

    if _wrapper_tracer is not None:
        return _wrapper_tracer

    provider = otel_trace.get_tracer_provider()
    if isinstance(provider, TracerProvider):
        _wrapper_tracer = provider.get_tracer("neatlogs.wrapper")
        return _wrapper_tracer

    api_key = _wrapper_config.get("api_key") or os.environ.get("NEATLOGS_API_KEY", "")
    if not api_key:
        if not _bootstrap_warned:
            _bootstrap_warned = True
            logger.warning(
                "neatlogs wrapper: no TracerProvider configured and NEATLOGS_API_KEY not set. "
                "Spans will not be exported. Call neatlogs.init() or set NEATLOGS_API_KEY."
            )
        _wrapper_tracer = otel_trace.get_tracer("neatlogs.wrapper.noop")
        return _wrapper_tracer

    if not _wrapper_bootstrapped:
        _wrapper_bootstrapped = True
        _bootstrap_from_env(api_key)

    _wrapper_tracer = otel_trace.get_tracer("neatlogs.wrapper")
    return _wrapper_tracer


def _bootstrap_from_env(api_key: str) -> None:
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import SERVICE_NAME, Resource
    from opentelemetry.sdk.trace import SpanLimits
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    endpoint = (
        _wrapper_config.get("endpoint")
        or os.environ.get("NEATLOGS_ENDPOINT", "https://cloud.neatlogs.com")
    )
    if not endpoint.endswith("/v1/traces"):
        endpoint = f"{endpoint.rstrip('/')}/v1/traces"

    workflow_name = _wrapper_config.get("workflow_name") or "neatlogs-app"

    resource_attrs: Dict[str, Any] = {
        SERVICE_NAME: workflow_name,
        "neatlogs.workflow_name": workflow_name,
    }
    session_id = _wrapper_config.get("session_id")
    if session_id:
        resource_attrs["session.id"] = session_id

    resource = Resource.create(resource_attrs)
    provider = TracerProvider(
        resource=resource,
        span_limits=SpanLimits(max_span_attributes=10_000),
    )
    exporter = OTLPSpanExporter(
        endpoint=endpoint,
        headers={"x-api-key": api_key},
    )
    provider.add_span_processor(BatchSpanProcessor(exporter))
    otel_trace.set_tracer_provider(provider)
    logger.debug(f"neatlogs wrapper: auto-bootstrapped TracerProvider → {endpoint}")


def is_suppressed() -> bool:
    """Check if a framework instrumentor already covers this call."""
    try:
        return bool(context_api.get_value("suppress_instrumentation"))
    except Exception:
        return False


def serialize(obj: Any, max_length: int = 100_000) -> str:
    """Safe JSON serialization with truncation."""
    try:
        s = json.dumps(obj, default=str, ensure_ascii=False)
    except (TypeError, ValueError):
        s = str(obj)
    if len(s) > max_length:
        return s[:max_length] + "...[truncated]"
    return s


class SyncStreamWrapper:
    """
    Wraps a sync streaming response. Transparently passes through chunks
    while recording timestamps. Calls finalizer on stream exhaustion.
    """

    def __init__(self, stream: Any, span: otel_trace.Span, finalizer: Callable):
        self._stream = stream
        self._span = span
        self._finalizer = finalizer
        self._start_time = time.perf_counter()
        self._first_chunk_time: Optional[float] = None
        self._chunks: List[Any] = []
        self._finalized = False

    def __iter__(self):
        return self

    def __next__(self):
        try:
            chunk = next(self._stream)
        except StopIteration:
            self._finalize()
            raise
        except Exception as e:
            self._finalize_error(e)
            raise

        if self._first_chunk_time is None:
            self._first_chunk_time = time.perf_counter()
        self._chunks.append(chunk)
        return chunk

    def __enter__(self):
        if hasattr(self._stream, "__enter__"):
            self._stream.__enter__()
        return self

    def __exit__(self, *args):
        if hasattr(self._stream, "__exit__"):
            self._stream.__exit__(*args)
        self._finalize()

    def _finalize(self):
        if self._finalized:
            return
        self._finalized = True
        elapsed_ms = (time.perf_counter() - self._start_time) * 1000
        ttft_ms = None
        if self._first_chunk_time is not None:
            ttft_ms = (self._first_chunk_time - self._start_time) * 1000
        self._finalizer(self._span, self._chunks, elapsed_ms, ttft_ms)

    def _finalize_error(self, error: Exception):
        if self._finalized:
            return
        self._finalized = True
        from opentelemetry.trace import StatusCode
        self._span.set_status(StatusCode.ERROR, str(error))
        self._span.record_exception(error)
        self._span.end()

    def __getattr__(self, name):
        return getattr(self._stream, name)


class AsyncStreamWrapper:
    """
    Wraps an async streaming response. Same contract as SyncStreamWrapper.
    """

    def __init__(self, stream: Any, span: otel_trace.Span, finalizer: Callable):
        self._stream = stream
        self._span = span
        self._finalizer = finalizer
        self._start_time = time.perf_counter()
        self._first_chunk_time: Optional[float] = None
        self._chunks: List[Any] = []
        self._finalized = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            chunk = await self._stream.__anext__()
        except StopAsyncIteration:
            self._finalize()
            raise
        except Exception as e:
            self._finalize_error(e)
            raise

        if self._first_chunk_time is None:
            self._first_chunk_time = time.perf_counter()
        self._chunks.append(chunk)
        return chunk

    async def __aenter__(self):
        if hasattr(self._stream, "__aenter__"):
            await self._stream.__aenter__()
        return self

    async def __aexit__(self, *args):
        if hasattr(self._stream, "__aexit__"):
            await self._stream.__aexit__(*args)
        self._finalize()

    def _finalize(self):
        if self._finalized:
            return
        self._finalized = True
        elapsed_ms = (time.perf_counter() - self._start_time) * 1000
        ttft_ms = None
        if self._first_chunk_time is not None:
            ttft_ms = (self._first_chunk_time - self._start_time) * 1000
        self._finalizer(self._span, self._chunks, elapsed_ms, ttft_ms)

    def _finalize_error(self, error: Exception):
        if self._finalized:
            return
        self._finalized = True
        from opentelemetry.trace import StatusCode
        self._span.set_status(StatusCode.ERROR, str(error))
        self._span.record_exception(error)
        self._span.end()

    def __getattr__(self, name):
        return getattr(self._stream, name)
