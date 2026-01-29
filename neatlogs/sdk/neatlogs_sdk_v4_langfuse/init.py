"""
Neatlogs SDK v4 initialization (new).

This module is a drop-in replacement for init.py while we experiment with
deduping spans emitted by dual instrumentation (OpenInference + OpenLLMetry).
"""

import os
import time
import uuid
from typing import Optional, List

from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.resources import Resource, SERVICE_NAME

from .core.exporter import NeatlogsExporter
from .core.span_processor import NeatlogsSpanProcessor
from .core.metrics_correlation import SpanMetricMeterProviderProxy
from .instrumentation.manager import InstrumentationManager


_initialized = False
_tracer_provider = None
_meter_provider = None
_span_processor = None
_session_config = {
    "session_id": None,
    "user_id": None,
}


def _patch_semconv_ai_for_openllmetry(debug: bool) -> None:
    """
    Some OpenLLMetry instrumentations reference SpanAttributes.GEN_AI_* constants
    that were renamed in opentelemetry-semconv-ai (the string values stayed the same).
    Add aliases at runtime to avoid noisy AttributeError logs and to preserve attributes.
    """
    try:
        from opentelemetry.semconv_ai import SpanAttributes
    except Exception:
        return

    aliases = (
        ("GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS", "LLM_USAGE_CACHE_READ_INPUT_TOKENS"),
        (
            "GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS",
            "LLM_USAGE_CACHE_CREATION_INPUT_TOKENS",
        ),
    )

    changed = False
    for missing, existing in aliases:
        if not hasattr(SpanAttributes, missing) and hasattr(SpanAttributes, existing):
            setattr(SpanAttributes, missing, getattr(SpanAttributes, existing))
            changed = True

    if debug and changed:
        # Debug-only so we don't pollute normal output.
        print(
            "Patched opentelemetry.semconv_ai.SpanAttributes GEN_AI_* aliases for OpenLLMetry compatibility"
        )


def init(
    api_key: Optional[str] = None,
    endpoint: str = "http://localhost:3000/api/data/v4/batch",
    workflow_name: Optional[str] = None,
    session_id: Optional[str] = None,
    auto_session: bool = False,
    user_id: Optional[str] = None,
    # Instrumentation control
    instrument_tags: Optional[List[str]] = None,
    instrumentations: Optional[List[str]] = None,
    enable_http_tracing: bool = True,
    # Performance
    sample_rate: float = 1.0,
    batch_size: int = 100,
    flush_interval: float = 5.0,
    # Debug
    debug: bool = False,
) -> None:
    """
    Initialize Neatlogs SDK.

    Same surface area as init.py; implemented in a new module so we can swap
    components without breaking downstream imports.
    """
    global _initialized

    if _initialized:
        if debug:
            print("Neatlogs already initialized")
        return

    api_key = api_key or os.getenv("NEATLOGS_API_KEY")
    if not api_key:
        raise ValueError(
            "api_key required. Either pass it to init() or set NEATLOGS_API_KEY environment variable."
        )

    if debug:
        import logging
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(name)s - %(levelname)s - %(message)s",
        )

    _patch_semconv_ai_for_openllmetry(debug=debug)

    # Determine final session ID
    final_session_id = None
    if session_id:
        final_session_id = session_id
    elif auto_session:
        timestamp = int(time.time())
        random_suffix = uuid.uuid4().hex[:8]
        final_session_id = f"session_{timestamp}_{random_suffix}"
        if debug:
            print(f"Auto-generated session_id: {final_session_id}")

    # Store session_id and user_id in global state for trace() access
    global _session_config
    _session_config["session_id"] = final_session_id
    _session_config["user_id"] = user_id

    # Setup resource with metadata (applies to all spans)
    resource_attrs = {
        SERVICE_NAME: workflow_name or "neatlogs-app",
        "neatlogs.workflow_name": workflow_name or "",
    }
    if final_session_id:
        resource_attrs["session.id"] = final_session_id
    if user_id:
        resource_attrs["user.id"] = user_id
    resource = Resource.create(resource_attrs)

    # Get or create tracer provider
    global _tracer_provider
    existing_provider = trace.get_tracer_provider()

    if existing_provider and hasattr(existing_provider, "add_span_processor"):
        provider = existing_provider
        if debug:
            print("Using existing tracer provider")
    else:
        provider = TracerProvider(resource=resource)
        trace.set_tracer_provider(provider)
        if debug:
            print("Created new tracer provider")

    _tracer_provider = provider

    exporter = NeatlogsExporter(
        api_key=api_key,
        endpoint=endpoint,
        batch_size=batch_size,
        flush_interval=flush_interval,
    )

    global _span_processor
    _span_processor = NeatlogsSpanProcessor(
        exporter=exporter,
        sample_rate=sample_rate,
        debug=debug,
    )
    provider.add_span_processor(_span_processor)

    if debug:
        print("Neatlogs tracer provider initialized")

    # Metrics provider
    global _meter_provider
    _meter_provider = MeterProvider(
        resource=resource,
    )
    # Wrap the provider so OpenLLMetry metric calls also emit per-span raw metric points
    # to NeatlogsExporter (with trace_id/span_id) for downstream "metrics on spans".
    metrics.set_meter_provider(SpanMetricMeterProviderProxy(_meter_provider, exporter))

    if debug:
        print("Neatlogs meter provider initialized")

    manager = InstrumentationManager(
        provider=provider,
        debug=debug,
        excluded_urls=endpoint,
    )

    manager.instrument_threading()
    if enable_http_tracing:
        manager.instrument_http()

    if instrument_tags or instrumentations:
        manager.instrument(tags=instrument_tags, libraries=instrumentations)
        if debug:
            print(f"Instrumented libraries: {manager.instrumented}")

    _initialized = True

    if debug:
        print("Neatlogs SDK initialized successfully")
        print(f"Endpoint: {endpoint}")
        print(f"Workflow: {workflow_name or '(none)'}")
        print(f"Session: {final_session_id or '(none)'}")
        print(f"User: {user_id or '(none)'}")
        print(f"Instrumentations: {manager.instrumented or '(none)'}")
        print(f"Tags: {instrument_tags or []}")
        print(f"Sample Rate: {sample_rate}")


def flush(timeout_millis: int = 30000) -> bool:
    """Flush all pending spans and metrics."""
    global _tracer_provider, _meter_provider
    success = True

    if _tracer_provider:
        try:
            ok = _tracer_provider.force_flush(timeout_millis=timeout_millis)
            success = bool(ok) and success
        except Exception as e:
            print(f"Error flushing spans: {e}")
            success = False

    if _meter_provider:
        try:
            ok = _meter_provider.force_flush(timeout_millis=timeout_millis)
            success = bool(ok) and success
        except Exception as e:
            print(f"Error flushing metrics: {e}")
            success = False

    return success


def get_session_config():
    """Get the current session configuration (session_id, user_id)."""
    return _session_config.copy()


def shutdown(timeout_millis: int = 30000) -> bool:
    """Shutdown the SDK and flush pending spans/metrics."""
    global _tracer_provider, _meter_provider, _span_processor, _initialized
    success = True

    if _span_processor:
        try:
            _span_processor._log_performance_stats()
        except Exception as e:
            print(f"Error logging performance stats: {e}")

    if _tracer_provider:
        try:
            ok = _tracer_provider.shutdown()
            success = (ok is None or bool(ok)) and success
        except Exception as e:
            print(f"Error shutting down tracer provider: {e}")
            success = False

    if _meter_provider:
        try:
            ok = _meter_provider.shutdown()
            success = (ok is None or bool(ok)) and success
        except Exception as e:
            print(f"Error shutting down meter provider: {e}")
            success = False

    _initialized = False
    _tracer_provider = None
    _meter_provider = None
    _span_processor = None
    _session_config["session_id"] = None
    _session_config["user_id"] = None

    return success
