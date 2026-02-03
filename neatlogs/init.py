"""
Neatlogs SDK.
"""

import atexit
import os
import time
import uuid
from typing import List, Optional

from opentelemetry import metrics, trace
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.sampling import TraceIdRatioBased

from .core.exporter import NeatlogsExporter
from .core.logger import get_logger
from .core.metrics_correlation import SpanMetricMeterProviderProxy
from .core.span_processor import NeatlogsSpanProcessor
from .instrumentation.manager import InstrumentationManager

logger = get_logger()


_initialized = False
_tracer_provider = None
_meter_provider = None
_span_processor = None
_session_config = {
    "session_id": None,
    "user_id": None,
}


def _patch_semconv_ai_for_openllmetry(debug: bool) -> None:
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
        from .core.logger import get_logger

        logger = get_logger()
        logger.debug(
            "Patched opentelemetry.semconv_ai.SpanAttributes GEN_AI_* aliases for OpenLLMetry compatibility"
        )


def init(
    api_key: Optional[str] = None,
    endpoint: str = "http://localhost:3000/api/data/v4/batch",
    workflow_name: Optional[str] = None,
    session_id: Optional[str] = None,
    auto_session: bool = False,
    user_id: Optional[str] = None,
    instrument_tags: Optional[List[str]] = None,
    instrumentations: Optional[List[str]] = None,
    sample_rate: float = 1.0,
    batch_size: int = 100,
    flush_interval: float = 5.0,
    debug: bool = False,
    disable_export: bool = False,
) -> None:
    """
    Initialize Neatlogs SDK.
    """
    global _initialized

    if _initialized:
        if debug:
            logger.warning("Neatlogs already initialized, skipping re-initialization")
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

    final_session_id = None
    if session_id:
        final_session_id = session_id
    elif auto_session:
        timestamp = int(time.time())
        random_suffix = uuid.uuid4().hex[:8]
        final_session_id = f"session_{timestamp}_{random_suffix}"
        if debug:
            logger.debug(f"Auto-generated session_id: {final_session_id}")

    global _session_config
    _session_config["session_id"] = final_session_id
    _session_config["user_id"] = user_id

    resource_attrs = {
        SERVICE_NAME: workflow_name or "neatlogs-app",
        "neatlogs.workflow_name": workflow_name or "",
    }
    if final_session_id:
        resource_attrs["session.id"] = final_session_id
    if user_id:
        resource_attrs["user.id"] = user_id
    resource = Resource.create(resource_attrs)

    global _tracer_provider
    existing_provider = trace.get_tracer_provider()

    if existing_provider and hasattr(existing_provider, "add_span_processor"):
        provider = existing_provider
        if debug:
            logger.debug("Using existing tracer provider")
    else:
        sampler = None
        if sample_rate < 1.0:
            sampler = TraceIdRatioBased(sample_rate)
            if debug:
                logger.debug(f"Using TraceIdRatioBased sampler with rate {sample_rate}")

        provider = TracerProvider(resource=resource, sampler=sampler)
        trace.set_tracer_provider(provider)
        if debug:
            logger.debug("Created new tracer provider")

    _tracer_provider = provider

    exporter = NeatlogsExporter(
        api_key=api_key,
        endpoint=endpoint,
        batch_size=batch_size,
        flush_interval=flush_interval,
        disable_export=disable_export,
    )

    global _span_processor
    _span_processor = NeatlogsSpanProcessor(
        exporter=exporter,
        sample_rate=sample_rate,
        debug=debug,
    )
    provider.add_span_processor(_span_processor)

    if debug:
        logger.debug("Neatlogs tracer provider initialized")

    global _meter_provider
    _meter_provider = MeterProvider(
        resource=resource,
    )
    metrics.set_meter_provider(SpanMetricMeterProviderProxy(_meter_provider, exporter))

    if debug:
        logger.debug("Neatlogs meter provider initialized")

    manager = InstrumentationManager(
        provider=provider,
        debug=debug,
        excluded_urls=endpoint,
    )

    manager.instrument_threading()
    manager.instrument_http()

    if instrument_tags or instrumentations:
        manager.instrument(tags=instrument_tags, libraries=instrumentations)
        if debug:
            logger.debug(f"Instrumented libraries: {manager.instrumented}")

    atexit.register(shutdown)

    _initialized = True

    if debug:
        logger.info("Neatlogs SDK initialized successfully")
        logger.info(f"Endpoint: {endpoint}")
        logger.info(f"Workflow: {workflow_name or '(none)'}")
        logger.info(f"Session: {final_session_id or '(none)'}")
        logger.info(f"User: {user_id or '(none)'}")
        logger.info(f"Instrumentations: {manager.instrumented or '(none)'}")
        logger.info(f"Tags: {instrument_tags or []}")
        logger.info(f"Sample Rate: {sample_rate}")


def flush(timeout_millis: int = 30000) -> bool:
    """Flush all pending spans and metrics."""
    global _tracer_provider, _meter_provider
    success = True

    if _tracer_provider:
        try:
            logger.debug("Flushing tracer provider...")
            ok = _tracer_provider.force_flush(timeout_millis=timeout_millis)
            success = bool(ok) and success
            logger.debug("Tracer provider flushed successfully")
        except Exception as e:
            logger.error(f"Error flushing spans: {e}", exc_info=True)
            success = False

    if _meter_provider:
        try:
            logger.debug("Flushing meter provider...")
            ok = _meter_provider.force_flush(timeout_millis=timeout_millis)
            success = bool(ok) and success
            logger.debug("Meter provider flushed successfully")
        except Exception as e:
            logger.error(f"Error flushing metrics: {e}", exc_info=True)
            success = False

    return success


def get_session_config():
    """Get the current session configuration (session_id, user_id)."""
    return _session_config.copy()


def shutdown(timeout_millis: int = 30000) -> bool:
    """Shutdown the SDK and flush pending spans/metrics."""
    global _tracer_provider, _meter_provider, _span_processor, _initialized

    try:
        atexit.unregister(shutdown)
    except Exception:
        pass

    success = True

    if _span_processor:
        try:
            _span_processor._log_performance_stats()
        except Exception as e:
            logger.warning(f"Error logging performance stats: {e}")

    if _tracer_provider:
        try:
            logger.debug("Shutting down tracer provider...")
            ok = _tracer_provider.shutdown()
            success = (ok is None or bool(ok)) and success
            logger.debug("Tracer provider shut down successfully")
        except Exception as e:
            logger.error(f"Error shutting down tracer provider: {e}", exc_info=True)
            success = False

    if _meter_provider:
        try:
            logger.debug("Shutting down meter provider...")
            ok = _meter_provider.shutdown()
            success = (ok is None or bool(ok)) and success
            logger.debug("Meter provider shut down successfully")
        except Exception as e:
            logger.error(f"Error shutting down meter provider: {e}", exc_info=True)
            success = False

    _initialized = False
    _tracer_provider = None
    _meter_provider = None
    _span_processor = None
    _session_config["session_id"] = None
    _session_config["user_id"] = None

    logger.info("Neatlogs SDK shutdown complete")
    return success
