"""
Neatlogs SDK.
"""

import atexit
import os
import re
import sys
import time
import uuid
from typing import Any, Callable, Dict, List, Optional

try:
    from opentelemetry import logs
except ImportError:
    from opentelemetry import _logs as logs  # type: ignore[no-redef]
from opentelemetry import metrics, trace
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import SimpleLogRecordProcessor
from opentelemetry.sdk.environment_variables import (
    OTEL_ATTRIBUTE_COUNT_LIMIT,
    OTEL_SPAN_ATTRIBUTE_COUNT_LIMIT,
)
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import SpanLimits, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.trace.sampling import TraceIdRatioBased

from .core.logger import get_logger
from .core.span_processor import NeatlogsSpanProcessor
from .instrumentation.manager import InstrumentationManager
from .version import __version__

logger = get_logger()


_initialized = False
_tracer_provider = None
_meter_provider = None
_log_provider = None
_log_span_exporter = None
_span_processor = None
_debug_mode = False
_session_config = {
    "session_id": None,
    "user_id": None,
    "workflow_name": None,
    "_api_key": None,
    "_base_url": None,
}


def is_debug_enabled() -> bool:
    """Return True if neatlogs was initialized with debug=True."""
    return _debug_mode

_DEFAULT_MAX_SPAN_ATTRIBUTES = 10_000


def _resolve_workflow_name(workflow_name: Optional[str]) -> str:
    """Return non-empty workflow name; derive from script when omitted."""
    provided = (workflow_name or "").strip()
    if provided:
        return provided

    script_name = os.path.splitext(os.path.basename(sys.argv[0] or ""))[0]
    script_slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", script_name).strip("-").lower()
    if script_slug and script_slug not in {"python", "python3", "ipython", "-c"}:
        return script_slug

    return "neatlogs-app"


def _span_limits_for_capture_everything() -> SpanLimits:
    """
    OpenTelemetry defaults to 128 span attributes, which can silently drop semantic
    attributes when instrumenting LLM apps (retrieval docs, tool IO, etc).

    If the user explicitly sets OTel limits via env vars, respect that. Otherwise
    default to a larger max-span-attributes value (matching OpenInference's approach).
    """
    span_limit = os.getenv(OTEL_SPAN_ATTRIBUTE_COUNT_LIMIT, "")
    general_limit = os.getenv(OTEL_ATTRIBUTE_COUNT_LIMIT, "")
    if span_limit.strip() or general_limit.strip():
        return SpanLimits()
    return SpanLimits(max_span_attributes=_DEFAULT_MAX_SPAN_ATTRIBUTES)



def init(
    api_key: Optional[str] = None,
    endpoint: str = "https://staging-cloud.neatlogs.com/api/data/v4/batch",
    workflow_name: Optional[str] = None,
    session_id: Optional[str] = None,
    auto_session: bool = False,
    user_id: Optional[str] = None,
    tags: Optional[List[str]] = None,
    instrumentations: Optional[List[str]] = None,
    sample_rate: float = 1.0,
    batch_size: int = 100,
    flush_interval: float = 5.0,
    debug: bool = False,
    disable_export: bool = False,
    capture_logs: bool = False,
    log_level: str = "INFO",
    mask: Optional[Callable[[Dict[str, Any]], Optional[Dict[str, Any]]]] = None,
    pii_enabled: Optional[bool] = None,
    pii_span_types: Optional[List[str]] = None,
) -> None:
    """
    Initialize Neatlogs SDK.

    Args:
        api_key: Neatlogs API key (or set NEATLOGS_API_KEY env var)
        endpoint: Neatlogs backend endpoint
        workflow_name: Logical grouping for traces
        session_id: Custom session ID (for multi-turn conversations)
        auto_session: Auto-generate session_id (useful for chatbots)
        user_id: User identifier (propagates to all spans)
        tags: Global tags for all traces (list of strings only, e.g., ['production', 'api-v2'])
        instrumentations: Specific libraries to instrument
        sample_rate: Trace sampling rate (0.0-1.0)
        batch_size: Max spans per batch
        flush_interval: Seconds between batch flushes
        debug: Enable debug logging
        disable_export: Disable data export (for testing)
        capture_logs: Capture Python logging.* calls and neatlogs.log() as LOG spans.
                      Default: False. Enable to see intermediate steps in the timeline.
        log_level: Minimum Python logging level to capture when capture_logs=True.
                   Default: "INFO".
        mask: Optional callable applied to every span dict before export.
              Receives the full span dict and must return the (possibly modified) dict.
              Use this to redact PII from inputs, outputs, and attributes.
              Per-span masks (set via @span(mask=fn) or with trace(..., mask=fn))
              take precedence over this global mask.
              Example::

                  def redact(span):
                      attrs = span.get("attributes", {})
                      for key in list(attrs):
                          if "email" in key:
                              attrs[key] = "***"
                      return span

                  neatlogs.init(mask=redact)
        pii_enabled: Override the team-level server-side PII redaction toggle for this
              project. True = enable redaction, False = disable redaction entirely.
              When None (default), the team setting in the Neatlogs dashboard is used.
        pii_span_types: Override which span types have server-side PII redaction applied.
              Pass a list of span kind strings, e.g. ["LLM", "TOOL"]. When None (default),
              the team setting is used. Pass an empty list to disable redaction for all types.
    """
    global _initialized

    if _initialized:
        if debug:
            logger.warning("Neatlogs already initialized, skipping re-initialization")
        return

    disable_export_resolved = bool(disable_export) or (
        os.getenv("NEATLOGS_DISABLE_EXPORT", "").lower() in ("true", "1", "yes")
    )

    if api_key is not None and str(api_key).strip():
        resolved_key = str(api_key).strip()
    else:
        resolved_key = (os.getenv("NEATLOGS_API_KEY") or "").strip()

    if not resolved_key:
        disable_export_resolved = True
        resolved_key = "disabled"
        if debug:
            logger.warning(
                "No NEATLOGS_API_KEY set; HTTP export disabled. "
                "Set NEATLOGS_API_KEY (or pass api_key=) to send spans to the backend."
            )

    if debug:
        import logging

        logging.basicConfig(
            level=logging.DEBUG,
            format="%(name)s - %(levelname)s - %(message)s",
        )

    global _debug_mode
    _debug_mode = debug

    resolved_workflow_name = _resolve_workflow_name(workflow_name)

    final_session_id = None
    if session_id:
        final_session_id = session_id
    elif auto_session:
        timestamp = int(time.time())
        random_suffix = uuid.uuid4().hex[:8]
        final_session_id = f"session_{timestamp}_{random_suffix}"
        if debug:
            logger.debug(f"Auto-generated session_id: {final_session_id}")

    from urllib.parse import urlparse as _urlparse
    _parsed = _urlparse(endpoint)
    _base_url = f"{_parsed.scheme}://{_parsed.netloc}"

    global _session_config
    _session_config["session_id"] = final_session_id
    _session_config["user_id"] = user_id
    _session_config["workflow_name"] = resolved_workflow_name
    _session_config["_api_key"] = resolved_key
    _session_config["_base_url"] = _base_url

    resource_attrs = {
        SERVICE_NAME: workflow_name or "neatlogs-app",
        "service.version": __version__,
        "neatlogs.workflow_name": resolved_workflow_name,
    }
    if final_session_id:
        resource_attrs["session.id"] = final_session_id
    if user_id:
        resource_attrs["user.id"] = user_id
    if tags:
        # Tags must be a list of strings
        if not isinstance(tags, list):
            raise ValueError(f"tags must be a list of strings, got {type(tags)}")
        # Validate all elements are strings
        if not all(isinstance(tag, str) for tag in tags):
            raise ValueError("All tags must be strings")
        # Store as comma-separated string for OTel resource attributes
        resource_attrs["neatlogs.tags"] = ",".join(tags)
    if pii_enabled is not None:
        resource_attrs["neatlogs.pii.enabled"] = "true" if pii_enabled else "false"
    if pii_span_types is not None:
        resource_attrs["neatlogs.pii.span_types"] = ",".join(pii_span_types)
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

        provider = TracerProvider(
            resource=resource,
            sampler=sampler,
            span_limits=_span_limits_for_capture_everything(),
        )
        trace.set_tracer_provider(provider)
        if debug:
            logger.debug("Created new tracer provider")

    _tracer_provider = provider

    # NeatlogsSpanProcessor: pure pre-processing (attribute normalization + file logging)
    global _span_processor
    _span_processor = NeatlogsSpanProcessor(
        sample_rate=sample_rate,
        debug=debug,
        mask=mask,
    )
    provider.add_span_processor(_span_processor)

    # BatchSpanProcessor + OTLPSpanExporter: standard transport
    if not disable_export_resolved:
        otlp_headers = {"x-api-key": resolved_key}
        # Always send to {base_url}/v1/traces regardless of what endpoint string was passed.
        # Users may pass the legacy /api/data/v4/batch path; we normalise to the OTLP path here.
        traces_endpoint = endpoint if endpoint.endswith("/v1/traces") else f"{_base_url}/v1/traces"
        otlp_exporter = OTLPSpanExporter(
            endpoint=traces_endpoint,
            headers=otlp_headers,
        )
        batch_processor = BatchSpanProcessor(
            otlp_exporter,
            max_export_batch_size=batch_size,
            schedule_delay_millis=int(flush_interval * 1000),
        )
        provider.add_span_processor(batch_processor)
        if debug:
            logger.debug(f"OTLP trace exporter configured: {traces_endpoint}")
    elif debug:
        logger.debug("Export disabled — spans will not be sent to backend")

    if debug:
        logger.debug("Neatlogs tracer provider initialized")

    global _meter_provider
    _meter_provider = MeterProvider(resource=resource)
    metrics.set_meter_provider(_meter_provider)

    if debug:
        logger.debug("Neatlogs meter provider initialized")

    # --- Logs signal (opt-in) ---
    # neatlogs.log(), capture_stdout=True, and logging.* auto-capture all require
    # capture_logs=True. When False, nothing is captured as LOG spans.
    global _log_provider, _log_span_exporter
    if capture_logs:
        from .core.exporter import NeatlogsExporter
        from .core.log_exporter import NeatlogsLogExporter

        logs_batch_endpoint = f"{_base_url}/api/data/v4/batch"
        _log_span_exporter = NeatlogsExporter(
            api_key=resolved_key,
            endpoint=logs_batch_endpoint,
            workflow_name=resolved_workflow_name,
            batch_size=batch_size,
            flush_interval=flush_interval,
            disable_export=disable_export_resolved,
        )
        _log_provider = LoggerProvider(resource=resource)
        _log_provider.add_log_record_processor(
            SimpleLogRecordProcessor(NeatlogsLogExporter(_log_span_exporter))
        )
        logs.set_logger_provider(_log_provider)

        try:
            import logging as _stdlib_logging

            from opentelemetry.instrumentation.logging import LoggingInstrumentor

            _stdlib_level = getattr(_stdlib_logging, log_level.upper(), _stdlib_logging.WARNING)
            LoggingInstrumentor().instrument(
                log_level=_stdlib_level,
                logger_provider=_log_provider,
            )
            if debug:
                logger.debug(
                    "Neatlogs log capture enabled "
                    f"(logging.* at {log_level.upper()}+, endpoint: {logs_batch_endpoint})"
                )
        except ImportError:
            if debug:
                logger.debug(
                    "opentelemetry-instrumentation-logging not installed — "
                    "Install with: pip install opentelemetry-instrumentation-logging"
                )
    elif debug:
        logger.debug("Log capture disabled (pass capture_logs=True to enable)")

    manager = InstrumentationManager(
        provider=provider,
        debug=debug,
        excluded_urls=endpoint,
    )

    manager.instrument_threading()
    manager.instrument_http()

    if instrumentations:
        manager.instrument(libraries=instrumentations)
        if debug:
            logger.debug(f"Instrumented libraries: {manager.instrumented}")

    atexit.register(shutdown)

    _initialized = True

    if debug:
        logger.info("Neatlogs SDK initialized successfully")
        logger.info(f"Endpoint: {endpoint}")
        logger.info(f"Workflow: {resolved_workflow_name}")
        logger.info(f"Session: {final_session_id or '(none)'}")
        logger.info(f"User: {user_id or '(none)'}")
        logger.info(f"Tags: {tags or []}")
        logger.info(f"Instrumentations: {manager.instrumented or '(none)'}")
        logger.info(f"Sample Rate: {sample_rate}")


def flush(timeout_millis: int = 30000) -> bool:
    """Flush all pending spans and metrics."""
    global _tracer_provider, _meter_provider, _log_span_exporter
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

    if _log_span_exporter:
        try:
            logger.debug("Flushing log span exporter...")
            _log_span_exporter.flush(timeout=timeout_millis / 1000.0)
            logger.debug("Log span exporter flushed successfully")
        except Exception as e:
            logger.error(f"Error flushing logs: {e}", exc_info=True)
            success = False

    return success


def get_session_config():
    """Get the current session configuration (session_id, user_id)."""
    return _session_config.copy()


def shutdown(timeout_millis: int = 30000) -> bool:
    """Shutdown the SDK and flush pending spans/metrics."""
    global _tracer_provider, _meter_provider, _log_provider, _log_span_exporter, _span_processor, _initialized

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

    if _log_provider:
        try:
            logger.debug("Shutting down log provider...")
            ok = _log_provider.shutdown()
            success = (ok is None or bool(ok)) and success
            logger.debug("Log provider shut down successfully")
        except Exception as e:
            logger.error(f"Error shutting down log provider: {e}", exc_info=True)
            success = False

    if _log_span_exporter:
        try:
            logger.debug("Shutting down log span exporter...")
            _log_span_exporter.shutdown()
            logger.debug("Log span exporter shut down successfully")
        except Exception as e:
            logger.error(f"Error shutting down log span exporter: {e}", exc_info=True)
            success = False

    try:
        from opentelemetry.instrumentation.logging import LoggingInstrumentor

        LoggingInstrumentor().uninstrument()
    except Exception:
        pass

    _initialized = False
    _tracer_provider = None
    _meter_provider = None
    _log_provider = None
    _log_span_exporter = None
    _span_processor = None
    _debug_mode = False
    _session_config["session_id"] = None
    _session_config["user_id"] = None
    _session_config["workflow_name"] = None

    logger.info("Neatlogs SDK shutdown complete")
    return success
