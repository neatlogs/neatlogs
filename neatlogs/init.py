"""
Neatlogs SDK.
"""

import atexit
import functools
import hashlib
import json
import math
import os
import queue
import re
import signal
import sys
import threading
import time
from typing import Any, Callable, Dict, List, Optional

from opentelemetry import trace as otel_trace
from opentelemetry.exporter.otlp.proto.http import Compression
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk.environment_variables import (
    OTEL_ATTRIBUTE_COUNT_LIMIT,
    OTEL_SPAN_ATTRIBUTE_COUNT_LIMIT,
)
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import SpanLimits, TracerProvider
from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased

from ._wrap_utils import _normalize_traces_endpoint
from .constants import DEFAULT_INGEST_ENDPOINT, export_queue_capacity
from .core.byte_limited_exporter import ByteLimitedSpanExporter
from .core.byte_limited_log_exporter import ByteLimitedLogExporter
from .core.deadline import DeadlineWorker, bounded_call
from .core.delivery import (
    DeliveryDiagnostics,
    ObservableBatchLogRecordProcessor,
    ObservableBatchSpanProcessor,
)
from .core.logger import get_logger
from .core.media import PendingMediaStore, set_default_media_store
from .core.media_exporter import TypedMediaLogExporter, TypedMediaSpanExporter
from .core.span_processor import CompletionMarkerSpanProcessor, NeatlogsSpanProcessor
from .core.transport import build_otlp_session
from .core.upload_authority import (
    AuthenticatedUploadAuthority,
    DisabledUploadAuthority,
)
from .core.upload_authority import uploads_enabled as resolve_uploads_enabled
from .errors import NeatlogsConfigurationError
from .instrumentation.manager import InstrumentationManager
from .version import __version__

logger = get_logger()


_initialized = False
_init_signature = None
_tracer_provider = None
_owns_tracer_provider = False  # True only when neatlogs created the provider (safe to shut down)


_log_provider = None
_span_processor = None
_transport_span_processors = []
_completion_span_processor = None
_instrumentation_manager = None
_debug_mode = False
_delivery_diagnostics = DeliveryDiagnostics()
_upload_authority = None
_media_store = None
_signal_handlers = {}
_signal_shutdown_in_progress = False
_shutdown_condition = threading.Condition(threading.RLock())
_shutdown_state = "idle"
_shutdown_owner = None
_shutdown_result = True
_shutdown_worker = None
_lifecycle_operation_lock = threading.RLock()
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


def _instrument_library(library: str) -> bool:
    """Activate one library through the current default instrumentation manager."""
    manager = _instrumentation_manager
    if manager is None:
        return False
    manager.instrument(libraries=[library])
    return library in manager.instrumented


def _trace_sampler(sample_rate: float) -> ParentBased:
    """Validate one trace-level rate and preserve the parent's sampling decision."""
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, (int, float)):
        raise NeatlogsConfigurationError("sample_rate must be a finite number from 0.0 to 1.0")
    rate = float(sample_rate)
    if not math.isfinite(rate) or rate < 0.0 or rate > 1.0:
        raise NeatlogsConfigurationError("sample_rate must be a finite number from 0.0 to 1.0")
    return ParentBased(root=TraceIdRatioBased(rate))


def _restore_shutdown_signal_handlers() -> None:
    """Restore handlers that were present before Neatlogs initialized."""
    global _signal_handlers
    if threading.current_thread() is not threading.main_thread():
        return
    for signum, previous in list(_signal_handlers.items()):
        try:
            signal.signal(signum, previous)
        except (OSError, RuntimeError, ValueError):
            pass
    _signal_handlers = {}


def _shutdown_signal_handler(signum, frame) -> None:
    """Close active spans, flush, then preserve the process' signal semantics."""
    global _signal_shutdown_in_progress
    if _signal_shutdown_in_progress:
        return
    _signal_shutdown_in_progress = True
    previous = _signal_handlers.get(signum, signal.SIG_DFL)
    if previous == signal.SIG_IGN:
        _signal_shutdown_in_progress = False
        return
    try:
        try:
            reason = signal.Signals(signum).name
        except ValueError:
            reason = f"signal-{signum}"
        shutdown(termination_reason=reason)
    finally:
        _signal_shutdown_in_progress = False

    if callable(previous) and previous is not _shutdown_signal_handler:
        previous(signum, frame)
        # The application owns this signal. A handler that returns may be
        # intentionally coordinating its own graceful shutdown; do not force an
        # additional KeyboardInterrupt/SystemExit after it regains control.
        return
    if signum == getattr(signal, "SIGINT", None):
        raise KeyboardInterrupt
    raise SystemExit(128 + int(signum))


def _register_shutdown_signal_handlers() -> None:
    """Best-effort SIGINT/SIGTERM registration; only legal on the main thread."""
    global _signal_handlers
    if threading.current_thread() is not threading.main_thread():
        logger.debug("Skipping Neatlogs signal handlers outside the main thread")
        return
    for signum in (getattr(signal, "SIGINT", None), getattr(signal, "SIGTERM", None)):
        if signum is None or signum in _signal_handlers:
            continue
        try:
            previous = signal.getsignal(signum)
            if previous == signal.SIG_IGN:
                continue
            signal.signal(signum, _shutdown_signal_handler)
            _signal_handlers[signum] = previous
        except (OSError, RuntimeError, ValueError) as exc:
            logger.debug(f"Could not register shutdown handler for signal {signum}: {exc}")


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


def _serialize_init(func):
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        with _lifecycle_operation_lock:
            return func(*args, **kwargs)

    return wrapped


def _configuration_signature(**values):
    api_key = values.pop("api_key")
    resolved_key = str(api_key).strip() if api_key is not None else ""
    if not resolved_key:
        resolved_key = (os.getenv("NEATLOGS_API_KEY") or "").strip()
    values["api_key_sha256"] = hashlib.sha256(resolved_key.encode()).hexdigest()
    values["disable_export"] = bool(values["disable_export"]) or os.getenv(
        "NEATLOGS_DISABLE_EXPORT", ""
    ).lower() in ("true", "1", "yes")
    mask = values.pop("mask")
    provider = values.pop("tracer_provider")
    doctor_probe = values.pop("_doctor_probe")
    doctor_probe_exporter = values.pop("_doctor_probe_exporter")
    values["mask_identity"] = id(mask) if mask is not None else None
    values["provider_identity"] = id(provider) if provider is not None else None
    # Doctor changes both the resource contract and the transport headers. It
    # must never be treated as an idempotent reuse of an ordinary initialized
    # runtime, because the probe owns and shuts down the pipeline it creates.
    values["doctor_probe"] = bool(doctor_probe)
    values["doctor_probe_exporter_identity"] = (
        id(doctor_probe_exporter) if doctor_probe_exporter is not None else None
    )
    return json.dumps(values, sort_keys=True, separators=(",", ":"), default=str)


@_serialize_init
def init(
    api_key: Optional[str] = None,
    endpoint: str = DEFAULT_INGEST_ENDPOINT,
    workflow_name: Optional[str] = None,
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
    pii_entities: Optional[List[str]] = None,
    pii_span_types: Optional[List[str]] = None,
    tracer_provider: Optional[Any] = None,
    isolate: Optional[bool] = None,
    register_shutdown_handlers: bool = True,
    uploads_enabled: Optional[bool] = None,
    _doctor_probe: bool = False,
    _doctor_probe_exporter: Optional[Any] = None,
) -> None:
    """
    Initialize Neatlogs SDK.

    Args:
        api_key: Neatlogs API key (or set NEATLOGS_API_KEY env var)
        endpoint: Neatlogs backend endpoint
        workflow_name: Logical grouping for traces
        user_id: Operator identifier — whoever is RUNNING the SDK (a developer, a
                 service account, the OS user). Propagates to all spans as a
                 resource attribute. NOT the end-user of your app.

                 NOTE: session and end-user identity are NOT set here. They are
                 per-request, not process-global. Set them at the trace root via
                 ``@span(kind="WORKFLOW", session_id=..., end_user_id=...)`` /
                 ``with neatlogs.trace(session_id=..., end_user_id=...)``, or for
                 wrapper-only code via ``with neatlogs.identify(session_id=...,
                 end_user_id=...)``.
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
              project and persist the setting so the dashboard reflects it.
              True = enable redaction, False = disable redaction entirely. When None
              (default), the project setting in the Neatlogs dashboard is used.
        pii_entities: Optional project-level Presidio entity selection to persist with
              the SDK override, e.g. ["PERSON", "EMAIL_ADDRESS"]. When None, the
              project's existing saved entity selection is preserved.
        pii_span_types: Override which span types have server-side PII redaction applied.
              Pass a list of span kind strings, e.g. ["LLM", "TOOL"]. This selection
              is persisted so the dashboard reflects it. When None (default), the
              project setting is preserved.
        tracer_provider: Pass a private ``TracerProvider`` that Neatlogs may configure.
              created but did NOT install as the OTel global) and neatlogs will emit ALL of
              its spans — auto-instrumented, wrap()/trace()/@span, and the internal
              completion marker — into that provider ONLY. Use this when another tool
              already owns the global provider (e.g. your own OpenTelemetry setup exporting
              to Langfuse) and you need neatlogs and that tool to share NO pipeline: no
              neatlogs span reaches the other backend and no foreign span reaches neatlogs.
              neatlogs never shuts this provider down and never claims the global
              meter/logger providers. When None (default), behaviour is auto-detected
              unnecessary because Neatlogs creates an SDK-private provider by default.
        isolate: Deprecated compatibility option. Neatlogs is always isolated and
              never installs or reuses the process-global tracer provider.
        register_shutdown_handlers: Register SIGINT and SIGTERM handlers that end
              active Neatlogs spans child-first and flush before preserving normal
              signal termination. Defaults to True. Set False only when the host
              application owns signal handling and calls ``neatlogs.shutdown()``.
        uploads_enabled: Enable the authenticated typed-media and oversized OTLP
              upload contract. Defaults to ``NEATLOGS_UPLOADS_ENABLED``, which
              is false when unset.
    """
    global _initialized, _init_signature, _shutdown_worker

    uploads_enabled_resolved = resolve_uploads_enabled(
        uploads_enabled, os.getenv("NEATLOGS_UPLOADS_ENABLED")
    )
    candidate_signature = _configuration_signature(
        api_key=api_key,
        endpoint=endpoint,
        workflow_name=workflow_name,
        user_id=user_id,
        tags=tags or [],
        instrumentations=instrumentations or [],
        sample_rate=sample_rate,
        batch_size=batch_size,
        flush_interval=flush_interval,
        debug=debug,
        disable_export=disable_export,
        capture_logs=capture_logs,
        log_level=log_level,
        mask=mask,
        pii_enabled=pii_enabled,
        pii_entities=pii_entities or [],
        pii_span_types=pii_span_types or [],
        tracer_provider=tracer_provider,
        isolate=isolate,
        register_shutdown_handlers=register_shutdown_handlers,
        uploads_enabled=uploads_enabled_resolved,
        _doctor_probe=_doctor_probe,
        _doctor_probe_exporter=_doctor_probe_exporter,
    )

    if _initialized:
        if _init_signature == candidate_signature:
            if debug:
                logger.warning("Neatlogs already initialized with the same configuration")
            return
        raise NeatlogsConfigurationError(
            "Neatlogs is already running with different configuration; "
            "call shutdown() before reinitializing"
        )

    sampler = _trace_sampler(sample_rate)
    if tracer_provider is not None and float(sample_rate) != 1.0:
        raise NeatlogsConfigurationError(
            "sample_rate cannot configure a caller-owned tracer_provider; "
            "configure its sampler directly or omit tracer_provider"
        )
    if tracer_provider is not None and tracer_provider is otel_trace.get_tracer_provider():
        raise NeatlogsConfigurationError(
            "tracer_provider must be private and must not be the process-global provider"
        )

    global _delivery_diagnostics
    _delivery_diagnostics = DeliveryDiagnostics()

    disable_export_resolved = bool(disable_export) or (
        os.getenv("NEATLOGS_DISABLE_EXPORT", "").lower() in ("true", "1", "yes")
    )
    # Probe is an explicit CLI action whose purpose is a controlled export. A
    # process-wide disable flag must not silently turn it into a false local-only
    # pass; local Doctor never sets this internal flag.
    if _doctor_probe:
        disable_export_resolved = False

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

    if debug:
        try:
            from .core.logger import set_log_level

            set_log_level(logging.DEBUG)  # ensure our debug lines emit from here on
            import inspect

            caller = inspect.stack()[1]
            logger.debug(
                f"[neatlogs.init] called from {caller.filename}:{caller.lineno} "
                f"(in {caller.function})"
            )
        except Exception:
            pass

    resolved_workflow_name = _resolve_workflow_name(workflow_name)

    from urllib.parse import urlparse as _urlparse

    traces_endpoint = _normalize_traces_endpoint(endpoint)
    _parsed = _urlparse(traces_endpoint)
    _base_url = f"{_parsed.scheme}://{_parsed.netloc}"

    # Session and end-user identity are deliberately NOT resource attributes:
    # they are per-request, set at the trace root (trace()/@span) or via
    # neatlogs.identify(). Only the operator user.id is process-global here.
    resource_attrs = {
        SERVICE_NAME: workflow_name or "neatlogs-app",
        "service.version": __version__,
        "neatlogs.workflow_name": resolved_workflow_name,
    }
    if _doctor_probe:
        resource_attrs["neatlogs.doctor"] = True
        resource_attrs["neatlogs.doctor.version"] = "v1"
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
    if pii_entities is not None:
        if not pii_entities or not all(
            isinstance(entity, str) and entity.strip() for entity in pii_entities
        ):
            raise ValueError("pii_entities must be a non-empty list of strings")
        resource_attrs["neatlogs.pii.entities"] = ",".join(
            entity.strip() for entity in pii_entities
        )
    if pii_span_types is not None:
        resource_attrs["neatlogs.pii.span_types"] = ",".join(pii_span_types)
    resource = Resource.create(resource_attrs)

    global _session_config
    _session_config["user_id"] = user_id
    _session_config["workflow_name"] = resolved_workflow_name
    _session_config["_api_key"] = resolved_key
    _session_config["_base_url"] = _base_url

    # Create and publish upload resources only after all user input validation
    # succeeds. A failed init must not leave raw-media storage or an HTTP session.
    global _upload_authority, _media_store
    _upload_authority = DisabledUploadAuthority()
    _media_store = None
    if uploads_enabled_resolved and not disable_export_resolved:
        _upload_authority = AuthenticatedUploadAuthority(
            base_url=_base_url,
            api_key=resolved_key,
        )
        _media_store = PendingMediaStore(max_bytes=_upload_authority.max_upload_bytes)
    set_default_media_store(_media_store)

    global _tracer_provider, _owns_tracer_provider
    if tracer_provider is not None:
        # ISOLATED MODE. The caller handed us a PRIVATE provider (never installed
        # as the OTel global) so neatlogs shares NO pipeline with a co-tenant such
        # as a user's own OpenTelemetry/Langfuse global provider. We attach our
        # processor/exporter to it, and — crucially — register it as the single
        # provider neatlogs resolves ALL its own tracers from (wrap()/trace()/@span
        # and the completion marker), so not one neatlogs span reaches the global
        # pipeline. We do NOT own it (never shut it down) and never touch global
        # meter/logger providers below. This is what makes isolation total.
        provider = tracer_provider
        _owns_tracer_provider = False
        # Merge neatlogs' resource (service.name + neatlogs.workflow_name + tags/pii)
        # onto the caller's provider so exported spans carry workflow_name — the
        # branches that build their OWN provider pass resource= at construction,
        # this branch cannot, so merge it in (neatlogs values win over the SDK's
        # unknown_service sentinel). Best-effort: never fail init on a custom provider.
        try:
            if getattr(provider, "resource", None) is not None:
                provider._resource = provider.resource.merge(resource)
        except Exception:  # noqa: BLE001
            if debug:
                logger.debug("Could not merge neatlogs resource onto provided provider")
        if debug:
            logger.debug("Using explicitly provided tracer provider (isolated mode)")
    else:
        provider = TracerProvider(
            resource=resource,
            sampler=sampler,
            span_limits=_span_limits_for_capture_everything(),
        )
        _owns_tracer_provider = True
        if debug:
            logger.debug("Created SDK-private tracer provider (owned by neatlogs)")

    _tracer_provider = provider

    # Register this provider as the single source of truth for every neatlogs
    # tracer (wrap()/trace()/@span + completion marker). In isolated mode this is
    # the private provider, so neatlogs' own spans never reach the global pipeline.
    from ._wrap_utils import set_neatlogs_provider

    set_neatlogs_provider(provider)

    # NeatlogsSpanProcessor: pure pre-processing (attribute normalization + file logging)
    global _span_processor
    _span_processor = NeatlogsSpanProcessor(
        debug=debug,
        mask=mask,
        emit_completion_markers=False,
        # A private/isolated pipeline contains only Neatlogs execution spans,
        # even when the caller retains provider shutdown ownership.
        own_all_spans=True,
    )
    provider.add_span_processor(_span_processor)

    # BatchSpanProcessor + OTLPSpanExporter: standard transport
    if not disable_export_resolved:
        otlp_headers = {"x-api-key": resolved_key}
        if _doctor_probe:
            otlp_headers["x-neatlogs-doctor"] = "v1"
        # Always send traces to the OTLP traces endpoint for the configured base URL.
        otlp_exporter = _doctor_probe_exporter or OTLPSpanExporter(
            endpoint=traces_endpoint,
            headers=otlp_headers,
            compression=Compression.Gzip,
            session=build_otlp_session(),
        )
        # Wrap the exporter so rootless infra-HTTP auto-spans (boot pings, dependency
        # warmups, outbound fetches outside any traced request) are never sent — on
        # their own they're junk rootless traces the backend can't simplify. Nested
        # HTTP spans (with a parent) still export normally.
        from .core.masking_exporter import MaskingSpanExporter
        from .core.span_processor import is_rootless_infra_http

        class _FilteredOTLPExporter:
            def __init__(self, inner):
                self._inner = inner

            def export(self, spans):
                from opentelemetry.sdk.trace.export import SpanExportResult

                kept = [s for s in spans if not is_rootless_infra_http(s)]
                if not kept:
                    return SpanExportResult.SUCCESS
                return self._inner.export(kept)

            def shutdown(self):
                return self._inner.shutdown()

            def force_flush(self, timeout_millis: int = 30000):
                return self._inner.force_flush(timeout_millis)

        limited_span_exporter = ByteLimitedSpanExporter(
            otlp_exporter,
            diagnostics=_delivery_diagnostics,
            upload_authority=_upload_authority,
        )
        limited_span_exporter = TypedMediaSpanExporter(
            limited_span_exporter,
            _upload_authority,
            _media_store,
            diagnostics=_delivery_diagnostics,
        )
        batch_processor = ObservableBatchSpanProcessor(
            _FilteredOTLPExporter(
                MaskingSpanExporter(
                    limited_span_exporter,
                    mask,
                    diagnostics=_delivery_diagnostics,
                    media_store=_media_store,
                    doctor_capture=_doctor_probe,
                )
            ),
            max_export_batch_size=batch_size,
            max_queue_size=export_queue_capacity(batch_size),
            schedule_delay_millis=int(flush_interval * 1000),
            diagnostics=_delivery_diagnostics,
        )
        provider.add_span_processor(batch_processor)
        # Registered after the batch processor so a root is queued for export
        # before the completion marker that triggers backend finalization.
        completion_processor = CompletionMarkerSpanProcessor(
            _span_processor,
            provider.get_tracer("neatlogs.internal"),
        )
        provider.add_span_processor(completion_processor)
        global _transport_span_processors
        _transport_span_processors = [batch_processor, completion_processor]
        global _completion_span_processor
        _completion_span_processor = completion_processor
        if debug:
            logger.debug(f"OTLP trace exporter configured: {traces_endpoint}")
    elif debug:
        logger.debug("Export disabled — spans will not be sent to backend")

    if debug:
        logger.debug("Neatlogs tracer provider initialized")

    # --- Logs signal (opt-in) ---
    # neatlogs.log(), capture_stdout=True, and logging.* auto-capture all require
    # capture_logs=True. When False, nothing is captured as LOG spans.
    global _log_provider
    if capture_logs:
        from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter

        from .core.log_exporter import NeatlogsLogFilter
        from .core.masking_exporter import MaskingLogExporter

        logs_endpoint = f"{_base_url}/v1/logs"
        _otlp_log_exporter = OTLPLogExporter(
            endpoint=logs_endpoint,
            headers={"x-api-key": resolved_key},
            compression=Compression.Gzip,
            session=build_otlp_session(),
        )
        _log_provider = LoggerProvider(resource=resource)
        # NeatlogsLogFilter drops external-module and no-trace records before
        # BatchLogRecordProcessor batches and sends them via OTLPLogExporter.
        limited_log_exporter = ByteLimitedLogExporter(
            _otlp_log_exporter,
            diagnostics=_delivery_diagnostics,
            upload_authority=_upload_authority,
        )
        limited_log_exporter = TypedMediaLogExporter(
            limited_log_exporter,
            _upload_authority,
            _media_store,
            diagnostics=_delivery_diagnostics,
        )
        _log_provider.add_log_record_processor(
            NeatlogsLogFilter(
                ObservableBatchLogRecordProcessor(
                    MaskingLogExporter(
                        limited_log_exporter,
                        mask,
                        diagnostics=_delivery_diagnostics,
                        media_store=_media_store,
                    ),
                    max_export_batch_size=batch_size,
                    max_queue_size=export_queue_capacity(batch_size),
                    schedule_delay_millis=int(flush_interval * 1000),
                    diagnostics=_delivery_diagnostics,
                )
            )
        )
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
                    f"(logging.* at {log_level.upper()}+, endpoint: {logs_endpoint})"
                )
        except ImportError:
            if debug:
                logger.debug(
                    "opentelemetry-instrumentation-logging not installed — "
                    "Install with: pip install opentelemetry-instrumentation-logging"
                )
    elif debug:
        logger.debug("Log capture disabled (pass capture_logs=True to enable)")

    global _instrumentation_manager
    manager = InstrumentationManager(
        provider=provider,
        debug=debug,
        excluded_urls=endpoint,
    )
    _instrumentation_manager = manager

    manager.instrument_threading()
    manager.instrument_http()

    if instrumentations:
        manager.instrument(libraries=instrumentations)
        if debug:
            logger.debug(f"Instrumented libraries: {manager.instrumented}")

    _shutdown_worker = DeadlineWorker("neatlogs-default-shutdown")
    atexit.register(_atexit_shutdown)
    _init_signature = candidate_signature
    _initialized = True
    if register_shutdown_handlers:
        _register_shutdown_signal_handlers()

    if debug:
        logger.info("Neatlogs SDK initialized successfully")
        logger.info(f"Endpoint: {endpoint}")
        logger.info(f"Workflow: {resolved_workflow_name}")
        logger.info(f"User: {user_id or '(none)'}")
        logger.info(f"Tags: {tags or []}")
        logger.info(f"Instrumentations: {manager.instrumented or '(none)'}")
        logger.info(f"Sample Rate: {sample_rate}")


def flush(timeout_millis: int = 30000) -> bool:
    """Flush the process-default Neatlogs pipeline only."""
    global _tracer_provider
    success = True

    # Log provider must flush BEFORE tracer provider: the tracer batch includes
    # the neatlogs.trace.complete marker which triggers server-side finalization.
    # Flushing logs first guarantees LOG records reach ClickHouse before the
    # completion marker fires the trace-finalizer query.
    if _log_provider:
        try:
            logger.debug("Flushing log provider...")
            _log_provider.force_flush(timeout_millis=timeout_millis)
            logger.debug("Log provider flushed successfully")
        except Exception as e:
            logger.error(f"Error flushing logs: {e}", exc_info=True)
            success = False

    for processor in _transport_span_processors:
        try:
            logger.debug("Flushing Neatlogs trace processor...")
            ok = processor.force_flush(timeout_millis=timeout_millis)
            success = (ok is None or bool(ok)) and success
        except Exception as e:
            logger.error(f"Error flushing spans: {e}", exc_info=True)
            success = False

    return success


def get_log_provider() -> Optional[LoggerProvider]:
    """Return the SDK-private default log provider, when log capture is enabled."""
    return _log_provider


def get_delivery_diagnostics() -> Dict[str, Any]:
    """Return loss counters for the current or most recently closed pipeline."""

    return _delivery_diagnostics.snapshot()


def flush_all(timeout_millis: int = 30000) -> Dict[str, bool]:
    """Flush every live Neatlogs pipeline concurrently under one deadline.

    Only processors/exporters installed by Neatlogs are flushed. The process-global
    OpenTelemetry provider and caller/foreign processors are never touched.
    """
    from .core.client_registry import snapshot_clients

    clients = snapshot_clients()
    operations = [("default", flush)] if _initialized else []
    operations.extend(
        (f"client:{client.workflow_name}:{id(client):x}", client.flush) for client in clients
    )
    if not operations:
        return {}

    results: Dict[str, bool] = {name: False for name, _ in operations}
    timeout_seconds = max(0, timeout_millis) / 1000
    deadline = time.monotonic() + timeout_seconds
    completed: queue.Queue[tuple[str, bool]] = queue.Queue()

    def run(name: str, operation: Callable[[int], bool]) -> None:
        try:
            completed.put((name, bool(operation(timeout_millis))))
        except BaseException:
            completed.put((name, False))

    if timeout_seconds <= 0:
        return results
    for name, operation in operations:
        threading.Thread(
            target=run,
            args=(name, operation),
            name="neatlogs-flush",
            daemon=True,
        ).start()

    remaining = len(operations)
    while remaining and time.monotonic() < deadline:
        try:
            name, success = completed.get(timeout=max(0.001, deadline - time.monotonic()))
        except queue.Empty:
            break
        results[name] = success
        remaining -= 1
    return results


def get_session_config():
    """Get the current session configuration (session_id, user_id)."""
    return _session_config.copy()


def _atexit_shutdown() -> None:
    """Synchronous interpreter-exit cleanup (Python 3.12 forbids new threads)."""

    shutdown(_synchronous=True)


def shutdown(
    timeout_millis: int = 30000,
    termination_reason: str = "shutdown",
    *,
    _synchronous: bool = False,
) -> bool:
    """Run one shutdown at a time and make same-thread re-entry non-blocking."""
    global _shutdown_state, _shutdown_owner, _shutdown_result, _shutdown_worker
    deadline = time.monotonic() + max(0, timeout_millis) / 1000
    current_thread = threading.get_ident()
    with _shutdown_condition:
        if _shutdown_worker is None:
            if _synchronous:
                return False
            _shutdown_worker = DeadlineWorker("neatlogs-default-shutdown")
        if _shutdown_state == "closing":
            # end_active_spans() invokes processors synchronously. If one of
            # those callbacks re-enters shutdown on this thread, waiting here
            # would deadlock the original shutdown.
            if _shutdown_owner == current_thread or (
                _shutdown_worker is not None and _shutdown_worker.is_current()
            ):
                return _shutdown_result
            completed = _shutdown_condition.wait_for(
                lambda: _shutdown_state == "idle",
                timeout=max(0.0, deadline - time.monotonic()),
            )
            return _shutdown_result if completed else False
        _shutdown_state = "closing"
        _shutdown_owner = current_thread

    acquired = _lifecycle_operation_lock.acquire(timeout=max(0.0, deadline - time.monotonic()))
    if not acquired:
        with _shutdown_condition:
            _shutdown_result = False
            _shutdown_state = "idle"
            _shutdown_owner = None
            _shutdown_condition.notify_all()
        return False
    shutdown_worker = _shutdown_worker
    try:
        try:
            result = _perform_shutdown(
                max(0, int((deadline - time.monotonic()) * 1000)),
                termination_reason,
                synchronous=_synchronous,
            )
        except BaseException:
            with _shutdown_condition:
                _shutdown_result = False
            raise
        else:
            with _shutdown_condition:
                _shutdown_result = result
            return result
    finally:
        if shutdown_worker is not None:
            shutdown_worker.close()
        if _shutdown_worker is shutdown_worker:
            _shutdown_worker = None
        _lifecycle_operation_lock.release()
        with _shutdown_condition:
            _shutdown_state = "idle"
            _shutdown_owner = None
            _shutdown_condition.notify_all()


def _perform_shutdown(
    timeout_millis: int,
    termination_reason: str,
    *,
    synchronous: bool = False,
) -> bool:
    """End active Neatlogs spans, then flush and shut down SDK providers."""
    global _tracer_provider, _owns_tracer_provider, _log_provider, _span_processor, _initialized
    global _init_signature
    global _instrumentation_manager, _transport_span_processors, _completion_span_processor
    global _debug_mode, _shutdown_worker
    global _upload_authority, _media_store

    try:
        atexit.unregister(_atexit_shutdown)
    except Exception:
        pass

    _restore_shutdown_signal_handlers()

    success = True
    deadline = time.monotonic() + max(0, timeout_millis) / 1000
    log_provider = _log_provider
    tracer_provider = _tracer_provider
    owns_tracer_provider = _owns_tracer_provider
    span_processor = _span_processor
    completion_span_processor = _completion_span_processor
    transport_span_processors = list(_transport_span_processors)
    instrumentation_manager = _instrumentation_manager
    shutdown_worker = _shutdown_worker
    from ._wrap_utils import take_bootstrap_resources

    bootstrap_provider, bootstrap_authority, bootstrap_media_store = take_bootstrap_resources()
    if tracer_provider is None and bootstrap_provider is not None:
        tracer_provider = bootstrap_provider
        owns_tracer_provider = True
    if completion_span_processor is not None:
        completion_span_processor.begin_shutdown()
    if span_processor is not None:
        span_processor.begin_shutdown(termination_reason)

    # LOG records must drain before trace completion. The tracer batch contains
    # neatlogs.trace.complete, which can trigger backend finalization as soon as
    # it is ingested.
    if log_provider:
        logger.debug("Shutting down log provider...")
        completed, result = bounded_call(
            log_provider.shutdown,
            deadline,
            synchronous=synchronous,
            worker=shutdown_worker,
        )
        if not completed:
            logger.error("Log provider shutdown failed or timed out: %s", result)
            success = False
        else:
            success = (result is None or bool(result)) and success
            logger.debug("Log provider shut down successfully")

    # Root end creates the completion marker, so it must happen only after all
    # buffered LOG records have drained.
    if span_processor:
        completed, result = bounded_call(
            lambda: span_processor.end_active_spans(termination_reason),
            deadline,
            synchronous=synchronous,
            worker=shutdown_worker,
        )
        if completed:
            ended = result
            if ended:
                logger.info(f"Ended {ended} active Neatlogs span(s) during {termination_reason}")
        else:
            logger.warning("Ending active Neatlogs spans failed or timed out: %s", result)
            success = False
        completed, result = bounded_call(
            span_processor._log_performance_stats,
            deadline,
            synchronous=synchronous,
            worker=shutdown_worker,
        )
        if not completed:
            logger.warning("Logging performance stats failed or timed out: %s", result)
            success = False
        remaining_millis = max(0, int((deadline - time.monotonic()) * 1000))
        if not span_processor.wait_for_downstream(remaining_millis):
            logger.warning("Timed out waiting for ending spans to reach the export queue")
            success = False
    if completion_span_processor is not None:
        completed, result = bounded_call(
            completion_span_processor.emit_deferred,
            deadline,
            synchronous=synchronous,
            worker=shutdown_worker,
        )
        if not completed:
            logger.warning("Completion marker emission failed or timed out: %s", result)
            success = False

    if tracer_provider:
        if owns_tracer_provider:
            completed, result = bounded_call(
                tracer_provider.shutdown,
                deadline,
                synchronous=synchronous,
                worker=shutdown_worker,
            )
            if completed:
                # neatlogs created this provider → safe to fully shut down.
                success = (result is None or bool(result)) and success
                logger.debug("Tracer provider shut down successfully")
            else:
                logger.error("Tracer provider shutdown failed or timed out: %s", result)
                success = False
        else:
            completed, result = bounded_call(
                lambda: tracer_provider.force_flush(
                    timeout_millis=max(0, int((deadline - time.monotonic()) * 1000))
                ),
                deadline,
                synchronous=synchronous,
                worker=shutdown_worker,
            )
            if completed:
                # Provider is shared (host app / Langfuse / another SDK set it). Calling
                # provider.shutdown() would tear down EVERY processor on it — including the
                # co-tenant's exporter — silently killing their telemetry. Only flush our
                # own spans and shut down JUST the neatlogs span processor, leaving the
                # shared provider and other exporters intact.
                logger.debug("Shared tracer provider — flushing without shutting it down")
                success = (result is None or bool(result)) and success
            else:
                logger.error("Tracer provider flush failed or timed out: %s", result)
                success = False
            for processor in reversed(transport_span_processors):
                completed, result = bounded_call(
                    processor.shutdown,
                    deadline,
                    synchronous=synchronous,
                    worker=shutdown_worker,
                )
                if not completed:
                    logger.warning("Neatlogs transport shutdown failed or timed out: %s", result)
                    success = False
            if span_processor is not None:
                completed, result = bounded_call(
                    span_processor.shutdown,
                    deadline,
                    synchronous=synchronous,
                    worker=shutdown_worker,
                )
                if not completed:
                    logger.warning(
                        "Neatlogs span processor shutdown failed or timed out: %s", result
                    )
                    success = False

    # Wrapper-only auto-bootstrap owns a distinct provider. If regular init()
    # subsequently installed another provider, both pipelines still need a
    # bounded shutdown; never strand the earlier wrapper exporter/session.
    if bootstrap_provider is not None and bootstrap_provider is not tracer_provider:
        completed, result = bounded_call(
            bootstrap_provider.shutdown,
            deadline,
            synchronous=synchronous,
            worker=shutdown_worker,
        )
        if completed:
            success = (result is None or bool(result)) and success
        else:
            logger.error("Wrapper bootstrap shutdown failed or timed out: %s", result)
            success = False

    def uninstrument_logging() -> None:
        # Only undo the logging patch NeatLogs installed. Touching an
        # unconfigured or foreign instrumentor violates provider ownership.
        if log_provider is not None:
            try:
                from opentelemetry.instrumentation.logging import LoggingInstrumentor

                LoggingInstrumentor().uninstrument()
            except Exception:
                pass

    completed, result = bounded_call(
        uninstrument_logging,
        deadline,
        synchronous=synchronous,
        worker=shutdown_worker,
    )
    if not completed:
        logger.debug("Logging uninstrument timed out: %s", result)
        success = False

    # Reverse the framework/provider instrumentation so a later init() (or a test
    # re-init with a fresh TracerProvider) rebinds cleanly instead of leaving the
    # old instrumentor pointing at the now-dead provider.
    if instrumentation_manager is not None:
        completed, result = bounded_call(
            instrumentation_manager.uninstrument_all,
            deadline,
            synchronous=synchronous,
            worker=shutdown_worker,
        )
        if not completed:
            logger.debug("Instrumentation cleanup failed or timed out: %s", result)
            success = False
        _instrumentation_manager = None

    # Drop the cached wrapper tracer (used by wrap()/trace processors like the
    # OpenAI Agents one) so the next init() rebinds it to the new provider. Also
    # clear the registered neatlogs provider so a later init() starts clean.
    try:
        from ._wrap_utils import reset_tracer, set_neatlogs_provider

        reset_tracer()
        set_neatlogs_provider(None)
    except Exception:
        pass

    set_default_media_store(None)
    if _media_store is not None:
        _media_store.clear()
    if bootstrap_media_store is not None:
        bootstrap_media_store.clear()
    close_uploads = getattr(_upload_authority, "close", None)
    if callable(close_uploads):
        close_uploads()
    close_bootstrap_uploads = getattr(bootstrap_authority, "close", None)
    if callable(close_bootstrap_uploads):
        close_bootstrap_uploads()
    _media_store = None
    _upload_authority = None

    try:
        from .prompt.client import _close_shared_prompt_clients

        _close_shared_prompt_clients()
    except Exception:
        success = False

    _initialized = False
    _init_signature = None
    _tracer_provider = None
    _owns_tracer_provider = False
    _log_provider = None
    _span_processor = None
    _transport_span_processors = []
    _completion_span_processor = None
    _debug_mode = False
    _session_config["session_id"] = None
    _session_config["user_id"] = None
    _session_config["workflow_name"] = None
    _session_config["_api_key"] = None
    _session_config["_base_url"] = None

    logger.info("Neatlogs SDK shutdown complete")
    return success
