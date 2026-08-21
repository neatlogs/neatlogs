"""
Neatlogs SDK.
"""

import atexit
import functools
import os
import re
import signal
import sys
import threading
from typing import Any, Callable, Dict, List, Optional

try:
    from opentelemetry import logs
except ImportError:
    from opentelemetry import _logs as logs  # type: ignore[no-redef]

from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.environment_variables import (
    OTEL_ATTRIBUTE_COUNT_LIMIT,
    OTEL_SPAN_ATTRIBUTE_COUNT_LIMIT,
)
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import SpanLimits, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.trace.sampling import TraceIdRatioBased

from ._wrap_utils import _normalize_traces_endpoint
from .core.logger import get_logger
from .core.span_processor import CompletionMarkerSpanProcessor, NeatlogsSpanProcessor
from .instrumentation.manager import InstrumentationManager
from .version import __version__

logger = get_logger()


_initialized = False
_tracer_provider = None
_owns_tracer_provider = False  # True only when neatlogs created the provider (safe to shut down)
_isolated_provider = False  # True when isolated (explicit tracer_provider= OR auto-detected)

# Foreign LLM-observability instrumentors that own the global OTel pipeline and
# resolve their span parent from the GLOBAL current-span. If one is active while
# neatlogs would otherwise reuse the global provider, spans cross-contaminate both
# ways. Their presence is the auto-detect signal to isolate onto a private provider.
#
# NOTE: `openinference` is deliberately NOT listed. The bare
# `openinference-instrumentation-*` packages are neatlogs' OWN instrumentation
# backend (hard dependency) — neatlogs imports them itself the moment it
# instruments anything, and they are always importable. So their presence in
# sys.modules / on disk carries no information about a co-tenant and would cause
# a false-positive isolation (spurious for every standalone user, and — because
# the modules stay resident after shutdown — for every re-init in one process).
# Genuine OpenInference-based co-tenants (Arize Phoenix, Arize) import `phoenix`/
# `arize` and are detected via those entries below.
_FOREIGN_LLM_INSTRUMENTORS = (
    "openlit",
    "langfuse",
    "traceloop",  # traceloop-sdk / openllmetry
    "phoenix",  # arize-phoenix (uses OpenInference under the hood)
    "arize",
    "logfire",
)


def _foreign_llm_instrumentor_active():
    """Return the name of a loaded foreign LLM-observability instrumentor, or None.

    Import presence in ``sys.modules`` is the signal: these packages install their
    instrumentation onto the global provider at import/init time, so being loaded
    in-process means their spans are (or will be) flowing through the global
    pipeline neatlogs would otherwise share.
    """
    for name in _FOREIGN_LLM_INSTRUMENTORS:
        if name in sys.modules:
            return name
    return None


def _foreign_llm_instrumentor_installed():
    """Return the name of an INSTALLED (importable) foreign LLM instrumentor, or None.

    Unlike ``_foreign_llm_instrumentor_active`` (which requires the package to be
    imported already), this only checks that it is importable. That is the signal
    we need when neatlogs is the FIRST tracing SDK to init: the foreign tool
    (openlit/langfuse/…) frequently loads LATER — e.g. in a FastAPI startup event
    that runs after an import-time ``neatlogs.init()``. If neatlogs claimed the
    global provider now, that later instrumentor would bind to OUR provider and
    both pipelines would cross-contaminate. Detecting the package on-disk lets us
    pre-emptively stay private and leave the global slot free for the co-tenant,
    WITHOUT requiring the customer to pass ``isolate=True``.

    A user who has a foreign package installed but genuinely wants neatlogs to own
    the global provider (e.g. to capture their own raw-OTel spans) can force the
    legacy behaviour with ``isolate=False``.
    """
    import importlib.util

    for name in _FOREIGN_LLM_INSTRUMENTORS:
        try:
            if importlib.util.find_spec(name) is not None:
                return name
        except (ImportError, ValueError):
            # find_spec can raise for namespace-package edge cases; treat as absent.
            continue
    return None


_meter_provider = None
_log_provider = None
_span_processor = None
_transport_span_processors = []
_completion_span_processor = None
_instrumentation_manager = None
_debug_mode = False
_signal_handlers = {}
_signal_shutdown_in_progress = False
_shutdown_condition = threading.Condition(threading.RLock())
_shutdown_state = "idle"
_shutdown_owner = None
_shutdown_result = True
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


@_serialize_init
def init(
    api_key: Optional[str] = None,
    endpoint: str = "https://ingest.neatlogs.com",
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
        tracer_provider: Opt-in FULL ISOLATION. Pass a private ``TracerProvider`` (one you
              created but did NOT install as the OTel global) and neatlogs will emit ALL of
              its spans — auto-instrumented, wrap()/trace()/@span, and the internal
              completion marker — into that provider ONLY. Use this when another tool
              already owns the global provider (e.g. your own OpenTelemetry setup exporting
              to Langfuse) and you need neatlogs and that tool to share NO pipeline: no
              neatlogs span reaches the other backend and no foreign span reaches neatlogs.
              neatlogs never shuts this provider down and never claims the global
              meter/logger providers. When None (default), behaviour is auto-detected
              (see ``isolate``): neatlogs isolates automatically when a foreign LLM
              instrumentor owns the global provider, and otherwise owns-or-reuses it.
        isolate: Override the auto-isolation decision. When None (default), neatlogs
              AUTO-DETECTS and isolates onto a private provider — with NO customer
              code change — in either of these cases:
                * a foreign LLM-observability instrumentor
                  (openlit/langfuse/traceloop/openinference/…) already owns the
                  global tracer provider (it loaded first), OR
                * neatlogs is the first tracing SDK to load but a foreign
                  instrumentor is INSTALLED (importable) and may attach to the
                  global provider later (the common FastAPI-startup ordering, where
                  neatlogs.init runs at import time and openlit.init runs in a
                  startup event).
              In both cases neatlogs routes ALL its spans through a private provider
              so the two pipelines share nothing — no neatlogs span reaches the
              foreign backend, no foreign span reaches neatlogs, and no foreign span
              inherits a neatlogs parent. When NO foreign instrumentor is installed,
              neatlogs owns-or-reuses the global provider exactly as before (standalone
              behaviour is unchanged). Pass True to force isolation unconditionally, or
              False to force the legacy own-or-reuse behaviour even when a foreign
              instrumentor is present. Passing ``tracer_provider=`` implies isolation
              regardless of this flag.
        register_shutdown_handlers: Register SIGINT and SIGTERM handlers that end
              active Neatlogs spans child-first and flush before preserving normal
              signal termination. Defaults to True. Set False only when the host
              application owns signal handling and calls ``neatlogs.shutdown()``.
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

    global _session_config
    _session_config["user_id"] = user_id
    _session_config["workflow_name"] = resolved_workflow_name
    _session_config["_api_key"] = resolved_key
    _session_config["_base_url"] = _base_url

    # Session and end-user identity are deliberately NOT resource attributes:
    # they are per-request, set at the trace root (trace()/@span) or via
    # neatlogs.identify(). Only the operator user.id is process-global here.
    resource_attrs = {
        SERVICE_NAME: workflow_name or "neatlogs-app",
        "service.version": __version__,
        "neatlogs.workflow_name": resolved_workflow_name,
    }
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

    global _tracer_provider, _owns_tracer_provider, _isolated_provider
    existing_provider = trace.get_tracer_provider()

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
        _isolated_provider = True
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
    elif existing_provider and hasattr(existing_provider, "add_span_processor"):
        # A real provider already owns the global pipeline. Two sub-cases:
        _foreign = None if isolate is False else _foreign_llm_instrumentor_active()
        if isolate is True or _foreign is not None:
            # AUTO-DETECT ISOLATION. A foreign LLM-observability instrumentor
            # (openlit/langfuse/traceloop/…) owns the global provider AND resolves
            # its parent from the global current-span. Reusing that provider would
            # cross-contaminate both pipelines: neatlogs spans export to the foreign
            # backend, and foreign spans (a) export to neatlogs and (b) inherit a
            # neatlogs parent. So we build a PRIVATE provider neatlogs never installs
            # globally — identical semantics to an explicit tracer_provider=.
            provider = TracerProvider(
                resource=resource,
                span_limits=_span_limits_for_capture_everything(),
            )
            _owns_tracer_provider = True  # we made it; safe to shut down
            _isolated_provider = True
            if debug:
                _why = (
                    f"detected foreign LLM instrumentor '{_foreign}'"
                    if _foreign is not None
                    else "isolate=True was requested"
                )
                logger.info(
                    f"Auto-isolation engaged: {_why} owning the global tracer "
                    f"provider — routing all neatlogs spans through a private "
                    f"provider so the two pipelines share nothing."
                )
        else:
            # Reuse a provider set by the host app / another SDK (plain OpenTelemetry
            # with no foreign LLM instrumentor). We do NOT own it — shutdown() must
            # never tear it down, or it would kill the co-tenant's exporter too.
            provider = existing_provider
            _owns_tracer_provider = False
            _isolated_provider = False
            if debug:
                logger.debug("Using existing tracer provider (not owned by neatlogs)")
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
        _owns_tracer_provider = True

        # neatlogs is the first tracing SDK to load (no real global provider yet).
        # We must still decide whether to claim the global slot or stay private. A
        # foreign LLM instrumentor (openlit/langfuse/…) frequently attaches to the
        # global provider LATER — e.g. in a FastAPI startup event that runs after an
        # import-time neatlogs.init(). If we claimed the global now, that later
        # instrumentor would bind to OUR provider and both pipelines would
        # cross-contaminate. So when isolation is requested OR a foreign instrumentor
        # is merely INSTALLED (importable), we pre-emptively keep a private provider
        # and leave the global slot free for the co-tenant to own — no isolate=True
        # required from the customer.
        _foreign_installed = None if isolate is False else _foreign_llm_instrumentor_installed()
        if isolate is True or _foreign_installed is not None:
            _isolated_provider = True
            if debug:
                _why = (
                    "isolate=True was requested"
                    if isolate is True
                    else f"foreign LLM instrumentor '{_foreign_installed}' is installed "
                    f"(may attach to the global provider later)"
                )
                logger.info(
                    f"Pre-emptive isolation: {_why} — neatlogs loaded first but keeps "
                    f"a PRIVATE tracer provider and leaves the global slot free for the "
                    f"co-tenant so the two pipelines share nothing."
                )
        else:
            trace.set_tracer_provider(provider)
            _isolated_provider = False
            if debug:
                logger.debug("Created new tracer provider (owned by neatlogs)")

    _tracer_provider = provider

    # Register this provider as the single source of truth for every neatlogs
    # tracer (wrap()/trace()/@span + completion marker). In isolated mode this is
    # the private provider, so neatlogs' own spans never reach the global pipeline.
    from ._wrap_utils import set_neatlogs_provider

    set_neatlogs_provider(provider)

    # NeatlogsSpanProcessor: pure pre-processing (attribute normalization + file logging)
    global _span_processor
    _span_processor = NeatlogsSpanProcessor(
        sample_rate=sample_rate,
        debug=debug,
        mask=mask,
        emit_completion_markers=False,
        # A private/isolated pipeline contains only Neatlogs execution spans,
        # even when the caller retains provider shutdown ownership.
        own_all_spans=_owns_tracer_provider or _isolated_provider,
    )
    provider.add_span_processor(_span_processor)

    # BatchSpanProcessor + OTLPSpanExporter: standard transport
    if not disable_export_resolved:
        otlp_headers = {"x-api-key": resolved_key}
        # Always send traces to the OTLP traces endpoint for the configured base URL.
        otlp_exporter = OTLPSpanExporter(
            endpoint=traces_endpoint,
            headers=otlp_headers,
        )
        # Wrap the exporter so rootless infra-HTTP auto-spans (boot pings, dependency
        # warmups, outbound fetches outside any traced request) are never sent — on
        # their own they're junk rootless traces the backend can't simplify. Nested
        # HTTP spans (with a parent) still export normally.
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

        batch_processor = BatchSpanProcessor(
            _FilteredOTLPExporter(otlp_exporter),
            max_export_batch_size=batch_size,
            schedule_delay_millis=int(flush_interval * 1000),
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

    global _meter_provider
    _meter_provider = MeterProvider(resource=resource)
    # In isolated mode, never claim the GLOBAL meter provider — that would clobber
    # a co-tenant's (e.g. their OpenTelemetry/Langfuse) meter pipeline. neatlogs
    # currently emits no metrics of its own, so a non-global provider is harmless.
    if not _isolated_provider:
        metrics.set_meter_provider(_meter_provider)
        if debug:
            logger.debug("Neatlogs meter provider initialized")
    elif debug:
        logger.debug("Isolated mode: skipping global meter provider registration")

    # --- Logs signal (opt-in) ---
    # neatlogs.log(), capture_stdout=True, and logging.* auto-capture all require
    # capture_logs=True. When False, nothing is captured as LOG spans.
    global _log_provider
    if capture_logs:
        from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter

        from .core.log_exporter import NeatlogsLogFilter

        logs_endpoint = f"{_base_url}/v1/logs"
        _otlp_log_exporter = OTLPLogExporter(
            endpoint=logs_endpoint,
            headers={"x-api-key": resolved_key},
        )
        _log_provider = LoggerProvider(resource=resource)
        # NeatlogsLogFilter drops external-module and no-trace records before
        # BatchLogRecordProcessor batches and sends them via OTLPLogExporter.
        _log_provider.add_log_record_processor(
            NeatlogsLogFilter(BatchLogRecordProcessor(_otlp_log_exporter))
        )
        # Isolated mode: don't claim the GLOBAL logger provider (would clobber a
        # co-tenant's log pipeline). LoggingInstrumentor below is still bound to
        # our _log_provider explicitly via logger_provider=, so capture still works.
        if not _isolated_provider:
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

    atexit.register(shutdown)
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
    """Flush all pending spans and metrics."""
    global _tracer_provider, _meter_provider
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


def shutdown(timeout_millis: int = 30000, termination_reason: str = "shutdown") -> bool:
    """Run one shutdown at a time and make same-thread re-entry non-blocking."""
    global _shutdown_state, _shutdown_owner, _shutdown_result
    current_thread = threading.get_ident()
    with _shutdown_condition:
        if _shutdown_state == "closing":
            # end_active_spans() invokes processors synchronously. If one of
            # those callbacks re-enters shutdown on this thread, waiting here
            # would deadlock the original shutdown.
            if _shutdown_owner == current_thread:
                return _shutdown_result
            _shutdown_condition.wait_for(lambda: _shutdown_state == "idle")
            return _shutdown_result
        _shutdown_state = "closing"
        _shutdown_owner = current_thread

    try:
        with _lifecycle_operation_lock:
            result = _perform_shutdown(timeout_millis, termination_reason)
    except BaseException:
        with _shutdown_condition:
            _shutdown_result = False
        raise
    else:
        with _shutdown_condition:
            _shutdown_result = result
        return result
    finally:
        with _shutdown_condition:
            _shutdown_state = "idle"
            _shutdown_owner = None
            _shutdown_condition.notify_all()


def _perform_shutdown(timeout_millis: int, termination_reason: str) -> bool:
    """End active Neatlogs spans, then flush and shut down SDK providers."""
    global _tracer_provider, _owns_tracer_provider, _meter_provider, _log_provider, _span_processor, _initialized
    global _instrumentation_manager, _transport_span_processors, _completion_span_processor

    try:
        atexit.unregister(shutdown)
    except Exception:
        pass

    _restore_shutdown_signal_handlers()

    success = True
    if _completion_span_processor is not None:
        _completion_span_processor.begin_shutdown()
    if _span_processor is not None:
        _span_processor.begin_shutdown(termination_reason)

    # LOG records must drain before trace completion. The tracer batch contains
    # neatlogs.trace.complete, which can trigger backend finalization as soon as
    # it is ingested.
    if _log_provider:
        try:
            logger.debug("Shutting down log provider...")
            ok = _log_provider.shutdown()
            success = (ok is None or bool(ok)) and success
            logger.debug("Log provider shut down successfully")
        except Exception as e:
            logger.error(f"Error shutting down log provider: {e}", exc_info=True)
            success = False

    # Root end creates the completion marker, so it must happen only after all
    # buffered LOG records have drained.
    if _span_processor:
        try:
            ended = _span_processor.end_active_spans(termination_reason)
            if ended:
                logger.info(f"Ended {ended} active Neatlogs span(s) during {termination_reason}")
            _span_processor._log_performance_stats()
        except Exception as e:
            logger.warning(f"Error logging performance stats: {e}")
        if not _span_processor.wait_for_downstream(timeout_millis):
            logger.warning("Timed out waiting for ending spans to reach the export queue")
            success = False
    if _completion_span_processor is not None:
        _completion_span_processor.emit_deferred()

    if _tracer_provider:
        try:
            if _owns_tracer_provider:
                # neatlogs created this provider → safe to fully shut down.
                logger.debug("Shutting down tracer provider (owned by neatlogs)...")
                ok = _tracer_provider.shutdown()
                success = (ok is None or bool(ok)) and success
                logger.debug("Tracer provider shut down successfully")
            else:
                # Provider is shared (host app / Langfuse / another SDK set it). Calling
                # provider.shutdown() would tear down EVERY processor on it — including the
                # co-tenant's exporter — silently killing their telemetry. Only flush our
                # own spans and shut down JUST the neatlogs span processor, leaving the
                # shared provider and other exporters intact.
                logger.debug("Shared tracer provider — flushing without shutting it down")
                ok = _tracer_provider.force_flush(timeout_millis=timeout_millis)
                success = (ok is None or bool(ok)) and success
                for processor in reversed(_transport_span_processors):
                    try:
                        processor.shutdown()
                    except Exception as e:
                        logger.warning(f"Error shutting down Neatlogs transport: {e}")
                        success = False
                if _span_processor is not None:
                    try:
                        _span_processor.shutdown()
                    except Exception as e:
                        logger.warning(f"Error shutting down neatlogs span processor: {e}")
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

    try:
        from opentelemetry.instrumentation.logging import LoggingInstrumentor

        LoggingInstrumentor().uninstrument()
    except Exception:
        pass

    # Reverse the framework/provider instrumentation so a later init() (or a test
    # re-init with a fresh TracerProvider) rebinds cleanly instead of leaving the
    # old instrumentor pointing at the now-dead provider.
    if _instrumentation_manager is not None:
        try:
            _instrumentation_manager.uninstrument_all()
        except Exception as e:
            logger.debug(f"Error uninstrumenting libraries: {e}")
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

    global _isolated_provider
    _isolated_provider = False
    _initialized = False
    _tracer_provider = None
    _meter_provider = None
    _log_provider = None
    _span_processor = None
    _transport_span_processors = []
    _completion_span_processor = None
    _debug_mode = False
    _session_config["session_id"] = None
    _session_config["user_id"] = None
    _session_config["workflow_name"] = None

    logger.info("Neatlogs SDK shutdown complete")
    return success
