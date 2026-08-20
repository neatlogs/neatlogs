"""
Shared infrastructure for Neatlogs provider wrappers.

Only contains truly shared concerns:
  - private TracerProvider bootstrap (auto from env or reuse from init())
  - configure() for wrapper-only mode
  - Sync/async stream wrapper classes
  - Safe JSON serialization
"""

import inspect
import json
import os
import time
from contextvars import ContextVar
from typing import Any, Callable, Dict, List, Mapping, Optional
from urllib.parse import urlparse

from .constants import DEFAULT_INGEST_ENDPOINT

from opentelemetry import context as context_api
from opentelemetry import trace as otel_trace
from opentelemetry.sdk.trace import TracerProvider

from .core.logger import get_logger

logger = get_logger()

_wrapper_tracer: Optional[otel_trace.Tracer] = None
_wrapper_bootstrapped = False
_bootstrap_warned = False

# Single source of truth for the private provider Neatlogs emits into. Every
# Neatlogs tracer resolves from it; the process-global provider is never adopted.
_neatlogs_provider: Optional[TracerProvider] = None

# Optional secondary Client selected only for the current async/thread context.
# The process-wide provider configured by neatlogs.init() remains the default.
_active_client: ContextVar[Optional[Any]] = ContextVar("neatlogs.active_client", default=None)

# True when Neatlogs has its required private provider. Neatlogs threads parents
# through a private context key so co-tenant instrumentors never inherit them.
_isolated: bool = False


def set_neatlogs_provider(provider: Optional[TracerProvider]) -> None:
    """Register the provider neatlogs must use for all its own spans."""
    global _neatlogs_provider, _wrapper_tracer, _isolated
    _neatlogs_provider = provider
    _wrapper_tracer = None  # force get_tracer() to rebind to the new provider
    try:
        _isolated = provider is not None and provider is not otel_trace.get_tracer_provider()
    except Exception:
        _isolated = False


def get_neatlogs_provider() -> Optional[TracerProvider]:
    """The provider neatlogs emits into, or None if init() hasn't set one."""
    client = _active_client.get()
    if client is not None:
        return client.tracer_provider
    return _neatlogs_provider


def get_active_client() -> Optional[Any]:
    """Return the secondary Client active in this context, if any."""
    return _active_client.get()


def activate_client(client: Any):
    """Select a secondary Client for this context and return its reset token."""
    return _active_client.set(client)


def reset_active_client(token: Any) -> None:
    _active_client.reset(token)


def _isolation_active() -> bool:
    """True when a private default or context-scoped Client pipeline is active."""
    return _active_client.get() is not None or _isolated


_wrapper_config: Dict[str, Any] = {}
_wrap_context: ContextVar[Optional[Dict[str, Any]]] = ContextVar(
    "neatlogs.wrap_context", default=None
)

_PROXY_PASSTHROUGH_TYPES = (
    str,
    bytes,
    int,
    float,
    bool,
    type(None),
    dict,
    list,
    tuple,
    set,
    frozenset,
)


def _normalize_traces_endpoint(endpoint: str) -> str:
    """Convert a Neatlogs base endpoint into the OTLP traces endpoint."""
    raw = (endpoint or "").strip().rstrip("/")
    if not raw:
        raw = DEFAULT_INGEST_ENDPOINT

    if raw.endswith("/v1/traces"):
        return raw

    parsed = urlparse(raw)
    if parsed.scheme and parsed.netloc:
        if parsed.path not in ("", "/"):
            raise ValueError(
                "NEATLOGS_ENDPOINT must be a base URL or an OTLP traces URL ending in /v1/traces."
            )
        return f"{parsed.scheme}://{parsed.netloc}/v1/traces"

    raise ValueError(
        "NEATLOGS_ENDPOINT must be a base URL or an OTLP traces URL ending in /v1/traces."
    )


def configure(**kwargs: Any) -> None:
    """
    Optional configuration for wrapper-only mode (no neatlogs.init() needed).

    Args:
        workflow_name: Logical grouping for traces
        session_id: Session identifier
        endpoint: Backend URL (default: https://ingest.neatlogs.com)
        api_key: Project write key (or set NEATLOGS_API_KEY env var)
    """
    _wrapper_config.update(kwargs)
    global _wrapper_tracer
    _wrapper_tracer = None


def _filtered_mapping(values: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if not values:
        return {}
    return {str(key): value for key, value in values.items() if value is not None}


def make_wrap_context(
    workflow_attributes: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Normalize metadata passed to ``neatlogs.wrap(...)``."""
    workflow_attrs = _filtered_mapping(workflow_attributes)
    context: Dict[str, Any] = {}
    if workflow_attrs:
        context["workflow"] = workflow_attrs
    return context


def _merged_wrap_context(context: Dict[str, Any]) -> Dict[str, Any]:
    current = _wrap_context.get() or {}
    if not current:
        merged: Dict[str, Any] = {}
        if context.get("workflow"):
            merged["workflow"] = dict(context["workflow"])
        return merged

    merged: Dict[str, Any] = {}
    workflow_attrs = dict(current.get("workflow") or {})
    workflow_attrs.update(context.get("workflow") or {})
    if workflow_attrs:
        merged["workflow"] = workflow_attrs
    return merged


def _call_with_wrap_context(
    fn: Callable[..., Any],
    context: Dict[str, Any],
    *args: Any,
    **kwargs: Any,
) -> Any:
    if inspect.iscoroutinefunction(fn):

        async def _async_call():
            token = _wrap_context.set(_merged_wrap_context(context))
            try:
                return await fn(*args, **kwargs)
            finally:
                _wrap_context.reset(token)

        return _async_call()

    token = _wrap_context.set(_merged_wrap_context(context))
    try:
        result = fn(*args, **kwargs)
        if hasattr(result, "__await__"):

            async def _await_result():
                await_token = _wrap_context.set(_merged_wrap_context(context))
                try:
                    return await result
                finally:
                    _wrap_context.reset(await_token)

            return _await_result()
        return result
    finally:
        _wrap_context.reset(token)


class _WrapContextProxy:
    """Forwarding proxy that makes wrap metadata active during method calls."""

    __slots__ = ("_neatlogs_target", "_neatlogs_context")

    def __init__(self, target: Any, context: Dict[str, Any]):
        object.__setattr__(self, "_neatlogs_target", target)
        object.__setattr__(self, "_neatlogs_context", context)

    @property
    def __class__(self):
        return object.__getattribute__(self, "_neatlogs_target").__class__

    def __getattr__(self, name: str) -> Any:
        target = object.__getattribute__(self, "_neatlogs_target")
        context = object.__getattribute__(self, "_neatlogs_context")
        value = getattr(target, name)
        if callable(value):

            def _wrapped_callable(*args: Any, **kwargs: Any) -> Any:
                return _call_with_wrap_context(value, context, *args, **kwargs)

            return _wrapped_callable
        if isinstance(value, _PROXY_PASSTHROUGH_TYPES):
            return value
        return _WrapContextProxy(value, context)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(object.__getattribute__(self, "_neatlogs_target"), name, value)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        target = object.__getattribute__(self, "_neatlogs_target")
        context = object.__getattribute__(self, "_neatlogs_context")
        return _call_with_wrap_context(target, context, *args, **kwargs)


def with_wrap_context(target: Any, context: Optional[Dict[str, Any]]) -> Any:
    if not context:
        return target
    return _WrapContextProxy(target, context)


def apply_wrap_context_attributes(span: Any, is_root: bool = True) -> None:
    if not is_root:
        return
    context = _wrap_context.get() or {}
    if not context:
        return

    for key, value in (context.get("workflow") or {}).items():
        try:
            span.set_attribute(f"neatlogs.workflow.{key}", str(value))
        except Exception:
            pass


def reset_tracer() -> None:
    """Drop the cached wrapper tracer so the next get_tracer() rebinds to the
    next private provider. Called by neatlogs.shutdown() — without it, a re-init
    (or a test that swaps the TracerProvider) keeps emitting wrapper spans through
    the previous, now-dead provider."""
    global _wrapper_tracer, _wrapper_bootstrapped
    _wrapper_tracer = None
    _wrapper_bootstrapped = False


class _ForeignParentGuardTracer:
    """Wraps a neatlogs Tracer so no span it starts ever inherits a FOREIGN
    (non-neatlogs) active parent.

    Invariant: only a neatlogs span may be the parent of a neatlogs span. When a
    span starts while a foreign span (a user's own OTel/Langfuse/openlit tracer,
    or any other instrumentation) is active in OTel context, nesting under it
    would give our span a parent the neatlogs backend never receives — a dangling
    parent, so the trace has no root, no completion marker fires, and it never
    finalizes. This guard detaches from such a parent (starts in an empty context)
    so the span becomes a true neatlogs root. Neatlogs parents, and explicit
    context=/callers, are passed through untouched. Applies uniformly to every
    integration that starts spans via get_tracer() (crewai, openai_agents,
    pydantic_ai, google_adk, agno, strands, hermes, claude_agent_sdk, ...)."""

    __slots__ = ("_tracer", "_context_transform")

    def __init__(
        self,
        tracer: otel_trace.Tracer,
        context_transform: Optional[
            Callable[[Optional[context_api.Context]], context_api.Context]
        ] = None,
    ):
        object.__setattr__(self, "_tracer", tracer)
        object.__setattr__(self, "_context_transform", context_transform)

    def _prepare_context(self, kwargs: Dict[str, Any]) -> None:
        if "context" not in kwargs:
            guard = _neatlogs_root_kwargs()  # {} or {"context": empty} if foreign parent
            if guard:
                kwargs.update(guard)
        transform = object.__getattribute__(self, "_context_transform")
        if transform is not None:
            kwargs["context"] = transform(kwargs.get("context"))

    def start_span(self, *args: Any, **kwargs: Any):
        tracer = object.__getattribute__(self, "_tracer")
        self._prepare_context(kwargs)
        return tracer.start_span(*args, **kwargs)

    def start_as_current_span(self, *args: Any, **kwargs: Any):
        tracer = object.__getattribute__(self, "_tracer")
        self._prepare_context(kwargs)
        return tracer.start_as_current_span(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_tracer"), name)


def get_tracer() -> otel_trace.Tracer:
    """
    Return a Tracer from init()'s provider, or auto-bootstrap from env.
    """
    global _wrapper_tracer, _wrapper_bootstrapped, _bootstrap_warned

    client = _active_client.get()
    if client is not None:
        return client.get_tracer("neatlogs.wrapper")

    if _wrapper_tracer is not None:
        return _wrapper_tracer

    # Neatlogs never adopts the process-global provider. Wrapper-only mode either
    # uses the private provider registered by init/configuration or bootstraps one.
    provider = _neatlogs_provider
    if isinstance(provider, TracerProvider):
        _wrapper_tracer = _ForeignParentGuardTracer(provider.get_tracer("neatlogs.wrapper"))
        return _wrapper_tracer

    api_key = _wrapper_config.get("api_key") or os.environ.get("NEATLOGS_API_KEY", "")
    if not api_key:
        if not _bootstrap_warned:
            _bootstrap_warned = True
            logger.warning(
                "neatlogs wrapper: no TracerProvider configured and NEATLOGS_API_KEY not set. "
                "Spans will not be exported. Call neatlogs.init() or set NEATLOGS_API_KEY."
            )
        _wrapper_tracer = _ForeignParentGuardTracer(otel_trace.get_tracer("neatlogs.wrapper.noop"))
        return _wrapper_tracer

    if not _wrapper_bootstrapped:
        _wrapper_bootstrapped = True
        _bootstrap_from_env(api_key)

    provider = get_neatlogs_provider()
    if provider is None:
        _wrapper_tracer = _ForeignParentGuardTracer(otel_trace.get_tracer("neatlogs.wrapper.noop"))
    else:
        _wrapper_tracer = _ForeignParentGuardTracer(provider.get_tracer("neatlogs.wrapper"))
    return _wrapper_tracer


def get_internal_tracer(scope: str) -> otel_trace.Tracer:
    """Guard-wrapped Tracer bound to the NEATLOGS provider for internal span
    sources (``@span`` decorators, ``trace()`` blocks, ...).

    These must resolve from the provider init() registered — NOT the OTel global
    ``get_tracer(__name__)`` — otherwise in isolated mode their spans export into a
    co-tenant's pipeline (a user's Langfuse provider). The guard wrapper also
    keeps them off a foreign active parent, identical to :func:`get_tracer`."""
    client = _active_client.get()
    if client is not None:
        return client.get_tracer(scope)
    provider = _neatlogs_provider
    if isinstance(provider, TracerProvider):
        return _ForeignParentGuardTracer(provider.get_tracer(scope))
    # Give wrapper-only configuration one chance to bootstrap a private provider.
    get_tracer()
    provider = _neatlogs_provider
    if isinstance(provider, TracerProvider):
        return _ForeignParentGuardTracer(provider.get_tracer(scope))
    return _ForeignParentGuardTracer(
        otel_trace.NoOpTracerProvider().get_tracer("neatlogs.unconfigured")
    )


class _NeatlogsSpanCM:
    """Context manager that creates a neatlogs span, makes it the neatlogs active
    parent, and ends it on exit — the isolation-safe replacement for
    ``tracer.start_as_current_span`` in internal span sources.

    In DEFAULT mode this is equivalent to ``start_as_current_span`` (the span
    becomes the OTel current-span). In ISOLATED mode :func:`attach_as_current`
    threads the parent on the private key WITHOUT touching the global current-span,
    so co-tenant instrumentation keeps nesting under the host while neatlogs' own
    children still nest under this span. Parent resolution and foreign-parent
    detachment come from the guard tracer, so the caller passes no ``context``."""

    __slots__ = ("_tracer", "_name", "_kwargs", "_span", "_token")

    def __init__(self, tracer: otel_trace.Tracer, name: str, **start_kwargs: Any):
        self._tracer = tracer
        self._name = name
        self._kwargs = start_kwargs
        self._span: Optional[otel_trace.Span] = None
        self._token: Any = None

    def __enter__(self) -> otel_trace.Span:
        self._span = self._tracer.start_span(self._name, **self._kwargs)
        self._token = attach_as_current(self._span)
        return self._span

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self._token is not None:
            detach(self._token)
        if self._span is not None:
            self._span.end()
        return False


def neatlogs_span(scope: str, name: str, **start_kwargs: Any) -> "_NeatlogsSpanCM":
    """Open an internal neatlogs span (decorator / trace() block) that is safe in
    both default and isolated modes. See :class:`_NeatlogsSpanCM`."""
    return _NeatlogsSpanCM(get_internal_tracer(scope), name, **start_kwargs)


def _bootstrap_from_env(api_key: str) -> None:
    from opentelemetry.exporter.otlp.proto.http import Compression
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import SERVICE_NAME, Resource
    from opentelemetry.sdk.trace import SpanLimits
    from .core.byte_limited_exporter import ByteLimitedSpanExporter
    from .core.delivery import DeliveryDiagnostics, ObservableBatchSpanProcessor
    from .core.transport import build_otlp_session

    endpoint = _wrapper_config.get("endpoint") or os.environ.get(
        "NEATLOGS_ENDPOINT", DEFAULT_INGEST_ENDPOINT
    )
    endpoint = _normalize_traces_endpoint(endpoint)

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
        compression=Compression.Gzip,
        session=build_otlp_session(),
    )
    diagnostics = DeliveryDiagnostics()
    provider.add_span_processor(
        ObservableBatchSpanProcessor(
            ByteLimitedSpanExporter(exporter, diagnostics=diagnostics),
            diagnostics=diagnostics,
        )
    )
    set_neatlogs_provider(provider)
    logger.debug(f"neatlogs wrapper: auto-bootstrapped private TracerProvider → {endpoint}")


def set_neatlogs_span_in_context(
    span: otel_trace.Span,
    context: Optional[context_api.Context] = None,
    *,
    force_owned: bool = False,
) -> context_api.Context:
    """Return a context with ``span`` installed as Neatlogs' private parent.

    ``force_owned`` is used by the OpenInference isolation adapter.  Those spans
    are exported by Neatlogs' private provider, but their instrumentation scope
    is ``openinference.*`` rather than ``neatlogs.*``.
    """
    ctx = context if context is not None else context_api.get_current()
    if force_owned or _is_neatlogs_span(span):
        ctx = context_api.set_value(_NEATLOGS_PARENT_KEY, span, ctx)
        ctx = context_api.set_value(_NEATLOGS_ACTIVE_KEY, True, ctx)
    return ctx


def attach_as_current(span: otel_trace.Span, *, force_owned: bool = False):
    """
    Make ``span`` the neatlogs *active parent* and return the context token.

    In the DEFAULT single-provider mode this makes ``span`` the OpenTelemetry
    active span so child operations nest correctly: provider auto-instrumentation
    spans, user ``@span`` decorators, ``trace()`` blocks, and ``log()`` LogRecords
    all resolve their parent via the standard OTel active-span context
    (``trace.get_current_span()`` / ``set_span_in_context``).

    In ISOLATED mode (init(tracer_provider=<private>)) it does NOT overwrite the
    global current-span. It instead records ``span`` under a PRIVATE context key
    that only neatlogs reads. This keeps a co-tenant instrumentor (openlit /
    langfuse) resolving its parent from the HOST's current-span — so foreign
    spans never nest under a neatlogs span, and neatlogs spans never borrow a
    foreign parent. neatlogs' own children still nest correctly because
    ``get_tracer()`` resolves their parent from this same private key.

    Detach the returned token (in a ``finally``) when the span completes.

        token = attach_as_current(span)
        try:
            ...
        finally:
            context_api.detach(token)
    """
    is_nl = force_owned or _is_neatlogs_span(span)
    if _isolation_active():
        # ISOLATED: thread the neatlogs parent privately; leave the global
        # current-span untouched so foreign instrumentation nests under the host.
        ctx = set_neatlogs_span_in_context(span, force_owned=force_owned)
        return context_api.attach(ctx)

    ctx = otel_trace.set_span_in_context(span)
    # Mark that a neatlogs span is active in this context subtree. A foreign
    # instrumentation (e.g. openlit) may push its OWN spans as the immediate
    # current span BETWEEN two neatlogs spans (crew kickoff → openlit agent span →
    # our llm.call). Inspecting only the immediate current span would then mistake
    # the openlit span for "no neatlogs parent" and detach our child into its own
    # trace, fragmenting the crew. This flag records that a neatlogs ancestor
    # exists somewhere up the active chain, so the root-guard nests instead of
    # detaching. Context is immutable+nested, so the flag scopes to the subtree.
    if is_nl:
        ctx = context_api.set_value(_NEATLOGS_ACTIVE_KEY, True, ctx)
    return context_api.attach(ctx)


def _current_neatlogs_parent(
    context: Optional[context_api.Context] = None,
) -> Optional[otel_trace.Span]:
    """The neatlogs span threaded via the private parent key (isolated mode), or
    None. Returns None for stale/ended local spans so a dead parent never dangles
    a child — but ACCEPTS a valid REMOTE parent (a non-recording span installed by
    ``extract_trace_context`` from an inbound W3C ``traceparent``), so a callee
    nests under the caller's cross-process trace."""
    try:
        parent = context_api.get_value(_NEATLOGS_PARENT_KEY, context)
    except Exception:
        parent = None
    if parent is None:
        return None
    try:
        if parent.is_recording():
            return parent
        # Remote parents are non-recording by construction; accept only when the
        # span context is both remote and valid. A stale ENDED LOCAL span is
        # non-recording AND is_remote=False, so it stays rejected.
        sc = parent.get_span_context()
        if getattr(sc, "is_remote", False) and sc.is_valid:
            return parent
    except Exception:
        pass
    return None


def active_neatlogs_context(
    context: Optional[context_api.Context] = None,
) -> Optional[context_api.Context]:
    """Return a context whose current-span is the active NEATLOGS span, or None.

    This is the isolation-aware analog of the TS SDK's ``getActiveNeatlogsSpan``:
    in ISOLATED mode the active neatlogs span lives on the private parent key
    (the global current-span is the host's), so we resolve it from there and
    rebuild a context with it as the current span — exactly what W3C
    ``inject(context=...)`` needs to write our ``traceparent``. In DEFAULT mode
    the neatlogs span already IS the global current-span, so the ambient context
    works and we return it unchanged.

    Returns None when no recording neatlogs span is active, so callers can no-op
    (the inject helper leaves the carrier untouched)."""
    if _isolation_active():
        parent = _current_neatlogs_parent(context)
        if parent is None:
            return None
        base = context if context is not None else context_api.Context()
        return otel_trace.set_span_in_context(parent, base)
    # DEFAULT mode: only report a context when a recording neatlogs span is the
    # active ancestor — never let a purely foreign current-span be injected as ours.
    current = otel_trace.get_current_span()
    if current and current.is_recording() and _has_neatlogs_ancestor():
        return context if context is not None else context_api.get_current()
    return None


def detach(token: Any) -> None:
    """Detach a context token returned by :func:`attach_as_current`."""
    try:
        context_api.detach(token)
    except Exception:
        pass


def is_suppressed() -> bool:
    """Check if a framework instrumentor already covers this call."""
    try:
        return bool(context_api.get_value("suppress_instrumentation"))
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Auto-root
#
# The backend only renders a trace once it contains a *parentless* span of a
# root-eligible kind (WORKFLOW / CHAIN / AGENT / MCP_TOOL). Direct-provider
# wrappers (openai, anthropic, bedrock, ...) only ever emit non-root spans
# (llm / embedding / reranker / tool). So a bare ``client = neatlogs.wrap(...)``
# call with no surrounding ``@span`` / ``trace()`` produces an orphan span and
# the trace never renders.
#
# ``get_provider_tracer()`` returns a tracer facade used *only* by those
# direct-provider wrappers: when a span would otherwise be parentless and is a
# non-root kind, it transparently opens a WORKFLOW root (named after the
# configured ``workflow_name``) and closes it when the provider span ends.
# Framework wrappers (langchain, crewai, agno, ...) keep using ``get_tracer()``
# unchanged — they already emit their own root and thread context explicitly,
# so auto-root must never fire for them.
# ---------------------------------------------------------------------------

# A parentless span of one of these kinds already satisfies the backend's
# root requirement, so it must NOT be wrapped in another root.
_ROOT_KINDS = frozenset({"workflow", "chain", "agent", "mcp_tool"})


def _auto_root_enabled() -> bool:
    """Auto-root is on unless explicitly disabled via NEATLOGS_AUTO_ROOT."""
    val = os.environ.get("NEATLOGS_AUTO_ROOT", "").strip().lower()
    return val not in ("false", "0", "no", "off")


def _resolve_root_workflow_name() -> str:
    """The name for an auto-created root: init()'s workflow_name, else the
    wrapper-mode workflow_name, else a neutral default."""
    client = _active_client.get()
    if client is not None:
        return client.workflow_name
    try:
        from .init import get_session_config

        name = (get_session_config() or {}).get("workflow_name")
        if name:
            return name
    except Exception:
        pass
    return _wrapper_config.get("workflow_name") or "workflow"


# Context flag: set by attach_as_current() whenever a neatlogs span is made
# active, so descendants know a neatlogs ancestor exists even if a foreign span
# (openlit, etc.) is the IMMEDIATE current span between them.
_NEATLOGS_ACTIVE_KEY = "neatlogs.active_span_present"

# Isolated-mode parent pointer. In isolated mode attach_as_current() stores the
# active neatlogs span HERE instead of on the global current-span, so neatlogs
# children resolve their parent from this key while foreign instrumentation
# (openlit / langfuse) keeps nesting under the untouched host current-span.
# Propagates across asyncio await AND threads (via ThreadingInstrumentor) — the
# same mechanism as any other OTel context value — so CrewAI's threaded task/tool
# execution keeps the neatlogs hierarchy intact.
_NEATLOGS_PARENT_KEY = "neatlogs.parent_span"


def _has_neatlogs_ancestor() -> bool:
    """True if a neatlogs span is active anywhere up the current context chain.

    Prefers the context flag (set by attach_as_current) so a foreign span sitting
    BETWEEN two neatlogs spans doesn't hide the neatlogs ancestor. Falls back to
    inspecting the immediate current span for spans neatlogs created without
    attach_as_current."""
    try:
        if context_api.get_value(_NEATLOGS_ACTIVE_KEY):
            return True
    except Exception:
        pass
    current = otel_trace.get_current_span()
    return bool(current and current.is_recording() and _is_neatlogs_span(current))


def _has_active_recording_parent() -> bool:
    """True when there is an active, recording NEATLOGS span (anywhere up the
    chain) to nest under.

    A purely foreign context (another OTel instrumentation — a user's Langfuse
    tracer, or openlit — with NO neatlogs ancestor) must NOT be treated as a
    parent: nesting under it produces a neatlogs span whose parent never reaches
    the neatlogs backend (dangling → no root → no completion marker → never
    finalizes). But a foreign span nested INSIDE a neatlogs span (e.g. openlit's
    crew span between our kickoff and our llm.call) still has a neatlogs ancestor,
    so we DO nest — otherwise the crew fragments into many single-span traces."""
    current = otel_trace.get_current_span()
    if not (current and current.is_recording()):
        # No active span at all — but a neatlogs ancestor flag could still be set
        # in a detached/propagated context; be conservative and treat as parent.
        return _has_neatlogs_ancestor()
    return _has_neatlogs_ancestor()


def _is_neatlogs_span(span: Any) -> bool:
    """True if the span was created by a neatlogs tracer (scope name 'neatlogs.*').
    Foreign spans (langfuse, openlit, user tracers) return False."""
    try:
        scope = getattr(span, "instrumentation_scope", None)
        name = getattr(scope, "name", "") or ""
        return name.startswith("neatlogs")
    except Exception:
        return False


def _neatlogs_root_kwargs() -> dict:
    """kwargs for start_span so a neatlogs span nests under its neatlogs parent and
    never inherits a purely FOREIGN context.

    ISOLATED mode (private provider): the global current-span is the HOST's, never
    ours, so we NEVER let a neatlogs span inherit it implicitly. If a neatlogs
    parent is threaded on the private key, return a context explicitly rooted at
    it; otherwise return an empty context so the span is a true root. This is what
    keeps neatlogs' own hierarchy intact while foreign instrumentation nests under
    the untouched host current-span.

    DEFAULT mode (shared/owned provider): if a neatlogs span is active anywhere up
    the chain, nest normally (return {}) — even when a foreign span (openlit) is
    the immediate current span. Only when there is NO neatlogs ancestor do we start
    in an empty context so the new span is a true root. A caller-supplied shared
    trace_id still wins via its own context kwarg."""
    if _isolation_active():
        parent = _current_neatlogs_parent()
        if parent is not None:
            return {"context": otel_trace.set_span_in_context(parent)}
        return {"context": context_api.Context()}
    if _has_neatlogs_ancestor():
        return {}
    current = otel_trace.get_current_span()
    if current and current.is_recording():
        # purely foreign parent active (no neatlogs ancestor) → detach from it
        return {"context": context_api.Context()}
    return {}


class _RootEndingSpan:
    """Transparent proxy around a provider span that also ends an auto-created
    WORKFLOW root when the provider span ends.

    Wrappers and stream finalizers only touch the span through duck-typed
    methods (set_attribute / set_status / record_exception / end / ...), so a
    delegating proxy is sufficient and avoids mutating OTel's Span instance.
    """

    __slots__ = ("_child", "_root", "_ended")

    def __init__(self, child: otel_trace.Span, root: otel_trace.Span):
        object.__setattr__(self, "_child", child)
        object.__setattr__(self, "_root", root)
        object.__setattr__(self, "_ended", False)

    def end(self, *args: Any, **kwargs: Any) -> None:
        if object.__getattribute__(self, "_ended"):
            return
        object.__setattr__(self, "_ended", True)
        child = object.__getattribute__(self, "_child")
        root = object.__getattribute__(self, "_root")
        try:
            child.end(*args, **kwargs)
        finally:
            try:
                root.end()
            except Exception:
                pass

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_child"), name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(object.__getattribute__(self, "_child"), name, value)


class _AutoRootTracer:
    """Tracer facade for direct-provider wrappers. ``start_span`` behaves like
    the underlying tracer, except it transparently opens a WORKFLOW root when
    the span would otherwise be parentless and is a non-root kind."""

    __slots__ = ("_tracer",)

    def __init__(self, tracer: otel_trace.Tracer):
        self._tracer = tracer

    def start_span(self, name: str, attributes: Optional[Dict[str, Any]] = None, **kwargs: Any):
        tracer = object.__getattribute__(self, "_tracer")
        attributes = attributes or {}
        kind = str(attributes.get("neatlogs.span.kind", "")).lower()

        needs_root = (
            _auto_root_enabled()
            and kind not in _ROOT_KINDS
            and "context" not in kwargs  # explicit-context callers opt out
            and not _has_active_recording_parent()
        )
        if not needs_root:
            # A root-kind span (or auto-root disabled): still must not inherit a
            # FOREIGN active parent. If only a foreign span is active, detach so
            # this span becomes a true neatlogs root. Explicit-context callers and
            # spans already under a neatlogs parent are untouched.
            if "context" not in kwargs:
                foreign_kwargs = _neatlogs_root_kwargs()
                if foreign_kwargs:
                    kwargs.update(foreign_kwargs)
            return tracer.start_span(name=name, attributes=attributes, **kwargs)

        # If only a foreign span is active, start the auto-root in an empty context
        # so it doesn't inherit the foreign (non-neatlogs) parent.
        root_ctx = _neatlogs_root_kwargs().get("context")
        root = tracer.start_span(
            name=_resolve_root_workflow_name(),
            attributes={"neatlogs.span.kind": "workflow", "neatlogs.auto_root": True},
            **({"context": root_ctx} if root_ctx is not None else {}),
        )
        # Stamp request-scoped identity (set via neatlogs.identify()) onto the
        # auto-root. This is the only path wrapper-only code has to a root span,
        # so it's where session/end-user attach for bare wrap() usage.
        try:
            from .core.end_user import apply_end_user_attributes
            from .core.session import apply_session_attributes

            apply_wrap_context_attributes(root, is_root=True)
            apply_session_attributes(root, None, is_root=True)
            apply_end_user_attributes(root, None, None, is_root=True)
        except Exception:
            pass
        token = attach_as_current(root)
        try:
            child = tracer.start_span(name=name, attributes=attributes, **kwargs)
        except Exception:
            detach(token)
            try:
                root.end()
            except Exception:
                pass
            raise
        # Restore context immediately — the child already captured root as its
        # parent, and provider spans never nest user code under themselves.
        detach(token)
        return _RootEndingSpan(child, root)

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_tracer"), name)


def get_provider_tracer() -> "_AutoRootTracer":
    """Tracer for direct-provider wrappers (openai, anthropic, bedrock, ...).

    Identical to :func:`get_tracer` but adds transparent auto-root so a bare
    ``neatlogs.wrap(client)`` renders a trace without a manual ``@span`` /
    ``trace()`` wrapper. Do NOT use for framework wrappers."""
    return _AutoRootTracer(get_tracer())


def serialize(obj: Any, max_length: Optional[int] = None) -> str:
    """Safe JSON serialization; truncation is only applied when explicitly requested."""
    try:
        s = json.dumps(obj, default=str, ensure_ascii=False)
    except (TypeError, ValueError):
        s = str(obj)
    if max_length is not None and len(s) > max_length:
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
        self._incremental = hasattr(finalizer, "on_chunk") and hasattr(finalizer, "finish")
        self._chunks: List[Any] = []
        self._finalized = False
        self._exhausted = False

    def __iter__(self):
        def consume():
            try:
                while True:
                    try:
                        yield self.__next__()
                    except StopIteration:
                        return
            finally:
                if not self._finalized:
                    self.close()

        return consume()

    def __next__(self):
        try:
            chunk = next(self._stream)
        except StopIteration:
            self._exhausted = True
            self._finalize(interrupted=False)
            raise
        except BaseException as e:
            self._finalize_error(e)
            raise

        if self._first_chunk_time is None:
            self._first_chunk_time = time.perf_counter()
        if self._incremental:
            self._finalizer.on_chunk(self._span, chunk)
        else:
            self._chunks.append(chunk)
        return chunk

    def __enter__(self):
        if hasattr(self._stream, "__enter__"):
            self._stream.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            if hasattr(self._stream, "__exit__"):
                self._stream.__exit__(exc_type, exc, tb)
        except BaseException as close_error:
            self._finalize_error(close_error)
            raise
        if exc is not None:
            self._finalize_error(exc)
        else:
            self._finalize(interrupted=not self._exhausted)

    def close(self):
        try:
            if hasattr(self._stream, "close"):
                self._stream.close()
        except BaseException as close_error:
            self._finalize_error(close_error)
            raise
        self._finalize(interrupted=not self._exhausted)

    def _finalize(self, *, interrupted: bool):
        if self._finalized:
            return
        self._finalized = True
        elapsed_ms = (time.perf_counter() - self._start_time) * 1000
        ttft_ms = None
        if self._first_chunk_time is not None:
            ttft_ms = (self._first_chunk_time - self._start_time) * 1000
        if self._incremental:
            self._finalizer.finish(
                self._span,
                elapsed_ms,
                ttft_ms,
                interrupted=interrupted,
            )
        else:
            if interrupted:
                self._span.set_attribute("neatlogs.stream.cancelled", True)
            self._finalizer(self._span, self._chunks, elapsed_ms, ttft_ms)

    def _finalize_error(self, error: BaseException):
        if self._finalized:
            return
        self._finalized = True
        if self._incremental and hasattr(self._finalizer, "fail"):
            self._finalizer.fail(self._span, error)
            return
        from opentelemetry.trace import StatusCode

        if error.__class__.__name__ in {"CancelledError", "GeneratorExit"}:
            self._span.set_attribute("neatlogs.stream.cancelled", True)
            self._span.set_status(StatusCode.UNSET)
        else:
            self._span.set_status(StatusCode.ERROR, str(error))
            if isinstance(error, Exception):
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
        self._incremental = hasattr(finalizer, "on_chunk") and hasattr(finalizer, "finish")
        self._chunks: List[Any] = []
        self._finalized = False
        self._exhausted = False

    def __aiter__(self):
        async def consume():
            try:
                while True:
                    try:
                        yield await self.__anext__()
                    except StopAsyncIteration:
                        return
            finally:
                if not self._finalized:
                    await self.aclose()

        return consume()

    async def __anext__(self):
        try:
            chunk = await self._stream.__anext__()
        except StopAsyncIteration:
            self._exhausted = True
            self._finalize(interrupted=False)
            raise
        except BaseException as e:
            self._finalize_error(e)
            raise

        if self._first_chunk_time is None:
            self._first_chunk_time = time.perf_counter()
        if self._incremental:
            self._finalizer.on_chunk(self._span, chunk)
        else:
            self._chunks.append(chunk)
        return chunk

    async def __aenter__(self):
        if hasattr(self._stream, "__aenter__"):
            await self._stream.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        try:
            if hasattr(self._stream, "__aexit__"):
                await self._stream.__aexit__(exc_type, exc, tb)
        except BaseException as close_error:
            self._finalize_error(close_error)
            raise
        if exc is not None:
            self._finalize_error(exc)
        else:
            self._finalize(interrupted=not self._exhausted)

    async def aclose(self):
        try:
            if hasattr(self._stream, "aclose"):
                await self._stream.aclose()
        except BaseException as close_error:
            self._finalize_error(close_error)
            raise
        self._finalize(interrupted=not self._exhausted)

    def _finalize(self, *, interrupted: bool):
        if self._finalized:
            return
        self._finalized = True
        elapsed_ms = (time.perf_counter() - self._start_time) * 1000
        ttft_ms = None
        if self._first_chunk_time is not None:
            ttft_ms = (self._first_chunk_time - self._start_time) * 1000
        if self._incremental:
            self._finalizer.finish(
                self._span,
                elapsed_ms,
                ttft_ms,
                interrupted=interrupted,
            )
        else:
            if interrupted:
                self._span.set_attribute("neatlogs.stream.cancelled", True)
            self._finalizer(self._span, self._chunks, elapsed_ms, ttft_ms)

    def _finalize_error(self, error: BaseException):
        if self._finalized:
            return
        self._finalized = True
        if self._incremental and hasattr(self._finalizer, "fail"):
            self._finalizer.fail(self._span, error)
            return
        from opentelemetry.trace import StatusCode

        if error.__class__.__name__ in {"CancelledError", "GeneratorExit"}:
            self._span.set_attribute("neatlogs.stream.cancelled", True)
            self._span.set_status(StatusCode.UNSET)
        else:
            self._span.set_status(StatusCode.ERROR, str(error))
            if isinstance(error, Exception):
                self._span.record_exception(error)
        self._span.end()

    def __getattr__(self, name):
        return getattr(self._stream, name)
