"""
Shared infrastructure for Neatlogs provider wrappers.

Only contains truly shared concerns:
  - TracerProvider bootstrap (auto from env or reuse from init())
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

from opentelemetry import context as context_api
from opentelemetry import trace as otel_trace
from opentelemetry.sdk.trace import TracerProvider

from .core.logger import get_logger

logger = get_logger()

_wrapper_tracer: Optional[otel_trace.Tracer] = None
_wrapper_bootstrapped = False
_bootstrap_warned = False

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
        raw = "https://ingest.neatlogs.com"

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
    current global provider. Called by neatlogs.shutdown() — without it, a re-init
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

    __slots__ = ("_tracer",)

    def __init__(self, tracer: otel_trace.Tracer):
        object.__setattr__(self, "_tracer", tracer)

    def start_span(self, *args: Any, **kwargs: Any):
        tracer = object.__getattribute__(self, "_tracer")
        if "context" not in kwargs:
            guard = _neatlogs_root_kwargs()  # {} or {"context": empty} if foreign parent
            if guard:
                kwargs.update(guard)
        return tracer.start_span(*args, **kwargs)

    def start_as_current_span(self, *args: Any, **kwargs: Any):
        tracer = object.__getattribute__(self, "_tracer")
        if "context" not in kwargs:
            guard = _neatlogs_root_kwargs()
            if guard:
                kwargs.update(guard)
        return tracer.start_as_current_span(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_tracer"), name)


def get_tracer() -> otel_trace.Tracer:
    """
    Return a Tracer from init()'s provider, or auto-bootstrap from env.
    """
    global _wrapper_tracer, _wrapper_bootstrapped, _bootstrap_warned

    if _wrapper_tracer is not None:
        return _wrapper_tracer

    provider = otel_trace.get_tracer_provider()
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

    _wrapper_tracer = _ForeignParentGuardTracer(otel_trace.get_tracer("neatlogs.wrapper"))
    return _wrapper_tracer


def _bootstrap_from_env(api_key: str) -> None:
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import SERVICE_NAME, Resource
    from opentelemetry.sdk.trace import SpanLimits
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    endpoint = (
        _wrapper_config.get("endpoint")
        or os.environ.get("NEATLOGS_ENDPOINT", "https://ingest.neatlogs.com")
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
    )
    provider.add_span_processor(BatchSpanProcessor(exporter))
    otel_trace.set_tracer_provider(provider)
    logger.debug(f"neatlogs wrapper: auto-bootstrapped TracerProvider → {endpoint}")


def attach_as_current(span: otel_trace.Span):
    """
    Make ``span`` the OpenTelemetry *active* span and return the context token.

    This is what makes child operations nest correctly: provider
    auto-instrumentation spans, user ``@span`` decorators, ``trace()`` blocks,
    and ``log()`` LogRecords all resolve their parent via the standard OTel
    active-span context (``trace.get_current_span()`` / ``set_span_in_context``).

    Detach the returned token (in a ``finally``) when the span completes.

        token = attach_as_current(span)
        try:
            ...
        finally:
            context_api.detach(token)
    """
    ctx = otel_trace.set_span_in_context(span)
    # Mark that a neatlogs span is active in this context subtree. A foreign
    # instrumentation (e.g. openlit) may push its OWN spans as the immediate
    # current span BETWEEN two neatlogs spans (crew kickoff → openlit agent span →
    # our llm.call). Inspecting only the immediate current span would then mistake
    # the openlit span for "no neatlogs parent" and detach our child into its own
    # trace, fragmenting the crew. This flag records that a neatlogs ancestor
    # exists somewhere up the active chain, so the root-guard nests instead of
    # detaching. Context is immutable+nested, so the flag scopes to the subtree.
    if _is_neatlogs_span(span):
        ctx = context_api.set_value(_NEATLOGS_ACTIVE_KEY, True, ctx)
    return context_api.attach(ctx)


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
    """kwargs for start_span so a neatlogs root ignores a purely FOREIGN context.

    If a neatlogs span is active anywhere up the chain, nest normally (return {}) —
    even when a foreign span (openlit) is the immediate current span. Only when
    there is NO neatlogs ancestor (no active span, or only foreign spans) do we
    start in an empty context so the new span is a true root (parent_span_id='')
    that finalizes. A caller-supplied shared trace_id still wins via its own
    context kwarg."""
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
            and "context" not in kwargs          # explicit-context callers opt out
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
