"""Provider-local context isolation for OpenInference instrumentors.

OpenTelemetry has one process-wide current-span slot.  Supplying a private
TracerProvider controls export, but OpenInference adapters still activate their
spans through that global slot.  This module marks only tracers obtained through
Neatlogs' private OpenInference provider and redirects activation of their spans
to Neatlogs' private parent key.
"""

import sys
from contextlib import contextmanager
from typing import Any, Iterator, Optional

from opentelemetry import context as context_api
from opentelemetry import trace as trace_api
from opentelemetry.trace import Span, Status, StatusCode
from opentelemetry.trace.propagation import _SPAN_KEY

from .._wrap_utils import (
    _current_neatlogs_parent,
    _isolation_active,
    set_neatlogs_span_in_context,
)

_ISOLATED_MARKER = "_neatlogs_openinference_isolated"
_ORIGINAL_USE_SPAN = trace_api.use_span
_ORIGINAL_SET_SPAN_IN_CONTEXT = trace_api.set_span_in_context
_ORIGINAL_GET_CURRENT_SPAN = trace_api.get_current_span
_ORIGINAL_SET_VALUE = context_api.set_value
_PATCHED = False


def _is_isolated_span(span: Any) -> bool:
    if bool(getattr(span, _ISOLATED_MARKER, False)):
        return True
    return bool(getattr(getattr(span, "__wrapped__", None), _ISOLATED_MARKER, False))


def _mark_isolated(target: Any) -> Any:
    try:
        setattr(target, _ISOLATED_MARKER, True)
    except Exception:
        pass
    wrapped = getattr(target, "__wrapped__", None)
    if wrapped is not None:
        try:
            setattr(wrapped, _ISOLATED_MARKER, True)
        except Exception:
            pass
    return target


def _sdk_parent_context(source: Optional[context_api.Context]) -> context_api.Context:
    """Translate the private parent pointer into an SDK-readable OTel context."""
    parent = _current_neatlogs_parent(source)
    if parent is not None:
        return _ORIGINAL_SET_SPAN_IN_CONTEXT(parent, context_api.Context())

    # Some callback-based OI integrations retain an explicit standard context.
    # Preserve it only when it points to another marked private OI span.
    if source is not None:
        explicit_parent = trace_api.get_current_span(source)
        if _is_isolated_span(explicit_parent) and explicit_parent.is_recording():
            return _ORIGINAL_SET_SPAN_IN_CONTEXT(explicit_parent, context_api.Context())
    return context_api.Context()


def _safe_set_span_in_context(
    span: Span, context: Optional[context_api.Context] = None
) -> context_api.Context:
    if not _is_isolated_span(span):
        return _ORIGINAL_SET_SPAN_IN_CONTEXT(span, context)
    return set_neatlogs_span_in_context(span, context, force_owned=True)


def _safe_get_current_span(context: Optional[context_api.Context] = None) -> Span:
    private_span = _current_neatlogs_parent(context)
    if private_span is not None and _is_isolated_span(private_span):
        return private_span
    return _ORIGINAL_GET_CURRENT_SPAN(context)


def _safe_set_value(
    key: str, value: Any, context: Optional[context_api.Context] = None
) -> context_api.Context:
    # AutoGen AgentChat writes OTel's private span key directly instead of
    # calling use_span()/set_span_in_context(). Redirect that one operation.
    if key == _SPAN_KEY and _is_isolated_span(value):
        return set_neatlogs_span_in_context(value, context, force_owned=True)
    return _ORIGINAL_SET_VALUE(key, value, context)


@contextmanager
def _safe_use_span(
    span: Span,
    end_on_exit: bool = False,
    record_exception: bool = True,
    set_status_on_exception: bool = True,
) -> Iterator[Span]:
    if not _is_isolated_span(span):
        with _ORIGINAL_USE_SPAN(
            span,
            end_on_exit=end_on_exit,
            record_exception=record_exception,
            set_status_on_exception=set_status_on_exception,
        ) as current_span:
            yield current_span
        return

    token = context_api.attach(set_neatlogs_span_in_context(span, force_owned=True))
    try:
        try:
            yield span
        finally:
            context_api.detach(token)
    except Exception as exc:
        if isinstance(span, Span) and span.is_recording():
            if record_exception:
                span.record_exception(exc)
            if set_status_on_exception:
                span.set_status(
                    Status(
                        status_code=StatusCode.ERROR,
                        description=f"{type(exc).__name__}: {exc}",
                    )
                )
        raise
    finally:
        if end_on_exit:
            span.end()


class _IsolatedOpenInferenceTracerProvider:
    """Delegating provider that marks tracers handed to OpenInference only."""

    __slots__ = ("_provider",)

    def __init__(self, provider: Any) -> None:
        self._provider = provider

    def get_tracer(self, *args: Any, **kwargs: Any) -> Any:
        return _mark_isolated(self._provider.get_tracer(*args, **kwargs))

    def __getattr__(self, name: str) -> Any:
        return getattr(self._provider, name)


def _patch_loaded_openinference_references() -> None:
    """Replace direct imports held by already-imported OI adapter modules."""
    for name, module in tuple(sys.modules.items()):
        if not name.startswith("openinference.instrumentation") or module is None:
            continue
        namespace = getattr(module, "__dict__", {})
        for attr_name, value in tuple(namespace.items()):
            if value is _ORIGINAL_USE_SPAN:
                namespace[attr_name] = _safe_use_span
            elif value is _ORIGINAL_SET_SPAN_IN_CONTEXT:
                namespace[attr_name] = _safe_set_span_in_context
            elif value is _ORIGINAL_GET_CURRENT_SPAN:
                namespace[attr_name] = _safe_get_current_span
            elif value is _ORIGINAL_SET_VALUE:
                namespace[attr_name] = _safe_set_value


def _install_patch() -> None:
    global _PATCHED
    if _PATCHED:
        _patch_loaded_openinference_references()
        return

    from openinference.instrumentation import OITracer

    original_start_span = OITracer.start_span

    def _isolated_start_span(self: Any, *args: Any, **kwargs: Any) -> Any:
        wrapped = getattr(self, "__wrapped__", None)
        if not bool(getattr(wrapped, _ISOLATED_MARKER, False)):
            return original_start_span(self, *args, **kwargs)

        positional = list(args)
        if len(positional) >= 2:
            positional[1] = _sdk_parent_context(positional[1])
        else:
            kwargs["context"] = _sdk_parent_context(kwargs.get("context"))
        span = original_start_span(self, *positional, **kwargs)
        return _mark_isolated(span)

    OITracer.start_span = _isolated_start_span
    trace_api.use_span = _safe_use_span
    trace_api.set_span_in_context = _safe_set_span_in_context
    context_api.set_value = _safe_set_value
    _PATCHED = True
    _patch_loaded_openinference_references()


def provider_for_openinference(provider: Any) -> Any:
    """Return ``provider`` adapted for OI when Neatlogs uses private isolation."""
    if not _isolation_active():
        return provider
    _install_patch()
    return _IsolatedOpenInferenceTracerProvider(provider)
