"""Neatlogs integration for Strands Agents."""

import copy
import threading
import weakref
from typing import Any, Optional

from openinference.instrumentation.strands_agents import (
    StrandsAgentsToOpenInferenceProcessor,
)

from ._wrap_utils import get_active_client, get_neatlogs_provider
from .instrumentation.openinference_isolation import provider_for_openinference
from .instrumentation.preprocessing import (
    add_provider_preprocessor,
    remove_provider_preprocessor,
)

_LOCK = threading.RLock()
_DEFAULT_OWNER = object()
_CONTEXTUAL_TRACER: Optional[Any] = None
_PREVIOUS_TRACER: Optional[Any] = None
_WRAPPED_AGENTS: "weakref.WeakKeyDictionary[Any, tuple[Any, Any]]" = weakref.WeakKeyDictionary()
_PROVIDER_TRACERS: "weakref.WeakKeyDictionary[Any, weakref.ReferenceType[Any]]" = (
    weakref.WeakKeyDictionary()
)
_PROVIDER_PROCESSORS: "weakref.WeakKeyDictionary[Any, Any]" = weakref.WeakKeyDictionary()
_PROVIDER_OWNERS: "weakref.WeakKeyDictionary[Any, set[Any]]" = weakref.WeakKeyDictionary()


class _NeatlogsStrandsProcessor(StrandsAgentsToOpenInferenceProcessor):
    def on_end(self, span: Any) -> None:
        status = span.status
        interrupted = bool(
            (getattr(span, "attributes", None) or {}).get("neatlogs.trace.interrupted")
        )
        super().on_end(span)
        if interrupted:
            span._status = status


class _ContextualStrandsTracer:
    def __init__(self, fallback: Any) -> None:
        self._fallback = fallback

    def _current(self) -> Any:
        provider = get_neatlogs_provider()
        if provider is None:
            if self._fallback is not None:
                return self._fallback
            from strands.telemetry.tracer import Tracer

            return Tracer()
        return _tracer_for_provider(provider, self._fallback)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._current(), name)


def _current_owner() -> Any:
    return get_active_client() or _DEFAULT_OWNER


def prepare_strands(provider: Any, owner: Any = None) -> bool:
    """Install Strands conversion before Neatlogs normalization and export."""
    with _LOCK:
        owner = _current_owner() if owner is None else owner
        owners = _PROVIDER_OWNERS.setdefault(provider, set())
        owners.add(owner)
        if provider not in _PROVIDER_PROCESSORS:
            processor = _NeatlogsStrandsProcessor()
            installed = add_provider_preprocessor(provider, "strands", processor)
            _PROVIDER_PROCESSORS[provider] = installed
    return True


def _tracer_for_provider(provider: Any, fallback: Any = None) -> Any:
    with _LOCK:
        existing_ref = _PROVIDER_TRACERS.get(provider)
        existing = existing_ref() if existing_ref is not None else None
        if existing is not None:
            prepare_strands(provider)
            return existing

        from strands.telemetry.tracer import Tracer

        prepare_strands(provider)
        base = fallback if fallback is not None else _PREVIOUS_TRACER
        tracer = copy.copy(base) if base is not None else Tracer()
        oi_provider = provider_for_openinference(provider)
        tracer.tracer_provider = oi_provider
        tracer.tracer = oi_provider.get_tracer(tracer.service_name)
        _PROVIDER_TRACERS[provider] = weakref.ref(tracer)
        return tracer


def _ensure_contextual_tracer_locked() -> Any:
    from strands.telemetry import tracer as tracer_module

    global _CONTEXTUAL_TRACER, _PREVIOUS_TRACER
    if _CONTEXTUAL_TRACER is None:
        _PREVIOUS_TRACER = tracer_module._tracer_instance
        _CONTEXTUAL_TRACER = _ContextualStrandsTracer(_PREVIOUS_TRACER)
    if tracer_module._tracer_instance is not _CONTEXTUAL_TRACER:
        tracer_module._tracer_instance = _CONTEXTUAL_TRACER
    return _CONTEXTUAL_TRACER


def instrument_strands(provider: Any, owner: Any = _DEFAULT_OWNER) -> bool:
    """Route Strands telemetry through the provider active for each invocation."""
    with _LOCK:
        prepare_strands(provider, owner)
        _ensure_contextual_tracer_locked()
        for agent, (previous, current_owner) in list(_WRAPPED_AGENTS.items()):
            if current_owner is None:
                _WRAPPED_AGENTS[agent] = (previous, owner)
    return True


def has_wrapped_agents() -> bool:
    with _LOCK:
        return bool(_WRAPPED_AGENTS)


def strands_hooks(agent: Any) -> Any:
    """Route an existing Strands agent through Neatlogs and return it unchanged."""
    with _LOCK:
        tracer = _ensure_contextual_tracer_locked()
        provider = get_neatlogs_provider()
        owner = _current_owner() if provider is not None else None
        if provider is not None:
            prepare_strands(provider, owner)
        try:
            if agent not in _WRAPPED_AGENTS:
                previous = getattr(agent, "tracer", None)
                if previous is tracer:
                    previous = _PREVIOUS_TRACER
                _WRAPPED_AGENTS[agent] = (previous, owner)
        except TypeError:
            pass
        if getattr(agent, "tracer", None) is not tracer:
            agent.tracer = tracer
        try:
            setattr(agent, "_neatlogs_patched", True)
        except Exception:
            pass
    return agent


def release_strands(provider: Any, owner: Any) -> None:
    try:
        from strands.telemetry import tracer as tracer_module
    except Exception:
        return

    global _CONTEXTUAL_TRACER, _PREVIOUS_TRACER
    with _LOCK:
        contextual = _CONTEXTUAL_TRACER
        for agent, (previous, agent_owner) in list(_WRAPPED_AGENTS.items()):
            if agent_owner is owner:
                if getattr(agent, "tracer", None) is contextual:
                    agent.tracer = previous
                _WRAPPED_AGENTS.pop(agent, None)

        owners = _PROVIDER_OWNERS.get(provider)
        if owners is not None:
            owners.discard(owner)
            if not owners:
                _PROVIDER_OWNERS.pop(provider, None)
                _PROVIDER_TRACERS.pop(provider, None)
                _PROVIDER_PROCESSORS.pop(provider, None)
                remove_provider_preprocessor(provider, "strands")

        if _PROVIDER_OWNERS or _WRAPPED_AGENTS:
            return
        if contextual is not None and tracer_module._tracer_instance is contextual:
            tracer_module._tracer_instance = _PREVIOUS_TRACER
        _CONTEXTUAL_TRACER = None
        _PREVIOUS_TRACER = None


def release_default_strands(provider: Any) -> None:
    release_strands(provider, _DEFAULT_OWNER)


def uninstrument_strands() -> None:
    """Restore the Strands singleton and explicitly wrapped agents."""
    try:
        from strands.telemetry import tracer as tracer_module
    except Exception:
        return

    global _CONTEXTUAL_TRACER, _PREVIOUS_TRACER
    with _LOCK:
        contextual = _CONTEXTUAL_TRACER
        if contextual is not None and tracer_module._tracer_instance is contextual:
            tracer_module._tracer_instance = _PREVIOUS_TRACER
        for agent, (previous, _) in list(_WRAPPED_AGENTS.items()):
            if getattr(agent, "tracer", None) is contextual:
                agent.tracer = previous
        _WRAPPED_AGENTS.clear()
        _PROVIDER_TRACERS.clear()
        for provider in list(_PROVIDER_PROCESSORS):
            remove_provider_preprocessor(provider, "strands")
        _PROVIDER_PROCESSORS.clear()
        _PROVIDER_OWNERS.clear()
        _CONTEXTUAL_TRACER = None
        _PREVIOUS_TRACER = None
