"""Neatlogs integration for Strands Agents."""

import copy
import threading
import weakref
from typing import Any, Optional

from openinference.instrumentation.strands_agents import (
    StrandsAgentsToOpenInferenceProcessor,
)

from .instrumentation.openinference_isolation import provider_for_openinference

_LOCK = threading.RLock()
_ACTIVE_PROVIDER: Optional[Any] = None
_ACTIVE_TRACER: Optional[Any] = None
_PREVIOUS_TRACER: Optional[Any] = None
_WRAPPED_AGENTS: list[tuple[Any, Any]] = []
_PROVIDER_PROCESSORS: "weakref.WeakKeyDictionary[Any, Any]" = weakref.WeakKeyDictionary()


class _NeatlogsStrandsProcessor(StrandsAgentsToOpenInferenceProcessor):
    def on_end(self, span: Any) -> None:
        status = span.status
        interrupted = bool(
            (getattr(span, "attributes", None) or {}).get("neatlogs.trace.interrupted")
        )
        super().on_end(span)
        if interrupted:
            # The upstream converter marks non-error spans OK. An interrupted
            # Neatlogs span deliberately retains the framework's prior status.
            span._status = status


def prepare_strands(provider: Any) -> bool:
    """Install the Strands-to-OpenInference processor before export processors."""
    with _LOCK:
        try:
            if provider in _PROVIDER_PROCESSORS:
                return True
        except TypeError:
            pass

        processor = _NeatlogsStrandsProcessor()
        provider.add_span_processor(processor)
        try:
            _PROVIDER_PROCESSORS[provider] = processor
        except TypeError:
            pass
    return True


def instrument_strands(provider: Any) -> bool:
    """Route future Strands agents through the current Neatlogs provider."""
    from strands.telemetry import tracer as tracer_module
    from strands.telemetry.tracer import Tracer

    prepare_strands(provider)
    oi_provider = provider_for_openinference(provider)

    global _ACTIVE_PROVIDER, _ACTIVE_TRACER, _PREVIOUS_TRACER
    with _LOCK:
        if (
            _ACTIVE_PROVIDER is provider
            and _ACTIVE_TRACER is not None
            and tracer_module._tracer_instance is _ACTIVE_TRACER
        ):
            return True

        _restore_strands_locked(tracer_module)
        previous = tracer_module._tracer_instance
        tracer = copy.copy(previous) if previous is not None else Tracer()
        tracer.tracer_provider = oi_provider
        tracer.tracer = oi_provider.get_tracer(tracer.service_name)
        tracer_module._tracer_instance = tracer

        _PREVIOUS_TRACER = previous
        _ACTIVE_TRACER = tracer
        _ACTIVE_PROVIDER = provider
    return True


def strands_hooks(agent: Any) -> Any:
    """Route an existing Strands agent through Neatlogs and return it unchanged."""
    from .init import _instrument_library

    _instrument_library("strands")
    with _LOCK:
        tracer = _ACTIVE_TRACER
        if tracer is not None and getattr(agent, "tracer", None) is not tracer:
            _remember_agent(agent, getattr(agent, "tracer", None))
            agent.tracer = tracer
        try:
            setattr(agent, "_neatlogs_patched", True)
        except Exception:
            pass
    return agent


def uninstrument_strands() -> None:
    """Restore the Strands singleton and explicitly wrapped agents."""
    try:
        from strands.telemetry import tracer as tracer_module
    except Exception:
        return

    with _LOCK:
        _restore_strands_locked(tracer_module)


def _remember_agent(agent: Any, previous_tracer: Any) -> None:
    for reference, _ in _WRAPPED_AGENTS:
        if reference() is agent:
            return
    try:
        reference = weakref.ref(agent)
    except TypeError:
        reference = lambda: agent
    _WRAPPED_AGENTS.append((reference, previous_tracer))


def _restore_strands_locked(tracer_module: Any) -> None:
    global _ACTIVE_PROVIDER, _ACTIVE_TRACER, _PREVIOUS_TRACER
    active = _ACTIVE_TRACER
    if active is None:
        return

    if tracer_module._tracer_instance is active:
        tracer_module._tracer_instance = _PREVIOUS_TRACER

    for reference, previous in _WRAPPED_AGENTS:
        agent = reference()
        if agent is not None and getattr(agent, "tracer", None) is active:
            agent.tracer = previous
    _WRAPPED_AGENTS.clear()
    _ACTIVE_PROVIDER = None
    _ACTIVE_TRACER = None
    _PREVIOUS_TRACER = None
