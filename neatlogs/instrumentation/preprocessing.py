"""Provider-local preprocessing that must run before Neatlogs export."""

import threading
import weakref
from typing import Any

from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor
from opentelemetry.trace import Span

_LOCK = threading.RLock()
_HUBS: "weakref.WeakKeyDictionary[Any, ProviderPreprocessor]" = weakref.WeakKeyDictionary()


class ProviderPreprocessor(SpanProcessor):
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._processors: dict[str, SpanProcessor] = {}

    def add(self, key: str, processor: SpanProcessor) -> SpanProcessor:
        with self._lock:
            existing = self._processors.get(key)
            if existing is not None:
                processor.shutdown()
                return existing
            self._processors[key] = processor
            return processor

    def remove(self, key: str) -> None:
        with self._lock:
            processor = self._processors.pop(key, None)
        if processor is not None:
            processor.shutdown()

    def _snapshot(self) -> tuple[SpanProcessor, ...]:
        with self._lock:
            return tuple(self._processors.values())

    def is_empty(self) -> bool:
        with self._lock:
            return not self._processors

    def on_start(self, span: Span, parent_context=None) -> None:
        for processor in self._snapshot():
            processor.on_start(span, parent_context)

    def on_end(self, span: ReadableSpan) -> None:
        for processor in self._snapshot():
            processor.on_end(span)

    def shutdown(self) -> None:
        with self._lock:
            processors = tuple(self._processors.values())
            self._processors.clear()
        for processor in processors:
            processor.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        succeeded = True
        for processor in self._snapshot():
            if processor.force_flush(timeout_millis) is False:
                succeeded = False
        return succeeded


def ensure_provider_preprocessor(provider: Any) -> ProviderPreprocessor:
    with _LOCK:
        existing = _HUBS.get(provider)
        if existing is not None:
            return existing
        hub = ProviderPreprocessor()
        multi = getattr(provider, "_active_span_processor", None)
        processors = getattr(multi, "_span_processors", None)
        lock = getattr(multi, "_lock", None)
        if processors is None or lock is None:
            if processors:
                raise RuntimeError(
                    "Cannot install provider preprocessing before existing processors"
                )
            provider.add_span_processor(hub)
        else:
            with lock:
                multi._span_processors = (hub,) + tuple(multi._span_processors)
        _HUBS[provider] = hub
        return hub


def add_provider_preprocessor(provider: Any, key: str, processor: SpanProcessor) -> SpanProcessor:
    return ensure_provider_preprocessor(provider).add(key, processor)


def remove_provider_preprocessor(provider: Any, key: str) -> None:
    with _LOCK:
        hub = _HUBS.get(provider)
        if hub is None:
            return
        hub.remove(key)
        if not hub.is_empty():
            return
        _HUBS.pop(provider, None)
        multi = getattr(provider, "_active_span_processor", None)
        processors = getattr(multi, "_span_processors", None)
        lock = getattr(multi, "_lock", None)
        if processors is not None and lock is not None:
            with lock:
                multi._span_processors = tuple(
                    processor for processor in multi._span_processors if processor is not hub
                )
