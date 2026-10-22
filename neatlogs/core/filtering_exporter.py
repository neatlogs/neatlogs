"""Final export filtering for unsupported HTTP transport spans."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

from .media_exporter import release_span_media
from .span_processor import is_http_span


class HttpFilteringSpanExporter(SpanExporter):
    """Drop HTTP spans while releasing their private media allocations."""

    def __init__(
        self,
        inner: SpanExporter,
        *,
        media_store=None,
        on_accepted: Callable[[ReadableSpan], None] | None = None,
        on_rejected: Callable[[ReadableSpan], None] | None = None,
    ) -> None:
        self._inner = inner
        self._media_store = media_store
        self._on_accepted = on_accepted
        self._on_rejected = on_rejected

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        kept: list[ReadableSpan] = []
        for span in spans:
            if is_http_span(span):
                release_span_media(self._media_store, span)
                if self._on_rejected is not None:
                    self._on_rejected(span)
                continue
            kept.append(span)
        if not kept:
            return SpanExportResult.SUCCESS
        result = self._inner.export(kept)
        callback = self._on_accepted if result == SpanExportResult.SUCCESS else self._on_rejected
        if callback is not None:
            for span in kept:
                callback(span)
        return result

    def shutdown(self) -> None:
        self._inner.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        result = self._inner.force_flush(timeout_millis)
        return True if result is None else bool(result)
