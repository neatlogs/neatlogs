"""Encoded-byte-aware batching for OTLP/protobuf span export."""

from __future__ import annotations

from collections.abc import Sequence

from opentelemetry.exporter.otlp.proto.common.trace_encoder import encode_spans
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

from .delivery import DeliveryDiagnostics

DEFAULT_MAX_EXPORT_BYTES = 4 * 1024 * 1024


class ByteLimitedSpanExporter(SpanExporter):
    """Split a row batch by a conservative encoded protobuf byte bound.

    Encoding each span as a one-span OTLP request repeats resource and scope
    framing that a combined request deduplicates. Summing those sizes is
    therefore a safe upper bound without repeatedly encoding every growing
    candidate batch. A single oversized span is forwarded intact: Phase 8 owns
    the overflow claim-check, and this layer never truncates or silently drops.
    """

    def __init__(
        self,
        inner: SpanExporter,
        max_export_bytes: int = DEFAULT_MAX_EXPORT_BYTES,
        diagnostics: DeliveryDiagnostics | None = None,
    ) -> None:
        if max_export_bytes <= 0:
            raise ValueError("max_export_bytes must be greater than zero")
        self._inner = inner
        self._max_export_bytes = max_export_bytes
        self._diagnostics = diagnostics

    @staticmethod
    def _encoded_upper_bound(span: ReadableSpan) -> int:
        return encode_spans((span,)).ByteSize()

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        batches: list[list[ReadableSpan]] = []
        current: list[ReadableSpan] = []
        current_bytes = 0

        for span in spans:
            span_bytes = self._encoded_upper_bound(span)
            if current and current_bytes + span_bytes > self._max_export_bytes:
                batches.append(current)
                current = []
                current_bytes = 0
            current.append(span)
            current_bytes += span_bytes

        if current:
            batches.append(current)

        for batch in batches:
            if self._inner.export(batch) is not SpanExportResult.SUCCESS:
                if self._diagnostics is not None:
                    self._diagnostics.record_export_failure("span", len(batch))
                return SpanExportResult.FAILURE
        return SpanExportResult.SUCCESS

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        result = self._inner.force_flush(timeout_millis)
        return True if result is None else bool(result)

    def shutdown(self) -> None:
        self._inner.shutdown()
