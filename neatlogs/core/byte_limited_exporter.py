"""Encoded-byte-aware batching for OTLP/protobuf span export."""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Sequence

from opentelemetry.exporter.otlp.proto.common.trace_encoder import encode_spans
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

from .capture import limit_span_capture
from .delivery import DeliveryDiagnostics
from .upload_authority import (
    DisabledUploadAuthority,
    OverflowExportReceipt,
    OverflowPayload,
    UploadAuthority,
    UploadError,
)

DEFAULT_MAX_EXPORT_BYTES = 4 * 1024 * 1024
logger = logging.getLogger(__name__)


class ByteLimitedSpanExporter(SpanExporter):
    """Split a row batch by a conservative encoded protobuf byte bound.

    Encoding each span as a one-span OTLP request repeats resource and scope
    framing that a combined request deduplicates. Summing those sizes is
    therefore a safe upper bound without repeatedly encoding every growing
    candidate batch. A single oversized span crosses the upload-authority
    boundary when explicitly enabled; the default-off path rejects it.
    """

    def __init__(
        self,
        inner: SpanExporter,
        max_export_bytes: int = DEFAULT_MAX_EXPORT_BYTES,
        diagnostics: DeliveryDiagnostics | None = None,
        upload_authority: UploadAuthority | None = None,
    ) -> None:
        if max_export_bytes <= 0:
            raise ValueError("max_export_bytes must be greater than zero")
        self._inner = inner
        self._max_export_bytes = max_export_bytes
        self._diagnostics = diagnostics
        self._upload_authority = upload_authority or DisabledUploadAuthority()
        if diagnostics is not None:
            diagnostics.configure_upload_authority(
                "span",
                self._upload_authority.available,
                self._upload_authority.unavailable_reason,
            )

    @staticmethod
    def _encoded_upper_bound(span: ReadableSpan) -> int:
        return encode_spans((span,)).ByteSize()

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        bounded_spans: list[ReadableSpan] = []
        for span in spans:
            if self._upload_authority.available:
                bounded, truncations = span, 0
            else:
                bounded, truncations = limit_span_capture(span)
            bounded_spans.append(bounded)
            if truncations and self._diagnostics is not None:
                self._diagnostics.record_capture_truncation("span", truncations)

        actions: list[tuple[str, object]] = []
        current: list[ReadableSpan] = []
        current_bytes = 0
        overflow_failed = False

        for span in bounded_spans:
            span_bytes = self._encoded_upper_bound(span)
            if span_bytes > self._max_export_bytes:
                if current:
                    actions.append(("batch", current))
                    current = []
                    current_bytes = 0
                payload = encode_spans((span,)).SerializeToString()
                actions.append(
                    (
                        "overflow",
                        OverflowPayload(
                            content=payload,
                            sha256=hashlib.sha256(payload).hexdigest(),
                            byte_length=len(payload),
                            signal="span",
                        ),
                    )
                )
                continue
            if current and current_bytes + span_bytes > self._max_export_bytes:
                actions.append(("batch", current))
                current = []
                current_bytes = 0
            current.append(span)
            current_bytes += span_bytes

        if current:
            actions.append(("batch", current))

        for index, (action_type, value) in enumerate(actions):
            if action_type == "overflow":
                candidate = value
                if not self._upload_authority.available:
                    overflow_failed = True
                    if self._diagnostics is not None:
                        self._diagnostics.record_overflow("span", "unavailable")
                        self._diagnostics.record_overflow("span", "failures")
                        self._diagnostics.record_export_failure("span", 1)
                    logger.error(
                        "[neatlogs] oversized span rejected: bytes=%d limit=%d reason=%s",
                        candidate.byte_length,
                        self._max_export_bytes,
                        self._upload_authority.unavailable_reason,
                    )
                    continue
                try:
                    result = self._upload_authority.export_overflow(candidate)
                except UploadError as exc:
                    if self._diagnostics is not None:
                        self._diagnostics.record_upload_failure(f"{exc.stage}:{exc.reason_code}")
                    logger.error(
                        "[neatlogs] oversized span upload failed (%s)",
                        type(exc).__name__,
                    )
                    result = SpanExportResult.FAILURE
                except Exception as exc:
                    if self._diagnostics is not None:
                        self._diagnostics.record_upload_failure("unexpected_error")
                    logger.error(
                        "[neatlogs] oversized span upload failed (%s)",
                        type(exc).__name__,
                    )
                    result = SpanExportResult.FAILURE
                if isinstance(result, OverflowExportReceipt) and result.complete:
                    if self._diagnostics is not None:
                        self._diagnostics.record_overflow("span", "exports")
                else:
                    overflow_failed = True
                    if self._diagnostics is not None:
                        self._diagnostics.record_overflow("span", "failures")
                        self._diagnostics.record_export_failure("span", 1)
                continue

            batch = value
            try:
                result = self._inner.export(batch)
            except Exception as exc:
                logger.error(
                    "[neatlogs] span batch export raised (%s)",
                    type(exc).__name__,
                )
                result = SpanExportResult.FAILURE
            if result is not SpanExportResult.SUCCESS:
                if self._diagnostics is not None:
                    self._diagnostics.record_export_failure(
                        "span",
                        sum(
                            len(unattempted) if kind == "batch" else 1
                            for kind, unattempted in actions[index:]
                        ),
                    )
                return SpanExportResult.FAILURE
        return SpanExportResult.FAILURE if overflow_failed else SpanExportResult.SUCCESS

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        result = self._inner.force_flush(timeout_millis)
        return True if result is None else bool(result)

    def shutdown(self) -> None:
        self._inner.shutdown()
