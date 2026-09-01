"""Encoded-byte-aware batching for OTLP/protobuf span export."""

from __future__ import annotations

import gzip
import hashlib
import logging
from collections.abc import Sequence

from opentelemetry.exporter.otlp.proto.common.trace_encoder import encode_spans
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

from .capture import limit_span_capture
from .delivery import DeliveryDiagnostics
from .upload_authority import (
    DEFAULT_MAX_OVERFLOW_EXPANDED_BYTES,
    DEFAULT_MAX_OVERFLOW_UPLOAD_BYTES,
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
            # Only a span that must cross the overflow boundary retains its full,
            # already-masked payload. Ordinary spans keep the Phase 3 per-value
            # and collection limits even when uploads are enabled.
            if (
                self._upload_authority.available
                and self._encoded_upper_bound(span) > self._max_export_bytes
            ):
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
                if self._upload_authority.available and span_bytes > getattr(
                    self._upload_authority,
                    "max_overflow_expanded_bytes",
                    DEFAULT_MAX_OVERFLOW_EXPANDED_BYTES,
                ):
                    actions.append(
                        (
                            "overflow_too_large",
                            (
                                span_bytes,
                                getattr(
                                    self._upload_authority,
                                    "max_overflow_expanded_bytes",
                                    DEFAULT_MAX_OVERFLOW_EXPANDED_BYTES,
                                ),
                            ),
                        )
                    )
                    continue
                # Defer serialization/compression until execution so a full
                # batch cannot retain hundreds of 20 MiB compressed payloads.
                actions.append(("overflow_span", (span, span_bytes)))
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
            if action_type == "overflow_too_large":
                rejected_bytes, rejected_limit = value
                overflow_failed = True
                if self._diagnostics is not None:
                    self._diagnostics.record_upload_failure("prepare:invalid_byte_length")
                    self._diagnostics.record_overflow("span", "failures")
                    self._diagnostics.record_export_failure("span", 1)
                logger.error(
                    "[neatlogs] oversized span rejected: bytes=%d upload_limit=%d",
                    rejected_bytes,
                    rejected_limit,
                )
                continue
            if action_type == "overflow_span":
                overflow_span, expanded_bytes = value
                if not self._upload_authority.available:
                    overflow_failed = True
                    if self._diagnostics is not None:
                        self._diagnostics.record_overflow("span", "unavailable")
                        self._diagnostics.record_overflow("span", "failures")
                        self._diagnostics.record_export_failure("span", 1)
                    logger.error(
                        "[neatlogs] oversized span rejected: bytes=%d limit=%d reason=%s",
                        expanded_bytes,
                        self._max_export_bytes,
                        self._upload_authority.unavailable_reason,
                    )
                    continue
                expanded_payload = encode_spans((overflow_span,)).SerializeToString()
                compressed = gzip.compress(expanded_payload, mtime=0)
                max_compressed = getattr(
                    self._upload_authority,
                    "max_overflow_bytes",
                    DEFAULT_MAX_OVERFLOW_UPLOAD_BYTES,
                )
                if len(compressed) > max_compressed:
                    overflow_failed = True
                    if self._diagnostics is not None:
                        self._diagnostics.record_upload_failure("prepare:invalid_byte_length")
                        self._diagnostics.record_overflow("span", "failures")
                        self._diagnostics.record_export_failure("span", 1)
                    logger.error(
                        "[neatlogs] oversized span rejected: bytes=%d upload_limit=%d",
                        len(compressed),
                        max_compressed,
                    )
                    continue
                candidate = OverflowPayload(
                    content=compressed,
                    sha256=hashlib.sha256(compressed).hexdigest(),
                    byte_length=len(compressed),
                    signal="span",
                    content_encoding="gzip",
                )
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
