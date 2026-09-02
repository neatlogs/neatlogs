"""Encoded-byte-aware batching for OTLP/protobuf log export."""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Sequence

from opentelemetry.exporter.otlp.proto.common._log_encoder import encode_logs

try:
    from opentelemetry.sdk._logs import ReadableLogRecord
    from opentelemetry.sdk._logs.export import LogRecordExporter, LogRecordExportResult
except ImportError:  # OpenTelemetry 1.35-1.38 compatibility
    from opentelemetry.sdk._logs import LogData as ReadableLogRecord
    from opentelemetry.sdk._logs.export import (
        LogExporter as LogRecordExporter,
        LogExportResult as LogRecordExportResult,
    )

from .byte_limited_exporter import DEFAULT_MAX_EXPORT_BYTES
from .capture import limit_log_capture
from .delivery import DeliveryDiagnostics
from .upload_authority import (
    DisabledUploadAuthority,
    OverflowExportReceipt,
    OverflowPayload,
    UploadAuthority,
    UploadError,
)

logger = logging.getLogger(__name__)


class ByteLimitedLogExporter(LogRecordExporter):
    """Split log batches and explicitly reject unsafe single-record overflow."""

    def __init__(
        self,
        inner: LogRecordExporter,
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
                "log",
                self._upload_authority.available,
                self._upload_authority.unavailable_reason,
            )

    @staticmethod
    def _encoded_upper_bound(item: ReadableLogRecord) -> int:
        return encode_logs((item,)).ByteSize()

    def export(self, batch: Sequence[ReadableLogRecord]) -> LogRecordExportResult:
        bounded = []
        for item in batch:
            # The v1 upload contract only accepts OTLP trace envelopes. Preserve
            # the Phase 3 bounded log path even when trace/media uploads are on.
            clone, truncations = limit_log_capture(item)
            bounded.append(clone)
            if truncations and self._diagnostics is not None:
                self._diagnostics.record_capture_truncation("log", truncations)

        actions: list[tuple[str, object]] = []
        current = []
        current_bytes = 0
        overflow_failed = False
        for item in bounded:
            item_bytes = self._encoded_upper_bound(item)
            if item_bytes > self._max_export_bytes:
                if current:
                    actions.append(("batch", current))
                    current = []
                    current_bytes = 0
                payload = encode_logs((item,)).SerializeToString()
                actions.append(
                    (
                        "overflow",
                        OverflowPayload(
                            content=payload,
                            sha256=hashlib.sha256(payload).hexdigest(),
                            byte_length=len(payload),
                            signal="log",
                        ),
                    )
                )
                continue
            if current and current_bytes + item_bytes > self._max_export_bytes:
                actions.append(("batch", current))
                current = []
                current_bytes = 0
            current.append(item)
            current_bytes += item_bytes
        if current:
            actions.append(("batch", current))

        for index, (action_type, value) in enumerate(actions):
            if action_type == "overflow":
                candidate = value
                if not self._upload_authority.available:
                    overflow_failed = True
                    if self._diagnostics is not None:
                        self._diagnostics.record_overflow("log", "unavailable")
                        self._diagnostics.record_overflow("log", "failures")
                        self._diagnostics.record_export_failure("log", 1)
                    logger.error(
                        "[neatlogs] oversized log rejected: bytes=%d limit=%d reason=%s",
                        candidate.byte_length,
                        self._max_export_bytes,
                        self._upload_authority.unavailable_reason,
                    )
                    continue
                try:
                    result = self._upload_authority.export_overflow(candidate)
                    succeeded = isinstance(result, OverflowExportReceipt) and result.complete
                except UploadError as exc:
                    if self._diagnostics is not None:
                        self._diagnostics.record_upload_failure(f"{exc.stage}:{exc.reason_code}")
                    logger.error(
                        "[neatlogs] oversized log upload failed (%s)",
                        type(exc).__name__,
                    )
                    succeeded = False
                except Exception as exc:
                    if self._diagnostics is not None:
                        self._diagnostics.record_upload_failure("unexpected_error")
                    logger.error(
                        "[neatlogs] oversized log upload failed (%s)",
                        type(exc).__name__,
                    )
                    succeeded = False
                if succeeded:
                    if self._diagnostics is not None:
                        self._diagnostics.record_overflow("log", "exports")
                else:
                    overflow_failed = True
                    if self._diagnostics is not None:
                        self._diagnostics.record_overflow("log", "failures")
                        self._diagnostics.record_export_failure("log", 1)
                continue

            records = value
            try:
                result = self._inner.export(records)
            except Exception as exc:
                logger.error(
                    "[neatlogs] log batch export raised (%s)",
                    type(exc).__name__,
                )
                result = LogRecordExportResult.FAILURE
            if result is not LogRecordExportResult.SUCCESS:
                if self._diagnostics is not None:
                    self._diagnostics.record_export_failure(
                        "log",
                        sum(
                            len(unattempted) if kind == "batch" else 1
                            for kind, unattempted in actions[index:]
                        ),
                    )
                return LogRecordExportResult.FAILURE
        return LogRecordExportResult.FAILURE if overflow_failed else LogRecordExportResult.SUCCESS

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        force_flush = getattr(self._inner, "force_flush", None)
        if force_flush is None:
            return True
        result = force_flush(timeout_millis)
        return True if result is None else bool(result)

    def shutdown(self) -> None:
        self._inner.shutdown()
