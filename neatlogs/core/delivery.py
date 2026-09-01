"""Bounded-queue and final-delivery loss counters for NeatLogs pipelines."""

from __future__ import annotations

import threading
from dataclasses import asdict, dataclass
from typing import Any

from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.trace.export import BatchSpanProcessor


@dataclass
class DeliveryDiagnosticsSnapshot:
    span_queue_drops: int = 0
    log_queue_drops: int = 0
    span_export_failures: int = 0
    log_export_failures: int = 0
    masked_span_drops: int = 0
    masked_log_drops: int = 0
    span_capture_truncations: int = 0
    log_capture_truncations: int = 0
    span_overflow_exports: int = 0
    span_overflow_failures: int = 0
    span_overflow_unavailable: int = 0
    log_overflow_exports: int = 0
    log_overflow_failures: int = 0
    log_overflow_unavailable: int = 0
    span_upload_authority_available: bool = False
    log_upload_authority_available: bool = False
    span_upload_authority_reason: str = "backend_upload_contract_unavailable"
    log_upload_authority_reason: str = "backend_upload_contract_unavailable"


class DeliveryDiagnostics:
    """Thread-safe per-pipeline counters retained for doctor diagnostics."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._values = DeliveryDiagnosticsSnapshot()

    def record_queue_drop(self, signal: str, count: int = 1) -> None:
        with self._lock:
            field = "span_queue_drops" if signal == "span" else "log_queue_drops"
            setattr(self._values, field, getattr(self._values, field) + count)

    def record_export_failure(self, signal: str, count: int) -> None:
        with self._lock:
            field = "span_export_failures" if signal == "span" else "log_export_failures"
            setattr(self._values, field, getattr(self._values, field) + count)

    def record_masked_drop(self, signal: str, count: int = 1) -> None:
        with self._lock:
            field = "masked_span_drops" if signal == "span" else "masked_log_drops"
            setattr(self._values, field, getattr(self._values, field) + count)

    def record_capture_truncation(self, signal: str, count: int = 1) -> None:
        with self._lock:
            field = "span_capture_truncations" if signal == "span" else "log_capture_truncations"
            setattr(self._values, field, getattr(self._values, field) + count)

    def record_overflow(self, signal: str, outcome: str, count: int = 1) -> None:
        if outcome not in {"exports", "failures", "unavailable"}:
            raise ValueError(f"unknown overflow outcome: {outcome}")
        with self._lock:
            field = f"{signal}_overflow_{outcome}"
            setattr(self._values, field, getattr(self._values, field) + count)

    def configure_upload_authority(self, signal: str, available: bool, reason: str) -> None:
        with self._lock:
            setattr(self._values, f"{signal}_upload_authority_available", available)
            setattr(self._values, f"{signal}_upload_authority_reason", reason)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return asdict(self._values)


class ObservableBatchSpanProcessor(BatchSpanProcessor):
    def __init__(self, *args, diagnostics: DeliveryDiagnostics, **kwargs) -> None:
        self._neatlogs_diagnostics = diagnostics
        super().__init__(*args, **kwargs)

    def on_end(self, span) -> None:
        if span.context and span.context.trace_flags.sampled:
            processor = self._batch_processor
            if len(processor._queue) >= processor._max_queue_size:
                self._neatlogs_diagnostics.record_queue_drop("span")
        super().on_end(span)


class ObservableBatchLogRecordProcessor(BatchLogRecordProcessor):
    def __init__(self, *args, diagnostics: DeliveryDiagnostics, **kwargs) -> None:
        self._neatlogs_diagnostics = diagnostics
        super().__init__(*args, **kwargs)

    def on_emit(self, log_record) -> None:
        processor = self._batch_processor
        if len(processor._queue) >= processor._max_queue_size:
            self._neatlogs_diagnostics.record_queue_drop("log")
        super().on_emit(log_record)
