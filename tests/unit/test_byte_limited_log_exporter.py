import pytest
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import (
    InMemoryLogRecordExporter,
    LogRecordExportResult,
    SimpleLogRecordProcessor,
)

from neatlogs.core.byte_limited_log_exporter import ByteLimitedLogExporter
from neatlogs.core.delivery import DeliveryDiagnostics
from neatlogs.core.upload_authority import UploadError


class RecordingExporter:
    def __init__(self, results=None):
        self.batches = []
        self.results = iter(results) if results is not None else None

    def export(self, batch):
        self.batches.append(list(batch))
        if self.results is None:
            return LogRecordExportResult.SUCCESS
        return next(self.results)

    def force_flush(self, timeout_millis=30000):
        return True

    def shutdown(self):
        return None


def _finished_logs(count=3, payload_size=2048):
    sink = InMemoryLogRecordExporter()
    provider = LoggerProvider()
    provider.add_log_record_processor(SimpleLogRecordProcessor(sink))
    logger = provider.get_logger("byte-test")
    for index in range(count):
        logger.emit(body=f"{index}:" + "x" * payload_size)
    provider.shutdown()
    return list(sink.get_finished_logs())


def test_splits_log_batches_using_encoded_protobuf_upper_bound():
    records = _finished_logs()
    one_record_bytes = max(ByteLimitedLogExporter._encoded_upper_bound(item) for item in records)
    sink = RecordingExporter()
    exporter = ByteLimitedLogExporter(sink, max_export_bytes=one_record_bytes * 2)

    assert exporter.export(records) is LogRecordExportResult.SUCCESS
    assert [len(batch) for batch in sink.batches] == [2, 1]


def test_rejects_oversized_log_when_upload_authority_is_unavailable():
    records = _finished_logs(count=1, payload_size=16_384)
    diagnostics = DeliveryDiagnostics()
    sink = RecordingExporter()
    exporter = ByteLimitedLogExporter(sink, max_export_bytes=128, diagnostics=diagnostics)

    assert exporter.export(records) is LogRecordExportResult.FAILURE
    assert sink.batches == []
    snapshot = diagnostics.snapshot()
    assert snapshot["log_overflow_unavailable"] == 1
    assert snapshot["log_overflow_failures"] == 1
    assert snapshot["log_export_failures"] == 1


@pytest.mark.parametrize(
    ("results", "expected_attempts", "expected_failures"),
    [
        ([LogRecordExportResult.FAILURE], 1, 3),
        ([LogRecordExportResult.SUCCESS, LogRecordExportResult.FAILURE], 2, 2),
        (
            [
                LogRecordExportResult.SUCCESS,
                LogRecordExportResult.SUCCESS,
                LogRecordExportResult.FAILURE,
            ],
            3,
            1,
        ),
    ],
)
def test_log_split_failure_counts_failed_and_unattempted_tail(
    results, expected_attempts, expected_failures
):
    records = _finished_logs(count=3, payload_size=256)
    max_bytes = max(ByteLimitedLogExporter._encoded_upper_bound(item) for item in records)
    sink = RecordingExporter(results)
    diagnostics = DeliveryDiagnostics()
    exporter = ByteLimitedLogExporter(
        sink,
        max_export_bytes=max_bytes,
        diagnostics=diagnostics,
    )

    assert exporter.export(records) is LogRecordExportResult.FAILURE
    assert len(sink.batches) == expected_attempts
    assert diagnostics.snapshot()["log_export_failures"] == expected_failures


def test_log_exporter_exception_counts_current_and_unattempted_tail():
    records = _finished_logs(count=3, payload_size=256)
    max_bytes = max(ByteLimitedLogExporter._encoded_upper_bound(item) for item in records)

    class RaisingExporter(RecordingExporter):
        def export(self, batch):
            self.batches.append(list(batch))
            raise RuntimeError("transport failed")

    diagnostics = DeliveryDiagnostics()
    sink = RaisingExporter()
    exporter = ByteLimitedLogExporter(
        sink,
        max_export_bytes=max_bytes,
        diagnostics=diagnostics,
    )

    assert exporter.export(records) is LogRecordExportResult.FAILURE
    assert len(sink.batches) == 1
    assert diagnostics.snapshot()["log_export_failures"] == 3


def test_unsupported_log_overflow_exposes_a_safe_failure_reason():
    records = _finished_logs(count=1, payload_size=16_384)

    class Authority:
        available = True
        unavailable_reason = ""

        def export_overflow(self, _payload):
            raise UploadError("prepare", "overflow_signal_unsupported")

    diagnostics = DeliveryDiagnostics()
    sink = RecordingExporter()
    exporter = ByteLimitedLogExporter(
        sink,
        max_export_bytes=128,
        diagnostics=diagnostics,
        upload_authority=Authority(),
    )

    assert exporter.export(records) is LogRecordExportResult.FAILURE
    assert sink.batches == []
    snapshot = diagnostics.snapshot()
    assert snapshot["log_overflow_failures"] == 1
    assert snapshot["upload_last_failure_reason"] == "prepare:overflow_signal_unsupported"
