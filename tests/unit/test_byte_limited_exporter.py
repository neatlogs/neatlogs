import hashlib

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExportResult

from neatlogs.core.byte_limited_exporter import ByteLimitedSpanExporter
from neatlogs.core.delivery import DeliveryDiagnostics
from neatlogs.core.upload_authority import OverflowExportReceipt


class RecordingExporter:
    def __init__(self, result=SpanExportResult.SUCCESS, results=None):
        self.batches = []
        self.result = result
        self.results = iter(results) if results is not None else None

    def export(self, spans):
        self.batches.append(list(spans))
        return next(self.results) if self.results is not None else self.result

    def force_flush(self, timeout_millis=30000):
        return True

    def shutdown(self):
        return None


def _finished_spans(count=3, payload_size=2048):
    sink = RecordingExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(sink))
    tracer = provider.get_tracer("byte-test")
    for index in range(count):
        with tracer.start_as_current_span(f"span-{index}") as span:
            span.set_attribute("neatlogs.llm.input", "x" * payload_size)
    return [span for batch in sink.batches for span in batch]


def test_splits_batches_using_encoded_protobuf_upper_bound():
    spans = _finished_spans()
    one_span_bytes = ByteLimitedSpanExporter._encoded_upper_bound(spans[0])
    sink = RecordingExporter()
    exporter = ByteLimitedSpanExporter(sink, max_export_bytes=one_span_bytes * 2)

    assert exporter.export(spans) is SpanExportResult.SUCCESS
    assert [len(batch) for batch in sink.batches] == [2, 1]


def test_rejects_one_oversized_span_when_backend_upload_authority_is_unavailable():
    spans = _finished_spans(count=1, payload_size=16_384)
    sink = RecordingExporter()
    diagnostics = DeliveryDiagnostics()
    exporter = ByteLimitedSpanExporter(sink, max_export_bytes=128, diagnostics=diagnostics)

    assert exporter.export(spans) is SpanExportResult.FAILURE
    assert sink.batches == []
    snapshot = diagnostics.snapshot()
    assert snapshot["span_overflow_unavailable"] == 1
    assert snapshot["span_overflow_failures"] == 1
    assert snapshot["span_export_failures"] == 1


def test_injectable_upload_authority_receives_complete_masked_envelope():
    spans = _finished_spans(count=1, payload_size=16_384)

    class Authority:
        available = True
        unavailable_reason = ""

        def __init__(self):
            self.payloads = []

        def export_overflow(self, payload):
            self.payloads.append(payload)
            return OverflowExportReceipt(
                upload_id="upload-1",
                project_id="project-1",
                state="ready",
                reference_exported=True,
            )

    authority = Authority()
    diagnostics = DeliveryDiagnostics()
    exporter = ByteLimitedSpanExporter(
        RecordingExporter(),
        max_export_bytes=128,
        diagnostics=diagnostics,
        upload_authority=authority,
    )

    assert exporter.export(spans) is SpanExportResult.SUCCESS
    assert len(authority.payloads) == 1
    payload = authority.payloads[0]
    assert payload.purpose == "otlp_overflow"
    assert payload.byte_length == len(payload.content)
    assert payload.sha256 == hashlib.sha256(payload.content).hexdigest()
    snapshot = diagnostics.snapshot()
    assert snapshot["span_overflow_exports"] == 1
    assert snapshot["span_upload_authority_available"] is True


def test_oversized_upload_contains_the_complete_masked_span_and_is_not_sent_twice():
    from neatlogs.core.masking_exporter import MaskingSpanExporter

    spans = _finished_spans(count=1, payload_size=16_384)

    class Authority:
        available = True
        unavailable_reason = ""

        def __init__(self):
            self.payloads = []

        def export_overflow(self, payload):
            self.payloads.append(payload)
            return OverflowExportReceipt(upload_id="upload-1")

    def mask(snapshot):
        snapshot["attributes"]["neatlogs.llm.input"] = "MASKED" * 3_000
        return snapshot

    authority = Authority()
    ordinary = RecordingExporter()
    exporter = MaskingSpanExporter(
        ByteLimitedSpanExporter(
            ordinary,
            max_export_bytes=128,
            upload_authority=authority,
        ),
        mask,
    )

    assert exporter.export(spans) is SpanExportResult.SUCCESS
    assert ordinary.batches == []
    assert len(authority.payloads) == 1
    assert b"MASKED" in authority.payloads[0].content
    assert b"xxxxxxxx" not in authority.payloads[0].content


def test_upload_authority_boolean_does_not_falsely_claim_delivery():
    spans = _finished_spans(count=1, payload_size=16_384)

    class IncompleteAuthority:
        available = True
        unavailable_reason = ""

        def export_overflow(self, _payload):
            return True

    diagnostics = DeliveryDiagnostics()
    exporter = ByteLimitedSpanExporter(
        RecordingExporter(),
        max_export_bytes=128,
        diagnostics=diagnostics,
        upload_authority=IncompleteAuthority(),
    )

    assert exporter.export(spans) is SpanExportResult.FAILURE
    snapshot = diagnostics.snapshot()
    assert snapshot["span_overflow_exports"] == 0
    assert snapshot["span_overflow_failures"] == 1


def test_exposes_final_export_failure_count():
    spans = _finished_spans(count=2)
    diagnostics = DeliveryDiagnostics()
    exporter = ByteLimitedSpanExporter(
        RecordingExporter(SpanExportResult.FAILURE), diagnostics=diagnostics
    )

    assert exporter.export(spans) is SpanExportResult.FAILURE
    assert diagnostics.snapshot()["span_export_failures"] == 2


def test_exporter_exception_counts_current_and_unattempted_tail():
    spans = _finished_spans(count=3, payload_size=256)
    max_bytes = max(ByteLimitedSpanExporter._encoded_upper_bound(span) for span in spans)

    class RaisingExporter(RecordingExporter):
        def export(self, spans):
            self.batches.append(list(spans))
            raise RuntimeError("transport failed")

    diagnostics = DeliveryDiagnostics()
    sink = RaisingExporter()
    exporter = ByteLimitedSpanExporter(
        sink,
        max_export_bytes=max_bytes,
        diagnostics=diagnostics,
    )

    assert exporter.export(spans) is SpanExportResult.FAILURE
    assert len(sink.batches) == 1
    assert diagnostics.snapshot()["span_export_failures"] == 3


@pytest.mark.parametrize(
    ("results", "expected_attempts", "expected_failures"),
    [
        ([SpanExportResult.FAILURE], 1, 3),
        ([SpanExportResult.SUCCESS, SpanExportResult.FAILURE], 2, 2),
        (
            [SpanExportResult.SUCCESS, SpanExportResult.SUCCESS, SpanExportResult.FAILURE],
            3,
            1,
        ),
    ],
)
def test_split_failure_counts_failed_and_all_unattempted_tail_batches(
    results, expected_attempts, expected_failures
):
    spans = _finished_spans(count=3, payload_size=256)
    max_bytes = max(ByteLimitedSpanExporter._encoded_upper_bound(span) for span in spans)
    sink = RecordingExporter(results=results)
    diagnostics = DeliveryDiagnostics()
    exporter = ByteLimitedSpanExporter(
        sink,
        max_export_bytes=max_bytes,
        diagnostics=diagnostics,
    )

    assert exporter.export(spans) is SpanExportResult.FAILURE
    assert len(sink.batches) == expected_attempts
    assert diagnostics.snapshot()["span_export_failures"] == expected_failures
