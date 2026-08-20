from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExportResult

from neatlogs.core.byte_limited_exporter import ByteLimitedSpanExporter
from neatlogs.core.delivery import DeliveryDiagnostics


class RecordingExporter:
    def __init__(self, result=SpanExportResult.SUCCESS):
        self.batches = []
        self.result = result

    def export(self, spans):
        self.batches.append(list(spans))
        return self.result

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


def test_forwards_one_oversized_span_without_truncation_or_drop():
    spans = _finished_spans(count=1, payload_size=16_384)
    sink = RecordingExporter()
    exporter = ByteLimitedSpanExporter(sink, max_export_bytes=128)

    assert exporter.export(spans) is SpanExportResult.SUCCESS
    assert sink.batches == [spans]


def test_exposes_final_export_failure_count():
    spans = _finished_spans(count=2)
    diagnostics = DeliveryDiagnostics()
    exporter = ByteLimitedSpanExporter(
        RecordingExporter(SpanExportResult.FAILURE), diagnostics=diagnostics
    )

    assert exporter.export(spans) is SpanExportResult.FAILURE
    assert diagnostics.snapshot()["span_export_failures"] == 2
