from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExportResult

from neatlogs.core.delivery import DeliveryDiagnostics, ObservableBatchSpanProcessor


class Exporter:
    def __init__(self):
        self.spans = []

    def export(self, spans):
        self.spans.extend(spans)
        return SpanExportResult.SUCCESS

    def force_flush(self, timeout_millis=30000):
        return True

    def shutdown(self):
        return None


def _finished_span():
    sink = Exporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(sink))
    provider.get_tracer("delivery-test").start_span("span").end()
    return sink.spans[0]


def test_exposes_queue_saturation_before_otel_drops():
    diagnostics = DeliveryDiagnostics()
    processor = ObservableBatchSpanProcessor(
        Exporter(),
        max_queue_size=1,
        max_export_batch_size=1,
        schedule_delay_millis=60_000,
        diagnostics=diagnostics,
    )
    span = _finished_span()
    processor._batch_processor._queue.appendleft(span)

    processor.on_end(span)

    assert diagnostics.snapshot()["span_queue_drops"] == 1
    processor.shutdown()
