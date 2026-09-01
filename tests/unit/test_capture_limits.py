from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import InMemoryLogRecordExporter, SimpleLogRecordProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExportResult
from opentelemetry.trace import StatusCode

from neatlogs._wrap_utils import serialize
from neatlogs.core.byte_limited_exporter import ByteLimitedSpanExporter
from neatlogs.core.byte_limited_log_exporter import ByteLimitedLogExporter
from neatlogs.core.capture import (
    DEFAULT_MAX_CAPTURE_ITEM_BYTES,
    DEFAULT_MAX_CAPTURE_VALUE_BYTES,
    limit_log_capture,
    limit_span_capture,
)
from neatlogs.core.delivery import DeliveryDiagnostics
from neatlogs.core.masking_exporter import MaskingLogExporter, MaskingSpanExporter
from neatlogs.core.upload_authority import OverflowExportReceipt


class _SpanSink:
    def __init__(self):
        self.spans = []

    def export(self, spans):
        self.spans.extend(spans)
        return SpanExportResult.SUCCESS

    def force_flush(self, timeout_millis=30000):
        return True

    def shutdown(self):
        return None


def test_serialize_has_a_byte_bound_and_self_describing_truncation():
    value = serialize({"payload": "é" * 80_000})

    assert len(value.encode("utf-8")) <= DEFAULT_MAX_CAPTURE_VALUE_BYTES
    assert "...[neatlogs-truncated" in value
    assert "original_bytes=" in value
    assert "sha256=" in value
    assert "overflow=backend_upload_contract_unavailable" in value


def test_post_mask_span_capture_is_bounded_and_reported():
    sink = _SpanSink()
    diagnostics = DeliveryDiagnostics()
    exporter = ByteLimitedSpanExporter(sink, diagnostics=diagnostics)
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    span = provider.get_tracer("capture-test").start_span("large")
    span.set_attribute("output.value", "x" * 150_000)
    span.end()
    provider.shutdown()

    exported = sink.spans[0]
    assert len(exported.attributes["output.value"].encode()) <= DEFAULT_MAX_CAPTURE_VALUE_BYTES
    assert "...[neatlogs-truncated" in exported.attributes["output.value"]
    assert exported.attributes["neatlogs.capture.truncated"] is True
    assert exported.attributes["neatlogs.capture.overflow.state"] == "disabled"
    assert diagnostics.snapshot()["span_capture_truncations"] == 1


def test_post_mask_log_capture_is_bounded_and_reported():
    diagnostics = DeliveryDiagnostics()
    inner = InMemoryLogRecordExporter()
    provider = LoggerProvider()
    provider.add_log_record_processor(
        SimpleLogRecordProcessor(
            MaskingLogExporter(
                ByteLimitedLogExporter(inner, diagnostics=diagnostics),
                None,
                diagnostics=diagnostics,
            )
        )
    )

    provider.get_logger("capture-test").emit(body="x" * 150_000)
    provider.shutdown()

    exported = inner.get_finished_logs()[0].log_record
    assert len(exported.body.encode()) <= DEFAULT_MAX_CAPTURE_VALUE_BYTES
    assert "...[neatlogs-truncated" in exported.body
    assert exported.attributes["neatlogs.capture.truncated"] is True
    assert diagnostics.snapshot()["log_capture_truncations"] == 1


def test_masking_precedes_injected_overflow_authority():
    class Authority:
        available = True
        unavailable_reason = ""

        def __init__(self):
            self.payload = None

        def export_overflow(self, payload):
            self.payload = payload
            return OverflowExportReceipt(
                upload_id="upload-1",
                project_id="project-1",
                state="ready",
                reference_exported=True,
            )

    def mask(snapshot):
        snapshot["attributes"]["secret"] = "masked"
        return snapshot

    authority = Authority()
    sink = _SpanSink()
    provider = TracerProvider()
    provider.add_span_processor(
        SimpleSpanProcessor(
            MaskingSpanExporter(
                ByteLimitedSpanExporter(
                    sink,
                    max_export_bytes=128,
                    upload_authority=authority,
                ),
                mask,
            )
        )
    )

    span = provider.get_tracer("capture-test").start_span("oversized")
    span.set_attribute("secret", "must-not-cross-authority")
    span.set_attribute("padding", "x" * 2048)
    span.end()
    provider.shutdown()

    assert authority.payload is not None
    assert b"must-not-cross-authority" not in authority.payload.content
    assert b"masked" in authority.payload.content
    assert sink.spans == []


def test_span_capture_budget_covers_name_status_resource_and_mapping_keys():
    sink = _SpanSink()
    provider = TracerProvider(
        resource=Resource(
            {
                "resource-key-" + "k" * 200: "resource-value-" + "r" * 200,
            }
        )
    )
    provider.add_span_processor(SimpleSpanProcessor(sink))
    span = provider.get_tracer("capture-test").start_span("name-" + "n" * 1_200_000)
    span.set_status(StatusCode.ERROR, "status-" + "s" * 200)
    span.add_event("event-" + "e" * 200, {"event-key-" + "k" * 200: "value"})
    span.end()
    provider.shutdown()

    bounded, truncations = limit_span_capture(
        sink.spans[0],
        max_item_bytes=2_000,
        max_value_bytes=80,
    )

    assert truncations >= 5
    assert len(bounded.name.encode()) <= 80
    assert len(bounded.status.description.encode()) <= 80
    assert all(len(key.encode()) <= 80 for key in bounded.resource.attributes)
    assert all(len(key.encode()) <= 80 for key in bounded.events[0].attributes)
    assert len(bounded.events[0].name.encode()) <= 80
    assert len(bounded.name.encode()) < DEFAULT_MAX_CAPTURE_ITEM_BYTES


def test_log_capture_bounds_bytes_resource_fields_and_collision_safe_mapping_keys():
    inner = InMemoryLogRecordExporter()
    provider = LoggerProvider(
        resource=Resource({"resource-key-" + "k" * 200: "resource-value-" + "r" * 200})
    )
    provider.add_log_record_processor(SimpleLogRecordProcessor(inner))
    provider.get_logger("capture-test").emit(
        body={
            "shared-prefix-" + "a" * 200: b"a" * 200,
            "shared-prefix-" + "b" * 200: b"b" * 200,
        },
        severity_text="severity-" + "s" * 200,
        event_name="event-" + "e" * 200,
    )
    provider.shutdown()

    bounded, truncations = limit_log_capture(
        inner.get_finished_logs()[0],
        max_item_bytes=2_000,
        max_value_bytes=80,
    )
    record = bounded.log_record
    resource = getattr(bounded, "resource", None) or getattr(record, "resource", None)

    assert truncations >= 7
    assert len(record.body) == 2
    assert len(set(record.body)) == 2
    assert all(len(key.encode()) <= 80 for key in record.body)
    assert all(isinstance(value, bytes) and len(value) <= 80 for value in record.body.values())
    assert len(record.severity_text.encode()) <= 80
    assert len(record.event_name.encode()) <= 80
    assert all(len(key.encode()) <= 80 for key in resource.attributes)
