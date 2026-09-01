import base64
import json

from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import (
    InMemoryLogRecordExporter,
    LogRecordExportResult,
    SimpleLogRecordProcessor,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExportResult
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from neatlogs._wrap_utils import serialize
from neatlogs.core.delivery import DeliveryDiagnostics
from neatlogs.core.masking_exporter import MaskingLogExporter, MaskingSpanExporter
from neatlogs.core.media import PendingMediaStore, set_default_media_store, set_media_attributes
from neatlogs.core.media_exporter import TypedMediaLogExporter, TypedMediaSpanExporter
from neatlogs.core.upload_authority import (
    MediaExportReceipt,
    UploadError,
    UploadReference,
)

UPLOAD_ID = "123e4567-e89b-12d3-a456-426614174000"


class Authority:
    available = True
    unavailable_reason = ""

    def __init__(self, *, fail=False):
        self.payloads = []
        self.fail = fail

    def export_media(self, payload):
        self.payloads.append(payload)
        if self.fail:
            raise UploadError("complete", "MEDIA_SIGNATURE_MISMATCH")
        reference = UploadReference(
            id=UPLOAD_ID,
            purpose="typed_media",
            sha256=payload.sha256,
            byte_length=payload.byte_length,
            mime_type=payload.mime_type,
            content_encoding="identity",
            state="ready",
        )
        return MediaExportReceipt(UPLOAD_ID, "ready", reference)

    def export_overflow(self, payload):  # pragma: no cover - not this layer's job
        raise AssertionError(payload)


def _large_image():
    raw = b"\x89PNG\r\n\x1a\n" + b"private-media-content" * 5_000
    value = {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{base64.b64encode(raw).decode()}"},
    }
    return raw, value


def test_span_media_is_staged_then_resolved_after_mask_to_canonical_reference():
    raw, value = _large_image()
    store = PendingMediaStore(max_bytes=25 * 1024 * 1024)
    authority = Authority()
    diagnostics = DeliveryDiagnostics()
    sink = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(
        SimpleSpanProcessor(
            MaskingSpanExporter(
                TypedMediaSpanExporter(sink, authority, store, diagnostics),
                lambda snapshot: snapshot,
            )
        )
    )
    set_default_media_store(store)
    try:
        span = provider.get_tracer("media-test").start_span("media")
        span.set_attribute("input.value", serialize(value))
        set_media_attributes(span, "neatlogs.llm.input_messages.0", value, "input")
        span.add_event("media", {"payload": serialize(value)})
        span.end()
        provider.shutdown()
    finally:
        set_default_media_store(None)

    # The captured body and flattened semantic media attributes intentionally
    # retain different purposes; a production authority deduplicates their
    # identical bytes with its deterministic idempotency key.
    assert len(authority.payloads) == 2
    assert all(payload.content == raw for payload in authority.payloads)
    assert store.snapshot() == {"items": 0, "bytes": 0}
    exported = sink.get_finished_spans()[0]
    rendered = repr(exported.attributes) + repr(exported.events[0].attributes)
    assert "private-media-content" not in rendered
    assert "nl_pending_media_" not in rendered
    assert "signed" not in rendered
    assert UPLOAD_ID in rendered
    assert '"source":"uploaded"' in exported.attributes["input.value"]
    prefix = "neatlogs.llm.input_messages.0.media.0"
    assert exported.attributes[f"{prefix}.source"] == "uploaded"
    assert exported.attributes[f"{prefix}.state"] == "available"
    assert diagnostics.snapshot()["span_media_uploads"] == 2


def test_mask_can_remove_media_before_any_upload_occurs():
    _, value = _large_image()
    store = PendingMediaStore(max_bytes=25 * 1024 * 1024)
    authority = Authority()
    sink = InMemorySpanExporter()
    provider = TracerProvider()

    def remove_all_attributes(snapshot):
        snapshot["attributes"] = {}
        return snapshot

    provider.add_span_processor(
        SimpleSpanProcessor(
            MaskingSpanExporter(
                TypedMediaSpanExporter(sink, authority, store), remove_all_attributes
            )
        )
    )
    set_default_media_store(store)
    try:
        span = provider.get_tracer("media-test").start_span("masked")
        span.set_attribute("input.value", serialize(value))
        span.end()
        provider.shutdown()
    finally:
        set_default_media_store(None)

    assert authority.payloads == []
    assert sink.get_finished_spans()[0].attributes == {}


def test_masked_upload_token_fails_closed_without_uploading_or_exporting_token():
    _, value = _large_image()
    store = PendingMediaStore(max_bytes=25 * 1024 * 1024)
    authority = Authority()
    diagnostics = DeliveryDiagnostics()
    sink = InMemorySpanExporter()
    provider = TracerProvider()

    def redact_token(snapshot):
        parsed = json.loads(snapshot["attributes"]["input.value"])
        parsed["image_url"]["url"]["neatlogs_media"]["upload_token"] = "***"
        snapshot["attributes"]["input.value"] = json.dumps(parsed)
        return snapshot

    provider.add_span_processor(
        SimpleSpanProcessor(
            MaskingSpanExporter(
                TypedMediaSpanExporter(sink, authority, store, diagnostics), redact_token
            )
        )
    )
    set_default_media_store(store)
    try:
        span = provider.get_tracer("media-test").start_span("masked-token")
        span.set_attribute("input.value", serialize(value))
        span.end()
        provider.shutdown()
    finally:
        set_default_media_store(None)

    assert authority.payloads == []
    rendered = sink.get_finished_spans()[0].attributes["input.value"]
    assert '"state":"failed"' in rendered
    assert "upload_token" not in json.loads(rendered)["image_url"]["url"]["neatlogs_media"]
    assert "***" not in rendered
    assert diagnostics.snapshot()["span_media_upload_failures"] == 1


def test_media_failure_exports_secret_free_failed_reference_and_reports_failure():
    _, value = _large_image()
    store = PendingMediaStore(max_bytes=25 * 1024 * 1024)
    authority = Authority(fail=True)
    diagnostics = DeliveryDiagnostics()
    sink = InMemorySpanExporter()
    set_default_media_store(store)
    try:
        captured = serialize(value)
        provider = TracerProvider()
        capture_sink = InMemorySpanExporter()
        provider.add_span_processor(SimpleSpanProcessor(capture_sink))
        span = provider.get_tracer("media-test").start_span("failed")
        span.set_attribute("input.value", captured)
        span.end()
        item = capture_sink.get_finished_spans()[0]
        result = TypedMediaSpanExporter(sink, authority, store, diagnostics).export((item,))
    finally:
        set_default_media_store(None)

    assert result is SpanExportResult.FAILURE
    rendered = sink.get_finished_spans()[0].attributes["input.value"]
    assert "nl_pending_media_" not in rendered
    assert "private-media-content" not in rendered
    assert "MEDIA_SIGNATURE_MISMATCH" in rendered
    snapshot = diagnostics.snapshot()
    assert snapshot["span_media_upload_failures"] == 1
    assert snapshot["upload_last_failure_reason"] == "complete:MEDIA_SIGNATURE_MISMATCH"


def test_log_body_media_is_replaced_without_a_second_ordinary_payload():
    _, value = _large_image()
    store = PendingMediaStore(max_bytes=25 * 1024 * 1024)
    authority = Authority()
    diagnostics = DeliveryDiagnostics()
    sink = InMemoryLogRecordExporter()
    provider = LoggerProvider()
    provider.add_log_record_processor(
        SimpleLogRecordProcessor(
            MaskingLogExporter(
                TypedMediaLogExporter(sink, authority, store, diagnostics), mask=None
            )
        )
    )
    set_default_media_store(store)
    try:
        provider.get_logger("media-test").emit(body=serialize(value))
        provider.shutdown()
    finally:
        set_default_media_store(None)

    assert len(authority.payloads) == 1
    record = sink.get_finished_logs()[0].log_record
    parsed = json.loads(record.body)
    reference = parsed["image_url"]["url"]["neatlogs_media"]
    assert reference["id"] == UPLOAD_ID
    assert reference["source"] == "uploaded"
    assert reference["state"] == "available"
    assert "upload_token" not in reference
    assert diagnostics.snapshot()["log_media_uploads"] == 1


def test_repeated_token_in_one_batch_uploads_once_and_resolves_every_span():
    _, value = _large_image()
    store = PendingMediaStore(max_bytes=25 * 1024 * 1024)
    authority = Authority()
    capture = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(capture))
    set_default_media_store(store)
    try:
        captured = serialize(value)
        for index in range(2):
            span = provider.get_tracer("media-test").start_span(f"media-{index}")
            span.set_attribute("input.value", captured)
            span.end()
        sink = InMemorySpanExporter()
        result = TypedMediaSpanExporter(sink, authority, store).export(capture.get_finished_spans())
    finally:
        set_default_media_store(None)

    assert result is SpanExportResult.SUCCESS
    assert len(authority.payloads) == 1
    assert len(sink.get_finished_spans()) == 2
    for span in sink.get_finished_spans():
        assert UPLOAD_ID in span.attributes["input.value"]
        assert "nl_pending_media_" not in span.attributes["input.value"]
