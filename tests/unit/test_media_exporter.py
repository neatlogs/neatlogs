import base64
import json

import pytest
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

    # The body, canonical attributes, and event share one staged object and one
    # authority call while preserving their individual telemetry purposes.
    assert len(authority.payloads) == 1
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
    assert diagnostics.snapshot()["span_media_uploads"] == 1


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
        retained_after_export = store.snapshot()
    finally:
        set_default_media_store(None)

    assert authority.payloads == []
    assert sink.get_finished_spans()[0].attributes == {}
    assert retained_after_export == {"items": 0, "bytes": 0}


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
        retained_after_export = store.snapshot()
    finally:
        set_default_media_store(None)

    assert authority.payloads == []
    rendered = sink.get_finished_spans()[0].attributes["input.value"]
    assert '"state":"failed"' in rendered
    assert "upload_token" not in json.loads(rendered)["image_url"]["url"]["neatlogs_media"]
    assert "***" not in rendered
    assert retained_after_export == {"items": 0, "bytes": 0}
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


def test_repeated_token_across_batches_retains_bytes_until_last_reference():
    _, value = _large_image()
    store = PendingMediaStore(max_bytes=25 * 1024 * 1024)
    authority = Authority()
    capture = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(capture))
    set_default_media_store(store)
    try:
        for index in range(2):
            span = provider.get_tracer("media-test").start_span(f"media-{index}")
            span.set_attribute("input.value", serialize(value))
            span.end()
        exporter = TypedMediaSpanExporter(InMemorySpanExporter(), authority, store)
        first, second = capture.get_finished_spans()
        assert exporter.export((first,)) is SpanExportResult.SUCCESS
        assert store.snapshot()["items"] == 1
        assert exporter.export((second,)) is SpanExportResult.SUCCESS
        assert store.snapshot() == {"items": 0, "bytes": 0}
    finally:
        set_default_media_store(None)

    assert len(authority.payloads) == 2


@pytest.mark.parametrize("provider", ["cohere", "groq", "mistral", "litellm", "together"])
def test_generic_provider_content_is_sanitized_and_promoted_after_mask(provider):
    original = b"unmasked-private-image" * 5_000
    masked = b"masked-private-image" * 5_000
    original_value = {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{base64.b64encode(original).decode()}"},
    }
    masked_value = {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{base64.b64encode(masked).decode()}"},
    }
    store = PendingMediaStore(max_bytes=25 * 1024 * 1024)
    authority = Authority()
    sink = InMemorySpanExporter()
    capture = InMemorySpanExporter()
    provider_instance = TracerProvider()
    provider_instance.add_span_processor(SimpleSpanProcessor(capture))
    span = provider_instance.get_tracer("generic-media").start_span("provider-media")
    span.set_attribute("gen_ai.system", provider)
    span.set_attribute(
        "neatlogs.llm.input_messages.0.content",
        json.dumps(original_value),
    )
    span.end()

    def mask(snapshot):
        snapshot["attributes"]["neatlogs.llm.input_messages.0.content"] = json.dumps(masked_value)
        return snapshot

    exporter = MaskingSpanExporter(
        TypedMediaSpanExporter(sink, authority, store),
        mask,
        media_store=store,
    )
    assert exporter.export(capture.get_finished_spans()) is SpanExportResult.SUCCESS

    assert len(authority.payloads) == 1
    assert authority.payloads[0].content == masked
    rendered = repr(sink.get_finished_spans()[0].attributes)
    assert base64.b64encode(original).decode()[:100] not in rendered
    assert base64.b64encode(masked).decode()[:100] not in rendered
    assert UPLOAD_ID in rendered


def test_generic_provider_media_fails_closed_when_uploads_are_disabled():
    raw = b"disabled-upload-private-image" * 5_000
    value = {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{base64.b64encode(raw).decode()}"},
    }
    capture = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(capture))
    span = provider.get_tracer("generic-media").start_span("disabled-media")
    span.set_attribute("neatlogs.llm.input_messages.0.content", json.dumps(value))
    span.end()
    sink = InMemorySpanExporter()

    result = TypedMediaSpanExporter(sink, authority=None, store=None).export(
        capture.get_finished_spans()
    )

    assert result is SpanExportResult.SUCCESS
    rendered = repr(sink.get_finished_spans()[0].attributes)
    assert base64.b64encode(raw).decode()[:100] not in rendered
    assert "backend_upload_contract_unavailable" in rendered
    assert "nl_pending_media_" not in rendered


def test_flattened_image_generation_base64_fails_closed():
    encoded = base64.b64encode(b"generated-private-image" * 5_000).decode()
    capture = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(capture))
    span = provider.get_tracer("generated-media").start_span("generated-image")
    span.set_attribute("gen_ai.response.images.0.b64_json", encoded)
    span.end()
    sink = InMemorySpanExporter()

    TypedMediaSpanExporter(sink, authority=None, store=None).export(capture.get_finished_spans())

    rendered = repr(sink.get_finished_spans()[0].attributes)
    assert encoded[:100] not in rendered
    assert "backend_upload_contract_unavailable" in rendered
