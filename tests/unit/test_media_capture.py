import base64
import hashlib

from neatlogs.core.media import media_references, set_media_attributes


def test_inline_and_remote_media_preserve_type_source_digest_and_reference(
    tracer_provider, in_memory_span_exporter
):
    raw = b"full-image-bytes"
    payload = [
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{base64.b64encode(raw).decode()}"},
        },
        {
            "type": "input_file",
            "file_id": "file-provider-123",
            "mime_type": "application/pdf",
        },
    ]
    records = media_references(payload, "input")
    assert records[0]["sha256"] == hashlib.sha256(raw).hexdigest()
    assert records[0]["byte_length"] == len(raw)
    assert records[0]["source"] == "inline"
    assert records[1]["reference"] == "file-provider-123"
    assert records[1]["source"] == "provider"

    span = tracer_provider.get_tracer("neatlogs.test").start_span("media")
    set_media_attributes(span, "neatlogs.llm.input_messages.0", payload, "input")
    span.end()
    attrs = in_memory_span_exporter.get_finished_spans()[0].attributes
    assert attrs["neatlogs.llm.input_messages.0.media.0.type"] == "image"
    assert attrs["neatlogs.llm.input_messages.0.media.1.type"] == "document"


def test_large_typed_media_reports_disabled_upload_instead_of_silent_hash_only():
    raw = b"x" * 80_000
    payload = {
        "inline_data": {
            "mime_type": "application/pdf",
            "data": base64.b64encode(raw).decode(),
        }
    }

    records = media_references(payload, "document")

    assert len(records) == 1
    assert records[0]["type"] == "document"
    assert records[0]["byte_length"] == len(raw)
    assert records[0]["sha256"] == hashlib.sha256(raw).hexdigest()
    assert records[0]["state"] == "failed"
    assert "backend_upload_contract_unavailable" in records[0]["safe_preview"]


def test_binary_media_requires_explicit_type_and_arbitrary_strings_are_not_media():
    assert media_references("plain user text", "input") == []
    assert media_references({"data": b"raw without a type"}, "input") == []

    records = media_references(
        {"type": "audio", "mime_type": "audio/wav", "data": b"RIFF"},
        "input",
    )
    assert records[0]["type"] == "audio"
    assert records[0]["byte_length"] == 4
