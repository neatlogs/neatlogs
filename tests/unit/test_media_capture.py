import base64
import hashlib

from neatlogs._wrap_utils import serialize
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


def test_remote_media_strips_userinfo_query_and_fragment_from_all_capture_fields():
    secret_url = (
        "https://user:password@bucket.example/private.pdf"
        "?X-Amz-Credential=AKIA_TEST&X-Amz-Signature=secret#fragment"
    )
    payload = {
        "type": "input_file",
        "mime_type": "application/pdf",
        "url": secret_url,
    }

    record = media_references(payload, "input")[0]
    captured = serialize(payload)

    assert record["reference"] == "https://bucket.example/private.pdf"
    assert "user" not in record["reference"]
    assert "password" not in record["reference"]
    assert "X-Amz" not in captured
    assert "secret" not in captured


def test_large_inline_media_body_is_replaced_by_typed_unavailable_metadata():
    raw = b"unique-secret-media" * 6000
    encoded = base64.b64encode(raw).decode()
    payload = [
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{encoded}"},
        }
    ]

    captured = serialize(payload)

    assert encoded[:100] not in captured
    assert "unique-secret-media" not in captured
    assert "backend_upload_contract_unavailable" in captured
    assert hashlib.sha256(raw).hexdigest() in captured
    assert len(captured.encode()) < 2_000


def test_anthropic_source_media_is_sanitized_without_losing_typed_metadata():
    raw = b"private-anthropic-image" * 5000
    encoded = base64.b64encode(raw).decode()
    payload = [
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": encoded,
            },
        },
        {
            "type": "document",
            "source": {
                "type": "url",
                "url": "https://user:pass@example.test/report.pdf?token=secret#page=1",
            },
        },
    ]

    records = media_references(payload, "input")
    captured = serialize(payload)

    assert any(record["sha256"] == hashlib.sha256(raw).hexdigest() for record in records)
    assert any(record.get("reference") == "https://example.test/report.pdf" for record in records)
    assert encoded[:100] not in captured
    assert "pass" not in captured
    assert "token" not in captured


def test_bedrock_nested_bytes_are_replaced_when_upload_is_unavailable():
    raw = b"private-bedrock-image" * 6000
    payload = {"image": {"format": "png", "source": {"bytes": raw}}}

    records = media_references(payload, "input")
    captured = serialize(payload)

    assert records[0]["type"] == "image"
    assert records[0]["state"] == "failed"
    assert hashlib.sha256(raw).hexdigest() in captured
    assert "private-bedrock-image" not in captured
