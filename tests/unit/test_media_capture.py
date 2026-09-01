import base64
import hashlib
import json

from neatlogs._wrap_utils import serialize
from neatlogs.core.media import (
    PendingMediaStore,
    media_references,
    promote_message_media_attributes,
    sanitize_media_payload,
    set_default_media_store,
    set_media_attributes,
)


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


def test_backend_mime_aliases_and_bedrock_format_are_canonicalized():
    raw = b"x" * 100_001
    store = PendingMediaStore(max_bytes=25 * 1024 * 1024)
    set_default_media_store(store)
    try:
        audio = media_references(
            {
                "type": "input_audio",
                "input_audio": {"format": "mp3", "data": base64.b64encode(raw).decode()},
            },
            "input",
        )[0]
        image = media_references({"image": {"format": "png", "source": {"bytes": raw}}}, "input")[0]
    finally:
        set_default_media_store(None)

    assert audio["mime_type"] == "audio/mpeg"
    assert image["mime_type"] == "image/png"


def test_media_signature_corrects_generic_or_incorrect_declared_mime():
    png = b"\x89PNG\r\n\x1a\n" + (b"x" * 64)

    generic = media_references(
        {"type": "image", "mime_type": "application/octet-stream", "data": png},
        "input",
    )[0]
    incorrect = media_references(
        {
            "type": "image_url",
            "mime_type": "image/jpeg",
            "image_url": {"url": f"data:image/jpeg;base64,{base64.b64encode(png).decode()}"},
        },
        "input",
    )[0]

    assert generic["mime_type"] == "image/png"
    assert incorrect["mime_type"] == "image/png"


def test_duplicate_capture_and_semantic_discovery_share_store_capacity():
    raw = b"x" * (128 * 1024)
    value = {"type": "image", "mime_type": "image/png", "data": raw}
    store = PendingMediaStore(max_bytes=160 * 1024)
    set_default_media_store(store)
    try:
        serialize(value)
        record = media_references(value, "input")[0]
        snapshot = store.snapshot()
    finally:
        set_default_media_store(None)

    assert record["state"] == "pending-upload"
    assert "upload_token" in record
    assert snapshot == {"items": 1, "bytes": len(raw)}


def test_duplicate_media_within_one_value_does_not_leak_store_references():
    raw = b"x" * (128 * 1024)
    value = {"type": "image", "mime_type": "image/png", "data": raw}
    store = PendingMediaStore(max_bytes=160 * 1024)
    set_default_media_store(store)
    try:
        records = media_references([value, value], "input")
        store.release(records[0]["upload_token"])
        snapshot = store.snapshot()
    finally:
        set_default_media_store(None)

    assert len(records) == 1
    assert snapshot == {"items": 0, "bytes": 0}


def test_serialized_provider_media_is_promoted_to_canonical_message_attributes():
    raw = b"x" * 100_001
    value = {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{base64.b64encode(raw).decode()}"},
    }
    store = PendingMediaStore(max_bytes=25 * 1024 * 1024)
    set_default_media_store(store)
    try:
        attributes = promote_message_media_attributes(
            {"neatlogs.llm.input_messages.2.content": serialize(value)}
        )
    finally:
        set_default_media_store(None)

    prefix = "neatlogs.llm.input_messages.2.media.0"
    assert attributes[f"{prefix}.mime_type"] == "image/png"
    assert attributes[f"{prefix}.state"] == "pending-upload"
    assert attributes[f"{prefix}.purpose"] == "input"


def test_malformed_and_case_varied_data_uris_fail_closed():
    invalid = "data:image/png;base64," + ("not-base64-SECRET" * 6_000)
    mixed_case = "DATA:image/png;base64," + base64.b64encode(b"private" * 20_000).decode()

    invalid_capture = serialize({"type": "image_url", "image_url": {"url": invalid}})
    mixed_capture = serialize({"type": "image_url", "image_url": {"url": mixed_case}})

    assert "not-base64-SECRET" not in invalid_capture
    assert "invalid_media_encoding" in invalid_capture
    assert base64.b64encode(b"private" * 20_000).decode()[:100] not in mixed_capture
    assert "neatlogs_media" in mixed_capture


def test_media_walkers_are_bounded_and_provider_serializers_cannot_escape():
    cyclic = {"type": "image"}
    cyclic["source"] = cyclic

    class BrokenModel:
        def model_dump(self, **_kwargs):
            raise RuntimeError("provider serializer failed")

    assert media_references(cyclic, "input") == []
    assert media_references(BrokenModel(), "input") == []
    captured = serialize(cyclic)
    assert "traversal_cycle" in captured
    assert "provider serializer failed" not in serialize(BrokenModel())

    bounded = sanitize_media_payload([{"value": index} for index in range(20_000)])
    assert len(bounded) <= 10_001
    assert "traversal_limit" in repr(bounded[-1])


def test_image_generation_base64_is_discovered_as_typed_output():
    raw = b"generated-image" * 8_000
    records = media_references(
        [{"b64_json": base64.b64encode(raw).decode()}],
        "output",
    )

    assert len(records) == 1
    assert records[0]["type"] == "image"
    assert records[0]["sha256"] == hashlib.sha256(raw).hexdigest()
    assert "b64_json" not in json.dumps(records)
