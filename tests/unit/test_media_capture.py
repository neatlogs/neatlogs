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
