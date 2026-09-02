import base64
import hashlib

import pytest

from neatlogs.azure_openai import _finalize_responses_response as finalize_azure_response
from neatlogs.azure_openai import _finalize_responses_stream as finalize_azure_stream
from neatlogs.azure_openai import _set_generated_image_media as set_azure_image_media
from neatlogs.azure_openai import _set_speech_response_media as set_azure_speech_media
from neatlogs.openai import _finalize_responses_response as finalize_openai_response
from neatlogs.openai import _finalize_responses_stream as finalize_openai_stream
from neatlogs.openai import _set_generated_image_media as set_openai_image_media
from neatlogs.openai import _set_speech_response_media as set_openai_speech_media
from neatlogs.openrouter import _finalize_responses as finalize_openrouter_response
from neatlogs.openrouter import _finalize_responses_stream as finalize_openrouter_stream


class ProviderModel:
    def __init__(self, **values):
        self._values = values
        for key, value in values.items():
            setattr(self, key, value)

    def model_dump(self, **_kwargs):
        return dict(self._values)


@pytest.mark.parametrize("set_media", [set_openai_image_media, set_azure_image_media])
def test_openai_image_resources_capture_url_only_outputs(
    set_media, tracer_provider, in_memory_span_exporter
):
    span = tracer_provider.get_tracer("neatlogs.provider").start_span("image")
    set_media(span, [ProviderModel(url="https://cdn.example/image.png?signature=secret")])
    span.end()

    attributes = in_memory_span_exporter.get_finished_spans()[0].attributes
    assert attributes["neatlogs.llm.output_messages.0.media.0.type"] == "image"
    assert (
        attributes["neatlogs.llm.output_messages.0.media.0.reference"]
        == "https://cdn.example/image.png"
    )


@pytest.mark.parametrize("set_media", [set_openai_speech_media, set_azure_speech_media])
def test_openai_speech_resources_capture_buffered_audio_outputs(
    set_media, tracer_provider, in_memory_span_exporter
):
    raw = b"ID3" + (b"audio" * 20)
    response = ProviderModel(content=raw, headers={"content-type": "audio/mpeg"})
    span = tracer_provider.get_tracer("neatlogs.provider").start_span("speech")
    set_media(span, response)
    span.end()

    attributes = in_memory_span_exporter.get_finished_spans()[0].attributes
    prefix = "neatlogs.llm.output_messages.0.media.0"
    assert attributes[f"{prefix}.type"] == "audio"
    assert attributes[f"{prefix}.sha256"] == hashlib.sha256(raw).hexdigest()


@pytest.mark.parametrize(
    "finalize,response",
    [
        (
            finalize_openai_response,
            lambda item: ProviderModel(output=[item], output_text=None, model=None, usage=None),
        ),
        (
            finalize_azure_response,
            lambda item: ProviderModel(output=[item], output_text=None, model=None, usage=None),
        ),
        (
            finalize_openrouter_response,
            lambda item: {"output": [item]},
        ),
    ],
)
def test_responses_sync_and_async_finalizers_capture_image_generation_results(
    finalize, response, tracer_provider, in_memory_span_exporter
):
    raw = b"\x89PNG\r\n\x1a\n" + (b"generated" * 20)
    item = ProviderModel(
        type="image_generation_call",
        result=base64.b64encode(raw).decode(),
    )
    if finalize is finalize_openrouter_response:
        item = item.model_dump()
    span = tracer_provider.get_tracer("neatlogs.provider").start_span("responses")
    finalize(span, response(item), 1.0)

    attributes = in_memory_span_exporter.get_finished_spans()[0].attributes
    assert (
        attributes["neatlogs.llm.output_messages.0.media.0.sha256"]
        == hashlib.sha256(raw).hexdigest()
    )


@pytest.mark.parametrize(
    "finalize,event",
    [
        (finalize_openai_stream, lambda values: ProviderModel(**values)),
        (finalize_azure_stream, lambda values: ProviderModel(**values)),
        (finalize_openrouter_stream, lambda values: values),
    ],
)
def test_responses_stream_finalizers_capture_image_generation_results(
    finalize, event, tracer_provider, in_memory_span_exporter
):
    raw = b"\x89PNG\r\n\x1a\n" + (b"streamed" * 20)
    chunk = event(
        {
            "type": "response.image_generation_call.completed",
            "result": base64.b64encode(raw).decode(),
        }
    )
    span = tracer_provider.get_tracer("neatlogs.provider").start_span("responses-stream")
    finalize(span, [chunk], 1.0, 0.5)

    attributes = in_memory_span_exporter.get_finished_spans()[0].attributes
    assert (
        attributes["neatlogs.llm.output_messages.0.media.0.sha256"]
        == hashlib.sha256(raw).hexdigest()
    )
