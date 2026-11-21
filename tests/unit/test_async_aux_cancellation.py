"""Regression: async responses/embeddings/images wrappers must end their span
when the awaited call is cancelled.

`asyncio.CancelledError` inherits BaseException (not Exception), so the
`except Exception` guards in the Azure OpenAI async responses/embeddings
wrappers and the shared `_patch_method` async wrapper (images/audio on both
OpenAI and Azure OpenAI) let the span leak open until shutdown.
"""

import asyncio
import types as _t

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

import neatlogs
from neatlogs._wrap_utils import set_neatlogs_provider


def _init():
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    set_neatlogs_provider(provider)
    neatlogs.init(
        api_key="test-key",
        disable_export=True,
        instrumentations=[],
        tracer_provider=provider,
        register_shutdown_handlers=False,
    )
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return exporter


async def _cancel_and_get_span(exporter, obj, call, span_name):
    task = asyncio.create_task(call())
    await asyncio.sleep(0.02)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0.02)
    spans = [s for s in exporter.get_finished_spans() if s.name == span_name]
    assert spans, f"{span_name} span leaked open on cancellation"
    return spans[0]


def _slow_create():
    async def create(self, *a, **k):
        await asyncio.sleep(10)

    return _t.MethodType(create, _t.SimpleNamespace())


@pytest.mark.asyncio
async def test_azure_async_responses_cancel_ends_span():
    from neatlogs.azure_openai import _patch_async_responses

    exporter = _init()
    responses = _t.SimpleNamespace()
    async def create(*a, **k):
        await asyncio.sleep(10)
    responses.create = create
    _patch_async_responses(responses)
    span = await _cancel_and_get_span(
        exporter, responses, lambda: responses.create(model="gpt-4o", input="hi"),
        "azure_openai.responses.create",
    )
    assert span.status.status_code == StatusCode.UNSET
    assert span.attributes.get("neatlogs.stream.cancelled") is True


@pytest.mark.asyncio
async def test_azure_async_embeddings_cancel_ends_span():
    from neatlogs.azure_openai import _patch_embeddings

    exporter = _init()
    embeddings = _t.SimpleNamespace()
    async def create(*a, **k):
        await asyncio.sleep(10)
    embeddings.create = create
    _patch_embeddings(embeddings, sync=False)
    span = await _cancel_and_get_span(
        exporter, embeddings,
        lambda: embeddings.create(model="text-embedding-3-small", input="hi"),
        "azure_openai.embeddings.create",
    )
    assert span.status.status_code == StatusCode.UNSET
    assert span.attributes.get("neatlogs.stream.cancelled") is True


@pytest.mark.asyncio
async def test_openai_async_images_cancel_ends_span():
    from neatlogs.openai import _patch_images

    exporter = _init()
    images = _t.SimpleNamespace()
    async def generate(*a, **k):
        await asyncio.sleep(10)
    images.generate = generate
    _patch_images(images, sync=False)
    span = await _cancel_and_get_span(
        exporter, images, lambda: images.generate(model="gpt-image-1", prompt="hi"),
        "openai.images.generate",
    )
    assert span.status.status_code == StatusCode.UNSET
    assert span.attributes.get("neatlogs.stream.cancelled") is True
