"""Regression: async provider wrappers must end their span when the awaited
call is cancelled. asyncio.CancelledError inherits BaseException (not
Exception), so an `except Exception` guard let the span leak open until
shutdown. Generalizes the cancellation fix from the Pydantic AI wrapper to the
OpenAI, Azure OpenAI, Google GenAI, Vertex AI, and Anthropic wrappers.
"""

import asyncio

import pytest

import neatlogs
from neatlogs._wrap_utils import set_neatlogs_provider


def _init(tracer_provider):
    set_neatlogs_provider(tracer_provider)
    neatlogs.init(
        api_key="test-key",
        instrumentations=[],
        tracer_provider=tracer_provider,
        register_shutdown_handlers=False,
    )


async def _cancel_and_get_span(exporter, patch_fn, obj, call):
    patch_fn(obj)
    task = asyncio.create_task(call())
    await asyncio.sleep(0.02)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0.02)
    return exporter.get_finished_spans()


@pytest.mark.asyncio
async def test_openai_async_ends_span_on_cancel(tracer_provider, in_memory_span_exporter):
    _init(tracer_provider)
    from neatlogs.openai import _patch_async_completions

    class C:
        async def create(self, *a, **k):
            await asyncio.sleep(10)

    obj = C()
    spans = await _cancel_and_get_span(
        in_memory_span_exporter,
        _patch_async_completions,
        obj,
        lambda: obj.create(model="gpt-4o", messages=[{"role": "user", "content": "hi"}]),
    )
    assert any(s.name == "openai.chat.completions.create" for s in spans)


@pytest.mark.asyncio
async def test_azure_openai_async_ends_span_on_cancel(tracer_provider, in_memory_span_exporter):
    _init(tracer_provider)
    from neatlogs.azure_openai import _patch_async_completions

    class C:
        async def create(self, *a, **k):
            await asyncio.sleep(10)

    obj = C()
    spans = await _cancel_and_get_span(
        in_memory_span_exporter,
        _patch_async_completions,
        obj,
        lambda: obj.create(model="gpt-4o", messages=[{"role": "user", "content": "hi"}]),
    )
    assert any(s.name == "azure_openai.chat.completions.create" for s in spans)


@pytest.mark.asyncio
async def test_google_genai_async_ends_span_on_cancel(tracer_provider, in_memory_span_exporter):
    _init(tracer_provider)
    from neatlogs.google_genai import _patch_async_models

    class M:
        async def generate_content(self, *a, **k):
            await asyncio.sleep(10)

    obj = M()
    spans = await _cancel_and_get_span(
        in_memory_span_exporter,
        _patch_async_models,
        obj,
        lambda: obj.generate_content(model="gemini-1.5", contents="hi"),
    )
    assert any(s.name == "google_genai.models.generate_content" for s in spans)


@pytest.mark.asyncio
async def test_vertex_ai_async_ends_span_on_cancel(tracer_provider, in_memory_span_exporter):
    _init(tracer_provider)
    from neatlogs.vertex_ai import _patch_async_models

    class M:
        async def generate_content(self, *a, **k):
            await asyncio.sleep(10)

    obj = M()
    spans = await _cancel_and_get_span(
        in_memory_span_exporter,
        _patch_async_models,
        obj,
        lambda: obj.generate_content(model="gemini-1.5", contents="hi"),
    )
    assert any(s.name == "vertex_ai.models.generate_content" for s in spans)


@pytest.mark.asyncio
async def test_anthropic_async_ends_span_on_cancel(tracer_provider, in_memory_span_exporter):
    pytest.importorskip("anthropic")
    _init(tracer_provider)
    from neatlogs.anthropic import _patch_async_messages

    class M:
        async def create(self, *a, **k):
            await asyncio.sleep(10)

    obj = M()
    spans = await _cancel_and_get_span(
        in_memory_span_exporter,
        _patch_async_messages,
        obj,
        lambda: obj.create(model="claude-3", messages=[{"role": "user", "content": "hi"}]),
    )
    assert any(s.name == "anthropic.messages.create" for s in spans)
