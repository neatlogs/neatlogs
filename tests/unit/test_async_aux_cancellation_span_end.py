"""Regression: async Google GenAI / Vertex AI auxiliary wrappers must end their
span when the awaited call is cancelled. asyncio.CancelledError inherits
BaseException (not Exception), so an `except Exception` guard let the span leak
open until shutdown. Extends the primary-call cancellation fix to the
embed_content, count_tokens and chat send paths.
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


async def _cancel_and_get_spans(exporter, call):
    task = asyncio.create_task(call())
    await asyncio.sleep(0.02)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0.02)
    return exporter.get_finished_spans()


@pytest.mark.asyncio
async def test_google_genai_embed_ends_span_on_cancel(tracer_provider, in_memory_span_exporter):
    _init(tracer_provider)
    from neatlogs.google_genai import _patch_models_extra

    class M:
        async def embed_content(self, *a, **k):
            await asyncio.sleep(10)

    obj = M()
    _patch_models_extra(obj, is_async=True)
    spans = await _cancel_and_get_spans(
        in_memory_span_exporter,
        lambda: obj.embed_content(model="text-embedding-004", contents="hi"),
    )
    assert any(s.name == "google_genai.models.embed_content" for s in spans)


@pytest.mark.asyncio
async def test_google_genai_count_tokens_ends_span_on_cancel(
    tracer_provider, in_memory_span_exporter
):
    _init(tracer_provider)
    from neatlogs.google_genai import _patch_models_extra

    class M:
        async def count_tokens(self, *a, **k):
            await asyncio.sleep(10)

    obj = M()
    _patch_models_extra(obj, is_async=True)
    spans = await _cancel_and_get_spans(
        in_memory_span_exporter,
        lambda: obj.count_tokens(model="gemini-2.0-flash", contents="hi"),
    )
    assert any(s.name == "google_genai.models.count_tokens" for s in spans)


@pytest.mark.asyncio
async def test_vertex_ai_embed_ends_span_on_cancel(tracer_provider, in_memory_span_exporter):
    _init(tracer_provider)
    from neatlogs.vertex_ai import _patch_models_extra

    class M:
        async def embed_content(self, *a, **k):
            await asyncio.sleep(10)

    obj = M()
    _patch_models_extra(obj, is_async=True)
    spans = await _cancel_and_get_spans(
        in_memory_span_exporter,
        lambda: obj.embed_content(model="text-embedding-004", contents="hi"),
    )
    assert any(s.name == "vertex_ai.models.embed_content" for s in spans)


@pytest.mark.asyncio
async def test_vertex_ai_count_tokens_ends_span_on_cancel(tracer_provider, in_memory_span_exporter):
    _init(tracer_provider)
    from neatlogs.vertex_ai import _patch_models_extra

    class M:
        async def count_tokens(self, *a, **k):
            await asyncio.sleep(10)

    obj = M()
    _patch_models_extra(obj, is_async=True)
    spans = await _cancel_and_get_spans(
        in_memory_span_exporter,
        lambda: obj.count_tokens(model="gemini-2.0-flash", contents="hi"),
    )
    assert any(s.name == "vertex_ai.models.count_tokens" for s in spans)


@pytest.mark.asyncio
async def test_google_genai_chat_send_ends_span_on_cancel(tracer_provider, in_memory_span_exporter):
    pytest.importorskip("google.genai")
    _init(tracer_provider)
    from google.genai.chats import AsyncChat

    import neatlogs.google_genai as gg

    async def fake_send(self, message, *a, **k):
        await asyncio.sleep(10)

    AsyncChat.send_message = fake_send
    AsyncChat._neatlogs_patched = False
    gg._patch_chat_classes()
    chat = object.__new__(AsyncChat)
    spans = await _cancel_and_get_spans(
        in_memory_span_exporter,
        lambda: chat.send_message("hi"),
    )
    assert any(s.name == "google_genai.chat.send_message" for s in spans)


@pytest.mark.asyncio
async def test_vertex_ai_chat_send_ends_span_on_cancel(tracer_provider, in_memory_span_exporter):
    pytest.importorskip("google.genai")
    _init(tracer_provider)
    from google.genai.chats import AsyncChat

    import neatlogs.vertex_ai as vx

    async def fake_send(self, message, *a, **k):
        await asyncio.sleep(10)

    AsyncChat.send_message = fake_send
    AsyncChat._neatlogs_vertex_patched = False
    vx._patch_chat_classes()
    chat = object.__new__(AsyncChat)
    spans = await _cancel_and_get_spans(
        in_memory_span_exporter,
        lambda: chat.send_message("hi"),
    )
    assert any(s.name == "vertex_ai.chat.send_message" for s in spans)
