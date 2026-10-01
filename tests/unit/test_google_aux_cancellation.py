import asyncio
import sys
import types
from types import SimpleNamespace

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from neatlogs._wrap_utils import set_neatlogs_provider


def _setup_provider():
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    set_neatlogs_provider(provider)
    return provider, exporter


def _assert_cancelled_span(exporter, expect_root=False):
    spans = exporter.get_finished_spans()
    if expect_root:
        assert len(spans) == 2
        child = [s for s in spans if s.attributes.get("neatlogs.auto_root") is not True][0]
        root = [s for s in spans if s.attributes.get("neatlogs.auto_root") is True][0]
        assert root.attributes.get("neatlogs.span.kind") == "workflow"
    else:
        assert len(spans) == 1
        child = spans[0]
    assert child.status.status_code.name == "UNSET"
    assert child.attributes.get("neatlogs.stream.cancelled") is True
    assert len(child.events) == 0
    return child


def _blocking_coro(started):
    async def _impl(*args, **kwargs):
        started.set()
        await asyncio.Future()

    return _impl


@pytest.mark.asyncio
async def test_google_embed_cancellation_ends_span():
    import neatlogs.google_genai as gg

    _, exporter = _setup_provider()
    started = asyncio.Event()
    fake = SimpleNamespace()
    fake.embed_content = _blocking_coro(started)
    gg._patch_models_extra(fake, is_async=True)

    task = asyncio.create_task(fake.embed_content(model="m", contents="hi"))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    _assert_cancelled_span(exporter, expect_root=True)


@pytest.mark.asyncio
async def test_google_count_tokens_cancellation_ends_span():
    import neatlogs.google_genai as gg

    _, exporter = _setup_provider()
    started = asyncio.Event()
    fake = SimpleNamespace()
    fake.count_tokens = _blocking_coro(started)
    gg._patch_models_extra(fake, is_async=True)

    task = asyncio.create_task(fake.count_tokens(model="m", contents="hi"))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    _assert_cancelled_span(exporter)


@pytest.mark.asyncio
async def test_vertex_embed_cancellation_ends_span():
    import neatlogs.vertex_ai as vx

    _, exporter = _setup_provider()
    started = asyncio.Event()
    fake = SimpleNamespace()
    fake.embed_content = _blocking_coro(started)
    vx._patch_models_extra(fake, is_async=True)

    task = asyncio.create_task(fake.embed_content(model="m", contents="hi"))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    _assert_cancelled_span(exporter, expect_root=True)


@pytest.mark.asyncio
async def test_vertex_count_tokens_cancellation_ends_span():
    import neatlogs.vertex_ai as vx

    _, exporter = _setup_provider()
    started = asyncio.Event()
    fake = SimpleNamespace()
    fake.count_tokens = _blocking_coro(started)
    vx._patch_models_extra(fake, is_async=True)

    task = asyncio.create_task(fake.count_tokens(model="m", contents="hi"))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    _assert_cancelled_span(exporter)


def _install_fake_chats(blocking_send, blocking_send_stream):
    class Chat:
        def send_message(self, message, *args, **kwargs):
            return {"ok": True}

        def send_message_stream(self, message, *args, **kwargs):
            return iter([])

    class AsyncChat:
        def __init__(self):
            self._model = "test-model"

        async def send_message(self, message, *args, **kwargs):
            return await blocking_send(message, *args, **kwargs)

        async def send_message_stream(self, message, *args, **kwargs):
            return await blocking_send_stream(message, *args, **kwargs)

    google_mod = types.ModuleType("google")
    genai_mod = types.ModuleType("google.genai")
    chats_mod = types.ModuleType("google.genai.chats")
    chats_mod.Chat = Chat
    chats_mod.AsyncChat = AsyncChat
    genai_mod.chats = chats_mod

    prev = {}
    for name, mod in {
        "google": google_mod,
        "google.genai": genai_mod,
        "google.genai.chats": chats_mod,
    }.items():
        if name in sys.modules:
            prev[name] = sys.modules[name]
        sys.modules[name] = mod
    return Chat, AsyncChat, prev


def _restore_chats(prev):
    for name in ["google", "google.genai", "google.genai.chats"]:
        sys.modules.pop(name, None)
    sys.modules.update(prev)


@pytest.mark.asyncio
async def test_google_chat_send_cancellation_ends_span():
    import neatlogs.google_genai as gg

    _, exporter = _setup_provider()
    started = asyncio.Event()
    _, AsyncChat, prev = _install_fake_chats(
        _blocking_coro(started), _blocking_coro(asyncio.Event())
    )
    try:
        gg._patch_chat_classes()
        chat = AsyncChat()
        task = asyncio.create_task(chat.send_message("hi"))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        _assert_cancelled_span(exporter)
    finally:
        _restore_chats(prev)


@pytest.mark.asyncio
async def test_vertex_chat_send_cancellation_ends_span():
    import neatlogs.vertex_ai as vx

    _, exporter = _setup_provider()
    started = asyncio.Event()
    _, AsyncChat, prev = _install_fake_chats(
        _blocking_coro(started), _blocking_coro(asyncio.Event())
    )
    try:
        vx._patch_chat_classes()
        chat = AsyncChat()
        task = asyncio.create_task(chat.send_message("hi"))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        _assert_cancelled_span(exporter)
    finally:
        _restore_chats(prev)


@pytest.mark.asyncio
async def test_google_chat_stream_open_cancellation_ends_span():
    import neatlogs.google_genai as gg

    _, exporter = _setup_provider()
    started = asyncio.Event()
    _, AsyncChat, prev = _install_fake_chats(
        _blocking_coro(asyncio.Event()), _blocking_coro(started)
    )
    try:
        gg._patch_chat_classes()
        chat = AsyncChat()
        task = asyncio.create_task(chat.send_message_stream("hi"))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        _assert_cancelled_span(exporter)
    finally:
        _restore_chats(prev)


@pytest.mark.asyncio
async def test_vertex_chat_stream_open_cancellation_ends_span():
    import neatlogs.vertex_ai as vx

    _, exporter = _setup_provider()
    started = asyncio.Event()
    _, AsyncChat, prev = _install_fake_chats(
        _blocking_coro(asyncio.Event()), _blocking_coro(started)
    )
    try:
        vx._patch_chat_classes()
        chat = AsyncChat()
        task = asyncio.create_task(chat.send_message_stream("hi"))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        _assert_cancelled_span(exporter)
    finally:
        _restore_chats(prev)
