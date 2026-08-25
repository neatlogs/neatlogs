import gc
from types import SimpleNamespace

import pytest

import neatlogs.google_genai as google_genai


class _Models:
    def __init__(self, factory):
        self._factory = factory

    def generate_content(self, *args, **kwargs):
        return SimpleNamespace(candidates=[], usage_metadata=None)

    def generate_content_stream(self, *args, **kwargs):
        return self._factory()


class _AsyncModels:
    def __init__(self, factory):
        self._factory = factory

    async def generate_content(self, *args, **kwargs):
        return SimpleNamespace(candidates=[], usage_metadata=None)

    async def generate_content_stream(self, *args, **kwargs):
        return self._factory()


def _chunk(text):
    part = SimpleNamespace(text=text, thought=False, function_call=None)
    content = SimpleNamespace(parts=[part])
    return SimpleNamespace(
        candidates=[SimpleNamespace(content=content, finish_reason=None)], usage_metadata=None
    )


def test_google_sync_iterator_ownership_all_terminal_paths(
    monkeypatch, tracer_provider, in_memory_span_exporter
):
    monkeypatch.setattr(
        google_genai, "get_provider_tracer", lambda: tracer_provider.get_tracer("google.test")
    )

    models = _Models(lambda: iter([_chunk("a"), _chunk("b")]))
    google_genai._patch_models(models)
    assert len(list(models.generate_content_stream(model="m", contents="q"))) == 2

    models = _Models(lambda: iter([_chunk("partial"), _chunk("later")]))
    google_genai._patch_models(models)
    stream = models.generate_content_stream(model="m", contents="q")
    next(stream)
    stream.close()
    stream.close()

    models = _Models(lambda: iter([_chunk("unused")]))
    google_genai._patch_models(models)
    stream = models.generate_content_stream(model="m", contents="q")
    del stream
    gc.collect()

    def failing():
        yield _chunk("partial")
        raise RuntimeError("provider boom")

    models = _Models(failing)
    google_genai._patch_models(models)
    stream = models.generate_content_stream(model="m", contents="q")
    next(stream)
    with pytest.raises(RuntimeError):
        next(stream)

    spans = in_memory_span_exporter.get_finished_spans()
    assert [span.attributes["neatlogs.stream.completion_state"] for span in spans] == [
        "complete",
        "consumer_cancelled",
        "consumer_cancelled",
        "provider_error",
    ]
    assert len(spans) == 4


@pytest.mark.asyncio
async def test_google_async_iterator_preserves_awaitable_contract_and_close_ownership(
    monkeypatch, tracer_provider, in_memory_span_exporter
):
    monkeypatch.setattr(
        google_genai, "get_provider_tracer", lambda: tracer_provider.get_tracer("google.test")
    )

    async def chunks():
        yield _chunk("a")
        yield _chunk("b")

    models = _AsyncModels(chunks)
    google_genai._patch_async_models(models)
    stream = await models.generate_content_stream(model="m", contents="q")
    assert [item async for item in stream]

    models = _AsyncModels(chunks)
    google_genai._patch_async_models(models)
    stream = await models.generate_content_stream(model="m", contents="q")
    await anext(stream)
    await stream.aclose()
    await stream.aclose()

    spans = in_memory_span_exporter.get_finished_spans()
    assert [span.attributes["neatlogs.stream.completion_state"] for span in spans] == [
        "complete",
        "consumer_cancelled",
    ]
