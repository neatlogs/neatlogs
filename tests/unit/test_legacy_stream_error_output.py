import asyncio

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from neatlogs._wrap_utils import AsyncStreamWrapper, SyncStreamWrapper


def pipeline():
    sink = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(sink))
    return provider.get_tracer("test").start_span("stream"), sink


def finalizer(span, chunks, duration, ttft, interrupted=False):
    span.set_attribute("partial", "".join(chunks))
    span.set_status(StatusCode.UNSET if interrupted else StatusCode.OK)
    span.end()


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_failed_legacy_stream_keeps_partial_output(mode):
    span, sink = pipeline()

    def source():
        yield "partial"
        raise ValueError("connection lost")

    async def asource():
        yield "partial"
        raise ValueError("connection lost")

    async def consume():
        async for _ in AsyncStreamWrapper(asource(), span, finalizer):
            pass

    with pytest.raises(ValueError, match="connection lost"):
        if mode == "sync":
            list(SyncStreamWrapper(source(), span, finalizer))
        else:
            asyncio.run(consume())
    result = sink.get_finished_spans()
    assert len(result) == 1
    assert result[0].status.status_code == StatusCode.ERROR
    assert result[0].attributes["partial"] == "partial"


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize("error", [ValueError("broken"), GeneratorExit()])
@pytest.mark.parametrize("modern", [True, False])
@pytest.mark.parametrize(
    "finalizer_error", [RuntimeError("telemetry error"), GeneratorExit(), asyncio.CancelledError()]
)
def test_failed_finalizer_cannot_hide_provider_error(mode, error, modern, finalizer_error):
    span, sink = pipeline()

    def broken(*args, **kwargs):
        raise finalizer_error

    if modern:
        finish = broken
    else:

        def finish(span, chunks, duration, ttft):
            broken()

    def source():
        yield "partial"
        raise error

    async def asource():
        yield "partial"
        raise error

    async def consume():
        async for _ in AsyncStreamWrapper(asource(), span, finish):
            pass

    with pytest.raises(type(error)) as raised:
        if mode == "sync":
            list(SyncStreamWrapper(source(), span, finish))
        else:
            asyncio.run(consume())
    assert raised.value is error
    result = sink.get_finished_spans()
    assert len(result) == 1
    assert result[0].status.status_code == (
        StatusCode.UNSET if isinstance(error, GeneratorExit) else StatusCode.ERROR
    )


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_anthropic_finalizer_keeps_partial_text_and_usage(mode):
    from types import SimpleNamespace as NS

    pytest.importorskip("anthropic")
    from neatlogs.anthropic import _finalize_stream

    span, sink = pipeline()
    events = [
        NS(type="message_start", message=NS(model="mock", usage=NS(input_tokens=5))),
        NS(type="content_block_delta", index=0, delta=NS(type="text_delta", text="partial")),
    ]

    def source():
        yield from events
        raise ValueError("connection lost")

    async def asource():
        for event in events:
            yield event
        raise ValueError("connection lost")

    async def consume():
        async for _ in AsyncStreamWrapper(asource(), span, _finalize_stream):
            pass

    with pytest.raises(ValueError):
        if mode == "sync":
            list(SyncStreamWrapper(source(), span, _finalize_stream))
        else:
            asyncio.run(consume())
    result = sink.get_finished_spans()[0]
    assert result.status.status_code == StatusCode.ERROR
    assert result.attributes["neatlogs.llm.output_messages.0.content"] == "partial"
    assert result.attributes["neatlogs.llm.token_count.prompt"] == 5
    assert len(result.events) == 1


def test_labeled_smoke_60_failed_and_60_clean_streams():
    for i in range(120):
        failed = i % 2 == 0
        span, sink = pipeline()

        def source():
            yield "partial"
            if failed:
                raise ValueError("connection lost")

        if failed:
            with pytest.raises(ValueError):
                list(SyncStreamWrapper(source(), span, finalizer))
        else:
            list(SyncStreamWrapper(source(), span, finalizer))
        result = sink.get_finished_spans()[0]
        assert result.status.status_code == (StatusCode.ERROR if failed else StatusCode.OK)
        assert result.attributes["partial"] == "partial"
