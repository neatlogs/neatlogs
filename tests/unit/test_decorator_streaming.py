import asyncio
import gc

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import neatlogs
from neatlogs._wrap_utils import set_neatlogs_provider


def _install(provider):
    set_neatlogs_provider(provider)


def test_sync_generator_exhaustion_and_exactly_once_close(tracer_provider, in_memory_span_exporter):
    _install(tracer_provider)

    @neatlogs.span(kind="CHAIN")
    def stream():
        yield "a"
        yield "b"

    result = stream()
    assert list(result) == ["a", "b"]
    result.close()
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].attributes["neatlogs.stream.completion_state"] == "complete"
    assert spans[0].attributes["neatlogs.stream.chunk_count"] == 2


def test_sync_early_close_partial_and_never_consumed(tracer_provider, in_memory_span_exporter):
    _install(tracer_provider)

    @neatlogs.span(kind="CHAIN")
    def stream():
        yield "first"
        yield "second"

    never = stream()
    del never
    gc.collect()
    assert in_memory_span_exporter.get_finished_spans() == ()

    result = stream()
    assert next(result) == "first"
    result.close()
    span = in_memory_span_exporter.get_finished_spans()[0]
    assert span.attributes["neatlogs.stream.completion_state"] == "consumer_cancelled"
    assert span.attributes["output.value"] == '["first"]'


def test_sync_midstream_error_and_gc_finalization(tracer_provider, in_memory_span_exporter):
    _install(tracer_provider)

    @neatlogs.span(kind="CHAIN")
    def broken():
        yield "partial"
        raise RuntimeError("boom")

    result = broken()
    assert next(result) == "partial"
    with pytest.raises(RuntimeError):
        next(result)
    assert (
        in_memory_span_exporter.get_finished_spans()[0].attributes[
            "neatlogs.stream.completion_state"
        ]
        == "provider_error"
    )

    @neatlogs.span(kind="CHAIN")
    def abandoned():
        yield "partial"
        yield "later"

    result = abandoned()
    next(result)
    del result
    gc.collect()
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 2
    assert spans[-1].attributes["neatlogs.stream.completion_state"] == "consumer_cancelled"


@pytest.mark.asyncio
async def test_async_exhaust_close_error_and_cancellation(tracer_provider, in_memory_span_exporter):
    _install(tracer_provider)

    @neatlogs.span(kind="CHAIN")
    async def complete():
        yield "a"
        yield "b"

    assert [item async for item in complete()] == ["a", "b"]

    @neatlogs.span(kind="CHAIN")
    async def partial():
        yield "first"
        await asyncio.sleep(10)

    result = partial()
    assert await anext(result) == "first"
    await result.aclose()

    result = partial()
    assert await anext(result) == "first"
    pending = asyncio.create_task(anext(result))
    await asyncio.sleep(0)
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending

    @neatlogs.span(kind="CHAIN")
    async def broken():
        yield "first"
        raise RuntimeError("boom")

    result = broken()
    await anext(result)
    with pytest.raises(RuntimeError):
        await anext(result)

    states = [
        s.attributes["neatlogs.stream.completion_state"]
        for s in in_memory_span_exporter.get_finished_spans()
    ]
    assert states == [
        "complete",
        "consumer_cancelled",
        "consumer_cancelled",
        "provider_error",
    ]


def test_stream_output_is_bounded(tracer_provider, in_memory_span_exporter):
    _install(tracer_provider)

    @neatlogs.span(kind="CHAIN")
    def stream():
        yield "x" * 120_000
        yield "small"

    list(stream())
    span = in_memory_span_exporter.get_finished_spans()[0]
    assert len(span.attributes["output.value"]) <= 100_000
    assert span.attributes["neatlogs.stream.output_truncated"] is True
    assert span.attributes["neatlogs.stream.chunk_count"] == 2


def test_never_started_generator_binds_to_current_reinit_generation():
    first, second = TracerProvider(), TracerProvider()
    first_export, second_export = InMemorySpanExporter(), InMemorySpanExporter()
    first.add_span_processor(SimpleSpanProcessor(first_export))
    second.add_span_processor(SimpleSpanProcessor(second_export))

    @neatlogs.span(kind="CHAIN")
    def stream():
        yield "value"

    set_neatlogs_provider(first)
    pending = stream()  # generator body and span have not started
    set_neatlogs_provider(second)
    assert list(pending) == ["value"]
    assert first_export.get_finished_spans() == ()
    assert len(second_export.get_finished_spans()) == 1
    first.shutdown()
    second.shutdown()
