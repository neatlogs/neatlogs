import asyncio

import pytest
from opentelemetry import trace as otel_trace

import neatlogs
from neatlogs._wrap_utils import set_neatlogs_provider
from neatlogs.core.span_processor import NeatlogsSpanProcessor


def test_sync_generator_span_stays_open_and_records_every_chunk(
    tracer_provider, in_memory_span_exporter
):
    otel_trace.set_tracer_provider(tracer_provider)
    set_neatlogs_provider(tracer_provider)

    @neatlogs.span(kind="CHAIN")
    def stream():
        yield {"delta": "a"}
        yield {"delta": "b"}

    result = stream()
    assert next(result) == {"delta": "a"}
    assert in_memory_span_exporter.get_finished_spans() == ()
    assert list(result) == [{"delta": "b"}]

    finished = in_memory_span_exporter.get_finished_spans()
    assert len(finished) == 1
    assert finished[0].attributes["neatlogs.stream.chunk_count"] == 2
    assert [event.name for event in finished[0].events] == [
        "neatlogs.stream.chunk",
        "neatlogs.stream.chunk",
    ]
    assert [event.attributes["neatlogs.stream.chunk.index"] for event in finished[0].events] == [
        0,
        1,
    ]
    assert all(
        "neatlogs.stream.chunk.value" not in event.attributes for event in finished[0].events
    )
    assert finished[0].attributes["output.value"] == '[{"delta": "a"}, {"delta": "b"}]'


def test_sync_generator_close_exports_partial_output(tracer_provider, in_memory_span_exporter):
    otel_trace.set_tracer_provider(tracer_provider)
    set_neatlogs_provider(tracer_provider)

    @neatlogs.span(kind="CHAIN")
    def stream():
        yield "first"
        yield "never-consumed"

    result = stream()
    assert next(result) == "first"
    result.close()

    finished = in_memory_span_exporter.get_finished_spans()
    assert len(finished) == 1
    assert finished[0].attributes["neatlogs.stream.cancelled"] is True
    assert finished[0].attributes["output.value"] == '["first"]'


@pytest.mark.asyncio
async def test_async_generator_span_stays_open_and_records_every_chunk(
    tracer_provider, in_memory_span_exporter
):
    otel_trace.set_tracer_provider(tracer_provider)
    set_neatlogs_provider(tracer_provider)

    @neatlogs.span(kind="CHAIN")
    async def stream():
        yield "a"
        yield "b"

    result = stream()
    assert await anext(result) == "a"
    assert in_memory_span_exporter.get_finished_spans() == ()
    assert [chunk async for chunk in result] == ["b"]

    finished = in_memory_span_exporter.get_finished_spans()
    assert len(finished) == 1
    assert finished[0].attributes["neatlogs.stream.chunk_count"] == 2
    assert len(finished[0].events) == 2


@pytest.mark.asyncio
async def test_cancelled_coroutine_is_interrupted_not_error(
    tracer_provider, in_memory_span_exporter
):
    otel_trace.set_tracer_provider(tracer_provider)
    set_neatlogs_provider(tracer_provider)

    started = asyncio.Event()

    @neatlogs.span(kind="CHAIN")
    async def wait_forever():
        started.set()
        await asyncio.Future()

    task = asyncio.create_task(wait_forever())
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    finished = in_memory_span_exporter.get_finished_spans()
    assert len(finished) == 1
    assert finished[0].attributes["neatlogs.stream.cancelled"] is True
    assert finished[0].status.status_code.name == "UNSET"


def test_shutdown_of_active_sync_generator_exports_partial_output_without_fake_status(
    tracer_provider, in_memory_span_exporter
):
    lifecycle = NeatlogsSpanProcessor(emit_completion_markers=False)
    tracer_provider.add_span_processor(lifecycle)
    set_neatlogs_provider(tracer_provider)

    @neatlogs.span(kind="CHAIN")
    def stream():
        yield "first"
        yield "never-consumed"

    result = stream()
    assert next(result) == "first"
    assert lifecycle.end_active_spans("shutdown") == 1
    finished = in_memory_span_exporter.get_finished_spans()[0]
    assert finished.attributes["output.value"] == '["first"]'
    assert finished.attributes["neatlogs.trace.interrupted"] is True
    assert finished.status.status_code.name == "UNSET"
    result.close()
    assert len(in_memory_span_exporter.get_finished_spans()) == 1


@pytest.mark.asyncio
async def test_shutdown_of_active_async_generator_exports_partial_output_without_fake_status(
    tracer_provider, in_memory_span_exporter
):
    lifecycle = NeatlogsSpanProcessor(emit_completion_markers=False)
    tracer_provider.add_span_processor(lifecycle)
    set_neatlogs_provider(tracer_provider)

    @neatlogs.span(kind="CHAIN")
    async def stream():
        yield "first"
        yield "never-consumed"

    result = stream()
    assert await anext(result) == "first"
    assert lifecycle.end_active_spans("shutdown") == 1
    finished = in_memory_span_exporter.get_finished_spans()[0]
    assert finished.attributes["output.value"] == '["first"]'
    assert finished.attributes["neatlogs.trace.interrupted"] is True
    assert finished.status.status_code.name == "UNSET"
    await result.aclose()
    assert len(in_memory_span_exporter.get_finished_spans()) == 1
