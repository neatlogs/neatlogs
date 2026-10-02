"""Stream ttft/duration are measured from the request start, not wrapper creation."""

import asyncio
import time

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from neatlogs._wrap_utils import AsyncStreamWrapper, SyncStreamWrapper

WAIT = 0.3


class _Finalizer:
    def __init__(self):
        self.seen = None

    def on_chunk(self, span, chunk):
        pass

    def finish(self, span, duration_ms, ttft_ms, *, interrupted=False):
        self.seen = (duration_ms, ttft_ms)
        span.end()

    def fail(self, span, error):
        span.end()


def _span():
    tp = TracerProvider()
    tp.add_span_processor(SimpleSpanProcessor(InMemorySpanExporter()))
    return tp.get_tracer("t").start_span("llm")


def test_sync_stream_counts_wait_before_wrapper():
    span = _span()
    time.sleep(WAIT)  # provider call blocks until response headers
    fin = _Finalizer()
    list(SyncStreamWrapper(iter(["a", "b"]), span, fin))
    duration_ms, ttft_ms = fin.seen
    assert ttft_ms >= WAIT * 1000 * 0.9
    assert duration_ms >= ttft_ms


def test_async_stream_counts_wait_before_wrapper():
    async def gen():
        yield "a"
        yield "b"

    async def run():
        span = _span()
        await asyncio.sleep(WAIT)
        fin = _Finalizer()
        async for _ in AsyncStreamWrapper(gen(), span, fin):
            pass
        return fin.seen

    duration_ms, ttft_ms = asyncio.run(run())
    assert ttft_ms >= WAIT * 1000 * 0.9
    assert duration_ms >= ttft_ms


def test_span_without_start_time_falls_back_to_wrapper_start():
    class Bare:
        def set_attribute(self, *a):
            pass

        def set_status(self, *a):
            pass

        def end(self):
            pass

    fin = _Finalizer()
    list(SyncStreamWrapper(iter(["a"]), Bare(), fin))
    duration_ms, ttft_ms = fin.seen
    assert 0 <= ttft_ms < 100
