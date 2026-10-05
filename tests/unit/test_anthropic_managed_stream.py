"""messages.stream() spans keep output/usage for text_stream and final-message helpers,
and mid-stream failures export an ERROR span."""

import asyncio
import json

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

anthropic = pytest.importorskip("anthropic")

from neatlogs import _wrap_utils as w  # noqa: E402
from neatlogs.anthropic import (  # noqa: E402
    wrap_anthropic_client,
    wrap_async_anthropic_client,
)

try:
    import httpx2 as httpx
except ImportError:  # older anthropic releases use httpx
    import httpx

KW = dict(model="claude-x", max_tokens=5, messages=[{"role": "user", "content": "hi"}])


def _event(name, data):
    return f"event: {name}\ndata: {json.dumps(data)}\n\n"


def _body(broken):
    usage = {"input_tokens": 5, "output_tokens": 1}
    body = _event(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": "m",
                "type": "message",
                "role": "assistant",
                "model": "claude-x",
                "content": [],
                "stop_reason": None,
                "usage": usage,
            },
        },
    )
    body += _event(
        "content_block_start",
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
    )
    for text in ("hel", "lo"):
        if broken and text == "lo":
            return body + "event: content_block_delta\ndata: {broken\n\n"
        body += _event(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": text},
            },
        )
    body += _event("content_block_stop", {"type": "content_block_stop", "index": 0})
    body += _event(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 3},
        },
    )
    return body + _event("message_stop", {"type": "message_stop"})


def _handler(broken):
    def handle(request):
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=_body(broken).encode(),
        )

    return handle


@pytest.fixture
def exporter():
    exp = InMemorySpanExporter()
    prov = TracerProvider()
    prov.add_span_processor(SimpleSpanProcessor(exp))
    w.set_neatlogs_provider(prov)
    yield exp
    exp.clear()


def _sync_client(broken):
    client = anthropic.Anthropic(
        api_key="k",
        http_client=httpx.Client(transport=httpx.MockTransport(_handler(broken))),
        max_retries=0,
    )
    return wrap_anthropic_client(client)


def _async_client(broken):
    client = anthropic.AsyncAnthropic(
        api_key="k",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(_handler(broken))),
        max_retries=0,
    )
    return wrap_async_anthropic_client(client)


def _span(exporter):
    spans = [s for s in exporter.get_finished_spans() if s.name.startswith("anthropic")]
    assert len(spans) == 1
    return spans[0]


def _assert_full(span):
    attrs = span.attributes
    assert span.status.status_code.name == "OK"
    assert attrs["neatlogs.llm.output_messages.0.content"] == "hello"
    assert attrs["neatlogs.llm.token_count.prompt"] == 5
    assert attrs["neatlogs.llm.token_count.completion"] == 3


def _assert_error(span):
    assert span.status.status_code.name == "ERROR"
    assert any(e.name == "exception" for e in span.events)


@pytest.mark.parametrize("mode", ["iter", "text_stream", "final_message", "final_text"])
def test_sync_stream_keeps_output(exporter, mode):
    with _sync_client(False).messages.stream(**KW) as stream:
        if mode == "iter":
            for _ in stream:
                pass
        elif mode == "text_stream":
            for _ in stream.text_stream:
                pass
        elif mode == "final_message":
            stream.get_final_message()
        else:
            stream.get_final_text()
    _assert_full(_span(exporter))


@pytest.mark.parametrize("mode", ["iter", "text_stream", "final_message"])
def test_sync_mid_stream_error_is_error_span(exporter, mode):
    with pytest.raises(Exception):
        with _sync_client(True).messages.stream(**KW) as stream:
            if mode == "iter":
                for _ in stream:
                    pass
            elif mode == "text_stream":
                for _ in stream.text_stream:
                    pass
            else:
                stream.get_final_message()
    _assert_error(_span(exporter))


@pytest.mark.parametrize("mode", ["iter", "text_stream", "final_message"])
def test_async_stream_keeps_output(exporter, mode):
    async def run():
        async with _async_client(False).messages.stream(**KW) as stream:
            if mode == "iter":
                async for _ in stream:
                    pass
            elif mode == "text_stream":
                async for _ in stream.text_stream:
                    pass
            else:
                await stream.get_final_message()

    asyncio.run(run())
    _assert_full(_span(exporter))


@pytest.mark.parametrize("mode", ["iter", "text_stream", "final_message"])
def test_async_mid_stream_error_is_error_span(exporter, mode):
    async def run():
        async with _async_client(True).messages.stream(**KW) as stream:
            if mode == "iter":
                async for _ in stream:
                    pass
            elif mode == "text_stream":
                async for _ in stream.text_stream:
                    pass
            else:
                await stream.get_final_message()

    with pytest.raises(Exception):
        asyncio.run(run())
    _assert_error(_span(exporter))
