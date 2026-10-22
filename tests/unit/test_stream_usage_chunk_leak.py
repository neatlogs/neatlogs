"""The wrapper's forced include_usage chunk must not leak into caller iteration."""

import json

import httpx
import openai
import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from neatlogs._wrap_utils import SyncStreamWrapper
from neatlogs.openai import wrap_openai_client


def _sse(obj):
    return f"data: {json.dumps(obj)}\n\n"


def _stream_body(include_usage):
    parts = []
    for i, tok in enumerate(["hi", " there"]):
        delta = {"role": "assistant", "content": tok} if i == 0 else {"content": tok}
        parts.append(
            _sse(
                {
                    "id": "c1",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": "gpt-x",
                    "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
                }
            )
        )
    parts.append(
        _sse(
            {
                "id": "c1",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "gpt-x",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
        )
    )
    if include_usage:
        parts.append(
            _sse(
                {
                    "id": "c1",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": "gpt-x",
                    "choices": [],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
                }
            )
        )
    parts.append("data: [DONE]\n\n")
    return "".join(parts)


def _handler(request):
    body = json.loads(request.content)
    include_usage = (body.get("stream_options") or {}).get("include_usage", False)
    return httpx.Response(
        200, headers={"content-type": "text/event-stream"}, text=_stream_body(include_usage)
    )


def _client():
    return openai.OpenAI(
        api_key="x", http_client=httpx.Client(transport=httpx.MockTransport(_handler))
    )


@pytest.fixture
def exporter():
    exp = InMemorySpanExporter()
    tp = TracerProvider()
    tp.add_span_processor(SimpleSpanProcessor(exp))
    import neatlogs

    neatlogs.init(tracer_provider=tp, api_key="test", debug=False)
    yield exp
    neatlogs.shutdown()


def _collect(stream):
    out = []
    for chunk in stream:
        out.append(chunk.choices[0].delta.content or "")
    return "".join(out)


def test_forced_usage_chunk_not_yielded_to_caller(exporter):
    wrapped = wrap_openai_client(_client())
    assert (
        _collect(
            wrapped.chat.completions.create(
                model="gpt-x", messages=[{"role": "user", "content": "hi"}], stream=True
            )
        )
        == "hi there"
    )


def test_forced_usage_chunk_still_feeds_span_usage(exporter):
    wrapped = wrap_openai_client(_client())
    _collect(
        wrapped.chat.completions.create(
            model="gpt-x", messages=[{"role": "user", "content": "hi"}], stream=True
        )
    )
    spans = exporter.get_finished_spans()
    llm = [s for s in spans if s.name == "openai.chat.completions.create"]
    assert llm, [s.name for s in spans]
    attrs = llm[-1].attributes
    assert attrs.get("neatlogs.llm.token_count.total") == 7
    assert attrs.get("neatlogs.llm.token_count.prompt") == 5


def test_caller_requested_usage_chunk_passes_through(exporter):
    wrapped = wrap_openai_client(_client())
    stream = wrapped.chat.completions.create(
        model="gpt-x",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
        stream_options={"include_usage": True},
    )
    kinds = [(len(c.choices), c.usage is not None) for c in stream]
    assert (0, True) in kinds  # the caller asked for usage, so the usage chunk is theirs


def test_caller_stream_options_dict_not_mutated(exporter):
    wrapped = wrap_openai_client(_client())
    opts = {"include_usage": False}
    _collect(
        wrapped.chat.completions.create(
            model="gpt-x",
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
            stream_options=opts,
        )
    )
    assert opts == {"include_usage": False}


def test_wrapper_skip_chunk_keeps_finalizer_informed():
    seen = []

    class Fin:
        def on_chunk(self, span, chunk):
            seen.append(chunk)

        def finish(self, span, duration_ms, ttft_ms, *, interrupted=False):
            span.end()

        def fail(self, span, error):
            span.end()

    tp = TracerProvider()
    tp.add_span_processor(SimpleSpanProcessor(InMemorySpanExporter()))
    span = tp.get_tracer("t").start_span("llm")
    wrapper = SyncStreamWrapper(
        iter(["a", "usage-chunk", "b"]), span, Fin(), skip_chunk=lambda c: c == "usage-chunk"
    )
    assert list(wrapper) == ["a", "b"]
    assert seen == ["a", "usage-chunk", "b"]


@pytest.mark.asyncio
@pytest.mark.parametrize("azure", [False, True])
@pytest.mark.parametrize("requested", [False, True])
async def test_async_openai_and_azure_usage_matrix(exporter, azure, requested):
    from neatlogs.azure_openai import wrap_azure_openai_client

    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return _handler(request)

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    if azure:
        client = openai.AsyncAzureOpenAI(
            api_key="test",
            azure_endpoint="https://azure.invalid",
            api_version="2024-10-21",
            http_client=http_client,
        )
        wrapped = wrap_azure_openai_client(client)
    else:
        client = openai.AsyncOpenAI(api_key="test", http_client=http_client)
        wrapped = wrap_openai_client(client)
    options = {"include_usage": requested}
    try:
        stream = await wrapped.chat.completions.create(
            model="gpt-x",
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
            stream_options=options,
        )
        chunks = [chunk async for chunk in stream]
        assert any(not c.choices and c.usage for c in chunks) is requested
        assert "".join(c.choices[0].delta.content or "" for c in chunks if c.choices) == "hi there"
        assert options == {"include_usage": requested}
        assert requests[0]["stream_options"]["include_usage"] is True
        llm = [
            s
            for s in exporter.get_finished_spans()
            if s.attributes.get("neatlogs.span.kind") == "llm"
        ][-1]
        assert llm.attributes["neatlogs.llm.token_count.total"] == 7
        assert llm.end_time is not None
    finally:
        await client.close()


@pytest.mark.parametrize("requested", [False, True])
def test_sync_azure_usage_matrix(exporter, requested):
    from neatlogs.azure_openai import wrap_azure_openai_client

    client = openai.AzureOpenAI(
        api_key="test",
        azure_endpoint="https://azure.invalid",
        api_version="2024-10-21",
        http_client=httpx.Client(transport=httpx.MockTransport(_handler)),
    )
    options = {"include_usage": requested}
    try:
        wrapped = wrap_azure_openai_client(client)
        chunks = list(
            wrapped.chat.completions.create(
                model="gpt-x",
                messages=[],
                stream=True,
                stream_options=options,
            )
        )
        assert any(not c.choices and c.usage for c in chunks) is requested
        assert options == {"include_usage": requested}
        llm = [
            s
            for s in exporter.get_finished_spans()
            if s.attributes.get("neatlogs.span.kind") == "llm"
        ][-1]
        assert llm.attributes["neatlogs.llm.token_count.total"] == 7
    finally:
        client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_async_hidden_chunk_error_and_cancellation_finalize(exporter, cancel):
    import asyncio

    from neatlogs._wrap_utils import AsyncStreamWrapper

    seen = []
    tp = TracerProvider()
    span = tp.get_tracer("test").start_span("async-boundary")

    class Finalizer:
        def on_chunk(self, span, chunk):
            seen.append(chunk)

        def finish(self, span, duration_ms, ttft_ms, *, interrupted=False):
            span.end()

        def fail(self, span, error):
            span.end()

    async def source():
        yield "usage"
        if cancel:
            raise asyncio.CancelledError()
        raise RuntimeError("stream failed")

    wrapper = AsyncStreamWrapper(source(), span, Finalizer(), skip_chunk=lambda c: c == "usage")
    with pytest.raises(asyncio.CancelledError if cancel else RuntimeError):
        await wrapper.__anext__()
    assert seen == ["usage"]
    assert not span.is_recording()
    tp.shutdown()
