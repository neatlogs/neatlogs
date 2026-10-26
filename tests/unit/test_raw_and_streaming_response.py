"""Regression tests for issue #234: with_raw_response and with_streaming_response
on Anthropic and OpenAI clients (sync and async, non-streaming and streaming).
"""

from __future__ import annotations

import json
from typing import Any

import anthropic
import httpx as _std_httpx
import openai
import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from neatlogs._wrap_utils import set_neatlogs_provider
from neatlogs.anthropic import wrap_anthropic_client, wrap_async_anthropic_client
from neatlogs.openai import wrap_async_openai_client, wrap_openai_client

httpx: Any = getattr(getattr(anthropic, "_base_client", None), "httpx2", _std_httpx)


@pytest.fixture
def exporter() -> InMemorySpanExporter:
    provider = TracerProvider()
    exp = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exp))
    set_neatlogs_provider(provider)
    return exp


def _anthropic_transport_handler(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    model = body.get("model", "claude-sonnet-4-20250514")
    if body.get("stream"):
        sse = (
            "event: message_start\ndata: "
            + json.dumps(
                {
                    "type": "message_start",
                    "message": {
                        "id": "msg_stream_1",
                        "type": "message",
                        "role": "assistant",
                        "model": model,
                        "content": [],
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {"input_tokens": 40, "output_tokens": 0},
                    },
                }
            )
            + "\n\n"
            + "event: content_block_delta\ndata: "
            + json.dumps(
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": "hello"},
                }
            )
            + "\n\n"
            + "event: message_delta\ndata: "
            + json.dumps(
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                    "usage": {"output_tokens": 250},
                }
            )
            + "\n\n"
            + "event: message_stop\ndata: "
            + json.dumps({"type": "message_stop"})
            + "\n\n"
        )
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream", "request-id": "req_ant_stream"},
            content=sse.encode("utf-8"),
        )

    return httpx.Response(
        200,
        headers={"content-type": "application/json", "request-id": "req_ant_json"},
        json={
            "id": "msg_json_1",
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [{"type": "text", "text": "hello"}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 40, "output_tokens": 250},
        },
    )


def _openai_chat_transport_handler(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    model = body.get("model", "gpt-4o")
    if body.get("stream"):
        sse = (
            "data: "
            + json.dumps(
                {
                    "id": "chatcmpl_stream_1",
                    "object": "chat.completion.chunk",
                    "created": 1700000000,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant", "content": "hello"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 40,
                        "completion_tokens": 250,
                        "total_tokens": 290,
                    },
                }
            )
            + "\n\ndata: [DONE]\n\n"
        )
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream", "x-request-id": "req_oai_stream"},
            content=sse.encode("utf-8"),
        )

    return httpx.Response(
        200,
        headers={"content-type": "application/json", "x-request-id": "req_oai_json"},
        json={
            "id": "chatcmpl_json_1",
            "object": "chat.completion",
            "created": 1700000000,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "hello"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 40,
                "completion_tokens": 250,
                "total_tokens": 290,
            },
        },
    )


def _openai_responses_transport_handler(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    model = body.get("model", "gpt-4o")
    if body.get("stream"):
        completed_resp = {
            "id": "resp_stream_1",
            "object": "response",
            "status": "completed",
            "model": model,
            "output": [
                {
                    "type": "message",
                    "id": "msg_1",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "hello", "annotations": []}],
                }
            ],
            "usage": {"input_tokens": 40, "output_tokens": 250, "total_tokens": 290},
        }
        sse = (
            "event: response.output_text.delta\ndata: "
            + json.dumps({"type": "response.output_text.delta", "delta": "hello"})
            + "\n\n"
            + "event: response.completed\ndata: "
            + json.dumps({"type": "response.completed", "response": completed_resp})
            + "\n\n"
        )
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=sse.encode("utf-8"),
        )

    return httpx.Response(
        200,
        headers={"content-type": "application/json"},
        json={
            "id": "resp_json_1",
            "object": "response",
            "created_at": 1700000000,
            "status": "completed",
            "model": model,
            "output": [
                {
                    "type": "message",
                    "id": "msg_1",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "hello", "annotations": []}],
                }
            ],
            "usage": {"input_tokens": 40, "output_tokens": 250, "total_tokens": 290},
        },
    )


def _assert_llm_span(span: Any, *, finish_reason: str = "end_turn") -> None:
    attrs = dict(span.attributes or {})
    assert attrs.get("neatlogs.llm.token_count.prompt") == 40
    assert attrs.get("neatlogs.llm.token_count.completion") == 250
    assert attrs.get("neatlogs.llm.token_count.total") == 290
    assert attrs.get("neatlogs.llm.output_messages.0.content") == "hello"
    assert attrs.get("neatlogs.llm.finish_reason") == finish_reason


# ---------------------------------------------------------------------------
# Anthropic Sync
# ---------------------------------------------------------------------------


def test_anthropic_sync_with_raw_response_non_streaming(exporter: InMemorySpanExporter):
    http_client = httpx.Client(transport=httpx.MockTransport(_anthropic_transport_handler))
    client = wrap_anthropic_client(anthropic.Anthropic(api_key="x", http_client=http_client))

    raw = client.messages.with_raw_response.create(
        model="claude-x", max_tokens=300, messages=[{"role": "user", "content": "hi"}]
    )
    assert raw.headers.get("request-id") == "req_ant_json"
    parsed = raw.parse()
    assert parsed.id == "msg_json_1"

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    _assert_llm_span(spans[0], finish_reason="end_turn")


def test_anthropic_sync_with_raw_response_streaming(exporter: InMemorySpanExporter):
    http_client = httpx.Client(transport=httpx.MockTransport(_anthropic_transport_handler))
    client = wrap_anthropic_client(anthropic.Anthropic(api_key="x", http_client=http_client))

    raw = client.messages.with_raw_response.create(
        model="claude-x",
        max_tokens=300,
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
    )
    assert raw.headers.get("request-id") == "req_ant_stream"
    events = list(raw.parse())
    assert len(events) == 4

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    _assert_llm_span(spans[0], finish_reason="end_turn")


def test_anthropic_sync_with_streaming_response_non_streaming(exporter: InMemorySpanExporter):
    http_client = httpx.Client(transport=httpx.MockTransport(_anthropic_transport_handler))
    client = wrap_anthropic_client(anthropic.Anthropic(api_key="x", http_client=http_client))

    with client.messages.with_streaming_response.create(
        model="claude-x", max_tokens=300, messages=[{"role": "user", "content": "hi"}]
    ) as response:
        assert response.headers.get("request-id") == "req_ant_json"
        msg = response.parse()
        assert msg.id == "msg_json_1"

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    _assert_llm_span(spans[0], finish_reason="end_turn")


def test_anthropic_sync_with_streaming_response_streaming(exporter: InMemorySpanExporter):
    http_client = httpx.Client(transport=httpx.MockTransport(_anthropic_transport_handler))
    client = wrap_anthropic_client(anthropic.Anthropic(api_key="x", http_client=http_client))

    with client.messages.with_streaming_response.create(
        model="claude-x",
        max_tokens=300,
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
    ) as response:
        events = list(response.parse())
        assert len(events) == 4

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    _assert_llm_span(spans[0], finish_reason="end_turn")


# ---------------------------------------------------------------------------
# Anthropic Async
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_anthropic_async_with_raw_and_streaming_response(exporter: InMemorySpanExporter):
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(_anthropic_transport_handler))
    client = wrap_async_anthropic_client(
        anthropic.AsyncAnthropic(api_key="x", http_client=http_client)
    )

    import inspect

    raw = await client.messages.with_raw_response.create(
        model="claude-x", max_tokens=300, messages=[{"role": "user", "content": "hi"}]
    )
    msg = (await raw.parse()) if inspect.iscoroutinefunction(raw.parse) else raw.parse()
    assert msg.id == "msg_json_1"

    raw_stream = await client.messages.with_raw_response.create(
        model="claude-x",
        max_tokens=300,
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
    )
    parsed_stream = (
        (await raw_stream.parse())
        if inspect.iscoroutinefunction(raw_stream.parse)
        else raw_stream.parse()
    )
    events = [ev async for ev in parsed_stream]
    assert len(events) == 4

    async with client.messages.with_streaming_response.create(
        model="claude-x", max_tokens=300, messages=[{"role": "user", "content": "hi"}]
    ) as resp:
        parsed_msg = await resp.parse()
        assert parsed_msg.id == "msg_json_1"

    async with client.messages.with_streaming_response.create(
        model="claude-x",
        max_tokens=300,
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
    ) as resp_stream:
        stream_events = [ev async for ev in await resp_stream.parse()]
        assert len(stream_events) == 4

    spans = exporter.get_finished_spans()
    assert len(spans) == 4
    for sp in spans:
        _assert_llm_span(sp, finish_reason="end_turn")


# ---------------------------------------------------------------------------
# OpenAI Sync (Chat Completions + Responses API)
# ---------------------------------------------------------------------------


def test_openai_sync_chat_with_raw_and_streaming_response(exporter: InMemorySpanExporter):
    http_client = httpx.Client(transport=httpx.MockTransport(_openai_chat_transport_handler))
    client = wrap_openai_client(openai.OpenAI(api_key="x", http_client=http_client))

    raw = client.chat.completions.with_raw_response.create(
        model="gpt-4o", messages=[{"role": "user", "content": "hi"}]
    )
    assert raw.headers.get("x-request-id") == "req_oai_json"
    parsed = raw.parse()
    assert parsed.id == "chatcmpl_json_1"

    raw_stream = client.chat.completions.with_raw_response.create(
        model="gpt-4o", messages=[{"role": "user", "content": "hi"}], stream=True
    )
    chunks = list(raw_stream.parse())
    assert len(chunks) == 1

    with client.chat.completions.with_streaming_response.create(
        model="gpt-4o", messages=[{"role": "user", "content": "hi"}]
    ) as resp:
        assert resp.parse().id == "chatcmpl_json_1"

    with client.chat.completions.with_streaming_response.create(
        model="gpt-4o", messages=[{"role": "user", "content": "hi"}], stream=True
    ) as resp_stream:
        stream_chunks = list(resp_stream.parse())
        assert len(stream_chunks) == 1

    spans = exporter.get_finished_spans()
    assert len(spans) == 4
    for sp in spans:
        _assert_llm_span(sp, finish_reason="stop")


def test_openai_sync_responses_with_raw_and_streaming_response(exporter: InMemorySpanExporter):
    http_client = httpx.Client(transport=httpx.MockTransport(_openai_responses_transport_handler))
    client = wrap_openai_client(openai.OpenAI(api_key="x", http_client=http_client))

    raw = client.responses.with_raw_response.create(model="gpt-4o", input="hi")
    assert raw.parse().id == "resp_json_1"

    raw_stream = client.responses.with_raw_response.create(model="gpt-4o", input="hi", stream=True)
    events = list(raw_stream.parse())
    assert len(events) == 2

    with client.responses.with_streaming_response.create(model="gpt-4o", input="hi") as resp:
        assert resp.parse().id == "resp_json_1"

    with client.responses.with_streaming_response.create(
        model="gpt-4o", input="hi", stream=True
    ) as resp_stream:
        stream_events = list(resp_stream.parse())
        assert len(stream_events) == 2

    spans = exporter.get_finished_spans()
    assert len(spans) == 4
    for sp in spans:
        _assert_llm_span(sp, finish_reason="completed")


# ---------------------------------------------------------------------------
# OpenAI Async (Chat Completions + Responses API)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_async_chat_and_responses_raw_and_streaming(exporter: InMemorySpanExporter):
    chat_http = httpx.AsyncClient(transport=httpx.MockTransport(_openai_chat_transport_handler))
    chat_client = wrap_async_openai_client(openai.AsyncOpenAI(api_key="x", http_client=chat_http))

    raw = await chat_client.chat.completions.with_raw_response.create(
        model="gpt-4o", messages=[{"role": "user", "content": "hi"}]
    )
    assert raw.parse().id == "chatcmpl_json_1"

    raw_stream = await chat_client.chat.completions.with_raw_response.create(
        model="gpt-4o", messages=[{"role": "user", "content": "hi"}], stream=True
    )
    chunks = [c async for c in raw_stream.parse()]
    assert len(chunks) == 1

    async with chat_client.chat.completions.with_streaming_response.create(
        model="gpt-4o", messages=[{"role": "user", "content": "hi"}]
    ) as resp:
        assert (await resp.parse()).id == "chatcmpl_json_1"

    async with chat_client.chat.completions.with_streaming_response.create(
        model="gpt-4o", messages=[{"role": "user", "content": "hi"}], stream=True
    ) as resp_stream:
        stream_chunks = [c async for c in await resp_stream.parse()]
        assert len(stream_chunks) == 1

    resp_http = httpx.AsyncClient(
        transport=httpx.MockTransport(_openai_responses_transport_handler)
    )
    resp_client = wrap_async_openai_client(openai.AsyncOpenAI(api_key="x", http_client=resp_http))

    r_raw = await resp_client.responses.with_raw_response.create(model="gpt-4o", input="hi")
    assert r_raw.parse().id == "resp_json_1"

    r_raw_stream = await resp_client.responses.with_raw_response.create(
        model="gpt-4o", input="hi", stream=True
    )
    r_events = [ev async for ev in r_raw_stream.parse()]
    assert len(r_events) == 2

    async with resp_client.responses.with_streaming_response.create(
        model="gpt-4o", input="hi"
    ) as r_resp:
        assert (await r_resp.parse()).id == "resp_json_1"

    async with resp_client.responses.with_streaming_response.create(
        model="gpt-4o", input="hi", stream=True
    ) as r_resp_stream:
        r_stream_events = [ev async for ev in await r_resp_stream.parse()]
        assert len(r_stream_events) == 2

    spans = exporter.get_finished_spans()
    assert len(spans) == 8
    for sp in spans[:4]:
        _assert_llm_span(sp, finish_reason="stop")
    for sp in spans[4:]:
        _assert_llm_span(sp, finish_reason="completed")


def test_precached_wrappers_and_unparsed_streaming_close(exporter: InMemorySpanExporter):
    from opentelemetry.trace import StatusCode

    from neatlogs.anthropic import _patch_anthropic_module, _unpatch_anthropic_module
    from neatlogs.openai import _patch_openai_module, _unpatch_openai_module

    _unpatch_anthropic_module()
    _unpatch_openai_module()
    try:
        a_http = httpx.Client(transport=httpx.MockTransport(_anthropic_transport_handler))
        a_client = anthropic.Anthropic(api_key="x", http_client=a_http)
        pre_a_raw = a_client.messages.with_raw_response
        pre_a_stream = a_client.messages.with_streaming_response
        wrap_anthropic_client(a_client)

        assert (
            pre_a_raw.create(
                model="claude-x", max_tokens=300, messages=[{"role": "user", "content": "hi"}]
            )
            .parse()
            .id
            == "msg_json_1"
        )

        with pre_a_stream.create(
            model="claude-x", max_tokens=300, messages=[{"role": "user", "content": "hi"}]
        ) as unparsed_resp:
            assert unparsed_resp.headers.get("request-id") == "req_ant_json"

        o_http = httpx.Client(transport=httpx.MockTransport(_openai_chat_transport_handler))
        o_client = openai.OpenAI(api_key="x", http_client=o_http)
        pre_o_raw = o_client.chat.completions.with_raw_response
        pre_o_stream = o_client.chat.completions.with_streaming_response
        wrap_openai_client(o_client)

        assert (
            pre_o_raw.create(model="gpt-4o", messages=[{"role": "user", "content": "hi"}])
            .parse()
            .id
            == "chatcmpl_json_1"
        )

        with pre_o_stream.create(
            model="gpt-4o", messages=[{"role": "user", "content": "hi"}]
        ) as unparsed_o_resp:
            assert unparsed_o_resp.headers.get("x-request-id") == "req_oai_json"

        spans = exporter.get_finished_spans()
        assert len(spans) == 4
        _assert_llm_span(spans[0], finish_reason="end_turn")
        assert spans[1].status.status_code == StatusCode.OK
        assert "neatlogs.llm.metrics.duration_ms" in spans[1].attributes
        assert "neatlogs.llm.output_messages.0.content" not in spans[1].attributes
        _assert_llm_span(spans[2], finish_reason="stop")
        assert spans[3].status.status_code == StatusCode.OK
        assert "neatlogs.llm.metrics.duration_ms" in spans[3].attributes
        assert "neatlogs.llm.output_messages.0.content" not in spans[3].attributes
    finally:
        _patch_anthropic_module()
        _patch_openai_module()


def test_simplenamespace_fake_raw_and_streaming_wrappers(exporter: InMemorySpanExporter):
    from contextlib import contextmanager
    from types import SimpleNamespace

    fake_msg = SimpleNamespace(
        id="msg_fake",
        model="claude-fake",
        stop_reason="end_turn",
        content=[SimpleNamespace(type="text", text="hello")],
        usage=SimpleNamespace(input_tokens=40, output_tokens=250),
    )

    @contextmanager
    def fake_stream_cm(**_kwargs):
        yield SimpleNamespace(parse=lambda: fake_msg, headers={"x-id": "1"})

    fake_anthropic = SimpleNamespace(
        messages=SimpleNamespace(
            create=lambda **_kw: fake_msg,
            with_raw_response=SimpleNamespace(
                create=lambda **_kw: SimpleNamespace(parse=lambda: fake_msg, headers={"x-id": "1"})
            ),
            with_streaming_response=SimpleNamespace(create=fake_stream_cm),
        )
    )
    wrap_anthropic_client(fake_anthropic)

    raw_res = fake_anthropic.messages.with_raw_response.create(
        model="claude-fake", messages=[{"role": "user", "content": "hi"}]
    )
    assert raw_res.parse().id == "msg_fake"

    with fake_anthropic.messages.with_streaming_response.create(
        model="claude-fake", messages=[{"role": "user", "content": "hi"}]
    ) as s_res:
        assert s_res.parse().id == "msg_fake"

    spans = exporter.get_finished_spans()
    assert len(spans) == 2
    for sp in spans:
        _assert_llm_span(sp, finish_reason="end_turn")


@pytest.mark.asyncio
async def test_custom_parse_types_bytes_and_dict_sync_and_async(exporter: InMemorySpanExporter):
    a_sync = wrap_anthropic_client(
        anthropic.Anthropic(
            api_key="x",
            http_client=httpx.Client(transport=httpx.MockTransport(_anthropic_transport_handler)),
        )
    )
    with a_sync.messages.with_streaming_response.create(
        model="claude-x", max_tokens=300, messages=[{"role": "user", "content": "hi"}]
    ) as resp:
        d = resp.parse(to=dict)
        assert isinstance(d, dict) and d["id"] == "msg_json_1"

    a_async = wrap_async_anthropic_client(
        anthropic.AsyncAnthropic(
            api_key="x",
            http_client=httpx.AsyncClient(
                transport=httpx.MockTransport(_anthropic_transport_handler)
            ),
        )
    )
    async with a_async.messages.with_streaming_response.create(
        model="claude-x", max_tokens=300, messages=[{"role": "user", "content": "hi"}]
    ) as a_resp:
        b = await a_resp.parse(to=bytes)
        assert isinstance(b, bytes) and b[0] == ord("{")

    o_async = wrap_async_openai_client(
        openai.AsyncOpenAI(
            api_key="x",
            http_client=httpx.AsyncClient(
                transport=httpx.MockTransport(_openai_responses_transport_handler)
            ),
        )
    )
    r_raw = await o_async.responses.with_raw_response.create(model="gpt-4o", input="hi")
    r_bytes = r_raw.parse(to=bytes)
    assert isinstance(r_bytes, bytes) and r_bytes[0] == ord("{")

    async with o_async.responses.with_streaming_response.create(
        model="gpt-4o", input="hi"
    ) as r_stream_resp:
        r_dict = await r_stream_resp.parse(to=dict)
        assert isinstance(r_dict, dict) and r_dict["id"] == "resp_json_1"

    spans = exporter.get_finished_spans()
    assert len(spans) == 4
    _assert_llm_span(spans[0], finish_reason="end_turn")
    _assert_llm_span(spans[1], finish_reason="end_turn")
    _assert_llm_span(spans[2], finish_reason="completed")
    _assert_llm_span(spans[3], finish_reason="completed")


@pytest.mark.asyncio
async def test_streaming_context_exceptions_after_parse_and_repeated_stream_parses(
    exporter: InMemorySpanExporter,
):
    import time
    from types import SimpleNamespace

    from opentelemetry.trace import StatusCode

    from neatlogs.anthropic import _instrument_async_raw_response as ant_async_raw
    from neatlogs.anthropic import _instrument_raw_response as ant_raw
    from neatlogs.openai import _instrument_async_raw_response as oai_async_raw
    from neatlogs.openai import _instrument_raw_response as oai_raw

    a_sync = wrap_anthropic_client(
        anthropic.Anthropic(
            api_key="x",
            http_client=httpx.Client(transport=httpx.MockTransport(_anthropic_transport_handler)),
        )
    )
    with pytest.raises(ValueError, match="body failed"):
        with a_sync.messages.with_streaming_response.create(
            model="claude-x", max_tokens=300, messages=[{"role": "user", "content": "hi"}]
        ) as resp:
            assert resp.parse().id == "msg_json_1"
            raise ValueError("body failed")

    o_async = wrap_async_openai_client(
        openai.AsyncOpenAI(
            api_key="x",
            http_client=httpx.AsyncClient(
                transport=httpx.MockTransport(_openai_chat_transport_handler)
            ),
        )
    )
    with pytest.raises(RuntimeError, match="async body failed"):
        async with o_async.chat.completions.with_streaming_response.create(
            model="gpt-4o", messages=[{"role": "user", "content": "hi"}], stream=True
        ) as a_resp:
            chunks = [c async for c in await a_resp.parse()]
            assert len(chunks) == 1
            raise RuntimeError("async body failed")

    spans = exporter.get_finished_spans()
    assert len(spans) == 2
    assert spans[0].status.status_code == StatusCode.ERROR
    assert "body failed" in (spans[0].status.description or "")
    assert spans[1].status.status_code == StatusCode.ERROR
    assert "async body failed" in (spans[1].status.description or "")

    # Verify repeated explicit stream parses and extraction-error recovery end span exactly once
    for raw_fn, async_raw_fn in ((ant_raw, ant_async_raw), (oai_raw, oai_async_raw)):
        ends = 0
        statuses: list = []
        fake_span = SimpleNamespace(
            end=lambda: None,
            set_status=lambda *a: statuses.append(a),
            set_attribute=lambda *_a: None,
            record_exception=lambda *_a: None,
        )

        def count_end():
            nonlocal ends
            ends += 1

        fake_span.end = count_end
        parse_calls: list = []
        r = SimpleNamespace(
            parse=lambda **kw: (parse_calls.append(kw), iter([1]))[1],
            close=lambda: None,
        )
        raw_fn(
            fake_span,
            r,
            time.perf_counter(),
            True,
            lambda s, _p, _t: s.end(),
            lambda s, *_a, **_k: s.end(),
            "raw",
        )
        assert parse_calls == []  # No eager parse
        s1 = r.parse(to=object)
        s2 = r.parse(to=object)
        assert s1 is s2
        list(s1)
        list(s2)
        r.close()
        assert ends == 1

        # Extraction error during parse leaves span open for close() to finalize
        ends = 0
        statuses.clear()
        r_err = SimpleNamespace(parse=lambda **_kw: b"abc", close=lambda: None)

        def failing_fin(_s, _p, _t):
            raise RuntimeError("finalize boom")

        await async_raw_fn(
            fake_span, r_err, time.perf_counter(), False, failing_fin, lambda *_a: None, "raw"
        )
        with pytest.raises(RuntimeError, match="finalize boom"):
            r_err.parse(to=bytes)
        assert ends == 0
        r_err.close()
        assert ends == 1


@pytest.mark.asyncio
async def test_with_raw_response_headers_only_without_parse_or_close(
    exporter: InMemorySpanExporter,
):
    import gc
    import weakref

    a_sync = wrap_anthropic_client(
        anthropic.Anthropic(
            api_key="x",
            http_client=httpx.Client(transport=httpx.MockTransport(_anthropic_transport_handler)),
        )
    )
    raw_a = a_sync.messages.with_raw_response.create(
        model="claude-x", max_tokens=300, messages=[{"role": "user", "content": "hi"}]
    )
    assert raw_a.headers.get("request-id") == "req_ant_json"
    ref_a = weakref.ref(raw_a)
    del raw_a
    gc.collect()
    assert ref_a() is None

    a_async = wrap_async_anthropic_client(
        anthropic.AsyncAnthropic(
            api_key="x",
            http_client=httpx.AsyncClient(
                transport=httpx.MockTransport(_anthropic_transport_handler)
            ),
        )
    )
    raw_a_async = await a_async.messages.with_raw_response.create(
        model="claude-x", max_tokens=300, messages=[{"role": "user", "content": "hi"}]
    )
    assert raw_a_async.headers.get("request-id") == "req_ant_json"
    ref_a_async = weakref.ref(raw_a_async)
    del raw_a_async
    gc.collect()
    assert ref_a_async() is None

    o_sync = wrap_openai_client(
        openai.OpenAI(
            api_key="x",
            http_client=httpx.Client(transport=httpx.MockTransport(_openai_chat_transport_handler)),
        )
    )
    raw_o = o_sync.chat.completions.with_raw_response.create(
        model="gpt-4o", messages=[{"role": "user", "content": "hi"}]
    )
    assert raw_o.headers.get("x-request-id") == "req_oai_json"
    ref_o = weakref.ref(raw_o)
    del raw_o
    gc.collect()
    assert ref_o() is None

    o_async = wrap_async_openai_client(
        openai.AsyncOpenAI(
            api_key="x",
            http_client=httpx.AsyncClient(
                transport=httpx.MockTransport(_openai_responses_transport_handler)
            ),
        )
    )
    raw_o_async = await o_async.responses.with_raw_response.create(model="gpt-4o", input="hi")
    assert raw_o_async.http_response.status_code == 200
    ref_o_async = weakref.ref(raw_o_async)
    del raw_o_async
    gc.collect()
    assert ref_o_async() is None

    spans = exporter.get_finished_spans()
    assert len(spans) == 4
    _assert_llm_span(spans[0], finish_reason="end_turn")
    _assert_llm_span(spans[1], finish_reason="end_turn")
    _assert_llm_span(spans[2], finish_reason="stop")
    _assert_llm_span(spans[3], finish_reason="completed")
