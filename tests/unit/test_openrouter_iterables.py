import asyncio
import json

import httpx
import pytest

import neatlogs.openrouter as integration


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("field", ["messages", "tools"])
@pytest.mark.parametrize("shape", ["generator", "list", "tuple"])
def test_native_chat_keeps_iterable_input(
    monkeypatch, tracer_provider, in_memory_span_exporter, asynchronous, field, shape, streaming
):
    sdk = pytest.importorskip("openrouter")
    requests = []
    values = (
        [{"role": "user", "content": "keep this prompt"}]
        if field == "messages"
        else [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "find a record",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
    )
    consumed = []

    def once():
        for value in values:
            consumed.append(value)
            yield value

    supplied = once() if shape == "generator" else (tuple(values) if shape == "tuple" else values)
    kwargs = {
        "model": "openai/test",
        "stream": streaming,
        "messages": [{"role": "user", "content": "hi"}],
        field: supplied,
    }

    def respond(request):
        requests.append(json.loads(request.content))
        if streaming:
            chunk = {
                "id": "offline",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "openai/test",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
            }
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                text="data: " + json.dumps(chunk) + "\n\ndata: [DONE]\n\n",
            )
        return httpx.Response(
            200,
            json={
                "id": "offline",
                "object": "chat.completion",
                "created": 1,
                "model": "openai/test",
                "system_fingerprint": "offline",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "ok"},
                    }
                ],
            },
        )

    monkeypatch.setattr(
        integration, "get_provider_tracer", lambda: tracer_provider.get_tracer("test")
    )
    if asynchronous:

        async def run():
            async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as transport:
                client = sdk.OpenRouter(api_key="offline", async_client=transport)
                integration.wrap_openrouter_client(client)
                result = await client.chat.send_async(**kwargs)
                return [chunk async for chunk in result] if streaming else result

        result = asyncio.run(run())
    else:
        with httpx.Client(transport=httpx.MockTransport(respond)) as transport:
            client = sdk.OpenRouter(api_key="offline", client=transport)
            integration.wrap_openrouter_client(client)
            result = client.chat.send(**kwargs)
            if streaming:
                result = list(result)
    if streaming:
        assert result[0].choices[0].delta.content == "ok"
    else:
        assert result.choices[0].message.content == "ok"
    assert len(requests) == 1
    assert requests[0][field] == values
    if shape == "generator":
        assert consumed == values
    spans = in_memory_span_exporter.get_finished_spans()
    span = next(s for s in spans if s.name == "openrouter.chat.send")
    attr = (
        "neatlogs.llm.input_messages.0.content"
        if field == "messages"
        else "neatlogs.llm.tools.0.name"
    )
    assert span.attributes[attr] == ("keep this prompt" if field == "messages" else "lookup")


@pytest.mark.parametrize("shape", ["generator", "list", "tuple"])
@pytest.mark.parametrize("wrapped", [False, True])
def test_native_rerank_keeps_iterable_documents(
    monkeypatch, tracer_provider, in_memory_span_exporter, shape, wrapped
):
    sdk = pytest.importorskip("openrouter")
    values = ["first document", "second document"]
    requests = []
    consumed = []

    def once():
        for value in values:
            consumed.append(value)
            yield value

    supplied = once() if shape == "generator" else (tuple(values) if shape == "tuple" else values)

    def respond(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={"model": "test/reranker", "results": []})

    monkeypatch.setattr(
        integration, "get_provider_tracer", lambda: tracer_provider.get_tracer("test")
    )
    with httpx.Client(transport=httpx.MockTransport(respond)) as transport:
        client = sdk.OpenRouter(api_key="offline", client=transport)
        if wrapped:
            integration.wrap_openrouter_client(client)
        result = client.rerank.rerank(model="test/reranker", query="first", documents=supplied)
    assert result.results == []
    assert len(requests) == 1
    assert requests[0]["documents"] == values
    if shape == "generator":
        assert consumed == values
    spans = in_memory_span_exporter.get_finished_spans()
    if wrapped:
        span = next(s for s in spans if s.name == "openrouter.rerank.rerank")
        assert span.attributes["neatlogs.reranker.input_documents.0"] == values[0]
        assert span.attributes["neatlogs.reranker.input_documents.1"] == values[1]
    else:
        assert spans == ()
