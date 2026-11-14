import asyncio
import json

import httpx
import pytest
from openai import AsyncAzureOpenAI, AzureOpenAI

import neatlogs.azure_openai as integration
import neatlogs.openai as openai_integration


@pytest.fixture(autouse=True)
def restore_azure_patch():
    openai_patched = openai_integration._PATCHED
    openai_integration._unpatch_openai_module()
    was_patched = integration._PATCHED
    integration._unpatch_azure_openai_module()
    try:
        yield
    finally:
        if was_patched:
            integration._patch_azure_openai_module()
        if openai_patched:
            openai_integration._patch_openai_module()


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("field", ["messages", "tools"])
@pytest.mark.parametrize("shape", ["generator", "list", "tuple"])
@pytest.mark.parametrize("wrapped", [False, True])
def test_native_azure_chat_keeps_iterable_inputs(
    monkeypatch,
    tracer_provider,
    in_memory_span_exporter,
    asynchronous,
    streaming,
    field,
    shape,
    wrapped,
):
    values = (
        [{"role": "user", "content": "preserve me"}]
        if field == "messages"
        else [
            {"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}
        ]
    )
    consumed = []
    requests = []

    def once():
        for value in values:
            consumed.append(value)
            yield value

    supplied = once() if shape == "generator" else (tuple(values) if shape == "tuple" else values)
    kwargs = {
        "model": "deployment",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": streaming,
        field: supplied,
    }

    def respond(request):
        requests.append(json.loads(request.content))
        if streaming:
            chunk = {
                "id": "offline",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "deployment",
                "choices": [{"index": 0, "delta": {"content": "ok"}, "finish_reason": "stop"}],
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
                "model": "deployment",
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
    config = {
        "api_key": "offline",
        "azure_endpoint": "https://offline.invalid",
        "api_version": "2024-10-21",
        "max_retries": 0,
    }
    if asynchronous:

        async def run():
            async with AsyncAzureOpenAI(
                **config, http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond))
            ) as client:
                if wrapped:
                    integration.wrap_async_azure_openai_client(client)
                result = await client.chat.completions.create(**kwargs)
                return [chunk async for chunk in result] if streaming else result

        result = asyncio.run(run())
    else:
        with AzureOpenAI(
            **config, http_client=httpx.Client(transport=httpx.MockTransport(respond))
        ) as client:
            if wrapped:
                integration.wrap_azure_openai_client(client)
            result = client.chat.completions.create(**kwargs)
            if streaming:
                result = list(result)
    assert (
        result[0].choices[0].delta.content if streaming else result.choices[0].message.content
    ) == "ok"
    assert len(requests) == 1
    assert requests[0][field] == values
    if shape == "generator":
        assert consumed == values
    if wrapped:
        span = next(
            s
            for s in in_memory_span_exporter.get_finished_spans()
            if s.name == "azure_openai.chat.completions.create"
        )
        attr = (
            "neatlogs.llm.input_messages.0.content"
            if field == "messages"
            else "neatlogs.llm.tools.0.name"
        )
        assert span.attributes[attr] == ("preserve me" if field == "messages" else "lookup")
