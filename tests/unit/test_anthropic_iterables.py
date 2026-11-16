import asyncio
import json

import httpx
import pytest

anthropic = pytest.importorskip("anthropic")

from anthropic import Anthropic, AsyncAnthropic  # noqa: E402

import neatlogs.anthropic as integration  # noqa: E402


@pytest.fixture(autouse=True)
def restore_patch():
    was_patched = integration._PATCHED
    integration._unpatch_anthropic_module()
    try:
        yield
    finally:
        if was_patched:
            integration._patch_anthropic_module()


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("field", ["messages", "tools"])
@pytest.mark.parametrize("mode", ["create", "create_stream", "stream"])
@pytest.mark.parametrize("shape", ["generator", "list", "tuple"])
@pytest.mark.parametrize("wrapped", [False, True])
def test_native_message_inputs(
    monkeypatch, tracer_provider, in_memory_span_exporter, asynchronous, field, shape, wrapped, mode
):
    values = (
        [{"role": "user", "content": "preserve me"}]
        if field == "messages"
        else [{"name": "lookup", "input_schema": {"type": "object"}}]
    )
    consumed, requests = [], []

    def once():
        for value in values:
            consumed.append(value)
            yield value

    supplied = once() if shape == "generator" else tuple(values) if shape == "tuple" else values
    kwargs = dict(model="claude-test", max_tokens=5, messages=[{"role": "user", "content": "hi"}])
    kwargs[field] = supplied

    def event(name, data):
        return f"event: {name}\ndata: {json.dumps(data)}\n\n"

    def respond(request):
        requests.append(json.loads(request.content))
        if mode != "create":
            message = dict(
                id="offline",
                type="message",
                role="assistant",
                model="claude-test",
                content=[],
                stop_reason=None,
                stop_sequence=None,
                usage={"input_tokens": 1, "output_tokens": 0},
            )
            body = event("message_start", {"type": "message_start", "message": message})
            body += event(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                },
            )
            body += event(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": "ok"},
                },
            )
            body += event("content_block_stop", {"type": "content_block_stop", "index": 0})
            body += event(
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                    "usage": {"output_tokens": 1},
                },
            )
            body += event("message_stop", {"type": "message_stop"})
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=body)
        return httpx.Response(
            200,
            json=dict(
                id="offline",
                type="message",
                role="assistant",
                model="claude-test",
                content=[{"type": "text", "text": "ok"}],
                stop_reason="end_turn",
                stop_sequence=None,
                usage={"input_tokens": 1, "output_tokens": 1},
            ),
        )

    monkeypatch.setattr(
        integration, "get_provider_tracer", lambda: tracer_provider.get_tracer("test")
    )
    if asynchronous:

        async def run():
            async with AsyncAnthropic(
                api_key="offline",
                max_retries=0,
                http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
            ) as client:
                if wrapped:
                    integration.wrap_async_anthropic_client(client)
                resource = client.messages
                if mode == "stream":
                    async with resource.stream(**kwargs) as stream:
                        return await stream.get_final_text()
                result = await resource.create(
                    **kwargs, **({"stream": True} if mode == "create_stream" else {})
                )
                if mode == "create_stream":
                    chunks = [chunk async for chunk in result]
                    return "".join(c.delta.text for c in chunks if c.type == "content_block_delta")
                return result.content[0].text

        result = asyncio.run(run())
    else:
        with Anthropic(
            api_key="offline",
            max_retries=0,
            http_client=httpx.Client(transport=httpx.MockTransport(respond)),
        ) as client:
            if wrapped:
                integration.wrap_anthropic_client(client)
            resource = client.messages
            if mode == "stream":
                with resource.stream(**kwargs) as stream:
                    result = stream.get_final_text()
            else:
                result = resource.create(
                    **kwargs, **({"stream": True} if mode == "create_stream" else {})
                )
                result = (
                    "".join(c.delta.text for c in result if c.type == "content_block_delta")
                    if mode == "create_stream"
                    else result.content[0].text
                )
    assert result == "ok"
    assert len(requests) == 1
    assert requests[0][field] == values
    if shape == "generator":
        assert consumed == values
    if wrapped:
        span = next(
            s
            for s in in_memory_span_exporter.get_finished_spans()
            if s.name == "anthropic.messages.create"
        )
        attr = (
            "neatlogs.llm.input_messages.0.content"
            if field == "messages"
            else "neatlogs.llm.tools.0.name"
        )
        assert span.attributes[attr] == ("preserve me" if field == "messages" else "lookup")
