import json
import logging

import httpx
import pytest

openai = pytest.importorskip("openai")

from neatlogs.openai import wrap_async_openai_client, wrap_openai_client  # noqa: E402

_CHUNKS = [
    {
        "id": "c1",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "gpt-4o",
        "choices": [
            {"index": 0, "delta": {"role": "assistant", "content": "hi"}, "finish_reason": None}
        ],
    },
    {
        "id": "c1",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "gpt-4o",
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    },
]


def _handler(_request):
    body = "".join(f"data: {json.dumps(c)}\n\n" for c in _CHUNKS) + "data: [DONE]\n\n"
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body.encode())


def _param_attrs(exporter):
    names = ("temperature", "top_p", "max_tokens", "frequency_penalty", "presence_penalty")
    return {
        k: v
        for span in exporter.get_finished_spans()
        for k, v in span.attributes.items()
        if k.startswith("neatlogs.llm.") and k.rsplit(".", 1)[-1] in names
    }


def _init(tracer_provider):
    import neatlogs

    neatlogs.init(
        api_key="test",
        disable_export=True,
        tracer_provider=tracer_provider,
        register_shutdown_handlers=False,
    )


def test_stream_helper_skips_omit_sentinels(tracer_provider, in_memory_span_exporter, caplog):
    _init(tracer_provider)
    client = wrap_openai_client(
        openai.OpenAI(
            api_key="x", http_client=httpx.Client(transport=httpx.MockTransport(_handler))
        )
    )
    with caplog.at_level(logging.WARNING, logger="opentelemetry"):
        with client.chat.completions.stream(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.3,
            max_tokens=50,
        ) as stream:
            for _ in stream:
                pass

    assert "Invalid type" not in caplog.text
    assert _param_attrs(in_memory_span_exporter) == {
        "neatlogs.llm.temperature": 0.3,
        "neatlogs.llm.max_tokens": 50,
    }


@pytest.mark.asyncio
async def test_async_stream_helper_skips_omit_sentinels(
    tracer_provider, in_memory_span_exporter, caplog
):
    _init(tracer_provider)
    client = wrap_async_openai_client(
        openai.AsyncOpenAI(
            api_key="x", http_client=httpx.AsyncClient(transport=httpx.MockTransport(_handler))
        )
    )
    with caplog.at_level(logging.WARNING, logger="opentelemetry"):
        async with client.chat.completions.stream(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
            top_p=0.9,
        ) as stream:
            async for _ in stream:
                pass

    assert "Invalid type" not in caplog.text
    assert _param_attrs(in_memory_span_exporter) == {"neatlogs.llm.top_p": 0.9}
