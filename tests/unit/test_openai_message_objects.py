"""Chat messages passed as SDK objects (not dicts) must not break the wrapped client."""

import json

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

openai = pytest.importorskip("openai")

from neatlogs import _wrap_utils as w  # noqa: E402
from neatlogs.openai import wrap_async_openai_client, wrap_openai_client  # noqa: E402

httpx = pytest.importorskip("httpx")


def _handler(calls):
    def handle(request):
        calls.append(json.loads(request.content))
        first = len(calls) == 1
        message = (
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_w", "arguments": "{}"},
                    }
                ],
            }
            if first
            else {"role": "assistant", "content": "sunny"}
        )
        return httpx.Response(
            200,
            json={
                "id": "c",
                "object": "chat.completion",
                "created": 1,
                "model": "m",
                "choices": [
                    {
                        "index": 0,
                        "message": message,
                        "finish_reason": "tool_calls" if first else "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
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


def _history(first_message):
    return [
        {"role": "user", "content": "weather?"},
        first_message,
        {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
    ]


def test_sync_accepts_message_object_in_history(exporter):
    calls = []
    client = openai.OpenAI(
        api_key="k", http_client=httpx.Client(transport=httpx.MockTransport(_handler(calls)))
    )
    wrap_openai_client(client)
    first = client.chat.completions.create(model="m", messages=[{"role": "user", "content": "x"}])
    client.chat.completions.create(model="m", messages=_history(first.choices[0].message))

    assert len(calls) == 2 and len(calls[1]["messages"]) == 3
    spans = [s for s in exporter.get_finished_spans() if s.name.startswith("openai")]
    assert len(spans) == 2
    assert spans[1].attributes["neatlogs.llm.input_messages.1.role"] == "assistant"
    assert spans[1].attributes["neatlogs.llm.input_messages.2.tool_call_id"] == "call_1"


@pytest.mark.asyncio
async def test_async_accepts_message_object_in_history(exporter):
    calls = []
    client = openai.AsyncOpenAI(
        api_key="k",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(_handler(calls))),
    )
    wrap_async_openai_client(client)
    first = await client.chat.completions.create(
        model="m", messages=[{"role": "user", "content": "x"}]
    )
    await client.chat.completions.create(model="m", messages=_history(first.choices[0].message))

    assert len(calls) == 2 and len(calls[1]["messages"]) == 3
    spans = [s for s in exporter.get_finished_spans() if s.name.startswith("openai")]
    assert len(spans) == 2
    assert spans[1].attributes["neatlogs.llm.input_messages.2.tool_call_id"] == "call_1"


def test_azure_sync_accepts_message_object_in_history(exporter):
    from neatlogs.azure_openai import wrap_azure_openai_client

    calls = []
    client = openai.AzureOpenAI(
        api_key="k",
        api_version="2024-06-01",
        azure_endpoint="https://example.openai.azure.com",
        http_client=httpx.Client(transport=httpx.MockTransport(_handler(calls))),
    )
    wrap_azure_openai_client(client)
    first = client.chat.completions.create(model="m", messages=[{"role": "user", "content": "x"}])
    client.chat.completions.create(model="m", messages=_history(first.choices[0].message))

    assert len(calls) == 2 and len(calls[1]["messages"]) == 3
    spans = [s for s in exporter.get_finished_spans() if s.name.startswith("azure_openai")]
    assert len(spans) == 2
    assert spans[1].attributes["neatlogs.llm.input_messages.1.role"] == "assistant"
    assert spans[1].attributes["neatlogs.llm.input_messages.2.tool_call_id"] == "call_1"


@pytest.mark.asyncio
async def test_azure_async_accepts_message_object_in_history(exporter):
    from neatlogs.azure_openai import wrap_async_azure_openai_client

    calls = []
    client = openai.AsyncAzureOpenAI(
        api_key="k",
        api_version="2024-06-01",
        azure_endpoint="https://example.openai.azure.com",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(_handler(calls))),
    )
    wrap_async_azure_openai_client(client)
    first = await client.chat.completions.create(
        model="m", messages=[{"role": "user", "content": "x"}]
    )
    await client.chat.completions.create(model="m", messages=_history(first.choices[0].message))

    assert len(calls) == 2 and len(calls[1]["messages"]) == 3
    spans = [s for s in exporter.get_finished_spans() if s.name.startswith("azure_openai")]
    assert len(spans) == 2
    assert spans[1].attributes["neatlogs.llm.input_messages.2.tool_call_id"] == "call_1"
