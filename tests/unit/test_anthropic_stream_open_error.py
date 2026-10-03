"""messages.stream() that fails while opening must export an ERROR span."""

import asyncio

import anthropic
import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from neatlogs import _wrap_utils as w
from neatlogs.anthropic import wrap_anthropic_client, wrap_async_anthropic_client

try:
    import httpx2 as httpx
except ImportError:  # older anthropic releases use httpx
    import httpx

KW = dict(model="claude-x", max_tokens=5, messages=[{"role": "user", "content": "hi"}])


def _unauthorized(request):
    return httpx.Response(
        401,
        json={
            "type": "error",
            "error": {"type": "authentication_error", "message": "bad key"},
        },
    )


@pytest.fixture
def exporter():
    exp = InMemorySpanExporter()
    prov = TracerProvider()
    prov.add_span_processor(SimpleSpanProcessor(exp))
    w.set_neatlogs_provider(prov)
    yield exp
    exp.clear()


def test_sync_stream_open_failure_exports_error_span(exporter):
    client = wrap_anthropic_client(
        anthropic.Anthropic(
            api_key="x",
            max_retries=0,
            http_client=httpx.Client(transport=httpx.MockTransport(_unauthorized)),
        )
    )
    with pytest.raises(anthropic.AuthenticationError):
        with client.messages.stream(**KW) as stream:
            for _ in stream:
                pass
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].status.status_code.name == "ERROR"


def test_async_stream_open_failure_exports_error_span(exporter):
    client = wrap_async_anthropic_client(
        anthropic.AsyncAnthropic(
            api_key="x",
            max_retries=0,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(_unauthorized)),
        )
    )

    async def run():
        async with client.messages.stream(**KW) as stream:
            async for _ in stream:
                pass

    with pytest.raises(anthropic.AuthenticationError):
        asyncio.run(run())
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].status.status_code.name == "ERROR"
