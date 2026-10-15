"""
Tests for the Supermemory wrapper (neatlogs.supermemory).

The client is a fake with the same call shape as supermemory.Supermemory, so
nothing here talks to the real API.
"""

import asyncio
from types import SimpleNamespace

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor


def _setup_tracer(exporter):
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    import neatlogs._wrap_utils as _wu

    _wu._wrapper_tracer = None
    return provider


def _span(exporter, name):
    spans = [s for s in exporter.get_finished_spans() if s.name == name]
    assert len(spans) == 1, [s.name for s in exporter.get_finished_spans()]
    return spans[0]


def test_search_is_a_retriever_span(in_memory_span_exporter):
    _setup_tracer(in_memory_span_exporter)
    from neatlogs.supermemory import wrap_supermemory_client

    def search(namespace, *, query, limit=None, **_):
        return SimpleNamespace(
            results=[SimpleNamespace(memory="likes tea"), {"chunk": "lives in pune"}]
        )

    client = wrap_supermemory_client(SimpleNamespace(search=search))
    out = client.search("user-1", query="what do they like?", limit=5)

    assert len(out.results) == 2
    attrs = _span(in_memory_span_exporter, "supermemory.search").attributes
    assert attrs["neatlogs.span.kind"] == "retriever"
    assert attrs["neatlogs.retriever.query"] == "what do they like?"
    assert attrs["neatlogs.retriever.top_k"] == 5
    assert attrs["supermemory.namespace"] == "user-1"
    assert attrs["neatlogs.retriever.document_count"] == 2
    assert attrs["neatlogs.retriever.documents.0.content"] == "likes tea"
    assert attrs["neatlogs.retriever.documents.1.content"] == "lives in pune"


def test_add_is_a_tool_span(in_memory_span_exporter):
    _setup_tracer(in_memory_span_exporter)
    from neatlogs.supermemory import wrap_supermemory_client

    client = wrap_supermemory_client(
        SimpleNamespace(add=lambda namespace, *, content, **_: SimpleNamespace(id="doc_1"))
    )
    client.add("user-1", content="prefers dark mode")

    attrs = _span(in_memory_span_exporter, "supermemory.add").attributes
    assert attrs["neatlogs.span.kind"] == "tool"
    assert attrs["neatlogs.tool.name"] == "supermemory.add"
    assert "prefers dark mode" in attrs["neatlogs.tool.parameters"]
    assert attrs["supermemory.document_id"] == "doc_1"


def test_profile_is_a_retriever_span(in_memory_span_exporter):
    _setup_tracer(in_memory_span_exporter)
    from neatlogs.supermemory import wrap_supermemory_client

    client = wrap_supermemory_client(
        SimpleNamespace(profile=lambda namespace, **_: SimpleNamespace(profile={"static": ["a"]}))
    )
    client.profile("user-1")

    attrs = _span(in_memory_span_exporter, "supermemory.profile").attributes
    assert attrs["neatlogs.span.kind"] == "retriever"
    assert "static" in attrs["neatlogs.retriever.profile"]


def test_async_client_is_awaited_and_traced(in_memory_span_exporter):
    _setup_tracer(in_memory_span_exporter)
    from neatlogs.supermemory import wrap_supermemory_client

    async def search(namespace, *, query, **_):
        return SimpleNamespace(results=[SimpleNamespace(memory="async hit")])

    client = wrap_supermemory_client(SimpleNamespace(search=search))
    out = asyncio.run(client.search("user-1", query="q"))

    assert out.results[0].memory == "async hit"
    attrs = _span(in_memory_span_exporter, "supermemory.search").attributes
    assert attrs["neatlogs.retriever.documents.0.content"] == "async hit"


def test_errors_are_recorded_and_reraised(in_memory_span_exporter):
    _setup_tracer(in_memory_span_exporter)
    from neatlogs.supermemory import wrap_supermemory_client

    def search(namespace, *, query, **_):
        raise RuntimeError("rate limited")

    client = wrap_supermemory_client(SimpleNamespace(search=search))
    with pytest.raises(RuntimeError, match="rate limited"):
        client.search("user-1", query="q")

    span = _span(in_memory_span_exporter, "supermemory.search")
    assert span.status.status_code == trace.StatusCode.ERROR


def test_wrap_is_idempotent(in_memory_span_exporter):
    _setup_tracer(in_memory_span_exporter)
    from neatlogs.supermemory import wrap_supermemory_client

    client = SimpleNamespace(search=lambda namespace, *, query, **_: SimpleNamespace(results=[]))
    wrap_supermemory_client(client)
    first = client.search
    wrap_supermemory_client(client)
    assert client.search is first
    client.search("user-1", query="q")
    _span(in_memory_span_exporter, "supermemory.search")


def test_real_sdk_client_with_mocked_transport(in_memory_span_exporter):
    httpx = pytest.importorskip("httpx")
    supermemory = pytest.importorskip("supermemory")
    _setup_tracer(in_memory_span_exporter)
    import neatlogs

    def handler(request):
        return httpx.Response(
            200,
            json={"results": [{"id": "m1", "memory": "likes tea", "similarity": 0.9}], "total": 1},
        )

    client = supermemory.Supermemory(
        api_key="test-key",
        base_url="https://supermemory.invalid",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    client = neatlogs.wrap(client)
    client.search("user-1", query="what do they like?")

    attrs = _span(in_memory_span_exporter, "supermemory.search").attributes
    assert attrs["neatlogs.retriever.query"] == "what do they like?"
    assert attrs["neatlogs.retriever.document_count"] == 1
    assert attrs["neatlogs.retriever.documents.0.content"] == "likes tea"
