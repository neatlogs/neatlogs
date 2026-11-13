import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

import neatlogs
from neatlogs._embedding import patch_embeddings
from neatlogs._wrap_utils import set_neatlogs_provider


@pytest.mark.asyncio
@pytest.mark.parametrize("azure", [False, True])
@pytest.mark.parametrize("subclass", [False, True])
async def test_wrap_langchain_embedding_resources_records_actual_usage(
    azure, subclass, tracer_provider, in_memory_span_exporter
):
    set_neatlogs_provider(tracer_provider)
    base = type(
        "AzureOpenAIEmbeddings" if azure else "OpenAIEmbeddings",
        (),
        {
            "__module__": (
                "langchain_openai.embeddings.azure" if azure else "langchain_openai.embeddings.base"
            )
        },
    )
    embedding = (type("CustomEmbeddings", (base,), {}) if subclass else base)()
    response = SimpleNamespace(
        model="text-embedding-3-large",
        usage=SimpleNamespace(prompt_tokens=7, total_tokens=7),
        data=[SimpleNamespace(embedding=[0.1, 0.2, 0.3])],
    )
    sync_create = Mock(return_value=response)
    async_create = AsyncMock(return_value=response)
    chat = Mock()
    embedding.client = SimpleNamespace(create=sync_create)
    embedding.async_client = SimpleNamespace(create=async_create)
    embedding.chat = SimpleNamespace(completions=SimpleNamespace(create=chat))

    import openai

    constructors = [
        openai.OpenAI.__init__,
        openai.AsyncOpenAI.__init__,
        openai.AzureOpenAI.__init__,
        openai.AsyncAzureOpenAI.__init__,
    ]
    assert neatlogs.wrap(embedding) is embedding
    assert constructors == [
        openai.OpenAI.__init__,
        openai.AsyncOpenAI.__init__,
        openai.AzureOpenAI.__init__,
        openai.AsyncAzureOpenAI.__init__,
    ]
    wrapped_sync = embedding.client.create
    wrapped_async = embedding.async_client.create
    assert neatlogs.wrap(embedding) is embedding
    assert embedding.client.create is wrapped_sync
    assert embedding.async_client.create is wrapped_async
    assert embedding.client.create(input="returns", model="deployment-alias") is response
    assert (
        await embedding.async_client.create(input="delivery", model="deployment-alias") is response
    )
    assert embedding.chat.completions.create is chat
    sync_create.assert_called_once_with(input="returns", model="deployment-alias")
    async_create.assert_awaited_once_with(input="delivery", model="deployment-alias")

    spans = [
        item
        for item in in_memory_span_exporter.get_finished_spans()
        if item.attributes.get("neatlogs.span.kind") == "embedding"
    ]
    assert len(spans) == 2
    for span in spans:
        assert span.attributes["neatlogs.embedding.model_name"] == response.model
        assert span.attributes["neatlogs.embedding.token_count"] == 7
        assert span.attributes["neatlogs.embedding.dimensions"] == 3
        assert span.attributes["neatlogs.embedding.count"] == 1
        assert "vector" not in str(span.attributes)


@pytest.mark.parametrize(
    "flag", ["_neatlogs_embedding_patched", "_neatlogs_patched", "_neatlogs_azure_patched"]
)
def test_embedding_resource_respects_existing_instrumentation(flag):
    create = Mock()
    resource = SimpleNamespace(create=create, **{flag: True})
    patch_embeddings(resource, provider="azure")
    assert resource.create is create


def test_embedding_telemetry_failure_does_not_retry_provider(monkeypatch):
    from neatlogs import _embedding

    monkeypatch.setattr(
        _embedding, "get_provider_tracer", Mock(side_effect=RuntimeError("tracer unavailable"))
    )
    result = object()
    create = Mock(return_value=result)
    resource = SimpleNamespace(create=create)
    patch_embeddings(resource, provider="azure")
    assert resource.create(input="returns") is result
    create.assert_called_once_with(input="returns")
    error = ValueError("provider failure")
    create.side_effect = error
    with pytest.raises(ValueError) as raised:
        resource.create(input="delivery")
    assert raised.value is error
    assert create.call_count == 2


@pytest.mark.asyncio
async def test_embedding_suppression_and_cancellation_preserve_async_provider(
    tracer_provider, in_memory_span_exporter
):
    from opentelemetry import context as context_api

    set_neatlogs_provider(tracer_provider)
    result = object()
    create = AsyncMock(return_value=result)
    resource = SimpleNamespace(create=create)
    patch_embeddings(resource, provider="azure", sync=False)
    token = context_api.attach(context_api.set_value("suppress_instrumentation", True))
    try:
        assert await resource.create(input="returns") is result
    finally:
        context_api.detach(token)
    assert not in_memory_span_exporter.get_finished_spans()
    error = asyncio.CancelledError()
    create.side_effect = error
    with pytest.raises(asyncio.CancelledError) as raised:
        await resource.create(input="delivery")
    assert raised.value is error
    assert create.await_count == 2
    assert (
        len(
            [
                span
                for span in in_memory_span_exporter.get_finished_spans()
                if span.attributes.get("neatlogs.span.kind") == "embedding"
            ]
        )
        == 1
    )
