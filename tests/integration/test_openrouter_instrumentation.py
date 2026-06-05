"""
Tests for OpenRouter Instrumentation
====================================
Verifies the neatlogs.openrouter wrapper traces the Chat Completions, Responses,
Embeddings, and Rerank APIs with provider="openrouter", system=<model vendor>,
and canonical neatlogs.* attrs — including the invocation_parameters blob the UI
reads for model settings.
"""

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


def _fake_client(*, chat=None, responses=None, embeddings=None, rerank=None):
    """Minimal fake openrouter.OpenRouter client exposing the sub-SDK surfaces."""
    chat_obj = SimpleNamespace(send=chat) if chat is not None else SimpleNamespace()
    beta = SimpleNamespace(
        responses=SimpleNamespace(send=responses) if responses is not None else SimpleNamespace()
    )
    emb = SimpleNamespace(generate=embeddings) if embeddings is not None else SimpleNamespace()
    rer = SimpleNamespace(rerank=rerank) if rerank is not None else SimpleNamespace()
    return SimpleNamespace(chat=chat_obj, beta=beta, embeddings=emb, rerank=rer)


class TestOpenRouterChat:
    def test_chat_send_traces_io_tokens_and_params(self, in_memory_span_exporter):
        _setup_tracer(in_memory_span_exporter)
        from neatlogs.openrouter import wrap_openrouter_client

        def send(**kwargs):
            return SimpleNamespace(
                id="gen-123",
                model="openai/gpt-4o-mini",
                choices=[
                    SimpleNamespace(
                        index=0,
                        finish_reason="stop",
                        message=SimpleNamespace(role="assistant", content="Hello from OpenRouter", tool_calls=None),
                    )
                ],
                usage=SimpleNamespace(prompt_tokens=16, completion_tokens=8, total_tokens=24),
            )

        client = _fake_client(chat=send)
        wrap_openrouter_client(client)

        client.chat.send(
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": "Hi"}],
            temperature=0.3,
            top_p=0.9,
            max_tokens=256,
        )

        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) == 1
        attrs = spans[0].attributes
        assert attrs.get("neatlogs.span.kind") == "llm"
        assert attrs.get("neatlogs.llm.provider") == "openrouter"
        assert attrs.get("neatlogs.llm.system") == "openai"
        assert attrs.get("neatlogs.llm.model_name") == "openai/gpt-4o-mini"
        assert attrs.get("neatlogs.llm.input_messages.0.role") == "user"
        assert attrs.get("neatlogs.llm.input_messages.0.content") == "Hi"
        assert attrs.get("neatlogs.llm.output_messages.0.content") == "Hello from OpenRouter"
        assert attrs.get("output.value") == "Hello from OpenRouter"
        assert attrs.get("neatlogs.llm.token_count.prompt") == 16
        assert attrs.get("neatlogs.llm.token_count.completion") == 8
        assert attrs.get("neatlogs.llm.token_count.total") == 24
        assert attrs.get("neatlogs.llm.finish_reason") == "stop"
        assert attrs.get("neatlogs.llm.response_id") == "gen-123"
        # Individual params AND the invocation_parameters blob (UI model_settings).
        assert attrs.get("neatlogs.llm.temperature") == 0.3
        assert attrs.get("neatlogs.llm.top_p") == 0.9
        assert attrs.get("neatlogs.llm.max_tokens") == 256
        import json
        blob = json.loads(attrs.get("neatlogs.llm.invocation_parameters"))
        assert blob == {"temperature": 0.3, "top_p": 0.9, "max_tokens": 256}

    def test_chat_send_streaming(self, in_memory_span_exporter):
        _setup_tracer(in_memory_span_exporter)
        from neatlogs.openrouter import wrap_openrouter_client

        def _chunk(text=None, finish=None, usage=None):
            delta = SimpleNamespace(content=text, tool_calls=None)
            choice = SimpleNamespace(index=0, delta=delta, finish_reason=finish)
            return SimpleNamespace(model="openai/gpt-4o-mini", choices=[choice], usage=usage)

        def send(**kwargs):
            return iter([
                _chunk("Hello"),
                _chunk(" world"),
                _chunk(finish="stop", usage=SimpleNamespace(prompt_tokens=5, completion_tokens=3, total_tokens=8)),
            ])

        client = _fake_client(chat=send)
        wrap_openrouter_client(client)

        stream = client.chat.send(
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": "Hi"}],
            temperature=0.5,
            stream=True,
        )
        collected = "".join((getattr(c.choices[0].delta, "content", "") or "") for c in stream)
        assert collected == "Hello world"

        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) == 1
        attrs = spans[0].attributes
        assert attrs.get("neatlogs.llm.is_streaming") is True
        assert attrs.get("neatlogs.llm.output_messages.0.content") == "Hello world"
        assert attrs.get("neatlogs.llm.token_count.prompt") == 5
        assert attrs.get("neatlogs.llm.token_count.completion") == 3
        assert attrs.get("neatlogs.llm.finish_reason") == "stop"
        import json
        assert json.loads(attrs.get("neatlogs.llm.invocation_parameters")) == {"temperature": 0.5}


class TestOpenRouterResponses:
    def test_responses_send(self, in_memory_span_exporter):
        _setup_tracer(in_memory_span_exporter)
        from neatlogs.openrouter import wrap_openrouter_client

        def send(**kwargs):
            return SimpleNamespace(
                id="resp-1",
                model="openai/gpt-4o-mini",
                status="completed",
                output_text="Responses output",
                output=[],
                usage=SimpleNamespace(input_tokens=10, output_tokens=4, total_tokens=14),
            )

        client = _fake_client(responses=send)
        wrap_openrouter_client(client)
        client.beta.responses.send(model="openai/gpt-4o-mini", input="Hi", max_output_tokens=128)

        attrs = in_memory_span_exporter.get_finished_spans()[0].attributes
        assert attrs.get("neatlogs.span.kind") == "llm"
        assert attrs.get("neatlogs.llm.provider") == "openrouter"
        assert attrs.get("neatlogs.llm.input_messages.0.content") == "Hi"
        assert attrs.get("neatlogs.llm.output_messages.0.content") == "Responses output"
        assert attrs.get("neatlogs.llm.token_count.prompt") == 10
        assert attrs.get("neatlogs.llm.token_count.completion") == 4
        # max_output_tokens normalizes to the max_tokens individual attr + blob.
        assert attrs.get("neatlogs.llm.max_tokens") == 128
        import json
        assert json.loads(attrs.get("neatlogs.llm.invocation_parameters")) == {"max_output_tokens": 128}


class TestOpenRouterEmbeddingsRerank:
    def test_embeddings(self, in_memory_span_exporter):
        _setup_tracer(in_memory_span_exporter)
        from neatlogs.openrouter import wrap_openrouter_client

        def generate(**kwargs):
            return SimpleNamespace(
                data=[SimpleNamespace(embedding=[0.1, 0.2, 0.3])],
                usage=SimpleNamespace(prompt_tokens=4, total_tokens=4),
            )

        client = _fake_client(embeddings=generate)
        wrap_openrouter_client(client)
        client.embeddings.generate(model="openai/text-embedding-3-small", input="hello")

        attrs = in_memory_span_exporter.get_finished_spans()[0].attributes
        assert attrs.get("neatlogs.span.kind") == "embedding"
        assert attrs.get("neatlogs.embedding.model_name") == "openai/text-embedding-3-small"
        assert attrs.get("neatlogs.embedding.dimensions") == 3
        assert attrs.get("neatlogs.llm.token_count.prompt") == 4

    def test_rerank(self, in_memory_span_exporter):
        _setup_tracer(in_memory_span_exporter)
        from neatlogs.openrouter import wrap_openrouter_client

        def rerank(**kwargs):
            return SimpleNamespace(
                model="cohere/rerank-v3.5",
                results=[
                    SimpleNamespace(index=1, relevance_score=0.9, document=SimpleNamespace(text="doc B")),
                    SimpleNamespace(index=0, relevance_score=0.2, document=SimpleNamespace(text="doc A")),
                ],
            )

        client = _fake_client(rerank=rerank)
        wrap_openrouter_client(client)
        client.rerank.rerank(model="cohere/rerank-v3.5", query="q", documents=["doc A", "doc B"])

        attrs = in_memory_span_exporter.get_finished_spans()[0].attributes
        assert attrs.get("neatlogs.span.kind") == "reranker"
        assert attrs.get("neatlogs.reranker.model_name") == "cohere/rerank-v3.5"
        assert attrs.get("neatlogs.reranker.query") == "q"
        assert attrs.get("neatlogs.reranker.input_documents.0") == "doc A"
        assert attrs.get("neatlogs.reranker.output_documents.0") == "doc B"


class TestOpenRouterVendorHelper:
    @pytest.mark.parametrize(
        "model,expected",
        [
            ("openai/gpt-4o-mini", "openai"),
            ("anthropic/claude-3.5-sonnet", "anthropic"),
            ("google/gemini-2.0-flash", "google"),
            ("meta-llama/llama-3.1-70b-instruct", "meta-llama"),
            ("gpt-4o", "gpt-4o"),
        ],
    )
    def test_vendor_from_model(self, model, expected):
        from neatlogs.openrouter import _vendor_from_model

        assert _vendor_from_model(model) == expected


class TestOpenRouterWrapIdempotent:
    def test_idempotent_and_dispatch(self, in_memory_span_exporter):
        _setup_tracer(in_memory_span_exporter)
        from neatlogs.openrouter import wrap_openrouter_client

        def send(**kwargs):
            return SimpleNamespace(choices=[], usage=None, model="openai/gpt-4o", id="x")

        client = _fake_client(chat=send)
        first = client.chat.send
        wrap_openrouter_client(client)
        patched = client.chat.send
        # Second wrap must not double-patch.
        wrap_openrouter_client(client)
        assert client.chat.send is patched
        assert patched is not first
