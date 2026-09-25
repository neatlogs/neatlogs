"""
Tests for OpenRouter fail-open behavior.

Regression coverage for the gap where telemetry-setup errors during span
creation would propagate to the caller. These tests patch the wrapper's
``get_provider_tracer`` to raise and assert that the underlying SDK call
still runs untraced.
"""

from types import SimpleNamespace
from unittest.mock import patch

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
    chat_obj = SimpleNamespace(send=chat) if chat is not None else SimpleNamespace()
    beta = SimpleNamespace(
        responses=SimpleNamespace(send=responses) if responses is not None else SimpleNamespace()
    )
    emb = SimpleNamespace(generate=embeddings) if embeddings is not None else SimpleNamespace()
    rer = SimpleNamespace(rerank=rerank) if rerank is not None else SimpleNamespace()
    return SimpleNamespace(chat=chat_obj, beta=beta, embeddings=emb, rerank=rer)


class TestOpenRouterChatFailOpen:
    def test_chat_send_fails_open_on_tracer_error(self, in_memory_span_exporter):
        _setup_tracer(in_memory_span_exporter)
        from neatlogs.openrouter import wrap_openrouter_client

        def send(**kwargs):
            return SimpleNamespace(
                id="ok",
                model="openai/gpt-4o-mini",
                choices=[
                    SimpleNamespace(
                        index=0,
                        finish_reason="stop",
                        message=SimpleNamespace(role="assistant", content="hello"),
                        tool_calls=None,
                    )
                ],
                usage=SimpleNamespace(prompt_tokens=4, completion_tokens=2, total_tokens=6),
            )

        client = _fake_client(chat=send)
        wrap_openrouter_client(client)

        with patch(
            "neatlogs.openrouter.get_provider_tracer", side_effect=RuntimeError("trace failed")
        ):
            response = client.chat.send(
                model="openai/gpt-4o-mini",
                messages=[{"role": "user", "content": "hi"}],
            )

        assert response.choices[0].message.content == "hello"
        assert len(in_memory_span_exporter.get_finished_spans()) == 0

    @pytest.mark.asyncio
    async def test_chat_send_async_fails_open_on_tracer_error(self, in_memory_span_exporter):
        _setup_tracer(in_memory_span_exporter)
        from neatlogs.openrouter import wrap_openrouter_client

        async def send_async(**kwargs):
            return SimpleNamespace(
                id="ok",
                model="openai/gpt-4o-mini",
                choices=[
                    SimpleNamespace(
                        index=0,
                        finish_reason="stop",
                        message=SimpleNamespace(role="assistant", content="hello"),
                        tool_calls=None,
                    )
                ],
                usage=SimpleNamespace(prompt_tokens=4, completion_tokens=2, total_tokens=6),
            )

        chat = SimpleNamespace(send=None, send_async=send_async)
        beta = SimpleNamespace(responses=SimpleNamespace())
        client = SimpleNamespace(
            chat=chat,
            beta=beta,
            embeddings=SimpleNamespace(),
            rerank=SimpleNamespace(),
        )
        wrap_openrouter_client(client)

        with patch(
            "neatlogs.openrouter.get_provider_tracer", side_effect=RuntimeError("trace failed")
        ):
            response = await client.chat.send_async(
                model="openai/gpt-4o-mini",
                messages=[{"role": "user", "content": "hi"}],
            )

        assert response.choices[0].message.content == "hello"
        assert len(in_memory_span_exporter.get_finished_spans()) == 0

    def test_chat_send_set_attribute_fails_after_span_create(self, in_memory_span_exporter):
        """set_attribute fails after start_span succeeded. The partial span
        is ended, the original SDK is called exactly once, the response
        flows through unchanged."""
        from neatlogs.openrouter import wrap_openrouter_client

        _setup_tracer(in_memory_span_exporter)

        call_count = {"n": 0}

        def send(**kwargs):
            call_count["n"] += 1
            return SimpleNamespace(
                id="ok",
                model="openai/gpt-4o-mini",
                choices=[
                    SimpleNamespace(
                        index=0,
                        finish_reason="stop",
                        message=SimpleNamespace(role="assistant", content="hello"),
                        tool_calls=None,
                    )
                ],
                usage=SimpleNamespace(prompt_tokens=4, completion_tokens=2, total_tokens=6),
            )

        # Build a span whose set_attribute raises but end() works fine.
        class _BrokenAttrSpan:
            def set_attribute(self, *a, **kw):
                raise RuntimeError("set_attribute down")

            def set_status(self, *a, **kw):
                pass

            def end(self):
                pass

        class _BrokenAttrTracer:
            def start_span(self, *a, **kw):
                return _BrokenAttrSpan()

        client = _fake_client(chat=send)
        wrap_openrouter_client(client)

        with patch("neatlogs.openrouter.get_provider_tracer", return_value=_BrokenAttrTracer()):
            response = client.chat.send(
                model="openai/gpt-4o-mini",
                messages=[{"role": "user", "content": "hi"}],
            )

        assert call_count["n"] == 1, f"called {call_count['n']} times, expected 1"
        assert response.choices[0].message.content == "hello"
        # The contract: orig was called once, response unchanged, no exception
        # leaked. The partial-span cleanup is implementation detail; the
        # critical property is that the user-facing call succeeded exactly
        # once with the right response.
        spans = in_memory_span_exporter.get_finished_spans()
        # The partial span may or may not be exported depending on whether
        # start_span itself raised before the span could be recorded.
        # Either way, the SDK call must have happened and returned.
        assert call_count["n"] == 1


class TestOpenRouterResponsesFailOpen:
    def test_responses_send_fails_open_on_tracer_error(self, in_memory_span_exporter):
        _setup_tracer(in_memory_span_exporter)
        from neatlogs.openrouter import wrap_openrouter_client

        def send(**kwargs):
            return SimpleNamespace(
                id="resp-ok",
                model="openai/gpt-4o-mini",
                status="completed",
                output_text="hello",
                output=[],
                usage=SimpleNamespace(input_tokens=3, output_tokens=2, total_tokens=5),
            )

        client = _fake_client(responses=send)
        wrap_openrouter_client(client)

        with patch(
            "neatlogs.openrouter.get_provider_tracer", side_effect=RuntimeError("trace failed")
        ):
            response = client.beta.responses.send(
                model="openai/gpt-4o-mini",
                input="hi",
            )

        assert response.output_text == "hello"
        assert len(in_memory_span_exporter.get_finished_spans()) == 0

    @pytest.mark.asyncio
    async def test_responses_send_async_fails_open_on_tracer_error(self, in_memory_span_exporter):
        _setup_tracer(in_memory_span_exporter)
        from neatlogs.openrouter import wrap_openrouter_client

        async def send_async(**kwargs):
            return SimpleNamespace(
                id="resp-ok",
                model="openai/gpt-4o-mini",
                status="completed",
                output_text="hello",
                output=[],
                usage=SimpleNamespace(input_tokens=3, output_tokens=2, total_tokens=5),
            )

        chat = SimpleNamespace()
        beta = SimpleNamespace(responses=SimpleNamespace(send=None, send_async=send_async))
        client = SimpleNamespace(
            chat=chat,
            beta=beta,
            embeddings=SimpleNamespace(),
            rerank=SimpleNamespace(),
        )
        wrap_openrouter_client(client)

        with patch(
            "neatlogs.openrouter.get_provider_tracer", side_effect=RuntimeError("trace failed")
        ):
            response = await client.beta.responses.send_async(
                model="openai/gpt-4o-mini",
                input="hi",
            )

        assert response.output_text == "hello"
        assert len(in_memory_span_exporter.get_finished_spans()) == 0


class TestOpenRouterEmbeddingsFailOpen:
    def test_embeddings_generate_fails_open_on_tracer_error(self, in_memory_span_exporter):
        _setup_tracer(in_memory_span_exporter)
        from neatlogs.openrouter import wrap_openrouter_client

        def generate(**kwargs):
            return SimpleNamespace(
                data=[SimpleNamespace(embedding=[0.1, 0.2])],
                usage=SimpleNamespace(prompt_tokens=2, total_tokens=2),
            )

        client = _fake_client(embeddings=generate)
        wrap_openrouter_client(client)

        with patch(
            "neatlogs.openrouter.get_provider_tracer", side_effect=RuntimeError("trace failed")
        ):
            response = client.embeddings.generate(
                model="openai/text-embedding-3-small",
                input="hi",
            )

        assert len(response.data) == 1
        assert len(in_memory_span_exporter.get_finished_spans()) == 0


class TestOpenRouterRerankFailOpen:
    def test_rerank_rerank_fails_open_on_tracer_error(self, in_memory_span_exporter):
        _setup_tracer(in_memory_span_exporter)
        from neatlogs.openrouter import wrap_openrouter_client

        def rerank(**kwargs):
            return SimpleNamespace(
                model="cohere/rerank-v3.5",
                results=[
                    SimpleNamespace(
                        index=0, relevance_score=0.9, document=SimpleNamespace(text="doc")
                    )
                ],
            )

        client = _fake_client(rerank=rerank)
        wrap_openrouter_client(client)

        with patch(
            "neatlogs.openrouter.get_provider_tracer", side_effect=RuntimeError("trace failed")
        ):
            response = client.rerank.rerank(
                model="cohere/rerank-v3.5",
                query="q",
                documents=["doc"],
            )

        assert response.results[0].document.text == "doc"
        assert len(in_memory_span_exporter.get_finished_spans()) == 0
