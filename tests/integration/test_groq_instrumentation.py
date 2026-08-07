"""
Tests for Groq Instrumentation
==============================
Verifies ``neatlogs.groq`` traces chat completions on both sync ``Groq`` and
async ``AsyncGroq`` clients — including streaming — with
``provider="groq"``, ``system="groq"``, and the canonical ``neatlogs.*``
attributes, plus regression coverage for fail-open on telemetry-setup errors.
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


def _fake_groq_client(create=None, async_create=None):
    """Minimal fake groq.Groq / groq.AsyncGroq client."""
    completions = SimpleNamespace()
    if create is not None:
        completions.create = create
    if async_create is not None:
        completions.create = async_create
    chat = SimpleNamespace(completions=completions)
    return SimpleNamespace(chat=chat)


def _llm_spans(exporter):
    return [
        s
        for s in exporter.get_finished_spans()
        if s.attributes.get("neatlogs.llm.provider") == "groq"
        and s.attributes.get("neatlogs.span.kind") == "llm"
    ]


class TestGroqChatCompletions:
    def test_create_traces_io_tokens_and_params(self, in_memory_span_exporter):
        _setup_tracer(in_memory_span_exporter)
        from neatlogs.groq import wrap_groq_client

        def create(*, messages, model, **kwargs):
            return SimpleNamespace(
                id="cmpl-groq-1",
                model=model,
                choices=[
                    SimpleNamespace(
                        index=0,
                        finish_reason="stop",
                        message=SimpleNamespace(
                            role="assistant",
                            content="hello from llama",
                            tool_calls=None,
                            reasoning=None,
                        ),
                    )
                ],
                usage=SimpleNamespace(prompt_tokens=5, completion_tokens=3, total_tokens=8),
            )

        client = _fake_groq_client(create=create)
        wrap_groq_client(client)

        client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.5,
            top_p=0.9,
            max_tokens=256,
        )

        spans = _llm_spans(in_memory_span_exporter)
        assert len(spans) == 1
        attrs = spans[0].attributes
        assert attrs.get("neatlogs.span.kind") == "llm"
        assert attrs.get("neatlogs.llm.provider") == "groq"
        assert attrs.get("neatlogs.llm.system") == "groq"
        assert attrs.get("neatlogs.llm.model_name") == "llama-3.3-70b-versatile"
        assert attrs.get("neatlogs.llm.input_messages.0.role") == "user"
        assert attrs.get("neatlogs.llm.input_messages.0.content") == "hi"
        assert attrs.get("neatlogs.llm.output_messages.0.content") == "hello from llama"
        assert attrs.get("output.value") == "hello from llama"
        assert attrs.get("neatlogs.llm.token_count.prompt") == 5
        assert attrs.get("neatlogs.llm.token_count.completion") == 3
        assert attrs.get("neatlogs.llm.token_count.total") == 8
        assert attrs.get("neatlogs.llm.finish_reason") == "stop"
        assert attrs.get("neatlogs.llm.response_id") == "cmpl-groq-1"
        assert attrs.get("neatlogs.llm.temperature") == 0.5
        assert attrs.get("neatlogs.llm.top_p") == 0.9
        assert attrs.get("neatlogs.llm.max_tokens") == 256
        import json

        blob = json.loads(attrs.get("neatlogs.llm.invocation_parameters"))
        assert blob == {"temperature": 0.5, "top_p": 0.9, "max_tokens": 256}

    def test_create_streaming(self, in_memory_span_exporter):
        _setup_tracer(in_memory_span_exporter)
        from neatlogs.groq import wrap_groq_client

        def _chunk(text=None, finish=None, usage=None):
            delta = SimpleNamespace(content=text, tool_calls=None, reasoning=None)
            choice = SimpleNamespace(index=0, delta=delta, finish_reason=finish)
            return SimpleNamespace(model="llama-3.3-70b-versatile", choices=[choice], usage=usage)

        def create(*, messages, model, stream=False, **kwargs):
            assert stream is True
            return iter(
                [
                    _chunk("Hello"),
                    _chunk(" world"),
                    _chunk(
                        finish="stop",
                        usage=SimpleNamespace(prompt_tokens=4, completion_tokens=2, total_tokens=6),
                    ),
                ]
            )

        client = _fake_groq_client(create=create)
        wrap_groq_client(client)

        stream = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
        )
        collected = "".join((getattr(c.choices[0].delta, "content", "") or "") for c in stream)
        assert collected == "Hello world"

        spans = _llm_spans(in_memory_span_exporter)
        assert len(spans) == 1
        attrs = spans[0].attributes
        assert attrs.get("neatlogs.llm.is_streaming") is True
        assert attrs.get("neatlogs.llm.output_messages.0.content") == "Hello world"
        assert attrs.get("neatlogs.llm.token_count.prompt") == 4
        assert attrs.get("neatlogs.llm.token_count.completion") == 2
        assert attrs.get("neatlogs.llm.finish_reason") == "stop"

    @pytest.mark.asyncio
    async def test_async_create_traces(self, in_memory_span_exporter):
        _setup_tracer(in_memory_span_exporter)
        from neatlogs.groq import wrap_groq_client

        async def create(*, messages, model, **kwargs):
            return SimpleNamespace(
                id="cmpl-async-1",
                model=model,
                choices=[
                    SimpleNamespace(
                        index=0,
                        finish_reason="stop",
                        message=SimpleNamespace(
                            role="assistant", content="async hi", tool_calls=None, reasoning=None
                        ),
                    )
                ],
                usage=SimpleNamespace(prompt_tokens=2, completion_tokens=1, total_tokens=3),
            )

        # AsyncGroq's class name contains "Async" — the wrapper detects it.
        from groq.resources.chat.completions import AsyncCompletions

        completions = AsyncCompletions.__new__(AsyncCompletions)
        completions.create = create
        chat = SimpleNamespace(completions=completions)
        client = SimpleNamespace(chat=chat)
        wrap_groq_client(client)

        await client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": "hi"}],
        )

        spans = _llm_spans(in_memory_span_exporter)
        assert len(spans) == 1
        attrs = spans[0].attributes
        assert attrs.get("neatlogs.llm.provider") == "groq"
        assert attrs.get("neatlogs.llm.output_messages.0.content") == "async hi"


class TestGroqReasoningFields:
    def test_reasoning_field_recorded(self, in_memory_span_exporter):
        _setup_tracer(in_memory_span_exporter)
        from neatlogs.groq import wrap_groq_client

        def create(*, messages, model, **kwargs):
            return SimpleNamespace(
                id="r1",
                model=model,
                choices=[
                    SimpleNamespace(
                        index=0,
                        finish_reason="stop",
                        message=SimpleNamespace(
                            role="assistant",
                            content="final answer",
                            tool_calls=None,
                            reasoning="step-by-step thoughts",
                        ),
                    )
                ],
                usage=SimpleNamespace(
                    prompt_tokens=10,
                    completion_tokens=20,
                    total_tokens=30,
                    completion_tokens_details=SimpleNamespace(reasoning_tokens=15),
                ),
            )

        client = _fake_groq_client(create=create)
        wrap_groq_client(client)
        client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": "explain"}],
        )

        attrs = _llm_spans(in_memory_span_exporter)[0].attributes
        assert attrs.get("neatlogs.llm.output_messages.0.reasoning") == "step-by-step thoughts"
        assert attrs.get("neatlogs.llm.token_count.reasoning") == 15


class TestGroqIdempotentAndSuppress:
    def test_wrap_is_idempotent(self, in_memory_span_exporter):
        _setup_tracer(in_memory_span_exporter)
        from neatlogs.groq import wrap_groq_client

        def create(*, messages, model, **kwargs):
            return SimpleNamespace(
                id="x",
                model=model,
                choices=[
                    SimpleNamespace(
                        index=0,
                        finish_reason="stop",
                        message=SimpleNamespace(
                            role="assistant", content="ok", tool_calls=None, reasoning=None
                        ),
                    )
                ],
                usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            )

        client = _fake_groq_client(create=create)
        wrap_groq_client(client)
        patched = client.chat.completions.create
        wrap_groq_client(client)
        assert client.chat.completions.create is patched


class TestGroqFailOpen:
    def test_create_fails_open_on_tracer_error(self, in_memory_span_exporter):
        _setup_tracer(in_memory_span_exporter)
        from neatlogs.groq import wrap_groq_client

        def create(*, messages, model, **kwargs):
            return SimpleNamespace(
                id="ok",
                model=model,
                choices=[
                    SimpleNamespace(
                        index=0,
                        finish_reason="stop",
                        message=SimpleNamespace(
                            role="assistant", content="ok", tool_calls=None, reasoning=None
                        ),
                    )
                ],
                usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            )

        client = _fake_groq_client(create=create)
        wrap_groq_client(client)

        with patch("neatlogs.groq.get_provider_tracer", side_effect=RuntimeError("trace failed")):
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": "hi"}],
            )

        assert response.choices[0].message.content == "ok"
        assert len(_llm_spans(in_memory_span_exporter)) == 0

    @pytest.mark.asyncio
    async def test_async_create_fails_open_on_tracer_error(self, in_memory_span_exporter):
        _setup_tracer(in_memory_span_exporter)
        from neatlogs.groq import wrap_groq_client

        async def create(*, messages, model, **kwargs):
            return SimpleNamespace(
                id="ok",
                model=model,
                choices=[
                    SimpleNamespace(
                        index=0,
                        finish_reason="stop",
                        message=SimpleNamespace(
                            role="assistant", content="async ok", tool_calls=None, reasoning=None
                        ),
                    )
                ],
                usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            )

        from groq.resources.chat.completions import AsyncCompletions

        completions = AsyncCompletions.__new__(AsyncCompletions)
        completions.create = create
        chat = SimpleNamespace(completions=completions)
        client = SimpleNamespace(chat=chat)
        wrap_groq_client(client)

        with patch("neatlogs.groq.get_provider_tracer", side_effect=RuntimeError("trace failed")):
            response = await client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": "hi"}],
            )

        assert response.choices[0].message.content == "async ok"
        assert len(_llm_spans(in_memory_span_exporter)) == 0
