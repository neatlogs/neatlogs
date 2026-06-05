"""
Tests for Azure OpenAI Instrumentation
======================================
Verifies the neatlogs.azure_openai wrapper traces AzureOpenAI chat completions
with provider="azure" and the canonical neatlogs.* attributes.
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
    # Reset the process-global wrapper-tracer cache so each test binds to its
    # own provider (correct in production; leaks across tests otherwise).
    import neatlogs._wrap_utils as _wu
    _wu._wrapper_tracer = None
    return provider


def _fake_chat_completion():
    return SimpleNamespace(
        model="gpt-4o",
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="Hi from Azure!", tool_calls=None),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=11,
            completion_tokens=4,
            total_tokens=15,
            prompt_tokens_details=None,
            completion_tokens_details=None,
        ),
    )


class TestAzureOpenAIInstrumentation:
    def test_wrap_traces_chat_completion(self, in_memory_span_exporter):
        _setup_tracer(in_memory_span_exporter)

        import neatlogs
        from neatlogs.azure_openai import wrap_azure_openai_client

        # A minimal fake AzureOpenAI client (avoids needing real Azure creds).
        captured = {}

        def fake_create(**kwargs):
            captured.update(kwargs)
            return _fake_chat_completion()

        client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create)),
            responses=None,
            embeddings=None,
        )

        wrap_azure_openai_client(client)
        result = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hi"}],
            temperature=0.5,
        )

        assert result.choices[0].message.content == "Hi from Azure!"

        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        attrs = span.attributes
        assert attrs.get("neatlogs.span.kind") == "llm"
        assert attrs.get("neatlogs.llm.provider") == "azure"
        assert attrs.get("neatlogs.llm.system") == "azure"
        assert attrs.get("neatlogs.llm.input_messages.0.role") == "user"
        assert attrs.get("neatlogs.llm.input_messages.0.content") == "Hi"
        assert attrs.get("neatlogs.llm.output_messages.0.content") == "Hi from Azure!"
        assert attrs.get("neatlogs.llm.token_count.prompt") == 11
        assert attrs.get("neatlogs.llm.token_count.completion") == 4
        assert attrs.get("neatlogs.llm.temperature") == 0.5

    def test_wrap_dispatch_via_neatlogs_wrap(self, in_memory_span_exporter):
        """neatlogs.wrap() should route a real AzureOpenAI instance to the azure wrapper."""
        _setup_tracer(in_memory_span_exporter)

        pytest.importorskip("openai")
        from openai import AzureOpenAI

        import neatlogs

        client = AzureOpenAI(
            azure_endpoint="https://example.openai.azure.com",
            api_key="fake-key",
            api_version="2024-02-01",
        )
        returned = neatlogs.wrap(client)
        # Same client returned, chat.completions.create patched.
        assert returned is client
        assert getattr(client.chat.completions, "_neatlogs_azure_patched", False) is True

    def test_error_records_exception(self, in_memory_span_exporter):
        _setup_tracer(in_memory_span_exporter)

        from neatlogs.azure_openai import wrap_azure_openai_client

        def fake_create(**kwargs):
            raise RuntimeError("boom")

        client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create)),
            responses=None,
            embeddings=None,
        )
        wrap_azure_openai_client(client)

        with pytest.raises(RuntimeError):
            client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "Hi"}])

        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].status.status_code.name == "ERROR"
