"""
Tests for Vertex AI Instrumentation
===================================
Verifies the neatlogs.vertex_ai wrapper traces google-genai clients running in
Vertex mode with provider="vertex_ai" / system="vertexai".
"""

from types import SimpleNamespace

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from neatlogs import vertex_ai


def _setup_tracer(exporter):
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    # Reset the wrapper-tracer cache so each test binds to its own provider
    # (the cache is process-global and correct in production, where init() sets
    # the provider once, but leaks across tests).
    import neatlogs._wrap_utils as _wu

    _wu._wrapper_tracer = None
    return provider


def _fake_generate_content_response():
    part = SimpleNamespace(text="Hello from Vertex", thought=False, function_call=None)
    content = SimpleNamespace(parts=[part])
    candidate = SimpleNamespace(content=content, finish_reason="STOP")
    usage = SimpleNamespace(
        prompt_token_count=14,
        candidates_token_count=5,
        total_token_count=19,
        cached_content_token_count=None,
        thoughts_token_count=None,
    )
    return SimpleNamespace(candidates=[candidate], usage_metadata=usage)


class TestVertexAIInstrumentation:
    def test_wrap_traces_generate_content(self, in_memory_span_exporter):
        _setup_tracer(in_memory_span_exporter)

        from neatlogs.vertex_ai import wrap_vertex_ai_client

        def generate_content(*args, **kwargs):
            return _fake_generate_content_response()

        client = SimpleNamespace(
            vertexai=True,
            models=SimpleNamespace(generate_content=generate_content),
        )
        wrap_vertex_ai_client(client)

        resp = client.models.generate_content(model="gemini-2.0-flash", contents="Hi Vertex")
        assert resp.candidates[0].content.parts[0].text == "Hello from Vertex"

        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) == 1
        attrs = spans[0].attributes
        assert attrs.get("neatlogs.span.kind") == "llm"
        assert attrs.get("neatlogs.llm.provider") == "vertex_ai"
        assert attrs.get("neatlogs.llm.system") == "vertexai"
        assert attrs.get("neatlogs.llm.model_name") == "gemini-2.0-flash"
        assert attrs.get("neatlogs.llm.input_messages.0.role") == "user"
        assert attrs.get("neatlogs.llm.input_messages.0.content") == "Hi Vertex"
        assert attrs.get("neatlogs.llm.output_messages.0.content") == "Hello from Vertex"
        assert attrs.get("neatlogs.llm.token_count.prompt") == 14
        assert attrs.get("neatlogs.llm.token_count.completion") == 5
        assert attrs.get("neatlogs.llm.token_count.total") == 19

    def test_is_vertex_client_detection(self):
        from neatlogs.vertex_ai import _is_vertex_client

        assert _is_vertex_client(SimpleNamespace(vertexai=True)) is True
        # Gemini (AI Studio) client: vertexai falsy on both client and _api_client
        gemini = SimpleNamespace(vertexai=False, _api_client=SimpleNamespace(vertexai=False))
        assert _is_vertex_client(gemini) is False
        # Falls back to _api_client.vertexai
        wrapped = SimpleNamespace(_api_client=SimpleNamespace(vertexai=True))
        assert _is_vertex_client(wrapped) is True

    def test_error_records_exception(self, in_memory_span_exporter):
        _setup_tracer(in_memory_span_exporter)
        from neatlogs.vertex_ai import wrap_vertex_ai_client

        def generate_content(*args, **kwargs):
            raise RuntimeError("vertex boom")

        client = SimpleNamespace(
            vertexai=True, models=SimpleNamespace(generate_content=generate_content)
        )
        wrap_vertex_ai_client(client)

        with pytest.raises(RuntimeError):
            client.models.generate_content(model="gemini-2.0-flash", contents="Hi")

        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].status.status_code.name == "ERROR"


def test_vertex_preserves_multiple_response_candidates():
    response = SimpleNamespace(
        candidates=[
            SimpleNamespace(
                index=0,
                content=SimpleNamespace(role="model", parts=[SimpleNamespace(text="alpha")]),
                finish_reason="STOP",
            ),
            SimpleNamespace(
                index=1,
                content=SimpleNamespace(role="model", parts=[SimpleNamespace(text="beta")]),
                finish_reason="MAX_TOKENS",
            ),
        ],
        usage_metadata=None,
    )
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    span = provider.get_tracer("test").start_span("vertex")

    vertex_ai._finalize_response(span, response, 1.0)

    attributes = exporter.get_finished_spans()[0].attributes
    assert attributes["neatlogs.llm.output_messages.0.content"] == "alpha"
    assert attributes["neatlogs.llm.output_messages.1.content"] == "beta"
    assert attributes["neatlogs.llm.choices.0.finish_reason"] == "STOP"
    assert attributes["neatlogs.llm.choices.1.finish_reason"] == "MAX_TOKENS"
