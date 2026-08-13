"""Regression tests for Vertex AI wrapper fail-open behavior."""

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


def _vertex_response():
    part = SimpleNamespace(text="hello", thought=False, function_call=None)
    content = SimpleNamespace(parts=[part])
    candidate = SimpleNamespace(content=content, finish_reason="STOP")
    usage = SimpleNamespace(
        prompt_token_count=5,
        candidates_token_count=3,
        total_token_count=8,
        cached_content_token_count=None,
        thoughts_token_count=None,
    )
    return SimpleNamespace(candidates=[candidate], usage_metadata=usage)


class TestVertexAIFailOpen:
    def test_sync_generate_content_fails_open_on_tracer_error(self, in_memory_span_exporter):
        from neatlogs.vertex_ai import wrap_vertex_ai_client

        _setup_tracer(in_memory_span_exporter)

        def generate_content(*args, **kwargs):
            return _vertex_response()

        models = SimpleNamespace(generate_content=generate_content, generate_content_stream=None)
        client = SimpleNamespace(models=models, aio=None)
        wrap_vertex_ai_client(client)

        with patch(
            "neatlogs.vertex_ai.get_provider_tracer",
            side_effect=RuntimeError("trace failed"),
        ):
            response = client.models.generate_content(model="gemini-2.0-flash", contents="hi")

        assert response.candidates[0].content.parts[0].text == "hello"
        assert len(in_memory_span_exporter.get_finished_spans()) == 0

    @pytest.mark.asyncio
    async def test_async_generate_content_fails_open_on_tracer_error(self, in_memory_span_exporter):
        from neatlogs.vertex_ai import wrap_vertex_ai_client

        _setup_tracer(in_memory_span_exporter)

        async def generate_content(*args, **kwargs):
            return _vertex_response()

        models = SimpleNamespace(generate_content=generate_content, generate_content_stream=None)
        aio = SimpleNamespace(models=models)
        client = SimpleNamespace(models=SimpleNamespace(), aio=aio)
        wrap_vertex_ai_client(client)

        with patch(
            "neatlogs.vertex_ai.get_provider_tracer",
            side_effect=RuntimeError("trace failed"),
        ):
            response = await client.aio.models.generate_content(
                model="gemini-2.0-flash", contents="hi"
            )

        assert response.candidates[0].content.parts[0].text == "hello"
        assert len(in_memory_span_exporter.get_finished_spans()) == 0

    def test_wrap_survives_patch_failure(self):
        from neatlogs.vertex_ai import wrap_vertex_ai_client

        client = SimpleNamespace(models=SimpleNamespace(), aio=None)

        with patch(
            "neatlogs.vertex_ai._patch_models",
            side_effect=RuntimeError("patch failed"),
        ):
            result = wrap_vertex_ai_client(client)

        assert result is client

    def test_sync_stream_called_once_when_setup_fails(self, in_memory_span_exporter):
        """When telemetry setup fails, orig_generate_content_stream must be called exactly once."""
        from neatlogs.vertex_ai import wrap_vertex_ai_client

        _setup_tracer(in_memory_span_exporter)

        call_count = {"n": 0}

        def generate_content_stream(*args, **kwargs):
            call_count["n"] += 1
            return SimpleNamespace()

        models = SimpleNamespace(
            generate_content=lambda *a, **k: None,
            generate_content_stream=generate_content_stream,
        )
        client = SimpleNamespace(models=models, aio=None)
        wrap_vertex_ai_client(client)

        with patch(
            "neatlogs.vertex_ai.get_provider_tracer",
            side_effect=RuntimeError("trace failed"),
        ):
            stream = client.models.generate_content_stream(model="gemini-1.5-pro", contents="hi")

        assert call_count["n"] == 1, f"called {call_count['n']} times, expected 1"
        assert stream is not None
        assert len(in_memory_span_exporter.get_finished_spans()) == 0

    @pytest.mark.asyncio
    async def test_async_stream_called_once_when_setup_fails(self, in_memory_span_exporter):
        """Async path: setup fail → orig_generate_content_stream called exactly once."""
        from neatlogs.vertex_ai import wrap_vertex_ai_client

        _setup_tracer(in_memory_span_exporter)

        call_count = {"n": 0}

        async def generate_content_stream(*args, **kwargs):
            call_count["n"] += 1
            return SimpleNamespace()

        models = SimpleNamespace(
            generate_content=lambda *a, **k: None,
            generate_content_stream=generate_content_stream,
        )
        aio = SimpleNamespace(models=models)
        client = SimpleNamespace(models=SimpleNamespace(), aio=aio)
        wrap_vertex_ai_client(client)

        with patch(
            "neatlogs.vertex_ai.get_provider_tracer",
            side_effect=RuntimeError("trace failed"),
        ):
            stream = await client.aio.models.generate_content_stream(
                model="gemini-1.5-pro", contents="hi"
            )

        assert call_count["n"] == 1, f"called {call_count['n']} times, expected 1"
        assert stream is not None
        assert len(in_memory_span_exporter.get_finished_spans()) == 0
