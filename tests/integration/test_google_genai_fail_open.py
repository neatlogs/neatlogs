"""Regression tests for Google GenAI wrapper fail-open behavior."""

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


def _genai_response():
    return SimpleNamespace(
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(parts=[SimpleNamespace(text="hello")])
            )
        ],
        usage_metadata=SimpleNamespace(prompt_token_count=5, candidates_token_count=3),
    )


class TestGoogleGenAIFailOpen:
    def test_sync_generate_content_fails_open_on_tracer_error(
        self, in_memory_span_exporter
    ):
        from neatlogs.google_genai import wrap_google_genai_client

        _setup_tracer(in_memory_span_exporter)

        def generate_content(*args, **kwargs):
            return _genai_response()

        models = SimpleNamespace(
            generate_content=generate_content, generate_content_stream=None
        )
        client = SimpleNamespace(models=models, aio=None)
        wrap_google_genai_client(client)

        with patch(
            "neatlogs.google_genai.get_provider_tracer",
            side_effect=RuntimeError("trace failed"),
        ):
            response = client.models.generate_content(
                model="gemini-2.0-flash", contents="hi"
            )

        assert response.candidates[0].content.parts[0].text == "hello"
        assert len(in_memory_span_exporter.get_finished_spans()) == 0

    @pytest.mark.asyncio
    async def test_async_generate_content_fails_open_on_tracer_error(
        self, in_memory_span_exporter
    ):
        from neatlogs.google_genai import wrap_google_genai_client

        _setup_tracer(in_memory_span_exporter)

        async def generate_content(*args, **kwargs):
            return _genai_response()

        models = SimpleNamespace(
            generate_content=generate_content, generate_content_stream=None
        )
        aio = SimpleNamespace(models=models)
        client = SimpleNamespace(models=SimpleNamespace(), aio=aio)
        wrap_google_genai_client(client)

        with patch(
            "neatlogs.google_genai.get_provider_tracer",
            side_effect=RuntimeError("trace failed"),
        ):
            response = await client.aio.models.generate_content(
                model="gemini-2.0-flash", contents="hi"
            )

        assert response.candidates[0].content.parts[0].text == "hello"
        assert len(in_memory_span_exporter.get_finished_spans()) == 0

    def test_wrap_survives_patch_failure(self):
        from neatlogs.google_genai import wrap_google_genai_client

        client = SimpleNamespace(models=SimpleNamespace(), aio=None)

        with patch(
            "neatlogs.google_genai._patch_models",
            side_effect=RuntimeError("patch failed"),
        ):
            result = wrap_google_genai_client(client)

        assert result is client
