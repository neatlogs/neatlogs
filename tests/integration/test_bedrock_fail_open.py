"""Regression tests for Bedrock wrapper fail-open behavior."""

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


def _fake_bedrock_client(**methods):
    """Minimal fake boto3 bedrock-runtime client with a service model fingerprint."""
    service_model = SimpleNamespace(service_name="bedrock-runtime")
    meta = SimpleNamespace(service_model=service_model)
    return SimpleNamespace(meta=meta, **methods)


def _converse_response():
    return {
        "output": {
            "message": {
                "role": "assistant",
                "content": [{"text": "hello"}],
            }
        },
        "stopReason": "end_turn",
        "usage": {"inputTokens": 5, "outputTokens": 3, "totalTokens": 8},
    }


class TestBedrockFailOpen:
    def test_converse_fails_open_on_tracer_error(self, in_memory_span_exporter):
        from neatlogs.bedrock import wrap_bedrock_client

        _setup_tracer(in_memory_span_exporter)

        def converse(*args, **kwargs):
            return _converse_response()

        client = _fake_bedrock_client(converse=converse)
        wrap_bedrock_client(client)

        with patch(
            "neatlogs.bedrock.get_provider_tracer",
            side_effect=RuntimeError("trace failed"),
        ):
            response = client.converse(
                modelId="anthropic.claude-3-5-sonnet-20240620-v1:0",
                messages=[{"role": "user", "content": [{"text": "hi"}]}],
            )

        assert response["output"]["message"]["content"][0]["text"] == "hello"
        assert len(in_memory_span_exporter.get_finished_spans()) == 0

    def test_converse_stream_fails_open_on_tracer_error(self, in_memory_span_exporter):
        from neatlogs.bedrock import wrap_bedrock_client

        _setup_tracer(in_memory_span_exporter)

        class _FakeStream:
            def __iter__(self_inner):
                return iter([])

        def converse_stream(*args, **kwargs):
            return {"stream": _FakeStream()}

        client = _fake_bedrock_client(converse_stream=converse_stream)
        wrap_bedrock_client(client)

        with patch(
            "neatlogs.bedrock.get_provider_tracer",
            side_effect=RuntimeError("trace failed"),
        ):
            response = client.converse_stream(
                modelId="anthropic.claude-3-5-sonnet-20240620-v1:0",
                messages=[{"role": "user", "content": [{"text": "hi"}]}],
            )

        assert isinstance(response["stream"], _FakeStream)
        assert len(in_memory_span_exporter.get_finished_spans()) == 0

    def test_wrap_survives_patch_failure(self):
        from neatlogs.bedrock import wrap_bedrock_client

        def converse(*args, **kwargs):
            return {}

        client = _fake_bedrock_client(converse=converse)

        with patch(
            "neatlogs.bedrock._patch_converse",
            side_effect=RuntimeError("patch failed"),
        ):
            result = wrap_bedrock_client(client)

        assert result is client
