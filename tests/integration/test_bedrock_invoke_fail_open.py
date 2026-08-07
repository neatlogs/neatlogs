"""Regression tests for Bedrock invoke_model + invoke_model_with_response_stream fail-open."""

import io
import json
from types import SimpleNamespace
from unittest.mock import patch

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
    service_model = SimpleNamespace(service_name="bedrock-runtime")
    meta = SimpleNamespace(service_model=service_model)
    return SimpleNamespace(meta=meta, **methods)


class TestBedrockInvokeFailOpen:
    def test_invoke_model_fails_open_on_tracer_error(self, in_memory_span_exporter):
        from neatlogs.bedrock import wrap_bedrock_client

        _setup_tracer(in_memory_span_exporter)

        body_out = json.dumps(
            {
                "content": [{"type": "text", "text": "hello"}],
                "usage": {"input_tokens": 4, "output_tokens": 2},
                "stop_reason": "end_turn",
            }
        ).encode()

        def invoke_model(**kwargs):
            return {"body": io.BytesIO(body_out)}

        client = _fake_bedrock_client(invoke_model=invoke_model)
        wrap_bedrock_client(client)

        with patch(
            "neatlogs.bedrock.get_provider_tracer",
            side_effect=RuntimeError("trace failed"),
        ):
            response = client.invoke_model(
                modelId="anthropic.claude-3-haiku-20240307-v1:0",
                body=json.dumps({"messages": [{"role": "user", "content": "hi"}]}),
            )

        assert json.loads(response["body"].read())["content"][0]["text"] == "hello"
        assert len(in_memory_span_exporter.get_finished_spans()) == 0

    def test_invoke_model_stream_fails_open_on_tracer_error(self, in_memory_span_exporter):
        from neatlogs.bedrock import wrap_bedrock_client

        _setup_tracer(in_memory_span_exporter)

        class _FakeStream:
            def __iter__(self_inner):
                return iter([])

        def invoke_model_with_response_stream(**kwargs):
            return {"body": _FakeStream()}

        client = _fake_bedrock_client(
            invoke_model_with_response_stream=invoke_model_with_response_stream
        )
        wrap_bedrock_client(client)

        with patch(
            "neatlogs.bedrock.get_provider_tracer",
            side_effect=RuntimeError("trace failed"),
        ):
            response = client.invoke_model_with_response_stream(
                modelId="anthropic.claude-3-haiku-20240307-v1:0",
                body=json.dumps({"messages": [{"role": "user", "content": "hi"}]}),
            )

        assert isinstance(response["body"], _FakeStream)
        assert len(in_memory_span_exporter.get_finished_spans()) == 0
