"""
Tests for AWS Bedrock Instrumentation
=====================================
Verifies the neatlogs.bedrock wrapper traces the Converse and InvokeModel APIs
with provider="bedrock", system=<model vendor>, and canonical neatlogs.* attrs.
"""

import io
import json
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


def _fake_bedrock_client(**methods):
    """Minimal fake boto3 bedrock-runtime client with a service model fingerprint."""
    service_model = SimpleNamespace(service_name="bedrock-runtime")
    meta = SimpleNamespace(service_model=service_model)
    return SimpleNamespace(meta=meta, **methods)


class TestBedrockConverse:
    def test_converse_traces_tokens_and_output(self, in_memory_span_exporter):
        _setup_tracer(in_memory_span_exporter)

        from neatlogs.bedrock import wrap_bedrock_client

        def converse(**kwargs):
            return {
                "output": {
                    "message": {
                        "role": "assistant",
                        "content": [{"text": "Hello from Claude on Bedrock"}],
                    }
                },
                "stopReason": "end_turn",
                "usage": {"inputTokens": 20, "outputTokens": 8, "totalTokens": 28},
            }

        client = _fake_bedrock_client(converse=converse)
        wrap_bedrock_client(client)

        resp = client.converse(
            modelId="anthropic.claude-3-5-sonnet-20240620-v1:0",
            messages=[{"role": "user", "content": [{"text": "Hi"}]}],
            inferenceConfig={"temperature": 0.7, "maxTokens": 256},
        )
        assert resp["stopReason"] == "end_turn"

        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) == 1
        attrs = spans[0].attributes
        assert attrs.get("neatlogs.span.kind") == "llm"
        assert attrs.get("neatlogs.llm.provider") == "bedrock"
        assert attrs.get("neatlogs.llm.system") == "anthropic"
        assert attrs.get("neatlogs.llm.input_messages.0.role") == "user"
        assert attrs.get("neatlogs.llm.input_messages.0.content") == "Hi"
        assert attrs.get("neatlogs.llm.output_messages.0.content") == "Hello from Claude on Bedrock"
        assert attrs.get("neatlogs.llm.token_count.prompt") == 20
        assert attrs.get("neatlogs.llm.token_count.completion") == 8
        assert attrs.get("neatlogs.llm.token_count.total") == 28
        assert attrs.get("neatlogs.llm.finish_reason") == "end_turn"
        assert attrs.get("neatlogs.llm.temperature") == 0.7
        assert attrs.get("neatlogs.llm.max_tokens") == 256

    def test_cross_region_inference_profile_vendor(self, in_memory_span_exporter):
        _setup_tracer(in_memory_span_exporter)
        from neatlogs.bedrock import wrap_bedrock_client

        def converse(**kwargs):
            return {"output": {"message": {"content": [{"text": "ok"}]}}, "usage": {}}

        client = _fake_bedrock_client(converse=converse)
        wrap_bedrock_client(client)
        client.converse(modelId="us.meta.llama3-1-70b-instruct-v1:0", messages=[])

        spans = in_memory_span_exporter.get_finished_spans()
        assert spans[0].attributes.get("neatlogs.llm.system") == "meta"


class TestBedrockInvokeModel:
    def test_invoke_model_claude_messages(self, in_memory_span_exporter):
        _setup_tracer(in_memory_span_exporter)
        from neatlogs.bedrock import wrap_bedrock_client

        body_out = json.dumps(
            {
                "content": [{"type": "text", "text": "Invoke output"}],
                "usage": {"input_tokens": 12, "output_tokens": 6},
                "stop_reason": "end_turn",
            }
        ).encode()

        def invoke_model(**kwargs):
            return {"body": io.BytesIO(body_out)}

        client = _fake_bedrock_client(invoke_model=invoke_model)
        wrap_bedrock_client(client)

        req_body = json.dumps(
            {"messages": [{"role": "user", "content": "Hi"}], "max_tokens": 100}
        )
        resp = client.invoke_model(modelId="anthropic.claude-3-haiku-20240307-v1:0", body=req_body)

        # Body must still be readable by the caller.
        parsed = json.loads(resp["body"].read())
        assert parsed["content"][0]["text"] == "Invoke output"

        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) == 1
        attrs = spans[0].attributes
        assert attrs.get("neatlogs.llm.provider") == "bedrock"
        assert attrs.get("neatlogs.llm.system") == "anthropic"
        assert attrs.get("neatlogs.llm.input_messages.0.content") == "Hi"
        assert attrs.get("neatlogs.llm.output_messages.0.content") == "Invoke output"
        assert attrs.get("neatlogs.llm.token_count.prompt") == 12
        assert attrs.get("neatlogs.llm.token_count.completion") == 6
        assert attrs.get("neatlogs.llm.finish_reason") == "end_turn"

    def test_invoke_model_titan(self, in_memory_span_exporter):
        _setup_tracer(in_memory_span_exporter)
        from neatlogs.bedrock import wrap_bedrock_client

        body_out = json.dumps(
            {
                "inputTextTokenCount": 9,
                "results": [
                    {"outputText": "Titan says hi", "tokenCount": 5, "completionReason": "FINISH"}
                ],
            }
        ).encode()

        def invoke_model(**kwargs):
            return {"body": io.BytesIO(body_out)}

        client = _fake_bedrock_client(invoke_model=invoke_model)
        wrap_bedrock_client(client)
        client.invoke_model(
            modelId="amazon.titan-text-express-v1",
            body=json.dumps({"inputText": "Hi", "textGenerationConfig": {"maxTokenCount": 50}}),
        )

        attrs = in_memory_span_exporter.get_finished_spans()[0].attributes
        assert attrs.get("neatlogs.llm.system") == "amazon"
        assert attrs.get("neatlogs.llm.output_messages.0.content") == "Titan says hi"
        assert attrs.get("neatlogs.llm.token_count.prompt") == 9
        assert attrs.get("neatlogs.llm.token_count.completion") == 5


class TestBedrockVendorHelper:
    @pytest.mark.parametrize(
        "model_id,expected",
        [
            ("anthropic.claude-3-5-sonnet-20240620-v1:0", "anthropic"),
            ("amazon.titan-text-express-v1", "amazon"),
            ("meta.llama3-70b-instruct-v1:0", "meta"),
            ("us.anthropic.claude-3-haiku-20240307-v1:0", "anthropic"),
            ("cohere.command-r-v1:0", "cohere"),
        ],
    )
    def test_vendor_from_model(self, model_id, expected):
        from neatlogs.bedrock import _vendor_from_model

        assert _vendor_from_model(model_id) == expected
