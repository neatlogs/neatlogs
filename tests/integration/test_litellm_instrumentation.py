"""
Tests for LiteLLM Instrumentation
=================================
Tests for LiteLLM's unified interface and Proxy patterns.
"""

import time

import httpx
import pytest
import respx
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor


class TestLiteLLMInstrumentation:

    @pytest.fixture(autouse=True)
    def setup_teardown(self, in_memory_span_exporter):
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(in_memory_span_exporter))
        trace.set_tracer_provider(provider)

        import neatlogs

        neatlogs.init(api_key="test-key", instrumentations=["litellm"])
        yield

    @respx.mock
    def test_litellm_basic_completion(self, in_memory_span_exporter):
        """
        Test 1: Basic Unified Call.
        Expected: PASS (LiteLLM behaves like OpenAI).
        """
        # LiteLLM calls OpenAI under the hood for this model
        respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={"choices": [{"message": {"role": "assistant", "content": "Hello LiteLLM"}}]},
            )
        )

        from litellm import completion

        # FIXED: Added api_key to bypass local validation
        completion(
            model="gpt-3.5-turbo", messages=[{"role": "user", "content": "Hi"}], api_key="sk-fake"
        )

        time.sleep(0.1)
        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) >= 1, "Basic LiteLLM span missing"
        assert spans[0].attributes.get("llm.system") == "openai"

    @respx.mock
    def test_litellm_metadata_injection(self, in_memory_span_exporter):
        """
        Test 2: Metadata/Variable Injection.
        Scenario: User passes custom metadata (critical for tracing users/envs).
        Expected: FAIL (SDK likely drops 'metadata' or kwargs).
        """
        respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(
                200, json={"choices": [{"message": {"role": "assistant", "content": "OK"}}]}
            )
        )

        from litellm import completion

        custom_metadata = {
            "metadata": {"user_id": "u_123", "environment": "production"},
            "temperature": 0.5,
        }

        # FIXED: Added api_key
        completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hi"}],
            api_key="sk-fake",
            **custom_metadata,
        )

        time.sleep(0.1)
        spans = in_memory_span_exporter.get_finished_spans()
        span = spans[0]

        # 🕵️‍♂️ BUG CHECK
        # We expect to see 'user_id' in the span attributes
        attributes_str = str(span.attributes)
        assert (
            "u_123" in attributes_str
        ), f"Failed to capture custom metadata. Span attributes: {span.attributes.keys()}"

    @respx.mock
    def test_litellm_streaming(self, in_memory_span_exporter):
        """
        Test 3: LiteLLM Streaming.
        Expected: FAIL (Universal streaming issue).
        """
        stream_chunks = [
            'data: {"choices":[{"delta":{"content":"Lite"}}]}\n\n',
            'data: {"choices":[{"delta":{"content":"LLM"}}]}\n\n',
            "data: [DONE]\n\n",
        ]

        respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(
                200, content="".join(stream_chunks), headers={"Content-Type": "text/event-stream"}
            )
        )

        from litellm import completion

        # FIXED: Added api_key
        response = completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Stream"}],
            stream=True,
            api_key="sk-fake",
        )

        for chunk in response:
            pass

        time.sleep(0.1)
        spans = in_memory_span_exporter.get_finished_spans()

        if len(spans) == 0:
            pytest.fail("LiteLLM Streaming failed to generate span.")
