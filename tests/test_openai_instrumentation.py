"""
Tests for OpenAI Instrumentation
=================================
Tests that verify OpenAI SDK calls are properly instrumented, covering
basic chat completions, streaming, and tool calling workflows.
"""

import pytest
import respx
import httpx
import time
from unittest.mock import Mock, patch
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor


class TestOpenAIInstrumentation:
    """
    Comprehensive Test Suite for OpenAI Instrumentation.
    Includes basic smoke tests, streaming, and tool usage.
    """

    @pytest.fixture(autouse=True)
    def setup_teardown(self, in_memory_span_exporter):
        """
        Setup fresh tracer provider and neatlogs instance before each test.
        """
        # 1. Reset Tracer Provider
        provider = TracerProvider()
        provider.add_span_processor(
            SimpleSpanProcessor(in_memory_span_exporter))
        trace.set_tracer_provider(provider)

        # 2. Initialize neatlogs (mocking preventing real network calls)
        import neatlogs
        # We assume clean-up is handled by conftest.py, but we re-init here
        neatlogs.init(api_key="test-key", instrumentations=["openai"])

        yield

    @pytest.fixture
    def openai_chat_response_json(self):
        """Mock OpenAI API standard response JSON."""
        return {
            "id": "chatcmpl-test123",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "gpt-3.5-turbo-0613",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello! How can I help you today?",
                },
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
        }

    # ==========================================
    # BASIC WORKFLOW TESTS
    # ==========================================

    @respx.mock
    def test_basic_chat_completion(self, openai_chat_response_json, in_memory_span_exporter):
        """Test that a standard non-streaming chat completion creates a span."""
        respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=openai_chat_response_json)
        )

        from openai import OpenAI
        client = OpenAI(api_key="fake-key")

        client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Say hello"}],
        )

        # Allow background processing
        time.sleep(0.1)

        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) >= 1, "At least one span should be created"

        # Verify specific attributes
        llm_span = spans[0]
        assert llm_span.name == "ChatCompletion"
        assert llm_span.attributes.get("llm.system") == "openai"
        assert llm_span.attributes.get("llm.request.model") == "gpt-3.5-turbo"

    # ==========================================
    # ADVANCED WORKFLOW TESTS (BUG HUNTING)
    # ==========================================

    @respx.mock
    def test_openai_streaming_attributes(self, in_memory_span_exporter):
        """
        Test: Streaming Response (Server-Sent Events).
        Goal: Verify if the SDK correctly aggregates text chunks into 'output.value'.
        """
        # Mock Streaming Chunks
        stream_chunks = [
            'data: {"id":"1","choices":[{"delta":{"content":""}}]}\n\n',
            'data: {"id":"1","choices":[{"delta":{"content":"Hello "}}]}\n\n',
            'data: {"id":"1","choices":[{"delta":{"content":"World"}}]}\n\n',
            'data: [DONE]\n\n'
        ]

        respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(200, content="".join(stream_chunks), headers={
                                        "Content-Type": "text/event-stream"})
        )

        from openai import OpenAI
        client = OpenAI(api_key="fake-key")

        # Act: Call with stream=True
        stream = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hi"}],
            stream=True
        )

        # Consume the stream
        for _ in stream:
            pass

        time.sleep(0.1)

        # Assert: Check Spans
        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) > 0, "Streaming should create a span"

        llm_span = spans[0]
        attributes = llm_span.attributes

        # Documenting the behavior:
        # We expect "output.value" to be the combined text "Hello World".
        # If the SDK captures a raw JSON string instead, this assertion will fail.
        actual_output = attributes.get("output.value")
        print(f"DEBUG: Actual Streaming Output: {actual_output}")

        assert actual_output == "Hello World", \
            f"Bug detected: Expected 'Hello World' but got {actual_output}"

    @respx.mock
    def test_openai_tool_calling_span_creation(self, in_memory_span_exporter):
        """
        Test: Tool/Function Calling.
        Goal: Verify if the SDK handles 'tool_calls' without crashing or dropping the span.
        """
        tool_response = {
            "id": "chatcmpl-tool",
            "model": "gpt-4-turbo",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,  # Content is None when tool is called
                    "tool_calls": [{
                        "id": "call_123",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": "{\"location\": \"Delhi\"}"
                        }
                    }]
                },
                "finish_reason": "tool_calls"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}
        }

        respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=tool_response)
        )

        from openai import OpenAI
        client = OpenAI(api_key="fake-key")

        tools = [{"type": "function", "function": {
            "name": "get_weather", "parameters": {}}}]

        try:
            client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[{"role": "user", "content": "Weather?"}],
                tools=tools
            )
        except Exception as e:
            pytest.fail(f"SDK crashed during tool call: {str(e)}")

        time.sleep(0.1)

        spans = in_memory_span_exporter.get_finished_spans()

        # Critical Check: Did we get a span?
        if len(spans) == 0:
            pytest.fail(
                "CRITICAL BUG: No span created for Tool Calling request. SDK likely failed to parse response.")

        llm_span = spans[0]
        # Check if tool info is present
        assert "llm.request.functions" in llm_span.attributes or "tool_calls" in str(
            llm_span.attributes)

    @respx.mock
    def test_openai_complex_content_array(self, in_memory_span_exporter):
        """
        Scenario: Content is a list of blocks (Realistic for Multi-modal/Vision).
        Pattern: Anthropic Pattern B / Google Pattern B
        """
        respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(
                200, json={"choices": [{"message": {"content": "Done"}}]})
        )

        from openai import OpenAI
        client = OpenAI(api_key="sk-fake")

        # Realistic Multi-part content
        complex_content = [
            {"type": "text", "text": "Check this logs:"},
            {"type": "text", "text": "ERROR: Connection Timeout"}
        ]

        client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": complex_content}]
        )

        time.sleep(0.1)
        span = in_memory_span_exporter.get_finished_spans()[0]

        # BUG CHECK: Does the SDK flatten the array into a readable string?
        input_val = span.attributes.get("input.value")
        assert "ERROR: Connection Timeout" in input_val
        assert isinstance(
            input_val, str), "input.value should be a string even if input was an array"

    @respx.mock
    def test_openai_extra_metadata_capture(self, in_memory_span_exporter):
        """
        Scenario: Passing custom metadata or tags.
        Pattern: LiteLLM / Custom App Metadata.
        """
        respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(
                200, json={"choices": [{"message": {"content": "OK"}}]})
        )

        from openai import OpenAI
        client = OpenAI(api_key="sk-fake")

        # Realistic scenario: Tracking which user or environment made the call
        client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hi"}],
            extra_body={
                "metadata": {"user_id": "cust_1234", "env": "production"}
            }
        )

        time.sleep(0.1)
        span = in_memory_span_exporter.get_finished_spans()[0]

        # CHECK: Are we capturing custom metadata?
        # OpenInference often uses 'llm.metadata' or 'metadata'
        assert "cust_1234" in str(span.attributes)
