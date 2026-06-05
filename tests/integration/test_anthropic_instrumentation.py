"""
Tests for Google GenAI (Gemini) Instrumentation
==============================================
Tests for the modern google-genai SDK covering Basic, Vertex, Streaming,
Multi-part content, and Tool Calling.
"""

import time
from unittest.mock import Mock, patch

import httpx
import pytest
import respx

# Import the new SDK
from google import genai
from google.genai import types
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor


class TestGoogleGenAIInstrumentation:
    """Test suite for the new Google GenAI SDK instrumentation."""

    @pytest.fixture(autouse=True)
    def setup_teardown(self, in_memory_span_exporter):
        """Setup fresh tracer and neatlogs before each test."""
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(in_memory_span_exporter))
        trace.set_tracer_provider(provider)

        import neatlogs

        # Re-init to ensure clean state
        neatlogs.init(api_key="test-key", instrumentations=["gemini"])
        yield

    @pytest.fixture
    def gemini_response_json(self):
        """Mock Gemini standard response JSON."""
        return {
            "candidates": [
                {
                    "content": {"parts": [{"text": "Hello from Gemini 2.0!"}], "role": "model"},
                    "finishReason": "STOP",
                    "index": 0,
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 15,
                "candidatesTokenCount": 5,
                "totalTokenCount": 20,
            },
        }

    @pytest.fixture
    def mock_google_auth(self):
        """Fixes TypeError in httpx by ensuring the credential token is a string."""
        mock_creds = Mock()
        mock_creds.token = "fake-oauth-token-string"
        mock_creds.valid = True
        with patch("google.auth.default", return_value=(mock_creds, "test-project")):
            yield mock_creds

    # ==========================================
    # 🟢 BASIC PATTERNS (Should PASS)
    # ==========================================

    @respx.mock
    def test_gemini_api_mode_creates_span(self, gemini_response_json, in_memory_span_exporter):
        """Test 1: Basic Developer Mode (API Key) - Should PASS."""
        respx.post(url__regex=r"https://generativelanguage\.googleapis\.com/.*").mock(
            return_value=httpx.Response(200, json=gemini_response_json)
        )

        client = genai.Client(api_key="fake-google-key")
        response = client.models.generate_content(model="gemini-2.0-flash", contents="Say hello")

        assert response.text == "Hello from Gemini 2.0!"

        time.sleep(0.1)
        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) >= 1, "Basic Gemini span missing"

        # Verify provider attribute
        span = spans[0]
        # Note: Check if your SDK sets 'google' or 'gemini'
        assert span.attributes.get("llm.system") in ["google", "gemini", "google_genai"]

    @respx.mock
    def test_gemini_vertex_mode_creates_span(
        self, gemini_response_json, in_memory_span_exporter, mock_google_auth
    ):
        """Test 2: Enterprise Vertex Mode - Should PASS."""
        respx.post(url__regex=r"https://.*-aiplatform\.googleapis\.com/.*").mock(
            return_value=httpx.Response(200, json=gemini_response_json)
        )

        client = genai.Client(vertexai=True, project="test-project", location="us-central1")
        response = client.models.generate_content(model="gemini-2.0-flash", contents="Hi Vertex")

        assert response.text == "Hello from Gemini 2.0!"

        time.sleep(0.1)
        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) >= 1, "Vertex span missing"

    # ==========================================
    # 🔴 COMPLEX PATTERNS (Likely FAIL - Bug Proof)
    # ==========================================

    @respx.mock
    def test_gemini_multipart_content_list(self, gemini_response_json, in_memory_span_exporter):
        """
        Test 3: Multi-part Content (List of strings).
        Expected: FAIL (IndexError/Crash) because SDK expects string only.
        """
        respx.post(url__regex=r"https://generativelanguage\.googleapis\.com/.*").mock(
            return_value=httpx.Response(200, json=gemini_response_json)
        )

        client = genai.Client(api_key="fake")

        # Passing a LIST instead of STRING
        complex_content = ["Context: You are a bot.", "Question: Who are you?"]

        try:
            client.models.generate_content(model="gemini-2.0-flash", contents=complex_content)
        except Exception:
            pass  # We catch crash to see if span was created

        time.sleep(0.1)
        spans = in_memory_span_exporter.get_finished_spans()

        # BUG CHECK
        if len(spans) == 0:
            pytest.fail("CRITICAL: Gemini failed to track Multi-part (List) content.")

        # Verify input was flattened
        input_val = spans[0].attributes.get("input.value")
        assert "Who are you?" in str(input_val), "Input list was not captured correctly"

    @respx.mock
    def test_gemini_tool_calling(self, in_memory_span_exporter):
        """
        Test 4: Tool Calling.
        Expected: FAIL (Missing attributes or No Span).
        """
        # Mock Tool Response (Content is often empty/null, FunctionCall is populated)
        tool_response = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"functionCall": {"name": "get_weather", "args": {"city": "Mumbai"}}}
                        ],
                        "role": "model",
                    },
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {"totalTokenCount": 10},
        }

        respx.post(url__regex=r"https://generativelanguage\.googleapis\.com/.*").mock(
            return_value=httpx.Response(200, json=tool_response)
        )

        client = genai.Client(api_key="fake")

        # We don't need real tools def here, just simulating the response parsing
        client.models.generate_content(
            model="gemini-2.0-flash",
            contents="Weather?",
            config={"tools": [{"function_declarations": [{"name": "get_weather"}]}]},
        )

        time.sleep(0.1)
        spans = in_memory_span_exporter.get_finished_spans()

        # BUG CHECK
        if len(spans) == 0:
            pytest.fail("CRITICAL: No span created for Gemini Tool Call.")

        span = spans[0]
        # Check if function name captured
        has_tool = "get_weather" in str(
            span.attributes.get("llm.output_messages")
        ) or "get_weather" in str(span.attributes.get("llm.request.functions"))

        assert has_tool, "Tool execution details missing from span attributes"
