"""
Tests for Google GenAI (Gemini) Instrumentation
==============================================
Tests for the modern google-genai SDK.
"""

import time
from unittest.mock import Mock, patch

import httpx
import pytest
import respx

# Import the new SDK
from google import genai
from google.genai import types
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor


class TestGoogleGenAIInstrumentation:
    """Test suite for the new Google GenAI SDK instrumentation."""

    @pytest.fixture
    def gemini_response_json(self):
        """Mock Gemini response JSON."""
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
        """
        Fixes TypeError in httpx by ensuring the credential token is a string, not a Mock.
        """
        mock_creds = Mock()
        # The SDK accesses .token or calls refresh. We make sure it behaves strings.
        mock_creds.token = "fake-oauth-token-string"
        mock_creds.valid = True
        mock_creds.quota_project_id = None

        with (
            patch("google.auth.default", return_value=(mock_creds, "test-project")),
            patch(
                "google.genai._api_client.get_token_from_credentials",
                return_value="fake-oauth-token-string",
            ),
        ):
            yield mock_creds

    @respx.mock
    def test_gemini_api_mode_creates_span(self, gemini_response_json, in_memory_span_exporter):
        """Test Gemini API (Developer Mode) creates LLM span."""

        # 1. ARRANGE
        api_url = r"https://generativelanguage\.googleapis\.com/.*"
        respx.post(url__regex=api_url).mock(
            return_value=httpx.Response(200, json=gemini_response_json)
        )

        # Setup Clean OTEL
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(in_memory_span_exporter))

        import neatlogs

        neatlogs.init(
            api_key="test-key",
            instrumentations=["google_genai"],
            tracer_provider=provider,
        )

        # 2. ACT
        client = genai.Client(api_key="fake-google-key")
        response = client.models.generate_content(model="gemini-2.0-flash", contents="Say hello")

        assert response.text == "Hello from Gemini 2.0!"

        # 3. ASSERT
        time.sleep(0.1)
        spans = in_memory_span_exporter.get_finished_spans()

        llm_spans = [s for s in spans if s.attributes.get("neatlogs.span.kind") == "llm"]

        assert len(llm_spans) >= 1
        span = llm_spans[0]

        assert span.attributes.get("neatlogs.llm.provider") == "google_genai"
        assert "gemini" in str(span.attributes)
        assert span.attributes.get("neatlogs.llm.token_count.total") == 20

    @respx.mock
    def test_gemini_vertex_mode_creates_span(
        self, gemini_response_json, in_memory_span_exporter, mock_google_auth
    ):
        """
        Test Gemini Vertex AI (Enterprise Mode) creates LLM span.
        Uses mock_google_auth fixture to fix the header string error.
        """

        # 1. ARRANGE
        vertex_url = r"https://.*-aiplatform\.googleapis\.com/.*"
        respx.post(url__regex=vertex_url).mock(
            return_value=httpx.Response(200, json=gemini_response_json)
        )

        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(in_memory_span_exporter))

        import neatlogs

        neatlogs.init(
            api_key="test-key",
            instrumentations=["google_genai"],
            tracer_provider=provider,
        )

        # 2. ACT
        # The client will use our mocked google.auth.default credentials
        client = genai.Client(vertexai=True, project="test-project", location="us-central1")

        response = client.models.generate_content(model="gemini-2.0-flash", contents="Hi Vertex")

        # 3. ASSERT
        assert response.text == "Hello from Gemini 2.0!"

        time.sleep(0.1)
        spans = in_memory_span_exporter.get_finished_spans()
        llm_span = next(s for s in spans if s.attributes.get("neatlogs.span.kind") == "llm")

        assert llm_span.attributes.get("neatlogs.llm.provider") == "google_genai"

    @respx.mock
    def test_gemini_streaming_capture(self, in_memory_span_exporter):
        """
        Test that streaming calls also generate spans.
        FIXED: Uses client.models.generate_content_stream()
        """
        # Streaming response format (SSE-like JSON array often)
        # We simulate a simple stream response body
        # Note: respx mocking of streams is tricky, we return a list which httpx iterates
        chunk1 = b'{"candidates": [{"content": {"parts": [{"text": "Hello "}]}}]}\n'
        chunk2 = b'{"candidates": [{"content": {"parts": [{"text": "World"}]}}]}\n'

        respx.post(url__regex=r"https://generativelanguage.*").mock(
            return_value=httpx.Response(200, content=chunk1 + chunk2)
        )

        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(in_memory_span_exporter))

        import neatlogs

        neatlogs.init(
            api_key="test-key",
            instrumentations=["google_genai"],
            tracer_provider=provider,
        )

        client = genai.Client(api_key="fake")

        # FIXED: Use the specific streaming method instead of config={'stream': True}
        response_stream = client.models.generate_content_stream(
            model="gemini-2.0-flash", contents="Stream this"
        )

        # Consume stream
        full_text = ""
        for chunk in response_stream:
            full_text += chunk.text

        assert "Hello World" in full_text

        time.sleep(0.1)
        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) >= 1
