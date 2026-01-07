"""
Google GenAI Instrumentation Audit
==================================
Tests observability for Google's native GenAI SDK (Gemini).
This is the underlying engine for 'google-adk'.
"""

import pytest
import respx
import httpx
import os
import time
import logging
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)

# Mock Keys
os.environ["GOOGLE_API_KEY"] = "fake-key"

# --- IMPORTS ---
try:
    from google import genai
    from google.genai import types
except ImportError:
    pytest.skip("google-genai not installed. Run: uv add google-genai", allow_module_level=True)

class TestGoogleGenAIInstrumentation:
    
    @pytest.fixture(autouse=True)
    def setup_teardown(self, in_memory_span_exporter):
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(in_memory_span_exporter))
        trace.set_tracer_provider(provider)
        
        import neatlogs
        # Instrument the specific Google GenAI library
        neatlogs.init(
            api_key="test-key", 
            instrumentations=["google-genai"], 
            enable_otel=True
        )
        
        self.exporter = in_memory_span_exporter
        yield
        self.exporter.clear()

    def wait_for_spans(self, min_spans=1, timeout=3.0):
        start = time.time()
        while time.time() - start < timeout:
            spans = self.exporter.get_finished_spans()
            if len(spans) >= min_spans:
                return spans
            time.sleep(0.1)
        return self.exporter.get_finished_spans()

    # =================================================================
    # 🟢 PATTERN 1: BASIC GENERATE CONTENT
    # =================================================================
    @respx.mock
    def test_gemini_generate_content(self):
        """
        Scenario: Basic sync generation call.
        """
        # Mock the Google API endpoint
        respx.post(url__regex=r"https://generativelanguage.googleapis.com.*").mock(
            return_value=httpx.Response(
                200, 
                json={
                    "candidates": [{
                        "content": {"parts": [{"text": "Hello from Gemini"}]},
                        "finishReason": "STOP",
                        "index": 0
                    }],
                    "usageMetadata": {"totalTokenCount": 10}
                }
            )
        )

        client = genai.Client(api_key="fake-key")
        
        # ACT
        response = client.models.generate_content(
            model="gemini-2.0-flash-exp", 
            contents="Say hi"
        )
        
        assert "Gemini" in response.text

        # ASSERT SPANS
        spans = self.wait_for_spans(min_spans=1)
        
        # Check for GenAI attributes
        llm_span = next((s for s in spans if "generate_content" in s.name), None)
        assert llm_span is not None, "Google GenAI span missing"
        
        # Verify attributes
        assert llm_span.attributes.get("gen_ai.system") == "google"
        assert "gemini" in llm_span.attributes.get("gen_ai.request.model")

    # =================================================================
    # 🔥 PATTERN 2: STREAMING
    # =================================================================
    @respx.mock
    def test_gemini_streaming(self):
        """
        Scenario: Streaming response aggregation.
        """
        # Mock Streaming Response (Server-Sent Events / JSON chunks)
        # Note: Google API often returns a list of JSON objects in stream
        chunk_1 = {"candidates": [{"content": {"parts": [{"text": "Streaming "}]}}]}
        chunk_2 = {"candidates": [{"content": {"parts": [{"text": "is working"}]}}]}
        
        # Respx expects an iterable for streaming
        mock_content = [
            json.dumps(chunk_1).encode("utf-8"),
            json.dumps(chunk_2).encode("utf-8")
        ]

        respx.post(url__regex=r"https://generativelanguage.googleapis.com.*").mock(
            return_value=httpx.Response(200, content=iter(mock_content))
        )

        client = genai.Client(api_key="fake-key")
        
        # ACT
        full_text = ""
        for chunk in client.models.generate_content_stream(model="gemini-1.5-pro", contents="Stream me"):
            full_text += chunk.text

        assert "working" in full_text

        # ASSERT SPANS
        spans = self.wait_for_spans(min_spans=1)
        stream_span = next((s for s in spans if "stream" in s.name), None)
        assert stream_span is not None

    # =================================================================
    # ⚡ PATTERN 3: ASYNC EXECUTION
    # =================================================================
    @respx.mock
    @pytest.mark.asyncio
    async def test_gemini_async(self):
        """
        Scenario: Async generate content.
        """
        respx.post(url__regex=r"https://generativelanguage.googleapis.com.*").mock(
            return_value=httpx.Response(
                200, 
                json={
                    "candidates": [{"content": {"parts": [{"text": "Async Hi"}]}}]
                }
            )
        )

        client = genai.Client(api_key="fake-key")
        
        # ACT
        response = await client.aio.models.generate_content(
            model="gemini-1.5-flash", 
            contents="Async test"
        )
        
        assert "Async" in response.text

        # ASSERT SPANS
        spans = self.wait_for_spans(min_spans=1)
        assert len(spans) > 0