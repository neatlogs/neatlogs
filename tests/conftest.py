"""
Pytest Configuration and Fixtures
=================================
Shared fixtures for Neatlogs tests.
"""

import os
from unittest.mock import Mock, patch

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


@pytest.fixture
def mock_openai_client():
    """Create a mock OpenAI client."""
    with patch("openai.OpenAI") as mock:
        yield mock


@pytest.fixture
def mock_openai_response():
    """Create a mock OpenAI chat completion response."""
    response = Mock()
    response.choices = [Mock()]
    response.choices[0].message.content = "Hello! How can I help you today?"
    response.choices[0].message.role = "assistant"
    response.model = "gpt-3.5-turbo"
    response.usage.prompt_tokens = 20
    response.usage.completion_tokens = 10
    response.usage.total_tokens = 30
    return response


@pytest.fixture
def mock_gemini_response():
    """Create a mock Gemini response."""
    response = Mock()
    # Gemini response format
    candidate = Mock()
    candidate.content.parts = [Mock(text="I am Gemini, helpful AI.")]
    candidate.content.role = "model"

    response.text = "I am Gemini, helpful AI."
    response.candidates = [candidate]

    # Mock usage metadata usually comes in usage_metadata
    response.usage_metadata.prompt_token_count = 15
    response.usage_metadata.candidates_token_count = 10
    response.usage_metadata.total_token_count = 25

    return response


@pytest.fixture
def in_memory_span_exporter():
    """Create an in-memory span exporter for testing."""
    return InMemorySpanExporter()


@pytest.fixture
def tracer_provider(in_memory_span_exporter):
    """Create a tracer provider with in-memory exporter."""
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(in_memory_span_exporter))
    return provider


@pytest.fixture(autouse=True)
def _disable_export_for_unit_tests(monkeypatch):
    """
    Ensure tests never attempt to call a real Neatlogs backend.

    Tests should assert on spans in-memory, not on HTTP side effects.
    """
    monkeypatch.setenv("NEATLOGS_DISABLE_EXPORT", "true")


@pytest.fixture(autouse=True)
def reset_neatlogs_and_otel_state():
    """
    Reset Neatlogs + OpenTelemetry globals between tests (Langfuse-style).

    Many OTel SDK globals are "set-once"; tests need a clean slate to set a new
    TracerProvider and capture spans deterministically.
    """
    from opentelemetry import trace

    import neatlogs

    # Hard reset OTel "set once" provider state to allow tests to set their own provider.
    trace._TRACER_PROVIDER = None
    trace._TRACER_PROVIDER_SET_ONCE = trace.Once()

    # Reset Neatlogs SDK state if a previous test initialized it.
    try:
        neatlogs.shutdown()
    except Exception:
        pass

    yield

    # Cleanup after test
    try:
        neatlogs.shutdown()
    except Exception:
        pass

    trace._TRACER_PROVIDER = None
    trace._TRACER_PROVIDER_SET_ONCE = trace.Once()


@pytest.fixture
def mock_neatlogs_server():
    """Mock the Neatlogs server endpoint."""
    with patch("requests.post") as mock_post:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "success"}
        mock_post.return_value = mock_response
        yield mock_post


def _noop():
    # Keep file ending stable for tooling.
    return None
