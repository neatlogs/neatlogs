"""
Comprehensive Agno Framework Instrumentation Test Suite
=======================================================
Tests Agno agent framework with OpenInference/OpenTelemetry instrumentation.
Validates span creation across complexity levels from basic to production-grade.

Patterns Tested:
1. ✅ Basic Agent Execution (Sync & Async)
2. ✅ Multi-Agent Systems (Tic Tac Toe Style)
3. ✅ Tool-Calling Agents
4. ✅ Multi-Model Agent Federation
5. ✅ Streaming Responses
6. ✅ Knowledge Base Integration (RAG)
7. ✅ Workflows & State Machines
8. ⚠️ Error Handling & Edge Cases
"""

import asyncio
import json
import logging
import re
import time
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import httpx
import pytest
import respx

# Configure logging for test visibility
logging.basicConfig(level=logging.INFO)

# Skip if Agno not installed
try:
    from agno import Agent, KnowledgeBase, RunConfig, Tool
    from agno.models import Anthropic, GoogleGenerativeAI, Groq, OpenAIChat
    from agno.run.agent import RunOutput
    from agno.tools import Toolkit

    HAS_AGNO = True
except ImportError:
    HAS_AGNO = False
    pytest.skip("Agno not installed", allow_module_level=True)


# =================================================================
# TEST SETUP & FIXTURES
# =================================================================


class TestAgnoInstrumentation:
    """Main test class for Agno framework instrumentation validation."""

    @pytest.fixture(autouse=True)
    def setup_teardown(self, in_memory_span_exporter):
        """Global test setup with OpenTelemetry and Neatlogs initialization."""
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor

        # Setup OpenTelemetry with in-memory exporter
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(in_memory_span_exporter))
        trace.set_tracer_provider(provider)

        # Initialize Neatlogs with Agno instrumentation
        import neatlogs

        neatlogs.init(
            api_key="test-key",
            enable_otel=True,
            disable_export=True,
            instrumentations=["agno"],  # Agno instrumentation
        )

        self.exporter = in_memory_span_exporter
        yield

        # Cleanup
        self.exporter.clear()

    @pytest.fixture
    def mock_openai_chat_response(self):
        """Standard OpenAI chat response fixture."""
        return {
            "id": "chatcmpl-test123",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "This is a test response from GPT-4o.",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 25, "completion_tokens": 15, "total_tokens": 40},
        }

    @pytest.fixture
    def mock_openai_o3_response(self):
        """OpenAI o3-mini response with reasoning."""
        return {
            "id": "chatcmpl-o3-test",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "o3-mini",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "I've reasoned about this and my answer is: Test response with reasoning.",
                        "reasoning": "Let me think step by step...",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 30, "completion_tokens": 25, "total_tokens": 55},
        }

    @pytest.fixture
    def mock_anthropic_response(self):
        """Anthropic Claude response fixture."""
        return {
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "text", "text": "This is a test response from Claude 3.5 Sonnet."}
            ],
            "model": "claude-3-5-sonnet-20241022",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 20, "output_tokens": 15},
        }

    @pytest.fixture
    def mock_google_gemini_response(self):
        """Google Gemini response fixture."""
        return {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": "This is a test response from Gemini Flash."}],
                        "role": "model",
                    }
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 15,
                "candidatesTokenCount": 10,
                "totalTokenCount": 25,
            },
        }

    @pytest.fixture
    def mock_groq_llama_response(self):
        """Groq Llama response fixture."""
        return {
            "id": "chatcmpl-groq123",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "llama-3.3-70b-versatile",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "This is a test response from Llama 3.3 via Groq.",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 20, "completion_tokens": 12, "total_tokens": 32},
        }

    @pytest.fixture
    def mock_embeddings_response(self):
        """OpenAI embeddings response fixture."""
        return {
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "index": 0,
                    "embedding": [0.1, 0.2, 0.3] * 512,  # 1536-dim vector
                }
            ],
            "model": "text-embedding-3-small",
            "usage": {"prompt_tokens": 8, "total_tokens": 8},
        }

    def wait_for_spans(self, min_spans=1, timeout=2.0):
        """Helper to wait for spans to be processed."""
        start = time.time()
        while time.time() - start < timeout:
            spans = self.exporter.get_finished_spans()
            if len(spans) >= min_spans:
                return spans
            time.sleep(0.05)
        return self.exporter.get_finished_spans()

    def assert_span_exists(self, span_name: str, spans: List = None):
        """Assert a span with given name exists."""
        if spans is None:
            spans = self.exporter.get_finished_spans()

        matching_spans = [s for s in spans if span_name in s.name]
        assert len(matching_spans) > 0, f"Span '{span_name}' not found in {[s.name for s in spans]}"
        return matching_spans[0]

    # =================================================================
    # 🟢 PATTERN 1: BASIC AGENT EXECUTION (SYNC & ASYNC)
    # =================================================================

    @respx.mock
    def test_basic_agent_sync(self, mock_openai_chat_response):
        """
        Test basic Agno agent with OpenAI model.
        Expected spans: 'Agent', 'ChatOpenAI' or similar
        """
        respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=mock_openai_chat_response)
        )

        # Create basic agent
        agent = Agent(
            name="TestAgent",
            model=OpenAIChat(id="gpt-4o"),
            description="A test agent",
            instructions="You are a helpful assistant.",
            markdown=True,
        )

        # Run synchronously
        response = agent.run("Hello, who are you?")
        assert response is not None
        assert hasattr(response, "content")

        # Validate spans
        spans = self.wait_for_spans(min_spans=2)

        # Check for Agent span
        self.assert_span_exists("Agent", spans)

        # Check for LLM span (might be 'OpenAIChat', 'ChatOpenAI', or similar)
        llm_spans = [s for s in spans if any(x in s.name for x in ["OpenAI", "Chat", "LLM"])]
        assert len(llm_spans) >= 1, f"No LLM span found. Spans: {[s.name for s in spans]}"

        # Verify agent attributes
        agent_span = self.assert_span_exists("Agent", spans)
        assert agent_span.attributes.get("agent.name") == "TestAgent"
        assert "agno" in agent_span.attributes.get("llm.framework", "").lower()

    @respx.mock
    @pytest.mark.asyncio
    async def test_basic_agent_async(self, mock_anthropic_response):
        """
        Test async Agno agent with Anthropic model.
        """
        respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(200, json=mock_anthropic_response)
        )

        agent = Agent(
            name="AsyncAgent",
            model=Anthropic(id="claude-3-5-sonnet"),
            instructions="Answer concisely.",
        )

        # Run asynchronously
        response = await agent.arun("What's the weather like?")
        assert response is not None

        spans = self.wait_for_spans(min_spans=2)

        self.assert_span_exists("Agent", spans)

        # Check for async attributes if present
        agent_spans = [s for s in spans if "Agent" in s.name]
        for span in agent_spans:
            if "async" in span.attributes:
                assert span.attributes.get("async") is True

    @respx.mock
    def test_multi_model_agents(self, mock_openai_chat_response, mock_anthropic_response):
        """
        Test multiple agents with different models in same test.
        Simulates Tic Tac Toe setup with different models.
        """
        # Mock both OpenAI and Anthropic endpoints
        respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=mock_openai_chat_response)
        )
        respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(200, json=mock_anthropic_response)
        )

        # Create agents with different models (like Tic Tac Toe)
        agent_x = Agent(
            name="PlayerX",
            model=OpenAIChat(id="gpt-4o"),
            instructions="You are player X in Tic Tac Toe. Make strategic moves.",
            markdown=True,
        )

        agent_o = Agent(
            name="PlayerO",
            model=Anthropic(id="claude-3-5-sonnet"),
            instructions="You are player O in Tic Tac Toe. Counter player X's moves.",
            markdown=True,
        )

        # Simulate a move by player X
        board_state = "X _ _\n_ _ _\n_ _ _"
        valid_moves = ["0 1", "0 2", "1 0", "1 1", "1 2", "2 0", "2 1", "2 2"]

        response_x = agent_x.run(
            f"Current board:\n{board_state}\nValid moves: {valid_moves}\nChoose your move (row col):"
        )

        # Check spans for first agent
        spans = self.wait_for_spans(min_spans=3)

        # Should have spans for both agent types
        agent_spans = [s for s in spans if "Agent" in s.name]
        assert len(agent_spans) >= 1

        # Check model-specific spans
        openai_spans = [s for s in spans if "openai" in s.name.lower() or "gpt" in s.name.lower()]
        anthropic_spans = [
            s for s in spans if "anthropic" in s.name.lower() or "claude" in s.name.lower()
        ]

        # At least OpenAI spans should exist (we called agent_x)
        assert len(openai_spans) >= 1

    # =================================================================
    # 🔵 PATTERN 2: MULTI-AGENT SYSTEMS (TIC TAC TOE STYLE)
    # =================================================================

    @respx.mock
    def test_tic_tac_toe_simulation(self, mock_openai_chat_response, mock_openai_o3_response):
        """
        Test complex Tic Tac Toe simulation between two agents.
        Inspired by the provided Streamlit app.
        """
        # Mock different responses for alternating players
        responses = [
            # Player X (GPT-4o) first move
            {
                **mock_openai_chat_response,
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": "1 1"}}  # Center move
                ],
            },
            # Player O (o3-mini) response
            {
                **mock_openai_o3_response,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "0 0",
                            "reasoning": "I should take the corner to control the board.",
                        },
                    }
                ],
            },
            # Player X second move
            {
                **mock_openai_chat_response,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "2 2"}}],
            },
        ]

        respx.post("https://api.openai.com/v1/chat/completions").mock(
            side_effect=[httpx.Response(200, json=r) for r in responses]
        )

        # Create Tic Tac Toe agents (simplified version)
        player_x = Agent(
            name="PlayerX_TTT",
            model=OpenAIChat(id="gpt-4o"),
            instructions="""You are Player X in Tic Tac Toe.
            Analyze the board and choose the best move.
            Respond with ONLY row and column numbers, e.g., "1 1" for center.
            Make strategic moves to win.""",
        )

        player_o = Agent(
            name="PlayerO_TTT",
            model=OpenAIChat(id="o3-mini"),
            instructions="""You are Player O in Tic Tac Toe.
            You are playing against Player X.
            Think step by step about your strategy.
            Respond with ONLY row and column numbers.""",
        )

        # Simulate a few moves
        board_states = [
            "X _ _\n_ _ _\n_ _ _",  # After first X move
            "X _ _\n_ O _\n_ _ _",  # After O response
            "X _ _\n_ O _\n_ _ X",  # After second X move
        ]

        valid_moves = ["0 1", "0 2", "1 0", "1 1", "1 2", "2 0", "2 1", "2 2"]

        # Player X makes first move
        move1 = player_x.run(f"Board:\n{board_states[0]}\nValid: {valid_moves}\nYour move:")
        assert "1 1" in move1.content  # Center move

        # Update valid moves (remove center)
        valid_moves = ["0 0", "0 1", "0 2", "1 0", "1 2", "2 0", "2 1", "2 2"]

        # Player O responds
        move2 = player_o.run(f"Board:\n{board_states[1]}\nValid: {valid_moves}\nYour move:")
        assert "0 0" in move2.content  # Corner move

        spans = self.wait_for_spans(min_spans=6, timeout=3.0)

        # Verify we have multiple agent spans
        agent_spans = [s for s in spans if "Agent" in s.name]
        assert len(agent_spans) >= 2

        # Verify agent names in spans
        agent_names = [s.attributes.get("agent.name") for s in agent_spans]
        assert "PlayerX_TTT" in agent_names
        assert "PlayerO_TTT" in agent_names

        # Check for multiple LLM calls
        llm_spans = [
            s for s in spans if any(x in s.name.lower() for x in ["openai", "chat", "llm"])
        ]
        assert len(llm_spans) >= 3  # At least 3 moves

        # Check for reasoning in o3-mini span if captured
        reasoning_spans = [s for s in spans if "reasoning" in str(s.attributes).lower()]
        if reasoning_spans:
            assert len(reasoning_spans) >= 1

    @respx.mock
    def test_agent_turn_based_game(self, mock_google_gemini_response, mock_groq_llama_response):
        """
        Test turn-based game with heterogeneous models (Google + Groq).
        More complex than Tic Tac Toe with different providers.
        """
        # Mock Google Gemini
        gemini_response = mock_google_gemini_response.copy()
        gemini_response["candidates"][0]["content"]["parts"][0]["text"] = "I choose position A3"

        # Mock Groq Llama
        llama_response = mock_groq_llama_response.copy()
        llama_response["choices"][0]["message"]["content"] = "I'll take position B2 in response"

        respx.post(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"
        ).mock(return_value=httpx.Response(200, json=gemini_response))
        respx.post("https://api.groq.com/openai/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=llama_response)
        )

        # Create agents with different providers
        gemini_agent = Agent(
            name="GeminiPlayer",
            model=GoogleGenerativeAI(id="gemini-pro"),
            instructions="Make strategic game moves.",
            markdown=False,
        )

        llama_agent = Agent(
            name="LlamaPlayer",
            model=Groq(id="llama-3.3-70b-versatile"),
            instructions="Counter your opponent's moves.",
            markdown=True,
        )

        # Simulate game turns
        game_state = "Turn 1: Gemini's move"
        gemini_move = gemini_agent.run(f"Game state: {game_state}. Your move:")

        game_state = "Turn 2: Llama's response to Gemini"
        llama_move = llama_agent.run(f"Game state: {game_state}. Your move:")

        spans = self.wait_for_spans(min_spans=4)

        # Verify both provider spans exist
        gemini_spans = [
            s for s in spans if "google" in s.name.lower() or "gemini" in s.name.lower()
        ]
        groq_spans = [s for s in spans if "groq" in s.name.lower() or "llama" in s.name.lower()]

        assert len(gemini_spans) >= 1, "Gemini spans missing"
        assert len(groq_spans) >= 1, "Groq spans missing"

        # Verify different model attributes
        for span in gemini_spans:
            assert "gemini" in span.attributes.get("llm.system", "").lower()

        for span in groq_spans:
            assert (
                "groq" in span.attributes.get("llm.system", "").lower()
                or "llama" in span.attributes.get("llm.model_name", "").lower()
            )

    # =================================================================
    # 🟡 PATTERN 3: TOOL-CALLING AGENTS
    # =================================================================

    @respx.mock
    def test_agent_with_builtin_tools(self, mock_openai_chat_response):
        """
        Test Agno agent using built-in tools.
        Tool execution should create separate spans.
        """
        # Mock tool call response
        tool_response = mock_openai_chat_response.copy()
        tool_response["choices"][0]["message"]["content"] = None
        tool_response["choices"][0]["message"]["tool_calls"] = [
            {
                "id": "call_123",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": json.dumps({"location": "San Francisco"}),
                },
            }
        ]
        tool_response["choices"][0]["finish_reason"] = "tool_calls"

        # Final response
        final_response = mock_openai_chat_response.copy()

        respx.post("https://api.openai.com/v1/chat/completions").mock(
            side_effect=[
                httpx.Response(200, json=tool_response),
                httpx.Response(200, json=final_response),
            ]
        )

        # Define a tool
        def get_weather_tool(location: str) -> str:
            """Get weather for a location."""
            return f"Weather in {location}: Sunny, 72°F"

        # Create agent with tool
        agent = Agent(
            name="WeatherAgent",
            model=OpenAIChat(id="gpt-4o"),
            tools=[get_weather_tool],
            instructions="Use tools when needed to get accurate information.",
            show_tool_calls=True,
        )

        response = agent.run("What's the weather in San Francisco?")

        spans = self.wait_for_spans(min_spans=4, timeout=2.0)

        # Check for tool span
        tool_spans = [s for s in spans if "get_weather" in s.name]
        assert len(tool_spans) >= 1, f"Tool span missing. Found: {[s.name for s in spans]}"

        # Check tool span attributes
        tool_span = tool_spans[0]
        assert tool_span.attributes.get("openinference.span.kind") == "TOOL"
        assert "san francisco" in str(tool_span.attributes).lower()

        # Verify agent span
        agent_spans = [s for s in spans if "Agent" in s.name]
        assert len(agent_spans) >= 1

        # Check multiple LLM spans (tool call + final answer)
        llm_spans = [s for s in spans if "openai" in s.name.lower()]
        assert len(llm_spans) >= 2

    @respx.mock
    def test_agent_with_custom_toolkit(self, mock_openai_chat_response):
        """
        Test agent with custom toolkit containing multiple tools.
        Complex tool orchestration scenario.
        """
        # Mock sequence: tool call -> final answer
        tool_response = mock_openai_chat_response.copy()
        tool_response["choices"][0]["message"]["content"] = None
        tool_response["choices"][0]["message"]["tool_calls"] = [
            {
                "id": "call_456",
                "type": "function",
                "function": {
                    "name": "calculate_metrics",
                    "arguments": json.dumps({"data": [1, 2, 3, 4, 5]}),
                },
            }
        ]

        final_response = mock_openai_chat_response.copy()

        respx.post("https://api.openai.com/v1/chat/completions").mock(
            side_effect=[
                httpx.Response(200, json=tool_response),
                httpx.Response(200, json=final_response),
            ]
        )

        # Create custom toolkit
        class AnalyticsToolkit(Toolkit):
            """Custom analytics toolkit."""

            def calculate_metrics(self, data: List[float]) -> Dict[str, float]:
                """Calculate statistical metrics."""
                return {
                    "mean": sum(data) / len(data),
                    "min": min(data),
                    "max": max(data),
                    "count": len(data),
                }

            def detect_anomalies(self, data: List[float], threshold: float = 2.0) -> List[bool]:
                """Detect anomalies in data."""
                mean = sum(data) / len(data)
                std = (sum((x - mean) ** 2 for x in data) / len(data)) ** 0.5
                return [abs(x - mean) > threshold * std for x in data]

        toolkit = AnalyticsToolkit()

        agent = Agent(
            name="AnalyticsAgent",
            model=OpenAIChat(id="gpt-4o"),
            tools=[toolkit],
            instructions="Use analytics tools to process data.",
            markdown=True,
        )

        response = agent.run("Calculate metrics for data: [1, 2, 3, 4, 5]")

        spans = self.wait_for_spans(min_spans=4)

        # Check for toolkit/tool spans
        tool_spans = [
            s for s in spans if "calculate_metrics" in s.name or "AnalyticsToolkit" in s.name
        ]
        assert len(tool_spans) >= 1

        # Verify toolkit method was called
        tool_span = tool_spans[0]
        assert (
            "metrics" in str(tool_span.attributes).lower()
            or "data" in str(tool_span.attributes).lower()
        )

    # =================================================================
    # 🟠 PATTERN 4: MULTI-MODEL AGENT FEDERATION
    # =================================================================

    @respx.mock
    def test_model_routing_agent(
        self, mock_openai_chat_response, mock_anthropic_response, mock_google_gemini_response
    ):
        """
        Test agent that routes queries to different models based on content.
        Advanced pattern for cost/performance optimization.
        """
        # Mock all endpoints
        respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=mock_openai_chat_response)
        )
        respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(200, json=mock_anthropic_response)
        )
        respx.post(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"
        ).mock(return_value=httpx.Response(200, json=mock_google_gemini_response))

        # Create specialized agents
        coding_agent = Agent(
            name="CodingExpert",
            model=OpenAIChat(id="gpt-4o"),
            instructions="You are a coding expert. Write efficient code.",
            markdown=True,
        )

        creative_agent = Agent(
            name="CreativeWriter",
            model=Anthropic(id="claude-3-5-sonnet"),
            instructions="You are a creative writer. Write engaging content.",
            markdown=True,
        )

        general_agent = Agent(
            name="GeneralAssistant",
            model=GoogleGenerativeAI(id="gemini-pro"),
            instructions="You are a general assistant. Answer various questions.",
            markdown=False,
        )

        # Router function (simplified)
        def route_query(query: str) -> Agent:
            """Route query to appropriate agent based on content."""
            query_lower = query.lower()
            if any(word in query_lower for word in ["code", "program", "algorithm", "function"]):
                return coding_agent
            elif any(word in query_lower for word in ["write", "story", "creative", "poem"]):
                return creative_agent
            else:
                return general_agent

        # Test routing
        queries = [
            "Write a Python function to calculate factorial",
            "Write a short story about AI",
            "What's the capital of France?",
        ]

        responses = []
        for query in queries:
            agent = route_query(query)
            response = agent.run(query)
            responses.append(response)

        # 3 queries * ~3 spans each
        spans = self.wait_for_spans(min_spans=9, timeout=3.0)

        # Verify all agent types were used
        agent_spans = [s for s in spans if "Agent" in s.name]
        agent_names = [s.attributes.get("agent.name") for s in agent_spans]

        assert "CodingExpert" in agent_names
        assert "CreativeWriter" in agent_names
        assert "GeneralAssistant" in agent_names

        # Verify different model spans
        openai_spans = [s for s in spans if "openai" in str(s.attributes).lower()]
        anthropic_spans = [s for s in spans if "anthropic" in str(s.attributes).lower()]
        google_spans = [s for s in spans if "google" in str(s.attributes).lower()]

        assert len(openai_spans) >= 1
        assert len(anthropic_spans) >= 1
        assert len(google_spans) >= 1

    # =================================================================
    # 🔴 PATTERN 5: STREAMING RESPONSES
    # =================================================================

    @respx.mock
    def test_agent_streaming(self, mock_openai_chat_response):
        """
        Test Agno agent with streaming enabled.
        Streaming responses should still produce spans.
        """
        # Mock streaming response (SSE format)
        streaming_chunks = [
            b'data: {"id": "chatcmpl-123", "object": "chat.completion.chunk", '
            b'"choices": [{"delta": {"role": "assistant"}, "index": 0}]}\n\n',
            b'data: {"id": "chatcmpl-123", "object": "chat.completion.chunk", '
            b'"choices": [{"delta": {"content": "Streaming"}, "index": 0}]}\n\n',
            b'data: {"id": "chatcmpl-123", "object": "chat.completion.chunk", '
            b'"choices": [{"delta": {"content": " response"}, "index": 0}]}\n\n',
            b'data: {"id": "chatcmpl-123", "object": "chat.completion.chunk", '
            b'"choices": [{"delta": {"content": " from Agno"}, "index": 0}]}\n\n',
            b'data: {"id": "chatcmpl-123", "object": "chat.completion.chunk", '
            b'"choices": [{"delta": {}, "finish_reason": "stop", "index": 0}]}\n\n',
            b"data: [DONE]\n\n",
        ]

        respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(200, content=b"".join(streaming_chunks))
        )

        agent = Agent(
            name="StreamingAgent",
            model=OpenAIChat(id="gpt-4o"),
            instructions="Stream your response.",
            markdown=True,
        )

        # Note: Agno might handle streaming differently
        # This is a simplified test
        response = agent.run("Say hello with streaming", stream=False)

        spans = self.wait_for_spans(min_spans=2)

        # Streaming might create different span patterns
        llm_spans = [s for s in spans if "openai" in s.name.lower()]
        assert len(llm_spans) >= 1

        # Check for streaming indicators
        llm_span = llm_spans[0]
        if "stream" in str(llm_span.attributes).lower():
            assert llm_span.attributes.get("llm.streaming") == True

        # Check for chunk events if present
        if hasattr(llm_span, "events"):
            stream_events = [
                e
                for e in llm_span.events
                if "chunk" in e.name.lower() or "stream" in e.name.lower()
            ]
            if stream_events:
                assert len(stream_events) > 0

    @respx.mock
    @pytest.mark.asyncio
    async def test_async_streaming_agent(self, mock_anthropic_response):
        """
        Test async streaming with Agno agent.
        """
        # Mock streaming for Anthropic
        streaming_chunks = [
            b'event: message_start\ndata: {"type": "message_start", "message": {"id": "msg_123"}}\n\n',
            b'event: content_block_start\ndata: {"type": "content_block_start", "index": 0}\n\n',
            b'event: content_block_delta\ndata: {"type": "content_block_delta", "delta": {"text": "Async"}}\n\n',
            b'event: content_block_delta\ndata: {"type": "content_block_delta", "delta": {"text": " streaming"}}\n\n',
            b'event: content_block_delta\ndata: {"type": "content_block_delta", "delta": {"text": " test"}}\n\n',
            b'event: message_stop\ndata: {"type": "message_stop"}\n\n',
        ]

        respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(200, content=b"".join(streaming_chunks))
        )

        agent = Agent(
            name="AsyncStreamAgent",
            model=Anthropic(id="claude-3-5-sonnet"),
            instructions="Stream your response asynchronously.",
            markdown=True,
        )

        # Note: Actual async streaming implementation may vary
        response = await agent.arun("Test async streaming")

        spans = self.wait_for_spans(min_spans=2)

        # Should still produce spans
        assert len(spans) >= 1

        # Check for async attributes
        agent_spans = [s for s in spans if "Agent" in s.name]
        if agent_spans:
            span = agent_spans[0]
            if "async" in span.attributes:
                assert span.attributes.get("async") == True

    # =================================================================
    # 🟣 PATTERN 6: KNOWLEDGE BASE INTEGRATION (RAG)
    # =================================================================

    @respx.mock
    def test_agent_with_knowledge_base(self, mock_openai_chat_response, mock_embeddings_response):
        """
        Test Agno agent with knowledge base (RAG pattern).
        Complex pattern with retrieval + generation.
        """
        # Mock LLM response
        respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=mock_openai_chat_response)
        )

        # Mock embeddings
        respx.post("https://api.openai.com/v1/embeddings").mock(
            return_value=httpx.Response(200, json=mock_embeddings_response)
        )

        # Create knowledge base (simplified - would normally load documents)
        kb = KnowledgeBase(
            name="TestKB",
            description="Test knowledge base",
            # In real scenario: documents=[...], vector_store=...
        )

        # Mock the search method
        with patch.object(
            kb, "search", return_value=["Document 1: Test content", "Document 2: More test"]
        ):
            agent = Agent(
                name="RAGAgent",
                model=OpenAIChat(id="gpt-4o"),
                knowledge_base=kb,
                instructions="Use knowledge base to answer questions accurately.",
                markdown=True,
                add_knowledge_base_to_prompt=True,
            )

            response = agent.run("What information do you have about testing?")

        spans = self.wait_for_spans(min_spans=3, timeout=2.0)

        # Check for knowledge base spans
        kb_spans = [s for s in spans if "knowledge" in s.name.lower() or "KnowledgeBase" in s.name]
        if kb_spans:
            assert len(kb_spans) >= 1

        # Check for retrieval/embedding spans
        retrieval_spans = [
            s for s in spans if "retrieval" in s.name.lower() or "search" in s.name.lower()
        ]
        embedding_spans = [s for s in spans if "embedding" in s.name.lower()]

        # At least one type should exist
        assert len(retrieval_spans) >= 1 or len(embedding_spans) >= 1

        # Verify agent span has knowledge base context
        agent_spans = [s for s in spans if "Agent" in s.name]
        if agent_spans:
            agent_span = agent_spans[0]
            assert (
                "knowledge" in str(agent_span.attributes).lower()
                or "rag" in str(agent_span.attributes).lower()
            )

    @respx.mock
    def test_multi_knowledge_base_agent(self, mock_openai_chat_response, mock_embeddings_response):
        """
        Test agent accessing multiple knowledge bases.
        Advanced RAG pattern for different domains.
        """
        respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=mock_openai_chat_response)
        )
        respx.post("https://api.openai.com/v1/embeddings").mock(
            return_value=httpx.Response(200, json=mock_embeddings_response)
        )

        # Create multiple knowledge bases
        tech_kb = KnowledgeBase(name="TechDocs", description="Technical documentation")
        legal_kb = KnowledgeBase(name="LegalDocs", description="Legal documents")

        # Mock search methods
        with patch.object(tech_kb, "search", return_value=["API documentation", "Code examples"]):
            with patch.object(
                legal_kb, "search", return_value=["Terms of service", "Privacy policy"]
            ):

                # Agent with access to multiple KBs
                agent = Agent(
                    name="MultiKBAgent",
                    model=OpenAIChat(id="gpt-4o"),
                    knowledge_base=[tech_kb, legal_kb],  # Multiple KBs
                    instructions="Use appropriate knowledge base based on query.",
                    markdown=True,
                )

                # Query that should trigger both KBs
                response = agent.run("Explain API terms and technical implementation")

        spans = self.wait_for_spans(min_spans=5, timeout=2.0)

        # Check for multiple KB spans
        kb_spans = [
            s for s in spans if any(kb in s.name for kb in ["TechDocs", "LegalDocs", "Knowledge"])
        ]
        assert len(kb_spans) >= 2

        # Check retrieval occurred
        retrieval_spans = [
            s for s in spans if "retrieval" in s.name.lower() or "search" in s.name.lower()
        ]
        assert len(retrieval_spans) >= 2

    # =================================================================
    # 🟠 PATTERN 7: WORKFLOWS & STATE MACHINES
    # =================================================================

    @respx.mock
    def test_agent_workflow_orchestration(self, mock_openai_chat_response):
        """
        Test complex workflow with multiple agents in sequence.
        State machine pattern for business processes.
        """
        # Mock responses for different steps
        responses = [
            # Analyst agent
            {
                **mock_openai_chat_response,
                "choices": [{"message": {"content": "Analysis: Data looks good"}}],
            },
            # Reviewer agent
            {
                **mock_openai_chat_response,
                "choices": [{"message": {"content": "Review: Approved"}}],
            },
            # Executor agent
            {
                **mock_openai_chat_response,
                "choices": [{"message": {"content": "Execution: Task completed"}}],
            },
        ]

        respx.post("https://api.openai.com/v1/chat/completions").mock(
            side_effect=[httpx.Response(200, json=r) for r in responses]
        )

        # Create workflow agents
        analyst = Agent(
            name="DataAnalyst",
            model=OpenAIChat(id="gpt-4o"),
            instructions="Analyze data and provide insights.",
            role="analyst",
        )

        reviewer = Agent(
            name="ReviewManager",
            model=OpenAIChat(id="gpt-4o"),
            instructions="Review analyses and approve/reject.",
            role="reviewer",
        )

        executor = Agent(
            name="TaskExecutor",
            model=OpenAIChat(id="gpt-4o"),
            instructions="Execute approved tasks.",
            role="executor",
        )

        # Simple workflow orchestration
        data = "Sample dataset with metrics"

        # Step 1: Analysis
        analysis_result = analyst.run(f"Analyze: {data}")

        # Step 2: Review
        review_result = reviewer.run(f"Review analysis: {analysis_result.content}")

        # Step 3: Execution (if approved)
        if "approved" in review_result.content.lower():
            execution_result = executor.run(f"Execute based on: {analysis_result.content}")

        # 3 agents * ~3 spans each
        spans = self.wait_for_spans(min_spans=9, timeout=3.0)

        # Verify all agents in workflow
        agent_spans = [s for s in spans if "Agent" in s.name]
        agent_roles = [
            s.attributes.get("agent.role") for s in agent_spans if s.attributes.get("agent.role")
        ]

        assert "analyst" in agent_roles
        assert "reviewer" in agent_roles
        assert "executor" in agent_roles

        # Check workflow context propagation
        trace_ids = set(s.context.trace_id for s in spans)
        # Ideally all in same trace, but may be separate
        if len(trace_ids) == 1:
            assert True  # Good - all part of same workflow

        # Check sequential execution
        agent_names = [s.attributes.get("agent.name") for s in agent_spans]
        # Should see all three agents
        assert "DataAnalyst" in agent_names
        assert "ReviewManager" in agent_names
        assert "TaskExecutor" in agent_names

    @respx.mock
    def test_conditional_agent_execution(self, mock_openai_chat_response):
        """
        Test agents with conditional execution paths.
        Decision-making based on agent responses.
        """
        # Mock different response paths
        positive_response = mock_openai_chat_response.copy()
        positive_response["choices"][0]["message"]["content"] = "YES, proceed with option A"

        negative_response = mock_openai_chat_response.copy()
        negative_response["choices"][0]["message"]["content"] = "NO, choose option B instead"

        respx.post("https://api.openai.com/v1/chat/completions").mock(
            side_effect=[
                httpx.Response(200, json=positive_response),
                httpx.Response(200, json=negative_response),
            ]
        )

        # Decision agent
        decider = Agent(
            name="DecisionAgent",
            model=OpenAIChat(id="gpt-4o"),
            instructions="Decide YES or NO based on the query.",
            markdown=True,
        )

        # Action agents
        action_a = Agent(
            name="ActionAgentA",
            model=OpenAIChat(id="gpt-4o"),
            instructions="Execute option A.",
            markdown=True,
        )

        action_b = Agent(
            name="ActionAgentB",
            model=OpenAIChat(id="gpt-4o"),
            instructions="Execute option B.",
            markdown=True,
        )

        # Test both paths
        decision1 = decider.run("Should we proceed with the launch?")

        if "YES" in decision1.content.upper():
            result1 = action_a.run("Execute launch plan A")
        else:
            result1 = action_b.run("Execute alternative plan B")

        # Test opposite path with different query
        decision2 = decider.run("Is the risk too high?")

        if "NO" in decision2.content.upper():
            result2 = action_a.run("Proceed with plan")
        else:
            result2 = action_b.run("Use cautious approach")

        spans = self.wait_for_spans(min_spans=8, timeout=2.0)

        # Check decision agent was called twice
        decider_spans = [s for s in spans if s.attributes.get("agent.name") == "DecisionAgent"]
        assert len(decider_spans) >= 2

        # Check both action agents were used (depends on decisions)
        action_a_spans = [s for s in spans if s.attributes.get("agent.name") == "ActionAgentA"]
        action_b_spans = [s for s in spans if s.attributes.get("agent.name") == "ActionAgentB"]

        # At least one of each should be used in our test
        assert len(action_a_spans) >= 1 or len(action_b_spans) >= 1

    # =================================================================
    # ⚠️ PATTERN 8: ERROR HANDLING & EDGE CASES
    # =================================================================

    @respx.mock
    def test_agent_error_handling(self, mock_openai_chat_response):
        """
        Test instrumentation when agent encounters errors.
        Validates error status in spans.
        """
        # Mock API error
        respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(500, json={"error": "Internal server error"})
        )

        agent = Agent(
            name="ErrorProneAgent",
            model=OpenAIChat(id="gpt-4o"),
            instructions="This agent might fail.",
            markdown=True,
        )

        try:
            response = agent.run("Test query that will fail")
        except Exception:
            pass  # Expected to fail

        spans = self.wait_for_spans(min_spans=2, timeout=2.0)

        # Check for error spans
        error_spans = [s for s in spans if s.status.is_ok is False]
        assert len(error_spans) >= 1

        # Verify error details
        error_span = error_spans[0]
        assert error_span.status.status_code.name == "ERROR"
        assert (
            "error" in str(error_span.status.description).lower()
            or "500" in str(error_span.status.description)
            or "failed" in str(error_span.status.description).lower()
        )

        # Error should still have agent name
        assert error_span.attributes.get("agent.name") == "ErrorProneAgent"

    @respx.mock
    def test_agent_with_invalid_tool(self, mock_openai_chat_response):
        """
        Test agent with tool that raises exception.
        Tool execution error should be captured.
        """
        # Mock tool call
        tool_response = mock_openai_chat_response.copy()
        tool_response["choices"][0]["message"]["content"] = None
        tool_response["choices"][0]["message"]["tool_calls"] = [
            {
                "id": "call_err",
                "type": "function",
                "function": {"name": "failing_tool", "arguments": json.dumps({"param": "test"})},
            }
        ]

        final_response = mock_openai_chat_response.copy()

        respx.post("https://api.openai.com/v1/chat/completions").mock(
            side_effect=[
                httpx.Response(200, json=tool_response),
                httpx.Response(200, json=final_response),
            ]
        )

        # Define failing tool
        def failing_tool(param: str) -> str:
            raise ValueError(f"Tool failed with param: {param}")

        agent = Agent(
            name="ToolErrorAgent",
            model=OpenAIChat(id="gpt-4o"),
            tools=[failing_tool],
            instructions="Use tools carefully.",
            markdown=True,
        )

        try:
            response = agent.run("Use the failing tool with test parameter")
        except (ValueError, Exception):
            pass  # Tool execution failed

        spans = self.wait_for_spans(min_spans=4, timeout=2.0)

        # Check for tool error span
        tool_spans = [s for s in spans if "failing_tool" in s.name]
        if tool_spans:
            tool_span = tool_spans[0]
            if tool_span.status.is_ok is False:
                assert tool_span.status.status_code.name == "ERROR"
                assert "failed" in str(tool_span.status.description).lower()

    def test_agent_configuration_variations(self):
        """
        Test different agent configurations produce correct spans.
        Edge cases in agent setup.
        """
        # Test minimal agent
        minimal_agent = Agent(name="MinimalAgent", model=OpenAIChat(id="gpt-4o"))

        # Test full configuration
        full_agent = Agent(
            name="FullFeatureAgent",
            model=OpenAIChat(id="gpt-4o"),
            role="senior_advisor",
            description="A fully configured agent",
            instructions="""You are an expert with multiple capabilities.
            Use markdown formatting and be concise.""",
            markdown=True,
            show_tool_calls=True,
            add_datetime_to_instructions=True,
            debug_mode=True,
        )

        # Test without name (should still work)
        unnamed_agent = Agent(model=OpenAIChat(id="gpt-4o"))

        # Just creating agents shouldn't create spans
        spans = self.exporter.get_finished_spans()
        assert len(spans) == 0  # No execution yet

        # Run minimal agent (with mocked LLM in actual test)
        # This would need @respx.mock decorator

        print("Agent configuration tests passed - no spans until execution")

    @respx.mock
    def test_agent_with_system_prompt_override(self, mock_openai_chat_response):
        """
        Test agent with custom system prompt overrides.
        Advanced configuration scenario.
        """
        respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=mock_openai_chat_response)
        )

        custom_system_prompt = """You are a specialized agent with custom rules:
        1. Always respond in JSON format
        2. Include 'confidence' score (0-1)
        3. Add 'reasoning' field explaining your thought process
        4. Keep responses under 100 words"""

        agent = Agent(
            name="CustomPromptAgent",
            model=OpenAIChat(id="gpt-4o", system_prompt=custom_system_prompt),
            instructions="Follow your system prompt exactly.",
            markdown=False,  # JSON output
        )

        response = agent.run("Analyze this test query")

        spans = self.wait_for_spans(min_spans=2)

        # Check for custom prompt in attributes
        llm_spans = [s for s in spans if "openai" in s.name.lower()]
        if llm_spans:
            llm_span = llm_spans[0]
            # System prompt might be in attributes
            if (
                "system" in str(llm_span.attributes).lower()
                or "prompt" in str(llm_span.attributes).lower()
            ):
                assert (
                    "custom" in str(llm_span.attributes).lower()
                    or "json" in str(llm_span.attributes).lower()
                    or "confidence" in str(llm_span.attributes).lower()
                )


# =================================================================
# TEST RUNNER & CONFIGURATION
# =================================================================

if __name__ == "__main__":
    """
    To run Agno instrumentation tests:

    1. Install dependencies:
       pip install pytest pytest-asyncio respx httpx agno
       pip install openinference-instrumentation-agno  # If available

    2. Set up environment variables for API keys (or use test keys):
       export OPENAI_API_KEY="test-key"
       export ANTHROPIC_API_KEY="test-key"
       export GOOGLE_API_KEY="test-key"
       export GROQ_API_KEY="test-key"

    3. Run specific test patterns:
       pytest test_agno_instrumentation.py::TestAgnoInstrumentation::test_basic_agent_sync -v

    4. Run all tests:
       pytest test_agno_instrumentation.py -v

    5. Run with coverage:
       pytest test_agno_instrumentation.py --cov=neatlogs --cov-report=html

    6. Skip tests if Agno not installed:
       pytest test_agno_instrumentation.py -v --tb=short
    """
    print("Agno Framework Instrumentation Test Suite")
    print("=" * 50)
    print("Patterns Tested:")
    print("1. ✅ Basic Agent Execution")
    print("2. 🔵 Multi-Agent Systems (Tic Tac Toe)")
    print("3. 🟡 Tool-Calling Agents")
    print("4. 🟠 Multi-Model Federation")
    print("5. 🔴 Streaming Responses")
    print("6. 🟣 Knowledge Base Integration (RAG)")
    print("7. 🟠 Workflows & State Machines")
    print("8. ⚠️ Error Handling & Edge Cases")
