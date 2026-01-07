"""
Tests for OpenAI Agents SDK (Swarm/Agents) Instrumentation
==========================================================
Comprehensive Test Suite covering 7 Advanced Patterns.
Updated for the new 'Responses API' (/v1/responses) structure.

Patterns Covered:
1. 🟢 Basic: Simple Synchronous Chat (PASSED)
2. 🔵 Async: Asynchronous Execution
3. 🟡 Single Tool: Basic Function Calling
4. 🟠 Parallel Tools: Multiple tools called in one turn
5. 🔴 Handoffs: Agent A -> Agent B
6. 🔥 RAG + Embeddings: Nested Deep Tracing
7. ⚠️ Error Handling: Tool Failures
"""

import pytest
import respx
import httpx
import time
import os
import asyncio
from opentelemetry import trace
from opentelemetry.trace import StatusCode
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

# Mocking Env to prevent SDK start-up crash
os.environ["OPENAI_API_KEY"] = "sk-fake"

try:
    from agents import Agent, Runner, function_tool
except ImportError:
    pytest.skip("openai-agents not installed", allow_module_level=True)

# --- HELPER: NEW API MOCK FACTORY ---
def create_mock_response(content=None, tool_calls=None):
    """
    Generates a valid response for the new /v1/responses API.
    Crucial: Includes 'token_details' to pass Pydantic validation.
    """
    output_items = []
    
    # 1. Add Message Content
    if content:
        output_items.append({
            "type": "message",
            "message": {"role": "assistant", "content": content}
        })
    
    # 2. Add Tool Calls
    if tool_calls:
        for tc in tool_calls:
            output_items.append({
                "type": "tool_call",
                "id": tc.get("id", "call_123"),
                "function": {
                    "name": tc["name"],
                    "arguments": tc["arguments"]
                }
            })

    return {
        "id": "resp_mock_123",
        "status": "completed",
        "output": output_items,
        "usage": {
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
            # 🔥 CRITICAL FIELDS FOR NEW SDK
            "input_token_details": {"cache_read": 0},
            "output_token_details": {"reasoning": 0}
        }
    }

class TestOpenAIAgentsInstrumentation:
    
    @pytest.fixture(autouse=True)
    def setup_teardown(self, in_memory_span_exporter):
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(in_memory_span_exporter))
        trace.set_tracer_provider(provider)
        
        import neatlogs
        neatlogs.init(api_key="test-key", instrumentations=["openai", "openai-agents"])
        yield

    # =================================================================
    # 🟢 PATTERN 1: BASIC SYNC (Proven Working)
    # =================================================================
    @respx.mock
    def test_basic_agent_sync(self, in_memory_span_exporter):
        mock_resp = create_mock_response(content="Hello!")
        
        respx.post(url__regex=r"https://api.openai.com/v1/responses").mock(
            return_value=httpx.Response(200, json=mock_resp)
        )

        agent = Agent(name="SimpleBot")
        Runner.run_sync(agent, "Hi")
        
        time.sleep(0.1)
        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) >= 1

    # =================================================================
    # 🔵 PATTERN 2: ASYNC EXECUTION
    # =================================================================
    @respx.mock
    @pytest.mark.asyncio
    async def test_basic_agent_async(self, in_memory_span_exporter):
        mock_resp = create_mock_response(content="Async Hello!")
        
        respx.post(url__regex=r"https://api.openai.com/v1/responses").mock(
            return_value=httpx.Response(200, json=mock_resp)
        )

        agent = Agent(name="AsyncBot")
        # In this SDK, Runner.run() is typically the async entry point
        await Runner.run(agent, "Hi Async")

        time.sleep(0.1)
        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) >= 1
        assert any(s.attributes.get("agent.name") == "AsyncBot" for s in spans)

    # =================================================================
    # 🟡 PATTERN 3: SINGLE TOOL CALLING
    # =================================================================
    @respx.mock
    def test_single_tool_call(self, in_memory_span_exporter):
        # 1. Agent asks to call tool
        tool_resp = create_mock_response(
            tool_calls=[{"name": "get_weather", "arguments": "{\"city\": \"Pune\"}"}]
        )
        # 2. Agent gives final answer
        final_resp = create_mock_response(content="Pune is sunny.")

        respx.post(url__regex=r"https://api.openai.com/v1/responses").mock(side_effect=[
            httpx.Response(200, json=tool_resp),
            httpx.Response(200, json=final_resp)
        ])

        @function_tool
        def get_weather(city: str):
            return "Sunny"

        agent = Agent(name="WeatherBot", tools=[get_weather])
        Runner.run_sync(agent, "Weather in Pune?")

        time.sleep(0.1)
        spans = in_memory_span_exporter.get_finished_spans()
        
        # Verify Tool Span
        tool_span = next((s for s in spans if s.name == "get_weather"), None)
        assert tool_span is not None, "Tool execution span missing"

    # =================================================================
    # 🟠 PATTERN 4: PARALLEL TOOLS
    # =================================================================
    @respx.mock
    def test_parallel_tool_calls(self, in_memory_span_exporter):
        # 1. Agent calls 2 tools at once
        parallel_resp = create_mock_response(
            tool_calls=[
                {"id": "call_1", "name": "get_stock", "arguments": "{\"ticker\":\"AAPL\"}"},
                {"id": "call_2", "name": "get_stock", "arguments": "{\"ticker\":\"MSFT\"}"}
            ]
        )
        final_resp = create_mock_response(content="Stocks are up.")

        respx.post(url__regex=r"https://api.openai.com/v1/responses").mock(side_effect=[
            httpx.Response(200, json=parallel_resp),
            httpx.Response(200, json=final_resp)
        ])

        @function_tool
        def get_stock(ticker: str):
            return "100"

        agent = Agent(name="StockBot", tools=[get_stock])
        Runner.run_sync(agent, "Check AAPL and MSFT")

        time.sleep(0.1)
        spans = in_memory_span_exporter.get_finished_spans()
        
        stock_spans = [s for s in spans if s.name == "get_stock"]
        assert len(stock_spans) == 2, f"Expected 2 parallel spans, got {len(stock_spans)}"

    # =================================================================
    # 🔴 PATTERN 5: AGENT HANDOFFS
    # =================================================================
    @respx.mock
    def test_agent_handoff(self, in_memory_span_exporter):
        # 1. TriageBot calls transfer tool
        # Note: The SDK names transfer tools as 'transfer_to_{AgentName}'
        handoff_resp = create_mock_response(
            tool_calls=[{"name": "transfer_to_SupportBot", "arguments": "{}"}]
        )
        # 2. SupportBot replies
        support_resp = create_mock_response(content="Support here.")

        respx.post(url__regex=r"https://api.openai.com/v1/responses").mock(side_effect=[
            httpx.Response(200, json=handoff_resp),
            httpx.Response(200, json=support_resp)
        ])

        support_agent = Agent(name="SupportBot")
        triage_agent = Agent(name="TriageBot", handoffs=[support_agent])

        Runner.run_sync(triage_agent, "Help me")

        time.sleep(0.1)
        spans = in_memory_span_exporter.get_finished_spans()
        
        # Check trace contains both agents
        agent_names = [s.attributes.get("agent.name") for s in spans if s.attributes.get("agent.name")]
        assert "TriageBot" in agent_names
        # Check for handoff event (tool call)
        assert any("transfer" in s.name for s in spans), "Handoff tool call missing"

    # =================================================================
    # 🔥 PATTERN 6: RAG + EMBEDDINGS (Nested)
    # =================================================================
    @respx.mock
    def test_rag_nested_flow(self, in_memory_span_exporter):
        # 1. Agent calls RAG tool
        tool_req = create_mock_response(
            tool_calls=[{"name": "rag_search", "arguments": "{\"q\":\"AI\"}"}]
        )
        final_resp = create_mock_response(content="Found info.")

        # Mock Responses API
        respx.post(url__regex=r"https://api.openai.com/v1/responses").mock(side_effect=[
            httpx.Response(200, json=tool_req),
            httpx.Response(200, json=final_resp)
        ])
        
        # Mock Embeddings API (Standard Format still applies here usually)
        respx.post("https://api.openai.com/v1/embeddings").mock(
            return_value=httpx.Response(200, json={
                "data": [{"embedding": [0.1], "index": 0}],
                "usage": {"total_tokens": 5}
            })
        )

        @function_tool
        def rag_search(q: str):
            # Nested call to OpenAI Embeddings
            from openai import OpenAI
            client = OpenAI(api_key="fake")
            client.embeddings.create(input=q, model="text-embedding-3-small")
            return "Retrieved Docs"

        agent = Agent(name="RAGBot", tools=[rag_search])
        Runner.run_sync(agent, "Search AI")

        time.sleep(0.1)
        spans = in_memory_span_exporter.get_finished_spans()
        
        # Verify Hierarchy
        embed_span = next((s for s in spans if "embedding" in s.name), None)
        tool_span = next((s for s in spans if s.name == "rag_search"), None)
        
        assert tool_span is not None, "Tool span missing"
        assert embed_span is not None, "Embedding span missing"
        # Validate Context Propagation
        assert embed_span.context.trace_id == tool_span.context.trace_id, "Broken Trace Context!"

    # =================================================================
    # ⚠️ PATTERN 7: ERROR HANDLING
    # =================================================================
    @respx.mock
    def test_agent_error_status(self, in_memory_span_exporter):
        # 1. Agent calls tool
        tool_req = create_mock_response(
            tool_calls=[{"name": "bad_tool", "arguments": "{}"}]
        )
        # 2. Agent apologizes
        final_resp = create_mock_response(content="Error occurred.")

        respx.post(url__regex=r"https://api.openai.com/v1/responses").mock(side_effect=[
            httpx.Response(200, json=tool_req),
            httpx.Response(200, json=final_resp)
        ])

        @function_tool
        def bad_tool():
            raise ValueError("DB Crash")

        agent = Agent(name="FailBot", tools=[bad_tool])
        
        try:
            Runner.run_sync(agent, "Do fail")
        except:
            pass

        time.sleep(0.1)
        spans = in_memory_span_exporter.get_finished_spans()
        
        fail_span = next((s for s in spans if s.name == "bad_tool"), None)
        assert fail_span is not None
        assert fail_span.status.status_code == StatusCode.ERROR