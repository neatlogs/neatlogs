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

import asyncio
import os
import time

import httpx
import pytest
import respx
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.trace import StatusCode

# Mocking Env to prevent SDK start-up crash
os.environ["OPENAI_API_KEY"] = "sk-fake"

try:
    from agents import Agent, Runner, function_tool
except ImportError:
    pytest.skip("openai-agents not installed", allow_module_level=True)


# --- HELPER: NEW API MOCK FACTORY ---
def create_mock_response(content=None, tool_calls=None):
    """
    Generates a valid response for the /v1/responses API as required by
    openai>=2.x. Output items use the real Responses schemas:
      - message  -> {type:"message", role, status, content:[{type:"output_text",...}]}
      - function_call -> {type:"function_call", call_id, name, arguments, ...}
    All top-level required fields (created_at, object, model, tools, etc.) and
    the usage *_tokens_details blocks are included so Pydantic validation and
    the `output_text` aggregation property both succeed.
    """
    output_items = []

    # 1. Message content (Responses API message shape)
    if content:
        output_items.append(
            {
                "type": "message",
                "id": "msg_mock",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": content, "annotations": []}],
            }
        )

    # 2. Tool calls (Responses API function_call shape)
    if tool_calls:
        for i, tc in enumerate(tool_calls):
            call_id = tc.get("id", f"call_{i}")
            output_items.append(
                {
                    "type": "function_call",
                    "id": f"fc_{call_id}",
                    "call_id": call_id,
                    "name": tc["name"],
                    "arguments": tc["arguments"],
                    "status": "completed",
                }
            )

    return {
        "id": "resp_mock_123",
        "created_at": 0,
        "object": "response",
        "status": "completed",
        "model": "gpt-4o",
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "output": output_items,
        "usage": {
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
    }


class TestOpenAIAgentsInstrumentation:

    @pytest.fixture(autouse=True)
    def setup_teardown(self, in_memory_span_exporter):
        import neatlogs

        # OTel set_tracer_provider + neatlogs.init are set-once; reset both so
        # each test installs its own provider + processor (avoids in-group
        # pollution where later tests' spans go to the first test's exporter).
        neatlogs.shutdown()
        trace._TRACER_PROVIDER = None
        trace._TRACER_PROVIDER_SET_ONCE._done = False

        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(in_memory_span_exporter))
        trace.set_tracer_provider(provider)

        neatlogs.init(
            api_key="test-key",
            disable_export=True,
            instrumentations=["openai", "openai-agents"],
        )

        # AGENT/tool/handoff spans come from the Agents SDK trace processor, which
        # is NOT auto-registered by instrumentations=["openai-agents"] (that only
        # wires the OpenInference LLM instrumentation). Register neatlogs' processor
        # explicitly. set_trace_processors replaces, so it won't stack across tests.
        from agents import set_trace_processors

        set_trace_processors([neatlogs.openai_agents_processor()])
        yield
        neatlogs.shutdown()

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
        assert any(s.attributes.get("neatlogs.agent.name") == "AsyncBot" for s in spans)

    # =================================================================
    # 🟡 PATTERN 3: SINGLE TOOL CALLING
    # =================================================================
    @respx.mock
    def test_single_tool_call(self, in_memory_span_exporter):
        # 1. Agent asks to call tool
        tool_resp = create_mock_response(
            tool_calls=[{"name": "get_weather", "arguments": '{"city": "Pune"}'}]
        )
        # 2. Agent gives final answer
        final_resp = create_mock_response(content="Pune is sunny.")

        respx.post(url__regex=r"https://api.openai.com/v1/responses").mock(
            side_effect=[httpx.Response(200, json=tool_resp), httpx.Response(200, json=final_resp)]
        )

        @function_tool
        def get_weather(city: str):
            return "Sunny"

        agent = Agent(name="WeatherBot", tools=[get_weather])
        Runner.run_sync(agent, "Weather in Pune?")

        time.sleep(0.1)
        spans = in_memory_span_exporter.get_finished_spans()

        # Verify Tool Span — match on the TOOL kind + tool name, not the exact span
        # name (the instrumentor names it e.g. "openai_agents.tool.get_weather").
        tool_span = next(
            (
                s
                for s in spans
                if s.attributes.get("neatlogs.span.kind") == "tool"
                and s.attributes.get("neatlogs.tool.name") == "get_weather"
            ),
            None,
        )
        assert tool_span is not None, "Tool execution span missing"

    # =================================================================
    # 🟠 PATTERN 4: PARALLEL TOOLS
    # =================================================================
    @respx.mock
    def test_parallel_tool_calls(self, in_memory_span_exporter):
        # 1. Agent calls 2 tools at once
        parallel_resp = create_mock_response(
            tool_calls=[
                {"id": "call_1", "name": "get_stock", "arguments": '{"ticker":"AAPL"}'},
                {"id": "call_2", "name": "get_stock", "arguments": '{"ticker":"MSFT"}'},
            ]
        )
        final_resp = create_mock_response(content="Stocks are up.")

        respx.post(url__regex=r"https://api.openai.com/v1/responses").mock(
            side_effect=[
                httpx.Response(200, json=parallel_resp),
                httpx.Response(200, json=final_resp),
            ]
        )

        @function_tool
        def get_stock(ticker: str):
            return "100"

        agent = Agent(name="StockBot", tools=[get_stock])
        Runner.run_sync(agent, "Check AAPL and MSFT")

        time.sleep(0.1)
        spans = in_memory_span_exporter.get_finished_spans()

        stock_spans = [
            s
            for s in spans
            if s.attributes.get("neatlogs.span.kind") == "tool"
            and s.attributes.get("neatlogs.tool.name") == "get_stock"
        ]
        assert len(stock_spans) == 2, f"Expected 2 parallel spans, got {len(stock_spans)}"

    # =================================================================
    # 🔴 PATTERN 5: AGENT HANDOFFS
    # =================================================================
    @respx.mock
    def test_agent_handoff(self, in_memory_span_exporter):
        # 1. TriageBot calls transfer tool. The SDK names transfer tools
        # 'transfer_to_<snake_case(agent_name)>' — "SupportBot" -> "supportbot".
        handoff_resp = create_mock_response(
            tool_calls=[{"name": "transfer_to_supportbot", "arguments": "{}"}]
        )
        # 2. SupportBot replies
        support_resp = create_mock_response(content="Support here.")

        respx.post(url__regex=r"https://api.openai.com/v1/responses").mock(
            side_effect=[
                httpx.Response(200, json=handoff_resp),
                httpx.Response(200, json=support_resp),
            ]
        )

        support_agent = Agent(name="SupportBot")
        triage_agent = Agent(name="TriageBot", handoffs=[support_agent])

        Runner.run_sync(triage_agent, "Help me")

        time.sleep(0.1)
        spans = in_memory_span_exporter.get_finished_spans()

        # Check the trace contains BOTH agents (handoff actually executed).
        agent_names = [
            s.attributes.get("neatlogs.agent.name")
            for s in spans
            if s.attributes.get("neatlogs.agent.name")
        ]
        assert "TriageBot" in agent_names
        assert "SupportBot" in agent_names, "Handoff target agent never ran"
        # Check for the dedicated handoff span (kind=agent, name contains 'handoff',
        # or carries the handoff_from attribute).
        assert any(
            "handoff" in s.name.lower() or any("handoff" in k.lower() for k in s.attributes)
            for s in spans
        ), "Handoff span missing"

    # =================================================================
    # 🔥 PATTERN 6: RAG + EMBEDDINGS (Nested)
    # =================================================================
    @respx.mock
    def test_rag_nested_flow(self, in_memory_span_exporter):
        # 1. Agent calls RAG tool
        tool_req = create_mock_response(
            tool_calls=[{"name": "rag_search", "arguments": '{"q":"AI"}'}]
        )
        final_resp = create_mock_response(content="Found info.")

        # Mock Responses API
        respx.post(url__regex=r"https://api.openai.com/v1/responses").mock(
            side_effect=[httpx.Response(200, json=tool_req), httpx.Response(200, json=final_resp)]
        )

        # Mock Embeddings API (Standard Format still applies here usually)
        respx.post("https://api.openai.com/v1/embeddings").mock(
            return_value=httpx.Response(
                200, json={"data": [{"embedding": [0.1], "index": 0}], "usage": {"total_tokens": 5}}
            )
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

        # Verify Hierarchy — match by kind/tool-name, not exact span names.
        embed_span = next(
            (
                s
                for s in spans
                if s.attributes.get("neatlogs.span.kind") == "embedding"
                or "embedding" in s.name.lower()
            ),
            None,
        )
        tool_span = next(
            (
                s
                for s in spans
                if s.attributes.get("neatlogs.span.kind") == "tool"
                and s.attributes.get("neatlogs.tool.name") == "rag_search"
            ),
            None,
        )

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
        tool_req = create_mock_response(tool_calls=[{"name": "bad_tool", "arguments": "{}"}])
        # 2. Agent apologizes
        final_resp = create_mock_response(content="Error occurred.")

        respx.post(url__regex=r"https://api.openai.com/v1/responses").mock(
            side_effect=[httpx.Response(200, json=tool_req), httpx.Response(200, json=final_resp)]
        )

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

        fail_span = next(
            (
                s
                for s in spans
                if s.attributes.get("neatlogs.span.kind") == "tool"
                and s.attributes.get("neatlogs.tool.name") == "bad_tool"
            ),
            None,
        )
        assert fail_span is not None
        assert fail_span.status.status_code == StatusCode.ERROR
