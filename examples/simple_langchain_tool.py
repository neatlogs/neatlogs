"""
Simple LangChain Agent with Tool example for Neatlogs SDK v4.

This example demonstrates:
- LangChain Agent execution (creates AGENT spans)
- Tool execution via LangChain callback system (creates TOOL spans)
- HTTP calls inside tools (creates HTTP child spans under TOOL spans)

IMPORTANT: Initialize Neatlogs BEFORE importing LangChain/requests!

Run: python examples/simple_langchain_tool.py
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
load_dotenv()

# ============================================================================
# INITIALIZE NEATLOGS FIRST - before any LangChain/requests imports!
# This ensures OpenInference + HTTPX instrumentation attaches properly.
# ============================================================================
from neatlogs.sdk.neatlogs_sdk_v4 import init, flush

init(
    api_key=os.getenv("NEATLOGS_API_KEY", "EQZO8zBsJkDRvmIL06m-EQ7faMUHEIN5"),
    workflow_name="langchain-agent-demo",
    session_id="agent_session_006",
    tags=["langchain", "agent", "tool", "http"],
    enable_http_tracing=True,
    instrumentations=["openai", "langchain"],
    debug=True,
)

# ============================================================================
# INITIALIZE LANGFUSE (for comparison)
# Uses LangChain-specific CallbackHandler integration
# Ref: https://langfuse.com/integrations/frameworks/langchain
# ============================================================================
from langfuse import get_client
from langfuse.langchain import CallbackHandler

# # Initialize Langfuse client
langfuse = get_client()


# # Initialize Langfuse CallbackHandler for LangChain tracing
# langfuse_handler = CallbackHandler()

# print("✅ Langfuse CallbackHandler initialized for LangChain")

# ============================================================================
# NOW import LangChain and requests (after init)
# ============================================================================
import requests
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool as langchain_tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor


# ============================================================================
# Define tool with @langchain_tool decorator
# When LangChain Agent executes this tool, LangChainInstrumentor creates a TOOL span
# The HTTP call inside becomes a child of the TOOL span (via RequestsInstrumentor)
# ============================================================================

@langchain_tool
# @neatlogs_tool(name="get_weather")
def get_weather(city: str) -> str:
    """Get current weather for a city. Returns weather information as text."""
    try:
        # Use httpbin as a reliable test endpoint (wttr.in can timeout)
        # This demonstrates HTTP spans being captured regardless of the API
        response = requests.get(
            "https://httpbin.org/json",
            timeout=15,
            headers={"X-City-Query": city}
        )
        response.raise_for_status()
        # Simulate weather response for demo purposes
        return f"Weather in {city}: Sunny, 22°C (demo data from httpbin)"
    except requests.exceptions.Timeout:
        return f"Weather service timeout for {city}. Please try again."
    except requests.exceptions.RequestException as e:
        return f"Could not fetch weather for {city}: {str(e)}"


def main():
    """
    Run a LangChain Agent that uses tools.
    
    Expected span hierarchy:
    ├── CHAIN span (AgentExecutor.invoke)
    │   ├── LLM span (ChatOpenAI - first call to decide tool usage)
    │   │   └── HTTP span (POST api.openai.com)
    │   │
    │   └── TOOL span (get_weather)  ← Created by LangChainInstrumentor!
    │       └── HTTP span (GET httpbin.org/json)  ← Child of TOOL span!
    │   │
    │   └── LLM span (ChatOpenAI - final response with tool result)
    │       └── HTTP span (POST api.openai.com)
    """
    
    # Create LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # Create prompt template for agent
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that can check the weather."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    
    # Create agent with tools
    tools = [get_weather]
    agent = create_tool_calling_agent(llm, tools, prompt)
    
    # Create executor - this is what actually runs the agent loop
    executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    # Invoke the agent - LangChain will:
    # 1. Call LLM to decide if tool is needed
    # 2. Execute tool (creates TOOL span, HTTP call becomes child)
    # 3. Call LLM again with tool result
    # 
    # Pass langfuse_handler to send traces to Langfuse as well
    # result = executor.invoke(
    #     {"input": "What's the weather in Tokyo?"},
    #     config={"callbacks": [langfuse_handler]}
    # )
    result = executor.invoke(
        {"input": "What's the weather in Tokyo?"}
    )
    
    print(f"\n{'='*60}")
    print(f"Agent Response: {result['output']}")
    print(f"{'='*60}")
    
    return result


if __name__ == "__main__":
    result = main()
    
    # Flush both observability platforms
    flush()  # Neatlogs
    # langfuse.flush()  # Langfuse
    
    print("\n✅ Done! Check your dashboards:")
    print("   • Neatlogs: http://localhost:3000")
    print("   • Langfuse: https://cloud.langfuse.com")
    print("\nExpected: AGENT → LLM → TOOL (with HTTP child) → LLM")
    print("\n🔍 Compare how each platform handles:")
    print("  1. ChatCompletion parent_id (orphan vs correct parent)")
    print("  2. HTTP span handling (separate spans vs merged attributes)")
    print("  3. Context propagation in ThreadPoolExecutor")
