"""
Simple CrewAI Agent with Tool example for Neatlogs SDK v4.

This example demonstrates:
- CrewAI Agent execution (creates AGENT spans)
- Tool execution via CrewAI (creates TOOL spans)
- HTTP calls inside tools (creates HTTP child spans under TOOL spans)

IMPORTANT: Initialize Neatlogs BEFORE importing CrewAI/requests!
Note: CrewAI uses its own Tool class (not LangChain's @tool decorator)

Run: python examples/simple_crewai_tool.py
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
load_dotenv()

# ============================================================================
# INITIALIZE NEATLOGS FIRST - before any CrewAI/requests imports!
# This ensures OpenInference + HTTP instrumentation attaches properly.
# ============================================================================
from neatlogs.sdk.neatlogs_sdk_v4_threading import init, flush

init(
    api_key=os.getenv("NEATLOGS_API_KEY", "EQZO8zBsJkDRvmIL06m-EQ7faMUHEIN5"),
    workflow_name="crewai-agent-demo",
    session_id="crewai_session_001",
    tags=["crewai", "agent", "tool", "http"],
    enable_http_tracing=True,
    instrumentations=["openai", "crewai"],  # CrewAI instrumentation
    debug=True,
)

# ============================================================================
# NOW import CrewAI and requests (after init)
# ============================================================================
import requests
from crewai import Agent, Task, Crew, Tool


# ============================================================================
# Define tool function (plain Python function)
# CrewAI will wrap it in a Tool object
# ============================================================================
def get_weather_func(city: str) -> str:
    """Get current weather for a city. Returns weather information as text."""
    try:
        # Use httpbin as a reliable test endpoint
        # This demonstrates HTTP spans being captured regardless of the API
        response = requests.get(
            "https://httpbin.org/json",
            timeout=15,
            headers={"X-City-Query": city}
        )
        response.raise_for_status()
        # Simulate weather response for demo purposes
        return f"Weather in {city}: Sunny, 25°C (demo data from httpbin)"
    except requests.exceptions.Timeout:
        return f"Weather service timeout for {city}. Please try again."
    except requests.exceptions.RequestException as e:
        return f"Could not fetch weather for {city}: {str(e)}"


# Create CrewAI Tool object
get_weather = Tool(
    name="get_weather",
    description="Get current weather for a city. Returns weather information as text.",
    func=get_weather_func
)


def main():
    """
    Run a CrewAI Agent that uses tools.
    
    Expected span hierarchy:
    ├── AGENT span (Agent.execute)
    │   ├── LLM span (OpenAI - first call to decide tool usage)
    │   │   └── HTTP span (POST api.openai.com)
    │   │
    │   └── TOOL span (get_weather)  ← Created by CrewAIInstrumentor!
    │       └── HTTP span (GET httpbin.org/json)  ← Child of TOOL span!
    │   │
    │   └── LLM span (OpenAI - final response with tool result)
    │       └── HTTP span (POST api.openai.com)
    """
    
    # Define agent with Tool object
    weather_agent = Agent(
        role="Weather Assistant",
        goal="Provide accurate weather information for any city",
        backstory="You are a helpful weather assistant that can check current weather conditions.",
        tools=[get_weather],  # CrewAI Tool object
        verbose=True,
        allow_delegation=False,
    )
    
    # Define task
    weather_task = Task(
        description="Get the current weather for Tokyo and provide a brief summary.",
        expected_output="A summary of the current weather in Tokyo.",
        agent=weather_agent,
    )
    
    # Create crew
    crew = Crew(
        agents=[weather_agent],
        tasks=[weather_task],
        verbose=True,
    )
    
    # Execute the crew - CrewAI will:
    # 1. Call LLM to decide if tool is needed
    # 2. Execute tool (creates TOOL span, HTTP call becomes child)
    # 3. Call LLM again with tool result
    print(f"\n{'='*60}")
    print("Starting CrewAI agent execution...")
    print(f"{'='*60}\n")
    
    result = crew.kickoff()
    
    print(f"\n{'='*60}")
    print(f"Agent Response: {result}")
    print(f"{'='*60}")
    
    return result


if __name__ == "__main__":
    result = main()
    flush()
    print("\n✅ Done! Check your Neatlogs dashboard for spans.")
    print("Expected: AGENT → LLM → TOOL (with HTTP child) → LLM")
