"""
Neatlogs OpenAI Agents SDK Example (Multi-span)
==============================================
This example demonstrates a flow with tool usage to generate multiple spans.
"""

import os
import sys

# Add parent directory to path to import local neatlogs
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from neatlogs import init

# Initialize neatlogs
init(
    api_key="test-key",
    debug=True,
    enable_otel=True,
    instrumentations=["openai-agents"],
)

print("=" * 60)
print("Neatlogs OpenAI Agents SDK Example")
print("=" * 60)

try:
    from agents import Agent, Runner, function_tool

    # 1. Define a custom tool
    @function_tool
    def get_weather(city: str) -> str:
        """Returns weather information for the specified city."""
        # This execution will appear as a child span in your trace
        return f"The weather in {city} is sunny and 75°F."

    # 2. Create an agent with the tool
    agent = Agent(
        name="WeatherAssistant",
        instructions="You are a helpful assistant that can check the weather.",
        tools=[get_weather],
    )

    print("\nRunning agent with tool...")

    # 3. Run with a prompt that requires the tool
    # This will generate:
    # - Span 1: Agent thinking
    # - Span 2: Tool execution (get_weather)
    # - Span 3: Agent final response
    result = Runner.run_sync(agent, "What is the weather like in San Francisco?")

    print(f"\nResponse: {result.final_output}")
    print("\n✓ Success!")
    print("=" * 60)

    import time

    time.sleep(1)

except ImportError:
    print("\n⚠ Error: OpenAI Agents library not installed")
    print("  Install with: uv add openai-agents")
    print("  or: pip install openai-agents")
    print("=" * 60)

except Exception as e:
    print(f"\n⚠ Error executing agent: {e}")
    print("  Make sure OPENAI_API_KEY is set in your environment.")
    print("=" * 60)
