"""
Example 2: Prompt Variable Capture (3 Layers)

This example shows all three ways to capture prompt variables:
1. Framework Instrumentation (LangChain, LlamaIndex) - Most automatic
2. Decorator Auto-Capture (@observe) - Best for custom functions  
3. Context Manager (with trace) - Observish style for inline usage

All methods ensure variables appear on LLM spans for consistent querying.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from neatlogs.sdk.neatlogs_sdk_v4_langfuse import init, observe, trace, flush, shutdown
from openai import OpenAI


def main():
    # Initialize
    os.environ["NEATLOGS_LOG_SPANS"] = "true"  # Log spans to see results
    
    init(
        api_key="EQZO8zBsJkDRvmIL06m-EQ7faMUHEIN5",
        workflow_name="prompt-capture-demo",
        instrumentations=["openai"],
        debug=True,
    )
    
    client = OpenAI()
    
    @observe(version="v1.0")
    def get_weather(city: str, date: str, units: str = "metric"):
        """Get weather forecast for a city on a specific date."""
        # Variables automatically captured: {"city": "SF", "date": "Jan 21", "units": "metric"}
        prompt = f"What's the weather in {city} on {date}? Use {units} units."
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    
    # Call the decorated function
    result = get_weather(city="San Francisco", date="January 21, 2026")
    print(f"Response: {result}\n")
    
    # Check spans.log - you'll see:
    # 1. get_weather (CHAIN span) - wrapper for logical grouping
    # 2. ChatCompletion (LLM span) - has llm.prompt_template_variables: {"city": "San Francisco", ...}
    
    
    city = "Tokyo"
    date = "February 1, 2026"
    
    # Observish style: wrap LLM call in context manager
    with trace(
        "weather_query",
        prompt_template="What's the weather in {city} on {date}?",
        prompt_variables={"city": city, "date": date},
        version="v2.0"
    ):
        # Make LLM call - it will inherit the prompt metadata via baggage
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": f"What's the weather in {city} on {date}?"}]
        )
        print(f"Response: {response.choices[0].message.content}\n")
    
    # Check spans.log - you'll see:
    # 1. weather_query (CHAIN) - wrapper with prompt template/variables
    # 2. ChatCompletion (LLM span) - inherits llm.prompt_template_variables via baggage
    
    # Flush and shutdown
    flush()
    shutdown()


if __name__ == "__main__":
    main()
