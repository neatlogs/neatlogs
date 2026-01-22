"""
Example 1: Minimal Usage - Zero Configuration

This example shows the simplest way to use Neatlogs SDK.
Just init() and your LLM calls are automatically traced!
"""

import os
import sys

# Add parent directory to path for local testing
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from neatlogs.sdk.neatlogs_sdk_v4_langfuse import init, flush, shutdown

# Import your LLM library
from openai import OpenAI


def main():
    # Optional: Enable span logging to file
    # Set NEATLOGS_LOG_SPANS=true to log all spans to spans.log in current directory
    os.environ["NEATLOGS_LOG_SPANS"] = "true"
    
    # Initialize Neatlogs with minimal config
    init(
        api_key="EQZO8zBsJkDRvmIL06m-EQ7faMUHEIN5",  # Or set NEATLOGS_API_KEY env var
        instrumentations=["openai"],   # Auto-instrument OpenAI
        debug=True,                    # See what's happening
    )
    
    # Use OpenAI normally - everything auto-traced!
    client = OpenAI()
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": "What is the capital of France?"}
        ]
    )
    
    print(f"Response: {response.choices[0].message.content}")
    
    # Spans are automatically exported with:
    # - llm.token_count.prompt (from both conventions)
    # - llm.token_count.completion (from both conventions)
    # - llm.cost.total (calculated if not provided)
    # - llm.is_streaming (from OpenLLMetry)
    # - openinference.span.kind = "LLM"
    
    # Flush all pending spans before exit
    print("\n💾 Flushing spans...")
    flush()
    
    # Shutdown SDK and clean up resources
    print("🛑 Shutting down SDK...")
    shutdown()
    print("✅ Done!")


if __name__ == "__main__":
    main()
