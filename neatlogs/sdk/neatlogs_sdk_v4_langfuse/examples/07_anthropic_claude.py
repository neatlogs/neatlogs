"""
Example 7: Anthropic Claude SDK

Both OpenInference and OpenLLMetry support Anthropic SDK.
Shows prompt caching, tool use, and vision capabilities.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from neatlogs.sdk.neatlogs_sdk_v4_langfuse import init, trace, flush, shutdown
from anthropic import Anthropic


def example_basic_chat():
    """Basic Claude chat."""
    client = Anthropic()
    
    topic = "quantum computing"
    
    with trace(
        "explain_topic",
        prompt_template="Explain {topic} in simple terms",
        prompt_variables={"topic": topic}
    ):
        message = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": f"Explain {topic} in simple terms"
            }]
        )
    
    print(f"Response: {message.content[0].text}")


def example_prompt_caching():
    """
    Claude's prompt caching - OpenLLMetry captures cache metrics!
    Cache creation and cache read tokens are tracked separately.
    """
    client = Anthropic()
    
    print("\n" + "="*60)
    print("Prompt Caching Example:")
    print("="*60)
    
    # Large system prompt (will be cached)
    system_prompt = """You are an AI assistant specializing in Python programming.
You have deep knowledge of:
- Python syntax and best practices
- Common libraries (numpy, pandas, requests, etc.)
- Design patterns and architectures
- Performance optimization
- Testing and debugging
""" * 10  # Make it large enough to benefit from caching
    
    message = client.messages.create(
        model="claude-sonnet-4-5-20250929",
    max_tokens=1024,
    system=[{
        "type": "text",
        "text": system_prompt,
        "cache_control": {"type": "ephemeral"}
    }],
    messages=[{
        "role": "user",
        "content": "Write a function to reverse a string"
    }])
    
    print(f"Response: {message.content[0].text[:100]}...")
    print(f"\nCache stats:")
    print(f"  Cache creation tokens: {message.usage.cache_creation_input_tokens}")
    print(f"  Cache read tokens: {message.usage.cache_read_input_tokens}")
    
    # Span will have (merged from both conventions):
    # - llm.token_count.prompt (total prompt tokens)
    # - llm.token_count.prompt.cache_creation (OpenLLMetry)
    # - llm.token_count.prompt.cache_read (OpenLLMetry)
    # - llm.cost.prompt (adjusted for cache pricing)


def example_tool_use():
    """Claude tool use (function calling)."""
    client = Anthropic()
    
    print("\n" + "="*60)
    print("Tool Use Example:")
    print("="*60)
    
    tools = [{
        "name": "get_stock_price",
        "description": "Get the current stock price for a company",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "The stock ticker symbol"
                }
            },
            "required": ["ticker"]
        }
    }]
    
    message = client.messages.create(
        model="claude-sonnet-4-5-20250929",
    max_tokens=1024,
    tools=tools,
    messages=[{
        "role": "user",
        "content": "What's the stock price of Apple?"
    }])
    
    # Check if Claude wants to use a tool
    for content in message.content:
        if content.type == "tool_use":
            print(f"Tool: {content.name}")
            print(f"Arguments: {content.input}")
    
    # Span will have:
    # - llm.tools (OpenInference - JSON list)
    # - Tool call details if made


def main():
    # Enable span logging
    os.environ['NEATLOGS_LOG_SPANS'] = 'true'
    
    # Initialize
    init(
        api_key=os.getenv("NEATLOGS_API_KEY", "test-key"),
        workflow_name="anthropic-claude",
        instrumentations=["anthropic"],
        debug=True,
    )
    
    try:
        with trace("anthropic_examples"):
            # Run examples
            example_basic_chat()
            example_prompt_caching()
            example_tool_use()
        
        print("\n✅ All Claude examples completed!")
        print("Check spans.log to see:")
        print("  - Token counts (including cache metrics)")
        print("  - Accurate costs (cache pricing is cheaper)")
        print("  - Tool use tracking")
        print("  - HTTP spans as children of LLM spans")
    except Exception as e:
        print(f"\nError during Anthropic Claude execution: {e}")
    finally:
        print("\n💾 Flushing spans...")
        flush()
        print("🛑 Shutting down SDK...")
        shutdown()
        print("✅ Done!")


if __name__ == "__main__":
    main()
