"""
Example 6: OpenAI SDK Direct Usage

Shows direct OpenAI SDK usage with streaming, function calling, and vision.
Both OpenInference and OpenLLMetry instrument OpenAI SDK.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from neatlogs.sdk.neatlogs_sdk_v4_langfuse import init, trace, flush, shutdown
# Enable span logging
os.environ['NEATLOGS_LOG_SPANS'] = 'true'

# Initialize
init(
    api_key=os.getenv("NEATLOGS_API_KEY", "test-key"),
    workflow_name="openai-direct",
    instrumentations=["openai"],
    debug=True,
)
from openai import OpenAI


def example_basic_chat():
    """Basic chat completion."""
    client = OpenAI()
    
    city = "Tokyo"
    
    with trace(
        "chat_about_city",
        prompt_template="Tell me about {city}",
        prompt_variables={"city": city}
    ):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": f"Tell me about {city}"}]
        )
    
    print(f"Response: {response.choices[0].message.content}")


def example_streaming():
    """Streaming response - OpenLLMetry captures streaming metrics."""
    client = OpenAI()
    
    print("\n" + "="*60)
    print("Streaming Example:")
    print("="*60)
    
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Count from 1 to 5"}],
        stream=True
    )
    
    for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    
    print("\n")
    
    # Span will have:
    # - llm.is_streaming = true (OpenLLMetry)
    # - Time to first token (OpenLLMetry)
    # - Token counts and cost (both conventions)


def example_function_calling():
    """Function calling with tools."""
    client = OpenAI()
    
    print("="*60)
    print("Function Calling Example:")
    print("="*60)
    
    tools = [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"}
                },
                "required": ["city"]
            }
        }
    }]
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "What's the weather in Paris?"}],
        tools=tools
    )
    
    if response.choices[0].message.tool_calls:
        tool_call = response.choices[0].message.tool_calls[0]
        print(f"Function: {tool_call.function.name}")
        print(f"Arguments: {tool_call.function.arguments}")
    
    # Span will have:
    # - llm.tools (OpenInference - JSON list of tools)
    # - llm.function_call (OpenInference - the actual call made)
    # - llm.request.functions (OpenLLMetry - tools definition)


def example_embeddings():
    """Embeddings generation."""
    client = OpenAI()
    
    print("="*60)
    print("Embeddings Example:")
    print("="*60)
    
    texts = [
        "The capital of France is Paris",
        "Machine learning is a subset of AI"
    ]
    
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    
    print(f"Generated {len(response.data)} embeddings")
    print(f"Embedding dimension: {len(response.data[0].embedding)}")
    
    # Span will have:
    # - openinference.span.kind = "EMBEDDING"
    # - embedding.model_name (OpenInference)
    # - embedding.embeddings (OpenInference - the vectors)
    # - Token count and cost


def main():
    
    
    try:
        with trace("openai_examples"):
            # Run examples
            example_basic_chat()
            example_streaming()
            example_function_calling()
            example_embeddings()
        
        print("\n✅ All examples completed!")
        print("Check spans.log to see:")
        print("  - Separate spans for each operation")
        print("  - Streaming metrics (time to first token)")
        print("  - Function calling details")
        print("  - Embedding vectors")
        print("  - Token counts and costs for everything")
    except Exception as e:
        print(f"\nError during OpenAI direct execution: {e}")
    finally:
        print("\n💾 Flushing spans...")
        flush()
        print("🛑 Shutting down SDK...")
        shutdown()
        print("✅ Done!")


if __name__ == "__main__":
    main()
