"""
Neatlogs OpenAI Example
========================
This example demonstrates how to use Neatlogs with OpenAI API calls.
Traces will be written to a local file (neatlogs.jsonl).
"""

import os
import sys

# Add parent directory to path to import local neatlogs
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import neatlogs

# Initialize neatlogs to write traces to a local file.
# dry_run=True prevents data from being sent to the remote server.
neatlogs.init(api_key="test-key", instrumentations=["openai"])

print("=" * 60)
print("Neatlogs OpenAI Example")
print("=" * 60)

try:
    from openai import AzureOpenAI

    # Create OpenAI client (uses OPENAI_API_KEY from environment)
    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    )

    print("\nMaking OpenAI API call...")

    # Make a simple chat completion request
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say hello in 5 words or less"},
        ],
        max_tokens=50,
        temperature=0.7,
    )

    print(f"\nResponse: {response.choices[0].message.content}")
    print("\n[SUCCESS] Trace sent successfully")
    print("\n" + "=" * 60)

    import time

    time.sleep(5)

except ImportError:
    print("\n⚠ Error: OpenAI library not installed")
    print("  Install with: uv add openai")
    print("  or: pip install openai")
    print("=" * 60)

except Exception as e:
    print(f"\n[ERROR] Error making OpenAI call: {e}")
    print("  Make sure OPENAI_API_KEY is set in your environment:")
    print("  export OPENAI_API_KEY='your-api-key-here'")
    print("=" * 60)
