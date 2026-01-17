"""
Neatlogs Anthropic Example
===========================
This example demonstrates how to use Neatlogs with Anthropic Claude API calls.
Traces will be sent to the local dev server.
"""

import os
import sys

# Add parent directory to path to import local neatlogs
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from neatlogs import init

# Initialize neatlogs with debug mode and OpenTelemetry enabled
init(api_key="test-key", debug=True, enable_otel=True, instrumentations=["anthropic"])

print("=" * 60)
print("Neatlogs Anthropic Example")
print("=" * 60)

try:
    from anthropic import Anthropic

    # Create Anthropic client (uses ANTHROPIC_API_KEY from environment)
    client = Anthropic()

    print("\nMaking Anthropic API call...")

    # Make a simple message request
    message = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=50,
        temperature=0.7,
        messages=[{"role": "user", "content": "Say hello in 5 words or less"}],
    )

    print(f"\nResponse: {message.content[0].text}")
    print("\n✓ Success! Trace sent to http://localhost:8000/api/data/v2")
    print("  Check your dev_server.py console for the trace data!")
    print("\n" + "=" * 60)

    # Give background thread time to complete the request
    import time

    time.sleep(1)

except ImportError:
    print("\n⚠ Error: Anthropic library not installed")
    print("  Install with: uv add anthropic")
    print("  or: pip install anthropic")
    print("=" * 60)

except Exception as e:
    print(f"\n⚠ Error making Anthropic call: {e}")
    print("  Make sure ANTHROPIC_API_KEY is set in your environment:")
    print("  export ANTHROPIC_API_KEY='your-api-key-here'")
    print("=" * 60)
