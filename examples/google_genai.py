"""
Neatlogs Google GenAI Example
=============================
This example demonstrates how to use Neatlogs with Google Generative AI API calls.
Traces will be written to a local file (neatlogs.jsonl).
"""

import os
import sys

# Add parent directory to path to import local neatlogs
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import neatlogs

# Initialize neatlogs to write traces to a local file.
neatlogs.init(
    api_key="test-key", tags=["google-genai", "demo"], instrumentations=["google-genai"]
)

print("=" * 60)
print("Neatlogs Google GenAI Example")
print("=" * 60)

try:
    from google import genai

    print("\nMaking Google GenAI API call...")

    # Initialize the GenAI client with API key from environment
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    # Make a simple content generation request
    response = client.models.generate_content(
        model="gemini-2.5-flash",  # or "gemini-1.5-flash"
        contents=[
            {"role": "user", "parts": [{"text": "Say hello in 5 words or less"}]}
        ],
    )

    print(f"\nResponse: {response.text}")
    print("\n[SUCCESS] Trace sent successfully")
    print("\n" + "=" * 60)

except ImportError:
    print("\n⚠ Error: Google GenAI library not installed")
    print("  Install with: pip install google-generativeai")
    print("=" * 60)

except Exception as e:
    print(f"\n[ERROR] Error making Google GenAI call: {e}")
    print("  Make sure GEMINI_API_KEY is set in your environment:")
    print("  export GEMINI_API_KEY='your-api-key-here'")
    print("=" * 60)
