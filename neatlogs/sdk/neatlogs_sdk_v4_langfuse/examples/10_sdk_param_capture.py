"""
Example 10: SDK Parameter Enrichment & Performance Tracking

Demonstrates automatic enrichment of invocation parameters with model defaults
and performance overhead tracking.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from neatlogs.sdk.neatlogs_sdk_v4_langfuse import init, trace, flush, shutdown
from openai import OpenAI


def example_with_enrichment():
    """Demonstrates parameter enrichment with model defaults."""
    print("\n" + "="*60)
    print("SDK Parameter Enrichment & Performance Tracking")
    print("="*60)
    
    # Enable span logging
    os.environ["NEATLOGS_LOG_SPANS"] = "true"
    
    # Initialize with capture_sdk_params=True
    init(
        api_key=os.getenv("NEATLOGS_API_KEY", "test-key"),
        workflow_name="sdk-params-enriched",
        instrumentations=["openai"],
        debug=True,
    )
    
    client = OpenAI()
    
    with trace("enriched_call"):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Say 'hello'"}],
            max_tokens=10
        )
    
    print(f"\nResponse: {response.choices[0].message.content}")
    print("✅ Check spans.log for enriched 'llm.invocation_parameters'")
    
    flush()
    shutdown()


if __name__ == "__main__":
    example_with_enrichment()
    
    print("\n" + "="*60)
    print("✅ Done! Compare the two traces in spans.log")
    print("="*60)
