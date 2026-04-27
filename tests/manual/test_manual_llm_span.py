"""
TEST 3: Manual LLM span attributes
Covers decorators-and-traces.md §5 (Manual LLM Span Attributes).
Also tests the attribute prefix question: do you set `llm.*` or `neatlogs.llm.*`?

The skill file says to use `llm.input_messages.N.message.role` etc. (without `neatlogs.` prefix).
The attribute_processor normalises these to `neatlogs.llm.*` before export.

What to verify:
  1. The pattern compiles and runs without errors.
  2. In the NeatLogs dashboard (workflow "test-manual-llm-span"), the LLM span appears
     with correct input/output messages, model name, and token counts.
  3. Attributes are stored under `neatlogs.llm.*` in the dashboard (normalisation confirmed).

Run:
    NEATLOGS_API_KEY=<your-key> python tests/manual/test_manual_llm_span.py

Expected output (no errors):
    [manual_llm] span created with flat indexed attributes
    [manual_llm] flush done
    [manual_llm] shutdown done
    PASS
"""

import os

from opentelemetry import trace as otel_trace
from opentelemetry.trace import StatusCode

import neatlogs


def main():
    neatlogs.init(
        api_key=None,  # reads NEATLOGS_API_KEY from env
        endpoint=os.environ.get(
            "NEATLOGS_ENDPOINT", "https://staging-cloud.neatlogs.com/api/data/v4/batch"
        ),
        workflow_name="test-manual-llm-span",
        disable_export=False,
    )

    tracer = otel_trace.get_tracer(__name__)

    user_query = "What is the capital of France?"
    response_text = "The capital of France is Paris."

    with tracer.start_as_current_span("manual_llm_call") as span:
        # Mark it as an LLM span
        span.set_attribute("openinference.span.kind", "LLM")

        # Input messages — flat indexed format (from decorators-and-traces.md §5)
        span.set_attribute("llm.input_messages.0.message.role", "system")
        span.set_attribute("llm.input_messages.0.message.content", "You are a helpful assistant.")
        span.set_attribute("llm.input_messages.1.message.role", "user")
        span.set_attribute("llm.input_messages.1.message.content", user_query)

        # Output messages
        span.set_attribute("llm.output_messages.0.message.role", "assistant")
        span.set_attribute("llm.output_messages.0.message.content", response_text)

        # Model metadata
        span.set_attribute("llm.model_name", "gpt-4o")
        span.set_attribute("llm.token_count.prompt", 20)
        span.set_attribute("llm.token_count.completion", 10)
        span.set_attribute("llm.token_count.total", 30)

    print("[manual_llm] span created with flat indexed attributes")

    neatlogs.flush()
    print("[manual_llm] flush done")

    neatlogs.shutdown()
    print("[manual_llm] shutdown done")

    print("PASS")


main()
