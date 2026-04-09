"""
Entry point for the reasoning model / LLM params verification workflow.

Verifies that the SDK correctly captures:
  1. reasoning_effort + max_completion_tokens in llm.invocation_parameters (o4-mini)
  2. temperature, top_p, presence_penalty, frequency_penalty, seed, max_tokens (gpt-4o)
  3. extended thinking config + thinking content blocks (claude-3-7-sonnet)
  4. neatlogs.llm.token_count.reasoning > 0 for o4-mini and claude-3-7-sonnet
  5. neatlogs.llm.metrics.ttft_ms on streaming spans (gpt-4o + claude-3-7-sonnet)

Usage:
    python main.py

Required env vars:
    NEATLOGS_API_KEY
    AZURE_OPENAI_ENDPOINT
    AZURE_OPENAI_API_KEY
    AZURE_LLM_DEPLOYMENT          (supports temperature, top_p, etc.)
    AZURE_REASONING_DEPLOYMENT    (supports max_completion_tokens)
    GEMINI_API_KEY
    AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION  (Bedrock, for Anthropic agent)

Optional env vars:
    OPENAI_API_VERSION       (default: 2025-01-01-preview)
    GEMINI_MODEL             (default: gemini-2.5-flash)
"""

import os
import sys

# Add local SDK to path
_sdk_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if _sdk_root not in sys.path:
    sys.path.insert(0, _sdk_root)

from dotenv import load_dotenv
load_dotenv()

os.environ.setdefault("NEATLOGS_LOG_SPANS", "true")
os.environ.setdefault("NEATLOGS_LOG_SPANS_FILE", "reasoning_model_spans.log")
os.environ.setdefault("NEATLOGS_LOG_RAW_SPANS", "true")
os.environ.setdefault("NEATLOGS_LOG_RAW_SPANS_FILE", "reasoning_model_raw_spans.log")

import neatlogs

neatlogs.init(
    api_key=os.getenv("NEATLOGS_API_KEY", ""),
    endpoint=os.getenv("NEATLOGS_ENDPOINT", "http://localhost:4100"),
    workflow_name="reasoning-model-verification",
    tags=["reasoning", "openai", "anthropic", "params-verification"],
    instrumentations=["openai", "anthropic", "langchain", "google_genai"],
    debug=True,
)

from agents import PROBLEM, openai_reasoning_agent, openai_full_params_agent, anthropic_thinking_agent, langchain_openai_agent, gemini_async_agent


@neatlogs.span(kind="WORKFLOW", name="reasoning_verification_workflow")
def run():
    print(f"\nProblem: {PROBLEM}\n")

    print("\n=== Agent 1: Azure OpenAI (non-streaming, reasoning_effort=high) ===")
    r1 = openai_reasoning_agent(PROBLEM)
    print(f"\nAnswer:\n{r1}")

    print("\n=== Agent 2: Azure OpenAI (streaming, temperature/top_p/seed/penalties) ===")
    r2 = openai_full_params_agent(PROBLEM)

    print("\n=== Agent 3: claude-sonnet-4-6 via Bedrock (streaming, extended thinking) ===")
    r3 = anthropic_thinking_agent(PROBLEM)

    print("\n=== Agent 4: LangChain AzureChatOpenAI (temperature/max_tokens/top_p) ===")
    langchain_openai_agent(PROBLEM)

    print("\n=== Agent 5: Gemini async streaming (temperature/maxOutputTokens/top_p) ===")
    gemini_async_agent(PROBLEM)

    print("\n\n--- WHAT TO CHECK IN reasoning_model_spans.log ---")
    print("1. o4-mini span:")
    print("   neatlogs.llm.invocation_parameters contains reasoning_effort='high' + max_completion_tokens=16000")
    print("   neatlogs.llm.token_count.reasoning > 0")
    print("2. gpt-4o span:")
    print("   neatlogs.llm.invocation_parameters contains temperature=0.7, top_p=0.9,")
    print("   presence_penalty=0.1, frequency_penalty=0.1, seed=42, max_tokens=1000")
    print("   neatlogs.llm.metrics.ttft_ms > 0 (streaming)")
    print("3. claude-3-7-sonnet span:")
    print("   neatlogs.llm.invocation_parameters contains thinking config")
    print("   llm.output_messages contains thinking content blocks")
    print("   neatlogs.llm.token_count.reasoning > 0")
    print("   neatlogs.llm.metrics.ttft_ms > 0 (streaming)")


if __name__ == "__main__":
    run()
    neatlogs.flush()
    neatlogs.shutdown()
