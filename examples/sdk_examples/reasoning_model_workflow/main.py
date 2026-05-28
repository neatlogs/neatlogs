"""
Entry point for the reasoning model / LLM params verification workflow.

Runs five agents across four providers (Azure OpenAI, Anthropic Bedrock,
LangChain AzureChatOpenAI, Gemini async) to verify that invocation parameters
(reasoning_effort, extended thinking, temperature, top_p, etc.) and reasoning
token counts appear correctly on captured LLM spans.

Usage:
    python main.py

Required env vars:
    NEATLOGS_API_KEY
    AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY
    AZURE_LLM_DEPLOYMENT        (standard params)
    AZURE_REASONING_DEPLOYMENT  (max_completion_tokens)
    GEMINI_API_KEY
    AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION  (Bedrock)
"""

import os

from dotenv import load_dotenv

load_dotenv()

# neatlogs.init() MUST come before any LLM client creation. Google GenAI is
# especially strict — the client caches its transport at construction time.
import neatlogs

neatlogs.init(
    api_key=os.getenv("NEATLOGS_API_KEY"),
    endpoint=os.getenv("NEATLOGS_ENDPOINT"),
    workflow_name="reasoning-model-verification",
    tags=[
        "sdk-examples",
        "reasoning",
        "multi-provider",
        "params-verification",
    ],
    instrumentations=["openai", "anthropic", "langchain", "google_genai"],
)

from agents import (
    PROBLEM,
    openai_reasoning_agent,
    openai_full_params_agent,
    anthropic_thinking_agent,
    langchain_openai_agent,
    gemini_async_agent,
)


@neatlogs.span(kind="WORKFLOW", name="reasoning_verification_workflow")
def run():
    print(f"\nProblem: {PROBLEM}\n")

    print("\n=== Agent 1: Azure OpenAI (non-streaming, reasoning_effort) ===")
    r1 = openai_reasoning_agent(PROBLEM)
    print(f"\nAnswer:\n{r1}")

    print("\n=== Agent 2: Azure OpenAI (streaming, full params) ===")
    openai_full_params_agent(PROBLEM)

    print("\n=== Agent 3: Anthropic Bedrock (streaming, extended thinking) ===")
    anthropic_thinking_agent(PROBLEM)

    print("\n=== Agent 4: LangChain AzureChatOpenAI ===")
    langchain_openai_agent(PROBLEM)

    print("\n=== Agent 5: Gemini async streaming ===")
    gemini_async_agent(PROBLEM)


if __name__ == "__main__":
    run()
    neatlogs.flush()
    neatlogs.shutdown()
