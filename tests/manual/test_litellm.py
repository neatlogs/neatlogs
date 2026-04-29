"""
TEST 5: LiteLLM integration
Covers framework-integrations.md §8 (LiteLLM).

What to verify:
  1. `instrumentations=["litellm"]` instruments LiteLLM calls automatically.
  2. The `@neatlogs.span(kind="WORKFLOW")` wrapper creates a parent WORKFLOW span.
  3. The inner `neatlogs.trace("llm_call", kind="LLM", ...)` captures the
     SystemPromptTemplate and UserPromptTemplate on the span.
  4. In the NeatLogs dashboard (workflow "test-litellm"), you see:
       - A WORKFLOW span "run"
       - A child LLM span "llm_call" with prompt template + variables captured
       - The LiteLLM call auto-instrumented as a child LLM span

Prerequisites:
    pip install neatlogs[litellm]
    Set OPENAI_API_KEY (or any provider key supported by LiteLLM)

Run:
    NEATLOGS_API_KEY=<your-key> OPENAI_API_KEY=<openai-key> python tests/manual/test_litellm.py

Expected output (no errors):
    [litellm] response: <some text>
    [litellm] flush done
    [litellm] shutdown done
    PASS
"""

import os

import neatlogs
from neatlogs import SystemPromptTemplate, UserPromptTemplate

neatlogs.init(
    api_key=None,  # reads NEATLOGS_API_KEY from env
    endpoint=os.environ.get("NEATLOGS_ENDPOINT", "https://staging-cloud.neatlogs.com"),
    workflow_name="test-litellm",
    instrumentations=["litellm"],
)

from litellm import completion

sys_tpl = SystemPromptTemplate("You are a helpful assistant.")
user_tpl = UserPromptTemplate("{{query}}")


@neatlogs.span(kind="WORKFLOW")
def run(query: str) -> str:
    with neatlogs.trace(
        "llm_call", kind="LLM", prompt_template=sys_tpl, user_prompt_template=user_tpl
    ):
        msgs = [
            {"role": "system", "content": sys_tpl.compile()},
            {"role": "user", "content": user_tpl.compile(query=query)},
        ]
        response = completion(
            model="azure/gpt-5-nano",
            messages=msgs,
            api_key=os.environ.get("AZURE_API_KEY"),
            api_base=os.environ.get("AZURE_API_BASE"),
            api_version=os.environ.get("AZURE_API_VERSION", "2025-01-01-preview"),
        )
    return response.choices[0].message.content


result = run("Say hello in one word.")
print(f"[litellm] response: {result!r}")

neatlogs.flush()
print("[litellm] flush done")

neatlogs.shutdown()
print("[litellm] shutdown done")

print("PASS")
