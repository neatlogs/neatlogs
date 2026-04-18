"""
TEST 5: LiteLLM integration
Covers framework-integrations.md §8 (LiteLLM).

What to verify:
  1. `instrumentations=["litellm"]` instruments LiteLLM calls automatically.
  2. The `@neatlogs.span(kind="WORKFLOW")` wrapper creates a parent WORKFLOW span.
  3. The inner `neatlogs.trace("llm_call", kind="LLM", ...)` captures the
     PromptTemplate and UserPromptTemplate on the span.
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

import neatlogs
from neatlogs import PromptTemplate, UserPromptTemplate

neatlogs.init(
    api_key=None,  # reads NEATLOGS_API_KEY from env
    workflow_name="test-litellm",
    instrumentations=["litellm"],
)

from litellm import completion

sys_tpl = PromptTemplate("You are a helpful assistant.")
user_tpl = UserPromptTemplate("{{query}}")


@neatlogs.span(kind="WORKFLOW")
def run(query: str) -> str:
    with neatlogs.trace("llm_call", kind="LLM",
                        prompt_template=sys_tpl,
                        user_prompt_template=user_tpl):
        msgs = sys_tpl.compile() + user_tpl.compile(query=query)
        response = completion(model="gpt-4o", messages=msgs)
    return response.choices[0].message.content


result = run("Say hello in one word.")
print(f"[litellm] response: {result!r}")

neatlogs.flush()
print("[litellm] flush done")

neatlogs.shutdown()
print("[litellm] shutdown done")

print("PASS")
