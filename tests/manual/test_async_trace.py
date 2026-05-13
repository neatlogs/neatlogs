"""
TEST 2: async trace() + SystemPromptTemplate
Covers troubleshooting.md line 172 — using neatlogs.trace() and SystemPromptTemplate
inside an async function (the reviewer asked if this was tested).

What to verify:
  - `with neatlogs.trace(...)` works correctly inside an async def.
  - The prompt template and variables are captured on the span and appear in
    the NeatLogs dashboard under workflow "test-async-trace".
  - No errors or warnings about event loop blocking.

Run:
    NEATLOGS_API_KEY=<your-key> python tests/manual/test_async_trace.py

Expected output (no errors):
    [async_trace] span + prompt template created
    [async_trace] flush done
    [async_trace] shutdown done
    PASS
"""

import asyncio
import os

import neatlogs
from neatlogs import SystemPromptTemplate, UserPromptTemplate


async def main():
    neatlogs.init(
        api_key=None,  # reads NEATLOGS_API_KEY from env
        endpoint=os.environ.get("NEATLOGS_ENDPOINT", "https://staging-cloud.neatlogs.com"),
        workflow_name="test-async-trace",
    )

    sys_tpl = SystemPromptTemplate("You are a helpful assistant in {{domain}}.")
    user_tpl = UserPromptTemplate("Answer this: {{question}}")

    @neatlogs.span(kind="CHAIN")
    async def async_agent(question: str):
        with neatlogs.trace("prompt", prompt_template=sys_tpl, user_prompt_template=user_tpl):
            # Compile templates — variables are auto-captured into the span
            sys_msg = sys_tpl.compile(domain="science")
            user_msg = user_tpl.compile(question=question)
            # In a real test you'd call an LLM here; we just verify no errors
            return f"sys={sys_msg!r}, user={user_msg!r}"

    result = await async_agent("What is photosynthesis?")
    print(f"[async_trace] span + prompt template created, result={result!r}")

    await asyncio.to_thread(neatlogs.flush)
    print("[async_trace] flush done")

    await asyncio.to_thread(neatlogs.shutdown)
    print("[async_trace] shutdown done")

    print("PASS")


asyncio.run(main())
