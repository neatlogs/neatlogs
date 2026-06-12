"""
Hermes Agent (NousResearch/hermes-agent) example with Neatlogs.

Hermes is a Python agentic loop whose public surface is the ``AIAgent`` class in
the top-level ``run_agent`` module. neatlogs.wrap(agent) patches
``AIAgent.run_conversation`` (AGENT span) and ``ToolRegistry.dispatch`` (TOOL
spans); Hermes' LLM calls go through the ``openai`` SDK (pointed at OpenRouter),
which neatlogs' OpenAI instrumentation captures as LLM spans.

So one run produces:
    AGENT  hermes.run_conversation
      ↳ LLM   chat.completions.create   (via OpenRouter)
      ↳ TOOL  hermes.tool.<name>        (if the model calls a tool)

Install (Hermes needs Python >= 3.11):
    git clone https://github.com/NousResearch/hermes-agent.git
    pip install -e ./hermes-agent
    pip install -e /path/to/neatlogs        # the local SDK

Run:
    OPENROUTER_API_KEY=... python examples/sdk_examples/hermes_agent_basic.py

Env:
    OPENROUTER_API_KEY (required — Hermes routes LLM calls through OpenRouter)
    HERMES_MODEL       (default: openai/gpt-4o-mini, OpenRouter slug)
    NEATLOGS_API_KEY, NEATLOGS_ENDPOINT (default http://localhost:4100)
"""

import os

import neatlogs
from run_agent import AIAgent


def main() -> None:
    neatlogs.init(
        api_key=os.getenv("NEATLOGS_API_KEY", ""),
        endpoint=os.getenv("NEATLOGS_ENDPOINT", "http://localhost:4100"),
        workflow_name="hermes-agent-basic-py",
        # Hermes' LLM calls flow through the openai SDK — enable that so LLM
        # spans are captured alongside the AGENT/TOOL spans the wrapper emits.
        instrumentations=["hermes", "openai"],
        tags=["hermes", "python", "basic"],
    )

    model = os.getenv("HERMES_MODEL", "openai/gpt-4o-mini")

    # AIAgent defaults to OpenRouter (base_url https://openrouter.ai/api/v1,
    # api_key from OPENROUTER_API_KEY). neatlogs.wrap patches the class so the
    # AGENT + TOOL spans are emitted; no per-call code changes needed.
    agent = neatlogs.wrap(
        AIAgent(
            model=model,
            api_key=os.getenv("OPENROUTER_API_KEY"),
            max_iterations=4,
            quiet_mode=True,
        )
    )

    result = agent.run_conversation(
        "In one short paragraph, explain what distributed tracing is and why it helps debug agents."
    )

    # run_conversation returns a dict with the loop summary.
    if isinstance(result, dict):
        msgs = result.get("messages") or []
        final = next(
            (m.get("content") for m in reversed(msgs) if m.get("role") == "assistant" and m.get("content")),
            None,
        )
        if final:
            print("\n=== Hermes response ===\n" + str(final))
        print(f"\nAPI calls: {result.get('api_calls')} | completed: {result.get('completed')}")

    neatlogs.flush()
    neatlogs.shutdown()


if __name__ == "__main__":
    main()
