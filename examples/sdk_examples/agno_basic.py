"""
Agno Python example with Neatlogs.

Wrap an Agno ``Agent`` with ``neatlogs.wrap()`` and its runs, model calls, and
tool calls are captured (AGENT root + LLM/TOOL children). The wrapped run
self-roots, so a single ``agent.run(...)`` renders with no extra wrapper.

Points Agno's OpenAIChat model at OpenRouter (OpenAI-compatible) so it runs with
just an OpenRouter key.

Run:
    python examples/sdk_examples/agno_basic.py

Env:
    OPENROUTER_API_KEY (required)
    AGNO_MODEL         (default: openai/gpt-4o-mini)
    NEATLOGS_API_KEY, NEATLOGS_ENDPOINT (default http://localhost:4100)
"""

import os

import neatlogs


def main() -> None:
    neatlogs.init(
        api_key=os.getenv("NEATLOGS_API_KEY", ""),
        endpoint=os.getenv("NEATLOGS_ENDPOINT", "http://localhost:4100"),
        workflow_name="agno-basic-py",
        tags=["agno", "python", "basic"],
    )

    # Import AFTER init().
    from agno.agent import Agent
    from agno.models.openai import OpenAIChat

    model_id = os.getenv("AGNO_MODEL", "openai/gpt-4o-mini")
    agent = Agent(
        model=OpenAIChat(
            id=model_id,
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY", ""),
        ),
        instructions="You are concise and helpful.",
        markdown=False,
    )

    agent = neatlogs.wrap(agent)

    result = agent.run("In one sentence, what is Agno?")
    print(getattr(result, "content", result))

    neatlogs.flush()
    neatlogs.shutdown()


if __name__ == "__main__":
    main()
