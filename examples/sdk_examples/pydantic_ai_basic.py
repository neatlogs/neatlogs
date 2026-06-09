"""
Pydantic AI Python example with Neatlogs.

Wrap a Pydantic AI ``Agent`` with ``neatlogs.wrap()`` and its runs, model calls,
and tool calls are captured (AGENT root + LLM/TOOL children). The wrapped agent
run self-roots, so a single ``agent.run_sync(...)`` renders with no extra wrapper.

Points the model at OpenRouter (OpenAI-compatible) so it runs with just an
OpenRouter key.

Run:
    python examples/sdk_examples/pydantic_ai_basic.py

Env:
    OPENROUTER_API_KEY (required)
    PYDANTIC_AI_MODEL  (default: openai/gpt-4o-mini)
    NEATLOGS_API_KEY, NEATLOGS_ENDPOINT (default http://localhost:4100)
"""

import os

import neatlogs


def main() -> None:
    neatlogs.init(
        api_key=os.getenv("NEATLOGS_API_KEY", ""),
        endpoint=os.getenv("NEATLOGS_ENDPOINT", "http://localhost:4100"),
        workflow_name="pydantic-ai-basic-py",
        tags=["pydantic-ai", "python", "basic"],
    )

    # Import AFTER init().
    from pydantic_ai import Agent
    from pydantic_ai.models.openai import OpenAIModel
    from pydantic_ai.providers.openai import OpenAIProvider

    model_name = os.getenv("PYDANTIC_AI_MODEL", "openai/gpt-4o-mini")
    model = OpenAIModel(
        model_name,
        provider=OpenAIProvider(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY", ""),
        ),
    )

    agent = Agent(model, system_prompt="You are concise and helpful.")

    @agent.tool_plain
    def word_count(text: str) -> int:
        """Return the number of words in text."""
        return len(text.split())

    agent = neatlogs.wrap(agent)

    result = agent.run_sync("In one sentence, what is Pydantic AI?")
    print(result.output)

    neatlogs.flush()
    neatlogs.shutdown()


if __name__ == "__main__":
    main()
