"""
OpenAI Agents SDK Python example with Neatlogs.

Register the Neatlogs trace processor once at startup and the OpenAI Agents SDK
emits its own WORKFLOW/AGENT root with LLM/TOOL children — the run self-roots,
so no extra wrapper is needed.

Points the Agents SDK at OpenRouter (OpenAI-compatible) so it runs with just an
OpenRouter key.

Run:
    python examples/sdk_examples/openai_agents_basic.py

Env:
    OPENROUTER_API_KEY (required)
    OPENAI_AGENTS_MODEL (default: openai/gpt-4o-mini)
    NEATLOGS_API_KEY, NEATLOGS_ENDPOINT (default http://localhost:4100)
"""

import os

import neatlogs


def main() -> None:
    neatlogs.init(
        api_key=os.getenv("NEATLOGS_API_KEY", ""),
        endpoint=os.getenv("NEATLOGS_ENDPOINT", "http://localhost:4100"),
        workflow_name="openai-agents-basic-py",
        tags=["openai-agents", "python", "basic"],
    )

    # Import AFTER init().
    from agents import (
        Agent,
        OpenAIChatCompletionsModel,
        Runner,
        add_trace_processor,
        function_tool,
        set_tracing_disabled,
    )
    from openai import AsyncOpenAI

    # Register the Neatlogs processor once. (OpenAI's own tracing exporter would
    # need an OpenAI key; disable it and rely on Neatlogs.)
    add_trace_processor(neatlogs.openai_agents_processor())
    set_tracing_disabled(False)

    client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY", ""),
    )
    model = OpenAIChatCompletionsModel(
        model=os.getenv("OPENAI_AGENTS_MODEL", "openai/gpt-4o-mini"),
        openai_client=client,
    )

    @function_tool
    def word_count(text: str) -> int:
        """Return the number of words in text."""
        return len(text.split())

    agent = Agent(
        name="Assistant",
        instructions="You are concise. Use tools when helpful.",
        model=model,
        tools=[word_count],
    )

    result = Runner.run_sync(agent, "In one sentence, what is the OpenAI Agents SDK?")
    print(result.final_output)

    neatlogs.flush()
    neatlogs.shutdown()


if __name__ == "__main__":
    main()
