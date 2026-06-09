"""
CrewAI Python example with Neatlogs.

Wrap a CrewAI ``Crew`` with ``neatlogs.wrap()`` and the crew run emits a
WORKFLOW root with AGENT / TASK / LLM children. CrewAI routes LLM calls through
LiteLLM, so pair the wrap with ``instrumentations=["crewai", "openai"]`` for full
LLM capture. The crew run self-roots, so no extra wrapper is needed.

Points CrewAI's LLM at OpenRouter (OpenAI-compatible) so it runs with just an
OpenRouter key.

Run:
    python examples/sdk_examples/crewai_basic.py

Env:
    OPENROUTER_API_KEY (required)
    CREWAI_MODEL       (default: openai/gpt-4o-mini  — litellm-style slug)
    NEATLOGS_API_KEY, NEATLOGS_ENDPOINT (default http://localhost:4100)
"""

import os

import neatlogs

neatlogs.init(
    api_key=os.getenv("NEATLOGS_API_KEY", ""),
    endpoint=os.getenv("NEATLOGS_ENDPOINT", "http://localhost:4100"),
    workflow_name="crewai-basic-py",
    tags=["crewai", "python", "basic"],
    instrumentations=["crewai", "openai"],
)

# Import AFTER init().
from crewai import Agent, Crew, LLM, Process, Task  # noqa: E402


def main() -> None:
    # CrewAI uses LiteLLM under the hood; "openai/<model>" + base_url routes to
    # OpenRouter. The OPENAI_API_KEY env is what LiteLLM reads for openai/* slugs.
    os.environ.setdefault("OPENAI_API_KEY", os.getenv("OPENROUTER_API_KEY", ""))
    os.environ.setdefault("OPENAI_API_BASE", "https://openrouter.ai/api/v1")

    llm = LLM(
        model=os.getenv("CREWAI_MODEL", "openai/gpt-4o-mini"),
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY", ""),
        temperature=0.3,
    )

    researcher = Agent(
        role="Explainer",
        goal="Explain technical topics in one clear sentence.",
        backstory="You are concise and precise.",
        llm=llm,
        verbose=False,
    )
    task = Task(
        description="Explain what CrewAI is in exactly one sentence.",
        expected_output="One sentence.",
        agent=researcher,
    )

    crew = neatlogs.wrap(Crew(agents=[researcher], tasks=[task], process=Process.sequential, verbose=False))

    result = crew.kickoff()
    print(getattr(result, "raw", result))

    neatlogs.flush()
    neatlogs.shutdown()


if __name__ == "__main__":
    main()
