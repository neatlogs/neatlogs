"""Generic CrewAI long-prompt fidelity example for NeatLogs.

The example creates a deterministic synthetic document whose task description exceeds
12,000 characters. Stable markers on both sides of the historical 10,000-character
boundary make it easy to verify that the complete runtime user message reaches a trace.

Run from the SDK repository root:

    NEATLOGS_ENV_FILE=/absolute/path/to/.env \
    NEATLOGS_ENDPOINT=http://127.0.0.1:4100 \
    CREWAI_MODEL=openai/gpt-4o-mini \
    .venv/bin/python examples/sdk_examples/crewai_prompt_fidelity.py
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from textwrap import dedent
from typing import Any

from dotenv import load_dotenv

import neatlogs

AGENT_ROLE = "Long Document Analyst"
AGENT_GOAL = "Summarize long synthetic documents without losing details near the end"
AGENT_BACKSTORY = (
    "You inspect technical documents carefully and return concise, structured findings."
)

PROMPT_START_MARKER = "PROMPT_FIDELITY_TASK_START"
NEAR_10000_MARKER = "PROMPT_FIDELITY_NEAR_10000"
TAIL_AFTER_12000_MARKER = "PROMPT_FIDELITY_TAIL_AFTER_12000"

EXPECTED_OUTPUT = dedent("""
    {
        "summary": "string",
        "key_points": ["string"],
        "open_questions": ["string"],
        "tail_marker_seen": "boolean"
    }
    """).strip()


def _synthetic_document(long_body: str) -> dict[str, Any]:
    return {
        "document_id": "synthetic-document-001",
        "title": "Synthetic reliability review",
        "sections": [
            {
                "heading": "Background",
                "content": "This document is generated solely for prompt-fidelity testing.",
            },
            {
                "heading": "Long analysis",
                "content": long_body,
            },
        ],
    }


def _render_task_description(document_json: str) -> str:
    return dedent(f"""
        **Objective:** Analyze the synthetic document and return a structured summary.

        **Instructions:**
        1. Read the complete input, including content near the end.
        2. Summarize the main topic in one sentence.
        3. List the most important technical points.
        4. List any open questions.
        5. Set `tail_marker_seen` to true only if the final fidelity marker is present.
        6. Return JSON only.

        **Input document:**
        {document_json}

        **Required output schema:**
        {EXPECTED_OUTPUT}
        """).strip()


def build_long_task_description() -> str:
    """Build a deterministic task with markers around the capture boundary."""
    probe_body = f"{PROMPT_START_MARKER}\n{NEAR_10000_MARKER}"
    probe_json = json.dumps(_synthetic_document(probe_body), sort_keys=True)
    probe_prompt = _render_task_description(probe_json)
    padding_length = 9_800 - probe_prompt.index(NEAR_10000_MARKER)
    if padding_length <= 0:
        raise AssertionError("The fixed task text already reaches the near marker")

    long_body = (
        f"{PROMPT_START_MARKER}\n"
        f"{'x' * padding_length}{NEAR_10000_MARKER}\n"
        f"{'y' * 2_500}{TAIL_AFTER_12000_MARKER}"
    )
    document_json = json.dumps(_synthetic_document(long_body), sort_keys=True)
    description = _render_task_description(document_json)

    near_offset = description.index(NEAR_10000_MARKER)
    tail_offset = description.index(TAIL_AFTER_12000_MARKER)
    assert 9_500 < near_offset < 10_000
    assert tail_offset > 12_000
    return description


def _load_runtime_environment() -> None:
    env_file = os.getenv("NEATLOGS_ENV_FILE")
    if env_file:
        load_dotenv(Path(env_file), override=False)
    else:
        load_dotenv(override=False)


def _first_env(*names: str) -> str:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    raise SystemExit(f"Missing required environment variable; expected one of: {', '.join(names)}")


def setup_neatlogs() -> None:
    """Initialize automatic CrewAI instrumentation before importing CrewAI."""
    neatlogs.init(
        api_key=_first_env("NEATLOGS_API_KEY"),
        endpoint=_first_env("NEATLOGS_ENDPOINT"),
        workflow_name="crewai-long-prompt-fidelity",
        instrumentations=["crewai"],
        debug=os.getenv("NEATLOGS_DEBUG", "").lower() in {"1", "true", "yes"},
    )


def _build_llm():
    # CrewAI must be imported only after setup_neatlogs() installs its hooks.
    from crewai import LLM

    return LLM(
        model=_first_env("CREWAI_MODEL"),
        api_key=_first_env("OPENAI_API_KEY"),
        max_completion_tokens=600,
        drop_params=True,
    )


async def run_example() -> str:
    # These imports deliberately occur after setup_neatlogs().
    from crewai import Agent, Crew, Task

    run_id = os.getenv("CREWAI_PROMPT_RUN_ID", str(int(time.time())))
    description = build_long_task_description()
    near_offset = description.index(NEAR_10000_MARKER)
    tail_offset = description.index(TAIL_AFTER_12000_MARKER)

    agent = Agent(
        role=AGENT_ROLE,
        goal=AGENT_GOAL,
        backstory=AGENT_BACKSTORY,
        allow_delegation=False,
        max_iter=1,
        llm=_build_llm(),
        verbose=False,
    )
    task = Task(
        description=description,
        expected_output=EXPECTED_OUTPUT,
        agent=agent,
    )
    crew = Crew(agents=[agent], tasks=[task], verbose=False)

    trace_id = ""
    try:
        with neatlogs.trace(
            name="long_prompt_fidelity",
            kind="WORKFLOW",
            metadata={
                "example": "crewai-long-prompt-fidelity",
                "run_id": run_id,
                "task_chars": len(description),
            },
        ) as trace_span:
            span_context = trace_span.get_span_context()
            trace_id = f"{span_context.trace_id:032x}"
            result = await crew.kickoff_async()
    finally:
        neatlogs.flush()
        await asyncio.sleep(3)

    result_text = str(getattr(result, "raw", result))
    print(f"run_id={run_id}")
    print(f"trace_id={trace_id}")
    print(f"task_chars={len(description)} near_offset={near_offset} " f"tail_offset={tail_offset}")
    print(f"result_preview={result_text[:120]!r}")
    return trace_id


def main() -> None:
    _load_runtime_environment()
    setup_neatlogs()
    try:
        asyncio.run(run_example())
    finally:
        neatlogs.shutdown()


if __name__ == "__main__":
    main()
