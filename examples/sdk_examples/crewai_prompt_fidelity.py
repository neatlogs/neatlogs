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
PROMPT_TARGET_TAIL_MARKER = "PROMPT_FIDELITY_TARGET_TAIL"

TOOL_OUTPUT_START_MARKER = "TOOL_FIDELITY_OUTPUT_START"
TOOL_NEAR_10000_MARKER = "TOOL_FIDELITY_NEAR_10000"
TOOL_TAIL_AFTER_10000_MARKER = "TOOL_FIDELITY_TAIL_AFTER_10000"

DEFAULT_PROMPT_TAIL_OFFSET = 12_000
DEFAULT_TOOL_OUTPUT_CHARS = 12_500

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


def build_long_task_description(
    target_tail_offset: int = DEFAULT_PROMPT_TAIL_OFFSET,
) -> str:
    """Build a deterministic task with markers around the capture boundary."""
    probe_body = f"{PROMPT_START_MARKER}\n{NEAR_10000_MARKER}"
    probe_json = json.dumps(_synthetic_document(probe_body), sort_keys=True)
    probe_prompt = _render_task_description(probe_json)
    padding_length = 9_800 - probe_prompt.index(NEAR_10000_MARKER)
    if padding_length <= 0:
        raise AssertionError("The fixed task text already reaches the near marker")

    base_long_body = (
        f"{PROMPT_START_MARKER}\n"
        f"{'x' * padding_length}{NEAR_10000_MARKER}\n"
        f"{'y' * 2_500}{TAIL_AFTER_12000_MARKER}"
    )
    target_probe_json = json.dumps(
        _synthetic_document(f"{base_long_body}\n{PROMPT_TARGET_TAIL_MARKER}"),
        sort_keys=True,
    )
    target_probe_prompt = _render_task_description(target_probe_json)
    target_padding_length = max(
        0,
        target_tail_offset
        - target_probe_prompt.index(PROMPT_TARGET_TAIL_MARKER),
    )
    long_body = (
        f"{base_long_body}\n"
        f"{'z' * target_padding_length}{PROMPT_TARGET_TAIL_MARKER}"
    )
    document_json = json.dumps(_synthetic_document(long_body), sort_keys=True)
    description = _render_task_description(document_json)

    near_offset = description.index(NEAR_10000_MARKER)
    tail_offset = description.index(TAIL_AFTER_12000_MARKER)
    target_tail_actual_offset = description.index(PROMPT_TARGET_TAIL_MARKER)
    assert 9_500 < near_offset < 10_000
    assert tail_offset > 12_000
    assert target_tail_actual_offset >= target_tail_offset
    return description


def build_tool_fidelity_output(
    target_chars: int = DEFAULT_TOOL_OUTPUT_CHARS,
) -> str:
    """Build a plain string whose final marker is beyond 10,000 characters."""
    prefix = f"{TOOL_OUTPUT_START_MARKER}\n"
    near_padding_length = 9_800 - len(prefix)
    if near_padding_length <= 0:
        raise AssertionError("The tool-output prefix already reaches the near marker")

    through_near_marker = (
        f"{prefix}{'u' * near_padding_length}{TOOL_NEAR_10000_MARKER}\n"
    )
    tail_padding_length = max(0, 10_050 - len(through_near_marker))
    output = (
        f"{through_near_marker}"
        f"{'v' * tail_padding_length}{TOOL_TAIL_AFTER_10000_MARKER}"
    )
    if len(output) < target_chars:
        output = f"{output}{'w' * (target_chars - len(output))}"

    assert output.index(TOOL_NEAR_10000_MARKER) < 10_000
    assert output.index(TOOL_TAIL_AFTER_10000_MARKER) > 10_000
    return output


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


def _positive_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise SystemExit(f"{name} must be a positive integer") from exc
    if value <= 0:
        raise SystemExit(f"{name} must be a positive integer")
    return value


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
    from crewai.tools import BaseTool

    run_id = os.getenv("CREWAI_PROMPT_RUN_ID", str(int(time.time())))
    run_label = os.getenv("CREWAI_FIDELITY_RUN_LABEL", "default")
    prompt_target_offset = _positive_int_env(
        "CREWAI_PROMPT_TARGET_CHARS",
        DEFAULT_PROMPT_TAIL_OFFSET,
    )
    tool_output_chars = _positive_int_env(
        "CREWAI_TOOL_OUTPUT_CHARS",
        DEFAULT_TOOL_OUTPUT_CHARS,
    )
    description = build_long_task_description(prompt_target_offset)
    near_offset = description.index(NEAR_10000_MARKER)
    tail_offset = description.index(TAIL_AFTER_12000_MARKER)
    target_tail_offset = description.index(PROMPT_TARGET_TAIL_MARKER)

    class FidelityPayloadProbeTool(BaseTool):
        name: str = "fidelity_payload_probe"
        description: str = (
            "Return a deterministic plain-text payload for telemetry fidelity checks."
        )

        def _run(self, target_chars: int = DEFAULT_TOOL_OUTPUT_CHARS) -> str:
            return build_tool_fidelity_output(target_chars)

    fidelity_tool = FidelityPayloadProbeTool()

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
                "run_label": run_label,
                "task_chars": len(description),
                "tool_output_chars": tool_output_chars,
            },
        ) as trace_span:
            span_context = trace_span.get_span_context()
            trace_id = f"{span_context.trace_id:032x}"
            tool_output = fidelity_tool.run(target_chars=tool_output_chars)
            result = await crew.kickoff_async()
    finally:
        neatlogs.flush()
        await asyncio.sleep(3)

    result_text = str(getattr(result, "raw", result))
    tool_tail_offset = tool_output.index(TOOL_TAIL_AFTER_10000_MARKER)
    print(f"run_label={run_label}")
    print(f"run_id={run_id}")
    print(f"trace_id={trace_id}")
    print(
        f"task_chars={len(description)} near_offset={near_offset} "
        f"tail_offset={tail_offset} target_tail_offset={target_tail_offset}"
    )
    print(
        f"tool_output_chars={len(tool_output)} "
        f"tool_tail_offset={tool_tail_offset}"
    )
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
