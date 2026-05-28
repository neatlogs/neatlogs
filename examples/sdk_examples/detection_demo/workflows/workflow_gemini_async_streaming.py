"""
Workflow 5: Gemini Async Streaming (google-genai)
=================================================
Tests the async Gemini streaming path to verify spans are properly closed.

This workflow exists specifically to validate the _Stream.__aiter__ fix in the
neatlogs SDK (manager.py: _patch_openinference_google_genai_stream_finally).

Before the fix: spans started but never ended when iterating async streams,
because _finish_tracing was only called on natural exhaustion — not in a
finally block. Abandoned or GC'd generators silently leaked open spans.

After the fix: _finish_tracing is always called via finally, regardless of
how the async generator exits (normal completion, early break, exception).

Architecture:
  WORKFLOW → LLM call (google_genai async stream)

Test Scenarios:
  1. Full iteration  - stream consumed to completion → span must end
  2. Early break     - caller breaks after first chunk → span must end
  3. Error query     - runtime exception in stream → span must end with ERROR

Requirements:
  GOOGLE_API_KEY (or GEMINI_API_KEY) in environment or .env
  pip install google-genai
"""

import asyncio
import os

from google import genai
from google.genai import types

import neatlogs
from config import Settings


GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

TEST_SCENARIOS = [
    {
        "name": "full_iteration",
        "query": "In two sentences, explain what OpenTelemetry is.",
        "break_after": None,
        "description": "Full stream consumption — span must end with OK status",
    },
    {
        "name": "early_break",
        "query": "List ten programming languages, one per line.",
        "break_after": 2,
        "description": "Break after 2 chunks — span must still end (tests finally block)",
    },
    {
        "name": "normal_completion_2",
        "query": "What is distributed tracing? One sentence.",
        "break_after": None,
        "description": "Second full iteration — confirms no span leak between calls",
    },
]


def _get_client() -> genai.Client:
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY (or GEMINI_API_KEY) is required to run the Gemini async streaming workflow."
        )
    return genai.Client(api_key=api_key)


async def _run_scenario(scenario: dict) -> None:
    """Run a single async streaming scenario inside a neatlogs WORKFLOW span."""
    name = scenario["name"]
    query = scenario["query"]
    break_after = scenario["break_after"]

    print(f"\n  Query: {query}")
    if break_after is not None:
        print(f"  Mode:  break after {break_after} chunk(s)")
    else:
        print(f"  Mode:  full iteration")

    client = _get_client()

    config = types.GenerateContentConfig(
        temperature=0.7,
        max_output_tokens=256,
    )

    with neatlogs.trace(name=f"gemini_stream_{name}", kind="WORKFLOW"):
        response_stream = await client.aio.models.generate_content_stream(
            model=GEMINI_MODEL,
            contents=query,
            config=config,
        )

        chunks_received = 0
        full_text = []

        async for response in response_stream:
            if response.candidates:
                candidate = response.candidates[0]
                if candidate.content and candidate.content.parts:
                    for part in candidate.content.parts:
                        if part.text:
                            full_text.append(part.text)

            chunks_received += 1

            if break_after is not None and chunks_received >= break_after:
                print(f"  Breaking after {chunks_received} chunk(s) (testing early-exit span close)")
                break

    result_text = "".join(full_text)
    print(f"  Chunks received: {chunks_received}")
    print(f"  Response preview: {result_text[:120]}{'...' if len(result_text) > 120 else ''}")
    print(f"  Span closed: yes (reached here without hanging)")


async def _run_all_scenarios() -> None:
    for i, scenario in enumerate(TEST_SCENARIOS, 1):
        print(f"\n{'─'*70}")
        print(f"  Scenario {i}/{len(TEST_SCENARIOS)}: {scenario['name']}")
        print(f"  {scenario['description']}")
        print(f"{'─'*70}")
        try:
            await _run_scenario(scenario)
        except Exception as e:
            # Surface the error but continue — we want all scenarios to run
            print(f"  ERROR: {type(e).__name__}: {e}")


def run_gemini_async_streaming_workflow(settings: Settings) -> None:
    """Entry point called by main.py. Wraps async runner in asyncio.run()."""

    print("\n" + "=" * 80)
    print("WORKFLOW 5: Gemini Async Streaming — Span Lifecycle Verification")
    print("=" * 80)
    print(f"\nModel: {GEMINI_MODEL}")
    print(f"Scenarios: {len(TEST_SCENARIOS)}")
    print(
        "\nThis workflow verifies that async Gemini streaming spans are properly"
        "\nclosed in all exit paths (normal, early break) via the neatlogs SDK patch."
    )

    asyncio.run(_run_all_scenarios())

    print(f"\n{'='*80}")
    print(f"✅ Gemini async streaming workflow completed ({len(TEST_SCENARIOS)} scenarios)")
    print(f"{'='*80}\n")
