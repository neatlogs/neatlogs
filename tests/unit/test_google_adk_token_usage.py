"""Regression: wrapped google_adk Runner spans must record token usage.

ADK events carry GenerateContentResponseUsageMetadata on `event.usage_metadata`;
EventActions has no usage fields. The wrapper read usage from `event.actions`,
so runner spans never recorded token counts.
"""

import asyncio

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import neatlogs
from neatlogs._wrap_utils import set_neatlogs_provider

pytest.importorskip("google.adk")


def _init():
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    set_neatlogs_provider(provider)
    neatlogs.init(
        api_key="test-key",
        disable_export=True,
        instrumentations=[],
        tracer_provider=provider,
        register_shutdown_handlers=False,
    )
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return exporter


async def _run(exporter, sync):
    from google.adk.agents import LlmAgent
    from google.adk.models.base_llm import BaseLlm
    from google.adk.models.llm_response import LlmResponse
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai import types

    class LocalLlm(BaseLlm):
        async def generate_content_async(self, llm_request, stream=False):
            yield LlmResponse(
                content=types.Content(role="model", parts=[types.Part(text="answer")]),
                turn_complete=True,
                finish_reason=types.FinishReason.STOP,
                usage_metadata=types.GenerateContentResponseUsageMetadata(
                    prompt_token_count=8,
                    candidates_token_count=5,
                    total_token_count=13,
                ),
            )

    sessions = InMemorySessionService()
    await sessions.create_session(app_name="app", user_id="u", session_id="s")
    runner = Runner(
        app_name="app",
        agent=LlmAgent(name="a", model=LocalLlm(model="m")),
        session_service=sessions,
    )
    runner = neatlogs.wrap(runner)
    message = types.Content(role="user", parts=[types.Part(text="q")])
    if sync:
        for _ in runner.run(user_id="u", session_id="s", new_message=message):
            pass
    else:
        async for _ in runner.run_async(user_id="u", session_id="s", new_message=message):
            pass
    name = "google_adk.runner.run" if sync else "google_adk.runner.run_async"
    spans = [s for s in exporter.get_finished_spans() if s.name == name]
    assert spans, f"no {name} span"
    return spans[0]


@pytest.mark.asyncio
async def test_runner_span_records_token_usage_async():
    exporter = _init()
    span = await _run(exporter, sync=False)
    attrs = dict(span.attributes)
    assert attrs["neatlogs.llm.token_count.prompt"] == 8
    assert attrs["neatlogs.llm.token_count.completion"] == 5
    assert attrs["neatlogs.llm.token_count.total"] == 13


@pytest.mark.asyncio
async def test_runner_span_records_token_usage_sync():
    exporter = _init()
    span = await _run(exporter, sync=True)
    attrs = dict(span.attributes)
    assert attrs["neatlogs.llm.token_count.prompt"] == 8
    assert attrs["neatlogs.llm.token_count.completion"] == 5
    assert attrs["neatlogs.llm.token_count.total"] == 13
