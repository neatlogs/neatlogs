"""Regression: an early close of run_async must not record ERROR.

Breaking out of `async for` throws GeneratorExit into the wrapper; the wrapper
recorded it as an exception and ended the span ERROR. A routine early close is
not a run failure.
"""

import asyncio

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

import neatlogs
from neatlogs._wrap_utils import set_neatlogs_provider
from neatlogs.google_adk import wrap_google_adk

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


def _event(text):
    part = type("P", (), {"text": text})()
    content = type("C", (), {"parts": [part]})()
    return type("Ev", (), {"content": content, "author": "a", "actions": None})()


class FakeRunner:
    app_name = "app"
    agent = None

    async def run_async(self, user_id=None, session_id=None, new_message=None):
        yield _event("one")
        yield _event("two")


@pytest.mark.asyncio
async def test_early_break_ends_span_ok_not_error():
    exporter = _init()
    runner = wrap_google_adk(FakeRunner())
    seen = []
    async for event in runner.run_async(user_id="u", session_id="s", new_message=None):
        seen.append(event.content.parts[0].text)
        break
    await asyncio.sleep(0.05)
    assert seen == ["one"]
    spans = [s for s in exporter.get_finished_spans() if s.name == "google_adk.runner.run_async"]
    assert spans, "wrapper span did not end after early close"
    span = spans[0]
    assert span.status.status_code != StatusCode.ERROR
    assert not [e for e in span.events if e.name == "exception"]
