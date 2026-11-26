"""Regression: an early stop of agno run(stream=True) must end the span.

The sync stream wrapper was a plain iterator, so a caller breaking out of
iteration never finalized anything: the span (and its attached context) leaked.
The wrapper is now a generator and finalizes on GeneratorExit.
"""

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import neatlogs
from neatlogs._wrap_utils import set_neatlogs_provider
from neatlogs.agno import _patch_agent


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


class _Event:
    def __init__(self, text):
        self.content = text


class FakeAgent:
    model = None

    def run(self, *args, **kwargs):
        yield _Event("one")
        yield _Event("two")


def test_early_break_ends_span():
    exporter = _init()
    agent = FakeAgent()
    _patch_agent(agent)
    seen = []
    for event in agent.run("hi", stream=True):
        seen.append(event.content)
        break
    assert seen == ["one"]
    import time

    time.sleep(0.05)
    spans = [s for s in exporter.get_finished_spans() if s.name == "agno.fakeagent.run"]
    assert spans, "run span leaked open after early break"
