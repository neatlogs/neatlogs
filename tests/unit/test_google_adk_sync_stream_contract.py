"""Regression: wrapped Runner.run() (sync) must stream events lazily.

The wrapper drained the entire event generator before returning, so callers saw
no events until the run finished, and a mid-run error lost every partial event.
"""

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

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

    def __init__(self, fail=False):
        self.progress = []
        self.fail = fail

    def run(self, user_id=None, session_id=None, new_message=None):
        self.progress.append("yield-1")
        yield _event("first")
        self.progress.append("yield-2")
        if self.fail:
            raise RuntimeError("mid-run failure")
        yield _event("second")
        self.progress.append("done")


def test_wrapped_run_streams_lazily():
    _init()
    runner = wrap_google_adk(FakeRunner())
    events = runner.run(user_id="u", session_id="s", new_message=None)
    assert runner.progress == []  # nothing consumed before iteration
    first = next(events)
    assert runner.progress == ["yield-1"]
    assert first.content.parts[0].text == "first"


def test_wrapped_run_preserves_partial_events_on_error():
    _init()
    runner = wrap_google_adk(FakeRunner(fail=True))
    events = runner.run(user_id="u", session_id="s", new_message=None)
    first = next(events)
    assert first.content.parts[0].text == "first"
    with pytest.raises(RuntimeError):
        next(events)
