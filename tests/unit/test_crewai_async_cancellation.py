import asyncio

import pytest

from neatlogs import crewai as instrumentation


class _Span:
    def __init__(self, name, attributes=None):
        self.name = name
        self.attributes = dict(attributes or {})
        self.status = None
        self.ended = False

    def set_attribute(self, key, value):
        self.attributes[key] = value

    def set_status(self, status, *_args):
        self.status = status

    def record_exception(self, *_args):
        pass

    def end(self):
        self.ended = True


class _Tracer:
    def __init__(self):
        self.spans = []

    def start_span(self, name, attributes=None, **_kwargs):
        span = _Span(name, attributes)
        self.spans.append(span)
        return span


@pytest.mark.asyncio
async def test_kickoff_for_each_async_cancellation_ends_span(monkeypatch):
    class Crew:
        async def kickoff_for_each_async(self, inputs):
            await asyncio.Event().wait()

    tracer = _Tracer()
    monkeypatch.setattr(instrumentation, "get_tracer", lambda: tracer)
    monkeypatch.setattr(instrumentation, "attach_as_current", lambda _span: object())
    monkeypatch.setattr(instrumentation, "detach", lambda _token: None)
    instrumentation._patch_crew_class(Crew)

    task = asyncio.create_task(Crew().kickoff_for_each_async([{"x": 1}]))
    await asyncio.sleep(0)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert len(tracer.spans) == 1
    assert tracer.spans[0].name == "crewai.crew.kickoff_for_each_async"
    assert tracer.spans[0].ended is True
    assert tracer.spans[0].status.name == "ERROR"
