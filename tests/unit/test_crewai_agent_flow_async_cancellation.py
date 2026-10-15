import asyncio

import pytest

from neatlogs import crewai as instrumentation


class _Span:
    def __init__(self, name, attributes=None):
        self.name = name
        self.attributes = dict(attributes or {})
        self.status = None
        self.ended = False
        self.exceptions = []

    def set_attribute(self, key, value):
        self.attributes[key] = value

    def set_status(self, status, *_args):
        self.status = status

    def record_exception(self, exc, *_args):
        self.exceptions.append(exc)

    def end(self):
        self.ended = True


class _Tracer:
    def __init__(self):
        self.spans = []

    def start_span(self, name, attributes=None, **_kwargs):
        span = _Span(name, attributes)
        self.spans.append(span)
        return span


@pytest.fixture
def tracer(monkeypatch):
    tracer = _Tracer()
    monkeypatch.setattr(instrumentation, "get_tracer", lambda: tracer)
    monkeypatch.setattr(instrumentation, "attach_as_current", lambda _span: object())
    monkeypatch.setattr(instrumentation, "detach", lambda _token: None)
    return tracer


@pytest.mark.asyncio
async def test_agent_kickoff_async_cancellation_ends_span(tracer):
    class Agent:
        role = "demo"

        async def kickoff_async(self, messages=None):
            await asyncio.Event().wait()

    agent = Agent()
    instrumentation._patch_agent_kickoff(agent)
    task = asyncio.create_task(agent.kickoff_async(messages=["x"]))
    await asyncio.sleep(0)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert len(tracer.spans) == 1
    assert tracer.spans[0].ended is True
    assert tracer.spans[0].status.name == "UNSET"
    assert tracer.spans[0].attributes["neatlogs.stream.cancelled"] is True
    assert tracer.spans[0].exceptions == []


@pytest.mark.asyncio
async def test_flow_kickoff_async_cancellation_ends_span(tracer):
    class Flow:
        async def kickoff_async(self, inputs=None):
            await asyncio.Event().wait()

    instrumentation._patch_flow_class(Flow)
    task = asyncio.create_task(Flow().kickoff_async(inputs={"x": 1}))
    await asyncio.sleep(0)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert len(tracer.spans) == 1
    assert tracer.spans[0].ended is True
    assert tracer.spans[0].status.name == "UNSET"
    assert tracer.spans[0].attributes["neatlogs.stream.cancelled"] is True
    assert tracer.spans[0].exceptions == []


@pytest.mark.asyncio
async def test_flow_kickoff_async_error_still_marks_error(tracer):
    class Flow:
        async def kickoff_async(self, inputs=None):
            raise RuntimeError("flow failed")

    instrumentation._patch_flow_class(Flow)

    with pytest.raises(RuntimeError, match="flow failed"):
        await Flow().kickoff_async(inputs={"x": 1})

    assert len(tracer.spans) == 1
    assert tracer.spans[0].ended is True
    assert tracer.spans[0].status.name == "ERROR"
    assert "neatlogs.stream.cancelled" not in tracer.spans[0].attributes
    assert isinstance(tracer.spans[0].exceptions[0], RuntimeError)
