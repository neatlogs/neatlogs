import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from neatlogs import agno


@pytest.mark.parametrize("workflow", [False, True])
@pytest.mark.parametrize("configured", [False, True])
@pytest.mark.parametrize("mode", ["stream", "response"])
def test_arun_keeps_native_dispatch_contract(monkeypatch, workflow, configured, mode):
    event = SimpleNamespace(content="synthetic event")
    response = SimpleNamespace(content="synthetic response")
    calls = []

    async def events():
        yield event

    async def result():
        return response

    class Entity:
        name = "contract"
        model = None
        stream = configured

        def arun(self, *args, **kwargs):
            calls.append((args, kwargs))
            stream = kwargs.get("stream")
            if stream is None:
                stream = self.stream
            return events() if stream else result()

    tracer = Mock()
    monkeypatch.setattr(agno, "get_tracer", lambda: tracer)
    monkeypatch.setattr(agno, "attach_as_current", lambda span: None)
    monkeypatch.setattr(agno, "detach", lambda token: None)
    entity = Entity()
    (agno._patch_workflow if workflow else agno._patch_agent)(entity)
    streaming = mode == "stream"
    kwargs = {} if streaming == configured else {"stream": streaming}

    async def run():
        output = entity.arun("synthetic input", **kwargs)
        if streaming:
            assert [item async for item in output] == [event]
        else:
            assert await output is response

    asyncio.run(run())
    assert calls == [(("synthetic input",), kwargs)]
    tracer.start_span.return_value.end.assert_called_once()


def test_real_workflow_streams_without_await(monkeypatch):
    pytest.importorskip("agno")
    from agno.workflow import Workflow
    from agno.workflow.types import StepInput, StepOutput

    async def step(step_input: StepInput):
        return StepOutput(content="synthetic output")

    monkeypatch.setattr(agno, "_install_class_hooks", lambda: None)
    workflow = agno.wrap_agno(Workflow(name="synthetic", steps=[step], telemetry=False))

    async def run():
        events = [event async for event in workflow.arun(input="test", stream=True)]
        assert events
        assert events[-1].content == "synthetic output"

    asyncio.run(run())


@pytest.mark.parametrize("workflow", [False, True])
def test_stream_close_finalizes_once(monkeypatch, workflow):
    closed = []

    async def events():
        try:
            yield SimpleNamespace(content="first")
            yield SimpleNamespace(content="second")
        finally:
            closed.append(True)

    entity = SimpleNamespace(arun=lambda *args, **kwargs: events(), model=None)
    tracer = Mock()
    monkeypatch.setattr(agno, "get_tracer", lambda: tracer)
    monkeypatch.setattr(agno, "attach_as_current", lambda span: None)
    monkeypatch.setattr(agno, "detach", lambda token: None)
    (agno._patch_workflow if workflow else agno._patch_agent)(entity)

    async def run():
        stream = entity.arun("synthetic", stream=True)
        assert (await anext(stream)).content == "first"
        await stream.aclose()
        await stream.aclose()

    asyncio.run(run())
    assert closed == [True]
    tracer.start_span.return_value.end.assert_called_once()
