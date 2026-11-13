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


@pytest.mark.parametrize("stream", [None, False, True])
@pytest.mark.parametrize("websocket", [False, True])
def test_workflow_background_preserves_native_transport(monkeypatch, stream, websocket):
    pytest.importorskip("agno")
    from agno.workflow import Workflow

    response = SimpleNamespace(content="polling response")
    event = SimpleNamespace(content="background event")
    calls = []

    async def polling(*args, **kwargs):
        calls.append("polling")
        return response

    async def sse(*args, **kwargs):
        calls.append("sse")
        yield event

    async def ws(*args, **kwargs):
        calls.append("websocket")
        return response

    monkeypatch.setattr(Workflow, "_arun_background", polling)
    monkeypatch.setattr(Workflow, "_arun_background_stream", sse)
    monkeypatch.setattr(Workflow, "_arun_background_stream_ws", ws)
    kwargs = {"input": "synthetic", "background": True}
    if stream is not None:
        kwargs["stream"] = stream
    if websocket:
        WebSocket = pytest.importorskip("starlette.websockets").WebSocket
        kwargs["websocket"] = WebSocket({"type": "websocket"}, Mock(), Mock())
    expected = "websocket" if stream and websocket else "sse" if stream else "polling"

    async def check(entity):
        output = entity.arun(**kwargs)
        if expected == "sse":
            assert [item async for item in output] == [event]
        else:
            assert await output is response

    # First prove the real native dispatcher shape, then require wrap parity.
    asyncio.run(check(Workflow(name="native", stream=True, telemetry=False)))
    tracer = Mock()
    monkeypatch.setattr(agno, "get_tracer", lambda: tracer)
    monkeypatch.setattr(agno, "attach_as_current", lambda span: None)
    monkeypatch.setattr(agno, "detach", lambda token: None)
    wrapped = Workflow(name="wrapped", stream=True, telemetry=False)
    agno._patch_workflow(wrapped)
    asyncio.run(check(wrapped))
    assert calls == [expected, expected]
    tracer.start_span.return_value.end.assert_called_once()


def test_real_configured_workflow_background_polling(monkeypatch):
    pytest.importorskip("agno")
    from agno.workflow import Workflow
    from agno.workflow.types import StepInput, StepOutput

    async def step(step_input: StepInput):
        return StepOutput(content="background synthetic output")

    monkeypatch.setattr(agno, "_install_class_hooks", lambda: None)
    workflow = agno.wrap_agno(
        Workflow(name="background synthetic", stream=True, steps=[step], telemetry=False)
    )

    async def run():
        response = await workflow.arun(input="test", background=True)
        assert response.run_id
        # Let the actual synthetic background task finish before loop shutdown.
        for _ in range(100):
            if response.content == "background synthetic output":
                break
            await asyncio.sleep(0.01)
        assert response.content == "background synthetic output"

    asyncio.run(run())
