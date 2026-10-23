"""Regression tests: the OpenAI Agents workflow span must be ERROR when the run fails, and OK when it recovers."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

pytest.importorskip("agents", reason="needs the openai-agents extra")

import openai
from agents import Agent, Runner, function_tool, set_trace_processors
from agents.exceptions import MaxTurnsExceeded, ModelBehaviorError
from agents.items import ModelResponse
from agents.models.interface import Model
from agents.models.openai_responses import OpenAIResponsesModel
from agents.usage import Usage
from openai.types.responses import (
    Response,
    ResponseCompletedEvent,
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputText,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from neatlogs._wrap_utils import set_neatlogs_provider
from neatlogs.openai_agents import openai_agents_processor

pytestmark = pytest.mark.asyncio


@pytest.fixture
def exporter() -> InMemorySpanExporter:
    provider = TracerProvider()
    exp = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exp))
    set_neatlogs_provider(provider)
    set_trace_processors([openai_agents_processor()])
    yield exp
    set_trace_processors([])


def _message(text: str) -> ResponseOutputMessage:
    return ResponseOutputMessage(
        id="m1",
        type="message",
        role="assistant",
        status="completed",
        content=[ResponseOutputText(type="output_text", text=text, annotations=[])],
    )


def _tool_call(
    name: str, call_id: str = "call-1", arguments: str = "{}"
) -> ResponseFunctionToolCall:
    return ResponseFunctionToolCall(
        type="function_call", call_id=call_id, name=name, arguments=arguments
    )


class _Scripted(Model):
    """Returns the given turns in order; a turn that is an exception is raised."""

    def __init__(self, *turns: Any) -> None:
        self._turns = list(turns)

    async def get_response(self, *args: Any, **kwargs: Any) -> ModelResponse:
        turn = self._turns.pop(0)
        if isinstance(turn, BaseException):
            raise turn
        return ModelResponse(output=turn, usage=Usage(), response_id=None)

    async def stream_response(self, *args: Any, **kwargs: Any) -> Any:
        turn = self._turns.pop(0)
        if isinstance(turn, BaseException):
            raise turn
        if turn is _NO_FINAL_RESPONSE:
            return
        yield ResponseCompletedEvent(
            type="response.completed",
            sequence_number=0,
            response=Response(
                id="r1",
                object="response",
                created_at=0,
                model="scripted",
                status="completed",
                output=turn,
                parallel_tool_calls=True,
                tool_choice="auto",
                tools=[],
            ),
        )


# A streamed turn that closes without a response.completed event.
_NO_FINAL_RESPONSE = object()


class _LoopForever(_Scripted):
    async def get_response(self, *args: Any, **kwargs: Any) -> ModelResponse:
        return ModelResponse(
            output=[_tool_call("get_weather", arguments='{"city": "Lagos"}')],
            usage=Usage(),
            response_id=None,
        )

    async def stream_response(self, *args: Any, **kwargs: Any) -> Any:
        self._turns = [[_tool_call("get_weather", arguments='{"city": "Lagos"}')]]
        async for event in super().stream_response(*args, **kwargs):
            yield event


class _ServerError(Exception):
    pass


@function_tool
def get_weather(city: str) -> str:
    """Get weather."""
    return f"sunny in {city}"


@function_tool
def broken_tool(city: str) -> str:
    """Always fails."""
    raise RuntimeError("tool broke")


def _workflow(exporter: InMemorySpanExporter):
    spans = [s for s in exporter.get_finished_spans() if s.name == "openai_agents.trace"]
    assert len(spans) == 1
    return spans[0]


async def test_workflow_is_error_when_model_call_fails(exporter: InMemorySpanExporter) -> None:
    agent = Agent(
        name="shop",
        model=_Scripted(
            [_tool_call("get_weather", arguments='{"city": "Lagos"}')],
            _ServerError("500 from provider"),
        ),
        tools=[get_weather],
    )
    with pytest.raises(_ServerError):
        await Runner.run(agent, "hi")
    assert _workflow(exporter).status.status_code == StatusCode.ERROR


async def test_workflow_is_error_when_max_turns_exceeded(exporter: InMemorySpanExporter) -> None:
    agent = Agent(name="shop", model=_LoopForever(), tools=[get_weather])
    with pytest.raises(MaxTurnsExceeded):
        await Runner.run(agent, "hi", max_turns=2)
    assert _workflow(exporter).status.status_code == StatusCode.ERROR


async def test_workflow_is_ok_on_success(exporter: InMemorySpanExporter) -> None:
    agent = Agent(name="shop", model=_Scripted([_message("Done.")]))
    result = await Runner.run(agent, "hi")
    assert result.final_output == "Done."
    assert _workflow(exporter).status.status_code == StatusCode.OK


async def test_workflow_is_ok_when_tool_error_is_handed_back(
    exporter: InMemorySpanExporter,
) -> None:
    agent = Agent(
        name="shop",
        model=_Scripted(
            [_tool_call("broken_tool", arguments='{"city": "Lagos"}')],
            [_message("Sorry, try later.")],
        ),
        tools=[broken_tool],
    )
    result = await Runner.run(agent, "hi")
    assert result.final_output == "Sorry, try later."
    assert _workflow(exporter).status.status_code == StatusCode.OK


async def test_workflow_is_ok_when_agent_tool_fails_but_run_succeeds(
    exporter: InMemorySpanExporter,
) -> None:
    sub = Agent(name="sub", model=_LoopForever(), tools=[get_weather])
    agent = Agent(
        name="helper",
        model=_Scripted([_tool_call("sub_tool", arguments='{"input": "hi"}')], [_message("Done.")]),
        tools=[sub.as_tool(tool_name="sub_tool", tool_description="sub", max_turns=1)],
    )
    result = await Runner.run(agent, "hi")
    assert result.final_output == "Done."
    assert _workflow(exporter).status.status_code == StatusCode.OK


async def test_workflow_is_ok_when_multiple_handoffs_requested(
    exporter: InMemorySpanExporter,
) -> None:
    first = Agent(name="first", model=_Scripted([_message("Hello!")]))
    second = Agent(name="second", model=_Scripted([_message("Hi!")]))
    agent = Agent(
        name="triage",
        model=_Scripted(
            [
                _tool_call("transfer_to_first", call_id="call-1"),
                _tool_call("transfer_to_second", call_id="call-2"),
            ]
        ),
        handoffs=[first, second],
    )
    result = await Runner.run(agent, "hi")
    assert result.final_output == "Hello!"
    assert _workflow(exporter).status.status_code == StatusCode.OK


def _give_up(_: Any) -> str:
    return "Sorry, I could not finish that."


async def _stream(agent: Agent, **kwargs: Any) -> Any:
    result = Runner.run_streamed(agent, "hi", **kwargs)
    async for _ in result.stream_events():
        pass
    return result.final_output


async def test_workflow_is_ok_when_max_turns_is_handled(exporter: InMemorySpanExporter) -> None:
    agent = Agent(name="shop", model=_LoopForever(), tools=[get_weather])
    result = await Runner.run(agent, "hi", max_turns=2, error_handlers={"max_turns": _give_up})
    assert result.final_output == "Sorry, I could not finish that."
    assert _workflow(exporter).status.status_code == StatusCode.OK


async def test_workflow_is_ok_on_success_inside_callers_except(
    exporter: InMemorySpanExporter,
) -> None:
    agent = Agent(name="shop", model=_Scripted([_message("Done.")]))
    try:
        raise ValueError("the caller's own error")
    except ValueError:
        result = await Runner.run(agent, "hi")
    assert result.final_output == "Done."
    assert _workflow(exporter).status.status_code == StatusCode.OK


async def test_streamed_workflow_is_ok_on_success(exporter: InMemorySpanExporter) -> None:
    agent = Agent(name="shop", model=_Scripted([_message("Done.")]))
    assert await _stream(agent) == "Done."
    assert _workflow(exporter).status.status_code == StatusCode.OK


async def test_streamed_workflow_is_error_when_model_call_fails(
    exporter: InMemorySpanExporter,
) -> None:
    agent = Agent(name="shop", model=_Scripted(_ServerError("500 from provider")))
    with pytest.raises(_ServerError):
        await _stream(agent)
    assert _workflow(exporter).status.status_code == StatusCode.ERROR


async def test_streamed_workflow_is_error_when_stream_ends_without_response(
    exporter: InMemorySpanExporter,
) -> None:
    agent = Agent(name="shop", model=_Scripted(_NO_FINAL_RESPONSE))
    with pytest.raises(ModelBehaviorError):
        await _stream(agent)
    assert _workflow(exporter).status.status_code == StatusCode.ERROR


async def test_streamed_workflow_is_error_when_max_turns_exceeded(
    exporter: InMemorySpanExporter,
) -> None:
    agent = Agent(name="shop", model=_LoopForever(), tools=[get_weather])
    with pytest.raises(MaxTurnsExceeded):
        await _stream(agent, max_turns=2)
    assert _workflow(exporter).status.status_code == StatusCode.ERROR


async def test_streamed_workflow_is_ok_when_max_turns_is_handled(
    exporter: InMemorySpanExporter,
) -> None:
    agent = Agent(name="shop", model=_LoopForever(), tools=[get_weather])
    output = await _stream(agent, max_turns=2, error_handlers={"max_turns": _give_up})
    assert output == "Sorry, I could not finish that."
    assert _workflow(exporter).status.status_code == StatusCode.OK


# init(instrumentations=["openai_agents"]) records through OpenInference's processor, which
# neatlogs patches so its root span ("Agent workflow") follows the same rule.


@pytest.fixture
def oi_exporter() -> InMemorySpanExporter:
    pytest.importorskip("openinference.instrumentation.openai_agents")
    from openinference.instrumentation.openai_agents._processor import (
        OpenInferenceTracingProcessor,
    )

    from neatlogs.instrumentation.manager import InstrumentationManager

    provider = TracerProvider()
    exp = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exp))
    InstrumentationManager(provider)._patch_openinference_openai_agents_run_status()
    set_trace_processors([OpenInferenceTracingProcessor(provider.get_tracer(__name__))])
    yield exp
    set_trace_processors([])


def _oi_root(exporter: InMemorySpanExporter):
    roots = [s for s in exporter.get_finished_spans() if s.parent is None]
    assert len(roots) == 1
    return roots[0]


async def test_oi_root_is_error_when_model_call_fails(oi_exporter: InMemorySpanExporter) -> None:
    agent = Agent(
        name="shop",
        model=_Scripted(
            [_tool_call("get_weather", arguments='{"city": "Lagos"}')],
            _ServerError("500 from provider"),
        ),
        tools=[get_weather],
    )
    with pytest.raises(_ServerError):
        await Runner.run(agent, "hi")
    assert _oi_root(oi_exporter).status.status_code == StatusCode.ERROR


async def test_oi_root_is_error_when_max_turns_exceeded(oi_exporter: InMemorySpanExporter) -> None:
    agent = Agent(name="shop", model=_LoopForever(), tools=[get_weather])
    with pytest.raises(MaxTurnsExceeded):
        await Runner.run(agent, "hi", max_turns=2)
    assert _oi_root(oi_exporter).status.status_code == StatusCode.ERROR


async def test_oi_streamed_root_is_error_when_stream_ends_without_response(
    oi_exporter: InMemorySpanExporter,
) -> None:
    agent = Agent(name="shop", model=_Scripted(_NO_FINAL_RESPONSE))
    with pytest.raises(ModelBehaviorError):
        await _stream(agent)
    assert _oi_root(oi_exporter).status.status_code == StatusCode.ERROR


async def test_oi_root_is_ok_on_success(oi_exporter: InMemorySpanExporter) -> None:
    agent = Agent(name="shop", model=_Scripted([_message("Done.")]))
    result = await Runner.run(agent, "hi")
    assert result.final_output == "Done."
    assert _oi_root(oi_exporter).status.status_code == StatusCode.OK


async def test_oi_root_is_ok_when_tool_error_is_handed_back(
    oi_exporter: InMemorySpanExporter,
) -> None:
    agent = Agent(
        name="shop",
        model=_Scripted(
            [_tool_call("broken_tool", arguments='{"city": "Lagos"}')],
            [_message("Sorry, try later.")],
        ),
        tools=[broken_tool],
    )
    result = await Runner.run(agent, "hi")
    assert result.final_output == "Sorry, try later."
    assert _oi_root(oi_exporter).status.status_code == StatusCode.OK


async def test_oi_root_is_ok_when_max_turns_is_handled(oi_exporter: InMemorySpanExporter) -> None:
    agent = Agent(name="shop", model=_LoopForever(), tools=[get_weather])
    result = await Runner.run(agent, "hi", max_turns=2, error_handlers={"max_turns": _give_up})
    assert result.final_output == "Sorry, I could not finish that."
    assert _oi_root(oi_exporter).status.status_code == StatusCode.OK


# The same rule end to end, through a real OpenAI client talking to a fake Responses API
# server, for both setups: init(instrumentations=["openai_agents"]) and
# openai_agents_processor(). The top span is ERROR exactly when the run raises.


class _FakeResponsesServer:
    """Serves scripted turns as Responses API JSON, or as an SSE stream when asked to stream.

    A turn is "tool:<name>", "text", "loop" (a tool call on every request), "http_500",
    "error_event", "cut" (the connection drops mid-stream) or "no_final" (no
    response.completed event).
    """

    def __init__(self, *turns: str) -> None:
        self._turns = list(turns)
        self._n = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        stream = json.loads(request.content).get("stream", False)
        self._n += 1
        turn = self._turns[0] if self._turns == ["loop"] else self._turns.pop(0)
        if turn == "http_500":
            return httpx.Response(500, json={"error": {"message": "boom", "type": "server_error"}})
        if turn.startswith("tool:") or turn == "loop":
            name = "get_weather" if turn == "loop" else turn[5:]
            item = {
                "type": "function_call",
                "id": f"fc_{self._n}",
                "call_id": f"call_{self._n}",
                "name": name,
                "arguments": '{"city": "Lagos"}',
                "status": "completed",
            }
        else:
            item = {
                "type": "message",
                "id": f"msg_{self._n}",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "Sunny.", "annotations": []}],
            }
        body = {
            "id": f"resp_{self._n}",
            "object": "response",
            "created_at": 0,
            "status": "completed",
            "model": "fake",
            "output": [item],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
        }
        if not stream:
            return httpx.Response(200, json=body)
        events = [("response.created", {"response": dict(body, status="in_progress", output=[])})]
        if turn == "error_event":
            events.append(
                ("error", {"code": "server_error", "message": "stream failed", "param": None})
            )
        elif turn != "cut":
            events.append(("response.output_item.done", {"output_index": 0, "item": item}))
            if turn != "no_final":
                events.append(("response.completed", {"response": body}))
        sse = b"".join(
            f"event: {t}\ndata: {json.dumps(dict(d, type=t, sequence_number=i))}\n\n".encode()
            for i, (t, d) in enumerate(events)
        )
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=_CutStream(sse) if turn == "cut" else httpx.ByteStream(sse),
        )


class _CutStream(httpx.AsyncByteStream):
    def __init__(self, data: bytes) -> None:
        self._data = data

    async def __aiter__(self) -> Any:
        yield self._data
        raise httpx.RemoteProtocolError("peer closed connection without sending complete message")


@pytest.fixture(params=["init", "processor"])
def setup_path(request: pytest.FixtureRequest) -> Any:
    import neatlogs

    provider = TracerProvider()
    exp = InMemorySpanExporter()
    if request.param == "init":
        neatlogs.init(
            api_key="test-key",
            disable_export=True,
            instrumentations=["openai_agents"],
            tracer_provider=provider,
            register_shutdown_handlers=False,
        )
    else:
        set_neatlogs_provider(provider)
        set_trace_processors([openai_agents_processor()])
    provider.add_span_processor(SimpleSpanProcessor(exp))
    yield exp
    if request.param == "init":
        neatlogs.shutdown()
    set_trace_processors([])


_SERVER_CASES = {
    # case: (turns, stream, run kwargs, expected exception or None)
    "success": (["text"], False, {}, None),
    "http_500": (["tool:get_weather", "http_500"], False, {}, openai.InternalServerError),
    "max_turns": (["loop"], False, {"max_turns": 2}, MaxTurnsExceeded),
    "max_turns_handled": (["loop"], False, {"max_turns": 2, "handled": True}, None),
    "unknown_tool": (["tool:no_such_tool"], False, {}, ModelBehaviorError),
    "tool_error_recovered": (["tool:broken_tool", "text"], False, {}, None),
    "stream_success": (["text"], True, {}, None),
    "stream_error_event": (["error_event"], True, {}, ModelBehaviorError),
    # openai 3.x wraps the dropped connection as APIConnectionError; 2.x raises it as is.
    "stream_cut": (["cut"], True, {}, (openai.APIConnectionError, httpx.TransportError)),
    "stream_no_final_response": (["no_final"], True, {}, ModelBehaviorError),
    "stream_max_turns": (["loop"], True, {"max_turns": 2}, MaxTurnsExceeded),
    "stream_max_turns_handled": (["loop"], True, {"max_turns": 2, "handled": True}, None),
    "stream_tool_error_recovered": (["tool:broken_tool", "text"], True, {}, None),
}


@pytest.mark.parametrize("case", list(_SERVER_CASES))
async def test_top_span_is_error_exactly_when_the_run_raises(
    setup_path: InMemorySpanExporter, case: str
) -> None:
    turns, stream, kwargs, expected = _SERVER_CASES[case]
    kwargs = dict(kwargs)
    if kwargs.pop("handled", False):
        kwargs["error_handlers"] = {"max_turns": _give_up}
    client = openai.AsyncOpenAI(
        api_key="test-key",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(_FakeResponsesServer(*turns))),
    )
    agent = Agent(
        name="shop",
        model=OpenAIResponsesModel(model="fake", openai_client=client),
        tools=[get_weather, broken_tool],
    )

    async def run() -> Any:
        if stream:
            return await _stream(agent, **kwargs)
        return (await Runner.run(agent, "hi", **kwargs)).final_output

    if expected is None:
        await run()
    else:
        with pytest.raises(expected):
            await run()

    roots = [s for s in setup_path.get_finished_spans() if s.parent is None]
    assert len(roots) == 1
    want = StatusCode.OK if expected is None else StatusCode.ERROR
    assert roots[0].status.status_code == want
