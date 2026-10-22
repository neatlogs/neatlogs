"""Regression tests: the OpenAI Agents workflow span must be ERROR when the run fails, and OK when it recovers."""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("agents", reason="needs the openai-agents extra")

from agents import Agent, Runner, function_tool, set_trace_processors
from agents.exceptions import MaxTurnsExceeded
from agents.items import ModelResponse
from agents.models.interface import Model
from agents.usage import Usage
from openai.types.responses import (
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

    def stream_response(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError


class _LoopForever(_Scripted):
    async def get_response(self, *args: Any, **kwargs: Any) -> ModelResponse:
        return ModelResponse(
            output=[_tool_call("get_weather", arguments='{"city": "Lagos"}')],
            usage=Usage(),
            response_id=None,
        )


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
