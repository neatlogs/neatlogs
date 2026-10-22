"""A call that failed, or a stream read through a provider helper, must not be recorded as an empty success.

No network: the providers are replaced by small fakes that behave like the real clients.
"""

from __future__ import annotations

from types import SimpleNamespace as NS
from typing import Any

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from neatlogs._wrap_utils import set_neatlogs_provider
from neatlogs.anthropic import wrap_anthropic_client, wrap_async_anthropic_client
from neatlogs.core.choice_accumulator import ChoiceAccumulator
from neatlogs.openai import wrap_async_openai_client, wrap_openai_client


@pytest.fixture
def exporter() -> InMemorySpanExporter:
    provider = TracerProvider()
    exp = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exp))
    set_neatlogs_provider(provider)
    return exp


def _only_span(exporter: InMemorySpanExporter):
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    return spans[0]


def _exception_messages(span) -> list:
    return [e.attributes.get("exception.message") for e in span.events if e.name == "exception"]


# ---------------------------------------------------------------------------
# Anthropic messages.stream()
# ---------------------------------------------------------------------------


def _message(text: str = "", stop_reason: Any = "end_turn", tool: bool = False) -> NS:
    content = [NS(type="text", text=text)] if text else []
    if tool:
        content.append(
            NS(type="tool_use", id="toolu_1", name="get_weather", input={"city": "Lagos"})
        )
    return NS(
        type="message",
        content=content,
        stop_reason=stop_reason,
        model="claude-test",
        usage=NS(
            input_tokens=5,
            output_tokens=3,
            cache_read_input_tokens=None,
            cache_creation_input_tokens=None,
        ),
    )


class _HelperStream:
    """Like anthropic's MessageStream: the caller reads ``text_stream``, never the events."""

    def __init__(self, pieces: list, final: NS, fail_with: BaseException | None = None):
        self._pieces = pieces
        self._final = final
        self._fail_with = fail_with
        self._seen: list = []

    @property
    def text_stream(self):
        for piece in self._pieces:
            self._seen.append(piece)
            yield piece
        if self._fail_with is not None:
            raise self._fail_with

    @property
    def current_message_snapshot(self) -> NS:
        if not self._seen:
            raise AssertionError("no message_start yet")
        if len(self._seen) == len(self._pieces) and self._fail_with is None:
            return self._final
        return _message("".join(self._seen), stop_reason=None)

    def get_final_message(self) -> NS:
        for _ in self.text_stream:
            pass
        return self._final


class _Manager:
    def __init__(self, stream: Any = None, enter_error: BaseException | None = None):
        self._stream = stream
        self._enter_error = enter_error

    def __enter__(self):
        if self._enter_error is not None:
            raise self._enter_error
        return self._stream

    def __exit__(self, *args):
        return None

    async def __aenter__(self):
        return self.__enter__()

    async def __aexit__(self, *args):
        return None


def _anthropic_client(manager: _Manager, is_async: bool = False) -> NS:
    messages = NS(create=lambda **kwargs: None, stream=lambda **kwargs: manager)
    client = NS(messages=messages)
    return (wrap_async_anthropic_client if is_async else wrap_anthropic_client)(client)


def _stream_kwargs() -> dict:
    return {
        "model": "claude-test",
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "hi"}],
    }


def test_anthropic_helper_records_the_reply_read_through_text_stream(exporter):
    client = _anthropic_client(_Manager(_HelperStream(["hel", "lo"], _message("hello"))))
    with client.messages.stream(**_stream_kwargs()) as stream:
        assert "".join(stream.text_stream) == "hello"

    span = _only_span(exporter)
    assert span.status.status_code.name == "OK"
    assert span.attributes["neatlogs.llm.output_messages.0.content"] == "hello"
    assert span.attributes["neatlogs.llm.finish_reason"] == "end_turn"
    assert span.attributes["neatlogs.llm.token_count.prompt"] == 5
    assert span.attributes["neatlogs.llm.token_count.completion"] == 3


def test_anthropic_helper_records_tool_calls(exporter):
    final = _message(stop_reason="tool_use", tool=True)
    client = _anthropic_client(_Manager(_HelperStream(["x"], final)))
    with client.messages.stream(**_stream_kwargs()) as stream:
        stream.get_final_message()

    span = _only_span(exporter)
    assert span.attributes["neatlogs.llm.tool_calls.0.name"] == "get_weather"
    assert span.attributes["neatlogs.llm.tool_calls.0.arguments"] == '{"city": "Lagos"}'
    assert span.attributes["neatlogs.llm.finish_reason"] == "tool_use"


def test_anthropic_helper_stream_that_breaks_is_an_error_and_keeps_the_partial_reply(exporter):
    cut = ConnectionError("peer closed connection without sending complete message body")
    client = _anthropic_client(_Manager(_HelperStream(["hel"], _message("hello"), fail_with=cut)))
    with pytest.raises(ConnectionError):
        with client.messages.stream(**_stream_kwargs()) as stream:
            for _ in stream.text_stream:
                pass

    span = _only_span(exporter)
    assert span.status.status_code.name == "ERROR"
    assert "peer closed connection" in span.status.description
    assert _exception_messages(span) == [str(cut)]
    assert span.attributes["neatlogs.llm.output_messages.0.content"] == "hel"


def test_anthropic_helper_request_that_fails_still_ends_its_span(exporter):
    overloaded = RuntimeError("Error code: 529 - overloaded")
    client = _anthropic_client(_Manager(enter_error=overloaded))
    with pytest.raises(RuntimeError):
        with client.messages.stream(**_stream_kwargs()):
            pass

    span = _only_span(exporter)
    assert span.status.status_code.name == "ERROR"
    assert _exception_messages(span) == [str(overloaded)]


def test_anthropic_helper_left_early_is_cancelled_not_an_error(exporter):
    client = _anthropic_client(_Manager(_HelperStream(["hel", "lo"], _message("hello"))))
    with client.messages.stream(**_stream_kwargs()) as stream:
        for _ in stream.text_stream:
            break

    span = _only_span(exporter)
    assert span.status.status_code.name == "UNSET"
    assert span.attributes["neatlogs.stream.cancelled"] is True
    assert span.attributes["neatlogs.llm.output_messages.0.content"] == "hel"


def test_anthropic_helper_without_a_snapshot_still_uses_the_events(exporter):
    events = [
        NS(type="message_start", message=NS(model="claude-test", usage=NS(input_tokens=5))),
        NS(type="content_block_delta", index=0, delta=NS(type="text_delta", text="hello")),
        NS(type="message_delta", delta=NS(stop_reason="end_turn"), usage=NS(output_tokens=3)),
    ]
    client = _anthropic_client(_Manager(iter(events)))
    with client.messages.stream(**_stream_kwargs()) as stream:
        assert len(list(stream)) == 3

    span = _only_span(exporter)
    assert span.status.status_code.name == "OK"
    assert span.attributes["neatlogs.llm.output_messages.0.content"] == "hello"
    assert span.attributes["neatlogs.llm.finish_reason"] == "end_turn"


@pytest.mark.asyncio
async def test_anthropic_async_helper_stream_that_breaks_is_an_error(exporter):
    cut = ConnectionError("connection reset")
    client = _anthropic_client(
        _Manager(_HelperStream(["hel"], _message("hello"), fail_with=cut)), is_async=True
    )
    with pytest.raises(ConnectionError):
        async with client.messages.stream(**_stream_kwargs()) as stream:
            for _ in stream.text_stream:
                pass

    span = _only_span(exporter)
    assert span.status.status_code.name == "ERROR"
    assert span.attributes["neatlogs.llm.output_messages.0.content"] == "hel"


# ---------------------------------------------------------------------------
# OpenAI Responses API
# ---------------------------------------------------------------------------


def _response(status: str = "completed", **fields: Any) -> NS:
    base = {
        "status": status,
        "model": "gpt-test",
        "output": [],
        "output_text": "",
        "error": None,
        "incomplete_details": None,
        "usage": NS(input_tokens=5, output_tokens=3),
    }
    return NS(**{**base, **fields})


def _function_call() -> NS:
    return NS(
        type="function_call", call_id="call_1", name="get_weather", arguments='{"city":"Lagos"}'
    )


def _openai_client(result: Any, is_async: bool = False) -> NS:
    if is_async:

        async def create(**kwargs):
            return result

    else:

        def create(**kwargs):
            return result

    client = NS(responses=NS(create=create))
    return (wrap_async_openai_client if is_async else wrap_openai_client)(client)


def test_responses_create_records_finish_and_tool_calls(exporter):
    client = _openai_client(_response(output=[_function_call()]))
    client.responses.create(model="gpt-test", input="weather?")

    span = _only_span(exporter)
    assert span.status.status_code.name == "OK"
    assert span.attributes["neatlogs.llm.finish_reason"] == "completed"
    assert span.attributes["neatlogs.llm.tool_calls.0.id"] == "call_1"
    assert span.attributes["neatlogs.llm.tool_calls.0.name"] == "get_weather"
    assert span.attributes["neatlogs.llm.tool_calls.0.arguments"] == '{"city":"Lagos"}'


def test_responses_create_that_failed_in_the_body_is_an_error(exporter):
    failed = _response("failed", error=NS(code="server_error", message="The model failed"))
    client = _openai_client(failed)
    client.responses.create(model="gpt-test", input="hi")

    span = _only_span(exporter)
    assert span.status.status_code.name == "ERROR"
    assert span.status.description == "The model failed"
    assert _exception_messages(span) == ["The model failed"]


def test_responses_create_cut_short_keeps_the_reason_and_is_not_an_error(exporter):
    cut_short = _response("incomplete", incomplete_details=NS(reason="max_output_tokens"))
    client = _openai_client(cut_short)
    client.responses.create(model="gpt-test", input="hi")

    span = _only_span(exporter)
    assert span.status.status_code.name == "OK"
    assert span.attributes["neatlogs.llm.finish_reason"] == "max_output_tokens"


def _delta(text: str) -> NS:
    return NS(type="response.output_text.delta", delta=text)


def test_responses_stream_records_finish_and_tool_calls(exporter):
    events = [
        _delta("hel"),
        _delta("lo"),
        NS(type="response.completed", response=_response(output=[_function_call()])),
    ]
    client = _openai_client(iter(events))
    assert len(list(client.responses.create(model="gpt-test", input="hi", stream=True))) == 3

    span = _only_span(exporter)
    assert span.status.status_code.name == "OK"
    assert span.attributes["neatlogs.llm.output_messages.0.content"] == "hello"
    assert span.attributes["neatlogs.llm.finish_reason"] == "completed"
    assert span.attributes["neatlogs.llm.tool_calls.0.name"] == "get_weather"


def test_responses_stream_error_event_is_an_error_even_when_the_app_stops_reading(exporter):
    events = [
        _delta("hel"),
        NS(type="error", code="server_error", message="The server had an error"),
    ]
    client = _openai_client(iter(events))

    # What an app does: it sees the error event and raises, abandoning the stream.
    with pytest.raises(RuntimeError):
        for event in client.responses.create(model="gpt-test", input="hi", stream=True):
            if event.type == "error":
                raise RuntimeError(event.message)

    span = _only_span(exporter)
    assert span.status.status_code.name == "ERROR"
    assert span.status.description == "The server had an error"
    assert _exception_messages(span) == ["The server had an error"]
    assert span.attributes["neatlogs.llm.output_messages.0.content"] == "hel"


def test_responses_stream_failed_event_is_an_error(exporter):
    failed = _response("failed", error=NS(code="server_error", message="The model failed"))
    events = [_delta("hel"), NS(type="response.failed", response=failed)]
    client = _openai_client(iter(events))
    list(client.responses.create(model="gpt-test", input="hi", stream=True))

    span = _only_span(exporter)
    assert span.status.status_code.name == "ERROR"
    assert _exception_messages(span) == ["The model failed"]


def test_responses_stream_left_early_without_an_error_stays_unset(exporter):
    client = _openai_client(iter([_delta("hel"), _delta("lo")]))
    for _ in client.responses.create(model="gpt-test", input="hi", stream=True):
        break

    span = _only_span(exporter)
    assert span.status.status_code.name == "UNSET"


@pytest.mark.asyncio
async def test_async_responses_stream_error_event_is_an_error(exporter):
    async def events():
        yield _delta("hel")
        yield NS(type="error", code="server_error", message="The server had an error")

    client = _openai_client(events(), is_async=True)
    stream = await client.responses.create(model="gpt-test", input="hi", stream=True)
    async for _ in stream:
        pass

    span = _only_span(exporter)
    assert span.status.status_code.name == "ERROR"


# ---------------------------------------------------------------------------
# OpenAI chat completions: a refusal
# ---------------------------------------------------------------------------


def test_chat_refusal_is_recorded_as_the_reply(exporter):
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    span = provider.get_tracer("test").start_span("chat")
    accumulator = ChoiceAccumulator()
    accumulator.add_response(
        {
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "refusal": "I can't help with that.",
                    },
                }
            ]
        }
    )
    accumulator.apply(span)
    span.end()

    recorded = _only_span(exporter)
    assert (
        recorded.attributes["neatlogs.llm.output_messages.0.content"] == "I can't help with that."
    )
