import asyncio

import pytest

from neatlogs._wrap_utils import AsyncStreamWrapper, SyncStreamWrapper
from neatlogs.core.choice_accumulator import ChoiceAccumulator, OpenAIStreamFinalizer


def _chunk(choices, *, usage=None, model=None, response_id=None):
    return {
        "choices": choices,
        "usage": usage,
        "model": model,
        "id": response_id,
    }


def test_interleaved_choices_and_tool_fragments_are_preserved(
    tracer_provider, in_memory_span_exporter
):
    span = tracer_provider.get_tracer("neatlogs.test").start_span("llm")
    accumulator = ChoiceAccumulator()
    accumulator.add_chunk(
        span,
        _chunk(
            [
                {
                    "index": 1,
                    "delta": {
                        "content": "B",
                        "tool_calls": [
                            {
                                "index": 0,
                                "function": {"name": "lookup", "arguments": '{"q":'},
                            }
                        ],
                    },
                },
                {"index": 0, "delta": {"content": "A"}},
            ],
            model="gpt-test",
            response_id="response-1",
        ),
    )
    accumulator.add_chunk(
        span,
        _chunk(
            [
                {
                    "index": 1,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "function": {"arguments": '"weather"}'},
                            }
                        ]
                    },
                    "finish_reason": "tool_calls",
                },
                {"index": 0, "delta": {"content": "0"}, "finish_reason": "stop"},
            ],
            usage={"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
        ),
    )
    accumulator.apply(span)
    span.end()

    attributes = in_memory_span_exporter.get_finished_spans()[0].attributes
    assert attributes["neatlogs.llm.output_messages.0.content"] == "A0"
    assert attributes["neatlogs.llm.output_messages.1.content"] == "B"
    assert attributes["neatlogs.llm.choices.0.finish_reason"] == "stop"
    assert attributes["neatlogs.llm.choices.1.finish_reason"] == "tool_calls"
    assert attributes["neatlogs.llm.tool_calls.0.name"] == "lookup"
    assert attributes["neatlogs.llm.tool_calls.0.arguments"] == '{"q":"weather"}'
    assert attributes["neatlogs.llm.tool_calls.0.choice_index"] == 1
    assert attributes["neatlogs.llm.tool_calls.0.tool_call_index"] == 0
    assert attributes["neatlogs.llm.tool_calls.0.id"].startswith("nl_")
    assert attributes["neatlogs.llm.tool_calls.0.id_synthetic"] is True
    assert attributes["neatlogs.llm.token_count.total"] == 6


def test_sync_early_close_exports_partial_content_as_interrupted(
    tracer_provider, in_memory_span_exporter
):
    span = tracer_provider.get_tracer("neatlogs.test").start_span("llm")
    stream = iter([_chunk([{"index": 0, "delta": {"content": "partial"}}])])
    wrapper = SyncStreamWrapper(stream, span, OpenAIStreamFinalizer())

    next(wrapper)
    assert not wrapper._chunks
    wrapper.close()

    finished = in_memory_span_exporter.get_finished_spans()[0]
    assert finished.attributes["neatlogs.llm.output_messages.0.content"] == "partial"
    assert finished.attributes["neatlogs.stream.cancelled"] is True
    assert finished.status.status_code.name == "UNSET"


def test_sync_for_loop_break_finalizes_without_explicit_close(
    tracer_provider, in_memory_span_exporter
):
    span = tracer_provider.get_tracer("neatlogs.test").start_span("llm")
    stream = iter(
        [
            _chunk([{"index": 0, "delta": {"content": "first"}}]),
            _chunk([{"index": 0, "delta": {"content": "second"}}]),
        ]
    )
    wrapper = SyncStreamWrapper(stream, span, OpenAIStreamFinalizer())

    for chunk in wrapper:
        assert chunk["choices"][0]["delta"]["content"] == "first"
        break

    finished = in_memory_span_exporter.get_finished_spans()[0]
    assert finished.attributes["neatlogs.llm.output_messages.0.content"] == "first"
    assert finished.attributes["neatlogs.stream.cancelled"] is True


@pytest.mark.asyncio
async def test_async_cancellation_exports_partial_content_as_interrupted(
    tracer_provider, in_memory_span_exporter
):
    async def stream():
        yield _chunk([{"index": 0, "delta": {"content": "partial"}}])
        await asyncio.Future()

    span = tracer_provider.get_tracer("neatlogs.test").start_span("llm")
    wrapper = AsyncStreamWrapper(stream(), span, OpenAIStreamFinalizer())
    await anext(wrapper)
    assert not wrapper._chunks
    await wrapper.aclose()

    finished = in_memory_span_exporter.get_finished_spans()[0]
    assert finished.attributes["neatlogs.llm.output_messages.0.content"] == "partial"
    assert finished.attributes["neatlogs.stream.cancelled"] is True
    assert finished.status.status_code.name == "UNSET"


def test_flattened_callback_can_report_capture_fidelity_explicitly(
    tracer_provider, in_memory_span_exporter
):
    span = tracer_provider.get_tracer("neatlogs.test").start_span("flattened")
    accumulator = ChoiceAccumulator(capture_fidelity="flattened")
    accumulator.add_response({"choices": [{"message": {"content": "flat"}}]})
    accumulator.apply(span)
    span.end()

    attributes = in_memory_span_exporter.get_finished_spans()[0].attributes
    assert attributes["neatlogs.capture_fidelity"] == "flattened"
