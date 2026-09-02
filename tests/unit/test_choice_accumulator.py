import asyncio
from types import SimpleNamespace

import pytest
from opentelemetry.trace import StatusCode

from neatlogs._wrap_utils import AsyncStreamWrapper, SyncStreamWrapper
from neatlogs.core.choice_accumulator import (
    MAX_MEDIA_RECORDS_PER_CHOICE,
    ChoiceAccumulator,
    GoogleStreamFinalizer,
    OpenAIStreamFinalizer,
)
from neatlogs.openai import _finalize_responses_stream


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


def test_sync_iterator_break_does_not_close_provider_stream(
    tracer_provider, in_memory_span_exporter
):
    class ProviderStream:
        def __init__(self):
            self.values = iter(
                [
                    _chunk([{"index": 0, "delta": {"content": "first"}}]),
                    _chunk([{"index": 0, "delta": {"content": "second"}}]),
                ]
            )
            self.closed = False

        def __next__(self):
            if self.closed:
                raise RuntimeError("provider stream was closed")
            return next(self.values)

        def close(self):
            self.closed = True

    span = tracer_provider.get_tracer("neatlogs.test").start_span("llm")
    source = ProviderStream()
    wrapper = SyncStreamWrapper(source, span, OpenAIStreamFinalizer())
    iterator = iter(wrapper)

    next(iterator)
    iterator.close()  # what generator cleanup does when a for-loop is abandoned

    assert source.closed is False
    assert next(source)["choices"][0]["delta"]["content"] == "second"
    finished = in_memory_span_exporter.get_finished_spans()[0]
    assert finished.status.status_code is StatusCode.UNSET


def test_legacy_provider_finalizer_preserves_unset_on_early_close(
    tracer_provider, in_memory_span_exporter
):
    event = SimpleNamespace(type="response.output_text.delta", delta="partial", response=None)
    span = tracer_provider.get_tracer("neatlogs.openai").start_span("responses")
    wrapper = SyncStreamWrapper(iter([event]), span, _finalize_responses_stream)

    next(wrapper)
    wrapper.close()

    finished = in_memory_span_exporter.get_finished_spans()[0]
    assert finished.attributes["neatlogs.stream.cancelled"] is True
    assert finished.status.status_code is StatusCode.UNSET


def test_reasoning_content_uses_backend_thinking_selector_for_response_and_stream(
    tracer_provider, in_memory_span_exporter
):
    response_span = tracer_provider.get_tracer("neatlogs.test").start_span("response")
    response = ChoiceAccumulator()
    response.add_response(
        {"choices": [{"message": {"content": "answer", "reasoning_content": "private"}}]}
    )
    response.apply(response_span)
    response_span.end()

    stream_span = tracer_provider.get_tracer("neatlogs.test").start_span("stream")
    stream = ChoiceAccumulator()
    stream.add_chunk(
        stream_span,
        _chunk([{"index": 0, "delta": {"reasoning_content": "stream-private"}}]),
    )
    stream.apply(stream_span)
    stream_span.end()

    response_attrs, stream_attrs = [
        item.attributes for item in in_memory_span_exporter.get_finished_spans()
    ]
    assert response_attrs["neatlogs.llm.output_messages.0.thinking"] == "private"
    assert stream_attrs["neatlogs.llm.output_messages.0.thinking"] == "stream-private"
    assert "neatlogs.llm.output_messages.0.reasoning" not in response_attrs
    assert "neatlogs.llm.output_messages.0.reasoning" not in stream_attrs


def test_incremental_choice_capture_is_memory_bounded_and_explicit(
    tracer_provider, in_memory_span_exporter
):
    span = tracer_provider.get_tracer("neatlogs.test").start_span("large-stream")
    accumulator = ChoiceAccumulator()
    for _ in range(120):
        accumulator.add_chunk(
            span,
            _chunk([{"index": 0, "delta": {"content": "x" * 1000}}]),
        )
    accumulator.apply(span)
    span.end()

    attributes = in_memory_span_exporter.get_finished_spans()[0].attributes
    content = attributes["neatlogs.llm.output_messages.0.content"]
    assert len(content.encode()) <= 100_000
    assert "...[neatlogs-truncated" in content
    assert "original_bytes=120000" in content


def test_legacy_stream_keeps_terminal_usage_beyond_1024_small_events(
    tracer_provider, in_memory_span_exporter
):
    events = [
        SimpleNamespace(type="response.output_text.delta", delta="x", response=None)
        for _ in range(1024)
    ]
    usage = SimpleNamespace(input_tokens=11, output_tokens=22)
    events.append(
        SimpleNamespace(
            type="response.completed",
            delta=None,
            response=SimpleNamespace(model="gpt-test", usage=usage),
        )
    )

    span = tracer_provider.get_tracer("neatlogs.test").start_span("responses")
    wrapper = SyncStreamWrapper(iter(events), span, _finalize_responses_stream)
    list(wrapper)

    finished = in_memory_span_exporter.get_finished_spans()[0]
    assert finished.status.status_code is StatusCode.OK
    assert finished.attributes["neatlogs.llm.token_count.prompt"] == 11
    assert finished.attributes["neatlogs.llm.token_count.completion"] == 22
    assert "neatlogs.stream.chunks_dropped" not in finished.attributes


def test_legacy_stream_huge_chunk_is_not_retained_or_marked_successful(
    tracer_provider, in_memory_span_exporter
):
    seen = []

    def finalizer(span, chunks, _duration, _ttft, *, interrupted=False):
        seen.extend(chunks)
        span.set_status(StatusCode.UNSET if interrupted else StatusCode.OK)
        span.end()

    huge = SimpleNamespace(payload="secret" * 250_000)
    span = tracer_provider.get_tracer("neatlogs.test").start_span("legacy-stream")
    wrapper = SyncStreamWrapper(iter([huge]), span, finalizer)
    list(wrapper)

    finished = in_memory_span_exporter.get_finished_spans()[0]
    attributes = finished.attributes
    assert seen == []
    assert finished.status.status_code is StatusCode.UNSET
    assert attributes["neatlogs.stream.chunks_dropped"] == 1
    assert attributes["neatlogs.stream.capture_incomplete"] is True
    assert attributes["neatlogs.capture.truncated"] is True
    assert attributes["neatlogs.capture.overflow.reason"] == "backend_upload_contract_unavailable"


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


def test_google_candidates_share_the_same_multi_choice_and_tool_schema(
    tracer_provider, in_memory_span_exporter
):
    span = tracer_provider.get_tracer("neatlogs.google_genai").start_span("gemini")
    accumulator = ChoiceAccumulator()
    accumulator.add_google_response(
        {
            "model_version": "gemini-test",
            "response_id": "google-response-1",
            "candidates": [
                {
                    "index": 0,
                    "finish_reason": "STOP",
                    "content": {"role": "model", "parts": [{"text": "first"}]},
                },
                {
                    "index": 1,
                    "finish_reason": "TOOL_CALL",
                    "content": {
                        "role": "model",
                        "parts": [
                            {"text": "thinking", "thought": True},
                            {
                                "function_call": {
                                    "name": "lookup",
                                    "args": {"query": "weather"},
                                }
                            },
                        ],
                    },
                },
            ],
            "usage_metadata": {
                "prompt_token_count": 4,
                "candidates_token_count": 2,
                "total_token_count": 6,
                "thoughts_token_count": 1,
            },
        }
    )
    accumulator.apply(span)
    span.end()

    attributes = in_memory_span_exporter.get_finished_spans()[0].attributes
    assert attributes["neatlogs.llm.output_messages.0.content"] == "first"
    assert attributes["neatlogs.llm.output_messages.1.thinking"] == "thinking"
    assert attributes["neatlogs.llm.choices.0.finish_reason"] == "STOP"
    assert attributes["neatlogs.llm.choices.1.finish_reason"] == "TOOL_CALL"
    assert attributes["neatlogs.llm.tool_calls.0.choice_index"] == 1
    assert attributes["neatlogs.llm.tool_calls.0.tool_call_index"] == 0
    assert attributes["neatlogs.llm.tool_calls.0.name"] == "lookup"
    assert attributes["neatlogs.llm.token_count.total"] == 6
    assert attributes["neatlogs.llm.token_count.reasoning"] == 1


def test_single_choice_flattened_family_reports_limitation_without_inventing_choices(
    tracer_provider, in_memory_span_exporter
):
    span = tracer_provider.get_tracer("neatlogs.callback").start_span("callback")
    accumulator = ChoiceAccumulator(capture_fidelity="flattened")
    accumulator.add_single_response("only output", finish_reason="stop", model="callback-model")
    accumulator.apply(span)
    span.end()

    attributes = in_memory_span_exporter.get_finished_spans()[0].attributes
    assert attributes["neatlogs.llm.output_messages.0.content"] == "only output"
    assert "neatlogs.llm.output_messages.1.content" not in attributes
    assert attributes["neatlogs.capture_fidelity"] == "flattened"


def test_google_stream_finalizer_is_incremental_and_preserves_partial_candidate(
    tracer_provider, in_memory_span_exporter
):
    span = tracer_provider.get_tracer("neatlogs.google_genai").start_span("gemini-stream")
    chunks = iter(
        [
            {
                "candidates": [
                    {"index": 1, "content": {"role": "model", "parts": [{"text": "partial"}]}}
                ]
            }
        ]
    )
    wrapper = SyncStreamWrapper(chunks, span, GoogleStreamFinalizer())
    next(wrapper)
    assert not wrapper._chunks
    wrapper.close()

    finished = in_memory_span_exporter.get_finished_spans()[0]
    assert finished.attributes["neatlogs.llm.output_messages.1.content"] == "partial"
    assert finished.attributes["neatlogs.stream.cancelled"] is True
    assert finished.status.status_code.name == "UNSET"


def test_streamed_media_accumulation_is_deduplicated_and_bounded(
    tracer_provider, in_memory_span_exporter
):
    span = tracer_provider.get_tracer("neatlogs.test").start_span("media-stream")
    accumulator = ChoiceAccumulator()
    repeated = {"type": "image_url", "image_url": {"url": "https://media.test/same.png"}}
    for _ in range(50):
        accumulator.add_chunk(span, _chunk([{"index": 0, "delta": {"content": [repeated]}}]))
    for index in range(MAX_MEDIA_RECORDS_PER_CHOICE + 5):
        media = {
            "type": "image_url",
            "image_url": {"url": f"https://media.test/{index}.png"},
        }
        accumulator.add_chunk(span, _chunk([{"index": 0, "delta": {"content": [media]}}]))
    assert len(accumulator.choices[0].media_records) == MAX_MEDIA_RECORDS_PER_CHOICE
    assert accumulator.choices[0].media_records_dropped == 6
    accumulator.apply(span)
    span.end()

    attributes = in_memory_span_exporter.get_finished_spans()[0].attributes
    assert attributes["neatlogs.llm.output_messages.0.media_dropped_count"] == 6
