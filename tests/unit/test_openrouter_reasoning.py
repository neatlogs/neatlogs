import pytest

pytest.importorskip("openrouter")

from openrouter import components as c
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import neatlogs.openrouter as orr

THINKING = "neatlogs.llm.output_messages.0.thinking"
CONTENT = "neatlogs.llm.output_messages.0.content"


def _tracer():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider.get_tracer("t"), exporter


def _result(message):
    return c.ChatResult.model_validate(
        {
            "id": "x",
            "object": "chat.completion",
            "created": 1,
            "system_fingerprint": "fp",
            "model": "anthropic/claude-3.7-sonnet",
            "choices": [{"index": 0, "finish_reason": "stop", "message": message}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 9, "total_tokens": 14},
        }
    )


def _chunk(delta):
    return c.ChatStreamChunk.model_validate(
        {
            "id": "x",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "m",
            "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
        }
    )


def _attrs(tracer, exporter, run):
    span = tracer.start_span("s")
    run(span)
    return dict(exporter.get_finished_spans()[-1].attributes)


def test_chat_keeps_reasoning():
    tracer, exporter = _tracer()
    res = _result({"role": "assistant", "content": "4", "reasoning": "add them"})
    attrs = _attrs(tracer, exporter, lambda s: orr._finalize_chat(s, res, 1.0))
    assert attrs[THINKING] == "add them"
    assert attrs[CONTENT] == "4"


def test_chat_without_reasoning_has_no_thinking():
    tracer, exporter = _tracer()
    res = _result({"role": "assistant", "content": "4"})
    attrs = _attrs(tracer, exporter, lambda s: orr._finalize_chat(s, res, 1.0))
    assert THINKING not in attrs


def test_stream_keeps_reasoning():
    tracer, exporter = _tracer()
    chunks = [
        _chunk({"role": "assistant", "reasoning": "add "}),
        _chunk({"reasoning": "them"}),
        _chunk({"content": "4"}),
    ]
    attrs = _attrs(tracer, exporter, lambda s: orr._finalize_chat_stream(s, chunks, 1.0, 0.5))
    assert attrs[THINKING] == "add them"
    assert attrs[CONTENT] == "4"


def test_stream_without_reasoning_has_no_thinking():
    tracer, exporter = _tracer()
    chunks = [_chunk({"role": "assistant", "content": "4"})]
    attrs = _attrs(tracer, exporter, lambda s: orr._finalize_chat_stream(s, chunks, 1.0, 0.5))
    assert THINKING not in attrs
