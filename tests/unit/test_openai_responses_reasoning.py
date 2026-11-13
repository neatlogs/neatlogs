from openai.types.responses import (
    Response,
    ResponseReasoningSummaryTextDeltaEvent,
    ResponseTextDeltaEvent,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import neatlogs.openai as oai

THINKING = "neatlogs.llm.output_messages.0.thinking"
CONTENT = "neatlogs.llm.output_messages.0.content"


def _run(finalize):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    finalize(provider.get_tracer("t").start_span("s"))
    return dict(exporter.get_finished_spans()[-1].attributes)


def _response(output):
    return Response.model_validate(
        {
            "id": "r",
            "object": "response",
            "created_at": 1.0,
            "model": "o4-mini",
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
            "output": output,
        }
    )


MESSAGE = {
    "type": "message",
    "id": "m1",
    "role": "assistant",
    "status": "completed",
    "content": [{"type": "output_text", "text": "4", "annotations": []}],
}


def _reasoning_delta(delta, summary_index=0, seq=1):
    return ResponseReasoningSummaryTextDeltaEvent.model_validate(
        {
            "type": "response.reasoning_summary_text.delta",
            "item_id": "rs_1",
            "output_index": 0,
            "summary_index": summary_index,
            "delta": delta,
            "sequence_number": seq,
        }
    )


def _text_delta(delta):
    return ResponseTextDeltaEvent.model_validate(
        {
            "type": "response.output_text.delta",
            "item_id": "m1",
            "output_index": 1,
            "content_index": 0,
            "delta": delta,
            "sequence_number": 9,
            "logprobs": [],
        }
    )


def test_response_keeps_reasoning_summary():
    reasoning = {
        "type": "reasoning",
        "id": "rs_1",
        "summary": [
            {"type": "summary_text", "text": "add the numbers"},
            {"type": "summary_text", "text": "check it"},
        ],
    }
    resp = _response([reasoning, MESSAGE])
    attrs = _run(lambda s: oai._finalize_responses_response(s, resp, 1.0))
    assert attrs[THINKING] == "add the numbers\n\ncheck it"
    assert attrs[CONTENT] == "4"


def test_response_without_reasoning_has_no_thinking():
    resp = _response([MESSAGE])
    attrs = _run(lambda s: oai._finalize_responses_response(s, resp, 1.0))
    assert THINKING not in attrs
    assert attrs[CONTENT] == "4"


def test_stream_keeps_reasoning_summary():
    events = [
        _reasoning_delta("add the ", seq=1),
        _reasoning_delta("numbers", seq=2),
        _reasoning_delta("check it", summary_index=1, seq=3),
        _text_delta("4"),
    ]
    attrs = _run(lambda s: oai._finalize_responses_stream(s, events, 1.0, 0.5))
    assert attrs[THINKING] == "add the numbers\n\ncheck it"
    assert attrs[CONTENT] == "4"


def test_stream_without_reasoning_has_no_thinking():
    attrs = _run(lambda s: oai._finalize_responses_stream(s, [_text_delta("4")], 1.0, 0.5))
    assert THINKING not in attrs
    assert attrs[CONTENT] == "4"
