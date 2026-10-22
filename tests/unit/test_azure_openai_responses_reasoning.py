from openai.types.responses import (
    Response,
    ResponseReasoningSummaryTextDeltaEvent,
    ResponseTextDeltaEvent,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import neatlogs.azure_openai as az

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


def _reasoning(*texts):
    return {
        "type": "reasoning",
        "id": "rs_1",
        "summary": [{"type": "summary_text", "text": t} for t in texts],
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


def _text_delta(delta, seq=10):
    return ResponseTextDeltaEvent.model_validate(
        {
            "type": "response.output_text.delta",
            "item_id": "m1",
            "output_index": 1,
            "content_index": 0,
            "delta": delta,
            "sequence_number": seq,
            "logprobs": [],
        }
    )


def _stream(chunks):
    return lambda span: az._finalize_responses_stream(span, chunks, 5.0, 1.0)


def test_response_keeps_reasoning_summary():
    resp = _response([_reasoning("add the numbers"), MESSAGE])
    attrs = _run(lambda s: az._finalize_responses_response(s, resp, 5.0))
    assert attrs[THINKING] == "add the numbers"
    assert attrs[CONTENT] == "4"


def test_response_joins_summary_parts():
    resp = _response([_reasoning("first", "second"), MESSAGE])
    attrs = _run(lambda s: az._finalize_responses_response(s, resp, 5.0))
    assert attrs[THINKING] == "first\n\nsecond"


def test_response_without_reasoning_has_no_thinking():
    attrs = _run(lambda s: az._finalize_responses_response(s, _response([MESSAGE]), 5.0))
    assert THINKING not in attrs
    assert attrs[CONTENT] == "4"


def test_stream_keeps_reasoning_summary():
    chunks = [_reasoning_delta("add the "), _reasoning_delta("numbers", seq=2), _text_delta("4")]
    attrs = _run(_stream(chunks))
    assert attrs[THINKING] == "add the numbers"
    assert attrs[CONTENT] == "4"


def test_stream_without_reasoning_has_no_thinking():
    attrs = _run(_stream([_text_delta("4")]))
    assert THINKING not in attrs
    assert attrs[CONTENT] == "4"
