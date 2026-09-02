import pytest

from neatlogs.azure_openai import _finalize_response as finalize_azure
from neatlogs.google_genai import _finalize_response as finalize_google
from neatlogs.openrouter import _finalize_chat as finalize_openrouter


@pytest.mark.parametrize("finalize", [finalize_azure, finalize_openrouter])
def test_openai_compatible_adapters_preserve_all_choices_and_tool_coordinates(
    finalize, tracer_provider, in_memory_span_exporter
):
    span = tracer_provider.get_tracer("neatlogs.provider").start_span("chat")
    finalize(
        span,
        {
            "id": "response-1",
            "model": "model-1",
            "choices": [
                {
                    "index": 1,
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "index": 2,
                                "id": "call-1",
                                "type": "function",
                                "function": {"name": "lookup", "arguments": "{}"},
                            }
                        ],
                    },
                },
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "answer"},
                },
            ],
            "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
        },
        10.0,
    )

    attributes = in_memory_span_exporter.get_finished_spans()[0].attributes
    assert attributes["neatlogs.llm.output_messages.0.content"] == "answer"
    assert attributes["neatlogs.llm.choices.0.finish_reason"] == "stop"
    assert attributes["neatlogs.llm.choices.1.finish_reason"] == "tool_calls"
    assert attributes["neatlogs.llm.tool_calls.0.choice_index"] == 1
    assert attributes["neatlogs.llm.tool_calls.0.tool_call_index"] == 2
    assert attributes["neatlogs.llm.tool_calls.0.id"] == "call-1"


def test_google_adapter_preserves_candidates_independently(
    tracer_provider, in_memory_span_exporter
):
    span = tracer_provider.get_tracer("neatlogs.google_genai").start_span("generate")
    finalize_google(
        span,
        {
            "candidates": [
                {
                    "index": 0,
                    "finish_reason": "STOP",
                    "content": {"role": "model", "parts": [{"text": "first"}]},
                },
                {
                    "index": 1,
                    "finish_reason": "MAX_TOKENS",
                    "content": {"role": "model", "parts": [{"text": "second"}]},
                },
            ]
        },
        10.0,
    )

    attributes = in_memory_span_exporter.get_finished_spans()[0].attributes
    assert attributes["neatlogs.llm.output_messages.0.content"] == "first"
    assert attributes["neatlogs.llm.output_messages.1.content"] == "second"
    assert attributes["neatlogs.llm.choices.0.finish_reason"] == "STOP"
    assert attributes["neatlogs.llm.choices.1.finish_reason"] == "MAX_TOKENS"
