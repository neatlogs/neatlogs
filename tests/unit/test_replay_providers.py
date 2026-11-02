"""Installed SDK-class capability proofs; no provider or application credentials."""

import json
from types import SimpleNamespace

import pytest

import neatlogs
from neatlogs.replay import ReplaySession
from neatlogs.replay.provider import (
    MODEL_CAPTURE_ATTRIBUTE,
    capture_digest,
    ecmascript_json,
    read_provider_capture,
    response_class,
    restore_response,
)

CHAT = "openai.chat.completions.create"
RESPONSES = "openai.responses.create"
GOOGLE = "google_genai.models.generate_content"
REQUESTS = {
    CHAT: {"model": "fixture", "messages": [{"role": "user", "content": "fixture"}]},
    RESPONSES: {"model": "fixture", "input": "fixture"},
    GOOGLE: {"model": "fixture", "contents": "fixture"},
}
OUTPUTS = {
    CHAT: {
        "id": "chat_fixture",
        "object": "chat.completion",
        "created": 1,
        "model": "fixture",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "hello"},
            }
        ],
        "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
    },
    RESPONSES: {
        "id": "resp_fixture",
        "object": "response",
        "created_at": 1,
        "model": "fixture",
        "status": "completed",
        "output": [
            {
                "id": "msg_fixture",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "hello", "annotations": []}],
            }
        ],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "output_text": "hello",
        "usage": {
            "input_tokens": 11,
            "output_tokens": 7,
            "total_tokens": 18,
            "input_tokens_details": {"cached_tokens": 2},
            "output_tokens_details": {"reasoning_tokens": 1},
        },
    },
    GOOGLE: {
        "candidates": [
            {"content": {"parts": [{"text": "hello"}], "role": "model"}, "finishReason": "STOP"}
        ],
        "usageMetadata": {"promptTokenCount": 11, "candidatesTokenCount": 7, "totalTokenCount": 18},
    },
}


def recording(operation, output=None):
    return {
        "version": 1,
        "sourceTraceId": "fixture-trace",
        "entries": [
            {
                "spanId": "model",
                "kind": "model",
                "name": operation,
                "input": REQUESTS[operation],
                "output": output or OUTPUTS[operation],
            },
            {
                "spanId": "checkpoint",
                "kind": "tool",
                "name": "checkpoint",
                "input": {},
                "output": {},
            },
        ],
    }


def wrapper(operation, resource, asynchronous=False):
    if operation == GOOGLE:
        from neatlogs.google_genai import _patch_async_models, _patch_models

        (_patch_async_models if asynchronous else _patch_models)(resource)
        return resource.generate_content
    from neatlogs.openai import (
        _patch_async_completions,
        _patch_async_responses,
        _patch_completions,
        _patch_responses,
    )

    if operation == CHAT:
        (_patch_async_completions if asynchronous else _patch_completions)(resource)
    else:
        (_patch_async_responses if asynchronous else _patch_responses)(resource)
    return resource.create


@pytest.mark.parametrize("operation", [CHAT, RESPONSES, GOOGLE])
def test_installed_sdk_response_classes_and_getters_restore_losslessly(operation):
    result = restore_response(operation, OUTPUTS[operation])
    assert type(result) is response_class(operation)
    if operation == CHAT:
        assert result.choices[0].message.content == "hello"
    elif operation == RESPONSES:
        assert result.output_text == "hello"
    else:
        assert result.text == "hello"
        assert result.usage_metadata.total_token_count == 18


def test_google_function_call_getter_preserves_saved_call_and_thought_signature():
    output = {
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": [
                        {
                            "functionCall": {"name": "lookup", "args": {"value": 1}},
                            "thoughtSignature": "AQID",
                        }
                    ],
                }
            }
        ]
    }
    result = restore_response(GOOGLE, output)
    assert result.function_calls[0].name == "lookup"
    assert result.function_calls[0].args == {"value": 1}
    assert result.candidates[0].content.parts[0].thought_signature == b"\x01\x02\x03"


@pytest.mark.parametrize("operation", [CHAT, RESPONSES, GOOGLE])
def test_model_hit_skips_live_callable_and_counts_no_billed_tokens(
    operation, tracer_provider, in_memory_span_exporter
):
    neatlogs.init(
        tracer_provider=tracer_provider, disable_export=True, register_shutdown_handlers=False
    )
    calls = []

    def live(**kwargs):
        calls.append(kwargs)
        return response_class(operation).model_validate(OUTPUTS[operation])

    resource = SimpleNamespace(
        **({"generate_content": live} if operation == GOOGLE else {"create": live})
    )
    call = wrapper(operation, resource)
    with ReplaySession(recording(operation), mode="until_span", until_span="checkpoint") as session:
        result = call(**REQUESTS[operation])
    assert type(result) is response_class(operation)
    assert calls == []
    assert session.get_status()["hits"] == 1
    span = next(
        span
        for span in in_memory_span_exporter.get_finished_spans()
        if span.attributes.get("neatlogs.replay.outcome") == "HIT"
    )
    metrics = {
        key: value
        for key, value in span.attributes.items()
        if key.startswith("neatlogs.llm.token_count.")
    }
    assert metrics and all(value == 0 for value in metrics.values())
    assert MODEL_CAPTURE_ATTRIBUTE not in span.attributes


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", [CHAT, RESPONSES, GOOGLE])
async def test_async_provider_hits_preserve_native_response_class(operation):
    calls = []

    async def live(**kwargs):
        calls.append(kwargs)
        return response_class(operation).model_validate(OUTPUTS[operation])

    resource = SimpleNamespace(
        **({"generate_content": live} if operation == GOOGLE else {"create": live})
    )
    call = wrapper(operation, resource, asynchronous=True)
    async with ReplaySession(
        recording(operation), mode="until_span", until_span="checkpoint"
    ) as session:
        result = await call(**REQUESTS[operation])
    assert type(result) is response_class(operation)
    assert calls == [] and session.get_status()["hits"] == 1


@pytest.mark.parametrize("operation", [CHAT, RESPONSES, GOOGLE])
def test_live_capture_is_verified_and_immutable_before_request_mutation(
    operation, tracer_provider, in_memory_span_exporter
):
    neatlogs.init(
        tracer_provider=tracer_provider, disable_export=True, register_shutdown_handlers=False
    )
    original = json.loads(json.dumps(REQUESTS[operation]))

    def live(**kwargs):
        if "messages" in kwargs:
            kwargs["messages"][0]["content"] = "mutated by provider"
        return response_class(operation).model_validate(OUTPUTS[operation])

    resource = SimpleNamespace(
        **({"generate_content": live} if operation == GOOGLE else {"create": live})
    )
    result = wrapper(operation, resource)(**json.loads(json.dumps(original)))
    assert type(result) is response_class(operation)
    span = next(
        span
        for span in in_memory_span_exporter.get_finished_spans()
        if MODEL_CAPTURE_ATTRIBUTE in span.attributes
    )
    saved = read_provider_capture(span.attributes[MODEL_CAPTURE_ATTRIBUTE])
    assert saved["input"] == original
    corrupted = json.loads(span.attributes[MODEL_CAPTURE_ATTRIBUTE])
    corrupted["input"]["model"] = "tampered"
    with pytest.raises(ValueError):
        read_provider_capture(json.dumps(corrupted))


def test_unsupported_transport_options_switch_live_and_never_capture(
    tracer_provider, in_memory_span_exporter
):
    neatlogs.init(
        tracer_provider=tracer_provider, disable_export=True, register_shutdown_handlers=False
    )
    calls = []

    def live(**kwargs):
        calls.append(kwargs)
        return response_class(CHAT).model_validate(OUTPUTS[CHAT])

    call = wrapper(CHAT, SimpleNamespace(create=live))
    with ReplaySession(recording(CHAT), mode="until_span", until_span="checkpoint") as session:
        call(**REQUESTS[CHAT], extra_headers={"Authorization": "PRIVATE FIXTURE"})
        call(**REQUESTS[CHAT])
    assert len(calls) == 2
    assert session.get_status()["firstMiss"] == {"kind": "model", "name": CHAT}
    first = next(
        span
        for span in in_memory_span_exporter.get_finished_spans()
        if span.attributes.get("neatlogs.replay.outcome") == "MISS"
    )
    assert MODEL_CAPTURE_ATTRIBUTE not in first.attributes


def test_raw_capture_is_removed_from_masked_export(tracer_provider, in_memory_span_exporter):
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from neatlogs.core.masking_exporter import MaskingSpanExporter

    span = tracer_provider.get_tracer("neatlogs.test").start_span("fixture")
    span.set_attribute(MODEL_CAPTURE_ATTRIBUTE, "private original copy")
    span.end()
    source = in_memory_span_exporter.get_finished_spans()[0]
    sink = InMemorySpanExporter()
    MaskingSpanExporter(sink, lambda snapshot: snapshot).export([source])
    assert MODEL_CAPTURE_ATTRIBUTE not in sink.get_finished_spans()[0].attributes
    assert MODEL_CAPTURE_ATTRIBUTE in source.attributes


def test_unsupported_google_stream_never_restores_a_cached_model_null_input():
    calls = []

    def live(**kwargs):
        calls.append(kwargs)
        return iter([response_class(GOOGLE).model_validate(OUTPUTS[GOOGLE])])

    resource = SimpleNamespace(generate_content=lambda **kwargs: None, generate_content_stream=live)
    wrapper(GOOGLE, resource)
    saved = recording(GOOGLE)
    saved["entries"][0]["input"] = None
    with ReplaySession(saved, mode="until_span", until_span="checkpoint") as session:
        assert len(list(resource.generate_content_stream(**REQUESTS[GOOGLE]))) == 1
    assert len(calls) == 1
    assert session.get_status()["hits"] == 0
    assert session.get_status()["liveCalls"] == 1
    assert session.get_status()["reason"] == "missing_recording"
