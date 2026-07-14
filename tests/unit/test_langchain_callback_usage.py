import importlib
from uuid import uuid4

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

messages = pytest.importorskip("langchain_core.messages")
outputs = pytest.importorskip("langchain_core.outputs")
AIMessage = messages.AIMessage
HumanMessage = messages.HumanMessage
ChatGeneration = outputs.ChatGeneration
LLMResult = outputs.LLMResult
langchain_integration = importlib.import_module("neatlogs.langchain")


async def _capture_callback_span(monkeypatch, in_memory_span_exporter, response):
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(in_memory_span_exporter))
    monkeypatch.setattr(
        langchain_integration,
        "get_tracer",
        lambda: provider.get_tracer("neatlogs.test.langchain"),
    )
    monkeypatch.setattr(langchain_integration, "_auto_root_enabled", lambda: False)

    handler = langchain_integration.NeatlogsCallbackHandler()
    run_id = uuid4()
    await handler.on_chat_model_start(
        {"id": ["langchain_google_genai", "ChatGoogleGenerativeAI"]},
        [[HumanMessage(content="Say hello")]],
        run_id=run_id,
        invocation_params={"model": "gemini-2.5-flash"},
    )

    await handler.on_llm_end(response, run_id=run_id)

    return next(
        span
        for span in in_memory_span_exporter.get_finished_spans()
        if span.name == "langchain.chat_model"
    )


def _chat_result(usage_metadata, llm_output=None):
    return LLMResult(
        generations=[
            [
                ChatGeneration(
                    message=AIMessage(
                        content="Hello",
                        usage_metadata=usage_metadata,
                    )
                )
            ]
        ],
        llm_output=llm_output or {},
    )


@pytest.mark.asyncio
async def test_callback_records_standard_message_usage_metadata(
    monkeypatch, in_memory_span_exporter
):
    response = _chat_result(
        {
            "input_tokens": 7,
            "output_tokens": 329,
            "total_tokens": 336,
            "input_token_details": {
                "cache_read": 5,
                "cache_creation": 2,
            },
            "output_token_details": {"reasoning": 325},
        }
    )
    span = await _capture_callback_span(monkeypatch, in_memory_span_exporter, response)

    assert span.attributes["neatlogs.llm.token_count.prompt"] == 7
    assert span.attributes["neatlogs.llm.token_count.completion"] == 329
    assert span.attributes["neatlogs.llm.token_count.total"] == 336
    assert span.attributes["neatlogs.llm.token_count.cache_read"] == 5
    assert span.attributes["neatlogs.llm.token_count.cache_write"] == 2
    assert span.attributes["neatlogs.llm.token_count.reasoning"] == 325


@pytest.mark.asyncio
async def test_callback_normalizes_standard_llm_output_usage(monkeypatch, in_memory_span_exporter):
    response = _chat_result(
        None,
        llm_output={
            "usage": {
                "input_tokens": 11,
                "output_tokens": 13,
                "total_tokens": 24,
                "input_token_details": {
                    "cache_read": 3,
                    "cache_creation": 1,
                },
                "output_token_details": {"reasoning": 8},
            }
        },
    )
    span = await _capture_callback_span(monkeypatch, in_memory_span_exporter, response)

    assert span.attributes["neatlogs.llm.token_count.prompt"] == 11
    assert span.attributes["neatlogs.llm.token_count.completion"] == 13
    assert span.attributes["neatlogs.llm.token_count.total"] == 24
    assert span.attributes["neatlogs.llm.token_count.cache_read"] == 3
    assert span.attributes["neatlogs.llm.token_count.cache_write"] == 1
    assert span.attributes["neatlogs.llm.token_count.reasoning"] == 8


@pytest.mark.asyncio
async def test_callback_fills_missing_legacy_fields_from_message_usage(
    monkeypatch, in_memory_span_exporter
):
    response = _chat_result(
        {
            "input_tokens": 100,
            "output_tokens": 20,
            "total_tokens": 120,
        },
        llm_output={"token_usage": {"prompt_tokens": 10}},
    )
    span = await _capture_callback_span(monkeypatch, in_memory_span_exporter, response)

    assert span.attributes["neatlogs.llm.token_count.prompt"] == 10
    assert span.attributes["neatlogs.llm.token_count.completion"] == 20
    assert span.attributes["neatlogs.llm.token_count.total"] == 120


@pytest.mark.asyncio
async def test_callback_combines_llm_output_usage_sources_before_message_fallback(
    monkeypatch, in_memory_span_exporter
):
    response = _chat_result(
        {
            "input_tokens": 100,
            "output_tokens": 200,
            "total_tokens": 300,
        },
        llm_output={
            "token_usage": {"prompt_tokens": 10},
            "usage": {"output_tokens": 15, "total_tokens": 25},
        },
    )
    span = await _capture_callback_span(monkeypatch, in_memory_span_exporter, response)

    assert span.attributes["neatlogs.llm.token_count.prompt"] == 10
    assert span.attributes["neatlogs.llm.token_count.completion"] == 15
    assert span.attributes["neatlogs.llm.token_count.total"] == 25


@pytest.mark.asyncio
async def test_callback_aggregates_message_usage_for_batched_prompts(
    monkeypatch, in_memory_span_exporter
):
    response = LLMResult(
        generations=[
            [
                ChatGeneration(
                    message=AIMessage(
                        content="First",
                        usage_metadata={
                            "input_tokens": 3,
                            "output_tokens": 4,
                            "total_tokens": 7,
                        },
                    )
                ),
                ChatGeneration(
                    message=AIMessage(
                        content="Alternate candidate",
                        usage_metadata={
                            "input_tokens": 100,
                            "output_tokens": 200,
                            "total_tokens": 300,
                        },
                    )
                ),
            ],
            [
                ChatGeneration(
                    message=AIMessage(
                        content="Second",
                        usage_metadata={
                            "input_tokens": 5,
                            "output_tokens": 6,
                            "total_tokens": 11,
                        },
                    )
                )
            ],
        ],
        llm_output={},
    )
    span = await _capture_callback_span(monkeypatch, in_memory_span_exporter, response)

    assert span.attributes["neatlogs.llm.token_count.prompt"] == 8
    assert span.attributes["neatlogs.llm.token_count.completion"] == 10
    assert span.attributes["neatlogs.llm.token_count.total"] == 18
