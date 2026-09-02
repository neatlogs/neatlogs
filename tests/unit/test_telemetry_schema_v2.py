import hashlib
import json

from jsonschema import Draft202012Validator
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

import neatlogs
from neatlogs.core.choice_accumulator import ChoiceAccumulator
from neatlogs.core.span_processor import NeatlogsSpanProcessor

EXPECTED_PRECEDENCE = [
    "native-v2",
    "neatlogs-direct",
    "otel-genai",
    "openinference",
    "provider-specific",
    "external-legacy",
    "unknown-raw",
]


def test_canonical_telemetry_schema_is_packaged_verbatim():
    schema_bytes = neatlogs.telemetry_schema_bytes()

    assert hashlib.sha256(schema_bytes).hexdigest() == neatlogs.TELEMETRY_SCHEMA_SHA256
    assert json.loads(schema_bytes) == neatlogs.telemetry_schema()
    neatlogs.verify_telemetry_schema()


def test_canonical_telemetry_policy_is_consumed_by_the_sdk():
    schema = neatlogs.telemetry_schema()
    policy = schema["x-neatlogs-policy"]

    assert neatlogs.TELEMETRY_SCHEMA_VERSION == 2
    assert policy["contract_version"] == neatlogs.TELEMETRY_CONTRACT_VERSION
    assert policy["conflict_precedence"] == EXPECTED_PRECEDENCE
    assert policy["tool_calls"]["execution_is_separate_tool_span"] is True
    assert policy["root_finalization"]["launch_sdk_auto_workflow_roots"] is True


def test_diagnostic_exporter_normalizes_post_processor_llm_span():
    provider = TracerProvider()
    lifecycle = NeatlogsSpanProcessor(emit_completion_markers=False, own_all_spans=True)
    diagnostic = neatlogs.InMemoryDiagnosticSpanExporter(max_spans=4, provider_generation=7)
    provider.add_span_processor(lifecycle)
    provider.add_span_processor(SimpleSpanProcessor(diagnostic))

    try:
        span = provider.get_tracer("neatlogs.openai", "1.2.3").start_span("chat.completions")
        span.set_attribute("openinference.span.kind", "LLM")
        span.set_attribute("input.value", '{"prompt":"hello"}')
        span.set_attribute("input.mime_type", "application/json")
        accumulator = ChoiceAccumulator()
        accumulator.add_response(
            {
                "id": "resp_1",
                "model": "gpt-test",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "type": "function",
                                    "function": {
                                        "name": "weather",
                                        "arguments": '{"city":"Kolkata"}',
                                    },
                                }
                            ],
                        },
                    },
                    {
                        "index": 1,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "sunny"},
                    },
                ],
                "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
            }
        )
        accumulator.apply(span)
        span.end()

        payload = diagnostic.get_finished_envelopes()[0].to_dict()
        Draft202012Validator(neatlogs.telemetry_schema()).validate(payload)
        assert payload["schema_version"] == 2
        assert payload["kind"] == "LLM"
        assert payload["ownership"]["provider_generation"] == 7
        assert payload["wrapper"] == {
            "captured": True,
            "integration": "neatlogs.openai",
            "integration_version": "1.2.3",
            "capture_fidelity": "native",
        }
        assert payload["input"] == {
            "type": "json",
            "value": {"prompt": "hello"},
            "media": [],
        }
        choices = payload["semantic"]["response"]["choices"]
        assert [choice["choice_index"] for choice in choices] == [0, 1]
        assert choices[0]["message"]["tool_calls"][0]["name"] == "weather"
        assert choices[0]["message"]["tool_calls"][0]["id_origin"] == ("deterministic-synthetic")
        assert payload["semantic"]["usage"]["total_tokens"] == 5
    finally:
        provider.shutdown()


def test_canonical_llm_messages_include_typed_media_parts():
    provider = TracerProvider()
    diagnostic = neatlogs.InMemoryDiagnosticSpanExporter(max_spans=1)
    provider.add_span_processor(SimpleSpanProcessor(diagnostic))
    span = provider.get_tracer("neatlogs.openai").start_span("chat")
    span.set_attribute("neatlogs.span.kind", "llm")
    span.set_attribute("neatlogs.llm.input_messages.0.role", "user")
    prefix = "neatlogs.llm.input_messages.0.media.0"
    span.set_attribute(f"{prefix}.id", "123e4567-e89b-12d3-a456-426614174000")
    span.set_attribute(f"{prefix}.type", "image")
    span.set_attribute(f"{prefix}.sha256", "a" * 64)
    span.set_attribute(f"{prefix}.mime_type", "image/png")
    span.set_attribute(f"{prefix}.byte_length", 123)
    span.set_attribute(f"{prefix}.source", "uploaded")
    span.set_attribute(f"{prefix}.state", "available")
    span.end()
    provider.shutdown()

    message = diagnostic.get_finished_envelopes()[0].to_dict()["semantic"]["request"]["messages"][0]
    assert message["content"] == [
        {
            "type": "image",
            "reference": {
                "id": "123e4567-e89b-12d3-a456-426614174000",
                "sha256": "a" * 64,
                "mime_type": "image/png",
                "byte_length": 123,
                "source": "uploaded",
                "purpose": "input",
                "state": "available",
                "safe_preview": None,
            },
        }
    ]


def test_diagnostic_exporter_is_bounded_and_reports_eviction():
    provider = TracerProvider()
    diagnostic = neatlogs.InMemoryDiagnosticSpanExporter(max_spans=2)
    provider.add_span_processor(SimpleSpanProcessor(diagnostic))
    tracer = provider.get_tracer("neatlogs.test")
    try:
        for index in range(3):
            tracer.start_span(f"span-{index}").end()
        assert [item.name for item in diagnostic.get_finished_envelopes()] == [
            "span-1",
            "span-2",
        ]
        assert diagnostic.dropped_count == 1
    finally:
        provider.shutdown()
