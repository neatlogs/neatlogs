from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from neatlogs.google_genai import _set_input_attributes


def test_multiple_tool_groups_keep_every_function_declaration():
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    span = provider.get_tracer("test").start_span("google-genai-input")

    _set_input_attributes(
        span,
        "hello",
        {
            "config": {
                "tools": [
                    {
                        "function_declarations": [
                            {"name": "alpha_zero"},
                            {"name": "alpha_one"},
                        ]
                    },
                    {
                        "function_declarations": [
                            {"name": "beta_zero"},
                            {"name": "beta_one"},
                        ]
                    },
                ]
            }
        },
    )
    span.end()

    attributes = exporter.get_finished_spans()[0].attributes
    assert [attributes[f"neatlogs.llm.tools.{index}.name"] for index in range(4)] == [
        "alpha_zero",
        "alpha_one",
        "beta_zero",
        "beta_one",
    ]
