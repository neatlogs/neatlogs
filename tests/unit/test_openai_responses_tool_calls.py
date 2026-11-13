from types import SimpleNamespace

from neatlogs.openai import wrap_openai_client


class _Responses:
    def create(self, **_kwargs):
        return SimpleNamespace(
            output=[
                SimpleNamespace(
                    type="function_call",
                    id="fc1",
                    call_id="call_1",
                    name="get_weather",
                    arguments='{"city": "Pune"}',
                )
            ],
            output_text="",
            model="test",
            usage=None,
        )


def test_responses_function_call_is_recorded(tracer_provider, in_memory_span_exporter):
    import neatlogs

    neatlogs.init(
        api_key="test",
        disable_export=True,
        tracer_provider=tracer_provider,
        register_shutdown_handlers=False,
    )
    wrapped = wrap_openai_client(SimpleNamespace(responses=_Responses()))
    wrapped.responses.create(model="test", input="weather in pune?")

    attrs = in_memory_span_exporter.get_finished_spans()[-1].attributes
    assert attrs["neatlogs.llm.tool_calls.0.name"] == "get_weather"
    assert attrs["neatlogs.llm.tool_calls.0.id"] == "call_1"
    assert attrs["neatlogs.llm.tool_calls.0.arguments"] == '{"city": "Pune"}'
