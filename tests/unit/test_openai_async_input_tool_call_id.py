from types import SimpleNamespace

import pytest

from neatlogs.openai import wrap_async_openai_client


class _Completions:
    async def create(self, **_kwargs):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    index=0,
                    message=SimpleNamespace(role="assistant", content="ok", tool_calls=[]),
                    finish_reason="stop",
                )
            ],
            usage=None,
            model="test",
            id="response-1",
        )


@pytest.mark.asyncio
async def test_async_openai_preserves_input_tool_call_id(tracer_provider, in_memory_span_exporter):
    import neatlogs

    neatlogs.init(
        api_key="test",
        disable_export=True,
        tracer_provider=tracer_provider,
        register_shutdown_handlers=False,
    )
    client = SimpleNamespace(chat=SimpleNamespace(completions=_Completions()))
    wrapped = wrap_async_openai_client(client)

    await wrapped.chat.completions.create(
        model="test",
        messages=[{"role": "tool", "content": "sunny", "tool_call_id": "call_123"}],
    )

    attributes = in_memory_span_exporter.get_finished_spans()[-1].attributes
    assert attributes["neatlogs.llm.input_messages.0.role"] == "tool"
    assert attributes["neatlogs.llm.input_messages.0.tool_call_id"] == "call_123"
