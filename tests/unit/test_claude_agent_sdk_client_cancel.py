import asyncio

import pytest
from opentelemetry.trace import StatusCode

from neatlogs import claude_agent_sdk as integration


@pytest.mark.parametrize("method", ["receive_messages", "receive_response"])
@pytest.mark.parametrize("with_message", [False, True])
def test_cancelled_native_receiver_ends_agent_span(
    monkeypatch, tracer_provider, in_memory_span_exporter, method, with_message
):
    cas = pytest.importorskip("claude_agent_sdk")
    integration._unpatch_claude_agent_sdk()
    monkeypatch.setattr(integration, "get_tracer", lambda: tracer_provider.get_tracer("test"))
    errors = []
    messages = []

    async def run():
        waiting = asyncio.Event()

        class Transport:
            async def receive_messages(self):
                if with_message:
                    yield {
                        "type": "assistant",
                        "message": {
                            "id": "offline",
                            "model": "claude-test",
                            "content": [{"type": "text", "text": "partial"}],
                        },
                    }
                waiting.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError as error:
                    errors.append(error)
                    raise

        client = cas.ClaudeSDKClient()
        client._query = Transport()
        integration._patch_claude_agent_sdk()

        async def consume():
            try:
                async for message in getattr(client, method)():
                    messages.append(message)
            except asyncio.CancelledError as error:
                assert error is errors[0]
                raise
            finally:
                assert integration._ACTIVE_RECEIVER.get() is None

        task = asyncio.create_task(consume())
        await waiting.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert task.cancelled()

    try:
        asyncio.run(run())
        spans = in_memory_span_exporter.get_finished_spans()
        agents = [span for span in spans if span.attributes.get("neatlogs.span.kind") == "agent"]
        assert [span.name for span in agents] == ["claude_agent." + method]
        assert agents[0].status.status_code == StatusCode.ERROR
        assert len(messages) == int(with_message)
        if with_message:
            assert (
                len([span for span in spans if span.attributes.get("neatlogs.span.kind") == "llm"])
                == 1
            )
    finally:
        integration._unpatch_claude_agent_sdk()
