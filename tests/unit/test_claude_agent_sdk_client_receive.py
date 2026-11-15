import asyncio

import pytest

from neatlogs import claude_agent_sdk as integration


@pytest.fixture
def sdk(monkeypatch, tracer_provider):
    integration._unpatch_claude_agent_sdk()
    monkeypatch.setattr(integration, "get_tracer", lambda: tracer_provider.get_tracer("test"))
    yield integration
    integration._unpatch_claude_agent_sdk()


def _result():
    return dict(
        type="result",
        subtype="success",
        duration_ms=1,
        duration_api_ms=1,
        is_error=False,
        num_turns=1,
        session_id="offline",
        result="done",
    )


@pytest.mark.parametrize("method", ["receive_messages", "receive_response"])
def test_native_client_response_has_one_agent_span(sdk, in_memory_span_exporter, method):
    cas = pytest.importorskip("claude_agent_sdk")

    class Transport:
        async def receive_messages(self):
            yield _result()

    async def run():
        client = cas.ClaudeSDKClient()
        client._query = Transport()
        sdk._patch_claude_agent_sdk()
        return [message async for message in getattr(client, method)()]

    messages = asyncio.run(run())
    assert len(messages) == 1
    agents = [
        s
        for s in in_memory_span_exporter.get_finished_spans()
        if s.attributes.get("neatlogs.span.kind") == "agent"
    ]
    assert [s.name for s in agents] == ["claude_agent." + method]


def test_delegating_receiver_does_not_duplicate_messages(sdk, in_memory_span_exporter):
    class Client:
        async def receive_messages(self):
            yield _result()

        async def receive_response(self):
            async for message in self.receive_messages():
                yield message

    Client.receive_messages = sdk._wrap_client_method(Client.receive_messages, "receive_messages")
    Client.receive_response = sdk._wrap_client_method(Client.receive_response, "receive_response")

    async def run():
        return [message async for message in Client().receive_response()]

    assert asyncio.run(run()) == [_result()]
    agents = in_memory_span_exporter.get_finished_spans()
    assert [s.name for s in agents] == ["claude_agent.receive_response"]


def test_another_client_inside_receiver_still_gets_its_span(sdk, in_memory_span_exporter):
    class Client:
        def __init__(self, child=None):
            self.child = child

        async def receive_messages(self):
            if self.child is not None:
                async for message in self.child.receive_messages():
                    yield message
            else:
                yield _result()

    Client.receive_messages = sdk._wrap_client_method(Client.receive_messages, "receive_messages")

    async def run():
        return [message async for message in Client(Client()).receive_messages()]

    assert asyncio.run(run()) == [_result()]
    assert len(in_memory_span_exporter.get_finished_spans()) == 2


def test_interleaved_clients_are_independent(sdk, in_memory_span_exporter):
    class Client:
        async def receive_messages(self):
            await asyncio.sleep(0)
            yield _result()

        async def receive_response(self):
            async for message in self.receive_messages():
                yield message

    Client.receive_messages = sdk._wrap_client_method(Client.receive_messages, "receive_messages")
    Client.receive_response = sdk._wrap_client_method(Client.receive_response, "receive_response")

    async def consume(client):
        return [message async for message in client.receive_response()]

    async def run():
        return await asyncio.gather(consume(Client()), consume(Client()))

    assert asyncio.run(run()) == [[_result()], [_result()]]
    agents = in_memory_span_exporter.get_finished_spans()
    assert len(agents) == 2
    assert all(s.name == "claude_agent.receive_response" for s in agents)
    assert all(s.parent is None for s in agents)


def test_receiver_scope_is_reset_before_yielding_to_consumer(sdk, in_memory_span_exporter):
    class Client:
        async def receive_messages(self):
            yield _result()

    Client.receive_messages = sdk._wrap_client_method(Client.receive_messages, "receive_messages")

    async def run():
        client = Client()
        async for _ in client.receive_messages():
            assert [m async for m in client.receive_messages()] == [_result()]

    asyncio.run(run())
    assert len(in_memory_span_exporter.get_finished_spans()) == 2


def test_receiver_scope_is_reset_after_provider_error(sdk, in_memory_span_exporter):
    class Client:
        async def receive_messages(self):
            raise RuntimeError("offline failure")
            yield

    Client.receive_messages = sdk._wrap_client_method(Client.receive_messages, "receive_messages")

    async def run():
        for _ in range(2):
            with pytest.raises(RuntimeError, match="offline failure"):
                async for _ in Client().receive_messages():
                    pass

    asyncio.run(run())
    assert len(in_memory_span_exporter.get_finished_spans()) == 2
