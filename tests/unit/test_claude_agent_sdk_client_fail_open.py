import asyncio
from unittest.mock import Mock

import pytest

from neatlogs import claude_agent_sdk as integration


@pytest.mark.parametrize("stage", ["tracer", "span", "context", "wrapper"])
@pytest.mark.parametrize("provider_error", [False, True])
def test_client_setup_failure_preserves_receiver(monkeypatch, stage, provider_error):
    failure = RuntimeError("provider failure")
    calls = []
    pulled = []

    def receive(client):
        calls.append(client)

        async def messages():
            pulled.append(True)
            if provider_error:
                raise failure
            yield "native message"

        return messages()

    span = Mock()
    tracer = Mock()
    tracer.start_span.return_value = span
    monkeypatch.setattr(integration, "get_tracer", lambda: tracer)
    monkeypatch.setattr(integration, "is_suppressed", lambda: False)

    def broken(*args, **kwargs):
        raise ValueError("telemetry failure")

    if stage == "tracer":
        monkeypatch.setattr(integration, "get_tracer", broken)
    elif stage == "span":
        tracer.start_span.side_effect = broken
    elif stage == "context":
        monkeypatch.setattr(integration.otel_trace, "set_span_in_context", broken)
    else:
        monkeypatch.setattr(integration, "_TracingQuery", broken)

    client = object()

    async def run():
        result = integration._wrap_client_method(receive, "receive_messages")(client)
        return [message async for message in result]

    if provider_error:
        with pytest.raises(RuntimeError) as caught:
            asyncio.run(run())
        assert caught.value is failure
    else:
        assert asyncio.run(run()) == ["native message"]
    assert calls == [client]
    assert pulled == [True]
    if stage in ("context", "wrapper"):
        span.end.assert_called_once_with()


def test_client_setup_cleanup_failure_does_not_replace_messages(monkeypatch):
    span = Mock()
    span.end.side_effect = RuntimeError("end failure")
    tracer = Mock()
    tracer.start_span.return_value = span
    monkeypatch.setattr(integration, "get_tracer", lambda: tracer)
    monkeypatch.setattr(integration, "is_suppressed", lambda: False)

    def broken(*args, **kwargs):
        raise ValueError("context failure")

    monkeypatch.setattr(integration.otel_trace, "set_span_in_context", broken)

    async def receive(client):
        yield "native message"

    async def run():
        return [
            message
            async for message in integration._wrap_client_method(receive, "receive_response")(
                object()
            )
        ]

    assert asyncio.run(run()) == ["native message"]
