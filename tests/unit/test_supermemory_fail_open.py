import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from neatlogs import supermemory as wrapper


@pytest.mark.parametrize("method", ["search", "profile", "add"])
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    ("fault", "telemetry_fault"),
    [
        ("start", "set_attribute"),
        ("finish", "set_attribute"),
        ("finish", "set_status"),
        ("finish", "end"),
        ("error", "set_status"),
        ("error", "record_exception"),
        ("error", "end"),
        ("cancel", "set_attribute"),
        ("cancel", "set_status"),
        ("cancel", "end"),
    ],
)
def test_telemetry_failure_preserves_provider(
    monkeypatch, method, asynchronous, fault, telemetry_fault
):
    if fault == "cancel" and not asynchronous:
        pytest.skip("cancellation is asynchronous")
    calls = []
    response = object()
    provider_error = ValueError("provider error")
    cancelled = asyncio.CancelledError("cancelled")

    def invoke():
        calls.append(True)
        if fault == "error":
            raise provider_error
        if fault == "cancel":
            raise cancelled
        return response

    async def async_call(*args, **kwargs):
        return invoke()

    def sync_call(*args, **kwargs):
        return invoke()

    span = Mock()
    if fault != "start":
        getattr(span, telemetry_fault).side_effect = RuntimeError("telemetry failed")
    tracer = Mock()
    if fault == "start":
        tracer.start_span.side_effect = RuntimeError("telemetry start failed")
    else:
        tracer.start_span.return_value = span
    monkeypatch.setattr(wrapper, "get_provider_tracer", lambda: tracer)
    monkeypatch.setattr(wrapper, "is_suppressed", lambda: False)
    client = SimpleNamespace(**{method: async_call if asynchronous else sync_call})
    wrapper.wrap_supermemory_client(client)

    def run():
        result = getattr(client, method)("user-1", query="q", content="x")
        return asyncio.run(result) if asynchronous else result

    if fault in {"error", "cancel"}:
        expected = provider_error if fault == "error" else cancelled
        if asynchronous:

            async def check_error():
                with pytest.raises(type(expected)) as caught:
                    await getattr(client, method)("user-1", query="q", content="x")
                assert caught.value is expected

            asyncio.run(check_error())
        else:
            with pytest.raises(type(expected)) as caught:
                run()
            assert caught.value is expected
    else:
        assert run() is response
    assert calls == [True]
    if fault != "start":
        span.end.assert_called_once()


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("fault", ["start", "finish"])
def test_real_sdk_with_telemetry_failure(monkeypatch, asynchronous, fault):
    httpx = pytest.importorskip("httpx")
    supermemory = pytest.importorskip("supermemory")

    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json={"results": [], "total": 0})

    tracer = Mock()
    if fault == "start":
        tracer.start_span.side_effect = RuntimeError("telemetry start failed")
    else:
        tracer.start_span.return_value.end.side_effect = RuntimeError("telemetry end failed")
    monkeypatch.setattr(wrapper, "get_provider_tracer", lambda: tracer)
    monkeypatch.setattr(wrapper, "is_suppressed", lambda: False)
    if asynchronous:

        async def run():
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
                client = supermemory.AsyncSupermemory(
                    api_key="test-key", base_url="https://supermemory.invalid", http_client=http
                )
                wrapper.wrap_supermemory_client(client)
                result = await client.search("user-1", query="q")
                assert result.results == []

        asyncio.run(run())
    else:
        with httpx.Client(transport=httpx.MockTransport(handler)) as http:
            client = supermemory.Supermemory(
                api_key="test-key", base_url="https://supermemory.invalid", http_client=http
            )
            wrapper.wrap_supermemory_client(client)
            assert client.search("user-1", query="q").results == []
    assert len(requests) == 1
