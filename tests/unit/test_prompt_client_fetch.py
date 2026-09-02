import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import AsyncMock, Mock, call

import pytest

from neatlogs.prompt.client import AsyncPromptClient, PromptClient, PromptHandle

PROMPT = {
    "id": "00000000-0000-4000-8000-000000000001",
    "name": "support prompt",
    "version": 123,
    "content": "Hello {{name}}",
    "messages": None,
    "config": {},
    "labels": [],
    "updatedAt": "2026-09-02T00:00:00.000Z",
}


@pytest.mark.parametrize(
    ("version", "expected_params"),
    [(123, {"version": 123}), (None, {"latest": "true"})],
)
def test_sync_fetch_uses_one_exact_endpoint(version, expected_params):
    client = PromptClient(base_url="https://example.test", api_key="test-key")
    client._request_json = Mock(return_value=PROMPT)

    handle = client._fetch_prompt("support prompt", version=version)

    assert handle.version == 123
    client._request_json.assert_called_once_with(
        method="GET",
        path="/api/v1/prompts/support%20prompt/fetch",
        params=expected_params,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("version", "expected_params"),
    [(123, {"version": 123}), (None, {"latest": "true"})],
)
async def test_async_fetch_uses_one_exact_endpoint(version, expected_params):
    client = AsyncPromptClient(base_url="https://example.test", api_key="test-key")
    client._request_json = AsyncMock(return_value=PROMPT)
    try:
        handle = await client._fetch_prompt("support prompt", version=version)
    finally:
        await client.close()

    assert handle.version == 123
    client._request_json.assert_awaited_once_with(
        method="GET",
        path="/api/v1/prompts/support%20prompt/fetch",
        params=expected_params,
    )


def test_sync_cold_cache_miss_is_coalesced():
    client = PromptClient(base_url="https://example.test", api_key="test-key")
    calls = 0

    def fetch(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        time.sleep(0.05)
        return PromptHandle(client_prompt())

    client._fetch_prompt = fetch
    with ThreadPoolExecutor(max_workers=2) as executor:
        handles = list(executor.map(lambda _: client.get_prompt("support prompt"), range(2)))

    assert calls == 1
    assert [handle.version for handle in handles] == [123, 123]


def test_sync_coalesced_failure_wakes_all_waiters():
    client = PromptClient(base_url="https://example.test", api_key="test-key")
    started = threading.Event()
    release = threading.Event()
    calls = 0

    def fetch(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        started.set()
        release.wait(timeout=1)
        raise RuntimeError("backend unavailable")

    client._fetch_prompt = fetch
    with ThreadPoolExecutor(max_workers=2) as executor:
        leader = executor.submit(client.get_prompt, "support prompt")
        assert started.wait(timeout=1)
        follower = executor.submit(client.get_prompt, "support prompt")
        release.set()

        with pytest.raises(RuntimeError, match="backend unavailable"):
            leader.result()
        with pytest.raises(RuntimeError, match="backend unavailable"):
            follower.result()

    assert calls == 1


def test_mutation_resolves_exact_version_without_listing_history():
    client = PromptClient(base_url="https://example.test", api_key="test-key")
    client._request_json = Mock(side_effect=[PROMPT, {"deleted": True}])

    result = client.delete_prompt("support prompt", 123)

    assert result == {"deleted": True}
    assert client._request_json.call_args_list == [
        call(
            method="GET",
            path="/api/v1/prompts/support%20prompt/fetch",
            params={"version": 123},
        ),
        call(
            method="DELETE",
            path="/api/managed-prompts/00000000-0000-4000-8000-000000000001",
        ),
    ]


@pytest.mark.asyncio
async def test_async_cold_cache_miss_is_coalesced():
    client = AsyncPromptClient(base_url="https://example.test", api_key="test-key")
    calls = 0

    async def fetch(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.05)
        return PromptHandle(client_prompt())

    client._fetch_prompt = fetch
    try:
        handles = await asyncio.gather(
            client.get_prompt("support prompt"),
            client.get_prompt("support prompt"),
        )
    finally:
        await client.close()

    assert calls == 1
    assert [handle.version for handle in handles] == [123, 123]


@pytest.mark.asyncio
async def test_async_caller_cancellation_does_not_cancel_shared_fetch():
    client = AsyncPromptClient(base_url="https://example.test", api_key="test-key")
    started = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    async def fetch(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        started.set()
        await release.wait()
        return PromptHandle(client_prompt())

    client._fetch_prompt = fetch
    try:
        cancelled_caller = asyncio.create_task(client.get_prompt("support prompt"))
        await started.wait()
        surviving_caller = asyncio.create_task(client.get_prompt("support prompt"))

        cancelled_caller.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled_caller

        release.set()
        handle = await surviving_caller
    finally:
        await client.close()

    assert calls == 1
    assert handle.version == 123


def client_prompt():
    from neatlogs.prompt.client import _normalize_prompt_object

    return _normalize_prompt_object(PROMPT)
