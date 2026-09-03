import asyncio
import importlib
import threading
from unittest.mock import AsyncMock, Mock, call

import httpx
import pytest
import requests

import neatlogs
import neatlogs.prompt.client as prompt_client_module
from neatlogs.prompt import PromptClientClosedError as ExportedPromptClientClosedError
from neatlogs.prompt.client import (
    AsyncPromptClient,
    CachedPrompt,
    PromptApiError,
    PromptCache,
    PromptClient,
    PromptClientClosedError,
    PromptHandle,
)

init_module = importlib.import_module("neatlogs.init")


def _prompt(version: int, content: str) -> PromptHandle:
    return PromptHandle(
        CachedPrompt(
            id=f"prompt-{version}",
            name="assistant",
            version=version,
            content=content,
            messages=None,
            config={},
            labels=[],
            updated_at=f"2026-09-02T00:00:0{version}Z",
        )
    )


def _raw_prompt(version: int, content: str = "content") -> dict:
    return {
        "id": f"prompt-{version}",
        "name": "assistant",
        "version": version,
        "content": content,
    }


def _key(name: str, *, label: str | None = None, version: int | None = None):
    return PromptCache.cache_key(name, label=label, version=version)


def _expire(client: PromptClient | AsyncPromptClient, key) -> None:
    entry = client._cache.get(key)
    assert entry is not None
    entry.fetched_at -= entry.ttl_seconds + 1


def test_closed_error_is_exported_from_prompt_and_package_namespaces():
    assert ExportedPromptClientClosedError is PromptClientClosedError
    assert neatlogs.PromptClientClosedError is PromptClientClosedError


def test_sync_stale_refresh_is_coalesced_and_preserves_entry_ttl():
    client = PromptClient(base_url="https://example.test", api_key="test", cache_ttl_seconds=60)
    initial = _prompt(1, "old")
    refreshed = _prompt(2, "new")
    refresh_started = threading.Event()
    release_refresh = threading.Event()

    def fetch(*_args, **_kwargs):
        if not client._cache.get(_key("assistant")):
            return initial
        refresh_started.set()
        assert release_refresh.wait(timeout=2)
        return refreshed

    client._fetch_prompt = Mock(side_effect=fetch)
    assert client.get_prompt("assistant", cache_ttl_seconds=7).content == "old"
    _expire(client, _key("assistant"))

    assert client.get_prompt("assistant").content == "old"
    assert refresh_started.wait(timeout=2)
    assert client.get_prompt("assistant").content == "old"
    assert client._fetch_prompt.call_count == 2

    release_refresh.set()
    for _ in range(100):
        entry = client._cache.get(_key("assistant"))
        if entry is not None and entry.value.content == "new":
            break
        threading.Event().wait(0.01)
    else:
        pytest.fail("background prompt refresh did not finish")

    entry = client._cache.get(_key("assistant"))
    assert entry is not None
    assert entry.ttl_seconds == 7


def test_sync_pinned_version_is_not_refreshed_after_ttl():
    client = PromptClient(base_url="https://example.test", api_key="test", cache_ttl_seconds=1)
    client._fetch_prompt = Mock(return_value=_prompt(3, "pinned"))

    first = client.get_prompt("assistant", version=3)
    _expire(client, _key("assistant", version=3))
    second = client.get_prompt("assistant", version=3)

    assert second is first
    client._fetch_prompt.assert_called_once()


def test_cache_is_lru_bounded_and_crud_invalidation_preserves_other_prompts():
    cache = PromptCache(max_entries=2)
    cache.set(_key("alpha"), _prompt(1, "alpha"))
    cache.set(_key("beta"), _prompt(1, "beta"))
    assert cache.get(_key("alpha")) is not None

    cache.set(_key("gamma"), _prompt(1, "gamma"))
    assert cache.get(_key("beta")) is None
    cache.set(_key("alpha", label="production"), _prompt(2, "alpha-label"))
    cache.invalidate_prompt("alpha")
    assert cache.get(_key("alpha")) is None
    assert cache.get(_key("alpha", label="production")) is None
    assert cache.get(_key("gamma")) is not None


def test_cache_keys_do_not_collide_for_names_containing_selector_text():
    assert _key("a@label:b") != _key("a", label="b@latest")


def test_cached_prompt_nested_values_are_returned_as_defensive_copies():
    original_messages = [{"role": "system", "content": "original"}]
    original_config = {"nested": {"temperature": 0.2}}
    handle = PromptHandle(
        CachedPrompt(
            id="prompt-1",
            name="assistant",
            version=1,
            content="",
            messages=original_messages,
            config=original_config,
            labels=[],
            updated_at="2026-09-03T00:00:00Z",
            type="chat",
        )
    )

    messages = handle.messages
    config = handle.config
    assert messages is not None
    messages[0]["content"] = "changed"
    config["nested"]["temperature"] = 1.0
    original_messages[0]["content"] = "changed at source"
    original_config["nested"]["temperature"] = 2.0

    assert handle.messages == [{"role": "system", "content": "original"}]
    assert handle.config == {"nested": {"temperature": 0.2}}


def test_chat_type_is_inferred_from_backend_messages():
    prompt = prompt_client_module._normalize_prompt_object(
        {
            "id": "prompt-1",
            "name": "assistant",
            "version": 1,
            "messages": [{"role": "system", "content": "hello"}],
        }
    )

    assert prompt.type == "chat"


def test_sync_refresh_workers_are_bounded_and_close_rejects_new_work():
    session = Mock()
    client = PromptClient(
        base_url="https://example.test",
        api_key="test",
        session=session,
        max_refresh_workers=1,
    )
    client._cache.set(_key("alpha"), _prompt(1, "alpha"), ttl=0)
    client._cache.set(_key("beta"), _prompt(1, "beta"), ttl=0)
    refresh_started = threading.Event()
    release_refresh = threading.Event()

    def fetch(*_args, **_kwargs):
        refresh_started.set()
        assert release_refresh.wait(timeout=2)
        return _prompt(2, "refreshed")

    client._fetch_prompt = Mock(side_effect=fetch)
    assert client.get_prompt("alpha").content == "alpha"
    assert refresh_started.wait(timeout=2)
    assert client.get_prompt("beta").content == "beta"
    assert client._fetch_prompt.call_count == 1

    release_refresh.set()
    client.close(timeout_seconds=2)
    session.close.assert_called_once()
    with pytest.raises(PromptClientClosedError):
        client.get_prompt("alpha")


def test_sync_close_rejects_every_public_network_method():
    client = PromptClient(base_url="https://example.test", api_key="test")
    client.close()

    with pytest.raises(PromptClientClosedError):
        client.list_prompts()
    with pytest.raises(PromptClientClosedError):
        client.fetch_prompt("assistant", label="production")
    with pytest.raises(PromptClientClosedError):
        client.create_prompt(name="assistant", prompt="hello", labels=["production"])
    with pytest.raises(PromptClientClosedError):
        client.save_as_version(prompt_name="assistant", content="hello")


def test_sync_cold_misses_are_coalesced():
    client = PromptClient(base_url="https://example.test", api_key="test")
    fetch_started = threading.Event()
    release_fetch = threading.Event()

    def fetch(*_args, **_kwargs):
        fetch_started.set()
        assert release_fetch.wait(timeout=2)
        return _prompt(1, "shared")

    client._fetch_prompt = Mock(side_effect=fetch)
    results = []
    first = threading.Thread(target=lambda: results.append(client.get_prompt("assistant")))
    second = threading.Thread(target=lambda: results.append(client.get_prompt("assistant")))
    first.start()
    assert fetch_started.wait(timeout=2)
    second.start()
    release_fetch.set()
    first.join(timeout=2)
    second.join(timeout=2)

    assert len(results) == 2
    assert client._fetch_prompt.call_count == 1
    assert results[0] is results[1]
    client.close()


def test_prompt_api_error_does_not_copy_response_content():
    response = Mock(status_code=400)
    response.text = 'invalid prompt: "private system prompt"'
    session = Mock()
    session.request.return_value = response
    client = PromptClient(base_url="https://example.test", api_key="test", session=session)

    with pytest.raises(PromptApiError) as captured:
        client.list_prompts()

    assert "private system prompt" not in str(captured.value)
    assert str(captured.value) == "GET /api/managed-prompts failed (400)"


def test_prompt_transport_error_does_not_copy_network_details():
    session = Mock()
    session.request.side_effect = requests.ConnectionError(
        "failed https://example.test/api/v1/prompts/private-name/fetch"
    )
    client = PromptClient(base_url="https://example.test", api_key="test", session=session)

    with pytest.raises(PromptApiError) as captured:
        client.list_prompts()

    assert "example.test" not in str(captured.value)
    assert str(captured.value) == "GET /api/managed-prompts request failed"


def test_prompt_transport_error_does_not_copy_prompt_selectors():
    session = Mock()
    session.request.side_effect = requests.ConnectionError("private prompt leaked")
    client = PromptClient(base_url="https://example.test", api_key="test", session=session)

    with pytest.raises(PromptApiError) as captured:
        client.fetch_prompt("private-customer-name", label="production")

    assert str(captured.value) == "GET /api/v1/prompts/:name/fetch request failed"
    assert "customer" not in str(captured.value)
    assert "production" not in str(captured.value)


@pytest.mark.asyncio
async def test_async_prompt_transport_error_does_not_copy_network_details():
    client = AsyncPromptClient(base_url="https://example.test", api_key="test")
    client._client.request = AsyncMock(
        side_effect=httpx.ConnectError(
            "failed https://example.test/private",
            request=httpx.Request("GET", "https://example.test/private"),
        )
    )
    try:
        with pytest.raises(PromptApiError) as captured:
            await client.get_prompt("private-name")

        assert "example.test" not in str(captured.value)
        assert str(captured.value) == "GET /api/managed-prompts request failed"
    finally:
        await client.close()


def test_prompt_crud_invalidates_mutable_selectors_and_deleted_versions():
    client = PromptClient(base_url="https://example.test", api_key="test")
    client._cache.set(_key("assistant"), _prompt(1, "latest"))
    client._cache.set(_key("assistant", label="production"), _prompt(1, "production"))
    client._cache.set(_key("assistant", version=1), _prompt(1, "pinned"))
    client._request_json = Mock(
        return_value={
            "prompt": {
                "id": "prompt-2",
                "name": "assistant",
                "version": 2,
                "content": "new",
            }
        }
    )

    client.create_prompt(name="assistant", prompt="new", labels=["production"])
    assert client._cache.get(_key("assistant")) is None
    assert client._cache.get(_key("assistant", label="production")) is None
    assert client._cache.get(_key("assistant", version=1)) is not None

    client._cache.set(_key("assistant"), _prompt(2, "new"))
    client.list_prompts = Mock(
        return_value={"items": [{"id": "prompt-1", "name": "assistant", "version": 1}]}
    )
    client.delete_prompt("assistant", 1)
    assert client._cache.get(_key("assistant")) is None
    assert client._cache.get(_key("assistant", version=1)) is None


def test_completed_stale_refresh_cannot_repopulate_cache_after_prompt_write():
    client = PromptClient(base_url="https://example.test", api_key="test")
    client._cache.set(_key("assistant"), _prompt(1, "old"), ttl=0)
    refresh_started = threading.Event()
    release_refresh = threading.Event()

    def fetch(*_args, **_kwargs):
        refresh_started.set()
        assert release_refresh.wait(timeout=2)
        return _prompt(1, "stale refresh")

    client._fetch_prompt = Mock(side_effect=fetch)
    assert client.get_prompt("assistant").content == "old"
    assert refresh_started.wait(timeout=2)
    client._request_json = Mock(return_value={"prompt": _raw_prompt(2, "new")})
    client.create_prompt(name="assistant", prompt="new", labels=["production"])

    release_refresh.set()
    for _ in range(100):
        with client._refresh_threads_lock:
            if not client._refresh_threads:
                break
        threading.Event().wait(0.01)
    else:
        pytest.fail("background refresh did not finish")
    assert client._cache.get(_key("assistant")) is None
    client.close()


def test_sync_pinned_version_lookup_pages_beyond_first_500_results():
    client = PromptClient(base_url="https://example.test", api_key="test")
    first_page = [_raw_prompt(version) for version in range(700, 200, -1)]
    client.list_prompts = Mock(
        side_effect=[
            {"items": first_page, "limit": 500, "offset": 0, "total": 501},
            {"items": [_raw_prompt(200, "target")], "limit": 500, "offset": 500, "total": 501},
        ]
    )

    assert client.get_prompt("assistant", version=200).content == "target"
    assert client.list_prompts.call_count == 2


def test_prompt_mutation_resolves_versions_beyond_first_500_results():
    client = PromptClient(base_url="https://example.test", api_key="test")
    first_page = [_raw_prompt(version) for version in range(700, 200, -1)]
    client.list_prompts = Mock(
        side_effect=[
            {"items": first_page, "limit": 500, "offset": 0, "total": 501},
            {"items": [_raw_prompt(200)], "limit": 500, "offset": 500, "total": 501},
        ]
    )
    client._request_json = Mock(return_value={"deletedAt": "2026-09-03T00:00:00Z"})

    client.delete_prompt("assistant", 200)

    assert client.list_prompts.call_count == 2
    client._request_json.assert_called_once_with(
        method="DELETE", path="/api/managed-prompts/prompt-200"
    )


def test_prompt_writes_match_backend_routes_and_field_names():
    client = PromptClient(base_url="https://example.test", api_key="test")
    client._request_json = Mock(
        side_effect=[
            {"prompt": _raw_prompt(1, "hello")},
            {"prompt": _raw_prompt(2, "updated")},
        ]
    )

    client.create_prompt(
        name="assistant",
        prompt="hello",
        labels=["production"],
        commit_message="initial",
    )
    client.save_as_version(
        prompt_name="assistant",
        content="updated",
        labels=["production"],
        commit_message="revision",
    )

    assert client._request_json.call_args_list == [
        call(
            method="POST",
            path="/api/managed-prompts",
            json_body={
                "name": "assistant",
                "content": "hello",
                "labels": ["production"],
                "commit_message": "initial",
            },
        ),
        call(
            method="POST",
            path="/api/prompt-playground/save-as-version",
            json_body={
                "promptName": "assistant",
                "content": "updated",
                "commitMessage": "revision",
                "labels": ["production"],
            },
        ),
    ]


def test_label_and_tag_mutations_use_resolved_uuid_routes():
    client = PromptClient(base_url="https://example.test", api_key="test")
    client.list_prompts = Mock(
        return_value={"items": [{"id": "prompt-1", "name": "assistant", "version": 1}]}
    )
    client._request_json = Mock(return_value={})

    client.update_prompt(name="assistant", version=1, new_labels=["production"])
    client.remove_tag("assistant", 1, "release")

    assert client._request_json.call_args_list == [
        call(
            method="POST",
            path="/api/managed-prompts/prompt-1/labels",
            json_body={"label": "production"},
        ),
        call(
            method="DELETE",
            path="/api/managed-prompts/prompt-1/tags",
            json_body={"tag": "release"},
        ),
    ]


def test_save_as_version_rejects_empty_content_before_network():
    client = PromptClient(base_url="https://example.test", api_key="test")
    client._request_json = Mock()

    with pytest.raises(ValueError, match="requires non-empty content or messages"):
        client.save_as_version(prompt_name="assistant", content="   ")

    client._request_json.assert_not_called()


def test_prompt_writes_validate_labels_and_tags_before_network():
    client = PromptClient(base_url="https://example.test", api_key="test")
    client._request_json = Mock()

    with pytest.raises(ValueError, match="at most one"):
        client.create_prompt(
            name="assistant",
            prompt="hello",
            labels=["production", "staging"],
        )
    with pytest.raises(ValueError, match="at most one"):
        client.save_as_version(
            prompt_name="assistant",
            content="hello",
            labels=["production", "staging"],
        )
    with pytest.raises(ValueError, match="1-50"):
        client.create_prompt(name="assistant", prompt="hello", labels=["  "])
    with pytest.raises(ValueError, match="1-50"):
        client.update_prompt(name="assistant", version=1, new_labels=[""])
    with pytest.raises(ValueError, match="1-50"):
        client.save_as_version(prompt_name="assistant", content="hello", labels=["\t"])
    with pytest.raises(ValueError, match="1-64"):
        client.remove_tag("assistant", 1, " unsafe ")
    with pytest.raises(ValueError, match="1-64"):
        client.save_as_version(prompt_name="assistant", content="hello", tags=["line\nbreak"])

    client._request_json.assert_not_called()


def test_create_prompt_allows_an_unlabeled_version():
    client = PromptClient(base_url="https://example.test", api_key="test")
    client._request_json = Mock(return_value={"prompt": _raw_prompt(1, "hello")})

    client.create_prompt(name="assistant", prompt="hello")

    assert client._request_json.call_args == call(
        method="POST",
        path="/api/managed-prompts",
        json_body={"name": "assistant", "content": "hello", "labels": []},
    )


def test_shared_sync_client_rotates_credentials_and_shutdown_detaches_it(monkeypatch):
    old_session = Mock()
    old = PromptClient(base_url="https://project-a.test", api_key="project-a", session=old_session)
    monkeypatch.setattr(prompt_client_module, "_shared_client", old)
    monkeypatch.setitem(init_module._session_config, "_api_key", "project-b")
    monkeypatch.setitem(init_module._session_config, "_base_url", "https://project-b.test")

    current = prompt_client_module._get_shared_client()

    assert current is not old
    assert current.api_key == "project-b"
    old_session.close.assert_called_once()
    prompt_client_module._close_shared_prompt_clients()
    assert prompt_client_module._shared_client is None


@pytest.mark.asyncio
async def test_async_stale_refresh_is_coalesced_and_preserves_entry_ttl():
    client = AsyncPromptClient(
        base_url="https://example.test", api_key="test", cache_ttl_seconds=60
    )
    initial = _prompt(1, "old")
    refreshed = _prompt(2, "new")
    refresh_started = asyncio.Event()
    release_refresh = asyncio.Event()

    async def fetch(*_args, **_kwargs):
        if not client._cache.get(_key("assistant")):
            return initial
        refresh_started.set()
        await asyncio.wait_for(release_refresh.wait(), timeout=2)
        return refreshed

    client._fetch_prompt = AsyncMock(side_effect=fetch)
    try:
        assert (await client.get_prompt("assistant", cache_ttl_seconds=7)).content == "old"
        _expire(client, _key("assistant"))

        assert (await client.get_prompt("assistant")).content == "old"
        await asyncio.wait_for(refresh_started.wait(), timeout=2)
        assert (await client.get_prompt("assistant")).content == "old"
        assert client._fetch_prompt.await_count == 2

        release_refresh.set()
        for _ in range(100):
            entry = client._cache.get(_key("assistant"))
            if entry is not None and entry.value.content == "new":
                break
            await asyncio.sleep(0.01)
        else:
            pytest.fail("background async prompt refresh did not finish")

        entry = client._cache.get(_key("assistant"))
        assert entry is not None
        assert entry.ttl_seconds == 7
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_async_pinned_version_is_not_refreshed_after_ttl():
    client = AsyncPromptClient(base_url="https://example.test", api_key="test", cache_ttl_seconds=1)
    client._fetch_prompt = AsyncMock(return_value=_prompt(3, "pinned"))
    try:
        first = await client.get_prompt("assistant", version=3)
        _expire(client, _key("assistant", version=3))
        second = await client.get_prompt("assistant", version=3)

        assert second is first
        client._fetch_prompt.assert_awaited_once()
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_async_refresh_tasks_are_bounded_and_cancelled_on_close():
    client = AsyncPromptClient(
        base_url="https://example.test",
        api_key="test",
        max_refresh_workers=1,
    )
    client._cache.set(_key("alpha"), _prompt(1, "alpha"), ttl=0)
    client._cache.set(_key("beta"), _prompt(1, "beta"), ttl=0)
    refresh_started = asyncio.Event()
    release_refresh = asyncio.Event()

    async def fetch(*_args, **_kwargs):
        refresh_started.set()
        await release_refresh.wait()
        return _prompt(2, "refreshed")

    client._fetch_prompt = AsyncMock(side_effect=fetch)
    assert (await client.get_prompt("alpha")).content == "alpha"
    await asyncio.wait_for(refresh_started.wait(), timeout=2)
    assert (await client.get_prompt("beta")).content == "beta"
    assert client._fetch_prompt.await_count == 1

    await client.close()
    assert not client._refresh_tasks
    with pytest.raises(PromptClientClosedError):
        await client.get_prompt("alpha")


@pytest.mark.asyncio
async def test_async_cold_misses_are_coalesced_and_close_cannot_repopulate_cache():
    client = AsyncPromptClient(base_url="https://example.test", api_key="test")
    fetch_started = asyncio.Event()
    release_fetch = asyncio.Event()

    async def fetch(*_args, **_kwargs):
        fetch_started.set()
        await release_fetch.wait()
        return _prompt(1, "shared")

    client._fetch_prompt = AsyncMock(side_effect=fetch)
    first = asyncio.create_task(client.get_prompt("assistant"))
    await asyncio.wait_for(fetch_started.wait(), timeout=2)
    second = asyncio.create_task(client.get_prompt("assistant"))
    await asyncio.sleep(0)
    await client.close()
    release_fetch.set()

    results = await asyncio.gather(first, second, return_exceptions=True)
    assert all(isinstance(result, asyncio.CancelledError) for result in results)
    assert client._fetch_prompt.await_count == 1
    assert client._cache.get(_key("assistant")) is None


@pytest.mark.asyncio
async def test_shared_async_client_rotates_credentials(monkeypatch):
    old = AsyncPromptClient(base_url="https://project-a.test", api_key="project-a")
    monkeypatch.setattr(prompt_client_module, "_shared_async_client", old)
    monkeypatch.setitem(init_module._session_config, "_api_key", "project-b")
    monkeypatch.setitem(init_module._session_config, "_base_url", "https://project-b.test")

    current = await prompt_client_module._get_shared_async_client()

    assert current is not old
    assert current.api_key == "project-b"
    assert old._transport_closed
    await current.close()
    monkeypatch.setattr(prompt_client_module, "_shared_async_client", None)


@pytest.mark.asyncio
async def test_async_pinned_version_lookup_pages_beyond_first_500_results():
    client = AsyncPromptClient(base_url="https://example.test", api_key="test")
    first_page = [_raw_prompt(version) for version in range(700, 200, -1)]
    client._request_json = AsyncMock(
        side_effect=[
            {"items": first_page, "limit": 500, "offset": 0, "total": 501},
            {"items": [_raw_prompt(200, "target")], "limit": 500, "offset": 500, "total": 501},
        ]
    )
    try:
        assert (await client.get_prompt("assistant", version=200)).content == "target"
        assert client._request_json.await_count == 2
    finally:
        await client.close()
