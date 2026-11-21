import asyncio

import pytest

import neatlogs
from neatlogs._wrap_utils import (
    _wrap_context,
    make_wrap_context,
    set_neatlogs_provider,
    with_wrap_context,
)


def _sdk_client(provider, asynchronous, handler):
    httpx = pytest.importorskip("httpx")
    sdk = pytest.importorskip(provider)
    cls = getattr(
        sdk, ("Async" if asynchronous else "") + ("OpenAI" if provider == "openai" else "Anthropic")
    )
    http_cls = httpx.AsyncClient if asynchronous else httpx.Client
    return cls(
        api_key="synthetic-not-a-key", http_client=http_cls(transport=httpx.MockTransport(handler))
    )


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
@pytest.mark.parametrize("metadata", [False, True])
@pytest.mark.parametrize("asynchronous", [False, True])
def test_real_client_context_manager(provider, metadata, asynchronous):
    requests = []

    def handler(request):
        requests.append(request)
        raise AssertionError("context entry must not make a provider request")

    client = _sdk_client(provider, asynchronous, handler)
    wrapped = neatlogs.wrap(client, **({"team": "context-test"} if metadata else {}))
    failure = ValueError("body failed")

    async def run_async():
        try:
            with pytest.raises(ValueError) as caught:
                async with wrapped as entered:
                    assert entered is wrapped
                    assert not client.is_closed()
                    raise failure
            assert caught.value is failure
            assert client.is_closed()
        finally:
            await client.close()

    def run_sync():
        try:
            with pytest.raises(ValueError) as caught:
                with wrapped as entered:
                    assert entered is wrapped
                    assert not client.is_closed()
                    raise failure
            assert caught.value is failure
            assert client.is_closed()
        finally:
            client.close()

    asyncio.run(run_async()) if asynchronous else run_sync()
    assert requests == []


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
@pytest.mark.parametrize("asynchronous", [False, True])
def test_real_client_keeps_workflow_metadata(
    provider, asynchronous, tracer_provider, in_memory_span_exporter
):
    httpx = pytest.importorskip("httpx")
    set_neatlogs_provider(tracer_provider)
    requests = []

    def handler(request):
        requests.append(request)
        if provider == "openai":
            data = {
                "id": "chatcmpl_test",
                "object": "chat.completion",
                "created": 0,
                "model": "gpt-test",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        else:
            data = {
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "model": "claude-test",
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }
        return httpx.Response(200, json=data)

    def call(client):
        if provider == "openai":
            return client.chat.completions.create(
                model="gpt-test", messages=[{"role": "user", "content": "hi"}]
            )
        return client.messages.create(
            model="claude-test", max_tokens=8, messages=[{"role": "user", "content": "hi"}]
        )

    async def run_async():
        async with neatlogs.wrap(
            _sdk_client(provider, True, handler), team="context-test"
        ) as wrapped:
            await call(wrapped)

    def run_sync():
        with neatlogs.wrap(_sdk_client(provider, False, handler), team="context-test") as wrapped:
            call(wrapped)

    try:
        asyncio.run(run_async()) if asynchronous else run_sync()
        roots = [s for s in in_memory_span_exporter.get_finished_spans() if s.parent is None]
        assert len(roots) == 1
        assert roots[0].attributes["neatlogs.workflow.team"] == "context-test"
        assert len(requests) == 1
        assert _wrap_context.get() is None
    finally:
        set_neatlogs_provider(None)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("failure_at", ["enter", "exit", None])
def test_context_failure_restores_outer_metadata(asynchronous, failure_at):
    failure = RuntimeError("context failed")
    seen = []

    class Target:
        def enter(self):
            seen.append(_wrap_context.get())
            if failure_at == "enter":
                raise failure
            return self

        def exit(self, *args):
            seen.append(_wrap_context.get())
            if failure_at == "exit":
                raise failure
            return False

        __enter__ = enter
        __exit__ = exit

        async def __aenter__(self):
            return self.enter()

        async def __aexit__(self, *args):
            return self.exit(*args)

    wrapped = with_wrap_context(Target(), make_wrap_context({"team": "inner"}))
    outer = {"workflow": {"outer": "unchanged"}}

    async def run_async():
        async with wrapped:
            assert _wrap_context.get() is outer

    def run_sync():
        with wrapped:
            assert _wrap_context.get() is outer

    token = _wrap_context.set(outer)
    try:
        if failure_at:
            with pytest.raises(RuntimeError) as caught:
                asyncio.run(run_async()) if asynchronous else run_sync()
            assert caught.value is failure
        else:
            asyncio.run(run_async()) if asynchronous else run_sync()
        assert _wrap_context.get() is outer
        assert all(item == {"workflow": {"outer": "unchanged", "team": "inner"}} for item in seen)
    finally:
        _wrap_context.reset(token)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("suppress", [False, True])
def test_context_preserves_other_entry_value_and_suppression(asynchronous, suppress):
    result = object()
    failure = ValueError("body failed")
    seen = []

    class Target:
        def __enter__(self):
            return result

        def __exit__(self, *args):
            seen.append(args)
            return suppress

        async def __aenter__(self):
            return result

        async def __aexit__(self, *args):
            seen.append(args)
            return suppress

    wrapped = with_wrap_context(Target(), make_wrap_context({"team": "test"}))

    async def run_async():
        async with wrapped as entered:
            assert entered is result
            raise failure

    def run_sync():
        with wrapped as entered:
            assert entered is result
            raise failure

    if suppress:
        asyncio.run(run_async()) if asynchronous else run_sync()
    else:
        with pytest.raises(ValueError) as caught:
            asyncio.run(run_async()) if asynchronous else run_sync()
        assert caught.value is failure
    assert len(seen) == 1
    assert seen[0][0] is ValueError
    assert seen[0][1] is failure
    assert seen[0][2] is not None
    assert _wrap_context.get() is None


def test_cancelled_async_context_entry_restores_metadata():
    failure = asyncio.CancelledError("cancel entry")

    class Target:
        async def __aenter__(self):
            raise failure

        async def __aexit__(self, *args):
            pytest.fail("exit must not be called after failed entry")

    wrapped = with_wrap_context(Target(), make_wrap_context({"team": "test"}))

    async def run():
        with pytest.raises(asyncio.CancelledError) as caught:
            async with wrapped:
                pytest.fail("entry must fail")
        assert caught.value is failure
        assert _wrap_context.get() is None

    asyncio.run(run())


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_concurrent_async_clients_keep_metadata_separate(
    provider, tracer_provider, in_memory_span_exporter
):
    httpx = pytest.importorskip("httpx")
    set_neatlogs_provider(tracer_provider)

    async def handler(request):
        await asyncio.sleep(0)
        if provider == "openai":
            data = {
                "id": "chatcmpl_test",
                "object": "chat.completion",
                "created": 0,
                "model": "gpt-test",
                "choices": [],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        else:
            data = {
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "model": "claude-test",
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }
        return httpx.Response(200, json=data)

    async def run(team):
        async with neatlogs.wrap(_sdk_client(provider, True, handler), team=team) as wrapped:
            if provider == "openai":
                await wrapped.chat.completions.create(
                    model="gpt-test", messages=[{"role": "user", "content": "hi"}]
                )
            else:
                await wrapped.messages.create(
                    model="claude-test", max_tokens=8, messages=[{"role": "user", "content": "hi"}]
                )
            assert _wrap_context.get() is None

    async def main():
        await asyncio.gather(run("first"), run("second"))
        assert _wrap_context.get() is None

    try:
        asyncio.run(main())
        roots = [s for s in in_memory_span_exporter.get_finished_spans() if s.parent is None]
        assert sorted(s.attributes["neatlogs.workflow.team"] for s in roots) == ["first", "second"]
    finally:
        set_neatlogs_provider(None)
