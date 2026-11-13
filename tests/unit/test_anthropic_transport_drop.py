import pytest
from test_anthropic_managed_stream import (
    KW,
    _assert_error,
    _body,
    _span,
    anthropic,
    exporter,
    httpx,
    wrap_anthropic_client,
    wrap_async_anthropic_client,
)


class Drop(httpx.SyncByteStream):
    def __iter__(self):
        yield _body(False).split("event: message_delta")[0].encode()
        raise httpx.ReadError("connection dropped mid-reply")


class AsyncDrop(httpx.AsyncByteStream):
    async def __aiter__(self):
        yield _body(False).split("event: message_delta")[0].encode()
        raise httpx.ReadError("connection dropped mid-reply")


@pytest.mark.parametrize("mode", ["iter", "text_stream", "final_message"])
def test_drop_sync(exporter, mode):
    c = wrap_anthropic_client(
        anthropic.Anthropic(
            api_key="dummy",
            max_retries=0,
            http_client=httpx.Client(
                transport=httpx.MockTransport(
                    lambda r: httpx.Response(
                        200, headers={"content-type": "text/event-stream"}, stream=Drop()
                    )
                )
            ),
        )
    )
    with pytest.raises(httpx.ReadError):
        with c.messages.stream(**KW) as s:
            if mode == "iter":
                list(s)
            elif mode == "text_stream":
                list(s.text_stream)
            else:
                s.get_final_message()
    _assert_error(_span(exporter))


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["iter", "text_stream", "final_message"])
async def test_drop_async(exporter, mode):
    c = wrap_async_anthropic_client(
        anthropic.AsyncAnthropic(
            api_key="dummy",
            max_retries=0,
            http_client=httpx.AsyncClient(
                transport=httpx.MockTransport(
                    lambda r: httpx.Response(
                        200, headers={"content-type": "text/event-stream"}, stream=AsyncDrop()
                    )
                )
            ),
        )
    )
    with pytest.raises(httpx.ReadError):
        async with c.messages.stream(**KW) as s:
            if mode == "iter":
                async for _ in s:
                    pass
            elif mode == "text_stream":
                async for _ in s.text_stream:
                    pass
            else:
                await s.get_final_message()
    _assert_error(_span(exporter))
