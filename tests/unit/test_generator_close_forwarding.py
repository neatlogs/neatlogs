import pytest

import neatlogs
from neatlogs._wrap_utils import set_neatlogs_provider


@pytest.mark.asyncio
async def test_async_generator_close_releases_inner_resource(tracer_provider):
    set_neatlogs_provider(tracer_provider)
    closed = []

    @neatlogs.span(kind="CHAIN")
    async def stream():
        try:
            yield "first"
            yield "second"
        finally:
            closed.append(True)

    result = stream()
    assert await anext(result) == "first"
    await result.aclose()
    assert closed == [True]


@pytest.mark.asyncio
async def test_async_generator_close_direct_control():
    closed = []

    async def stream():
        try:
            yield "first"
        finally:
            closed.append(True)

    result = stream()
    assert await anext(result) == "first"
    await result.aclose()
    assert closed == [True]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["complete", "error"])
async def test_async_generator_cleanup_on_normal_end_or_error(tracer_provider, mode):
    set_neatlogs_provider(tracer_provider)
    closed = []

    @neatlogs.span(kind="CHAIN")
    async def stream():
        try:
            yield "first"
            if mode == "error":
                raise ValueError("provider failure")
        finally:
            closed.append(True)

    if mode == "error":
        with pytest.raises(ValueError, match="provider failure"):
            _ = [item async for item in stream()]
    else:
        assert [item async for item in stream()] == ["first"]
    assert closed == [True]


@pytest.mark.asyncio
async def test_async_generator_close_smoke(tracer_provider):
    set_neatlogs_provider(tracer_provider)
    cleaned = 0
    controls = 0
    for index in range(120):
        closed = []

        async def raw():
            try:
                yield "first"
                yield "second"
            finally:
                closed.append(True)

        wrapped = neatlogs.span(kind="CHAIN")(raw)
        iterator = wrapped()
        assert await anext(iterator) == "first"
        if index % 2 == 0:
            await iterator.aclose()
            assert closed == [True]
            cleaned += 1
        else:
            assert [item async for item in iterator] == ["second"]
            assert closed == [True]
            controls += 1
    assert (cleaned, controls) == (60, 60)
