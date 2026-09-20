import asyncio
from types import SimpleNamespace

import pytest

from neatlogs import openai as instrumentation


class _Span:
    def __init__(self, name, attributes):
        self.name = name
        self.attributes = dict(attributes)
        self.status = SimpleNamespace(name="UNSET")
        self.ended = False
        self.exceptions = []

    def set_attribute(self, key, value):
        self.attributes[key] = value

    def set_status(self, status, *args):
        self.status = status

    def record_exception(self, exc):
        self.exceptions.append(exc)

    def end(self):
        self.ended = True


class _Tracer:
    def __init__(self):
        self.spans = []

    def start_span(self, name, attributes):
        span = _Span(name, attributes)
        self.spans.append(span)
        return span


class _Completions:
    async def create(self, *args, **kwargs):
        await asyncio.Future()


class _Responses:
    async def create(self, *args, **kwargs):
        await asyncio.Future()


@pytest.fixture
def tracer(monkeypatch):
    tracer = _Tracer()
    monkeypatch.setattr(instrumentation, "get_provider_tracer", lambda: tracer)
    monkeypatch.setattr(instrumentation, "is_suppressed", lambda: False)
    return tracer


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("resource", "patch", "kwargs", "name"),
    [
        (
            _Completions,
            instrumentation._patch_async_completions,
            {"model": "demo", "messages": []},
            "openai.chat.completions.create",
        ),
        (
            _Responses,
            instrumentation._patch_async_responses,
            {"model": "demo", "input": "hi"},
            "openai.responses.create",
        ),
    ],
)
async def test_openai_async_cancellation_ends_span(tracer, resource, patch, kwargs, name):
    target = resource()
    patch(target)
    task = asyncio.create_task(target.create(**kwargs))
    await asyncio.sleep(0)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert len(tracer.spans) == 1
    assert tracer.spans[0].name == name
    assert tracer.spans[0].ended is True
    assert tracer.spans[0].status.name == "ERROR"
    assert isinstance(tracer.spans[0].exceptions[0], asyncio.CancelledError)
