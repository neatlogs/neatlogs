import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from neatlogs._wrap_utils import set_neatlogs_provider
from neatlogs.decorators.orchestration import span


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_mappers_project_before_capture_and_preserve_business_result(
    asynchronous, tracer_provider, in_memory_span_exporter
):
    set_neatlogs_provider(tracer_provider)
    account = SimpleNamespace(id=7, api_key="must-not-capture")
    result = SimpleNamespace(allowed=False, count=0, token="must-not-capture")
    calls = []

    async def async_action(account):
        calls.append(account)
        return result

    def sync_action(account):
        calls.append(account)
        return result

    action = span(
        kind="CHAIN",
        name="projected",
        input_mapper=lambda args: {"account_id": args["account"].id},
        output_mapper=lambda value: {"allowed": value.allowed, "count": value.count},
    )(async_action if asynchronous else sync_action)
    actual = await action(account) if asynchronous else action(account)
    assert actual is result
    assert calls == [account]
    captured = next(
        item for item in in_memory_span_exporter.get_finished_spans() if item.name == "projected"
    )
    assert json.loads(captured.attributes["input.value"]) == {"account_id": 7}
    assert json.loads(captured.attributes["output.value"]) == {"allowed": False, "count": 0}
    assert "must-not-capture" not in str(captured.attributes)


def test_mapper_failure_omits_raw_data_and_preserves_result(
    tracer_provider, in_memory_span_exporter
):
    set_neatlogs_provider(tracer_provider)
    mapper = Mock(side_effect=ValueError("projection failed"))

    @span(kind="WORKFLOW", name="failed-projection", input_mapper=mapper, output_mapper=mapper)
    def action(secret):
        return secret

    assert action("must-not-capture") == "must-not-capture"
    captured = next(
        item
        for item in in_memory_span_exporter.get_finished_spans()
        if item.name == "failed-projection"
    )
    assert "input.value" not in captured.attributes
    assert "output.value" not in captured.attributes
    assert len(captured.events) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_stream_mappers_preserve_original_chunks(
    asynchronous, tracer_provider, in_memory_span_exporter
):
    set_neatlogs_provider(tracer_provider)

    async def async_chunks():
        yield False
        yield 0

    def sync_chunks():
        yield False
        yield 0

    action = span(
        kind="CHAIN", name="stream-projection", output_mapper=lambda chunks: {"values": chunks}
    )(async_chunks if asynchronous else sync_chunks)
    values = [value async for value in action()] if asynchronous else list(action())
    assert values == [False, 0]
    captured = next(
        item
        for item in in_memory_span_exporter.get_finished_spans()
        if item.name == "stream-projection"
    )
    assert json.loads(captured.attributes["output.value"]) == {"values": [False, 0]}
