from types import SimpleNamespace

import pytest

import neatlogs.bedrock as integration


@pytest.mark.parametrize(
    "method,key", [("converse_stream", "stream"), ("invoke_model_with_response_stream", "body")]
)
@pytest.mark.parametrize("phase", ["unread", "partial", "exhausted"])
@pytest.mark.parametrize("close_error", [False, True])
@pytest.mark.parametrize("wrapped", [False, True])
def test_native_event_stream_close_reaches_body(
    monkeypatch, tracer_provider, in_memory_span_exporter, method, key, phase, wrapped, close_error
):
    eventstream = pytest.importorskip("botocore.eventstream")

    failure = OSError("close failed")

    class Raw:
        closed = False
        closes = 0

        def close(self):
            self.closed = True
            self.closes += 1
            if close_error:
                raise failure

        def stream(self):
            return iter(())

    raw = Raw()
    event = (
        {"contentBlockDelta": {"delta": {"text": "hi"}}}
        if key == "stream"
        else {"chunk": {"bytes": b'{"generation":"hi"}'}}
    )
    parser = SimpleNamespace(parse=lambda response, shape: event)
    native = eventstream.EventStream(raw, None, parser, "offline")
    native._event_generator = iter(
        [
            SimpleNamespace(to_response_dict=lambda: {"status_code": 200}),
            SimpleNamespace(to_response_dict=lambda: {"status_code": 200}),
        ]
    )
    monkeypatch.setattr(
        integration, "get_provider_tracer", lambda: tracer_provider.get_tracer("test")
    )
    client = SimpleNamespace(**{method: lambda **kwargs: {key: native}})
    if wrapped:
        integration.wrap_bedrock_client(client)
    response = getattr(client, method)(modelId="anthropic.test", messages=[])
    iterator = iter(response[key])
    if phase == "partial":
        assert next(iterator) == event
    elif phase == "exhausted":
        assert list(iterator) == [event, event]
    if close_error:
        with pytest.raises(OSError) as caught:
            response[key].close()
        assert caught.value is failure
    else:
        response[key].close()
    assert raw.closed
    assert raw.closes == 1
    if wrapped:
        response[key].close()
        assert raw.closes == 1
        assert list(response[key]) == []
        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) == 1
        if phase != "unread":
            assert spans[0].attributes["neatlogs.llm.output_messages.0.content"] == (
                "hi" if phase == "partial" else "hihi"
            )
