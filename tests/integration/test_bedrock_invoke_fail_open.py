"""Regression tests for Bedrock invoke_model + invoke_model_with_response_stream fail-open."""

import io
import json
from types import SimpleNamespace
from unittest.mock import patch

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor


def _setup_tracer(exporter):
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    import neatlogs._wrap_utils as _wu

    _wu._wrapper_tracer = None
    return provider


def _fake_bedrock_client(**methods):
    service_model = SimpleNamespace(service_name="bedrock-runtime")
    meta = SimpleNamespace(service_model=service_model)
    return SimpleNamespace(meta=meta, **methods)


class TestBedrockInvokeFailOpen:
    def test_invoke_model_fails_open_on_tracer_error(self, in_memory_span_exporter):
        from neatlogs.bedrock import wrap_bedrock_client

        _setup_tracer(in_memory_span_exporter)

        body_out = json.dumps(
            {
                "content": [{"type": "text", "text": "hello"}],
                "usage": {"input_tokens": 4, "output_tokens": 2},
                "stop_reason": "end_turn",
            }
        ).encode()

        def invoke_model(**kwargs):
            return {"body": io.BytesIO(body_out)}

        client = _fake_bedrock_client(invoke_model=invoke_model)
        wrap_bedrock_client(client)

        with patch(
            "neatlogs.bedrock.get_provider_tracer",
            side_effect=RuntimeError("trace failed"),
        ):
            response = client.invoke_model(
                modelId="anthropic.claude-3-haiku-20240307-v1:0",
                body=json.dumps({"messages": [{"role": "user", "content": "hi"}]}),
            )

        assert json.loads(response["body"].read())["content"][0]["text"] == "hello"
        assert len(in_memory_span_exporter.get_finished_spans()) == 0

    def test_invoke_model_stream_fails_open_on_tracer_error(self, in_memory_span_exporter):
        from neatlogs.bedrock import wrap_bedrock_client

        _setup_tracer(in_memory_span_exporter)

        class _FakeStream:
            def __iter__(self_inner):
                return iter([])

        def invoke_model_with_response_stream(**kwargs):
            return {"body": _FakeStream()}

        client = _fake_bedrock_client(
            invoke_model_with_response_stream=invoke_model_with_response_stream
        )
        wrap_bedrock_client(client)

        with patch(
            "neatlogs.bedrock.get_provider_tracer",
            side_effect=RuntimeError("trace failed"),
        ):
            response = client.invoke_model_with_response_stream(
                modelId="anthropic.claude-3-haiku-20240307-v1:0",
                body=json.dumps({"messages": [{"role": "user", "content": "hi"}]}),
            )

        assert isinstance(response["body"], _FakeStream)
        assert len(in_memory_span_exporter.get_finished_spans()) == 0

    def test_invoke_model_embedding_path_exact_call_once(self, in_memory_span_exporter):
        """Embedding models create the span via a different path (direct start_span).
        Assert: when telemetry setup fails, the original invoke_model is called
        exactly once, response flows through unchanged, no leaked span."""
        from neatlogs.bedrock import wrap_bedrock_client

        _setup_tracer(in_memory_span_exporter)

        body_out = json.dumps(
            {
                "embedding": [0.1, 0.2, 0.3],
                "inputTextTokenCount": 5,
            }
        ).encode()

        call_count = {"n": 0}

        def invoke_model(**kwargs):
            call_count["n"] += 1
            return {"body": io.BytesIO(body_out)}

        client = _fake_bedrock_client(invoke_model=invoke_model)
        wrap_bedrock_client(client)

        with patch(
            "neatlogs.bedrock.get_provider_tracer",
            side_effect=RuntimeError("trace failed"),
        ):
            response = client.invoke_model(
                modelId="amazon.titan-embed-text-v1",
                body=json.dumps({"inputText": "hi"}),
            )

        assert call_count["n"] == 1, f"called {call_count['n']} times, expected 1"
        body = json.loads(response["body"].read())
        assert body["embedding"] == [0.1, 0.2, 0.3]
        assert len(in_memory_span_exporter.get_finished_spans()) == 0

    def test_invoke_model_set_attribute_fails_exact_call_once(self, in_memory_span_exporter):
        """When set_attribute raises after span creation, the partial span is
        ended and the original invoke_model is called exactly once."""
        from neatlogs.bedrock import wrap_bedrock_client

        _setup_tracer(in_memory_span_exporter)

        body_out = json.dumps(
            {
                "content": [{"type": "text", "text": "hello"}],
                "usage": {"input_tokens": 4, "output_tokens": 2},
                "stop_reason": "end_turn",
            }
        ).encode()

        call_count = {"n": 0}

        def invoke_model(**kwargs):
            call_count["n"] += 1
            return {"body": io.BytesIO(body_out)}

        client = _fake_bedrock_client(invoke_model=invoke_model)
        wrap_bedrock_client(client)

        # Patch _set_invoke_input to raise AFTER span creation. This exercises
        # the partial-span cleanup path: span is open, but input-attribute
        # recording fails.
        with patch(
            "neatlogs.bedrock._set_invoke_input",
            side_effect=RuntimeError("set_attribute failed"),
        ):
            response = client.invoke_model(
                modelId="anthropic.claude-3-haiku-20240307-v1:0",
                body=json.dumps({"messages": [{"role": "user", "content": "hi"}]}),
            )

        assert call_count["n"] == 1, f"called {call_count['n']} times, expected 1"
        assert json.loads(response["body"].read())["content"][0]["text"] == "hello"
        # The partial span was ended before the SDK call. There may be a
        # second housekeeping span from the SDK setup path; what matters is
        # that the LLM span (the partial one) has no input_messages.
        spans = in_memory_span_exporter.get_finished_spans()
        llm_spans = [s for s in spans if s.attributes.get("neatlogs.span.kind") == "llm"]
        assert len(llm_spans) == 1, f"expected 1 LLM span, got {len(llm_spans)}"
        # The partial LLM span has no input_messages attributes recorded
        # (since _set_invoke_input failed before recording them).
        assert not any(a.startswith("neatlogs.llm.input_messages") for a in llm_spans[0].attributes)

    def test_invoke_model_ok_does_not_mask_successful_response(self, in_memory_span_exporter):
        """A broken span must NOT turn a successful AWS response into an
        application error. _ok and _err are now defensive — they swallow
        any span-method exception internally."""
        from neatlogs.bedrock import wrap_bedrock_client

        _setup_tracer(in_memory_span_exporter)

        body_out = json.dumps(
            {
                "content": [{"type": "text", "text": "hello"}],
                "usage": {"input_tokens": 4, "output_tokens": 2},
                "stop_reason": "end_turn",
            }
        ).encode()

        def invoke_model(**kwargs):
            return {"body": io.BytesIO(body_out)}

        # Build a span whose every set_attribute / set_status / end raises.
        # This is the worst case for the fallback _ok path.
        class _BrokenSpan:
            def set_attribute(self, *a, **kw):
                raise RuntimeError("broken")

            def set_status(self, *a, **kw):
                raise RuntimeError("broken")

            def end(self):
                raise RuntimeError("broken")

        class _BrokenTracer:
            def start_span(self, *a, **kw):
                return _BrokenSpan()

        client = _fake_bedrock_client(invoke_model=invoke_model)
        wrap_bedrock_client(client)

        with patch("neatlogs.bedrock.get_provider_tracer", return_value=_BrokenTracer()):
            response = client.invoke_model(
                modelId="anthropic.claude-3-haiku-20240307-v1:0",
                body=json.dumps({"messages": [{"role": "user", "content": "hi"}]}),
            )

        # The successful AWS response flows through unchanged.
        assert json.loads(response["body"].read())["content"][0]["text"] == "hello"
