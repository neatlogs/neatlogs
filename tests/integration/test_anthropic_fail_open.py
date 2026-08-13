"""Regression tests for Anthropic wrapper fail-open behavior."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
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


def _message_response():
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text="hello")],
        stop_reason="end_turn",
        model="claude-3-haiku-20240307",
        usage=SimpleNamespace(input_tokens=5, output_tokens=3),
    )


class TestAnthropicFailOpen:
    def test_sync_create_fails_open_on_tracer_error(self, in_memory_span_exporter):
        from neatlogs.anthropic import wrap_anthropic_client

        _setup_tracer(in_memory_span_exporter)

        def create(**kwargs):
            return _message_response()

        messages = SimpleNamespace(create=create, stream=None)
        client = SimpleNamespace(messages=messages, completions=None, beta=None)
        wrap_anthropic_client(client)

        with patch(
            "neatlogs.anthropic.get_provider_tracer",
            side_effect=RuntimeError("trace failed"),
        ):
            response = client.messages.create(
                model="claude-3-haiku-20240307",
                messages=[{"role": "user", "content": "hi"}],
            )

        assert getattr(response.content[0], "text", None) == "hello"
        assert len(in_memory_span_exporter.get_finished_spans()) == 0

    @pytest.mark.asyncio
    async def test_async_create_fails_open_on_tracer_error(self, in_memory_span_exporter):
        from neatlogs.anthropic import wrap_async_anthropic_client

        _setup_tracer(in_memory_span_exporter)

        async def create(**kwargs):
            return _message_response()

        messages = SimpleNamespace(create=create, stream=None)
        client = SimpleNamespace(messages=messages, completions=None, beta=None)
        wrap_async_anthropic_client(client)

        with patch(
            "neatlogs.anthropic.get_provider_tracer",
            side_effect=RuntimeError("trace failed"),
        ):
            response = await client.messages.create(
                model="claude-3-haiku-20240307",
                messages=[{"role": "user", "content": "hi"}],
            )

        assert getattr(response.content[0], "text", None) == "hello"
        assert len(in_memory_span_exporter.get_finished_spans()) == 0

    def test_sync_create_propagates_sdk_errors(self, in_memory_span_exporter):
        from neatlogs.anthropic import wrap_anthropic_client

        _setup_tracer(in_memory_span_exporter)

        def create(**kwargs):
            raise ValueError("sdk failure")

        messages = SimpleNamespace(create=create, stream=None)
        client = SimpleNamespace(messages=messages, completions=None, beta=None)
        wrap_anthropic_client(client)

        with pytest.raises(ValueError, match="sdk failure"):
            client.messages.create(
                model="claude-3-haiku-20240307",
                messages=[{"role": "user", "content": "hi"}],
            )

    def test_sync_stream_called_once_when_setup_fails(self, in_memory_span_exporter):
        """When telemetry setup fails, orig_stream must be called exactly once (not double-called)."""
        from neatlogs.anthropic import wrap_anthropic_client

        _setup_tracer(in_memory_span_exporter)

        call_count = {"n": 0}

        def stream_fn(**kwargs):
            call_count["n"] += 1
            return SimpleNamespace()

        messages = SimpleNamespace(create=lambda **k: None, stream=stream_fn)
        client = SimpleNamespace(messages=messages, completions=None, beta=None)
        wrap_anthropic_client(client)

        with patch(
            "neatlogs.anthropic.get_provider_tracer",
            side_effect=RuntimeError("trace failed"),
        ):
            result = client.messages.stream(
                model="claude-3-haiku-20240307",
                messages=[{"role": "user", "content": "hi"}],
            )

        assert call_count["n"] == 1, f"orig_stream was called {call_count['n']} times, expected 1"
        assert result is not None
        assert len(in_memory_span_exporter.get_finished_spans()) == 0

    @pytest.mark.asyncio
    async def test_async_stream_called_once_when_setup_fails(self, in_memory_span_exporter):
        """Async path: when telemetry setup fails, orig_stream must be called exactly once."""
        from neatlogs.anthropic import wrap_async_anthropic_client

        _setup_tracer(in_memory_span_exporter)

        call_count = {"n": 0}

        async def stream_fn(**kwargs):
            call_count["n"] += 1
            return SimpleNamespace()

        messages = SimpleNamespace(create=lambda **k: None, stream=stream_fn)
        client = SimpleNamespace(messages=messages, completions=None, beta=None)
        wrap_async_anthropic_client(client)

        with patch(
            "neatlogs.anthropic.get_provider_tracer",
            side_effect=RuntimeError("trace failed"),
        ):
            result = await client.messages.stream(
                model="claude-3-haiku-20240307",
                messages=[{"role": "user", "content": "hi"}],
            )

        assert call_count["n"] == 1, f"orig_stream was called {call_count['n']} times, expected 1"
        assert result is not None
        assert len(in_memory_span_exporter.get_finished_spans()) == 0

    def test_sync_stream_called_once_when_orig_stream_raises(self, in_memory_span_exporter):
        """When orig_stream raises, the fallback must NOT re-invoke it (no double-call)."""
        from neatlogs.anthropic import wrap_anthropic_client

        _setup_tracer(in_memory_span_exporter)

        call_count = {"n": 0}

        def stream_fn(**kwargs):
            call_count["n"] += 1
            raise RuntimeError("stream failed")

        messages = SimpleNamespace(create=lambda **k: None, stream=stream_fn)
        client = SimpleNamespace(messages=messages, completions=None, beta=None)
        wrap_anthropic_client(client)

        with pytest.raises(RuntimeError, match="stream failed"):
            client.messages.stream(
                model="claude-3-haiku-20240307",
                messages=[{"role": "user", "content": "hi"}],
            )

        assert call_count["n"] == 1, f"orig_stream was called {call_count['n']} times, expected 1"

    def test_parse_fails_open_on_tracer_error(self, in_memory_span_exporter):
        """messages.parse has the same fail-open contract when telemetry setup fails."""
        from neatlogs.anthropic import wrap_anthropic_client

        _setup_tracer(in_memory_span_exporter)

        call_count = {"n": 0}

        def parse(**kwargs):
            call_count["n"] += 1
            return _message_response()

        messages = SimpleNamespace(
            create=lambda **k: None, stream=None, parse=parse, count_tokens=None
        )
        client = SimpleNamespace(messages=messages, completions=None, beta=None)
        wrap_anthropic_client(client)

        with patch(
            "neatlogs.anthropic.get_provider_tracer",
            side_effect=RuntimeError("trace failed"),
        ):
            response = client.messages.parse(
                model="claude-3-haiku-20240307",
                messages=[{"role": "user", "content": "hi"}],
            )

        assert call_count["n"] == 1, f"called {call_count['n']} times, expected 1"
        assert getattr(response.content[0], "text", None) == "hello"
        assert len(in_memory_span_exporter.get_finished_spans()) == 0

    def test_count_tokens_fails_open_on_tracer_error(self, in_memory_span_exporter):
        """messages.count_tokens has the same fail-open contract when telemetry setup fails."""
        from neatlogs.anthropic import wrap_anthropic_client

        _setup_tracer(in_memory_span_exporter)

        call_count = {"n": 0}

        def count_tokens(**kwargs):
            call_count["n"] += 1
            return SimpleNamespace(input_tokens=5)

        messages = SimpleNamespace(
            create=lambda **k: None, stream=None, parse=None, count_tokens=count_tokens
        )
        client = SimpleNamespace(messages=messages, completions=None, beta=None)
        wrap_anthropic_client(client)

        with patch(
            "neatlogs.anthropic.get_provider_tracer",
            side_effect=RuntimeError("trace failed"),
        ):
            response = client.messages.count_tokens(
                model="claude-3-haiku-20240307",
                messages=[{"role": "user", "content": "hi"}],
            )

        assert call_count["n"] == 1, f"called {call_count['n']} times, expected 1"
        assert response.input_tokens == 5
        assert len(in_memory_span_exporter.get_finished_spans()) == 0
