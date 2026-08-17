"""Tests for per-call fail-open: if the neatlogs wrapper's pre-execution or
finalize phase throws, the user's original (untraced) LLM call must still run.

The host application must never crash due to a telemetry library bug. This is
the third leg of the fail-open contract:

  1. WRAP-time fail-open:  ``neatlogs.wrap(client)`` must succeed even if
     telemetry setup fails. (fixed in PRs #23, #29, #31, #34, #37, #43, #46)
  2. WRAP-context fail-open: workflow attribute coercion must not raise
     or silently corrupt types. (fixed in PR #70)
  3. PER-CALL fail-open:    the wrapper's pre-exec / finalize phase must not
     propagate exceptions to the caller; the untraced call must run as a
     fallback. (fixed here)

Issue: #15 — Telemetry errors can crash the host application due to lack of
       try-catch wrappers around hook pre-execution.
"""

import types
from unittest.mock import MagicMock

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import neatlogs._wrap_utils as _wu
from neatlogs._wrap_utils import _telemetry_fallback

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _setup_provider():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider, exporter


def _reset_wrapper_tracer():
    _wu._neatlogs_provider = None
    _wu._wrapper_tracer = None


class ThrowingMsg:
    """A messages entry that throws on any access — simulates a user passing
    a non-dict or malformed message that breaks the pre-exec iteration."""

    def get(self, key, default=None):
        raise RuntimeError(f"msg.get('{key}') raised unexpectedly")

    def __getitem__(self, key):
        raise RuntimeError(f"msg['{key}'] raised unexpectedly")


class ThrowingTool:
    """A tool entry that returns a non-serializable `parameters` schema."""

    def get(self, key, default=None):
        if key == "function":
            return {"parameters": object()}  # cannot be JSON-serialized
        return default


class _CountingOrig:
    """Stand-in for an SDK method that records every invocation."""

    def __init__(self, return_value="ok"):
        self.return_value = return_value
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.return_value


# ---------------------------------------------------------------------------
# _telemetry_fallback helper
# ---------------------------------------------------------------------------


def test_telemetry_fallback_calls_orig_with_same_args():
    orig = _CountingOrig(return_value="fallback-ok")
    result = _telemetry_fallback(orig, "a", kw1=1, kw2="x")
    assert result == "fallback-ok"
    assert orig.calls == [(("a",), {"kw1": 1, "kw2": "x"})]


def test_telemetry_fallback_propagates_orig_exception():
    def orig(*a, **kw):
        raise ValueError("real LLM error from orig")

    with pytest.raises(ValueError, match="real LLM error from orig"):
        _telemetry_fallback(orig)


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------


def test_openai_pre_exec_throwing_message_fails_open_to_orig():
    """Issue #15: pre-exec throws on a malformed message — orig_create must run."""
    provider, exporter = _setup_provider()
    _wu._neatlogs_provider = provider
    _wu._wrapper_tracer = None

    from neatlogs.openai import wrap_openai_client

    client = types.SimpleNamespace()
    orig = _CountingOrig(return_value="response-ok")
    client.chat = types.SimpleNamespace(
        completions=types.SimpleNamespace(create=orig)
    )

    wrap_openai_client(client)

    result = client.chat.completions.create(
        model="gpt-4", messages=[ThrowingMsg()]
    )
    assert result == "response-ok"
    assert len(orig.calls) == 1
    _reset_wrapper_tracer()


def test_openai_pre_exec_throwing_tool_fails_open_to_orig():
    provider, exporter = _setup_provider()
    _wu._neatlogs_provider = provider
    _wu._wrapper_tracer = None

    from neatlogs.openai import wrap_openai_client

    client = types.SimpleNamespace()
    orig = _CountingOrig(return_value="response-ok")
    client.chat = types.SimpleNamespace(
        completions=types.SimpleNamespace(create=orig)
    )

    wrap_openai_client(client)

    result = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "hi"}],
        tools=[ThrowingTool()],
    )
    assert result == "response-ok"
    assert len(orig.calls) == 1
    _reset_wrapper_tracer()


def test_openai_orig_exception_still_propagates_after_telemetry_setup():
    """The fix must not swallow real LLM errors that occur inside orig_create.
    The inner try/except around orig_create must still record on the span
    and re-raise."""
    provider, exporter = _setup_provider()
    _wu._neatlogs_provider = provider
    _wu._wrapper_tracer = None

    from neatlogs.openai import wrap_openai_client

    class Boom:
        def __call__(self, *a, **kw):
            raise ValueError("real LLM error")

    client = types.SimpleNamespace()
    client.chat = types.SimpleNamespace(
        completions=types.SimpleNamespace(create=Boom())
    )

    wrap_openai_client(client)

    with pytest.raises(ValueError, match="real LLM error"):
        client.chat.completions.create(
            model="gpt-4", messages=[{"role": "user", "content": "hi"}]
        )
    # The error span should have been emitted.
    spans = exporter.get_finished_spans()
    assert any(s.status.status_code.name == "ERROR" for s in spans)
    _reset_wrapper_tracer()


def test_openai_happy_path_still_emits_span():
    """The pre-exec wrapper must not change the normal-success telemetry path."""
    provider, exporter = _setup_provider()
    _wu._neatlogs_provider = provider
    _wu._wrapper_tracer = None

    from neatlogs.openai import wrap_openai_client

    fake_response = types.SimpleNamespace(
        choices=[
            types.SimpleNamespace(
                message=types.SimpleNamespace(
                    role="assistant", content="hello", tool_calls=None
                ),
                finish_reason="stop",
            )
        ],
        usage=types.SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            prompt_tokens_details=None,
            completion_tokens_details=None,
        ),
        model="gpt-4",
    )

    client = types.SimpleNamespace()
    client.chat = types.SimpleNamespace(
        completions=types.SimpleNamespace(create=lambda *a, **kw: fake_response)
    )

    wrap_openai_client(client)

    result = client.chat.completions.create(
        model="gpt-4", messages=[{"role": "user", "content": "hi"}]
    )
    assert result is fake_response
    spans = exporter.get_finished_spans()
    assert any(s.name == "openai.chat.completions.create" for s in spans)
    _reset_wrapper_tracer()


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------


def test_anthropic_pre_exec_throwing_messages_fails_open_to_orig():
    provider, _ = _setup_provider()
    _wu._neatlogs_provider = provider
    _wu._wrapper_tracer = None

    from neatlogs.anthropic import _patch_messages

    messages = types.SimpleNamespace()
    orig = _CountingOrig(return_value="anthropic-ok")
    messages.create = orig
    messages.stream = None
    messages._neatlogs_patched = False

    _patch_messages(messages)

    result = messages.create(
        model="claude-3-5-sonnet", messages=[ThrowingMsg()]
    )
    assert result == "anthropic-ok"
    assert len(orig.calls) == 1
    _reset_wrapper_tracer()


# ---------------------------------------------------------------------------
# Google GenAI
# ---------------------------------------------------------------------------


def test_google_genai_pre_exec_throwing_contents_fails_open_to_orig():
    provider, _ = _setup_provider()
    _wu._neatlogs_provider = provider
    _wu._wrapper_tracer = None

    from neatlogs.google_genai import _patch_models

    models = types.SimpleNamespace()
    orig = _CountingOrig(return_value="gemini-ok")
    models.generate_content = orig
    models.generate_content_stream = None
    models._neatlogs_patched = False

    _patch_models(models)

    # Pass contents as a list with a throwing entry to trip the pre-exec loop.
    result = models.generate_content(model="gemini-2.0-flash", contents=[ThrowingMsg()])
    assert result == "gemini-ok"
    assert len(orig.calls) == 1
    _reset_wrapper_tracer()


# ---------------------------------------------------------------------------
# Vertex AI
# ---------------------------------------------------------------------------


def test_vertex_ai_pre_exec_throwing_contents_fails_open_to_orig():
    provider, _ = _setup_provider()
    _wu._neatlogs_provider = provider
    _wu._wrapper_tracer = None

    from neatlogs.vertex_ai import _patch_models

    models = types.SimpleNamespace()
    orig = _CountingOrig(return_value="vertex-ok")
    models.generate_content = orig
    models.generate_content_stream = None
    models._neatlogs_vertex_patched = False

    _patch_models(models)

    result = models.generate_content(model="gemini-2.0-flash", contents=[ThrowingMsg()])
    assert result == "vertex-ok"
    assert len(orig.calls) == 1
    _reset_wrapper_tracer()


# ---------------------------------------------------------------------------
# Bedrock
# ---------------------------------------------------------------------------


def test_bedrock_pre_exec_throwing_body_fails_open_to_orig():
    provider, _ = _setup_provider()
    _wu._neatlogs_provider = provider
    _wu._wrapper_tracer = None

    from neatlogs.bedrock import _patch_invoke_model

    sentinel = {"sentinel": "bedrock-ok"}
    client = types.SimpleNamespace()
    orig = _CountingOrig(return_value=sentinel)
    client.invoke_model = orig

    _patch_invoke_model(client)

    # Pass a non-bytes body that will fail JSON decode in pre-exec.
    class BadBody:
        def read(self):
            raise RuntimeError("body is not readable")

    result = client.invoke_model(
        modelId="anthropic.claude-3-5-sonnet-20240620-v1:0",
        body=BadBody(),
    )
    assert result is sentinel
    assert len(orig.calls) == 1
    _reset_wrapper_tracer()


# ---------------------------------------------------------------------------
# OpenRouter
# ---------------------------------------------------------------------------


def test_openrouter_pre_exec_throwing_input_fails_open_to_orig():
    provider, _ = _setup_provider()
    _wu._neatlogs_provider = provider
    _wu._wrapper_tracer = None

    from neatlogs.openrouter import _patch_chat

    chat = types.SimpleNamespace()
    orig = _CountingOrig(return_value="openrouter-ok")
    chat.send = orig
    chat.send_async = None
    chat._neatlogs_openrouter_patched = False

    _patch_chat(chat)

    result = chat.send(
        model="openai/gpt-4o-mini",
        messages=[ThrowingMsg()],
    )
    assert result == "openrouter-ok"
    assert len(orig.calls) == 1
    _reset_wrapper_tracer()
