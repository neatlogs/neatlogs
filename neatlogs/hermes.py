"""
Neatlogs Hermes Agent integration.

Hermes (NousResearch/hermes-agent) is a Python agent. Its public surface is the
``AIAgent`` class in the top-level ``run_agent`` module, whose
``run_conversation()`` drives an agentic loop. Inside the loop:

  * LLM calls go through the standard ``openai`` SDK
    (``openai.OpenAI(...).chat.completions.create`` in
    ``agent/chat_completion_helpers.py``) — so we let neatlogs' existing OpenAI
    instrumentation capture LLM spans (auto-loaded via the registry).
  * Tool execution funnels through ``tools.registry.ToolRegistry.dispatch`` — the
    single chokepoint that every built-in, parallel, and MCP tool flows through.

So this integration produces:

  AGENT  hermes.run_conversation        (one agentic run; conversation.id = session_id)
    ↳ LLM   chat.completions.create      (from neatlogs.openai — auto-loaded)
    ↳ TOOL  tool dispatch                (from ToolRegistry.dispatch)

Enable via ``neatlogs.init(instrumentations=["hermes"])`` (which auto-loads
``openai``), or wrap an instance with ``neatlogs.wrap(AIAgent(...))``.

Note: Hermes' non-OpenAI provider adapters (anthropic / bedrock / gemini / codex)
do NOT flow through the openai SDK; enable those provider instrumentations
alongside ``hermes`` if you use them (e.g. ``["hermes", "anthropic"]``).
"""

from typing import Any

from opentelemetry.trace import StatusCode

from ._wrap_utils import attach_as_current, detach, get_tracer, is_suppressed, serialize

_PROVIDER = "hermes"


class HermesInstrumentor:
    """Instrumentor class for InstrumentationManager integration."""

    def instrument(self, tracer_provider=None):
        _patch_hermes()

    def uninstrument(self):
        _unpatch_hermes()


def wrap_hermes(agent: Any) -> Any:
    """
    Enable Neatlogs tracing for Hermes. ``AIAgent.run_conversation`` and
    ``ToolRegistry.dispatch`` are patched at the class level, so a single call
    covers this instance and every other AIAgent created in the process.
    Returns the same agent.
    """
    _patch_hermes()
    return agent


# ---------------------------------------------------------------------------
# Class-level patches
# ---------------------------------------------------------------------------

_PATCHED = False


def _patch_hermes() -> None:
    global _PATCHED
    if _PATCHED:
        return
    _PATCHED = True

    _patch_run_conversation()
    _patch_tool_dispatch()


def _patch_run_conversation() -> None:
    try:
        from run_agent import AIAgent
    except Exception:
        return

    if getattr(AIAgent, "_neatlogs_patched", False):
        return

    orig = AIAgent.run_conversation

    def patched_run_conversation(self, user_message="", *args, **kwargs):
        if is_suppressed():
            return orig(self, user_message, *args, **kwargs)

        session_id = getattr(self, "session_id", None)
        model = getattr(self, "model", None)

        span = get_tracer().start_span(
            name="hermes.run_conversation",
            attributes={
                "neatlogs.span.kind": "agent",
                "neatlogs.agent.framework": _PROVIDER,
                "neatlogs.llm.provider": _PROVIDER,
            },
        )
        if session_id:
            span.set_attribute("neatlogs.conversation.id", str(session_id))
        if model:
            span.set_attribute("neatlogs.agent.model", str(model))
        if isinstance(user_message, str) and user_message:
            span.set_attribute("input.value", user_message[:10000])

        # Make the AGENT span the active span so LLM spans (from the openai SDK
        # wrapper) and TOOL spans (from ToolRegistry.dispatch) nest under it.
        token = attach_as_current(span)
        try:
            result = orig(self, user_message, *args, **kwargs)
        except Exception as e:
            detach(token)
            span.set_status(StatusCode.ERROR, str(e))
            span.record_exception(e)
            span.end()
            raise

        detach(token)
        _finalize_conversation(span, result)
        return result

    AIAgent.run_conversation = patched_run_conversation
    AIAgent._neatlogs_patched = True


def _finalize_conversation(span: Any, result: Any) -> None:
    try:
        if isinstance(result, dict):
            # The loop returns a dict; surface the assistant response if present.
            content = (
                result.get("response")
                or result.get("content")
                or result.get("message")
                or result.get("final_response")
            )
            if content is not None:
                span.set_attribute(
                    "output.value",
                    (content if isinstance(content, str) else serialize(content))[:10000],
                )
            if result.get("api_call_count") is not None:
                span.set_attribute("neatlogs.agent.num_turns", result["api_call_count"])
        elif isinstance(result, str):
            span.set_attribute("output.value", result[:10000])
    except Exception:
        pass
    span.set_status(StatusCode.OK)
    span.end()


def _patch_tool_dispatch() -> None:
    try:
        from tools.registry import ToolRegistry
    except Exception:
        return

    if getattr(ToolRegistry, "_neatlogs_patched", False):
        return

    orig_dispatch = ToolRegistry.dispatch

    def patched_dispatch(self, name, args, **kwargs):
        if is_suppressed():
            return orig_dispatch(self, name, args, **kwargs)

        span = get_tracer().start_span(
            name=f"hermes.tool.{name}",
            attributes={
                "neatlogs.span.kind": "tool",
                "neatlogs.tool.name": str(name),
            },
        )
        try:
            span.set_attribute("input.value", serialize(args)[:10000])
        except Exception:
            pass
        for key, attr in (("session_id", "neatlogs.conversation.id"),
                          ("tool_call_id", "neatlogs.tool_call.id"),
                          ("turn_id", "neatlogs.turn.id")):
            val = kwargs.get(key)
            if val:
                span.set_attribute(attr, str(val))

        token = attach_as_current(span)
        try:
            result = orig_dispatch(self, name, args, **kwargs)
        except Exception as e:
            detach(token)
            span.set_status(StatusCode.ERROR, str(e))
            span.record_exception(e)
            span.end()
            raise

        detach(token)
        try:
            if result is not None:
                span.set_attribute(
                    "output.value",
                    (result if isinstance(result, str) else serialize(result))[:10000],
                )
        except Exception:
            pass
        span.set_status(StatusCode.OK)
        span.end()
        return result

    ToolRegistry.dispatch = patched_dispatch
    ToolRegistry._neatlogs_patched = True


def _unpatch_hermes() -> None:
    global _PATCHED
    if not _PATCHED:
        return
    try:
        from run_agent import AIAgent

        if getattr(AIAgent, "_neatlogs_patched", False):
            # We didn't stash the original; leave the wrapper (idempotent + cheap).
            pass
    except Exception:
        pass
    _PATCHED = False
