"""
Tests for Hermes Agent Instrumentation
=======================================
Hermes (NousResearch/hermes-agent) is not a test dependency, so we register
fake `run_agent` and `tools.registry` modules in sys.modules that mirror the
real surface (AIAgent.run_conversation + ToolRegistry.dispatch), then verify the
neatlogs.hermes patches emit AGENT and TOOL spans.
"""

import sys
import types

import pytest

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor


def _setup_tracer(exporter):
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    # Reset the process-global wrapper-tracer cache so each test binds to its
    # own provider (correct in production; leaks across tests otherwise).
    import neatlogs._wrap_utils as _wu
    _wu._wrapper_tracer = None
    return provider


@pytest.fixture
def fake_hermes(monkeypatch):
    """Install fake run_agent.AIAgent + tools.registry.ToolRegistry modules."""
    # --- run_agent.AIAgent ---
    run_agent_mod = types.ModuleType("run_agent")

    class AIAgent:
        def __init__(self, session_id="sess-xyz", model="claude-opus-4"):
            self.session_id = session_id
            self.model = model

        def run_conversation(self, user_message="", **kwargs):
            # Real loop would call the LLM + tools here; return a result dict.
            return {"response": f"Echo: {user_message}", "api_call_count": 2}

    run_agent_mod.AIAgent = AIAgent

    # --- tools.registry.ToolRegistry ---
    tools_pkg = types.ModuleType("tools")
    registry_mod = types.ModuleType("tools.registry")

    class ToolRegistry:
        def dispatch(self, name, args, **kwargs):
            return f"ran {name} with {args}"

    registry_mod.ToolRegistry = ToolRegistry
    registry_mod.registry = ToolRegistry()
    tools_pkg.registry = registry_mod

    monkeypatch.setitem(sys.modules, "run_agent", run_agent_mod)
    monkeypatch.setitem(sys.modules, "tools", tools_pkg)
    monkeypatch.setitem(sys.modules, "tools.registry", registry_mod)

    # Reset the neatlogs.hermes patch flag so each test patches fresh classes.
    import neatlogs.hermes as h
    h._PATCHED = False
    yield {"AIAgent": AIAgent, "ToolRegistry": ToolRegistry}
    h._PATCHED = False


class TestHermesInstrumentation:
    def test_run_conversation_emits_agent_span(self, fake_hermes, in_memory_span_exporter):
        _setup_tracer(in_memory_span_exporter)

        from neatlogs.hermes import wrap_hermes

        agent = fake_hermes["AIAgent"](session_id="sess-xyz", model="claude-opus-4")
        wrap_hermes(agent)

        result = agent.run_conversation("Hello Hermes")
        assert result["response"] == "Echo: Hello Hermes"

        spans = in_memory_span_exporter.get_finished_spans()
        agent_spans = [s for s in spans if s.attributes.get("neatlogs.span.kind") == "agent"]
        assert len(agent_spans) == 1
        attrs = agent_spans[0].attributes
        assert attrs.get("neatlogs.agent.framework") == "hermes"
        assert attrs.get("neatlogs.conversation.id") == "sess-xyz"
        assert attrs.get("neatlogs.agent.model") == "claude-opus-4"
        assert attrs.get("input.value") == "Hello Hermes"
        assert attrs.get("output.value") == "Echo: Hello Hermes"
        assert attrs.get("neatlogs.agent.num_turns") == 2

    def test_tool_dispatch_emits_tool_span(self, fake_hermes, in_memory_span_exporter):
        _setup_tracer(in_memory_span_exporter)

        from neatlogs.hermes import wrap_hermes

        wrap_hermes(fake_hermes["AIAgent"]())
        registry = fake_hermes["ToolRegistry"]()
        out = registry.dispatch(
            "web_search",
            {"query": "neatlogs"},
            session_id="sess-xyz",
            tool_call_id="tc-1",
            turn_id="turn-1",
        )
        assert out == "ran web_search with {'query': 'neatlogs'}"

        spans = in_memory_span_exporter.get_finished_spans()
        tool_spans = [s for s in spans if s.attributes.get("neatlogs.span.kind") == "tool"]
        assert len(tool_spans) == 1
        attrs = tool_spans[0].attributes
        assert attrs.get("neatlogs.tool.name") == "web_search"
        assert attrs.get("neatlogs.conversation.id") == "sess-xyz"
        assert attrs.get("neatlogs.tool_call.id") == "tc-1"
        assert "neatlogs" in attrs.get("input.value", "")
        assert "ran web_search" in attrs.get("output.value", "")

    def test_tool_dispatch_records_error(self, fake_hermes, in_memory_span_exporter):
        _setup_tracer(in_memory_span_exporter)

        from neatlogs.hermes import wrap_hermes

        ToolRegistry = fake_hermes["ToolRegistry"]

        def boom(self, name, args, **kwargs):
            raise RuntimeError("tool failed")

        ToolRegistry.dispatch = boom
        wrap_hermes(fake_hermes["AIAgent"]())

        with pytest.raises(RuntimeError):
            ToolRegistry().dispatch("bad_tool", {})

        spans = in_memory_span_exporter.get_finished_spans()
        tool_spans = [s for s in spans if s.attributes.get("neatlogs.span.kind") == "tool"]
        assert len(tool_spans) == 1
        assert tool_spans[0].status.status_code.name == "ERROR"
