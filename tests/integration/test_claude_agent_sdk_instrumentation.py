"""
Tests for Claude Agent SDK Instrumentation (faithful port of claude-agent-sdk.ts).

`claude_agent_sdk` is not a test dependency; we register a fake module whose `query()` yields the
REAL message shapes (system init -> assistant text -> assistant tool_use -> user tool_result -> ...
-> result), each carrying `parent_tool_use_id` (None=orchestrator, a Task id=subagent). We verify
the wrapper builds the canonical tree:

    AGENT claude_agent.query (root) -> LLM (one per buffered model turn) -> TOOL (closed by tool_result)
      -> subagent AGENT nested under the Task TOOL span

All attribute keys are the canonical ones (neatlogs/config/attribute-mapping.json); the agent root
is the trace ROOT (NO WORKFLOW wrapper).
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
    import neatlogs._wrap_utils as _wu

    _wu._wrapper_tracer = None
    return provider


def _run(coro):
    import asyncio

    return asyncio.run(coro)


# message factories (dict-shaped, like the SDK's typed messages read via attrs/keys) ----------


def _system(session_id="sess-1", model="claude-haiku-4-5"):
    return {"type": "system", "session_id": session_id, "model": model}


def _assistant(
    text=None,
    tool_use=None,
    usage=None,
    model="claude-haiku-4-5",
    stop_reason=None,
    parent_tool_use_id=None,
    subagent_type=None,
):
    content = []
    if text is not None:
        content.append({"type": "text", "text": text})
    if tool_use is not None:
        content.append({"type": "tool_use", **tool_use})
    msg = {
        "type": "assistant",
        "parent_tool_use_id": parent_tool_use_id,
        "message": {"content": content, "model": model, "usage": usage, "stop_reason": stop_reason},
    }
    # the SDK tags a subagent's own messages with subagent_type (the wrapper reads it to name the
    # nested AGENT span — same as the TS reference's msg.subagent_type)
    if subagent_type is not None:
        msg["subagent_type"] = subagent_type
    return msg


def _user_tool_result(tool_use_id, output, is_error=False, parent_tool_use_id=None):
    return {
        "type": "user",
        "parent_tool_use_id": parent_tool_use_id,
        "message": {
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": output,
                    "is_error": is_error,
                }
            ]
        },
    }


def _result(text="final answer", usage=None, total_cost_usd=None, num_turns=None, is_error=False):
    return {
        "type": "result",
        "result": text,
        "usage": usage,
        "total_cost_usd": total_cost_usd,
        "num_turns": num_turns,
        "is_error": is_error,
    }


@pytest.fixture
def fake_cas(monkeypatch):
    mod = types.ModuleType("claude_agent_sdk")

    # default script: a one-tool run — system, assistant(text+tool_use), tool_result, assistant(text), result
    async def query(*, prompt=None, options=None):
        for m in _SCRIPT[0]:
            yield m

    mod.query = query
    mod.ClaudeSDKClient = None
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", mod)
    import neatlogs.claude_agent_sdk as c

    c._PATCHED = False
    c._ORIGINALS.clear()
    yield mod
    c._unpatch_claude_agent_sdk()
    c._PATCHED = False


# a mutable holder so a test can swap the script the fake query() yields
_SCRIPT = [[]]


def _set_script(messages):
    _SCRIPT[0] = messages


def _consume(mod, prompt="how are my agents?"):
    async def go():
        out = []
        async for m in mod.query(prompt=prompt):
            out.append(m.get("type"))
        return out

    return _run(go())


def _kinds(spans, k):
    return [s for s in spans if s.attributes.get("neatlogs.span.kind") == k]


class TestClaudeAgentSDKInstrumentation:
    def test_agent_is_root_no_workflow(self, fake_cas, in_memory_span_exporter):
        # the canonical design: AGENT is the trace root, NO WORKFLOW wrapper
        _set_script(
            [
                _system(),
                _assistant(
                    text="checking",
                    tool_use={"id": "t1", "name": "triage_signal", "input": {"signal": "X"}},
                    usage={"input_tokens": 1200, "output_tokens": 90},
                ),
                _user_tool_result("t1", "fired 322 times"),
                _assistant(
                    text="all good",
                    usage={"input_tokens": 1500, "output_tokens": 40},
                    stop_reason="end_turn",
                ),
                _result(text="final answer", usage={"input_tokens": 2700, "output_tokens": 130}),
            ]
        )
        _setup_tracer(in_memory_span_exporter)
        from neatlogs.claude_agent_sdk import wrap_claude_agent_sdk

        wrap_claude_agent_sdk(fake_cas)
        _consume(fake_cas)

        spans = in_memory_span_exporter.get_finished_spans()
        assert len(_kinds(spans, "workflow")) == 0  # NO workflow wrapper
        agents = _kinds(spans, "agent")
        assert len(agents) == 1
        root = agents[0]
        assert root.name == "claude_agent.query"
        assert root.parent is None  # it IS the trace root
        assert root.attributes.get("neatlogs.agent.framework") == "claude_agent_sdk"

    def test_one_llm_span_per_buffered_turn(self, fake_cas, in_memory_span_exporter):
        # turn 1 = text + tool_use (one model turn across blocks); turn 2 = final text → 2 LLM spans
        _set_script(
            [
                _system(),
                _assistant(
                    text="let me check",
                    tool_use={"id": "t1", "name": "triage_signal", "input": {"signal": "X"}},
                    usage={"input_tokens": 1200, "output_tokens": 90},
                    stop_reason="tool_use",
                ),
                _user_tool_result("t1", "ok"),
                _assistant(
                    text="done",
                    usage={"input_tokens": 1500, "output_tokens": 40},
                    stop_reason="end_turn",
                ),
                _result(),
            ]
        )
        _setup_tracer(in_memory_span_exporter)
        from neatlogs.claude_agent_sdk import wrap_claude_agent_sdk

        wrap_claude_agent_sdk(fake_cas)
        _consume(fake_cas)
        spans = in_memory_span_exporter.get_finished_spans()
        llm = _kinds(spans, "llm")
        assert len(llm) == 2
        # first LLM turn carries canonical model + token + tool_call attrs
        first = next(s for s in llm if s.attributes.get("neatlogs.llm.token_count.prompt") == 1200)
        a = first.attributes
        assert a.get("neatlogs.llm.model_name") == "claude-haiku-4-5"
        assert a.get("neatlogs.llm.provider") == "anthropic"
        assert a.get("neatlogs.llm.token_count.completion") == 90
        assert a.get("neatlogs.llm.token_count.total") == 1290
        assert a.get("neatlogs.llm.finish_reason") == "tool_use"
        assert a.get("neatlogs.llm.tool_calls.0.name") == "triage_signal"
        # both flat input.value and indexed input_messages are set (UI panel + drawer)
        assert a.get("input.value") and "messages" in a["input.value"]
        assert a.get("neatlogs.llm.input_messages.0.role") == "user"

    def test_tool_span_opened_then_closed_by_tool_result(self, fake_cas, in_memory_span_exporter):
        _set_script(
            [
                _system(),
                _assistant(
                    tool_use={"id": "t1", "name": "triage_signal", "input": {"signal": "X"}},
                    usage={"input_tokens": 10, "output_tokens": 5},
                ),
                _user_tool_result("t1", "fired 322 times"),
                _result(),
            ]
        )
        _setup_tracer(in_memory_span_exporter)
        from neatlogs.claude_agent_sdk import wrap_claude_agent_sdk

        wrap_claude_agent_sdk(fake_cas)
        _consume(fake_cas)
        spans = in_memory_span_exporter.get_finished_spans()
        tool = _kinds(spans, "tool")
        assert len(tool) == 1
        a = tool[0].attributes
        assert a.get("neatlogs.tool.name") == "triage_signal"
        assert a.get("neatlogs.tool_call.id") == "t1"
        assert "signal" in a.get("input.value", "")  # tool args
        assert "fired 322" in a.get("output.value", "")  # closed by the matching tool_result

    def test_tool_error_marks_tool_span_error(self, fake_cas, in_memory_span_exporter):
        _set_script(
            [
                _system(),
                _assistant(
                    tool_use={"id": "t1", "name": "bad_tool", "input": {}},
                    usage={"input_tokens": 1, "output_tokens": 1},
                ),
                _user_tool_result("t1", "boom", is_error=True),
                _result(),
            ]
        )
        _setup_tracer(in_memory_span_exporter)
        from neatlogs.claude_agent_sdk import wrap_claude_agent_sdk

        wrap_claude_agent_sdk(fake_cas)
        _consume(fake_cas)
        spans = in_memory_span_exporter.get_finished_spans()
        tool = _kinds(spans, "tool")[0]
        assert tool.status.status_code.name == "ERROR"
        assert tool.attributes.get("neatlogs.tool.is_error") is True

    def test_subagent_nests_under_task_tool(self, fake_cas, in_memory_span_exporter):
        # orchestrator spawns a Task (id=task1); the subagent's messages carry parent_tool_use_id=task1
        _set_script(
            [
                _system(),
                _assistant(
                    tool_use={
                        "id": "task1",
                        "name": "Task",
                        "input": {"subagent_type": "investigator"},
                    },
                    usage={"input_tokens": 100, "output_tokens": 20},
                ),
                # subagent activity (parent_tool_use_id = the Task id; tagged with subagent_type)
                _assistant(
                    text="subagent working",
                    parent_tool_use_id="task1",
                    subagent_type="investigator",
                    usage={"input_tokens": 50, "output_tokens": 10},
                    tool_use={"id": "s1", "name": "run_query", "input": {"q": "x"}},
                ),
                _user_tool_result("s1", "rows", parent_tool_use_id="task1"),
                _assistant(
                    text="subagent done",
                    parent_tool_use_id="task1",
                    subagent_type="investigator",
                    usage={"input_tokens": 60, "output_tokens": 5},
                ),
                # Task tool result closes the orchestrator's Task TOOL span
                _user_tool_result("task1", "subagent result"),
                _assistant(
                    text="orchestrator done", usage={"input_tokens": 200, "output_tokens": 8}
                ),
                _result(),
            ]
        )
        _setup_tracer(in_memory_span_exporter)
        from neatlogs.claude_agent_sdk import wrap_claude_agent_sdk

        wrap_claude_agent_sdk(fake_cas)
        _consume(fake_cas)
        spans = in_memory_span_exporter.get_finished_spans()
        agents = _kinds(spans, "agent")
        # two AGENT spans: orchestrator + subagent
        assert len(agents) == 2
        root = next(s for s in agents if s.name == "claude_agent.query")
        sub = next(s for s in agents if s.name == "claude_agent.subagent.investigator")
        task_tool = next(
            s for s in _kinds(spans, "tool") if s.attributes.get("neatlogs.tool.name") == "Task"
        )
        # subagent AGENT nests under the Task TOOL span; everything one trace
        assert sub.parent is not None and sub.parent.span_id == task_tool.context.span_id
        assert len({s.context.trace_id for s in agents}) == 1
        assert sub.attributes.get("neatlogs.agent.name") == "investigator"

    def test_result_sets_agent_attrs_and_output(self, fake_cas, in_memory_span_exporter):
        _set_script(
            [
                _system(session_id="sess-42", model="claude-haiku-4-5"),
                _assistant(text="answer", usage={"input_tokens": 5, "output_tokens": 3}),
                _result(
                    text="final answer",
                    usage={"input_tokens": 2700, "output_tokens": 130},
                    total_cost_usd=0.0123,
                    num_turns=2,
                ),
            ]
        )
        _setup_tracer(in_memory_span_exporter)
        from neatlogs.claude_agent_sdk import wrap_claude_agent_sdk

        wrap_claude_agent_sdk(fake_cas)
        _consume(fake_cas)
        spans = in_memory_span_exporter.get_finished_spans()
        root = _kinds(spans, "agent")[0]
        a = root.attributes
        assert a.get("output.value") and "final answer" in a["output.value"]
        # canonical attrs the TS reference sets on the root agent (verified against claude-agent-sdk.ts)
        assert a.get("neatlogs.conversation.id") == "sess-42"
        assert a.get("neatlogs.agent.model") == "claude-haiku-4-5"
        assert a.get("neatlogs.agent.cost_usd") == 0.0123
        assert a.get("neatlogs.agent.num_turns") == 2
        assert a.get("neatlogs.llm.token_count.prompt") == 2700

    def test_string_prompt_captured_as_input(self, fake_cas, in_memory_span_exporter):
        _set_script(
            [
                _system(),
                _assistant(text="hi", usage={"input_tokens": 1, "output_tokens": 1}),
                _result(),
            ]
        )
        _setup_tracer(in_memory_span_exporter)
        from neatlogs.claude_agent_sdk import wrap_claude_agent_sdk

        wrap_claude_agent_sdk(fake_cas)
        _consume(fake_cas, prompt="how are my agents?")
        spans = in_memory_span_exporter.get_finished_spans()
        root = _kinds(spans, "agent")[0]
        assert "agents" in root.attributes.get("input.value", "")

    def test_wrap_via_dispatcher(self, fake_cas, in_memory_span_exporter):
        _set_script(
            [
                _system(),
                _assistant(text="x", usage={"input_tokens": 1, "output_tokens": 1}),
                _result(),
            ]
        )
        _setup_tracer(in_memory_span_exporter)
        import neatlogs

        neatlogs.wrap(fake_cas)
        _consume(fake_cas)
        spans = in_memory_span_exporter.get_finished_spans()
        assert _kinds(spans, "agent")

    def test_idempotent_double_wrap(self, fake_cas, in_memory_span_exporter):
        _set_script(
            [
                _system(),
                _assistant(text="x", usage={"input_tokens": 1, "output_tokens": 1}),
                _result(),
            ]
        )
        _setup_tracer(in_memory_span_exporter)
        from neatlogs.claude_agent_sdk import wrap_claude_agent_sdk

        wrap_claude_agent_sdk(fake_cas)
        wrap_claude_agent_sdk(fake_cas)
        _consume(fake_cas)
        spans = in_memory_span_exporter.get_finished_spans()
        assert len(_kinds(spans, "agent")) == 1  # not double-wrapped

    def test_no_uncon_sumed_agent_attrs(self, fake_cas, in_memory_span_exporter):
        # every neatlogs.agent.* / neatlogs.* key we emit must be a real canonical key (no invented
        # cosmetic attrs). These are the keys the TS reference + attribute-mapping.json establish.
        _set_script(
            [
                _system(session_id="s", model="m"),
                _assistant(text="a", usage={"input_tokens": 1, "output_tokens": 1}),
                _result(total_cost_usd=0.01, num_turns=1),
            ]
        )
        _setup_tracer(in_memory_span_exporter)
        from neatlogs.claude_agent_sdk import wrap_claude_agent_sdk

        wrap_claude_agent_sdk(fake_cas)
        _consume(fake_cas)
        spans = in_memory_span_exporter.get_finished_spans()
        root = _kinds(spans, "agent")[0]
        CANONICAL_AGENT = {
            "neatlogs.agent.framework",
            "neatlogs.agent.name",
            "neatlogs.agent.model",
            "neatlogs.agent.cost_usd",
            "neatlogs.agent.num_turns",
            "neatlogs.agent.is_error",
        }
        emitted = {k for k in root.attributes if k.startswith("neatlogs.agent.")}
        assert emitted <= CANONICAL_AGENT, f"unexpected agent attrs: {emitted - CANONICAL_AGENT}"

    def test_fail_open_on_tracer_error(self, fake_cas, in_memory_span_exporter, monkeypatch):
        _set_script(
            [
                _system(),
                _assistant(text="x", usage={"input_tokens": 1, "output_tokens": 1}),
                _result(),
            ]
        )
        _setup_tracer(in_memory_span_exporter)
        import neatlogs.claude_agent_sdk as c
        from neatlogs.claude_agent_sdk import wrap_claude_agent_sdk

        monkeypatch.setattr(c, "get_tracer", lambda: (_ for _ in ()).throw(RuntimeError("down")))
        wrap_claude_agent_sdk(fake_cas)
        # iteration must still complete despite tracing failure
        assert _consume(fake_cas) == ["system", "assistant", "result"]

    def test_real_python_sdk_object_shapes(self, in_memory_span_exporter, monkeypatch):
        """THE regression guard for the live bug: the Python claude_agent_sdk yields typed OBJECTS
        (AssistantMessage/UserMessage/SystemMessage/ResultMessage) with NO `type`/`message` fields —
        dispatch is by CLASS NAME, content is typed blocks. The dict-shaped fakes above did NOT catch
        that the wrapper read msg['type'] (always None on objects) → zero LLM/TOOL spans live. This
        builds messages from object-like shims matching the real classes' field names + class names.
        """
        import sys
        import types

        def _obj(cls_name, **fields):
            return type(cls_name, (), fields)()  # an instance whose class NAME drives dispatch

        def TextBlock(text):
            return _obj("TextBlock", text=text)

        def ToolUseBlock(id, name, inp):
            return _obj("ToolUseBlock", id=id, name=name, input=inp)

        def ToolResultBlock(tool_use_id, content, is_error=False):
            return _obj(
                "ToolResultBlock", tool_use_id=tool_use_id, content=content, is_error=is_error
            )

        # real objects: NO 'type' field, content DIRECTLY on the message, no '.message' wrapper
        msgs = [
            _obj(
                "SystemMessage",
                subtype="init",
                data={"session_id": "s1", "model": "claude-haiku-4-5"},
            ),
            _obj(
                "AssistantMessage",
                parent_tool_use_id=None,
                model="claude-haiku-4-5",
                stop_reason="tool_use",
                usage={"input_tokens": 10, "output_tokens": 5},
                content=[
                    TextBlock("checking"),
                    ToolUseBlock("t1", "triage_signal", {"signal": "X"}),
                ],
            ),
            _obj(
                "UserMessage",
                parent_tool_use_id=None,
                content=[ToolResultBlock("t1", "fired 322 times")],
                tool_use_result=None,
            ),
            _obj(
                "AssistantMessage",
                parent_tool_use_id=None,
                model="claude-haiku-4-5",
                stop_reason="end_turn",
                usage={"input_tokens": 12, "output_tokens": 3},
                content=[TextBlock("all good")],
            ),
            _obj(
                "ResultMessage",
                subtype="success",
                result="final answer",
                is_error=False,
                num_turns=2,
                total_cost_usd=0.0123,
                session_id="s1",
                usage={"input_tokens": 2700, "output_tokens": 130},
            ),
        ]
        mod = types.ModuleType("claude_agent_sdk")

        async def query(*, prompt=None, options=None):
            for m in msgs:
                yield m

        mod.query = query
        mod.ClaudeSDKClient = None
        monkeypatch.setitem(sys.modules, "claude_agent_sdk", mod)
        import neatlogs.claude_agent_sdk as c

        c._PATCHED = False
        c._ORIGINALS.clear()

        _setup_tracer(in_memory_span_exporter)
        c.wrap_claude_agent_sdk(mod)
        _run((lambda: _drain(mod))())

        spans = in_memory_span_exporter.get_finished_spans()
        # the live bug = these were ALL zero on real objects. Now they must populate.
        assert len(_kinds(spans, "agent")) == 1
        assert len(_kinds(spans, "llm")) == 2, "object-shaped assistant turns produced no LLM spans"
        assert len(_kinds(spans, "tool")) == 1, "object-shaped tool_use produced no TOOL span"
        llm = next(
            s
            for s in _kinds(spans, "llm")
            if s.attributes.get("neatlogs.llm.token_count.prompt") == 10
        )
        assert llm.attributes.get("neatlogs.llm.model_name") == "claude-haiku-4-5"
        assert llm.attributes.get("neatlogs.llm.tool_calls.0.name") == "triage_signal"
        tool = _kinds(spans, "tool")[0]
        assert "fired 322" in tool.attributes.get("output.value", "")  # closed by ToolResultBlock
        root = _kinds(spans, "agent")[0]
        assert root.attributes.get("neatlogs.conversation.id") == "s1"
        assert root.attributes.get("neatlogs.agent.cost_usd") == 0.0123
        c._unpatch_claude_agent_sdk()
        c._PATCHED = False


async def _drain(mod):
    async for _ in mod.query(prompt="hi"):
        pass


class TestQueryObjectShape:
    """The real SDK's query() returns a Query OBJECT (with its own methods/attrs + a permissive
    __getattr__), NOT a bare async generator. A wrapper that stored internal state via getattr/
    __getattr__ delegation broke iteration on the real object (live: only the first span emitted,
    the rest of the stream was lost). This guards that the wrapper iterates a Query-object correctly.
    """

    def test_iterates_query_object_with_permissive_getattr(
        self, in_memory_span_exporter, monkeypatch
    ):
        import sys
        import types

        class FakeQuery:
            """Mimics the SDK Query object: async-iterable + permissive __getattr__ (returns a no-op
            for any unknown attr, like a proxy) — the shape that broke the wrapper."""

            def __init__(self, msgs):
                self._msgs = msgs

            def __aiter__(self):
                async def gen():
                    for m in self._msgs:
                        yield m

                return gen()

            def __getattr__(self, name):
                # permissive proxy: any unknown attribute returns a callable no-op (this is what
                # made `getattr(self, "_inner_iter", None)` NOT return None in the buggy wrapper)
                return lambda *a, **k: None

        msgs = [
            {"type": "system", "session_id": "s", "model": "m"},
            {
                "type": "assistant",
                "parent_tool_use_id": None,
                "message": {
                    "content": [
                        {"type": "text", "text": "hi"},
                        {"type": "tool_use", "id": "t1", "name": "tool_x", "input": {}},
                    ],
                    "model": "m",
                    "usage": {"input_tokens": 5, "output_tokens": 2},
                },
            },
            {
                "type": "user",
                "parent_tool_use_id": None,
                "message": {
                    "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}]
                },
            },
            {
                "type": "assistant",
                "parent_tool_use_id": None,
                "message": {
                    "content": [{"type": "text", "text": "done"}],
                    "model": "m",
                    "usage": {"input_tokens": 6, "output_tokens": 1},
                },
            },
            {"type": "result", "result": "final", "usage": None},
        ]
        mod = types.ModuleType("claude_agent_sdk")
        mod.query = lambda *, prompt=None, options=None: FakeQuery(msgs)  # returns an OBJECT
        mod.ClaudeSDKClient = None
        monkeypatch.setitem(sys.modules, "claude_agent_sdk", mod)
        import neatlogs.claude_agent_sdk as c

        c._PATCHED = False
        c._ORIGINALS.clear()

        _setup_tracer(in_memory_span_exporter)
        c.wrap_claude_agent_sdk(mod)
        _run(_drain(mod))

        spans = in_memory_span_exporter.get_finished_spans()
        # the WHOLE stream must be traced (the bug lost everything after the first span)
        assert len(_kinds(spans, "agent")) == 1
        assert (
            len(_kinds(spans, "llm")) == 2
        ), "Query-object iteration dropped LLM spans (the live bug)"
        assert len(_kinds(spans, "tool")) == 1, "Query-object iteration dropped TOOL spans"
        c._unpatch_claude_agent_sdk()
        c._PATCHED = False
