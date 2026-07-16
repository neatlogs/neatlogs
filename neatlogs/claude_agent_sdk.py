"""
Neatlogs Claude Agent SDK integration.

Wraps Anthropic's ``claude_agent_sdk`` so every ``query()`` run is traced. The SDK's ``query()``
returns an async-iterable of messages (system init -> assistant (text + tool_use) -> user
(tool_result) -> ... -> result). Each message carries ``parent_tool_use_id``: None for the main
(orchestrator) agent, or the id of the spawning Task tool call for a subagent. We translate that
into a neatlogs span tree:

    AGENT  claude_agent.query                (orchestrator = trace root)
      |- LLM   orchestrator turn             (one per model turn; text + tool_calls)
      |- TOOL  Read / Edit / Bash ...
      |- TOOL  Task                          (spawns a subagent)
           |- AGENT  claude_agent.subagent.<type>   (each subagent, nested)
                |- LLM   subagent turn
                |- TOOL  subagent tool call

The orchestrator is the single root AGENT span -- there is NO redundant WORKFLOW wrapper (a single
query() is one agent run; subagents nest below as their own AGENT spans). Subagents (Task-tool
invocations) get their own AGENT spans nested under the Task TOOL span that spawned them.

This is a faithful 1:1 port of the TypeScript ``neatlogs-typescript/src/claude-agent-sdk.ts``;
attribute names are the canonical ones from ``neatlogs/config/attribute-mapping.json`` (every span
emits the flat ``input.value``/``output.value`` blobs that the UI panel renders, plus the indexed
``neatlogs.llm.input_messages.*``/``output_messages.*`` for the LLM drawer). Payloads are NOT
truncated client-side (the backend caps at 1MB with S3 offload). Fail-open: a tracing error never
breaks the agent run.

Enable via ``neatlogs.init(instrumentations=["claude_agent_sdk"])`` or ``neatlogs.wrap(module)``.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from opentelemetry import context as otel_context
from opentelemetry import trace as otel_trace
from opentelemetry.trace import StatusCode

from ._wrap_utils import attach_as_current, detach as _nl_detach, get_tracer, is_suppressed

_PROVIDER = "claude_agent_sdk"
_ROOT_SCOPE = "__root__"


class ClaudeAgentSDKInstrumentor:
    """Instrumentor class for InstrumentationManager integration."""

    def instrument(self, tracer_provider=None):
        _patch_claude_agent_sdk()

    def uninstrument(self):
        _unpatch_claude_agent_sdk()


def wrap_claude_agent_sdk(target: Any = None) -> Any:
    """Enable Neatlogs tracing for the Claude Agent SDK. Patches ``claude_agent_sdk.query`` (and the
    ``ClaudeSDKClient`` streaming methods) at module/class level so one call covers every query in the
    process. ``target`` (the module or a client) is accepted for the ``neatlogs.wrap(...)`` idiom and
    returned unchanged."""
    _patch_claude_agent_sdk()
    return target


# ---------------------------------------------------------------------------
# Module/class-level patches (idempotent, fail-open)
# ---------------------------------------------------------------------------

_PATCHED = False
_ORIGINALS: Dict[str, Any] = {}


def _patch_claude_agent_sdk() -> None:
    global _PATCHED
    if _PATCHED:
        return
    _PATCHED = True
    try:
        import claude_agent_sdk as cas
    except Exception:
        return  # SDK not installed — nothing to patch

    orig = getattr(cas, "query", None)
    if orig is not None and not getattr(orig, "_neatlogs_wrapped", False):
        _ORIGINALS["query"] = orig
        wrapper = _wrap_query(orig)
        wrapper._neatlogs_wrapped = True  # type: ignore[attr-defined]
        cas.query = wrapper

    # ClaudeSDKClient.receive_messages / receive_response are also async-iterables of the same
    # message types — wrap them the same way.
    client_cls = getattr(cas, "ClaudeSDKClient", None)
    if client_cls is not None:
        for meth in ("receive_messages", "receive_response"):
            m = getattr(client_cls, meth, None)
            if m is None or getattr(m, "_neatlogs_wrapped", False):
                continue
            _ORIGINALS[f"client.{meth}"] = m
            setattr(client_cls, meth, _wrap_client_method(m, meth))


def _unpatch_claude_agent_sdk() -> None:
    global _PATCHED
    if not _PATCHED:
        return
    try:
        import claude_agent_sdk as cas

        if "query" in _ORIGINALS:
            cas.query = _ORIGINALS["query"]
        client_cls = getattr(cas, "ClaudeSDKClient", None)
        if client_cls is not None:
            for meth in ("receive_messages", "receive_response"):
                key = f"client.{meth}"
                if key in _ORIGINALS:
                    setattr(client_cls, meth, _ORIGINALS[key])
    except Exception:
        pass
    _ORIGINALS.clear()
    _PATCHED = False


def _wrap_query(original):
    def query_wrapper(*args: Any, **kwargs: Any):
        if is_suppressed():
            return original(*args, **kwargs)
        # FAIL-OPEN: any error building the trace setup must fall back to the un-traced query — a
        # tracing failure can NEVER break the agent run.
        try:
            # query(prompt=..., options=...) — prompt may be a string or an async-iterable.
            prompt = kwargs.get("prompt") if "prompt" in kwargs else (args[0] if args else None)
            prompt_ref = {"text": ""}
            # If the prompt is a streaming async-iterable, tap it so we capture the first user text
            # as the SDK pulls it (it isn't echoed back as a `user` output message before turn 1).
            if prompt is not None and not isinstance(prompt, str) and _is_async_iterable(prompt):
                tapped = _tap_prompt_stream(prompt, prompt_ref)
                if "prompt" in kwargs:
                    kwargs = {**kwargs, "prompt": tapped}
                elif args:
                    args = (tapped,) + tuple(args[1:])
            elif isinstance(prompt, str):
                prompt_ref["text"] = prompt

            tracer = get_tracer()
            # The orchestrator AGENT span is the trace ROOT (no WORKFLOW wrapper).
            agent_span = tracer.start_span(
                "claude_agent.query",
                attributes={"neatlogs.span.kind": "agent", "neatlogs.agent.framework": _PROVIDER},
            )
            if prompt_ref["text"]:
                try:
                    agent_span.set_attribute("input.value", prompt_ref["text"])
                except Exception:
                    pass
            agent_ctx = otel_trace.set_span_in_context(agent_span)
            # Mode-aware: in DEFAULT mode makes the agent span the current-span so
            # nested SDK work nests under it; in ISOLATED mode threads the parent on
            # the private key WITHOUT touching the global current-span, so co-tenant
            # instrumentation (openlit) keeps nesting under the host.
            token = attach_as_current(agent_span)
            try:
                query_obj = original(*args, **kwargs)
            finally:
                _nl_detach(token)
            return _TracingQuery(query_obj, agent_span, agent_ctx, tracer, prompt_ref)
        except Exception:
            # tracing setup failed — run the original un-traced (never break the run)
            return original(*args, **kwargs)

    return query_wrapper


def _wrap_client_method(original, label: str):
    def wrapper(self, *args: Any, **kwargs: Any):
        inner = original(self, *args, **kwargs)
        if is_suppressed():
            return inner
        tracer = get_tracer()
        agent_span = tracer.start_span(
            f"claude_agent.{label}",
            attributes={"neatlogs.span.kind": "agent", "neatlogs.agent.framework": _PROVIDER},
        )
        agent_ctx = otel_trace.set_span_in_context(agent_span)
        return _TracingQuery(inner, agent_span, agent_ctx, tracer, {"text": ""})

    wrapper._neatlogs_wrapped = True  # type: ignore[attr-defined]
    return wrapper


# ---------------------------------------------------------------------------
# Per-agent scope + run state
# ---------------------------------------------------------------------------

class _Scope:
    """One agent's tracing scope. The root scope is the orchestrator (keyed _ROOT_SCOPE); each
    subagent (identified by the parent_tool_use_id of the Task call that spawned it) gets its own
    scope lazily, with its AGENT span nested under that Task TOOL span."""

    __slots__ = ("span", "ctx", "input_messages", "assistant_buffer", "final_text", "input_captured")

    def __init__(self, span, ctx, input_messages, input_captured):
        self.span = span
        self.ctx = ctx
        self.input_messages: List[Dict[str, str]] = input_messages
        self.assistant_buffer: Optional[Dict[str, Any]] = None  # in-progress model turn
        self.final_text: str = ""
        self.input_captured: bool = input_captured


class _TracingQuery:
    """Wraps the SDK's async-iterable query object: instruments iteration while preserving its own
    methods (interrupt, set_model, ...). Owns the span tree built from the message stream."""

    def __init__(self, query_obj, agent_span, agent_ctx, tracer, prompt_ref):
        self._q = query_obj
        self._tracer = tracer
        self._agent_span = agent_span
        self._agent_ctx = agent_ctx
        self._prompt_ref = prompt_ref
        root = _Scope(
            span=agent_span, ctx=agent_ctx,
            input_messages=([{"role": "user", "content": prompt_ref["text"]}] if prompt_ref["text"] else []),
            input_captured=bool(prompt_ref["text"]),
        )
        self._tool_spans: Dict[str, Any] = {}
        self._scopes: Dict[str, _Scope] = {_ROOT_SCOPE: root}
        self._session_id: Optional[str] = None
        self._model: Optional[str] = None
        self._finished = False
        self._inner_iter = None    # set on first __anext__ (NOT via __getattr__ — see below)

    # pass through the wrapped Query object's OWN methods (interrupt, set_model, …). CRITICAL: never
    # delegate our internal/dunder attrs (leading underscore) — otherwise `self._inner_iter` etc.
    # would resolve against self._q and break iteration. __getattr__ only fires for names NOT found
    # normally, so our real attributes (set in __init__) are always served from the instance.
    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self.__dict__["_q"], name)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._inner_iter is None:
            self._inner_iter = self._q.__aiter__()
        inner_iter = self._inner_iter
        # Mode-aware attach (see _wrap_query): never leaks the neatlogs agent span
        # onto the global current-span in isolated mode, so foreign instrumentation
        # emitted while the SDK pulls the next message nests under the host, not us.
        token = attach_as_current(self._agent_span)
        try:
            try:
                msg = await inner_iter.__anext__()
            except StopAsyncIteration:
                self._finalize("ok")
                raise
            except Exception as e:
                self._finalize("error", e)
                raise
        finally:
            _nl_detach(token)
        try:
            self._handle_message(msg)
        except Exception:
            pass  # never let tracing break the run
        return msg

    # ClaudeSDKClient streams may be used as async context managers
    async def __aenter__(self):
        if hasattr(self._q, "__aenter__"):
            self._q = await self._q.__aenter__()
        return self

    async def __aexit__(self, *exc):
        self._finalize("error" if exc and exc[1] else "ok", exc[1] if exc and exc[1] else None)
        if hasattr(self._q, "__aexit__"):
            return await self._q.__aexit__(*exc)
        return False

    # -- finalization ----------------------------------------------------------

    def _finalize(self, status: str, err: Any = None) -> None:
        if self._finished:
            return
        self._finished = True
        root = self._scopes[_ROOT_SCOPE]
        # flush every scope's in-progress turn
        for scope in self._scopes.values():
            self._flush_turn(scope)
        # close any tool spans that never got a matching result
        for ts in self._tool_spans.values():
            try:
                ts.end()
            except Exception:
                pass
        self._tool_spans.clear()
        # close subagent scopes first, then the root
        for key, scope in self._scopes.items():
            if key == _ROOT_SCOPE:
                continue
            self._close_scope(scope)
        if root.final_text:
            try:
                self._agent_span.set_attribute("output.value", root.final_text)
            except Exception:
                pass
        try:
            if status == "error":
                if isinstance(err, BaseException):
                    self._agent_span.set_status(StatusCode.ERROR, str(err))
                    self._agent_span.record_exception(err)
                else:
                    self._agent_span.set_status(StatusCode.ERROR)
            else:
                self._agent_span.set_status(StatusCode.OK)
            self._agent_span.end()
        except Exception:
            pass

    @staticmethod
    def _close_scope(scope: _Scope) -> None:
        try:
            if scope.final_text:
                scope.span.set_attribute("output.value", scope.final_text)
            scope.span.set_status(StatusCode.OK)
            scope.span.end()
        except Exception:
            pass

    # -- message handling ------------------------------------------------------

    def _get_scope(self, msg: Any) -> _Scope:
        """Resolve the scope for a message. parent_tool_use_id is None for the orchestrator and the
        spawning Task tool_use_id for a subagent. Subagent scopes are created lazily, AGENT span
        nested under the Task TOOL span (orchestrator -> Task TOOL -> subagent AGENT)."""
        parent_id = _get(msg, "parent_tool_use_id")
        if not parent_id:
            return self._scopes[_ROOT_SCOPE]
        existing = self._scopes.get(parent_id)
        if existing:
            return existing

        root = self._scopes[_ROOT_SCOPE]
        # the spawning Task tool_use may still be buffered in the root turn — flush so its TOOL span
        # exists and this subagent AGENT can nest under it.
        if parent_id not in self._tool_spans and root.assistant_buffer:
            self._flush_turn(root)

        parent_tool_span = self._tool_spans.get(parent_id)
        parent_ctx = (otel_trace.set_span_in_context(parent_tool_span)
                      if parent_tool_span is not None else root.ctx)

        sub_type = str(_get(msg, "subagent_type") or "subagent")
        attrs: Dict[str, Any] = {"neatlogs.span.kind": "agent", "neatlogs.agent.name": sub_type}
        task_desc = _get(msg, "task_description")
        if task_desc:
            attrs["input.value"] = str(task_desc)
        span = self._tracer.start_span(f"claude_agent.subagent.{sub_type}",
                                       attributes=attrs, context=parent_ctx)
        scope = _Scope(
            span=span, ctx=otel_trace.set_span_in_context(span),
            input_messages=([{"role": "user", "content": str(task_desc)}] if task_desc else []),
            input_captured=bool(task_desc),
        )
        self._scopes[parent_id] = scope
        return scope

    def _handle_message(self, msg: Any) -> None:
        if msg is None:
            return
        mtype = _msg_type(msg)
        root = self._scopes[_ROOT_SCOPE]

        # backfill root input from a tapped streaming prompt as soon as available
        if not root.input_captured and self._prompt_ref["text"]:
            root.input_captured = True
            try:
                root.span.set_attribute("input.value", self._prompt_ref["text"])
            except Exception:
                pass

        if mtype == "system":
            # SystemMessage (Python SDK) carries session_id/model under `.data`; the dict shape has
            # them flat. Read both.
            data = _get(msg, "data") if isinstance(_get(msg, "data"), dict) else {}
            sid = _get(msg, "session_id") or data.get("session_id")
            if sid:
                self._session_id = sid
                _safe_set(root.span, "neatlogs.conversation.id", str(sid))
            model = _get(msg, "model") or data.get("model")
            if model:
                self._model = model
                _safe_set(root.span, "neatlogs.agent.model", str(model))

        elif mtype == "user":
            scope = self._get_scope(msg)
            self._flush_turn(scope)
            user_text = _user_message_text(msg)
            if user_text:
                if not scope.input_captured:
                    scope.input_captured = True
                    _safe_set(scope.span, "input.value", user_text)
                scope.input_messages.append({"role": "user", "content": user_text})
            self._close_tool_spans_from_user(scope, msg)

        elif mtype == "assistant":
            scope = self._get_scope(msg)
            self._buffer_assistant(scope, msg)

        elif mtype == "result":
            self._flush_turn(root)
            result = _get(msg, "result")
            text = result if isinstance(result, str) else ""
            if text:
                root.final_text = text
            sid = _get(msg, "session_id")
            if sid and not self._session_id:
                _safe_set(root.span, "neatlogs.conversation.id", str(sid))
            usage = _get(msg, "usage")
            if usage:
                _set_usage(root.span, usage)
            cost = _get(msg, "total_cost_usd")
            if cost is not None:
                _safe_set(root.span, "neatlogs.agent.cost_usd", cost)
            num_turns = _get(msg, "num_turns")
            if num_turns is not None:
                _safe_set(root.span, "neatlogs.agent.num_turns", num_turns)
            if _get(msg, "is_error"):
                _safe_set(root.span, "neatlogs.agent.is_error", True)
            err = ValueError(str(text or "agent run failed")) if _get(msg, "is_error") else None
            self._finalize("error" if err else "ok", err)

    def _buffer_assistant(self, scope: _Scope, msg: Any) -> None:
        """Append one `assistant` message to the in-progress turn buffer. The SDK splits one model
        turn into multiple assistant messages (a text block, then tool_use blocks); they share usage.
        Merge text/thinking/tool_use so the turn becomes ONE LLM span on flush."""
        # Python AssistantMessage has content/model/usage/stop_reason DIRECTLY (no `.message`
        # wrapper); the dict/raw shape nests them under `.message`. Support both.
        message = _get(msg, "message") if _get(msg, "message") is not None else msg
        content = _get(message, "content")
        if content is None:
            content = _get(msg, "content") or []
        if scope.assistant_buffer is None:
            scope.assistant_buffer = {"text": [], "thinking": [], "tool_calls": [],
                                      "usage": None, "model": None, "stop_reason": None}
        buf = scope.assistant_buffer
        model = _get(message, "model") or _get(msg, "model")
        if model:
            buf["model"] = model
        stop = _get(message, "stop_reason") or _get(msg, "stop_reason")
        if stop:
            buf["stop_reason"] = stop
        usage = _get(message, "usage") or _get(msg, "usage")
        if usage:
            buf["usage"] = usage  # same turn total — keep last, don't sum
        for block in (content if isinstance(content, list) else []):
            bt = _block_type(block)
            if bt == "text" and isinstance(_get(block, "text"), str):
                buf["text"].append(_get(block, "text"))
            elif bt == "thinking" and isinstance(_get(block, "thinking"), str):
                buf["thinking"].append(_get(block, "thinking"))
            elif bt == "tool_use":
                buf["tool_calls"].append({"id": _get(block, "id") or "",
                                          "name": _get(block, "name") or "",
                                          "input": _get(block, "input")})

    def _flush_turn(self, scope: _Scope) -> None:
        """Emit the buffered model turn as ONE LLM span (nested under this scope's AGENT span), then
        open TOOL spans for its tool calls. No-op if nothing buffered."""
        buf = scope.assistant_buffer
        if not buf:
            return
        scope.assistant_buffer = None
        model = buf["model"] or self._model or ""

        attrs: Dict[str, Any] = {
            "neatlogs.span.kind": "llm",
            "neatlogs.llm.provider": "anthropic",
            "neatlogs.llm.system": "anthropic",
        }
        if model:
            attrs["neatlogs.llm.model_name"] = str(model)

        # seed root input from tapped prompt if this is the first turn with no user msg yet
        if not scope.input_messages and self._prompt_ref["text"]:
            scope.input_messages.append({"role": "user", "content": self._prompt_ref["text"]})

        # input = the exact accumulated conversation up to this turn (indexed + flat blob).
        for i, m in enumerate(scope.input_messages):
            attrs[f"neatlogs.llm.input_messages.{i}.role"] = m["role"]
            attrs[f"neatlogs.llm.input_messages.{i}.content"] = m["content"]
        if scope.input_messages:
            attrs["input.value"] = _safe_json({"messages": scope.input_messages})

        out_text = "".join(buf["text"])
        out_value = out_text or "\n".join(
            f"{tc['name']}({_safe_json(tc['input'] or {})})" for tc in buf["tool_calls"]
        )
        if out_value:
            attrs["neatlogs.llm.output_messages.0.role"] = "assistant"
            attrs["neatlogs.llm.output_messages.0.content"] = out_value
            attrs["output.value"] = out_value
        if buf["thinking"]:
            attrs["neatlogs.llm.output_messages.0.thinking"] = "".join(buf["thinking"])
        for j, tc in enumerate(buf["tool_calls"]):
            attrs[f"neatlogs.llm.tool_calls.{j}.id"] = tc["id"]
            attrs[f"neatlogs.llm.tool_calls.{j}.name"] = tc["name"]
            attrs[f"neatlogs.llm.tool_calls.{j}.arguments"] = _safe_json(tc["input"] or {})
        if buf["stop_reason"]:
            attrs["neatlogs.llm.finish_reason"] = str(buf["stop_reason"])

        span = self._tracer.start_span(f"claude_agent.llm.{model or 'model'}",
                                       attributes=attrs, context=scope.ctx)
        if buf["usage"]:
            _set_usage(span, buf["usage"])
        span.set_status(StatusCode.OK)
        span.end()

        if out_text:
            scope.final_text = out_text

        # record this turn so the NEXT turn's LLM span sees it as context
        turn_parts: List[str] = []
        if out_text:
            turn_parts.append(out_text)
        for tc in buf["tool_calls"]:
            turn_parts.append(f"[tool_call {tc['name']} {_safe_json(tc['input'] or {})}]")
        if turn_parts:
            scope.input_messages.append({"role": "assistant", "content": "\n".join(turn_parts)})

        # open TOOL spans (closed by their tool_result); a Task tool's id becomes a subagent key
        for tc in buf["tool_calls"]:
            t_attrs = {"neatlogs.span.kind": "tool", "neatlogs.tool.name": str(tc["name"] or ""),
                       "input.value": _safe_json(tc["input"] or {})}
            if tc["id"]:
                t_attrs["neatlogs.tool_call.id"] = str(tc["id"])
            tool_span = self._tracer.start_span(f"claude_agent.tool.{tc['name'] or 'tool'}",
                                                attributes=t_attrs, context=scope.ctx)
            if tc["id"]:
                self._tool_spans[tc["id"]] = tool_span

    def _close_tool_spans_from_user(self, scope: _Scope, msg: Any) -> None:
        # UserMessage.content is DIRECTLY on the object (Python SDK); dict shape nests under .message
        message = _get(msg, "message") if _get(msg, "message") is not None else msg
        content = _get(message, "content")
        if content is None:
            content = _get(msg, "content") or []
        for block in (content if isinstance(content, list) else []):
            if _block_type(block) != "tool_result":
                continue
            tid = _get(block, "tool_use_id") or ""
            out = _get(block, "content")
            out_text = out if isinstance(out, str) else _safe_json(out)
            if out_text:
                scope.input_messages.append({"role": "tool", "content": out_text})
            span = self._tool_spans.get(tid)
            if span is None:
                continue
            try:
                span.set_attribute("output.value", out_text)
                if _get(block, "is_error"):
                    span.set_status(StatusCode.ERROR)
                    span.set_attribute("neatlogs.tool.is_error", True)
                else:
                    span.set_status(StatusCode.OK)
                span.end()
            except Exception:
                pass
            self._tool_spans.pop(tid, None)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CLASS_TO_TYPE = {
    "SystemMessage": "system",
    "AssistantMessage": "assistant",
    "UserMessage": "user",
    "ResultMessage": "result",
}


def _msg_type(msg: Any) -> str:
    """Normalize a message to one of system/assistant/user/result. The Python claude_agent_sdk uses
    typed CLASSES (AssistantMessage, …) with NO `type` field; the dict/raw shape (and our tests) use
    a `type` key. Support both: prefer an explicit dict `type`, else map the class name."""
    if isinstance(msg, dict):
        t = msg.get("type")
        if t:
            return t
    return _CLASS_TO_TYPE.get(type(msg).__name__, "")


def _block_type(block: Any) -> str:
    """A content block's type — from a dict `type` key OR the Python block class name
    (TextBlock/ThinkingBlock/ToolUseBlock/ToolResultBlock)."""
    if isinstance(block, dict):
        return block.get("type") or ""
    name = type(block).__name__
    return {"TextBlock": "text", "ThinkingBlock": "thinking",
            "ToolUseBlock": "tool_use", "ToolResultBlock": "tool_result"}.get(name, "")


def _is_async_iterable(obj: Any) -> bool:
    return hasattr(obj, "__aiter__")


async def _tap_prompt_stream(prompt: Any, ref: Dict[str, str]):
    """Pass-through wrapper over a streaming-input prompt that records the first user message's text
    into ``ref`` as the SDK consumes it. Never alters what the SDK receives."""
    async for message in prompt:
        if not ref["text"]:
            text = _user_message_text(message)
            if text:
                ref["text"] = text
        yield message


def _get(obj: Any, key: str) -> Any:
    """Read a field from either a dict-shaped message or an attribute-shaped object."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _user_message_text(msg: Any) -> str:
    """Text content of a `user` message — the prompt, not tool_result blocks. Handles the Python SDK
    object (content directly on the message) and the dict shape (content under `.message`)."""
    message = _get(msg, "message") if _get(msg, "message") is not None else msg
    content = _get(message, "content")
    if content is None:
        content = _get(msg, "content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: List[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif _block_type(block) == "text" and isinstance(_get(block, "text"), str):
            parts.append(_get(block, "text"))
    return "\n".join(parts)


def _set_usage(span: Any, usage: Any) -> None:
    """Canonical token-count attributes (same keys as anthropic.py). Accepts a dict or object."""
    inp = _get(usage, "input_tokens")
    out = _get(usage, "output_tokens")
    if inp is not None:
        _safe_set(span, "neatlogs.llm.token_count.prompt", inp)
    if out is not None:
        _safe_set(span, "neatlogs.llm.token_count.completion", out)
    if isinstance(inp, (int, float)) and isinstance(out, (int, float)):
        _safe_set(span, "neatlogs.llm.token_count.total", inp + out)
    cr = _get(usage, "cache_read_input_tokens")
    if cr is not None:
        _safe_set(span, "neatlogs.llm.token_count.cache_read", cr)
    cw = _get(usage, "cache_creation_input_tokens")
    if cw is not None:
        _safe_set(span, "neatlogs.llm.token_count.cache_write", cw)


def _safe_set(span: Any, key: str, value: Any) -> None:
    try:
        span.set_attribute(key, value)
    except Exception:
        pass


def _safe_json(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, default=str, ensure_ascii=False)
    except Exception:
        return ""
