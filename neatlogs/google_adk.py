"""
Neatlogs Google ADK wrapper.

Usage:
    >>> import neatlogs
    >>> from google.adk.runners import Runner
    >>> runner = neatlogs.wrap(Runner(agent=my_agent, app_name="my_app"))
    >>> # runner.run() / runner.run_async() are now traced
"""

import time
from typing import Any

from opentelemetry import context as otel_context
from opentelemetry.trace import StatusCode

from ._wrap_utils import attach_as_current, detach, get_tracer, serialize


_ADK_HOOKS_INSTALLED = False

# The inner `generate_content` span (trace_inference_result) only receives the
# response, not the request. We stash the most recent request's structured messages
# in a plain module global so the inner span can reuse them. Calls are effectively
# sequential per agent turn, and we set it both at use_inference_span entry (which
# DOES receive llm_request) and in trace_call_llm — whichever fires first wins.
_LAST_REQUEST_MESSAGES: list = []


def wrap_google_adk(runner: Any) -> Any:
    """
    Wrap a Google ADK Runner instance. Patches run() and run_async() to emit the
    user-facing WORKFLOW span, and installs hooks on ADK's own telemetry helpers
    so its native per-call spans carry neatlogs I/O + kind.
    Returns the same runner instance.
    """
    _install_adk_telemetry_hooks()

    if getattr(runner, "_neatlogs_patched", False):
        return runner

    _patch_run(runner)
    _patch_run_async(runner)
    return runner


def _install_adk_telemetry_hooks() -> None:
    """
    ADK self-instruments (like the Vercel AI SDK): it emits its own OTel spans
    (`call_llm`, `execute_tool`, `invoke_agent`, …) via google.adk.telemetry.tracing,
    classifying them with the gen_ai.* semantic conventions and stashing request/
    response content under `gcp.vertex.agent.*`. neatlogs' generic mapper reads
    `input.value`/`output.value` + `neatlogs.span.kind`, which ADK does NOT set — so
    those native spans would show no I/O and no kind.

    Rather than teach the shared processor about ADK's vendor-specific attributes,
    we map them HERE, in the ADK wrapper: wrap ADK's telemetry helpers so that after
    ADK populates its own span we additionally set the neatlogs keys on the SAME span
    from the request/response objects ADK already hands us. All ADK-specific knowledge
    stays in this module.
    """
    global _ADK_HOOKS_INSTALLED
    if _ADK_HOOKS_INSTALLED:
        return

    try:
        from google.adk.telemetry import tracing as adk_tracing
    except Exception:
        return

    from opentelemetry import trace as otel_trace

    # --- LLM calls: trace_call_llm(invocation_context, event_id, llm_request, llm_response, span=None)
    orig_call_llm = getattr(adk_tracing, "trace_call_llm", None)
    if orig_call_llm is not None and not getattr(orig_call_llm, "_neatlogs_wrapped", False):
        def patched_trace_call_llm(*args, **kwargs):
            orig_call_llm(*args, **kwargs)
            try:
                # positional: (invocation_context, event_id, llm_request, llm_response, span?)
                llm_request = args[2] if len(args) > 2 else kwargs.get("llm_request")
                llm_response = args[3] if len(args) > 3 else kwargs.get("llm_response")
                span = (args[4] if len(args) > 4 else kwargs.get("span")) or otel_trace.get_current_span()
                if span is not None:
                    span.set_attribute("neatlogs.span.kind", "llm")
                    # Emit PROPERLY-ROLED indexed messages (system / user / assistant /
                    # tool), like the other wrappers — NOT one flattened blob. The backend
                    # LLM finalizer reads neatlogs.llm.input_messages.<i>.role/content per
                    # role; a single jumbled string makes it mis-extract (e.g. "City: Paris").
                    msgs = _adk_request_messages(llm_request)
                    for i, m in enumerate(msgs):
                        span.set_attribute(f"neatlogs.llm.input_messages.{i}.role", m["role"])
                        span.set_attribute(f"neatlogs.llm.input_messages.{i}.content", m["content"])
                    if msgs:
                        span.set_attribute("input.value", serialize({"messages": msgs}))
                        # Stash for the inner generate_content span (same turn).
                        global _LAST_REQUEST_MESSAGES
                        _LAST_REQUEST_MESSAGES = msgs
                    out_text = _adk_llm_response_text(llm_response)
                    if out_text:
                        span.set_attribute("output.value", out_text)
                        span.set_attribute("neatlogs.llm.output_messages.0.role", "assistant")
                        span.set_attribute("neatlogs.llm.output_messages.0.content", out_text)
            except Exception:
                pass
        patched_trace_call_llm._neatlogs_wrapped = True
        _rebind_adk_symbol("trace_call_llm", patched_trace_call_llm, adk_tracing)

    # --- use_inference_span(llm_request, ctx, event): async CM that opens the inner
    #     `generate_content` span. It receives llm_request FIRST (before
    #     trace_inference_result fires), so stash the request messages here to avoid a
    #     first-turn race where the inner span has no input yet.
    orig_use_inf = getattr(adk_tracing, "use_inference_span", None)
    if orig_use_inf is not None and not getattr(orig_use_inf, "_neatlogs_wrapped", False):
        def patched_use_inference_span(*args, **kwargs):
            try:
                llm_request = args[0] if args else kwargs.get("llm_request")
                msgs = _adk_request_messages(llm_request)
                if msgs:
                    global _LAST_REQUEST_MESSAGES
                    _LAST_REQUEST_MESSAGES = msgs
            except Exception:
                pass
            return orig_use_inf(*args, **kwargs)
        patched_use_inference_span._neatlogs_wrapped = True
        _rebind_adk_symbol("use_inference_span", patched_use_inference_span, adk_tracing)

    # --- Inference result on the inner `generate_content` span:
    #     trace_inference_result(span, llm_response). ADK creates BOTH a `call_llm`
    #     (outer) and a `generate_content` (inner) LLM span per call; the inner one
    #     only carries tokens + a gen_ai.choice EVENT (no I/O attribute), so it would
    #     render empty. Enrich it with the same neatlogs I/O so neither LLM span is blank.
    orig_inf_result = getattr(adk_tracing, "trace_inference_result", None)
    if orig_inf_result is not None and not getattr(orig_inf_result, "_neatlogs_wrapped", False):
        def patched_trace_inference_result(*args, **kwargs):
            orig_inf_result(*args, **kwargs)
            try:
                span = args[0] if args else kwargs.get("span")
                # ADK may pass a GenerateContentSpan wrapper — unwrap to the real
                # OTel span (`.span`); set_attribute on the wrapper would no-op.
                inner = getattr(span, "span", None)
                if inner is not None:
                    span = inner
                llm_response = args[1] if len(args) > 1 else kwargs.get("llm_response")
                # Skip partial streaming chunks — wait for the final/non-partial one.
                if llm_response is not None and getattr(llm_response, "partial", False):
                    return
                if span is not None:
                    span.set_attribute("neatlogs.span.kind", "llm")
                    # input: reuse the request captured on the parent call_llm turn.
                    in_msgs = _LAST_REQUEST_MESSAGES
                    if in_msgs:
                        for i, m in enumerate(in_msgs):
                            span.set_attribute(f"neatlogs.llm.input_messages.{i}.role", m["role"])
                            span.set_attribute(f"neatlogs.llm.input_messages.{i}.content", m["content"])
                        span.set_attribute("input.value", serialize({"messages": in_msgs}))
                    out_text = _adk_llm_response_text(llm_response)
                    if out_text:
                        span.set_attribute("output.value", out_text)
                        span.set_attribute("neatlogs.llm.output_messages.0.role", "assistant")
                        span.set_attribute("neatlogs.llm.output_messages.0.content", out_text)
            except Exception:
                pass
        patched_trace_inference_result._neatlogs_wrapped = True
        _rebind_adk_symbol("trace_inference_result", patched_trace_inference_result, adk_tracing)

    # --- Tool calls: trace_tool_call(tool, args, function_response_event)
    orig_tool_call = getattr(adk_tracing, "trace_tool_call", None)
    if orig_tool_call is not None and not getattr(orig_tool_call, "_neatlogs_wrapped", False):
        def patched_trace_tool_call(*args, **kwargs):
            orig_tool_call(*args, **kwargs)
            try:
                span = otel_trace.get_current_span()
                tool_args = args[1] if len(args) > 1 else kwargs.get("args")
                resp_event = args[2] if len(args) > 2 else kwargs.get("function_response_event")
                if span is not None:
                    span.set_attribute("neatlogs.span.kind", "tool")
                    if tool_args is not None:
                        span.set_attribute("input.value", serialize(tool_args))
                    out_text = _adk_tool_response_text(resp_event)
                    if out_text:
                        span.set_attribute("output.value", out_text)
            except Exception:
                pass
        patched_trace_tool_call._neatlogs_wrapped = True
        _rebind_adk_symbol("trace_tool_call", patched_trace_tool_call, adk_tracing)

    # --- Agent invocation: trace_agent_invocation(span, agent, ctx)
    orig_agent_inv = getattr(adk_tracing, "trace_agent_invocation", None)
    if orig_agent_inv is not None and not getattr(orig_agent_inv, "_neatlogs_wrapped", False):
        def patched_trace_agent_invocation(*args, **kwargs):
            orig_agent_inv(*args, **kwargs)
            try:
                span = args[0] if args else kwargs.get("span")
                agent = args[1] if len(args) > 1 else kwargs.get("agent")
                ctx = args[2] if len(args) > 2 else kwargs.get("ctx")
                if span is not None:
                    span.set_attribute("neatlogs.span.kind", "agent")
                    # Agent system prompt = the agent's instruction.
                    instruction = getattr(agent, "instruction", None)
                    if instruction and isinstance(instruction, str):
                        span.set_attribute("neatlogs.llm.input_messages.0.role", "system")
                        span.set_attribute("neatlogs.llm.input_messages.0.content", instruction)
                    # User input for this invocation.
                    user_text = _content_to_text(getattr(ctx, "user_content", None)) if ctx is not None else ""
                    if user_text:
                        span.set_attribute("input.value", user_text)
                    name = getattr(agent, "name", None)
                    if name:
                        span.set_attribute("neatlogs.agent.name", str(name))
            except Exception:
                pass
        patched_trace_agent_invocation._neatlogs_wrapped = True
        _rebind_adk_symbol("trace_agent_invocation", patched_trace_agent_invocation, adk_tracing)

    _ADK_HOOKS_INSTALLED = True


def _rebind_adk_symbol(name: str, patched: Any, adk_tracing: Any) -> None:
    """
    Rebind a telemetry helper everywhere it's used. ADK's flow modules do
    `from ...telemetry.tracing import trace_call_llm` (a direct name import), so
    patching only `adk_tracing.<name>` leaves those already-imported references
    pointing at the original. Patch the source module AND every loaded module
    that imported the symbol by value.
    """
    import sys

    setattr(adk_tracing, name, patched)
    for mod in list(sys.modules.values()):
        if mod is None:
            continue
        mod_name = getattr(mod, "__name__", "")
        if not mod_name.startswith("google.adk"):
            continue
        if getattr(mod, name, None) is not None and mod is not adk_tracing:
            try:
                setattr(mod, name, patched)
            except Exception:
                pass


def _content_to_text(content: Any) -> str:
    """Flatten a google.genai Content (or list) into readable text + tool calls."""
    if content is None:
        return ""
    items = content if isinstance(content, list) else [content]
    out = []
    for c in items:
        parts = getattr(c, "parts", None) or []
        for p in parts:
            text = getattr(p, "text", None)
            if text:
                out.append(text)
                continue
            fc = getattr(p, "function_call", None)
            if fc is not None:
                out.append(f"{getattr(fc, 'name', '')}({serialize(dict(getattr(fc, 'args', {}) or {}))})")
                continue
            fr = getattr(p, "function_response", None)
            if fr is not None:
                out.append(serialize(getattr(fr, "response", None)))
    return "\n".join(s for s in out if s)


def _adk_request_messages(llm_request: Any) -> list:
    """
    Structured, per-role messages from an ADK LlmRequest:
    [{role: system, content}, {role: user/model/tool, content}, ...].
    ADK's google.genai content roles are 'user' and 'model'; map 'model'→'assistant'.
    """
    if llm_request is None:
        return []
    out = []
    config = getattr(llm_request, "config", None)
    sys_inst = getattr(config, "system_instruction", None) if config is not None else None
    if sys_inst:
        sys_text = sys_inst if isinstance(sys_inst, str) else _content_to_text(sys_inst)
        if sys_text:
            out.append({"role": "system", "content": sys_text})
    contents = getattr(llm_request, "contents", None) or []
    if not isinstance(contents, list):
        contents = [contents]
    for c in contents:
        role = getattr(c, "role", None) or "user"
        if role == "model":
            role = "assistant"
        # google.genai sends BOTH user turns AND tool/function responses with
        # role "user". Re-label function-response turns as "tool" so the backend's
        # "last user message" heuristic picks the real user question, not the tool
        # result (which would render as e.g. "City: Paris").
        parts = getattr(c, "parts", None) or []
        has_fn_response = any(getattr(p, "function_response", None) is not None for p in parts)
        has_fn_call = any(getattr(p, "function_call", None) is not None for p in parts)
        if has_fn_response:
            role = "tool"
        elif has_fn_call and role == "user":
            role = "assistant"
        text = _content_to_text(c)
        if text:
            out.append({"role": role, "content": text})
    return out


def _adk_llm_response_text(llm_response: Any) -> str:
    if llm_response is None:
        return ""
    return _content_to_text(getattr(llm_response, "content", None))


def _adk_tool_response_text(resp_event: Any) -> str:
    if resp_event is None:
        return ""
    return _content_to_text(getattr(resp_event, "content", None))


def _get_runner_attributes(runner: Any) -> dict:
    """Extract runner metadata as span attributes."""
    attrs = {"neatlogs.span.kind": "workflow"}

    app_name = getattr(runner, "app_name", None)
    if app_name:
        attrs["neatlogs.workflow.name"] = app_name

    agent = getattr(runner, "agent", None)
    if agent:
        agent_name = getattr(agent, "name", None)
        if agent_name:
            attrs["neatlogs.agent.name"] = agent_name
        model = getattr(agent, "model", None)
        if model:
            attrs["neatlogs.llm.model_name"] = str(model)

    return attrs


def _collect_events(events) -> tuple:
    """Consume a generator/iterator of events, collecting them and extracting attributes."""
    collected = []
    total_input_tokens = 0
    total_output_tokens = 0
    last_content = None
    tool_calls = []
    author = None

    for event in events:
        collected.append(event)

        content = getattr(event, "content", None)
        if content:
            last_content = content

        event_author = getattr(event, "author", None)
        if event_author:
            author = event_author

        # Token usage from event actions
        actions = getattr(event, "actions", None)
        if actions:
            for action in (actions if isinstance(actions, list) else [actions]):
                usage = getattr(action, "usage_metadata", None) or getattr(action, "usage", None)
                if usage:
                    input_t = getattr(usage, "prompt_token_count", None) or getattr(usage, "input_tokens", 0)
                    output_t = getattr(usage, "candidates_token_count", None) or getattr(usage, "output_tokens", 0)
                    if input_t:
                        total_input_tokens += input_t
                    if output_t:
                        total_output_tokens += output_t

        # Tool use from function calls
        parts = None
        if content and hasattr(content, "parts"):
            parts = content.parts
        elif hasattr(event, "parts"):
            parts = event.parts

        if parts:
            for part in parts:
                fn_call = getattr(part, "function_call", None)
                if fn_call:
                    tool_calls.append({
                        "name": getattr(fn_call, "name", ""),
                        "arguments": serialize(getattr(fn_call, "args", {})),
                    })

    attrs = {}
    if total_input_tokens:
        attrs["neatlogs.llm.token_count.prompt"] = total_input_tokens
    if total_output_tokens:
        attrs["neatlogs.llm.token_count.completion"] = total_output_tokens
    if total_input_tokens and total_output_tokens:
        attrs["neatlogs.llm.token_count.total"] = total_input_tokens + total_output_tokens

    if last_content:
        text_parts = []
        if hasattr(last_content, "parts"):
            for part in last_content.parts:
                text = getattr(part, "text", None)
                if text:
                    text_parts.append(text)
        if text_parts:
            attrs["output.value"] = "\n".join(text_parts)

    if author:
        attrs["neatlogs.agent.name"] = str(author)

    for i, tc in enumerate(tool_calls):
        attrs[f"neatlogs.llm.tool_calls.{i}.name"] = tc["name"]
        attrs[f"neatlogs.llm.tool_calls.{i}.arguments"] = tc["arguments"]

    return collected, attrs


async def _collect_events_async(events) -> tuple:
    """Async version of _collect_events."""
    collected = []
    total_input_tokens = 0
    total_output_tokens = 0
    last_content = None
    tool_calls = []
    author = None

    async for event in events:
        collected.append(event)

        content = getattr(event, "content", None)
        if content:
            last_content = content

        event_author = getattr(event, "author", None)
        if event_author:
            author = event_author

        actions = getattr(event, "actions", None)
        if actions:
            for action in (actions if isinstance(actions, list) else [actions]):
                usage = getattr(action, "usage_metadata", None) or getattr(action, "usage", None)
                if usage:
                    input_t = getattr(usage, "prompt_token_count", None) or getattr(usage, "input_tokens", 0)
                    output_t = getattr(usage, "candidates_token_count", None) or getattr(usage, "output_tokens", 0)
                    if input_t:
                        total_input_tokens += input_t
                    if output_t:
                        total_output_tokens += output_t

        parts = None
        if content and hasattr(content, "parts"):
            parts = content.parts
        elif hasattr(event, "parts"):
            parts = event.parts

        if parts:
            for part in parts:
                fn_call = getattr(part, "function_call", None)
                if fn_call:
                    tool_calls.append({
                        "name": getattr(fn_call, "name", ""),
                        "arguments": serialize(getattr(fn_call, "args", {})),
                    })

    attrs = {}
    if total_input_tokens:
        attrs["neatlogs.llm.token_count.prompt"] = total_input_tokens
    if total_output_tokens:
        attrs["neatlogs.llm.token_count.completion"] = total_output_tokens
    if total_input_tokens and total_output_tokens:
        attrs["neatlogs.llm.token_count.total"] = total_input_tokens + total_output_tokens

    if last_content:
        text_parts = []
        if hasattr(last_content, "parts"):
            for part in last_content.parts:
                text = getattr(part, "text", None)
                if text:
                    text_parts.append(text)
        if text_parts:
            attrs["output.value"] = "\n".join(text_parts)

    if author:
        attrs["neatlogs.agent.name"] = str(author)

    for i, tc in enumerate(tool_calls):
        attrs[f"neatlogs.llm.tool_calls.{i}.name"] = tc["name"]
        attrs[f"neatlogs.llm.tool_calls.{i}.arguments"] = tc["arguments"]

    return collected, attrs


def _patch_run(runner: Any) -> None:
    """Patch Runner.run() (synchronous generator)."""
    if not hasattr(runner, "run"):
        return

    orig_run = runner.run

    def patched_run(*args, **kwargs):
        tracer = get_tracer()
        attrs = _get_runner_attributes(runner)

        user_id = kwargs.get("user_id")
        if user_id:
            attrs["neatlogs.user.id"] = str(user_id)
        session_id = kwargs.get("session_id")
        if session_id:
            attrs["neatlogs.session.id"] = str(session_id)

        new_message = kwargs.get("new_message")
        if new_message:
            if hasattr(new_message, "parts"):
                text_parts = [getattr(p, "text", "") for p in new_message.parts if getattr(p, "text", None)]
                if text_parts:
                    attrs["input.value"] = "\n".join(text_parts)
            else:
                attrs["input.value"] = str(new_message)

        span = tracer.start_span(name="google_adk.runner.run", attributes=attrs)
        token = attach_as_current(span)
        start = time.perf_counter()

        try:
            events = orig_run(*args, **kwargs)
            collected, event_attrs = _collect_events(events)
        except Exception as e:
            span.set_status(StatusCode.ERROR, str(e))
            span.record_exception(e)
            span.end()
            raise
        finally:
            detach(token)

        duration_ms = (time.perf_counter() - start) * 1000
        for attr_name, value in event_attrs.items():
            span.set_attribute(attr_name, value)
        span.set_attribute("neatlogs.llm.metrics.duration_ms", round(duration_ms, 3))
        span.set_status(StatusCode.OK)
        span.end()

        return iter(collected)

    runner.run = patched_run
    runner._neatlogs_patched = True


def _patch_run_async(runner: Any) -> None:
    """Patch Runner.run_async() (async generator)."""
    if not hasattr(runner, "run_async"):
        return

    orig_run_async = runner.run_async

    # Must itself be an async GENERATOR (use yield), because callers consume
    # run_async() with `async for`. A plain `async def` would return a coroutine
    # and break iteration. We stream events through, then finalize the span.
    async def patched_run_async(*args, **kwargs):
        tracer = get_tracer()
        attrs = _get_runner_attributes(runner)

        user_id = kwargs.get("user_id")
        if user_id:
            attrs["neatlogs.user.id"] = str(user_id)
        session_id = kwargs.get("session_id")
        if session_id:
            attrs["neatlogs.session.id"] = str(session_id)

        new_message = kwargs.get("new_message")
        if new_message:
            if hasattr(new_message, "parts"):
                text_parts = [getattr(p, "text", "") for p in new_message.parts if getattr(p, "text", None)]
                if text_parts:
                    attrs["input.value"] = "\n".join(text_parts)
            else:
                attrs["input.value"] = str(new_message)

        span = tracer.start_span(name="google_adk.runner.run_async", attributes=attrs)
        token = attach_as_current(span)
        start = time.perf_counter()

        collected = []
        try:
            # Stream events to the caller as they arrive (keeps run_async an async
            # generator), buffering them so we can extract span attrs at the end.
            async for event in orig_run_async(*args, **kwargs):
                collected.append(event)
                yield event
        except Exception as e:
            span.set_status(StatusCode.ERROR, str(e))
            span.record_exception(e)
            span.end()
            raise
        finally:
            detach(token)

        duration_ms = (time.perf_counter() - start) * 1000
        # Reuse the sync extractor over the buffered events (no awaiting needed).
        _, event_attrs = _collect_events(iter(collected))
        for attr_name, value in event_attrs.items():
            span.set_attribute(attr_name, value)
        span.set_attribute("neatlogs.llm.metrics.duration_ms", round(duration_ms, 3))
        span.set_status(StatusCode.OK)
        span.end()

    runner.run_async = patched_run_async
