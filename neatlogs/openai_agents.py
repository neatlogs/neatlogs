"""
Neatlogs OpenAI Agents SDK trace processor.

Usage:
    >>> import neatlogs
    >>> from agents import add_trace_processor
    >>> add_trace_processor(neatlogs.openai_agents_processor())

Maps the OpenAI Agents SDK tracing protocol to neatlogs OTel spans:

    WORKFLOW   trace
      ↳ AGENT       agent span
      ↳ LLM         generation / response span
      ↳ TOOL        function span
      ↳ AGENT       handoff span
      ↳ GUARDRAIL   guardrail span
      ↳ TOOL        mcp_tools (MCP list-tools) span
      ↳ LLM         speech / transcription span
      ↳ CHAIN       custom / unknown span

Parent-child nesting follows the SDK's span.parent_id (falling back to the
trace), and each span is attached as the active OTel span so user @span /
trace() / log() calls nest correctly too.
"""

import time
from typing import Any, Dict

from opentelemetry.trace import StatusCode

from ._wrap_utils import attach_as_current, detach, get_tracer, serialize


def openai_agents_processor():
    """Return a trace processor for the OpenAI Agents SDK."""
    return _NeatlogsTraceProcessor()


class _NeatlogsTraceProcessor:
    """
    Implements the OpenAI Agents SDK TracingProcessor protocol:
      on_trace_start(trace) / on_trace_end(trace)
      on_span_start(span)   / on_span_end(span)

    The SDK passes a Trace (name, trace_id) and Span objects (span_id,
    parent_id, trace_id, span_data, error). The meaningful payload lives on
    span.span_data, whose ``.type`` discriminates the kind.
    """

    def __init__(self):
        self._spans: Dict[str, Any] = {}
        self._tokens: Dict[str, Any] = {}
        self._start_times: Dict[str, float] = {}

    # -- trace lifecycle -------------------------------------------------------

    def on_trace_start(self, trace: Any) -> None:
        tracer = get_tracer()
        attrs = {"neatlogs.span.kind": "workflow"}
        workflow_name = getattr(trace, "name", None) or getattr(trace, "workflow_name", None)
        if workflow_name:
            attrs["neatlogs.workflow.name"] = workflow_name
        trace_id = getattr(trace, "trace_id", None)
        if trace_id:
            attrs["neatlogs.agent.trace_id"] = str(trace_id)

        span = tracer.start_span(name="openai_agents.trace", attributes=attrs)
        token = attach_as_current(span)
        key = str(trace_id or id(trace))
        self._spans[key] = span
        self._tokens[key] = token
        self._start_times[key] = time.perf_counter()

    def on_trace_end(self, trace: Any) -> None:
        key = str(getattr(trace, "trace_id", id(trace)))
        span = self._spans.pop(key, None)
        token = self._tokens.pop(key, None)
        start = self._start_times.pop(key, None)
        if not span:
            return
        if token:
            detach(token)
        if start:
            span.set_attribute("neatlogs.llm.metrics.duration_ms", round((time.perf_counter() - start) * 1000, 3))
        span.set_status(StatusCode.OK)
        span.end()

    # -- span lifecycle --------------------------------------------------------

    def on_span_start(self, span: Any) -> None:
        tracer = get_tracer()
        data = getattr(span, "span_data", None) or span
        span_type = getattr(data, "type", None) or getattr(data, "span_type", "") or ""
        span_id = str(getattr(span, "span_id", id(span)))

        # Parent context: nest under parent span (or the trace).
        parent_key = str(getattr(span, "parent_id", None) or getattr(span, "trace_id", "") or "")
        parent_span = self._spans.get(parent_key)
        attrs, name = _build_start_attrs(span_type, data)

        if parent_span is not None:
            from opentelemetry import trace as _ot
            from opentelemetry import context as _ctx
            otel_span = tracer.start_span(name=name, attributes=attrs, context=_ot.set_span_in_context(parent_span))
        else:
            otel_span = tracer.start_span(name=name, attributes=attrs)

        token = attach_as_current(otel_span)
        self._spans[span_id] = otel_span
        self._tokens[span_id] = token
        self._start_times[span_id] = time.perf_counter()

    def on_span_end(self, span: Any) -> None:
        span_id = str(getattr(span, "span_id", id(span)))
        otel_span = self._spans.pop(span_id, None)
        token = self._tokens.pop(span_id, None)
        start = self._start_times.pop(span_id, None)
        if not otel_span:
            return
        if token:
            detach(token)

        data = getattr(span, "span_data", None) or span
        span_type = getattr(data, "type", None) or getattr(data, "span_type", "") or ""
        _apply_end_attrs(otel_span, span_type, data)

        # Error: SDK puts it on span.error (a dict) ; span_data may also carry one.
        error = getattr(span, "error", None) or getattr(data, "error", None)
        if error:
            msg = error.get("message") if isinstance(error, dict) else str(error)
            otel_span.set_status(StatusCode.ERROR, str(msg))
            if isinstance(error, BaseException):
                otel_span.record_exception(error)
        else:
            otel_span.set_status(StatusCode.OK)

        if start:
            otel_span.set_attribute("neatlogs.llm.metrics.duration_ms", round((time.perf_counter() - start) * 1000, 3))
        otel_span.end()

    def shutdown(self) -> None:
        for span_id in list(self._spans.keys()):
            span = self._spans.pop(span_id, None)
            token = self._tokens.pop(span_id, None)
            if token:
                detach(token)
            if span:
                span.set_status(StatusCode.ERROR, "Processor shutdown before span completed")
                span.end()

    def force_flush(self) -> None:
        pass


# ---------------------------------------------------------------------------
# span_data → attributes
# ---------------------------------------------------------------------------


def _build_start_attrs(span_type: str, data: Any):
    if span_type == "agent":
        name = getattr(data, "name", None) or "agent"
        attrs = {"neatlogs.span.kind": "agent", "neatlogs.agent.name": str(name)}
        tools = getattr(data, "tools", None)
        if tools:
            for i, t in enumerate(tools):
                attrs[f"neatlogs.llm.tools.{i}.name"] = str(t)
        handoffs = getattr(data, "handoffs", None)
        if handoffs:
            attrs["neatlogs.agent.handoffs"] = serialize(handoffs)[:2000]
        return attrs, f"openai_agents.agent.{name}"

    if span_type in ("generation", "llm"):
        attrs = {"neatlogs.span.kind": "llm", "neatlogs.llm.provider": "openai"}
        model = getattr(data, "model", None)
        if model:
            attrs["neatlogs.llm.model_name"] = str(model)
        _set_input_messages(attrs, getattr(data, "input", None))
        return attrs, "openai_agents.generation"

    if span_type == "response":
        attrs = {"neatlogs.span.kind": "llm", "neatlogs.llm.provider": "openai"}
        _set_input_messages(attrs, getattr(data, "input", None))
        return attrs, "openai_agents.response"

    if span_type == "function":
        name = getattr(data, "name", None) or "tool"
        attrs = {"neatlogs.span.kind": "tool", "neatlogs.tool.name": str(name)}
        tool_input = getattr(data, "input", None)
        if tool_input is not None:
            attrs["input.value"] = tool_input if isinstance(tool_input, str) else serialize(tool_input)
        return attrs, f"openai_agents.tool.{name}"

    if span_type == "handoff":
        attrs = {"neatlogs.span.kind": "agent"}
        if getattr(data, "from_agent", None):
            attrs["neatlogs.agent.handoff_from"] = str(data.from_agent)
        if getattr(data, "to_agent", None):
            attrs["neatlogs.agent.name"] = str(data.to_agent)
        return attrs, "openai_agents.handoff"

    if span_type == "guardrail":
        attrs = {"neatlogs.span.kind": "guardrail"}
        name = getattr(data, "name", None)
        if name:
            attrs["neatlogs.guardrail.name"] = str(name)
        return attrs, f"openai_agents.guardrail.{name or 'guardrail'}"

    if span_type == "mcp_tools":
        attrs = {"neatlogs.span.kind": "tool", "neatlogs.tool.type": "mcp_list_tools"}
        server = getattr(data, "server", None)
        if server:
            attrs["neatlogs.tool.mcp_server"] = str(server)
        return attrs, "openai_agents.mcp_list_tools"

    if span_type == "speech":
        attrs = {"neatlogs.span.kind": "llm", "neatlogs.llm.task": "speech"}
        model = getattr(data, "model", None)
        if model:
            attrs["neatlogs.llm.model_name"] = str(model)
        inp = getattr(data, "input", None)
        if inp:
            attrs["input.value"] = inp if isinstance(inp, str) else serialize(inp)
        return attrs, "openai_agents.speech"

    if span_type == "transcription":
        attrs = {"neatlogs.span.kind": "llm", "neatlogs.llm.task": "transcription"}
        model = getattr(data, "model", None)
        if model:
            attrs["neatlogs.llm.model_name"] = str(model)
        return attrs, "openai_agents.transcription"

    if span_type == "custom":
        attrs = {"neatlogs.span.kind": "chain"}
        name = getattr(data, "name", None)
        custom = getattr(data, "data", None)
        if custom:
            attrs["neatlogs.custom.data"] = serialize(custom)[:10000]
        return attrs, f"openai_agents.custom.{name or 'custom'}"

    attrs = {"neatlogs.span.kind": "chain"}
    return attrs, f"openai_agents.{span_type or 'span'}"


def _apply_end_attrs(otel_span: Any, span_type: str, data: Any) -> None:
    if span_type in ("generation", "llm"):
        _set_output_messages(otel_span, getattr(data, "output", None))
        _set_usage(otel_span, getattr(data, "usage", None))
        model = getattr(data, "model", None)
        if model:
            otel_span.set_attribute("neatlogs.llm.model_name", str(model))

    elif span_type == "response":
        response = getattr(data, "response", None)
        if response is not None:
            text = getattr(response, "output_text", None)
            if text:
                otel_span.set_attribute("neatlogs.llm.output_messages.0.role", "assistant")
                otel_span.set_attribute("neatlogs.llm.output_messages.0.content", str(text)[:10000])
            model = getattr(response, "model", None)
            if model:
                otel_span.set_attribute("neatlogs.llm.model_name", str(model))
            usage = getattr(response, "usage", None)
            _set_usage(otel_span, usage)
        _set_usage(otel_span, getattr(data, "usage", None))

    elif span_type == "function":
        output = getattr(data, "output", None)
        if output is not None:
            otel_span.set_attribute("output.value", str(output)[:10000])

    elif span_type == "guardrail":
        triggered = getattr(data, "triggered", None)
        if triggered is not None:
            otel_span.set_attribute("neatlogs.guardrail.triggered", bool(triggered))

    elif span_type == "mcp_tools":
        result = getattr(data, "result", None)
        if result is not None:
            otel_span.set_attribute("output.value", serialize(result)[:10000])

    elif span_type in ("speech", "transcription"):
        output = getattr(data, "output", None)
        if output is not None:
            otel_span.set_attribute("output.value", str(output)[:10000])


def _set_input_messages(attrs: dict, input_msgs: Any) -> None:
    if not input_msgs or not isinstance(input_msgs, list):
        if isinstance(input_msgs, str):
            attrs["neatlogs.llm.input_messages.0.role"] = "user"
            attrs["neatlogs.llm.input_messages.0.content"] = input_msgs[:10000]
        return
    for i, msg in enumerate(input_msgs):
        role = msg.get("role", "") if isinstance(msg, dict) else getattr(msg, "role", "")
        content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
        if role:
            attrs[f"neatlogs.llm.input_messages.{i}.role"] = role
        if content:
            attrs[f"neatlogs.llm.input_messages.{i}.content"] = (content if isinstance(content, str) else serialize(content))[:10000]


def _set_output_messages(otel_span: Any, output: Any) -> None:
    if not output:
        return
    if isinstance(output, list):
        for i, msg in enumerate(output):
            role = msg.get("role", "assistant") if isinstance(msg, dict) else "assistant"
            content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
            otel_span.set_attribute(f"neatlogs.llm.output_messages.{i}.role", role)
            if content:
                otel_span.set_attribute(f"neatlogs.llm.output_messages.{i}.content", (content if isinstance(content, str) else serialize(content))[:10000])
    elif hasattr(output, "content"):
        otel_span.set_attribute("neatlogs.llm.output_messages.0.role", "assistant")
        c = output.content
        otel_span.set_attribute("neatlogs.llm.output_messages.0.content", (c if isinstance(c, str) else serialize(c))[:10000])
    elif isinstance(output, str):
        otel_span.set_attribute("neatlogs.llm.output_messages.0.role", "assistant")
        otel_span.set_attribute("neatlogs.llm.output_messages.0.content", output[:10000])


def _set_usage(otel_span: Any, usage: Any) -> None:
    if not usage:
        return
    if isinstance(usage, dict):
        input_tokens = usage.get("input_tokens") or usage.get("prompt_tokens")
        output_tokens = usage.get("output_tokens") or usage.get("completion_tokens")
        total = usage.get("total_tokens")
    else:
        input_tokens = getattr(usage, "input_tokens", None) or getattr(usage, "prompt_tokens", None)
        output_tokens = getattr(usage, "output_tokens", None) or getattr(usage, "completion_tokens", None)
        total = getattr(usage, "total_tokens", None)
    if input_tokens:
        otel_span.set_attribute("neatlogs.llm.token_count.prompt", input_tokens)
    if output_tokens:
        otel_span.set_attribute("neatlogs.llm.token_count.completion", output_tokens)
    if total:
        otel_span.set_attribute("neatlogs.llm.token_count.total", total)
    elif input_tokens and output_tokens:
        otel_span.set_attribute("neatlogs.llm.token_count.total", input_tokens + output_tokens)
