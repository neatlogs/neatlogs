"""
Neatlogs OpenAI Agents SDK trace processor.

Usage:
    >>> import neatlogs
    >>> from agents import add_trace_processor
    >>> add_trace_processor(neatlogs.openai_agents_processor())

Creates spans: AGENT (agent runs), LLM (generations), TOOL (function calls)
"""

import time
from typing import Any, Dict, List, Optional

from opentelemetry import context as otel_context
from opentelemetry.trace import StatusCode

from ._wrap_utils import get_tracer, serialize


def openai_agents_processor():
    """
    Return a trace processor for the OpenAI Agents SDK.

    Register with:
        from agents import add_trace_processor
        add_trace_processor(neatlogs.openai_agents_processor())
    """
    return _NeatlogsTraceProcessor()


class _NeatlogsTraceProcessor:
    """
    Implements the OpenAI Agents SDK TracingProcessor protocol.

    The SDK calls:
      - on_trace_start(trace) / on_trace_end(trace)
      - on_span_start(span) / on_span_end(span)

    We map these to OTel spans with proper neatlogs.* attributes.
    """

    def __init__(self):
        self._spans: Dict[str, Any] = {}
        self._tokens: Dict[str, Any] = {}
        self._start_times: Dict[str, float] = {}

    def on_trace_start(self, trace: Any) -> None:
        """Called when a new agent trace begins."""
        tracer = get_tracer()
        attrs = {"neatlogs.span.kind": "WORKFLOW"}

        workflow_name = getattr(trace, "workflow_name", None) or getattr(trace, "name", None)
        if workflow_name:
            attrs["neatlogs.workflow.name"] = workflow_name

        trace_id = getattr(trace, "trace_id", None)
        if trace_id:
            attrs["neatlogs.agent.trace_id"] = str(trace_id)

        span = tracer.start_span(name="openai_agents.trace", attributes=attrs)
        ctx = otel_context.set_value("current_span", span)
        token = otel_context.attach(ctx)

        key = str(getattr(trace, "trace_id", id(trace)))
        self._spans[key] = span
        self._tokens[key] = token
        self._start_times[key] = time.perf_counter()

    def on_trace_end(self, trace: Any) -> None:
        """Called when an agent trace completes."""
        key = str(getattr(trace, "trace_id", id(trace)))
        span = self._spans.pop(key, None)
        token = self._tokens.pop(key, None)
        start_time = self._start_times.pop(key, None)
        if not span:
            return

        if token:
            otel_context.detach(token)

        if start_time:
            duration_ms = (time.perf_counter() - start_time) * 1000
            span.set_attribute("neatlogs.llm.metrics.duration_ms", round(duration_ms, 3))

        span.set_status(StatusCode.OK)
        span.end()

    def on_span_start(self, span_data: Any) -> None:
        """Called when a span within a trace starts (agent run, LLM call, tool call, etc)."""
        tracer = get_tracer()

        span_type = getattr(span_data, "span_type", None) or getattr(span_data, "type", "")
        span_id = str(getattr(span_data, "span_id", id(span_data)))

        if span_type in ("agent", "agent_run"):
            attrs = {"neatlogs.span.kind": "AGENT"}
            agent_name = getattr(span_data, "agent_name", None) or getattr(span_data, "name", None)
            if agent_name:
                attrs["neatlogs.agent.name"] = agent_name
            otel_span = tracer.start_span(name=f"openai_agents.agent.{agent_name or 'run'}", attributes=attrs)

        elif span_type in ("generation", "llm"):
            attrs = {
                "neatlogs.span.kind": "LLM",
                "neatlogs.llm.provider": "openai",
            }
            model = getattr(span_data, "model", None)
            if model:
                attrs["neatlogs.llm.model_name"] = model

            # Input messages
            input_msgs = getattr(span_data, "input", None) or getattr(span_data, "messages", None)
            if input_msgs and isinstance(input_msgs, list):
                for i, msg in enumerate(input_msgs):
                    role = msg.get("role", "") if isinstance(msg, dict) else getattr(msg, "role", "")
                    content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
                    if role:
                        attrs[f"neatlogs.llm.input_messages.{i}.role"] = role
                    if content:
                        attrs[f"neatlogs.llm.input_messages.{i}.content"] = (content if isinstance(content, str) else serialize(content))[:10000]

            otel_span = tracer.start_span(name="openai_agents.generation", attributes=attrs)

        elif span_type in ("function", "tool", "tool_call"):
            attrs = {"neatlogs.span.kind": "TOOL"}
            tool_name = getattr(span_data, "name", None) or getattr(span_data, "function_name", "")
            if tool_name:
                attrs["neatlogs.tool.name"] = tool_name

            tool_input = getattr(span_data, "input", None) or getattr(span_data, "arguments", None)
            if tool_input:
                attrs["input.value"] = serialize(tool_input) if not isinstance(tool_input, str) else tool_input

            otel_span = tracer.start_span(name=f"openai_agents.tool.{tool_name}", attributes=attrs)

        elif span_type == "handoff":
            attrs = {"neatlogs.span.kind": "AGENT"}
            from_agent = getattr(span_data, "from_agent", None)
            to_agent = getattr(span_data, "to_agent", None)
            if from_agent:
                attrs["neatlogs.agent.handoff_from"] = str(from_agent)
            if to_agent:
                attrs["neatlogs.agent.name"] = str(to_agent)
            otel_span = tracer.start_span(name=f"openai_agents.handoff", attributes=attrs)

        else:
            attrs = {"neatlogs.span.kind": "CHAIN"}
            otel_span = tracer.start_span(name=f"openai_agents.{span_type or 'span'}", attributes=attrs)

        ctx = otel_context.set_value("current_span", otel_span)
        token = otel_context.attach(ctx)

        self._spans[span_id] = otel_span
        self._tokens[span_id] = token
        self._start_times[span_id] = time.perf_counter()

    def on_span_end(self, span_data: Any) -> None:
        """Called when a span within a trace ends."""
        span_id = str(getattr(span_data, "span_id", id(span_data)))
        otel_span = self._spans.pop(span_id, None)
        token = self._tokens.pop(span_id, None)
        start_time = self._start_times.pop(span_id, None)
        if not otel_span:
            return

        if token:
            otel_context.detach(token)

        span_type = getattr(span_data, "span_type", None) or getattr(span_data, "type", "")

        # Extract output/response data
        if span_type in ("generation", "llm"):
            output = getattr(span_data, "output", None) or getattr(span_data, "response", None)
            if output:
                # Output message content
                if isinstance(output, list):
                    for i, msg in enumerate(output):
                        role = msg.get("role", "assistant") if isinstance(msg, dict) else "assistant"
                        content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
                        otel_span.set_attribute(f"neatlogs.llm.output_messages.{i}.role", role)
                        if content:
                            otel_span.set_attribute(f"neatlogs.llm.output_messages.{i}.content", (content if isinstance(content, str) else serialize(content))[:10000])
                elif hasattr(output, "content"):
                    otel_span.set_attribute("neatlogs.llm.output_messages.0.role", "assistant")
                    content = output.content
                    otel_span.set_attribute("neatlogs.llm.output_messages.0.content", (content if isinstance(content, str) else serialize(content))[:10000])

            # Token usage
            usage = getattr(span_data, "usage", None)
            if usage:
                input_tokens = getattr(usage, "input_tokens", None) or getattr(usage, "prompt_tokens", None)
                output_tokens = getattr(usage, "output_tokens", None) or getattr(usage, "completion_tokens", None)
                if input_tokens:
                    otel_span.set_attribute("neatlogs.llm.token_count.prompt", input_tokens)
                if output_tokens:
                    otel_span.set_attribute("neatlogs.llm.token_count.completion", output_tokens)
                if input_tokens and output_tokens:
                    otel_span.set_attribute("neatlogs.llm.token_count.total", input_tokens + output_tokens)

            model = getattr(span_data, "model", None)
            if model:
                otel_span.set_attribute("neatlogs.llm.model_name", model)

        elif span_type in ("function", "tool", "tool_call"):
            output = getattr(span_data, "output", None) or getattr(span_data, "result", None)
            if output is not None:
                otel_span.set_attribute("output.value", str(output)[:10000])

        elif span_type in ("agent", "agent_run"):
            output = getattr(span_data, "output", None)
            if output is not None:
                otel_span.set_attribute("output.value", str(output)[:10000])

        # Error handling
        error = getattr(span_data, "error", None)
        if error:
            otel_span.set_status(StatusCode.ERROR, str(error))
            if isinstance(error, BaseException):
                otel_span.record_exception(error)
        else:
            otel_span.set_status(StatusCode.OK)

        if start_time:
            duration_ms = (time.perf_counter() - start_time) * 1000
            otel_span.set_attribute("neatlogs.llm.metrics.duration_ms", round(duration_ms, 3))

        otel_span.end()

    def shutdown(self) -> None:
        """Clean up any remaining spans."""
        for span_id in list(self._spans.keys()):
            span = self._spans.pop(span_id, None)
            token = self._tokens.pop(span_id, None)
            if token:
                otel_context.detach(token)
            if span:
                span.set_status(StatusCode.ERROR, "Processor shutdown before span completed")
                span.end()

    def force_flush(self) -> None:
        pass
