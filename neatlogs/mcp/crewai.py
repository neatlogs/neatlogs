"""CrewAI callback adapter that captures WORKFLOW, AGENT, CHAIN, TOOL, and LLM spans.

CrewAI provides:
  - step_callback(AgentAction) — fires per tool use / agent step
  - task_callback(TaskOutput)  — fires when a task completes
  - before_kickoff_callbacks   — fires before crew starts
  - after_kickoff_callbacks    — fires after crew finishes

LLM detail comes via LiteLLM's CustomLogger — CrewAI routes all LLM calls through LiteLLM.

Usage::

    nl = NeatlogsCallback(mcp_url="...")
    crew = Crew(
        agents=[...],
        tasks=[...],
        step_callback=nl.crewai.step,
        task_callback=nl.crewai.task,
        before_kickoff_callbacks=[nl.crewai.before_kickoff],
        after_kickoff_callbacks=[nl.crewai.after_kickoff],
    )
    result = crew.kickoff(inputs={...})
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from . import context as ctx
from .buffer import SpanBuffer, SpanRecord


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_json(obj: Any, max_len: int = 50_000) -> str:
    try:
        s = json.dumps(obj, default=str, ensure_ascii=False)
    except (TypeError, ValueError):
        s = str(obj)
    return s[:max_len] if len(s) > max_len else s


class CrewAIAdapter:
    """Captures CrewAI execution spans into the shared SpanBuffer."""

    def __init__(self, buffer: SpanBuffer, workflow_name: str):
        self._buffer = buffer
        self._workflow_name = workflow_name
        self._framework = "crewai"

        # State tracking
        self._trace_id: Optional[str] = None
        self._workflow_span_id: Optional[str] = None
        self._current_agent_span_id: Optional[str] = None
        self._current_task_span_id: Optional[str] = None
        self._current_agent_name: Optional[str] = None

        # LiteLLM logger reference for cleanup
        self._litellm_logger: Optional[Any] = None

    def _ensure_trace(self) -> str:
        if self._trace_id is None:
            self._trace_id = ctx.generate_trace_id()
            ctx.set_trace_id(self._trace_id)
            self._buffer.get_or_create_trace(
                self._trace_id, self._workflow_name, self._framework
            )
        return self._trace_id

    # -----------------------------------------------------------------------
    # Crew lifecycle callbacks
    # -----------------------------------------------------------------------

    def before_kickoff(self, inputs: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """before_kickoff_callbacks — creates WORKFLOW root span, installs LiteLLM hook."""
        self._trace_id = None
        trace_id = self._ensure_trace()

        self._workflow_span_id = ctx.generate_span_id()
        attrs: Dict[str, Any] = {
            "neatlogs.span.kind": "workflow",
            "neatlogs.framework": self._framework,
        }
        if inputs:
            attrs["input.value"] = _safe_json(inputs)

        span = SpanRecord(
            span_id=self._workflow_span_id,
            parent_span_id=None,
            name=self._workflow_name,
            kind="WORKFLOW",
            start_time=_now_iso(),
            attributes=attrs,
            is_root=True,
        )
        self._buffer.add_span(trace_id, span)
        buf = self._buffer.get_or_create_trace(trace_id, self._workflow_name)
        buf.root_span_id = self._workflow_span_id

        self._install_litellm_logger()
        return inputs

    def after_kickoff(self, result: Any) -> Any:
        """after_kickoff_callbacks — closes WORKFLOW span, removes LiteLLM hook."""
        if self._trace_id and self._workflow_span_id:
            self._close_current_agent()

            output_attrs: Dict[str, Any] = {"output.value": _safe_json(result)}
            # CrewOutput has usage_metrics
            usage_metrics = getattr(result, "usage_metrics", None)
            if usage_metrics:
                output_attrs["neatlogs.crew.token_usage"] = _safe_json(usage_metrics)

            self._buffer.complete_span(
                self._trace_id,
                self._workflow_span_id,
                _now_iso(),
                status_code="OK",
                output_attrs=output_attrs,
            )

        self._uninstall_litellm_logger()
        self._current_agent_span_id = None
        self._current_task_span_id = None
        self._current_agent_name = None
        return result

    # -----------------------------------------------------------------------
    # step_callback — fires per agent action (tool use)
    # -----------------------------------------------------------------------

    def step(self, step_output: Any) -> None:
        """step_callback for Crew(step_callback=nl.crewai.step)."""
        trace_id = self._ensure_trace()

        tool_name = getattr(step_output, "tool", None) or "unknown_tool"
        tool_input = getattr(step_output, "tool_input", None) or ""
        result = getattr(step_output, "result", None) or getattr(step_output, "log", "")

        parent = self._current_task_span_id or self._current_agent_span_id or self._workflow_span_id

        attrs: Dict[str, Any] = {
            "neatlogs.span.kind": "tool",
            "neatlogs.tool.name": str(tool_name),
            "neatlogs.tool.input": _safe_json(tool_input),
            "neatlogs.tool.output": _safe_json(result),
        }

        span = SpanRecord(
            span_id=ctx.generate_span_id(),
            parent_span_id=parent,
            name=str(tool_name),
            kind="TOOL",
            start_time=_now_iso(),
            end_time=_now_iso(),
            status_code="OK",
            attributes=attrs,
        )
        self._buffer.add_span(trace_id, span)

    # -----------------------------------------------------------------------
    # task_callback — fires when a task completes
    # -----------------------------------------------------------------------

    def task(self, task_output: Any) -> None:
        """task_callback for Crew(task_callback=nl.crewai.task)."""
        trace_id = self._ensure_trace()

        description = getattr(task_output, "description", "") or ""
        raw_output = getattr(task_output, "raw", "") or str(task_output)
        agent_name = getattr(task_output, "agent", "") or ""

        # If agent changed, close previous and start new agent span
        if agent_name and agent_name != self._current_agent_name:
            self._close_current_agent()
            self._start_agent_span(str(agent_name))

        parent = self._current_agent_span_id or self._workflow_span_id
        span_id = ctx.generate_span_id()
        self._current_task_span_id = span_id

        task_name = str(description)[:80] or "task"
        attrs: Dict[str, Any] = {
            "neatlogs.span.kind": "chain",
            "input.value": _safe_json(description),
            "output.value": _safe_json(raw_output),
        }

        span = SpanRecord(
            span_id=span_id,
            parent_span_id=parent,
            name=task_name,
            kind="CHAIN",
            start_time=_now_iso(),
            end_time=_now_iso(),
            status_code="OK",
            attributes=attrs,
        )
        self._buffer.add_span(trace_id, span)

    # -----------------------------------------------------------------------
    # Agent span management
    # -----------------------------------------------------------------------

    def _start_agent_span(self, agent_name: str) -> None:
        trace_id = self._ensure_trace()
        self._current_agent_name = agent_name
        self._current_agent_span_id = ctx.generate_span_id()

        span = SpanRecord(
            span_id=self._current_agent_span_id,
            parent_span_id=self._workflow_span_id,
            name=agent_name,
            kind="AGENT",
            start_time=_now_iso(),
            attributes={
                "neatlogs.span.kind": "agent",
                "input.value": _safe_json({"agent": agent_name}),
            },
        )
        self._buffer.add_span(trace_id, span)

    def _close_current_agent(self) -> None:
        if self._current_agent_span_id and self._trace_id:
            self._buffer.complete_span(
                self._trace_id,
                self._current_agent_span_id,
                _now_iso(),
                status_code="OK",
            )
        self._current_agent_span_id = None
        self._current_agent_name = None
        self._current_task_span_id = None

    # -----------------------------------------------------------------------
    # LiteLLM integration — captures LLM-level detail
    # -----------------------------------------------------------------------

    def _install_litellm_logger(self) -> None:
        try:
            import litellm
            from litellm.integrations.custom_logger import CustomLogger

            adapter = self

            class _NeatlogsLiteLLMLogger(CustomLogger):
                def log_success_event(self_logger, kwargs, response_obj, start_time, end_time):
                    adapter._on_litellm_success(kwargs, response_obj, start_time, end_time)

                def log_failure_event(self_logger, kwargs, response_obj, start_time, end_time):
                    adapter._on_litellm_failure(kwargs, start_time, end_time)

                async def async_log_success_event(self_logger, kwargs, response_obj, start_time, end_time):
                    adapter._on_litellm_success(kwargs, response_obj, start_time, end_time)

                async def async_log_failure_event(self_logger, kwargs, response_obj, start_time, end_time):
                    adapter._on_litellm_failure(kwargs, start_time, end_time)

            self._litellm_logger = _NeatlogsLiteLLMLogger()
            litellm.callbacks.append(self._litellm_logger)
        except ImportError:
            pass

    def _uninstall_litellm_logger(self) -> None:
        if self._litellm_logger:
            try:
                import litellm
                litellm.callbacks.remove(self._litellm_logger)
            except (ImportError, ValueError):
                pass
            self._litellm_logger = None

    def _on_litellm_success(
        self,
        kwargs: Dict[str, Any],
        response_obj: Any,
        start_time: Any,
        end_time: Any,
    ) -> None:
        trace_id = self._ensure_trace()
        parent = (
            self._current_task_span_id
            or self._current_agent_span_id
            or self._workflow_span_id
        )

        model = kwargs.get("model", "") or ""
        messages = kwargs.get("messages", [])

        # Provider detection
        provider = ""
        custom_llm_provider = kwargs.get("litellm_params", {}).get("custom_llm_provider", "")
        if custom_llm_provider:
            provider = custom_llm_provider
        elif "gpt" in model.lower() or "o1" in model.lower() or "o3" in model.lower():
            provider = "openai"
        elif "claude" in model.lower():
            provider = "anthropic"
        elif "gemini" in model.lower():
            provider = "google_genai"

        attrs: Dict[str, Any] = {
            "neatlogs.span.kind": "llm",
            "neatlogs.llm.model_name": model,
            "neatlogs.llm.request_type": "chat",
        }
        if provider:
            attrs["neatlogs.llm.provider"] = provider
            attrs["neatlogs.llm.system"] = provider

        # Input messages
        if isinstance(messages, list):
            for i, msg in enumerate(messages):
                if isinstance(msg, dict):
                    attrs[f"neatlogs.llm.input_messages.{i}.role"] = msg.get("role", "")
                    content = msg.get("content", "")
                    if isinstance(content, list):
                        content = " ".join(
                            p.get("text", str(p)) if isinstance(p, dict) else str(p)
                            for p in content
                        )
                    attrs[f"neatlogs.llm.input_messages.{i}.content"] = str(content)[:50_000]

        # Invocation params
        optional_params = kwargs.get("optional_params", {})
        for src, tgt in [
            ("temperature", "neatlogs.llm.temperature"),
            ("max_tokens", "neatlogs.llm.max_tokens"),
            ("top_p", "neatlogs.llm.top_p"),
            ("stream", "neatlogs.llm.is_streaming"),
        ]:
            if src in optional_params:
                attrs[tgt] = optional_params[src]
        attrs["neatlogs.llm.invocation_parameters"] = _safe_json(optional_params)

        # Output messages
        choices = getattr(response_obj, "choices", None) or []
        for i, choice in enumerate(choices):
            message = getattr(choice, "message", None)
            if message:
                attrs[f"neatlogs.llm.output_messages.{i}.role"] = getattr(message, "role", "assistant")
                attrs[f"neatlogs.llm.output_messages.{i}.content"] = str(getattr(message, "content", "") or "")[:50_000]

                finish_reason = getattr(choice, "finish_reason", None)
                if finish_reason:
                    attrs[f"neatlogs.llm.output_messages.{i}.message.finish_reason"] = str(finish_reason)
                    if i == 0:
                        attrs["neatlogs.llm.finish_reason"] = str(finish_reason)

                tool_calls = getattr(message, "tool_calls", None) or []
                for ti, tc in enumerate(tool_calls):
                    attrs[f"neatlogs.llm.tool_calls.{ti}"] = _safe_json({
                        "id": getattr(tc, "id", ""),
                        "type": getattr(tc, "type", "function"),
                        "function": {
                            "name": getattr(getattr(tc, "function", None), "name", ""),
                            "arguments": getattr(getattr(tc, "function", None), "arguments", ""),
                        },
                    })

        # Token usage
        usage = getattr(response_obj, "usage", None)
        if usage:
            if hasattr(usage, "prompt_tokens"):
                attrs["neatlogs.llm.token_count.prompt"] = usage.prompt_tokens or 0
            if hasattr(usage, "completion_tokens"):
                attrs["neatlogs.llm.token_count.completion"] = usage.completion_tokens or 0
            if hasattr(usage, "total_tokens"):
                attrs["neatlogs.llm.token_count.total"] = usage.total_tokens or 0

            cdetails = getattr(usage, "completion_tokens_details", None)
            if cdetails and getattr(cdetails, "reasoning_tokens", None):
                attrs["neatlogs.llm.token_count.reasoning"] = cdetails.reasoning_tokens

            pdetails = getattr(usage, "prompt_tokens_details", None)
            if pdetails and getattr(pdetails, "cached_tokens", None):
                attrs["neatlogs.llm.token_count.cache_read"] = pdetails.cached_tokens

            if hasattr(usage, "cache_read_input_tokens") and usage.cache_read_input_tokens:
                attrs["neatlogs.llm.token_count.cache_read"] = usage.cache_read_input_tokens
            if hasattr(usage, "cache_creation_input_tokens") and usage.cache_creation_input_tokens:
                attrs["neatlogs.llm.token_count.cache_write"] = usage.cache_creation_input_tokens

        # Prompt template from context
        pt = ctx.get_prompt_template()
        if pt:
            attrs["neatlogs.llm.prompt_template"] = pt
            pv = ctx.get_prompt_variables()
            if pv:
                attrs["neatlogs.llm.prompt_template_variables"] = _safe_json(pv)

        # Timestamps
        start_iso = _now_iso()
        end_iso = _now_iso()
        if isinstance(start_time, datetime):
            start_iso = start_time.astimezone(timezone.utc).isoformat()
        if isinstance(end_time, datetime):
            end_iso = end_time.astimezone(timezone.utc).isoformat()

        span = SpanRecord(
            span_id=ctx.generate_span_id(),
            parent_span_id=parent,
            name=model or "llm",
            kind="LLM",
            start_time=start_iso,
            end_time=end_iso,
            status_code="OK",
            attributes=attrs,
        )
        self._buffer.add_span(trace_id, span)

    def _on_litellm_failure(
        self,
        kwargs: Dict[str, Any],
        start_time: Any,
        end_time: Any,
    ) -> None:
        trace_id = self._ensure_trace()
        parent = (
            self._current_task_span_id
            or self._current_agent_span_id
            or self._workflow_span_id
        )

        model = kwargs.get("model", "llm")
        error_msg = str(kwargs.get("exception", "unknown error"))

        start_iso = _now_iso()
        end_iso = _now_iso()
        if isinstance(start_time, datetime):
            start_iso = start_time.astimezone(timezone.utc).isoformat()
        if isinstance(end_time, datetime):
            end_iso = end_time.astimezone(timezone.utc).isoformat()

        span = SpanRecord(
            span_id=ctx.generate_span_id(),
            parent_span_id=parent,
            name=model,
            kind="LLM",
            start_time=start_iso,
            end_time=end_iso,
            status_code="ERROR",
            status_message=error_msg,
            attributes={
                "neatlogs.span.kind": "llm",
                "neatlogs.llm.model_name": model,
            },
        )
        self._buffer.add_span(trace_id, span)
