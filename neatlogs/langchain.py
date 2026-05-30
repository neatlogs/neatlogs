"""
Neatlogs LangChain/LangGraph callback handler.

Usage:
    >>> import neatlogs
    >>> handler = neatlogs.langchain_handler()
    >>> result = chain.invoke(input, config={"callbacks": [handler]})

Works with LangChain, LangGraph, and any framework using LangChain callbacks
(e.g., Deep Agents).
"""

from typing import Any, Dict, List, Optional
from uuid import UUID

from opentelemetry import context as otel_context
from opentelemetry.trace import StatusCode

from ._wrap_utils import get_tracer, serialize

try:
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.messages import BaseMessage
    from langchain_core.outputs import LLMResult
except ImportError:
    raise ImportError(
        "langchain-core is required for neatlogs.langchain_handler(). "
        "Install it with: pip install langchain-core"
    )


def _set_invocation_params(span: Any, kwargs: Dict[str, Any]) -> None:
    """Extract and set invocation parameters on span."""
    invocation_params = kwargs.get("invocation_params", {})
    if not invocation_params:
        return

    # Store raw invocation_parameters as JSON (backend maps this)
    span.set_attribute("neatlogs.llm.invocation_parameters", serialize(invocation_params))

    # Also set individual params for structured queries
    param_mapping = {
        "temperature": "neatlogs.llm.temperature",
        "max_tokens": "neatlogs.llm.max_tokens",
        "max_output_tokens": "neatlogs.llm.max_tokens",
        "top_p": "neatlogs.llm.top_p",
        "top_k": "neatlogs.llm.top_k",
        "frequency_penalty": "neatlogs.llm.frequency_penalty",
        "presence_penalty": "neatlogs.llm.presence_penalty",
    }

    for param, attr_name in param_mapping.items():
        val = invocation_params.get(param)
        if val is not None:
            span.set_attribute(attr_name, val)

    stop = invocation_params.get("stop")
    if stop is not None:
        span.set_attribute("neatlogs.llm.stop_sequences", serialize(stop) if isinstance(stop, list) else str(stop))

    if invocation_params.get("stream") or invocation_params.get("streaming"):
        span.set_attribute("neatlogs.llm.is_streaming", True)

    # Tools/functions from invocation params
    tools = invocation_params.get("tools") or invocation_params.get("functions")
    if tools:
        for i, tool in enumerate(tools):
            if isinstance(tool, dict):
                fn = tool.get("function", tool)
                name = fn.get("name", "")
                if name:
                    span.set_attribute(f"neatlogs.llm.tools.{i}.name", name)
                desc = fn.get("description")
                if desc:
                    span.set_attribute(f"neatlogs.llm.tools.{i}.description", desc)
                params = fn.get("parameters")
                if params:
                    span.set_attribute(f"neatlogs.llm.tools.{i}.input_schema", serialize(params))


class NeatlogsCallbackHandler(BaseCallbackHandler):
    """LangChain callback handler that creates neatlogs.llm.* spans."""

    def __init__(self, workflow_name: Optional[str] = None):
        super().__init__()
        self._spans: Dict[UUID, Any] = {}
        self._tokens: Dict[UUID, List[str]] = {}
        self._contexts: Dict[UUID, Any] = {}
        self._workflow_name = workflow_name

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        tracer = get_tracer()

        invocation_params = kwargs.get("invocation_params", {})
        model = ""
        for key in ("model_name", "model", "model_id"):
            model = invocation_params.get(key, "")
            if model:
                break
        if not model:
            model = serialized.get("kwargs", {}).get("model_name", "")
        if not model:
            model = serialized.get("id", [""])[-1] if serialized.get("id") else ""

        parent_ctx = self._contexts.get(parent_run_id) if parent_run_id else None
        ctx = parent_ctx if parent_ctx else otel_context.get_current()

        span = tracer.start_span(
            name="langchain.chat_model",
            context=ctx,
            attributes={
                "neatlogs.span.kind": "LLM",
                "neatlogs.llm.provider": "langchain",
                "neatlogs.llm.model_name": model,
            },
        )

        _set_invocation_params(span, kwargs)

        # Capture input messages
        if messages:
            idx = 0
            for msg in messages[0]:
                role = getattr(msg, "type", "unknown")
                if role == "human":
                    role = "user"
                elif role == "ai":
                    role = "assistant"
                content = getattr(msg, "content", "")
                span.set_attribute(f"neatlogs.llm.input_messages.{idx}.role", role)
                if isinstance(content, str):
                    span.set_attribute(f"neatlogs.llm.input_messages.{idx}.content", content)
                else:
                    span.set_attribute(f"neatlogs.llm.input_messages.{idx}.content", serialize(content))
                idx += 1

        self._spans[run_id] = span
        new_ctx = otel_context.set_value("current_span", span, ctx)
        self._contexts[run_id] = new_ctx
        self._tokens[run_id] = []

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        tracer = get_tracer()

        invocation_params = kwargs.get("invocation_params", {})
        model = ""
        for key in ("model_name", "model", "model_id"):
            model = invocation_params.get(key, "")
            if model:
                break
        if not model:
            model = serialized.get("id", [""])[-1] if serialized.get("id") else ""

        parent_ctx = self._contexts.get(parent_run_id) if parent_run_id else None
        ctx = parent_ctx if parent_ctx else otel_context.get_current()

        span = tracer.start_span(
            name="langchain.llm",
            context=ctx,
            attributes={
                "neatlogs.span.kind": "LLM",
                "neatlogs.llm.provider": "langchain",
                "neatlogs.llm.model_name": model,
            },
        )

        _set_invocation_params(span, kwargs)

        if prompts:
            for i, prompt in enumerate(prompts):
                span.set_attribute(f"neatlogs.llm.input_messages.{i}.role", "user")
                span.set_attribute(f"neatlogs.llm.input_messages.{i}.content", prompt)

        self._spans[run_id] = span
        new_ctx = otel_context.set_value("current_span", span, ctx)
        self._contexts[run_id] = new_ctx
        self._tokens[run_id] = []

    def on_llm_new_token(
        self,
        token: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        if run_id in self._tokens:
            self._tokens[run_id].append(token)

    def on_llm_end(
        self,
        response: "LLMResult",
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        span = self._spans.pop(run_id, None)
        self._contexts.pop(run_id, None)
        self._tokens.pop(run_id, None)
        if not span:
            return

        if response.generations:
            for gen_list in response.generations:
                if gen_list:
                    gen = gen_list[0]
                    text = getattr(gen, "text", "")
                    if text:
                        span.set_attribute("neatlogs.llm.output_messages.0.role", "assistant")
                        span.set_attribute("neatlogs.llm.output_messages.0.content", text)

                    message = getattr(gen, "message", None)
                    if message:
                        content = getattr(message, "content", "")
                        if content and not text:
                            span.set_attribute("neatlogs.llm.output_messages.0.role", "assistant")
                            span.set_attribute("neatlogs.llm.output_messages.0.content", content if isinstance(content, str) else serialize(content))

                        tool_calls = getattr(message, "tool_calls", None)
                        if tool_calls:
                            for j, tc in enumerate(tool_calls):
                                span.set_attribute(f"neatlogs.llm.tool_calls.{j}.name", tc.get("name", ""))
                                span.set_attribute(f"neatlogs.llm.tool_calls.{j}.arguments", serialize(tc.get("args", {})))
                                if tc.get("id"):
                                    span.set_attribute(f"neatlogs.llm.tool_calls.{j}.id", tc["id"])

                        thinking_blocks = getattr(message, "thinking_blocks", None)
                        if thinking_blocks:
                            thinking_text = "".join(
                                block.get("thinking", "") for block in thinking_blocks if isinstance(block, dict)
                            )
                            if thinking_text:
                                span.set_attribute("neatlogs.llm.output_messages.0.thinking", thinking_text)

                    gen_info = getattr(gen, "generation_info", None) or {}
                    finish_reason = gen_info.get("finish_reason")
                    if finish_reason:
                        span.set_attribute("neatlogs.llm.finish_reason", finish_reason)

        # Token usage
        llm_output = getattr(response, "llm_output", None) or {}
        token_usage = llm_output.get("token_usage") or llm_output.get("usage") or {}
        if isinstance(token_usage, dict):
            if "prompt_tokens" in token_usage:
                span.set_attribute("neatlogs.llm.token_count.prompt", token_usage["prompt_tokens"])
            if "completion_tokens" in token_usage:
                span.set_attribute("neatlogs.llm.token_count.completion", token_usage["completion_tokens"])
            if "total_tokens" in token_usage:
                span.set_attribute("neatlogs.llm.token_count.total", token_usage["total_tokens"])
            if "cache_read_input_tokens" in token_usage:
                span.set_attribute("neatlogs.llm.token_count.cache_read", token_usage["cache_read_input_tokens"])
            if "cache_creation_input_tokens" in token_usage:
                span.set_attribute("neatlogs.llm.token_count.cache_write", token_usage["cache_creation_input_tokens"])
            if "reasoning_tokens" in token_usage:
                span.set_attribute("neatlogs.llm.token_count.reasoning", token_usage["reasoning_tokens"])

        model_name = llm_output.get("model_name") or llm_output.get("model")
        if model_name:
            span.set_attribute("neatlogs.llm.model_name", model_name)

        span.set_status(StatusCode.OK)
        span.end()

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        span = self._spans.pop(run_id, None)
        self._contexts.pop(run_id, None)
        self._tokens.pop(run_id, None)
        if not span:
            return
        span.set_status(StatusCode.ERROR, str(error))
        span.record_exception(error)
        span.end()

    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        tracer = get_tracer()
        name = serialized.get("id", [""])[-1] if serialized.get("id") else "chain"

        parent_ctx = self._contexts.get(parent_run_id) if parent_run_id else None
        ctx = parent_ctx if parent_ctx else otel_context.get_current()

        span = tracer.start_span(
            name=f"langchain.chain.{name}",
            context=ctx,
            attributes={
                "neatlogs.span.kind": "CHAIN",
            },
        )

        if inputs:
            span.set_attribute("input.value", serialize(inputs))

        self._spans[run_id] = span
        new_ctx = otel_context.set_value("current_span", span, ctx)
        self._contexts[run_id] = new_ctx

    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        span = self._spans.pop(run_id, None)
        self._contexts.pop(run_id, None)
        if not span:
            return

        if outputs:
            span.set_attribute("output.value", serialize(outputs))

        span.set_status(StatusCode.OK)
        span.end()

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        span = self._spans.pop(run_id, None)
        self._contexts.pop(run_id, None)
        if not span:
            return
        span.set_status(StatusCode.ERROR, str(error))
        span.record_exception(error)
        span.end()

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        tracer = get_tracer()
        name = serialized.get("name", "") or (serialized.get("id", [""])[-1] if serialized.get("id") else "tool")

        parent_ctx = self._contexts.get(parent_run_id) if parent_run_id else None
        ctx = parent_ctx if parent_ctx else otel_context.get_current()

        span = tracer.start_span(
            name=f"langchain.tool.{name}",
            context=ctx,
            attributes={
                "neatlogs.span.kind": "TOOL",
                "neatlogs.tool.name": name,
            },
        )

        if input_str:
            span.set_attribute("input.value", input_str)

        self._spans[run_id] = span
        new_ctx = otel_context.set_value("current_span", span, ctx)
        self._contexts[run_id] = new_ctx

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        span = self._spans.pop(run_id, None)
        self._contexts.pop(run_id, None)
        if not span:
            return

        if output is not None:
            output_str = str(output) if not isinstance(output, str) else output
            span.set_attribute("output.value", output_str[:10000])

        span.set_status(StatusCode.OK)
        span.end()

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        span = self._spans.pop(run_id, None)
        self._contexts.pop(run_id, None)
        if not span:
            return
        span.set_status(StatusCode.ERROR, str(error))
        span.record_exception(error)
        span.end()
