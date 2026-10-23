"""Default provider caller for replay (OpenAI chat completions)."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from .types import LLMRequest, LLMResponse, ProviderError, ToolCallData


def _request_tools(request: LLMRequest) -> Optional[List[Dict[str, Any]]]:
    tools: List[Dict[str, Any]] = []
    for tool in request.tools:
        definition = tool.get("definition")
        if isinstance(definition, dict) and definition.get("type"):
            tools.append(definition)
        elif tool.get("name"):
            tools.append(
                {
                    "type": tool.get("type") or "function",
                    "function": {"name": tool["name"]},
                }
            )
    return tools or None


def openai_caller(request: LLMRequest) -> LLMResponse:
    """Issue a live OpenAI chat completion for a reconstructed request.

    Uses the caller's own ``OPENAI_API_KEY`` credentials at call time; nothing
    is read from span attributes. Any SDK/API failure is wrapped as
    :class:`ProviderError` so replay callers handle one error type.
    """
    try:
        import openai
    except ImportError as exc:
        raise ProviderError("openai package is required for the default caller") from exc

    kwargs: Dict[str, Any] = {"model": request.model, "messages": request.messages}
    if request.temperature is not None:
        kwargs["temperature"] = request.temperature
    if request.top_p is not None:
        kwargs["top_p"] = request.top_p
    if request.max_tokens is not None:
        kwargs["max_tokens"] = request.max_tokens
    if request.frequency_penalty is not None:
        kwargs["frequency_penalty"] = request.frequency_penalty
    if request.presence_penalty is not None:
        kwargs["presence_penalty"] = request.presence_penalty
    tools = _request_tools(request)
    if tools:
        kwargs["tools"] = tools

    client = openai.OpenAI()
    start = time.perf_counter()
    try:
        response = client.chat.completions.create(**kwargs)
    except Exception as exc:
        raise ProviderError(f"OpenAI call failed during replay: {exc}") from exc
    latency_ms = (time.perf_counter() - start) * 1000

    try:
        choice = (response.choices or [None])[0]
        message = getattr(choice, "message", None)
        content = getattr(message, "content", None) or ""
        raw_calls = getattr(message, "tool_calls", None) or []
        tool_calls = [
            ToolCallData(
                name=getattr(getattr(c, "function", None), "name", "") or "",
                arguments=getattr(getattr(c, "function", None), "arguments", "") or "",
                id=getattr(c, "id", "") or "",
            )
            for c in raw_calls
        ]
        usage = getattr(response, "usage", None)
        return LLMResponse(
            content=str(content),
            tool_calls=tool_calls,
            model=str(getattr(response, "model", "") or request.model),
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
            total_tokens=getattr(usage, "total_tokens", None),
            latency_ms=round(latency_ms, 3),
            finish_reason=str(getattr(choice, "finish_reason", "") or "") or None,
        )
    except ProviderError:
        raise
    except Exception as exc:
        raise ProviderError(f"Failed to parse OpenAI replay response: {exc}") from exc
