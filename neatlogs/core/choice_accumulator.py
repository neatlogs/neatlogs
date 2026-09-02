"""Incremental OpenAI-compatible multi-choice and tool-fragment capture."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from opentelemetry.trace import StatusCode

from .capture import BoundedTextAccumulator
from .media import current_media_store, media_references, sanitize_media_payload

MAX_SEMANTIC_STREAM_EVENTS = 128
MAX_MEDIA_RECORDS_PER_CHOICE = 32


def _get(value: Any, name: str, default=None):
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _string(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(
            sanitize_media_payload(value, "output"),
            default=str,
            ensure_ascii=False,
            separators=(",", ":"),
        )
    except (TypeError, ValueError):
        return str(value or "")


@dataclass
class _ToolCall:
    id: str = ""
    name: str = ""
    arguments: BoundedTextAccumulator = field(default_factory=BoundedTextAccumulator)
    type: str = ""
    details: str = ""
    synthetic: bool = False


@dataclass
class _Choice:
    role: str = "assistant"
    content: BoundedTextAccumulator = field(default_factory=BoundedTextAccumulator)
    reasoning: BoundedTextAccumulator = field(default_factory=BoundedTextAccumulator)
    media_records: list[dict[str, Any]] = field(default_factory=list)
    media_records_dropped: int = 0
    finish_reason: str | None = None
    tool_calls: dict[int, _ToolCall] = field(default_factory=dict)


class ChoiceAccumulator:
    """Preserve every choice and key tool fragments by choice + tool index."""

    def __init__(self, capture_fidelity: str = "native") -> None:
        if capture_fidelity not in {"native", "normalized", "flattened", "unknown"}:
            raise ValueError("unsupported capture fidelity")
        self.capture_fidelity = capture_fidelity
        self.choices: dict[int, _Choice] = {}
        self.usage: Any = None
        self.model = ""
        self.response_id = ""
        self.chunk_count = 0
        self.finalized = False

    def add_response(self, response: Any) -> None:
        self._capture_envelope(response)
        for position, value in enumerate(_get(response, "choices", None) or []):
            index = _get(value, "index", position)
            index = index if isinstance(index, int) else position
            message = _get(value, "message", None)
            if message is None:
                continue
            choice = self._choice(index)
            choice.role = str(_get(message, "role", None) or "assistant")
            content = _get(message, "content", None)
            if content is not None:
                choice.content.append(_string(content))
                if not isinstance(content, str):
                    self._add_media(choice, media_references(content, "output"))
            reasoning = _get(message, "reasoning_content", None)
            if reasoning is not None:
                choice.reasoning.append(_string(reasoning))
            self._add_tools(index, _get(message, "tool_calls", None))
            finish_reason = _get(value, "finish_reason", None)
            if finish_reason is not None:
                choice.finish_reason = str(finish_reason)

    def add_single_response(
        self,
        content: Any,
        *,
        role: str = "assistant",
        finish_reason: str | None = None,
        tool_calls: Any = None,
        usage: Any = None,
        model: str | None = None,
        response_id: str | None = None,
    ) -> None:
        """Normalize providers/callbacks that expose one flattened completion."""

        self.add_response(
            {
                "id": response_id,
                "model": model,
                "usage": usage,
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": finish_reason,
                        "message": {
                            "role": role,
                            "content": content,
                            "tool_calls": tool_calls or [],
                        },
                    }
                ],
            }
        )

    def add_google_response(self, response: Any) -> None:
        """Normalize every Gemini candidate without flattening candidate zero."""

        self._capture_google_envelope(response)
        for position, candidate in enumerate(_get(response, "candidates", None) or []):
            index = _get(candidate, "index", position)
            index = index if isinstance(index, int) else position
            choice = self._choice(index)
            content = _get(candidate, "content", None)
            if content is not None:
                role = _get(content, "role", None)
                if role:
                    choice.role = "assistant" if str(role) == "model" else str(role)
                tool_index = 0
                for part in _get(content, "parts", None) or []:
                    text = _get(part, "text", None)
                    if text is not None and _get(part, "thought", False):
                        choice.reasoning.append(str(text))
                    elif text is not None:
                        choice.content.append(str(text))
                    function_call = _get(part, "function_call", None)
                    if function_call is not None:
                        tool = choice.tool_calls.setdefault(tool_index, _ToolCall(type="function"))
                        call_id = _get(function_call, "id", None)
                        name = _get(function_call, "name", None)
                        arguments = _get(function_call, "args", None)
                        if call_id:
                            tool.id = str(call_id)
                        if name:
                            tool.name = str(name)
                        if arguments is not None:
                            tool.arguments.append(_string(arguments))
                        tool_index += 1
                    inline_data = _get(part, "inline_data", None) or _get(part, "file_data", None)
                    if inline_data is not None:
                        self._add_media(choice, media_references(inline_data, "output"))
            finish_reason = _get(candidate, "finish_reason", None)
            if finish_reason is not None:
                choice.finish_reason = str(finish_reason)

    def add_google_chunk(self, span: Any, chunk: Any) -> None:
        chunk_index = self.chunk_count
        self.chunk_count += 1
        before = {
            index: (
                choice.content.original_bytes,
                choice.reasoning.original_bytes,
                len(choice.tool_calls),
            )
            for index, choice in self.choices.items()
        }
        self.add_google_response(chunk)
        summary = []
        for index in sorted(self.choices):
            choice = self.choices[index]
            previous = before.get(index, (0, 0, 0))
            summary.append(
                {
                    "choice_index": index,
                    "content_bytes": choice.content.original_bytes - previous[0],
                    "reasoning_bytes": choice.reasoning.original_bytes - previous[1],
                    "tool_calls": len(choice.tool_calls) - previous[2],
                    "finish_reason": choice.finish_reason,
                }
            )
        if chunk_index < MAX_SEMANTIC_STREAM_EVENTS and summary:
            span.add_event(
                "neatlogs.stream.chunk",
                {
                    "neatlogs.stream.chunk.index": chunk_index,
                    "neatlogs.stream.chunk.summary": _string({"choices": summary}),
                },
            )

    def add_chunk(self, span: Any, chunk: Any) -> None:
        chunk_index = self.chunk_count
        self.chunk_count += 1
        self._capture_envelope(chunk)
        summary = []
        for position, value in enumerate(_get(chunk, "choices", None) or []):
            index = _get(value, "index", position)
            index = index if isinstance(index, int) else position
            delta = _get(value, "delta", None)
            if delta is None:
                continue
            choice = self._choice(index)
            role = _get(delta, "role", None)
            if role:
                choice.role = str(role)
            content = (
                _string(_get(delta, "content", ""))
                if _get(delta, "content", None) is not None
                else ""
            )
            reasoning = (
                _string(_get(delta, "reasoning_content", ""))
                if _get(delta, "reasoning_content", None) is not None
                else ""
            )
            if content:
                choice.content.append(content)
            raw_content = _get(delta, "content", None)
            if raw_content is not None and not isinstance(raw_content, str):
                self._add_media(choice, media_references(raw_content, "output"))
            if reasoning:
                choice.reasoning.append(reasoning)
            tools = _get(delta, "tool_calls", None) or []
            self._add_tools(index, tools)
            finish_reason = _get(value, "finish_reason", None)
            if finish_reason is not None:
                choice.finish_reason = str(finish_reason)
            summary.append(
                {
                    "choice_index": index,
                    "content_bytes": len(content.encode()),
                    "reasoning_bytes": len(reasoning.encode()),
                    "tool_fragments": len(tools),
                    "finish_reason": finish_reason,
                }
            )
        if chunk_index < MAX_SEMANTIC_STREAM_EVENTS and (summary or self.usage is not None):
            span.add_event(
                "neatlogs.stream.chunk",
                {
                    "neatlogs.stream.chunk.index": chunk_index,
                    "neatlogs.stream.chunk.summary": _string(
                        {"choices": summary, "usage": _get(chunk, "usage", None) is not None}
                    ),
                },
            )

    def apply(self, span: Any) -> None:
        indexes = sorted(self.choices)
        flattened_tool_index = 0
        for choice_index in indexes:
            choice = self.choices[choice_index]
            prefix = f"neatlogs.llm.output_messages.{choice_index}"
            span.set_attribute(f"{prefix}.role", choice.role)
            content = choice.content.value()
            if content:
                span.set_attribute(f"{prefix}.content", content)
            reasoning = choice.reasoning.value()
            if reasoning:
                span.set_attribute(f"{prefix}.thinking", reasoning)
            unique_media = {
                (record.get("sha256"), record.get("reference"), record.get("type")): record
                for record in choice.media_records
            }
            for media_index, record in enumerate(unique_media.values()):
                for key, item in record.items():
                    span.set_attribute(f"{prefix}.media.{media_index}.{key}", item)
            if choice.media_records_dropped:
                span.set_attribute(f"{prefix}.media_dropped_count", choice.media_records_dropped)
            if choice.finish_reason is not None:
                span.set_attribute(
                    f"neatlogs.llm.choices.{choice_index}.finish_reason",
                    choice.finish_reason,
                )
                if choice_index == indexes[0]:
                    span.set_attribute("neatlogs.llm.finish_reason", choice.finish_reason)
            for tool_index in sorted(choice.tool_calls):
                tool = choice.tool_calls[tool_index]
                if not tool.id:
                    context = span.get_span_context()
                    raw = (
                        f"{context.trace_id:032x}:{context.span_id:016x}:{choice_index}:"
                        f"{tool_index}:{tool.name}:"
                        f"{hashlib.sha256(tool.arguments.value().encode()).hexdigest()}"
                    )
                    tool.id = f"nl_{hashlib.sha256(raw.encode()).hexdigest()[:24]}"
                    tool.synthetic = True
                tool_prefix = f"neatlogs.llm.tool_calls.{flattened_tool_index}"
                span.set_attribute(f"{tool_prefix}.id", tool.id)
                if tool.type:
                    span.set_attribute(f"{tool_prefix}.type", tool.type)
                span.set_attribute(f"{tool_prefix}.name", tool.name)
                span.set_attribute(f"{tool_prefix}.arguments", tool.arguments.value())
                if tool.details:
                    span.set_attribute(f"{tool_prefix}.details", tool.details)
                span.set_attribute(f"{tool_prefix}.choice_index", choice_index)
                span.set_attribute(f"{tool_prefix}.tool_call_index", tool_index)
                if tool.synthetic:
                    span.set_attribute(f"{tool_prefix}.id_synthetic", True)
                flattened_tool_index += 1
        if self.model:
            span.set_attribute("neatlogs.llm.model_name", self.model)
        if self.response_id:
            span.set_attribute("neatlogs.llm.response_id", self.response_id)
        self._apply_usage(span)
        span.set_attribute("neatlogs.capture_fidelity", self.capture_fidelity)
        if self.chunk_count:
            span.set_attribute("neatlogs.stream.chunk_count", self.chunk_count)
            if self.chunk_count > MAX_SEMANTIC_STREAM_EVENTS:
                span.set_attribute(
                    "neatlogs.stream.events_dropped",
                    self.chunk_count - MAX_SEMANTIC_STREAM_EVENTS,
                )

    @staticmethod
    def _add_media(choice: _Choice, records: list[dict[str, Any]]) -> None:
        identities = {
            (record.get("sha256"), record.get("reference"), record.get("type"))
            for record in choice.media_records
        }
        for record in records:
            identity = (record.get("sha256"), record.get("reference"), record.get("type"))
            if identity in identities:
                token = record.get("upload_token")
                store = current_media_store()
                if isinstance(token, str) and store is not None:
                    store.release(token)
                continue
            if len(choice.media_records) >= MAX_MEDIA_RECORDS_PER_CHOICE:
                choice.media_records_dropped += 1
                token = record.get("upload_token")
                store = current_media_store()
                if isinstance(token, str) and store is not None:
                    store.release(token)
                continue
            choice.media_records.append(record)
            identities.add(identity)

    def _choice(self, index: int) -> _Choice:
        return self.choices.setdefault(index, _Choice())

    def _add_tools(self, choice_index: int, fragments: Any) -> None:
        for position, fragment in enumerate(fragments or []):
            index = _get(fragment, "index", position)
            index = index if isinstance(index, int) else position
            tool = self._choice(choice_index).tool_calls.setdefault(index, _ToolCall())
            tool_id = _get(fragment, "id", None)
            tool_type = _get(fragment, "type", None)
            function = _get(fragment, "function", None)
            if tool_id:
                tool.id = str(tool_id)
            if tool_type:
                tool.type = str(tool_type)
            name = _get(function, "name", None)
            if name:
                tool.name = str(name)
            arguments = _get(function, "arguments", None)
            if arguments is not None:
                tool.arguments.append(_string(arguments))
            if function is None:
                tool.details = _string(fragment)

    def _capture_envelope(self, value: Any) -> None:
        usage = _get(value, "usage", None)
        if usage is not None:
            self.usage = usage
        model = _get(value, "model", None)
        if model:
            self.model = str(model)
        response_id = _get(value, "id", None)
        if response_id:
            self.response_id = str(response_id)

    def _capture_google_envelope(self, value: Any) -> None:
        usage = _get(value, "usage_metadata", None)
        if usage is not None:
            self.usage = {
                "prompt_tokens": _get(usage, "prompt_token_count", None),
                "completion_tokens": _get(usage, "candidates_token_count", None),
                "total_tokens": _get(usage, "total_token_count", None),
                "prompt_tokens_details": {
                    "cached_tokens": _get(usage, "cached_content_token_count", None)
                },
                "completion_tokens_details": {
                    "reasoning_tokens": _get(usage, "thoughts_token_count", None)
                },
            }
        model = _get(value, "model_version", None) or _get(value, "model", None)
        if model:
            self.model = str(model)
        response_id = _get(value, "response_id", None)
        if response_id:
            self.response_id = str(response_id)

    def _apply_usage(self, span: Any) -> None:
        usage = self.usage
        if usage is None:
            return
        span.set_attribute("neatlogs.llm.usage", _string(usage))
        for source, target in (
            ("prompt_tokens", "prompt"),
            ("completion_tokens", "completion"),
            ("total_tokens", "total"),
        ):
            value = _get(usage, source, None)
            if value is not None:
                span.set_attribute(f"neatlogs.llm.token_count.{target}", value)
        prompt_details = _get(usage, "prompt_tokens_details", None)
        cached = _get(prompt_details, "cached_tokens", None)
        if cached is not None:
            span.set_attribute("neatlogs.llm.token_count.cache_read", cached)
        completion_details = _get(usage, "completion_tokens_details", None)
        reasoning = _get(completion_details, "reasoning_tokens", None)
        if reasoning is not None:
            span.set_attribute("neatlogs.llm.token_count.reasoning", reasoning)


class OpenAIStreamFinalizer:
    """Incremental finalizer consumed by Sync/AsyncStreamWrapper."""

    def __init__(self) -> None:
        self.accumulator = ChoiceAccumulator()

    def on_chunk(self, span: Any, chunk: Any) -> None:
        self.accumulator.add_chunk(span, chunk)

    def finish(
        self,
        span: Any,
        duration_ms: float,
        ttft_ms: float | None,
        *,
        interrupted: bool = False,
    ) -> None:
        if self.accumulator.finalized:
            return
        self.accumulator.finalized = True
        self.accumulator.apply(span)
        span.set_attribute("neatlogs.llm.metrics.duration_ms", round(duration_ms, 3))
        if ttft_ms is not None:
            span.set_attribute("neatlogs.llm.metrics.ttft_ms", round(ttft_ms, 3))
            if duration_ms > ttft_ms:
                span.set_attribute(
                    "neatlogs.llm.metrics.streaming_time_to_generate_ms",
                    round(duration_ms - ttft_ms, 3),
                )
        if interrupted:
            span.set_attribute("neatlogs.stream.cancelled", True)
            span.set_status(StatusCode.UNSET)
        else:
            span.set_status(StatusCode.OK)
        span.end()

    def fail(self, span: Any, error: BaseException) -> None:
        if self.accumulator.finalized:
            return
        self.accumulator.finalized = True
        self.accumulator.apply(span)
        if error.__class__.__name__ in {"CancelledError", "GeneratorExit"}:
            span.set_attribute("neatlogs.stream.cancelled", True)
            span.set_status(StatusCode.UNSET)
        else:
            span.set_status(StatusCode.ERROR, str(error))
            if isinstance(error, Exception):
                span.record_exception(error)
        span.end()


class GoogleStreamFinalizer(OpenAIStreamFinalizer):
    """Incremental Gemini finalizer using the same canonical choice state."""

    def on_chunk(self, span: Any, chunk: Any) -> None:
        self.accumulator.add_google_chunk(span, chunk)
