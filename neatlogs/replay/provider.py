"""Lossless non-streaming OpenAI/Google restoration and bounded raw capture."""

from __future__ import annotations

import hashlib
import re
from functools import lru_cache

import rfc8785

from ..core.media import media_references, sanitize_media_payload
from .contracts import (
    MAX_INPUT_BYTES,
    ReplayError,
    canonical_json,
    json_value,
    object_fields,
    parse_json_bytes,
)
from .session import ReplayHit, current_replay, lookup_replay

MODEL_CAPTURE_ATTRIBUTE = "neatlogs.replay.model_capture"
OPERATIONS = {
    "openai.chat.completions.create",
    "openai.responses.create",
    "google_genai.models.generate_content",
}
_TRANSPORT = re.compile(
    r"^(?:api_?key|authorization|proxy_?authorization|headers|extra_?headers|http_?options|abort_?signal|credentials|access_token|client_secret)$",
    re.I,
)


def ecmascript_json(value):
    """JSON.stringify bytes, with ECMAScript numbers and integer-property ordering."""
    value = json_value(value)

    def emit(item):
        if type(item) is list:
            return b"[" + b",".join(emit(child) for child in item) + b"]"
        if type(item) is dict:
            indexes = sorted(
                (
                    key
                    for key in item
                    if key.isascii()
                    and key.isdigit()
                    and str(int(key)) == key
                    and int(key) < 2**32 - 1
                ),
                key=int,
            )
            keys = indexes + [key for key in item if key not in indexes]
            return (
                b"{" + b",".join(rfc8785.dumps(key) + b":" + emit(item[key]) for key in keys) + b"}"
            )
        return rfc8785.dumps(item)

    return emit(value)


def safe_body(value):
    body = json_value(value, limit=MAX_INPUT_BYTES)
    nodes = 0

    def inspect(item, depth):
        nonlocal nodes
        nodes += 1
        if nodes > 4096 or depth > 32:
            raise ReplayError("Provider capture exceeds its traversal limit.")
        if type(item) is dict:
            for key, child in item.items():
                if _TRANSPORT.fullmatch(key) or key in {"__proto__", "prototype", "constructor"}:
                    raise ReplayError("Provider transport data cannot be captured.")
                inspect(child, depth + 1)
        elif type(item) is list:
            for child in item:
                inspect(child, depth + 1)

    inspect(body, 0)
    if media_references(body, "replay", store=None) or canonical_json(
        sanitize_media_payload(body, "replay", store=None)
    ) != canonical_json(body):
        raise ReplayError("Provider media cannot be losslessly replayed.")
    return body


def is_complete_provider_response(operation, value):
    if type(value) is not dict:
        return False
    if operation == "openai.chat.completions.create":
        choices = value.get("choices")
        return (
            value.get("object") == "chat.completion"
            and type(value.get("id")) is str
            and type(value.get("created")) in (int, float)
            and type(value.get("model")) is str
            and type(choices) is list
            and all(
                type(choice) is dict
                and type(choice.get("index")) is int
                and (
                    choice.get("finish_reason") is None or type(choice.get("finish_reason")) is str
                )
                and type(choice.get("message")) is dict
                and choice["message"].get("role") == "assistant"
                for choice in choices
            )
        )
    if operation == "openai.responses.create":
        return (
            value.get("object") == "response"
            and value.get("status") == "completed"
            and type(value.get("id")) is str
            and type(value.get("created_at")) in (int, float)
            and type(value.get("model")) is str
            and type(value.get("output_text")) is str
            and type(value.get("output")) is list
            and all(type(item) is dict for item in value["output"])
        )
    if operation == "google_genai.models.generate_content":
        candidates = value.get("candidates")
        return (
            type(candidates) is list
            and len(candidates) > 0
            and all(
                type(candidate) is dict
                and type(candidate.get("content")) is dict
                and type(candidate["content"].get("parts")) is list
                and all(type(part) is dict for part in candidate["content"]["parts"])
                for candidate in candidates
            )
        )
    return False


@lru_cache(maxsize=3)
def response_class(operation):
    if operation == "openai.chat.completions.create":
        from openai.types.chat import ChatCompletion

        return ChatCompletion
    if operation == "openai.responses.create":
        from openai.types.responses import Response

        return Response
    if operation == "google_genai.models.generate_content":
        from google.genai.types import GenerateContentResponse

        return GenerateContentResponse
    raise ReplayError("Unsupported replay provider operation.")


def response_data(operation, response):
    if type(response) is not response_class(operation):
        raise ReplayError("Provider response does not have a supported SDK shape.")
    body = response.model_dump(mode="json", by_alias=True, exclude_unset=True)
    if operation == "openai.responses.create":
        body["output_text"] = response.output_text
    body.pop("sdkHttpResponse", None)
    body.pop("_request_id", None)
    return safe_body(body)


def restore_response(operation, saved):
    if not is_complete_provider_response(operation, saved):
        raise ReplayError("Saved provider response is incomplete.")
    body = safe_body(saved)
    response = response_class(operation).model_validate(body)
    if canonical_json(response_data(operation, response)) != canonical_json(body):
        raise ReplayError("The installed SDK cannot losslessly restore this response.")
    return response


def _can_restore(operation, value):
    try:
        restore_response(operation, value)
        return True
    except Exception:
        return False


def provider_replay(operation, input, *, supported, span):
    if current_replay() is None:
        return None
    request = object()
    if supported:
        try:
            request = safe_body(input)
        except Exception:
            pass
    decision = lookup_replay(
        "model",
        operation,
        request,
        span=span,
        can_restore=lambda output: _can_restore(operation, output),
    )
    return (
        restore_response(operation, decision.entry.output)
        if isinstance(decision, ReplayHit)
        else None
    )


def capture_digest(operation, input, output):
    return hashlib.sha256(
        ecmascript_json({"version": 1, "operation": operation, "input": input, "output": output})
    ).hexdigest()


def prepare_provider_capture(span, operation, request, *, supported):
    if not supported or not span.is_recording():
        return lambda response: None
    try:
        input = safe_body(request)
    except Exception:
        return lambda response: None

    def capture(response):
        try:
            output = response_data(operation, response)
            if not is_complete_provider_response(operation, output):
                return
            content = ecmascript_json(
                {
                    "version": 1,
                    "operation": operation,
                    "input": input,
                    "output": output,
                    "digest": capture_digest(operation, input, output),
                }
            )
            if len(content) <= MAX_INPUT_BYTES:
                span.set_attribute(MODEL_CAPTURE_ATTRIBUTE, content.decode())
        except Exception:
            return

    return capture


def read_provider_capture(value):
    if type(value) is not str:
        raise ReplayError("Missing complete provider capture.")
    fields = object_fields(
        parse_json_bytes(value.encode(), limit=MAX_INPUT_BYTES),
        {"version", "operation", "input", "output", "digest"},
    )
    operation = fields["operation"]
    if (
        type(fields["version"]) is not int
        or fields["version"] != 1
        or type(operation) is not str
        or operation not in OPERATIONS
    ):
        raise ReplayError("Unsupported provider capture.")
    input, output = safe_body(fields["input"]), safe_body(fields["output"])
    if capture_digest(operation, input, output) != fields[
        "digest"
    ] or not is_complete_provider_response(operation, output):
        raise ReplayError("Provider capture changed or is incomplete.")
    return {"name": operation, "input": input, "output": output}


class _CachedModelSpan:
    def __init__(self, span):
        self._span = span
        for metric in ("prompt", "completion", "total", "reasoning", "cache_read", "cache_write"):
            span.set_attribute(f"neatlogs.llm.token_count.{metric}", 0)

    def set_attribute(self, key, value):
        if key.startswith(
            ("neatlogs.llm.token_count.", "neatlogs.llm.cost", "llm.token_count.", "gen_ai.usage.")
        ):
            value = 0
        return self._span.set_attribute(key, value)

    def set_attributes(self, values):
        for key, value in values.items():
            self.set_attribute(key, value)

    def __getattr__(self, key):
        return getattr(self._span, key)


def cached_model_span(span):
    return _CachedModelSpan(span)
