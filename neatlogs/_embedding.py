"""OpenAI-compatible embedding resource instrumentation without constructor patches."""

import contextlib
import functools
import time
from typing import Any

from opentelemetry.trace import StatusCode

from ._wrap_utils import get_provider_tracer, is_suppressed, serialize


def patch_embeddings(resource: Any, *, provider: str, sync: bool = True) -> None:
    flags = ("_neatlogs_embedding_patched", "_neatlogs_patched", "_neatlogs_azure_patched")
    if resource is None or any(getattr(resource, flag, False) for flag in flags):
        return
    original = getattr(resource, "create", None)
    if not callable(original):
        return
    name = "azure_openai" if provider == "azure" else "openai"

    @contextlib.contextmanager
    def capture(kwargs):
        span = None
        started = time.perf_counter()
        if not is_suppressed():
            with contextlib.suppress(Exception):
                attrs = {
                    "neatlogs.span.kind": "embedding",
                    "neatlogs.llm.provider": provider,
                    "neatlogs.embedding.model_name": kwargs.get("model", ""),
                }
                value = kwargs.get("input", "")
                if isinstance(value, str):
                    attrs["neatlogs.embedding.text"] = value[:10000]
                elif isinstance(value, list):
                    attrs["neatlogs.embedding.text"] = serialize(value[:20])[:10000]
                span = get_provider_tracer().start_span(
                    name=f"{name}.embeddings.create", attributes=attrs
                )
        try:
            yield span
        except BaseException as error:
            if span is not None:
                with contextlib.suppress(Exception):
                    span.set_status(StatusCode.ERROR, str(error))
                    span.record_exception(error)
            raise
        else:
            if span is not None:
                with contextlib.suppress(Exception):
                    span.set_status(StatusCode.OK)
        finally:
            if span is not None:
                with contextlib.suppress(Exception):
                    span.set_attribute(
                        "neatlogs.llm.metrics.duration_ms",
                        round((time.perf_counter() - started) * 1000, 3),
                    )
                with contextlib.suppress(Exception):
                    span.end()

    def record(span, response):
        if span is None:
            return
        with contextlib.suppress(Exception):
            usage = getattr(response, "usage", None)
            if usage is not None:
                if getattr(usage, "prompt_tokens", None) is not None:
                    span.set_attribute("neatlogs.llm.token_count.prompt", usage.prompt_tokens)
                if getattr(usage, "total_tokens", None) is not None:
                    span.set_attribute("neatlogs.embedding.token_count", usage.total_tokens)
            data = getattr(response, "data", None)
            if data is not None:
                span.set_attribute("neatlogs.embedding.count", len(data))
                if data and getattr(data[0], "embedding", None) is not None:
                    span.set_attribute("neatlogs.embedding.dimensions", len(data[0].embedding))
            model = getattr(response, "model", None)
            if model:
                span.set_attribute("neatlogs.embedding.model_name", model)

    if sync:

        @functools.wraps(original)
        def patched(*args, **kwargs):
            with capture(kwargs) as span:
                response = original(*args, **kwargs)
                record(span, response)
                return response

    else:

        @functools.wraps(original)
        async def patched(*args, **kwargs):
            with capture(kwargs) as span:
                response = await original(*args, **kwargs)
                record(span, response)
                return response

    resource.create = patched
    resource._neatlogs_embedding_patched = True
    setattr(
        resource, "_neatlogs_azure_patched" if provider == "azure" else "_neatlogs_patched", True
    )
