"""
Neatlogs wrapper for the Supermemory Python SDK.

Traces the memory calls on a ``supermemory.Supermemory`` or
``supermemory.AsyncSupermemory`` client instance:

  - ``client.search(...)``   -> retriever span
  - ``client.profile(...)``  -> retriever span
  - ``client.add(...)``      -> tool span

Usage (the client is patched in place and also returned):

  >>> import neatlogs
  >>> from supermemory import Supermemory
  >>> client = neatlogs.wrap(Supermemory(api_key=os.environ["SUPERMEMORY_API_KEY"]))
  >>> client.search(namespace="user-1", query="what do they like?")
"""

import inspect
import time
from typing import Any, Callable

from opentelemetry.trace import StatusCode

from ._wrap_utils import get_provider_tracer, is_suppressed, serialize

_PROVIDER = "supermemory"
_TEXT_FIELDS = ("memory", "chunk", "content", "text")


def wrap_supermemory_client(client: Any) -> Any:
    """Wrap a Supermemory client instance. Returns the same client. Idempotent."""
    if getattr(client, "_neatlogs_supermemory_wrapped", False):
        return client

    _patch(client, "search", "retriever", _search_attrs, _search_result)
    _patch(client, "profile", "retriever", _profile_attrs, _profile_result)
    _patch(client, "add", "tool", _add_attrs, _add_result)

    client._neatlogs_supermemory_wrapped = True
    return client


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _namespace(args: tuple, kwargs: dict) -> str:
    value = kwargs.get("namespace", args[0] if args else "")
    return str(value) if value else ""


def _search_attrs(args: tuple, kwargs: dict) -> dict:
    attrs = {
        "neatlogs.retriever.name": "supermemory.search",
        "neatlogs.retriever.query": str(kwargs.get("query", "")),
        "supermemory.namespace": _namespace(args, kwargs),
    }
    if kwargs.get("limit") is not None:
        attrs["neatlogs.retriever.top_k"] = kwargs["limit"]
    return attrs


def _search_result(span: Any, response: Any) -> None:
    results = _get(response, "results", None)
    if results is None:
        return
    span.set_attribute("neatlogs.retriever.document_count", len(results))
    for i, item in enumerate(results):
        for field in _TEXT_FIELDS:
            text = _get(item, field, None)
            if text:
                span.set_attribute(f"neatlogs.retriever.documents.{i}.content", str(text))
                break


def _profile_attrs(args: tuple, kwargs: dict) -> dict:
    return {
        "neatlogs.retriever.name": "supermemory.profile",
        "supermemory.namespace": _namespace(args, kwargs),
    }


def _profile_result(span: Any, response: Any) -> None:
    profile = _get(response, "profile", None)
    if profile is not None:
        span.set_attribute("neatlogs.retriever.profile", serialize(profile))


def _add_attrs(args: tuple, kwargs: dict) -> dict:
    return {
        "neatlogs.tool.name": "supermemory.add",
        "neatlogs.tool.parameters": serialize(
            {"namespace": _namespace(args, kwargs), "content": kwargs.get("content", "")}
        ),
        "supermemory.namespace": _namespace(args, kwargs),
    }


def _add_result(span: Any, response: Any) -> None:
    doc_id = _get(response, "id", None)
    if doc_id:
        span.set_attribute("supermemory.document_id", str(doc_id))


def _patch(client: Any, method: str, kind: str, attrs_fn: Callable, result_fn: Callable) -> None:
    orig = getattr(client, method, None)
    if not callable(orig):
        return

    def _start(args: tuple, kwargs: dict) -> Any:
        attrs = {
            "neatlogs.span.kind": kind,
            "neatlogs.llm.provider": _PROVIDER,
            **attrs_fn(args, kwargs),
        }
        return get_provider_tracer().start_span(name=f"supermemory.{method}", attributes=attrs)

    def _finish(span: Any, response: Any, start: float) -> None:
        try:
            result_fn(span, response)
        except Exception:
            pass
        span.set_attribute("neatlogs.duration_ms", round((time.perf_counter() - start) * 1000, 3))
        span.set_status(StatusCode.OK)
        span.end()

    def _fail(span: Any, exc: Exception) -> None:
        span.set_status(StatusCode.ERROR, str(exc))
        span.record_exception(exc)
        span.end()

    if inspect.iscoroutinefunction(orig):

        async def patched_async(*args: Any, **kwargs: Any) -> Any:
            if is_suppressed():
                return await orig(*args, **kwargs)
            span = _start(args, kwargs)
            start = time.perf_counter()
            try:
                response = await orig(*args, **kwargs)
            except Exception as exc:
                _fail(span, exc)
                raise
            _finish(span, response, start)
            return response

        setattr(client, method, patched_async)
    else:

        def patched(*args: Any, **kwargs: Any) -> Any:
            if is_suppressed():
                return orig(*args, **kwargs)
            span = _start(args, kwargs)
            start = time.perf_counter()
            try:
                response = orig(*args, **kwargs)
            except Exception as exc:
                _fail(span, exc)
                raise
            _finish(span, response, start)
            return response

        setattr(client, method, patched)
