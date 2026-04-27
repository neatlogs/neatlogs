"""
Best-effort HTTP context propagation helpers.

Goal: keep cross-service calls in the same trace even when users are not using
OpenTelemetry server/client auto-instrumentation packages.

This module only injects/extracts W3C trace context (traceparent/tracestate/baggage)
into/from HTTP headers. It does not create HTTP spans.
"""

from __future__ import annotations

import importlib.util
from typing import Any, Iterable, Mapping, MutableMapping, Optional

import wrapt
from opentelemetry import context as context_api
from opentelemetry.propagate import extract, inject
from opentelemetry.propagators.textmap import Getter


def _module_exists(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def _has_traceparent(headers: Mapping[str, Any]) -> bool:
    try:
        for k in headers.keys():
            if str(k).lower() == "traceparent":
                return True
    except Exception:
        pass
    return False


def _coerce_headers(headers: Any) -> MutableMapping[str, str]:
    if headers is None:
        return {}
    if isinstance(headers, dict):
        return headers  # type: ignore[return-value]
    try:
        return dict(headers)
    except Exception:
        return {}


def _inject_trace_context(headers: MutableMapping[str, str]) -> None:
    if _has_traceparent(headers):
        return
    try:
        inject(headers)
    except Exception:
        return


class _WSGIEnvironGetter(Getter[Mapping[str, Any]]):
    """
    Map WSGI environ keys (HTTP_TRACEPARENT) to header names (traceparent).
    """

    def get(self, carrier: Mapping[str, Any], key: str) -> Optional[Iterable[str]]:
        if not carrier or not key:
            return None
        try:
            env_key = "HTTP_" + key.upper().replace("-", "_")
            val = carrier.get(env_key)
            if val is None:
                return None
            return [str(val)]
        except Exception:
            return None

    def keys(self, carrier: Mapping[str, Any]) -> Iterable[str]:
        return []


def patch_requests() -> None:
    """
    Inject trace context into outgoing requests via `requests`.
    """
    if not _module_exists("requests"):
        return
    if _module_exists("opentelemetry.instrumentation.requests"):
        return

    try:
        import requests  # noqa: F401
    except Exception:
        return

    if getattr(patch_requests, "_patched", False):
        return

    def _wrapper(wrapped, instance, args, kwargs):
        headers = _coerce_headers(kwargs.get("headers"))
        _inject_trace_context(headers)
        kwargs["headers"] = headers
        return wrapped(*args, **kwargs)

    try:
        wrapt.wrap_function_wrapper(
            module="requests.sessions",
            name="Session.request",
            wrapper=_wrapper,
        )
        patch_requests._patched = True  # type: ignore[attr-defined]
    except Exception:
        return


def patch_aiohttp_client() -> None:
    """
    Inject trace context into outgoing requests via `aiohttp.ClientSession`.
    """
    if not _module_exists("aiohttp"):
        return
    if _module_exists("opentelemetry.instrumentation.aiohttp_client"):
        return

    try:
        import aiohttp  # noqa: F401
    except Exception:
        return

    if getattr(patch_aiohttp_client, "_patched", False):
        return

    def _wrapper(wrapped, instance, args, kwargs):
        headers = _coerce_headers(kwargs.get("headers"))
        _inject_trace_context(headers)
        kwargs["headers"] = headers
        return wrapped(*args, **kwargs)

    try:
        wrapt.wrap_function_wrapper(
            module="aiohttp.client",
            name="ClientSession._request",
            wrapper=_wrapper,
        )
        patch_aiohttp_client._patched = True  # type: ignore[attr-defined]
    except Exception:
        return


def patch_flask_server() -> None:
    """
    Extract trace context from incoming Flask/Werkzeug requests.

    This attaches the extracted OpenTelemetry context for the lifetime of the request.
    Any spans started inside the request (workflows, LLM calls, tools) will share the
    same `trace_id` and will have a `parent_span_id` corresponding to the remote parent.
    """
    if not _module_exists("flask"):
        return
    if _module_exists("opentelemetry.instrumentation.flask"):
        return

    try:
        import flask  # noqa: F401
    except Exception:
        return

    if getattr(patch_flask_server, "_patched", False):
        return

    getter = _WSGIEnvironGetter()

    def _wrapper(wrapped, instance, args, kwargs):
        environ = args[0] if args else None
        token = None
        try:
            if isinstance(environ, Mapping):
                ctx = extract(environ, getter=getter)
                token = context_api.attach(ctx)
        except Exception:
            token = None
        try:
            return wrapped(*args, **kwargs)
        finally:
            if token is not None:
                try:
                    context_api.detach(token)
                except Exception:
                    pass

    try:
        wrapt.wrap_function_wrapper(
            module="flask.app",
            name="Flask.wsgi_app",
            wrapper=_wrapper,
        )
        patch_flask_server._patched = True  # type: ignore[attr-defined]
    except Exception:
        return


def patch_http_context_propagation() -> None:
    """
    Apply best-effort context propagation patches for commonly used HTTP libs/frameworks.
    """
    patch_requests()
    patch_aiohttp_client()
    patch_flask_server()
