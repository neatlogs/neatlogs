"""
Neatlogs Anthropic wrapper.

Two usage patterns:

  1. Explicit wrap (primary):
     >>> import neatlogs, anthropic
     >>> client = neatlogs.wrap(anthropic.Anthropic())
     >>> client.messages.create(...)

  2. Import replacement (Langfuse-style):
     >>> from neatlogs.anthropic import anthropic
     >>> client = anthropic.Anthropic()
     >>> client.messages.create(...)
"""

import inspect
import sys
import time
from contextvars import ContextVar
from types import SimpleNamespace
from typing import Any, List, Optional

from opentelemetry.trace import StatusCode

from ._wrap_utils import (
    AsyncStreamWrapper,
    SyncStreamWrapper,
    get_provider_tracer,
    is_suppressed,
    serialize,
)
from .core.media import set_media_attributes

_PATCHED = False
_ORIG_INIT = None
_ORIG_ASYNC_INIT = None
_RAW_CALL_MARKER: ContextVar[Optional[dict]] = ContextVar(
    "_neatlogs_anthropic_raw_marker", default=None
)


def _is_raw_response(response: Any) -> bool:
    return callable(getattr(response, "parse", None))


def _is_stream_object(obj: Any, is_stream: bool = False) -> bool:
    if hasattr(obj, "__next__") or hasattr(obj, "__anext__"):
        return True
    if is_stream and (hasattr(obj, "__iter__") or hasattr(obj, "__aiter__")):
        return not isinstance(obj, (str, bytes, dict, list, tuple))
    return False


def _raw_response_mode(kwargs: dict) -> Optional[str]:
    extra_headers = kwargs.get("extra_headers")
    if isinstance(extra_headers, dict):
        for k, v in extra_headers.items():
            if str(k).lower() == "x-stainless-raw-response":
                return str(v).lower()
    return None


def _reset_raw_wrappers(resource: Any) -> None:
    d = getattr(resource, "__dict__", None)
    if not isinstance(d, dict):
        return
    cls = type(resource)
    for attr in ("with_raw_response", "with_streaming_response"):
        if attr in d and hasattr(cls, attr):
            existing = d.pop(attr, None)
            if existing is not None:
                try:
                    existing.__init__(resource)
                except Exception:
                    pass


def _instrument_raw_response(
    span: Any,
    response: Any,
    start: float,
    is_stream: bool,
    finalize_response: Any,
    stream_finalizer: Any,
    raw_mode: Optional[str] = None,
) -> Any:
    orig_parse = response.parse
    orig_close = getattr(response, "close", None)
    defer_end = raw_mode == "stream"
    state: dict = {
        "finalized": False,
        "finalized_with_custom": False,
        "span_ended": False,
        "end_requested": False,
        "closed": False,
        "pending_status": None,
        "wrapped_streams": [],
        "cache": {},
    }

    class _SpanProxy:
        def set_attribute(self, key: str, value: Any) -> None:
            if not state["span_ended"] and not state["end_requested"]:
                span.set_attribute(key, value)

        def set_status(self, *args: Any, **kwargs: Any) -> None:
            if not state["span_ended"] and not state["end_requested"]:
                if defer_end:
                    state["pending_status"] = (args, kwargs)
                else:
                    span.set_status(*args, **kwargs)

        def record_exception(self, *args: Any, **kwargs: Any) -> None:
            if (
                not state["span_ended"]
                and not state["end_requested"]
                and hasattr(span, "record_exception")
            ):
                span.record_exception(*args, **kwargs)

        def add_event(self, *args: Any, **kwargs: Any) -> None:
            if (
                not state["span_ended"]
                and not state["end_requested"]
                and hasattr(span, "add_event")
            ):
                span.add_event(*args, **kwargs)

        def get_span_context(self, *args: Any, **kwargs: Any) -> Any:
            if hasattr(span, "get_span_context"):
                return span.get_span_context(*args, **kwargs)
            return None

        def end(self, *args: Any, **kwargs: Any) -> None:
            state["end_requested"] = True
            if not defer_end and not state["span_ended"]:
                state["span_ended"] = True
                span.end(*args, **kwargs)

        def __getattr__(self, name: str) -> Any:
            return getattr(span, name)

    span_proxy = _SpanProxy()

    def _flush_span_end() -> None:
        if not state["span_ended"]:
            pending = state.get("pending_status")
            if pending is not None:
                state["pending_status"] = None
                span.set_status(*pending[0], **pending[1])
            state["span_ended"] = True
            span.end()

    def _cache_key(p_args: tuple, p_kwargs: dict) -> Any:
        try:
            key = (p_args, tuple(sorted(p_kwargs.items())))
            hash(key)
            return key
        except TypeError:
            return (
                tuple((id(a), type(a)) for a in p_args),
                tuple(sorted((k, id(v), type(v)) for k, v in p_kwargs.items())),
            )

    def _extract_for_span(parsed_obj: Any) -> Any:
        import json as _json

        if isinstance(parsed_obj, (bytes, bytearray, str)):
            try:
                loaded = _json.loads(parsed_obj)
                if isinstance(loaded, dict):
                    return loaded
            except Exception:
                pass
        elif not isinstance(parsed_obj, dict) and getattr(parsed_obj, "usage", None) is None:
            http_resp = getattr(response, "http_response", None)
            if http_resp is not None and "_content" in getattr(http_resp, "__dict__", {}):
                try:
                    loaded = http_resp.json()
                    if isinstance(loaded, dict):
                        return loaded
                except Exception:
                    pass
        return parsed_obj

    def patched_parse(*p_args, **p_kwargs):
        key = _cache_key(p_args, p_kwargs)
        if key in state["cache"]:
            cached = state["cache"][key]
            if not state["finalized"] and not _is_stream_object(cached, is_stream):
                state["end_requested"] = False
                finalize_response(
                    span_proxy,
                    _extract_for_span(cached),
                    (time.perf_counter() - start) * 1000,
                )
                state["finalized"] = True
            return cached

        try:
            try:
                parsed = orig_parse(*p_args, **p_kwargs)
            except (ValueError, RuntimeError, TypeError):
                http_resp = getattr(response, "http_response", None)
                if p_kwargs.get("to") is dict and http_resp is not None:
                    if hasattr(http_resp, "read"):
                        http_resp.read()
                    parsed = http_resp.json()
                elif p_kwargs.get("to") is bytes and http_resp is not None:
                    parsed = (
                        http_resp.read()
                        if hasattr(http_resp, "read")
                        else getattr(http_resp, "content", b"")
                    )
                else:
                    raise
        except BaseException as err:
            if not state["finalized"]:
                state["finalized"] = True
                span_proxy.set_status(StatusCode.ERROR, str(err))
                if isinstance(err, Exception):
                    span_proxy.record_exception(err)
                span_proxy.end()
            raise

        if _is_stream_object(parsed, is_stream):
            span_proxy.set_attribute("neatlogs.llm.is_streaming", True)
            finalizer = (
                stream_finalizer() if isinstance(stream_finalizer, type) else stream_finalizer
            )
            wrapped = SyncStreamWrapper(parsed, span_proxy, finalizer)
            state["wrapped_streams"].append(wrapped)
            state["cache"][key] = wrapped
            state["finalized"] = True
            return wrapped

        state["cache"][key] = parsed
        is_custom = bool(p_args or p_kwargs)
        if not state["finalized"] or (
            defer_end and state["finalized_with_custom"] and not is_custom
        ):
            state["end_requested"] = False
            finalize_response(
                span_proxy,
                _extract_for_span(parsed),
                (time.perf_counter() - start) * 1000,
            )
            state["finalized"] = True
            state["finalized_with_custom"] = is_custom
        return parsed

    def _finalize_on_close(active_exc: Optional[BaseException]) -> None:
        if state["closed"]:
            return
        state["closed"] = True
        streams = state["wrapped_streams"]
        if streams:
            ws = streams[0]
            if not ws._finalized:
                if active_exc is not None:
                    ws._finalize_error(active_exc)
                else:
                    ws._finalize(interrupted=not ws._exhausted)
            elif active_exc is not None and not state["span_ended"]:
                state["pending_status"] = None
                span.set_status(StatusCode.ERROR, str(active_exc))
                if isinstance(active_exc, Exception) and hasattr(span, "record_exception"):
                    span.record_exception(active_exc)
            for s_wrap in streams:
                s_wrap._finalized = True
            _flush_span_end()
            return

        if state["finalized"]:
            if active_exc is not None and not state["span_ended"]:
                state["pending_status"] = None
                span.set_status(StatusCode.ERROR, str(active_exc))
                if isinstance(active_exc, Exception) and hasattr(span, "record_exception"):
                    span.record_exception(active_exc)
            _flush_span_end()
            return

        state["finalized"] = True
        if active_exc is not None:
            state["pending_status"] = None
            span.set_status(StatusCode.ERROR, str(active_exc))
            if isinstance(active_exc, Exception) and hasattr(span, "record_exception"):
                span.record_exception(active_exc)
            _flush_span_end()
        elif is_stream or getattr(response, "_is_sse_stream", False):
            finalizer = (
                stream_finalizer() if isinstance(stream_finalizer, type) else stream_finalizer
            )
            dummy = SyncStreamWrapper(iter(()), span_proxy, finalizer)
            dummy._finalize(interrupted=True)
            _flush_span_end()
        else:
            state["pending_status"] = None
            span.set_attribute(
                "neatlogs.llm.metrics.duration_ms",
                round((time.perf_counter() - start) * 1000, 3),
            )
            span.set_status(StatusCode.OK)
            _flush_span_end()

    def patched_close(*c_args, **c_kwargs):
        active_exc = getattr(response, "_neatlogs_active_exc", None) or sys.exc_info()[1]
        response._neatlogs_active_exc = None
        _finalize_on_close(active_exc)
        if callable(orig_close):
            return orig_close(*c_args, **c_kwargs)
        return None

    def force_close(active_exc: Optional[BaseException] = None):
        exc = active_exc or getattr(response, "_neatlogs_active_exc", None) or sys.exc_info()[1]
        response._neatlogs_active_exc = None
        if not state["closed"] and callable(orig_close):
            response._neatlogs_active_exc = exc
            return patched_close()
        _finalize_on_close(exc)
        return None

    response.parse = patched_parse
    response._neatlogs_state = state
    response._neatlogs_force_close = force_close
    response.close = patched_close
    if not defer_end and not is_stream and not getattr(response, "_is_sse_stream", False):
        http_resp = getattr(response, "http_response", None)
        if http_resp is not None and "_content" in getattr(http_resp, "__dict__", {}):
            try:
                loaded = http_resp.json()
                if isinstance(loaded, dict):
                    state["end_requested"] = False
                    finalize_response(
                        span_proxy,
                        loaded,
                        (time.perf_counter() - start) * 1000,
                    )
                    state["finalized"] = True
            except Exception:
                pass
    return response


async def _instrument_async_raw_response(
    span: Any,
    response: Any,
    start: float,
    is_stream: bool,
    finalize_response: Any,
    stream_finalizer: Any,
    raw_mode: Optional[str] = None,
) -> Any:
    orig_parse = response.parse
    orig_close = getattr(response, "close", None)
    parse_is_coro = inspect.iscoroutinefunction(orig_parse)
    close_is_coro = inspect.iscoroutinefunction(orig_close)
    defer_end = raw_mode == "stream"
    state: dict = {
        "finalized": False,
        "finalized_with_custom": False,
        "span_ended": False,
        "end_requested": False,
        "closed": False,
        "pending_status": None,
        "wrapped_streams": [],
        "cache": {},
    }

    class _SpanProxy:
        def set_attribute(self, key: str, value: Any) -> None:
            if not state["span_ended"] and not state["end_requested"]:
                span.set_attribute(key, value)

        def set_status(self, *args: Any, **kwargs: Any) -> None:
            if not state["span_ended"] and not state["end_requested"]:
                if defer_end:
                    state["pending_status"] = (args, kwargs)
                else:
                    span.set_status(*args, **kwargs)

        def record_exception(self, *args: Any, **kwargs: Any) -> None:
            if (
                not state["span_ended"]
                and not state["end_requested"]
                and hasattr(span, "record_exception")
            ):
                span.record_exception(*args, **kwargs)

        def add_event(self, *args: Any, **kwargs: Any) -> None:
            if (
                not state["span_ended"]
                and not state["end_requested"]
                and hasattr(span, "add_event")
            ):
                span.add_event(*args, **kwargs)

        def get_span_context(self, *args: Any, **kwargs: Any) -> Any:
            if hasattr(span, "get_span_context"):
                return span.get_span_context(*args, **kwargs)
            return None

        def end(self, *args: Any, **kwargs: Any) -> None:
            state["end_requested"] = True
            if not defer_end and not state["span_ended"]:
                state["span_ended"] = True
                span.end(*args, **kwargs)

        def __getattr__(self, name: str) -> Any:
            return getattr(span, name)

    span_proxy = _SpanProxy()

    def _flush_span_end() -> None:
        if not state["span_ended"]:
            pending = state.get("pending_status")
            if pending is not None:
                state["pending_status"] = None
                span.set_status(*pending[0], **pending[1])
            state["span_ended"] = True
            span.end()

    def _cache_key(p_args: tuple, p_kwargs: dict) -> Any:
        try:
            key = (p_args, tuple(sorted(p_kwargs.items())))
            hash(key)
            return key
        except TypeError:
            return (
                tuple((id(a), type(a)) for a in p_args),
                tuple(sorted((k, id(v), type(v)) for k, v in p_kwargs.items())),
            )

    def _extract_for_span(parsed_obj: Any) -> Any:
        import json as _json

        if isinstance(parsed_obj, (bytes, bytearray, str)):
            try:
                loaded = _json.loads(parsed_obj)
                if isinstance(loaded, dict):
                    return loaded
            except Exception:
                pass
        elif not isinstance(parsed_obj, dict) and getattr(parsed_obj, "usage", None) is None:
            http_resp = getattr(response, "http_response", None)
            if http_resp is not None and "_content" in getattr(http_resp, "__dict__", {}):
                try:
                    loaded = http_resp.json()
                    if isinstance(loaded, dict):
                        return loaded
                except Exception:
                    pass
        return parsed_obj

    def _handle_parsed(parsed: Any, key: Any, is_custom: bool) -> Any:
        if _is_stream_object(parsed, is_stream):
            span_proxy.set_attribute("neatlogs.llm.is_streaming", True)
            finalizer = (
                stream_finalizer() if isinstance(stream_finalizer, type) else stream_finalizer
            )
            wrapped = AsyncStreamWrapper(parsed, span_proxy, finalizer)
            state["wrapped_streams"].append(wrapped)
            state["cache"][key] = wrapped
            state["finalized"] = True
            return wrapped

        state["cache"][key] = parsed
        if not state["finalized"] or (
            defer_end and state["finalized_with_custom"] and not is_custom
        ):
            state["end_requested"] = False
            finalize_response(
                span_proxy,
                _extract_for_span(parsed),
                (time.perf_counter() - start) * 1000,
            )
            state["finalized"] = True
            state["finalized_with_custom"] = is_custom
        return parsed

    async def patched_async_parse(*p_args, **p_kwargs):
        key = _cache_key(p_args, p_kwargs)
        if key in state["cache"]:
            cached = state["cache"][key]
            if not state["finalized"] and not _is_stream_object(cached, is_stream):
                state["end_requested"] = False
                finalize_response(
                    span_proxy,
                    _extract_for_span(cached),
                    (time.perf_counter() - start) * 1000,
                )
                state["finalized"] = True
            return cached

        try:
            try:
                parsed = await orig_parse(*p_args, **p_kwargs)
            except (ValueError, RuntimeError, TypeError):
                http_resp = getattr(response, "http_response", None)
                if p_kwargs.get("to") is dict and http_resp is not None:
                    if hasattr(http_resp, "aread") and inspect.iscoroutinefunction(http_resp.aread):
                        await http_resp.aread()
                    elif hasattr(http_resp, "read"):
                        http_resp.read()
                    parsed = http_resp.json()
                elif p_kwargs.get("to") is bytes and http_resp is not None:
                    if hasattr(http_resp, "aread") and inspect.iscoroutinefunction(http_resp.aread):
                        parsed = await http_resp.aread()
                    elif hasattr(http_resp, "read"):
                        parsed = http_resp.read()
                    else:
                        parsed = getattr(http_resp, "content", b"")
                else:
                    raise
        except BaseException as err:
            if not state["finalized"]:
                state["finalized"] = True
                span_proxy.set_status(StatusCode.ERROR, str(err))
                if isinstance(err, Exception):
                    span_proxy.record_exception(err)
                span_proxy.end()
            raise

        return _handle_parsed(parsed, key, bool(p_args or p_kwargs))

    def patched_sync_parse(*p_args, **p_kwargs):
        key = _cache_key(p_args, p_kwargs)
        if key in state["cache"]:
            cached = state["cache"][key]
            if not state["finalized"] and not _is_stream_object(cached, is_stream):
                state["end_requested"] = False
                finalize_response(
                    span_proxy,
                    _extract_for_span(cached),
                    (time.perf_counter() - start) * 1000,
                )
                state["finalized"] = True
            return cached

        try:
            try:
                parsed = orig_parse(*p_args, **p_kwargs)
            except (ValueError, RuntimeError, TypeError):
                http_resp = getattr(response, "http_response", None)
                if p_kwargs.get("to") is dict and http_resp is not None:
                    if hasattr(http_resp, "read"):
                        http_resp.read()
                    parsed = http_resp.json()
                elif p_kwargs.get("to") is bytes and http_resp is not None:
                    parsed = (
                        http_resp.read()
                        if hasattr(http_resp, "read")
                        else getattr(http_resp, "content", b"")
                    )
                else:
                    raise
        except BaseException as err:
            if not state["finalized"]:
                state["finalized"] = True
                span_proxy.set_status(StatusCode.ERROR, str(err))
                if isinstance(err, Exception):
                    span_proxy.record_exception(err)
                span_proxy.end()
            raise

        return _handle_parsed(parsed, key, bool(p_args or p_kwargs))

    def _finalize_on_close(active_exc: Optional[BaseException]) -> None:
        if state["closed"]:
            return
        state["closed"] = True
        streams = state["wrapped_streams"]
        if streams:
            ws = streams[0]
            if not ws._finalized:
                if active_exc is not None:
                    ws._finalize_error(active_exc)
                else:
                    ws._finalize(interrupted=not ws._exhausted)
            elif active_exc is not None and not state["span_ended"]:
                state["pending_status"] = None
                span.set_status(StatusCode.ERROR, str(active_exc))
                if isinstance(active_exc, Exception) and hasattr(span, "record_exception"):
                    span.record_exception(active_exc)
            for s_wrap in streams:
                s_wrap._finalized = True
            _flush_span_end()
            return

        if state["finalized"]:
            if active_exc is not None and not state["span_ended"]:
                state["pending_status"] = None
                span.set_status(StatusCode.ERROR, str(active_exc))
                if isinstance(active_exc, Exception) and hasattr(span, "record_exception"):
                    span.record_exception(active_exc)
            _flush_span_end()
            return

        state["finalized"] = True
        if active_exc is not None:
            state["pending_status"] = None
            span.set_status(StatusCode.ERROR, str(active_exc))
            if isinstance(active_exc, Exception) and hasattr(span, "record_exception"):
                span.record_exception(active_exc)
            _flush_span_end()
        elif is_stream or getattr(response, "_is_sse_stream", False):
            finalizer = (
                stream_finalizer() if isinstance(stream_finalizer, type) else stream_finalizer
            )
            dummy = AsyncStreamWrapper(iter(()), span_proxy, finalizer)
            dummy._finalize(interrupted=True)
            _flush_span_end()
        else:
            state["pending_status"] = None
            span.set_attribute(
                "neatlogs.llm.metrics.duration_ms",
                round((time.perf_counter() - start) * 1000, 3),
            )
            span.set_status(StatusCode.OK)
            _flush_span_end()

    async def patched_async_close(*c_args, **c_kwargs):
        active_exc = getattr(response, "_neatlogs_active_exc", None) or sys.exc_info()[1]
        response._neatlogs_active_exc = None
        _finalize_on_close(active_exc)
        if callable(orig_close):
            return (
                await orig_close(*c_args, **c_kwargs)
                if close_is_coro
                else orig_close(*c_args, **c_kwargs)
            )
        return None

    def patched_sync_close(*c_args, **c_kwargs):
        active_exc = getattr(response, "_neatlogs_active_exc", None) or sys.exc_info()[1]
        response._neatlogs_active_exc = None
        _finalize_on_close(active_exc)
        if callable(orig_close):
            return orig_close(*c_args, **c_kwargs)
        return None

    async def force_async_close(active_exc: Optional[BaseException] = None):
        exc = active_exc or getattr(response, "_neatlogs_active_exc", None) or sys.exc_info()[1]
        response._neatlogs_active_exc = None
        if not state["closed"] and callable(orig_close):
            response._neatlogs_active_exc = exc
            return await patched_async_close()
        _finalize_on_close(exc)
        return None

    def force_sync_close(active_exc: Optional[BaseException] = None):
        exc = active_exc or getattr(response, "_neatlogs_active_exc", None) or sys.exc_info()[1]
        response._neatlogs_active_exc = None
        if not state["closed"] and callable(orig_close):
            response._neatlogs_active_exc = exc
            return patched_sync_close()
        _finalize_on_close(exc)
        return None

    response.parse = patched_async_parse if parse_is_coro else patched_sync_parse
    response._neatlogs_state = state
    response._neatlogs_force_close = force_async_close if close_is_coro else force_sync_close
    response.close = patched_async_close if close_is_coro else patched_sync_close
    if not defer_end and not is_stream and not getattr(response, "_is_sse_stream", False):
        http_resp = getattr(response, "http_response", None)
        if http_resp is not None and "_content" in getattr(http_resp, "__dict__", {}):
            try:
                loaded = http_resp.json()
                if isinstance(loaded, dict):
                    state["end_requested"] = False
                    finalize_response(
                        span_proxy,
                        loaded,
                        (time.perf_counter() - start) * 1000,
                    )
                    state["finalized"] = True
            except Exception:
                pass
    return response


class _StreamingContextManagerWrapper:
    def __init__(
        self,
        cm: Any,
        start_span_fn: Any,
        finalize_response: Any,
        stream_finalizer: Any,
        kwargs: dict,
        marker: dict,
        start: float,
    ) -> None:
        self._cm = cm
        self._start_span_fn = start_span_fn
        self._finalize_response = finalize_response
        self._stream_finalizer = stream_finalizer
        self._kwargs = kwargs
        self._marker = marker
        self._start = start
        self._response = None
        self._wrapped_stream = None
        self._span = None

    def __enter__(self):
        if is_suppressed():
            return self._cm.__enter__()
        ctx_tok = _RAW_CALL_MARKER.set(self._marker)
        try:
            resp = self._cm.__enter__()
        except BaseException as err:
            _RAW_CALL_MARKER.reset(ctx_tok)
            if not self._marker["handled"]:
                span = self._start_span_fn(self._kwargs)
                span.set_status(StatusCode.ERROR, str(err))
                if isinstance(err, Exception):
                    span.record_exception(err)
                span.end()
            raise
        _RAW_CALL_MARKER.reset(ctx_tok)
        if self._marker["handled"]:
            self._response = resp
            return resp
        span = self._start_span_fn(self._kwargs)
        self._span = span
        is_stream = bool(self._kwargs.get("stream", False))
        if _is_raw_response(resp):
            self._response = _instrument_raw_response(
                span,
                resp,
                self._start,
                is_stream,
                self._finalize_response,
                self._stream_finalizer,
                "stream",
            )
            return self._response
        if _is_stream_object(resp, is_stream):
            span.set_attribute("neatlogs.llm.is_streaming", True)
            finalizer = (
                self._stream_finalizer()
                if isinstance(self._stream_finalizer, type)
                else self._stream_finalizer
            )
            self._wrapped_stream = SyncStreamWrapper(resp, span, finalizer)
            return self._wrapped_stream
        self._finalize_response(span, resp, (time.perf_counter() - self._start) * 1000)
        return resp

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._response is not None and exc_val is not None:
            self._response._neatlogs_active_exc = exc_val
        try:
            return self._cm.__exit__(exc_type, exc_val, exc_tb)
        finally:
            if self._wrapped_stream is not None:
                if not self._wrapped_stream._finalized:
                    if exc_val is not None:
                        self._wrapped_stream._finalize_error(exc_val)
                    else:
                        self._wrapped_stream._finalize(
                            interrupted=not self._wrapped_stream._exhausted
                        )
            elif self._response is not None:
                fc = getattr(self._response, "_neatlogs_force_close", None) or getattr(
                    self._response, "close", None
                )
                if callable(fc):
                    try:
                        fc(exc_val)
                    except TypeError:
                        try:
                            fc()
                        except BaseException:
                            pass
                    except BaseException:
                        pass

    async def __aenter__(self):
        if is_suppressed():
            return await self._cm.__aenter__()
        ctx_tok = _RAW_CALL_MARKER.set(self._marker)
        try:
            resp = await self._cm.__aenter__()
        except BaseException as err:
            _RAW_CALL_MARKER.reset(ctx_tok)
            if not self._marker["handled"]:
                span = self._start_span_fn(self._kwargs)
                span.set_status(StatusCode.ERROR, str(err))
                if isinstance(err, Exception):
                    span.record_exception(err)
                span.end()
            raise
        _RAW_CALL_MARKER.reset(ctx_tok)
        if self._marker["handled"]:
            self._response = resp
            return resp
        span = self._start_span_fn(self._kwargs)
        self._span = span
        is_stream = bool(self._kwargs.get("stream", False))
        if _is_raw_response(resp):
            self._response = await _instrument_async_raw_response(
                span,
                resp,
                self._start,
                is_stream,
                self._finalize_response,
                self._stream_finalizer,
                "stream",
            )
            return self._response
        if _is_stream_object(resp, is_stream):
            span.set_attribute("neatlogs.llm.is_streaming", True)
            finalizer = (
                self._stream_finalizer()
                if isinstance(self._stream_finalizer, type)
                else self._stream_finalizer
            )
            self._wrapped_stream = AsyncStreamWrapper(resp, span, finalizer)
            return self._wrapped_stream
        self._finalize_response(span, resp, (time.perf_counter() - self._start) * 1000)
        return resp

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._response is not None and exc_val is not None:
            self._response._neatlogs_active_exc = exc_val
        try:
            return await self._cm.__aexit__(exc_type, exc_val, exc_tb)
        finally:
            if self._wrapped_stream is not None:
                if not self._wrapped_stream._finalized:
                    if exc_val is not None:
                        self._wrapped_stream._finalize_error(exc_val)
                    else:
                        self._wrapped_stream._finalize(
                            interrupted=not self._wrapped_stream._exhausted
                        )
            elif self._response is not None:
                fc = getattr(self._response, "_neatlogs_force_close", None) or getattr(
                    self._response, "close", None
                )
                if callable(fc):
                    try:
                        try:
                            res = fc(exc_val)
                        except TypeError:
                            res = fc()
                        if inspect.isawaitable(res):
                            await res
                    except BaseException:
                        pass

    def __getattr__(self, name: str) -> Any:
        return getattr(self._cm, name)


def _patch_non_stainless_raw_wrappers(
    resource: Any,
    start_span_fn: Any,
    finalize_response: Any,
    stream_finalizer: Any,
    is_async: bool,
) -> None:
    cls = type(resource)
    for attr in ("with_raw_response", "with_streaming_response"):
        desc = getattr(cls, attr, None)
        if type(desc).__name__ == "cached_property":
            continue
        sub = getattr(resource, attr, None)
        if sub is None or not hasattr(sub, "create") or getattr(sub, "_neatlogs_patched", False):
            continue
        orig_sub_create = sub.create
        if getattr(orig_sub_create, "__wrapped__", None) == getattr(resource, "create", None):
            continue

        if attr == "with_raw_response":
            if is_async:

                async def patched_raw_create(*args, _orig=orig_sub_create, **kwargs):
                    if is_suppressed():
                        return await _orig(*args, **kwargs)
                    marker = {"handled": False}
                    ctx_tok = _RAW_CALL_MARKER.set(marker)
                    start = time.perf_counter()
                    try:
                        resp = await _orig(*args, **kwargs)
                    except BaseException as err:
                        _RAW_CALL_MARKER.reset(ctx_tok)
                        if not marker["handled"]:
                            span = start_span_fn(kwargs)
                            span.set_status(StatusCode.ERROR, str(err))
                            if isinstance(err, Exception):
                                span.record_exception(err)
                            span.end()
                        raise
                    _RAW_CALL_MARKER.reset(ctx_tok)
                    if marker["handled"]:
                        return resp
                    span = start_span_fn(kwargs)
                    is_stream = bool(kwargs.get("stream", False))
                    if _is_raw_response(resp):
                        return await _instrument_async_raw_response(
                            span,
                            resp,
                            start,
                            is_stream,
                            finalize_response,
                            stream_finalizer,
                            "raw",
                        )
                    if is_stream or _is_stream_object(resp, is_stream):
                        finalizer = (
                            stream_finalizer()
                            if isinstance(stream_finalizer, type)
                            else stream_finalizer
                        )
                        return AsyncStreamWrapper(resp, span, finalizer)
                    finalize_response(span, resp, (time.perf_counter() - start) * 1000)
                    return resp

            else:

                def patched_raw_create(*args, _orig=orig_sub_create, **kwargs):
                    if is_suppressed():
                        return _orig(*args, **kwargs)
                    marker = {"handled": False}
                    ctx_tok = _RAW_CALL_MARKER.set(marker)
                    start = time.perf_counter()
                    try:
                        resp = _orig(*args, **kwargs)
                    except BaseException as err:
                        _RAW_CALL_MARKER.reset(ctx_tok)
                        if not marker["handled"]:
                            span = start_span_fn(kwargs)
                            span.set_status(StatusCode.ERROR, str(err))
                            if isinstance(err, Exception):
                                span.record_exception(err)
                            span.end()
                        raise
                    _RAW_CALL_MARKER.reset(ctx_tok)
                    if marker["handled"]:
                        return resp
                    span = start_span_fn(kwargs)
                    is_stream = bool(kwargs.get("stream", False))
                    if _is_raw_response(resp):
                        return _instrument_raw_response(
                            span,
                            resp,
                            start,
                            is_stream,
                            finalize_response,
                            stream_finalizer,
                            "raw",
                        )
                    if is_stream or _is_stream_object(resp, is_stream):
                        finalizer = (
                            stream_finalizer()
                            if isinstance(stream_finalizer, type)
                            else stream_finalizer
                        )
                        return SyncStreamWrapper(resp, span, finalizer)
                    finalize_response(span, resp, (time.perf_counter() - start) * 1000)
                    return resp

            sub.create = patched_raw_create
            sub._neatlogs_patched = True
        else:

            def patched_stream_create(*args, _orig=orig_sub_create, **kwargs):
                if is_suppressed():
                    return _orig(*args, **kwargs)
                marker = {"handled": False}
                ctx_tok = _RAW_CALL_MARKER.set(marker)
                start = time.perf_counter()
                try:
                    cm = _orig(*args, **kwargs)
                finally:
                    _RAW_CALL_MARKER.reset(ctx_tok)
                return _StreamingContextManagerWrapper(
                    cm,
                    start_span_fn,
                    finalize_response,
                    stream_finalizer,
                    kwargs,
                    marker,
                    start,
                )

            sub.create = patched_stream_create
            sub._neatlogs_patched = True


class AnthropicInstrumentor:
    """Instrumentor class for InstrumentationManager integration."""

    def instrument(self, tracer_provider=None):
        _patch_anthropic_module()

    def uninstrument(self):
        _unpatch_anthropic_module()


def wrap_anthropic_client(client: Any) -> Any:
    """
    Wrap an Anthropic client instance. Patches messages (create/stream/parse/
    count_tokens), legacy completions, and beta.messages.
    """
    _patch_messages(client.messages)
    _extra_message_methods(client.messages, is_async=False)
    _patch_legacy_completions(getattr(client, "completions", None), is_async=False)
    beta = getattr(client, "beta", None)
    if beta is not None and getattr(beta, "messages", None) is not None:
        _patch_messages(beta.messages)
        _extra_message_methods(beta.messages, is_async=False)
    return client


def wrap_async_anthropic_client(client: Any) -> Any:
    """Wrap an AsyncAnthropic client instance — full coverage."""
    _patch_async_messages(client.messages)
    _extra_message_methods(client.messages, is_async=True)
    _patch_legacy_completions(getattr(client, "completions", None), is_async=True)
    beta = getattr(client, "beta", None)
    if beta is not None and getattr(beta, "messages", None) is not None:
        _patch_async_messages(beta.messages)
        _extra_message_methods(beta.messages, is_async=True)
    return client


def _start_create_span(kwargs: dict) -> Any:
    model = kwargs.get("model", "")
    input_messages = kwargs.get("messages", [])
    system = kwargs.get("system")
    is_stream = kwargs.get("stream", False)
    tracer = get_provider_tracer()
    span = tracer.start_span(
        name="anthropic.messages.create",
        attributes={
            "neatlogs.span.kind": "llm",
            "neatlogs.llm.provider": "anthropic",
            "neatlogs.llm.system": "anthropic",
            "neatlogs.llm.model_name": model,
            "neatlogs.llm.is_streaming": is_stream,
        },
    )
    _set_input_attributes(span, input_messages, system, kwargs)
    return span


def _patch_messages(messages: Any) -> None:
    if getattr(messages, "_neatlogs_patched", False):
        return

    orig_create = messages.create
    orig_stream = getattr(messages, "stream", None)

    def patched_create(*args, **kwargs):
        if is_suppressed():
            return orig_create(*args, **kwargs)

        marker = _RAW_CALL_MARKER.get()
        if marker is not None:
            marker["handled"] = True

        is_stream = kwargs.get("stream", False)
        raw_mode = _raw_response_mode(kwargs)
        span = _start_create_span(kwargs)
        start = time.perf_counter()

        try:
            response = orig_create(*args, **kwargs)
        except Exception as e:
            span.set_status(StatusCode.ERROR, str(e))
            span.record_exception(e)
            span.end()
            raise

        if _is_raw_response(response):
            return _instrument_raw_response(
                span,
                response,
                start,
                bool(is_stream),
                _finalize_response,
                _finalize_stream,
                raw_mode,
            )

        if is_stream:
            return SyncStreamWrapper(response, span, _finalize_stream)

        duration_ms = (time.perf_counter() - start) * 1000
        _finalize_response(span, response, duration_ms)
        return response

    messages.create = patched_create

    if orig_stream:

        def patched_stream(*args, **kwargs):
            if is_suppressed():
                return orig_stream(*args, **kwargs)

            model = kwargs.get("model", "")
            input_messages = kwargs.get("messages", [])
            system = kwargs.get("system")

            tracer = get_provider_tracer()
            span = tracer.start_span(
                name="anthropic.messages.create",
                attributes={
                    "neatlogs.span.kind": "llm",
                    "neatlogs.llm.provider": "anthropic",
                    "neatlogs.llm.system": "anthropic",
                    "neatlogs.llm.model_name": model,
                    "neatlogs.llm.is_streaming": True,
                },
            )

            _set_input_attributes(span, input_messages, system, kwargs)

            stream_mgr = orig_stream(*args, **kwargs)
            return _SyncStreamManagerWrapper(stream_mgr, span)

        messages.stream = patched_stream

    _reset_raw_wrappers(messages)
    _patch_non_stainless_raw_wrappers(
        messages, _start_create_span, _finalize_response, _finalize_stream, is_async=False
    )
    messages._neatlogs_patched = True


def _patch_async_messages(messages: Any) -> None:
    if getattr(messages, "_neatlogs_patched", False):
        return

    orig_create = messages.create
    orig_stream = getattr(messages, "stream", None)

    async def patched_create(*args, **kwargs):
        if is_suppressed():
            return await orig_create(*args, **kwargs)

        marker = _RAW_CALL_MARKER.get()
        if marker is not None:
            marker["handled"] = True

        is_stream = kwargs.get("stream", False)
        raw_mode = _raw_response_mode(kwargs)
        span = _start_create_span(kwargs)
        start = time.perf_counter()

        try:
            response = await orig_create(*args, **kwargs)
        except Exception as e:
            span.set_status(StatusCode.ERROR, str(e))
            span.record_exception(e)
            span.end()
            raise

        if _is_raw_response(response):
            return await _instrument_async_raw_response(
                span,
                response,
                start,
                bool(is_stream),
                _finalize_response,
                _finalize_stream,
                raw_mode,
            )

        if is_stream:
            return AsyncStreamWrapper(response, span, _finalize_stream)

        duration_ms = (time.perf_counter() - start) * 1000
        _finalize_response(span, response, duration_ms)
        return response

    messages.create = patched_create

    if orig_stream:

        def patched_stream(*args, **kwargs):
            if is_suppressed():
                return orig_stream(*args, **kwargs)

            model = kwargs.get("model", "")
            input_messages = kwargs.get("messages", [])
            system = kwargs.get("system")

            tracer = get_provider_tracer()
            span = tracer.start_span(
                name="anthropic.messages.create",
                attributes={
                    "neatlogs.span.kind": "llm",
                    "neatlogs.llm.provider": "anthropic",
                    "neatlogs.llm.system": "anthropic",
                    "neatlogs.llm.model_name": model,
                    "neatlogs.llm.is_streaming": True,
                },
            )

            _set_input_attributes(span, input_messages, system, kwargs)

            stream_mgr = orig_stream(*args, **kwargs)
            return _AsyncStreamManagerWrapper(stream_mgr, span)

        messages.stream = patched_stream

    _reset_raw_wrappers(messages)
    _patch_non_stainless_raw_wrappers(
        messages, _start_create_span, _finalize_response, _finalize_stream, is_async=True
    )
    messages._neatlogs_patched = True


def _set_input_attributes(span: Any, messages: list, system: Any, kwargs: dict) -> None:
    """Set input message and tool attributes on the span."""
    idx = 0

    if system:
        span.set_attribute(f"neatlogs.llm.input_messages.{idx}.role", "system")
        if isinstance(system, str):
            span.set_attribute(f"neatlogs.llm.input_messages.{idx}.content", system)
        else:
            span.set_attribute(f"neatlogs.llm.input_messages.{idx}.content", serialize(system))
            set_media_attributes(span, f"neatlogs.llm.input_messages.{idx}", system, "input")
        idx += 1

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        span.set_attribute(f"neatlogs.llm.input_messages.{idx}.role", role)
        if isinstance(content, str):
            span.set_attribute(f"neatlogs.llm.input_messages.{idx}.content", content)
        else:
            span.set_attribute(f"neatlogs.llm.input_messages.{idx}.content", serialize(content))
        set_media_attributes(span, f"neatlogs.llm.input_messages.{idx}", content, "input")
        if msg.get("tool_call_id"):
            span.set_attribute(
                f"neatlogs.llm.input_messages.{idx}.tool_call_id", msg["tool_call_id"]
            )
        idx += 1

    tools = kwargs.get("tools")
    if tools:
        for i, tool in enumerate(tools):
            span.set_attribute(f"neatlogs.llm.tools.{i}.name", tool.get("name", ""))
            if tool.get("description"):
                span.set_attribute(f"neatlogs.llm.tools.{i}.description", tool["description"])
            if tool.get("input_schema"):
                span.set_attribute(
                    f"neatlogs.llm.tools.{i}.input_schema", serialize(tool["input_schema"])
                )

    for param in ("temperature", "top_p", "top_k", "max_tokens"):
        if param in kwargs and kwargs[param] is not None:
            span.set_attribute(f"neatlogs.llm.{param}", kwargs[param])


def _finalize_response(span: Any, response: Any, duration_ms: float) -> None:
    """Extract attributes from a non-streaming Anthropic Message response."""
    _set_message_attributes(span, response)
    span.set_attribute("neatlogs.llm.metrics.duration_ms", round(duration_ms, 3))
    span.set_status(StatusCode.OK)
    span.end()


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _set_message_attributes(span: Any, response: Any) -> None:
    """Record the output, tool calls, stop reason, model and usage of a Message."""
    content = _field(response, "content", []) or []
    text_parts: List[str] = []
    tool_call_idx = 0

    for block in content:
        block_type = _field(block, "type", "")
        if block_type == "text":
            text_parts.append(_field(block, "text", "") or "")
        elif block_type == "tool_use":
            span.set_attribute(
                f"neatlogs.llm.tool_calls.{tool_call_idx}.id", _field(block, "id", "") or ""
            )
            span.set_attribute(
                f"neatlogs.llm.tool_calls.{tool_call_idx}.name", _field(block, "name", "") or ""
            )
            span.set_attribute(
                f"neatlogs.llm.tool_calls.{tool_call_idx}.arguments",
                serialize(_field(block, "input", {})),
            )
            tool_call_idx += 1
        elif block_type == "thinking":
            thinking_text = _field(block, "thinking", "")
            if thinking_text:
                span.set_attribute("neatlogs.llm.output_messages.0.thinking", thinking_text)

    if text_parts:
        span.set_attribute("neatlogs.llm.output_messages.0.role", "assistant")
        span.set_attribute("neatlogs.llm.output_messages.0.content", "".join(text_parts))
    set_media_attributes(span, "neatlogs.llm.output_messages.0", content, "output")

    stop_reason = _field(response, "stop_reason", None)
    if stop_reason:
        span.set_attribute("neatlogs.llm.finish_reason", stop_reason)

    model = _field(response, "model", None)
    if model:
        span.set_attribute("neatlogs.llm.model_name", model)

    usage = _field(response, "usage", None)
    if usage:
        _set_usage_attributes(span, usage)


def _set_usage_attributes(span: Any, usage: Any) -> None:
    """Set token usage attributes from Anthropic usage object."""
    input_tokens = _field(usage, "input_tokens", None)
    if input_tokens is not None:
        span.set_attribute("neatlogs.llm.token_count.prompt", input_tokens)

    output_tokens = _field(usage, "output_tokens", None)
    if output_tokens is not None:
        span.set_attribute("neatlogs.llm.token_count.completion", output_tokens)

    if input_tokens is not None and output_tokens is not None:
        span.set_attribute("neatlogs.llm.token_count.total", input_tokens + output_tokens)

    cache_read = _field(usage, "cache_read_input_tokens", None)
    if cache_read is not None:
        span.set_attribute("neatlogs.llm.token_count.cache_read", cache_read)

    cache_write = _field(usage, "cache_creation_input_tokens", None)
    if cache_write is not None:
        span.set_attribute("neatlogs.llm.token_count.cache_write", cache_write)


def _finalize_stream(
    span: Any,
    chunks: List[Any],
    duration_ms: float,
    ttft_ms: Optional[float],
    *,
    interrupted: bool = False,
) -> None:
    """Finalize a streaming response span from accumulated Anthropic stream events."""
    _set_stream_attributes(span, chunks)
    _set_stream_timing(span, duration_ms, ttft_ms)
    span.set_status(StatusCode.UNSET if interrupted else StatusCode.OK)
    span.end()


def _set_stream_attributes(span: Any, chunks: List[Any]) -> Optional[str]:
    """Record what the stream events carried. Returns the stop reason, if one arrived."""
    text_parts: List[str] = []
    tool_calls_acc: dict = {}
    thinking_parts: List[str] = []
    stop_reason = None
    model = None
    input_tokens = None
    output_tokens = None
    cache_read = None
    cache_write = None

    for event in chunks:
        event_type = _field(event, "type", "")

        if event_type == "message_start":
            message = _field(event, "message", None)
            if message:
                model = _field(message, "model", None)
                msg_usage = _field(message, "usage", None)
                if msg_usage:
                    input_tokens = _field(msg_usage, "input_tokens", None)
                    cache_read = _field(msg_usage, "cache_read_input_tokens", None)
                    cache_write = _field(msg_usage, "cache_creation_input_tokens", None)

        elif event_type == "content_block_start":
            block = _field(event, "content_block", None)
            if block and _field(block, "type", "") == "tool_use":
                idx = _field(event, "index", len(tool_calls_acc))
                tool_calls_acc[idx] = {
                    "id": _field(block, "id", "") or "",
                    "name": _field(block, "name", "") or "",
                    "arguments": "",
                }

        elif event_type == "content_block_delta":
            delta = _field(event, "delta", None)
            if not delta:
                continue
            delta_type = _field(delta, "type", "")
            if delta_type == "text_delta":
                text_parts.append(_field(delta, "text", "") or "")
            elif delta_type == "input_json_delta":
                idx = _field(event, "index", 0)
                if idx in tool_calls_acc:
                    tool_calls_acc[idx]["arguments"] += _field(delta, "partial_json", "") or ""
            elif delta_type == "thinking_delta":
                thinking_parts.append(_field(delta, "thinking", "") or "")

        elif event_type == "message_delta":
            delta = _field(event, "delta", None)
            if delta:
                stop_reason = _field(delta, "stop_reason", None)
            msg_usage = _field(event, "usage", None)
            if msg_usage:
                output_tokens = _field(msg_usage, "output_tokens", None)

    # Output messages
    full_text = "".join(text_parts)
    if full_text:
        span.set_attribute("neatlogs.llm.output_messages.0.role", "assistant")
        span.set_attribute("neatlogs.llm.output_messages.0.content", full_text)

    # Thinking
    full_thinking = "".join(thinking_parts)
    if full_thinking:
        span.set_attribute("neatlogs.llm.output_messages.0.thinking", full_thinking)

    # Tool calls
    for j, tc in enumerate(tool_calls_acc.values()):
        span.set_attribute(f"neatlogs.llm.tool_calls.{j}.id", tc["id"])
        span.set_attribute(f"neatlogs.llm.tool_calls.{j}.name", tc["name"])
        span.set_attribute(f"neatlogs.llm.tool_calls.{j}.arguments", tc["arguments"])

    if model:
        span.set_attribute("neatlogs.llm.model_name", model)
    if stop_reason:
        span.set_attribute("neatlogs.llm.finish_reason", stop_reason)

    # Usage
    if input_tokens is not None:
        span.set_attribute("neatlogs.llm.token_count.prompt", input_tokens)
    if output_tokens is not None:
        span.set_attribute("neatlogs.llm.token_count.completion", output_tokens)
    if input_tokens is not None and output_tokens is not None:
        span.set_attribute("neatlogs.llm.token_count.total", input_tokens + output_tokens)
    if cache_read is not None:
        span.set_attribute("neatlogs.llm.token_count.cache_read", cache_read)
    if cache_write is not None:
        span.set_attribute("neatlogs.llm.token_count.cache_write", cache_write)
    return stop_reason


def _set_stream_timing(span: Any, duration_ms: float, ttft_ms: Optional[float]) -> None:
    span.set_attribute("neatlogs.llm.metrics.duration_ms", round(duration_ms, 3))
    if ttft_ms is not None:
        span.set_attribute("neatlogs.llm.metrics.ttft_ms", round(ttft_ms, 3))
        if duration_ms > ttft_ms:
            span.set_attribute(
                "neatlogs.llm.metrics.streaming_time_to_generate_ms",
                round(duration_ms - ttft_ms, 3),
            )


def _finalize_helper_stream(
    span: Any,
    stream: Any,
    chunks: List[Any],
    duration_ms: float,
    ttft_ms: Optional[float],
    error: Optional[BaseException],
) -> None:
    """End the span of a ``messages.stream()`` block.

    Anthropic's helper keeps the message it has assembled so far. That snapshot
    is the same whichever way the caller read the stream (``text_stream``,
    ``get_final_message``, ``until_done`` or plain iteration), so the span no
    longer depends on the caller iterating through this wrapper.
    """
    snapshot = None
    if stream is not None:
        try:
            snapshot = stream.current_message_snapshot
        except Exception:
            snapshot = None
    if getattr(snapshot, "type", None) != "message":
        snapshot = None
    if snapshot is not None:
        _set_message_attributes(span, snapshot)
        stop_reason = getattr(snapshot, "stop_reason", None)
    else:
        stop_reason = _set_stream_attributes(span, chunks)
    _set_stream_timing(span, duration_ms, ttft_ms)

    cancelled = error is not None and error.__class__.__name__ in {
        "CancelledError",
        "GeneratorExit",
    }
    # A reply that started and never reached its stop reason was left early.
    left_early = (snapshot is not None or bool(chunks)) and not stop_reason
    if error is not None and not cancelled:
        span.set_status(StatusCode.ERROR, str(error))
        if isinstance(error, Exception):
            span.record_exception(error)
    elif cancelled or left_early:
        span.set_attribute("neatlogs.stream.cancelled", True)
        span.set_status(StatusCode.UNSET)
    else:
        span.set_status(StatusCode.OK)
    span.end()


class _SyncStreamManagerWrapper:
    """Wraps Anthropic's MessageStreamManager context manager for messages.stream()."""

    def __init__(self, stream_mgr: Any, span: Any):
        self._stream_mgr = stream_mgr
        self._span = span
        self._start_time = time.perf_counter()
        self._first_chunk_time: Optional[float] = None
        self._chunks: List[Any] = []
        self._finalized = False
        self._stream: Any = None

    def __enter__(self):
        try:
            self._stream = self._stream_mgr.__enter__()
        except BaseException as error:
            # The request itself failed (for example HTTP 529): no block will
            # run, so nothing else would ever end this span.
            self._finalize(error)
            raise
        return _SyncStreamIterator(self)

    def __exit__(self, exc_type, exc, tb):
        error = exc
        try:
            self._stream_mgr.__exit__(exc_type, exc, tb)
        except BaseException as close_error:
            error = close_error
            raise
        finally:
            self._finalize(error)

    def _finalize(self, error: Optional[BaseException] = None):
        if self._finalized:
            return
        self._finalized = True
        elapsed_ms = (time.perf_counter() - self._start_time) * 1000
        ttft_ms = None
        if self._first_chunk_time is not None:
            ttft_ms = (self._first_chunk_time - self._start_time) * 1000
        _finalize_helper_stream(self._span, self._stream, self._chunks, elapsed_ms, ttft_ms, error)

    def __getattr__(self, name):
        return getattr(self._stream_mgr, name)


class _SyncStreamIterator:
    """Iterates events from an Anthropic MessageStream, collecting chunks."""

    def __init__(self, wrapper: _SyncStreamManagerWrapper):
        self._wrapper = wrapper

    def __iter__(self):
        for event in self._wrapper._stream:
            if self._wrapper._first_chunk_time is None:
                self._wrapper._first_chunk_time = time.perf_counter()
            self._wrapper._chunks.append(event)
            yield event

    def __getattr__(self, name):
        return getattr(self._wrapper._stream, name)


class _AsyncStreamManagerWrapper:
    """Wraps Anthropic's AsyncMessageStreamManager for async messages.stream()."""

    def __init__(self, stream_mgr: Any, span: Any):
        self._stream_mgr = stream_mgr
        self._span = span
        self._start_time = time.perf_counter()
        self._first_chunk_time: Optional[float] = None
        self._chunks: List[Any] = []
        self._finalized = False
        self._stream: Any = None

    async def __aenter__(self):
        try:
            self._stream = await self._stream_mgr.__aenter__()
        except BaseException as error:
            self._finalize(error)
            raise
        return _AsyncStreamIterator(self)

    async def __aexit__(self, exc_type, exc, tb):
        error = exc
        try:
            await self._stream_mgr.__aexit__(exc_type, exc, tb)
        except BaseException as close_error:
            error = close_error
            raise
        finally:
            self._finalize(error)

    def _finalize(self, error: Optional[BaseException] = None):
        if self._finalized:
            return
        self._finalized = True
        elapsed_ms = (time.perf_counter() - self._start_time) * 1000
        ttft_ms = None
        if self._first_chunk_time is not None:
            ttft_ms = (self._first_chunk_time - self._start_time) * 1000
        _finalize_helper_stream(self._span, self._stream, self._chunks, elapsed_ms, ttft_ms, error)

    def __getattr__(self, name):
        return getattr(self._stream_mgr, name)


class _AsyncStreamIterator:
    """Iterates events from an Anthropic AsyncMessageStream, collecting chunks."""

    def __init__(self, wrapper: _AsyncStreamManagerWrapper):
        self._wrapper = wrapper

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            event = await self._wrapper._stream.__anext__()
        except StopAsyncIteration:
            raise

        if self._wrapper._first_chunk_time is None:
            self._wrapper._first_chunk_time = time.perf_counter()
        self._wrapper._chunks.append(event)
        return event

    def __getattr__(self, name):
        return getattr(self._wrapper._stream, name)


# ---------------------------------------------------------------------------
# Additional message methods: parse (structured) + count_tokens; legacy completions
# ---------------------------------------------------------------------------


def _extra_message_methods(messages: Any, is_async: bool) -> None:
    """Patch messages.parse (structured output) and messages.count_tokens."""
    # parse → same shape as a Message response
    if hasattr(messages, "parse") and not getattr(messages, "_neatlogs_parse_patched", False):
        orig_parse = messages.parse
        if is_async:

            async def patched_parse(*args, **kwargs):
                if is_suppressed():
                    return await orig_parse(*args, **kwargs)
                span = _start_message_span(kwargs, "anthropic.messages.parse", structured=True)
                start = time.perf_counter()
                try:
                    resp = await orig_parse(*args, **kwargs)
                except Exception as e:
                    _err(span, e)
                    raise
                _finalize_response(span, resp, (time.perf_counter() - start) * 1000)
                return resp

        else:

            def patched_parse(*args, **kwargs):
                if is_suppressed():
                    return orig_parse(*args, **kwargs)
                span = _start_message_span(kwargs, "anthropic.messages.parse", structured=True)
                start = time.perf_counter()
                try:
                    resp = orig_parse(*args, **kwargs)
                except Exception as e:
                    _err(span, e)
                    raise
                _finalize_response(span, resp, (time.perf_counter() - start) * 1000)
                return resp

        messages.parse = patched_parse
        messages._neatlogs_parse_patched = True

    # count_tokens → small utility call
    if hasattr(messages, "count_tokens") and not getattr(
        messages, "_neatlogs_count_patched", False
    ):
        orig_count = messages.count_tokens
        if is_async:

            async def patched_count(*args, **kwargs):
                if is_suppressed():
                    return await orig_count(*args, **kwargs)
                span = get_provider_tracer().start_span(
                    name="anthropic.messages.count_tokens",
                    attributes={
                        "neatlogs.span.kind": "llm",
                        "neatlogs.llm.provider": "anthropic",
                        "neatlogs.llm.task": "count_tokens",
                        "neatlogs.llm.model_name": kwargs.get("model", ""),
                    },
                )
                try:
                    resp = await orig_count(*args, **kwargs)
                except Exception as e:
                    _err(span, e)
                    raise
                _finalize_count_tokens(span, resp)
                return resp

        else:

            def patched_count(*args, **kwargs):
                if is_suppressed():
                    return orig_count(*args, **kwargs)
                span = get_provider_tracer().start_span(
                    name="anthropic.messages.count_tokens",
                    attributes={
                        "neatlogs.span.kind": "llm",
                        "neatlogs.llm.provider": "anthropic",
                        "neatlogs.llm.task": "count_tokens",
                        "neatlogs.llm.model_name": kwargs.get("model", ""),
                    },
                )
                try:
                    resp = orig_count(*args, **kwargs)
                except Exception as e:
                    _err(span, e)
                    raise
                _finalize_count_tokens(span, resp)
                return resp

        messages.count_tokens = patched_count
        messages._neatlogs_count_patched = True


def _start_message_span(kwargs: dict, name: str, structured: bool = False) -> Any:
    span = get_provider_tracer().start_span(
        name=name,
        attributes={
            "neatlogs.span.kind": "llm",
            "neatlogs.llm.provider": "anthropic",
            "neatlogs.llm.system": "anthropic",
            "neatlogs.llm.model_name": kwargs.get("model", ""),
        },
    )
    if structured:
        span.set_attribute("neatlogs.llm.structured_output", True)
    _set_input_attributes(span, kwargs.get("messages", []), kwargs.get("system"), kwargs)
    return span


def _finalize_count_tokens(span: Any, resp: Any) -> None:
    tokens = getattr(resp, "input_tokens", None)
    if tokens is not None:
        span.set_attribute("neatlogs.llm.token_count.prompt", tokens)
    span.set_status(StatusCode.OK)
    span.end()


def _patch_legacy_completions(completions: Any, is_async: bool) -> None:
    """Legacy text completions API (client.completions.create)."""
    if completions is None or not hasattr(completions, "create"):
        return
    if getattr(completions, "_neatlogs_patched", False):
        return
    orig = completions.create

    def _attrs(kwargs):
        prompt = kwargs.get("prompt", "")
        return {
            "neatlogs.span.kind": "llm",
            "neatlogs.llm.provider": "anthropic",
            "neatlogs.llm.system": "anthropic",
            "neatlogs.llm.model_name": kwargs.get("model", ""),
            "neatlogs.llm.input_messages.0.role": "user",
            "neatlogs.llm.input_messages.0.content": str(prompt)[:10000],
        }

    def _finalize(span, resp, duration_ms):
        completion = getattr(resp, "completion", None)
        if completion:
            span.set_attribute("neatlogs.llm.output_messages.0.role", "assistant")
            span.set_attribute("neatlogs.llm.output_messages.0.content", str(completion)[:10000])
        stop = getattr(resp, "stop_reason", None)
        if stop:
            span.set_attribute("neatlogs.llm.finish_reason", str(stop))
        span.set_attribute("neatlogs.llm.metrics.duration_ms", round(duration_ms, 3))
        span.set_status(StatusCode.OK)
        span.end()

    if is_async:

        async def patched(*args, **kwargs):
            if is_suppressed():
                return await orig(*args, **kwargs)
            span = get_provider_tracer().start_span(
                name="anthropic.completions.create", attributes=_attrs(kwargs)
            )
            start = time.perf_counter()
            try:
                resp = await orig(*args, **kwargs)
            except Exception as e:
                _err(span, e)
                raise
            _finalize(span, resp, (time.perf_counter() - start) * 1000)
            return resp

    else:

        def patched(*args, **kwargs):
            if is_suppressed():
                return orig(*args, **kwargs)
            span = get_provider_tracer().start_span(
                name="anthropic.completions.create", attributes=_attrs(kwargs)
            )
            start = time.perf_counter()
            try:
                resp = orig(*args, **kwargs)
            except Exception as e:
                _err(span, e)
                raise
            _finalize(span, resp, (time.perf_counter() - start) * 1000)
            return resp

    completions.create = patched
    completions._neatlogs_patched = True


def _err(span: Any, e: Exception) -> None:
    span.set_status(StatusCode.ERROR, str(e))
    span.record_exception(e)
    span.end()


# ---------------------------------------------------------------------------
# Import-replacement: `from neatlogs.anthropic import anthropic`
# Patches Anthropic/AsyncAnthropic.__init__ so every client is auto-wrapped.
# ---------------------------------------------------------------------------


def _patch_anthropic_module() -> None:
    global _PATCHED, _ORIG_INIT, _ORIG_ASYNC_INIT
    if _PATCHED:
        return
    _PATCHED = True

    import anthropic as _anthropic

    _ORIG_INIT = _anthropic.Anthropic.__init__
    _ORIG_ASYNC_INIT = _anthropic.AsyncAnthropic.__init__

    def _patched_init(self, *args, **kwargs):
        _ORIG_INIT(self, *args, **kwargs)
        wrap_anthropic_client(self)

    _anthropic.Anthropic.__init__ = _patched_init

    def _patched_async_init(self, *args, **kwargs):
        _ORIG_ASYNC_INIT(self, *args, **kwargs)
        wrap_async_anthropic_client(self)

    _anthropic.AsyncAnthropic.__init__ = _patched_async_init


def _unpatch_anthropic_module() -> None:
    global _PATCHED, _ORIG_INIT, _ORIG_ASYNC_INIT
    if not _PATCHED:
        return

    import anthropic as _anthropic

    if _ORIG_INIT is not None:
        _anthropic.Anthropic.__init__ = _ORIG_INIT
    if _ORIG_ASYNC_INIT is not None:
        _anthropic.AsyncAnthropic.__init__ = _ORIG_ASYNC_INIT

    _PATCHED = False
    _ORIG_INIT = None
    _ORIG_ASYNC_INIT = None


_patch_anthropic_module()

import anthropic  # noqa: E402
