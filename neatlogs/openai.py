"""
Neatlogs OpenAI wrapper.

Two usage patterns:

  1. Explicit wrap (primary):
     >>> import neatlogs, openai
     >>> client = neatlogs.wrap(openai.OpenAI())
     >>> client.chat.completions.create(...)

  2. Import replacement (Langfuse-style):
     >>> from neatlogs.openai import openai
     >>> client = openai.OpenAI()
     >>> client.chat.completions.create(...)
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
from .core.choice_accumulator import ChoiceAccumulator, OpenAIStreamFinalizer
from .core.media import set_media_attributes

_PATCHED = False
_ORIG_INIT = None
_ORIG_ASYNC_INIT = None
_RAW_CALL_MARKER: ContextVar[Optional[dict]] = ContextVar(
    "_neatlogs_openai_raw_marker", default=None
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

        if _is_stream_object(parsed, is_stream) and stream_finalizer is not None:
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
        elif (
            is_stream or getattr(response, "_is_sse_stream", False)
        ) and stream_finalizer is not None:
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
        if _is_stream_object(parsed, is_stream) and stream_finalizer is not None:
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
        elif (
            is_stream or getattr(response, "_is_sse_stream", False)
        ) and stream_finalizer is not None:
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
        if _is_stream_object(resp, is_stream) and self._stream_finalizer is not None:
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
        if _is_stream_object(resp, is_stream) and self._stream_finalizer is not None:
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
                    if (
                        is_stream or _is_stream_object(resp, is_stream)
                    ) and stream_finalizer is not None:
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
                    if (
                        is_stream or _is_stream_object(resp, is_stream)
                    ) and stream_finalizer is not None:
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


class OpenAIInstrumentor:
    """Instrumentor class for InstrumentationManager integration."""

    def instrument(self, tracer_provider=None):
        _patch_openai_module()

    def uninstrument(self):
        _unpatch_openai_module()


def wrap_openai_client(client: Any) -> Any:
    """
    Wrap an OpenAI client instance. Returns the same client with all
    LLM-relevant resources patched to auto-trace.
    """
    _safe(_patch_completions, _resource(client, "chat", "completions"))
    _safe(_patch_chat_parse, _resource(client, "chat", "completions"), sync=True)
    _safe(_patch_legacy_completions, getattr(client, "completions", None), sync=True)
    _safe(_patch_responses, getattr(client, "responses", None))
    _safe(_patch_responses_parse, getattr(client, "responses", None), sync=True)
    _safe(_patch_embeddings, getattr(client, "embeddings", None), sync=True)
    _safe(_patch_images, getattr(client, "images", None), sync=True)
    _safe(_patch_audio, getattr(client, "audio", None), sync=True)
    _safe(_patch_moderations, getattr(client, "moderations", None), sync=True)
    _safe(_patch_batches, getattr(client, "batches", None), sync=True)
    # beta.chat.completions.parse
    beta = getattr(client, "beta", None)
    if beta is not None:
        _safe(_patch_chat_parse, _resource(beta, "chat", "completions"), sync=True)
    return client


def wrap_async_openai_client(client: Any) -> Any:
    """Wrap an AsyncOpenAI client instance — full resource coverage."""
    _safe(_patch_async_completions, _resource(client, "chat", "completions"))
    _safe(_patch_chat_parse, _resource(client, "chat", "completions"), sync=False)
    _safe(_patch_legacy_completions, getattr(client, "completions", None), sync=False)
    _safe(_patch_async_responses, getattr(client, "responses", None))
    _safe(_patch_responses_parse, getattr(client, "responses", None), sync=False)
    _safe(_patch_embeddings, getattr(client, "embeddings", None), sync=False)
    _safe(_patch_images, getattr(client, "images", None), sync=False)
    _safe(_patch_audio, getattr(client, "audio", None), sync=False)
    _safe(_patch_moderations, getattr(client, "moderations", None), sync=False)
    _safe(_patch_batches, getattr(client, "batches", None), sync=False)
    beta = getattr(client, "beta", None)
    if beta is not None:
        _safe(_patch_chat_parse, _resource(beta, "chat", "completions"), sync=False)
    return client


def _resource(client: Any, *path):
    """Safely walk a nested resource path (e.g. client.chat.completions)."""
    obj = client
    for p in path:
        obj = getattr(obj, p, None)
        if obj is None:
            return None
    return obj


def _safe(fn, resource, **kw):
    """Call a patch fn only if the resource exists; never raise."""
    if resource is None:
        return
    try:
        fn(resource, **kw) if kw else fn(resource)
    except Exception:
        pass


def _start_completions_span(kwargs: dict) -> Any:
    model = kwargs.get("model", "")
    messages = kwargs.get("messages", [])
    is_stream = kwargs.get("stream", False)

    if is_stream:
        opts = kwargs.get("stream_options") or {}
        if not opts.get("include_usage"):
            opts["include_usage"] = True
            kwargs["stream_options"] = opts

    tracer = get_provider_tracer()
    span = tracer.start_span(
        name="openai.chat.completions.create",
        attributes={
            "neatlogs.span.kind": "llm",
            "neatlogs.llm.provider": "openai",
            "neatlogs.llm.system": "openai",
            "neatlogs.llm.model_name": model,
            "neatlogs.llm.is_streaming": is_stream,
        },
    )

    for i, msg in enumerate(messages):
        role = msg.get("role", "")
        content = msg.get("content", "")
        span.set_attribute(f"neatlogs.llm.input_messages.{i}.role", role)
        if isinstance(content, str):
            span.set_attribute(f"neatlogs.llm.input_messages.{i}.content", content)
        else:
            span.set_attribute(f"neatlogs.llm.input_messages.{i}.content", serialize(content))
        set_media_attributes(span, f"neatlogs.llm.input_messages.{i}", content, "input")
        if msg.get("tool_call_id"):
            span.set_attribute(f"neatlogs.llm.input_messages.{i}.tool_call_id", msg["tool_call_id"])

    tools = kwargs.get("tools")
    if tools:
        for i, tool in enumerate(tools):
            fn = tool.get("function", {})
            span.set_attribute(f"neatlogs.llm.tools.{i}.type", tool.get("type", "function"))
            span.set_attribute(f"neatlogs.llm.tools.{i}.definition", serialize(tool))
            span.set_attribute(f"neatlogs.llm.tools.{i}.name", fn.get("name", ""))
            if fn.get("description"):
                span.set_attribute(f"neatlogs.llm.tools.{i}.description", fn["description"])
            if fn.get("parameters"):
                span.set_attribute(
                    f"neatlogs.llm.tools.{i}.input_schema", serialize(fn["parameters"])
                )

    for param in (
        "temperature",
        "top_p",
        "max_tokens",
        "frequency_penalty",
        "presence_penalty",
    ):
        if param in kwargs and kwargs[param] is not None:
            span.set_attribute(f"neatlogs.llm.{param}", kwargs[param])

    _set_request_metadata(span, kwargs)
    return span


def _patch_completions(completions: Any) -> None:
    if getattr(completions, "_neatlogs_patched", False):
        return

    orig_create = completions.create

    def patched_create(*args, **kwargs):
        if is_suppressed():
            return orig_create(*args, **kwargs)

        marker = _RAW_CALL_MARKER.get()
        if marker is not None:
            marker["handled"] = True

        is_stream = kwargs.get("stream", False)
        raw_mode = _raw_response_mode(kwargs)
        span = _start_completions_span(kwargs)
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
                OpenAIStreamFinalizer,
                raw_mode,
            )

        if is_stream:
            return SyncStreamWrapper(response, span, OpenAIStreamFinalizer())

        duration_ms = (time.perf_counter() - start) * 1000
        _finalize_response(span, response, duration_ms)
        return response

    completions.create = patched_create
    _reset_raw_wrappers(completions)
    _patch_non_stainless_raw_wrappers(
        completions,
        _start_completions_span,
        _finalize_response,
        OpenAIStreamFinalizer,
        is_async=False,
    )
    completions._neatlogs_patched = True


def _patch_async_completions(completions: Any) -> None:
    if getattr(completions, "_neatlogs_patched", False):
        return

    orig_create = completions.create

    async def patched_create(*args, **kwargs):
        if is_suppressed():
            return await orig_create(*args, **kwargs)

        marker = _RAW_CALL_MARKER.get()
        if marker is not None:
            marker["handled"] = True

        is_stream = kwargs.get("stream", False)
        raw_mode = _raw_response_mode(kwargs)
        span = _start_completions_span(kwargs)
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
                OpenAIStreamFinalizer,
                raw_mode,
            )

        if is_stream:
            return AsyncStreamWrapper(response, span, OpenAIStreamFinalizer())

        duration_ms = (time.perf_counter() - start) * 1000
        _finalize_response(span, response, duration_ms)
        return response

    completions.create = patched_create
    _reset_raw_wrappers(completions)
    _patch_non_stainless_raw_wrappers(
        completions,
        _start_completions_span,
        _finalize_response,
        OpenAIStreamFinalizer,
        is_async=True,
    )
    completions._neatlogs_patched = True


def _start_responses_span(kwargs: dict) -> Any:
    model = kwargs.get("model", "")
    is_stream = kwargs.get("stream", False)
    tracer = get_provider_tracer()
    return tracer.start_span(
        name="openai.responses.create",
        attributes={
            "neatlogs.span.kind": "llm",
            "neatlogs.llm.provider": "openai",
            "neatlogs.llm.system": "openai",
            "neatlogs.llm.model_name": model,
            "neatlogs.llm.is_streaming": bool(is_stream),
            "neatlogs.llm.input_messages.0.role": "user",
            "neatlogs.llm.input_messages.0.content": serialize(kwargs.get("input", "")),
        },
    )


def _patch_responses(responses: Any) -> None:
    if getattr(responses, "_neatlogs_patched", False):
        return

    orig_create = responses.create

    def patched_create(*args, **kwargs):
        if is_suppressed():
            return orig_create(*args, **kwargs)

        marker = _RAW_CALL_MARKER.get()
        if marker is not None:
            marker["handled"] = True

        is_stream = kwargs.get("stream", False)
        raw_mode = _raw_response_mode(kwargs)
        span = _start_responses_span(kwargs)

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
                _finalize_responses_response,
                _finalize_responses_stream,
                raw_mode,
            )

        if is_stream:
            return SyncStreamWrapper(response, span, _finalize_responses_stream)

        duration_ms = (time.perf_counter() - start) * 1000
        _finalize_responses_response(span, response, duration_ms)
        return response

    responses.create = patched_create
    _reset_raw_wrappers(responses)
    _patch_non_stainless_raw_wrappers(
        responses,
        _start_responses_span,
        _finalize_responses_response,
        _finalize_responses_stream,
        is_async=False,
    )
    responses._neatlogs_patched = True


def _set_request_metadata(span: Any, kwargs: dict) -> None:
    """
    Capture caller-supplied request metadata. OpenAI accepts a top-level
    ``metadata=`` dict; many gateways/apps pass it via ``extra_body``. Surface it
    as ``neatlogs.metadata`` (the canonical target) so user tags reach the UI.
    """
    meta = kwargs.get("metadata")
    if meta is None:
        extra_body = kwargs.get("extra_body")
        if isinstance(extra_body, dict):
            meta = extra_body.get("metadata")
    if meta:
        span.set_attribute("neatlogs.metadata", meta if isinstance(meta, str) else serialize(meta))


def _finalize_response(span: Any, response: Any, duration_ms: float) -> None:
    """Extract neatlogs attributes from a non-streaming ChatCompletion response."""
    accumulator = ChoiceAccumulator()
    accumulator.add_response(response)
    accumulator.apply(span)
    span.set_attribute("neatlogs.llm.metrics.duration_ms", round(duration_ms, 3))
    span.set_status(StatusCode.OK)
    span.end()


def _field(value: Any, name: str) -> Any:
    return value.get(name) if isinstance(value, dict) else getattr(value, name, None)


def _extract_responses_output_text(response: Any) -> Optional[str]:
    output_text = _field(response, "output_text")
    if output_text:
        return str(output_text)
    parts: List[str] = []
    for item in _field(response, "output") or []:
        if _field(item, "type") in (None, "message"):
            for content_item in _field(item, "content") or []:
                if _field(content_item, "type") in ("output_text", "text"):
                    txt = _field(content_item, "text")
                    if txt:
                        parts.append(str(txt))
    return "".join(parts) if parts else None


def _finalize_responses_response(span: Any, response: Any, duration_ms: float) -> None:
    """Extract attributes from a Responses API response."""
    output_items = _field(response, "output") or []
    output_text = _extract_responses_output_text(response)
    set_media_attributes(
        span,
        "neatlogs.llm.output_messages.0",
        output_items,
        "output",
    )
    if output_text:
        span.set_attribute("neatlogs.llm.output_messages.0.role", "assistant")
        span.set_attribute("neatlogs.llm.output_messages.0.content", output_text)

    model = _field(response, "model")
    if model:
        span.set_attribute("neatlogs.llm.model_name", model)

    usage = _field(response, "usage")
    if usage:
        input_tokens = _field(usage, "input_tokens")
        output_tokens = _field(usage, "output_tokens")
        total_tokens = _field(usage, "total_tokens")
        if total_tokens is None and input_tokens is not None and output_tokens is not None:
            total_tokens = input_tokens + output_tokens
        if input_tokens is not None:
            span.set_attribute("neatlogs.llm.token_count.prompt", input_tokens)
        if output_tokens is not None:
            span.set_attribute("neatlogs.llm.token_count.completion", output_tokens)
        if total_tokens is not None:
            span.set_attribute("neatlogs.llm.token_count.total", total_tokens)

    _set_responses_tool_calls(span, output_items)
    failure = _set_responses_outcome(span, response)
    span.set_attribute("neatlogs.llm.metrics.duration_ms", round(duration_ms, 3))
    if failure:
        _set_responses_failure(span, *failure)
    else:
        span.set_status(StatusCode.OK)
    span.end()


def _set_responses_tool_calls(span: Any, items: Any) -> None:
    """Record the function calls among a response's output items."""
    index = 0
    for item in items:
        if _field(item, "type") != "function_call":
            continue
        prefix = f"neatlogs.llm.tool_calls.{index}"
        span.set_attribute(f"{prefix}.id", str(_field(item, "call_id") or _field(item, "id") or ""))
        span.set_attribute(f"{prefix}.type", "function")
        span.set_attribute(f"{prefix}.name", str(_field(item, "name") or ""))
        span.set_attribute(f"{prefix}.arguments", str(_field(item, "arguments") or ""))
        index += 1
    if index:
        span.set_attribute("neatlogs.llm.output_messages.0.role", "assistant")


def _set_responses_outcome(span: Any, response: Any) -> Optional[tuple]:
    """Record how a response ended. Returns (kind, message) when it failed.

    The Responses API reports a failed generation in the body, with HTTP 200
    and no exception, so the status has to be read from the response itself.
    """
    status = _field(response, "status")
    reason = _field(_field(response, "incomplete_details"), "reason")
    if status:
        span.set_attribute("neatlogs.llm.finish_reason", str(reason or status))
    error = _field(response, "error")
    if status == "failed" or error:
        return (
            str(_field(error, "code") or "response_failed"),
            str(_field(error, "message") or "The response failed"),
        )
    return None


def _set_responses_failure(span: Any, kind: str, message: str) -> None:
    span.add_event("exception", {"exception.type": kind, "exception.message": message})
    span.set_status(StatusCode.ERROR, message)


# ---------------------------------------------------------------------------
# Generic method-patching helper (used by the additional resources below)
# ---------------------------------------------------------------------------


def _patch_method(
    resource: Any, method_name: str, flag: str, start_attrs, finalize, is_async: bool
) -> None:
    """
    Wrap resource.<method_name> with a span. start_attrs(kwargs)->dict builds the
    initial attributes; finalize(span, response) records the result. Idempotent
    per-resource via `flag`.
    """
    if resource is None or getattr(resource, flag, False) or not hasattr(resource, method_name):
        return
    orig = getattr(resource, method_name)

    def _safe_finalize(span, resp, duration_ms):
        try:
            finalize(span, resp, duration_ms)
        except Exception:
            span.set_status(StatusCode.OK)
            span.end()

    if is_async:

        async def patched(*args, **kwargs):
            if is_suppressed():
                return await orig(*args, **kwargs)
            marker = _RAW_CALL_MARKER.get()
            if marker is not None:
                marker["handled"] = True
            raw_mode = _raw_response_mode(kwargs)
            tracer = get_provider_tracer()
            span = tracer.start_span(name=start_attrs.__name__, attributes=start_attrs(kwargs))
            start = time.perf_counter()
            try:
                response = await orig(*args, **kwargs)
            except Exception as e:
                span.set_status(StatusCode.ERROR, str(e))
                span.record_exception(e)
                span.end()
                raise
            if _is_raw_response(response):
                return await _instrument_async_raw_response(
                    span, response, start, False, _safe_finalize, None, raw_mode
                )
            _safe_finalize(span, response, (time.perf_counter() - start) * 1000)
            return response

    else:

        def patched(*args, **kwargs):
            if is_suppressed():
                return orig(*args, **kwargs)
            marker = _RAW_CALL_MARKER.get()
            if marker is not None:
                marker["handled"] = True
            raw_mode = _raw_response_mode(kwargs)
            tracer = get_provider_tracer()
            span = tracer.start_span(name=start_attrs.__name__, attributes=start_attrs(kwargs))
            start = time.perf_counter()
            try:
                response = orig(*args, **kwargs)
            except Exception as e:
                span.set_status(StatusCode.ERROR, str(e))
                span.record_exception(e)
                span.end()
                raise
            if _is_raw_response(response):
                return _instrument_raw_response(
                    span, response, start, False, _safe_finalize, None, raw_mode
                )
            _safe_finalize(span, response, (time.perf_counter() - start) * 1000)
            return response

    setattr(resource, method_name, patched)
    if method_name == "create":
        _reset_raw_wrappers(resource)
    setattr(resource, flag, True)


def _ok(span, duration_ms):
    span.set_attribute("neatlogs.llm.metrics.duration_ms", round(duration_ms, 3))
    span.set_status(StatusCode.OK)
    span.end()


# ---------------------------------------------------------------------------
# Async + streaming Responses API
# ---------------------------------------------------------------------------


def _patch_async_responses(responses: Any) -> None:
    if getattr(responses, "_neatlogs_patched", False):
        return
    orig_create = responses.create

    async def patched_create(*args, **kwargs):
        if is_suppressed():
            return await orig_create(*args, **kwargs)
        marker = _RAW_CALL_MARKER.get()
        if marker is not None:
            marker["handled"] = True
        is_stream = kwargs.get("stream", False)
        raw_mode = _raw_response_mode(kwargs)
        span = _start_responses_span(kwargs)
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
                _finalize_responses_response,
                _finalize_responses_stream,
                raw_mode,
            )
        if is_stream:
            return AsyncStreamWrapper(response, span, _finalize_responses_stream)
        _finalize_responses_response(span, response, (time.perf_counter() - start) * 1000)
        return response

    responses.create = patched_create
    _reset_raw_wrappers(responses)
    _patch_non_stainless_raw_wrappers(
        responses,
        _start_responses_span,
        _finalize_responses_response,
        _finalize_responses_stream,
        is_async=True,
    )
    responses._neatlogs_patched = True


def _finalize_responses_stream(
    span: Any,
    chunks: List[Any],
    duration_ms: float,
    ttft_ms: Optional[float],
    *,
    interrupted: bool = False,
) -> None:
    """Finalize a streaming Responses API span (events carry .type / .delta / .response)."""
    text_parts: List[str] = []
    set_media_attributes(span, "neatlogs.llm.output_messages.0", chunks, "output")
    model = None
    usage = None
    final_response = None
    finished_items: List[Any] = []
    failure: Optional[tuple] = None
    for ev in chunks:
        ev_type = _field(ev, "type") or ""
        if ev_type == "response.output_text.delta":
            d = _field(ev, "delta")
            if d:
                text_parts.append(d)
        elif ev_type == "response.output_item.done":
            finished_items.append(_field(ev, "item"))
        elif ev_type == "error":
            # The server reports the failure as an event; nothing is raised.
            failure = (
                str(_field(ev, "code") or "error"),
                str(_field(ev, "message") or "The stream reported an error"),
            )
        resp = _field(ev, "response")
        if resp is not None:
            if _field(resp, "model"):
                model = _field(resp, "model")
            if _field(resp, "usage"):
                usage = _field(resp, "usage")
            if ev_type in ("response.completed", "response.failed", "response.incomplete"):
                final_response = resp
    if final_response is not None:
        _set_responses_tool_calls(span, _field(final_response, "output") or [])
        failure = _set_responses_outcome(span, final_response) or failure
    else:
        _set_responses_tool_calls(span, finished_items)
    if text_parts:
        span.set_attribute("neatlogs.llm.output_messages.0.role", "assistant")
        span.set_attribute("neatlogs.llm.output_messages.0.content", "".join(text_parts)[:10000])
    if model:
        span.set_attribute("neatlogs.llm.model_name", model)
    if usage:
        it = _field(usage, "input_tokens")
        ot_ = _field(usage, "output_tokens")
        tt = _field(usage, "total_tokens")
        if tt is None and it is not None and ot_ is not None:
            tt = it + ot_
        if it is not None:
            span.set_attribute("neatlogs.llm.token_count.prompt", it)
        if ot_ is not None:
            span.set_attribute("neatlogs.llm.token_count.completion", ot_)
        if tt is not None:
            span.set_attribute("neatlogs.llm.token_count.total", tt)
    span.set_attribute("neatlogs.llm.metrics.duration_ms", round(duration_ms, 3))
    if ttft_ms is not None:
        span.set_attribute("neatlogs.llm.metrics.ttft_ms", round(ttft_ms, 3))
    if failure:
        # An app usually stops reading after an error event, which reaches here
        # as an interrupted stream; the failure still decides the status.
        _set_responses_failure(span, *failure)
    else:
        span.set_status(StatusCode.UNSET if interrupted else StatusCode.OK)
    span.end()


# Extend the sync responses.create to support streaming too.
def _patch_responses_streaming_aware(responses: Any) -> None:
    pass  # sync streaming handled inline in _patch_responses below (kept for clarity)


# ---------------------------------------------------------------------------
# chat.completions.parse  /  beta.chat.completions.parse  (structured outputs)
# ---------------------------------------------------------------------------


def _patch_chat_parse(completions: Any, sync: bool = True) -> None:
    def start_attrs(kwargs):
        return {
            "neatlogs.span.kind": "llm",
            "neatlogs.llm.provider": "openai",
            "neatlogs.llm.system": "openai",
            "neatlogs.llm.model_name": kwargs.get("model", ""),
            "neatlogs.llm.structured_output": True,
        }

    start_attrs.__name__ = "openai.chat.completions.parse"

    def finalize(span, response, duration_ms):
        _finalize_response(span, response, duration_ms)

    _patch_method(
        completions, "parse", "_neatlogs_parse_patched", start_attrs, finalize, is_async=not sync
    )


def _patch_responses_parse(responses: Any, sync: bool = True) -> None:
    def start_attrs(kwargs):
        return {
            "neatlogs.span.kind": "llm",
            "neatlogs.llm.provider": "openai",
            "neatlogs.llm.model_name": kwargs.get("model", ""),
            "neatlogs.llm.structured_output": True,
            "neatlogs.llm.input_messages.0.role": "user",
            "neatlogs.llm.input_messages.0.content": serialize(kwargs.get("input", "")),
        }

    start_attrs.__name__ = "openai.responses.parse"

    def finalize(span, response, duration_ms):
        _finalize_responses_response(span, response, duration_ms)

    _patch_method(
        responses, "parse", "_neatlogs_parse_patched", start_attrs, finalize, is_async=not sync
    )


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------


def _patch_embeddings(embeddings: Any, sync: bool = True) -> None:
    def start_attrs(kwargs):
        inp = kwargs.get("input", "")
        attrs = {
            "neatlogs.span.kind": "embedding",
            "neatlogs.embedding.model_name": kwargs.get("model", ""),
        }
        if isinstance(inp, str):
            attrs["neatlogs.embedding.text"] = inp[:10000]
        elif isinstance(inp, list):
            attrs["neatlogs.embedding.text"] = serialize(inp[:20])[:10000]
        return attrs

    start_attrs.__name__ = "openai.embeddings.create"

    def finalize(span, response, duration_ms):
        usage = getattr(response, "usage", None)
        if usage is not None:
            if getattr(usage, "prompt_tokens", None) is not None:
                span.set_attribute("neatlogs.llm.token_count.prompt", usage.prompt_tokens)
            if getattr(usage, "total_tokens", None) is not None:
                span.set_attribute("neatlogs.embedding.token_count", usage.total_tokens)
        data = getattr(response, "data", None)
        if data is not None:
            try:
                span.set_attribute("neatlogs.embedding.count", len(data))
                if data and getattr(data[0], "embedding", None) is not None:
                    span.set_attribute("neatlogs.embedding.dimensions", len(data[0].embedding))
            except (TypeError, AttributeError):
                pass
        model = getattr(response, "model", None)
        if model:
            span.set_attribute("neatlogs.embedding.model_name", model)
        _ok(span, duration_ms)

    _patch_method(
        embeddings, "create", "_neatlogs_patched", start_attrs, finalize, is_async=not sync
    )


# ---------------------------------------------------------------------------
# Legacy completions (text)
# ---------------------------------------------------------------------------


def _patch_legacy_completions(completions: Any, sync: bool = True) -> None:
    def start_attrs(kwargs):
        prompt = kwargs.get("prompt", "")
        return {
            "neatlogs.span.kind": "llm",
            "neatlogs.llm.provider": "openai",
            "neatlogs.llm.system": "openai",
            "neatlogs.llm.model_name": kwargs.get("model", ""),
            "neatlogs.llm.input_messages.0.role": "user",
            "neatlogs.llm.input_messages.0.content": (
                prompt if isinstance(prompt, str) else serialize(prompt)
            )[:10000],
        }

    start_attrs.__name__ = "openai.completions.create"

    def finalize(span, response, duration_ms):
        choices = getattr(response, "choices", []) or []
        if choices:
            text = getattr(choices[0], "text", None)
            if text:
                span.set_attribute("neatlogs.llm.output_messages.0.role", "assistant")
                span.set_attribute("neatlogs.llm.output_messages.0.content", text[:10000])
            fr = getattr(choices[0], "finish_reason", None)
            if fr:
                span.set_attribute("neatlogs.llm.finish_reason", fr)
        usage = getattr(response, "usage", None)
        if usage:
            if getattr(usage, "prompt_tokens", None) is not None:
                span.set_attribute("neatlogs.llm.token_count.prompt", usage.prompt_tokens)
            if getattr(usage, "completion_tokens", None) is not None:
                span.set_attribute("neatlogs.llm.token_count.completion", usage.completion_tokens)
            if getattr(usage, "total_tokens", None) is not None:
                span.set_attribute("neatlogs.llm.token_count.total", usage.total_tokens)
        _ok(span, duration_ms)

    _patch_method(
        completions, "create", "_neatlogs_legacy_patched", start_attrs, finalize, is_async=not sync
    )


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------


def _set_generated_image_media(span: Any, data: Any) -> None:
    # Image API items do not carry a ``type`` discriminator. Supply the known
    # resource context so URL-only and base64 responses share canonical media.
    set_media_attributes(span, "neatlogs.llm.output_messages.0", {"image": data}, "output")


def _patch_images(images: Any, sync: bool = True) -> None:
    for method, span_name in (
        ("generate", "openai.images.generate"),
        ("edit", "openai.images.edit"),
        ("create_variation", "openai.images.create_variation"),
    ):

        def make(method=method, span_name=span_name):
            def start_attrs(kwargs):
                attrs = {
                    "neatlogs.span.kind": "llm",
                    "neatlogs.llm.provider": "openai",
                    "neatlogs.llm.task": "image",
                }
                if kwargs.get("model"):
                    attrs["neatlogs.llm.model_name"] = kwargs["model"]
                if kwargs.get("prompt"):
                    attrs["input.value"] = str(kwargs["prompt"])[:10000]
                if kwargs.get("size"):
                    attrs["neatlogs.image.size"] = str(kwargs["size"])
                return attrs

            start_attrs.__name__ = span_name

            def finalize(span, response, duration_ms):
                data = getattr(response, "data", None)
                if data is not None:
                    _set_generated_image_media(span, data)
                    try:
                        span.set_attribute("neatlogs.image.count", len(data))
                    except TypeError:
                        pass
                _ok(span, duration_ms)

            return start_attrs, finalize

        sa, fin = make()
        _patch_method(images, method, f"_neatlogs_{method}_patched", sa, fin, is_async=not sync)


# ---------------------------------------------------------------------------
# Audio (speech / transcriptions / translations)
# ---------------------------------------------------------------------------


def _set_speech_response_media(span: Any, response: Any) -> None:
    """Capture the buffered speech body without exposing raw bytes as attributes."""

    try:
        content = response if isinstance(response, (bytes, bytearray)) else response.content
        if not isinstance(content, (bytes, bytearray)) or not content:
            return
        headers = getattr(response, "headers", {}) if not isinstance(response, bytes) else {}
        content_type = headers.get("content-type", "audio/mpeg") if hasattr(headers, "get") else ""
        set_media_attributes(
            span,
            "neatlogs.llm.output_messages.0",
            {
                "type": "audio",
                "mime_type": content_type or "audio/mpeg",
                "data": content,
            },
            "output",
        )
    except Exception:
        return


def _patch_audio(audio: Any, sync: bool = True) -> None:
    speech = getattr(audio, "speech", None)
    if speech is not None:

        def start_attrs(kwargs):
            attrs = {
                "neatlogs.span.kind": "llm",
                "neatlogs.llm.provider": "openai",
                "neatlogs.llm.task": "tts",
            }
            if kwargs.get("model"):
                attrs["neatlogs.llm.model_name"] = kwargs["model"]
            if kwargs.get("input"):
                attrs["input.value"] = str(kwargs["input"])[:10000]
            if kwargs.get("voice"):
                attrs["neatlogs.audio.voice"] = str(kwargs["voice"])
            return attrs

        start_attrs.__name__ = "openai.audio.speech.create"

        def finalize_speech(span, response, duration_ms):
            _set_speech_response_media(span, response)
            _ok(span, duration_ms)

        _patch_method(
            speech,
            "create",
            "_neatlogs_patched",
            start_attrs,
            finalize_speech,
            is_async=not sync,
        )

    for sub, task in (("transcriptions", "stt"), ("translations", "translation")):
        res = getattr(audio, sub, None)
        if res is None:
            continue

        def make(task=task, sub=sub):
            def start_attrs(kwargs):
                attrs = {
                    "neatlogs.span.kind": "llm",
                    "neatlogs.llm.provider": "openai",
                    "neatlogs.llm.task": task,
                }
                if kwargs.get("model"):
                    attrs["neatlogs.llm.model_name"] = kwargs["model"]
                return attrs

            start_attrs.__name__ = f"openai.audio.{sub}.create"

            def finalize(span, response, duration_ms):
                text = getattr(response, "text", None)
                if text:
                    span.set_attribute("neatlogs.llm.output_messages.0.role", "assistant")
                    span.set_attribute("neatlogs.llm.output_messages.0.content", str(text)[:10000])
                _ok(span, duration_ms)

            return start_attrs, finalize

        sa, fin = make()
        _patch_method(res, "create", "_neatlogs_patched", sa, fin, is_async=not sync)


# ---------------------------------------------------------------------------
# Moderations
# ---------------------------------------------------------------------------


def _patch_moderations(moderations: Any, sync: bool = True) -> None:
    def start_attrs(kwargs):
        attrs = {
            "neatlogs.span.kind": "llm",
            "neatlogs.llm.provider": "openai",
            "neatlogs.llm.task": "moderation",
        }
        if kwargs.get("model"):
            attrs["neatlogs.llm.model_name"] = kwargs["model"]
        inp = kwargs.get("input")
        if inp:
            attrs["input.value"] = (inp if isinstance(inp, str) else serialize(inp))[:10000]
        return attrs

    start_attrs.__name__ = "openai.moderations.create"

    def finalize(span, response, duration_ms):
        results = getattr(response, "results", None)
        if results:
            try:
                span.set_attribute(
                    "neatlogs.moderation.flagged", bool(getattr(results[0], "flagged", False))
                )
            except (TypeError, AttributeError):
                pass
        _ok(span, duration_ms)

    _patch_method(
        moderations, "create", "_neatlogs_patched", start_attrs, finalize, is_async=not sync
    )


# ---------------------------------------------------------------------------
# Batches
# ---------------------------------------------------------------------------


def _patch_batches(batches: Any, sync: bool = True) -> None:
    def start_attrs(kwargs):
        attrs = {
            "neatlogs.span.kind": "task",
            "neatlogs.batch.endpoint": kwargs.get("endpoint", ""),
        }
        return attrs

    start_attrs.__name__ = "openai.batches.create"

    def finalize(span, response, duration_ms):
        bid = getattr(response, "id", None)
        if bid:
            span.set_attribute("neatlogs.batch.id", str(bid))
        status = getattr(response, "status", None)
        if status:
            span.set_attribute("neatlogs.batch.status", str(status))
        _ok(span, duration_ms)

    _patch_method(batches, "create", "_neatlogs_patched", start_attrs, finalize, is_async=not sync)


# ---------------------------------------------------------------------------
# Import-replacement: `from neatlogs.openai import openai`
# Patches OpenAI/AsyncOpenAI.__init__ so every client is auto-wrapped.
# ---------------------------------------------------------------------------


def _patch_openai_module() -> None:
    global _PATCHED, _ORIG_INIT, _ORIG_ASYNC_INIT
    if _PATCHED:
        return
    _PATCHED = True

    import openai as _openai

    _ORIG_INIT = _openai.OpenAI.__init__
    _ORIG_ASYNC_INIT = _openai.AsyncOpenAI.__init__

    def _patched_init(self, *args, **kwargs):
        _ORIG_INIT(self, *args, **kwargs)
        wrap_openai_client(self)

    _openai.OpenAI.__init__ = _patched_init

    def _patched_async_init(self, *args, **kwargs):
        _ORIG_ASYNC_INIT(self, *args, **kwargs)
        wrap_async_openai_client(self)

    _openai.AsyncOpenAI.__init__ = _patched_async_init


def _unpatch_openai_module() -> None:
    global _PATCHED, _ORIG_INIT, _ORIG_ASYNC_INIT
    if not _PATCHED:
        return

    import openai as _openai

    if _ORIG_INIT is not None:
        _openai.OpenAI.__init__ = _ORIG_INIT
    if _ORIG_ASYNC_INIT is not None:
        _openai.AsyncOpenAI.__init__ = _ORIG_ASYNC_INIT

    _PATCHED = False
    _ORIG_INIT = None
    _ORIG_ASYNC_INIT = None


_patch_openai_module()

import openai  # noqa: E402
