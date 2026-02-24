"""
Decorator primitives for Neatlogs custom orchestration spans.
"""

from __future__ import annotations

import functools
import inspect
import json
import os
from typing import Any, Callable, Dict, Optional, Tuple, TypeVar

from opentelemetry import trace as otel_trace
from opentelemetry.trace import Status, StatusCode

F = TypeVar("F", bound=Callable[..., Any])


def _should_capture_content() -> bool:
    v = os.getenv("NEATLOGS_TRACE_CONTENT")
    if v is None:
        v = os.getenv("TRACELOOP_TRACE_CONTENT")
    if v is None:
        return True
    return v.lower() not in ("false", "0", "no")


def _serialize_obj(obj: Any) -> Any:
    """
    Convert complex objects to JSON-serializable dicts.
    Handles common Python library objects (Pydantic, dataclasses, ORM models, etc.)
    """
    # Handle None, primitives
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    
    # Handle lists and tuples
    if isinstance(obj, (list, tuple)):
        return [_serialize_obj(item) for item in obj]
    
    # Handle dicts
    if isinstance(obj, dict):
        return {k: _serialize_obj(v) for k, v in obj.items()}
    
    # Try common serialization methods (Pydantic, dataclasses, etc.)
    for method in ['model_dump', 'dict', 'to_dict', 'to_json', 'as_dict']:
        if hasattr(obj, method):
            try:
                result = getattr(obj, method)()
                # to_json returns string, need to parse it
                if method == 'to_json' and isinstance(result, str):
                    return json.loads(result)
                return _serialize_obj(result) if isinstance(result, dict) else result
            except Exception:
                continue
    
    # Try extracting __dict__ (works for many custom classes)
    if hasattr(obj, '__dict__'):
        try:
            # Filter out private attributes and methods
            obj_dict = {
                k: _serialize_obj(v) 
                for k, v in obj.__dict__.items() 
                if not k.startswith('_') and not callable(v)
            }
            if obj_dict:  # Only return if we got some data
                return obj_dict
        except Exception:
            pass
    
    # Last resort: convert to string
    return str(obj)


def _safe_json_dumps(value: Any) -> str:
    try:
        # Use custom serializer that handles complex objects
        serialized = _serialize_obj(value)
        return json.dumps(serialized)
    except Exception:
        # Final fallback: convert entire value to string
        return json.dumps(str(value))


def _bind_call_args(
    func: Callable[..., Any], args: Tuple[Any, ...], kwargs: Dict[str, Any]
) -> Dict[str, Any]:
    try:
        sig = inspect.signature(func)
        bound = sig.bind_partial(*args, **kwargs)
        bound.apply_defaults()
        return dict(bound.arguments)
    except Exception:
        return {"args": list(args), "kwargs": kwargs}


def _set_common_span_attrs(
    span,
    *,
    openinference_kind: str,
    name: str,
    version: Optional[str] = None,
    tags: Optional[list[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    attributes: Optional[Dict[str, Any]] = None,
) -> None:
    span.set_attribute("neatlogs.internal", True)

    span.set_attribute("openinference.span.kind", openinference_kind)
    
    # Set neatlogs.span.kind for simplified view
    span.set_attribute("neatlogs.span.kind", openinference_kind.lower())

    if tags:
        span.set_attribute("tag.tags", tags)
    if metadata:
        span.set_attribute("metadata", _safe_json_dumps(metadata))

    if version:
        span.set_attribute("neatlogs.version", version)

    if attributes:
        for k, v in attributes.items():
            if v is None:
                continue
            try:
                span.set_attribute(k, v)
            except Exception:
                span.set_attribute(k, str(v))


def _decorate_span(
    *,
    openinference_kind: str,
    name: Optional[str] = None,
    description: Optional[str] = None,
    version: Optional[str] = None,
    tags: Optional[list[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    attributes: Optional[Dict[str, Any]] = None,
    capture_input: Optional[bool] = None,
    capture_output: Optional[bool] = None,
    postprocess_result: Optional[Callable[[Any, Any, Dict[str, Any]], None]] = None,
) -> Callable[[F], F]:
    """
    Generic decorator factory for a single span boundary.
    """

    def decorator(func: F) -> F:
        span_name = name or func.__name__
        tracer = otel_trace.get_tracer(__name__)

        cap = _should_capture_content()
        cap_in = cap if capture_input is None else capture_input
        cap_out = cap if capture_output is None else capture_output

        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                with tracer.start_as_current_span(
                    span_name, kind=otel_trace.SpanKind.INTERNAL
                ) as span:
                    _set_common_span_attrs(
                        span,
                        openinference_kind=openinference_kind,
                        name=span_name,
                        version=version,
                        tags=tags,
                        metadata=metadata,
                        attributes=attributes,
                    )
                    if description:
                        span.set_attribute("neatlogs.description", description)

                    bound_inputs: Optional[Dict[str, Any]] = None
                    if cap_in or postprocess_result is not None:
                        bound_inputs = _bind_call_args(func, args, kwargs)
                    if cap_in and bound_inputs is not None:
                        span.set_attribute("input.value", _safe_json_dumps(bound_inputs))
                        span.set_attribute("input.mime_type", "application/json")

                    try:
                        result = await func(*args, **kwargs)
                        if postprocess_result is not None:
                            try:
                                postprocess_result(span, result, bound_inputs or {})
                            except Exception:
                                pass
                        if cap_out:
                            span.set_attribute("output.value", _safe_json_dumps(result))
                            span.set_attribute("output.mime_type", "application/json")
                        span.set_status(Status(StatusCode.OK))
                        return result
                    except Exception as e:
                        span.record_exception(e)
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        raise

            return async_wrapper

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            with tracer.start_as_current_span(span_name, kind=otel_trace.SpanKind.INTERNAL) as span:
                _set_common_span_attrs(
                    span,
                    openinference_kind=openinference_kind,
                    name=span_name,
                    version=version,
                    tags=tags,
                    metadata=metadata,
                    attributes=attributes,
                )
                if description:
                    span.set_attribute("neatlogs.description", description)

                bound_inputs: Optional[Dict[str, Any]] = None
                if cap_in or postprocess_result is not None:
                    bound_inputs = _bind_call_args(func, args, kwargs)
                if cap_in and bound_inputs is not None:
                    span.set_attribute("input.value", _safe_json_dumps(bound_inputs))
                    span.set_attribute("input.mime_type", "application/json")

                try:
                    result = func(*args, **kwargs)
                    if postprocess_result is not None:
                        try:
                            postprocess_result(span, result, bound_inputs or {})
                        except Exception:
                            pass
                    if cap_out:
                        span.set_attribute("output.value", _safe_json_dumps(result))
                        span.set_attribute("output.mime_type", "application/json")
                    span.set_status(Status(StatusCode.OK))
                    return result
                except Exception as e:
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise

        return sync_wrapper

    return decorator
