"""
Neatlogs DSPy wrapper.

Usage:
    >>> import neatlogs
    >>> import dspy
    >>> predict = neatlogs.wrap(dspy.Predict("question -> answer"))
    >>> result = predict(question="What is 2+2?")
"""

import time
from typing import Any

from opentelemetry import context as otel_context
from opentelemetry.trace import StatusCode

from ._wrap_utils import get_tracer, serialize


def wrap_dspy(module: Any) -> Any:
    """
    Wrap a DSPy module (Predict, ChainOfThought, custom Module subclass).
    Patches forward() / __call__ to auto-trace. Returns the same module.
    """
    if getattr(module, "_neatlogs_patched", False):
        return module

    _patch_forward(module)
    return module


def _get_module_attributes(module: Any) -> dict:
    """Extract DSPy module metadata as span attributes."""
    attrs = {"neatlogs.span.kind": "CHAIN"}

    cls_name = type(module).__name__
    attrs["neatlogs.entity.name"] = cls_name

    signature = getattr(module, "signature", None)
    if signature:
        sig_str = str(signature) if not isinstance(signature, str) else signature
        attrs["neatlogs.llm.invocation_parameters"] = serialize({"signature": sig_str})

    return attrs


def _extract_usage(module: Any) -> dict:
    """Extract token usage from DSPy's LM history."""
    attrs = {}
    try:
        import dspy
        lm = dspy.settings.lm
        if lm is None:
            return attrs

        history = getattr(lm, "history", None)
        if history and len(history) > 0:
            last = history[-1]
            response = last.get("response", None)
            if response:
                usage = getattr(response, "usage", None)
                if usage:
                    if hasattr(usage, "prompt_tokens") and usage.prompt_tokens:
                        attrs["neatlogs.llm.token_count.prompt"] = usage.prompt_tokens
                    if hasattr(usage, "completion_tokens") and usage.completion_tokens:
                        attrs["neatlogs.llm.token_count.completion"] = usage.completion_tokens
                    if hasattr(usage, "total_tokens") and usage.total_tokens:
                        attrs["neatlogs.llm.token_count.total"] = usage.total_tokens

                choices = getattr(response, "choices", None)
                if choices and len(choices) > 0:
                    finish_reason = getattr(choices[0], "finish_reason", None)
                    if finish_reason:
                        attrs["neatlogs.llm.finish_reason"] = finish_reason

                model = getattr(response, "model", None)
                if model:
                    attrs["neatlogs.llm.model_name"] = model
    except (ImportError, AttributeError, IndexError, KeyError):
        pass

    return attrs


def _patch_forward(module: Any) -> None:
    """Patch the module's forward or __call__ method."""
    if hasattr(module, "forward"):
        orig = module.forward

        def patched_forward(*args, **kwargs):
            tracer = get_tracer()
            attrs = _get_module_attributes(module)

            if kwargs:
                attrs["input.value"] = serialize(kwargs)

            span = tracer.start_span(
                name=f"dspy.{type(module).__name__}.forward",
                attributes=attrs,
            )
            ctx = otel_context.set_value("current_span", span)
            token = otel_context.attach(ctx)
            start = time.perf_counter()

            try:
                result = orig(*args, **kwargs)
            except Exception as e:
                span.set_status(StatusCode.ERROR, str(e))
                span.record_exception(e)
                span.end()
                raise
            finally:
                otel_context.detach(token)

            duration_ms = (time.perf_counter() - start) * 1000
            _finalize_dspy_span(span, module, result, duration_ms)
            return result

        module.forward = patched_forward
        module._neatlogs_patched = True

    elif hasattr(module, "__call__"):
        orig_call = module.__call__

        def patched_call(*args, **kwargs):
            tracer = get_tracer()
            attrs = _get_module_attributes(module)

            if kwargs:
                attrs["input.value"] = serialize(kwargs)

            span = tracer.start_span(
                name=f"dspy.{type(module).__name__}.__call__",
                attributes=attrs,
            )
            ctx = otel_context.set_value("current_span", span)
            token = otel_context.attach(ctx)
            start = time.perf_counter()

            try:
                result = orig_call(*args, **kwargs)
            except Exception as e:
                span.set_status(StatusCode.ERROR, str(e))
                span.record_exception(e)
                span.end()
                raise
            finally:
                otel_context.detach(token)

            duration_ms = (time.perf_counter() - start) * 1000
            _finalize_dspy_span(span, module, result, duration_ms)
            return result

        module.__call__ = patched_call
        module._neatlogs_patched = True


def _finalize_dspy_span(span: Any, module: Any, result: Any, duration_ms: float) -> None:
    """Finalize a DSPy span with result data."""
    if result is not None:
        if hasattr(result, "toDict"):
            span.set_attribute("output.value", serialize(result.toDict()))
        elif hasattr(result, "__dict__"):
            output = {k: str(v) for k, v in result.__dict__.items() if not k.startswith("_")}
            span.set_attribute("output.value", serialize(output))
        else:
            span.set_attribute("output.value", str(result)[:10000])

    for attr_name, value in _extract_usage(module).items():
        span.set_attribute(attr_name, value)

    span.set_attribute("neatlogs.llm.metrics.duration_ms", round(duration_ms, 3))
    span.set_status(StatusCode.OK)
    span.end()
