"""
Neatlogs DSPy wrapper.

Usage:
    >>> import neatlogs
    >>> import dspy
    >>> predict = neatlogs.wrap(dspy.Predict("question -> answer"))
    >>> result = predict(question="What is 2+2?")

Patches at the DSPy class level (idempotent, global) so nested module
composition is fully traced:

    CHAIN     dspy.Module.__call__   (Predict / ChainOfThought / ReAct / custom — every module call, nested)
      ↳ LLM        dspy.LM.__call__       (the underlying language-model request)
      ↳ RETRIEVER  dspy.Retrieve.forward  (retrieval calls)

Calling neatlogs.wrap(module) installs the class hooks once and returns the
module unchanged. Because hooks live on the base classes, sub-modules invoked
internally (e.g. the Predict inside a ChainOfThought, or each ReAct iteration)
all nest correctly under the active span and under user @span / trace() blocks.
"""

import time
from typing import Any

from opentelemetry.trace import StatusCode

from ._wrap_utils import attach_as_current, detach, get_tracer, serialize

_CLASS_HOOKS_INSTALLED = False


def wrap_dspy(module: Any) -> Any:
    """
    Install DSPy class-level tracing hooks (Module / LM / Retrieve) and return
    the module unchanged. Idempotent.
    """
    _install_class_hooks()
    return module


def _install_class_hooks() -> None:
    global _CLASS_HOOKS_INSTALLED
    if _CLASS_HOOKS_INSTALLED:
        return
    _CLASS_HOOKS_INSTALLED = True
    _patch_module_class()
    _patch_lm_class()
    _patch_retrieve_class()


# ---------------------------------------------------------------------------
# Module (CHAIN span) — universal entry point for every module call
# ---------------------------------------------------------------------------


def _patch_module_class() -> None:
    try:
        import dspy
    except Exception:
        return
    Module = getattr(dspy, "Module", None)
    if Module is None or getattr(Module, "_neatlogs_patched", False):
        return

    orig_call = Module.__call__

    def patched_call(self, *args, **kwargs):
        # Skip the abstract base itself being called directly (shouldn't happen).
        tracer = get_tracer()
        cls_name = type(self).__name__
        attrs = {"neatlogs.span.kind": "chain", "neatlogs.entity.name": cls_name}

        signature = getattr(self, "signature", None)
        if signature is not None:
            sig_str = signature if isinstance(signature, str) else str(signature)
            attrs["neatlogs.dspy.signature"] = sig_str[:2000]

        if kwargs:
            attrs["input.value"] = serialize(kwargs)[:10000]
        elif args:
            attrs["input.value"] = serialize(args)[:10000]

        span = tracer.start_span(name=f"dspy.{cls_name}", attributes=attrs)
        token = attach_as_current(span)
        start = time.perf_counter()
        try:
            result = orig_call(self, *args, **kwargs)
        except Exception as e:
            _err(span, e); raise
        finally:
            detach(token)

        _finalize_module(span, result, (time.perf_counter() - start) * 1000)
        return result

    Module.__call__ = patched_call
    Module._neatlogs_patched = True


def _finalize_module(span: Any, result: Any, duration_ms: float) -> None:
    if result is not None:
        if hasattr(result, "toDict"):
            try:
                span.set_attribute("output.value", serialize(result.toDict())[:10000])
            except Exception:
                span.set_attribute("output.value", str(result)[:10000])
        elif hasattr(result, "__dict__"):
            output = {k: str(v) for k, v in vars(result).items() if not k.startswith("_")}
            span.set_attribute("output.value", serialize(output)[:10000])
        else:
            span.set_attribute("output.value", str(result)[:10000])
    span.set_attribute("neatlogs.llm.metrics.duration_ms", round(duration_ms, 3))
    span.set_status(StatusCode.OK)
    span.end()


# ---------------------------------------------------------------------------
# LM (LLM span) — the underlying language-model request
# ---------------------------------------------------------------------------


def _patch_lm_class() -> None:
    try:
        import dspy
    except Exception:
        return
    LM = getattr(dspy, "LM", None)
    if LM is None or getattr(LM, "_neatlogs_patched", False):
        return
    if not hasattr(LM, "__call__"):
        return

    orig_call = LM.__call__

    def patched_call(self, prompt=None, messages=None, **kwargs):
        tracer = get_tracer()
        attrs = {"neatlogs.span.kind": "llm", "neatlogs.llm.provider": "dspy"}

        model = getattr(self, "model", None)
        if model:
            attrs["neatlogs.llm.model_name"] = str(model)

        # Input messages. Not truncated here — the backend enforces the 1 MB
        # payload cap (S3 offload + preview) on ingest.
        collected = []
        if messages and isinstance(messages, list):
            for i, msg in enumerate(messages):
                role = msg.get("role", "") if isinstance(msg, dict) else getattr(msg, "role", "")
                content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
                content_str = content if isinstance(content, str) else serialize(content)
                if role:
                    attrs[f"neatlogs.llm.input_messages.{i}.role"] = role
                if content:
                    attrs[f"neatlogs.llm.input_messages.{i}.content"] = content_str
                collected.append({"role": role or "user", "content": content_str})
        elif prompt:
            attrs["neatlogs.llm.input_messages.0.role"] = "user"
            attrs["neatlogs.llm.input_messages.0.content"] = str(prompt)
            collected.append({"role": "user", "content": str(prompt)})
        # Flat input blob the UI renders for LLM spans.
        if collected:
            attrs["neatlogs.llm.input"] = serialize({"messages": collected})

        for param in ("temperature", "max_tokens", "top_p"):
            val = kwargs.get(param) or (getattr(self, "kwargs", {}) or {}).get(param)
            if val is not None:
                attrs[f"neatlogs.llm.{param}"] = val

        span = tracer.start_span(name="dspy.lm", attributes=attrs)
        token = attach_as_current(span)
        start = time.perf_counter()
        try:
            result = orig_call(self, prompt=prompt, messages=messages, **kwargs)
        except Exception as e:
            _err(span, e); raise
        finally:
            detach(token)

        _finalize_lm(span, self, result, (time.perf_counter() - start) * 1000)
        return result

    LM.__call__ = patched_call
    LM._neatlogs_patched = True


def _finalize_lm(span: Any, lm: Any, result: Any, duration_ms: float) -> None:
    # result is typically a list[str] of completions; usage lives on lm.history[-1]
    if result is not None:
        if isinstance(result, list) and result:
            content = str(result[0])
        else:
            content = str(result)
        span.set_attribute("neatlogs.llm.output_messages.0.role", "assistant")
        span.set_attribute("neatlogs.llm.output_messages.0.content", content)
        # Flat output blob the UI renders for LLM spans. Not truncated here — the
        # backend enforces the 1 MB payload cap (S3 offload + preview) on ingest.
        span.set_attribute(
            "neatlogs.llm.output",
            serialize({"role": "assistant", "content": content}),
        )

    try:
        history = getattr(lm, "history", None)
        if history:
            last = history[-1]
            resp = last.get("response") if isinstance(last, dict) else None

            # Token usage: read from the response object's `usage`, NOT from
            # history[-1]["usage"]. DSPy appends the history entry and only fills
            # in the "usage"/"cost" keys AFTER LM.__call__ returns — so at finalize
            # time history[-1]["usage"] is still {}. The ModelResponse returned by
            # the call already carries usage (litellm attaches it synchronously).
            usage = getattr(resp, "usage", None)
            # Fallback to the history dict's usage in case the response shape differs.
            if not usage and isinstance(last, dict):
                usage = last.get("usage") or None

            def _u(key):
                if usage is None:
                    return None
                if isinstance(usage, dict):
                    return usage.get(key)
                return getattr(usage, key, None)

            prompt_t = _u("prompt_tokens")
            completion_t = _u("completion_tokens")
            total_t = _u("total_tokens")
            if prompt_t:
                span.set_attribute("neatlogs.llm.token_count.prompt", prompt_t)
            if completion_t:
                span.set_attribute("neatlogs.llm.token_count.completion", completion_t)
            if total_t:
                span.set_attribute("neatlogs.llm.token_count.total", total_t)

            # Cached prompt tokens / reasoning tokens, when present (parity with the
            # OpenAI wrapper). Cost itself is computed by the backend from tokens +
            # model — the SDK does not emit a cost attribute.
            pd = _u("prompt_tokens_details")
            cached = getattr(pd, "cached_tokens", None) if pd is not None else None
            if cached:
                span.set_attribute("neatlogs.llm.token_count.cache_read", cached)
            cd = _u("completion_tokens_details")
            reasoning = getattr(cd, "reasoning_tokens", None) if cd is not None else None
            if reasoning:
                span.set_attribute("neatlogs.llm.token_count.reasoning", reasoning)

            model = getattr(resp, "model", None) if resp is not None else None
            if model:
                span.set_attribute("neatlogs.llm.model_name", str(model))
    except (AttributeError, IndexError, KeyError, TypeError):
        pass

    span.set_attribute("neatlogs.llm.metrics.duration_ms", round(duration_ms, 3))
    span.set_status(StatusCode.OK)
    span.end()


# ---------------------------------------------------------------------------
# Retrieve (RETRIEVER span)
# ---------------------------------------------------------------------------


def _patch_retrieve_class() -> None:
    try:
        import dspy
    except Exception:
        return
    Retrieve = getattr(dspy, "Retrieve", None)
    if Retrieve is None or getattr(Retrieve, "_neatlogs_patched", False):
        return
    if "forward" not in Retrieve.__dict__:
        return

    orig_forward = Retrieve.forward

    def patched_forward(self, query_or_queries, k=None, *args, **kwargs):
        tracer = get_tracer()
        attrs = {"neatlogs.span.kind": "retriever"}
        if query_or_queries is not None:
            attrs["neatlogs.retrieval.query"] = (
                query_or_queries if isinstance(query_or_queries, str) else serialize(query_or_queries)
            )[:10000]
        effective_k = k if k is not None else getattr(self, "k", None)
        if effective_k is not None:
            attrs["neatlogs.retrieval.top_k"] = effective_k

        span = tracer.start_span(name="dspy.retrieve", attributes=attrs)
        token = attach_as_current(span)
        start = time.perf_counter()
        try:
            result = orig_forward(self, query_or_queries, k, *args, **kwargs)
        except Exception as e:
            _err(span, e); raise
        finally:
            detach(token)

        passages = getattr(result, "passages", None) if result is not None else None
        if passages is None and result is not None:
            passages = result
        if passages is not None:
            try:
                span.set_attribute("neatlogs.retrieval.document_count", len(passages))
            except TypeError:
                pass
            span.set_attribute("neatlogs.retrieval.documents", serialize(passages)[:10000])
        span.set_attribute("neatlogs.llm.metrics.duration_ms", round((time.perf_counter() - start) * 1000, 3))
        span.set_status(StatusCode.OK)
        span.end()
        return result

    Retrieve.forward = patched_forward
    Retrieve._neatlogs_patched = True


def _err(span: Any, e: Exception) -> None:
    span.set_status(StatusCode.ERROR, str(e))
    span.record_exception(e)
    span.end()
