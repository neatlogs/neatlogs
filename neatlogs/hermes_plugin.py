"""Neatlogs observer plugin for Hermes (NousResearch/hermes-agent).

This is the **recommended** way to trace the standalone ``hermes`` CLI / gateway:
it plugs into Hermes' native observer-hook contract (``hermes.observer.v1``) and
emits Neatlogs spans with **zero changes to user code**. Hermes discovers it via
the ``hermes_agent.plugins`` entry-point declared in this SDK's packaging.

Enable it once Hermes and neatlogs are installed in the same environment::

    hermes plugins enable neatlogs

and provide credentials (env, or ~/.hermes/.env)::

    NEATLOGS_API_KEY   - Neatlogs project write key (required)
    NEATLOGS_ENDPOINT  - Backend URL (optional; default https://staging-cloud.neatlogs.com,
                         the same canonical default neatlogs.init() uses)

Traces are grouped under the workflow name "hermes". If the API key is missing the
hooks no-op silently (fail-open).

Span tree produced — ONE session = ONE trace, with a single WORKFLOW root::

    WORKFLOW  hermes.session             (the root; one per Hermes session)
      AGENT   hermes.turn                 (one per user message / run_conversation)
        LLM   hermes.api_request          (one per provider attempt; usage + I/O)
        TOOL  hermes.tool.<name>          (one per tool dispatch)
        AGENT hermes.subagent.<role>      (a delegated child, under the spawning TURN)
          AGENT hermes.turn               (the child agent's own turns)
            LLM / TOOL

Deterministic IDs (like @neatlogs/claude-code): the session's trace_id and root
span_id are derived from the session_id via SHA-256, so EVERY process — the agent
loop AND kanban worker subprocesses, which run in separate Python interpreters —
recomputes the same IDs and parents its spans under the one session trace without
shared memory. The root span is emitted (and ended) the moment the session starts
so the trace renders immediately; each child span exports as it completes, so the
trace grows live.

Why this plugin and not the openai instrumentor: in plugin mode the user never
calls ``neatlogs.init()``, so the auto-instrumentors are not active — these hooks
are the only tracer, so we build the LLM span from the sanitized request/response
payloads Hermes hands us. (For the *library* usage — importing ``run_agent`` in
your own code — use ``neatlogs.init(instrumentations=["hermes", "openai"])`` +
``neatlogs.wrap(AIAgent(...))`` instead; see ``neatlogs/hermes.py``.)

The observer contract is read-only: every callback accepts ``**kwargs`` so future
additive fields stay backward-compatible, and we never alter Hermes' behavior.
"""

from __future__ import annotations

import hashlib
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from opentelemetry import trace as otel_trace
from opentelemetry.trace import (
    NonRecordingSpan,
    SpanContext,
    StatusCode,
    TraceFlags,
    set_span_in_context,
)

from opentelemetry.sdk.trace.id_generator import RandomIdGenerator

from ._wrap_utils import configure, get_tracer, serialize
from .core.logger import get_logger


class _FixedIdGen(RandomIdGenerator):
    """Forces a specific trace_id + span_id for a single span start (used to pin
    the session root to its deterministic ids). Subclasses the real generator so
    the SDK's id_generator interface (e.g. is_trace_id_random) stays intact."""

    def __init__(self, trace_id: int, span_id: int):
        self._trace_id = trace_id
        self._span_id = span_id

    def generate_span_id(self) -> int:
        return self._span_id

    def generate_trace_id(self) -> int:
        return self._trace_id

logger = get_logger()

_PROVIDER = "hermes"
_MAX_TURN_STATE = 256       # bound the leak from turns that never finalize cleanly
_MAX_SESSION_STATE = 256    # bound session roots that never finalize


# ---------------------------------------------------------------------------
# Span state
#
# Hierarchy is threaded by EXPLICIT parent context (hook callbacks are separate
# invocations and can't share the ambient active span):
#   session root  ── parent of ──▶  turn  ── parent of ──▶  llm / tool / subagent
#   subagent      ── parent of ──▶  child-session turns
# ---------------------------------------------------------------------------
@dataclass
class _TurnState:
    """An AGENT turn span plus the LLM/TOOL children currently open under it."""

    span: Any
    session_id: Optional[str] = None  # owning session — lets cleanup find turns
    llm_spans: Dict[str, Any] = field(default_factory=dict)   # keyed by api_request_id
    tool_spans: Dict[str, Any] = field(default_factory=dict)  # keyed by tool span key
    last_updated_at: float = field(default_factory=time.time)


_STATE_LOCK = threading.Lock()
# session_ids whose deterministic WORKFLOW root span has already been emitted in
# THIS process (so we emit it once per process; other processes emit their own
# copy — same deterministic id, so the backend dedups/merges on span id).
_ROOTS_EMITTED: set = set()
# turn_key -> _TurnState.
_TURN_STATE: Dict[str, _TurnState] = {}
# Delegated-child AGENT spans. A subagent parents under the spawning TURN; the
# child's own turns (which fire with the CHILD's session_id) parent under the
# subagent span via _SUBAGENT_BY_SESSION.
_SUBAGENT_STATE: Dict[str, Any] = {}        # subagent key -> subagent AGENT span
_SUBAGENT_BY_SESSION: Dict[str, Any] = {}   # child_session_id -> subagent AGENT span
# Dangerous-command approval GUARDRAIL spans, keyed by approval key, open between
# pre_approval_request and post_approval_response.
_APPROVAL_SPANS: Dict[str, Any] = {}
_CONFIGURED = False

# Approval choices that mean the command was allowed through the gate; anything
# else (deny / timeout) means the guardrail blocked it.
_APPROVAL_ALLOWED = frozenset({"once", "session", "always"})


# ---------------------------------------------------------------------------
# Bootstrap (lazy, fail-open)
# ---------------------------------------------------------------------------
def _ensure_configured() -> bool:
    """Wire the Neatlogs tracer from env on first use. Returns False (and stays
    silent after one warning) when no API key is available, so all hooks no-op."""
    global _CONFIGURED
    if _CONFIGURED:
        return True

    api_key = os.environ.get("NEATLOGS_API_KEY", "").strip()
    if not api_key:
        logger.warning(
            "neatlogs hermes plugin: NEATLOGS_API_KEY not set — tracing disabled. "
            "Set it in your environment or ~/.hermes/.env to capture Hermes traces."
        )
        # Mark configured so we don't warn on every hook. Re-enable on a fresh process.
        _CONFIGURED = True
        return False

    # Resolve the endpoint to the SAME canonical default as neatlogs.init()
    # (staging-cloud). We can't omit it: configure()'s wrapper-mode bootstrap
    # defaults to a DIFFERENT host (cloud.neatlogs.com), so relying on that path
    # would silently export to the wrong backend. Honor NEATLOGS_ENDPOINT when set.
    endpoint = (
        os.environ.get("NEATLOGS_ENDPOINT", "").strip()
        or "https://staging-cloud.neatlogs.com"
    )
    # workflow_name is an init()/configure() argument, not an env var, so we pass
    # our default directly. configure() only sets wrapper-mode defaults; if
    # neatlogs.init() already ran in this process (mixed library+plugin use),
    # get_tracer() reuses that provider and these values are ignored — no double export.
    configure(api_key=api_key, workflow_name="hermes", endpoint=endpoint)
    _CONFIGURED = True
    return True


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _turn_key(turn_id: Any, api_request_id: Any, session_id: Any, task_id: Any) -> Optional[str]:
    """Stable key joining a turn's hooks. Prefer turn_id (shared by API attempts
    and tool calls in a turn); fall back through the other correlation IDs."""
    for candidate in (turn_id, api_request_id, session_id, task_id):
        if candidate:
            return str(candidate)
    return None


def _session_of(session_id: Any = None, turn_id: Any = None) -> Optional[str]:
    """Resolve the owning session id for a span. Hermes' turn_id is compound —
    ``"<session_id>:<uuid>:<frag>"`` — so when an explicit session_id is absent
    (e.g. some approval payloads carry only turn_id), the session is the prefix
    before the first ':'. This is what lets approval/late spans attach to the
    SAME deterministic session root instead of starting an orphan trace."""
    if session_id:
        return str(session_id)
    if turn_id:
        return str(turn_id).split(":", 1)[0]
    return None


def _safe_end(span: Any) -> None:
    try:
        if span is not None:
            span.end()
    except Exception:
        pass


def _child_context(parent: Any):
    """OTel context with ``parent`` as the parent span — passed explicitly to
    start_span so children nest correctly across separate hook invocations.
    Valid even when ``parent`` has already ended."""
    return otel_trace.set_span_in_context(parent)


def _set_text(span: Any, attr: str, value: Any) -> None:
    if value is None:
        return
    span.set_attribute(attr, (value if isinstance(value, str) else serialize(value))[:10000])


# ---------------------------------------------------------------------------
# Deterministic session IDs — derived from session_id via SHA-256 so EVERY
# process (agent loop + kanban worker subprocesses) recomputes the same trace
# and root span id, joining one trace without shared memory.
# ---------------------------------------------------------------------------
def _det_trace_id(session_id: str) -> int:
    return int.from_bytes(hashlib.sha256(session_id.encode()).digest()[:16], "big")


def _det_span_id(session_id: str) -> int:
    return int.from_bytes(hashlib.sha256((session_id + ":root").encode()).digest()[:8], "big")


def _root_span_context(session_id: str) -> SpanContext:
    """The deterministic, non-recording SpanContext for a session's WORKFLOW root.
    Children parent under this; any process can rebuild it from session_id."""
    return SpanContext(
        trace_id=_det_trace_id(session_id),
        span_id=_det_span_id(session_id),
        is_remote=True,
        trace_flags=TraceFlags(TraceFlags.SAMPLED),
    )


def _session_parent_ctx(session_id: str):
    """An OTel Context whose current span is the deterministic root — pass as
    ``context=`` to start_span to nest a span directly under the session root."""
    return set_span_in_context(NonRecordingSpan(_root_span_context(session_id)))


def _emit_session_root(session_id: str, *, model: Any = None,
                       user_message: Any = None) -> None:
    """Emit the WORKFLOW root span ONCE per process, with the deterministic
    trace/span id so it lines up with children emitted here and in other
    processes. Ended immediately (zero-duration) for live tailing."""
    sid = str(session_id)
    with _STATE_LOCK:
        if sid in _ROOTS_EMITTED:
            return
        _ROOTS_EMITTED.add(sid)
        if len(_ROOTS_EMITTED) > _MAX_SESSION_STATE:
            for old in list(_ROOTS_EMITTED)[: len(_ROOTS_EMITTED) - _MAX_SESSION_STATE]:
                _ROOTS_EMITTED.discard(old)

    tracer = get_tracer()
    # Force the deterministic trace_id + span_id for just this start by swapping
    # the tracer's id generator (subclass of the real one so the SDK interface is
    # intact), then restoring it. Verified safe + reset in the POC.
    fixed = _FixedIdGen(_det_trace_id(sid), _det_span_id(sid))
    orig = getattr(tracer, "id_generator", None)
    try:
        if orig is not None:
            tracer.id_generator = fixed
        root = tracer.start_span(
            name="hermes.session",
            attributes={
                "neatlogs.span.kind": "workflow",
                "neatlogs.agent.framework": _PROVIDER,
                "neatlogs.llm.provider": _PROVIDER,
                "neatlogs.conversation.id": sid,
                "neatlogs.session.id": sid,
            },
        )
    finally:
        if orig is not None:
            tracer.id_generator = orig
    if model:
        root.set_attribute("neatlogs.agent.model", str(model))
    _set_text(root, "neatlogs.input.value", user_message)
    root.set_status(StatusCode.OK)
    _safe_end(root)


def _ensure_session_root(session_id: Any, *, model: Any = None,
                         user_message: Any = None):
    """Ensure the session's WORKFLOW root is emitted, and return a parenting
    Context for it. Returns None when tracing is disabled / no session id."""
    if not _ensure_configured() or not session_id:
        return None
    sid = str(session_id)
    _emit_session_root(sid, model=model, user_message=user_message)
    return _session_parent_ctx(sid)


def _parent_for_session(session_id: Any, *, model: Any = None,
                        user_message: Any = None):
    """The parenting Context a turn nests under: the subagent span if this session
    is a delegated child, otherwise the deterministic session root context."""
    if session_id:
        sub = _SUBAGENT_BY_SESSION.get(str(session_id))
        if sub is not None:
            return _child_context(sub)
    return _ensure_session_root(session_id, model=model, user_message=user_message)


def _emit_completion_marker(session_id: Any) -> None:
    """Emit a ``neatlogs.trace.complete`` marker span on the session's trace.

    REQUIRED for the trace to render: the backend's trace-finalizer only
    processes (finalizes) a trace when it sees this marker span name on the
    trace_id (kafka-consumer matches purely on span_name; the span is filtered
    out, never persisted). Without it, spans ingest (HTTP 200) but never appear
    in the UI. We emit it after every turn (incremental finalize → live tailing)
    and at session finalize, mirroring @neatlogs/claude-code."""
    if not _ensure_configured() or not session_id:
        return
    sid = str(session_id)
    # Only finalize a trace we actually started. Without this, finalize/reset on a
    # session that never emitted a root (no turns) would mark a nonexistent trace.
    with _STATE_LOCK:
        if sid not in _ROOTS_EMITTED:
            return
    try:
        marker = get_tracer().start_span(
            name="neatlogs.trace.complete",
            context=_session_parent_ctx(sid),
        )
        _safe_end(marker)
    except Exception as exc:
        logger.debug("hermes plugin completion marker failed: %s", exc)


# ---------------------------------------------------------------------------
# Turn  (AGENT) — one per run_conversation / user message
# ---------------------------------------------------------------------------
def _ensure_turn(key: str, *, session_id: Any = None, model: Any = None,
                 user_message: Any = None) -> Optional[_TurnState]:
    """Get-or-create the AGENT turn span, parented under the session root (or the
    subagent span for delegated child turns)."""
    if not _ensure_configured():
        return None
    with _STATE_LOCK:
        state = _TURN_STATE.get(key)
        if state is not None:
            state.last_updated_at = time.time()
            return state

    # Resolve the parenting context OUTSIDE the turn lock (it takes its own locks).
    parent_ctx = _parent_for_session(session_id, model=model, user_message=user_message)

    with _STATE_LOCK:
        state = _TURN_STATE.get(key)
        if state is not None:
            state.last_updated_at = time.time()
            return state

        start_kwargs = {"context": parent_ctx} if parent_ctx is not None else {}
        span = get_tracer().start_span(
            name="hermes.turn",
            attributes={
                "neatlogs.span.kind": "agent",
                "neatlogs.agent.framework": _PROVIDER,
                "neatlogs.llm.provider": _PROVIDER,
            },
            **start_kwargs,
        )
        if session_id:
            span.set_attribute("neatlogs.conversation.id", str(session_id))
        if model:
            span.set_attribute("neatlogs.agent.model", str(model))
        _set_text(span, "neatlogs.input.value", user_message)

        state = _TurnState(span=span, session_id=str(session_id) if session_id else None)
        _TURN_STATE[key] = state
        if len(_TURN_STATE) > _MAX_TURN_STATE:
            stale = sorted(_TURN_STATE.items(), key=lambda kv: kv[1].last_updated_at)
            for k, st in stale[: len(_TURN_STATE) - _MAX_TURN_STATE]:
                _safe_end(st.span)
                _TURN_STATE.pop(k, None)
        return state


# ---------------------------------------------------------------------------
# Hook callbacks — all keyword-only, **kwargs for forward-compat, fail-open.
# ---------------------------------------------------------------------------
def on_session_start(**kwargs: Any) -> None:
    """A new session began. We do NOT emit the root here: a session that never
    produces a turn/tool/subagent (user opens hermes, types nothing, quits) would
    otherwise leave a childless WORKFLOW root — a lone-root orphan trace. The root
    is created lazily on the first real child (see _parent_for_session), so an
    empty session produces no trace at all. Kept registered for forward-compat."""
    return None


def on_pre_llm_call(**kwargs: Any) -> None:
    """Turn begins — open the AGENT turn span and stamp the user input."""
    try:
        key = _turn_key(kwargs.get("turn_id"), kwargs.get("api_request_id"),
                        kwargs.get("session_id"), kwargs.get("task_id"))
        if not key:
            return
        _ensure_turn(key, session_id=kwargs.get("session_id"),
                     model=kwargs.get("model"), user_message=kwargs.get("user_message"))
    except Exception as exc:
        logger.debug("hermes plugin on_pre_llm_call failed: %s", exc)


def on_post_llm_call(**kwargs: Any) -> None:
    """Turn ends — stamp the assistant output and close the AGENT turn span."""
    try:
        key = _turn_key(kwargs.get("turn_id"), kwargs.get("api_request_id"),
                        kwargs.get("session_id"), kwargs.get("task_id"))
        if not key:
            return
        with _STATE_LOCK:
            state = _TURN_STATE.pop(key, None)
        if state is None:
            return
        # Close any LLM/TOOL children left open by an interrupted path.
        for child in list(state.llm_spans.values()) + list(state.tool_spans.values()):
            _safe_end(child)
        out = kwargs.get("assistant_response")
        if out is None:
            out = kwargs.get("assistant_message")
        _set_text(state.span, "neatlogs.output.value", out)
        state.span.set_status(StatusCode.OK)
        _safe_end(state.span)
        # Marker AFTER the turn span ends → the finalizer renders the trace
        # incrementally (live tailing) with this turn's spans complete.
        _emit_completion_marker(kwargs.get("session_id"))
    except Exception as exc:
        logger.debug("hermes plugin on_post_llm_call failed: %s", exc)


def on_pre_api_request(**kwargs: Any) -> None:
    """Provider attempt begins — open an LLM child under the turn span."""
    try:
        key = _turn_key(kwargs.get("turn_id"), kwargs.get("api_request_id"),
                        kwargs.get("session_id"), kwargs.get("task_id"))
        if not key:
            return
        # A turn may begin at the API layer (legacy paths skip pre_llm_call).
        state = _ensure_turn(key, session_id=kwargs.get("session_id"), model=kwargs.get("model"))
        if state is None:
            return
        req_id = str(kwargs.get("api_request_id") or kwargs.get("api_call_count") or "0")
        span = get_tracer().start_span(
            name="hermes.api_request",
            context=_child_context(state.span),
            attributes={
                "neatlogs.span.kind": "llm",
                "neatlogs.llm.provider": str(kwargs.get("provider") or _PROVIDER),
                "neatlogs.llm.system": str(kwargs.get("provider") or _PROVIDER),
            },
        )
        if kwargs.get("model"):
            span.set_attribute("neatlogs.llm.model_name", str(kwargs["model"]))
        if kwargs.get("max_tokens") is not None:
            span.set_attribute("neatlogs.llm.max_tokens", kwargs["max_tokens"])
        # Input = the prompt messages only. The request payload also carries the
        # model + invocation params (method/temperature/tools/...), but model is
        # already a structured field; dumping the whole envelope is input jargon.
        _set_text(span, "neatlogs.input.value", _request_messages(kwargs.get("request")))
        with _STATE_LOCK:
            state.llm_spans[req_id] = span
            state.last_updated_at = time.time()
    except Exception as exc:
        logger.debug("hermes plugin on_pre_api_request failed: %s", exc)


def on_post_api_request(**kwargs: Any) -> None:
    """Provider attempt succeeded — stamp usage/output and close the LLM span."""
    _finish_api_span(kwargs, error=False)


def on_api_request_error(**kwargs: Any) -> None:
    """Provider attempt failed — record the error and close the LLM span."""
    _finish_api_span(kwargs, error=True)


def _finish_api_span(kwargs: Dict[str, Any], *, error: bool) -> None:
    try:
        key = _turn_key(kwargs.get("turn_id"), kwargs.get("api_request_id"),
                        kwargs.get("session_id"), kwargs.get("task_id"))
        if not key:
            return
        with _STATE_LOCK:
            state = _TURN_STATE.get(key)
            req_id = str(kwargs.get("api_request_id") or kwargs.get("api_call_count") or "0")
            span = state.llm_spans.pop(req_id, None) if state else None
        if span is None:
            return

        if kwargs.get("finish_reason"):
            span.set_attribute("neatlogs.llm.finish_reason", str(kwargs["finish_reason"]))
        if kwargs.get("response_model"):
            span.set_attribute("neatlogs.llm.model_name", str(kwargs["response_model"]))
        _set_usage(span, kwargs.get("usage"))

        if error:
            err = kwargs.get("error") or {}
            msg = err.get("message") if isinstance(err, dict) else str(err)
            span.set_status(StatusCode.ERROR, str(msg or "api_request_error"))
            # http.status_code is kept as-is by the backend attribute mapping.
            if kwargs.get("status_code") is not None:
                span.set_attribute("http.status_code", kwargs["status_code"])
        else:
            # Output = the assistant's actual content only. The response payload
            # also carries model/finish_reason/usage, but those are already
            # captured as structured neatlogs.llm.* fields (→ span_metadata), so
            # dumping the whole dict here just duplicates them as output jargon.
            _set_text(span, "neatlogs.output.value", _response_content(kwargs.get("response")))
            span.set_status(StatusCode.OK)
        _safe_end(span)

        # Close the turn on a NON-RETRYABLE error. Hermes aborts such turns with
        # an early `return` from run_conversation that bypasses the turn finalizer
        # (and post_llm_call), so without this the AGENT turn would linger until
        # eviction — leaving the error trace looking incomplete. A retryable error
        # is followed by a retry/fallback, so we leave the turn open.
        if error and kwargs.get("retryable") is False:
            _finalize_error_turn(key, kwargs.get("error"), kwargs.get("status_code"))
    except Exception as exc:
        logger.debug("hermes plugin _finish_api_span failed: %s", exc)


def _finalize_error_turn(key: str, error: Any, status_code: Any) -> None:
    """End a turn span that aborted on a non-retryable error, stamping the error
    so the turn renders as completed (failed), not open."""
    with _STATE_LOCK:
        state = _TURN_STATE.pop(key, None)
    if state is None:
        return
    for child in list(state.llm_spans.values()) + list(state.tool_spans.values()):
        _safe_end(child)
    msg = error.get("message") if isinstance(error, dict) else (str(error) if error else None)
    state.span.set_status(StatusCode.ERROR, str(msg or "api_request_error"))
    if status_code is not None:
        state.span.set_attribute("http.status_code", status_code)
    _safe_end(state.span)
    # Marker so the aborted-turn error trace finalizes and renders.
    _emit_completion_marker(state.session_id)


def _request_messages(request: Any) -> Any:
    """Pull the prompt messages from Hermes' sanitized request payload
    ``{method, body: {model, messages, tools, temperature, ...}}``. Falls back to
    the body, then the whole request, for unexpected shapes."""
    if not isinstance(request, dict):
        return request
    body = request.get("body")
    if isinstance(body, dict):
        return body.get("messages") or body.get("input") or body
    return request


def _response_content(response: Any) -> Any:
    """Pull just the assistant content from Hermes' sanitized response payload
    ``{model, finish_reason, assistant_message: {role, content, tool_calls}, usage}``.
    Model / finish_reason / usage are captured as structured neatlogs.llm.* fields,
    so the output should be the content (or the tool_calls when there's no text)."""
    if not isinstance(response, dict):
        return response
    msg = response.get("assistant_message")
    if isinstance(msg, dict):
        content = msg.get("content")
        if content:
            return content
        tool_calls = msg.get("tool_calls")
        if tool_calls:
            return tool_calls  # serialized by _set_text — tool-only assistant turn
        return content
    # Unexpected shape — fall back to common content keys, else the whole payload.
    return response.get("content") or response.get("message") or response


def _set_usage(span: Any, usage: Any) -> None:
    """Map Hermes' sanitized usage dict → canonical neatlogs token attributes.

    Hermes' ``_usage_summary_for_api_request_hook`` emits an ``asdict`` of its
    CanonicalUsage (``input_tokens`` / ``output_tokens`` / ``cache_read_tokens`` /
    ``reasoning_tokens`` / ...) plus computed ``prompt_tokens`` and
    ``total_tokens``. We prefer the computed buckets and fall back to the raw
    OpenAI-style names for non-Hermes shapes."""
    if not isinstance(usage, dict):
        return
    prompt = usage.get("prompt_tokens", usage.get("input_tokens"))
    completion = usage.get("completion_tokens", usage.get("output_tokens"))
    total = usage.get("total_tokens")
    if prompt is not None:
        span.set_attribute("neatlogs.llm.token_count.prompt", prompt)
    if completion is not None:
        span.set_attribute("neatlogs.llm.token_count.completion", completion)
    if total is None and prompt is not None and completion is not None:
        total = prompt + completion
    if total is not None:
        span.set_attribute("neatlogs.llm.token_count.total", total)
    cache_read = usage.get("cache_read_tokens")
    if cache_read:
        span.set_attribute("neatlogs.llm.token_count.cache_read", cache_read)
    reasoning = usage.get("reasoning_tokens")
    if reasoning:
        span.set_attribute("neatlogs.llm.token_count.reasoning", reasoning)


def on_pre_tool_call(**kwargs: Any) -> None:
    """Tool dispatch begins — open a TOOL child under the turn span."""
    try:
        key = _turn_key(kwargs.get("turn_id"), kwargs.get("api_request_id"),
                        kwargs.get("session_id"), kwargs.get("task_id"))
        if not key:
            return
        state = _ensure_turn(key, session_id=kwargs.get("session_id"))
        if state is None:
            return
        name = kwargs.get("tool_name") or "tool"
        span = get_tracer().start_span(
            name=f"hermes.tool.{name}",
            context=_child_context(state.span),
            attributes={
                "neatlogs.span.kind": "tool",
                "neatlogs.tool.name": str(name),
            },
        )
        if kwargs.get("tool_call_id"):
            span.set_attribute("neatlogs.tool_call.id", str(kwargs["tool_call_id"]))
        _set_text(span, "neatlogs.input.value", kwargs.get("args"))
        with _STATE_LOCK:
            state.tool_spans[_tool_span_key(kwargs)] = span
            state.last_updated_at = time.time()
    except Exception as exc:
        logger.debug("hermes plugin on_pre_tool_call failed: %s", exc)


def on_post_tool_call(**kwargs: Any) -> None:
    """Tool dispatch finished (ok / error / blocked / cancelled) — close TOOL."""
    try:
        key = _turn_key(kwargs.get("turn_id"), kwargs.get("api_request_id"),
                        kwargs.get("session_id"), kwargs.get("task_id"))
        if not key:
            return
        with _STATE_LOCK:
            state = _TURN_STATE.get(key)
            span = state.tool_spans.pop(_tool_span_key(kwargs), None) if state else None
        if span is None:
            return
        _set_text(span, "neatlogs.output.value", kwargs.get("result"))
        # Hermes' observer-grade status: ok | error | blocked | cancelled. The
        # non-ok variants map to span ERROR; the specific variant is folded into
        # the status message rather than a custom attribute the mapping omits.
        status = str(kwargs.get("status") or "ok")
        if status in ("error", "blocked", "cancelled"):
            span.set_status(StatusCode.ERROR, str(kwargs.get("error_message") or status))
        else:
            span.set_status(StatusCode.OK)
        _safe_end(span)
    except Exception as exc:
        logger.debug("hermes plugin on_post_tool_call failed: %s", exc)


def _tool_span_key(kwargs: Dict[str, Any]) -> str:
    """Key a TOOL span within a turn. tool_call_id is the provider-supplied
    identity; fall back to the tool name when absent."""
    return str(kwargs.get("tool_call_id") or kwargs.get("tool_name") or "tool")


# ---------------------------------------------------------------------------
# Subagents — a delegated child agent nests under the spawning TURN
# ---------------------------------------------------------------------------
def _subagent_key(child_subagent_id: Any, child_session_id: Any) -> Optional[str]:
    # Prefer child_session_id: it's present on BOTH subagent_start and
    # subagent_stop, whereas child_subagent_id is set on start but None on stop
    # (verified against Hermes' delegate_tool emit). Keying on the latter would
    # mismatch → the span never closes/exports. child_session_id is also the key
    # _SUBAGENT_BY_SESSION uses to nest the child's own turns, so they align.
    for c in (child_session_id, child_subagent_id):
        if c:
            return str(c)
    return None


def on_subagent_start(**kwargs: Any) -> None:
    """A delegated child agent was created — open an AGENT span under the turn
    that spawned it (via parent_turn_id), and map the child's session so the
    child's own turns nest beneath this subagent span."""
    try:
        if not _ensure_configured():
            return
        key = _subagent_key(kwargs.get("child_subagent_id"), kwargs.get("child_session_id"))
        if not key:
            return
        # Parent = the spawning TURN span (looked up by parent_turn_id). Fall back
        # to the parent session root context if the turn isn't tracked here.
        parent_turn_id = kwargs.get("parent_turn_id")
        with _STATE_LOCK:
            parent_state = _TURN_STATE.get(str(parent_turn_id)) if parent_turn_id else None
        if parent_state is not None:
            parent_ctx = _child_context(parent_state.span)
        else:
            parent_ctx = _ensure_session_root(kwargs.get("parent_session_id"))

        role = kwargs.get("child_role") or "subagent"
        start_kwargs = {"context": parent_ctx} if parent_ctx is not None else {}
        span = get_tracer().start_span(
            name=f"hermes.subagent.{role}",
            attributes={
                "neatlogs.span.kind": "agent",
                "neatlogs.agent.framework": _PROVIDER,
                "neatlogs.agent.role": str(role),
            },
            **start_kwargs,
        )
        if kwargs.get("child_session_id"):
            span.set_attribute("neatlogs.conversation.id", str(kwargs["child_session_id"]))
        _set_text(span, "neatlogs.input.value", kwargs.get("child_goal"))

        with _STATE_LOCK:
            _SUBAGENT_STATE[key] = span
            if kwargs.get("child_session_id"):
                _SUBAGENT_BY_SESSION[str(kwargs["child_session_id"])] = span
    except Exception as exc:
        logger.debug("hermes plugin on_subagent_start failed: %s", exc)


def on_subagent_stop(**kwargs: Any) -> None:
    """A delegated child agent finished — stamp its summary/status and close it."""
    try:
        key = _subagent_key(kwargs.get("child_subagent_id"), kwargs.get("child_session_id"))
        if not key:
            return
        with _STATE_LOCK:
            span = _SUBAGENT_STATE.pop(key, None)
            csid = kwargs.get("child_session_id")
            if csid:
                _SUBAGENT_BY_SESSION.pop(str(csid), None)
        if span is None:
            return
        _set_text(span, "neatlogs.output.value", kwargs.get("child_summary"))
        status = str(kwargs.get("child_status") or "ok").lower()
        if status in ("error", "failed", "blocked", "cancelled"):
            span.set_status(StatusCode.ERROR, status)
        else:
            span.set_status(StatusCode.OK)
        _safe_end(span)
    except Exception as exc:
        logger.debug("hermes plugin on_subagent_stop failed: %s", exc)


# ---------------------------------------------------------------------------
# Approvals — a dangerous-command approval gate is a GUARDRAIL span under the turn
# ---------------------------------------------------------------------------
def _approval_key(kwargs: Dict[str, Any]) -> Optional[str]:
    """Key an approval span. tool_call_id ties it to the specific tool awaiting
    approval; session_key/pattern_key are fallbacks when that's absent."""
    for c in (kwargs.get("tool_call_id"), kwargs.get("session_key"), kwargs.get("pattern_key")):
        if c:
            return str(c)
    return None


def on_pre_approval_request(**kwargs: Any) -> None:
    """A dangerous command needs user approval — open a GUARDRAIL span under the
    turn that triggered it. (Hermes injects turn_id/tool_call_id from contextvars
    into this hook, so it correlates to the active turn.)"""
    try:
        if not _ensure_configured():
            return
        key = _approval_key(kwargs)
        if not key:
            return
        # Parent = the active turn (by turn_id) when one is live; otherwise the
        # SESSION ROOT (never a fresh root — that would orphan the approval onto
        # its own trace, which is the bug this fixes). The session is resolved
        # from session_id/turn_id-prefix so the guardrail joins the agent's trace.
        turn_id = kwargs.get("turn_id")
        with _STATE_LOCK:
            state = _TURN_STATE.get(str(turn_id)) if turn_id else None
        if state is not None:
            parent_ctx = _child_context(state.span)
        else:
            sid = _session_of(kwargs.get("session_id"), turn_id)
            parent_ctx = _ensure_session_root(sid) if sid else None
        start_kwargs = {"context": parent_ctx} if parent_ctx is not None else {}
        name = kwargs.get("pattern_key") or "approval"
        span = get_tracer().start_span(
            name=f"hermes.approval.{name}",
            attributes={
                "neatlogs.span.kind": "guardrail",
                "neatlogs.guardrail.name": str(name),
                "neatlogs.guardrail.triggered": True,
            },
            **start_kwargs,
        )
        if kwargs.get("tool_call_id"):
            span.set_attribute("neatlogs.tool_call.id", str(kwargs["tool_call_id"]))
        # The command (+ description) is what the user is being asked to approve.
        cmd = kwargs.get("command")
        desc = kwargs.get("description")
        _set_text(span, "neatlogs.input.value", cmd if not desc else f"{cmd}\n\n{desc}")
        with _STATE_LOCK:
            _APPROVAL_SPANS[key] = span
    except Exception as exc:
        logger.debug("hermes plugin on_pre_approval_request failed: %s", exc)


def on_post_approval_response(**kwargs: Any) -> None:
    """The user responded (or it timed out) — record the decision and close the
    GUARDRAIL span. passed=True when allowed (once/session/always), False on
    deny/timeout."""
    try:
        key = _approval_key(kwargs)
        if not key:
            return
        with _STATE_LOCK:
            span = _APPROVAL_SPANS.pop(key, None)
        if span is None:
            return
        choice = str(kwargs.get("choice") or "").lower()
        allowed = choice in _APPROVAL_ALLOWED
        span.set_attribute("neatlogs.guardrail.passed", allowed)
        _set_text(span, "neatlogs.output.value", choice or "no_response")
        if allowed:
            span.set_status(StatusCode.OK)
        else:
            # deny / timeout — the guardrail blocked the command.
            span.set_status(StatusCode.ERROR, f"approval {choice or 'no_response'}")
        _safe_end(span)
    except Exception as exc:
        logger.debug("hermes plugin on_post_approval_response failed: %s", exc)


# ---------------------------------------------------------------------------
# Kanban — task lifecycle, often in a worker SUBPROCESS. The hook payload has no
# session_id, so we recover it from HERMES_SESSION_ID (env) — set by the agent /
# gateway before spawning the worker. With the deterministic root context built
# from that session_id, the kanban span joins the session trace even though the
# worker is a different interpreter with no in-memory root.
# ---------------------------------------------------------------------------
def _kanban_session_id(kwargs: Dict[str, Any]) -> Optional[str]:
    """Resolve the session a kanban task belongs to — WITHOUT any Hermes change.

    The kanban hooks don't carry session_id, and HERMES_SESSION_ID (env) is
    GLOBAL and gets overwritten with the CHILD session during delegation and
    never restored — so a post-delegation kanban task read from env would attach
    to the wrong (child) trace. Instead we look the task up in Hermes' own kanban
    DB via the public get_task() (the task row stores the session that created
    it). The hook fires AFTER the write txn commits, so a read is safe. Fall back
    to the hook's session_id (if ever present) then the env."""
    sid = kwargs.get("session_id")
    if sid:
        return str(sid)
    task_id = kwargs.get("task_id")
    if task_id:
        try:
            from hermes_cli import kanban_db as _kdb
            with _kdb.connect_closing(board=kwargs.get("board")) as conn:
                task = _kdb.get_task(conn, str(task_id))
            row_sid = getattr(task, "session_id", None) if task else None
            if row_sid:
                return str(row_sid)
        except Exception as exc:
            logger.debug("hermes plugin kanban session lookup failed: %s", exc)
    env = os.environ.get("HERMES_SESSION_ID", "").strip()
    return env or None


def _emit_kanban_span(status: str, kwargs: Dict[str, Any]) -> None:
    """Emit a self-contained TASK span for a terminal kanban transition. Claim and
    completion may happen in different processes, so we don't hold a span open
    across them — each terminal event is one span under the session root."""
    try:
        if not _ensure_configured():
            return
        sid = _kanban_session_id(kwargs)
        # _fire_kanban_lifecycle_hook normalizes the positional task_id into a
        # task_id kwarg before invoke_hook, so it's always a keyword here.
        task_id = kwargs.get("task_id")
        start_kwargs = {"context": _session_parent_ctx(sid)} if sid else {}
        # Ensure the root exists in this process too (idempotent, deterministic id).
        if sid:
            _emit_session_root(sid)
        name = task_id or "task"
        span = get_tracer().start_span(
            name=f"hermes.kanban.{name}",
            attributes={
                "neatlogs.span.kind": "task",
                "neatlogs.agent.framework": _PROVIDER,
            },
            **start_kwargs,
        )
        if task_id:
            span.set_attribute("neatlogs.task.id", str(task_id))
        for field_name, attr in (("board", "neatlogs.task.board"),
                                  ("assignee", "neatlogs.task.assignee"),
                                  ("run_id", "neatlogs.task.run_id")):
            val = kwargs.get(field_name)
            if val is not None:
                span.set_attribute(attr, str(val))
        _set_text(span, "neatlogs.output.value", kwargs.get("summary") or kwargs.get("reason"))
        if status in ("blocked",):
            span.set_status(StatusCode.ERROR, "kanban task blocked")
        else:
            span.set_status(StatusCode.OK)
        _safe_end(span)
        # A kanban worker is often a standalone subprocess with no turn hooks, so
        # emit the completion marker here too — otherwise its TASK span ingests
        # but the trace never finalizes/renders.
        if sid:
            _emit_completion_marker(sid)
    except Exception as exc:
        logger.debug("hermes plugin kanban span (%s) failed: %s", status, exc)


def on_kanban_task_completed(**kwargs: Any) -> None:
    _emit_kanban_span("completed", kwargs)


def on_kanban_task_blocked(**kwargs: Any) -> None:
    _emit_kanban_span("blocked", kwargs)


# ---------------------------------------------------------------------------
# Session lifecycle cleanup
# ---------------------------------------------------------------------------
def _close_session(sid: str) -> None:
    """End any turns/subagents still open for a session and drop its root handle.
    The root span itself was already ended (and exported) at creation."""
    with _STATE_LOCK:
        turn_keys = [k for k, st in _TURN_STATE.items() if st.session_id == sid]
        for k in turn_keys:
            st = _TURN_STATE.pop(k)
            for child in list(st.llm_spans.values()) + list(st.tool_spans.values()):
                _safe_end(child)
            _safe_end(st.span)
        sub = _SUBAGENT_BY_SESSION.pop(sid, None)
        if sub is not None:
            for k in [k for k, v in list(_SUBAGENT_STATE.items()) if v is sub]:
                _SUBAGENT_STATE.pop(k, None)
            _safe_end(sub)
    # Final completion marker so the finished session's trace is finalized.
    # Emit BEFORE discarding the root marker — _emit_completion_marker only fires
    # for sessions whose root was emitted (guards against marking empty sessions).
    _emit_completion_marker(sid)
    with _STATE_LOCK:
        # The root span was emitted+ended lazily on first child (deterministic id);
        # just drop the once-per-process emission marker.
        _ROOTS_EMITTED.discard(sid)


def on_session_end(**kwargs: Any) -> None:
    """Per-turn safety net (fires at the end of every run_conversation): close a
    turn that didn't reach post_llm_call. Does NOT drop the session root — the
    session can continue with more turns; that's on_session_finalize/reset."""
    try:
        sid = kwargs.get("session_id")
        if not sid:
            return
        sid = str(sid)
        with _STATE_LOCK:
            for k in [k for k, st in _TURN_STATE.items() if st.session_id == sid]:
                st = _TURN_STATE.pop(k)
                for child in list(st.llm_spans.values()) + list(st.tool_spans.values()):
                    _safe_end(child)
                st.span.set_status(StatusCode.OK)
                _safe_end(st.span)
    except Exception as exc:
        logger.debug("hermes plugin on_session_end failed: %s", exc)


def on_session_finalize(**kwargs: Any) -> None:
    """Session identity torn down — final cleanup of any open spans + root handle."""
    try:
        sid = kwargs.get("session_id")
        if sid:
            _close_session(str(sid))
    except Exception as exc:
        logger.debug("hermes plugin on_session_finalize failed: %s", exc)


def on_session_reset(**kwargs: Any) -> None:
    """Session switched to a new identity — close out the old session's spans."""
    try:
        old = kwargs.get("old_session_id") or kwargs.get("session_id")
        if old:
            _close_session(str(old))
    except Exception as exc:
        logger.debug("hermes plugin on_session_reset failed: %s", exc)


# ---------------------------------------------------------------------------
# Entry point — Hermes calls register(ctx) after discovering this via the
# ``hermes_agent.plugins`` entry-point group and the user enabling it.
# ---------------------------------------------------------------------------
def register(ctx) -> None:
    """Register Neatlogs observer hooks with Hermes' plugin manager."""
    # Session root (the single trace root).
    ctx.register_hook("on_session_start", on_session_start)
    # Turn-scoped (frame the AGENT turn + final summary).
    ctx.register_hook("pre_llm_call", on_pre_llm_call)
    ctx.register_hook("post_llm_call", on_post_llm_call)
    # Request-scoped (the canonical LLM-span source — preferred over turn hooks).
    ctx.register_hook("pre_api_request", on_pre_api_request)
    ctx.register_hook("post_api_request", on_post_api_request)
    ctx.register_hook("api_request_error", on_api_request_error)
    # Tool lifecycle.
    ctx.register_hook("pre_tool_call", on_pre_tool_call)
    ctx.register_hook("post_tool_call", on_post_tool_call)
    # Subagent (delegated child) lifecycle.
    ctx.register_hook("subagent_start", on_subagent_start)
    ctx.register_hook("subagent_stop", on_subagent_stop)
    # Dangerous-command approval gate (GUARDRAIL span).
    ctx.register_hook("pre_approval_request", on_pre_approval_request)
    ctx.register_hook("post_approval_response", on_post_approval_response)
    # Kanban task lifecycle (TASK span; joins the session trace via HERMES_SESSION_ID).
    ctx.register_hook("kanban_task_completed", on_kanban_task_completed)
    ctx.register_hook("kanban_task_blocked", on_kanban_task_blocked)
    # Session lifecycle cleanup.
    ctx.register_hook("on_session_end", on_session_end)
    ctx.register_hook("on_session_finalize", on_session_finalize)
    ctx.register_hook("on_session_reset", on_session_reset)
