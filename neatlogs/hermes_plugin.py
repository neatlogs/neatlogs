"""Neatlogs observer plugin for Hermes (NousResearch/hermes-agent).

This is the **recommended** way to trace the standalone ``hermes`` CLI / gateway:
it plugs into Hermes' native observer-hook contract (``hermes.observer.v1``) and
emits Neatlogs spans with **zero changes to user code**. Hermes discovers it via
the ``hermes_agent.plugins`` entry-point declared in this SDK's packaging.

Enable it once Hermes and neatlogs are installed in the same environment::

    hermes plugins enable neatlogs

and provide credentials (env, or ~/.hermes/.env)::

    NEATLOGS_API_KEY   - Neatlogs project write key (required)
    NEATLOGS_ENDPOINT  - Backend URL (optional; default https://ingest.neatlogs.com,
                         the same canonical default neatlogs.init() uses)

Traces are grouped under the workflow name "hermes". If the API key is missing the
hooks no-op silently (fail-open).

Span tree produced — ONE turn = ONE trace, all turns grouped into a session::

    WORKFLOW  hermes.turn                 (the root; one TRACE per user message / run_conversation)
      LLM   hermes.api_request            (one per provider attempt; usage + I/O)
      TOOL  hermes.tool.<name>            (one per tool dispatch)
      AGENT hermes.subagent.<role>        (a delegated child, under the spawning TURN)
        AGENT hermes.turn                 (the child agent's own turns, nested in this trace)
          LLM / TOOL

Every turn's WORKFLOW root carries ``neatlogs.session.id`` so the backend groups a
session's turn-traces into one multi-turn session in the UI. Kanban tasks (async,
often in a worker subprocess) each become their own trace in the session too.

Deterministic IDs (like @neatlogs/claude-code): each turn's trace_id and root
span_id are derived from the turn key (Hermes' session-prefixed, per-turn turn_id)
via SHA-256, so children emitted live can parent under the not-yet-emitted root and
any process recomputes the same IDs without shared memory. The WORKFLOW root is
emitted ONCE at turn close carrying input+output; children export as they complete
(nesting under the deterministic root context) and the trace finalizes on the
per-turn completion marker.

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
from opentelemetry.sdk.trace.id_generator import RandomIdGenerator
from opentelemetry.trace import (
    NonRecordingSpan,
    SpanContext,
    StatusCode,
    TraceFlags,
    set_span_in_context,
)

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
_MAX_TURN_STATE = 256  # bound the leak from turns that never finalize cleanly


# ---------------------------------------------------------------------------
# Span state
#
# Hierarchy is threaded by EXPLICIT parent context (hook callbacks are separate
# invocations and can't share the ambient active span). Each TURN is its own
# trace, rooted at a WORKFLOW ``hermes.turn`` span:
#   turn (WORKFLOW root)  ── parent of ──▶  llm / tool / subagent
#   subagent              ── parent of ──▶  the child agent's own turns
# All of a session's turn-traces are grouped by ``neatlogs.session.id``.
# ---------------------------------------------------------------------------
@dataclass
class _TurnState:
    """A WORKFLOW turn span (the root of its own trace) plus the LLM/TOOL
    children currently open under it."""

    span: Any
    turn_key: str  # deterministic-id key for this turn's trace
    session_id: Optional[str] = None  # owning session — lets cleanup find turns
    model: Any = None  # remembered for the root at close
    input_value: Any = None  # user message, stamped on the root at close
    llm_spans: Dict[str, Any] = field(default_factory=dict)  # keyed by api_request_id
    tool_spans: Dict[str, Any] = field(default_factory=dict)  # keyed by tool span key
    last_updated_at: float = field(default_factory=time.time)


_STATE_LOCK = threading.Lock()
# turn_keys whose deterministic WORKFLOW root span has already been emitted in
# THIS process (so we emit it once per turn; other processes emit their own copy
# — same deterministic id, so the backend dedups/merges on span id).
_ROOTS_EMITTED: set = set()
# turn_key -> _TurnState.
_TURN_STATE: Dict[str, _TurnState] = {}
# Open TOOL spans keyed by tool-span key (tool_call_id, else tool_name) → (span,
# owning _TurnState). Process-global so on_post_tool_call closes the exact span it
# opened even if the pre/post turn-key resolution diverges (re-entrant dispatch).
_OPEN_TOOL_SPANS: Dict[str, Any] = {}
# Delegated-child AGENT spans. A subagent parents under the spawning TURN; the
# child's own turns (which fire with the CHILD's session_id) parent under the
# subagent span via _SUBAGENT_BY_SESSION.
_SUBAGENT_STATE: Dict[str, Any] = {}  # subagent key -> subagent AGENT span
_SUBAGENT_BY_SESSION: Dict[str, Any] = {}  # child_session_id -> subagent AGENT span
# Dangerous-command approval GUARDRAIL spans, keyed by approval key, open between
# pre_approval_request and post_approval_response.
_APPROVAL_SPANS: Dict[str, Any] = {}
_CONFIGURED = False

# Compression-root resolution. When a conversation gets long Hermes compresses it
# and ROTATES agent.session_id to a fresh id, linking the old one via
# parent_session_id (end_reason="compression"). Those post-compression turns are
# still ONE conversation, so we resolve every raw session_id back to its
# compression root before stamping ``neatlogs.session.id`` — otherwise the session
# splits in the UI. Only the grouping id is resolved; the raw id still keys
# turn/subagent state so the other hooks keep matching. Read-only DB, cached.
_SESSION_ROOT_CACHE: Dict[str, str] = {}
_SESSION_ROOT_LOCK = threading.Lock()

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
    # (ingest.neatlogs.com) and honor NEATLOGS_ENDPOINT when set.
    endpoint = os.environ.get("NEATLOGS_ENDPOINT", "").strip() or "https://ingest.neatlogs.com"
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
# Deterministic per-trace IDs — derived from a TRACE KEY (a turn key, or a
# kanban task key) via SHA-256 so the live root span, the completion marker, and
# any cross-path span (a late approval) all resolve to the SAME trace_id + root
# span_id without shared memory. A cross-process worker recomputes them too.
# ---------------------------------------------------------------------------
def _det_trace_id(key: str) -> int:
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:16], "big")


def _det_root_span_id(key: str) -> int:
    return int.from_bytes(hashlib.sha256((key + ":root").encode()).digest()[:8], "big")


def _root_ctx(key: str):
    """A non-recording OTel Context whose current span is the deterministic root
    for ``key``'s trace — pass as ``context=`` to parent a span (completion
    marker, fallback approval/subagent parent) onto that trace without holding
    the live root span."""
    ctx = SpanContext(
        trace_id=_det_trace_id(key),
        span_id=_det_root_span_id(key),
        is_remote=True,
        trace_flags=TraceFlags(TraceFlags.SAMPLED),
    )
    return set_span_in_context(NonRecordingSpan(ctx))


def _open_session_db():
    """Open Hermes' state.db READ-ONLY (no write lock → never contends with a
    live Hermes writer). Returns None if Hermes isn't importable or the DB is
    absent — callers then fall back to the raw session id."""
    try:
        from hermes_state import DEFAULT_DB_PATH, SessionDB

        if not DEFAULT_DB_PATH.exists():
            return None
        return SessionDB(read_only=True)
    except Exception as exc:
        logger.debug("hermes plugin: session DB unavailable for root resolution: %s", exc)
        return None


def _resolve_session_root(session_id: Any) -> Optional[str]:
    """Resolve a raw per-turn session_id to its COMPRESSION ROOT so all turns of
    one conversation share a ``neatlogs.session.id``.

    Walks ``parent_session_id`` backward following ONLY compression edges — the
    same rule Hermes uses internally (web_server.compression_root): a parent
    counts only when ``parent.end_reason == "compression"`` AND the child started
    at/after the parent ended. Branch/delegate edges are genuine forks and stop
    the walk. Fully defensive: any failure returns the raw id unchanged, so
    grouping degrades to today's behaviour rather than breaking."""
    if not session_id:
        return None
    sid = str(session_id)
    with _SESSION_ROOT_LOCK:
        cached = _SESSION_ROOT_CACHE.get(sid)
    if cached is not None:
        return cached

    root = sid
    db = _open_session_db()
    if db is not None:
        try:
            chain = []
            cur = sid
            visited: set = set()
            # The visited-set guard makes this walk cycle-proof: every node is
            # recorded before we follow its parent, so a repeat exits the loop.
            while cur and cur not in visited:
                visited.add(cur)
                chain.append(cur)
                s = db.get_session(cur)
                if not s:
                    root = cur
                    break
                parent = s.get("parent_session_id")
                if not parent:
                    root = cur
                    break
                parent_session = db.get_session(parent)
                if not parent_session:
                    root = cur
                    break
                parent_ended_at = parent_session.get("ended_at")
                started_at = s.get("started_at")
                is_compression_edge = (
                    parent_session.get("end_reason") == "compression"
                    and parent_ended_at is not None
                    and started_at is not None
                    and started_at >= parent_ended_at
                )
                if not is_compression_edge:
                    root = cur
                    break
                cur = parent
            with _SESSION_ROOT_LOCK:
                for node in chain:
                    _SESSION_ROOT_CACHE[node] = root
        except Exception as exc:
            logger.debug("hermes plugin: session root walk failed for %s: %s", sid, exc)
            root = sid
        finally:
            try:
                db.close()
            except Exception:
                pass

    with _SESSION_ROOT_LOCK:
        _SESSION_ROOT_CACHE[sid] = root
    return root


def _start_root_span(key: str, name: str, kind: str, *, session_id: Any = None, model: Any = None):
    """Start a span forced to ``key``'s deterministic trace_id + root span_id, so
    it IS the root of that trace. Used for the WORKFLOW turn root and standalone
    kanban task traces. Caller ends it (turn roots stay open across the turn;
    kanban roots end immediately)."""
    tracer = get_tracer()
    # get_tracer() hands back a _ForeignParentGuardTracer whose start_span
    # DELEGATES to the wrapped SDK tracer — and the SDK reads id_generator off
    # that inner tracer, not the wrapper (which is slotted and rejects the set).
    # So unwrap to the real SDK tracer and swap its id generator; the guard still
    # applies its parenting behaviour because we call start_span through it.
    sdk_tracer = getattr(tracer, "_tracer", tracer)
    # Force the deterministic ids for just this start by swapping the tracer's id
    # generator (subclass of the real one so the SDK interface stays intact),
    # then restoring it.
    fixed = _FixedIdGen(_det_trace_id(key), _det_root_span_id(key))
    orig = getattr(sdk_tracer, "id_generator", None)
    try:
        if orig is not None:
            sdk_tracer.id_generator = fixed
        span = tracer.start_span(
            name=name,
            attributes={
                "neatlogs.span.kind": kind,
                "neatlogs.agent.framework": _PROVIDER,
                "neatlogs.llm.provider": _PROVIDER,
            },
        )
    finally:
        if orig is not None:
            sdk_tracer.id_generator = orig
    if session_id:
        # neatlogs.session.id drives grouping in the backend, so it carries the
        # compression ROOT (all turns of one conversation → one session).
        span.set_attribute(
            "neatlogs.session.id", _resolve_session_root(session_id) or str(session_id)
        )
        span.set_attribute("neatlogs.conversation.id", str(session_id))
    if model:
        span.set_attribute("neatlogs.agent.model", str(model))
    return span


def _emit_completion_marker(key: Any) -> None:
    """Emit a ``neatlogs.trace.complete`` marker span on ``key``'s trace.

    REQUIRED for the trace to render: the backend's trace-finalizer only
    processes (finalizes) a trace when it sees this marker span name on the
    trace_id (kafka-consumer matches purely on span_name; the span is filtered
    out, never persisted). Without it, spans ingest (HTTP 200) but never appear
    in the UI. Emitted once per turn (its own trace) and per kanban task trace."""
    if not _ensure_configured() or not key:
        return
    k = str(key)
    # Only finalize a trace we actually started here.
    with _STATE_LOCK:
        if k not in _ROOTS_EMITTED:
            return
    try:
        marker = get_tracer().start_span(
            name="neatlogs.trace.complete",
            context=_root_ctx(k),
        )
        _safe_end(marker)
    except Exception as exc:
        logger.debug("hermes plugin completion marker failed: %s", exc)


# ---------------------------------------------------------------------------
# Turn  (WORKFLOW root) — one per run_conversation / user message = one TRACE
# ---------------------------------------------------------------------------
def _ensure_turn(
    key: str, *, session_id: Any = None, model: Any = None, user_message: Any = None
) -> Optional[_TurnState]:
    """Get-or-create the WORKFLOW turn span that ROOTS this turn's trace. Its
    trace_id + span_id are the deterministic ids for ``key`` (so children and the
    completion marker line up), and it carries ``neatlogs.session.id`` so the
    backend groups the session's per-turn traces. A delegated child turn instead
    nests under the spawning subagent span (same trace as the parent turn)."""
    if not _ensure_configured():
        return None
    with _STATE_LOCK:
        state = _TURN_STATE.get(key)
        if state is not None:
            state.last_updated_at = time.time()
            return state
        # A delegated child turn belongs to the parent turn's trace, not its own.
        sub = _SUBAGENT_BY_SESSION.get(str(session_id)) if session_id else None

    with _STATE_LOCK:
        state = _TURN_STATE.get(key)
        if state is not None:
            state.last_updated_at = time.time()
            return state

        if sub is not None:
            # Child turn: an AGENT span nested under the subagent (no new trace).
            span = get_tracer().start_span(
                name="hermes.turn",
                context=_child_context(sub),
                attributes={
                    "neatlogs.span.kind": "agent",
                    "neatlogs.agent.framework": _PROVIDER,
                    "neatlogs.llm.provider": _PROVIDER,
                },
            )
            if session_id:
                span.set_attribute("neatlogs.conversation.id", str(session_id))
            if model:
                span.set_attribute("neatlogs.agent.model", str(model))
            _set_text(span, "neatlogs.input.value", user_message)
        else:
            # Top-level turn: the WORKFLOW root of its own per-turn trace. Left
            # OPEN across the turn; input/output stamped and ended at turn close.
            span = _start_root_span(
                key, "hermes.turn", "workflow", session_id=session_id, model=model
            )
            # A kanban worker (`hermes chat -q "work kanban task <id>"`) is a
            # subprocess dedicated to ONE task, pinned via env. Tag its turn roots
            # with the task identity so they're attributable to the task — the
            # terminal kanban_complete/kanban_block already lands as a TOOL child.
            _stamp_kanban_task_context(span)
            _ROOTS_EMITTED.add(key)

        state = _TurnState(
            span=span,
            turn_key=key,
            session_id=str(session_id) if session_id else None,
            model=model,
            input_value=user_message,
        )
        _TURN_STATE[key] = state
        if len(_TURN_STATE) > _MAX_TURN_STATE:
            stale = sorted(_TURN_STATE.items(), key=lambda kv: kv[1].last_updated_at)
            for k, st in stale[: len(_TURN_STATE) - _MAX_TURN_STATE]:
                # Purge any open tool spans from the global registry (still under
                # _STATE_LOCK, so inline rather than via _drain_tool_spans).
                for tk, sp in list(st.tool_spans.items()):
                    _OPEN_TOOL_SPANS.pop(tk, None)
                    _safe_end(sp)
                st.tool_spans.clear()
                _safe_end(st.span)
                _ROOTS_EMITTED.discard(k)
                _TURN_STATE.pop(k, None)
        return state


# ---------------------------------------------------------------------------
# Hook callbacks — all keyword-only, **kwargs for forward-compat, fail-open.
# ---------------------------------------------------------------------------
def on_session_start(**kwargs: Any) -> None:
    """A new session began. Nothing to emit: there is no session-level root
    anymore — each turn is its own trace, opened lazily on its first hook. A
    session with no turns produces no trace at all. Kept registered for
    forward-compat."""
    return None


def on_pre_llm_call(**kwargs: Any) -> None:
    """Turn begins — open the WORKFLOW turn trace and remember the user input."""
    try:
        key = _turn_key(
            kwargs.get("turn_id"),
            kwargs.get("api_request_id"),
            kwargs.get("session_id"),
            kwargs.get("task_id"),
        )
        if not key:
            return
        state = _ensure_turn(
            key,
            session_id=kwargs.get("session_id"),
            model=kwargs.get("model"),
            user_message=kwargs.get("user_message"),
        )
        # If the API layer opened the turn first (no user_message then), backfill
        # the input/model now so the root carries them at close.
        if state is not None:
            with _STATE_LOCK:
                if kwargs.get("user_message") is not None:
                    state.input_value = kwargs.get("user_message")
                if kwargs.get("model") is not None and state.model is None:
                    state.model = kwargs.get("model")
    except Exception as exc:
        logger.debug("hermes plugin on_pre_llm_call failed: %s", exc)


def on_post_llm_call(**kwargs: Any) -> None:
    """Turn ends — stamp input+output on the WORKFLOW root, close it, and finalize
    this turn's trace with a completion marker."""
    try:
        key = _turn_key(
            kwargs.get("turn_id"),
            kwargs.get("api_request_id"),
            kwargs.get("session_id"),
            kwargs.get("task_id"),
        )
        if not key:
            return
        with _STATE_LOCK:
            state = _TURN_STATE.pop(key, None)
        if state is None:
            return
        # Close any LLM/TOOL children left open by an interrupted path.
        for child in list(state.llm_spans.values()):
            _safe_end(child)
        _drain_tool_spans(state)
        _set_text(state.span, "neatlogs.input.value", state.input_value)
        out = kwargs.get("assistant_response")
        if out is None:
            out = kwargs.get("assistant_message")
        _set_text(state.span, "neatlogs.output.value", out)
        state.span.set_status(StatusCode.OK)
        _safe_end(state.span)
        # Marker on the TURN'S trace key AFTER the root ends → the finalizer
        # renders this turn's trace complete. A child turn (no own root) doesn't
        # emit a marker — its spans belong to the parent turn's trace.
        _finalize_turn_trace(state)
    except Exception as exc:
        logger.debug("hermes plugin on_post_llm_call failed: %s", exc)


def _finalize_turn_trace(state: _TurnState) -> None:
    """Emit the completion marker for a top-level turn's own trace, and drop its
    once-per-trace root flag. Child turns (which nest in the parent turn's trace)
    have no own root flag, so this no-ops for them."""
    with _STATE_LOCK:
        is_root = state.turn_key in _ROOTS_EMITTED
    if is_root:
        _emit_completion_marker(state.turn_key)
        with _STATE_LOCK:
            _ROOTS_EMITTED.discard(state.turn_key)


def on_pre_api_request(**kwargs: Any) -> None:
    """Provider attempt begins — open an LLM child under the turn span."""
    try:
        key = _turn_key(
            kwargs.get("turn_id"),
            kwargs.get("api_request_id"),
            kwargs.get("session_id"),
            kwargs.get("task_id"),
        )
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
        key = _turn_key(
            kwargs.get("turn_id"),
            kwargs.get("api_request_id"),
            kwargs.get("session_id"),
            kwargs.get("task_id"),
        )
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
    for child in list(state.llm_spans.values()):
        _safe_end(child)
    _drain_tool_spans(state)
    _set_text(state.span, "neatlogs.input.value", state.input_value)
    msg = error.get("message") if isinstance(error, dict) else (str(error) if error else None)
    state.span.set_status(StatusCode.ERROR, str(msg or "api_request_error"))
    if status_code is not None:
        state.span.set_attribute("http.status_code", status_code)
    _safe_end(state.span)
    # Marker so the aborted-turn's trace finalizes and renders.
    _finalize_turn_trace(state)


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


def _resolve_open_turn(kwargs: Dict[str, Any]) -> Optional[_TurnState]:
    """Find the OPEN turn a tool call belongs to — WITHOUT ever minting a root.

    A tool is always requested by the LLM mid-turn, so an open turn already
    exists. But a re-entrant dispatch (e.g. a tool run from inside execute_code)
    can reach pre_tool_call with a turn_id no LLM hook used, so its _turn_key
    misses _TURN_STATE. Calling _ensure_turn there would mint a fresh WORKFLOW
    root that never gets a post_llm_call → never ends/exports → stuck trace with
    an orphan tool child. So we only ATTACH to a live turn: try the key, then the
    session's most-recent open turn, then (single live turn) that one. Returns
    None if truly no turn is open — the caller then skips rather than orphaning."""
    with _STATE_LOCK:
        key = _turn_key(
            kwargs.get("turn_id"),
            kwargs.get("api_request_id"),
            kwargs.get("session_id"),
            kwargs.get("task_id"),
        )
        state = _TURN_STATE.get(key) if key else None
        if state is not None:
            return state
        sid = kwargs.get("session_id")
        if sid:
            sid = str(sid)
            candidates = [s for s in _TURN_STATE.values() if s.session_id == sid]
            if candidates:
                return max(candidates, key=lambda s: s.last_updated_at)
        if len(_TURN_STATE) == 1:
            return next(iter(_TURN_STATE.values()))
    return None


def on_pre_tool_call(**kwargs: Any) -> None:
    """Tool dispatch begins — open a TOOL child under the OPEN turn span. Never
    mints a root: an unmatched tool attaches to the live turn (see
    _resolve_open_turn), else is skipped so it can't orphan a ghost trace."""
    try:
        if not _ensure_configured():
            return
        state = _resolve_open_turn(kwargs)
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
        tskey = _tool_span_key(kwargs)
        with _STATE_LOCK:
            state.tool_spans[tskey] = span
            _OPEN_TOOL_SPANS[tskey] = (span, state)
            state.last_updated_at = time.time()
    except Exception as exc:
        logger.debug("hermes plugin on_pre_tool_call failed: %s", exc)


def on_post_tool_call(**kwargs: Any) -> None:
    """Tool dispatch finished (ok / error / blocked / cancelled) — close TOOL.

    Closes the exact span opened in pre_tool_call via the process-global
    _OPEN_TOOL_SPANS registry, so key divergence between pre/post (re-entrant
    dispatch) can't leave the span open."""
    try:
        tskey = _tool_span_key(kwargs)
        with _STATE_LOCK:
            entry = _OPEN_TOOL_SPANS.pop(tskey, None)
            span = None
            if entry is not None:
                span, owner = entry
                owner.tool_spans.pop(tskey, None)
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
        # NOTE: terminal kanban tools (kanban_complete / kanban_block) need no
        # special handling — this very TOOL span already captures the action, and
        # it lives inside the worker's WORKFLOW turn trace (which finalizes). The
        # task identity is stamped on that turn root (_stamp_kanban_task_context).
        # A separate TASK-rooted trace would be redundant AND wouldn't finalize:
        # the backend only treats WORKFLOW/CHAIN/AGENT/MCP_TOOL roots as
        # simplifiable (trace-finalizer/root-readiness.ts) — a task is not a root.
    except Exception as exc:
        logger.debug("hermes plugin on_post_tool_call failed: %s", exc)


def _tool_span_key(kwargs: Dict[str, Any]) -> str:
    """Key a TOOL span within a turn. tool_call_id is the provider-supplied
    identity; fall back to the tool name when absent."""
    return str(kwargs.get("tool_call_id") or kwargs.get("tool_name") or "tool")


def _drain_tool_spans(state: _TurnState) -> None:
    """End every TOOL span still open under ``state`` and purge its entries from
    the process-global _OPEN_TOOL_SPANS registry. Called from every turn-teardown
    path so an interrupted tool can't leak a registry entry (which would later
    mis-close a same-keyed tool in another turn)."""
    with _STATE_LOCK:
        keys = list(state.tool_spans.keys())
        spans = list(state.tool_spans.values())
        state.tool_spans.clear()
        for k in keys:
            _OPEN_TOOL_SPANS.pop(k, None)
    for span in spans:
        _safe_end(span)


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
        # Parent = the spawning TURN span (looked up by parent_turn_id), so the
        # subagent joins that turn's trace. Fall back to the parent turn's
        # deterministic root context (same trace) when its live span isn't tracked
        # in this process.
        parent_turn_id = kwargs.get("parent_turn_id")
        with _STATE_LOCK:
            parent_state = _TURN_STATE.get(str(parent_turn_id)) if parent_turn_id else None
        if parent_state is not None:
            parent_ctx = _child_context(parent_state.span)
        elif parent_turn_id:
            parent_ctx = _root_ctx(str(parent_turn_id))
        else:
            parent_ctx = None

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
        # turn's deterministic root context (same trace) so the guardrail joins
        # the turn it triggered instead of orphaning onto its own trace.
        turn_id = kwargs.get("turn_id")
        with _STATE_LOCK:
            state = _TURN_STATE.get(str(turn_id)) if turn_id else None
        if state is not None:
            parent_ctx = _child_context(state.span)
        elif turn_id:
            parent_ctx = _root_ctx(str(turn_id))
        else:
            parent_ctx = None
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
# Kanban — task attribution. A kanban worker runs as a dedicated SUBPROCESS
# (`hermes chat -q "work kanban task <id>"`) pinned to ONE task via env
# (HERMES_KANBAN_TASK / _BOARD / _RUN_ID, HERMES_PROFILE). It traces exactly like
# any interactive session — the worker's per-turn WORKFLOW roots and the
# kanban_complete/kanban_block TOOL children are already captured. We do NOT mint
# a separate TASK-rooted trace: it would be redundant with that TOOL span and, by
# design, would never finalize — a task is not a trace root (the backend only
# treats WORKFLOW/CHAIN/AGENT/MCP_TOOL as simplifiable roots). Instead we stamp
# the task identity onto the worker's turn root, so its execution is attributable
# to the task without inventing an artificial root.
# ---------------------------------------------------------------------------
def _stamp_kanban_task_context(span: Any) -> None:
    """If this process is a kanban worker (task pinned in env), tag its turn root
    with the task identity. No-op for ordinary interactive/oneshot turns."""
    try:
        task_id = os.environ.get("HERMES_KANBAN_TASK", "").strip()
        if not task_id:
            return
        span.set_attribute("neatlogs.task.id", task_id)
        board = os.environ.get("HERMES_KANBAN_BOARD", "").strip()
        if board:
            span.set_attribute("neatlogs.task.board", board)
        run_id = os.environ.get("HERMES_KANBAN_RUN_ID", "").strip()
        if run_id:
            span.set_attribute("neatlogs.task.run_id", run_id)
        assignee = os.environ.get("HERMES_PROFILE", "").strip()
        if assignee:
            span.set_attribute("neatlogs.task.assignee", assignee)
    except Exception as exc:
        logger.debug("hermes plugin kanban task stamp failed: %s", exc)


# ---------------------------------------------------------------------------
# Session lifecycle cleanup
# ---------------------------------------------------------------------------
def _close_session(sid: str) -> None:
    """End any turns/subagents still open for a session. Each open top-level turn
    finalizes ITS OWN trace (completion marker on the turn key); child turns just
    close their span in the parent turn's trace."""
    with _STATE_LOCK:
        open_states = [st for st in _TURN_STATE.values() if st.session_id == sid]
    for st in open_states:
        with _STATE_LOCK:
            _TURN_STATE.pop(st.turn_key, None)
        for child in list(st.llm_spans.values()):
            _safe_end(child)
        _drain_tool_spans(st)
        _set_text(st.span, "neatlogs.input.value", st.input_value)
        st.span.set_status(StatusCode.OK)
        _safe_end(st.span)
        _finalize_turn_trace(st)
    with _STATE_LOCK:
        sub = _SUBAGENT_BY_SESSION.pop(sid, None)
        if sub is not None:
            for k in [k for k, v in list(_SUBAGENT_STATE.items()) if v is sub]:
                _SUBAGENT_STATE.pop(k, None)
            _safe_end(sub)


def on_session_end(**kwargs: Any) -> None:
    """Per-turn safety net (fires at the end of every run_conversation): close and
    finalize any turn that didn't reach post_llm_call. Subsequent turns open
    their own new traces, so closing these here is safe."""
    try:
        sid = kwargs.get("session_id")
        if not sid:
            return
        sid = str(sid)
        with _STATE_LOCK:
            open_states = [st for st in _TURN_STATE.values() if st.session_id == sid]
        for st in open_states:
            with _STATE_LOCK:
                _TURN_STATE.pop(st.turn_key, None)
            for child in list(st.llm_spans.values()):
                _safe_end(child)
            _drain_tool_spans(st)
            _set_text(st.span, "neatlogs.input.value", st.input_value)
            st.span.set_status(StatusCode.OK)
            _safe_end(st.span)
            _finalize_turn_trace(st)
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
    # Kanban TASK spans are emitted from on_post_tool_call (kanban_complete /
    # kanban_block) — Hermes has no dedicated kanban lifecycle hook.
    # Session lifecycle cleanup.
    ctx.register_hook("on_session_end", on_session_end)
    ctx.register_hook("on_session_finalize", on_session_finalize)
    ctx.register_hook("on_session_reset", on_session_reset)
