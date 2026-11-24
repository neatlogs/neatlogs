"""Request-scoped FIFO replay with a one-way transition to live execution."""

from __future__ import annotations

import copy
import inspect
import threading
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Literal

from opentelemetry import context as otel_context

from .contracts import (
    STEP_KINDS,
    ReplayEntry,
    ReplayError,
    ReplayRecordedError,
    ReplayRunBinding,
    json_value,
    parse_replay_recording,
    replay_key,
)

_current = ContextVar("neatlogs.local_replay", default=None)
_environment = None
_context_key = otel_context.create_key("neatlogs.local_replay")


@dataclass(frozen=True)
class ReplayHit:
    entry: ReplayEntry
    outcome: Literal["HIT"] = "HIT"
    overridden: bool = False


@dataclass(frozen=True)
class ReplayErrorHit:
    entry: ReplayEntry
    outcome: Literal["ERROR_HIT"] = "ERROR_HIT"


@dataclass(frozen=True)
class ReplayLive:
    outcome: Literal["MISS", "LIVE"]


class ReplaySession:
    """Replay scoped calls; use Thread(target=...) or explicitly copy thread context."""

    def __init__(
        self,
        recording,
        *,
        mode="tools_only",
        until_span=None,
        run=None,
        overrides=None,
    ):
        self.recording = parse_replay_recording(recording)
        if mode not in ("tools_only", "until_span"):
            raise ReplayError("Unsupported local replay mode.")
        if mode == "tools_only" and until_span is not None:
            raise ReplayError("tools_only does not accept a checkpoint.")
        identities = [entry.span_id for entry in self.recording.entries]
        if mode == "until_span" and until_span not in identities:
            raise ReplayError("The replay checkpoint must exist in this recording.")
        validated_overrides = {}
        if overrides is not None:
            from .provider import safe_body

            if not isinstance(overrides, dict):
                raise ReplayError("Replay overrides must be a mapping of span_id to value.")
            known = set(identities)
            for span_id, replacement in overrides.items():
                if span_id not in known:
                    raise ReplayError("Replay override target must exist in this recording.")
                if callable(replacement):
                    validated_overrides[span_id] = replacement
                else:
                    validated_overrides[span_id] = safe_body(
                        json_value(replacement), provider_transport=False
                    )
        self.mode = mode
        self.until_span = until_span
        self.run_binding = ReplayRunBinding.parse(run) if run is not None else None
        self._overrides = validated_overrides
        self._override_applied = False
        self._checkpoint = identities.index(until_span) if until_span is not None else -1
        self._entries = {}
        self._positions = {}
        for index, entry in enumerate(self.recording.entries):
            try:
                key = replay_key(entry.kind, entry.name, entry.input)
            except ReplayError:
                continue
            self._entries.setdefault(key, []).append((index, entry))
        self._lock = threading.RLock()
        self._used = False
        self._token = None
        self._failure = None
        self._status = {
            "sourceTraceId": self.recording.source_trace_id,
            "mode": mode,
            "hits": 0,
            "liveCalls": 0,
            "switchedToLive": False,
            "reason": None,
            "firstMiss": None,
            "traceIds": [],
        }
        if overrides is not None:
            self._status["overridesApplied"] = 0

    def __enter__(self):
        if current_replay() is not None:
            raise ReplayError("Nested local replay sessions are not supported.")
        with self._lock:
            if self._used:
                raise ReplayError("Create a new ReplaySession for each local run.")
            self._used = True
        from opentelemetry.instrumentation.threading import ThreadingInstrumentor

        instrumentor = ThreadingInstrumentor()
        if not instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.instrument()
        self._context_token = otel_context.attach(otel_context.set_value(_context_key, self))
        self._token = _current.set(self)
        return self

    def __exit__(self, exc_type, *_):
        _current.reset(self._token)
        otel_context.detach(self._context_token)
        self._token = None
        if exc_type is None:
            with self._lock:
                self._raise_if_failed()

    async def __aenter__(self):
        return self.__enter__()

    async def __aexit__(self, *args):
        self.__exit__(*args)

    def run(self, execute, *args, **kwargs):
        if inspect.iscoroutinefunction(execute):
            return self.arun(execute, *args, **kwargs)
        with self:
            result = execute(*args, **kwargs)
            if inspect.isawaitable(result):
                if inspect.iscoroutine(result):
                    result.close()
                raise ReplayError("Use ReplaySession.arun for an awaitable entry function.")
            return result

    async def arun(self, execute, *args, **kwargs):
        async with self:
            result = execute(*args, **kwargs)
            return await result if inspect.isawaitable(result) else result

    def get_status(self):
        with self._lock:
            self._raise_if_failed()
            return copy.deepcopy(self._status)

    def get_fork_lineage(self):
        with self._lock:
            self._raise_if_failed()
            return {
                "forkedFromTraceId": self.recording.source_trace_id,
                "forkedFromSpanId": self.until_span,
                "mode": self.mode,
                "overrideApplied": self._override_applied,
                "overridesApplied": self._status.get("overridesApplied", 0),
                "traceIds": list(self._status["traceIds"]),
            }

    def _resolve_override(self, entry: ReplayEntry) -> ReplayEntry:
        from .provider import safe_body

        replacement = self._overrides[entry.span_id]
        value = replacement(copy.deepcopy(entry)) if callable(replacement) else replacement
        validated = copy.deepcopy(safe_body(json_value(value), provider_transport=False))
        return ReplayEntry(
            span_id=entry.span_id,
            kind=entry.kind,
            name=entry.name,
            input=copy.deepcopy(entry.input),
            output=validated,
            error=None,
        )

    def _raise_if_failed(self):
        if self._failure:
            raise ReplayError(self._failure)

    def _reject_thread_context(self):
        with self._lock:
            self._failure = "This Thread.run override did not forward replay and trace context. Use Thread(target=...) or explicitly copy the context."
            self._raise_if_failed()

    def _miss(self, kind, name):
        self._status.update(
            switchedToLive=True, reason="missing_recording", firstMiss={"kind": kind, "name": name}
        )
        self._status["liveCalls"] += 1
        return ReplayLive("MISS")

    def _live_step(self):
        # Step spans (RETRIEVER/MEMORY/GUARDRAIL/RERANKER/VECTOR_STORE) are optional
        # in CLI and App-supplied recordings; a missed step falls back to live for
        # that call only and must never switch downstream tool/model calls to live.
        self._status["liveCalls"] += 1
        return ReplayLive("LIVE")

    def decide(self, kind, name, input, can_restore=None):
        with self._lock:
            self._raise_if_failed()
            if self._status["switchedToLive"] or self.mode == "tools_only" and kind == "model":
                self._status["liveCalls"] += 1
                return ReplayLive("LIVE")
            try:
                key = replay_key(kind, name, input)
            except ReplayError:
                return self._live_step() if kind in STEP_KINDS else self._miss(kind, name)
            position = self._positions.get(key, 0)
            sequence = self._entries.get(key, [])
            if position >= len(sequence):
                return self._live_step() if kind in STEP_KINDS else self._miss(kind, name)
            index, entry = sequence[position]
            if entry.span_id == self.until_span:
                if entry.span_id in self._overrides:
                    overridden_entry = self._resolve_override(entry)
                    self._positions[key] = position + 1
                    self._status["hits"] += 1
                    self._status["overridesApplied"] = self._status.get("overridesApplied", 0) + 1
                    self._override_applied = True
                    self._status.update(switchedToLive=True, reason="override_fork")
                    return ReplayHit(overridden_entry, overridden=True)
                self._status.update(switchedToLive=True, reason="checkpoint")
                self._status["liveCalls"] += 1
                return ReplayLive("LIVE")
            if self.until_span is not None and index > self._checkpoint:
                return self._live_step() if kind in STEP_KINDS else self._miss(kind, name)
            if entry.span_id in self._overrides:
                overridden_entry = self._resolve_override(entry)
                self._positions[key] = position + 1
                self._status["hits"] += 1
                self._status["overridesApplied"] = self._status.get("overridesApplied", 0) + 1
                self._override_applied = True
                return ReplayHit(overridden_entry, overridden=True)
            if entry.error is not None:
                self._positions[key] = position + 1
                self._status["hits"] += 1
                return ReplayErrorHit(copy.deepcopy(entry))
            if can_restore is not None and not can_restore(entry.output):
                return self._live_step() if kind in STEP_KINDS else self._miss(kind, name)
            self._positions[key] = position + 1
            self._status["hits"] += 1
            return ReplayHit(copy.deepcopy(entry))

    def stamp(self, span):
        context = span.get_span_context()
        trace_id = f"{context.trace_id:032x}"
        with self._lock:
            identities = self._status["traceIds"]
            if context.trace_id and trace_id not in identities and len(identities) < 2:
                identities.append(trace_id)
            override_applied = self._override_applied
        attributes = {
            "neatlogs.origin": "simulation",
            "neatlogs.replay.source_trace_id": self.recording.source_trace_id,
            "neatlogs.replay.mode": self.mode,
        }
        if self.until_span is not None:
            attributes["neatlogs.replay.forked_from_trace_id"] = self.recording.source_trace_id
            attributes["neatlogs.replay.forked_from_span_id"] = self.until_span
        if override_applied:
            attributes["neatlogs.replay.override_applied"] = True
        if self.run_binding is not None:
            attributes.update(
                {
                    "neatlogs.replay.run_id": self.run_binding.run_id,
                    "neatlogs.replay.case_id": self.run_binding.case_id,
                    "neatlogs.replay.trial": self.run_binding.trial,
                }
            )
        span.set_attributes(attributes)


def current_replay():
    scoped = _current.get() or otel_context.get_value(_context_key)
    session = scoped or _environment
    thread = threading.current_thread()
    if scoped is None and inspect.getattr_static(type(thread), "run") is not inspect.getattr_static(
        threading.Thread, "run"
    ):
        context = getattr(thread, "_otel_context", None)
        inherited = otel_context.get_value(_context_key, context) if context is not None else None
        session = session or inherited
        if session is not None:
            from opentelemetry import trace

            from .._wrap_utils import _current_neatlogs_parent

            parent = _current_neatlogs_parent(context) or trace.get_current_span(context)
            current = _current_neatlogs_parent() or trace.get_current_span()
            if not (
                parent.is_recording()
                and current.is_recording()
                and parent.get_span_context().trace_id == current.get_span_context().trace_id
            ):
                session._reject_thread_context()
    return session


def set_environment_replay(session):
    global _environment
    if _environment is not None:
        raise ReplayError("Environment replay is already configured.")
    _environment = session


def get_replay_status():
    session = current_replay()
    return session.get_status() if session is not None else None


def stamp_replay_span(span):
    session = current_replay()
    if session is not None:
        session.stamp(span)


def lookup_replay(kind, name, input, *, span=None, can_restore=None):
    session = current_replay()
    if session is None:
        return None
    decision = session.decide(kind, name, input, can_restore)
    if span is None:
        from opentelemetry.trace import get_current_span

        from .._wrap_utils import active_neatlogs_context

        context = active_neatlogs_context()
        span = get_current_span(context) if context is not None else None
    if span is not None:
        session.stamp(span)
        span.set_attribute("neatlogs.replay.outcome", decision.outcome)
        if isinstance(decision, (ReplayHit, ReplayErrorHit)):
            span.set_attribute("neatlogs.replay.source_span_id", decision.entry.span_id)
        if isinstance(decision, ReplayHit) and decision.overridden:
            span.set_attribute("neatlogs.replay.override_applied", True)
    return decision


def replay_model(name, input, decode, live, *, span=None):
    decision = lookup_replay("model", name, input, span=span)
    if isinstance(decision, ReplayErrorHit):
        err = decision.entry.error or {}
        raise ReplayRecordedError(err.get("type", "RecordedError"), err.get("message", ""))
    return decode(decision.entry.output) if isinstance(decision, ReplayHit) else live()
