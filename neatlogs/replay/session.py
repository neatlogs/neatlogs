"""Request-scoped FIFO replay with a one-way transition to live execution."""

from __future__ import annotations

import copy
import inspect
import threading
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Literal

from .contracts import (
    ReplayEntry,
    ReplayError,
    ReplayRunBinding,
    parse_replay_recording,
    replay_key,
)

_current = ContextVar("neatlogs.local_replay", default=None)
_environment = None


@dataclass(frozen=True)
class ReplayHit:
    entry: ReplayEntry
    outcome: Literal["HIT"] = "HIT"


@dataclass(frozen=True)
class ReplayLive:
    outcome: Literal["MISS", "LIVE"]


class ReplaySession:
    def __init__(self, recording, *, mode="tools_only", until_span=None, run=None):
        self.recording = parse_replay_recording(recording)
        if mode not in ("tools_only", "until_span"):
            raise ReplayError("Unsupported local replay mode.")
        if mode == "tools_only" and until_span is not None:
            raise ReplayError("tools_only does not accept a checkpoint.")
        identities = [entry.span_id for entry in self.recording.entries]
        if mode == "until_span" and until_span not in identities:
            raise ReplayError("The replay checkpoint must exist in this recording.")
        self.mode = mode
        self.until_span = until_span
        self.run_binding = ReplayRunBinding.parse(run) if run is not None else None
        self._checkpoint = identities.index(until_span) if until_span is not None else -1
        self._entries = {}
        self._positions = {}
        for index, entry in enumerate(self.recording.entries):
            self._entries.setdefault(replay_key(entry.kind, entry.name, entry.input), []).append(
                (index, entry)
            )
        self._lock = threading.RLock()
        self._used = False
        self._token = None
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

    def __enter__(self):
        if current_replay() is not None:
            raise ReplayError("Nested local replay sessions are not supported.")
        with self._lock:
            if self._used:
                raise ReplayError("Create a new ReplaySession for each local run.")
            self._used = True
        self._token = _current.set(self)
        return self

    def __exit__(self, *_):
        _current.reset(self._token)
        self._token = None

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
            return copy.deepcopy(self._status)

    def _miss(self, kind, name):
        self._status.update(
            switchedToLive=True, reason="missing_recording", firstMiss={"kind": kind, "name": name}
        )
        self._status["liveCalls"] += 1
        return ReplayLive("MISS")

    def decide(self, kind, name, input, can_restore=None):
        with self._lock:
            if self._status["switchedToLive"] or self.mode == "tools_only" and kind == "model":
                self._status["liveCalls"] += 1
                return ReplayLive("LIVE")
            try:
                key = replay_key(kind, name, input)
            except ReplayError:
                return self._miss(kind, name)
            position = self._positions.get(key, 0)
            sequence = self._entries.get(key, [])
            if position >= len(sequence):
                return self._miss(kind, name)
            index, entry = sequence[position]
            if entry.span_id == self.until_span:
                self._status.update(switchedToLive=True, reason="checkpoint")
                self._status["liveCalls"] += 1
                return ReplayLive("LIVE")
            if self.until_span is not None and index > self._checkpoint:
                return self._miss(kind, name)
            if can_restore is not None and not can_restore(entry.output):
                return self._miss(kind, name)
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
        attributes = {
            "neatlogs.origin": "simulation",
            "neatlogs.replay.source_trace_id": self.recording.source_trace_id,
            "neatlogs.replay.mode": self.mode,
        }
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
    return _current.get() or _environment


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
        if isinstance(decision, ReplayHit):
            span.set_attribute("neatlogs.replay.source_span_id", decision.entry.span_id)
    return decision


def replay_model(name, input, decode, live, *, span=None):
    decision = lookup_replay("model", name, input, span=span)
    return decode(decision.entry.output) if isinstance(decision, ReplayHit) else live()
