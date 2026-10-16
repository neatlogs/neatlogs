"""
Did an OpenAI Agents SDK run fail? Decided at the moment the SDK ends the run's trace.

Used by both Agents integrations: neatlogs' own processor (``openai_agents_processor()``) and
the OpenInference processor that ``init(instrumentations=["openai_agents"])`` installs.

Errors recorded on spans can't answer this. A tool error, a failed agent-as-tool, an extra
handoff and a ``max_turns`` caught by ``error_handlers`` all record an error, yet the run
finishes normally. So we look at what the SDK itself does:

- ``Runner.run`` ends the trace in ``TraceCtxManager.__exit__``, and a streamed run ends it in
  an ``except`` block, so a run that raises has its exception active (``sys.exc_info()``) when
  ``on_trace_end`` runs. An exception that was already active when the trace started belongs
  to the caller (the run was started inside their ``except``), so it's ignored.
- A streamed run that hits ``max_turns`` without a handler doesn't raise there: the loop stops,
  ends the trace and ``stream_events()`` raises afterwards. For that one case we read the
  streamed result the SDK will raise from, with the same test it uses (``_check_errors``).
"""

import sys
import weakref
from typing import Any, Optional

# trace_id -> RunResultStreaming, for streamed runs that created their own trace.
_streamed_results: "weakref.WeakValueDictionary[str, Any]" = weakref.WeakValueDictionary()


def active_exception() -> Optional[BaseException]:
    return sys.exc_info()[1]


def run_failure(trace: Any, start_exception: Optional[BaseException]) -> Optional[str]:
    """Return why the run failed, or None if it finished normally."""
    exc = active_exception()
    if exc is not None and exc is not start_exception:
        return f"{type(exc).__name__}: {exc}"
    result = _streamed_results.get(str(getattr(trace, "trace_id", "") or ""))
    if result is not None:
        stored = getattr(result, "_stored_exception", None)
        if stored is not None:
            return f"{type(stored).__name__}: {stored}"
        max_turns = getattr(result, "max_turns", None)
        if (
            max_turns is not None
            and getattr(result, "current_turn", 0) > max_turns
            and not getattr(result, "_max_turns_handled", False)
        ):
            return f"MaxTurnsExceeded: Max turns ({max_turns}) exceeded"
    return None


def link_streamed_results() -> None:
    """Remember each streamed run's result by trace id, so run_failure() can read it."""
    try:
        from agents.run import AgentRunner
    except Exception:
        return
    if getattr(AgentRunner, "_NEATLOGS_LINKED_STREAMED_RESULTS", False):
        return
    orig = AgentRunner.run_streamed

    def _run_streamed(self, *args: Any, **kwargs: Any):
        result = orig(self, *args, **kwargs)
        try:
            trace = getattr(result, "trace", None)
            trace_id = getattr(trace, "trace_id", None)
            if trace_id:
                _streamed_results[str(trace_id)] = result
        except Exception:
            pass
        return result

    AgentRunner.run_streamed = _run_streamed
    AgentRunner._NEATLOGS_LINKED_STREAMED_RESULTS = True
