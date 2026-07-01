"""
Request-scoped session & end-user identity.

Session and end-user identity are PER-REQUEST, not process-global, so they are
never set on ``init()``. The primary way to set them is at the trace root
(``trace(session_id=...)`` / ``@span(session_id=...)``). For code that only uses
``wrap()`` — where the trace root is created internally by the wrapper and the
caller has no root to put arguments on — use the ``identify()`` context manager:

    with neatlogs.identify(session_id="chat_123", end_user_id="user_456",
                           end_user_metadata={"plan": "pro"}):
        wrapped_client.chat.completions.create(...)   # auto-root reads this

Resolution at the trace root (per field): explicit per-call argument wins, then
the ``identify()`` context, then nothing. Identity is stamped on the ROOT span
only; the backend rolls it up to the trace and its session.

Backed by ``contextvars`` so it propagates across async tasks (and threads via
``contextvars.copy_context()``), exactly like OpenTelemetry's own context.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Dict, Iterator, Optional

_session_id: ContextVar[Optional[str]] = ContextVar("neatlogs.session_id", default=None)
_end_user_id: ContextVar[Optional[str]] = ContextVar("neatlogs.end_user_id", default=None)
_end_user_metadata: ContextVar[Optional[Dict[str, Any]]] = ContextVar(
    "neatlogs.end_user_metadata", default=None
)


def current_session_id() -> Optional[str]:
    return _session_id.get()


def current_end_user_id() -> Optional[str]:
    return _end_user_id.get()


def current_end_user_metadata() -> Optional[Dict[str, Any]]:
    return _end_user_metadata.get()


@contextmanager
def identify(
    session_id: Optional[str] = None,
    end_user_id: Optional[str] = None,
    end_user_metadata: Optional[Dict[str, Any]] = None,
) -> Iterator[None]:
    """Bind session / end-user identity for the enclosed block.

    Only non-None values are set, so nested ``identify()`` calls override one
    field without clearing the others. Restores the previous values on exit.
    """
    tokens = []
    if session_id is not None:
        tokens.append((_session_id, _session_id.set(session_id)))
    if end_user_id is not None:
        tokens.append((_end_user_id, _end_user_id.set(end_user_id)))
    if end_user_metadata is not None:
        tokens.append((_end_user_metadata, _end_user_metadata.set(end_user_metadata)))
    try:
        yield
    finally:
        for var, token in reversed(tokens):
            var.reset(token)
