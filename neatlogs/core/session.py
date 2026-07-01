"""
Session identity.

A *session* is one conversation / thread / journey — it groups the multiple
traces of a multi-turn interaction (one turn = one trace). Identity is declared
once at a trace boundary — ``init()`` (process-global default), ``trace()``
(per-turn), or ``@span`` (per WORKFLOW root) — via the ``session_id`` keyword on
those existing constructs. The SDK stamps only the trace's root span; the backend
rolls the value up to group the traces of a session together.

For single-turn workflows a session maps 1:1 to a trace: when no session is
declared the backend backfills ``session_id = trace_id``.

Canonical span attribute:
    ``neatlogs.session.id`` — the session identifier (string)
"""

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

SESSION_ID_KEY = "neatlogs.session.id"


def apply_session_attributes(
    span: Any,
    session_id: Optional[str] = None,
    is_root: bool = True,
) -> None:
    """Stamp the session id onto a ROOT span (best-effort).

    Session identity belongs to the trace as a whole, so it is only stamped on
    the trace's root span (any span kind, created via ``trace()`` / ``@span``).
    When ``is_root`` is False the span is a child and the value is ignored — the
    backend reads session from the root span and groups traces by it.

    Falls back to the ``identify()`` context when no explicit ``session_id`` is
    given — this is how wrapper-only code (whose root is an auto-root) gets a
    session without a user-controlled root.
    """
    if not session_id and is_root:
        from .identity import current_session_id

        session_id = current_session_id()
    if not session_id:
        return
    if not is_root:
        logger.debug(
            "[session] Ignoring session_id on a non-root span — declare it on "
            "the trace root (top-level trace()/@span) or init()."
        )
        return
    try:
        span.set_attribute(SESSION_ID_KEY, str(session_id))
    except Exception:
        pass
