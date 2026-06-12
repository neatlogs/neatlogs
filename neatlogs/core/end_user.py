"""
End-user identity.

The *end user* is the user of OUR CUSTOMER'S application — the person interacting
with the AI product our customer built. This is deliberately distinct from the
operator-level ``user_id`` set in :func:`neatlogs.init`, which identifies whoever
is running the SDK (a developer, a service account, the OS user in Cloud Code).

Model: **one end-user per trace.** Identity is declared once at a trace boundary —
``init()`` (process-global default), ``trace()`` (per-request), or ``@span`` (per
WORKFLOW root) — via new keyword arguments on those existing constructs. There is
no separate ``identify()`` call and no nested per-span override: a trace belongs
to a single end-user. The SDK only stamps the declaring span; the backend rolls
the value up to the trace (and its session) so filtering/analytics are trace- and
session-level, not per-span.

Canonical span attributes:
    ``neatlogs.end_user.id``        — the end-user identifier (string)
    ``neatlogs.end_user.metadata``  — JSON object of arbitrary end-user fields
"""

import json
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

END_USER_ID_KEY = "neatlogs.end_user.id"
END_USER_METADATA_KEY = "neatlogs.end_user.metadata"


def normalize_metadata(metadata: Optional[Any]) -> Optional[str]:
    """Coerce end-user metadata to a JSON string. Returns None when empty."""
    if metadata is None:
        return None
    if isinstance(metadata, str):
        return metadata or None
    try:
        return json.dumps(metadata, default=str)
    except Exception:
        return json.dumps(str(metadata))


def apply_end_user_attributes(
    span: Any,
    end_user_id: Optional[str] = None,
    end_user_metadata: Optional[Any] = None,
    is_root: bool = True,
) -> None:
    """Stamp end-user id/metadata onto a ROOT span (best-effort).

    End-user identity belongs to the trace as a whole, so it is only stamped on
    the trace's root span (any span kind, created via ``trace()`` / ``@span``).
    When ``is_root`` is False the span is a child and the value is ignored — the
    backend reads end-user from the root span and rolls it up to the trace and
    session. ``end_user_id`` is set as-is; ``end_user_metadata`` is JSON-encoded.
    """
    if not (end_user_id or end_user_metadata):
        return
    if not is_root:
        logger.debug(
            "[end_user] Ignoring end_user_id/metadata on a non-root span — "
            "declare it on the trace root (top-level trace()/@span) or init()."
        )
        return
    try:
        if end_user_id:
            span.set_attribute(END_USER_ID_KEY, str(end_user_id))
        meta_json = normalize_metadata(end_user_metadata)
        if meta_json:
            span.set_attribute(END_USER_METADATA_KEY, meta_json)
    except Exception:
        pass
