"""
PII masking support for Neatlogs spans.

Users supply a callable that receives the full span dict and returns
the (possibly modified) span dict. The callable is responsible for
traversing and redacting any sensitive fields.

Example::

    def redact(span: dict) -> dict:
        attrs = span.get("attributes", {})
        for key in list(attrs):
            if "email" in key or "phone" in key:
                attrs[key] = "***"
        return span

    neatlogs.init(mask=redact)
"""

import logging
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

# Module-level registry: str(id(fn)) -> callable
# Entries are permanent for the lifetime of the callable object; no cleanup needed.
_MASK_REGISTRY: Dict[str, Callable[[Dict[str, Any]], Optional[Dict[str, Any]]]] = {}


def register_mask(fn: Callable) -> str:
    """Register a mask callable and return its lookup key."""
    key = str(id(fn))
    _MASK_REGISTRY[key] = fn
    return key


def apply_mask(
    span_data: Dict[str, Any],
    global_mask: Optional[Callable],
) -> Dict[str, Any]:
    """Apply the effective mask callable to *span_data*.

    Per-span mask (stored in ``attributes["neatlogs.mask_id"]``) takes
    precedence over the global mask.  Returns the (possibly modified) dict.
    """
    mask_id = (span_data.get("attributes") or {}).get("neatlogs.mask_id")
    mask_fn: Optional[Callable] = None

    if mask_id:
        mask_fn = _MASK_REGISTRY.get(str(mask_id))

    if mask_fn is None:
        mask_fn = global_mask

    if mask_fn is None:
        return span_data

    try:
        result = mask_fn(span_data)
        return result if result is not None else span_data
    except Exception as exc:
        logger.warning(
            "[neatlogs] mask callable raised an exception for span '%s': %s — "
            "original span data will be exported unchanged.",
            span_data.get("name"),
            exc,
        )
        return span_data
