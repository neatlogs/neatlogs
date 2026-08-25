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

from typing import Any, Callable, Dict, Optional

# Module-level registry: str(id(fn)) -> callable
# Entries are permanent for the lifetime of the callable object; no cleanup needed.
_MASK_REGISTRY: Dict[str, Callable[[Dict[str, Any]], Optional[Dict[str, Any]]]] = {}


def register_mask(fn: Callable) -> str:
    """Register a mask callable and return its lookup key."""
    key = str(id(fn))
    _MASK_REGISTRY[key] = fn
    return key


def effective_mask(span_data: Dict[str, Any], global_mask: Optional[Callable]):
    """Return the registered per-span mask, otherwise the global mask."""
    mask_id = (span_data.get("attributes") or {}).get("neatlogs.mask_id")
    if mask_id:
        registered = _MASK_REGISTRY.get(str(mask_id))
        if registered is not None:
            return registered
    return global_mask
