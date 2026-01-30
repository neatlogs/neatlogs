"""
Span kind definitions and mapping between OpenInference and OpenLLMetry conventions.
"""

from .mapping import (
    OPENINFERENCE_TO_TRACELOOP,
    TRACELOOP_TO_OPENINFERENCE,
    infer_span_kind_from_name,
)

__all__ = [
    "OPENINFERENCE_TO_TRACELOOP",
    "TRACELOOP_TO_OPENINFERENCE",
    "infer_span_kind_from_name",
]
