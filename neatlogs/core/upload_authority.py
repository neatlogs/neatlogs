"""Injectable boundary for the not-yet-deployed authenticated upload contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .capture import UPLOAD_UNAVAILABLE_REASON


@dataclass(frozen=True)
class OverflowPayload:
    """A complete, masked OTLP item offered to an upload implementation."""

    content: bytes
    sha256: str
    byte_length: int
    signal: str
    purpose: str = "otlp_overflow"
    mime_type: str = "application/x-protobuf"
    encoding: str = "identity"


class UploadAuthority(Protocol):
    """Internal seam; no backend URL or authentication contract is assumed."""

    available: bool
    unavailable_reason: str

    def export_overflow(self, payload: OverflowPayload) -> bool:
        """Upload bytes and export the resulting small canonical reference."""


class DisabledUploadAuthority:
    """Production default until the authenticated backend contract is deployed."""

    available = False
    unavailable_reason = UPLOAD_UNAVAILABLE_REASON

    def export_overflow(self, payload: OverflowPayload) -> bool:
        del payload
        return False
