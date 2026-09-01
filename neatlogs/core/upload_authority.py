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


@dataclass(frozen=True)
class OverflowExportReceipt:
    """Proof that backend validation completed and its small reference exported."""

    upload_id: str
    project_id: str
    state: str
    reference_exported: bool

    @property
    def complete(self) -> bool:
        return bool(
            self.upload_id and self.project_id and self.state == "ready" and self.reference_exported
        )


class UploadAuthority(Protocol):
    """Internal seam; no backend URL or authentication contract is assumed."""

    available: bool
    unavailable_reason: str

    def export_overflow(self, payload: OverflowPayload) -> OverflowExportReceipt | None:
        """Upload, validate, and export a small reference, returning its receipt."""


class DisabledUploadAuthority:
    """Production default until the authenticated backend contract is deployed."""

    available = False
    unavailable_reason = UPLOAD_UNAVAILABLE_REASON

    def export_overflow(self, payload: OverflowPayload) -> None:
        del payload
        return None
