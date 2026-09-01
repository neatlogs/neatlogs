"""Typed multimodal discovery without fetching remote content or truncating data."""

from __future__ import annotations

import base64
import binascii
import hashlib
import mimetypes
from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import urlparse

from .capture import DEFAULT_MAX_CAPTURE_VALUE_BYTES, UPLOAD_UNAVAILABLE_REASON


def _kind(mime_type: str, declared: str = "") -> str:
    value = declared.lower().replace("input_", "")
    if value in {"image", "audio", "video", "document"}:
        return value
    if mime_type.startswith("image/"):
        return "image"
    if mime_type.startswith("audio/"):
        return "audio"
    if mime_type.startswith("video/"):
        return "video"
    if mime_type == "application/pdf" or mime_type.startswith("text/"):
        return "document"
    return "media"


def _base64_metadata(encoded: str) -> tuple[int, str] | None:
    """Validate/digest base64 incrementally without duplicating the full media."""

    digest = hashlib.sha256()
    byte_length = 0
    chunk_size = 64 * 1024  # divisible by four, so quartets never cross chunks
    try:
        for offset in range(0, len(encoded), chunk_size):
            decoded = base64.b64decode(encoded[offset : offset + chunk_size], validate=True)
            digest.update(decoded)
            byte_length += len(decoded)
    except (binascii.Error, ValueError, TypeError):
        return None
    return byte_length, digest.hexdigest()


def _inline(data: str, mime_type: str, declared: str, purpose: str) -> dict[str, Any] | None:
    encoded = data
    if data.startswith("data:"):
        header, separator, encoded = data.partition(",")
        if not separator:
            return None
        mime_type = header[5:].split(";", 1)[0] or mime_type
    metadata = _base64_metadata(encoded)
    if metadata is None:
        return None
    byte_length, digest = metadata
    record = {
        "id": f"nl_media_{digest[:24]}",
        "type": _kind(mime_type, declared),
        "source": "inline",
        "mime_type": mime_type or "application/octet-stream",
        "byte_length": byte_length,
        "sha256": digest,
        "purpose": purpose,
    }
    if len(data.encode("utf-8")) > DEFAULT_MAX_CAPTURE_VALUE_BYTES:
        record.update(
            {
                "state": "failed",
                "safe_preview": f"upload unavailable: {UPLOAD_UNAVAILABLE_REASON}",
            }
        )
    else:
        record["state"] = "inline"
    return record


def _reference(reference: str, mime_type: str, declared: str, purpose: str) -> dict[str, Any]:
    digest = hashlib.sha256(reference.encode()).hexdigest()
    parsed = urlparse(reference)
    guessed = mimetypes.guess_type(parsed.path)[0] or ""
    source = "provider" if not parsed.scheme or parsed.scheme in {"gs", "s3"} else "url"
    return {
        "id": f"nl_media_{digest[:24]}",
        "type": _kind(mime_type or guessed, declared),
        "source": source,
        "mime_type": mime_type or guessed or "application/octet-stream",
        "reference": reference,
        "purpose": purpose,
        "state": "available",
    }


def media_references(value: Any, purpose: str) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []

    def visit(node: Any) -> None:
        if isinstance(node, Mapping):
            declared = str(node.get("type") or "")
            mime_type = str(node.get("mime_type") or node.get("mimeType") or "")
            image = node.get("image_url") or node.get("imageUrl")
            if isinstance(image, Mapping):
                image = image.get("url")
            if isinstance(image, str):
                record = (
                    _inline(image, mime_type, "image", purpose)
                    if image.startswith("data:")
                    else _reference(image, mime_type, "image", purpose)
                )
                if record:
                    found.append(record)
            audio = node.get("input_audio") or node.get("inputAudio")
            if isinstance(audio, Mapping) and isinstance(audio.get("data"), str):
                fmt = str(audio.get("format") or "")
                record = _inline(
                    audio["data"],
                    mime_type or (f"audio/{fmt}" if fmt else "audio/unknown"),
                    "audio",
                    purpose,
                )
                if record:
                    found.append(record)
            inline_data = node.get("inline_data") or node.get("inlineData")
            if isinstance(inline_data, Mapping) and isinstance(inline_data.get("data"), str):
                record = _inline(
                    inline_data["data"],
                    str(inline_data.get("mime_type") or inline_data.get("mimeType") or mime_type),
                    declared,
                    purpose,
                )
                if record:
                    found.append(record)
            file_data = node.get("file_data") or node.get("fileData")
            if isinstance(file_data, str):
                record = _inline(file_data, mime_type, declared or "document", purpose)
                if record:
                    found.append(record)
            raw_data = node.get("data")
            if isinstance(raw_data, (bytes, bytearray)) and (declared or mime_type):
                digest = hashlib.sha256(raw_data).hexdigest()
                found.append(
                    {
                        "id": f"nl_media_{digest[:24]}",
                        "type": _kind(mime_type, declared),
                        "source": "inline",
                        "mime_type": mime_type or "application/octet-stream",
                        "byte_length": len(raw_data),
                        "sha256": digest,
                        "purpose": purpose,
                        "state": (
                            "inline"
                            if len(raw_data) <= DEFAULT_MAX_CAPTURE_VALUE_BYTES
                            else "failed"
                        ),
                        **(
                            {}
                            if len(raw_data) <= DEFAULT_MAX_CAPTURE_VALUE_BYTES
                            else {
                                "safe_preview": (f"upload unavailable: {UPLOAD_UNAVAILABLE_REASON}")
                            }
                        ),
                    }
                )
            reference = (
                node.get("file_id")
                or node.get("file_uri")
                or node.get("fileUri")
                or node.get("url")
            )
            if isinstance(reference, str) and (declared in {"file", "input_file"} or mime_type):
                found.append(_reference(reference, mime_type, declared or "document", purpose))
            for child in node.values():
                visit(child)
        elif isinstance(node, Sequence) and not isinstance(node, (str, bytes, bytearray)):
            for child in node:
                visit(child)

    visit(value)
    unique: dict[tuple[Any, ...], dict[str, Any]] = {}
    for record in found:
        if record is not None:
            key = (record.get("sha256"), record.get("reference"), record.get("type"))
            unique[key] = record
    return list(unique.values())


def set_media_attributes(span: Any, prefix: str, value: Any, purpose: str) -> None:
    for index, record in enumerate(media_references(value, purpose)):
        for key, item in record.items():
            span.set_attribute(f"{prefix}.media.{index}.{key}", item)
