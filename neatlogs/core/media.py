"""Typed multimodal discovery without fetching remote content or truncating data."""

from __future__ import annotations

import base64
import hashlib
import mimetypes
from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import urlparse


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


def _inline(data: str, mime_type: str, declared: str, purpose: str) -> dict[str, Any] | None:
    encoded = data
    if data.startswith("data:"):
        header, separator, encoded = data.partition(",")
        if not separator:
            return None
        mime_type = header[5:].split(";", 1)[0] or mime_type
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, TypeError):
        return None
    digest = hashlib.sha256(raw).hexdigest()
    return {
        "id": f"nl_media_{digest[:24]}",
        "type": _kind(mime_type, declared),
        "source": "inline",
        "mime_type": mime_type or "application/octet-stream",
        "byte_length": len(raw),
        "sha256": digest,
        "purpose": purpose,
        "state": "inline",
    }


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
            file_data = node.get("file_data") or node.get("fileData")
            if isinstance(file_data, str):
                record = _inline(file_data, mime_type, declared or "document", purpose)
                if record:
                    found.append(record)
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
