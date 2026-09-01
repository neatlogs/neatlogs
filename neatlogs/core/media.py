"""Typed multimodal discovery without fetching remote content or truncating data."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import mimetypes
import re
import threading
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from .capture import DEFAULT_MAX_CAPTURE_VALUE_BYTES, UPLOAD_UNAVAILABLE_REASON
from .upload_authority import MediaPayload

_MEDIA_KINDS = {"image", "audio", "video", "document", "file", "input_file"}
_MESSAGE_CONTENT = re.compile(r"^neatlogs\.llm\.(input|output)_messages\.(\d+)\.content$")
_MIME_ALIASES = {
    "audio/mp3": "audio/mpeg",
    "audio/x-flac": "audio/flac",
    "audio/x-wav": "audio/wav",
    "image/jpg": "image/jpeg",
    "image/pjpeg": "image/jpeg",
}
_FORMAT_MIME_TYPES = {
    ("audio", "flac"): "audio/flac",
    ("audio", "mp3"): "audio/mpeg",
    ("audio", "mpeg"): "audio/mpeg",
    ("audio", "ogg"): "audio/ogg",
    ("audio", "wav"): "audio/wav",
    ("document", "pdf"): "application/pdf",
    ("file", "pdf"): "application/pdf",
    ("image", "gif"): "image/gif",
    ("image", "jpeg"): "image/jpeg",
    ("image", "jpg"): "image/jpeg",
    ("image", "png"): "image/png",
    ("image", "webp"): "image/webp",
}


def _sniff_media_mime(content: bytes | bytearray) -> str:
    """Match the backend's bounded typed-media signature detection."""

    prefix = bytes(content[:12])
    if prefix.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if prefix.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if prefix.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if prefix.startswith(b"RIFF") and prefix[8:12] == b"WEBP":
        return "image/webp"
    if prefix.startswith(b"%PDF-"):
        return "application/pdf"
    if prefix.startswith(b"fLaC"):
        return "audio/flac"
    if prefix.startswith(b"OggS"):
        return "audio/ogg"
    if prefix.startswith(b"RIFF") and prefix[8:12] == b"WAVE":
        return "audio/wav"
    if prefix.startswith(b"ID3") or (
        len(prefix) >= 2 and prefix[0] == 0xFF and prefix[1] & 0xE0 == 0xE0
    ):
        return "audio/mpeg"
    return ""


def normalize_media_mime(
    mime_type: str,
    declared: str = "",
    format_name: str = "",
    content: bytes | bytearray = b"",
) -> str:
    """Return the backend-contract MIME spelling, preferring byte signatures."""

    value = str(mime_type or "").strip().lower().split(";", 1)[0]
    sniffed = _sniff_media_mime(content) if content else ""
    if sniffed:
        return sniffed
    if value:
        return _MIME_ALIASES.get(value, value)
    kind = str(declared or "").lower().replace("input_", "")
    fmt = str(format_name or "").strip().lower().lstrip(".")
    return _FORMAT_MIME_TYPES.get((kind, fmt), "")


def _model_mapping(value: Any) -> Mapping[str, Any] | None:
    """Read provider model data through documented mapping serializers only."""

    for name in ("model_dump", "to_json_dict", "dict"):
        method = getattr(value, name, None)
        if not callable(method):
            continue
        try:
            candidate = method(exclude_none=True) if name == "model_dump" else method()
        except (TypeError, ValueError):
            continue
        if isinstance(candidate, Mapping):
            return candidate
    return None


@dataclass(frozen=True)
class PendingMedia:
    token: str
    payload: MediaPayload


class PendingMediaStore:
    """Bounded in-memory handoff from provider capture to post-mask export."""

    def __init__(self, *, max_bytes: int, max_items: int = 32) -> None:
        if max_bytes <= 0 or max_items <= 0:
            raise ValueError("pending media bounds must be greater than zero")
        self.max_bytes = max_bytes
        self.max_items = max_items
        self._items: OrderedDict[str, MediaPayload] = OrderedDict()
        self._references: dict[str, int] = {}
        self._retained_bytes = 0
        self._lock = threading.RLock()

    def stage(self, content: bytes, mime_type: str, media_purpose: str) -> str | None:
        if not content or len(content) > self.max_bytes:
            return None
        digest = hashlib.sha256(content).hexdigest()
        mime_type = normalize_media_mime(mime_type, content=content)
        identity = hashlib.sha256(f"{digest}:{mime_type}".encode("utf-8")).hexdigest()[:24]
        token = f"nl_pending_media_{identity}"
        payload = MediaPayload(
            content=content,
            sha256=digest,
            byte_length=len(content),
            mime_type=mime_type or "application/octet-stream",
            media_purpose=media_purpose,
        )
        with self._lock:
            if token in self._items:
                self._items.move_to_end(token)
                self._references[token] += 1
                return token
            if (
                len(self._items) >= self.max_items
                or self._retained_bytes + len(content) > self.max_bytes
            ):
                return None
            self._items[token] = payload
            self._references[token] = 1
            self._retained_bytes += len(content)
        return token

    def get(self, token: str) -> MediaPayload | None:
        with self._lock:
            return self._items.get(token)

    def retain(self, token: str, count: int = 1) -> bool:
        if count <= 0:
            return False
        with self._lock:
            if token not in self._items:
                return False
            self._references[token] += count
            self._items.move_to_end(token)
            return True

    def release(self, token: str, count: int = 1) -> None:
        if count <= 0:
            return
        with self._lock:
            references = self._references.get(token, 0)
            if references > count:
                self._references[token] = references - count
                return
            self._references.pop(token, None)
            payload = self._items.pop(token, None)
            if payload is not None:
                self._retained_bytes -= payload.byte_length

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
            self._references.clear()
            self._retained_bytes = 0

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {"items": len(self._items), "bytes": self._retained_bytes}


_default_media_store: PendingMediaStore | None = None
_default_media_store_lock = threading.RLock()


def set_default_media_store(store: PendingMediaStore | None) -> None:
    global _default_media_store
    with _default_media_store_lock:
        previous = _default_media_store
        _default_media_store = store
    if previous is not None and previous is not store:
        previous.clear()


def current_media_store() -> PendingMediaStore | None:
    # A secondary Client is context-scoped; its staging store must follow the
    # same routing decision as its private tracer/exporter pipeline.
    try:
        from .._wrap_utils import get_active_client

        client = get_active_client()
        if client is not None:
            return getattr(client, "_media_store", None)
    except Exception:
        pass
    with _default_media_store_lock:
        return _default_media_store


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


def _base64_metadata(encoded: str) -> tuple[int, str, bytes] | None:
    """Validate/digest base64 incrementally without duplicating the full media."""

    digest = hashlib.sha256()
    byte_length = 0
    prefix = b""
    chunk_size = 64 * 1024  # divisible by four, so quartets never cross chunks
    try:
        for offset in range(0, len(encoded), chunk_size):
            decoded = base64.b64decode(encoded[offset : offset + chunk_size], validate=True)
            if not prefix:
                prefix = decoded[:12]
            digest.update(decoded)
            byte_length += len(decoded)
    except (binascii.Error, ValueError, TypeError):
        return None
    return byte_length, digest.hexdigest(), prefix


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
    byte_length, digest, prefix = metadata
    mime_type = normalize_media_mime(mime_type, declared, content=prefix)
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
        store = current_media_store()
        token = None
        if store is not None and byte_length <= store.max_bytes:
            try:
                content = base64.b64decode(encoded, validate=True)
            except (binascii.Error, ValueError, TypeError):
                content = b""
            if content:
                token = store.stage(content, record["mime_type"], purpose)
        if token is not None:
            record.update(
                {
                    "state": "pending-upload",
                    "upload_token": token,
                    "safe_preview": "pending authenticated upload",
                }
            )
        else:
            record.update(
                {
                    "state": "failed",
                    "safe_preview": f"upload unavailable: {UPLOAD_UNAVAILABLE_REASON}",
                }
            )
    else:
        record["state"] = "inline"
    return record


def _sanitized_reference(reference: str) -> str:
    """Remove credentials and bearer material from a captured media locator."""

    try:
        parsed = urlsplit(reference)
    except ValueError:
        # Malformed bracketed hosts must still fail closed instead of falling
        # back to serializing their credential-bearing original value.
        safe = reference.split("#", 1)[0].split("?", 1)[0]
        if "://" in safe:
            scheme, remainder = safe.split("://", 1)
            safe = f"{scheme}://{remainder.rsplit('@', 1)[-1]}"
        return safe
    # ``urlsplit`` leaves opaque provider IDs in ``path``. Preserve those, but
    # never retain URL userinfo, query credentials, or fragments in telemetry.
    netloc = parsed.netloc.rsplit("@", 1)[-1]
    return urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))


def _reference(reference: str, mime_type: str, declared: str, purpose: str) -> dict[str, Any]:
    safe_reference = _sanitized_reference(reference)
    digest = hashlib.sha256(safe_reference.encode()).hexdigest()
    parsed = urlsplit(safe_reference)
    guessed = normalize_media_mime(mimetypes.guess_type(parsed.path)[0] or "", declared)
    mime_type = normalize_media_mime(mime_type, declared)
    source = "provider" if not parsed.scheme or parsed.scheme in {"gs", "s3"} else "url"
    return {
        "id": f"nl_media_{digest[:24]}",
        "type": _kind(mime_type or guessed, declared),
        "source": source,
        "mime_type": mime_type or guessed or "application/octet-stream",
        "reference": safe_reference,
        "purpose": purpose,
        "state": "available",
    }


def _unavailable_placeholder(record: Mapping[str, Any]) -> dict[str, Any]:
    """A secret-free replacement for inline bytes that cannot be uploaded."""

    return {
        "neatlogs_media": {
            key: record[key]
            for key in (
                "id",
                "type",
                "source",
                "mime_type",
                "byte_length",
                "sha256",
                "purpose",
                "state",
                "safe_preview",
                "upload_token",
            )
            if key in record
        }
    }


def sanitize_media_payload(value: Any, purpose: str = "capture") -> Any:
    """Clone provider-shaped media while removing unsafe inline/URL secrets.

    Small inline values remain available to ordinary capture. Large inline
    values are replaced by typed failure metadata because no authenticated
    upload authority exists yet. Remote locators retain only scheme/host/path.
    """

    def sanitize(
        node: Any,
        inherited_declared: str = "",
        inherited_mime_type: str = "",
    ) -> Any:
        if isinstance(node, Mapping):
            declared = str(node.get("type") or "")
            media_declared = declared if declared.lower() in _MEDIA_KINDS else inherited_declared
            mime_type = str(
                node.get("mime_type")
                or node.get("mimeType")
                or node.get("media_type")
                or node.get("mediaType")
                or inherited_mime_type
            )
            mime_type = normalize_media_mime(
                mime_type, media_declared, str(node.get("format") or "")
            )
            result = {}
            for key, item in node.items():
                keyed_kind = str(key).lower().replace("input_", "")
                child_declared = keyed_kind if keyed_kind in _MEDIA_KINDS else media_declared
                result[key] = sanitize(item, child_declared, mime_type)

            for key in ("image_url", "imageUrl"):
                image = node.get(key)
                if isinstance(image, Mapping) and isinstance(image.get("url"), str):
                    original = image["url"]
                    record = (
                        _inline(original, mime_type, "image", purpose)
                        if original.startswith("data:")
                        else _reference(original, mime_type, "image", purpose)
                    )
                    image_result = dict(result[key])
                    if record is not None and record.get("state") in {"failed", "pending-upload"}:
                        image_result["url"] = _unavailable_placeholder(record)
                    elif record is not None:
                        image_result["url"] = record.get("reference", original)
                    result[key] = image_result
                elif isinstance(image, str):
                    record = (
                        _inline(image, mime_type, "image", purpose)
                        if image.startswith("data:")
                        else _reference(image, mime_type, "image", purpose)
                    )
                    if record is not None and record.get("state") in {"failed", "pending-upload"}:
                        result[key] = _unavailable_placeholder(record)
                    elif record is not None:
                        result[key] = record.get("reference", image)

            for key in ("input_audio", "inputAudio"):
                audio = node.get(key)
                if isinstance(audio, Mapping) and isinstance(audio.get("data"), str):
                    fmt = str(audio.get("format") or "")
                    record = _inline(
                        audio["data"],
                        mime_type or (f"audio/{fmt}" if fmt else "audio/unknown"),
                        "audio",
                        purpose,
                    )
                    if record is not None and record.get("state") in {"failed", "pending-upload"}:
                        audio_result = dict(result[key])
                        audio_result["data"] = _unavailable_placeholder(record)
                        result[key] = audio_result

            for key in ("inline_data", "inlineData"):
                inline_data = node.get(key)
                if isinstance(inline_data, Mapping) and isinstance(inline_data.get("data"), str):
                    record = _inline(
                        inline_data["data"],
                        str(
                            inline_data.get("mime_type") or inline_data.get("mimeType") or mime_type
                        ),
                        declared,
                        purpose,
                    )
                    if record is not None and record.get("state") in {
                        "failed",
                        "pending-upload",
                    }:
                        inline_result = dict(result[key])
                        inline_result["data"] = _unavailable_placeholder(record)
                        result[key] = inline_result

            file_data = node.get("file_data") or node.get("fileData")
            if isinstance(file_data, str):
                record = _inline(file_data, mime_type, declared or "document", purpose)
                if record is not None and record.get("state") in {"failed", "pending-upload"}:
                    key = "file_data" if "file_data" in node else "fileData"
                    result[key] = _unavailable_placeholder(record)

            raw_key = "data" if "data" in node else "bytes" if "bytes" in node else None
            raw_data = node.get(raw_key) if raw_key is not None else None
            if isinstance(raw_data, str) and (media_declared or mime_type):
                record = _inline(raw_data, mime_type, media_declared, purpose)
                if record is not None and record.get("state") in {"failed", "pending-upload"}:
                    result[raw_key] = _unavailable_placeholder(record)
            if (
                isinstance(raw_data, (bytes, bytearray))
                and (media_declared or mime_type)
                and len(raw_data) > DEFAULT_MAX_CAPTURE_VALUE_BYTES
            ):
                digest = hashlib.sha256(raw_data).hexdigest()
                store = current_media_store()
                actual_mime = (
                    normalize_media_mime(mime_type, media_declared, content=raw_data)
                    or "application/octet-stream"
                )
                token = (
                    store.stage(
                        bytes(raw_data),
                        actual_mime,
                        purpose,
                    )
                    if store is not None and len(raw_data) <= store.max_bytes
                    else None
                )
                result[raw_key] = _unavailable_placeholder(
                    {
                        "id": f"nl_media_{digest[:24]}",
                        "type": _kind(actual_mime, media_declared),
                        "source": "inline",
                        "mime_type": actual_mime,
                        "byte_length": len(raw_data),
                        "sha256": digest,
                        "purpose": purpose,
                        "state": "pending-upload" if token else "failed",
                        "safe_preview": (
                            "pending authenticated upload"
                            if token
                            else f"upload unavailable: {UPLOAD_UNAVAILABLE_REASON}"
                        ),
                        **({"upload_token": token} if token else {}),
                    }
                )

            reference_key = next(
                (key for key in ("file_id", "file_uri", "fileUri", "url") if key in node),
                None,
            )
            if reference_key is not None:
                reference = node.get(reference_key)
                if isinstance(reference, str) and (
                    declared.lower() == "url" or media_declared or mime_type
                ):
                    result[reference_key] = _sanitized_reference(reference)
            return result
        if isinstance(node, Sequence) and not isinstance(node, (str, bytes, bytearray)):
            return [sanitize(item) for item in node]
        modeled = _model_mapping(node)
        if modeled is not None:
            return sanitize(modeled, inherited_declared, inherited_mime_type)
        return node

    return sanitize(value)


def media_references(value: Any, purpose: str) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []

    def visit(
        node: Any,
        inherited_declared: str = "",
        inherited_mime_type: str = "",
    ) -> None:
        if isinstance(node, Mapping):
            declared = str(node.get("type") or "")
            media_declared = declared if declared.lower() in _MEDIA_KINDS else inherited_declared
            mime_type = str(
                node.get("mime_type")
                or node.get("mimeType")
                or node.get("media_type")
                or node.get("mediaType")
                or inherited_mime_type
            )
            mime_type = normalize_media_mime(
                mime_type, media_declared, str(node.get("format") or "")
            )
            wrapped = node.get("neatlogs_media")
            if isinstance(wrapped, Mapping):
                found.append(dict(wrapped) | {"purpose": purpose})
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
            raw_key = "data" if "data" in node else "bytes" if "bytes" in node else None
            raw_data = node.get(raw_key) if raw_key is not None else None
            if isinstance(raw_data, str) and (media_declared or mime_type):
                record = _inline(raw_data, mime_type, media_declared, purpose)
                if record:
                    found.append(record)
            if isinstance(raw_data, (bytes, bytearray)) and (media_declared or mime_type):
                mime_type = normalize_media_mime(
                    mime_type,
                    media_declared,
                    content=raw_data,
                )
                digest = hashlib.sha256(raw_data).hexdigest()
                store = current_media_store()
                token = (
                    store.stage(
                        bytes(raw_data),
                        mime_type or "application/octet-stream",
                        purpose,
                    )
                    if (
                        store is not None
                        and len(raw_data) > DEFAULT_MAX_CAPTURE_VALUE_BYTES
                        and len(raw_data) <= store.max_bytes
                    )
                    else None
                )
                found.append(
                    {
                        "id": f"nl_media_{digest[:24]}",
                        "type": _kind(mime_type, media_declared),
                        "source": "inline",
                        "mime_type": mime_type or "application/octet-stream",
                        "byte_length": len(raw_data),
                        "sha256": digest,
                        "purpose": purpose,
                        "state": (
                            "inline"
                            if len(raw_data) <= DEFAULT_MAX_CAPTURE_VALUE_BYTES
                            else "pending-upload" if token else "failed"
                        ),
                        **(
                            {}
                            if len(raw_data) <= DEFAULT_MAX_CAPTURE_VALUE_BYTES
                            else {
                                "safe_preview": (
                                    "pending authenticated upload"
                                    if token
                                    else f"upload unavailable: {UPLOAD_UNAVAILABLE_REASON}"
                                ),
                                **({"upload_token": token} if token else {}),
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
            if isinstance(reference, str) and (
                declared.lower() == "url" or media_declared or mime_type
            ):
                found.append(
                    _reference(reference, mime_type, media_declared or "document", purpose)
                )
            for key, child in node.items():
                if key == "neatlogs_media":
                    continue
                keyed_kind = str(key).lower().replace("input_", "")
                child_declared = keyed_kind if keyed_kind in _MEDIA_KINDS else media_declared
                visit(child, child_declared, mime_type)
        elif isinstance(node, Sequence) and not isinstance(node, (str, bytes, bytearray)):
            for child in node:
                visit(child, inherited_declared, inherited_mime_type)
        else:
            modeled = _model_mapping(node)
            if modeled is not None:
                visit(modeled, inherited_declared, inherited_mime_type)

    visit(value)
    unique: dict[tuple[Any, ...], dict[str, Any]] = {}
    for record in found:
        if record is not None:
            key = (record.get("sha256"), record.get("reference"), record.get("type"))
            if key in unique:
                token = record.get("upload_token")
                store = current_media_store()
                if isinstance(token, str) and store is not None:
                    store.release(token)
                continue
            unique[key] = record
    return list(unique.values())


def set_media_attributes(span: Any, prefix: str, value: Any, purpose: str) -> None:
    for index, record in enumerate(media_references(value, purpose)):
        for key, item in record.items():
            span.set_attribute(f"{prefix}.media.{index}.{key}", item)


def promote_message_media_attributes(attributes: Mapping[str, Any]) -> dict[str, Any]:
    """Promote provider-shaped message content into canonical media attributes.

    This is a shared fallback for existing adapters. Adapters may still emit the
    fields directly; direct fields always win.
    """

    result = {str(key): value for key, value in attributes.items()}
    for key, value in list(result.items()):
        match = _MESSAGE_CONTENT.fullmatch(key)
        if match is None:
            continue
        direction, index = match.groups()
        prefix = f"neatlogs.llm.{direction}_messages.{index}"
        if any(existing.startswith(f"{prefix}.media.") for existing in result):
            continue
        parsed = value
        if isinstance(value, str):
            stripped = value.lstrip()
            if not stripped.startswith(("{", "[")):
                continue
            try:
                parsed = json.loads(value)
            except (TypeError, ValueError):
                continue
        records = media_references(parsed, "input" if direction == "input" else "output")
        for media_index, record in enumerate(records):
            token = record.get("upload_token")
            store = current_media_store()
            if isinstance(token, str) and store is not None:
                store.retain(token)
            for field, item in record.items():
                result[f"{prefix}.media.{media_index}.{field}"] = item
    return result
