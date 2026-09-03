from __future__ import annotations

import re
import threading
import time as _time
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)
from urllib.parse import quote

import requests

from ..core.logger import get_logger

logger = get_logger()

DEFAULT_CACHE_TTL_SECONDS = 60
DEFAULT_MAX_CACHE_ENTRIES = 100
DEFAULT_MAX_REFRESH_WORKERS = 4
DEFAULT_CONNECT_TIMEOUT = 2.0
DEFAULT_READ_TIMEOUT = 5.0

_PLACEHOLDER_PATTERN = re.compile(r"\{\{\s*([A-Za-z0-9_.-]+)\s*\}\}")
_LABEL_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,50}$")
CacheKey = Tuple[str, Optional[str], Optional[int]]


class PromptClientError(Exception):
    """Base exception for prompt client failures."""


class PromptApiError(PromptClientError):
    """Raised when the backend returns an API error."""


class PromptNotFoundError(PromptClientError):
    """Raised when a prompt/label/version is not found and no fallback is provided."""


class PromptClientClosedError(PromptClientError):
    """Raised when work is attempted after the prompt client has closed."""


# ---------------------------------------------------------------------------
# In-memory prompt cache with stale-while-revalidate
# ---------------------------------------------------------------------------


@dataclass
class _CacheEntry:
    value: Any
    fetched_at: float
    ttl_seconds: float
    refreshing: bool = field(default=False, repr=False)

    def is_expired(self) -> bool:
        return (_time.monotonic() - self.fetched_at) >= self.ttl_seconds


class PromptCache:
    """Thread-safe in-memory cache with stale-while-revalidate semantics."""

    def __init__(
        self,
        default_ttl: float = DEFAULT_CACHE_TTL_SECONDS,
        max_entries: int = DEFAULT_MAX_CACHE_ENTRIES,
    ):
        if max_entries <= 0:
            raise ValueError("max_entries must be greater than zero")
        self._store: "OrderedDict[CacheKey, _CacheEntry]" = OrderedDict()
        self._lock = threading.Lock()
        self._default_ttl = default_ttl
        self._max_entries = max_entries

    @staticmethod
    def cache_key(
        name: str, label: Optional[str] = None, version: Optional[int] = None
    ) -> CacheKey:
        return (name, label, version)

    def get(self, key: CacheKey) -> Optional[_CacheEntry]:
        with self._lock:
            entry = self._store.get(key)
            if entry is not None:
                self._store.move_to_end(key)
            return entry

    def set(self, key: CacheKey, value: Any, ttl: Optional[float] = None) -> None:
        with self._lock:
            self._store.pop(key, None)
            self._store[key] = _CacheEntry(
                value=value,
                fetched_at=_time.monotonic(),
                ttl_seconds=ttl if ttl is not None else self._default_ttl,
            )
            while len(self._store) > self._max_entries:
                self._store.popitem(last=False)

    def mark_refreshing(self, key: CacheKey) -> bool:
        """Mark entry as being refreshed. Returns False if already refreshing."""
        with self._lock:
            entry = self._store.get(key)
            if entry is None or entry.refreshing:
                return False
            entry.refreshing = True
            return True

    def clear_refreshing(self, key: CacheKey) -> None:
        with self._lock:
            entry = self._store.get(key)
            if entry:
                entry.refreshing = False

    def invalidate_prompt(self, name: str, *, include_versions: bool = False) -> None:
        with self._lock:
            for key in list(self._store):
                key_name, label, version = key
                if key_name != name:
                    continue
                if label is not None or version is None:
                    self._store.pop(key, None)
                elif include_versions:
                    self._store.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()


@dataclass(frozen=True)
class CachedPrompt:
    id: str
    name: str
    version: int
    content: str
    messages: Optional[List[Dict[str, str]]]
    config: Dict[str, Any]
    labels: List[str]
    updated_at: str
    type: str = "text"


@dataclass
class _SyncPromptFlight:
    completed: threading.Event = field(default_factory=threading.Event)
    result: Optional["PromptHandle"] = None
    error: Optional[BaseException] = None


class PromptHandle:
    """Compiled prompt handle returned by PromptClient.get_prompt()."""

    def __init__(self, prompt: CachedPrompt):
        self._prompt = deepcopy(prompt)

    @property
    def id(self) -> str:
        return self._prompt.id

    @property
    def name(self) -> str:
        return self._prompt.name

    @property
    def version(self) -> int:
        return self._prompt.version

    @property
    def config(self) -> Dict[str, Any]:
        return deepcopy(self._prompt.config)

    @property
    def labels(self) -> List[str]:
        return list(self._prompt.labels)

    @property
    def updated_at(self) -> str:
        return self._prompt.updated_at

    @property
    def type(self) -> str:
        return self._prompt.type

    @property
    def content(self) -> str:
        return self._prompt.content

    @property
    def messages(self) -> Optional[List[Dict[str, str]]]:
        return deepcopy(self._prompt.messages) if self._prompt.messages else None

    def compile(self, variables: Mapping[str, str]) -> str:
        """Compile string content with {{variable}} replacement."""
        if self._prompt.content:
            return _render_template(self._prompt.content, variables)

        if self._prompt.messages:
            rendered = [
                _render_template(message.get("content", ""), variables)
                for message in self._prompt.messages
            ]
            return "\n\n".join(part for part in rendered if part)

        return ""

    def compile_messages(self, variables: Mapping[str, str]) -> List[Dict[str, str]]:
        """
        Compile message list with {{variable}} replacement.

        If no messages exist, returns a single synthetic system message from content.
        """
        if self._prompt.messages:
            return [
                {
                    "role": str(message.get("role", "system")),
                    "content": _render_template(str(message.get("content", "")), variables),
                }
                for message in self._prompt.messages
            ]

        return [
            {
                "role": "system",
                "content": _render_template(self._prompt.content, variables),
            }
        ]


def _validate_label(label: Any, operation: str) -> None:
    if not isinstance(label, str) or not _LABEL_PATTERN.fullmatch(label):
        raise ValueError(
            f"{operation} labels must contain 1-50 letters, numbers, underscores, or hyphens."
        )


def _validate_tag(tag: Any, operation: str) -> None:
    if (
        not isinstance(tag, str)
        or tag != tag.strip()
        or not tag
        or len(tag) > 64
        or "\n" in tag
        or "\r" in tag
    ):
        raise ValueError(
            f"{operation} tags must contain 1-64 characters without surrounding whitespace or newlines."
        )


def _safe_error_path(path: str) -> str:
    route = path.split("?", 1)[0]
    if re.fullmatch(r"/api/v1/prompts/[^/]+/fetch", route):
        return "/api/v1/prompts/:name/fetch"
    return re.sub(
        r"^/api/managed-prompts/[^/]+(?=/|$)",
        "/api/managed-prompts/:promptId",
        route,
    )


class PromptClient:
    """
    Prompt client for Neatlogs managed prompts.

    Fetches prompts from the backend (Redis-backed, falls back to Postgres).
    Includes an in-memory cache with stale-while-revalidate: after the first
    fetch, subsequent calls return from cache instantly and refresh in the
    background when the TTL expires.
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        session: Optional[requests.Session] = None,
        cache_ttl_seconds: float = DEFAULT_CACHE_TTL_SECONDS,
        max_cache_entries: int = DEFAULT_MAX_CACHE_ENTRIES,
        max_refresh_workers: int = DEFAULT_MAX_REFRESH_WORKERS,
    ):
        if max_refresh_workers <= 0:
            raise ValueError("max_refresh_workers must be greater than zero")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._session = session or requests.Session()
        self._cache = PromptCache(
            default_ttl=cache_ttl_seconds,
            max_entries=max_cache_entries,
        )
        self._refresh_slots = threading.BoundedSemaphore(max_refresh_workers)
        self._refresh_threads: Set[threading.Thread] = set()
        self._refresh_threads_lock = threading.Lock()
        self._inflight: Dict[CacheKey, _SyncPromptFlight] = {}
        self._inflight_lock = threading.Lock()
        self._cache_epoch = 0
        self._cache_epoch_lock = threading.Lock()
        self._closed = False

    def _assert_open(self) -> None:
        with self._cache_epoch_lock:
            if self._closed:
                raise PromptClientClosedError("PromptClient is closed.")

    def get_prompt(
        self,
        name: str,
        *,
        label: Optional[str] = None,
        version: Optional[int] = None,
        type: str = "text",
        cache_ttl_seconds: Optional[float] = None,
    ) -> PromptHandle:
        """
        Fetch a prompt from the backend (Redis → Postgres fallback).

        Uses an in-memory cache with stale-while-revalidate:
        - Cache hit (fresh): returns immediately, no network call.
        - Cache hit (stale): returns immediately, refreshes in background.
        - Cache miss: fetches from backend, caches, then returns.

        Args:
            name: Prompt name.
            label: Return the version holding this label.
            version: Return this specific version number.
            type: Prompt type ("text" or "chat").
            cache_ttl_seconds: Override the default cache TTL for this prompt.
        """
        self._assert_open()
        if label is not None and version is not None:
            raise ValueError("Cannot specify both label and version.")

        cache_key = PromptCache.cache_key(name, label=label, version=version)
        entry = self._cache.get(cache_key)

        if entry is not None:
            # Prompt versions are immutable. Once this client has fetched an
            # explicitly pinned version, refreshing it cannot produce a newer
            # version and only adds avoidable prompt-API traffic.
            if version is not None or not entry.is_expired():
                return entry.value
            # Stale — return immediately, refresh in background
            self._background_refresh(
                cache_key,
                name,
                label=label,
                version=version,
                ttl=entry.ttl_seconds if cache_ttl_seconds is None else cache_ttl_seconds,
            )
            return entry.value

        # Coalesce a cold miss so concurrent callers for one selector share a
        # single bounded backend request.
        with self._inflight_lock:
            flight = self._inflight.get(cache_key)
            owns_flight = flight is None
            if flight is None:
                flight = _SyncPromptFlight()
                self._inflight[cache_key] = flight

        if not owns_flight:
            if not flight.completed.wait(DEFAULT_CONNECT_TIMEOUT + DEFAULT_READ_TIMEOUT + 1.0):
                raise PromptApiError("Prompt request did not complete before its deadline")
            if flight.error is not None:
                raise flight.error
            if flight.result is None:
                raise PromptApiError("Prompt request completed without a result")
            return flight.result

        try:
            with self._cache_epoch_lock:
                cache_epoch = self._cache_epoch
            handle = self._fetch_prompt(name, label=label, version=version)
            with self._cache_epoch_lock:
                if self._closed:
                    raise PromptClientClosedError("PromptClient is closed.")
                if self._cache_epoch == cache_epoch:
                    self._cache.set(cache_key, handle, cache_ttl_seconds)
            flight.result = handle
            return handle
        except BaseException as exc:
            flight.error = exc
            raise
        finally:
            flight.completed.set()
            with self._inflight_lock:
                if self._inflight.get(cache_key) is flight:
                    self._inflight.pop(cache_key, None)

    def _fetch_prompt(
        self,
        name: str,
        *,
        label: Optional[str] = None,
        version: Optional[int] = None,
    ) -> PromptHandle:
        """Fetch prompt from backend (no cache involved)."""
        if label is not None:
            return PromptHandle(self.fetch_prompt(name, label=label))

        listing = self.list_prompts(name=name, limit=500)
        items = listing.get("items", [])

        if not items:
            raise PromptNotFoundError(f"No versions found for prompt '{name}'")

        if version is not None:
            offset = 0
            while True:
                for item in items:
                    if item.get("version") == version:
                        return PromptHandle(_normalize_prompt_object(item))
                offset += len(items)
                total = int(listing.get("total") or offset)
                if not items or offset >= total:
                    break
                listing = self.list_prompts(name=name, limit=500, offset=offset)
                items = listing.get("items", [])
            raise PromptNotFoundError(f"Prompt '{name}' version {version} not found")

        latest = max(items, key=lambda x: x.get("createdAt") or x.get("created_at") or "")
        return PromptHandle(_normalize_prompt_object(latest))

    def _background_refresh(
        self,
        cache_key: CacheKey,
        name: str,
        *,
        label: Optional[str] = None,
        version: Optional[int] = None,
        ttl: Optional[float] = None,
    ) -> None:
        """Refresh a stale cache entry in a background thread (deduped)."""
        if not self._cache.mark_refreshing(cache_key):
            return
        if not self._refresh_slots.acquire(blocking=False):
            self._cache.clear_refreshing(cache_key)
            return
        with self._cache_epoch_lock:
            if self._closed:
                self._refresh_slots.release()
                self._cache.clear_refreshing(cache_key)
                return
            cache_epoch = self._cache_epoch

        def _refresh():
            try:
                handle = self._fetch_prompt(name, label=label, version=version)
                with self._cache_epoch_lock:
                    if not self._closed and self._cache_epoch == cache_epoch:
                        self._cache.set(cache_key, handle, ttl)
            except Exception as e:
                logger.debug(
                    "Background prompt refresh failed (%s)",
                    type(e).__name__,
                )
            finally:
                self._cache.clear_refreshing(cache_key)
                self._refresh_slots.release()
                with self._refresh_threads_lock:
                    self._refresh_threads.discard(threading.current_thread())

        thread = threading.Thread(target=_refresh, daemon=True)
        with self._refresh_threads_lock:
            with self._cache_epoch_lock:
                if self._closed:
                    self._refresh_slots.release()
                    self._cache.clear_refreshing(cache_key)
                    return
            self._refresh_threads.add(thread)
            thread.start()

    def close(self, timeout_seconds: float = 0.5) -> None:
        """Stop accepting work, release cached prompts, and bound refresh cleanup."""
        with self._cache_epoch_lock:
            if self._closed:
                return
            self._closed = True
            self._cache_epoch += 1
        self._cache.clear()
        deadline = _time.monotonic() + max(0.0, timeout_seconds)
        with self._refresh_threads_lock:
            threads = list(self._refresh_threads)
        for thread in threads:
            remaining = deadline - _time.monotonic()
            if remaining <= 0:
                break
            thread.join(remaining)
        self._session.close()

    def _invalidate_prompt_cache(self, name: str, *, include_versions: bool = False) -> None:
        with self._cache_epoch_lock:
            self._cache_epoch += 1
            self._cache.invalidate_prompt(name, include_versions=include_versions)

    # ----------------------------
    # API helpers
    # ----------------------------

    def fetch_prompt(self, name: str, *, label: str) -> CachedPrompt:
        """
        Fetch one prompt by name+label from /api/v1/prompts/:name/fetch.
        Backend checks Redis first, then Postgres.
        """
        self._assert_open()
        path = f"/api/v1/prompts/{quote(name, safe='')}/fetch"
        payload = self._request_json(method="GET", path=path, params={"label": label})
        return _normalize_prompt_object(payload)

    def list_prompts(
        self,
        *,
        name: Optional[str] = None,
        source: Optional[str] = None,
        label: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """List prompt versions from /api/managed-prompts."""
        self._assert_open()
        params: Dict[str, Any] = {
            "limit": max(1, min(limit, 500)),
            "offset": max(0, offset),
        }
        if name:
            params["name"] = name
        if source:
            params["source"] = source
        if label:
            params["label"] = label

        return self._request_json(method="GET", path="/api/managed-prompts", params=params)

    def create_prompt(
        self,
        *,
        name: str,
        prompt: Union[str, Sequence[Dict[str, str]]],
        type: str = "text",
        labels: Sequence[str] = (),
        tags: Optional[Sequence[str]] = None,
        config: Optional[Mapping[str, Any]] = None,
        commit_message: Optional[str] = None,
    ) -> PromptHandle:
        """
        Create a new prompt version via /api/managed-prompts.

        For type="text", prompt must be a str.
        For type="chat", prompt must be a list of {"role", "content"} dicts.
        labels may contain one active label (for example, "production").
        """
        self._assert_open()
        if len(labels) > 1:
            raise ValueError("labels may contain at most one label.")
        for label in labels:
            _validate_label(label, "create_prompt")
        for tag in tags or ():
            _validate_tag(tag, "create_prompt")
        if type == "text" and not isinstance(prompt, str):
            raise ValueError("For type='text', prompt must be a string.")
        if type == "chat" and not isinstance(prompt, list):
            raise ValueError("For type='chat', prompt must be a list of message dicts.")

        body: Dict[str, Any] = {"name": name}
        if type == "chat":
            body["messages"] = list(prompt)  # type: ignore[arg-type]
        else:
            body["content"] = prompt
        if labels is not None:
            body["labels"] = list(labels)
        if tags is not None:
            body["tags"] = list(tags)
        if config is not None:
            body["config"] = dict(config)
        if commit_message is not None:
            body["commit_message"] = commit_message

        payload = self._request_json(method="POST", path="/api/managed-prompts", json_body=body)
        self._invalidate_prompt_cache(name)
        return PromptHandle(_normalize_prompt_object(payload.get("prompt", payload)))

    def update_prompt(
        self,
        *,
        name: str,
        version: int,
        new_labels: Sequence[str] = (),
    ) -> Dict[str, Any]:
        """
        Move labels onto a specific prompt version via /api/managed-prompts/:promptId/labels.

        new_labels must contain the one active label to assign.
        """
        self._assert_open()
        if len(new_labels) != 1:
            raise ValueError(
                "new_labels must contain exactly one label, e.g. new_labels=['production']."
            )
        _validate_label(new_labels[0], "update_prompt")

        prompt_id = self._find_prompt_id(name, version)

        path = f"/api/managed-prompts/{quote(prompt_id, safe='')}/labels"
        label = new_labels[0]
        last_response = self._request_json(method="POST", path=path, json_body={"label": label})

        self._invalidate_prompt_cache(name)
        return {"name": name, "version": version, "labels": list(new_labels), **last_response}

    def delete_prompt(
        self,
        name: str,
        version: int,
    ) -> Dict[str, Any]:
        """
        Soft-delete a specific prompt version via DELETE /api/managed-prompts/:promptId.
        """
        self._assert_open()
        prompt_id = self._find_prompt_id(name, version)

        path = f"/api/managed-prompts/{quote(prompt_id, safe='')}"
        response = self._request_json(method="DELETE", path=path)
        self._invalidate_prompt_cache(name, include_versions=True)
        return response

    def remove_tag(
        self,
        name: str,
        version: int,
        tag: str,
    ) -> Dict[str, Any]:
        """
        Remove a tag from a prompt version via DELETE /api/managed-prompts/:promptId/tags.
        """
        self._assert_open()
        _validate_tag(tag, "remove_tag")
        prompt_id = self._find_prompt_id(name, version)

        path = f"/api/managed-prompts/{quote(prompt_id, safe='')}/tags"
        response = self._request_json(method="DELETE", path=path, json_body={"tag": tag})
        self._invalidate_prompt_cache(name)
        return response

    def save_as_version(
        self,
        *,
        prompt_name: str,
        content: Optional[str] = None,
        messages: Optional[Sequence[Dict[str, str]]] = None,
        config: Optional[Mapping[str, Any]] = None,
        commit_message: Optional[str] = None,
        labels: Optional[Sequence[str]] = None,
        tags: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        """Save prompt content/messages as a new version via the playground endpoint."""
        self._assert_open()
        has_content = isinstance(content, str) and bool(content.strip())
        has_messages = messages is not None and len(messages) > 0
        if not has_content and not has_messages:
            raise ValueError("save_as_version requires non-empty content or messages.")
        if labels is not None and len(labels) > 1:
            raise ValueError("labels may contain at most one label.")
        for label in labels or ():
            _validate_label(label, "save_as_version")
        for tag in tags or ():
            _validate_tag(tag, "save_as_version")
        body: Dict[str, Any] = {"promptName": prompt_name}
        if content is not None:
            body["content"] = content
        if messages is not None:
            body["messages"] = list(messages)
        if config is not None:
            body["config"] = dict(config)
        if commit_message is not None:
            body["commitMessage"] = commit_message
        if labels is not None:
            body["labels"] = list(labels)
        if tags is not None:
            body["tags"] = list(tags)

        response = self._request_json(
            method="POST", path="/api/prompt-playground/save-as-version", json_body=body
        )
        self._invalidate_prompt_cache(prompt_name)
        return response

    def _find_prompt_id(self, name: str, version: int) -> str:
        """Resolve a prompt version to its backend UUID across paginated history."""
        offset = 0
        while True:
            listing = self.list_prompts(name=name, limit=500, offset=offset)
            items = listing.get("items", [])
            for item in items:
                if item.get("version") == version and isinstance(item.get("id"), str):
                    return item["id"]
            offset += len(items)
            total = int(listing.get("total") or offset)
            if not items or offset >= total:
                raise PromptNotFoundError(f"Prompt '{name}' version {version} not found")

    # ----------------
    # Internal helpers
    # ----------------

    def _auth_headers(self) -> Dict[str, str]:
        return {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "x-api-key": self.api_key,
        }

    def _request_json(
        self,
        *,
        method: str,
        path: str,
        params: Optional[Mapping[str, Any]] = None,
        json_body: Optional[Mapping[str, Any]] = None,
        timeout_seconds: float = DEFAULT_READ_TIMEOUT,
    ) -> Dict[str, Any]:
        self._assert_open()
        url = f"{self.base_url}{path}"
        safe_path = _safe_error_path(path)

        try:
            from opentelemetry.context import attach, detach, set_value
            from opentelemetry.instrumentation.utils import _SUPPRESS_INSTRUMENTATION_KEY

            _token = attach(set_value(_SUPPRESS_INSTRUMENTATION_KEY, True))
        except Exception:
            _token = None

        try:
            try:
                response = self._session.request(
                    method=method,
                    url=url,
                    params=params,
                    json=json_body,
                    headers={**self._auth_headers(), "Content-Type": "application/json"},
                    timeout=(DEFAULT_CONNECT_TIMEOUT, timeout_seconds),
                )
            except requests.RequestException:
                raise PromptApiError(f"{method} {safe_path} request failed") from None
        finally:
            if _token is not None:
                try:
                    from opentelemetry.context import detach

                    detach(_token)
                except Exception:
                    pass

        if response.status_code >= 400:
            raise PromptApiError(f"{method} {safe_path} failed ({response.status_code})")

        try:
            payload = response.json()
        except Exception:
            raise PromptApiError(f"{method} {safe_path} returned non-JSON response") from None

        if not isinstance(payload, MutableMapping):
            raise PromptApiError(f"{method} {safe_path} returned unexpected response shape")

        return dict(payload)


def _render_template(template: str, variables: Mapping[str, str]) -> str:
    return _PLACEHOLDER_PATTERN.sub(
        lambda match: str(variables.get(match.group(1), match.group(0))),
        template,
    )


def _normalize_prompt_object(raw: Mapping[str, Any]) -> CachedPrompt:
    raw_messages = raw.get("messages")
    messages: Optional[List[Dict[str, str]]] = None
    if isinstance(raw_messages, Sequence) and not isinstance(raw_messages, (str, bytes, bytearray)):
        message_list: List[Dict[str, str]] = []
        for item in raw_messages:
            if isinstance(item, Mapping):
                message_list.append(
                    {
                        "role": str(item.get("role", "system")),
                        "content": str(item.get("content", "")),
                    }
                )
        if message_list:
            messages = message_list

    raw_labels = raw.get("labels")
    labels: List[str] = []
    if isinstance(raw_labels, Sequence) and not isinstance(raw_labels, (str, bytes, bytearray)):
        labels = [str(label) for label in raw_labels if str(label).strip()]

    config = raw.get("config")
    if not isinstance(config, Mapping):
        config = {}

    content = raw.get("content")
    if not isinstance(content, str):
        content = ""

    prompt_id = raw.get("id")
    if not isinstance(prompt_id, str):
        prompt_id = ""

    name = raw.get("name")
    if not isinstance(name, str):
        name = ""

    version_value = raw.get("version")
    try:
        version = int(version_value) if version_value is not None else 0
    except Exception:
        version = 0

    updated_at = raw.get("updatedAt")
    if not isinstance(updated_at, str):
        updated_at = str(raw.get("updated_at") or "")

    prompt_type = raw.get("type")
    if not isinstance(prompt_type, str) or prompt_type not in ("text", "chat"):
        prompt_type = "text"

    return CachedPrompt(
        id=prompt_id,
        name=name,
        version=version,
        content=content,
        messages=messages,
        config=deepcopy(dict(config)),
        labels=labels,
        updated_at=updated_at,
        type="chat" if messages else prompt_type,
    )


# ---------------------------------------------------------------------------
# Module-level prompt API — credentials sourced from neatlogs.init()
# ---------------------------------------------------------------------------

_shared_client: Optional[PromptClient] = None


def _get_shared_client() -> PromptClient:
    global _shared_client
    from ..init import _session_config

    api_key = _session_config.get("_api_key") or ""
    base_url = _session_config.get("_base_url") or ""

    if not api_key or api_key == "disabled":
        raise PromptClientError(
            "No API key available. Call neatlogs.init(api_key=...) before using prompt methods."
        )

    if _shared_client is not None:
        if (
            not _shared_client._closed
            and _shared_client.api_key == api_key
            and _shared_client.base_url == base_url.rstrip("/")
        ):
            return _shared_client
        _shared_client.close()

    _shared_client = PromptClient(base_url=base_url, api_key=api_key)
    return _shared_client


def get_prompt(
    name: str,
    *,
    label: Optional[str] = None,
    version: Optional[int] = None,
    type: str = "text",
) -> PromptHandle:
    return _get_shared_client().get_prompt(name, label=label, version=version, type=type)


def fetch_prompt(name: str, *, label: str) -> CachedPrompt:
    return _get_shared_client().fetch_prompt(name, label=label)


def list_prompts(
    *,
    name: Optional[str] = None,
    source: Optional[str] = None,
    label: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
) -> Dict[str, Any]:
    return _get_shared_client().list_prompts(
        name=name, source=source, label=label, limit=limit, offset=offset
    )


def create_prompt(
    *,
    name: str,
    prompt: Union[str, Sequence[Dict[str, str]]],
    type: str = "text",
    labels: Sequence[str] = (),
    tags: Optional[Sequence[str]] = None,
    config: Optional[Mapping[str, Any]] = None,
    commit_message: Optional[str] = None,
) -> PromptHandle:
    return _get_shared_client().create_prompt(
        name=name,
        prompt=prompt,
        type=type,
        labels=labels,
        tags=tags,
        config=config,
        commit_message=commit_message,
    )


def update_prompt(
    *,
    name: str,
    version: int,
    new_labels: Sequence[str] = (),
) -> Dict[str, Any]:
    return _get_shared_client().update_prompt(name=name, version=version, new_labels=new_labels)


def save_as_version(
    *,
    prompt_name: str,
    content: Optional[str] = None,
    messages: Optional[Sequence[Dict[str, str]]] = None,
    config: Optional[Mapping[str, Any]] = None,
    commit_message: Optional[str] = None,
    labels: Optional[Sequence[str]] = None,
    tags: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    return _get_shared_client().save_as_version(
        prompt_name=prompt_name,
        content=content,
        messages=messages,
        config=config,
        commit_message=commit_message,
        labels=labels,
        tags=tags,
    )


def delete_prompt(name: str, version: int) -> Dict[str, Any]:
    return _get_shared_client().delete_prompt(name, version)


def remove_tag(name: str, version: int, tag: str) -> Dict[str, Any]:
    return _get_shared_client().remove_tag(name, version, tag)


# ---------------------------------------------------------------------------
# Async prompt client — uses httpx, runs on the event loop without threads
# ---------------------------------------------------------------------------


class AsyncPromptClient:
    """
    Async prompt client for Neatlogs managed prompts.

    Uses httpx.AsyncClient — no thread pool needed, runs directly on the event loop.
    Includes the same in-memory stale-while-revalidate cache as the sync client.
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        connect_timeout: float = DEFAULT_CONNECT_TIMEOUT,
        read_timeout: float = DEFAULT_READ_TIMEOUT,
        cache_ttl_seconds: float = DEFAULT_CACHE_TTL_SECONDS,
        max_cache_entries: int = DEFAULT_MAX_CACHE_ENTRIES,
        max_refresh_workers: int = DEFAULT_MAX_REFRESH_WORKERS,
    ):
        import httpx

        if max_refresh_workers <= 0:
            raise ValueError("max_refresh_workers must be greater than zero")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=connect_timeout, read=read_timeout, write=5.0, pool=5.0),
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
                "x-api-key": api_key,
            },
        )
        self._cache = PromptCache(
            default_ttl=cache_ttl_seconds,
            max_entries=max_cache_entries,
        )
        self._refresh_tasks: Set[Any] = set()
        self._inflight_tasks: Dict[CacheKey, Any] = {}
        self._max_refresh_workers = max_refresh_workers
        self._cache_epoch = 0
        self._closed = False
        self._transport_closed = False

    async def get_prompt(
        self,
        name: str,
        *,
        label: Optional[str] = None,
        version: Optional[int] = None,
        type: str = "text",
        cache_ttl_seconds: Optional[float] = None,
    ) -> PromptHandle:
        if self._closed:
            raise PromptClientClosedError("AsyncPromptClient is closed.")
        if label is not None and version is not None:
            raise ValueError("Cannot specify both label and version.")

        cache_key = PromptCache.cache_key(name, label=label, version=version)
        entry = self._cache.get(cache_key)

        if entry is not None:
            if version is not None or not entry.is_expired():
                return entry.value
            # Stale — return immediately, refresh in background task
            self._background_refresh(
                cache_key,
                name,
                label=label,
                version=version,
                ttl=entry.ttl_seconds if cache_ttl_seconds is None else cache_ttl_seconds,
            )
            return entry.value

        # Concurrent cold misses share one task. Shielding prevents one caller's
        # cancellation from cancelling the request for every other waiter.
        import asyncio

        request = self._inflight_tasks.get(cache_key)
        if request is None:
            cache_epoch = self._cache_epoch

            async def _fetch_and_cache() -> PromptHandle:
                handle = await self._fetch_prompt(name, label=label, version=version)
                if self._closed:
                    raise PromptClientClosedError("AsyncPromptClient is closed.")
                if self._cache_epoch == cache_epoch:
                    self._cache.set(cache_key, handle, cache_ttl_seconds)
                return handle

            request = asyncio.create_task(_fetch_and_cache())
            self._inflight_tasks[cache_key] = request

            def _remove_inflight(done: Any) -> None:
                if self._inflight_tasks.get(cache_key) is done:
                    self._inflight_tasks.pop(cache_key, None)

            request.add_done_callback(_remove_inflight)

        return await asyncio.shield(request)

    async def _fetch_prompt(
        self,
        name: str,
        *,
        label: Optional[str] = None,
        version: Optional[int] = None,
    ) -> PromptHandle:
        if label is not None:
            path = f"/api/v1/prompts/{quote(name, safe='')}/fetch"
            payload = await self._request_json(method="GET", path=path, params={"label": label})
            return PromptHandle(_normalize_prompt_object(payload))

        params: Dict[str, Any] = {"limit": 500, "offset": 0, "name": name}
        listing = await self._request_json(method="GET", path="/api/managed-prompts", params=params)
        items = listing.get("items", [])

        if not items:
            raise PromptNotFoundError(f"No versions found for prompt '{name}'")

        if version is not None:
            offset = 0
            while True:
                for item in items:
                    if item.get("version") == version:
                        return PromptHandle(_normalize_prompt_object(item))
                offset += len(items)
                total = int(listing.get("total") or offset)
                if not items or offset >= total:
                    break
                params["offset"] = offset
                listing = await self._request_json(
                    method="GET", path="/api/managed-prompts", params=params
                )
                items = listing.get("items", [])
            raise PromptNotFoundError(f"Prompt '{name}' version {version} not found")

        latest = max(items, key=lambda x: x.get("createdAt") or x.get("created_at") or "")
        return PromptHandle(_normalize_prompt_object(latest))

    def _background_refresh(
        self,
        cache_key: CacheKey,
        name: str,
        *,
        label: Optional[str] = None,
        version: Optional[int] = None,
        ttl: Optional[float] = None,
    ) -> None:
        if not self._cache.mark_refreshing(cache_key):
            return
        if self._closed or len(self._refresh_tasks) >= self._max_refresh_workers:
            self._cache.clear_refreshing(cache_key)
            return

        cache_epoch = self._cache_epoch

        import asyncio

        async def _refresh():
            try:
                handle = await self._fetch_prompt(name, label=label, version=version)
                if not self._closed and self._cache_epoch == cache_epoch:
                    self._cache.set(cache_key, handle, ttl)
            except Exception as e:
                logger.debug(
                    "Background async prompt refresh failed (%s)",
                    type(e).__name__,
                )
            finally:
                self._cache.clear_refreshing(cache_key)

        try:
            loop = asyncio.get_running_loop()
            task = loop.create_task(_refresh())
            self._refresh_tasks.add(task)
            task.add_done_callback(self._refresh_tasks.discard)
        except RuntimeError:
            self._cache.clear_refreshing(cache_key)

    async def _request_json(
        self,
        *,
        method: str,
        path: str,
        params: Optional[Mapping[str, Any]] = None,
        json_body: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        import httpx

        if self._closed:
            raise PromptClientClosedError("AsyncPromptClient is closed.")
        url = f"{self.base_url}{path}"
        safe_path = _safe_error_path(path)

        try:
            from opentelemetry.context import attach, detach, set_value
            from opentelemetry.instrumentation.utils import _SUPPRESS_INSTRUMENTATION_KEY

            _token = attach(set_value(_SUPPRESS_INSTRUMENTATION_KEY, True))
        except Exception:
            _token = None

        try:
            try:
                response = await self._client.request(
                    method=method,
                    url=url,
                    params=params,
                    json=json_body,
                )
            except httpx.HTTPError:
                raise PromptApiError(f"{method} {safe_path} request failed") from None
        finally:
            if _token is not None:
                try:
                    from opentelemetry.context import detach

                    detach(_token)
                except Exception:
                    pass

        if response.status_code >= 400:
            raise PromptApiError(f"{method} {safe_path} failed ({response.status_code})")

        try:
            payload = response.json()
        except Exception:
            raise PromptApiError(f"{method} {safe_path} returned non-JSON response") from None

        if not isinstance(payload, MutableMapping):
            raise PromptApiError(f"{method} {safe_path} returned unexpected response shape")

        return dict(payload)

    async def close(self):
        if self._transport_closed:
            return
        if not self._closed:
            self._closed = True
            self._cache_epoch += 1
        tasks = list(self._refresh_tasks | set(self._inflight_tasks.values()))
        for task in tasks:
            task.cancel()
        if tasks:
            import asyncio

            await asyncio.gather(*tasks, return_exceptions=True)
        self._refresh_tasks.clear()
        self._inflight_tasks.clear()
        self._cache.clear()
        await self._client.aclose()
        self._transport_closed = True


# ---------------------------------------------------------------------------
# Module-level async API
# ---------------------------------------------------------------------------

_shared_async_client: Optional[AsyncPromptClient] = None


async def _get_shared_async_client() -> AsyncPromptClient:
    global _shared_async_client
    from ..init import _session_config

    api_key = _session_config.get("_api_key") or ""
    base_url = _session_config.get("_base_url") or ""

    if not api_key or api_key == "disabled":
        raise PromptClientError(
            "No API key available. Call neatlogs.init(api_key=...) before using prompt methods."
        )

    if _shared_async_client is not None:
        if (
            not _shared_async_client._closed
            and _shared_async_client.api_key == api_key
            and _shared_async_client.base_url == base_url.rstrip("/")
        ):
            return _shared_async_client
        await _shared_async_client.close()

    _shared_async_client = AsyncPromptClient(base_url=base_url, api_key=api_key)
    return _shared_async_client


async def aget_prompt(
    name: str,
    *,
    label: Optional[str] = None,
    version: Optional[int] = None,
    type: str = "text",
    cache_ttl_seconds: Optional[float] = None,
) -> PromptHandle:
    """Async version of get_prompt — no thread pool needed."""
    client = await _get_shared_async_client()
    return await client.get_prompt(
        name, label=label, version=version, type=type, cache_ttl_seconds=cache_ttl_seconds
    )


def _close_shared_prompt_clients() -> None:
    """Detach prompt clients from a completed default SDK generation."""
    global _shared_client, _shared_async_client
    if _shared_client is not None:
        _shared_client.close()
        _shared_client = None
    if _shared_async_client is not None:
        import asyncio

        async_client = _shared_async_client
        async_client._closed = True
        async_client._cache_epoch += 1
        async_client._cache.clear()
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(async_client.close())
        else:
            loop.create_task(async_client.close())
        _shared_async_client = None
