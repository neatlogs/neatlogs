from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, MutableMapping, Optional, Sequence, Union
from urllib.parse import quote

import requests

from ..core.logger import get_logger

logger = get_logger()

_PLACEHOLDER_PATTERN = re.compile(r"\{\{\s*([A-Za-z0-9_.-]+)\s*\}\}")


class PromptClientError(Exception):
    """Base exception for prompt client failures."""


class PromptApiError(PromptClientError):
    """Raised when the backend returns an API error."""


class PromptNotFoundError(PromptClientError):
    """Raised when a prompt/label pair is not in cache and no fallback is provided."""


class PromptConnectionTimeoutError(PromptClientError):
    """Raised when initial SSE snapshot is not received in time."""


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


class PromptHandle:
    """Compiled prompt handle returned by PromptStreamClient.get_prompt()."""

    def __init__(self, prompt: CachedPrompt):
        self._prompt = prompt

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
        return dict(self._prompt.config)

    @property
    def labels(self) -> List[str]:
        return list(self._prompt.labels)

    @property
    def updated_at(self) -> str:
        return self._prompt.updated_at

    @property
    def type(self) -> str:
        return self._prompt.type

    def compile(self, variables: Mapping[str, str]) -> str:
        """
        Compile string content with {{variable}} replacement.

        Unmatched placeholders are left unchanged.
        """
        if self._prompt.content:
            return _render_template(self._prompt.content, variables)

        if self._prompt.messages:
            rendered = [_render_template(message.get("content", ""), variables) for message in self._prompt.messages]
            return "\n\n".join(part for part in rendered if part)

        return ""

    def compile_messages(self, variables: Mapping[str, str]) -> List[Dict[str, str]]:
        """
        Compile message list with {{variable}} replacement.

        If no messages exist, this returns a single synthetic system message from content.
        """
        if self._prompt.messages:
            compiled_messages: List[Dict[str, str]] = []
            for message in self._prompt.messages:
                role = str(message.get("role", "system"))
                content = _render_template(str(message.get("content", "")), variables)
                compiled_messages.append({"role": role, "content": content})
            return compiled_messages

        return [
            {
                "role": "system",
                "content": _render_template(self._prompt.content, variables),
            }
        ]


class PromptStreamClient:
    """
    Event-driven prompt client for Neatlogs managed prompts.

    Features:
    - Persistent SSE stream for prompt snapshot + label move updates
    - In-memory cache keyed by prompt name + label
    - Synchronous get_prompt() after connect() receives first snapshot
    - Optional prompt API helpers (fetch/list/create/set-label/save-as-version)
    """

    def __init__(
        self,
        *,
        base_url: str,
        project_id: str,
        api_key: str,
        on_error: Optional[Callable[[Exception], None]] = None,
        connect_timeout_seconds: float = 10.0,
        read_timeout_seconds: float = 65.0,
        reconnect_initial_seconds: float = 2.0,
        reconnect_max_seconds: float = 30.0,
        dev_user_id: Optional[str] = None,
        session: Optional[requests.Session] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.project_id = project_id
        self.api_key = api_key
        self.on_error = on_error
        self.connect_timeout_seconds = connect_timeout_seconds
        self.read_timeout_seconds = read_timeout_seconds
        self.reconnect_initial_seconds = max(0.5, reconnect_initial_seconds)
        self.reconnect_max_seconds = max(self.reconnect_initial_seconds, reconnect_max_seconds)
        self.dev_user_id = dev_user_id

        self._session = session or requests.Session()

        self._cache: Dict[str, Dict[str, CachedPrompt]] = {}
        self._cache_lock = threading.Lock()

        self._snapshot_received = threading.Event()
        self._stop_event = threading.Event()
        self._connected_lock = threading.Lock()
        self._connected = False

        self._stream_thread: Optional[threading.Thread] = None
        self._active_stream_response: Optional[requests.Response] = None

    @property
    def is_connected(self) -> bool:
        with self._connected_lock:
            return self._connected

    def connect(self, timeout_seconds: float = 10.0) -> None:
        """
        Start the SSE stream and wait for the first snapshot.

        Raises PromptConnectionTimeoutError if the initial snapshot is not received
        before timeout.
        """
        if self._stream_thread and self._stream_thread.is_alive():
            if self._snapshot_received.wait(timeout=timeout_seconds):
                return
            raise PromptConnectionTimeoutError(
                f"Timed out after {timeout_seconds}s waiting for prompt snapshot"
            )

        self._stop_event.clear()
        self._stream_thread = threading.Thread(
            target=self._run_stream_loop,
            name="neatlogs-prompt-stream",
            daemon=True,
        )
        self._stream_thread.start()

        if not self._snapshot_received.wait(timeout=timeout_seconds):
            raise PromptConnectionTimeoutError(
                f"Timed out after {timeout_seconds}s waiting for prompt snapshot"
            )

    def disconnect(self) -> None:
        self._stop_event.set()
        response = self._active_stream_response
        if response is not None:
            try:
                response.close()
            except Exception:
                pass

        if self._stream_thread and self._stream_thread.is_alive():
            self._stream_thread.join(timeout=2.0)

        with self._connected_lock:
            self._connected = False

    def get_prompt(
        self,
        name: str,
        *,
        label: Optional[str] = None,
        version: Optional[int] = None,
        type: str = "text",
        fallback: Optional[Union[str, Sequence[Dict[str, str]]]] = None,
    ) -> PromptHandle:
        """
        Return prompt handle from cache (or fallback).

        connect() must be called beforehand for deterministic behavior.
        If neither label nor version is provided, defaults to label="production".
        """
        if label is not None and version is not None:
            raise ValueError("Cannot specify both label and version.")

        resolved_label = label if label is not None else ("production" if version is None else None)

        with self._cache_lock:
            by_label = self._cache.get(name, {})
            cached: Optional[CachedPrompt] = None
            if resolved_label is not None:
                cached = by_label.get(resolved_label)
            elif version is not None:
                cached = next(
                    (p for p in by_label.values() if p.version == version),
                    None,
                )
            has_any_cached = bool(self._cache)

        if cached is not None:
            return PromptHandle(cached)

        if fallback is not None:
            fallback_label = resolved_label or f"v{version}"
            fallback_prompt = _build_fallback_prompt(name=name, label=fallback_label, fallback=fallback)
            return PromptHandle(fallback_prompt)

        if not has_any_cached and not self.is_connected:
            raise PromptNotFoundError(
                "Prompt cache is empty and stream is disconnected. Call connect() first."
            )

        ref = f"label '{resolved_label}'" if resolved_label else f"version {version}"
        raise PromptNotFoundError(f"Prompt '{name}' with {ref} not found in cache")

    def get_cached_prompt(self, name: str, label: str) -> Optional[CachedPrompt]:
        with self._cache_lock:
            by_label = self._cache.get(name, {})
            return by_label.get(label)

    # ----------------------------
    # API helpers (Langfuse-style)
    # ----------------------------

    def fetch_prompt(self, name: str, *, label: str, project_id: Optional[str] = None) -> CachedPrompt:
        """
        Fetch one prompt by name+label from /api/v1/prompts/:name/fetch.

        This endpoint supports API key auth.
        """
        pid = project_id or self.project_id
        path = f"/api/v1/prompts/{quote(name, safe='')}/fetch"
        payload = self._request_json(
            method="GET",
            path=path,
            params={"projectId": pid, "label": label},
        )
        cached = _normalize_prompt_object(payload)
        self._merge_prompt_into_cache(cached)
        return cached

    def list_prompts(
        self,
        *,
        project_id: Optional[str] = None,
        name: Optional[str] = None,
        source: Optional[str] = None,
        label: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """
        List prompt versions from /api/managed-prompts.

        Note: This route requires managed-prompt auth (session or x-dev-user-id in dev).
        """
        pid = project_id or self.project_id
        params: Dict[str, Any] = {
            "project_id": pid,
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
        labels: Sequence[str],
        tags: Optional[Sequence[str]] = None,
        config: Optional[Mapping[str, Any]] = None,
        commit_message: Optional[str] = None,
        project_id: Optional[str] = None,
    ) -> PromptHandle:
        """
        Create a new prompt version via /api/managed-prompts.

        For type="text", prompt must be a str.
        For type="chat", prompt must be a list of {"role", "content"} dicts.
        labels is required — specify at least one label (e.g. "production", "staging").
        Labels are moved atomically — existing holders of each label are updated.
        """
        if not labels:
            raise ValueError("labels is required. Specify at least one label, e.g. labels=['production'].")
        if type == "text" and not isinstance(prompt, str):
            raise ValueError("For type='text', prompt must be a string.")
        if type == "chat" and not isinstance(prompt, list):
            raise ValueError("For type='chat', prompt must be a list of message dicts.")

        pid = project_id or self.project_id
        body: Dict[str, Any] = {"project_id": pid, "name": name, "type": type}

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
        cached = _normalize_prompt_object(payload.get("prompt", payload))
        self._merge_prompt_into_cache(cached)
        return PromptHandle(cached)

    def update_prompt(
        self,
        *,
        name: str,
        version: int,
        new_labels: Sequence[str] = (),
        project_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Move labels onto a specific prompt version via /api/managed-prompts/:promptId/labels.

        Resolves prompt_id from name+version (cache first, then list_prompts fallback).
        Each label is moved atomically — the old holder loses the label.
        new_labels is required — specify at least one label (e.g. new_labels=["production"]).
        """
        if not new_labels:
            raise ValueError("new_labels is required. Specify at least one label, e.g. new_labels=['production'].")

        pid = project_id or self.project_id

        # Resolve prompt_id from cache
        prompt_id: Optional[str] = None
        with self._cache_lock:
            by_label = self._cache.get(name, {})
            for cached in by_label.values():
                if cached.version == version:
                    prompt_id = cached.id
                    break

        # Fallback to list_prompts
        if not prompt_id:
            listing = self.list_prompts(name=name, project_id=pid)
            for item in listing.get("items", []):
                if item.get("version") == version:
                    prompt_id = item.get("id")
                    break

        if not prompt_id:
            raise PromptNotFoundError(f"Prompt '{name}' version {version} not found")

        path = f"/api/managed-prompts/{quote(prompt_id, safe='')}/labels"
        last_response: Dict[str, Any] = {}
        for label in new_labels:
            last_response = self._request_json(
                method="POST",
                path=path,
                json_body={"project_id": pid, "label": label},
            )

        return {"name": name, "version": version, "labels": list(new_labels), **last_response}

    def save_as_version(
        self,
        *,
        prompt_name: str,
        project_id: Optional[str] = None,
        content: Optional[str] = None,
        messages: Optional[Sequence[Dict[str, str]]] = None,
        config: Optional[Mapping[str, Any]] = None,
        commit_message: Optional[str] = None,
        labels: Optional[Sequence[str]] = None,
        tags: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        """
        Save prompt content/messages to managed prompt versions via playground endpoint.

        This route supports API-key auth.
        """
        pid = project_id or self.project_id

        body: Dict[str, Any] = {
            "projectId": pid,
            "promptName": prompt_name,
        }
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

        return self._request_json(
            method="POST",
            path="/api/prompt-playground/save-as-version",
            json_body=body,
        )

    # ----------------
    # Internal helpers
    # ----------------

    def _run_stream_loop(self) -> None:
        reconnect_delay = self.reconnect_initial_seconds

        while not self._stop_event.is_set():
            try:
                self._stream_once()
                reconnect_delay = self.reconnect_initial_seconds
            except Exception as exc:
                with self._connected_lock:
                    self._connected = False

                if not self._stop_event.is_set():
                    self._emit_error(exc)

                if self._stop_event.is_set():
                    break

                time.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2.0, self.reconnect_max_seconds)

    def _stream_once(self) -> None:
        url = f"{self.base_url}/api/managed-prompts/stream"

        headers = self._auth_headers()

        with self._session.get(
            url,
            params={"project_id": self.project_id},
            headers=headers,
            stream=True,
            timeout=(self.connect_timeout_seconds, self.read_timeout_seconds),
        ) as response:
            self._active_stream_response = response

            if response.status_code >= 400:
                body = _safe_response_text(response)
                raise PromptApiError(
                    f"Prompt SSE stream failed ({response.status_code}): {body}"
                )

            with self._connected_lock:
                self._connected = True

            current_event = "message"
            data_lines: List[str] = []

            for raw_line in response.iter_lines(decode_unicode=True):
                if self._stop_event.is_set():
                    return

                if raw_line is None:
                    continue

                line = raw_line.rstrip("\r")

                if not line:
                    self._handle_sse_event(current_event, data_lines)
                    current_event = "message"
                    data_lines = []
                    continue

                if line.startswith(":"):
                    continue

                if line.startswith("event:"):
                    current_event = line[6:].strip() or "message"
                    continue

                if line.startswith("data:"):
                    data_lines.append(line[5:].lstrip())
                    continue

            raise PromptClientError("Prompt SSE stream disconnected")

    def _handle_sse_event(self, event_name: str, data_lines: Sequence[str]) -> None:
        if not data_lines:
            return

        raw_data = "\n".join(data_lines)
        try:
            payload = json.loads(raw_data)
        except json.JSONDecodeError:
            logger.debug("Ignoring non-JSON prompt SSE event: %s", raw_data[:180])
            return

        if event_name == "snapshot":
            if not isinstance(payload, list):
                logger.debug("Ignoring snapshot event with invalid payload type: %s", type(payload))
                return

            prompts = [_normalize_prompt_object(item) for item in payload if isinstance(item, Mapping)]
            with self._cache_lock:
                self._cache = {}
                for prompt in prompts:
                    self._merge_prompt_into_cache_locked(prompt)

            self._snapshot_received.set()
            return

        if isinstance(payload, Mapping) and str(payload.get("type", "")) == "label_moved":
            prompt = _normalize_prompt_event(payload)
            self._merge_prompt_into_cache(prompt)
            return

    def _merge_prompt_into_cache(self, prompt: CachedPrompt) -> None:
        with self._cache_lock:
            self._merge_prompt_into_cache_locked(prompt)

    def _merge_prompt_into_cache_locked(self, prompt: CachedPrompt) -> None:
        name_bucket = self._cache.setdefault(prompt.name, {})
        for label in prompt.labels:
            name_bucket[label] = prompt

    def _emit_error(self, error: Exception) -> None:
        if self.on_error:
            try:
                self.on_error(error)
            except Exception:
                logger.exception("PromptStreamClient on_error callback failed")
        else:
            logger.warning("PromptStreamClient error: %s", error)

    def _auth_headers(self) -> Dict[str, str]:
        headers = {
            "Accept": "text/event-stream, application/json",
            "Authorization": f"Bearer {self.api_key}",
            "x-api-key": self.api_key,
        }

        if self.dev_user_id:
            headers["x-dev-user-id"] = self.dev_user_id

        return headers

    def _request_json(
        self,
        *,
        method: str,
        path: str,
        params: Optional[Mapping[str, Any]] = None,
        json_body: Optional[Mapping[str, Any]] = None,
        timeout_seconds: float = 20.0,
    ) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"

        response = self._session.request(
            method=method,
            url=url,
            params=params,
            json=json_body,
            headers={
                **self._auth_headers(),
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
            timeout=timeout_seconds,
        )

        if response.status_code >= 400:
            body = _safe_response_text(response)
            raise PromptApiError(f"{method} {path} failed ({response.status_code}): {body}")

        try:
            payload = response.json()
        except Exception as exc:
            raise PromptApiError(f"{method} {path} returned non-JSON response") from exc

        if not isinstance(payload, MutableMapping):
            raise PromptApiError(f"{method} {path} returned unexpected response shape")

        return dict(payload)


def _render_template(template: str, variables: Mapping[str, str]) -> str:
    return _PLACEHOLDER_PATTERN.sub(
        lambda match: str(variables.get(match.group(1), match.group(0))),
        template,
    )


def _safe_response_text(response: requests.Response, limit: int = 400) -> str:
    try:
        text = response.text.strip()
    except Exception:
        return "<unavailable>"
    return text[:limit] if text else "<empty>"


def _normalize_prompt_object(raw: Mapping[str, Any]) -> CachedPrompt:
    # Accept both camelCase and snake_case keys for resilience.
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
        config=dict(config),
        labels=labels,
        updated_at=updated_at,
        type=prompt_type,
    )


def _normalize_prompt_event(event: Mapping[str, Any]) -> CachedPrompt:
    labels_raw = event.get("labels")
    if isinstance(labels_raw, list) and labels_raw:
        labels = [str(l) for l in labels_raw if str(l).strip()]
    else:
        single = str(event.get("label", ""))
        labels = [single] if single.strip() else []

    return CachedPrompt(
        id=str(event.get("promptId", "")),
        name=str(event.get("promptName", "")),
        version=int(event.get("version", 0) or 0),
        content=str(event.get("content", "")),
        messages=_coerce_messages(event.get("messages")),
        config=dict(event.get("config") or {}),
        labels=labels,
        updated_at=str(event.get("updatedAt", "")),
        type="text",
    )


def _coerce_messages(raw_messages: Any) -> Optional[List[Dict[str, str]]]:
    if not isinstance(raw_messages, Sequence) or isinstance(raw_messages, (str, bytes, bytearray)):
        return None

    messages: List[Dict[str, str]] = []
    for item in raw_messages:
        if not isinstance(item, Mapping):
            continue
        messages.append(
            {
                "role": str(item.get("role", "system")),
                "content": str(item.get("content", "")),
            }
        )

    return messages or None


def _build_fallback_prompt(
    *,
    name: str,
    label: str,
    fallback: Union[str, Sequence[Dict[str, str]]],
) -> CachedPrompt:
    if isinstance(fallback, str):
        return CachedPrompt(
            id="fallback",
            name=name,
            version=0,
            content=fallback,
            messages=None,
            config={},
            labels=[label],
            updated_at="",
            type="text",
        )

    fallback_messages: List[Dict[str, str]] = []
    for message in fallback:
        if isinstance(message, Mapping):
            fallback_messages.append(
                {
                    "role": str(message.get("role", "system")),
                    "content": str(message.get("content", "")),
                }
            )

    return CachedPrompt(
        id="fallback",
        name=name,
        version=0,
        content="",
        messages=fallback_messages or None,
        config={},
        labels=[label],
        updated_at="",
        type="chat",
    )
