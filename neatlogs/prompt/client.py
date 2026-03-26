from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Union
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
    """Raised when a prompt/label/version is not found and no fallback is provided."""


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
    """Compiled prompt handle returned by PromptClient.get_prompt()."""

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

    @property
    def content(self) -> str:
        return self._prompt.content

    @property
    def messages(self) -> Optional[List[Dict[str, str]]]:
        return list(self._prompt.messages) if self._prompt.messages else None

    def compile(self, variables: Mapping[str, str]) -> str:
        """Compile string content with {{variable}} replacement."""
        if self._prompt.content:
            return _render_template(self._prompt.content, variables)

        if self._prompt.messages:
            rendered = [_render_template(message.get("content", ""), variables) for message in self._prompt.messages]
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


class PromptClient:
    """
    Prompt client for Neatlogs managed prompts.

    Fetches prompts on-demand from the backend (Redis-backed, falls back to Postgres).
    No persistent connection or in-memory cache is maintained.
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        session: Optional[requests.Session] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._session = session or requests.Session()

    def get_prompt(
        self,
        name: str,
        *,
        label: Optional[str] = None,
        version: Optional[int] = None,
        type: str = "text",
    ) -> PromptHandle:
        """
        Fetch a prompt from the backend (Redis → Postgres fallback).

        - Default (no label, no version): returns the most recently created version.
        - label: returns the version holding that label.
        - version: returns that specific version number.
        """
        if label is not None and version is not None:
            raise ValueError("Cannot specify both label and version.")

        if label is not None:
            return PromptHandle(self.fetch_prompt(name, label=label))

        listing = self.list_prompts(name=name)
        items = listing.get("items", [])

        if not items:
            raise PromptNotFoundError(f"No versions found for prompt '{name}'")

        if version is not None:
            for item in items:
                if item.get("version") == version:
                    return PromptHandle(_normalize_prompt_object(item))
            raise PromptNotFoundError(f"Prompt '{name}' version {version} not found")

        # Default: most recently created version
        latest = max(items, key=lambda x: x.get("createdAt") or x.get("created_at") or "")
        return PromptHandle(_normalize_prompt_object(latest))

    # ----------------------------
    # API helpers
    # ----------------------------

    def fetch_prompt(self, name: str, *, label: str) -> CachedPrompt:
        """
        Fetch one prompt by name+label from /api/v1/prompts/:name/fetch.
        Backend checks Redis first, then Postgres.
        """
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
        labels: Sequence[str],
        tags: Optional[Sequence[str]] = None,
        config: Optional[Mapping[str, Any]] = None,
        commit_message: Optional[str] = None,
    ) -> PromptHandle:
        """
        Create a new prompt version via /api/managed-prompts.

        For type="text", prompt must be a str.
        For type="chat", prompt must be a list of {"role", "content"} dicts.
        labels is required — specify at least one label (e.g. "production", "staging").
        """
        if not labels:
            raise ValueError("labels is required. Specify at least one label, e.g. labels=['production'].")
        if type == "text" and not isinstance(prompt, str):
            raise ValueError("For type='text', prompt must be a string.")
        if type == "chat" and not isinstance(prompt, list):
            raise ValueError("For type='chat', prompt must be a list of message dicts.")

        body: Dict[str, Any] = {"name": name, "type": type}
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

        new_labels is required — specify at least one label (e.g. new_labels=["production"]).
        """
        if not new_labels:
            raise ValueError("new_labels is required. Specify at least one label, e.g. new_labels=['production'].")

        listing = self.list_prompts(name=name)
        prompt_id: Optional[str] = None
        for item in listing.get("items", []):
            if item.get("version") == version:
                prompt_id = item.get("id")
                break

        if not prompt_id:
            raise PromptNotFoundError(f"Prompt '{name}' version {version} not found")

        path = f"/api/managed-prompts/{quote(prompt_id, safe='')}/labels"
        last_response: Dict[str, Any] = {}
        for label in new_labels:
            last_response = self._request_json(method="POST", path=path, json_body={"label": label})

        return {"name": name, "version": version, "labels": list(new_labels), **last_response}

    def delete_prompt(
        self,
        name: str,
        version: int,
    ) -> Dict[str, Any]:
        """
        Soft-delete a specific prompt version via DELETE /api/managed-prompts/:promptId.
        """
        listing = self.list_prompts(name=name)
        prompt_id: Optional[str] = None
        for item in listing.get("items", []):
            if item.get("version") == version:
                prompt_id = item.get("id")
                break

        if not prompt_id:
            raise PromptNotFoundError(f"Prompt '{name}' version {version} not found")

        path = f"/api/managed-prompts/{quote(prompt_id, safe='')}"
        return self._request_json(method="DELETE", path=path)

    def remove_tag(
        self,
        name: str,
        version: int,
        tag: str,
    ) -> Dict[str, Any]:
        """
        Remove a tag from a prompt version via DELETE /api/managed-prompts/:promptId/tags.
        """
        listing = self.list_prompts(name=name)
        prompt_id: Optional[str] = None
        for item in listing.get("items", []):
            if item.get("version") == version:
                prompt_id = item.get("id")
                break

        if not prompt_id:
            raise PromptNotFoundError(f"Prompt '{name}' version {version} not found")

        path = f"/api/managed-prompts/{quote(prompt_id, safe='')}/tags"
        return self._request_json(method="DELETE", path=path, json_body={"tag": tag})

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

        return self._request_json(method="POST", path="/api/prompt-playground/save-as-version", json_body=body)

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
        timeout_seconds: float = 20.0,
    ) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"

        response = self._session.request(
            method=method,
            url=url,
            params=params,
            json=json_body,
            headers={**self._auth_headers(), "Content-Type": "application/json"},
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


# ---------------------------------------------------------------------------
# Module-level prompt API — credentials sourced from neatlogs.init()
# ---------------------------------------------------------------------------

_shared_client: Optional[PromptClient] = None


def _get_shared_client() -> PromptClient:
    global _shared_client
    if _shared_client is not None:
        return _shared_client

    from ..init import _session_config

    api_key = _session_config.get("_api_key") or ""
    base_url = _session_config.get("_base_url") or ""

    if not api_key or api_key == "disabled":
        raise PromptClientError(
            "No API key available. Call neatlogs.init(api_key=...) before using prompt methods."
        )

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
    return _get_shared_client().list_prompts(name=name, source=source, label=label, limit=limit, offset=offset)


def create_prompt(
    *,
    name: str,
    prompt: Union[str, Sequence[Dict[str, str]]],
    type: str = "text",
    labels: Sequence[str],
    tags: Optional[Sequence[str]] = None,
    config: Optional[Mapping[str, Any]] = None,
    commit_message: Optional[str] = None,
) -> PromptHandle:
    return _get_shared_client().create_prompt(
        name=name, prompt=prompt, type=type, labels=labels,
        tags=tags, config=config, commit_message=commit_message,
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
        prompt_name=prompt_name, content=content, messages=messages,
        config=config, commit_message=commit_message, labels=labels, tags=tags,
    )


def delete_prompt(name: str, version: int) -> Dict[str, Any]:
    return _get_shared_client().delete_prompt(name, version)


def remove_tag(name: str, version: int, tag: str) -> Dict[str, Any]:
    return _get_shared_client().remove_tag(name, version, tag)
