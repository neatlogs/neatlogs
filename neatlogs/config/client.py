from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence
from urllib.parse import quote

import requests

from ..core.logger import get_logger

logger = get_logger()


class ConfigClientError(Exception):
    """Base exception for config client failures."""


class ConfigApiError(ConfigClientError):
    """Raised when the backend returns an API error."""


class ConfigNotFoundError(ConfigClientError):
    """Raised when a config is not found."""


@dataclass(frozen=True)
class CachedConfig:
    id: str
    name: str
    provider: Optional[str]
    model: Optional[str]
    temperature: Optional[float]
    max_tokens: Optional[int]
    top_p: Optional[float]
    top_k: Optional[int]
    description: Optional[str]
    labels: List[str]
    created_at: str
    updated_at: str


def _get_first(raw: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in raw and raw[key] is not None:
            return raw[key]
    return None


def _as_optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    return str(value)


def _as_optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_config_object(data: Mapping[str, Any]) -> CachedConfig:
    """Normalize API response (camelCase) to CachedConfig (snake_case)."""
    raw_labels = data.get("labels")
    labels: List[str] = []
    if isinstance(raw_labels, Sequence) and not isinstance(raw_labels, (str, bytes, bytearray)):
        labels = [str(label) for label in raw_labels if str(label).strip()]

    config_id = data.get("id")
    if not isinstance(config_id, str):
        config_id = str(config_id) if config_id is not None else ""

    name = data.get("name")
    if not isinstance(name, str):
        name = str(name) if name is not None else ""

    created_at = _get_first(data, "createdAt", "created_at")
    updated_at = _get_first(data, "updatedAt", "updated_at")

    return CachedConfig(
        id=config_id,
        name=name,
        provider=_as_optional_str(data.get("provider")),
        model=_as_optional_str(data.get("model")),
        temperature=_as_optional_float(data.get("temperature")),
        max_tokens=_as_optional_int(_get_first(data, "maxTokens", "max_tokens")),
        top_p=_as_optional_float(_get_first(data, "topP", "top_p")),
        top_k=_as_optional_int(_get_first(data, "topK", "top_k")),
        description=_as_optional_str(data.get("description")),
        labels=labels,
        created_at=str(created_at) if created_at is not None else "",
        updated_at=str(updated_at) if updated_at is not None else "",
    )


class ConfigClient:
    """
    Config client for Neatlogs prompt configs.

    Provides full CRUD over /api/prompt-configs and the label-fetch endpoint
    /api/v1/configs/:name/fetch.
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

    # ----------------------------
    # Public API
    # ----------------------------

    def get_config(self, name: str, *, label: Optional[str] = None) -> CachedConfig:
        """Fetch a config by name. If label is given, uses the fetch-by-label endpoint."""
        if label is not None:
            path = f"/api/v1/configs/{quote(name, safe='')}/fetch"
            payload = self._request_json(method="GET", path=path, params={"label": label})
            return _normalize_config_object(payload)

        listing = self.list_configs(name=name)
        items = listing.get("items", [])
        if not items:
            raise ConfigNotFoundError(f"No config found with name '{name}'")
        return _normalize_config_object(items[0])

    def list_configs(
        self,
        *,
        name: Optional[str] = None,
        label: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """List configs from /api/prompt-configs."""
        params: Dict[str, Any] = {
            "limit": max(1, min(limit, 500)),
            "offset": max(0, offset),
        }
        if name is not None:
            params["name"] = name
        if label is not None:
            params["label"] = label

        return self._request_json(method="GET", path="/api/prompt-configs", params=params)

    def create_config(
        self,
        name: str,
        *,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        description: Optional[str] = None,
        labels: Optional[Sequence[str]] = None,
    ) -> CachedConfig:
        """Create a new config via POST /api/prompt-configs."""
        body: Dict[str, Any] = {"name": name}
        if provider is not None:
            body["provider"] = provider
        if model is not None:
            body["model"] = model
        if temperature is not None:
            body["temperature"] = temperature
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if top_p is not None:
            body["top_p"] = top_p
        if top_k is not None:
            body["top_k"] = top_k
        if description is not None:
            body["description"] = description
        if labels is not None:
            body["labels"] = list(labels)

        payload = self._request_json(method="POST", path="/api/prompt-configs", json_body=body)
        return _normalize_config_object(payload.get("config", payload))

    def update_config(
        self,
        name: str,
        *,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        description: Optional[str] = None,
    ) -> CachedConfig:
        """Update fields on a config via PATCH /api/prompt-configs/:id."""
        body: Dict[str, Any] = {}
        if provider is not None:
            body["provider"] = provider
        if model is not None:
            body["model"] = model
        if temperature is not None:
            body["temperature"] = temperature
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if top_p is not None:
            body["top_p"] = top_p
        if top_k is not None:
            body["top_k"] = top_k
        if description is not None:
            body["description"] = description

        # If no fields are provided, short-circuit to avoid an empty PATCH.
        if not body:
            return self.get_config(name)

        existing = self.get_config(name)
        path = f"/api/prompt-configs/{quote(existing.id, safe='')}"
        payload = self._request_json(method="PATCH", path=path, json_body=body)
        return _normalize_config_object(payload.get("config", payload))

    def set_config_labels(
        self,
        name: str,
        *,
        new_labels: Sequence[str],
    ) -> Dict[str, Any]:
        """Attach one or more labels to a config via POST /api/prompt-configs/:id/labels."""
        if not new_labels:
            raise ValueError(
                "new_labels is required. Specify at least one label, e.g. new_labels=['production']."
            )

        existing = self.get_config(name)
        path = f"/api/prompt-configs/{quote(existing.id, safe='')}/labels"
        last_response: Dict[str, Any] = {}
        for label in new_labels:
            last_response = self._request_json(method="POST", path=path, json_body={"label": label})

        return {"name": name, "labels": list(new_labels), **last_response}

    def delete_config(self, name: str) -> Dict[str, Any]:
        """Delete a config via DELETE /api/prompt-configs/:id."""
        existing = self.get_config(name)
        path = f"/api/prompt-configs/{quote(existing.id, safe='')}"
        return self._request_json(method="DELETE", path=path)

    def remove_config_label(self, name: str, label: str) -> Dict[str, Any]:
        """Remove a label from a config via DELETE /api/prompt-configs/:id/labels."""
        existing = self.get_config(name)
        path = f"/api/prompt-configs/{quote(existing.id, safe='')}/labels"
        return self._request_json(method="DELETE", path=path, json_body={"label": label})

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

        try:
            from opentelemetry.context import attach, set_value
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
                    timeout=timeout_seconds,
                )
            except requests.RequestException as exc:
                raise ConfigClientError(f"{method} {path} request failed: {exc}") from exc
        finally:
            if _token is not None:
                try:
                    from opentelemetry.context import detach

                    detach(_token)
                except Exception:
                    pass

        if response.status_code == 404:
            body = _safe_response_text(response)
            raise ConfigNotFoundError(f"{method} {path} not found (404): {body}")

        if response.status_code >= 400:
            body = _safe_response_text(response)
            raise ConfigApiError(f"{method} {path} failed ({response.status_code}): {body}")

        try:
            payload = response.json()
        except Exception as exc:
            raise ConfigApiError(f"{method} {path} returned non-JSON response") from exc

        if not isinstance(payload, MutableMapping):
            raise ConfigApiError(f"{method} {path} returned unexpected response shape")

        return dict(payload)


def _safe_response_text(response: requests.Response, limit: int = 400) -> str:
    try:
        text = response.text.strip()
    except Exception:
        return "<unavailable>"
    return text[:limit] if text else "<empty>"


# ---------------------------------------------------------------------------
# Module-level config API — credentials sourced from neatlogs.init()
# ---------------------------------------------------------------------------

_shared_config_client: Optional[ConfigClient] = None


def _get_config_client() -> ConfigClient:
    global _shared_config_client

    from ..init import _session_config

    api_key = _session_config.get("_api_key") or ""
    base_url = _session_config.get("_base_url") or ""

    if not api_key or api_key == "disabled":
        raise ConfigClientError(
            "No API key available. Call neatlogs.init(api_key=...) before using config methods."
        )

    if _shared_config_client is None:
        _shared_config_client = ConfigClient(base_url=base_url, api_key=api_key)
    return _shared_config_client


def get_config(name: str, *, label: Optional[str] = None) -> CachedConfig:
    return _get_config_client().get_config(name, label=label)


def list_configs(
    *,
    name: Optional[str] = None,
    label: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
) -> Dict[str, Any]:
    return _get_config_client().list_configs(name=name, label=label, limit=limit, offset=offset)


def create_config(
    name: str,
    *,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    description: Optional[str] = None,
    labels: Optional[Sequence[str]] = None,
) -> CachedConfig:
    return _get_config_client().create_config(
        name,
        provider=provider,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        top_k=top_k,
        description=description,
        labels=labels,
    )


def update_config(
    name: str,
    *,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    description: Optional[str] = None,
) -> CachedConfig:
    return _get_config_client().update_config(
        name,
        provider=provider,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        top_k=top_k,
        description=description,
    )


def set_config_labels(name: str, *, new_labels: Sequence[str]) -> Dict[str, Any]:
    return _get_config_client().set_config_labels(name, new_labels=new_labels)


def delete_config(name: str) -> Dict[str, Any]:
    return _get_config_client().delete_config(name)


def remove_config_label(name: str, label: str) -> Dict[str, Any]:
    return _get_config_client().remove_config_label(name, label)
