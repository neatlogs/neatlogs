from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence
from urllib.parse import quote

import requests

from ..core.logger import get_logger

logger = get_logger()

# ---------------------------------------------------------------------------
# OpenTelemetry imports — guarded once at module load time so we don't pay the
# import cost on every HTTP call. Either both imports succeed (instrumentation
# is suppressed around each request) or both are None (instrumentation is not
# installed and requests go out as-is).
# ---------------------------------------------------------------------------
try:
    from opentelemetry.context import attach as _otel_attach
    from opentelemetry.context import detach as _otel_detach
    from opentelemetry.context import set_value as _otel_set_value
    from opentelemetry.instrumentation.utils import _SUPPRESS_INSTRUMENTATION_KEY
except Exception:  # pragma: no cover - depends on optional dependency
    _otel_attach = None
    _otel_detach = None
    _otel_set_value = None
    _SUPPRESS_INSTRUMENTATION_KEY = None


class ConfigClientError(Exception):
    """Base exception for config client failures."""


class ConfigApiError(ConfigClientError):
    """Raised when the backend returns an API error."""


class ConfigConflictError(ConfigApiError):
    """Raised when the backend returns 409 (e.g. duplicate config name)."""


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
    return next((raw[k] for k in keys if raw.get(k) is not None), None)


def _as_optional_str(value: Any) -> Optional[str]:
    return None if value is None else str(value)


def _as_optional_number(value: Any, cast):
    if value is None:
        return None
    try:
        return cast(value)
    except (TypeError, ValueError):
        return None


def _normalize_config_object(data: Mapping[str, Any]) -> CachedConfig:
    """Normalize API response (camelCase) to CachedConfig (snake_case)."""
    raw_labels = data.get("labels")
    # JSON arrays always decode as `list`; guard narrowly to avoid the
    # `str is Sequence` gotcha.
    labels: List[str] = (
        [str(label) for label in raw_labels if str(label).strip()]
        if isinstance(raw_labels, list)
        else []
    )

    config_id = data.get("id")
    config_id = str(config_id) if config_id is not None else ""

    name = data.get("name")
    name = str(name) if name is not None else ""

    created_at = _get_first(data, "createdAt", "created_at")
    updated_at = _get_first(data, "updatedAt", "updated_at")

    return CachedConfig(
        id=config_id,
        name=name,
        provider=_as_optional_str(data.get("provider")),
        model=_as_optional_str(data.get("model")),
        temperature=_as_optional_number(data.get("temperature"), float),
        max_tokens=_as_optional_number(_get_first(data, "maxTokens", "max_tokens"), int),
        top_p=_as_optional_number(_get_first(data, "topP", "top_p"), float),
        top_k=_as_optional_number(_get_first(data, "topK", "top_k"), int),
        description=_as_optional_str(data.get("description")),
        labels=labels,
        created_at=str(created_at) if created_at is not None else "",
        updated_at=str(updated_at) if updated_at is not None else "",
    )


# ---------------------------------------------------------------------------
# Inference-parameter body builder
# ---------------------------------------------------------------------------

_SENTINEL = object()  # distinguishes "not passed" from None


def _build_inference_body(
    *,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    description: Optional[str] = None,
    labels=_SENTINEL,
) -> Dict[str, Any]:
    """Build the snake_case request body for create/update, skipping None fields.

    `labels` is included only when explicitly passed; passing `labels=[]`
    sends an empty array (clears all labels).
    """
    values = {
        "provider": provider,
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
        "top_k": top_k,
        "description": description,
    }
    body: Dict[str, Any] = {k: v for k, v in values.items() if v is not None}
    if labels is not _SENTINEL and labels is not None:
        body["labels"] = list(labels)
    return body


class ConfigClient:
    """
    Config client for Neatlogs prompt configs.

    Provides CRUD over /api/prompt-configs:
      - create_config  — create a new config
      - get_config     — fetch by name (optionally filtered by label)
      - list_configs   — list with optional name/label filters
      - update_config  — update fields; pass labels=[...] to replace the labels array
      - delete_config  — delete a config
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
        """Create a new config via POST /api/prompt-configs.

        Raises :class:`ConfigConflictError` (HTTP 409) if a config with this
        name already exists in the project.
        """
        body: Dict[str, Any] = {
            "name": name,
            **_build_inference_body(
                provider=provider,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                top_k=top_k,
                description=description,
                labels=labels if labels is not None else _SENTINEL,
            ),
        }

        payload = self._request_json(method="POST", path="/api/prompt-configs", json_body=body)
        return _normalize_config_object(payload)

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
        labels=_SENTINEL,
    ) -> CachedConfig:
        """Update fields on a config via PATCH /api/prompt-configs/:id.

        Pass ``labels=[...]`` to replace the entire labels array.
        Pass ``labels=[]`` to clear all labels.
        Omit ``labels`` to leave the existing labels unchanged.
        """
        body = _build_inference_body(
            provider=provider,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            top_k=top_k,
            description=description,
            labels=labels,
        )

        # If no fields are provided, short-circuit to avoid an empty PATCH.
        if not body:
            return self.get_config(name)

        config_id = self._resolve_config_id(name)
        path = f"/api/prompt-configs/{quote(config_id, safe='')}"
        payload = self._request_json(method="PATCH", path=path, json_body=body)
        return _normalize_config_object(payload)

    def delete_config(self, name: str) -> Dict[str, Any]:
        """Delete a config via DELETE /api/prompt-configs/:id."""
        config_id = self._resolve_config_id(name)
        path = f"/api/prompt-configs/{quote(config_id, safe='')}"
        return self._request_json(method="DELETE", path=path)

    # ----------------
    # Internal helpers
    # ----------------

    def _resolve_config_id(self, name: str) -> str:
        """Resolve a config name to its UUID. Raises ConfigNotFoundError if missing."""
        return self.get_config(name).id

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

        _token = None
        if _otel_attach is not None and _SUPPRESS_INSTRUMENTATION_KEY is not None:
            try:
                _token = _otel_attach(_otel_set_value(_SUPPRESS_INSTRUMENTATION_KEY, True))
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
            if _token is not None and _otel_detach is not None:
                try:
                    _otel_detach(_token)
                except Exception:
                    pass

        if response.status_code == 404:
            body = _safe_response_text(response)
            raise ConfigNotFoundError(f"{method} {path} not found (404): {body}")

        if response.status_code == 409:
            body = _safe_response_text(response)
            raise ConfigConflictError(f"{method} {path} conflict (409): {body}")

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

    # Re-create the client if credentials have changed (e.g., after shutdown() + init()
    # with a different api_key). Without this check the stale client from the previous
    # init() call would be silently reused.
    if _shared_config_client is None or (
        _shared_config_client.api_key != api_key or _shared_config_client.base_url != base_url
    ):
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
    labels=_SENTINEL,
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
        labels=labels,
    )


def delete_config(name: str) -> Dict[str, Any]:
    return _get_config_client().delete_config(name)
