"""Unit tests for neatlogs.config.client."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests

import neatlogs
from neatlogs.config.client import (
    CachedConfig,
    ConfigApiError,
    ConfigClient,
    ConfigClientError,
    ConfigNotFoundError,
    _normalize_config_object,
)
import neatlogs.config.client as config_client_module


BASE_URL = "https://api.example.com"
API_KEY = "test-api-key"


def _mock_response(*, status_code: int = 200, json_body=None, text: str = "") -> MagicMock:
    response = MagicMock(spec=requests.Response)
    response.status_code = status_code
    response.json.return_value = json_body if json_body is not None else {}
    response.text = text
    return response


def _make_client() -> ConfigClient:
    return ConfigClient(base_url=BASE_URL, api_key=API_KEY)


def _sample_config_payload(**overrides):
    base = {
        "id": "cfg-1",
        "name": "foo",
        "provider": "openai",
        "model": "gpt-4",
        "temperature": 0.7,
        "maxTokens": 1024,
        "topP": 0.9,
        "topK": 40,
        "description": "test config",
        "labels": ["production"],
        "createdAt": "2024-01-01T00:00:00Z",
        "updatedAt": "2024-01-02T00:00:00Z",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_get_config_by_name():
    client = _make_client()
    payload = {"items": [_sample_config_payload()]}

    with patch("requests.Session.request") as mock_request:
        mock_request.return_value = _mock_response(json_body=payload)
        result = client.get_config("foo")

    assert isinstance(result, CachedConfig)
    assert result.id == "cfg-1"
    assert result.name == "foo"
    assert result.provider == "openai"
    assert result.model == "gpt-4"
    assert result.temperature == 0.7
    assert result.max_tokens == 1024
    assert result.top_p == 0.9
    assert result.top_k == 40
    assert result.labels == ["production"]

    mock_request.assert_called_once()
    call = mock_request.call_args
    assert call.kwargs["method"] == "GET"
    assert call.kwargs["url"] == f"{BASE_URL}/api/prompt-configs"
    assert call.kwargs["params"]["name"] == "foo"


def test_get_config_by_label():
    client = _make_client()
    payload = _sample_config_payload()

    with patch("requests.Session.request") as mock_request:
        mock_request.return_value = _mock_response(json_body=payload)
        result = client.get_config("foo", label="production")

    assert isinstance(result, CachedConfig)
    assert result.name == "foo"
    assert result.max_tokens == 1024

    call = mock_request.call_args
    assert call.kwargs["method"] == "GET"
    assert call.kwargs["url"] == f"{BASE_URL}/api/v1/configs/foo/fetch"
    assert call.kwargs["params"] == {"label": "production"}


def test_get_config_not_found_raises():
    client = _make_client()

    with patch("requests.Session.request") as mock_request:
        mock_request.return_value = _mock_response(status_code=404, text="not found")
        with pytest.raises(ConfigNotFoundError):
            client.get_config("missing", label="production")


def test_list_configs_pagination():
    client = _make_client()

    with patch("requests.Session.request") as mock_request:
        mock_request.return_value = _mock_response(json_body={"items": [], "total": 0})
        client.list_configs(limit=25, offset=50)

    call = mock_request.call_args
    assert call.kwargs["method"] == "GET"
    assert call.kwargs["url"] == f"{BASE_URL}/api/prompt-configs"
    assert call.kwargs["params"]["limit"] == 25
    assert call.kwargs["params"]["offset"] == 50
    # name/label omitted when None
    assert "name" not in call.kwargs["params"]
    assert "label" not in call.kwargs["params"]


def test_create_config():
    client = _make_client()
    payload = _sample_config_payload()

    with patch("requests.Session.request") as mock_request:
        mock_request.return_value = _mock_response(json_body=payload)
        result = client.create_config(
            "foo",
            provider="openai",
            model="gpt-4",
            temperature=0.7,
            # max_tokens intentionally omitted → must NOT be in body
            top_p=0.9,
            labels=["production"],
        )

    assert isinstance(result, CachedConfig)
    assert result.name == "foo"
    assert result.provider == "openai"

    call = mock_request.call_args
    assert call.kwargs["method"] == "POST"
    assert call.kwargs["url"] == f"{BASE_URL}/api/prompt-configs"
    body = call.kwargs["json"]
    assert body["name"] == "foo"
    assert body["provider"] == "openai"
    assert body["model"] == "gpt-4"
    assert body["temperature"] == 0.7
    assert body["top_p"] == 0.9
    assert body["labels"] == ["production"]
    # None kwargs must be absent
    assert "max_tokens" not in body
    assert "top_k" not in body
    assert "description" not in body


def test_update_config_fields():
    client = _make_client()
    list_payload = {"items": [_sample_config_payload()]}
    patch_payload = _sample_config_payload(model="gpt-4-turbo")

    with patch("requests.Session.request") as mock_request:
        mock_request.side_effect = [
            _mock_response(json_body=list_payload),  # get_config() list call
            _mock_response(json_body=patch_payload),  # PATCH call
        ]
        result = client.update_config("foo", model="gpt-4-turbo", temperature=0.5)

    assert isinstance(result, CachedConfig)
    assert result.model == "gpt-4-turbo"

    assert mock_request.call_count == 2
    patch_call = mock_request.call_args_list[1]
    assert patch_call.kwargs["method"] == "PATCH"
    assert patch_call.kwargs["url"] == f"{BASE_URL}/api/prompt-configs/cfg-1"
    body = patch_call.kwargs["json"]
    assert body == {"model": "gpt-4-turbo", "temperature": 0.5}
    # None-only kwargs must be absent
    assert "provider" not in body
    assert "max_tokens" not in body
    assert "top_p" not in body
    assert "top_k" not in body
    assert "description" not in body


def test_update_config_no_fields_returns_existing():
    client = _make_client()
    list_payload = {"items": [_sample_config_payload()]}

    with patch("requests.Session.request") as mock_request:
        mock_request.return_value = _mock_response(json_body=list_payload)
        result = client.update_config("foo")

    assert isinstance(result, CachedConfig)
    assert result.id == "cfg-1"
    # Only the list/get call, no PATCH
    assert mock_request.call_count == 1
    assert mock_request.call_args.kwargs["method"] == "GET"


def test_set_config_labels():
    client = _make_client()
    list_payload = {"items": [_sample_config_payload()]}
    post_response = {"ok": True}

    with patch("requests.Session.request") as mock_request:
        mock_request.side_effect = [
            _mock_response(json_body=list_payload),  # get_config list
            _mock_response(json_body=post_response),  # POST label
        ]
        result = client.set_config_labels("foo", new_labels=["staging"])

    assert result["name"] == "foo"
    assert result["labels"] == ["staging"]
    assert result["ok"] is True

    post_call = mock_request.call_args_list[1]
    assert post_call.kwargs["method"] == "POST"
    assert post_call.kwargs["url"] == f"{BASE_URL}/api/prompt-configs/cfg-1/labels"
    assert post_call.kwargs["json"] == {"label": "staging"}


def test_set_config_labels_empty_raises():
    client = _make_client()
    with pytest.raises(ValueError):
        client.set_config_labels("foo", new_labels=[])


def test_delete_config():
    client = _make_client()
    list_payload = {"items": [_sample_config_payload()]}
    delete_payload = {"deleted": True}

    with patch("requests.Session.request") as mock_request:
        mock_request.side_effect = [
            _mock_response(json_body=list_payload),
            _mock_response(json_body=delete_payload),
        ]
        result = client.delete_config("foo")

    assert result == {"deleted": True}
    delete_call = mock_request.call_args_list[1]
    assert delete_call.kwargs["method"] == "DELETE"
    assert delete_call.kwargs["url"] == f"{BASE_URL}/api/prompt-configs/cfg-1"


def test_remove_config_label():
    client = _make_client()
    list_payload = {"items": [_sample_config_payload()]}
    remove_payload = {"ok": True}

    with patch("requests.Session.request") as mock_request:
        mock_request.side_effect = [
            _mock_response(json_body=list_payload),
            _mock_response(json_body=remove_payload),
        ]
        result = client.remove_config_label("foo", "staging")

    assert result == {"ok": True}
    delete_call = mock_request.call_args_list[1]
    assert delete_call.kwargs["method"] == "DELETE"
    assert delete_call.kwargs["url"] == f"{BASE_URL}/api/prompt-configs/cfg-1/labels"
    assert delete_call.kwargs["json"] == {"label": "staging"}


def test_api_error_on_500():
    client = _make_client()

    with patch("requests.Session.request") as mock_request:
        mock_request.return_value = _mock_response(status_code=500, text="server error")
        with pytest.raises(ConfigApiError):
            client.list_configs()


def test_not_initialized_raises():
    # Ensure no cached module-level client
    config_client_module._shared_config_client = None
    # Ensure init session_config does not already have an api_key.
    # Note: neatlogs.init is the function (shadows the submodule), so grab
    # the real submodule via sys.modules.
    import sys

    init_module = sys.modules["neatlogs.init"]
    original_api_key = init_module._session_config.get("_api_key")
    original_base_url = init_module._session_config.get("_base_url")
    init_module._session_config["_api_key"] = None
    init_module._session_config["_base_url"] = None

    try:
        with pytest.raises(ConfigClientError):
            neatlogs.get_config("foo")
    finally:
        init_module._session_config["_api_key"] = original_api_key
        init_module._session_config["_base_url"] = original_base_url
        config_client_module._shared_config_client = None


def test_normalize_camelcase():
    data = {
        "id": "cfg-9",
        "name": "bar",
        "provider": "anthropic",
        "model": "claude-3",
        "temperature": 0.2,
        "maxTokens": 2048,
        "topP": 0.8,
        "topK": 20,
        "description": "desc",
        "labels": ["production", "canary"],
        "createdAt": "2024-05-01T00:00:00Z",
        "updatedAt": "2024-05-02T00:00:00Z",
    }

    cfg = _normalize_config_object(data)

    assert cfg.id == "cfg-9"
    assert cfg.name == "bar"
    assert cfg.provider == "anthropic"
    assert cfg.model == "claude-3"
    assert cfg.temperature == 0.2
    assert cfg.max_tokens == 2048
    assert cfg.top_p == 0.8
    assert cfg.top_k == 20
    assert cfg.description == "desc"
    assert cfg.labels == ["production", "canary"]
    assert cfg.created_at == "2024-05-01T00:00:00Z"
    assert cfg.updated_at == "2024-05-02T00:00:00Z"


def test_reinit_with_new_api_key_creates_new_client():
    """After shutdown() + init() with a different key, module-level functions must
    use the NEW key, not the cached client from the previous init().

    Regression test for: _shared_config_client is never invalidated when credentials
    change between init() calls (e.g. key rotation or project switch).
    """
    import sys

    init_module = sys.modules["neatlogs.init"]
    original_api_key = init_module._session_config.get("_api_key")
    original_base_url = init_module._session_config.get("_base_url")
    original_client = config_client_module._shared_config_client

    try:
        # Simulate first init() → client created with key-A
        init_module._session_config["_api_key"] = "key-A"
        init_module._session_config["_base_url"] = "https://api.example.com"
        config_client_module._shared_config_client = None  # fresh start

        payload = {"items": [{"id": "c1", "name": "foo", "labels": [], "createdAt": "t", "updatedAt": "t"}]}
        with patch("requests.Session.request") as mock_request:
            mock_request.return_value = MagicMock(
                spec=requests.Response,
                status_code=200,
                json=lambda: payload,
                text="",
            )
            neatlogs.get_config("foo")
        client_a = config_client_module._shared_config_client
        assert client_a is not None
        assert client_a.api_key == "key-A"

        # Simulate shutdown() + init() with key-B
        init_module._session_config["_api_key"] = "key-B"
        init_module._session_config["_base_url"] = "https://api.example.com"
        # _shared_config_client is still the stale key-A client — this is the bug scenario

        with patch("requests.Session.request") as mock_request:
            mock_request.return_value = MagicMock(
                spec=requests.Response,
                status_code=200,
                json=lambda: payload,
                text="",
            )
            neatlogs.get_config("foo")
        client_b = config_client_module._shared_config_client

        # After the fix: a NEW client must be created with key-B
        assert client_b is not client_a, "must create a new client after key rotation"
        assert client_b.api_key == "key-B", (
            f"new client must use key-B, got {client_b.api_key!r}"
        )
    finally:
        init_module._session_config["_api_key"] = original_api_key
        init_module._session_config["_base_url"] = original_base_url
        config_client_module._shared_config_client = original_client
