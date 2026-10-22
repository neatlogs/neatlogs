import pytest

import neatlogs
from neatlogs.constants import DEFAULT_INGEST_ENDPOINT
from neatlogs.errors import NeatlogsConfigurationError
from neatlogs.init import _resolve_init_endpoint


@pytest.fixture(autouse=True)
def _clean_endpoint_env(monkeypatch):
    monkeypatch.delenv("NEATLOGS_ENDPOINT", raising=False)
    yield


def test_explicit_endpoint_wins_over_env(monkeypatch):
    monkeypatch.setenv("NEATLOGS_ENDPOINT", "https://dev-cloud.neatlogs.com")
    assert _resolve_init_endpoint("https://staging.neatlogs.com") == "https://staging.neatlogs.com"


def test_env_used_when_no_explicit_endpoint(monkeypatch):
    monkeypatch.setenv("NEATLOGS_ENDPOINT", "https://dev-cloud.neatlogs.com")
    assert _resolve_init_endpoint(None) == "https://dev-cloud.neatlogs.com"
    assert _resolve_init_endpoint("") == "https://dev-cloud.neatlogs.com"
    assert _resolve_init_endpoint("   ") == "https://dev-cloud.neatlogs.com"


def test_default_when_no_explicit_or_env():
    assert _resolve_init_endpoint(None) == DEFAULT_INGEST_ENDPOINT
    assert _resolve_init_endpoint("") == DEFAULT_INGEST_ENDPOINT


def test_init_with_env_endpoint_is_idempotent_with_explicit_same(monkeypatch):
    monkeypatch.setenv("NEATLOGS_ENDPOINT", "https://dev-cloud.neatlogs.com")
    neatlogs.init(
        api_key="key-a",
        workflow_name="endpoint-env",
        disable_export=True,
        register_shutdown_handlers=False,
    )
    # Same effective configuration via explicit endpoint: must not raise.
    neatlogs.init(
        api_key="key-a",
        endpoint="https://dev-cloud.neatlogs.com",
        workflow_name="endpoint-env",
        disable_export=True,
        register_shutdown_handlers=False,
    )
    assert neatlogs.shutdown()


def test_init_conflicting_endpoint_still_typed_error(monkeypatch):
    monkeypatch.setenv("NEATLOGS_ENDPOINT", "https://dev-cloud.neatlogs.com")
    neatlogs.init(
        api_key="key-a",
        workflow_name="endpoint-env",
        disable_export=True,
        register_shutdown_handlers=False,
    )
    with pytest.raises(NeatlogsConfigurationError, match="different configuration"):
        neatlogs.init(
            api_key="key-a",
            endpoint="https://staging.neatlogs.com",
            workflow_name="endpoint-env",
            disable_export=True,
            register_shutdown_handlers=False,
        )
    assert neatlogs.shutdown()


def test_init_without_env_still_defaults_to_prod():
    neatlogs.init(
        api_key="key-a",
        workflow_name="endpoint-env-default",
        disable_export=True,
        register_shutdown_handlers=False,
    )
    neatlogs.init(
        api_key="key-a",
        endpoint=DEFAULT_INGEST_ENDPOINT,
        workflow_name="endpoint-env-default",
        disable_export=True,
        register_shutdown_handlers=False,
    )
    assert neatlogs.shutdown()
