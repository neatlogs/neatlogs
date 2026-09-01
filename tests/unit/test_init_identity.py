import pytest

import neatlogs
from neatlogs.errors import NeatlogsConfigurationError


def test_identical_init_is_idempotent_but_conflicting_init_is_typed_error():
    neatlogs.init(
        api_key="key-a",
        workflow_name="identity",
        disable_export=True,
        register_shutdown_handlers=False,
    )
    neatlogs.init(
        api_key="key-a",
        workflow_name="identity",
        disable_export=True,
        register_shutdown_handlers=False,
    )

    with pytest.raises(NeatlogsConfigurationError, match="different configuration"):
        neatlogs.init(
            api_key="key-b",
            workflow_name="identity",
            disable_export=True,
            register_shutdown_handlers=False,
        )

    assert neatlogs.shutdown()


def test_explicit_shutdown_allows_a_new_configuration_generation():
    neatlogs.init(
        api_key="first",
        disable_export=True,
        register_shutdown_handlers=False,
    )
    assert neatlogs.shutdown()


def test_uploads_are_default_off_and_explicit_opt_in_is_visible_in_diagnostics(
    monkeypatch,
):
    import importlib

    init_module = importlib.import_module("neatlogs.init")

    class Authority:
        available = True
        unavailable_reason = ""
        max_upload_bytes = 25 * 1024 * 1024
        closed = False

        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def close(self):
            self.closed = True

    monkeypatch.setenv("NEATLOGS_DISABLE_EXPORT", "false")
    monkeypatch.delenv("NEATLOGS_UPLOADS_ENABLED", raising=False)
    monkeypatch.setattr(init_module, "AuthenticatedUploadAuthority", Authority)

    neatlogs.init(
        api_key="project-key",
        workflow_name="uploads-off",
        instrumentations=[],
        register_shutdown_handlers=False,
    )
    assert neatlogs.get_delivery_diagnostics()["span_upload_authority_available"] is False
    assert neatlogs.shutdown()

    neatlogs.init(
        api_key="project-key",
        workflow_name="uploads-on",
        instrumentations=[],
        uploads_enabled=True,
        register_shutdown_handlers=False,
    )
    authority = init_module._upload_authority
    assert authority.kwargs == {
        "base_url": "https://ingest.neatlogs.com",
        "api_key": "project-key",
    }
    assert neatlogs.get_delivery_diagnostics()["span_upload_authority_available"] is True
    assert neatlogs.shutdown()
    assert authority.closed is True
    neatlogs.init(
        api_key="second",
        disable_export=True,
        register_shutdown_handlers=False,
    )
    assert neatlogs.shutdown()
