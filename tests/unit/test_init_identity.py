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
    neatlogs.init(
        api_key="second",
        disable_export=True,
        register_shutdown_handlers=False,
    )
    assert neatlogs.shutdown()
