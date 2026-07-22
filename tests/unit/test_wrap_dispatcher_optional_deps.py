"""
Regression tests for ``neatlogs.wrap()`` dispatching behavior with optional
provider dependencies.

The dispatcher in ``neatlogs/__init__.py`` matches incoming client objects
against well-known provider class names (``OpenAI``, ``AsyncOpenAI``,
``Anthropic``, ``Crew``, ``Flow``, ...) and lazily imports the matching
provider wrapper module. Each provider is an *optional* extra — ``openai``,
``anthropic``, ``crewai``, etc. are declared in ``pyproject.toml`` only as
installable extras — so the lazy import must not crash the dispatch when the
underlying package isn't installed.

The bug we are guarding against: with ``openai`` uninstalled, calling
``neatlogs.wrap(SomeUserDefinedClassNamedOpenAI(...), project_id="...")``
used to raise ``ModuleNotFoundError: No module named 'openai'`` from inside
``neatlogs/openai.py::_patch_openai_module`` because the dispatcher eagerly
imported the wrapper module the moment ``cls_name`` matched the well-known
SDK class name.

These tests inject sentinel modules into ``sys.modules`` to simulate the
"optional provider SDK not installed" condition without having to uninstall
each provider package from the test environment. With the dispatcher
fallback in place, ``wrap()`` must always return a wrapped client — falling
back to the universal ``_WrapContextProxy`` when any provider wrapper
module fails to import.
"""

import pytest

import neatlogs


# Provider wrapper modules that the dispatcher in ``neatlogs/__init__.py``
# imports lazily. The set must be kept in sync with the dispatcher.
_PROVIDER_MODULES = [
    "neatlogs.openai",
    "neatlogs.anthropic",
    "neatlogs.azure_openai",
    "neatlogs.crewai",
    "neatlogs.dspy",
    "neatlogs.agno",
    "neatlogs.google_adk",
    "neatlogs.strands",
    "neatlogs.hermes",
    "neatlogs.openrouter",
    "neatlogs.claude_agent_sdk",
    "neatlogs.bedrock",
    "neatlogs.vertex_ai",
    "neatlogs.google_genai",
    "neatlogs.pydantic_ai",
]


class _Boom:
    """Sentinel module replacement: any attribute access raises ImportError.

    Mirrors what happens in real life when ``import openai`` (or another
    optional provider SDK) is executed in an environment where the package
    isn't installed.
    """

    def __getattr__(self, name):
        raise ImportError("simulated missing optional dep")


@pytest.fixture
def poisoned_providers(monkeypatch):
    """Replace every provider wrapper module in ``sys.modules`` with a
    ``_Boom`` sentinel for the duration of the test, then restore them
    automatically via ``monkeypatch``'s implicit fixture teardown."""

    import sys

    for mod in _PROVIDER_MODULES:
        monkeypatch.setitem(sys.modules, mod, _Boom())


def test_wrap_does_not_raise_when_provider_modules_unimportable(poisoned_providers):
    """Regression: ``neatlogs.wrap()`` must not raise ``ImportError`` when
    none of the optional provider SDKs are installed. Previously the
    dispatcher's eager import of ``neatlogs.openai`` (because the test
    class was named ``OpenAI``) crashed with ``ModuleNotFoundError``.

    The fix detects the failed import and falls back to the universal
    ``_WrapContextProxy`` so the call returns successfully and workflow
    metadata can still flow through the wrap context.
    """

    # Class names that exercise the dispatcher's most common matchers.
    class OpenAI:
        def chat(self):
            return None

    class AsyncOpenAI:
        async def chat(self):
            return None

    class Anthropic:
        def messages(self):
            return None

    class Crew:
        def kickoff(self):
            return None

    class Flow:
        def kickoff(self):
            return None

    # Each of these used to raise ``ImportError`` before the dispatcher
    # fallback. They must all now return a wrapped client.
    for client, label in (
        (OpenAI(), "OpenAI"),
        (AsyncOpenAI(), "AsyncOpenAI"),
        (Anthropic(), "Anthropic"),
        (Crew(), "Crew"),
        (Flow(), "Flow"),
    ):
        wrapped = neatlogs.wrap(
            client,
            workflow_name=f"{label} shadow",
            project_id="project_x",
        )
        assert wrapped is not None, f"wrap() returned None for {label}"


def test_wrap_dispatcher_handles_unknown_class_cleanly(poisoned_providers):
    """For an unrecognized class (no name match, no module match), the
    dispatcher must still raise the original ``TypeError`` — the fallback
    only fires when a recognized name matches but the wrapper module is
    unavailable. This guards against accidentally swallowing
    ``TypeError`` for truly unsupported types.
    """

    class SomeCompletelyUnrelatedClass:
        def hello(self):
            return "world"

    with pytest.raises(
        TypeError, match="neatlogs.wrap.. does not support SomeCompletelyUnrelatedClass"
    ):
        neatlogs.wrap(SomeCompletelyUnrelatedClass(), project_id="x")


def test_realistic_openai_shadow_does_not_crash_test_session_identity(monkeypatch):
    """Reproduces the exact ``test_session_identity`` failure: a class
    named ``OpenAI`` with no real ``openai`` SDK available. The test was
    originally marked as needing ``openai`` in the env, but the real
    environment-level fix is the dispatcher fallback — so this test
    exercises the production code path users actually hit.
    """

    import sys

    monkeypatch.setitem(sys.modules, "openai", _Boom())
    # Force re-evaluation of the wrapper module the next time it's imported.
    monkeypatch.delitem(sys.modules, "neatlogs.openai", raising=False)

    class OpenAI:
        def __init__(self):
            pass

        def chat(self):
            return None

    # Before the fallback, this raised ``ModuleNotFoundError``. After the
    # fallback, this returns the universal proxy successfully.
    wrapped = neatlogs.wrap(
        OpenAI(),
        workflow_name="Copilot chat",
        project_id="project_123",
    )
    assert wrapped is not None
