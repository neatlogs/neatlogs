"""
Tests that @neatlogs.span() decorators auto-capture code location attributes
(code.file.path, code.function.name, code.line.number, code.namespace) on the
resulting span, and that auto-instrumented spans (not decorated) do NOT carry
these attributes.
"""

import asyncio
import functools
import inspect
from pathlib import Path

import pytest
from opentelemetry import trace

from neatlogs._wrap_utils import set_neatlogs_provider
from neatlogs.core.context import trace as neatlogs_trace
from neatlogs.decorators._base import _decorate_span
from neatlogs.decorators.orchestration import span as neatlogs_span

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _install(tracer_provider):
    """Register the test provider explicitly with both test pipelines.

    The autouse ``reset_neatlogs_and_otel_state`` fixture in ``conftest.py``
    clears the OTel ``set_tracer_provider`` guard between tests, so this is
    safe to call repeatedly.
    """
    trace.set_tracer_provider(tracer_provider)
    set_neatlogs_provider(tracer_provider)


def _semantic_span(spans, kind):
    return next(span for span in spans if span.attributes.get("openinference.span.kind") == kind)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["GUARDRAIL", "EVALUATOR", "MEMORY"])
def test_semantic_span_kinds_auto_capture_io(kind, tracer_provider, in_memory_span_exporter):
    _install(tracer_provider)

    @neatlogs_span(kind=kind, name=f"test.{kind.lower()}")
    def semantic_stage(value: str):
        return {"seen": value}

    assert semantic_stage("evidence") == {"seen": "evidence"}

    spans = in_memory_span_exporter.get_finished_spans()
    finished = _semantic_span(spans, kind)
    attrs = finished.attributes
    assert attrs["neatlogs.span.kind"] == kind.lower()
    assert "evidence" in attrs["input.value"]
    assert "evidence" in attrs["output.value"]


def test_sync_decorator_sets_code_file_path(tracer_provider, in_memory_span_exporter):
    _install(tracer_provider)

    @_decorate_span(openinference_kind="CHAIN")
    def my_function():
        return 42

    my_function()

    spans = in_memory_span_exporter.get_finished_spans()
    attrs = _semantic_span(spans, "CHAIN").attributes
    assert "code.file.path" in attrs
    assert Path(attrs["code.file.path"]).name == Path(__file__).name


def test_sync_decorator_sets_code_function_name(tracer_provider, in_memory_span_exporter):
    _install(tracer_provider)

    @_decorate_span(openinference_kind="TOOL")
    def my_tool_function():
        pass

    my_tool_function()

    spans = in_memory_span_exporter.get_finished_spans()
    attrs = _semantic_span(spans, "TOOL").attributes
    assert (
        attrs.get("code.function.name")
        == "test_sync_decorator_sets_code_function_name.<locals>.my_tool_function"
    )


def test_sync_decorator_sets_code_line_number(tracer_provider, in_memory_span_exporter):
    _install(tracer_provider)

    @_decorate_span(openinference_kind="AGENT")
    def my_agent():
        pass

    expected_lineno = inspect.getsourcelines(
        my_agent.__wrapped__ if hasattr(my_agent, "__wrapped__") else my_agent
    )[1]

    my_agent()

    spans = in_memory_span_exporter.get_finished_spans()
    attrs = _semantic_span(spans, "AGENT").attributes
    assert attrs["code.line.number"] == expected_lineno


def test_sync_decorator_sets_code_namespace(tracer_provider, in_memory_span_exporter):
    _install(tracer_provider)

    @_decorate_span(openinference_kind="WORKFLOW")
    def my_workflow():
        pass

    my_workflow()

    spans = in_memory_span_exporter.get_finished_spans()
    attrs = _semantic_span(spans, "WORKFLOW").attributes
    assert attrs.get("code.namespace") == __name__


def test_async_decorator_sets_code_attributes(tracer_provider, in_memory_span_exporter):
    _install(tracer_provider)

    @_decorate_span(openinference_kind="CHAIN")
    async def my_async_function():
        return "result"

    asyncio.run(my_async_function())

    spans = in_memory_span_exporter.get_finished_spans()
    attrs = _semantic_span(spans, "CHAIN").attributes
    assert "code.file.path" in attrs
    assert "code.function.name" in attrs
    assert "code.line.number" in attrs
    assert "code.namespace" in attrs


def test_caller_attributes_preserved_alongside_code_attrs(tracer_provider, in_memory_span_exporter):
    """User-supplied (disjoint-key) attributes coexist with auto code params."""
    _install(tracer_provider)

    @_decorate_span(openinference_kind="TOOL", attributes={"custom.key": "custom_value"})
    def my_tool():
        pass

    my_tool()

    spans = in_memory_span_exporter.get_finished_spans()
    attrs = _semantic_span(spans, "TOOL").attributes
    assert attrs.get("custom.key") == "custom_value"
    # code attrs are also present
    assert "code.file.path" in attrs
    assert "code.function.name" in attrs


def test_user_supplied_code_attrs_take_precedence(tracer_provider, in_memory_span_exporter):
    """User-supplied code.* attributes must win over auto-captured ones."""
    _install(tracer_provider)

    @_decorate_span(
        openinference_kind="TOOL",
        attributes={
            "code.file.path": "user_provided_path.py",
            "code.function.name": "user_provided_name",
            "code.namespace": "user.provided.namespace",
        },
    )
    def my_tool():
        pass

    my_tool()

    spans = in_memory_span_exporter.get_finished_spans()
    attrs = _semantic_span(spans, "TOOL").attributes
    assert attrs.get("code.file.path") == "user_provided_path.py"
    assert attrs.get("code.function.name") == "user_provided_name"
    assert attrs.get("code.namespace") == "user.provided.namespace"


def test_code_attrs_do_not_appear_on_plain_otel_span(tracer_provider, in_memory_span_exporter):
    """Spans created without the neatlogs decorator should not have code params."""
    _install(tracer_provider)

    tracer = trace.get_tracer("test")
    with tracer.start_as_current_span("plain-span"):
        pass

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    attrs = spans[0].attributes
    assert "code.file.path" not in attrs
    assert "code.function.name" not in attrs
    assert "code.line.number" not in attrs


# ---------------------------------------------------------------------------
# Stacked-decorator unwrap behaviour
# ---------------------------------------------------------------------------


def _noop_wraps(func):
    """A simple decorator that correctly preserves __wrapped__ via functools.wraps."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


def test_stacked_decorator_reports_inner_function_location(
    tracer_provider, in_memory_span_exporter
):
    """When @neatlogs.span wraps another decorator that preserves __wrapped__,
    code attributes should point at the user's inner function, not the wrapper.
    """
    _install(tracer_provider)

    @_decorate_span(openinference_kind="TOOL")
    @_noop_wraps
    def inner_tool():
        pass

    expected_lineno = inspect.getsourcelines(inspect.unwrap(inner_tool))[1]

    inner_tool()

    spans = in_memory_span_exporter.get_finished_spans()
    attrs = _semantic_span(spans, "TOOL").attributes
    # File path should be this test file, not functools / some decorator module.
    assert Path(attrs["code.file.path"]).name == Path(__file__).name
    assert attrs["code.function.name"].endswith("inner_tool")
    assert attrs["code.namespace"] == __name__
    assert attrs["code.line.number"] == expected_lineno


def test_parentless_tool_gets_one_root_with_identity(tracer_provider, in_memory_span_exporter):
    _install(tracer_provider)

    @neatlogs_span(kind="TOOL", session_id="session-1")
    def standalone_tool():
        return "done"

    assert standalone_tool() == "done"
    spans = in_memory_span_exporter.get_finished_spans()
    tool = _semantic_span(spans, "TOOL")
    root = next(span for span in spans if span.attributes.get("neatlogs.auto_root") is True)

    assert tool.parent.span_id == root.context.span_id
    assert root.attributes["neatlogs.session.id"] == "session-1"
    assert "neatlogs.session.id" not in tool.attributes


def test_parentless_llm_remains_the_backend_supported_root(
    tracer_provider, in_memory_span_exporter
):
    _install(tracer_provider)

    @_decorate_span(openinference_kind="LLM")
    def standalone_llm():
        return "done"

    standalone_llm()
    spans = in_memory_span_exporter.get_finished_spans()

    assert len(spans) == 1
    assert spans[0].parent is None
    assert spans[0].attributes["openinference.span.kind"] == "LLM"
    assert "neatlogs.auto_root" not in spans[0].attributes


def test_parentless_tool_context_gets_one_root_with_identity(
    tracer_provider, in_memory_span_exporter
):
    _install(tracer_provider)

    with neatlogs_trace("standalone.tool", kind="TOOL", session_id="session-2"):
        pass

    spans = in_memory_span_exporter.get_finished_spans()
    tool = _semantic_span(spans, "TOOL")
    root = next(span for span in spans if span.attributes.get("neatlogs.auto_root") is True)

    assert tool.parent.span_id == root.context.span_id
    assert root.attributes["neatlogs.session.id"] == "session-2"
    assert "neatlogs.session.id" not in tool.attributes


# ---------------------------------------------------------------------------
# MCP_TOOL parity — the public @neatlogs.span(kind="MCP_TOOL") must also set
# code.* attributes, even though it goes through _create_mcp_tool_decorator.
# ---------------------------------------------------------------------------


def test_mcp_tool_decorator_sets_code_attributes(tracer_provider, in_memory_span_exporter):
    _install(tracer_provider)

    @neatlogs_span(kind="MCP_TOOL", tool_name="my_mcp_tool")
    def my_mcp_tool(x: int) -> str:
        return f"got {x}"

    my_mcp_tool(1)

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    attrs = spans[0].attributes
    assert "code.file.path" in attrs
    assert Path(attrs["code.file.path"]).name == Path(__file__).name
    assert attrs["code.function.name"].endswith("my_mcp_tool")
    assert attrs["code.namespace"] == __name__
    assert isinstance(attrs["code.line.number"], int)
    assert attrs["code.line.number"] > 0


def test_mcp_tool_async_decorator_sets_code_attributes(tracer_provider, in_memory_span_exporter):
    _install(tracer_provider)

    @neatlogs_span(kind="MCP_TOOL", tool_name="my_async_mcp_tool")
    async def my_async_mcp_tool(x: int) -> str:
        return f"async {x}"

    asyncio.run(my_async_mcp_tool(2))

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    attrs = spans[0].attributes
    assert "code.file.path" in attrs
    assert attrs["code.function.name"].endswith("my_async_mcp_tool")
    assert attrs["code.namespace"] == __name__
