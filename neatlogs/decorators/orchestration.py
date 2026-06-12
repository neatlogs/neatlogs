"""
Decorators for custom orchestration.

This module provides the unified @span decorator for instrumenting custom code.
"""

from __future__ import annotations

import functools
import inspect
import json
import logging
from typing import Any, Callable, Dict, Optional, TypeVar

from opentelemetry import trace as otel_trace
from opentelemetry.trace import Status, StatusCode

from ._base import _capture_code_attrs, _decorate_span, _safe_json_dumps

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def _create_mcp_tool_decorator(
    name: Optional[str] = None,
    *,
    tool_name: Optional[str] = None,
    description: Optional[str] = None,
    tool_json_schema: Optional[dict] = None,
) -> Callable[[F], F]:
    """
    Create MCP tool decorator with EXACT logic from original mcp_tool.py.

    This preserves the exact behavior:
    - Checks args[0] for Pydantic models BEFORE signature binding
    - Wraps string results as {"result": "..."} for output.value
    - Sets both mcp.* and standard attributes explicitly
    """

    def decorator(func: F) -> F:
        span_name = name or f"{func.__name__}.tool"
        tool_name_attr = tool_name or func.__name__
        tracer = otel_trace.get_tracer(__name__)

        # Capture code-location attrs once at decoration time so that MCP_TOOL
        # spans carry the same ``code.*`` metadata as every other kind.
        code_attrs = _capture_code_attrs(func)

        def _apply_code_attrs(span):
            for k, v in code_attrs.items():
                span.set_attribute(k, v)

        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                with tracer.start_as_current_span(
                    span_name,
                    kind=otel_trace.SpanKind.INTERNAL,
                ) as span:
                    span.set_attribute("mcp.tool.name", tool_name_attr)
                    span.set_attribute("openinference.span.kind", "MCP_TOOL")
                    span.set_attribute("tool.name", tool_name_attr)
                    _apply_code_attrs(span)

                    if description:
                        span.set_attribute("tool.description", description)
                    if tool_json_schema is not None:
                        span.set_attribute("tool.json_schema", json.dumps(tool_json_schema))

                    try:
                        # EXACT original logic: check args[0] first
                        if args and hasattr(args[0], "model_dump"):
                            arguments = args[0].model_dump()
                        else:
                            sig = inspect.signature(func)
                            bound = sig.bind_partial(*args, **kwargs)
                            bound.apply_defaults()
                            arguments = dict(bound.arguments)

                        span.set_attribute("mcp.tool.arguments", json.dumps(arguments))
                        span.set_attribute("tool.parameters", json.dumps(arguments))
                        span.set_attribute("input.value", json.dumps(arguments))
                        span.set_attribute("input.mime_type", "application/json")
                    except Exception as e:
                        logger.debug(f"Failed to set MCP tool input attributes: {e}")

                    try:
                        result = await func(*args, **kwargs)
                        try:
                            # EXACT original logic: wrap strings, set both attributes
                            if isinstance(result, dict):
                                span.set_attribute("mcp.response.value", json.dumps(result))
                                span.set_attribute("output.value", json.dumps(result))
                                span.set_attribute("output.mime_type", "application/json")
                            elif isinstance(result, str):
                                output = {"result": result}
                                span.set_attribute("mcp.response.value", json.dumps(output))
                                span.set_attribute("output.value", json.dumps(output))
                                span.set_attribute("output.mime_type", "application/json")
                        except Exception as e:
                            logger.debug(f"Failed to set MCP tool output attributes: {e}")

                        span.set_status(Status(StatusCode.OK))
                        return result

                    except Exception as e:
                        span.record_exception(e)
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        raise

            return async_wrapper

        else:

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                with tracer.start_as_current_span(
                    span_name,
                    kind=otel_trace.SpanKind.INTERNAL,
                ) as span:
                    span.set_attribute("mcp.tool.name", tool_name_attr)
                    span.set_attribute("openinference.span.kind", "MCP_TOOL")
                    span.set_attribute("tool.name", tool_name_attr)
                    _apply_code_attrs(span)

                    if description:
                        span.set_attribute("tool.description", description)
                    if tool_json_schema is not None:
                        span.set_attribute("tool.json_schema", json.dumps(tool_json_schema))

                    try:
                        # EXACT original logic: check args[0] first
                        if args and hasattr(args[0], "model_dump"):
                            arguments = args[0].model_dump()
                        else:
                            sig = inspect.signature(func)
                            bound = sig.bind_partial(*args, **kwargs)
                            bound.apply_defaults()
                            arguments = dict(bound.arguments)

                        span.set_attribute("mcp.tool.arguments", json.dumps(arguments))
                        span.set_attribute("tool.parameters", json.dumps(arguments))
                        span.set_attribute("input.value", json.dumps(arguments))
                        span.set_attribute("input.mime_type", "application/json")
                    except Exception as e:
                        logger.debug(f"Failed to set MCP tool input attributes: {e}")

                    try:
                        result = func(*args, **kwargs)
                        try:
                            # EXACT original logic: wrap strings, set both attributes
                            if isinstance(result, dict):
                                span.set_attribute("mcp.response.value", json.dumps(result))
                                span.set_attribute("output.value", json.dumps(result))
                                span.set_attribute("output.mime_type", "application/json")
                            elif isinstance(result, str):
                                output = {"result": result}
                                span.set_attribute("mcp.response.value", json.dumps(output))
                                span.set_attribute("output.value", json.dumps(output))
                                span.set_attribute("output.mime_type", "application/json")
                        except Exception as e:
                            logger.debug(f"Failed to set MCP tool output attributes: {e}")

                        span.set_status(Status(StatusCode.OK))
                        return result

                    except Exception as e:
                        span.record_exception(e)
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        raise

            return sync_wrapper

    return decorator


def span(
    kind: str,
    name: Optional[str] = None,
    *,
    # Common attributes
    description: Optional[str] = None,
    version: Optional[str] = None,
    tags: Optional[list[str]] = None,
    capture_input: Optional[bool] = None,
    capture_output: Optional[bool] = None,
    capture_stdout: bool = False,
    mask: Optional[Callable] = None,
    # Agent-specific
    role: Optional[str] = None,
    goal: Optional[str] = None,
    # Tool-specific
    tool_name: Optional[str] = None,
    parameters: Optional[Dict[str, Any]] = None,
    # End-user identity (one end-user per trace; usually set on the WORKFLOW root)
    end_user_id: Optional[str] = None,
    end_user_metadata: Optional[Dict[str, Any]] = None,
) -> Callable[[F], F]:
    """
    Universal decorator for instrumenting custom code with observability spans.

    This is the single, unified decorator that replaces @workflow, @agent, @chain, etc.
    Specify the span type using the `kind` parameter.

    Args:
        kind: OpenInference span kind - one of:
              "WORKFLOW" - Top-level orchestration entry point
              "AGENT" - Agent execution with optional role/goal
              "CHAIN" - Generic sequential steps/processing
              "TOOL" - Tool or function call
              "RETRIEVER" - RAG retrieval operation
              "EMBEDDING" - Embedding generation
              "MCP_TOOL" - MCP protocol tool call
        name: Optional span name (defaults to function name)
        description: Human-readable description
        version: Version identifier for tracking changes
        tags: List of tags for filtering/grouping
        capture_input: Whether to capture function input (default: True)
        capture_output: Whether to capture function output (default: True)

        # Agent-specific parameters (when kind="AGENT"):
        role: Agent role description (also sets agent.name)
        goal: Agent goal/objective

        # Tool-specific parameters (when kind="TOOL" or kind="MCP_TOOL"):
        tool_name: Tool identifier
        parameters: Tool parameter schema

        # End-user identity (typically only on the WORKFLOW root):
        end_user_id: Identifier of the END-USER this trace belongs to — the user
              of your application, not the operator running the SDK. One end-user
              per trace; the backend rolls it up to the trace + session for
              filtering/analytics. Distinct from init(user_id=...).
        end_user_metadata: Optional dict of arbitrary end-user fields, stored as
              JSON on the trace (e.g. {"plan": "pro"}).

        Note: MCP_TOOL automatically handles:

    Returns:
        Decorated function with span instrumentation

    Examples:
        Top-level workflow:

        >>> @span(kind="WORKFLOW", name="support_workflow")
        >>> def handle_support(query: str):
        ...     docs = retrieve_docs(query)
        ...     return generate_answer(docs)

        Agent with role:

        >>> @span(kind="AGENT", role="Researcher", goal="Find relevant papers")
        >>> def research_agent(topic: str):
        ...     return search_and_analyze(topic)

        Generic processing step:

        >>> @span(kind="CHAIN", name="data_preprocessing")
        >>> def preprocess(data):
        ...     cleaned = clean(data)
        ...     return validate(cleaned)

        Tool/function call:

        >>> @span(kind="TOOL", tool_name="web_search")
        >>> def search_web(query: str):
        ...     return requests.get(f"https://api.search.com?q={query}")

        Retrieval operation:

        >>> @span(kind="RETRIEVER", name="vector_search")
        >>> def retrieve_docs(query: str):
        ...     return vector_db.search(query)

        MCP tool with auto Pydantic handling:

        >>> @span(kind="MCP_TOOL", tool_name="add_numbers", description="Add two numbers")
        >>> async def add(a: int, b: int) -> str:
        ...     return f"Result: {a + b}"

    Note:
        This decorator is for instrumenting YOUR custom code. Framework code
        (LangChain, LlamaIndex, OpenAI, etc) is auto-instrumented and doesn't
        need decoration.
    """
    # Validate kind
    valid_kinds = {
        "WORKFLOW",
        "AGENT",
        "CHAIN",
        "TOOL",
        "RETRIEVER",
        "EMBEDDING",
        "GUARDRAIL",
        "MCP_TOOL",
    }
    kind_upper = kind.upper()
    if kind_upper not in valid_kinds:
        raise ValueError(f"Invalid span kind: {kind}. Must be one of {valid_kinds}")

    # MCP_TOOL requires special handling to match exact original logic
    # (needs access to raw args[0] before binding, special output wrapping)
    if kind_upper == "MCP_TOOL":
        return _create_mcp_tool_decorator(
            name=name,
            tool_name=tool_name,
            description=description,
        )

    # Build attributes based on kind
    extra: Dict[str, Any] = {}

    # Agent-specific attributes
    if kind_upper == "AGENT":
        if role:
            extra["agent.name"] = role
            extra["neatlogs.agent.role"] = role
        if goal:
            extra["neatlogs.agent.goal"] = goal

    # Tool-specific attributes
    elif kind_upper == "TOOL":
        if tool_name:
            extra["tool.name"] = tool_name
        if description:
            extra["tool.description"] = description
        if parameters is not None:
            extra["tool.parameters"] = _safe_json_dumps(parameters)

    # Use specialized postprocessor for retriever
    postprocess_result = None
    if kind_upper == "RETRIEVER":
        postprocess_result = _retriever_postprocessor

    return _decorate_span(
        openinference_kind=kind_upper,
        name=name,
        description=description if kind_upper not in ("TOOL", "MCP_TOOL") else None,
        version=version,
        tags=tags,
        attributes=extra,
        capture_input=capture_input,
        capture_output=capture_output,
        capture_stdout=capture_stdout,
        postprocess_result=postprocess_result,
        mask=mask,
        # End-user belongs to the trace root only. _decorate_span stamps these
        # at call time, and only when the decorated span is a root (no active
        # parent). The backend rolls the value up to the trace and session.
        end_user_id=end_user_id,
        end_user_metadata=end_user_metadata,
    )


def _retriever_postprocessor(span: Any, result: Any, bound_inputs: Dict[str, Any]) -> None:
    """Helper to set retrieval-specific attributes."""
    # Extract query
    query = None
    for k in ("query", "question", "text"):
        v = bound_inputs.get(k)
        if isinstance(v, str) and v:
            query = v
            break
    if query is None:
        for v in bound_inputs.values():
            if isinstance(v, str) and v:
                query = v
                break
    if query:
        span.set_attribute("retrieval.query", query)

    # Extract documents
    docs: Any = None
    if isinstance(result, (list, tuple)):
        docs = list(result)
    elif isinstance(result, dict):
        for key in ("documents", "docs", "results", "matches", "items", "data"):
            v = result.get(key)
            if isinstance(v, (list, tuple)):
                docs = list(v)
                break
    if not docs:
        return

    # Set document attributes
    for i, doc in enumerate(docs[:20]):
        if isinstance(doc, str):
            span.set_attribute(f"retrieval.documents.{i}.document.content", doc)
            continue

        if isinstance(doc, dict):
            doc_id = doc.get("id") or doc.get("_id") or doc.get("doc_id")
            content = doc.get("content") or doc.get("document") or doc.get("text")
            score = doc.get("score") or doc.get("_score") or doc.get("distance")
            metadata = doc.get("metadata") or {}

            if doc_id is not None:
                span.set_attribute(f"retrieval.documents.{i}.document.id", str(doc_id))
            if content is not None:
                span.set_attribute(f"retrieval.documents.{i}.document.content", str(content))
            if score is not None:
                try:
                    span.set_attribute(f"retrieval.documents.{i}.document.score", float(score))
                except Exception:
                    span.set_attribute(f"retrieval.documents.{i}.document.score", str(score))
            if metadata:
                span.set_attribute(
                    f"retrieval.documents.{i}.document.metadata",
                    _safe_json_dumps(metadata),
                )
