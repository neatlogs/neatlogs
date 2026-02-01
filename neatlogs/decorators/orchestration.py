"""
Decorators for custom orchestration.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, TypeVar

from ._base import _decorate_span, _safe_json_dumps

F = TypeVar("F", bound=Callable[..., Any])


def workflow(
    name: Optional[str] = None,
    *,
    description: Optional[str] = None,
    version: Optional[str] = None,
    tags: Optional[list[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    attributes: Optional[Dict[str, Any]] = None,
    capture_input: Optional[bool] = None,
    capture_output: Optional[bool] = None,
) -> Callable[[F], F]:
    """
    Root/top-level orchestration span.
    """
    return _decorate_span(
        openinference_kind="CHAIN",
        name=name,
        description=description,
        version=version,
        tags=tags,
        metadata=metadata,
        attributes=attributes,
        capture_input=capture_input,
        capture_output=capture_output,
    )


def chain(
    name: Optional[str] = None,
    *,
    description: Optional[str] = None,
    version: Optional[str] = None,
    tags: Optional[list[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    attributes: Optional[Dict[str, Any]] = None,
    capture_input: Optional[bool] = None,
    capture_output: Optional[bool] = None,
) -> Callable[[F], F]:
    """
    Generic orchestration step.
    """
    return _decorate_span(
        openinference_kind="CHAIN",
        name=name,
        description=description,
        version=version,
        tags=tags,
        metadata=metadata,
        attributes=attributes,
        capture_input=capture_input,
        capture_output=capture_output,
    )


def agent(
    name: Optional[str] = None,
    *,
    agent_name: Optional[str] = None,
    role: Optional[str] = None,
    goal: Optional[str] = None,
    description: Optional[str] = None,
    version: Optional[str] = None,
    tags: Optional[list[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    attributes: Optional[Dict[str, Any]] = None,
    capture_input: Optional[bool] = None,
    capture_output: Optional[bool] = None,
) -> Callable[[F], F]:
    extra = dict(attributes or {})
    if agent_name:
        extra["agent.name"] = agent_name
    elif role:
        extra["agent.name"] = role
    if role:
        extra["neatlogs.agent.role"] = role
    if goal:
        extra["neatlogs.agent.goal"] = goal

    return _decorate_span(
        openinference_kind="AGENT",
        name=name,
        description=description,
        version=version,
        tags=tags,
        metadata=metadata,
        attributes=extra,
        capture_input=capture_input,
        capture_output=capture_output,
    )


def tool(
    name: Optional[str] = None,
    *,
    tool_name: Optional[str] = None,
    description: Optional[str] = None,
    parameters: Optional[Dict[str, Any]] = None,
    version: Optional[str] = None,
    tags: Optional[list[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    attributes: Optional[Dict[str, Any]] = None,
    capture_input: Optional[bool] = None,
    capture_output: Optional[bool] = None,
) -> Callable[[F], F]:
    extra = dict(attributes or {})
    if tool_name:
        extra["tool.name"] = tool_name
    if description:
        extra["tool.description"] = description
    if parameters is not None:
        extra["tool.parameters"] = _safe_json_dumps(parameters)

    return _decorate_span(
        openinference_kind="TOOL",
        name=name,
        description=None,
        version=version,
        tags=tags,
        metadata=metadata,
        attributes=extra,
        capture_input=capture_input,
        capture_output=capture_output,
    )


def retriever(
    name: Optional[str] = None,
    *,
    description: Optional[str] = None,
    version: Optional[str] = None,
    tags: Optional[list[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    attributes: Optional[Dict[str, Any]] = None,
    capture_input: Optional[bool] = None,
    capture_output: Optional[bool] = None,
) -> Callable[[F], F]:
    """
    Retrieval boundary for custom RAG.

    Note: OpenInference only standardizes `retrieval.documents`. We keep the query
    in `input.value` (JSON) by default when capture_input is enabled.
    """

    def _set_retrieval_attrs(span: Any, result: Any, bound_inputs: Dict[str, Any]) -> None:
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

    return _decorate_span(
        openinference_kind="RETRIEVER",
        name=name,
        description=description,
        version=version,
        tags=tags,
        metadata=metadata,
        attributes=attributes,
        capture_input=capture_input,
        capture_output=capture_output,
        postprocess_result=_set_retrieval_attrs,
    )
