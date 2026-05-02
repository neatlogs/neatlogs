"""Shared contextvars for MCP callback span hierarchy and prompt template propagation."""

from __future__ import annotations

import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Dict, Optional

_trace_id: ContextVar[Optional[str]] = ContextVar("nl.mcp.trace_id", default=None)
_parent_span_id: ContextVar[Optional[str]] = ContextVar("nl.mcp.parent_span_id", default=None)
_root_span_id: ContextVar[Optional[str]] = ContextVar("nl.mcp.root_span_id", default=None)

_prompt_template: ContextVar[Optional[str]] = ContextVar("nl.mcp.prompt_template", default=None)
_prompt_variables: ContextVar[Optional[Dict[str, Any]]] = ContextVar(
    "nl.mcp.prompt_variables", default=None
)
_user_prompt_template: ContextVar[Optional[str]] = ContextVar(
    "nl.mcp.user_prompt_template", default=None
)
_user_prompt_variables: ContextVar[Optional[Dict[str, Any]]] = ContextVar(
    "nl.mcp.user_prompt_variables", default=None
)


def generate_span_id() -> str:
    return uuid.uuid4().hex[:16]


def generate_trace_id() -> str:
    return uuid.uuid4().hex


def get_trace_id() -> Optional[str]:
    return _trace_id.get()


def set_trace_id(tid: str) -> None:
    _trace_id.set(tid)


def get_parent_span_id() -> Optional[str]:
    return _parent_span_id.get()


def set_parent_span_id(sid: Optional[str]) -> None:
    _parent_span_id.set(sid)


def get_root_span_id() -> Optional[str]:
    return _root_span_id.get()


def set_root_span_id(sid: str) -> None:
    _root_span_id.set(sid)


@contextmanager
def span_scope(span_id: str):
    """Context manager that sets span_id as the current parent for nested calls."""
    prev = _parent_span_id.get()
    _parent_span_id.set(span_id)
    try:
        yield
    finally:
        _parent_span_id.set(prev)


def get_prompt_template() -> Optional[str]:
    return _prompt_template.get()


def get_prompt_variables() -> Optional[Dict[str, Any]]:
    return _prompt_variables.get()


def set_prompt_template(template: Optional[str], variables: Optional[Dict[str, Any]] = None):
    _prompt_template.set(template)
    _prompt_variables.set(variables)


def get_user_prompt_template() -> Optional[str]:
    return _user_prompt_template.get()


def get_user_prompt_variables() -> Optional[Dict[str, Any]]:
    return _user_prompt_variables.get()


def set_user_prompt_template(
    template: Optional[str], variables: Optional[Dict[str, Any]] = None
):
    _user_prompt_template.set(template)
    _user_prompt_variables.set(variables)


def clear_prompt_context():
    _prompt_template.set(None)
    _prompt_variables.set(None)
    _user_prompt_template.set(None)
    _user_prompt_variables.set(None)
