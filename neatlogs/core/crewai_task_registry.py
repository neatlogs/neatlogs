"""
Registry for CrewAI task template bindings.

Call register_crewai_task(task, user_tpl, **vars) in tasks.py after creating
each Task. The span processor reads and clears the entry when the corresponding
_execute_core AGENT span ends, stamping the template onto that span and
relabeling its kind to CREWAI_TASK.
"""

from __future__ import annotations

import json
import threading
from typing import Any

_lock = threading.Lock()
_registry: dict[str, tuple[str, str | None]] = {}  # task_id → (template, vars_json)


def register_crewai_task(task: Any, user_tpl: Any, **vars: Any) -> None:
    """
    Register a user prompt template for a CrewAI task.

    Args:
        task: A crewai.Task instance (must have a .id attribute).
        user_tpl: A neatlogs.UserPromptTemplate describing the task prompt.
        **vars: Variable values passed to user_tpl at task-creation time.
    """
    task_id = str(task.id)
    tpl_str = str(user_tpl.template)
    vars_json = json.dumps(vars, default=str) if vars else None
    with _lock:
        _registry[task_id] = (tpl_str, vars_json)


def pop_entry(task_id: str) -> tuple[str, str | None] | None:
    """Remove and return the registry entry for task_id, or None if absent."""
    with _lock:
        return _registry.pop(task_id, None)
