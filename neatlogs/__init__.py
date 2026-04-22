"""
Neatlogs SDK - Simple, powerful LLM observability.

Primary API:
    - init(), flush(), shutdown() - Lifecycle management
    - @span(kind="...") - Universal decorator for custom code instrumentation
    - trace() - Context manager for prompt tracking and session management
    - PromptTemplate - Structured prompt versioning

Quick Start:
    >>> from neatlogs import init, span, trace, PromptTemplate
    >>>
    >>> init(api_key="...", instrumentations=["openai"])
    >>>
    >>> @span(kind="WORKFLOW")
    >>> def my_workflow(query: str):
    ...     return process(query)

Available span kinds:
    - "WORKFLOW" - Top-level orchestration
    - "AGENT" - Agent execution
    - "CHAIN" - Sequential processing
    - "TOOL" - Tool/function call
    - "RETRIEVER" - RAG retrieval
    - "EMBEDDING" - Embedding generation
    - "MCP_TOOL" - MCP protocol tool (auto Pydantic handling)
"""

from .config import (
    CachedConfig,
    ConfigApiError,
    ConfigClient,
    ConfigClientError,
    ConfigConflictError,
    ConfigNotFoundError,
    create_config,
    delete_config,
    get_config,
    list_configs,
    update_config,
)
from .core.context import trace
from .core.crewai_task_registry import register_crewai_task
from .core.llm_binder import bind_templates
from .core.log import log
from .decorators import span
from .init import flush, init, shutdown
from .prompt.client import (
    CachedPrompt,
    PromptApiError,
    PromptClient,
    PromptClientError,
    PromptHandle,
    PromptNotFoundError,
    create_prompt,
    delete_prompt,
    fetch_prompt,
    get_prompt,
    list_prompts,
    remove_tag,
    save_as_version,
    update_prompt,
)
from .prompt.template import PromptTemplate, UserPromptTemplate
from .version import __version__

__all__ = [
    "init",
    "flush",
    "shutdown",
    "span",
    "trace",
    "log",
    "PromptTemplate",
    "UserPromptTemplate",
    "CachedPrompt",
    "PromptHandle",
    "PromptClient",
    "PromptClientError",
    "PromptApiError",
    "PromptNotFoundError",
    "get_prompt",
    "fetch_prompt",
    "list_prompts",
    "create_prompt",
    "update_prompt",
    "save_as_version",
    "delete_prompt",
    "remove_tag",
    "CachedConfig",
    "ConfigClient",
    "ConfigClientError",
    "ConfigApiError",
    "ConfigConflictError",
    "ConfigNotFoundError",
    "get_config",
    "list_configs",
    "create_config",
    "update_config",
    "delete_config",
    "bind_templates",
    "register_crewai_task",
    "__version__",
]
