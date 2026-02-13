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

from .core.context import trace
from .decorators import span
from .init import flush, init, shutdown
from .prompt.template import PromptTemplate

__all__ = [
    "init",
    "flush",
    "shutdown",
    "span",
    "trace",
    "PromptTemplate",
]

__version__ = "4.0.0"
