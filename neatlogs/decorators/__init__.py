"""
Decorators for Neatlogs SDK.

Primary decorator:
    @span(kind="...") - Universal decorator for custom code instrumentation

Available kinds:
    - "WORKFLOW" - Top-level orchestration entry point
    - "AGENT" - Agent execution with role/goal
    - "CHAIN" - Generic sequential processing
    - "TOOL" - Tool or function call
    - "RETRIEVER" - RAG retrieval operation
    - "EMBEDDING" - Embedding generation
    - "MCP_TOOL" - MCP protocol tool (with Pydantic auto-handling)
"""

from neatlogs.decorators.orchestration import span

__all__ = [
    "span",
]
