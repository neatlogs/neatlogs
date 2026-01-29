"""
Neatlogs SDK v4 - Langfuse Architecture

A production-ready observability SDK with:
- Dual instrumentation (OpenInference + OpenLLMetry)
- Smart attribute merging
- Traceloop-style context propagation
- Explicit prompt capture
- OpenInference span kinds (9 granular types)
- Universal PromptTemplate (eliminates variable duplication)
"""

from .init import init, flush, shutdown
from .core.context import trace, track_prompt
from .prompt.capture import capture_prompt, capture_vars
from .prompt.decorators import observe
from .prompt.template import PromptTemplate
from .decorators import mcp_tool, workflow, chain, agent, tool, retriever

__all__ = [
    "init",
    "flush",
    "shutdown",
    "trace",
    "observe",
    "track_prompt",
    "capture_prompt",
    "capture_vars",
    "PromptTemplate",
    "mcp_tool",
    "workflow",
    "chain",
    "agent",
    "tool",
    "retriever",
]

__version__ = "4.0.0"
