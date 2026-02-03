"""
Neatlogs SDK
"""

from .core.context import trace, track_prompt
from .decorators import agent, chain, mcp_tool, retriever, tool, workflow
from .init import flush, init, shutdown
from .prompt.capture import capture_prompt, capture_vars
from .prompt.decorators import observe
from .prompt.template import PromptTemplate

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
