"""
Neatlogs SDK - Simple, powerful LLM observability.

Primary API:
    - init(), flush(), shutdown() - Lifecycle management
    - @span(kind="...") - Universal decorator for custom code instrumentation
    - trace() - Context manager for prompt tracking and session management
    - SystemPromptTemplate - Structured prompt versioning (formerly PromptTemplate)

Quick Start:
    >>> from neatlogs import init, span, trace, SystemPromptTemplate
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
from .core.crewai_task_registry import register_crewai_task
from .core.llm_binder import bind_templates
from .core.log import log
from .decorators import span
from .init import flush, init, shutdown
from .prompt.client import (
    AsyncPromptClient,
    CachedPrompt,
    PromptApiError,
    PromptClient,
    PromptClientError,
    PromptHandle,
    PromptNotFoundError,
    aget_prompt,
    create_prompt,
    delete_prompt,
    fetch_prompt,
    get_prompt,
    list_prompts,
    remove_tag,
    save_as_version,
    update_prompt,
)
from .prompt.template import PromptTemplate, SystemPromptTemplate, UserPromptTemplate
from .version import __version__


def langchain_handler(**kwargs):
    """
    Create a LangChain/LangGraph callback handler for tracing.

    Works with LangChain, LangGraph, and any framework using LangChain callbacks
    (e.g., Deep Agents).

        >>> import neatlogs
        >>> handler = neatlogs.langchain_handler()
        >>> result = chain.invoke(input, config={"callbacks": [handler]})
    """
    from .langchain import NeatlogsCallbackHandler

    return NeatlogsCallbackHandler(**kwargs)


def openai_agents_processor():
    """
    Return a trace processor for the OpenAI Agents SDK.

        >>> import neatlogs
        >>> from agents import add_trace_processor
        >>> add_trace_processor(neatlogs.openai_agents_processor())
    """
    from .openai_agents import openai_agents_processor as _proc

    return _proc()


def strands_hooks(agent):
    """
    Register Neatlogs tracing hooks on a Strands Agent.

        >>> import neatlogs
        >>> from strands import Agent
        >>> agent = Agent(model=model)
        >>> neatlogs.strands_hooks(agent)
    """
    from .strands import strands_hooks as _hooks

    return _hooks(agent)


def wrap(client):
    """
    Wrap an LLM client or agent instance to auto-trace all calls.

    Supports: OpenAI, AsyncOpenAI, Anthropic, AsyncAnthropic, google.genai.Client,
    CrewAI Crew, Pydantic AI Agent, DSPy modules, Agno Agent/Team/Workflow,
    Google ADK Runner, Strands Agent.

        >>> import neatlogs, openai
        >>> client = neatlogs.wrap(openai.OpenAI())
        >>> client.chat.completions.create(...)
    """
    cls_name = type(client).__name__
    module = type(client).__module__ or ""

    if "openai" in module or cls_name in ("OpenAI", "AsyncOpenAI"):
        from .openai import wrap_async_openai_client, wrap_openai_client

        if "Async" in cls_name:
            return wrap_async_openai_client(client)
        return wrap_openai_client(client)

    if "anthropic" in module or cls_name in ("Anthropic", "AsyncAnthropic"):
        from .anthropic import wrap_anthropic_client, wrap_async_anthropic_client

        if "Async" in cls_name:
            return wrap_async_anthropic_client(client)
        return wrap_anthropic_client(client)

    if ("google" in module and "genai" in module) or cls_name == "Client":
        from .google_genai import wrap_google_genai_client

        return wrap_google_genai_client(client)

    if "crewai" in module and cls_name == "Crew":
        from .crewai import wrap_crewai

        return wrap_crewai(client)

    if "pydantic_ai" in module and cls_name == "Agent":
        from .pydantic_ai import wrap_pydantic_ai

        return wrap_pydantic_ai(client)

    if "dspy" in module:
        from .dspy import wrap_dspy

        return wrap_dspy(client)

    if "agno" in module:
        from .agno import wrap_agno

        return wrap_agno(client)

    if "google.adk" in module or "google_adk" in module:
        from .google_adk import wrap_google_adk

        return wrap_google_adk(client)

    if "strands" in module:
        from .strands import strands_hooks

        return strands_hooks(client)

    raise TypeError(
        f"neatlogs.wrap() does not support {cls_name} from {module}. "
        "Supported: OpenAI, AsyncOpenAI, Anthropic, AsyncAnthropic, google.genai.Client, "
        "CrewAI Crew, Pydantic AI Agent, DSPy modules, Agno agents, Google ADK Runner, Strands Agent"
    )


__all__ = [
    "init",
    "flush",
    "shutdown",
    "span",
    "trace",
    "log",
    "SystemPromptTemplate",
    "PromptTemplate",  # backward-compatible alias
    "UserPromptTemplate",
    "CachedPrompt",
    "PromptHandle",
    "PromptClient",
    "AsyncPromptClient",
    "PromptClientError",
    "PromptApiError",
    "PromptNotFoundError",
    "get_prompt",
    "aget_prompt",
    "fetch_prompt",
    "list_prompts",
    "create_prompt",
    "update_prompt",
    "save_as_version",
    "delete_prompt",
    "remove_tag",
    "bind_templates",
    "register_crewai_task",
    "wrap",
    "langchain_handler",
    "openai_agents_processor",
    "strands_hooks",
    "__version__",
]
