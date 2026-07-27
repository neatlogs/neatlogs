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
from .core.identity import identify
from .core.propagation import extract_trace_context, inject_trace_context
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


def wrap(client, **workflow_attributes):
    """
    Wrap an LLM client or agent instance to auto-trace all calls.

    Supports: OpenAI, AsyncOpenAI, AzureOpenAI, AsyncAzureOpenAI, Anthropic,
    AsyncAnthropic, google.genai.Client (Gemini + Vertex AI), boto3
    bedrock-runtime, CrewAI Crew, Pydantic AI Agent, DSPy modules, Agno
    Agent/Team/Workflow, Google ADK Runner, Strands Agent, Hermes AIAgent,
    OpenRouter.

        >>> import neatlogs, openai
        >>> client = neatlogs.wrap(openai.OpenAI())
        >>> client.chat.completions.create(...)

    Workflow metadata can be bound at wrap time. Keyword arguments are stamped
    on the root WORKFLOW span as ``neatlogs.workflow.*`` so they show in the
    metadata drawer and trace search. Session identity belongs in
    ``neatlogs.identify(...)``.

        >>> client = neatlogs.wrap(
        ...     openai.OpenAI(),
        ...     project_id="project_123",
        ...     route="/api/v1/copilot/chat",
        ...     surface="copilot",
        ...     mode="fast",
        ... )
        >>> with neatlogs.identify(session_id="chat_123", end_user_id="user_456"):
        ...     client.chat.completions.create(...)
    """
    from ._wrap_utils import make_wrap_context, with_wrap_context

    wrap_context = make_wrap_context(workflow_attributes)

    def _finish(wrapped):
        return with_wrap_context(wrapped, wrap_context)

    cls_name = type(client).__name__
    module = type(client).__module__ or ""

    # Debug: record every wrap() call — the object type and the call site — so a
    # user can confirm WHICH object was wrapped and from WHERE.
    try:
        from .init import is_debug_enabled

        if is_debug_enabled():
            import inspect

            from .core.logger import get_logger

            _c = inspect.stack()[1]
            get_logger().debug(
                f"[neatlogs.wrap] wrapping {cls_name} (module={module}) "
                f"— called from {_c.filename}:{_c.lineno} (in {_c.function})"
            )
    except Exception:
        pass

    # Also fingerprint the full MRO so subclasses defined in user code are
    # detected — e.g. `class QAPipeline(dspy.Module)` has __module__ "src.pipeline"
    # but `dspy.module` appears in a base class. Used as a fallback when the leaf
    # class's own module doesn't reveal the framework.
    mro_modules = " ".join(
        (getattr(base, "__module__", "") or "") for base in type(client).__mro__
    )

    # Azure OpenAI must be checked before plain OpenAI: AzureOpenAI subclasses
    # OpenAI and its module ("openai.lib.azure") contains "openai".
    if cls_name in ("AzureOpenAI", "AsyncAzureOpenAI"):
        from .azure_openai import wrap_async_azure_openai_client, wrap_azure_openai_client

        if "Async" in cls_name:
            return _finish(wrap_async_azure_openai_client(client))
        return _finish(wrap_azure_openai_client(client))

    if "openai" in module or cls_name in ("OpenAI", "AsyncOpenAI"):
        from .openai import wrap_async_openai_client, wrap_openai_client

        if "Async" in cls_name:
            return _finish(wrap_async_openai_client(client))
        return _finish(wrap_openai_client(client))

    if "anthropic" in module or cls_name in ("Anthropic", "AsyncAnthropic"):
        from .anthropic import wrap_anthropic_client, wrap_async_anthropic_client

        if "Async" in cls_name:
            return _finish(wrap_async_anthropic_client(client))
        return _finish(wrap_anthropic_client(client))

    # boto3 bedrock-runtime client — detect by the botocore service model.
    if "botocore" in module or "boto3" in module:
        service = getattr(getattr(client, "meta", None), "service_model", None)
        if getattr(service, "service_name", None) == "bedrock-runtime":
            from .bedrock import wrap_bedrock_client

            return _finish(wrap_bedrock_client(client))

    if ("google" in module and "genai" in module) or cls_name == "Client":
        # A google-genai Client in Vertex mode is traced as vertex_ai; otherwise
        # as google_genai (Gemini / AI Studio).
        from .vertex_ai import _is_vertex_client

        if _is_vertex_client(client):
            from .vertex_ai import wrap_vertex_ai_client

            return _finish(wrap_vertex_ai_client(client))

        from .google_genai import wrap_google_genai_client

        return _finish(wrap_google_genai_client(client))

    if cls_name in ("Crew", "Flow") or "crewai" in mro_modules:
        from .crewai import wrap_crewai

        return _finish(wrap_crewai(client))

    if cls_name == "Agent" and ("pydantic_ai" in module or "pydantic_ai" in mro_modules):
        from .pydantic_ai import wrap_pydantic_ai

        return _finish(wrap_pydantic_ai(client))

    if "dspy" in module or "dspy" in mro_modules:
        from .dspy import wrap_dspy

        return _finish(wrap_dspy(client))

    if "agno" in module or "agno" in mro_modules:
        from .agno import wrap_agno

        return _finish(wrap_agno(client))

    if "google.adk" in module or "google_adk" in module:
        from .google_adk import wrap_google_adk

        return _finish(wrap_google_adk(client))

    if "strands" in module:
        from .strands import strands_hooks

        return _finish(strands_hooks(client))

    # Hermes (NousResearch/hermes-agent): AIAgent lives in the top-level
    # `run_agent` module.
    if cls_name == "AIAgent" or "run_agent" in module:
        from .hermes import wrap_hermes

        return _finish(wrap_hermes(client))

    # OpenRouter official Python SDK — OpenRouter client from `openrouter.sdk`.
    if cls_name == "OpenRouter" or module.startswith("openrouter"):
        from .openrouter import wrap_openrouter_client

        return _finish(wrap_openrouter_client(client))

    # Claude Agent SDK (Anthropic): the agentic loop runs in a CLI SUBPROCESS and is consumed as an
    # async iterator of typed messages (query() / ClaudeSDKClient). There's no in-process LLM client
    # to wrap — we patch the streaming entry points and build spans from the message stream. Accept
    # either the module itself (`neatlogs.wrap(claude_agent_sdk)`) or a ClaudeSDKClient instance.
    if (cls_name == "module" and getattr(client, "__name__", "") == "claude_agent_sdk") \
            or cls_name == "ClaudeSDKClient" or "claude_agent_sdk" in module:
        from .claude_agent_sdk import wrap_claude_agent_sdk

        return _finish(wrap_claude_agent_sdk(client))

    raise TypeError(
        f"neatlogs.wrap() does not support {cls_name} from {module}. "
        "Supported: OpenAI, AsyncOpenAI, AzureOpenAI, AsyncAzureOpenAI, Anthropic, "
        "AsyncAnthropic, google.genai.Client (Gemini + Vertex AI), boto3 bedrock-runtime, "
        "CrewAI Crew, Pydantic AI Agent, DSPy modules, Agno agents, Google ADK Runner, Strands Agent, "
        "Hermes AIAgent, OpenRouter"
    )


__all__ = [
    "init",
    "flush",
    "shutdown",
    "span",
    "trace",
    "identify",
    "inject_trace_context",
    "extract_trace_context",
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
