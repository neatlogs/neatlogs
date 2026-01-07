"""
Neatlogs Instrumentation Manager
===============================
This module manages auto-instrumentation for LLM providers and frameworks
within the Neatlogs system using OpenInference.
"""

import importlib.util
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


def instrument_all(instrumentations: Optional[List[str]] = None):
    """
    Instrument all supported and available libraries.

    Args:
        instrumentations (List[str], optional): List of frameworks to instrument.
                                                If None, all available supported frameworks are instrumented.
                                                Supported: "openai", "openai-agents", "langchain", "anthropic",
                                                "google-genai", "crewai", "groq", "litellm", "llama-index",
                                                "google-adk", "agno", "bedrock", "dspy", "guardrails", "haystack",
                                                "instructor", "mcp", "mistralai", "portkey", "pydantic-ai",
                                                "smolagents", "vertexai", "beeai", "autogen-agentchat".

    This function checks for the presence of OpenInference instrumentation libraries
    and initializes them if found.
    """

    # Helper to check if we should instrument a specific framework
    def should_instrument(name):
        if instrumentations is not None:
            return name in instrumentations
        return True

    # OpenAI Agents
    # The OpenAI Agents SDK is imported as 'agents'
    # Check for OpenAI Agents first, as it uses OpenAI SDK internally
    if should_instrument("openai-agents"):
        has_agents = importlib.util.find_spec("agents") is not None

        if has_agents:
            if importlib.util.find_spec("openinference.instrumentation.openai_agents"):
                try:
                    from openinference.instrumentation.openai_agents import (
                        OpenAIAgentsInstrumentor,
                    )

                    OpenAIAgentsInstrumentor().instrument()
                    logger.info(
                        "Neatlogs: OpenAI Agents instrumentation enabled.")
                except Exception as e:
                    logger.warning(
                        f"Neatlogs: Failed to instrument OpenAI Agents: {e}")
            else:
                logger.warning(
                    "Neatlogs: Detected 'agents' but 'openinference-instrumentation-openai-agents' is not installed. "
                    "Install with: uv add neatlogs[openai-agents]"
                )

    # OpenAI - instrument regardless of whether Agents is present (they can coexist)
    if should_instrument("openai"):
        if importlib.util.find_spec("openai"):
            if importlib.util.find_spec("openinference.instrumentation.openai"):
                try:
                    from openinference.instrumentation.openai import OpenAIInstrumentor

                    OpenAIInstrumentor().instrument()
                    logger.info("Neatlogs: OpenAI instrumentation enabled.")
                except Exception as e:
                    logger.warning(
                        f"Neatlogs: Failed to instrument OpenAI: {e}")
            else:
                logger.warning(
                    "Neatlogs: Detected 'openai' but 'openinference-instrumentation-openai' is not installed. "
                    "Install with: uv add neatlogs[openai]"
                )

    # LangChain
    if should_instrument("langchain"):
        if importlib.util.find_spec("langchain"):
            if importlib.util.find_spec("openinference.instrumentation.langchain"):
                try:
                    from openinference.instrumentation.langchain import (
                        LangChainInstrumentor,
                    )

                    LangChainInstrumentor().instrument()
                    
                    logger.info("Neatlogs: LangChain instrumentation enabled.")
                except Exception as e:
                    logger.warning(
                        f"Neatlogs: Failed to instrument LangChain: {e}")
            else:
                logger.warning(
                    "Neatlogs: Detected 'langchain' but 'openinference-instrumentation-langchain' is not installed. "
                    "Install with: uv add neatlogs[langchain]"
                )

    # Anthropic
    if should_instrument("anthropic"):
        if importlib.util.find_spec("anthropic"):
            if importlib.util.find_spec("openinference.instrumentation.anthropic"):
                try:
                    from openinference.instrumentation.anthropic import (
                        AnthropicInstrumentor,
                    )

                    AnthropicInstrumentor().instrument()
                    logger.info("Neatlogs: Anthropic instrumentation enabled.")
                except Exception as e:
                    logger.warning(
                        f"Neatlogs: Failed to instrument Anthropic: {e}")
            else:
                logger.warning(
                    "Neatlogs: Detected 'anthropic' but 'openinference-instrumentation-anthropic' is not installed. "
                    "Install with: uv add neatlogs[anthropic]"
                )

    # Google GenAI
    if should_instrument("google-genai"):
        if importlib.util.find_spec("google.genai"):
            if importlib.util.find_spec("openinference.instrumentation.google_genai"):
                try:
                    from openinference.instrumentation.google_genai import (
                        GoogleGenAIInstrumentor,
                    )

                    GoogleGenAIInstrumentor().instrument()
                    logger.info(
                        "Neatlogs: Google GenAI instrumentation enabled.")
                except Exception as e:
                    logger.warning(
                        f"Neatlogs: Failed to instrument Google GenAI: {e}")
            else:
                logger.warning(
                    "Neatlogs: Detected 'google.genai' but 'openinference-instrumentation-google-genai' is not installed. "
                    "Install with: uv add neatlogs[google-genai]"
                )

    # CrewAI
    if should_instrument("crewai"):
        if importlib.util.find_spec("crewai"):
            if importlib.util.find_spec("openinference.instrumentation.crewai"):
                try:
                    from openinference.instrumentation.crewai import CrewAIInstrumentor

                    CrewAIInstrumentor().instrument()
                    logger.info("Neatlogs: CrewAI instrumentation enabled.")
                except Exception as e:
                    logger.warning(
                        f"Neatlogs: Failed to instrument CrewAI: {e}")
            else:
                logger.warning(
                    "Neatlogs: Detected 'crewai' but 'openinference-instrumentation-crewai' is not installed. "
                    "Install with: uv add neatlogs[crewai]"
                )

    # Groq
    if should_instrument("groq"):
        if importlib.util.find_spec("groq"):
            if importlib.util.find_spec("openinference.instrumentation.groq"):
                try:
                    from openinference.instrumentation.groq import GroqInstrumentor

                    GroqInstrumentor().instrument()
                    logger.info("Neatlogs: Groq instrumentation enabled.")
                except Exception as e:
                    logger.warning(f"Neatlogs: Failed to instrument Groq: {e}")
            else:
                logger.warning(
                    "Neatlogs: Detected 'groq' but 'openinference-instrumentation-groq' is not installed. "
                    "Install with: uv add neatlogs[groq]"
                )

    # LiteLLM
    if should_instrument("litellm"):
        if importlib.util.find_spec("litellm"):
            if importlib.util.find_spec("openinference.instrumentation.litellm"):
                try:
                    from openinference.instrumentation.litellm import (
                        LiteLLMInstrumentor,
                    )

                    LiteLLMInstrumentor().instrument()
                    logger.info("Neatlogs: LiteLLM instrumentation enabled.")
                except Exception as e:
                    logger.warning(
                        f"Neatlogs: Failed to instrument LiteLLM: {e}")
            else:
                logger.warning(
                    "Neatlogs: Detected 'litellm' but 'openinference-instrumentation-litellm' is not installed. "
                    "Install with: uv add neatlogs[litellm]"
                )

    # LlamaIndex
    if should_instrument("llama-index"):
        if importlib.util.find_spec("llama-index"):
            if importlib.util.find_spec("openinference.instrumentation.llama-index"):
                try:
                    from openinference.instrumentation.llama_index import LlamaIndexInstrumentor

                    LlamaIndexInstrumentor().instrument()
                    logger.info(
                        "Neatlogs: LlamaIndex instrumentation enabled.")
                except Exception as e:
                    logger.warning(
                        f"Neatlogs: Failed to instrument LlamaIndex: {e}")
            else:
                logger.warning(
                    "Neatlogs: Detected 'llama-index' but 'openinference-instrumentation-llama-index' is not installed. "
                    "Install with: uv add neatlogs[llama-index]"
                )

    # Google ADK
    if should_instrument("google-adk"):
        if importlib.util.find_spec("google-adk"):
            if importlib.util.find_spec("openinference.instrumentation.google-adk"):
                try:
                    from openinference.instrumentation.google_adk import GoogleADKInstrumentor

                    GoogleADKInstrumentor().instrument()
                    logger.info(
                        "Neatlogs: Google ADK instrumentation enabled.")
                except Exception as e:
                    logger.warning(
                        f"Neatlogs: Failed to instrument Google ADK: {e}")
            else:
                logger.warning(
                    "Neatlogs: Detected 'google-adk' but 'openinference-instrumentation-google-adk' is not installed. "
                    "Install with: uv add neatlogs[google-adk]"
                )

    # Agno
    if should_instrument("agno"):
        if importlib.util.find_spec("agno"):
            if importlib.util.find_spec("openinference.instrumentation.agno"):
                try:
                    from openinference.instrumentation.agno import AgnoInstrumentor

                    AgnoInstrumentor().instrument()
                    logger.info("Neatlogs: Agno instrumentation enabled.")
                except Exception as e:
                    logger.warning(
                        f"Neatlogs: Failed to instrument Agno: {e}")
            else:
                logger.warning(
                    "Neatlogs: Detected 'agno' but 'openinference-instrumentation-agno' is not installed. "
                    "Install with: uv add neatlogs[agno]"
                )

    # AWS Bedrock
    if should_instrument("bedrock"):
        if importlib.util.find_spec("boto3"):
            if importlib.util.find_spec("openinference.instrumentation.bedrock"):
                try:
                    from openinference.instrumentation.bedrock import BedrockInstrumentor

                    BedrockInstrumentor().instrument()
                    logger.info(
                        "Neatlogs: AWS Bedrock instrumentation enabled.")
                except Exception as e:
                    logger.warning(
                        f"Neatlogs: Failed to instrument AWS Bedrock: {e}")
            else:
                logger.warning(
                    "Neatlogs: Detected 'boto3' but 'openinference-instrumentation-bedrock' is not installed. "
                    "Install with: uv add neatlogs[bedrock]"
                )

    # DSPy
    if should_instrument("dspy"):
        if importlib.util.find_spec("dspy"):
            if importlib.util.find_spec("openinference.instrumentation.dspy"):
                try:
                    from openinference.instrumentation.dspy import DSPyInstrumentor

                    DSPyInstrumentor().instrument()
                    logger.info("Neatlogs: DSPy instrumentation enabled.")
                except Exception as e:
                    logger.warning(
                        f"Neatlogs: Failed to instrument DSPy: {e}")
            else:
                logger.warning(
                    "Neatlogs: Detected 'dspy' but 'openinference-instrumentation-dspy' is not installed. "
                    "Install with: uv add neatlogs[dspy]"
                )

    # Guardrails
    if should_instrument("guardrails"):
        if importlib.util.find_spec("guardrails") or importlib.util.find_spec("guardrails_ai"):
            if importlib.util.find_spec("openinference.instrumentation.guardrails"):
                try:
                    from openinference.instrumentation.guardrails import GuardrailsInstrumentor

                    GuardrailsInstrumentor().instrument()
                    logger.info(
                        "Neatlogs: Guardrails instrumentation enabled.")
                except Exception as e:
                    logger.warning(
                        f"Neatlogs: Failed to instrument Guardrails: {e}")
            else:
                logger.warning(
                    "Neatlogs: Detected 'guardrails' but 'openinference-instrumentation-guardrails' is not installed. "
                    "Install with: uv add neatlogs[guardrails]"
                )

    # Haystack
    if should_instrument("haystack"):
        if importlib.util.find_spec("haystack"):
            if importlib.util.find_spec("openinference.instrumentation.haystack"):
                try:
                    from openinference.instrumentation.haystack import HaystackInstrumentor

                    HaystackInstrumentor().instrument()
                    logger.info("Neatlogs: Haystack instrumentation enabled.")
                except Exception as e:
                    logger.warning(
                        f"Neatlogs: Failed to instrument Haystack: {e}")
            else:
                logger.warning(
                    "Neatlogs: Detected 'haystack' but 'openinference-instrumentation-haystack' is not installed. "
                    "Install with: uv add neatlogs[haystack]"
                )

    # Instructor
    if should_instrument("instructor"):
        if importlib.util.find_spec("instructor"):
            if importlib.util.find_spec("openinference.instrumentation.instructor"):
                try:
                    from openinference.instrumentation.instructor import InstructorInstrumentor

                    InstructorInstrumentor().instrument()
                    logger.info(
                        "Neatlogs: Instructor instrumentation enabled.")
                except Exception as e:
                    logger.warning(
                        f"Neatlogs: Failed to instrument Instructor: {e}")
            else:
                logger.warning(
                    "Neatlogs: Detected 'instructor' but 'openinference-instrumentation-instructor' is not installed. "
                    "Install with: uv add neatlogs[instructor]"
                )

    # MCP
    if should_instrument("mcp"):
        if importlib.util.find_spec("mcp"):
            if importlib.util.find_spec("openinference.instrumentation.mcp"):
                try:
                    from openinference.instrumentation.mcp import MCPInstrumentor

                    MCPInstrumentor().instrument()
                    logger.info("Neatlogs: MCP instrumentation enabled.")
                except Exception as e:
                    logger.warning(
                        f"Neatlogs: Failed to instrument MCP: {e}")
            else:
                logger.warning(
                    "Neatlogs: Detected 'mcp' but 'openinference-instrumentation-mcp' is not installed. "
                    "Install with: uv add neatlogs[mcp]"
                )

    # MistralAI
    if should_instrument("mistralai"):
        if importlib.util.find_spec("mistralai"):
            if importlib.util.find_spec("openinference.instrumentation.mistralai"):
                try:
                    from openinference.instrumentation.mistralai import MistralAIInstrumentor

                    MistralAIInstrumentor().instrument()
                    logger.info("Neatlogs: MistralAI instrumentation enabled.")
                except Exception as e:
                    logger.warning(
                        f"Neatlogs: Failed to instrument MistralAI: {e}")
            else:
                logger.warning(
                    "Neatlogs: Detected 'mistralai' but 'openinference-instrumentation-mistralai' is not installed. "
                    "Install with: uv add neatlogs[mistralai]"
                )

    # Portkey
    if should_instrument("portkey"):
        if importlib.util.find_spec("portkey"):
            if importlib.util.find_spec("openinference.instrumentation.portkey"):
                try:
                    from openinference.instrumentation.portkey import PortkeyInstrumentor

                    PortkeyInstrumentor().instrument()
                    logger.info("Neatlogs: Portkey instrumentation enabled.")
                except Exception as e:
                    logger.warning(
                        f"Neatlogs: Failed to instrument Portkey: {e}")
            else:
                logger.warning(
                    "Neatlogs: Detected 'portkey' but 'openinference-instrumentation-portkey' is not installed. "
                    "Install with: uv add neatlogs[portkey]"
                )

    # PydanticAI
    if should_instrument("pydantic-ai"):
        if importlib.util.find_spec("pydantic_ai"):
            if importlib.util.find_spec("openinference.instrumentation.pydantic_ai"):
                try:
                    from openinference.instrumentation.pydantic_ai import PydanticAIInstrumentor

                    PydanticAIInstrumentor().instrument()
                    logger.info(
                        "Neatlogs: PydanticAI instrumentation enabled.")
                except Exception as e:
                    logger.warning(
                        f"Neatlogs: Failed to instrument PydanticAI: {e}")
            else:
                logger.warning(
                    "Neatlogs: Detected 'pydantic_ai' but 'openinference-instrumentation-pydantic-ai' is not installed. "
                    "Install with: uv add neatlogs[pydantic-ai]"
                )

    # smolagents
    if should_instrument("smolagents"):
        if importlib.util.find_spec("smolagents"):
            if importlib.util.find_spec("openinference.instrumentation.smolagents"):
                try:
                    from openinference.instrumentation.smolagents import SmolagentsInstrumentor

                    SmolagentsInstrumentor().instrument()
                    logger.info(
                        "Neatlogs: smolagents instrumentation enabled.")
                except Exception as e:
                    logger.warning(
                        f"Neatlogs: Failed to instrument smolagents: {e}")
            else:
                logger.warning(
                    "Neatlogs: Detected 'smolagents' but 'openinference-instrumentation-smolagents' is not installed. "
                    "Install with: uv add neatlogs[smolagents]"
                )

    # VertexAI
    if should_instrument("vertexai"):
        if importlib.util.find_spec("vertexai") or importlib.util.find_spec("google.cloud.aiplatform"):
            if importlib.util.find_spec("openinference.instrumentation.vertexai"):
                try:
                    from openinference.instrumentation.vertexai import VertexAIInstrumentor

                    VertexAIInstrumentor().instrument()
                    logger.info("Neatlogs: VertexAI instrumentation enabled.")
                except Exception as e:
                    logger.warning(
                        f"Neatlogs: Failed to instrument VertexAI: {e}")
            else:
                logger.warning(
                    "Neatlogs: Detected 'vertexai' but 'openinference-instrumentation-vertexai' is not installed. "
                    "Install with: uv add neatlogs[vertexai]"
                )

    # Autogen AgentChat
    if should_instrument("autogen-agentchat"):
        if importlib.util.find_spec("autogen") or importlib.util.find_spec("autogen_agentchat"):
            if importlib.util.find_spec("openinference.instrumentation.autogen_agentchat"):
                try:
                    from openinference.instrumentation.autogen_agentchat import AutogenAgentchatInstrumentor

                    AutogenAgentchatInstrumentor().instrument()
                    logger.info(
                        "Neatlogs: Autogen AgentChat instrumentation enabled.")
                except Exception as e:
                    logger.warning(
                        f"Neatlogs: Failed to instrument Autogen AgentChat: {e}")
            else:
                logger.warning(
                    "Neatlogs: Detected 'autogen' but 'openinference-instrumentation-autogen-agentchat' is not installed. "
                    "Install with: uv add neatlogs[autogen-agentchat]"
                )

    logger.info("Neatlogs: Auto-instrumentation setup complete.")
