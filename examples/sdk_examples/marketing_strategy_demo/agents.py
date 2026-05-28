"""
Marketing Strategy Agents.

Three specialized agents that collaborate sequentially:
  1. Lead Market Analyst       — researches the company, competitors, and audience
  2. Chief Marketing Strategist — synthesises research into a strategy
  3. Creative Content Creator   — produces campaign ideas and ad copy

Uses Azure OpenAI via crewai.LLM with neatlogs.bind_templates(). bind_templates
attaches the system prompt template to every LLM span this agent creates,
making the prompt visible in the NeatLogs trace view.

System templates MUST NOT have required placeholders because bind_templates()
calls system_tpl.compile() with no arguments. Pre-render dynamic values.
"""

import os
from typing import Optional

import neatlogs
from crewai import Agent, LLM
from neatlogs import SystemPromptTemplate
from tools import search_web, analyze_website


def _make_llm() -> LLM:
    """Create a fresh Azure OpenAI LLM instance (required by each Agent)."""
    return LLM(
        model="azure/" + os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-5-nano"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        base_url=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview"),
        drop_params=True,
        additional_drop_params=["stop", "temperature"],
    )


def _make_agent(
    role: str,
    goal: str,
    backstory: str,
    tools: Optional[list] = None,
    allow_delegation: bool = False,
    max_iter: int = 3,
) -> Agent:
    """Create an Agent with a fresh LLM instance and bound system prompt template."""
    system_tpl = SystemPromptTemplate(backstory)
    bound_llm = neatlogs.bind_templates(_make_llm(), system_tpl)
    return Agent(
        role=role,
        goal=goal,
        backstory=str(system_tpl.template),
        llm=bound_llm,
        tools=tools or [],
        verbose=True,
        allow_delegation=allow_delegation,
        max_iter=max_iter,
    )


_ANALYST_BACKSTORY = (
    "You are a Lead Market Analyst at a premier digital marketing firm. "
    "You specialise in dissecting online business landscapes, identifying "
    "competitor positioning, and uncovering audience demographics. "
    "You always ground your analysis in data and cite your sources. "
    "Be efficient: use at most 3 web searches total — each search should be "
    "targeted and purposeful. Do not repeat similar queries."
)


def create_lead_market_analyst() -> Agent:
    return _make_agent(
        role="Lead Market Analyst",
        goal=(
            "Conduct thorough analysis of the company and competitors, "
            "providing data-driven insights to guide marketing strategies."
        ),
        backstory=_ANALYST_BACKSTORY,
        tools=[search_web, analyze_website],
    )


_STRATEGIST_BACKSTORY = (
    "You are the Chief Marketing Strategist at a leading digital marketing "
    "agency, known for crafting bespoke strategies that drive measurable "
    "results. You synthesise market research into actionable plans with "
    "clear KPIs. "
    "Be efficient: use at most 2 web searches — only search when the research "
    "context is insufficient. Prioritise synthesis over additional lookups."
)


def create_chief_marketing_strategist() -> Agent:
    return _make_agent(
        role="Chief Marketing Strategist",
        goal=(
            "Synthesise market research insights into a comprehensive, "
            "actionable marketing strategy with clear KPIs."
        ),
        backstory=_STRATEGIST_BACKSTORY,
        allow_delegation=False,
    )


_CREATOR_BACKSTORY = (
    "You are a Creative Content Creator at a top-tier digital marketing "
    "agency. You excel at turning marketing strategies into engaging "
    "stories and compelling ad copy that captures attention and inspires "
    "action."
)


def create_creative_content_creator() -> Agent:
    return _make_agent(
        role="Creative Content Creator",
        goal=(
            "Develop compelling and innovative campaign ideas and ad copy "
            "that align with the marketing strategy and resonate with the "
            "target audience."
        ),
        backstory=_CREATOR_BACKSTORY,
    )
