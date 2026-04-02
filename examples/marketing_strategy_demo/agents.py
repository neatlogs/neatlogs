"""
Marketing Strategy Agents.

Three specialized agents that collaborate sequentially:
  1. Lead Market Analyst       -- researches the company, competitors, and audience
  2. Chief Marketing Strategist -- synthesises research into a strategy
  3. Creative Content Creator   -- produces campaign ideas and ad copy

Uses Azure OpenAI via crewai.LLM with Neatlogs prompt-template tracking.
Thoughts (Thought / Action / Observation) are captured automatically by
CrewAI's ReAct loop and the Neatlogs SDK.
"""

import os
from typing import Optional

import neatlogs
from crewai import Agent, LLM
from tools import search_web, analyze_website


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_RUNTIME_VARS = {
    "customer_domain": "(provided at runtime)",
    "project_description": "(provided at runtime)",
}


def _make_llm() -> LLM:
    """Create a fresh Azure OpenAI LLM instance (crewAI-native)."""
    return LLM(
        model="azure/" + os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        base_url=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
    )


def _make_agent(
    role: str,
    goal: str,
    system_tpl: neatlogs.PromptTemplate,
    user_tpl: neatlogs.UserPromptTemplate,
    tools: Optional[list] = None,
    allow_delegation: bool = False,
) -> Agent:
    """Create an Agent with Neatlogs prompt-template binding."""
    return Agent(
        role=role,
        goal=goal,
        backstory=str(system_tpl.template),
        llm=neatlogs.bind_templates(
            _make_llm(), system_tpl, user_tpl, **_RUNTIME_VARS,
        ),
        tools=tools or [],
        verbose=True,
        allow_delegation=allow_delegation,
    )


# ---------------------------------------------------------------------------
# 1. Lead Market Analyst
# ---------------------------------------------------------------------------
_analyst_system = neatlogs.PromptTemplate(
    "You are a Lead Market Analyst at a premier digital marketing firm. "
    "You specialise in dissecting online business landscapes, identifying "
    "competitor positioning, and uncovering audience demographics. "
    "You always ground your analysis in data and cite your sources. "
    "Think step-by-step through your research process."
)

_analyst_user = neatlogs.UserPromptTemplate(
    "Research the company at {{customer_domain}} and its competitors. "
    "Provide an in-depth report covering:\n"
    "1. What the company does and its unique value proposition\n"
    "2. Target audience demographics and preferences\n"
    "3. Top 3 competitors and their positioning\n"
    "4. Market trends relevant to the industry\n"
    "5. Opportunities the company can exploit\n\n"
    "Project context: {{project_description}}"
)


def create_lead_market_analyst() -> Agent:
    """Create the Lead Market Analyst agent."""
    return _make_agent(
        role="Lead Market Analyst",
        goal=(
            "Conduct thorough analysis of the company and competitors, "
            "providing data-driven insights to guide marketing strategies."
        ),
        system_tpl=_analyst_system,
        user_tpl=_analyst_user,
        tools=[search_web, analyze_website],
    )


# ---------------------------------------------------------------------------
# 2. Chief Marketing Strategist
# ---------------------------------------------------------------------------
_strategist_system = neatlogs.PromptTemplate(
    "You are the Chief Marketing Strategist at a leading digital marketing "
    "agency, known for crafting bespoke strategies that drive measurable "
    "results. You synthesise market research into actionable plans with "
    "clear KPIs. Think carefully about which channels and tactics will "
    "have the highest ROI for the target audience."
)

_strategist_user = neatlogs.UserPromptTemplate(
    "Using the market research provided, formulate a comprehensive "
    "marketing strategy for {{customer_domain}}.\n\n"
    "Project: {{project_description}}\n\n"
    "Your strategy must include:\n"
    "1. Strategy name and executive summary\n"
    "2. Specific tactics (at least 3)\n"
    "3. Marketing channels to use\n"
    "4. KPIs to measure success\n"
    "5. Rationale for each recommendation"
)


def create_chief_marketing_strategist() -> Agent:
    """Create the Chief Marketing Strategist agent."""
    return _make_agent(
        role="Chief Marketing Strategist",
        goal=(
            "Synthesise market research insights into a comprehensive, "
            "actionable marketing strategy with clear KPIs."
        ),
        system_tpl=_strategist_system,
        user_tpl=_strategist_user,
        tools=[search_web],
        allow_delegation=True,   # can delegate research back to analyst
    )


# ---------------------------------------------------------------------------
# 3. Creative Content Creator
# ---------------------------------------------------------------------------
_creator_system = neatlogs.PromptTemplate(
    "You are a Creative Content Creator at a top-tier digital marketing "
    "agency. You excel at turning marketing strategies into engaging "
    "stories and compelling ad copy that captures attention and inspires "
    "action. You think about what will resonate emotionally with the "
    "target audience and always provide multiple creative options."
)

_creator_user = neatlogs.UserPromptTemplate(
    "Based on the marketing strategy, create campaign content for "
    "{{customer_domain}}.\n\n"
    "Project: {{project_description}}\n\n"
    "Deliver:\n"
    "1. Campaign idea: name, description, target audience, primary channel\n"
    "2. Ad copy: a compelling title and body text\n"
    "3. The copy must align with the strategy and speak to the audience"
)


def create_creative_content_creator() -> Agent:
    """Create the Creative Content Creator agent."""
    return _make_agent(
        role="Creative Content Creator",
        goal=(
            "Develop compelling and innovative campaign ideas and ad copy "
            "that align with the marketing strategy and resonate with the "
            "target audience."
        ),
        system_tpl=_creator_system,
        user_tpl=_creator_user,
    )
