"""
Marketing Strategy Agents.

Three specialized agents that collaborate sequentially:
  1. Lead Market Analyst       -- researches the company, competitors, and audience
  2. Chief Marketing Strategist -- synthesises research into a strategy
  3. Creative Content Creator   -- produces campaign ideas and ad copy

Uses Azure OpenAI via crewai.LLM with Neatlogs auto-instrumentation.
Thoughts (Thought / Action / Observation) are captured automatically by
CrewAI's ReAct loop and the Neatlogs SDK.
"""

import os
from typing import Optional

from crewai import Agent, LLM
from tools import search_web, analyze_website


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
    backstory: str,
    tools: Optional[list] = None,
    allow_delegation: bool = False,
) -> Agent:
    """Create an Agent with a fresh LLM instance."""
    return Agent(
        role=role,
        goal=goal,
        backstory=backstory,
        llm=_make_llm(),
        tools=tools or [],
        verbose=True,
        allow_delegation=allow_delegation,
    )


# ---------------------------------------------------------------------------
# 1. Lead Market Analyst
# ---------------------------------------------------------------------------
_ANALYST_BACKSTORY = (
    "You are a Lead Market Analyst at a premier digital marketing firm. "
    "You specialise in dissecting online business landscapes, identifying "
    "competitor positioning, and uncovering audience demographics. "
    "You always ground your analysis in data and cite your sources. "
    "Think step-by-step through your research process."
)


def create_lead_market_analyst() -> Agent:
    """Create the Lead Market Analyst agent."""
    return _make_agent(
        role="Lead Market Analyst",
        goal=(
            "Conduct thorough analysis of the company and competitors, "
            "providing data-driven insights to guide marketing strategies."
        ),
        backstory=_ANALYST_BACKSTORY,
        tools=[search_web, analyze_website],
    )


# ---------------------------------------------------------------------------
# 2. Chief Marketing Strategist
# ---------------------------------------------------------------------------
_STRATEGIST_BACKSTORY = (
    "You are the Chief Marketing Strategist at a leading digital marketing "
    "agency, known for crafting bespoke strategies that drive measurable "
    "results. You synthesise market research into actionable plans with "
    "clear KPIs. Think carefully about which channels and tactics will "
    "have the highest ROI for the target audience."
)


def create_chief_marketing_strategist() -> Agent:
    """Create the Chief Marketing Strategist agent."""
    return _make_agent(
        role="Chief Marketing Strategist",
        goal=(
            "Synthesise market research insights into a comprehensive, "
            "actionable marketing strategy with clear KPIs."
        ),
        backstory=_STRATEGIST_BACKSTORY,
        tools=[search_web],
        allow_delegation=True,
    )


# ---------------------------------------------------------------------------
# 3. Creative Content Creator
# ---------------------------------------------------------------------------
_CREATOR_BACKSTORY = (
    "You are a Creative Content Creator at a top-tier digital marketing "
    "agency. You excel at turning marketing strategies into engaging "
    "stories and compelling ad copy that captures attention and inspires "
    "action. You think about what will resonate emotionally with the "
    "target audience and always provide multiple creative options."
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
        backstory=_CREATOR_BACKSTORY,
    )
