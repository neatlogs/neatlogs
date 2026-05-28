"""
Marketing Strategy Crew — assembles agents + tasks and runs them sequentially.
"""

import neatlogs
from crewai import Crew, Process

from agents import (
    create_lead_market_analyst,
    create_chief_marketing_strategist,
    create_creative_content_creator,
)
from task import create_tasks


@neatlogs.span(
    kind="WORKFLOW",
    name="marketing_strategy_workflow",
    description="Run the 3-agent marketing strategy CrewAI workflow",
)
def run_marketing_crew(inputs: dict) -> str:
    """
    Build and run the marketing strategy crew.

    Args:
        inputs: dict with keys 'customer_domain' and 'project_description'.

    Returns:
        The final crew output as a string.
    """
    analyst = create_lead_market_analyst()
    strategist = create_chief_marketing_strategist()
    creator = create_creative_content_creator()

    tasks = create_tasks(analyst, strategist, creator, inputs)

    crew = Crew(
        agents=[analyst, strategist, creator],
        tasks=tasks,
        process=Process.sequential,
    )

    result = crew.kickoff(inputs=inputs)
    return str(result)
