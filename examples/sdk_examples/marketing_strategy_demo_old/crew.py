"""
Marketing Strategy Crew.

Assembles the agents and tasks into a sequential CrewAI crew and runs it.
"""

from crewai import Crew, Process
from agents import (
    create_lead_market_analyst,
    create_chief_marketing_strategist,
    create_creative_content_creator,
)
from task import create_tasks


def run_marketing_crew(inputs: dict) -> str:
    """
    Build and run the marketing strategy crew.

    Args:
        inputs: dict with keys 'customer_domain' and 'project_description'.
                These are interpolated into every task description via CrewAI's
                {variable} syntax.

    Returns:
        The final crew output as a string.
    """
    # Create agents
    analyst = create_lead_market_analyst()
    strategist = create_chief_marketing_strategist()
    creator = create_creative_content_creator()

    # Create tasks wired to agents (inputs needed for template compilation)
    tasks = create_tasks(analyst, strategist, creator, inputs)

    # Assemble the crew
    crew = Crew(
        agents=[analyst, strategist, creator],
        tasks=tasks,
        process=Process.sequential,
        verbose=True,
    )

    # Run
    result = crew.kickoff(inputs=inputs)

    return str(result)
