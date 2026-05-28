"""
OG (L2) Crew — handles complex technical support tickets.

Four agents (question_extractor → kb_rag_expert → past_tickets_expert →
response_generator) chained via task context.
"""

from crewai import Crew

import neatlogs
from crews.og_crew.tasks import make_tasks


@neatlogs.span(kind="CHAIN", name="og_crew_kickoff")
def og_crew_kickoff(ticket: dict) -> str:
    tasks, agents = make_tasks(ticket)

    crew = Crew(
        agents=agents,
        tasks=tasks,
        verbose=False,
    )

    result = crew.kickoff()
    return result.raw if hasattr(result, "raw") else str(result)
