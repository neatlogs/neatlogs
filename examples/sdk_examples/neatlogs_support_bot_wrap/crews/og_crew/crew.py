"""
OG (L2) Crew — handles complex technical support tickets.

Four agents (question_extractor → kb_rag_expert → past_tickets_expert →
response_generator) chained via task context.
"""
import neatlogs

from crewai import Crew

from crews.og_crew.tasks import make_tasks
from tools import ticket_workflow_metadata


def og_crew_kickoff(ticket: dict) -> str:
    tasks, agents = make_tasks(ticket)

    crew = neatlogs.wrap(
        Crew(
            agents=agents,
            tasks=tasks,
            verbose=False,
        ),
        **ticket_workflow_metadata(ticket, crew="og"),
    )

    result = crew.kickoff()
    return result.raw if hasattr(result, "raw") else str(result)
