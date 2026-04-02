"""
OG (L2) Crew — handles complex technical support tickets.

Mirrors the original support_bot og_crew/crew.py: QueryCrew class + kickoff_crew function.

Span hierarchy (per ticket):
  og_crew_kickoff          [CHAIN  — @neatlogs.span]
  ├── question_extractor   [AGENT/LLM spans — CrewAI + LiteLLM → OI]
  │   └── ticket_details   [tool call]
  ├── kb_rag_expert        [AGENT/LLM spans]
  │   └── kb_search        [RETRIEVER + EMBEDDING spans]
  ├── past_tickets_expert  [AGENT/LLM spans]
  │   └── past_tickets_search [RETRIEVER + EMBEDDING spans]
  └── response_generator   [AGENT/LLM spans]
"""

from crewai import Crew

import neatlogs
from neatlogs.examples.neatlogs_support_bot.crews.og_crew.tasks import make_tasks


@neatlogs.span(kind="CHAIN", name="og_crew_kickoff")
def og_crew_kickoff(ticket: dict) -> str:
    """
    Entry point for the OG (L2) crew. Returns the final email reply text.
    The @span decorator wraps the full 4-agent crew execution in a CHAIN span.
    """
    tasks, agents = make_tasks(ticket)

    crew = Crew(
        agents=agents,
        tasks=tasks,
        verbose=False,
    )

    result = crew.kickoff()
    return result.raw if hasattr(result, "raw") else str(result)
