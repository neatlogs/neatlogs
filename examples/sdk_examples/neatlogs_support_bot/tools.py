"""
CrewAI tools for the support bot.

  - kb_search_tool:           searches KB articles via real OpenAI embeddings
  - past_tickets_search_tool: searches past resolved tickets via embeddings
  - ticket_details_tool:      returns structured ticket information for the crew

The embedding calls inside KB.search() are auto-instrumented and appear as
EMBEDDING spans via the `openai` instrumentation.

RETRIEVER spans wrap the KB/past-ticket search tools. `@span(kind="TOOL")`
would lose the `neatlogs.retrieval.*` attributes, so we use `trace()` inside
the CrewAI tool body per the skill's custom-retriever pattern.
"""

import json
import threading

from crewai.tools import tool

import neatlogs


_ticket_ctx: threading.local = threading.local()


def set_ticket_context(ticket: dict) -> None:
    """Set the current ticket for this thread. Called before crew kickoff."""
    _ticket_ctx.ticket = ticket


def get_ticket_context() -> dict:
    return getattr(_ticket_ctx, "ticket", {})


@tool("kb_search")
def kb_search_tool(query: str) -> str:
    """
    Search the product knowledge base for articles relevant to the given query.
    Returns the top 3 matching articles with their content.
    """
    from kb import KB

    with neatlogs.trace("kb_search", kind="RETRIEVER") as span:
        span.set_attribute("neatlogs.retrieval.query", query)
        span.set_attribute("neatlogs.retrieval.top_k", 3)
        results = KB.search(query, top_k=3)
        span.set_attribute("neatlogs.retrieval.documents", json.dumps(results))

    if not results:
        return "No relevant KB articles found."
    return KB.format_results(results)


@tool("past_tickets_search")
def past_tickets_search_tool(query: str) -> str:
    """
    Search past resolved support tickets for similar issues and their resolutions.
    Returns the top 3 most similar past tickets with resolution notes.
    """
    from kb import PAST_KB

    with neatlogs.trace("past_tickets_search", kind="RETRIEVER") as span:
        span.set_attribute("neatlogs.retrieval.query", query)
        span.set_attribute("neatlogs.retrieval.top_k", 3)
        results = PAST_KB.search(query, top_k=3)
        span.set_attribute("neatlogs.retrieval.documents", json.dumps(results))

    if not results:
        return "No similar past tickets found."
    return PAST_KB.format_results(results)


@tool("get_ticket_details")
def ticket_details_tool(field: str = "all") -> str:
    """
    Retrieve details of the current support ticket.
    Pass field="all" for the full ticket, or a specific field name.
    """
    ticket = get_ticket_context()
    if not ticket:
        return "No ticket context available."

    if field == "all":
        return json.dumps(ticket, indent=2)

    value = ticket.get(field)
    if value is None:
        return f"Field '{field}' not found. Available: {list(ticket.keys())}"
    return str(value)
