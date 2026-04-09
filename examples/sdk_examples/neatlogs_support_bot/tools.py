"""
CrewAI tools for the support bot.

All tools make real calls — no hardcoded mock responses.
  - kb_search_tool:           searches KB articles via real OpenAI embeddings
  - past_tickets_search_tool: searches past resolved tickets via embeddings
  - ticket_details_tool:      returns structured ticket information for the crew

The embedding calls inside KB.search() are auto-instrumented by OpenInference
and appear as EMBEDDING spans in the trace.
"""

import json
import threading

from crewai.tools import tool

# KB is imported at call time (inside each tool function) to avoid circular
# imports and to ensure neatlogs.init() has already run when modules are loaded.

# ---------------------------------------------------------------------------
# Thread-local storage for passing the current ticket context to tools
# (same pattern as the original support bot's ticket_detail_tools)
# ---------------------------------------------------------------------------

_ticket_ctx: threading.local = threading.local()


def set_ticket_context(ticket: dict) -> None:
    """Set the current ticket for this thread. Called by crew_selector before kickoff."""
    _ticket_ctx.ticket = ticket


def get_ticket_context() -> dict:
    return getattr(_ticket_ctx, "ticket", {})


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@tool("kb_search")
def kb_search_tool(query: str) -> str:
    """
    Search the product knowledge base for articles relevant to the given query.
    Returns the top 3 matching articles with their content.
    Use this to find how-to guides, policies, and feature documentation.
    """
    from neatlogs.examples.neatlogs_support_bot.kb import KB
    import neatlogs

    with neatlogs.trace(kind="RETRIEVER", name="kb_search") as span:
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
    Use this to find patterns, precedents, and proven solutions.
    """
    from neatlogs.examples.neatlogs_support_bot.kb import PAST_KB
    import neatlogs

    with neatlogs.trace(kind="RETRIEVER", name="past_tickets_search") as span:
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
    Pass field="all" for the full ticket, or a specific field name such as:
    "subject", "body", "customer_email", "account_plan", "priority".
    """
    ticket = get_ticket_context()
    if not ticket:
        return "No ticket context available."

    if field == "all":
        return json.dumps(ticket, indent=2)

    value = ticket.get(field)
    if value is None:
        return f"Field '{field}' not found in ticket. Available fields: {list(ticket.keys())}"
    return str(value)
