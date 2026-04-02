"""Shared state definition for the LangGraph research workflow."""

import operator
from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class ResearchState(TypedDict):
    query: str
    plan: str
    # Per-researcher message lists — each branch has its own isolated LLM↔tools loop
    web_messages: Annotated[list, add_messages]
    wiki_messages: Annotated[list, add_messages]
    arxiv_messages: Annotated[list, add_messages]
    # Final extracted results written when each branch loop completes
    web_results: str
    wiki_results: str
    arxiv_results: str
    synthesis: str
    final_report: str
    messages: Annotated[list, operator.add]
