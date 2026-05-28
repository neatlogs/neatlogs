"""
OG (L2) Crew Agents.

Four agents:
  1. question_extractor      — parses the ticket and extracts precise questions
  2. kb_rag_expert           — retrieves relevant KB articles via semantic search
  3. past_tickets_rag_expert — finds similar resolved tickets and their patterns
  4. response_generator      — synthesizes all context into a professional email reply
"""

from crewai import Agent, LLM

import neatlogs
from neatlogs import SystemPromptTemplate

from config import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_ENDPOINT,
    OPENAI_API_VERSION,
)
from tools import kb_search_tool, past_tickets_search_tool, ticket_details_tool


def _build_llm() -> LLM:
    return LLM(
        model="azure/gpt-4o-mini",
        base_url=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=OPENAI_API_VERSION,
        temperature=0.2,
        top_p=0.9,
        max_completion_tokens=800,
    )


def make_question_extractor() -> Agent:
    system_tpl = SystemPromptTemplate(
        "You are a meticulous support triage specialist. You read customer tickets "
        "carefully to identify all questions, implied concerns, and the account context."
    )
    return Agent(
        role="Ticket Question Extractor",
        goal=(
            "Read the full support ticket and extract every distinct question or issue."
        ),
        backstory=str(system_tpl.template),
        tools=[ticket_details_tool],
        llm=neatlogs.bind_templates(_build_llm(), system_tpl),
        allow_delegation=False,
        max_iter=8,
        max_retry_limit=3,
    )


def make_kb_rag_expert() -> Agent:
    system_tpl = SystemPromptTemplate(
        "You are an expert in retrieval-augmented generation. You rephrase questions "
        "into short, high-signal search queries. You cite the KB article IDs you used."
    )
    return Agent(
        role="KB RAG Expert",
        goal=(
            "For each question, formulate the best semantic search query and retrieve "
            "the most relevant knowledge base articles."
        ),
        backstory=str(system_tpl.template),
        tools=[kb_search_tool],
        llm=neatlogs.bind_templates(_build_llm(), system_tpl),
        allow_delegation=False,
        max_iter=12,
        max_retry_limit=3,
    )


def make_past_tickets_rag_expert() -> Agent:
    system_tpl = SystemPromptTemplate(
        "You have deep experience with support history. You search past ticket archives "
        "with precision. You always note the ticket IDs of cases you reference."
    )
    return Agent(
        role="Past Tickets RAG Expert",
        goal=(
            "Find similar past resolved tickets and extract their resolution approaches."
        ),
        backstory=str(system_tpl.template),
        tools=[past_tickets_search_tool],
        llm=neatlogs.bind_templates(_build_llm(), system_tpl),
        allow_delegation=False,
        max_iter=10,
        max_retry_limit=3,
    )


def make_response_generator() -> Agent:
    system_tpl = SystemPromptTemplate(
        "You are a senior customer success engineer. You synthesize complex information "
        "into clear, actionable replies that customers can follow immediately."
    )
    return Agent(
        role="Support Response Generator",
        goal=(
            "Write a complete, accurate, and professional email reply to the customer."
        ),
        backstory=str(system_tpl.template),
        tools=[],
        llm=neatlogs.bind_templates(_build_llm(), system_tpl),
        allow_delegation=False,
        max_iter=8,
        max_retry_limit=3,
    )
