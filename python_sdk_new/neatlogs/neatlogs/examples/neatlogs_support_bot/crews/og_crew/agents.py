"""
OG (L2) Crew Agents — mirrors the original support_bot og_crew/agents.py structure.

Four agents, each with a distinct specialization:
  1. question_extractor      — parses the ticket and extracts precise questions
  2. kb_rag_expert           — retrieves relevant KB articles via semantic search
  3. past_tickets_rag_expert — finds similar resolved tickets and their patterns
  4. response_generator      — synthesizes all context into a professional email reply

Each agent's LLM call goes through LiteLLM → OpenAI, instrumented by OI LiteLLM
instrumentor → LLM span with full inputs/outputs/token counts.
"""

from crewai import Agent, LLM

import neatlogs
from neatlogs.examples.neatlogs_support_bot.config import AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_LLM_DEPLOYMENT, OPENAI_API_VERSION
from neatlogs.examples.neatlogs_support_bot.tools import kb_search_tool, past_tickets_search_tool, ticket_details_tool


def make_question_extractor() -> Agent:
    """
    Analyzes ticket content (subject, body, attachments described) and extracts
    the precise technical questions the customer is asking.

    Mirrors original: QueryAgents.question_extractor
    """
    system_tpl = neatlogs.PromptTemplate(
        "You are a meticulous support triage specialist. You read customer tickets "
        "carefully to identify all questions, implied concerns, and the account context. "
        "You never assume — if something is unclear you note it as 'requires clarification'."
    )
    bound_llm = neatlogs.bind_templates(LLM(model=f"azure/{AZURE_LLM_DEPLOYMENT}", base_url=AZURE_OPENAI_ENDPOINT, api_key=AZURE_OPENAI_API_KEY, api_version=OPENAI_API_VERSION), system_tpl)
    return Agent(
        role="Ticket Question Extractor",
        goal=(
            "Read the full support ticket and extract every distinct question or issue "
            "the customer is raising. Format them as a numbered list with context for each."
        ),
        backstory=str(system_tpl.template),
        tools=[ticket_details_tool],
        llm=bound_llm,
        allow_delegation=False,
        max_iter=8,
        max_retry_limit=3,
    )


def make_kb_rag_expert() -> Agent:
    """
    Transforms extracted questions into optimized search queries and retrieves
    relevant KB articles.

    Mirrors original: QueryAgents.kb_rag_expert
    """
    system_tpl = neatlogs.PromptTemplate(
        "You are an expert in retrieval-augmented generation. You know how to rephrase "
        "questions into short, high-signal search queries that maximize semantic similarity. "
        "You always cite the KB article IDs you used."
    )
    bound_llm = neatlogs.bind_templates(LLM(model=f"azure/{AZURE_LLM_DEPLOYMENT}", base_url=AZURE_OPENAI_ENDPOINT, api_key=AZURE_OPENAI_API_KEY, api_version=OPENAI_API_VERSION), system_tpl)
    return Agent(
        role="KB RAG Expert",
        goal=(
            "For each question extracted from the ticket, formulate the best semantic search "
            "query and retrieve the most relevant knowledge base articles. "
            "Return a synthesis of the KB findings relevant to this ticket."
        ),
        backstory=str(system_tpl.template),
        tools=[kb_search_tool],
        llm=bound_llm,
        allow_delegation=False,
        max_iter=12,
        max_retry_limit=3,
    )


def make_past_tickets_rag_expert() -> Agent:
    """
    Searches past resolved tickets for patterns and proven resolutions.

    Mirrors original: QueryAgents.past_tickets_rag_expert
    """
    system_tpl = neatlogs.PromptTemplate(
        "You have deep experience with support history. You search past ticket archives "
        "with precision, identifying which prior resolutions are truly analogous to the "
        "current case. You always note the ticket IDs of cases you reference."
    )
    bound_llm = neatlogs.bind_templates(LLM(model=f"azure/{AZURE_LLM_DEPLOYMENT}", base_url=AZURE_OPENAI_ENDPOINT, api_key=AZURE_OPENAI_API_KEY, api_version=OPENAI_API_VERSION), system_tpl)
    return Agent(
        role="Past Tickets RAG Expert",
        goal=(
            "Search the repository of past resolved tickets to find cases similar to the "
            "current one. Extract the resolution approaches that worked and any known pitfalls "
            "to avoid."
        ),
        backstory=str(system_tpl.template),
        tools=[past_tickets_search_tool],
        llm=bound_llm,
        allow_delegation=False,
        max_iter=10,
        max_retry_limit=3,
    )


def make_response_generator() -> Agent:
    """
    Synthesizes all gathered context into a polished, accurate email response.

    Mirrors original: QueryAgents + email generation task
    """
    system_tpl = neatlogs.PromptTemplate(
        "You are a senior customer success engineer who combines technical depth with "
        "excellent written communication. You synthesize complex information into clear, "
        "actionable replies that customers can follow immediately."
    )
    bound_llm = neatlogs.bind_templates(LLM(model=f"azure/{AZURE_LLM_DEPLOYMENT}", base_url=AZURE_OPENAI_ENDPOINT, api_key=AZURE_OPENAI_API_KEY, api_version=OPENAI_API_VERSION), system_tpl)
    return Agent(
        role="Support Response Generator",
        goal=(
            "Write a complete, accurate, and professional email reply to the customer "
            "that addresses every question raised, draws on KB articles and past ticket "
            "patterns, and provides clear action steps."
        ),
        backstory=str(system_tpl.template),
        tools=[],
        llm=bound_llm,
        allow_delegation=False,
        max_iter=8,
        max_retry_limit=3,
    )
