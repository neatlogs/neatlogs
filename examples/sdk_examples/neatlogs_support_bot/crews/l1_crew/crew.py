"""
L1 Crew — handles simpler support tickets.

  Agent 1: context_analyzer — reads ticket, searches KB, extracts key context
  Agent 2: email_generator  — writes the final customer-facing reply
"""

from crewai import Agent, Crew, LLM, Task

import neatlogs
from neatlogs import SystemPromptTemplate, UserPromptTemplate

from config import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_ENDPOINT,
    OPENAI_API_VERSION,
)
from tools import kb_search_tool, ticket_details_tool


def _make_agents():
    # System templates — no placeholders (bind_templates compiles with no args).
    context_analyzer_tpl = SystemPromptTemplate(
        "You are an experienced Tier-1 support analyst. You quickly identify what "
        "the customer needs, look up the correct policy or how-to from the knowledge "
        "base, and summarize the key facts for the email writer."
    )
    context_analyzer = Agent(
        role="Support Context Analyzer",
        goal=(
            "Analyze the customer support ticket, fetch relevant KB articles, "
            "and produce a structured summary."
        ),
        backstory=str(context_analyzer_tpl.template),
        tools=[ticket_details_tool, kb_search_tool],
        llm=neatlogs.bind_templates(
            LLM(
                model="azure/gpt-4o-mini",
                base_url=AZURE_OPENAI_ENDPOINT,
                api_key=AZURE_OPENAI_API_KEY,
                api_version=OPENAI_API_VERSION,
                temperature=0.2,
                top_p=0.9,
                max_completion_tokens=800,
            ),
            context_analyzer_tpl,
        ),
        allow_delegation=False,
        max_iter=10,
        max_retry_limit=3,
    )

    email_generator_tpl = SystemPromptTemplate(
        "You are a skilled customer success writer. You take technical summaries "
        "and turn them into clear, friendly, and professional email replies."
    )
    email_generator = Agent(
        role="Customer Email Generator",
        goal=(
            "Write a concise, empathetic, and accurate customer-facing email reply."
        ),
        backstory=str(email_generator_tpl.template),
        tools=[],
        llm=neatlogs.bind_templates(
            LLM(
                model="azure/gpt-4o-mini",
                base_url=AZURE_OPENAI_ENDPOINT,
                api_key=AZURE_OPENAI_API_KEY,
                api_version=OPENAI_API_VERSION,
                temperature=0.2,
                top_p=0.9,
                max_completion_tokens=800,
            ),
            email_generator_tpl,
        ),
        allow_delegation=False,
        max_iter=5,
        max_retry_limit=3,
    )

    return context_analyzer, email_generator


def _make_tasks(context_analyzer: Agent, email_generator: Agent, ticket: dict) -> list:
    subject = ticket.get("subject", "Support request")
    customer_name = ticket.get("customer_name", "Customer")

    context_tpl = UserPromptTemplate(
        "The customer '{{customer_name}}' has submitted a support ticket with subject: "
        "'{{subject}}'.\n\n"
        "1. Use get_ticket_details to read the full ticket.\n"
        "2. Use kb_search to find 2-3 relevant KB articles.\n"
        "3. Produce a structured summary."
    )
    context_task = Task(
        description=context_tpl.compile(customer_name=customer_name, subject=subject),
        expected_output=(
            "A structured markdown summary with sections: Core Issue, KB Findings, "
            "Resolution Steps, Recommended Tone."
        ),
        agent=context_analyzer,
    )
    neatlogs.register_crewai_task(
        context_task, context_tpl,
        customer_name=customer_name, subject=subject,
    )

    email_tpl = UserPromptTemplate(
        "Using the context summary from the previous task, write a complete "
        "email reply to {{customer_name}}.\n\n"
        "Requirements: professional warm tone, directly address every point, "
        "include specific action steps, under 200 words, no subject line."
    )
    email_task = Task(
        description=email_tpl.compile(customer_name=customer_name),
        expected_output="A complete customer-facing email reply body (no subject, plain text).",
        agent=email_generator,
        context=[context_task],
    )
    neatlogs.register_crewai_task(email_task, email_tpl, customer_name=customer_name)

    return [context_task, email_task]


@neatlogs.span(kind="CHAIN", name="l1_crew_kickoff")
def l1_crew_kickoff(ticket: dict) -> str:
    context_analyzer, email_generator = _make_agents()
    tasks = _make_tasks(context_analyzer, email_generator, ticket)

    crew = Crew(
        agents=[context_analyzer, email_generator],
        tasks=tasks,
        verbose=False,
    )

    result = crew.kickoff()
    return result.raw if hasattr(result, "raw") else str(result)
