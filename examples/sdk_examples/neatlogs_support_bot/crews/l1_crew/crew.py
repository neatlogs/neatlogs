"""
L1 Crew — handles simpler support tickets.

Mirrors the original support_bot L1 crew structure:
  Agent 1: context_analyzer   — reads ticket, searches KB, extracts key context
  Agent 2: email_generator    — writes the final customer-facing reply

Span hierarchy (per ticket):
  l1_crew_kickoff      [CHAIN  — @neatlogs.span]
  ├── context_analyzer agent tasks  [AGENT/LLM — CrewAI + LiteLLM → OI]
  │   └── kb_search tool calls      [RETRIEVER + EMBEDDING — neatlogs.trace + OI]
  └── email_generator agent tasks   [AGENT/LLM — CrewAI + LiteLLM → OI]
"""

from crewai import Agent, Crew, LLM, Task

import neatlogs
from neatlogs.examples.neatlogs_support_bot.config import AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_LLM_DEPLOYMENT, OPENAI_API_VERSION
from neatlogs.examples.neatlogs_support_bot.tools import kb_search_tool, ticket_details_tool


def _make_agents():
    context_analyzer_tpl = neatlogs.PromptTemplate(
        "You are an experienced Tier-1 support analyst. You quickly identify what "
        "the customer needs, look up the correct policy or how-to from the knowledge "
        "base, and summarize the key facts for the email writer."
    )
    context_analyzer = Agent(
        role="Support Context Analyzer",
        goal=(
            "Analyze the customer support ticket, fetch relevant KB articles, "
            "and produce a structured summary of: the core issue, applicable policies, "
            "and the recommended resolution steps."
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

    email_generator_tpl = neatlogs.PromptTemplate(
        "You are a skilled customer success writer. You take technical summaries "
        "and turn them into clear, friendly, and professional email replies that "
        "leave customers feeling heard and helped."
    )
    email_generator = Agent(
        role="Customer Email Generator",
        goal=(
            "Write a concise, empathetic, and accurate customer-facing email reply "
            "that directly addresses the customer's issue using the context summary."
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

    context_task = Task(
        description=(
            f"The customer '{customer_name}' has submitted a support ticket with subject: "
            f"'{subject}'.\n\n"
            "1. Use get_ticket_details to read the full ticket.\n"
            "2. Use kb_search to find 2-3 relevant KB articles.\n"
            "3. Produce a structured summary with:\n"
            "   - Core issue (1-2 sentences)\n"
            "   - Relevant KB findings (bullet points)\n"
            "   - Recommended resolution steps (numbered list)\n"
            "   - Tone to use in reply (based on customer sentiment)"
        ),
        expected_output=(
            "A structured markdown summary with sections: "
            "Core Issue, KB Findings, Resolution Steps, Recommended Tone."
        ),
        agent=context_analyzer,
    )
    neatlogs.register_crewai_task(
        context_task,
        neatlogs.UserPromptTemplate(context_task.description + "\n" + context_task.expected_output),
    )

    email_task = Task(
        description=(
            f"Using the context summary from the previous task, write a complete "
            f"email reply to {customer_name}.\n\n"
            "Requirements:\n"
            "- Professional, warm, and empathetic tone\n"
            "- Directly address every point raised in the ticket\n"
            "- Include specific action steps the customer should take\n"
            "- End with an offer for further assistance\n"
            "- Keep it under 200 words\n"
            "- Do NOT include a subject line"
        ),
        expected_output=(
            "A complete customer-facing email reply body (no subject line, plain text)."
        ),
        agent=email_generator,
        context=[context_task],
    )
    neatlogs.register_crewai_task(
        email_task,
        neatlogs.UserPromptTemplate(email_task.description + "\n" + email_task.expected_output),
    )

    return [context_task, email_task]


@neatlogs.span(kind="CHAIN", name="l1_crew_kickoff")
def l1_crew_kickoff(ticket: dict) -> str:
    """
    Entry point for the L1 crew. Returns the final email reply text.
    The @span decorator wraps the full crew execution in a CHAIN span.
    """
    context_analyzer, email_generator = _make_agents()
    tasks = _make_tasks(context_analyzer, email_generator, ticket)

    crew = Crew(
        agents=[context_analyzer, email_generator],
        tasks=tasks,
        verbose=False,
    )

    result = crew.kickoff()
    return result.raw if hasattr(result, "raw") else str(result)
