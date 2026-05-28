"""
OG (L2) Crew Tasks — one task per agent, chained via context dependencies.

  extraction_task → kb_task ── ┐
                    past_tickets_task ──┤
                                        └→ response_task (context=[all of above])
"""

from crewai import Task

import neatlogs
from neatlogs import UserPromptTemplate

from crews.og_crew.agents import (
    make_kb_rag_expert,
    make_past_tickets_rag_expert,
    make_question_extractor,
    make_response_generator,
)


def make_tasks(ticket: dict) -> tuple[list[Task], list]:
    subject = ticket.get("subject", "Support request")
    customer_name = ticket.get("customer_name", "Customer")
    plan = ticket.get("account_plan", "unknown")

    question_extractor = make_question_extractor()
    kb_rag_expert = make_kb_rag_expert()
    past_tickets_rag_expert = make_past_tickets_rag_expert()
    response_generator = make_response_generator()

    extraction_tpl = UserPromptTemplate(
        "A {{plan}}-plan customer named '{{customer_name}}' has submitted a ticket "
        "with subject: '{{subject}}'.\n\n"
        "1. Use get_ticket_details to read the full ticket.\n"
        "2. Identify and list ALL questions and concerns.\n"
        "3. For each: question, category, urgency.\n"
        "4. Note any account context clues."
    )
    extraction_task = Task(
        description=extraction_tpl.compile(plan=plan, customer_name=customer_name, subject=subject),
        expected_output=(
            "A numbered list of customer questions with category and urgency for each."
        ),
        agent=question_extractor,
    )
    neatlogs.register_crewai_task(
        extraction_task, extraction_tpl,
        plan=plan, customer_name=customer_name, subject=subject,
    )

    kb_tpl = UserPromptTemplate(
        "Based on the extracted questions:\n\n"
        "1. Create an optimized semantic search query for each.\n"
        "2. Use kb_search to retrieve articles.\n"
        "3. Synthesize findings and note coverage gaps."
    )
    kb_task = Task(
        description=kb_tpl.compile(),
        expected_output=(
            "A structured report with relevant KB articles per question, plus gaps section."
        ),
        agent=kb_rag_expert,
        context=[extraction_task],
    )
    neatlogs.register_crewai_task(kb_task, kb_tpl)

    past_tpl = UserPromptTemplate(
        "Based on the extracted questions:\n\n"
        "1. Search past resolved tickets using past_tickets_search.\n"
        "2. Identify the 2-3 most similar past cases.\n"
        "3. For each, extract root cause, resolution, caveats."
    )
    past_tickets_task = Task(
        description=past_tpl.compile(),
        expected_output=(
            "'Similar Past Cases' section with 2-3 entries, plus a 'Pattern Match' note."
        ),
        agent=past_tickets_rag_expert,
        context=[extraction_task],
    )
    neatlogs.register_crewai_task(past_tickets_task, past_tpl)

    response_tpl = UserPromptTemplate(
        "Using all prior research, write a complete email reply to {{customer_name}}.\n\n"
        "Address every question, reference KB policies, apply past resolutions, "
        "200-350 words, warm professional tone, no subject line."
    )
    response_task = Task(
        description=response_tpl.compile(customer_name=customer_name),
        expected_output=(
            "A complete, professional customer email reply body (plain text, no subject)."
        ),
        agent=response_generator,
        context=[extraction_task, kb_task, past_tickets_task],
    )
    neatlogs.register_crewai_task(response_task, response_tpl, customer_name=customer_name)

    tasks = [extraction_task, kb_task, past_tickets_task, response_task]
    agents = [question_extractor, kb_rag_expert, past_tickets_rag_expert, response_generator]

    return tasks, agents
