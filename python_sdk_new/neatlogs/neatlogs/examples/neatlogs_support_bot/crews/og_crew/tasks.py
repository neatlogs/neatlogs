"""
OG (L2) Crew Tasks — one task per agent, chained via context dependencies.

Task flow:
  extraction_task  →  kb_task  →  past_tickets_task  →  response_task
                                                             ↑
                                      context=[extraction, kb, past_tickets]
"""

from crewai import Task

import neatlogs
from neatlogs.examples.neatlogs_support_bot.crews.og_crew.agents import (
    make_kb_rag_expert,
    make_past_tickets_rag_expert,
    make_question_extractor,
    make_response_generator,
)


def make_tasks(ticket: dict) -> tuple[list[Task], list]:
    """
    Create all tasks and agents for the OG crew.
    Returns (tasks, agents) so the Crew can be assembled.
    """
    subject = ticket.get("subject", "Support request")
    customer_name = ticket.get("customer_name", "Customer")
    plan = ticket.get("account_plan", "unknown")

    # Build agents
    question_extractor = make_question_extractor()
    kb_rag_expert = make_kb_rag_expert()
    past_tickets_rag_expert = make_past_tickets_rag_expert()
    response_generator = make_response_generator()

    # Task 1: Extract questions
    extraction_task = Task(
        description=(
            f"A {plan}-plan customer named '{customer_name}' has submitted a ticket "
            f"with subject: '{subject}'.\n\n"
            "1. Use get_ticket_details to read the full ticket.\n"
            "2. Identify and list ALL questions and concerns the customer raises.\n"
            "3. For each item include:\n"
            "   - The question/concern (exact wording if short, paraphrase if long)\n"
            "   - Category: billing / account / technical / feature_request / other\n"
            "   - Urgency signal: high / medium / low\n"
            "4. Note any account context clues (plan type, error messages, URLs mentioned)."
        ),
        expected_output=(
            "A numbered list of customer questions with category and urgency for each. "
            "Plus a brief account context note."
        ),
        agent=question_extractor,
    )
    neatlogs.register_crewai_task(
        extraction_task,
        neatlogs.UserPromptTemplate(extraction_task.description + "\n" + extraction_task.expected_output),
    )

    # Task 2: KB retrieval
    kb_task = Task(
        description=(
            "Based on the extracted questions from the previous task:\n\n"
            "1. For each question, create an optimized semantic search query (short, "
            "   high-signal, noun-phrase style).\n"
            "2. Search the KB using kb_search for each query.\n"
            "3. Synthesize the findings: which articles are most relevant, what do they say, "
            "   and does the KB fully address each question?\n"
            "4. Note any gaps — questions not covered by existing KB articles."
        ),
        expected_output=(
            "A structured report: for each customer question, list the relevant KB articles "
            "(with IDs and key points), plus a 'Coverage Gaps' section."
        ),
        agent=kb_rag_expert,
        context=[extraction_task],
    )
    neatlogs.register_crewai_task(
        kb_task,
        neatlogs.UserPromptTemplate(kb_task.description + "\n" + kb_task.expected_output),
    )

    # Task 3: Past tickets retrieval
    past_tickets_task = Task(
        description=(
            "Based on the extracted questions from the first task:\n\n"
            "1. Search past resolved tickets using past_tickets_search for queries derived "
            "   from the main issues in this ticket.\n"
            "2. Identify the 2-3 most similar past cases.\n"
            "3. For each, extract:\n"
            "   - What was the root cause\n"
            "   - How it was resolved\n"
            "   - Any caveats or follow-up actions that were needed\n"
            "4. Note whether the current ticket pattern matches any known recurring issue."
        ),
        expected_output=(
            "A section 'Similar Past Cases' with 2-3 entries (ticket ID, summary, resolution) "
            "and a 'Pattern Match' note."
        ),
        agent=past_tickets_rag_expert,
        context=[extraction_task],
    )
    neatlogs.register_crewai_task(
        past_tickets_task,
        neatlogs.UserPromptTemplate(past_tickets_task.description + "\n" + past_tickets_task.expected_output),
    )

    # Task 4: Generate response
    response_task = Task(
        description=(
            f"Using all prior research, write a complete email reply to {customer_name}.\n\n"
            "Requirements:\n"
            "- Address EVERY question raised in the ticket\n"
            "- Reference specific KB policies or steps where applicable\n"
            "- If a past ticket resolution is directly applicable, apply it\n"
            "- Be technically precise but accessible — avoid jargon without explanation\n"
            "- Warm, professional tone; empathetic opening sentence\n"
            "- End with next steps and offer for further help\n"
            "- 200-350 words; NO subject line"
        ),
        expected_output=(
            "A complete, professional customer email reply body (plain text, no subject line)."
        ),
        agent=response_generator,
        context=[extraction_task, kb_task, past_tickets_task],
    )
    neatlogs.register_crewai_task(
        response_task,
        neatlogs.UserPromptTemplate(response_task.description + "\n" + response_task.expected_output),
    )

    tasks = [extraction_task, kb_task, past_tickets_task, response_task]
    agents = [question_extractor, kb_rag_expert, past_tickets_rag_expert, response_generator]

    return tasks, agents
