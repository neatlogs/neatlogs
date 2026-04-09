"""
Crew selector — routes each ticket to the L1 or OG (L2) crew.

Mirrors the original support_bot crews/crew_selector.py logic.

Routing rules:
  L1 (simple):  billing, account management, password reset, plan changes,
                feature questions with clear KB answers
  OG (complex): technical debugging, API/webhook/SDK issues, data export problems,
                SSO configuration, multi-issue tickets

The selector uses a lightweight OpenAI classification call — auto-instrumented
by OpenInference → LLM span.
"""

import json

from openai import AzureOpenAI

import neatlogs
from neatlogs.examples.neatlogs_support_bot.config import AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, OPENAI_API_VERSION, AZURE_LLM_DEPLOYMENT
from neatlogs.examples.neatlogs_support_bot.tools import set_ticket_context

_client = AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version=OPENAI_API_VERSION,
)

_ROUTE_SYSTEM = """\
You are a support ticket routing classifier.
Given a support ticket subject and body, decide which crew should handle it.

Reply with JSON only:
{
  "crew": "<l1 | og>",
  "reason": "<one sentence>"
}

Route to "l1" if the issue is:
  - Billing, refunds, plan upgrades/downgrades
  - Password reset or simple account access
  - General feature questions with clear answers
  - Short single-question tickets

Route to "og" if the issue is:
  - Technical debugging (API errors, webhooks, SDK integration)
  - SSO/SAML configuration
  - Data export or large-scale account operations
  - Multi-issue tickets with unclear root cause
  - Any issue that likely needs code-level investigation"""


@neatlogs.span(kind="CHAIN", name="process_ticket")
def get_ticket_information(ticket: dict) -> dict:
    """
    Route a ticket to the correct crew and run it.

    Returns a dict with:
      crew:     "l1" or "og"
      response: final email reply from the crew
      reason:   why this crew was selected
    """
    # Register ticket context so tools can access it during crew execution
    set_ticket_context(ticket)

    # Lightweight routing call — produces an LLM span
    routing = _classify_ticket(ticket)
    crew_name = routing.get("crew", "og")
    reason = routing.get("reason", "")

    if crew_name == "l1":
        from neatlogs.examples.neatlogs_support_bot.crews.l1_crew.crew import l1_crew_kickoff
        response = l1_crew_kickoff(ticket)
    else:
        from neatlogs.examples.neatlogs_support_bot.crews.og_crew.crew import og_crew_kickoff
        response = og_crew_kickoff(ticket)

    return {
        "crew": crew_name,
        "reason": reason,
        "response": response,
    }


def _classify_ticket(ticket: dict) -> dict:
    subject = ticket.get("subject", "")
    body = ticket.get("body", "")
    resp = _client.chat.completions.create(
        model=AZURE_LLM_DEPLOYMENT,
        messages=[
            {"role": "system", "content": _ROUTE_SYSTEM},
            {"role": "user", "content": f"Subject: {subject}\n\nBody:\n{body}"},
        ],
        response_format={"type": "json_object"},
    )
    raw = resp.choices[0].message.content or "{}"
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"crew": "og", "reason": "parse error — defaulting to OG crew"}
