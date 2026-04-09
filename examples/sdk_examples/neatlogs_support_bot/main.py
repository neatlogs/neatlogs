"""
Neatlogs Support Bot — main entry point.

Runs 4 sample tickets through the crew selector:
  - 2 L1 tickets (billing, password reset)
  - 2 OG tickets (API webhook debugging, SSO configuration)

Each ticket goes through:
  1. _classify_ticket()  → lightweight routing call (LLM span)
  2. l1_crew_kickoff()   → 2-agent CrewAI crew (CHAIN span)
     OR og_crew_kickoff() → 4-agent CrewAI crew (CHAIN span)

Spans are written to NEATLOGS_LOG_SPANS_FILE (default: support_bot_spans.log).
Set NEATLOGS_LOG_SPANS=true and optionally NEATLOGS_LOG_SPANS_FILE to capture them.

Usage (from python_sdk_new/neatlogs/):
  python neatlogs/examples/neatlogs_support_bot/main.py
"""

import os
import sys

# Add repo root (python_sdk_new/neatlogs/) to sys.path so that
# `import neatlogs` and `neatlogs.examples.*` resolve correctly.
# main.py is 4 levels deep: neatlogs/examples/neatlogs_support_bot/main.py
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import neatlogs.examples.neatlogs_support_bot.config  # noqa: F401 — triggers neatlogs.init() (must be first)
import neatlogs

from neatlogs.examples.neatlogs_support_bot.crew_selector import get_ticket_information

SAMPLE_TICKETS = [
    # --- L1: billing question ---
    {
        "ticket_id": "TKT-1001",
        "customer_name": "Alice Chen",
        "account_plan": "pro",
        "subject": "Charge on my credit card I don't recognize",
        "body": (
            "Hi, I noticed a charge of $79 on my credit card last Tuesday from your company. "
            "I'm on the Pro plan which should be $49/month. Can you explain the extra $30 charge? "
            "If it was a mistake please refund it."
        ),
    },
    # --- L1: password reset ---
    # {
    #     "ticket_id": "TKT-1002",
    #     "customer_name": "Bob Patel",
    #     "account_plan": "starter",
    #     "subject": "Can't log in — forgot password",
    #     "body": (
    #         "I can't log into my account. I tried the 'Forgot Password' link but I'm not "
    #         "receiving the reset email. I've checked spam. My email is bob@example.com. "
    #         "Please help me regain access."
    #     ),
    # },
    # # --- OG: API / webhook debugging ---
    # {
    #     "ticket_id": "TKT-2001",
    #     "customer_name": "Carol Nguyen",
    #     "account_plan": "enterprise",
    #     "subject": "Webhook payloads missing 'metadata' field since last Wednesday",
    #     "body": (
    #         "Since last Wednesday our webhook receiver has been getting payloads without the "
    #         "'metadata' field that used to be present. Our pipeline depends on this field to "
    #         "route events to the correct downstream service. I've verified on our end — the "
    #         "field is simply absent in the JSON. API version we're using: v2.4. "
    #         "We're hitting POST /events/subscribe. Is this a breaking change or a regression? "
    #         "What's the ETA for a fix? We're currently blocking a production deployment."
    #     ),
    # },
    # # --- OG: SSO configuration ---
    # {
    #     "ticket_id": "TKT-2002",
    #     "customer_name": "David Kim",
    #     "account_plan": "enterprise",
    #     "subject": "SAML SSO login loop — users redirected back to IdP indefinitely",
    #     "body": (
    #         "We configured SAML SSO using Okta as our IdP. When users try to log in they get "
    #         "redirected to Okta, authenticate successfully, then get sent back to our login page "
    #         "instead of being logged in. This creates an infinite loop. "
    #         "We've verified the ACS URL and Entity ID match exactly. "
    #         "The Okta app assignment is correct. SAML response looks valid in the browser "
    #         "network tab — we see a 200 on the callback but then get redirected back. "
    #         "Could there be a session cookie / SameSite issue? Or a misconfiguration on your end?"
    #     ),
    # },
]


def run_ticket(ticket: dict) -> None:
    print(f"\n{'=' * 70}")
    print(f"Ticket: [{ticket['ticket_id']}] {ticket['subject']}")
    print(f"Customer: {ticket['customer_name']} ({ticket['account_plan']} plan)")
    print("Processing...")

    result = get_ticket_information(ticket)

    print(f"\nRouted to:  {result['crew'].upper()} crew")
    print(f"Reason:     {result['reason']}")
    print(f"\n--- Response ---\n{result['response']}")


def main() -> None:
    print("Neatlogs Support Bot")
    print(f"Running {len(SAMPLE_TICKETS)} sample tickets\n")

    for ticket in SAMPLE_TICKETS:
        run_ticket(ticket)

    print(f"\n{'=' * 70}")
    print("All tickets processed.")
    print("Span log: check NEATLOGS_LOG_SPANS_FILE (default /tmp/neatlogs_spans.jsonl)")
    neatlogs.flush()
    neatlogs.shutdown()


if __name__ == "__main__":
    main()
