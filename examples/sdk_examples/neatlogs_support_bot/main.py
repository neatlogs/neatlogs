"""
Neatlogs Support Bot — entry point.

Runs sample tickets through the crew selector:
  - L1 tickets (billing, password reset) → 2-agent CrewAI crew
  - OG tickets (technical) → 4-agent CrewAI crew

Each ticket goes through:
  1. _classify_ticket()  → lightweight routing call (LLM span)
  2. l1_crew_kickoff()   OR og_crew_kickoff()  (CHAIN span)

Usage:
    python main.py
"""

import config  # noqa: F401 — triggers neatlogs.init() (must be first import)
import os

import neatlogs

from crew_selector import get_ticket_information


SAMPLE_TICKETS = [
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
    neatlogs.flush()
    neatlogs.shutdown()


if __name__ == "__main__":
    main()
