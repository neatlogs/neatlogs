"""
GobbleCube Agent — Main Entry Point
======================================
Runs GobbsGPT across all four demo scenarios and wraps each run in a
Neatlogs session trace so every query appears as a separate trace in the dashboard.

Usage:
    python main.py                  # run all 4 demo scenarios
    python main.py --query "..."    # run a custom query

Prerequisites (see README.md):
    1. Copy .env.example → .env and fill in your Azure OpenAI + Neatlogs keys
    2. pip install -r requirements.txt
"""

import argparse
import json
import os
import sys

# Allow running from repo root  (python examples/gobblecube/main.py)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import neatlogs
from neatlogs.examples.gobblecube.supervisor import build_gobbs_gpt_supervisor

# ---------------------------------------------------------------------------
# Demo scenarios (from the architecture doc)
# ---------------------------------------------------------------------------

DEMO_QUERIES = [
    {
        "title": "Revenue Diagnostic",
        "query": "Why did our revenue drop 15% in Mumbai last week?",
        "expected_agent": "Gobbs Edge (Analytics)",
        # --- Neatlogs business-use-case tags ---
        "use_case": "revenue_diagnostic",
        "agent": "gobbs_edge",
        "session_id": "demo-revenue-diagnostic",
    },
    {
        "title": "Ad Campaign Optimisation",
        "query": "Our ROAS on Blinkit dropped below 2. What should we change in our ad campaigns?",
        "expected_agent": "Gobbs Boost (Ads)",
        "use_case": "ad_optimisation",
        "agent": "gobbs_boost",
        "session_id": "demo-ad-optimisation",
    },
    {
        "title": "Stockout Emergency",
        "query": "Which SKUs are at risk of stocking out in the next 48 hours, and what should we order?",
        "expected_agent": "Gobbs Flow (Inventory)",
        "use_case": "stockout_emergency",
        "agent": "gobbs_flow",
        "session_id": "demo-stockout-emergency",
    },
    {
        "title": "Market Opportunity Discovery",
        "query": "What are the fastest growing subcategories in health snacks that we should enter?",
        "expected_agent": "Gobbs Discover (Market Intel)",
        "use_case": "market_opportunity",
        "agent": "gobbs_discover",
        "session_id": "demo-market-opportunity",
    },
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_query(supervisor, query: str, session_id: str,
              use_case: str = "general", agent: str = "auto") -> dict:
    """
    Run a single query through the GobbsGPT supervisor.

    use_case / agent — flat string attributes attached to the Neatlogs trace
                       so you can filter traces by business use case in the dashboard.
    """
    initial_state = {
        "messages": [],
        "user_query": query,
        "query_classification": None,
        "delegated_to": None,
        "sub_agent_result": None,
        "final_response": None,
        "follow_up_suggestions": None,
    }

    with neatlogs.trace(
        name="gobbs_gpt_query",
        session_id=session_id,
        use_case=use_case,
        agent=agent,
    ):
        result = supervisor.invoke(initial_state)

    return result


def print_result(scenario: dict, result: dict) -> None:
    width = 70
    print(f"\n{'=' * width}")
    print(f"  📌 {scenario.get('title', 'Custom Query')}")
    print(f"{'=' * width}")
    print(f"  ❓ Query:       {scenario['query']}")
    print(f"  🤖 Routed to:   {result.get('delegated_to', '—')}")
    print(f"\n{'-' * width}")
    print("  📋 GobbsGPT Response:\n")
    final = result.get("final_response", "No response generated.")
    for line in final.split("\n"):
        print(f"  {line}")
    print(f"\n{'-' * width}")
    follow_ups = result.get("follow_up_suggestions", [])
    if follow_ups:
        print("  💡 Suggested follow-ups:")
        for fu in follow_ups:
            print(f"     • {fu}")
    print(f"{'=' * width}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="GobbleCube AI Agent powered by LangGraph + Azure OpenAI"
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        help="Run a single custom query instead of all demo scenarios",
    )
    parser.add_argument(
        "--scenario", "-s",
        type=int,
        choices=[1, 2, 3, 4],
        help="Run a specific demo scenario (1–4)",
    )
    args = parser.parse_args()

    print("\n🚀 GobbleCube AI Agent — Powered by LangGraph + Azure OpenAI")
    print("   Observability: Neatlogs SDK\n")

    supervisor = build_gobbs_gpt_supervisor()

    if args.query:
        # Single custom query — tag it generically
        scenario = {"title": "Custom Query", "query": args.query}
        result = run_query(
            supervisor, args.query,
            session_id="custom-query",
            use_case="custom", agent="auto",
        )
        print_result(scenario, result)

    elif args.scenario:
        # Specific scenario — use its pre-defined session_id + tags
        scenario = DEMO_QUERIES[args.scenario - 1]
        result = run_query(
            supervisor, scenario["query"],
            session_id=scenario["session_id"],
            use_case=scenario["use_case"], agent=scenario["agent"],
        )
        print_result(scenario, result)

    else:
        # All 4 demo scenarios — each with its own session_id + use_case tags
        for i, scenario in enumerate(DEMO_QUERIES, start=1):
            print(f"\n⏳ Running demo scenario {i}/4: {scenario['title']}…")
            result = run_query(
                supervisor, scenario["query"],
                session_id=scenario["session_id"],
                use_case=scenario["use_case"], agent=scenario["agent"],
            )
            print_result(scenario, result)

    neatlogs.flush()
    neatlogs.shutdown()
    print("\n✅ All traces flushed to Neatlogs. Done!\n")


if __name__ == "__main__":
    main()
