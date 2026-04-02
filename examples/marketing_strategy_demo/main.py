"""
Marketing Strategy Demo -- Entry point.

Run:  python main.py

Neatlogs SDK is initialised HERE, before any CrewAI / Gemini imports,
so that auto-instrumentation hooks are registered first.
"""

import os
import sys
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Validate required environment variables upfront
# ---------------------------------------------------------------------------
_REQUIRED_VARS = [
    "NEATLOGS_API_KEY",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "GEMINI_API_KEY",
]
_missing = [v for v in _REQUIRED_VARS if not os.getenv(v)]
if _missing:
    print(f"ERROR: Missing required environment variables: {', '.join(_missing)}")
    print("       Copy .env.example to .env and fill in your API keys.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Neatlogs init -- MUST come before any LLM / CrewAI imports
# ---------------------------------------------------------------------------
import neatlogs  # noqa: E402

neatlogs.init(
    api_key=os.getenv("NEATLOGS_API_KEY"),
    endpoint=os.getenv(
        "NEATLOGS_ENDPOINT",
        "https://staging-cloud.neatlogs.com/api/data/v4/batch",
    ),
    instrumentations=["openai", "crewai", "google_genai"],
    workflow_name="marketing-strategy-demo",
    tags=["demo", "crewai", "marketing-strategy"],
    debug=os.getenv("NEATLOGS_DEBUG", "false").lower() == "true",
)

# -- Now safe to import the rest ------------------------------------------------
from crew import run_marketing_crew  # noqa: E402


# ---------------------------------------------------------------------------
# Demo inputs -- customise these for each customer demo
# ---------------------------------------------------------------------------
DEMO_INPUTS = {
    "customer_domain": "crewai.com",
    "project_description": (
        "CrewAI, a leading provider of multi-agent AI systems, wants to "
        "boost adoption of its platform among enterprise engineering teams. "
        "The campaign should highlight ease of use, production-readiness, "
        "and the ability to orchestrate complex AI workflows. Target audience: "
        "CTOs, VP Engineering, and senior developers at mid-to-large companies."
    ),
}


def main():
    print("\n" + "=" * 70)
    print("  MARKETING STRATEGY DEMO  --  Neatlogs + CrewAI")
    print("=" * 70)
    print(f"  Company : {DEMO_INPUTS['customer_domain']}")
    print(f"  Project : {DEMO_INPUTS['project_description'][:80]}...")
    print("=" * 70 + "\n")

    try:
        with neatlogs.trace(
            name="marketing_strategy_workflow",
            kind="WORKFLOW",
        ):
            result = run_marketing_crew(DEMO_INPUTS)

        print("\n" + "=" * 70)
        print("  FINAL RESULT")
        print("=" * 70)
        print(result)
    finally:
        # Always flush spans -- even on error the partial trace is valuable
        neatlogs.flush()
        neatlogs.shutdown()


if __name__ == "__main__":
    main()
