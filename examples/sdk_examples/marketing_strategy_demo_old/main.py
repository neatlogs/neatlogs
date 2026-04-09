"""
Marketing Strategy Demo -- Entry point.

Run:  python main.py

Set MOCK_MODE = True to skip all API calls and return the real cached result
from trace d021f6e44c40b01ee0d0687678594a0a instantly.
Set MOCK_MODE = False to run the actual CrewAI + Gemini workflow.

Neatlogs SDK is initialised HERE, before any CrewAI / Gemini imports,
so that auto-instrumentation hooks are registered first.
"""

import os
import sys
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Toggle: set True for instant cached result, False to run real API calls
# ---------------------------------------------------------------------------
MOCK_MODE = False

# Real output from trace d021f6e44c40b01ee0d0687678594a0a (2026-04-02)
# Company: crewai.com | ~316s run | gpt-5-nano + gemini-2.5-flash
_MOCK_RESULT = (
    "title='FlowForge: Enterprise AI Crews, Orchestrated at Scale' "
    "body=\"In large organizations, automation isn\u2019t single-task\u2014it\u2019s a "
    "coordinated crew of intelligent agents. FlowForge merges Studio\u2019s no-code/low-code "
    "crew orchestration with AMP\u2019s production-grade governance and AMP Factory\u2019s "
    "on-prem/hybrid deployment to deliver scalable, auditable multi-agent workflows across "
    "finance, IT, and operations. Say goodbye to fragmentation: centralize control, "
    "observability, memory across steps, and real-time tracing with a single platform. "
    "FlowForge is built for CTOs, VPs of Engineering, and AI leads who demand data residency, "
    "RBAC, SOC 2 controls, and seamless integrations with Salesforce, Slack, Gmail, Teams, "
    "and more. Start small with Studio, scale to production with AMP, and deploy where you "
    "need\u2014cloud, on-prem, or hybrid. See faster time-to-value, reduced risk, and "
    "measurable ROI as you deploy millions of coordinated tasks across your organization. "
    "Ready to see it in action? Book a personalized executive demo today.\""
)

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
    instrumentations=[
        "openai",
        "crewai",
        "langchain",
        "azure_ai_inference",
        "google_genai",
        # google_genai intentionally excluded: Gemini is used only as a search
        # tool inside search_web/analyze_website. Those calls are already captured
        # as TOOL spans by crewai OI. Including google_genai would also log them
        # as LLM spans, which is misleading.
    ],
    workflow_name="Marketing Strategy Demo",
    tags=["demo", "crewai", "marketing-strategy"],
    pii_enabled=True,
    debug=True,
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
    if MOCK_MODE:
        print("  MODE: MOCK (cached result from trace d021f6e44c40b01ee0d0687678594a0a)")
    print("=" * 70)
    print(f"  Company : {DEMO_INPUTS['customer_domain']}")
    print(f"  Project : {DEMO_INPUTS['project_description'][:80]}...")
    print("=" * 70 + "\n")

    try:
        if MOCK_MODE:
            result = _MOCK_RESULT
        else:
            result = run_marketing_crew(DEMO_INPUTS)

        print("\n" + "=" * 70)
        print("  FINAL RESULT")
        print("=" * 70)
        print(result)
    finally:
        neatlogs.flush()
        neatlogs.shutdown()


if __name__ == "__main__":
    main()
