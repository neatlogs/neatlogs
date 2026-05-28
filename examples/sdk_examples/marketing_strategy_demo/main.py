"""
Marketing Strategy Demo — CrewAI + Azure OpenAI agents with Gemini grounded search tools.

Run:
    python main.py

Required env:
    NEATLOGS_API_KEY
    AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT_NAME
    GOOGLE_API_KEY  (for the Gemini grounded search tool)
"""

import os
import sys

from dotenv import load_dotenv

load_dotenv()

# Validate required env vars up-front.
_REQUIRED_VARS = [
    "NEATLOGS_API_KEY",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "GOOGLE_API_KEY",
]
_missing = [v for v in _REQUIRED_VARS if not os.getenv(v)]
if _missing:
    sys.exit(f"Missing required env vars: {', '.join(_missing)}")

# neatlogs.init() MUST come before any CrewAI / Gemini imports.
#
# CrewAI auto-loads LiteLLM internally. When the underlying LLM is backed by a
# direct provider SDK, add that provider key too — here we use Azure OpenAI via
# `azure_ai_inference`. `google_genai` is added because tools.py uses the
# direct google.genai SDK for Google Search grounding.
import neatlogs  # noqa: E402

neatlogs.init(
    api_key=os.getenv("NEATLOGS_API_KEY"),
    endpoint=os.getenv("NEATLOGS_ENDPOINT"),
    workflow_name="marketing-strategy",
    tags=["sdk-examples", "crewai", "marketing-strategy", "demo"],
    instrumentations=["crewai", "azure_ai_inference", "google_genai"],
)

from crew import run_marketing_crew  # noqa: E402


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
    print("  MARKETING STRATEGY DEMO — NeatLogs + CrewAI")
    print("=" * 70)
    print(f"  Company : {DEMO_INPUTS['customer_domain']}")
    print(f"  Project : {DEMO_INPUTS['project_description'][:80]}...")
    print("=" * 70 + "\n")

    try:
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
