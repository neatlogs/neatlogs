"""
Entry point for the OpenAI investment research workflow.

Custom Python orchestration — no framework.
@neatlogs.span decorators create the WORKFLOW + AGENT span hierarchy.

Usage:
    python main.py
    python main.py "Tesla"

Required env vars:
    NEATLOGS_API_KEY
    AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_LLM_DEPLOYMENT
"""

import os
import sys

from dotenv import load_dotenv

load_dotenv()

# neatlogs.init() MUST come before any LLM library imports.
import neatlogs

neatlogs.init(
    api_key=os.getenv("NEATLOGS_API_KEY"),
    endpoint=os.getenv("NEATLOGS_ENDPOINT"),
    workflow_name="investment-research",
    tags=["sdk-examples", "openai", "investment", "research", "multi-agent"],
    instrumentations=["openai"],
    capture_logs=True,
)

from agents import planner_agent, researcher_agent, analyst_agent, reporter_agent


@neatlogs.span(kind="WORKFLOW", name="investment_research_workflow")
def run_investment_research(company: str) -> str:
    print(f"\n=== Investment Research: {company} ===\n")

    print("--- Planner: generating research questions ---")
    questions = planner_agent(company)
    for i, q in enumerate(questions, 1):
        print(f"  {i}. {q}")

    print("\n--- Researcher: gathering findings ---")
    findings = researcher_agent(questions)

    print("\n--- Analyst: analyzing findings ---")
    analysis = analyst_agent(company, findings)

    print("\n--- Reporter: writing investment brief ---")
    report = reporter_agent(company, analysis)

    return report


if __name__ == "__main__":
    company = sys.argv[1] if len(sys.argv) > 1 else "NVIDIA"
    run_investment_research(company)
    neatlogs.flush()
    neatlogs.shutdown()
