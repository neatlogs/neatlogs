"""
Entry point for the OpenAI investment research workflow.

Custom Python orchestration — no framework.
@neatlogs.span decorators create the WORKFLOW + AGENT span hierarchy.

Usage:
    python main.py
    python main.py "Tesla"

Required env vars:
    NEATLOGS_API_KEY
    AZURE_OPENAI_ENDPOINT
    AZURE_OPENAI_API_KEY
    AZURE_LLM_DEPLOYMENT
"""

import os
import sys

# Add local SDK to path
_sdk_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if _sdk_root not in sys.path:
    sys.path.insert(0, _sdk_root)

os.environ.setdefault("NEATLOGS_LOG_SPANS", "true")
os.environ.setdefault("NEATLOGS_LOG_SPANS_FILE", "openai_multiagent_spans.log")
os.environ.setdefault("NEATLOGS_LOG_RAW_SPANS", "true")
os.environ.setdefault("NEATLOGS_LOG_RAW_SPANS_FILE", "openai_multiagent_raw_spans.log")

import neatlogs

neatlogs.init(
    api_key=os.getenv("NEATLOGS_API_KEY", ""),
    endpoint=os.getenv("NEATLOGS_ENDPOINT", "http://localhost:4100"),
    workflow_name="openai-investment-research",
    tags=["openai", "investment", "research"],
    instrumentations=["openai"],
    capture_logs=True,
    debug=True,
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
