"""
Entry point for the Anthropic code review workflow.

Custom Python orchestration with @neatlogs.span decorators.

Usage:
    python main.py

Required env vars:
    NEATLOGS_API_KEY
    AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION  (Bedrock)
"""

import os
import sys

from dotenv import load_dotenv

load_dotenv()

# neatlogs.init() MUST come before any LLM library imports so that
# auto-instrumentation can patch the modules at import time.
import neatlogs

neatlogs.init(
    api_key=os.getenv("NEATLOGS_API_KEY"),
    endpoint=os.getenv("NEATLOGS_ENDPOINT"),
    workflow_name="code-review",
    tags=["sdk-examples", "anthropic", "multi-agent", "code-review"],
    instrumentations=["anthropic"],
)

from agents import reviewer_agent, fixer_agent, tester_agent, documenter_agent

SAMPLE_CODE = '''
def calculate_average(numbers):
    total = 0
    for n in numbers:
        total = total + n
    avg = total / len(numbers)
    return avg

def find_duplicates(lst):
    duplicates = []
    for i in range(len(lst)):
        for j in range(len(lst)):
            if i != j and lst[i] == lst[j]:
                if lst[i] not in duplicates:
                    duplicates.append(lst[i])
    return duplicates

def parse_config(config_str):
    parts = config_str.split("=")
    key = parts[0]
    value = parts[1]
    return {key: value}
'''


@neatlogs.span(kind="WORKFLOW", name="code_review_workflow")
def run_code_review(code: str) -> dict:
    print("\n=== Code Review Pipeline ===\n")

    print("--- Reviewer: identifying issues ---")
    issues = reviewer_agent(code)
    print(f"  Found {len(issues)} issue(s)")

    print("\n--- Fixer: applying fixes ---")
    fixed_code = fixer_agent(code, issues)

    print("\n--- Tester: writing tests ---")
    tests = tester_agent(fixed_code)

    print("\n--- Documenter: adding documentation ---")
    documented_code = documenter_agent(fixed_code)
    print("\n--- Documented Code ---")
    print(documented_code)

    return {
        "issues": issues,
        "fixed_code": fixed_code,
        "tests": tests,
        "documented_code": documented_code,
    }


if __name__ == "__main__":
    run_code_review(SAMPLE_CODE)
    neatlogs.flush()
    neatlogs.shutdown()
