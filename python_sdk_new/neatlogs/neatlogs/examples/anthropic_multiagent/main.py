"""
Entry point for the Anthropic code review workflow.

Custom Python orchestration with @neatlogs.span decorators.

Usage:
    python main.py

Required env vars:
    NEATLOGS_API_KEY
    ANTHROPIC_API_KEY
"""

import os
import sys

# Add local SDK to path
_sdk_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if _sdk_root not in sys.path:
    sys.path.insert(0, _sdk_root)

os.environ.setdefault("NEATLOGS_LOG_SPANS", "true")
os.environ.setdefault("NEATLOGS_LOG_SPANS_FILE", "anthropic_multiagent_spans.log")
os.environ.setdefault("NEATLOGS_LOG_RAW_SPANS", "true")
os.environ.setdefault("NEATLOGS_LOG_RAW_SPANS_FILE", "anthropic_multiagent_raw_spans.log")

import neatlogs

neatlogs.init(
    api_key=os.getenv("NEATLOGS_API_KEY", ""),
    endpoint=os.getenv("NEATLOGS_ENDPOINT", "http://localhost:4100"),
    workflow_name="anthropic-code-review",
    tags=["anthropic", "code-review", "python"],
    instrumentations=["anthropic"],
    debug=True,
)

from agents import reviewer_agent, fixer_agent, tester_agent, documenter_agent

# Sample Python code with intentional issues for demonstration
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
    print(f"  Found {len(issues)} issue(s):")
    for issue in issues:
        print(f"  [{issue.get('severity', '?').upper()}] {issue.get('description', '')}")

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
