"""
Stress Test 2: Cross-File Multi-Agent Custom Orchestration
==========================================================

Tests whether `with trace()` blocks defined in DIFFERENT FILES all share
the same trace_id when called from a main orchestrator.

Scenario:
- Main workflow imports agent modules from separate files
- Each agent file has its own @agent decorator + with trace() logic
- Verify: ALL spans from all files have SAME trace_id

File structure:
  51_cross_file_multi_agent_stress_test.py  (main orchestrator)
  51_agents/planner.py                      (agent 1)
  51_agents/executor.py                     (agent 2)
  51_agents/reviewer.py                     (agent 3)

Run:
  python neatlogs/examples/51_cross_file_multi_agent_stress_test.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from neatlogs import flush, init, shutdown, trace, workflow


def _env_default(k: str, v: str) -> None:
    os.environ.setdefault(k, v)


_env_default("LANGCHAIN_TRACING_V2", "false")
_env_default("NEATLOGS_LOG_SPANS", "true")
_env_default("NEATLOGS_LOG_METRICS", "true")
_env_default("NEATLOGS_LOG_RAW_SPANS", "true")
_env_default("NEATLOGS_LOG_SPANS_FILE", "spans_51_cross_file_stress.jsonl")
_env_default("NEATLOGS_LOG_RAW_SPANS_FILE", "spans_raw_51_cross_file_stress.jsonl")
_env_default("NEATLOGS_LOG_METRICS_FILE", "metrics_51_cross_file_stress.jsonl")


def _require_env(key: str) -> str:
    val = os.getenv(key)
    if not val:
        raise SystemExit(f"Missing required env var: {key}")
    return val


# Initialize SDK
init(
    api_key=_require_env("NEATLOGS_API_KEY"),
    endpoint="http://localhost:3000/api/data/v4/batch",
    workflow_name="cross-file-multi-agent-stress",
    instrumentations=["openai"],
    debug=True,
)

# Import agents AFTER init
from neatlogs.examples.agents_51 import executor, planner, reviewer


# ============================================================================
# Main Workflow (defined in main file)
# ============================================================================


@workflow(name="cross_file_orchestration")
def run_cross_file_workflow(task: str) -> str:
    """
    Orchestrate agents defined in separate files.
    
    CRITICAL: All agent calls should inherit the trace context from this
    @workflow decorator, resulting in a single trace_id across all files.
    """
    print(f"\n{'='*60}")
    print(f"🚀 CROSS-FILE WORKFLOW: {task}")
    print(f"{'='*60}")

    # Agent 1: Planner (from 51_agents/planner.py)
    plan = planner.plan_task(task)

    # Agent 2: Executor (from 51_agents/executor.py)
    execution_results = executor.execute_plan(plan)

    # Agent 3: Reviewer (from 51_agents/reviewer.py)
    review = reviewer.review_execution(execution_results)

    print(f"\n{'='*60}")
    print(f"✅ WORKFLOW COMPLETE")
    print(f"{'='*60}")

    return review


def main():
    try:
        # Outer wrapper to ensure all cross-file agents share one trace
        with trace("cross_file_session", kind="WORKFLOW"):
            task = "Design a multi-agent system for customer support automation"
            final_output = run_cross_file_workflow(task)

            print(f"\n📝 Final Review:\n{final_output}\n")

    except KeyboardInterrupt:
        raise SystemExit(130)
    finally:
        flush()
        shutdown()
        print("\n✅ Check output files:")
        print(f"   - spans_51_cross_file_stress.jsonl")
        print(f"   - spans_raw_51_cross_file_stress.jsonl")
        print(f"   - metrics_51_cross_file_stress.jsonl")
        print("\nVerify: All spans (from main + 3 agent files) should have SAME trace_id")
        print("Run: jq '.trace_id' spans_51_cross_file_stress.jsonl | sort | uniq -c")


if __name__ == "__main__":
    main()
