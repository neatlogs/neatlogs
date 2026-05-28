"""
Detection Demo - Multi-Framework Workflows
===========================================
Demonstrates NeatLogs detections across LangGraph, CrewAI, LangChain, and Gemini.

Usage:
    python main.py                    # Run all workflows
    python main.py --workflow 1       # Customer Support (LangGraph)
    python main.py --workflow 2       # Content Moderation (CrewAI)
    python main.py --workflow 3       # Research Assistant (LangChain)
    python main.py --workflow 4       # Sales Lead Qualification (LangGraph)
    python main.py --workflow 5       # Gemini async streaming

Prerequisites:
    cp .env.example .env
    pip install -r requirements.txt
"""

import argparse
import sys

import neatlogs

from config import init_neatlogs, load_settings


def print_banner() -> None:
    print(
        """
╔══════════════════════════════════════════════════════════════════════════════╗
║                  DETECTION DEMO - MULTI-FRAMEWORK WORKFLOWS                   ║
║  Workflows: Customer Support, Content Moderation, Research, Sales, Gemini    ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
    )


def print_summary(workflows_run: list[str]) -> None:
    print("\n" + "=" * 80)
    print("DETECTION DEMO - SUMMARY")
    print("=" * 80)
    print(f"\nCompleted {len(workflows_run)} workflow(s):\n")
    for workflow in workflows_run:
        print(f"  • {workflow}")
    print("\nTraces were sent to NeatLogs. Open your dashboard to inspect detections.")
    print("=" * 80 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Detection Demo - Multi-Framework Workflows")
    parser.add_argument(
        "--workflow",
        "-w",
        type=int,
        choices=[1, 2, 3, 4, 5],
        help="Run one workflow (1=Customer Support, 2=Content Moderation, 3=Research, 4=Sales, 5=Gemini)",
    )
    args = parser.parse_args()

    print_banner()

    try:
        settings = load_settings()
    except RuntimeError as e:
        print(f"\nConfiguration error:\n{e}\n")
        sys.exit(1)

    try:
        init_neatlogs(settings)
    except Exception as e:
        print(f"\nNeatLogs initialization error:\n{e}\n")
        sys.exit(1)

    workflows: list[tuple[str, object]] = []

    if args.workflow is None:
        from workflows.workflow_customer_support import run_customer_support_workflow
        from workflows.workflow_research_assistant import run_research_assistant_workflow
        from workflows.workflow_sales_qualification import run_sales_qualification_workflow

        workflows = [
            ("Customer Support (LangGraph)", run_customer_support_workflow),
            ("Research Assistant (LangChain)", run_research_assistant_workflow),
            ("Sales Lead Qualification (LangGraph)", run_sales_qualification_workflow),
        ]
        try:
            from workflows.workflow_content_moderation import run_content_moderation_workflow

            workflows.insert(1, ("Content Moderation (CrewAI)", run_content_moderation_workflow))
        except ImportError:
            print("Skipping Content Moderation (CrewAI not installed)\n")
        try:
            from workflows.workflow_gemini_async_streaming import run_gemini_async_streaming_workflow

            workflows.append(("Gemini Async Streaming", run_gemini_async_streaming_workflow))
        except ImportError:
            print("Skipping Gemini Async Streaming (google-genai not installed)\n")
    elif args.workflow == 1:
        from workflows.workflow_customer_support import run_customer_support_workflow

        workflows = [("Customer Support (LangGraph)", run_customer_support_workflow)]
    elif args.workflow == 2:
        from workflows.workflow_content_moderation import run_content_moderation_workflow

        workflows = [("Content Moderation (CrewAI)", run_content_moderation_workflow)]
    elif args.workflow == 3:
        from workflows.workflow_research_assistant import run_research_assistant_workflow

        workflows = [("Research Assistant (LangChain)", run_research_assistant_workflow)]
    elif args.workflow == 4:
        from workflows.workflow_sales_qualification import run_sales_qualification_workflow

        workflows = [("Sales Lead Qualification (LangGraph)", run_sales_qualification_workflow)]
    elif args.workflow == 5:
        from workflows.workflow_gemini_async_streaming import run_gemini_async_streaming_workflow

        workflows = [("Gemini Async Streaming", run_gemini_async_streaming_workflow)]

    completed: list[str] = []

    for name, workflow_func in workflows:
        try:
            workflow_func(settings)
            completed.append(name)
        except KeyboardInterrupt:
            print("\nInterrupted by user.")
            break
        except Exception as e:
            print(f"\nWorkflow '{name}' failed: {type(e).__name__}: {e}\n")

    print("\nFlushing traces to NeatLogs...")
    neatlogs.flush()
    neatlogs.shutdown()

    if completed:
        print_summary(completed)
    else:
        print("\nNo workflows completed successfully.\n")


if __name__ == "__main__":
    main()
