"""
Detection Demo - Multi-Framework Workflows
===========================================
Runs 4 multi-agent workflows to demonstrate detection system:
1. Customer Support (LangGraph) - 6 test queries
2. Content Moderation (CrewAI) - 6 test samples  
3. Research Assistant (LangChain) - 6 test queries
4. Sales Lead Qualification (LangGraph) - 8 test leads

Total: 26 test scenarios triggering classifiers, regex, and conditional detections

Usage:
    python main.py                    # Run all workflows
    python main.py --workflow 1       # Run only Customer Support (LangGraph)
    python main.py --workflow 2       # Run only Content Moderation (CrewAI)
    python main.py --workflow 3       # Run only Research Assistant (LangChain)
    python main.py --workflow 4       # Run only Sales Lead Qualification (LangGraph)

Prerequisites:
    1. Copy .env.example to .env and fill in your keys
    2. Install dependencies: pip install -r requirements.txt
"""

import argparse
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

from examples.sdk_examples.detection_demo.config import load_settings, init_neatlogs
import neatlogs


def print_banner():
    """Print startup banner."""
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                  DETECTION DEMO - MULTI-FRAMEWORK WORKFLOWS                  ║
║                                                                              ║
║  Showcasing AI Agent Detection System with:                                 ║
║  • 5 Workflows: Customer Support, Content Mod, Research, Sales, PII Demo    ║
║  • 33 Test Scenarios                                                         ║
║  • Detections: Classifiers, Regex, Conditional, PII Redaction               ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
    print(banner)


def print_summary(workflows_run: list):
    """Print final summary."""
    print("\n" + "="*80)
    print("DETECTION DEMO - SUMMARY")
    print("="*80)
    print(f"\n✅ Successfully completed {len(workflows_run)} workflow(s):\n")
    for workflow in workflows_run:
        print(f"   • {workflow}")
    print(f"\n{'─'*80}")
    print("Traces have been sent to Neatlogs.")
    print("Check your dashboard to view:")
    print("  • Multi-agent orchestration")
    print("  • Span nesting (WORKFLOW → AGENT → LLM/TOOL)")
    print("  • Detection triggers (nsfw, hate, jailbreaking, refusals)")
    print("  • RAG pipeline (RETRIEVER → RERANKER)")
    print("  • PII redaction (SDK mask + server-side Presidio, all operators)")
    print(f"{'─'*80}\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Detection Demo - Multi-Framework Workflows",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                 # Run all 4 workflows
  python main.py --workflow 1    # Customer Support only
  python main.py --workflow 2    # Content Moderation only (requires CrewAI)
  python main.py --workflow 3    # Research Assistant only
  python main.py --workflow 4    # Sales Lead Qualification only
        """
    )
    parser.add_argument(
        "--workflow", "-w",
        type=int,
        choices=[1, 2, 3, 4, 5, 6, 7],
        help="Run specific workflow (1=Customer Support, 2=Content Moderation, 3=Research Assistant, 4=Sales Qualification, 5=PII Redaction, 7=Gemini Async Streaming)",
    )
    args = parser.parse_args()

    # Print banner
    print_banner()

    # Load configuration
    try:
        settings = load_settings()
    except RuntimeError as e:
        print(f"\n❌ Configuration Error:\n{e}\n")
        sys.exit(1)

    # Initialize Neatlogs
    try:
        init_neatlogs(settings)
    except Exception as e:
        print(f"\n❌ Neatlogs Initialization Error:\n{e}\n")
        sys.exit(1)

    # Determine which workflows to run (lazy imports to avoid dependency issues)
    workflows = []
    
    if args.workflow is None:
        # Run all workflows (skip CrewAI if not installed)
        from examples.sdk_examples.detection_demo.workflows.workflow_customer_support import run_customer_support_workflow
        from examples.sdk_examples.detection_demo.workflows.workflow_research_assistant import run_research_assistant_workflow
        from examples.sdk_examples.detection_demo.workflows.workflow_sales_qualification import run_sales_qualification_workflow
        from examples.sdk_examples.detection_demo.workflows.workflow_pii_redaction import run_pii_redaction_workflow
        workflows = [
            ("Customer Support (LangGraph)", run_customer_support_workflow),
            ("Research Assistant (LangChain)", run_research_assistant_workflow),
            ("Sales Lead Qualification (LangGraph)", run_sales_qualification_workflow),
            ("PII Redaction Demo", run_pii_redaction_workflow),
        ]
        # Try to add CrewAI workflow if available
        try:
            from examples.sdk_examples.detection_demo.workflows.workflow_content_moderation import run_content_moderation_workflow
            workflows.insert(1, ("Content Moderation (CrewAI)", run_content_moderation_workflow))
        except ImportError:
            print("⚠️  Skipping Content Moderation (CrewAI not installed)\n")
    elif args.workflow == 1:
        from examples.sdk_examples.detection_demo.workflows.workflow_customer_support import run_customer_support_workflow
        workflows = [("Customer Support (LangGraph)", run_customer_support_workflow)]
    elif args.workflow == 2:
        from examples.sdk_examples.detection_demo.workflows.workflow_content_moderation import run_content_moderation_workflow
        workflows = [("Content Moderation (CrewAI)", run_content_moderation_workflow)]
    elif args.workflow == 3:
        from examples.sdk_examples.detection_demo.workflows.workflow_research_assistant import run_research_assistant_workflow
        workflows = [("Research Assistant (LangChain)", run_research_assistant_workflow)]
    elif args.workflow == 4:
        from examples.sdk_examples.detection_demo.workflows.workflow_sales_qualification import run_sales_qualification_workflow
        workflows = [("Sales Lead Qualification (LangGraph)", run_sales_qualification_workflow)]
    elif args.workflow == 5:
        from examples.sdk_examples.detection_demo.workflows.workflow_pii_redaction import run_pii_redaction_workflow
        workflows = [("PII Redaction Demo", run_pii_redaction_workflow)]
    elif args.workflow == 6:
        from examples.sdk_examples.detection_demo.workflows.workflow_prompt_management import run_prompt_management_workflow
        workflows = [("Prompt Management", run_prompt_management_workflow)]
    elif args.workflow == 7:
        from examples.sdk_examples.detection_demo.workflows.workflow_gemini_async_streaming import run_gemini_async_streaming_workflow
        workflows = [("Gemini Async Streaming", run_gemini_async_streaming_workflow)]

    # Run workflows
    completed_workflows = []
    
    for name, workflow_func in workflows:
        try:
            workflow_func(settings)
            completed_workflows.append(name)
        except KeyboardInterrupt:
            print(f"\n\n⚠️  Workflow interrupted by user")
            break
        except Exception as e:
            print(f"\n❌ Workflow '{name}' failed:")
            print(f"   {type(e).__name__}: {e}")
            if settings.debug:
                import traceback
                print("\nFull traceback:")
                traceback.print_exc()
            print(f"\nContinuing to next workflow...\n")

    # Flush traces
    print("\n" + "="*80)
    print("Flushing traces to Neatlogs...")
    print("="*80)
    
    try:
        neatlogs.flush()
        neatlogs.shutdown()
        print("✓ Traces flushed successfully")
    except Exception as e:
        print(f"⚠️  Warning: Failed to flush traces: {e}")

    # Print summary
    if completed_workflows:
        print_summary(completed_workflows)
    else:
        print("\n⚠️  No workflows completed successfully.\n")

    print("Done!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Program interrupted by user. Exiting...\n")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)
