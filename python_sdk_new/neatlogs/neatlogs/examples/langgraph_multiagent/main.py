"""
Entry point for the LangGraph multi-provider research workflow.

Usage:
    python main.py                    # non-streaming
    python main.py --stream           # streaming (prints node completions live)

Required env vars:
    NEATLOGS_API_KEY
    AZURE_OPENAI_ENDPOINT
    AZURE_OPENAI_API_KEY
    AZURE_LLM_DEPLOYMENT
    ANTHROPIC_API_KEY
    GOOGLE_API_KEY
"""

import os
import sys

# Add local SDK to path
_sdk_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if _sdk_root not in sys.path:
    sys.path.insert(0, _sdk_root)

os.environ.setdefault("NEATLOGS_LOG_SPANS", "true")
os.environ.setdefault("NEATLOGS_LOG_SPANS_FILE", "langgraph_multiagent_spans.log")
os.environ.setdefault("NEATLOGS_LOG_RAW_SPANS", "true")
os.environ.setdefault("NEATLOGS_LOG_RAW_SPANS_FILE", "langgraph_multiagent_raw_spans.log")

import neatlogs

neatlogs.init(
    api_key=os.getenv("NEATLOGS_API_KEY", ""),
    endpoint=os.getenv("NEATLOGS_ENDPOINT", "http://localhost:4100"),
    workflow_name="langgraph-multiagent",
    tags=["langgraph", "multi-provider", "research"],
    instrumentations=["langchain"],
    debug=True,
)

import sys, os; sys.path.insert(0, os.path.dirname(__file__))
from graph import graph  # noqa: E402 — must import after neatlogs.init()


@neatlogs.span(kind="WORKFLOW", name="research_workflow")
def run_workflow(query: str, stream: bool = False) -> str:
    initial_state = {
        "query": query,
        "plan": "",
        "web_messages": [],
        "wiki_messages": [],
        "arxiv_messages": [],
        "web_results": "",
        "wiki_results": "",
        "arxiv_results": "",
        "synthesis": "",
        "final_report": "",
        "messages": [],
    }

    if stream:
        print(f"\nResearching: {query}\n")
        for event in graph.stream(initial_state):
            for node_name in event:
                print(f"[{node_name}] completed")
        return ""
    else:
        result = graph.invoke(initial_state)
        return result.get("final_report", "")


if __name__ == "__main__":
    stream_mode = "--stream" in sys.argv
    topic = "CRISPR gene editing in cancer treatment"
    report = run_workflow(topic, stream=stream_mode)
    if not stream_mode and report:
        print("\n--- Final Report ---")
        print(report)
    neatlogs.flush()
    neatlogs.shutdown()
