"""
Entry point for the LangGraph multi-provider research workflow.

Usage:
    python main.py                    # non-streaming
    python main.py --stream           # streaming (prints node completions live)

Required env vars:
    NEATLOGS_API_KEY
    AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_LLM_DEPLOYMENT
    GOOGLE_API_KEY  (for Gemini nodes)
"""

import os
import sys
import uuid

from dotenv import load_dotenv

load_dotenv()

# neatlogs.init() MUST come before importing LangChain / LangGraph.
import neatlogs

neatlogs.init(
    api_key=os.getenv("NEATLOGS_API_KEY"),
    endpoint=os.getenv("NEATLOGS_ENDPOINT"),
    workflow_name="multi-provider-research",
    tags=["sdk-examples", "langgraph", "multi-provider", "research"],
    instrumentations=["langchain"],
)

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

    run_config = {
        "configurable": {"thread_id": str(uuid.uuid4())},
        "recursion_limit": 25,
    }

    if stream:
        print(f"\nResearching: {query}\n")
        for event in graph.stream(initial_state, config=run_config):
            for node_name in event:
                print(f"[{node_name}] completed")
        return ""
    result = graph.invoke(initial_state, config=run_config)
    return result.get("final_report", "")


if __name__ == "__main__":
    stream_mode = "--stream" in sys.argv
    topic = "CRISPR gene editing in cancer treatment"
    try:
        report = run_workflow(topic, stream=stream_mode)
        if not stream_mode and report:
            print("\n--- Final Report ---")
            print(report)
    finally:
        neatlogs.flush()
        neatlogs.shutdown()
