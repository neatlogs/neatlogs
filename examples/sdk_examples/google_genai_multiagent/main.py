"""
Entry point for the Google GenAI blog post creation workflow.

Custom Python orchestration with @neatlogs.span decorators.

Usage:
    python main.py
    python main.py "The future of renewable energy"

Required env vars:
    NEATLOGS_API_KEY
    GOOGLE_API_KEY
"""

import os
import sys

# Add local SDK to path
_sdk_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if _sdk_root not in sys.path:
    sys.path.insert(0, _sdk_root)

from dotenv import load_dotenv
load_dotenv()

os.environ.setdefault("NEATLOGS_LOG_SPANS", "true")
os.environ.setdefault("NEATLOGS_LOG_SPANS_FILE", "google_genai_multiagent_spans.log")
os.environ.setdefault("NEATLOGS_LOG_RAW_SPANS", "true")
os.environ.setdefault("NEATLOGS_LOG_RAW_SPANS_FILE", "google_genai_multiagent_raw_spans.log")

import neatlogs

neatlogs.init(
    api_key=os.getenv("NEATLOGS_API_KEY", ""),
    endpoint=os.getenv("NEATLOGS_ENDPOINT", "http://localhost:4100"),
    workflow_name="google-genai-content-creation",
    tags=["google-genai", "content", "blog"],
    instrumentations=["google_genai"],
    debug=True,
)

from agents import ideation_agent, writer_agent, editor_agent, finalizer_agent


@neatlogs.span(kind="WORKFLOW", name="blog_creation_workflow")
def run_blog_creation(topic: str) -> str:
    print(f"\n=== Blog Creation: {topic} ===\n")

    print("--- Ideation: generating content ideas ---")
    idea = ideation_agent(topic)
    print(f"  Selected idea: {idea.get('title')}")

    print("\n--- Writer: drafting post ---")
    draft = writer_agent(topic, idea)

    print("\n--- Editor: improving draft ---")
    edited = editor_agent(topic, draft)

    print("\n--- Finalizer: polishing post ---")
    final = finalizer_agent(topic, edited)
    print("\n--- Final Post ---")
    print(final)

    return final


if __name__ == "__main__":
    topic = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "The future of AI in healthcare"
    run_blog_creation(topic)
    neatlogs.flush()
    neatlogs.shutdown()
