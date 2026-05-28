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

from dotenv import load_dotenv

load_dotenv()

# neatlogs.init() MUST come before creating the Google GenAI client.
# google.genai.Client caches its transport at construction time, so the client
# must be created after init() for auto-instrumentation to take effect.
import neatlogs

neatlogs.init(
    api_key=os.getenv("NEATLOGS_API_KEY"),
    endpoint=os.getenv("NEATLOGS_ENDPOINT"),
    workflow_name="blog-creation",
    tags=["sdk-examples", "google-genai", "content", "blog"],
    instrumentations=["google_genai"],
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
