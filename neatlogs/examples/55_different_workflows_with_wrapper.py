"""
Stress Test 55: Different Workflows WITH Outer Wrapper
=======================================================

Tests whether calling DIFFERENT workflow functions within ONE outer trace
all share the same trace_id.

Scenario:
- Outer: with trace("batch_session", kind="WORKFLOW")
- Inside: workflow_A (2-agent: research + analysis)
- Inside: workflow_B (1-agent: summary only)
- Verify: Both different workflows share the SAME trace_id

Expected: 1 trace containing spans from both workflows

Run:
  python neatlogs/examples/55_different_workflows_with_wrapper.py
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from neatlogs import (
    PromptTemplate,
    agent,
    chain,
    embedding,
    flush,
    init,
    retriever,
    shutdown,
    tool,
    trace,
    workflow,
)


def _env_default(k: str, v: str) -> None:
    os.environ.setdefault(k, v)


_env_default("LANGCHAIN_TRACING_V2", "false")
_env_default("NEATLOGS_LOG_SPANS", "true")
_env_default("NEATLOGS_LOG_METRICS", "true")
_env_default("NEATLOGS_LOG_RAW_SPANS", "true")
_env_default("NEATLOGS_LOG_SPANS_FILE", "spans_55_different_workflows_with_wrapper.jsonl")
_env_default("NEATLOGS_LOG_RAW_SPANS_FILE", "spans_raw_55_different_workflows_with_wrapper.jsonl")
_env_default("NEATLOGS_LOG_METRICS_FILE", "metrics_55_different_workflows_with_wrapper.jsonl")


def _require_env(key: str) -> str:
    val = os.getenv(key)
    if not val:
        raise SystemExit(f"Missing required env var: {key}")
    return val


# Initialize SDK
init(
    api_key=_require_env("NEATLOGS_API_KEY"),
    endpoint="http://localhost:3000/api/data/v4/batch",
    workflow_name="different-workflows-with-wrapper",
    instrumentations=["openai"],
    debug=True,
)

from openai import OpenAI

client = OpenAI()


# ============================================================================
# Prompts for 3 Agents
# ============================================================================

research_prompt = PromptTemplate(
    "You are a research agent. Analyze the following context and extract key insights:\n\n{{context}}\n\nProvide 3 bullet points."
)

analysis_prompt = PromptTemplate(
    "You are an analysis agent. Given research findings, identify patterns:\n\n{{research}}\n\nProvide analysis in 2 sentences."
)

summary_prompt = PromptTemplate(
    "You are a summary agent. Synthesize the analysis:\n\n{{analysis}}\n\nProvide a 1-sentence summary."
)


# ============================================================================
# Helper Functions (Tools, Retriever, etc.)
# ============================================================================


@tool(name="web_search", tool_name="web_search")
def web_search(query: str) -> str:
    """Simulate web search."""
    return json.dumps(
        {
            "query": query,
            "results": [
                "AI agents use memory to store context",
                "Vector databases enable semantic search",
                "RAG improves LLM accuracy",
            ],
        }
    )


@tool(name="validate_output", tool_name="validate_output")
def validate_output(text: str) -> Dict[str, Any]:
    """Validate agent output."""
    return {
        "valid": len(text) > 10,
        "length": len(text),
        "has_bullet_points": "•" in text or "-" in text,
    }


@retriever(name="retrieve_context")
def retrieve_context(query: str, top_k: int = 3) -> List[Dict[str, str]]:
    """Simulate document retrieval."""
    docs = [
        {"content": "AI agents combine LLMs with tools and memory.", "score": 0.95},
        {"content": "Vector search enables semantic similarity matching.", "score": 0.89},
        {"content": "RAG retrieves relevant context before generation.", "score": 0.87},
        {"content": "Multi-agent systems decompose complex tasks.", "score": 0.82},
    ]
    return docs[:top_k]


@embedding(name="embed_query")
def embed_query(text: str) -> List[float]:
    """Simulate embedding generation."""
    # Fake embedding (384-dim for testing)
    return [0.1] * 384


@chain(name="rerank_documents")
def rerank_documents(query: str, docs: List[Dict]) -> List[Dict]:
    """Simulate reranking."""
    # Sort by score descending
    return sorted(docs, key=lambda d: d.get("score", 0), reverse=True)[:2]


# ============================================================================
# Agent Functions
# ============================================================================


@agent(name="research_agent")
def research_agent(query: str) -> str:
    """Research agent: retrieves and analyzes context."""
    print("\n🔍 RESEARCH AGENT")

    with trace("research_agent_step", kind="AGENT", prompt_template=research_prompt):
        # Step 1: Embed query
        query_embedding = embed_query(query)

        # Step 2: Retrieve documents
        retrieved_docs = retrieve_context(query, top_k=4)

        # Step 3: Rerank
        reranked_docs = rerank_documents(query, retrieved_docs)

        # Step 4: LLM call with context
        context = "\n".join([d["content"] for d in reranked_docs])
        messages = research_prompt.compile(context=context)

        with trace("research_llm_call", kind="LLM", prompt_template=research_prompt):
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": messages}],
                temperature=0,
            )
            result = response.choices[0].message.content or ""

        # Step 5: Validate output
        validation = validate_output(result)
        print(f"  Research Result: {result[:100]}...")
        print(f"  Validation: {validation}")

        return result


@agent(name="analysis_agent")
def analysis_agent(research: str) -> str:
    """Analysis agent: identifies patterns."""
    print("\n📊 ANALYSIS AGENT")

    with trace("analysis_agent_step", kind="AGENT", prompt_template=analysis_prompt):
        messages = analysis_prompt.compile(research=research)

        with trace("analysis_llm_call", kind="LLM", prompt_template=analysis_prompt):
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": messages}],
                temperature=0,
            )
            result = response.choices[0].message.content or ""

        # Web search for additional context
        search_result = web_search(f"patterns in {result[:50]}")
        print(f"  Analysis Result: {result[:100]}...")

        return result


@agent(name="summary_agent")
def summary_agent(analysis: str) -> str:
    """Summary agent: synthesizes final output."""
    print("\n✨ SUMMARY AGENT")

    with trace("summary_agent_step", kind="AGENT", prompt_template=summary_prompt):
        messages = summary_prompt.compile(analysis=analysis)

        with trace("summary_llm_call", kind="LLM", prompt_template=summary_prompt):
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": messages}],
                temperature=0,
            )
            result = response.choices[0].message.content or ""

        print(f"  Summary Result: {result}")

        return result


# ============================================================================
# Main Workflow
# ============================================================================


@workflow(name="workflow_a_research_analysis")
def workflow_a(query: str) -> str:
    """
    Workflow A: Research + Analysis (2 agents)
    """
    print(f"\n{'='*60}")
    print(f"🚀 WORKFLOW A (Research + Analysis): {query}")
    print(f"{'='*60}")

    research_result = research_agent(query)
    analysis_result = analysis_agent(research_result)

    print(f"\n{'='*60}")
    print(f"✅ WORKFLOW A COMPLETE")
    print(f"{'='*60}")

    return analysis_result


@workflow(name="workflow_b_summary")
def workflow_b(analysis: str) -> str:
    """
    Workflow B: Summary only (1 agent)
    """
    print(f"\n{'='*60}")
    print(f"🚀 WORKFLOW B (Summary): {analysis[:50]}...")
    print(f"{'='*60}")

    summary_result = summary_agent(analysis)

    print(f"\n{'='*60}")
    print(f"✅ WORKFLOW B COMPLETE")
    print(f"{'='*60}")

    return summary_result


def main():
    try:
        # Outer wrapper - both DIFFERENT workflows should share this trace
        with trace("batch_processing_session", kind="WORKFLOW"):
            print("\n🔹 WORKFLOW A: Research + Analysis")
            query = "What are the key components of modern AI agent systems?"
            analysis_result = workflow_a(query)

            print("\n🔹 WORKFLOW B: Summary")
            final_result = workflow_b(analysis_result)

            print(f"\n📝 Final Answer:\n{final_result}\n")

    except KeyboardInterrupt:
        raise SystemExit(130)
    finally:
        flush()
        shutdown()
        print("\n✅ Check output files:")
        print(f"   - spans_55_different_workflows_with_wrapper.jsonl")
        print(f"   - spans_raw_55_different_workflows_with_wrapper.jsonl")
        print(f"   - metrics_55_different_workflows_with_wrapper.jsonl")
        print("\nVerify: Should have 1 SINGLE trace_id (both workflows share it)")
        print("Run: jq '.trace_id' spans_55_different_workflows_with_wrapper.jsonl | sort | uniq -c")


if __name__ == "__main__":
    main()
