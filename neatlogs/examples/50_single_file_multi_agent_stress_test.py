"""
Stress Test 1: Single-File Multi-Agent Custom Orchestration
============================================================

Tests whether multiple sequential `with trace()` blocks within ONE outer trace
all share the same trace_id.

Scenario:
- Outer workflow span
- 3 sequential agents (each with different prompt templates)
- Each agent uses: embedding → retriever → reranker → LLM → tool
- Verify: ALL spans have SAME trace_id

Run:
  python neatlogs/examples/50_single_file_multi_agent_stress_test.py
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List

import requests

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
_env_default("NEATLOGS_LOG_SPANS_FILE", "spans_50_single_file_stress.jsonl")
_env_default("NEATLOGS_LOG_RAW_SPANS_FILE", "spans_raw_50_single_file_stress.jsonl")
_env_default("NEATLOGS_LOG_METRICS_FILE", "metrics_50_single_file_stress.jsonl")


def _require_env(key: str) -> str:
    val = os.getenv(key)
    if not val:
        raise SystemExit(f"Missing required env var: {key}")
    return val


# Initialize SDK
init(
    api_key=_require_env("NEATLOGS_API_KEY"),
    endpoint="http://localhost:3000/api/data/v4/batch",
    workflow_name="single-file-multi-agent-stress",
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
    """Search the web using Tavily API."""
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if not tavily_api_key:
        raise SystemExit("Missing required env var: TAVILY_API_KEY")
    
    response = requests.post(
        "https://api.tavily.com/search",
        json={
            "api_key": tavily_api_key,
            "query": query,
            "max_results": 3,
        },
        timeout=10,
    )
    response.raise_for_status()
    
    data = response.json()
    results = [result.get("content", "") for result in data.get("results", [])]
    
    return json.dumps({"query": query, "results": results})


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


@workflow(name="multi_agent_orchestration")
def run_multi_agent_workflow(query: str) -> str:
    """
    Orchestrate 3 agents sequentially.
    
    CRITICAL: All agent calls are nested under this @workflow decorator,
    so they should all share the same trace_id.
    """
    print(f"\n{'='*60}")
    print(f"🚀 MULTI-AGENT WORKFLOW: {query}")
    print(f"{'='*60}")

    # Sequential agent execution
    research_result = research_agent(query)
    analysis_result = analysis_agent(research_result)
    summary_result = summary_agent(analysis_result)

    print(f"\n{'='*60}")
    print(f"✅ WORKFLOW COMPLETE")
    print(f"{'='*60}")

    return summary_result


def main():
    try:
        query = "What are the key components of modern AI agent systems?"
        final_answer = run_multi_agent_workflow(query)

        print(f"\n📝 Final Answer:\n{final_answer}\n")

    except KeyboardInterrupt:
        raise SystemExit(130)
    finally:
        flush()
        shutdown()
        print("\n✅ Check output files:")
        print(f"   - spans_50_single_file_stress.jsonl")
        print(f"   - spans_raw_50_single_file_stress.jsonl")
        print(f"   - metrics_50_single_file_stress.jsonl")
        print("\nVerify: All spans should have the SAME trace_id")
        print("Run: jq '.trace_id' spans_50_single_file_stress.jsonl | sort | uniq -c")


if __name__ == "__main__":
    main()
