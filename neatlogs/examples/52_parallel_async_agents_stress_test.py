"""
Stress Test 3: Parallel + Async Multi-Agent Custom Orchestration
================================================================

Tests whether `with trace()` blocks executed in PARALLEL and ASYNC contexts
all share the same trace_id.

Scenario:
- Main workflow spawns 3 agents in parallel using asyncio
- Each agent has async operations (embedding, retrieval, LLM calls)
- Uses ThreadPoolExecutor for MCP tool simulation
- Verify: ALL spans from all parallel agents have SAME trace_id

Run:
  python neatlogs/examples/52_parallel_async_agents_stress_test.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from neatlogs import (
    PromptTemplate,
    agent,
    chain,
    embedding,
    flush,
    init,
    mcp_tool,
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
_env_default("NEATLOGS_LOG_SPANS_FILE", "spans_52_parallel_async_stress.jsonl")
_env_default("NEATLOGS_LOG_RAW_SPANS_FILE", "spans_raw_52_parallel_async_stress.jsonl")
_env_default("NEATLOGS_LOG_METRICS_FILE", "metrics_52_parallel_async_stress.jsonl")


def _require_env(key: str) -> str:
    val = os.getenv(key)
    if not val:
        raise SystemExit(f"Missing required env var: {key}")
    return val


# Initialize SDK
init(
    api_key=_require_env("NEATLOGS_API_KEY"),
    endpoint="http://localhost:3000/api/data/v4/batch",
    workflow_name="parallel-async-multi-agent-stress",
    instrumentations=["openai"],
    debug=True,
)

from openai import AsyncOpenAI

client = AsyncOpenAI()

# Thread pool for MCP tool simulation
executor = ThreadPoolExecutor(max_workers=4)


# ============================================================================
# Prompts for 3 Parallel Agents
# ============================================================================

researcher_prompt = PromptTemplate(
    "You are a researcher agent. Research this topic: {{topic}}. Provide 2 key findings."
)

analyzer_prompt = PromptTemplate(
    "You are an analyzer agent. Analyze this topic: {{topic}}. Provide 2 insights."
)

validator_prompt = PromptTemplate(
    "You are a validator agent. Validate this topic: {{topic}}. Provide validation status."
)


# ============================================================================
# Async Tools and Decorators
# ============================================================================


@tool(name="async_web_search")
async def async_web_search(query: str) -> str:
    """Async web search simulation."""
    await asyncio.sleep(0.1)  # Simulate network delay
    return json.dumps(
        {
            "query": query,
            "results": [
                "AI agents enable autonomous task execution",
                "Async operations improve agent scalability",
                "Context propagation ensures trace consistency",
            ],
        }
    )


@mcp_tool(name="mcp_database_query", description="Simulate database query via MCP")
def mcp_database_query(query: str) -> Dict[str, Any]:
    """MCP tool: Database query (blocking, runs in thread pool)."""
    # Simulate blocking I/O
    import time

    time.sleep(0.05)
    return {
        "query": query,
        "results": [
            {"id": 1, "name": "Agent A", "status": "active"},
            {"id": 2, "name": "Agent B", "status": "idle"},
        ],
    }


@retriever(name="async_retrieve_docs")
async def async_retrieve_docs(query: str, top_k: int = 2) -> List[Dict[str, str]]:
    """Async document retrieval."""
    await asyncio.sleep(0.1)  # Simulate vector DB query
    docs = [
        {"content": "Parallel agents improve throughput", "score": 0.94},
        {"content": "Async operations reduce latency", "score": 0.91},
        {"content": "Context propagation maintains trace integrity", "score": 0.88},
    ]
    return docs[:top_k]


@embedding(name="async_embed_query")
async def async_embed_query(text: str) -> List[float]:
    """Async embedding generation."""
    await asyncio.sleep(0.05)  # Simulate embedding API call
    return [0.4] * 384


@chain(name="async_rerank_docs")
async def async_rerank_docs(query: str, docs: List[Dict]) -> List[Dict]:
    """Async reranking."""
    await asyncio.sleep(0.05)  # Simulate reranker API
    return sorted(docs, key=lambda d: d.get("score", 0), reverse=True)[:2]


# ============================================================================
# Async Agent Functions
# ============================================================================


@agent(name="researcher_agent_async")
async def researcher_agent_async(topic: str) -> str:
    """Async researcher agent."""
    print(f"\n🔬 RESEARCHER AGENT (async, parallel)")

    with trace("researcher_step", kind="AGENT", prompt_template=researcher_prompt):
        # Embed query
        query_embedding = await async_embed_query(topic)

        # Retrieve docs
        retrieved_docs = await async_retrieve_docs(topic, top_k=3)

        # Rerank
        reranked_docs = await async_rerank_docs(topic, retrieved_docs)

        # MCP tool call (blocking, run in thread pool)
        loop = asyncio.get_event_loop()
        db_results = await loop.run_in_executor(
            executor, mcp_database_query, f"SELECT * FROM agents WHERE topic='{topic}'"
        )

        # LLM call
        context = "\n".join([d["content"] for d in reranked_docs])
        messages = researcher_prompt.compile(topic=topic)

        with trace("researcher_llm_call", kind="LLM", prompt_template=researcher_prompt):
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": messages}],
                temperature=0,
            )
            result = response.choices[0].message.content or ""

        print(f"  Researcher Result: {result[:80]}...")
        return result


@agent(name="analyzer_agent_async")
async def analyzer_agent_async(topic: str) -> str:
    """Async analyzer agent."""
    print(f"\n📈 ANALYZER AGENT (async, parallel)")

    with trace("analyzer_step", kind="AGENT", prompt_template=analyzer_prompt):
        # Embed query
        query_embedding = await async_embed_query(topic)

        # Web search (async tool)
        search_results = await async_web_search(topic)

        # LLM call
        messages = analyzer_prompt.compile(topic=topic)

        with trace("analyzer_llm_call", kind="LLM", prompt_template=analyzer_prompt):
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": messages}],
                temperature=0,
            )
            result = response.choices[0].message.content or ""

        print(f"  Analyzer Result: {result[:80]}...")
        return result


@agent(name="validator_agent_async")
async def validator_agent_async(topic: str) -> str:
    """Async validator agent."""
    print(f"\n✅ VALIDATOR AGENT (async, parallel)")

    with trace("validator_step", kind="AGENT", prompt_template=validator_prompt):
        # Retrieve validation criteria
        criteria_docs = await async_retrieve_docs(f"{topic} validation", top_k=2)

        # MCP tool call (blocking, run in thread pool)
        loop = asyncio.get_event_loop()
        validation_status = await loop.run_in_executor(
            executor, mcp_database_query, f"SELECT status FROM validations WHERE topic='{topic}'"
        )

        # LLM call
        messages = validator_prompt.compile(topic=topic)

        with trace("validator_llm_call", kind="LLM", prompt_template=validator_prompt):
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": messages}],
                temperature=0,
            )
            result = response.choices[0].message.content or ""

        print(f"  Validator Result: {result[:80]}...")
        return result


# ============================================================================
# Main Workflow (Async + Parallel)
# ============================================================================


@workflow(name="parallel_async_orchestration")
async def run_parallel_async_workflow(topic: str) -> Dict[str, str]:
    """
    Orchestrate 3 agents in parallel using asyncio.gather.
    
    CRITICAL: All parallel agents should share the same trace_id because
    they're spawned under this @workflow decorator which sets the trace context.
    """
    print(f"\n{'='*60}")
    print(f"🚀 PARALLEL ASYNC WORKFLOW: {topic}")
    print(f"{'='*60}")

    # Launch all 3 agents in parallel
    results = await asyncio.gather(
        researcher_agent_async(topic),
        analyzer_agent_async(topic),
        validator_agent_async(topic),
    )

    research, analysis, validation = results

    print(f"\n{'='*60}")
    print(f"✅ WORKFLOW COMPLETE (all agents executed in parallel)")
    print(f"{'='*60}")

    return {
        "research": research,
        "analysis": analysis,
        "validation": validation,
    }


def main():
    try:
        # Outer wrapper to ensure all parallel async agents share one trace
        with trace("parallel_async_session", kind="WORKFLOW"):
            topic = "Multi-agent orchestration with trace context propagation"

            # Run async workflow
            results = asyncio.run(run_parallel_async_workflow(topic))

            print(f"\n📝 Final Results:")
            for key, value in results.items():
                print(f"  {key.upper()}: {value[:100]}...")

    except KeyboardInterrupt:
        raise SystemExit(130)
    finally:
        executor.shutdown(wait=True)
        flush()
        shutdown()
        print("\n✅ Check output files:")
        print(f"   - spans_52_parallel_async_stress.jsonl")
        print(f"   - spans_raw_52_parallel_async_stress.jsonl")
        print(f"   - metrics_52_parallel_async_stress.jsonl")
        print("\nVerify: All spans (from 3 parallel async agents) should have SAME trace_id")
        print("Run: jq '.trace_id' spans_52_parallel_async_stress.jsonl | sort | uniq -c")


if __name__ == "__main__":
    main()
