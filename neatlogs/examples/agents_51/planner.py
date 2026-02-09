"""
Agent 1: Planner (for cross-file stress test)
"""

from __future__ import annotations

from neatlogs import PromptTemplate, agent, embedding, retriever, tool, trace
from openai import OpenAI

client = OpenAI()

planner_prompt = PromptTemplate(
    "You are a planning agent. Break down this task into 3 concrete steps:\n\n{{task}}"
)


@tool(name="fetch_similar_plans")
def fetch_similar_plans(query: str) -> str:
    """Retrieve similar historical plans."""
    return "Historical plans: [Step-by-step planning, Define scope, Allocate resources]"


@retriever(name="retrieve_planning_docs")
def retrieve_planning_docs(query: str):
    """Simulate retrieval of planning documentation."""
    return [
        {"content": "Best practices: Start with requirements", "score": 0.92},
        {"content": "Break tasks into milestones", "score": 0.88},
    ]


@embedding(name="embed_task_for_planning")
def embed_task_for_planning(text: str):
    """Embed task for semantic search."""
    return [0.2] * 384


@agent(name="planner_agent")
def plan_task(task: str) -> str:
    """
    Plan a task by breaking it into steps.
    
    CRITICAL: This function is called from main file, so it should inherit
    the trace context from the parent @workflow decorator.
    """
    print("\n📋 PLANNER AGENT (in separate file)")

    with trace("planner_step", kind="AGENT", prompt_template=planner_prompt):
        # Embed task
        task_embedding = embed_task_for_planning(task)

        # Retrieve planning docs
        docs = retrieve_planning_docs(task)

        # Fetch similar plans (tool call)
        similar = fetch_similar_plans(task)

        # LLM call
        messages = planner_prompt.compile(task=task)
        with trace("planner_llm_call", kind="LLM", prompt_template=planner_prompt):
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": messages}],
                temperature=0,
            )
            plan = response.choices[0].message.content or ""

        print(f"  Plan: {plan[:100]}...")
        return plan
