"""
Agent 3: Reviewer (for cross-file stress test)
"""

from __future__ import annotations

from neatlogs import PromptTemplate, agent, embedding, retriever, tool, trace
from openai import OpenAI

client = OpenAI()

reviewer_prompt = PromptTemplate(
    "You are a review agent. Assess the execution quality:\n\n{{execution}}\n\nProvide a quality score and feedback."
)


@tool(name="quality_check")
def quality_check(text: str) -> dict:
    """Quality assurance check."""
    return {
        "completeness": len(text) > 50,
        "clarity": "step" in text.lower(),
        "score": 0.85,
    }


@retriever(name="retrieve_review_criteria")
def retrieve_review_criteria(query: str):
    """Retrieve review criteria from knowledge base."""
    return [
        {"content": "Execution should be complete and documented", "score": 0.9},
        {"content": "All steps should have validation", "score": 0.87},
    ]


@embedding(name="embed_execution_for_review")
def embed_execution_for_review(text: str):
    """Embed execution text for semantic search."""
    return [0.3] * 384


@agent(name="reviewer_agent")
def review_execution(execution: str) -> str:
    """
    Review execution quality.
    
    CRITICAL: Called from main file, should inherit trace context.
    """
    print("\n🔍 REVIEWER AGENT (in separate file)")

    with trace("reviewer_step", kind="AGENT", prompt_template=reviewer_prompt):
        # Embed execution
        execution_embedding = embed_execution_for_review(execution)

        # Retrieve review criteria
        criteria = retrieve_review_criteria(execution)

        # Quality check (tool call)
        quality = quality_check(execution)

        # LLM call to generate review
        messages = reviewer_prompt.compile(execution=execution)
        with trace("reviewer_llm_call", kind="LLM", prompt_template=reviewer_prompt):
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": messages}],
                temperature=0,
            )
            review = response.choices[0].message.content or ""

        print(f"  Review: {review[:100]}...")
        print(f"  Quality Check: {quality}")
        return review
