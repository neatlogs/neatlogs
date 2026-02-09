"""
Agent 2: Executor (for cross-file stress test)
"""

from __future__ import annotations

from neatlogs import PromptTemplate, agent, chain, tool, trace
from openai import OpenAI

client = OpenAI()

executor_prompt = PromptTemplate(
    "You are an execution agent. Execute this plan step-by-step:\n\n{{plan}}\n\nProvide execution results."
)


@tool(name="execute_step")
def execute_step(step: str) -> str:
    """Simulate step execution."""
    return f"Executed: {step} ✅"


@chain(name="validate_execution")
def validate_execution(result: str) -> dict:
    """Validate execution results."""
    return {"valid": "✅" in result, "length": len(result)}


@agent(name="executor_agent")
def execute_plan(plan: str) -> str:
    """
    Execute a plan step-by-step.
    
    CRITICAL: Called from main file, should inherit trace context.
    """
    print("\n⚙️  EXECUTOR AGENT (in separate file)")

    with trace("executor_step", kind="AGENT", prompt_template=executor_prompt):
        # Simulate executing each step
        steps = ["Step 1", "Step 2", "Step 3"]
        for step in steps:
            result = execute_step(step)
            validation = validate_execution(result)

        # LLM call to summarize execution
        messages = executor_prompt.compile(plan=plan)
        with trace("executor_llm_call", kind="LLM", prompt_template=executor_prompt):
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": messages}],
                temperature=0,
            )
            execution_summary = response.choices[0].message.content or ""

        print(f"  Execution Summary: {execution_summary[:100]}...")
        return execution_summary
