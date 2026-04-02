"""
Agent functions for the Anthropic code review workflow.

Agents:
  - Reviewer    (claude-sonnet-4-6,         tool-calling)  — checks syntax via check_syntax tool,
                                                              then returns issues as JSON
  - Fixer       (claude-sonnet-4-6,         streaming)     — rewrites code with fixes
  - Tester      (claude-haiku-4-5-20251001, streaming)     — writes pytest test cases
  - Documenter  (claude-haiku-4-5-20251001, non-streaming) — adds docstrings and docs
"""

import ast
import json

import anthropic
import neatlogs
from neatlogs import PromptTemplate, UserPromptTemplate

client = anthropic.Anthropic()

# ---------------------------------------------------------------------------
# Tool implementation — called only when the LLM requests it
# ---------------------------------------------------------------------------

@neatlogs.span(kind="TOOL", name="check_syntax", description="Check Python code for syntax errors")
def check_syntax(code: str) -> str:
    try:
        ast.parse(code)
        return "No syntax errors found."
    except SyntaxError as e:
        return f"SyntaxError at line {e.lineno}: {e.msg}"


# Anthropic tool definition (passed to the LLM)
_CHECK_SYNTAX_TOOL = {
    "name": "check_syntax",
    "description": "Check Python code for syntax errors using the AST parser. Call this before reviewing the code.",
    "input_schema": {
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "The Python code to check for syntax errors."},
        },
        "required": ["code"],
    },
}

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_reviewer_sys = PromptTemplate([{
    "role": "system",
    "content": "You are an expert code reviewer. Analyze the code and return a JSON array of issue objects with 'severity' (high/medium/low), 'line' (approximate), and 'description' fields. No other text.",
}])
_reviewer_user = UserPromptTemplate([{
    "role": "user",
    "content": "Review this Python code:\n\n```python\n{{code}}\n```",
}])

_fixer_sys = PromptTemplate([{
    "role": "system",
    "content": "You are a Python expert. Fix all the identified issues in the code. Return only the corrected code in a python code block, no explanations.",
}])
_fixer_user = UserPromptTemplate([{
    "role": "user",
    "content": "Original code:\n```python\n{{code}}\n```\n\nIssues to fix:\n{{issues}}\n\nReturn the fixed code.",
}])

_tester_sys = PromptTemplate([{
    "role": "system",
    "content": "You are a Python testing expert. Write comprehensive pytest test cases for the provided code. Include edge cases and error conditions.",
}])
_tester_user = UserPromptTemplate([{
    "role": "user",
    "content": "Write pytest tests for this code:\n\n```python\n{{code}}\n```",
}])

_documenter_sys = PromptTemplate([{
    "role": "system",
    "content": "You are a Python documentation specialist. Add clear docstrings to all functions and classes, and add a module-level docstring. Return only the documented code.",
}])
_documenter_user = UserPromptTemplate([{
    "role": "user",
    "content": "Add documentation to this code:\n\n```python\n{{code}}\n```",
}])

# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------

@neatlogs.span(kind="AGENT", name="reviewer", role="Code Reviewer", goal="Identify code issues")
def reviewer_agent(code: str) -> list[dict]:
    with neatlogs.trace("review_code", kind="LLM", prompt_template=_reviewer_sys,
                        user_prompt_template=_reviewer_user):
        system_msg = _reviewer_sys.compile()[0]["content"]
        user_msg = _reviewer_user.compile(code=code)[0]["content"]
        messages = [{"role": "user", "content": user_msg}]

        # First call — model may call check_syntax tool before reviewing
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2048,
            system=system_msg,
            messages=messages,
            tools=[_CHECK_SYNTAX_TOOL],
            tool_choice={"type": "auto"},
        )

        # Execute any tool calls the model requested
        while response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result = check_syntax(block.input["code"])
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })
            messages.append({"role": "user", "content": tool_results})
            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=2048,
                system=system_msg,
                messages=messages,
                tools=[_CHECK_SYNTAX_TOOL],
                tool_choice={"type": "auto"},
            )

        raw = next(
            (block.text for block in response.content if hasattr(block, "text")), ""
        ).strip()

    try:
        issues = json.loads(raw)
    except json.JSONDecodeError:
        issues = [{"severity": "medium", "line": 0, "description": raw}]
    return issues


@neatlogs.span(kind="AGENT", name="fixer", role="Code Fixer", goal="Fix identified code issues")
def fixer_agent(code: str, issues: list[dict]) -> str:
    issues_text = "\n".join(
        f"- [{i['severity'].upper()}] line {i.get('line', '?')}: {i['description']}"
        for i in issues
    )
    with neatlogs.trace("fix_code", kind="LLM", prompt_template=_fixer_sys,
                        user_prompt_template=_fixer_user):
        system_msg = _fixer_sys.compile()[0]["content"]
        user_msg = _fixer_user.compile(code=code, issues=issues_text)[0]["content"]
        print("\n--- Fixer (streaming) ---")
        full = ""
        with client.messages.stream(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            system=system_msg,
            messages=[{"role": "user", "content": user_msg}],
        ) as stream:
            for text in stream.text_stream:
                print(text, end="", flush=True)
                full += text
        print("\n------------------------\n")
    return full


@neatlogs.span(kind="AGENT", name="tester", role="Test Writer", goal="Write pytest test cases")
def tester_agent(code: str) -> str:
    with neatlogs.trace("write_tests", kind="LLM", prompt_template=_tester_sys,
                        user_prompt_template=_tester_user):
        system_msg = _tester_sys.compile()[0]["content"]
        user_msg = _tester_user.compile(code=code)[0]["content"]
        print("\n--- Tester (streaming) ---")
        full = ""
        with client.messages.stream(
            model="claude-haiku-4-5-20251001",
            max_tokens=4096,
            system=system_msg,
            messages=[{"role": "user", "content": user_msg}],
        ) as stream:
            for text in stream.text_stream:
                print(text, end="", flush=True)
                full += text
        print("\n-------------------------\n")
    return full


@neatlogs.span(kind="AGENT", name="documenter", role="Documentation Writer", goal="Add docstrings and module docs")
def documenter_agent(code: str) -> str:
    with neatlogs.trace("add_docs", kind="LLM", prompt_template=_documenter_sys,
                        user_prompt_template=_documenter_user):
        system_msg = _documenter_sys.compile()[0]["content"]
        user_msg = _documenter_user.compile(code=code)[0]["content"]
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=4096,
            system=system_msg,
            messages=[{"role": "user", "content": user_msg}],
        )
    return response.content[0].text
