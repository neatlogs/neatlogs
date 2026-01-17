"""
Test LangChain with various tool types to see what OpenInference captures.

This file tests:
1. HTTP API calls (via requests)
2. Subprocess calls
3. Custom Python functions
4. LangChain built-in tools
5. Tool error handling
"""

import os
import sys
import subprocess
import requests
from typing import Optional

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

import neatlogs
from langchain_classic.agents import AgentExecutor, create_openai_functions_agent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import Tool, StructuredTool
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

# Initialize NeatLogs
neatlogs.init(
    api_key=os.getenv("NEATLOGS_API_KEY", "EQZO8zBsJkDRvmIL06m-EQ7faMUHEIN5"),
    tags=["v4", "langchain", "tool-testing"],
    workflow_name="langchain-tool-test",
    instrumentations=["langchain"]
)

# ============================================================================
# TOOL 1: HTTP API Call (GitHub API)
# ============================================================================
def fetch_github_user(username: str) -> str:
    """Fetch GitHub user information via API."""
    print(f"[Tool] Fetching GitHub user: {username}")
    try:
        # HTTP call - should be auto-captured if requests is instrumented
        response = requests.get(
            f"https://api.github.com/users/{username}",
            headers={"Accept": "application/vnd.github.v3+json"},
            timeout=5
        )
        if response.status_code == 200:
            data = response.json()
            return f"User: {data['login']}, Name: {data.get('name', 'N/A')}, Repos: {data['public_repos']}"
        else:
            return f"Failed to fetch user (status: {response.status_code})"
    except Exception as e:
        return f"Error: {str(e)}"


github_tool = Tool(
    name="fetch_github_user",
    func=fetch_github_user,
    description="Fetch GitHub user information by username. Input should be a GitHub username."
)


# ============================================================================
# TOOL 2: Subprocess Call
# ============================================================================
def execute_command(command: str) -> str:
    """Execute a shell command."""
    print(f"[Tool] Executing: {command}")
    try:
        # Subprocess - NOT auto-captured
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=5
        )
        output = result.stdout if result.stdout else result.stderr
        return f"[Exit {result.returncode}] {output[:200]}"
    except Exception as e:
        return f"Error executing command: {str(e)}"


command_tool = Tool(
    name="execute_command",
    func=execute_command,
    description="Execute a shell command. Input should be a valid shell command."
)


# ============================================================================
# TOOL 3: Custom Python Function (Complex Logic)
# ============================================================================
class CalculatorInput(BaseModel):
    """Input schema for calculator."""
    expression: str = Field(description="Mathematical expression to evaluate, e.g., '2 + 2 * 3'")


def safe_calculator(expression: str) -> str:
    """Safely evaluate mathematical expressions."""
    print(f"[Tool] Calculating: {expression}")
    try:
        # Pure Python logic - NOT auto-captured
        # Use safe evaluation (no exec/eval for security)
        allowed_chars = set("0123456789+-*/()., ")
        if not all(c in allowed_chars for c in expression):
            return "Error: Invalid characters in expression"
        
        result = eval(expression, {"__builtins__": {}}, {})
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"


calculator_tool = StructuredTool.from_function(
    func=safe_calculator,
    name="calculator",
    description="Calculate mathematical expressions. Input should be a valid math expression like '2 + 2'.",
    args_schema=CalculatorInput
)


# ============================================================================
# TOOL 4: Tool that Intentionally Fails (Test Error Capture)
# ============================================================================
def failing_tool(input_text: str) -> str:
    """A tool that always fails to test error handling."""
    print(f"[Tool] Failing tool called with: {input_text}")
    # This will raise an exception
    raise ValueError("This tool is designed to fail for testing error capture!")


error_tool = Tool(
    name="failing_tool",
    func=failing_tool,
    description="A tool that always fails. Use for testing error handling."
)


# ============================================================================
# TOOL 5: HTTP POST with JSON (More Complex API Call)
# ============================================================================
def post_to_webhook(message: str) -> str:
    """Send a POST request to a webhook (httpbin for testing)."""
    print(f"[Tool] Posting to webhook: {message}")
    try:
        # POST request - should be auto-captured
        response = requests.post(
            "https://httpbin.org/post",
            json={"message": message, "source": "neatlogs-test"},
            timeout=5
        )
        if response.status_code == 200:
            return f"Webhook success: {response.status_code}"
        else:
            return f"Webhook failed: {response.status_code}"
    except Exception as e:
        return f"Error: {str(e)}"


webhook_tool = Tool(
    name="post_to_webhook",
    func=post_to_webhook,
    description="Send a POST request to a webhook. Input should be a message string."
)


# ============================================================================
# CREATE LANGCHAIN AGENT
# ============================================================================

# LLM
llm = ChatOpenAI(
    api_key="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",
    model="gpt-4o-mini",
    temperature=0
)

# Tools list
tools = [
    github_tool,
    command_tool,
    calculator_tool,
    webhook_tool,
    # error_tool,  # Commented out to avoid breaking the flow
]

# Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant that can use tools to help users.
    
Available tools:
- fetch_github_user: Get GitHub user info
- execute_command: Run shell commands
- calculator: Calculate math expressions
- post_to_webhook: Send HTTP POST requests

Use these tools to complete the user's request."""),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Create agent
agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=10,
    return_intermediate_steps=True
)


# ============================================================================
# TEST SCENARIOS
# ============================================================================

def test_scenario_1():
    """Test HTTP API call tool."""
    print("\n" + "="*80)
    print("TEST 1: HTTP API Call (GitHub)")
    print("="*80)
    result = agent_executor.invoke({
        "input": "Fetch information about the GitHub user 'torvalds'"
    })
    print(f"\n✅ Result: {result['output']}")


def test_scenario_2():
    """Test subprocess call."""
    print("\n" + "="*80)
    print("TEST 2: Subprocess Call")
    print("="*80)
    result = agent_executor.invoke({
        "input": "Use execute_command to get the current date and time (run 'date' command)"
    })
    print(f"\n✅ Result: {result['output']}")


def test_scenario_3():
    """Test custom Python function."""
    print("\n" + "="*80)
    print("TEST 3: Custom Calculator Function")
    print("="*80)
    result = agent_executor.invoke({
        "input": "Calculate the result of (15 + 35) * 2 - 10"
    })
    print(f"\n✅ Result: {result['output']}")


def test_scenario_4():
    """Test HTTP POST."""
    print("\n" + "="*80)
    print("TEST 4: HTTP POST Request")
    print("="*80)
    result = agent_executor.invoke({
        "input": "Send a POST request to the webhook with message 'Hello from NeatLogs!'"
    })
    print(f"\n✅ Result: {result['output']}")


def test_scenario_5():
    """Test multi-tool scenario."""
    print("\n" + "="*80)
    print("TEST 5: Multi-Tool Complex Query")
    print("="*80)
    result = agent_executor.invoke({
        "input": """Do the following:
        1. Calculate 100 * 0.08 (to find 8% tax)
        2. Fetch GitHub user info for 'gvanrossum'
        3. Tell me the results"""
    })
    print(f"\n✅ Result: {result['output']}")


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    print("🚀 Starting LangChain Tool Testing...")
    print(f"📍 Backend URL: http://localhost:3000")
    print(f"🔧 Testing 5 scenarios:")
    print("   1. HTTP GET (GitHub API)")
    print("   2. Subprocess (shell command)")
    print("   3. Custom Python function (calculator)")
    print("   4. HTTP POST (webhook)")
    print("   5. Multi-tool complex query")
    print()
    
    try:
        test_scenario_1()
        test_scenario_2()
        test_scenario_3()
        test_scenario_4()
        test_scenario_5()
        
        print("\n" + "="*80)
        print("✅ All tests completed!")
        print("="*80)
        print("\n📊 Now check your NeatLogs UI at http://localhost:3000")
        print("\nExpected Captures:")
        print("  ✅ LLM calls (function calling) - YES")
        print("  ✅ Tool decisions - YES (as TOOL spans)")
        print("  ✅ HTTP requests (GET/POST) - YES (if auto-instrumented)")
        print("  ❌ Subprocess calls - NO")
        print("  ❌ Pure Python logic - NO")
        print("\nNote: Tool NAMES and INPUTS/OUTPUTS should be captured.")
        print("      External HTTP calls within tools should be captured as nested HTTP spans.")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
