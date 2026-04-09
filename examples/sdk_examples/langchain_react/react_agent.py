"""
LangChain ReAct agent with custom tools, a retriever, and a report-formatting chain.

Topology:
  run_workflow
    ├── react_agent_executor  (ReAct loop: LLM ↔ tools)
    │     ├── knowledge_base_search  (custom retriever — mocked vector store)
    │     ├── web_search             (custom tool)
    │     ├── arxiv_search           (custom tool)
    │     └── calculate              (custom tool)
    └── report_chain  (LCEL runnable: prompt | llm, formats final report)

Tools are mocked (no real HTTP calls or embeddings).
Uses Anthropic claude-haiku-4-5.

Usage:
    python react_agent.py

Required env vars:
    NEATLOGS_API_KEY
    ANTHROPIC_API_KEY
"""

import json
import os
import sys

# ---------------------------------------------------------------------------
# Path + env setup — must happen before neatlogs import
# ---------------------------------------------------------------------------

_sdk_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if _sdk_root not in sys.path:
    sys.path.insert(0, _sdk_root)

from dotenv import load_dotenv
load_dotenv()

os.environ.setdefault("NEATLOGS_LOG_SPANS", "true")
os.environ.setdefault("NEATLOGS_LOG_SPANS_FILE", "react_spans.log")
os.environ.setdefault("NEATLOGS_LOG_RAW_SPANS", "true")
os.environ.setdefault("NEATLOGS_LOG_RAW_SPANS_FILE", "react_raw_spans.log")

import neatlogs
from neatlogs import PromptTemplate, UserPromptTemplate

neatlogs.init(
    api_key=os.getenv("NEATLOGS_API_KEY", ""),
    endpoint=os.getenv("NEATLOGS_ENDPOINT", "http://localhost:4100"),
    workflow_name="langchain-react-agent",
    tags=["langchain", "react", "research", "retriever"],
    instrumentations=["langchain"],
    debug=True,
)

# ---------------------------------------------------------------------------
# Imports that must come after neatlogs.init()
# ---------------------------------------------------------------------------

from typing import List
from langchain_aws import ChatBedrockConverse
from langchain_core.tools import tool, render_text_description
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate as LCPromptTemplate

# ---------------------------------------------------------------------------
# Mocked vector store retriever
# ---------------------------------------------------------------------------

_KNOWLEDGE_BASE = {
    "quantum": [
        "Quantum computers use qubits that can exist in superposition, enabling parallel computation.",
        "Variational Quantum Eigensolvers (VQE) can simulate molecular energy surfaces cheaply.",
        "Quantum annealing is used for combinatorial optimization problems in drug binding.",
    ],
    "drug": [
        "Drug discovery involves target identification, hit finding, lead optimization, and clinical trials.",
        "Molecular docking predicts how small molecules bind to protein targets.",
        "ADMET properties (absorption, distribution, metabolism, excretion, toxicity) filter drug candidates.",
    ],
    "default": [
        "Recent advances combine classical ML with domain-specific algorithms for improved accuracy.",
        "Hybrid quantum-classical approaches show promise for near-term NISQ devices.",
        "Benchmarking studies show 10-100x speedup on specific problem classes.",
    ],
}


class MockVectorStoreRetriever(BaseRetriever):
    """Mocked retriever that returns pre-defined documents based on query keywords."""

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        with neatlogs.trace("knowledge_base_retriever", kind="RETRIEVER") as span:
            span.set_attribute("neatlogs.retrieval.query", query)
            span.set_attribute("neatlogs.retrieval.top_k", 4)
            query_lower = query.lower()
            docs = []
            for keyword, passages in _KNOWLEDGE_BASE.items():
                if keyword in query_lower:
                    docs.extend([Document(page_content=p, metadata={"source": f"kb_{keyword}"}) for p in passages])
            if not docs:
                docs = [Document(page_content=p, metadata={"source": "kb_default"}) for p in _KNOWLEDGE_BASE["default"]]
            docs = docs[:4]
            span.set_attribute(
                "neatlogs.retrieval.documents",
                json.dumps([{"content": d.page_content, "metadata": d.metadata} for d in docs]),
            )
        return docs


_retriever = MockVectorStoreRetriever()

# ---------------------------------------------------------------------------
# Custom tools
# ---------------------------------------------------------------------------

@tool
def knowledge_base_search(query: str) -> str:
    """Search the internal knowledge base for background facts on a topic."""
    docs = _retriever.invoke(query)
    results = "\n".join(f"- {doc.page_content}" for doc in docs)
    return f"Knowledge base results for '{query}':\n{results}"


@tool
def web_search(query: str) -> str:
    """Search the web for current news and information on a topic."""
    return (
        f"Web search results for '{query}':\n"
        f"- Recent developments show significant progress in this area.\n"
        f"- Industry experts highlight growing investment and adoption.\n"
        f"- Key players are actively publishing findings and case studies.\n"
        f"- Multiple startups and research groups are advancing the field."
    )


@tool
def arxiv_search(query: str) -> str:
    """Search ArXiv for recent academic papers on a topic."""
    return (
        f"ArXiv papers for '{query}':\n"
        f"- [2024] 'Advances in {query}: A Systematic Review' — 94% accuracy improvement.\n"
        f"- [2024] 'Benchmarking Methods for {query}' — new state-of-the-art baselines.\n"
        f"- [2025] 'Scaling {query} to Production' — practical deployment framework.\n"
        f"- [2025] 'Hybrid Approaches in {query}' — combines classical and quantum methods."
    )


@tool
def calculate(expression: str) -> str:
    """Evaluate a simple arithmetic expression (e.g. '2 ** 10' or '1024 / 8')."""
    try:
        allowed = set("0123456789+-*/.() ")
        if not all(c in allowed for c in expression):
            return "Error: only basic arithmetic is supported."
        result = eval(expression, {"__builtins__": {}})  # noqa: S307
        return str(result)
    except Exception as e:
        return f"Error: {e}"


@tool
def failing_tool(query: str) -> str:
    """A tool that always raises an error — used to test error span capture."""
    raise RuntimeError(f"Simulated tool failure for query: '{query}'")


TOOLS = [knowledge_base_search, web_search, arxiv_search, calculate]

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

_llm = ChatBedrockConverse(
    model=os.getenv("BEDROCK_MODEL", "us.anthropic.claude-haiku-4-5-20251001-v1:0"),
    region_name=os.getenv("AWS_REGION", "us-west-1"),
)

# ---------------------------------------------------------------------------
# Prompt templates (neatlogs)
# ---------------------------------------------------------------------------


_report_sys = PromptTemplate([{
    "role": "system",
    "content": "You are a technical writer. Format the research findings into a concise structured report.",
}])
_report_user = UserPromptTemplate([{
    "role": "user",
    "content": "Topic: {{topic}}\n\nResearch findings:\n{{findings}}\n\nWrite a short structured report.",
}])

# ---------------------------------------------------------------------------
# ReAct agent
# ---------------------------------------------------------------------------

_react_prompt = LCPromptTemplate.from_template(
    "You are a research assistant with access to the following tools:\n\n"
    "{tools}\n\n"
    "Use the following format:\n\n"
    "Question: the input question you must answer\n"
    "Thought: you should always think about what to do\n"
    "Action: the action to take, should be one of [{tool_names}]\n"
    "Action Input: the input to the action\n"
    "Observation: the result of the action\n"
    "... (this Thought/Action/Action Input/Observation can repeat N times)\n"
    "Thought: I now know the final answer\n"
    "Final Answer: the final answer to the original input question\n\n"
    "Begin!\n\n"
    "Question: {input}\n"
    "Thought:{agent_scratchpad}"
)

_react_neatlogs_sys = PromptTemplate(
    _react_prompt.template
    .replace("Thought:{agent_scratchpad}", "")
    .replace("{", "{{")
    .replace("}", "}}")
)

_agent = create_react_agent(_llm, TOOLS, _react_prompt)
_agent_executor = AgentExecutor(agent=_agent, tools=TOOLS, verbose=True, max_iterations=10)

# Error-case agent: same prompt, but toolset includes failing_tool
_error_tools = [failing_tool]
_error_agent = create_react_agent(_llm, _error_tools, _react_prompt)
_error_agent_executor = AgentExecutor(
    agent=_error_agent, tools=_error_tools, verbose=True, max_iterations=3
)

# ---------------------------------------------------------------------------
# Report-formatting chain
# ---------------------------------------------------------------------------

_report_lc_prompt = ChatPromptTemplate.from_messages([
    ("system", _report_sys.template[0]["content"]),
    ("user", _report_user.template[0]["content"].replace("{{", "{").replace("}}", "}")),
])
_report_chain = _report_lc_prompt | _llm

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

@neatlogs.span(kind="WORKFLOW", name="react_research_workflow")
def run_workflow(topic: str) -> str:
    # Step 1: ReAct agent gathers information using tools
    with neatlogs.trace("react_agent", kind="LLM", prompt_template=_react_neatlogs_sys):
        _react_neatlogs_sys.compile(
            tools=render_text_description(TOOLS),
            tool_names=", ".join(t.name for t in TOOLS),
            input=topic,
        )
        agent_result = _agent_executor.invoke({"input": topic})
    findings = agent_result.get("output", "")

    # Step 2: Format findings into a report
    with neatlogs.trace("report_writer", kind="LLM", prompt_template=_report_sys, user_prompt_template=_report_user):
        _report_sys.compile()
        _report_user.compile(topic=topic, findings=findings)
        report_result = _report_chain.invoke({"topic": topic, "findings": findings})
    if hasattr(report_result, "content"):
        return str(report_result.content)
    if isinstance(report_result, dict):
        return str(report_result.get("text") or report_result.get("output") or findings)
    return str(report_result) if report_result is not None else findings


@neatlogs.span(kind="WORKFLOW", name="error_research_workflow")
def run_error_workflow(topic: str) -> str:
    """Same structure as run_workflow but raises after the agent step to produce ERROR spans."""
    # Step 1: ReAct agent gathers information
    with neatlogs.trace("error_react_agent", kind="LLM", prompt_template=_react_neatlogs_sys):
        _react_neatlogs_sys.compile(
            tools=render_text_description(_error_tools),
            tool_names=", ".join(t.name for t in _error_tools),
            input=topic,
        )
        agent_result = _error_agent_executor.invoke({"input": topic})
        raise RuntimeError(f"Simulated pipeline failure after agent step for topic: '{topic}'")
    findings = agent_result.get("output", "")

    # Step 2: Format findings into a report (only reached if agent recovers)
    with neatlogs.trace("report_writer", kind="LLM", prompt_template=_report_sys, user_prompt_template=_report_user):
        _report_sys.compile()
        _report_user.compile(topic=topic, findings=findings)
        report_result = _report_chain.invoke({"topic": topic, "findings": findings})
    if hasattr(report_result, "content"):
        return str(report_result.content)
    return str(report_result) if report_result is not None else findings


if __name__ == "__main__":
    topic = "quantum computing in drug discovery"
    print(f"Researching: {topic}\n")
    report = run_workflow(topic)
    print("\n--- Final Report ---")
    print(report)

    # print("\n--- Running error case ---")
    # try:
    #     run_error_workflow(topic)
    # except Exception as exc:
    #     print(f"Caught expected error: {exc}")

    neatlogs.flush()
    neatlogs.shutdown()
