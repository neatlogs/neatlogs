"""
Simple single-file LangGraph multi-agent research workflow.

Same topology as langgraph_multiagent:
  START → supervisor
            ↓           ↓           ↓        (parallel fan-out)
    web_researcher  wiki_researcher  arxiv_researcher
         ⇅ tools        ⇅ tools         ⇅ tools      (LLM↔tool loops)
    web_done        wiki_done       arxiv_done
            ↓           ↓           ↓        (fan-in)
                    synthesizer
                        ↓
                  report_writer → END

Tools are mocked (no real HTTP calls).
Uses a single LLM provider (Anthropic claude-haiku-4-5) for all nodes.

Usage:
    python simple.py

Required env vars:
    NEATLOGS_API_KEY
    ANTHROPIC_API_KEY
"""

import operator
import os
import sys

# ---------------------------------------------------------------------------
# Path + env setup — must happen before neatlogs import
# ---------------------------------------------------------------------------

_sdk_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if _sdk_root not in sys.path:
    sys.path.insert(0, _sdk_root)

os.environ.setdefault("NEATLOGS_LOG_SPANS", "true")
os.environ.setdefault("NEATLOGS_LOG_SPANS_FILE", "simple_spans.log")
os.environ.setdefault("NEATLOGS_LOG_RAW_SPANS", "true")
os.environ.setdefault("NEATLOGS_LOG_RAW_SPANS_FILE", "simple_raw_spans.log")

import neatlogs
from neatlogs import PromptTemplate, UserPromptTemplate

neatlogs.init(
    api_key=os.getenv("NEATLOGS_API_KEY", ""),
    endpoint=os.getenv("NEATLOGS_ENDPOINT", "http://localhost:4100"),
    workflow_name="simple-multiagent",
    tags=["langgraph", "simple", "research"],
    instrumentations=["langchain"],
    debug=True,
)

# ---------------------------------------------------------------------------
# Imports that must come after neatlogs.init()
# ---------------------------------------------------------------------------

from typing import Annotated, TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class ResearchState(TypedDict):
    query: str
    plan: str
    web_messages: Annotated[list, add_messages]
    wiki_messages: Annotated[list, add_messages]
    arxiv_messages: Annotated[list, add_messages]
    web_results: str
    wiki_results: str
    arxiv_results: str
    synthesis: str
    final_report: str
    messages: Annotated[list, operator.add]

# ---------------------------------------------------------------------------
# Mocked tools
# ---------------------------------------------------------------------------

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
def wiki_search(query: str) -> str:
    """Search Wikipedia for encyclopedic background on a topic."""
    return (
        f"Wikipedia summary for '{query}':\n"
        f"- The field has foundational theoretical work dating back decades.\n"
        f"- Core principles are widely studied across disciplines.\n"
        f"- Applications span medicine, finance, logistics, and materials science.\n"
        f"- Leading research institutions include MIT, Stanford, and national labs."
    )


@tool
def arxiv_search(query: str) -> str:
    """Search ArXiv for recent academic papers on a topic."""
    return (
        f"ArXiv papers for '{query}':\n"
        f"- [2024] 'Advances in {query}: A Systematic Review' — 94% accuracy improvement.\n"
        f"- [2024] 'Benchmarking Methods for {query}' — new state-of-the-art baselines.\n"
        f"- [2025] 'Scaling {query} to Production' — practical deployment framework."
    )


WEB_TOOLS = [web_search]
WIKI_TOOLS = [wiki_search]
ARXIV_TOOLS = [arxiv_search]

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

_llm = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0)

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_supervisor_sys = PromptTemplate([{
    "role": "system",
    "content": "You are a research supervisor. Given a topic, write a concise 1-2 sentence research plan.",
}])
_supervisor_user = UserPromptTemplate([{"role": "user", "content": "Research topic: {{topic}}"}])

_web_sys = PromptTemplate([{
    "role": "system",
    "content": "You are a web research specialist. Use the web_search tool to find current information. Return findings as bullet points.",
}])
_web_user = UserPromptTemplate([{"role": "user", "content": "Topic: {{topic}}\nPlan: {{plan}}"}])

_wiki_sys = PromptTemplate([{
    "role": "system",
    "content": "You are a Wikipedia specialist. Use the wiki_search tool to find encyclopedic background. Return key facts as bullet points.",
}])
_wiki_user = UserPromptTemplate([{"role": "user", "content": "Topic: {{topic}}"}])

_arxiv_sys = PromptTemplate([{
    "role": "system",
    "content": "You are an academic research specialist. Use the arxiv_search tool to find recent papers. Summarize key findings as bullet points.",
}])
_arxiv_user = UserPromptTemplate([{"role": "user", "content": "Topic: {{topic}}"}])

_synth_sys = PromptTemplate([{
    "role": "system",
    "content": "You are a research synthesizer. Combine findings from multiple sources into a coherent summary.",
}])
_synth_user = UserPromptTemplate([{
    "role": "user",
    "content": (
        "Topic: {{topic}}\n\n"
        "Web findings:\n{{web_results}}\n\n"
        "Wikipedia findings:\n{{wiki_results}}\n\n"
        "Academic findings:\n{{arxiv_results}}\n\n"
        "Synthesize these into a unified summary."
    ),
}])

_writer_sys = PromptTemplate([{
    "role": "system",
    "content": "You are a report writer. Write a short structured report with key findings and conclusion.",
}])
_writer_user = UserPromptTemplate([{
    "role": "user",
    "content": "Topic: {{topic}}\n\nSynthesis:\n{{synthesis}}\n\nWrite a concise research report.",
}])

# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------

def supervisor_node(state: ResearchState) -> dict:
    topic = state["query"]
    with neatlogs.trace("supervisor", kind="LLM", prompt_template=_supervisor_sys,
                        user_prompt_template=_supervisor_user):
        msgs = _supervisor_sys.compile() + _supervisor_user.compile(topic=topic)
        response = _llm.invoke(msgs)
    return {"plan": response.content}


def web_researcher_node(state: ResearchState) -> dict:
    topic = state["query"]
    plan = state.get("plan", "")
    messages = state.get("web_messages") or []
    with neatlogs.trace("web_researcher", kind="LLM", prompt_template=_web_sys,
                        user_prompt_template=_web_user):
        if not messages:
            initial = _web_sys.compile() + _web_user.compile(topic=topic, plan=plan)
            msgs = initial
        else:
            initial = None
            msgs = messages
        ai_msg = _llm.bind_tools(WEB_TOOLS).invoke(msgs)
    if initial is not None:
        return {"web_messages": initial + [ai_msg]}
    return {"web_messages": [ai_msg]}


def web_done_node(state: ResearchState) -> dict:
    last = state["web_messages"][-1]
    return {"web_results": last.content or ""}


def wiki_researcher_node(state: ResearchState) -> dict:
    topic = state["query"]
    messages = state.get("wiki_messages") or []
    with neatlogs.trace("wiki_researcher", kind="LLM", prompt_template=_wiki_sys,
                        user_prompt_template=_wiki_user):
        if not messages:
            initial = _wiki_sys.compile() + _wiki_user.compile(topic=topic)
            msgs = initial
        else:
            initial = None
            msgs = messages
        ai_msg = _llm.bind_tools(WIKI_TOOLS).invoke(msgs)
    if initial is not None:
        return {"wiki_messages": initial + [ai_msg]}
    return {"wiki_messages": [ai_msg]}


def wiki_done_node(state: ResearchState) -> dict:
    last = state["wiki_messages"][-1]
    return {"wiki_results": last.content or ""}


def arxiv_researcher_node(state: ResearchState) -> dict:
    topic = state["query"]
    messages = state.get("arxiv_messages") or []
    with neatlogs.trace("arxiv_researcher", kind="LLM", prompt_template=_arxiv_sys,
                        user_prompt_template=_arxiv_user):
        if not messages:
            initial = _arxiv_sys.compile() + _arxiv_user.compile(topic=topic)
            msgs = initial
        else:
            initial = None
            msgs = messages
        ai_msg = _llm.bind_tools(ARXIV_TOOLS).invoke(msgs)
    if initial is not None:
        return {"arxiv_messages": initial + [ai_msg]}
    return {"arxiv_messages": [ai_msg]}


def arxiv_done_node(state: ResearchState) -> dict:
    last = state["arxiv_messages"][-1]
    return {"arxiv_results": last.content or ""}


def synthesizer_node(state: ResearchState) -> dict:
    with neatlogs.trace("synthesizer", kind="LLM", prompt_template=_synth_sys,
                        user_prompt_template=_synth_user):
        msgs = _synth_sys.compile() + _synth_user.compile(
            topic=state["query"],
            web_results=state.get("web_results", "N/A"),
            wiki_results=state.get("wiki_results", "N/A"),
            arxiv_results=state.get("arxiv_results", "N/A"),
        )
        response = _llm.invoke(msgs)
    return {"synthesis": response.content}


def report_writer_node(state: ResearchState) -> dict:
    with neatlogs.trace("report_writer", kind="LLM", prompt_template=_writer_sys,
                        user_prompt_template=_writer_user):
        msgs = _writer_sys.compile() + _writer_user.compile(
            topic=state["query"],
            synthesis=state.get("synthesis", ""),
        )
        response = _llm.invoke(msgs)
    return {"final_report": response.content}

# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

def build_graph():
    g = StateGraph(ResearchState)

    g.add_node("supervisor", supervisor_node)
    g.add_node("web_researcher", web_researcher_node)
    g.add_node("wiki_researcher", wiki_researcher_node)
    g.add_node("arxiv_researcher", arxiv_researcher_node)
    g.add_node("web_tools", ToolNode(WEB_TOOLS, messages_key="web_messages"))
    g.add_node("wiki_tools", ToolNode(WIKI_TOOLS, messages_key="wiki_messages"))
    g.add_node("arxiv_tools", ToolNode(ARXIV_TOOLS, messages_key="arxiv_messages"))
    g.add_node("web_done", web_done_node)
    g.add_node("wiki_done", wiki_done_node)
    g.add_node("arxiv_done", arxiv_done_node)
    g.add_node("synthesizer", synthesizer_node)
    g.add_node("report_writer", report_writer_node)

    g.add_edge(START, "supervisor")

    g.add_edge("supervisor", "web_researcher")
    g.add_edge("supervisor", "wiki_researcher")
    g.add_edge("supervisor", "arxiv_researcher")

    g.add_conditional_edges(
        "web_researcher",
        lambda s: tools_condition(s, messages_key="web_messages"),
        {"tools": "web_tools", "__end__": "web_done"},
    )
    g.add_edge("web_tools", "web_researcher")

    g.add_conditional_edges(
        "wiki_researcher",
        lambda s: tools_condition(s, messages_key="wiki_messages"),
        {"tools": "wiki_tools", "__end__": "wiki_done"},
    )
    g.add_edge("wiki_tools", "wiki_researcher")

    g.add_conditional_edges(
        "arxiv_researcher",
        lambda s: tools_condition(s, messages_key="arxiv_messages"),
        {"tools": "arxiv_tools", "__end__": "arxiv_done"},
    )
    g.add_edge("arxiv_tools", "arxiv_researcher")

    g.add_edge("web_done", "synthesizer")
    g.add_edge("wiki_done", "synthesizer")
    g.add_edge("arxiv_done", "synthesizer")

    g.add_edge("synthesizer", "report_writer")
    g.add_edge("report_writer", END)

    return g.compile()


graph = build_graph()

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

@neatlogs.span(kind="WORKFLOW", name="research_workflow")
def run_workflow(query: str) -> str:
    result = graph.invoke({
        "query": query,
        "plan": "",
        "web_messages": [],
        "wiki_messages": [],
        "arxiv_messages": [],
        "web_results": "",
        "wiki_results": "",
        "arxiv_results": "",
        "synthesis": "",
        "final_report": "",
        "messages": [],
    })
    return result.get("final_report", "")


if __name__ == "__main__":
    topic = "quantum computing in drug discovery"
    print(f"Researching: {topic}\n")
    report = run_workflow(topic)
    print("\n--- Final Report ---")
    print(report)
    neatlogs.flush()
    neatlogs.shutdown()
