"""
LangGraph StateGraph definition for the multi-provider research workflow.

Topology:
  START → supervisor
            ↓              ↓             ↓          (parallel fan-out)
    web_researcher    wiki_researcher  arxiv_researcher
         ⇅ web_tools       ⇅ wiki_tools     ⇅ arxiv_tools   (LLM↔tools loops)
    web_done          wiki_done        arxiv_done
            ↓              ↓             ↓          (fan-in — waits for all three)
                      synthesizer
                          ↓
                    report_writer → END

Each researcher branch uses ToolNode with its own messages_key so parallel
branches are fully isolated. tools_condition routes back to the LLM node until
the model stops requesting tools, then goes to the *_done node.
"""

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

from state import ResearchState
from agents import (
    supervisor_node,
    web_researcher_node,
    web_done_node,
    wiki_researcher_node,
    wiki_done_node,
    arxiv_researcher_node,
    arxiv_done_node,
    synthesizer_node,
    report_writer_node,
    WEB_TOOLS,
    WIKI_TOOLS,
    ARXIV_TOOLS,
)


def build_graph() -> StateGraph:
    g = StateGraph(ResearchState)

    # Core nodes
    g.add_node("supervisor", supervisor_node)
    g.add_node("synthesizer", synthesizer_node)
    g.add_node("report_writer", report_writer_node)

    # Per-branch LLM nodes
    g.add_node("web_researcher", web_researcher_node)
    g.add_node("wiki_researcher", wiki_researcher_node)
    g.add_node("arxiv_researcher", arxiv_researcher_node)

    # Per-branch ToolNode — each reads/writes its own messages_key
    g.add_node("web_tools", ToolNode(WEB_TOOLS, messages_key="web_messages"))
    g.add_node("wiki_tools", ToolNode(WIKI_TOOLS, messages_key="wiki_messages"))
    g.add_node("arxiv_tools", ToolNode(ARXIV_TOOLS, messages_key="arxiv_messages"))

    # Per-branch "done" nodes — extract final text result
    g.add_node("web_done", web_done_node)
    g.add_node("wiki_done", wiki_done_node)
    g.add_node("arxiv_done", arxiv_done_node)

    # Entry
    g.add_edge(START, "supervisor")

    # Parallel fan-out after supervisor
    g.add_edge("supervisor", "web_researcher")
    g.add_edge("supervisor", "wiki_researcher")
    g.add_edge("supervisor", "arxiv_researcher")

    # Web researcher loop
    g.add_conditional_edges(
        "web_researcher",
        lambda s: tools_condition(s, messages_key="web_messages"),
        {"tools": "web_tools", "__end__": "web_done"},
    )
    g.add_edge("web_tools", "web_researcher")

    # Wiki researcher loop
    g.add_conditional_edges(
        "wiki_researcher",
        lambda s: tools_condition(s, messages_key="wiki_messages"),
        {"tools": "wiki_tools", "__end__": "wiki_done"},
    )
    g.add_edge("wiki_tools", "wiki_researcher")

    # ArXiv researcher loop
    g.add_conditional_edges(
        "arxiv_researcher",
        lambda s: tools_condition(s, messages_key="arxiv_messages"),
        {"tools": "arxiv_tools", "__end__": "arxiv_done"},
    )
    g.add_edge("arxiv_tools", "arxiv_researcher")

    # Fan-in — synthesizer waits for all three branches to complete
    g.add_edge("web_done", "synthesizer")
    g.add_edge("wiki_done", "synthesizer")
    g.add_edge("arxiv_done", "synthesizer")

    g.add_edge("synthesizer", "report_writer")
    g.add_edge("report_writer", END)

    return g.compile()


graph = build_graph()
