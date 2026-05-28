"""
LangGraph research assistant — classify → research → summarize → respond
(or classify → respond for general chitchat).

Usage:
    python main.py

Required env:
    NEATLOGS_API_KEY
    AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY
    AZURE_LLM_DEPLOYMENT (or AZURE_OPENAI_DEPLOYMENT_NAME)
"""

from __future__ import annotations

import os
from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv

load_dotenv()

# neatlogs.init() MUST come before importing LangChain / LangGraph.
import neatlogs

neatlogs.init(
    api_key=os.getenv("NEATLOGS_API_KEY"),
    endpoint=os.getenv("NEATLOGS_ENDPOINT"),
    workflow_name="research-assistant",
    tags=["sdk-examples", "langgraph", "routing", "research-assistant"],
    instrumentations=["langchain"],
    capture_logs=True,
)

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    query_type: str
    research_notes: str
    summary: str


_deployment = os.getenv("AZURE_LLM_DEPLOYMENT") or os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
if not _deployment:
    raise SystemExit(
        "Set AZURE_LLM_DEPLOYMENT or AZURE_OPENAI_DEPLOYMENT_NAME in your environment."
    )

llm = AzureChatOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version=os.getenv("OPENAI_API_VERSION", "2025-01-01-preview"),
    azure_deployment=_deployment,
)


@neatlogs.span(kind="CHAIN", name="classify_query")
def classify_query(state: AgentState) -> dict:
    last_msg = state["messages"][-1].content
    if not isinstance(last_msg, str):
        last_msg = str(last_msg)

    result = llm.invoke(
        [
            SystemMessage(
                content=(
                    "You are a query classifier. Respond with ONLY one word:\n"
                    "- 'research' if the query needs factual research, analysis, or explanation\n"
                    "- 'general' if it's a greeting, chitchat, or simple question"
                )
            ),
            HumanMessage(content=last_msg),
        ]
    )

    query_type = (result.content or "").strip().lower()
    if query_type not in ("research", "general"):
        query_type = "research"

    print(f"  [classify] → {query_type}")
    neatlogs.log("classified query as {query_type}", query_type=query_type)
    return {"query_type": query_type}


@neatlogs.span(kind="TOOL", tool_name="deep_research")
def research_node(state: AgentState) -> dict:
    last_msg = state["messages"][-1].content
    if not isinstance(last_msg, str):
        last_msg = str(last_msg)

    result = llm.invoke(
        [
            SystemMessage(
                content=(
                    "You are a research analyst. Given the user's question, provide "
                    "detailed research notes with 4-5 key findings. Include specific "
                    "facts, comparisons, or data points. Format as bullet points."
                )
            ),
            HumanMessage(content=last_msg),
        ]
    )

    notes = result.content or ""
    print(f"  [research] → gathered {len(notes)} chars of notes")
    neatlogs.log("research notes length={n} chars", n=len(notes))
    return {"research_notes": notes}


@neatlogs.span(kind="CHAIN", name="summarize")
def summarize_node(state: AgentState) -> dict:
    research = state.get("research_notes", "")

    result = llm.invoke(
        [
            SystemMessage(
                content=(
                    "You are a summarizer. Take the following research notes and "
                    "distill them into 2-3 crisp, actionable takeaways. Be concise."
                )
            ),
            HumanMessage(content=research),
        ]
    )

    summary = result.content or ""
    print(f"  [summarize] → {len(summary)} chars")
    return {"summary": summary}


@neatlogs.span(kind="CHAIN", name="generate_response")
def respond_node(state: AgentState) -> dict:
    last_msg = state["messages"][-1].content
    if not isinstance(last_msg, str):
        last_msg = str(last_msg)
    summary = state.get("summary", "")
    query_type = state.get("query_type", "general")

    if query_type == "research" and summary:
        system = (
            "You are a helpful assistant. Answer the user's question using "
            "the following research summary. Be clear and well-structured.\n\n"
            f"Research Summary:\n{summary}"
        )
    else:
        system = "You are a friendly assistant. Respond naturally and helpfully."

    result = llm.invoke(
        [
            SystemMessage(content=system),
            HumanMessage(content=last_msg),
        ]
    )

    out = result.content or ""
    print(f"  [respond] → {len(out)} chars")
    return {"messages": [AIMessage(content=out)]}


def route_by_type(state: AgentState) -> Literal["research", "respond"]:
    return "research" if state.get("query_type") == "research" else "respond"


graph = StateGraph(AgentState)
graph.add_node("classify", classify_query)
graph.add_node("research", research_node)
graph.add_node("summarize", summarize_node)
graph.add_node("respond", respond_node)

graph.add_edge(START, "classify")
graph.add_conditional_edges(
    "classify",
    route_by_type,
    {"research": "research", "respond": "respond"},
)
graph.add_edge("research", "summarize")
graph.add_edge("summarize", "respond")
graph.add_edge("respond", END)

app = graph.compile()


@neatlogs.span(kind="WORKFLOW", name="run_workflow")
def run_workflow(user_query: str) -> str:
    result = app.invoke(
        {
            "messages": [HumanMessage(content=user_query)],
            "query_type": "",
            "research_notes": "",
            "summary": "",
        },
    )
    final = result["messages"][-1].content
    return final if isinstance(final, str) else str(final)


QUERIES = [
    "What are the main differences between RAG and fine-tuning for LLM customization?",
]


def run_all_queries() -> list[str]:
    """Run the full QUERIES list, return list of answers. Used by both
    the __main__ block and platform wrapper scripts so all four telemetry
    paths see the same workflow shape."""
    answers: list[str] = []
    for i, q in enumerate(QUERIES, 1):
        print(f"\n{'─' * 60}")
        print(f"Query {i}: {q}")
        print(f"{'─' * 60}")
        answer = run_workflow(q)
        preview = answer[:300] + ("..." if len(answer) > 300 else "")
        print(f"\nAnswer: {preview}")
        answers.append(answer)
    return answers


if __name__ == "__main__":
    print("=" * 60)
    print("  LangGraph research assistant (NeatLogs)")
    print("=" * 60)

    run_all_queries()

    neatlogs.flush()
    neatlogs.shutdown()
