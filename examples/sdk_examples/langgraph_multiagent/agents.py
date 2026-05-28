"""
Agent node functions for the LangGraph multi-provider research workflow.

Providers:
  - Azure OpenAI  : supervisor, web researcher, report writer (streaming)
  - Google Gemini : wiki researcher, arxiv researcher, synthesizer

Graph topology per researcher branch (LangGraph handles tool calling natively):
  supervisor → web_researcher ⇄ web_tools (loop until no tool_calls) → web_done
             → wiki_researcher ⇄ wiki_tools (loop)                   → wiki_done
             → arxiv_researcher ⇄ arxiv_tools (loop)                 → arxiv_done
  web_done + wiki_done + arxiv_done → synthesizer → report_writer → END

Each LLM node uses neatlogs.trace() for prompt template capture.
@neatlogs.span(kind="AGENT") is used on the top-level researcher functions.
"""

import os

import neatlogs
from neatlogs import SystemPromptTemplate, UserPromptTemplate

from langchain_openai import AzureChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

from state import ResearchState
from tools import web_search, wiki_search, arxiv_search

WEB_TOOLS = [web_search]
WIKI_TOOLS = [wiki_search]
ARXIV_TOOLS = [arxiv_search]

# ---------------------------------------------------------------------------
# LLM clients
# ---------------------------------------------------------------------------

_azure_kwargs = dict(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version=os.getenv("OPENAI_API_VERSION"),
    azure_deployment=os.environ["AZURE_LLM_DEPLOYMENT"],
)

_supervisor_llm = AzureChatOpenAI(**_azure_kwargs)
_web_llm = AzureChatOpenAI(**_azure_kwargs)
_wiki_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
_arxiv_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
_synth_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
_writer_llm = AzureChatOpenAI(**_azure_kwargs)

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_supervisor_sys = SystemPromptTemplate([{
    "role": "system",
    "content": "You are a research supervisor. Given a topic, write a concise 1-2 sentence research plan.",
}])
_supervisor_user = UserPromptTemplate([{"role": "user", "content": "Research topic: {{topic}}"}])

_web_sys = SystemPromptTemplate([{
    "role": "system",
    "content": "You are a web research specialist. Use the search tool to find current information. Return findings as bullet points.",
}])
_web_user = UserPromptTemplate([{"role": "user", "content": "Topic: {{topic}}\nPlan: {{plan}}"}])

_wiki_sys = SystemPromptTemplate([{
    "role": "system",
    "content": "You are a Wikipedia specialist. Use the search tool to find encyclopedic background. Return key facts as bullet points.",
}])
_wiki_user = UserPromptTemplate([{"role": "user", "content": "Topic: {{topic}}"}])

_arxiv_sys = SystemPromptTemplate([{
    "role": "system",
    "content": "You are an academic research specialist. Use the search tool to find recent papers. Summarize key findings as bullet points.",
}])
_arxiv_user = UserPromptTemplate([{"role": "user", "content": "Topic: {{topic}}"}])

_synth_sys = SystemPromptTemplate([{
    "role": "system",
    "content": "You are a research synthesizer. Combine findings from multiple sources into a coherent summary. Identify common themes.",
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

_writer_sys = SystemPromptTemplate([{
    "role": "system",
    "content": "You are a research report writer. Write a clear, structured report with executive summary, key findings, and conclusion. Use markdown.",
}])
_writer_user = UserPromptTemplate([{
    "role": "user",
    "content": "Topic: {{topic}}\n\nSynthesis:\n{{synthesis}}\n\nWrite a complete research report.",
}])

# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------

@neatlogs.span(kind="AGENT", name="supervisor", role="Research Supervisor", goal="Generate a research plan for the topic")
def supervisor_node(state: ResearchState) -> dict:
    topic = state["query"]
    with neatlogs.trace("supervisor", kind="LLM",
                        prompt_template=_supervisor_sys,
                        user_prompt_template=_supervisor_user):
        msgs = _supervisor_sys.compile() + _supervisor_user.compile(topic=topic)
        response = _supervisor_llm.invoke(msgs)
    return {"plan": response.content}


@neatlogs.span(kind="AGENT", name="web_researcher", role="Web Researcher", goal="Search the web for relevant information")
def web_researcher_node(state: ResearchState) -> dict:
    topic = state["query"]
    plan = state.get("plan", "")
    messages = state.get("web_messages") or []
    with neatlogs.trace("web_researcher", kind="LLM",
                        prompt_template=_web_sys,
                        user_prompt_template=_web_user):
        if not messages:
            initial = _web_sys.compile() + _web_user.compile(topic=topic, plan=plan)
            msgs = initial
        else:
            initial = None
            msgs = messages
        llm_with_tool = _web_llm.bind_tools(WEB_TOOLS)
        ai_msg = llm_with_tool.invoke(msgs)
    if initial is not None:
        return {"web_messages": initial + [ai_msg]}
    return {"web_messages": [ai_msg]}


def web_done_node(state: ResearchState) -> dict:
    last_msg = state["web_messages"][-1]
    return {"web_results": last_msg.content or ""}


@neatlogs.span(kind="AGENT", name="wiki_researcher", role="Wiki Researcher", goal="Find encyclopedic background on Wikipedia")
def wiki_researcher_node(state: ResearchState) -> dict:
    topic = state["query"]
    messages = state.get("wiki_messages") or []
    with neatlogs.trace("wiki_researcher", kind="LLM",
                        prompt_template=_wiki_sys,
                        user_prompt_template=_wiki_user):
        if not messages:
            initial = _wiki_sys.compile() + _wiki_user.compile(topic=topic)
            msgs = initial
        else:
            initial = None
            msgs = messages
        llm_with_tool = _wiki_llm.bind_tools(WIKI_TOOLS)
        ai_msg = llm_with_tool.invoke(msgs)
    if initial is not None:
        return {"wiki_messages": initial + [ai_msg]}
    return {"wiki_messages": [ai_msg]}


def wiki_done_node(state: ResearchState) -> dict:
    last_msg = state["wiki_messages"][-1]
    return {"wiki_results": last_msg.content or ""}


@neatlogs.span(kind="AGENT", name="arxiv_researcher", role="ArXiv Researcher", goal="Find recent academic papers on ArXiv")
def arxiv_researcher_node(state: ResearchState) -> dict:
    topic = state["query"]
    messages = state.get("arxiv_messages") or []
    with neatlogs.trace("arxiv_researcher", kind="LLM",
                        prompt_template=_arxiv_sys,
                        user_prompt_template=_arxiv_user):
        if not messages:
            initial = _arxiv_sys.compile() + _arxiv_user.compile(topic=topic)
            msgs = initial
        else:
            initial = None
            msgs = messages
        llm_with_tool = _arxiv_llm.bind_tools(ARXIV_TOOLS)
        ai_msg = llm_with_tool.invoke(msgs)
    if initial is not None:
        return {"arxiv_messages": initial + [ai_msg]}
    return {"arxiv_messages": [ai_msg]}


def arxiv_done_node(state: ResearchState) -> dict:
    last_msg = state["arxiv_messages"][-1]
    return {"arxiv_results": last_msg.content or ""}


@neatlogs.span(kind="AGENT", name="synthesizer", role="Research Synthesizer", goal="Combine findings from all researchers into a unified summary")
def synthesizer_node(state: ResearchState) -> dict:
    with neatlogs.trace("synthesizer", kind="LLM",
                        prompt_template=_synth_sys,
                        user_prompt_template=_synth_user):
        msgs = _synth_sys.compile() + _synth_user.compile(
            topic=state["query"],
            web_results=state.get("web_results", "N/A"),
            wiki_results=state.get("wiki_results", "N/A"),
            arxiv_results=state.get("arxiv_results", "N/A"),
        )
        response = _synth_llm.invoke(msgs)
    return {"synthesis": response.content}


@neatlogs.span(kind="AGENT", name="report_writer", role="Report Writer", goal="Write a complete structured research report")
def report_writer_node(state: ResearchState) -> dict:
    with neatlogs.trace("report_writer", kind="LLM",
                        prompt_template=_writer_sys,
                        user_prompt_template=_writer_user):
        msgs = _writer_sys.compile() + _writer_user.compile(
            topic=state["query"],
            synthesis=state.get("synthesis", ""),
        )
        response = _writer_llm.invoke(msgs)
        full = response.content if hasattr(response, "content") else str(response)
        print(full)
    return {"final_report": full}
