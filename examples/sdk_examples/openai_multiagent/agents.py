"""
Agent functions for the OpenAI investment research workflow.

Agents:
  - Planner     (Azure OpenAI, non-streaming) — generates 3 research questions
  - Researcher  (Azure OpenAI, tool-calling)  — LLM calls web_search tool
  - Analyst     (Azure OpenAI, streaming)     — identifies investment themes
  - Reporter    (Azure OpenAI, streaming)     — writes final investment brief
"""

import json
import os

import neatlogs
from neatlogs import SystemPromptTemplate, UserPromptTemplate

from openai import AzureOpenAI

client = AzureOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version=os.getenv("OPENAI_API_VERSION", "2024-08-01-preview"),
)
DEPLOYMENT = os.environ["AZURE_LLM_DEPLOYMENT"]

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_planner_sys = SystemPromptTemplate([{
    "role": "system",
    "content": (
        "You are a financial research planner. Given a company or stock, return exactly 3 "
        "research questions as a JSON array of strings. No other text."
    ),
}])
_planner_user = UserPromptTemplate([{"role": "user", "content": "Company: {{company}}"}])

_researcher_sys = SystemPromptTemplate([{
    "role": "system",
    "content": (
        "You are a web research assistant. Use the web_search tool to find information "
        "for the given question, then summarize the findings as concise bullet points "
        "relevant to investment analysis."
    ),
}])
_researcher_user = UserPromptTemplate([{"role": "user", "content": "Research question: {{question}}"}])

_analyst_sys = SystemPromptTemplate([{
    "role": "system",
    "content": (
        "You are a senior investment analyst. Identify key investment themes, risks, "
        "and opportunities from the research findings."
    ),
}])
_analyst_user = UserPromptTemplate([{
    "role": "user",
    "content": "Company: {{company}}\n\nResearch findings:\n{{findings}}\n\nProvide a structured analysis.",
}])

_reporter_sys = SystemPromptTemplate([{
    "role": "system",
    "content": (
        "You are an investment report writer. Write a clear, professional investment brief "
        "with an executive summary, key findings, risks, and recommendation. Use markdown."
    ),
}])
_reporter_user = UserPromptTemplate([{
    "role": "user",
    "content": "Company: {{company}}\n\nAnalysis:\n{{analysis}}\n\nWrite a complete investment brief.",
}])

# ---------------------------------------------------------------------------
# Tool definition + implementation
# ---------------------------------------------------------------------------

WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web for current information on a topic.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query."},
            },
            "required": ["query"],
        },
    },
}


@neatlogs.span(kind="TOOL", tool_name="web_search", description="Mocked web search")
def web_search(query: str) -> str:
    return (
        f"- Mock result 1 for '{query}': Strong revenue growth and expanding market share.\n"
        f"- Mock result 2 for '{query}': Recent product launches receiving positive analyst coverage.\n"
        f"- Mock result 3 for '{query}': Management reaffirmed full-year guidance above consensus."
    )


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------

@neatlogs.span(kind="AGENT", name="planner", role="Research Planner", goal="Generate targeted research questions")
def planner_agent(company: str) -> list[str]:
    with neatlogs.trace("plan_questions", kind="LLM",
                        prompt_template=_planner_sys,
                        user_prompt_template=_planner_user):
        msgs = _planner_sys.compile() + _planner_user.compile(company=company)
        response = client.chat.completions.create(
            model=DEPLOYMENT,
            messages=msgs,
        )
        raw = response.choices[0].message.content.strip()
    try:
        questions = json.loads(raw)
    except json.JSONDecodeError:
        questions = [q.strip("- ").strip() for q in raw.split("\n") if q.strip()]
    questions = questions[:3]
    neatlogs.log("planner generated {count} questions for {company}",
                 count=len(questions), company=company)
    return questions


@neatlogs.span(kind="AGENT", name="researcher", role="Web Researcher", goal="Find current information on each question")
def researcher_agent(questions: list[str]) -> str:
    all_summaries = []
    for question in questions:
        neatlogs.log("researching question: {question}", question=question)
        with neatlogs.trace("research_question", kind="LLM",
                            prompt_template=_researcher_sys,
                            user_prompt_template=_researcher_user):
            msgs = _researcher_sys.compile() + _researcher_user.compile(question=question)

            response = client.chat.completions.create(
                model=DEPLOYMENT,
                messages=msgs,
                tools=[WEB_SEARCH_TOOL],
                tool_choice="auto",
            )
            ai_msg = response.choices[0].message
            msgs.append(ai_msg.model_dump(exclude_unset=True))

            if ai_msg.tool_calls:
                for tc in ai_msg.tool_calls:
                    args = json.loads(tc.function.arguments)
                    result = web_search(args["query"])
                    msgs.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    })

                final = client.chat.completions.create(
                    model=DEPLOYMENT,
                    messages=msgs,
                )
                summary = final.choices[0].message.content
            else:
                summary = ai_msg.content

        all_summaries.append(f"Q: {question}\n{summary}")
    return "\n\n".join(all_summaries)


@neatlogs.span(kind="AGENT", name="analyst", role="Investment Analyst", goal="Identify investment themes and risks")
def analyst_agent(company: str, findings: str) -> str:
    with neatlogs.trace("analyze_findings", kind="LLM",
                        prompt_template=_analyst_sys,
                        user_prompt_template=_analyst_user):
        msgs = _analyst_sys.compile() + _analyst_user.compile(
            company=company, findings=findings
        )
        stream = client.chat.completions.create(
            model=DEPLOYMENT,
            messages=msgs,
            stream=True,
            stream_options={"include_usage": True},
        )
        full = ""
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                text = chunk.choices[0].delta.content
                print(text, end="", flush=True)
                full += text
        print("\n")
    return full


@neatlogs.span(kind="AGENT", name="reporter", role="Report Writer", goal="Write the final investment brief")
def reporter_agent(company: str, analysis: str) -> str:
    with neatlogs.trace("write_report", kind="LLM",
                        prompt_template=_reporter_sys,
                        user_prompt_template=_reporter_user):
        msgs = _reporter_sys.compile() + _reporter_user.compile(
            company=company, analysis=analysis
        )
        stream = client.chat.completions.create(
            model=DEPLOYMENT,
            messages=msgs,
            stream=True,
            stream_options={"include_usage": True},
        )
        full = ""
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                text = chunk.choices[0].delta.content
                print(text, end="", flush=True)
                full += text
        print("\n")
    return full
