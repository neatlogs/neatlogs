"""
Agent functions for the Google GenAI blog post creation workflow.

Agents:
  - Ideation   (gemini-2.5-flash, non-streaming)           — returns 5 content ideas as JSON
  - Writer     (gemini-2.5-flash, tool-calling + streaming) — researches facts via web_search,
                                                               then drafts the full post
  - Editor     (gemini-2.5-flash, streaming)               — rewrites weak sections, adds examples
  - Finalizer  (gemini-2.5-flash, non-streaming)           — SEO polish and final formatting
"""

import json

import neatlogs
from neatlogs import PromptTemplate, UserPromptTemplate
from google import genai
from google.genai import types
from duckduckgo_search import DDGS

client = genai.Client()
MODEL = "gemini-2.5-flash"

# ---------------------------------------------------------------------------
# Tool implementation — called only when the LLM requests it
# ---------------------------------------------------------------------------

@neatlogs.span(kind="TOOL", name="web_search", description="DuckDuckGo web search for supporting facts")
def web_search(query: str) -> str:
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=3):
            results.append(f"- {r['title']}: {r['body']}")
    return "\n".join(results) if results else "No results found."


# Google GenAI function declaration (passed to the LLM)
_SEARCH_TOOL_DEF = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="web_search",
            description="Search the web for current facts, statistics, and examples to support the blog post.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "query": types.Schema(type="STRING", description="The search query."),
                },
                required=["query"],
            ),
        )
    ]
)

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_ideation_sys = PromptTemplate([{
    "role": "system",
    "content": "You are a creative content strategist. Return exactly 5 blog post ideas as a JSON array of objects with 'title' and 'hook' fields. No other text.",
}])
_ideation_user = UserPromptTemplate([{"role": "user", "content": "Topic: {{topic}}"}])

_research_sys = PromptTemplate([{
    "role": "system",
    "content": "You are a research assistant. Use the web_search tool to find 2-3 relevant facts or statistics for the blog post topic. Call the tool, then summarize the findings.",
}])
_research_user = UserPromptTemplate([{
    "role": "user",
    "content": "Find supporting facts for a blog post titled '{{title}}' about {{topic}}.",
}])

_writer_sys = PromptTemplate([{
    "role": "system",
    "content": "You are an expert blog writer. Write an engaging, well-structured blog post with an introduction, 3-4 main sections, and a conclusion. Use markdown.",
}])
_writer_user = UserPromptTemplate([{
    "role": "user",
    "content": "Topic: {{topic}}\nTitle: {{title}}\nHook: {{hook}}\n\nSupporting facts:\n{{facts}}\n\nWrite a complete blog post.",
}])

_editor_sys = PromptTemplate([{
    "role": "system",
    "content": "You are a sharp content editor. Improve the draft by strengthening weak sections, adding concrete examples, and improving clarity. Return the full revised post in markdown.",
}])
_editor_user = UserPromptTemplate([{
    "role": "user",
    "content": "Topic: {{topic}}\n\nDraft:\n{{draft}}\n\nRevise and improve this post.",
}])

_finalizer_sys = PromptTemplate([{
    "role": "system",
    "content": "You are an SEO and content specialist. Polish the post: add a meta description, improve headings for SEO, ensure consistent tone, and format cleanly in markdown.",
}])
_finalizer_user = UserPromptTemplate([{
    "role": "user",
    "content": "Topic: {{topic}}\n\nEdited post:\n{{edited}}\n\nProduce the final polished version.",
}])

# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------

@neatlogs.span(kind="AGENT", name="ideation", role="Content Strategist", goal="Generate blog post ideas")
def ideation_agent(topic: str) -> dict:
    with neatlogs.trace("generate_ideas", kind="LLM", prompt_template=_ideation_sys,
                        user_prompt_template=_ideation_user):
        system_prompt = _ideation_sys.compile()[0]["content"]
        user_prompt = _ideation_user.compile(topic=topic)[0]["content"]
        response = client.models.generate_content(
            model=MODEL,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.8,
            ),
        )
        raw = response.text.strip()
    try:
        ideas = json.loads(raw)
    except json.JSONDecodeError:
        ideas = [{"title": topic, "hook": "Explore this topic in depth."}]
    return ideas[0] if ideas else {"title": topic, "hook": ""}


@neatlogs.span(kind="AGENT", name="writer", role="Blog Writer", goal="Research facts and draft the full blog post")
def writer_agent(topic: str, idea: dict) -> str:
    title = idea.get("title", topic)
    hook = idea.get("hook", "")

    # Step 1: Research step — LLM calls web_search via function calling
    facts = ""
    with neatlogs.trace("research_facts", kind="LLM", prompt_template=_research_sys,
                        user_prompt_template=_research_user):
        system_prompt = _research_sys.compile()[0]["content"]
        user_prompt = _research_user.compile(title=title, topic=topic)[0]["content"]
        response = client.models.generate_content(
            model=MODEL,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                tools=[_SEARCH_TOOL_DEF],
                temperature=0,
            ),
        )
        # Execute any tool calls the model requested
        contents = [user_prompt]
        for part in response.candidates[0].content.parts:
            if part.function_call:
                search_result = web_search(part.function_call.args["query"])
                contents = [
                    types.Content(role="user", parts=[types.Part(text=user_prompt)]),
                    response.candidates[0].content,
                    types.Content(
                        role="user",
                        parts=[types.Part(
                            function_response=types.FunctionResponse(
                                name="web_search",
                                response={"result": search_result},
                            )
                        )],
                    ),
                ]
                # Second call — model summarizes tool results
                summary_resp = client.models.generate_content(
                    model=MODEL,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        temperature=0,
                    ),
                )
                facts = summary_resp.text or ""
                break
        else:
            facts = response.text or ""

    # Step 2: Write the draft using the researched facts (streaming)
    with neatlogs.trace("write_draft", kind="LLM", prompt_template=_writer_sys,
                        user_prompt_template=_writer_user):
        system_prompt = _writer_sys.compile()[0]["content"]
        user_prompt = _writer_user.compile(
            topic=topic, title=title, hook=hook, facts=facts
        )[0]["content"]
        print("\n--- Writer (streaming) ---")
        full = ""
        for chunk in client.models.generate_content_stream(
            model=MODEL,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.7,
            ),
        ):
            if chunk.text:
                print(chunk.text, end="", flush=True)
                full += chunk.text
        print("\n-------------------------\n")
    return full


@neatlogs.span(kind="AGENT", name="editor", role="Content Editor", goal="Improve and enrich the draft")
def editor_agent(topic: str, draft: str) -> str:
    with neatlogs.trace("edit_draft", kind="LLM", prompt_template=_editor_sys,
                        user_prompt_template=_editor_user):
        system_prompt = _editor_sys.compile()[0]["content"]
        user_prompt = _editor_user.compile(topic=topic, draft=draft)[0]["content"]
        print("\n--- Editor (streaming) ---")
        full = ""
        for chunk in client.models.generate_content_stream(
            model=MODEL,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.5,
            ),
        ):
            if chunk.text:
                print(chunk.text, end="", flush=True)
                full += chunk.text
        print("\n--------------------------\n")
    return full


@neatlogs.span(kind="AGENT", name="finalizer", role="SEO Specialist", goal="Polish and format the final post")
def finalizer_agent(topic: str, edited: str) -> str:
    with neatlogs.trace("finalize_post", kind="LLM", prompt_template=_finalizer_sys,
                        user_prompt_template=_finalizer_user):
        system_prompt = _finalizer_sys.compile()[0]["content"]
        user_prompt = _finalizer_user.compile(topic=topic, edited=edited)[0]["content"]
        response = client.models.generate_content(
            model=MODEL,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.3,
            ),
        )
    return response.text
