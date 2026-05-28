"""
Agent functions for the reasoning model / LLM params verification workflow.

Agents:
  - openai_reasoning_agent    — Azure OpenAI, non-streaming, reasoning_effort + max_completion_tokens
  - openai_full_params_agent  — Azure OpenAI, streaming, standard params
  - anthropic_thinking_agent  — claude-sonnet-4-6 via Bedrock, streaming, extended thinking
  - langchain_openai_agent    — LangChain AzureChatOpenAI, temperature + max_tokens
  - gemini_async_agent        — Gemini async streaming, temperature + maxOutputTokens + top_p

Each agent wraps its call in neatlogs.trace() with prompt templates so the
llm.invocation_parameters and token counts land on the captured span.
"""

import asyncio
import os

import anthropic
import neatlogs
from neatlogs import SystemPromptTemplate, UserPromptTemplate

from google import genai as google_genai
from google.genai import types as genai_types
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from openai import AzureOpenAI

openai_client = AzureOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version=os.getenv("OPENAI_API_VERSION", "2025-01-01-preview"),
)
anthropic_client = anthropic.AnthropicBedrock(
    aws_region=os.getenv("AWS_REGION", "us-west-1"),
)
gemini_client = google_genai.Client(api_key=os.environ["GEMINI_API_KEY"])

REASONING_DEPLOYMENT = os.environ["AZURE_REASONING_DEPLOYMENT"]
LLM_DEPLOYMENT = os.environ["AZURE_LLM_DEPLOYMENT"]
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

PROBLEM = (
    "A bat and a ball cost $1.10 in total. "
    "The bat costs $1.00 more than the ball. "
    "How much does the ball cost? Show your full reasoning step by step."
)

_reasoning_sys = SystemPromptTemplate(
    "You are a careful logical reasoner. Show all your work step by step."
)
_reasoning_user = UserPromptTemplate("{{problem}}")


@neatlogs.span(kind="AGENT", name="openai_reasoning_agent",
               role="Logical Reasoner", goal="Solve with deep chain-of-thought reasoning")
def openai_reasoning_agent(problem: str) -> str:
    with neatlogs.trace("reasoning_llm", kind="LLM",
                        prompt_template=_reasoning_sys,
                        user_prompt_template=_reasoning_user):
        system_text = _reasoning_sys.compile()
        user_text = _reasoning_user.compile(problem=problem)
        response = openai_client.chat.completions.create(
            model=REASONING_DEPLOYMENT,
            messages=[
                {"role": "system", "content": system_text},
                {"role": "user", "content": user_text},
            ],
            max_completion_tokens=16000,
        )
    return response.choices[0].message.content or ""


@neatlogs.span(kind="AGENT", name="openai_full_params_agent",
               role="Logical Reasoner", goal="Solve with full params")
def openai_full_params_agent(problem: str) -> str:
    with neatlogs.trace("full_params_llm", kind="LLM",
                        prompt_template=_reasoning_sys,
                        user_prompt_template=_reasoning_user):
        system_text = _reasoning_sys.compile()
        user_text = _reasoning_user.compile(problem=problem)
        stream = openai_client.chat.completions.create(
            model=LLM_DEPLOYMENT,
            messages=[
                {"role": "system", "content": system_text},
                {"role": "user", "content": user_text},
            ],
            max_completion_tokens=1000,
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


@neatlogs.span(kind="AGENT", name="anthropic_thinking_agent",
               role="Extended Thinker", goal="Solve using Anthropic extended thinking")
def anthropic_thinking_agent(problem: str) -> str:
    with neatlogs.trace("extended_thinking_llm", kind="LLM",
                        prompt_template=_reasoning_sys,
                        user_prompt_template=_reasoning_user):
        system_msg = _reasoning_sys.compile()
        user_msg = _reasoning_user.compile(problem=problem)
        full = ""
        with anthropic_client.messages.stream(
            model=os.getenv("BEDROCK_SONNET_MODEL", "us.anthropic.claude-sonnet-4-6"),
            max_tokens=16000,
            temperature=1,
            thinking={"type": "enabled", "budget_tokens": 10000},
            system=system_msg,
            messages=[{"role": "user", "content": user_msg}],
        ) as stream:
            for event in stream:
                if event.type == "content_block_delta" and event.delta.type == "text_delta":
                    print(event.delta.text, end="", flush=True)
                    full += event.delta.text
        print("\n")
    return full


@neatlogs.span(kind="AGENT", name="langchain_openai_agent",
               role="Logical Reasoner", goal="Solve via LangChain ChatOpenAI")
def langchain_openai_agent(problem: str) -> str:
    llm = AzureChatOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        azure_deployment=LLM_DEPLOYMENT,
        model=LLM_DEPLOYMENT,
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.getenv("OPENAI_API_VERSION", "2025-01-01-preview"),
        max_completion_tokens=500,
    )
    with neatlogs.trace("langchain_llm", kind="LLM",
                        prompt_template=_reasoning_sys,
                        user_prompt_template=_reasoning_user):
        system_text = _reasoning_sys.compile()
        _reasoning_user.compile(problem=problem)
        messages = [
            SystemMessage(content=system_text),
            HumanMessage(content=problem),
        ]
        response = llm.invoke(messages)
        print(f"\n{response.content}\n")
    return response.content


@neatlogs.span(kind="AGENT", name="gemini_async_agent",
               role="Logical Reasoner", goal="Solve via Gemini async streaming")
def gemini_async_agent(problem: str) -> str:
    async def _run() -> str:
        system_text = _reasoning_sys.compile()
        user_text = _reasoning_user.compile(problem=problem)
        contents = [{"role": "user", "parts": [{"text": user_text}]}]

        with neatlogs.trace("gemini_llm", kind="LLM",
                            prompt_template=_reasoning_sys,
                            user_prompt_template=_reasoning_user):
            full = ""
            async for chunk in await gemini_client.aio.models.generate_content_stream(
                model=GEMINI_MODEL,
                contents=contents,
                config=genai_types.GenerateContentConfig(
                    system_instruction=system_text,
                    temperature=0.7,
                    max_output_tokens=1000,
                    top_p=0.9,
                ),
            ):
                if chunk.text:
                    print(chunk.text, end="", flush=True)
                    full += chunk.text
            print("\n")
        return full

    return asyncio.run(_run())
