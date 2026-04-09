"""
Agent functions for the reasoning model / LLM params verification workflow.

Agents:
  - openai_reasoning_agent    — Azure OpenAI (AZURE_LLM_DEPLOYMENT), non-streaming, reasoning_effort + max_completion_tokens
  - openai_full_params_agent  — Azure OpenAI (AZURE_LLM_DEPLOYMENT), streaming, all 6 standard params
  - anthropic_thinking_agent  — claude-sonnet-4-6 via Bedrock, streaming, extended thinking
  - langchain_openai_agent    — LangChain AzureChatOpenAI (AZURE_LLM_DEPLOYMENT), temperature + max_tokens
  - gemini_async_agent        — google-genai gemini-2.0-flash, async streaming, temperature + maxOutputTokens

Each agent wraps its call in neatlogs.trace() with prompt templates so that
llm.invocation_parameters and token counts appear in the captured spans.
"""

import asyncio
import os

import anthropic
import neatlogs
from google import genai as google_genai
from google.genai import types as genai_types
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from neatlogs import PromptTemplate, UserPromptTemplate
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

REASONING_DEPLOYMENT = os.environ["AZURE_REASONING_DEPLOYMENT"]  # high max_completion_tokens
LLM_DEPLOYMENT = os.environ["AZURE_LLM_DEPLOYMENT"]              # temperature, top_p, etc.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

PROBLEM = (
    "A bat and a ball cost $1.10 in total. "
    "The bat costs $1.00 more than the ball. "
    "How much does the ball cost? Show your full reasoning step by step."
)

# ---------------------------------------------------------------------------
# Shared prompt templates
# ---------------------------------------------------------------------------

_reasoning_sys = PromptTemplate([{
    "role": "system",
    "content": "You are a careful logical reasoner. Show all your work step by step.",
}])
_reasoning_user = UserPromptTemplate([{"role": "user", "content": "{{problem}}"}])

# ---------------------------------------------------------------------------
# Agent 1: o4-mini — non-streaming, reasoning_effort + max_completion_tokens
# ---------------------------------------------------------------------------

@neatlogs.span(kind="AGENT", name="openai_reasoning_agent",
               role="Logical Reasoner", goal="Solve with deep chain-of-thought reasoning")
def openai_reasoning_agent(problem: str) -> str:
    with neatlogs.trace("o4_mini_reasoning", kind="LLM", prompt_template=_reasoning_sys,
                        user_prompt_template=_reasoning_user):
        system_msgs = _reasoning_sys.compile()
        user_msgs = _reasoning_user.compile(problem=problem)
        response = openai_client.chat.completions.create(
            model=REASONING_DEPLOYMENT,
            messages=system_msgs + user_msgs,
            max_completion_tokens=16000,    # → llm.invocation_parameters
        )
        reasoning_tokens = 0
        if response.usage and response.usage.completion_tokens_details:
            reasoning_tokens = response.usage.completion_tokens_details.reasoning_tokens or 0
        print(f"  reasoning_tokens={reasoning_tokens}  completion_tokens={response.usage.completion_tokens}")
    return response.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Agent 2: gpt-4o — streaming, all standard params
# ---------------------------------------------------------------------------

@neatlogs.span(kind="AGENT", name="openai_full_params_agent",
               role="Logical Reasoner", goal="Solve with all LLM params explicitly set")
def openai_full_params_agent(problem: str) -> str:
    with neatlogs.trace("gpt4o_full_params", kind="LLM", prompt_template=_reasoning_sys,
                        user_prompt_template=_reasoning_user):
        system_msgs = _reasoning_sys.compile()
        user_msgs = _reasoning_user.compile(problem=problem)
        stream = openai_client.chat.completions.create(
            model=LLM_DEPLOYMENT,
            messages=system_msgs + user_msgs,
            # presence_penalty=0.1,        # unsupported by gpt-5-nano
            # frequency_penalty=0.1,       # unsupported by gpt-5-nano
            # seed=42,                     # unsupported by gpt-5-nano
            max_completion_tokens=1000,   # → llm.invocation_parameters
            stream=True,
        )
        print(f"\n--- {LLM_DEPLOYMENT} (streaming) ---")
        full = ""
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                text = chunk.choices[0].delta.content
                print(text, end="", flush=True)
                full += text
        print("\n-------------------------\n")
    return full


# ---------------------------------------------------------------------------
# Agent 3: claude-sonnet-4-6 — streaming, extended thinking
# ---------------------------------------------------------------------------

@neatlogs.span(kind="AGENT", name="anthropic_thinking_agent",
               role="Extended Thinker", goal="Solve using Anthropic extended thinking")
def anthropic_thinking_agent(problem: str) -> str:
    with neatlogs.trace("claude_extended_thinking", kind="LLM", prompt_template=_reasoning_sys,
                        user_prompt_template=_reasoning_user):
        system_msg = _reasoning_sys.compile()[0]["content"]
        user_msg = _reasoning_user.compile(problem=problem)[0]["content"]
        print("\n--- claude-sonnet-4-6 extended thinking (streaming) ---")
        full = ""
        with anthropic_client.messages.stream(
            model=os.getenv("BEDROCK_SONNET_MODEL", "us.anthropic.claude-sonnet-4-6"),
            max_tokens=16000,
            temperature=1,            # required for extended thinking
            thinking={"type": "enabled", "budget_tokens": 10000},  # → llm.invocation_parameters
            system=system_msg,
            messages=[{"role": "user", "content": user_msg}],
        ) as stream:
            for event in stream:
                if event.type == "content_block_delta" and event.delta.type == "text_delta":
                    print(event.delta.text, end="", flush=True)
                    full += event.delta.text
        print("\n------------------------------------------------------\n")
    return full


# ---------------------------------------------------------------------------
# Agent 4: LangChain ChatOpenAI — temperature + max_tokens
# ---------------------------------------------------------------------------

@neatlogs.span(kind="AGENT", name="langchain_openai_agent",
               role="Logical Reasoner", goal="Solve using LangChain ChatOpenAI with explicit params")
def langchain_openai_agent(problem: str) -> str:
    llm = AzureChatOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        azure_deployment=LLM_DEPLOYMENT,
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.getenv("OPENAI_API_VERSION", "2025-01-01-preview"),
        max_completion_tokens=500, # → llm.invocation_parameters
    )
    with neatlogs.trace("langchain_azure_openai", kind="LLM", prompt_template=_reasoning_sys,
                        user_prompt_template=_reasoning_user):
        system_text = _reasoning_sys.compile()[0]["content"]
        messages = [
            SystemMessage(content=system_text),
            HumanMessage(content=problem),
        ]
        response = llm.invoke(messages)
        result = response.content
        print(f"\n--- LangChain AzureChatOpenAI ({LLM_DEPLOYMENT}) ---\n{result}\n----------------------------\n")
    return result


# ---------------------------------------------------------------------------
# Agent 5: Gemini async streaming — mirrors the backend copilot
# ---------------------------------------------------------------------------

@neatlogs.span(kind="AGENT", name="gemini_async_agent",
               role="Logical Reasoner", goal="Solve using Gemini async streaming with temperature + maxOutputTokens")
def gemini_async_agent(problem: str) -> str:
    async def _run() -> str:
        system_text = _reasoning_sys.compile()[0]["content"]
        user_text = _reasoning_user.compile(problem=problem)[0]["content"]

        contents = [{"role": "user", "parts": [{"text": user_text}]}]

        with neatlogs.trace("gemini_flash_streaming", kind="LLM", prompt_template=_reasoning_sys,
                            user_prompt_template=_reasoning_user):
            full = ""
            print(f"\n--- {GEMINI_MODEL} (async streaming) ---")
            async for chunk in await gemini_client.aio.models.generate_content_stream(
                model=GEMINI_MODEL,
                contents=contents,
                config=genai_types.GenerateContentConfig(
                    system_instruction=system_text,
                    temperature=0.7,          # → llm.invocation_parameters
                    max_output_tokens=1000,   # → llm.invocation_parameters
                    top_p=0.9,                # → llm.invocation_parameters
                ),
            ):
                if chunk.text:
                    print(chunk.text, end="", flush=True)
                    full += chunk.text
            print("\n-----------------------------------------\n")
        return full

    return asyncio.run(_run())
