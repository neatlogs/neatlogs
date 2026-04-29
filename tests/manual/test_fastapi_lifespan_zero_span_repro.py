"""
FastAPI lifespan reproduction for 0-span rows from non-AI HTTP traffic.

This mirrors the customer setup:
- FastAPI app uses a lifespan context manager.
- neatlogs.init() runs inside lifespan startup.
- Startup code performs an outgoing HTTP call after init.
- A normal non-AI API endpoint performs an outgoing HTTP call.
- A chat endpoint runs an AI-like workflow wrapped with NeatLogs decorators.

Important behavior being tested:
- NeatLogs does NOT auto-instrument inbound FastAPI/ASGI server request spans.
- NeatLogs DOES always instrument outgoing HTTP clients (requests/httpx/urllib3/aiohttp)
  after init.
- If the dashboard shows 0-span rows for startup/non-AI HTTP calls, the row is created
  from HTTP-only/non-AI traces, not because init is inside lifespan.

Run server:
    NEATLOGS_API_KEY=<dev-key> uvicorn tests.manual.test_fastapi_lifespan_zero_span_repro:app --reload --port 8088

In another terminal:
    curl http://127.0.0.1:8088/health
    curl http://127.0.0.1:8088/non-ai
    curl 'http://127.0.0.1:8088/chat?q=hello'

Then stop uvicorn with Ctrl+C so lifespan shutdown flushes spans.

Expected dashboard check:
- If rows appear for workflow "fastapi-lifespan-zero-span-repro" with 0 spans after
  startup or /non-ai, backend/UI is creating rows for outgoing HTTP-only traffic.
- /chat should create a meaningful trace with WORKFLOW/AGENT/LLM-like spans because it
  uses NeatLogs decorators/trace around the AI workflow.
"""

from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager

import requests
from fastapi import FastAPI

import neatlogs


@neatlogs.span(kind="AGENT", name="chat_agent", role="assistant")
def run_agent_step(query: str) -> str:
    with neatlogs.trace("mock_llm_call", kind="LLM") as span:
        span.set_attribute("llm.model_name", "mock-model")
        span.set_attribute("llm.input", query)
        answer = f"mock answer for: {query}"
        span.set_attribute("llm.output", answer)
        return answer


@neatlogs.span(kind="WORKFLOW", name="chat_workflow")
def run_chat_workflow(query: str) -> str:
    return run_agent_step(query)


@asynccontextmanager
async def lifespan(app: FastAPI):
    neatlogs.init(
        api_key=None,  # reads NEATLOGS_API_KEY from env
        endpoint=os.environ.get("NEATLOGS_ENDPOINT", "https://staging-cloud.neatlogs.com"),
        workflow_name="fastapi-lifespan-zero-span-repro",
        instrumentations=[],
    )

    # Simulate customer startup work that calls external services after NeatLogs init.
    requests.get("https://httpbin.org/status/204", timeout=10)

    try:
        yield
    finally:
        await asyncio.to_thread(neatlogs.flush)
        await asyncio.to_thread(neatlogs.shutdown)


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/non-ai")
def non_ai() -> dict[str, int]:
    response = requests.get("https://httpbin.org/status/204", timeout=10)
    return {"status": response.status_code}


@app.get("/chat")
def chat(q: str = "hello") -> dict[str, str]:
    return {"answer": run_chat_workflow(q)}
