---
name: neatlogs
description: >
  NeatLogs is an AI agent debugging and observability platform. Use this skill when
  instrumenting Python LLM applications with neatlogs for tracing, monitoring, debugging,
  observability, decorators, spans, prompt template tracking, or auto-instrumentation of
  LLM providers and agent frameworks.
---

# NeatLogs SDK v3 — Agent Skill

NeatLogs auto-instruments LLM calls, agent frameworks, and custom code with just 6 exports:
`init()`, `flush()`, `shutdown()`, `@span()`, `trace()`, and `SystemPromptTemplate`.

---

## Installation

Base install — includes the SDK plus lightweight instrumentation adapters for the customer-facing integrations listed below. Optional extras install the actual LLM/framework libraries when needed:

```bash
pip install neatlogs
```

**Optional extras** install the actual underlying LLM/framework libraries:

```bash
pip install neatlogs[openai]
pip install neatlogs[anthropic]
pip install neatlogs[google-genai]
pip install neatlogs[langchain]
pip install neatlogs[langchain,langgraph]
pip install neatlogs[crewai]
pip install neatlogs[crewai,google-genai,litellm]
```

Combine multiple extras with commas: `pip install neatlogs[crewai,google-genai,litellm]`

Customer-facing tested extras: `openai`, `anthropic`, `langchain`, `langgraph`, `crewai`, `litellm`, `google-genai`, `mcp`

Requires Python >= 3.10, < 3.14. Notable version pins: `crewai >= 1.9.3`.

---

## Core Principles

1. **Import order matters**: `neatlogs.init()` MUST be called **before** importing any LLM libraries (OpenAI, Anthropic, etc.) for auto-instrumentation patching to work.
2. **Scripts**: end with `neatlogs.flush()` then `neatlogs.shutdown()`. **Servers**: call `init()` once at startup; do NOT call `flush()` or `shutdown()` on every request — see [Long-Running Servers](#long-running-servers-fastapi-flask-django) below.
3. **Use `@span` decorators** for custom code; use `trace()` context manager for prompt template tracking or span kinds not supported by `@span` (`RERANKER`, `VECTOR_STORE`, `LLM`).
4. **Prefer auto-instrumentation** (`instrumentations=["openai"]`) over manual wrapping when possible.
5. **Init is single-shot**: `neatlogs.init()` configures the global telemetry provider. Calling it again is a no-op (with a debug warning when `debug=True`). If you need to reinitialize, call `neatlogs.shutdown()` first (rare).
6. **Read reference docs** before implementing — NeatLogs updates frequently.

---

## Long-Running Servers (FastAPI, Flask, Django)

For server applications, `neatlogs.init()` is called **once at startup**. Do NOT call `flush()` or `shutdown()` on every request — spans batch automatically every `flush_interval` (default 5 seconds).

`neatlogs.init()` does **not** auto-instrument inbound FastAPI/ASGI server request spans. It does always instrument outgoing HTTP clients (`requests`, `httpx`, `urllib3`, `aiohttp`) for context propagation. Wrap AI endpoints in a NeatLogs `WORKFLOW` span so they appear with proper trace structure in the dashboard.

```python
import neatlogs
from fastapi import FastAPI
from contextlib import asynccontextmanager

neatlogs.init(api_key="...", workflow_name="my-api", instrumentations=["openai"])

from openai import OpenAI  # Import AFTER init()

client = OpenAI()

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    # Called once when the server shuts down — flush remaining spans
    import asyncio
    await asyncio.to_thread(neatlogs.flush)
    await asyncio.to_thread(neatlogs.shutdown)

app = FastAPI(lifespan=lifespan)

@app.get("/ask")
@neatlogs.span(kind="WORKFLOW", name="ask_workflow")
async def ask(q: str):
    # Auto-instrumented LLM call becomes a child of this WORKFLOW span
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": q}],
    )
    return {"answer": response.choices[0].message.content}
    # DO NOT call flush() here — it would flush on every request (performance issue)
```

For Flask/Django, call `neatlogs.flush()` and `neatlogs.shutdown()` via an `atexit` handler or framework shutdown hook. See [`references/troubleshooting.md` §6](references/troubleshooting.md#6-flushshutdown-gotcha) for the async gotcha.

---

## Quick Start

Complete minimal working example:

```python
import neatlogs

neatlogs.init(
    api_key="your-api-key",       # or set NEATLOGS_API_KEY env var
    workflow_name="my-app",
    instrumentations=["openai"],
)

# NOW import the LLM library (after init)
from openai import OpenAI

client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)

neatlogs.flush()
neatlogs.shutdown()
```

---

## Instrumentation Workflow

1. **Assess**: Detect what LLM providers/frameworks the project uses.
2. **Instrument**: Choose the correct approach:
   - Auto-instrumentation for providers → add to `instrumentations=[]`
   - `@span` decorators for custom orchestration code
   - `trace()` for prompt template tracking or span kinds not available in `@span` (`RERANKER`, `VECTOR_STORE`, `LLM`)
3. **Init**: Add `neatlogs.init()` **BEFORE** any LLM library imports with the correct `instrumentations` list.
4. **Verify**: Check the NeatLogs dashboard for incoming traces.

---

## `neatlogs.init()` Reference

| Parameter | Type | Default | Description |
|---|---|---|---|
| `api_key` | `str` | `None` | API key (or set `NEATLOGS_API_KEY` env var). If neither is set, spans are created locally but **silently not exported** — no error is raised |
| `endpoint` | `str` | `"https://staging-cloud.neatlogs.com"` | Backend base URL. Trace export is normalized to `{base_url}/v1/traces`; legacy `/api/data/v4/batch` inputs are accepted and rewritten |
| `workflow_name` | `str` | `None` | Name for this workflow/application |
| `instrumentations` | `list[str]` | `None` | Libraries to auto-instrument (e.g. `["openai", "langchain"]`) |
| `tags` | `list[str]` | `None` | Tags for filtering in dashboard |
| `user_id` | `str` | `None` | User identifier for trace attribution |
| `auto_session` | `bool` | `False` | Auto-generate a session ID on first use and reuse it for the process lifetime. Useful for chatbots/multi-turn conversations |
| `session_id` | `str` | `None` | Explicit session ID — overrides `auto_session`. Pass a per-user or per-conversation ID to group turns in the dashboard |
| `sample_rate` | `float` | `1.0` | Sampling rate (0.0 to 1.0) |
| `flush_interval` | `float` | `5.0` | Seconds between batch flushes |
| `batch_size` | `int` | `100` | Max spans per batch |
| `debug` | `bool` | `False` | Enable verbose logging to stderr |
| `pii_enabled` | `Optional[bool]` | `None` | Override the team-level server-side PII redaction setting. `True` = enable, `False` = disable, `None` (default) = use the team setting in the NeatLogs dashboard |
| `pii_span_types` | `Optional[list[str]]` | `None` | Override which span types have PII redaction applied. `None` = use team dashboard config |
| `capture_logs` | `bool` | `False` | Capture `neatlogs.log()`, stdlib `logging.*()`, and `print()` (via `capture_stdout=True`) as LOG spans. Required for log capture |
| `log_level` | `str` | `"INFO"` | Minimum Python logging level to auto-capture as LOG spans when `capture_logs=True` |
| `mask` | `callable` | `None` | Client-side mask function `(span_dict) -> span_dict` |

---

## Supported Instrumentations

Pass these string values in the `instrumentations=[]` list to `neatlogs.init()`.

### LLM Providers

| Key | Library | Notes |
|---|---|---|
| `openai` | OpenAI | Tested |
| `anthropic` | Anthropic | Tested |
| `google_genai` | Google Generative AI (`google.genai`) | Tested. Client must be created **after** `init()` — see troubleshooting |
| `azure_ai_inference` | Azure AI Inference | For Azure OpenAI / Azure AI models |
| `bedrock` | AWS Bedrock | `boto3>=1.42.11` |
| `litellm` | LiteLLM | Tested |

### Agent Frameworks

| Key | Framework | Notes |
|---|---|---|
| `langchain` | LangChain | Tested. Also covers LangGraph execution — see below |
| `crewai` | CrewAI | Tested. Auto-loads `litellm`. If the CrewAI LLM is backed by a direct provider SDK, also add that provider key: Azure OpenAI / Azure AI Inference → `azure_ai_inference`, OpenAI → `openai`, Google GenAI → `google_genai`, Anthropic → `anthropic` |
| ⚠️ `langgraph` | LangGraph | Tested via LangChain. No direct instrumentor; use `instrumentations=["langchain"]` |

### Vector Databases

| Key | Library | Notes |
|---|---|---|
| `chromadb` | ChromaDB | Auto-instrumented via OpenLLMetry |
| `pinecone` | Pinecone | Auto-instrumented via OpenLLMetry |
| `qdrant` | Qdrant | Auto-instrumented via OpenLLMetry |
| `weaviate` | Weaviate | Auto-instrumented via OpenLLMetry/OpenInference |
| `milvus` | Milvus | `pymilvus>=2.4.0,<2.5.0` |
| `opensearch` | OpenSearch | Auto-instrumented |
| `elasticsearch` | Elasticsearch | Auto-instrumented |
| `redis` | Redis | Auto-instrumented |
| `marqo` | Marqo | Auto-instrumented |

> **Tip**: If you use LangChain retrievers, add `"langchain"` to `instrumentations=[]` — retriever spans are captured automatically via the LangChain instrumentor.

### Other

| Key | Library | Notes |
|---|---|---|
| `mcp` | Model Context Protocol | Tested |
| `instructor` | Instructor | Structured output library |
| `guardrails` | Guardrails AI | Safety/validation framework |

> **HTTP libraries** (`requests`, `httpx`, `urllib3`, `aiohttp`) are always auto-instrumented by `neatlogs.init()` for trace context propagation — you do not need to list them in `instrumentations=[]`.

---

## Reference Docs

For deep dives, see the companion reference files:

- **Custom instrumentation** with decorators and traces → [`references/decorators-and-traces.md`](references/decorators-and-traces.md)
- **Prompt template** tracking and management → [`references/prompt-templates.md`](references/prompt-templates.md)
- **Framework-specific** integration patterns → [`references/framework-integrations.md`](references/framework-integrations.md)
- **Troubleshooting** and common mistakes → [`references/troubleshooting.md`](references/troubleshooting.md)

---

## Environment Variables

| Variable | Description |
|---|---|
| `NEATLOGS_API_KEY` | API key (alternative to `api_key` param) |
| `NEATLOGS_DISABLE_EXPORT` | Set to `"true"` to disable span export |

---

## Data Masking and PII

NeatLogs supports both client-side and server-side PII redaction.

### Client-Side Masking

Provide a `mask` callback to `init()` to redact sensitive data before spans leave the process. You can also pass `mask=fn` per-span via `@span(mask=fn)` or `trace(mask=fn)`.

```python
def redact_pii(span):
    attrs = span.get("attributes", {})
    for key in list(attrs):
        if "email" in key or "password" in key:
            attrs[key] = "[REDACTED]"
    return span

neatlogs.init(mask=redact_pii)
```

### Server-Side PII Redaction

Enable automatic server-side redaction by setting `pii_enabled=True`:

```python
neatlogs.init(
    pii_enabled=True,
)
```

---

## Documentation

Full documentation: [https://docs.neatlogs.com/](https://docs.neatlogs.com/)
