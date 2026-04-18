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
`init()`, `flush()`, `shutdown()`, `@span()`, `trace()`, and `PromptTemplate`.

---

## Installation

Base install — includes lightweight OpenInference instrumentation adapters for all 45+ supported libraries (thin wrappers that do **not** pull in heavy LLM/framework dependencies):

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

Full list of available extras: `openai`, `anthropic`, `langchain`, `langgraph`, `crewai`, `litellm`, `google-genai`, `google-adk`, `bedrock`, `groq`, `agno`, `dspy`, `openai-agents`, `guardrails`, `haystack`, `instructor`, `mcp`, `mistralai`, `portkey`, `pydantic-ai`, `smolagents`, `vertexai`, `autogen-agentchat`, `milvus`, `llama-index`

Requires Python >= 3.10, < 3.14. Notable version pins: `crewai >= 1.9.3`, `qdrant-client < 1.16` (langchain extra).

---

## Core Principles

1. **Import order matters**: `neatlogs.init()` MUST be called **before** importing any LLM libraries (OpenAI, Anthropic, etc.) for auto-instrumentation patching to work.
2. **Scripts**: end with `neatlogs.flush()` then `neatlogs.shutdown()`. **Servers**: call `init()` once at startup; do NOT call `flush()` or `shutdown()` on every request — see [Long-Running Servers](#long-running-servers-fastapi-flask-django) below.
3. **Use `@span` decorators** for custom code; use `trace()` context manager for prompt template tracking or span kinds not supported by `@span` (`RERANKER`, `VECTOR_STORE`, `LLM`).
4. **Prefer auto-instrumentation** (`instrumentations=["openai"]`) over manual wrapping when possible.
5. **Init is single-shot**: `neatlogs.init()` configures the global telemetry provider. Calling it a second time raises `ValueError`. If you need to reinitialize, call `neatlogs.shutdown()` first (rare).
6. **Read reference docs** before implementing — NeatLogs updates frequently.

---

## Long-Running Servers (FastAPI, Flask, Django)

For server applications, `neatlogs.init()` is called **once at startup**. Do NOT call `flush()` or `shutdown()` on every request — spans batch automatically every `flush_interval` (default 5 seconds).

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
async def ask(q: str):
    # Auto-instrumented — spans are batched and exported automatically
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
| `endpoint` | `str` | `"https://staging-cloud.neatlogs.com/api/data/v4/batch"` | Backend endpoint URL |
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
| `pii_span_types` | `list[str]` | `None` | Span types for PII redaction (e.g. `["LLM", "TOOL"]`) |
| `mask` | `callable` | `None` | Client-side mask function `(span_dict) -> span_dict` |

---

## Supported Instrumentations

Pass these string values in the `instrumentations=[]` list to `neatlogs.init()`.

> **How it works**: `_instrument_dual()` tries a NeatLogs custom instrumentor first, then OpenInference. Libraries that have neither are silently skipped. Keys marked ⚠️ below have no direct instrumentor — use the noted alternative instead.

### LLM Providers

| Key | Library | Notes |
|---|---|---|
| `openai` | OpenAI | |
| `anthropic` | Anthropic | |
| `google_genai` | Google Generative AI (`google.genai`) | Client must be created **after** `init()` — see troubleshooting. Preferred key for the `google-genai` SDK |
| `google_generativeai` | Google Generative AI (`google.generativeai`) | For the older `google-generativeai` SDK. Has OpenLLMetry instrumentor |
| `azure_ai_inference` | Azure AI Inference | |
| `litellm` | LiteLLM | |
| `bedrock` | AWS Bedrock | |
| `groq` | Groq | |
| `vertexai` | Google Vertex AI | |
| `mistralai` | Mistral AI | |
| `portkey` | Portkey | |
| `watsonx` | IBM watsonx.ai | |
| `replicate` | Replicate | |
| `sagemaker` | AWS SageMaker | |
| `alephalpha` | Aleph Alpha | |
| ⚠️ `huggingface_hub` | Hugging Face Hub | No direct instrumentor — key is in the registry but silently skipped |
| ⚠️ `together` | Together AI | No direct instrumentor — use `litellm` as a proxy or call via OpenAI-compatible endpoint with `openai` key |
| ⚠️ `cohere` | Cohere | No direct instrumentor — use `litellm` as a proxy |
| ⚠️ `ollama` | Ollama | No direct instrumentor — call via OpenAI-compatible endpoint with `openai` key |

### Agent Frameworks

| Key | Framework | Notes |
|---|---|---|
| `langchain` | LangChain | Also covers LangGraph execution — see below |
| `crewai` | CrewAI | Auto-loads `litellm`; also add provider keys (e.g. `openai`) |
| `llamaindex` | LlamaIndex | |
| `autogen` | AutoGen | |
| `haystack` | Haystack | |
| `dspy` | DSPy | |
| `agno` | Agno | |
| `pydantic_ai` | Pydantic AI | |
| `openai_agents` | OpenAI Agents | |
| `smolagents` | SmolAgents | |
| `strands` | Strands | |
| `pipecat` | Pipecat | |
| `beeai` | BeeAI | |
| ⚠️ `langgraph` | LangGraph | No direct instrumentor. Use `instrumentations=["langchain"]` — LangGraph is built on LangChain and is traced via the LangChain instrumentor |

### Retrieval / Vector Stores

| Key | Library | Status |
|---|---|---|
| `weaviate` | Weaviate | ✅ Has OpenInference instrumentor — auto-instrumented when weaviate is installed |
| `chromadb` | ChromaDB | ⚠️ No direct instrumentor — traced indirectly via LangChain retriever instrumentation |
| `pinecone` | Pinecone | ⚠️ No direct instrumentor |
| `qdrant` | Qdrant | ⚠️ No direct instrumentor |
| `milvus` | Milvus | ⚠️ No direct instrumentor |
| `elasticsearch` | Elasticsearch | ⚠️ No direct instrumentor |
| `redis` | Redis | ⚠️ No direct instrumentor |
| `marqo` | Marqo | ⚠️ No direct instrumentor |
| `opensearch` | OpenSearch | ⚠️ No direct instrumentor |

> **Note**: Libraries marked ⚠️ above have no NeatLogs or OpenInference instrumentor — passing them to `instrumentations=[]` is silently skipped. Use `trace("op", kind="VECTOR_STORE")` with manual attributes for custom vector DB spans, or rely on higher-level framework instrumentation (e.g. LangChain retriever auto-instrumentation).

### Other

| Key | Library | Notes |
|---|---|---|
| `mcp` | Model Context Protocol | |
| `instructor` | Instructor | |
| `guardrails` | Guardrails AI | |
| `google_adk` | Google ADK | |
| `promptflow` | PromptFlow | ⚠️ No pip extra — `pip install openinference-instrumentation-promptflow` separately |

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
| `NEATLOGS_TRACE_CONTENT` | Set to `"false"` to globally disable input/output content capture on spans |

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

Enable automatic server-side redaction by setting `pii_enabled=True` and optionally scoping to specific span types:

```python
neatlogs.init(
    pii_enabled=True,          # Override team default — enable redaction for this project
    pii_span_types=["LLM", "TOOL"],  # Limit to specific span kinds; None = all kinds
)
```

---

## Documentation

Full documentation: [https://docs.neatlogs.com/](https://docs.neatlogs.com/)
