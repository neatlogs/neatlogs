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

Built on OpenTelemetry + OpenInference standards.

---

## Installation

Base install — includes lightweight OpenInference instrumentation adapters for all 40+ supported libraries (thin wrappers that do **not** pull in heavy LLM/framework dependencies):

```bash
pip install neatlogs
```

**Optional extras** install the actual underlying LLM/framework libraries:

```bash
pip install neatlogs[openai]==1.2.7
pip install neatlogs[anthropic]==1.2.7
pip install neatlogs[google-genai]==1.2.7
pip install neatlogs[langchain]==1.2.7
pip install neatlogs[langchain,langgraph]==1.2.7
pip install neatlogs[crewai]==1.2.7
pip install neatlogs[crewai,google-genai,litellm,azure-ai-inference]==1.2.7
```

Combine multiple extras with commas: `pip install neatlogs[crewai,google-genai,litellm]==1.2.7`

Full list of available extras: `openai`, `anthropic`, `langchain`, `langgraph`, `crewai`, `litellm`, `google-genai`, `google-adk`, `bedrock`, `groq`, `agno`, `dspy`, `openai-agents`, `guardrails`, `haystack`, `instructor`, `mcp`, `mistralai`, `portkey`, `pydantic-ai`, `smolagents`, `vertexai`, `autogen-agentchat`, `milvus`, `llama-index`, `azure-ai-inference`

Requires Python >= 3.10, < 3.14. Notable version pins: `crewai >= 1.9.3`, `qdrant-client < 1.16` (langchain extra).

---

## Core Principles

1. **Import order matters**: `neatlogs.init()` MUST be called **before** importing any LLM libraries (OpenAI, Anthropic, etc.) for auto-instrumentation patching to work.
2. **Always end scripts** with `neatlogs.flush()` then `neatlogs.shutdown()`.
3. **Use `@span` decorators** for custom code; use `trace()` context manager only for prompt template tracking, session management, or span kinds not supported by `@span` (`RERANKER`, `VECTOR_STORE`).
4. **Prefer auto-instrumentation** (`instrumentations=["openai"]`) over manual wrapping when possible.
5. **Read reference docs** before implementing — NeatLogs updates frequently.

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
   - `trace()` for prompt template tracking or span kinds not available in `@span` (`RERANKER`, `VECTOR_STORE`)
3. **Init**: Add `neatlogs.init()` **BEFORE** any LLM library imports with the correct `instrumentations` list.
4. **Verify**: Enable `debug=True` and check stderr output, or check the NeatLogs dashboard.

---

## `neatlogs.init()` Reference

| Parameter | Type | Default | Description |
|---|---|---|---|
| `api_key` | `str` | `None` | API key (or set `NEATLOGS_API_KEY` env var) |
| `endpoint` | `str` | `"https://staging-cloud.neatlogs.com/api/data/v4/batch"` | Backend endpoint URL |
| `workflow_name` | `str` | `None` | Name for this workflow/application |
| `instrumentations` | `list[str]` | `None` | Libraries to auto-instrument (e.g. `["openai", "langchain"]`) |
| `tags` | `list[str]` | `None` | Tags for filtering in dashboard |
| `user_id` | `str` | `None` | User identifier for trace attribution |
| `auto_session` | `bool` | `False` | Auto-generate session IDs for multi-turn conversations |
| `session_id` | `str` | `None` | Explicit session ID (overrides `auto_session`) |
| `sample_rate` | `float` | `1.0` | Sampling rate (0.0 to 1.0) |
| `flush_interval` | `float` | `5.0` | Seconds between batch flushes |
| `batch_size` | `int` | `100` | Max spans per batch |
| `debug` | `bool` | `False` | Enable verbose logging to stderr |
| `log_level` | `str` | `"INFO"` | Stdlib logging level for auto-capture. Captures `logging.info()`, `logging.warning()`, `logging.error()` as LOG spans inside `@span` or `trace()` blocks |
| `capture_logs` | `bool` | `False` | Enable stdlib logging auto-capture |
| `disable_export` | `bool` | `False` | Disable span export (for local testing) |
| `pii_enabled` | `bool` | `False` | Enable server-side PII redaction |
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
| `google_genai` | Google Generative AI | Client must be created **after** `init()` — see troubleshooting |
| `azure_ai_inference` | Azure AI Inference | |
| `litellm` | LiteLLM | |
| `bedrock` | AWS Bedrock | |
| `groq` | Groq | |
| `vertexai` | Google Vertex AI | |
| `mistralai` | Mistral AI | |
| `portkey` | Portkey | |
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
| ⚠️ `langgraph` | LangGraph | No direct instrumentor. Use `instrumentations=["langchain"]` — LangGraph is built on LangChain and is traced via the LangChain instrumentor |

### Retrieval / Vector Stores

> **Note**: The vector store libraries below are registered in the SDK but do not currently have a NeatLogs or OpenInference instrumentor. Passing them to `instrumentations=[]` will be silently skipped. Use `trace("op", kind="VECTOR_STORE")` with manual attributes for custom vector DB spans, or rely on higher-level framework instrumentation (e.g. LangChain retriever auto-instrumentation).

| Key | Library | Status |
|---|---|---|
| `weaviate` | Weaviate | ✅ Has OpenInference instrumentor |
| `chromadb` | ChromaDB | ⚠️ No direct instrumentor — traced indirectly via LangChain retriever instrumentation |
| `pinecone` | Pinecone | ⚠️ No direct instrumentor |
| `qdrant` | Qdrant | ⚠️ No direct instrumentor |
| `milvus` | Milvus | ⚠️ No direct instrumentor |
| `elasticsearch` | Elasticsearch | ⚠️ No direct instrumentor |
| `redis` | Redis | ⚠️ No direct instrumentor |
| `marqo` | Marqo | ⚠️ No direct instrumentor |

### Other

| Key | Library | Notes |
|---|---|---|
| `mcp` | Model Context Protocol | |
| `instructor` | Instructor | |
| `guardrails` | Guardrails AI | |
| `google_adk` | Google ADK | |
| `promptflow` | PromptFlow | |

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
| `NEATLOGS_ENDPOINT` | Backend endpoint URL |
| `NEATLOGS_DISABLE_EXPORT` | Set to `"true"` to disable span export |
| `NEATLOGS_LOG_SPANS` | Set to `"true"` to log spans to file |
| `NEATLOGS_LOG_SPANS_FILE` | File path for span logs (default: `spans.log`) |
| `NEATLOGS_LOG_RAW_SPANS` | Set to `"true"` to log raw span JSON |
| `NEATLOGS_LOG_RAW_SPANS_FILE` | File path for raw span logs |
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
    pii_enabled=True,
    pii_span_types=["LLM", "TOOL"],
)
```

---

## Documentation

Full documentation: [https://docs.neatlogs.com/](https://docs.neatlogs.com/)
