---
name: neatlogs
description: >
  NeatLogs is an AI agent debugging and observability platform. Use this skill when
  instrumenting Python LLM applications with neatlogs for tracing, monitoring, debugging,
  observability, decorators, spans, prompt template tracking, or auto-instrumentation of
  LLM providers and agent frameworks.
---

# NeatLogs — Agent Skill

NeatLogs auto-instruments LLM calls, agent frameworks, and custom code. The small public API most integrations need:
`init()`, `flush()`, `shutdown()`, `@span()`, `trace()`, `SystemPromptTemplate`, `UserPromptTemplate`, `bind_templates()`, `register_crewai_task()`.

---

## Installation

```bash
pip install neatlogs
```

Optional extras install the actual underlying LLM / framework libraries:

```bash
pip install neatlogs[openai]
pip install neatlogs[anthropic]
pip install neatlogs[google-genai]
pip install neatlogs[langchain]
pip install neatlogs[langchain,langgraph]
pip install neatlogs[crewai]
pip install neatlogs[litellm]
pip install neatlogs[mcp]
```

Combine multiple extras with commas: `pip install neatlogs[crewai,google-genai]`

Requires Python >= 3.10, < 3.14. Notable version pins: `crewai >= 1.9.3`.

---

## Core Principles

1. **Import order matters**: `neatlogs.init()` MUST be called **before** importing any LLM libraries for auto-instrumentation patching to work.
2. **Scripts**: end with `neatlogs.flush()` then `neatlogs.shutdown()`. **Servers**: call `init()` once at startup; do NOT call `flush()` / `shutdown()` per request — see [Long-Running Servers](#long-running-servers) below.
3. **`@span` for custom code**, **`trace()` for prompt templates** — use `@span(kind="...")` to decorate orchestration functions, use `with neatlogs.trace(kind="LLM", system_prompt_template=..., user_prompt_template=...)` to attach prompt templates to LLM calls.
4. **Prefer auto-instrumentation** (`instrumentations=["openai"]`) over manual wrapping when a supported library is available.
5. **Init is single-shot**: `neatlogs.init()` configures the global telemetry provider. Calling it again is a no-op. If you need to reinitialize, call `neatlogs.shutdown()` first (rare).
6. **Read reference docs** before implementing — NeatLogs updates frequently.

---

## Quick Start

End-to-end example showing auto-instrumentation + `@span` decorators + `trace()` with prompt templates:

```python
import neatlogs
from neatlogs import SystemPromptTemplate, UserPromptTemplate

neatlogs.init(
    api_key="your-api-key",       # Get from https://app.neatlogs.com/settings/api-keys (or set NEATLOGS_API_KEY env var)
    workflow_name="my-app",
    instrumentations=["openai"],
)

# Import the LLM library AFTER init() so auto-instrumentation takes effect.
from openai import OpenAI

client = OpenAI()

sys_tpl = SystemPromptTemplate([
    {"role": "system", "content": "You are a helpful {{role}} assistant."}
])
user_tpl = UserPromptTemplate([
    {"role": "user", "content": "{{query}}"}
])


@neatlogs.span(kind="AGENT", name="researcher", role="Research Analyst")
def researcher(query: str) -> str:
    with neatlogs.trace(
        "llm_call",
        kind="LLM",
        system_prompt_template=sys_tpl,
        user_prompt_template=user_tpl,
    ):
        msgs = sys_tpl.compile(role="research") + user_tpl.compile(query=query)
        response = client.chat.completions.create(model="gpt-4o", messages=msgs)
    return response.choices[0].message.content


@neatlogs.span(kind="WORKFLOW")
def run(query: str) -> str:
    return researcher(query)


if __name__ == "__main__":
    print(run("Explain quantum computing briefly."))
    neatlogs.flush()
    neatlogs.shutdown()
```

This produces a trace with `WORKFLOW → AGENT → LLM` nesting, system + user prompt templates captured on the LLM span, and variable bindings visible in the UI.

---

## Long-Running Servers

For server applications, call `neatlogs.init()` **once at startup** and flush/shutdown **once at shutdown**. Spans batch automatically every `flush_interval` (default 5 s) — do not call `flush()` / `shutdown()` per request.

Wrap AI endpoints in a `@span(kind="WORKFLOW")` so each request appears as a root in the trace tree.

```python
import neatlogs
from fastapi import FastAPI
from contextlib import asynccontextmanager

neatlogs.init(
    api_key="...",  # Get from https://app.neatlogs.com/settings/api-keys (or set NEATLOGS_API_KEY env var)
    workflow_name="my-api",
    instrumentations=["openai"],
)

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
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": q}],
    )
    return {"answer": response.choices[0].message.content}
    # DO NOT call flush() here — would flush on every request (performance issue)
```

For non-FastAPI servers, hook `neatlogs.flush()` + `neatlogs.shutdown()` into the framework's shutdown event (or an `atexit` handler). See [`references/troubleshooting.md` §5](references/troubleshooting.md#5-flush--shutdown-gotcha) for the async gotcha.

---

## Instrumentation Workflow

1. **Assess**: Detect what LLM providers / frameworks the project uses.
2. **Instrument**: Pick the right approach:
   - **Auto-instrumentation** for LLM providers → add the key to `instrumentations=[]`
   - **`@span` decorators** for your own orchestration functions
   - **`trace()`** for prompt template tracking on LLM calls, and for direct-API spans (`RERANKER`, `VECTOR_STORE`, `LLM`) when you don't go through an instrumented SDK
3. **Init**: Add `neatlogs.init()` **BEFORE** any LLM library imports with the correct `instrumentations=[...]` list.
4. **Verify**: Check the NeatLogs dashboard for incoming traces. Use `debug=True` in `init()` to confirm each instrumentor loaded (prints `✅ Instrumented …` lines).

---

## `neatlogs.init()` Reference

| Parameter | Type | Default | Description |
|---|---|---|---|
| `api_key` | `str` | `None` | API key (or set `NEATLOGS_API_KEY` env var). If neither is set, spans are created locally but **silently not exported** — no error is raised |
| `endpoint` | `str` | `"https://staging-cloud.neatlogs.com"` | Backend base URL. Trace export is normalized to `{base_url}/v1/traces` |
| `workflow_name` | `str` | `None` | Name for this workflow / application |
| `instrumentations` | `list[str]` | `None` | Libraries to auto-instrument (e.g. `["openai", "langchain"]`) |
| `tags` | `list[str]` | `None` | Tags for filtering in dashboard |
| `auto_session` | `bool` | `False` | Auto-generate a session ID on first use and reuse it for the process lifetime. Useful for chatbots / multi-turn conversations |
| `session_id` | `str` | `None` | Explicit session ID — overrides `auto_session`. Pass a per-user or per-conversation ID to group turns in the dashboard |
| `debug` | `bool` | `False` | Enable verbose logging to stderr |
| `pii_enabled` | `Optional[bool]` | `None` | Override the team-level server-side PII redaction setting. `True` = enable, `False` = disable, `None` (default) = use the team setting in the NeatLogs dashboard |
| `pii_span_types` | `Optional[list[str]]` | `None` | Override which span types have PII redaction applied. `None` = use team dashboard config |
| `capture_logs` | `bool` | `False` | Capture `neatlogs.log()`, stdlib `logging.*()`, and `print()` (via `capture_stdout=True` on `@span`) as LOG spans |
| `mask` | `callable` | `None` | Client-side mask function `(span_dict) -> span_dict` — see [Data Masking](#data-masking-and-pii) |

---

## Supported Instrumentations

Pass these string values in the `instrumentations=[]` list to `neatlogs.init()`.

### LLM Providers

| Key | Library | Notes |
|---|---|---|
| `openai` | OpenAI (`OpenAI()` and `AzureOpenAI()`) | Tested end-to-end |
| `anthropic` | Anthropic | Tested |
| `google_genai` | Google Generative AI (`google.genai`) | Tested. Client must be created **after** `init()` — see troubleshooting |
| `azure_ai_inference` | Azure AI Inference | Tested. Required when CrewAI dispatches to Azure — see §CrewAI below |
| `litellm` | LiteLLM | Tested end-to-end with `gemini/*` + message-list templates |
| `bedrock` | AWS Bedrock | Tested. `boto3>=1.42.11` |

### Agent Frameworks

| Key | Framework | Notes |
|---|---|---|
| `langchain` | LangChain (incl. LangGraph execution) | Tested end-to-end |
| `langgraph` | LangGraph — use `instrumentations=["langchain"]` | Tested via LangChain |
| `crewai` | CrewAI | Tested. **Must also add the direct provider key** matching `crewai.LLM(model=...)` — see below |

#### CrewAI routing rules

CrewAI dispatches LLM calls internally via LiteLLM; adding only `"crewai"` is not enough. Match the provider key to your `crewai.LLM(model=...)` prefix:

| `crewai.LLM(model=...)` | Required instrumentations |
|---|---|
| `"gpt-4o"` (OpenAI proper) | `["crewai", "openai"]` |
| `"azure/..."` | `["crewai", "azure_ai_inference"]` |
| `"gemini/..."` | `["crewai", "google_genai"]` |
| `"claude-..."` | `["crewai", "anthropic"]` |

Picking the wrong key makes the LLM call silently untraced — the trace UI shows only the Agent parent with no LLM child. See [`references/troubleshooting.md` §4](references/troubleshooting.md#4-crewai-instrumentation-key-selection) for the full diagnostic.

### Vector Databases

| Key | Library | Notes |
|---|---|---|
| `chromadb` | ChromaDB | Auto-instrumented |
| `pinecone` | Pinecone | Auto-instrumented |
| `qdrant` | Qdrant | Auto-instrumented |
| `weaviate` | Weaviate | Auto-instrumented |

> If you use LangChain retrievers wrapping these (very common for RAG apps), `instrumentations=["langchain"]` already captures the retrieval spans automatically — a dedicated vector-DB key is only needed when you call the DB client directly.

### Other

| Key | Library | Notes |
|---|---|---|
| `mcp` | Model Context Protocol | Tested |

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
| `NEATLOGS_ENDPOINT` | Backend base URL (alternative to `endpoint` param) |

---

## Data Masking and PII

NeatLogs supports both client-side and server-side PII redaction.

### Client-Side Masking

Provide a `mask` callback to `init()` to redact sensitive data before spans leave the process. You can also pass `mask=fn` per-span via `@span(mask=fn)` or `trace(..., mask=fn)`.

The mask function receives a span dict and should return the (possibly mutated) span dict:

```python
def redact_pii(span):
    attrs = span.get("attributes", {})
    for key in list(attrs):
        if "email" in key or "password" in key:
            attrs[key] = "[REDACTED]"
    return span

neatlogs.init(mask=redact_pii)
```

The example above is illustrative — real redaction logic should target the specific attribute names your application writes (common ones: `input.value`, `output.value`, `llm.input_messages.*.message.content`).

### Server-Side PII Redaction

Enable automatic server-side redaction via `init()`:

```python
neatlogs.init(
    pii_enabled=True,
)
```

---

## Documentation

Full documentation: [https://docs.neatlogs.com/](https://docs.neatlogs.com/)
