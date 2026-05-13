# Decorators and Traces Reference

Complete reference for the manual instrumentation APIs in the NeatLogs SDK.

---

## 1. `@neatlogs.span()` Decorator

The primary manual instrumentation API for custom code. Wraps a function to create an OpenTelemetry span with NeatLogs-specific attributes.

### Signature

```python
@neatlogs.span(
    kind,                    # Required: span kind string
    name=None,               # Optional: span name (defaults to function name)
    description=None,        # Optional: span description (used as tool.description for TOOL/MCP_TOOL)
    mask=None,               # Optional: per-span mask function
    # Kind-specific:
    role=None,               # AGENT: agent role (also sets agent.name)
    goal=None,               # AGENT: agent goal
    tool_name=None,          # TOOL / MCP_TOOL: tool identifier
)
```

### Valid Kinds

`@span()` raises `ValueError` for any kind not in this set:

`WORKFLOW`, `AGENT`, `CHAIN`, `TOOL`, `RETRIEVER`, `EMBEDDING`, `GUARDRAIL`, `MCP_TOOL`

> `RERANKER`, `VECTOR_STORE`, and `LLM` are not accepted by `@span()`. Create them via `trace()` — see §2.

### When to Use Each Kind

#### WORKFLOW

Top-level entry point that orchestrates the full pipeline. Use this for the outermost function that ties together agents, tools, and processing steps.

```python
@neatlogs.span(kind="WORKFLOW")
def run_research_pipeline(topic: str) -> str:
    analysis = researcher_agent(topic)
    report = writer_agent(analysis)
    return report
```

#### AGENT

Function representing an AI agent with a specific role/goal. The `role` parameter sets `agent.name` on the span.

```python
@neatlogs.span(kind="AGENT", name="researcher", role="Research Analyst", goal="Find relevant information")
def researcher_agent(topic: str) -> str:
    # ... agent logic with LLM calls ...
    return findings
```

#### CHAIN

Sequential processing step. Use for any intermediate processing, transformation, or pipeline stage.

```python
@neatlogs.span(kind="CHAIN")
def process_documents(docs: list) -> list:
    return [clean(d) for d in docs]
```

#### TOOL

Tool/function call (web search, calculator, API call, etc.).

```python
@neatlogs.span(kind="TOOL", tool_name="web_search", description="Search the web")
def web_search(query: str) -> str:
    return search_api.search(query)
```

To attach a JSON schema or other tool-level metadata, set it inside a nested `trace()`:

```python
import json

@neatlogs.span(kind="TOOL", tool_name="web_search")
def web_search(query: str) -> str:
    with neatlogs.trace("web_search_schema") as span:
        span.set_attribute("tool.json_schema", json.dumps({
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        }))
    return search_api.search(query)
```

#### RETRIEVER

RAG retrieval. For supported retrieval libraries (LangChain retrievers, chromadb, pinecone, etc.) auto-instrumentation handles everything — no decorator needed. For a **custom** retriever, use `@span(kind="RETRIEVER")` plus a nested `trace()` block to set `neatlogs.retrieval.*` attributes:

```python
import json

@neatlogs.span(kind="RETRIEVER")
def retrieve_docs(query: str, top_k: int = 5) -> list:
    with neatlogs.trace("retrieval_details") as span:
        span.set_attribute("neatlogs.retrieval.query", query)
        span.set_attribute("neatlogs.retrieval.top_k", top_k)
        docs = vector_db.search(query, top_k=top_k)
        span.set_attribute("neatlogs.retrieval.documents", json.dumps(docs))
    return docs
```

See §3 for the full attribute list.

#### EMBEDDING

Embedding generation. For supported providers (OpenAI, Cohere, etc.) auto-instrumentation captures model and dimensions automatically. For a **custom** embedding implementation, use a nested `trace()` block to set `neatlogs.embedding.*` attributes — the decorator alone does not auto-extract embedding metadata.

```python
import json

@neatlogs.span(kind="EMBEDDING")
def embed_texts(texts: list[str]) -> list[list[float]]:
    with neatlogs.trace("embedding_details") as span:
        span.set_attribute("neatlogs.embedding.model_name", "text-embedding-3-small")
        span.set_attribute("neatlogs.embedding.text", json.dumps(texts))
        vectors = embedding_model.encode(texts)
        span.set_attribute("neatlogs.embedding.vector", json.dumps(vectors))
        span.set_attribute("neatlogs.embedding.token_count", sum(len(t.split()) for t in texts))
    return vectors
```

#### GUARDRAIL

Input/output validation and safety checks. For supported guardrail libraries (`instrumentations=["guardrails"]`) attributes are captured automatically. For a **custom** guardrail, use a nested `trace()` to set `neatlogs.guardrail.*` attributes:

```python
@neatlogs.span(kind="GUARDRAIL")
def check_toxicity(text: str) -> dict:
    with neatlogs.trace("toxicity_check") as span:
        span.set_attribute("neatlogs.guardrail.input", text)
        result = toxicity_model.check(text)
        span.set_attribute("neatlogs.guardrail.passed", result.score < 0.5)
        span.set_attribute("neatlogs.guardrail.output", f"score={result.score:.2f}")
    return {"passed": result.score < 0.5, "score": result.score}
```

#### MCP_TOOL

MCP protocol tool handlers. Auto-handles Pydantic model args via `.model_dump()` and wraps string results as `{"result": "..."}` for `output.value`.

```python
@neatlogs.span(kind="MCP_TOOL", tool_name="get_weather", description="Get current weather")
async def get_weather(location: str) -> str:
    return f"Weather in {location}: Sunny, 72°F"
```

### Complete Multi-Agent Example

```python
import neatlogs
from neatlogs import SystemPromptTemplate, UserPromptTemplate

neatlogs.init(
    api_key="...",  # Get from https://app.neatlogs.com/settings/api-keys (or set NEATLOGS_API_KEY env var)
    workflow_name="research-app",
    instrumentations=["openai"],
)

from openai import OpenAI  # Import AFTER init() for auto-instrumentation

client = OpenAI()

sys_tpl = SystemPromptTemplate([
    {"role": "system", "content": "You are a senior {{role}}. Be thorough."}
])
user_tpl = UserPromptTemplate([
    {"role": "user", "content": "Analyze: {{search_results}}"}
])

@neatlogs.span(kind="TOOL", tool_name="web_search")
def web_search(query: str) -> str:
    return f"Results for: {query}"

@neatlogs.span(kind="AGENT", name="researcher", role="Research Analyst")
def researcher(topic: str) -> str:
    search_results = web_search(topic)
    with neatlogs.trace(
        "llm_call",
        kind="LLM",
        system_prompt_template=sys_tpl,
        user_prompt_template=user_tpl,
    ):
        messages = sys_tpl.compile(role="research analyst") + user_tpl.compile(search_results=search_results)
        response = client.chat.completions.create(model="gpt-4o", messages=messages)
    return response.choices[0].message.content

@neatlogs.span(kind="WORKFLOW")
def run_pipeline(topic: str) -> str:
    return researcher(topic)

result = run_pipeline("quantum computing")
neatlogs.flush()
neatlogs.shutdown()
```

### Using `@span()` on Class Methods

`@span()` works on both regular functions and class methods. Place the decorator directly on the method:

```python
class ResearchAgent:
    def __init__(self, client):
        self.client = client

    @neatlogs.span(kind="AGENT", name="researcher", role="Research Analyst")
    def run(self, topic: str) -> str:
        with neatlogs.trace(
            "llm_call",
            kind="LLM",
            system_prompt_template=sys_tpl,
            user_prompt_template=user_tpl,
        ):
            messages = sys_tpl.compile() + user_tpl.compile(topic=topic)
            response = self.client.chat.completions.create(model="gpt-4o", messages=messages)
        return response.choices[0].message.content

    @neatlogs.span(kind="TOOL", tool_name="summarize")
    def summarize(self, text: str) -> str:
        return text[:200]
```

---

## 2. `neatlogs.trace()` Context Manager

Use `trace()` for:

1. **Prompt template tracking** — pass `system_prompt_template=` / `user_prompt_template=` to capture template + variables on LLM spans (primary use case)
2. **Setting custom attributes** on a span for any kind (call `span.set_attribute(...)` inside the block)
3. **Span kinds that `@span()` doesn't accept**: `RERANKER`, `VECTOR_STORE`, `LLM`

### Signature

```python
with neatlogs.trace(
    name,                             # Required: span name
    kind=None,                        # Optional: span kind (any string accepted)
    system_prompt_template=None,      # Optional: SystemPromptTemplate instance
    user_prompt_template=None,        # Optional: UserPromptTemplate instance
    mask=None,                        # Optional: per-span mask function
) as span:
    ...
```

> `prompt_template=` is a backward-compat alias for `system_prompt_template=`. Existing code using the old name keeps working; new code should prefer the canonical name. For legacy string templates (not `SystemPromptTemplate` instances), variables can be passed via `system_prompt_variables=` / `user_prompt_variables=` — rarely needed.

### When you need `as span:` (and when you don't)

- **Prompt template tracking only** — `as span:` is not needed. Variables captured via `SystemPromptTemplate.compile(...)` inside the block land on the span automatically.

  ```python
  # Templates constructed with a message list (see §1) so .compile() returns a
  # list[dict] ready for the chat messages= param.
  sys_tpl = SystemPromptTemplate([{"role": "system", "content": "You are a {{role}}."}])
  user_tpl = UserPromptTemplate([{"role": "user", "content": "{{query}}"}])

  with neatlogs.trace("llm_call", kind="LLM",
                      system_prompt_template=sys_tpl,
                      user_prompt_template=user_tpl):
      msgs = sys_tpl.compile(role="researcher") + user_tpl.compile(query=query)
      response = client.chat.completions.create(model="gpt-4o", messages=msgs)
  ```

- **Setting custom attributes** — bind with `as span:` so you can call `span.set_attribute(...)`.

  ```python
  with neatlogs.trace("rerank", kind="RERANKER") as span:
      span.set_attribute("neatlogs.reranker.query", query)
      span.set_attribute("neatlogs.reranker.top_k", top_n)
      ...
  ```

### Common Anti-Pattern

Do NOT wrap a function that already has `@span(kind="WORKFLOW")` in `trace()` — it creates a redundant extra span:

```python
# WRONG: Redundant wrapper
@neatlogs.span(kind="WORKFLOW")
def my_workflow():
    pass

with neatlogs.trace(name="main"):
    my_workflow()  # Already traced by @span decorator

# RIGHT: Just call it directly
my_workflow()
```

---

## 3. Using `trace()` for Custom Attributes and Non-Decorator Kinds

Use `trace()` with manual attributes whenever you need to enrich a span for a custom (non-standard-lib) implementation of any kind. The three kinds below can only be created via `trace()`.

### RERANKER

For reranking retrieved documents. (Not accepted by `@span()`.)

| Attribute | Type | Description |
|-----------|------|-------------|
| `neatlogs.reranker.query` | `str` | The reranking query |
| `neatlogs.reranker.top_k` | `int` | Number of top results requested |
| `neatlogs.reranker.model_name` | `str` | Reranker model name |
| `neatlogs.reranker.input_documents` | `JSON str` | Documents before reranking |
| `neatlogs.reranker.output_documents` | `JSON str` | Documents after reranking |

```python
import json
import neatlogs

def rerank(query: str, docs: list, top_n: int = 3) -> list:
    with neatlogs.trace("rerank", kind="RERANKER") as span:
        span.set_attribute("neatlogs.reranker.query", query)
        span.set_attribute("neatlogs.reranker.top_k", top_n)
        span.set_attribute("neatlogs.reranker.model_name", "cohere-rerank-v3")
        span.set_attribute("neatlogs.reranker.input_documents", json.dumps(docs))
        reranked = reranker_model.rerank(query, docs, top_n=top_n)
        span.set_attribute("neatlogs.reranker.output_documents", json.dumps(reranked))
    return reranked
```

### VECTOR_STORE

For direct vector database operations (insert, index, query) on a custom/unsupported store. For chromadb, pinecone, qdrant, weaviate, milvus, etc., add them to `instrumentations=[]` instead of using `trace()`.

| Attribute | Type | Description |
|-----------|------|-------------|
| `neatlogs.vectordb.index_name` | `str` | Name of the vector index/collection |
| `neatlogs.vectordb.embedding_model` | `str` | Embedding model used |
| `neatlogs.vectordb.vector_dimension` | `int` | Dimension of stored vectors |
| `neatlogs.vectordb.similarity_algorithm` | `str` | Distance metric (e.g. `cosine`, `dot_product`) |

```python
import neatlogs

def index_documents(docs: list):
    with neatlogs.trace("index_documents", kind="VECTOR_STORE") as span:
        span.set_attribute("neatlogs.vectordb.index_name", "support_kb")
        span.set_attribute("neatlogs.vectordb.embedding_model", "text-embedding-3-small")
        span.set_attribute("neatlogs.vectordb.vector_dimension", 1536)
        span.set_attribute("neatlogs.vectordb.similarity_algorithm", "cosine")
        my_custom_store.upsert(docs)
```

### LLM

For provider auto-instrumentation, use `instrumentations=["openai"]` (etc.) — `trace(kind="LLM")` is only needed to attach prompt templates to an already-instrumented call. See §4.

### RETRIEVER Attributes (custom implementations)

| Attribute | Type | Description |
|---|---|---|
| `neatlogs.retrieval.query` | `str` | The retrieval query |
| `neatlogs.retrieval.top_k` | `int` | Number of results requested |
| `neatlogs.retrieval.documents` | `JSON str` | Retrieved documents |

### GUARDRAIL Attributes (custom implementations)

| Attribute | Type | Description |
|---|---|---|
| `neatlogs.guardrail.input` | `str` | Input to the guardrail |
| `neatlogs.guardrail.passed` | `bool` | Whether the guardrail check passed |
| `neatlogs.guardrail.output` | `str` | Output / result of the guardrail |

### EMBEDDING Attributes (custom implementations)

| Attribute | Type | Description |
|---|---|---|
| `neatlogs.embedding.model_name` | `str` | Embedding model name |
| `neatlogs.embedding.text` | `str` | Input text |
| `neatlogs.embedding.vector` | `list[float]` | Embedding vector |
| `neatlogs.embedding.token_count` | `int` | Token count |

---

## 4. Prompt-Template Wrapper for LLM Calls

`trace(kind="LLM")` attaches prompt templates to an already-instrumented LLM call. The OpenInference instrumentor creates the LLM span itself; `trace()` just stamps the templates:

```python
sys_tpl = SystemPromptTemplate("You are a helpful assistant.")
user_tpl = UserPromptTemplate("{{query}}")

with neatlogs.trace(
    "llm_call",
    kind="LLM",
    system_prompt_template=sys_tpl,
    user_prompt_template=user_tpl,
):
    system_prompt = sys_tpl.compile()
    user_prompt = user_tpl.compile(query=user_query)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": system_prompt},
                  {"role": "user", "content": user_prompt}],
    )
```

Do not set `neatlogs.llm.*` attributes manually on auto-instrumented LLM spans — the SDK's attribute processor already populates them from vendor attributes.

---

## 5. `neatlogs.log()` Structured Logging

```python
neatlogs.log("retrieved {count} docs in {ms}ms", count=len(docs), ms=elapsed)
```

- **Signature**: `log(msg_template, level="info", **data)`
- Template with `{key}` placeholders — stored as span name (`log.template` attribute)
- The rendered message is stored as the log body
- Each keyword arg stored as `log.{key}` attribute
- **Requires** being inside an active `@span` or `trace()` context — the log record inherits `trace_id` and `span_id` from the active span
- Requires `capture_logs=True` in `neatlogs.init()`

---

## 6. Span Nesting Pattern

Decorators and context managers create parent-child relationships. Each nested `@span` or `trace()` becomes a child of the enclosing span:

```
@span(kind="WORKFLOW")     -> top-level span
  @span(kind="AGENT")      -> child of workflow
    trace("llm_call")       -> child of agent (captures prompt template)
      LLM API call           -> auto-instrumented child span
    @span(kind="TOOL")      -> child of agent
```

The nesting is automatic — OpenTelemetry's context propagation ensures any span created within an active span becomes its child.

---

## 7. Async Support

- `@span()` works with both sync and async functions automatically. It detects `async def` and wraps them correctly.
- `trace()` is a sync `@contextmanager` but works in async code — the context manager itself is sync, the code inside can be async.

```python
# sys_tpl and user_tpl use the message-list form (see §1) so .compile()
# returns a list[dict] that can be passed straight to chat messages=.
sys_tpl = SystemPromptTemplate([{"role": "system", "content": "You are a researcher."}])
user_tpl = UserPromptTemplate([{"role": "user", "content": "Research: {{topic}}"}])


@neatlogs.span(kind="AGENT", name="async_researcher")
async def async_researcher(topic: str) -> str:
    with neatlogs.trace("llm_call", kind="LLM",
                        system_prompt_template=sys_tpl,
                        user_prompt_template=user_tpl):
        msgs = sys_tpl.compile() + user_tpl.compile(topic=topic)
        response = await async_client.chat.completions.create(
            model="gpt-4o", messages=msgs
        )
    return response.choices[0].message.content
```
