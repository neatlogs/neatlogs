# Decorators and Traces Reference

Complete reference for all manual instrumentation APIs in NeatLogs SDK v3.

---

## 1. `@neatlogs.span()` Decorator

The primary manual instrumentation API for custom code. Wraps a function to create an OpenTelemetry span with NeatLogs-specific attributes.

### Full Signature

```python
@neatlogs.span(
    kind,                    # Required: span kind string
    name=None,               # Optional: span name (defaults to function name)
    description=None,        # Optional: span description
    version=None,            # Optional: version string
    tags=None,               # Optional: list of tags
    metadata=None,           # Optional: dict of metadata
    attributes=None,         # Optional: dict of custom attributes
    capture_input=True,      # Serialize function args to span attributes
    capture_output=True,     # Serialize return value to span attributes
    capture_stdout=False,    # Capture stdout during execution
    mask=None,               # Per-span mask function
    # Kind-specific parameters:
    agent_name=None,         # AGENT: agent name
    role=None,               # AGENT: agent role (sets agent.name if agent_name not set)
    goal=None,               # AGENT: agent goal
    tool_name=None,          # TOOL/MCP_TOOL: tool name
    parameters=None,         # TOOL: tool parameters
    tool_json_schema=None,   # MCP_TOOL: JSON schema for the tool
    model=None,              # EMBEDDING: model name
    dimension=None,          # EMBEDDING: embedding dimension
)
```

### Valid Kinds

`@span()` raises `ValueError` for any kind not in this set:

`WORKFLOW`, `AGENT`, `CHAIN`, `TOOL`, `RETRIEVER`, `EMBEDDING`, `GUARDRAIL`, `MCP_TOOL`

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

Function representing an AI agent with a specific role/goal. The `role` parameter sets `agent.name` if `agent_name` is not explicitly provided.

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

#### RETRIEVER

RAG retrieval. Auto-extracts query from the first function argument and documents from the return value. Supports extracting query from args named `query`, `question`, or `text` (falls back to first string arg). Documents are extracted from list/tuple return values, or from dict return values with keys `documents`, `docs`, `results`, `matches`, `items`, or `data`.

```python
@neatlogs.span(kind="RETRIEVER")
def retrieve_docs(query: str, top_k: int = 5) -> list:
    return vector_db.search(query, top_k=top_k)
```

#### EMBEDDING

Embedding generation.

```python
@neatlogs.span(kind="EMBEDDING", model="text-embedding-3-small", dimension=1536)
def embed_texts(texts: list[str]) -> list[list[float]]:
    return embedding_model.encode(texts)
```

#### GUARDRAIL

Input/output validation and safety checks.

```python
@neatlogs.span(kind="GUARDRAIL")
def check_toxicity(text: str) -> dict:
    result = toxicity_model.check(text)
    return {"passed": result.score < 0.5, "score": result.score}
```

#### MCP_TOOL

MCP protocol tool handlers. Auto-handles Pydantic model args via `.model_dump()` and wraps string results as `{"result": "..."}` for `output.value`.

```python
@neatlogs.span(kind="MCP_TOOL", tool_name="get_weather", description="Get current weather")
async def get_weather(location: str) -> str:
    return f"Weather in {location}: Sunny, 72°F"
```

### `capture_input` / `capture_output` Behavior

Default is `True` for both. When enabled, function arguments are serialized to the `input.value` span attribute, and the return value is serialized to the `output.value` span attribute (both as JSON with `application/json` mime type).

Set to `False` to suppress serialization — useful for large payloads or sensitive data. The `NEATLOGS_TRACE_CONTENT` environment variable can also be set to `"false"` to globally disable content capture.

### Complete Multi-Agent Example

```python
import neatlogs

neatlogs.init(api_key="...", workflow_name="research-app", instrumentations=["openai"])

from openai import OpenAI  # Import AFTER init() for auto-instrumentation

client = OpenAI()

@neatlogs.span(kind="TOOL", tool_name="web_search")
def web_search(query: str) -> str:
    return f"Results for: {query}"

@neatlogs.span(kind="AGENT", name="researcher", role="Research Analyst")
def researcher(topic: str) -> str:
    search_results = web_search(topic)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": f"Analyze: {search_results}"}],
    )
    return response.choices[0].message.content

@neatlogs.span(kind="WORKFLOW")
def run_pipeline(topic: str) -> str:
    return researcher(topic)

result = run_pipeline("quantum computing")
neatlogs.flush()
neatlogs.shutdown()
```

---

## 2. `neatlogs.trace()` Context Manager

For prompt template tracking AND additional span kinds not available in `@span()`.

### Full Signature

```python
with neatlogs.trace(
    name,                        # Required: span name
    kind=None,                   # Optional: span kind (any string accepted)
    prompt_template=None,        # Optional: PromptTemplate instance
    user_prompt_template=None,   # Optional: UserPromptTemplate instance
    prompt_variables=None,       # Optional: dict of prompt variables
    user_prompt_variables=None,  # Optional: dict of user prompt variables
    version=None,                # Optional: version string
    capture_stdout=False,        # Capture stdout
    mask=None,                   # Per-span mask function
    **attributes,                # Additional span attributes
) as span:
    # span.set_attribute(key, value) to add custom attributes
    ...
```

**IMPORTANT**: Unlike `@span()`, `trace()` does NOT validate the kind string. It accepts any value. This is the ONLY way to create `RERANKER`, `VECTOR_STORE`, and `LLM` kind spans.

When `kind` is not provided, it defaults to `"CHAIN"`.

### Use Cases for `trace()`

1. **Prompt template tracking** — captures template + variables for versioning on LLM spans
2. **Multi-turn session management** — groups turns within a session (when `session_id` is set and no active parent span exists, creates a new root trace)
3. **Span kinds not available in `@span()`**: `RERANKER`, `VECTOR_STORE`, `LLM`
4. **Grouping multiple operations** under a single span

### Common Anti-Pattern

Do NOT wrap a function that already has `@span(kind="WORKFLOW")` in `trace()` — it's redundant and creates a useless extra span:

```python
# ❌ WRONG: Redundant wrapper
@neatlogs.span(kind="WORKFLOW")
def my_workflow():
    pass

with neatlogs.trace(name="main"):
    my_workflow()  # Already traced by @span decorator

# ✅ CORRECT: Just call it directly
my_workflow()
```

---

## 3. Span Kinds Available Only via `trace()`

### RERANKER

For reranking retrieved documents.

Manual attributes:

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

For direct vector database operations (insert, index, query).

Manual attributes:

| Attribute | Type | Description |
|-----------|------|-------------|
| `neatlogs.vectordb.index_name` | `str` | Name of the vector index/collection |
| `neatlogs.vectordb.embedding_model` | `str` | Embedding model used |
| `neatlogs.vectordb.vector_dimension` | `int` | Dimension of stored vectors |
| `neatlogs.vectordb.similarity_algorithm` | `str` | Distance metric (e.g. `cosine`, `dot_product`) |

> **Note**: Supported vector DBs (Chroma, Pinecone, Qdrant, Weaviate, Milvus, Elasticsearch, Redis, Marqo) auto-create `VECTOR_STORE` spans via `instrumentations`. Only use manual spans for custom implementations.

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

---

## 4. Manual Attribute Tables for `@span()` Kinds

### RETRIEVER Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `neatlogs.retrieval.query` | `str` | The retrieval query |
| `neatlogs.retrieval.top_k` | `int` | Number of results requested |
| `neatlogs.retrieval.documents` | `JSON str` | Retrieved documents |

> `@span(kind="RETRIEVER")` auto-extracts query from the first function argument and documents from the return value. Manual `set_attribute` is only needed for custom behavior.

### GUARDRAIL Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `neatlogs.guardrail.input` | `str` | Input to the guardrail |
| `neatlogs.guardrail.passed` | `bool` | Whether the guardrail check passed |
| `neatlogs.guardrail.output` | `str` | Output/result of the guardrail |

### TOOL Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `neatlogs.tool.name` | `str` | Tool name (set via `tool_name` param in `@span`) |
| `neatlogs.tool.description` | `str` | Tool description (set via `description` param) |
| `neatlogs.tool.parameters` | `JSON str` | Tool parameters |
| `neatlogs.tool.json_schema` | `JSON str` | Tool JSON schema |

> **IMPORTANT gotcha**: When using `trace()` for TOOL spans, the attribute key is `tool.name` (dotted, not underscored). Python kwargs can't have dots, so use `span.set_attribute("tool.name", "my_tool")`. Using `tool_name` (underscore) will NOT show the tool name in the NeatLogs dashboard.

---

## 5. Manual LLM Span Attributes (OpenInference Format)

When creating manual LLM spans via `trace(kind="LLM")`, the NeatLogs dashboard requires OpenInference flat indexed attributes to render structured message views:

```python
with neatlogs.trace("llm_call", kind="LLM") as span:
    span.set_attribute("llm.input_messages.0.message.role", "system")
    span.set_attribute("llm.input_messages.0.message.content", "You are a helpful assistant.")
    span.set_attribute("llm.input_messages.1.message.role", "user")
    span.set_attribute("llm.input_messages.1.message.content", user_query)

    response = call_llm(messages)

    span.set_attribute("llm.output_messages.0.message.role", "assistant")
    span.set_attribute("llm.output_messages.0.message.content", response_text)
    span.set_attribute("llm.model_name", "gpt-4o")
```

| Wrong | Right |
|-------|-------|
| Setting `input.value` as a JSON blob | Use flat indexed attributes: `llm.input_messages.N.message.role`, `llm.input_messages.N.message.content` |

> Auto-instrumented LLM calls (via `instrumentations=["openai"]` etc.) handle this automatically. This is only needed for manual LLM spans.

---

## 6. Error Handling on Manual Spans

```python
from opentelemetry.trace import StatusCode, Status

with neatlogs.trace("my_operation", kind="CHAIN") as span:
    try:
        result = do_work()
    except Exception as e:
        span.record_exception(e)
        span.set_status(Status(StatusCode.ERROR, str(e)))
        raise
```

| Wrong | Right |
|-------|-------|
| `span.set_attribute("error", str(e))` | `span.record_exception(e)` + `span.set_status(Status(StatusCode.ERROR, str(e)))` |

Setting an `"error"` attribute does NOT mark the span as failed in the backend. You must use OTel's `record_exception()` and `set_status()` methods. Note: `set_status()` requires a `Status` object, not bare `StatusCode` + string.

> **Note**: `@span()` automatically calls `record_exception()` and `set_status(StatusCode.ERROR)` when the decorated function raises an exception. Manual error handling is only needed inside `trace()` blocks.

---

## 7. `neatlogs.log()` Structured Logging

```python
neatlogs.log("retrieved {count} docs in {ms}ms", count=len(docs), ms=elapsed)
```

- **Signature**: `log(msg_template, level="info", **data)`
- Template with `{key}` placeholders — stored as span name (`log.template` attribute)
- The rendered message is stored as the log body
- Each keyword arg stored as `log.{key}` attribute
- **Requires** being inside an active `@span` or `trace()` context — the log record automatically inherits `trace_id` and `span_id` from the active span
- When `debug=True` is set in `init()`, the rendered message is also echoed to stderr immediately

---

## 8. Stdlib Logging Auto-Capture

```python
neatlogs.init(capture_logs=True, log_level="INFO")
```

- Auto-captures stdlib `logging.info()`, `logging.warning()`, `logging.error()` calls as LOG spans inside `@span` or `trace()` blocks
- `log_level` (default `"INFO"`) sets the minimum level to capture
- Only captures logs that occur within an active span context

---

## 9. Span Nesting Pattern

Decorators and context managers create parent-child relationships. Each nested `@span` or `trace()` becomes a child of the enclosing span:

```
@span(kind="WORKFLOW")     → top-level span
  @span(kind="AGENT")      → child of workflow
    trace("llm_call")       → child of agent (captures prompt template)
      LLM API call           → auto-instrumented child span
    @span(kind="TOOL")      → child of agent
```

The nesting is automatic — OpenTelemetry's context propagation ensures that any span created within an active span becomes its child.

---

## 10. Async Support

- `@span()` works with both sync and async functions automatically. It detects `async def` functions and wraps them correctly.
- `trace()` is a sync `@contextmanager` but works in async code — the context manager itself is sync, the code inside can be async.

```python
@neatlogs.span(kind="AGENT", name="async_researcher")
async def async_researcher(topic: str) -> str:
    with neatlogs.trace("llm_call", kind="LLM",
                        prompt_template=sys_tpl,
                        user_prompt_template=user_tpl):
        msgs = sys_tpl.compile() + user_tpl.compile(topic=topic)
        response = await async_client.chat.completions.create(
            model="gpt-4o", messages=msgs
        )
    return response.choices[0].message.content
```
