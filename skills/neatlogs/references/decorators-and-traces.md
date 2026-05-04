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
    capture_input=True,      # Serialize function args to span attributes
    capture_output=True,     # Serialize return value to span attributes
    capture_stdout=False,    # Capture print() as LOG spans (requires capture_logs=True in init)
    mask=None,               # Per-span mask function
    # Kind-specific parameters:
    role=None,               # AGENT: agent role (sets agent.name)
    goal=None,               # AGENT: agent goal
    tool_name=None,          # TOOL/MCP_TOOL: tool name
    parameters=None,         # TOOL: tool parameters
)
```

### Valid Kinds

`@span()` raises `ValueError` for any kind not in this set:

`WORKFLOW`, `AGENT`, `CHAIN`, `TOOL`, `RETRIEVER`, `EMBEDDING`, `GUARDRAIL`, `MCP_TOOL`

> **Note**: `RERANKER`, `VECTOR_STORE`, and `LLM` are **not** valid for `@span()`. Use `trace()` for these kinds — see §3.

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

To attach a JSON schema for the tool, use `span.set_attribute()` inside a nested `trace()`:

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

RAG retrieval. Use `@span(kind="RETRIEVER")` for supported retrieval libraries — attributes like `top_k` and documents are captured automatically. For custom implementations, add `span.set_attribute()` calls inside a nested `trace()` block (see §4 for attribute names).

```python
@neatlogs.span(kind="RETRIEVER")
def retrieve_docs(query: str, top_k: int = 5) -> list:
    return vector_db.search(query, top_k=top_k)
```

#### EMBEDDING

Embedding generation. Use `@span(kind="EMBEDDING")` for supported embedding libraries — model and dimension are captured automatically. For custom implementations, add `span.set_attribute()` calls with the attribute names from §4.

```python
@neatlogs.span(kind="EMBEDDING")
def embed_texts(texts: list[str]) -> list[list[float]]:
    return embedding_model.encode(texts)
```

#### GUARDRAIL

Input/output validation and safety checks. Use `@span(kind="GUARDRAIL")` for supported guardrail libraries — attributes are captured automatically. For custom implementations, add `span.set_attribute()` calls with the attribute names from §4.

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

### `capture_input` / `capture_output`

Default is `True` for both. Set to `False` to suppress serialization — useful for large payloads or sensitive data.

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

### Using `@span()` on Class Methods

`@span()` works on both regular functions and class methods. Place the decorator directly on the method:

```python
class ResearchAgent:
    def __init__(self, client):
        self.client = client

    @neatlogs.span(kind="AGENT", name="researcher", role="Research Analyst")
    def run(self, topic: str) -> str:
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": topic}],
        )
        return response.choices[0].message.content

    @neatlogs.span(kind="TOOL", tool_name="summarize")
    def summarize(self, text: str) -> str:
        return text[:200]
```

---

## 2. `neatlogs.trace()` Context Manager

For prompt template tracking AND additional span kinds not available in `@span()`.

### Full Signature

```python
with neatlogs.trace(
    name,                        # Required: span name
    kind=None,                   # Optional: span kind (any string accepted)
    prompt_template=None,        # Optional: SystemPromptTemplate instance
    user_prompt_template=None,   # Optional: UserPromptTemplate instance
    prompt_variables=None,       # Optional: dict of prompt variables
    user_prompt_variables=None,  # Optional: dict of user prompt variables
    version=None,                # Optional: version string
    capture_stdout=False,        # Capture stdout
    mask=None,                   # Per-span mask function
) as span:
    # span.set_attribute(key, value) to add custom attributes
    ...
```

**IMPORTANT**: Unlike `@span()`, `trace()` does NOT validate the kind string. It accepts any value. This is the ONLY way to create `RERANKER`, `VECTOR_STORE`, and `LLM` kind spans.

When `kind` is not provided, it defaults to `"CHAIN"`.

### Span Object Methods

The `span` object yielded by `trace()` is an OpenTelemetry `Span`. Available methods:

```python
with neatlogs.trace("my_op", kind="CHAIN") as span:
    span.set_attribute("key", "value")          # Add a custom attribute
    span.record_exception(exception)             # Record an exception (use in except block)
    span.set_status(Status(StatusCode.ERROR, "msg"))  # Mark span as failed
    span.add_event("event_name", {"key": "val"}) # Add a timestamped event
```

### Use Cases for `trace()`

1. **Prompt template tracking** — pass `prompt_template=` / `user_prompt_template=` to capture template + variables on LLM spans (primary use case)
2. **Custom attribute capture** — use `span.set_attribute()` for non-standard libraries where `@span()` can't auto-extract attributes
3. **Span kinds not available in `@span()`**: `RERANKER`, `VECTOR_STORE`, `LLM`

> **`as span:` is optional** when only tracking prompt templates — the `with neatlogs.trace(...):` block is sufficient. The `as span` binding is only needed when you want to call `span.set_attribute()` or other span methods inside the block.

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

> **Note**: For supported vector DBs (chromadb, pinecone, qdrant, weaviate, milvus, etc.), add them to `instrumentations=[]` for automatic span creation. Use `trace("op", kind="VECTOR_STORE")` with manual attributes only for custom/unsupported vector DB implementations.

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

## 4. Manual Attribute Tables

> **How attributes work**: The SDK's `_apply_namespace_mapping` in `attribute_processor.py` does two things:
> 1. Any attribute you set with `neatlogs.*` prefix **passes through unchanged** (line 905)
> 2. Vendor-specific attributes (from auto-instrumentation) get mapped from their source names to `neatlogs.*` targets using `attribute-mapping.json`
>
> This means you CAN set any `neatlogs.*` attribute and it will arrive at the backend. The documented attributes below are the ones the **dashboard renders with specialized views**. Custom `neatlogs.my_app.*` attributes pass through and appear in the raw span data.

### RETRIEVER Attributes

When using `@span(kind="RETRIEVER")`, the decorator auto-sets retrieval metadata for supported libraries. For custom implementations inside `trace()` blocks, set these manually:

| Attribute | Type | Description |
|---|---|---|
| `neatlogs.retrieval.query` | `str` | The retrieval query |
| `neatlogs.retrieval.top_k` | `int` | Number of results requested |
| `neatlogs.retrieval.documents` | `JSON str` | Retrieved documents |

> **Note**: Auto-instrumented libraries set `retrieval.query` (no prefix) which the mapper normalizes to `neatlogs.retriever.query`. When setting manually, use the `neatlogs.retrieval.*` names as shown above — they pass through directly.

### RERANKER Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `neatlogs.reranker.query` | `str` | The reranking query |
| `neatlogs.reranker.top_k` | `int` | Number of results to keep |
| `neatlogs.reranker.model_name` | `str` | Reranker model name |
| `neatlogs.reranker.input_documents` | `JSON str` | Documents before reranking |
| `neatlogs.reranker.output_documents` | `JSON str` | Documents after reranking |

### GUARDRAIL Attributes

> **Note**: Guardrail attributes are NOT defined in `attribute-mapping.json` — they pass through as custom `neatlogs.*` attributes. The dashboard displays them when the span kind is `GUARDRAIL`.

| Attribute | Type | Description |
|-----------|------|-------------|
| `neatlogs.guardrail.input` | `str` | Input to the guardrail |
| `neatlogs.guardrail.passed` | `bool` | Whether the guardrail check passed |
| `neatlogs.guardrail.output` | `str` | Output/result of the guardrail |

### VECTOR_STORE Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `neatlogs.vectordb.index_name` | `str` | Name of the vector index/collection |
| `neatlogs.vectordb.embedding_model` | `str` | Embedding model used |
| `neatlogs.vectordb.vector_dimension` | `int` | Dimension of stored vectors |
| `neatlogs.vectordb.similarity_algorithm` | `str` | Distance metric (e.g. `cosine`, `dot_product`) |

> Additional auto-mapped attributes from supported vector DB libraries: `neatlogs.vectordb.retrieval_query`, `neatlogs.vectordb.retrieval_time_taken`, `neatlogs.vectordb.document_attributes`, `neatlogs.vectordb.retrieval_input_params`, `neatlogs.vectordb.retrieval_documents`.

### TOOL Attributes (auto-handled)

> Prefer `@span(kind="TOOL", tool_name="...", description="...")` — the decorator sets all attributes automatically. Manual `neatlogs.tool.*` attributes are rarely needed.

### Custom Application Attributes

Any `neatlogs.*` attribute passes through to the backend. Use a namespace like `neatlogs.my_app.*` for application-specific data:

```python
with neatlogs.trace("my_step") as span:
    span.set_attribute("neatlogs.my_app.customer_tier", "enterprise")
    span.set_attribute("neatlogs.my_app.request_id", request_id)
```

> **Do NOT** manually set attributes that auto-instrumentation already handles (`neatlogs.llm.model_name`, `neatlogs.llm.token_count.*`, etc.) — those are auto-mapped from vendor-specific attributes by the SDK's attribute processor.

---

## 5. Manual LLM Span Attributes

> Prefer provider auto-instrumentation plus prompt templates. Manual LLM spans are only needed when calling a model's REST endpoint directly (no SDK wrapper).

### 5a. Prompt-template wrapper around an already-instrumented call

This is the common case — an OpenInference instrumentor is patching the SDK and creating the canonical LLM span for you; `trace(kind="LLM")` just attaches prompt templates so they show on the span:

```python
sys_tpl = SystemPromptTemplate("You are a helpful assistant.")
user_tpl = UserPromptTemplate("{{query}}")

with neatlogs.trace(
    "llm_call",
    kind="LLM",
    prompt_template=sys_tpl,
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

> **No custom attributes needed on auto-instrumented LLM spans.** The SDK's `attribute_processor.py` auto-maps vendor attributes (`llm.model_name`, `gen_ai.usage.input_tokens`, etc.) to `neatlogs.llm.*` targets via `attribute-mapping.json`. The `trace(kind="LLM")` wrapper is only for prompt template tracking.

| Wrong | Right |
|-------|-------|
| Setting `input.value` as a JSON blob | Use prompt templates and compile them inside `trace(kind="LLM")` |
| Manually setting `neatlogs.llm.model_name` on auto-instrumented calls | Let auto-instrumentation handle it (mapped from `llm.model_name` / `gen_ai.response.model`) |

> Auto-instrumented LLM calls (via `instrumentations=["openai"]` etc.) handle provider-specific message formatting automatically. The manual `trace(kind="LLM")` block is only for prompt tracking.

### 5b. Manual LLM span when there's NO SDK to patch

If your code bypasses the LLM vendor SDK and POSTs raw JSON to the model endpoint (e.g. via `httpx.AsyncClient`, `requests.post`, or a streaming REST client), the OpenInference instrumentor has nothing to patch — **no LLM span will be created automatically**. You must create the span yourself AND populate it completely, because no canonical sibling exists.

Two non-obvious requirements apply in this case:

#### Requirement 1: Override `neatlogs.internal=False`

By default, every `neatlogs.trace()` span is stamped with `neatlogs.internal=True`. The backend trace finalizer uses this flag to drop "wrapper" spans that duplicate a canonical OI-instrumented sibling. In the no-SDK case there is no sibling — YOUR span IS the canonical record — so if you leave the flag at its default, the finalizer will delete your LLM span and the trace will appear to have only the AGENT parent in `spans_simplified` (the UI renders an empty 2s card).

Override it on the first line inside the `with` block:

```python
with neatlogs.trace("raw_api_llm_call", kind="LLM") as llm_span:
    # Opt out of the "internal wrapper" default so the finalizer keeps this span.
    llm_span.set_attribute("neatlogs.internal", False)
    # ... rest of the span setup ...
```

#### Requirement 2: Set OpenInference source attributes (not `neatlogs.llm.*` targets)

The SDK's attribute mapper rewrites OpenInference-style source names to the canonical `neatlogs.llm.*` targets. The kafka consumer and finalizer populate `spans.input_value`, `spans.output_value`, and the token columns from the **source** names via that mapping — writing directly to the target names passes through unchanged, but the backend columns stay empty.

Set these attributes on the span:

| Source attribute (set by you) | Maps to |
|---|---|
| `input.value` (JSON string — list of `{role, content}` dicts) | `spans.input_value`, `neatlogs.llm.input` |
| `input.mime_type` = `"application/json"` | (hint to UI renderers) |
| `output.value` (JSON string — list of one `{role, content, tool_calls?}` dict) | `spans.output_value`, `neatlogs.llm.output` |
| `output.mime_type` = `"application/json"` | (hint to UI renderers) |
| `llm.model_name` | `neatlogs.llm.model_name` |
| `llm.provider`, `llm.system` | pass-through |
| `llm.invocation_parameters` (JSON string) | `neatlogs.llm.invocation_parameters` |
| `llm.input_messages.{i}.message.role` / `.content` (indexed) | `neatlogs.llm.input_messages.{i}.*` |
| `llm.output_messages.0.message.role` / `.content` | `neatlogs.llm.output_messages.0.*` |
| `llm.output_messages.0.message.tool_calls.{i}.tool_call.function.name` / `.arguments` | tool-call nesting |
| `llm.finish_reason` | pass-through |
| `llm.token_count.prompt` / `.completion` / `.total` | `neatlogs.llm.token_count.*` |
| `llm.token_count.completion_details.reasoning` | `neatlogs.llm.token_count.reasoning` |

Complete minimal example for a raw HTTP call to a model endpoint:

```python
import json, neatlogs, httpx

async def call_model_raw(contents: list[dict], system_prompt: str, model: str):
    payload = {
        "contents": contents,
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "generationConfig": {"temperature": 0.7, "maxOutputTokens": 4096},
    }

    with neatlogs.trace("raw_api_llm_call", kind="LLM") as llm_span:
        # 1. Opt out of the internal-wrapper default
        llm_span.set_attribute("neatlogs.internal", False)

        # 2. Static attributes (before the call)
        llm_span.set_attribute("llm.model_name", model)
        llm_span.set_attribute("llm.provider", "google")
        llm_span.set_attribute(
            "llm.invocation_parameters", json.dumps(payload["generationConfig"])
        )

        # 3. Input — both the JSON blob and the indexed per-message view
        input_messages = [{"role": "system", "content": system_prompt}] + [
            {"role": c["role"], "content": "\n".join(p.get("text", "") for p in c["parts"])}
            for c in contents
        ]
        llm_span.set_attribute("input.value", json.dumps(input_messages))
        llm_span.set_attribute("input.mime_type", "application/json")
        for i, msg in enumerate(input_messages):
            llm_span.set_attribute(f"llm.input_messages.{i}.message.role", msg["role"])
            llm_span.set_attribute(f"llm.input_messages.{i}.message.content", msg["content"])

        # 4. Make the real HTTP call
        async with httpx.AsyncClient() as client:
            r = await client.post(URL, json=payload, headers=HEADERS)
            data = r.json()

        # 5. Extract output and tokens, set them on the span
        output_text = data["candidates"][0]["content"]["parts"][0].get("text", "")
        usage = data.get("usageMetadata", {})
        output_msg = {"role": "assistant", "content": output_text}

        llm_span.set_attribute("output.value", json.dumps([output_msg]))
        llm_span.set_attribute("output.mime_type", "application/json")
        llm_span.set_attribute("llm.output_messages.0.message.role", "assistant")
        llm_span.set_attribute("llm.output_messages.0.message.content", output_text)
        llm_span.set_attribute("llm.token_count.prompt", usage.get("promptTokenCount", 0))
        llm_span.set_attribute("llm.token_count.completion", usage.get("candidatesTokenCount", 0))
        llm_span.set_attribute("llm.token_count.total", usage.get("totalTokenCount", 0))
        if usage.get("thoughtsTokenCount"):
            llm_span.set_attribute(
                "llm.token_count.completion_details.reasoning",
                usage["thoughtsTokenCount"],
            )

        return output_text
```

This span will show up in the trace UI with model, input, output, token counts, and server-side cost calculation — **identical to an auto-instrumented LLM span**.

---

## 6. Error Handling on Manual Spans

> **Only needed inside `trace()` blocks.** `@span()` automatically calls `record_exception()` and `set_status(StatusCode.ERROR)` when the decorated function raises — no manual handling needed.

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

## 8. Span Nesting Pattern

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

## 9. Async Support

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

### Streaming async generators — set span attributes BEFORE the terminal yield

When a `trace()` block sits inside an `async def … yield` generator that streams chunks to a caller, and the caller `break`s out of its `async for` after a terminal chunk (e.g. your own `FINISH` / `DONE` marker), **Python closes the generator without executing any code after that `yield`**. Any `span.set_attribute(...)` calls placed after the terminal yield silently never run — even though the `with neatlogs.trace(...)` block's `__exit__` still fires and closes the span.

Observable symptom: the span exists with its pre-loop attributes (model, input) populated, but post-loop attributes (output, tokens, finish_reason) are blank.

```python
# ❌ WRONG — post-yield attrs never land when the caller breaks on FINISH
async def stream_chunks():
    with neatlogs.trace("raw_llm_call", kind="LLM") as span:
        span.set_attribute("neatlogs.internal", False)
        span.set_attribute("llm.model_name", MODEL)
        span.set_attribute("input.value", json.dumps(input_messages))

        full = ""
        async for chunk in raw_http_stream():
            text = parse(chunk)
            full += text
            if done(chunk):
                yield {"type": "FINISH"}   # ← caller breaks here, generator is closed

        # These never run — generator was garbage-collected after the yield above.
        span.set_attribute("output.value", json.dumps([{"role": "assistant", "content": full}]))
        span.set_attribute("llm.token_count.prompt", prompt_tokens)

# ✅ RIGHT — attach terminal attrs BEFORE yielding the FINISH marker
async def stream_chunks():
    with neatlogs.trace("raw_llm_call", kind="LLM") as span:
        span.set_attribute("neatlogs.internal", False)
        span.set_attribute("llm.model_name", MODEL)
        span.set_attribute("input.value", json.dumps(input_messages))

        full = ""
        async for chunk in raw_http_stream():
            text = parse(chunk)
            full += text
            if done(chunk):
                # Attach final-state attrs here, THEN yield.
                span.set_attribute(
                    "output.value",
                    json.dumps([{"role": "assistant", "content": full}]),
                )
                span.set_attribute("llm.token_count.prompt", prompt_tokens)
                span.set_attribute("llm.token_count.completion", completion_tokens)
                yield {"type": "FINISH"}
```

This is only a concern when YOU own the generator and emit a terminal chunk that causes the caller to stop pulling. If the generator exhausts naturally (runs past the last `yield` and returns), post-yield code runs fine. When in doubt, extract the attribute-writing logic into a helper and call it just before each `yield` that could be terminal.
