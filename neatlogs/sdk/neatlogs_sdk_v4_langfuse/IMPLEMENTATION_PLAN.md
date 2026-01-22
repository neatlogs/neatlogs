# Neatlogs SDK v4 Implementation Plan

**Date:** January 21, 2026  
**Goal:** Create a production-ready, OTel-native SDK combining the best features from Langfuse, LangSmith, Traceloop, Braintrust, Raindrop, OpenLit, and HoneyHive.

---

## Executive Summary

This SDK redesign combines:
- **Traceloop's context propagation** (HTTP spans as children of LLM/TOOL spans)
- **OpenLLMetry + OpenInference semantic conventions** (unified schema)
- **Raindrop's streaming/manual span patterns** (for async/distributed operations)
- **OpenLit's Guards & Evals** (pre/post execution validation)
- **Langfuse's clean decorator API** (`@observe` with span types)
- **Braintrust's evaluation framework** (dataset management, evaluators)

**Core Philosophy:** Production-first observability with analytics capabilities.

---

## What We Liked from Each SDK

### ✅ Traceloop (PRIMARY FOUNDATION)
1. **Strong OTel context propagation** - HTTP spans become children of parent spans
2. **Threading instrumentation** - Context propagates across ThreadPoolExecutor
3. **Comprehensive auto-instrumentation** - 40+ libraries out of the box
4. **`Traceloop.init()` static method** - Clean initialization pattern
5. **Semantic decorators** - `@workflow`, `@agent`, `@tool`, `@task`
6. **`set_association_properties`** - Trace-level metadata propagation

### ✅ Langfuse
1. **Clean `@observe` decorator** - Single decorator with span type parameter
2. **`start_as_current_span` context managers** - Explicit span management
3. **`propagate_attributes` context manager** - Trace-level attributes without creating spans
4. **Filtered span export** - Only export relevant spans (not generic HTTP)
5. **Input/output capture** - Automatic serialization with safeguards

### ✅ LangSmith
1. **`@traceable` decorator simplicity** - No explicit span type needed
2. **`tracing_context()` for trace-level attributes** - Clean API for session/user/tags
3. **Evaluation framework** - Dataset loading, evaluator decorators, concurrent execution

### ✅ Braintrust
1. **Manual spans** - `start_span()` with explicit `start()` and `end()` for async/distributed
2. **Distributed tracing via baggage** - `add_parent_to_baggage`, `parent_from_headers`
3. **Evaluation framework** - `@evaluator` decorators, dataset management
4. **Dual context management** - Flexibility between native and OTel modes (we'll use OTel only)

### ✅ Raindrop
1. **Streaming support** - `Interaction` API with `_track_ai_partial` for incremental updates
2. **Manual span management** - `ManualSpan` with `start()` and `finish()`
3. **PII redaction** - Built-in regex-based redaction for sensitive data
4. **Analytics + tracing hybrid** - Product analytics layer on top of tracing
5. **Wraps Traceloop** - Validates Traceloop as a solid foundation

### ✅ OpenLit
1. **Guards (pre-execution)** - Input validation, prompt injection detection, PII checks
2. **Evals (post-execution)** - Hallucination, bias, toxicity, relevance checks
3. **GPU metrics** - OTel Observable Gauges for NVIDIA/AMD GPU monitoring
4. **Prompt Hub & Vault** - Centralized prompt/secret management
5. **Comprehensive auto-instrumentation** - 40+ libraries, VectorDBs, frameworks

### ✅ HoneyHive
1. **Session management** - Explicit session lifecycle with enrichment API
2. **Evaluation orchestration** - Dataset loading, concurrent evaluation, result reporting
3. **Git metadata collection** - Automatic git info for reproducibility
4. **Wraps Traceloop** - Further validates Traceloop's architecture

### ✅ Semantic Conventions Analysis
1. **OpenLLMetry (Traceloop)** - Metrics, streaming, vendor-specific attributes
2. **OpenInference** - Cost tracking, multimodal, reranker/guardrail/evaluator span kinds
3. **Hybrid approach** - Combine both for production + analytics

---

## Architecture Overview

### Layer 1: OpenTelemetry Foundation
```
┌─────────────────────────────────────────────────────────────┐
│                   OpenTelemetry Core                         │
│  ┌──────────────┐  ┌───────────────┐  ┌─────────────────┐  │
│  │TracerProvider│  │Context Manager│  │   Propagators   │  │
│  └──────────────┘  └───────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Layer 2: Semantic Conventions (Unified Schema)
```
┌─────────────────────────────────────────────────────────────┐
│              Neatlogs Semantic Conventions                   │
│                                                              │
│  ┌──────────────────────┐  ┌──────────────────────────┐    │
│  │  OpenLLMetry Base    │  │  OpenInference Base      │    │
│  │  - Metrics           │  │  - Cost tracking         │    │
│  │  - Streaming         │  │  - Multimodal            │    │
│  │  - Vendor-specific   │  │  - Reranker/Guardrail    │    │
│  └──────────────────────┘  └──────────────────────────┘    │
│                                                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │          Neatlogs Custom Extensions                │    │
│  │  - neatlogs.session_id                             │    │
│  │  - neatlogs.user_id                                │    │
│  │  - neatlogs.workflow_name                          │    │
│  │  - neatlogs.tags                                   │    │
│  │  - neatlogs.thread_id                              │    │
│  │  - neatlogs.parent_span_id (HTTP fallback)         │    │
│  └────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

### Layer 3: Auto-Instrumentation
```
┌─────────────────────────────────────────────────────────────┐
│            Auto-Instrumentation Layer                        │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ LLM Providers│  │  VectorDBs   │  │   Frameworks     │  │
│  │ - OpenAI     │  │ - Pinecone   │  │ - LangChain      │  │
│  │ - Anthropic  │  │ - Chroma     │  │ - LlamaIndex     │  │
│  │ - Cohere     │  │ - Weaviate   │  │ - DSPy           │  │
│  │ - etc.       │  │ - etc.       │  │ - CrewAI         │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │        HTTP Instrumentation (CRITICAL)                │  │
│  │  - requests, httpx, aiohttp, urllib3                 │  │
│  │  - Context propagation in ThreadPoolExecutor         │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Layer 4: Neatlogs SDK (Public API)
```
┌─────────────────────────────────────────────────────────────┐
│                  Neatlogs SDK v4 API                         │
│                                                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │  Initialization                                     │    │
│  │  - neatlogs.init()                                  │    │
│  │  - neatlogs.flush(), neatlogs.shutdown()           │    │
│  └────────────────────────────────────────────────────┘    │
│                                                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │  Decorators                                         │    │
│  │  - @observe (Langfuse-style with kind parameter)   │    │
│  │  - @workflow, @agent, @tool (Traceloop-style)      │    │
│  │  - @llm, @embedding, @retriever, @reranker         │    │
│  │  - @guardrail, @evaluator (OpenLit-style)          │    │
│  └────────────────────────────────────────────────────┘    │
│                                                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │  Context Managers                                   │    │
│  │  - start_span() (manual spans for async/streaming) │    │
│  │  - start_as_current_span() (Langfuse-style)        │    │
│  │  - trace_context() (trace-level attributes)        │    │
│  │  - Interaction() (Raindrop-style streaming)        │    │
│  └────────────────────────────────────────────────────┘    │
│                                                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │  Evaluation Framework                               │    │
│  │  - Dataset loading (local/remote)                   │    │
│  │  - @evaluator decorators                            │    │
│  │  - Evaluation orchestration                         │    │
│  └────────────────────────────────────────────────────┘    │
│                                                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │  Guards & Evals (OpenLit-inspired)                  │    │
│  │  - Pre-execution: PII, prompt injection, toxicity   │    │
│  │  - Post-execution: hallucination, bias, relevance   │    │
│  └────────────────────────────────────────────────────┘    │
│                                                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │  Metrics Collection                                 │    │
│  │  - GPU metrics (Observable Gauges)                  │    │
│  │  - LLM metrics (latency, tokens, TTFT)             │    │
│  │  - VectorDB metrics (query duration, distance)      │    │
│  └────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

### Layer 5: Backend Export
```
┌─────────────────────────────────────────────────────────────┐
│              Span/Metric Export Layer                        │
│                                                              │
│  ┌──────────────────────┐  ┌──────────────────────────┐    │
│  │ NeatlogsSpanProcessor│  │  NeatlogsMetricExporter  │    │
│  │ - Filters spans      │  │  - GPU metrics           │    │
│  │ - Enriches attributes│  │  - LLM metrics           │    │
│  │ - Batches & exports  │  │  - VectorDB metrics      │    │
│  └──────────────────────┘  └──────────────────────────┘    │
│                                                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │         Neatlogs Backend (Kafka + ClickHouse)      │    │
│  └────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

---

## File Structure

```
neatlogs_sdk_v4_langfuse/
├── __init__.py                    # Public API exports
├── IMPLEMENTATION_PLAN.md         # This document
│
├── init.py                        # neatlogs.init() implementation
├── config.py                      # Configuration dataclass
├── state.py                       # Global state management
│
├── decorators/
│   ├── __init__.py
│   ├── observe.py                 # @observe (Langfuse-style)
│   ├── semantic.py                # @workflow, @agent, @tool, etc.
│   ├── guards.py                  # @guardrail (OpenLit-style)
│   └── evaluators.py              # @evaluator (Braintrust/HoneyHive-style)
│
├── context_managers/
│   ├── __init__.py
│   ├── spans.py                   # start_span(), start_as_current_span()
│   ├── trace_context.py           # trace_context() for trace-level attrs
│   └── interaction.py             # Interaction() for streaming (Raindrop-style)
│
├── semantic_conventions/
│   ├── __init__.py
│   ├── openllmetry.py             # Traceloop/OpenLLMetry conventions
│   ├── openinference.py           # OpenInference conventions
│   ├── neatlogs.py                # Neatlogs custom attributes
│   └── unified.py                 # Unified schema (combines both)
│
├── instrumentation/
│   ├── __init__.py
│   ├── manager.py                 # Auto-instrumentation coordinator
│   ├── http.py                    # HTTP instrumentation (requests/httpx/aiohttp)
│   ├── threading_patch.py         # ThreadPoolExecutor context propagation
│   └── registry.py                # Registry of available instrumentations
│
├── otel/
│   ├── __init__.py
│   ├── tracer_provider.py         # TracerProvider setup
│   ├── span_processor.py          # NeatlogsSpanProcessor (filtering + export)
│   ├── context_propagation.py     # Context propagation utilities
│   └── baggage.py                 # Distributed tracing via baggage
│
├── metrics/
│   ├── __init__.py
│   ├── gpu.py                     # GPU metrics (Observable Gauges)
│   ├── llm.py                     # LLM metrics (latency, tokens, TTFT)
│   └── vectordb.py                # VectorDB metrics
│
├── guards/
│   ├── __init__.py
│   ├── base.py                    # Base Guard class
│   ├── pii.py                     # PII detection
│   ├── prompt_injection.py        # Prompt injection detection
│   └── toxicity.py                # Toxicity detection
│
├── evals/
│   ├── __init__.py
│   ├── base.py                    # Base Eval class
│   ├── hallucination.py           # Hallucination detection
│   ├── bias.py                    # Bias detection
│   └── relevance.py               # Relevance scoring
│
├── evaluation/
│   ├── __init__.py
│   ├── dataset.py                 # Dataset loading/management
│   ├── evaluator.py               # Evaluator decorators
│   └── orchestrator.py            # Evaluation orchestration
│
├── utils/
│   ├── __init__.py
│   ├── serialization.py           # Safe input/output serialization
│   ├── redaction.py               # PII redaction utilities
│   └── git_info.py                # Git metadata collection
│
└── tracker.py                     # Backend communication (Kafka export)
```

---

## Phase 1: Foundation (Week 1)

### 1.1 Core Infrastructure
**Goal:** Set up OTel foundation with Traceloop-style context propagation.

**Files:**
- `init.py` - `neatlogs.init()` with OTel setup
- `config.py` - Configuration dataclass
- `state.py` - Global state management
- `otel/tracer_provider.py` - TracerProvider initialization
- `otel/span_processor.py` - NeatlogsSpanProcessor for filtering/export
- `tracker.py` - Backend communication (reuse from v4)

**Key Features:**
1. Initialize OTel TracerProvider with Neatlogs resource attributes
2. Set up NeatlogsSpanProcessor for span filtering and Kafka export
3. Enable context propagation across async/threading boundaries
4. Resource attributes: `neatlogs.session_id`, `neatlogs.user_id`, `neatlogs.workflow_name`, `neatlogs.tags`

**Example:**
```python
import neatlogs

neatlogs.init(
    api_key="...",
    workflow_name="my-workflow",
    user_id="user123",
    session_id="session456",
    tags=["demo", "production"],
    debug=True,
)
```

### 1.2 HTTP Instrumentation (CRITICAL)
**Goal:** Ensure HTTP spans become children of parent spans (Traceloop pattern).

**Files:**
- `instrumentation/http.py` - HTTP client instrumentation
- `instrumentation/threading_patch.py` - ThreadPoolExecutor context propagation

**Key Features:**
1. Instrument `requests`, `httpx`, `aiohttp` with OTel
2. Patch `ThreadPoolExecutor` to propagate context (like Traceloop)
3. HTTP spans automatically become children of active TOOL/LLM/AGENT spans
4. No manual context passing required

**Implementation Details:**
- Use `opentelemetry.instrumentation.requests.RequestsInstrumentor`
- Use `opentelemetry.instrumentation.httpx.HTTPXClientInstrumentor`
- Patch `concurrent.futures.ThreadPoolExecutor.submit` to wrap callables with `contextvars.copy_context()`

### 1.3 Semantic Conventions (Unified Schema)
**Goal:** Define unified semantic conventions combining OpenLLMetry + OpenInference.

**Files:**
- `semantic_conventions/openllmetry.py` - OpenLLMetry constants
- `semantic_conventions/openinference.py` - OpenInference constants
- `semantic_conventions/neatlogs.py` - Neatlogs custom attributes
- `semantic_conventions/unified.py` - Unified schema

**Schema Design:**

```python
# OpenLLMetry (Traceloop) - Primary for operational monitoring
gen_ai.system = "openai"
gen_ai.request.model = "gpt-4"
gen_ai.usage.prompt_tokens = 100
gen_ai.usage.completion_tokens = 50
llm.is_streaming = True
llm.response.finish_reason = "stop"

# OpenInference - Add cost tracking and multimodal
llm.cost.total = 0.0025  # USD
llm.cost.prompt = 0.0015
llm.cost.completion = 0.0010
image.url = "data:image/png;base64,..."
audio.transcript = "Hello world"

# Neatlogs Custom - Session/user tracking
neatlogs.session_id = "sess_123"
neatlogs.user_id = "user_456"
neatlogs.workflow_name = "customer_support"
neatlogs.tags = "demo,production"
neatlogs.thread_id = "thread_789"

# Span kinds (combined)
openinference.span.kind = "LLM"  # OpenInference primary
traceloop.span.kind = "workflow"  # Also set for compatibility
```

**Span Kind Mapping:**
```python
NEATLOGS_SPAN_KINDS = {
    # OpenInference (9 kinds)
    "LLM", "EMBEDDING", "CHAIN", "RETRIEVER", "RERANKER",
    "TOOL", "AGENT", "GUARDRAIL", "EVALUATOR",
    
    # Traceloop (5 kinds) - subset
    "WORKFLOW", "TASK",
    
    # HTTP (for internal use)
    "HTTP",
}
```

---

## Phase 2: Decorators & Context Managers (Week 2)

### 2.1 @observe Decorator (Langfuse-style)
**Goal:** Single decorator with span kind parameter.

**File:** `decorators/observe.py`

**API:**
```python
from neatlogs import observe

@observe(kind="LLM", name="call_openai", capture_input=True, capture_output=True)
def call_openai(prompt: str) -> str:
    # LLM call
    return response

@observe(kind="TOOL", capture_input=True)
async def search_database(query: str) -> list:
    # Tool call
    return results

@observe(kind="AGENT")
async def agent_loop():
    # Async generator for streaming
    async for chunk in stream:
        yield chunk
```

**Features:**
1. Supports sync/async/generators/async generators
2. Automatic input/output capture with serialization
3. Exception recording with error attributes
4. Span kind validation against unified schema

### 2.2 Semantic Decorators (Traceloop-style)
**Goal:** Specialized decorators for common span types.

**File:** `decorators/semantic.py`

**API:**
```python
from neatlogs import workflow, agent, tool, llm, retriever, embedding, reranker, guardrail

@workflow
def my_workflow():
    pass

@agent
async def my_agent():
    pass

@tool(name="search_tool")
def search(query: str):
    pass

@llm
def call_llm(prompt: str):
    pass

@retriever
async def retrieve_docs(query: str):
    pass

@embedding
def generate_embedding(text: str):
    pass

@reranker  # OpenInference span kind
def rerank_results(query: str, docs: list):
    pass

@guardrail  # OpenLit-inspired
def check_toxicity(text: str):
    pass
```

**Implementation:**
- All delegate to `@observe` with appropriate `kind` parameter
- Syntactic sugar for common use cases

### 2.3 Manual Spans (Braintrust/Raindrop-style)
**Goal:** Explicit span management for async/distributed operations.

**File:** `context_managers/spans.py`

**API:**
```python
from neatlogs import start_span

# Context manager (auto start/end)
with start_span(kind="TOOL", name="external_api_call") as span:
    span.set_attribute("api.endpoint", "https://api.example.com")
    result = make_api_call()
    span.set_attribute("api.status_code", 200)

# Manual start/end (for distributed/async operations)
span = start_span(kind="LLM", name="streaming_llm")
span.start()
try:
    for chunk in stream_response():
        span.add_event("chunk_received", {"chunk": chunk})
        process_chunk(chunk)
finally:
    span.end()

# Async support
async with start_span(kind="AGENT", name="async_agent") as span:
    result = await async_operation()
```

**Features:**
1. Context manager for automatic lifecycle
2. Manual `start()` and `end()` for complex flows
3. `set_attribute()`, `add_event()`, `record_exception()` methods
4. Async context manager support

### 2.4 Streaming (Raindrop-style Interaction API)
**Goal:** Incremental updates for streaming operations.

**File:** `context_managers/interaction.py`

**API:**
```python
from neatlogs import Interaction

# Create interaction for streaming LLM call
interaction = Interaction(kind="LLM", name="chat_completion")
interaction.set_input(messages=[...])
interaction.set_properties(model="gpt-4", temperature=0.7)

# Start streaming
interaction.start()

# Incremental updates
full_response = ""
for chunk in stream_openai():
    full_response += chunk
    interaction.update(output=full_response)  # Partial updates

# Finish with final state
interaction.set_output(full_response)
interaction.set_tokens(prompt=100, completion=50)
interaction.set_cost(0.0025)
interaction.finish()

# Can also manually create child spans
with interaction.start_span(kind="TOOL", name="function_call") as tool_span:
    tool_span.set_attribute("function.name", "search")
    result = call_function()
```

**Implementation:**
1. Wraps OTel span with high-level API
2. `update()` creates span events for incremental states
3. Final `finish()` sets span attributes
4. Supports child span creation

### 2.5 Trace Context (LangSmith-style)
**Goal:** Set trace-level attributes without creating spans.

**File:** `context_managers/trace_context.py`

**API:**
```python
from neatlogs import trace_context

# Set trace-level metadata
with trace_context(
    session_id="sess_123",
    user_id="user_456",
    tags=["production", "experiment-a"],
    metadata={"environment": "prod", "version": "1.2.3"}
):
    # All spans created within this context inherit these attributes
    call_llm()
    call_tool()

# Can also use functional API
trace_context.set(session_id="sess_789")
trace_context.set_tags(["debug"])
trace_context.add_metadata("request_id", "req_123")
```

**Implementation:**
1. Uses OTel Baggage for context propagation (like Braintrust)
2. Attributes automatically added to all child spans
3. Does NOT create a new span (unlike Langfuse's `propagate_attributes`)

---

## Phase 3: Auto-Instrumentation (Week 3)

### 3.1 Instrumentation Manager
**Goal:** Coordinate auto-instrumentation of 40+ libraries.

**File:** `instrumentation/manager.py`

**API:**
```python
# In init.py
from .instrumentation import instrument_all

instrument_all(
    instrumentations=["openai", "anthropic", "langchain", "pinecone"],
    tracer_provider=provider,
)

# Or instrument selectively
from .instrumentation import instrument_openai, instrument_langchain

instrument_openai(tracer_provider=provider)
instrument_langchain(tracer_provider=provider)
```

**Registry:**
```python
INSTRUMENTATION_REGISTRY = {
    # LLM Providers
    "openai": "opentelemetry.instrumentation.openai_v2",
    "anthropic": "opentelemetry.instrumentation.anthropic",
    "cohere": "opentelemetry.instrumentation.cohere",
    "bedrock": "opentelemetry.instrumentation.bedrock",
    
    # Vector DBs
    "pinecone": "opentelemetry.instrumentation.pinecone",
    "chroma": "opentelemetry.instrumentation.chromadb",
    "weaviate": "opentelemetry.instrumentation.weaviate",
    "qdrant": "opentelemetry.instrumentation.qdrant",
    
    # Frameworks
    "langchain": "opentelemetry.instrumentation.langchain",
    "llamaindex": "opentelemetry.instrumentation.llamaindex",
    "dspy": "opentelemetry.instrumentation.dspy",
    "crewai": "opentelemetry.instrumentation.crewai",
    
    # HTTP (critical)
    "requests": "opentelemetry.instrumentation.requests",
    "httpx": "opentelemetry.instrumentation.httpx",
    "aiohttp": "opentelemetry.instrumentation.aiohttp_client",
}
```

**Implementation:**
1. Dynamically load instrumentations based on user config
2. Gracefully handle missing packages
3. Log which instrumentations were successfully enabled
4. HTTP instrumentation always enabled by default

### 3.2 HTTP Instrumentation Details
**File:** `instrumentation/http.py`

**Critical Features:**
1. **Auto parent-child relationships** - HTTP spans become children of active TOOL/LLM spans
2. **Context propagation in ThreadPoolExecutor** - Patch executor to copy context
3. **No manual context passing** - Works automatically with decorators

**Implementation:**
```python
# Patch ThreadPoolExecutor to propagate context
import concurrent.futures
import contextvars

original_submit = concurrent.futures.ThreadPoolExecutor.submit

def patched_submit(self, fn, /, *args, **kwargs):
    ctx = contextvars.copy_context()
    return original_submit(self, ctx.run, fn, *args, **kwargs)

concurrent.futures.ThreadPoolExecutor.submit = patched_submit
```

**Why This Matters:**
- LangChain tools run in ThreadPoolExecutor
- Without this patch, HTTP calls in tools become root spans
- With this patch, they become children of TOOL spans

---

## Phase 4: Semantic Conventions Implementation (Week 4)

### 4.1 Unified Schema Implementation
**Goal:** Implement helpers for setting attributes according to unified schema.

**Files:**
- `semantic_conventions/unified.py`
- `utils/serialization.py`

**API:**
```python
from neatlogs.semantic_conventions import SpanAttributes, set_llm_attributes, set_cost_attributes

# In your code
span.set_attribute(SpanAttributes.LLM_REQUEST_MODEL, "gpt-4")
span.set_attribute(SpanAttributes.LLM_REQUEST_TEMPERATURE, 0.7)

# Helper functions
set_llm_attributes(
    span,
    model="gpt-4",
    temperature=0.7,
    max_tokens=1000,
    prompt_tokens=100,
    completion_tokens=50,
    is_streaming=True,
    finish_reason="stop"
)

set_cost_attributes(
    span,
    total_cost=0.0025,
    prompt_cost=0.0015,
    completion_cost=0.0010,
    currency="USD"
)

set_multimodal_attributes(
    span,
    images=[{"url": "...", "type": "image/png"}],
    audio=[{"url": "...", "transcript": "..."}]
)
```

**Attribute Categories:**
1. **LLM Attributes** - Model, params, tokens, streaming, finish reason
2. **Cost Attributes** - Total, prompt, completion, cache, reasoning
3. **Multimodal Attributes** - Images, audio, MIME types
4. **Vector DB Attributes** - Query, results, distances, metadata
5. **Tool Attributes** - Name, description, parameters, results
6. **Session Attributes** - Session ID, user ID, tags, metadata

### 4.2 Cost Calculation
**Goal:** Automatic cost calculation based on model pricing.

**File:** `utils/cost_calculator.py`

**Features:**
1. Pricing table for major LLM providers (OpenAI, Anthropic, Cohere, etc.)
2. Automatic cost calculation from token counts
3. Support for cache hits/misses pricing (OpenAI prompt caching, Anthropic cache)
4. Support for reasoning tokens (o1, o3 models)
5. Regular pricing updates via config

**API:**
```python
from neatlogs.utils.cost_calculator import calculate_cost

cost = calculate_cost(
    model="gpt-4-turbo",
    prompt_tokens=100,
    completion_tokens=50,
    cache_read_tokens=20,  # Cache hits
    cache_write_tokens=10,  # Cache misses
)

# Returns
{
    "total": 0.0025,
    "prompt": 0.0015,
    "completion": 0.0010,
    "prompt_details": {
        "input": 0.0008,
        "cache_read": 0.0002,
        "cache_write": 0.0005,
    }
}
```

---

## Phase 5: Metrics (Week 5)

### 5.1 LLM Metrics (OpenLLMetry)
**Goal:** Implement operational metrics for dashboards/alerting.

**File:** `metrics/llm.py`

**Metrics:**
```python
# Histograms
gen_ai.client.operation.duration      # Request latency
gen_ai.client.token.usage             # Token consumption
llm.chat_completions.streaming_time_to_generate  # TTFT

# Counters
gen_ai.client.generation.choices      # Number of completions
llm.openai.chat_completions.exceptions  # Error count
```

**Implementation:**
- Use OTel Metrics API
- Create metrics in span processor
- Export to Neatlogs backend (ClickHouse)

### 5.2 GPU Metrics (OpenLit)
**Goal:** Monitor GPU usage for on-premise deployments.

**File:** `metrics/gpu.py`

**Metrics:**
```python
# Observable Gauges (async collection)
gpu.utilization          # GPU utilization %
gpu.memory.used          # Memory used (GB)
gpu.memory.available     # Memory available (GB)
gpu.temperature          # Temperature (°C)
gpu.power.usage          # Power usage (W)
gpu.compute.processes    # Active processes
```

**Implementation:**
```python
from opentelemetry.metrics import get_meter
import pynvml  # NVIDIA
import amdsmi  # AMD

class GPUInstrumentor:
    def __init__(self):
        self.meter = get_meter("neatlogs.gpu")
        self.gpu_type = self._detect_gpu()
        
    def _detect_gpu(self):
        try:
            pynvml.nvmlInit()
            return "nvidia"
        except:
            try:
                amdsmi.amdsmi_init()
                return "amd"
            except:
                return None
    
    def instrument(self):
        if self.gpu_type == "nvidia":
            self._create_nvidia_gauges()
        elif self.gpu_type == "amd":
            self._create_amd_gauges()
    
    def _create_nvidia_gauges(self):
        self.meter.create_observable_gauge(
            name="gpu.utilization",
            callbacks=[self._collect_nvidia_utilization],
            unit="%",
            description="GPU utilization percentage"
        )
        # ... more gauges
    
    def _collect_nvidia_utilization(self, options):
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        yield Observation(value=util.gpu, attributes={})
```

**Optional Feature:** Can be enabled via `enable_gpu_metrics=True` in `init()`

### 5.3 Vector DB Metrics
**File:** `metrics/vectordb.py`

**Metrics:**
```python
# Histograms
db.client.query.duration      # Query latency
db.client.search.distance     # Similarity scores

# Counters (vendor-specific)
db.pinecone.usage.read_units
db.pinecone.usage.write_units
```

---

## Phase 6: Guards & Evals (Week 6)

### 6.1 Guards (Pre-Execution Validation)
**Goal:** Validate inputs before execution (OpenLit-inspired).

**File:** `guards/base.py`

**API:**
```python
from neatlogs.guards import PII Guard, PromptInjectionGuard, ToxicityGuard

# Attach guards to decorators
@observe(kind="LLM", guards=[PIIGuard(), PromptInjectionGuard()])
def call_llm(prompt: str):
    return openai.chat.completions.create(...)

# Or use explicitly
guard = PromptInjectionGuard()
result = guard.check(prompt="Ignore all previous instructions...")
if not result.passed:
    raise ValueError(f"Guard failed: {result.reason}")
```

**Built-in Guards:**
1. **PIIGuard** - Detect PII (email, phone, SSN, credit cards)
2. **PromptInjectionGuard** - Detect prompt injection attempts
3. **ToxicityGuard** - Detect toxic/offensive content
4. **JailbreakGuard** - Detect jailbreak attempts

**Implementation:**
```python
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class GuardResult:
    passed: bool
    reason: Optional[str] = None
    score: Optional[float] = None

class Guard(ABC):
    @abstractmethod
    def check(self, text: str) -> GuardResult:
        pass

class PIIGuard(Guard):
    def check(self, text: str) -> GuardResult:
        # Regex-based PII detection
        if email_pattern.search(text):
            return GuardResult(passed=False, reason="Email detected")
        return GuardResult(passed=True)
```

**Span Attributes:**
```python
# Guards create GUARDRAIL spans
openinference.span.kind = "GUARDRAIL"
guardrail.name = "PII_DETECTION"
guardrail.passed = False
guardrail.reason = "Email detected"
guardrail.score = 0.95
```

### 6.2 Evals (Post-Execution Validation)
**Goal:** Evaluate outputs after execution (OpenLit-inspired).

**File:** `evals/base.py`

**API:**
```python
from neatlogs.evals import HallucinationEval, BiasEval, RelevanceEval

# Attach evals to decorators
@observe(kind="LLM", evals=[HallucinationEval(), RelevanceEval()])
def call_llm(prompt: str):
    return openai.chat.completions.create(...)

# Or use explicitly
eval = HallucinationEval()
result = eval.evaluate(
    input="What is the capital of France?",
    output="Paris is the capital of France.",
    context=["Paris is the capital and most populous city of France."]
)
print(result.score)  # 0.95
print(result.passed)  # True
```

**Built-in Evals:**
1. **HallucinationEval** - Check for factual consistency
2. **BiasEval** - Detect bias in responses
3. **ToxicityEval** - Detect toxic content in outputs
4. **RelevanceEval** - Check if output is relevant to input

**Implementation:**
```python
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class EvalResult:
    score: float  # 0.0 - 1.0
    passed: bool
    reason: Optional[str] = None
    metadata: Optional[dict] = None

class Eval(ABC):
    @abstractmethod
    def evaluate(self, input: str, output: str, context: Optional[list] = None) -> EvalResult:
        pass

class HallucinationEval(Eval):
    def evaluate(self, input: str, output: str, context: Optional[list] = None) -> EvalResult:
        # Use LLM-as-judge or model-based detection
        score = self._calculate_hallucination_score(output, context)
        passed = score < 0.3  # Threshold
        return EvalResult(score=score, passed=passed)
```

**Span Attributes:**
```python
# Evals create EVALUATOR spans
openinference.span.kind = "EVALUATOR"
evaluator.name = "HALLUCINATION"
evaluator.score = 0.15
evaluator.passed = True
evaluator.threshold = 0.3
```

---

## Phase 7: Evaluation Framework (Week 7)

### 7.1 Dataset Management
**Goal:** Load and manage evaluation datasets (Braintrust/HoneyHive-style).

**File:** `evaluation/dataset.py`

**API:**
```python
from neatlogs.evaluation import Dataset

# Load from Neatlogs backend
dataset = Dataset.from_neatlogs("my-dataset-id")

# Load from local file
dataset = Dataset.from_json("./dataset.json")
dataset = Dataset.from_csv("./dataset.csv")

# Create programmatically
dataset = Dataset.from_list([
    {"input": "What is 2+2?", "expected": "4", "metadata": {...}},
    {"input": "What is the capital of France?", "expected": "Paris", "metadata": {...}},
])

# Iterate
for item in dataset:
    print(item.input, item.expected)
```

### 7.2 Evaluator Decorators
**Goal:** Define evaluation functions with decorators (Braintrust/HoneyHive-style).

**File:** `evaluation/evaluator.py`

**API:**
```python
from neatlogs.evaluation import evaluator

@evaluator
def check_correctness(input: str, output: str, expected: str) -> float:
    """Returns score 0.0-1.0"""
    return 1.0 if output.strip() == expected.strip() else 0.0

@evaluator
def check_relevance(input: str, output: str) -> dict:
    """Returns dict with score and metadata"""
    relevance_score = calculate_relevance(input, output)
    return {
        "score": relevance_score,
        "metadata": {"method": "cosine_similarity"}
    }

# Async evaluators
@evaluator
async def check_with_llm(input: str, output: str) -> float:
    """Use LLM as judge"""
    result = await openai.chat.completions.create(...)
    return parse_score(result)
```

### 7.3 Evaluation Orchestration
**Goal:** Run evaluations on datasets with concurrent execution.

**File:** `evaluation/orchestrator.py`

**API:**
```python
from neatlogs.evaluation import run_evaluation

# Define function to evaluate
def my_agent(input: str) -> str:
    # Your agent logic
    return process(input)

# Run evaluation
results = run_evaluation(
    name="agent-v1-evaluation",
    dataset=dataset,
    fn=my_agent,
    evaluators=[check_correctness, check_relevance],
    max_concurrency=10,
    timeout_per_item=30.0,
)

# Results
print(f"Average correctness: {results.metrics['correctness_avg']}")
print(f"Average relevance: {results.metrics['relevance_avg']}")
print(f"Pass rate: {results.pass_rate}")

# View failed cases
for item in results.failures:
    print(f"Input: {item.input}")
    print(f"Expected: {item.expected}")
    print(f"Actual: {item.output}")
    print(f"Scores: {item.scores}")
```

**Implementation:**
```python
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context

class EvaluationOrchestrator:
    def run(self, dataset, fn, evaluators, max_concurrency):
        # Use ThreadPoolExecutor with context propagation
        with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
            futures = []
            for item in dataset:
                ctx = copy_context()
                future = executor.submit(ctx.run, self._run_item, item, fn, evaluators)
                futures.append(future)
            
            results = [f.result() for f in futures]
        
        return self._aggregate_results(results)
    
    def _run_item(self, item, fn, evaluators):
        # Run function
        output = fn(item.input)
        
        # Run evaluators
        scores = {}
        for evaluator in evaluators:
            score = evaluator(item.input, output, item.expected)
            scores[evaluator.__name__] = score
        
        return EvaluationResult(
            input=item.input,
            expected=item.expected,
            output=output,
            scores=scores
        )
```

---

## Phase 8: PII Redaction & Utilities (Week 8)

### 8.1 PII Redaction
**Goal:** Automatic PII redaction for sensitive data (Raindrop-inspired).

**File:** `utils/redaction.py`

**API:**
```python
from neatlogs import init

# Enable automatic PII redaction
init(
    api_key="...",
    redact_pii=True,  # Automatic redaction
    redact_patterns=["custom_pattern"],  # Custom regex patterns
)

# Or use manually
from neatlogs.utils.redaction import redact_pii

text = "My email is john@example.com and my SSN is 123-45-6789"
redacted = redact_pii(text)
# "My email is [EMAIL_REDACTED] and my SSN is [SSN_REDACTED]"
```

**Redaction Patterns:**
1. Email addresses
2. Credit card numbers
3. Phone numbers (US/international)
4. SSNs
5. IP addresses
6. API keys (common patterns)
7. Passwords (in JSON/form data)
8. Custom patterns via regex

**Implementation:**
- Regex-based detection
- Applied in span processor before export
- Configurable on/off per attribute type

### 8.2 Git Metadata Collection
**Goal:** Automatic git info for reproducibility (HoneyHive-inspired).

**File:** `utils/git_info.py`

**Features:**
1. Collect git commit hash, branch, author, timestamp
2. Detect dirty working directory
3. Attach to Resource attributes
4. Optional manual override

**API:**
```python
# Automatic (in init)
init(api_key="...", collect_git_info=True)

# Manual
from neatlogs.utils.git_info import get_git_info

info = get_git_info()
print(info)
# {
#     "commit": "abc123...",
#     "branch": "main",
#     "author": "John Doe",
#     "timestamp": "2026-01-21T10:00:00Z",
#     "dirty": False
# }
```

### 8.3 Safe Serialization
**Goal:** Safely serialize inputs/outputs with size limits and type handling.

**File:** `utils/serialization.py`

**Features:**
1. Handle primitive types, lists, dicts
2. Truncate large strings
3. Omit binary data
4. JSON serialization with fallback to `repr()`

**API:**
```python
from neatlogs.utils.serialization import safe_serialize

# Serialize for span attribute
value = safe_serialize(
    obj={"key": "value", "data": large_binary_data},
    max_length=500,
    omit_binary=True
)
span.set_attribute("input.value", value)
```

---

## Phase 9: Span Filtering & Export (Week 9)

### 9.1 Span Filtering (Langfuse-inspired)
**Goal:** Only export relevant spans to backend.

**File:** `otel/span_processor.py`

**Filter Rules:**
1. **Always Export:**
   - LLM, EMBEDDING, TOOL, AGENT, WORKFLOW, CHAIN
   - RETRIEVER, RERANKER, GUARDRAIL, EVALUATOR
   - Spans with `neatlogs.*` attributes
   
2. **Conditionally Export:**
   - HTTP spans IF they are children of LLM/TOOL/AGENT spans
   - Generic HTTP spans to root are FILTERED OUT
   
3. **Never Export:**
   - Internal library spans (e.g., JSON parsing, logging)
   - Health check endpoints
   - Spans without meaningful attributes

**Implementation:**
```python
class NeatlogsSpanProcessor(SpanProcessor):
    def on_end(self, span: ReadableSpan) -> None:
        # Filter logic
        if not self._should_export(span):
            return
        
        # Enrich with resource attributes
        self._enrich_span(span)
        
        # Export to backend
        self.tracker.track_span(span)
    
    def _should_export(self, span: ReadableSpan) -> bool:
        attrs = dict(span.attributes)
        
        # Check span kind
        span_kind = attrs.get("openinference.span.kind")
        if span_kind in ALWAYS_EXPORT_KINDS:
            return True
        
        # Check if HTTP with parent
        if span_kind == "HTTP":
            if span.parent is not None:
                parent_kind = self._get_parent_kind(span.parent)
                if parent_kind in ["LLM", "TOOL", "AGENT"]:
                    return True
            return False
        
        # Check Neatlogs attributes
        if any(k.startswith("neatlogs.") for k in attrs):
            return True
        
        return False
```

### 9.2 Span Enrichment
**Goal:** Add resource attributes and computed fields to spans before export.

**Features:**
1. Copy resource attributes to span attributes
2. Calculate costs if token counts present
3. Add git metadata if available
4. Add computed fields (duration_ms, etc.)

**Implementation:**
```python
def _enrich_span(self, span: ReadableSpan) -> None:
    attrs = dict(span.attributes)
    
    # Add resource attributes
    attrs["neatlogs.session_id"] = self.tracker.session_id
    attrs["neatlogs.user_id"] = self.tracker.user_id
    
    # Calculate cost if tokens present
    if "gen_ai.usage.prompt_tokens" in attrs and "gen_ai.request.model" in attrs:
        cost = calculate_cost(
            model=attrs["gen_ai.request.model"],
            prompt_tokens=attrs["gen_ai.usage.prompt_tokens"],
            completion_tokens=attrs.get("gen_ai.usage.completion_tokens", 0)
        )
        attrs["llm.cost.total"] = cost["total"]
        attrs["llm.cost.prompt"] = cost["prompt"]
        attrs["llm.cost.completion"] = cost["completion"]
    
    # Add git metadata
    if self.git_info:
        attrs["neatlogs.git.commit"] = self.git_info["commit"]
        attrs["neatlogs.git.branch"] = self.git_info["branch"]
    
    # Update span
    for k, v in attrs.items():
        span.set_attribute(k, v)
```

---

## Phase 10: Documentation & Examples (Week 10)

### 10.1 Documentation
**Files:**
- `README.md` - Quick start guide
- `docs/API.md` - Complete API reference
- `docs/SEMANTIC_CONVENTIONS.md` - Attribute schema
- `docs/MIGRATION_V3_TO_V4.md` - Migration guide
- `docs/GUIDES/` - Usage guides (decorators, streaming, evaluation, etc.)

### 10.2 Examples
**Directory:** `examples/`

**Examples to Create:**
1. `basic_usage.py` - Simple LLM call with @observe
2. `langchain_integration.py` - LangChain agent with tools
3. `streaming_example.py` - Streaming LLM with Interaction API
4. `manual_spans.py` - Manual span management for distributed operations
5. `guards_and_evals.py` - Using guards and evals
6. `evaluation_framework.py` - Running evaluations on datasets
7. `gpu_monitoring.py` - GPU metrics collection
8. `cost_tracking.py` - Cost tracking and optimization
9. `distributed_tracing.py` - Distributed tracing with baggage
10. `custom_instrumentation.py` - Custom span creation and attributes

---

## Unified API Reference

### Initialization
```python
import neatlogs

neatlogs.init(
    api_key: str,
    workflow_name: Optional[str] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    tags: Optional[List[str]] = None,
    debug: bool = False,
    
    # Auto-instrumentation
    instrumentations: Optional[List[str]] = None,  # ["openai", "langchain", "pinecone"]
    enable_http_tracing: bool = True,
    
    # Optional features
    enable_gpu_metrics: bool = False,
    redact_pii: bool = False,
    collect_git_info: bool = True,
)
```

### Decorators
```python
# Langfuse-style (single decorator with kind)
from neatlogs import observe

@observe(kind="LLM", name="call_openai", capture_input=True, capture_output=True)
def my_function():
    pass

# Traceloop-style (semantic decorators)
from neatlogs import workflow, agent, tool, llm, retriever, embedding, reranker

@workflow
def my_workflow():
    pass

@agent
def my_agent():
    pass

@tool
def my_tool():
    pass

# OpenLit-style (guards and evals)
from neatlogs import guardrail, evaluator

@guardrail
def check_pii(text: str):
    pass

@evaluator
def check_relevance(input: str, output: str) -> float:
    pass
```

### Context Managers
```python
# Manual spans (Braintrust-style)
from neatlogs import start_span

with start_span(kind="TOOL", name="api_call") as span:
    span.set_attribute("api.endpoint", "https://...")
    result = call_api()

# Manual lifecycle
span = start_span(kind="LLM", name="streaming")
span.start()
try:
    for chunk in stream():
        span.add_event("chunk", {"text": chunk})
finally:
    span.end()

# Streaming (Raindrop-style)
from neatlogs import Interaction

interaction = Interaction(kind="LLM", name="chat")
interaction.set_input(messages=[...])
interaction.start()

for chunk in stream():
    interaction.update(output=chunk)

interaction.finish()

# Trace context (LangSmith-style)
from neatlogs import trace_context

with trace_context(session_id="sess_123", tags=["prod"]):
    # All spans inherit these attributes
    call_llm()
```

### Evaluation
```python
from neatlogs.evaluation import Dataset, run_evaluation, evaluator

# Load dataset
dataset = Dataset.from_neatlogs("dataset-id")

# Define evaluators
@evaluator
def check_correctness(input: str, output: str, expected: str) -> float:
    return 1.0 if output == expected else 0.0

# Run evaluation
results = run_evaluation(
    name="eval-v1",
    dataset=dataset,
    fn=my_agent,
    evaluators=[check_correctness],
    max_concurrency=10
)
```

### Guards & Evals
```python
from neatlogs.guards import PIIGuard, PromptInjectionGuard
from neatlogs.evals import HallucinationEval, RelevanceEval

# Attach to decorators
@observe(
    kind="LLM",
    guards=[PIIGuard(), PromptInjectionGuard()],
    evals=[HallucinationEval(), RelevanceEval()]
)
def call_llm(prompt: str):
    return openai.chat.completions.create(...)

# Or use explicitly
guard = PIIGuard()
if not guard.check(prompt).passed:
    raise ValueError("PII detected")
```

### Utilities
```python
# Flush/shutdown
neatlogs.flush(timeout=10.0)
neatlogs.shutdown()

# PII redaction
from neatlogs.utils.redaction import redact_pii
redacted = redact_pii("email: john@example.com")

# Git info
from neatlogs.utils.git_info import get_git_info
info = get_git_info()

# Cost calculation
from neatlogs.utils.cost_calculator import calculate_cost
cost = calculate_cost(model="gpt-4", prompt_tokens=100, completion_tokens=50)
```

---

## Key Architectural Decisions

### 1. **OpenTelemetry as Foundation** ✅
- Use OTel for all tracing, context propagation, and metrics
- No custom context management (unlike LangSmith/Braintrust native mode)
- Ensures compatibility with OTel ecosystem

### 2. **Traceloop's Context Propagation** ✅
- Patch ThreadPoolExecutor to propagate context
- HTTP instrumentation creates child spans automatically
- No manual context passing required

### 3. **Unified Semantic Conventions** ✅
- Combine OpenLLMetry (operational) + OpenInference (analytics)
- Support both attribute sets for maximum flexibility
- Neatlogs custom attributes for platform-specific features

### 4. **Filtered Span Export** ✅
- Only export relevant spans (LLM, TOOL, AGENT, etc.)
- Filter out generic HTTP root spans
- Include HTTP spans if they're children of LLM/TOOL

### 5. **Multiple Decorator Styles** ✅
- `@observe` (Langfuse-style) for explicit span kinds
- `@workflow`, `@tool`, etc. (Traceloop-style) for semantic clarity
- Both delegate to same underlying implementation

### 6. **Manual Spans for Async/Distributed** ✅
- `start_span()` with manual `start()` and `end()`
- Necessary for distributed tracing, streaming, complex flows
- Inspired by Braintrust and Raindrop

### 7. **Streaming Support** ✅
- `Interaction` API (Raindrop-inspired) for incremental updates
- Span events for streaming chunks
- Final attributes set on completion

### 8. **Guards & Evals** ✅
- Pre-execution validation (Guards) - OpenLit-inspired
- Post-execution evaluation (Evals) - OpenLit-inspired
- Create GUARDRAIL and EVALUATOR spans

### 9. **Evaluation Framework** ✅
- Dataset management (local/remote)
- Evaluator decorators (Braintrust/HoneyHive-style)
- Concurrent execution with context propagation

### 10. **Cost Tracking** ✅
- Automatic cost calculation from token counts
- OpenInference cost attributes
- Support for cache hits/misses, reasoning tokens

---

## Success Criteria

### Functional Requirements
- ✅ HTTP spans are children of parent LLM/TOOL spans (not root spans)
- ✅ Context propagates across threads/async boundaries automatically
- ✅ Decorators work with sync/async/generators/async generators
- ✅ Manual spans support distributed/streaming operations
- ✅ Evaluation framework runs concurrently with context propagation
- ✅ Guards and Evals create appropriate spans
- ✅ Cost tracking works automatically for major LLM providers
- ✅ PII redaction can be enabled globally
- ✅ GPU metrics collection works for NVIDIA and AMD

### Performance Requirements
- Minimal overhead (<5% latency impact)
- Efficient span batching and export
- Async metrics collection (Observable Gauges)
- No blocking operations in critical path

### User Experience Requirements
- Simple `init()` with sensible defaults
- Clean decorator API with minimal boilerplate
- Automatic instrumentation of 40+ libraries
- Helpful error messages and debug logging
- Comprehensive documentation and examples

---

## Migration Path from v3 to v4

### Breaking Changes
1. Rename decorators: `@task` → `@chain` (to match OpenInference)
2. `capture_input`/`capture_output` now default to `True`
3. Session ID auto-generated if not provided
4. Removed v3 custom context management (now OTel-only)

### Migration Guide
```python
# v3
from neatlogs.sdk.neatlogs_sdk_v3 import init, task, agent, tool

init(api_key="...")

@task
def my_task():
    pass

# v4
from neatlogs import init, chain, agent, tool  # or use @observe

init(api_key="...")

@chain  # task → chain
def my_task():
    pass

# Or use @observe for explicit control
@observe(kind="CHAIN", capture_input=True)
def my_task():
    pass
```

---

## Timeline Summary

| Week | Phase | Deliverables |
|------|-------|--------------|
| 1 | Foundation | OTel setup, HTTP instrumentation, semantic conventions |
| 2 | Decorators & Context Managers | @observe, semantic decorators, manual spans, streaming |
| 3 | Auto-Instrumentation | Instrumentation manager, HTTP/threading patches |
| 4 | Semantic Conventions | Unified schema, helpers, cost calculation |
| 5 | Metrics | LLM metrics, GPU metrics, VectorDB metrics |
| 6 | Guards & Evals | Pre/post-execution validation |
| 7 | Evaluation Framework | Datasets, evaluators, orchestration |
| 8 | PII & Utilities | Redaction, git info, serialization |
| 9 | Span Filtering & Export | Filtering rules, enrichment, export |
| 10 | Documentation & Examples | Docs, guides, 10+ examples |

**Total:** 10 weeks (2.5 months)

---

## Dependencies

### Required Packages
```
opentelemetry-api>=1.20.0
opentelemetry-sdk>=1.20.0
opentelemetry-instrumentation>=0.41b0

# HTTP instrumentation
opentelemetry-instrumentation-requests
opentelemetry-instrumentation-httpx
opentelemetry-instrumentation-aiohttp-client

# LLM instrumentation (optional, user installs)
opentelemetry-instrumentation-openai-v2
opentelemetry-instrumentation-anthropic
opentelemetry-instrumentation-langchain
opentelemetry-instrumentation-llamaindex

# GPU metrics (optional)
pynvml  # NVIDIA
amdsmi  # AMD

# Utilities
requests>=2.28.0
pydantic>=2.0.0
```

---

## Open Questions & Future Enhancements

### Open Questions
1. **Prompt Hub implementation** - Should we build a centralized prompt management system?
2. **Vault for secrets** - Should we manage API keys/secrets centrally?
3. **Model Context Protocol (MCP)** - Should we add MCP support like Traceloop?
4. **Multi-backend export** - Should we support OTLP, Console, and custom exporters simultaneously?

### Future Enhancements (Post-v4)
1. **Distributed tracing headers** - Full support for W3C TraceContext propagation
2. **Custom span exporters** - Plugin system for custom backends
3. **Real-time dashboards** - WebSocket streaming of live spans
4. **Cost optimization advisor** - Analyze traces and suggest cost reductions
5. **Anomaly detection** - ML-based detection of unusual patterns
6. **Multi-modal visualization** - View images/audio in traces
7. **A/B testing framework** - Compare different prompts/models
8. **Caching layer** - Semantic caching for LLM responses

---

## Conclusion

This plan combines the best features from 7+ leading observability SDKs into a unified, production-ready solution. By adopting:

- **Traceloop's context propagation** for robust HTTP span parenting
- **OpenLLMetry + OpenInference** for comprehensive semantic conventions
- **Langfuse's clean API** for ease of use
- **Raindrop's streaming patterns** for real-time updates
- **OpenLit's Guards & Evals** for quality assurance
- **Braintrust's evaluation framework** for systematic testing

We create an SDK that is:
- **Production-ready** - Metrics, monitoring, alerting
- **Developer-friendly** - Clean API, automatic instrumentation
- **Comprehensive** - Cost tracking, GPU monitoring, evaluation
- **Future-proof** - OTel-native, extensible, standards-compliant

Let's build it! 🚀
