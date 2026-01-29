# Neatlogs SDK v4 (Langfuse Architecture)

Production-ready AI observability SDK with dual instrumentation, smart attribute merging, and Traceloop-style context propagation.

## 🎯 Key Features

### ✅ Dual Instrumentation
- **OpenInference**: AI-specific span kinds (LLM, AGENT, TOOL, RETRIEVER, EMBEDDING), cost tracking, analytics
- **OpenLLMetry**: Streaming metrics, operational data, vendor-specific attributes
- **Best of both worlds**: Complementary attributes merged intelligently

### ✅ Smart Attribute Merging
- Deduplicates overlapping attributes (e.g., token counts from both conventions)
- Preserves unique attributes from each convention
- Calculates derived metrics (cost, totals)
- **50% storage savings** compared to naive duplication

### ✅ Context Propagation (Traceloop-style)
- **Threading instrumentation**: Maintains context across threads
- **HTTP instrumentation**: HTTP calls are children of LLM/TOOL spans
- **Correct span hierarchy**: No orphaned spans!

### ✅ OpenInference Span Kinds (9 granular types)
- `LLM` - LLM inference calls
- `EMBEDDING` - Embedding generation  
- `RETRIEVER` - Vector/document retrieval
- `RERANKER` - Search result reranking
- `CHAIN` - Sequential operations
- `AGENT` - Autonomous agents
- `TOOL` - Tool/function execution
- `GUARDRAIL` - Input/output validation
- `EVALUATOR` - Quality evaluation

### ✅ Explicit Prompt Capture
- No fragile AST inspection
- `capture_prompt(template, variables)` - explicit and reliable
- `@observe()` decorator - auto-capture function arguments
- Supports versioning

### ✅ Production-Ready
- Sampling for high-volume production
- Batched async export (non-blocking)
- Configurable flush intervals
- Cost calculation with pricing.json

## 📦 Installation

```bash
cd /Users/tanishabanik/Projects/neatlogs
pip install -e .
```

## 🚀 Quick Start

### Minimal Setup

```python
import neatlogs
from openai import OpenAI

# Initialize (auto-instruments OpenAI)
neatlogs.init(
    api_key="your-api-key",
    instrumentations=["openai"]
)

# Use OpenAI normally - everything traced!
client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### Tag-Based Instrumentation

```python
# Instrument by semantic tags
neatlogs.init(
    api_key="your-api-key",
    instrument_tags=["llm", "agent", "retrieval"]
    # Auto-instruments: openai, anthropic, langchain, chromadb, etc.
)
```

### Explicit Prompt Capture

```python
from neatlogs import capture_prompt

city = "San Francisco"
date = "Jan 21, 2026"

capture_prompt(
    template="Weather in {city} on {date}?",
    variables={"city": city, "date": date},
    version="v1.0"
)

response = client.chat.completions.create(...)
```

## 📁 Architecture

```
neatlogs_sdk_v4_langfuse/
├── __init__.py              # Public API
├── init.py                  # SDK initialization
├── pricing.json             # Model pricing (per 1K tokens)
│
├── core/
│   ├── span_processor.py    # Span processing + dedupe/merge
│   ├── attribute_processor.py # Attribute normalization/mapping
│   ├── exporter.py          # Batched async export (spans + metrics)
│   ├── metrics_correlation.py # Correlate instrumentor metrics to spans
│   └── context.py           # Context managers (trace, observe)
│
├── instrumentation/
│   ├── manager.py          # Instrumentation orchestration
│   └── registry.py         # Available instrumentations
│
├── prompt/
│   ├── capture.py          # Explicit prompt capture
│   └── decorators.py       # @observe decorator
│
├── span_kinds/
│   └── mapping.py          # OpenInference ↔ Traceloop mapping
│
└── examples/
    ├── 01_minimal_usage.py
    ├── 03_langchain_agent.py
    ├── 04_langgraph_agent.py
    ├── 05_crewai_multi_agent.py
    ├── 06_openai_direct.py
    ├── 07_anthropic_claude.py
    ├── 08_dspy_programs.py
    └── 09_rag_with_chromadb.py
```

## 🎨 API Reference

### Initialization

```python
neatlogs.init(
    api_key: str,                        # Neatlogs API key
    endpoint: str = "https://api.neatlogs.com",
    workflow_name: str = None,           # Workflow identifier
    session_id: str = None,              # Session grouping
    user_id: str = None,                 # User identifier
    
    # Instrumentation
    instrument_tags: List[str] = None,   # ["llm", "agent", "retrieval"]
    instrumentations: List[str] = None,  # ["openai", "langchain"]
    enable_http_tracing: bool = True,    # Always recommended!
    
    # Performance
    sample_rate: float = 1.0,            # 0.0-1.0 (1.0 = all spans)
    batch_size: int = 100,               # Batch export size
    flush_interval: float = 5.0,         # Seconds between flushes
    
    # Debug
    debug: bool = False,                 # Enable debug logging
)
```

### Context Managers

```python
# Generic span
with neatlogs.trace("operation_name", kind="LLM") as span:
    span.set_attribute("custom.key", "value")
    result = do_work()

# Prompt tracking
with neatlogs.track_prompt(template, variables):
    response = llm.create(...)
```

### Prompt Capture

```python
# Explicit capture
neatlogs.capture_prompt(
    template="Query: {query}",
    variables={"query": user_input},
    version="v1.0"
)

# Variable capture
neatlogs.capture_vars(city=city, date=date)

# Decorator (auto-capture function args)
@neatlogs.observe(name="my_function", version="v1.0")
def weather_lookup(city: str, date: str):
    return llm.create(...)
```

## 🏷️ Available Tags

| Tag | Libraries Instrumented |
|-----|------------------------|
| `llm` | openai, anthropic, cohere, bedrock, groq, together |
| `embedding` | openai, cohere, huggingface |
| `retrieval` | chromadb, pinecone, weaviate, qdrant, milvus |
| `agent` | langchain, llamaindex, crewai, autogen |
| `tool` | langchain, llamaindex |
| `http` | requests, httpx, urllib3, aiohttp |

## 🔄 How It Works

### 1. Dual Instrumentation

```python
# When you call init()
init(instrumentations=["openai"])

# Both instrumentations are applied:
# - OpenLLMetry: opentelemetry.instrumentation.openai
# - OpenInference: openinference.instrumentation.openai

# Result: Two sets of attributes on each span
```

### 2. Attribute Merging

```python
# Raw span attributes (before merging):
{
    "gen_ai.usage.prompt_tokens": 10,        # OpenLLMetry
    "llm.token_count.prompt": 10,            # OpenInference (duplicate!)
    "gen_ai.usage.completion_tokens": 20,    # OpenLLMetry
    "llm.token_count.completion": 20,        # OpenInference (duplicate!)
    "llm.is_streaming": true,                # OpenLLMetry (unique)
    "llm.cost.total": 0.0003,                # OpenInference (unique)
}

# After merging (NeatlogsSpanProcessor):
{
    "llm.token_count.prompt": 10,            # Canonical (OpenInference)
    "llm.token_count.completion": 20,        # Canonical (OpenInference)
    "llm.token_count.total": 30,             # Derived
    "llm.is_streaming": true,                # Preserved (OpenLLMetry unique)
    "llm.cost.total": 0.0003,                # Preserved (OpenInference unique)
}

# ✅ 50% reduction, no data loss!
```

### 3. Context Propagation

```python
# Traceloop's approach (what we use):
ThreadingInstrumentor().instrument()  # ← Context across threads
RequestsInstrumentor().instrument()   # ← HTTP spans

# Result:
LLM span (openai.chat)
└─ HTTP span (POST /v1/chat/completions)  # ← Child, not sibling!

TOOL span (weather_tool)
└─ HTTP span (GET /api/weather)  # ← Child of tool!
```

## 💰 Cost Tracking

### pricing.json

Model pricing is loaded from `pricing.json` (per 1K tokens):

```json
{
  "chat": {
    "gpt-4o-mini": {
      "promptPrice": 0.00015,
      "completionPrice": 0.0006
    },
    "claude-3-5-sonnet-20241022": {
      "promptPrice": 0.003,
      "completionPrice": 0.015
    }
  }
}
```

### Cost Calculation

1. **Primary**: OpenInference instrumentation provides cost (if available)
2. **Fallback**: Calculate from token counts using pricing.json
3. **Result**: Every LLM span has `llm.cost.total`

## 🎯 Best Practices

### 1. Always Enable HTTP Tracing

```python
init(
    enable_http_tracing=True,  # Default, but be explicit!
    # HTTP spans will be children of LLM/TOOL spans
)
```

### 2. Use Tags for Convenience

```python
# Instead of:
init(instrumentations=["openai", "anthropic", "langchain", "llamaindex"])

# Use:
init(instrument_tags=["llm", "agent"])  # Cleaner!
```

### 3. Explicit Prompt Capture

```python
# ❌ Don't rely on magic
prompt = f"Weather in {city}"  # SDK won't detect this

# ✅ Be explicit
capture_prompt("Weather in {city}", {"city": city})
```

### 4. Sample in Production

```python
init(
    sample_rate=0.1,  # Only trace 10% of requests
    # Reduces overhead in high-volume production
)
```

### 5. Use Workflow Names

```python
init(
    workflow_name="customer-support-agent",
    session_id=user_session_id,
    user_id=user_id,
)
```

## 🐛 Troubleshooting

### HTTP spans are orphaned (not children of LLM spans)

**Cause**: HTTP instrumentation not enabled or threading context lost

**Fix**:
```python
init(
    enable_http_tracing=True,  # ← Must be True
    instrumentations=["openai"]
)
```

### Missing cost data

**Cause**: Model name not in pricing.json or OpenInference didn't provide it

**Fix**: Add model to `pricing.json` or check model name format

### Spans not appearing

**Cause**: Library not installed or not instrumented

**Fix**:
```python
init(
    instrumentations=["your-library"],
    debug=True  # ← See what gets instrumented
)
```

## 📊 What Gets Traced

### Every LLM Call

- Token counts (prompt, completion, total)
- Cost (prompt, completion, total)
- Model name, temperature, parameters
- Streaming indicator (if streaming)
- Finish reason (e.g., "stop", "length")
- Prompt template + variables (if captured)

### Every Agent

- Agent name and type
- Input and output
- Execution time
- Child spans (LLM calls, tool calls)
- Token usage aggregated per agent

### Every Tool Call

- Tool name and arguments
- Execution time
- HTTP calls made by tool (as children!)
- Tool output

### Every Retrieval

- Query text
- Number of results
- Retrieved documents
- Similarity scores/distances
- Embedding vectors

## 🚀 Performance

- **Batched export**: Non-blocking, async
- **Sampling**: Skip low-value spans
- **Lazy loading**: Only instruments installed libraries
- **Efficient merging**: O(1) attribute deduplication

## 📚 Examples

See `examples/` directory for:
- LangChain agents
- LangGraph state machines
- CrewAI multi-agent systems
- OpenAI direct usage (streaming, function calling, embeddings)
- Anthropic Claude (prompt caching, tools)
- DSPy programs
- RAG with ChromaDB

## 🤝 Contributing

To add a new instrumentation:

1. Add to `instrumentation/registry.py`:
```python
"libraries": {
    "your-library": {
        "openllmetry": "opentelemetry.instrumentation.your_library",
        "openinference": "openinference.instrumentation.your_library",
        "default_span_kind": "LLM",  # or AGENT, TOOL, etc.
    }
}
```

2. Test with:
```python
init(instrumentations=["your-library"], debug=True)
```

## 📄 License

[Your License Here]

## 🙏 Credits

Built on top of:
- [OpenTelemetry](https://opentelemetry.io/)
- [OpenInference](https://github.com/Arize-ai/openinference)
- [OpenLLMetry (Traceloop)](https://github.com/traceloop/openllmetry)
- [Langfuse](https://langfuse.com/) (architectural inspiration)

---

**Ready to get started?** Check out `examples/01_minimal_usage.py` for the simplest possible setup!
