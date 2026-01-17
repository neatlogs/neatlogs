# Raindrop AI vs NeatLogs: Trace Collection Comparison

## 🏗️ **Architecture Comparison**

### **Raindrop AI:**
```
User App (Python/TypeScript)
  ↓
Raindrop SDK (decorators/wrappers)
  ↓
OpenTelemetry Span Processor (custom)
  ↓
OTLP HTTP Exporter
  ↓
https://api.raindrop.ai/v1/traces (Authorization: Bearer <token>)
  ↓
Raindrop Backend → Tinybird (columnar DB for analytics)
```

### **NeatLogs:**
```
User App (Python)
  ↓
NeatLogs SDK (auto-instrumentation)
  ↓
OpenInference Instrumentation (LangChain/CrewAI)
  ↓
Custom Span Processor → Queue
  ↓
HTTP POST → http://localhost:3000/api/data/v4/batch
  ↓
Kafka Topic (ingest-events)
  ↓
Kafka Consumer → ClickHouse (spans) + Postgres (metadata)
```

---

## 📊 **What Gets Captured**

### **Raindrop AI - Requires Manual Wrapping:**

```python
import raindrop

# Initialize with tracing enabled
raindrop.init(
    api_key="rd_xxx",
    tracing_enabled=True  # ← Must enable!
)

# MUST wrap interactions
@raindrop.interaction("chat_session")
def handle_chat(query):
    # MUST wrap tools manually
    with raindrop.tool_span("fetch_weather"):
        response = requests.get(f"https://api.weather.com/{query}")
        raindrop.record_input({"query": query})
        raindrop.record_output({"data": response.json()})
    
    # WITHOUT wrapping, HTTP call is NOT captured!
    return response

# For Vercel AI SDK
await generateText({
  model: openai('gpt-4'),
  prompt: 'Hello',
  experimental_telemetry: { isEnabled: true }  // ← Must enable per call!
})
```

**Result:**
- ✅ Captures: Wrapped interactions, tools, LLM calls
- ❌ Misses: Any HTTP/DB calls NOT manually wrapped
- ⚠️ Opt-in: Tracing disabled by default

---

### **NeatLogs - Auto-Instrumentation:**

```python
import neatlogs

# Initialize once
neatlogs.init(
    api_key="xxx",
    workflow_name="my-workflow",
    instrumentations=["langchain"]  # Auto-instruments LangChain
)

# NO wrapping needed - automatic capture!
from langchain.agents import create_openai_functions_agent
from langchain_openai import ChatOpenAI
from langchain.tools import Tool

def fetch_weather(city: str) -> str:
    # This HTTP call is NOT captured (no HTTP instrumentation)
    response = requests.get(f"https://api.weather.com/{city}")
    return response.json()

weather_tool = Tool(name="fetch_weather", func=fetch_weather)
agent = create_openai_functions_agent(llm, [weather_tool], prompt)
result = agent_executor.invoke({"input": "What's the weather in SF?"})

# Automatically captured:
# - LLM spans (prompts, responses, tokens)
# - TOOL spans (name, input, output)
# - AGENT spans (decisions)
# - CHAIN spans (execution flow)
```

**Result:**
- ✅ Captures: LLM, tools, agents, chains (via OpenInference)
- ❌ Misses: HTTP calls within tools (no HTTP instrumentation)
- ✅ Opt-out: Auto-instrumentation enabled by default

---

## 🔍 **Span Attributes Comparison**

### **Raindrop AI Spans:**
```json
{
  "span_kind": "tool",
  "name": "fetch_weather",
  "properties": {
    "tool_name": "fetch_weather",
    "input": "{\"city\": \"SF\"}",
    "output": "{\"temp\": 72, \"weather\": \"sunny\"}"
  },
  "trace_id": "xxx",
  "event_id": "evt_xxx",
  "user_id": "user_123",
  "metadata": {
    "session_id": "session_456"
  }
}
```

### **NeatLogs (OpenInference) Spans:**
```json
{
  "span_kind": "TOOL",
  "name": "fetch_weather",
  "attributes": {
    "openinference.span.kind": "TOOL",
    "tool.name": "fetch_weather",
    "tool.description": "Fetch weather data from external API",
    "input.value": "SF",
    "output.value": "Weather in SF: 72°F, sunny"
  },
  "resource": {
    "attributes": {
      "neatlogs.session_id": "xxx",
      "neatlogs.agent_id": "default-agent",
      "neatlogs.workflow_name": "weather-agent",
      "neatlogs.tags": ["v4", "production"]
    }
  },
  "trace_id": "xxx",
  "context": {
    "trace_id": "otel_trace_xxx",
    "span_id": "otel_span_xxx"
  }
}
```

---

## ⚠️ **Key Differences**

| Feature | Raindrop AI | NeatLogs |
|---------|-------------|----------|
| **Instrumentation Approach** | Manual (decorators/wrappers) | Auto (OpenInference) |
| **HTTP Call Capture** | ❌ Requires `withTool()` wrapper | ❌ Not captured (no instrumentation) |
| **DB Call Capture** | ❌ Requires manual wrapping | ❌ Not captured |
| **LLM Call Capture** | ✅ Yes (if wrapped or SDK enabled) | ✅ Yes (automatic via OpenInference) |
| **Tool Call Capture** | ✅ Yes (if wrapped) | ✅ Yes (automatic via LangChain) |
| **Trace Format** | Custom JSON + OpenTelemetry | OpenTelemetry + OpenInference |
| **Ingestion Protocol** | OTLP HTTP | HTTP → Kafka → Processor |
| **Backend Storage** | Tinybird (columnar) | ClickHouse (columnar) + Postgres |
| **Default Behavior** | Tracing OFF (opt-in) | Tracing ON (opt-out) |
| **RAG Support** | ⚠️ Requires wrapping embeddings/retrieval | ✅ Automatic (EMBEDDING + RETRIEVER spans) |

---

## 🎯 **What This Means**

### **Raindrop AI Philosophy:**
- **Explicit is better than implicit**
- You control exactly what gets traced
- More boilerplate, but more control
- Better for teams that want fine-grained control

### **NeatLogs Philosophy:**
- **Convention over configuration**
- Framework-aware auto-instrumentation
- Less boilerplate, but less control
- Better for teams that want "just works" observability

---

## 🔧 **How to Add HTTP Instrumentation to NeatLogs**

To match Raindrop's capability to capture HTTP calls (without manual wrapping):

```python
# In neatlogs/core.py, add:
def _setup_auto_instrumentation(self):
    """Auto-instrument HTTP clients to capture tool API calls."""
    try:
        from opentelemetry.instrumentation.requests import RequestsInstrumentor
        RequestsInstrumentor().instrument()
        
        from opentelemetry.instrumentation.httpx import HTTPXInstrumentor
        HTTPXInstrumentor().instrument()
        
        from opentelemetry.instrumentation.aiohttp_client import AioHttpClientInstrumentor
        AioHttpClientInstrumentor().instrument()
        
        logging.info("✅ Auto-instrumented HTTP clients for tool tracking")
    except Exception as e:
        logging.warning(f"⚠️ HTTP instrumentation failed: {e}")
```

**Install packages:**
```bash
pip install opentelemetry-instrumentation-requests
pip install opentelemetry-instrumentation-httpx
pip install opentelemetry-instrumentation-aiohttp-client
```

**Result:**
```
TOOL span: fetch_weather
  ├─ input: "SF"
  ├─ output: "Weather in SF: 72°F, sunny"
  └─ HTTP span: GET https://api.weather.com/SF  ← NEW!
      ├─ status: 200
      ├─ duration: 234ms
      └─ headers: {...}
```

---

## ✅ **Recommendation**

**For NeatLogs to match Raindrop AI's capability:**

1. ✅ **Add HTTP auto-instrumentation** (already written in `core.py`)
2. ❌ **Skip DB instrumentation** (TOOL spans already capture input/output)
3. ❌ **Skip metrics** (compute from traces in ClickHouse)
4. ✅ **Add ERROR logs only** (for exception stack traces)

**This gives you:**
- ✅ Automatic LLM, tool, agent, chain capture (via OpenInference)
- ✅ Automatic HTTP call capture within tools (via OTel instrumentation)
- ✅ Less boilerplate than Raindrop (no manual wrapping)
- ✅ More detail than Raindrop (nested HTTP spans)

---

## 📊 **Run the RAG Test to Verify**

```bash
cd /Users/tanishabanik/Projects/neatlogs
poetry run python test_langchain_rag_embeddings.py
```

**Expected spans:**
1. EMBEDDING (batch): 5 documents → vectors
2. EMBEDDING (query): "What is Python..." → vector
3. RETRIEVER: Returns relevant docs
4. LLM: Generates answer with context

**Check in NeatLogs UI:** http://localhost:3000
