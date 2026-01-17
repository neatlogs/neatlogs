# NeatLogs SDK: Three Levels of Control

## Design Philosophy

NeatLogs provides **three levels of control** to balance ease-of-use with flexibility:

1. **Level 1: Auto-Capture** - Zero code changes, full observability
2. **Level 2: Enhanced Capture** - Optional decorators for custom functions
3. **Level 3: Granular Control** - Fine-tune what gets collected

---

## Level 1: Auto-Capture (Zero Code Changes)

### Philosophy
**"Just call `neatlogs.init()` and everything works!"**

No decorators needed. No manual instrumentation. Full observability out-of-the-box.

### What Gets Auto-Captured

✅ **All LLM calls** (via OpenInference instrumentation)
- OpenAI, Anthropic, Google, Cohere, Mistral, Groq, Bedrock, etc.
- Prompts, completions, token usage, costs, latency
- Model parameters (temperature, max_tokens, etc.)

✅ **All framework operations** (via OpenInference instrumentation)
- LangChain (chains, agents, tools, retrievers)
- CrewAI (agents, tasks, tools)
- LlamaIndex (queries, retrievers, embeddings)
- AutoGen, Haystack, DSPy, etc.

✅ **All vector DB operations** (via OpenInference instrumentation)
- Chroma, Pinecone, Qdrant, Weaviate, Milvus
- Queries, embeddings, document retrieval

✅ **HTTP client calls** (if enabled)
- `requests`, `httpx`, `aiohttp`
- External API calls from tools

### Global Control Switches

```python
neatlogs.init(
    api_key="...",
    
    # Framework selection - only instrument these
    instrumentations=["openai", "langchain", "crewai"],
    
    # HTTP tracing
    enable_http_tracing=True,     # Auto-trace HTTP calls (default: True)
    
    # Privacy control
    disable_content=False,         # Skip LLM prompts/completions (default: False)
    
    # OpenTelemetry
    enable_otel=True,              # Enable all auto-instrumentation (default: True)
)
```

### Example (Zero Decorators!)

```python
import neatlogs
from openai import OpenAI

# Just init
neatlogs.init(api_key="...")

# This is automatically traced - NO DECORATOR NEEDED!
client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello"}]
)
# ↑ Full trace captured: prompt, response, tokens, cost, latency
```

**What gets captured:**
- `llm.input_messages`: `[{"role": "user", "content": "Hello"}]`
- `llm.output_messages`: `[{"role": "assistant", "content": "Hi there!"}]`
- `llm.model_name`: `"gpt-4o-mini-2024-07-18"`
- `llm.token_count.prompt`: `10`
- `llm.token_count.completion`: `15`
- `llm.token_count.total`: `25`
- Cost, latency, provider, etc.

**All without a single decorator!**

---

## Level 2: Enhanced Capture (Optional Decorators)

### Philosophy
**"Use decorators for YOUR custom functions"**

Auto-instrumentation captures LLM/framework operations. But YOUR custom business logic needs decorators.

### When to Use Decorators

✅ **DO use decorators for:**
- Your custom business logic functions
- Custom tool implementations
- Data processing pipelines
- Functions you want to trace explicitly

❌ **DON'T use decorators for:**
- LLM calls (already auto-traced!)
- Framework operations (already auto-traced!)
- Vector DB calls (already auto-traced!)
- HTTP calls (already auto-traced if enabled!)

### Basic Decorator Usage

```python
@neatlogs.trace()
def process_user_data(user_id: str, data: dict):
    """Your custom function - add decorator to trace"""
    # Your business logic here
    return processed_data
```

**What gets captured:**
- Span name: `"process_user_data"`
- Input arguments: `user_id="user_123"`, `data={...}`
- Output: Return value
- Duration, errors

### Categorize Spans with `span_kind`

```python
@neatlogs.trace(span_kind="TOOL")
def call_external_api():
    """Marked as TOOL for better categorization"""
    return api_response

@neatlogs.trace(span_kind="CHAIN")
def orchestrate_workflow():
    """Marked as CHAIN for workflow steps"""
    return result

@neatlogs.trace(span_kind="AGENT")
def run_agent_logic():
    """Marked as AGENT"""
    return agent_output
```

**Supported span kinds:**
- `"CHAIN"` - Workflow orchestration, multi-step logic
- `"TOOL"` - External tool calls, API interactions
- `"AGENT"` - Agent decision-making logic
- `"RETRIEVER"` - Document retrieval (RAG)
- `"EMBEDDING"` - Embedding generation

### Add Custom Attributes

```python
@neatlogs.trace(
    name="data_validation",
    priority="high",
    data_source="postgres",
    region="us-west"
)
def validate_data(data):
    """All kwargs become span attributes"""
    return validated
```

### Use as Context Manager

```python
with neatlogs.trace("batch_processing"):
    for item in batch:
        process(item)
```

---

## Level 3: Granular Control (Fine-Tuning)

### Philosophy
**"Control exactly what gets collected"**

Fine-tune data collection at the function level and runtime.

### 3A: Disable Tracing for Specific Functions

```python
@neatlogs.trace(enabled=False)
def internal_helper():
    """This won't be traced - useful for high-volume internal functions"""
    return result
```

**Use case:** High-frequency internal functions that add noise

### 3B: Privacy - Don't Capture Inputs/Outputs

```python
@neatlogs.trace(capture_input=False, capture_output=False)
def process_ssn(ssn: str):
    """Function is traced but sensitive data NOT captured"""
    return masked_ssn
```

**What gets captured:**
- ✅ Function name, duration, errors
- ❌ Input arguments (SSN)
- ❌ Output value (masked SSN)

**Use case:** Compliance, PII protection

### 3C: Sampling - Only Trace X% of Calls

```python
@neatlogs.trace(sample_rate=0.1)
def high_volume_operation():
    """Only 10% of calls are traced"""
    return result

# Run 100 times → only ~10 traces created
for i in range(100):
    high_volume_operation()
```

**Use case:** Reduce overhead for high-volume functions

### 3D: Global Disable/Enable

```python
# Pause ALL tracing (including auto-instrumentation!)
neatlogs.disable_tracing()

# These won't be traced
llm.invoke("query 1")
agent.run("task 1")

@neatlogs.trace()
def custom_func():
    pass
custom_func()  # Also NOT traced!

# Resume tracing
neatlogs.enable_tracing()

# These will be traced again
llm.invoke("query 2")
```

**Use case:**
- Testing/debugging specific code paths
- Performance-critical sections
- Background tasks where traces aren't needed

### 3E: Runtime Context Updates

```python
# Initialize with user/session
neatlogs.init(
    api_key="...",
    user_id="user_123",
    session_id="session_abc"
)

# Update mid-workflow
neatlogs.set_user("user_456")
neatlogs.set_session("session_xyz")

# Set custom metadata
neatlogs.set_context(
    user_id="user_789",
    metadata={"tier": "premium", "region": "us-east"}
)
```

**Context propagates to all child spans via OpenTelemetry Baggage!**

---

## Complete Example: All Three Levels

```python
import neatlogs
from openai import OpenAI

# LEVEL 1: Auto-capture setup
neatlogs.init(
    api_key="...",
    user_id="user_123",
    session_id="session_abc",
    instrumentations=["openai"],      # Only OpenAI
    enable_http_tracing=True,         # Auto-trace HTTP
    disable_content=False,            # Capture prompts/completions
)

# LEVEL 1: Auto-traced (no decorator!)
client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello"}]
)

# LEVEL 2: Enhanced capture (with decorator)
@neatlogs.trace(span_kind="CHAIN")
def orchestrate_workflow(user_id: str):
    """Your custom function"""
    # Call LLM (auto-traced)
    response = client.chat.completions.create(...)
    
    # Your business logic (traced via decorator)
    result = process_data(response)
    return result

# LEVEL 3: Granular control
@neatlogs.trace(
    enabled=True,                     # This function is traced
    capture_input=False,              # Don't capture inputs
    sample_rate=0.5,                  # Only 50% of calls
)
def sensitive_operation(ssn: str):
    return masked

# LEVEL 3: Runtime control
neatlogs.disable_tracing()
# ... code not traced ...
neatlogs.enable_tracing()

neatlogs.set_user("user_456")        # Update context
neatlogs.annotate({"key": "value"})  # Add attributes
```

---

## Decision Tree: Should I Use a Decorator?

```
┌─────────────────────────────────────┐
│ Is it an LLM call?                  │
│ (OpenAI, Anthropic, etc.)           │
└──────────┬──────────────────────────┘
           │
    YES ←──┘───→ NO
     │             │
     v             v
  ❌ DON'T    ┌─────────────────────┐
   USE        │ Is it a framework   │
  DECORATOR   │ operation?          │
              │ (LangChain, CrewAI) │
              └──────┬──────────────┘
                     │
              YES ←──┘───→ NO
               │             │
               v             v
            ❌ DON'T    ┌─────────────────┐
             USE        │ Is it YOUR      │
            DECORATOR   │ custom function?│
                        └──────┬──────────┘
                               │
                        YES ←──┘───→ NO
                         │             │
                         v             v
                      ✅ USE      ❌ DON'T USE
                     DECORATOR    DECORATOR
```

---

## Comparison with Competitors

### OpenLLMetry (Traceloop)

| Feature | OpenLLMetry | NeatLogs |
|---------|-------------|----------|
| Auto-capture LLMs | ✅ | ✅ |
| Auto-capture frameworks | ✅ | ✅ |
| Auto-capture HTTP | ✅ | ✅ |
| Custom decorators | ✅ `@workflow`, `@task` | ✅ `@trace` |
| Per-function enable/disable | ❌ | ✅ `enabled=False` |
| Per-function privacy control | ❌ | ✅ `capture_input/output=False` |
| Per-function sampling | ❌ | ✅ `sample_rate=0.1` |
| Global disable/enable | ❌ | ✅ `disable_tracing()` |

### Langfuse

| Feature | Langfuse | NeatLogs |
|---------|----------|----------|
| Auto-capture LLMs | ✅ | ✅ |
| Custom decorators | ✅ `@observe` | ✅ `@trace` |
| Context propagation | ✅ | ✅ |
| Per-function control | ⚠️ Limited | ✅ Full control |
| Span categorization | ✅ | ✅ `span_kind` |

**NeatLogs advantage:** More granular per-function control!

---

## Best Practices

### ✅ DO:
1. **Start with Level 1** - Just `neatlogs.init()`, see what auto-captures
2. **Add Level 2 decorators** only for YOUR custom functions
3. **Use Level 3 controls** for privacy, sampling, high-volume functions
4. **Use `span_kind`** to categorize your custom spans
5. **Use `capture_input/output=False`** for sensitive data

### ❌ DON'T:
1. **Don't decorate LLM calls** - Already auto-traced!
2. **Don't decorate framework operations** - Already auto-traced!
3. **Don't over-decorate** - Only trace what adds value
4. **Don't capture sensitive data** - Use `capture_input/output=False`
5. **Don't trace high-volume internals** - Use `enabled=False` or `sample_rate`

---

## Testing

Run the comprehensive demo:

```bash
cd /Users/tanishabanik/Projects/neatlogs
poetry run python test_three_levels_control.py
```

This demonstrates all three levels with real examples!

---

## Summary

| Level | When to Use | Code Changes | Control |
|-------|-------------|--------------|---------|
| **Level 1: Auto-Capture** | Always! Default behavior | Just `init()` | Global switches |
| **Level 2: Enhanced Capture** | For YOUR custom functions | Add `@trace()` | Per-function categorization |
| **Level 3: Granular Control** | Privacy, sampling, tuning | Add parameters | Per-function fine-tuning |

**The NeatLogs way: Start simple (Level 1), add decorators only where needed (Level 2), fine-tune for production (Level 3).**

---

## Migration from Competitors

### From OpenLLMetry:

```python
# Before (OpenLLMetry)
from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import workflow, task

Traceloop.init(app_name="app")

@workflow(name="my_workflow")
def run_workflow():
    task1()

@task(name="task1")
def task1():
    pass
```

```python
# After (NeatLogs)
import neatlogs

neatlogs.init(api_key="...", workflow_name="app")

@neatlogs.trace(name="my_workflow", span_kind="CHAIN")
def run_workflow():
    task1()

@neatlogs.trace(name="task1")
def task1():
    pass
```

**Key difference:** NeatLogs uses ONE decorator (`@trace`) with `span_kind` parameter instead of multiple decorators.

### From Langfuse:

```python
# Before (Langfuse)
from langfuse.decorators import observe

@observe()
def my_function():
    pass
```

```python
# After (NeatLogs)
import neatlogs

@neatlogs.trace()
def my_function():
    pass
```

**Nearly identical!** Just replace `@observe` with `@trace`.

---

## Conclusion

NeatLogs provides **the best of both worlds:**

1. **Ease of use** - Auto-capture everything by default (like OpenLLMetry)
2. **Flexibility** - Granular per-function control (better than competitors)
3. **Privacy** - Built-in controls for sensitive data
4. **Performance** - Sampling and selective tracing

**Three levels give you full control without complexity!**
