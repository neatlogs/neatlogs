# Troubleshooting — NeatLogs SDK v3 Reference

Common mistakes, anti-patterns, and diagnostic steps for NeatLogs SDK v3 instrumentation.

---

## 1. Import Order Issues (Most Common Mistake)

| Wrong | Right |
|-------|-------|
| `from openai import OpenAI` then `neatlogs.init()` | `neatlogs.init(instrumentations=["openai"])` then `from openai import OpenAI` |

Auto-instrumentation works by patching library modules at import time. If the library is already imported, patching has no effect and LLM calls will not be traced.

```python
# WRONG
from openai import OpenAI  # Already imported — patching won't work
import neatlogs
neatlogs.init(instrumentations=["openai"])

# RIGHT
import neatlogs
neatlogs.init(instrumentations=["openai"])
from openai import OpenAI  # Imported after init — patching works
```

---

## 2. Google GenAI Instantiation Ordering

Unlike most providers where only the `import` order matters, Google GenAI caches the transport at `Client()` construction time. The client must be **created** AFTER `neatlogs.init()`.

| Wrong | Right |
|-------|-------|
| `client = google.genai.Client()` then `neatlogs.init()` | `neatlogs.init(instrumentations=["google_genai"])` then `client = google.genai.Client()` |

```python
# WRONG — client caches transport before init
from google import genai
client = genai.Client(api_key="...")
neatlogs.init(instrumentations=["google_genai"])  # Too late!

# RIGHT
import neatlogs
neatlogs.init(instrumentations=["google_genai"])
from google import genai
client = genai.Client(api_key="...")  # Transport hooks now active
```

---

## 3. Missing Traces Diagnostic Flowchart

If traces are not appearing in the NeatLogs dashboard, check these in order:

1. **Is `neatlogs.init()` called?** → No → Add `neatlogs.init(...)` as the **very first NeatLogs call** at the top of your entry file, before any other imports or logic.
2. **Is it called BEFORE LLM library imports?** → No → Move `neatlogs.init()` before `import openai` / `import anthropic` / etc.
3. **Is the provider listed in `instrumentations=[]`?** → No → Add it (e.g. `instrumentations=["openai"]`). See the [Supported Instrumentations table in SKILL.md](../SKILL.md#supported-instrumentations) for valid keys.
4. **Is `NEATLOGS_API_KEY` set?** → No → Set it via env var or `api_key=` param. Without it, export is **silently disabled** with no error.
5. **Still missing?** → Enable `debug=True` in `neatlogs.init()` and check stderr output for clues.

---

## 4. HTTP Auto-Instrumentation (Always On)

`neatlogs.init()` **always** instruments `requests`, `httpx`, `urllib3`, and `aiohttp` (if installed), regardless of what you put in the `instrumentations` parameter. This is by design for trace context propagation across HTTP boundaries.

**Gotcha**: In services that call themselves (e.g., a webhook handler that triggers another endpoint on the same service), this can cause **infinite trace loops**.

There is no built-in parameter to disable HTTP auto-instrumentation. However, you can uninstrument each HTTP library immediately after `neatlogs.init()`:

```python
import importlib
import neatlogs

neatlogs.init(api_key="...", instrumentations=["google_genai"])

# Uninstrument HTTP libs — init() always enables them regardless of instrumentations=[]
for cls_path in [
    "opentelemetry.instrumentation.httpx.HTTPXClientInstrumentor",
    "opentelemetry.instrumentation.requests.RequestsInstrumentor",
    "opentelemetry.instrumentation.urllib3.URLLib3Instrumentor",
    "opentelemetry.instrumentation.aiohttp_client.AioHttpClientInstrumentor",
]:
    try:
        mod_path, cls_name = cls_path.rsplit(".", 1)
        mod = importlib.import_module(mod_path)
        getattr(mod, cls_name)().uninstrument()
    except Exception:
        pass  # Library not installed — safe to skip
```

---

---

## 5. Duplicate Span Issues

When using CrewAI, `"crewai"` auto-loads LiteLLM instrumentation. If the CrewAI LLM is backed by a direct provider SDK, add the matching provider key too:

- Azure OpenAI / Azure AI Inference → `["crewai", "azure_ai_inference"]`
- OpenAI SDK → `["crewai", "openai"]`
- Google GenAI → `["crewai", "google_genai"]`
- Anthropic → `["crewai", "anthropic"]`

Do NOT add `"litellm"` alongside a direct provider key for the same CrewAI call path unless verified. Example duplicate combination: `["crewai", "openai", "litellm"]` can make LiteLLM and OpenAI both fire for the same internal LLM call.

---

## 6. Flush/Shutdown Gotcha

Scripts (not long-running servers) **MUST** call `neatlogs.flush()` then `neatlogs.shutdown()` before exit — these two calls are compulsory. Without them, the last batch of spans may not be exported.

```python
# At the end of your script
neatlogs.flush()
neatlogs.shutdown()
```

### Long-Running Servers (FastAPI, Flask, Django)

For servers, call `neatlogs.init()` **once at startup** and `flush()` / `shutdown()` **once at shutdown** — NOT on every request:

```python
# WRONG — flush on every request is a performance disaster
@app.get("/ask")
async def ask(q: str):
    response = client.chat.completions.create(...)
    neatlogs.flush()    # ← Don't do this
    return {"answer": response.choices[0].message.content}

# RIGHT — use FastAPI lifespan (or a framework shutdown hook)
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield  # Server runs
    import asyncio
    await asyncio.to_thread(neatlogs.flush)
    await asyncio.to_thread(neatlogs.shutdown)

app = FastAPI(lifespan=lifespan)
```

**Why?** `flush()` on every request sends one HTTP batch per request instead of one every 5 seconds — this risks API throttling and adds latency. Spans batch automatically via `flush_interval` (default 5s).

### Async Gotcha

`flush()` and `shutdown()` are **synchronous** and will block the event loop if called directly in async code. Use `asyncio.to_thread()`:

```python
import asyncio
import neatlogs

async def main():
    # ... your async application code ...

    # DON'T do this — blocks the event loop
    # neatlogs.flush()

    # DO this instead
    await asyncio.to_thread(neatlogs.flush)
    await asyncio.to_thread(neatlogs.shutdown)

asyncio.run(main())
```

---

## 7. Debug Mode

```python
neatlogs.init(debug=True)
```

- Enables verbose logging to stderr (instrumentation status, span creation, export status)
- Enables `neatlogs.log()` echo to terminal
- For file-based span logging:
  ```bash
  export NEATLOGS_LOG_SPANS=true
  export NEATLOGS_LOG_SPANS_FILE=spans.log
  ```

---

## 8. Common Anti-Patterns Table

| Anti-Pattern | Why It's Wrong | Fix |
|-------------|----------------|-----|
| Wrapping `@span(kind="WORKFLOW")` in `trace()` | Redundant — `@span` already creates a span | Just call the decorated function directly |
| Using `trace()` for custom functions where `@span` would work | That's what `@span` is for | Use `@span(kind="CHAIN")` or the appropriate kind instead |
| Calling `.compile()` outside `trace()` context | Variable bindings won't be captured on the span | Move `.compile()` inside the `with trace(...)` block |
| Not listing all providers in `instrumentations` | Some LLM calls won't be traced | Add all providers your code uses |
| Mixing `mask` on `init()` and per-span | Both can coexist — per-span mask takes precedence over the global mask for that specific span | This is expected behavior, not a bug |
| Using `@span` on `StreamingResponse` endpoints | Decorator closes span when function returns, before async generator produces data | Use `trace()` inside the generator body instead |
| Setting `input.value` as JSON for manual LLM spans (when the SDK is patched) | Dashboard won't render structured prompt views | Use `SystemPromptTemplate` / `UserPromptTemplate` and call `.compile()` inside `trace(kind="LLM")`. Only set `input.value` / `output.value` directly on spans where NO SDK wrapper exists — see [`decorators-and-traces.md` §5b](decorators-and-traces.md#5b-manual-llm-span-when-theres-no-sdk-to-patch) |
| Using `tool_name` attribute with `trace()` | Dashboard expects the public NeatLogs tool metadata key | Use `span.set_attribute("neatlogs.tool.name", "my_tool")` |
| Using `@span(kind="RERANKER")` or `@span(kind="VECTOR_STORE")` | `@span()` raises `ValueError` for these kinds | Use `trace("name", kind="RERANKER")` or `trace("name", kind="VECTOR_STORE")` instead |

---

## 9. Manual `trace(kind="LLM")` Span Disappears From the Dashboard

**Symptom**: a chat / agent step shows its parent AGENT span with no children in the UI (empty card, 2s duration, no input/output). The raw `spans` table contains the LLM span you created with full model / input / output / tokens, but `spans_simplified` (the table the UI reads) only has the AGENT row.

**Root cause**: `neatlogs.trace()` stamps `neatlogs.internal=True` on every span by default. The backend trace finalizer drops every internal LLM span, assuming a canonical OpenInference-instrumented sibling already carries the same data. In the no-SDK path (raw `httpx.post`, streaming REST, anywhere the vendor SDK is bypassed) there is no sibling — YOUR span IS the canonical record — so the finalizer deletes the only LLM row in the trace.

**Fix**: opt out of the internal flag on the first line inside the `with` block:

```python
with neatlogs.trace("raw_api_llm_call", kind="LLM") as llm_span:
    llm_span.set_attribute("neatlogs.internal", False)   # ← required
    # ... rest of span setup, http call, attribute writes ...
```

Full pattern (attributes, streaming, token extraction) is documented in [`decorators-and-traces.md` §5b "Manual LLM span when there's NO SDK to patch"](decorators-and-traces.md#5b-manual-llm-span-when-theres-no-sdk-to-patch).

**Do NOT** override `neatlogs.internal=False` on a `trace()` that wraps an already-auto-instrumented call (case §5a). There the OpenInference LLM span IS the canonical record, and the default flag correctly removes your wrapper after prompt-template data has been merged across — leaving it in place would give you two overlapping LLM spans for the same call.

**Diagnosis quickstart**:

```bash
# compare row counts — if raw > simplified, the finalizer is dropping something
clickhouse-client -q "SELECT 'raw', count() FROM spans WHERE trace_id='<id>' UNION ALL \
                      SELECT 'simp', count() FROM spans_simplified WHERE trace_id='<id>'"
```

---

## 10. Data Masking

For the full client-side masking example and server-side PII redaction configuration, see the [Data Masking and PII section in SKILL.md](../SKILL.md#data-masking-and-pii).

**Per-span mask override**: You can pass `mask=fn` to `@span(mask=fn)` or `trace(..., mask=fn)` to override the global mask for a specific span:

```python
@neatlogs.span(kind="TOOL", tool_name="lookup_user", mask=redact_pii)
def lookup_user(email: str) -> dict:
    return db.find_user(email)
```

> **Note**: Per-span mask takes precedence — the global `init(mask=fn)` mask is skipped for that span.
