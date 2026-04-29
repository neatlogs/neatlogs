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

## 5. Zero-Span Rows from Non-AI HTTP Traffic

`neatlogs.init()` always instruments outgoing HTTP clients (`requests`, `httpx`, `urllib3`, `aiohttp`) for trace context propagation. It does **not** auto-instrument inbound FastAPI/ASGI server spans; those appear only if the application enables FastAPI/ASGI OpenTelemetry instrumentation separately.

If the dashboard shows rows with 0 spans for non-AI activity, distinguish these cases:

1. **Outgoing HTTP-only traces** from background jobs, health checks, or non-AI endpoints. These can be produced by NeatLogs' always-on HTTP client instrumentation.
2. **Inbound FastAPI/ASGI request root spans** only when the customer has separately enabled FastAPI/ASGI OTel instrumentation.

Customer-side initialization changes alone do not robustly suppress these rows. The correct product fix is backend-side trace row creation/finalization: only create or display a workflow trace row when the trace contains at least one NeatLogs semantic application span (`neatlogs.span.kind` such as `workflow`, `agent`, `llm`, `tool`, etc.), or preserve a non-AI HTTP root only when it has semantic children. Frontend filtering (for example, span count > 1) is acceptable as a temporary workaround but hides the symptom rather than fixing ingestion/finalization.

For local confirmation, run:

```bash
NEATLOGS_API_KEY=<your-key> python tests/manual/test_http_zero_span_repro.py
```

This creates one outgoing HTTP-only trace and one proper `@span(kind="WORKFLOW")` trace so you can compare what the UI displays.

---

## 6. Duplicate Span Issues

When using CrewAI, adding both provider-specific and framework instrumentations creates intentional parent-child hierarchies — but the wrong combination causes duplicate spans:

- **Correct** `["crewai", "openai"]` → CrewAI wraps OpenAI in a parent-child hierarchy (expected)
- **Duplicate** `["crewai", "openai", "litellm"]` → LiteLLM and OpenAI both fire for the same internal LLM call → duplicate LLM spans

Do NOT add both `"litellm"` and a provider-specific key (e.g. `"openai"`) when CrewAI is routing through LiteLLM internally.

---

## 7. Flush/Shutdown Gotcha

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

## 8. Debug Mode

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

## 9. Common Anti-Patterns Table

| Anti-Pattern | Why It's Wrong | Fix |
|-------------|----------------|-----|
| Wrapping `@span(kind="WORKFLOW")` in `trace()` | Redundant — `@span` already creates a span | Just call the decorated function directly |
| Using `trace()` for custom functions where `@span` would work | That's what `@span` is for | Use `@span(kind="CHAIN")` or the appropriate kind instead |
| Calling `.compile()` outside `trace()` context | Variable bindings won't be captured on the span | Move `.compile()` inside the `with trace(...)` block |
| Not listing all providers in `instrumentations` | Some LLM calls won't be traced | Add all providers your code uses |
| Mixing `mask` on `init()` and per-span | Both can coexist — per-span mask takes precedence over the global mask for that specific span | This is expected behavior, not a bug |
| Using `@span` on `StreamingResponse` endpoints | Decorator closes span when function returns, before async generator produces data | Use `trace()` inside the generator body instead |
| Setting `input.value` as JSON for manual LLM spans | Dashboard won't render structured message views | Use flat indexed attributes: `llm.input_messages.0.message.role` etc. |
| Using `tool_name` attribute with `trace()` | Dashboard expects `tool.name` (dotted) | Use `span.set_attribute("tool.name", "my_tool")` |
| Using `@span(kind="RERANKER")` or `@span(kind="VECTOR_STORE")` | `@span()` raises `ValueError` for these kinds | Use `trace("name", kind="RERANKER")` or `trace("name", kind="VECTOR_STORE")` instead |

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
