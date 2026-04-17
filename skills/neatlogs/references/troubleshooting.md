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

1. **Is `neatlogs.init()` called?** → No → Add `neatlogs.init(...)` at the top of your application.
2. **Is it called BEFORE LLM library imports?** → No → Move `neatlogs.init()` before `import openai` / `import anthropic` / etc.
3. **Is the provider listed in `instrumentations=[]`?** → No → Add it (e.g. `instrumentations=["openai"]`).
4. **Is `NEATLOGS_API_KEY` set?** → No → Set it via env var or `api_key=` param. Without it, export is **silently disabled** with no error.
5. **Is `disable_export=True`?** → Yes → Remove it or set to `False`.
6. **Still missing?** → Enable `debug=True` in `neatlogs.init()` and check stderr output for clues.

```python
# Enable debug mode to diagnose
neatlogs.init(
    api_key="...",
    instrumentations=["openai"],
    debug=True,  # Verbose logging to stderr
)
```

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

**Alternative workaround**: Filter out self-referencing HTTP spans server-side in the NeatLogs dashboard, or ensure your service doesn't recursively call its own endpoints within a traced context.

---

## 5. Duplicate Span Issues

- When using CrewAI with `instrumentations=["crewai", "openai"]`, both CrewAI and OpenAI instrumentations may fire for the same LLM call. This is **expected** — CrewAI spans wrap the OpenAI LLM spans in a parent-child hierarchy.
- Do NOT add both `"litellm"` and provider-specific instrumentations (e.g. `"openai"`) if CrewAI is managing LLM calls through LiteLLM internally — this can create duplicate LLM spans.

---

## 6. Flush/Shutdown Gotcha

Scripts (not long-running servers) **MUST** call `neatlogs.flush()` then `neatlogs.shutdown()` before exit. Without this, the last batch of spans may be lost because the `BatchSpanProcessor` hasn't flushed yet.

An `atexit` handler is registered automatically, but explicit flush is recommended for reliability.

```python
# At the end of your script
neatlogs.flush()
neatlogs.shutdown()
```

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

## 8. Error Tracking on Manual Spans

Setting `span.set_attribute("error", str(e))` does **NOT** mark the span as failed. You must use `span.record_exception(e)` + `span.set_status(Status(StatusCode.ERROR, str(e)))`.

See [`decorators-and-traces.md` §6](decorators-and-traces.md#6-error-handling-on-manual-spans) for the full pattern and code example. Note that `@span()` handles this automatically — manual error handling is only needed inside `trace()` blocks.

---

## 9. Common Anti-Patterns Table

| Anti-Pattern | Why It's Wrong | Fix |
|-------------|----------------|-----|
| Wrapping `@span(kind="WORKFLOW")` in `trace()` | Redundant — `@span` already creates a span | Just call the decorated function directly |
| Using `trace()` for custom functions | That's what `@span` is for (unless you need RERANKER/VECTOR_STORE kind) | Use `@span(kind="CHAIN")` or appropriate kind |
| Calling `.compile()` outside `trace()` context | Variable bindings won't be captured on the span | Move `.compile()` inside the `with trace(...)` block |
| Not listing all providers in `instrumentations` | Some LLM calls won't be traced | Add all providers your code uses |
| Mixing `mask` on `init()` and per-span | Per-span mask takes precedence; global mask is skipped for that span | Use one or the other consistently |
| Using `@span` on `StreamingResponse` endpoints | Decorator closes span when function returns, before async generator produces data | Use `trace()` inside the generator body instead |
| Setting `input.value` as JSON for manual LLM spans | Dashboard won't render structured message views | Use flat indexed attributes: `llm.input_messages.0.message.role` etc. |
| Using `tool_name` attribute with `trace()` | Dashboard expects `tool.name` (dotted) | Use `span.set_attribute("tool.name", "my_tool")` |
| Using `@span(kind="RERANKER")` or `@span(kind="VECTOR_STORE")` | `@span()` raises `ValueError` for these kinds | Use `trace("name", kind="RERANKER")` or `trace("name", kind="VECTOR_STORE")` instead |

> For error handling anti-patterns, see [§8 above](#8-error-tracking-on-manual-spans).

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
