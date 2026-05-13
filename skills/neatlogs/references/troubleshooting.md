# Troubleshooting — NeatLogs SDK Reference

Common mistakes, anti-patterns, and diagnostic steps for NeatLogs SDK instrumentation. This file is meant to be consumed by AI agents debugging integration issues in user code — many of the steps below are intentionally prescriptive so that the agent can self-serve without a round-trip to a human.

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

---

## 4. CrewAI Instrumentation Key Selection

When using CrewAI, pair `"crewai"` with the provider key that matches your `crewai.LLM(model=...)`:

| `crewai.LLM(model=...)` | Correct `instrumentations=[...]` |
|---|---|
| `"gpt-4o"` (OpenAI proper) | `["crewai", "openai"]` |
| `"azure/..."` | `["crewai", "azure_ai_inference"]` |
| `"gemini/..."` | `["crewai", "google_genai"]` |
| `"claude-..."` | `["crewai", "anthropic"]` |

> Don't add `"litellm"` alongside a direct provider key — the two can double-fire and produce duplicate LLM spans.

### Few-shot examples

```python
# Example 1: CrewAI + Azure OpenAI
neatlogs.init(instrumentations=["crewai", "azure_ai_inference"])
llm = LLM(model="azure/gpt-5-nano", base_url=..., api_key=..., api_version=...)

# Example 2: CrewAI + OpenAI proper
neatlogs.init(instrumentations=["crewai", "openai"])
llm = LLM(model="gpt-4o")

# Example 3: CrewAI + Gemini
neatlogs.init(instrumentations=["crewai", "google_genai"])
llm = LLM(model="gemini/gemini-2.5-flash")
```

---

## 5. Flush / Shutdown Gotcha

Scripts (not long-running servers) **must** call `neatlogs.flush()` then `neatlogs.shutdown()` before exit — without them the last batch of spans may not be exported.

```python
# At the end of your script
neatlogs.flush()
neatlogs.shutdown()
```

### Long-Running Servers (FastAPI / Flask)

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

**Why?** `flush()` on every request sends one HTTP batch per request instead of one every 5 seconds — this risks API throttling and adds latency. Spans batch automatically via `flush_interval` (default 5 s).

### Async Gotcha

`flush()` and `shutdown()` are **synchronous** and will block the event loop if called directly in async code. Use `asyncio.to_thread()`:

```python
import asyncio
import neatlogs

async def main():
    # ... your async application code ...

    # DON'T — blocks the event loop
    # neatlogs.flush()

    # DO — run sync calls in a worker thread
    await asyncio.to_thread(neatlogs.flush)
    await asyncio.to_thread(neatlogs.shutdown)

asyncio.run(main())
```

---

## 6. Common Anti-Patterns Table

| Anti-Pattern | Why It's Wrong | Fix |
|---|---|---|
| Wrapping `@span(kind="WORKFLOW")` in `trace()` | Redundant — `@span` already creates a span | Just call the decorated function directly |
| Using `trace()` for custom functions where `@span` would work | That's what `@span` is for | Use `@span(kind="CHAIN")` or the appropriate kind instead |
| Calling `.compile()` outside `trace()` context | Variable bindings won't be captured on the span | Move `.compile()` inside the `with trace(...)` block |
| Not listing all providers in `instrumentations` | Some LLM calls won't be traced | Add all providers your code uses (see §4 for CrewAI) |
| Mixing `mask` on `init()` and per-span | Per-span mask takes precedence over the global mask for that span | Expected behavior, not a bug |
| Setting `input.value` as a JSON blob on an auto-instrumented LLM span | Dashboard won't render structured prompt views | Use `SystemPromptTemplate` / `UserPromptTemplate` and call `.compile()` inside `trace(kind="LLM")` |
| Using `tool_name` as a manual span attribute with `trace()` | The decorator already wires this — `@span(kind="TOOL", tool_name=...)` sets `tool.name` on the span automatically | Use the `@span(kind="TOOL", tool_name="my_tool")` form instead of `trace()` + `set_attribute` |

---

## 7. Data Masking

For the full client-side masking example and server-side PII redaction configuration, see the [Data Masking and PII section in SKILL.md](../SKILL.md#data-masking-and-pii).

**Per-span mask override**: pass `mask=fn` to `@span(mask=fn)` or `trace(..., mask=fn)` to override the global mask for a specific span:

```python
@neatlogs.span(kind="TOOL", tool_name="lookup_user", mask=redact_pii)
def lookup_user(email: str) -> dict:
    return db.find_user(email)
```

> Per-span mask takes precedence — the global `init(mask=fn)` is skipped for that specific span.
