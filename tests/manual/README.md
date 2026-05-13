# Pending Tests

These 5 scripts correspond to PR #6 comments that need testing before the threads can be marked resolved.

## Setup

```bash
cd <repo-root>
pip install -e ".[dev]"
export NEATLOGS_API_KEY=<your-key>
```

---

## Test 1 — Async flush/shutdown (`test_async_flush.py`)

**PR comment**: troubleshooting.md lines 148–161 — "have you tested this?"

**What it tests**: `await asyncio.to_thread(neatlogs.flush)` and `await asyncio.to_thread(neatlogs.shutdown)` complete without blocking the event loop or raising errors.

**What to check in the dashboard**: A CHAIN span appears under workflow `test-async-flush`.

```bash
python tests/manual/test_async_flush.py
```

**Pass if**: Prints `PASS` with no exceptions.

---

## Test 2 — Async `trace()` + SystemPromptTemplate (`test_async_trace.py`)

**PR comment**: troubleshooting.md line 172 — "same question as above"

**What it tests**: `with neatlogs.trace(prompt_template=..., user_prompt_template=...)` works correctly inside an `async def`. Template and variables are captured on the span.

**What to check in the dashboard**: A CHAIN span under workflow `test-async-trace` with `llm.prompt_template` and `llm.user_prompt_template` attributes visible.

```bash
python tests/manual/test_async_trace.py
```

**Pass if**: Prints `PASS` with no exceptions.

---

## Test 3 — Manual LLM span attributes (`test_manual_llm_span.py`)

**PR comment**: decorators-and-traces.md lines 191–194 — "have you tested this? If it is manual that is"

**What it tests**: Setting `llm.input_messages.N.message.role/content`, `llm.output_messages.N.message.role/content`, `llm.model_name`, and `llm.token_count.*` directly on a span via `span.set_attribute()`. These use the `llm.*` prefix (not `neatlogs.llm.*`) — the SDK normalises them before export.

**What to check in the dashboard**: An LLM span under workflow `test-manual-llm-span` showing:
- Input messages: system + user
- Output message: assistant
- Model: `gpt-4o`
- Token counts: prompt=20, completion=10, total=30
- Attributes stored as `neatlogs.llm.*` (confirming normalisation)

```bash
python tests/manual/test_manual_llm_span.py
```

**Pass if**: Prints `PASS` with no exceptions, and the dashboard shows the expected attributes.

---

## Test 4 — Client-side PII masking (`test_pii_masking.py`)

**PR comment**: SKILL.md line 288 — "Have you tested this example?"

**What it tests**: The `mask=` callback passed to `neatlogs.init()` is applied before export, redacting span attributes whose keys contain "email" or "password".

**What to check in the dashboard**: A CHAIN span under workflow `test-pii-masking` where:
- `input.user_email` → `[REDACTED]`
- `input.password` → `[REDACTED]`
- `input.query` → `"What is the weather?"` (unchanged)

```bash
python tests/manual/test_pii_masking.py
```

**Pass if**: Prints `PASS` and the dashboard shows `[REDACTED]` for the PII fields.

---

## Test 5 — LiteLLM integration (`test_litellm.py`)

**PR comment**: framework-integrations.md line ~390 — "same comment as above and did you test this?"

**What it tests**: `instrumentations=["litellm"]` auto-instruments LiteLLM, and the `@neatlogs.span(kind="WORKFLOW")` + `neatlogs.trace("llm_call", kind="LLM", prompt_template=..., user_prompt_template=...)` pattern works end-to-end.

**Prerequisites**:
```bash
pip install "neatlogs[litellm]"
export OPENAI_API_KEY=<your-openai-key>   # or any LiteLLM-supported provider key
```

**What to check in the dashboard**: Workflow `test-litellm` shows:
- WORKFLOW span `run`
  - LLM span `llm_call` with prompt template + variables
    - Auto-instrumented LiteLLM child span

```bash
NEATLOGS_API_KEY=<your-key> OPENAI_API_KEY=<openai-key> python tests/manual/test_litellm.py
```

**Pass if**: Prints `PASS` and the dashboard shows the full span hierarchy.
