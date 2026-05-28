# Support Copilot Demo

Flagship NeatLogs demo: classify → RAG → draft reply → send email. Three scripted run modes produce different trace stories (broken email API, silent KB/prompt bug, fixed path).

## Setup

```bash
cd examples/sdk_examples/support_copilot_demo
cp .env.example .env   # fill in NeatLogs + Azure keys
pip install -r requirements.txt
```

## Demo run sequence

**1. Broken email (intentional SendGrid 401 + screenshot in workflow input)**

```bash
RUN=A python support_copilot.py
```

**2. Silent refund-window bug (missing KB v3.0 + hardcoded prompt v3)**

```bash
SENDGRID_FAKE_SUCCESS=1 RUN=B python support_copilot.py
```

`SENDGRID_FAKE_SUCCESS=1` stubs delivery via httpbin so you don't need a real SendGrid key; the `requests` span still appears.

**3. Fixed path (same ticket, v4 prompt + full KB)**

```bash
SENDGRID_FAKE_SUCCESS=1 RUN=B_FIXED python support_copilot.py
```

## Run modes

| `RUN` | Story | Expected outcome |
|---|---|---|
| `A` | Broken SendGrid key | TOOL span fails with 401 body in logs |
| `B` | Stale prompt + KB missing v3.0 | Trace succeeds but reply cites wrong refund window |
| `B_FIXED` | Prompt v4 + full KB | Trace succeeds with correct policy citation |

`RUN=A` exits 0 even on `EmailDeliveryError` — the failure is intentional and captured in the trace.

## What's instrumented

| Span / signal | How |
|---|---|
| WORKFLOW, AGENT, RETRIEVER, TOOL | `@neatlogs.span(kind=...)` |
| Azure OpenAI | `instrumentations=["openai"]` |
| Chroma query | `instrumentations=["chromadb"]` |
| SendGrid / httpbin POST | `instrumentations=["requests"]` |
| Step logs | `capture_logs=True` + `neatlogs.log()` |
| Prompt templates | `SystemPromptTemplate` / `UserPromptTemplate` in `neatlogs.trace()` |
| PII redaction | `pii_enabled=True` (server-side) |
| Ticket screenshot | Markdown + embedded PNG data URL on WORKFLOW input |

## Assets

PNG fixtures live in `assets/`. Re-render from HTML only if sources change:

```bash
pip install playwright
playwright install chromium
cd assets && python make_assets.py
```

KB URLs point at `KB_SITE_BASE_URL` (default: GitHub Pages demo KB).
