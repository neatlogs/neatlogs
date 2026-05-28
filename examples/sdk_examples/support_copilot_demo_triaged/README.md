# Support Copilot Demo (Triaged)

Same pipeline as [`support_copilot_demo/`](../support_copilot_demo/), after applying Triage-style fixes:

- KB articles carry `status` metadata; retriever excludes `archived` docs
- Prompt v3 no longer hardcodes a refund window
- `RUN=B` uses the full KB seed (v3.0 present) — compare against the pre-triage demo

## Setup

```bash
cd examples/sdk_examples/support_copilot_demo_triaged
cp .env.example .env
pip install -r requirements.txt
```

## Demo run sequence

**1. Broken email (same as pre-triage demo)**

```bash
RUN=A python support_copilot.py
```

**2. Post-triage RUN=B (archived filter + updated prompt v3)**

```bash
SENDGRID_FAKE_SUCCESS=1 RUN=B python support_copilot.py
```

Run the same command in `support_copilot_demo/` first to show the silent bug, then here to show the triaged fix.

**3. Optional: prompt v4 hardening**

```bash
SENDGRID_FAKE_SUCCESS=1 RUN=B_FIXED python support_copilot.py
```

## Difference from `support_copilot_demo/`

| Area | Pre-triage | Triaged (this folder) |
|---|---|---|
| `workflow_name` | `support-copilot` | `support-copilot-triaged` |
| KB metadata | no `status` field | `current` / `archived` |
| Retriever | all docs | `where status != archived` |
| Prompt v3 | hardcodes 30-day window | reads window from KB |
| `RUN=B` KB seed | missing v3.0 | full `SEEDS_NORMAL` |

See [`support_copilot_demo/README.md`](../support_copilot_demo/README.md) for span-type and instrumentation details.
