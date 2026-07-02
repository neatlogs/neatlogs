# Multi-turn support chat — sessions + evolving end-user metadata

A realistic **multi-turn** conversation with a SaaS support bot ("SaaS Genius"),
built on the Google GenAI SDK and instrumented with Neatlogs. It exercises the
**session** and **end-user identity** features end-to-end — ideal for verifying
the Sessions UI, the session timeline, and last-seen-wins end-user metadata.

## What it does

One customer, "Dave", talks to the bot across **5 turns** in **one session**:

| Turn | User asks | Tools called | End-user state |
|------|-----------|--------------|----------------|
| 1 | Annual price of Pro? | `get_pricing_plans` | plan=free, tier=basic |
| 2 | Does Pro include Custom Dashboards? | `get_feature_details` | plan=free, tier=basic |
| 3 | What plan am I on / how many projects? | `get_user_subscription_info` | plan=free, tier=basic |
| 4 | Upgrade me to Pro annual | `upgrade_subscription` | **plan=pro, tier=premium** |
| 5 | Open a ticket — dashboard is slow | `create_support_ticket` | **active_ticket_id set** |

The bot keeps the conversation history across turns, so later turns genuinely
depend on earlier ones (it remembers the plan and the price it quoted).

## How it maps to Neatlogs

- **Session** — every turn is its own WORKFLOW-root trace, but all share one
  `session_id` (`conv_dave_<ts>`). The dashboard groups them into a single
  conversation timeline.
- **End-user** — each turn's root carries `end_user_id="u_dave"` plus the
  end-user's metadata **as of that turn**. The metadata **evolves**: the plan
  upgrades (free → pro), the support tier rises (basic → premium), and a ticket
  id appears on the last turn. The `end_users` catalog reflects the **latest**
  seen metadata (last-seen-wins).
- **Agentic content** — each turn runs a real Gemini function-calling loop; each
  tool is a `TOOL` span under the turn, and the Gemini calls are auto-captured
  via the `google_genai` instrumentation.

Identity is set on the **trace root only** (the `@neatlogs.span(kind="WORKFLOW", …)`
decorator), mirroring how session/end-user work across all Neatlogs SDKs.

## Run

```bash
cd examples/sdk_examples/support_chat_multiturn
cp -n .env .env 2>/dev/null || true   # .env is present; fill in the keys
pip install -r requirements.txt
python main.py
```

Set `NEATLOGS_API_KEY` (project key `nl_…`) and `GOOGLE_API_KEY` in `.env`. To
send to a local/self-hosted backend, also set `NEATLOGS_ENDPOINT`
(e.g. `http://localhost:4100`).

## Verify

In the UI: open **Sessions** → the `conv_dave_…` session shows **5 traces** on
one timeline; each trace has tool spans; the end-user `u_dave` shows metadata
`plan=pro, support_tier=premium, active_ticket_id=TICKET-XYZ-123`.

In Postgres:

```sql
-- one row per turn, all sharing one sdk_session_id, each end_user_id='u_dave'
SELECT session_id, sdk_session_id, end_user_id FROM traces
  WHERE sdk_session_id = 'conv_dave_<ts>' ORDER BY started_at;

-- one end_users row, metadata = LATEST turn (pro / premium / ticket set)
SELECT workflow_name, end_user_id, metadata, first_seen_at, last_seen_at
  FROM end_users WHERE end_user_id = 'u_dave';
```
