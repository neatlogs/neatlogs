"""
Multi-turn SaaS support conversation — one session, one end-user, evolving metadata.

A single customer ("Dave") talks to the "SaaS Genius" support bot across 5 turns.
Each turn is its own WORKFLOW-root trace; all turns share ONE session, so the
dashboard shows them as a single conversation timeline. The end-user's metadata
EVOLVES mid-conversation (Free -> Pro plan, Basic -> Premium tier, a ticket id
appears) — exactly what "last-seen-wins" end-user metadata is meant to capture.

Later turns depend on earlier ones (the bot remembers the plan and price it
quoted), because the bot keeps the google-genai conversation history across turns.

Usage:
    cd examples/sdk_examples/support_chat_multiturn
    cp .env.example .env    # fill in NEATLOGS_API_KEY + GOOGLE_API_KEY
    pip install -r requirements.txt
    python main.py

Required env vars:
    NEATLOGS_API_KEY, GOOGLE_API_KEY
    NEATLOGS_ENDPOINT (optional — for local/self-hosted ingest)
"""

import os
import time

from dotenv import load_dotenv

load_dotenv()

# init() MUST run before the google.genai client is created (it caches transport
# at construction), so the "google_genai" instrumentation can attach.
import neatlogs

WORKFLOW = "saas-support-chat"

neatlogs.init(
    api_key=os.getenv("NEATLOGS_API_KEY"),
    endpoint=os.getenv("NEATLOGS_ENDPOINT"),
    workflow_name=WORKFLOW,
    tags=["sdk-examples", "google-genai", "multi-turn", "session", "end-user"],
    instrumentations=["google_genai"],
)

from support_bot import SupportBot, USER_ID

# One session id for the whole conversation (use your own conversation/thread id).
SESSION_ID = f"conv_dave_{int(time.time())}"

# The scripted conversation. `metadata` is the end-user's state AS OF this turn —
# it evolves: the plan upgrades on turn 5, the support tier rises, a ticket opens.
BASE_META = {"email": "dave@acme.test", "company": "Acme Inc"}

TURNS = [
    {
        "user": "Hi! I'm on the Free plan and thinking about upgrading to Pro. "
                "Before I do — what's the annual price?",
        "metadata": {**BASE_META, "plan": "free", "support_tier": "basic",
                     "topic": "upgrade-inquiry", "active_ticket_id": None},
    },
    {
        "user": "Got it. Does that Pro plan include Custom Dashboards?",
        "metadata": {**BASE_META, "plan": "free", "support_tier": "basic",
                     "topic": "feature-question", "active_ticket_id": None},
    },
    {
        "user": "Perfect. And just to confirm — what plan am I on right now, "
                "and how many projects am I using?",
        "metadata": {**BASE_META, "plan": "free", "support_tier": "basic",
                     "topic": "account-status", "active_ticket_id": None},
    },
    {
        "user": "Great, please go ahead and upgrade me to Pro annual.",
        # Upgrade happens THIS turn → metadata flips to pro / premium.
        "metadata": {**BASE_META, "plan": "pro", "support_tier": "premium",
                     "topic": "upgrade-complete", "active_ticket_id": None},
    },
    {
        "user": "Thanks! One more thing — my project dashboard has been loading "
                "really slowly. Can you open a ticket for that?",
        # Ticket opens THIS turn → active_ticket_id set.
        "metadata": {**BASE_META, "plan": "pro", "support_tier": "premium",
                     "topic": "dashboard-issue", "active_ticket_id": "TICKET-XYZ-123"},
    },
]


def run_turn(bot: SupportBot, turn_no: int, user_message: str, metadata: dict) -> str:
    """One user turn = one WORKFLOW-root trace, stamped with the shared session +
    the end-user's CURRENT metadata (root-only; child spans inherit nothing extra)."""

    @neatlogs.span(
        kind="WORKFLOW",
        name="support_turn",
        session_id=SESSION_ID,
        end_user_id=USER_ID,
        end_user_metadata=metadata,
    )
    def _turn() -> str:
        return bot.ask(user_message)

    return _turn()


def main() -> None:
    if not os.getenv("NEATLOGS_API_KEY"):
        raise SystemExit("NEATLOGS_API_KEY required (put it in .env)")
    if not os.getenv("GOOGLE_API_KEY"):
        raise SystemExit("GOOGLE_API_KEY required (put it in .env)")

    print(f"=== SaaS support chat — session={SESSION_ID}  end_user={USER_ID} ===\n")
    bot = SupportBot()

    for i, turn in enumerate(TURNS, start=1):
        print(f"--- Turn {i}  (plan={turn['metadata']['plan']}, "
              f"tier={turn['metadata']['support_tier']}) ---")
        print(f"  user: {turn['user']}")
        reply = run_turn(bot, i, turn["user"], turn["metadata"])
        print(f"  bot : {reply.strip()}\n")

    neatlogs.flush()
    neatlogs.shutdown()

    print("Done. Verify in the UI / Postgres:")
    print(f"  - ONE session '{SESSION_ID}' with {len(TURNS)} traces (one per turn), shown as a timeline.")
    print(f"  - Each trace: end_user_id='{USER_ID}'; tool spans under each turn.")
    print("  - end_users catalog metadata = LATEST turn (plan=pro, support_tier=premium,")
    print("    active_ticket_id=TICKET-XYZ-123) — last-seen-wins across the conversation.")
    print("\n  SQL:")
    print(f"    SELECT session_id, sdk_session_id, end_user_id FROM traces")
    print(f"      WHERE sdk_session_id='{SESSION_ID}' ORDER BY started_at;")
    print(f"    SELECT workflow_name, end_user_id, metadata, first_seen_at, last_seen_at")
    print(f"      FROM end_users WHERE end_user_id='{USER_ID}';")


if __name__ == "__main__":
    main()
