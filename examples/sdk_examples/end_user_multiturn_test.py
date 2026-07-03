"""
Run:
    NEATLOGS_API_KEY=<local key> python examples/sdk_examples/end_user_multiturn_test.py
"""

import os
import sys
import time

import neatlogs

WORKFLOW = "support-chat-multiturn"
END_USER = "u_dave"
# One session id for the whole conversation. Session/end-user identity is
# per-request — declared at the trace root (here the @span WORKFLOW decorator),
# NOT on init(). Every turn passes the same SESSION_ID → one grouped session.
SESSION_ID = f"conv_{int(time.time())}"

# One conversation, several turns. Metadata evolves across turns (free → pro).
TURNS = [
    {"q": "How do I reset my password?", "metadata": {"plan": "free", "email": "dave@acme.test"}},
    {"q": "The reset link isn't working.", "metadata": {"plan": "free", "email": "dave@acme.test"}},
    {"q": "I just upgraded — can you help faster now?", "metadata": {"plan": "pro", "email": "dave@acme.test"}},
]


# Per-turn WORKFLOW root via the DECORATOR. end_user_id/metadata are set on the
# decorator (the trace root) — child spans inside inherit nothing extra. A new
# decorated call with no active parent + an active session => a new root trace.
def make_turn(metadata: dict):
    @neatlogs.span(
        kind="WORKFLOW",
        name=WORKFLOW,
        session_id=SESSION_ID,
        end_user_id=END_USER,
        end_user_metadata=metadata,
    )
    def turn(question: str) -> str:
        # A child TOOL span so the trace has agentic content (won't be dropped).
        @neatlogs.span(kind="TOOL", tool_name="lookup")
        def lookup(q: str) -> str:
            return f"answer for {END_USER} ({metadata['plan']} plan)"

        return lookup(question)

    return turn


def main() -> None:
    api_key = os.getenv("NEATLOGS_API_KEY", "").strip()
    if not api_key:
        sys.exit(
            "NEATLOGS_API_KEY required.\n"
            "python examples/sdk_examples/end_user_multiturn_test.py"
        )

    endpoint = os.getenv("NEATLOGS_ENDPOINT")

    # Session/end-user identity is NOT set on init() (it's per-request). Each turn
    # declares session_id + end_user on its WORKFLOW root (see make_turn).
    neatlogs.init(
        api_key=api_key,
        endpoint=endpoint,
        workflow_name=WORKFLOW,
        tags=["end-user-multiturn"],
    )

    print(f"Session: {SESSION_ID}  end_user: {END_USER}")

    for i, turn in enumerate(TURNS, start=1):
        make_turn(turn["metadata"])(turn["q"])
        print(f"  turn {i}: plan={turn['metadata']['plan']}")

    neatlogs.flush()
    neatlogs.shutdown()

    print(
        f"\nDone. {len(TURNS)} turns, one auto session, one end-user ({END_USER}).\n"
        "Verify in Postgres:\n"
        "  -- one traces row per turn; all share ONE sdk_session_id; each end_user_id=u_dave:\n"
        f"  SELECT session_id, sdk_session_id, end_user_id FROM traces\n"
        f"  WHERE workflow_name='{WORKFLOW}' AND end_user_id='{END_USER}' ORDER BY started_at;\n"
        "  -- ONE end_users row, metadata = LATEST (pro):\n"
        f"  SELECT workflow_name, end_user_id, metadata, first_seen_at, last_seen_at\n"
        f"  FROM end_users WHERE end_user_id='{END_USER}';"
    )


if __name__ == "__main__":
    main()
