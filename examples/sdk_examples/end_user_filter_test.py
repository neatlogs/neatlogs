"""
End-user identity test workflow — for verifying the "filter traces by end-user"
feature and the end_users catalog table locally.

It sends several traces, each attributed to a DIFFERENT end-user via the
per-request path `neatlogs.trace(end_user_id=..., end_user_metadata=...)`, across
TWO workflow names — so you can:
  1. Open the Traces page → Filters → "End-user" and filter by e.g. `u_alice`.
  2. Inspect Postgres `end_users` → one row per (project, workflow_name, end_user)
     with the latest metadata.

No LLM key required — each trace contains a manual TOOL span + output content so
the backend keeps it (content-free traces can be dropped).

Run:
    NEATLOGS_API_KEY=<your local project key> \
    NEATLOGS_ENDPOINT=http://localhost:4100 \
    python examples/sdk_examples/end_user_filter_test.py

Env:
    NEATLOGS_API_KEY  (required — your LOCAL project's API key)
    NEATLOGS_ENDPOINT (default http://localhost:4100 — local ingest)
"""

import os
import sys

import neatlogs


# A few end-users, each with arbitrary metadata (plan/email/region → JSON).
END_USERS = [
    {"id": "u_alice", "metadata": {"plan": "pro", "email": "alice@acme.test", "region": "us"}},
    {"id": "u_bob", "metadata": {"plan": "free", "email": "bob@acme.test", "region": "eu"}},
    {"id": "u_carol", "metadata": {"plan": "enterprise", "email": "carol@acme.test", "region": "apac"}},
]

# Two workflows so the end_users table shows the (project, workflow, user) grain.
WORKFLOWS = ["support-chat", "doc-search"]


def handle_request(workflow: str, end_user: dict, turn: int) -> None:
    """One traced request attributed to a single end-user.

    The top-level trace() is the ROOT span and carries the end-user — every child
    span inherits the trace; the backend rolls the id up to the trace + session
    and upserts the end_users catalog.
    """
    with neatlogs.trace(
        name=workflow,
        kind="WORKFLOW",
        end_user_id=end_user["id"],
        end_user_metadata=end_user["metadata"],
    ) as span:
        question = f"[{workflow}] turn {turn} from {end_user['id']}"
        span.set_attribute("input.value", question)

        # A child TOOL span so the trace has agentic content (won't be dropped).
        with neatlogs.trace(name="lookup", kind="TOOL") as tool:
            tool.set_attribute("input.value", question)
            answer = f"Resolved request for {end_user['id']} ({end_user['metadata']['plan']} plan)"
            tool.set_attribute("output.value", answer)

        span.set_attribute("output.value", answer)
        print(f"  {workflow} ← {end_user['id']} (turn {turn})")


def main() -> None:
    api_key = os.getenv("NEATLOGS_API_KEY", "").strip()
    if not api_key:
        sys.exit(
            "NEATLOGS_API_KEY is required — set it to your LOCAL project's API key.\n"
            "  e.g. NEATLOGS_API_KEY=nl_... NEATLOGS_ENDPOINT=http://localhost:4100 "
            "python examples/sdk_examples/end_user_filter_test.py"
        )

    endpoint = os.getenv("NEATLOGS_ENDPOINT", "http://localhost:4100")
    neatlogs.init(
        api_key=api_key,
        endpoint=endpoint,
        workflow_name="end-user-filter-test",  # overridden per-trace below
        tags=["end-user-test"],
        debug=True,
    )

    print(f"Sending end-user traces → {endpoint}")
    # 2 turns each, so there's volume to filter and metadata is re-seen (last-wins).
    for turn in range(1, 3):
        for workflow in WORKFLOWS:
            for end_user in END_USERS:
                handle_request(workflow, end_user, turn)

    neatlogs.flush()
    neatlogs.shutdown()

    users = ", ".join(u["id"] for u in END_USERS)
    print(
        "\nDone. In the UI: Traces → Filters → 'End-user' → try one of: "
        f"{users}\n"
        "Postgres: SELECT project_id, workflow_name, end_user_id, metadata, "
        "last_seen_at FROM end_users ORDER BY last_seen_at DESC;"
    )


if __name__ == "__main__":
    main()
