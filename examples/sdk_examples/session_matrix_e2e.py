"""
Run:
  set -a; source <staging .env with AZURE_*>; set +a
  NEATLOGS_API_KEY=<key> \
  AZURE_API_KEY=.. AZURE_API_BASE=.. AZURE_API_VERSION=.. AZURE_API_DEPLOYMENT_NAME=.. \
    python examples/sdk_examples/session_matrix_e2e.py
"""

import os
import sys

import neatlogs

ENDPOINT = os.getenv("NEATLOGS_ENDPOINT")
API_KEY = os.getenv("NEATLOGS_API_KEY", "").strip()
PREFIX = "py-matrix"


def azure_client():
    from openai import AzureOpenAI

    return AzureOpenAI(
        api_key=os.environ["AZURE_API_KEY"],
        azure_endpoint=os.environ["AZURE_API_BASE"],
        api_version=os.environ["AZURE_API_VERSION"],
    )


def llm_call(client, prompt: str) -> str:
    deployment = os.environ["AZURE_API_DEPLOYMENT_NAME"]
    # gpt-5-nano is a reasoning model: needs generous max_completion_tokens.
    resp = client.chat.completions.create(
        model=deployment,
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=2000,
    )
    return resp.choices[0].message.content or ""


def reinit(**overrides):
    """Re-init the SDK for a scenario (shutdown any prior session first)."""
    try:
        neatlogs.shutdown()
    except Exception:
        pass
    base = dict(api_key=API_KEY, endpoint=ENDPOINT)
    base.update(overrides)
    neatlogs.init(**base)


def scenario_1_wrapper_only():
    # Wrapper-only: identity comes from neatlogs.identify() (NOT init). The
    # auto-root created inside wrap() picks up session + end-user from context.
    reinit(workflow_name=f"{PREFIX}-s1")
    client = neatlogs.wrap(azure_client())
    with neatlogs.identify(session_id="s1_session", end_user_id="s1_user",
                           end_user_metadata={"plan": "pro"}):
        llm_call(client, "Say hi in 3 words.")
    neatlogs.flush()


def scenario_2_wrapper_plus_decorator():
    reinit(workflow_name=f"{PREFIX}-s2")
    client = neatlogs.wrap(azure_client())

    @neatlogs.span(kind="WORKFLOW", name=f"{PREFIX}-s2",
                   session_id="s2_session", end_user_id="s2_user")
    def turn():
        return llm_call(client, "Say bye in 3 words.")

    turn()
    neatlogs.flush()


def scenario_3_decorator_only():
    reinit(workflow_name=f"{PREFIX}-s3")

    @neatlogs.span(kind="WORKFLOW", name=f"{PREFIX}-s3", session_id="s3_session")
    def turn():
        @neatlogs.span(kind="TOOL", tool_name="noop")
        def tool():
            return "ok"
        return tool()

    turn()
    neatlogs.flush()


def scenario_4_workflow():
    reinit(workflow_name=f"{PREFIX}-s4")

    @neatlogs.span(kind="WORKFLOW", name=f"{PREFIX}-s4", session_id="s4_session")
    def wf():
        return "done"

    wf()
    neatlogs.flush()


def scenario_5_multiturn_not_workflow():
    # trace() context manager (not a @span workflow), multi-turn one session.
    reinit(workflow_name=f"{PREFIX}-s5")
    for q in ("turn one", "turn two", "turn three"):
        with neatlogs.trace("chat_turn", session_id="s5_session"):
            pass
    neatlogs.flush()


def scenario_6_enduser_per_session():
    # Two sessions, each a unique end-user (user is unique per session).
    reinit(workflow_name=f"{PREFIX}-s6")
    with neatlogs.trace("turn", session_id="s6_sessionA", end_user_id="s6_userA",
                        end_user_metadata={"plan": "free"}):
        pass
    with neatlogs.trace("turn", session_id="s6_sessionB", end_user_id="s6_userB",
                        end_user_metadata={"plan": "pro"}):
        pass
    neatlogs.flush()


def scenario_7_no_session():
    reinit(workflow_name=f"{PREFIX}-s7")

    @neatlogs.span(kind="WORKFLOW", name=f"{PREFIX}-s7")
    def wf():
        return "no session set"

    wf()
    neatlogs.flush()


def scenario_8_session_context_and_root():
    # session via identify() context; a root decorator overrides it -> per-call wins.
    reinit(workflow_name=f"{PREFIX}-s8")
    with neatlogs.identify(session_id="s8_ctx_session"):
        @neatlogs.span(kind="WORKFLOW", name=f"{PREFIX}-s8", session_id="s8_root_session")
        def wf():
            return "override"

        wf()
        # And one with no per-call session -> inherits the identify() context.
        @neatlogs.span(kind="WORKFLOW", name=f"{PREFIX}-s8b")
        def wf2():
            return "inherit"

        wf2()
    neatlogs.flush()


def scenario_9_enduser_root_and_context():
    # end-user via identify() context AND a root override -> per-call wins.
    reinit(workflow_name=f"{PREFIX}-s9")
    with neatlogs.identify(end_user_id="s9_ctx_user", end_user_metadata={"plan": "free"}):
        @neatlogs.span(kind="WORKFLOW", name=f"{PREFIX}-s9", session_id="s9_session",
                       end_user_id="s9_root_user", end_user_metadata={"plan": "enterprise"})
        def wf():
            return "root enduser"

        wf()
    neatlogs.flush()


SCENARIOS = [
    ("1 wrapper-only", scenario_1_wrapper_only),
    ("2 wrapper+decorator", scenario_2_wrapper_plus_decorator),
    ("3 decorator-only", scenario_3_decorator_only),
    ("4 workflow", scenario_4_workflow),
    ("5 multiturn-not-workflow", scenario_5_multiturn_not_workflow),
    ("6 enduser-per-session", scenario_6_enduser_per_session),
    ("7 no-session", scenario_7_no_session),
    ("8 session-context-and-root", scenario_8_session_context_and_root),
    ("9 enduser-root-and-context", scenario_9_enduser_root_and_context),
]


def main():
    if not API_KEY:
        sys.exit("NEATLOGS_API_KEY required")
    only = sys.argv[1] if len(sys.argv) > 1 else None
    for label, fn in SCENARIOS:
        if only and not label.startswith(only):
            continue
        print(f"--- scenario {label} ---")
        fn()
        print(f"    done: {label}")
    try:
        neatlogs.shutdown()
    except Exception:
        pass
    print(f"\nAll scenarios emitted. Verify rows WHERE workflow_name LIKE '{PREFIX}-%'.")


if __name__ == "__main__":
    main()
