"""
Run:
  set -a; source <staging .env with AZURE_*>; set +a
  NEATLOGS_API_KEY=<key> python examples/sdk_examples/multiturn_chat_e2e.py mt-py-wrapper
  NEATLOGS_API_KEY=<key> python examples/sdk_examples/multiturn_chat_e2e.py mt-py-decorator
"""

import os
import sys

import neatlogs

ENDPOINT = os.getenv("NEATLOGS_ENDPOINT")
API_KEY = os.getenv("NEATLOGS_API_KEY", "").strip()
TURNS = 3


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


def run_wrapper():
    neatlogs.init(api_key=API_KEY, endpoint=ENDPOINT, workflow_name="mt-py-wrapper")
    client = neatlogs.wrap(azure_client())
    for i in range(1, TURNS + 1):
        with neatlogs.identify(
            session_id="mt_py_conv1",
            end_user_id="mt_py_user1",
            end_user_metadata={"plan": "pro"},
        ):
            out = llm_call(client, f"Turn {i}: say hi in 3 words.")
            print(f"    wrapper turn {i}: {out!r}")
    neatlogs.flush()
    neatlogs.shutdown()


def run_decorator():
    neatlogs.init(api_key=API_KEY, endpoint=ENDPOINT, workflow_name="mt-py-decorator")

    @neatlogs.span(
        kind="WORKFLOW",
        name="turn",
        session_id="mt_py_conv2",
        end_user_id="mt_py_user2",
        end_user_metadata={"plan": "team"},
    )
    def turn(i):
        @neatlogs.span(kind="TOOL", tool_name="noop")
        def noop():
            return "ok"

        return noop()

    for i in range(1, TURNS + 1):
        turn(i)
        print(f"    decorator turn {i}: done")
    neatlogs.flush()
    neatlogs.shutdown()


SCENARIOS = {
    "mt-py-wrapper": run_wrapper,
    "mt-py-decorator": run_decorator,
}


def main():
    if not API_KEY:
        sys.exit("NEATLOGS_API_KEY required")
    which = sys.argv[1] if len(sys.argv) > 1 else ""
    fn = SCENARIOS.get(which)
    if not fn:
        sys.exit(f"usage: {sys.argv[0]} <{'|'.join(SCENARIOS)}>")
    print(f"--- scenario {which} ({TURNS} turns) ---")
    fn()
    print(f"    done: {which}")


if __name__ == "__main__":
    main()
