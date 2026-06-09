"""
OpenRouter Python example with Neatlogs.

Verifies neatlogs.wrap(openrouter.OpenRouter(...)) traces the Chat Completions
API (client.chat.send / send_async) with provider="openrouter", system=<model
vendor>, full input/output, token usage, and invocation parameters (temperature,
top_p, max_tokens — surfaced as model_settings in the UI).

Run:
    python examples/sdk_examples/openrouter_basic.py

Env:
    OPENROUTER_API_KEY (required)
    OPENROUTER_MODEL   (default: openai/gpt-4o-mini)
    NEATLOGS_API_KEY, NEATLOGS_ENDPOINT (default http://localhost:4100)
"""

import os

import neatlogs
from openrouter import OpenRouter


def main() -> None:
    neatlogs.init(
        api_key=os.getenv("NEATLOGS_API_KEY", ""),
        endpoint=os.getenv("NEATLOGS_ENDPOINT", "http://localhost:4100"),
        workflow_name="openrouter-basic-py",
        tags=["openrouter", "python", "basic"],
    )

    model = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
    client = neatlogs.wrap(OpenRouter(api_key=os.getenv("OPENROUTER_API_KEY", "")))

    # wrap() auto-creates a WORKFLOW root, so each call renders on its own — no
    # manual trace() wrapper needed. (To group several calls into ONE trace,
    # decorate an entry function with @neatlogs.span(kind="WORKFLOW").)
    _run(client, model)

    neatlogs.flush()
    neatlogs.shutdown()


def _run(client, model):
    print("--- chat.send (non-streaming) ---")
    # Pass sampling params so they land in the UI as model_settings.
    resp = client.chat.send(
        model=model,
        messages=[{"role": "user", "content": "In one sentence, what is OpenRouter?"}],
        temperature=0.3,
        top_p=0.9,
        max_tokens=256,
    )
    print(resp.choices[0].message.content)

    print("\n--- chat.send (streaming) ---")
    stream = client.chat.send(
        model=model,
        messages=[{"role": "user", "content": "List three uses for distributed tracing."}],
        temperature=0.5,
        max_tokens=256,
        stream=True,
    )
    for chunk in stream:
        choices = getattr(chunk, "choices", None) or []
        if choices:
            delta = getattr(choices[0], "delta", None)
            text = getattr(delta, "content", None) if delta else None
            if text:
                print(text, end="", flush=True)
    print()


if __name__ == "__main__":
    main()
