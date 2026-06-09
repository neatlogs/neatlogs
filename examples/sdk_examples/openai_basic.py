"""
OpenAI Python example with Neatlogs.

Wrap the OpenAI client with ``neatlogs.wrap()`` and every chat / responses /
embeddings call is captured as an ``LLM`` span. The wrapped call self-roots — a
WORKFLOW root is opened automatically — so a single call renders with no extra
wrapper. (To group several calls into ONE trace, decorate an entry function
with ``@neatlogs.span(kind="WORKFLOW")``.)

This example points the OpenAI SDK at OpenRouter (OpenAI-compatible) so it runs
with just an OpenRouter key, but the instrumentation is identical for api.openai.com.

Run:
    python examples/sdk_examples/openai_basic.py

Env:
    OPENROUTER_API_KEY (required here; or OPENAI_API_KEY + drop base_url for real OpenAI)
    OPENAI_MODEL       (default: openai/gpt-4o-mini)
    NEATLOGS_API_KEY, NEATLOGS_ENDPOINT (default http://localhost:4100)
"""

import os

import neatlogs
from openai import OpenAI


def main() -> None:
    neatlogs.init(
        api_key=os.getenv("NEATLOGS_API_KEY", ""),
        endpoint=os.getenv("NEATLOGS_ENDPOINT", "http://localhost:4100"),
        workflow_name="openai-basic-py",
        tags=["openai", "python", "basic"],
    )

    model = os.getenv("OPENAI_MODEL", "openai/gpt-4o-mini")
    client = neatlogs.wrap(
        OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY", ""),
        )
    )

    print("--- chat.completions.create (non-streaming) ---")
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "In one sentence, what is OpenAI?"}],
        temperature=0.3,
        max_tokens=256,
    )
    print(resp.choices[0].message.content)

    print("\n--- chat.completions.create (streaming) ---")
    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "List three uses for tracing."}],
        max_tokens=256,
        stream=True,
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print()

    neatlogs.flush()
    neatlogs.shutdown()


if __name__ == "__main__":
    main()
