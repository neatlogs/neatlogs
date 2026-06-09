"""
Vertex AI Python example with Neatlogs.

Verifies neatlogs.wrap(google.genai.Client(vertexai=True, ...)) traces the
generate_content + generate_content_stream APIs with provider/system="vertexai",
full input/output, token usage, and invocation parameters (temperature, top_p,
max_output_tokens — surfaced as model_settings in the UI).

Auth — two modes (auto-detected from env):
  - Express mode (API key): set GOOGLE_API_KEY. No project/location needed.
    Express keys route to a fixed region, so use a model like gemini-2.5-flash
    (gemini-2.0-flash 404s on Express keys).
  - ADC mode (service account): set GOOGLE_CLOUD_PROJECT + GOOGLE_CLOUD_LOCATION
    and authenticate via Application Default Credentials.

Run:
    python examples/sdk_examples/vertex_ai_basic.py

Env:
    GOOGLE_API_KEY                          (Express mode), OR
    GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_LOCATION (ADC mode)
    VERTEX_MODEL   (default: gemini-2.5-flash)
    NEATLOGS_API_KEY, NEATLOGS_ENDPOINT (default http://localhost:4100)
"""

import os

import neatlogs
from google import genai
from google.genai import types


def _make_client():
    """Express mode if GOOGLE_API_KEY is set, else ADC (project/location)."""
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_CLOUD_API_KEY")
    if api_key:
        return genai.Client(vertexai=True, api_key=api_key)
    return genai.Client(
        vertexai=True,
        project=os.getenv("GOOGLE_CLOUD_PROJECT"),
        location=os.getenv("GOOGLE_CLOUD_LOCATION", "global"),
    )


def main() -> None:
    neatlogs.init(
        api_key=os.getenv("NEATLOGS_API_KEY", ""),
        endpoint=os.getenv("NEATLOGS_ENDPOINT", "http://localhost:4100"),
        workflow_name="vertex-ai-basic-py",
        tags=["vertex_ai", "python", "basic"],
    )

    model = os.getenv("VERTEX_MODEL", "gemini-2.5-flash")
    client = neatlogs.wrap(_make_client())

    # wrap() auto-creates a WORKFLOW root, so each call renders on its own — no
    # manual trace() wrapper needed. (To group several calls into ONE trace,
    # decorate an entry function with @neatlogs.span(kind="WORKFLOW").)
    _run(client, model)

    neatlogs.flush()
    neatlogs.shutdown()


def _run(client, model):
    # Pass sampling params so they land in the UI as model_settings.
    config = types.GenerateContentConfig(temperature=0.3, top_p=0.9, max_output_tokens=256)

    print("--- generate_content (non-streaming) ---")
    resp = client.models.generate_content(
        model=model,
        contents="In one sentence, what is Vertex AI?",
        config=config,
    )
    print(resp.text)

    print("\n--- generate_content_stream (streaming) ---")
    stream = client.models.generate_content_stream(
        model=model,
        contents="List three benefits of distributed tracing.",
        config=config,
    )
    for chunk in stream:
        if getattr(chunk, "text", None):
            print(chunk.text, end="", flush=True)
    print()


if __name__ == "__main__":
    main()
