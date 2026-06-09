"""
Azure OpenAI Python example with Neatlogs.

Verifies neatlogs.wrap(AzureOpenAI(...)) traces chat completions with
provider="azure": a system+user turn, a tool call, and a streaming turn.

Run:
    python examples/sdk_examples/azure_openai_basic.py

Env (read from process env; this script reads the same names the TS example uses):
    AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_DEPLOYMENT (or AZURE_OPENAI_DEPLOYMENT_NAME)
    NEATLOGS_API_KEY, NEATLOGS_ENDPOINT (default http://localhost:4100)
    AZURE_TEMPERATURE (optional; reasoning deployments only accept the default)
"""

import json
import os

import neatlogs
from openai import AzureOpenAI


def main() -> None:
    neatlogs.init(
        api_key=os.getenv("NEATLOGS_API_KEY", ""),
        endpoint=os.getenv("NEATLOGS_ENDPOINT", "http://localhost:4100"),
        workflow_name="azure-openai-basic-py",
        tags=["azure-openai", "python", "basic"],
    )

    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT") or os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    client = neatlogs.wrap(
        AzureOpenAI(
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),
        )
    )

    temp = os.getenv("AZURE_TEMPERATURE")
    extra = {"temperature": float(temp)} if temp else {}

    # This run is ONE logical turn made of several calls (LLM -> tool -> LLM ->
    # stream), so a WORKFLOW root groups them into a single trace; the LLM/TOOL
    # spans nest under it. (A lone wrapped call auto-roots on its own — this
    # explicit root is purely for grouping multiple calls. The wrapper detects
    # the active WORKFLOW and does NOT add a second root.)
    with neatlogs.trace("azure-weather-chat", kind="WORKFLOW"):
        _run(client, deployment, extra)

    neatlogs.flush()
    neatlogs.shutdown()


def _run(client, deployment, extra):
    print("--- non-streaming chat.completions.create (system + user + tool) ---")
    first = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": "You are a concise weather assistant. Use the get_weather tool when asked about weather."},
            {"role": "user", "content": "What is the weather in San Francisco?"},
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                        "required": ["location"],
                    },
                },
            }
        ],
        **extra,
    )

    tool_call = (first.choices[0].message.tool_calls or [None])[0]
    if tool_call:
        weather = {"location": "San Francisco", "temperature": 72, "conditions": "sunny"}
        print("tool result:", weather)
        second = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": "You are a concise weather assistant. Use the get_weather tool when asked about weather."},
                {"role": "user", "content": "What is the weather in San Francisco?"},
                first.choices[0].message.model_dump(),
                {"role": "tool", "tool_call_id": tool_call.id, "content": json.dumps(weather)},
            ],
            **extra,
        )
        print("final:", second.choices[0].message.content)
    else:
        print("final:", first.choices[0].message.content)

    print("\n--- streaming chat.completions.create ---")
    stream = client.chat.completions.create(
        model=deployment,
        messages=[{"role": "user", "content": "In one sentence, what is OpenTelemetry?"}],
        stream=True,
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print()


if __name__ == "__main__":
    main()
