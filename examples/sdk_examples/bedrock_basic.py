"""
AWS Bedrock Python example with Neatlogs.

Verifies neatlogs.wrap(boto3 bedrock-runtime client) traces the Converse and
ConverseStream APIs with provider="bedrock", system=<model vendor>.

Run:
    python examples/sdk_examples/bedrock_basic.py

Env:
    AWS_REGION, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY (standard AWS chain)
    BEDROCK_MODEL_ID (default: us.anthropic.claude-haiku-4-5-20251001-v1:0)
    NEATLOGS_API_KEY, NEATLOGS_ENDPOINT (default http://localhost:4100)
"""

import os

import boto3
import neatlogs


def main() -> None:
    neatlogs.init(
        api_key=os.getenv("NEATLOGS_API_KEY", ""),
        endpoint=os.getenv("NEATLOGS_ENDPOINT", "http://localhost:4100"),
        workflow_name="bedrock-basic-py",
        tags=["bedrock", "python", "basic"],
    )

    region = os.getenv("AWS_REGION", "us-east-1")
    model_id = os.getenv("BEDROCK_MODEL_ID", "us.anthropic.claude-haiku-4-5-20251001-v1:0")

    client = neatlogs.wrap(boto3.client("bedrock-runtime", region_name=region))

    # WORKFLOW root so the trace has a root span (LLM spans nest under it).
    with neatlogs.trace("bedrock-demo", kind="WORKFLOW"):
        _run(client, model_id)

    neatlogs.flush()
    neatlogs.shutdown()


def _run(client, model_id):
    print("--- Converse ---")
    resp = client.converse(
        modelId=model_id,
        messages=[{"role": "user", "content": [{"text": "In one sentence, what is AWS Bedrock?"}]}],
        inferenceConfig={"temperature": 0.2, "maxTokens": 512},
    )
    print(resp["output"]["message"]["content"][0]["text"])

    print("\n--- ConverseStream ---")
    stream = client.converse_stream(
        modelId=model_id,
        messages=[{"role": "user", "content": [{"text": "List three uses for distributed tracing."}]}],
    )
    for event in stream["stream"]:
        delta = event.get("contentBlockDelta", {}).get("delta", {})
        if delta.get("text"):
            print(delta["text"], end="", flush=True)
    print()


if __name__ == "__main__":
    main()
