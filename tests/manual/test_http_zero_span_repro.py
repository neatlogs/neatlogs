"""
Manual regression check that non-AI HTTP traffic produces no Neatlogs spans.

This script creates two traces in one process:

1. non_ai_outgoing_http_only
   - This performs only an outgoing requests.get() without a NeatLogs semantic span.
   - It must not create a trace row.

2. ai_workflow_with_http_child
   - A real NeatLogs WORKFLOW span wraps the operation.
   - The outgoing HTTP call is not captured; only the workflow is exported.

Important: the current SDK does NOT auto-instrument inbound FastAPI/ASGI server spans.
If a customer sees root FastAPI request spans, they likely enabled FastAPI/ASGI
OpenTelemetry instrumentation separately. NeatLogs itself does not instrument
outgoing HTTP transports.

Run:
    NEATLOGS_API_KEY=<your-key> python tests/manual/test_http_zero_span_repro.py

Optional:
    NEATLOGS_ENDPOINT=https://ingest.neatlogs.com python tests/manual/test_http_zero_span_repro.py

Dashboard checks:
    - No row appears for the standalone outgoing HTTP request.
    - The workflow span "ai_workflow_with_http_child" appears without an HTTP child.
"""

import os

import requests

import neatlogs


def non_ai_outgoing_http_only() -> int:
    response = requests.get("https://httpbin.org/status/204", timeout=10)
    return response.status_code


def main() -> None:
    neatlogs.init(
        api_key=None,  # reads NEATLOGS_API_KEY from env
        endpoint=os.environ.get("NEATLOGS_ENDPOINT", "https://ingest.neatlogs.com"),
        workflow_name="zero-span-non-ai-http-repro",
        instrumentations=[],
    )

    @neatlogs.span(kind="WORKFLOW", name="ai_workflow_with_http_child")
    def ai_workflow_with_http_child() -> int:
        response = requests.get("https://httpbin.org/status/204", timeout=10)
        return response.status_code

    print(f"[non_ai] status={non_ai_outgoing_http_only()}")
    print(f"[workflow] status={ai_workflow_with_http_child()}")
    neatlogs.flush()
    neatlogs.shutdown()
    print("PASS")


if __name__ == "__main__":
    main()
