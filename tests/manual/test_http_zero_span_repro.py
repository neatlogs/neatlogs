"""
Manual reproduction for investigating 0-span dashboard rows caused by non-AI HTTP traffic.

This script creates two traces in one process:

1. non_ai_outgoing_http_only
   - neatlogs.init() always instruments outgoing HTTP client libraries (requests/httpx/etc.).
   - This performs only an outgoing requests.get() without any NeatLogs @span or trace()
     wrapper. It validates whether NeatLogs' always-on HTTP client instrumentation alone
     can create a non-AI trace row.

2. ai_workflow_with_http_child
   - A real NeatLogs WORKFLOW span wraps the operation.
   - The outgoing HTTP call is a child of that workflow span.
   - This should appear as an AI/application trace, not as a confusing standalone 0-span row.

Important: the current SDK does NOT auto-instrument inbound FastAPI/ASGI server spans.
If a customer sees root FastAPI request spans, they likely enabled FastAPI/ASGI
OpenTelemetry instrumentation separately. This script focuses on the behavior NeatLogs
itself enables by default: outgoing HTTP client instrumentation.

Run:
    NEATLOGS_API_KEY=<your-key> python tests/manual/test_http_zero_span_repro.py

Optional:
    NEATLOGS_ENDPOINT=https://staging-cloud.neatlogs.com python tests/manual/test_http_zero_span_repro.py

Dashboard checks:
    - If a row appears for the standalone outgoing HTTP request, the backend/UI is
      creating a trace row for HTTP-only traffic with no NeatLogs semantic application span.
    - The workflow span "ai_workflow_with_http_child" should appear as a normal
      application trace.
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
        endpoint=os.environ.get("NEATLOGS_ENDPOINT", "https://staging-cloud.neatlogs.com"),
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
