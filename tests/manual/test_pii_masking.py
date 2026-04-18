"""
TEST 4: Client-side PII masking
Covers SKILL.md §Data Masking — client-side `mask=` callback to `init()`.

What to verify:
  1. The mask function is called before export.
  2. Span attributes containing "email" or "password" are replaced with "[REDACTED]"
     in the NeatLogs dashboard (workflow "test-pii-masking").
  3. Other attributes (e.g. "query") are NOT redacted.

Run:
    NEATLOGS_API_KEY=<your-key> python tests/manual/test_pii_masking.py

Expected output (no errors):
    [pii_masking] span created with sensitive attributes
    [pii_masking] flush done
    [pii_masking] shutdown done
    PASS

In the dashboard, the "test-pii" span should show:
  - input.user_email  → [REDACTED]
  - input.password    → [REDACTED]
  - input.query       → "What is the weather?"   (NOT redacted)
"""

import neatlogs


def redact_pii(span):
    """Redact any attribute whose key contains 'email' or 'password'."""
    attrs = span.get("attributes", {})
    for key in list(attrs):
        if "email" in key or "password" in key:
            attrs[key] = "[REDACTED]"
    return span


def main():
    neatlogs.init(
        api_key=None,  # reads NEATLOGS_API_KEY from env
        workflow_name="test-pii-masking",
        mask=redact_pii,
        disable_export=False,
    )

    @neatlogs.span(kind="CHAIN")
    def handle_request(query: str, user_email: str, password: str):
        # Simulate setting span attributes that contain PII
        from opentelemetry import trace as otel_trace
        span = otel_trace.get_current_span()
        span.set_attribute("input.user_email", user_email)
        span.set_attribute("input.password", password)
        span.set_attribute("input.query", query)
        return f"Handled: {query}"

    result = handle_request(
        query="What is the weather?",
        user_email="alice@example.com",
        password="supersecret123",
    )
    print(f"[pii_masking] span created with sensitive attributes, result={result!r}")

    neatlogs.flush()
    print("[pii_masking] flush done")

    neatlogs.shutdown()
    print("[pii_masking] shutdown done")

    print("PASS")


main()
