"""Send an email via SendGrid. Used by the send_email_sendgrid TOOL span.

In Trace A, SENDGRID_API_KEY is set to the broken key, so SendGrid returns 401
and the response body lands in the trace via neatlogs.log() inside the span.

In Trace B / B_FIXED, the valid key is used and the call succeeds (status 202).
"""
from __future__ import annotations

import os

import requests

import neatlogs


SENDGRID_URL = "https://api.sendgrid.com/v3/mail/send"


class EmailDeliveryError(RuntimeError):
    """Raised when SendGrid returns a non-2xx response."""


def send_email(*, to_addr: str, from_addr: str, subject: str, body: str) -> dict:
    api_key = os.environ["SENDGRID_API_KEY"]
    payload = {
        "personalizations": [{"to": [{"email": to_addr}]}],
        "from": {"email": from_addr},
        "subject": subject,
        "content": [{"type": "text/plain", "value": body}],
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    # Stub path for happy-path runs when there is no real SendGrid key. The HTTP
    # auto-instrumentation child span still fires (real httpbin call). RUN=A
    # always uses the SendGrid URL so the 401 reveal is genuine.
    use_stub = os.environ.get("SENDGRID_FAKE_SUCCESS") == "1" and api_key != os.environ.get("SENDGRID_API_KEY_BROKEN", "")
    target_url = "https://httpbin.org/status/202" if use_stub else SENDGRID_URL

    neatlogs.log("POST {url} to={to} subject={subject}", url=target_url, to=to_addr, subject=subject)

    resp = requests.post(target_url, json=payload, headers=headers, timeout=10)

    if resp.status_code >= 400:
        # Capture the full SendGrid error body inside the span via neatlogs.log so
        # the failure root cause is visible in the trace itself, no CloudWatch needed.
        neatlogs.log(
            "ERROR {status}: {body}",
            status=resp.status_code,
            body=resp.text[:2000],
        )
        raise EmailDeliveryError(
            f"SendGrid {resp.status_code}: {resp.text[:500]}"
        )

    msg_id = resp.headers.get("X-Message-Id") or "stub-202" if use_stub else resp.headers.get("X-Message-Id", "")
    neatlogs.log("delivered status={status} message_id={mid}", status=resp.status_code, mid=msg_id)
    return {"status": resp.status_code, "message_id": msg_id}
