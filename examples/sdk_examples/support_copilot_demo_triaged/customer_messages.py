"""Customer ticket bodies used in the two demo traces.

Trace A intentionally contains PII (email + last-4 card digits) so the
PII-redaction story has something to redact.
"""

TRACE_A_BILLING_DISPUTE = (
    "Hi, I see a charge of $89.99 from your company on my card ending 4242 last "
    "Tuesday but I never purchased anything. My account email is "
    "alice.chen@gmail.com. Please refund this immediately."
)

TRACE_B_REFUND_INQUIRY = (
    "I requested a refund 20 days after purchase and you said no. Your website's "
    "FAQ says 30 days, see screenshot. Please process the refund."
)
