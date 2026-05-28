"""Canonical KB documents that get seeded into Chroma.

Each entry: id (matches HTML filename slug), title, body text, metadata dict
including the public GH Pages URL. This is the demo's ground truth for what
the retriever can return.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class KbDoc:
    id: str
    title: str
    body: str
    last_updated: str
    version: str | None
    url_path: str  # appended to KB_SITE_BASE_URL at seed time
    status: str = "current"  # "current" | "archived" — used as a Chroma metadata filter


REFUND_POLICY_V2_1 = KbDoc(
    id="refund-policy-v2.1",
    title="Refund Policy (archived)",
    body=(
        "If you are not satisfied with your purchase, you may request a refund within "
        "30 days of the original transaction date. Refunds are processed to the original "
        "payment method within 5 business days."
    ),
    last_updated="2026-02-18",
    version="2.1",
    url_path="kb/refund-policy-v2.1.html",
    status="archived",
)

REFUND_POLICY_V3_0 = KbDoc(
    id="refund-policy-v3.0",
    title="Refund Policy",
    body=(
        "If you are not satisfied with your purchase, you may request a refund within "
        "14 days of the original transaction date. Refunds are processed to the original "
        "payment method within 5 business days."
    ),
    last_updated="2026-03-04",
    version="3.0",
    url_path="kb/refund-policy-v3.0.html",
    status="current",
)

BILLING_DISPUTE_SOP = KbDoc(
    id="billing-dispute-sop",
    title="Resolving a Billing Dispute",
    body=(
        "If you see a charge from ACME that you do not recognize, confirm the charge "
        "amount and date, check for shared-account use, then contact support with your "
        "account email and the last four digits of the card."
    ),
    last_updated="2026-04-03",
    version=None,
    url_path="kb/billing-dispute-sop.html",
    status="current",
)

PASSWORD_RESET = KbDoc(
    id="password-reset",
    title="Reset Your Password",
    body=(
        "To reset your password, visit the sign-in page and click 'Forgot password'. "
        "A reset link is emailed to the address on file. The link expires after 60 minutes."
    ),
    last_updated="2026-04-12",
    version=None,
    url_path="kb/password-reset.html",
    status="current",
)


# Seed sets per run mode. Trace B's bug is the absence of v3.0.
SEEDS_NORMAL = [REFUND_POLICY_V2_1, REFUND_POLICY_V3_0, BILLING_DISPUTE_SOP, PASSWORD_RESET]
SEEDS_BROKEN_RUN_B = [REFUND_POLICY_V2_1, BILLING_DISPUTE_SOP, PASSWORD_RESET]


def kb_url(doc: KbDoc) -> str:
    """Resolve a KB doc's full public URL using KB_SITE_BASE_URL env var."""
    base = os.environ.get("KB_SITE_BASE_URL", "https://neatlogs.github.io/support-copilot-demo-kb").rstrip("/")
    return f"{base}/{doc.url_path}"
