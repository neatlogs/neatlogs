"""
In-memory knowledge base with real Azure OpenAI embeddings.

Articles are embedded once on first search, then cosine similarity is computed
in Python. Embedding API calls are auto-instrumented by the `openai`
instrumentation and appear as EMBEDDING spans.
"""

import math

from openai import AzureOpenAI

from config import (
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_KEY,
    AZURE_EMBEDDING_API_VERSION,
    AZURE_EMBEDDING_DEPLOYMENT,
)

_client = AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_EMBEDDING_API_VERSION,
)


ARTICLES = [
    {
        "id": "billing-001",
        "title": "How to upgrade or downgrade your subscription plan",
        "content": (
            "You can change your plan at any time from Settings > Billing > Change Plan. "
            "Upgrades take effect immediately and you are charged a prorated amount."
        ),
    },
    {
        "id": "billing-002",
        "title": "Refund policy and how to request a refund",
        "content": (
            "We offer a 14-day money-back guarantee for all new subscriptions. "
            "Email billing@acme.io within 14 days of your initial purchase."
        ),
    },
    {
        "id": "billing-003",
        "title": "Failed payment and account suspension",
        "content": (
            "If a payment fails we retry automatically on days 3, 7, and 14. "
            "After 14 days of failed payments your account is suspended but data is retained for 30 days."
        ),
    },
    {
        "id": "account-001",
        "title": "How to reset your password",
        "content": (
            "Click 'Forgot password' on the login page and enter your account email. "
            "You will receive a password reset link within 2 minutes."
        ),
    },
    {
        "id": "technical-001",
        "title": "API rate limits and handling 429 errors",
        "content": (
            "The API allows 1,000 requests per minute on Standard and 10,000 on Enterprise. "
            "When exceeded the API returns HTTP 429 with a Retry-After header."
        ),
    },
    {
        "id": "technical-002",
        "title": "Webhook setup and troubleshooting",
        "content": (
            "Webhooks are configured in Settings > Developer > Webhooks. "
            "All payloads are signed with HMAC-SHA256 using your webhook secret."
        ),
    },
    {
        "id": "feature-002",
        "title": "Single Sign-On (SSO) configuration",
        "content": (
            "SSO is available on Enterprise plans. Supported providers: Okta, Azure AD, "
            "Google Workspace, and any SAML 2.0-compatible provider."
        ),
    },
]


PAST_TICKETS = [
    {
        "id": "past-001",
        "title": "Resolved: Account locked after too many failed login attempts",
        "content": (
            "Customer was locked out after 10 failed login attempts. Resolution: used admin "
            "unlock endpoint. Also advised 2FA. Resolved in 1 reply."
        ),
    },
    {
        "id": "past-002",
        "title": "Resolved: API key not working after plan downgrade",
        "content": (
            "Customer downgraded from Enterprise to Standard. Resolution: explained new limits, "
            "suggested caching. Resolved in 2 replies."
        ),
    },
    {
        "id": "past-003",
        "title": "Resolved: Webhook signature verification failing",
        "content": (
            "Customer was hashing parsed JSON instead of raw body. Resolution: provided correct "
            "HMAC verification code sample. Resolved in 1 reply."
        ),
    },
    {
        "id": "past-005",
        "title": "Resolved: SSO login loop after Okta configuration change",
        "content": (
            "Customer rotated their Okta signing certificate without updating it in our system. "
            "Resolution: re-uploaded new IdP metadata XML."
        ),
    },
]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


class _KnowledgeBase:
    """Lazy-embedded in-memory knowledge base."""

    def __init__(self, articles: list[dict]):
        self._articles = articles
        self._embeddings: list[list[float]] | None = None

    def _ensure_indexed(self) -> None:
        if self._embeddings is not None:
            return
        texts = [f"{a['title']}\n{a['content']}" for a in self._articles]
        resp = _client.embeddings.create(model=AZURE_EMBEDDING_DEPLOYMENT, input=texts)
        self._embeddings = [d.embedding for d in resp.data]

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        self._ensure_indexed()
        q_resp = _client.embeddings.create(model=AZURE_EMBEDDING_DEPLOYMENT, input=[query])
        q_emb = q_resp.data[0].embedding
        scores = [_cosine(q_emb, emb) for emb in self._embeddings]
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        return [
            {**self._articles[idx], "score": round(score, 4)}
            for idx, score in ranked
        ]

    def format_results(self, results: list[dict]) -> str:
        return "\n\n".join(
            f"[{r['id']}] {r['title']} (score={r['score']})\n{r['content']}"
            for r in results
        )


KB = _KnowledgeBase(ARTICLES)
PAST_KB = _KnowledgeBase(PAST_TICKETS)
