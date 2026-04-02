"""
In-memory knowledge base with real OpenAI embeddings.

Articles are embedded once on first search (one batched API call to
text-embedding-3-small), then cosine similarity is computed in Python.

Two corpora are exposed:
  KB        — product KB articles (FAQs, how-tos)
  PAST_KB   — summaries of past resolved tickets

Both use the same _KnowledgeBase class so you get the same
EMBEDDING span structure for both.
"""

import math

from openai import AzureOpenAI

# config must have been imported (neatlogs.init done) before we reach here
from neatlogs.examples.neatlogs_support_bot.config import EMBEDDING_MODEL, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_EMBEDDING_API_VERSION, AZURE_EMBEDDING_DEPLOYMENT

_client = AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_EMBEDDING_API_VERSION,
)


# ---------------------------------------------------------------------------
# KB articles
# ---------------------------------------------------------------------------

ARTICLES = [
    {
        "id": "billing-001",
        "title": "How to upgrade or downgrade your subscription plan",
        "content": (
            "You can change your plan at any time from Settings > Billing > Change Plan. "
            "Upgrades take effect immediately and you are charged a prorated amount for the "
            "remainder of the billing cycle. Downgrades take effect at the start of the next "
            "billing cycle so you keep current features until then. Annual plans can only be "
            "downgraded once per year. If you downgrade from annual, you receive a credit for "
            "unused months applied to your next invoice."
        ),
    },
    {
        "id": "billing-002",
        "title": "Refund policy and how to request a refund",
        "content": (
            "We offer a 14-day money-back guarantee for all new subscriptions. "
            "Email billing@acme.io within 14 days of your initial purchase with your account "
            "email and order ID. Refunds are processed within 5-10 business days to the original "
            "payment method. Renewals are not eligible for refunds. Enterprise plans have separate "
            "refund terms negotiated during contract signing."
        ),
    },
    {
        "id": "billing-003",
        "title": "Failed payment and account suspension",
        "content": (
            "If a payment fails we retry automatically on days 3, 7, and 14. You receive email "
            "notification on each retry. After 14 days of failed payments your account is suspended "
            "but data is retained for 30 days. Update your payment method via Settings > Billing > "
            "Payment Methods. Once a valid payment is processed your account is reactivated instantly."
        ),
    },
    {
        "id": "account-001",
        "title": "How to reset your password",
        "content": (
            "Click 'Forgot password' on the login page and enter your account email. "
            "You will receive a password reset link within 2 minutes. The link expires after "
            "30 minutes. If you do not receive the email check your spam folder or ensure you "
            "are using the correct address. SSO-enabled accounts must reset passwords through "
            "their identity provider (Google Workspace, Okta, etc.)."
        ),
    },
    {
        "id": "account-002",
        "title": "Transferring account ownership to another user",
        "content": (
            "Account owners can transfer ownership from Settings > Team > Members. Select the "
            "member, click 'Transfer Ownership', and confirm via email. The new owner must already "
            "be a team member. After transfer you remain as an admin unless you remove yourself. "
            "Ownership transfer cannot be reversed by support — the new owner must transfer back."
        ),
    },
    {
        "id": "technical-001",
        "title": "API rate limits and handling 429 errors",
        "content": (
            "The API allows 1,000 requests per minute on Standard and 10,000 on Enterprise. "
            "When exceeded the API returns HTTP 429 with a Retry-After header indicating seconds "
            "to wait. Implement exponential backoff starting at 1 second. Rate limits are per API "
            "key, not per account. Create multiple keys from Settings > Developer > API Keys."
        ),
    },
    {
        "id": "technical-002",
        "title": "Webhook setup and troubleshooting",
        "content": (
            "Webhooks are configured in Settings > Developer > Webhooks. Each webhook requires a "
            "URL and event types to subscribe to. All payloads are signed with HMAC-SHA256 using "
            "your webhook secret — verify the X-Acme-Signature header. Failed deliveries are "
            "retried up to 5 times over 24 hours. Inspect delivery history and replay failed "
            "events from the webhook dashboard."
        ),
    },
    {
        "id": "technical-003",
        "title": "Data export and backup options",
        "content": (
            "Export all data from Settings > Data > Export in JSON or CSV format. Large exports "
            "are delivered via email as a download link within 30 minutes. Automated daily backups "
            "are enabled by default on Pro and Enterprise plans. Data is retained for 90 days after "
            "account cancellation. For GDPR data subject requests use Settings > Privacy > Data Requests."
        ),
    },
    {
        "id": "feature-001",
        "title": "Enabling two-factor authentication (2FA)",
        "content": (
            "Enable 2FA from Settings > Security > 2FA. Supported methods: authenticator apps "
            "(Google Authenticator, Authy) and SMS. Admins can enforce 2FA for all team members "
            "from Settings > Team > Security Policy. If you lose access to your 2FA device use a "
            "backup recovery code saved at setup, or contact support with proof of identity."
        ),
    },
    {
        "id": "feature-002",
        "title": "Single Sign-On (SSO) configuration",
        "content": (
            "SSO is available on Enterprise plans. Supported providers: Okta, Azure AD, "
            "Google Workspace, and any SAML 2.0-compatible provider. Configure from Settings > "
            "Security > Single Sign-On using your IdP's metadata URL or XML file. Users can be "
            "provisioned automatically via SCIM once SSO is enabled. Mixed authentication is "
            "supported but can be restricted to SSO-only from security policy settings."
        ),
    },
    {
        "id": "feature-003",
        "title": "Team member roles and permissions",
        "content": (
            "Four roles: Owner (full access including billing), Admin (all except billing and "
            "ownership transfer), Member (create/edit resources, no team settings), Viewer "
            "(read-only). Roles are per workspace so a user can have different roles across "
            "workspaces. Custom roles are available on Enterprise plans."
        ),
    },
    {
        "id": "technical-004",
        "title": "SDK integration and installation",
        "content": (
            "Install our JavaScript SDK with: npm install @acme/sdk. Initialize with your API key: "
            "import { AcmeClient } from '@acme/sdk'; const client = new AcmeClient({ apiKey: YOUR_KEY }). "
            "Python SDK: pip install acme-sdk. Initialize with: from acme import Client; client = Client(api_key=KEY). "
            "SDK documentation and changelogs are at docs.acme.io/sdk. "
            "For server-side usage ensure ACME_API_KEY is set as an environment variable."
        ),
    },
]

# ---------------------------------------------------------------------------
# Past ticket summaries (used by past_tickets_rag_expert)
# ---------------------------------------------------------------------------

PAST_TICKETS = [
    {
        "id": "past-001",
        "title": "Resolved: Account locked after too many failed login attempts",
        "content": (
            "Customer was locked out after 10 failed login attempts. Resolution: used admin "
            "unlock endpoint POST /admin/accounts/{id}/unlock. Also advised customer to use "
            "password manager and enable 2FA to prevent future lockouts. Ticket resolved in 1 reply."
        ),
    },
    {
        "id": "past-002",
        "title": "Resolved: API key not working after plan downgrade",
        "content": (
            "Customer downgraded from Enterprise to Standard. Their API key hit the new 1,000 "
            "req/min rate limit. Resolution: explained new limits, suggested caching responses "
            "and implementing request queuing on their side. Offered to whitelist their key "
            "temporarily for migration. Resolved in 2 replies."
        ),
    },
    {
        "id": "past-003",
        "title": "Resolved: Webhook signature verification failing",
        "content": (
            "Customer was verifying HMAC signature incorrectly — they were hashing the parsed "
            "JSON rather than the raw request body. Resolution: provided code sample showing "
            "correct verification using the raw bytes from the request. Resolved in 1 reply."
        ),
    },
    {
        "id": "past-004",
        "title": "Resolved: Unable to export data larger than 100MB",
        "content": (
            "Customer's export kept timing out for large datasets. Resolution: advised splitting "
            "the export by date range using the ?from= and ?to= query parameters. Also escalated "
            "to engineering who increased the async export timeout. Resolved in 3 replies."
        ),
    },
    {
        "id": "past-005",
        "title": "Resolved: SSO login loop after Okta configuration change",
        "content": (
            "Customer's SSO integration broke after they rotated their Okta signing certificate "
            "without updating it in our system. Resolution: customer re-uploaded the new IdP "
            "metadata XML from Settings > Security > SSO. Resolved immediately after update."
        ),
    },
    {
        "id": "past-006",
        "title": "Resolved: Team member not receiving invite email",
        "content": (
            "Customer sent team invite but recipient never received it. Root cause: recipient's "
            "email provider was blocking emails from our domain. Resolution: customer resent "
            "invite, recipient whitelisted noreply@acme.io, and added to safe senders list."
        ),
    },
]


# ---------------------------------------------------------------------------
# Core KB class
# ---------------------------------------------------------------------------

def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


class _KnowledgeBase:
    """
    Lazy-initialized in-memory knowledge base backed by OpenAI embeddings.
    First call to search() triggers a batched embedding of all articles.
    That batch call is auto-instrumented by OpenInference → EMBEDDING span.
    Each query also creates an EMBEDDING span for the query vector.
    """

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


# Singletons
KB = _KnowledgeBase(ARTICLES)
PAST_KB = _KnowledgeBase(PAST_TICKETS)
