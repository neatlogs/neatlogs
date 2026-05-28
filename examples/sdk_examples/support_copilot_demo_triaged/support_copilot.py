"""support-copilot — the demo customer-support agent.

Three run modes via env var RUN={A,B,B_FIXED}:
  A         — broken email API (uses SENDGRID_API_KEY_BROKEN). All other parts succeed.
  B         — silent prompt regression (Chroma missing v3.0, prompt v3 hardcodes 30 days).
  B_FIXED   — fix applied: Chroma has v3.0, prompt v4 cites canonical URL.

Run a single mode:
    RUN=A   ../venv/bin/python support_copilot.py
    RUN=B   ../venv/bin/python support_copilot.py
    RUN=B_FIXED ../venv/bin/python support_copilot.py
"""
from __future__ import annotations

import base64
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- 1. Init MUST happen before any LLM/Chroma/requests imports
import neatlogs

neatlogs.init(
    api_key=os.environ["NEATLOGS_API_KEY"],
    endpoint=os.environ.get("NEATLOGS_ENDPOINT"),
    workflow_name="support-copilot-triaged",
    tags=["sdk-examples", "support-copilot", "triaged", "azure-openai", "chroma"],
    instrumentations=["openai", "chromadb", "requests"],
    capture_logs=True,
    pii_enabled=True,
    pii_span_types=["LLM"],
)

# --- 2. NOW import LLM/Chroma/requests-using modules
from openai import AzureOpenAI

from customer_messages import TRACE_A_BILLING_DISPUTE, TRACE_B_REFUND_INQUIRY
from email_tool import EmailDeliveryError, send_email
from kb_data import SEEDS_BROKEN_RUN_B, SEEDS_NORMAL
from prompts import (
    SUPPORT_DRAFTER_SYSTEM_V3,
    SUPPORT_DRAFTER_SYSTEM_V4,
    SUPPORT_DRAFTER_USER,
)
from seed_chroma import build_collection


def _azure_client() -> AzureOpenAI:
    return AzureOpenAI(
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    )


def _load_image_data_url(asset_filename: str) -> str:
    """Read an asset PNG, return a data: URL so it rides into the WORKFLOW input."""
    here = Path(__file__).parent
    png = (here / "assets" / asset_filename).read_bytes()
    b64 = base64.b64encode(png).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _customer_ticket(text: str, asset_filename: str) -> str:
    """Return a clean markdown string with the screenshot embedded as a data-URL image.

    The simplifier renders Markdown for plain-string inputs, so we get inline image
    rendering AND no `Type: text / Text: ... / Type: image_url` boilerplate.
    """
    return (
        "## 📩 Customer Support Ticket\n\n"
        "### Message\n\n"
        f"{text}\n\n"
        "### Attachment\n\n"
        f"![{asset_filename}]({_load_image_data_url(asset_filename)})"
    )


# ---------- Run-scoped state ----------
#
# The decorated span functions below use module-scoped state for the Chroma
# collection, prompt templates, and the markdown ticket. Zero-arg WORKFLOW keeps
# the trace Input panel as rendered markdown (headings + inline screenshot),
# not a JSON blob like `Ticket Markdown: ## …`.
_ACTIVE: dict = {}


# ---------- Spans ----------

CLASSIFIER_SYSTEM_PROMPT = (
    "You are a customer-support ticket classifier for ACME. "
    "Read the customer message and return strict JSON with two fields: "
    "`category` (one of: billing_dispute, refund_inquiry, account_access, general) "
    "and `priority` (one of: high, medium, low). "
    "Reply with ONLY the JSON object, no prose."
)


@neatlogs.span(kind="AGENT", name="Classify Ticket")
def classify_ticket(message: str) -> dict:
    """Classify the ticket using a small LLM call."""
    client = _azure_client()
    resp = client.chat.completions.create(
        model=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
        messages=[
            {"role": "system", "content": CLASSIFIER_SYSTEM_PROMPT},
            {"role": "user", "content": message},
        ],
        response_format={"type": "json_object"},
    )
    raw = resp.choices[0].message.content or "{}"
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = {"category": "general", "priority": "low"}
    # Defensive defaults in case the model omits a field.
    return {
        "category": parsed.get("category", "general"),
        "priority": parsed.get("priority", "low"),
    }


@neatlogs.span(kind="RETRIEVER", name="Search Knowledge Base")
def kb_search(query: str, n_results: int = 3) -> list[dict]:
    coll = _ACTIVE["coll"]
    # Triage fix: exclude archived KB articles from retrieval so the agent never
    # grounds itself in stale policy. Chroma `where` accepts a metadata filter
    # using the `$ne` operator.
    res = coll.query(
        query_texts=[query],
        n_results=n_results,
        where={"status": {"$ne": "archived"}},
    )
    docs: list[dict] = []
    for i, doc_id in enumerate(res["ids"][0]):
        docs.append(
            {
                "id": doc_id,
                "title": res["metadatas"][0][i].get("title"),
                "url": res["metadatas"][0][i].get("url"),
                "version": res["metadatas"][0][i].get("version"),
                "last_updated": res["metadatas"][0][i].get("last_updated"),
                "snippet": res["documents"][0][i][:280],
                "score": (float(res["distances"][0][i]) if "distances" in res else None),
            }
        )
    # The trace UI surfaces retrieved docs as the RETRIEVER span's Output
    # (markdown numbered list with title + version + URL). This summary log is
    # an additional inline note — useful for quick scanning in the timeline.
    top_title = docs[0].get("title") if docs else "—"
    neatlogs.log(
        "📚 Found **{n}** matching article(s). Top result: _{top}_",
        n=len(docs),
        top=top_title,
    )
    return docs


def _format_kb_context(kb_docs: list[dict]) -> str:
    """Render retrieved KB articles as a markdown numbered list for the LLM."""
    if not kb_docs:
        return "_No matching KB articles._"
    lines: list[str] = []
    for idx, doc in enumerate(kb_docs, start=1):
        title = doc.get("title") or doc["id"]
        version = doc.get("version") or "n/a"
        last_updated = doc.get("last_updated") or "n/a"
        url = doc["url"]
        snippet = doc["snippet"].strip()
        lines.append(
            f"{idx}. **{title}** (version `{version}`, last updated {last_updated})  \n"
            f"   {url}  \n"
            f"   > {snippet}"
        )
    return "\n\n".join(lines)


@neatlogs.span(kind="AGENT", name="Draft Reply", capture_input=False)
def draft_reply() -> str:
    """Draft a reply from templates + run-scoped state.

    No function args — `message` and `kb` are captured as prompt template
    variables on the nested LLM span and surfaced on this agent's Prompt panel
    as `{{message}}` / `{{kb}}` chips (hover to see values).
    """
    customer_message = _ACTIVE["customer_text"]
    kb_context = _ACTIVE["kb_context"]
    system_tpl = _ACTIVE["system_tpl"]
    user_tpl = _ACTIVE["user_tpl"]
    with neatlogs.trace(
        "Generate Reply with LLM",
        kind="LLM",
        system_prompt_template=system_tpl,
        user_prompt_template=user_tpl,
    ):
        compiled_system = system_tpl.compile()
        compiled_user = user_tpl.compile(message=customer_message, kb=kb_context)
        client = _azure_client()
        resp = client.chat.completions.create(
            model=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
            messages=[
                {"role": "system", "content": compiled_system},
                {"role": "user", "content": compiled_user},
            ],
        )
    return resp.choices[0].message.content


@neatlogs.span(kind="TOOL", name="Send Email", tool_name="Send Email")
def send_email_span(reply_text: str) -> dict:
    return send_email(
        to_addr=os.environ["SENDGRID_TO"],
        from_addr=os.environ["SENDGRID_FROM"],
        subject="Re: your support ticket",
        body=reply_text,
    )


@neatlogs.span(kind="WORKFLOW", name="Handle Support Ticket", capture_input=False)
def handle_support_ticket() -> dict:
    """No positional args — otherwise the SDK serializes kwargs as JSON and the
    UI flattens `{"ticket_markdown": "## …"}` to `Ticket Markdown: ## …`, which
    breaks markdown rendering. The ticket lives in `_ACTIVE`; we set `input.value`
    to the bare markdown string so headings and the embedded screenshot render.
    """
    from opentelemetry import trace as _otel_trace

    ticket_markdown = _ACTIVE["ticket_markdown"]
    span = _otel_trace.get_current_span()
    span.set_attribute("input.value", ticket_markdown)
    span.set_attribute("input.mime_type", "text/markdown")
    span.set_attribute("neatlogs.workflow.input", ticket_markdown)

    customer_text = _ACTIVE["customer_text"]
    image_filename = _ACTIVE["image_filename"]
    neatlogs.log(
        "New support ticket received - {chars} characters, attachment {att}",
        chars=len(customer_text),
        att=image_filename,
    )

    classification = classify_ticket(customer_text)
    docs = kb_search(query=customer_text)
    kb_context = _format_kb_context(docs)
    _ACTIVE["kb_context"] = kb_context
    reply = draft_reply()
    delivery = send_email_span(reply)
    return {
        "classification": classification,
        "kb_top": docs[0]["id"] if docs else None,
        "delivery": delivery,
        "reply_preview": reply[:500],
    }


# ---------- Run modes ----------

def _activate_run(*, customer_text: str, image_filename: str, coll, system_tpl, user_tpl) -> None:
    """Stash run-scoped state including the markdown ticket shown in the workflow Input panel."""
    _ACTIVE.clear()
    _ACTIVE.update(
        {
            "customer_text": customer_text,
            "image_filename": image_filename,
            "ticket_markdown": _customer_ticket(customer_text, image_filename),
            "coll": coll,
            "system_tpl": system_tpl,
            "user_tpl": user_tpl,
        }
    )


def run_a():
    os.environ["SENDGRID_API_KEY"] = os.environ["SENDGRID_API_KEY_BROKEN"]
    _activate_run(
        customer_text=TRACE_A_BILLING_DISPUTE,
        image_filename="bank_statement.png",
        coll=build_collection(SEEDS_NORMAL),
        system_tpl=SUPPORT_DRAFTER_SYSTEM_V3,  # v3 used in A; the v3 issue is irrelevant here
        user_tpl=SUPPORT_DRAFTER_USER,
    )
    return handle_support_ticket()


def run_b():
    # Triage Step 4 simulated: KB has been re-indexed so v3.0 is now present
    # alongside the archived v2.1. The metadata filter (added in kb_search) will
    # exclude v2.1, and the retriever will return v3.0 — the current policy.
    os.environ["SENDGRID_API_KEY"] = os.environ["SENDGRID_API_KEY_VALID"]
    _activate_run(
        customer_text=TRACE_B_REFUND_INQUIRY,
        image_filename="help_center_30day.png",
        coll=build_collection(SEEDS_NORMAL),
        system_tpl=SUPPORT_DRAFTER_SYSTEM_V3,  # post-Triage v3 (no hardcoded window)
        user_tpl=SUPPORT_DRAFTER_USER,
    )
    return handle_support_ticket()


def run_b_fixed():
    os.environ["SENDGRID_API_KEY"] = os.environ["SENDGRID_API_KEY_VALID"]
    _activate_run(
        customer_text=TRACE_B_REFUND_INQUIRY,
        image_filename="help_center_30day.png",
        coll=build_collection(SEEDS_NORMAL),
        system_tpl=SUPPORT_DRAFTER_SYSTEM_V4,  # fixed
        user_tpl=SUPPORT_DRAFTER_USER,
    )
    return handle_support_ticket()


RUNS = {"A": run_a, "B": run_b, "B_FIXED": run_b_fixed}


def main() -> int:
    mode = os.environ.get("RUN", "A").upper()
    if mode not in RUNS:
        print(f"Unknown RUN={mode}. Choose from {sorted(RUNS)}", file=sys.stderr)
        return 2

    result_payload = None
    error_payload = None
    try:
        result_payload = RUNS[mode]()
    except EmailDeliveryError as e:
        # Trace A: expected. The error is captured inside the TOOL span via neatlogs.log;
        # we let the WORKFLOW span see the exception so the trace is marked failed.
        error_payload = {"type": "EmailDeliveryError", "message": str(e)}
    finally:
        neatlogs.flush()
        import time
        time.sleep(3)
        neatlogs.shutdown()

    print(json.dumps({"mode": mode, "result": result_payload, "error": error_payload}, indent=2, default=str))
    return 0 if error_payload is None or mode == "A" else 1


if __name__ == "__main__":
    raise SystemExit(main())
