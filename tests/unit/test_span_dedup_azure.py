"""
Unit tests for the Azure LangChain + OpenAI duplicate LLM span deduplication fixes.

Tests exercise _dedupe_trace() directly using span dicts (no OTel SDK needed).
"""

from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest


def _make_processor():
    """Create a NeatlogsSpanProcessor with a mock exporter (no I/O)."""
    from neatlogs.core.span_processor import NeatlogsSpanProcessor

    exporter = MagicMock()
    proc = NeatlogsSpanProcessor(exporter=exporter, debug=False)
    # Stop background thread immediately to avoid side effects
    proc._stop_background.set()
    return proc


def _llm(
    span_id: str,
    name: str,
    parent: str | None = None,
    prompt: int = 0,
    completion: int = 0,
    cost: float = 0.0,
    provider: str = "",
    system: str = "",
    internal: bool = False,
    start_ns: int = 1_000_000_000,
    end_ns: int = 2_000_000_000,
) -> Dict[str, Any]:
    attrs: Dict[str, Any] = {"neatlogs.span.kind": "llm"}
    if provider:
        attrs["neatlogs.llm.provider"] = provider
    if system:
        attrs["neatlogs.llm.system"] = system
    if prompt:
        attrs["neatlogs.llm.token_count.prompt"] = prompt
    if completion:
        attrs["neatlogs.llm.token_count.completion"] = completion
    if cost:
        attrs["neatlogs.llm.cost.total"] = cost
        attrs["neatlogs.llm.cost.prompt"] = cost * 0.3
        attrs["neatlogs.llm.cost.completion"] = cost * 0.7
    if internal:
        attrs["neatlogs.internal"] = True
    return {
        "span_id": span_id,
        "name": name,
        "parent_span_id": parent,
        "kind": "llm",
        "start_time": start_ns,
        "end_time": end_ns,
        "attributes": attrs,
        "status": {"code": "OK"},
        "events": [],
    }


def _http(
    span_id: str,
    parent: str,
    start_ns: int = 1_100_000_000,
    end_ns: int = 1_900_000_000,
) -> Dict[str, Any]:
    return {
        "span_id": span_id,
        "name": "POST",
        "parent_span_id": parent,
        "kind": "http",
        "start_time": start_ns,
        "end_time": end_ns,
        "attributes": {"neatlogs.span.kind": "http"},
        "status": {"code": "UNSET"},
        "events": [],
    }


def _chain(
    span_id: str,
    name: str,
    parent: str | None = None,
    start_ns: int = 1_000_000_000,
    end_ns: int = 2_000_000_000,
) -> Dict[str, Any]:
    return {
        "span_id": span_id,
        "name": name,
        "parent_span_id": parent,
        "kind": "chain",
        "start_time": start_ns,
        "end_time": end_ns,
        "attributes": {"neatlogs.span.kind": "chain"},
        "status": {"code": "UNSET"},
        "events": [],
    }


# ---------------------------------------------------------------------------
# Fix 1: has_http_child uses lowercase "http"
# ---------------------------------------------------------------------------


def test_fix1_has_http_child_lowercase():
    """Provider-like spans are correctly identified when neatlogs.span.kind='http'."""
    proc = _make_processor()

    # Without fix (old "HTTP" check), ChatCompletion wouldn't be in provider_like
    # → dedup would early-return.  With fix it should suppress AzureChatOpenAI.
    spans = [
        _llm("user_trace", "UserTrace", internal=True),
        _llm("azure_llm", "AzureChatOpenAI", parent="user_trace",
             provider="azure", prompt=28, completion=139, cost=0.001),
        _llm("chat_compl", "ChatCompletion", parent="user_trace",
             provider="azure", prompt=28, completion=139, cost=0.001),
        _http("post_span", parent="chat_compl"),
    ]

    result = proc._suppress_overlapping_llm_spans(spans)
    names = [s["name"] for s in result]

    assert "AzureChatOpenAI" not in names, "AzureChatOpenAI should be suppressed"
    assert "ChatCompletion" in names, "ChatCompletion (provider-like) should survive"
    assert "UserTrace" in names, "User trace should survive"
    assert "POST" in names, "POST should survive"


# ---------------------------------------------------------------------------
# Fix 2: no self-parenting when provider is a child of framework
# ---------------------------------------------------------------------------


def test_fix2_no_self_parenting_provider_child_of_framework():
    """When provider span is a direct child of framework span, re-parenting
    must promote provider to framework's parent instead of making it self-parent."""
    proc = _make_processor()

    # Hierarchy: root → fw → pv → POST
    spans = [
        _llm("root", "RootSpan"),
        _llm("fw", "AzureChatOpenAI.chat", parent="root",
             provider="azure", prompt=100, completion=200),
        _llm("pv", "chat gpt-4", parent="fw",
             provider="openai", prompt=100, completion=200),
        _http("post", parent="pv"),
    ]

    result = proc._suppress_overlapping_llm_spans(spans)

    pv_span = next(s for s in result if s["name"] == "chat gpt-4")
    assert pv_span["parent_span_id"] == "root", (
        "Provider promoted to framework's parent, not itself"
    )
    assert pv_span["span_id"] != pv_span["parent_span_id"], "No self-parenting"


# ---------------------------------------------------------------------------
# Fix 3: fuzzy provider matching — azure ↔ openai family
# ---------------------------------------------------------------------------


def test_fix3_fuzzy_provider_azure_openai():
    """'azure' and 'openai' belong to the same family and match fuzzily."""
    proc = _make_processor()
    assert proc._providers_match_fuzzy("azure", "openai") is True
    assert proc._providers_match_fuzzy("openai", "azure") is True
    assert proc._providers_match_fuzzy("azure_openai", "openai") is True
    assert proc._providers_match_fuzzy("azure", "anthropic") is False
    assert proc._providers_match_fuzzy("openai", "anthropic") is False


def test_fix3_fuzzy_provider_suppresses_framework_with_azure_provider():
    """AzureChatOpenAI.chat (provider=azure) suppressed against chat gpt-4 (provider=openai)."""
    proc = _make_processor()

    spans = [
        _llm("root", "root_span"),
        _llm("fw", "AzureChatOpenAI.chat", parent="root",
             provider="azure", prompt=100, completion=200),
        _llm("pv", "chat gpt-4", parent="root",
             provider="openai", prompt=100, completion=200),
        _http("post", parent="pv"),
    ]

    result = proc._suppress_overlapping_llm_spans(spans)
    names = [s["name"] for s in result]

    assert "AzureChatOpenAI.chat" not in names, "Framework span should be suppressed"
    assert "chat gpt-4" in names, "Provider span should survive"


# ---------------------------------------------------------------------------
# Fix 4: _suppress_identical_llm_siblings
# ---------------------------------------------------------------------------


def test_fix4_identical_siblings_suppressed():
    """Two LLM siblings with same (parent, name, prompt, completion) and time overlap
    → one suppressed, one kept."""
    proc = _make_processor()

    spans = [
        _llm("parent", "ParentChain"),
        _llm("sibling1", "chat gpt-4", parent="parent",
             provider="openai", prompt=50, completion=100),
        _llm("sibling2", "chat gpt-4", parent="parent",
             provider="openai", prompt=50, completion=100),
    ]

    result = proc._suppress_identical_llm_siblings(spans)
    survivors = [s for s in result if s["name"] == "chat gpt-4"]
    assert len(survivors) == 1, "Exactly one sibling should survive"


def test_fix4_zero_token_siblings_not_grouped():
    """Sibling LLM spans with zero tokens (user decorator spans) are NOT grouped."""
    proc = _make_processor()

    spans = [
        _llm("s1", "UserTrace", parent=None, prompt=0, completion=0),
        _llm("s2", "UserTrace", parent=None, prompt=0, completion=0),
    ]

    result = proc._suppress_identical_llm_siblings(spans)
    assert len([s for s in result if s["name"] == "UserTrace"]) == 2, (
        "Zero-token siblings must not be deduplicated"
    )


def test_fix4_non_overlapping_siblings_not_suppressed():
    """Siblings that do NOT overlap in time are independent calls — keep both."""
    proc = _make_processor()

    spans = [
        _llm("parent", "Parent"),
        _llm("s1", "chat gpt-4", parent="parent", prompt=10, completion=20,
             start_ns=1_000_000_000, end_ns=2_000_000_000),
        _llm("s2", "chat gpt-4", parent="parent", prompt=10, completion=20,
             start_ns=3_000_000_000, end_ns=4_000_000_000),
    ]

    result = proc._suppress_identical_llm_siblings(spans)
    survivors = [s for s in result if s["name"] == "chat gpt-4"]
    assert len(survivors) == 2, "Non-overlapping siblings are different calls"


# ---------------------------------------------------------------------------
# Fix 5: _zero_duplicate_parent_tokens
# ---------------------------------------------------------------------------


def test_fix5_parent_tokens_zeroed():
    """Parent LLM span with identical non-zero tokens as child → parent tokens zeroed."""
    proc = _make_processor()

    spans = [
        _llm("parent", "AzureChatOpenAI.chat", provider="azure",
             prompt=100, completion=200, cost=0.005),
        _llm("child", "chat gpt-4", parent="parent", provider="openai",
             prompt=100, completion=200, cost=0.005),
    ]

    result = proc._zero_duplicate_parent_tokens(spans)
    parent = next(s for s in result if s["name"] == "AzureChatOpenAI.chat")
    pa = parent["attributes"]

    assert pa.get("neatlogs.llm.token_count.prompt") == 0
    assert pa.get("neatlogs.llm.token_count.completion") == 0
    assert pa.get("neatlogs.llm.cost.total") == 0


def test_fix5_different_tokens_not_zeroed():
    """Parent and child with different token counts → parent unchanged."""
    proc = _make_processor()

    spans = [
        _llm("parent", "ParentLLM", provider="azure", prompt=100, completion=200),
        _llm("child", "ChildLLM", parent="parent", provider="openai",
             prompt=50, completion=100),
    ]

    result = proc._zero_duplicate_parent_tokens(spans)
    parent = next(s for s in result if s["name"] == "ParentLLM")
    pa = parent["attributes"]

    assert pa.get("neatlogs.llm.token_count.prompt") == 100, "Different tokens — no zeroing"
    assert pa.get("neatlogs.llm.token_count.completion") == 200


# ---------------------------------------------------------------------------
# Integration: full Azure hierarchy same-batch
# ---------------------------------------------------------------------------


def test_integration_azure_langchain_same_batch():
    """Full hierarchy after Pass 1 re-parenting:
    UserTrace → RunnableSequence → AzureChatOpenAI (fw, no http child)
    UserTrace → ChatCompletion (pv, has http child POST)  [re-parented from openai.chat]
    → dedup should suppress AzureChatOpenAI, keep ChatCompletion with tokens.
    """
    proc = _make_processor()

    # After Pass 1: openai.chat suppressed, ChatCompletion re-parented to user_trace
    user_trace = _llm("user_trace", "langchainAzureChatOpenAI", internal=True)
    runnable = _chain("runnable", "RunnableSequence", parent="user_trace")
    azure_llm = _llm("azure_llm", "AzureChatOpenAI", parent="runnable",
                     provider="azure", prompt=28, completion=139, cost=0.0014)
    chat_compl = _llm("chat_compl", "ChatCompletion", parent="user_trace",
                      provider="azure", prompt=28, completion=139, cost=0.0014)
    post = _http("post", parent="chat_compl")

    spans = [user_trace, runnable, azure_llm, chat_compl, post]
    result = proc._suppress_overlapping_llm_spans(spans)
    names = [s["name"] for s in result]

    assert "AzureChatOpenAI" not in names, "AzureChatOpenAI should be suppressed"
    assert "ChatCompletion" in names, "ChatCompletion should survive"
    assert "langchainAzureChatOpenAI" in names, "User trace must survive"
    assert "RunnableSequence" in names, "RunnableSequence must survive"

    # Verify no self-parenting
    for s in result:
        assert s.get("parent_span_id") != s["span_id"], (
            f"Span {s['name']} self-parents!"
        )

    # Exactly 1 LLM span with tokens
    llm_with_tokens = [s for s in result if s.get("kind") == "llm"
                       and (s.get("attributes", {}).get("neatlogs.llm.token_count.prompt") or 0) > 0]
    assert len(llm_with_tokens) == 1, f"Expected 1 LLM span with tokens, got {len(llm_with_tokens)}"
    assert llm_with_tokens[0]["name"] == "ChatCompletion"


def test_integration_direct_invoke_same_batch():
    """Direct invoke pattern:
    UserTrace → AzureChatOpenAI (fw, same parent as ChatCompletion)
    UserTrace → ChatCompletion (pv, has http child POST)
    → suppress AzureChatOpenAI (same-parent +2 + provider +1 + time +3 = 6).
    """
    proc = _make_processor()

    user_trace = _llm("user_trace", "directAzureInvoke", internal=True)
    azure_llm = _llm("azure_llm", "AzureChatOpenAI", parent="user_trace",
                     provider="azure", prompt=12, completion=203, cost=0.002)
    chat_compl = _llm("chat_compl", "ChatCompletion", parent="user_trace",
                      provider="azure", prompt=12, completion=203, cost=0.002)
    post = _http("post", parent="chat_compl")

    spans = [user_trace, azure_llm, chat_compl, post]
    result = proc._suppress_overlapping_llm_spans(spans)
    names = [s["name"] for s in result]

    assert "AzureChatOpenAI" not in names
    assert "ChatCompletion" in names
    assert "directAzureInvoke" in names

    llm_with_tokens = [s for s in result if s.get("kind") == "llm"
                       and (s.get("attributes", {}).get("neatlogs.llm.token_count.prompt") or 0) > 0]
    assert len(llm_with_tokens) == 1


def test_integration_cross_batch_identical_siblings():
    """Cross-batch case: two AzureChatOpenAI siblings with same tokens (no HTTP child).
    _suppress_identical_llm_siblings should keep one.
    """
    proc = _make_processor()

    parent = _chain("parent", "RunnableSequence")
    sib1 = _llm("sib1", "AzureChatOpenAI", parent="parent",
                provider="azure", prompt=26, completion=75)
    sib2 = _llm("sib2", "AzureChatOpenAI", parent="parent",
                provider="azure", prompt=26, completion=75)

    spans = [parent, sib1, sib2]
    result = proc._suppress_identical_llm_siblings(spans)
    survivors = [s for s in result if s["name"] == "AzureChatOpenAI"]
    assert len(survivors) == 1
