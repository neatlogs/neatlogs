"""
Error Injection Utilities
=========================
Shared error classes and retry utilities for demonstrating Neatlogs SDK
error tracking, retry patterns, and guardrail capabilities.

The @neatlogs.span decorator automatically calls span.record_exception(e)
and span.set_status(ERROR) when a decorated function raises — so simply
raising from within a @neatlogs.span-decorated function produces the
correct error trace with no manual instrumentation.
"""

import time
import json
import re

import neatlogs


# ---------------------------------------------------------------------------
# Custom Error Classes
# ---------------------------------------------------------------------------

class DatabaseTimeoutError(Exception):
    """Simulated ClickHouse connection timeout."""
    pass


class ExternalAPIError(Exception):
    """Simulated external HTTP API error."""

    def __init__(self, status_code: int, service: str, message: str):
        self.status_code = status_code
        self.service = service
        super().__init__(f"HTTP {status_code} from {service}: {message}")


class TokenLimitError(Exception):
    """Simulated token limit / context length exceeded."""
    pass


# ---------------------------------------------------------------------------
# Retry Utility
# ---------------------------------------------------------------------------

def with_retry(func, max_retries=3, backoff_base=1.0,
               transient_errors=(ConnectionError, TimeoutError, ExternalAPIError)):
    """
    Returns a wrapper that retries `func` on transient errors.
    Each attempt is a visible child span so retries show up in traces.
    """

    @neatlogs.span(kind="CHAIN", name=f"{func.__name__}_with_retry",
                   metadata={"max_retries": max_retries, "backoff_base": backoff_base})
    def wrapper(*args, **kwargs):
        last_error = None
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except transient_errors as e:
                last_error = e
                wait = backoff_base * (2 ** attempt)
                print(f"   ⚠️  Attempt {attempt + 1}/{max_retries} failed: {e}. "
                      f"Retrying in {wait:.1f}s…")
                time.sleep(wait)
        raise last_error

    wrapper.__name__ = f"{func.__name__}_with_retry"
    return wrapper


# ---------------------------------------------------------------------------
# Content Guardrail Logic
# ---------------------------------------------------------------------------

INJECTION_PATTERNS = [
    "ignore your instructions",
    "admin password",
    "system prompt",
    "reveal your",
    "disregard previous",
    "override safety",
]

PROFANITY_PATTERNS = [
    r"\bf+[\*\#]+ *i?n?g?\b",
    r"\bsh[\*\#]+t\b",
    r"\bdamn\b",
    r"\bass\b",
    r"\bhell\b",
    r"\bcrap\b",
    r"\bbullsh",
    r"\bf+u+c+k",
    r"\bsh+i+t",
]

HOSTILE_PATTERNS = [
    r"fire[d]?\b",
    r"useless",
    r"incompetent",
    r"threaten",
    r"sue you",
    r"shut.?up",
    r"stupid",
    r"idiot",
]


def detect_content_issues(query: str) -> dict:
    """
    Analyse a query for content issues.
    Returns: {action, reason, flagged_terms, sanitized_query}
    """
    lower = query.lower()

    # 1. Prompt injection (highest severity → BLOCK)
    for pattern in INJECTION_PATTERNS:
        if pattern in lower:
            return {
                "action": "BLOCK",
                "reason": "prompt_injection_detected",
                "flagged_terms": [pattern],
                "sanitized_query": None,
            }

    # 2. Profanity (medium severity → SANITIZE)
    profanity_found = []
    for pattern in PROFANITY_PATTERNS:
        if re.search(pattern, lower):
            profanity_found.append(pattern)
    if profanity_found:
        # Sanitize: strip obvious profanity, keep the business question
        sanitized = query
        for p in PROFANITY_PATTERNS:
            sanitized = re.sub(p, "", sanitized, flags=re.IGNORECASE)
        # Clean up extra whitespace
        sanitized = re.sub(r"\s+", " ", sanitized).strip()
        return {
            "action": "SANITIZE",
            "reason": "profanity_detected",
            "flagged_terms": profanity_found,
            "sanitized_query": sanitized,
        }

    # 3. Hostile/threatening (low severity → FLAG_AND_PROCEED)
    hostile_found = []
    for pattern in HOSTILE_PATTERNS:
        if re.search(pattern, lower):
            hostile_found.append(pattern)
    if hostile_found:
        return {
            "action": "FLAG_AND_PROCEED",
            "reason": "hostile_language_detected",
            "flagged_terms": hostile_found,
            "sanitized_query": query,  # serve as-is but flag
        }

    # 4. Clean
    return {
        "action": "ALLOW",
        "reason": None,
        "flagged_terms": [],
        "sanitized_query": query,
    }


# ---------------------------------------------------------------------------
# SQL Hallucination Validator
# ---------------------------------------------------------------------------

VALID_TABLES = {"orders", "products", "search_rankings", "inventory",
                "campaigns", "pricing_history"}

VALID_COLUMNS = {
    "orders": {"order_id", "brand", "platform", "city", "dark_store_id",
               "sku_id", "quantity", "revenue", "order_date", "status"},
    "products": {"sku_id", "brand", "category", "subcategory", "mrp",
                 "selling_price"},
    "search_rankings": {"sku_id", "platform", "keyword", "rank_position",
                        "search_date", "city"},
    "inventory": {"sku_id", "dark_store_id", "city", "stock_level",
                  "last_updated", "reorder_point"},
    "campaigns": {"campaign_id", "brand", "platform", "keyword", "bid_amount",
                  "impressions", "clicks", "spend", "roas", "campaign_date"},
    "pricing_history": {"sku_id", "platform", "price", "discount_pct",
                        "recorded_at", "city"},
}


def validate_sql(sql: str) -> dict:
    """
    Check SQL for references to non-existent tables.
    Returns: {valid, invalid_tables, hallucination_detected}
    """
    sql_lower = sql.lower()
    # Extract table names after FROM and JOIN
    table_refs = re.findall(r'(?:from|join)\s+([a-z_]+)', sql_lower)
    invalid = [t for t in table_refs if t not in VALID_TABLES]

    return {
        "valid": len(invalid) == 0,
        "invalid_tables": invalid,
        "hallucination_detected": len(invalid) > 0,
        "hallucination_type": "schema_violation" if invalid else None,
    }


# ---------------------------------------------------------------------------
# Data Consistency Checker (for fabricated metrics)
# ---------------------------------------------------------------------------

def check_data_consistency(llm_output: str, source_data: dict) -> dict:
    """
    Check if LLM output references data not present in the source.
    Specifically checks for Tier-3 references when source only has Tier-1/2.
    """
    issues = []

    # Check for tier-3 references when source has no tier-3 data
    city_tiers = source_data.get("city_tier_insights", {})
    has_tier_3 = "tier_3" in city_tiers
    llm_lower = llm_output.lower()

    if not has_tier_3 and ("tier-3" in llm_lower or "tier 3" in llm_lower
                           or "tier_3" in llm_lower):
        issues.append({
            "type": "fabricated_metrics",
            "detail": "LLM references Tier-3 data but source contains only Tier-1 and Tier-2",
        })

    # Check for fabricated city names not in source
    source_cities = set()
    for tier_data in city_tiers.values():
        if isinstance(tier_data, dict):
            source_cities.update(c.lower() for c in tier_data.get("top_cities", []))

    # Common Tier-3 cities that might be hallucinated
    tier3_cities = ["jaipur", "lucknow", "kanpur", "nagpur", "indore",
                    "patna", "bhopal", "coimbatore", "kochi", "vizag"]
    for city in tier3_cities:
        if city in llm_lower and city not in source_cities:
            issues.append({
                "type": "fabricated_location",
                "detail": f"LLM references {city} which is not in the source data",
            })

    return {
        "consistent": len(issues) == 0,
        "hallucination_detected": len(issues) > 0,
        "issues": issues,
    }


# ---------------------------------------------------------------------------
# Routing Validator (for classifier hallucination)
# ---------------------------------------------------------------------------

ROUTING_KEYWORDS = {
    "INVENTORY": ["stockout", "stock", "inventory", "expir", "shelf life",
                  "warehouse", "depot", "supply chain", "out of stock",
                  "purchase order", "reorder"],
    "ANALYTICS": ["revenue", "sales", "growth", "decline", "trend",
                  "performance", "why did", "root cause", "breakdown"],
    "ADS": ["roas", "campaign", "bid", "ad spend", "cpc", "impressions",
            "clicks", "marketing", "advertising", "sov", "share of voice"],
    "MARKET_INTEL": ["competitor", "market share", "opportunity", "trend",
                     "white space", "subcategory", "enter", "launch"],
}


def validate_routing(query: str, classification: str) -> dict:
    """
    Check if the classifier's routing matches keyword signals in the query.
    Returns correction if a strong mismatch is detected.
    """
    query_lower = query.lower()

    # Score each category by keyword match count
    scores = {}
    for category, keywords in ROUTING_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in query_lower)
        scores[category] = score

    best_match = max(scores, key=scores.get) if any(scores.values()) else classification

    if classification != best_match and scores.get(best_match, 0) >= 2:
        return {
            "valid": False,
            "original_classification": classification,
            "corrected_classification": best_match,
            "hallucination_detected": True,
            "hallucination_type": "misclassification",
            "reason": f"Query keywords strongly match {best_match} "
                      f"(score {scores[best_match]}) but was classified as {classification} "
                      f"(score {scores.get(classification, 0)})",
            "scores": scores,
        }

    return {
        "valid": True,
        "hallucination_detected": False,
    }
