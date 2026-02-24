"""
Gobbs Edge — Analytics Agent
==============================
Mirrors GobbleCube's NLQ-to-SQL pipeline + decision-tree root-cause analysis.

Pipeline:
  recognize_intent → generate_sql → execute_query
                                         │
                           (anomaly?) ───┤
                                         ├─ analyze_root_cause → END
                                         └─ END

Error variants:
  db_timeout       — execute_query raises DatabaseTimeoutError
  token_limit      — analyze_root_cause raises TokenLimitError, retries with truncated context
  retry_storm      — execute_query fails twice then succeeds (exponential backoff)
  hallucination_sql — generate_sql references invalid tables, sql_validator catches it
"""

import json
import time
from typing import Optional, Annotated

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

import neatlogs
from config import llm
from error_injection import (
    DatabaseTimeoutError,
    TokenLimitError,
    validate_sql,
    with_retry,
)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class AnalyticsState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    user_query: str
    intent: Optional[str]
    entities: Optional[dict]
    generated_sql: Optional[str]
    query_results: Optional[dict]
    root_cause: Optional[str]
    framework_used: Optional[str]
    confidence_score: Optional[float]
    error_variant: Optional[str]


# ---------------------------------------------------------------------------
# Nodes (Original — happy path)
# ---------------------------------------------------------------------------

@neatlogs.span(kind="AGENT", name="intent_recognition", role="Intent Classifier",
               goal="Classify analytics query intent and extract entities")
def recognize_intent(state: AnalyticsState) -> dict:
    """
    Step 1 of NLQ-to-SQL: classify intent and extract entities.
    Mirrors GobbleCube's first pipeline stage.
    """
    prompt = (
        "You are an e-commerce analytics intent classifier for a quick-commerce platform.\n"
        "Analyze the query and return a JSON object with:\n"
        '  - "intent": one of [revenue_analysis, share_of_search, pricing_analysis, '
        "availability_check, campaign_performance, root_cause_analysis]\n"
        '  - "entities": dict with any of: brand, platform (blinkit/zepto/instamart), '
        "city, category, time_period, metric\n\n"
        f"Query: {state['user_query']}\n\n"
        "Return ONLY valid JSON."
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    try:
        parsed = json.loads(response.content)
    except json.JSONDecodeError:
        parsed = {"intent": "revenue_analysis", "entities": {}}

    return {
        "intent": parsed.get("intent", "revenue_analysis"),
        "entities": parsed.get("entities", {}),
        "messages": [response],
    }


@neatlogs.span(kind="CHAIN", name="nlq_to_sql", goal="Generate ClickHouse SQL from NL query")
def generate_sql(state: AnalyticsState) -> dict:
    """
    Step 2: zero-shot NLQ-to-SQL generation.
    GobbleCube found zero-shot to outperform few-shot on their domain datasets.
    """
    schema = """
Tables (ClickHouse):
  orders        (order_id, brand, platform, city, dark_store_id, sku_id, quantity, revenue, order_date, status)
  products      (sku_id, brand, category, subcategory, mrp, selling_price)
  search_rankings (sku_id, platform, keyword, rank_position, search_date, city)
  inventory     (sku_id, dark_store_id, city, stock_level, last_updated, reorder_point)
  campaigns     (campaign_id, brand, platform, keyword, bid_amount, impressions, clicks, spend, roas, campaign_date)
  pricing_history (sku_id, platform, price, discount_pct, recorded_at, city)
"""
    prompt = (
        "You are a ClickHouse SQL expert generating queries for an e-commerce analytics platform.\n\n"
        f"{schema}\n"
        f"Intent: {state['intent']}\n"
        f"Entities: {json.dumps(state.get('entities', {}))}\n"
        f"Original Query: {state['user_query']}\n\n"
        "Return ONLY the SQL query, no explanation or markdown fences."
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    return {
        "generated_sql": response.content.strip(),
        "messages": [response],
    }


@neatlogs.span(kind="TOOL", name="execute_query", tool_name="antman_analytics_engine")
def execute_query(state: AnalyticsState) -> dict:
    """
    Step 3: Simulated query execution against GobbleCube's Antman engine.
    Returns realistic dummy results keyed by intent.
    """
    simulated: dict[str, dict] = {
        "revenue_analysis": {
            "total_revenue": 2_450_000,
            "revenue_change_pct": -12.3,
            "period": "last_week",
            "top_declining_cities": ["Mumbai", "Delhi", "Bangalore"],
            "top_declining_skus": ["SKU-1234", "SKU-5678"],
        },
        "share_of_search": {
            "brand_sov": 23.5,
            "competitor_sov": 31.2,
            "total_search_volume": 485_000,
            "top_keywords": ["protein bar", "healthy snacks"],
            "rank_changes": [{"keyword": "protein bar", "from": 2, "to": 5}],
        },
        "availability_check": {
            "overall_availability": 78.5,
            "oos_dark_stores": 42,
            "critical_skus_oos": ["SKU-1234", "SKU-9012"],
            "cities_below_threshold": ["Pune", "Hyderabad"],
        },
        "pricing_analysis": {
            "avg_selling_price": 178,
            "avg_mrp": 220,
            "discount_pct": 19.1,
            "competitor_avg_price": 165,
            "price_gap": 13,
        },
        "campaign_performance": {
            "total_campaigns": 14,
            "avg_roas": 3.1,
            "total_spend": 68_000,
            "top_performing_keyword": "protein bar",
            "underperforming_campaigns": 3,
        },
        "root_cause_analysis": {
            "symptoms_detected": ["revenue_drop", "sov_decline"],
            "preliminary_root_cause": "Competitor promotional surge on Blinkit",
            "confidence": 0.72,
        },
    }

    intent = state.get("intent", "")
    result = simulated.get(intent, {"status": "no_data", "message": "No matching data found"})
    return {
        "query_results": result,
        "messages": [AIMessage(content=f"[Antman Engine] Query executed. Results: {json.dumps(result)}")],
    }


@neatlogs.span(kind="AGENT", name="root_cause_analysis", role="Diagnostic Agent",
               goal="Walk decision-tree frameworks to identify root causes")
def analyze_root_cause(state: AnalyticsState) -> dict:
    """
    Step 4 (conditional): Decision-tree root-cause analysis.
    Mirrors GobbleCube's productized 'why' frameworks.
    """
    prompt = (
        "You are a root-cause analysis engine for an e-commerce analytics platform.\n"
        "Use structured decision-tree frameworks to diagnose business problems.\n\n"
        "Framework rules:\n"
        "  Revenue drop → availability → pricing → visibility → competition → seasonality\n"
        "  SOV decline  → keyword ranking → ad spend → new competitors → content quality\n"
        "  Availability → supply chain → demand spike → warehouse allocation\n\n"
        f"Query: {state['user_query']}\n"
        f"Intent: {state.get('intent')}\n"
        f"Data: {json.dumps(state.get('query_results', {}))}\n\n"
        "Return JSON with: framework, root_cause, contributing_factors (list), "
        "confidence (0–1), recommended_actions (list).\n"
        "Return ONLY valid JSON."
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    try:
        parsed = json.loads(response.content)
    except json.JSONDecodeError:
        parsed = {"framework": "revenue_diagnostic", "root_cause": "Analysis complete", "confidence": 0.75}

    return {
        "root_cause": response.content,
        "framework_used": parsed.get("framework", "general_diagnostic"),
        "confidence_score": parsed.get("confidence", 0.7),
        "messages": [response],
    }


# ---------------------------------------------------------------------------
# Error Variant: Database Timeout
# ---------------------------------------------------------------------------

@neatlogs.span(kind="TOOL", name="execute_query", tool_name="antman_analytics_engine")
def execute_query_timeout(state: AnalyticsState) -> dict:
    """Simulates a ClickHouse connection timeout after delay."""
    print("   ⏳ Connecting to Antman analytics engine…")
    time.sleep(2)
    raise DatabaseTimeoutError(
        "Connection timed out after 30000ms: ClickHouse cluster 'antman-prod' "
        "is not responding. Last successful query was 45 minutes ago. "
        "Possible cause: cluster maintenance or network partition."
    )


# ---------------------------------------------------------------------------
# Error Variant: Token Limit Exceeded
# ---------------------------------------------------------------------------

@neatlogs.span(kind="AGENT", name="root_cause_analysis_attempt",
               role="Diagnostic Agent", goal="Root cause analysis — attempt")
def _root_cause_attempt(state: AnalyticsState, context_data: str) -> dict:
    """Single attempt at root cause analysis with given context."""
    prompt = (
        "You are a root-cause analysis engine for an e-commerce analytics platform.\n"
        "Use structured decision-tree frameworks to diagnose business problems.\n\n"
        "Framework rules:\n"
        "  Revenue drop → availability → pricing → visibility → competition → seasonality\n"
        "  SOV decline  → keyword ranking → ad spend → new competitors → content quality\n\n"
        f"Query: {state['user_query']}\n"
        f"Intent: {state.get('intent')}\n"
        f"Data: {context_data}\n\n"
        "Return JSON with: framework, root_cause, contributing_factors (list), "
        "confidence (0–1), recommended_actions (list).\n"
        "Return ONLY valid JSON."
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    try:
        parsed = json.loads(response.content)
    except json.JSONDecodeError:
        parsed = {"framework": "revenue_diagnostic", "root_cause": "Analysis complete", "confidence": 0.75}
    return {
        "root_cause": response.content,
        "framework_used": parsed.get("framework", "general_diagnostic"),
        "confidence_score": parsed.get("confidence", 0.7),
        "messages": [response],
    }


@neatlogs.span(kind="CHAIN", name="root_cause_with_retry",
               metadata={"pattern": "token_limit_retry"})
def analyze_root_cause_with_token_retry(state: AnalyticsState) -> dict:
    """
    Root cause analysis that simulates token limit exceeded.
    First attempt: oversized context → TokenLimitError
    Second attempt: truncated context → success
    """
    query_results = state.get("query_results", {})
    full_context = json.dumps(query_results, indent=2) * 50  # Intentionally oversized

    # Attempt 1: oversized → raises TokenLimitError
    try:
        estimated_tokens = len(full_context) // 4
        if estimated_tokens > 16_000:
            raise TokenLimitError(
                f"Request too large: estimated {estimated_tokens} tokens "
                f"exceeds maximum context length of 16384 tokens. "
                f"Input contained {len(full_context)} characters."
            )
        return _root_cause_attempt(state, full_context)
    except TokenLimitError as e:
        print(f"   ⚠️  Token limit exceeded: {e}")
        print("   ✂️  Truncating context and retrying…")

    # Attempt 2: truncated context → success
    truncated = json.dumps(query_results, indent=2)[:4000]
    return _root_cause_attempt(state, truncated)


# ---------------------------------------------------------------------------
# Error Variant: Retry Storm
# ---------------------------------------------------------------------------

@neatlogs.span(kind="TOOL", name="execute_query_attempt", tool_name="antman_analytics_engine")
def _execute_query_attempt(state: AnalyticsState, attempt: int) -> dict:
    """Single query execution attempt."""
    if attempt == 1:
        time.sleep(0.5)
        raise ConnectionResetError(
            "Connection reset by peer: Antman cluster rejected the connection. "
            "Too many concurrent queries (limit: 100, current: 103)."
        )
    elif attempt == 2:
        time.sleep(1.0)
        raise TimeoutError(
            "Query execution timed out after 15000ms. "
            "Table 'orders' scan exceeded memory budget (4GB limit)."
        )
    else:
        # Attempt 3: success
        return execute_query(state)


@neatlogs.span(kind="CHAIN", name="execute_query_with_retry",
               metadata={"pattern": "retry_storm", "max_retries": 3})
def execute_query_with_retries(state: AnalyticsState) -> dict:
    """
    Execute query with retry storm: 3 attempts with exponential backoff.
    Attempt 1: ConnectionResetError
    Attempt 2: TimeoutError
    Attempt 3: Success
    """
    for attempt in range(1, 4):
        try:
            print(f"   🔄 Query attempt {attempt}/3…")
            return _execute_query_attempt(state, attempt)
        except (ConnectionResetError, TimeoutError) as e:
            wait = 1.0 * (2 ** (attempt - 1))
            print(f"   ❌ Attempt {attempt} failed: {type(e).__name__}: {e}")
            if attempt < 3:
                print(f"   ⏳ Waiting {wait:.0f}s before retry…")
                time.sleep(wait)
            else:
                raise


# ---------------------------------------------------------------------------
# Error Variant: SQL Hallucination
# ---------------------------------------------------------------------------

@neatlogs.span(kind="CHAIN", name="nlq_to_sql", goal="Generate ClickHouse SQL — hallucination variant")
def generate_sql_hallucinating(state: AnalyticsState) -> dict:
    """
    SQL generation that intentionally asks about non-existent data.
    The query asks for customer satisfaction / delivery partner data that
    doesn't exist in the schema.
    """
    prompt = (
        "You are a ClickHouse SQL expert. The user wants customer satisfaction scores "
        "by delivery partner across all cities.\n\n"
        "Generate a SQL query that retrieves customer satisfaction ratings grouped by "
        "delivery partner and city from the customer_satisfaction and delivery_partners tables.\n\n"
        "Return ONLY the SQL query."
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    return {
        "generated_sql": response.content.strip(),
        "messages": [response],
    }


@neatlogs.span(kind="GUARDRAIL", name="sql_validator",
               metadata={"guardrail_type": "hallucination_detection"})
def validate_generated_sql(state: AnalyticsState) -> dict:
    """
    Validates generated SQL against known schema.
    Catches hallucinated table/column references.
    """
    sql = state.get("generated_sql", "")
    result = validate_sql(sql)

    if result["hallucination_detected"]:
        print(f"   🚨 SQL Hallucination detected: invalid tables {result['invalid_tables']}")
        return {
            "query_results": {
                "status": "hallucination_detected",
                "hallucination_type": "schema_violation",
                "invalid_tables": result["invalid_tables"],
                "message": (
                    f"The requested data (tables: {', '.join(result['invalid_tables'])}) "
                    f"does not exist in the analytics schema. "
                    f"Available tables: orders, products, search_rankings, inventory, campaigns, pricing_history."
                ),
            },
            "messages": [AIMessage(
                content=f"[SQL Validator] Hallucination detected. "
                        f"Tables {result['invalid_tables']} do not exist in schema."
            )],
        }

    return {}


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def should_analyze_root_cause(state: AnalyticsState) -> str:
    """Trigger root-cause branch only when the data shows an anomaly."""
    results = state.get("query_results", {})
    if isinstance(results, dict):
        if results.get("revenue_change_pct", 0) < -5:
            return "analyze"
        if results.get("overall_availability", 100) < 85:
            return "analyze"
        if "symptoms_detected" in results:
            return "analyze"
    return "report"


def should_analyze_root_cause_token_limit(state: AnalyticsState) -> str:
    """Always route to root cause analysis for token limit demo."""
    results = state.get("query_results", {})
    if isinstance(results, dict):
        if results.get("revenue_change_pct", 0) < -5:
            return "analyze"
        if results.get("overall_availability", 100) < 85:
            return "analyze"
        if "symptoms_detected" in results:
            return "analyze"
    return "analyze"  # Force root cause for token limit demo


def route_after_sql_validation(state: AnalyticsState) -> str:
    """Skip execute_query if SQL hallucination was detected."""
    results = state.get("query_results", {})
    if isinstance(results, dict) and results.get("status") == "hallucination_detected":
        return "report"
    return "execute"


# ---------------------------------------------------------------------------
# Graph Factory
# ---------------------------------------------------------------------------

def build_analytics_agent(error_variant: str = None) -> StateGraph:
    """
    Build the analytics agent graph. Pass error_variant to get
    error-injecting variants for demo scenarios.
    """
    graph = StateGraph(AnalyticsState)

    if error_variant == "db_timeout":
        # Scenario 5: Database timeout
        graph.add_node("recognize_intent", recognize_intent)
        graph.add_node("generate_sql", generate_sql)
        graph.add_node("execute_query", execute_query_timeout)

        graph.add_edge(START, "recognize_intent")
        graph.add_edge("recognize_intent", "generate_sql")
        graph.add_edge("generate_sql", "execute_query")
        graph.add_edge("execute_query", END)

    elif error_variant == "token_limit":
        # Scenario 7: Token limit exceeded with retry
        graph.add_node("recognize_intent", recognize_intent)
        graph.add_node("generate_sql", generate_sql)
        graph.add_node("execute_query", execute_query)
        graph.add_node("analyze_root_cause", analyze_root_cause_with_token_retry)

        graph.add_edge(START, "recognize_intent")
        graph.add_edge("recognize_intent", "generate_sql")
        graph.add_edge("generate_sql", "execute_query")
        graph.add_conditional_edges(
            "execute_query",
            should_analyze_root_cause_token_limit,
            {"analyze": "analyze_root_cause", "report": END},
        )
        graph.add_edge("analyze_root_cause", END)

    elif error_variant == "retry_storm":
        # Scenario 12: Retry storm with backoff
        graph.add_node("recognize_intent", recognize_intent)
        graph.add_node("generate_sql", generate_sql)
        graph.add_node("execute_query", execute_query_with_retries)
        graph.add_node("analyze_root_cause", analyze_root_cause)

        graph.add_edge(START, "recognize_intent")
        graph.add_edge("recognize_intent", "generate_sql")
        graph.add_edge("generate_sql", "execute_query")
        graph.add_conditional_edges(
            "execute_query",
            should_analyze_root_cause,
            {"analyze": "analyze_root_cause", "report": END},
        )
        graph.add_edge("analyze_root_cause", END)

    elif error_variant == "hallucination_sql":
        # Scenario 13: SQL hallucination
        graph.add_node("recognize_intent", recognize_intent)
        graph.add_node("generate_sql", generate_sql_hallucinating)
        graph.add_node("validate_sql", validate_generated_sql)
        graph.add_node("execute_query", execute_query)
        graph.add_node("analyze_root_cause", analyze_root_cause)

        graph.add_edge(START, "recognize_intent")
        graph.add_edge("recognize_intent", "generate_sql")
        graph.add_edge("generate_sql", "validate_sql")
        graph.add_conditional_edges(
            "validate_sql",
            route_after_sql_validation,
            {"execute": "execute_query", "report": END},
        )
        graph.add_conditional_edges(
            "execute_query",
            should_analyze_root_cause,
            {"analyze": "analyze_root_cause", "report": END},
        )
        graph.add_edge("analyze_root_cause", END)

    else:
        # Default: original happy-path graph
        graph.add_node("recognize_intent", recognize_intent)
        graph.add_node("generate_sql", generate_sql)
        graph.add_node("execute_query", execute_query)
        graph.add_node("analyze_root_cause", analyze_root_cause)

        graph.add_edge(START, "recognize_intent")
        graph.add_edge("recognize_intent", "generate_sql")
        graph.add_edge("generate_sql", "execute_query")
        graph.add_conditional_edges(
            "execute_query",
            should_analyze_root_cause,
            {"analyze": "analyze_root_cause", "report": END},
        )
        graph.add_edge("analyze_root_cause", END)

    return graph.compile()
