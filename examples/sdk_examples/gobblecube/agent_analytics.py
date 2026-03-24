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
"""

import json
import os
from typing import Optional, Annotated

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

import neatlogs
from config import llm


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


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

@neatlogs.span(kind="AGENT", name="intent_recognition", role="Intent Classifier",
               goal="Classify analytics query intent and extract entities")
def recognize_intent(state: AnalyticsState) -> dict:
    """
    Step 1 of NLQ-to-SQL: classify intent and extract entities.
    Mirrors GobbleCube's first pipeline stage.
    """
    prompt_template = neatlogs.PromptTemplate(
        "You are an e-commerce analytics intent classifier for a quick-commerce platform.\n"
        "Analyze the query and return a JSON object with:\n"
        '  - "intent": one of [revenue_analysis, share_of_search, pricing_analysis, '
        "availability_check, campaign_performance, root_cause_analysis]\n"
        '  - "entities": dict with any of: brand, platform (blinkit/zepto/instamart), '
        "city, category, time_period, metric\n\n"
        "Query: {{user_query}}\n\n"
        "Return ONLY valid JSON."
    )
    with neatlogs.trace("analytics_intent_prompt", kind="LLM", prompt_template=prompt_template):
        prompt = prompt_template.compile(user_query=state["user_query"])
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
    prompt_template = neatlogs.PromptTemplate(
        "You are a ClickHouse SQL expert generating queries for an e-commerce analytics platform.\n\n"
        "{{schema}}\n"
        "Intent: {{intent}}\n"
        "Entities: {{entities_json}}\n"
        "Original Query: {{user_query}}\n\n"
        "Return ONLY the SQL query, no explanation or markdown fences."
    )
    with neatlogs.trace("analytics_sql_prompt", kind="LLM", prompt_template=prompt_template):
        prompt = prompt_template.compile(
            schema=schema.strip(),
            intent=state["intent"],
            entities_json=json.dumps(state.get("entities", {})),
            user_query=state["user_query"],
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
    force_error = True
    query_text = str(state.get("user_query", "")).lower()
    if force_error or "[force_tool_error]" in query_text:
        raise ConnectionResetError(
            "Connection reset by peer: Antman cluster rejected the connection. "
            "Too many concurrent queries (limit: 100, current: 103)."
        )

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
    prompt_template = neatlogs.PromptTemplate(
        "You are a root-cause analysis engine for an e-commerce analytics platform.\n"
        "Use structured decision-tree frameworks to diagnose business problems.\n\n"
        "Framework rules:\n"
        "  Revenue drop → availability → pricing → visibility → competition → seasonality\n"
        "  SOV decline  → keyword ranking → ad spend → new competitors → content quality\n"
        "  Availability → supply chain → demand spike → warehouse allocation\n\n"
        "Query: {{user_query}}\n"
        "Intent: {{intent}}\n"
        "Data: {{query_results_json}}\n\n"
        "Return JSON with: framework, root_cause, contributing_factors (list), "
        "confidence (0–1), recommended_actions (list).\n"
        "Return ONLY valid JSON."
    )
    with neatlogs.trace("analytics_root_cause_prompt", kind="LLM", prompt_template=prompt_template):
        prompt = prompt_template.compile(
            user_query=state["user_query"],
            intent=state.get("intent"),
            query_results_json=json.dumps(state.get("query_results", {})),
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


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

def build_analytics_agent() -> StateGraph:
    graph = StateGraph(AnalyticsState)

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
