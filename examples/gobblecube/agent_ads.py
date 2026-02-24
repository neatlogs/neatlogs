"""
Gobbs Boost — Ad Automation Agent
====================================
Mirrors GobbleCube's agentic performance marketing tool.
Integrates digital-shelf data (stock + competitive signals) with bid decisions.

Pipeline:
  gather_context → generate_bid_recommendations → allocate_budget → generate_execution_plan → END

Error variants:
  auth_error — gather_campaign_context raises ExternalAPIError(401) for auth failure
"""

import json
from typing import Optional, Annotated

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

import neatlogs
from config import llm
from error_injection import ExternalAPIError


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class AdAutomationState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    brand: str
    platform: str
    campaign_goal: Optional[str]
    budget: Optional[float]
    current_performance: Optional[dict]
    stock_context: Optional[dict]
    competitive_context: Optional[dict]
    bid_recommendations: Optional[list]
    budget_allocation: Optional[dict]
    execution_plan: Optional[dict]
    error_variant: Optional[str]


# ---------------------------------------------------------------------------
# Nodes (Original — happy path)
# ---------------------------------------------------------------------------

@neatlogs.span(kind="TOOL", name="gather_campaign_context",
               tool_name="digital_shelf_data_api")
def gather_campaign_context(state: AdAutomationState) -> dict:
    """
    Pulls current campaign performance, live stock levels, and competitive data.
    Gobbs Boost is distinguished by fusing digital-shelf data into ad decisions.
    [DUMMY API] — simulates calls to Gobbs Edge + Gobbs Flow data APIs.
    """
    performance = {
        "active_campaigns": 12,
        "total_spend_today": 45_000,
        "avg_roas": 3.2,
        "top_keywords": [
            {"keyword": "protein bar",   "spend": 8_000, "roas": 4.1, "rank": 2},
            {"keyword": "healthy snacks","spend": 6_000, "roas": 2.8, "rank": 5},
            {"keyword": "energy bar",    "spend": 4_000, "roas": 1.9, "rank": 8},
            {"keyword": "keto bar",      "spend": 2_000, "roas": 5.2, "rank": 1},
        ],
    }

    stock = {
        "sku_availability": {
            "SKU-001": {"available_stores": 85, "total_stores": 100, "name": "Protein Bar 60g"},
            "SKU-002": {"available_stores": 42, "total_stores": 100, "name": "Oat Granola 200g"},
            "SKU-003": {"available_stores": 95, "total_stores": 100, "name": "Keto Bar 50g"},
            "SKU-004": {"available_stores": 30, "total_stores": 100, "name": "Energy Bar 80g"},
        },
    }

    competitive = {
        "competitor_sov": {"Competitor A": 28.5, "Competitor B": 22.1, "Competitor C": 18.0},
        "brand_sov": 23.5,
        "emerging_keywords": ["keto bar", "plant protein", "sugar free snack"],
        "competitor_price_cuts": ["Competitor A dropped protein bar price by 10%"],
    }

    return {
        "current_performance": performance,
        "stock_context": stock,
        "competitive_context": competitive,
        "messages": [AIMessage(content="[Digital Shelf API] Campaign context fetched from Gobbs Edge + Gobbs Flow.")],
    }


@neatlogs.span(kind="AGENT", name="bid_recommendation_engine",
               role="Bid Optimizer", goal="Generate keyword-level bid adjustments")
def generate_bid_recommendations(state: AdAutomationState) -> dict:
    """
    LLM-powered bid adjustments driven by stock × competition × ROAS signals.
    Uses GobbleCube's digital-shelf-powered ruleset approach.
    """
    prompt = (
        "You are an AI ad optimisation engine for quick-commerce platforms (Blinkit, Zepto, Instamart).\n"
        "Generate bid recommendations using these rules:\n\n"
        "  1. SKU availability < 50%  → PAUSE or REDUCE bids for its keywords\n"
        "  2. Brand rank #1-2 for keyword → REDUCE bid (already dominant)\n"
        "  3. Competitor SOV rising on keyword → INCREASE bid defensively (+15%)\n"
        "  4. ROAS < 2.0 → REDUCE bid or PAUSE keyword\n"
        "  5. ROAS > 4.0 AND availability > 80% → INCREASE bid to capture more (+20%)\n"
        "  6. Allocate freed budget to emerging keyword opportunities\n\n"
        f"Performance: {json.dumps(state['current_performance'])}\n"
        f"Stock:       {json.dumps(state['stock_context'])}\n"
        f"Competitive: {json.dumps(state['competitive_context'])}\n"
        f"Goal:   {state.get('campaign_goal', 'maximize_roas')}\n"
        f"Budget: ₹{state.get('budget', 50_000):,}\n\n"
        "Return a JSON array. Each item: keyword, action (INCREASE/REDUCE/PAUSE/MAINTAIN), "
        "reason, current_bid_amount, new_bid_amount, expected_roas_impact.\n"
        "Return ONLY valid JSON."
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    try:
        recommendations = json.loads(response.content)
    except json.JSONDecodeError:
        recommendations = []

    return {"bid_recommendations": recommendations, "messages": [response]}


@neatlogs.span(kind="CHAIN", name="budget_allocator",
               goal="Optimally distribute budget across keywords and time slots")
def allocate_budget(state: AdAutomationState) -> dict:
    """
    Distributes total budget optimally across keywords, cities, and time windows.
    Quick-commerce has strong morning/evening peaks — that is factored in.
    """
    prompt = (
        "You are a budget allocation optimiser for quick-commerce ads.\n\n"
        f"Bid recommendations: {json.dumps(state.get('bid_recommendations', []))}\n"
        f"Total budget: ₹{state.get('budget', 50_000):,}\n"
        f"Goal: {state.get('campaign_goal', 'maximize_roas')}\n\n"
        "Create an allocation plan. Consider:\n"
        "  - Morning (7-10 AM) and evening (6-9 PM) peaks for quick commerce\n"
        "  - City-level stock availability (don't spend in cities where SKUs are OOS)\n"
        "  - ROAS expectations per keyword\n\n"
        "Return JSON with: keyword_allocations (list with keyword + amount + pct), "
        "time_split (morning_pct, afternoon_pct, evening_pct), "
        "city_priorities (list), expected_total_roas.\n"
        "Return ONLY valid JSON."
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    try:
        allocation = json.loads(response.content)
    except json.JSONDecodeError:
        allocation = {"status": "allocation_generated"}

    return {"budget_allocation": allocation, "messages": [response]}


@neatlogs.span(kind="TOOL", name="execution_plan_compiler",
               tool_name="campaign_execution_api")
def generate_execution_plan(state: AdAutomationState) -> dict:
    """
    Compiles the final execution plan: bid changes + budget allocation + auto-rules.
    [DUMMY API] — in production this would push changes to Blinkit/Zepto ad APIs.
    """
    plan = {
        "platform": state.get("platform", "blinkit"),
        "brand": state.get("brand", "Demo Brand"),
        "bid_changes": state.get("bid_recommendations", []),
        "budget_allocation": state.get("budget_allocation", {}),
        "auto_rules": [
            "Pause keywords automatically when SKU availability drops below 40%",
            "Increase bids by 15% when any competitor SOV rises above 30%",
            "Shift budget to evening slots on Friday–Sunday",
            "Alert if daily spend exceeds 120% of planned budget",
        ],
        "monitoring_triggers": [
            "ROAS drops below 2.0 for any keyword",
            "Any keyword loses top-3 ranking",
            "Competitor launches a flash sale (price drop > 20%)",
        ],
        "estimated_roas_improvement": "+0.4–0.8x",
    }

    return {
        "execution_plan": plan,
        "messages": [AIMessage(content=f"[Campaign API] Execution plan ready. Changes staged for review.")],
    }


# ---------------------------------------------------------------------------
# Error Variant: Authentication Error
# ---------------------------------------------------------------------------

@neatlogs.span(kind="TOOL", name="gather_campaign_context",
               tool_name="ad_platform_api")
def gather_campaign_context_auth_error(state: AdAutomationState) -> dict:
    """
    Simulates a 401 authentication failure from the ad platform API.
    This represents an expired OAuth token or revoked API credentials.
    """
    raise ExternalAPIError(
        status_code=401,
        service="ad_platform_api",
        message=(
            "Authentication failed: OAuth token expired at 2026-02-24T08:30:00Z. "
            "The ad_platform_api service requires a valid bearer token. "
            "Last successful authentication was 25 hours ago. "
            "Please refresh credentials via the platform settings dashboard."
        ),
    )


# ---------------------------------------------------------------------------
# Graph Factory
# ---------------------------------------------------------------------------

def build_ad_automation_agent(error_variant: str = None) -> StateGraph:
    """
    Build the ad automation agent graph. Pass error_variant to get
    error-injecting variants for demo scenarios.
    """
    graph = StateGraph(AdAutomationState)

    if error_variant == "auth_error":
        # Scenario 9 (partial failure): auth error on gather_context
        graph.add_node("gather_context", gather_campaign_context_auth_error)
        graph.add_node("generate_bids", generate_bid_recommendations)
        graph.add_node("allocate_budget", allocate_budget)
        graph.add_node("execution_plan", generate_execution_plan)

        graph.add_edge(START,            "gather_context")
        graph.add_edge("gather_context", "generate_bids")
        graph.add_edge("generate_bids",  "allocate_budget")
        graph.add_edge("allocate_budget","execution_plan")
        graph.add_edge("execution_plan", END)

    else:
        # Default: original happy-path graph
        graph.add_node("gather_context",    gather_campaign_context)
        graph.add_node("generate_bids",     generate_bid_recommendations)
        graph.add_node("allocate_budget",   allocate_budget)
        graph.add_node("execution_plan",    generate_execution_plan)

        graph.add_edge(START,            "gather_context")
        graph.add_edge("gather_context", "generate_bids")
        graph.add_edge("generate_bids",  "allocate_budget")
        graph.add_edge("allocate_budget","execution_plan")
        graph.add_edge("execution_plan", END)

    return graph.compile()
