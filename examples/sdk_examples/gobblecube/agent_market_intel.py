"""
Gobbs Discover — Market Intelligence Agent
==========================================
Spots micro-category growth trends, white-space opportunities, and competitive moves
at a hyperlocal level.

Pipeline:
  gather_market_data → detect_trends → find_white_space → generate_strategy → END
"""

import json
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

class MarketIntelState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    brand: str
    category: str
    analysis_type: Optional[str]         # "trend" | "whitespace" | "competitive" | "comprehensive"
    market_data: Optional[dict]
    trends: Optional[list]
    opportunities: Optional[list]
    competitive_moves: Optional[list]
    strategic_recommendations: Optional[str]


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

@neatlogs.span(kind="TOOL", name="market_data_aggregator",
               tool_name="market_data_api")
def gather_market_data(state: MarketIntelState) -> dict:
    """
    Aggregates market-level data across platforms and subcategories.
    [DUMMY API] — simulates Gobbs Discover's data aggregation layer.
    """
    data = {
        "category": state.get("category", "health_snacks"),
        "market_size_inr_cr": 4_200,
        "market_size_trend": {"6mo_growth_pct": 18.5, "yoy_growth_pct": 42.3},
        "top_brands": [
            {"brand": "Brand A",                           "market_share_pct": 22.1, "trend": "stable"},
            {"brand": "Brand B",                           "market_share_pct": 18.7, "trend": "growing"},
            {"brand": state.get("brand", "Demo Brand"),    "market_share_pct": 15.3, "trend": "growing"},
            {"brand": "Brand C",                           "market_share_pct": 11.2, "trend": "declining"},
        ],
        "subcategory_breakdown": [
            {"subcategory": "Protein Bars",   "growth_pct": 35.2, "saturation": "medium", "yoy_new_skus": 45},
            {"subcategory": "Granola",        "growth_pct": 12.1, "saturation": "high",   "yoy_new_skus": 8},
            {"subcategory": "Keto Snacks",    "growth_pct": 68.5, "saturation": "low",    "yoy_new_skus": 22},
            {"subcategory": "Trail Mix",      "growth_pct": 28.0, "saturation": "low",    "yoy_new_skus": 12},
            {"subcategory": "Makhana Snacks", "growth_pct": 92.0, "saturation": "very_low","yoy_new_skus": 18},
        ],
        "pricing_landscape": {
            "avg_category_price_inr": 180,
            "price_range": [99, 450],
            "gap_zones": [
                "₹150–200 (underserved — no strong brand)",
                "₹300–350 (no premium options except imports)",
            ],
        },
        "platform_distribution": {
            "blinkit":   {"gmv_share_pct": 42, "trend": "gaining"},
            "zepto":     {"gmv_share_pct": 31, "trend": "gaining"},
            "instamart": {"gmv_share_pct": 27, "trend": "stable"},
        },
        "city_tier_insights": {
            "tier_1": {"growth_pct": 28, "top_cities": ["Mumbai", "Delhi", "Bangalore"]},
            "tier_2": {"growth_pct": 58, "top_cities": ["Pune", "Hyderabad", "Chennai"]},
        },
    }

    return {
        "market_data": data,
        "messages": [AIMessage(content=f"[Market Data API] Data aggregated for category: {data['category']}")],
    }


@neatlogs.span(kind="AGENT", name="trend_detector",
               role="Trend Analyst", goal="Identify micro-category growth trends")
def detect_trends(state: MarketIntelState) -> dict:
    """
    Identifies emerging micro-trends, demand velocity shifts, and new search patterns.
    """
    prompt_template = neatlogs.PromptTemplate(
        "You are a market intelligence analyst for quick commerce.\n\n"
        "Market data: {{market_data_json}}\n\n"
        "Identify emerging trends. Return JSON array where each item has:\n"
        "  trend_name, description, growth_signal_strength (1–10), "
        "  time_horizon (immediate/3mo/6mo), relevant_cities (list), "
        "  supporting_data_points (list of strings).\n"
        "Return ONLY valid JSON."
    )
    with neatlogs.trace("market_trends_prompt", kind="LLM", prompt_template=prompt_template):
        prompt = prompt_template.compile(
            market_data_json=json.dumps(state.get("market_data", {})),
        )
        response = llm.invoke([HumanMessage(content=prompt)])
    try:
        trends = json.loads(response.content)
    except json.JSONDecodeError:
        trends = []

    return {"trends": trends, "messages": [response]}


@neatlogs.span(kind="AGENT", name="whitespace_finder",
               role="Opportunity Analyst", goal="Identify white-space opportunities and pricing gaps")
def find_white_space(state: MarketIntelState) -> dict:
    """
    Pinpoints untapped market opportunities: pricing gaps, unsaturated subcategories,
    geographic voids, and format gaps.
    """
    prompt_template = neatlogs.PromptTemplate(
        "You are a strategic market analyst for consumer brands in quick commerce.\n\n"
        "Market data: {{market_data_json}}\n"
        "Trends: {{trends_json}}\n\n"
        "Find white-space opportunities:\n"
        "  1. Price points with no strong brand presence\n"
        "  2. Subcategories with high growth but low competition\n"
        "  3. Geographic markets with unmet demand (especially Tier-2)\n"
        "  4. Product format gaps (size, packaging, flavour)\n\n"
        "Return JSON array. Each item: opportunity_name, type "
        "(price_gap/subcategory_gap/geo_gap/format_gap), "
        "market_size_estimate_inr_cr, competition_level (low/medium/high), "
        "recommended_action, priority_score (1–10), time_to_market_months.\n"
        "Return ONLY valid JSON."
    )
    with neatlogs.trace("market_whitespace_prompt", kind="LLM", prompt_template=prompt_template):
        prompt = prompt_template.compile(
            market_data_json=json.dumps(state.get("market_data", {})),
            trends_json=json.dumps(state.get("trends", [])),
        )
        response = llm.invoke([HumanMessage(content=prompt)])
    try:
        opportunities = json.loads(response.content)
    except json.JSONDecodeError:
        opportunities = []

    return {"opportunities": opportunities, "messages": [response]}


@neatlogs.span(kind="AGENT", name="strategy_synthesiser",
               role="Growth Strategy Advisor", goal="Synthesise actionable CXO-level strategy")
def generate_strategy(state: MarketIntelState) -> dict:
    """
    Synthesises market data, trends, and opportunities into a CXO-grade strategic brief.
    """
    prompt_template = neatlogs.PromptTemplate(
        "You are a growth strategy advisor for a consumer brand in quick commerce.\n\n"
        "Market data: {{market_data_json}}\n"
        "Trends:       {{trends_json}}\n"
        "White space:  {{opportunities_json}}\n\n"
        "Write an executive strategic brief (structured, concise):\n"
        "  1. Top 3 strategic priorities for next quarter\n"
        "  2. Quick wins (actionable in 1–2 weeks)\n"
        "  3. Medium-term bets (1–3 months)\n"
        "  4. Risks to monitor\n"
        "  5. One bold contrarian move\n\n"
        "Be specific with numbers and subcategory names. "
        "Write in a professional but direct tone suitable for a Brand CXO."
    )
    with neatlogs.trace("market_strategy_prompt", kind="LLM", prompt_template=prompt_template):
        prompt = prompt_template.compile(
            market_data_json=json.dumps(state.get("market_data", {})),
            trends_json=json.dumps(state.get("trends", [])),
            opportunities_json=json.dumps(state.get("opportunities", [])),
        )
        response = llm.invoke([HumanMessage(content=prompt)])
    return {
        "strategic_recommendations": response.content,
        "messages": [response],
    }


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

def build_market_intel_agent() -> StateGraph:
    graph = StateGraph(MarketIntelState)

    graph.add_node("gather_data",       gather_market_data)
    graph.add_node("detect_trends",     detect_trends)
    graph.add_node("find_whitespace",   find_white_space)
    graph.add_node("generate_strategy", generate_strategy)

    graph.add_edge(START,              "gather_data")
    graph.add_edge("gather_data",      "detect_trends")
    graph.add_edge("detect_trends",    "find_whitespace")
    graph.add_edge("find_whitespace",  "generate_strategy")
    graph.add_edge("generate_strategy", END)

    return graph.compile()
