"""
GobbsGPT — Supervisor / Orchestrator Agent
============================================
The central AI CXO copilot. Classifies every user query and routes it to the
appropriate specialised sub-agent, then synthesises a CXO-grade final response.

Pipeline:
  classify → (conditional) ─► analytics_agent ─┐
                             ─► ad_agent        ├─► synthesize → END
                             ─► inventory_agent ─┤
                             ─► market_intel     ┘
"""

import json
from typing import Optional, Annotated

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

import neatlogs
from config import llm, BRAND, CATEGORY
from agent_analytics import build_analytics_agent
from agent_ads import build_ad_automation_agent
from agent_inventory import build_inventory_agent
from agent_market_intel import build_market_intel_agent


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class SupervisorState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    user_query: str
    query_classification: Optional[str]
    delegated_to: Optional[str]
    sub_agent_result: Optional[dict]
    final_response: Optional[str]
    follow_up_suggestions: Optional[list]


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

@neatlogs.span(kind="AGENT", name="gobbs_gpt_classifier",
               role="GobbsGPT Router", goal="Classify query and route to correct sub-agent")
def classify_query(state: SupervisorState) -> dict:
    """
    GobbsGPT's first action: understand query intent and identify the owning agent.
    """
    prompt = (
        "You are GobbsGPT, an AI CXO copilot for e-commerce and quick-commerce brands.\n"
        "Classify the user query into EXACTLY one category:\n\n"
        "  ANALYTICS    — revenue, sales, SOV, pricing analysis, 'why did X happen'\n"
        "  ADS          — ad campaigns, ROAS, bidding, marketing spend, optimisation\n"
        "  INVENTORY    — stock levels, availability, stockouts, purchase orders, supply chain\n"
        "  MARKET_INTEL — market trends, competition, new opportunities, white space, NPD\n\n"
        f"Query: {state['user_query']}\n\n"
        "Return ONLY the category name: ANALYTICS, ADS, INVENTORY, or MARKET_INTEL."
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    classification = response.content.strip().upper().split()[0]   # first word, safety net

    valid = {"ANALYTICS", "ADS", "INVENTORY", "MARKET_INTEL"}
    if classification not in valid:
        classification = "ANALYTICS"

    print(f"\n🧠 GobbsGPT classified query → {classification}")
    return {"query_classification": classification, "messages": [response]}


# ---------------------  Sub-agent wrappers  --------------------------------

@neatlogs.span(kind="WORKFLOW", name="run_analytics_agent")
def run_analytics_agent(state: SupervisorState) -> dict:
    """Delegates to Gobbs Edge (Analytics Agent)."""
    print("📊 Delegating to Gobbs Edge (Analytics)…")
    agent = build_analytics_agent()
    result = agent.invoke({
        "messages": [],
        "user_query": state["user_query"],
        "intent": None, "entities": None,
        "generated_sql": None, "query_results": None,
        "root_cause": None, "framework_used": None,
        "confidence_score": None,
    })
    return {
        "delegated_to": "Gobbs Edge (Analytics)",
        "sub_agent_result": {
            "intent": result.get("intent"),
            "sql": result.get("generated_sql"),
            "query_results": result.get("query_results"),
            "root_cause": result.get("root_cause"),
            "framework": result.get("framework_used"),
            "confidence": result.get("confidence_score"),
        },
        "messages": [AIMessage(content="Gobbs Edge analysis complete.")],
    }


@neatlogs.span(kind="WORKFLOW", name="run_ad_automation_agent")
def run_ad_agent(state: SupervisorState) -> dict:
    """Delegates to Gobbs Boost (Ad Automation Agent)."""
    print("📣 Delegating to Gobbs Boost (Ad Automation)…")
    agent = build_ad_automation_agent()
    result = agent.invoke({
        "messages": [],
        "brand": BRAND,
        "platform": "blinkit",
        "campaign_goal": "maximize_roas",
        "budget": 50_000,
        "current_performance": None, "stock_context": None,
        "competitive_context": None, "bid_recommendations": None,
        "budget_allocation": None, "execution_plan": None,
    })
    return {
        "delegated_to": "Gobbs Boost (Ads)",
        "sub_agent_result": {
            "bid_recommendations": result.get("bid_recommendations"),
            "budget_allocation": result.get("budget_allocation"),
            "execution_plan": result.get("execution_plan"),
        },
        "messages": [AIMessage(content="Gobbs Boost optimisation complete.")],
    }


@neatlogs.span(kind="WORKFLOW", name="run_inventory_agent")
def run_inventory_agent(state: SupervisorState) -> dict:
    """Delegates to Gobbs Flow (Inventory Agent)."""
    print("📦 Delegating to Gobbs Flow (Inventory)…")
    agent = build_inventory_agent()
    result = agent.invoke({
        "messages": [],
        "brand": BRAND,
        "query_type": "stock_check",
        "inventory_snapshot": None, "demand_signals": None,
        "stockout_alerts": None, "po_recommendations": None,
        "forecast": None,
    })
    return {
        "delegated_to": "Gobbs Flow (Inventory)",
        "sub_agent_result": {
            "inventory_snapshot": result.get("inventory_snapshot"),
            "stockout_alerts": result.get("stockout_alerts"),
            "po_recommendations": result.get("po_recommendations"),
        },
        "messages": [AIMessage(content="Gobbs Flow inventory analysis complete.")],
    }


@neatlogs.span(kind="WORKFLOW", name="run_market_intel_agent")
def run_market_intel_agent(state: SupervisorState) -> dict:
    """Delegates to Gobbs Discover (Market Intelligence Agent)."""
    print("🌐 Delegating to Gobbs Discover (Market Intelligence)…")
    agent = build_market_intel_agent()
    result = agent.invoke({
        "messages": [],
        "brand": BRAND,
        "category": CATEGORY,
        "analysis_type": "comprehensive",
        "market_data": None, "trends": None,
        "opportunities": None, "competitive_moves": None,
        "strategic_recommendations": None,
    })
    return {
        "delegated_to": "Gobbs Discover (Market Intel)",
        "sub_agent_result": {
            "trends": result.get("trends"),
            "opportunities": result.get("opportunities"),
            "strategy": result.get("strategic_recommendations"),
        },
        "messages": [AIMessage(content="Gobbs Discover market intelligence complete.")],
    }


@neatlogs.span(kind="AGENT", name="gobbs_gpt_synthesiser",
               role="GobbsGPT CXO Advisor", goal="Synthesise sub-agent findings into executive brief")
def synthesize_response(state: SupervisorState) -> dict:
    """
    GobbsGPT's final step: distil sub-agent results into a CXO-friendly response
    with a clear narrative, key insights, and prioritised next actions.
    """
    prompt = (
        "You are GobbsGPT, an AI CXO copilot for a quick-commerce brand.\n"
        "Synthesise the analysis below into a clear, executive-level response.\n\n"
        f"Original question:   {state['user_query']}\n"
        f"Analysis source:     {state.get('delegated_to', 'Multi-agent')}\n"
        f"Analysis results:    {json.dumps(state.get('sub_agent_result', {}), indent=2)}\n\n"
        "Structure your response as:\n"
        "  1. **TL;DR** — one-sentence answer\n"
        "  2. **Key Insights** — 3–5 bullet points with specific numbers\n"
        "  3. **Recommended Actions** — prioritised list (label P0/P1/P2)\n"
        "  4. **Follow-up Questions** — 3 questions the CXO should ask next\n\n"
        "Be specific, reference actual numbers from the data. "
        "Professional but direct tone."
    )
    response = llm.invoke([HumanMessage(content=prompt)])

    follow_ups = [
        "What is the city-level breakdown of this impact?",
        "How does this compare to the same period last month?",
        "What budget or resource reallocation would you recommend?",
    ]

    return {
        "final_response": response.content,
        "follow_up_suggestions": follow_ups,
        "messages": [response],
    }


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def route_to_agent(state: SupervisorState) -> str:
    routing = {
        "ANALYTICS":    "analytics_agent",
        "ADS":          "ad_agent",
        "INVENTORY":    "inventory_agent",
        "MARKET_INTEL": "market_intel_agent",
    }
    return routing.get(state.get("query_classification", "ANALYTICS"), "analytics_agent")


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

def build_gobbs_gpt_supervisor() -> StateGraph:
    graph = StateGraph(SupervisorState)

    # Nodes
    graph.add_node("classify",           classify_query)
    graph.add_node("analytics_agent",    run_analytics_agent)
    graph.add_node("ad_agent",           run_ad_agent)
    graph.add_node("inventory_agent",    run_inventory_agent)
    graph.add_node("market_intel_agent", run_market_intel_agent)
    graph.add_node("synthesize",         synthesize_response)

    # Edges
    graph.add_edge(START, "classify")
    graph.add_conditional_edges(
        "classify",
        route_to_agent,
        {
            "analytics_agent":    "analytics_agent",
            "ad_agent":           "ad_agent",
            "inventory_agent":    "inventory_agent",
            "market_intel_agent": "market_intel_agent",
        },
    )

    # All sub-agents converge to synthesis
    for node in ("analytics_agent", "ad_agent", "inventory_agent", "market_intel_agent"):
        graph.add_edge(node, "synthesize")

    graph.add_edge("synthesize", END)

    return graph.compile()
