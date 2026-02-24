"""
GobbsGPT — Supervisor / Orchestrator Agent
============================================
The central AI CXO copilot. Classifies every user query and routes it to the
appropriate specialised sub-agent, then synthesises a CXO-grade final response.

Enhanced with:
  - Content guardrail (prompt injection, profanity, hostile language)
  - Multi-agent routing (queries needing 2+ agents)
  - Error variant passthrough for demo scenarios
  - Hallucination detection (routing validator)

Pipeline:
  guardrail → classify → (conditional) ─► analytics_agent ─┐
                                          ─► ad_agent        ├─► synthesize → END
                                          ─► inventory_agent ─┤
                                          ─► market_intel     │
                                          ─► multi_agent ─────┘
              └──(BLOCKED)────────────────────────────────────► synthesize → END
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
from error_injection import (
    detect_content_issues,
    validate_routing,
    ExternalAPIError,
)


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
    # --- Enhanced fields ---
    error_variant: Optional[str]
    guardrail_action: Optional[str]
    guardrail_reason: Optional[str]
    sanitized_query: Optional[str]
    moderation_metadata: Optional[dict]


# ---------------------------------------------------------------------------
# Multi-Agent Combos
# ---------------------------------------------------------------------------

MULTI_AGENT_COMBOS = {
    "ANALYTICS_INVENTORY": ["analytics", "inventory"],
    "ADS_INVENTORY": ["ads", "inventory"],
    "ANALYTICS_ADS": ["analytics", "ads"],
}


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

@neatlogs.span(kind="GUARDRAIL", name="content_guardrail",
               metadata={"guardrail_type": "content_moderation"})
def check_content_guardrail(state: SupervisorState) -> dict:
    """
    Content guardrail: checks for prompt injection, profanity, and hostile language.
    Actions: BLOCK, SANITIZE, FLAG_AND_PROCEED, ALLOW
    """
    query = state["user_query"]
    result = detect_content_issues(query)

    action = result["action"]
    print(f"\n🛡️  Guardrail: {action}" + (f" — {result['reason']}" if result["reason"] else ""))

    updates = {
        "guardrail_action": action,
        "guardrail_reason": result["reason"],
        "moderation_metadata": {
            "flagged_terms": result["flagged_terms"],
            "original_query": query,
        },
    }

    if action == "BLOCK":
        updates["query_classification"] = "BLOCKED"
        updates["sub_agent_result"] = {
            "blocked": True,
            "reason": result["reason"],
        }

    elif action == "SANITIZE":
        updates["sanitized_query"] = result["sanitized_query"]
        updates["user_query"] = result["sanitized_query"]
        print(f"   ✏️  Sanitized query: {result['sanitized_query']}")

    elif action == "FLAG_AND_PROCEED":
        updates["moderation_metadata"]["escalation"] = "human_review_required"
        print("   🚩 Flagged for human review — proceeding with query")

    return updates


@neatlogs.span(kind="AGENT", name="gobbs_gpt_classifier",
               role="GobbsGPT Router", goal="Classify query and route to correct sub-agent")
def classify_query(state: SupervisorState) -> dict:
    """
    GobbsGPT's first action: understand query intent and identify the owning agent.
    Enhanced to support multi-agent classifications.
    """
    query = state.get("sanitized_query") or state["user_query"]

    prompt = (
        "You are GobbsGPT, an AI CXO copilot for e-commerce and quick-commerce brands.\n"
        "Classify the user query into EXACTLY one category:\n\n"
        "  ANALYTICS          — revenue, sales, SOV, pricing analysis, 'why did X happen'\n"
        "  ADS                — ad campaigns, ROAS, bidding, marketing spend, optimisation\n"
        "  INVENTORY          — stock levels, availability, stockouts, purchase orders, supply chain, expiry\n"
        "  MARKET_INTEL       — market trends, competition, new opportunities, white space, NPD\n"
        "  ANALYTICS_INVENTORY — query needs BOTH revenue/sales analysis AND stock/inventory data\n"
        "  ADS_INVENTORY      — query needs BOTH ad performance AND stock/inventory data\n\n"
        f"Query: {query}\n\n"
        "Return ONLY the category name."
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    classification = response.content.strip().upper().replace(" ", "_")
    # Extract first valid classification token
    for token in classification.split():
        token = token.strip()
        if token in {"ANALYTICS", "ADS", "INVENTORY", "MARKET_INTEL",
                      "ANALYTICS_INVENTORY", "ADS_INVENTORY", "ANALYTICS_ADS"}:
            classification = token
            break
    else:
        valid_single = {"ANALYTICS", "ADS", "INVENTORY", "MARKET_INTEL"}
        if classification not in valid_single and classification not in MULTI_AGENT_COMBOS:
            classification = "ANALYTICS"

    print(f"\n🧠 GobbsGPT classified query → {classification}")
    return {"query_classification": classification, "messages": [response]}


@neatlogs.span(kind="GUARDRAIL", name="routing_validator",
               metadata={"guardrail_type": "hallucination_detection"})
def validate_classification(state: SupervisorState) -> dict:
    """
    Validates the classifier's routing decision against keyword signals.
    Catches misclassification hallucinations and corrects them.
    """
    classification = state.get("query_classification", "ANALYTICS")

    # Only validate single-agent classifications
    if classification in MULTI_AGENT_COMBOS or classification == "BLOCKED":
        return {}

    query = state.get("sanitized_query") or state["user_query"]
    result = validate_routing(query, classification)

    if result.get("hallucination_detected"):
        corrected = result["corrected_classification"]
        print(f"   🔄 Routing correction: {classification} → {corrected} ({result['reason']})")
        return {
            "query_classification": corrected,
            "moderation_metadata": {
                **(state.get("moderation_metadata") or {}),
                "routing_correction": {
                    "original": classification,
                    "corrected": corrected,
                    "reason": result["reason"],
                    "hallucination_type": "misclassification",
                },
            },
        }

    return {}


# ---------------------  Sub-agent wrappers  --------------------------------

@neatlogs.span(kind="AGENT", name="run_analytics_agent",
               agent_name="gobbs_edge", role="Gobbs Edge Analytics")
def run_analytics_agent(state: SupervisorState) -> dict:
    """Delegates to Gobbs Edge (Analytics Agent)."""
    print("📊 Delegating to Gobbs Edge (Analytics)…")
    error_variant = state.get("error_variant")
    agent = build_analytics_agent(error_variant=error_variant)
    result = agent.invoke({
        "messages": [],
        "user_query": state.get("sanitized_query") or state["user_query"],
        "intent": None, "entities": None,
        "generated_sql": None, "query_results": None,
        "root_cause": None, "framework_used": None,
        "confidence_score": None,
        "error_variant": error_variant,
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


@neatlogs.span(kind="AGENT", name="run_ad_automation_agent",
               agent_name="gobbs_boost", role="Gobbs Boost Ads")
def run_ad_agent(state: SupervisorState) -> dict:
    """Delegates to Gobbs Boost (Ad Automation Agent)."""
    print("📣 Delegating to Gobbs Boost (Ad Automation)…")
    error_variant = state.get("error_variant")
    agent = build_ad_automation_agent(error_variant=error_variant)
    result = agent.invoke({
        "messages": [],
        "brand": BRAND,
        "platform": "blinkit",
        "campaign_goal": "maximize_roas",
        "budget": 50_000,
        "current_performance": None, "stock_context": None,
        "competitive_context": None, "bid_recommendations": None,
        "budget_allocation": None, "execution_plan": None,
        "error_variant": error_variant,
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


@neatlogs.span(kind="AGENT", name="run_inventory_agent",
               agent_name="gobbs_flow", role="Gobbs Flow Inventory")
def run_inventory_agent(state: SupervisorState) -> dict:
    """Delegates to Gobbs Flow (Inventory Agent)."""
    print("📦 Delegating to Gobbs Flow (Inventory)…")
    error_variant = state.get("error_variant")
    agent = build_inventory_agent(error_variant=error_variant)
    result = agent.invoke({
        "messages": [],
        "brand": BRAND,
        "query_type": "stock_check",
        "inventory_snapshot": None, "demand_signals": None,
        "stockout_alerts": None, "po_recommendations": None,
        "forecast": None,
        "error_variant": error_variant,
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


@neatlogs.span(kind="AGENT", name="run_market_intel_agent",
               agent_name="gobbs_discover", role="Gobbs Discover Market Intel")
def run_market_intel_agent(state: SupervisorState) -> dict:
    """Delegates to Gobbs Discover (Market Intelligence Agent)."""
    print("🌐 Delegating to Gobbs Discover (Market Intelligence)…")
    error_variant = state.get("error_variant")
    agent = build_market_intel_agent(error_variant=error_variant)
    result = agent.invoke({
        "messages": [],
        "brand": BRAND,
        "category": CATEGORY,
        "analysis_type": "comprehensive",
        "market_data": None, "trends": None,
        "opportunities": None, "competitive_moves": None,
        "strategic_recommendations": None,
        "error_variant": error_variant,
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


# ---------------------  Multi-agent cascade  --------------------------------

AGENT_RUNNERS = {
    "analytics": run_analytics_agent,
    "ads": run_ad_agent,
    "inventory": run_inventory_agent,
    "market_intel": run_market_intel_agent,
}


@neatlogs.span(kind="WORKFLOW", name="run_multi_agent_cascade",
               metadata={"pattern": "sequential_cascade"})
def run_multi_agents(state: SupervisorState) -> dict:
    """
    Run multiple sub-agents sequentially and merge results.
    Handles partial failures gracefully — if one agent fails, the others' results
    are still available for synthesis.
    """
    combo = state.get("query_classification", "")
    agent_keys = MULTI_AGENT_COMBOS.get(combo, [])

    print(f"\n🔀 Multi-agent cascade: {', '.join(agent_keys)}")

    combined_results = {}
    delegated_names = []
    errors = []

    for agent_key in agent_keys:
        runner = AGENT_RUNNERS.get(agent_key)
        if not runner:
            continue
        try:
            result = runner(state)
            combined_results[agent_key] = result.get("sub_agent_result", {})
            delegated_names.append(result.get("delegated_to", agent_key))
        except Exception as e:
            print(f"   ❌ {agent_key} failed: {e}")
            errors.append({"agent": agent_key, "error": str(e), "error_type": type(e).__name__})

    return {
        "delegated_to": f"Multi-Agent: {', '.join(delegated_names)}",
        "sub_agent_result": {
            "combined": combined_results,
            "errors": errors,
            "partial_failure": len(errors) > 0,
            "agents_succeeded": len(combined_results),
            "agents_failed": len(errors),
        },
        "messages": [AIMessage(content="Multi-agent cascade complete.")],
    }


# ---------------------  Synthesizer  ----------------------------------------

@neatlogs.span(kind="AGENT", name="gobbs_gpt_synthesiser",
               role="GobbsGPT CXO Advisor", goal="Synthesise sub-agent findings into executive brief")
def synthesize_response(state: SupervisorState) -> dict:
    """
    GobbsGPT's final step: distil sub-agent results into a CXO-friendly response
    with a clear narrative, key insights, and prioritised next actions.
    Handles blocked queries, partial failures, and moderation flags.
    """
    # Handle blocked queries
    if state.get("guardrail_action") == "BLOCK":
        return {
            "final_response": (
                "I'm sorry, but I cannot process that request. "
                "Please rephrase your question about business analytics, "
                "ad campaigns, inventory, or market intelligence."
            ),
            "follow_up_suggestions": [
                "What is the current revenue trend for our brand?",
                "Show me our ad campaign performance on Blinkit",
                "Which SKUs are at risk of stocking out?",
            ],
            "messages": [AIMessage(content="Query blocked by content guardrail.")],
        }

    # Build context for synthesis
    sub_result = state.get("sub_agent_result", {})
    partial_failure = sub_result.get("partial_failure", False)

    extra_context = ""
    if partial_failure:
        extra_context = (
            "\n\n⚠️ NOTE: This was a multi-agent query. Some agents failed:\n"
            f"  Succeeded: {sub_result.get('agents_succeeded', 0)}\n"
            f"  Failed: {sub_result.get('agents_failed', 0)}\n"
            f"  Errors: {json.dumps(sub_result.get('errors', []))}\n"
            "Please acknowledge the partial data in your response."
        )

    moderation_note = ""
    if state.get("guardrail_action") == "FLAG_AND_PROCEED":
        moderation_note = "\nNote: This query was flagged for tone. Respond professionally and helpfully."
    elif state.get("guardrail_action") == "SANITIZE":
        moderation_note = "\nNote: The original query contained inappropriate language but the intent is valid."

    prompt = (
        "You are GobbsGPT, an AI CXO copilot for a quick-commerce brand.\n"
        "Synthesise the analysis below into a clear, executive-level response.\n\n"
        f"Original question:   {state['user_query']}\n"
        f"Analysis source:     {state.get('delegated_to', 'Multi-agent')}\n"
        f"Analysis results:    {json.dumps(sub_result, indent=2, default=str)}\n"
        f"{extra_context}{moderation_note}\n\n"
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

def route_after_guardrail(state: SupervisorState) -> str:
    if state.get("guardrail_action") == "BLOCK":
        return "synthesize"
    return "classify"


def route_to_agent(state: SupervisorState) -> str:
    classification = state.get("query_classification", "ANALYTICS")

    if classification in MULTI_AGENT_COMBOS:
        return "multi_agent"

    single_routing = {
        "ANALYTICS":    "analytics_agent",
        "ADS":          "ad_agent",
        "INVENTORY":    "inventory_agent",
        "MARKET_INTEL": "market_intel_agent",
    }
    return single_routing.get(classification, "analytics_agent")


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

def build_gobbs_gpt_supervisor() -> StateGraph:
    graph = StateGraph(SupervisorState)

    # Nodes
    graph.add_node("guardrail",          check_content_guardrail)
    graph.add_node("classify",           classify_query)
    graph.add_node("validate_routing",   validate_classification)
    graph.add_node("analytics_agent",    run_analytics_agent)
    graph.add_node("ad_agent",           run_ad_agent)
    graph.add_node("inventory_agent",    run_inventory_agent)
    graph.add_node("market_intel_agent", run_market_intel_agent)
    graph.add_node("multi_agent",        run_multi_agents)
    graph.add_node("synthesize",         synthesize_response)

    # Edges: START → guardrail
    graph.add_edge(START, "guardrail")

    # Guardrail → classify (if ALLOW/SANITIZE/FLAG) or synthesize (if BLOCKED)
    graph.add_conditional_edges(
        "guardrail",
        route_after_guardrail,
        {
            "classify": "classify",
            "synthesize": "synthesize",
        },
    )

    # Classify → validate routing
    graph.add_edge("classify", "validate_routing")

    # Validate routing → route to agent
    graph.add_conditional_edges(
        "validate_routing",
        route_to_agent,
        {
            "analytics_agent":    "analytics_agent",
            "ad_agent":           "ad_agent",
            "inventory_agent":    "inventory_agent",
            "market_intel_agent": "market_intel_agent",
            "multi_agent":        "multi_agent",
        },
    )

    # All sub-agents and multi-agent converge to synthesis
    for node in ("analytics_agent", "ad_agent", "inventory_agent",
                 "market_intel_agent", "multi_agent"):
        graph.add_edge(node, "synthesize")

    graph.add_edge("synthesize", END)

    return graph.compile()
