"""
Gobbs Flow — Inventory Agent
==============================
Tracks the availability lifecycle from company depot to dark store and generates
AI-smart purchase order recommendations.

Pipeline:
  check_inventory ──(fill_rate ≥ 85%)──► END
                  └─(fill_rate < 85%)──► analyze_demand → generate_alerts → recommend_po → END

Error variants:
  cascade_failure — generate_alerts raises ExternalAPIError(500),
                    recommend_purchase_orders raises ValueError
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

class InventoryState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    brand: str
    query_type: Optional[str]          # "stock_check" | "po_planning" | "demand_forecast"
    inventory_snapshot: Optional[dict]
    demand_signals: Optional[dict]
    stockout_alerts: Optional[list]
    po_recommendations: Optional[list]
    forecast: Optional[dict]
    error_variant: Optional[str]


# ---------------------------------------------------------------------------
# Nodes (Original — happy path)
# ---------------------------------------------------------------------------

@neatlogs.span(kind="TOOL", name="inventory_snapshot",
               tool_name="gobbs_flow_inventory_api")
def check_inventory(state: InventoryState) -> dict:
    """
    Pulls real-time inventory data across all dark stores.
    [DUMMY API] — simulates Gobbs Flow's depot-to-dark-store tracking system.
    """
    snapshot = {
        "total_skus_tracked": 150,
        "overall_fill_rate": 82.3,   # below 85 → triggers action pipeline
        "critical_stockouts": [
            {
                "sku": "SKU-001", "name": "Protein Bar 60g",
                "oos_stores": 35, "total_stores": 100,
                "cities_affected": ["Mumbai", "Pune"],
                "revenue_at_risk_daily": 85_000,
            },
            {
                "sku": "SKU-015", "name": "Oat Milk 1L",
                "oos_stores": 28, "total_stores": 100,
                "cities_affected": ["Delhi", "Noida"],
                "revenue_at_risk_daily": 42_000,
            },
            {
                "sku": "SKU-033", "name": "Keto Bar 50g",
                "oos_stores": 15, "total_stores": 100,
                "cities_affected": ["Bangalore"],
                "revenue_at_risk_daily": 28_000,
            },
        ],
        "depot_stock": {
            "SKU-001": {"depot_qty": 5_000, "transit_qty": 2_000, "eta_days": 2},
            "SKU-015": {"depot_qty":   200, "transit_qty":     0, "eta_days": None},
            "SKU-033": {"depot_qty": 3_500, "transit_qty":   500, "eta_days": 1},
        },
        "platform_breakdown": {
            "blinkit":   {"fill_rate": 85.0, "oos_skus": 12},
            "zepto":     {"fill_rate": 78.5, "oos_skus": 18},
            "instamart": {"fill_rate": 83.4, "oos_skus": 15},
        },
        "velocity_data": {
            "SKU-001": {"daily_units_sold": 820, "trend": "increasing"},
            "SKU-015": {"daily_units_sold": 340, "trend": "stable"},
            "SKU-033": {"daily_units_sold": 210, "trend": "increasing"},
        },
    }

    return {
        "inventory_snapshot": snapshot,
        "messages": [AIMessage(content=f"[Gobbs Flow API] Snapshot loaded. Fill rate: {snapshot['overall_fill_rate']}%")],
    }


@neatlogs.span(kind="AGENT", name="demand_signal_analysis",
               role="Demand Forecaster", goal="Analyse cross-platform demand patterns")
def analyze_demand(state: InventoryState) -> dict:
    """
    Cross-platform demand signal analysis for short-term forecasting (48 h–7 d).
    """
    prompt = (
        "You are a demand forecasting engine for a quick-commerce analytics platform.\n\n"
        f"Inventory data: {json.dumps(state.get('inventory_snapshot', {}))}\n\n"
        "Analyse patterns and return JSON with:\n"
        "  demand_trend: dict of sku → (up/down/stable)\n"
        "  velocity_by_platform: dict of platform → avg_daily_units\n"
        "  city_demand_rank: list of cities sorted by demand urgency\n"
        "  risk_skus: list of SKUs likely to stock out in 48 h\n"
        "  seasonal_note: string (any upcoming events/promotions affecting demand)\n"
        "Return ONLY valid JSON."
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    try:
        signals = json.loads(response.content)
    except json.JSONDecodeError:
        signals = {"status": "demand_analysis_complete"}

    return {"demand_signals": signals, "messages": [response]}


@neatlogs.span(kind="TOOL", name="stockout_alert_generator",
               tool_name="alert_service_api")
def generate_alerts(state: InventoryState) -> dict:
    """
    Derives severity-ranked stockout alerts from inventory snapshot.
    [DUMMY LOGIC] — in production alerts are pushed to Slack / PagerDuty.
    """
    snapshot = state.get("inventory_snapshot", {})
    alerts = []

    for item in snapshot.get("critical_stockouts", []):
        oos_pct = (item["oos_stores"] / item["total_stores"]) * 100
        depot = snapshot.get("depot_stock", {}).get(item["sku"], {})

        if oos_pct > 30 or depot.get("depot_qty", 0) < 500:
            severity = "CRITICAL"
        else:
            severity = "WARNING"

        alerts.append({
            "sku": item["sku"],
            "name": item["name"],
            "severity": severity,
            "oos_percentage": round(oos_pct, 1),
            "cities_affected": item["cities_affected"],
            "depot_stock_remaining": depot.get("depot_qty", 0),
            "transit_qty": depot.get("transit_qty", 0),
            "eta_days": depot.get("eta_days"),
            "revenue_at_risk_daily": item.get("revenue_at_risk_daily", 0),
        })

    # Sort CRITICAL first
    alerts.sort(key=lambda a: (0 if a["severity"] == "CRITICAL" else 1))

    return {
        "stockout_alerts": alerts,
        "messages": [AIMessage(content=f"[Alert Service] {len(alerts)} alerts generated. "
                                        f"Critical: {sum(1 for a in alerts if a['severity'] == 'CRITICAL')}")],
    }


@neatlogs.span(kind="AGENT", name="po_recommendation_engine",
               role="Supply Planner", goal="Generate AI-smart purchase order recommendations")
def recommend_purchase_orders(state: InventoryState) -> dict:
    """
    AI-smart PO planning based on demand velocity, depot levels, and stockout urgency.
    """
    prompt = (
        "You are a purchase-order planning AI for a quick-commerce consumer brand.\n\n"
        f"Inventory: {json.dumps(state.get('inventory_snapshot', {}))}\n"
        f"Demand signals: {json.dumps(state.get('demand_signals', {}))}\n"
        f"Stockout alerts: {json.dumps(state.get('stockout_alerts', []))}\n\n"
        "Generate PO recommendations. For each critical SKU:\n"
        "  1. Calculate recommended order quantity (consider lead time, transit, demand velocity)\n"
        "  2. Factor in 14-day cover target\n"
        "  3. Prioritise by revenue impact × stockout severity\n"
        "  4. Suggest warehouse-level allocation (city → dark-store mapping)\n\n"
        "Return JSON array. Each item: sku, name, recommended_qty, priority (1–5), "
        "reason, suggested_delivery_date, estimated_revenue_recovery_daily, "
        "suggested_warehouse_split (dict of city → pct).\n"
        "Return ONLY valid JSON."
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    try:
        recommendations = json.loads(response.content)
    except json.JSONDecodeError:
        recommendations = []

    return {"po_recommendations": recommendations, "messages": [response]}


# ---------------------------------------------------------------------------
# Error Variant: Cascading Tool Failures
# ---------------------------------------------------------------------------

@neatlogs.span(kind="TOOL", name="stockout_alert_generator",
               tool_name="alert_service_api")
def generate_alerts_api_down(state: InventoryState) -> dict:
    """
    Simulates alert service API returning 500 Internal Server Error.
    The alert service depends on an upstream notification queue (SQS/Kafka)
    that is experiencing issues.
    """
    raise ExternalAPIError(
        status_code=500,
        service="alert_service_api",
        message=(
            "Internal Server Error: Alert service failed to process request. "
            "Upstream dependency 'notification-queue' returned: "
            "KafkaProducerError — broker not available (leader election in progress). "
            "Circuit breaker OPEN — 15/20 requests failed in last 60 seconds."
        ),
    )


@neatlogs.span(kind="AGENT", name="po_recommendation_engine",
               role="Supply Planner", goal="Generate PO recommendations — missing alerts")
def recommend_purchase_orders_missing_data(state: InventoryState) -> dict:
    """
    PO recommendation that fails because required alerts data is missing
    (cascading from the upstream alert service failure).
    """
    alerts = state.get("stockout_alerts")
    if not alerts:
        raise ValueError(
            "Cannot generate purchase order recommendations: "
            "stockout_alerts data is missing or empty. "
            "The PO recommendation engine requires severity-ranked alerts "
            "to prioritise orders. Upstream dependency: alert_service_api."
        )
    return recommend_purchase_orders(state)


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def route_inventory_query(state: InventoryState) -> str:
    """Trigger the full action pipeline only when fill rate drops below SLA threshold."""
    snapshot = state.get("inventory_snapshot", {})
    fill_rate = snapshot.get("overall_fill_rate", 100)
    return "needs_action" if fill_rate < 85 else "report_only"


# ---------------------------------------------------------------------------
# Graph Factory
# ---------------------------------------------------------------------------

def build_inventory_agent(error_variant: str = None) -> StateGraph:
    """
    Build the inventory agent graph. Pass error_variant to get
    error-injecting variants for demo scenarios.
    """
    graph = StateGraph(InventoryState)

    if error_variant == "cascade_failure":
        # Scenario 10: Cascading tool failures
        graph.add_node("check_inventory",  check_inventory)
        graph.add_node("analyze_demand",   analyze_demand)
        graph.add_node("generate_alerts",  generate_alerts_api_down)
        graph.add_node("recommend_po",     recommend_purchase_orders_missing_data)

        graph.add_edge(START, "check_inventory")
        graph.add_conditional_edges(
            "check_inventory",
            route_inventory_query,
            {
                "needs_action": "analyze_demand",
                "report_only":  END,
            },
        )
        graph.add_edge("analyze_demand",  "generate_alerts")
        graph.add_edge("generate_alerts", "recommend_po")
        graph.add_edge("recommend_po",    END)

    else:
        # Default: original happy-path graph
        graph.add_node("check_inventory",  check_inventory)
        graph.add_node("analyze_demand",   analyze_demand)
        graph.add_node("generate_alerts",  generate_alerts)
        graph.add_node("recommend_po",     recommend_purchase_orders)

        graph.add_edge(START, "check_inventory")
        graph.add_conditional_edges(
            "check_inventory",
            route_inventory_query,
            {
                "needs_action": "analyze_demand",
                "report_only":  END,
            },
        )
        graph.add_edge("analyze_demand",  "generate_alerts")
        graph.add_edge("generate_alerts", "recommend_po")
        graph.add_edge("recommend_po",    END)

    return graph.compile()
