# GobbleCube AI Agent Architecture: Reverse Engineering & LangGraph Implementation Guide

## Executive Summary

GobbleCube is an AI-powered "agentic operational layer" for consumer brands operating in e-commerce and quick commerce. It processes billions of hyperlocal data points across pricing, availability, visibility, and demand to deliver real-time, actionable insights. This document reverse-engineers the probable agent architecture behind GobbleCube's product suite—Gobbs Edge, Gobbs Boost, Gobbs Flow, Gobbs Discover, and GobbsGPT—and provides a detailed LangGraph implementation blueprint for building representative dummy agents. These agents can serve as a compelling demo for Neatlogs' observability platform, showcasing multi-agent tracing, state management, and LLM call instrumentation.[^1][^2]

***

## GobbleCube Product Suite Overview

GobbleCube's platform consists of five interconnected products, each likely powered by specialized AI agents:[^3]

| Product | Function | Agent Type |
|---------|----------|------------|
| **Gobbs Edge** | Real-time analytics engine — cross-platform insights on pricing, visibility, availability, share of search[^3] | Data Aggregation + Analytics Agent |
| **Gobbs Boost** | AI-powered ad automation — goal-based campaigns that auto-adapt to stock, competition, and performance in real-time[^3] | Ad Optimization Agent |
| **Gobbs Flow** | Supply chain nerve center — tracks availability lifecycle from depot to dark store, AI-smart PO planning[^3] | Inventory/Supply Chain Agent |
| **Gobbs Discover** | Strategic intelligence — spots micro-category growth trends, white space, pricing gaps, NPD insights[^3] | Market Intelligence Agent |
| **GobbsGPT** | AI CXO copilot — answers what/why/what-next, diagnoses campaign underperformance, suggests next steps[^4] | Supervisor/Orchestrator Agent |

***

## Reverse-Engineered Agent Architecture

### How GobbleCube Likely Works Under the Hood

Based on GobbleCube's public technical blogs and product descriptions, the following architectural patterns are evident:

**NLQ-to-SQL Pipeline:** GobbleCube is an "LLM-based guided analytics platform" that converts natural language questions into SQL queries. They evaluated multiple LLMs (GPT-4, GPT-3.5-turbo, Google Codey) and settled on a zero-shot prompting approach achieving 80% accuracy on the Spider benchmark. The system performs intent recognition, entity recognition, query logic comprehension, SQL generation, and syntax/semantic validation.[^5][^6]

**Decision Tree Frameworks:** Rather than relying on raw NLQ-to-SQL, GobbleCube productizes business problem-solving mental frameworks as decision trees. When a user asks "Why are orders dropping?", the LLM identifies the relevant framework, navigates the tree, and surfaces root causes automatically.[^5]

**Custom Analytics Engine (Antman):** GobbleCube replaced Cube.js with a purpose-built analytics engine called Antman, written in Go, optimized for three jobs: application-facing analytics with strict latency SLAs, multi-destination query execution with intelligent routing, and context-aware data access for AI pipelines. The backend uses PostgreSQL for transactional state and ClickHouse for high-volume time-series data.[^7]

**Proprietary SLM Models:** GobbsGPT combines billions of data points with proprietary Small Language Models (SLMs) to turn raw data into actionable strategies. It goes beyond answering "what" to uncovering the deeper "why" and shaping the "what next".[^4]

**Agentic Ad Automation:** Gobbs Boost runs goal-based campaigns that auto-adapt to stock levels, competition, and performance using digital shelf-powered rulesets. This is described as an "agentic performance marketing tool" built from scratch.[^8][^3]

### Probable Agent Graph

The overall system likely follows a **supervisor pattern** where GobbsGPT acts as the central orchestrator, routing user queries to specialized sub-agents:

```
User Query → GobbsGPT (Supervisor)
                ├── Analytics Agent (Gobbs Edge)
                │      ├── NLQ-to-SQL Sub-agent
                │      ├── Root Cause Analysis Sub-agent
                │      └── Anomaly Detection Sub-agent
                ├── Ad Automation Agent (Gobbs Boost)
                │      ├── Budget Allocation Sub-agent
                │      ├── Keyword Bidding Sub-agent
                │      └── Campaign Performance Sub-agent
                ├── Inventory Agent (Gobbs Flow)
                │      ├── Stock Monitor Sub-agent
                │      ├── PO Planning Sub-agent
                │      └── Demand Forecast Sub-agent
                └── Market Intelligence Agent (Gobbs Discover)
                       ├── Trend Detection Sub-agent
                       ├── White Space Analysis Sub-agent
                       └── Competitive Intelligence Sub-agent
```

***

## LangGraph Implementation Blueprint

### Architecture Overview

The implementation uses LangGraph's `StateGraph` with a supervisor pattern. A central GobbsGPT supervisor agent delegates to four specialized sub-agents, each with their own tools and state management. This creates rich, multi-layered traces ideal for demonstrating observability.[^9]

### Prerequisites and Setup

```python
# requirements.txt
langgraph>=0.2.0
langchain-openai>=0.1.0
langchain-core>=0.2.0
langchain-community>=0.2.0
pydantic>=2.0
```

```python
import os
os.environ["OPENAI_API_KEY"] = "your-key-here"

from typing import TypedDict, List, Optional, Annotated, Literal
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
import operator
import json
```

***

### Agent 1: Analytics Agent (Gobbs Edge)

This agent handles the NLQ-to-SQL pipeline and root cause analysis—the core of GobbleCube's analytics layer.[^6][^5]

#### State Definition

```python
class AnalyticsState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    user_query: str
    intent: Optional[str]           # e.g., "revenue_analysis", "share_of_search"
    entities: Optional[dict]        # extracted entities (brand, platform, city, etc.)
    generated_sql: Optional[str]
    query_results: Optional[dict]
    root_cause: Optional[str]
    framework_used: Optional[str]   # decision tree framework identifier
    confidence_score: Optional[float]
```

#### Node Implementations

```python
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# --- Node 1: Intent & Entity Recognition ---
def recognize_intent(state: AnalyticsState) -> dict:
    """
    Identifies the business intent and extracts entities from the user query.
    Mirrors GobbleCube's first step in NLQ-to-SQL pipeline.
    """
    prompt = f"""You are an e-commerce analytics intent classifier for a quick-commerce 
    analytics platform. Analyze this query and extract:
    
    1. Intent: One of [revenue_analysis, share_of_search, pricing_analysis, 
       availability_check, campaign_performance, root_cause_analysis]
    2. Entities: Extract brand, platform (blinkit/zepto/instamart), city, 
       category, time_period, metric
    
    Query: {state['user_query']}
    
    Return JSON with keys: intent, entities (dict with extracted fields)"""
    
    response = llm.invoke([HumanMessage(content=prompt)])
    try:
        parsed = json.loads(response.content)
    except json.JSONDecodeError:
        parsed = {"intent": "revenue_analysis", "entities": {}}
    
    return {
        "intent": parsed.get("intent", "revenue_analysis"),
        "entities": parsed.get("entities", {}),
        "messages": [response]
    }


# --- Node 2: SQL Generation (Zero-Shot NLQ-to-SQL) ---
def generate_sql(state: AnalyticsState) -> dict:
    """
    Converts the recognized intent + entities into a SQL query.
    Uses the zero-shot approach that GobbleCube found most effective.
    """
    schema_context = """
    Tables:
    - orders (order_id, brand, platform, city, dark_store_id, sku_id, 
              quantity, revenue, order_date, status)
    - products (sku_id, brand, category, subcategory, mrp, selling_price)
    - search_rankings (sku_id, platform, keyword, rank_position, 
                       search_date, city)
    - inventory (sku_id, dark_store_id, city, stock_level, 
                 last_updated, reorder_point)
    - campaigns (campaign_id, brand, platform, keyword, bid_amount, 
                 impressions, clicks, spend, roas, campaign_date)
    - pricing_history (sku_id, platform, price, discount_pct, 
                       recorded_at, city)
    
    Database: ClickHouse (use ClickHouse SQL syntax)
    """
    
    prompt = f"""You are a SQL expert for an e-commerce analytics platform.
    Generate a ClickHouse-compatible SQL query for this request.
    
    {schema_context}
    
    Intent: {state['intent']}
    Entities: {json.dumps(state['entities'])}
    Original Query: {state['user_query']}
    
    Return ONLY the SQL query, no explanation."""
    
    response = llm.invoke([HumanMessage(content=prompt)])
    
    return {
        "generated_sql": response.content.strip(),
        "messages": [response]
    }


# --- Node 3: Execute Query (Simulated) ---
def execute_query(state: AnalyticsState) -> dict:
    """
    Simulates query execution against the analytics engine.
    In production, this would hit GobbleCube's Antman engine.
    """
    # Simulated results based on intent
    simulated_results = {
        "revenue_analysis": {
            "total_revenue": 2450000,
            "revenue_change_pct": -12.3,
            "top_declining_cities": ["Mumbai", "Delhi", "Bangalore"],
            "top_declining_skus": ["SKU-1234", "SKU-5678"]
        },
        "share_of_search": {
            "brand_sov": 23.5,
            "competitor_sov": 31.2,
            "top_keywords": ["protein bar", "healthy snacks"],
            "rank_changes": [{"keyword": "protein bar", "from": 2, "to": 5}]
        },
        "availability_check": {
            "overall_availability": 78.5,
            "oos_dark_stores": 42,
            "critical_skus_oos": ["SKU-1234", "SKU-9012"],
            "cities_below_threshold": ["Pune", "Hyderabad"]
        }
    }
    
    results = simulated_results.get(
        state["intent"], 
        {"status": "no_data", "message": "No matching data found"}
    )
    
    return {
        "query_results": results,
        "messages": [AIMessage(content=f"Query executed. Results: {json.dumps(results)}")]
    }


# --- Node 4: Root Cause Analysis ---
def analyze_root_cause(state: AnalyticsState) -> dict:
    """
    Navigates decision tree frameworks to identify root causes.
    This mirrors GobbleCube's productized problem-solving frameworks.
    """
    prompt = f"""You are a root cause analysis engine for an e-commerce analytics platform.
    You use structured decision tree frameworks to diagnose business problems.
    
    Framework Selection Rules:
    - Revenue drop → Check: availability → pricing → visibility → competition → seasonality
    - SOV decline → Check: keyword ranking → ad spend → new competitors → content quality
    - Availability issues → Check: supply chain → demand spike → warehouse allocation
    
    Query: {state['user_query']}
    Intent: {state['intent']}
    Data: {json.dumps(state['query_results'])}
    
    Walk through the relevant decision tree framework. Identify:
    1. The framework used
    2. The primary root cause
    3. Contributing factors  
    4. Confidence score (0-1)
    5. Recommended actions
    
    Return structured JSON."""
    
    response = llm.invoke([HumanMessage(content=prompt)])
    
    try:
        parsed = json.loads(response.content)
    except json.JSONDecodeError:
        parsed = {
            "framework": "revenue_diagnostic",
            "root_cause": "Analysis complete",
            "confidence": 0.75
        }
    
    return {
        "root_cause": response.content,
        "framework_used": parsed.get("framework", "general_diagnostic"),
        "confidence_score": parsed.get("confidence", 0.7),
        "messages": [response]
    }


# --- Routing Logic ---
def should_analyze_root_cause(state: AnalyticsState) -> str:
    """Route to root cause analysis only if data suggests anomalies."""
    results = state.get("query_results", {})
    if isinstance(results, dict):
        change = results.get("revenue_change_pct", 0)
        availability = results.get("overall_availability", 100)
        if change < -5 or availability < 85:
            return "analyze"
    return "report"
```

#### Graph Assembly

```python
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
        {
            "analyze": "analyze_root_cause",
            "report": END
        }
    )
    graph.add_edge("analyze_root_cause", END)
    
    return graph.compile()
```

***

### Agent 2: Ad Automation Agent (Gobbs Boost)

This agent mirrors GobbleCube's agentic performance marketing tool that runs goal-based campaigns auto-adapting to stock, competition, and performance.[^3][^8]

#### State Definition

```python
class AdAutomationState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    brand: str
    platform: str
    campaign_goal: Optional[str]      # "maximize_roas", "increase_sov", "launch_push"
    budget: Optional[float]
    current_performance: Optional[dict]
    stock_context: Optional[dict]      # inventory data from Gobbs Flow
    competitive_context: Optional[dict] # competitive data from Gobbs Edge
    bid_recommendations: Optional[list]
    budget_allocation: Optional[dict]
    execution_plan: Optional[dict]
```

#### Node Implementations

```python
# --- Node 1: Gather Campaign Context ---
def gather_campaign_context(state: AdAutomationState) -> dict:
    """
    Pulls current campaign performance, stock levels, and competitive data.
    Gobbs Boost integrates digital shelf data with ad decisions.
    """
    # Simulated cross-system data pull
    performance = {
        "active_campaigns": 12,
        "total_spend_today": 45000,
        "avg_roas": 3.2,
        "top_keywords": [
            {"keyword": "protein bar", "spend": 8000, "roas": 4.1, "rank": 2},
            {"keyword": "healthy snacks", "spend": 6000, "roas": 2.8, "rank": 5},
            {"keyword": "energy bar", "spend": 4000, "roas": 1.9, "rank": 8}
        ]
    }
    
    stock = {
        "sku_availability": {
            "SKU-001": {"available_stores": 85, "total_stores": 100},
            "SKU-002": {"available_stores": 42, "total_stores": 100},
            "SKU-003": {"available_stores": 95, "total_stores": 100}
        }
    }
    
    competitive = {
        "competitor_sov": {"Competitor A": 28.5, "Competitor B": 22.1},
        "brand_sov": 23.5,
        "emerging_keywords": ["keto bar", "plant protein"]
    }
    
    return {
        "current_performance": performance,
        "stock_context": stock,
        "competitive_context": competitive,
        "messages": [AIMessage(content="Campaign context gathered from Edge and Flow.")]
    }


# --- Node 2: Generate Bid Recommendations ---
def generate_bid_recommendations(state: AdAutomationState) -> dict:
    """
    Uses LLM to generate intelligent bid adjustments based on 
    stock + competition + performance signals.
    """
    prompt = f"""You are an AI ad optimization engine for quick commerce platforms.
    Generate bid recommendations based on these rules:
    
    RULES (digital shelf-powered rulesets):
    1. If a SKU has <50% store availability, PAUSE or REDUCE bids for its keywords
    2. If brand ranks #1-2 for a keyword, REDUCE bid (already dominant)
    3. If competitor SOV is rising on a keyword, INCREASE bid defensively
    4. If ROAS < 2.0, REDUCE bid or PAUSE keyword
    5. If ROAS > 4.0 and availability > 80%, INCREASE bid to capture more
    6. Allocate freed budget to emerging keyword opportunities
    
    Current Performance: {json.dumps(state['current_performance'])}
    Stock Context: {json.dumps(state['stock_context'])}
    Competitive Context: {json.dumps(state['competitive_context'])}
    Campaign Goal: {state.get('campaign_goal', 'maximize_roas')}
    Budget: {state.get('budget', 50000)}
    
    Return JSON array of bid recommendations with: keyword, action, 
    reason, new_bid_amount, expected_impact."""
    
    response = llm.invoke([HumanMessage(content=prompt)])
    
    try:
        recommendations = json.loads(response.content)
    except json.JSONDecodeError:
        recommendations = []
    
    return {
        "bid_recommendations": recommendations,
        "messages": [response]
    }


# --- Node 3: Allocate Budget ---
def allocate_budget(state: AdAutomationState) -> dict:
    """Optimally distributes budget across keywords and campaigns."""
    prompt = f"""You are a budget allocation optimizer for e-commerce ads.
    
    Given these bid recommendations: {json.dumps(state.get('bid_recommendations', []))}
    Total budget: {state.get('budget', 50000)}
    Goal: {state.get('campaign_goal', 'maximize_roas')}
    
    Create an optimal budget allocation plan. Consider:
    - Time-of-day patterns (morning/evening peaks for quick commerce)
    - City-level stock availability
    - Keyword-level ROAS expectations
    
    Return JSON with: keyword_allocations (list), time_split (dict), 
    city_priorities (list), expected_total_roas."""
    
    response = llm.invoke([HumanMessage(content=prompt)])
    
    try:
        allocation = json.loads(response.content)
    except json.JSONDecodeError:
        allocation = {"status": "allocation_generated"}
    
    return {
        "budget_allocation": allocation,
        "messages": [response]
    }


# --- Node 4: Generate Execution Plan ---
def generate_execution_plan(state: AdAutomationState) -> dict:
    """Compiles the final execution plan for campaign changes."""
    plan = {
        "bid_changes": state.get("bid_recommendations", []),
        "budget_allocation": state.get("budget_allocation", {}),
        "auto_rules": [
            "Pause keywords when SKU availability drops below 40%",
            "Increase bids by 15% when competitor SOV rises above 30%",
            "Shift budget to evening slots on weekends"
        ],
        "monitoring_triggers": [
            "Alert if ROAS drops below 2.0 for any keyword",
            "Alert if daily spend exceeds 120% of planned budget",
            "Alert if any campaign keyword loses top-3 ranking"
        ]
    }
    
    return {
        "execution_plan": plan,
        "messages": [AIMessage(content=f"Execution plan ready: {json.dumps(plan)}")]
    }
```

#### Graph Assembly

```python
def build_ad_automation_agent() -> StateGraph:
    graph = StateGraph(AdAutomationState)
    
    graph.add_node("gather_context", gather_campaign_context)
    graph.add_node("generate_bids", generate_bid_recommendations)
    graph.add_node("allocate_budget", allocate_budget)
    graph.add_node("execution_plan", generate_execution_plan)
    
    graph.add_edge(START, "gather_context")
    graph.add_edge("gather_context", "generate_bids")
    graph.add_edge("generate_bids", "allocate_budget")
    graph.add_edge("allocate_budget", "execution_plan")
    graph.add_edge("execution_plan", END)
    
    return graph.compile()
```

***

### Agent 3: Inventory Agent (Gobbs Flow)

This agent tracks the availability lifecycle from company depot to dark store and generates AI-smart purchase order recommendations.[^3]

#### State Definition

```python
class InventoryState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    brand: str
    query_type: Optional[str]  # "stock_check", "po_planning", "demand_forecast"
    inventory_snapshot: Optional[dict]
    demand_signals: Optional[dict]
    stockout_alerts: Optional[list]
    po_recommendations: Optional[list]
    forecast: Optional[dict]
```

#### Node Implementations

```python
# --- Node 1: Inventory Snapshot ---
def check_inventory(state: InventoryState) -> dict:
    """
    Pulls real-time inventory data across dark stores.
    Simulates Gobbs Flow's depot-to-dark-store tracking.
    """
    snapshot = {
        "total_skus_tracked": 150,
        "overall_fill_rate": 82.3,
        "critical_stockouts": [
            {"sku": "SKU-001", "name": "Protein Bar 60g", "oos_stores": 35, 
             "total_stores": 100, "cities_affected": ["Mumbai", "Pune"]},
            {"sku": "SKU-015", "name": "Oat Milk 1L", "oos_stores": 28, 
             "total_stores": 100, "cities_affected": ["Delhi", "Noida"]}
        ],
        "depot_stock": {
            "SKU-001": {"depot_qty": 5000, "transit_qty": 2000, "eta_days": 2},
            "SKU-015": {"depot_qty": 200, "transit_qty": 0, "eta_days": None}
        },
        "platform_breakdown": {
            "blinkit": {"fill_rate": 85.0, "oos_skus": 12},
            "zepto": {"fill_rate": 78.5, "oos_skus": 18},
            "instamart": {"fill_rate": 83.4, "oos_skus": 15}
        }
    }
    
    return {
        "inventory_snapshot": snapshot,
        "messages": [AIMessage(content="Inventory snapshot loaded.")]
    }


# --- Node 2: Analyze Demand Signals ---
def analyze_demand(state: InventoryState) -> dict:
    """Cross-platform demand signal analysis for forecasting."""
    prompt = f"""You are a demand forecasting engine for a quick-commerce analytics platform.
    
    Analyze these inventory patterns and generate demand signals:
    
    Inventory Data: {json.dumps(state.get('inventory_snapshot', {}))}
    
    Consider:
    - Seasonal trends (current month patterns)
    - Platform-specific demand velocity
    - City-level consumption patterns
    - Stockout-induced demand shift to competitors
    
    Return JSON with: demand_trend (up/down/stable per SKU), 
    velocity_by_platform, city_demand_rank, risk_skus (likely to stockout in 48h)."""
    
    response = llm.invoke([HumanMessage(content=prompt)])
    
    try:
        signals = json.loads(response.content)
    except json.JSONDecodeError:
        signals = {"status": "demand_analysis_complete"}
    
    return {
        "demand_signals": signals,
        "messages": [response]
    }


# --- Node 3: Generate Stockout Alerts ---
def generate_alerts(state: InventoryState) -> dict:
    """Identifies critical stockout situations requiring immediate action."""
    snapshot = state.get("inventory_snapshot", {})
    alerts = []
    
    for item in snapshot.get("critical_stockouts", []):
        oos_pct = (item["oos_stores"] / item["total_stores"]) * 100
        depot = snapshot.get("depot_stock", {}).get(item["sku"], {})
        
        severity = "CRITICAL" if oos_pct > 30 else "WARNING"
        if depot.get("depot_qty", 0) < 500:
            severity = "CRITICAL"
        
        alerts.append({
            "sku": item["sku"],
            "name": item["name"],
            "severity": severity,
            "oos_percentage": oos_pct,
            "cities_affected": item["cities_affected"],
            "depot_stock_remaining": depot.get("depot_qty", 0),
            "transit_qty": depot.get("transit_qty", 0),
            "eta_days": depot.get("eta_days")
        })
    
    return {
        "stockout_alerts": alerts,
        "messages": [AIMessage(content=f"Generated {len(alerts)} stockout alerts.")]
    }


# --- Node 4: PO Recommendations ---
def recommend_purchase_orders(state: InventoryState) -> dict:
    """AI-smart PO planning based on cross-platform demand."""
    prompt = f"""You are a purchase order planning AI for a quick-commerce brand.
    
    Inventory: {json.dumps(state.get('inventory_snapshot', {}))}
    Demand Signals: {json.dumps(state.get('demand_signals', {}))}
    Stockout Alerts: {json.dumps(state.get('stockout_alerts', []))}
    
    Generate purchase order recommendations:
    1. For each critical SKU, calculate recommended order quantity
    2. Factor in lead times, current transit, and demand velocity
    3. Prioritize by revenue impact and stockout severity
    4. Suggest warehouse-level allocation
    
    Return JSON array with: sku, recommended_qty, priority (1-5), 
    reason, suggested_delivery_date, estimated_revenue_recovery."""
    
    response = llm.invoke([HumanMessage(content=prompt)])
    
    try:
        recommendations = json.loads(response.content)
    except json.JSONDecodeError:
        recommendations = []
    
    return {
        "po_recommendations": recommendations,
        "messages": [response]
    }


# --- Routing ---
def route_inventory_query(state: InventoryState) -> str:
    snapshot = state.get("inventory_snapshot", {})
    fill_rate = snapshot.get("overall_fill_rate", 100)
    if fill_rate < 85:
        return "needs_action"
    return "report_only"
```

#### Graph Assembly

```python
def build_inventory_agent() -> StateGraph:
    graph = StateGraph(InventoryState)
    
    graph.add_node("check_inventory", check_inventory)
    graph.add_node("analyze_demand", analyze_demand)
    graph.add_node("generate_alerts", generate_alerts)
    graph.add_node("recommend_po", recommend_purchase_orders)
    
    graph.add_edge(START, "check_inventory")
    graph.add_conditional_edges(
        "check_inventory",
        route_inventory_query,
        {
            "needs_action": "analyze_demand",
            "report_only": END
        }
    )
    graph.add_edge("analyze_demand", "generate_alerts")
    graph.add_edge("generate_alerts", "recommend_po")
    graph.add_edge("recommend_po", END)
    
    return graph.compile()
```

***

### Agent 4: Market Intelligence Agent (Gobbs Discover)

This agent spots micro-category growth trends, white space opportunities, and competitive intelligence at a hyperlocal level.[^3]

#### State Definition

```python
class MarketIntelState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    brand: str
    category: str
    analysis_type: Optional[str]   # "trend", "whitespace", "competitive"
    market_data: Optional[dict]
    trends: Optional[list]
    opportunities: Optional[list]
    competitive_moves: Optional[list]
    strategic_recommendations: Optional[str]
```

#### Node Implementations

```python
# --- Node 1: Gather Market Data ---
def gather_market_data(state: MarketIntelState) -> dict:
    """Aggregates market-level data across platforms and categories."""
    data = {
        "category": state.get("category", "health_snacks"),
        "market_size_trend": {"6mo_growth": 18.5, "yoy_growth": 42.3},
        "top_brands": [
            {"brand": "Brand A", "market_share": 22.1, "trend": "stable"},
            {"brand": "Brand B", "market_share": 18.7, "trend": "growing"},
            {"brand": state.get("brand", "Our Brand"), "market_share": 15.3, "trend": "growing"}
        ],
        "subcategory_breakdown": [
            {"subcategory": "Protein Bars", "growth": 35.2, "saturation": "medium"},
            {"subcategory": "Granola", "growth": 12.1, "saturation": "high"},
            {"subcategory": "Keto Snacks", "growth": 68.5, "saturation": "low"}
        ],
        "pricing_landscape": {
            "avg_category_price": 180,
            "price_range": [99, 450],
            "gap_zones": ["150-200 (underserved)", "300-350 (no premium options)"]
        }
    }
    
    return {
        "market_data": data,
        "messages": [AIMessage(content="Market data aggregated.")]
    }


# --- Node 2: Detect Trends ---
def detect_trends(state: MarketIntelState) -> dict:
    """Identifies micro-category growth trends and emerging patterns."""
    prompt = f"""You are a market intelligence analyst for quick commerce.
    
    Analyze this market data and identify:
    1. Emerging micro-trends in the category
    2. Demand velocity changes by city tier
    3. New search keywords gaining traction
    4. Seasonal or event-driven demand patterns
    
    Market Data: {json.dumps(state.get('market_data', {}))}
    
    Return JSON array of trends with: trend_name, description, 
    growth_signal_strength (1-10), time_horizon, relevant_cities."""
    
    response = llm.invoke([HumanMessage(content=prompt)])
    
    try:
        trends = json.loads(response.content)
    except json.JSONDecodeError:
        trends = []
    
    return {"trends": trends, "messages": [response]}


# --- Node 3: White Space Analysis ---
def find_white_space(state: MarketIntelState) -> dict:
    """Identifies untapped market opportunities and pricing gaps."""
    prompt = f"""You are a strategic market analyst for consumer brands.
    
    Based on market data and trends, identify white space opportunities:
    
    Market Data: {json.dumps(state.get('market_data', {}))}
    Trends: {json.dumps(state.get('trends', []))}
    
    Find:
    1. Price points with no strong brand presence
    2. Subcategories with high growth but low competition
    3. Geographic markets with unmet demand
    4. Product format gaps (size, packaging, flavor)
    
    Return JSON array of opportunities with: opportunity_name, type, 
    market_size_estimate, competition_level, recommended_action, 
    priority_score (1-10)."""
    
    response = llm.invoke([HumanMessage(content=prompt)])
    
    try:
        opportunities = json.loads(response.content)
    except json.JSONDecodeError:
        opportunities = []
    
    return {"opportunities": opportunities, "messages": [response]}


# --- Node 4: Strategic Recommendations ---
def generate_strategy(state: MarketIntelState) -> dict:
    """Synthesizes all intelligence into actionable strategy."""
    prompt = f"""You are a growth strategy advisor for a consumer brand 
    in quick commerce.
    
    Synthesize these inputs into a strategic recommendation:
    
    Market Data: {json.dumps(state.get('market_data', {}))}
    Trends: {json.dumps(state.get('trends', []))}
    White Space: {json.dumps(state.get('opportunities', []))}
    
    Provide:
    1. Top 3 strategic priorities for next quarter
    2. Quick wins (actionable in 1-2 weeks)
    3. Medium-term bets (1-3 months)
    4. Risks to monitor
    
    Be specific and actionable."""
    
    response = llm.invoke([HumanMessage(content=prompt)])
    
    return {
        "strategic_recommendations": response.content,
        "messages": [response]
    }
```

#### Graph Assembly

```python
def build_market_intel_agent() -> StateGraph:
    graph = StateGraph(MarketIntelState)
    
    graph.add_node("gather_data", gather_market_data)
    graph.add_node("detect_trends", detect_trends)
    graph.add_node("find_whitespace", find_white_space)
    graph.add_node("generate_strategy", generate_strategy)
    
    graph.add_edge(START, "gather_data")
    graph.add_edge("gather_data", "detect_trends")
    graph.add_edge("detect_trends", "find_whitespace")
    graph.add_edge("find_whitespace", "generate_strategy")
    graph.add_edge("generate_strategy", END)
    
    return graph.compile()
```

***

### Agent 5: GobbsGPT Supervisor (Orchestrator)

This is the central supervisor agent that mirrors GobbsGPT—the AI CXO copilot that routes queries to the appropriate specialized agent. It uses LangGraph's supervisor pattern where the orchestrator decides which sub-agent to invoke based on query classification.[^4][^9]

#### State Definition

```python
class SupervisorState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    user_query: str
    query_classification: Optional[str]
    delegated_to: Optional[str]
    sub_agent_result: Optional[dict]
    final_response: Optional[str]
    follow_up_suggestions: Optional[list]
```

#### Supervisor Implementation

```python
# --- Node 1: Classify & Route ---
def classify_query(state: SupervisorState) -> dict:
    """
    GobbsGPT's first step: understand what the user is asking 
    and route to the right specialized agent.
    """
    prompt = f"""You are GobbsGPT, an AI CXO copilot for e-commerce brands.
    Classify this query into exactly one category:
    
    - ANALYTICS: Questions about revenue, sales performance, SOV, 
      pricing analysis, "why" questions about metrics
    - ADS: Questions about ad campaigns, ROAS, bidding, marketing spend, 
      campaign optimization
    - INVENTORY: Questions about stock levels, availability, stockouts, 
      purchase orders, supply chain
    - MARKET_INTEL: Questions about market trends, competition, 
      new opportunities, white space, NPD
    
    Query: {state['user_query']}
    
    Return ONLY the category name (ANALYTICS, ADS, INVENTORY, or MARKET_INTEL)."""
    
    response = llm.invoke([HumanMessage(content=prompt)])
    classification = response.content.strip().upper()
    
    valid = ["ANALYTICS", "ADS", "INVENTORY", "MARKET_INTEL"]
    if classification not in valid:
        classification = "ANALYTICS"
    
    return {
        "query_classification": classification,
        "messages": [response]
    }


def route_to_agent(state: SupervisorState) -> str:
    """Routes to the appropriate sub-agent based on classification."""
    classification = state.get("query_classification", "ANALYTICS")
    routing = {
        "ANALYTICS": "analytics_agent",
        "ADS": "ad_agent",
        "INVENTORY": "inventory_agent",
        "MARKET_INTEL": "market_intel_agent"
    }
    return routing.get(classification, "analytics_agent")


# --- Sub-Agent Wrapper Nodes ---
def run_analytics_agent(state: SupervisorState) -> dict:
    """Delegates to the Analytics (Gobbs Edge) agent."""
    agent = build_analytics_agent()
    result = agent.invoke({
        "messages": [],
        "user_query": state["user_query"],
        "intent": None, "entities": None,
        "generated_sql": None, "query_results": None,
        "root_cause": None, "framework_used": None,
        "confidence_score": None
    })
    return {
        "delegated_to": "Gobbs Edge (Analytics)",
        "sub_agent_result": {
            "intent": result.get("intent"),
            "sql": result.get("generated_sql"),
            "results": result.get("query_results"),
            "root_cause": result.get("root_cause"),
            "framework": result.get("framework_used"),
            "confidence": result.get("confidence_score")
        },
        "messages": [AIMessage(content="Analytics analysis complete.")]
    }


def run_ad_agent(state: SupervisorState) -> dict:
    """Delegates to the Ad Automation (Gobbs Boost) agent."""
    agent = build_ad_automation_agent()
    result = agent.invoke({
        "messages": [],
        "brand": "Demo Brand",
        "platform": "blinkit",
        "campaign_goal": "maximize_roas",
        "budget": 50000,
        "current_performance": None, "stock_context": None,
        "competitive_context": None, "bid_recommendations": None,
        "budget_allocation": None, "execution_plan": None
    })
    return {
        "delegated_to": "Gobbs Boost (Ads)",
        "sub_agent_result": {
            "bids": result.get("bid_recommendations"),
            "allocation": result.get("budget_allocation"),
            "plan": result.get("execution_plan")
        },
        "messages": [AIMessage(content="Ad optimization complete.")]
    }


def run_inventory_agent(state: SupervisorState) -> dict:
    """Delegates to the Inventory (Gobbs Flow) agent."""
    agent = build_inventory_agent()
    result = agent.invoke({
        "messages": [],
        "brand": "Demo Brand",
        "query_type": "stock_check",
        "inventory_snapshot": None, "demand_signals": None,
        "stockout_alerts": None, "po_recommendations": None,
        "forecast": None
    })
    return {
        "delegated_to": "Gobbs Flow (Inventory)",
        "sub_agent_result": {
            "snapshot": result.get("inventory_snapshot"),
            "alerts": result.get("stockout_alerts"),
            "po_recommendations": result.get("po_recommendations")
        },
        "messages": [AIMessage(content="Inventory analysis complete.")]
    }


def run_market_intel_agent(state: SupervisorState) -> dict:
    """Delegates to the Market Intelligence (Gobbs Discover) agent."""
    agent = build_market_intel_agent()
    result = agent.invoke({
        "messages": [],
        "brand": "Demo Brand",
        "category": "health_snacks",
        "analysis_type": "comprehensive",
        "market_data": None, "trends": None,
        "opportunities": None, "competitive_moves": None,
        "strategic_recommendations": None
    })
    return {
        "delegated_to": "Gobbs Discover (Market Intel)",
        "sub_agent_result": {
            "trends": result.get("trends"),
            "opportunities": result.get("opportunities"),
            "strategy": result.get("strategic_recommendations")
        },
        "messages": [AIMessage(content="Market intelligence complete.")]
    }


# --- Node: Synthesize Final Response ---
def synthesize_response(state: SupervisorState) -> dict:
    """
    GobbsGPT synthesizes the sub-agent results into a CXO-friendly response
    with actionable next steps and follow-up suggestions.
    """
    prompt = f"""You are GobbsGPT, an AI CXO copilot. 
    Synthesize this analysis into a clear, executive-level response.
    
    Original Question: {state['user_query']}
    Analysis Source: {state.get('delegated_to', 'Unknown')}
    Analysis Results: {json.dumps(state.get('sub_agent_result', {}))}
    
    Provide:
    1. A clear, concise answer to the question
    2. Key insights with supporting data
    3. Recommended immediate actions (prioritized)
    4. Three follow-up questions the CXO should ask next
    
    Use a professional but conversational tone. Be specific with numbers."""
    
    response = llm.invoke([HumanMessage(content=prompt)])
    
    return {
        "final_response": response.content,
        "follow_up_suggestions": [
            "What is the city-level breakdown of this impact?",
            "How does this compare to last month?",
            "What budget reallocation would you recommend?"
        ],
        "messages": [response]
    }
```

#### Supervisor Graph Assembly

```python
def build_gobbs_gpt_supervisor() -> StateGraph:
    graph = StateGraph(SupervisorState)
    
    # Nodes
    graph.add_node("classify", classify_query)
    graph.add_node("analytics_agent", run_analytics_agent)
    graph.add_node("ad_agent", run_ad_agent)
    graph.add_node("inventory_agent", run_inventory_agent)
    graph.add_node("market_intel_agent", run_market_intel_agent)
    graph.add_node("synthesize", synthesize_response)
    
    # Edges
    graph.add_edge(START, "classify")
    graph.add_conditional_edges(
        "classify",
        route_to_agent,
        {
            "analytics_agent": "analytics_agent",
            "ad_agent": "ad_agent",
            "inventory_agent": "inventory_agent",
            "market_intel_agent": "market_intel_agent"
        }
    )
    
    # All sub-agents converge to synthesis
    graph.add_edge("analytics_agent", "synthesize")
    graph.add_edge("ad_agent", "synthesize")
    graph.add_edge("inventory_agent", "synthesize")
    graph.add_edge("market_intel_agent", "synthesize")
    graph.add_edge("synthesize", END)
    
    return graph.compile()
```

***

## Observability Integration

### Instrumenting with Neatlogs

This multi-agent architecture generates rich, multi-layered traces that are ideal for demonstrating observability capabilities. Each agent invocation creates nested spans across the supervisor and sub-agent hierarchy.

#### Key Observability Points

| Trace Point | What to Capture | Why It Matters |
|-------------|-----------------|----------------|
| **Supervisor routing** | Query classification, routing decision, latency | Shows agent orchestration decisions |
| **NLQ-to-SQL generation** | Input query, generated SQL, model used, tokens | Critical for debugging query accuracy |
| **Root cause analysis** | Framework selected, decision tree path, confidence | Shows reasoning chain transparency |
| **Ad bid decisions** | Stock context, competitive signals, bid changes | Multi-signal decision audit trail |
| **Inventory alerts** | Threshold triggers, severity calculations | Automated decision monitoring |
| **LLM calls** | Prompt, response, tokens, latency, cost per call | Core cost and performance tracking |
| **Cross-agent data flow** | Data passed between Edge → Boost (stock → ads) | Shows system-level data dependencies |

#### Integration Code (Langfuse-Style Callbacks)

```python
# Example: Adding observability callbacks to the supervisor
from langfuse.callback import CallbackHandler

# Initialize observability handler
obs_handler = CallbackHandler(
    public_key="your-public-key",
    secret_key="your-secret-key",
    host="https://your-neatlogs-instance.com"
)

# Run with tracing enabled
supervisor = build_gobbs_gpt_supervisor()

result = supervisor.invoke(
    {
        "messages": [],
        "user_query": "Why did our revenue drop 15% in Mumbai last week?",
        "query_classification": None,
        "delegated_to": None,
        "sub_agent_result": None,
        "final_response": None,
        "follow_up_suggestions": None
    },
    config={"callbacks": [obs_handler]}
)
```

This produces traces showing:[^10]
- **Top-level span**: GobbsGPT Supervisor invocation
- **Child span 1**: Query classification LLM call
- **Child span 2**: Analytics Agent invocation (nested sub-graph)
  - **Grandchild spans**: Intent recognition → SQL generation → query execution → root cause analysis
- **Child span 3**: Response synthesis LLM call

***

## Demo Scenarios

### Scenario 1: Revenue Diagnostic

**Query**: "Why did our revenue drop 15% in Mumbai last week?"

**Flow**: Supervisor → Analytics Agent → Intent (revenue_analysis) → SQL Generation → Execute → Root Cause Analysis → Synthesize

**Observability value**: Shows the full decision tree traversal, SQL generation accuracy, and multi-step reasoning chain.

### Scenario 2: Ad Campaign Optimization

**Query**: "Our ROAS on Blinkit dropped below 2. What should we change?"

**Flow**: Supervisor → Ad Agent → Gather Context (pulls from Edge + Flow) → Generate Bids → Allocate Budget → Execution Plan → Synthesize

**Observability value**: Demonstrates cross-agent data dependencies (stock data influencing ad decisions) and multi-signal reasoning.

### Scenario 3: Stockout Emergency

**Query**: "Which SKUs are at risk of stocking out in the next 48 hours?"

**Flow**: Supervisor → Inventory Agent → Check Inventory → Analyze Demand → Generate Alerts → PO Recommendations → Synthesize

**Observability value**: Shows conditional routing (only triggers PO planning if fill rate is below threshold) and automated alert generation.

### Scenario 4: Market Opportunity Discovery

**Query**: "What are the fastest growing subcategories we should enter?"

**Flow**: Supervisor → Market Intel Agent → Gather Data → Detect Trends → Find White Space → Generate Strategy → Synthesize

**Observability value**: Shows strategic reasoning chain from raw market data through trend analysis to actionable recommendations.

***

## Running the Complete System

```python
def main():
    """
    Run the full GobbsGPT supervisor with all sub-agents.
    """
    supervisor = build_gobbs_gpt_supervisor()
    
    test_queries = [
        "Why did our revenue drop 15% in Mumbai last week?",
        "Optimize our ad campaigns on Blinkit for maximum ROAS",
        "Which SKUs are at risk of stocking out in the next 48 hours?",
        "What are the fastest growing subcategories we should enter?"
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"QUERY: {query}")
        print(f"{'='*60}")
        
        result = supervisor.invoke({
            "messages": [],
            "user_query": query,
            "query_classification": None,
            "delegated_to": None,
            "sub_agent_result": None,
            "final_response": None,
            "follow_up_suggestions": None
        })
        
        print(f"\nRouted to: {result.get('delegated_to')}")
        print(f"\nResponse:\n{result.get('final_response')}")
        print(f"\nFollow-ups: {result.get('follow_up_suggestions')}")


if __name__ == "__main__":
    main()
```

***

## Technical Architecture Summary

The implemented system mirrors GobbleCube's architecture with five agent components organized in a supervisor pattern:

| Component | GobbleCube Product | LangGraph Pattern | LLM Calls per Run |
|-----------|-------------------|-------------------|-------------------|
| Supervisor | GobbsGPT | Conditional routing graph | 2 (classify + synthesize) |
| Analytics Agent | Gobbs Edge | Sequential with conditional branch | 3-4 (intent + SQL + execute + root cause) |
| Ad Automation Agent | Gobbs Boost | Linear pipeline | 3 (bids + budget + plan) |
| Inventory Agent | Gobbs Flow | Conditional pipeline | 2-4 (check + optional demand/alerts/PO) |
| Market Intel Agent | Gobbs Discover | Linear pipeline | 3 (trends + whitespace + strategy) |

A single end-to-end query through the supervisor generates 5-6 LLM calls minimum, creating deep trace trees with 10+ spans—ideal for showcasing observability features like latency tracking, token usage monitoring, cost attribution, and reasoning chain inspection.

---

## References

1. [GobbleCube: The AI Growth Copilot](https://unlistedzone.com/gobblecube-the-ai-growth-copilot-fueling-global-brand-success) - GobbleCube's core product is an AI-based "growth copilot" designed for consumer brands. It helps com...

2. [Kae Capital - Lead - Strategic Partnerships - GobbleCube - IIM Jobs](https://www.iimjobs.com/j/kae-capital-lead-strategic-partnerships-gobblecube-1671675?jobPos=1) - GobbleCube is an agentic operational layer with hyperlocal intelligence that helps brands drive reve...

3. [42Signals vs. GobbleCube: Choosing Your E-commerce Data Partner](https://www.42signals.com/blog/gobblecube-vs-42signals/) - In contrast, GobbleCube is the specialist for speed and surgical precision, excelling in near real-t...

4. [Our AI engine helps brands make decisions at “Quick-Commerce ...](https://cio.economictimes.indiatimes.com/news/artificial-intelligence/our-ai-engine-helps-brands-make-decisions-at-quick-commerce-like-speed-says-satyam-krishna-gobblecube/125221986) - In the age of quick commerce with 10-minute deliveries, GobbleCube's AI engine is redefining how fas...

5. [The value of abstracting the 'Why' - GobbleCube](https://gobblecube.ai/blog/the-value-of-abstracting-the-why/) - GobbleCube is an LLM-based guided analytics platform that can help you productize your business prob...

6. [NLQ-to-SQL with LLMs: Our Journey and Learnings - GobbleCube](https://gobblecube.ai/blog/nlq-to-sql-with-llm/) - Why We Replaced Cube.js with an In-House Analytics Engine. Technology · Why We Replaced Cube.js with...

7. [Why We Replaced Cube.js with an In-House Analytics Engine](https://gobblecube.ai/blog/why-we-replaced-cubejs-with-an-in-house-analytics-engine/) - This blog walks through why we replaced Cube.js with a purpose-built in-house analytics engine calle...

8. [The Pillar Behind Our Agentic Performance Marketing Tool](https://gobblecube.ai/blog/the-pillar-behind-our-agentic-performance-marketing-tool/) - Around the same time, we were ideating our Agentic Ad Automation product. ... GobbleCube is an AI-po...

9. [langchain-ai/langgraph-supervisor-py - GitHub](https://github.com/langchain-ai/langgraph-supervisor-py) - A Python library for creating hierarchical multi-agent systems using LangGraph. Hierarchical systems...

10. [Example - Trace and Evaluate LangGraph Agents - Langfuse](https://langfuse.com/guides/cookbook/example_langgraph_agents) - In this tutorial, we will learn how to monitor the internal steps (traces) of LangGraph agents and e...

