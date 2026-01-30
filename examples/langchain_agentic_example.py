"""
Neatlogs SDK v4 - LangChain Agentic Systems Example

This example demonstrates proper decorator usage with LangChain.

Key insight: LangChain + OpenInference auto-instrumentation already creates
LLM, AGENT, and CHAIN spans. We only need decorators for:
- @workflow: Entry point (always required)
- @tool: Custom functions with HTTP calls or external actions
- @chain: Custom multi-step pipelines (not LangChain Runnables)

See docs/decorator_guide.md for comprehensive framework guidance.
"""

import json
import os
import sys
from typing import List, Dict, Any

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dotenv import load_dotenv
load_dotenv()

# Initialize Neatlogs v4 FIRST (before any LangChain imports)
from neatlogs.sdk.neatlogs_sdk_v4 import init, workflow, tool, chain, flush

init(
    api_key=os.getenv("NEATLOGS_API_KEY", "EQZO8zBsJkDRvmIL06m-EQ7faMUHEIN5"),
    workflow_name="langchain-agentic-demo",
    user_id="demo_user",
    session_id="agentic_session_002",
    tags=["langchain", "agents", "demo"],
    enable_http_tracing=True,
    instrumentations=["openai", "langchain"],
    debug=True
)

# Now import LangChain
import requests
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.tools import tool as langchain_tool


# =============================================================================
# TOOLS - Stack @langchain_tool + @tool for both LangChain and tracing
# =============================================================================

@langchain_tool
# @tool(name="fetch_stock_price")
def fetch_stock_price(symbol: str) -> str:
    """Fetch real stock price from API for a given symbol like AAPL, GOOGL, MSFT."""
    try:
        response = requests.get(
            f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=10
        )
        if response.status_code == 200:
            data = response.json()
            result = data.get("chart", {}).get("result", [{}])[0]
            meta = result.get("meta", {})
            return json.dumps({
                "symbol": symbol,
                "price": meta.get("regularMarketPrice", "N/A"),
                "currency": meta.get("currency", "USD"),
                "exchange": meta.get("exchangeName", "Unknown")
            })
        return json.dumps({"error": f"Failed to fetch: {response.status_code}"})
    except Exception as e:
        return json.dumps({"error": str(e)})


@langchain_tool
# @tool(name="fetch_company_news")
def fetch_company_news(company: str) -> str:
    """Fetch recent news (useful for market sentiment)."""
    try:
        response = requests.get(
            "https://hacker-news.firebaseio.com/v0/topstories.json",
            timeout=10
        )
        if response.status_code == 200:
            story_ids = response.json()[:3]
            news = []
            for story_id in story_ids:
                story_resp = requests.get(
                    f"https://hacker-news.firebaseio.com/v0/item/{story_id}.json",
                    timeout=5
                )
                if story_resp.status_code == 200:
                    story = story_resp.json()
                    news.append({
                        "title": story.get("title", ""),
                        "url": story.get("url", ""),
                        "score": story.get("score", 0)
                    })
            return json.dumps(news)
        return json.dumps([{"error": "Failed to fetch news"}])
    except Exception as e:
        return json.dumps([{"error": str(e)}])


# @tool(name="calculate_portfolio_value")
def calculate_portfolio_value(holdings: Dict[str, int]) -> str:
    """Calculate total portfolio value given holdings."""
    total = 0
    breakdown = {}
    
    for symbol, shares in holdings.items():
        stock_data = json.loads(fetch_stock_price(symbol))
        if "price" in stock_data and stock_data["price"] != "N/A":
            value = float(stock_data["price"]) * shares
            breakdown[symbol] = {
                "shares": shares,
                "price": stock_data["price"],
                "value": round(value, 2)
            }
            total += value
    
    return json.dumps({
        "total_value": round(total, 2),
        "breakdown": breakdown,
        "currency": "USD"
    })


# =============================================================================
# CHAINS - Use @chain for custom multi-step pipelines
# NOT for LangChain Runnables (those are auto-instrumented)
# =============================================================================

# @chain(name="data_enrichment_pipeline")
def enrich_stock_data(symbol: str) -> str:
    """
    Custom pipeline that enriches stock data.
    We use @chain because this is OUR function, not a LangChain Runnable.
    """
    # Step 1: Get stock price
    price_data = json.loads(fetch_stock_price(symbol))
    
    # Step 2: Get related news
    news_data = json.loads(fetch_company_news(symbol))
    
    # Step 3: Use LangChain for sentiment (auto-instrumented, no decorator needed)
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Analyze sentiment. Return: POSITIVE, NEGATIVE, or NEUTRAL."),
        ("human", "{text}")
    ])
    sentiment_chain = prompt | model | StrOutputParser()
    
    sentiment = "NEUTRAL"
    if news_data and isinstance(news_data, list) and len(news_data) > 0 and "title" in news_data[0]:
        titles = " ".join([n.get("title", "") for n in news_data[:2]])
        sentiment = sentiment_chain.invoke({"text": titles})
    
    return json.dumps({
        "symbol": symbol,
        "price_data": price_data,
        "news": news_data,
        "market_sentiment": sentiment
    })


# @chain(name="portfolio_analysis_pipeline")
def analyze_portfolio(holdings: Dict[str, int]) -> str:
    """
    Custom pipeline for portfolio analysis.
    """
    # Step 1: Calculate portfolio value
    portfolio_value = json.loads(calculate_portfolio_value(holdings))
    
    # Step 2: Get enriched data for top holding
    top_holding = max(holdings.items(), key=lambda x: x[1])[0]
    enriched_data = json.loads(enrich_stock_data(top_holding))
    
    # Step 3: Generate advice using LangChain (auto-instrumented)
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Provide brief investment advice (2-3 sentences)."),
        ("human", "Portfolio: {portfolio}\nNews: {news}")
    ])
    advice_chain = prompt | model | StrOutputParser()
    
    advice = advice_chain.invoke({
        "portfolio": str(portfolio_value),
        "news": str(enriched_data.get("news", [])[:2])
    })
    
    return json.dumps({
        "portfolio": portfolio_value,
        "top_holding_analysis": enriched_data,
        "advice": advice
    })


# =============================================================================
# LANGCHAIN AGENT - NO @agent decorator needed!
# LangChain OpenInference instrumentation auto-creates AGENT spans
# =============================================================================

def run_financial_agent(query: str) -> str:
    """
    LangChain Agent - auto-instrumented by OpenInference.
    
    NO @agent decorator needed because:
    - AgentExecutor creates AGENT spans automatically
    - Tool calls create TOOL spans automatically  
    - LLM calls create LLM spans automatically
    """
    
    # Tools are stacked with @langchain_tool + @tool
    # So they work with AgentExecutor AND have Neatlogs tracing
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a financial research assistant. Be concise."),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    tools = [fetch_stock_price, fetch_company_news]
    agent = create_openai_functions_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    result = agent_executor.invoke({"input": query})
    return result["output"]


# =============================================================================
# WORKFLOW - Always required for the entry point
# =============================================================================

# @workflow(name="financial_assistant_workflow")
def main():
    """
    Entry point - @workflow is always required.
    
    Trace hierarchy:
    
    WORKFLOW: financial_assistant_workflow
    │
    ├── [LangChain auto-instrumented]
    │   ├── AGENT: AgentExecutor
    │   ├── LLM: ChatOpenAI, ChatCompletion
    │   └── CHAIN: RunnableSequence, ChatPromptTemplate
    │
    ├── [Our @tool decorators]
    │   ├── TOOL: fetch_stock_price (with HTTP)
    │   ├── TOOL: fetch_company_news (with HTTP)
    │   └── TOOL: calculate_portfolio_value
    │
    └── [Our @chain decorators]
        ├── CHAIN: data_enrichment_pipeline
        └── CHAIN: portfolio_analysis_pipeline
    """
    
    print("\n" + "="*60)
    print("EXAMPLE 1: LangChain Agent (auto-instrumented)")
    print("="*60)
    
    agent_result = run_financial_agent(
        "What's the current price of Apple stock?"
    )
    print(f"\nAgent Result: {agent_result[:200]}...")
    
    # print("\n" + "="*60)
    # print("EXAMPLE 2: Custom Pipeline with @chain")
    # print("="*60)
    
    # portfolio_result = analyze_portfolio({
    #     "AAPL": 10,
    #     "GOOGL": 5
    # })
    # print(f"\nPortfolio Value: ${portfolio_result['portfolio']['total_value']}")
    # print(f"Advice: {portfolio_result['advice'][:150]}...")
    
    # print("\n" + "="*60)
    # print("DONE - Check Neatlogs dashboard for traces!")
    # print("="*60)
    
    # return {
    #     "agent_result": agent_result,
    #     "portfolio_result": portfolio_result
    # }


if __name__ == "__main__":
    result = main()
    flush()
    print("\n✅ All spans sent to Neatlogs!")
