"""
Example 11: Multi-Agent System with Parallel Execution + PromptTemplate

This example demonstrates:
1. 5 AI agents in a chat assistant workflow
2. Multiple tools with HTTP requests (some with error simulation)
3. RAG tool with ChromaDB for knowledge retrieval
4. Parallel agent execution (multiple agents running simultaneously)
5. Parallel tool calls (multiple tools called at once)
6. ✨ NEW: PromptTemplate - NO variable duplication!

Architecture:
- Coordinator Agent: Routes queries to specialized agents
- Research Agent: Gathers information (parallel with Fact Checker)
- Fact Checker Agent: Verifies information (parallel with Research)
- RAG Agent: Retrieves knowledge from vector DB
- Writer Agent: Synthesizes final response
- Error Handler Agent: Handles failures gracefully

Tools:
- Web Search (HTTP, with error simulation)
- Weather API (HTTP)
- Stock Price API (HTTP, with error simulation)
- Calculator (local)
- RAG Search (ChromaDB)

PromptTemplate Feature:
Instead of specifying variables twice:
  ❌ OLD: with trace("query", prompt_template="...", prompt_variables={...})
  ✅ NEW: template.compile(query=...) - variables specified ONCE!
"""

import os
import sys
import time
import random
from typing import TypedDict, Annotated, Sequence
from operator import add

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from neatlogs.sdk.neatlogs_sdk_v4_langfuse import init, trace, flush, shutdown, PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
import requests
import chromadb
from chromadb.utils import embedding_functions
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["NEATLOGS_LOG_SPANS"] = "true"
os.environ["NEATLOGS_LOG_METRICS"] = "true"


# ============================================================================
# 1. TOOLS WITH HTTP REQUESTS & ERROR SIMULATION
# ============================================================================

@tool
def web_search(query: str) -> str:
    """Search the web for information. May fail occasionally."""
    # Simulate 20% failure rate
    if random.random() < 0.2:
        raise Exception(f"Web search API error: Rate limit exceeded for query '{query}'")
    
    # Simulate HTTP request
    time.sleep(0.1)
    return f"Web results for '{query}': [Mock results - latest news and articles]"


@tool
def get_weather(city: str) -> str:
    """Get current weather for a city via HTTP API."""
    try:
        # Real HTTP call (using a free weather API)
        response = requests.get(
            f"https://wttr.in/{city}?format=%C+%t",
            timeout=5
        )
        if response.status_code == 200:
            return f"Weather in {city}: {response.text}"
        else:
            return f"Weather in {city}: Sunny, 72°F (mock fallback)"
    except Exception as e:
        return f"Weather service unavailable: {str(e)}"


@tool
def get_stock_price(ticker: str) -> str:
    """Get stock price. Simulates API errors."""
    # Simulate 30% failure rate
    if random.random() < 0.3:
        raise Exception(f"Stock API error: Failed to fetch {ticker}")
    
    # Mock stock prices
    prices = {"AAPL": "$175.23", "GOOGL": "$142.50", "MSFT": "$380.12"}
    return f"{ticker}: {prices.get(ticker.upper(), '$100.00')}"


@tool
def calculator(expression: str) -> str:
    """Evaluate mathematical expressions."""
    try:
        result = eval(expression)
        return f"{expression} = {result}"
    except Exception as e:
        return f"Calculator error: {str(e)}"


# ============================================================================
# 2. RAG TOOL WITH CHROMADB
# ============================================================================

def setup_rag_knowledge_base():
    """Setup ChromaDB with sample knowledge base."""
    client = chromadb.Client()
    
    # Create or get collection
    try:
        collection = client.get_collection("knowledge_base")
        client.delete_collection("knowledge_base")
    except:
        pass
    
    embedding_fn = embedding_functions.DefaultEmbeddingFunction()
    collection = client.create_collection(
        "knowledge_base",
        embedding_function=embedding_fn
    )
    
    # Add knowledge documents
    documents = [
        "Artificial Intelligence is the simulation of human intelligence by machines.",
        "Machine Learning is a subset of AI that learns from data without explicit programming.",
        "Deep Learning uses neural networks with multiple layers to learn complex patterns.",
        "Natural Language Processing enables computers to understand human language.",
        "Computer Vision allows machines to interpret and understand visual information.",
        "Python is the most popular programming language for AI and ML development.",
        "TensorFlow and PyTorch are leading frameworks for deep learning.",
        "OpenAI developed GPT models which are large language models.",
        "LangChain is a framework for building applications with large language models.",
        "Vector databases store embeddings for semantic search and RAG applications.",
    ]
    
    collection.add(
        documents=documents,
        ids=[f"doc_{i}" for i in range(len(documents))],
        metadatas=[{"source": f"knowledge_{i}"} for i in range(len(documents))]
    )
    
    return collection


@tool
def rag_search(query: str) -> str:
    """Search knowledge base using RAG."""
    try:
        client = chromadb.Client()
        collection = client.get_collection("knowledge_base")
        
        results = collection.query(
            query_texts=[query],
            n_results=2
        )
        
        if results['documents'] and results['documents'][0]:
            docs = results['documents'][0]
            return f"Knowledge base results: {' | '.join(docs)}"
        else:
            return "No relevant knowledge found."
    except Exception as e:
        return f"RAG search error: {str(e)}"


# ============================================================================
# 3. STATE DEFINITION
# ============================================================================

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add]
    next_agents: list[str]
    research_result: str
    fact_check_result: str
    rag_result: str
    final_answer: str
    error_log: Annotated[list[str], add]


# ============================================================================
# 4. AGENT DEFINITIONS
# ============================================================================

def coordinator_agent(state: AgentState) -> AgentState:
    """Coordinator: Routes query to appropriate agents."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # ✅ Use PromptTemplate for coordinator
    template = PromptTemplate([
        {"role": "system", "content": """You are a coordinator. Analyze the query and decide which agents to invoke.
        Options: research, fact_checker, rag, writer, error_handler
        For complex queries, you can invoke multiple agents in parallel.
        Respond with: AGENTS: [agent1, agent2, ...]"""},
        {"role": "user", "content": "{{query}}"}
    ])

    user_query = state["messages"][-1].content

    with trace("coordinator_agent", prompt_template=template):
        # Variables specified ONCE - no duplication!
        messages_compiled = template.compile(query=user_query)

        # Convert to LangChain messages
        messages = [
            SystemMessage(content=msg["content"]) if msg["role"] == "system"
            else HumanMessage(content=msg["content"])
            for msg in messages_compiled
        ]

        response = llm.invoke(messages)
        content = response.content.lower()

        # Determine next agents (parallel execution if multiple)
        next_agents = []
        if "research" in content or "search" in content:
            next_agents.append("research")
        if "fact" in content or "verify" in content:
            next_agents.append("fact_checker")
        if "knowledge" in content or "learn" in content:
            next_agents.append("rag")

        # Default to research + RAG if uncertain
        if not next_agents:
            next_agents = ["research", "rag"]

        return {
            "messages": [AIMessage(content=f"Routing to: {', '.join(next_agents)}")],
            "next_agents": next_agents,
            "error_log": []
        }


def research_agent(state: AgentState) -> AgentState:
    """Research Agent: Gathers information using tools."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    tools = [web_search, get_weather, calculator]
    llm_with_tools = llm.bind_tools(tools)

    # ✅ Use PromptTemplate for research agent
    template = PromptTemplate([
        {"role": "system", "content": "You are a research agent. Use tools to gather information."},
        {"role": "user", "content": "{{query}}"}
    ])

    user_query = state["messages"][-1].content

    try:
        with trace("research_agent", prompt_template=template):
            # Variables specified ONCE - no duplication!
            messages_compiled = template.compile(query=user_query)

            # Convert to LangChain messages
            messages = [
                SystemMessage(content=msg["content"]) if msg["role"] == "system"
                else HumanMessage(content=msg["content"])
                for msg in messages_compiled
            ]

            response = llm_with_tools.invoke(messages)

            # Execute tool calls
            tool_results = []
            if hasattr(response, 'tool_calls') and response.tool_calls:
                for tool_call in response.tool_calls:
                    try:
                        if tool_call['name'] == 'web_search':
                            result = web_search.invoke(tool_call['args'])
                        elif tool_call['name'] == 'get_weather':
                            result = get_weather.invoke(tool_call['args'])
                        elif tool_call['name'] == 'calculator':
                            result = calculator.invoke(tool_call['args'])
                        tool_results.append(result)
                    except Exception as e:
                        error_msg = f"Research tool error: {str(e)}"
                        return {
                            "research_result": f"Partial results (error occurred): {'; '.join(tool_results)}",
                            "error_log": [error_msg]
                        }

            result = '; '.join(tool_results) if tool_results else response.content

            return {
                "research_result": result,
                "messages": [AIMessage(content=f"Research complete: {result[:100]}...")]
            }
    except Exception as e:
        return {
            "research_result": f"Research failed: {str(e)}",
            "error_log": [f"Research agent error: {str(e)}"]
        }


def fact_checker_agent(state: AgentState) -> AgentState:
    """Fact Checker: Verifies information (runs parallel with research)."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    tools = [get_stock_price]
    llm_with_tools = llm.bind_tools(tools)

    # ✅ Use PromptTemplate for fact checker
    template = PromptTemplate([
        {"role": "system", "content": "You are a fact checker. Verify claims and check data accuracy."},
        {"role": "user", "content": "{{query}}"}
    ])

    user_query = state["messages"][-1].content

    try:
        with trace("fact_checker_agent", prompt_template=template):
            # Variables specified ONCE - no duplication!
            messages_compiled = template.compile(query=user_query)

            # Convert to LangChain messages
            messages = [
                SystemMessage(content=msg["content"]) if msg["role"] == "system"
                else HumanMessage(content=msg["content"])
                for msg in messages_compiled
            ]

            response = llm_with_tools.invoke(messages)

            # Execute tool calls
            tool_results = []
            if hasattr(response, 'tool_calls') and response.tool_calls:
                for tool_call in response.tool_calls:
                    try:
                        if tool_call['name'] == 'get_stock_price':
                            result = get_stock_price.invoke(tool_call['args'])
                        tool_results.append(result)
                    except Exception as e:
                        error_msg = f"Fact check tool error: {str(e)}"
                        return {
                            "fact_check_result": f"Verification incomplete (error occurred)",
                            "error_log": [error_msg]
                        }

            result = '; '.join(tool_results) if tool_results else "No facts to verify"

            return {
                "fact_check_result": result,
                "messages": [AIMessage(content=f"Fact check: {result}")]
            }
    except Exception as e:
        return {
            "fact_check_result": f"Fact check failed: {str(e)}",
            "error_log": [f"Fact checker error: {str(e)}"]
        }


def rag_agent(state: AgentState) -> AgentState:
    """RAG Agent: Retrieves knowledge from vector DB."""
    # ✅ Use PromptTemplate for RAG agent
    template = PromptTemplate("{{query}}")

    try:
        query = state["messages"][-1].content

        with trace("rag_agent", prompt_template=template):
            # Variables specified ONCE - no duplication!
            _compiled_query = template.compile(query=query)

            result = rag_search.invoke({"query": query})

            return {
                "rag_result": result,
                "messages": [AIMessage(content=f"RAG search: {result[:100]}...")]
            }
    except Exception as e:
        return {
            "rag_result": f"RAG failed: {str(e)}",
            "error_log": [f"RAG agent error: {str(e)}"]
        }


def writer_agent(state: AgentState) -> AgentState:
    """Writer: Synthesizes final response from all agent results."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

    # ✅ Use PromptTemplate for writer agent
    # Collect all results
    context = []
    if state.get("research_result"):
        context.append(f"Research: {state['research_result']}")
    if state.get("fact_check_result"):
        context.append(f"Facts: {state['fact_check_result']}")
    if state.get("rag_result"):
        context.append(f"Knowledge: {state['rag_result']}")

    context_str = "\n".join(context)
    original_query = state["messages"][0].content

    template = PromptTemplate([
        {"role": "system", "content": "Synthesize a comprehensive answer using this context:\n{{context}}"},
        {"role": "user", "content": "{{query}}"}
    ])

    with trace("writer_agent", prompt_template=template):
        # Variables specified ONCE - no duplication!
        messages_compiled = template.compile(
            context=context_str,
            query=original_query
        )

        # Convert to LangChain messages
        messages = [
            SystemMessage(content=msg["content"]) if msg["role"] == "system"
            else HumanMessage(content=msg["content"])
            for msg in messages_compiled
        ]

        response = llm.invoke(messages)

        return {
            "final_answer": response.content,
            "messages": [AIMessage(content=response.content)]
        }


def error_handler_agent(state: AgentState) -> AgentState:
    """Error Handler: Manages failures gracefully."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)

    errors = state.get("error_log", [])
    if errors:
        # ✅ Use PromptTemplate for error handler
        template = PromptTemplate([
            {"role": "system", "content": "Errors occurred: {{errors}}. Provide a helpful fallback response."},
            {"role": "user", "content": "{{query}}"}
        ])

        errors_str = "; ".join(errors)
        original_query = state["messages"][0].content

        with trace("error_handler_agent", prompt_template=template):
            # Variables specified ONCE - no duplication!
            messages_compiled = template.compile(
                errors=errors_str,
                query=original_query
            )

            # Convert to LangChain messages
            messages = [
                SystemMessage(content=msg["content"]) if msg["role"] == "system"
                else HumanMessage(content=msg["content"])
                for msg in messages_compiled
            ]

            response = llm.invoke(messages)

            return {
                "final_answer": f"[Partial Response] {response.content}",
                "messages": [AIMessage(content=response.content)]
            }

    return state


# ============================================================================
# 5. GRAPH CONSTRUCTION WITH PARALLEL EXECUTION
# ============================================================================

def build_multi_agent_graph():
    """Build LangGraph with parallel agent execution."""
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("coordinator", coordinator_agent)
    workflow.add_node("research", research_agent)
    workflow.add_node("fact_checker", fact_checker_agent)
    workflow.add_node("rag", rag_agent)
    workflow.add_node("writer", writer_agent)
    workflow.add_node("error_handler", error_handler_agent)
    
    # Entry point
    workflow.set_entry_point("coordinator")
    
    # Parallel execution after coordinator
    def route_after_coordinator(state: AgentState):
        """Route to multiple agents in parallel."""
        next_agents = state.get("next_agents", ["research"])
        # Return list of agents to execute in parallel
        return next_agents
    
    workflow.add_conditional_edges(
        "coordinator",
        route_after_coordinator,
        {
            "research": "research",
            "fact_checker": "fact_checker",
            "rag": "rag",
        }
    )
    
    # All parallel agents converge to writer
    workflow.add_edge("research", "writer")
    workflow.add_edge("fact_checker", "writer")
    workflow.add_edge("rag", "writer")
    
    # Writer checks for errors, routes accordingly
    def route_after_writer(state: AgentState):
        if state.get("error_log"):
            return "error_handler"
        return "end"
    
    workflow.add_conditional_edges(
        "writer",
        route_after_writer,
        {
            "error_handler": "error_handler",
            "end": END
        }
    )
    
    workflow.add_edge("error_handler", END)
    
    return workflow.compile()


# ============================================================================
# 6. MAIN EXECUTION
# ============================================================================

def main():
    os.environ["NEATLOGS_LOG_SPANS"] = "true"
    
    init(
        api_key=os.getenv("NEATLOGS_API_KEY", "test-key"),
        workflow_name="multi-agent-parallel",
        instrumentations=["openai", "langchain", "chromadb"],
        debug=True,
    )
    
    # User query
    user_query = "What is machine learning and what's the weather in San Francisco?"

    print(f"\n{'='*60}")
    print(f"User Query: {user_query}")
    print('='*60)

    # ✅ NEW: Use PromptTemplate - no duplication!
    template = PromptTemplate("User asks: {{query}}")

    with trace("multi_agent_chat", prompt_template=template):
        # Variables specified ONCE in compile() - no duplication!
        _compiled_prompt = template.compile(query=user_query)
        print(f"📝 Compiled prompt: {_compiled_prompt}")

        # Setup RAG knowledge base
        print("Setting up RAG knowledge base...")
        setup_rag_knowledge_base()
        
        # Build graph
        print("Building multi-agent workflow...")
        app = build_multi_agent_graph()
        try:
            result = app.invoke({
                "messages": [HumanMessage(content=user_query)],
                "next_agents": [],
                "research_result": "",
                "fact_check_result": "",
                "rag_result": "",
                "final_answer": "",
                "error_log": []
            })
            
            print(f"\n✅ Final Answer:\n{result.get('final_answer', 'No answer')}")
            
            if result.get('error_log'):
                print(f"\n⚠️  Errors: {'; '.join(result['error_log'])}")
        
        except Exception as e:
            print(f"\n❌ Query failed: {e}")
    
    print("\n" + "="*60)
    print("✅ All queries processed!")
    print("Check spans.log to see:")
    print("  - Parallel agent execution (research + fact_checker + rag)")
    print("  - Parallel tool calls within agents")
    print("  - Error handling and recovery")
    print("  - ✨ PromptTemplate usage (no variable duplication!)")
    print("="*60)
    
    flush()
    shutdown()


if __name__ == "__main__":
    main()
