"""
Example 14: Interactive Multi-Agent Chatbot + PromptTemplate

This example demonstrates:
1. Continuous interactive conversation session
2. 5 AI agents in a chat assistant workflow
3. Multiple tools with HTTP requests (some with error simulation)
4. RAG tool with ChromaDB for knowledge retrieval
5. Parallel agent execution (multiple agents running simultaneously)
6. Parallel tool calls (multiple tools called at once)
7. ✨ NEW: PromptTemplate - NO variable duplication!

Architecture:
- Coordinator Agent: Routes queries to specialized agents
- Research Agent: Gathers information (parallel with Fact Checker)
- Fact Checker Agent: Verifies information (parallel with Research)
- RAG Agent: Retrieves knowledge from vector DB
- Writer Agent: Synthesizes final response
- Error Handler Agent: Handles failures gracefully

Tools:
- Web Search (HTTP, with error simulation)
- Weather API (REAL HTTP call to wttr.in)
- Stock Price API (HTTP, with error simulation)
- Calculator (local)
- RAG Search (ChromaDB)

Interactive Features:
- Type 'quit', 'exit', or 'q' to end session
- Type 'history' to see conversation history
- Type 'clear' to reset conversation
- Each query maintains context from previous messages

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
# 1. TOOLS WITH HTTP REQUESTS & ERROR SIMULATION (Exact copy from example 11)
# ============================================================================

@tool
def web_search(query: str) -> str:
    """Search the web for information. May fail occasionally."""
    import random
    if random.random() < 0.2:
        raise Exception(f"Web search API error: Rate limit exceeded for query '{query}'")
    
    time.sleep(0.1)
    return f"Web results for '{query}': [Mock results - latest news and articles]"


@tool
def get_weather(city: str) -> str:
    """Get current weather for a city via HTTP API."""
    try:
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
def calculator(expression: str) -> str:
    """Evaluate mathematical expressions."""
    try:
        result = eval(expression)
        return f"{expression} = {result}"
    except Exception as e:
        return f"Calculator error: {str(e)}"


@tool
def get_stock_price(ticker: str) -> str:
    """Get stock price. Simulates API errors."""
    import random
    if random.random() < 0.3:
        raise Exception(f"Stock API error: Failed to fetch {ticker}")
    
    prices = {"AAPL": "$175.23", "GOOGL": "$142.50", "MSFT": "$380.12"}
    return f"{ticker}: {prices.get(ticker.upper(), '$100.00')}"


# ============================================================================
# 2. RAG SETUP (ChromaDB)
# ============================================================================

def setup_rag_knowledge_base():
    """Setup ChromaDB with sample knowledge base."""
    client = chromadb.Client()
    
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
# 3. AGENT STATE
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
# 4. AGENT NODES
# ============================================================================

def coordinator_node(state: AgentState) -> dict:
    """Coordinator: Routes query to appropriate agents."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # ✅ Use PromptTemplate for coordinator
    template = PromptTemplate([
        {"role": "system", "content": """You are a coordinator for a chatbot. Analyze the query and decide which agents to invoke.

Available agents:
- writer: For conversational queries, greetings, small talk, opinions, simple questions
- research: For queries needing web search or external information
- fact_checker: For verifying facts or checking stock prices
- rag: For queries about stored knowledge
- error_handler: For handling errors

Rules:
1. For greetings (hi, hello, hey), small talk, or simple conversational queries → ONLY writer
2. For complex informational queries → research, rag (parallel)
3. For fact verification → fact_checker
4. You can invoke multiple agents in parallel for complex queries

Respond with EXACTLY: AGENTS: [agent1, agent2, ...]
Examples:
- "Hi" → AGENTS: [writer]
- "How are you?" → AGENTS: [writer]
- "What is quantum computing?" → AGENTS: [research, rag]
- "Check if the sky is blue" → AGENTS: [fact_checker]"""},
        {"role": "user", "content": "{{query}}"}
    ])

    # Get the last HumanMessage (user query)
    user_query = next(msg.content for msg in reversed(state["messages"]) if isinstance(msg, HumanMessage))

    with trace("coordinator_node", prompt_template=template):
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

        # Parse the coordinator's response
        next_agents = []
        if "writer" in content:
            next_agents.append("writer")
        if "research" in content:
            next_agents.append("research")
        if "fact_checker" in content or "fact" in content:
            next_agents.append("fact_checker")
        if "rag" in content or "knowledge" in content:
            next_agents.append("rag")

        # Default to writer for conversational queries if no agents detected
        if not next_agents:
            next_agents = ["writer"]

        return {
            "messages": [AIMessage(content=f"Routing to: {', '.join(next_agents)}")],
            "next_agents": next_agents,
            "error_log": []
        }


def research_node(state: AgentState) -> dict:
    """Research Agent: Gathers information using tools."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    tools = [web_search, get_weather, calculator]
    llm_with_tools = llm.bind_tools(tools)

    # ✅ Use PromptTemplate for research agent
    template = PromptTemplate([
        {"role": "system", "content": "You are a research agent. Use tools to gather information."},
        {"role": "user", "content": "{{query}}"}
    ])

    # Get the last HumanMessage (user query)
    user_query = next(msg.content for msg in reversed(state["messages"]) if isinstance(msg, HumanMessage))

    try:
        with trace("research_node", prompt_template=template):
            # Variables specified ONCE - no duplication!
            messages_compiled = template.compile(query=user_query)

            # Convert to LangChain messages
            messages = [
                SystemMessage(content=msg["content"]) if msg["role"] == "system"
                else HumanMessage(content=msg["content"])
                for msg in messages_compiled
            ]

            response = llm_with_tools.invoke(messages)

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


def fact_checker_node(state: AgentState) -> dict:
    """Fact Checker: Verifies information (runs parallel with research)."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    tools = [get_stock_price]
    llm_with_tools = llm.bind_tools(tools)

    # ✅ Use PromptTemplate for fact checker
    template = PromptTemplate([
        {"role": "system", "content": "You are a fact checker. Verify claims and check data accuracy."},
        {"role": "user", "content": "{{query}}"}
    ])

    # Get the last HumanMessage (user query)
    user_query = next(msg.content for msg in reversed(state["messages"]) if isinstance(msg, HumanMessage))

    try:
        with trace("fact_checker_node", prompt_template=template):
            # Variables specified ONCE - no duplication!
            messages_compiled = template.compile(query=user_query)

            # Convert to LangChain messages
            messages = [
                SystemMessage(content=msg["content"]) if msg["role"] == "system"
                else HumanMessage(content=msg["content"])
                for msg in messages_compiled
            ]

            response = llm_with_tools.invoke(messages)

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


def rag_node(state: AgentState) -> dict:
    """RAG Agent: Retrieves knowledge from vector DB."""
    # ✅ Use PromptTemplate for RAG agent
    template = PromptTemplate("{{query}}")

    try:
        # Get the last HumanMessage (user query)
        query = next(msg.content for msg in reversed(state["messages"]) if isinstance(msg, HumanMessage))

        with trace("rag_node", prompt_template=template):
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


def writer_node(state: AgentState) -> dict:
    """Writer: Synthesizes final response from all agent results."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

    # ✅ Use PromptTemplate for writer agent
    context = []
    if state.get("research_result"):
        context.append(f"Research: {state['research_result']}")
    if state.get("fact_check_result"):
        context.append(f"Facts: {state['fact_check_result']}")
    if state.get("rag_result"):
        context.append(f"Knowledge: {state['rag_result']}")

    context_str = "\n".join(context) if context else "No additional context available."
    # Get the last HumanMessage (user query)
    user_query = next(msg.content for msg in reversed(state["messages"]) if isinstance(msg, HumanMessage))

    template = PromptTemplate([
        {"role": "system", "content": "Synthesize a comprehensive answer using:\n{{context}}"},
        {"role": "user", "content": "{{query}}"}
    ])

    with trace("writer_node", prompt_template=template):
        # Variables specified ONCE - no duplication!
        messages_compiled = template.compile(
            context=context_str,
            query=user_query
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


def error_handler_node(state: AgentState) -> dict:
    """Error Handler: Provides fallback when tools fail."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)

    # ✅ Use PromptTemplate for error handler
    errors = state.get("error_log", [])
    error_summary = "; ".join(errors) if errors else "Unknown errors"
    # Get the last HumanMessage (user query)
    user_query = next(msg.content for msg in reversed(state["messages"]) if isinstance(msg, HumanMessage))

    template = PromptTemplate([
        {"role": "system", "content": "Tool errors occurred: {{errors}}. Provide a helpful fallback response."},
        {"role": "user", "content": "{{query}}"}
    ])

    with trace("error_handler_node", prompt_template=template):
        # Variables specified ONCE - no duplication!
        messages_compiled = template.compile(
            errors=error_summary,
            query=user_query
        )

        # Convert to LangChain messages
        messages = [
            SystemMessage(content=msg["content"]) if msg["role"] == "system"
            else HumanMessage(content=msg["content"])
            for msg in messages_compiled
        ]

        response = llm.invoke(messages)

        return {
            "final_answer": f"[Partial Response - Some tools failed] {response.content}",
            "messages": [AIMessage(content=response.content)]
        }


# ============================================================================
# 5. GRAPH CONSTRUCTION WITH PARALLEL EXECUTION
# ============================================================================

def build_multi_agent_graph():
    """Build LangGraph with parallel agent execution."""
    workflow = StateGraph(AgentState)
    
    workflow.add_node("coordinator", coordinator_node)
    workflow.add_node("research", research_node)
    workflow.add_node("fact_checker", fact_checker_node)
    workflow.add_node("rag", rag_node)
    workflow.add_node("writer", writer_node)
    workflow.add_node("error_handler", error_handler_node)
    
    workflow.set_entry_point("coordinator")
    
    def route_after_coordinator(state: AgentState):
        """Route to multiple agents in parallel."""
        next_agents = state.get("next_agents", ["writer"])
        return next_agents

    workflow.add_conditional_edges(
        "coordinator",
        route_after_coordinator,
        {
            "research": "research",
            "fact_checker": "fact_checker",
            "rag": "rag",
            "writer": "writer",
        }
    )
    
    workflow.add_edge("research", "writer")
    workflow.add_edge("fact_checker", "writer")
    workflow.add_edge("rag", "writer")
    
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
# 6. INTERACTIVE CHATBOT WRAPPER
# ============================================================================

def print_banner():
    """Print chatbot banner."""
    print("\n" + "=" * 80)
    print("🤖 NEATLOGS MULTI-AGENT CHATBOT")
    print("=" * 80)
    print("Features:")
    print("  • Multi-agent system with parallel execution")
    print("  • Tools: Web Search, Weather API, Calculator, Stock Prices, RAG")
    print("  • Conversation history maintained across queries")
    print("  • Full observability with Neatlogs")
    print("\nCommands:")
    print("  • Type your question to chat")
    print("  • 'history' - Show conversation history")
    print("  • 'clear' - Reset conversation")
    print("  • 'quit', 'exit', 'q' - End session")
    print("=" * 80 + "\n")


def interactive_chatbot():
    """Run interactive chatbot session."""
    print_banner()
    
    # Setup (wrapped in trace to avoid orphan ChromaDB spans)
    with trace("setup_session"):
        print("🔧 Setting up knowledge base...")
        rag_collection = setup_rag_knowledge_base()
        print("✓ Knowledge base ready")
        
        print("🔧 Building multi-agent graph...")
        graph = build_multi_agent_graph()
        print("✓ Agent graph ready\n")
    
    # Conversation state
    conversation_history = []
    session_start = time.time()
    query_count = 0
    
    print("💬 Chatbot ready! Ask me anything...\n")
    
    # Main chat loop
    while True:
        try:
            # Get user input
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Ending session...")
                break
            
            if user_input.lower() == 'history':
                print("\n" + "=" * 80)
                print("📜 CONVERSATION HISTORY")
                print("=" * 80)
                for i, msg in enumerate(conversation_history, 1):
                    role = "You" if isinstance(msg, HumanMessage) else "Bot"
                    print(f"{i}. {role}: {msg.content[:100]}...")
                print("=" * 80 + "\n")
                continue
            
            if user_input.lower() == 'clear':
                conversation_history = []
                query_count = 0
                print("✓ Conversation cleared\n")
                continue
            
            # Process query
            query_count += 1
            human_msg = HumanMessage(content=user_input)
            conversation_history.append(human_msg)
            
            print("\n🤔 Processing...\n")

            # ✅ Use PromptTemplate to capture query variable
            template = PromptTemplate("User: {{query}}")

            # Each query creates a NEW ROOT TRACE (separate trace_id, same session_id)
            with trace(f"chatbot_query_{query_count}", prompt_template=template):
                # Compile template to capture variables (only 1 variable: query)
                template.compile(query=user_input)

                # Run graph
                initial_state = {
                    "messages": conversation_history.copy(),
                    "next_agents": [],
                    "research_result": "",
                    "fact_check_result": "",
                    "rag_result": "",
                    "final_answer": "",
                    "error_log": []
                }
                
                result = graph.invoke(initial_state)
                
                # Extract response
                final_answer = result.get("final_answer", "I couldn't generate a response.")
                
                # Add to history
                ai_msg = AIMessage(content=final_answer)
                conversation_history.append(ai_msg)
                
                # Display response
                print(f"🤖 Bot: {final_answer}\n")
                
                # Show errors if any
                if result.get("error_log"):
                    print("⚠️  Partial response due to errors:")
                    for error in result["error_log"]:
                        print(f"   - {error}")
                    print()
        
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted. Type 'quit' to exit or continue chatting...\n")
            continue
        except Exception as e:
            print(f"\n❌ Error: {e}\n")
            continue
    
    # Session summary
    session_duration = time.time() - session_start
    print("\n" + "=" * 80)
    print("📊 SESSION SUMMARY")
    print("=" * 80)
    print(f"Duration: {session_duration:.1f}s")
    print(f"Total queries: {query_count}")
    print(f"Messages in history: {len(conversation_history)}")
    print("=" * 80 + "\n")


def main():
    """Main entry point."""
    # Enable span logging for debugging (metrics are in span attributes)
    os.environ["NEATLOGS_LOG_SPANS"] = "true"
    
    # Initialize Neatlogs with auto-session for testing
    init(
        api_key=os.getenv("NEATLOGS_API_KEY", "test-key"),
        workflow_name="interactive_chatbot",
        auto_session=True,  # Auto-generate session ID for testing
        instrumentations=["openai", "langchain", "chromadb"],
        debug=False,
    )
    
    try:
        # Run interactive chatbot
        # Note: Each trace() inside the loop will create a NEW root trace
        # due to auto_session=True, allowing multi-turn conversation tracking
        interactive_chatbot()
    
    finally:
        print("📤 Flushing traces...")
        flush()
        shutdown()
        print("✓ Session ended. Check spans.log for observability data (metrics embedded as span attributes).\n")


if __name__ == "__main__":
    main()
