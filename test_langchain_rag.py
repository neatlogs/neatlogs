"""
Test LangChain RAG with database tools to see what OpenInference captures.

This file tests RAG-style workflows with:
1. Vector database queries (simulated with ClickHouse)
2. Postgres metadata queries
3. Document retrieval
4. LLM with retrieved context
"""

import os
import sys
import requests

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

import neatlogs
from langchain_classic.agents import AgentExecutor, create_openai_functions_agent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import Tool
from dotenv import load_dotenv

load_dotenv()

# Initialize NeatLogs
neatlogs.init(
    api_key=os.getenv("NEATLOGS_API_KEY", "EQZO8zBsJkDRvmIL06m-EQ7faMUHEIN5"),
    tags=["v4", "langchain", "rag-testing"],
    workflow_name="langchain-rag-test"
)

# ============================================================================
# RAG TOOL 1: Query Postgres for Trace Metadata
# ============================================================================
def query_postgres_traces(search_term: str) -> str:
    """Query Postgres to find traces by workflow name or session ID."""
    print(f"[RAG Tool] Searching Postgres for traces: {search_term}")
    try:
        import psycopg2
        
        conn = psycopg2.connect(
            host="localhost",
            port=5432,
            dbname="neatlogs",
            user="neatlogs",
            password="neatlogs"
        )
        cursor = conn.cursor()
        
        # Search for traces
        query = """
        SELECT 
            trace_id,
            workflow_name,
            status,
            created_at,
            total_tokens
        FROM traces
        WHERE workflow_name ILIKE %s
        ORDER BY created_at DESC
        LIMIT 5
        """
        
        cursor.execute(query, (f"%{search_term}%",))
        rows = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        if rows:
            result = f"Found {len(rows)} traces matching '{search_term}':\n"
            for row in rows:
                result += f"  - {row[0][:8]}... | {row[1]} | {row[2]} | tokens: {row[4]}\n"
            return result
        else:
            return f"No traces found matching '{search_term}'"
            
    except Exception as e:
        return f"Error querying Postgres: {str(e)}"


postgres_tool = Tool(
    name="query_postgres_traces",
    func=query_postgres_traces,
    description="Search Postgres database for trace metadata by workflow name or session ID"
)


# ============================================================================
# RAG TOOL 2: Vector Search in ClickHouse (Simulated RAG)
# ============================================================================
def vector_search_clickhouse(query: str) -> str:
    """Search ClickHouse for semantically similar LLM interactions."""
    print(f"[RAG Tool] Vector search in ClickHouse: {query}")
    try:
        # Use ClickHouse HTTP API
        clickhouse_url = "http://localhost:8123"
        
        # Simulate vector search with keyword matching
        sql_query = f"""
        SELECT 
            trace_id,
            span_name,
            span_type,
            llm_model,
            llm_input_messages,
            llm_output_messages,
            total_tokens,
            cost_usd
        FROM neatlogs.spans
        WHERE 
            span_type = 'LLM'
            AND (
                llm_input_messages ILIKE '%{query}%'
                OR llm_output_messages ILIKE '%{query}%'
            )
        ORDER BY start_ts DESC
        LIMIT 3
        FORMAT JSON
        """
        
        response = requests.post(
            clickhouse_url,
            data=sql_query,
            timeout=5
        )
        
        if response.status_code == 200:
            data = response.json()
            rows = data.get('data', [])
            
            if rows:
                result = f"Found {len(rows)} similar LLM interactions for '{query}':\n\n"
                for i, row in enumerate(rows, 1):
                    result += f"{i}. Model: {row['llm_model']}, Tokens: {row['total_tokens']}\n"
                    result += f"   Input: {row['llm_input_messages'][:100]}...\n"
                    result += f"   Output: {row['llm_output_messages'][:100]}...\n\n"
                return result
            else:
                return f"No similar interactions found for '{query}'"
        else:
            return f"ClickHouse query failed (status: {response.status_code})"
            
    except Exception as e:
        return f"Error searching ClickHouse: {str(e)}"


clickhouse_tool = Tool(
    name="vector_search_clickhouse",
    func=vector_search_clickhouse,
    description="Search ClickHouse for similar LLM interactions using semantic/keyword search"
)


# ============================================================================
# RAG TOOL 3: Retrieve Agent Execution Details
# ============================================================================
def retrieve_agent_details(trace_id: str) -> str:
    """Retrieve detailed agent execution information from Postgres."""
    print(f"[RAG Tool] Retrieving agent details for trace: {trace_id}")
    try:
        import psycopg2
        
        conn = psycopg2.connect(
            host="localhost",
            port=5432,
            dbname="neatlogs",
            user="neatlogs",
            password="neatlogs"
        )
        cursor = conn.cursor()
        
        # Get agent information
        query = """
        SELECT 
            agent_name,
            agent_role,
            agent_goal,
            status,
            total_tokens,
            cost_usd,
            duration_ms
        FROM agents
        WHERE trace_id = %s
        """
        
        cursor.execute(query, (trace_id,))
        rows = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        if rows:
            result = f"Agent execution details for trace {trace_id[:8]}...:\n\n"
            for row in rows:
                result += f"Agent: {row[0]}\n"
                result += f"Role: {row[1]}\n"
                result += f"Goal: {row[2]}\n"
                result += f"Status: {row[3]}\n"
                result += f"Tokens: {row[4]}, Cost: ${row[5]}, Duration: {row[6]}ms\n\n"
            return result
        else:
            return f"No agent details found for trace {trace_id}"
            
    except Exception as e:
        return f"Error retrieving agent details: {str(e)}"


agent_details_tool = Tool(
    name="retrieve_agent_details",
    func=retrieve_agent_details,
    description="Retrieve detailed agent execution information from Postgres by trace ID"
)


# ============================================================================
# CREATE LANGCHAIN RAG AGENT
# ============================================================================

# LLM
llm = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-4o-mini",
    temperature=0
)

# RAG Tools
rag_tools = [
    postgres_tool,
    clickhouse_tool,
    agent_details_tool
]

# Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful RAG assistant that helps users analyze LLM traces and agent executions.

You have access to:
- query_postgres_traces: Search trace metadata by workflow name
- vector_search_clickhouse: Search for similar LLM interactions
- retrieve_agent_details: Get detailed agent execution info by trace ID

Use these tools to answer user questions about their LLM traces."""),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Create RAG agent
agent = create_openai_functions_agent(llm, rag_tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=rag_tools,
    verbose=True,
    max_iterations=10,
    return_intermediate_steps=True
)


# ============================================================================
# TEST RAG SCENARIOS
# ============================================================================

def test_rag_scenario_1():
    """Test searching for traces in Postgres."""
    print("\n" + "="*80)
    print("RAG TEST 1: Search Postgres for Traces")
    print("="*80)
    result = agent_executor.invoke({
        "input": "Find traces from workflows related to 'langchain'"
    })
    print(f"\n✅ Result: {result['output']}")


def test_rag_scenario_2():
    """Test semantic search in ClickHouse."""
    print("\n" + "="*80)
    print("RAG TEST 2: Semantic Search in ClickHouse")
    print("="*80)
    result = agent_executor.invoke({
        "input": "Find LLM interactions where the user asked about 'programming languages'"
    })
    print(f"\n✅ Result: {result['output']}")


def test_rag_scenario_3():
    """Test retrieving agent details."""
    print("\n" + "="*80)
    print("RAG TEST 3: Retrieve Agent Details")
    print("="*80)
    # First, get a trace ID
    result1 = agent_executor.invoke({
        "input": "Find the most recent langchain trace and tell me its trace_id"
    })
    print(f"\n✅ Step 1 Result: {result1['output']}")
    
    # Then get details (note: this may fail if no traces exist)
    # result2 = agent_executor.invoke({
    #     "input": f"Get detailed agent execution information for the trace you just found"
    # })
    # print(f"\n✅ Step 2 Result: {result2['output']}")


def test_rag_scenario_4():
    """Test complex RAG query."""
    print("\n" + "="*80)
    print("RAG TEST 4: Complex RAG Query")
    print("="*80)
    result = agent_executor.invoke({
        "input": """Do the following:
        1. Search for traces with 'test' in the workflow name
        2. Find similar LLM interactions about 'weather' or 'temperature'
        3. Summarize what you found"""
    })
    print(f"\n✅ Result: {result['output']}")


# ============================================================================
# RUN RAG TESTS
# ============================================================================

if __name__ == "__main__":
    print("🚀 Starting LangChain RAG Testing...")
    print(f"📍 Backend URL: http://localhost:3000")
    print(f"🔧 Testing RAG scenarios:")
    print("   1. Postgres trace metadata search")
    print("   2. ClickHouse semantic/vector search")
    print("   3. Agent execution detail retrieval")
    print("   4. Complex multi-step RAG query")
    print()
    print("🔍 What to observe:")
    print("   - LLM spans for agent decisions")
    print("   - TOOL spans for each database query")
    print("   - Nested HTTP spans for ClickHouse API calls")
    print("   - Database query spans (if psycopg2 instrumented)")
    print()
    
    try:
        test_rag_scenario_1()
        test_rag_scenario_2()
        # test_rag_scenario_3()  # May fail if no traces
        test_rag_scenario_4()
        
        print("\n" + "="*80)
        print("✅ All RAG tests completed!")
        print("="*80)
        print("\n📊 Check NeatLogs UI at http://localhost:3000")
        print("\nExpected Captures:")
        print("  ✅ LLM decision-making spans")
        print("  ✅ TOOL spans for each database query")
        print("  ✅ HTTP spans for ClickHouse API calls")
        print("  ⚠️  Postgres query spans (if psycopg2 instrumented)")
        print("\nThis demonstrates RAG pattern: Agent retrieves context from DB, then answers")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
