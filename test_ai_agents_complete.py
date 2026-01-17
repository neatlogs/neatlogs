"""
NeatLogs SDK - Complete AI Agent Tracing Demo
==============================================

This script demonstrates NeatLogs with REAL AI agent frameworks:
- LangChain agents with tools
- CrewAI multi-agent crews
- RAG workflows with embeddings and retrieval
- Custom decorators on top of auto-instrumentation
"""

import os
import sys
from dotenv import load_dotenv
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

import neatlogs

load_dotenv()

print("=" * 100)
print("🤖 NeatLogs SDK - Complete AI Agent Tracing Demo")
print("=" * 100)

# ============================================================================
# INITIALIZE NEATLOGS
# ============================================================================

neatlogs.init(
    api_key="EQZO8zBsJkDRvmIL06m-EQ7faMUHEIN5",
    workflow_name="ai-agents-demo",
    user_id="demo_user",
    session_id="demo_session_001",
    tags=["ai-agents", "demo", "v4"],
    enable_http_tracing=True,
    instrumentations=["openai", "langchain", "crewai"],  # Only what we need
    debug=True
)

print("\n✅ NeatLogs initialized for AI agent tracing")
print("   📊 Auto-tracing: OpenAI, LangChain, CrewAI")
print("   🌐 HTTP tracing: Enabled (captures tool API calls)")


# ============================================================================
# EXAMPLE 1: LANGCHAIN AGENT WITH TOOLS (AUTO-TRACED!)
# ============================================================================

print("\n\n" + "=" * 100)
print("📊 EXAMPLE 1: LangChain Agent with Tools")
print("=" * 100)
print("✨ Everything is AUTO-TRACED by OpenInference - no decorators needed!")

def demo_langchain_agent():
    """LangChain agent with custom tools - fully auto-traced"""
    try:
        from langchain_classic.agents import AgentExecutor, create_openai_functions_agent
        from langchain_classic.tools import tool
        from langchain_openai import ChatOpenAI
        from langchain_classic import hub
        import requests
        
        # Define custom tools (these will be auto-traced as TOOL spans!)
        @tool
        def get_weather(city: str) -> str:
            """Get current weather for a city"""
            try:
                response = requests.get(
                    f"https://wttr.in/{city}?format=j1",
                    timeout=3
                )
                if response.status_code == 200:
                    data = response.json()
                    temp = data['current_condition'][0]['temp_C']
                    weather = data['current_condition'][0]['weatherDesc'][0]['value']
                    return f"Weather in {city}: {temp}°C, {weather}"
            except Exception as e:
                return f"Could not fetch weather: {e}"
            return "Weather unavailable"
        
        @tool
        def calculate(expression: str) -> str:
            """Evaluate a mathematical expression"""
            try:
                result = eval(expression, {"__builtins__": {}}, {})
                return f"Result: {result}"
            except Exception as e:
                return f"Error: {e}"
        
        @tool
        def search_docs(query: str) -> str:
            """Search documentation (simulated)"""
            docs = {
                "python": "Python is a high-level programming language known for simplicity.",
                "javascript": "JavaScript is a scripting language for web development.",
                "rust": "Rust is a systems programming language focused on safety."
            }
            query_lower = query.lower()
            for key, doc in docs.items():
                if key in query_lower:
                    return doc
            return "No documentation found"
        
        # Create LLM (auto-traced!)
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            openai_api_key="",
            max_completion_tokens=200
        )
        
        # Get prompt template
        prompt = hub.pull("hwchase17/openai-functions-agent")
        
        # Create agent (auto-traced!)
        tools = [get_weather, calculate, search_docs]
        agent = create_openai_functions_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
        
        # Run agent (fully auto-traced: agent → LLM calls → tool calls → HTTP requests!)
        print("\n🤖 Running LangChain agent...")
        result = agent_executor.invoke({
            "input": "What's the weather in London? Also, what is 15 * 23?"
        })
        
        print(f"\n✅ Agent Result: {result['output']}")
        print("\n📊 What got traced (AUTO-CAPTURED):")
        print("   • AGENT span - AgentExecutor")
        print("   • LLM spans - ChatOpenAI calls (prompts, completions, tokens, cost)")
        print("   • TOOL spans - get_weather, calculate tools")
        print("   • HTTP span - wttr.in API call (captured by HTTP instrumentation)")
        print("   • CHAIN spans - intermediate reasoning steps")
        
        return result
        
    except ImportError as e:
        print(f"⚠️  Skipping LangChain demo: {e}")
        print("   Run: poetry add langchain langchain-openai langchain-community")
        return None
    except Exception as e:
        print(f"⚠️  Error in LangChain demo: {e}")
        return None

demo_langchain_agent()


# ============================================================================
# EXAMPLE 2: CREWAI MULTI-AGENT WORKFLOW (AUTO-TRACED!)
# ============================================================================

# print("\n\n" + "=" * 100)
# print("📊 EXAMPLE 2: CrewAI Multi-Agent Crew")
# print("=" * 100)
# print("✨ Multi-agent workflows are AUTO-TRACED - see agent collaboration!")

# def demo_crewai_agents():
#     """CrewAI multi-agent crew - fully auto-traced"""
#     try:
#         from crewai import Agent, Task, Crew
#         from crewai_tools import tool
#         import requests
        
#         # Define tools for agents (auto-traced as TOOL spans!)
#         @tool("Search Web")
#         def search_web(query: str) -> str:
#             """Search the web for information"""
#             # Simulated search
#             return f"Search results for '{query}': Found relevant articles about {query}."
        
#         @tool("Fetch GitHub Stats")
#         def fetch_github_stats(username: str) -> str:
#             """Fetch GitHub user statistics"""
#             try:
#                 response = requests.get(
#                     f"https://api.github.com/users/{username}",
#                     timeout=3
#                 )
#                 if response.status_code == 200:
#                     data = response.json()
#                     return f"GitHub user {username}: {data.get('public_repos', 0)} repos, {data.get('followers', 0)} followers"
#             except Exception:
#                 pass
#             return f"Could not fetch stats for {username}"
        
#         # Create specialized agents (each will be an AGENT span!)
#         researcher = Agent(
#             role="Research Analyst",
#             goal="Find and analyze information about AI topics",
#             backstory="Expert at finding relevant information and insights",
#             tools=[search_web],
#             verbose=True
#         )
        
#         developer = Agent(
#             role="Developer",
#             goal="Analyze GitHub profiles and code repositories",
#             backstory="Experienced developer who can assess code quality",
#             tools=[fetch_github_stats],
#             verbose=True
#         )
        
#         writer = Agent(
#             role="Technical Writer",
#             goal="Create clear technical summaries",
#             backstory="Expert at distilling complex information into clear content",
#             verbose=True
#         )
        
#         # Create tasks (each will be a CHAIN/TASK span!)
#         research_task = Task(
#             description="Research the latest trends in AI agent frameworks",
#             agent=researcher,
#             expected_output="Summary of AI agent framework trends"
#         )
        
#         code_task = Task(
#             description="Check the GitHub profile of 'langchain-ai' organization",
#             agent=developer,
#             expected_output="GitHub statistics and analysis"
#         )
        
#         writing_task = Task(
#             description="Write a brief report combining research and code analysis",
#             agent=writer,
#             expected_output="Technical report",
#             context=[research_task, code_task]
#         )
        
#         # Create crew and run (fully auto-traced!)
#         crew = Crew(
#             agents=[researcher, developer, writer],
#             tasks=[research_task, code_task, writing_task],
#             verbose=True
#         )
        
#         print("\n🤖 Running CrewAI multi-agent crew...")
#         result = crew.kickoff()
        
#         print(f"\n✅ Crew Result:\n{result}")
#         print("\n📊 What got traced (AUTO-CAPTURED):")
#         print("   • AGENT spans - Each agent (Researcher, Developer, Writer)")
#         print("   • TASK spans - Each task execution")
#         print("   • LLM spans - All AI model calls with prompts/completions/tokens")
#         print("   • TOOL spans - search_web, fetch_github_stats")
#         print("   • HTTP span - GitHub API call")
#         print("   • Full agent collaboration flow with parent-child relationships!")
        
#         return result
        
#     except ImportError as e:
#         print(f"⚠️  Skipping CrewAI demo: {e}")
#         print("   Run: poetry add 'crewai[tools]'")
#         return None
#     except Exception as e:
#         print(f"⚠️  Error in CrewAI demo: {e}")
#         return None

# demo_crewai_agents()


# ============================================================================
# EXAMPLE 3: RAG WORKFLOW WITH EMBEDDINGS (AUTO-TRACED!)
# ============================================================================

# print("\n\n" + "=" * 100)
# print("📊 EXAMPLE 3: RAG Workflow with Embeddings & Retrieval")
# print("=" * 100)
# print("✨ Embeddings and retrieval are AUTO-TRACED - see your RAG pipeline!")

# def demo_rag_workflow():
#     """RAG workflow with embeddings and retrieval - fully auto-traced"""
#     try:
#         from langchain_openai import OpenAIEmbeddings, ChatOpenAI
#         from langchain_community.vectorstores import Chroma
#         from langchain_classic.text_splitter import CharacterTextSplitter
#         from langchain_classic.chains import RetrievalQA
#         from langchain_classic.schema import Document
        
#         # Sample documents
#         docs = [
#             Document(page_content="NeatLogs is an AI agent observability platform.", metadata={"source": "doc1"}),
#             Document(page_content="OpenInference is a semantic convention for LLM tracing.", metadata={"source": "doc2"}),
#             Document(page_content="LangChain is a framework for building AI applications.", metadata={"source": "doc3"}),
#             Document(page_content="CrewAI enables multi-agent AI workflows.", metadata={"source": "doc4"}),
#         ]
        
#         # Split documents (CHAIN span)
#         text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
#         splits = text_splitter.split_documents(docs)
        
#         # Create embeddings (EMBEDDING spans - auto-traced!)
#         print("\n📝 Creating embeddings...")
#         embeddings = OpenAIEmbeddings(openai_api_key="")
        
#         # Create vector store (EMBEDDING spans for each document!)
#         vectorstore = Chroma.from_documents(
#             documents=splits,
#             embedding=embeddings,
#             collection_name="neatlogs_demo"
#         )
        
#         # Create retriever (RETRIEVER spans - auto-traced!)
#         retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
        
#         # Create QA chain
#         llm = ChatOpenAI(
#             model="gpt-4o-mini",
#             temperature=0,
#             openai_api_key=""
#         )
#         qa_chain = RetrievalQA.from_chain_type(
#             llm=llm,
#             chain_type="stuff",
#             retriever=retriever,
#             return_source_documents=True
#         )
        
#         # Run RAG query (fully auto-traced: query → embedding → retrieval → LLM!)
#         print("🤖 Running RAG query...")
#         result = qa_chain({"query": "What is NeatLogs?"})
        
#         print(f"\n✅ RAG Result: {result['result']}")
#         print(f"   Sources: {[doc.metadata['source'] for doc in result['source_documents']]}")
#         print("\n📊 What got traced (AUTO-CAPTURED):")
#         print("   • EMBEDDING spans - Query embedding + document embeddings")
#         print("   •   ├─ embedding.model_name: 'text-embedding-ada-002'")
#         print("   •   ├─ embedding.text: Query and document texts")
#         print("   •   └─ embedding.token_count: Token usage")
#         print("   • RETRIEVER span - Vector similarity search")
#         print("   •   ├─ retrieval.query: User query")
#         print("   •   ├─ retrieval.top_k: Number of documents")
#         print("   •   └─ Retrieved documents with scores")
#         print("   • LLM span - Final answer generation with retrieved context")
#         print("   • CHAIN spans - RAG orchestration")
        
#         # Clean up
#         vectorstore.delete_collection()
        
#         return result
        
#     except ImportError as e:
#         print(f"⚠️  Skipping RAG demo: {e}")
#         print("   Run: poetry add langchain-openai langchain-community chromadb")
#         return None
#     except Exception as e:
#         print(f"⚠️  Error in RAG demo: {e}")
#         return None

# demo_rag_workflow()


# ============================================================================
# EXAMPLE 4: CUSTOM DECORATORS ON TOP OF AUTO-TRACING
# ============================================================================

# print("\n\n" + "=" * 100)
# print("📊 EXAMPLE 4: Custom Decorators + Auto-Tracing")
# print("=" * 100)
# print("🎨 Add decorators for YOUR custom orchestration logic")

# @neatlogs.trace(span_kind="CHAIN", name="ai_agent_orchestrator")
# def orchestrate_ai_workflow(user_query: str):
#     """
#     Custom workflow orchestrator - YOUR business logic
#     This creates a parent CHAIN span that contains all auto-traced operations
#     """
#     print(f"\n🎯 Orchestrating workflow for: '{user_query}'")
    
#     # Add custom annotations to the parent span
#     neatlogs.annotate({
#         "workflow_stage": "initialization",
#         "user_query": user_query,
#         "priority": "high"
#     })
    
#     # Step 1: Validate input (your custom function)
#     @neatlogs.trace(span_kind="CHAIN", name="input_validation")
#     def validate_input(query: str):
#         neatlogs.annotate({"validation_status": "passed"})
#         return len(query) > 0
    
#     if not validate_input(user_query):
#         return {"error": "Invalid input"}
    
#     # Step 2: Call LangChain agent (AUTO-TRACED by OpenInference!)
#     try:
#         from langchain_openai import ChatOpenAI
        
#         llm = ChatOpenAI(
#             model="gpt-4o-mini",
#             temperature=0.7,
#             openai_api_key=""
#         )
        
#         # This LLM call is AUTO-TRACED as a child of our custom span!
#         response = llm.invoke(user_query)
        
#         # Step 3: Post-process (your custom function)
#         @neatlogs.trace(span_kind="CHAIN", name="post_processing")
#         def post_process(ai_response):
#             neatlogs.annotate({
#                 "response_length": len(ai_response.content),
#                 "processing_status": "success"
#             })
#             return ai_response.content
        
#         final_result = post_process(response)
        
#         neatlogs.annotate({
#             "workflow_stage": "completed",
#             "success": True
#         })
        
#         print(f"✅ Workflow completed: {final_result[:100]}...")
#         print("\n📊 Trace hierarchy:")
#         print("   CHAIN: ai_agent_orchestrator (your decorator)")
#         print("   ├─ CHAIN: input_validation (your decorator)")
#         print("   ├─ LLM: ChatOpenAI (auto-traced by OpenInference)")
#         print("   │  ├─ prompt_text, completion_text, tokens, cost")
#         print("   │  └─ model parameters (temperature, etc.)")
#         print("   └─ CHAIN: post_processing (your decorator)")
        
#         return {"result": final_result}
        
#     except ImportError:
#         print("⚠️  Skipping (OpenAI not available)")
#         return None
#     except Exception as e:
#         neatlogs.annotate({
#             "workflow_stage": "error",
#             "error": str(e)
#         })
#         print(f"⚠️  Error: {e}")
#         return {"error": str(e)}

# orchestrate_ai_workflow("Explain quantum computing in simple terms")


# ============================================================================
# EXAMPLE 5: TRACK USER FEEDBACK
# ============================================================================

# print("\n\n" + "=" * 100)
# print("📊 EXAMPLE 5: Track User Feedback")
# print("=" * 100)

# # Get current trace for feedback
# trace_id = neatlogs.get_current_trace_id()
# if trace_id:
#     # Simulate user giving positive feedback
#     neatlogs.track_feedback(
#         trace_id=trace_id,
#         rating="positive",
#         score=5,
#         comment="Great AI agent response! Very helpful.",
#         metadata={"feature": "ai_orchestration", "session_length": 120}
#     )
#     print(f"✅ Tracked positive feedback for trace {trace_id}")
# else:
#     print("⚠️  No active trace for feedback")


# ============================================================================
# SUMMARY
# ============================================================================

print("\n\n" + "=" * 100)
print("🎉 COMPLETE AI AGENT TRACING SUMMARY")
print("=" * 100)

print("""
✅ What You Just Saw:

1. LANGCHAIN AGENT (Example 1)
   • Agent with multiple tools (weather, calculator, search)
   • AUTO-TRACED: agents, LLM calls, tool calls, HTTP requests
   • Full conversation flow with reasoning

2. CREWAI MULTI-AGENT (Example 2)
   • Multiple specialized agents collaborating
   • AUTO-TRACED: each agent, tasks, LLM calls, tool calls
   • Agent-to-agent communication and handoffs

3. RAG WORKFLOW (Example 3)
   • Document embeddings, vector search, retrieval, QA
   • AUTO-TRACED: embedding spans, retriever spans, LLM spans
   • Full RAG pipeline visibility

4. CUSTOM ORCHESTRATION (Example 4)
   • Your custom decorators + auto-traced framework calls
   • Parent-child span relationships
   • Custom business logic tracking

5. USER FEEDBACK (Example 5)
   • Link user feedback to specific traces
   • Track satisfaction, errors, improvements

📊 ALL CAPTURED AUTOMATICALLY:
   • OpenInference span types: LLM, CHAIN, AGENT, TOOL, EMBEDDING, RETRIEVER
   • LLM details: prompts, completions, tokens, costs, model parameters
   • Tool executions: inputs, outputs, duration, errors
   • HTTP calls: external API calls from tools
   • Error tracking: types, messages, stacktraces
   • User/session context: user_id, session_id
   • Custom attributes: via @neatlogs.trace() and annotate()

🎯 NO MANUAL INSTRUMENTATION NEEDED!
   Just call neatlogs.init() and use your favorite AI frameworks.
   Everything is automatically traced via OpenInference.

💡 WHEN TO ADD DECORATORS:
   Only for YOUR custom orchestration/business logic.
   Framework operations (LangChain, CrewAI, LlamaIndex) are auto-traced!
""")

print("\n🔍 Check your NeatLogs UI at http://localhost:3000")
print("   • Filter by user_id: 'demo_user'")
print("   • Filter by session_id: 'demo_session_001'")
print("   • See full agent workflows with nested spans")
print("   • View costs, tokens, errors, feedback")

# Final flush
neatlogs.flush()
print("\n✅ All traces flushed to NeatLogs!")
