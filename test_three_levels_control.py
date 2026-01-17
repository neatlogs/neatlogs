"""
NeatLogs SDK - Three Levels of Control Demo
============================================

This script demonstrates the three levels of control in NeatLogs:
1. Auto-Capture: Zero code changes, full observability
2. Enhanced Capture: Optional decorators for custom functions
3. Granular Control: Fine-tune what gets collected
"""

import os
import sys
from dotenv import load_dotenv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

import neatlogs

load_dotenv()

print("=" * 100)
print("🎯 NeatLogs SDK - Three Levels of Control")
print("=" * 100)

# ============================================================================
# LEVEL 1: AUTO-CAPTURE (Zero Code Changes)
# ============================================================================

print("\n📊 LEVEL 1: AUTO-CAPTURE")
print("-" * 100)
print("✨ Just call neatlogs.init() and everything is auto-captured!")
print()

neatlogs.init(
    api_key=os.getenv("NEATLOGS_API_KEY", "test-key"),
    workflow_name="three-levels-demo",
    user_id="demo_user",
    session_id="demo_session",
    tags=["demo", "control-levels"],
    
    # Global control switches
    enable_http_tracing=True,      # Auto-capture HTTP calls
    disable_content=False,          # Capture LLM prompts/completions
    
    # Selective framework instrumentation
    instrumentations=["openai", "langchain", "crewai"],  # Only these frameworks
)

print("✅ Auto-capture enabled for:")
print("   • All LLM calls (OpenAI, Anthropic, etc.)")
print("   • All framework operations (LangChain, CrewAI, etc.)")
print("   • All vector DB operations (Chroma, Pinecone, etc.)")
print("   • HTTP client calls (requests, httpx, aiohttp)")
print()
print("👉 No decorators needed! Everything just works.")

# LangChain agent - automatically traced!
def run_langchain_agent_auto_traced():
    """LangChain agent with tools - fully auto-traced by OpenInference"""
    try:
        from langchain_openai import ChatOpenAI
        from langchain_classic.tools import tool
        from langchain_classic.agents import AgentExecutor, create_openai_functions_agent
        from langchain_classic import hub
        
        @tool
        def get_time() -> str:
            """Get current time"""
            import datetime
            return f"Current time: {datetime.datetime.now().strftime('%H:%M:%S')}"
        
        llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=os.getenv("OPENAI_API_KEY"))
        prompt = hub.pull("hwchase17/openai-functions-agent")
        agent = create_openai_functions_agent(llm, [get_time], prompt)
        executor = AgentExecutor(agent=agent, tools=[get_time], verbose=False)
        
        result = executor.invoke({"input": "What time is it?"})
        return result['output']
    except Exception as e:
        return f"(Skipped: {e})"

print("\n🤖 Running LangChain agent (auto-traced)...")
result = run_langchain_agent_auto_traced()
print(f"   Agent response: {result}")
print("   ✨ Captured: AGENT span, LLM spans, TOOL span - all automatic!")


# ============================================================================
# LEVEL 2: ENHANCED CAPTURE (Optional Decorators)
# ============================================================================

print("\n\n📊 LEVEL 2: ENHANCED CAPTURE (Optional Decorators)")
print("-" * 100)
print("🎨 Add decorators to YOUR custom functions for better visibility")
print()

# Your custom AI agent orchestration logic - add decorator to trace it
@neatlogs.trace(span_kind="CHAIN", name="ai_workflow_orchestrator")
def orchestrate_ai_agents(user_query: str):
    """
    Your custom AI workflow orchestrator - needs decorator to trace
    The LangChain/CrewAI calls inside are auto-traced!
    """
    print(f"   Orchestrating AI workflow for: {user_query}")
    
    # Add custom metadata to your orchestration span
    neatlogs.annotate({
        "workflow_type": "multi_agent",
        "priority": "high",
        "user_query": user_query
    })
    
    # Your business logic here
    # (LangChain/CrewAI operations inside will be auto-traced as children)
    result = {
        "status": "orchestrated",
        "agents_triggered": ["research_agent", "analysis_agent"],
        "query": user_query
    }
    return result

print("✅ Using @neatlogs.trace() on AI workflow orchestrator")
result = orchestrate_ai_agents("Analyze Q4 financial trends")
print(f"   Result: {result}")
print("   💡 This creates a parent CHAIN span containing auto-traced agent operations")


# Custom RAG retrieval logic with decorator
@neatlogs.trace(span_kind="RETRIEVER", name="custom_document_retriever")
def retrieve_relevant_docs(query: str):
    """
    Custom document retrieval logic - marked as RETRIEVER span
    If you use LangChain/LlamaIndex retrieval inside, it's auto-traced!
    """
    # Your custom retrieval logic
    neatlogs.annotate({
        "retrieval.query": query,
        "retrieval.top_k": 5,
        "db_type": "custom_vector_db"
    })
    
    docs = ["Doc 1", "Doc 2", "Doc 3"]
    return docs

print("\n✅ Using span_kind='RETRIEVER' for custom RAG component")
docs = retrieve_relevant_docs("AI agent frameworks")
print(f"   Retrieved: {len(docs)} documents")
print("   💡 Categorized as RETRIEVER span for RAG analytics")


# ============================================================================
# LEVEL 3: GRANULAR CONTROL
# ============================================================================

print("\n\n📊 LEVEL 3: GRANULAR CONTROL")
print("-" * 100)
print("🎛️  Fine-tune exactly what gets collected")
print()

# 3A: Disable tracing for specific function
@neatlogs.trace(enabled=False)
def internal_helper_not_traced():
    """This won't be traced - useful for high-volume internal functions"""
    return "helper result"

print("🔕 Function with enabled=False (not traced):")
internal_helper_not_traced()
print("   ✓ Executed but no span created")


# 3B: Privacy - don't capture sensitive LLM prompts
@neatlogs.trace(span_kind="CHAIN", capture_input=False, capture_output=False)
def process_sensitive_ai_query(user_query: str, personal_data: dict):
    """
    Traced but inputs/outputs NOT captured (privacy)
    Perfect for: PII data, medical records, financial info in prompts
    """
    print(f"   Processing sensitive AI query with PII...")
    # Your LLM call with sensitive data
    # The LLM call IS traced (tokens, cost, model) but prompts/completions are NOT
    return "AI response (not captured)"

print("\n🔒 Privacy mode for sensitive AI queries:")
process_sensitive_ai_query(
    "Analyze this patient record: John Doe, SSN 123-45-6789",
    {"credit_card": "4111-1111-1111-1111"}
)
print("   ✓ Trace exists but sensitive prompts/data NOT captured")
print("   ✓ Still captures: tokens, cost, duration, model, errors")


# 3C: Sampling - only trace 10% of high-volume LLM calls
@neatlogs.trace(span_kind="CHAIN", sample_rate=0.1)
def auto_classify_email(email_text: str):
    """
    Only 10% of calls are traced - for high-volume operations
    Perfect for: email classification, spam detection, auto-tagging
    """
    # Your LLM-based classification
    return "spam" if "lottery" in email_text.lower() else "inbox"

print("\n📊 Sampling high-volume AI operations (sample_rate=0.1):")
for i in range(10):
    auto_classify_email(f"Email {i}: Regular message")
print("   ✓ Ran 10 times, only ~1 should be traced (10% sample)")
print("   ✓ Reduces observability cost for high-volume endpoints")


# 3D: Global disable/enable
print("\n🌐 Global tracing control:")
print("   ⚡ All tracing enabled (current state)")

# Temporarily disable ALL tracing
neatlogs.disable_tracing()
print("   🔕 neatlogs.disable_tracing() called")

@neatlogs.trace()
def wont_be_traced():
    return "not traced"

wont_be_traced()
# Even decorated functions won't be traced!
print("   ✓ Even @neatlogs.trace() functions are NOT traced")

# Re-enable tracing
neatlogs.enable_tracing()
print("   🔔 neatlogs.enable_tracing() called")

@neatlogs.trace()
def now_traced_again():
    return "traced"

now_traced_again()
print("   ✓ Tracing resumed")


# 3E: Dynamic context updates
print("\n🔄 Runtime context updates:")
neatlogs.set_user("user_456")
print("   ✓ Updated user to user_456")

neatlogs.set_session("new_session_789")
print("   ✓ Updated session to new_session_789")

neatlogs.set_context(
    user_id="user_final",
    metadata={"tier": "premium", "region": "us-east"}
)
print("   ✓ Set custom metadata")


# ============================================================================
# SUMMARY
# ============================================================================

print("\n\n" + "=" * 100)
print("📋 SUMMARY: Three Levels of Control")
print("=" * 100)

print("""
┌─────────────────────────────────────────────────────────────────────────────┐
│ LEVEL 1: AUTO-CAPTURE (Zero Code Changes) 🎯 FOR AI AGENTS                 │
├─────────────────────────────────────────────────────────────────────────────┤
│ • Just call neatlogs.init()                                                 │
│ • ALL AI FRAMEWORKS AUTO-TRACED via OpenInference:                         │
│   ✓ LangChain: agents, chains, tools, prompts, LLM calls                   │
│   ✓ CrewAI: crews, agents, tasks, tool executions                          │
│   ✓ LlamaIndex: query engines, retrievers, embeddings                      │
│   ✓ OpenAI/Anthropic: direct SDK calls                                     │
│   ✓ Vector DBs: Chroma, Pinecone, Weaviate queries                         │
│   ✓ HTTP calls: external API calls from tools                              │
│ • NO decorators needed for framework operations!                           │
│                                                                             │
│ Global Controls (via init):                                                │
│   - enable_http_tracing=True/False   # Auto-trace HTTP tool calls          │
│   - disable_content=True/False       # Privacy: skip prompts/completions   │
│   - instrumentations=["openai"]      # Select specific frameworks          │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ LEVEL 2: ENHANCED CAPTURE (Optional Decorators) 🎨 FOR CUSTOM LOGIC        │
├─────────────────────────────────────────────────────────────────────────────┤
│ Use decorators ONLY for YOUR custom orchestration/business logic:          │
│                                                                             │
│   @neatlogs.trace(span_kind="CHAIN")          # Workflow orchestrator      │
│   @neatlogs.trace(span_kind="RETRIEVER")      # Custom retrieval logic     │
│   @neatlogs.trace(span_kind="TOOL")           # Custom tool wrapper        │
│   @neatlogs.trace(name="rag_pipeline")        # Custom span name           │
│                                                                             │
│ When to use decorators:                                                    │
│   ✓ Your AI workflow orchestrators                                         │
│   ✓ Custom RAG retrieval logic                                             │
│   ✓ Custom tool wrappers/integrations                                      │
│   ✓ Business logic around AI operations                                    │
│                                                                             │
│ When NOT to use decorators:                                                │
│   ✗ LangChain agents/chains (already auto-traced!)                         │
│   ✗ CrewAI crews/tasks (already auto-traced!)                              │
│   ✗ OpenAI/Anthropic calls (already auto-traced!)                          │
│   ✗ Vector DB operations (already auto-traced!)                            │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ LEVEL 3: GRANULAR CONTROL 🎛️  FOR PRIVACY & COST                           │
├─────────────────────────────────────────────────────────────────────────────┤
│ Fine-tune data collection for AI operations:                                │
│                                                                             │
│   @neatlogs.trace(capture_input=False)        # Privacy: skip PII prompts  │
│   @neatlogs.trace(capture_output=False)       # Privacy: skip AI responses │
│   @neatlogs.trace(sample_rate=0.1)            # Sample high-volume calls   │
│   @neatlogs.trace(enabled=False)              # Don't trace this function  │
│                                                                             │
│ Use cases:                                                                 │
│   • capture_input=False: Sensitive prompts (PII, medical, financial)       │
│   • capture_output=False: Sensitive AI responses                           │
│   • sample_rate=0.1: High-volume classification/tagging endpoints          │
│   • enabled=False: Internal helpers that clutter traces                    │
│                                                                             │
│ Runtime controls:                                                          │
│   neatlogs.disable_tracing()                  # Pause ALL tracing          │
│   neatlogs.enable_tracing()                   # Resume tracing             │
│   neatlogs.set_user("user_id")                # Update user context        │
│   neatlogs.set_session("session_id")          # Update session context     │
│   neatlogs.annotate({"key": "value"})         # Add span attributes        │
└─────────────────────────────────────────────────────────────────────────────┘

💡 KEY INSIGHT:
   • 95% of AI agent operations are AUTO-TRACED by OpenInference
   • Decorators are ONLY for YOUR custom orchestration/business logic
   • Use granular controls for privacy, sampling, and cost optimization
""")

print("✅ All three levels demonstrated!")
print("🔍 Check your NeatLogs UI at http://localhost:3000")

# Clean up
neatlogs.flush()
