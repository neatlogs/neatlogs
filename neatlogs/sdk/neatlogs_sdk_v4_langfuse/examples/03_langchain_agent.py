"""
Example 3: LangChain Agent with Tools

This example shows dual instrumentation with LangChain.
Both OpenInference and OpenLLMetry instrument LangChain, giving us:
- OpenInference: Span kinds (AGENT, TOOL, LLM), cost tracking
- OpenLLMetry: Streaming info, operational metrics
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from neatlogs.sdk.neatlogs_sdk_v4_langfuse import init, trace, flush, shutdown

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_classic import hub  # In LangChain v1, hub is in langchain_classic
from langchain_core.tools import tool



# Define custom tools
@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    # In real app, this would make an HTTP call
    # HTTP span will be a child of this TOOL span!
    return f"The weather in {city} is sunny and 72°F"


@tool
def get_population(city: str) -> str:
    """Get the population of a city."""
    return f"The population of {city} is approximately 800,000"


def main():
    # Enable span logging to file for debugging
    os.environ['NEATLOGS_LOG_SPANS'] = 'true'
    
    # Initialize with explicit library instrumentation
    init(
        api_key="EQZO8zBsJkDRvmIL06m-EQ7faMUHEIN5",
        workflow_name="langchain-agent",
        
        # Explicitly specify only the libraries you're using
        instrumentations=["openai", "langchain"],
        # This is cleaner than tags and only checks what you need
        
        debug=True,
    )
    
    # Run agent with prompt tracking
    query = "What's the weather like in San Francisco and what is its population?"
    
    try:
        with trace(
            "langchain_agent_workflow",
            prompt_template="User query: {query}",
            prompt_variables={"query": query},
            version="v1.0"
        ):
            # Setup: Create LangChain agent
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
            tools = [get_weather, get_population]
            
            # Get prompt template from LangChain hub (makes HTTP calls)
            prompt = hub.pull("hwchase17/react")
            
            # Create agent with tools
            agent = create_react_agent(llm, tools, prompt)
            agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
            
            # Run agent - all spans are automatically traced!
            result = agent_executor.invoke({"input": query})
        
        print(f"\nResult: {result['output']}")
    except Exception as e:
        print(f"\n❌ Error running agent: {e}")
        import traceback
        traceback.print_exc()
    
    # What gets traced:
    # - CHAIN span (langchain_agent_workflow) - root span
    #   └─ HTTP spans (hub.pull calls to LangChain hub)
    #   └─ AGENT span (langchain agent execution)
    #      └─ LLM span (reasoning)
    #      └─ TOOL span (get_weather)
    #      └─ LLM span (reasoning)
    #      └─ TOOL span (get_population)
    #      └─ LLM span (final answer)
    #
    # All HTTP calls are properly parented under their context!
    # Each span has merged attributes from both conventions:
    # - Token counts (from both)
    # - Cost (OpenInference or calculated fallback)
    # - Streaming info (OpenLLMetry)
    # - Span kinds (OpenInference)
    
    # Flush and shutdown to ensure all spans are exported
    print("\n💾 Flushing spans...")
    flush()
    
    print("🛑 Shutting down SDK...")
    shutdown()
    
    print("✅ Done! Check spans.log for trace data.")


if __name__ == "__main__":
    main()
