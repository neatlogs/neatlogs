"""
Example 4: LangGraph State Machine Agent

LangGraph uses state machines for complex agent workflows.
Shows how context propagation works across graph nodes.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from neatlogs.sdk.neatlogs_sdk_v4_langfuse import init, trace, flush, shutdown
from typing import TypedDict

# LangGraph imports
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage


# Define state
class AgentState(TypedDict):
    messages: list
    next_action: str
    result: str


def main():
    # Enable span logging
    os.environ['NEATLOGS_LOG_SPANS'] = 'true'
    
    # Initialize
    init(
        api_key=os.getenv("NEATLOGS_API_KEY", "test-key"),
        workflow_name="langgraph-agent",
        instrumentations=["openai", "langchain"],
        debug=True,
    )
    
    query = "What is the capital of France?"
    
    try:
        with trace(
            "langgraph_workflow",
            prompt_template="User query: {query}",
            prompt_variables={"query": query}
        ):
            llm = ChatOpenAI(model="gpt-4o-mini")
            
            # Define graph nodes - auto-instrumented by LangGraph!
            def analyze_query(state: AgentState) -> AgentState:
                """Analyze the user query."""
                messages = [
                    SystemMessage(content="Analyze the user query and decide next action."),
                    HumanMessage(content=state["messages"][0])
                ]
                response = llm.invoke(messages)
                
                state["next_action"] = "search" if "search" in response.content.lower() else "answer"
                return state
            
            def search_information(state: AgentState) -> AgentState:
                """Search for information."""
                # Simulated search - in real app would use HTTP
                result = "France is a country in Europe. Capital: Paris."
                state["result"] = result
                return state
            
            def generate_answer(state: AgentState) -> AgentState:
                """Generate final answer."""
                context = state.get("result", "")
                messages = [
                    SystemMessage(content=f"Answer based on context: {context}"),
                    HumanMessage(content=state["messages"][0])
                ]
                response = llm.invoke(messages)
                state["result"] = response.content
                return state
            
            # Build graph
            workflow = StateGraph(AgentState)
            
            workflow.add_node("analyze", analyze_query)
            workflow.add_node("search", search_information)
            workflow.add_node("answer", generate_answer)
            
            workflow.set_entry_point("analyze")
            workflow.add_conditional_edges(
                "analyze",
                lambda state: state["next_action"],
                {"search": "search", "answer": "answer"}
            )
            workflow.add_edge("search", "answer")
            workflow.add_edge("answer", END)
            
            app = workflow.compile()
            
            # Run the graph
            result = app.invoke({
                "messages": [query],
                "next_action": "",
                "result": ""
            })
            
            print(f"\nFinal Result: {result['result']}")
    except Exception as e:
        print(f"\nError during LangGraph execution: {e}")
    finally:
    
        print("\n💾 Flushing spans...")
        flush()
        print("🛑 Shutting down SDK...")
        shutdown()
        print("✅ Done!")


if __name__ == "__main__":
    main()
