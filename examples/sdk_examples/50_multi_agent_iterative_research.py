"""
Multi-agent iterative research system with cyclic workflow.
Demonstrates same agent running multiple times with identical prompt templates for deduplication testing.

Prereqs:
  pip install langchain-google-genai langgraph langchain-community
  export GOOGLE_API_KEY=your-key
  export TAVILY_API_KEY=your-key
"""

import json
import os
import sys
from typing import Annotated, TypedDict

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from neatlogs import flush, init, shutdown, trace, PromptTemplate

os.environ["NEATLOGS_LOG_SPANS"] = "true"
os.environ["NEATLOGS_LOG_METRICS"] = "true"
os.environ["NEATLOGS_LOG_RAW_SPANS"] = "true"
os.environ["NEATLOGS_LOG_SPANS_FILE"] = "spans_50.log"
os.environ["NEATLOGS_LOG_RAW_SPANS_FILE"] = "spans_raw_50.log"
os.environ["NEATLOGS_LOG_METRICS_FILE"] = "metrics_50.log"


init(
    api_key=os.getenv("NEATLOGS_API_KEY", "test-key"),
    workflow_name="multi-agent-research",
    instrumentations=["google_genai", "langchain"],
    debug=True,
)


class ResearchState(TypedDict):
    messages: Annotated[list, add_messages]
    question: str
    search_queries: list[str]
    current_query_idx: int
    findings: list[dict]
    iterations: int
    max_iterations: int
    decision: str
    final_answer: str


def main():

    question = "Compare the latest developments in quantum computing, AI chip architectures, and neuromorphic computing in 2024. What are the key breakthroughs in each area and how do they relate to each other?"

    try:
        with trace("iterative_research_workflow", kind="WORKFLOW"):
            llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)
            
            tavily_tool = TavilySearchResults(
                max_results=5,
                search_depth="advanced",
                name="web_search",
                description="Search the web for current information on quantum computing and technology"
            )
            tools = [tavily_tool]
            llm_with_tools = llm.bind_tools(tools)

            def query_planner(state: ResearchState) -> ResearchState:
                """Break down question into search queries"""
                planner_prompt = PromptTemplate(
                    """You are a research query planner. The user has a complex multi-faceted question.
Break it down into EXACTLY 3 specific, focused search queries that together will comprehensively answer the question.

Make each query specific and distinct - avoid overlap.

Return ONLY a valid JSON array of exactly 3 strings, nothing else.

Question: {{question}}

Output format: ["query1", "query2", "query3"]"""
                )

                with trace("query_planner_call", kind="AGENT"):
                    prompt = planner_prompt.compile(question=state["question"])
                    response = llm.invoke([HumanMessage(content=prompt)])

                    try:
                        queries = json.loads(response.content)
                        if not isinstance(queries, list):
                            queries = [
                                "quantum computing breakthroughs 2024",
                                "AI chip architectures developments 2024",
                                "neuromorphic computing advances 2024"
                            ]
                    except:
                        queries = [
                            "quantum computing breakthroughs 2024",
                            "AI chip architectures developments 2024",
                            "neuromorphic computing advances 2024"
                        ]

                    # Ensure exactly 3 queries
                    if len(queries) < 3:
                        queries = queries + [state["question"]] * (3 - len(queries))
                    
                    state["search_queries"] = queries[:3]
                    state["current_query_idx"] = 0
                    state["iterations"] = 0
                    state["max_iterations"] = 5
                    state["findings"] = []
                    return state

            def researcher(state: ResearchState) -> ResearchState:
                """Core research agent - runs multiple times with same prompt template"""
                idx = state["current_query_idx"]
                if idx >= len(state["search_queries"]):
                    state["decision"] = "SYNTHESIZE"
                    return state

                current_query = state["search_queries"][idx]
                accumulated = "\n".join([f["insights"] for f in state["findings"]])

                researcher_prompt = PromptTemplate(
                    """You are a meticulous research analyst. 

Use the web_search tool to search for: {{current_query}}

After getting results, analyze them and extract:
1. Key facts and insights
2. Information gaps

Previous Findings:
{{accumulated}}

Call the web_search tool now with the query."""
                )

                with trace("researcher_call", kind="AGENT"):
                    prompt = researcher_prompt.compile(
                        current_query=current_query,
                        accumulated=accumulated or "None yet",
                    )
                    response = llm_with_tools.invoke([HumanMessage(content=prompt)])
                    return {"messages": [response]}

            def process_research_results(state: ResearchState) -> ResearchState:
                """Process tool results into findings"""
                messages = state.get("messages", [])
                if not messages:
                    return state
                
                last_message = messages[-1]
                
                if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                    return state
                
                idx = state["current_query_idx"]
                if idx >= len(state["search_queries"]):
                    return state
                
                current_query = state["search_queries"][idx]
                
                analysis_prompt = PromptTemplate(
                    """Based on the search results, provide analysis in JSON format:
{{
  "insights": "key findings as single paragraph",
  "gaps": ["missing info1", "missing info2"],
  "confidence": 0-100
}}

Search query was: {{current_query}}
Previous findings: {{accumulated}}

Provide your analysis:"""
                )
                
                accumulated = "\n".join([f["insights"] for f in state["findings"]])
                
                with trace("process_research_results", kind="AGENT"):
                    prompt = analysis_prompt.compile(
                        current_query=current_query,
                        accumulated=accumulated or "None yet"
                    )
                    response = llm.invoke([HumanMessage(content=prompt)])
                    
                    try:
                        analysis = json.loads(response.content)
                    except:
                        analysis = {
                            "insights": response.content,
                            "gaps": [],
                            "confidence": 50,
                        }
                    
                    state["findings"].append({
                        "query": current_query,
                        "insights": analysis.get("insights", ""),
                        "confidence": analysis.get("confidence", 50),
                    })
                    state["current_query_idx"] += 1
                    state["iterations"] += 1
                    
                    return state

            def quality_check(state: ResearchState) -> ResearchState:
                """Evaluate if more research needed"""
                processed = state["current_query_idx"]
                total = len(state["search_queries"])

                all_findings = "\n\n".join(
                    [f"Query: {f['query']}\n{f['insights']}" for f in state["findings"]]
                )

                quality_check_prompt = PromptTemplate(
                    """You are a quality assessor. Review research findings carefully.

Original Question: {{question}}

Findings so far:
{{all_findings}}

Progress: {{processed}}/{{total}} queries processed, {{iterations}}/{{max_iterations}} iterations

Rules:
- If we have processed LESS than {{total}} queries, output "CONTINUE" - we need all perspectives
- Only output "SYNTHESIZE" when we have processed ALL {{total}} queries
- The question has multiple facets - we need comprehensive coverage

Output ONLY: "CONTINUE" or "SYNTHESIZE"

Decision:"""
                )

                with trace("quality_check_call", kind="AGENT"):
                    prompt = quality_check_prompt.compile(
                        question=state["question"],
                        all_findings=all_findings,
                        processed=processed,
                        total=total,
                        iterations=state["iterations"],
                        max_iterations=state["max_iterations"],
                    )
                    response = llm.invoke([HumanMessage(content=prompt)])

                    decision = (
                        "SYNTHESIZE"
                        if "SYNTHESIZE" in response.content.upper()
                        else "CONTINUE"
                    )

                    # Safety: always continue if we haven't processed all queries
                    if processed < total:
                        decision = "CONTINUE"
                    
                    # Safety: always stop if we hit max iterations
                    if state["iterations"] >= state["max_iterations"]:
                        decision = "SYNTHESIZE"

                    state["decision"] = decision
                    return state

            def synthesizer(state: ResearchState) -> ResearchState:
                """Compile final answer"""
                all_findings = "\n\n".join(
                    [
                        f"Finding {i+1} (from: {f['query']}):\n{f['insights']}"
                        for i, f in enumerate(state["findings"])
                    ]
                )

                synthesizer_prompt = PromptTemplate(
                    """You are a synthesis expert. Compile research findings into a comprehensive, well-structured answer with clear sections.

Question: {{question}}

All Research Findings:
{{all_findings}}

Provide a comprehensive answer:"""
                )

                with trace("synthesizer_call", kind="AGENT"):
                    prompt = synthesizer_prompt.compile(
                        question=state["question"], all_findings=all_findings
                    )
                    response = llm.invoke([HumanMessage(content=prompt)])

                    state["final_answer"] = response.content
                    return state

            def route_after_quality_check(state: ResearchState) -> str:
                """Route based on quality check decision"""
                if state["decision"] == "CONTINUE":
                    return "researcher"
                return "synthesizer"

            workflow = StateGraph(ResearchState)

            workflow.add_node("query_planner", query_planner)
            workflow.add_node("researcher", researcher)
            workflow.add_node("tools", ToolNode(tools))
            workflow.add_node("process_results", process_research_results)
            workflow.add_node("quality_check", quality_check)
            workflow.add_node("synthesizer", synthesizer)

            workflow.set_entry_point("query_planner")
            workflow.add_edge("query_planner", "researcher")
            
            workflow.add_conditional_edges(
                "researcher",
                tools_condition,
                {
                    "tools": "tools",
                    END: "process_results"
                }
            )
            workflow.add_edge("tools", "process_results")
            workflow.add_edge("process_results", "quality_check")
            workflow.add_conditional_edges(
                "quality_check", route_after_quality_check, ["researcher", "synthesizer"]
            )
            workflow.add_edge("synthesizer", END)

            app = workflow.compile()

            result = app.invoke(
                {
                    "messages": [],
                    "question": question,
                    "search_queries": [],
                    "current_query_idx": 0,
                    "findings": [],
                    "iterations": 0,
                    "max_iterations": 5,
                    "decision": "",
                    "final_answer": "",
                }
            )

            print("\n" + "=" * 80)
            print("RESEARCH COMPLETE")
            print("=" * 80)
            print(f"\nQuestion: {question}")
            print(f"\nQueries Executed: {len(result['findings'])}")
            print(f"Total Iterations: {result['iterations']}")
            print(f"\nFinal Answer:\n{result['final_answer']}")

            flush()

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        shutdown()


if __name__ == "__main__":
    main()
