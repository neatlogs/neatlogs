from langgraph.graph import StateGraph, START, END
import neatlogs
from langgraph.graph.message import add_messages
from langgraph.graph.message import add_messages
from langchain_openai import AzureChatOpenAI
from langchain_core.runnables import RunnableConfig

from typing import TypedDict
import os
from dotenv import load_dotenv

load_dotenv()


neatlogs.init(api_key=os.getenv('NEATLOGS_API_KEY'), tags=[
    "v3", "langchain", "demo"], instrumentations=["langchain"], debug=True)

model = AzureChatOpenAI(api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
                        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),)


class GraphState(TypedDict):
    query: str
    analysis: str
    status: str


def analyst_node(state: GraphState):
    print(f"   [Node] Analyst working on: {state['query']}")
    # Using model instance inside node for realistic tracing
    response = model.invoke(state['query'])
    return {"analysis": response.content, "status": "analyzed"}


# def test_langgraph_fstring():
#     print("\n>>> Scenario 9: LangGraph with Python F-Strings")
#     topic = "Dark Matter"
#     # Manual rendering before passing to graph
#     rendered_input = f"Analyze {topic}"

#     workflow = StateGraph(GraphState)
#     workflow.add_node("analyst", analyst_node)
#     workflow.add_edge(START, "analyst")
#     workflow.add_edge("analyst", END)
#     app = workflow.compile()

#     # Graph only sees 'query' key, doesn't know 'topic' was a variable
#     app.invoke({"query": rendered_input})


# test_langgraph_fstring()


# def test_langgraph_flow():
#     print("\n>>> Scenario 4: LangGraph State Flow")
#     workflow = StateGraph(GraphState)
#     workflow.add_node("analyst", analyst_node)
#     workflow.add_edge(START, "analyst")
#     workflow.add_edge("analyst", END)
#     app = workflow.compile()

#     # Neatlogs should capture 'query' in initial state and 'analysis' in output state
#     app.invoke({"query": "Market trends in 2025", "status": "new"})


# test_langgraph_flow()


def test_langgraph_configurable():
    print("\n>>> Scenario 5: LangGraph Configurable Parameters")

    def config_node(state: GraphState, config: RunnableConfig):
        user_id = config.get("configurable", {}).get("user_id", "unknown")
        print(f"   [Node] Config node for user: {user_id}")
        response = model.invoke(
            f"Processing request for user {user_id}: {state['query']}")
        print(response.content)
        return {"status": f"processed_for_{user_id}", "analysis": response.content}

    workflow = StateGraph(GraphState)
    workflow.add_node("config_node", config_node)
    workflow.add_edge(START, "config_node")
    workflow.add_edge("config_node", END)
    app = workflow.compile()

    # Neatlogs should capture the configurable user_id
    app.invoke(
        {"query": "test", "status": "start"},
        config={"configurable": {"user_id": "user_123"}}
    )


test_langgraph_configurable()
