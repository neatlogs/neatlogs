import os
import sys
import time
import logging
from typing import TypedDict, Annotated, List, Dict
from langchain_core.runnables import RunnableConfig
from uuid import uuid4

# --- ENVIRONMENT HOTFIXES ---
# 1. Fix aiohttp/litellm timeout attributes
try:
    import aiohttp
    if not hasattr(aiohttp, "ConnectionTimeoutError"):
        class ConnectionTimeoutError(Exception): pass
        aiohttp.ConnectionTimeoutError = ConnectionTimeoutError
    if not hasattr(aiohttp, "SocketTimeoutError"):
        class SocketTimeoutError(Exception): pass
        aiohttp.SocketTimeoutError = SocketTimeoutError
except ImportError:
    pass

# 2. Force local neatlogs prioritization
local_path = os.path.join(os.getcwd(), "neatlogs")
if local_path not in sys.path:
    sys.path.insert(0, local_path)

# --- INITIALIZE NEATLOGS ---
import neatlogs


# --- LANGCHAIN SCENARIOS ---
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

model = ChatOpenAI(model="gpt-4o-mini")

def test_langchain_standard():
    print("\n>>> Scenario 1: LangChain Standard Invoke")
    prompt = ChatPromptTemplate.from_template("Explain {topic} in one sentence.")
    chain = prompt | model | StrOutputParser()
    # Neatlogs should capture 'topic' variable
    chain.invoke({"topic": "Quantum Computing"})

def test_langchain_partial():
    print("\n>>> Scenario 2: LangChain Partialing")
    prompt = ChatPromptTemplate.from_template("As a {role}, tell me a joke about {subject}.")
    partial_prompt = prompt.partial(role="Robot")
    chain = partial_prompt | model | StrOutputParser()
    # Neatlogs should capture 'subject' variable
    chain.invoke({"subject": "humans"})

def test_langchain_messages():
    print("\n>>> Scenario 3: LangChain Message Placeholders")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant for {company}."),
        ("user", "{query}")
    ])
    chain = prompt | model | StrOutputParser()
    # Neatlogs should capture 'company' and 'query'
    chain.invoke({"company": "Neatlogs", "query": "How do variables work?"})

# --- LANGCHAIN MODEL DIRECT SCENARIOS ---

def test_langchain_model_template():
    print("\n>>> Scenario 3.1: LangChain model.invoke with Template (Direct)")
    prompt = ChatPromptTemplate.from_template("Explain {topic}")
    try:
        # Most models can't handle the template object directly, but let's test capture
        model.invoke(prompt)
    except Exception as e:
        print(f"   [EXPECTED ERROR] {type(e).__name__}: {e}")

def test_langchain_model_prompt_value():
    print("\n>>> Scenario 3.2: LangChain model.invoke with PromptValue")
    prompt = ChatPromptTemplate.from_template("Explain {topic}")
    prompt_value = prompt.invoke({"topic": "Dark Matter"})
    model.invoke(prompt_value)

def test_langchain_model_partial():
    print("\n>>> Scenario 3.3: LangChain model.invoke with Partial Template")
    prompt = ChatPromptTemplate.from_template("As a {role}, tell me about {topic}")
    partial_prompt = prompt.partial(role="Scientist")
    prompt_value = partial_prompt.invoke({"topic": "Evolution"})
    model.invoke(prompt_value)

def test_langchain_model_messages():
    print("\n>>> Scenario 3.4: LangChain model.invoke with direct Messages")
    from langchain_core.messages import HumanMessage, SystemMessage
    messages = [
        SystemMessage(content="You are a helpful assistant"),
        HumanMessage(content="What is Neatlogs?")
    ]
    model.invoke(messages)

def test_langchain_model_invoke_string():
    print("\n>>> Scenario 12: LangChain model.invoke with STRING (F-String variant)")
    topic = "Quantum Gravity"
    model.invoke(f"Tell me about {topic}")

# --- LANGGRAPH SCENARIOS ---
from langgraph.graph import StateGraph, START, END

class GraphState(TypedDict):
    query: str
    analysis: str
    status: str

def analyst_node(state: GraphState):
    print(f"   [Node] Analyst working on: {state['query']}")
    # Using model instance inside node for realistic tracing
    response = model.invoke(state['query'])
    return {"analysis": response.content, "status": "analyzed"}

def test_langgraph_flow():
    print("\n>>> Scenario 4: LangGraph State Flow")
    workflow = StateGraph(GraphState)
    workflow.add_node("analyst", analyst_node)
    workflow.add_edge(START, "analyst")
    workflow.add_edge("analyst", END)
    app = workflow.compile()
    
    # Neatlogs should capture 'query' in initial state and 'analysis' in output state
    app.invoke({"query": "Market trends in 2025", "status": "new"})

def test_langgraph_configurable():
    print("\n>>> Scenario 5: LangGraph Configurable Parameters")
    def config_node(state: GraphState, config: Dict):
        user_id = config.get("configurable", {}).get("user_id", "unknown")
        print(f"   [Node] Config node for user: {user_id}")
        # Using model instance inside node for realistic tracing
        model.invoke(f"Processing request for user {user_id}: {state['query']}")
        return {"status": f"processed_for_{user_id}"}

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

# --- CREWAI SCENARIOS ---
from crewai import Agent, Task, Crew

def test_crewai_kickoff():
    print("\n>>> Scenario 6: CrewAI Kickoff Inputs")
    researcher = Agent(
        role='Researcher',
        goal='Find info about {topic}',
        backstory='Expert researcher',
        llm=model
    )
    task = Task(
        description='Research the {topic} and provide 3 key points.',
        agent=researcher,
        expected_output='3 bullet points'
    )
    crew = Crew(agents=[researcher], tasks=[task])
    
    # Neatlogs should capture 'topic' from kickoff inputs
    crew.kickoff(inputs={'topic': 'Space Exploration'})

def test_crewai_context_flow():
    print("\n>>> Scenario 7: CrewAI Context/Output Flow")
    writer = Agent(role='Writer', goal='Write a post', backstory='Pro writer', llm=model)
    
    task1 = Task(description='Research {subject}', agent=writer, expected_output='Short summary')
    task2 = Task(description='Write a tweet based on the summary', agent=writer, expected_output='One tweet', context=[task1])
    
    crew = Crew(agents=[writer], tasks=[task1, task2])
    # Neatlogs should capture 'subject' and the flow of data from task1 to task2
    crew.kickoff(inputs={'subject': 'AI Observability'})

# --- F-STRING SCENARIOS (Observability Failure Cases) ---

def test_langchain_fstring():
    print("\n>>> Scenario 8: LangChain with Python F-Strings")
    topic = "Superconductivity"
    # Manual rendering - variable 'topic' is lost to framework
    rendered_prompt = f"Explain {topic} in one sentence."
    # LangChain just sees a static string
    model.invoke(rendered_prompt)

def test_langgraph_fstring():
    print("\n>>> Scenario 9: LangGraph with Python F-Strings")
    topic = "Dark Matter"
    # Manual rendering before passing to graph
    rendered_input = f"Analyze {topic}"
    
    workflow = StateGraph(GraphState)
    workflow.add_node("analyst", analyst_node)
    workflow.add_edge(START, "analyst")
    workflow.add_edge("analyst", END)
    app = workflow.compile()
    
    # Graph only sees 'query' key, doesn't know 'topic' was a variable
    app.invoke({"query": rendered_input})

def test_crewai_fstring():
    print("\n>>> Scenario 10: CrewAI with Python F-Strings")
    target = "Mars Rover"
    # Manual rendering in Task description
    rendered_desc = f"Research the {target} mission."
    
    agent = Agent(role='Researcher', goal='Research', backstory='Researcher', llm=model)
    task = Task(description=rendered_desc, agent=agent, expected_output='Summary')
    crew = Crew(agents=[agent], tasks=[task])
    
    # Kickoff without inputs dict - variables are hardcoded strings now
    crew.kickoff()

if __name__ == "__main__":
    print("Starting Comprehensive Variable Capture Test...")
    
    # LangChain
    # handler = NeatlogsLangchainCallbackHandler(api_key="kL-zN954K3s_lz4FMy9-p0QLqI5S4zFK")
    # def test_langchain_model_template():
    #     print("\n>>> Scenario 3.1: LangChain model.invoke with Template (Direct)")
    #     prompt = ChatPromptTemplate.from_template("Explain {topic}")
    
    #     # Format the template into a PromptValue first
    #     prompt_value = prompt.invoke({"topic": "Dark Matter"}, config={"callbacks": [handler]})
        
    #     # Now this will work
    #     model.invoke(prompt_value, config={"callbacks": [handler]})
    # test_langchain_model_template()

    # def test_langchain_model_prompt_value():
    #     print("\n>>> Scenario 3.2: LangChain model.invoke with PromptValue")
    #     prompt = ChatPromptTemplate.from_template("Explain {topic}")
    #     prompt_value = prompt.invoke({"topic": "Dark Matter"}, config={"callbacks": [handler]})
    #     model.invoke(prompt_value, config={"callbacks": [handler]})

    # test_langchain_model_prompt_value()

    # def test_langchain_model_partial():
    #     print("\n>>> Scenario 3.3: LangChain model.invoke with Partial Template")
    #     prompt = ChatPromptTemplate.from_template("As a {role}, tell me about {topic}")
    #     partial_prompt = prompt.partial(role="Scientist")
    #     prompt_value = partial_prompt.invoke({"topic": "Evolution"}, config={"callbacks": [handler]})
    #     model.invoke(prompt_value)

    # test_langchain_model_partial()

    # def test_langchain_model_messages_structured():
    #     print("\n>>> Scenario 3.4a: Messages with Structured Variables (Capturable)")
    #     # We define placeholders in the template
    #     prompt = ChatPromptTemplate.from_messages([
    #         ("system", "You are a helpful assistant for {company}"),
    #         ("user", "Explain {topic}")
    #     ])
        
    #     # We pass the variables as a dictionary
    #     prompt_value = prompt.invoke({"company": "Neatlogs", "topic": "V3 Architecture"}, config={"callbacks": [handler]})
        
    #     # Neatlogs captures 'company' and 'topic'
    #     model.invoke(prompt_value, config={"callbacks": [handler]})

    # test_langchain_model_messages_structured()

    # def test_langchain_model_messages_fstring():
    #     print("\n>>> Scenario 3.4b: Messages with F-Strings (Invisible to SDK)")
    #     from langchain_core.messages import HumanMessage, SystemMessage
    #     company = "Neatlogs"
    #     topic = "V3 Architecture"
        
    #     # Python renders the f-string into a static string immediately
    #     messages = [
    #         SystemMessage(content=f"You are a helpful assistant for {company}"),
    #         HumanMessage(content=f"Explain {topic}")
    #     ]
        
    #     # To the SDK, this looks like a list of hardcoded text with no variables
    #     model.invoke(messages, config={"callbacks": [handler]})
    # test_langchain_model_messages_fstring()

    # def test_langchain_model_invoke_string():
    #     print("\n>>> Scenario 12: LangChain model.invoke with STRING (F-String variant)")
    #     topic = "Quantum Gravity"
    #     model.invoke(f"Tell me about {topic}", config={"callbacks": [handler]})
    # test_langchain_model_invoke_string()
    # def test_langchain_fstring():
    #     print("\n>>> Scenario 8: LangChain with Python F-Strings")
    #     topic = "Superconductivity"
    #     # Manual rendering - variable 'topic' is lost to framework
    #     rendered_prompt = f"Explain {topic} in one sentence."
    #     # LangChain just sees a static string
    #     model.invoke(rendered_prompt, config={"callbacks": [handler]})
    # test_langchain_fstring()
    # def test_langchain_standard():
    #     print("\n>>> Scenario 1: LangChain Standard Invoke")
    #     prompt = ChatPromptTemplate.from_template("Explain {topic} in one sentence.")
    #     chain = prompt | model | StrOutputParser()
    #     # Neatlogs should capture 'topic' variable
    #     chain.invoke({"topic": "Quantum Computing"}, config={"callbacks": [handler]})
    # test_langchain_standard()
    # def test_langchain_partial():
    #     print("\n>>> Scenario 2: LangChain Partialing")
    #     prompt = ChatPromptTemplate.from_template("As a {role}, tell me a joke about {subject}.")
    #     partial_prompt = prompt.partial(role="Robot")
    #     chain = partial_prompt | model | StrOutputParser()
    #     chain.invoke({"subject": "humans"}, config={"callbacks": [handler]})
    # test_langchain_partial()
    # def test_langchain_messages():
    #     print("\n>>> Scenario 3: LangChain Message Placeholders")
    #     prompt = ChatPromptTemplate.from_messages([
    #         ("system", "You are a helpful assistant for {company}."),
    #         ("user", "{query}")
    #     ])
    #     chain = prompt | model | StrOutputParser()
    #     # Neatlogs should capture 'company' and 'query'
    #     chain.invoke({"company": "Neatlogs", "query": "How do variables work?"}, config={"callbacks": [handler]})
    # test_langchain_messages()
    
    # LangGraph
    neatlogs.init(
        api_key="kL-zN954K3s_lz4FMy9-p0QLqI5S4zFK", 
        base_url="http://localhost:3000",
        debug=True
    )
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
    # class GraphState(TypedDict):
    #     query: str
    #     analysis: str
    #     status: str

    # def analyst_node(state: GraphState):
    #     print(f"   [Node] Analyst working on: {state['query']}")
    #     # Using model instance inside node for realistic tracing
    #     response = model.invoke(state['query'])
    #     return {"analysis": response.content, "status": "analyzed"}

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
    # def test_langgraph_configurable():
    #     print("\n>>> Scenario 5: LangGraph Configurable Parameters")
    #     def config_node(state: GraphState, config: RunnableConfig):
    #         user_id = config.get("configurable", {}).get("user_id", "unknown")
    #         print(f"   [Node] Config node for user: {user_id}")
    #         model.invoke(f"Processing request for user {user_id}: {state['query']}")
    #         return {"status": f"processed_for_{user_id}"}

    #     workflow = StateGraph(GraphState)
    #     workflow.add_node("config_node", config_node)
    #     workflow.add_edge(START, "config_node")
    #     workflow.add_edge("config_node", END)
    #     app = workflow.compile()
        
    #     # Neatlogs should capture the configurable user_id
    #     app.invoke(
    #         {"query": "test", "status": "start"}, 
    #         config={"configurable": {"user_id": "user_123"}}
    #     )
    # test_langgraph_configurable()
    
    # CrewAI
    # We'll run one CrewAI test after verifying the others
    def test_crewai_kickoff():
        print("\n>>> Scenario 6: CrewAI Kickoff Inputs")
        researcher = Agent(
            role='Researcher',
            goal='Find info about {topic}',
            backstory='Expert researcher',
            llm=model
        )
        task = Task(
            description='Research the {topic} and provide 3 key points.',
            agent=researcher,
            expected_output='3 bullet points'
        )
        crew = Crew(agents=[researcher], tasks=[task])
        
        # Neatlogs should capture 'topic' from kickoff inputs
        crew.kickoff(inputs={'topic': 'Space Exploration'})
    test_crewai_kickoff()
    def test_crewai_context_flow():
        print("\n>>> Scenario 7: CrewAI Context/Output Flow")
        writer = Agent(role='Writer', goal='Write a post', backstory='Pro writer', llm=model)
        
        task1 = Task(description='Research {subject}', agent=writer, expected_output='Short summary')
        task2 = Task(description='Write a tweet based on the summary', agent=writer, expected_output='One tweet', context=[task1])
        
        crew = Crew(agents=[writer], tasks=[task1, task2])
        # Neatlogs should capture 'subject' and the flow of data from task1 to task2
        crew.kickoff(inputs={'subject': 'AI Observability'})
    test_crewai_context_flow()
    def test_crewai_fstring():
        print("\n>>> Scenario 10: CrewAI with Python F-Strings")
        target = "Mars Rover"
        # Manual rendering in Task description
        rendered_desc = f"Research the {target} mission."
        
        agent = Agent(role='Researcher', goal='Research', backstory='Researcher', llm=model)
        task = Task(description=rendered_desc, agent=agent, expected_output='Summary')
        crew = Crew(agents=[agent], tasks=[task])
        
        # Kickoff without inputs dict - variables are hardcoded strings now
        crew.kickoff()
    test_crewai_fstring()
    
    print("\nAll tests completed. Waiting for background threads to finish...")
    time.sleep(5)
    print("Done.")
