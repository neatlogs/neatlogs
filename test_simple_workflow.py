import os
import sys

# Add parent directory to path to import local neatlogs
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

import neatlogs
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

# Initialize NeatLogs with local backend
neatlogs.init(
    api_key="EQZO8zBsJkDRvmIL06m-EQ7faMUHEIN5",
    tags=["v4", "langchain", "local-test"],
    instrumentations=["langchain"],
    workflow_name="test-langchain-1",
    debug=True
)

# Create a simple LangChain model
model = ChatOpenAI(
    api_key="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",
    model="gpt-4o-mini"
)

def test_simple_chain():
    print("\n>>> Test 1: Simple LangChain Chain")
    prompt = ChatPromptTemplate.from_template("Tell me a {length} joke about {topic}")
    chain = prompt | model | StrOutputParser()
    
    result = chain.invoke({"length": "short", "topic": "programming"})
    print(f"Result: {result}")

def test_message_template():
    print("\n>>> Test 2: Message Template with Variables")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant for {company}."),
        ("user", "Explain {concept} in simple terms.")
    ])
    chain = prompt | model | StrOutputParser()
    
    result = chain.invoke({
        "company": "NeatLogs",
        "concept": "OpenTelemetry tracing"
    })
    print(f"Result: {result}")

def test_multi_step():
    print("\n>>> Test 3: Multi-step Chain")
    prompt1 = ChatPromptTemplate.from_template("List 3 {item_type} in one line")
    prompt2 = ChatPromptTemplate.from_template("Pick the best item from this list and explain why: {items}")
    
    chain1 = prompt1 | model | StrOutputParser()
    chain2 = prompt2 | model | StrOutputParser()
    
    items = chain1.invoke({"item_type": "programming languages"})
    result = chain2.invoke({"items": items})
    print(f"Result: {result}")

if __name__ == "__main__":
    print("🚀 Starting LangChain + NeatLogs Local Test...")
    print(f"📍 Backend URL: http://localhost:3000")
    print(f"🔑 API Key: {os.getenv('NEATLOGS_API_KEY')[:20]}..." if os.getenv('NEATLOGS_API_KEY') else "❌ No API Key")
    
    # test_simple_chain()
    # test_message_template()
    test_multi_step()
    
    print("\n✅ All tests completed! Check your local NeatLogs UI at http://localhost:3000")
