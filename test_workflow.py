import os
import sys

# Add parent directory to path to import local neatlogs
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

import neatlogs
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from dotenv import load_dotenv

load_dotenv()

# Initialize NeatLogs with local backend
neatlogs.init(
    api_key=os.getenv("NEATLOGS_API_KEY"),
    base_url="http://localhost:3000",  # Point to local NeatLogs backend
    tags=["v4", "langchain", "agents", "local-test"],
    instrumentations=["langchain"]
)

# Define tools
@tool
def calculate_sum(a: int, b: int) -> int:
    """Calculate the sum of two numbers."""
    return a + b

@tool
def calculate_product(a: int, b: int) -> int:
    """Calculate the product of two numbers."""
    return a * b

@tool
def get_weather(city: str) -> str:
    """Get the weather for a city (simulated)."""
    # Simulated weather data
    weather_data = {
        "New York": "Sunny, 72°F",
        "London": "Rainy, 15°C",
        "Tokyo": "Cloudy, 20°C",
        "Paris": "Partly Cloudy, 18°C",
    }
    return weather_data.get(city, f"Weather data not available for {city}")

# Create LangChain agent
model = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-4o-mini",
    temperature=0
)

tools = [calculate_sum, calculate_product, get_weather]

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant with access to tools. Use them when needed."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(model, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

def test_math_workflow():
    print("\n>>> Test 1: Math Calculation Workflow")
    result = agent_executor.invoke({
        "input": "What is the sum of 15 and 27, and then what is the product of that result and 3?"
    })
    print(f"Result: {result['output']}")

def test_weather_workflow():
    print("\n>>> Test 2: Weather Query Workflow")
    result = agent_executor.invoke({
        "input": "What's the weather like in Tokyo and Paris?"
    })
    print(f"Result: {result['output']}")

def test_mixed_workflow():
    print("\n>>> Test 3: Mixed Math and Weather Workflow")
    result = agent_executor.invoke({
        "input": "If it's sunny in New York and the temperature is above 70F, calculate 10 times 5. Otherwise, calculate 10 plus 5."
    })
    print(f"Result: {result['output']}")

if __name__ == "__main__":
    print("🚀 Starting LangChain Agent + NeatLogs Local Test...")
    print(f"📍 Backend URL: http://localhost:3000")
    print(f"🔑 API Key: {os.getenv('NEATLOGS_API_KEY')[:20]}..." if os.getenv('NEATLOGS_API_KEY') else "❌ No API Key")
    
    test_math_workflow()
    test_weather_workflow()
    test_mixed_workflow()
    
    print("\n✅ All agent tests completed! Check your local NeatLogs UI at http://localhost:3000")
