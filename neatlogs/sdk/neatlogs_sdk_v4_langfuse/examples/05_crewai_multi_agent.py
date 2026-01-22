"""
Example 5: CrewAI Multi-Agent System

CrewAI orchestrates multiple agents working together.
OpenLLMetry instruments CrewAI, showing agent collaboration.
"""

import os
import sys


os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from neatlogs.sdk.neatlogs_sdk_v4_langfuse import init, flush, shutdown, trace

# Enable span logging
os.environ['NEATLOGS_LOG_SPANS'] = 'true'

# Initialize
init(
    api_key=os.getenv("NEATLOGS_API_KEY", "test-key"),
    workflow_name="openai-direct",
    instrumentations=["openai"],
    debug=True,
)

# CrewAI imports
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI


def main():
    # Enable span logging
    os.environ['NEATLOGS_LOG_SPANS'] = 'true'
    
    # Initialize - CrewAI instrumented via OpenLLMetry
    init(
        api_key=os.getenv("NEATLOGS_API_KEY", "test-key"),
        workflow_name="crewai-research",
        instrumentations=["crewai", "openai", "langchain"],
        debug=True,
    )
    
    topic = "AI observability and tracing"
    
    try:
        with trace(
            "crewai_research_workflow",
            prompt_template="Research and write about {topic}",
            prompt_variables={"topic": topic}
        ):
            llm = ChatOpenAI(model="gpt-4o-mini")
            
            # Define agents
            researcher = Agent(
                role="Research Analyst",
                goal=f"Research {topic} trends",
                backstory="Expert at finding and analyzing information",
                llm=llm,
                verbose=True
            )
            
            writer = Agent(
                role="Content Writer",
                goal="Write engaging content about AI topics",
                backstory="Skilled writer who explains complex topics simply",
                llm=llm,
                verbose=True
            )
            
            # Define tasks
            research_task = Task(
                description=f"Research the latest trends in {topic}",
                agent=researcher,
                expected_output="A comprehensive research report"
            )
            
            write_task = Task(
                description=f"Write a blog post based on the research about {topic}",
                agent=writer,
                expected_output="An engaging blog post",
                context=[research_task]
            )
            
            # Create crew
            crew = Crew(
                agents=[researcher, writer],
                tasks=[research_task, write_task],
                process=Process.sequential,
                verbose=True,
                tracing=False
            )
            
            # Run crew - all agents traced!
            result = crew.kickoff()
            
            print(f"\n{'='*60}")
            print("Final Result:")
            print(result)
            print('='*60)
    except Exception as e:
        print(f"\nError during CrewAI execution: {e}")
    finally:
        print("\n💾 Flushing spans...")
        flush()
        print("🛑 Shutting down SDK...")
        shutdown()
        print("✅ Done!")


if __name__ == "__main__":
    main()
