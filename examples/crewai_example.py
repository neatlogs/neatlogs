from crewai import Agent, Task, Crew, LLM
import neatlogs
from dotenv import load_dotenv
import os

load_dotenv()

neatlogs.init(
    api_key=os.getenv("NEATLOGS_API_KEY"),
    tags=["v3", "crewai", "demo"],
    instrumentations=["crewai"],
)

llm = LLM(
    model=os.getenv("model"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    base_url=os.getenv("AZURE_OPENAI_ENDPOINT"),
)


# def test_crewai_kickoff():
#     print("\n>>> Scenario 6: CrewAI Kickoff Inputs")
#     researcher = Agent(
#         role='Researcher',
#         goal='Find info about {topic}',
#         backstory='Expert researcher',
#         llm=llm
#     )
#     task = Task(
#         description='Research the {topic} and provide 3 key points.',
#         agent=researcher,
#         expected_output='3 bullet points'
#     )
#     crew = Crew(agents=[researcher], tasks=[task])

#     # Neatlogs should capture 'topic' from kickoff inputs
#     crew.kickoff(inputs={'topic': 'Space Exploration'})


# test_crewai_kickoff()


# def test_crewai_context_flow():
#     print("\n>>> Scenario 7: CrewAI Context/Output Flow")
#     writer = Agent(role='Writer', goal='Write a post',
#                    backstory='Pro writer', llm=llm)

#     task1 = Task(description='Research {subject}',
#                  agent=writer, expected_output='Short summary')
#     task2 = Task(description='Write a tweet based on the summary',
#                  agent=writer, expected_output='One tweet', context=[task1])

#     crew = Crew(agents=[writer], tasks=[task1, task2])
#     # Neatlogs should capture 'subject' and the flow of data from task1 to task2
#     crew.kickoff(inputs={'subject': 'AI Observability'})


# test_crewai_context_flow()


def test_crewai_fstring():
    print("\n>>> Scenario 10: CrewAI with Python F-Strings")
    target = "Mars Rover"
    # Manual rendering in Task description
    rendered_desc = f"Research the {target} mission."

    agent = Agent(role='Researcher', goal='Research',
                  backstory='Researcher', llm=llm)
    task = Task(description=rendered_desc,
                agent=agent, expected_output='Summary')
    crew = Crew(agents=[agent], tasks=[task])

    # Kickoff without inputs dict - variables are hardcoded strings now
    crew.kickoff(inputs={'mission': target})


test_crewai_fstring()
