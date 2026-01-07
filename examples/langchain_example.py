import neatlogs
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv

load_dotenv()

model = AzureChatOpenAI(api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
                        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),)

neatlogs.init(api_key=os.getenv('NEATLOGS_API_KEY'), tags=[
    "v3", "langchain", "demo"], instrumentations=["langchain"])


# def test_langchain_model_template():
#     print("\n>>> Scenario 3.1: LangChain model.invoke with Template (Direct)")
#     prompt = ChatPromptTemplate.from_template("Explain {topic}")

#     # Format the template into a PromptValue first
#     prompt_value = prompt.invoke({"topic": "Dark Matter"})

#     # Now this will work
#     model.invoke(prompt_value)


# test_langchain_model_template()


# def test_langchain_model_prompt_value():
#     print("\n>>> Scenario 3.2: LangChain model.invoke with PromptValue")
#     prompt = ChatPromptTemplate.from_template("Explain {topic}")
#     prompt_value = prompt.invoke({"topic": "Dark Matter"})
#     model.invoke(prompt_value)


# test_langchain_model_prompt_value()


# def test_langchain_model_partial():
#     print("\n>>> Scenario 3.3: LangChain model.invoke with Partial Template")
#     prompt = ChatPromptTemplate.from_template(
#         "As a {role}, tell me about {topic}")
#     partial_prompt = prompt.partial(role="Scientist")
#     prompt_value = partial_prompt.invoke(
#         {"topic": "Evolution"})
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
#     prompt_value = prompt.invoke(
#         {"company": "Neatlogs", "topic": "V3 Architecture"})

#     # Neatlogs captures 'company' and 'topic'
#     model.invoke(prompt_value)


# test_langchain_model_messages_structured()


# def test_langchain_model_messages_fstring():
#     print("\n>>> Scenario 3.4b: Messages with F-Strings (Invisible to SDK)")
#     from langchain_core.messages import HumanMessage, SystemMessage
#     company = "Neatlogs"
#     topic = "V3 Architecture"

#     # Python renders the f-string into a static string immediately
#     messages = [
#         SystemMessage(
#             content=f"You are a helpful assistant for {company}"),
#         HumanMessage(content=f"Explain {topic}")
#     ]

#     # To the SDK, this looks like a list of hardcoded text with no variables
#     model.invoke(messages)


# test_langchain_model_messages_fstring()


# def test_langchain_model_invoke_string():
#     print("\n>>> Scenario 12: LangChain model.invoke with STRING (F-String variant)")
#     topic = "Quantum Gravity"
#     model.invoke(f"Tell me about {topic}")


# test_langchain_model_invoke_string()


# def test_langchain_fstring():
#     print("\n>>> Scenario 8: LangChain with Python F-Strings")
#     topic = "Superconductivity"
#     # Manual rendering - variable 'topic' is lost to framework
#     rendered_prompt = f"Explain {topic} in one sentence."
#     # LangChain just sees a static string
#     model.invoke(rendered_prompt)


# test_langchain_fstring()


# def test_langchain_standard():
#     print("\n>>> Scenario 1: LangChain Standard Invoke")
#     prompt = ChatPromptTemplate.from_template(
#         "Explain {topic} in one sentence.")
#     chain = prompt | model | StrOutputParser()
#     # Neatlogs should capture 'topic' variable
#     chain.invoke({"topic": "Quantum Computing"})


# test_langchain_standard()


# def test_langchain_partial():
#     print("\n>>> Scenario 2: LangChain Partialing")
#     prompt = ChatPromptTemplate.from_template(
#         "As a {role}, tell me a joke about {subject}.")
#     partial_prompt = prompt.partial(role="Robot")
#     chain = partial_prompt | model | StrOutputParser()
#     chain.invoke({"subject": "humans"})


# test_langchain_partial()


def test_langchain_messages():
    print("\n>>> Scenario 3: LangChain Message Placeholders")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant for {company}."),
        ("user", "{query}")
    ])
    chain = prompt | model | StrOutputParser()
    # Neatlogs should capture 'company' and 'query'
    chain.invoke({"company": "Neatlogs", "query": "How do variables work?"})


test_langchain_messages()

"""
Input variables not getting captured for:
fstring
stroutput

"""