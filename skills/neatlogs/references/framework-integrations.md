# Framework Integrations — NeatLogs SDK v3 Reference

Framework-specific integration patterns for NeatLogs SDK v3. Covers auto-instrumentation setup, init ordering, install commands, and representative code examples for each supported LLM provider and agent framework.

---

## 1. Integration Approaches (Decision Tree)

Choose the approach that matches your application architecture:

### 1a. Pure Auto-Instrumentation

For applications that call LLM providers directly (OpenAI, Anthropic, Google GenAI, etc.). Just add the provider to `instrumentations=[]`. No decorators needed for LLM calls — every SDK call is traced automatically.

```python
neatlogs.init(instrumentations=["openai"])
```

### 1b. Auto-Instrumentation + `@span` Decorators

For custom multi-agent orchestration. Add providers to `instrumentations=[]` for LLM call tracing, then use `@span` decorators on your orchestration functions to capture the full call graph.

```python
neatlogs.init(instrumentations=["openai", "anthropic"])

@neatlogs.span(kind="WORKFLOW")
def pipeline(query: str) -> str:
    result_a = agent_a(query)       # @span(kind="AGENT")
    result_b = agent_b(result_a)    # @span(kind="AGENT")
    return result_b
```

### 1c. Auto-Instrumentation + `bind_templates`

For framework-managed LLM calls (e.g. CrewAI). Add `"crewai"` to instrumentations, use `bind_templates()` to attach prompt context to framework-owned LLMs, and `register_crewai_task()` for task-level prompt tracking.

```python
neatlogs.init(instrumentations=["openai", "crewai", "langchain"])
bound_llm = neatlogs.bind_templates(llm, system_tpl, domain="finance")
neatlogs.register_crewai_task(task, user_tpl, topic="AI chips")
```

---

## 2. OpenAI

- **Instrumentation key**: `instrumentations=["openai"]`
- **Covers**: The `openai` Python SDK — both `OpenAI()` and `AzureOpenAI()` (which is part of the same `openai` package)
- **Import order critical**: `neatlogs.init()` BEFORE `from openai import OpenAI`
- **Supports**: Sync, async, streaming
- **Install**: `pip install neatlogs[openai]==1.2.7`

```python
import neatlogs

neatlogs.init(
    api_key="...",
    workflow_name="my-app",
    instrumentations=["openai"],
)

from openai import OpenAI

client = OpenAI()

@neatlogs.span(kind="WORKFLOW")
def run(query: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": query}],
    )
    return response.choices[0].message.content

result = run("Explain quantum computing")
neatlogs.flush()
neatlogs.shutdown()
```

---

## 3. Azure AI Inference

- **Instrumentation key**: `instrumentations=["azure_ai_inference"]`
- **Covers**: Microsoft's **separate** `azure-ai-inference` SDK (`azure.ai.inference`), NOT `openai.AzureOpenAI`
- **If you use the `openai` package with Azure endpoints** (i.e. the `AzureOpenAI` class), use `instrumentations=["openai"]` instead
- **Install**: `pip install neatlogs[azure-ai-inference]==1.2.7`

```python
import neatlogs

neatlogs.init(
    api_key="...",
    workflow_name="azure-app",
    instrumentations=["azure_ai_inference"],
)

from azure.ai.inference import ChatCompletionsClient
# ... use the Azure AI Inference SDK
```

---

## 4. Anthropic

- **Instrumentation key**: `instrumentations=["anthropic"]`
- **Supports**: Extended thinking, streaming, tool use
- **Also works with `AnthropicBedrock`**: Still use `instrumentations=["anthropic"]`
- **Install**: `pip install neatlogs[anthropic]==1.2.7`

```python
import neatlogs

neatlogs.init(
    api_key="...",
    workflow_name="anthropic-app",
    instrumentations=["anthropic"],
)

from anthropic import Anthropic

client = Anthropic()

@neatlogs.span(kind="AGENT", name="analyst")
def analyst(query: str) -> str:
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{"role": "user", "content": query}],
    )
    return response.content[0].text

result = analyst("Analyze market trends")
neatlogs.flush()
neatlogs.shutdown()
```

---

## 5. Google GenAI (Gemini)

- **Instrumentation key**: `instrumentations=["google_genai"]`
- **Stricter init ordering**: `neatlogs.init()` must precede `google.genai.Client()` **instantiation**, not just import. The client object caches the transport at construction time.
- **Supports**: Sync, streaming, async streaming
- **Install**: `pip install neatlogs[google-genai]==1.2.7`

```python
import neatlogs

neatlogs.init(
    api_key="...",
    workflow_name="gemini-app",
    instrumentations=["google_genai"],
)

# Client MUST be created AFTER init()
from google import genai

client = genai.Client(api_key="...")

@neatlogs.span(kind="AGENT", name="researcher")
def researcher(topic: str) -> str:
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=topic,
    )
    return response.text

result = researcher("quantum computing advances")
neatlogs.flush()
neatlogs.shutdown()
```

### Wrong vs Right (Google GenAI)

See [`troubleshooting.md` §2](troubleshooting.md#2-google-genai-instantiation-ordering) for the full wrong/right code examples on Google GenAI client ordering.

---

## 6. LangChain

- **Instrumentation key**: `instrumentations=["langchain"]`
- **Auto-instruments**: LLM calls, chains, agents, tools, retrievers
- **Works with**: AgentExecutor, ReAct agents, LCEL chains
- **Install**: `pip install neatlogs[langchain]==1.2.7`

```python
import neatlogs
from neatlogs import PromptTemplate

neatlogs.init(
    api_key="...",
    workflow_name="langchain-app",
    instrumentations=["langchain"],
)

from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor

llm = ChatOpenAI(model="gpt-4o")

sys_tpl = PromptTemplate("You are a helpful research assistant for {{domain}}.")

@neatlogs.span(kind="WORKFLOW")
def run_agent(query: str) -> str:
    with neatlogs.trace("agent_llm", kind="LLM", prompt_template=sys_tpl):
        sys_tpl.compile(domain="science")
        result = agent_executor.invoke({"input": query})
    return result["output"]

result = run_agent("Explain black holes")
neatlogs.flush()
neatlogs.shutdown()
```

---

## 7. LangGraph

- **Instrumentation key**: `instrumentations=["langchain"]` (LangGraph uses LangChain instrumentation)
- **Tracks**: Graph execution, nodes, tool loops, fan-out/fan-in
- **Install**: `pip install neatlogs[langchain,langgraph]==1.2.7`

```python
import neatlogs
from neatlogs import PromptTemplate, UserPromptTemplate

neatlogs.init(
    api_key="...",
    workflow_name="langgraph-app",
    instrumentations=["langchain"],
)

from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o")

# Define templates per node
supervisor_tpl = PromptTemplate("You are a supervisor routing tasks to agents...")
researcher_tpl = PromptTemplate("You are a research agent for {{domain}}.")

def supervisor_node(state):
    with neatlogs.trace("supervisor_llm", kind="LLM", prompt_template=supervisor_tpl):
        msgs = supervisor_tpl.compile()
        response = llm.invoke(msgs + [{"role": "user", "content": state["query"]}])
    return {"next": response.content}

def researcher_node(state):
    with neatlogs.trace("researcher_llm", kind="LLM", prompt_template=researcher_tpl):
        msgs = researcher_tpl.compile(domain="technology")
        response = llm.invoke(msgs + [{"role": "user", "content": state["query"]}])
    return {"result": response.content}

# Build graph
graph = StateGraph(dict)
graph.add_node("supervisor", supervisor_node)
graph.add_node("researcher", researcher_node)
# ... add edges ...
app = graph.compile()

result = app.invoke({"query": "latest AI trends"})
neatlogs.flush()
neatlogs.shutdown()
```

---

## 8. CrewAI

- **Instrumentation key**: `instrumentations=["openai", "crewai", "langchain"]` (CrewAI uses LiteLLM internally; also include provider instrumentations)
- **Use `bind_templates()`** to attach prompt context to agent LLMs
- **Use `register_crewai_task(task, user_tpl, **vars)`** for task-level prompt tracking
- **Install**: `pip install neatlogs[crewai]==1.2.7` (pulls in `crewai >= 1.9.3` and `litellm`)
- **Note**: SDK pins `crewai >= 1.9.3`. CrewAI API has changed significantly between versions — ensure version compatibility.

```python
import neatlogs
from neatlogs import PromptTemplate, UserPromptTemplate

neatlogs.init(
    api_key="...",
    workflow_name="crewai-app",
    instrumentations=["openai", "crewai", "langchain"],
)

from crewai import Agent, Task, Crew, LLM

# System template for the agent's LLM — must NOT have required placeholders
# because bind_templates() calls system_tpl.compile() with no arguments.
# Pre-render the template string if you need dynamic values.
analyst_tpl = PromptTemplate("You are a senior market analyst.")

# User template for the task
task_tpl = UserPromptTemplate("Analyze {{topic}} trends for {{year}}.")

# Create and bind LLM
llm = LLM(model="gpt-4o")
bound_llm = neatlogs.bind_templates(llm, analyst_tpl)

# Create agent with bound LLM
analyst = Agent(
    role="Market Analyst",
    goal="Provide market analysis",
    backstory="Expert analyst with 10 years experience",
    llm=bound_llm,
)

# Create task and register template
task = Task(
    description="Analyze AI chip market trends",
    expected_output="Market analysis report",
    agent=analyst,
)
neatlogs.register_crewai_task(task, task_tpl, topic="AI chips", year="2025")

# Run crew
crew = Crew(agents=[analyst], tasks=[task])
result = crew.kickoff()

neatlogs.flush()
neatlogs.shutdown()
```

---

## 9. LiteLLM

- **Instrumentation key**: `instrumentations=["litellm"]`
- **Auto-instruments**: LiteLLM's unified API across all providers
- **Install**: `pip install neatlogs[litellm]==1.2.7`

```python
import neatlogs

neatlogs.init(
    api_key="...",
    workflow_name="litellm-app",
    instrumentations=["litellm"],
)

from litellm import completion

response = completion(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}],
)
```

---

## 10. Multi-Provider

When using multiple LLM providers, list all of them in `instrumentations`:

```python
import neatlogs

neatlogs.init(
    api_key="...",
    workflow_name="multi-provider-app",
    instrumentations=["openai", "anthropic", "google_genai", "langchain"],
)

from openai import OpenAI
from anthropic import Anthropic
from google import genai

openai_client = OpenAI()
anthropic_client = Anthropic()
gemini_client = genai.Client(api_key="...")

@neatlogs.span(kind="WORKFLOW")
def multi_model_pipeline(query: str) -> dict:
    # Each provider's LLM calls are auto-instrumented independently
    gpt_response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": query}],
    )
    claude_response = anthropic_client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{"role": "user", "content": query}],
    )
    gemini_response = gemini_client.models.generate_content(
        model="gemini-2.0-flash",
        contents=query,
    )
    return {
        "gpt": gpt_response.choices[0].message.content,
        "claude": claude_response.content[0].text,
        "gemini": gemini_response.text,
    }

result = multi_model_pipeline("Compare approaches to AGI")
neatlogs.flush()
neatlogs.shutdown()
```
